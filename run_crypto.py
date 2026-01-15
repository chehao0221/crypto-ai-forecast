from __future__ import annotations

import os
import json
import warnings
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple

import pandas as pd
import requests
from xgboost import XGBRegressor

from utils.safe_yfinance import safe_yf_download

warnings.filterwarnings("ignore")
...


def _today_tw() -> str:
    return _now_tw().strftime("%Y-%m-%d")


def settle_date_plus_days(today: str, days: int = 5) -> str:
    # Crypto 24/7：維持簡單，直接 +5 天
    dt = datetime.strptime(today, "%Y-%m-%d") + timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


# -----------------------------
# Helpers
#
# (中間原本內容保持不動)
# -----------------------------


def settle_history(today: str) -> Tuple[pd.DataFrame, str]:
    hist = _read_history()
    if hist.empty:
        return hist, ""

    if hist["settle_date"].astype(str).str.len().eq(0).all():
        return hist, ""

    pending = hist[
        (hist["status"].astype(str) == "pending")
        & (hist["settle_date"].astype(str) <= today)
        & (hist["settle_date"].astype(str).str.len() > 0)
    ]
    if pending.empty:
        return hist, ""

    tickers = sorted(pending["ticker"].astype(str).unique().tolist())
    data = safe_yf_download(tickers, period="6mo", max_chunk=60)

    settled_lines: List[str] = []
    now_str = _now_tw().strftime("%Y-%m-%d %H:%M:%S")

    for idx, row in pending.iterrows():
        t = str(row["ticker"])
        settle_date = str(row["settle_date"])

        d = data.get(t)
        if d is None or d.empty:
            continue

        d2 = d.copy()
        d2.index = pd.to_datetime(d2.index).strftime("%Y-%m-%d")
        if settle_date not in d2.index:
            continue

        settle_close = float(d2.loc[settle_date, "Close"])
        try:
            price_at_run = float(row["price_at_run"])
        except Exception:
            price_at_run = float("nan")

        # FIX: avoid ZeroDivisionError / invalid historical values
        if (not math.isfinite(price_at_run)) or price_at_run <= 0:
            hist.at[idx, "status"] = "invalid"
            hist.at[idx, "updated_at"] = now_str
            continue
        if (not math.isfinite(settle_close)) or settle_close <= 0:
            hist.at[idx, "status"] = "invalid"
            hist.at[idx, "updated_at"] = now_str
            continue

        rr = (settle_close / price_at_run) - 1.0

        try:
            pred_f = float(row.get("pred", pd.NA))
        except Exception:
            pred_f = None

        hit = int(rr > 0)
        mark = "✅" if hit == 1 else "❌"

        hist.at[idx, "settle_close"] = round(settle_close, 6)
        hist.at[idx, "realized_return"] = rr
        hist.at[idx, "hit"] = hit
        hist.at[idx, "status"] = "settled"
        hist.at[idx, "updated_at"] = now_str

        if pred_f is None:
            settled_lines.append(f"• {t}: 實際 {rr:+.2%} {mark}")
        else:
            settled_lines.append(f"• {t}: 預估 {pred_f:+.2%} | 實際 {rr:+.2%} {mark}")

    msg = "\n".join(settled_lines[:10])
    if len(settled_lines) > 10:
        msg += f"\n… 另外還有 {len(settled_lines) - 10} 筆已結算"
    return hist, msg


def run() -> None:
    today = _today_tw()

    # 0) 先結算
    hist, settle_detail = settle_history(today)
    _write_history(hist)
    if settle_detail:
        _post("🧾 已結算歷史紀錄：\n" + settle_detail)

    # 1) 下載資料
    universe = get_universe(today)
    data = safe_yf_download(universe, period="2y", max_chunk=60)

    feats = ["mom20", "bias", "vol_ratio"]
    results: Dict[str, dict] = {}

    for s, df in data.items():
        if df is None or len(df) < 160:
            continue

        df = df.copy()
        df["mom20"] = df["Close"].pct_change(20)
        ma20 = df["Close"].rolling(20).mean()
        df["bias"] = (df["Close"] - ma20) / ma20

        # FIX: Volume 在 crypto 資料常有缺值，避免 dropna() 把整個幣清空
        vr = df["Volume"] / df["Volume"].rolling(20).mean()
        df["vol_ratio"] = vr.replace([math.inf, -math.inf], pd.NA).ffill()

        df["target"] = df["Close"].shift(-5) / df["Close"] - 1

        # 只針對模型需要的欄位做 dropna，避免 results 永遠空
        df = df.dropna(subset=["mom20", "bias", "vol_ratio", "target"])
        if len(df) < 120:
            continue

        train = df.iloc[:-1]

        model = XGBRegressor(
            n_estimators=90,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(train[feats], train["target"])

        pred = float(model.predict(df[feats].iloc[-1:])[0])
        sup, res = calc_pivot(df)

        # 價格顯示：小於 10 顯示 4 位，否則 2 位
        price = float(df["Close"].iloc[-1])
        price_disp = round(price, 4) if price < 10 else round(price, 2)

        results[s] = {
            "pred": pred,
            "price": price_disp,
            "sup": sup,
            "res": res,
        }

    if not results:
        _post("⚠️ 今日無可用結果（可能資料不足或抓取失敗）")
        return

    top = sorted(results.items(), key=lambda kv: kv[1]["pred"], reverse=True)[:5]

    # 3) 寫入 history（今日 Top5）
    new_rows = []
    for t, r in top:
        settle_date = settle_date_plus_days(today, 5)

        try:
            _p = float(r["price"])
        except Exception:
            _p = float("nan")
        # FIX: 不把 0/NaN 寫進 history，避免之後結算除以 0
        if (not math.isfinite(_p)) or _p <= 0:
            continue

        new_rows.append(
            {
                "ticker": t,
                "pred": r["pred"],
                "price_at_run": _p,
                "sup": r["sup"],
                "res": r["res"],
                "settle_date": settle_date,
                "settle_close": pd.NA,
                "realized_return": pd.NA,
                "hit": pd.NA,
            }
        )

    hist = append_today_predictions(hist, today, new_rows)
    _write_history(hist)

    stats_line = last20_stats_line(hist)

    # 4) Discord 顯示（跟台股一致）
    # ✅ 這一段（含下面所有 msg 組字）我完全沒有改，保持你原始顯示方式
    msg = f"₿ 加密貨幣 AI 進階預測報告 ({today})\n"
    msg += "-" * 42 + "\n\n"

    msg += "🏆 AI 海選 Top 5 (潛力幣)\n"
    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    for i, (t, r) in enumerate(top):
        msg += f"{medals[i]} {t}: 預估 {r['pred']:+.2%}\n"
        msg += f" └ 現價: {r['price']} (支撐: {r['sup']} / 壓力: {r['res']})\n"

    msg += "\n💎 指定主流幣監控 (固定顯示)\n"
    for t in MAIN_5:
        if t not in results:
            continue
        r = results[t]
        msg += f"• {t}: 預估 {r['pred']:+.2%} | 現價 {r['price']}\n"

    msg += "\n" + stats_line
    _post(msg)


if __name__ == "__main__":
    run()
