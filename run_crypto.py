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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(CACHE_DIR, "history.csv")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

TW_TZ = ZoneInfo("Asia/Taipei")


# -----------------------------
# Settings / Discord
# -----------------------------
def _now_tw() -> datetime:
    return datetime.now(TW_TZ)


def _today_tw() -> str:
    return _now_tw().strftime("%Y-%m-%d")


def _load_settings() -> Dict:
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _post(content: str) -> None:
    settings = _load_settings()
    url = settings.get("DISCORD_WEBHOOK", "") or os.getenv("DISCORD_WEBHOOK", "")
    if not url:
        print("[discord] webhook not set, skip")
        print(content)
        return

    try:
        r = requests.post(url, json={"content": content}, timeout=20)
        if r.status_code >= 300:
            print(f"[discord] failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[discord] post exception: {e}")
        print(content)


# -----------------------------
# History I/O
# -----------------------------
def _read_history() -> pd.DataFrame:
    cols = [
        "date",
        "ticker",
        "pred",
        "price_at_run",
        "sup",
        "res",
        "settle_date",
        "settle_close",
        "realized_return",
        "hit",
        "status",
        "updated_at",
    ]
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(HISTORY_FILE)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def _write_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8")


def settle_date_plus_days(date_str: str, days: int) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return (d + timedelta(days=days)).strftime("%Y-%m-%d")


def append_today_predictions(hist: pd.DataFrame, today: str, new_rows: List[Dict]) -> pd.DataFrame:
    now_str = _now_tw().strftime("%Y-%m-%d %H:%M:%S")

    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        return hist

    df_new["date"] = today
    df_new["status"] = "pending"
    df_new["updated_at"] = now_str

    # 避免同日同 ticker 重複
    if not hist.empty:
        keep = ~(
            (hist["date"].astype(str) == today)
            & (hist["ticker"].astype(str).isin(df_new["ticker"].astype(str)))
        )
        hist = hist[keep].copy()

    return pd.concat([hist, df_new], ignore_index=True)


# -----------------------------
# Settle history
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
        # 防呆：price_at_run 可能因過去抓價失敗而為 0/NaN，避免除以 0
        try:
            price_at_run = float(row["price_at_run"])
        except Exception:
            price_at_run = float("nan")

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

    if not settled_lines:
        return hist, ""

    msg = "\n".join(settled_lines[:10])
    if len(settled_lines) > 10:
        msg += f"\n… 另外還有 {len(settled_lines) - 10} 筆已結算"
    return hist, msg


def last20_stats_line(hist: pd.DataFrame) -> str:
    if hist is None or hist.empty:
        return "最近 20 筆命中率：--% / 平均報酬：--%"

    done = hist[hist["status"].astype(str).isin(["settled"])]
    if done.empty:
        return "最近 20 筆命中率：--% / 平均報酬：--%"

    tail = done.tail(20).copy()
    try:
        hit_rate = float(tail["hit"].astype(float).mean())
        avg_rr = float(tail["realized_return"].astype(float).mean())
        return f"最近 20 筆命中率：{hit_rate:.0%} / 平均報酬：{avg_rr:+.2%}"
    except Exception:
        return "最近 20 筆命中率：--% / 平均報酬：--%"


# -----------------------------
# Universe
# -----------------------------
def get_universe(today: str) -> List[str]:
    # 你可以自行擴充/調整
    return [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "SOL-USD",
        "XRP-USD",
        "ADA-USD",
        "DOGE-USD",
        "AVAX-USD",
        "DOT-USD",
        "LINK-USD",
        "MATIC-USD",
        "LTC-USD",
        "BCH-USD",
        "ATOM-USD",
        "TRX-USD",
        "ETC-USD",
        "FIL-USD",
        "NEAR-USD",
        "APT-USD",
        "ARB-USD",
        "OP-USD",
        "SUI-USD",
        "INJ-USD",
        "AAVE-USD",
        "UNI-USD",
        "FTM-USD",
        "PEPE-USD",
        "SHIB-USD",
    ]


def calc_pivot(df: pd.DataFrame) -> Tuple[float, float]:
    d = df.copy().tail(20)
    lo = float(d["Low"].min()) if "Low" in d.columns else float(d["Close"].min())
    hi = float(d["High"].max()) if "High" in d.columns else float(d["Close"].max())
    return round(lo, 4), round(hi, 4)


# -----------------------------
# Main
# -----------------------------
def run() -> None:
    today = _today_tw()

    # 1) 先結算
    hist, settle_detail = settle_history(today)
    _write_history(hist)

    if settle_detail:
        _post("🧾 已結算歷史紀錄：\n" + settle_detail)

    # 2) 今日預測
    universe = get_universe(today)
    data = safe_yf_download(universe, period="2y", max_chunk=60)

    feats = ["mom20", "bias", "vol_ratio"]
    results: Dict[str, dict] = {}

    for s, df in data.items():
        if df is None or len(df) < 160:
            continue

        df = df.copy()
        if "Close" not in df.columns:
            continue

        # 建特徵
        df["mom20"] = df["Close"].pct_change(20)
        ma20 = df["Close"].rolling(20).mean()
        df["bias"] = (df["Close"] - ma20) / ma20
        if "Volume" in df.columns:
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        else:
            df["vol_ratio"] = pd.NA

        df["target"] = df["Close"].shift(-5) / df["Close"] - 1

        df = df.dropna()
        if len(df) < 120:
            continue

        train = df.tail(500).copy()

        model = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
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

        # 防呆：若當次抓到的現價為 0/NaN，避免寫入 history 造成之後結算除以 0
        try:
            price_at_run = float(r["price"])
        except Exception:
            price_at_run = float("nan")

        if (not math.isfinite(price_at_run)) or price_at_run <= 0:
            print(f"[history] skip write: {t} invalid price_at_run={r.get('price')}")
            continue

        new_rows.append(
            {
                "ticker": t,
                "pred": r["pred"],
                "price_at_run": price_at_run,
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

    # 4) Discord 報告
    msg = f"₿ 加密貨幣 AI 進階預測報告 ({today})\n"
    msg += "-" * 42 + "\n\n"
    msg += "🏆 AI 海選 Top 5 (潛力幣)\n"

    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    for i, (t, r) in enumerate(top):
        msg += f"{medals[i]} {t}: 預估 {r['pred']:+.2%}\n"
        msg += f" └ 現價: {r['price']} (支撐: {r['sup']} / 壓力: {r['res']})\n"

    msg += "\n" + stats_line
    _post(msg)


if __name__ == "__main__":
    run()
