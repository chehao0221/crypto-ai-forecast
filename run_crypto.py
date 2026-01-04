from __future__ import annotations

import os
import json
import warnings
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

HISTORY_FILE = os.path.join(BASE_DIR, "crypto_history.csv")
UNIVERSE_CACHE_FILE = os.path.join(CACHE_DIR, "crypto_universe.json")

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

MAIN_5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]


# -----------------------------
# Time helpers
# -----------------------------
def _now_tw() -> datetime:
    return datetime.now(ZoneInfo("Asia/Taipei"))


def _today_tw() -> str:
    return _now_tw().strftime("%Y-%m-%d")


def settle_date_plus_days(today: str, days: int = 5) -> str:
    # Crypto 24/7：維持簡單，直接 +5 天
    dt = datetime.strptime(today, "%Y-%m-%d") + timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


# -----------------------------
# Helpers
# -----------------------------
def calc_pivot(df: pd.DataFrame) -> Tuple[float, float]:
    r = df.iloc[-20:]
    h, l, c = float(r["High"].max()), float(r["Low"].min()), float(r["Close"].iloc[-1])
    p = (h + l + c) / 3
    prec = 4 if c < 10 else 2
    return round(2 * p - h, prec), round(2 * p - l, prec)


def _read_history() -> pd.DataFrame:
    cols = [
        "run_date",
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
    df["status"] = df["status"].fillna("pending")
    df["run_date"] = df["run_date"].astype(str)
    df["ticker"] = df["ticker"].astype(str)
    df["settle_date"] = df["settle_date"].fillna("").astype(str)
    return df


def _write_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def append_today_predictions(hist: pd.DataFrame, today: str, rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return hist

    now_str = _now_tw().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame(rows)
    df_new["run_date"] = today
    df_new["status"] = "pending"
    df_new["updated_at"] = now_str

    if not hist.empty:
        existing = set(zip(hist["run_date"].astype(str), hist["ticker"].astype(str)))
        df_new = df_new[~df_new.apply(lambda r: (today, str(r["ticker"])) in existing, axis=1)]

    if df_new.empty:
        return hist
    return pd.concat([hist, df_new], ignore_index=True)


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
        price_at_run = float(row["price_at_run"])
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

    df = hist.copy()
    df = df[df["status"].astype(str) == "settled"]
    df = df[pd.to_numeric(df["realized_return"], errors="coerce").notna()]
    if df.empty:
        return "最近 20 筆命中率：--% / 平均報酬：--%"

    df["settle_date_sort"] = pd.to_datetime(df["settle_date"], errors="coerce")
    df["updated_at_sort"] = pd.to_datetime(df["updated_at"], errors="coerce")
    df = df.sort_values(by=["settle_date_sort", "updated_at_sort"], ascending=True).tail(20)

    hit = pd.to_numeric(df["hit"], errors="coerce")
    rr = pd.to_numeric(df["realized_return"], errors="coerce")

    hit_rate = float(hit.mean()) if hit.notna().any() else float("nan")
    avg_rr = float(rr.mean()) if rr.notna().any() else float("nan")

    if not pd.notna(hit_rate) or not pd.notna(avg_rr):
        return "最近 20 筆命中率：--% / 平均報酬：--%"

    return f"最近 20 筆命中率：{hit_rate:.0%} / 平均報酬：{avg_rr:+.2%}"


# -----------------------------
# Universe (simple + cached)
# -----------------------------
def _load_universe_cache(today: str) -> List[str] | None:
    try:
        with open(UNIVERSE_CACHE_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("date") == today and isinstance(obj.get("tickers"), list):
            return obj["tickers"]
    except Exception:
        pass
    return None


def _save_universe_cache(today: str, tickers: List[str]) -> None:
    try:
        with open(UNIVERSE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": today, "tickers": tickers}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_universe(today: str) -> List[str]:
    """
    維持簡單：用一份常見幣清單 + MAIN_5 保底
    同日快取避免重跑
    """
    cached = _load_universe_cache(today)
    if cached:
        return cached

    # 你原本也有一段 backup，我沿用概念並擴成穩定的常見清單
    common = [
        "ADA-USD","DOGE-USD","DOT-USD","LINK-USD","AVAX-USD",
        "BCH-USD","NEAR-USD","ARB-USD","RNDR-USD","APT-USD",
        "MATIC-USD","LTC-USD","TRX-USD","ATOM-USD","ETC-USD",
        "FIL-USD","ICP-USD","OP-USD","INJ-USD","TIA-USD",
        "XLM-USD","UNI-USD","AAVE-USD","SUI-USD","FTM-USD",
        "IMX-USD","GRT-USD","PEPE-USD","SHIB-USD","HBAR-USD",
    ]

    out = list(dict.fromkeys(MAIN_5 + common))
    _save_universe_cache(today, out)
    return out


# -----------------------------
# Discord post
# -----------------------------
def _post(content: str) -> None:
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": content}, timeout=15)
        except Exception as e:
            print(f"⚠️ Discord 發送失敗: {e}")
            print(content)
    else:
        print(content)


# -----------------------------
# Main
# -----------------------------
def run() -> None:
    today = _today_tw()

    # 1) 先結算
    hist, settle_detail = settle_history(today)

    # 2) 今日預測
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
        df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["target"] = df["Close"].shift(-5) / df["Close"] - 1

        df = df.dropna()
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
        new_rows.append(
            {
                "ticker": t,
                "pred": r["pred"],
                "price_at_run": r["price"],
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
        msg += f"{t}: 預估 {r['pred']:+.2%}\n"
        msg += f" └ 現價: {r['price']} (支撐: {r['sup']} / 壓力: {r['res']})\n"

    msg += "\n🏁 加密貨幣 5 日回測結算報告\n"
    if settle_detail.strip():
        msg += settle_detail + "\n"

    msg += f"\n{stats_line}\n"
    msg += "\n💡 AI 為機率模型，僅供研究參考"

    _post(msg[:1900])


if __name__ == "__main__":
    run()
