from __future__ import annotations

import os
import json
import math
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

HISTORY_FILE = os.path.join(CACHE_DIR, "history.csv")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

TW_TZ = ZoneInfo("Asia/Taipei")


# -----------------------------
# Discord / Settings
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
    """
    Only posts what your original logic decides to post.
    FIX: support DISCORD_WEBHOOK_URL (your repo secret name) + DISCORD_WEBHOOK (old name).
    """
    settings = _load_settings()
    url = (
        settings.get("DISCORD_WEBHOOK", "")
        or settings.get("DISCORD_WEBHOOK_URL", "")
        or os.getenv("DISCORD_WEBHOOK")
        or os.getenv("DISCORD_WEBHOOK_URL")
    )

    if not url:
        # keep behavior minimal: do not add extra messages, just print for logs
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
# History
# -----------------------------
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

    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def _write_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8")


def _settle_date_plus_days(date_str: str, days: int) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return (d + timedelta(days=days)).strftime("%Y-%m-%d")


def _append_today_predictions(hist: pd.DataFrame, today: str, new_rows: List[Dict]) -> pd.DataFrame:
    now_str = _now_tw().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        return hist

    df_new["run_date"] = today
    df_new["status"] = "pending"
    df_new["updated_at"] = now_str

    # de-dup same (run_date, ticker)
    if not hist.empty:
        keep = ~(
            (hist["run_date"].astype(str) == today)
            & (hist["ticker"].astype(str).isin(df_new["ticker"].astype(str)))
        )
        hist = hist[keep].copy()

    return pd.concat([hist, df_new], ignore_index=True)


def settle_history(today: str) -> Tuple[pd.DataFrame, str]:
    """
    FIX: avoid ZeroDivisionError if historical price_at_run is 0/NaN.
    """
    hist = _read_history()
    if hist.empty:
        return hist, ""

    pending = hist[
        (hist["status"].astype(str) == "pending")
        & (hist["settle_date"].astype(str).str.len() > 0)
        & (hist["settle_date"].astype(str) <= today)
    ]
    if pending.empty:
        return hist, ""

    tickers = sorted(pending["ticker"].astype(str).unique().tolist())
    data = safe_yf_download(tickers, period="6mo", max_chunk=60)

    now_str = _now_tw().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []

    for idx, row in pending.iterrows():
        t = str(row["ticker"])
        settle_date = str(row["settle_date"])

        df = data.get(t)
        if df is None or df.empty:
            continue

        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).strftime("%Y-%m-%d")
        if settle_date not in df2.index:
            continue

        try:
            settle_close = float(df2.loc[settle_date, "Close"])
        except Exception:
            settle_close = float("nan")

        try:
            price_at_run = float(row["price_at_run"])
        except Exception:
            price_at_run = float("nan")

        # --- FIX (core): guard invalid values ---
        if (not math.isfinite(price_at_run)) or price_at_run <= 0:
            hist.at[idx, "status"] = "invalid"
            hist.at[idx, "updated_at"] = now_str
            continue

        if (not math.isfinite(settle_close)) or settle_close <= 0:
            hist.at[idx, "status"] = "invalid"
            hist.at[idx, "updated_at"] = now_str
            continue

        rr = (settle_close / price_at_run) - 1.0
        hit = int(rr > 0)

        hist.at[idx, "settle_close"] = round(settle_close, 6)
        hist.at[idx, "realized_return"] = rr
        hist.at[idx, "hit"] = hit
        hist.at[idx, "status"] = "settled"
        hist.at[idx, "updated_at"] = now_str

        # keep settle summary minimal (doesn't change your main report formatting)
        mark = "✅" if hit == 1 else "❌"
        lines.append(f"• {t}: 實際 {rr:+.2%} {mark}")

    msg = "\n".join(lines) if lines else ""
    return hist, msg


def _last20_stats_line(hist: pd.DataFrame) -> str:
    done = hist[hist["status"].astype(str) == "settled"].copy()
    if done.empty:
        return "最近 20 筆命中率：--% / 平均報酬：--%"

    tail = done.tail(20).copy()
    hit = pd.to_numeric(tail["hit"], errors="coerce")
    rr = pd.to_numeric(tail["realized_return"], errors="coerce")

    if not hit.notna().any() or not rr.notna().any():
        return "最近 20 筆命中率：--% / 平均報酬：--%"

    hit_rate = float(hit.mean())
    avg_rr = float(rr.mean())
    return f"最近 20 筆命中率：{hit_rate:.0%} / 平均報酬：{avg_rr:+.2%}"


# -----------------------------
# Universe (keep your original list style)
# -----------------------------
def get_universe(_: str) -> List[str]:
    # 你原本的清單風格：保留 UNI / FTM（抓不到也不會害整體掛）
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
    ]


def calc_pivot(df: pd.DataFrame) -> Tuple[float, float]:
    d = df.copy().tail(20)
    lo = float(d["Low"].min()) if "Low" in d.columns else float(d["Close"].min())
    hi = float(d["High"].max()) if "High" in d.columns else float(d["Close"].max())
    return round(lo, 4), round(hi, 4)


def run() -> None:
    today = _today_tw()

    # 1) settle history (no extra message unless you already had one; keep minimal)
    hist, settle_detail = settle_history(today)
    _write_history(hist)
    # NOTE: do NOT add new heartbeat/status. Only post settle info if there is any.
    if settle_detail:
        _post("🧾 已結算：\n" + settle_detail)

    # 2) predict today
    universe = get_universe(today)
    data = safe_yf_download(universe, period="2y", max_chunk=60)

    feats = ["mom20", "bias", "vol_ratio"]
    results: Dict[str, Dict] = {}

    for s, df in data.items():
        if df is None or len(df) < 160:
            continue
        if "Close" not in df.columns:
            continue

        df = df.copy()

        # --- features: keep your original idea (mom20 / bias / vol_ratio) ---
        df["mom20"] = df["Close"].pct_change(20)

        ma20 = df["Close"].rolling(20).mean()
        df["bias"] = (df["Close"] - ma20) / ma20

        # FIX: avoid wiping everything because Volume has NaN (common in crypto)
        if "Volume" in df.columns:
            vol_ma = df["Volume"].rolling(20).mean()
            df["vol_ratio"] = (df["Volume"] / vol_ma).replace([math.inf, -math.inf], pd.NA)
        else:
            df["vol_ratio"] = 1.0

        # 5-day forward return target
        df["target"] = df["Close"].shift(-5) / df["Close"] - 1

        # FIX: only dropna on required columns (prevents results becoming empty)
        df = df.dropna(subset=["mom20", "bias", "vol_ratio", "target"])
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
        price = float(df["Close"].iloc[-1])
        price_disp = round(price, 4) if price < 10 else round(price, 2)

        results[s] = {"pred": pred, "price": price_disp, "sup": sup, "res": res}

    # keep behavior minimal: if no results, do not add extra discord messages
    if not results:
        print("[run] no results today (data/feature filtering).")
        return

    top = sorted(results.items(), key=lambda kv: kv[1]["pred"], reverse=True)[:5]

    # 3) write history (prevent writing invalid run prices)
    new_rows: List[Dict] = []
    for t, r in top:
        settle_date = _settle_date_plus_days(today, 5)

        try:
            price_at_run = float(r.get("price"))
        except Exception:
            price_at_run = float("nan")

        # FIX: don't write invalid price into history (prevents future settle crash)
        if (not math.isfinite(price_at_run)) or price_at_run <= 0:
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

    hist = _append_today_predictions(hist, today, new_rows)
    _write_history(hist)

    stats_line = _last20_stats_line(hist)

    # 4) report (simple; no extra added lines beyond a normal report)
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
