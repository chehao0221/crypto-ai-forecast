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

HISTORY_FILE = os.path.join(BASE_DIR, "crypto_history.csv")
UNIVERSE_CACHE_FILE = os.path.join(CACHE_DIR, "crypto_universe.json")

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

MAIN_5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

DEFAULT_UNIVERSE = [
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

# -----------------------------
# 小加固參數（向台股看齊，用途：參考）
# -----------------------------
MIN_PRED = 0.005   # 0.5%（5 日預測報酬門檻）
MAX_VOL20 = 0.07   # 7%（近 20 日日報酬波動上限）

TW_TZ = ZoneInfo("Asia/Taipei")


def _now_tw() -> datetime:
    return datetime.now(TW_TZ)


def _today_tw() -> str:
    return _now_tw().strftime("%Y-%m-%d")


def settle_date_plus_days(today: str, days: int = 5) -> str:
    dt = datetime.strptime(today, "%Y-%m-%d") + timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


def _load_universe_cache(today: str) -> List[str] | None:
    if not os.path.exists(UNIVERSE_CACHE_FILE):
        return None
    try:
        obj = json.loads(open(UNIVERSE_CACHE_FILE, "r", encoding="utf-8").read())
        if obj.get("date") == today and isinstance(obj.get("tickers"), list):
            return obj["tickers"]
    except Exception:
        return None
    return None


def _save_universe_cache(today: str, tickers: List[str]) -> None:
    try:
        with open(UNIVERSE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": today, "tickers": tickers}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_universe(today: str) -> List[str]:
    cached = _load_universe_cache(today)
    if cached:
        return cached
    _save_universe_cache(today, DEFAULT_UNIVERSE)
    return DEFAULT_UNIVERSE


def calc_pivot(df: pd.DataFrame) -> Tuple[float, float]:
    d = df.copy().tail(20)
    lo = float(d["Low"].min()) if "Low" in d.columns else float(d["Close"].min())
    hi = float(d["High"].max()) if "High" in d.columns else float(d["Close"].max())
    return round(lo, 4), round(hi, 4)


def _post(content: str) -> None:
    if WEBHOOK_URL:
        try:
            r = requests.post(WEBHOOK_URL, json={"content": content}, timeout=15)
            if r.status_code >= 300:
                print(f"⚠️ Discord 發送失敗: {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"⚠️ Discord 發送失敗: {e}")
            print(content)
    else:
        print(content)


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
    return df[cols]


def _write_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8")


def append_today_predictions(hist: pd.DataFrame, today: str, new_rows: List[Dict]) -> pd.DataFrame:
    now_str = _now_tw().strftime("%Y-%m-%d %H:%M:%S")

    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        return hist

    df_new["run_date"] = today
    df_new["status"] = "pending"
    df_new["updated_at"] = now_str

    if not hist.empty:
        keep = ~(
            (hist["run_date"].astype(str) == today)
            & (hist["ticker"].astype(str).isin(df_new["ticker"].astype(str)))
        )
        hist = hist[keep].copy()

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
        try:
            price_at_run = float(row["price_at_run"])
        except Exception:
            price_at_run = float("nan")

        # ✅ FIX: avoid ZeroDivisionError / invalid historical run price
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

    df = hist.copy()
    df = df[df["status"].astype(str) == "settled"]
    df = df[pd.to_numeric(df["realized_return"], errors="coerce").notna()]
    if df.empty:
        return "最近 20 筆命中率：--% / 平均報酬：--%"

    tail = df.tail(20).copy()
    hit = pd.to_numeric(tail["hit"], errors="coerce")
    rr = pd.to_numeric(tail["realized_return"], errors="coerce")

    if hit.notna().sum() == 0 or rr.notna().sum() == 0:
        return "最近 20 筆命中率：--% / 平均報酬：--%"

    hit_rate = float(hit.mean())
    avg_rr = float(rr.mean())
    return f"最近 20 筆命中率：{hit_rate:.0%} / 平均報酬：{avg_rr:+.2%}"


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

        # ✅ FIX: Volume 常缺值，避免 df.dropna() 把整個幣清空
        vr = df["Volume"] / df["Volume"].rolling(20).mean()
        df["vol_ratio"] = vr.replace([math.inf, -math.inf], pd.NA).ffill()

        df["target"] = df["Close"].shift(-5) / df["Close"] - 1

        # ✅ FIX: 只針對必要欄位 dropna
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

        # 小加固（向台股看齊）：近 20 日波動（用日報酬 std）
        vol20 = float(df["Close"].pct_change().rolling(20).std().iloc[-1])

        price = float(df["Close"].iloc[-1])
        price_disp = round(price, 4) if price < 10 else round(price, 2)

        results[s] = {
            "pred": pred,
            "price": price_disp,
            "sup": sup,
            "res": res,
            "vol20": vol20,
        }

    # 3) 寫回歷史
    if not results:
        _post("⚠️ 今日無可用結果（可能資料不足或抓取失敗）")
        return

    # -----------------------------
    # 海選 Top5（小加固版，向台股看齊）
    # 1) 先挑 pred 達門檻 + 波動不極端 的「主選」
    # 2) 不足 5 檔用「備取」依 pred 補滿
    # -----------------------------
    items = list(results.items())

    def _vol_ok(v: float) -> bool:
        # vol20 可能是 nan；nan 視為未知，不擋（交給 pred 去排序）
        try:
            if pd.isna(v):
                return True
            return float(v) <= MAX_VOL20
        except Exception:
            return True

    primary = [
        (t, r) for (t, r) in items
        if (float(r.get("pred", 0.0)) >= MIN_PRED) and _vol_ok(r.get("vol20", float("nan")))
    ]
    primary_set = set([t for (t, _) in primary])
    backup = [(t, r) for (t, r) in items if t not in primary_set]

    primary_sorted = sorted(primary, key=lambda kv: kv[1]["pred"], reverse=True)
    backup_sorted = sorted(backup, key=lambda kv: kv[1]["pred"], reverse=True)

    top = (primary_sorted + backup_sorted)[:5]

    # 3) 寫入 history（今日 Top5）
    new_rows = []
    for t, r in top:
        settle_date = settle_date_plus_days(today, 5)

        # ✅ FIX: 不把 0/NaN 寫進 history，避免未來結算除以 0
        try:
            _p = float(r["price"])
        except Exception:
            _p = float("nan")
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

    # 4) Discord 顯示（跟你原本的一樣，不動）
    msg = f"₿ 加密貨幣 AI 進階預測報告 ({today})\n"
    msg += "-" * 42 + "\n\n"
    msg += "🏆 AI 海選 Top 5 (潛力幣)\n"

    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    for i, (t, r) in enumerate(top):
        msg += f"{medals[i]} {t}: 預估 {r['pred']:+.2%}\n"
        msg += f" └ 現價: {r['price']} (支撐: {r['sup']} / 壓力: {r['res']})\n"

    # --- Fixed major coins (like TW fixed large-caps) ---
    msg += "\n💎 主流幣監控 (固定顯示)\n"
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
