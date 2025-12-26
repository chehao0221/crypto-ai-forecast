import yfinance as yf
import pandas as pd
import requests
import os
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# =========================
# 基本設定（不動）
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "crypto_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# 固定監控（不動）
MAIN_5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

# =========================
# 海選池抓取（原邏輯保留）
# =========================
def get_top_volume_pool():
    headers = {'User-agent': 'Mozilla/5.0'}
    tickers = []

    try:
        for offset in [0, 100, 200]:
            url = f"https://finance.yahoo.com/crypto/?count=100&offset={offset}"
            resp = requests.get(url, headers=headers, timeout=15)
            tables = pd.read_html(resp.text, flavor='html5lib')
            if tables and 'Symbol' in tables[0]:
                tickers.extend(tables[0]['Symbol'].dropna().astype(str).tolist())
    except:
        pass

    backup_list = [
        "ADA-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "AVAX-USD",
        "SHIB-USD", "TRX-USD", "LTC-USD", "BCH-USD", "UNI-USD", "NEAR-USD",
        "FIL-USD", "APT-USD", "ARB-USD", "OP-USD", "STX-USD", "RNDR-USD"
    ]

    exclude = ["USDT-USD", "USDC-USD", "DAI-USD", "FDUSD-USD", "PYUSD-USD"]
    clean = [t for t in tickers if t.endswith("-USD") and t not in exclude]

    return list(dict.fromkeys(clean + backup_list))

# =========================
# 技術指標
# =========================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_pivot(df):
    r = df.iloc[-20:]
    h, l, c = r["High"].max(), r["Low"].min(), r["Close"].iloc[-1]
    p = (h + l + c) / 3
    prec = 4 if c < 10 else 2
    return round(2 * p - h, prec), round(2 * p - l, prec)

# =========================
# 回測結算（顯示不動）
# =========================
def get_settle_report():
    if not os.path.exists(HISTORY_FILE):
        return ""

    df = pd.read_csv(HISTORY_FILE)
    if "settled" not in df.columns:
        return ""

    unsettled = df[df["settled"] == False]
    if unsettled.empty:
        return "\n📊 **加密貨幣 5 日回測結算報告**：暫無待結算項目\n"

    report = "\n🏁 **加密貨幣 5 日回測結算報告**\n"
    for idx, row in unsettled.iterrows():
        try:
            p = yf.download(row["symbol"], period="7d", auto_adjust=True, progress=False)
            if p.empty:
                continue
            exit_p = p["Close"].iloc[-1]
            ret = (exit_p - row["entry_price"]) / row["entry_price"]
            win = (ret > 0 and row["pred_ret"] > 0) or (ret < 0 and row["pred_ret"] < 0)
            report += f"• `{row['symbol']}` 預估 {row['pred_ret']:+.2%} | 實際 `{ret:+.2%}` {'✅' if win else '❌'}\n"
            df.at[idx, "settled"] = True
        except:
            continue

    # 只保留 180 天
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= datetime.now() - timedelta(days=180)]
    df.to_csv(HISTORY_FILE, index=False)

    return report

# =========================
# 主程式
# =========================
def run():
    pool = list(set(MAIN_5 + get_top_volume_pool()))
    data = yf.download(pool, period="2y", auto_adjust=True, group_by="ticker", progress=False)

    results = {}
    feats = ["mom20", "bias", "vol_ratio", "rsi", "volatility"]

    for s in pool:
        try:
            df = data[s].dropna()
            if len(df) < 120:
                continue

            df["mom20"] = df["Close"].pct_change(20)
            df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df["rsi"] = calc_rsi(df["Close"])
            df["volatility"] = df["Close"].pct_change().rolling(20).std()
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1

            train = df.iloc[:-5].dropna()
            if len(train) < 60:
                continue

            model = XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(train[feats], train["target"])
            pred = float(model.predict(df[feats].iloc[-1:])[0])

            sup, res = calc_pivot(df)
            results[s] = {
                "pred": pred,
                "price": df["Close"].iloc[-1],
                "sup": sup,
                "res": res
            }
        except:
            continue

    msg = f"₿ **加密貨幣 AI 進階預測報**
