import yfinance as yf
import pandas as pd
import requests
import os
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "crypto_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# 固定主流幣（不動）
MAIN_5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

def get_top_volume_pool():
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = []

    try:
        for offset in [0, 100, 200]:
            url = f"https://finance.yahoo.com/crypto/?count=100&offset={offset}"
            r = requests.get(url, headers=headers, timeout=15)
            tables = pd.read_html(r.text, flavor="html5lib")
            if tables and "Symbol" in tables[0]:
                tickers.extend(tables[0]["Symbol"].dropna().tolist())
    except:
        pass

    backup = [
        "ADA-USD","DOGE-USD","DOT-USD","MATIC-USD","LINK-USD","AVAX-USD",
        "SHIB-USD","TRX-USD","LTC-USD","BCH-USD","UNI-USD","NEAR-USD",
        "FIL-USD","APT-USD","ARB-USD","OP-USD","STX-USD","RNDR-USD"
    ]

    exclude = ["USDT-USD","USDC-USD","DAI-USD","FDUSD-USD","PYUSD-USD"]
    clean = [t for t in tickers if isinstance(t, str) and t.endswith("-USD") and t not in exclude]

    return list(dict.fromkeys(clean + backup))

def calc_rsi(close, n=14):
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(n).mean()
    loss = -diff.clip(upper=0).rolling(n).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def calc_pivot(df):
    r = df.iloc[-20:]
    h, l, c = r["High"].max(), r["Low"].min(), r["Close"].iloc[-1]
    p = (h + l + c) / 3
    prec = 4 if c < 10 else 2
    return round(2*p - h, prec), round(2*p - l, prec)

def get_settle_report():
    if not os.path.exists(HISTORY_FILE):
        return "\n 加密貨幣 5 日回測結算報告\n"

    df = pd.read_csv(HISTORY_FILE)
    if "settled" not in df.columns:
        return "\n 加密貨幣 5 日回測結算報告\n"

    unsettled = df[df["settled"] == False]
    report = "\n 加密貨幣 5 日回測結算報告\n"

    for i, row in unsettled.iterrows():
        try:
            p = yf.download(row["symbol"], period="7d", auto_adjust=True, progress=False)
            if p.empty:
                continue
            exit_p = p["Close"].iloc[-1]
            ret = (exit_p - row["entry_price"]) / row["entry_price"]
            df.at[i, "settled"] = True
        except:
            continue

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= datetime.now() - timedelta(days=180)]
    df.to_csv(HISTORY_FILE, index=False)

    return report

def run():
    pool = list(set(MAIN_5 + get_top_volume_pool()))
    data = yf.download(pool, period="2y", auto_adjust=True, group_by="ticker", progress=False)

    results = {}
    feats = ["mom20","bias","vol_ratio","rsi","volatility"]

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

    msg = f"₿ 加密貨幣 AI 進階預測報告 ({datetime.now():%Y-%m-%d})\n"
    msg += "------------------------------------------\n\n"

    msg += " AI 海選 Top 5 (潛力標的)\n"
    top5 = sorted(
        [(k, v) for k, v in results.items() if k not in MAIN_5],
        key=lambda x: x[1]["pred"],
        reverse=True
    )[:5]

    for s, r in top5:
        msg += f" {s}: 預估 {r['pred']:+.2%}\n"
        msg += f"  └ 現價: {r['price']:.4f} (支撐: {r['sup']} / 壓力: {r['res']})\n"

    msg += "\n 主流幣監控 (固定顯示)\n"
    for s in MAIN_5:
        if s in results:
            r = results[s]
            msg += f"{s}: 預估 {r['pred']:+.2%}\n"
            msg += f" └ 現價: {r['price']:.4f} (支撐: {r['sup']} / 壓力: {r['res']})\n"

    msg += get_settle_report()
    msg += "\n AI 為機率模型，僅供研究參考。投資請謹慎。"

    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:2000]})
    else:
        print(msg)

if __name__ == "__main__":
    run()
