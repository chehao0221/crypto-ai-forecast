from utils.safe_yfinance import safe_yf_download

import pandas as pd
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "crypto_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

MAIN_5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

def get_crypto_universe():
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = []

    for offset in [0, 100, 200]:
        try:
            url = f"https://finance.yahoo.com/crypto?count=100&offset={offset}"
            df = pd.read_html(requests.get(url, headers=headers, timeout=10).text)[0]
            tickers += df["Symbol"].dropna().tolist()
        except:
            pass

    exclude = ["USDT-USD", "USDC-USD", "DAI-USD"]
    tickers = [t for t in tickers if t.endswith("-USD") and t not in exclude]

    backup = [
        "ADA-USD","DOGE-USD","DOT-USD","LINK-USD","AVAX-USD",
        "BCH-USD","NEAR-USD","ARB-USD","RNDR-USD","APT-USD"
    ]

    return list(dict.fromkeys(MAIN_5 + tickers + backup))

def calc_pivot(df):
    r = df.iloc[-20:]
    h, l, c = r["High"].max(), r["Low"].min(), r["Close"].iloc[-1]
    p = (h + l + c) / 3
    prec = 4 if c < 10 else 2
    return round(2*p - h, prec), round(2*p - l, prec)

def run():
    universe = get_crypto_universe()
    data = safe_yf_download(universe, period="2y", max_chunk=100)

    feats = ["mom20", "bias", "vol_ratio"]
    results = {}

    for s, df in data.items():
        if len(df) < 160:
            continue

        df["mom20"] = df["Close"].pct_change(20)
        df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["target"] = df["Close"].shift(-5) / df["Close"] - 1

        train = df.iloc[:-5].dropna()
        if len(train) < 80:
            continue

        model = XGBRegressor(
            n_estimators=90,
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

    msg = f"₿ 加密貨幣 AI 進階預測報告 ({datetime.now():%Y-%m-%d})\n"
    msg += "------------------------------------------\n\n"

    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    horses = {k: v for k, v in results.items() if k not in MAIN_5 and v["pred"] > 0}
    top5 = sorted(horses, key=lambda x: horses[x]["pred"], reverse=True)[:5]

    msg += "🏆 AI 海選 Top 5 (潛力標的)\n"
    for i, s in enumerate(top5):
        r = results[s]
        msg += f"{medals[i]} {s}: 預估 `{r['pred']:+.2%}`\n"
        msg += f" └ 現價: {r['price']:.4f} (支撐: {r['sup']} / 壓力: {r['res']})\n"

    msg += "\n💎 主流幣監控 (固定顯示)\n"
    for s in MAIN_5:
        if s in results:
            r = results[s]
            msg += f"{s}: 預估 `{r['pred']:+.2%}`\n"
            msg += f" └ 現價: {r['price']:.4f} (支撐: {r['sup']} / 壓力: {r['res']})\n"

    msg += "\n🏁 加密貨幣 5 日回測結算報告\n\n💡 AI 為機率模型，僅供研究參考"

    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:1900]})
    else:
        print(msg)

if __name__ == "__main__":
    run()
