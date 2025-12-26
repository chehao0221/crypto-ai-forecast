import yfinance as yf
import pandas as pd
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# =========================
# 基本設定
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "crypto_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# =========================
# 工具函數
# =========================
def calc_pivot(df):
    """計算支撐與壓力位"""
    r = df.iloc[-20:]
    h, l, c = r["High"].max(), r["Low"].min(), r["Close"].iloc[-1]
    p = (h + l + c) / 3
    return round(2*p - h, 2), round(2*p - l, 2)

def get_crypto_list():
    """僅追蹤最知名的 5 個加密貨幣"""
    return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

# =========================
# 5 日回測結算 (核心邏輯不變)
# =========================
def get_settle_report():
    if not os.path.exists(HISTORY_FILE):
        return "\n📊 **回測結算**：尚無可結算資料\n"

    df = pd.read_csv(HISTORY_FILE)
    if "settled" not in df.columns:
        return ""
        
    unsettled = df[df["settled"] == False]
    if unsettled.empty:
        return "\n📊 **回測結算**：目前暫無待結算項目\n"

    report = "\n🏁 **加密貨幣 5 日預測結算報告**\n"
    for idx, row in unsettled.iterrows():
        try:
            # 下載最新數據比對
            price_df = yf.download(row["symbol"], period="7d", auto_adjust=True, progress=False)
            exit_price = price_df["Close"].iloc[-1]
            ret = (exit_price - row["entry_price"]) / row["entry_price"]
            
            # 判斷方向是否正確 (同正或同負)
            win = (ret > 0 and row["pred_ret"] > 0) or (ret < 0 and row["pred_ret"] < 0)

            report += (
                f"• `{row['symbol']}` 預估 {row['pred_ret']:+.2%} | "
                f"實際 `{ret:+.2%}` {'✅' if win else '❌'}\n"
            )
            df.at[idx, "settled"] = True
        except:
            continue

    df.to_csv(HISTORY_FILE, index=False)
    return report

# =========================
# 主程式
# =========================
def run():
    watch = get_crypto_list()
    
    # 下載數據
    data = yf.download(watch, period="2y", auto_adjust=True, group_by="ticker", progress=False)

    feats = ["mom20", "bias", "vol_ratio"]
    results = {}

    for s in watch:
        try:
            df = data[s].dropna()
            if len(df) < 150:
                continue

            # 計算特徵
            df["mom20"] = df["Close"].pct_change(20)
            df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1 # 預測目標：5日後報酬

            # 訓練模型
            train = df.iloc[:-5].dropna()
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=42
            )
            model.fit(train[feats], train["target"])

            # 進行當前預測
            pred = float(model.predict(df[feats].iloc[-1:])[0])
            sup, res = calc_pivot(df)

            results[s] = {
                "pred": pred,
                "price": round(df["Close"].iloc[-1], 2),
                "sup": sup,
                "res": res
            }
        except Exception:
            continue

    # 組合訊息
    msg = f"₿ **加密貨幣 AI 精選報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"

    # 按預測報酬排序
    sorted_coins = sorted(results.items(), key=lambda x: x[1]["pred"], reverse=True)

    for s, r in sorted_coins:
        icon = "📈" if r["pred"] > 0 else "📉"
        msg += f"{icon} **{s}**: 預估 5 日 `{r['pred']:+.2%}`\n"
        msg += f" └ 現價: `{r['price']:.2f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += get_settle_report()
    msg += "\n💡 AI 模型基於歷史數據，加密市場波動大，請務必做好風控。"

    # 輸出
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:1900]}, timeout=15)
    else:
        print(msg)

    # 儲存歷史紀錄
    hist = [{
        "date": datetime.now().date(),
        "symbol": s,
        "entry_price": r["price"],
        "pred_ret": r["pred"],
        "settled": False
    } for s, r in results.items()]

    pd.DataFrame(hist).to_csv(
        HISTORY_FILE,
        mode="a",
        header=not os.path.exists(HISTORY_FILE),
        index=False
    )

if __name__ == "__main__":
    run()
