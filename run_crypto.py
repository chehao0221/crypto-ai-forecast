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

# 固定監控的標的
MAIN_5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

# =========================
# 工具函數：動態抓取交易量前 300
# =========================
def get_top_volume_tickers(limit=300):
    """從 Yahoo Finance 抓取交易量排名領先的幣種"""
    try:
        # Yahoo Finance 的 Crypto 篩選頁面，每頁 100 筆，我們抓前 3 頁
        tickers = []
        for offset in [0, 100, 200]:
            url = f"https://finance.yahoo.com/crypto/?count=100&offset={offset}"
            # 使用 pandas 讀取網頁表格
            tables = pd.read_html(requests.get(url, headers={'User-agent': 'Mozilla/5.0'}).text)
            df = tables[0]
            tickers.extend(df['Symbol'].tolist())
        
        # 過濾掉穩定幣 (USDT, USDC 等) 以確保海選到的是波動標的
        exclude = ["USDT-USD", "USDC-USD", "DAI-USD", "FDUSD-USD"]
        final_list = [t for t in tickers if t not in exclude][:limit]
        return final_list
    except Exception as e:
        print(f"無法獲取交易量清單: {e}，改用預設清單")
        return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "LINK-USD"]

def calc_pivot(df):
    r = df.iloc[-20:]
    h, l, c = r["High"].max(), r["Low"].min(), r["Close"].iloc[-1]
    p = (h + l + c) / 3
    return round(2*p - h, 4 if c < 10 else 2), round(2*p - l, 4 if c < 10 else 2)

def get_settle_report():
    if not os.path.exists(HISTORY_FILE): return ""
    df = pd.read_csv(HISTORY_FILE)
    if "settled" not in df.columns: return ""
    unsettled = df[df["settled"] == False]
    if unsettled.empty: return ""
    
    report = "\n🏁 **加密貨幣 5 日回測結算報告**\n"
    for idx, row in unsettled.iterrows():
        try:
            p_df = yf.download(row["symbol"], period="7d", auto_adjust=True, progress=False)
            exit_p = p_df["Close"].iloc[-1]
            ret = (exit_p - row["entry_price"]) / row["entry_price"]
            win = (ret > 0 and row["pred_ret"] > 0) or (ret < 0 and row["pred_ret"] < 0)
            report += f"• `{row['symbol']}` 預估 {row['pred_ret']:+.2%} | 實際 `{ret:+.2%}` {'✅' if win else '❌'}\n"
            df.at[idx, "settled"] = True
        except: continue
    df.to_csv(HISTORY_FILE, index=False)
    return report

# =========================
# 主執行程序
# =========================
def run():
    # 1. 動態獲取交易量前 300 的池子
    scan_pool = get_top_volume_tickers(300)
    # 確保固定監控的標的也在掃描名單中
    full_scan = list(set(MAIN_5 + scan_pool))
    
    print(f"🔍 正在下載與分析 {len(full_scan)} 個高交易量標的...")
    data = yf.download(full_scan, period="1y", auto_adjust=True, group_by="ticker", progress=False)
    
    results = {}
    feats = ["mom20", "bias", "vol_ratio"]

    for s in full_scan:
        try:
            df = data[s].dropna()
            if len(df) < 60: continue
            df["mom20"] = df["Close"].pct_change(20)
            df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            
            train = df.iloc[:-5].dropna()
            if len(train) < 20: continue
            
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])
            
            pred = float(model.predict(df[feats].iloc[-1:])[0])
            sup, res = calc_pivot(df)
            results[s] = {"pred": pred, "price": df["Close"].iloc[-1], "sup": sup, "res": res}
        except: continue

    # 2. 生成 Discord 報告
    msg = f"₿ **加密貨幣 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"

    # 海選 Top 5 (排除固定標的，取報酬率最高)
    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    candidates = {k: v for k, v in results.items() if k not in MAIN_5 and v["pred"] > 0}
    top_5_keys = sorted(candidates, key=lambda x: candidates[x]["pred"], reverse=True)[:5]

    msg += "🏆 **AI 海選 Top 5 (高交易量潛力股)**\n"
    for i, s in enumerate(top_5_keys):
        r = results[s]
        msg += f"{medals[i]} {s}: 預估 `{r['pred']:+.2%}`\n"
        msg += f" └ 現價: `{r['price']:.4f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    # 主流幣監控
    msg += "\n💎 **主流幣監控 (固定顯示)**\n"
    for s in MAIN_5:
        if s in results:
            r = results[s]
            msg += f"{s}: 預估 `{r['pred']:+.2%}`\n"
            msg += f" └ 現價: `{r['price']:.4f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += get_settle_report()
    msg += "\n⚠️ AI 為機率模型，僅供研究參考。"

    if WEBHOOK_URL: requests.post(WEBHOOK_URL, json={"content": msg[:1900]}, timeout=15)
    else: print(msg)

    # 存檔
    hist_list = [{"date": datetime.now().date(), "symbol": s, "entry_price": r["price"], "pred_ret": r["pred"], "settled": False} 
                 for s, r in ([(k, results[k]) for k in top_5_keys] + [(k, results[k]) for k in MAIN_5 if k in results])]
    pd.DataFrame(hist_list).to_csv(HISTORY_FILE, mode="a", header=not os.path.exists(HISTORY_FILE), index=False)

if __name__ == "__main__":
    run()
