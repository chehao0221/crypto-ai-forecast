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

# 指定監控標的 (固定顯示區塊)
MAIN_5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

# =========================
# 工具函數
# =========================
def get_top_volume_pool():
    """自動抓取 Yahoo Finance 當日交易量最高的前 300 名標的 (包含資料清洗)"""
    try:
        headers = {'User-agent': 'Mozilla/5.0'}
        tickers = []
        for offset in [0, 100, 200]:
            url = f"https://finance.yahoo.com/crypto/?count=100&offset={offset}"
            resp = requests.get(url, headers=headers, timeout=15)
            tables = pd.read_html(resp.text)
            if not tables: continue
            
            df = tables[0]
            if 'Symbol' in df.columns:
                raw_symbols = df['Symbol'].dropna().astype(str).tolist()
                tickers.extend(raw_symbols)
        
        exclude = ["USDT-USD", "USDC-USD", "DAI-USD", "FDUSD-USD", "PYUSD-USD", "USDE-USD"]
        clean_tickers = [t for t in tickers if isinstance(t, str) and t.endswith("-USD") and t not in exclude]
        clean_tickers = list(dict.fromkeys(clean_tickers))
        return clean_tickers
    except Exception as e:
        print(f"⚠️ 抓取海選清單失敗: {e}")
        return MAIN_5

def calc_pivot(df):
    """計算支撐與壓力位"""
    r = df.iloc[-20:]
    h, l, c = r["High"].max(), r["Low"].min(), r["Close"].iloc[-1]
    p = (h + l + c) / 3
    prec = 4 if c < 10 else 2
    return round(2*p - h, prec), round(2*p - l, prec)

def get_settle_report():
    """5 日回測結算邏輯"""
    if not os.path.exists(HISTORY_FILE): return ""
    try:
        df = pd.read_csv(HISTORY_FILE)
        if "settled" not in df.columns or df[df["settled"] == False].empty:
            return "\n📊 **加密貨幣 5 日回測結算報告**：暫無待結算項目\n"
        
        unsettled = df[df["settled"] == False]
        report = "\n🏁 **加密貨幣 5 日回測結算報告**\n"
        for idx, row in unsettled.iterrows():
            try:
                p_df = yf.download(row["symbol"], period="7d", auto_adjust=True, progress=False)
                if p_df.empty: continue
                exit_p = p_df["Close"].iloc[-1]
                ret = (exit_p - row["entry_price"]) / row["entry_price"]
                win = (ret > 0 and row["pred_ret"] > 0) or (ret < 0 and row["pred_ret"] < 0)
                report += f"• `{row['symbol']}` 預估 {row['pred_ret']:+.2%} | 實際 `{ret:+.2%}` {'✅' if win else '❌'}\n"
                df.at[idx, "settled"] = True
            except: continue
        df.to_csv(HISTORY_FILE, index=False)
        return report
    except: return ""

# =========================
# 主程式：AI 分析引擎
# =========================
def run():
    vol_pool = get_top_volume_pool()
    full_scan = list(set(MAIN_5 + vol_pool))
    
    print(f"🔍 正在對 {len(full_scan)} 個高交易量標的進行 AI 分析...")
    data = yf.download(full_scan, period="2y", auto_adjust=True, group_by="ticker", progress=False)
    
    results = {}
    feats = ["mom20", "bias", "vol_ratio"]

    for s in full_scan:
        try:
            df = data[s].dropna()
            if len(df) < 100: continue
            
            df["mom20"] = df["Close"].pct_change(20)
            df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            
            train = df.iloc[:-5].dropna()
            if len(train) < 50: continue
            
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])
            
            pred = float(model.predict(df[feats].iloc[-1:])[0])
            sup, res = calc_pivot(df)
            
            results[s] = {"pred": pred, "price": df["Close"].iloc[-1], "sup": sup, "res": res}
        except: continue

    # =========================
    # 生成報告 (格式修正)
    # =========================
    msg = f"₿ **加密貨幣 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"

    # 海選邏輯：移除 pred > 0 的限制，保證選出相對最強的 5 名
    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    candidates = {k: v for k, v in results.items() if k not in MAIN_5}
    
    if not candidates:
        msg += "⚠️ AI 海選區塊：今日掃描數據不足，暫無海選結果。\n\n"
        top_5_list = []
    else:
        top_5_list = sorted(candidates.items(), key=lambda x: x[1]["pred"], reverse=True)[:5]
        msg += "🏆 **AI 海選 Top 5 (高交易量潛力幣)**\n"
        for i, (s, r) in enumerate(top_5_list):
            msg += f"{medals[i]} {s}: 預估 `{r['pred']:+.2%}`\n"
            msg += f" └ 現價: `{r['price']:.4f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"
        msg += "\n"

    # 固定監控區塊
    msg += "💎 **主流幣監控 (固定顯示)**\n"
    for s in MAIN_5:
        if s in results:
            r = results[s]
            msg += f"{s}: 預估 `{r['pred']:+.2%}`\n"
            msg += f" └ 現價: `{r['price']:.4f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += get_settle_report()
    msg += "\n💡 AI 為機率模型，僅供研究參考。投資請謹慎。"

    # 發送訊息
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:2000]}, timeout=15)
    else:
        print(msg)

    # 存檔供 5 日後結算
    save_items = top_5_list + [(k, results[k]) for k in MAIN_5 if k in results]
    hist_data = [{
        "date": datetime.now().date(),
        "symbol": s,
        "entry_price": r["price"],
        "pred_ret": r["pred"],
        "settled": False
    } for s, r in save_items]

    if hist_data:
        pd.DataFrame(hist_data).to_csv(HISTORY_FILE, mode="a", header=not os.path.exists(HISTORY_FILE), index=False)

if __name__ == "__main__":
    run()
