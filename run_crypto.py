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

# 固定監控區
MAIN_5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

# =========================
# 工具函數：最強健的海選池抓取
# =========================
def get_top_volume_pool():
    headers = {'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    tickers = []
    
    # 策略 A: 抓取 Yahoo Crypto 篩選器 (前 300 名)
    try:
        for offset in [0, 100, 200]:
            url = f"https://finance.yahoo.com/crypto/?count=100&offset={offset}"
            resp = requests.get(url, headers=headers, timeout=15)
            # 使用 html5lib 增加解析穩定性
            tables = pd.read_html(resp.text, flavor='html5lib')
            if tables:
                df = tables[0]
                if 'Symbol' in df.columns:
                    tickers.extend(df['Symbol'].dropna().astype(str).tolist())
    except Exception as e:
        print(f"策略 A 失敗: {e}")

    # 策略 B: 如果 A 抓到的太少，嘗試抓取 Trending 或熱門標的
    if len(tickers) < 10:
        try:
            url = "https://finance.yahoo.com/crypto"
            resp = requests.get(url, headers=headers, timeout=15)
            tables = pd.read_html(resp.text)
            if tables:
                tickers.extend(tables[0]['Symbol'].dropna().astype(str).tolist())
        except: pass

# 策略 C: 強制保底名單 (擴展至 300 檔，涵蓋全市場熱門標的)
    backup_list = [
        # --- 前 50 大主流與基礎設施 ---
        "ADA-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "AVAX-USD", "SHIB-USD", "TRX-USD",
        "LTC-USD", "BCH-USD", "UNI-USD", "NEAR-USD", "FIL-USD", "APT-USD", "ARB-USD", "OP-USD",
        "STX-USD", "ICP-USD", "ETC-USD", "HBAR-USD", "KAS-USD", "INJ-USD", "TIA-USD", "SUI-USD",
        "SEI-USD", "ATOM-USD", "IMX-USD", "VET-USD", "EGLD-USD", "ALGO-USD", "FTM-USD", "FLOW-USD",
        "MNT-USD", "ASTR-USD", "METIS-USD", "SKL-USD", "STRK-USD", "MANTA-USD", "ZK-USD", "AVAX-USD",
        "RENDER-USD", "HNT-USD", "KAVA-USD", "AXL-USD", "MINA-USD", "RON-USD", "DYM-USD", "PYTH-USD",
        
        # --- AI 賽道與預言機 ---
        "FET-USD", "TAO-USD", "AKT-USD", "THETA-USD", "GRT-USD", "AGIX-USD", "OCEAN-USD", "ARKM-USD",
        "NOS-USD", "GLM-USD", "PRIME-USD", "AIOZ-USD", "GPT-USD", "PHB-USD", "RSS3-USD", "TRAC-USD",
        "RLC-USD", "NMR-USD", "BAND-USD", "API3-USD", "LINK-USD", "UMA-USD", "TRB-USD", "ORAI-USD",
        
        # --- Meme 幣 (高波動、海選常客) ---
        "PEPE-USD", "WIF-USD", "FLOKI-USD", "BONK-USD", "MEME-USD", "BOME-USD", "TURBO-USD", "MYRO-USD",
        "WEN-USD", "MOG-USD", "POPCAT-USD", "BRETT-USD", "DEGEN-USD", "SLERF-USD", "MEW-USD", "MANEKI-USD",
        "ELON-USD", "BABYDOGE-USD", "VINU-USD", "COQ-USD", "TOSHI-USD", "LADYS-USD", "AIDOGE-USD", "PEOPLE-USD",
        
        # --- DeFi, DEX 與質押 ---
        "AAVE-USD", "MKR-USD", "LDO-USD", "CRV-USD", "SNX-USD", "JUP-USD", "PENDLE-USD", "DYDX-USD",
        "COMP-USD", "RUNE-USD", "CAKE-USD", "ENA-USD", "ETHFI-USD", "RAY-USD", "COW-USD", "BAL-USD",
        "SUSHI-USD", "JOE-USD", "ORCA-USD", "GMX-USD", "ID-USD", "ENS-USD", "RPL-USD", "SSV-USD",
        "LQTY-USD", "ANKR-USD", "ZRX-USD", "1INCH-USD", "KNC-USD", "BICO-USD", "CVX-USD", "FXS-USD",
        
        # --- GameFi, Metaverse 與 NFT ---
        "SAND-USD", "MANA-USD", "AXS-USD", "GALA-USD", "ENJ-USD", "BEAM-USD", "PIXEL-USD", "ILV-USD",
        "YGG-USD", "MAGIC-USD", "BIGTIME-USD", "APE-USD", "ALICE-USD", "WAXP-USD", "GMT-USD", "SUPER-USD",
        "MBOX-USD", "XAI-USD", "PORTAL-USD", "ACE-USD", "GHST-USD", "CHR-USD", "UOS-USD", "NAKA-USD",
        
        # --- Layer 1 / Layer 2 擴展 ---
        "METIS-USD", "LOOM-USD", "EVMOS-USD", "CANTO-USD", "KTS-USD", "GLMR-USD", "MOVR-USD", "ROSE-USD",
        "IOTX-USD", "ONT-USD", "NEO-USD", "QTUM-USD", "GAS-USD", "ZIL-USD", "ONE-USD", "CELO-USD",
        "KSM-USD", "SCRT-USD", "ZEN-USD", "ICX-USD", "LSK-USD", "WAVES-USD", "RVN-USD", "CORE-USD",
        
        # --- 存儲、隱私與計算 ---
        "AR-USD", "STORJ-USD", "SC-USD", "BLZ-USD", "XMR-USD", "ZEC-USD", "DASH-USD", "DCR-USD",
        "BEAM-USD", "BEL-USD", "PHA-USD", "CTSI-USD", "CVC-USD", "STPT-USD", "UTK-USD", "POND-USD",
        
        # --- 銘文與比特幣生態 ---
        "ORDI-USD", "SATS-USD", "RATS-USD", "MUBI-USD", "STX-USD", "ALEX-USD", "RIF-USD", "BADGER-USD",
        "CKB-USD", "LEVER-USD", "AUCTION-USD", "BAKE-USD", "T-USD", "REN-USD", "WBTC-USD",
        
        # --- 其他熱門與潛力標的 ---
        "CHZ-USD", "IOTA-USD", "XLM-USD", "XEC-USD", "WOO-USD", "JASMY-USD", "BTT-USD", "HOT-USD",
        "ZIG-USD", "TEL-USD", "CSPR-USD", "TWT-USD", "RVN-USD", "LRC-USD", "KDA-USD", "SYS-USD",
        "NKN-USD", "DGB-USD", "XVG-USD", "STEEM-USD", "SXP-USD", "ARK-USD", "SCR-USD", "GNS-USD",
        "POLY-USD", "LIT-USD", "MASK-USD", "LPT-USD", "TRB-USD", "GTC-USD", "NMR-USD", "OXT-USD",
        "RNDR-USD", "HIGH-USD", "PROMS-USD", "HOOK-USD", "EDU-USD", "ID-USD", "MAV-USD", "RDNT-USD",
        "ARKM-USD", "CYBER-USD", "WLD-USD", "IQ-USD", "OAX-USD", "MDT-USD", "NFP-USD", "AI-USD",
        "XVS-USD", "FORTH-USD", "OOKI-USD", "VGX-USD", "PROM-USD", "VTHO-USD", "DENT-USD", "FUN-USD",
        "STMX-USD", "STRAX-USD", "DATA-USD", "NULS-USD", "MTL-USD", "PNT-USD", "QLC-USD", "POWR-USD",
        "AMB-USD", "KMD-USD", "ARK-USD", "ADX-USD", "GAS-USD", "VIB-USD", "REQ-USD", "RDN-USD",
        "GTO-USD", "OST-USD", "LUN-USD", "MTH-USD", "WPR-USD", "SNM-USD", "NCASH-USD", "TNB-USD",
        "VIC-USD", "LOOM-USD", "KEY-USD", "PIVX-USD", "CLV-USD", "BAKE-USD", "REI-USD", "TKO-USD"
    ]
    
    exclude = ["USDT-USD", "USDC-USD", "DAI-USD", "FDUSD-USD", "PYUSD-USD"]
    clean_tickers = [t for t in tickers if isinstance(t, str) and t.endswith("-USD") and t not in exclude]
    
    # 合併並去重
    final_pool = list(dict.fromkeys(clean_tickers + backup_list))
    return final_pool

def calc_pivot(df):
    r = df.iloc[-20:]
    h, l, c = r["High"].max(), r["Low"].min(), r["Close"].iloc[-1]
    p = (h + l + c) / 3
    prec = 4 if c < 10 else 2
    return round(2*p - h, prec), round(2*p - l, prec)

def get_settle_report():
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
# 主程式
# =========================
def run():
    full_pool = get_top_volume_pool()
    scan_list = list(set(MAIN_5 + full_pool))
    
    print(f"🔍 掃描池總數: {len(scan_list)}")
    data = yf.download(scan_list, period="2y", auto_adjust=True, group_by="ticker", progress=False)
    
    results = {}
    feats = ["mom20", "bias", "vol_ratio"]

    for s in scan_list:
        try:
            df = data[s].dropna()
            if len(df) < 80: continue
            df["mom20"] = df["Close"].pct_change(20)
            df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            train = df.iloc[:-5].dropna()
            if len(train) < 40: continue
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])
            pred = float(model.predict(df[feats].iloc[-1:])[0])
            sup, res = calc_pivot(df)
            results[s] = {"pred": pred, "price": df["Close"].iloc[-1], "sup": sup, "res": res}
        except: continue

    msg = f"₿ **加密貨幣 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"

    # 海選 Top 5 (排除固定監控)
    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    candidates = {k: v for k, v in results.items() if k not in MAIN_5}
    
    if not candidates:
        msg += "⚠️ 警報：海選數據抓取異常，請檢查網路連接。\n\n"
        top_5_list = []
    else:
        top_5_list = sorted(candidates.items(), key=lambda x: x[1]["pred"], reverse=True)[:5]
        msg += "🏆 **AI 海選 Top 5 (潛力標的)**\n"
        for i, (s, r) in enumerate(top_5_list):
            msg += f"{medals[i]} {s}: 預估 `{r['pred']:+.2%}`\n"
            msg += f" └ 現價: `{r['price']:.4f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"
        msg += "\n"

    # 固定監控
    msg += "💎 **主流幣監控 (固定顯示)**\n"
    for s in MAIN_5:
        if s in results:
            r = results[s]
            msg += f"{s}: 預估 `{r['pred']:+.2%}`\n"
            msg += f" └ 現價: `{r['price']:.4f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += get_settle_report()
    msg += "\n💡 AI 為機率模型，僅供研究參考。投資請謹慎。"

    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:2000]}, timeout=15)
    else:
        print(msg)

    save_items = top_5_list + [(k, results[k]) for k in MAIN_5 if k in results]
    hist_data = [{"date": datetime.now().date(), "symbol": s, "entry_price": r["price"], "pred_ret": r["pred"], "settled": False} for s, r in save_items]
    if hist_data:
        pd.DataFrame(hist_data).to_csv(HISTORY_FILE, mode="a", header=not os.path.exists(HISTORY_FILE), index=False)

if __name__ == "__main__":
    run()
