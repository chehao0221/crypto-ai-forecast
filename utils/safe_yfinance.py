import yfinance as yf
import pandas as pd
import time
import random

def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def safe_yf_download(
    tickers,
    period="2y",
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    max_chunk=80,
    max_retry=3,
    base_sleep=1.5
):
    """
    最安心的 yfinance 抓取架構
    - 分批下載
    - 批次間休息
    - 429 自動退避
    """
    all_data = {}

    for chunk in chunk_list(tickers, max_chunk):
        for attempt in range(1, max_retry + 1):
            try:
                data = yf.download(
                    chunk,
                    period=period,
                    interval=interval,
                    group_by=group_by,
                    auto_adjust=auto_adjust,
                    progress=False,
                    threads=True
                )

                if isinstance(data.columns, pd.MultiIndex):
                    for t in chunk:
                        if t in data.columns.levels[0]:
                            all_data[t] = data[t].dropna()
                else:
                    if len(chunk) == 1:
                        all_data[chunk[0]] = data.dropna()

                break

            except Exception as e:
                if attempt == max_retry:
                    print(f"⚠️ yfinance 放棄該批: {chunk[:3]}... | {e}")
                else:
                    wait = base_sleep * attempt * 20
                    print(f"⏳ yfinance retry {attempt}/{max_retry}，等待 {wait:.0f}s")
                    time.sleep(wait)

        time.sleep(base_sleep + random.uniform(0.5, 1.5))

    return all_data
