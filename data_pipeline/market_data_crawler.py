import akshare as ak
import pandas as pd
import numpy as np
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime, timedelta

proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
for key in proxy_keys:
    if key in os.environ:
        del os.environ[key]

class MarketDataEngineer:
    def __init__(self, start_date="20250101", end_date="20251231"):
        self.start_date = start_date
        self.end_date = end_date
        self.start_date_dash = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        self.end_date_dash = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

        # 调整取数起点以满足滚动指标 (Rolling Windows) 的预热期需求
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        fetch_start_dt = start_dt - timedelta(days=40)
        self.fetch_start_date = fetch_start_dt.strftime("%Y%m%d")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=20))
    def _fetch_ssec_history(self):
        print(f"[*] Fetching SSEC daily data (start: {self.fetch_start_date})...")
        df = ak.stock_zh_index_daily_em(symbol="sh000001", start_date=self.fetch_start_date, end_date=self.end_date)
        if df.empty:
            raise ValueError("Empty data returned for SSEC.")
        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
    def _fetch_valuation_features(self):
        print("[*] Fetching CSIndex valuation data...")
        df = ak.stock_zh_index_value_csindex(symbol="000001")
        if not df.empty:
            df['date'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
            mask = (df['date'] >= self.start_date_dash) & (df['date'] <= self.end_date_dash)
            return df.loc[mask]
        return pd.DataFrame()

    def build_feature_matrix(self):
        print("-" * 60)
        print(f"Building Feature Matrix | Target Window: {self.start_date_dash} to {self.end_date_dash}")
        print("-" * 60)

        try:
            ssec_df = self._fetch_ssec_history()
            ssec_df['date'] = pd.to_datetime(ssec_df['date']).dt.strftime('%Y-%m-%d')
            ssec_df = ssec_df.sort_values(by='date').reset_index(drop=True)

            ssec_df['pre_close'] = ssec_df['close'].shift(1)
            ssec_df['log_return'] = np.log(ssec_df['close'] / ssec_df['pre_close'])
            ssec_df['amplitude_pct'] = (ssec_df['high'] - ssec_df['low']) / ssec_df['pre_close'] * 100

            # 计算历史已实现波动率 (HRV, 年化)
            ssec_df['volatility_5d'] = ssec_df['log_return'].rolling(window=5).std() * np.sqrt(252) * 100
            ssec_df['volatility_20d'] = ssec_df['log_return'].rolling(window=20).std() * np.sqrt(252) * 100

            # 计算乖离率 (Bias Ratio)
            ssec_df['ma_20'] = ssec_df['close'].rolling(window=20).mean()
            ssec_df['bias_ratio_20d'] = (ssec_df['close'] - ssec_df['ma_20']) / ssec_df['ma_20'] * 100

            # 截断至目标时间窗口
            mask = (ssec_df['date'] >= self.start_date_dash) & (ssec_df['date'] <= self.end_date_dash)
            ssec_df = ssec_df.loc[mask].copy()

            try:
                time.sleep(3)
                val_df = self._fetch_valuation_features()
                if not val_df.empty:
                    val_df = val_df[['date', '市盈率1', '股息率1']].rename(columns={
                        '市盈率1': 'pe_ratio',
                        '股息率1': 'dividend_yield'
                    })
                    ssec_df = pd.merge(ssec_df, val_df, on='date', how='left')
                    ssec_df['pe_ratio'] = ssec_df['pe_ratio'].ffill()
                    ssec_df['dividend_yield'] = ssec_df['dividend_yield'].ffill()
            except Exception as e:
                print(f"[WARN] Failed to merge Valuation Features: {e}")

            # 剔除了容易卡死和报错的冗余特征，只保留最核心的学术因子
            final_columns = [
                'date', 'open', 'close', 'high', 'low', 'volume', 'amount',
                'log_return', 'amplitude_pct', 'volatility_5d', 'volatility_20d', 'bias_ratio_20d',
                'pe_ratio', 'dividend_yield'
            ]

            final_columns = [col for col in final_columns if col in ssec_df.columns]
            ssec_df = ssec_df[final_columns]

            file_name = "../data/raw/ssec_ultimate_features_2025.csv"
            ssec_df.to_csv(file_name, index=False, encoding="utf-8-sig")

            print(f"[*] Matrix exported to {file_name}")

        except Exception as e:
            print(f"[ERROR] Pipeline execution failed: {e}")

if __name__ == "__main__":
    engineer = MarketDataEngineer(start_date="20250101", end_date="20251231")
    engineer.build_feature_matrix()