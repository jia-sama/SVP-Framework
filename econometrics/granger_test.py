import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent


def check_stationarity_rigorous(series, name):
    """ADF 单位根检验报告"""
    result = adfuller(series.dropna())
    stat, p, lags, n = result[0], result[1], result[2], result[3]

    print(f"\n[ADF Test] {name}")
    print(f"  ├─ Test Statistic : {stat:.4f}")
    print(f"  ├─ p-value        : {p:.4f}")
    print(f"  ├─ Lags Used      : {lags}")
    print(f"  └─ Observations   : {n}")

    is_stationary = p < 0.05
    print(f"  => Conclusion     : {'Stationary' if is_stationary else 'Non-Stationary'}")
    return is_stationary


def run_rigorous_causality():
    print("=" * 65)
    print("Rigorous Econometrics: Bidirectional Granger Causality")
    print("=" * 65)

    csv_path = BASE_DIR / "data" / "processed" / "LSTM_Multimodal_Dataset_2025.csv"
    df = pd.read_csv(str(csv_path))

    # STEP 1: 平稳性检验
    sentiment = df['guba_avg_sentiment']
    returns = df['log_return']

    if not check_stationarity_rigorous(sentiment, 'Guba Sentiment'):
        df['guba_avg_sentiment'] = df['guba_avg_sentiment'].diff()
    if not check_stationarity_rigorous(returns, 'Log Return'):
        df['log_return'] = df['log_return'].diff()

    df_clean = df[['guba_avg_sentiment', 'log_return']].dropna().reset_index(drop=True)

    max_lag = 5
    alpha = 0.05
    bonferroni_thresh = alpha / max_lag
    print(f"\n[*] Bonferroni Adjusted Alpha (maxlag={max_lag}): {bonferroni_thresh:.4f}")

    # STEP 2: 正向检验 (Sentiment -> Return)
    # grangercausalitytests 测试第二列是否 Granger-cause 第一列
    print("\n[Hypothesis A] Sentiment -> Return (Predictive Power)")
    test_forward = df_clean[['log_return', 'guba_avg_sentiment']]
    res_forward = grangercausalitytests(test_forward, maxlag=max_lag, verbose=False)

    for lag in range(1, max_lag + 1):
        p_val = res_forward[lag][0]['ssr_ftest'][1]
        marker = "✔ (Significant)" if p_val < bonferroni_thresh else "❌"
        print(f"  Lag {lag}: p-value = {p_val:.4f} {marker}")

    # STEP 3: 反向检验 (Return -> Sentiment)
    print("\n[Hypothesis B] Return -> Sentiment (Reverse Causality / Market Reaction)")
    test_reverse = df_clean[['guba_avg_sentiment', 'log_return']]
    res_reverse = grangercausalitytests(test_reverse, maxlag=max_lag, verbose=False)

    for lag in range(1, max_lag + 1):
        p_val = res_reverse[lag][0]['ssr_ftest'][1]
        marker = "✔ (Significant)" if p_val < bonferroni_thresh else "❌"
        print(f"  Lag {lag}: p-value = {p_val:.4f} {marker}")


if __name__ == "__main__":
    run_rigorous_causality()