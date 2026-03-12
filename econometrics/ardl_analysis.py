import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ardl import ARDL
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent


def run_ardl_analysis():
    print("=" * 70)
    print("Econometrics: Autoregressive Distributed Lag (ARDL) Modeling")
    print("=" * 70)

    csv_path = BASE_DIR / "data" / "processed" / "LSTM_Multimodal_Dataset_2025.csv"
    if not csv_path.exists():
        print(f"[Error] Dataset not found at {csv_path}")
        return

    df = pd.read_csv(str(csv_path))

    # 提取序列并去空值
    data = df[['log_return', 'guba_avg_sentiment']].dropna().reset_index(drop=True)

    # 目标变量 (Endogenous): 收益率
    y = data['log_return']
    # 外生变量 (Exogenous): 股吧情绪
    X = data[['guba_avg_sentiment']]

    print("[*] Fitting ARDL Model: log_return ~ lags(log_return) + lags(guba_sentiment)")
    print("[*] Maximum lags set to 5 (Based on Granger causality tests)...")

    # 构建 ARDL 模型
    # lags=5 表示 y (收益率) 自身滞后最多 5 阶
    # order=5 表示 X (情绪) 滞后最多 5 阶
    # 实际应用中，可以通过循环对比 AIC 找出最优组合，这里为了清晰直接使用最大阶数，
    # 观察哪些滞后阶数的 p-value < 0.05
    model = ARDL(endog=y, lags=5, exog=X, order=5)
    results = model.fit()

    print("\n" + "=" * 70)
    print("ARDL MODEL SUMMARY")
    print("=" * 70)
    # 打印完整的学术统计表
    print(results.summary())
    print("=" * 70)

    # 提取核心结论：寻找显著的情绪滞后项
    print("\n[*] Academic Diagnosis (Focusing on Sentiment Lags):")
    pvalues = results.pvalues
    significant_found = False

    for idx, p_val in pvalues.items():
        if 'guba_avg_sentiment' in idx and p_val < 0.05:
            coef = results.params[idx]
            direction = "Positive (+)" if coef > 0 else "Negative (-)"
            print(f"  -> {idx}: Significant at 5% level (p={p_val:.4f}). Effect: {direction}")
            significant_found = True

    if not significant_found:
        print("  -> No statistically significant sentiment lags found at the 5% level in the ARDL framework.")
        print(
            "     (This indicates sentiment effects might be fully absorbed by return autocorrelations, or the relationship is purely non-linear.)")


if __name__ == "__main__":
    run_ardl_analysis()