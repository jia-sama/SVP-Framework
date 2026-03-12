import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from pathlib import Path
import warnings

# 忽略不影响学术结果的底层警告
warnings.filterwarnings("ignore")

# 动态获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent


def check_stationarity(series, name):
    """进行 ADF 单位根检验"""
    result = adfuller(series.dropna())
    p_value = result[1]
    is_stationary = p_value < 0.05
    print(
        f"[*] ADF Test for {name:<20}: p-value = {p_value:.4f} -> {'Stationary' if is_stationary else 'Non-Stationary'}")
    return is_stationary


def run_econometrics_workflow():
    print("=" * 65)
    print("Rigorous Econometrics Pipeline: ADF -> Granger -> VAR -> IRF")
    print("=" * 65)

    csv_path = BASE_DIR / "data" / "processed" / "LSTM_Multimodal_Dataset_2025.csv"
    if not csv_path.exists():
        print(f"[Error] Dataset not found at {csv_path}")
        return

    df = pd.read_csv(str(csv_path))

    # 提取核心变量
    sentiment = df['guba_avg_sentiment']
    returns = df['log_return']

    print("\n[STEP 1] Augmented Dickey-Fuller (ADF) Stationarity Test")
    # 如果序列不平稳，必须进行一阶差分
    if not check_stationarity(sentiment, 'Guba Sentiment'):
        df['guba_avg_sentiment'] = df['guba_avg_sentiment'].diff()

    if not check_stationarity(returns, 'Log Return'):
        df['log_return'] = df['log_return'].diff()

    # 清理差分可能产生的 NaN
    df_clean = df[['guba_avg_sentiment', 'log_return']].dropna().reset_index(drop=True)

    print("\n[STEP 2] Granger Causality Test with Bonferroni Correction")
    # 测试 X(情绪) 是否 Granger-cause Y(收益率)
    test_data = df_clean[['log_return', 'guba_avg_sentiment']]
    max_lag = 5

    # Bonferroni 修正阈值
    alpha = 0.05
    bonferroni_threshold = alpha / max_lag
    print(f"[*] Testing lags 1 to {max_lag}. Bonferroni adjusted alpha = {bonferroni_threshold:.4f}")

    gc_res = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

    for lag in range(1, max_lag + 1):
        # 提取 SSR based F-test 的 p-value
        p_val = gc_res[lag][0]['ssr_ftest'][1]
        is_sig = p_val < bonferroni_threshold
        marker = "✔ (Robustly Significant)" if is_sig else "❌"
        print(f"    Lag {lag}: p-value = {p_val:.4f} {marker}")

    print("\n[STEP 3] Vector Autoregression (VAR) & Cholesky Ordering")
    # 明确变量顺序：[冲击变量(Sentiment), 响应变量(Return)]
    var_data = df_clean[['guba_avg_sentiment', 'log_return']]

    model = VAR(var_data)
    # 限制 maxlags=3 以防止小样本下自由度过度消耗
    results = model.fit(maxlags=3, ic='aic')
    print(f"[*] VAR Model Selected Lag Order (AIC): {results.k_ar}")

    print("\n[STEP 4] Generating Publication-Ready Impulse Response Function (IRF)")
    # 计算 10 期的脉冲响应
    irf = results.irf(10)

    # 设置全局学术字体与样式
    plt.style.use('default')

    # 绘制正交化脉冲响应图 (仅展示 Sentiment -> Return)
    fig = irf.plot(impulse='guba_avg_sentiment', response='log_return', orth=True, figsize=(8, 5))

    # 提取当前图表的 Axis 句柄，强行覆盖 statsmodels 的默认简陋格式
    ax = fig.axes[0]

    # 1. 设定明确的学术主标题
    ax.set_title("Impulse Response of Log Return to a Guba Sentiment Shock", fontsize=13, fontweight='bold', pad=15)

    # 2. 标注 X 轴 (时间刻度)
    ax.set_xlabel("Lag (Trading Days)", fontsize=11, fontweight='medium')

    # 3. 标注 Y 轴 (响应幅度)
    ax.set_ylabel("Response Magnitude (Log Return)", fontsize=11, fontweight='medium')

    # 4. 细节优化：添加辅助网格线，增强数据对齐的可读性
    ax.grid(True, linestyle='--', alpha=0.5)

    # 5. 清除 statsmodels 自带的可能产生重叠的默认总标题
    fig.suptitle("")

    # 紧凑布局防止边缘标签被裁剪
    plt.tight_layout()

    # 确保输出目录存在
    output_dir = BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存高精度 (300 dpi) 论文配图
    save_path = output_dir / "Fig1_VAR_IRF_Sentiment_to_Return.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')

    print(f"[SUCCESS] Academic IRF plot saved strictly for thesis inclusion: {save_path}")


if __name__ == "__main__":
    run_econometrics_workflow()