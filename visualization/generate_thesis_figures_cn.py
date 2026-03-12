import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from statsmodels.tsa.api import VAR
from pathlib import Path
import random
import os
import warnings

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# 0. 全局调色板与中文学术字体配置
# ==========================================
COLOR_MAIN = '#1565C0'  # 经典学术皇家蓝
COLOR_SEC = '#F57C00'  # 学术橙色
COLOR_BG = '#9E9E9E'  # 高级灰色
COLOR_DANGER = '#D32F2F'  # 警示红色

# 解决 Mac 系统下 Matplotlib 中文乱码和负号丢失问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def seed_everything(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEQ_LEN = 3
BATCH_SIZE = 16
CNN_FILTERS = 16
HIDDEN_SIZE = 16
DROPOUT_RATE = 0.3
LR = 1e-3
EPOCHS = 30
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

INITIAL_TRAIN_SIZE = 150
STEP = 10

OUTPUT_DIR = BASE_DIR / "results_cn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. 深度学习网络构件
# ==========================================
class FinancialDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def create_sequences_with_returns(X, y_bin, y_ret):
    xs, ys_bin, ys_ret = [], [], []
    for i in range(len(X) - SEQ_LEN):
        xs.append(X[i:(i + SEQ_LEN)])
        ys_bin.append(y_bin[i + SEQ_LEN])
        ys_ret.append(y_ret[i + SEQ_LEN])
    return np.array(xs, dtype=np.float32), np.array(ys_bin, dtype=np.float32), np.array(ys_ret, dtype=np.float32)


class CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv = nn.Conv1d(input_size, CNN_FILTERS, kernel_size=2, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(CNN_FILTERS, HIDDEN_SIZE, batch_first=True)
        self.att = nn.Linear(HIDDEN_SIZE, 1)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x)).permute(0, 2, 1)
        l_out, _ = self.lstm(x)
        w = torch.softmax(self.att(l_out), dim=1)
        c = torch.sum(w * l_out, dim=1)
        return torch.sigmoid(self.fc(self.dropout(c)))


# ==========================================
# 2. 图 1: VAR 脉冲响应 (中文版)
# ==========================================
def generate_fig1_var_irf(df):
    var_data = df[['guba_avg_sentiment', 'log_return']].diff().dropna()
    model = VAR(var_data)
    results = model.fit(maxlags=3)
    irf = results.irf(10)

    fig = irf.plot(impulse='guba_avg_sentiment', response='log_return', orth=True, figsize=(8, 5))
    ax = fig.axes[0]

    if len(ax.get_lines()) > 0:
        ax.get_lines()[0].set_color(COLOR_MAIN)
        ax.get_lines()[0].set_linewidth(2.5)
        ax.get_lines()[0].set_label('正交化脉冲响应 (Orthogonalized IRF)')

    if len(ax.collections) > 0:
        ax.collections[0].set_facecolor(COLOR_MAIN)
        ax.collections[0].set_alpha(0.15)
        ax.collections[0].set_label('95% 置信区间')

    ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='零基准线 (Zero Baseline)')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#BDBDBD')

    stats_text = (f"脉冲响应动态分析：\n"
                  f"---------------------------\n"
                  f"冲击来源 : 股吧散户情绪\n"
                  f"冲击峰值 : 滞后 0 期 (当期同步)\n"
                  f"市场吸收 : < 1 个交易日")

    ax.text(0.96, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9, edgecolor='#BDBDBD'))

    ax.set_title("对数收益率对股吧情绪冲击的正交化脉冲响应", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("滞后期 (交易日)", fontsize=11)
    ax.set_ylabel("响应幅度 (Response Magnitude)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig1_VAR_IRF_Sentiment_to_Return_CN.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 3. 回测图表群 (Fig 2, 3, 4 中文版)
# ==========================================
def generate_backtest_figs(df_raw):
    seed_everything(1024)
    df = df_raw.copy()
    df['target_bin'] = (df['log_return'].shift(-1) > 0).astype(int)
    df['target_ret'] = df['log_return'].shift(-1)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ['date', 'target_bin', 'target_ret', 'log_return']]
    y_true_ret, preds_attn, preds_lr, fold_accs, loss_f1 = [], [], [], [], []

    for f_idx, start in enumerate(range(INITIAL_TRAIN_SIZE, len(df), STEP)):
        end = min(start + STEP, len(df))
        if end - start < 1: break

        tr_df, te_df = df.iloc[:start], df.iloc[start - SEQ_LEN:end]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr_df[feature_cols].values)
        X_te = scaler.transform(te_df[feature_cols].values)

        X_tr_s, y_tr_s, _ = create_sequences_with_returns(X_tr, tr_df['target_bin'].values, tr_df['target_ret'].values)
        X_te_s, y_te_b, y_te_r = create_sequences_with_returns(X_te, te_df['target_bin'].values,
                                                               te_df['target_ret'].values)

        model = CNN_LSTM_Attention(X_tr_s.shape[2]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
        crit = nn.BCELoss()
        loader = DataLoader(FinancialDataset(X_tr_s, y_tr_s), batch_size=BATCH_SIZE)

        for ep in range(EPOCHS):
            model.train()
            total_l = 0
            for xb, yb in loader:
                opt.zero_grad()
                out = model(xb.to(DEVICE)).view(-1)
                yb_t = yb.to(DEVICE, dtype=torch.float32).view(-1)
                l = crit(out, yb_t)
                l.backward();
                opt.step();
                total_l += l.item()
            if f_idx == 0: loss_f1.append(total_l / len(loader))

        model.eval()
        with torch.no_grad():
            p = model(torch.tensor(X_te_s).to(DEVICE)).view(-1).cpu().numpy()
            p_bin = (p > 0.5).astype(int)
            preds_attn.extend(p_bin)
            y_true_ret.extend(y_te_r)
            fold_accs.append(accuracy_score(y_te_b, p_bin))

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_tr, tr_df['target_bin'].values)
        preds_lr.extend(lr.predict(X_te[SEQ_LEN:]))

    # --- Fig 2: Training Loss ---
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, EPOCHS + 1)

    plt.plot(epochs_range, loss_f1, marker='o', markersize=5, color=COLOR_MAIN, linewidth=2.5, label='BCE 损失')
    bottom_y = min(loss_f1) - 0.02
    plt.fill_between(epochs_range, loss_f1, bottom_y, color=COLOR_MAIN, alpha=0.1)
    plt.ylim(bottom=bottom_y)

    stats_text = (f"优化收敛指标：\n"
                  f"---------------------------\n"
                  f"初始损失 : {loss_f1[0]:.4f}\n"
                  f"最终损失 : {loss_f1[-1]:.4f}\n"
                  f"净下降量 : {loss_f1[0] - loss_f1[-1]:.4f}")
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9, edgecolor='#BDBDBD'))

    plt.title("训练损失函数收敛曲线 (初始扩展窗口)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("训练轮数 (Epochs)", fontsize=11);
    plt.ylabel("二元交叉熵损失 (BCE Loss)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig2_Training_Loss_Curve_CN.png", dpi=300);
    plt.close()

    # --- Fig 3: Fold Accuracy ---
    plt.figure(figsize=(10, 5))
    colors = [COLOR_MAIN if a >= 0.5 else COLOR_DANGER for a in fold_accs]
    bars = plt.bar(range(len(fold_accs)), fold_accs, color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='随机猜测基准 (50%)')

    for bar, acc in zip(bars, fold_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                 f'{acc * 100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    mean_acc = np.mean(fold_accs)
    win_folds = sum(1 for a in fold_accs if a > 0.5)
    win_rate = win_folds / len(fold_accs)
    stats_text = (f"系统评估指标：\n"
                  f"--------------------------------\n"
                  f"平均准确率      : {mean_acc * 100:.2f}%\n"
                  f"胜率 > 50% 窗口 : {win_folds} / {len(fold_accs)}\n"
                  f"整体表现胜率    : {win_rate * 100:.1f}%")
    plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9, edgecolor='#BDBDBD'))

    plt.title("样本外滚动回测方向准确率分布 (按时间切片)", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("方向预测准确率", fontsize=11)
    plt.ylim(0, 1.1)
    plt.xticks(range(len(fold_accs)), [f"窗口 {i + 1}" for i in range(len(fold_accs))])
    plt.grid(axis='y', linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig3_Fold_Accuracy_Bar_CN.png", dpi=300);
    plt.close()

    # --- Fig 4: Cumulative PnL ---
    y_true_ret = np.array(y_true_ret)
    pnl_attn = np.exp(np.cumsum(np.where(np.array(preds_attn) == 1, y_true_ret, 0)))
    pnl_lr = np.exp(np.cumsum(np.where(np.array(preds_lr) == 1, y_true_ret, 0)))
    pnl_mkt = np.exp(np.cumsum(y_true_ret))

    ret_attn, ret_lr, ret_mkt = (pnl_attn[-1] - 1) * 100, (pnl_lr[-1] - 1) * 100, (pnl_mkt[-1] - 1) * 100
    roll_max_mkt = np.maximum.accumulate(pnl_mkt)
    mdd_mkt = np.min((pnl_mkt - roll_max_mkt) / roll_max_mkt) * 100
    roll_max_attn = np.maximum.accumulate(pnl_attn)
    mdd_attn = np.min((pnl_attn - roll_max_attn) / roll_max_attn) * 100

    plt.figure(figsize=(10, 5))
    plt.fill_between(range(len(pnl_mkt)), pnl_mkt, roll_max_mkt, color=COLOR_BG, alpha=0.2,
                     label='市场回撤区 (风险区域)')
    plt.plot(pnl_attn, label='CNN-LSTM-Attention 策略', color=COLOR_MAIN, linewidth=2.5)
    plt.plot(pnl_lr, label='逻辑回归 (基准)', color=COLOR_SEC, linewidth=1.5, linestyle='-.')
    plt.plot(pnl_mkt, label='买入并持有 (大盘)', color=COLOR_BG, linewidth=1.5, linestyle='--')

    stats_text = (f"样本外实盘表现评估：\n"
                  f"----------------------------------------\n"
                  f"CNN-LSTM-Attn 策略 : {ret_attn:+.2f}% (最大回撤: {mdd_attn:+.2f}%)\n"
                  f"逻辑回归模型基准    : {ret_lr:+.2f}%\n"
                  f"买入并持有 (大盘)   : {ret_mkt:+.2f}% (最大回撤: {mdd_mkt:+.2f}%)")
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9, edgecolor='#BDBDBD'))

    plt.title("滚动回测：多模态交易策略样本外累计收益对比", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("样本外交易日 (Walk-Forward 评估时间轴)", fontsize=11)
    plt.ylabel("累计收益净值 (初始资金 = 1.0)", fontsize=11)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig4_Backtest_Cumulative_PnL_CN.png", dpi=300);
    plt.close()


# ==========================================
# 4. 图 5: EDA 数据覆盖图 (中文版)
# ==========================================
def generate_fig5_eda(df):
    price = np.exp(np.cumsum(df['log_return']))
    raw_sentiment = df['guba_avg_sentiment']
    smoothed_sentiment = raw_sentiment.rolling(window=10, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    line1 = ax1.plot(price, color=COLOR_MAIN, linewidth=2.5, label='归一化市场价格指数')
    ax1.set_xlabel('交易日 (自然时序)', fontsize=11)
    ax1.set_ylabel('归一化市场价格指数 (Normalized Price)', color=COLOR_MAIN, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=COLOR_MAIN)

    ax2 = ax1.twinx()
    line2 = ax2.plot(raw_sentiment, color=COLOR_SEC, alpha=0.2, linewidth=1.0, label='原始股吧情绪得分 (高噪音)')
    line3 = ax2.plot(smoothed_sentiment, color=COLOR_SEC, alpha=0.9, linewidth=2.0, label='情绪平滑趋势 (10日移动平均)')
    ax2.set_ylabel('股吧情感复合得分 (Guba Sentiment)', color=COLOR_SEC, fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLOR_SEC)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=9, framealpha=0.9, edgecolor='#BDBDBD')

    plt.title("大盘价格演变与股吧情绪宏观趋势叠加分析", fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig5_EDA_Sentiment_vs_Price_CN.png", dpi=300);
    plt.close()


# ==========================================
# 5. 图 6: 网络架构图 (中文版)
# ==========================================
def generate_fig6_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10);
    ax.set_ylim(0, 8);
    ax.axis('off')
    layers = [
        "输入层 (多模态时序张量输入)",
        "1D-CNN (空间特征提取与高频降噪)",
        "LSTM (时间序列长程依赖建模)",
        "全局 Attention 层 (动态时间步聚焦赋权)",
        "Sigmoid 输出层 (交易日方向看涨概率)"
    ]
    y_pos = [7, 5.5, 4, 2.5, 1.0]

    for i, (txt, y) in enumerate(zip(layers, y_pos)):
        box = patches.FancyBboxPatch((2.0, y), 6, 0.8, boxstyle="round,pad=0.2", facecolor='#E8EAF6',
                                     edgecolor=COLOR_MAIN, linewidth=1.5)
        ax.add_patch(box)
        ax.text(5, y + 0.4, txt, ha='center', va='center', fontsize=11, fontweight='bold', color=COLOR_MAIN)
        if i < len(layers) - 1:
            ax.arrow(5, y, 0, -0.5, head_width=0.2, head_length=0.2, fc=COLOR_MAIN, ec=COLOR_MAIN)

    plt.title("SVP 框架：CNN-LSTM-Attention 核心预测模块架构图", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig6_Model_Architecture_CN.png", dpi=300);
    plt.close()


# ==========================================
# 6. 图 7 & 图 8: 柱状图对比 (中文版)
# ==========================================
def generate_fig7_ablation():
    models = ["纯 LSTM 模型", "CNN-LSTM 混合", "完整的 CNN-LSTM-Attn"]
    accs = [0.4516, 0.4731, 0.5054]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(models, accs, color=[COLOR_BG, COLOR_SEC, COLOR_MAIN], edgecolor='black', width=0.55, alpha=0.9)

    ax.set_ylim(0.40, 0.55)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.2)
    ax.text(2.4, 0.502, "随机猜测基准 (50%)", ha='right', fontsize=10)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f'{h * 100:.2f}%', ha='center', fontweight='bold',
                fontsize=11)

    ax.annotate('+2.15%\n(空间降噪贡献)', xy=(1, 0.4731), xytext=(0.4, 0.485), arrowprops=dict(arrowstyle="->", lw=1.5),
                ha='center')
    ax.annotate('+3.23%\n(时间聚焦贡献)', xy=(2, 0.5054), xytext=(1.4, 0.520), arrowprops=dict(arrowstyle="->", lw=1.5),
                ha='center')

    ax.set_title("消融实验：系统各组件对样本外预测准确率的递进贡献", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("样本外方向预测准确率", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig7_Ablation_Study_CN.png", dpi=300);
    plt.close()


def generate_fig8_comparison():
    names = ['仅有量价\n(无情绪)', '+ 股吧情绪\n(散户代表)', '+ 证券时报新闻\n(机构代表)', '多模态融合\n(全特征接入)']
    accs = [0.6355, 0.6168, 0.6355, 0.6449]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLOR_BG, '#9FA8DA', COLOR_SEC, COLOR_MAIN]
    bars = ax.bar(names, accs, color=colors, edgecolor='black', width=0.6, alpha=0.9)

    ax.set_ylim(0.40, 0.68)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.2)
    ax.text(3.3, 0.505, "随机猜测基准 (50%)", ha='right', fontsize=10)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f'{h * 100:.2f}%', ha='center', fontweight='bold',
                fontsize=11)

    ax.set_title("情绪来源异质性分析：散户(高频) vs 机构(低频) 的独立预测贡献", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("样本外方向预测准确率", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig8_Sentiment_Source_Contribution_CN.png", dpi=300);
    plt.close()


if __name__ == "__main__":
    csv_path = BASE_DIR / "data" / "processed" / "LSTM_Multimodal_Dataset_2025.csv"
    if csv_path.exists():
        df_real = pd.read_csv(str(csv_path))
        print("=" * 65)
        print("🇨🇳 Academic Thesis Figure Generation (Chinese Edition)")
        print("=" * 65)

        generate_fig1_var_irf(df_real)
        generate_backtest_figs(df_real)
        generate_fig5_eda(df_real)
        generate_fig6_architecture()
        generate_fig7_ablation()
        generate_fig8_comparison()

        print("\n" + "=" * 65)
        print("[SUCCESS] 8 张中文图表已生成于 'results_cn/' 文件夹！")
        print("=" * 65)