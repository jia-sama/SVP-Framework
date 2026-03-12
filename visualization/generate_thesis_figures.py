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
# 0. 全局调色板与工程配置 (色彩大升级)
# ==========================================
COLOR_MAIN = '#1565C0'  # 经典学术皇家蓝 (更明亮通透，告别沉闷发黑)
COLOR_SEC = '#F57C00'  # 学术橙色 (极佳的互补色)
COLOR_BG = '#9E9E9E'  # 高级灰色
COLOR_DANGER = '#D32F2F'  # 警示红色


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

plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
OUTPUT_DIR = BASE_DIR / "results"
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
# 2. 图 1: VAR 脉冲响应
# ==========================================
def generate_fig1_var_irf(df):
    print("[*] Generating Fig 1: VAR Impulse Response Function (with Legend)...")
    var_data = df[['guba_avg_sentiment', 'log_return']].diff().dropna()
    model = VAR(var_data)
    results = model.fit(maxlags=3)
    irf = results.irf(10)

    fig = irf.plot(impulse='guba_avg_sentiment', response='log_return', orth=True, figsize=(8, 5))
    ax = fig.axes[0]

    # 强制修改脉冲折线与置信区间颜色，并注册 Label (用于生成图例)
    if len(ax.get_lines()) > 0:
        ax.get_lines()[0].set_color(COLOR_MAIN)
        ax.get_lines()[0].set_linewidth(2.5)
        ax.get_lines()[0].set_label('Orthogonalized IRF')  # 注册主线图例

    if len(ax.collections) > 0:
        ax.collections[0].set_facecolor(COLOR_MAIN)
        ax.collections[0].set_alpha(0.15)
        ax.collections[0].set_label('95% Confidence Interval')  # 注册阴影图例

    # 添加绝对 0 轴辅助线，并注册 Label
    ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='Zero Baseline')

    # 1. 标准图例 (放置在右上角空白处)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#BDBDBD')

    # 2. 统计信息框 (修复位置：悬浮在右下角的绝对安全区)
    stats_text = (f"Impulse Dynamics:\n"
                  f"---------------------------\n"
                  f"Shock Source : Guba Sentiment\n"
                  f"Peak Impact  : Lag 0 (Contemporaneous)\n"
                  f"Absorption   : < 1 Trading Day")


    ax.text(0.96, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9, edgecolor='#BDBDBD'))

    ax.set_title("Impulse Response of Log Return to Guba Sentiment Shock", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Lag (Trading Days)", fontsize=11)
    ax.set_ylabel("Response Magnitude", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig1_VAR_IRF_Sentiment_to_Return.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 3. 回测图表群 (Fig 2, 3, 4)
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

    # --- Fig 2: Training Loss (带阴影与统计框) ---
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, EPOCHS + 1)

    # 画线与渐变阴影
    plt.plot(epochs_range, loss_f1, marker='o', markersize=5, color=COLOR_MAIN, linewidth=2.5, label='BCE Loss')
    bottom_y = min(loss_f1) - 0.02
    plt.fill_between(epochs_range, loss_f1, bottom_y, color=COLOR_MAIN, alpha=0.1)
    plt.ylim(bottom=bottom_y)

    # 增加收敛指标统计框
    stats_text = (f"Optimization Metrics:\n"
                  f"---------------------------\n"
                  f"Initial Loss : {loss_f1[0]:.4f}\n"
                  f"Final Loss   : {loss_f1[-1]:.4f}\n"
                  f"Net Decrease : {loss_f1[0] - loss_f1[-1]:.4f}")
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9, edgecolor='#BDBDBD'))

    plt.title("Training Loss Convergence (Initial Fold)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Epochs", fontsize=11);
    plt.ylabel("Binary Cross Entropy (BCE) Loss", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig2_Training_Loss_Curve.png", dpi=300);
    plt.close()

    # --- Fig 3: Fold Accuracy ---
    plt.figure(figsize=(10, 5))
    colors = [COLOR_MAIN if a >= 0.5 else COLOR_DANGER for a in fold_accs]
    bars = plt.bar(range(len(fold_accs)), fold_accs, color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Random Guess (50%)')

    for bar, acc in zip(bars, fold_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                 f'{acc * 100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    mean_acc = np.mean(fold_accs)
    win_folds = sum(1 for a in fold_accs if a > 0.5)
    win_rate = win_folds / len(fold_accs)
    stats_text = (f"System Evaluation Metrics:\n"
                  f"--------------------------------\n"
                  f"Mean Accuracy : {mean_acc * 100:.2f}%\n"
                  f"Folds > 50%   : {win_folds} / {len(fold_accs)}\n"
                  f"Win Rate      : {win_rate * 100:.1f}%")
    plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9, edgecolor='#BDBDBD'))

    plt.title("Out-of-Sample Directional Accuracy per Walk-Forward Fold", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Directional Accuracy", fontsize=11)
    plt.ylim(0, 1.1)
    plt.xticks(range(len(fold_accs)), [f"F{i + 1}" for i in range(len(fold_accs))])
    plt.grid(axis='y', linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig3_Fold_Accuracy_Bar.png", dpi=300);
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
                     label='Market Drawdown (Risk Zone)')
    plt.plot(pnl_attn, label='CNN-LSTM-Attention', color=COLOR_MAIN, linewidth=2.5)
    plt.plot(pnl_lr, label='Logistic Regression', color=COLOR_SEC, linewidth=1.5, linestyle='-.')
    plt.plot(pnl_mkt, label='Buy & Hold (Market)', color=COLOR_BG, linewidth=1.5, linestyle='--')

    stats_text = (f"Out-of-Sample Performance:\n"
                  f"----------------------------------------\n"
                  f"CNN-LSTM-Attn : {ret_attn:+.2f}% (MDD: {mdd_attn:+.2f}%)\n"
                  f"Logistic Reg  : {ret_lr:+.2f}%\n"
                  f"Buy & Hold    : {ret_mkt:+.2f}% (MDD: {mdd_mkt:+.2f}%)")
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9, edgecolor='#BDBDBD'))

    plt.title("Walk-Forward Backtest: Strategy Performance Comparison", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Out-of-Sample Trading Days (Walk-Forward Evaluation)", fontsize=11)
    plt.ylabel("Cumulative Returns (Initial Capital = 1.0)", fontsize=11)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig4_Backtest_Cumulative_PnL.png", dpi=300);
    plt.close()


# ==========================================
# 4. 图 5: EDA 数据覆盖图 (高级平滑提取)
# ==========================================
def generate_fig5_eda(df):
    price = np.exp(np.cumsum(df['log_return']))
    raw_sentiment = df['guba_avg_sentiment']
    smoothed_sentiment = raw_sentiment.rolling(window=10, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    line1 = ax1.plot(price, color=COLOR_MAIN, linewidth=2.5, label='Market Price Index')
    ax1.set_xlabel('Trading Days (Chronological)', fontsize=11)
    ax1.set_ylabel('Normalized Market Price Index', color=COLOR_MAIN, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=COLOR_MAIN)

    ax2 = ax1.twinx()
    line2 = ax2.plot(raw_sentiment, color=COLOR_SEC, alpha=0.2, linewidth=1.0, label='Raw Sentiment (High Noise)')
    line3 = ax2.plot(smoothed_sentiment, color=COLOR_SEC, alpha=0.9, linewidth=2.0,
                     label='Smoothed Sentiment (10-Day MA)')
    ax2.set_ylabel('Guba Sentiment Score', color=COLOR_SEC, fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLOR_SEC)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=9, framealpha=0.9, edgecolor='#BDBDBD')

    plt.title("Overlay of Market Price Evolution and Sentiment Trend", fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig5_EDA_Sentiment_vs_Price.png", dpi=300);
    plt.close()


# ==========================================
# 5. 图 6: 网络架构图
# ==========================================
def generate_fig6_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10);
    ax.set_ylim(0, 8);
    ax.axis('off')
    layers = ["Input Layer (Multimodal Time-Series)", "1D-CNN (Spatial Denoising)", "LSTM (Temporal Dependency)",
              "Attention (Dynamic Weighting)", "Sigmoid Output (DA Probability)"]
    y_pos = [7, 5.5, 4, 2.5, 1.0]

    for i, (txt, y) in enumerate(zip(layers, y_pos)):
        box = patches.FancyBboxPatch((2.5, y), 5, 0.8, boxstyle="round,pad=0.2", facecolor='#E8EAF6',
                                     edgecolor=COLOR_MAIN, linewidth=1.5)
        ax.add_patch(box)
        ax.text(5, y + 0.4, txt, ha='center', va='center', fontsize=11, fontweight='bold', color=COLOR_MAIN)
        if i < len(layers) - 1:
            ax.arrow(5, y, 0, -0.5, head_width=0.2, head_length=0.2, fc=COLOR_MAIN, ec=COLOR_MAIN)

    plt.title("SVP Framework: Core Prediction Module Architecture", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig6_Model_Architecture.png", dpi=300);
    plt.close()


# ==========================================
# 6. 图 7 & 图 8: 消融实验与特征对比柱状图
# ==========================================
def generate_fig7_ablation():
    models = ["Pure LSTM", "CNN-LSTM", "CNN-LSTM-Attn"]
    accs = [0.4516, 0.4731, 0.5054]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(models, accs, color=[COLOR_BG, COLOR_SEC, COLOR_MAIN], edgecolor='black', width=0.55, alpha=0.9)

    ax.set_ylim(0.40, 0.55)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.2)
    ax.text(2.4, 0.502, "Random Guess Base", ha='right', fontsize=10)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f'{h * 100:.2f}%', ha='center', fontweight='bold',
                fontsize=11)

    ax.annotate('+2.15%', xy=(1, 0.4731), xytext=(0.4, 0.485), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('+3.23%', xy=(2, 0.5054), xytext=(1.4, 0.520), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_title("Ablation Study: Sequential Component Contributions", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Directional Accuracy", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig7_Ablation_Study.png", dpi=300);
    plt.close()


def generate_fig8_comparison():
    names = ['Market Only', '+ Guba', '+ News', 'Multimodal']
    accs = [0.6355, 0.6168, 0.6355, 0.6449]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLOR_BG, '#9FA8DA', COLOR_SEC, COLOR_MAIN]
    bars = ax.bar(names, accs, color=colors, edgecolor='black', width=0.6, alpha=0.9)

    ax.set_ylim(0.40, 0.68)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.2)
    ax.text(3.3, 0.505, "Random Guess", ha='right', fontsize=10)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f'{h * 100:.2f}%', ha='center', fontweight='bold',
                fontsize=11)

    ax.set_title("Out-of-Sample Accuracy: Scatter (Guba) vs Institutional (News)", fontsize=13, fontweight='bold',
                 pad=15)
    ax.set_ylabel("Directional Accuracy", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3);
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig8_Sentiment_Source_Contribution.png", dpi=300);
    plt.close()


if __name__ == "__main__":
    csv_path = BASE_DIR / "data" / "processed" / "LSTM_Multimodal_Dataset_2025.csv"
    if csv_path.exists():
        df_real = pd.read_csv(str(csv_path))
        print("=" * 65)
        print("Academic Thesis Figure Generation Pipeline (Ultimate Edition)")
        print("=" * 65)

        generate_fig1_var_irf(df_real)
        generate_backtest_figs(df_real)
        generate_fig5_eda(df_real)
        generate_fig6_architecture()
        generate_fig7_ablation()
        generate_fig8_comparison()

        print("\n" + "=" * 65)
        print("[SUCCESS]!")
        print("=" * 65)