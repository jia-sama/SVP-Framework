import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import binomtest, pearsonr
from pathlib import Path
import random
import os
import warnings

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================
# 0. 严格锁死随机种子 (MPS 兼容版)
# ==========================================
def seed_everything(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. 极简 LSTM 架构 (The Occam's Razor Fix)
# ==========================================
# 【重构点】：严格依据 Granger 滞后 2-3 天的先验，斩断冗余记忆
SEQ_LEN = 3
BATCH_SIZE = 16  # 小样本下减小 Batch Size 以增加梯度更新震荡，跳出局部最优
HIDDEN_SIZE = 8  # 极度压缩神经元容量，剥夺死记硬背的能力
NUM_LAYERS = 1  # 砍为单层
DROPOUT_RATE = 0.3
LR = 1e-3
EPOCHS = 30
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

INITIAL_TRAIN_SIZE = 150
STEP = 10


class FinancialDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class MiniatureLSTM(nn.Module):
    def __init__(self, input_size):
        super(MiniatureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        # 单层 LSTM 无法在内部使用 dropout 参数，必须显式在外部加一层 Dropout
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # 仅对最后一个时间步施加极强的 Dropout
        return self.sigmoid(self.fc(out))


# 修改序列构造器：不仅返回二分类 label，还返回【真实的连续收益率】以计算 IC
def create_sequences_with_returns(X, y_bin, y_ret):
    xs, ys_bin, ys_ret = [], [], []
    for i in range(len(X) - SEQ_LEN):
        xs.append(X[i:(i + SEQ_LEN)])
        ys_bin.append(y_bin[i + SEQ_LEN])
        ys_ret.append(y_ret[i + SEQ_LEN])
    return np.array(xs, dtype=np.float32), np.array(ys_bin, dtype=np.float32), np.array(ys_ret, dtype=np.float32)


# ==========================================
# 2. Walk-Forward 核心引擎
# ==========================================
def run_rigorous_walk_forward():
    seed_everything(1024)
    print("=" * 75)
    print("Academic Grade Walk-Forward Validation (Miniature LSTM & IC Analysis)")
    print(f"[*] Computing on Device: {DEVICE} | SEQ_LEN: {SEQ_LEN} | HIDDEN: {HIDDEN_SIZE}")
    print("=" * 75)

    csv_path = BASE_DIR / "data" / "processed" / "LSTM_Multimodal_Dataset_2025.csv"
    df = pd.read_csv(str(csv_path))

    df['target_bin'] = (df['log_return'].shift(-1) > 0).astype(int)
    df['target_ret'] = df['log_return'].shift(-1)  # 保留真实的未来连续收益率
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ['date', 'target_bin', 'target_ret', 'log_return']]

    y_true_bin_all, y_true_ret_all = [], []
    probs_lstm_all, preds_lstm_all, preds_lr_all = [], [], []

    for start in range(INITIAL_TRAIN_SIZE, len(df), STEP):
        end = min(start + STEP, len(df))
        if end - start < 1: break

        train_df = df.iloc[:start]
        test_df = df.iloc[start - SEQ_LEN: end]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[feature_cols].values).astype(np.float32)
        X_test_scaled = scaler.transform(test_df[feature_cols].values).astype(np.float32)

        X_tr, y_tr_bin, _ = create_sequences_with_returns(X_train_scaled, train_df['target_bin'].values,
                                                          train_df['target_ret'].values)
        X_te, y_te_bin, y_te_ret = create_sequences_with_returns(X_test_scaled, test_df['target_bin'].values,
                                                                 test_df['target_ret'].values)

        # --- 模型 A: 极简抗拟合 LSTM ---
        model = MiniatureLSTM(X_tr.shape[2]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)  # 增大 L2 正则化惩罚
        crit = nn.BCELoss()

        loader = DataLoader(FinancialDataset(X_tr, y_tr_bin), batch_size=BATCH_SIZE, shuffle=False)
        for _ in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                xb = xb.to(DEVICE, dtype=torch.float32)
                yb = yb.to(DEVICE, dtype=torch.float32)

                output = model(xb).squeeze()
                if output.dim() == 0: output = output.unsqueeze(0)
                if yb.dim() == 0: yb = yb.unsqueeze(0)

                loss = crit(output, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_te_tensor = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
            probs = model(X_te_tensor).squeeze()
            if probs.dim() == 0: probs = probs.unsqueeze(0)

            probs_np = probs.cpu().numpy()
            preds_np = (probs_np > 0.5).astype(int)

        # --- 模型 B: 逻辑回归 (Baseline) ---
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train_scaled, train_df['target_bin'].values)
        preds_lr = lr_model.predict(X_test_scaled[SEQ_LEN:])

        y_true_bin_all.extend(y_te_bin)
        y_true_ret_all.extend(y_te_ret)
        probs_lstm_all.extend(probs_np)
        preds_lstm_all.extend(preds_np)
        preds_lr_all.extend(preds_lr)

        print(f"[*] Trained [0:{start}] -> Test [{start}:{end}] | DA: {accuracy_score(y_te_bin, preds_np) * 100:.1f}%")

    # ==========================================
    # 3. 顶级金融实证学术评估 (Academic Evaluation)
    # ==========================================
    y_true_bin_all = np.array(y_true_bin_all)
    y_true_ret_all = np.array(y_true_ret_all)
    probs_lstm_all = np.array(probs_lstm_all)
    preds_lstm_all = np.array(preds_lstm_all)
    preds_lr_all = np.array(preds_lr_all)

    n_samples = len(y_true_bin_all)
    lstm_correct = np.sum(y_true_bin_all == preds_lstm_all)
    lstm_da = lstm_correct / n_samples
    lr_da = accuracy_score(y_true_bin_all, preds_lr_all)

    # 1. 二项检验 (Binomial Test)
    # 检验假设：我们的胜率是否显著大于随机抛硬币 (50%)
    binom_res = binomtest(k=int(lstm_correct), n=n_samples, p=0.5, alternative='two-sided')
    p_value_binom = binom_res.pvalue

    # 2. 信息系数 (Information Coefficient, IC)
    # 计算预测概率与真实收益率的皮尔逊相关性
    ic_val, p_value_ic = pearsonr(probs_lstm_all, y_true_ret_all)

    print("\n" + "=" * 75)
    print(f"FINAL ACADEMIC EVALUATION (N={n_samples} Out-of-Sample Days)")
    print("=" * 75)
    print(f"[LSTM] Directional Accuracy (DA) : {lstm_da * 100:.2f}%")
    print(f"       ├─ Correct Predictions    : {lstm_correct} / {n_samples}")
    print(f"       └─ Binomial Test p-value  : {p_value_binom:.4f} " + (
        "(Significant! 🌟)" if p_value_binom < 0.05 else "(Not statistically diff from random)"))
    print("-" * 75)
    print(f"[LSTM] Information Coefficient   : {ic_val:.4f}")
    print(f"       └─ Pearson IC p-value     : {p_value_ic:.4f} " + (
        "(Signal exists! 📈)" if p_value_ic < 0.1 else "(No linear correlation)"))
    print("-" * 75)
    print(f"[LR Baseline] Directional Acc    : {lr_da * 100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    run_rigorous_walk_forward()