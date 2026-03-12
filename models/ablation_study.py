import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
# 1. 超参数与系统配置
# ==========================================
SEQ_LEN = 3  # 根据 Granger 检验确定的最优时滞
BATCH_SIZE = 16
CNN_FILTERS = 16  # CNN 提取的特征通道数
HIDDEN_SIZE = 16  # LSTM 隐藏层维度
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


def create_sequences(X, y):
    xs, ys = [], []
    for i in range(len(X) - SEQ_LEN):
        xs.append(X[i:(i + SEQ_LEN)])
        ys.append(y[i + SEQ_LEN])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ==========================================
# 2. 系统模块：定义三种消融网络架构
# ==========================================

# 变体 A: 纯 LSTM (基准深度模型)
class PureLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return torch.sigmoid(self.fc(out))


# 变体 B: CNN-LSTM (引入空间特征提取)
class CNN_LSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # PyTorch Conv1d 要求输入格式为 (Batch, Channels, Length)
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=CNN_FILTERS, kernel_size=2, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(CNN_FILTERS, HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        # x: (B, L, F) -> permute -> (B, F, L)
        x = x.permute(0, 2, 1)
        c_out = self.relu(self.conv1d(x))
        # c_out: (B, C, L_out) -> permute back -> (B, L_out, C)
        c_out = c_out.permute(0, 2, 1)

        l_out, _ = self.lstm(c_out)
        out = self.dropout(l_out[:, -1, :])
        return torch.sigmoid(self.fc(out))


# 变体 C: CNN-LSTM-Attention (完整系统架构)
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=CNN_FILTERS, kernel_size=2, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(CNN_FILTERS, HIDDEN_SIZE, batch_first=True)

        # Global Attention 机制
        self.attention_weights = nn.Linear(HIDDEN_SIZE, 1)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        c_out = self.relu(self.conv1d(x))
        c_out = c_out.permute(0, 2, 1)

        l_out, _ = self.lstm(c_out)  # l_out shape: (B, L, H)

        # 计算 Attention 分数
        attn_scores = self.attention_weights(l_out)  # (B, L, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L, 1)

        # 加权求和上下文向量
        context_vector = torch.sum(attn_weights * l_out, dim=1)  # (B, H)

        out = self.dropout(context_vector)
        return torch.sigmoid(self.fc(out))


# ==========================================
# 3. 统一训练与评估管道
# ==========================================
def train_and_predict(model_class, input_size, X_tr, y_tr, X_te):
    model = model_class(input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    crit = nn.BCELoss()

    loader = DataLoader(FinancialDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=False)

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
        preds = (probs.cpu().numpy() > 0.5).astype(int)
    return preds


def run_ablation_study():
    seed_everything(1024)
    print("=" * 70)
    print("System Evaluation: CNN-LSTM-Attention Ablation Study (Walk-Forward)")
    print("=" * 70)

    csv_path = BASE_DIR / "data" / "processed" / "LSTM_Multimodal_Dataset_2025.csv"
    df = pd.read_csv(str(csv_path))
    df['target'] = (df['log_return'].shift(-1) > 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ['date', 'target', 'log_return']]

    y_true_all = []
    preds_lstm, preds_cnnlstm, preds_attn = [], [], []

    for start in range(INITIAL_TRAIN_SIZE, len(df), STEP):
        end = min(start + STEP, len(df))
        if end - start < 1: break
        print(f"[*] Processing Walk-Forward Fold: [{start} -> {end}]...")

        train_df = df.iloc[:start]
        test_df = df.iloc[start - SEQ_LEN: end]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(train_df[feature_cols].values).astype(np.float32)
        X_te_scaled = scaler.transform(test_df[feature_cols].values).astype(np.float32)

        X_tr, y_tr = create_sequences(X_tr_scaled, train_df['target'].values)
        X_te, y_te = create_sequences(X_te_scaled, test_df['target'].values)

        input_dim = X_tr.shape[2]

        # 依次训练三个模型
        p_lstm = train_and_predict(PureLSTM, input_dim, X_tr, y_tr, X_te)
        p_cnnlstm = train_and_predict(CNN_LSTM, input_dim, X_tr, y_tr, X_te)
        p_attn = train_and_predict(CNN_LSTM_Attention, input_dim, X_tr, y_tr, X_te)

        y_true_all.extend(y_te)
        preds_lstm.extend(p_lstm)
        preds_cnnlstm.extend(p_cnnlstm)
        preds_attn.extend(p_attn)

    # 结果核算
    print("\n" + "=" * 70)
    print(f"ABLATION STUDY RESULTS (Out-of-Sample Days N={len(y_true_all)})")
    print("=" * 70)
    print(f"[Model 1] Pure LSTM           DA: {accuracy_score(y_true_all, preds_lstm) * 100:.2f}%")
    print(f"[Model 2] CNN-LSTM            DA: {accuracy_score(y_true_all, preds_cnnlstm) * 100:.2f}%")
    print(f"[Model 3] CNN-LSTM-Attention  DA: {accuracy_score(y_true_all, preds_attn) * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_ablation_study()