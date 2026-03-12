import random
import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pathlib import Path

# 1. 随机种子置于最顶层
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# 2. 防过拟合超参数
# ==========================================
SEQ_LEN = 10
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
EPOCHS = 30  # 【修改】从100降至30，杜绝严重过拟合
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class FinancialDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def build_timeseries_dataset():
    csv_path = BASE_DIR / "data" / "processed" / "LSTM_Multimodal_Dataset_2025.csv"
    df = pd.read_csv(str(csv_path))

    df['target_next_day'] = (df['log_return'].shift(-1) > 0).astype(int)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ['date', 'target_next_day', 'log_return']]
    X_raw, y_raw = df[feature_cols].values, df['target_next_day'].values

    train_size = int(len(df) * 0.8)
    X_train_raw, y_train_raw = X_raw[:train_size], y_raw[:train_size]
    X_test_raw, y_test_raw = X_raw[train_size:], y_raw[train_size:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    def create_sequences(X, y, seq_length):
        xs, ys = [], []
        for i in range(len(X) - seq_length):
            xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(xs), np.array(ys)

    X_train, y_train = create_sequences(X_train_scaled, y_train_raw, SEQ_LEN)
    X_test, y_test = create_sequences(X_test_scaled, y_test_raw, SEQ_LEN)
    return X_train, y_train, X_test, y_test


class MultimodalLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


def train_and_evaluate(seed):
    seed_everything(seed)
    X_tr, y_tr, X_te, y_te = build_timeseries_dataset()

    # 【修改】shuffle=False 严格保留批次间的宏观时序特征
    train_loader = DataLoader(FinancialDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(FinancialDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

    model = MultimodalLSTM(input_size=X_tr.shape[2]).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch).squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    y_true, y_pred_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = (model(X_batch.to(DEVICE)).squeeze() > 0.5).float()
            if preds.dim() == 0: preds = preds.unsqueeze(0)
            if y_batch.dim() == 0: y_batch = y_batch.unsqueeze(0)
            y_true.extend(y_batch.cpu().numpy())
            y_pred_labels.extend(preds.cpu().numpy())

    return accuracy_score(y_true, y_pred_labels) * 100


if __name__ == "__main__":
    print("=" * 60)
    print("Rigorous Multi-Seed Validation (N=10 runs)")
    print("=" * 60)

    seeds = [0, 42, 123, 345, 666, 777, 888, 999, 1024, 2026]
    results = []

    for s in seeds:
        acc = train_and_evaluate(s)
        results.append(acc)
        print(f"Seed {s:<5} | Test Accuracy: {acc:.2f}%")

    mean_acc = np.mean(results)
    std_acc = np.std(results)

    print("-" * 60)
    print(f"FINAL ROBUST ACCURACY: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print("-" * 60)