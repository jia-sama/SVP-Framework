import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def run_ml_baselines():
    csv_path = BASE_DIR / "data" / "processed" / "LSTM_Multimodal_Dataset_2025.csv"
    df = pd.read_csv(str(csv_path))

    df['target_next_day'] = (df['log_return'].shift(-1) > 0).astype(int)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ['date', 'target_next_day', 'log_return']]
    X_raw, y_raw = df[feature_cols].values, df['target_next_day'].values

    train_size = int(len(df) * 0.8)
    X_tr_raw, y_tr = X_raw[:train_size], y_raw[:train_size]
    X_te_raw, y_te = X_raw[train_size:], y_raw[train_size:]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_te_scaled = scaler.transform(X_te_raw)

    # 传统机器学习不需要时序张量，直接用滞后 1 天的数据特征预测下一天
    # (这里为了对比简单，直接使用 flattened 或单步特征)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVC (RBF)": SVC(kernel='rbf', random_state=42)
    }

    print("=" * 50)
    print("Traditional Machine Learning Baselines (No Sliding Window)")
    print("=" * 50)

    for name, model in models.items():
        model.fit(X_tr_scaled, y_tr)
        preds = model.predict(X_te_scaled)
        acc = accuracy_score(y_te, preds) * 100
        print(f"[*] {name:<20} : {acc:.2f}%")


if __name__ == "__main__":
    run_ml_baselines()