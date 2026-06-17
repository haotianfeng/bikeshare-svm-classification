import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TEST_SIZE, RANDOM_SEED, MODELS_DIR


def preprocess_and_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """分层划分训练/测试集，在训练集上拟合 StandardScaler，再分别变换两个集合。

    返回 (X_train_scaled, X_test_scaled, y_train, y_test, scaler)。
    关键原则：scaler 仅 fit 训练集，测试集只 transform，防止数据泄露。
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_scaler(scaler: StandardScaler, filepath: str | None = None) -> str:
    """保存 StandardScaler 到 joblib 文件。"""
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, filepath)
    return filepath
