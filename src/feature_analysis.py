import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from src.config import FIGURES_DIR, TABLES_DIR, RANDOM_SEED, N_JOBS
from src.visualize import setup_chinese_font, set_style, save_figure


def compute_permutation_importance(
    model, X: np.ndarray, y: np.ndarray, feature_names: list[str],
    n_repeats: int = 20,
) -> pd.DataFrame:
    """多次重复计算排列重要性以获得稳定的重要性估计。"""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=RANDOM_SEED,
        n_jobs=N_JOBS, scoring="accuracy",
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df.to_csv(f"{TABLES_DIR}/permutation_importance.csv", index=False)
    return df


def logistic_regression_coefficients(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str],
) -> pd.DataFrame:
    """训练逻辑回归并提取标准化特征系数。"""
    lr = LogisticRegression(
        C=1.0, class_weight="balanced",
        max_iter=5000, random_state=RANDOM_SEED,
    )
    lr.fit(X_train, y_train)
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": lr.coef_[0],
        "abs_coefficient": np.abs(lr.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df.to_csv(f"{TABLES_DIR}/logistic_regression_coeffs.csv", index=False)
    return df


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    output_dir: str = FIGURES_DIR,
) -> str:
    """绘制排列重要性柱状图（含误差线），展示 Top 15 特征。"""
    set_style()
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df = importance_df.head(15).iloc[::-1]

    bars = ax.barh(
        range(len(plot_df)),
        plot_df["importance_mean"].values,
        xerr=plot_df["importance_std"].values,
        color=sns.color_palette("Reds_d", len(plot_df)),
        capsize=3, error_kw={"linewidth": 1.5},
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=9)
    ax.set_title("排列重要性（Permutation Importance）", fontsize=16, fontweight="bold")
    ax.set_xlabel("准确率下降 (mean ± std)", fontsize=12)
    return save_figure(fig, "feature_permutation_importance.png", output_dir)


def plot_logistic_coefficients(
    coef_df: pd.DataFrame,
    output_dir: str = FIGURES_DIR,
) -> str:
    """绘制逻辑回归系数大小和方向（正=会员倾向，负=散客倾向）。"""
    set_style()
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df = coef_df.head(15).iloc[::-1]
    colors = ["#A23B72" if c > 0 else "#2E86AB" for c in plot_df["coefficient"].values]
    ax.barh(range(len(plot_df)), plot_df["coefficient"].values, color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=9)
    ax.set_title("逻辑回归系数（标准化特征）", fontsize=16, fontweight="bold")
    ax.set_xlabel("系数值", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.8)
    return save_figure(fig, "feature_lr_coefficients.png", output_dir)


import seaborn as sns


def run_feature_analysis(
    model, X_test: np.ndarray, y_test: np.ndarray,
    X_train: np.ndarray, y_train: np.ndarray,
    feature_names: list[str],
) -> dict:
    """执行排列重要性分析和逻辑回归系数对比。"""
    perm_df = compute_permutation_importance(model, X_test, y_test, feature_names)
    lr_df = logistic_regression_coefficients(X_train, y_train, feature_names)

    paths = []
    paths.append(plot_permutation_importance(perm_df))
    paths.append(plot_logistic_coefficients(lr_df))

    return {
        "permutation_importance": perm_df,
        "logistic_regression": lr_df,
        "plot_paths": paths,
    }
