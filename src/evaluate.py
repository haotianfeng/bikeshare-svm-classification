import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc,
)
from sklearn.model_selection import learning_curve
from src.config import FIGURES_DIR, TABLES_DIR, CV_FOLDS, RANDOM_SEED, N_JOBS
from src.visualize import setup_chinese_font, set_style, save_figure


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """Compute comprehensive evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_casual": precision_score(y_true, y_pred, pos_label=0),
        "recall_casual": recall_score(y_true, y_pred, pos_label=0),
        "f1_casual": f1_score(y_true, y_pred, pos_label=0),
        "precision_member": precision_score(y_true, y_pred, pos_label=1),
        "recall_member": recall_score(y_true, y_pred, pos_label=1),
        "f1_member": f1_score(y_true, y_pred, pos_label=1),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_dir: str = FIGURES_DIR
) -> str:
    """Plot normalized + raw confusion matrix."""
    set_style()
    setup_chinese_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    for ax, data, title, fmt in [
        (ax1, cm, "混淆矩阵（原始计数）", "d"),
        (ax2, cm_norm, "混淆矩阵（归一化）", ".2%"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
            xticklabels=["散客", "会员"], yticklabels=["散客", "会员"],
            cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 16},
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("预测标签", fontsize=13)
        ax.set_ylabel("真实标签", fontsize=13)

    return save_figure(fig, "eval_confusion_matrix.png", output_dir)


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, output_dir: str = FIGURES_DIR
) -> str:
    """Plot ROC curve with AUC."""
    set_style()
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=3, color="#A23B72",
            label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="随机分类器")
    ax.fill_between(fpr, tpr, alpha=0.2, color="#A23B72")
    ax.set_title("ROC 曲线", fontsize=18, fontweight="bold")
    ax.set_xlabel("假阳性率 (FPR)", fontsize=13)
    ax.set_ylabel("真阳性率 (TPR)", fontsize=13)
    ax.legend(fontsize=13, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return save_figure(fig, "eval_roc_curve.png", output_dir)


def plot_precision_recall_curve(
    y_true: np.ndarray, y_proba: np.ndarray, output_dir: str = FIGURES_DIR
) -> str:
    """Plot Precision-Recall curve."""
    set_style()
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 8))
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, linewidth=3, color="#2E86AB",
            label=f"PR (AUC = {pr_auc:.3f})")
    ax.fill_between(recall, precision, alpha=0.2, color="#2E86AB")
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"基准 ({baseline:.3f})")
    ax.set_title("精确率-召回率曲线", fontsize=18, fontweight="bold")
    ax.set_xlabel("召回率", fontsize=13)
    ax.set_ylabel("精确率", fontsize=13)
    ax.legend(fontsize=13)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return save_figure(fig, "eval_pr_curve.png", output_dir)


def plot_classification_report_heatmap(
    y_true: np.ndarray, y_pred: np.ndarray, output_dir: str = FIGURES_DIR
) -> str:
    """Visualize classification report as a heatmap."""
    set_style()
    setup_chinese_font()
    report = classification_report(
        y_true, y_pred,
        target_names=["散客", "会员"],
        output_dict=True,
    )

    rows = []
    for cls in ["散客", "会员", "macro avg", "weighted avg"]:
        d = report[cls]
        rows.append({
            "类别": cls,
            "精确率": d["precision"],
            "召回率": d["recall"],
            "F1": d["f1-score"],
            "样本数": report[cls].get("support", 0),
        })
    table_df = pd.DataFrame(rows)
    table_df.to_csv(f"{TABLES_DIR}/classification_report.csv", index=False)

    # Heatmap of the three metrics
    plot_df = table_df.set_index("类别")[["精确率", "召回率", "F1"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        plot_df * 100, annot=True, fmt=".1f", cmap="YlOrRd",
        ax=ax, cbar_kws={"shrink": 0.8, "label": "%"},
        annot_kws={"fontsize": 14}, vmin=50, vmax=100,
    )
    ax.set_title("分类报告总览", fontsize=18, fontweight="bold")
    return save_figure(fig, "eval_classification_report.png", output_dir)


def plot_learning_curve(
    model, X: np.ndarray, y: np.ndarray, output_dir: str = FIGURES_DIR
) -> str:
    """Plot learning curve: train vs CV score vs training size."""
    set_style()
    setup_chinese_font()
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, cv=CV_FOLDS, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=RANDOM_SEED, n_jobs=N_JOBS,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="#A23B72")
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color="#2E86AB")
    ax.plot(train_sizes_abs, train_mean, "o-", linewidth=2.5, color="#A23B72",
            label="训练集")
    ax.plot(train_sizes_abs, val_mean, "s-", linewidth=2.5, color="#2E86AB",
            label="交叉验证")
    ax.set_title("学习曲线", fontsize=18, fontweight="bold")
    ax.set_xlabel("训练样本数", fontsize=13)
    ax.set_ylabel("准确率", fontsize=13)
    ax.legend(fontsize=13)
    return save_figure(fig, "eval_learning_curve.png", output_dir)


def save_metrics_table(metrics: dict, output_dir: str = TABLES_DIR) -> str:
    """Save all metrics to CSV."""
    df = pd.DataFrame([metrics])
    filepath = f"{output_dir}/final_metrics.csv"
    df.to_csv(filepath, index=False)
    return filepath


def run_full_evaluation(
    model, X_test: np.ndarray, y_test: np.ndarray,
    output_dir: str = FIGURES_DIR,
) -> dict:
    """Run comprehensive evaluation: metrics, confusion matrix, ROC, PR, learning curve."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)

    paths = []
    paths.append(plot_confusion_matrix(y_test, y_pred, output_dir))
    paths.append(plot_roc_curve(y_test, y_proba, output_dir))
    paths.append(plot_precision_recall_curve(y_test, y_proba, output_dir))
    paths.append(plot_classification_report_heatmap(y_test, y_pred, output_dir))
    paths.append(plot_learning_curve(model, X_test, y_test, output_dir))
    paths.append(save_metrics_table(metrics))

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return {"metrics": metrics, "plot_paths": paths}
