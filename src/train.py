import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import (
    PARAM_GRID,
    PARAM_GRID_EXTENDED,
    CV_FOLDS,
    SCORING,
    RANDOM_SEED,
    FIGURES_DIR,
    TABLES_DIR,
    MODELS_DIR,
    N_JOBS,
    SVC_CACHE_SIZE,
)
from src.visualize import setup_chinese_font, set_style, save_figure


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """训练 3 个基准 SVM 模型（LinearSVC / Poly / RBF）并对比性能。"""
    models = {
        "LinearSVC": LinearSVC(
            C=1.0, max_iter=5000, dual=False, class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "SVC-Poly(d=2)": SVC(
            kernel="poly", degree=2, C=1.0, class_weight="balanced",
            random_state=RANDOM_SEED, cache_size=SVC_CACHE_SIZE,
        ),
        "SVC-RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            random_state=RANDOM_SEED, cache_size=SVC_CACHE_SIZE,
        ),
    }

    results = {"model": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "time_sec": []}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        y_pred = model.predict(X_test)
        results["model"].append(name)
        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["precision"].append(precision_score(y_test, y_pred, average="weighted"))
        results["recall"].append(recall_score(y_test, y_pred, average="weighted"))
        results["f1"].append(f1_score(y_test, y_pred, average="weighted"))
        results["time_sec"].append(round(elapsed, 1))
        print(f"  {name}: Acc={results['accuracy'][-1]:.4f}, F1={results['f1'][-1]:.4f}, Time={elapsed:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv(f"{TABLES_DIR}/baseline_model_comparison.csv", index=False)
    return results


def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> GridSearchCV:
    """使用 5 折交叉验证对 RBF 核 SVC 执行网格搜索。"""
    svc = SVC(
        class_weight="balanced", random_state=RANDOM_SEED, cache_size=SVC_CACHE_SIZE
    )
    print(f"  Searching {len(PARAM_GRID['C']) * len(PARAM_GRID['gamma'])} param combinations with {CV_FOLDS}-fold CV...")
    print(f"  并行任务数: {N_JOBS} (限制以控制内存)")
    t0 = time.time()

    grid = GridSearchCV(
        svc, PARAM_GRID, cv=CV_FOLDS, scoring=SCORING,
        n_jobs=N_JOBS, verbose=1, return_train_score=True,
        pre_dispatch="2*n_jobs",  # 限制预分配任务数，减少内存占用
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV score: {grid.best_score_:.4f}")
    print(f"  GridSearch completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # 保存交叉验证结果
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df.to_csv(f"{TABLES_DIR}/grid_search_results.csv", index=False)
    return grid


def run_extended_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> GridSearchCV:
    """当最优 C 值位于搜索边界时执行扩展网格搜索。"""
    svc = SVC(
        class_weight="balanced", random_state=RANDOM_SEED, cache_size=SVC_CACHE_SIZE
    )
    print(f"  Extended search: C={PARAM_GRID_EXTENDED['C']}")
    grid = GridSearchCV(
        svc, PARAM_GRID_EXTENDED, cv=CV_FOLDS, scoring=SCORING,
        n_jobs=N_JOBS, verbose=1, return_train_score=True,
        pre_dispatch="2*n_jobs",
    )
    grid.fit(X_train, y_train)
    print(f"  Extended best params: {grid.best_params_}")
    print(f"  Extended best CV score: {grid.best_score_:.4f}")
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df.to_csv(f"{TABLES_DIR}/grid_search_extended_results.csv", index=False)
    return grid


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
) -> CalibratedClassifierCV:
    """使用最优超参数训练最终 SVC 模型，外层包裹 CalibratedClassifierCV
    以输出校准概率（内部使用 5 折 CV 做 Platt Scaling）。"""
    svc = SVC(
        kernel="rbf",
        C=best_params["C"],
        gamma=best_params["gamma"],
        class_weight="balanced",
        random_state=RANDOM_SEED,
        cache_size=SVC_CACHE_SIZE,
    )
    model = CalibratedClassifierCV(svc, cv=5, ensemble=False)
    model.fit(X_train, y_train)
    return model


def plot_model_comparison(
    baseline_results: dict,
    cv_best_score: float,
    test_accuracy: float,
    output_dir: str = FIGURES_DIR,
) -> str:
    """绘制所有模型的准确率对比柱状图。"""
    set_style()
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(baseline_results["model"]) + ["SVC-RBF (优化)"]
    accs = list(baseline_results["accuracy"]) + [test_accuracy]
    colors = ["#95A5A6", "#95A5A6", "#95A5A6", "#A23B72"]
    bars = ax.bar(models, [a * 100 for a in accs], color=colors, width=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc*100:.2f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_title("模型准确率对比", fontsize=18, fontweight="bold")
    ax.set_ylabel("准确率 (%)", fontsize=13)
    ax.set_ylim(0, max(accs) * 100 * 1.15)
    ax.tick_params(axis="x", rotation=15)
    return save_figure(fig, "model_comparison.png", output_dir)


def save_model(
    model: SVC,
    feature_names: list[str],
    output_dir: str = MODELS_DIR,
) -> tuple[str, str]:
    """保存训练好的模型和特征名列表到 joblib 文件。"""
    model_path = os.path.join(output_dir, "best_svm_model.joblib")
    names_path = os.path.join(output_dir, "feature_names.joblib")
    joblib.dump(model, model_path)
    joblib.dump(feature_names, names_path)
    return model_path, names_path
