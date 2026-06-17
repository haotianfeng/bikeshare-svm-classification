#!/usr/bin/env python3
"""SVM 共享单车用户分类 — 完整实验流程

按照 CLAUDE.md 定义的 12 个步骤依次执行。
运行: python main.py
"""

import sys
import os
import gc
import time
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免 Windows 下 tkinter 线程报错
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    RAW_DATA_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, ensure_dirs,
    TOTAL_SAMPLE_SIZE, RANDOM_SEED, PARAM_GRID_EXTENDED, N_JOBS, SVC_CACHE_SIZE,
)
from src.data_loader import discover_csv_files, load_single_month, compute_monthly_stats
from src.sampling import stratified_sample_by_month, save_sampling_summary
from src.cleaning import clean_dataset, report_cleaning_stats
from src.feature_engineering import engineer_all_features
from src.eda import run_all_eda
from src.feature_selection import run_feature_selection
from src.preprocessing import preprocess_and_split, save_scaler
from src.train import (
    train_baseline_models, run_grid_search, run_extended_grid_search,
    train_final_model, plot_model_comparison, save_model,
)
from src.evaluate import run_full_evaluation
from src.feature_analysis import run_feature_analysis
from src.findings import run_all_findings
from src.visualize import setup_chinese_font, set_style


def _mem_mb() -> str:
    """返回当前进程的内存占用量，格式为人类可读的字符串。"""
    try:
        import psutil
        p = psutil.Process(os.getpid())
        mem = p.memory_info().rss / 1024 / 1024  # MB
        return f"{mem:.0f} MB"
    except ImportError:
        return "N/A (install psutil for memory monitoring)"


def main():
    t0_total = time.time()
    ensure_dirs()
    print("=" * 60)
    print("SVM 共享单车用户分类 — 完整实验流程")
    print(f"配置: 样本={TOTAL_SAMPLE_SIZE}, 并行={N_JOBS}核, SVC缓存={SVC_CACHE_SIZE}MB")
    print(f"当前内存: {_mem_mb()}")
    print("=" * 60)

    # ── Step 1: Data Sampling ──────────────────────
    print("\n[Step 1/12] 数据采样 (每月分层抽样)...")
    t0 = time.time()
    df_raw = stratified_sample_by_month(RAW_DATA_DIR)
    print(f"  抽样完成: {len(df_raw)} 条记录 ({time.time()-t0:.1f}s)")
    print(f"  当前内存: {_mem_mb()}")
    save_sampling_summary(df_raw)
    gc.collect()

    # ── Step 2: Data Cleaning ──────────────────────
    print("\n[Step 2/12] 数据清洗...")
    t0 = time.time()
    df_clean = clean_dataset(df_raw)
    report_cleaning_stats(df_raw, df_clean)
    print(f"  清洗完成 ({time.time()-t0:.1f}s)")
    print(f"  当前内存: {_mem_mb()}")
    gc.collect()

    # ── Step 3: Feature Engineering ────────────────
    print("\n[Step 3/12] 特征工程...")
    t0 = time.time()
    X, y, feature_names = engineer_all_features(df_clean)
    print(f"  生成 {len(feature_names)} 个特征: {feature_names}")
    print(f"  特征矩阵形状: {X.shape}")
    print(f"  类别分布: member={y.sum()}, casual={len(y)-y.sum()}")
    print(f"  特征工程完成 ({time.time()-t0:.1f}s)")
    print(f"  当前内存: {_mem_mb()}")
    gc.collect()

    # Add engineered features back to df for EDA
    for i, name in enumerate(feature_names):
        df_clean[name] = X[:, i]
    df_clean["target"] = y

    # ── Step 4: Exploratory Data Analysis ──────────
    print("\n[Step 4/12] 探索性数据分析 (EDA)...")
    t0 = time.time()
    eda_paths = run_all_eda(df_clean, feature_names, FIGURES_DIR)
    print(f"  EDA 完成: {len(eda_paths)} 张图 ({time.time()-t0:.1f}s)")

    # ── Step 5: Feature Selection ──────────────────
    print("\n[Step 5/12] 特征选择...")
    t0 = time.time()
    selection_results = run_feature_selection(X, y, feature_names)
    print(f"  Top 5 (MI): {list(selection_results['mi'].head(5)['feature'])}")
    print(f"  Top 5 (Chi2): {list(selection_results['chi2'].head(5)['feature'])}")
    print(f"  Top 5 (RF): {list(selection_results['rf'].head(5)['feature'])}")
    print(f"  Top 5 (Ensemble): {list(selection_results['ensemble'].head(5)['feature'])}")
    print(f"  特征选择完成 ({time.time()-t0:.1f}s)")

    # ── Step 6 & 7: Split + Scale ──────────────────
    print("\n[Step 6-7/12] 训练/测试划分 + 标准化...")
    t0 = time.time()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(X, y)
    save_scaler(scaler)
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"  标准化完成 ({time.time()-t0:.1f}s)")
    print(f"  当前内存: {_mem_mb()}")
    gc.collect()

    # ── Step 8-9: SVM Modeling ─────────────────────
    print("\n[Step 8/12] SVM 建模...")
    t0 = time.time()
    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)
    print(f"  基准模型完成 ({time.time()-t0:.1f}s)")

    print("\n[Step 9/12] 超参数优化 (GridSearchCV)...")
    t0 = time.time()
    grid_search = run_grid_search(X_train, y_train)

    # Extended search if best C is at boundary (100)
    best_c = grid_search.best_params_["C"]
    if best_c == PARAM_GRID_EXTENDED["C"][0]:  # 100
        print("\n  最佳 C 在搜索边界，扩展搜索...")
        grid_ext = run_extended_grid_search(X_train, y_train)
        best_params = grid_ext.best_params_
        cv_best_score = grid_ext.best_score_
    else:
        best_params = grid_search.best_params_
        cv_best_score = grid_search.best_score_

    print(f"\n  最终超参数: {best_params}")
    print(f"  交叉验证最优得分: {cv_best_score:.4f}")
    print(f"  GridSearch 完成 ({time.time()-t0:.1f}s)")

    # Train final model
    final_model = train_final_model(X_train, y_train, best_params)
    model_path, names_path = save_model(final_model, feature_names)
    print(f"  最终模型已保存: {model_path}")

    # Model comparison plot
    y_pred_final = final_model.predict(X_test)
    from sklearn.metrics import accuracy_score
    test_acc = accuracy_score(y_test, y_pred_final)
    plot_model_comparison(baseline_results, cv_best_score, test_acc)

    # ── Step 10: Evaluation ────────────────────────
    print("\n[Step 10/12] 模型评估...")
    t0 = time.time()
    eval_results = run_full_evaluation(final_model, X_test, y_test, FIGURES_DIR)
    print(f"  评估完成 ({time.time()-t0:.1f}s)")

    # ── Step 11: Feature Importance ────────────────
    print("\n[Step 11/12] 特征重要性分析 (Permutation Importance)...")
    t0 = time.time()
    fa_results = run_feature_analysis(
        final_model, X_test, y_test, X_train, y_train, feature_names
    )
    top5_perm = list(fa_results["permutation_importance"].head(5)["feature"])
    print(f"  Top 5 (Permutation): {top5_perm}")
    print(f"  特征重要性分析完成 ({time.time()-t0:.1f}s)")

    # ── Step 12: Interesting Findings ──────────────
    print("\n[Step 12/12] 有趣发现分析...")
    t0 = time.time()
    findings = run_all_findings(df_clean)
    print(f"  发现分析完成 ({time.time()-t0:.1f}s)")

    # ── Final Summary ──────────────────────────────
    total_time = time.time() - t0_total
    print("\n" + "=" * 60)
    print("实验完成！")
    print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n最终结果:")
    metrics = eval_results["metrics"]
    print(f"  准确率 (Accuracy):      {metrics['accuracy']:.4f}")
    print(f"  F1 (加权):               {metrics['f1_weighted']:.4f}")
    print(f"  准确率 (会员):            {metrics['precision_member']:.4f}")
    print(f"  召回率 (会员):            {metrics['recall_member']:.4f}")
    print(f"  准确率 (散客):            {metrics['precision_casual']:.4f}")
    print(f"  召回率 (散客):            {metrics['recall_casual']:.4f}")
    roc_auc = metrics.get("roc_auc", None)
    if isinstance(roc_auc, float):
        print(f"  ROC-AUC:                {roc_auc:.4f}")
    else:
        print(f"  ROC-AUC:                N/A")
    print(f"\n输出文件:")
    print(f"  图片: {FIGURES_DIR}")
    print(f"  表格: {TABLES_DIR}")
    print(f"  模型: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
