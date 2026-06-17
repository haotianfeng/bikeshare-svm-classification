import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from src.config import FIGURES_DIR, TABLES_DIR, RANDOM_SEED, N_JOBS
from src.visualize import setup_chinese_font, set_style, save_figure


def mutual_information_ranking(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Compute Mutual Information scores and return ranked DataFrame."""
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    df = pd.DataFrame(
        {"feature": feature_names, "mutual_information": mi_scores}
    ).sort_values("mutual_information", ascending=False)
    df["rank_mi"] = range(1, len(df) + 1)
    df.to_csv(f"{TABLES_DIR}/feature_ranking_mi.csv", index=False)
    return df


def chi_square_ranking(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Compute Chi-Square scores (after MinMax scaling to [0,1]) and return ranked DF."""
    scaler = MinMaxScaler()
    X_01 = scaler.fit_transform(X)
    chi2_scores, p_values = chi2(X_01, y)
    df = pd.DataFrame(
        {"feature": feature_names, "chi2_score": chi2_scores, "chi2_pvalue": p_values}
    ).sort_values("chi2_score", ascending=False)
    df["rank_chi2"] = range(1, len(df) + 1)
    df.to_csv(f"{TABLES_DIR}/feature_ranking_chi2.csv", index=False)
    return df


def random_forest_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Train a Random Forest (auxiliary only) to get Gini importance."""
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=RANDOM_SEED, n_jobs=N_JOBS
    )
    rf.fit(X, y)
    df = pd.DataFrame(
        {"feature": feature_names, "rf_importance": rf.feature_importances_}
    ).sort_values("rf_importance", ascending=False)
    df["rank_rf"] = range(1, len(df) + 1)
    df.to_csv(f"{TABLES_DIR}/feature_ranking_rf.csv", index=False)
    return df


def ensemble_feature_ranking(
    rankings: list[pd.DataFrame],
) -> pd.DataFrame:
    """Average normalized ranks from multiple methods into consensus ranking."""
    # Each input df has 'feature' and 'rank_*' column
    merged = None
    rank_cols = []
    for i, r in enumerate(rankings):
        rank_col = [c for c in r.columns if c.startswith("rank_")][0]
        rank_cols.append(rank_col)
        if merged is None:
            merged = r[["feature", rank_col]]
        else:
            merged = merged.merge(r[["feature", rank_col]], on="feature")

    merged["mean_rank"] = merged[rank_cols].mean(axis=1)
    merged = merged.sort_values("mean_rank")
    merged["final_rank"] = range(1, len(merged) + 1)
    merged.to_csv(f"{TABLES_DIR}/feature_ranking_ensemble.csv", index=False)
    return merged


def plot_feature_rankings(
    mi_df: pd.DataFrame,
    chi2_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    output_dir: str = FIGURES_DIR,
) -> list[str]:
    """Plot 3 individual ranking charts + 1 ensemble chart."""
    set_style()
    setup_chinese_font()
    paths = []

    # MI chart
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df = mi_df.head(15).iloc[::-1]
    ax.barh(range(len(plot_df)), plot_df["mutual_information"].values,
            color=sns.color_palette("Blues_d", len(plot_df)))
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=9)
    ax.set_title("互信息特征重要性排序", fontsize=16, fontweight="bold")
    ax.set_xlabel("互信息得分", fontsize=12)
    paths.append(save_figure(fig, "feature_ranking_mi.png", output_dir))

    # Chi2 chart
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df = chi2_df.head(15).iloc[::-1]
    ax.barh(range(len(plot_df)), plot_df["chi2_score"].values,
            color=sns.color_palette("Greens_d", len(plot_df)))
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=9)
    ax.set_title("卡方检验特征重要性排序", fontsize=16, fontweight="bold")
    ax.set_xlabel("卡方值", fontsize=12)
    paths.append(save_figure(fig, "feature_ranking_chi2.png", output_dir))

    # RF chart
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df = rf_df.head(15).iloc[::-1]
    ax.barh(range(len(plot_df)), plot_df["rf_importance"].values,
            color=sns.color_palette("Oranges_d", len(plot_df)))
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=9)
    ax.set_title("随机森林特征重要性排序", fontsize=16, fontweight="bold")
    ax.set_xlabel("Gini Importance", fontsize=12)
    paths.append(save_figure(fig, "feature_ranking_rf.png", output_dir))

    # Ensemble chart
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df = ensemble_df.head(17).iloc[::-1]
    bars = ax.barh(
        range(len(plot_df)),
        plot_df["mean_rank"].values,
        color=sns.color_palette("Reds_d", len(plot_df)),
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=9)
    ax.set_title("综合特征重要性排序（三种方法平均排名）", fontsize=16, fontweight="bold")
    ax.set_xlabel("平均排名 (越小越重要)", fontsize=12)
    ax.invert_xaxis()
    paths.append(save_figure(fig, "feature_ranking_ensemble.png", output_dir))

    return paths


import seaborn as sns


def run_feature_selection(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict:
    """Run all three feature selection methods + ensemble."""
    mi_df = mutual_information_ranking(X, y, feature_names)
    chi2_df = chi_square_ranking(X, y, feature_names)
    rf_df = random_forest_importance(X, y, feature_names)
    ensemble_df = ensemble_feature_ranking([mi_df, chi2_df, rf_df])
    plot_paths = plot_feature_rankings(
        mi_df, chi2_df, rf_df, ensemble_df, FIGURES_DIR
    )
    return {
        "mi": mi_df,
        "chi2": chi2_df,
        "rf": rf_df,
        "ensemble": ensemble_df,
        "plot_paths": plot_paths,
    }
