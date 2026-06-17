import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.config import FIGURES_DIR
from src.visualize import setup_chinese_font, set_style, save_figure

SEASONS = {
    1: "冬季", 2: "冬季", 3: "春季", 4: "春季", 5: "春季",
    6: "夏季", 7: "夏季", 8: "夏季", 9: "秋季", 10: "秋季",
    11: "秋季", 12: "冬季",
}
DOW_LABELS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
SEASON_ORDER = ["春季", "夏季", "秋季", "冬季"]


def _fig() -> tuple[plt.Figure, plt.Axes]:
    """创建标准尺寸的 figure，应用 seaborn 样式和中文字体。

    注意：必须先调 set_style()（seaborn 设置样式），再调
    setup_chinese_font()（覆盖 seaborn 重置的字体配置）。
    """
    set_style()
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))
    return fig, ax


# ── 图1：用户类型分布 ──────────────────────────────
def plot_user_distribution(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    counts = df["member_casual"].value_counts()
    colors = ["#2E86AB", "#A23B72"]
    ax.pie(
        counts.values,
        labels=["会员", "散客"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0, 0.05),
        textprops={"fontsize": 14},
    )
    ax.set_title("用户类型分布", fontsize=18, fontweight="bold")
    return save_figure(fig, "01_user_distribution.png", save_path)


# ── 图2：月度骑行量分布 ────────────────────────────
def plot_monthly_volume(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    monthly = (
        df.groupby(["month", "member_casual"]).size().unstack(fill_value=0)
    )
    monthly = monthly.loc[sorted(monthly.index)]
    monthly.index = [m[-2:] + "月" for m in monthly.index]  # "202501" -> "01月"
    monthly.columns = ["散客", "会员"]
    colors = ["#2E86AB", "#A23B72"]
    monthly.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8)
    ax.set_title("月度骑行量分布", fontsize=18, fontweight="bold")
    ax.set_xlabel("月份", fontsize=13)
    ax.set_ylabel("骑行次数", fontsize=13)
    ax.legend(fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    return save_figure(fig, "02_monthly_volume.png", save_path)


# ── 图3：各时段使用模式 ────────────────────────────
def plot_hourly_pattern(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    df_copy = df.copy()
    df_copy["hour"] = df_copy["started_at"].dt.hour
    hourly = (
        df_copy.groupby(["hour", "member_casual"]).size().unstack(fill_value=0)
    )
    # 归一化为占比
    hourly_prop = hourly.div(hourly.sum(axis=1), axis=0)
    hourly_prop.columns = ["散客", "会员"]
    hourly_prop.plot(ax=ax, linewidth=2.5, marker="o", markersize=6)
    ax.set_title("各时段用户类型占比", fontsize=18, fontweight="bold")
    ax.set_xlabel("小时", fontsize=13)
    ax.set_ylabel("占比", fontsize=13)
    ax.legend(fontsize=12)
    ax.set_xticks(range(0, 24, 2))
    return save_figure(fig, "03_hourly_pattern.png", save_path)


# ── 图4：每日骑行分布 ──────────────────────────────
def plot_daily_pattern(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    df_copy = df.copy()
    df_copy["dow"] = df_copy["started_at"].dt.dayofweek
    daily = df_copy.groupby(["dow", "member_casual"]).size().unstack(fill_value=0)
    daily.index = DOW_LABELS
    daily.columns = ["散客", "会员"]
    colors = ["#2E86AB", "#A23B72"]
    daily.plot(kind="bar", ax=ax, color=colors, width=0.7)
    ax.set_title("每日骑行分布", fontsize=18, fontweight="bold")
    ax.set_xlabel("星期", fontsize=13)
    ax.set_ylabel("骑行次数", fontsize=13)
    ax.legend(fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    return save_figure(fig, "04_daily_pattern.png", save_path)


# ── 图5：季节性使用分布 ────────────────────────────
def plot_seasonal_usage(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    df_copy = df.copy()
    df_copy["season"] = df_copy["started_at"].dt.month.map(SEASONS)
    seasonal = (
        df_copy.groupby(["season", "member_casual"]).size().unstack(fill_value=0)
    )
    seasonal = seasonal.reindex([s for s in SEASON_ORDER if s in seasonal.index])
    seasonal.columns = ["散客", "会员"]
    seasonal["散客占比"] = seasonal["散客"] / seasonal.sum(axis=1) * 100

    colors = ["#2E86AB", "#A23B72"]
    ax2 = ax.twinx()
    seasonal[["散客", "会员"]].plot(kind="bar", ax=ax, color=colors, width=0.7)
    seasonal["散客占比"].plot(
        ax=ax2, linewidth=3, marker="D", markersize=8, color="#D81159", label="散客占比(%)"
    )
    ax.set_title("季节性使用分布", fontsize=18, fontweight="bold")
    ax.set_xlabel("季节", fontsize=13)
    ax.set_ylabel("骑行次数", fontsize=13)
    ax2.set_ylabel("散客占比 (%)", fontsize=13)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    return save_figure(fig, "05_seasonal_usage.png", save_path)


# ── 图6：骑行时长分布 ──────────────────────────────
def plot_duration_distribution(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    for label, color, style in [
        ("member", "#A23B72", "-"),
        ("casual", "#2E86AB", "--"),
    ]:
        subset = df[df["member_casual"] == label]["duration_minutes"]
        subset = subset[subset <= subset.quantile(0.99)]
        sns.kdeplot(subset, ax=ax, label="会员" if label == "member" else "散客",
                    color=color, linestyle=style, linewidth=2.5, fill=True, alpha=0.15)
    ax.set_title("骑行时长分布（会员 vs 散客）", fontsize=18, fontweight="bold")
    ax.set_xlabel("时长 (分钟)", fontsize=13)
    ax.set_ylabel("密度", fontsize=13)
    ax.legend(fontsize=12)
    return save_figure(fig, "06_duration_distribution.png", save_path)


# ── 图7：骑行距离分布 ──────────────────────────────
def plot_distance_distribution(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    for label, color, style in [
        ("member", "#A23B72", "-"),
        ("casual", "#2E86AB", "--"),
    ]:
        subset = df[df["member_casual"] == label]["haversine_distance_km"]
        sns.kdeplot(subset, ax=ax, label="会员" if label == "member" else "散客",
                    color=color, linestyle=style, linewidth=2.5, fill=True, alpha=0.15)
    ax.set_title("骑行距离分布（会员 vs 散客）", fontsize=18, fontweight="bold")
    ax.set_xlabel("距离 (km)", fontsize=13)
    ax.set_ylabel("密度", fontsize=13)
    ax.legend(fontsize=12)
    return save_figure(fig, "07_distance_distribution.png", save_path)


# ── 图8：平均速度箱线图 ────────────────────────────
def plot_speed_boxplot(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    plot_df = df[["member_casual", "avg_speed_kmh"]].copy()
    plot_df["用户类型"] = plot_df["member_casual"].map({"member": "会员", "casual": "散客"})
    sns.boxplot(
        data=plot_df, x="用户类型", y="avg_speed_kmh",
        hue="用户类型",
        palette={"会员": "#A23B72", "散客": "#2E86AB"}, ax=ax, width=0.5,
        legend=False,
    )
    ax.set_title("平均速度对比（会员 vs 散客）", fontsize=18, fontweight="bold")
    ax.set_xlabel("用户类型", fontsize=13)
    ax.set_ylabel("平均速度 (km/h)", fontsize=13)
    return save_figure(fig, "08_speed_boxplot.png", save_path)


# ── 图9：高峰时段分布 ──────────────────────────────
def plot_rush_hour_split(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    df_copy = df.copy()
    hour = df_copy["started_at"].dt.hour
    dow = df_copy["started_at"].dt.dayofweek
    df_copy["period"] = "其他时段"
    ml = (dow < 5) & (hour >= 7) & (hour <= 9)
    el = (dow < 5) & (hour >= 16) & (hour <= 19)
    df_copy.loc[ml, "period"] = "早高峰(7-9h)"
    df_copy.loc[el, "period"] = "晚高峰(16-19h)"

    cross = (
        df_copy.groupby(["period", "member_casual"]).size().unstack(fill_value=0)
    )
    cross = cross.reindex(["早高峰(7-9h)", "晚高峰(16-19h)", "其他时段"])
    cross.columns = ["散客", "会员"]
    colors = ["#2E86AB", "#A23B72"]
    cross.plot(kind="bar", ax=ax, color=colors, width=0.7)
    ax.set_title("高峰时段用户分布", fontsize=18, fontweight="bold")
    ax.set_xlabel("时段", fontsize=13)
    ax.set_ylabel("骑行次数", fontsize=13)
    ax.legend(fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    return save_figure(fig, "09_rush_hour_split.png", save_path)


# ── 图10：最热门起点站 Top 15 ──────────────────────
def plot_top_stations(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    station_counts = (
        df["start_station_name"].value_counts().head(15).sort_values()
    )
    ax.barh(range(len(station_counts)), station_counts.values,
            color=sns.color_palette("viridis", len(station_counts)))
    # 截断过长的站点名
    names = [n[:30] if len(str(n)) > 30 else str(n) for n in station_counts.index]
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("最热门起点站 Top 15", fontsize=18, fontweight="bold")
    ax.set_xlabel("骑行次数", fontsize=13)
    return save_figure(fig, "10_top_stations.png", save_path)


# ── 图11：单车类型偏好 ─────────────────────────────
def plot_bike_type(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    cross = (
        df.groupby(["rideable_type", "member_casual"]).size().unstack(fill_value=0)
    )
    cross.index = ["经典单车", "电动单车"]
    cross.columns = ["散客", "会员"]
    colors = ["#2E86AB", "#A23B72"]
    cross.plot(kind="bar", ax=ax, color=colors, width=0.7)
    ax.set_title("单车类型偏好（会员 vs 散客）", fontsize=18, fontweight="bold")
    ax.set_xlabel("单车类型", fontsize=13)
    ax.set_ylabel("骑行次数", fontsize=13)
    ax.legend(fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    return save_figure(fig, "11_bike_type.png", save_path)


# ── 图12：无桩骑行用户分布 ────────────────────────
def plot_dockless_proportion(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    df_copy = df.copy()
    df_copy["有无站点"] = df_copy["start_station_name"].notna().map(
        {True: "有站点", False: "无站点（无桩）"}
    )
    cross = (
        df_copy.groupby(["有无站点", "member_casual"]).size().unstack(fill_value=0)
    )
    cross.columns = ["散客", "会员"]
    colors = ["#2E86AB", "#A23B72"]
    cross.plot(kind="bar", ax=ax, color=colors, width=0.7)
    ax.set_title("无桩骑行用户分布", fontsize=18, fontweight="bold")
    ax.set_xlabel("站点状态", fontsize=13)
    ax.set_ylabel("骑行次数", fontsize=13)
    ax.legend(fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    return save_figure(fig, "12_dockless_proportion.png", save_path)


# ── 图13：特征相关性热力图 ────────────────────────
def plot_correlation_heatmap(
    df: pd.DataFrame, feature_cols: list[str], save_path: str | None = None
) -> str:
    fig, ax = plt.subplots(figsize=(14, 10))
    setup_chinese_font()
    set_style()
    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 7},
    )
    ax.set_title("特征相关性热力图", fontsize=18, fontweight="bold")
    return save_figure(fig, "13_correlation_heatmap.png", save_path)


# ── 图14：骑行时长 vs 骑行距离散点图 ──────────────
def plot_duration_vs_distance(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    sample_plot = df.sample(min(5000, len(df)), random_state=42)
    colors = sample_plot["member_casual"].map(
        {"member": "#A23B72", "casual": "#2E86AB"}
    )
    ax.scatter(
        sample_plot["duration_minutes"],
        sample_plot["haversine_distance_km"],
        c=colors, alpha=0.4, s=15, edgecolors="none",
    )
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#A23B72",
               markersize=10, label="会员"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E86AB",
               markersize=10, label="散客"),
    ]
    ax.legend(handles=legend_elements, fontsize=12)
    ax.set_title("骑行时长 vs 骑行距离", fontsize=18, fontweight="bold")
    ax.set_xlabel("时长 (分钟)", fontsize=13)
    ax.set_ylabel("距离 (km)", fontsize=13)
    return save_figure(fig, "14_duration_vs_distance.png", save_path)


# ── 图15：月度散客占比变化趋势 ────────────────────
def plot_monthly_class_balance(df: pd.DataFrame, save_path: str | None = None) -> str:
    fig, ax = _fig()
    monthly_ratio = (
        df.groupby("month")["member_casual"]
        .apply(lambda x: (x == "casual").mean() * 100)
        .sort_index()
    )
    months = [m[-2:] + "月" for m in monthly_ratio.index]
    ax.plot(months, monthly_ratio.values, marker="s", linewidth=2.5,
            markersize=8, color="#D81159")
    ax.fill_between(range(len(months)), monthly_ratio.values, alpha=0.2, color="#D81159")
    ax.set_title("月度散客占比变化趋势", fontsize=18, fontweight="bold")
    ax.set_xlabel("月份", fontsize=13)
    ax.set_ylabel("散客占比 (%)", fontsize=13)
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(y=monthly_ratio.mean(), color="gray", linestyle="--", alpha=0.7,
               label=f"年均 {monthly_ratio.mean():.1f}%")
    ax.legend(fontsize=12)
    return save_figure(fig, "15_monthly_class_balance.png", save_path)


# ── 编排函数 ────────────────────────────────────────
def run_all_eda(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    output_dir: str = FIGURES_DIR,
) -> list[str]:
    """依次执行全部 15 张 EDA 图并返回保存的文件路径列表。"""
    paths = []
    paths.append(plot_user_distribution(df, output_dir))
    paths.append(plot_monthly_volume(df, output_dir))
    paths.append(plot_hourly_pattern(df, output_dir))
    paths.append(plot_daily_pattern(df, output_dir))
    paths.append(plot_seasonal_usage(df, output_dir))
    paths.append(plot_duration_distribution(df, output_dir))
    paths.append(plot_distance_distribution(df, output_dir))
    paths.append(plot_speed_boxplot(df, output_dir))
    paths.append(plot_rush_hour_split(df, output_dir))
    paths.append(plot_top_stations(df, output_dir))
    paths.append(plot_bike_type(df, output_dir))
    paths.append(plot_dockless_proportion(df, output_dir))
    if feature_cols:
        paths.append(plot_correlation_heatmap(
            df, feature_cols, output_dir))
    paths.append(plot_duration_vs_distance(df, output_dir))
    paths.append(plot_monthly_class_balance(df, output_dir))
    print(f"Saved {len(paths)} EDA figures to {output_dir}")
    return paths
