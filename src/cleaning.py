import numpy as np
import pandas as pd
from src.config import (
    MIN_DURATION_MINUTES,
    MAX_DURATION_MINUTES,
    MIN_AVG_SPEED_KMH,
    MAX_AVG_SPEED_KMH,
    RANDOM_SEED,
    TABLES_DIR,
)


def fix_dst_negative_durations(df: pd.DataFrame) -> pd.DataFrame:
    """修正因 11 月夏令时回拨导致的负骑行时长。

    将负时长加回 3600 秒（1 小时）以纠正 DST 时钟误差。
    """
    mask = df["ended_at"] < df["started_at"]
    if mask.any():
        df.loc[mask, "ended_at"] = df.loc[mask, "ended_at"] + pd.Timedelta(hours=1)
    return df


def compute_haversine_km(
    lat1: pd.Series, lng1: pd.Series, lat2: pd.Series, lng2: pd.Series
) -> np.ndarray:
    """使用 Haversine 公式计算两组经纬度坐标之间的球面距离（单位 km）。"""
    r = 6371.0  # 地球半径，单位km
    lat1_r = np.radians(lat1.values)
    lat2_r = np.radians(lat2.values)
    dlat = np.radians(lat2.values - lat1.values)
    dlng = np.radians(lng2.values - lng1.values)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """完整的数据清洗流水线。

    按顺序执行：DST 修正 → 时长/距离计算 → 移除极端值 →
    IQR 缩尾 → 计算平均速度，返回清洗后的 DataFrame。
    """
    initial_rows = len(df)

    # 1. 修复夏令时导致的负时长
    df = fix_dst_negative_durations(df)

    # 2. 计算骑行时长和Haversine距离
    df["duration_minutes"] = (
        df["ended_at"] - df["started_at"]
    ).dt.total_seconds() / 60.0
    df["haversine_distance_km"] = compute_haversine_km(
        df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]
    )

    # 3. 移除终点经纬度缺失的行（占比极少）
    missing_end_coords = df["end_lat"].isna() | df["end_lng"].isna()
    df = df[~missing_end_coords]

    # 4. 移除时长异常值（<1分钟或>24小时）
    df = df[df["duration_minutes"] >= MIN_DURATION_MINUTES]
    df = df[df["duration_minutes"] <= MAX_DURATION_MINUTES]

    # 5. 移除距离为0但起止站点不同的矛盾记录
    zero_dist_diff_station = (df["haversine_distance_km"] == 0) & (
        df["start_station_id"] != df["end_station_id"]
    )
    df = df[~zero_dist_diff_station]

    # 6. 对距离做 IQR 截尾
    q1_dist = df["haversine_distance_km"].quantile(0.25)
    q3_dist = df["haversine_distance_km"].quantile(0.75)
    iqr_dist = q3_dist - q1_dist
    lower_dist = max(q1_dist - 1.5 * iqr_dist, 0)
    upper_dist = q3_dist + 1.5 * iqr_dist
    df["haversine_distance_km"] = df["haversine_distance_km"].clip(
        lower_dist, upper_dist
    )

    # 7. 对时长做 IQR 截尾（在异常值移除之后）
    q1_dur = df["duration_minutes"].quantile(0.25)
    q3_dur = df["duration_minutes"].quantile(0.75)
    iqr_dur = q3_dur - q1_dur
    lower_dur = max(q1_dur - 1.5 * iqr_dur, MIN_DURATION_MINUTES)
    upper_dur = min(q3_dur + 1.5 * iqr_dur, MAX_DURATION_MINUTES)
    df["duration_minutes"] = df["duration_minutes"].clip(lower_dur, upper_dur)

    # 8. 截尾后计算平均骑行速度
    df["avg_speed_kmh"] = df["haversine_distance_km"] / (df["duration_minutes"] / 60.0)
    df["avg_speed_kmh"] = df["avg_speed_kmh"].clip(
        MIN_AVG_SPEED_KMH, MAX_AVG_SPEED_KMH
    )

    print(
        f"Cleaning: {initial_rows} -> {len(df)} rows "
        f"({(initial_rows - len(df)) / initial_rows * 100:.1f}% removed)"
    )
    return df


def report_cleaning_stats(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """生成清洗前后对比统计表并保存为 CSV。"""
    import os

    stats = pd.DataFrame(
        {
            "metric": [
                "total_rows",
                "missing_start_station_pct",
                "missing_end_station_pct",
                "duration_lt_1min_pct",
                "duration_gt_24h_pct",
            ],
            "before": [
                len(before),
                before["start_station_name"].isna().mean() * 100,
                before["end_station_name"].isna().mean() * 100,
                (
                    (before["ended_at"] - before["started_at"]).dt.total_seconds() / 60
                    < 1
                ).mean()
                * 100,
                (
                    (before["ended_at"] - before["started_at"]).dt.total_seconds() / 60
                    > 1440
                ).mean()
                * 100,
            ],
            "after": [
                len(after),
                after["start_station_name"].isna().mean() * 100,
                after["end_station_name"].isna().mean() * 100,
                0.0,
                0.0,
            ],
        }
    )
    stats.to_csv(os.path.join(TABLES_DIR, "cleaning_stats.csv"), index=False)
    return stats
