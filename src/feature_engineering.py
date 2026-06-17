import numpy as np
import pandas as pd
from src.config import TARGET_MAP


def _cyclic_encode(values: pd.Series, period: float) -> tuple[np.ndarray, np.ndarray]:
    """将循环变量（取值范围 0..period-1）映射到 (sin, cos) 坐标。"""
    scaled = 2 * np.pi * values / period
    return np.sin(scaled), np.cos(scaled)


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 started_at 中提取循环编码和二元时间特征。"""
    feats = pd.DataFrame(index=df.index)

    hour = df["started_at"].dt.hour.astype(float)
    feats["hour_sin"], feats["hour_cos"] = _cyclic_encode(hour, 24.0)

    dow = df["started_at"].dt.dayofweek.astype(float)  # 周一=0, 周日=6
    feats["dayofweek_sin"], feats["dayofweek_cos"] = _cyclic_encode(dow, 7.0)

    month = df["started_at"].dt.month.astype(float)  # 1..12
    feats["month_sin"], feats["month_cos"] = _cyclic_encode(month, 12.0)

    feats["is_weekend"] = (dow >= 5).astype(int)
    feats["is_morning_rush"] = (
        (dow < 5) & (hour >= 7) & (hour <= 9)
    ).astype(int)
    feats["is_evening_rush"] = (
        (dow < 5) & (hour >= 16) & (hour <= 19)
    ).astype(int)

    return feats


def extract_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取骑行相关的数值特征。"""
    feats = pd.DataFrame(index=df.index)
    feats["duration_minutes_log"] = np.log1p(df["duration_minutes"])
    feats["haversine_distance_km"] = df["haversine_distance_km"]
    feats["is_electric_bike"] = (df["rideable_type"] == "electric_bike").astype(int)
    return feats


def compute_station_frequency_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """对起止站点进行频率编码。

    站点频率 = 该站点的骑行次数 / 总骑行次数。
    缺失站点频率记为 0。
    """
    feats = pd.DataFrame(index=df.index)
    total_rides = len(df)

    start_counts = df["start_station_name"].value_counts()
    end_counts = df["end_station_name"].value_counts()

    feats["has_start_station"] = df["start_station_name"].notna().astype(int)
    feats["has_end_station"] = df["end_station_name"].notna().astype(int)

    feats["start_station_freq"] = (
        df["start_station_name"].map(start_counts) / total_rides
    ).fillna(0)
    feats["end_station_freq"] = (
        df["end_station_name"].map(end_counts) / total_rides
    ).fillna(0)

    return feats


def extract_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取复合衍生特征。"""
    feats = pd.DataFrame(index=df.index)
    feats["avg_speed_kmh"] = df["avg_speed_kmh"]
    return feats


def engineer_all_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """执行全部特征工程步骤，返回 (X, y, feature_names)。

    X: (样本数, 特征数) float64 数组
    y: (样本数,) int 数组 (0=casual, 1=member)
    feature_names: 特征列名列表
    """
    temporal = extract_temporal_features(df)
    trip = extract_trip_features(df)
    station = compute_station_frequency_encoding(df)
    composite = extract_composite_features(df)

    feature_df = pd.concat([temporal, trip, station, composite], axis=1)
    X = feature_df.values.astype(np.float64)
    feature_names = list(feature_df.columns)
    y = df["member_casual"].map(TARGET_MAP).values.astype(int)

    assert not np.any(np.isnan(X)), "特征矩阵中存在 NaN 值"
    assert not np.any(np.isinf(X)), "特征矩阵中存在 Inf 值"

    return X, y, feature_names
