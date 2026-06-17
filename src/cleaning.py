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
    """Correct negative durations caused by DST clock rollback in November.

    Adds 3600 seconds (1 hour) to negative durations.
    """
    mask = df["ended_at"] < df["started_at"]
    if mask.any():
        df.loc[mask, "ended_at"] = df.loc[mask, "ended_at"] + pd.Timedelta(hours=1)
    return df


def compute_haversine_km(
    lat1: pd.Series, lng1: pd.Series, lat2: pd.Series, lng2: pd.Series
) -> np.ndarray:
    """Compute haversine distance in km between two sets of lat/lng coordinates."""
    r = 6371.0  # Earth radius in km
    lat1_r = np.radians(lat1.values)
    lat2_r = np.radians(lat2.values)
    dlat = np.radians(lat2.values - lat1.values)
    dlng = np.radians(lng2.values - lng1.values)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full data cleaning pipeline.

    Returns cleaned DataFrame.
    """
    initial_rows = len(df)

    # 1. Fix DST negative durations
    df = fix_dst_negative_durations(df)

    # 2. Compute duration and distance
    df["duration_minutes"] = (
        df["ended_at"] - df["started_at"]
    ).dt.total_seconds() / 60.0
    df["haversine_distance_km"] = compute_haversine_km(
        df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]
    )

    # 3. Remove rows missing end lat/lng (too few to matter)
    missing_end_coords = df["end_lat"].isna() | df["end_lng"].isna()
    df = df[~missing_end_coords]

    # 4. Remove duration outliers
    df = df[df["duration_minutes"] >= MIN_DURATION_MINUTES]
    df = df[df["duration_minutes"] <= MAX_DURATION_MINUTES]

    # 5. Remove rows with haversine=0 but different start/end station
    zero_dist_diff_station = (df["haversine_distance_km"] == 0) & (
        df["start_station_id"] != df["end_station_id"]
    )
    df = df[~zero_dist_diff_station]

    # 6. IQR capping for distance
    q1_dist = df["haversine_distance_km"].quantile(0.25)
    q3_dist = df["haversine_distance_km"].quantile(0.75)
    iqr_dist = q3_dist - q1_dist
    lower_dist = max(q1_dist - 1.5 * iqr_dist, 0)
    upper_dist = q3_dist + 1.5 * iqr_dist
    df["haversine_distance_km"] = df["haversine_distance_km"].clip(
        lower_dist, upper_dist
    )

    # 7. IQR capping for duration (after outlier removal)
    q1_dur = df["duration_minutes"].quantile(0.25)
    q3_dur = df["duration_minutes"].quantile(0.75)
    iqr_dur = q3_dur - q1_dur
    lower_dur = max(q1_dur - 1.5 * iqr_dur, MIN_DURATION_MINUTES)
    upper_dur = min(q3_dur + 1.5 * iqr_dur, MAX_DURATION_MINUTES)
    df["duration_minutes"] = df["duration_minutes"].clip(lower_dur, upper_dur)

    # 8. Compute avg speed after capping
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
    """Generate before/after cleaning comparison table."""
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
