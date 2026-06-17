import os
import glob
import pandas as pd
from src.config import RAW_DATA_DIR


def discover_csv_files(data_dir: str | None = None) -> dict[str, str]:
    """Return {month_label: csv_path} for all 12 monthly tripdata files."""
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    pattern = os.path.join(data_dir, "2025*-capitalbikeshare-tripdata", "*.csv")
    files = sorted(glob.glob(pattern))
    result = {}
    for f in files:
        month = os.path.basename(os.path.dirname(f))[:6]  # 目录名前6位，如 "202501"
        result[month] = f
    return result


def load_single_month(csv_path: str) -> pd.DataFrame:
    """Load one monthly CSV, parse timestamps, add month column."""
    df = pd.read_csv(csv_path)
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    df["month"] = df["started_at"].dt.to_period("M").astype(str)
    return df


def load_all_months(data_dir: str | None = None) -> pd.DataFrame:
    """Load all 12 months, concatenate, return combined DataFrame."""
    files = discover_csv_files(data_dir)
    frames = []
    for month, path in files.items():
        df = load_single_month(path)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def compute_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-month row counts and class distribution."""
    stats = (
        df.groupby("month")
        .agg(
            total_rows=("ride_id", "count"),
            member_pct=("member_casual", lambda x: (x == "member").mean() * 100),
            casual_pct=("member_casual", lambda x: (x == "casual").mean() * 100),
        )
        .reset_index()
    )
    stats["member_pct"] = stats["member_pct"].round(1)
    stats["casual_pct"] = stats["casual_pct"].round(1)
    return stats
