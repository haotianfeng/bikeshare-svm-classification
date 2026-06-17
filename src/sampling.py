"""Memory-optimized stratified sampling by month.

Key optimizations vs original:
1. Only reads columns needed for sampling (member_casual, started_at) plus key cols
2. Uses chunked reading for large files to avoid loading full month into memory
3. Calls gc.collect() after each month to release memory immediately
"""

import gc
import pandas as pd
import numpy as np
from src.config import TOTAL_SAMPLE_SIZE, RANDOM_SEED, TABLES_DIR
from src.data_loader import discover_csv_files

# Columns we actually need from the raw CSV for the full pipeline
NEEDED_COLS = [
    "ride_id", "rideable_type",
    "started_at", "ended_at",
    "start_station_name", "start_station_id",
    "end_station_name", "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng",
    "member_casual",
]

# Minimal columns needed for counting + stratified sampling
SAMPLING_COLS = ["member_casual"]


def _count_lines_fast(path: str) -> int:
    """Count lines in a CSV file quickly (for proportional allocation)."""
    n = 0
    with open(path, "rb") as f:
        # Use buffered binary read for speed
        buf = f.read(8192 * 1024)  # 8MB buffer
        while buf:
            n += buf.count(b"\n")
            buf = f.read(8192 * 1024)
    return n - 1  # minus header


def _load_and_sample_month(
    path: str,
    month_samples: int,
    random_seed: int,
) -> pd.DataFrame:
    """Load one month's CSV with only needed columns, then stratified-sample.

    Uses chunked reading for large files: reads in chunks of 100k rows,
    samples within each chunk, and concatenates. This avoids loading a
    700k-row DataFrame into memory at once.
    """
    # Determine chunk size based on target samples
    # We sample ~1-10% of rows, so chunk of 100k → ~1-10k sampled per chunk
    chunk_size = 100_000
    sampled_chunks = []

    # First pass: get column names
    header_df = pd.read_csv(path, nrows=0)
    available_cols = [c for c in NEEDED_COLS if c in header_df.columns]
    # We also need month info from started_at, so ensure it's included
    if "started_at" not in available_cols:
        available_cols.append("started_at")

    # Second pass: read in chunks, sample each
    for chunk in pd.read_csv(
        path,
        usecols=available_cols,
        chunksize=chunk_size,
        parse_dates=["started_at", "ended_at"] if "started_at" in available_cols else False,
    ):
        # Stratified sampling within this chunk
        for _, g in chunk.groupby("member_casual"):
            # Sample proportionally within chunk
            chunk_n = max(
                int(round(month_samples * len(g) / len(chunk))), 1
            )
            if chunk_n < len(g):
                sampled_chunks.append(
                    g.sample(n=chunk_n, random_state=random_seed)
                )
            else:
                sampled_chunks.append(g)

    if not sampled_chunks:
        return pd.DataFrame(columns=available_cols)

    result = pd.concat(sampled_chunks, ignore_index=True)

    # If we have more than month_samples, subsample
    if len(result) > month_samples:
        result = result.sample(n=month_samples, random_state=random_seed)

    return result


def stratified_sample_by_month(
    data_dir: str | None = None,
    total_samples: int = TOTAL_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Load all 12 months, sample proportionally with per-month stratification.

    Each month gets sample rows proportional to its original volume.
    Within each month, stratified sampling preserves the member/casual ratio.

    Memory-optimized: only reads needed columns, uses chunked reading,
    and forces garbage collection after each month.
    """
    files = discover_csv_files(data_dir)

    # Count rows per month for proportional allocation (fast, no full load)
    monthly_counts = {}
    for month, path in files.items():
        monthly_counts[month] = _count_lines_fast(path)

    total_rows = sum(monthly_counts.values())
    samples = []
    sampling_log = []

    print(f"  原始总行数: {total_rows:,}, 目标抽样: {total_samples:,}")
    print(f"  抽样比例: {total_samples / total_rows * 100:.2f}%")

    for month, path in sorted(files.items()):
        month_n = monthly_counts[month]
        # Proportional allocation
        month_samples = max(int(round(month_n / total_rows * total_samples)), 1)

        print(f"    处理 {month}: {month_n:,} 行 → 抽样 {month_samples:,} 行...")

        # Memory-optimized: only read needed cols, chunked
        sampled = _load_and_sample_month(path, month_samples, random_seed)

        # Add month column
        sampled["started_at"] = pd.to_datetime(sampled["started_at"])
        sampled["ended_at"] = pd.to_datetime(sampled["ended_at"])
        sampled["month"] = sampled["started_at"].dt.to_period("M").astype(str)

        samples.append(sampled)

        member_n = (sampled["member_casual"] == "member").sum()
        casual_n = (sampled["member_casual"] == "casual").sum()
        sampling_log.append({
            "month": month,
            "original_rows": month_n,
            "sampled_rows": len(sampled),
            "member_rows": member_n,
            "casual_rows": casual_n,
        })
        print(f"      成员: {member_n}, 散客: {casual_n}")

        # Force garbage collection after each month
        gc.collect()

    combined = pd.concat(samples, ignore_index=True)

    # Free memory
    del samples
    gc.collect()

    print(f"  最终抽样: {len(combined):,} 条记录")
    return combined


def save_sampling_summary(sampled_df: pd.DataFrame, output_dir: str = TABLES_DIR) -> None:
    """Save monthly sampling summary table."""
    import os

    summary = (
        sampled_df.groupby("month")
        .agg(
            total=("ride_id", "count"),
            member=("member_casual", lambda x: (x == "member").sum()),
            casual=("member_casual", lambda x: (x == "casual").sum()),
            member_pct=("member_casual", lambda x: (x == "member").mean() * 100),
        )
        .reset_index()
    )
    summary["member_pct"] = summary["member_pct"].round(1)
    summary.to_csv(os.path.join(output_dir, "sampling_summary.csv"), index=False)
    print(f"  Sampling summary saved to {output_dir}/sampling_summary.csv")
