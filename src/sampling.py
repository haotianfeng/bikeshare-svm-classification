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

# 完整流程所需的原始CSV列
NEEDED_COLS = [
    "ride_id", "rideable_type",
    "started_at", "ended_at",
    "start_station_name", "start_station_id",
    "end_station_name", "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng",
    "member_casual",
]

# 计数和分层抽样所需的最小列集合
SAMPLING_COLS = ["member_casual"]


def _count_lines_fast(path: str) -> int:
    """Count lines in a CSV file quickly (for proportional allocation)."""
    n = 0
    with open(path, "rb") as f:
        # 使用缓冲二进制读取以提速
        buf = f.read(8192 * 1024)  # 8MB 缓冲区
        while buf:
            n += buf.count(b"\n")
            buf = f.read(8192 * 1024)
    return n - 1  # 减去表头行


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
    # 基于目标样本数确定分块大小
    # 我们抽样约1-10%的行，所以10万行的chunk → 每块抽1-1万条
    chunk_size = 100_000
    sampled_chunks = []

    # 第一遍：获取列名
    header_df = pd.read_csv(path, nrows=0)
    available_cols = [c for c in NEEDED_COLS if c in header_df.columns]
    # 还需要从started_at中提取月份信息
    if "started_at" not in available_cols:
        available_cols.append("started_at")

    # 第二遍：分块读取，每块内抽样
    for chunk in pd.read_csv(
        path,
        usecols=available_cols,
        chunksize=chunk_size,
        parse_dates=["started_at", "ended_at"] if "started_at" in available_cols else False,
    ):
        # 块内分层抽样
        for _, g in chunk.groupby("member_casual"):
            # 块内按比例抽样
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

    # 如果结果超过目标抽样数，二次抽样
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

    # 快速统计每月行数，用于比例分配（不加载完整文件）
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
        # 按比例分配
        month_samples = max(int(round(month_n / total_rows * total_samples)), 1)

        print(f"    处理 {month}: {month_n:,} 行 → 抽样 {month_samples:,} 行...")

        # 内存优化：只读所需列，分块加载
        sampled = _load_and_sample_month(path, month_samples, random_seed)

        # 添加月份列
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

        # 每月处理后强制垃圾回收
        gc.collect()

    combined = pd.concat(samples, ignore_index=True)

    # 释放内存
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
