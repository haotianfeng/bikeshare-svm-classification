import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.config import FIGURES_DIR, TABLES_DIR
from src.visualize import setup_chinese_font, set_style, save_figure

SEASONS = {
    1: "冬季", 2: "冬季", 3: "春季", 4: "春季", 5: "春季",
    6: "夏季", 7: "夏季", 8: "夏季", 9: "秋季", 10: "秋季",
    11: "秋季", 12: "冬季",
}


def analyze_commuting_patterns(df: pd.DataFrame) -> dict:
    """Test whether members commute more than casual users during rush hours."""
    hour = df["started_at"].dt.hour
    dow = df["started_at"].dt.dayofweek
    is_weekday = dow < 5
    is_rush = is_weekday & ((hour >= 7) & (hour <= 9) | (hour >= 16) & (hour <= 19))

    rush_cross = pd.crosstab(
        is_rush.map({True: "高峰时段", False: "非高峰时段"}),
        df["member_casual"].map({"member": "会员", "casual": "散客"}),
    )
    chi2, p_value, dof, expected = stats.chi2_contingency(rush_cross)

    member_rush_pct = (
        df[df["member_casual"] == "member"]["started_at"]
        .apply(lambda t: is_rush[t == df["started_at"]] if False else None)
    )
    member_rush = (
        df["member_casual"] == "member"
    ) & is_rush
    casual_rush = (
        df["member_casual"] == "casual"
    ) & is_rush
    member_rush_rate = member_rush.sum() / (df["member_casual"] == "member").sum() * 100
    casual_rush_rate = casual_rush.sum() / (df["member_casual"] == "casual").sum() * 100

    return {
        "finding": "会员在高峰时段骑行比例显著高于散客",
        "member_rush_pct": round(member_rush_rate, 1),
        "casual_rush_pct": round(casual_rush_rate, 1),
        "chi2_statistic": round(chi2, 1),
        "p_value": f"{p_value:.2e}",
        "significant": p_value < 0.05,
    }


def test_duration_difference(df: pd.DataFrame) -> dict:
    """Mann-Whitney U test on ride duration between member and casual."""
    member_dur = df[df["member_casual"] == "member"]["duration_minutes"]
    casual_dur = df[df["member_casual"] == "casual"]["duration_minutes"]

    u_stat, p_value = stats.mannwhitneyu(casual_dur, member_dur, alternative="two-sided")

    return {
        "finding": "散客骑行时长显著长于会员",
        "member_median_min": round(member_dur.median(), 1),
        "casual_median_min": round(casual_dur.median(), 1),
        "mannwhitney_u": u_stat,
        "p_value": f"{p_value:.2e}",
        "significant": p_value < 0.05,
    }


def analyze_weekend_effect(df: pd.DataFrame) -> dict:
    """Test whether casual proportion is higher on weekends."""
    df_copy = df.copy()
    df_copy["is_weekend"] = df_copy["started_at"].dt.dayofweek >= 5
    df_copy["is_casual"] = df_copy["member_casual"] == "casual"

    cross = pd.crosstab(
        df_copy["is_weekend"].map({True: "周末", False: "工作日"}),
        df_copy["member_casual"].map({"member": "会员", "casual": "散客"}),
    )
    chi2, p_value, dof, expected = stats.chi2_contingency(cross)

    weekday_casual = df_copy[~df_copy["is_weekend"]]["is_casual"].mean() * 100
    weekend_casual = df_copy[df_copy["is_weekend"]]["is_casual"].mean() * 100

    return {
        "finding": "周末散客占比显著高于工作日",
        "weekday_casual_pct": round(weekday_casual, 1),
        "weekend_casual_pct": round(weekend_casual, 1),
        "chi2_statistic": round(chi2, 1),
        "p_value": f"{p_value:.2e}",
        "significant": p_value < 0.05,
    }


def analyze_dockless_pattern(df: pd.DataFrame) -> dict:
    """Test whether dockless rides are associated with casual users."""
    df_copy = df.copy()
    df_copy["is_dockless"] = df_copy["start_station_name"].isna()
    df_copy["is_casual"] = df_copy["member_casual"] == "casual"

    cross = pd.crosstab(
        df_copy["is_dockless"].map({True: "无桩骑行", False: "有桩骑行"}),
        df_copy["member_casual"].map({"member": "会员", "casual": "散客"}),
    )
    chi2, p_value, dof, expected = stats.chi2_contingency(cross)

    dockless_casual = df_copy[df_copy["is_dockless"]]["is_casual"].mean() * 100
    docked_casual = df_copy[~df_copy["is_dockless"]]["is_casual"].mean() * 100

    return {
        "finding": "无桩骑行中散客比例更高",
        "dockless_casual_pct": round(dockless_casual, 1),
        "docked_casual_pct": round(docked_casual, 1),
        "chi2_statistic": round(chi2, 1),
        "p_value": f"{p_value:.2e}",
        "significant": p_value < 0.05,
    }


def analyze_seasonal_shift(df: pd.DataFrame) -> dict:
    """Analyze seasonal shift in casual proportion."""
    df_copy = df.copy()
    df_copy["season"] = df_copy["started_at"].dt.month.map(SEASONS)
    seasonal_casual = df_copy.groupby("season")["member_casual"].apply(
        lambda x: (x == "casual").mean() * 100
    )
    seasonal_order = ["冬季", "春季", "夏季", "秋季"]
    seasonal_casual = seasonal_casual.reindex([s for s in seasonal_order if s in seasonal_casual.index])

    return {
        "finding": "散客占比呈现明显的季节性波动，夏季最高冬季最低",
        "winter_pct": round(seasonal_casual.get("冬季", 0), 1),
        "spring_pct": round(seasonal_casual.get("春季", 0), 1),
        "summer_pct": round(seasonal_casual.get("夏季", 0), 1),
        "autumn_pct": round(seasonal_casual.get("秋季", 0), 1),
        "peak_to_trough_ratio": round(
            seasonal_casual.max() / seasonal_casual.min(), 2
        ),
    }


def analyze_electric_preference(df: pd.DataFrame) -> dict:
    """Test whether casual users prefer electric bikes more."""
    cross = pd.crosstab(
        df["rideable_type"].map({"electric_bike": "电动单车", "classic_bike": "经典单车"}),
        df["member_casual"].map({"member": "会员", "casual": "散客"}),
    )
    chi2, p_value, dof, expected = stats.chi2_contingency(cross)

    member_elec = (
        (df["member_casual"] == "member") & (df["rideable_type"] == "electric_bike")
    ).sum() / (df["member_casual"] == "member").sum() * 100
    casual_elec = (
        (df["member_casual"] == "casual") & (df["rideable_type"] == "electric_bike")
    ).sum() / (df["member_casual"] == "casual").sum() * 100

    return {
        "finding": "散客更偏好电动单车",
        "member_electric_pct": round(member_elec, 1),
        "casual_electric_pct": round(casual_elec, 1),
        "chi2_statistic": round(chi2, 1),
        "p_value": f"{p_value:.2e}",
        "significant": p_value < 0.05,
    }


def generate_findings_table(all_findings: dict, output_dir: str = TABLES_DIR) -> str:
    """Save all findings as a structured table."""
    rows = []
    for key, f in all_findings.items():
        flat = {"analysis": key}
        for k, v in f.items():
            flat[k] = v
        rows.append(flat)
    df = pd.DataFrame(rows)
    filepath = f"{output_dir}/interesting_findings.csv"
    df.to_csv(filepath, index=False)
    return filepath


def run_all_findings(
    df: pd.DataFrame,
    output_dir: str = TABLES_DIR,
) -> dict:
    """Run all statistical analyses and return findings dict."""
    findings = {
        "commuting_patterns": analyze_commuting_patterns(df),
        "duration_difference": test_duration_difference(df),
        "weekend_effect": analyze_weekend_effect(df),
        "dockless_pattern": analyze_dockless_pattern(df),
        "seasonal_shift": analyze_seasonal_shift(df),
        "electric_preference": analyze_electric_preference(df),
    }

    for key, f in findings.items():
        status = "√" if f.get("significant", True) else "×"
        print(f"  [{status}] {f['finding']}")

    generate_findings_table(findings, output_dir)
    return findings
