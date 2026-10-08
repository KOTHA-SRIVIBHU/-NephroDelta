"""
Delta Analytics, Trend Analysis, and Analytical State Module for NephroDelta.

Calculates consecutive change metrics (delta), longitudinal slopes, and assigns
pathophysiology-inspired simulation analytical states for synthetic patient trajectories.

IMPORTANT DISCLAIMER:
Analytical states (0-3) and observed longitudinal slopes are simplified abstractions
created for simulation and research analytics. They are NOT clinical diagnoses,
staging systems, or medical prognoses.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


STATE_LABELS: Dict[int, str] = {
    0: "Stable analytical pattern",
    1: "Early abnormality pattern",
    2: "Moderate impairment pattern",
    3: "Advanced impairment pattern"
}


def calculate_safe_pct_change(current: float, previous: float) -> float:
    """
    Safely calculates percentage change between two numeric values.

    Args:
        current: Current observation value.
        previous: Previous observation value.

    Returns:
        Percentage change as a float, or NaN if invalid/zero denominator.
    """
    if pd.isna(current) or pd.isna(previous):
        return np.nan
    if abs(previous) < 1e-9:
        return np.nan
    return float(((current - previous) / previous) * 100.0)


def calculate_patient_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates consecutive observation deltas and percentage changes for each patient.

    Args:
        df: Input DataFrame (must be cleaned and sorted by patient_id and date).

    Returns:
        pd.DataFrame with added delta and percentage change columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_out = df.copy()

    # Delta targets
    metrics = ["egfr", "uacr", "creatinine", "hba1c", "systolic_bp", "weight"]

    for m in metrics:
        df_out[f"delta_{m}"] = df_out.groupby("patient_id")[m].diff()
        
        # Calculate percentage change safely per group
        prev_series = df_out.groupby("patient_id")[m].shift(1)
        curr_series = df_out[m]
        
        pct_series = np.where(
            (prev_series.notnull()) & (curr_series.notnull()) & (prev_series.abs() > 1e-9),
            ((curr_series - prev_series) / prev_series) * 100.0,
            np.nan
        )
        df_out[f"pct_change_{m}"] = pct_series

    return df_out


def calculate_patient_trends(df_patient: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates longitudinal summary metrics and observed slopes for a single patient's data.

    Args:
        df_patient: pd.DataFrame containing data for a single patient sorted by date.

    Returns:
        Dictionary containing initial/latest values, total changes, and longitudinal slopes.
    """
    if df_patient is None or df_patient.empty:
        return {
            "initial_egfr": np.nan, "latest_egfr": np.nan, "total_egfr_change": 0.0, "egfr_slope": 0.0,
            "initial_uacr": np.nan, "latest_uacr": np.nan, "total_uacr_change": 0.0, "uacr_slope": 0.0,
            "time_duration_years": 0.0, "num_observations": 0
        }

    df_sorted = df_patient.sort_values(by="date")
    num_obs = len(df_sorted)

    initial_row = df_sorted.iloc[0]
    latest_row = df_sorted.iloc[-1]

    # Calculate duration in years
    if num_obs > 1 and "date" in df_sorted.columns:
        date_start = pd.to_datetime(initial_row["date"])
        date_end = pd.to_datetime(latest_row["date"])
        duration_days = (date_end - date_start).days
        time_duration_years = max(duration_days / 365.25, 0.001)
    else:
        time_duration_years = 0.0

    # eGFR metrics
    initial_egfr = float(initial_row["egfr"]) if pd.notnull(initial_row["egfr"]) else 0.0
    latest_egfr = float(latest_row["egfr"]) if pd.notnull(latest_row["egfr"]) else 0.0
    total_egfr_change = latest_egfr - initial_egfr
    egfr_slope = total_egfr_change / time_duration_years if time_duration_years > 0 else 0.0

    # UACR metrics
    initial_uacr = float(initial_row["uacr"]) if pd.notnull(initial_row["uacr"]) else 0.0
    latest_uacr = float(latest_row["uacr"]) if pd.notnull(latest_row["uacr"]) else 0.0
    total_uacr_change = latest_uacr - initial_uacr
    uacr_slope = total_uacr_change / time_duration_years if time_duration_years > 0 else 0.0

    return {
        "patient_id": str(initial_row["patient_id"]),
        "initial_egfr": round(initial_egfr, 2),
        "latest_egfr": round(latest_egfr, 2),
        "total_egfr_change": round(total_egfr_change, 2),
        "egfr_slope": round(egfr_slope, 2),
        "initial_uacr": round(initial_uacr, 2),
        "latest_uacr": round(latest_uacr, 2),
        "total_uacr_change": round(total_uacr_change, 2),
        "uacr_slope": round(uacr_slope, 2),
        "time_duration_years": round(time_duration_years, 2),
        "num_observations": num_obs
    }


def classify_analytical_state(egfr: float, uacr: float) -> int:
    """
    Classifies synthetic eGFR and UACR into a pathophysiology-inspired analytical state.

    States:
        0: Stable analytical pattern (eGFR >= 90 and UACR < 30)
        1: Early abnormality pattern (eGFR 60-89 or UACR 30-300)
        2: Moderate impairment pattern (eGFR 30-59 or UACR > 300)
        3: Advanced impairment pattern (eGFR < 30)

    Note:
        The states are simplified analytical abstractions created for simulation
        and are not clinical classifications.

    Args:
        egfr: Estimated Glomerular Filtration Rate.
        uacr: Urine Albumin-to-Creatinine Ratio.

    Returns:
        Integer state (0, 1, 2, or 3).
    """
    if pd.isna(egfr) or pd.isna(uacr):
        return 0

    if egfr < 30.0:
        return 3
    elif egfr < 60.0 or uacr > 300.0:
        return 2
    elif egfr < 90.0 or uacr >= 30.0:
        return 1
    else:
        return 0


def get_state_description(state: int) -> str:
    """Returns the descriptive text label for an analytical state code."""
    return STATE_LABELS.get(state, "Unknown pattern")
