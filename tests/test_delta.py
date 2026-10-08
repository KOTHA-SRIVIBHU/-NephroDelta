"""
Unit tests for delta analytics, percentage calculations, trend slopes, and analytical states.
"""

import pytest
import pandas as pd
import numpy as np

from src.delta_analysis import (
    calculate_safe_pct_change,
    calculate_patient_deltas,
    calculate_patient_trends,
    classify_analytical_state
)


def test_safe_percentage_change():
    assert calculate_safe_pct_change(110.0, 100.0) == 10.0
    assert calculate_safe_pct_change(80.0, 100.0) == -20.0
    # Zero denominator
    assert np.isnan(calculate_safe_pct_change(10.0, 0.0))
    # NaN inputs
    assert np.isnan(calculate_safe_pct_change(np.nan, 100.0))


def test_calculate_patient_deltas():
    raw = {
        "patient_id": ["P001", "P001", "P001"],
        "date": ["2025-01-01", "2025-06-01", "2026-01-01"],
        "egfr": [90.0, 86.0, 82.0],
        "uacr": [20.0, 30.0, 40.0],
        "creatinine": [1.0, 1.1, 1.2],
        "hba1c": [6.0, 6.2, 6.4],
        "systolic_bp": [120.0, 124.0, 128.0],
        "weight": [70.0, 70.0, 71.0]
    }
    df = pd.DataFrame(raw)
    df_deltas = calculate_patient_deltas(df)

    assert pd.isna(df_deltas.iloc[0]["delta_egfr"])
    assert df_deltas.iloc[1]["delta_egfr"] == -4.0
    assert df_deltas.iloc[2]["delta_egfr"] == -4.0

    assert df_deltas.iloc[1]["pct_change_uacr"] == pytest.approx(50.0)


def test_calculate_patient_trends():
    raw = {
        "patient_id": ["P001", "P001"],
        "date": ["2024-01-01", "2025-01-01"],  # Exactly 1 year duration
        "egfr": [100.0, 90.0],
        "uacr": [10.0, 30.0]
    }
    df = pd.DataFrame(raw)
    trends = calculate_patient_trends(df)

    assert trends["initial_egfr"] == 100.0
    assert trends["latest_egfr"] == 90.0
    assert trends["total_egfr_change"] == -10.0
    assert trends["egfr_slope"] == pytest.approx(-10.0, rel=1e-2)
    assert trends["uacr_slope"] == pytest.approx(20.0, rel=1e-2)


def test_single_observation_trends():
    raw = {
        "patient_id": ["P001"],
        "date": ["2025-01-01"],
        "egfr": [88.0],
        "uacr": [25.0]
    }
    df = pd.DataFrame(raw)
    trends = calculate_patient_trends(df)

    assert trends["total_egfr_change"] == 0.0
    assert trends["egfr_slope"] == 0.0
    assert trends["num_observations"] == 1


def test_classify_analytical_state():
    # State 0: Stable
    assert classify_analytical_state(95.0, 20.0) == 0
    # State 1: Early abnormality
    assert classify_analytical_state(75.0, 25.0) == 1
    assert classify_analytical_state(95.0, 50.0) == 1
    # State 2: Moderate impairment
    assert classify_analytical_state(45.0, 100.0) == 2
    assert classify_analytical_state(80.0, 400.0) == 2
    # State 3: Advanced impairment
    assert classify_analytical_state(25.0, 50.0) == 3
