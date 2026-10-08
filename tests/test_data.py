"""
Unit tests for data generation and data processing modules in NephroDelta.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_generator import generate_synthetic_data
from src.data_processing import (
    load_data,
    clean_data,
    validate_data,
    sort_by_patient_and_date,
    REQUIRED_COLUMNS
)


def test_synthetic_data_generation(tmp_path):
    output_csv = os.path.join(tmp_path, "patients.csv")
    df = generate_synthetic_data(num_patients=10, obs_per_patient=5, seed=42, output_path=output_csv)

    assert os.path.exists(output_csv)
    assert len(df) == 50  # 10 * 5
    assert df["patient_id"].nunique() == 10
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_data_validation():
    # Valid dataframe
    df_valid = pd.DataFrame(columns=REQUIRED_COLUMNS)
    df_valid.loc[0] = ["P001", "2025-01-01", 50, "M", 100, 6.0, 120, 80, 1.0, 90, 15, 75]
    assert validate_data(df_valid) is True

    # Empty dataframe
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_data(pd.DataFrame())

    # Missing required column
    df_missing = df_valid.drop(columns=["egfr"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df_missing)


def test_data_cleaning_and_imputation():
    raw_data = {
        "patient_id": ["P001", "P001", "P002"],
        "date": ["2025-01-01", "2025-06-01", "2025-01-01"],
        "age": [50, 50, 60],
        "sex": ["M", "M", "F"],
        "glucose": [110, np.nan, 130],
        "hba1c": [6.5, 6.7, np.nan],
        "systolic_bp": [120, 125, 140],
        "diastolic_bp": [80, 82, 88],
        "creatinine": [1.0, 1.1, 1.5],
        "egfr": [90, 85, 45],
        "uacr": [20, 25, 150],
        "weight": [80.0, 80.5, 65.0]
    }
    df_raw = pd.DataFrame(raw_data)
    df_cleaned = clean_data(df_raw)

    # Check missing values are imputed
    assert df_cleaned["glucose"].isnull().sum() == 0
    assert df_cleaned["hba1c"].isnull().sum() == 0
    # Patient P001 glucose NaN should be imputed with P001's median (110)
    assert df_cleaned.loc[df_cleaned["patient_id"] == "P001", "glucose"].iloc[1] == 110.0


def test_sort_by_patient_and_date():
    raw_data = {
        "patient_id": ["P002", "P001", "P001"],
        "date": ["2025-01-01", "2025-06-01", "2025-01-01"],
        "age": [60, 50, 50],
        "sex": ["F", "M", "M"],
        "glucose": [130, 110, 105],
        "hba1c": [6.5, 6.7, 6.4],
        "systolic_bp": [140, 125, 120],
        "diastolic_bp": [88, 82, 80],
        "creatinine": [1.5, 1.1, 1.0],
        "egfr": [45, 85, 90],
        "uacr": [150, 25, 20],
        "weight": [65.0, 80.5, 80.0]
    }
    df = pd.DataFrame(raw_data)
    df_sorted = sort_by_patient_and_date(df)

    assert df_sorted.iloc[0]["patient_id"] == "P001"
    d0 = df_sorted.iloc[0]["date"]
    date_str0 = str(d0.date()) if hasattr(d0, "date") else str(d0)[:10]
    assert date_str0 == "2025-01-01"
    assert df_sorted.iloc[1]["patient_id"] == "P001"
    d1 = df_sorted.iloc[1]["date"]
    date_str1 = str(d1.date()) if hasattr(d1, "date") else str(d1)[:10]
    assert date_str1 == "2025-06-01"
