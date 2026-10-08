"""
Data Processing Module for NephroDelta.

Provides data loading, cleaning, median imputation, validation, and sorting
functions for longitudinal patient datasets.
"""

import os
from typing import List
import pandas as pd
import numpy as np

REQUIRED_COLUMNS: List[str] = [
    "patient_id",
    "date",
    "age",
    "sex",
    "glucose",
    "hba1c",
    "systolic_bp",
    "diastolic_bp",
    "creatinine",
    "egfr",
    "uacr",
    "weight"
]

NUMERICAL_COLUMNS: List[str] = [
    "age",
    "glucose",
    "hba1c",
    "systolic_bp",
    "diastolic_bp",
    "creatinine",
    "egfr",
    "uacr",
    "weight"
]


def load_data(filepath: str = "data/patients.csv") -> pd.DataFrame:
    """
    Loads raw CSV patient data from file.

    Args:
        filepath: Absolute or relative path to CSV file.

    Returns:
        pd.DataFrame containing raw patient observations.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Patient data file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validates that the input DataFrame contains all required columns and non-empty data.

    Args:
        df: Input pandas DataFrame.

    Returns:
        True if valid.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset by converting datatypes, dropping duplicates,
    and performing median imputation for missing numerical values.

    Args:
        df: Raw or loaded pandas DataFrame.

    Returns:
        Cleaned pd.DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df_cleaned = df.copy()

    # Drop duplicate rows if any
    df_cleaned = df_cleaned.drop_duplicates()

    # Validate schema
    validate_data(df_cleaned)

    # Convert datatypes
    df_cleaned["patient_id"] = df_cleaned["patient_id"].astype(str)
    df_cleaned["sex"] = df_cleaned["sex"].astype(str)
    df_cleaned["date"] = pd.to_datetime(df_cleaned["date"], errors="coerce")

    for col in NUMERICAL_COLUMNS:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

    # Median imputation: first try per-patient median, fallback to global column median
    for col in NUMERICAL_COLUMNS:
        if df_cleaned[col].isnull().any():
            # Per-patient median imputation
            patient_medians = df_cleaned.groupby("patient_id")[col].transform("median")
            df_cleaned[col] = df_cleaned[col].fillna(patient_medians)
            
            # Global median fallback if all values for a patient are null
            global_median = df_cleaned[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            df_cleaned[col] = df_cleaned[col].fillna(global_median)

    return df_cleaned


def sort_by_patient_and_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts DataFrame by patient_id and date in ascending order.

    Args:
        df: Input pd.DataFrame.

    Returns:
        Sorted pd.DataFrame with reset index.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df_sorted = df.sort_values(by=["patient_id", "date"], ascending=[True, True])
    return df_sorted.reset_index(drop=True)
