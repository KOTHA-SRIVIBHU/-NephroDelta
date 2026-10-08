"""
Synthetic Data Generator for NephroDelta.

IMPORTANT NOTICE:
This module generates EXPLICITLY SYNTHETIC longitudinal health data for educational
and research analytics purposes ONLY. The values produced do NOT represent real patient
physiology, clinical diagnosis, or medical outcomes.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_data(
    num_patients: int = 100,
    obs_per_patient: int = 10,
    seed: int = 42,
    output_path: str = "data/patients.csv"
) -> pd.DataFrame:
    """
    Generates correlated longitudinal synthetic data for NephroDelta.

    Args:
        num_patients: Number of synthetic patients to generate (default: 100).
        obs_per_patient: Number of longitudinal observations per patient (default: 10).
        seed: Random seed for reproducibility (default: 42).
        output_path: File path to save the generated CSV.

    Returns:
        pd.DataFrame containing the 1000 synthetic patient observations.
    """
    np.random.seed(seed)

    records = []
    base_date = pd.Timestamp("2021-01-01")

    for i in range(1, num_patients + 1):
        patient_id = f"P{i:03d}"
        base_age = int(np.random.randint(35, 78))
        sex = np.random.choice(["M", "F"], p=[0.52, 0.48])
        base_weight = np.round(np.random.normal(78, 12 if sex == "M" else 10), 1)
        base_weight = float(np.clip(base_weight, 50.0, 130.0))

        # Baseline physiological parameters
        # Baseline eGFR between 45 and 115 mL/min/1.73m2
        base_egfr = np.random.uniform(45.0, 115.0)
        # Progression trajectory slope per visit (-2.0 to +0.5 mL/min)
        egfr_drift = np.random.normal(-0.8, 0.6)

        # Baseline HbA1c (5.2 - 9.5 %)
        base_hba1c = np.random.uniform(5.2, 9.5)
        hba1c_drift = np.random.normal(0.05, 0.1)

        # Baseline Systolic BP (115 - 155 mmHg)
        base_sys_bp = np.random.uniform(115.0, 155.0)

        for obs in range(obs_per_patient):
            # Time progression in 6-month steps
            obs_date = base_date + pd.DateOffset(months=obs * 6)
            obs_date_str = obs_date.strftime("%Y-%m-%d")

            # Age increases slightly over time
            age = base_age + (obs * 6) // 12

            # eGFR with correlated longitudinal drift and slight random noise
            egfr_val = base_egfr + (obs * egfr_drift) + np.random.normal(0, 1.5)
            egfr_val = float(np.clip(egfr_val, 10.0, 130.0))

            # Serum Creatinine is inversely related to eGFR (approx CKD-EPI formula relationship)
            # creatinine approx 80 / eGFR for M, 65 / eGFR for F
            base_creat_factor = 85.0 if sex == "M" else 70.0
            creatinine_val = (base_creat_factor / egfr_val) + np.random.normal(0, 0.05)
            creatinine_val = float(np.clip(creatinine_val, 0.5, 6.5))

            # UACR is higher when eGFR is lower and HbA1c is higher
            base_uacr = 15.0 + (100.0 - egfr_val) * 1.2 + (base_hba1c - 5.5) * 15.0
            uacr_val = max(5.0, base_uacr + (obs * np.random.uniform(1.0, 5.0)) + np.random.normal(0, 5.0))
            uacr_val = float(np.clip(uacr_val, 5.0, 2000.0))

            # HbA1c and Glucose correlation
            hba1c_val = base_hba1c + (obs * hba1c_drift * 0.1) + np.random.normal(0, 0.15)
            hba1c_val = float(np.clip(hba1c_val, 4.5, 13.0))
            # Estimated average glucose (eAG) = 28.7 * HbA1c - 46.7
            glucose_val = (28.7 * hba1c_val - 46.7) + np.random.normal(0, 10.0)
            glucose_val = float(np.clip(glucose_val, 70.0, 350.0))

            # Blood pressure
            sys_bp = base_sys_bp + np.random.normal(0, 4.0)
            dia_bp = (sys_bp * 0.62) + np.random.normal(0, 3.0)

            # Weight fluctuates slightly
            weight_val = base_weight + np.random.normal(0, 0.8)

            records.append({
                "patient_id": patient_id,
                "date": obs_date_str,
                "age": int(age),
                "sex": sex,
                "glucose": round(glucose_val, 1),
                "hba1c": round(hba1c_val, 2),
                "systolic_bp": round(sys_bp, 1),
                "diastolic_bp": round(dia_bp, 1),
                "creatinine": round(creatinine_val, 2),
                "egfr": round(egfr_val, 1),
                "uacr": round(uacr_val, 1),
                "weight": round(weight_val, 1)
            })

    df = pd.DataFrame(records)

    # Introduce a small controlled fraction of missing values (~2% randomly in selective columns)
    num_rows = len(df)
    missing_indices = np.random.choice(num_rows, size=int(num_rows * 0.02), replace=False)
    for idx in missing_indices:
        col = np.random.choice(["glucose", "hba1c", "uacr", "weight"])
        df.loc[idx, col] = np.nan

    # Ensure output directory exists and save CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Synthetic dataset generated successfully at data/patients.csv with shape {df.shape}")
