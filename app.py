"""
NephroDelta - Streamlit Dashboard Application.

Longitudinal Kidney Health Delta Analytics with Stochastic Simulation and Generative AI.
Educational & Research Analytics Tool (Not a medical diagnostic system).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_generator import generate_synthetic_data
from src.data_processing import load_data, clean_data, sort_by_patient_and_date, validate_data
from src.delta_analysis import (
    calculate_patient_deltas,
    calculate_patient_trends,
    classify_analytical_state,
    get_state_description,
    STATE_LABELS
)
from src.simulation import run_monte_carlo_simulation
from src.rag import retrieve_context
from src.ai_explanation import generate_explanation, get_gemini_api_key

st.set_page_config(
    page_title="NephroDelta Analytics",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def get_processed_data() -> pd.DataFrame:
    """Loads, cleans, and computes deltas for the patient dataset."""
    data_path = "data/patients.csv"
    if not os.path.exists(data_path):
        generate_synthetic_data(output_path=data_path)

    try:
        raw_df = load_data(data_path)
        cleaned_df = clean_data(raw_df)
        sorted_df = sort_by_patient_and_date(cleaned_df)
        df_deltas = calculate_patient_deltas(sorted_df)
        return df_deltas
    except Exception as e:
        st.error(f"Error loading patient data: {e}")
        return pd.DataFrame()


def main():
    st.title("NephroDelta")
    st.caption("Longitudinal Kidney Health Delta Analytics with Stochastic Simulation and Generative AI")

    # Load dataset
    df = get_processed_data()

    if df.empty:
        st.warning("No patient data available. Please generate synthetic dataset.")
        if st.button("Generate Synthetic Data"):
            generate_synthetic_data()
            st.cache_data.clear()
            st.rerun()
        return

    patient_ids = sorted(df["patient_id"].unique().tolist())

    # Sidebar
    st.sidebar.header("Patient Selector & Controls")
    selected_patient = st.sidebar.selectbox("Select Patient ID", patient_ids, index=0)

    st.sidebar.markdown("---")
    st.sidebar.header("Stochastic Simulation Settings")
    num_simulations = st.sidebar.number_input(
        "Number of Simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )
    time_horizon = st.sidebar.slider(
        "Simulation Horizon (Time Steps)",
        min_value=3,
        max_value=36,
        value=12,
        step=1
    )

    # Filter data for selected patient
    patient_df = df[df["patient_id"] == selected_patient].sort_values("date").reset_index(drop=True)

    if patient_df.empty:
        st.error(f"Patient {selected_patient} not found.")
        return

    latest_row = patient_df.iloc[-1]
    patient_trends = calculate_patient_trends(patient_df)
    current_state = classify_analytical_state(latest_row["egfr"], latest_row["uacr"])

    # SECTION 1: Patient Summary Cards
    st.markdown("### Patient Profile & Latest Measurements")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Patient ID", str(latest_row["patient_id"]))
    c2.metric("Age", f"{int(latest_row['age'])} yrs")
    c3.metric("Sex", str(latest_row["sex"]))
    c4.metric("Latest eGFR", f"{latest_row['egfr']:.1f}", delta=f"{latest_row.get('delta_egfr', 0.0):+.1f}" if pd.notnull(latest_row.get('delta_egfr')) else None)
    c5.metric("Latest UACR", f"{latest_row['uacr']:.1f}", delta=f"{latest_row.get('delta_uacr', 0.0):+.1f}" if pd.notnull(latest_row.get('delta_uacr')) else None)
    c6.metric("Creatinine", f"{latest_row['creatinine']:.2f} mg/dL")
    c7.metric("Blood Pressure", f"{int(latest_row['systolic_bp'])}/{int(latest_row['diastolic_bp'])} mmHg")

    st.markdown("---")

    # SECTION 2: Longitudinal Trends
    st.subheader("Longitudinal Trends")
    t1, t2, t3 = st.tabs(["eGFR Trajectory", "UACR Trajectory", "Creatinine Trajectory"])

    with t1:
        fig_egfr = px.line(
            patient_df,
            x="date",
            y="egfr",
            markers=True,
            title=f"eGFR Over Time ({selected_patient})",
            labels={"date": "Observation Date", "egfr": "eGFR (mL/min/1.73m²)"}
        )
        fig_egfr.update_traces(line_color="#1f77b4", line_width=2.5)
        st.plotly_chart(fig_egfr, use_container_width=True)

    with t2:
        fig_uacr = px.line(
            patient_df,
            x="date",
            y="uacr",
            markers=True,
            title=f"UACR Over Time ({selected_patient})",
            labels={"date": "Observation Date", "uacr": "UACR (mg/g)"}
        )
        fig_uacr.update_traces(line_color="#ff7f0e", line_width=2.5)
        st.plotly_chart(fig_uacr, use_container_width=True)

    with t3:
        fig_creat = px.line(
            patient_df,
            x="date",
            y="creatinine",
            markers=True,
            title=f"Creatinine Over Time ({selected_patient})",
            labels={"date": "Observation Date", "creatinine": "Serum Creatinine (mg/dL)"}
        )
        fig_creat.update_traces(line_color="#2ca02c", line_width=2.5)
        st.plotly_chart(fig_creat, use_container_width=True)

    st.markdown("---")

    # SECTION 3: Delta Analytics
    st.subheader("Delta Analytics")
    st.info("Delta represents the change between consecutive observations.")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### Consecutive Metric Changes (Latest Observation Delta)")
        d_df = pd.DataFrame([
            {"Metric": "eGFR (mL/min/1.73m²)", "Latest Delta": latest_row.get("delta_egfr"), "% Change": latest_row.get("pct_change_egfr")},
            {"Metric": "UACR (mg/g)", "Latest Delta": latest_row.get("delta_uacr"), "% Change": latest_row.get("pct_change_uacr")},
            {"Metric": "Creatinine (mg/dL)", "Latest Delta": latest_row.get("delta_creatinine"), "% Change": latest_row.get("pct_change_creatinine")},
            {"Metric": "HbA1c (%)", "Latest Delta": latest_row.get("delta_hba1c"), "% Change": latest_row.get("pct_change_hba1c")},
            {"Metric": "Systolic BP (mmHg)", "Latest Delta": latest_row.get("delta_systolic_bp"), "% Change": latest_row.get("pct_change_systolic_bp")},
            {"Metric": "Weight (kg)", "Latest Delta": latest_row.get("delta_weight"), "% Change": latest_row.get("pct_change_weight")},
        ])
        st.dataframe(d_df.style.format({"Latest Delta": "{:+.2f}", "% Change": "{:+.2f}%"}, na_rep="N/A"), use_container_width=True)

    with col_d2:
        st.markdown("#### Observed Longitudinal Slopes")
        st.write(f"**Time Horizon:** {patient_trends['time_duration_years']} years ({patient_trends['num_observations']} observations)")
        st.metric("Observed eGFR Slope", f"{patient_trends['egfr_slope']:+.2f} mL/min/1.73m²/yr")
        st.metric("Observed UACR Slope", f"{patient_trends['uacr_slope']:+.2f} mg/g/yr")
        st.caption("Note: Slopes reflect calculated historical observations only, not medical prognosis.")

    st.markdown("---")

    # SECTION 4: Stochastic Simulation
    st.subheader("Stochastic Simulation")

    sim_res = run_monte_carlo_simulation(
        initial_state=current_state,
        initial_egfr=float(latest_row["egfr"]),
        num_simulations=num_simulations,
        time_horizon=time_horizon,
        seed=42
    )

    fig_sim = go.Figure()
    steps = sim_res["time_steps"]

    # 90th percentile
    fig_sim.add_trace(go.Scatter(
        x=steps,
        y=sim_res["p90"],
        mode="lines",
        name="90th Percentile",
        line=dict(color="rgba(31, 119, 180, 0.4)", dash="dash")
    ))

    # 10th percentile with fill
    fig_sim.add_trace(go.Scatter(
        x=steps,
        y=sim_res["p10"],
        mode="lines",
        name="10th Percentile",
        fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.15)",
        line=dict(color="rgba(31, 119, 180, 0.4)", dash="dash")
    ))

    # Median trajectory
    fig_sim.add_trace(go.Scatter(
        x=steps,
        y=sim_res["median"],
        mode="lines+markers",
        name="Median Trajectory",
        line=dict(color="#1f77b4", width=3)
    ))

    fig_sim.update_layout(
        title=f"Monte Carlo Simulated eGFR Trajectories ({num_simulations} Runs, {time_horizon} Future Steps)",
        xaxis_title="Future Time Steps",
        yaxis_title="Simulated eGFR (mL/min/1.73m²)",
        hovermode="x unified"
    )

    st.plotly_chart(fig_sim, use_container_width=True)
    st.warning("These trajectories represent hypothetical model-generated scenarios and should not be interpreted as predictions of an individual's clinical future.")

    st.markdown("---")

    # SECTION 5: Pathophysiology-Inspired Model
    st.subheader("Pathophysiology-Inspired Model")
    state_name = get_state_description(current_state)

    st.markdown(f"**Current Analytical State:** `State {current_state}: {state_name}`")

    # Visual State Flow
    cols_s = st.columns(4)
    state_colors = ["green", "blue", "orange", "red"]

    for idx, (s_code, s_label) in enumerate(STATE_LABELS.items()):
        with cols_s[idx]:
            if s_code == current_state:
                st.success(f"**[{s_label}]**\n\nCurrent Analytical Pattern")
            else:
                st.caption(f"{s_label}")

    st.caption("The states are simplified analytical abstractions created for simulation and are not clinical classifications.")

    st.markdown("---")

    # SECTION 6: Generative AI Explanation
    st.subheader("Generative AI Explanation")

    api_key_status = get_gemini_api_key()
    if not api_key_status:
        st.info("Gemini API key not configured. Analytical features remain available.")

    if st.button("Generate Explanation", type="primary"):
        with st.spinner("Retrieving educational context and generating AI explanation..."):
            query_str = f"eGFR slope {patient_trends['egfr_slope']} UACR slope {patient_trends['uacr_slope']} kidney function"
            rag_docs = retrieve_context(query_str, k=3)

            latest_dict = {
                "egfr": latest_row["egfr"],
                "uacr": latest_row["uacr"],
                "creatinine": latest_row["creatinine"],
                "hba1c": latest_row["hba1c"],
                "systolic_bp": latest_row["systolic_bp"],
                "diastolic_bp": latest_row["diastolic_bp"]
            }

            explanation_text = generate_explanation(
                patient_id=selected_patient,
                latest_metrics=latest_dict,
                trends=patient_trends,
                deltas={
                    "delta_egfr": latest_row.get("delta_egfr"),
                    "delta_uacr": latest_row.get("delta_uacr")
                },
                simulation_results=sim_res,
                analytical_state_name=state_name,
                rag_context=rag_docs
            )

            st.markdown(explanation_text)

    st.markdown("---")

    # SECTION 7: Safety Notice
    st.markdown("## Safety Notice")
    st.error(
        "**Research/Educational Use Only.**\n\n"
        "This application does not diagnose disease, predict individual clinical outcomes, "
        "or provide treatment recommendations."
    )


if __name__ == "__main__":
    main()
