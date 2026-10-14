"""
Groq AI Explanation Module for NephroDelta.

Generates natural-language educational explanations of pre-computed analytical results
and retrieved RAG context using Groq's fast LLM inference API (default: openai/gpt-oss-120b).

SAFETY & COMPLIANCE:
- Pre-computed metrics only; LLM does not perform calculations.
- Strict system prompt enforcing non-diagnostic, educational role.
- Graceful handling when GROQ_API_KEY is not configured or API fails.
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# llama-3.3-70b-versatile is no longer served by Groq; override with GROQ_MODEL if needed.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_SAFETY_INSTRUCTION = (
    "You are an educational healthcare analytics assistant.\n"
    "You must not diagnose disease or provide treatment recommendations.\n"
    "Explain only the supplied analytical results and retrieved educational information.\n"
    "Clearly distinguish observed patient data, calculated changes, hypothetical simulation results, "
    "and general educational information.\n"
    "Never present simulation output as a clinical prediction.\n"
    "Do not invent measurements or medical facts.\n"
    "If information is insufficient, say so.\n"
    "This application is for research and educational purposes only."
)


def get_groq_api_key() -> Optional[str]:
    """Retrieves the GROQ_API_KEY from environment variables."""
    key = os.getenv("GROQ_API_KEY")
    if key and key.strip():
        return key.strip()
    return None


def generate_explanation(
    patient_id: str,
    latest_metrics: Dict[str, Any],
    trends: Dict[str, Any],
    deltas: Dict[str, Any],
    simulation_results: Dict[str, Any],
    analytical_state_name: str,
    rag_context: List[str]
) -> str:
    """
    Formulates a structured prompt from pre-computed analytics + RAG context,
    then queries Groq LLM for a natural-language educational explanation.

    Args:
        patient_id: Identifier of the patient.
        latest_metrics: Dictionary of latest measurement values.
        trends: Pre-computed trend slopes and total changes.
        deltas: Pre-computed delta changes between observations.
        simulation_results: Monte Carlo simulation summary (median, p10, p90).
        analytical_state_name: Descriptive label of current analytical state.
        rag_context: Retrieved text chunks from RAG vector search.

    Returns:
        String containing the AI explanation or a fallback status message.
    """
    api_key = get_groq_api_key()
    if not api_key:
        return "Groq API key not configured. Analytical features remain available."

    # Format retrieved educational context
    context_str = "\n".join([f"- {c}" for c in rag_context]) if rag_context else "No extra context retrieved."

    # Final median / percentile values
    median_final = simulation_results.get("median", [None])[-1]
    p10_final    = simulation_results.get("p10",    [None])[-1]
    p90_final    = simulation_results.get("p90",    [None])[-1]

    user_prompt = f"""
Please provide an educational explanation of the following pre-computed longitudinal analytics:

PATIENT IDENTIFIER: {patient_id}

LATEST OBSERVATIONAL MEASUREMENTS:
- eGFR: {latest_metrics.get('egfr')} mL/min/1.73m²
- UACR: {latest_metrics.get('uacr')} mg/g
- Creatinine: {latest_metrics.get('creatinine')} mg/dL
- HbA1c: {latest_metrics.get('hba1c')} %
- Blood Pressure: {latest_metrics.get('systolic_bp')}/{latest_metrics.get('diastolic_bp')} mmHg

PRE-COMPUTED OBSERVED LONGITUDINAL TRENDS & DELTAS:
- eGFR Total Change: {trends.get('total_egfr_change')} mL/min/1.73m² over {trends.get('time_duration_years')} years
- Observed eGFR Slope: {trends.get('egfr_slope')} mL/min/1.73m²/year
- UACR Total Change: {trends.get('total_uacr_change')} mg/g over {trends.get('time_duration_years')} years
- Observed UACR Slope: {trends.get('uacr_slope')} mg/g/year

SIMULATION ANALYTICAL STATE (NOT A DIAGNOSIS):
- Current State: {analytical_state_name}

STOCHASTIC MONTE CARLO SIMULATION RESULTS ({simulation_results.get('num_simulations')} runs, {simulation_results.get('time_horizon')} steps):
- Final Step Median eGFR:        {median_final} mL/min/1.73m²
- Final Step 10th Percentile:    {p10_final} mL/min/1.73m²
- Final Step 90th Percentile:    {p90_final} mL/min/1.73m²

RETRIEVED EDUCATIONAL KNOWLEDGE CONTEXT:
{context_str}

EXPLANATION STRUCTURE REQUIRED:
1. Observed Trends Summary
2. Delta & Metric Interpretation
3. Stochastic Simulation Interpretation
4. General Educational Context
5. Analytical Limitations & Safety Disclaimer
"""

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        # gpt-oss models spend part of the token budget on reasoning; keep it short.
        extra = {"reasoning_effort": "low"} if GROQ_MODEL.startswith("openai/gpt-oss") else {}
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_SAFETY_INSTRUCTION},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.2,
            max_completion_tokens=3000,
            **extra,
        )
        result = response.choices[0].message.content
        if result and result.strip():
            return result.strip()

    except Exception as e:
        print(f"Warning: Groq API call failed: {e}")
        return (
            "AI explanation unavailable because the Groq API call failed: "
            f"{type(e).__name__}: {e}. Analytical results are still available."
        )

    return "AI explanation unavailable. Analytical results are still available."
