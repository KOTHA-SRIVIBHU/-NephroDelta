"""
Gemini AI Explanation Module for NephroDelta.

Generates natural-language educational explanations of pre-computed analytical results
and retrieved RAG context using Google Gemini API.

SAFETY & COMPLIANCE:
- Pre-computed metrics only; LLM does not perform calculations.
- Strict system prompt enforcing non-diagnostic, educational role.
- Graceful handling when GEMINI_API_KEY is not configured or API fails.
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

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


def get_gemini_api_key() -> Optional[str]:
    """Retrieves the GEMINI_API_KEY from environment variables."""
    key = os.getenv("GEMINI_API_KEY")
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
    Formulates prompt from pre-computed metrics and retrieved RAG context,
    then queries Google Gemini for natural-language explanation.

    Args:
        patient_id: Identifier of the patient.
        latest_metrics: Dictionary of latest measurement values.
        trends: Pre-computed trend slopes and total changes.
        deltas: Pre-computed delta changes between observations.
        simulation_results: Monte Carlo simulation summary (median, p10, p90).
        analytical_state_name: Descriptive label of current analytical state.
        rag_context: Retrieved text chunks from RAG vector search.

    Returns:
        String containing Gemini AI explanation or standard status fallback.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini API key not configured. Analytical features remain available."

    # Format structured context into prompt
    context_str = "\n".join([f"- {c}" for c in rag_context]) if rag_context else "No extra context retrieved."

    prompt = f"""
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
- Final Step Median eGFR: {simulation_results.get('median', [])[-1] if simulation_results.get('median') else 'N/A'} mL/min/1.73m²
- Final Step 10th Percentile: {simulation_results.get('p10', [])[-1] if simulation_results.get('p10') else 'N/A'} mL/min/1.73m²
- Final Step 90th Percentile: {simulation_results.get('p90', [])[-1] if simulation_results.get('p90') else 'N/A'} mL/min/1.73m²

RETRIEVED EDUCATIONAL KNOWLEDGE CONTEXT:
{context_str}

EXPLANATION STRUCTURE REQUIRED:
1. Observed Trends Summary (Explain observed eGFR/UACR changes and slopes)
2. Delta & Metric Interpretation (Explain what eGFR, UACR, and creatinine deltas generally indicate)
3. Stochastic Simulation Interpretation (Explain simulation median and percentile uncertainty bands)
4. General Educational Context (Synthesize the retrieved educational concepts)
5. Analytical Limitations & Safety Disclaimer (Emphasize educational/research scope and absence of clinical prediction)
"""

    # Attempt calling Gemini using available SDKs
    try:
        # Try new google-genai SDK first
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_SAFETY_INSTRUCTION,
                temperature=0.2
            )
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e1:
        # Fallback to google-generativeai legacy SDK if installed
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            model = genai_legacy.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=SYSTEM_SAFETY_INSTRUCTION
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e2:
            print(f"Warning: Gemini API call failed: {e1} / {e2}")

    return "AI explanation unavailable. Analytical results are still available."
