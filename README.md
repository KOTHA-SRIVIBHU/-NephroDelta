# NephroDelta

**Title:** *NephroDelta: Longitudinal Kidney Health Delta Analytics with Stochastic Simulation and Generative AI*

---

## 📋 Overview

**NephroDelta** is a local-only research and educational healthcare analytics application designed to inspect longitudinal kidney-health measurements, compute delta changes and longitudinal slopes, simulate hypothetical stochastic trajectories via Monte Carlo transition models, retrieve context with local Retrieval-Augmented Generation (RAG), and generate natural language summaries using Google Gemini.

> ⚠️ **MEDICAL & RESEARCH DISCLAIMER**
> **This application is built exclusively for educational and research analytics purposes.**
> It does NOT diagnose disease, prescribe medication, recommend medical treatment, or claim to predict an individual patient's medical future. All simulation outputs represent theoretical, model-generated scenarios.

---

## ✨ Features

1. **Synthetic Data Generator:** Generates correlated 100-patient, 10-observation longitudinal datasets (`data/patients.csv`) with explicit synthetic labeling and realistic physiological correlations.
2. **Data Cleaning & Imputation:** Handles missing values via median imputation, verifies datatypes, and sorts patient timelines accurately.
3. **Delta Analytics & Slope Trends:** Calculates consecutive observation deltas ($\Delta eGFR$, $\Delta UACR$, $\Delta Creatinine$, $\Delta HbA1c$, $\Delta BP$, $\Delta Weight$) and observed longitudinal slopes ($mL/min/1.73m^2/yr$).
4. **Pathophysiology-Inspired Model:** Categorizes observations into analytical simulation patterns (Stable, Early Abnormality, Moderate Impairment, Advanced Impairment).
5. **Monte Carlo Stochastic Simulation:** Runs Markov chain transition simulations across customizable horizons (e.g. 1000 runs, 12 steps) and renders median, 10th, and 90th percentile bands.
6. **Local RAG Retrieval:** Uses SentenceTransformers and FAISS to index and retrieve educational text passages from local knowledge files.
7. **Gemini AI Explanation:** Sends pre-computed analytics and retrieved context to Google Gemini for structured, non-diagnostic natural-language summaries.
8. **Interactive Streamlit Dashboard:** Complete visualization suite with metric cards and interactive Plotly charts.

---

## 🏗️ Simple Architecture

```
                    USER
                     │
                     ▼
             STREAMLIT DASHBOARD (app.py)
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Analytics   Simulation      RAG
        │            │            │
        └────────────┼────────────┘
                     ▼
                 Gemini AI
                     │
                     ▼
              Explanation
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.11 or Python 3.12 installed on Linux/macOS/Windows.

### 2. Environment Setup

Clone or place the project files in your directory, then create a virtual environment:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate

# Windows PowerShell / CMD:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuration & API Key Setup

NephroDelta functions completely without an API key for all data processing, delta calculations, plots, and simulations.

To enable optional Gemini AI natural language explanations:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and set your API key:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

*If no API key is provided, the dashboard will gracefully inform you that AI explanation is unavailable while all analytical features remain fully accessible.*

---

## 📊 Usage Guide

### 1. Generate Synthetic Patient Data
```bash
python -m src.data_generator
```
This produces `data/patients.csv` with 1,000 synthetic observations across 100 patients.

### 2. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 3. Run Test Suite
```bash
pytest -q
```

---

## 🧪 Project Structure

```
nephrodelta/
│
├── app.py                   # Streamlit interactive dashboard UI
├── requirements.txt         # Core Python dependencies
├── README.md                # Documentation & guide
├── .env.example             # Environment key template
├── .gitignore               # Git exclusions
│
├── data/
│   ├── patients.csv         # Synthetic longitudinal dataset
│   └── knowledge/
│       ├── kidney_health.txt # Educational kidney physiology notes
│       └── nephropathy.txt   # Educational nephropathy notes
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py    # Synthetic correlated data generator
│   ├── data_processing.py   # Load, validate, clean, & impute data
│   ├── delta_analysis.py    # Delta calculations & state classifier
│   ├── simulation.py        # Markov chain & Monte Carlo engine
│   ├── rag.py               # Local SentenceTransformer + FAISS RAG
│   └── ai_explanation.py    # Gemini API wrapper with safety prompt
│
└── tests/
    ├── test_data.py         # Unit tests for data loading & cleaning
    ├── test_delta.py        # Unit tests for delta analytics & slopes
    └── test_simulation.py   # Unit tests for Markov model & simulation
```

---

## 🔒 Limitations & Medical Disclaimer

- **Purely Synthetic Data:** The dataset is synthetically generated using random seed 42 for research demonstration.
- **Model Trajectories Are Not Clinical Predictions:** Stochastic percentiles reflect mathematical Markov simulations under hypothetical parameters.
- **Non-Diagnostic:** This tool must never be used for diagnosis, medical treatment, or patient management.
