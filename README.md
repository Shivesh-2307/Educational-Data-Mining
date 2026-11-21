# Student–LLM Interaction Analysis

This project analyzes how student interaction patterns relate to experimental treatment groups in an LLM-based educational study.

> The current data has a binary treatment target (`aug` vs. `vanilla`) but no independent academic outcome. Accordingly, this project predicts treatment and explains predictive associations; it does not establish that either treatment caused learning gains.

## Quick Start

### Interactive notebook

Open:

- `SHAP_Analysis/Student_LLM_Unified_Pipeline_SHAP.ipynb`

The notebook is organized into five stages: audit/EDA, clustering, E1–E3 classification, SHAP, and statistical auditing.

### Script

From the repository root:

```powershell
./.venv/Scripts/python.exe SHAP_Analysis/run_full_pipeline.py
```

## Repository Layout

```text
ProcessedDatasets/
  student_features_final.csv       518 student-level records
  cleaned_user_messages.csv        28,664 prompt-level records
  preprocessing_config.json        feature and preprocessing configuration

SHAP_Analysis/
  Student_LLM_Unified_Pipeline_SHAP.ipynb
  run_full_pipeline.py             script version of the notebook
  Research_Report.md               complete research report
  README.md                        package-specific instructions
  outputs/
    figures/                       generated visualizations
    tables/                        generated CSV evidence
    summary_metrics.json           run summary

docs/
  PROJECT_OVERVIEW.md              macro architecture and methodology
  NOTEBOOK_GUIDE.md                how to run and interpret notebook stages

Regression.py                      future/legacy continuous-outcome template
requirements.txt                   reproducible Python dependencies
.gitignore
```

## Research Pipeline

```text
raw prompt records
  -> prompt-level cleaning and feature engineering
  -> 518 student-level records
  -> data audit and EDA
  -> unsupervised clustering
  -> E1 / E2 / E3 classification
  -> SHAP explanation
  -> grade and statistical validation
```

## Feature Experiments

- **E1 — Primary:** nine core interaction features
- **E2 — Primary + sparse:** E1 plus rare prompt rates and flags
- **E3 — Combined:** E2 plus grade metadata

Preprocessing uses `log1p` and `StandardScaler`. Transformations are fitted inside cross-validation pipelines to prevent training-fold leakage.

## Environment

Create or reuse the local environment:

```powershell
C:/Python312/python.exe -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install ipykernel
```

VS Code/Jupyter uses `.venv/Scripts/python.exe` for notebook execution.

## Results Are Observational

The model distinguishes treatment groups strongly, but SHAP does not demonstrate causality and treatment classification is not a substitute for an academic outcome analysis.
