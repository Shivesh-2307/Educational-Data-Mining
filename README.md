# Student-AI Learning Outcome Research

This project asks whether the language students use while working with AI relates to their later math scores, and whether conversation patterns differ between GPT Base and GPT Tutor.

**Treatment labels:** `vanilla` = GPT Base; `aug` = GPT Tutor.

## Start here

1. Read [the essential research guide](docs/RESEARCH_GUIDE.md).
2. Open the notebooks in order: [01 - Text data](notebooks/01_text_features_and_data.ipynb), [02 - Profiles and treatment](notebooks/02_interaction_profiles_and_treatment.ipynb), [03 - Learning outcomes](notebooks/03_learning_outcomes.ipynb).
3. Open the [editable full project report](docs/Student_AI_Learning_Outcome_Research_Report.docx) for presentation material.
4. Use [the audit](docs/PROJECT_AUDIT.md) for data provenance and [the output catalog](docs/OUTPUT_CATALOG.md) to find saved files.

Notebook 01 runs the text pipeline. Notebook 02 displays those results. Notebook 03 runs the outcome analysis. The matching Python scripts are the source of the calculations and can also be run directly from the repository root:

```powershell
./.venv/Scripts/python.exe Research_Analysis/run_research_analysis.py
./.venv/Scripts/python.exe Research_Analysis/run_outcome_analysis.py
```

The first script must run before the second. Both reuse the existing processed data and save generated files under the single top-level `outputs/` folder. They do not modify the original processed datasets. Regenerate the DOCX report with `./.venv/Scripts/python.exe reports/build_project_report.py`.

## Project map

```text
ProcessedDatasets/                 original processed inputs
PreProcessing/                     original preprocessing notebook + full conversation export
AcademicOutcomes/                  exact-ID matched public score input
Research_Analysis/                 current analysis scripts
notebooks/                         three notebooks for the current workflow
reports/                          editable DOCX report and its builder
outputs/research/                  current derived datasets, tables, figures, and results
outputs/legacy_treatment_classifier/ saved older classifier results
archive/legacy_treatment_classifier/ older treatment classifier code and notebook
```

The archived `Student_LLM_Unified_Pipeline_SHAP.ipynb` is relevant only as a record of the earlier treatment-classification work. It predicts GPT Base versus GPT Tutor; it does not analyze exam scores. The current work is presented in the three notebooks above.
