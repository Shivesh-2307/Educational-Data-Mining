# Output Catalog

All generated files are now centralized in one place: `outputs/`. The original input files remain in `ProcessedDatasets/`, `PreProcessing/`, and `AcademicOutcomes/`.

```text
outputs/
??? research/                         current text and score analysis (43 files)
?   ??? datasets/                     derived prompt/student/session/problem/conversation data
?   ??? tables/                       model, cluster, topic, treatment, and review tables
?   ??? figures/                      current text/profile figures
?   ??? outcomes/                     score-linked data, outcome tables, figures, and summary
??? legacy_treatment_classifier/      archived earlier analysis (25 files)
    ??? tables/
    ??? figures/
    ??? summary_metrics.json
```

There are many CSVs because each one records a different **row level** or a different research question. A prompt file has one row per message; session and problem files aggregate within those units; treatment/cluster/model tables answer separate comparisons. Keeping these separate avoids mixing messages with students or score predictions. They are generated artifacts, not extra copies of the original processed inputs.

## Start with these current outputs

| Path under `outputs/research/` | Use |
|---|---|
| `research_run_summary.json` | Input checks, treatment mapping, clustering summary, and run status. |
| `datasets/message_text_features.csv` | Original student prompt plus newly extracted wording clues. |
| `datasets/student_session_features.csv` | One row per student-session; this is the score-join input. |
| `datasets/session_outcomes_with_chat_features.csv` | Exact student-session chat/score matches. |
| `tables/cluster_feature_profiles.csv` | Interpretable averages for the interaction clusters. |
| `tables/class_clustered_treatment_text_associations.csv` | Classroom-adjusted text-feature comparisons by arm. |
| `outcomes/tables/outcome_grouped_cv_results.csv` | Classroom-held-out exam/practice model metrics. |
| `outcomes/tables/exam_incremental_feature_value.csv` | Tests whether text/behavior features improve Ridge exam prediction over context. |
| `outcomes/tables/treatment_arm_score_contrasts.csv` | Adjusted GPT Tutor minus GPT Base score contrasts. |
| `outcomes/tables/exam_combined_model_shap.csv` | SHAP feature importance for one score prediction model. |

Current plots are in `research/figures/` and `research/outcomes/figures/`. The 25 files in `legacy_treatment_classifier/` belong to the earlier model that predicts GPT Tutor vs GPT Base; they are retained as an archive and are not exam-score findings.

## Editable project report

`reports/Student_AI_Learning_Outcome_Research_Report.docx` is the presentation-ready editable report. It embeds all 20 generated figures, presents the main statistical/model tables, and inventories all 45 generated CSVs. Regenerate it with `./.venv/Scripts/python.exe reports/build_project_report.py`.
