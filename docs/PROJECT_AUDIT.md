# Project Audit

Audit updated: 2026-11-22. Paths below are relative to the repository root.

## Repository inventory and status

| Path | Purpose and observed state |
|---|---|
| `PreProcessing/PreProcessing.ipynb` | Existing 95-cell preprocessing notebook. It cleans conversation records, filters user messages, defines rule labels and aggregates student features. It contains Colab-specific upload/download cells and hardcoded `/content/...` paths. Preserve it as the source of the processed inputs. |
| `PreProcessing/conversations_audit.csv` | Full 64,486 × 10 conversation export; includes `system`, `user`, and `assistant` rows. This is the only assistant-response source found. |
| `ProcessedDatasets/cleaned_user_messages.csv` | 28,664 × 23, student-message-only analysis table. |
| `ProcessedDatasets/student_features_final.csv` | 518 × 33, canonical one-row-per-student features. |
| `ProcessedDatasets/preprocessing_config.json` | Source of truth for the existing nine primary, three sparse rate, and three sparse binary features; `log1p`, `StandardScaler`, `username` and `treatment` definitions. Read as part of the added pipeline. |
| `AcademicOutcomes/matched_session_outcomes.csv` | 1,820 × 13 public study score rows for the 518 local student IDs, keyed by `username`, `session_id`; only 1,735 rows have same-session chat. See `AcademicOutcomes/README.md`. |
| `archive/legacy_treatment_classifier/Student_LLM_Unified_Pipeline_SHAP.ipynb` | Archived 15-cell notebook for treatment classification and SHAP; it does not model an academic outcome. |
| `notebooks/01_text_features_and_data.ipynb`, `02_interaction_profiles_and_treatment.ipynb`, `03_learning_outcomes.ipynb` | Current workflow notebooks; they follow the current Python scripts and read/write the centralized outputs. |
| `reports/Student_AI_Learning_Outcome_Research_Report.docx` | Editable full report with current results, all generated figures, selected result tables, and a complete CSV inventory. Rebuild with `reports/build_project_report.py`. |
| `archive/legacy_treatment_classifier/run_full_pipeline.py` | Script version of the existing analysis. |
| `outputs/legacy_treatment_classifier/` | Archived tables, figures and `summary_metrics.json` for treatment prediction and interaction profiles, not learning-outcome results. |
| `archive/legacy_treatment_classifier/Regression.py` | Incomplete historical template, not used by the current pipeline; it requires a real numeric target. |
| `docs/RESEARCH_GUIDE.md`, `docs/OUTPUT_CATALOG.md` | Essential research explanation and generated-output index. |

The local repository inventory itself contained no score table. After checking the paper link supplied for this research, the authors' public repository was found to contain outcomes and transcript data. A matched score subset is stored separately as `AcademicOutcomes/matched_session_outcomes.csv`. This corrects the initial local-only conclusion.

## Processed input validation

### `ProcessedDatasets/cleaned_user_messages.csv`

- Shape: **28,664 rows × 23 columns**; duplicate full rows: **0**.
- Dtypes: `unnamed:_0` int64; `username`, `grade`, `problem_id`, `session_id`, `prompt_length`, `word_count`, `sentence_count`, `question_count`, and all nine binary labels int64; `message`, `role`, `conversation_id`, `time_stamp`, and `treatment` strings.
- Columns in source order: `unnamed:_0`, `role`, `message`, `conversation_id`, `username`, `grade`, `problem_id`, `session_id`, `time_stamp`, `treatment`, `prompt_length`, `word_count`, `sentence_count`, `question_count`, `explanation_request`, `hint_request`, `complete_solution_request`, `verification_request`, `conceptual_question`, `refinement_request`, `has_multiple_questions`, `has_code`, `has_followup_language`.
- The nine integer binary label columns are the nine named fields from `explanation_request` through `has_followup_language` in the list above; the first 14 columns have the dtypes described above.
- Missingness: `time_stamp` has **23,215 missing** values; all other listed columns have no missing values.
- Values/cardinality: role has one value (`user`); 518 usernames, 7,089 conversation IDs, four sessions, five problem IDs, three grades, and two treatments. Treatment counts are `aug` 20,390 and `vanilla` 8,274 at message level.
- Identifiers and keys: `username` links to the student table. `conversation_id` links turns to the raw conversation export. `session_id` and `problem_id` support within-student aggregation. `unnamed:_0` is retained as a source row/order field. Timestamps are not complete enough to establish ordering for every message.
- Text: `message` is the cleaned student prompt. This file does **not** contain AI responses.

### `ProcessedDatasets/student_features_final.csv`

- Shape: **518 × 33**; duplicate full rows: **0**; missing values: **0**.
- Dtypes: `username`, counts, grade, treatment consistency, conversation/session/problem counts, and sparse presence flags are integers except `treatment` (string); averages and rates are float64.
- Columns in source order: `username`, `student_message_count`, `average_prompt_length`, `average_word_count`, `average_sentence_count`, `average_question_count`, `explanation_request_count`, `hint_request_count`, `complete_solution_request_count`, `verification_request_count`, `conceptual_question_count`, `refinement_request_count`, `multiple_question_count`, `code_prompt_count`, `followup_prompt_count`, `grade`, `treatment`, `unique_treatments`, `conversation_count`, `session_count`, `problem_count`, `explanation_request_rate`, `hint_request_rate`, `complete_solution_request_rate`, `verification_request_rate`, `conceptual_question_rate`, `refinement_request_rate`, `multiple_question_rate`, `code_prompt_rate`, `followup_prompt_rate`, `hint_request_present`, `refinement_request_present`, `multiple_question_present`.
- Identifiers and cardinality: `username` is unique in all 518 rows. Treatment counts are `aug` 277 and `vanilla` 241; grades 9, 10 and 11 are represented.
- It is the canonical existing student-level table; new text features are added in a separate derived file and validated with a one-to-one `username` merge.

### `PreProcessing/conversations_audit.csv`

- Shape: **64,486 × 10**; duplicate full rows: **0**.
- Columns: `Unnamed: 0` int64, `role` string, `message` string, `conversation_id` string, `username` int64, `grade` int64, `problem_id` int64, `session_id` int64, `time_stamp` string, `treatment` string.
- Missingness: `time_stamp` has **9,949** missing values; other columns have no missing values.
- Roles: assistant 28,731; user 28,666; system 7,089. Contains the complete role stream needed for turn adjacency and AI response analysis. The two additional raw user rows versus the processed student table are preserved in the raw source and not silently added to processed analyses.
- Identifier cardinalities: username 518; conversation ID 7,089; session 4; problem 5; treatment 2; grade 3. Use `conversation_id` for turn context and `username` for validated student joins.

## Public academic outcome data

`AcademicOutcomes/matched_session_outcomes.csv` is a filtered subset of the study authors' public `main_regressions/final_data.csv` (repository commit `2f63dae1a01d51453826fe07ef5cf6678e339588`). It has **1,820 rows × 13 columns**, 518 exact student ID matches, 27 classes, and four sessions. The exact modeling key is (`username`, `session_id`) = (`Student_ID`, `Session`). There are no duplicate session keys; `Part2Tot` and `Part3Tot` have no missing values; `gpa_prev` is missing in 36 rows and `Grader` once.

All 518 local students appear in the outcome file. Of its 1,820 score sessions, 1,735 have same-session chat; 85 do not. Of 1,771 local chat sessions, 36 have no score row. Treatment mappings are checked against the published indicators: `vanilla` = GPT Base and `aug` = GPT Tutor. This local transcript subset has no control-arm students. The authors' source and score definitions are described in `AcademicOutcomes/README.md`.

## Existing feature definitions

`preprocessing_config.json` defines the primary feature set: `student_message_count`, `average_word_count`, `average_question_count`, `explanation_request_rate`, `complete_solution_request_rate`, `verification_request_rate`, `conceptual_question_rate`, `code_prompt_rate`, and `followup_prompt_rate`. Supplementary rates are hint, refinement, and multiple-question rates; supplementary flags are the corresponding `*_present` indicators. Metadata are grade and conversation/session/problem counts. The added analysis honors the saved primary set and reports any additional text features separately; it does not rewrite this config or relabel old fields.

## Existing completed and incomplete work

- Completed: original cleaned student prompts and engineered rule labels; one-row-per-student aggregates; primary/supplementary feature config; existing clustering and quality metrics; treatment-group classifier comparisons; SHAP for treatment prediction; treatment/grade descriptive tests; saved figures, tables, report and summary JSON.
- Existing completed task: `archive/legacy_treatment_classifier/` predicts treatment assignment and describes student interaction features. This is not academic outcome prediction.
- Newly completed task: exact numeric ID linkage to study-author outcomes for all 518 local students; 1,735 same-session chat/score rows across 27 classes are eligible for analysis. The 85 score rows without matching same-session chat and 36 chat sessions without scores are explicitly excluded from text-linked analysis.
- Remaining limitations: the matched conversation cohort has only GPT Base and GPT Tutor, no control arm; automated text labels are not independently validated gold labels; the simple local NLP does not replace the richer message analyses in the paper's repository. Existing preprocessing notebooks retain Colab assumptions, while the added scripts run from the repository.
- Newly supported by source data: assistant response and adjacent-turn analysis, because the raw export contains assistant turns.
- Not supported: learning-gain measurement beyond the available exam score, or causal effects of individual text behaviors. The exam outcome is now available, but the prompt-feature slopes remain observational.
- Cleanup: removed the added compatibility-only outcome wrapper and an unclustered HC3 table/code that duplicated the classroom-clustered treatment analysis. The original preprocessing notebook, legacy SHAP pipeline, `archive/legacy_treatment_classifier/Regression.py`, processed files, and research outputs are retained.

## Implemented research pipeline and next research steps

The added pipeline derives message-level text features before aggregation, writes student/session/problem/conversation datasets, analyzes actual assistant turns from the raw export, evaluates profiles and treatment differences, and joins same-session outcomes using validated exact keys. It compares five regressors with classroom-held-out validation, class-clustered treatment contrasts, corrected treatment interactions, a classroom bootstrap, and held-out-fold SHAP. All current artifacts are saved under `outputs/research/`.

Further work should focus on independently validating annotations, reproducing/comparing the paper's richer NLP features, and testing the selected text/outcome patterns on an independent cohort. Do not interpret the current text coefficients as causal learning effects. The current scripts are under `Research_Analysis/`; notebooks are under `notebooks/` and outputs under `outputs/`.
