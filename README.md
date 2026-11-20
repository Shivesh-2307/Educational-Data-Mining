# Data Mining Project

Educational Data Mining project analysing how student prompts to an LLM relate to learning.

## Regression models

`Regression.py`

- Linear Regression, Random Forest and Gradient Boosting trained for each experiment (E1, E2, E3), giving 9 model evaluations.
- student-level evaluation with 5-fold `GroupKFold` grouped on student ID, so no student appears in both training and validation folds. Metrics: MAE, RMSE, R².

Preprocessing (`log1p` + `StandardScaler`) is fitted inside the pipeline on training folds only.

### Usage

1. Set `TARGET` to the outcome column and `DATA_PATH` to the dataset.
2. Run the script (or paste the `# %%` cells into Colab).
3. Results are saved to `regression_results.csv`.
