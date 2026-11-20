
# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
DATA_PATH = "ProcessedDatasets/student_features_final.csv"
GROUP_COL = "username"
TARGET = None  # set to your outcome column

primary = [
    "student_message_count", "average_word_count", "average_question_count",
    "explanation_request_rate", "complete_solution_request_rate",
    "verification_request_rate", "conceptual_question_rate",
    "code_prompt_rate", "followup_prompt_rate",
]
sparse_rates = ["hint_request_rate", "refinement_request_rate", "multiple_question_rate"]
sparse_flags = ["hint_request_present", "refinement_request_present", "multiple_question_present"]

experiments = {
    "E1": {"num": primary, "bin": [], "cat": []},
    "E2": {"num": primary + sparse_rates, "bin": sparse_flags, "cat": []},
    "E3": {"num": primary + sparse_rates, "bin": sparse_flags, "cat": ["grade"]},
}

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
}

# %%
df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET])
print(df.shape, "| students:", df[GROUP_COL].nunique())

# %%
def make_preprocessor(exp):
    steps = [("num", Pipeline([("log", FunctionTransformer(np.log1p)),
                               ("scale", StandardScaler())]), exp["num"])]
    if exp["bin"]:
        steps.append(("bin", "passthrough", exp["bin"]))
    if exp["cat"]:
        steps.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), exp["cat"]))
    return ColumnTransformer(steps)

# %%
def group_cv(pipe, X, y, groups, k=5):
    scores = []
    for train, val in GroupKFold(n_splits=k).split(X, y, groups):
        # make sure no student is in both train and validation
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[val])
        pipe.fit(X.iloc[train], y.iloc[train])
        pred = pipe.predict(X.iloc[val])
        scores.append({
            "MAE": mean_absolute_error(y.iloc[val], pred),
            "RMSE": np.sqrt(mean_squared_error(y.iloc[val], pred)),
            "R2": r2_score(y.iloc[val], pred),
        })
    return pd.DataFrame(scores)

# %%
y, groups = df[TARGET], df[GROUP_COL]
rows = []

for exp_name, exp in experiments.items():
    X = df[exp["num"] + exp["bin"] + exp["cat"]]
    for model_name, model in models.items():
        pipe = Pipeline([("prep", make_preprocessor(exp)), ("model", model)])
        s = group_cv(pipe, X, y, groups)
        rows.append({"experiment": exp_name, "model": model_name,
                     **s.mean().add_suffix("_mean"), **s.std().add_suffix("_std")})

results = pd.DataFrame(rows)
print(results.round(3).to_string(index=False))
results.to_csv("regression_results.csv", index=False)
