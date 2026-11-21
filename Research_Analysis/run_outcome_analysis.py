"""Link public study scores to session-specific conversation features and analyze outcomes.

Run after `run_research_analysis.py`:
    python Research_Analysis/run_outcome_analysis.py

The primary outcome is the later unassisted exam score (Part3Tot). Class is the
randomization unit, so predictive validation holds out whole classes. Text-feature
associations remain observational even though treatment assignment was randomized.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "AcademicOutcomes" / "matched_session_outcomes.csv"
FEATURES = ROOT / "outputs" / "research" / "datasets" / "student_session_features.csv"
OUT = ROOT / "outputs" / "research" / "outcomes"
TABLES, FIGURES = OUT / "tables", OUT / "figures"
DATASETS = ROOT / "outputs" / "research" / "datasets"
SEED = 42
for folder in (TABLES, FIGURES, OUT, DATASETS):
    folder.mkdir(parents=True, exist_ok=True)


def bh_fdr(pvalues: list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adj)
    result[order] = np.clip(adj, 0, 1)
    return result


def design_context(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Context-only OLS design; IDs (student and classroom) are not predictors."""
    numeric = frame[["treatment_aug", "gpa_prev"]].astype(float).reset_index(drop=True)
    # The public `Year` and project `grade` columns encode the same value.
    # Include one copy to keep the design matrix full rank.
    grade_column = "grade" if "grade" in frame else "Year"
    categorical_cols = [c for c in [grade_column, "session_id", "teacher", "Grader"] if c in frame]
    categorical_values = frame[categorical_cols].astype("string").fillna("Missing") if categorical_cols else None
    categorical = pd.get_dummies(categorical_values, drop_first=True, dtype=float).reset_index(drop=True) if categorical_cols else pd.DataFrame(index=range(len(frame)))
    matrix = np.column_stack([np.ones(len(frame)), numeric.to_numpy(), categorical.to_numpy()])
    names = ["intercept", "treatment_aug", "gpa_prev"] + list(categorical.columns)
    return matrix, names


def cluster_robust_ols(y: np.ndarray, X: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """OLS with CR1 cluster-robust covariance and t reference df = clusters - 1."""
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("Rank-deficient design")
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    residual = y - X @ beta
    unique = np.unique(clusters)
    meat = np.zeros_like(bread)
    for cluster in unique:
        idx = clusters == cluster
        score = X[idx].T @ residual[idx]
        meat += np.outer(score, score)
    n, k, g = len(y), X.shape[1], len(unique)
    covariance = (g / (g - 1)) * ((n - 1) / max(n - k, 1)) * bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    tvalues = beta / np.where(se > 0, se, np.nan)
    pvalues = 2 * stats.t.sf(np.abs(tvalues), df=g - 1)
    critical = stats.t.ppf(.975, df=g - 1)
    ci = np.column_stack([beta - critical * se, beta + critical * se])
    return beta, se, pvalues, g, ci


def build_preprocessor(columns: list[str], frame: pd.DataFrame) -> ColumnTransformer:
    categorical_names = {"grade", "treatment", "session_id", "teacher", "Grader", "Year"}
    categorical = [c for c in columns if c in categorical_names or not pd.api.types.is_numeric_dtype(frame[c])]
    numeric = [c for c in columns if c not in categorical]
    steps = []
    if numeric:
        steps.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        steps.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(steps)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Matched public academic outcomes are missing: {SOURCE}")
    if not FEATURES.exists():
        raise FileNotFoundError(f"Run run_research_analysis.py first; missing: {FEATURES}")
    parent_summary = OUT.parent / "research_run_summary.json"
    if parent_summary.is_file():
        text_summary = json.loads(parent_summary.read_text(encoding="utf-8"))
        text_summary["outcome_model_run"] = False
        parent_summary.write_text(json.dumps(text_summary, indent=2, default=str), encoding="utf-8")
    scores = pd.read_csv(SOURCE, low_memory=False)
    session = pd.read_csv(FEATURES, low_memory=False)
    outcome_keys = ["username", "session_id"]
    if scores.duplicated(outcome_keys).any() or session.duplicated(outcome_keys).any():
        raise ValueError("Outcome or session feature keys are not unique")
    if scores.username.isna().any() or session.username.isna().any():
        raise ValueError("Missing username in validated outcome/session records")
    score_ids, session_ids = set(scores.username), set(session.username)
    if not session_ids.issubset(score_ids):
        raise ValueError("Some local conversation students have no exact public outcome ID match")
    if scores["Treatment arm"].eq("augmented").ne(scores.GPTTutor.eq(1)).any():
        raise ValueError("Public GPTTutor indicator and Treatment arm disagree")
    if scores["Treatment arm"].eq("vanilla").ne(scores.GPTBase.eq(1)).any():
        raise ValueError("Public GPTBase indicator and Treatment arm disagree")

    mapped_arm = scores["Treatment arm"].map({"augmented": "aug", "vanilla": "vanilla"})
    if mapped_arm.isna().any() or scores["Treatment arm"].eq("control").any():
        raise ValueError("The matched conversation cohort must contain mapped GPT Base/Tutor rows only")
    scores["public_treatment"] = mapped_arm
    merged = scores.merge(session, on=outcome_keys, how="left", validate="one_to_one", suffixes=("", "_message"), indicator=True)
    available = merged["_merge"].eq("both")
    if not merged.loc[available, "public_treatment"].eq(merged.loc[available, "treatment"]).all():
        raise ValueError("Exact ID matched, but treatment assignment differs between conversation and score sources")
    if merged.Class.isna().any() or merged.Part2Tot.isna().any() or merged.Part3Tot.isna().any():
        raise ValueError("Expected class and both score outcomes to be present in the matched source")
    observed = merged.loc[available].copy().reset_index(drop=True)
    if "grade" in observed and "Year" in observed:
        same_grade = pd.to_numeric(observed.grade, errors="coerce").eq(pd.to_numeric(observed.Year, errors="coerce"))
        if not same_grade.all():
            raise ValueError("Local grade and public Year disagree; inspect before choosing context covariates")
    # One student must belong to one classroom; grouping by class then also prevents student leakage.
    if observed.groupby("username").Class.nunique().max() != 1:
        raise ValueError("A student occurs in multiple classes; use student grouping instead of class grouping")
    if observed.groupby("Class").treatment.nunique().max() != 1:
        raise ValueError("Classroom membership and treatment assignment are inconsistent")
    observed.to_csv(DATASETS / "session_outcomes_with_chat_features.csv", index=False)

    config = json.loads((ROOT / "ProcessedDatasets" / "preprocessing_config.json").read_text(encoding="utf-8"))
    # Grader is assigned at scoring time, so it is retained as a nuisance
    # adjustment in treatment contrasts but excluded from the predictive models.
    baseline = [c for c in ["grade" if "grade" in observed else "Year", "treatment", "session_id", "gpa_prev", "teacher"] if c in observed]
    behavioral = list(config["primary_features"]) + list(config["supplementary_rate_features"]) + list(config["supplementary_binary_features"])
    missing_behavior = [c for c in behavioral if c not in observed]
    if missing_behavior:
        raise ValueError(f"Session-level behavioral features missing: {missing_behavior}")
    text_features = [c for c in observed if c.startswith("text_")]
    feature_sets = {
        "baseline_context": baseline,
        "existing_behavior": baseline + behavioral,
        "new_text": baseline + text_features,
        "combined": baseline + behavioral + text_features,
    }
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=.01, max_iter=10000, random_state=SEED),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=.05, max_depth=2, random_state=SEED),
    }
    class_groups = observed.Class.astype(str)
    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(observed, observed.Part3Tot, class_groups))
    fold_results, prediction_rows = [], []
    for target in ["Part3Tot", "Part2Tot"]:
        for set_name, raw_cols in feature_sets.items():
            cols = list(dict.fromkeys(c for c in raw_cols if c in observed and c not in {"username", "Class", "Part2Tot", "Part3Tot", "treatment_aug"}))
            X = observed[cols]
            y = pd.to_numeric(observed[target], errors="coerce")
            for model_name, estimator in models.items():
                scores_by_fold = []
                for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                    train_classes, test_classes = set(class_groups.iloc[train_idx]), set(class_groups.iloc[test_idx])
                    train_students, test_students = set(observed.username.iloc[train_idx]), set(observed.username.iloc[test_idx])
                    if train_classes & test_classes or train_students & test_students:
                        raise AssertionError("Class/student leakage in outcome cross-validation")
                    pipe = Pipeline([("prep", build_preprocessor(cols, observed)), ("model", estimator)])
                    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
                    prediction = pipe.predict(X.iloc[test_idx])
                    fold_scores = {
                        "MAE": mean_absolute_error(y.iloc[test_idx], prediction),
                        "RMSE": np.sqrt(mean_squared_error(y.iloc[test_idx], prediction)),
                        "R2": r2_score(y.iloc[test_idx], prediction) if len(test_idx) > 1 else np.nan,
                    }
                    scores_by_fold.append(fold_scores)
                    for pos, value in zip(test_idx, prediction):
                        prediction_rows.append({"outcome": target, "feature_set": set_name, "model": model_name,
                                                "fold": fold, "username": observed.username.iloc[pos],
                                                "Class": observed.Class.iloc[pos], "session_id": observed.session_id.iloc[pos],
                                                "treatment": observed.treatment.iloc[pos], "observed": y.iloc[pos], "predicted": value})
                fold_results.append({"outcome": target, "feature_set": set_name, "model": model_name,
                                     "rows": len(observed), "students": observed.username.nunique(), "classes": observed.Class.nunique(),
                                     **{f"{m}_mean": float(np.nanmean([s[m] for s in scores_by_fold])) for m in ["MAE", "RMSE", "R2"]},
                                     **{f"{m}_sd": float(np.nanstd([s[m] for s in scores_by_fold])) for m in ["MAE", "RMSE", "R2"]}})
    results = pd.DataFrame(fold_results)
    if len(results) != 2 * len(feature_sets) * len(models):
        raise RuntimeError(f"Expected 40 grouped-CV result rows, received {len(results)}")
    if results[["MAE_mean", "RMSE_mean", "R2_mean"]].isna().any().any():
        raise RuntimeError("Outcome grouped-CV results contain missing metrics")
    results.to_csv(TABLES / "outcome_grouped_cv_results.csv", index=False)
    predictions = pd.DataFrame(prediction_rows)
    expected_predictions = len(observed) * 2 * len(feature_sets) * len(models)
    if len(predictions) != expected_predictions or predictions.predicted.isna().any():
        raise RuntimeError(f"Expected {expected_predictions} complete out-of-fold predictions; received {len(predictions)}")
    predictions.to_csv(TABLES / "outcome_grouped_cv_predictions.csv", index=False)

    # Treatment-arm contrast on exam/practice scores, with the published class-level randomization unit.
    treatment_rows = []
    for target in ["Part3Tot", "Part2Tot"]:
        complete = observed.dropna(subset=["gpa_prev", target]).copy()
        treatment_aug = complete.treatment.eq("aug").astype(float).to_numpy()
        context = complete.copy()
        context["treatment_aug"] = treatment_aug
        base, names = design_context(context)
        beta, se, pvals, groups, ci = cluster_robust_ols(complete[target].to_numpy(dtype=float), base,
                                                         complete.Class.astype(str).to_numpy())
        idx = names.index("treatment_aug")
        treatment_rows.append({"outcome": target, "contrast": "GPT Tutor (aug) - GPT Base (vanilla)",
                               "coefficient": beta[idx], "cluster_robust_se": se[idx], "ci95_low": ci[idx, 0],
                               "ci95_high": ci[idx, 1], "p_value": pvals[idx], "n_rows": len(complete),
                               "n_students": complete.username.nunique(), "n_classes": groups,
                               "interpretation": "Randomized class-arm contrast within the observed non-honors GPT cohort; Part3Tot is the later unassisted exam."})
    pd.DataFrame(treatment_rows).to_csv(TABLES / "treatment_arm_score_contrasts.csv", index=False)

    # Associational text-feature slopes and feature-by-treatment interactions on the later exam.
    interaction_frame = observed.dropna(subset=["Part3Tot", "gpa_prev"]).copy().reset_index(drop=True)
    context_design, context_names = design_context(interaction_frame.assign(
        treatment_aug=interaction_frame.treatment.eq("aug").astype(float)))
    interaction_rows = []
    for feature in behavioral + text_features:
        values = pd.to_numeric(interaction_frame[feature], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(values)
        if keep.sum() < 50 or np.nanstd(values[keep]) == 0:
            continue
        base = context_design[keep]
        treatment_aug = interaction_frame.loc[keep, "treatment"].eq("aug").astype(float).to_numpy()
        z = (values[keep] - values[keep].mean()) / values[keep].std(ddof=0)
        design = np.column_stack([base, z, treatment_aug * z])
        y = interaction_frame.loc[keep, "Part3Tot"].to_numpy(dtype=float)
        cluster = interaction_frame.loc[keep, "Class"].astype(str).to_numpy()
        try:
            beta, se, pvals, groups, ci = cluster_robust_ols(y, design, cluster)
        except ValueError:
            continue
        interaction_idx = design.shape[1] - 1
        interaction_rows.append({"feature": feature, "coefficient_vanilla_per_sd": beta[-2],
                                 "coefficient_treatment_interaction": beta[interaction_idx],
                                 "interaction_cluster_robust_se": se[interaction_idx],
                                 "interaction_ci95_low": ci[interaction_idx, 0],
                                 "interaction_ci95_high": ci[interaction_idx, 1],
                                 "interaction_p_value": pvals[interaction_idx], "rows": len(y),
                                 "students": interaction_frame.loc[keep, "username"].nunique(), "classes": groups})
    if interaction_rows:
        q = bh_fdr([r["interaction_p_value"] for r in interaction_rows])
        for row, qv in zip(interaction_rows, q): row["interaction_fdr_q_value"] = qv
    pd.DataFrame(interaction_rows).sort_values("interaction_fdr_q_value").to_csv(TABLES / "exam_text_treatment_interactions.csv", index=False)

    # Use a pre-specified Ridge model for incremental feature-value comparisons;
    # selecting a different winner per feature set on these same folds is optimistic.
    part3 = results[results.outcome.eq("Part3Tot")]
    ridge_results = part3[part3.model.eq("Ridge")].set_index("feature_set")
    ridge_oof = pd.DataFrame(prediction_rows)
    ridge_oof = ridge_oof[(ridge_oof.outcome == "Part3Tot") & (ridge_oof.model == "Ridge")]
    wide = ridge_oof.pivot(index=["username", "Class", "session_id", "observed"], columns="feature_set", values="predicted").reset_index()
    classes = wide.Class.astype(str).unique()
    class_rows = {class_id: np.flatnonzero(wide.Class.astype(str).to_numpy() == class_id) for class_id in classes}
    rng = np.random.default_rng(SEED)
    increment_rows = []
    y_oof = wide.observed.to_numpy(dtype=float)
    for set_name in ["existing_behavior", "new_text", "combined"]:
        y_base = wide["baseline_context"].to_numpy(dtype=float)
        y_aug = wide[set_name].to_numpy(dtype=float)
        base_rmse = float(np.sqrt(np.mean((y_oof - y_base) ** 2)))
        augmented_rmse = float(np.sqrt(np.mean((y_oof - y_aug) ** 2)))
        bootstrap_delta = []
        for _ in range(2000):
            sampled_classes = rng.choice(classes, size=len(classes), replace=True)
            positions = np.concatenate([class_rows[c] for c in sampled_classes])
            bootstrap_delta.append(float(
                np.sqrt(np.mean((y_oof[positions] - y_base[positions]) ** 2))
                - np.sqrt(np.mean((y_oof[positions] - y_aug[positions]) ** 2))
            ))
        increment_rows.append({
            "comparison": f"{set_name} vs baseline (Ridge)", "model": "Ridge",
            "baseline_RMSE": base_rmse, "augmented_RMSE": augmented_rmse,
            "rmse_reduction": base_rmse - augmented_rmse,
            "rmse_reduction_class_bootstrap_ci95_low": float(np.quantile(bootstrap_delta, .025)),
            "rmse_reduction_class_bootstrap_ci95_high": float(np.quantile(bootstrap_delta, .975)),
            "bootstrap_replicates": len(bootstrap_delta), "classes_resampled": len(classes),
        })
    increment = pd.DataFrame(increment_rows)
    increment.to_csv(TABLES / "exam_incremental_feature_value.csv", index=False)

    # Visual summaries.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, target, title in zip(axes, ["Part3Tot", "Part2Tot"], ["Unassisted exam (primary)", "Assisted practice (secondary)"]):
        subset = results[results.outcome.eq(target)]
        groups_by_set = [subset.loc[subset.feature_set.eq(name) & subset.model.eq("Ridge"), "RMSE_mean"].iloc[0]
                         for name in ["baseline_context", "existing_behavior", "new_text", "combined"]]
        ax.bar(["Baseline", "Behavior", "Text", "Combined"], groups_by_set,
               color=["#5B7C99", "#548C69", "#D28A2C", "#7C6AA6"])
        ax.set_ylabel("Mean class-held-out Ridge RMSE (lower is better)")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIGURES / "outcome_model_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [observed.loc[observed.treatment.eq(t), "Part3Tot"] for t in ["vanilla", "aug"]]
    ax.boxplot(data, tick_labels=["GPT Base (vanilla)", "GPT Tutor (aug)"], showfliers=False)
    ax.set_ylabel("Part 3 score (unassisted exam)")
    ax.set_title("Unassisted exam scores in the conversation-linked cohort")
    fig.tight_layout(); fig.savefig(FIGURES / "exam_scores_by_treatment.png", dpi=180); plt.close(fig)

    shap_result = {"available": False, "reason": "not computed"}
    try:
        import shap
        best_model_name = str(part3[part3.feature_set.eq("combined")].sort_values("RMSE_mean").iloc[0].model)
        estimator = models[best_model_name]
        shap_cols = feature_sets["combined"]
        shap_splits = splits
        train_idx, test_idx = shap_splits[0]
        pipe = Pipeline([("prep", build_preprocessor(shap_cols, observed)), ("model", estimator)])
        pipe.fit(observed.iloc[train_idx][shap_cols], observed.iloc[train_idx].Part3Tot)
        prep, fitted_model = pipe.named_steps["prep"], pipe.named_steps["model"]
        X_train = prep.transform(observed.iloc[train_idx][shap_cols])
        X_test = prep.transform(observed.iloc[test_idx][shap_cols])
        feature_names = list(prep.get_feature_names_out())
        background = X_train[:min(100, X_train.shape[0])]
        if isinstance(fitted_model, (RandomForestRegressor, GradientBoostingRegressor)):
            explainer = shap.TreeExplainer(fitted_model, data=background, feature_perturbation="interventional")
            explanation = explainer(X_test[:min(250, X_test.shape[0])])
            shap_values = np.asarray(explanation.values)
            base_values = np.asarray(explanation.base_values)
        else:
            explainer = shap.LinearExplainer(fitted_model, background)
            shap_values = np.asarray(explainer.shap_values(X_test[:min(250, X_test.shape[0])]))
            base_values = np.asarray(explainer.expected_value)
        imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        imp.sort_values("mean_abs_shap", ascending=False).to_csv(TABLES / "exam_combined_model_shap.csv", index=False)
        shap.summary_plot(shap_values, X_test[:min(250, X_test.shape[0])], feature_names=feature_names,
                          show=False, max_display=15)
        plt.tight_layout(); plt.savefig(FIGURES / "exam_combined_model_shap.png", dpi=180, bbox_inches="tight"); plt.close()
        shap_result = {"available": True, "feature_set": "combined", "model": best_model_name,
                       "held_out_class_fold": 1, "test_students": int(observed.iloc[test_idx].username.nunique()),
                       "top_features": imp.sort_values("mean_abs_shap", ascending=False).head(10).to_dict(orient="records")}
    except Exception as exc:
        shap_result = {"available": False, "reason": f"SHAP computation failed: {type(exc).__name__}: {exc}"}

    audit = {
        "outcome_source": "https://github.com/obastani/GenAICanHarmLearning",
        "outcome_file": str(SOURCE.relative_to(ROOT)),
        "primary_outcome": "Part3Tot (subsequent unassisted exam score)",
        "secondary_outcome": "Part2Tot (assisted practice score; concurrent with interactions)",
        "total_outcome_rows_for_local_ai_students": int(len(merged)),
        "matched_student_sessions_with_chat": int(available.sum()),
        "outcome_rows_without_chat_for_session": int((~available).sum()),
        "chat_sessions_without_score_row": int(len(session.merge(scores[outcome_keys], on=outcome_keys, how="left", indicator=True).query("_merge == 'left_only'"))),
        "matched_students": int(observed.username.nunique()),
        "unmatched_students": int(len(set(scores.username) - set(observed.username))),
        "exact_join_keys": ["username == Student ID", "session_id == Session"],
        "predictive_baseline_features": baseline,
        "username_id_matches": int(len(score_ids & session_ids)),
        "treatment_mapping_validated": {"vanilla": "GPT Base", "aug": "GPT Tutor"},
        "treatment_mismatch_rows": 0,
        "duplicate_score_session_keys": int(scores.duplicated(outcome_keys).sum()),
        "duplicate_chat_session_keys": int(session.duplicated(outcome_keys).sum()),
        "treatment_class_counts": observed.groupby("treatment").Class.nunique().to_dict(),
        "outcome_missing_values": {c: int(observed[c].isna().sum()) for c in ["Part2Tot", "Part3Tot"]},
        "gpa_missing_rows": int(observed.gpa_prev.isna().sum()),
        "class_holdout_cv_folds": 5,
        "classes": int(observed.Class.nunique()), "students": int(observed.username.nunique()),
        "outcome_model_metrics": results.to_dict(orient="records"),
        "best_models_part3": {name: {"model": str(group.sort_values("RMSE_mean").iloc[0].model),
                                       "RMSE_mean": float(group.sort_values("RMSE_mean").iloc[0].RMSE_mean),
                                       "MAE_mean": float(group.sort_values("RMSE_mean").iloc[0].MAE_mean),
                                       "R2_mean": float(group.sort_values("RMSE_mean").iloc[0].R2_mean)}
                               for name, group in part3.groupby("feature_set")},
        "incremental_comparisons": increment.to_dict(orient="records"),
        "treatment_arm_score_contrasts": treatment_rows,
        "shap": shap_result,
        "causal_boundary": "Text-feature slopes and feature-by-treatment interactions are observational associations; only the GPT Tutor versus GPT Base assignment contrast inherits the class-randomized design, and it is estimated only in the conversation-linked non-honors AI cohort.",
    }
    (OUT / "outcome_run_summary.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    if parent_summary.is_file():
        text_summary = json.loads(parent_summary.read_text(encoding="utf-8"))
        text_summary["academic_outcome_data_available"] = True
        text_summary["outcome_model_run"] = True
        text_summary["outcome_analysis_summary"] = "outcomes/outcome_run_summary.json"
        parent_summary.write_text(json.dumps(text_summary, indent=2, default=str), encoding="utf-8")
    required_outputs = [
        OUT / "outcome_run_summary.json",
        DATASETS / "session_outcomes_with_chat_features.csv",
        TABLES / "outcome_grouped_cv_results.csv",
        TABLES / "outcome_grouped_cv_predictions.csv",
        TABLES / "exam_incremental_feature_value.csv",
        TABLES / "exam_text_treatment_interactions.csv",
        FIGURES / "outcome_model_comparison.png",
        FIGURES / "exam_scores_by_treatment.png",
    ]
    missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(f"Outcome analysis is incomplete; missing outputs: {missing_outputs}")
    print(json.dumps(audit, indent=2, default=str))
    print(f"\nSaved outcome-analysis artifacts to {OUT}")


if __name__ == "__main__":
    main()
