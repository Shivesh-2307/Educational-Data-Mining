# %% [markdown]
# # Student–LLM Interaction Analysis: Unified End-to-End Pipeline (Clustering, E1–E3 Classification, SHAP Explainability & Statistical Analysis)
#
# This unified script continues Srijan's preprocessing (`ProcessedDatasets/`) and Shivesh's
# E1/E2/E3 experimental framework (`Regression.py`) through Shreyansh's full remaining workflow:
# 1. Data Audit & Exploratory Analysis
# 2. Unsupervised Learning: Clustering & Student Interaction Profiles
# 3. Supervised Learning: Treatment Classification across E1, E2, E3 (continuing Shivesh's pipeline)
# 4. Explainable AI: SHAP Global, Feature-Level, and Individual Explanations (Shreyansh's focus)
# 5. Additional Analysis: Statistical Tests, Grade Confounding & Prompt Pattern Audit

# %%
import json
import os
import platform
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import shap
import sklearn
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Reproducibility & Paths
SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ProcessedDatasets" if (BASE_DIR / "ProcessedDatasets").exists() else BASE_DIR.parent / "ProcessedDatasets"
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

for d in (OUT_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.size": 9.5,
})

# %% [markdown]
# ## STAGE 1: Data Audit & Verification (Part I)

# %%
print("=" * 72)
print("STAGE 1: DATA AUDIT & VERIFICATION")
print("=" * 72)

with open(DATA_DIR / "preprocessing_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

df = pd.read_csv(DATA_DIR / "student_features_final.csv")
msgs = pd.read_csv(DATA_DIR / "cleaned_user_messages.csv")

PRIMARY_FEATURES = config["primary_features"]
SPARSE_RATES = config["supplementary_rate_features"]
SPARSE_FLAGS = config["supplementary_binary_features"]
ID_COL = config["identifier"]
TARGET_COL = config["target"]  # 'treatment'

audit = {
    "student_rows": int(len(df)),
    "student_cols": int(df.shape[1]),
    "unique_students": int(df[ID_COL].nunique()),
    "duplicate_usernames": int(df[ID_COL].duplicated().sum()),
    "missing_values_total": int(df.isna().sum().sum()),
    "infinite_values_total": int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum()),
    "message_rows": int(len(msgs)),
    "message_cols": int(msgs.shape[1]),
    "treatment_counts": {k: int(v) for k, v in df[TARGET_COL].value_counts().items()},
    "grade_counts": {int(k): int(v) for k, v in df["grade"].value_counts().sort_index().items()},
    "grade_x_treatment": {
        int(g): {t: int(v) for t, v in row.items()}
        for g, row in pd.crosstab(df["grade"], df[TARGET_COL]).iterrows()
    },
    "primary_skewness": {
        col: round(float(df[col].skew()), 3) for col in PRIMARY_FEATURES
    },
    "sparse_nonzero_pct": {
        col: round(float((df[col] > 0).mean() * 100), 2) for col in SPARSE_RATES
    },
}

print(f"Students: {audit['student_rows']} rows x {audit['student_cols']} cols | Unique: {audit['unique_students']}")
print(f"Messages: {audit['message_rows']} rows x {audit['message_cols']} cols")
print(f"Missing: {audit['missing_values_total']} | Duplicates: {audit['duplicate_usernames']}")
print(f"Treatment distribution: {audit['treatment_counts']}")
print(f"Grade distribution: {audit['grade_counts']}")
print("Primary feature skewness:", audit["primary_skewness"])

# EDA Figure: Correlation Heatmap & Grade-Treatment Balance
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.4, 1.0]})

corr = df[PRIMARY_FEATURES].corr(method="spearman")
short_labels = [
    "Msg Count", "Avg Words", "Avg Questions", "Explain Rate",
    "Solution Rate", "Verify Rate", "Conceptual Rate", "Code Rate", "Follow-up Rate"
]
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
    xticklabels=short_labels, yticklabels=short_labels,
    ax=axes[0], cbar_kws={"shrink": 0.8}, annot_kws={"size": 8}
)
axes[0].set_title("Spearman Correlation: 9 Primary Interaction Features")
axes[0].tick_params(axis="x", rotation=35)

ct_gt = pd.crosstab(df["grade"], df[TARGET_COL])
ct_gt.plot(kind="bar", ax=axes[1], color=["#2b6cb0", "#ed8936"], edgecolor="black", linewidth=0.6)
axes[1].set_title("Student Count by Grade and Treatment Group")
axes[1].set_xlabel("Grade")
axes[1].set_ylabel("Number of Students")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Treatment")
for container in axes[1].containers:
    axes[1].bar_label(container, padding=2, fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "01_eda_correlations_and_distributions.png", bbox_inches="tight")
plt.close()

# %% [markdown]
# ## STAGE 2: Unsupervised Learning — Clustering & Student Profiles (Part II)

# %%
print("\n" + "=" * 72)
print("STAGE 2: UNSUPERVISED LEARNING (CLUSTERING & STUDENT PROFILES)")
print("=" * 72)

# Preprocessing for clustering: log1p + StandardScaler on 9 primary features only
# (excluding username, treatment, grade)
X_cluster_raw = df[PRIMARY_FEATURES].copy()
X_cluster_scaled = StandardScaler().fit_transform(np.log1p(X_cluster_raw))

k_range = range(2, 9)
cluster_eval_rows = []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=25, random_state=SEED)
    labels = km.fit_predict(X_cluster_scaled)
    cluster_eval_rows.append({
        "k": k,
        "inertia": round(float(km.inertia_), 2),
        "silhouette": round(float(silhouette_score(X_cluster_scaled, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_cluster_scaled, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_cluster_scaled, labels)), 2),
    })

cluster_eval_df = pd.DataFrame(cluster_eval_rows)
cluster_eval_df.to_csv(TAB_DIR / "clustering_metrics.csv", index=False)
print(cluster_eval_df.to_string(index=False))

# Select optimal k balancing statistical quality and interpretable student profiles
# Check k=3 vs k=4; pick the best k in {3, 4} by silhouette, or overall best k >= 3 for nuanced profiles
best_k_row = cluster_eval_df.loc[cluster_eval_df["silhouette"].idxmax()]
best_k_overall = int(best_k_row["k"])
# Also check k=3 and k=4 explicitly
sil_k3 = float(cluster_eval_df.loc[cluster_eval_df["k"] == 3, "silhouette"].values[0])
sil_k4 = float(cluster_eval_df.loc[cluster_eval_df["k"] == 4, "silhouette"].values[0])
CHOSEN_K = 3 if sil_k3 >= sil_k4 else 4
if best_k_overall in (3, 4):
    CHOSEN_K = best_k_overall
print(f"\nHighest silhouette overall: k={best_k_overall} ({best_k_row['silhouette']:.4f}); "
      f"selected k={CHOSEN_K} for interpretable multi-profile analysis.")

# Fit K-Means, Agglomerative, and GMM at CHOSEN_K
kmeans_final = KMeans(n_clusters=CHOSEN_K, n_init=50, random_state=SEED)
km_labels = kmeans_final.fit_predict(X_cluster_scaled)

agg_final = AgglomerativeClustering(n_clusters=CHOSEN_K, linkage="ward")
agg_labels = agg_final.fit_predict(X_cluster_scaled)

gmm_final = GaussianMixture(n_components=CHOSEN_K, covariance_type="full", n_init=10, random_state=SEED)
gmm_labels = gmm_final.fit_predict(X_cluster_scaled)

algo_comp = pd.DataFrame([
    {
        "algorithm": f"K-Means (k={CHOSEN_K})",
        "silhouette": round(float(silhouette_score(X_cluster_scaled, km_labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_cluster_scaled, km_labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_cluster_scaled, km_labels)), 2),
    },
    {
        "algorithm": f"Agglomerative Ward (k={CHOSEN_K})",
        "silhouette": round(float(silhouette_score(X_cluster_scaled, agg_labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_cluster_scaled, agg_labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_cluster_scaled, agg_labels)), 2),
    },
    {
        "algorithm": f"Gaussian Mixture (k={CHOSEN_K})",
        "silhouette": round(float(silhouette_score(X_cluster_scaled, gmm_labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_cluster_scaled, gmm_labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_cluster_scaled, gmm_labels)), 2),
    },
])
algo_comp.to_csv(TAB_DIR / "clustering_algorithm_comparison.csv", index=False)
print("\nClustering Algorithm Comparison:")
print(algo_comp.to_string(index=False))

df["cluster"] = km_labels

# Compute cluster profiles (size, %, mean, median)
profile_means = df.groupby("cluster")[PRIMARY_FEATURES].mean()
profile_medians = df.groupby("cluster")[PRIMARY_FEATURES].median()
cluster_sizes = df["cluster"].value_counts().sort_index()
cluster_pcts = (cluster_sizes / len(df) * 100).round(1)

# Assign data-grounded names based on standardized z-scores of cluster centroids
z_df = pd.DataFrame(X_cluster_scaled, columns=PRIMARY_FEATURES)
z_df["cluster"] = km_labels
z_means = z_df.groupby("cluster").mean()

cluster_names = {}
used_names = set()
for cid in sorted(df["cluster"].unique()):
    z = z_means.loc[cid]
    top_feat = z.idxmax()
    if z["code_prompt_rate"] > 0.7:
        name = "Code-Heavy & Verification Users"
    elif z["average_question_count"] > 0.5 and z["conceptual_question_rate"] > 0.4:
        name = "Inquiry-Driven & Conceptual Users"
    elif z["average_word_count"] > 0.4 and z["complete_solution_request_rate"] > 0.3:
        name = "Detailed Solution-Seeking Users"
    elif z["student_message_count"] > 0.4 and z["followup_prompt_rate"] > 0.2:
        name = "High-Volume Interactive Users"
    elif z["student_message_count"] < -0.2 and z["average_question_count"] < 0:
        name = "Brief / Low-Inquiry Users"
    else:
        readable = top_feat.replace("_rate", "").replace("_", " ").title()
        name = f"{readable}-Focused Users"
    if name in used_names:
        name = f"{name} (Cluster {cid})"
    used_names.add(name)
    cluster_names[int(cid)] = name

df["cluster_name"] = df["cluster"].map(cluster_names)

profile_summary = profile_means.round(4).copy()
profile_summary.insert(0, "pct", cluster_pcts.values)
profile_summary.insert(0, "count", cluster_sizes.values)
profile_summary.insert(0, "profile_name", [cluster_names[i] for i in profile_summary.index])
profile_summary.to_csv(TAB_DIR / "cluster_profiles.csv")
profile_medians.round(4).to_csv(TAB_DIR / "cluster_profiles_medians.csv")
print("\nCluster Profiles (Means):")
print(profile_summary[["profile_name", "count", "pct", "student_message_count", "average_word_count", "average_question_count", "conceptual_question_rate", "complete_solution_request_rate", "code_prompt_rate"]].to_string())

# Post-hoc Chi-square + Cramér's V: Cluster vs Treatment & Cluster vs Grade
def chi2_cramers_v(ct):
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
    return {"chi2": round(float(chi2), 4), "p_value": round(float(p), 6), "dof": int(dof), "cramers_v": round(float(v), 4)}

ct_cl_treat = pd.crosstab(df["cluster_name"], df[TARGET_COL])
ct_cl_grade = pd.crosstab(df["cluster_name"], df["grade"])
chi_cl_treat = chi2_cramers_v(ct_cl_treat)
chi_cl_grade = chi2_cramers_v(ct_cl_grade)
print("\nPost-hoc Cluster vs Treatment:", chi_cl_treat)
print(ct_cl_treat)
print("\nPost-hoc Cluster vs Grade:", chi_cl_grade)
print(ct_cl_grade)

# Figure 2: Clustering Evaluation (Elbow + Silhouette + Davies-Bouldin)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].plot(cluster_eval_df["k"], cluster_eval_df["inertia"], marker="o", color="#2b6cb0")
axes[0].axvline(CHOSEN_K, color="#e53e3e", linestyle="--", alpha=0.8, label=f"Selected k={CHOSEN_K}")
axes[0].set_title("Elbow Method (Inertia)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].legend()

axes[1].plot(cluster_eval_df["k"], cluster_eval_df["silhouette"], marker="s", color="#2f855a")
axes[1].axvline(CHOSEN_K, color="#e53e3e", linestyle="--", alpha=0.8)
axes[1].set_title("Silhouette Score (Higher = Better)")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")

axes[2].plot(cluster_eval_df["k"], cluster_eval_df["davies_bouldin"], marker="^", color="#dd6b20")
axes[2].axvline(CHOSEN_K, color="#e53e3e", linestyle="--", alpha=0.8)
axes[2].set_title("Davies–Bouldin Index (Lower = Better)")
axes[2].set_xlabel("Number of Clusters (k)")
axes[2].set_ylabel("Davies–Bouldin Index")

plt.tight_layout()
plt.savefig(FIG_DIR / "02_clustering_evaluation.png", bbox_inches="tight")
plt.close()

# Figure 3: PCA Projection + Cluster Centroid Z-score Heatmap
pca = PCA(n_components=2, random_state=SEED)
coords = pca.fit_transform(X_cluster_scaled)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.1, 1.3]})
palette = sns.color_palette("Set2", n_colors=CHOSEN_K)
for idx, cid in enumerate(sorted(df["cluster"].unique())):
    mask = df["cluster"] == cid
    axes[0].scatter(
        coords[mask, 0], coords[mask, 1],
        label=f"C{cid}: {cluster_names[cid]} (n={mask.sum()})",
        alpha=0.72, s=36, color=palette[idx], edgecolors="white", linewidth=0.4
    )
axes[0].set_title(f"PCA Projection of Student Clusters (Var Explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].legend(fontsize=8, loc="best")

z_plot = z_means.copy()
z_plot.index = [f"C{i}: {cluster_names[i]}" for i in z_plot.index]
z_plot.columns = short_labels
sns.heatmap(z_plot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1], cbar_kws={"label": "Mean Z-Score"})
axes[1].set_title("Cluster Centroid Profiles (Standardized log1p Features)")
axes[1].tick_params(axis="x", rotation=35)
axes[1].tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig(FIG_DIR / "03_cluster_profiles.png", bbox_inches="tight")
plt.close()

# %% [markdown]
# ## STAGE 3: Supervised Learning — Treatment Classification across E1, E2, E3 (Part III + Shivesh's Framework + Step 17)

# %%
print("\n" + "=" * 72)
print("STAGE 3: SUPERVISED LEARNING (TREATMENT CLASSIFICATION: E1, E2, E3)")
print("=" * 72)

# Binary target: aug = 1, vanilla = 0
y = (df[TARGET_COL] == "aug").astype(int)

# Exact E1 / E2 / E3 feature groups continuing Shivesh's Regression.py design
experiments = {
    "E1_Primary": {"num": PRIMARY_FEATURES, "bin": [], "cat": []},
    "E2_Primary_Sparse": {"num": PRIMARY_FEATURES + SPARSE_RATES, "bin": SPARSE_FLAGS, "cat": []},
    "E3_Combined_Grade": {"num": PRIMARY_FEATURES + SPARSE_RATES, "bin": SPARSE_FLAGS, "cat": ["grade"]},
}

def make_preprocessor(exp):
    steps = [
        (
            "num",
            Pipeline([
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]),
            exp["num"],
        )
    ]
    if exp["bin"]:
        steps.append(("bin", "passthrough", exp["bin"]))
    if exp["cat"]:
        steps.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), exp["cat"]))
    return ColumnTransformer(steps, verbose_feature_names_out=False)

def build_models():
    return {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.05, max_depth=4, min_samples_leaf=15, random_state=SEED
        ),
    }

# Stratified 80/20 Train/Test Split (same split indices across E1, E2, E3 for fair comparison)
train_idx, test_idx = train_test_split(
    df.index, test_size=0.20, stratify=y, random_state=SEED
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}

cv_rows = []
test_rows = []
fitted_pipelines = {}
test_predictions = {}

for exp_name, exp in experiments.items():
    cols = exp["num"] + exp["bin"] + exp["cat"]
    X_train = df.loc[train_idx, cols]
    y_train = y.loc[train_idx]
    X_test = df.loc[test_idx, cols]
    y_test = y.loc[test_idx]

    for model_name, model in build_models().items():
        pipe = Pipeline([("prep", make_preprocessor(exp)), ("model", model)])
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1
        )
        cv_rows.append({
            "experiment": exp_name,
            "model": model_name,
            "cv_roc_auc_mean": round(float(cv_res["test_roc_auc"].mean()), 4),
            "cv_roc_auc_std": round(float(cv_res["test_roc_auc"].std()), 4),
            "train_roc_auc_mean": round(float(cv_res["train_roc_auc"].mean()), 4),
            "cv_f1_macro_mean": round(float(cv_res["test_f1_macro"].mean()), 4),
            "cv_f1_mean": round(float(cv_res["test_f1"].mean()), 4),
            "cv_accuracy_mean": round(float(cv_res["test_accuracy"].mean()), 4),
            "cv_precision_mean": round(float(cv_res["test_precision"].mean()), 4),
            "cv_recall_mean": round(float(cv_res["test_recall"].mean()), 4),
        })

        # Fit on full training set, evaluate on untouched 20% test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred

        test_rows.append({
            "experiment": exp_name,
            "model": model_name,
            "test_roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "test_f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "test_f1_aug": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "test_precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        })
        fitted_pipelines[(exp_name, model_name)] = pipe
        test_predictions[(exp_name, model_name)] = (y_pred, y_prob)

cv_df = pd.DataFrame(cv_rows)
test_df = pd.DataFrame(test_rows)
cv_df.to_csv(TAB_DIR / "supervised_cv_results.csv", index=False)
test_df.to_csv(TAB_DIR / "supervised_test_results.csv", index=False)

print("\n5-Fold Stratified CV Results (sorted by Experiment & CV ROC-AUC):")
print(cv_df.sort_values(["experiment", "cv_roc_auc_mean"], ascending=[True, False]).to_string(index=False))

print("\nHold-out Test Set Results (20% stratified, n=104):")
print(test_df.sort_values(["experiment", "test_roc_auc"], ascending=[True, False]).to_string(index=False))

# Identify best models by CV ROC-AUC (avoiding test-set peeking for model selection!)
non_dummy_cv = cv_df[cv_df["model"] != "Dummy Baseline"]
best_behav_cv = non_dummy_cv[non_dummy_cv["experiment"].isin(["E1_Primary", "E2_Primary_Sparse"])].sort_values(
    ["cv_roc_auc_mean", "cv_f1_macro_mean"], ascending=False
).iloc[0]
best_e3_cv = non_dummy_cv[non_dummy_cv["experiment"] == "E3_Combined_Grade"].sort_values(
    ["cv_roc_auc_mean", "cv_f1_macro_mean"], ascending=False
).iloc[0]

BEST_BEHAV_EXP = best_behav_cv["experiment"]
BEST_BEHAV_MODEL = best_behav_cv["model"]
BEST_E3_MODEL = best_e3_cv["model"]

print(f"\nBest Behavioral-Only Model (by CV ROC-AUC): {BEST_BEHAV_EXP} | {BEST_BEHAV_MODEL} "
      f"(CV AUC={best_behav_cv['cv_roc_auc_mean']:.4f}, CV Macro-F1={best_behav_cv['cv_f1_macro_mean']:.4f})")
print(f"Best Combined E3 Model (by CV ROC-AUC): E3_Combined_Grade | {BEST_E3_MODEL} "
      f"(CV AUC={best_e3_cv['cv_roc_auc_mean']:.4f}, CV Macro-F1={best_e3_cv['cv_f1_macro_mean']:.4f})")

# Grade Confounding & Per-Grade Stratified Evaluation (Part III Step 5)
ct_grade_treat = pd.crosstab(df["grade"], df[TARGET_COL])
chi_grade_treat = chi2_cramers_v(ct_grade_treat)
print("\nGrade vs Treatment Association:", chi_grade_treat)

grade_rows = []
for exp_name, model_name in [(BEST_BEHAV_EXP, BEST_BEHAV_MODEL), ("E3_Combined_Grade", BEST_E3_MODEL)]:
    y_pred, y_prob = test_predictions[(exp_name, model_name)]
    test_sub = df.loc[test_idx, ["grade"]].copy()
    test_sub["y_true"] = y.loc[test_idx].values
    test_sub["y_pred"] = y_pred
    test_sub["y_prob"] = y_prob
    for g, grp in test_sub.groupby("grade"):
        auc_g = roc_auc_score(grp["y_true"], grp["y_prob"]) if grp["y_true"].nunique() > 1 else np.nan
        grade_rows.append({
            "experiment": exp_name,
            "model": model_name,
            "grade": int(g),
            "test_n": int(len(grp)),
            "aug_rate": round(float(grp["y_true"].mean()), 3),
            "accuracy": round(float(accuracy_score(grp["y_true"], grp["y_pred"])), 4),
            "f1_macro": round(float(f1_score(grp["y_true"], grp["y_pred"], average="macro")), 4),
            "roc_auc": round(float(auc_g), 4) if not np.isnan(auc_g) else None,
        })

grade_eval_df = pd.DataFrame(grade_rows)
grade_eval_df.to_csv(TAB_DIR / "grade_stratified_results.csv", index=False)
print("\nPer-Grade Test Performance:")
print(grade_eval_df.to_string(index=False))

# Figure 4: Experimental Comparison across E1, E2, E3 (Step 17)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
plot_cv = cv_df[cv_df["model"] != "Dummy Baseline"]
sns.barplot(data=plot_cv, x="model", y="cv_roc_auc_mean", hue="experiment", ax=axes[0], palette="viridis")
axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.50)")
axes[0].set_title("5-Fold CV ROC-AUC Comparison: E1 vs E2 vs E3")
axes[0].set_ylabel("Mean CV ROC-AUC")
axes[0].set_xlabel("")
axes[0].set_ylim(0.45, 0.97)
axes[0].tick_params(axis="x", rotation=22)
axes[0].legend(title="Experiment", fontsize=8.5, loc="lower right")

plot_test = test_df[test_df["model"] != "Dummy Baseline"]
sns.barplot(data=plot_test, x="model", y="test_roc_auc", hue="experiment", ax=axes[1], palette="viridis")
axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Hold-Out Test ROC-AUC Comparison: E1 vs E2 vs E3")
axes[1].set_ylabel("Test ROC-AUC (n=104)")
axes[1].set_xlabel("")
axes[1].set_ylim(0.45, 0.97)
axes[1].tick_params(axis="x", rotation=22)
axes[1].legend(title="Experiment", fontsize=8.5, loc="lower right")

plt.tight_layout()
plt.savefig(FIG_DIR / "04_model_comparison_E1_E2_E3.png", bbox_inches="tight")
plt.close()

# Figure 5: Confusion Matrices & ROC Curves for Best Behavioral & Best E3 Models
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
y_test_arr = y.loc[test_idx].values

for i, (exp_name, model_name, title_prefix) in enumerate([
    (BEST_BEHAV_EXP, BEST_BEHAV_MODEL, "Best Behavioral"),
    ("E3_Combined_Grade", BEST_E3_MODEL, "Best Combined (E3)"),
]):
    y_pred, _ = test_predictions[(exp_name, model_name)]
    cm = confusion_matrix(y_test_arr, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Pred: vanilla", "Pred: aug"],
        yticklabels=["True: vanilla", "True: aug"],
        ax=axes[i], annot_kws={"size": 12}
    )
    axes[i].set_title(f"{title_prefix}\n{exp_name} — {model_name}")

# ROC Curves
for (exp_name, model_name, label, col) in [
    ("E1_Primary", BEST_BEHAV_MODEL if BEST_BEHAV_EXP == "E1_Primary" else "Random Forest", f"E1 ({BEST_BEHAV_MODEL if BEST_BEHAV_EXP == 'E1_Primary' else 'Random Forest'})", "#2b6cb0"),
    (BEST_BEHAV_EXP, BEST_BEHAV_MODEL, f"Best Behav: {BEST_BEHAV_EXP} ({BEST_BEHAV_MODEL})", "#2f855a"),
    ("E3_Combined_Grade", BEST_E3_MODEL, f"Best E3: Combined ({BEST_E3_MODEL})", "#dd6b20"),
]:
    _, y_prob = test_predictions[(exp_name, model_name)]
    fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
    auc_v = roc_auc_score(y_test_arr, y_prob)
    axes[2].plot(fpr, tpr, label=f"{label} (AUC={auc_v:.3f})", linewidth=2)

axes[2].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC=0.500)")
axes[2].set_title("Hold-Out Test ROC Curves")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(FIG_DIR / "05_confusion_matrices_and_roc.png", bbox_inches="tight")
plt.close()

# %% [markdown]
# ## STAGE 4: Explainable AI — SHAP & Feature Importance (Part IV & Step 18 — Shreyansh's Core Focus)

# %%
print("\n" + "=" * 72)
print("STAGE 4: EXPLAINABLE AI (SHAP & FEATURE IMPORTANCE)")
print("=" * 72)

def compute_shap_for_pipeline(pipe, X_train_raw, X_eval_raw, model_name):
    """Compute positive-class (aug=1) SHAP values using the model-appropriate SHAP explainer."""
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["model"]

    X_train_t = prep.transform(X_train_raw)
    X_eval_t = prep.transform(X_eval_raw)
    feat_names = list(prep.get_feature_names_out())

    X_train_df = pd.DataFrame(X_train_t, columns=feat_names, index=X_train_raw.index)
    X_eval_df = pd.DataFrame(X_eval_t, columns=feat_names, index=X_eval_raw.index)

    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_train_df)
        sv = explainer.shap_values(X_eval_df)
        base_val = explainer.expected_value
        explainer_type = "LinearExplainer"
    elif isinstance(clf, (RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, DecisionTreeClassifier)):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_eval_df)
        base_val = explainer.expected_value
        explainer_type = "TreeExplainer"
    else:
        explainer = shap.Explainer(clf.predict_proba, X_train_df)
        exp_obj = explainer(X_eval_df)
        sv = exp_obj.values
        base_val = exp_obj.base_values
        explainer_type = "Permutation/Kernel Explainer"

    # Normalize shape to 2D (n_samples, n_features) for positive class (aug = 1)
    if isinstance(sv, list):
        sv_pos = np.asarray(sv[1])
        bv_pos = float(base_val[1]) if np.ndim(base_val) > 0 else float(base_val)
    elif sv.ndim == 3:
        sv_pos = sv[:, :, 1]
        bv_pos = float(base_val[1]) if np.ndim(base_val) > 0 else float(base_val)
    else:
        sv_pos = np.asarray(sv)
        bv_pos = float(np.mean(base_val)) if np.ndim(base_val) > 0 else float(base_val)

    exp_out = shap.Explanation(
        values=sv_pos,
        base_values=np.full(len(X_eval_df), bv_pos),
        data=X_eval_df.values,
        feature_names=feat_names,
    )
    return exp_out, X_eval_df, feat_names, explainer_type

# 4A. SHAP on Best Behavioral-Only Model (evaluated on full hold-out test set AND full dataset for stable patterns)
behav_exp = experiments[BEST_BEHAV_EXP]
behav_cols = behav_exp["num"] + behav_exp["bin"] + behav_exp["cat"]
behav_pipe = fitted_pipelines[(BEST_BEHAV_EXP, BEST_BEHAV_MODEL)]

shap_behav_test, X_behav_test_df, behav_feat_names, behav_exp_type = compute_shap_for_pipeline(
    behav_pipe, df.loc[train_idx, behav_cols], df.loc[test_idx, behav_cols], BEST_BEHAV_MODEL
)
# Also compute SHAP on all 518 students (using training-fitted pipeline) for robust global directionality check
shap_behav_all, X_behav_all_df, _, _ = compute_shap_for_pipeline(
    behav_pipe, df.loc[train_idx, behav_cols], df[behav_cols], BEST_BEHAV_MODEL
)

# Permutation importance on hold-out test set for comparison with SHAP
perm_res = permutation_importance(
    behav_pipe, df.loc[test_idx, behav_cols], y.loc[test_idx],
    n_repeats=30, scoring="roc_auc", random_state=SEED, n_jobs=-1
)

# Build SHAP summary table (mean |SHAP| + Spearman correlation between feature value and SHAP value for directionality)
shap_behav_rows = []
for j, fname in enumerate(behav_feat_names):
    sv_col = shap_behav_test.values[:, j]
    fv_col = X_behav_test_df[fname].values
    rho_dir = stats.spearmanr(fv_col, sv_col).statistic if np.std(fv_col) > 0 and np.std(sv_col) > 0 else 0.0
    direction = "Higher -> aug (+)" if rho_dir > 0.1 else ("Higher -> vanilla (-)" if rho_dir < -0.1 else "Non-linear / Mixed")
    shap_behav_rows.append({
        "feature": fname,
        "mean_abs_shap_test": round(float(np.abs(sv_col).mean()), 5),
        "mean_abs_shap_all": round(float(np.abs(shap_behav_all.values[:, j]).mean()), 5),
        "shap_feature_spearman_r": round(float(rho_dir), 3),
        "direction_effect": direction,
        "perm_importance_roc_auc": round(float(perm_res.importances_mean[j]), 5) if j < len(perm_res.importances_mean) else np.nan,
    })

shap_behav_df = pd.DataFrame(shap_behav_rows).sort_values("mean_abs_shap_test", ascending=False)
shap_behav_df.to_csv(TAB_DIR / "shap_feature_importance_behavioral.csv", index=False)
print(f"\nSHAP Global Importance — Best Behavioral Model ({BEST_BEHAV_EXP} | {BEST_BEHAV_MODEL} | {behav_exp_type}):")
print(shap_behav_df.to_string(index=False))

# 4B. SHAP on Best Combined E3 Model (with Grade)
e3_exp = experiments["E3_Combined_Grade"]
e3_cols = e3_exp["num"] + e3_exp["bin"] + e3_exp["cat"]
e3_pipe = fitted_pipelines[("E3_Combined_Grade", BEST_E3_MODEL)]

shap_e3_test, X_e3_test_df, e3_feat_names, e3_exp_type = compute_shap_for_pipeline(
    e3_pipe, df.loc[train_idx, e3_cols], df.loc[test_idx, e3_cols], BEST_E3_MODEL
)

shap_e3_rows = []
for j, fname in enumerate(e3_feat_names):
    sv_col = shap_e3_test.values[:, j]
    fv_col = X_e3_test_df[fname].values
    rho_dir = stats.spearmanr(fv_col, sv_col).statistic if np.std(fv_col) > 0 and np.std(sv_col) > 0 else 0.0
    direction = "Higher -> aug (+)" if rho_dir > 0.1 else ("Higher -> vanilla (-)" if rho_dir < -0.1 else "Non-linear / Mixed")
    shap_e3_rows.append({
        "feature": fname,
        "mean_abs_shap_test": round(float(np.abs(sv_col).mean()), 5),
        "shap_feature_spearman_r": round(float(rho_dir), 3),
        "direction_effect": direction,
    })

shap_e3_df = pd.DataFrame(shap_e3_rows).sort_values("mean_abs_shap_test", ascending=False)
shap_e3_df.to_csv(TAB_DIR / "shap_feature_importance_combined.csv", index=False)
print(f"\nSHAP Global Importance — Best Combined E3 Model ({BEST_E3_MODEL} | {e3_exp_type}):")
print(shap_e3_df.head(10).to_string(index=False))

# Figure 6: SHAP Beeswarm Summary Plot (Best Behavioral Model)
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_behav_test, max_display=12, show=False)
plt.title(f"SHAP Summary (Beeswarm) — {BEST_BEHAV_EXP} | {BEST_BEHAV_MODEL} (Positive Class: aug)", pad=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_shap_summary_beeswarm_best_behavioral.png", bbox_inches="tight")
plt.close()

# Figure 7: Side-by-Side SHAP Mean |SHAP| Bar Plot vs Model/Permutation Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
top_behav = shap_behav_df.head(10).iloc[::-1]
axes[0].barh(top_behav["feature"], top_behav["mean_abs_shap_test"], color="#2b6cb0", edgecolor="black", linewidth=0.5)
axes[0].set_title(f"SHAP Global Feature Importance (Mean |SHAP|)\n{BEST_BEHAV_EXP} — {BEST_BEHAV_MODEL}")
axes[0].set_xlabel("Mean |SHAP value| (impact on aug prediction)")

clf_behav = behav_pipe.named_steps["model"]
if hasattr(clf_behav, "feature_importances_"):
    imp_vals = pd.Series(clf_behav.feature_importances_, index=behav_feat_names).sort_values(ascending=False).head(10).iloc[::-1]
    imp_label = "Tree Impurity Feature Importance"
elif hasattr(clf_behav, "coef_"):
    imp_vals = pd.Series(np.abs(clf_behav.coef_[0]), index=behav_feat_names).sort_values(ascending=False).head(10).iloc[::-1]
    imp_label = "|Logistic Regression Coefficient|"
else:
    imp_vals = shap_behav_df.set_index("feature")["perm_importance_roc_auc"].sort_values(ascending=False).head(10).iloc[::-1]
    imp_label = "Permutation Importance (Δ ROC-AUC)"

axes[1].barh(imp_vals.index, imp_vals.values, color="#2f855a", edgecolor="black", linewidth=0.5)
axes[1].set_title(f"{imp_label}\n{BEST_BEHAV_EXP} — {BEST_BEHAV_MODEL}")
axes[1].set_xlabel(imp_label)

plt.tight_layout()
plt.savefig(FIG_DIR / "07_shap_bar_and_model_importance_comparison.png", bbox_inches="tight")
plt.close()

# Figure 8: SHAP Beeswarm Summary Plot (Combined E3 Model with Grade)
plt.figure(figsize=(10, 6.2))
shap.plots.beeswarm(shap_e3_test, max_display=12, show=False)
plt.title(f"SHAP Summary (Beeswarm) — E3 Combined (+Grade) | {BEST_E3_MODEL} (Positive Class: aug)", pad=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "08_shap_summary_beeswarm_E3_combined.png", bbox_inches="tight")
plt.close()

# Figure 9: SHAP Dependence Plots for Top 4 Behavioral Features
top4_feats = list(shap_behav_df["feature"].head(4))
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, fname in zip(axes.ravel(), top4_feats):
    j = behav_feat_names.index(fname)
    x_raw_vals = df.loc[test_idx, fname].values if fname in df.columns else X_behav_test_df[fname].values
    sv_vals = shap_behav_test.values[:, j]
    sc = ax.scatter(
        x_raw_vals, sv_vals,
        c=y.loc[test_idx].values, cmap="coolwarm", alpha=0.75, edgecolors="white", linewidth=0.4, s=42
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_title(f"SHAP Dependence: {fname}")
    ax.set_xlabel(f"Observed {fname}")
    ax.set_ylabel("SHAP value (impact on aug)")
fig.subplots_adjust(right=0.88, hspace=0.38, wspace=0.26)
cbar_ax = fig.add_axes([0.905, 0.20, 0.018, 0.60])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["vanilla (0)", "aug (1)"])
cbar.set_label("Actual Treatment")
fig.suptitle(f"Feature-Level SHAP Effects — Top 4 Features ({BEST_BEHAV_EXP} | {BEST_BEHAV_MODEL})", y=0.98, fontsize=13)
plt.savefig(FIG_DIR / "09_shap_dependence_top_features.png", bbox_inches="tight")
plt.close()

# Figure 10: Individual Student Waterfall Explanations (1 high-prob aug, 1 high-prob vanilla)
_, behav_probs = test_predictions[(BEST_BEHAV_EXP, BEST_BEHAV_MODEL)]
idx_high_aug = int(np.argmax(behav_probs))
idx_high_van = int(np.argmin(behav_probs))
student_aug_id = int(df.loc[test_idx[idx_high_aug], ID_COL])
student_van_id = int(df.loc[test_idx[idx_high_van], ID_COL])

fig = plt.figure(figsize=(10, 5.2))
shap.plots.waterfall(shap_behav_test[idx_high_aug], max_display=9, show=False)
plt.title(f"Individual Explanation — Student {student_aug_id} (True: {df.loc[test_idx[idx_high_aug], TARGET_COL]}, P(aug)={behav_probs[idx_high_aug]:.2f})", pad=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "10a_shap_waterfall_student_aug.png", bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(10, 5.2))
shap.plots.waterfall(shap_behav_test[idx_high_van], max_display=9, show=False)
plt.title(f"Individual Explanation — Student {student_van_id} (True: {df.loc[test_idx[idx_high_van], TARGET_COL]}, P(aug)={behav_probs[idx_high_van]:.2f})", pad=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "10b_shap_waterfall_student_vanilla.png", bbox_inches="tight")
plt.close()

# %% [markdown]
# ## STAGE 5: Additional Statistical Analysis — Treatment, Grade & Prompt Patterns (Part V & Step 19)

# %%
print("\n" + "=" * 72)
print("STAGE 5: STATISTICAL ANALYSIS & PROMPT PATTERN AUDIT")
print("=" * 72)

# 5A. Mann-Whitney U + Rank-Biserial r comparing aug vs vanilla across all 12 interaction features
all_interaction = PRIMARY_FEATURES + SPARSE_RATES
stat_treat_rows = []
pvals_t = []
for col in all_interaction:
    g_aug = df.loc[df[TARGET_COL] == "aug", col].values
    g_van = df.loc[df[TARGET_COL] == "vanilla", col].values
    u_stat, p_val = stats.mannwhitneyu(g_aug, g_van, alternative="two-sided")
    # Rank-biserial correlation: r = 1 - (2U)/(n1*n2); positive when aug > vanilla
    r_rb = (2 * u_stat) / (len(g_aug) * len(g_van)) - 1
    pvals_t.append(p_val)
    stat_treat_rows.append({
        "feature": col,
        "aug_mean": round(float(g_aug.mean()), 4),
        "aug_median": round(float(np.median(g_aug)), 4),
        "vanilla_mean": round(float(g_van.mean()), 4),
        "vanilla_median": round(float(np.median(g_van)), 4),
        "mannwhitney_u": round(float(u_stat), 1),
        "p_value": float(p_val),
        "rank_biserial_r": round(float(r_rb), 4),
    })

# Benjamini-Hochberg FDR adjustment
def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out

fdr_t = bh_fdr(pvals_t)
for row, q in zip(stat_treat_rows, fdr_t):
    row["p_value"] = round(row["p_value"], 6)
    row["fdr_q_value"] = round(float(q), 6)
    row["significant_fdr_05"] = bool(q < 0.05)

stat_treat_df = pd.DataFrame(stat_treat_rows).sort_values("p_value")
stat_treat_df.to_csv(TAB_DIR / "statistical_tests_treatment.csv", index=False)
print("\nMann-Whitney U Tests (aug vs vanilla, BH-FDR corrected):")
print(stat_treat_df[["feature", "aug_median", "vanilla_median", "p_value", "fdr_q_value", "rank_biserial_r", "significant_fdr_05"]].to_string(index=False))

# 5B. Kruskal-Wallis across Grades 9, 10, 11
stat_grade_rows = []
pvals_g = []
for col in PRIMARY_FEATURES:
    groups_g = [df.loc[df["grade"] == g, col].values for g in [9, 10, 11]]
    h_stat, p_val = stats.kruskal(*groups_g)
    # Epsilon-squared effect size for Kruskal-Wallis
    n_tot = len(df)
    eps2 = max(0.0, (h_stat - 2) / (n_tot - 3))
    pvals_g.append(p_val)
    stat_grade_rows.append({
        "feature": col,
        "grade9_median": round(float(np.median(groups_g[0])), 4),
        "grade10_median": round(float(np.median(groups_g[1])), 4),
        "grade11_median": round(float(np.median(groups_g[2])), 4),
        "kruskal_h": round(float(h_stat), 3),
        "p_value": float(p_val),
        "epsilon_sq": round(float(eps2), 4),
    })

fdr_g = bh_fdr(pvals_g)
for row, q in zip(stat_grade_rows, fdr_g):
    row["p_value"] = round(row["p_value"], 6)
    row["fdr_q_value"] = round(float(q), 6)
    row["significant_fdr_05"] = bool(q < 0.05)

stat_grade_df = pd.DataFrame(stat_grade_rows).sort_values("p_value")
stat_grade_df.to_csv(TAB_DIR / "statistical_tests_grade.csv", index=False)
print("\nKruskal-Wallis Tests across Grades 9, 10, 11:")
print(stat_grade_df[["feature", "grade9_median", "grade10_median", "grade11_median", "kruskal_h", "fdr_q_value", "epsilon_sq"]].to_string(index=False))

# 5C. Prompt-level Audit on cleaned_user_messages.csv (28,664 messages)
msg_texts = msgs["message"].fillna("").astype(str)
non_ascii_mask = msg_texts.apply(lambda s: any(ord(ch) > 127 for ch in s))
turkish_chars_mask = msg_texts.str.contains(r"[çğıöşüÇĞİÖŞÜ]", regex=True)
encoding_artefact_mask = msg_texts.str.contains(r"Ã|Ä|Å|â€|Â", regex=True)

behav_msg_cols = [
    "explanation_request", "hint_request", "complete_solution_request",
    "verification_request", "conceptual_question", "refinement_request",
    "has_multiple_questions", "has_code", "has_followup_language"
]
any_label_mask = msgs[behav_msg_cols].sum(axis=1) > 0

prompt_audit = {
    "total_messages": int(len(msgs)),
    "messages_with_any_rule_label_pct": round(float(any_label_mask.mean() * 100), 2),
    "messages_unlabelled_pct": round(float((~any_label_mask).mean() * 100), 2),
    "non_ascii_messages_count": int(non_ascii_mask.sum()),
    "non_ascii_messages_pct": round(float(non_ascii_mask.mean() * 100), 2),
    "turkish_char_messages_count": int(turkish_chars_mask.sum()),
    "encoding_artefact_messages_count": int(encoding_artefact_mask.sum()),
    "high_solution_seekers_95th_pct_threshold": round(float(df["complete_solution_request_rate"].quantile(0.95)), 4),
    "high_solution_seekers_count": int((df["complete_solution_request_rate"] >= df["complete_solution_request_rate"].quantile(0.95)).sum()),
    "session_message_counts": {int(k): int(v) for k, v in msgs["session_id"].value_counts().sort_index().items()},
}
pd.DataFrame([prompt_audit]).to_csv(TAB_DIR / "prompt_audit_summary.csv", index=False)
print("\nPrompt-Level Audit Summary:")
for k, v in prompt_audit.items():
    print(f"  {k}: {v}")

# Figure 11: Top Differentiating Features by Treatment & Grade
top_diff_feats = list(stat_treat_df["feature"].head(4))
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for ax, col in zip(axes.ravel(), top_diff_feats):
    sns.boxplot(data=df, x="grade", y=col, hue=TARGET_COL, ax=ax, palette=["#2b6cb0", "#ed8936"], showfliers=False)
    row_s = stat_treat_df[stat_treat_df["feature"] == col].iloc[0]
    ax.set_title(f"{col}\n(Treatment Mann-Whitney FDR q={row_s['fdr_q_value']:.4f}, r={row_s['rank_biserial_r']:+.2f})")
    ax.set_xlabel("Grade")
    ax.legend(title="Treatment", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "11_treatment_and_grade_patterns.png", bbox_inches="tight")
plt.close()

# Save comprehensive JSON summary for reporting & reproducibility
summary = {
    "environment": {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "shap": shap.__version__,
        "scipy": scipy.__version__,
        "seed": SEED,
    },
    "audit": audit,
    "clustering": {
        "chosen_k": CHOSEN_K,
        "best_k_overall_silhouette": best_k_overall,
        "metrics_table": cluster_eval_rows,
        "algorithm_comparison": algo_comp.to_dict(orient="records"),
        "cluster_names": cluster_names,
        "cluster_sizes": {int(k): int(v) for k, v in cluster_sizes.items()},
        "chi_cluster_vs_treatment": chi_cl_treat,
        "chi_cluster_vs_grade": chi_cl_grade,
    },
    "supervised": {
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "best_behavioral_experiment": BEST_BEHAV_EXP,
        "best_behavioral_model": BEST_BEHAV_MODEL,
        "best_e3_model": BEST_E3_MODEL,
        "chi_grade_vs_treatment": chi_grade_treat,
        "best_behavioral_cv": best_behav_cv.to_dict(),
        "best_e3_cv": best_e3_cv.to_dict(),
        "best_behavioral_test": test_df[(test_df["experiment"] == BEST_BEHAV_EXP) & (test_df["model"] == BEST_BEHAV_MODEL)].iloc[0].to_dict(),
        "best_e3_test": test_df[(test_df["experiment"] == "E3_Combined_Grade") & (test_df["model"] == BEST_E3_MODEL)].iloc[0].to_dict(),
    },
    "shap": {
        "behavioral_explainer": behav_exp_type,
        "e3_explainer": e3_exp_type,
        "top_behavioral_features": shap_behav_df.head(6).to_dict(orient="records"),
        "top_e3_features": shap_e3_df.head(6).to_dict(orient="records"),
        "waterfall_student_aug": {"username": student_aug_id, "prob_aug": round(float(behav_probs[idx_high_aug]), 4), "true_label": str(df.loc[test_idx[idx_high_aug], TARGET_COL])},
        "waterfall_student_vanilla": {"username": student_van_id, "prob_aug": round(float(behav_probs[idx_high_van]), 4), "true_label": str(df.loc[test_idx[idx_high_van], TARGET_COL])},
    },
    "prompt_audit": prompt_audit,
}

with open(OUT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 72)
print("PIPELINE COMPLETE! All figures, tables, and summary_metrics.json saved to:")
print(f"  {OUT_DIR}")
print("=" * 72)
