"""Reproducible text-first analysis layered on existing processed project data.

The script never rewrites ProcessedDatasets or archived legacy outputs.
Run from any working directory: python Research_Analysis/run_research_analysis.py
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             calinski_harabasz_score, davies_bouldin_score,
                             f1_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ProcessedDatasets"
RAW = ROOT / "PreProcessing" / "conversations_audit.csv"
OUT = ROOT / "outputs" / "research"
TABLES, FIGURES, DERIVED = OUT / "tables", OUT / "figures", OUT / "datasets"
SEED = 42
for folder in (TABLES, FIGURES, DERIVED):
    folder.mkdir(parents=True, exist_ok=True)


def patterns(text: str) -> dict[str, int]:
    """Transparent English regex indicators; these are observable text cues."""
    s = str(text).lower()
    rx = {
        "direct_answer_request": r"\b(?:just|only) (?:tell|give|show) me (?:the )?(?:answer|solution|result)\b|\b(?:tell|give|show) me (?:the )?(?:answer|solution|result)\b|\bwhat is the answer\b|\bprovide (?:the )?(?:answer|solution|result)\b|\bjust the answer\b",
        "clarification_request": r"\b(?:i mean|what do you mean|can you clarify|clarify|could you rephrase|say that again)\b",
        "example_request": r"\b(?:give|show|provide|need) (?:me )?(?:an? )?example\b|\bexample of\b",
        "mistake_explanation_request": r"\b(?:where|what) did i (?:do|go) wrong\b|\bexplain my mistake\b|\bwhy is my (?:answer|solution|work) wrong\b",
        "alternative_method_request": r"\bcan you .{0,35}(?:another|alternative|different) (?:way|method|approach)\b|\bcould you .{0,35}(?:another|alternative|different) (?:way|method|approach)\b|\bis there (?:another|a different|an alternative) (?:way|method|approach)\b|\b(?:another|alternative) way to solve\b",
        "independent_attempt": r"\b(?:i tried|i got|my answer|i think|i calculated|i solved|i started|i did)\b|(?:\d\s*[+*/=\-]\s*\d)",
        "reasoning_explanation": r"\b(?:because|therefore|so that|this means|my reasoning|i thought|since)\b",
        "specific_difficulty": r"\b(?:i don't understand|i do not understand|i'm confused|i am confused|i am stuck|i'm stuck|where should i start|which step do i|i can't proceed|i cannot proceed)\b",
        "why_correct_request": r"\bwhy (?:is|are|does|did) (?:that|this|it|the answer|my answer|the solution) (?:correct|right|true|work)\b|\bhow do you know (?:that|this|the answer)\b|\bwhy does that work\b",
        "approach_comparison": r"\b(?:compare|difference between|versus|vs\.?|which method|better method)\b",
        "challenge_or_verification": r"\b(?:check|verify|is this correct|are you sure|that seems wrong|i think that's wrong)\b",
        "builds_on_explanation": r"\b(?:next step|based on that|what about|can you explain that|following that)\b",
    }
    return {k: int(bool(re.search(v, s))) for k, v in rx.items()}


def text_features(message: str) -> dict[str, float | int]:
    s = str(message)
    words = re.findall(r"\b[\w']+\b", s.lower(), flags=re.UNICODE)
    unique = len(set(words))
    sentences = len(re.findall(r"[.!?]+", s))
    math_terms = re.findall(r"\b(?:equation|factor|multiple|prime|integer|fraction|ratio|variable|coefficient|polynomial|function|probability|theorem|angle|area|perimeter|slope|matrix|derivative|sum|product|quotient|expression|solve|simplify)\w*\b", s.lower())
    out = {
        "lexical_diversity": unique / len(words) if words else 0.0,
        "mean_token_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "mean_words_per_sentence": len(words) / sentences if sentences else float(len(words)),
        "clause_marker_count": len(re.findall(r"\b(?:because|although|however|therefore|while|since|if|then)\b", s.lower())),
        "math_terminology_count": len(math_terms),
        "math_terminology_present": int(bool(math_terms)),
        "punctuation_count": sum(s.count(c) for c in ",;:!?"),
    }
    out.update(patterns(s))
    return out


def response_features(message: str) -> dict[str, int]:
    """Observable response style cues on actual assistant messages; labels may overlap."""
    s = str(message).lower()
    tests = {
        "direct_answer": r"\b(?:the answer is|therefore,? the answer|final answer|so the value is|equals?)\b|\$\\boxed",
        "hint_or_scaffold": r"\b(?:hint|try|consider|start by|begin by|look at|what if you|think about)\b",
        "explanation": r"\b(?:because|this means|this is because|the reason|we can see|therefore|since)\b",
        "guided_question": r"\b(?:what do you think|can you identify|could you identify|what is your next step|how might you|what happens if|which step would you)\b",
    }
    return {f"response_{name}": int(bool(re.search(pattern, s))) for name, pattern in tests.items()}


def bh_fdr(pvalues: list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adj)
    result[order] = np.clip(adj, 0, 1)
    return result


def class_clustered_ols(y: np.ndarray, X: np.ndarray, classes: np.ndarray):
    """OLS CR1 covariance with classroom as the independent assignment cluster."""
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("Rank-deficient class-clustered design")
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    residual = y - X @ beta
    class_ids = np.unique(classes)
    meat = np.zeros_like(bread)
    for class_id in class_ids:
        mask = classes == class_id
        score = X[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k, g = len(y), X.shape[1], len(class_ids)
    covariance = (g / (g - 1)) * ((n - 1) / max(n - k, 1)) * bread @ meat @ bread
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_values = beta / np.where(standard_errors > 0, standard_errors, np.nan)
    p_values = 2 * stats.t.sf(np.abs(t_values), df=g - 1)
    critical = stats.t.ppf(.975, df=g - 1)
    confidence_intervals = np.column_stack([beta - critical * standard_errors,
                                            beta + critical * standard_errors])
    return beta, standard_errors, p_values, g, confidence_intervals


def bootstrap_median_difference(a, b, *, seed: int = SEED, n_resamples: int = 1000) -> tuple[float, float, float]:
    """Independent-student bootstrap CI for median(a) - median(b)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        diffs[i] = np.median(rng.choice(a, size=len(a), replace=True)) - np.median(rng.choice(b, size=len(b), replace=True))
    return float(np.median(a) - np.median(b)), float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    config = json.loads((DATA / "preprocessing_config.json").read_text(encoding="utf-8"))
    outcome_source = ROOT / "AcademicOutcomes" / "matched_session_outcomes.csv"
    students = pd.read_csv(DATA / "student_features_final.csv")
    messages = pd.read_csv(DATA / "cleaned_user_messages.csv", low_memory=False)
    required = {"username", "message", "conversation_id", "session_id", "problem_id", "time_stamp", "treatment", "grade"}
    missing = required - set(messages.columns)
    if missing:
        raise ValueError(f"Required processed message columns missing: {sorted(missing)}")
    if students.username.duplicated().any():
        raise ValueError("Student feature dataset must have one row per username")
    if set(students.username) != set(messages.username.unique()):
        raise ValueError("Student/message username sets differ; refusing unvalidated join")
    original_message_count = len(messages)
    features = pd.DataFrame([text_features(s) for s in messages.message.fillna("")])
    msg = pd.concat([messages.reset_index(drop=True), features], axis=1)
    msg["time_stamp_parsed"] = pd.to_datetime(msg.time_stamp, errors="coerce", dayfirst=False)
    msg.to_csv(DERIVED / "message_text_features.csv", index=False)

    # Validate old student features against the message records; preserve the canonical source fields.
    # Message-level 0/1 text cues become student-level rates; numeric linguistic measures become means.
    aggregation = {c: "mean" for c in features.columns}
    student_new = msg.groupby("username").agg(aggregation)
    student_new.columns = [f"text_{c}_mean" for c in student_new.columns]
    student_new["text_mean_prompt_similarity_previous"] = np.nan

    # Message sequence is ordered by source-row index where available; timestamps are preserved and parsed.
    sort_cols = [c for c in ["conversation_id", "unnamed:_0"] if c in msg.columns]
    if len(sort_cols) < 2:
        sort_cols = ["conversation_id", "time_stamp_parsed"]
    ordered = msg.sort_values(sort_cols, kind="stable").copy()
    prior = ordered.groupby("conversation_id").message.shift(1)
    # Character/token Jaccard similarity gives a transparent, local similarity measure without external models.
    def jaccard_pair(a, b):
        aa, bb = set(re.findall(r"\w+", str(a).lower())), set(re.findall(r"\w+", str(b).lower()))
        return len(aa & bb) / len(aa | bb) if aa | bb else 0.0
    ordered["previous_prompt_token_jaccard"] = [jaccard_pair(a, b) if pd.notna(a) else np.nan for a, b in zip(prior, ordered.message)]
    msg.loc[ordered.index, "previous_prompt_token_jaccard"] = ordered.previous_prompt_token_jaccard
    msg.to_csv(DERIVED / "message_text_features.csv", index=False)
    sim = ordered.loc[ordered.previous_prompt_token_jaccard.notna()].groupby("username").previous_prompt_token_jaccard.mean()
    student_new["text_mean_prompt_similarity_previous"] = sim
    student_dataset = students.merge(student_new.reset_index(), how="left", on="username", validate="one_to_one")
    student_dataset.to_csv(DERIVED / "student_features_with_text.csv", index=False)

    # Nested analytic levels, derived only after message-level text extraction.
    num_cols = list(features.columns) + ["previous_prompt_token_jaccard"]
    def aggregate_level(keys: list[str], name: str) -> pd.DataFrame:
        aggregation = {
            "student_message_count": ("message", "size"),
            "average_prompt_length": ("prompt_length", "mean"),
            "average_word_count": ("word_count", "mean"),
            "average_sentence_count": ("sentence_count", "mean"),
            "average_question_count": ("question_count", "mean"),
            "grade": ("grade", "first"), "treatment": ("treatment", "first"),
            "conversation_count": ("conversation_id", "nunique"),
            "first_timestamp": ("time_stamp_parsed", "min"),
            "last_timestamp": ("time_stamp_parsed", "max"),
        }
        # Existing labels become rates at the session/problem grain, matching the saved config.
        label_rates = {
            "explanation_request_rate": "explanation_request", "hint_request_rate": "hint_request",
            "complete_solution_request_rate": "complete_solution_request", "verification_request_rate": "verification_request",
            "conceptual_question_rate": "conceptual_question", "refinement_request_rate": "refinement_request",
            "multiple_question_rate": "has_multiple_questions", "code_prompt_rate": "has_code",
            "followup_prompt_rate": "has_followup_language",
        }
        aggregation.update({out: (source, "mean") for out, source in label_rates.items()})
        aggregation.update({"hint_request_present": ("hint_request", "max"),
                            "refinement_request_present": ("refinement_request", "max"),
                            "multiple_question_present": ("has_multiple_questions", "max")})
        aggregation.update({f"text_{c}_mean": (c, "mean") for c in num_cols})
        agg = msg.groupby(keys, dropna=False).agg(**aggregation).reset_index()
        agg.to_csv(DERIVED / name, index=False)
        return agg
    session = aggregate_level(["username", "session_id"], "student_session_features.csv")
    problem = aggregate_level(["username", "problem_id"], "student_problem_features.csv")
    conv_user = msg.groupby("conversation_id").agg(
        username=("username", "first"), session_id=("session_id", "first"), problem_id=("problem_id", "first"),
        grade=("grade", "first"), treatment=("treatment", "first"), student_message_count=("message", "size"),
        first_timestamp=("time_stamp_parsed", "min"), last_timestamp=("time_stamp_parsed", "max"),
        student_messages=("message", lambda x: json.dumps(list(x), ensure_ascii=False)),
        **{f"{c}_mean": (c, "mean") for c in num_cols},
    ).reset_index()
    conv_user.to_csv(DERIVED / "conversation_student_features.csv", index=False)

    # Raw full conversation exists and includes system, user, and assistant rows. Preserve all turns and adjacency.
    turn_summary = {"available": RAW.exists()}
    if RAW.exists():
        raw = pd.read_csv(RAW, low_memory=False)
        if "Unnamed: 0" in raw:
            raw = raw.sort_values("Unnamed: 0", kind="stable")
        raw["time_stamp_parsed"] = pd.to_datetime(raw.time_stamp, errors="coerce", dayfirst=False)
        raw["message_order_in_conversation"] = raw.groupby("conversation_id").cumcount()
        raw["previous_role"] = raw.groupby("conversation_id").role.shift()
        raw["previous_message"] = raw.groupby("conversation_id").message.shift()
        raw["next_role"] = raw.groupby("conversation_id").role.shift(-1)
        raw["next_message"] = raw.groupby("conversation_id").message.shift(-1)
        raw.to_csv(DERIVED / "conversation_turns_with_context.csv", index=False)
        assistant = raw.loc[raw.role.eq("assistant")].copy()
        asst_feats = pd.DataFrame([response_features(s) for s in assistant.message.fillna("")], index=assistant.index)
        assistant = pd.concat([assistant, asst_feats], axis=1)
        assistant["response_word_count"] = assistant.message.fillna("").str.findall(r"\b\w+\b").str.len()
        assistant.to_csv(DERIVED / "assistant_response_features.csv", index=False)
        adjacent = assistant.loc[assistant.next_role.eq("user") & assistant.next_message.notna()].copy()
        follow_features = pd.DataFrame([text_features(s) for s in adjacent.next_message], index=adjacent.index).add_prefix("followup_")
        adjacent = pd.concat([adjacent, follow_features], axis=1)
        adjacent["followup_word_count"] = adjacent.next_message.str.findall(r"\b\w+\b").str.len()
        adjacent["followup_question_mark_count"] = adjacent.next_message.str.count(r"\?")
        adjacent.to_csv(DERIVED / "assistant_student_followup_pairs.csv", index=False)
        style_cols = list(asst_feats.columns)
        long_pairs = pd.concat([
            adjacent.loc[adjacent[style].eq(1), ["username", "treatment", "conversation_id", style, "followup_word_count",
                                                  "followup_question_mark_count", "followup_independent_attempt",
                                                  "followup_reasoning_explanation", "followup_specific_difficulty"]]
            .assign(response_type=style) for style in style_cols
        ], ignore_index=True)
        follow_by_student = long_pairs.groupby(["username", "treatment", "response_type"]).agg(
            pair_count=("conversation_id", "size"), mean_followup_word_count=("followup_word_count", "mean"),
            question_followup_rate=("followup_question_mark_count", lambda x: (x > 0).mean()),
            attempt_cue_rate=("followup_independent_attempt", "mean"),
            reasoning_cue_rate=("followup_reasoning_explanation", "mean"),
            difficulty_cue_rate=("followup_specific_difficulty", "mean"),
        ).reset_index()
        follow_by_student.to_csv(DERIVED / "student_followup_by_response_type.csv", index=False)
        follow_summary = follow_by_student.groupby("response_type").agg(
            student_count=("username", "nunique"), mean_student_followup_words=("mean_followup_word_count", "mean"),
            median_student_followup_words=("mean_followup_word_count", "median"),
            mean_question_followup_rate=("question_followup_rate", "mean"),
            mean_attempt_cue_rate=("attempt_cue_rate", "mean"),
            mean_reasoning_cue_rate=("reasoning_cue_rate", "mean"),
        ).reset_index()
        follow_summary.to_csv(TABLES / "student_followup_after_response_summary.csv", index=False)
        response_cols = [c for c in asst_feats.columns]
        student_assistant = assistant.groupby("username").agg(
            treatment=("treatment", "first"), grade=("grade", "first"), assistant_turn_count=("message", "size"),
            mean_response_word_count=("response_word_count", "mean"),
            **{f"{c}_rate": (c, "mean") for c in response_cols},
        ).reset_index()
        student_assistant.to_csv(DERIVED / "student_assistant_response_features.csv", index=False)
        response_test_rows, response_ps = [], []
        for col in ["mean_response_word_count"] + [f"{c}_rate" for c in response_cols]:
            aug = student_assistant.loc[student_assistant.treatment.eq("aug"), col].dropna()
            van = student_assistant.loc[student_assistant.treatment.eq("vanilla"), col].dropna()
            if aug.nunique() == 1 and van.nunique() == 1 and aug.iloc[0] == van.iloc[0]:
                u, pvalue = len(aug) * len(van) / 2, 1.0
            else:
                u, pvalue = stats.mannwhitneyu(aug, van, alternative="two-sided", method="asymptotic")
            med_diff, ci_low, ci_high = bootstrap_median_difference(aug, van, seed=SEED + len(response_test_rows))
            response_test_rows.append({"feature": col, "aug_median": aug.median(), "vanilla_median": van.median(),
                                       "median_difference_aug_minus_vanilla": med_diff, "median_difference_ci95_low": ci_low,
                                       "median_difference_ci95_high": ci_high,
                                       "rank_biserial_r": 2 * u / (len(aug) * len(van)) - 1, "p_value": pvalue})
            response_ps.append(pvalue)
        response_qs = bh_fdr(response_ps)
        for row, qvalue in zip(response_test_rows, response_qs): row["fdr_q_value"] = qvalue
        pd.DataFrame(response_test_rows).to_csv(TABLES / "assistant_response_treatment_tests.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        response_rates = [f"{c}_rate" for c in response_cols]
        treatment_order = ["vanilla", "aug"]
        colors = {"vanilla": "#3B82A0", "aug": "#D17A22"}
        x = np.arange(len(response_rates))
        width = .36
        for j, treatment in enumerate(treatment_order):
            medians = [student_assistant.loc[student_assistant.treatment.eq(treatment), c].median() for c in response_rates]
            axes[0].bar(x + (j - .5) * width, medians, width, label=treatment, color=colors[treatment])
        axes[0].set_xticks(x, [c.replace("response_", "").replace("_rate", "").replace("_", "\n") for c in response_rates], rotation=0)
        axes[0].set_ylabel("Median response-style rate per student")
        axes[0].set_title("Assistant response cues by treatment")
        axes[0].legend(title="Treatment")
        for j, treatment in enumerate(treatment_order):
            vals = student_assistant.loc[student_assistant.treatment.eq(treatment), "mean_response_word_count"]
            axes[1].boxplot(vals, positions=[j + 1], widths=.55, patch_artist=True,
                            boxprops={"facecolor": colors[treatment], "alpha": .65}, medianprops={"color": "black"},
                            flierprops={"marker": ".", "markersize": 2, "alpha": .25})
        axes[1].set_xticks([1, 2], treatment_order)
        axes[1].set_ylabel("Mean assistant response words per student")
        axes[1].set_title("Assistant response length")
        fig.tight_layout(); fig.savefig(FIGURES / "assistant_response_treatment_comparison.png", dpi=180); plt.close(fig)
        turn_summary.update({"rows": len(raw), "role_counts": raw.role.value_counts().to_dict(), "assistant_rows": len(assistant),
                             "assistant_followed_by_user_pairs": len(adjacent),
                             "assistant_response_categories": list(asst_feats.columns),
                             "assistant_response_flag_counts": {c: int(assistant[c].sum()) for c in response_cols},
                             "raw_user_rows": int(raw.role.eq('user').sum())})

    # Sampled review sheet: automated semantic labels are explicitly reviewable, not ground truth.
    review = msg.sample(min(150, len(msg)), random_state=SEED)[["username", "conversation_id", "message"] + list(patterns(""))].copy()
    review.insert(3, "manual_review_label", "")
    review.insert(4, "reviewer_notes", "")
    review.to_csv(TABLES / "annotation_review_sample.csv", index=False)

    # Text themes via TF-IDF and NMF, no external model/API. English-oriented and lexical.
    corpus = msg.message.fillna("").astype(str)
    # Topic-only copy removes known encoding artifacts; source `message` is never modified.
    topic_corpus = corpus.map(lambda s: re.sub(r"\b(?:\w*u0131\w*|\w*ufffd\w*|nu\d+)\b", " ", s))
    topic_stop_words = sorted(set(ENGLISH_STOP_WORDS) | {"u0131", "ufffd", "nu", "mathbf"})
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=5,
                                 max_df=0.80, max_features=3000, token_pattern=r"(?u)\b\w\w+\b",
                                 stop_words=topic_stop_words)
    try:
        tfidf = vectorizer.fit_transform(topic_corpus)
        n_topics = min(8, max(2, tfidf.shape[0] // 500), tfidf.shape[1] - 1)
        nmf = NMF(n_components=n_topics, init="nndsvda", random_state=SEED, max_iter=400)
        topic_values = nmf.fit_transform(tfidf)
        topic_totals = topic_values.sum(axis=1, keepdims=True)
        topic_valid = topic_totals[:, 0] > 0
        terms = np.asarray(vectorizer.get_feature_names_out())
        topic_rows = []
        for i, comp in enumerate(nmf.components_):
            top = terms[np.argsort(comp)[-12:][::-1]].tolist()
            dominant = np.where(topic_valid, topic_values.argmax(axis=1), -1)
            topic_rows.append({"topic": i, "top_terms": ", ".join(top), "message_count": int((dominant == i).sum())})
        pd.DataFrame(topic_rows).to_csv(TABLES / "lexical_nmf_topics.csv", index=False)
        topic_shares = np.divide(topic_values, topic_totals, out=np.full_like(topic_values, np.nan), where=topic_totals > 0)
        pd.DataFrame({"message_row": msg.index, "username": msg.username, "conversation_id": msg.conversation_id,
                      "dominant_lexical_topic": np.where(topic_valid, topic_values.argmax(axis=1), -1),
                      "topic_strength": topic_values.max(axis=1)}).to_csv(DERIVED / "message_lexical_topics.csv", index=False)
        topic_message = pd.DataFrame(topic_shares, columns=[f"topic_{i}_share" for i in range(n_topics)])
        topic_message["username"] = msg.username.to_numpy()
        topic_student = topic_message.groupby("username").mean().reset_index().merge(
            students[["username", "treatment", "grade"]], on="username", how="left", validate="one_to_one")
        topic_student.to_csv(DERIVED / "student_lexical_topic_profiles.csv", index=False)
        topic_tests, topic_ps = [], []
        for col in [c for c in topic_student if c.startswith("topic_")]:
            aug = topic_student.loc[topic_student.treatment.eq("aug"), col].dropna()
            van = topic_student.loc[topic_student.treatment.eq("vanilla"), col].dropna()
            if not len(aug) or not len(van):
                continue
            if aug.nunique() == 1 and van.nunique() == 1 and aug.iloc[0] == van.iloc[0]:
                u, pvalue = len(aug) * len(van) / 2, 1.0
            else:
                u, pvalue = stats.mannwhitneyu(aug, van, alternative="two-sided", method="asymptotic")
            med_diff, ci_low, ci_high = bootstrap_median_difference(aug, van, seed=SEED + len(topic_tests))
            topic_tests.append({"topic_share": col, "aug_mean": aug.mean(), "vanilla_mean": van.mean(),
                                "median_difference_aug_minus_vanilla": med_diff, "median_difference_ci95_low": ci_low,
                                "median_difference_ci95_high": ci_high,
                                "rank_biserial_r": 2 * u / (len(aug) * len(van)) - 1, "p_value": pvalue})
            topic_ps.append(pvalue)
        topic_qs = bh_fdr(topic_ps)
        for row, qvalue in zip(topic_tests, topic_qs): row["fdr_q_value"] = qvalue
        pd.DataFrame(topic_tests).to_csv(TABLES / "lexical_topic_treatment_comparison.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(n_topics)
        width = .36
        for j, treatment in enumerate(["vanilla", "aug"]):
            vals = topic_student.loc[topic_student.treatment.eq(treatment), [f"topic_{i}_share" for i in range(n_topics)]]
            means = vals.mean().to_numpy()
            sems = vals.sem().to_numpy()
            ax.bar(x + (j - .5) * width, means, width, yerr=sems, capsize=3,
                   label=treatment, color={"vanilla": "#3B82A0", "aug": "#D17A22"}[treatment])
        ax.set_xticks(x, [f"Topic {i}" for i in range(n_topics)])
        ax.set_ylabel("Mean student topic proportion (± SE)")
        ax.set_title("Lexical topic composition by treatment")
        ax.legend(title="Treatment")
        fig.tight_layout(); fig.savefig(FIGURES / "lexical_topics_by_treatment.png", dpi=180); plt.close(fig)
        nmf_status = {"algorithm": "TF-IDF + NMF", "topics": n_topics, "vocabulary_size": int(tfidf.shape[1]), "external_pretrained_model": False}
    except ValueError as exc:
        nmf_status = {"available": False, "reason": str(exc)}

    # Existing canonical feature set + selective, interpretable text signals. No treatment/outcomes in clustering inputs.
    primary = list(config["primary_features"])
    added_cluster = ["text_lexical_diversity_mean", "text_mean_words_per_sentence_mean", "text_math_terminology_present_mean",
                     "text_independent_attempt_mean", "text_reasoning_explanation_mean", "text_specific_difficulty_mean",
                     "text_why_correct_request_mean", "text_direct_answer_request_mean", "text_challenge_or_verification_mean"]
    cluster_cols = primary + added_cluster
    X = student_dataset[cluster_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median()).fillna(0)
    # Honor saved transformation/scaling for canonical features; leave proportions/text signals untransformed.
    X.loc[:, primary] = np.log1p(X[primary].clip(lower=0))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    metrics = []
    labels_by_k = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(X_scaled)
        labels_by_k[k] = km.labels_
        metrics.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X_scaled, km.labels_),
                        "davies_bouldin": davies_bouldin_score(X_scaled, km.labels_), "calinski_harabasz": calinski_harabasz_score(X_scaled, km.labels_)})
    pd.DataFrame(metrics).to_csv(TABLES / "cluster_k_selection_metrics.csv", index=False)
    # Select k=4 to maintain interpretability and comparability with the existing pipeline; report its metrics.
    k_chosen = 4
    km = KMeans(n_clusters=k_chosen, random_state=SEED, n_init=20).fit(X_scaled)
    agg = AgglomerativeClustering(n_clusters=k_chosen, linkage="ward").fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=SEED).fit_transform(X_scaled)
    student_dataset["kmeans_cluster"] = km.labels_
    student_dataset["agglomerative_cluster"] = agg
    student_dataset["pca_1"] = pca[:, 0]
    student_dataset["pca_2"] = pca[:, 1]
    student_dataset.to_csv(DERIVED / "student_features_with_text.csv", index=False)
    profiles = student_dataset.groupby("kmeans_cluster")[cluster_cols].mean().reset_index()
    profiles.to_csv(TABLES / "cluster_feature_profiles.csv", index=False)
    cluster_hypotheses = pd.DataFrame([
        {"kmeans_cluster": 0, "profile_hypothesis": "Higher verification and complete-solution request rates", "basis": "Observed cluster means in verification_request_rate and complete_solution_request_rate"},
        {"kmeans_cluster": 1, "profile_hypothesis": "Higher message volume with shorter, lower-intent prompts", "basis": "Observed cluster means in student_message_count and average_word_count; low intent rates"},
        {"kmeans_cluster": 2, "profile_hypothesis": "Conceptual and solution-seeking prompts with math terminology", "basis": "Observed cluster means in conceptual_question_rate, complete_solution_request_rate and math terminology"},
        {"kmeans_cluster": 3, "profile_hypothesis": "Longer math-heavy prompts with more code", "basis": "Observed cluster means in average_word_count, code_prompt_rate and math terminology"},
    ])
    cluster_hypotheses.to_csv(TABLES / "cluster_profile_hypotheses.csv", index=False)
    cluster_group = pd.crosstab(student_dataset.kmeans_cluster, student_dataset.treatment).reset_index()
    cluster_group.to_csv(TABLES / "cluster_by_treatment.csv", index=False)
    pd.crosstab(student_dataset.kmeans_cluster, student_dataset.grade).to_csv(TABLES / "cluster_by_grade.csv")
    algo = []
    for name, lab in [("KMeans", km.labels_), ("Agglomerative_Ward", agg)]:
        algo.append({"algorithm": name, "k": k_chosen, "silhouette": silhouette_score(X_scaled, lab),
                     "davies_bouldin": davies_bouldin_score(X_scaled, lab), "calinski_harabasz": calinski_harabasz_score(X_scaled, lab)})
    pd.DataFrame(algo).to_csv(TABLES / "cluster_algorithm_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca[:, 0], pca[:, 1], c=km.labels_, cmap="tab10", alpha=.72, s=24)
    ax.set(title="Student interaction profiles (PCA of scaled interaction + text features)", xlabel="PCA 1", ylabel="PCA 2")
    fig.colorbar(scatter, ax=ax, label="K-Means cluster")
    fig.tight_layout(); fig.savefig(FIGURES / "student_profiles_pca.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([m["k"] for m in metrics], [m["inertia"] for m in metrics], marker="o")
    ax.set(title="K-Means elbow curve", xlabel="Number of clusters (k)", ylabel="Within-cluster sum of squares")
    fig.tight_layout(); fig.savefig(FIGURES / "kmeans_elbow.png", dpi=180); plt.close(fig)

    # Group treatment comparisons use per-student rows; no repeated-message pseudo-replication.
    compare = list(dict.fromkeys(primary + config["supplementary_rate_features"] +
                                 config["supplementary_binary_features"] + list(student_new.columns)))
    rows, pvals = [], []
    for col in compare:
        aug = pd.to_numeric(student_dataset.loc[student_dataset.treatment.eq("aug"), col], errors="coerce").dropna()
        van = pd.to_numeric(student_dataset.loc[student_dataset.treatment.eq("vanilla"), col], errors="coerce").dropna()
        if not len(aug) or not len(van):
            continue
        if aug.nunique() == 1 and van.nunique() == 1 and aug.iloc[0] == van.iloc[0]:
            stat, p = (len(aug) * len(van)) / 2, 1.0
        else:
            stat, p = stats.mannwhitneyu(aug, van, alternative="two-sided", method="asymptotic")
        # Rank-biserial effect size; U here counts aug ranks above vanilla.
        rb = 2 * stat / (len(aug) * len(van)) - 1
        med_diff, ci_low, ci_high = bootstrap_median_difference(aug, van, seed=SEED + len(rows))
        rows.append({"feature": col, "aug_n": len(aug), "aug_median": aug.median(), "vanilla_n": len(van),
                     "vanilla_median": van.median(), "median_difference_aug_minus_vanilla": med_diff,
                     "median_difference_ci95_low": ci_low, "median_difference_ci95_high": ci_high,
                     "mann_whitney_u": stat, "p_value": p, "rank_biserial_r": rb})
        pvals.append(p)
    if rows:
        p_array = np.asarray(pvals, dtype=float)
        qvals = np.full(len(p_array), np.nan)
        valid = np.isfinite(p_array)
        if valid.any():
            qvals[valid] = bh_fdr(p_array[valid].tolist())
        for row, q in zip(rows, qvals): row["fdr_q_value"] = q
        pd.DataFrame(rows).sort_values("fdr_q_value").to_csv(TABLES / "treatment_text_feature_tests.csv", index=False)
        effect_data = pd.DataFrame(rows).sort_values("rank_biserial_r", key=lambda col: col.abs(), ascending=False).head(12).sort_values("median_difference_aug_minus_vanilla")
        fig, ax = plt.subplots(figsize=(10, 7))
        y_pos = np.arange(len(effect_data))
        lower = effect_data["median_difference_aug_minus_vanilla"] - effect_data["median_difference_ci95_low"]
        upper = effect_data["median_difference_ci95_high"] - effect_data["median_difference_aug_minus_vanilla"]
        ax.errorbar(effect_data["median_difference_aug_minus_vanilla"], y_pos, xerr=[lower, upper], fmt="o", color="#315A7D", capsize=3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos, effect_data.feature)
        ax.set_xlabel("Median difference (aug − vanilla), bootstrap 95% CI")
        ax.set_title("Student-level text and interaction differences")
        fig.tight_layout(); fig.savefig(FIGURES / "treatment_text_feature_effects.png", dpi=180); plt.close(fig)

    # Treatment classification remains secondary, student-level, stratified CV; not an outcome model.
    X_sup = student_dataset[primary + [c for c in student_new.columns]].apply(pd.to_numeric, errors="coerce")
    y = student_dataset.treatment.eq("aug").astype(int)
    classifier_grouping = "student-level stratified folds (no class key available)"
    classifier_groups = None
    if outcome_source.is_file():
        class_rows = pd.read_csv(outcome_source, usecols=["username", "Class", "Treatment arm"])
        if class_rows.groupby("username").Class.nunique().max() != 1:
            raise ValueError("A student has multiple classes in the public outcome source")
        class_map = class_rows.groupby("username", as_index=False).agg(
            Class=("Class", "first"), source_treatment=("Treatment arm", "first"))
        class_map["source_treatment"] = class_map.source_treatment.map({"vanilla": "vanilla", "augmented": "aug"})
        classifier_frame = student_dataset[["username", "treatment"]].merge(
            class_map, on="username", how="left", validate="one_to_one")
        if classifier_frame.Class.isna().any() or not classifier_frame.treatment.eq(classifier_frame.source_treatment).all():
            raise ValueError("Class/treatment keys do not validate for grouped treatment classification")
        classifier_groups = classifier_frame.Class.astype(str).to_numpy()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        classifier_grouping = "five-fold class-held-out StratifiedGroupKFold"
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "LogisticRegression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1)),
        "GradientBoosting": make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(n_estimators=100, learning_rate=.05, max_depth=2, random_state=SEED)),
    }
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro", "roc_auc": "roc_auc"}
    model_rows = []
    for name, model in models.items():
        res = cross_validate(model, X_sup, y, cv=cv, groups=classifier_groups,
                             scoring=scoring, n_jobs=1)
        model_rows.append({"model": name, **{f"{m}_mean": float(np.mean(res[f"test_{m}"])) for m in scoring},
                           **{f"{m}_sd": float(np.std(res[f"test_{m}"])) for m in scoring}})
    pd.DataFrame(model_rows).to_csv(TABLES / "treatment_classifier_cv.csv", index=False)

    # Inferential student-level treatment associations use classroom clusters,
    # because the study assigned GPT Base/Tutor by class. The raw Mann-Whitney
    # table above remains a descriptive sensitivity analysis.
    class_clustered_summary = {"available": False, "reason": "matched academic source not present"}
    if outcome_source.is_file():
        outcome_classes = pd.read_csv(outcome_source, usecols=["username", "Class", "Treatment arm"])
        if outcome_classes.groupby("username").Class.nunique().max() != 1:
            raise ValueError("A student has multiple classes in the public outcome source")
        if outcome_classes.groupby("username")["Treatment arm"].nunique().max() != 1:
            raise ValueError("A student has multiple treatment assignments in the public outcome source")
        if outcome_classes.groupby("Class")["Treatment arm"].nunique().max() != 1:
            raise ValueError("A classroom has mixed treatment assignment in the public outcome source")
        class_lookup = outcome_classes.groupby("username", as_index=False).agg(
            Class=("Class", "first"), source_treatment=("Treatment arm", "first"))
        class_lookup["source_treatment"] = class_lookup.source_treatment.map({"vanilla": "vanilla", "augmented": "aug"})
        if class_lookup.source_treatment.isna().any():
            raise ValueError("Unexpected public treatment value in class-clustered text analysis")
        inference_frame = student_dataset[["username", "treatment", "grade"] + compare].merge(
            class_lookup, on="username", how="left", validate="one_to_one")
        if inference_frame.Class.isna().any() or not inference_frame.treatment.eq(inference_frame.source_treatment).all():
            raise ValueError("Class/treatment lookup does not exactly match local student rows")
        # Assistant response features are from actual assistant turns and remain
        # distinct from the student-prompt features.
        response_path = DERIVED / "student_assistant_response_features.csv"
        response_cols = []
        if response_path.is_file():
            response_frame = pd.read_csv(response_path)
            if response_frame.username.duplicated().any():
                raise ValueError("Assistant response student table is not unique by username")
            response_cols = [c for c in response_frame.select_dtypes(include=np.number).columns
                             if c != "username" and c not in inference_frame.columns]
            if response_cols:
                inference_frame = inference_frame.merge(
                    response_frame[["username"] + response_cols], on="username", how="left", validate="one_to_one")
        robust_rows = []
        infer_cols = compare + response_cols
        for col in infer_cols:
            frame = inference_frame[["treatment", "grade", "Class", col]].copy()
            frame["value"] = pd.to_numeric(frame[col], errors="coerce")
            frame = frame.dropna(subset=["value", "grade", "Class"])
            if len(frame) < 50 or frame.value.nunique() < 2 or frame.value.std(ddof=0) == 0:
                continue
            y_std = (frame.value.to_numpy(dtype=float) - frame.value.mean()) / frame.value.std(ddof=0)
            treatment_aug = frame.treatment.eq("aug").astype(float).to_numpy()
            grade_dummies = pd.get_dummies(frame.grade.astype("string"), drop_first=True, dtype=float).to_numpy()
            design = np.column_stack([np.ones(len(frame)), treatment_aug, grade_dummies])
            try:
                beta, se, pvals, n_classes, ci = class_clustered_ols(
                    y_std, design, frame.Class.astype(str).to_numpy())
            except (ValueError, np.linalg.LinAlgError):
                continue
            robust_rows.append({"feature": col, "treatment_contrast": "GPT Tutor (aug) - GPT Base (vanilla)",
                                "coefficient_sd": float(beta[1]), "cluster_robust_se": float(se[1]),
                                "ci95_low": float(ci[1, 0]), "ci95_high": float(ci[1, 1]),
                                "p_value": float(pvals[1]), "students": len(frame), "classes": n_classes,
                                "grade_adjusted": True})
        if robust_rows:
            q = bh_fdr([row["p_value"] for row in robust_rows])
            for row, qv in zip(robust_rows, q): row["fdr_q_value"] = float(qv)
        clustered_path = TABLES / "class_clustered_treatment_text_associations.csv"
        robust_table = pd.DataFrame(robust_rows)
        if not robust_table.empty:
            robust_table = robust_table.sort_values("fdr_q_value")
        robust_table.to_csv(clustered_path, index=False)
        class_clustered_summary = {"available": True, "student_rows": int(len(inference_frame)),
                                   "class_count": int(inference_frame.Class.nunique()),
                                   "tested_features": len(robust_rows), "result_file": str(clustered_path.relative_to(ROOT))}

    # Audit final state and machine-readable research summary.
    summary = {
        "seed": SEED,
        "input_student_rows": len(students), "input_student_unique": students.username.nunique(),
        "input_cleaned_student_message_rows": original_message_count,
        "student_message_role_values": messages.role.unique().tolist(),
        "timestamp_missing_student_messages": int(messages.time_stamp.isna().sum()),
        "validated_student_username_join": True,
        "derived_rows": {"message_text_features": len(msg), "students": len(student_dataset), "student_session": len(session),
                         "student_problem": len(problem), "conversation_user_summary": len(conv_user)},
        "assistant_turn_audit": turn_summary,
        "academic_outcome_data_available": (ROOT / "AcademicOutcomes" / "matched_session_outcomes.csv").is_file(),
        "outcome_model_run": False,
        "treatment_mapping": {"vanilla": "GPT Base", "aug": "GPT Tutor"},
        "treatment_classifier_cv_grouping": classifier_grouping,
        "class_clustered_treatment_text_associations": class_clustered_summary,
        "clustering": {"input_feature_count": len(cluster_cols), "input_features": cluster_cols, "selected_k": k_chosen,
                       "kmeans_silhouette": float(silhouette_score(X_scaled, km.labels_)),
                       "kmeans_davies_bouldin": float(davies_bouldin_score(X_scaled, km.labels_)),
                       "kmeans_calinski_harabasz": float(calinski_harabasz_score(X_scaled, km.labels_))},
        "semantic_analysis": nmf_status,
        "automated_annotations_are_ground_truth": False,
        "manual_review_sample_rows": len(review),
        "no_api_models_or_paid_calls": True,
        "algorithm_families_executed": ["PCA", "K-Means", "Agglomerative Ward", "TF-IDF", "NMF", "Logistic Regression", "Random Forest", "Gradient Boosting", "Mann-Whitney U", "class-clustered OLS"],
    }
    save_json(OUT / "research_run_summary.json", summary)
    # Validate expected artifacts before reporting successful run.
    required_files = [DERIVED / "message_text_features.csv", DERIVED / "student_features_with_text.csv",
                      DERIVED / "student_session_features.csv", DERIVED / "student_problem_features.csv",
                      DERIVED / "conversation_student_features.csv", DERIVED / "assistant_student_followup_pairs.csv",
                      TABLES / "cluster_algorithm_comparison.csv", TABLES / "treatment_classifier_cv.csv",
                      TABLES / "cluster_profile_hypotheses.csv",
                      FIGURES / "treatment_text_feature_effects.png", FIGURES / "assistant_response_treatment_comparison.png",
                      FIGURES / "lexical_topics_by_treatment.png", OUT / "research_run_summary.json"]
    if class_clustered_summary["available"]:
        required_files.append(TABLES / "class_clustered_treatment_text_associations.csv")
    missing_outputs = [str(p) for p in required_files if not p.is_file()]
    if missing_outputs:
        raise RuntimeError(f"Missing expected outputs: {missing_outputs}")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved research outputs to {OUT}")


if __name__ == "__main__":
    main()
