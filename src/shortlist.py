"""
model_shortlist.py - "Shortlist promising models" for BINARY classification.
Functional style: no custom classes, small pure functions, data passed in and
returned explicitly. Side effects (printing, writing files) live only in
`write_iteration_outputs` and `run_shortlist`.

Tested with numpy 2.2.0, pandas 2.2.3, scikit-learn 1.9.0, scipy 1.16.3.
xgboost / lightgbm are optional and skipped unless installed.

Usage
-----
    from src.shortlist import run_shortlist
    summary = run_shortlist(train_df, test_df, label_column="Class",
                            output_dir="outputs/model_shortlist")

Call tree
---------
run_shortlist
 |- prepare_data                      encode label, schema check, stratified sample
 |- for each iteration in ITERATIONS:
 |   '- run_iteration
 |       |- cross_validate_all        STEPS 1-2  train + N-fold CV
 |       |   '- cross_validate_one
 |       |       |- fit_bundle        learn_feature_plan -> apply_feature_plan
 |       |       |                    -> make_preprocessor -> fit model
 |       |       |- score_bundle      apply plan -> preprocess -> score
 |       |       '- compute_metrics
 |       |- importance_table          STEP 3    permutation importance
 |       |   '- model_importance -> permutation_importance_auc
 |       '- analyze_errors            STEP 4    error matrix / overlap / hard rows
 |- write_iteration_outputs           save CSVs (side effect)
 |- choose_best_iteration             STEP 6    best mean top-k score
 '- pick_shortlist                    STEP 7    score minus error-overlap penalty

Leakage note: only `train_df` drives decisions. `test_df` is used for a column
check only and is never scored.
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

METRICS = ("roc_auc", "avg_precision", "f1", "precision", "recall",
           "accuracy", "balanced_accuracy")

# One entry per iteration = one setting of the feature selection/engineering step.
ITERATIONS = {
    "iter1_baseline": {},
    "iter2_corr_filter": {"drop_correlated": True},
    "iter3_filter_engineer": {"drop_correlated": True, "add_log": True,
                              "add_ratios": True},
}


# =========================================================================== #
# Step 1: model specs  ->  {name: {"estimator": ..., "scale": bool}}
# =========================================================================== #
def _optional_specs(seed: int) -> dict:
    """XGBoost / LightGBM, only if they are installed."""
    found = {}
    try:
        from xgboost import XGBClassifier
        found["XGBoost"] = (XGBClassifier(eval_metric="logloss", random_state=seed,
                                          n_jobs=-1), False)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        found["LightGBM"] = (LGBMClassifier(random_state=seed, verbose=-1), False)
    except Exception:
        pass
    return found


def model_specs(seed: int = 42) -> dict:
    """Default-parameter models from different families; `scale` marks the
    ones that need standardised inputs (linear, distance, kernel)."""
    base = {
        "LogisticRegression": (LogisticRegression(max_iter=2000), True),
        "RidgeClassifier": (RidgeClassifier(), True),
        "LDA": (LinearDiscriminantAnalysis(), True),
        "GaussianNB": (GaussianNB(), False),
        "KNN": (KNeighborsClassifier(), True),
        "SVC_rbf": (SVC(random_state=seed), True),
        "DecisionTree": (DecisionTreeClassifier(random_state=seed), False),
        "RandomForest": (RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                random_state=seed), False),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=200, n_jobs=-1,
                                            random_state=seed), False),
        "GradientBoosting": (GradientBoostingClassifier(random_state=seed), False),
        "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=seed), False),
        "AdaBoost": (AdaBoostClassifier(random_state=seed), False),
        "MLPClassifier": (MLPClassifier(max_iter=400, random_state=seed), True),
    }
    return {name: {"estimator": est, "scale": scale}
            for name, (est, scale) in {**base, **_optional_specs(seed)}.items()}


# =========================================================================== #
# Step 5: feature selection + engineering as a learned "plan" (a plain dict)
#   learn_feature_plan(train fold)  ->  plan
#   apply_feature_plan(any data, plan) -> new DataFrame
# The plan is learned on training folds only, so there is no leakage.
# =========================================================================== #
def _relevance(filled: pd.DataFrame, y: np.ndarray, max_rows: int = 5000) -> pd.Series:
    """Mutual information of each numeric column with the target."""
    idx = np.arange(len(filled))
    if len(idx) > max_rows:
        idx = np.random.default_rng(0).choice(idx, max_rows, replace=False)
    mi = mutual_info_classif(filled.iloc[idx], y[idx], random_state=0)
    return pd.Series(mi, index=filled.columns)


def _correlation_filter(filled: pd.DataFrame, order: list, threshold: float) -> list:
    """Greedy: walk columns from most to least relevant, keep one only if it is
    not highly correlated with something already kept. Drops constants."""
    corr = filled.corr().abs().fillna(0)

    def step(kept, col):
        ok = filled[col].nunique() > 1 and all(corr.loc[col, k] < threshold for k in kept)
        return kept + [col] if ok else kept

    return list(itertools.accumulate(order, step, initial=[]))[-1]


def _skewed_columns(num: pd.DataFrame, keep: list, skew_threshold: float) -> list:
    return [c for c in keep
            if num[c].min() >= 0 and num[c].nunique() > 10
            and abs(num[c].skew()) > skew_threshold]


def learn_feature_plan(X: pd.DataFrame, y: np.ndarray, cfg: dict) -> dict:
    num_cols = X.select_dtypes("number").columns.tolist()
    empty = {"num_cols": num_cols, "keep_num": num_cols, "log_cols": [], "ratio_pairs": []}
    if not num_cols:
        return empty

    num = X[num_cols]
    filled = num.fillna(num.median())
    needs_relevance = cfg.get("drop_correlated") or cfg.get("add_ratios")
    relevance = (_relevance(filled, y) if needs_relevance
                 else pd.Series(0.0, index=num_cols))
    order = relevance.sort_values(ascending=False).index.tolist()

    kept = (_correlation_filter(filled, order, cfg.get("corr_threshold", 0.9))
            if cfg.get("drop_correlated") else num_cols)
    keep = [c for c in num_cols if c in kept]
    top = [c for c in order if c in keep][: cfg.get("n_ratio_features", 3)]

    return {
        **empty,
        "keep_num": keep,
        "log_cols": (_skewed_columns(num, keep, cfg.get("skew_threshold", 1.0))
                     if cfg.get("add_log") else []),
        "ratio_pairs": (list(itertools.combinations(top, 2))
                        if cfg.get("add_ratios") else []),
    }


def apply_feature_plan(X: pd.DataFrame, plan: dict) -> pd.DataFrame:
    dropped = [c for c in plan["num_cols"] if c not in plan["keep_num"]]
    logs = {f"{c}_log1p": np.log1p(X[c].clip(lower=0)) for c in plan["log_cols"]}
    ratios = {f"{a}_per_{b}": (X[a] / X[b].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
              for a, b in plan["ratio_pairs"]}
    return pd.concat([X.drop(columns=dropped), pd.DataFrame({**logs, **ratios}, index=X.index)],
                     axis=1)


# =========================================================================== #
# Preprocessing + model "bundle" (plain dict: plan, prep, model)
# =========================================================================== #
def make_preprocessor(needs_scaling: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if needs_scaling:                       # scale only where the model needs it
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), make_column_selector(dtype_include=np.number)),
         ("cat", Pipeline(cat_steps), make_column_selector(dtype_exclude=np.number))],
        remainder="drop")


def fit_bundle(spec: dict, cfg: dict, X: pd.DataFrame, y: np.ndarray) -> dict:
    plan = learn_feature_plan(X, y, cfg)#gives which columns are relvant basis on config and which requires transformation and which is required ratio calculation features
    X_feat = apply_feature_plan(X, plan)# this will contain chosen features based on config  for only one fold  and also tranformation like log of ratio if qualified
    prep = make_preprocessor(spec["scale"]).fit(X_feat)#This will give column trasnformer basis on number of categorical features impute and standardcaler for number and 
    model = clone(spec["estimator"]).fit(prep.transform(X_feat), y)#Fit individual model and post scaling 
    return {"plan": plan, "prep": prep, "model": model}


def score_bundle(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Return (continuous score for the positive class, decision threshold)."""
    Z = bundle["prep"].transform(apply_feature_plan(X, bundle["plan"]))
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Z)[:, 1], 0.5
    return model.decision_function(Z), 0.0


def compute_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = (score > thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, score),
        "avg_precision": average_precision_score(y, score),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
    }


# =========================================================================== #
# Steps 1-2: cross-validation (mean +/- std) and out-of-fold scores
# =========================================================================== #
def make_folds(X: pd.DataFrame, y: np.ndarray, n_folds: int, seed: int) -> list:
    return list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                random_state=seed).split(X, y))


def cross_validate_one(spec, cfg, X, y, folds):
    """-> (per-fold metrics DataFrame, out-of-fold scores, threshold)."""
    oof = np.full(len(y), np.nan)
    fold_rows, thr = [], 0.5
    for tr, va in folds:
        bundle = fit_bundle(spec, cfg, X.iloc[tr], y[tr])#for each fold it will calculate
        scores, thr = score_bundle(bundle, X.iloc[va])
        oof[va] = scores
        fold_rows.append(compute_metrics(y[va], scores, thr))
    return pd.DataFrame(fold_rows), oof, thr


def summarize_folds(name: str, fold_df: pd.DataFrame, seconds: float) -> dict:
    stats = {f"{m}_{kind}": getattr(fold_df[m], kind)()
             for m in METRICS for kind in ("mean", "std")}
    return {"model": name, **stats, "fit_seconds": round(seconds, 2)}


def cross_validate_all(specs, cfg, X, y, n_folds, seed):
    """-> (results DataFrame, {model: (oof_scores, thr)}, {model: error text})."""
    folds = make_folds(X, y, n_folds, seed)
    rows, oof, failed = [], {}, {}
    for name, spec in specs.items():
        start = time.time()
        try:
            fold_df, scores, thr = cross_validate_one(spec, cfg, X, y, folds)
        except Exception as exc:            # one bad model must not stop the run
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        rows.append(summarize_folds(name, fold_df, time.time() - start))
        oof[name] = (scores, thr)
    return pd.DataFrame(rows), oof, failed


# =========================================================================== #
# Step 3: variable importance (permutation, drop in ROC-AUC)
# =========================================================================== #
def permutation_importance_auc(bundle, X_va, y_va, n_repeats, seed) -> pd.Series:
    """Model-agnostic: shuffle one column at a time, measure the AUC drop."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y_va, score_bundle(bundle, X_va)[0])

    def drop_for(col):
        def one_repeat(_):
            shuffled = X_va.assign(**{col: rng.permutation(X_va[col].to_numpy())})
            return base - roc_auc_score(y_va, score_bundle(bundle, shuffled)[0])
        return float(np.mean([one_repeat(r) for r in range(n_repeats)]))

    return pd.Series({col: drop_for(col) for col in X_va.columns})


def model_importance(spec, cfg, X_tr, y_tr, X_va, y_va, n_repeats, seed):
    try:
        bundle = fit_bundle(spec, cfg, X_tr, y_tr)
        return permutation_importance_auc(bundle, X_va, y_va, n_repeats, seed)
    except Exception:
        return None


def importance_table(specs, cfg, X, y, seed, n_repeats=5, max_val_rows=2000):
    """Caveat: correlated features share / mask each other's importance."""
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=seed)
    if len(X_va) > max_val_rows:
        X_va, _, y_va, _ = train_test_split(X_va, y_va, train_size=max_val_rows,
                                            stratify=y_va, random_state=seed)
    per_model = {name: model_importance(spec, cfg, X_tr, y_tr, X_va, y_va, n_repeats, seed)
                 for name, spec in specs.items()}
    table = pd.DataFrame({k: v for k, v in per_model.items() if v is not None})
    if table.empty:
        return table
    return (table.assign(mean_rank=table.rank(ascending=False).mean(axis=1))
                 .sort_values("mean_rank"))


# =========================================================================== #
# Step 4: error analysis (from out-of-fold predictions)
# =========================================================================== #
def error_matrix(oof: dict, y: np.ndarray) -> pd.DataFrame:
    """rows = samples, columns = models, value = True when the model is wrong."""
    return pd.DataFrame({name: (scores > thr).astype(int) != y
                         for name, (scores, thr) in oof.items()})


def error_summary(oof: dict, y: np.ndarray, errors: pd.DataFrame) -> pd.DataFrame:
    def row(name):
        pred = (oof[name][0] > oof[name][1]).astype(int)
        return {"false_positives": int(((pred == 1) & (y == 0)).sum()),
                "false_negatives": int(((pred == 0) & (y == 1)).sum()),
                "error_rate": float(errors[name].mean())}
    return pd.DataFrame({name: row(name) for name in oof}).T


def error_overlap(errors: pd.DataFrame) -> pd.DataFrame:
    """Jaccard similarity of the sets of rows each pair of models gets wrong."""
    E = errors.to_numpy().astype(int)
    inter = E.T @ E
    counts = np.diag(inter)
    union = counts[:, None] + counts[None, :] - inter
    jac = np.divide(inter, union, out=np.zeros(inter.shape, dtype=float), where=union > 0)
    return pd.DataFrame(jac, index=errors.columns, columns=errors.columns)


def hard_rows(X, y, errors, fraction=0.8) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows that at least `fraction` of the models get wrong."""
    n_wrong = errors.sum(axis=1)
    mask = (n_wrong >= max(1, int(np.ceil(fraction * errors.shape[1])))).to_numpy()
    rows = (X[mask].assign(true_label=y[mask], n_models_wrong=n_wrong[mask].to_numpy())
            .sort_values("n_models_wrong", ascending=False))
    return rows, mask


def hard_row_profile(X: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Standardised mean difference of each numeric feature: hard rows vs rest."""
    if not mask.any() or mask.all():
        return pd.DataFrame()
    num = X.select_dtypes("number")
    diff = (num[mask].mean() - num[~mask].mean()) / num.std().replace(0, np.nan)
    return (diff.rename("std_mean_diff").to_frame()
                .sort_values("std_mean_diff", key=np.abs, ascending=False))


def analyze_errors(oof, X, y) -> dict:
    errors = error_matrix(oof, y)
    hard, mask = hard_rows(X, y, errors)
    return {"per_model": error_summary(oof, y, errors),
            "overlap": error_overlap(errors),
            "hard_rows": hard,
            "hard_profile": hard_row_profile(X, mask)}


# =========================================================================== #
# Step 7: shortlist, preferring models whose errors overlap less
# =========================================================================== #
def pick_shortlist(results, overlap, primary, top_k, penalty=0.1, max_gap=0.05):
    col = f"{primary}_mean"
    ranked = results.sort_values(col, ascending=False).reset_index(drop=True)
    score = ranked.set_index("model")[col]
    pool = ranked.loc[ranked[col] >= ranked[col].iloc[0] - max_gap, "model"].tolist()

    def add_next(chosen):
        rest = [m for m in pool if m not in chosen]
        return chosen + [max(rest, key=lambda m: score[m] - penalty * overlap.loc[m, chosen].mean())]

    chosen = [pool[0]]
    while len(chosen) < min(top_k, len(pool)):
        chosen = add_next(chosen)

    def avg_overlap(m):
        others = [c for c in chosen if c != m]
        return overlap.loc[m, others].mean() if others else 0.0

    table = ranked[ranked.model.isin(chosen)].copy()
    table["avg_error_overlap_with_others"] = table.model.map(avg_overlap)
    table["pick_order"] = table.model.map({m: i + 1 for i, m in enumerate(chosen)})
    return table.sort_values("pick_order")


# =========================================================================== #
# One iteration = steps 1-4 for one feature-engineering config
# =========================================================================== #
def run_iteration(specs, cfg, X, y, n_folds, seed, primary, top_k) -> dict:
    results, oof, failed = cross_validate_all(specs, cfg, X, y, n_folds, seed)
    results = results.sort_values(f"{primary}_mean", ascending=False)
    ok_specs = {name: spec for name, spec in specs.items() if name in oof}
    return {
        "results": results,
        "failed": failed,
        "importance": importance_table(ok_specs, cfg, X, y, seed),
        "errors": analyze_errors(oof, X, y),
        "top_k_mean": float(results[f"{primary}_mean"].head(top_k).mean()),
    }


def choose_best_iteration(iterations: dict) -> str:
    return max(iterations, key=lambda name: iterations[name]["top_k_mean"])


# =========================================================================== #
# Data preparation (train only)
# =========================================================================== #
def prepare_data(train_df, test_df, label_column, max_rows, seed):
    """-> (X, y, class labels). Encodes the binary label and samples if huge."""
    if label_column not in train_df:
        raise ValueError(f"'{label_column}' not found in train_df")
    if test_df is not None and set(test_df.columns) != set(train_df.columns):
        raise ValueError("train_df and test_df have different columns")

    encoder = LabelEncoder().fit(train_df[label_column])
    if len(encoder.classes_) != 2:
        raise ValueError(f"Binary target expected, found: {list(encoder.classes_)}")
    y = encoder.transform(train_df[label_column])
    X = train_df.drop(columns=label_column).reset_index(drop=True)

    if len(X) > max_rows:                   # stratified sample keeps the loop fast
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y,
                                      random_state=seed)
        X = X.reset_index(drop=True)
    return X, y, encoder.classes_


# =========================================================================== #
# Side effects: printing and writing files
# =========================================================================== #
def write_iteration_outputs(folder: Path, it: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    it["results"].to_csv(folder / "cv_results.csv", index=False)
    it["importance"].to_csv(folder / "feature_importance.csv")
    it["errors"]["per_model"].to_csv(folder / "error_summary.csv")
    it["errors"]["overlap"].to_csv(folder / "error_overlap.csv")
    it["errors"]["hard_rows"].to_csv(folder / "hard_rows.csv", index=False)
    it["errors"]["hard_profile"].to_csv(folder / "hard_rows_profile.csv")


def _print_iteration(name, cfg, it, primary) -> None:
    print(f"\n=== {name} {cfg or '(no feature engineering)'} ===")
    for model, msg in it["failed"].items():
        print(f"  ! {model} failed: {msg}")
    cols = ["model", f"{primary}_mean", f"{primary}_std", "f1_mean", "f1_std", "fit_seconds"]
    print(it["results"][cols].round(4).to_string(index=False))


# =========================================================================== #
# Orchestration (steps 1-7)
# =========================================================================== #
def _run_shortlist(train_df, test_df, label_column, output_dir, n_folds, max_rows,
                   seed, top_k, primary_metric, diversity_penalty) -> dict:
    out = Path(output_dir)
    X, y, classes = prepare_data(train_df, test_df, label_column, max_rows, seed)#X feature without target varibale and y in encoded 0 and 1 and classes are two classes of class
    print(f"Positive class (1) = {classes[1]!r}, negative (0) = {classes[0]!r}; "
          f"rows used: {len(X)}")

    specs = model_specs(seed)#this will give all model name with value as instance of model class and true/false if scaling is required or not 
    iterations = {name: run_iteration(specs, cfg, X, y, n_folds, seed, primary_metric, top_k)
                  for name, cfg in ITERATIONS.items()}#iterations are with different configs

    for name, it in iterations.items():
        _print_iteration(name, ITERATIONS[name], it, primary_metric)
        write_iteration_outputs(out / name, it)

    best = choose_best_iteration(iterations)
    shortlist = pick_shortlist(iterations[best]["results"], iterations[best]["errors"]["overlap"],
                               primary_metric, top_k, diversity_penalty)
    shortlist.to_csv(out / "shortlist.csv", index=False)

    print(f"\nBest iteration by mean top-{top_k} {primary_metric}: {best}")
    print(shortlist[["pick_order", "model", f"{primary_metric}_mean", f"{primary_metric}_std",
                     "avg_error_overlap_with_others"]].round(4).to_string(index=False))

    summary = {
        "positive_class": str(classes[1]),
        "rows_used": int(len(X)),
        "iteration_scores": {n: it["top_k_mean"] for n, it in iterations.items()},
        "best_iteration": best,
        "best_iteration_feature_config": ITERATIONS[best],
        "shortlist": shortlist.model.tolist(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return {**summary, "shortlist_table": shortlist, "iteration_outputs": iterations}


def run_shortlist(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None,
                  label_column: str = "target", output_dir: str = "outputs/model_shortlist",
                  n_folds: int = 5, max_rows: int = 20000, seed: int = 42, top_k: int = 5,
                  primary_metric: str = "roc_auc", diversity_penalty: float = 0.1) -> dict:
    """Public entry point. Warnings are silenced only while this runs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _run_shortlist(train_df, test_df, label_column, output_dir, n_folds,
                              max_rows, seed, top_k, primary_metric, diversity_penalty)