"""Fine-tune shortlisted models, try ensembles, and save the final pipeline.

Notes covered
-------------
1. Hyperparameter search with cross-validation. Data-transform choices
   (imputation, correlation filter, log/ratio features) are searched with
   the model parameters. Randomized search is the default.
2. Soft voting and stacking ensembles of the tuned models.
3. The winner is refit on all training rows and saved for prediction.

The held-out test set is scored once after the winner is chosen.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.shortlist import apply_feature_plan, learn_feature_plan, model_specs

DEFAULT_BUNDLE = Path("outputs/optimize_model/final_model.joblib")
SHORTLIST_SUMMARY = Path("outputs/model_shortlist/summary.json")


class FeaturePlanTransformer(BaseEstimator, TransformerMixin):
    """Learn a shortlist-style feature plan on each training fold."""

    def __init__(
        self,
        drop_correlated: bool = False,
        add_log: bool = False,
        add_ratios: bool = False,
        corr_threshold: float = 0.9,
        skew_threshold: float = 1.0,
        n_ratio_features: int = 3,
    ):
        self.drop_correlated = drop_correlated
        self.add_log = add_log
        self.add_ratios = add_ratios
        self.corr_threshold = corr_threshold
        self.skew_threshold = skew_threshold
        self.n_ratio_features = n_ratio_features

    def fit(self, X, y=None):
        frame = self._as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        cfg = {
            "drop_correlated": self.drop_correlated,
            "add_log": self.add_log,
            "add_ratios": self.add_ratios,
            "corr_threshold": self.corr_threshold,
            "skew_threshold": self.skew_threshold,
            "n_ratio_features": self.n_ratio_features,
        }
        self.plan_ = learn_feature_plan(frame, np.asarray(y), cfg)
        return self

    def transform(self, X):
        return apply_feature_plan(self._as_frame(X), self.plan_)

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        columns = getattr(self, "feature_names_in_", None)
        if columns is None:
            raise ValueError("FeaturePlanTransformer must be fitted before transform.")
        return pd.DataFrame(X, columns=list(columns))


def make_search_preprocessor(needs_scaling: bool, num_strategy: str = "median") -> ColumnTransformer:
    if num_strategy == "constant":
        num_imputer = SimpleImputer(strategy="constant", fill_value=0)
    else:
        num_imputer = SimpleImputer(strategy=num_strategy)
    num_steps = [("impute", num_imputer)]
    if needs_scaling:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), make_column_selector(dtype_include=np.number)),
            ("cat", Pipeline(cat_steps), make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )


def build_pipeline(spec: dict) -> Pipeline:
    return Pipeline(
        [
            ("features", FeaturePlanTransformer()),
            ("prep", make_search_preprocessor(spec["scale"])),
            ("model", clone(spec["estimator"])),
        ]
    )


def transform_search_space() -> dict:
    """Data-preparation choices treated as hyperparameters."""
    return {
        "features__drop_correlated": [False, True],
        "features__add_log": [False, True],
        "features__add_ratios": [False, True],
        "prep__num__impute__strategy": ["median", "mean", "constant"],
        "prep__num__impute__fill_value": [0],
    }


def model_search_space(name: str) -> dict:
    spaces = {
        "MLPClassifier": {
            "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "model__alpha": loguniform(1e-5, 1e-1),
            "model__learning_rate_init": loguniform(1e-4, 1e-2),
            "n_iter": 8,
        },
        "HistGradientBoosting": {
            "model__max_depth": [None, 3, 5, 8],
            "model__learning_rate": loguniform(0.03, 0.3),
            "model__max_iter": [100, 200, 300],
            "model__min_samples_leaf": [10, 20, 50],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "n_iter": 12,
        },
        "LDA": {
            "model__solver": ["svd", "lsqr"],
            "n_iter": 8,
        },
        "LogisticRegression": {
            "model__C": loguniform(0.01, 20),
            "model__class_weight": [None, "balanced"],
            "n_iter": 12,
        },
        "ExtraTrees": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": [None, "balanced"],
            "n_iter": 12,
        },
        "RandomForest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2"],
            "n_iter": 12,
        },
        "GradientBoosting": {
            "model__n_estimators": [80, 120, 200],
            "model__learning_rate": loguniform(0.03, 0.3),
            "model__max_depth": [2, 3, 5],
            "n_iter": 10,
        },
        "SVC_rbf": {
            "model__C": loguniform(0.1, 20),
            "model__gamma": ["scale", "auto"],
            "model__probability": [True],
            "n_iter": 8,
        },
    }
    extra = dict(spaces.get(name, {"n_iter": 8}))
    n_iter = extra.pop("n_iter", 10)
    return {**transform_search_space(), **extra}, n_iter


def encode_xy(train_df: pd.DataFrame, test_df: pd.DataFrame, label_column: str):
    if label_column not in train_df.columns:
        raise ValueError(f"'{label_column}' not found in training data")
    encoder = LabelEncoder().fit(train_df[label_column])
    X_train = train_df.drop(columns=label_column).reset_index(drop=True)
    y_train = encoder.transform(train_df[label_column])
    X_test = test_df.drop(columns=label_column).reset_index(drop=True)
    y_test = encoder.transform(test_df[label_column])
    return X_train, y_train, X_test, y_test, encoder


def load_shortlist_names(shortlist=None, summary_path: Path = SHORTLIST_SUMMARY) -> list[str]:
    if shortlist:
        return list(shortlist)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No shortlist found at {summary_path.as_posix()}. Run training through the shortlist step first."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    names = payload.get("shortlist") or []
    if not names:
        raise ValueError("Shortlist summary does not contain any model names.")
    return names


def tune_one_model(name: str, spec: dict, X, y, cv, seed: int) -> dict | None:
    space, n_iter = model_search_space(name)
    pipeline = build_pipeline(spec)
    search = RandomizedSearchCV(
        pipeline,
        space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
        error_score=np.nan,
    )
    print(f"Tuning {name} with {n_iter} random-search draws...")
    search.fit(X, y)
    if np.isnan(search.best_score_):
        print(f"  {name} search produced no valid score; skipped.")
        return None
    print(f"  {name} CV ROC-AUC = {search.best_score_:.4f}")
    return {
        "name": name,
        "search": search,
        "estimator": search.best_estimator_,
        "cv_roc_auc": float(search.best_score_),
        "params": search.best_params_,
    }


def ensemble_cv_score(estimator, X, y, cv) -> float:
    scores = cross_val_score(estimator, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    return float(np.nanmean(scores))


def build_ensembles(tuned: list[dict], X, y, cv) -> list[dict]:
    if len(tuned) < 2:
        return []

    ranked = sorted(tuned, key=lambda item: item["cv_roc_auc"], reverse=True)
    top = ranked[: min(3, len(ranked))]
    candidates = []

    vote = VotingClassifier(
        [(item["name"], clone(item["estimator"])) for item in top],
        voting="soft",
    )
    print("Evaluating soft-voting ensemble...")
    vote_score = ensemble_cv_score(vote, X, y, cv)
    vote.fit(X, y)
    print(f"  SoftVote CV ROC-AUC = {vote_score:.4f}")
    candidates.append(
        {"name": "SoftVote", "estimator": vote, "cv_roc_auc": vote_score, "params": {"members": [m["name"] for m in top]}}
    )

    stack = StackingClassifier(
        [(item["name"], clone(item["estimator"])) for item in ranked],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    print("Evaluating stacking ensemble...")
    stack_score = ensemble_cv_score(stack, X, y, cv)
    stack.fit(X, y)
    print(f"  Stacking CV ROC-AUC = {stack_score:.4f}")
    candidates.append(
        {
            "name": "Stacking",
            "estimator": stack,
            "cv_roc_auc": stack_score,
            "params": {"members": [m["name"] for m in ranked]},
        }
    )
    return candidates


def score_test(estimator, X_test, y_test) -> dict:
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X_test)[:, 1]
    else:
        scores = estimator.decision_function(X_test)
    pred = estimator.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_optimize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = "Class",
    shortlist=None,
    output_dir: str = "outputs/optimize_model",
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Fine-tune shortlisted models on all training rows and save the winner."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = load_shortlist_names(shortlist)
    X_train, y_train, X_test, y_test, encoder = encode_xy(train_df, test_df, label_column)
    print(f"Fine-tuning on {len(X_train)} training rows; test held out until the end.")
    print(f"Shortlisted models: {', '.join(names)}")

    specs = model_specs(seed)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tuned = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            if name not in specs:
                print(f"Skipping {name}: not in the current model registry.")
                continue
            try:
                result = tune_one_model(name, specs[name], X_train, y_train, cv, seed)
            except Exception as exc:
                print(f"  {name} failed: {type(exc).__name__}: {exc}")
                continue
            if result is not None:
                tuned.append(result)

    if not tuned:
        raise RuntimeError("No shortlisted model completed hyperparameter search.")

    ensembles = build_ensembles(tuned, X_train, y_train, cv)
    contest = tuned + ensembles
    winner = max(contest, key=lambda item: item["cv_roc_auc"])
    print(f"\nWinner: {winner['name']} (CV ROC-AUC {winner['cv_roc_auc']:.4f})")

    final_estimator = clone(winner["estimator"])
    final_estimator.fit(X_train, y_train)
    test_metrics = score_test(final_estimator, X_test, y_test)
    print(
        "Held-out test: "
        f"ROC-AUC {test_metrics['roc_auc']:.4f}, "
        f"accuracy {test_metrics['accuracy']:.4f}, "
        f"precision {test_metrics['precision']:.4f}, "
        f"recall {test_metrics['recall']:.4f}"
    )

    rows = []
    for item in contest:
        rows.append(
            {
                "model": item["name"],
                "cv_roc_auc": item["cv_roc_auc"],
                "params": json.dumps(item.get("params", {}), default=str),
                "selected": item["name"] == winner["name"],
            }
        )
    results_path = output / "search_results.csv"
    pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).to_csv(results_path, index=False)

    bundle = {
        "pipeline": final_estimator,
        "label_encoder": encoder,
        "feature_names": list(X_train.columns),
        "model_name": winner["name"],
        "cv_roc_auc": winner["cv_roc_auc"],
        "test_metrics": test_metrics,
        "params": winner.get("params", {}),
    }
    bundle_path = output / "final_model.joblib"
    joblib.dump(bundle, bundle_path)

    summary = {
        "model_name": winner["name"],
        "cv_roc_auc": winner["cv_roc_auc"],
        "test_metrics": test_metrics,
        "feature_names": list(X_train.columns),
        "class_labels": [str(label) for label in encoder.classes_],
        "bundle_path": bundle_path.as_posix(),
        "rows_used": int(len(X_train)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved final model to: {bundle_path.as_posix()}")
    return summary


def load_final_model(path: str | Path = DEFAULT_BUNDLE) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path.as_posix()}. Choose 1 in main.py to train first."
        )
    return joblib.load(path)


def parse_feature_input(text: str, feature_names: list[str]) -> pd.DataFrame:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    values = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Expected FeatureName=value, got: {part}")
        name, raw = part.split("=", 1)
        values[name.strip()] = float(raw.strip())
    missing = [name for name in feature_names if name not in values]
    extra = [name for name in values if name not in feature_names]
    if missing:
        raise ValueError("Missing features: " + ", ".join(missing))
    if extra:
        raise ValueError("Unknown features: " + ", ".join(extra))
    return pd.DataFrame([{name: values[name] for name in feature_names}])


def format_input_prompt(feature_names: list[str]) -> str:
    example = {
        "Area": 87524,
        "MajorAxisLength": 442.246,
        "MinorAxisLength": 253.291,
        "Eccentricity": 0.820,
        "ConvexArea": 90546,
        "Extent": 0.759,
        "Perimeter": 1184.04,
    }
    sample = ", ".join(
        f"{name}={example.get(name, 0)}"
        for name in feature_names
    )
    return (
        "Enter one raisin as FeatureName=value pairs, comma-separated.\n"
        f"Required features: {', '.join(feature_names)}\n"
        f"Example: {sample}"
    )


def predict_row(bundle: dict, row: pd.DataFrame) -> dict:
    pipeline = bundle["pipeline"]
    encoder: LabelEncoder = bundle["label_encoder"]
    pred = pipeline.predict(row)
    label = encoder.inverse_transform(pred)[0]
    result = {"class": str(label)}
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(row)[0]
        result["probabilities"] = {
            str(cls): float(score) for cls, score in zip(encoder.classes_, proba)
        }
    return result


def predict_interactive(path: str | Path = DEFAULT_BUNDLE) -> None:
    bundle = load_final_model(path)
    feature_names = list(bundle["feature_names"])
    print(f"Loaded {bundle['model_name']} from {Path(path).as_posix()}")
    print(format_input_prompt(feature_names))
    while True:
        try:
            raw = input("\nValues: ").strip()
        except EOFError:
            return
        if not raw:
            print("Empty input; prediction cancelled.")
            return
        try:
            row = parse_feature_input(raw, feature_names)
        except ValueError as exc:
            print(exc)
            continue
        result = predict_row(bundle, row)
        print(f"Predicted class: {result['class']}")
        if "probabilities" in result:
            pretty = ", ".join(
                f"{name}={score:.3f}" for name, score in result["probabilities"].items()
            )
            print(f"Probabilities: {pretty}")
        return
