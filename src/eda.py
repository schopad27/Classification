"""Explore the training data after a train/test split."""

import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_COLUMN = "Class"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def run_eda(frame: pd.DataFrame, output_dir: str = "outputs/eda") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the table, run EDA on the training set, and save the reports."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_df, test_df = _split_train_test(frame, output)
    _run_deepchecks(train_df, test_df, output)
    _run_autoviz(train_df, output)
    _run_ydata(train_df, output)
    return train_df, test_df


def _split_train_test(frame: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Splitting train and test sets...")
    train_df, test_df = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame[LABEL_COLUMN],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = output / "train.csv"
    test_path = output / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved training data ({len(train_df)} rows) to: {train_path.as_posix()}")
    print(f"Saved test data ({len(test_df)} rows) to: {test_path.as_posix()}")
    return train_df, test_df


def _prepare_deepchecks_import() -> None:
    """Let deepchecks 0.19.1 import against NumPy 2 and scikit-learn 1.9."""
    import numpy as np
    import sklearn.metrics as metrics
    from sklearn.metrics import max_error, mean_squared_error, root_mean_squared_error
    from sklearn.metrics._scorer import _SCORERS, make_scorer

    aliases = {
        "Inf": np.inf,
        "Infinity": np.inf,
        "NaN": np.nan,
        "PINF": np.inf,
        "NINF": -np.inf,
    }
    for name, value in aliases.items():
        if getattr(np, name, None) is None:
            setattr(np, name, value)

    if "max_error" not in _SCORERS:
        _SCORERS["max_error"] = make_scorer(max_error, greater_is_better=False)

    if getattr(metrics.mean_squared_error, "_raisin_squared_compat", False):
        return

    def _mean_squared_error(y_true, y_pred, *, squared=True, **kwargs):
        if squared:
            return mean_squared_error(y_true, y_pred, **kwargs)
        return root_mean_squared_error(y_true, y_pred, **kwargs)

    _mean_squared_error._raisin_squared_compat = True
    metrics.mean_squared_error = _mean_squared_error


def _run_deepchecks(train_df: pd.DataFrame, test_df: pd.DataFrame, output: Path) -> dict:
    _prepare_deepchecks_import()
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity, train_test_validation

    print("Running the Deepchecks data-integrity suite on training data...")
    train_dataset = Dataset(train_df, label=LABEL_COLUMN, cat_features=[])
    test_dataset = Dataset(test_df, label=LABEL_COLUMN, cat_features=[])

    integrity = data_integrity().run(train_dataset)
    integrity_path = output / "deepchecks_integrity.html"
    integrity.save_as_html(str(integrity_path), as_widget=False)
    print(f"Saved Deepchecks integrity report to: {integrity_path.as_posix()}")

    print("Running the Deepchecks train-test validation suite...")
    split_result = train_test_validation().run(train_dataset, test_dataset)
    split_path = output / "deepchecks_train_test.html"
    split_result.save_as_html(str(split_path), as_widget=False)
    print(f"Saved Deepchecks train-test report to: {split_path.as_posix()}")

    return {
        "integrity_path": integrity_path.as_posix(),
        "train_test_path": split_path.as_posix(),
    }


def _run_autoviz(frame: pd.DataFrame, output: Path) -> dict:
    from autoviz import AutoViz_Class

    print("Drawing AutoViz charts on training data...")
    chart_dir = output / "autoviz"
    if chart_dir.exists():
        shutil.rmtree(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    visualizer = AutoViz_Class()
    visualizer.AutoViz(
        "",
        sep=",",
        depVar=LABEL_COLUMN,
        dfte=frame,
        header=0,
        verbose=0,
        chart_format="html",
        save_plot_dir=str(chart_dir),
    )
    charts = sorted(
        path.as_posix()
        for path in chart_dir.rglob("*")
        if path.suffix.lower() in {".html", ".png", ".svg"}
    )
    print(f"Saved AutoViz charts to: {chart_dir.as_posix()}")
    return {
        "chart_dir": chart_dir.as_posix(),
        "chart_count": len(charts),
        "charts": charts,
        "target": LABEL_COLUMN,
    }


def _run_ydata(frame: pd.DataFrame, output: Path) -> dict:
    from ydata_profiling import ProfileReport

    print("Building the ydata-profiling report on training data...")
    profile = ProfileReport(frame, title="Raisin training profile", explorative=True)
    report_path = output / "ydata_profile.html"
    profile.to_file(report_path)
    print(f"Saved ydata-profiling report to: {report_path.as_posix()}")

    description = json.loads(profile.to_json())
    table = description.get("table", {})
    return {
        "report_path": report_path.as_posix(),
        "missing_cells": int(table.get("n_cells_missing", frame.isna().sum().sum())),
        "duplicate_rows": int(table.get("n_duplicates", frame.duplicated().sum())),
    }
