"""Load a table and store the Get Data checks, charts, and insight deck."""

import json
import math
import shutil
import time
from pathlib import Path

import matplotlib
import pandas as pd
import requests
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

matplotlib.use("Agg")

LABEL_COLUMN = "Class"
LABEL_VALUES = ("Kecimen", "Besni")
NON_NEGATIVE_COLUMNS = (
    "Area",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "Perimeter",
)
UNIT_INTERVAL_COLUMNS = ("Eccentricity", "Extent")

NAVY = RGBColor(0x0E, 0x1C, 0x36)
GOLD = RGBColor(0xC4, 0xA3, 0x5A)
CREAM = RGBColor(0xF7, 0xF4, 0xEE)
INK = RGBColor(0x1F, 0x29, 0x37)


def load_data(path: str) -> pd.DataFrame:
    """Read a CSV into a DataFrame without changing its values."""
    return pd.read_csv(path)


def run_get_data(path: str, output_dir: str = "outputs/get_data") -> pd.DataFrame:
    """Load the CSV, write Get Data artifacts, and build the insight deck."""
    source = Path(path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {source}...")
    frame = load_data(str(source))

    insights = {
        "source_path": source.as_posix(),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns),
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "class_counts": {
            str(label): int(count)
            for label, count in frame[LABEL_COLUMN].value_counts(dropna=False).items()
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_counts": {
            name: int(count) for name, count in frame.isna().sum().items()
        },
    }

    insights["missingno"] = _run_missingno(frame, output)
    insights["pandera"] = _run_pandera(frame, output)
    insights["great_expectations"] = _run_great_expectations(frame, output)
    insights["dtale"] = _run_dtale(frame, output)

    deck_path = output / "raisin_get_data_insights.pptx"
    _write_presentation(insights, deck_path)
    print(f"Saved insight deck to: {deck_path.as_posix()}")
    return frame


def _run_missingno(frame: pd.DataFrame, output: Path) -> dict:
    import matplotlib.pyplot as plt
    import missingno as msno

    print("Drawing missing-value charts...")
    paths = {
        "matrix": output / "missingno_matrix.png",
        "bar": output / "missingno_bar.png",
        "heatmap": output / "missingno_heatmap.png",
    }
    _save_missingno_plot(lambda: msno.matrix(frame), paths["matrix"])
    _save_missingno_plot(lambda: msno.bar(frame), paths["bar"])
    heatmap_note = _save_missingno_heatmap(frame, paths["heatmap"])
    plt.close("all")
    return {
        "matrix": paths["matrix"].as_posix(),
        "bar": paths["bar"].as_posix(),
        "heatmap": paths["heatmap"].as_posix(),
        "heatmap_note": heatmap_note,
    }


def _save_missingno_plot(draw, path: Path) -> None:
    import matplotlib.pyplot as plt

    draw()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def _save_missingno_heatmap(frame: pd.DataFrame, path: Path) -> str:
    import matplotlib.pyplot as plt
    import missingno as msno

    columns_with_gaps = int((frame.isna().any()).sum())
    if columns_with_gaps < 2:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.axis("off")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.text(
            0.5,
            0.5,
            "No nullity correlation.\nFewer than two columns contain missing values.",
            ha="center",
            va="center",
            fontsize=14,
        )
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return (
            "missingno could not draw a nullity heatmap because fewer than two "
            "columns contain missing values. The saved figure records that fact."
        )

    msno.heatmap(frame)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return "missingno drew a nullity correlation heatmap."


def _run_pandera(frame: pd.DataFrame, output: Path) -> dict:
    import pandera.pandas as pa
    from pandera.errors import SchemaErrors

    print("Checking the Pandera schema...")
    schema = pa.infer_schema(frame)
    for column in NON_NEGATIVE_COLUMNS:
        schema = schema.update_column(column, nullable=False, checks=pa.Check.ge(0))
    for column in UNIT_INTERVAL_COLUMNS:
        schema = schema.update_column(
            column,
            nullable=False,
            checks=[pa.Check.ge(0), pa.Check.le(1)],
        )
    schema = schema.update_column(
        LABEL_COLUMN,
        nullable=False,
        checks=pa.Check.isin(list(LABEL_VALUES)),
    )

    schema_path = output / "pandera_schema.yaml"
    schema.to_yaml(schema_path)

    try:
        schema.validate(frame, lazy=True)
        report = {"success": True, "failure_counts": {}}
    except SchemaErrors as error:
        counts = (
            error.failure_cases.groupby(["column", "check"], dropna=False)
            .size()
            .reset_index(name="failures")
        )
        report = {
            "success": False,
            "failure_counts": {
                f"{row.column}: {row.check}": int(row.failures)
                for row in counts.itertuples(index=False)
            },
        }

    report_path = output / "pandera_validation.json"
    _write_json(report_path, report)
    report["schema_path"] = schema_path.as_posix()
    report["report_path"] = report_path.as_posix()
    return report


def _run_great_expectations(frame: pd.DataFrame, output: Path) -> dict:
    import great_expectations as gx
    import great_expectations.expectations as expectations

    print("Validating Great Expectations...")
    project_dir = output / "great_expectations"
    for context_dir in (project_dir, output / "gx"):
        if context_dir.exists():
            shutil.rmtree(context_dir)

    context = gx.get_context(mode="file", context_root_dir=str(project_dir))
    data_source = context.data_sources.add_pandas("raisin_source")
    data_asset = data_source.add_dataframe_asset("raisin_frame")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_table")

    suite = context.suites.add(gx.ExpectationSuite(name="raisin_measurements"))
    suite.add_expectation(
        expectations.ExpectTableColumnsToMatchSet(column_set=list(frame.columns))
    )
    for column in frame.columns:
        suite.add_expectation(expectations.ExpectColumnToExist(column=column))
        suite.add_expectation(expectations.ExpectColumnValuesToNotBeNull(column=column))

    suite.add_expectation(
        expectations.ExpectColumnValuesToBeInSet(
            column=LABEL_COLUMN,
            value_set=list(LABEL_VALUES),
        )
    )
    for column in NON_NEGATIVE_COLUMNS:
        suite.add_expectation(
            expectations.ExpectColumnValuesToBeBetween(column=column, min_value=0)
        )
    for column in UNIT_INTERVAL_COLUMNS:
        suite.add_expectation(
            expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=0,
                max_value=1,
            )
        )

    validation = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="raisin_validation",
            data=batch_definition,
            suite=suite,
        )
    )
    result = validation.run(batch_parameters={"dataframe": frame})
    result_path = project_dir / "raisin_validation.json"
    _write_json(result_path, result.to_json_dict())
    docs = context.build_data_docs()

    failed = []
    for item in result.results:
        if item.success:
            continue
        kwargs = item.expectation_config.kwargs
        column = kwargs.get("column", "table")
        failed.append(f"{item.expectation_config.type} on {column}")

    statistics = result.statistics or {}
    return {
        "success": bool(result.success),
        "successful": int(statistics.get("successful_expectations", 0)),
        "evaluated": int(statistics.get("evaluated_expectations", 0)),
        "failed": failed,
        "result_path": result_path.as_posix(),
        "docs_path": str(next(iter(docs.values()), "")),
        "context_dir": project_dir.as_posix(),
    }


def _run_dtale(frame: pd.DataFrame, output: Path) -> dict:
    import dtale
    from dtale.utils import build_shutdown_url

    print("Summarizing the table with D-Tale...")
    dtale.global_state.cleanup()
    instance = dtale.show(
        frame,
        name="raisin",
        subprocess=True,
        open_browser=False,
        ignore_duplicate=True,
    )
    if instance is None:
        raise RuntimeError("D-Tale did not start an instance for this table.")

    held = instance.data
    numeric = held.select_dtypes(include="number")
    summary = {
        "url": instance.main_url(),
        "dtypes": {name: str(dtype) for name, dtype in held.dtypes.items()},
        "missing_counts": {name: int(count) for name, count in held.isna().sum().items()},
        "describe": _clean_json(numeric.describe().round(4).to_dict()),
        "correlations": _clean_json(numeric.corr().round(4).to_dict()),
    }
    _shutdown_dtale(instance, build_shutdown_url)
    dtale.global_state.cleanup(instance._data_id)

    summary_path = output / "dtale_summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = summary_path.as_posix()
    summary["area_min"] = float(numeric["Area"].min())
    summary["area_max"] = float(numeric["Area"].max())
    summary["eccentricity_min"] = float(numeric["Eccentricity"].min())
    summary["eccentricity_max"] = float(numeric["Eccentricity"].max())
    return summary


def _shutdown_dtale(instance, build_shutdown_url) -> None:
    shutdown_url = build_shutdown_url(instance._url)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            requests.get(shutdown_url, timeout=2)
            return
        except requests.RequestException:
            time.sleep(0.4)


def _write_presentation(insights: dict, path: Path) -> None:
    presentation = Presentation()
    presentation.slide_width = Inches(13.333)
    presentation.slide_height = Inches(7.5)
    presentation.core_properties.title = "Raisin Get Data insights"

    missing_total = sum(insights["missing_counts"].values())
    _add_title_slide(presentation, insights, missing_total)
    _add_content_slide(presentation, "What was loaded", _loaded_lines(insights))
    _add_content_slide(presentation, "Missing values", _missing_lines(insights, missing_total))
    _add_content_slide(presentation, "Balance, duplicates, and assumptions", _assumption_lines(insights))
    _add_content_slide(presentation, "Schema and expectations", _contract_lines(insights))
    _add_content_slide(presentation, "Saved outputs", _output_lines(insights, path))
    try:
        presentation.save(path)
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        presentation.save(fallback)
        print(f"Could not overwrite {path.name} because it is open. Saved {fallback.as_posix()} instead.")


def _add_title_slide(presentation: Presentation, insights: dict, missing_total: int) -> None:
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    _fill_slide(slide, NAVY)
    _add_text(
        slide,
        "Get the Data",
        Inches(0.8),
        Inches(2.1),
        Inches(11.5),
        Inches(1.2),
        GOLD,
        48,
        bold=True,
    )
    _add_text(
        slide,
        "Raisin morphology for Kecimen and Besni",
        Inches(0.8),
        Inches(3.4),
        Inches(11.5),
        Inches(0.6),
        CREAM,
        24,
    )
    _add_text(
        slide,
        (
            f"{insights['rows']} rows, {insights['columns']} columns, "
            f"{missing_total} missing cells. Source: {insights['source_path']}"
        ),
        Inches(0.8),
        Inches(4.4),
        Inches(11.5),
        Inches(0.8),
        CREAM,
        16,
    )


def _add_content_slide(presentation: Presentation, title: str, lines: list[str]) -> None:
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    _fill_slide(slide, CREAM)
    header = slide.shapes.add_shape(1, Emu(0), Emu(0), presentation.slide_width, Inches(1.05))
    header.fill.solid()
    header.fill.fore_color.rgb = NAVY
    header.line.fill.background()
    accent = slide.shapes.add_shape(1, Emu(0), Inches(1.05), presentation.slide_width, Inches(0.08))
    accent.fill.solid()
    accent.fill.fore_color.rgb = GOLD
    accent.line.fill.background()
    _add_text(slide, title, Inches(0.5), Inches(0.22), Inches(12.2), Inches(0.65), CREAM, 28, bold=True)

    box = slide.shapes.add_textbox(Inches(0.55), Inches(1.4), Inches(12.2), Inches(5.7))
    frame = box.text_frame
    frame.word_wrap = True
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.level = 0
        paragraph.font.size = Pt(18)
        paragraph.font.color.rgb = INK
        paragraph.font.name = "Calibri"
        paragraph.space_after = Pt(12)


def _add_text(slide, text, left, top, width, height, color, size, bold=False) -> None:
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.font.size = Pt(size)
    paragraph.font.color.rgb = color
    paragraph.font.bold = bold
    paragraph.font.name = "Calibri"
    paragraph.alignment = PP_ALIGN.LEFT


def _fill_slide(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _loaded_lines(insights: dict) -> list[str]:
    dtype_text = ", ".join(
        f"{name} ({dtype})" for name, dtype in insights["dtypes"].items()
    )
    memory_mb = insights["memory_bytes"] / (1024 * 1024)
    dtale = insights["dtale"]
    return [
        f"Path: {insights['source_path']}. The loader returned the CSV unchanged.",
        f"Shape: {insights['rows']} rows and {insights['columns']} columns.",
        f"Types: {dtype_text}.",
        f"In-memory size: {memory_mb:.2f} MB.",
        (
            "D-Tale describes Area from "
            f"{dtale['area_min']:,.0f} to {dtale['area_max']:,.0f}, and Eccentricity from "
            f"{dtale['eccentricity_min']:.3f} to {dtale['eccentricity_max']:.3f}."
        ),
        "Those ranges match length and area measurements rather than coded categories.",
    ]


def _missing_lines(insights: dict, missing_total: int) -> list[str]:
    cells = insights["rows"] * insights["columns"] or 1
    percent = (missing_total / cells) * 100
    if missing_total == 0:
        interpretation = (
            "Every cell is filled. A later step that drops missing rows would leave this table unchanged."
        )
    else:
        gaps = ", ".join(
            f"{name} ({count})"
            for name, count in insights["missing_counts"].items()
            if count
        )
        interpretation = f"Missing cells are concentrated in: {gaps}."
    return [
        f"Missing cells: {missing_total} ({percent:.2f}% of the table).",
        interpretation,
        insights["missingno"]["heatmap_note"],
        "The matrix, bar chart, and heatmap are saved beside this deck.",
    ]


def _assumption_lines(insights: dict) -> list[str]:
    total = insights["rows"] or 1
    counts = insights["class_counts"]
    count_text = ", ".join(
        f"{label} {count} ({count / total:.1%})" for label, count in counts.items()
    )
    minority = min((count / total) for count in counts.values()) if counts else 0
    if minority >= 0.4:
        balance = "The split supports the assumption that the two varieties are fairly balanced."
    else:
        balance = "The split is uneven, so the 'fairly balanced' assumption does not hold."

    duplicates = insights["duplicate_rows"]
    if duplicates == 0:
        duplicate_text = (
            "There are no duplicate rows. That does not prove each raisin was measured independently, "
            "but the file does not repeat the same record."
        )
    else:
        duplicate_text = (
            f"The table has {duplicates} duplicate rows. Repeated records weaken the assumption "
            "that every row is an independent raisin."
        )
    return [
        f"Class counts: {count_text}.",
        balance,
        duplicate_text,
        "The file has no time or batch column, so independence cannot be tested beyond duplicate rows.",
        "Completeness and the range checks below are the available evidence that the measurements were recorded cleanly.",
    ]


def _contract_lines(insights: dict) -> list[str]:
    pandera = insights["pandera"]
    if pandera["success"]:
        pandera_text = (
            "Pandera passed. The schema rejects nulls, keeps Class inside Kecimen and Besni, "
            "requires the size measures to be at least zero, and keeps Eccentricity and Extent between 0 and 1."
        )
    else:
        failures = ", ".join(
            f"{name} ({count})" for name, count in list(pandera["failure_counts"].items())[:4]
        )
        pandera_text = f"Pandera failed. Largest mismatches: {failures}."

    gx = insights["great_expectations"]
    if gx["success"]:
        gx_text = (
            f"Great Expectations passed all {gx['evaluated']} expectations. "
            "The same column, label, and range contract holds in the validation result and data docs."
        )
    else:
        failed = "; ".join(gx["failed"][:4]) or "see the validation JSON"
        gx_text = (
            f"Great Expectations passed {gx['successful']} of {gx['evaluated']} expectations. "
            f"Failed: {failed}."
        )
    return [pandera_text, gx_text]


def _output_lines(insights: dict, deck_path: Path) -> list[str]:
    return [
        f"Deck: {deck_path.as_posix()}",
        f"D-Tale summary: {insights['dtale']['summary_path']}",
        f"Pandera schema: {insights['pandera']['schema_path']}",
        f"Great Expectations: {insights['great_expectations']['context_dir']}",
        (
            "missingno: "
            f"{insights['missingno']['matrix']}, "
            f"{insights['missingno']['bar']}, "
            f"{insights['missingno']['heatmap']}"
        ),
    ]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean_json(payload), indent=2), encoding="utf-8")


def _clean_json(value):
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return _clean_json(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, Path):
        return value.as_posix()
    return value
