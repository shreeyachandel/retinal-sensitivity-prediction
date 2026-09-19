# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

"""Error and sensitivity analyses calculated only from frozen OOF predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    observed = frame["sensitivity_db"].to_numpy(float)
    predicted = frame["prediction_db"].to_numpy(float)
    difference = predicted - observed
    sd = difference.std(ddof=1) if len(difference) > 1 else np.nan
    return {
        "n_points": len(frame),
        "mae_db": float(np.mean(np.abs(difference))),
        "rmse_db": float(np.sqrt(np.mean(difference ** 2))),
        "bland_altman_bias_db": float(np.mean(difference)),
        "bland_altman_lower_loa_db": float(np.mean(difference) - 1.96 * sd),
        "bland_altman_upper_loa_db": float(np.mean(difference) + 1.96 * sd),
    }


def generate(predictions: pd.DataFrame, output_root: Path) -> dict:
    """Create prespecified descriptive robustness tables without refitting."""
    output_root.mkdir(parents=True, exist_ok=True)
    required = {"model_id", "study_id", "outer_fold", "sensitivity_db", "prediction_db", "sensitivity_floor_flag"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"OOF predictions lack required columns: {missing}")

    participant_rows = []
    fold_rows = []
    stratum_rows = []
    for model_id, model in predictions.groupby("model_id", sort=False):
        floor_flag = (
            model["sensitivity_floor_flag"].astype(str).str.lower().eq("true")
            if model["sensitivity_floor_flag"].dtype == object
            else model["sensitivity_floor_flag"].astype(bool)
        )
        for participant, frame in model.groupby("study_id"):
            participant_rows.append({"model_id": model_id, "study_id": participant, **_metrics(frame)})
        for outer_fold, frame in model.groupby("outer_fold"):
            fold_rows.append({"model_id": model_id, "outer_fold": int(outer_fold), **_metrics(frame)})
        strata = {
            "minus_1db_floor": model[floor_flag],
            "above_floor": model[~floor_flag],
            "observed_below_10db": model[(model["sensitivity_db"] > -1) & (model["sensitivity_db"] < 10)],
            "observed_10_to_20db": model[(model["sensitivity_db"] >= 10) & (model["sensitivity_db"] < 20)],
            "observed_20db_or_more": model[model["sensitivity_db"] >= 20],
        }
        for stratum, frame in strata.items():
            if len(frame):
                stratum_rows.append({"model_id": model_id, "stratum": stratum, **_metrics(frame)})

    tables = {
        "participant_error_metrics.csv": pd.DataFrame(participant_rows),
        "outer_fold_error_metrics.csv": pd.DataFrame(fold_rows),
        "sensitivity_stratum_metrics.csv": pd.DataFrame(stratum_rows),
    }
    for name, frame in tables.items():
        frame.to_csv(output_root / name, index=False, lineterminator="\n")
    return {
        "status": "passed",
        "tables": len(tables),
        "analysis_root": str(output_root),
        "rules": [
            "all analyses use frozen participant-held-out OOF predictions",
            "no model is refitted or selected in this stage",
            "floor and above-floor results are sensitivity analyses, not replacements for the all-point endpoint",
        ],
    }
