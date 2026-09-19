# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Evaluate primary full-cohort OOF predictions for error and agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "null_training_mean",
    "null_minus_1db_reference",
    "location_ridge",
    "retfound_ridge",
    "location_retfound_ridge",
]
METRICS = (
    "mae_db", "rmse_db", "bland_altman_bias_db", "bland_altman_lower_loa_db",
    "bland_altman_upper_loa_db", "calibration_intercept", "calibration_slope", "r_squared",
)


def calculate(frame: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, float]:
    observed = frame.sensitivity_db.to_numpy(float)
    predicted = frame.prediction_db.to_numpy(float)
    weights = np.ones(len(frame), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    keep = weights > 0
    observed, predicted, weights = observed[keep], predicted[keep], weights[keep]
    total = float(weights.sum())
    difference = predicted - observed
    bias = float(weights @ difference / total)
    variance = float((weights @ difference**2 - total * bias**2) / max(total - 1, 1))
    sd = float(np.sqrt(max(variance, 0.0)))
    root_w = np.sqrt(weights)
    design = np.column_stack([np.ones(len(predicted)), predicted])
    intercept, slope = np.linalg.lstsq(design * root_w[:, None], observed * root_w, rcond=None)[0]
    observed_mean = float(weights @ observed / total)
    squared_error = float(weights @ difference**2)
    total_variance = float(weights @ (observed - observed_mean) ** 2)
    return {
        "mae_db": float(weights @ np.abs(difference) / total),
        "rmse_db": float(np.sqrt(squared_error / total)),
        "bland_altman_bias_db": bias,
        "bland_altman_lower_loa_db": bias - 1.96 * sd,
        "bland_altman_upper_loa_db": bias + 1.96 * sd,
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "r_squared": float(1 - squared_error / total_variance) if total_variance > 0 else float("nan"),
    }


def bootstrap(frame: pd.DataFrame, repetitions: int, seed: int) -> dict[str, float]:
    result = calculate(frame)
    participants = sorted(frame.study_id.astype(str).unique())
    codes = frame.study_id.astype(str).map({value: i for i, value in enumerate(participants)}).to_numpy(int)
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(repetitions) for name in METRICS}
    for index in range(repetitions):
        draw = rng.integers(0, len(participants), len(participants))
        counts = np.bincount(draw, minlength=len(participants)).astype(float)
        sampled = calculate(frame, counts[codes])
        for name in METRICS:
            samples[name][index] = sampled[name]
    for name, values in samples.items():
        finite = values[np.isfinite(values)]
        low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
        result[f"{name}_ci95_low"] = float(low)
        result[f"{name}_ci95_high"] = float(high)
    return result


def run(predictions_path: Path, output_root: Path, bootstrap_repetitions: int = 5000) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    predictions = pd.read_csv(predictions_path)
    required = {"model_position", "model_id", "study_id", "case_id", "maia_timepoint", "sensitivity_db", "prediction_db", "sensitivity_floor_flag"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise RuntimeError(f"Primary predictions lack required columns: {missing}")
    floor_flag = predictions["sensitivity_floor_flag"]
    if floor_flag.dtype == object:
        floor_flag = floor_flag.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    else:
        floor_flag = floor_flag.astype(bool)
    predictions = predictions.copy()
    predictions["sensitivity_floor_flag"] = floor_flag
    overall, strata = [], []
    for position, model_id in zip(range(1, len(MODEL_ORDER) + 1), MODEL_ORDER):
        frame = predictions[predictions.model_id.eq(model_id)].copy()
        if len(frame) != 2886:
            raise RuntimeError(f"{model_id} has {len(frame)} predictions, expected 2886")
        overall.append({
            "model_position": position, "model_id": model_id,
            "n_participants": frame.study_id.nunique(), "n_cases": frame.case_id.nunique(),
            "n_points": len(frame),
            **bootstrap(frame, bootstrap_repetitions, 20260826 + position * 1000),
        })
        subsets = {
            "baseline_visit": frame[frame.maia_timepoint.eq("BL")],
            "year_one_visit": frame[frame.maia_timepoint.eq("Y01")],
            "minus_1db_floor": frame[frame.sensitivity_floor_flag.astype(bool)],
            "above_floor": frame[~frame.sensitivity_floor_flag.astype(bool)],
        }
        for stratum, subset in subsets.items():
            if len(subset):
                strata.append({
                    "model_position": position, "model_id": model_id, "stratum": stratum,
                    "n_participants": subset.study_id.nunique(), "n_cases": subset.case_id.nunique(),
                    "n_points": len(subset),
                    **bootstrap(subset, bootstrap_repetitions, 20260826 + position * 1000 + len(strata)),
                })
    overall = pd.DataFrame(overall).sort_values("model_position")
    strata = pd.DataFrame(strata)
    ranking = overall.sort_values(["mae_db", "model_position"]).reset_index(drop=True)
    ranking.insert(0, "mae_rank", np.arange(1, len(ranking) + 1))
    output_root.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output_root / "all_point_metrics_primary_v1.csv", index=False, lineterminator="\n")
    strata.to_csv(output_root / "visit_and_floor_stratum_metrics_primary_v1.csv", index=False, lineterminator="\n")
    ranking.to_csv(output_root / "model_ranking_by_mae_primary_v1.csv", index=False, lineterminator="\n")
    checks = {
        "five_models": len(overall) == 5,
        "four_strata_each": len(strata) == 20,
        "finite_error_agreement_metrics": bool(np.isfinite(overall[["mae_db", "rmse_db", "bland_altman_bias_db", "bland_altman_lower_loa_db", "bland_altman_upper_loa_db"]]).all().all()),
        "limits_ordered": bool((overall.bland_altman_lower_loa_db < overall.bland_altman_bias_db).all() & (overall.bland_altman_bias_db < overall.bland_altman_upper_loa_db).all()),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "status": "passed_primary_full_cohort_error_agreement_evaluation_v1" if all(checks.values()) else "failed",
        "models": MODEL_ORDER,
        "difference_definition": "prediction minus observed sensitivity",
        "bootstrap_unit": "participant",
        "bootstrap_repetitions": bootstrap_repetitions,
        "checks": checks,
    }
    (output_root / "PRIMARY_FULL_COHORT_EVALUATION_V1.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] == "failed":
        raise RuntimeError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(run(args.predictions.resolve(), args.output_root.resolve(), args.bootstrap_repetitions), indent=2))


if __name__ == "__main__":
    main()
