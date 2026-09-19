# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Evaluate combined-visit OOF predictions for error, calibration and agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_NAMES = ("mae_db", "rmse_db", "bland_altman_bias_db", "bland_altman_lower_loa_db", "bland_altman_upper_loa_db", "calibration_intercept", "calibration_slope", "r_squared")


def metrics(frame: pd.DataFrame, weights: np.ndarray | None = None) -> dict:
    observed = frame["sensitivity_db"].to_numpy(float)
    predicted = frame["prediction_db"].to_numpy(float)
    weights = np.ones(len(frame), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    keep = weights > 0
    observed, predicted, weights = observed[keep], predicted[keep], weights[keep]
    difference = predicted - observed
    total = float(weights.sum())
    bias = float(weights @ difference / total)
    variance = float((weights @ difference**2 - total * bias**2) / max(total - 1, 1))
    sd = float(np.sqrt(max(variance, 0)))
    design = np.column_stack([np.ones(len(predicted)), predicted])
    root_w = np.sqrt(weights)
    intercept, slope = np.linalg.lstsq(design * root_w[:, None], observed * root_w, rcond=None)[0]
    observed_mean = float(weights @ observed / total)
    squared_error = float(weights @ difference**2)
    total_variance = float(weights @ (observed - observed_mean)**2)
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


def evaluated(frame: pd.DataFrame, repetitions: int, seed: int) -> dict:
    point = metrics(frame)
    participants = sorted(frame["study_id"].astype(str).unique())
    participant_index = {value: index for index, value in enumerate(participants)}
    codes = frame["study_id"].astype(str).map(participant_index).to_numpy(int)
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(repetitions) for name in METRIC_NAMES}
    for repetition in range(repetitions):
        draw = rng.integers(0, len(participants), len(participants))
        counts = np.bincount(draw, minlength=len(participants)).astype(float)
        result = metrics(frame, counts[codes])
        for name in METRIC_NAMES:
            samples[name][repetition] = result[name]
    for name, values in samples.items():
        finite = values[np.isfinite(values)]
        low, high = (np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan))
        point[f"{name}_ci95_low"] = float(low); point[f"{name}_ci95_high"] = float(high)
    return point


def run(prediction_path: Path, output_root: Path, bootstrap_repetitions: int) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_csv(prediction_path)
    overall, strata = [], []
    for (position, model_id), frame in predictions.groupby(["model_position", "model_id"], sort=True):
        overall.append({"model_position": position, "model_id": model_id,
            "n_participants": frame.study_id.nunique(), "n_cases": frame.case_id.nunique(), "n_points": len(frame), **evaluated(frame, bootstrap_repetitions, 20260826 + int(position) * 1000)})
        for name, subset in {
            "baseline_visit": frame[frame.maia_timepoint.eq("BL")],
            "year_one_visit": frame[frame.maia_timepoint.eq("Y01")],
            "minus_1db_floor": frame[frame.sensitivity_floor_flag],
            "above_floor": frame[~frame.sensitivity_floor_flag],
        }.items():
            strata.append({"model_position": position, "model_id": model_id, "stratum": name,
                "n_participants": subset.study_id.nunique(), "n_cases": subset.case_id.nunique(), "n_points": len(subset), **evaluated(subset, bootstrap_repetitions, 20260826 + int(position) * 1000 + len(strata))})
    overall = pd.DataFrame(overall).sort_values("model_position")
    strata = pd.DataFrame(strata)
    ranking = overall.sort_values(["mae_db", "model_position"]).reset_index(drop=True)
    ranking.insert(0, "mae_rank", np.arange(1, len(ranking) + 1))
    overall.to_csv(output_root / "all_point_metrics_v2.csv", index=False, lineterminator="\n")
    strata.to_csv(output_root / "visit_and_floor_stratum_metrics_v2.csv", index=False, lineterminator="\n")
    ranking.to_csv(output_root / "model_ranking_by_mae_v2.csv", index=False, lineterminator="\n")
    checks = {"nine_models": len(overall) == 9, "four_strata_each": len(strata) == 36,
        "finite_primary_metrics": bool(np.isfinite(overall[["mae_db","rmse_db","bland_altman_bias_db","bland_altman_lower_loa_db","bland_altman_upper_loa_db"]]).all().all()),
        "limits_ordered": bool((overall.bland_altman_lower_loa_db < overall.bland_altman_bias_db).all() and (overall.bland_altman_bias_db < overall.bland_altman_upper_loa_db).all())}
    result = {"status": "passed_combined_error_agreement_evaluation_v2" if all(checks.values()) else "failed",
        "difference_definition": "prediction minus observed sensitivity", "bootstrap_unit": "participant", "bootstrap_repetitions": bootstrap_repetitions, "checks": checks}
    (output_root / "COMBINED_EVALUATION_V2.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] == "failed": raise RuntimeError(result)
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--predictions",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--bootstrap-repetitions",type=int,default=5000)
    a=p.parse_args(); print(json.dumps(run(a.predictions.resolve(),a.output_root.resolve(),a.bootstrap_repetitions),indent=2))


if __name__ == "__main__": main()
