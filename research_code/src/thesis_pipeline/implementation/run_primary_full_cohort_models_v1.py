# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Fit the all-eligible-point primary model ladder with grouped nested validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


MODEL_ORDER = [
    "null_training_mean",
    "null_minus_1db_reference",
    "location_ridge",
    "retfound_ridge",
    "location_retfound_ridge",
]
LOCATION = [
    "location_eccentricity_normalised",
    "location_angle_sin",
    "location_angle_cos",
]
IDENTITY = [
    "study_id", "case_id", "eye", "maia_timepoint", "outer_fold", "point_number",
]
PCA_GRID = [8, 16, 32, 64, 128]
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def design(model_id: str, representation: np.ndarray, location: np.ndarray, components: int) -> np.ndarray:
    if model_id == "retfound_ridge":
        return representation[:, :components]
    if model_id == "location_retfound_ridge":
        return np.column_stack([location, representation[:, :components]])
    if model_id == "location_ridge":
        return location
    raise ValueError(model_id)


def transform_representation(train: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    valid_scaled = scaler.transform(valid)
    n_components = min(128, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(
        n_components=n_components,
        whiten=True,
        svd_solver="randomized",
        iterated_power=4,
        n_oversamples=10,
        random_state=20260826,
    )
    return pca.fit_transform(train_scaled), pca.transform(valid_scaled), pca


def grouped_inner_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, int]]:
    splitter = GroupKFold(n_splits=4)
    return [
        (train_idx, valid_idx, split_index + 1)
        for split_index, (train_idx, valid_idx) in enumerate(
            splitter.split(frame, groups=frame["study_id"])
        )
    ]


def select_candidate(
    model_id: str,
    train_frame: pd.DataFrame,
    representation: np.ndarray,
    locations: np.ndarray,
    y: np.ndarray,
) -> tuple[dict, list[dict]]:
    candidates = (
        [{"pca_components": 0, "alpha": alpha} for alpha in ALPHA_GRID]
        if model_id == "location_ridge"
        else [
            {"pca_components": components, "alpha": alpha}
            for components in PCA_GRID
            for alpha in ALPHA_GRID
        ]
    )
    prepared = []
    for train_idx, valid_idx, inner_fold in grouped_inner_splits(train_frame):
        loc_scaler = StandardScaler()
        loc_train = loc_scaler.fit_transform(locations[train_idx])
        loc_valid = loc_scaler.transform(locations[valid_idx])
        if model_id == "location_ridge":
            rep_train = rep_valid = None
        else:
            rep_train, rep_valid, _ = transform_representation(
                representation[train_idx], representation[valid_idx]
            )
        prepared.append((train_idx, valid_idx, inner_fold, rep_train, rep_valid, loc_train, loc_valid))

    scores = []
    for candidate_index, candidate in enumerate(candidates):
        fold_scores = []
        for train_idx, valid_idx, inner_fold, rep_train, rep_valid, loc_train, loc_valid in prepared:
            x_train = loc_train if model_id == "location_ridge" else design(model_id, rep_train, loc_train, int(candidate["pca_components"]))
            x_valid = loc_valid if model_id == "location_ridge" else design(model_id, rep_valid, loc_valid, int(candidate["pca_components"]))
            fitted = Ridge(alpha=float(candidate["alpha"]))
            fitted.fit(x_train, y[train_idx])
            fold_scores.append(float(np.mean(np.abs(fitted.predict(x_valid) - y[valid_idx]))))
        scores.append({
            "candidate_index": candidate_index,
            **candidate,
            **{f"inner_{fold}_mae": value for fold, value in enumerate(fold_scores, start=1)},
            "mean_inner_mae": float(np.mean(fold_scores)),
            "std_inner_mae": float(np.std(fold_scores)),
        })
    selected = sorted(scores, key=lambda row: (row["mean_inner_mae"], row["candidate_index"]))[0]
    selected_index = selected["candidate_index"]
    for row in scores:
        row["selected"] = row["candidate_index"] == selected_index
    return selected, scores


def run(table_path: Path, representation_path: Path, output_root: Path, seed: int = 20260826) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    rows = pd.read_csv(table_path).sort_values("point_embedding_row_index").reset_index(drop=True)
    representation = np.load(representation_path, mmap_mode="r", allow_pickle=False)
    if tuple(representation.shape) != (2886, 2048):
        raise RuntimeError(f"Unexpected primary representation shape: {representation.shape}")
    if len(rows) != 2886 or not np.isfinite(representation).all():
        raise RuntimeError("Primary modelling inputs are incomplete or non-finite")
    prediction_rows: list[dict] = []
    tuning_rows: list[dict] = []
    configuration_rows: list[dict] = []
    pca_rows: list[dict] = []

    for outer_fold in sorted(rows["outer_fold"].astype(int).unique()):
        train_mask = rows["outer_fold"].astype(int).ne(outer_fold).to_numpy()
        test_mask = ~train_mask
        train = rows.loc[train_mask].reset_index(drop=True)
        test = rows.loc[test_mask].reset_index(drop=True)
        train_indices = rows.index[train_mask].to_numpy()
        test_indices = rows.index[test_mask].to_numpy()
        y_train = train["sensitivity_db"].to_numpy(float)
        y_test = test["sensitivity_db"].to_numpy(float)
        location_train = train[LOCATION].to_numpy(float)
        location_test = test[LOCATION].to_numpy(float)

        # Null and fixed floor are deliberately not learned from OCT or location.
        for position, model_id, prediction in [
            (1, "null_training_mean", np.repeat(float(y_train.mean()), len(test))),
            (2, "null_minus_1db_reference", np.repeat(-1.0, len(test))),
        ]:
            configuration_rows.append({
                "model_position": position, "model_id": model_id, "outer_fold": outer_fold,
                "role": "fixed baseline", "train_points": len(train), "test_points": len(test),
                "train_participants": train.study_id.nunique(), "test_participants": test.study_id.nunique(),
                "selected_parameters_json": json.dumps({"value": float(prediction[0])}),
            })
            for source, predicted in zip(test.to_dict("records"), prediction):
                prediction_rows.append({
                    "model_position": position, "model_id": model_id,
                    "source_row_index": int(source["point_embedding_row_index"]),
                    **{field: source[field] for field in IDENTITY},
                    "sensitivity_db": float(source["sensitivity_db"]),
                    "sensitivity_floor_flag": bool(source["sensitivity_floor_flag"]),
                    "prediction_db": float(predicted),
                    "prediction_below_minus_1": bool(predicted < -1),
                    "prediction_above_33": bool(predicted > 33),
                })

        location_scaler = StandardScaler()
        location_train_scaled = location_scaler.fit_transform(location_train)
        location_test_scaled = location_scaler.transform(location_test)
        rep_outer_train, rep_outer_test, outer_pca = transform_representation(
            np.asarray(representation[train_indices]), np.asarray(representation[test_indices])
        )
        pca_cumulative = np.cumsum(outer_pca.explained_variance_ratio_)
        for component, (ratio, cumulative) in enumerate(zip(outer_pca.explained_variance_ratio_, pca_cumulative), start=1):
            pca_rows.append({
                "outer_fold": outer_fold, "component": component,
                "explained_variance_ratio": float(ratio),
                "cumulative_explained_variance_ratio": float(cumulative),
            })

        for position, model_id in [(3, "location_ridge"), (4, "retfound_ridge"), (5, "location_retfound_ridge")]:
            if model_id == "location_ridge":
                rep_train = rep_test = np.empty((len(train), 0))
                loc_for_selection = location_train
                loc_test_for_model = location_test
            else:
                rep_train = np.asarray(representation[train_indices])
                rep_test = np.asarray(representation[test_indices])
                loc_for_selection = location_train
                loc_test_for_model = location_test
            selected, scores = select_candidate(
                model_id, train, rep_train, loc_for_selection, y_train
            )
            for row in scores:
                tuning_rows.append({"outer_fold": outer_fold, "model_position": position, "model_id": model_id, **row})
            components = int(selected["pca_components"])
            alpha = float(selected["alpha"])
            if model_id == "location_ridge":
                x_train, x_test = location_train_scaled, location_test_scaled
            else:
                x_train = design(model_id, rep_outer_train, location_train_scaled, components)
                x_test = design(model_id, rep_outer_test, location_test_scaled, components)
            fitted = Ridge(alpha=alpha)
            fitted.fit(x_train, y_train)
            prediction = fitted.predict(x_test)
            configuration_rows.append({
                "model_position": position, "model_id": model_id, "outer_fold": outer_fold,
                "role": "nested participant-grouped ridge", "train_points": len(train), "test_points": len(test),
                "train_participants": train.study_id.nunique(), "test_participants": test.study_id.nunique(),
                "selected_candidate_index": int(selected["candidate_index"]),
                "selected_parameters_json": json.dumps({"alpha": alpha, "pca_components": components}),
                "selected_mean_inner_mae": float(selected["mean_inner_mae"]),
                "predictor_count": int(x_train.shape[1]),
                "standardised_intercept": float(fitted.intercept_),
            })
            for source, predicted in zip(test.to_dict("records"), prediction):
                prediction_rows.append({
                    "model_position": position, "model_id": model_id,
                    "source_row_index": int(source["point_embedding_row_index"]),
                    **{field: source[field] for field in IDENTITY},
                    "sensitivity_db": float(source["sensitivity_db"]),
                    "sensitivity_floor_flag": bool(source["sensitivity_floor_flag"]),
                    "prediction_db": float(predicted),
                    "prediction_below_minus_1": bool(predicted < -1),
                    "prediction_above_33": bool(predicted > 33),
                })
            print(f"[outer {outer_fold}/5] fitted {model_id}", flush=True)

    predictions = pd.DataFrame(prediction_rows).sort_values(["model_position", "source_row_index"]).reset_index(drop=True)
    tuning = pd.DataFrame(tuning_rows).sort_values(["model_position", "outer_fold", "candidate_index"]).reset_index(drop=True)
    configurations = pd.DataFrame(configuration_rows).sort_values(["model_position", "outer_fold"]).reset_index(drop=True)
    pca = pd.DataFrame(pca_rows).sort_values(["outer_fold", "component"]).reset_index(drop=True)
    checks = {
        "five_models": predictions.model_id.nunique() == 5,
        "model_order_exact": predictions[["model_position", "model_id"]].drop_duplicates().sort_values("model_position")["model_id"].tolist() == MODEL_ORDER,
        "prediction_rows_equal_14430": len(predictions) == 2886 * 5,
        "one_prediction_per_point_model": not predictions.duplicated(["model_id", "source_row_index"]).any(),
        "finite_predictions": bool(np.isfinite(predictions.prediction_db).all()),
        "configurations_equal_25": len(configurations) == 25,
        "participant_fold_isolation": rows.groupby("study_id").outer_fold.nunique().eq(1).all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, value in checks.items() if not bool(value)]
    if failed:
        raise RuntimeError(f"Primary model checks failed: {failed}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building_", dir=output_root.parent))
    try:
        predictions.to_csv(building / "oof_predictions_primary_v1.csv", index=False, lineterminator="\n")
        tuning.to_csv(building / "inner_cv_tuning_primary_v1.csv", index=False, lineterminator="\n")
        configurations.to_csv(building / "outer_fold_configurations_primary_v1.csv", index=False, lineterminator="\n")
        pca.to_csv(building / "pca_diagnostics_primary_v1.csv", index=False, lineterminator="\n")
        metadata = {
            "status": "passed_primary_full_cohort_oof_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": MODEL_ORDER,
            "counts": {"participants": 22, "eye_visits": 78, "points": 2886, "outer_folds": 5, "inner_folds": 4},
            "table_sha256": sha256(table_path),
            "representation_sha256": sha256(representation_path),
            "runtime": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
            "checks": checks,
            "validation": "fixed participant-grouped outer folds; grouped inner folds; no participant appears in both train and test",
            "primary_endpoint": "all eligible pointwise sensitivity rows; boundary availability is not an inclusion criterion",
        }
        (building / "PRIMARY_FULL_COHORT_MODEL_RUN_V1.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (building / "README.md").write_text(
            "# Primary full-cohort model run v1\n\n"
            "Out-of-fold predictions for the all-eligible-point primary ladder. The fixed -1 dB "
            "reference is a descriptive floor baseline, not a learned model. Metrics are calculated "
            "only in the following evaluation stage.\n", encoding="utf-8"
        )
        building.rename(output_root)
        return metadata
    finally:
        if building.exists():
            shutil.rmtree(building, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    print(json.dumps(run(args.table.resolve(), args.representations.resolve(), args.output_root.resolve(), args.seed), indent=2))


if __name__ == "__main__":
    main()
