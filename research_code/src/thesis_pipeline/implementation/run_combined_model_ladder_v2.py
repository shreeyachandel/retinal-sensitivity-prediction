# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Run nine prespecified comparators on combined visits with grouped nested validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LOCATION = ["location_eccentricity_normalised", "location_angle_sin", "location_angle_cos"]
BOUNDARY = [f"B{i}_B{i + 1}_{s}" for i in range(4) for s in ("mean_um", "sd_um", "between_scan_mean_sd_um")]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MODELS = [
    ("null_training_mean", "mean", []),
    ("null_minus_1db_reference", "fixed", []),
    ("location_ridge", "ridge", LOCATION),
    ("boundary_linear", "linear", BOUNDARY),
    ("boundary_ridge", "ridge", BOUNDARY),
    ("location_boundary_ridge", "ridge", LOCATION + BOUNDARY),
    ("location_boundary_random_forest", "rf", LOCATION + BOUNDARY),
    ("boundary_aligned_retfound_ridge", "retfound", []),
    ("location_boundary_retfound_ridge", "retfound", LOCATION + BOUNDARY),
]


def choose_alpha_tabular(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    splitter = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    scores = {alpha: [] for alpha in RIDGE_ALPHAS}
    for train, valid in splitter.split(x, y, groups):
        for alpha in RIDGE_ALPHAS:
            fitted = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)).fit(x[train], y[train])
            scores[alpha].append(mean_absolute_error(y[valid], fitted.predict(x[valid])))
    return min(RIDGE_ALPHAS, key=lambda a: (np.mean(scores[a]), a))


def scale_pca(train: np.ndarray, valid: np.ndarray, seed: int):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    valid_scaled = scaler.transform(valid)
    components = min(64, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(train_scaled), pca.transform(valid_scaled)


def retfound_design(tab_train: np.ndarray, tab_valid: np.ndarray, emb_train: np.ndarray,
                    emb_valid: np.ndarray, seed: int):
    pc_train, pc_valid = scale_pca(emb_train, emb_valid, seed)
    if tab_train.shape[1] == 0:
        return pc_train, pc_valid
    imputer = SimpleImputer(strategy="median")
    tab_train = imputer.fit_transform(tab_train)
    tab_valid = imputer.transform(tab_valid)
    scaler = StandardScaler()
    z_train = scaler.fit_transform(tab_train)
    z_valid = scaler.transform(tab_valid)
    return np.column_stack([z_train, pc_train]), np.column_stack([z_valid, pc_valid])


def choose_alpha_retfound(tab: np.ndarray, emb: np.ndarray, y: np.ndarray,
                          groups: np.ndarray, seed: int) -> float:
    splitter = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    scores = {alpha: [] for alpha in RIDGE_ALPHAS}
    for split_number, (train, valid) in enumerate(splitter.split(emb, y, groups), start=1):
        x_train, x_valid = retfound_design(tab[train], tab[valid], emb[train], emb[valid], seed + split_number)
        for alpha in RIDGE_ALPHAS:
            fitted = Ridge(alpha=alpha).fit(x_train, y[train])
            scores[alpha].append(mean_absolute_error(y[valid], fitted.predict(x_valid)))
    return min(RIDGE_ALPHAS, key=lambda a: (np.mean(scores[a]), a))


def run(table_path: Path, embedding_rows_path: Path, representation_path: Path,
        output_root: Path, seed: int) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    table = pd.read_csv(table_path).sort_values(["case_id", "point_number"]).reset_index(drop=True)
    embedding_rows = pd.read_csv(embedding_rows_path).sort_values(["case_id", "point_number"]).reset_index(drop=True)
    embeddings = np.load(representation_path, mmap_mode="r")
    if not table[["case_id", "point_number"]].equals(embedding_rows[["case_id", "point_number"]]):
        raise RuntimeError("RETFound rows do not match the combined modelling table")
    if embeddings.shape != (len(table), 3072) or not np.isfinite(embeddings).all():
        raise RuntimeError("Invalid boundary-aligned RETFound representation")

    prediction_rows, tuning_rows = [], []
    for outer_fold in sorted(table["outer_fold"].astype(int).unique()):
        train_mask = table["outer_fold"].astype(int).ne(outer_fold).to_numpy()
        test_mask = ~train_mask
        train, test = table.loc[train_mask], table.loc[test_mask]
        y_train = train["sensitivity_db"].to_numpy(float)
        groups = train["study_id"].astype(str).to_numpy()
        for position, (model_id, family, columns) in enumerate(MODELS, start=1):
            model_seed = seed + 100 * outer_fold + position
            selected = {}
            if family == "mean":
                selected = {"training_mean_db": float(y_train.mean())}
                prediction = np.repeat(y_train.mean(), len(test))
            elif family == "fixed":
                selected = {"fixed_prediction_db": -1.0}
                prediction = np.repeat(-1.0, len(test))
            elif family == "linear":
                fitted = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()).fit(train[columns], y_train)
                prediction = fitted.predict(test[columns])
            elif family == "ridge":
                alpha = choose_alpha_tabular(train[columns].to_numpy(float), y_train, groups)
                selected = {"alpha": alpha}
                fitted = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)).fit(train[columns], y_train)
                prediction = fitted.predict(test[columns])
            elif family == "rf":
                candidates = [
                    {"max_depth": depth, "min_samples_leaf": leaf}
                    for depth in (4, 8, None) for leaf in (5, 15)
                ]
                splitter = GroupKFold(n_splits=min(4, len(np.unique(groups))))
                scored = []
                for candidate in candidates:
                    scores = []
                    for a, b in splitter.split(train, y_train, groups):
                        fitted = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=500, max_features="sqrt", n_jobs=-1,
                            random_state=model_seed, **candidate)).fit(train.iloc[a][columns], y_train[a])
                        scores.append(mean_absolute_error(y_train[b], fitted.predict(train.iloc[b][columns])))
                    scored.append((np.mean(scores), candidate))
                selected = min(scored, key=lambda item: (item[0], json.dumps(item[1], sort_keys=True)))[1]
                fitted = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=1000, max_features="sqrt", n_jobs=-1,
                    random_state=model_seed, **selected)).fit(train[columns], y_train)
                prediction = fitted.predict(test[columns])
            else:
                tab_train = train[columns].to_numpy(float) if columns else np.empty((len(train), 0))
                tab_test = test[columns].to_numpy(float) if columns else np.empty((len(test), 0))
                emb_train, emb_test = np.asarray(embeddings[train_mask]), np.asarray(embeddings[test_mask])
                alpha = choose_alpha_retfound(tab_train, emb_train, y_train, groups, model_seed)
                selected = {"alpha": alpha, "pca_components": min(64, len(train) - 1, embeddings.shape[1])}
                x_train, x_test = retfound_design(tab_train, tab_test, emb_train, emb_test, model_seed)
                prediction = Ridge(alpha=alpha).fit(x_train, y_train).predict(x_test)
            tuning_rows.append({"model_id": model_id, "outer_fold": outer_fold, "selected_parameters_json": json.dumps(selected, sort_keys=True)})
            for (_, row), value in zip(test.iterrows(), prediction):
                prediction_rows.append({
                    "model_position": position, "model_id": model_id, "study_id": row.study_id,
                    "case_id": row.case_id, "maia_timepoint": row.maia_timepoint,
                    "point_number": int(row.point_number), "outer_fold": outer_fold,
                    "sensitivity_db": float(row.sensitivity_db),
                    "sensitivity_floor_flag": bool(row.sensitivity_floor_flag),
                    "prediction_db": float(value), "selected_parameters_json": json.dumps(selected, sort_keys=True),
                })
    predictions = pd.DataFrame(prediction_rows)
    checks = {
        "nine_models": predictions["model_id"].nunique() == 9,
        "one_oof_prediction_per_model_point": not predictions[["model_id", "case_id", "point_number"]].duplicated().any(),
        "complete_rows": len(predictions) == len(table) * 9,
        "predictions_finite": bool(np.isfinite(predictions["prediction_db"]).all()),
        "participant_outer_fold_isolation": bool(predictions.groupby("study_id")["outer_fold"].nunique().eq(1).all()),
    }
    predictions.to_csv(output_root / "oof_predictions_v2.csv", index=False, lineterminator="\n")
    pd.DataFrame(tuning_rows).to_csv(output_root / "selected_configurations_v2.csv", index=False, lineterminator="\n")
    result = {"status": "passed_combined_nine_model_oof_v2" if all(checks.values()) else "failed", "models": [m[0] for m in MODELS], "points": len(table), "checks": checks}
    (output_root / "COMBINED_MODEL_RUN_V2.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] == "failed": raise RuntimeError(result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, required=True); p.add_argument("--embedding-rows", type=Path, required=True)
    p.add_argument("--representations", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260826)
    a = p.parse_args(); print(json.dumps(run(a.table.resolve(), a.embedding_rows.resolve(), a.representations.resolve(), a.output_root.resolve(), a.seed), indent=2))


if __name__ == "__main__": main()
