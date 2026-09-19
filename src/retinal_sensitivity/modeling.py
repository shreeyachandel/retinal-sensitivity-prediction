"""Participant-grouped model evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS: dict[str, list[str]] = {
    "Location ridge": ["eccentricity", "angle_sin", "angle_cos"],
    "Structure + location ridge": [
        "eccentricity",
        "angle_sin",
        "angle_cos",
        "retinal_thickness",
        "reflectivity",
        "roughness",
    ],
    "Structure + location forest": [
        "eccentricity",
        "angle_sin",
        "angle_cos",
        "retinal_thickness",
        "reflectivity",
        "roughness",
    ],
}


def build_models(random_state: int = 42) -> dict[str, RegressorMixin]:
    """Return small, transparent models suitable for the synthetic demo."""

    return {
        "Location ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Structure + location ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Structure + location forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def grouped_predictions(
    data: pd.DataFrame,
    models: Mapping[str, RegressorMixin] | None = None,
    *,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Generate out-of-fold predictions with participants kept intact.

    Each participant is assigned wholly to either train or test data in a fold.
    The returned table contains synthetic identifiers only and is safe to use in
    the demo output.
    """

    models = models or build_models()
    missing_models = set(models) - set(FEATURE_SETS)
    if missing_models:
        raise ValueError(f"No feature set configured for: {sorted(missing_models)}")

    groups = data["participant_id"].to_numpy()
    target = data["sensitivity_db"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    prediction_frames: list[pd.DataFrame] = []

    for model_name, estimator in models.items():
        features = FEATURE_SETS[model_name]
        matrix = data[features].to_numpy()
        predictions = np.full(len(data), np.nan, dtype=float)
        fold_numbers = np.full(len(data), -1, dtype=int)

        for fold_number, (train_indices, test_indices) in enumerate(
            splitter.split(matrix, target, groups), start=1
        ):
            train_groups = set(groups[train_indices])
            test_groups = set(groups[test_indices])
            if train_groups & test_groups:
                raise RuntimeError("Participant leakage detected between train and test sets")

            fitted_model = clone(estimator)
            fitted_model.fit(matrix[train_indices], target[train_indices])
            predictions[test_indices] = fitted_model.predict(matrix[test_indices])
            fold_numbers[test_indices] = fold_number

        if np.isnan(predictions).any() or (fold_numbers < 1).any():
            raise RuntimeError(f"Incomplete out-of-fold predictions for {model_name}")

        prediction_frames.append(
            pd.DataFrame(
                {
                    "participant_id": groups,
                    "observed_db": target,
                    "predicted_db": predictions,
                    "model": model_name,
                    "fold": fold_numbers,
                }
            )
        )

    return pd.concat(prediction_frames, ignore_index=True)

