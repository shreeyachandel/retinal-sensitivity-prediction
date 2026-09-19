# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

"""Deterministic, code-generated result figures."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def sampling_schematic(path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(8, 3.8))
    for rank, y in enumerate([2, 1, 0], start=1):
        axis.hlines(y, -0.18, 0.18, color="#525252", linewidth=2)
        samples = np.linspace(-0.05, 0.05, 17)
        axis.scatter(samples, np.repeat(y, len(samples)), s=18, color="#2563eb")
        axis.scatter([0], [y], s=70, marker="x", color="#dc2626", linewidth=2)
        axis.text(0.19, y, f"nearest B-scan {rank}", va="center")
    axis.axvspan(-0.05, 0.05, color="#93c5fd", alpha=0.25, label="100 µm footprint")
    axis.set(xlabel="Along-scan distance from registered MAIA point (mm)", yticks=[])
    axis.set_xlim(-0.20, 0.34)
    axis.legend(frameon=False, loc="upper left")
    axis.spines[["left", "right", "top"]].set_visible(False)
    figure.tight_layout()
    return _save(figure, path)


def _panel_grid(predictions: pd.DataFrame):
    models = predictions["model_id"].drop_duplicates().tolist()
    rows = math.ceil(len(models) / 2)
    figure, axes = plt.subplots(rows, 2, figsize=(10, 4.2 * rows), squeeze=False)
    return models, figure, axes.ravel()


def observed_predicted(predictions: pd.DataFrame, path: Path) -> Path:
    models, figure, axes = _panel_grid(predictions)
    limits = [-1, 33]
    for axis, model_id in zip(axes, models):
        frame = predictions[predictions["model_id"].eq(model_id)]
        axis.scatter(frame["sensitivity_db"], frame["prediction_db"], s=8, alpha=0.25)
        axis.plot(limits, limits, "--", color="#dc2626", linewidth=1)
        axis.set(title=model_id, xlabel="Observed (dB)", ylabel="Predicted (dB)", xlim=limits, ylim=limits)
    for axis in axes[len(models):]:
        axis.set_visible(False)
    figure.tight_layout()
    return _save(figure, path)


def bland_altman(predictions: pd.DataFrame, path: Path) -> Path:
    models, figure, axes = _panel_grid(predictions)
    for axis, model_id in zip(axes, models):
        frame = predictions[predictions["model_id"].eq(model_id)]
        observed = frame["sensitivity_db"].to_numpy(float)
        predicted = frame["prediction_db"].to_numpy(float)
        average = (observed + predicted) / 2
        difference = predicted - observed
        bias, sd = difference.mean(), difference.std(ddof=1)
        axis.scatter(average, difference, s=8, alpha=0.25)
        axis.axhline(bias, color="#111827")
        axis.axhline(bias - 1.96 * sd, color="#dc2626", linestyle="--")
        axis.axhline(bias + 1.96 * sd, color="#dc2626", linestyle="--")
        axis.set(title=model_id, xlabel="Mean observed and predicted (dB)", ylabel="Predicted − observed (dB)")
    for axis in axes[len(models):]:
        axis.set_visible(False)
    figure.tight_layout()
    return _save(figure, path)


def model_comparison(metrics: pd.DataFrame, path: Path) -> Path:
    frame = metrics.sort_values("model_position")
    lower = frame["mae_db"] - frame["mae_db_ci95_low"]
    upper = frame["mae_db_ci95_high"] - frame["mae_db"]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.errorbar(frame["model_id"], frame["mae_db"], yerr=np.vstack([lower, upper]), fmt="o", capsize=4)
    axis.set(xlabel="Model", ylabel="Out-of-fold MAE (dB)")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    return _save(figure, path)


def residual_by_sensitivity(predictions: pd.DataFrame, path: Path) -> Path:
    models, figure, axes = _panel_grid(predictions)
    for axis, model_id in zip(axes, models):
        frame = predictions[predictions["model_id"].eq(model_id)]
        residual = frame["prediction_db"] - frame["sensitivity_db"]
        axis.scatter(frame["sensitivity_db"], residual, s=8, alpha=0.25)
        axis.axhline(0, color="#dc2626", linestyle="--", linewidth=1)
        axis.set(title=model_id, xlabel="Observed sensitivity (dB)", ylabel="Prediction error (dB)")
    for axis in axes[len(models):]:
        axis.set_visible(False)
    figure.tight_layout()
    return _save(figure, path)


def sensitivity_distribution(predictions: pd.DataFrame, path: Path) -> Path:
    frame = predictions.drop_duplicates(["case_id", "point_number"])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-1.5, 34.5, 1)
    axis.hist(frame["sensitivity_db"], bins=bins, color="#2563eb", edgecolor="white")
    axis.axvline(-1, color="#dc2626", linestyle="--", label="−1 dB floor")
    axis.set(xlabel="Observed pointwise MAIA sensitivity (dB)", ylabel="Number of points")
    axis.legend(frameon=False)
    figure.tight_layout()
    return _save(figure, path)


def generate(predictions: pd.DataFrame, metrics: pd.DataFrame, output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "sampling": sampling_schematic(output_root / "figure_100um_sampling.png"),
        "model_comparison": model_comparison(metrics, output_root / "figure_model_comparison.png"),
        "observed_predicted": observed_predicted(predictions, output_root / "figure_observed_predicted.png"),
        "bland_altman": bland_altman(predictions, output_root / "figure_bland_altman.png"),
        "residual_by_sensitivity": residual_by_sensitivity(predictions, output_root / "figure_residual_by_sensitivity.png"),
        "sensitivity_distribution": sensitivity_distribution(predictions, output_root / "figure_sensitivity_distribution.png"),
    }
