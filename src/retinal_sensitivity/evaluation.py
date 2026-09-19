"""Evaluation metrics and figures for out-of-fold predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metric_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise error, agreement and explained variance by model."""

    rows: list[dict[str, float | str]] = []
    for model_name, model_rows in predictions.groupby("model", sort=False):
        observed = model_rows["observed_db"].to_numpy()
        predicted = model_rows["predicted_db"].to_numpy()
        difference = predicted - observed
        bias = float(np.mean(difference))
        difference_sd = float(np.std(difference, ddof=1))
        rows.append(
            {
                "model": model_name,
                "mae_db": mean_absolute_error(observed, predicted),
                "rmse_db": np.sqrt(mean_squared_error(observed, predicted)),
                "bias_db": bias,
                "lower_loa_db": bias - 1.96 * difference_sd,
                "upper_loa_db": bias + 1.96 * difference_sd,
                "r_squared": r2_score(observed, predicted),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_db", ignore_index=True)


def save_demo_figure(predictions: pd.DataFrame, output_path: Path) -> None:
    """Save model comparison, observed/predicted and agreement panels."""

    metrics = metric_summary(predictions)
    best_model = str(metrics.iloc[0]["model"])
    best_rows = predictions[predictions["model"] == best_model]
    observed = best_rows["observed_db"].to_numpy()
    predicted = best_rows["predicted_db"].to_numpy()
    mean_values = (observed + predicted) / 2
    differences = predicted - observed
    bias = differences.mean()
    loa_offset = 1.96 * differences.std(ddof=1)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    axes[0].barh(metrics["model"], metrics["mae_db"], color=["#3465a4", "#6aaed6", "#9ecae1"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE (dB; lower is better)")
    axes[0].set_title("Participant-held-out performance")

    axes[1].hexbin(observed, predicted, gridsize=28, mincnt=1, cmap="Blues")
    limit_low = min(observed.min(), predicted.min())
    limit_high = max(observed.max(), predicted.max())
    axes[1].plot([limit_low, limit_high], [limit_low, limit_high], "--", color="#444444")
    axes[1].set_xlabel("Observed sensitivity (dB)")
    axes[1].set_ylabel("Predicted sensitivity (dB)")
    axes[1].set_title(best_model)

    axes[2].scatter(mean_values, differences, s=8, alpha=0.22, color="#3465a4", edgecolors="none")
    axes[2].axhline(bias, color="#222222", linewidth=1.5, label="Bias")
    axes[2].axhline(bias - loa_offset, color="#d95f02", linestyle="--", label="95% limits")
    axes[2].axhline(bias + loa_offset, color="#d95f02", linestyle="--")
    axes[2].set_xlabel("Mean of observed and predicted (dB)")
    axes[2].set_ylabel("Predicted − observed (dB)")
    axes[2].set_title("Bland–Altman agreement")
    axes[2].legend(frameon=False, fontsize=8)

    figure.suptitle("Synthetic demonstration — not thesis data", fontweight="bold")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

