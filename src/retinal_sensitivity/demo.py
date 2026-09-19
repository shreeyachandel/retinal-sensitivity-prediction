"""Command-line entry point for the public synthetic demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import DemoConfig, generate_synthetic_cohort
from .evaluation import metric_summary, save_demo_figure
from .modeling import grouped_predictions


def run_demo(output_dir: Path, random_state: int = 42) -> tuple[Path, Path]:
    """Run the synthetic workflow and write aggregate demo artefacts."""

    data = generate_synthetic_cohort(DemoConfig(random_state=random_state))
    predictions = grouped_predictions(data)
    metrics = metric_summary(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "synthetic_metrics.csv"
    figure_path = output_dir / "synthetic_model_evaluation.png"
    metrics.to_csv(metrics_path, index=False, float_format="%.4f")
    save_demo_figure(predictions, figure_path)
    return metrics_path, figure_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the privacy-safe synthetic retinal sensitivity demo."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for generated metrics and figure (default: artifacts).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Synthetic random seed.")
    args = parser.parse_args()

    metrics_path, figure_path = run_demo(args.output_dir, args.seed)
    print(f"Metrics: {metrics_path}")
    print(f"Figure:  {figure_path}")


if __name__ == "__main__":
    main()

