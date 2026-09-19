"""Generate synthetic registration schematics for the public dissertation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NAVY = "#12233F"
BLUE = "#2563EB"
CYAN = "#0EA5A8"
CORAL = "#E76F51"
GREY = "#CBD5E1"


def _style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, loc="left", color=NAVY, fontsize=12, fontweight="bold")
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(GREY)


def _synthetic_landmarks(angle: float = 0.0, scale: float = 1.0) -> np.ndarray:
    radii = np.array([0.16, 0.32, 0.50, 0.68])
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False) + angle
    points = np.vstack(
        [np.column_stack((radius * np.cos(theta), radius * np.sin(theta))) for radius in radii]
    )
    return points * scale


def _draw_abstract_vessels(axis: plt.Axes, phase: float = 0.0) -> None:
    x = np.linspace(-1.0, 1.0, 300)
    for offset, width in [(-0.45, 2.0), (-0.12, 1.5), (0.24, 1.8), (0.55, 1.2)]:
        y = offset + 0.10 * np.sin((x + phase) * np.pi * width)
        axis.plot(x, y, color="#94A3B8", linewidth=1.2, alpha=0.8)


def registration_schematic(output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)
    points = _synthetic_landmarks()

    _draw_abstract_vessels(axes[0])
    axes[0].scatter(points[:, 0], points[:, 1], s=22, color=BLUE, edgecolor="white", linewidth=0.5)
    axes[0].scatter([0], [0], marker="x", s=90, color=CORAL, linewidth=2.4)
    _style_axis(axes[0], "Synthetic functional test grid")

    transformed = points @ np.array([[0.96, -0.08], [0.08, 0.96]]) + np.array([0.07, -0.04])
    _draw_abstract_vessels(axes[1], phase=0.18)
    for y in np.linspace(-0.82, 0.82, 18):
        axes[1].hlines(y, -0.82, 0.82, color=CYAN, linewidth=0.65, alpha=0.45)
    axes[1].scatter(
        transformed[:, 0], transformed[:, 1], s=22, color=BLUE, edgecolor="white", linewidth=0.5
    )
    axes[1].scatter([0.07], [-0.04], marker="x", s=90, color=CORAL, linewidth=2.4)
    _style_axis(axes[1], "Registered grid and candidate B-scans")

    figure.suptitle(
        "REGISTRATION AND POINT-ALIGNMENT SCHEMATIC - SYNTHETIC COORDINATES",
        color=NAVY,
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def supplementary_schematic(output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.6), constrained_layout=True)
    configurations = [(-0.10, 0.94), (0.05, 1.02), (0.13, 0.91), (-0.04, 0.98)]

    for index, (axis, (angle, scale)) in enumerate(zip(axes.flat, configurations), start=1):
        _draw_abstract_vessels(axis, phase=index * 0.11)
        points = _synthetic_landmarks(angle=angle, scale=scale)
        axis.scatter(points[:, 0], points[:, 1], s=16, color=BLUE, edgecolor="white", linewidth=0.4)
        axis.scatter([0], [0], marker="x", s=65, color=CORAL, linewidth=2.0)
        _style_axis(axis, f"Synthetic QC example {index}")

    figure.suptitle(
        "ILLUSTRATIVE REGISTRATION VARIATION - SYNTHETIC EXAMPLES",
        color=NAVY,
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    output = Path("docs/thesis-public-source/figures")
    output.mkdir(parents=True, exist_ok=True)
    registration_schematic(output / "figure_registration_qc_public.png")
    supplementary_schematic(output / "figure_supplementary_qc_public.png")


if __name__ == "__main__":
    main()
