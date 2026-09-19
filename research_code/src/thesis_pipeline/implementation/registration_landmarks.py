# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Prepare, collect, and fit landmark-based MAIA-to-OCT registrations.

The script uses only the privacy-minimised working copy. It preserves native
pixel coordinates and writes one reproducible output directory per pilot run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
WORKING_ROOT = PROJECT_ROOT / "outputs" / "2026-07-14_deidentified_working_data"
PILOT_CSV = WORKING_ROOT / "working_files" / "pilot_cases.csv"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "registration_runs"
DEFAULT_ZOOM_RADIUS_PX = 60


def read_pilot(pilot_id: str) -> dict[str, str]:
    with PILOT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["pilot_id"] == pilot_id:
                return row
    raise SystemExit(f"Unknown pilot_id {pilot_id!r}; see {PILOT_CSV}")


def case_paths(row: dict[str, str]) -> dict[str, Path]:
    oct_root = WORKING_ROOT / row["oct_path"]
    return {
        "moving": WORKING_ROOT / row["maia_clean_path"],
        "point_reference": WORKING_ROOT / row["maia_overlay_path"],
        "fixed": oct_root / "slo.png",
        "scan_area": oct_root / "slo_area.csv",
        "scan_lines": oct_root / "slo_coordinates.csv",
    }


def ensure_case_inputs(row: dict[str, str], paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing case inputs:\n" + "\n".join(missing))
    if int(row["n_bscans"]) != int(row["n_coordinate_rows"]):
        raise SystemExit(
            "Registration blocked: B-scan and coordinate-row counts disagree "
            f"({row['n_bscans']} vs {row['n_coordinate_rows']})."
        )


def relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def prepare_case(pilot_id: str, run_root: Path) -> Path:
    row = read_pilot(pilot_id)
    paths = case_paths(row)
    ensure_case_inputs(row, paths)
    run_dir = run_root / pilot_id
    run_dir.mkdir(parents=True, exist_ok=True)

    moving_size = Image.open(paths["moving"]).size
    fixed_size = Image.open(paths["fixed"]).size
    metadata = {
        "pilot_id": pilot_id,
        "study_id": row["study_id"],
        "eye": row["eye"],
        "maia_timepoint": row["provisional_maia_timepoint"],
        "oct_visit": row["oct_visit"],
        "visit_mapping_status": "provisional_unverified",
        "moving_image": relative_to_project(paths["moving"]),
        "fixed_image": relative_to_project(paths["fixed"]),
        "point_reference_image": relative_to_project(paths["point_reference"]),
        "scan_area_csv": relative_to_project(paths["scan_area"]),
        "scan_lines_csv": relative_to_project(paths["scan_lines"]),
        "moving_native_size_px": list(moving_size),
        "fixed_native_size_px": list(fixed_size),
        "n_bscans": int(row["n_bscans"]),
        "n_coordinate_rows": int(row["n_coordinate_rows"]),
        "expected_difficulty": row["expected_difficulty"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run_dir


def click_in_axis(plt, figure, axis, prompt: str) -> tuple[float, float]:
    """Wait for a left click inside the requested axis only."""
    clicked: list[tuple[float, float]] = []
    cancelled: list[bool] = []

    def on_click(event):
        if event.button == 1 and event.inaxes is axis and event.xdata is not None and event.ydata is not None:
            clicked.append((float(event.xdata), float(event.ydata)))

    def on_key(event):
        if event.key in {"escape", "q"}:
            cancelled.append(True)

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.suptitle(prompt)
    plt.draw()
    while not clicked and not cancelled:
        plt.pause(0.05)
    if cancelled:
        raise RuntimeError("Landmark selection cancelled")
    return clicked[0]


def click_precisely(
    plt,
    figure,
    axis,
    prompt: str,
    zoom_radius_px: int = DEFAULT_ZOOM_RADIUS_PX,
) -> tuple[float, float]:
    """Collect a coarse click followed by an exact click in a magnified view."""
    coarse_x, coarse_y = click_in_axis(
        plt,
        figure,
        axis,
        f"{prompt} — first click NEAR the junction to magnify it",
    )
    original_xlim = axis.get_xlim()
    original_ylim = axis.get_ylim()
    image_height, image_width = axis.images[0].get_array().shape[:2]
    x_min = max(-0.5, coarse_x - zoom_radius_px)
    x_max = min(image_width - 0.5, coarse_x + zoom_radius_px)
    y_min = max(-0.5, coarse_y - zoom_radius_px)
    y_max = min(image_height - 0.5, coarse_y + zoom_radius_px)
    coarse_marker = axis.plot(
        coarse_x,
        coarse_y,
        "+",
        color="magenta",
        markersize=14,
        markeredgewidth=2,
    )[0]
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_max, y_min)
    figure.canvas.draw_idle()
    try:
        precise_x, precise_y = click_in_axis(
            plt,
            figure,
            axis,
            f"{prompt} — MAGNIFIED: click the exact centre of the junction",
        )
    finally:
        coarse_marker.remove()
        axis.set_xlim(original_xlim)
        axis.set_ylim(original_ylim)
        figure.canvas.draw_idle()
    return precise_x, precise_y


def collect_points(plt, figure, ax_moving, ax_fixed, count: int, kind: str) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for index in range(1, count + 1):
        plt.suptitle(
            f"{kind.upper()} pair {index}/{count} — Step 1: click the centre of one vessel junction on MAIA (LEFT)"
        )
        plt.draw()
        mx, my = click_precisely(
            plt,
            figure,
            ax_moving,
            f"{kind.upper()} pair {index}/{count} — click the vessel junction in the LEFT MAIA panel",
        )
        ax_moving.plot(mx, my, "+", color="yellow", markersize=14, markeredgewidth=2)
        plt.suptitle(
            f"{kind.upper()} pair {index}/{count} — Step 2: find that EXACT SAME junction on OCT SLO (RIGHT)"
        )
        plt.draw()
        fx, fy = click_precisely(
            plt,
            figure,
            ax_fixed,
            f"{kind.upper()} pair {index}/{count} — click the SAME junction in the RIGHT OCT/SLO panel",
        )
        label = f"{kind}_{index:02d}"
        ax_moving.plot(mx, my, "o", color="yellow" if kind == "fit" else "cyan")
        ax_fixed.plot(fx, fy, "o", color="yellow" if kind == "fit" else "cyan")
        ax_moving.text(mx + 5, my + 5, label, color="yellow", fontsize=8)
        ax_fixed.text(fx + 5, fy + 5, label, color="yellow", fontsize=8)
        points.append(
            {
                "landmark_id": label,
                "moving_x_px": round(mx, 3),
                "moving_y_px": round(my, 3),
                "fixed_x_px": round(fx, 3),
                "fixed_y_px": round(fy, 3),
                "landmark_type": "vessel_crossing_or_bifurcation",
                "uncertainty_note": "",
            }
        )
    return points


def write_landmarks(path: Path, points: list[dict[str, object]]) -> None:
    fields = [
        "landmark_id",
        "moving_x_px",
        "moving_y_px",
        "fixed_x_px",
        "fixed_y_px",
        "landmark_type",
        "uncertainty_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)


def replace_landmarks(pilot_id: str, run_root: Path, landmark_ids: list[str]) -> None:
    """Interactively replace named fitting or QC landmarks while retaining all others."""
    run_dir = prepare_case(pilot_id, run_root)
    landmark_paths = {
        "fit": run_dir / "landmarks_fit.csv",
        "qc": run_dir / "landmarks_qc.csv",
    }
    rows_by_kind: dict[str, list[dict[str, str]]] = {}
    for kind, landmark_path in landmark_paths.items():
        if not landmark_path.exists():
            raise SystemExit(f"No {kind} landmark file exists at {landmark_path}")
        with landmark_path.open(newline="", encoding="utf-8") as handle:
            rows_by_kind[kind] = list(csv.DictReader(handle))
    available = {
        row["landmark_id"]
        for rows in rows_by_kind.values()
        for row in rows
    }
    unknown = set(landmark_ids) - available
    if unknown:
        raise SystemExit(f"Unknown fitting landmark IDs: {', '.join(sorted(unknown))}")

    import matplotlib.pyplot as plt

    row = read_pilot(pilot_id)
    paths = case_paths(row)
    moving = Image.open(paths["moving"]).convert("L")
    fixed = Image.open(paths["fixed"]).convert("L")
    figure, (ax_moving, ax_fixed) = plt.subplots(1, 2, figsize=(15, 8))
    ax_moving.imshow(moving, cmap="gray", origin="upper")
    ax_fixed.imshow(fixed, cmap="gray", origin="upper")
    ax_moving.set_title("Moving: MAIA clean (native pixels)")
    ax_fixed.set_title("Fixed: OCT SLO (native pixels)")
    for axis in (ax_moving, ax_fixed):
        axis.set_xlabel("x (px)")
        axis.set_ylabel("y (px)")
    for rows in rows_by_kind.values():
        for existing in rows:
            colour = "orange" if existing["landmark_id"] in landmark_ids else "yellow"
            ax_moving.plot(float(existing["moving_x_px"]), float(existing["moving_y_px"]), "+", color=colour, markersize=12, markeredgewidth=2)
            ax_fixed.plot(float(existing["fixed_x_px"]), float(existing["fixed_y_px"]), "+", color=colour, markersize=12, markeredgewidth=2)
    figure.tight_layout()
    replacements = {}
    for landmark_id in landmark_ids:
        mx, my = click_precisely(
            plt, figure, ax_moving,
            f"Replace {landmark_id}: click a distinctive junction in the LEFT MAIA panel",
        )
        ax_moving.plot(mx, my, "+", color="lime", markersize=15, markeredgewidth=2)
        fx, fy = click_precisely(
            plt, figure, ax_fixed,
            f"Replace {landmark_id}: click the SAME junction in the RIGHT OCT/SLO panel",
        )
        ax_fixed.plot(fx, fy, "+", color="lime", markersize=15, markeredgewidth=2)
        replacements[landmark_id] = (mx, my, fx, fy)
    for kind, rows in rows_by_kind.items():
        for existing in rows:
            if existing["landmark_id"] in replacements:
                mx, my, fx, fy = replacements[existing["landmark_id"]]
                existing["moving_x_px"] = round(mx, 3)
                existing["moving_y_px"] = round(my, 3)
                existing["fixed_x_px"] = round(fx, 3)
                existing["fixed_y_px"] = round(fy, 3)
                existing["uncertainty_note"] = "reselected with magnified view after residual review"
        write_landmarks(landmark_paths[kind], rows)
    figure.savefig(run_dir / "selected_landmarks_revised.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Replaced {', '.join(landmark_ids)} in {run_dir}")


def select_landmarks(pilot_id: str, run_root: Path, fit_count: int, qc_count: int) -> None:
    import matplotlib.pyplot as plt

    if fit_count < 3:
        raise SystemExit("At least 3 fitting landmarks are required")
    if qc_count < 2:
        raise SystemExit("At least 2 held-out QC landmarks are required")
    run_dir = prepare_case(pilot_id, run_root)
    row = read_pilot(pilot_id)
    paths = case_paths(row)
    moving = Image.open(paths["moving"]).convert("L")
    fixed = Image.open(paths["fixed"]).convert("L")

    figure, (ax_moving, ax_fixed) = plt.subplots(1, 2, figsize=(15, 8))
    ax_moving.imshow(moving, cmap="gray", origin="upper")
    ax_fixed.imshow(fixed, cmap="gray", origin="upper")
    ax_moving.set_title("Moving: MAIA clean (native pixels)")
    ax_fixed.set_title("Fixed: OCT SLO (native pixels)")
    for axis in (ax_moving, ax_fixed):
        axis.set_axis_on()
        axis.set_xlabel("x (px)")
        axis.set_ylabel("y (px)")
    figure.tight_layout()

    plt.suptitle("Read this before clicking", fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        0.015,
        "For each panel, first click NEAR the chosen Y- or T-shaped vessel junction. The panel will magnify "
        "that area; then click its EXACT centre. Do this on the LEFT MAIA image, then on the SAME anatomical "
        "junction in the RIGHT OCT image. The pixel positions will differ. Press any key to begin.",
        ha="center",
        va="bottom",
        wrap=True,
        fontsize=10,
    )
    plt.draw()
    plt.waitforbuttonpress(timeout=-1)

    fit = collect_points(plt, figure, ax_moving, ax_fixed, fit_count, "fit")
    qc = collect_points(plt, figure, ax_moving, ax_fixed, qc_count, "qc")
    write_landmarks(run_dir / "landmarks_fit.csv", fit)
    write_landmarks(run_dir / "landmarks_qc.csv", qc)
    figure.savefig(run_dir / "selected_landmarks.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Landmarks saved to {run_dir}. Fit them with the SimpleITK runner next.")


def read_landmarks(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No landmarks found in {path}")
    labels = [row["landmark_id"] for row in rows]
    moving = np.array(
        [[float(row["moving_x_px"]), float(row["moving_y_px"])] for row in rows]
    )
    fixed = np.array(
        [[float(row["fixed_x_px"]), float(row["fixed_y_px"])] for row in rows]
    )
    return labels, moving, fixed


def fit_similarity(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moving_mean = moving.mean(axis=0)
    fixed_mean = fixed.mean(axis=0)
    x = moving - moving_mean
    y = fixed - fixed_mean
    covariance = x.T @ y
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
        singular_values[-1] *= -1
    denominator = np.sum(x * x)
    if denominator <= 0:
        raise SystemExit("Degenerate moving landmarks")
    scale = float(np.sum(singular_values) / denominator)
    matrix = scale * rotation
    translation = fixed_mean - matrix @ moving_mean
    return matrix, translation


def fit_affine(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([moving, np.ones(len(moving))])
    coefficients, _, rank, _ = np.linalg.lstsq(design, fixed, rcond=None)
    if rank < 3:
        raise SystemExit("Degenerate fitting landmarks for affine transform")
    matrix = coefficients[:2, :].T
    translation = coefficients[2, :]
    return matrix, translation


def apply_transform(points: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (matrix @ points.T).T + translation


def tre_rows(
    labels: list[str], moving: np.ndarray, fixed: np.ndarray, matrix: np.ndarray, translation: np.ndarray
) -> list[dict[str, object]]:
    predicted = apply_transform(moving, matrix, translation)
    errors = np.linalg.norm(predicted - fixed, axis=1)
    return [
        {
            "landmark_id": label,
            "predicted_fixed_x_px": round(float(pred[0]), 3),
            "predicted_fixed_y_px": round(float(pred[1]), 3),
            "observed_fixed_x_px": round(float(obs[0]), 3),
            "observed_fixed_y_px": round(float(obs[1]), 3),
            "tre_px": round(float(error), 3),
        }
        for label, pred, obs, error in zip(labels, predicted, fixed, errors)
    ]


def write_tre(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def warp_to_fixed(moving_path: Path, fixed_size: tuple[int, int], matrix, translation) -> Image.Image:
    inverse = np.linalg.inv(matrix)
    offset = -inverse @ translation
    coefficients = (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(offset[0]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(offset[1]),
    )
    moving = Image.open(moving_path).convert("L")
    return moving.transform(
        fixed_size,
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )


def make_overlay(fixed_path: Path, warped: Image.Image) -> Image.Image:
    fixed = np.asarray(Image.open(fixed_path).convert("L"), dtype=np.float32)
    moving = np.asarray(warped, dtype=np.float32)
    fixed = 255 * (fixed - fixed.min()) / max(float(np.ptp(fixed)), 1.0)
    moving = 255 * (moving - moving.min()) / max(float(np.ptp(moving)), 1.0)
    rgb = np.zeros((*fixed.shape, 3), dtype=np.uint8)
    rgb[..., 0] = moving.astype(np.uint8)
    rgb[..., 1] = fixed.astype(np.uint8)
    rgb[..., 2] = fixed.astype(np.uint8)
    return Image.fromarray(rgb)


def annotate_qc(overlay: Image.Image, rows: list[dict[str, object]]) -> Image.Image:
    result = overlay.copy()
    draw = ImageDraw.Draw(result)
    for row in rows:
        px = float(row["predicted_fixed_x_px"])
        py = float(row["predicted_fixed_y_px"])
        ox = float(row["observed_fixed_x_px"])
        oy = float(row["observed_fixed_y_px"])
        draw.line((px, py, ox, oy), fill=(255, 255, 0), width=3)
        radius = 6
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=(255, 0, 255), width=3)
        draw.ellipse((ox - radius, oy - radius, ox + radius, oy + radius), outline=(0, 255, 0), width=3)
    return result


def transform_summary(matrix: np.ndarray, translation: np.ndarray) -> dict[str, object]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return {
        "matrix_2x2": matrix.tolist(),
        "translation_xy": translation.tolist(),
        "determinant": float(np.linalg.det(matrix)),
        "axis_scales": singular_values.tolist(),
        "condition_number": float(np.linalg.cond(matrix)),
    }


def fit_case(pilot_id: str, run_root: Path) -> None:
    run_dir = prepare_case(pilot_id, run_root)
    row = read_pilot(pilot_id)
    paths = case_paths(row)
    fit_labels, fit_moving, fit_fixed = read_landmarks(run_dir / "landmarks_fit.csv")
    qc_labels, qc_moving, qc_fixed = read_landmarks(run_dir / "landmarks_qc.csv")
    fixed_size = Image.open(paths["fixed"]).size

    results: dict[str, object] = {
        "pilot_id": pilot_id,
        "visit_mapping_status": "provisional_unverified",
        "fit_landmark_count": len(fit_labels),
        "held_out_qc_landmark_count": len(qc_labels),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "transforms": {},
        "accepted_transform": None,
        "acceptance_status": "pending_visual_review_and_supervisor_threshold",
    }

    for name, fitter in (("similarity", fit_similarity), ("affine", fit_affine)):
        matrix, translation = fitter(fit_moving, fit_fixed)
        fit_tre = tre_rows(fit_labels, fit_moving, fit_fixed, matrix, translation)
        qc_tre = tre_rows(qc_labels, qc_moving, qc_fixed, matrix, translation)
        write_tre(run_dir / f"tre_fit_{name}.csv", fit_tre)
        write_tre(run_dir / f"tre_qc_{name}.csv", qc_tre)
        warped = warp_to_fixed(paths["moving"], fixed_size, matrix, translation)
        warped.save(run_dir / f"warped_maia_{name}.png")
        overlay = make_overlay(paths["fixed"], warped)
        annotate_qc(overlay, qc_tre).save(run_dir / f"overlay_{name}_qc.png")
        fit_errors = [float(item["tre_px"]) for item in fit_tre]
        qc_errors = [float(item["tre_px"]) for item in qc_tre]
        results["transforms"][name] = {
            **transform_summary(matrix, translation),
            "fit_tre_mean_px": float(np.mean(fit_errors)),
            "fit_tre_median_px": float(np.median(fit_errors)),
            "qc_tre_mean_px": float(np.mean(qc_errors)),
            "qc_tre_median_px": float(np.median(qc_errors)),
            "qc_tre_max_px": float(np.max(qc_errors)),
        }

    similarity = results["transforms"]["similarity"]
    affine = results["transforms"]["affine"]
    if (
        similarity["fit_tre_median_px"] > 50
        or affine["determinant"] <= 0
    ):
        results["acceptance_status"] = "rejected_landmark_correspondence_sanity_check"
        results["rejection_note"] = (
            "The selected pairs are geometrically inconsistent or imply reflection. "
            "Repeat landmark selection; this is an entry sanity check, not the final clinical TRE threshold."
        )

    (run_dir / "registration_result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Registration outputs written to {run_dir}")
    print(f"Status: {results['acceptance_status']}")
    print("No transform has been accepted automatically; review both QC overlays and held-out TRE.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=f"Output root (default: {DEFAULT_RUN_ROOT})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Validate inputs and create case.json")
    prepare.add_argument("--pilot-id", required=True)
    select = subparsers.add_parser("select", help="Interactively select landmarks, then fit")
    select.add_argument("--pilot-id", required=True)
    select.add_argument("--fit-count", type=int, default=6)
    select.add_argument("--qc-count", type=int, default=2)
    fit = subparsers.add_parser("fit", help="Fit transforms from existing landmark CSV files")
    fit.add_argument("--pilot-id", required=True)
    replace = subparsers.add_parser("replace", help="Replace named fitting or QC landmark pairs")
    replace.add_argument("--pilot-id", required=True)
    replace.add_argument("--ids", required=True, help="Comma-separated landmark IDs, e.g. fit_01,qc_01")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    if args.command == "prepare":
        print(prepare_case(args.pilot_id, run_root))
    elif args.command == "select":
        select_landmarks(args.pilot_id, run_root, args.fit_count, args.qc_count)
    elif args.command == "fit":
        fit_case(args.pilot_id, run_root)
    elif args.command == "replace":
        replace_landmarks(args.pilot_id, run_root, [item.strip() for item in args.ids.split(",") if item.strip()])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Cancelled; no partial landmark set was saved.")
