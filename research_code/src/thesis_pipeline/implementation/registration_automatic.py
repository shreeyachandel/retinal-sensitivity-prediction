# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Automatic MAIA-to-OCT registration with landmark-free SimpleITK fitting.

The optimizer never reads manual landmarks. When landmark CSVs are available,
they are loaded only after fitting to benchmark the automatic transform.
All optimization and reported automatic-validation coordinates use a
standardized 512 x 512 grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

import registration_landmarks as common


STANDARD_SIZE = 512
DEFAULT_OUTPUT_ROOT = common.PROJECT_ROOT / "outputs" / "automatic_registration_runs"


def standardized_image(path: Path) -> sitk.Image:
    image = Image.open(path).convert("L").resize(
        (STANDARD_SIZE, STANDARD_SIZE), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32)
    result = sitk.GetImageFromArray(array)
    result.SetSpacing((1.0, 1.0))
    return sitk.RescaleIntensity(result, 0.0, 1.0)


def vessel_response(image: sitk.Image) -> sitk.Image:
    """Emphasize dark retinal vessels while reducing broad illumination drift."""
    fine = sitk.SmoothingRecursiveGaussian(image, 0.8)
    background = sitk.SmoothingRecursiveGaussian(image, 7.0)
    response = sitk.Maximum(background - fine, 0.0)
    return sitk.RescaleIntensity(response, 0.0, 1.0)


def registration_method(iterations: int, learning_rate: float) -> sitk.ImageRegistrationMethod:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsCorrelation()
    method.SetMetricSamplingStrategy(method.NONE)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=0.0005,
        numberOfIterations=iterations,
        gradientMagnitudeTolerance=1e-7,
        relaxationFactor=0.5,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return method


def fit_similarity(fixed: sitk.Image, moving: sitk.Image):
    centre = ((STANDARD_SIZE - 1) / 2.0, (STANDARD_SIZE - 1) / 2.0)
    candidates = []
    # MAIA and OCT SLO images can cover different retinal fields even after
    # both are displayed on a 512 grid. Multi-start scale initialization is
    # therefore essential; starting only at scale 1 can converge to a visually
    # plausible but anatomically incorrect local optimum.
    for initial_scale in (0.70, 0.80, 0.90, 1.00, 1.10):
        for angle_degrees in (-6.0, 0.0, 6.0):
            transform = sitk.Similarity2DTransform()
            transform.SetCenter(centre)
            transform.SetScale(initial_scale)
            transform.SetAngle(math.radians(angle_degrees))
            method = registration_method(iterations=250, learning_rate=1.5)
            method.SetInitialTransform(transform, inPlace=True)
            try:
                method.Execute(fixed, moving)
                candidates.append(
                    (
                        float(method.GetMetricValue()),
                        transform,
                        method.GetOptimizerStopConditionDescription(),
                    )
                )
            except RuntimeError:
                continue
    if not candidates:
        raise RuntimeError("All automatic similarity initializations failed")
    return min(candidates, key=lambda item: item[0])


def fit_affine(fixed: sitk.Image, moving: sitk.Image, similarity_fixed_to_moving):
    affine = sitk.AffineTransform(2)
    affine.SetCenter(similarity_fixed_to_moving.GetCenter())
    affine.SetMatrix(similarity_fixed_to_moving.GetMatrix())
    affine.SetTranslation(similarity_fixed_to_moving.GetTranslation())
    method = registration_method(iterations=300, learning_rate=0.75)
    method.SetInitialTransform(affine, inPlace=True)
    method.Execute(fixed, moving)
    return float(method.GetMetricValue()), affine, method.GetOptimizerStopConditionDescription()


def image_correlation(fixed: sitk.Image, moving: sitk.Image, fixed_to_moving) -> tuple[float, float]:
    warped = sitk.Resample(moving, fixed, fixed_to_moving, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    valid = sitk.Resample(
        sitk.Image(moving.GetSize(), sitk.sitkUInt8) + 1,
        fixed,
        fixed_to_moving,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    fixed_array = sitk.GetArrayViewFromImage(fixed)
    warped_array = sitk.GetArrayViewFromImage(warped)
    mask = sitk.GetArrayViewFromImage(valid).astype(bool)
    overlap_fraction = float(mask.mean())
    if mask.sum() < 100:
        return float("nan"), overlap_fraction
    x = fixed_array[mask].astype(np.float64)
    y = warped_array[mask].astype(np.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan"), overlap_fraction
    return float(np.corrcoef(x, y)[0, 1]), overlap_fraction


def affine_plausibility(transform) -> dict[str, object]:
    matrix = np.array(transform.GetMatrix(), dtype=float).reshape(2, 2)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    determinant = float(np.linalg.det(matrix))
    condition_number = float(singular_values.max() / max(singular_values.min(), 1e-12))
    plausible = bool(
        determinant > 0
        and 0.70 <= singular_values.min()
        and singular_values.max() <= 1.35
        and condition_number <= 1.30
    )
    return {
        "determinant": determinant,
        "singular_values": singular_values.tolist(),
        "condition_number": condition_number,
        "plausible": plausible,
    }


def save_outputs(
    output_dir: Path,
    kind: str,
    fixed_display: sitk.Image,
    moving_display: sitk.Image,
    fixed_to_moving,
) -> None:
    moving_to_fixed = fixed_to_moving.GetInverse()
    sitk.WriteTransform(fixed_to_moving, str(output_dir / f"{kind}_fixed_to_moving_512.tfm"))
    sitk.WriteTransform(moving_to_fixed, str(output_dir / f"{kind}_moving_to_fixed_512.tfm"))
    warped = sitk.Resample(
        moving_display,
        fixed_display,
        fixed_to_moving,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    fixed_u8 = sitk.Cast(sitk.RescaleIntensity(fixed_display, 0, 255), sitk.sitkUInt8)
    warped_u8 = sitk.Cast(sitk.RescaleIntensity(warped, 0, 255), sitk.sitkUInt8)
    sitk.WriteImage(warped_u8, str(output_dir / f"warped_maia_{kind}_512.png"))
    sitk.WriteImage(
        sitk.Compose(warped_u8, fixed_u8, fixed_u8),
        str(output_dir / f"overlay_{kind}_512.png"),
    )


def read_landmarks(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["landmark_id"] for row in rows]
    moving = np.array([[float(row["moving_x_px"]), float(row["moving_y_px"])] for row in rows])
    fixed = np.array([[float(row["fixed_x_px"]), float(row["fixed_y_px"])] for row in rows])
    return labels, moving, fixed


def landmark_benchmark(
    transform_moving_to_fixed,
    landmark_path: Path,
    moving_native_size: tuple[int, int],
    fixed_native_size: tuple[int, int],
) -> dict[str, object] | None:
    if not landmark_path.exists():
        return None
    labels, moving, fixed = read_landmarks(landmark_path)
    moving_scale = np.array([STANDARD_SIZE / moving_native_size[0], STANDARD_SIZE / moving_native_size[1]])
    fixed_scale = np.array([STANDARD_SIZE / fixed_native_size[0], STANDARD_SIZE / fixed_native_size[1]])
    moving_512 = moving * moving_scale
    fixed_512 = fixed * fixed_scale
    predicted = np.array([transform_moving_to_fixed.TransformPoint(tuple(point)) for point in moving_512])
    errors = np.linalg.norm(predicted - fixed_512, axis=1)
    return {
        "count": len(labels),
        "mean_px_512": float(errors.mean()),
        "median_px_512": float(np.median(errors)),
        "max_px_512": float(errors.max()),
        "by_landmark": [
            {"landmark_id": label, "error_px_512": float(error)}
            for label, error in zip(labels, errors)
        ],
    }


def run_registration(
    case_id: str,
    moving_path: Path,
    fixed_path: Path,
    output_dir: Path,
    manual_dir: Path | None = None,
    case_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run one landmark-free registration and return its saved result."""
    output_dir.mkdir(parents=True, exist_ok=True)

    moving_native_size = Image.open(moving_path).size
    fixed_native_size = Image.open(fixed_path).size
    moving_display = standardized_image(moving_path)
    fixed_display = standardized_image(fixed_path)
    moving_registration = vessel_response(moving_display)
    fixed_registration = vessel_response(fixed_display)

    similarity_metric, similarity, similarity_stop = fit_similarity(
        fixed_registration, moving_registration
    )
    affine_metric, affine, affine_stop = fit_affine(
        fixed_registration, moving_registration, similarity
    )

    landmark_dir = manual_dir if manual_dir is not None else Path("/__no_manual_landmarks__")
    result: dict[str, object] = {
        "case_id": case_id,
        "tool": "SimpleITK",
        "mode": "automatic_landmark_free",
        "standardized_grid_px": [STANDARD_SIZE, STANDARD_SIZE],
        "moving_native_size_px": list(moving_native_size),
        "fixed_native_size_px": list(fixed_native_size),
        "manual_landmarks_used_for_optimization": False,
        "transforms": {},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if case_metadata:
        result["case_metadata"] = case_metadata

    for kind, metric, transform, stop_condition in (
        ("similarity", similarity_metric, similarity, similarity_stop),
        ("affine", affine_metric, affine, affine_stop),
    ):
        moving_to_fixed = transform.GetInverse()
        correlation, overlap_fraction = image_correlation(
            fixed_registration, moving_registration, transform
        )
        save_outputs(output_dir, kind, fixed_display, moving_display, transform)
        transform_result: dict[str, object] = {
            "optimizer_metric": metric,
            "vessel_correlation": correlation,
            "overlap_fraction": overlap_fraction,
            "optimizer_stop_condition": stop_condition,
            "parameters_fixed_to_moving_512": list(transform.GetParameters()),
            "fit_landmark_benchmark": landmark_benchmark(
                moving_to_fixed,
                landmark_dir / "landmarks_fit.csv",
                moving_native_size,
                fixed_native_size,
            ),
            "qc_landmark_benchmark": landmark_benchmark(
                moving_to_fixed,
                landmark_dir / "landmarks_qc.csv",
                moving_native_size,
                fixed_native_size,
            ),
        }
        if kind == "similarity":
            transform_result["scale"] = float(transform.GetScale())
            transform_result["angle_degrees"] = float(math.degrees(transform.GetAngle()))
        else:
            transform_result["plausibility"] = affine_plausibility(transform)
        result["transforms"][kind] = transform_result

    similarity_score = result["transforms"]["similarity"]["vessel_correlation"]
    affine_score = result["transforms"]["affine"]["vessel_correlation"]
    affine_is_plausible = result["transforms"]["affine"]["plausibility"]["plausible"]
    if (
        affine_is_plausible
        and np.isfinite(affine_score)
        and affine_score >= similarity_score + 0.01
    ):
        recommendation = "affine"
    else:
        recommendation = "similarity"
    result["provisional_recommendation"] = recommendation
    result["acceptance_status"] = "pending_manual_overlay_review"

    (output_dir / "automatic_registration_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manual-run-root", type=Path, default=common.DEFAULT_RUN_ROOT)
    args = parser.parse_args()

    row = common.read_pilot(args.pilot_id)
    paths = common.case_paths(row)
    common.ensure_case_inputs(row, paths)
    output_dir = args.output_root.resolve() / args.pilot_id
    result = run_registration(
        case_id=args.pilot_id,
        moving_path=paths["moving"],
        fixed_path=paths["fixed"],
        output_dir=output_dir,
        manual_dir=args.manual_run_root.resolve() / args.pilot_id,
        case_metadata={
            "study_id": row["study_id"],
            "eye": row["eye"],
            "maia_timepoint": row["provisional_maia_timepoint"],
            "oct_visit": row["oct_visit"],
        },
    )
    print(f"Automatic registration outputs written to {output_dir}")
    print(f"Provisional transform recommendation: {result['provisional_recommendation']}")
    print("No automatic result accepted without overlay review.")


if __name__ == "__main__":
    main()
