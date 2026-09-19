# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Fit and QC landmark registrations using SimpleITK.

Landmark CSVs are produced by registration_landmarks.py. The saved transform
maps MAIA native pixel coordinates to OCT/SLO native pixel coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import registration_landmarks as common


def read_landmarks(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No landmarks found in {path}")
    moving = np.array([[float(r["moving_x_px"]), float(r["moving_y_px"])] for r in rows])
    fixed = np.array([[float(r["fixed_x_px"]), float(r["fixed_y_px"])] for r in rows])
    return rows, moving, fixed


def initialise(kind: str, moving: np.ndarray, fixed: np.ndarray):
    if kind == "similarity":
        # Estimate the constrained translation/rotation/uniform-scale model
        # from landmarks, then use SimpleITK for the stored transform and
        # resampling. This prevents affine shear or anisotropic scaling.
        matrix, translation = common.fit_similarity(moving, fixed)
        transform = sitk.AffineTransform(2)
        transform.SetMatrix(matrix.ravel().tolist())
        transform.SetTranslation(translation.tolist())
        return transform
    # SimpleITK's landmark initializer maps fixedLandmarks -> movingLandmarks.
    # Pass MAIA as fixedLandmarks and OCT as movingLandmarks so the saved
    # transform maps MAIA -> OCT, which is the direction required for point
    # transfer. Affine is the first established-tool transform to validate.
    transform = sitk.AffineTransform(2)
    return sitk.LandmarkBasedTransformInitializer(
        transform,
        fixedLandmarks=moving.ravel().tolist(),
        movingLandmarks=fixed.ravel().tolist(),
    )


def tre(transform, moving: np.ndarray, fixed: np.ndarray):
    predicted = np.array([transform.TransformPoint(tuple(p)) for p in moving])
    errors = np.linalg.norm(predicted - fixed, axis=1)
    return predicted, errors


def errors_on_512_grid(predicted: np.ndarray, fixed: np.ndarray, fixed_size: tuple[int, int]):
    """Express coordinate residuals on a standardized 512 x 512 output grid."""
    scale = np.array([512.0 / fixed_size[0], 512.0 / fixed_size[1]])
    return np.linalg.norm((predicted - fixed) * scale, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--run-root", type=Path, default=common.DEFAULT_RUN_ROOT)
    args = parser.parse_args()

    run_dir = args.run_root.resolve() / args.pilot_id
    row = common.read_pilot(args.pilot_id)
    paths = common.case_paths(row)
    common.ensure_case_inputs(row, paths)
    fit_rows, fit_moving, fit_fixed = read_landmarks(run_dir / "landmarks_fit.csv")
    qc_rows, qc_moving, qc_fixed = read_landmarks(run_dir / "landmarks_qc.csv")
    moving_image = sitk.ReadImage(str(paths["moving"]), sitk.sitkFloat32)
    fixed_image = sitk.ReadImage(str(paths["fixed"]), sitk.sitkFloat32)
    fixed_size = fixed_image.GetSize()
    result = {
        "pilot_id": args.pilot_id,
        "tool": "SimpleITK",
        "simpleitk_version": sitk.Version_VersionString(),
        "transform_direction": "MAIA_native_pixels_to_OCT_SLO_native_pixels",
        "visit_mapping_status": "provisional_unverified",
        "fit_landmark_count": len(fit_rows),
        "held_out_qc_landmark_count": len(qc_rows),
        "error_units": {
            "native": "fixed OCT/SLO native pixels",
            "standardized_512": "residual coordinates scaled to a 512 x 512 OCT/SLO grid",
            "fixed_native_size_px": list(fixed_size),
        },
        "transforms": {},
        "accepted_transform": None,
        "acceptance_status": "pending_visual_review_and_supervisor_threshold",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    for kind in ("similarity", "affine"):
        transform = initialise(kind, fit_moving, fit_fixed)
        fit_predicted, fit_errors = tre(transform, fit_moving, fit_fixed)
        qc_predicted, qc_errors = tre(transform, qc_moving, qc_fixed)
        fit_errors_512 = errors_on_512_grid(fit_predicted, fit_fixed, fixed_size)
        qc_errors_512 = errors_on_512_grid(qc_predicted, qc_fixed, fixed_size)
        # Resample expects output->input, hence the inverse of the saved
        # MAIA->OCT transform.
        warped = sitk.Resample(
            moving_image,
            fixed_image,
            transform.GetInverse(),
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        sitk.WriteImage(warped, str(run_dir / f"warped_maia_{kind}_simpleitk.mha"))
        warped_display = sitk.Cast(sitk.RescaleIntensity(warped, 0, 255), sitk.sitkUInt8)
        fixed_display = sitk.Cast(sitk.RescaleIntensity(fixed_image, 0, 255), sitk.sitkUInt8)
        sitk.WriteImage(warped_display, str(run_dir / f"warped_maia_{kind}_simpleitk.png"))
        # Red = warped MAIA, green/blue = OCT SLO. Matching dark vessels appear
        # dark/neutral; coloured vessel edges indicate a spatial mismatch.
        sitk.WriteImage(
            sitk.Compose(warped_display, fixed_display, fixed_display),
            str(run_dir / f"overlay_{kind}_simpleitk.png"),
        )
        sitk.WriteTransform(transform, str(run_dir / f"transform_{kind}_maia_to_oct.tfm"))
        result["transforms"][kind] = {
            "model": kind,
            "fit_tre_mean_px": float(fit_errors.mean()),
            "fit_tre_median_px": float(np.median(fit_errors)),
            "qc_tre_mean_px": float(qc_errors.mean()),
            "qc_tre_median_px": float(np.median(qc_errors)),
            "qc_tre_max_px": float(qc_errors.max()),
            "fit_tre_mean_px_512": float(fit_errors_512.mean()),
            "fit_tre_median_px_512": float(np.median(fit_errors_512)),
            "qc_tre_mean_px_512": float(qc_errors_512.mean()),
            "qc_tre_median_px_512": float(np.median(qc_errors_512)),
            "qc_tre_max_px_512": float(qc_errors_512.max()),
            "parameters": list(transform.GetParameters()),
            "fixed_parameters": list(transform.GetFixedParameters()),
        }

    (run_dir / "registration_result_simpleitk.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SimpleITK outputs written to {run_dir}")
    print("No transform accepted automatically; review QC and supervisor-approved threshold.")


if __name__ == "__main__":
    main()
