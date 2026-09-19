# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Build a label-blind RETFound preprocessing pilot from frozen OCT patches."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/point_alignment_cohort/frozen_v1/patch_manifest.csv"
PILOT_POINTS = ROOT / "outputs/feature_engineering_pilot_v1/pilot_points.csv"
OUTPUT_ROOT = ROOT / "outputs/retfound_preprocessing_pilot_v1"
RETFOUND_ROOT = ROOT / "external/RETFound"

MODEL_SIZE = 224
CROP_HEIGHT = 192
ANCHOR_OFFSET_FROM_TOP = 144
DETECTION_LOW_PERCENTILE = 1.0
DETECTION_HIGH_PERCENTILE = 99.5
MODEL_LOW_PERCENTILE = 1.0
MODEL_HIGH_PERCENTILE = 99.0
PROFILE_PERCENTILE = 50.0
PROFILE_SMOOTHING_SIGMA = 3.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def git_commit(repo: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def detect_anchor(raw: np.ndarray) -> tuple[int, float, float]:
    low, high = np.percentile(raw, [DETECTION_LOW_PERCENTILE, DETECTION_HIGH_PERCENTILE])
    normalised = np.clip((raw - low) / max(float(high - low), 1.0), 0.0, 1.0)
    profile = gaussian_filter1d(np.percentile(normalised, PROFILE_PERCENTILE, axis=1), PROFILE_SMOOTHING_SIGMA)
    return int(np.argmax(profile)), float(low), float(high)


def crop_bounds(anchor: int, height: int) -> tuple[int, int]:
    y0 = int(np.clip(anchor - ANCHOR_OFFSET_FROM_TOP, 0, height - CROP_HEIGHT))
    return y0, y0 + CROP_HEIGHT


def preprocess(path: Path) -> tuple[np.ndarray, dict[str, float | int | str]]:
    raw = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if raw.shape != (496, 128):
        raise ValueError(f"Unexpected patch shape {raw.shape}: {path}")
    anchor, detection_low, detection_high = detect_anchor(raw)
    y0, y1 = crop_bounds(anchor, raw.shape[0])
    crop = raw[y0:y1]
    low, high = np.percentile(crop, [MODEL_LOW_PERCENTILE, MODEL_HIGH_PERCENTILE])
    crop_norm = np.clip((crop - low) / max(float(high - low), 1.0), 0.0, 1.0)
    crop_u8 = np.rint(crop_norm * 255.0).astype(np.uint8)
    resized_width = int(round(crop_u8.shape[1] * MODEL_SIZE / crop_u8.shape[0]))
    resized = Image.fromarray(crop_u8, mode="L").resize((resized_width, MODEL_SIZE), Image.Resampling.LANCZOS)
    pad_left = (MODEL_SIZE - resized_width) // 2
    pad_right = MODEL_SIZE - resized_width - pad_left
    canvas = Image.new("L", (MODEL_SIZE, MODEL_SIZE), color=0)
    canvas.paste(resized, (pad_left, 0))
    model_rgb = np.repeat(np.asarray(canvas, dtype=np.uint8)[..., None], 3, axis=2)
    metadata: dict[str, float | int | str] = {
        "anchor_row": anchor, "crop_y0": y0, "crop_y1_exclusive": y1,
        "crop_top_clipped": int(y0 == 0), "crop_bottom_clipped": int(y1 == raw.shape[0]),
        "detection_p1": detection_low, "detection_p99_5": detection_high,
        "crop_p1": float(low), "crop_p99": float(high),
        "resized_width": resized_width, "resized_height": MODEL_SIZE,
        "pad_left": pad_left, "pad_right": pad_right,
        "processed_rgb_sha256": array_sha256(model_rgb),
    }
    return model_rgb, metadata


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def panel_for_patch(path: Path, metadata: dict[str, float | int | str], rank: int) -> Image.Image:
    raw_image = Image.open(path).convert("L")
    display_original = raw_image.resize((128, 496), Image.Resampling.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(display_original)
    y0, y1 = int(metadata["crop_y0"]), int(metadata["crop_y1_exclusive"])
    draw.rectangle((0, y0, 127, y1 - 1), outline=(0, 255, 0), width=3)
    draw.line((0, int(metadata["anchor_row"]), 127, int(metadata["anchor_row"])), fill=(255, 0, 255), width=2)
    raw = np.asarray(raw_image, dtype=np.float32)[y0:y1]
    low, high = float(metadata["crop_p1"]), float(metadata["crop_p99"])
    crop = np.rint(np.clip((raw - low) / max(high - low, 1.0), 0, 1) * 255).astype(np.uint8)
    crop_display = Image.fromarray(crop, mode="L").resize((256, 384), Image.Resampling.NEAREST).convert("RGB")
    processed, _ = preprocess(path)
    final_display = Image.fromarray(processed).resize((384, 384), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (128 + 256 + 384 + 40, 540), "white")
    panel.paste(display_original, (0, 36))
    panel.paste(crop_display, (148, 80))
    panel.paste(final_display, (424, 80))
    d = ImageDraw.Draw(panel)
    d.text((0, 6), f"Rank {rank}: original (green crop; magenta anchor)", fill="black", font=_font(15))
    d.text((148, 56), "normalised retinal crop", fill="black", font=_font(15))
    d.text((424, 56), "224x224 aspect-preserved RETFound input", fill="black", font=_font(15))
    return panel


def build_sheet(case_id: str, point_number: int, rows: list[dict[str, object]], destination: Path) -> None:
    width, header, row_height = 808, 86, 540
    sheet = Image.new("RGB", (width, header + 3 * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 12), f"{case_id} | point {point_number}", fill="black", font=_font(22))
    draw.text((15, 45), "Review only whether retinal tissue is retained without vertical distortion; sensitivity was not loaded.", fill="black", font=_font(14))
    for index, row in enumerate(sorted(rows, key=lambda item: int(item["scan_rank"]))):
        panel = panel_for_patch(Path(str(row["patch_path"])), row["metadata"], int(row["scan_rank"]))
        sheet.paste(panel, (0, header + index * row_height))
    sheet.save(destination)


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT_ROOT}")
    manifest = pd.read_csv(
        MANIFEST,
        usecols=["case_id", "study_id", "eye", "maia_timepoint", "point_number", "scan_rank", "patch_path", "output_width_px", "output_height_px", "output_sha256"],
    )
    if len(manifest) != 8658 or manifest[["case_id", "point_number", "scan_rank"]].duplicated().any():
        raise RuntimeError("Frozen patch manifest coverage is invalid.")
    if not manifest["output_width_px"].eq(128).all() or not manifest["output_height_px"].eq(496).all():
        raise RuntimeError("Frozen patch dimensions are not uniform 128x496.")

    work = Path(tempfile.mkdtemp(prefix=".retfound_preprocessing_pilot_v1_", dir=OUTPUT_ROOT.parent))
    try:
        cohort_rows = []
        for row in manifest.itertuples(index=False):
            path = ROOT / row.patch_path
            if not path.is_file() or sha256(path) != row.output_sha256:
                raise RuntimeError(f"Patch missing or hash mismatch: {path}")
            processed, metadata = preprocess(path)
            cohort_rows.append({
                "case_id": row.case_id, "point_number": int(row.point_number), "scan_rank": int(row.scan_rank),
                "anchor_row": metadata["anchor_row"], "crop_y0": metadata["crop_y0"],
                "crop_y1_exclusive": metadata["crop_y1_exclusive"],
                "crop_top_clipped": metadata["crop_top_clipped"], "crop_bottom_clipped": metadata["crop_bottom_clipped"],
                "processed_rgb_sha256": array_sha256(processed),
            })
        cohort_audit = pd.DataFrame(cohort_rows)
        cohort_audit.to_csv(work / "cohort_preprocessing_geometry_audit_label_blind.csv", index=False)

        pilot = pd.read_csv(PILOT_POINTS, usecols=["pilot_case_number", "case_id", "study_id", "outer_fold", "eye", "maia_timepoint", "point_number", "location_id", "ring", "rank1_patch_path", "rank2_patch_path", "rank3_patch_path"])
        review_dir = work / "review_sheets"
        review_dir.mkdir()
        pilot_scan_rows: list[dict[str, object]] = []
        decisions = []
        for pilot_row in pilot.itertuples(index=False):
            point_rows = []
            for rank in (1, 2, 3):
                path = ROOT / getattr(pilot_row, f"rank{rank}_patch_path")
                manifest_match = manifest.loc[
                    manifest["case_id"].eq(pilot_row.case_id)
                    & manifest["point_number"].eq(pilot_row.point_number)
                    & manifest["scan_rank"].eq(rank)
                ]
                if len(manifest_match) != 1 or path != ROOT / manifest_match.iloc[0]["patch_path"]:
                    raise RuntimeError("Pilot path does not match frozen manifest.")
                processed, metadata = preprocess(path)
                row = {
                    "pilot_case_number": int(pilot_row.pilot_case_number), "case_id": pilot_row.case_id,
                    "study_id": pilot_row.study_id, "outer_fold": int(pilot_row.outer_fold),
                    "eye": pilot_row.eye, "maia_timepoint": pilot_row.maia_timepoint,
                    "point_number": int(pilot_row.point_number), "location_id": pilot_row.location_id,
                    "ring": pilot_row.ring, "scan_rank": rank,
                    "patch_path": str(path.relative_to(ROOT)), "patch_sha256": sha256(path), **metadata,
                }
                pilot_scan_rows.append(row)
                point_rows.append({"scan_rank": rank, "patch_path": path, "metadata": metadata})
            sheet_name = f"{pilot_row.case_id}__point_{int(pilot_row.point_number):03d}.png"
            build_sheet(pilot_row.case_id, int(pilot_row.point_number), point_rows, review_dir / sheet_name)
            decisions.append({
                "case_id": pilot_row.case_id, "point_number": int(pilot_row.point_number),
                "review_sheet": str((review_dir / sheet_name).relative_to(work)),
                "retina_retained_all_three": "PENDING", "no_vertical_distortion_all_three": "PENDING",
                "decision": "PENDING", "notes": "",
            })
        pd.DataFrame(pilot_scan_rows).to_csv(work / "pilot_preprocessing_measurements_label_blind.csv", index=False)
        pd.DataFrame(decisions).to_csv(work / "PILOT_REVIEW_DECISIONS.csv", index=False)

        audit = cohort_audit["anchor_row"].to_numpy()
        specification = {
            "preprocessing_id": "retfound_preprocessing_v1_candidate",
            "status": "candidate_pending_human_visual_review_and_weighted_embedding_preflight",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sensitivity_loaded": False,
            "source_patch_manifest": str(MANIFEST.relative_to(ROOT)), "source_patch_manifest_sha256": sha256(MANIFEST),
            "pilot_points": str(PILOT_POINTS.relative_to(ROOT)), "pilot_points_sha256": sha256(PILOT_POINTS),
            "retfound_repository": "https://github.com/rmaphoh/RETFound",
            "retfound_commit": git_commit(RETFOUND_ROOT),
            "checkpoint_repo_id": "YukunZhou/RETFound_mae_natureOCT",
            "checkpoint_filename": "RETFound_mae_natureOCT.pth",
            "architecture": "RETFound_mae ViT-Large", "embedding_dimensions": 1024,
            "input": {"mode": "8-bit grayscale", "width": 128, "height": 496},
            "anchor_detection": {
                "description": "argmax of Gaussian-smoothed per-row median after full-patch percentile normalisation",
                "low_percentile": DETECTION_LOW_PERCENTILE, "high_percentile": DETECTION_HIGH_PERCENTILE,
                "row_percentile": PROFILE_PERCENTILE, "smoothing_sigma_rows": PROFILE_SMOOTHING_SIGMA,
            },
            "crop": {"height": CROP_HEIGHT, "anchor_offset_from_top": ANCHOR_OFFSET_FROM_TOP,
                     "boundary_handling": "translate fixed-height crop inside image; never rescale vertically to alter anatomy"},
            "normalisation": {"crop_low_percentile": MODEL_LOW_PERCENTILE, "crop_high_percentile": MODEL_HIGH_PERCENTILE,
                              "clip_to_0_1": True, "convert_to_uint8": True},
            "resize": {"canvas": [224, 224], "preserve_aspect_ratio": True, "interpolation": "Pillow LANCZOS",
                       "letterbox_value_uint8": 0, "grayscale_repeated_to_three_channels": True,
                       "expected_resized_width": 149, "expected_horizontal_padding": [37, 38]},
            "model_tensor_standardisation": "After letterboxing, divide by 255 and apply the official feature-notebook per-image, per-channel z-score with epsilon guard; freeze after checkpoint preflight.",
            "cohort_audit": {
                "patches": len(cohort_audit), "points": 2886, "eye_visits": 78, "participants": 22,
                "anchor_min": int(audit.min()), "anchor_p01": float(np.percentile(audit, 1)),
                "anchor_median": float(np.median(audit)), "anchor_p99": float(np.percentile(audit, 99)),
                "anchor_max": int(audit.max()), "top_clipped_crops": int(cohort_audit["crop_top_clipped"].sum()),
                "bottom_clipped_crops": int(cohort_audit["crop_bottom_clipped"].sum()),
            },
        }
        (work / "PREPROCESSING_SPECIFICATION_CANDIDATE_V1.json").write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
        (work / "README.md").write_text(
            "# RETFound preprocessing pilot v1\n\n"
            "Label-blind 16-point/48-patch visual pilot plus full 8,658-patch geometry audit. "
            "Green boxes mark the retained crop; magenta lines mark the image-derived anchor. "
            "No sensitivity, prediction, residual, or model metric was loaded.\n",
            encoding="utf-8",
        )
        names = [p for p in work.rglob("*") if p.is_file()]
        pd.DataFrame([{"file": str(p.relative_to(work)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(names)]).to_csv(work / "OUTPUT_MANIFEST_V1.csv", index=False)
        work.replace(OUTPUT_ROOT)
        print(json.dumps({"status": "candidate_preprocessing_pilot_created", "pilot_points": len(pilot),
                          "pilot_patches": len(pilot_scan_rows), "cohort_patches_audited": len(cohort_audit),
                          "sensitivity_loaded": False, "next_gate": "human visual review then official-weight deterministic embedding preflight"}, indent=2))
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
