# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Run the small post-registration point-alignment pilot only.

This script deliberately processes a fixed six-case pilot. It does not scale
to the full frozen cohort. MAIA marker centres and printed values are recovered
from clean/overlay image differences, then reconciled to the Pointwise sheet
export before any OCT mapping is attempted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import SimpleITK as sitk


PROJECT_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
WORKING_ROOT = PROJECT_ROOT / "outputs" / "2026-07-14_deidentified_working_data"
MANIFEST = WORKING_ROOT / "working_files" / "registration_manifest_frozen_v1.csv"
RECONCILIATION = WORKING_ROOT / "working_files" / "registration_manifest_reconciliation_v1.csv"
DECISIONS = PROJECT_ROOT / "outputs" / "automatic_registration_batch" / "automatic_visual_review_decisions.csv"
MANUAL_SUMMARY = PROJECT_ROOT / "outputs" / "registration_runs" / "registration_summary.csv"
WORKBOOK = WORKING_ROOT / "mesopic_MAIA_deidentified.xlsx"
DEFAULT_LABELS = WORKING_ROOT / "working_files" / "maia_pointwise_labels_from_workbook_v1.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "post_registration_point_alignment_pilot"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

# Frozen after numeric and visual inspection of the six-case pilot geometry,
# before any model fitting or performance review. The horizontal field is
# physical rather than source-pixel based so that 512- and 1024-A-scan exports
# cover the same retinal extent. The common axial sampling is retained in full.
# Supervisor-confirmed approximate MAIA stimulus diameter.  Modelling must use
# the v2 within-scan sample manifest as well as these three scan-level crops.
PATCH_WIDTH_MM = 0.100
PATCH_OUTPUT_WIDTH_PX = 128
PATCH_OUTPUT_HEIGHT_PX = 496
PATCH_PADDING_MODE = "reflect"
PATCH_EXCESSIVE_PADDING_FRACTION = 0.25

ALIGNMENT_PILOT_CASE_IDS = (
    "PUBLIC-PARTICIPANT-001__R__BL__OCT_V1",
    "PUBLIC-PARTICIPANT-001__L__Y01__OCT_V2",
    "PUBLIC-PARTICIPANT-006__R__BL__OCT_V1",
    "PUBLIC-PARTICIPANT-008__R__Y01__OCT_V2",
    "PUBLIC-PARTICIPANT-005__L__Y01__OCT_V2",
    "PUBLIC-PARTICIPANT-007__L__BL__OCT_V1",
)

# Three additional MAIA-only calibration cases resolve repeated sensitivity
# signatures that make point numbers non-identifiable from only three cases per
# eye. They are not transformed, mapped to OCT, or used for patch extraction.
NUMBERING_CALIBRATION_CASE_IDS = (
    "PUBLIC-PARTICIPANT-002__R__BL__OCT_V1",
    "PUBLIC-PARTICIPANT-003__R__BL__OCT_V1",
    "PUBLIC-PARTICIPANT-007__L__Y01__OCT_V2",
)
PILOT_CASE_IDS = ALIGNMENT_PILOT_CASE_IDS + NUMBERING_CALIBRATION_CASE_IDS

# One inner marker is substantially overpainted by the fixation trace. Its
# modal circle colour reads as 25 dB, while the adjacent printed text clearly
# reads 27 dB on the enlarged clean/overlay-difference QC crop.
# The governed study run contained a documented manual visual-value override.
# Its case-level value is intentionally omitted from the public release.
VISUAL_VALUE_OVERRIDES: dict[tuple[str, str], int] = {}

RINGS = ("outer", "middle", "inner")
RING_SCALE = {"outer": 1.0, "middle": 0.6, "inner": 0.2}

VALUE_TO_RGB = {
    -1: (1, 1, 1),
    0: (105, 1, 150),
    1: (135, 1, 120),
    2: (165, 1, 90),
    3: (195, 1, 60),
    4: (225, 1, 30),
    5: (255, 0, 1),
    **{value: (255, 10 * (value - 5), 1) for value in range(6, 18)},
    18: (255, 141, 1),
    19: (255, 162, 1),
    20: (255, 183, 1),
    21: (255, 204, 1),
    22: (255, 255, 1),
    23: (235, 255, 1),
    24: (215, 255, 1),
    25: (185, 255, 1),
    26: (155, 245, 1),
    27: (125, 235, 1),
    28: (95, 225, 1),
    29: (65, 215, 1),
    30: (35, 205, 1),
    31: (0, 195, 1),
    32: (0, 185, 1),
    33: (0, 175, 1),
    34: (0, 165, 1),
    35: (0, 145, 1),
    36: (0, 125, 1),
}
RGB_TO_VALUE = {rgb: value for value, rgb in VALUE_TO_RGB.items()}


@dataclass(frozen=True)
class GridPoint:
    location_id: str
    ring: str
    angle_index: int | None
    x: float
    y: float
    marker_score: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_working_path(relative: str) -> Path:
    return WORKING_ROOT / relative


def disk_offsets(radius: int) -> list[tuple[int, int]]:
    return [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]


DISK5 = disk_offsets(5)


def disk_score(mask: np.ndarray, x: int, y: int) -> float:
    height, width = mask.shape
    values = [
        mask[y + dy, x + dx]
        for dx, dy in DISK5
        if 0 <= x + dx < width and 0 <= y + dy < height
    ]
    return float(np.mean(values)) if values else 0.0


def refine_marker(mask: np.ndarray, expected_x: float, expected_y: float) -> tuple[float, float, float]:
    centre_x = int(round(expected_x))
    centre_y = int(round(expected_y))
    candidates = []
    for y in range(centre_y - 5, centre_y + 6):
        for x in range(centre_x - 5, centre_x + 6):
            candidates.append((disk_score(mask, x, y), x, y))
    score, x, y = max(candidates)
    return float(x), float(y), float(score)


def extract_grid(clean_path: Path, overlay_path: Path) -> tuple[list[GridPoint], dict[str, object], np.ndarray]:
    clean = np.asarray(Image.open(clean_path).convert("RGB"), dtype=np.int16)
    overlay = np.asarray(Image.open(overlay_path).convert("RGB"), dtype=np.int16)
    if clean.shape != overlay.shape:
        raise ValueError(f"Clean/overlay shape mismatch: {clean.shape} versus {overlay.shape}")
    difference = np.max(np.abs(overlay - clean), axis=2)
    marker_mask = difference >= 40

    eroded = sitk.BinaryErode(
        sitk.GetImageFromArray(marker_mask.astype(np.uint8)),
        [4, 4],
        sitk.sitkBall,
    )
    connected = sitk.ConnectedComponent(eroded)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(connected)
    small_centres = []
    for label in stats.GetLabels():
        area = stats.GetNumberOfPixels(label)
        _, _, width, height = stats.GetBoundingBox(label)
        if 8 <= area <= 25 and width <= 5 and height <= 5:
            small_centres.append(stats.GetCentroid(label))
    if len(small_centres) < 20:
        raise ValueError(f"Only {len(small_centres)} round-marker candidates survived erosion")

    candidates = np.asarray(small_centres, dtype=float)
    median = np.median(candidates, axis=0)
    distances_from_median = np.linalg.norm(candidates - median, axis=1)
    clustered = candidates[distances_from_median <= 190.0]
    provisional_centre = np.mean(clustered, axis=0)
    provisional_radii = np.linalg.norm(clustered - provisional_centre, axis=1)
    outer = clustered[provisional_radii >= 110.0]
    if len(outer) != 12:
        raise ValueError(f"Expected 12 outer-ring markers, found {len(outer)}")

    centre = np.mean(outer, axis=0)
    outer_radius = float(np.median(np.linalg.norm(outer - centre, axis=1)))
    top = outer[np.argmin(outer[:, 1])]
    base_angle = math.atan2(top[1] - centre[1], top[0] - centre[0])
    step = math.tau / 12.0

    grid: list[GridPoint] = []
    for ring in RINGS:
        radius = outer_radius * RING_SCALE[ring]
        for angle_index in range(12):
            angle = base_angle + angle_index * step
            expected_x = centre[0] + radius * math.cos(angle)
            expected_y = centre[1] + radius * math.sin(angle)
            x, y, score = refine_marker(marker_mask, expected_x, expected_y)
            grid.append(
                GridPoint(
                    location_id=f"{ring}_{angle_index:02d}",
                    ring=ring,
                    angle_index=angle_index,
                    x=x,
                    y=y,
                    marker_score=score,
                )
            )
    centre_score = disk_score(marker_mask, int(round(centre[0])), int(round(centre[1])))
    grid.append(
        GridPoint(
            location_id="centre",
            ring="centre",
            angle_index=None,
            x=float(centre[0]),
            y=float(centre[1]),
            marker_score=centre_score,
        )
    )

    if len({(round(point.x), round(point.y)) for point in grid}) != 37:
        raise ValueError("Recovered grid contains duplicate marker centres")
    ring_counts = Counter(point.ring for point in grid)
    audit = {
        "native_image_size_px": [int(clean.shape[1]), int(clean.shape[0])],
        "difference_pixels_ge_20": int((difference >= 20).sum()),
        "round_marker_candidates": len(small_centres),
        "clustered_round_marker_candidates": len(clustered),
        "grid_centre_px": [float(centre[0]), float(centre[1])],
        "outer_radius_px": outer_radius,
        "base_angle_degrees": math.degrees(base_angle),
        "ring_counts": dict(ring_counts),
        "minimum_marker_score": min(point.marker_score for point in grid if point.ring != "centre"),
        "centre_marker_score": centre_score,
    }
    return grid, audit, difference


def render_template(text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    bbox = font.getbbox(text)
    image = Image.new("L", (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4), 0)
    ImageDraw.Draw(image).text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=255)
    return np.asarray(image) >= 128


def label_window(point: GridPoint, centre: GridPoint, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    if point.ring == "centre":
        x0, x1 = int(round(point.x + 4)), int(round(point.x + 43))
        y0, y1 = int(round(point.y - 18)), int(round(point.y + 19))
    else:
        dx = point.x - centre.x
        dy = point.y - centre.y
        if abs(dx) >= abs(dy):
            if dx >= 0:
                x0, x1 = int(round(point.x + 4)), int(round(point.x + 43))
            else:
                x0, x1 = int(round(point.x - 43)), int(round(point.x - 4))
            y0, y1 = int(round(point.y - 18)), int(round(point.y + 19))
        else:
            x0, x1 = int(round(point.x - 23)), int(round(point.x + 24))
            if dy >= 0:
                y0, y1 = int(round(point.y + 4)), int(round(point.y + 36))
            else:
                y0, y1 = int(round(point.y - 36)), int(round(point.y - 4))
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def template_score(template: np.ndarray, observed: np.ndarray) -> tuple[float, tuple[int, int]]:
    template_height, template_width = template.shape
    observed_height, observed_width = observed.shape
    best_score = -1.0
    best_shift = (0, 0)
    if template_height > observed_height or template_width > observed_width:
        return best_score, best_shift
    template_pixels = int(template.sum())
    for y in range(observed_height - template_height + 1):
        for x in range(observed_width - template_width + 1):
            sample = observed[y : y + template_height, x : x + template_width]
            denominator = template_pixels + int(sample.sum())
            score = (
                2.0 * int(np.logical_and(template, sample).sum()) / denominator
                if denominator
                else 0.0
            )
            if score > best_score:
                best_score = score
                best_shift = (x, y)
    return best_score, best_shift


def ocr_values(
    difference: np.ndarray,
    grid: list[GridPoint],
    candidate_values: Iterable[int],
    overlay: np.ndarray,
) -> list[dict[str, object]]:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Required annotation font not found: {FONT_PATH}")
    font = ImageFont.truetype(str(FONT_PATH), 14)
    templates = {
        value: render_template("<0" if value == -1 else str(value), font)
        for value in candidate_values
    }
    observed_mask = difference >= 20
    centre = next(point for point in grid if point.ring == "centre")
    yy, xx = np.ogrid[: observed_mask.shape[0], : observed_mask.shape[1]]
    marker_exclusion = np.zeros_like(observed_mask)
    for marker in grid:
        marker_exclusion[(xx - marker.x) ** 2 + (yy - marker.y) ** 2 <= 7.0 ** 2] = True

    rows = []
    for point in grid:
        marker_pixels = []
        for dx, dy in DISK5:
            x = int(round(point.x)) + dx
            y = int(round(point.y)) + dy
            if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
                marker_pixels.append(tuple(int(value) for value in overlay[y, x, :]))
        palette_pixels = [rgb for rgb in marker_pixels if rgb in RGB_TO_VALUE]
        marker_rgb = list(Counter(palette_pixels).most_common(1)[0][0]) if palette_pixels else list(Counter(marker_pixels).most_common(1)[0][0])
        r, g, b = marker_rgb
        if max(marker_rgb) <= 12:
            allowed_values = [-1]
        elif b >= 40 and b > g:
            allowed_values = list(range(0, 5))
        elif r >= 220 and g < 50:
            allowed_values = list(range(5, 10))
        elif r >= 200 and g >= 50:
            allowed_values = list(range(10, 26))
        elif g >= 180 and r < 200:
            allowed_values = list(range(24, 37))
        else:
            allowed_values = list(candidate_values)
        x0, y0, x1, y1 = label_window(point, centre, observed_mask.shape)
        ranked = []
        for value in allowed_values:
            template = templates[value]
            candidate_rgb = np.asarray(VALUE_TO_RGB[value], dtype=np.int16)
            colour_distance = np.max(
                np.abs(overlay.astype(np.int16) - candidate_rgb),
                axis=2,
            )
            point_label_mask = np.logical_and(observed_mask, colour_distance <= 4)
            point_label_mask[marker_exclusion] = False
            observed = point_label_mask[y0:y1, x0:x1]
            score, shift = template_score(template, observed)
            ranked.append((score, value, shift))
        ranked.sort(reverse=True)
        best_score, best_value, best_shift = ranked[0]
        if len(ranked) > 1:
            second_score, second_value, _ = ranked[1]
        else:
            second_score, second_value = 0.0, ""
        rows.append(
            {
                "location_id": point.location_id,
                "ocr_candidate_db": best_value,
                "ocr_score": best_score,
                "ocr_margin": best_score - second_score,
                "ocr_second_value": second_value,
                "ocr_allowed_values": "|".join(str(value) for value in allowed_values),
                "marker_r": marker_rgb[0],
                "marker_g": marker_rgb[1],
                "marker_b": marker_rgb[2],
                "ocr_scores_json": json.dumps({str(value): score for score, value, _ in ranked}),
                "label_window_x0": x0,
                "label_window_y0": y0,
                "label_window_x1": x1,
                "label_window_y1": y1,
                "label_template_x": x0 + best_shift[0],
                "label_template_y": y0 + best_shift[1],
            }
        )
    return rows


def hungarian(cost: list[list[int]]) -> list[int]:
    """Return the assigned column for every row (square minimisation matrix)."""
    n = len(cost)
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [10**9] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 10**9
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j] = current
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def transform_points_to_oct(
    manifest: dict[str, str],
    decision: dict[str, str] | None,
    points: list[dict[str, object]],
) -> tuple[list[tuple[float, float]], dict[str, object]]:
    moving_path = resolve_working_path(manifest["maia_clean_path"])
    fixed_path = resolve_working_path(manifest["oct_slo_path"])
    moving_size = Image.open(moving_path).size
    fixed_size = Image.open(fixed_path).size
    transform_kind = decision["final_transform"] if decision else manifest["provisional_transform"]
    output_dir = PROJECT_ROOT / manifest["registration_output_path"]

    if decision:
        transform_path = output_dir / f"{transform_kind}_moving_to_fixed_512.tfm"
        result_path = output_dir / "automatic_registration_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if tuple(result["moving_native_size_px"]) != moving_size:
            raise ValueError(f"Automatic moving-size metadata mismatch for {manifest['case_id']}")
        if tuple(result["fixed_native_size_px"]) != fixed_size:
            raise ValueError(f"Automatic fixed-size metadata mismatch for {manifest['case_id']}")
        transform = sitk.ReadTransform(str(transform_path))
        transformed = []
        for point in points:
            moving_512 = (
                float(point["maia_x_px"]) * 512.0 / moving_size[0],
                float(point["maia_y_px"]) * 512.0 / moving_size[1],
            )
            fixed_512 = transform.TransformPoint(moving_512)
            transformed.append(
                (
                    fixed_512[0] * fixed_size[0] / 512.0,
                    fixed_512[1] * fixed_size[1] / 512.0,
                )
            )
        audit = {
            "transform_kind": transform_kind,
            "transform_source": "automatic_final_visual_decision",
            "transform_file": str(transform_path.relative_to(PROJECT_ROOT)),
            "transform_direction": "MAIA_512_to_OCT_SLO_512_then_rescaled_to_native",
            "moving_native_size_px": list(moving_size),
            "fixed_native_size_px": list(fixed_size),
            "moving_to_512_scale": [512.0 / moving_size[0], 512.0 / moving_size[1]],
            "fixed_512_to_native_scale": [fixed_size[0] / 512.0, fixed_size[1] / 512.0],
            "registration_result_file": str(result_path.relative_to(PROJECT_ROOT)),
        }
    else:
        transform_path = output_dir / f"transform_{transform_kind}_maia_to_oct.tfm"
        result_path = output_dir / "registration_result_simpleitk.json"
        transform = sitk.ReadTransform(str(transform_path))
        transformed = [
            transform.TransformPoint((float(point["maia_x_px"]), float(point["maia_y_px"])))
            for point in points
        ]
        audit = {
            "transform_kind": transform_kind,
            "transform_source": "accepted_manual_reference",
            "transform_file": str(transform_path.relative_to(PROJECT_ROOT)),
            "transform_direction": "MAIA_native_to_OCT_SLO_native",
            "moving_native_size_px": list(moving_size),
            "fixed_native_size_px": list(fixed_size),
            "moving_to_512_scale": "not_used",
            "fixed_512_to_native_scale": "not_used",
            "registration_result_file": str(result_path.relative_to(PROJECT_ROOT)),
        }
    return [(float(x), float(y)) for x, y in transformed], audit


def project_to_segment(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> dict[str, float | bool]:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 0:
        raise ValueError("Zero-length OCT scan-line segment")
    t_raw = ((px - ax) * dx + (py - ay) * dy) / denominator
    t = min(1.0, max(0.0, t_raw))
    projected_x = ax + t * dx
    projected_y = ay + t * dy
    distance = math.hypot(px - projected_x, py - projected_y)
    return {
        "t_raw": t_raw,
        "t": t,
        "projection_clipped_to_segment": not (0.0 <= t_raw <= 1.0),
        "projected_x": projected_x,
        "projected_y": projected_y,
        "distance_px": distance,
    }


def oct_geometry(manifest: dict[str, str]) -> dict[str, object]:
    slo_path = resolve_working_path(manifest["oct_slo_path"])
    oct_dir = slo_path.parent
    required = {
        "slo": slo_path,
        "scan_area": oct_dir / "slo_area.csv",
        "scan_coordinates": resolve_working_path(manifest["scan_coordinates_path"]),
        "slo_metadata": oct_dir / "slo_metadata.csv",
        "bscan_metadata": oct_dir / "bscan_metadata.csv",
        "scan_positions_image": oct_dir / "slo_scan_positions.png",
        "volume": resolve_working_path(manifest["oct_volume_path"]),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing OCT inputs for {manifest['case_id']}: {missing}")

    scan_rows = read_csv(required["scan_coordinates"])
    area_rows = read_csv(required["scan_area"])
    slo_metadata = read_csv(required["slo_metadata"])[0]
    bscan_metadata = read_csv(required["bscan_metadata"])[0]
    bscan_paths = sorted(required["volume"].glob("bscan_*.png"))
    if len(scan_rows) != int(manifest["n_coordinate_rows"]):
        raise ValueError(f"Coordinate-row count mismatch for {manifest['case_id']}")
    if len(bscan_paths) != int(manifest["n_bscans"]):
        raise ValueError(f"B-scan file count mismatch for {manifest['case_id']}")

    bscan_sizes = [Image.open(path).size for path in bscan_paths]
    if len(set(bscan_sizes)) != 1:
        raise ValueError(f"Variable B-scan dimensions within {manifest['case_id']}")
    bscan_size = bscan_sizes[0]
    if int(float(bscan_metadata["width (pixel)"])) != bscan_size[0]:
        raise ValueError(f"B-scan metadata width mismatch for {manifest['case_id']}")
    if int(float(bscan_metadata["height (pixel)"])) != bscan_size[1]:
        raise ValueError(f"B-scan metadata height mismatch for {manifest['case_id']}")

    lines = []
    for index, row in enumerate(scan_rows):
        lines.append(
            {
                "scan_index": index,
                "start": (float(row["start_x_px"]), float(row["start_y_px"])),
                "end": (float(row["end_x_px"]), float(row["end_y_px"])),
                "bscan_path": bscan_paths[index],
            }
        )
    area_x = [float(row[key]) for row in area_rows for key in ("start_x_px", "end_x_px")]
    area_y = [float(row[key]) for row in area_rows for key in ("start_y_px", "end_y_px")]
    area_bbox = (min(area_x), min(area_y), max(area_x), max(area_y))
    midpoints = np.asarray(
        [[(line["start"][0] + line["end"][0]) / 2, (line["start"][1] + line["end"][1]) / 2] for line in lines]
    )
    spacing = np.linalg.norm(np.diff(midpoints, axis=0), axis=1)
    median_spacing = float(np.median(spacing)) if len(spacing) else float("nan")
    return {
        "paths": required,
        "lines": lines,
        "area_bbox": area_bbox,
        "slo_size": Image.open(slo_path).size,
        "slo_metadata": slo_metadata,
        "bscan_metadata": bscan_metadata,
        "bscan_size": bscan_size,
        "median_scan_spacing_px": median_spacing,
        "scan_spacing_min_px": float(spacing.min()) if len(spacing) else float("nan"),
        "scan_spacing_max_px": float(spacing.max()) if len(spacing) else float("nan"),
    }


def draw_qc_overlays(
    output_root: Path,
    manifest: dict[str, str],
    points: list[dict[str, object]],
    geometry: dict[str, object],
) -> None:
    overlay_dir = output_root / "qc_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    case_id = manifest["case_id"]
    font = ImageFont.truetype(str(FONT_PATH), 14)

    maia = Image.open(resolve_working_path(manifest["maia_clean_path"])).convert("RGB")
    maia_draw = ImageDraw.Draw(maia)
    for point in points:
        x, y = float(point["maia_x_px"]), float(point["maia_y_px"])
        maia_draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 0, 255), width=2)
        maia_draw.text(
            (x + 6, y - 7),
            f"{point['point_number']}:{point['printed_sensitivity_db']}",
            fill=(255, 255, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    maia.save(overlay_dir / f"{case_id}__maia_points_1_37.png")

    slo = Image.open(resolve_working_path(manifest["oct_slo_path"])).convert("RGB")
    draw = ImageDraw.Draw(slo, "RGBA")
    for line in geometry["lines"]:
        draw.line((*line["start"], *line["end"]), fill=(0, 220, 255, 90), width=1)
    x0, y0, x1, y1 = geometry["area_bbox"]
    draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0, 220), width=3)
    rank_colours = {1: (255, 0, 80, 170), 2: (255, 140, 0, 140), 3: (255, 230, 0, 120)}
    for point in points:
        x, y = float(point["oct_slo_x_px"]), float(point["oct_slo_y_px"])
        for rank in (3, 2, 1):
            draw.line(
                (x, y, float(point[f"scan_{rank}_projected_x_px"]), float(point[f"scan_{rank}_projected_y_px"])),
                fill=rank_colours[rank],
                width=2,
            )
        point_colour = (0, 255, 0, 255) if point["inside_scan_area"] else (255, 0, 0, 255)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=point_colour, outline=(0, 0, 0, 255), width=1)
        draw.text(
            (x + 5, y - 7),
            str(point["point_number"]),
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
    slo.save(overlay_dir / f"{case_id}__oct_three_nearest_scans.png")


def draw_maia_only_qc_overlay(
    output_root: Path,
    manifest: dict[str, str],
    points: list[dict[str, object]],
) -> None:
    """Draw the same MAIA label audit for numbering-only calibration cases."""
    overlay_dir = output_root / "qc_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT_PATH), 14)
    maia = Image.open(resolve_working_path(manifest["maia_clean_path"])).convert("RGB")
    draw = ImageDraw.Draw(maia)
    for point in points:
        x, y = float(point["maia_x_px"]), float(point["maia_y_px"])
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 0, 255), width=2)
        draw.text(
            (x + 6, y - 7),
            f"{point['point_number']}:{point['printed_sensitivity_db']}",
            fill=(255, 255, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    maia.save(overlay_dir / f"{manifest['case_id']}__maia_points_1_37.png")


def extract_patch(
    bscan_path: Path,
    centre_x_px: float,
    scale_x_mm_per_px: float,
    output_path: Path,
) -> dict[str, object]:
    """Extract the frozen physical-width, full-depth pilot patch."""
    image = Image.open(bscan_path).convert("L")
    source = np.asarray(image)
    height, width = source.shape
    if height != PATCH_OUTPUT_HEIGHT_PX:
        raise ValueError(
            f"Unexpected B-scan height {height} for {bscan_path}; "
            f"frozen patch height is {PATCH_OUTPUT_HEIGHT_PX}"
        )
    source_width_px = max(3, int(round(PATCH_WIDTH_MM / scale_x_mm_per_px)))
    centre_index_px = int(round(centre_x_px))
    source_x0 = centre_index_px - source_width_px // 2
    source_x1 = source_x0 + source_width_px
    pad_left = max(0, -source_x0)
    pad_right = max(0, source_x1 - width)
    padded = np.pad(source, ((0, 0), (pad_left, pad_right)), mode=PATCH_PADDING_MODE)
    crop_x0 = source_x0 + pad_left
    crop_x1 = source_x1 + pad_left
    crop = padded[:, crop_x0:crop_x1]
    if crop.shape != (PATCH_OUTPUT_HEIGHT_PX, source_width_px):
        raise ValueError(f"Unexpected patch source shape {crop.shape} for {bscan_path}")
    resized = Image.fromarray(crop).resize(
        (PATCH_OUTPUT_WIDTH_PX, PATCH_OUTPUT_HEIGHT_PX),
        resample=Image.Resampling.BILINEAR,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)
    padded_fraction = (pad_left + pad_right) / source_width_px
    return {
        "bscan_width_px": width,
        "bscan_height_px": height,
        "source_centre_x_px": centre_x_px,
        "source_centre_index_px": centre_index_px,
        "source_patch_x0_px": source_x0,
        "source_patch_x1_exclusive_px": source_x1,
        "source_patch_width_px": source_width_px,
        "requested_patch_width_mm": PATCH_WIDTH_MM,
        "realised_patch_width_mm": source_width_px * scale_x_mm_per_px,
        "pad_left_px": pad_left,
        "pad_right_px": pad_right,
        "padded_fraction": padded_fraction,
        "edge_padding_applied": bool(pad_left or pad_right),
        "excessive_edge_padding": padded_fraction > PATCH_EXCESSIVE_PADDING_FRACTION,
        "padding_mode": PATCH_PADDING_MODE,
        "output_width_px": PATCH_OUTPUT_WIDTH_PX,
        "output_height_px": PATCH_OUTPUT_HEIGHT_PX,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest_by_case = {row["case_id"]: row for row in read_csv(MANIFEST)}
    decisions_by_case = {row["case_id"]: row for row in read_csv(DECISIONS)}
    label_rows = read_csv(args.labels_csv)
    labels_by_case: dict[tuple[str, str, str], dict[int, int]] = {}
    for row in label_rows:
        key = (row["study_id"], row["eye"], row["timepoint"])
        labels_by_case.setdefault(key, {})[int(row["point_number"])] = int(row["sensitivity_db"])

    args.output_root.mkdir(parents=True, exist_ok=True)
    extraction_rows: list[dict[str, object]] = []
    case_audits = []
    candidate_values = range(-1, 37)

    for case_id in PILOT_CASE_IDS:
        manifest = manifest_by_case[case_id]
        clean_path = resolve_working_path(manifest["maia_clean_path"])
        overlay_path = resolve_working_path(manifest["maia_overlay_path"])
        grid, grid_audit, difference = extract_grid(clean_path, overlay_path)
        overlay = np.asarray(Image.open(overlay_path).convert("RGB"), dtype=np.uint8)
        ocr = {row["location_id"]: row for row in ocr_values(difference, grid, candidate_values, overlay)}
        workbook = labels_by_case[(manifest["study_id"], manifest["eye"], manifest["maia_timepoint"])]
        endpoint_values = [workbook[point_number] for point_number in range(1, 38)]
        point0 = workbook[0]
        ring_values = []
        for point in grid:
            if point.ring == "centre":
                continue
            row = ocr[point.location_id]
            rgb = (int(row["marker_r"]), int(row["marker_g"]), int(row["marker_b"]))
            if rgb not in RGB_TO_VALUE:
                raise ValueError(f"No MAIA annotation-palette value for {case_id} {point.location_id}: {rgb}")
            palette_value = RGB_TO_VALUE[rgb]
            row["marker_palette_value_db"] = palette_value
            recovered_value = VISUAL_VALUE_OVERRIDES.get((case_id, point.location_id), palette_value)
            row["printed_sensitivity_db"] = recovered_value
            row["value_recovery_method"] = (
                "enlarged_difference_crop_visual_text_override_due_fixation_overpaint"
                if (case_id, point.location_id) in VISUAL_VALUE_OVERRIDES
                else "overlay_marker_palette_with_numeric_text_template_check"
            )
            ring_values.append(recovered_value)
        remaining = Counter(endpoint_values) - Counter(ring_values)
        excess = Counter(ring_values) - Counter(endpoint_values)
        if excess or sum(remaining.values()) != 1:
            raise ValueError(
                f"Ring-marker palette values do not leave one centre value for {case_id}: "
                f"remaining={remaining}, excess={excess}"
            )
        centre_value = next(remaining.elements())
        for point in grid:
            row = ocr[point.location_id]
            if point.ring == "centre":
                row["marker_palette_value_db"] = ""
                row["printed_sensitivity_db"] = centre_value
                row["value_recovery_method"] = (
                    "centre_numeric_text_visually_confirmed_after_ring_multiset_residual"
                )
            scores = json.loads(str(row["ocr_scores_json"]))
            row["recovered_value_template_score"] = scores.get(str(row["printed_sensitivity_db"]), 0.0)
        recovered_values = [int(ocr[point.location_id]["printed_sensitivity_db"]) for point in grid]
        multiset_match = Counter(recovered_values) == Counter(endpoint_values)
        decision = decisions_by_case.get(case_id)
        case_audits.append(
            {
                "case_id": case_id,
                "case_role": (
                    "alignment_patch_pilot"
                    if case_id in ALIGNMENT_PILOT_CASE_IDS
                    else "maia_numbering_calibration_only"
                ),
                "study_id": manifest["study_id"],
                "eye": manifest["eye"],
                "maia_timepoint": manifest["maia_timepoint"],
                "oct_visit": manifest["oct_visit"],
                "registration_source": "automatic" if decision else "manual_reference",
                "final_transform": decision["final_transform"] if decision else manifest["provisional_transform"],
                "visual_grade": decision["visual_grade"] if decision else "manual_reference_accepted",
                "n_bscans": int(manifest["n_bscans"]),
                "n_coordinate_rows": int(manifest["n_coordinate_rows"]),
                "point0_workbook_db": point0,
                "centre_printed_sensitivity_db": centre_value,
                "recovered_marker_count": len(grid),
                "printed_value_multiset_matches_points_1_37": multiset_match,
                "minimum_ocr_score": min(float(row["ocr_score"]) for row in ocr.values()),
                "minimum_ocr_margin": min(float(row["ocr_margin"]) for row in ocr.values()),
                **grid_audit,
            }
        )
        for point in grid:
            extraction_rows.append(
                {
                    "case_id": case_id,
                    "case_role": (
                        "alignment_patch_pilot"
                        if case_id in ALIGNMENT_PILOT_CASE_IDS
                        else "maia_numbering_calibration_only"
                    ),
                    "study_id": manifest["study_id"],
                    "eye": manifest["eye"],
                    "maia_timepoint": manifest["maia_timepoint"],
                    "location_id": point.location_id,
                    "ring": point.ring,
                    "angle_index": "" if point.angle_index is None else point.angle_index,
                    "maia_x_px": point.x,
                    "maia_y_px": point.y,
                    "marker_score": point.marker_score,
                    **ocr[point.location_id],
                }
            )

    locations = sorted({row["location_id"] for row in extraction_rows})
    point_numbers = list(range(1, 38))
    case_order = list(PILOT_CASE_IDS)
    extracted_by_case_location = {
        (row["case_id"], row["location_id"]): int(row["printed_sensitivity_db"])
        for row in extraction_rows
    }
    point_by_eye_location: dict[str, dict[str, int]] = {}
    mapping_mismatches_by_eye: dict[str, int] = {}
    point_signature_audit_by_eye: dict[str, dict[str, object]] = {}
    for eye in ("R", "L"):
        eye_cases = [case_id for case_id in case_order if manifest_by_case[case_id]["eye"] == eye]
        signature_groups: dict[tuple[int, ...], list[int]] = {}
        for point_number in point_numbers:
            signature = tuple(
                labels_by_case[
                    (
                        manifest_by_case[case_id]["study_id"],
                        eye,
                        manifest_by_case[case_id]["maia_timepoint"],
                    )
                ][point_number]
                for case_id in eye_cases
            )
            signature_groups.setdefault(signature, []).append(point_number)
        ambiguous_groups = [group for group in signature_groups.values() if len(group) > 1]
        point_signature_audit_by_eye[eye] = {
            "case_ids": eye_cases,
            "unique_point_signatures": len(signature_groups),
            "ambiguous_point_groups": ambiguous_groups,
        }
        cost = []
        for location in locations:
            location_cost = []
            for point_number in point_numbers:
                mismatches = 0
                for case_id in eye_cases:
                    manifest = manifest_by_case[case_id]
                    workbook = labels_by_case[(manifest["study_id"], eye, manifest["maia_timepoint"])]
                    if extracted_by_case_location[(case_id, location)] != workbook[point_number]:
                        mismatches += 1
                location_cost.append(mismatches)
            cost.append(location_cost)
        assignment = hungarian(cost)
        point_by_eye_location[eye] = {
            locations[row]: point_numbers[column] for row, column in enumerate(assignment)
        }
        mapping_mismatches_by_eye[eye] = sum(cost[row][column] for row, column in enumerate(assignment))

    total_mismatches = 0
    for row in extraction_rows:
        point_number = point_by_eye_location[row["eye"]][row["location_id"]]
        manifest = manifest_by_case[row["case_id"]]
        workbook_value = labels_by_case[(manifest["study_id"], manifest["eye"], manifest["maia_timepoint"])][point_number]
        row["point_number"] = point_number
        row["workbook_sensitivity_db"] = workbook_value
        row["printed_matches_workbook"] = int(row["printed_sensitivity_db"]) == workbook_value
        total_mismatches += int(not row["printed_matches_workbook"])

    for case_id in NUMBERING_CALIBRATION_CASE_IDS:
        draw_maia_only_qc_overlay(
            args.output_root,
            manifest_by_case[case_id],
            sorted(
                [row for row in extraction_rows if row["case_id"] == case_id],
                key=lambda row: int(row["point_number"]),
            ),
        )

    scan_mapping_rows: list[dict[str, object]] = []
    patch_rows: list[dict[str, object]] = []
    source_audit_rows: list[dict[str, object]] = [
        {
            "case_id": "GLOBAL",
            "source_name": source_name,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": path.exists(),
            "is_directory": path.is_dir(),
            "note": note,
        }
        for source_name, path, note in (
            ("frozen_registration_manifest", MANIFEST, "computational cohort and source-path authority"),
            ("manifest_reconciliation", RECONCILIATION, "audit-only reconciliation reference"),
            ("automatic_visual_decisions", DECISIONS, "final automatic transform and QC authority"),
            ("manual_registration_summary", MANUAL_SUMMARY, "accepted manual-reference audit"),
            ("deidentified_maia_workbook", WORKBOOK, "authoritative Point 0-37 sensitivity source"),
            ("workbook_pointwise_label_export", args.labels_csv, "deterministic workbook label export used by pilot"),
        )
    ]
    case_audit_by_id = {row["case_id"]: row for row in case_audits}
    for case_id in ALIGNMENT_PILOT_CASE_IDS:
        manifest = manifest_by_case[case_id]
        decision = decisions_by_case.get(case_id)
        case_points = sorted(
            [row for row in extraction_rows if row["case_id"] == case_id],
            key=lambda row: int(row["point_number"]),
        )
        transformed, transform_audit = transform_points_to_oct(manifest, decision, case_points)
        geometry = oct_geometry(manifest)
        slo_width, slo_height = geometry["slo_size"]
        area_x0, area_y0, area_x1, area_y1 = geometry["area_bbox"]
        bscan_width, bscan_height = geometry["bscan_size"]
        median_spacing = float(geometry["median_scan_spacing_px"])

        for point, (oct_x, oct_y) in zip(case_points, transformed):
            projections = []
            for line in geometry["lines"]:
                projected = project_to_segment((oct_x, oct_y), line["start"], line["end"])
                projections.append({**line, **projected})
            projections.sort(key=lambda row: (float(row["distance_px"]), int(row["scan_index"])))
            nearest = projections[:3]
            point["oct_slo_x_px"] = oct_x
            point["oct_slo_y_px"] = oct_y
            point["inside_native_slo"] = 0.0 <= oct_x < slo_width and 0.0 <= oct_y < slo_height
            point["inside_scan_area"] = area_x0 <= oct_x <= area_x1 and area_y0 <= oct_y <= area_y1
            point["missing_neighbouring_scans"] = len(nearest) < 3
            indices = sorted(int(row["scan_index"]) for row in nearest)
            point["three_scan_indices_contiguous"] = (
                len(indices) == 3 and indices[1] == indices[0] + 1 and indices[2] == indices[1] + 1
            )
            for rank, mapping in enumerate(nearest, start=1):
                along_x = float(mapping["t"]) * (bscan_width - 1)
                point[f"scan_{rank}_index"] = int(mapping["scan_index"])
                point[f"scan_{rank}_along_fraction"] = float(mapping["t"])
                point[f"scan_{rank}_along_x_px"] = along_x
                point[f"scan_{rank}_distance_slo_px"] = float(mapping["distance_px"])
                point[f"scan_{rank}_distance_spacing_units"] = float(mapping["distance_px"]) / median_spacing
                point[f"scan_{rank}_projected_x_px"] = float(mapping["projected_x"])
                point[f"scan_{rank}_projected_y_px"] = float(mapping["projected_y"])
                point[f"scan_{rank}_projection_clipped"] = bool(mapping["projection_clipped_to_segment"])
                point[f"scan_{rank}_bscan_path"] = str(mapping["bscan_path"].relative_to(PROJECT_ROOT))
                scan_mapping_rows.append(
                    {
                        "case_id": case_id,
                        "study_id": point["study_id"],
                        "eye": point["eye"],
                        "maia_timepoint": point["maia_timepoint"],
                        "point_number": point["point_number"],
                        "sensitivity_db": point["printed_sensitivity_db"],
                        "scan_rank": rank,
                        "bscan_index": int(mapping["scan_index"]),
                        "bscan_path": str(mapping["bscan_path"].relative_to(PROJECT_ROOT)),
                        "line_start_x_px": mapping["start"][0],
                        "line_start_y_px": mapping["start"][1],
                        "line_end_x_px": mapping["end"][0],
                        "line_end_y_px": mapping["end"][1],
                        "projected_x_px": mapping["projected_x"],
                        "projected_y_px": mapping["projected_y"],
                        "along_fraction": mapping["t"],
                        "along_x_bscan_px": along_x,
                        "point_to_line_distance_slo_px": mapping["distance_px"],
                        "distance_in_median_spacing_units": float(mapping["distance_px"]) / median_spacing,
                        "projection_clipped_to_segment": mapping["projection_clipped_to_segment"],
                    }
                )
                patch_path = (
                    args.output_root
                    / "patches"
                    / case_id
                    / f"point_{int(point['point_number']):03d}"
                    / f"rank_{rank}_bscan_{int(mapping['scan_index']):03d}.png"
                )
                patch_audit = extract_patch(
                    mapping["bscan_path"],
                    along_x,
                    float(geometry["bscan_metadata"]["Scale X"]),
                    patch_path,
                )
                patch_rows.append(
                    {
                        "case_id": case_id,
                        "study_id": point["study_id"],
                        "eye": point["eye"],
                        "maia_timepoint": point["maia_timepoint"],
                        "point_number": point["point_number"],
                        "sensitivity_db": point["printed_sensitivity_db"],
                        "scan_rank": rank,
                        "bscan_index": int(mapping["scan_index"]),
                        "bscan_path": str(mapping["bscan_path"].relative_to(PROJECT_ROOT)),
                        "patch_path": str(patch_path.relative_to(PROJECT_ROOT)),
                        "bscan_scale_x_mm_per_px": float(geometry["bscan_metadata"]["Scale X"]),
                        "bscan_scale_z_mm_per_px": float(geometry["bscan_metadata"]["Scale Z"]),
                        "along_scan_direction_convention": (
                            "slo_coordinate_start_to_end_maps_to_bscan_left_to_right"
                        ),
                        **patch_audit,
                    }
                )
            point["excessive_nearest_distance_pilot_rule"] = (
                float(point["scan_1_distance_spacing_units"]) > 0.55
            )
            point["excessive_third_distance_pilot_rule"] = (
                float(point["scan_3_distance_spacing_units"]) > 1.55
            )
            point["any_projection_clipped"] = any(
                bool(point[f"scan_{rank}_projection_clipped"]) for rank in (1, 2, 3)
            )
            point["bscan_width_px"] = bscan_width
            point["bscan_height_px"] = bscan_height
            point["bscan_scale_x_mm_per_px"] = float(geometry["bscan_metadata"]["Scale X"])
            point["bscan_scale_z_mm_per_px"] = float(geometry["bscan_metadata"]["Scale Z"])

        draw_qc_overlays(args.output_root, manifest, case_points, geometry)
        nearest_distances = np.asarray([float(point["scan_1_distance_slo_px"]) for point in case_points])
        third_distances = np.asarray([float(point["scan_3_distance_slo_px"]) for point in case_points])
        audit = case_audit_by_id[case_id]
        audit.update(transform_audit)
        audit.update(
            {
                "slo_width_px": slo_width,
                "slo_height_px": slo_height,
                "scan_area_bbox_px": list(geometry["area_bbox"]),
                "bscan_width_px": bscan_width,
                "bscan_height_px": bscan_height,
                "median_scan_spacing_px": median_spacing,
                "scan_spacing_min_px": geometry["scan_spacing_min_px"],
                "scan_spacing_max_px": geometry["scan_spacing_max_px"],
                "points_outside_native_slo": sum(not bool(point["inside_native_slo"]) for point in case_points),
                "points_outside_scan_area": sum(not bool(point["inside_scan_area"]) for point in case_points),
                "points_with_projection_clipping": sum(bool(point["any_projection_clipped"]) for point in case_points),
                "points_with_missing_neighbouring_scans": sum(bool(point["missing_neighbouring_scans"]) for point in case_points),
                "points_with_noncontiguous_three_scan_indices": sum(
                    not bool(point["three_scan_indices_contiguous"]) for point in case_points
                ),
                "points_excessive_nearest_distance": sum(
                    bool(point["excessive_nearest_distance_pilot_rule"]) for point in case_points
                ),
                "points_excessive_third_distance": sum(
                    bool(point["excessive_third_distance_pilot_rule"]) for point in case_points
                ),
                "nearest_distance_median_px": float(np.median(nearest_distances)),
                "nearest_distance_max_px": float(nearest_distances.max()),
                "third_distance_median_px": float(np.median(third_distances)),
                "third_distance_max_px": float(third_distances.max()),
                "scan_lines_with_start_x_greater_than_end_x": sum(
                    float(line["start"][0]) > float(line["end"][0])
                    for line in geometry["lines"]
                ),
                "patch_count": sum(row["case_id"] == case_id for row in patch_rows),
                "patches_with_edge_padding": sum(
                    row["case_id"] == case_id and bool(row["edge_padding_applied"])
                    for row in patch_rows
                ),
                "patches_with_excessive_edge_padding": sum(
                    row["case_id"] == case_id and bool(row["excessive_edge_padding"])
                    for row in patch_rows
                ),
            }
        )

        source_paths = {
            "maia_clean": resolve_working_path(manifest["maia_clean_path"]),
            "maia_overlay": resolve_working_path(manifest["maia_overlay_path"]),
            "registration_transform": PROJECT_ROOT / str(transform_audit["transform_file"]),
            "registration_result_metadata": PROJECT_ROOT / str(transform_audit["registration_result_file"]),
            "oct_slo": geometry["paths"]["slo"],
            "oct_scan_area": geometry["paths"]["scan_area"],
            "oct_scan_coordinates": geometry["paths"]["scan_coordinates"],
            "oct_slo_metadata": geometry["paths"]["slo_metadata"],
            "oct_bscan_metadata": geometry["paths"]["bscan_metadata"],
            "oct_scan_positions_image": geometry["paths"]["scan_positions_image"],
            "oct_volume": geometry["paths"]["volume"],
        }
        for source_name, path in source_paths.items():
            source_audit_rows.append(
                {
                    "case_id": case_id,
                    "source_name": source_name,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "exists": path.exists(),
                    "is_directory": path.is_dir(),
                    "note": "generated privacy-minimised source/output; raw data not modified",
                }
            )

    pilot_label_rows = [
        row
        for row in label_rows
        if any(
            row["study_id"] == manifest_by_case[case_id]["study_id"]
            and row["eye"] == manifest_by_case[case_id]["eye"]
            and row["timepoint"] == manifest_by_case[case_id]["maia_timepoint"]
            for case_id in case_order
        )
    ]
    write_csv(
        args.output_root / "pilot_workbook_labels_points_0_37.csv",
        pilot_label_rows,
        ["study_id", "eye", "timepoint", "point_number", "sensitivity_db", "floor_flag", "analysis_role"],
    )

    extraction_fields = [
        "case_id", "case_role", "study_id", "eye", "maia_timepoint", "point_number", "location_id",
        "ring", "angle_index", "maia_x_px", "maia_y_px", "marker_score",
        "printed_sensitivity_db", "marker_palette_value_db", "workbook_sensitivity_db",
        "printed_matches_workbook", "value_recovery_method",
        "ocr_candidate_db", "ocr_score", "ocr_margin", "ocr_second_value", "ocr_allowed_values",
        "recovered_value_template_score", "label_window_x0", "label_window_y0",
        "label_window_x1", "label_window_y1", "label_template_x", "label_template_y",
        "marker_r", "marker_g", "marker_b",
        "oct_slo_x_px", "oct_slo_y_px", "inside_native_slo", "inside_scan_area",
        "missing_neighbouring_scans", "three_scan_indices_contiguous",
        *[
            field
            for rank in (1, 2, 3)
            for field in (
                f"scan_{rank}_index", f"scan_{rank}_along_fraction", f"scan_{rank}_along_x_px",
                f"scan_{rank}_distance_slo_px", f"scan_{rank}_distance_spacing_units",
                f"scan_{rank}_projected_x_px", f"scan_{rank}_projected_y_px",
                f"scan_{rank}_projection_clipped", f"scan_{rank}_bscan_path",
            )
        ],
        "excessive_nearest_distance_pilot_rule", "excessive_third_distance_pilot_rule",
        "any_projection_clipped", "bscan_width_px", "bscan_height_px",
        "bscan_scale_x_mm_per_px", "bscan_scale_z_mm_per_px",
    ]
    write_csv(args.output_root / "maia_point_extraction_audit.csv", extraction_rows, extraction_fields)
    case_fields = [
        "case_id", "case_role", "study_id", "eye", "maia_timepoint", "oct_visit", "registration_source",
        "final_transform", "visual_grade", "n_bscans", "n_coordinate_rows", "point0_workbook_db",
        "centre_printed_sensitivity_db",
        "recovered_marker_count", "printed_value_multiset_matches_points_1_37",
        "minimum_ocr_score", "minimum_ocr_margin", "native_image_size_px",
        "difference_pixels_ge_20", "round_marker_candidates", "clustered_round_marker_candidates",
        "grid_centre_px", "outer_radius_px", "base_angle_degrees", "ring_counts",
        "minimum_marker_score", "centre_marker_score",
        "transform_kind", "transform_source", "transform_file", "transform_direction",
        "moving_native_size_px", "fixed_native_size_px", "moving_to_512_scale",
        "fixed_512_to_native_scale", "registration_result_file", "slo_width_px", "slo_height_px",
        "scan_area_bbox_px", "bscan_width_px", "bscan_height_px", "median_scan_spacing_px",
        "scan_spacing_min_px", "scan_spacing_max_px", "points_outside_native_slo",
        "points_outside_scan_area", "points_with_projection_clipping",
        "points_with_missing_neighbouring_scans", "points_with_noncontiguous_three_scan_indices",
        "points_excessive_nearest_distance", "points_excessive_third_distance",
        "nearest_distance_median_px", "nearest_distance_max_px",
        "third_distance_median_px", "third_distance_max_px",
        "scan_lines_with_start_x_greater_than_end_x", "patch_count",
        "patches_with_edge_padding", "patches_with_excessive_edge_padding",
    ]
    write_csv(args.output_root / "pilot_case_audit.csv", case_audits, case_fields)
    write_csv(
        args.output_root / "point_scan_mapping_long.csv",
        scan_mapping_rows,
        [
            "case_id", "study_id", "eye", "maia_timepoint", "point_number", "sensitivity_db",
            "scan_rank", "bscan_index", "bscan_path", "line_start_x_px", "line_start_y_px",
            "line_end_x_px", "line_end_y_px", "projected_x_px", "projected_y_px",
            "along_fraction", "along_x_bscan_px", "point_to_line_distance_slo_px",
            "distance_in_median_spacing_units", "projection_clipped_to_segment",
        ],
    )
    write_csv(
        args.output_root / "patch_manifest.csv",
        patch_rows,
        [
            "case_id", "study_id", "eye", "maia_timepoint", "point_number",
            "sensitivity_db", "scan_rank", "bscan_index", "bscan_path", "patch_path",
            "bscan_width_px", "bscan_height_px", "bscan_scale_x_mm_per_px",
            "bscan_scale_z_mm_per_px", "along_scan_direction_convention",
            "source_centre_x_px", "source_centre_index_px", "source_patch_x0_px",
            "source_patch_x1_exclusive_px", "source_patch_width_px",
            "requested_patch_width_mm", "realised_patch_width_mm", "pad_left_px",
            "pad_right_px", "padded_fraction", "edge_padding_applied",
            "excessive_edge_padding", "padding_mode", "output_width_px",
            "output_height_px", "output_sha256",
        ],
    )
    write_csv(
        args.output_root / "pilot_source_audit.csv",
        source_audit_rows,
        ["case_id", "source_name", "path", "exists", "is_directory", "note"],
    )
    (args.output_root / "point_number_mapping.json").write_text(
        json.dumps(
            {
                "method": (
                    "minimum-mismatch assignment of six-case printed-value signatures to workbook "
                    "Points 1-37, fitted separately by eye because fundus display orientation differs"
                ),
                "point_0_handling": "kept separate; not used in spatial assignment",
                "pilot_case_ids": case_order,
                "alignment_patch_pilot_case_ids": list(ALIGNMENT_PILOT_CASE_IDS),
                "maia_numbering_calibration_only_case_ids": list(NUMBERING_CALIBRATION_CASE_IDS),
                "location_to_point_number_by_eye": point_by_eye_location,
                "mapping_mismatches_by_eye": mapping_mismatches_by_eye,
                "point_signature_audit_by_eye": point_signature_audit_by_eye,
                "visual_value_overrides": [
                    {"case_id": case_id, "location_id": location_id, "printed_sensitivity_db": value}
                    for (case_id, location_id), value in VISUAL_VALUE_OVERRIDES.items()
                ],
                "total_printed_workbook_mismatches": total_mismatches,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_root / "patch_rule_frozen_pilot.json").write_text(
        json.dumps(
            {
                "status": "frozen_after_six_case_geometry_verification_before_model_performance",
                "selection": "three_nearest_bscan_line_segments_from_registered_geometry_only",
                "horizontal_width_mm": PATCH_WIDTH_MM,
                "output_size_px": [PATCH_OUTPUT_WIDTH_PX, PATCH_OUTPUT_HEIGHT_PX],
                "vertical_rule": (
                    "retain_full_496_pixel_axial_depth; all pilot Scale Z values are "
                    "0.00387167 mm/pixel"
                ),
                "horizontal_resampling": "bilinear_to_128_pixels_after_physical_width_crop",
                "source_width_rule": "round(0.75_mm / bscan_Scale_X_mm_per_pixel)",
                "centering_rule": "round(projected_along_scan_x_to_nearest_source_pixel)",
                "padding_rule": "reflect_horizontal_only_when_crop_crosses_a_bscan_edge",
                "edge_flag_rule": "flag_any_padding; excessive_if_more_than_25_percent_of_width",
                "scan_direction_convention": (
                    "slo_coordinates start endpoint maps to B-scan left edge and end endpoint "
                    "maps to B-scan right edge"
                ),
                "pilot_direction_check": (
                    "all coordinate rows run left-to-right; left/right-eye B-scans and SLO "
                    "anatomy were visually checked for consistency"
                ),
                "point_0_handling": "excluded_from_patch_extraction_and retained separately",
                "model_performance_inspected_before_freeze": False,
                "pilot_case_ids": list(ALIGNMENT_PILOT_CASE_IDS),
                "maia_numbering_calibration_only_case_ids": list(NUMBERING_CALIBRATION_CASE_IDS),
                "n_pilot_points": len(ALIGNMENT_PILOT_CASE_IDS) * 37,
                "n_pilot_patches": len(patch_rows),
                "n_patches_with_edge_padding": sum(bool(row["edge_padding_applied"]) for row in patch_rows),
                "n_patches_with_excessive_edge_padding": sum(
                    bool(row["excessive_edge_padding"]) for row in patch_rows
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Pilot MAIA extraction written to {args.output_root}")
    print(f"Printed/workbook mismatches after assignment: {total_mismatches} of {len(extraction_rows)}")
    print(
        "Multiset matches by case:",
        sum(bool(row["printed_value_multiset_matches_points_1_37"]) for row in case_audits),
        "of",
        len(case_audits),
    )


if __name__ == "__main__":
    main()
