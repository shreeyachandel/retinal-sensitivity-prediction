# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Apply the frozen post-registration point pipeline to the full cohort.

The script reuses the verified pilot geometry, point-number mapping, transform
handling, and patch rule. It writes a new cohort output directory and never
modifies raw data, the frozen manifest, registration results, or pilot outputs.

This run prepares automatic QC and a visual-review queue. It does not declare
the point dataset frozen; human review and exception decisions remain a gate.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image
import SimpleITK as sitk

from post_registration_point_alignment_pilot import (
    DECISIONS,
    DEFAULT_LABELS,
    GridPoint,
    MANIFEST,
    MANUAL_SUMMARY,
    PROJECT_ROOT,
    RECONCILIATION,
    RGB_TO_VALUE,
    VISUAL_VALUE_OVERRIDES,
    WORKBOOK,
    draw_qc_overlays,
    disk_score,
    extract_grid,
    extract_patch,
    oct_geometry,
    ocr_values,
    project_to_segment,
    read_csv,
    refine_marker,
    resolve_working_path,
    transform_points_to_oct,
    write_csv,
)


PILOT_ROOT = PROJECT_ROOT / "outputs" / "post_registration_point_alignment_pilot"
POINT_MAPPING = PILOT_ROOT / "point_number_mapping.json"
PATCH_RULE = PILOT_ROOT / "patch_rule_frozen_pilot.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "point_alignment_cohort"

NEAREST_DISTANCE_LIMIT = 0.55
THIRD_DISTANCE_LIMIT = 1.55

POINT_FIELDS = [
    "case_id", "study_id", "eye", "maia_timepoint", "oct_visit",
    "registration_source", "final_transform", "visual_grade", "point_number",
    "location_id", "ring", "angle_index", "maia_x_px", "maia_y_px",
    "marker_score", "grid_location_reconstructed", "workbook_sensitivity_db", "floor_flag",
    "printed_sensitivity_db", "marker_palette_value_db", "printed_value_recovered",
    "printed_matches_workbook", "value_recovery_method", "ocr_candidate_db",
    "ocr_score", "ocr_margin", "ocr_second_value", "ocr_allowed_values",
    "recovered_value_template_score", "label_window_x0", "label_window_y0",
    "label_window_x1", "label_window_y1", "label_template_x", "label_template_y",
    "marker_r", "marker_g", "marker_b", "oct_slo_x_px", "oct_slo_y_px",
    "inside_native_slo", "inside_scan_area", "missing_neighbouring_scans",
    "three_scan_indices_contiguous",
    *[
        field
        for rank in (1, 2, 3)
        for field in (
            f"scan_{rank}_index", f"scan_{rank}_along_fraction",
            f"scan_{rank}_along_x_px", f"scan_{rank}_distance_slo_px",
            f"scan_{rank}_distance_spacing_units", f"scan_{rank}_projected_x_px",
            f"scan_{rank}_projected_y_px", f"scan_{rank}_projection_clipped",
            f"scan_{rank}_bscan_path",
        )
    ],
    "excessive_nearest_distance_frozen_rule", "excessive_third_distance_frozen_rule",
    "any_projection_clipped", "bscan_width_px", "bscan_height_px",
    "bscan_scale_x_mm_per_px", "bscan_scale_z_mm_per_px",
]

CASE_FIELDS = [
    "case_id", "study_id", "eye", "maia_timepoint", "oct_visit",
    "registration_source", "final_transform", "visual_grade", "n_bscans",
    "n_coordinate_rows", "processing_status", "failure_stage", "failure_message",
    "registration_output_path_manifest", "registration_output_path_effective",
    "registration_path_resolution",
    "point0_workbook_db", "centre_printed_sensitivity_db", "recovered_marker_count",
    "printed_value_multiset_recovery_ok", "printed_value_unreadable_count",
    "printed_workbook_mismatch_count", "minimum_ocr_score", "minimum_ocr_margin",
    "native_image_size_px", "difference_pixels_ge_20", "round_marker_candidates",
    "clustered_round_marker_candidates", "grid_centre_px", "outer_radius_px",
    "base_angle_degrees", "ring_counts", "grid_recovery_mode",
    "reconstructed_grid_location_count", "reconstructed_grid_locations",
    "minimum_marker_score", "centre_marker_score",
    "transform_kind", "transform_source", "transform_file", "transform_direction",
    "moving_native_size_px", "fixed_native_size_px", "moving_to_512_scale",
    "fixed_512_to_native_scale", "registration_result_file", "slo_width_px",
    "slo_height_px", "scan_area_bbox_px", "bscan_width_px", "bscan_height_px",
    "median_scan_spacing_px", "scan_spacing_min_px", "scan_spacing_max_px",
    "points_outside_native_slo", "points_outside_scan_area",
    "points_with_projection_clipping", "points_with_missing_neighbouring_scans",
    "points_with_noncontiguous_three_scan_indices", "points_excessive_nearest_distance",
    "points_excessive_third_distance", "nearest_distance_median_px",
    "nearest_distance_max_px", "third_distance_median_px", "third_distance_max_px",
    "scan_lines_with_start_x_greater_than_end_x", "point_count", "patch_count",
    "patches_with_edge_padding", "patches_with_excessive_edge_padding",
    "automatic_qc_issue_count", "requires_visual_review",
]

SCAN_MAPPING_FIELDS = [
    "case_id", "study_id", "eye", "maia_timepoint", "point_number",
    "sensitivity_db", "scan_rank", "bscan_index", "bscan_path",
    "line_start_x_px", "line_start_y_px", "line_end_x_px", "line_end_y_px",
    "projected_x_px", "projected_y_px", "along_fraction", "along_x_bscan_px",
    "point_to_line_distance_slo_px", "distance_in_median_spacing_units",
    "projection_clipped_to_segment",
]

PATCH_FIELDS = [
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
]

SOURCE_FIELDS = ["case_id", "source_name", "path", "exists", "is_directory", "note"]
EXCEPTION_FIELDS = [
    "case_id", "study_id", "eye", "maia_timepoint", "point_number", "scan_rank",
    "issue_type", "severity", "observed_value", "threshold_or_expectation",
    "automatic_action", "human_decision", "decision_reason",
]


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def source_row(case_id: str, name: str, path: Path, note: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "source_name": name,
        "path": rel(path),
        "exists": path.exists(),
        "is_directory": path.is_dir(),
        "note": note,
    }


def expected_case_sources(
    manifest: dict[str, str], decision: dict[str, str] | None
) -> list[tuple[str, Path]]:
    clean = resolve_working_path(manifest["maia_clean_path"])
    overlay = resolve_working_path(manifest["maia_overlay_path"])
    slo = resolve_working_path(manifest["oct_slo_path"])
    oct_dir = slo.parent
    registration_dir = PROJECT_ROOT / manifest["registration_output_path"]
    transform_kind = decision["final_transform"] if decision else manifest["provisional_transform"]
    if decision:
        transform = registration_dir / f"{transform_kind}_moving_to_fixed_512.tfm"
        result = registration_dir / "automatic_registration_result.json"
    else:
        transform = registration_dir / f"transform_{transform_kind}_maia_to_oct.tfm"
        result = registration_dir / "registration_result_simpleitk.json"
    return [
        ("maia_clean", clean),
        ("maia_overlay", overlay),
        ("registration_transform", transform),
        ("registration_result_metadata", result),
        ("oct_slo", slo),
        ("oct_scan_area", oct_dir / "slo_area.csv"),
        ("oct_scan_coordinates", resolve_working_path(manifest["scan_coordinates_path"])),
        ("oct_slo_metadata", oct_dir / "slo_metadata.csv"),
        ("oct_bscan_metadata", oct_dir / "bscan_metadata.csv"),
        ("oct_scan_positions_image", oct_dir / "slo_scan_positions.png"),
        ("oct_volume", resolve_working_path(manifest["oct_volume_path"])),
    ]


def add_exception(
    rows: list[dict[str, object]], manifest: dict[str, str], issue_type: str,
    severity: str, observed: object, expected: object, point_number: object = "",
    scan_rank: object = "", automatic_action: str = "flag_for_visual_review",
) -> None:
    rows.append(
        {
            "case_id": manifest["case_id"],
            "study_id": manifest["study_id"],
            "eye": manifest["eye"],
            "maia_timepoint": manifest["maia_timepoint"],
            "point_number": point_number,
            "scan_rank": scan_rank,
            "issue_type": issue_type,
            "severity": severity,
            "observed_value": observed,
            "threshold_or_expectation": expected,
            "automatic_action": automatic_action,
            "human_decision": "pending",
            "decision_reason": "",
        }
    )


def disk_offsets(radius: int) -> list[tuple[int, int]]:
    return [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]


def fit_regular_grid_geometry(candidates: np.ndarray) -> tuple[float, float, float]:
    """Fit the known three-ring MAIA geometry with a robust bounded search."""
    initial_x, initial_y = np.median(candidates, axis=0)

    def score(cx: float, cy: float, radius: float) -> float:
        distances = np.linalg.norm(candidates - np.asarray([cx, cy]), axis=1)
        residuals = np.min(
            np.abs(distances[:, None] - radius * np.asarray([0.2, 0.6, 1.0])[None, :]),
            axis=1,
        )
        retained = np.sort(np.minimum(residuals, 12.0))[: max(20, int(len(residuals) * 0.9))]
        return float(np.mean(retained**2))

    best = (float("inf"), float(initial_x), float(initial_y), 140.0)
    for cx in np.arange(initial_x - 30.0, initial_x + 30.1, 2.0):
        for cy in np.arange(initial_y - 30.0, initial_y + 30.1, 2.0):
            for radius in np.arange(132.0, 148.1, 1.0):
                candidate = (score(cx, cy, radius), float(cx), float(cy), float(radius))
                if candidate < best:
                    best = candidate
    _, coarse_x, coarse_y, coarse_radius = best
    for cx in np.arange(coarse_x - 2.0, coarse_x + 2.01, 0.5):
        for cy in np.arange(coarse_y - 2.0, coarse_y + 2.01, 0.5):
            for radius in np.arange(coarse_radius - 1.0, coarse_radius + 1.01, 0.25):
                candidate = (score(cx, cy, radius), float(cx), float(cy), float(radius))
                if candidate < best:
                    best = candidate
    _, centre_x, centre_y, outer_radius = best
    return centre_x, centre_y, outer_radius


def extract_grid_tolerant(
    clean_path: Path, overlay_path: Path,
) -> tuple[list[GridPoint], dict[str, object], np.ndarray]:
    """Recover the regular 37-point grid when one or more markers are obscured.

    A circle is fitted to the visible outer-ring candidates. The regular
    12-angle, three-radius geometry is then reconstructed. Locations without a
    nearby connected marker candidate retain their fitted geometric position
    and are explicitly flagged for visual review.
    """
    clean = np.asarray(Image.open(clean_path).convert("RGB"), dtype=np.int16)
    overlay = np.asarray(Image.open(overlay_path).convert("RGB"), dtype=np.int16)
    if clean.shape != overlay.shape:
        raise ValueError(f"Clean/overlay shape mismatch: {clean.shape} versus {overlay.shape}")
    difference = np.max(np.abs(overlay - clean), axis=2)
    marker_mask = difference >= 40
    eroded = sitk.BinaryErode(
        sitk.GetImageFromArray(marker_mask.astype(np.uint8)), [4, 4], sitk.sitkBall
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
        raise ValueError(
            f"Only {len(small_centres)} round-marker candidates survived tolerant extraction"
        )

    candidates = np.asarray(small_centres, dtype=float)
    median = np.median(candidates, axis=0)
    clustered = candidates[np.linalg.norm(candidates - median, axis=1) <= 190.0]
    centre_x, centre_y, outer_radius = fit_regular_grid_geometry(clustered)
    centre = np.asarray([centre_x, centre_y], dtype=float)
    if not 132.0 <= outer_radius <= 148.0:
        raise ValueError(f"Implausible fitted outer radius: {outer_radius:.3f} px")

    step = math.tau / 12.0
    candidate_distances = np.linalg.norm(clustered - centre, axis=1)
    radial_residuals = np.min(
        np.abs(
            candidate_distances[:, None]
            - outer_radius * np.asarray([0.2, 0.6, 1.0])[None, :]
        ),
        axis=1,
    )
    phase_candidates = clustered[radial_residuals <= 7.0]
    if len(phase_candidates) < 8:
        raise ValueError(f"Only {len(phase_candidates)} candidates support the fitted grid phase")
    candidate_angles = np.arctan2(
        phase_candidates[:, 1] - centre_y, phase_candidates[:, 0] - centre_x
    )
    phase = float(np.angle(np.mean(np.exp(1j * 12.0 * candidate_angles))) / 12.0)
    angle_options = [phase + index * step for index in range(12)]
    base_angle = min(
        angle_options,
        key=lambda angle: abs(math.atan2(math.sin(angle + math.pi / 2), math.cos(angle + math.pi / 2))),
    )

    grid: list[GridPoint] = []
    reconstructed: list[str] = []
    for ring, scale in (("outer", 1.0), ("middle", 0.6), ("inner", 0.2)):
        radius = outer_radius * scale
        for angle_index in range(12):
            angle = base_angle + angle_index * step
            expected_x = centre_x + radius * math.cos(angle)
            expected_y = centre_y + radius * math.sin(angle)
            location_id = f"{ring}_{angle_index:02d}"
            nearest_candidate_distance = float(
                np.min(np.linalg.norm(clustered - np.asarray([expected_x, expected_y]), axis=1))
            )
            if nearest_candidate_distance <= 9.0:
                x, y, score = refine_marker(marker_mask, expected_x, expected_y)
            else:
                x, y = float(expected_x), float(expected_y)
                score = disk_score(marker_mask, int(round(x)), int(round(y)))
                reconstructed.append(location_id)
            grid.append(GridPoint(location_id, ring, angle_index, x, y, score))

    centre_score = disk_score(marker_mask, int(round(centre_x)), int(round(centre_y)))
    grid.append(GridPoint("centre", "centre", None, centre_x, centre_y, centre_score))
    if len({(round(point.x, 3), round(point.y, 3)) for point in grid}) != 37:
        raise ValueError("Tolerant grid recovery contains duplicate marker centres")
    audit = {
        "native_image_size_px": [int(clean.shape[1]), int(clean.shape[0])],
        "difference_pixels_ge_20": int((difference >= 20).sum()),
        "round_marker_candidates": len(small_centres),
        "clustered_round_marker_candidates": len(clustered),
        "grid_centre_px": [float(centre_x), float(centre_y)],
        "outer_radius_px": outer_radius,
        "base_angle_degrees": math.degrees(base_angle),
        "ring_counts": {"outer": 12, "middle": 12, "inner": 12, "centre": 1},
        "grid_recovery_mode": "tolerant_regular_grid_fit",
        "reconstructed_grid_location_count": len(reconstructed),
        "reconstructed_grid_locations": "|".join(reconstructed),
        "minimum_marker_score": min(point.marker_score for point in grid if point.ring != "centre"),
        "centre_marker_score": centre_score,
    }
    return grid, audit, difference


def extract_grid_for_cohort(
    clean_path: Path, overlay_path: Path,
) -> tuple[list[GridPoint], dict[str, object], np.ndarray]:
    try:
        grid, audit, difference = extract_grid(clean_path, overlay_path)
        audit.update(
            {
                "grid_recovery_mode": "standard_pilot_method",
                "reconstructed_grid_location_count": 0,
                "reconstructed_grid_locations": "",
            }
        )
        return grid, audit, difference
    except ValueError as error:
        if "Expected 12 outer-ring markers" not in str(error):
            raise
        return extract_grid_tolerant(clean_path, overlay_path)


def recover_case_points(
    manifest: dict[str, str], labels: dict[int, int],
    location_to_point: dict[str, int], registration_source: str,
    final_transform: str, visual_grade: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    clean_path = resolve_working_path(manifest["maia_clean_path"])
    overlay_path = resolve_working_path(manifest["maia_overlay_path"])
    grid, grid_audit, difference = extract_grid_for_cohort(clean_path, overlay_path)
    overlay = np.asarray(Image.open(overlay_path).convert("RGB"), dtype=np.uint8)
    ocr_by_location = {
        row["location_id"]: row
        for row in ocr_values(difference, grid, range(-1, 37), overlay)
    }

    recovered: dict[str, int | None] = {}
    recovery_method: dict[str, str] = {}
    palette_value_by_location: dict[str, int | None] = {}
    ring_values: list[int] = []
    unreadable_ring_count = 0
    for point in grid:
        if point.ring == "centre":
            continue
        ocr = ocr_by_location[point.location_id]
        rgb = (int(ocr["marker_r"]), int(ocr["marker_g"]), int(ocr["marker_b"]))
        palette_value = RGB_TO_VALUE.get(rgb)
        palette_value_by_location[point.location_id] = palette_value
        if (manifest["case_id"], point.location_id) in VISUAL_VALUE_OVERRIDES:
            value = VISUAL_VALUE_OVERRIDES[(manifest["case_id"], point.location_id)]
            recovered[point.location_id] = value
            recovery_method[point.location_id] = (
                "frozen_enlarged_difference_crop_text_override_due_fixation_overpaint"
            )
            ring_values.append(value)
        elif palette_value is None:
            recovered[point.location_id] = None
            recovery_method[point.location_id] = "unreadable_overlay_marker_palette"
            unreadable_ring_count += 1
        else:
            recovered[point.location_id] = palette_value
            recovery_method[point.location_id] = (
                "overlay_marker_palette_with_numeric_text_template_check"
            )
            ring_values.append(palette_value)

    endpoint_values = [labels[number] for number in range(1, 38)]
    remaining = Counter(endpoint_values) - Counter(ring_values)
    excess = Counter(ring_values) - Counter(endpoint_values)
    centre_recovery_ok = (
        unreadable_ring_count == 0 and not excess and sum(remaining.values()) == 1
    )
    centre_value = next(remaining.elements()) if centre_recovery_ok else None
    recovered["centre"] = centre_value
    palette_value_by_location["centre"] = None
    recovery_method["centre"] = (
        "centre_numeric_text_checked_after_ring_multiset_residual"
        if centre_recovery_ok
        else "centre_not_recoverable_after_ring_multiset_failure"
    )

    points: list[dict[str, object]] = []
    for point in grid:
        point_number = int(location_to_point[point.location_id])
        workbook_value = int(labels[point_number])
        printed_value = recovered[point.location_id]
        ocr = dict(ocr_by_location[point.location_id])
        scores = json.loads(str(ocr.pop("ocr_scores_json")))
        points.append(
            {
                "case_id": manifest["case_id"],
                "study_id": manifest["study_id"],
                "eye": manifest["eye"],
                "maia_timepoint": manifest["maia_timepoint"],
                "oct_visit": manifest["oct_visit"],
                "registration_source": registration_source,
                "final_transform": final_transform,
                "visual_grade": visual_grade,
                "point_number": point_number,
                "location_id": point.location_id,
                "ring": point.ring,
                "angle_index": "" if point.angle_index is None else point.angle_index,
                "maia_x_px": point.x,
                "maia_y_px": point.y,
                "marker_score": point.marker_score,
                "grid_location_reconstructed": point.location_id in set(
                    str(grid_audit["reconstructed_grid_locations"]).split("|")
                ),
                "workbook_sensitivity_db": workbook_value,
                "floor_flag": workbook_value == -1,
                "printed_sensitivity_db": "" if printed_value is None else printed_value,
                "marker_palette_value_db": (
                    "" if palette_value_by_location[point.location_id] is None
                    else palette_value_by_location[point.location_id]
                ),
                "printed_value_recovered": printed_value is not None,
                "printed_matches_workbook": (
                    "" if printed_value is None else int(printed_value) == workbook_value
                ),
                "value_recovery_method": recovery_method[point.location_id],
                "recovered_value_template_score": (
                    "" if printed_value is None else scores.get(str(printed_value), 0.0)
                ),
                **ocr,
            }
        )
    points.sort(key=lambda row: int(row["point_number"]))
    if [int(row["point_number"]) for row in points] != list(range(1, 38)):
        raise ValueError(f"Point-number map did not yield exactly Points 1-37 for {manifest['case_id']}")

    unreadable_count = sum(not bool(row["printed_value_recovered"]) for row in points)
    mismatch_count = sum(
        bool(row["printed_value_recovered"]) and not bool(row["printed_matches_workbook"])
        for row in points
    )
    audit = {
        "point0_workbook_db": labels[0],
        "centre_printed_sensitivity_db": "" if centre_value is None else centre_value,
        "recovered_marker_count": len(grid),
        "printed_value_multiset_recovery_ok": centre_recovery_ok,
        "printed_value_unreadable_count": unreadable_count,
        "printed_workbook_mismatch_count": mismatch_count,
        "minimum_ocr_score": min(float(row["ocr_score"]) for row in points),
        "minimum_ocr_margin": min(float(row["ocr_margin"]) for row in points),
        **grid_audit,
    }
    return points, audit


def process_geometry_and_patches(
    output_root: Path, manifest: dict[str, str], decision: dict[str, str] | None,
    points: list[dict[str, object]], scan_rows: list[dict[str, object]],
    patch_rows: list[dict[str, object]], exceptions: list[dict[str, object]],
    write_patches: bool,
) -> dict[str, object]:
    transformed, transform_audit = transform_points_to_oct(manifest, decision, points)
    geometry = oct_geometry(manifest)
    slo_width, slo_height = geometry["slo_size"]
    area_x0, area_y0, area_x1, area_y1 = geometry["area_bbox"]
    bscan_width, bscan_height = geometry["bscan_size"]
    median_spacing = float(geometry["median_scan_spacing_px"])

    for point, (oct_x, oct_y) in zip(points, transformed):
        projections = []
        for line in geometry["lines"]:
            projections.append(
                {**line, **project_to_segment((oct_x, oct_y), line["start"], line["end"])}
            )
        projections.sort(key=lambda row: (float(row["distance_px"]), int(row["scan_index"])))
        nearest = projections[:3]
        point["oct_slo_x_px"] = oct_x
        point["oct_slo_y_px"] = oct_y
        point["inside_native_slo"] = 0.0 <= oct_x < slo_width and 0.0 <= oct_y < slo_height
        point["inside_scan_area"] = area_x0 <= oct_x <= area_x1 and area_y0 <= oct_y <= area_y1
        point["missing_neighbouring_scans"] = len(nearest) < 3
        indices = sorted(int(row["scan_index"]) for row in nearest)
        point["three_scan_indices_contiguous"] = (
            len(indices) == 3
            and indices[1] == indices[0] + 1
            and indices[2] == indices[1] + 1
        )

        if not point["inside_native_slo"]:
            add_exception(exceptions, manifest, "point_outside_native_slo", "high",
                          f"({oct_x:.3f},{oct_y:.3f})", f"0<=x<{slo_width}; 0<=y<{slo_height}",
                          point["point_number"])
        if not point["inside_scan_area"]:
            add_exception(exceptions, manifest, "point_outside_scan_area", "high",
                          f"({oct_x:.3f},{oct_y:.3f})", str(geometry["area_bbox"]),
                          point["point_number"])
        if point["missing_neighbouring_scans"]:
            add_exception(exceptions, manifest, "missing_three_nearest_scans", "high",
                          len(nearest), 3, point["point_number"])
        if not point["three_scan_indices_contiguous"]:
            add_exception(exceptions, manifest, "noncontiguous_three_nearest_scans", "medium",
                          "|".join(str(index) for index in indices), "three consecutive indices",
                          point["point_number"])

        for rank, mapping in enumerate(nearest, start=1):
            along_x = float(mapping["t"]) * (bscan_width - 1)
            point[f"scan_{rank}_index"] = int(mapping["scan_index"])
            point[f"scan_{rank}_along_fraction"] = float(mapping["t"])
            point[f"scan_{rank}_along_x_px"] = along_x
            point[f"scan_{rank}_distance_slo_px"] = float(mapping["distance_px"])
            point[f"scan_{rank}_distance_spacing_units"] = (
                float(mapping["distance_px"]) / median_spacing
            )
            point[f"scan_{rank}_projected_x_px"] = float(mapping["projected_x"])
            point[f"scan_{rank}_projected_y_px"] = float(mapping["projected_y"])
            point[f"scan_{rank}_projection_clipped"] = bool(
                mapping["projection_clipped_to_segment"]
            )
            point[f"scan_{rank}_bscan_path"] = rel(mapping["bscan_path"])
            scan_rows.append(
                {
                    "case_id": manifest["case_id"],
                    "study_id": manifest["study_id"],
                    "eye": manifest["eye"],
                    "maia_timepoint": manifest["maia_timepoint"],
                    "point_number": point["point_number"],
                    "sensitivity_db": point["workbook_sensitivity_db"],
                    "scan_rank": rank,
                    "bscan_index": int(mapping["scan_index"]),
                    "bscan_path": rel(mapping["bscan_path"]),
                    "line_start_x_px": mapping["start"][0],
                    "line_start_y_px": mapping["start"][1],
                    "line_end_x_px": mapping["end"][0],
                    "line_end_y_px": mapping["end"][1],
                    "projected_x_px": mapping["projected_x"],
                    "projected_y_px": mapping["projected_y"],
                    "along_fraction": mapping["t"],
                    "along_x_bscan_px": along_x,
                    "point_to_line_distance_slo_px": mapping["distance_px"],
                    "distance_in_median_spacing_units": (
                        float(mapping["distance_px"]) / median_spacing
                    ),
                    "projection_clipped_to_segment": mapping["projection_clipped_to_segment"],
                }
            )
            if mapping["projection_clipped_to_segment"]:
                add_exception(exceptions, manifest, "projection_clipped_to_scan_segment", "high",
                              mapping["t_raw"], "0<=raw_fraction<=1", point["point_number"], rank)

            if write_patches:
                patch_path = (
                    output_root / "patches" / manifest["case_id"]
                    / f"point_{int(point['point_number']):03d}"
                    / f"rank_{rank}_bscan_{int(mapping['scan_index']):03d}.png"
                )
                patch_audit = extract_patch(
                    mapping["bscan_path"], along_x,
                    float(geometry["bscan_metadata"]["Scale X"]), patch_path,
                )
                patch_row = {
                    "case_id": manifest["case_id"],
                    "study_id": manifest["study_id"],
                    "eye": manifest["eye"],
                    "maia_timepoint": manifest["maia_timepoint"],
                    "point_number": point["point_number"],
                    "sensitivity_db": point["workbook_sensitivity_db"],
                    "scan_rank": rank,
                    "bscan_index": int(mapping["scan_index"]),
                    "bscan_path": rel(mapping["bscan_path"]),
                    "patch_path": rel(patch_path),
                    "bscan_scale_x_mm_per_px": float(
                        geometry["bscan_metadata"]["Scale X"]
                    ),
                    "bscan_scale_z_mm_per_px": float(
                        geometry["bscan_metadata"]["Scale Z"]
                    ),
                    "along_scan_direction_convention": (
                        "slo_coordinate_start_to_end_maps_to_bscan_left_to_right"
                    ),
                    **patch_audit,
                }
                patch_rows.append(patch_row)
                if patch_row["edge_padding_applied"]:
                    add_exception(exceptions, manifest, "patch_edge_padding", "medium",
                                  patch_row["padded_fraction"], "0 preferred",
                                  point["point_number"], rank)
                if patch_row["excessive_edge_padding"]:
                    add_exception(exceptions, manifest, "excessive_patch_edge_padding", "high",
                                  patch_row["padded_fraction"], "<=0.25",
                                  point["point_number"], rank)

        point["excessive_nearest_distance_frozen_rule"] = (
            float(point["scan_1_distance_spacing_units"]) > NEAREST_DISTANCE_LIMIT
        )
        point["excessive_third_distance_frozen_rule"] = (
            float(point["scan_3_distance_spacing_units"]) > THIRD_DISTANCE_LIMIT
        )
        point["any_projection_clipped"] = any(
            bool(point[f"scan_{rank}_projection_clipped"]) for rank in (1, 2, 3)
        )
        point["bscan_width_px"] = bscan_width
        point["bscan_height_px"] = bscan_height
        point["bscan_scale_x_mm_per_px"] = float(geometry["bscan_metadata"]["Scale X"])
        point["bscan_scale_z_mm_per_px"] = float(geometry["bscan_metadata"]["Scale Z"])
        if point["excessive_nearest_distance_frozen_rule"]:
            add_exception(exceptions, manifest, "excessive_nearest_scan_distance", "high",
                          point["scan_1_distance_spacing_units"],
                          f"<={NEAREST_DISTANCE_LIMIT} median spacings", point["point_number"], 1)
        if point["excessive_third_distance_frozen_rule"]:
            add_exception(exceptions, manifest, "excessive_third_scan_distance", "high",
                          point["scan_3_distance_spacing_units"],
                          f"<={THIRD_DISTANCE_LIMIT} median spacings", point["point_number"], 3)

    draw_qc_overlays(output_root, manifest, points, geometry)
    nearest_distances = np.asarray([float(point["scan_1_distance_slo_px"]) for point in points])
    third_distances = np.asarray([float(point["scan_3_distance_slo_px"]) for point in points])
    case_patch_rows = [row for row in patch_rows if row["case_id"] == manifest["case_id"]]
    return {
        **transform_audit,
        "slo_width_px": slo_width,
        "slo_height_px": slo_height,
        "scan_area_bbox_px": list(geometry["area_bbox"]),
        "bscan_width_px": bscan_width,
        "bscan_height_px": bscan_height,
        "median_scan_spacing_px": median_spacing,
        "scan_spacing_min_px": geometry["scan_spacing_min_px"],
        "scan_spacing_max_px": geometry["scan_spacing_max_px"],
        "points_outside_native_slo": sum(not bool(p["inside_native_slo"]) for p in points),
        "points_outside_scan_area": sum(not bool(p["inside_scan_area"]) for p in points),
        "points_with_projection_clipping": sum(bool(p["any_projection_clipped"]) for p in points),
        "points_with_missing_neighbouring_scans": sum(bool(p["missing_neighbouring_scans"]) for p in points),
        "points_with_noncontiguous_three_scan_indices": sum(
            not bool(p["three_scan_indices_contiguous"]) for p in points
        ),
        "points_excessive_nearest_distance": sum(
            bool(p["excessive_nearest_distance_frozen_rule"]) for p in points
        ),
        "points_excessive_third_distance": sum(
            bool(p["excessive_third_distance_frozen_rule"]) for p in points
        ),
        "nearest_distance_median_px": float(np.median(nearest_distances)),
        "nearest_distance_max_px": float(nearest_distances.max()),
        "third_distance_median_px": float(np.median(third_distances)),
        "third_distance_max_px": float(third_distances.max()),
        "scan_lines_with_start_x_greater_than_end_x": sum(
            float(line["start"][0]) > float(line["end"][0]) for line in geometry["lines"]
        ),
        "point_count": len(points),
        "patch_count": len(case_patch_rows),
        "patches_with_edge_padding": sum(bool(row["edge_padding_applied"]) for row in case_patch_rows),
        "patches_with_excessive_edge_padding": sum(
            bool(row["excessive_edge_padding"]) for row in case_patch_rows
        ),
    }


def build_visual_review_queue(
    case_audits: list[dict[str, object]], exceptions: list[dict[str, object]]
) -> list[dict[str, object]]:
    reasons_by_case: dict[str, set[str]] = defaultdict(set)
    for row in exceptions:
        reasons_by_case[str(row["case_id"])].add(str(row["issue_type"]))
    queue: list[dict[str, object]] = []
    queued = set()
    for audit in sorted(case_audits, key=lambda row: str(row["case_id"])):
        case_id = str(audit["case_id"])
        if reasons_by_case.get(case_id):
            queue.append(
                {
                    "case_id": case_id,
                    "study_id": audit["study_id"],
                    "eye": audit["eye"],
                    "maia_timepoint": audit["maia_timepoint"],
                    "registration_source": audit["registration_source"],
                    "final_transform": audit["final_transform"],
                    "visual_grade": audit["visual_grade"],
                    "slo_width_px": audit.get("slo_width_px", ""),
                    "review_reason": "automatic_qc_flagged",
                    "issue_types": "|".join(sorted(reasons_by_case[case_id])),
                    "maia_overlay_path": (
                        f"outputs/point_alignment_cohort/qc_overlays/"
                        f"{case_id}__maia_points_1_37.png"
                    ),
                    "oct_overlay_path": (
                        f"outputs/point_alignment_cohort/qc_overlays/"
                        f"{case_id}__oct_three_nearest_scans.png"
                    ),
                    "human_decision": "pending",
                    "decision_reason": "",
                }
            )
            queued.add(case_id)

    strata_seen = set()
    sample_count = 0
    for audit in sorted(case_audits, key=lambda row: str(row["case_id"])):
        case_id = str(audit["case_id"])
        if case_id in queued or audit.get("processing_status") != "completed":
            continue
        stratum = (
            audit["eye"], audit["maia_timepoint"], audit["registration_source"],
            audit["final_transform"], audit.get("slo_width_px", ""),
        )
        if stratum in strata_seen:
            continue
        strata_seen.add(stratum)
        queue.append(
            {
                "case_id": case_id,
                "study_id": audit["study_id"],
                "eye": audit["eye"],
                "maia_timepoint": audit["maia_timepoint"],
                "registration_source": audit["registration_source"],
                "final_transform": audit["final_transform"],
                "visual_grade": audit["visual_grade"],
                "slo_width_px": audit.get("slo_width_px", ""),
                "review_reason": "deterministic_unflagged_sample",
                "issue_types": "",
                "maia_overlay_path": (
                    f"outputs/point_alignment_cohort/qc_overlays/"
                    f"{case_id}__maia_points_1_37.png"
                ),
                "oct_overlay_path": (
                    f"outputs/point_alignment_cohort/qc_overlays/"
                    f"{case_id}__oct_three_nearest_scans.png"
                ),
                "human_decision": "pending",
                "decision_reason": "",
            }
        )
        sample_count += 1
        if sample_count >= 12:
            break
    return queue


def write_report(
    output_root: Path, manifests: list[dict[str, str]], case_audits: list[dict[str, object]],
    points: list[dict[str, object]], patches: list[dict[str, object]],
    exceptions: list[dict[str, object]], queue: list[dict[str, object]], write_patches: bool,
) -> None:
    completed = [row for row in case_audits if row["processing_status"] == "completed"]
    failed = [row for row in case_audits if row["processing_status"] != "completed"]
    issue_counts = Counter(str(row["issue_type"]) for row in exceptions)
    flagged_cases = len({str(row["case_id"]) for row in exceptions})
    report = [
        "# Full-cohort point-alignment automatic QC run",
        "",
        f"**Run date:** {date.today().isoformat()}",
        "**Status:** Not frozen. Automatic processing is complete only to the human visual-QC gate.",
        "",
        "## Scope",
        "",
        f"- Frozen-manifest eye-visits requested: {len(manifests)}.",
        f"- Eye-visits completed: {len(completed)}.",
        f"- Eye-visits with processing failure: {len(failed)}.",
        f"- Point rows produced: {len(points)}.",
        f"- Patch rows produced: {len(patches)}{' (patch writing disabled)' if not write_patches else ''}.",
        f"- Cases with one or more automatic QC flags: {flagged_cases}.",
        f"- Cases queued for visual review: {len(queue)}.",
        "",
        "## Frozen inputs and rules",
        "",
        "- Cohort membership and file pairing come from the frozen registration manifest.",
        "- Point numbering comes from the pilot point-number mapping, separately by eye.",
        "- Workbook sensitivity is the analysis label; printed MAIA values are an independent QC field.",
        "- Three B-scans are selected by registered geometry alone.",
        "- Patch dimensions, resampling, and padding follow the frozen pilot patch rule.",
        "- Point 0 is written separately and excluded from patch extraction.",
        "",
        "## Automatic QC issue counts",
        "",
    ]
    if issue_counts:
        report.extend(f"- {name}: {count}" for name, count in sorted(issue_counts.items()))
    else:
        report.append("- No automatic QC exceptions were detected.")
    report.extend(
        [
            "",
            "## Required human gate",
            "",
            "1. Review every case in `visual_review_queue.csv` using both QC overlays.",
            "2. Record accept, correct, or exclude and a reason for every queued case.",
            "3. Resolve every row in `qc_exceptions.csv`.",
            "4. Verify any corrections by rerunning the affected case under a versioned configuration.",
            "5. Freeze the accepted point table and patch manifest only after those decisions are complete.",
            "",
            "No feature extraction or modelling is authorised from this provisional run.",
            "",
        ]
    )
    output_root.joinpath("COHORT_RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--no-patches", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifests = read_csv(MANIFEST)
    if args.case_id:
        requested = set(args.case_id)
        manifests = [row for row in manifests if row["case_id"] in requested]
        missing_case_ids = requested - {row["case_id"] for row in manifests}
        if missing_case_ids:
            raise ValueError(f"Requested case IDs are absent from the frozen manifest: {sorted(missing_case_ids)}")
    if args.case_limit is not None:
        manifests = manifests[: args.case_limit]
    if not manifests:
        raise ValueError("No manifest rows selected")
    decisions_by_case = {row["case_id"]: row for row in read_csv(DECISIONS)}
    labels_by_case: dict[tuple[str, str, str], dict[int, int]] = {}
    label_rows = read_csv(args.labels_csv)
    for row in label_rows:
        key = (row["study_id"], row["eye"], row["timepoint"])
        labels_by_case.setdefault(key, {})[int(row["point_number"])] = int(row["sensitivity_db"])

    mapping_document = json.loads(POINT_MAPPING.read_text(encoding="utf-8"))
    location_map = mapping_document["location_to_point_number_by_eye"]
    if set(location_map) != {"R", "L"}:
        raise ValueError("Frozen point-number map must contain R and L mappings")
    for eye in ("R", "L"):
        if sorted(int(value) for value in location_map[eye].values()) != list(range(1, 38)):
            raise ValueError(f"Frozen {eye}-eye point-number map is not a bijection over Points 1-37")

    source_rows = [
        source_row("GLOBAL", name, path, note)
        for name, path, note in (
            ("frozen_registration_manifest", MANIFEST, "cohort and pairing authority"),
            ("manifest_reconciliation", RECONCILIATION, "audit-only reconciliation reference"),
            ("automatic_visual_decisions", DECISIONS, "final automatic transform and QC authority"),
            ("manual_registration_summary", MANUAL_SUMMARY, "accepted manual-reference audit"),
            ("deidentified_maia_workbook", WORKBOOK, "authoritative sensitivity source"),
            ("workbook_pointwise_label_export", args.labels_csv, "deterministic workbook export"),
            ("frozen_point_number_mapping", POINT_MAPPING, "pilot-verified point numbering"),
            ("frozen_patch_rule", PATCH_RULE, "pilot-verified patch specification"),
        )
    ]
    point_rows: list[dict[str, object]] = []
    scan_rows: list[dict[str, object]] = []
    patch_rows: list[dict[str, object]] = []
    case_audits: list[dict[str, object]] = []
    exceptions: list[dict[str, object]] = []
    point0_rows: list[dict[str, object]] = []

    for position, manifest in enumerate(manifests, start=1):
        case_id = manifest["case_id"]
        decision = decisions_by_case.get(case_id)
        effective_manifest = dict(manifest)
        registration_path_resolution = "frozen_manifest"
        if decision and not manifest["registration_output_path"]:
            fallback_path = PROJECT_ROOT / "outputs" / "automatic_registration_batch" / case_id
            effective_manifest["registration_output_path"] = rel(fallback_path)
            registration_path_resolution = "deterministic_automatic_batch_case_directory_fallback"
        registration_source = "automatic" if decision else "manual_reference"
        final_transform = decision["final_transform"] if decision else manifest["provisional_transform"]
        visual_grade = decision["visual_grade"] if decision else "manual_reference_accepted"
        audit: dict[str, object] = {
            "case_id": case_id,
            "study_id": manifest["study_id"],
            "eye": manifest["eye"],
            "maia_timepoint": manifest["maia_timepoint"],
            "oct_visit": manifest["oct_visit"],
            "registration_source": registration_source,
            "final_transform": final_transform,
            "visual_grade": visual_grade,
            "n_bscans": int(manifest["n_bscans"]),
            "n_coordinate_rows": int(manifest["n_coordinate_rows"]),
            "processing_status": "started",
            "failure_stage": "",
            "failure_message": "",
            "registration_output_path_manifest": manifest["registration_output_path"],
            "registration_output_path_effective": effective_manifest["registration_output_path"],
            "registration_path_resolution": registration_path_resolution,
        }
        case_audits.append(audit)
        if registration_path_resolution != "frozen_manifest":
            add_exception(
                exceptions, manifest, "registration_output_path_missing_in_frozen_manifest",
                "medium", "blank", effective_manifest["registration_output_path"],
                automatic_action="resolved_deterministically_and_flagged_for_audit",
            )
        for source_name, path in expected_case_sources(effective_manifest, decision):
            source_rows.append(
                source_row(case_id, source_name, path,
                           "privacy-minimised source or accepted registration output; raw data unchanged")
            )
            if not path.exists():
                add_exception(exceptions, manifest, "missing_required_source", "critical",
                              rel(path), "source exists", automatic_action="processing_failure")

        stage = "point_recovery"
        try:
            key = (manifest["study_id"], manifest["eye"], manifest["maia_timepoint"])
            labels = labels_by_case[key]
            if sorted(labels) != list(range(0, 38)):
                raise ValueError(f"Workbook export does not contain exactly Points 0-37 for {case_id}")
            points, recovery_audit = recover_case_points(
                manifest, labels, location_map[manifest["eye"]], registration_source,
                final_transform, visual_grade,
            )
            audit.update(recovery_audit)
            point0_rows.append(
                {
                    "case_id": case_id,
                    "study_id": manifest["study_id"],
                    "eye": manifest["eye"],
                    "maia_timepoint": manifest["maia_timepoint"],
                    "point_number": 0,
                    "sensitivity_db": labels[0],
                    "floor_flag": labels[0] == -1,
                    "analysis_role": "separate_not_primary_endpoint",
                }
            )
            for point in points:
                if point["grid_location_reconstructed"]:
                    add_exception(exceptions, manifest, "grid_location_reconstructed", "medium",
                                  point["location_id"], "directly detected marker centre",
                                  point["point_number"])
                if not point["printed_value_recovered"]:
                    add_exception(exceptions, manifest, "printed_value_unreadable", "medium", "",
                                  point["workbook_sensitivity_db"], point["point_number"])
                elif not point["printed_matches_workbook"]:
                    add_exception(exceptions, manifest, "printed_workbook_mismatch", "high",
                                  point["printed_sensitivity_db"], point["workbook_sensitivity_db"],
                                  point["point_number"])

            stage = "geometry_mapping_and_patch_extraction"
            geometry_audit = process_geometry_and_patches(
                args.output_root, effective_manifest, decision, points, scan_rows, patch_rows,
                exceptions, not args.no_patches,
            )
            audit.update(geometry_audit)
            point_rows.extend(points)
            if visual_grade == "borderline":
                add_exception(exceptions, manifest, "borderline_registration_qc", "medium",
                              visual_grade, "manual review required")
            if int(audit["scan_lines_with_start_x_greater_than_end_x"]) > 0:
                add_exception(exceptions, manifest, "scan_direction_start_x_greater_than_end_x",
                              "high", audit["scan_lines_with_start_x_greater_than_end_x"], 0)
            audit["processing_status"] = "completed"
        except Exception as error:  # continue the batch and expose every failed case
            audit["processing_status"] = "failed"
            audit["failure_stage"] = stage
            audit["failure_message"] = f"{type(error).__name__}: {error}"
            add_exception(exceptions, manifest, "case_processing_failure", "critical",
                          audit["failure_message"], "successful processing",
                          automatic_action="exclude_from_provisional_point_table_pending_resolution")
        print(f"[{position:02d}/{len(manifests):02d}] {case_id}: {audit['processing_status']}", flush=True)

    issue_count_by_case = Counter(str(row["case_id"]) for row in exceptions)
    for audit in case_audits:
        audit["automatic_qc_issue_count"] = issue_count_by_case[str(audit["case_id"])]
        audit["requires_visual_review"] = (
            int(audit["automatic_qc_issue_count"]) > 0
            or audit["processing_status"] != "completed"
        )

    queue = build_visual_review_queue(case_audits, exceptions)
    queue_fields = [
        "case_id", "study_id", "eye", "maia_timepoint", "registration_source",
        "final_transform", "visual_grade", "slo_width_px", "review_reason",
        "issue_types", "maia_overlay_path", "oct_overlay_path", "human_decision",
        "decision_reason",
    ]
    write_csv(args.output_root / "full_point_table_provisional.csv", point_rows, POINT_FIELDS)
    write_csv(args.output_root / "point_scan_mapping_long.csv", scan_rows, SCAN_MAPPING_FIELDS)
    write_csv(args.output_root / "patch_manifest_provisional.csv", patch_rows, PATCH_FIELDS)
    write_csv(args.output_root / "cohort_case_audit.csv", case_audits, CASE_FIELDS)
    write_csv(args.output_root / "source_audit.csv", source_rows, SOURCE_FIELDS)
    write_csv(args.output_root / "qc_exceptions.csv", exceptions, EXCEPTION_FIELDS)
    write_csv(args.output_root / "visual_review_queue.csv", queue, queue_fields)
    write_csv(
        args.output_root / "point0_labels_separate.csv", point0_rows,
        ["case_id", "study_id", "eye", "maia_timepoint", "point_number",
         "sensitivity_db", "floor_flag", "analysis_role"],
    )

    configuration = {
        "run_date": date.today().isoformat(),
        "status": "provisional_pending_human_visual_qc_not_for_modelling",
        "case_count_requested": len(manifests),
        "write_patches": not args.no_patches,
        "frozen_manifest": rel(MANIFEST),
        "frozen_manifest_sha256": sha256_file(MANIFEST),
        "workbook_label_export": rel(args.labels_csv),
        "workbook_label_export_sha256": sha256_file(args.labels_csv),
        "point_number_mapping": rel(POINT_MAPPING),
        "point_number_mapping_sha256": sha256_file(POINT_MAPPING),
        "patch_rule": rel(PATCH_RULE),
        "patch_rule_sha256": sha256_file(PATCH_RULE),
        "automatic_decisions": rel(DECISIONS),
        "automatic_decisions_sha256": sha256_file(DECISIONS),
        "nearest_distance_limit_median_spacings": NEAREST_DISTANCE_LIMIT,
        "third_distance_limit_median_spacings": THIRD_DISTANCE_LIMIT,
        "sensitivity_authority": "deidentified_workbook_export",
        "printed_maia_value_role": "independent_qc_only",
        "point_0_role": "separate_not_primary_endpoint",
        "scan_selection": "three_nearest_segments_from_registered_geometry_only",
        "participant_folds_created": False,
        "model_performance_inspected": False,
    }
    (args.output_root / "run_configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(POINT_MAPPING, args.output_root / "frozen_point_number_mapping_used.json")
    shutil.copy2(PATCH_RULE, args.output_root / "frozen_patch_rule_used.json")
    write_report(
        args.output_root, manifests, case_audits, point_rows, patch_rows,
        exceptions, queue, not args.no_patches,
    )

    print(f"Output: {args.output_root}")
    print(f"Completed cases: {sum(row['processing_status'] == 'completed' for row in case_audits)}")
    print(f"Failed cases: {sum(row['processing_status'] != 'completed' for row in case_audits)}")
    print(f"Points: {len(point_rows)}")
    print(f"Patches: {len(patch_rows)}")
    print(f"QC exceptions: {len(exceptions)}")
    print(f"Visual review queue: {len(queue)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
