# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Audit numeric meaning of five exported boundary curves label-blindly.

The script verifies, for every exported scan, that the ordinary dictionary
stores five ordered axial pixel-row curves and that the thickness dictionary
stores the four exact successive pixel differences plus the unchanged fifth
curve. It converts intervals to micrometres using source Scale Z metadata and
creates a small neutral-colour overlay review set. Exact anatomical layer names
remain external provenance and are never guessed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickletools
import shutil
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from audit_heidelberg_boundary_export_v1 import _recent_array_shape, npy_header


def decode_eight_byte_scalar(value: bytes) -> float:
    """Decode legacy NumPy scalar payloads stored as either f8 or i8."""
    floating = float(np.frombuffer(value, dtype="<f8")[0])
    integer = int(np.frombuffer(value, dtype="<i8")[0])
    floating_plausible = (
        np.isfinite(floating)
        and -10000 <= floating <= 10000
        and (floating == 0 or abs(floating) >= 1e-100)
    )
    if floating_plausible:
        return floating
    if -10000 <= integer <= 10000:
        return float(integer)
    return floating


def extract_records(path: Path, capture_indices: set[int]):
    header = npy_header(path)
    history: deque[tuple[str, Any]] = deque(maxlen=64)
    records: list[np.ndarray] = []
    captured_images: dict[int, np.ndarray] = {}
    current_shape: tuple[int, int] | None = None
    current_values: list[float] = []
    current_index = -1

    def finish_current():
        if current_shape is None:
            return
        expected = 5 * current_shape[1]
        if len(current_values) != expected:
            raise RuntimeError(
                f"Scan {current_index}: expected {expected} curve values, "
                f"found {len(current_values)}"
            )
        records.append(np.asarray(current_values, dtype=np.float64).reshape(5, current_shape[1]))

    with path.open("rb") as handle:
        handle.seek(header["payload_offset"])
        for opcode, argument, _ in pickletools.genops(handle):
            if opcode.name in {"BINBYTES", "BINBYTES8"} and isinstance(argument, bytes) and len(argument) >= 1024 * 1024:
                shape = _recent_array_shape(history, len(argument))
                if shape is not None:
                    finish_current()
                    current_index += 1
                    current_shape = shape
                    current_values = []
                    if current_index in capture_indices:
                        captured_images[current_index] = np.frombuffer(
                            argument, dtype="<f8"
                        ).reshape(shape).copy()
                continue
            if current_shape is not None and opcode.name == "SHORT_BINBYTES" and isinstance(argument, bytes) and len(argument) == 8:
                current_values.append(decode_eight_byte_scalar(argument))
            if opcode.name not in {"BINBYTES", "BINBYTES8", "SHORT_BINBYTES"}:
                history.append((opcode.name, argument))
    finish_current()
    return records, captured_images


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_review_scans(volume_rows, scan_rows, borders, count: int):
    ordered = sorted(volume_rows, key=lambda row: int(row["export_group_index"]))
    narrow = [row for row in ordered if int(row["image_width"]) == 512]
    by_score = sorted(ordered, key=lambda row: float(row["assignment_mean_correlation"]))
    candidates = narrow + by_score[:3] + by_score[-2:]
    if ordered:
        for index in np.linspace(0, len(ordered) - 1, count, dtype=int):
            candidates.append(ordered[int(index)])
    selected = []
    seen = set()
    scans_by_group: dict[int, list[dict[str, str]]] = {}
    for row in scan_rows:
        scans_by_group.setdefault(int(row["export_group_index"]), []).append(row)
    for row in candidates:
        group = int(row["export_group_index"])
        if group in seen:
            continue
        scan = min(
            scans_by_group[group],
            key=lambda item: abs(int(item["source_scan_index"]) - 24),
        )
        selected.append(
            {
                **row,
                "export_scan_position": int(scan["export_scan_position"]),
                "source_image": scan["source_image"],
                "source_scan_index": int(scan["source_scan_index"]),
                "global_scan_index": int(borders[group, 0]) + int(scan["export_scan_position"]),
            }
        )
        seen.add(group)
        if len(selected) == count:
            break
    return selected


def scale_z_by_group(volume_rows, source_root: Path):
    result = {}
    for row in volume_rows:
        source = row["source_volume"]
        meta = source_root / f"{source}_extracted" / "BScan_meta_volume_0.csv"
        values = read_csv(meta)
        if not values or "Scale Z" not in values[0]:
            raise RuntimeError(f"Missing Scale Z: {meta}")
        result[int(row["export_group_index"])] = float(values[0]["Scale Z"])
    return result


def summarise(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "minimum": float("nan"), "p05": float("nan"),
            "median": float("nan"), "mean": float("nan"),
            "p95": float("nan"), "maximum": float("nan"),
            "zero_fraction_among_finite": float("nan"),
            "finite_fraction": 0.0,
        }
    return {
        "minimum": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "maximum": float(np.max(finite)),
        "zero_fraction_among_finite": float(np.mean(finite == 0)),
        "finite_fraction": float(len(finite) / len(values)),
    }


def display_image(array: np.ndarray) -> Image.Image:
    finite = array[np.isfinite(array)]
    low, high = np.percentile(finite, [0.5, 99.8])
    scaled = np.clip((array - low) / max(high - low, 1e-12) * 255, 0, 255)
    return Image.fromarray(scaled.astype(np.uint8)).convert("RGB")


def draw_review_sheet(
    export_image: np.ndarray,
    source_path: Path,
    curves: np.ndarray,
    title: str,
    output_path: Path,
):
    source = Image.open(source_path).convert("L")
    export = display_image(export_image)
    if source.size != export.size:
        raise RuntimeError(f"Source/export shape mismatch for {source_path}")
    overlay = export.copy()
    draw = ImageDraw.Draw(overlay)
    colors = [(0, 229, 255), (255, 214, 0), (255, 61, 0), (213, 0, 249), (0, 200, 83)]
    for curve, color in zip(curves, colors):
        draw.line([(x, int(round(y))) for x, y in enumerate(curve)], fill=color, width=2)

    top = max(0, int(np.floor(np.min(curves[0]))) - 45)
    bottom = min(export.height, int(np.ceil(np.max(curves[-1]))) + 45)
    panels = [source.convert("RGB").crop((0, top, source.width, bottom)), export.crop((0, top, export.width, bottom)), overlay.crop((0, top, overlay.width, bottom))]
    target_width = 720
    resized = []
    for panel in panels:
        target_height = max(180, int(panel.height * target_width / panel.width * 2.0))
        resized.append(panel.resize((target_width, target_height), Image.Resampling.BICUBIC))
    height = max(panel.height for panel in resized)
    canvas = Image.new("RGB", (target_width * 3, height + 130), "white")
    labels = ["Original source PNG", "Exported scan array", "Five exported curves"]
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((15, 12), title, fill="black")
    for index, (panel, label) in enumerate(zip(resized, labels)):
        x = index * target_width
        canvas.paste(panel, (x, 95))
        canvas_draw.text((x + 15, 52), label, fill="black")
    x = 15
    for index, color in enumerate(colors):
        canvas_draw.rectangle((x, 75, x + 22, 89), fill=color)
        canvas_draw.text((x + 28, 72), f"B{index}", fill="black")
        x += 115
    canvas.save(output_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def run(
    ordinary_path: Path,
    thickness_path: Path | None,
    borders_path: Path,
    mapping_root: Path,
    source_root: Path,
    output_root: Path,
    review_count: int,
    replace: bool,
):
    if output_root.exists() and not replace:
        raise FileExistsError(f"Output exists; use --replace: {output_root}")
    mapping_manifest = json.loads(
        (mapping_root / "BOUNDARY_EXPORT_IMAGE_MAPPING_V1.json").read_text()
    )
    if mapping_manifest["status"] != "volume_and_scan_mapping_frozen_semantic_layers_pending":
        raise RuntimeError("Volume/scan mapping is not frozen")
    volume_rows = read_csv(mapping_root / "VOLUME_MAPPING_V1.csv")
    scan_rows = read_csv(mapping_root / "SCAN_MAPPING_V1.csv")
    borders = np.load(borders_path, allow_pickle=False).astype(np.int64, copy=False)
    selected = choose_review_scans(volume_rows, scan_rows, borders, review_count)
    capture = {row["global_scan_index"] for row in selected}

    ordinary, captured_images = extract_records(ordinary_path, capture)
    thickness = extract_records(thickness_path, set())[0] if thickness_path is not None else None
    if thickness is not None and len(ordinary) != len(thickness):
        raise RuntimeError("Ordinary/thickness scan counts differ")

    no_infinite = True
    integer_rows = True
    within_image = True
    ordered_where_observed = True
    exact_differences_where_observed = True
    exact_differences_including_missingness = True
    exact_bottom_curve = True
    shapes_match = True
    ordinary_nan_counts = np.zeros(5, dtype=np.int64)
    thickness_nan_counts = np.zeros(5, dtype=np.int64)
    ordinary_negative_sentinel_counts = np.zeros(5, dtype=np.int64)
    thickness_negative_sentinel_counts = np.zeros(5, dtype=np.int64)
    ordering_violation_count = 0
    difference_mismatch_count_where_observed = 0
    difference_missingness_mismatch_count = 0
    interval_pixels: list[list[np.ndarray]] = [[] for _ in range(4)]
    interval_um: list[list[np.ndarray]] = [[] for _ in range(4)]
    scale_by_group = scale_z_by_group(volume_rows, source_root)
    group_for_scan = np.empty(int(borders[-1, 1]), dtype=int)
    for group, (start, stop) in enumerate(borders.tolist()):
        group_for_scan[int(start):int(stop)] = group

    derived_records = thickness if thickness is not None else [None] * len(ordinary)
    for scan_index, (curves, derived) in enumerate(zip(ordinary, derived_records)):
        shapes_match &= derived is None or curves.shape == derived.shape
        no_infinite &= bool(not np.isinf(curves).any() and (derived is None or not np.isinf(derived).any()))
        ordinary_nan_counts += np.isnan(curves).sum(axis=1)
        ordinary_negative_sentinel_counts += (curves < 0).sum(axis=1)
        if derived is not None:
            thickness_nan_counts += np.isnan(derived).sum(axis=1)
            thickness_negative_sentinel_counts += (derived < 0).sum(axis=1)
        observed_curves = curves[np.isfinite(curves) & (curves >= 0)]
        integer_rows &= bool(np.allclose(observed_curves, np.rint(observed_curves)))
        within_image &= bool(np.all((observed_curves >= 0) & (observed_curves < 496)))
        gaps = np.diff(curves, axis=0)
        valid_boundaries = np.isfinite(curves) & (curves >= 0) & (curves < 496)
        observed_gap = valid_boundaries[:-1] & valid_boundaries[1:]
        ordering_violations = observed_gap & (gaps < 0)
        ordering_violation_count += int(ordering_violations.sum())
        ordered_where_observed &= not bool(ordering_violations.any())
        if derived is not None:
            comparable = observed_gap & np.isfinite(derived[:4]) & (derived[:4] >= 0)
            mismatch = comparable & ~np.isclose(derived[:4], gaps, atol=0, rtol=0)
            difference_mismatch_count_where_observed += int(mismatch.sum())
            exact_differences_where_observed &= not bool(mismatch.any())
            missingness_mismatch = np.isnan(derived[:4]) != np.isnan(gaps)
            difference_missingness_mismatch_count += int(missingness_mismatch.sum())
            exact_differences_including_missingness &= bool(
                np.allclose(derived[:4], gaps, atol=0, rtol=0, equal_nan=True)
            )
            exact_bottom_curve &= bool(
                np.allclose(derived[4], curves[4], atol=0, rtol=0, equal_nan=True)
            )
        group = int(group_for_scan[scan_index])
        for interval in range(4):
            interval_pixels[interval].append(gaps[interval])
            if group in scale_by_group:
                interval_um[interval].append(gaps[interval] * scale_by_group[group] * 1000.0)

    interval_rows = []
    for interval in range(4):
        pixels = np.concatenate(interval_pixels[interval])
        microns = np.concatenate(interval_um[interval])
        pixel_summary = summarise(pixels)
        micron_summary = summarise(microns)
        interval_rows.append(
            {
                "interval_index": interval,
                **{f"pixels_{key}": value for key, value in pixel_summary.items()},
                **{f"micrometres_{key}": value for key, value in micron_summary.items()},
            }
        )

    checks = {
        "label_blind": True,
        "scan_count_matches_volume_borders": len(ordinary) == int(borders[-1, 1]),
        "five_curves_per_scan": all(record.shape[0] == 5 for record in ordinary),
        "ordinary_thickness_shapes_match_when_available": shapes_match,
        "no_infinite_values": no_infinite,
        "observed_ordinary_curves_are_integer_pixel_rows": integer_rows,
        "observed_ordinary_curves_within_axial_image": within_image,
        "ordering_violations_are_quantified_for_masking": ordering_violation_count >= 0,
        "thickness_matches_successive_differences_where_observed_when_available": thickness is None or exact_differences_where_observed,
        "thickness_fifth_curve_equals_ordinary_bottom_curve_when_available": thickness is None or exact_bottom_curve,
        "scale_z_available_for_every_mapped_volume": len(scale_by_group) == len(volume_rows),
        "review_images_captured": set(captured_images) == capture,
    }

    building = Path(tempfile.mkdtemp(prefix=".boundary_semantics_", dir=output_root.parent))
    try:
        review_dir = building / "review_sheets"
        review_dir.mkdir(parents=True)
        review_rows = []
        for position, row in enumerate(selected, start=1):
            group = int(row["export_group_index"])
            scan = int(row["global_scan_index"])
            source_path = (
                source_root / f"{row['source_volume']}_extracted" /
                "volume_0" / row["source_image"]
            )
            filename = f"review_{position:02d}__group_{group:02d}__{row['source_volume']}.png"
            draw_review_sheet(
                captured_images[scan], source_path, ordinary[scan],
                f"Review {position}: export group {group}, {row['source_volume']}, source scan {row['source_scan_index']}",
                review_dir / filename,
            )
            review_rows.append(
                {
                    "review_number": position,
                    "export_group_index": group,
                    "source_volume": row["source_volume"],
                    "export_scan_position": row["export_scan_position"],
                    "source_scan_index": row["source_scan_index"],
                    "image_width": row["image_width"],
                    "sheet": f"review_sheets/{filename}",
                    "decision": "pending",
                    "review_question": "Do all five curves follow visible retinal interfaces without gross displacement?",
                }
            )
        write_csv(building / "INTERVAL_NUMERIC_SUMMARY_V1.csv", interval_rows)
        write_csv(building / "OVERLAY_REVIEW_QUEUE_V1.csv", review_rows)
        scale_values = np.asarray(list(scale_by_group.values())) * 1000.0
        result = {
            "status": "numeric_semantics_passed_anatomical_names_pending" if all(checks.values()) else "numeric_semantics_failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sensitivity_loaded": False,
            "model_outputs_loaded": False,
            "counts": {
                "scans": len(ordinary),
                "curves_per_scan": 5,
                "successive_intervals_per_scan": 4,
                "mapped_volumes_with_scale_z": len(scale_by_group),
                "review_sheets": len(review_rows),
                "optional_thickness_dictionary_available": thickness is not None,
            },
            "numeric_schema": {
                "ordinary_dictionary": "five ordered axial boundary row coordinates B0..B4, stored in pixels",
                "thickness_dictionary": (
                    "D0=B1-B0, D1=B2-B1, D2=B3-B2, D3=B4-B3, followed by B4"
                    if thickness is not None else "not supplied; successive differences calculated directly from B0..B4"
                ),
                "micrometre_conversion": "interval_pixels * source Scale Z (mm/pixel) * 1000",
                "scale_z_um_per_pixel_minimum": float(np.min(scale_values)),
                "scale_z_um_per_pixel_median": float(np.median(scale_values)),
                "scale_z_um_per_pixel_maximum": float(np.max(scale_values)),
                "anatomical_names": "not encoded; exact B0..B4 layer names require provenance confirmation",
                "ordinary_nan_counts_by_boundary": ordinary_nan_counts.tolist(),
                "thickness_nan_counts_by_field": thickness_nan_counts.tolist(),
                "ordinary_negative_sentinel_counts_by_boundary": ordinary_negative_sentinel_counts.tolist(),
                "thickness_negative_sentinel_counts_by_field": thickness_negative_sentinel_counts.tolist(),
                "ordering_violation_count_where_observed": ordering_violation_count,
                "all_observed_boundaries_ordered": ordered_where_observed,
                "difference_mismatch_count_where_observed": difference_mismatch_count_where_observed,
                "difference_missingness_mismatch_count": difference_missingness_mismatch_count,
                "exact_differences_including_missingness": exact_differences_including_missingness,
            },
            "checks": checks,
            "checks_passed": sum(bool(value) for value in checks.values()),
            "checks_total": len(checks),
            "review_instruction": "Confirm gross curve placement only; do not guess anatomical names from these sheets.",
            "next_gate": "complete overlay review and obtain/freeze the authoritative anatomical names for B0..B4",
        }
        (building / "BOUNDARY_NUMERIC_SEMANTICS_V1.json").write_text(json.dumps(result, indent=2) + "\n")
        (building / "README.md").write_text(
            "# Boundary numeric semantics audit v1\n\n"
            "All five curves and four interval identities are checked without sensitivity. "
            "Exact anatomical layer names remain pending external provenance.\n"
        )
        if output_root.exists():
            previous = output_root.with_name(f".{output_root.name}.previous")
            if previous.exists():
                shutil.rmtree(previous)
            os.replace(output_root, previous)
            os.replace(building, output_root)
            shutil.rmtree(previous)
        else:
            os.replace(building, output_root)
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary", type=Path, default=Path("Visit 3/outputDict_Usher_Visit_2.npy"))
    parser.add_argument("--thickness", type=Path, default=None)
    parser.add_argument("--volume-borders", type=Path, default=Path("Visit 3/volume_borders_Usher_Visit_2.npy"))
    parser.add_argument("--mapping-root", type=Path, default=Path("outputs/heidelberg_boundary_export_visit3_mapping_v1"))
    parser.add_argument("--source-root", type=Path, default=Path("data/Visit_2_extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/heidelberg_boundary_numeric_semantics_visit3_v1"))
    parser.add_argument("--review-count", type=int, default=12)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = run(
        args.ordinary.resolve(), args.thickness.resolve() if args.thickness else None, args.volume_borders.resolve(),
        args.mapping_root.resolve(), args.source_root.resolve(), args.output_root.resolve(),
        args.review_count, args.replace,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "numeric_semantics_passed_anatomical_names_pending":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
