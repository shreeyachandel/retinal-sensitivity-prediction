# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Build label-blind boundary features over each 100 µm MAIA footprint.

The unit of modelling remains one MAIA point.  Each point retains its three
registered nearest B-scans and all native A-scan columns inside ±50 µm.  The
five verified Heidelberg curves are kept under neutral names B0..B4 until the
authoritative anatomical name mapping is supplied.  Invalid negative sentinels
and locally crossed boundary pairs are masked, never interpreted as thickness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_heidelberg_boundary_numeric_semantics_v1 import extract_records


INTERVALS = tuple(range(4))
POINT_SCAN_KEYS = ["case_id", "point_number", "scan_rank"]
POINT_KEYS = ["case_id", "point_number"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_scales_mm(source_root: Path, source_volume: str) -> tuple[float, float]:
    path = source_root / f"{source_volume}_extracted" / "BScan_meta_volume_0.csv"
    rows = read_csv(path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one metadata row: {path}")
    return float(rows[0]["Scale X"]), float(rows[0]["Scale Z"])


def summarise(values: np.ndarray, total_count: int, prefix: str) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    result: dict[str, float | int] = {
        f"{prefix}_valid_count": int(len(finite)),
        f"{prefix}_total_count": int(total_count),
        f"{prefix}_valid_fraction": float(len(finite) / total_count) if total_count else float("nan"),
    }
    if not len(finite):
        result.update(
            {
                f"{prefix}_mean_um": float("nan"),
                f"{prefix}_sd_um": float("nan"),
                f"{prefix}_median_um": float("nan"),
                f"{prefix}_p10_um": float("nan"),
                f"{prefix}_p90_um": float("nan"),
            }
        )
        return result
    result.update(
        {
            f"{prefix}_mean_um": float(np.mean(finite)),
            f"{prefix}_sd_um": float(np.std(finite, ddof=0)),
            f"{prefix}_median_um": float(np.median(finite)),
            f"{prefix}_p10_um": float(np.percentile(finite, 10)),
            f"{prefix}_p90_um": float(np.percentile(finite, 90)),
        }
    )
    return result


def run(
    ordinary_dictionary: Path,
    borders_path: Path,
    boundary_mapping_root: Path,
    private_crosswalk_root: Path,
    footprint_root: Path,
    source_root: Path,
    semantic_audit_root: Path,
    output_root: Path,
    replace: bool,
) -> dict[str, Any]:
    if output_root.exists() and not replace:
        raise FileExistsError(f"Output exists; use --replace: {output_root}")
    semantics = json.loads(
        (semantic_audit_root / "BOUNDARY_NUMERIC_SEMANTICS_V1.json").read_text(encoding="utf-8")
    )
    if semantics["status"] != "numeric_semantics_passed_anatomical_names_pending":
        raise RuntimeError("Boundary numeric semantics audit has not passed")
    footprint_report = json.loads((footprint_root / "FOOTPRINT_SAMPLING_V2.json").read_text(encoding="utf-8"))
    if footprint_report["status"] != "passed":
        raise RuntimeError("100 µm footprint sampling has not passed")
    crosswalk_report = json.loads(
        (private_crosswalk_root / "PRIVATE_CROSSWALK_AUDIT_V1.json").read_text(encoding="utf-8")
    )
    if crosswalk_report["status"] != "passed_private_crosswalk_frozen":
        raise RuntimeError("Private source/case crosswalk has not passed")

    borders = np.load(borders_path, allow_pickle=False).astype(np.int64, copy=False)
    curves, _ = extract_records(ordinary_dictionary, set())
    if len(curves) != int(borders[-1, 1]):
        raise RuntimeError("Boundary dictionary scan count changed")

    volume_map = {
        row["source_volume"]: int(row["export_group_index"])
        for row in read_csv(boundary_mapping_root / "VOLUME_MAPPING_V1.csv")
    }
    scan_map = {
        (row["source_volume"], int(row["source_scan_index"])): (
            int(row["export_scan_position"]), row["source_image"]
        )
        for row in read_csv(boundary_mapping_root / "SCAN_MAPPING_V1.csv")
    }
    crosswalk_rows = read_csv(private_crosswalk_root / "PRIVATE_VOLUME_CASE_CROSSWALK_V1.csv")
    # The crosswalk can contain accepted OCT cases whose corresponding
    # boundary export group is unavailable (for example, a 193-scan export
    # without its source-image/scan mapping).  They are not eligible for a
    # boundary-derived feature table.  Restrict the feature table to the
    # intersection of accepted cases and mapped boundary volumes, and record
    # the omission in the manifest rather than failing on a KeyError or
    # silently inventing a scan correspondence.
    case_to_source = {
        row["case_id"]: row["source_volume"]
        for row in crosswalk_rows
        if row["accepted_point_alignment_case"].lower() == "true"
        and row["source_volume"] in volume_map
    }
    accepted_cases_without_boundary_mapping = sorted(
        row["case_id"]
        for row in crosswalk_rows
        if row["accepted_point_alignment_case"].lower() == "true"
        and row["source_volume"] not in volume_map
    )
    source_scales = {
        source: source_scales_mm(source_root, source)
        for source in set(case_to_source.values())
    }

    usecols = [
        "case_id", "study_id", "eye", "maia_timepoint", "point_number",
        "scan_rank", "bscan_index", "ascan_x_px", "within_scan_sample_rank",
        "offset_from_registered_centre_um",
    ]
    samples = pd.read_csv(footprint_root / "along_scan_samples_v2.csv", usecols=usecols)
    samples = samples[samples["case_id"].isin(case_to_source)].copy()
    samples.sort_values(POINT_SCAN_KEYS + ["within_scan_sample_rank"], inplace=True)
    samples.reset_index(drop=True, inplace=True)
    expected_case_ids = set(case_to_source)
    observed_case_ids = set(samples["case_id"].unique())
    if observed_case_ids != expected_case_ids:
        raise RuntimeError("Footprint/crosswalk Visit 3 case sets differ")

    boundary_values = np.full((len(samples), 4), np.nan, dtype=np.float64)
    invalid_reasons = np.empty(len(samples), dtype=object)
    for output_index, row in enumerate(samples.itertuples(index=False)):
        source = case_to_source[row.case_id]
        group = volume_map[source]
        export_position, source_image = scan_map[(source, int(row.bscan_index))]
        global_scan = int(borders[group, 0]) + export_position
        scan_curves = curves[global_scan]
        x = int(row.ascan_x_px)
        if x < 0 or x >= scan_curves.shape[1]:
            invalid_reasons[output_index] = "ascan_out_of_bounds"
            continue
        values = scan_curves[:, x]
        samples.at[output_index, "source_volume"] = source
        samples.at[output_index, "source_image_path"] = str(
            source_root / f"{source}_extracted" / "volume_0" / source_image
        )
        scale_x_mm, scale_z_mm = source_scales[source]
        samples.at[output_index, "scale_x_mm_per_pixel"] = scale_x_mm
        samples.at[output_index, "scale_z_mm_per_pixel"] = scale_z_mm
        for boundary_index, value in enumerate(values):
            samples.at[output_index, f"B{boundary_index}_row_px"] = value
        valid_boundary = np.isfinite(values) & (values >= 0) & (values < 496)
        gaps = np.diff(values)
        valid_intervals = valid_boundary[:-1] & valid_boundary[1:] & (gaps >= 0)
        boundary_values[output_index, valid_intervals] = gaps[valid_intervals] * scale_z_mm * 1000.0
        if valid_intervals.all():
            invalid_reasons[output_index] = ""
        elif not valid_boundary.all():
            invalid_reasons[output_index] = "negative_or_missing_boundary_sentinel"
        else:
            invalid_reasons[output_index] = "locally_crossed_boundary_pair"

    for interval in INTERVALS:
        samples[f"B{interval}_to_B{interval + 1}_um"] = boundary_values[:, interval]
        samples[f"B{interval}_to_B{interval + 1}_valid"] = np.isfinite(boundary_values[:, interval])
    samples["boundary_invalid_reason"] = invalid_reasons
    point_scan_rows = []
    for keys, frame in samples.groupby(POINT_SCAN_KEYS, sort=True):
        first = frame.iloc[0]
        output = {
            "case_id": keys[0],
            "study_id": first["study_id"],
            "eye": first["eye"],
            "maia_timepoint": first["maia_timepoint"],
            "point_number": int(keys[1]),
            "scan_rank": int(keys[2]),
            "bscan_index": int(first["bscan_index"]),
            "within_scan_sample_count": len(frame),
        }
        for interval in INTERVALS:
            values = frame[f"B{interval}_to_B{interval + 1}_um"].to_numpy(float)
            output.update(summarise(values, len(frame), f"B{interval}_B{interval + 1}"))
        point_scan_rows.append(output)
    point_scans = pd.DataFrame(point_scan_rows)

    point_rows = []
    for keys, frame in samples.groupby(POINT_KEYS, sort=True):
        first = frame.iloc[0]
        output = {
            "case_id": keys[0],
            "study_id": first["study_id"],
            "eye": first["eye"],
            "maia_timepoint": first["maia_timepoint"],
            "point_number": int(keys[1]),
            "three_nearest_bscans": int(frame["scan_rank"].nunique()),
            "total_footprint_samples": len(frame),
        }
        point_scan_subset = point_scans[
            (point_scans["case_id"] == keys[0])
            & (point_scans["point_number"] == int(keys[1]))
        ]
        for interval in INTERVALS:
            prefix = f"B{interval}_B{interval + 1}"
            values = frame[f"B{interval}_to_B{interval + 1}_um"].to_numpy(float)
            output.update(summarise(values, len(frame), prefix))
            scan_means = point_scan_subset[f"{prefix}_mean_um"].to_numpy(float)
            finite_scan_means = scan_means[np.isfinite(scan_means)]
            output[f"{prefix}_between_scan_mean_sd_um"] = (
                float(np.std(finite_scan_means, ddof=0)) if len(finite_scan_means) else float("nan")
            )
        point_rows.append(output)
    points = pd.DataFrame(point_rows)

    expected_points = len(case_to_source) * 37
    expected_point_scans = expected_points * 3
    feature_columns = [column for column in points if column.startswith("B")]
    checks = {
        "label_blind": True,
        "semantic_audit_passed": semantics["checks_passed"] == semantics["checks_total"],
        "private_crosswalk_passed": crosswalk_report["checks_passed"] == crosswalk_report["checks_total"],
        "accepted_boundary_cases_present": len(case_to_source) > 0,
        "point_rows_match_case_count_times_37": len(points) == expected_points,
        "point_scan_rows_match_points_times_3": len(point_scans) == expected_point_scans,
        "three_nearest_scans_per_point": bool((points["three_nearest_bscans"] == 3).all()),
        "multiple_along_scan_samples": int(point_scans["within_scan_sample_count"].min()) >= 2,
        "no_negative_valid_thicknesses": bool(np.nanmin(boundary_values) >= 0),
        "no_sensitivity_loaded_or_saved": not any("sensitivity" in column.lower() for column in [*samples.columns, *point_scans.columns, *points.columns]),
        "neutral_boundary_names_only": not any(name in " ".join(feature_columns).lower() for name in ("ez", "rpe", "ilm", "elm", "bruch")),
    }
    result = {
        "status": "calculated_pending_overlay_review_and_anatomical_name_mapping" if all(checks.values()) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sensitivity_loaded": False,
        "model_outputs_loaded": False,
        "anatomical_names_frozen": False,
        "feature_definition": "masked successive B0-B4 intervals across all native A-scans inside each 100 micrometre footprint and three nearest B-scans",
        "invalid_rule": "mask interval when either boundary is negative/non-finite/outside the image or the lower boundary is above the upper boundary",
        "counts": {
            "accepted_boundary_cases": len(case_to_source),
            "accepted_cases_without_boundary_mapping": len(accepted_cases_without_boundary_mapping),
            "points": len(points),
            "point_scans": len(point_scans),
            "along_scan_samples": len(samples),
            "interval_sample_values": int(boundary_values.size),
            "valid_interval_sample_values": int(np.isfinite(boundary_values).sum()),
            "masked_interval_sample_values": int(np.isnan(boundary_values).sum()),
        },
        "checks": checks,
        "accepted_cases_without_boundary_mapping": accepted_cases_without_boundary_mapping,
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "next_gate": "validate the neutral feature table and freeze the human-verified source boundaries without inventing anatomical names",
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=".visit3_boundary_features_", dir=output_root.parent))
    try:
        sample_path = building / "boundary_along_scan_samples_v1.csv"
        point_scan_path = building / "boundary_point_scan_features_v1.csv"
        point_path = building / "boundary_point_features_v1.csv"
        samples.to_csv(sample_path, index=False, lineterminator="\n")
        point_scans.to_csv(point_scan_path, index=False, lineterminator="\n")
        points.to_csv(point_path, index=False, lineterminator="\n")
        result["outputs"] = {
            sample_path.name: sha256(sample_path),
            point_scan_path.name: sha256(point_scan_path),
            point_path.name: sha256(point_path),
        }
        (building / "BOUNDARY_FOOTPRINT_FEATURES_V2.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (building / "README.md").write_text(
            "# Boundary footprint features v2\n\n"
            "Label-blind neutral B0-B4 interval features for all accepted mapped cases. "
            "The output is calculated but not yet frozen for modelling because overlay review "
            "and the authoritative anatomical name mapping remain pending.\n",
            encoding="utf-8",
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
    if result["status"] == "failed":
        raise RuntimeError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary", type=Path, default=Path("Visit 3/outputDict_Usher_Visit_2.npy"))
    parser.add_argument("--volume-borders", type=Path, default=Path("Visit 3/volume_borders_Usher_Visit_2.npy"))
    parser.add_argument("--boundary-mapping-root", type=Path, default=Path("outputs/heidelberg_boundary_export_visit3_mapping_v1"))
    parser.add_argument("--private-crosswalk-root", type=Path, default=Path("outputs/private_visit3_boundary_case_crosswalk_v1"))
    parser.add_argument("--footprint-root", type=Path, default=Path("outputs/point_footprint_sampling_v2"))
    parser.add_argument("--source-root", type=Path, default=Path("data/Visit_2_extracted"))
    parser.add_argument("--semantic-audit-root", type=Path, default=Path("outputs/heidelberg_boundary_numeric_semantics_visit3_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/visit3_boundary_footprint_features_v1"))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = run(
        args.ordinary.resolve(), args.volume_borders.resolve(),
        args.boundary_mapping_root.resolve(), args.private_crosswalk_root.resolve(),
        args.footprint_root.resolve(), args.source_root.resolve(),
        args.semantic_audit_root.resolve(), args.output_root.resolve(), args.replace,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
