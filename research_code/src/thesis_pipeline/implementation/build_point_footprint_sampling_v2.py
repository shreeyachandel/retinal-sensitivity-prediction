# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Build the label-blind ~100 µm MAIA-footprint sampling manifest.

Each MAIA point keeps its three already registered nearest B-scans.  Within
each B-scan, every native A-scan column whose centre lies inside the 100 µm
stimulus diameter is enumerated.  These rows are repeated image measurements
belonging to one MAIA point; they are never independent labelled observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


# The source bundle is extracted outside the working data tree in Colab.  Allow
# the stage runner to tell this script where the restored workspace lives.
PROJECT_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path.cwd())).resolve()
DEFAULT_MAPPING = PROJECT_ROOT / "outputs/point_alignment_cohort/frozen_v1/point_scan_mapping_long.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/point_footprint_sampling_v2"
DEFAULT_EXCLUSIONS = PROJECT_ROOT / "outputs/point_alignment_cohort/frozen_v1/excluded_eye_visits.csv"
FOOTPRINT_DIAMETER_MM = 0.100
EXPECTED_POINTS = 2_886
EXPECTED_POINT_SCANS = EXPECTED_POINTS * 3
KEYS = ["case_id", "point_number", "scan_rank"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def bscan_geometry(relative_bscan: str, bscan_index: int) -> tuple[float, int, int, str]:
    image_path = PROJECT_ROOT / relative_bscan
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    metadata_path = image_path.parent.parent / "bscan_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = pd.read_csv(metadata_path)
    # The privacy-minimised working data intentionally retain one volume-level
    # Heidelberg geometry row, rather than 49 duplicate per-B-scan rows.
    if len(metadata) != 1:
        raise RuntimeError(f"Expected one volume geometry row for {relative_bscan}; found {len(metadata)}")
    scale_x = float(metadata.iloc[0]["Scale X"])
    width, height = Image.open(image_path).size
    if not math.isfinite(scale_x) or scale_x <= 0:
        raise ValueError(f"Invalid Scale X for {relative_bscan}: {scale_x}")
    if int(metadata.iloc[0]["width (pixel)"]) != width or int(metadata.iloc[0]["height (pixel)"]) != height:
        raise RuntimeError(f"Image/metadata dimensions disagree for {relative_bscan}")
    return scale_x, width, height, project_path(metadata_path)


def integer_columns_inside_footprint(centre_x: float, scale_x: float, width: int) -> tuple[list[int], float, float]:
    radius_px = (FOOTPRINT_DIAMETER_MM / 2.0) / scale_x
    requested_left = centre_x - radius_px
    requested_right = centre_x + radius_px
    first = max(0, math.ceil(requested_left - 1e-12))
    last = min(width - 1, math.floor(requested_right + 1e-12))
    columns = list(range(first, last + 1))
    if not columns:
        nearest = min(width - 1, max(0, int(round(centre_x))))
        columns = [nearest]
    return columns, requested_left, requested_right


def run(mapping_path: Path, output_root: Path, exclusions_path: Path | None = DEFAULT_EXCLUSIONS) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    required = {
        "case_id", "study_id", "eye", "maia_timepoint", "point_number",
        "scan_rank", "bscan_index", "bscan_path", "along_x_bscan_px",
    }
    available_columns = set(pd.read_csv(mapping_path, nrows=0).columns)
    missing = sorted(required - available_columns)
    if missing:
        raise ValueError(f"Point-scan mapping lacks columns: {missing}")
    # Explicit usecols means sensitivity is not merely ignored; it is never
    # loaded into the geometry-construction process.
    mapping = pd.read_csv(mapping_path, usecols=sorted(required))
    excluded_cases: list[str] = []
    if exclusions_path is not None and exclusions_path.is_file():
        exclusions = pd.read_csv(exclusions_path)
        if "case_id" not in exclusions.columns:
            raise ValueError(f"Exclusion ledger lacks case_id: {exclusions_path}")
        if "decision" in exclusions.columns:
            excluded_cases = sorted(
                exclusions.loc[
                    exclusions["decision"].astype(str).str.lower().eq("exclude"), "case_id"
                ].astype(str).unique()
            )
        else:
            excluded_cases = sorted(exclusions["case_id"].astype(str).unique())
        mapping = mapping.loc[~mapping["case_id"].astype(str).isin(excluded_cases)].copy()
    if len(mapping) != EXPECTED_POINT_SCANS or mapping[KEYS].duplicated().any():
        raise RuntimeError("Expected one row for each of 2,886 points × three B-scans")
    ranks = mapping.groupby(["case_id", "point_number"])["scan_rank"].agg(lambda values: sorted(map(int, values)))
    if len(ranks) != EXPECTED_POINTS or not ranks.map(lambda values: values == [1, 2, 3]).all():
        raise RuntimeError("Every MAIA point must retain scan ranks 1, 2, and 3")

    geometry_cache: dict[tuple[str, int], tuple[float, int, int, str]] = {}
    footprint_rows: list[dict] = []
    sample_rows: list[dict] = []
    radius_mm = FOOTPRINT_DIAMETER_MM / 2.0

    for row in mapping.sort_values(KEYS).itertuples(index=False):
        geometry_key = (str(row.bscan_path), int(row.bscan_index))
        if geometry_key not in geometry_cache:
            geometry_cache[geometry_key] = bscan_geometry(*geometry_key)
        scale_x, width, height, metadata_path = geometry_cache[geometry_key]
        centre = float(row.along_x_bscan_px)
        columns, requested_left, requested_right = integer_columns_inside_footprint(centre, scale_x, width)
        clipped = requested_left < 0 or requested_right > width - 1
        realised_min = (columns[0] - centre) * scale_x
        realised_max = (columns[-1] - centre) * scale_x
        identity = {
            "case_id": row.case_id,
            "study_id": row.study_id,
            "eye": row.eye,
            "maia_timepoint": row.maia_timepoint,
            "point_number": int(row.point_number),
            "scan_rank": int(row.scan_rank),
            "bscan_index": int(row.bscan_index),
            "bscan_path": str(row.bscan_path),
        }
        footprint_rows.append(
            {
                **identity,
                "bscan_metadata_path": metadata_path,
                "bscan_width_px": width,
                "bscan_height_px": height,
                "scale_x_mm_per_ascan": scale_x,
                "registered_centre_x_px": centre,
                "stimulus_diameter_mm": FOOTPRINT_DIAMETER_MM,
                "stimulus_radius_mm": radius_mm,
                "requested_left_x_px": requested_left,
                "requested_right_x_px": requested_right,
                "first_sample_x_px": columns[0],
                "last_sample_x_px": columns[-1],
                "within_scan_sample_count": len(columns),
                "realised_min_offset_mm": realised_min,
                "realised_max_offset_mm": realised_max,
                "footprint_clipped_at_bscan_edge": clipped,
            }
        )
        for sample_rank, x_px in enumerate(columns, start=1):
            offset_mm = (x_px - centre) * scale_x
            if abs(offset_mm) > radius_mm + 1e-10:
                raise AssertionError("Enumerated A-scan lies outside the 100 µm footprint")
            sample_rows.append(
                {
                    **identity,
                    "within_scan_sample_rank": sample_rank,
                    "ascan_x_px": x_px,
                    "offset_from_registered_centre_mm": offset_mm,
                    "offset_from_registered_centre_um": offset_mm * 1000.0,
                    "scale_x_mm_per_ascan": scale_x,
                    "stimulus_diameter_mm": FOOTPRINT_DIAMETER_MM,
                }
            )

    footprints = pd.DataFrame(footprint_rows)
    samples = pd.DataFrame(sample_rows)
    if "sensitivity_db" in footprints or "sensitivity_db" in samples:
        raise AssertionError("Sensitivity must not enter footprint construction")
    if footprints[KEYS].duplicated().any() or samples[KEYS + ["within_scan_sample_rank"]].duplicated().any():
        raise AssertionError("Footprint outputs contain duplicate identities")
    if len(footprints) != EXPECTED_POINT_SCANS:
        raise AssertionError("Point-scan footprint row count changed")
    if samples.groupby(KEYS).size().min() < 2:
        raise AssertionError("Every selected B-scan should contribute multiple within-scan samples")

    building = output_root.with_name(output_root.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    footprint_path = building / "point_scan_footprints_v2.csv"
    sample_path = building / "along_scan_samples_v2.csv"
    footprints.to_csv(footprint_path, index=False, lineterminator="\n")
    samples.to_csv(sample_path, index=False, lineterminator="\n")
    counts = footprints["within_scan_sample_count"]
    checks = {
        "mapping_rows_equal_8658": len(mapping) == EXPECTED_POINT_SCANS,
        "point_scan_rows_equal_8658": len(footprints) == EXPECTED_POINT_SCANS,
        "unique_points_equal_2886": len(footprints[["case_id", "point_number"]].drop_duplicates()) == EXPECTED_POINTS,
        "three_nearest_scans_per_point": bool((footprints.groupby(["case_id", "point_number"]).size() == 3).all()),
        "multiple_within_scan_samples": int(counts.min()) >= 2,
        "all_sample_offsets_within_50_um": bool(samples["offset_from_registered_centre_mm"].abs().le(radius_mm + 1e-10).all()),
        "no_sensitivity_in_outputs": "sensitivity_db" not in footprints and "sensitivity_db" not in samples,
        "no_footprints_clipped_at_edges": not bool(footprints["footprint_clipped_at_bscan_edge"].any()),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "method": "all native A-scan centres inside the registered 100 µm MAIA stimulus diameter on each of three nearest B-scans",
        "label_blind": True,
        "stimulus_diameter_mm": FOOTPRINT_DIAMETER_MM,
        "counts": {
            "points": EXPECTED_POINTS,
            "point_scans": len(footprints),
            "within_scan_samples": len(samples),
            "minimum_samples_per_point_scan": int(counts.min()),
            "median_samples_per_point_scan": float(counts.median()),
            "maximum_samples_per_point_scan": int(counts.max()),
        },
        "checks": checks,
        "inputs": {project_path(mapping_path): sha256(mapping_path)},
        "excluded_cases": excluded_cases,
        "outputs": {
            footprint_path.name: sha256(footprint_path),
            sample_path.name: sha256(sample_path),
        },
        "modelling_unit": "one MAIA point; within-scan samples and three B-scans must be pooled before regression",
        "next_gate": "join human-verified Heidelberg boundary samples or extract a footprint-aware image representation without loading sensitivity",
    }
    (building / "FOOTPRINT_SAMPLING_V2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise RuntimeError(report)
    building.rename(output_root)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=DEFAULT_EXCLUSIONS,
        help="Frozen eye-visit exclusion ledger; excluded cases are removed before footprint construction.",
    )
    args = parser.parse_args()
    exclusions = args.exclusions.resolve() if args.exclusions else None
    print(json.dumps(run(args.mapping.resolve(), args.output_root.resolve(), exclusions), indent=2))


if __name__ == "__main__":
    main()
