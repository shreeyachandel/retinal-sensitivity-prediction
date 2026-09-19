# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Freeze a private source-volume to project-case crosswalk for either visit.

This script is deliberately label-blind.  It matches the original Heidelberg
volume folders to the privacy-minimised OCT folders using retained acquisition
geometry and verifies the match from B-scan pixels.  The resulting crosswalk
links source identifiers to pseudonymous project identifiers and therefore
must remain in approved private storage; it is not a submission artefact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


MATCH_FIELDS = (
    "imageQuality",
    "images average",
    "Scale X",
    "Scale Z",
    "Angle",
    "width (pixel)",
    "height (pixel)",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one volume-level metadata row: {path}")
    return rows[0]


def canonical_number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metadata value: {value!r}")
    return result


def metadata_equal(left: dict[str, str], right: dict[str, str]) -> bool:
    return all(
        math.isclose(
            canonical_number(left[field]),
            canonical_number(right[field]),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        for field in MATCH_FIELDS
    )


def image_correlation(left_path: Path, right_path: Path, stride: int = 8) -> float:
    with Image.open(left_path) as image:
        left = np.asarray(image.convert("F"), dtype=np.float32)[::stride, ::stride]
    with Image.open(right_path) as image:
        right = np.asarray(image.convert("F"), dtype=np.float32)[::stride, ::stride]
    if left.shape != right.shape:
        return float("nan")
    left = left.ravel()
    right = right.ravel()
    left_sd = float(left.std())
    right_sd = float(right.std())
    if left_sd <= 0 or right_sd <= 0:
        return float(left_sd == right_sd == 0 and np.array_equal(left, right))
    return float(((left - left.mean()) @ (right - right.mean())) / (left.size * left_sd * right_sd))


def source_eye(source_volume: str) -> str:
    if "_OD" in source_volume:
        return "R"
    if "_OS" in source_volume:
        return "L"
    raise ValueError(f"Cannot infer eye from {source_volume}")


def load_raw(raw_root: Path) -> list[dict]:
    rows = []
    for directory in sorted(raw_root.glob("*_extracted")):
        metadata_path = directory / "BScan_meta_volume_0.csv"
        images = sorted((directory / "volume_0").glob("volume_*.png"))
        if not metadata_path.is_file() or not images:
            continue
        rows.append(
            {
                "source_volume": directory.name.removesuffix("_extracted"),
                "eye": source_eye(directory.name),
                "metadata": read_single_row(metadata_path),
                "metadata_path": metadata_path,
                "images": images,
            }
        )
    return rows


def load_private(private_oct_root: Path, oct_visit: str, maia_timepoint: str) -> list[dict]:
    rows = []
    for metadata_path in sorted(private_oct_root.glob(f"*/{oct_visit}/*/bscan_metadata.csv")):
        case_root = metadata_path.parent
        images = sorted((case_root / "volume").glob("bscan_*.png"))
        if not images:
            continue
        study_id = metadata_path.parents[2].name
        eye = metadata_path.parent.name
        rows.append(
            {
                "case_id": f"{study_id}__{eye}__{maia_timepoint}__{oct_visit}",
                "study_id": study_id,
                "eye": eye,
                "metadata": read_single_row(metadata_path),
                "metadata_path": metadata_path,
                "images": images,
            }
        )
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def run(
    raw_root: Path,
    private_oct_root: Path,
    boundary_mapping_root: Path,
    cohort_audit_path: Path,
    output_root: Path,
    oct_visit: str,
    maia_timepoint: str,
    replace: bool,
) -> dict:
    if output_root.exists() and not replace:
        raise FileExistsError(f"Output exists; use --replace: {output_root}")
    raw = load_raw(raw_root)
    private = load_private(private_oct_root, oct_visit, maia_timepoint)
    mapped_sources = {
        row["source_volume"]: int(row["export_group_index"])
        for row in read_csv(boundary_mapping_root / "VOLUME_MAPPING_V1.csv")
    }
    accepted_cases = {
        row["case_id"]
        for row in read_csv(cohort_audit_path)
        if row["oct_visit"] == oct_visit and row["processing_status"] == "completed"
    }

    private_rows = []
    for source in raw:
        matches = [
            candidate
            for candidate in private
            if candidate["eye"] == source["eye"]
            and metadata_equal(source["metadata"], candidate["metadata"])
        ]
        if len(matches) > 1:
            raise RuntimeError(f"Non-unique metadata match for {source['source_volume']}")
        if not matches:
            continue
        candidate = matches[0]
        common_indices = sorted(
            set(int(path.stem.split("_")[-1]) for path in source["images"])
            & set(int(path.stem.split("_")[-1]) for path in candidate["images"])
        )
        if not common_indices:
            raise RuntimeError(f"No common B-scan indices for {source['source_volume']}")
        verification_indices = sorted({common_indices[0], common_indices[len(common_indices) // 2], common_indices[-1]})
        correlations = []
        for index in verification_indices:
            raw_path = next(path for path in source["images"] if int(path.stem.split("_")[-1]) == index)
            private_path = next(path for path in candidate["images"] if int(path.stem.split("_")[-1]) == index)
            correlations.append(image_correlation(raw_path, private_path))
        private_rows.append(
            {
                "source_volume": source["source_volume"],
                "export_group_index": mapped_sources.get(source["source_volume"], ""),
                "case_id": candidate["case_id"],
                "study_id": candidate["study_id"],
                "eye": candidate["eye"],
                "metadata_fields_matched": len(MATCH_FIELDS),
                "pixel_verification_scan_indices": "|".join(map(str, verification_indices)),
                "minimum_pixel_correlation": min(correlations),
                "boundary_export_present": source["source_volume"] in mapped_sources,
                "accepted_point_alignment_case": candidate["case_id"] in accepted_cases,
            }
        )

    matched_case_ids = [row["case_id"] for row in private_rows]
    boundary_rows = [row for row in private_rows if row["boundary_export_present"]]
    accepted_boundary_rows = [row for row in boundary_rows if row["accepted_point_alignment_case"]]
    linked_boundary_sources = {row["source_volume"] for row in boundary_rows}
    unlinked_boundary_sources = sorted(set(mapped_sources) - linked_boundary_sources)
    checks = {
        "label_blind": True,
        "raw_source_volumes_present": len(raw) > 0,
        "private_visit_volumes_present": len(private) > 0,
        "every_private_volume_has_one_raw_match": len(private_rows) == len(private),
        "all_case_matches_unique": len(matched_case_ids) == len(set(matched_case_ids)),
        "all_metadata_fields_exact": all(row["metadata_fields_matched"] == len(MATCH_FIELDS) for row in private_rows),
        # Images were losslessly re-encoded, but float32 correlation can differ
        # from 1.0 by a few parts per million through rounding.
        "pixel_verification_near_exact": all(row["minimum_pixel_correlation"] >= 0.99999 for row in private_rows),
        "all_accepted_cases_covered_by_raw_sources": accepted_cases.issubset(set(matched_case_ids)),
        # Boundary exports may contain acquisitions outside the frozen modelling
        # cohort.  They must be reported and excluded, not force-matched to a
        # pseudonymous case.  Every link that is retained for modelling must be
        # an accepted point-alignment case.
        "all_linked_boundary_sources_are_accepted_cases": all(
            row["accepted_point_alignment_case"] for row in boundary_rows
        ),
        "unlinked_boundary_sources_explicitly_recorded": (
            len(linked_boundary_sources) + len(unlinked_boundary_sources)
            == len(mapped_sources)
        ),
    }
    result = {
        "status": "passed_private_crosswalk_frozen" if all(checks.values()) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "privacy_classification": "private re-identification crosswalk; exclude from submission and public repositories",
        "sensitivity_loaded": False,
        "model_outputs_loaded": False,
        "method": "exact retained Heidelberg geometry match plus three-scan pixel verification",
        "oct_visit": oct_visit,
        "maia_timepoint": maia_timepoint,
        "match_fields": list(MATCH_FIELDS),
        "counts": {
            "raw_source_volumes": len(raw),
            "private_visit_volumes": len(private),
            "matched_private_volumes": len(private_rows),
            "boundary_export_volumes_linked": len(boundary_rows),
            "boundary_export_volumes_unlinked_and_excluded": len(unlinked_boundary_sources),
            "accepted_point_alignment_cases": len(accepted_cases),
            "accepted_cases_with_boundaries": len(accepted_boundary_rows),
        },
        "unlinked_boundary_sources_excluded": unlinked_boundary_sources,
        "minimum_pixel_verification_correlation": min(row["minimum_pixel_correlation"] for row in private_rows),
        "checks": checks,
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "next_gate": "extract masked B0-B4 interval measurements across each accepted 100 micrometre point footprint",
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=".private_boundary_crosswalk_", dir=output_root.parent))
    try:
        write_csv(building / "PRIVATE_VOLUME_CASE_CROSSWALK_V1.csv", private_rows)
        (building / "PRIVATE_CROSSWALK_AUDIT_V1.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (building / "README.md").write_text(
            f"# Private {oct_visit} boundary crosswalk\n\n"
            "This folder links original Heidelberg source identifiers to pseudonymous project case IDs. "
            "Keep it in approved private storage and exclude it from the thesis submission and public repositories.\n",
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
    if result["status"] != "passed_private_crosswalk_frozen":
        raise RuntimeError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/Visit_2_extracted"))
    parser.add_argument("--private-oct-root", type=Path, default=Path("outputs/2026-07-14_deidentified_working_data/OCT"))
    parser.add_argument("--boundary-mapping-root", type=Path, default=Path("outputs/heidelberg_boundary_export_visit3_mapping_v1"))
    parser.add_argument("--cohort-audit", type=Path, default=Path("outputs/point_alignment_cohort/frozen_v1/cohort_case_audit.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/private_visit3_boundary_case_crosswalk_v1"))
    parser.add_argument("--oct-visit", choices=("OCT_V1", "OCT_V2"), default="OCT_V2")
    parser.add_argument("--maia-timepoint", choices=("BL", "Y01"), default="Y01")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = run(
        args.raw_root.resolve(),
        args.private_oct_root.resolve(),
        args.boundary_mapping_root.resolve(),
        args.cohort_audit.resolve(),
        args.output_root.resolve(),
        args.oct_visit,
        args.maia_timepoint,
        args.replace,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
