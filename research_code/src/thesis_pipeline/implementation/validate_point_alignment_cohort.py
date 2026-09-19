# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Validate the provisional full-cohort point-alignment output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "point_alignment_cohort"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    paths = {
        "points": args.output_root / "full_point_table_provisional.csv",
        "scans": args.output_root / "point_scan_mapping_long.csv",
        "patches": args.output_root / "patch_manifest_provisional.csv",
        "cases": args.output_root / "cohort_case_audit.csv",
        "sources": args.output_root / "source_audit.csv",
        "exceptions": args.output_root / "qc_exceptions.csv",
        "queue": args.output_root / "visual_review_queue.csv",
        "point0": args.output_root / "point0_labels_separate.csv",
    }
    missing_tables = [name for name, path in paths.items() if not path.exists()]
    if missing_tables:
        raise FileNotFoundError(f"Missing validation inputs: {missing_tables}")

    points = read_csv(paths["points"])
    scans = read_csv(paths["scans"])
    patches = read_csv(paths["patches"])
    cases = read_csv(paths["cases"])
    sources = read_csv(paths["sources"])
    exceptions = read_csv(paths["exceptions"])
    queue = read_csv(paths["queue"])
    point0 = read_csv(paths["point0"])
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(len(cases) == 79, f"Expected 79 case rows, found {len(cases)}")
    require(len(points) == 2923, f"Expected 2923 point rows, found {len(points)}")
    require(len(scans) == 8769, f"Expected 8769 scan mappings, found {len(scans)}")
    require(len(patches) == 8769, f"Expected 8769 patch rows, found {len(patches)}")
    require(len(point0) == 79, f"Expected 79 separate Point 0 rows, found {len(point0)}")
    require(all(row["processing_status"] == "completed" for row in cases),
            "One or more cases did not complete")
    require(all(row["exists"] == "True" for row in sources), "One or more audited sources are missing")

    point_keys = [(row["case_id"], int(row["point_number"])) for row in points]
    scan_keys = [
        (row["case_id"], int(row["point_number"]), int(row["scan_rank"]))
        for row in scans
    ]
    patch_keys = [
        (row["case_id"], int(row["point_number"]), int(row["scan_rank"]))
        for row in patches
    ]
    require(len(point_keys) == len(set(point_keys)), "Point table contains duplicate case-point keys")
    require(len(scan_keys) == len(set(scan_keys)), "Scan mapping contains duplicate keys")
    require(len(patch_keys) == len(set(patch_keys)), "Patch manifest contains duplicate keys")
    require(set(scan_keys) == set(patch_keys), "Scan and patch keys do not match exactly")
    require(all(1 <= number <= 37 for _, number in point_keys), "Primary point table contains Point 0 or an invalid point")
    require(all(row["point_number"] == "0" for row in point0), "Separate Point 0 table contains another point")

    points_by_case = Counter(case_id for case_id, _ in point_keys)
    scans_by_point = Counter((case_id, number) for case_id, number, _ in scan_keys)
    patches_by_point = Counter((case_id, number) for case_id, number, _ in patch_keys)
    require(all(count == 37 for count in points_by_case.values()), "A case does not contain exactly 37 primary points")
    require(len(points_by_case) == 79, "Point table does not contain all 79 cases")
    require(all(count == 3 for count in scans_by_point.values()), "A point does not have exactly three scan mappings")
    require(all(count == 3 for count in patches_by_point.values()), "A point does not have exactly three patches")
    require(all({int(row["scan_rank"]) for row in scans if row["case_id"] == case_id and int(row["point_number"]) == number} == {1, 2, 3}
                for case_id, number in point_keys), "A point is missing scan rank 1, 2, or 3")

    point_label = {
        (row["case_id"], int(row["point_number"])): int(row["workbook_sensitivity_db"])
        for row in points
    }
    require(all(int(row["sensitivity_db"]) == point_label[(row["case_id"], int(row["point_number"]))]
                for row in scans), "A scan mapping sensitivity differs from the point table")
    require(all(int(row["sensitivity_db"]) == point_label[(row["case_id"], int(row["point_number"]))]
                for row in patches), "A patch sensitivity differs from the point table")

    missing_patch_files = 0
    wrong_patch_dimensions = 0
    hash_mismatches = 0
    for index, row in enumerate(patches, start=1):
        patch_path = PROJECT_ROOT / row["patch_path"]
        if not patch_path.exists():
            missing_patch_files += 1
            continue
        with Image.open(patch_path) as image:
            if image.size != (128, 496):
                wrong_patch_dimensions += 1
        if sha256_file(patch_path) != row["output_sha256"]:
            hash_mismatches += 1
        if index % 1000 == 0:
            print(f"Validated {index}/{len(patches)} patches", flush=True)

    require(missing_patch_files == 0, f"Missing patch files: {missing_patch_files}")
    require(wrong_patch_dimensions == 0, f"Patches with incorrect dimensions: {wrong_patch_dimensions}")
    require(hash_mismatches == 0, f"Patch SHA-256 mismatches: {hash_mismatches}")

    geometry_totals = {
        field: sum(int(float(row[field] or 0)) for row in cases)
        for field in (
            "points_outside_native_slo", "points_outside_scan_area",
            "points_with_projection_clipping", "points_with_missing_neighbouring_scans",
            "points_with_noncontiguous_three_scan_indices",
            "points_excessive_nearest_distance", "points_excessive_third_distance",
            "scan_lines_with_start_x_greater_than_end_x",
            "patches_with_edge_padding", "patches_with_excessive_edge_padding",
        )
    }
    issue_counts = dict(sorted(Counter(row["issue_type"] for row in exceptions).items()))
    status = "passed_pending_human_visual_qc" if not failures else "failed"
    result = {
        "status": status,
        "dataset_status": "provisional_not_frozen_not_for_modelling",
        "case_count": len(cases),
        "point_count": len(points),
        "scan_mapping_count": len(scans),
        "patch_manifest_count": len(patches),
        "point0_separate_count": len(point0),
        "qc_exception_count": len(exceptions),
        "flagged_case_count": len({row["case_id"] for row in exceptions}),
        "visual_review_queue_count": len(queue),
        "geometry_and_patch_totals": geometry_totals,
        "qc_issue_counts": issue_counts,
        "missing_patch_files": missing_patch_files,
        "wrong_patch_dimensions": wrong_patch_dimensions,
        "patch_hash_mismatches": hash_mismatches,
        "validation_failures": failures,
    }
    args.output_root.joinpath("validation_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Point-alignment cohort validation",
        "",
        f"**Validation status:** {status}",
        "**Dataset status:** Provisional; pending human visual QC and not authorised for modelling.",
        "",
        f"- Cases: {len(cases)}",
        f"- Points 1-37: {len(points)}",
        f"- Three-scan mappings: {len(scans)}",
        f"- Patches: {len(patches)}",
        f"- Separate Point 0 rows: {len(point0)}",
        f"- QC exception rows: {len(exceptions)}",
        f"- Visual-review queue rows: {len(queue)}",
        f"- Missing patch files: {missing_patch_files}",
        f"- Incorrect patch dimensions: {wrong_patch_dimensions}",
        f"- Patch hash mismatches: {hash_mismatches}",
        "",
        "## Geometry and patch flags",
        "",
        *[f"- {name}: {value}" for name, value in geometry_totals.items()],
        "",
        "## QC issue counts",
        "",
        *[f"- {name}: {value}" for name, value in issue_counts.items()],
        "",
        "## Next gate",
        "",
        "Review every flagged case and the deterministic unflagged sample in `visual_review_queue.csv`. Record accept, correct, or exclude with a reason before freezing the dataset.",
        "",
    ]
    if failures:
        markdown.extend(["## Validation failures", "", *[f"- {item}" for item in failures], ""])
    args.output_root.joinpath("VALIDATION_REPORT.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
