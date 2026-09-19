# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Validate the frozen point-alignment cohort v1 tables and patch references."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
FROZEN_ROOT = PROJECT_ROOT / "outputs" / "point_alignment_cohort" / "frozen_v1"
EXCLUDED_CASE = "PUBLIC-PARTICIPANT-008__R__Y01__OCT_V2"


def read_rows(name: str) -> list[dict[str, str]]:
    with (FROZEN_ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    points = read_rows("point_table.csv")
    mappings = read_rows("point_scan_mapping_long.csv")
    patches = read_rows("patch_manifest.csv")
    audits = read_rows("cohort_case_audit.csv")
    point0 = read_rows("point0_labels_separate.csv")
    decisions = read_rows("visual_review_decisions.csv")
    exclusions = read_rows("excluded_eye_visits.csv")

    errors: list[str] = []
    expected = {
        "accepted_eye_visits": 78,
        "endpoint_points": 2886,
        "scan_mappings": 8658,
        "patches": 8658,
        "point0_rows": 78,
    }
    actual = {
        "accepted_eye_visits": len({row["case_id"] for row in points}),
        "endpoint_points": len(points),
        "scan_mappings": len(mappings),
        "patches": len(patches),
        "point0_rows": len(point0),
    }
    for key, value in expected.items():
        if actual[key] != value:
            errors.append(f"{key}: expected {value}, got {actual[key]}")

    for table_name, rows in {
        "points": points, "mappings": mappings, "patches": patches,
        "audits": audits, "point0": point0,
    }.items():
        if any(row["case_id"] == EXCLUDED_CASE for row in rows):
            errors.append(f"Excluded case present in {table_name}")

    point_keys = [(row["case_id"], row["point_number"]) for row in points]
    if len(point_keys) != len(set(point_keys)):
        errors.append("Duplicate endpoint point rows")
    if set(row["point_number"] for row in points) != {str(index) for index in range(1, 38)}:
        errors.append("Endpoint point table does not contain exactly Points 1-37")

    map_counts = Counter((row["case_id"], row["point_number"]) for row in mappings)
    patch_counts = Counter((row["case_id"], row["point_number"]) for row in patches)
    if any(value != 3 for value in map_counts.values()) or len(map_counts) != len(points):
        errors.append("Every endpoint point does not have exactly three scan mappings")
    if any(value != 3 for value in patch_counts.values()) or len(patch_counts) != len(points):
        errors.append("Every endpoint point does not have exactly three patches")

    map_ranks = defaultdict(set)
    patch_ranks = defaultdict(set)
    for row in mappings:
        map_ranks[(row["case_id"], row["point_number"])].add(row["scan_rank"])
    for row in patches:
        patch_ranks[(row["case_id"], row["point_number"])].add(row["scan_rank"])
    expected_ranks = {"1", "2", "3"}
    if any(ranks != expected_ranks for ranks in map_ranks.values()):
        errors.append("One or more points lack mapping ranks 1, 2, 3")
    if any(ranks != expected_ranks for ranks in patch_ranks.values()):
        errors.append("One or more points lack patch ranks 1, 2, 3")

    missing_patch_files = []
    hash_mismatches = []
    for row in patches:
        path = PROJECT_ROOT / row["patch_path"]
        if not path.is_file():
            missing_patch_files.append(row["patch_path"])
        elif sha256(path) != row["output_sha256"]:
            hash_mismatches.append(row["patch_path"])
    if missing_patch_files:
        errors.append(f"Missing patch files: {len(missing_patch_files)}")
    if hash_mismatches:
        errors.append(f"Patch hash mismatches: {len(hash_mismatches)}")

    decision_counts = Counter(row["human_decision"] for row in decisions)
    if decision_counts != Counter({"accept": 27, "exclude": 1}):
        errors.append(f"Unexpected visual-decision counts: {dict(decision_counts)}")
    if len(exclusions) != 1 or exclusions[0].get("case_id") != EXCLUDED_CASE:
        errors.append("Exclusion log does not contain exactly the resolved Sheet 16 exclusion")

    result = {
        "validation_status": "passed" if not errors else "failed",
        "expected_counts": expected,
        "actual_counts": actual,
        "visual_decision_counts": dict(sorted(decision_counts.items())),
        "excluded_case_id": EXCLUDED_CASE,
        "missing_patch_files": len(missing_patch_files),
        "patch_hash_mismatches": len(hash_mismatches),
        "errors": errors,
    }
    (FROZEN_ROOT / "validation_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (FROZEN_ROOT / "VALIDATION_REPORT.md").write_text(
        "# Frozen cohort validation\n\n"
        f"**Status:** {result['validation_status']}\n\n"
        f"- Included eye-visits: {actual['accepted_eye_visits']}\n"
        f"- Endpoint points (1-37): {actual['endpoint_points']}\n"
        f"- Three-scan mappings: {actual['scan_mappings']}\n"
        f"- Patches: {actual['patches']}\n"
        f"- Separate Point 0 rows: {actual['point0_rows']}\n"
        f"- Visual decisions: {dict(sorted(decision_counts.items()))}\n"
        f"- Excluded after human review: `{EXCLUDED_CASE}`\n"
        f"- Missing patch files: {len(missing_patch_files)}\n"
        f"- Patch hash mismatches: {len(hash_mismatches)}\n\n"
        + ("All freeze checks passed.\n" if not errors else "## Errors\n\n" + "\n".join(f"- {error}" for error in errors) + "\n"),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
