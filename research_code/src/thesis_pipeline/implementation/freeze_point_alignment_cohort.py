# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Freeze the accepted point-alignment cohort without altering provisional outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SOURCE_ROOT = PROJECT_ROOT / "outputs" / "point_alignment_cohort"
OUTPUT_ROOT = SOURCE_ROOT / "frozen_v1"
EXCLUDED_DECISION = "exclude"
ACCEPTED_DECISION = "accept"

TABLES = {
    "point_table.csv": "full_point_table_provisional.csv",
    "point_scan_mapping_long.csv": "point_scan_mapping_long.csv",
    "patch_manifest.csv": "patch_manifest_provisional.csv",
    "cohort_case_audit.csv": "cohort_case_audit.csv",
    "point0_labels_separate.csv": "point0_labels_separate.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"No header row in {path}")
        return reader.fieldnames, list(reader)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing frozen output: {OUTPUT_ROOT}")

    queue_path = SOURCE_ROOT / "visual_review_queue_with_sheets.csv"
    queue_fields, queue_rows = read_rows(queue_path)
    decision_counts = Counter(row["human_decision"] for row in queue_rows)
    unresolved = [row["case_id"] for row in queue_rows if row["human_decision"] not in {ACCEPTED_DECISION, EXCLUDED_DECISION}]
    if unresolved:
        raise ValueError(f"Human review decisions unresolved for: {unresolved}")
    if decision_counts[EXCLUDED_DECISION] != 1:
        raise ValueError(f"Expected exactly one excluded case, found {decision_counts[EXCLUDED_DECISION]}")

    excluded_rows = [row for row in queue_rows if row["human_decision"] == EXCLUDED_DECISION]
    excluded_case_ids = {row["case_id"] for row in excluded_rows}

    audit_fields, audit_rows = read_rows(SOURCE_ROOT / "cohort_case_audit.csv")
    completed_case_ids = {
        row["case_id"]
        for row in audit_rows
        if row["processing_status"] == "completed"
    }
    accepted_case_ids = completed_case_ids - excluded_case_ids
    if len(completed_case_ids) != 79 or len(accepted_case_ids) != 78:
        raise ValueError(
            f"Unexpected case count: completed={len(completed_case_ids)}, accepted={len(accepted_case_ids)}"
        )

    OUTPUT_ROOT.mkdir(parents=True)
    for output_name, source_name in TABLES.items():
        source_path = SOURCE_ROOT / source_name
        fields, rows = read_rows(source_path)
        frozen_rows = [row for row in rows if row["case_id"] in accepted_case_ids]
        write_rows(OUTPUT_ROOT / output_name, fields, frozen_rows)

    excluded_fields = [
        "case_id", "study_id", "eye", "maia_timepoint", "oct_visit",
        "original_registration_grade", "decision", "decision_reason",
        "removed_endpoint_points", "removed_scan_mappings", "removed_patches",
        "decision_source",
    ]
    exclusions = []
    for row in excluded_rows:
        audit = next(item for item in audit_rows if item["case_id"] == row["case_id"])
        exclusions.append(
            {
                "case_id": row["case_id"],
                "study_id": row["study_id"],
                "eye": row["eye"],
                "maia_timepoint": row["maia_timepoint"],
                "oct_visit": audit["oct_visit"],
                "original_registration_grade": audit["visual_grade"],
                "decision": EXCLUDED_DECISION,
                "decision_reason": row["decision_reason"],
                "removed_endpoint_points": "37",
                "removed_scan_mappings": "111",
                "removed_patches": "111",
                "decision_source": "final human visual QC after candidate overlay investigation",
            }
        )
    write_rows(OUTPUT_ROOT / "excluded_eye_visits.csv", excluded_fields, exclusions)
    write_rows(OUTPUT_ROOT / "visual_review_decisions.csv", queue_fields, queue_rows)

    expected_counts = {
        "point_table.csv": 78 * 37,
        "point_scan_mapping_long.csv": 78 * 37 * 3,
        "patch_manifest.csv": 78 * 37 * 3,
        "cohort_case_audit.csv": 78,
        "point0_labels_separate.csv": 78,
    }
    actual_counts = {}
    for filename, expected in expected_counts.items():
        _, rows = read_rows(OUTPUT_ROOT / filename)
        actual_counts[filename] = len(rows)
        if len(rows) != expected:
            raise ValueError(f"Unexpected row count for {filename}: {len(rows)} != {expected}")

    all_patch_paths = [
        row["patch_path"] for _, rows in [read_rows(OUTPUT_ROOT / "patch_manifest.csv")] for row in rows
    ]
    missing_patches = [path for path in all_patch_paths if not (PROJECT_ROOT / path).is_file()]
    if missing_patches:
        raise FileNotFoundError(f"Frozen manifest contains missing patch files: {missing_patches[:5]}")

    source_hashes = {
        source_name: sha256(SOURCE_ROOT / source_name)
        for source_name in set(TABLES.values()) | {queue_path.name}
    }
    frozen_hashes = {
        path.name: sha256(path)
        for path in sorted(OUTPUT_ROOT.glob("*.csv"))
    }
    metadata = {
        "freeze_id": "point_alignment_cohort_frozen_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_for_feature_extraction_and_participant_grouped_modelling",
        "provisional_source_root": str(SOURCE_ROOT.relative_to(PROJECT_ROOT)),
        "decision_queue": str(queue_path.relative_to(PROJECT_ROOT)),
        "included_eye_visits": len(accepted_case_ids),
        "excluded_eye_visits": len(excluded_case_ids),
        "included_endpoint_points": actual_counts["point_table.csv"],
        "included_scan_mappings": actual_counts["point_scan_mapping_long.csv"],
        "included_patches": actual_counts["patch_manifest.csv"],
        "included_point0_rows_separate": actual_counts["point0_labels_separate.csv"],
        "queue_decision_counts": dict(sorted(decision_counts.items())),
        "source_sha256": source_hashes,
        "frozen_csv_sha256": frozen_hashes,
        "patch_storage": "Patches remain under outputs/point_alignment_cohort/patches and are referenced by immutable project-relative paths and SHA256 hashes.",
    }
    (OUTPUT_ROOT / "freeze_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "# Frozen point-alignment cohort v1\n\n"
        "This is the modelling-ready cohort frozen after final human visual QC on 7 August 2026. "
        "It contains 78 accepted eye-visits (2,886 endpoint records, 8,658 three-scan mappings, and 8,658 patches). "
        "Point 0 remains separate. One eye-visit was excluded after final review because both saved registration candidates showed vessel misalignment.\n\n"
        "The original provisional cohort is preserved in the parent directory. Patch image files are not copied here; "
        "the frozen patch manifest references the existing project-relative paths and their SHA256 hashes. "
        "Use only the tables in this directory for feature extraction, validation-fold creation, and modelling.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
