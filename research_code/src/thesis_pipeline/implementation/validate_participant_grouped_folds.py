# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Validate participant-held-out fold assignments against frozen cohort v1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    args = parser.parse_args()

    frozen_root = args.frozen_root.resolve()
    fold_root = args.fold_root.resolve()
    source_cases = read_csv(frozen_root / "cohort_case_audit.csv")
    participant_rows = read_csv(fold_root / "participant_fold_assignments.csv")
    assigned_cases = read_csv(fold_root / "case_fold_assignments.csv")
    summary_rows = read_csv(fold_root / "fold_summary.csv")
    config = json.loads((fold_root / "fold_configuration.json").read_text(encoding="utf-8"))

    errors: list[str] = []
    source_case_ids = {row["case_id"] for row in source_cases}
    assigned_case_ids = {row["case_id"] for row in assigned_cases}
    if source_case_ids != assigned_case_ids:
        errors.append("Assigned case IDs do not exactly equal frozen cohort case IDs.")
    if len(assigned_cases) != len(assigned_case_ids):
        errors.append("Duplicate case IDs found in case fold assignments.")

    source_participants = {row["study_id"] for row in source_cases}
    participant_ids = [row["study_id"] for row in participant_rows]
    if set(participant_ids) != source_participants:
        errors.append("Participant assignment IDs do not exactly equal frozen cohort participant IDs.")
    if len(participant_ids) != len(set(participant_ids)):
        errors.append("Duplicate participant IDs found in participant fold assignments.")

    participant_to_fold = {row["study_id"]: row["outer_fold"] for row in participant_rows}
    valid_folds = {str(index) for index in range(1, int(config["n_outer_folds"]) + 1)}
    if set(participant_to_fold.values()) - valid_folds:
        errors.append("Invalid outer fold values found in participant assignments.")
    case_participant_folds = defaultdict(set)
    for row in assigned_cases:
        case_participant_folds[row["study_id"]].add(row["outer_fold"])
        if participant_to_fold.get(row["study_id"]) != row["outer_fold"]:
            errors.append(f"Case {row['case_id']} does not match its participant fold.")
    crossed = [study_id for study_id, folds in case_participant_folds.items() if len(folds) != 1]
    if crossed:
        errors.append(f"Participants crossing outer folds: {crossed}")

    fold_case_counts = Counter(row["outer_fold"] for row in assigned_cases)
    fold_participant_counts = Counter(participant_to_fold.values())
    if set(fold_case_counts) != valid_folds or set(fold_participant_counts) != valid_folds:
        errors.append("At least one configured fold is empty.")
    if sum(fold_case_counts.values()) != len(source_cases):
        errors.append("Fold case count does not total the frozen cohort.")

    summary_by_fold = {row["outer_fold"]: row for row in summary_rows}
    for fold in valid_folds:
        if fold not in summary_by_fold:
            errors.append(f"Missing fold summary row for fold {fold}.")
            continue
        if int(summary_by_fold[fold]["accepted_eye_visits"]) != fold_case_counts[fold]:
            errors.append(f"Fold {fold} summary eye-visit count does not match case assignments.")
        if int(summary_by_fold[fold]["participants"]) != fold_participant_counts[fold]:
            errors.append(f"Fold {fold} summary participant count does not match participant assignments.")

    # Resolve every frozen downstream row by case_id, confirming the assignment
    # can be propagated without introducing a split within a participant.
    downstream_checks = {}
    for filename in ("point_table.csv", "point_scan_mapping_long.csv", "patch_manifest.csv", "point0_labels_separate.csv"):
        rows = read_csv(frozen_root / filename)
        unknown = sorted({row["case_id"] for row in rows} - source_case_ids)
        downstream_checks[filename] = {"rows": len(rows), "unknown_case_ids": unknown}
        if unknown:
            errors.append(f"{filename} contains case IDs outside frozen cohort.")

    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "participants": len(source_participants),
        "accepted_eye_visits": len(source_cases),
        "outer_folds": int(config["n_outer_folds"]),
        "participants_per_fold": {fold: fold_participant_counts[fold] for fold in sorted(valid_folds, key=int)},
        "eye_visits_per_fold": {fold: fold_case_counts[fold] for fold in sorted(valid_folds, key=int)},
        "downstream_case_id_checks": downstream_checks,
    }
    (fold_root / "FOLD_VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Participant-fold validation", "", f"**Status:** {report['status']}", ""]
    lines += [f"- Participants in frozen audit: {report['participants']}", f"- Accepted eye-visits: {report['accepted_eye_visits']}", f"- Outer folds: {report['outer_folds']}"]
    lines += [f"- Participants per fold: {report['participants_per_fold']}", f"- Eye-visits per fold: {report['eye_visits_per_fold']}", ""]
    if errors:
        lines += ["## Errors", ""] + [f"- {error}" for error in errors] + [""]
    else:
        lines += ["All cohort cases are assigned exactly once, every participant has exactly one fold, and each frozen downstream table resolves only to frozen cohort cases.", ""]
    (fold_root / "FOLD_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
