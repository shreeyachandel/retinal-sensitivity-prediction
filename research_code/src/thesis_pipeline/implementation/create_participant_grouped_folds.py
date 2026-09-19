# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Create deterministic participant-held-out outer folds from frozen cohort v1.

This deliberately uses cohort membership and visit metadata only.  It does not
read sensitivity values, OCT pixels, patch features, or QC scores, so outcomes
cannot influence the split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = {"case_id", "study_id", "eye", "maia_timepoint", "oct_visit", "processing_status"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    frozen_root = args.frozen_root.resolve()
    output_root = args.output_root.resolve()
    audit_path = frozen_root / "cohort_case_audit.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")
    if args.n_folds < 2:
        raise ValueError("At least two folds are required.")

    with audit_path.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    missing = REQUIRED_COLUMNS - set(cases[0] if cases else [])
    if missing:
        raise ValueError(f"Missing required cohort audit columns: {sorted(missing)}")
    if not cases:
        raise ValueError("Frozen cohort audit is empty.")
    if any(row["processing_status"] != "completed" for row in cases):
        raise ValueError("Frozen cohort audit includes non-completed eye-visits.")

    participant_cases: dict[str, list[dict]] = defaultdict(list)
    for row in cases:
        participant_cases[row["study_id"]].append(row)
    if len(participant_cases) < args.n_folds:
        raise ValueError("More folds than participants.")

    participant_rows = []
    for study_id, rows in participant_cases.items():
        baseline = sum(row["maia_timepoint"] == "BL" for row in rows)
        year_one = sum(row["maia_timepoint"] == "Y01" for row in rows)
        participant_rows.append(
            {
                "study_id": study_id,
                "accepted_eye_visits": len(rows),
                "baseline_eye_visits": baseline,
                "year_one_eye_visits": year_one,
                "endpoint_points_1_to_37": len(rows) * 37,
                "patches_three_per_point": len(rows) * 111,
            }
        )

    # Largest and most longitudinal participant blocks are placed first; at
    # each step the lightest fold is selected.  The final tie break is the fold
    # number, making the result reproducible without a random seed.
    participant_rows.sort(
        key=lambda row: (
            -row["accepted_eye_visits"],
            -row["baseline_eye_visits"],
            -row["year_one_eye_visits"],
            row["study_id"],
        )
    )
    folds = {
        fold: {"participants": 0, "eye_visits": 0, "baseline": 0, "year_one": 0}
        for fold in range(1, args.n_folds + 1)
    }
    assignment: dict[str, int] = {}
    for row in participant_rows:
        fold = min(
            folds,
            key=lambda candidate: (
                folds[candidate]["eye_visits"],
                folds[candidate]["participants"],
                folds[candidate]["baseline"],
                folds[candidate]["year_one"],
                candidate,
            ),
        )
        assignment[row["study_id"]] = fold
        folds[fold]["participants"] += 1
        folds[fold]["eye_visits"] += row["accepted_eye_visits"]
        folds[fold]["baseline"] += row["baseline_eye_visits"]
        folds[fold]["year_one"] += row["year_one_eye_visits"]

    participant_rows.sort(key=lambda row: (assignment[row["study_id"]], row["study_id"]))
    for row in participant_rows:
        row["outer_fold"] = assignment[row["study_id"]]

    case_rows = []
    for row in sorted(cases, key=lambda item: (assignment[item["study_id"]], item["study_id"], item["case_id"])):
        case_rows.append(
            {
                "case_id": row["case_id"],
                "study_id": row["study_id"],
                "outer_fold": assignment[row["study_id"]],
                "eye": row["eye"],
                "maia_timepoint": row["maia_timepoint"],
                "oct_visit": row["oct_visit"],
                "endpoint_points_1_to_37": 37,
                "patches_three_per_point": 111,
            }
        )

    summary_rows = []
    for fold in range(1, args.n_folds + 1):
        summary_rows.append(
            {
                "outer_fold": fold,
                "participants": folds[fold]["participants"],
                "accepted_eye_visits": folds[fold]["eye_visits"],
                "baseline_eye_visits": folds[fold]["baseline"],
                "year_one_eye_visits": folds[fold]["year_one"],
                "endpoint_points_1_to_37": folds[fold]["eye_visits"] * 37,
                "patches_three_per_point": folds[fold]["eye_visits"] * 111,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_root / "participant_fold_assignments.csv",
        participant_rows,
        [
            "study_id",
            "outer_fold",
            "accepted_eye_visits",
            "baseline_eye_visits",
            "year_one_eye_visits",
            "endpoint_points_1_to_37",
            "patches_three_per_point",
        ],
    )
    write_csv(
        output_root / "case_fold_assignments.csv",
        case_rows,
        [
            "case_id",
            "study_id",
            "outer_fold",
            "eye",
            "maia_timepoint",
            "oct_visit",
            "endpoint_points_1_to_37",
            "patches_three_per_point",
        ],
    )
    write_csv(
        output_root / "fold_summary.csv",
        summary_rows,
        [
            "outer_fold",
            "participants",
            "accepted_eye_visits",
            "baseline_eye_visits",
            "year_one_eye_visits",
            "endpoint_points_1_to_37",
            "patches_three_per_point",
        ],
    )

    config = {
        "fold_set_id": "participant_grouped_outer_folds_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_frozen_root": str(frozen_root),
        "source_cohort_case_audit_sha256": sha256(audit_path),
        "n_outer_folds": args.n_folds,
        "allocation_unit": "study_id",
        "allocation_method": "deterministic greedy allocation using only accepted eye-visit and visit-timepoint counts",
        "explicitly_not_used_for_allocation": [
            "MAIA sensitivity values",
            "OCT images, intensities, features, or patches",
            "registration QC scores or visual grades",
        ],
        "participants_in_frozen_case_audit": len(participant_cases),
        "accepted_eye_visits": len(cases),
        "endpoint_points_1_to_37": len(cases) * 37,
        "patches_three_per_point": len(cases) * 111,
        "fold_summary": summary_rows,
        "cohort_accounting_note": (
            "The frozen cohort audit contains 22 unique study_id values. Earlier project-level "
            "documentation refers to 23 registered participants; this historical-count discrepancy "
            "must be reconciled before the thesis methods/counts are finalised."
        ),
    }
    (output_root / "fold_configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_root / "README.md").write_text(
        "# Participant-grouped outer folds v1\n\n"
        "These are the fixed five outer validation folds for the frozen point-alignment cohort. "
        "Every eye, visit, MAIA point, mapping and OCT patch belonging to a participant is assigned "
        "to that participant's one outer fold. Use the held-out fold for final testing only; perform "
        "feature decisions and any tuning only inside the remaining participants.\n\n"
        "The allocation uses cohort membership and visit-timepoint counts only, never sensitivity values "
        "or image-derived data. See `fold_configuration.json` for the deterministic rule and source hash.\n\n"
        "Important accounting note: the frozen audit contains 22 unique participant IDs, even though older "
        "project documentation refers to 23 registered participants. This is recorded here for reconciliation "
        "and does not change the no-leakage rule.\n",
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
