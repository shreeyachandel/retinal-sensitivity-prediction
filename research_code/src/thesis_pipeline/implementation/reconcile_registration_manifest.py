# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Create the frozen same-visit registration cohort from the audit manifest.

The original 132-row registration_manifest.csv is retained unchanged. This
script creates:

* registration_manifest_frozen_v1.csv - the 79 pre-QC same-visit pairs.
* registration_manifest_reconciliation_v1.csv - the decision for all 132 rows.
* registration_manifest_frozen_v1.json - version, counts, rules, and hashes.

The two anatomy-resolved year-one MAIA image pairs must already be present in
the privacy-minimised working tree. No source identifiers are written.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKING_ROOT = (
    PROJECT_ROOT / "outputs" / "2026-07-14_deidentified_working_data"
)
WORKING_FILES = WORKING_ROOT / "working_files"
SOURCE_MANIFEST = WORKING_FILES / "registration_manifest.csv"
FROZEN_MANIFEST = WORKING_FILES / "registration_manifest_frozen_v1.csv"
RECONCILIATION_MANIFEST = (
    WORKING_FILES / "registration_manifest_reconciliation_v1.csv"
)
VERSION_METADATA = WORKING_FILES / "registration_manifest_frozen_v1.json"
PILOT_CASES = WORKING_FILES / "pilot_cases.csv"
MANUAL_SUMMARY = PROJECT_ROOT / "outputs" / "registration_runs" / "registration_summary.csv"
AUTOMATIC_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "automatic_registration_batch"
    / "automatic_batch_qc_summary.csv"
)

MANIFEST_VERSION = "v1_2026-07-27"
CROSS_VISIT_KEY = ("PUBLIC-PARTICIPANT-008", "R", "BL")
RECOVERED_KEYS = {
    ("PUBLIC-PARTICIPANT-004", "R", "Y01"),
    ("PUBLIC-PARTICIPANT-004", "L", "Y01"),
}

FROZEN_FIELDS = [
    "manifest_version",
    "case_id",
    "study_id",
    "eye",
    "maia_timepoint",
    "oct_visit",
    "acquisition_date_match",
    "pairing_basis",
    "maia_clean_path",
    "maia_overlay_path",
    "oct_slo_path",
    "oct_volume_path",
    "scan_coordinates_path",
    "n_bscans",
    "n_coordinate_rows",
    "geometry_valid",
    "point_labels_available",
    "point_coordinates_available",
    "registration_status",
    "provisional_transform",
    "transform_decision_status",
    "registration_output_path",
    "registration_qc",
    "inclusion_status",
    "exclusion_reason",
]

RECONCILIATION_FIELDS = [
    "manifest_version",
    "source_manifest_row",
    "study_id",
    "eye",
    "maia_timepoint",
    "original_maia_status",
    "corrected_maia_status",
    "selected_oct_visit",
    "selected_oct_path",
    "n_bscans",
    "n_coordinate_rows",
    "same_visit_pair_included",
    "analysis_disposition",
    "pairing_basis",
    "registration_status",
    "exclusion_reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manual_results() -> dict[tuple[str, str, str], dict[str, str]]:
    pilots = {row["pilot_id"]: row for row in read_csv(PILOT_CASES)}
    results: dict[tuple[str, str, str], dict[str, str]] = {}
    for summary in read_csv(MANUAL_SUMMARY):
        if summary["status"] != "successful" or summary["pilot_id"] == "P06":
            continue
        pilot = pilots[summary["pilot_id"]]
        key = (
            pilot["study_id"],
            pilot["eye"],
            pilot["provisional_maia_timepoint"],
        )
        results[key] = {
            **summary,
            "registration_output_path": (
                f"outputs/registration_runs/{summary['pilot_id']}"
            ),
        }
    return results


def automatic_results() -> dict[tuple[str, str, str], dict[str, str]]:
    results: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_csv(AUTOMATIC_SUMMARY):
        if row["run_state"] == "failed":
            continue
        key = (row["study_id"], row["eye"], row["maia_timepoint"])
        results[key] = {
            **row,
            "registration_output_path": (
                f"outputs/automatic_registration_batch/{row['case_id']}"
            ),
        }
    return results


def recovered_paths(eye: str) -> tuple[str, str]:
    base = f"MAIA/PUBLIC-PARTICIPANT-004/Y01/{eye}"
    return f"{base}/maia_clean.png", f"{base}/maia_overlay.png"


def reconcile() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_rows = read_csv(SOURCE_MANIFEST)
    manual = manual_results()
    automatic = automatic_results()
    frozen: list[dict[str, object]] = []
    reconciled: list[dict[str, object]] = []

    for index, source in enumerate(source_rows, start=2):
        row = dict(source)
        key = (row["study_id"], row["eye"], row["maia_timepoint"])
        original_maia_status = row["maia_status"]
        pairing_basis = ""
        exclusion_reason = ""
        analysis_disposition = "excluded_from_primary_registration"
        included = False
        oct_visit = ""
        prefix = ""

        if key in RECOVERED_KEYS:
            clean_path, overlay_path = recovered_paths(row["eye"])
            row["maia_clean_path"] = clean_path
            row["maia_overlay_path"] = overlay_path
            row["maia_status"] = "paired"
            pairing_basis = (
                "Raw acquisition date and participant-eye match; duplicated "
                "workbook examination identifier resolved by optic-disc side "
                "and vessel anatomy against the same eye at baseline."
            )

        if row["maia_timepoint"] == "M06":
            analysis_disposition = "longitudinal_functional_only"
            exclusion_reason = (
                "No six-month OCT acquisition was supplied; retain the "
                "functional examination for separate longitudinal analysis."
            )
        elif key == CROSS_VISIT_KEY:
            analysis_disposition = "technical_cross_visit_only"
            exclusion_reason = (
                "OCT_V1 is byte-identical to OCT_V2 and represents a "
                "duplicated year-one acquisition, not a baseline acquisition."
            )
        else:
            if row["maia_timepoint"] == "BL":
                oct_visit, prefix = "OCT_V1", "oct_v1"
            elif row["maia_timepoint"] == "Y01":
                oct_visit, prefix = "OCT_V2", "oct_v2"

            if not oct_visit:
                exclusion_reason = "Timepoint is outside the primary registration design."
            elif row["maia_status"] != "paired" or not row["maia_clean_path"]:
                exclusion_reason = (
                    "A complete clean/overlay MAIA image pair is not available."
                )
            elif row[f"{prefix}_status"] != "ready" or not row[f"{prefix}_path"]:
                if (
                    row[f"{prefix}_n_bscans"]
                    and row[f"{prefix}_n_coordinate_rows"]
                    and row[f"{prefix}_n_bscans"]
                    != row[f"{prefix}_n_coordinate_rows"]
                ):
                    exclusion_reason = (
                        "Invalid OCT geometry: B-scan count does not match "
                        "scan-coordinate row count."
                    )
                else:
                    exclusion_reason = (
                        "No usable OCT acquisition is available for the same visit."
                    )
            elif row[f"{prefix}_n_bscans"] != row[f"{prefix}_n_coordinate_rows"]:
                exclusion_reason = (
                    "Invalid OCT geometry: B-scan count does not match "
                    "scan-coordinate row count."
                )
            else:
                included = True
                analysis_disposition = "primary_registration_candidate"
                if not pairing_basis:
                    pairing_basis = (
                        "Participant, eye, visit, acquisition date, image "
                        "availability, and OCT geometry verified in the raw-data audit."
                    )

        selected_oct_path = row[f"{prefix}_path"] if prefix else ""
        n_bscans = row[f"{prefix}_n_bscans"] if prefix else ""
        n_coordinate_rows = row[f"{prefix}_n_coordinate_rows"] if prefix else ""
        registration_status = "not_applicable"
        provisional_transform = ""
        transform_decision_status = "not_applicable"
        registration_output_path = ""
        registration_qc = ""

        if included:
            if key in manual:
                result = manual[key]
                registration_status = "manual_reference_accepted"
                provisional_transform = result["chosen_transform"]
                transform_decision_status = "accepted_manual_reference"
                registration_output_path = result["registration_output_path"]
                registration_qc = result["visual_assessment"]
            elif key in automatic:
                result = automatic[key]
                registration_status = "automatic_complete_pending_visual_qc"
                provisional_transform = result["provisional_recommendation"]
                transform_decision_status = "provisional_pending_visual_qc"
                registration_output_path = result["registration_output_path"]
                registration_qc = "pending_visual_review"
            else:
                registration_status = "not_started"
                transform_decision_status = "not_selected"
                registration_qc = "not_started"

            case_id = (
                f"{row['study_id']}__{row['eye']}__"
                f"{row['maia_timepoint']}__{oct_visit}"
            )
            oct_base = selected_oct_path
            frozen.append(
                {
                    "manifest_version": MANIFEST_VERSION,
                    "case_id": case_id,
                    "study_id": row["study_id"],
                    "eye": row["eye"],
                    "maia_timepoint": row["maia_timepoint"],
                    "oct_visit": oct_visit,
                    "acquisition_date_match": "verified_in_raw_audit",
                    "pairing_basis": pairing_basis,
                    "maia_clean_path": row["maia_clean_path"],
                    "maia_overlay_path": row["maia_overlay_path"],
                    "oct_slo_path": f"{oct_base}/slo.png",
                    "oct_volume_path": f"{oct_base}/volume",
                    "scan_coordinates_path": f"{oct_base}/slo_coordinates.csv",
                    "n_bscans": int(n_bscans),
                    "n_coordinate_rows": int(n_coordinate_rows),
                    "geometry_valid": "yes",
                    "point_labels_available": "yes",
                    "point_coordinates_available": "pending_overlay_extraction",
                    "registration_status": registration_status,
                    "provisional_transform": provisional_transform,
                    "transform_decision_status": transform_decision_status,
                    "registration_output_path": registration_output_path,
                    "registration_qc": registration_qc,
                    "inclusion_status": "included_pre_qc",
                    "exclusion_reason": "",
                }
            )

        reconciled.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "source_manifest_row": index,
                "study_id": row["study_id"],
                "eye": row["eye"],
                "maia_timepoint": row["maia_timepoint"],
                "original_maia_status": original_maia_status,
                "corrected_maia_status": row["maia_status"],
                "selected_oct_visit": oct_visit,
                "selected_oct_path": selected_oct_path,
                "n_bscans": n_bscans,
                "n_coordinate_rows": n_coordinate_rows,
                "same_visit_pair_included": "yes" if included else "no",
                "analysis_disposition": analysis_disposition,
                "pairing_basis": pairing_basis,
                "registration_status": registration_status,
                "exclusion_reason": exclusion_reason,
            }
        )

    return frozen, reconciled


def validate(
    frozen: list[dict[str, object]],
    reconciled: list[dict[str, object]],
) -> dict[str, object]:
    assert len(reconciled) == 132
    assert len(frozen) == 79
    assert len({row["case_id"] for row in frozen}) == 79
    assert len({(row["study_id"], row["eye"], row["maia_timepoint"]) for row in frozen}) == 79
    assert Counter(row["maia_timepoint"] for row in frozen) == {
        "BL": 41,
        "Y01": 38,
    }
    assert len({row["study_id"] for row in frozen}) == 23
    assert CROSS_VISIT_KEY not in {
        (row["study_id"], row["eye"], row["maia_timepoint"]) for row in frozen
    }
    assert RECOVERED_KEYS.issubset(
        {
            (row["study_id"], row["eye"], row["maia_timepoint"])
            for row in frozen
        }
    )

    required_paths = [
        "maia_clean_path",
        "maia_overlay_path",
        "oct_slo_path",
        "oct_volume_path",
        "scan_coordinates_path",
    ]
    for row in frozen:
        assert row["n_bscans"] == row["n_coordinate_rows"]
        for field in required_paths:
            path = WORKING_ROOT / str(row[field])
            assert path.exists(), f"Missing {field} for {row['case_id']}: {path}"

    status_counts = Counter(row["registration_status"] for row in frozen)
    assert status_counts == {
        "automatic_complete_pending_visual_qc": 71,
        "manual_reference_accepted": 6,
        "not_started": 2,
    }
    disposition_counts = Counter(row["analysis_disposition"] for row in reconciled)
    assert disposition_counts == {
        "primary_registration_candidate": 79,
        "longitudinal_functional_only": 44,
        "excluded_from_primary_registration": 8,
        "technical_cross_visit_only": 1,
    }
    return {
        "rows_in_source_manifest": len(reconciled),
        "included_pre_qc_pairs": len(frozen),
        "included_by_timepoint": dict(
            sorted(Counter(row["maia_timepoint"] for row in frozen).items())
        ),
        "participants_included": len({row["study_id"] for row in frozen}),
        "registration_status_counts": dict(sorted(status_counts.items())),
        "reconciliation_disposition_counts": dict(
            sorted(disposition_counts.items())
        ),
    }


def main() -> None:
    frozen, reconciled = reconcile()
    counts = validate(frozen, reconciled)
    write_csv(FROZEN_MANIFEST, frozen, FROZEN_FIELDS)
    write_csv(RECONCILIATION_MANIFEST, reconciled, RECONCILIATION_FIELDS)

    metadata = {
        "manifest_version": MANIFEST_VERSION,
        "frozen_on": date.today().isoformat(),
        "purpose": "Pre-QC same-visit MAIA-OCT registration cohort.",
        "source_manifest_preserved": str(SOURCE_MANIFEST.relative_to(PROJECT_ROOT)),
        "rules": [
            "Primary registration uses baseline with OCT_V1 and year one with OCT_V2.",
            "Six-month functional records are retained outside the registration cohort because no six-month OCT was supplied.",
            "P06 baseline is excluded because its OCT_V1 and OCT_V2 acquisitions are byte-identical year-one data.",
            "Rows require a complete MAIA clean/overlay pair, same-visit OCT, matching eye, and valid B-scan/coordinate geometry.",
            "Two year-one eye assignments were recovered from raw acquisition date, optic-disc side, and vessel anatomy without writing source identifiers.",
        ],
        "counts": counts,
        "hashes": {
            "source_registration_manifest_sha256": sha256(SOURCE_MANIFEST),
            "frozen_manifest_sha256": sha256(FROZEN_MANIFEST),
            "reconciliation_manifest_sha256": sha256(RECONCILIATION_MANIFEST),
        },
    }
    VERSION_METADATA.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
