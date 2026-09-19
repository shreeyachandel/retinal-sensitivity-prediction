# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Run automatic registration on the frozen same-visit registration cohort.

The input manifest has already been reconciled against raw identifiers,
acquisition dates, eye, image anatomy, file availability, and OCT geometry.
Accepted manual reference cases are skipped. Existing automatic results are
reused unless --overwrite is supplied. All new results remain pending visual
review.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import SimpleITK as sitk

import registration_automatic as automatic
import registration_landmarks as common


MANIFEST = (
    common.WORKING_ROOT
    / "working_files"
    / "registration_manifest_frozen_v1.csv"
)
DEFAULT_OUTPUT_ROOT = common.PROJECT_ROOT / "outputs" / "automatic_registration_batch"


def build_candidates() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    candidates: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        key = (row["study_id"], row["eye"], row["maia_timepoint"])
        if row["inclusion_status"] != "included_pre_qc":
            excluded.append({"case_key": "|".join(key), "reason": "not_in_frozen_pre_qc_cohort"})
            continue
        if row["registration_status"] == "manual_reference_accepted":
            excluded.append({"case_key": "|".join(key), "reason": "manual_success_already_complete"})
            continue
        if row["geometry_valid"] != "yes" or row["n_bscans"] != row["n_coordinate_rows"]:
            excluded.append({"case_key": "|".join(key), "reason": "bscan_coordinate_count_mismatch"})
            continue
        moving_path = common.WORKING_ROOT / row["maia_clean_path"]
        fixed_path = common.WORKING_ROOT / row["oct_slo_path"]
        if not moving_path.exists() or not fixed_path.exists():
            excluded.append({"case_key": "|".join(key), "reason": "required_image_file_missing"})
            continue
        case_id = row["case_id"]
        candidates.append(
            {
                "case_id": case_id,
                "moving_path": moving_path,
                "fixed_path": fixed_path,
                "metadata": {
                    "study_id": row["study_id"],
                    "eye": row["eye"],
                    "maia_timepoint": row["maia_timepoint"],
                    "oct_visit": row["oct_visit"],
                    "visit_mapping_status": "verified_raw_audit",
                    "pairing_basis": row["pairing_basis"],
                    "manifest_version": row["manifest_version"],
                    "n_bscans": int(row["n_bscans"]),
                    "n_coordinate_rows": int(row["n_coordinate_rows"]),
                },
            }
        )
    return candidates, excluded


def run_one(candidate: dict[str, object], output_root: Path, overwrite: bool):
    case_id = str(candidate["case_id"])
    output_dir = output_root / case_id
    result_path = output_dir / "automatic_registration_result.json"
    if result_path.exists() and not overwrite:
        return json.loads(result_path.read_text(encoding="utf-8")), "existing"
    result = automatic.run_registration(
        case_id=case_id,
        moving_path=Path(candidate["moving_path"]),
        fixed_path=Path(candidate["fixed_path"]),
        output_dir=output_dir,
        manual_dir=None,
        case_metadata=dict(candidate["metadata"]),
    )
    return result, "completed"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    candidates, excluded = build_candidates()
    write_csv(
        output_root / "excluded_manifest_rows.csv",
        excluded,
        ["case_key", "reason"],
    )
    print(f"Runnable remaining cases: {len(candidates)}", flush=True)
    print(f"Excluded or already complete: {len(excluded)}", flush=True)

    summary_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_one, candidate, output_root, args.overwrite): candidate
            for candidate in candidates
        }
        for index, future in enumerate(as_completed(futures), start=1):
            candidate = futures[future]
            case_id = str(candidate["case_id"])
            metadata = dict(candidate["metadata"])
            try:
                result, run_state = future.result()
                similarity = result["transforms"]["similarity"]
                affine = result["transforms"]["affine"]
                summary_rows.append(
                    {
                        "case_id": case_id,
                        "study_id": metadata["study_id"],
                        "eye": metadata["eye"],
                        "maia_timepoint": metadata["maia_timepoint"],
                        "oct_visit": metadata["oct_visit"],
                        "visit_mapping_status": "verified_raw_audit",
                        "run_state": run_state,
                        "pipeline_status": result["acceptance_status"],
                        "provisional_recommendation": result["provisional_recommendation"],
                        "similarity_vessel_correlation": similarity["vessel_correlation"],
                        "affine_vessel_correlation": affine["vessel_correlation"],
                        "affine_plausible": affine["plausibility"]["plausible"],
                        "affine_condition_number": affine["plausibility"]["condition_number"],
                        "affine_overlap_fraction": affine["overlap_fraction"],
                        "error": "",
                    }
                )
            except Exception as exc:
                summary_rows.append(
                    {
                        "case_id": case_id,
                        "study_id": metadata["study_id"],
                        "eye": metadata["eye"],
                        "maia_timepoint": metadata["maia_timepoint"],
                        "oct_visit": metadata["oct_visit"],
                        "visit_mapping_status": "verified_raw_audit",
                        "run_state": "failed",
                        "pipeline_status": "not_generated",
                        "provisional_recommendation": "",
                        "similarity_vessel_correlation": "",
                        "affine_vessel_correlation": "",
                        "affine_plausible": "",
                        "affine_condition_number": "",
                        "affine_overlap_fraction": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            print(f"[{index}/{len(candidates)}] {case_id}", flush=True)

    summary_rows.sort(key=lambda row: str(row["case_id"]))
    fields = [
        "case_id", "study_id", "eye", "maia_timepoint", "oct_visit",
        "visit_mapping_status", "run_state", "pipeline_status",
        "provisional_recommendation", "similarity_vessel_correlation",
        "affine_vessel_correlation", "affine_plausible",
        "affine_condition_number", "affine_overlap_fraction", "error",
    ]
    write_csv(output_root / "automatic_batch_qc_summary.csv", summary_rows, fields)
    failures = sum(row["run_state"] == "failed" for row in summary_rows)
    print(f"Batch complete: {len(summary_rows) - failures} generated, {failures} failed", flush=True)
    print("All generated results remain pending visual review.", flush=True)


if __name__ == "__main__":
    main()
