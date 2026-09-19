# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Combine valid Visit 1 and Visit 3 boundary points with outcomes, location and folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# Outcomes contain the full identity (including eye and visit), whereas the
# compact registered-location cache deliberately stores only the geometry
# identity.  Keep those join contracts explicit instead of requiring the
# location cache to duplicate metadata supplied by the boundary/outcome rows.
KEYS = ["study_id", "case_id", "eye", "maia_timepoint", "point_number"]
LOCATION_KEYS = ["study_id", "case_id", "point_number"]
LOCATION = ["location_eccentricity_normalised", "location_angle_sin", "location_angle_cos"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(boundary_roots: list[Path], outcome_path: Path, location_path: Path,
        folds_path: Path, output_root: Path, replace: bool) -> dict:
    if output_root.exists() and not replace:
        raise FileExistsError(f"Output exists; use --replace: {output_root}")
    blocks, hashes, feature_columns = [], {}, None
    for root in boundary_roots:
        manifest_path = root / "BOUNDARY_MODELLING_INPUT_FREEZE_V2.json"
        table_path = root / "boundary_predictors_frozen_v2.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "frozen_neutral_boundary_modelling_input_v2":
            raise RuntimeError(f"Boundary block is not frozen: {root}")
        columns = list(manifest["feature_columns"])
        if feature_columns is None:
            feature_columns = columns
        elif columns != feature_columns:
            raise RuntimeError("Visit feature definitions differ")
        blocks.append(pd.read_csv(table_path))
        hashes[str(table_path)] = sha256(table_path)
        hashes[str(manifest_path)] = sha256(manifest_path)
    boundary = pd.concat(blocks, ignore_index=True)
    if boundary[["case_id", "point_number"]].duplicated().any():
        raise RuntimeError("Duplicate case/point keys across visit blocks")

    outcomes = pd.read_csv(outcome_path)
    sensitivity = "point_table_sensitivity_db" if "point_table_sensitivity_db" in outcomes else "sensitivity_db"
    outcomes = outcomes[KEYS + [sensitivity]].rename(columns={sensitivity: "sensitivity_db"})
    # The frozen location predictor cache is intentionally compact and has
    # study_id/case_id/point_number plus the three location columns.  Older
    # local caches may also contain eye/timepoint; accepting either shape keeps
    # the table builder reproducible without manufacturing metadata.
    location_header = pd.read_csv(location_path, nrows=0)
    missing_location_columns = sorted(set(LOCATION_KEYS + LOCATION) - set(location_header.columns))
    if missing_location_columns:
        raise ValueError(
            "Location predictor cache is missing required columns: "
            f"{missing_location_columns}. Expected the compact geometry schema."
        )
    location = pd.read_csv(location_path, usecols=LOCATION_KEYS + LOCATION)
    if location[LOCATION_KEYS].duplicated().any():
        raise RuntimeError("Duplicate location predictor keys")
    folds = pd.read_csv(folds_path, usecols=["study_id", "case_id", "outer_fold"])
    table = boundary.merge(location, on=LOCATION_KEYS, how="left", validate="one_to_one")
    table = table.merge(outcomes, on=KEYS, how="left", validate="one_to_one")
    table = table.merge(folds, on=["study_id", "case_id"], how="left", validate="many_to_one")
    table["sensitivity_floor_flag"] = table["sensitivity_db"].eq(-1.0)
    table.sort_values(KEYS, inplace=True)
    table.reset_index(drop=True, inplace=True)
    predictors = LOCATION + list(feature_columns or [])
    checks = {
        "both_visits_present": set(table["maia_timepoint"]) == {"BL", "Y01"},
        "point_keys_unique": not table[["case_id", "point_number"]].duplicated().any(),
        "thirty_seven_points_per_case": bool(table.groupby("case_id").size().eq(37).all()),
        "outcomes_finite": bool(np.isfinite(table["sensitivity_db"]).all()),
        "predictors_contain_no_infinity": bool(not np.isinf(table[predictors].to_numpy(float)).any()),
        "each_predictor_has_observed_values": bool(table[predictors].notna().any().all()),
        "folds_present": bool(table["outer_fold"].notna().all()),
        "participant_grouped_folds": bool(table.groupby("study_id")["outer_fold"].nunique().eq(1).all()),
        "five_outer_folds": set(table["outer_fold"].astype(int)) == {1, 2, 3, 4, 5},
    }
    result = {
        "status": "frozen_combined_boundary_modelling_table_v2" if all(checks.values()) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictor_blocks": {"location": LOCATION, "neutral_boundary": feature_columns},
        "outcome": "pointwise MAIA sensitivity_db",
        "counts": {
            "participants": int(table["study_id"].nunique()),
            "cases": int(table["case_id"].nunique()),
            "points": len(table),
            "baseline_cases": int(table.loc[table["maia_timepoint"].eq("BL"), "case_id"].nunique()),
            "year_one_cases": int(table.loc[table["maia_timepoint"].eq("Y01"), "case_id"].nunique()),
            "floor_points_minus_1_db": int(table["sensitivity_floor_flag"].sum()),
        },
        "missing_predictor_values": {column: int(table[column].isna().sum()) for column in predictors},
        "checks": checks,
        "checks_passed": sum(bool(v) for v in checks.values()),
        "checks_total": len(checks),
        "input_hashes": {**hashes, str(outcome_path): sha256(outcome_path), str(location_path): sha256(location_path), str(folds_path): sha256(folds_path)},
        "model_performance_inspected": False,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=".combined_boundary_table_v2_", dir=output_root.parent))
    try:
        table_path = building / "combined_boundary_modelling_table_v2.csv"
        table.to_csv(table_path, index=False, lineterminator="\n")
        result["output_sha256"] = sha256(table_path)
        (building / "COMBINED_BOUNDARY_MODELLING_TABLE_V2.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(building, output_root)
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise
    if result["status"] == "failed":
        raise RuntimeError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-root", type=Path, action="append", required=True)
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--location", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run([p.resolve() for p in args.boundary_root], args.outcome.resolve(), args.location.resolve(), args.folds.resolve(), args.output_root.resolve(), args.replace), indent=2))


if __name__ == "__main__":
    main()
