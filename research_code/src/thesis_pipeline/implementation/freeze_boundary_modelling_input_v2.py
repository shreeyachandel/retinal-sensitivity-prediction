# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Freeze one visit's neutral B0-B4 predictors from a human-verified source."""

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


IDENTITY_COLUMNS = ["case_id", "study_id", "eye", "maia_timepoint", "point_number"]
FEATURE_COLUMNS = [
    f"B{i}_B{i + 1}_{summary}"
    for i in range(4)
    for summary in ("mean_um", "sd_um", "between_scan_mean_sd_um")
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(feature_root: Path, output_root: Path, visit_label: str, replace: bool) -> dict:
    if output_root.exists() and not replace:
        raise FileExistsError(f"Output exists; use --replace: {output_root}")
    manifest_path = feature_root / "BOUNDARY_FOOTPRINT_FEATURES_V2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = feature_root / "boundary_point_features_v1.csv"
    source = pd.read_csv(source_path)
    missing = sorted(set(IDENTITY_COLUMNS + FEATURE_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    frozen = source[IDENTITY_COLUMNS + FEATURE_COLUMNS].copy()
    frozen.sort_values(["study_id", "eye", "maia_timepoint", "point_number"], inplace=True)
    frozen.reset_index(drop=True, inplace=True)

    checks = {
        "source_feature_checks_passed": manifest["checks_passed"] == manifest["checks_total"],
        "source_boundaries_declared_human_verified": True,
        "neutral_names_retained": all(column.startswith("B") for column in FEATURE_COLUMNS),
        "twelve_predictors": len(FEATURE_COLUMNS) == 12,
        "point_keys_unique": not frozen[["case_id", "point_number"]].duplicated().any(),
        "no_infinite_predictors": bool(not np.isinf(frozen[FEATURE_COLUMNS].to_numpy(float)).any()),
        "every_predictor_has_observed_values": bool(frozen[FEATURE_COLUMNS].notna().any().all()),
        "points_1_to_37": bool(frozen["point_number"].between(1, 37).all()),
        "thirty_seven_points_per_case": bool(frozen.groupby("case_id").size().eq(37).all()),
        "no_outcome_or_predictions": not any(
            term in column.lower() for column in frozen.columns
            for term in ("sensitivity", "prediction", "residual")
        ),
    }
    result = {
        "status": "frozen_neutral_boundary_modelling_input_v2" if all(checks.values()) else "failed",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "visit_label": visit_label,
        "source_boundary_status": (
            "provided Heidelberg boundaries reported by the data provider as human verified; "
            "this pipeline audits identity, scan mapping, numeric validity and retinal placement, "
            "but does not redraw or anatomically rename B0-B4"
        ),
        "feature_columns": FEATURE_COLUMNS,
        "missing_predictor_values": {column: int(frozen[column].isna().sum()) for column in FEATURE_COLUMNS},
        "counts": {
            "cases": int(frozen["case_id"].nunique()),
            "participants": int(frozen["study_id"].nunique()),
            "points": len(frozen),
        },
        "source_hashes": {
            source_path.name: sha256(source_path),
            manifest_path.name: sha256(manifest_path),
        },
        "checks": checks,
        "checks_passed": sum(bool(v) for v in checks.values()),
        "checks_total": len(checks),
        "outcome_loaded": False,
        "model_performance_inspected": False,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=".boundary_freeze_v2_", dir=output_root.parent))
    try:
        table_path = building / "boundary_predictors_frozen_v2.csv"
        frozen.to_csv(table_path, index=False, lineterminator="\n")
        result["output_sha256"] = sha256(table_path)
        (building / "BOUNDARY_MODELLING_INPUT_FREEZE_V2.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
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
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visit-label", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.feature_root.resolve(), args.output_root.resolve(), args.visit_label, args.replace), indent=2))


if __name__ == "__main__":
    main()
