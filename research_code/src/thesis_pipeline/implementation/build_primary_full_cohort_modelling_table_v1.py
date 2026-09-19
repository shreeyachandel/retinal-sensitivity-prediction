# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Build the all-eligible-point primary modelling table from frozen embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["study_id", "case_id", "eye", "maia_timepoint", "point_number"]
LOCATION = [
    "location_eccentricity_normalised",
    "location_angle_sin",
    "location_angle_cos",
]
EXPECTED_EMBEDDING_ZIP_SHA256 = "2d9afaf1f5f5e14c1e21476155c0580e30ac62763387db19bb9f91d45e6504b6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"Unsafe ZIP member: {name}")
        handle.extractall(destination)


def run(
    embedding_zip: Path,
    outcome_csv: Path,
    location_csv: Path,
    output_root: Path,
    expected_embedding_sha256: str = EXPECTED_EMBEDDING_ZIP_SHA256,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    actual_zip_sha256 = sha256(embedding_zip)
    if expected_embedding_sha256 and actual_zip_sha256 != expected_embedding_sha256:
        raise RuntimeError(
            f"Full-cohort embedding archive hash mismatch: {actual_zip_sha256}"
        )
    staging = Path(tempfile.mkdtemp(prefix=".primary_embedding_", dir=output_root.parent))
    building = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building_", dir=output_root.parent))
    try:
        safe_extract(embedding_zip, staging)
        point_rows = pd.read_csv(staging / "point_embedding_rows_v1.csv")
        representation_path = staging / "point_representation_mean_sd_v1.npy"
        representation = np.load(representation_path, mmap_mode="r", allow_pickle=False)
        outcome = pd.read_csv(outcome_csv)
        location = pd.read_csv(location_csv)

        if "point_table_sensitivity_db" not in outcome.columns:
            raise RuntimeError("Validated outcome table lacks point_table_sensitivity_db")
        outcome = outcome.rename(columns={"point_table_sensitivity_db": "sensitivity_db"})
        outcome = outcome[KEYS + ["outer_fold", "sensitivity_db", "sensitivity_floor_flag"]]
        location = location[KEYS[:2] + ["point_number"] + LOCATION]
        joined = point_rows.merge(outcome, on=KEYS, how="left", validate="one_to_one")
        joined = joined.merge(location, on=KEYS[:2] + ["point_number"], how="left", validate="one_to_one")
        joined = joined.sort_values("point_embedding_row_index").reset_index(drop=True)
        checks = {
            "rows_equal_2886": len(joined) == 2886,
            "representation_shape_2886_by_2048": tuple(representation.shape) == (2886, 2048),
            "representation_finite": bool(np.isfinite(representation).all()),
            "participants_equal_22": joined["study_id"].nunique() == 22,
            "eye_visits_equal_78": joined["case_id"].nunique() == 78,
            "points_1_to_37": joined["point_number"].between(1, 37).all(),
            "outcome_complete": not joined["sensitivity_db"].isna().any(),
            "location_complete": not joined[LOCATION].isna().any().any(),
            "outer_folds_exact": set(joined["outer_fold"].astype(int)) == {1, 2, 3, 4, 5},
            "participant_fold_isolation": joined.groupby("study_id")["outer_fold"].nunique().eq(1).all(),
            "embedding_row_order": joined["point_embedding_row_index"].tolist() == list(range(2886)),
        }
        checks = {name: bool(value) for name, value in checks.items()}
        failed = [name for name, value in checks.items() if not bool(value)]
        if failed:
            raise RuntimeError(f"Primary table checks failed: {failed}")

        selected = joined[
            [
                "point_embedding_row_index",
                *KEYS,
                "outer_fold",
                *LOCATION,
                "sensitivity_db",
                "sensitivity_floor_flag",
            ]
        ]
        selected.to_csv(building / "primary_full_cohort_modelling_rows_v1.csv", index=False, lineterminator="\n")
        shutil.copy2(representation_path, building / "point_representation_mean_sd_v1.npy")
        metadata = {
            "status": "passed_primary_full_cohort_modelling_table_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_archive_sha256": actual_zip_sha256,
            "outcome_sha256": sha256(outcome_csv),
            "location_sha256": sha256(location_csv),
            "rows": 2886,
            "eye_visits": 78,
            "participants": 22,
            "representation_shape": [2886, 2048],
            "primary_scope": "all eligible Visit 1 and Visit 3 points; no boundary-availability exclusion",
            "checks": checks,
            "next_gate": "fit the five-model primary ladder using fixed participant-grouped nested validation",
        }
        (building / "PRIMARY_FULL_COHORT_MODELLING_TABLE_V1.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        (building / "README.md").write_text(
            "# Primary full-cohort modelling table v1\n\n"
            "One row per MAIA Point 1–37 from all eligible matched eye-visits. The point-level "
            "RETFound mean+variability representation is label-blind and joined only after the "
            "validated outcome and location tables are checked. This is the primary all-cohort input; "
            "the verified-boundary table is a secondary structural analysis.\n",
            encoding="utf-8",
        )
        files = sorted(path for path in building.iterdir() if path.is_file())
        pd.DataFrame(
            [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]
        ).to_csv(building / "OUTPUT_MANIFEST_V1.csv", index=False, lineterminator="\n")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        building.rename(output_root)
        return metadata
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if building.exists():
            shutil.rmtree(building, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-zip", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--location", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-embedding-sha256", default=EXPECTED_EMBEDDING_ZIP_SHA256)
    args = parser.parse_args()
    print(json.dumps(run(
        embedding_zip=args.embedding_zip,
        outcome_csv=args.outcome,
        location_csv=args.location,
        output_root=args.output_root,
        expected_embedding_sha256=args.expected_embedding_sha256,
    ), indent=2))


if __name__ == "__main__":
    main()
