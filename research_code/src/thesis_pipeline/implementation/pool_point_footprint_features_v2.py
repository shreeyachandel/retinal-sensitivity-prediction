# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Pool per-A-scan features into one label-blind row per MAIA point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/point_footprint_sampling_v2/along_scan_samples_v2.csv"
IDENTITY = ["case_id", "point_number", "scan_rank", "within_scan_sample_rank"]
POINT_IDENTITY = ["case_id", "study_id", "eye", "maia_timepoint", "point_number"]
RESERVED = set(
    IDENTITY
    + POINT_IDENTITY
    + [
        "bscan_index", "bscan_path", "ascan_x_px",
        "offset_from_registered_centre_mm", "offset_from_registered_centre_um",
        "scale_x_mm_per_ascan", "stimulus_diameter_mm", "sensitivity_db",
    ]
)


def pool(manifest_path: Path, feature_path: Path, output_path: Path) -> dict:
    manifest = pd.read_csv(manifest_path)
    features = pd.read_csv(feature_path)
    missing = sorted(set(IDENTITY) - set(features.columns))
    if missing:
        raise ValueError(f"Feature table lacks identity columns: {missing}")
    if any("sensitivity" in column.lower() for column in features.columns):
        raise ValueError("Sensitivity is forbidden during footprint feature pooling")
    if features[IDENTITY].duplicated().any():
        raise ValueError("Feature table contains duplicate within-scan sample identities")
    numeric_features = [
        column for column in features.columns
        if column not in RESERVED and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not numeric_features:
        raise ValueError("No numeric sample features were supplied")
    joined = manifest.merge(features[IDENTITY + numeric_features], on=IDENTITY, how="left", validate="one_to_one")
    if joined[numeric_features].isna().any().any():
        raise ValueError("Feature table does not contain complete finite values for every footprint sample")
    if not np.isfinite(joined[numeric_features].to_numpy(dtype=float)).all():
        raise ValueError("Feature table contains non-finite values")

    scan_keys = POINT_IDENTITY + ["scan_rank"]
    scan_mean = joined.groupby(scan_keys, sort=True)[numeric_features].mean()
    scan_sd = joined.groupby(scan_keys, sort=True)[numeric_features].std(ddof=0)
    if len(scan_mean) != 8_658 or not (scan_mean.groupby(level=POINT_IDENTITY).size() == 3).all():
        raise RuntimeError("Expected exactly three pooled B-scans for each of 2,886 points")

    point_mean = scan_mean.groupby(level=POINT_IDENTITY).mean()
    point_between_scan_sd = scan_mean.groupby(level=POINT_IDENTITY).std(ddof=0)
    point_within_scan_sd = scan_sd.groupby(level=POINT_IDENTITY).mean()
    point_mean.columns = [f"{column}__scan_mean3" for column in numeric_features]
    point_between_scan_sd.columns = [f"{column}__between_scan_sd3" for column in numeric_features]
    point_within_scan_sd.columns = [f"{column}__mean_within_scan_sd3" for column in numeric_features]
    pooled = pd.concat([point_mean, point_between_scan_sd, point_within_scan_sd], axis=1).reset_index()
    if len(pooled) != 2_886 or pooled[["case_id", "point_number"]].duplicated().any():
        raise RuntimeError("Pooling did not produce exactly one row per MAIA point")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output_path, index=False, lineterminator="\n")
    report = {
        "status": "passed",
        "label_blind": True,
        "input_sample_rows": len(joined),
        "point_scan_rows": len(scan_mean),
        "point_rows": len(pooled),
        "input_features": numeric_features,
        "output_features_per_input": 3,
        "pooling": {
            "within_scan": "population mean and population SD over native A-scan samples",
            "across_three_scans": "mean and population SD of scan means, plus mean within-scan SD",
        },
        "modelling_unit": "one MAIA point",
    }
    (output_path.with_suffix(output_path.suffix + ".json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pool(args.manifest.resolve(), args.features.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
