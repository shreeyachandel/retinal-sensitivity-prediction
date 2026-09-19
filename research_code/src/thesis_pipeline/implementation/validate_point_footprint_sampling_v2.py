# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Independently validate the 100 µm point-footprint sampling output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "outputs/point_footprint_sampling_v2"
KEYS = ["case_id", "point_number", "scan_rank"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> dict:
    report = json.loads((root / "FOOTPRINT_SAMPLING_V2.json").read_text(encoding="utf-8"))
    footprints = pd.read_csv(root / "point_scan_footprints_v2.csv")
    samples = pd.read_csv(root / "along_scan_samples_v2.csv")
    observed_counts = samples.groupby(KEYS).size().rename("observed_count")
    declared_counts = footprints.set_index(KEYS)["within_scan_sample_count"]
    checks = {
        "primary_report_passed": report.get("status") == "passed",
        "point_scan_rows_equal_8658": len(footprints) == 8_658,
        "unique_points_equal_2886": len(footprints[["case_id", "point_number"]].drop_duplicates()) == 2_886,
        "ranks_equal_1_2_3": bool(footprints.groupby(["case_id", "point_number"])["scan_rank"].agg(lambda x: sorted(map(int, x)) == [1, 2, 3]).all()),
        "sample_counts_match": observed_counts.equals(declared_counts.astype("int64")),
        "all_offsets_inside_50_um": bool(samples["offset_from_registered_centre_um"].abs().le(50.0 + 1e-7).all()),
        "sample_columns_inside_images": bool(samples["ascan_x_px"].ge(0).all()) and bool(samples.merge(footprints[KEYS + ["bscan_width_px"]], on=KEYS)["ascan_x_px"].lt(samples.merge(footprints[KEYS + ["bscan_width_px"]], on=KEYS)["bscan_width_px"]).all()),
        "multiple_samples_per_scan": int(observed_counts.min()) >= 2,
        "no_sensitivity_columns": not any("sensitivity" in column.lower() for column in [*footprints.columns, *samples.columns]),
        "output_hashes_match": all(sha256(root / name) == expected for name, expected in report["outputs"].items()),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "counts": report["counts"],
    }
    if result["status"] != "passed":
        raise RuntimeError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
