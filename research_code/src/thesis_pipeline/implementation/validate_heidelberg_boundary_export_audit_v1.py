# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Independently validate a saved Heidelberg boundary-export audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(audit_root: Path) -> dict[str, Any]:
    manifest_path = audit_root / "BOUNDARY_EXPORT_AUDIT_V1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_root = Path(manifest["input_root"])
    inventory = {row["role"]: row for row in manifest["file_inventory"]}

    thickness_available = bool(manifest.get("thickness_dictionary_available", "thickness_dictionary" in inventory))
    expected_roles = {"ordinary_dictionary", "filename_index", "volume_border_index"}
    if thickness_available:
        expected_roles.add("thickness_dictionary")
    checks: dict[str, bool] = {
        "manifest_status_is_structural_pass": manifest["status"] in {
            "structural_audit_passed_semantic_mapping_pending",
            "structural_audit_passed_optional_thickness_crosscheck_pending",
        },
        "expected_inventory_roles_present": set(inventory) == expected_roles,
        "audit_checks_all_true": all(bool(value) for value in manifest["checks"].values()),
        "audit_check_count_consistent": manifest["checks_passed"] == manifest["checks_total"] == len(manifest["checks"]),
        "source_was_not_modified": manifest["input_was_modified"] is False,
        "semantic_gates_remain_explicit": len(manifest["unresolved_semantic_or_mapping_gates"]) >= 1,
        "coverage_csv_present": (audit_root / "FILENAME_SOURCE_COVERAGE_V1.csv").is_file(),
        "inventory_csv_present": (audit_root / "FILE_INVENTORY_V1.csv").is_file(),
    }

    file_results = []
    for role, row in inventory.items():
        path = input_root / row["filename"]
        exists = path.is_file()
        size_matches = exists and path.stat().st_size == row["bytes"]
        hash_matches = exists and sha256_file(path) == row["sha256"]
        file_results.append({"role": role, "exists": exists, "size_matches": size_matches, "sha256_matches": hash_matches})
    checks["all_source_files_exist"] = all(row["exists"] for row in file_results)
    checks["all_source_sizes_match"] = all(row["size_matches"] for row in file_results)
    checks["all_source_hashes_match"] = all(row["sha256_matches"] for row in file_results)

    filenames = np.load(input_root / inventory["filename_index"]["filename"], allow_pickle=False)
    borders = np.load(input_root / inventory["volume_border_index"]["filename"], allow_pickle=False)
    checks["filename_count_recalculated"] = len(filenames) == manifest["filename_index_count"]
    checks["volume_count_recalculated"] = len(borders) == manifest["volume_borders"]["volume_count"]
    checks["border_endpoint_recalculated"] = int(borders[-1, 1]) == manifest["volume_borders"]["final_scan_exclusive"]
    checks["ordinary_scan_count_matches_recalculated_endpoint"] = manifest["ordinary_dictionary"]["inferred_scan_payload_count"] == int(borders[-1, 1])
    if thickness_available:
        checks["thickness_scan_count_matches_recalculated_endpoint"] = manifest["thickness_dictionary"]["inferred_scan_payload_count"] == int(borders[-1, 1])
    else:
        checks["ordinary_only_mode_preserved"] = manifest["thickness_dictionary"] is None

    passed = all(checks.values())
    result = {
        "status": "passed" if passed else "failed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_root": str(audit_root),
        "file_results": file_results,
        "checks": checks,
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
    }
    (audit_root / "INDEPENDENT_VALIDATION_V1.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.audit_root.resolve())
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
