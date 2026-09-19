# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Audit a Heidelberg NumPy boundary export without deserialising huge objects.

The supplied ``outputDict`` files are scalar object arrays backed by Python
pickles. Calling ``numpy.load(..., allow_pickle=True)`` would materialise many
gigabytes of arrays and millions of Python objects. This audit instead parses
the pickle opcode stream. Pickle opcodes are inspected but never executed.

The same command is reusable for Visit 1 and Visit 3.  The ordinary dictionary,
filename index and volume-border index are the required core.  A thickness
dictionary is an optional stronger cross-check: its absence does not prevent
the ordinary boundary stream from being structurally audited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickletools
import re
import shutil
import tempfile
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npformat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = ROOT / "outputs/point_alignment_cohort/frozen_v1/cohort_case_audit.csv"
BINARY_OPS = {"BINBYTES", "SHORT_BINBYTES", "BINSTRING", "SHORT_BINSTRING"}
STRING_OPS = {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def npy_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        version = npformat.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = npformat.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = npformat.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY version {version}: {path}")
        payload_offset = handle.tell()
    return {
        "npy_version": list(version),
        "shape": list(shape),
        "dtype": str(dtype),
        "fortran_order": bool(fortran),
        "payload_offset": payload_offset,
        "is_scalar_object": shape == () and dtype.kind == "O",
    }


def _recent_array_shape(history: deque[tuple[str, Any]], payload_size: int):
    values = list(history)
    for index in range(len(values) - 1, 1, -1):
        if values[index][0] != "TUPLE2":
            continue
        first = values[index - 2]
        second = values[index - 1]
        if first[0] not in {"BININT", "BININT1", "BININT2"}:
            continue
        if second[0] not in {"BININT", "BININT1", "BININT2"}:
            continue
        shape = (int(first[1]), int(second[1]))
        if shape[0] * shape[1] * 8 == payload_size:
            return shape
    return None


def scan_object_pickle(path: Path) -> dict[str, Any]:
    """Stream the pickle and count payload structure without unpickling it."""
    header = npy_header(path)
    if not header["is_scalar_object"]:
        raise ValueError(f"Expected a scalar object NPY: {path}")

    opcode_counts: Counter[str] = Counter()
    binary_sizes: Counter[int] = Counter()
    inferred_scan_shapes: Counter[tuple[int, int]] = Counter()
    strings: Counter[str] = Counter()
    stop_seen = False
    history: deque[tuple[str, Any]] = deque(maxlen=64)

    with path.open("rb") as handle:
        handle.seek(header["payload_offset"])
        for opcode, argument, _position in pickletools.genops(handle):
            opcode_counts[opcode.name] += 1
            if opcode.name in BINARY_OPS:
                size = len(argument)
                binary_sizes[size] += 1
                if size >= 1024 * 1024:
                    shape = _recent_array_shape(history, size)
                    if shape is not None:
                        inferred_scan_shapes[shape] += 1
            elif opcode.name in STRING_OPS:
                strings[str(argument)] += 1
            elif opcode.name == "STOP":
                stop_seen = True
            if opcode.name not in BINARY_OPS:
                history.append((opcode.name, argument))

    return {
        **header,
        "pickle_stop_seen": stop_seen,
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "binary_payload_sizes": [
            {"bytes": size, "count": count}
            for size, count in sorted(binary_sizes.items())
        ],
        "inferred_scan_shapes": [
            {"height": shape[0], "width": shape[1], "count": count}
            for shape, count in sorted(inferred_scan_shapes.items())
        ],
        "inferred_scan_payload_count": int(sum(inferred_scan_shapes.values())),
        "embedded_strings": dict(sorted(strings.items())),
    }


def first_difference(left: Path, right: Path) -> int | None:
    """Return the zero-based first differing byte, or None for identical files."""
    offset = 0
    with left.open("rb") as lhs, right.open("rb") as rhs:
        while True:
            left_block = lhs.read(4 * 1024 * 1024)
            right_block = rhs.read(4 * 1024 * 1024)
            if left_block != right_block:
                common = min(len(left_block), len(right_block))
                for index in range(common):
                    if left_block[index] != right_block[index]:
                        return offset + index
                return offset + common
            if not left_block and not right_block:
                return None
            offset += len(left_block)


def normalise_source_name(name: str) -> str:
    """Normalise only cosmetic subject-number zero padding and case."""
    value = Path(name).name.upper()
    match = re.match(r"^(\d+)(_.+)$", value)
    if match:
        value = f"{int(match.group(1)):03d}{match.group(2)}"
    return value


def locate_one(input_root: Path, pattern: str, *, exclude: str | None = None) -> Path:
    matches = sorted(
        path for path in input_root.glob(pattern)
        if path.is_file() and (exclude is None or exclude.lower() not in path.name.lower())
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern!r} in {input_root}, found {len(matches)}")
    return matches[0]


def locate_optional_one(input_root: Path, pattern: str) -> Path | None:
    matches = sorted(path for path in input_root.glob(pattern) if path.is_file())
    if len(matches) > 1:
        raise RuntimeError(f"Expected at most one {pattern!r} in {input_root}, found {len(matches)}")
    return matches[0] if matches else None


def load_indexes(filename_path: Path, borders_path: Path) -> tuple[list[str], np.ndarray]:
    filenames = np.load(filename_path, allow_pickle=False)
    borders = np.load(borders_path, allow_pickle=False)
    if filenames.ndim != 1 or filenames.dtype.kind not in {"U", "S"}:
        raise ValueError("Filename index must be a one-dimensional string array")
    if borders.ndim != 2 or borders.shape[1] != 2 or borders.dtype.kind not in {"i", "u"}:
        raise ValueError("Volume borders must be an integer N x 2 array")
    return [str(value) for value in filenames.tolist()], borders.astype(np.int64, copy=False)


def border_audit(borders: np.ndarray) -> dict[str, Any]:
    spans = borders[:, 1] - borders[:, 0]
    span_counts = Counter(int(value) for value in spans.tolist())
    return {
        "volume_count": int(len(borders)),
        "starts_at_zero": bool(len(borders) and borders[0, 0] == 0),
        "strictly_positive_spans": bool(len(borders) and np.all(spans > 0)),
        "contiguous": bool(len(borders) and np.array_equal(borders[1:, 0], borders[:-1, 1])),
        "final_scan_exclusive": int(borders[-1, 1]) if len(borders) else 0,
        "span_counts": {str(key): value for key, value in sorted(span_counts.items())},
    }


def source_coverage(filenames: list[str], existing_source_root: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    indexed_normalised = {normalise_source_name(name): name for name in filenames}
    rows: list[dict[str, Any]] = []
    if existing_source_root is None:
        for name in filenames:
            rows.append({"indexed_filename": name, "normalised_filename": normalise_source_name(name), "existing_source_match": ""})
        return {"existing_source_root": None, "existing_source_count": None, "normalised_match_count": None}, rows

    existing = sorted(
        path.name.removesuffix("_extracted")
        for path in existing_source_root.glob("*.E2E_extracted")
        if path.is_dir()
    )
    existing_normalised = {normalise_source_name(name): name for name in existing}
    for name in filenames:
        normalised = normalise_source_name(name)
        rows.append({
            "indexed_filename": name,
            "normalised_filename": normalised,
            "existing_source_match": existing_normalised.get(normalised, ""),
        })
    return {
        "existing_source_root": str(existing_source_root),
        "existing_source_count": len(existing),
        "normalised_match_count": len(set(indexed_normalised) & set(existing_normalised)),
        "indexed_without_existing_source": sorted(set(indexed_normalised) - set(existing_normalised)),
        "existing_source_without_index": sorted(set(existing_normalised) - set(indexed_normalised)),
    }, rows


def cohort_summary(cohort_csv: Path | None, cohort_visit: str | None) -> dict[str, Any]:
    if cohort_csv is None or cohort_visit is None:
        return {"cohort_csv": None, "cohort_visit": cohort_visit}
    rows: list[dict[str, str]] = []
    with cohort_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("oct_visit") == cohort_visit:
                rows.append(row)
    participants = {row["study_id"] for row in rows}
    points = sum(int(float(row.get("point_count", 0))) for row in rows)
    bscan_counts = Counter(int(float(row["n_bscans"])) for row in rows)
    return {
        "cohort_csv": str(cohort_csv),
        "cohort_visit": cohort_visit,
        "accepted_eye_visits": len(rows),
        "accepted_participants": len(participants),
        "accepted_points": points,
        "accepted_bscan_count_distribution": {str(key): value for key, value in sorted(bscan_counts.items())},
        "case_level_crosswalk_available_in_export": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def run_audit(
    input_root: Path,
    output_root: Path,
    visit_label: str,
    existing_source_root: Path | None,
    cohort_csv: Path | None,
    cohort_visit: str | None,
    replace: bool,
) -> dict[str, Any]:
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if output_root.exists() and not replace:
        raise FileExistsError(f"Output exists; rerun with --replace: {output_root}")

    ordinary = locate_one(input_root, "outputDict_*.npy", exclude="thickness")
    thickness = locate_optional_one(input_root, "outputDict_thickness_*.npy")
    filename_path = locate_one(input_root, "filenames_*.npy")
    borders_path = locate_one(input_root, "volume_borders_*.npy")
    filenames, borders = load_indexes(filename_path, borders_path)
    borders_result = border_audit(borders)

    ordinary_scan = scan_object_pickle(ordinary)
    thickness_scan = scan_object_pickle(thickness) if thickness is not None else None
    difference_offset = first_difference(ordinary, thickness) if thickness is not None else None
    coverage, coverage_rows = source_coverage(filenames, existing_source_root)
    cohort = cohort_summary(cohort_csv, cohort_visit)

    file_paths = [ordinary, filename_path, borders_path]
    if thickness is not None:
        file_paths.insert(1, thickness)
    file_rows = [
        {
            "role": (
                "ordinary_dictionary" if path == ordinary else
                "thickness_dictionary" if thickness is not None and path == thickness else
                "filename_index" if path == filename_path else
                "volume_border_index"
            ),
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in file_paths
    ]

    checks = {
        "three_core_files_present": ordinary.is_file() and filename_path.is_file() and borders_path.is_file(),
        "ordinary_is_scalar_object_npy": ordinary_scan["is_scalar_object"],
        "ordinary_pickle_reaches_stop": ordinary_scan["pickle_stop_seen"],
        "volume_borders_start_at_zero": borders_result["starts_at_zero"],
        "volume_borders_are_contiguous": borders_result["contiguous"],
        "volume_borders_have_positive_spans": borders_result["strictly_positive_spans"],
        "ordinary_scan_payload_count_matches_border_endpoint": ordinary_scan["inferred_scan_payload_count"] == borders_result["final_scan_exclusive"],
        "filename_values_are_unique": len(filenames) == len(set(filenames)),
    }
    if thickness_scan is not None:
        checks.update({
            "optional_thickness_is_scalar_object_npy": thickness_scan["is_scalar_object"],
            "optional_thickness_pickle_reaches_stop": thickness_scan["pickle_stop_seen"],
            "optional_thickness_scan_payload_count_matches_border_endpoint": thickness_scan["inferred_scan_payload_count"] == borders_result["final_scan_exclusive"],
            "ordinary_and_optional_thickness_are_not_duplicates": difference_offset is not None,
        })
    else:
        checks["ordinary_only_mode_declared"] = True
    hard_pass = all(checks.values())
    unresolved = []
    if len(filenames) != borders_result["volume_count"]:
        unresolved.append(
            f"filename index has {len(filenames)} entries but volume borders define {borders_result['volume_count']} volumes"
        )
    border_spans = {int(key) for key in borders_result["span_counts"]}
    cohort_bscans = {int(key) for key in cohort.get("accepted_bscan_count_distribution", {})}
    if cohort_bscans and border_spans != cohort_bscans:
        unresolved.append(
            f"boundary-export scan counts {sorted(border_spans)} do not exactly match accepted cohort scan counts {sorted(cohort_bscans)}"
        )
    unresolved.extend([
        "nested field order and anatomical layer names are not encoded as descriptive strings",
        "boundary coordinate units and thickness units require empirical confirmation",
        "human-correction provenance is supplied by supervisor statement rather than embedded metadata",
        "case-level mapping to privacy-minimised study IDs must be established without exposing identifiers",
    ])
    if thickness is None:
        unresolved.append(
            "optional thickness dictionary is absent: B0-B4 coordinates can be audited, but saved thickness channels cannot yet be checked against successive boundary differences"
        )

    if hard_pass and thickness is None:
        status = "structural_audit_passed_optional_thickness_crosscheck_pending"
    elif hard_pass:
        status = "structural_audit_passed_semantic_mapping_pending"
    else:
        status = "structural_audit_failed"

    result = {
        "status": status,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "visit_label": visit_label,
        "input_root": str(input_root),
        "input_was_modified": False,
        "file_inventory": file_rows,
        "filename_index_count": len(filenames),
        "volume_borders": borders_result,
        "ordinary_dictionary": ordinary_scan,
        "thickness_dictionary": thickness_scan,
        "thickness_dictionary_available": thickness is not None,
        "first_dictionary_difference_zero_based_byte": difference_offset,
        "source_coverage": coverage,
        "cohort_summary": cohort,
        "checks": checks,
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "unresolved_semantic_or_mapping_gates": unresolved,
        "next_gate": (
            "resolve filename/scan mapping and audit ordinary B0-B4 coordinates; "
            "add the optional thickness cross-check if that file becomes available"
        ),
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=".boundary_export_audit_", dir=output_root.parent))
    try:
        write_csv(building / "FILE_INVENTORY_V1.csv", file_rows)
        write_csv(building / "FILENAME_SOURCE_COVERAGE_V1.csv", coverage_rows)
        (building / "BOUNDARY_EXPORT_AUDIT_V1.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (building / "README.md").write_text(
            "# Heidelberg boundary export structural audit v1\n\n"
            "This is a read-only, memory-safe structural audit. It parses pickle opcodes but never "
            "executes or deserialises the supplied object dictionaries. A structural pass does not "
            "assign anatomical names to nested fields and does not authorise modelling.\n",
            encoding="utf-8",
        )
        if output_root.exists():
            previous = output_root.with_name(f".{output_root.name}.previous")
            if previous.exists():
                shutil.rmtree(previous)
            os.replace(output_root, previous)
            os.replace(building, output_root)
            shutil.rmtree(previous)
        else:
            os.replace(building, output_root)
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visit-label", required=True)
    parser.add_argument("--existing-source-root", type=Path)
    parser.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--cohort-visit")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = run_audit(
        input_root=args.input_root.resolve(),
        output_root=args.output_root.resolve(),
        visit_label=args.visit_label,
        existing_source_root=args.existing_source_root.resolve() if args.existing_source_root else None,
        cohort_csv=args.cohort_csv.resolve() if args.cohort_csv else None,
        cohort_visit=args.cohort_visit,
        replace=args.replace,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
