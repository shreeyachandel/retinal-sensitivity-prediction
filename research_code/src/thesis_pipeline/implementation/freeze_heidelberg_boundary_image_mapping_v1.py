# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Freeze label-blind export-volume and scan-to-PNG mappings for either visit.

The ordinary boundary dictionary is parsed as a pickle opcode stream; it is
never deserialised. Every exported scan is represented by a standardised,
downsampled image fingerprint. Compatible export/source volume pairs are
scored by optimal one-to-one scan assignment, followed by a global one-to-one
volume assignment. No MAIA sensitivity or model output is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickletools
import shutil
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from audit_heidelberg_boundary_export_v1 import (
    _recent_array_shape,
    npy_header,
    normalise_source_name,
)


def fingerprint(array: np.ndarray, stride: int) -> np.ndarray:
    values = np.asarray(array[::stride, ::stride], dtype=np.float32).ravel()
    sd = float(values.std())
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return (values - float(values.mean())) / sd


def extract_export_groups(
    dictionary_path: Path, borders: np.ndarray, stride: int
) -> list[dict[str, Any]]:
    header = npy_header(dictionary_path)
    if not header["is_scalar_object"]:
        raise ValueError(f"Expected scalar object NPY: {dictionary_path}")
    spans = (borders[:, 1] - borders[:, 0]).astype(int)
    expected = int(borders[-1, 1])
    arrays: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
    history: deque[tuple[str, Any]] = deque(maxlen=64)
    with dictionary_path.open("rb") as handle:
        handle.seek(header["payload_offset"])
        for opcode, argument, _ in pickletools.genops(handle):
            if opcode.name not in {"BINBYTES", "BINBYTES8"}:
                history.append((opcode.name, argument))
                continue
            if not isinstance(argument, bytes) or len(argument) < 1024 * 1024:
                continue
            shape = _recent_array_shape(history, len(argument))
            if shape is None:
                continue
            image = np.frombuffer(argument, dtype="<f8").reshape(shape)
            arrays.append(fingerprint(image, stride))
            shapes.append(image.shape)
    if len(arrays) != expected:
        raise RuntimeError(f"Expected {expected} scan payloads, found {len(arrays)}")

    groups = []
    for group_index, (start, stop) in enumerate(borders.astype(int).tolist()):
        group_shapes = shapes[start:stop]
        if len(set(group_shapes)) != 1:
            raise RuntimeError(f"Mixed image shapes in export group {group_index}")
        groups.append(
            {
                "group_index": group_index,
                "shape": group_shapes[0],
                "matrix": np.stack(arrays[start:stop]),
                "scan_count": int(spans[group_index]),
            }
        )
    return groups


def load_source_volumes(source_root: Path, stride: int) -> list[dict[str, Any]]:
    volumes = []
    for case_dir in sorted(source_root.glob("*_extracted")):
        paths = sorted((case_dir / "volume_0").glob("*.png"))
        if not paths:
            continue
        matrices = []
        shapes = []
        kept_paths = []
        for path in paths:
            with Image.open(path) as image:
                array = np.asarray(image.convert("F"), dtype=np.float32)
            matrices.append(fingerprint(array, stride))
            shapes.append(tuple(array.shape))
            kept_paths.append(path)
        if len(set(shapes)) != 1:
            raise RuntimeError(f"Mixed source image shapes: {case_dir}")
        volumes.append(
            {
                "source_volume": case_dir.name.removesuffix("_extracted"),
                "shape": shapes[0],
                "matrix": np.stack(matrices),
                "paths": kept_paths,
            }
        )
    return volumes


def pair_assignment(export: dict[str, Any], source: dict[str, Any]):
    if export["shape"] != source["shape"]:
        return None
    left = export["matrix"]
    right = source["matrix"]
    similarity = (left @ right.T) / left.shape[1]
    export_rows, source_cols = linear_sum_assignment(-similarity)
    return {
        "score": float(similarity[export_rows, source_cols].mean()),
        "export_rows": export_rows,
        "source_cols": source_cols,
        "correlations": similarity[export_rows, source_cols],
    }


def filename_sequence_inference(
    filenames: list[str], assigned_pairs: list[dict[str, Any]], group_count: int
) -> dict[str, Any]:
    lookup = {normalise_source_name(name): index for index, name in enumerate(filenames)}
    evidence = []
    for row in assigned_pairs:
        index = lookup.get(normalise_source_name(row["source_volume"]))
        if index is not None:
            evidence.append((row["export_group_index"], index))
    candidates = []
    for omitted in range(len(filenames)):
        matches = 0
        for group_index, filename_index in evidence:
            expected = group_index if group_index < omitted else group_index + 1
            matches += int(filename_index == expected)
        candidates.append({"omitted_filename_index": omitted, "matches": matches})
    candidates.sort(key=lambda row: row["matches"], reverse=True)
    best_matches = candidates[0]["matches"] if candidates else 0
    best = [row for row in candidates if row["matches"] == best_matches]
    unique_complete = len(best) == 1 and best_matches == len(evidence)
    omitted = best[0]["omitted_filename_index"] if unique_complete else None
    inferred = []
    if omitted is not None:
        for group_index in range(group_count):
            filename_index = group_index if group_index < omitted else group_index + 1
            inferred.append(
                {
                    "export_group_index": group_index,
                    "filename_index": filename_index,
                    "indexed_filename": filenames[filename_index],
                }
            )
    return {
        "known_image_mapping_evidence_count": len(evidence),
        "unique_complete_single_omission_solution": unique_complete,
        "omitted_filename_index": omitted,
        "omitted_filename": filenames[omitted] if omitted is not None else None,
        "candidate_omissions": candidates,
        "inferred_group_filenames": inferred,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def run(
    dictionary_path: Path,
    filename_index_path: Path,
    borders_path: Path,
    source_root: Path,
    structural_audit_root: Path,
    output_root: Path,
    stride: int,
    replace: bool,
) -> dict[str, Any]:
    if output_root.exists() and not replace:
        raise FileExistsError(f"Output exists; use --replace: {output_root}")
    structural = json.loads(
        (structural_audit_root / "BOUNDARY_EXPORT_AUDIT_V1.json").read_text()
    )
    validation = json.loads(
        (structural_audit_root / "INDEPENDENT_VALIDATION_V1.json").read_text()
    )
    inventory = {row["role"]: row for row in structural["file_inventory"]}
    recorded_dictionary = (
        Path(structural["input_root"]) / inventory["ordinary_dictionary"]["filename"]
    ).resolve()
    if dictionary_path.resolve() != recorded_dictionary:
        raise RuntimeError("Dictionary differs from frozen structural audit path")
    if validation["status"] != "passed":
        raise RuntimeError("Independent structural validation did not pass")

    filenames = [
        str(value)
        for value in np.load(filename_index_path, allow_pickle=False).tolist()
    ]
    borders = np.load(borders_path, allow_pickle=False).astype(np.int64, copy=False)
    groups = extract_export_groups(dictionary_path, borders, stride)
    sources = load_source_volumes(source_root, stride)

    score_matrix = np.full((len(groups), len(sources)), -1e6, dtype=np.float64)
    pair_scores = []
    for group in groups:
        for source_index, source in enumerate(sources):
            result = pair_assignment(group, source)
            if result is None:
                continue
            score_matrix[group["group_index"], source_index] = result["score"]
            pair_scores.append(
                {
                    "export_group_index": group["group_index"],
                    "source_volume": source["source_volume"],
                    "assignment_mean_correlation": result["score"],
                }
            )

    source_by_name = {
        normalise_source_name(source["source_volume"]): index
        for index, source in enumerate(sources)
    }
    if len(filenames) == len(groups):
        # Visit 1 provides one filename for every exported block.  Use that
        # explicit identity first and leave blocks without an exact compatible
        # source unmatched (notably the 193-scan acquisitions).
        direct_pairs = []
        for group_index, filename in enumerate(filenames):
            source_index = source_by_name.get(normalise_source_name(filename))
            if source_index is not None and score_matrix[group_index, source_index] > -100:
                direct_pairs.append((group_index, source_index))
        group_rows = np.asarray([pair[0] for pair in direct_pairs], dtype=int)
        source_cols = np.asarray([pair[1] for pair in direct_pairs], dtype=int)
        assignment_mode = "exact indexed filename plus within-volume image assignment"
    else:
        # Visit 3 has one additional filename entry and therefore uses the
        # original global image assignment followed by sequence inference.
        group_rows, source_cols = linear_sum_assignment(-score_matrix)
        keep = score_matrix[group_rows, source_cols] > -100
        group_rows, source_cols = group_rows[keep], source_cols[keep]
        assignment_mode = "global image assignment plus filename-sequence inference"
    assigned_pairs = []
    scan_rows = []
    assigned_group_indexes = set()
    for group_index, source_index in zip(group_rows.tolist(), source_cols.tolist()):
        if score_matrix[group_index, source_index] < -100:
            raise RuntimeError("Global assignment used an incompatible shape")
        group = groups[group_index]
        source = sources[source_index]
        result = pair_assignment(group, source)
        compatible = score_matrix[group_index] > -100
        alternatives = np.sort(score_matrix[group_index, compatible])[::-1]
        second = float(alternatives[1]) if len(alternatives) > 1 else float("nan")
        rank = int(
            1 + np.sum(score_matrix[group_index, compatible] > result["score"] + 1e-12)
        )
        omitted = sorted(
            set(range(len(source["paths"]))) - set(result["source_cols"].tolist())
        )
        assigned_pairs.append(
            {
                "export_group_index": group_index,
                "source_volume": source["source_volume"],
                "export_scan_count": group["scan_count"],
                "source_scan_count": len(source["paths"]),
                "image_height": group["shape"][0],
                "image_width": group["shape"][1],
                "assignment_mean_correlation": result["score"],
                "within_group_candidate_rank": rank,
                "best_unconstrained_alternative_score": float(
                    np.max(score_matrix[group_index, compatible])
                ),
                "second_unconstrained_alternative_score": second,
                "omitted_source_images": "|".join(
                    source["paths"][index].name for index in omitted
                ),
            }
        )
        assigned_group_indexes.add(group_index)
        for export_position, source_position, correlation in zip(
            result["export_rows"], result["source_cols"], result["correlations"]
        ):
            source_path = source["paths"][int(source_position)]
            scan_rows.append(
                {
                    "export_group_index": group_index,
                    "export_scan_position": int(export_position),
                    "source_volume": source["source_volume"],
                    "source_image": source_path.name,
                    "source_scan_index": int(source_path.stem.split("_")[-1]),
                    "image_correlation": float(correlation),
                }
            )

    assigned_pairs.sort(key=lambda row: row["export_group_index"])
    scan_rows.sort(key=lambda row: (row["export_group_index"], row["export_scan_position"]))
    unmatched = [
        {
            "export_group_index": group["group_index"],
            "export_scan_count": group["scan_count"],
            "image_height": group["shape"][0],
            "image_width": group["shape"][1],
        }
        for group in groups
        if group["group_index"] not in assigned_group_indexes
    ]
    if len(filenames) == len(groups):
        sequence = {
            "known_image_mapping_evidence_count": len(assigned_pairs),
            "unique_complete_single_omission_solution": False,
            "omitted_filename_index": None,
            "omitted_filename": None,
            "candidate_omissions": [],
            "inferred_group_filenames": [
                {"export_group_index": index, "filename_index": index, "indexed_filename": name}
                for index, name in enumerate(filenames)
            ],
            "mode": "one indexed filename per export group",
        }
    else:
        sequence = filename_sequence_inference(filenames, assigned_pairs, len(groups))
    sequence_lookup = {
        row["export_group_index"]: row for row in sequence["inferred_group_filenames"]
    }
    for row in assigned_pairs:
        inferred = sequence_lookup.get(row["export_group_index"], {})
        row["filename_index"] = inferred.get("filename_index", "")
        row["indexed_filename"] = inferred.get("indexed_filename", "")
        row["indexed_filename_matches_image_mapping"] = bool(
            inferred
            and normalise_source_name(inferred["indexed_filename"])
            == normalise_source_name(row["source_volume"])
        )
    for row in unmatched:
        inferred = sequence_lookup.get(row["export_group_index"], {})
        row["filename_index"] = inferred.get("filename_index", "")
        row["inferred_indexed_filename"] = inferred.get("indexed_filename", "")

    checks = {
        "label_blind": True,
        "structural_audit_passed": structural["checks_passed"] == structural["checks_total"],
        "export_groups_match_volume_borders": len(groups) == len(borders),
        "source_volumes_present": len(sources) > 0,
        "at_least_one_export_volume_assigned": len(assigned_pairs) > 0,
        "assigned_source_volumes_unique": len(assigned_pairs) == len({r["source_volume"] for r in assigned_pairs}),
        "all_assigned_groups_unique": len(assigned_group_indexes) == len(assigned_pairs),
        "all_assigned_scan_rows_present": len(scan_rows) == sum(int(r["export_scan_count"]) for r in assigned_pairs),
        "filename_index_relationship_resolved": (
            len(filenames) == len(groups)
            or sequence["unique_complete_single_omission_solution"]
        ),
        "every_image_mapping_agrees_with_inferred_filename": all(r["indexed_filename_matches_image_mapping"] for r in assigned_pairs),
    }
    result = {
        "status": "volume_and_scan_mapping_frozen_semantic_layers_pending" if all(checks.values()) else "mapping_requires_review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sensitivity_loaded": False,
        "model_outputs_loaded": False,
        "method": assignment_mode,
        "fingerprint_stride": stride,
        "dictionary_path": str(dictionary_path.resolve()),
        "dictionary_sha256": inventory["ordinary_dictionary"]["sha256"],
        "filename_index_path": str(filename_index_path.resolve()),
        "volume_borders_path": str(borders_path.resolve()),
        "source_root": str(source_root.resolve()),
        "counts": {
            "export_groups": len(groups),
            "export_scans": sum(group["scan_count"] for group in groups),
            "existing_source_volumes": len(sources),
            "assigned_source_volumes": len(assigned_pairs),
            "unmatched_export_groups": len(unmatched),
            "saved_scan_mappings": len(scan_rows),
        },
        "filename_sequence_inference": sequence,
        "checks": checks,
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "minimum_assigned_volume_score": min((r["assignment_mean_correlation"] for r in assigned_pairs), default=float("nan")),
        "median_assigned_volume_score": float(np.median([r["assignment_mean_correlation"] for r in assigned_pairs])) if assigned_pairs else float("nan"),
        "minimum_assigned_scan_correlation": min((r["image_correlation"] for r in scan_rows), default=float("nan")),
        "next_gate": "decode nested boundary/thickness fields and validate their anatomical meaning on label-blind overlays",
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=".boundary_mapping_", dir=output_root.parent))
    try:
        write_csv(building / "VOLUME_MAPPING_V1.csv", assigned_pairs)
        write_csv(building / "SCAN_MAPPING_V1.csv", scan_rows)
        write_csv(building / "UNMATCHED_EXPORT_GROUPS_V1.csv", unmatched)
        write_csv(building / "ALL_COMPATIBLE_VOLUME_PAIR_SCORES_V1.csv", pair_scores)
        (building / "BOUNDARY_EXPORT_IMAGE_MAPPING_V1.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        (building / "README.md").write_text(
            "# Boundary export image mapping v1\n\n"
            "Label-blind mapping of exported scan blocks to existing source PNG volumes. "
            "This freezes geometry only; anatomical boundary identities remain pending.\n"
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
    parser.add_argument("--dictionary", type=Path, default=Path("Visit 3/outputDict_Usher_Visit_2.npy"))
    parser.add_argument("--filename-index", type=Path, default=Path("Visit 3/filenames_Usher_Visit_2.npy"))
    parser.add_argument("--volume-borders", type=Path, default=Path("Visit 3/volume_borders_Usher_Visit_2.npy"))
    parser.add_argument("--source-root", type=Path, default=Path("data/Visit_2_extracted"))
    parser.add_argument("--structural-audit-root", type=Path, default=Path("outputs/heidelberg_boundary_export_visit3_audit_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/heidelberg_boundary_export_visit3_mapping_v1"))
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = run(
        args.dictionary.resolve(), args.filename_index.resolve(),
        args.volume_borders.resolve(), args.source_root.resolve(),
        args.structural_audit_root.resolve(), args.output_root.resolve(),
        args.stride, args.replace,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "volume_and_scan_mapping_frozen_semantic_layers_pending":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
