# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Extract label-blind RETFound embeddings from B0-B4-aligned local OCT tiles."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO_ID = "YukunZhou/RETFound_mae_natureOCT"
CHECKPOINT_FILENAME = "RETFound_mae_natureOCT.pth"
CHECKPOINT_SHA256 = "e9ff7864f40334885062953cb3894739d9fff52aaeb22c8b4f47d92270068d18"
CHECKPOINT_BYTES = 3_952_489_221
CONTEXT_WIDTH_MM = 0.300
VERTICAL_MARGIN_MM = 0.050
EMBEDDING_DIM = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("retfound_models_vit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def five_centres(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("within_scan_sample_rank")
    if len(frame) < 5:
        raise RuntimeError(
            f"Fewer than five native A-scan centres inside a 100 µm footprint: "
            f"{frame[['case_id','point_number','scan_rank']].iloc[0].to_dict()}"
        )
    positions = np.rint(np.linspace(0, len(frame) - 1, 5)).astype(int)
    selected = frame.iloc[positions].copy()
    selected["encoder_sample_rank"] = np.arange(1, 6)
    return selected


def build_manifest(feature_roots: list[Path], private_oct_root: Path) -> pd.DataFrame:
    blocks = []
    for root in feature_roots:
        frame = pd.read_csv(root / "boundary_along_scan_samples_v1.csv")
        required = {
            "case_id", "study_id", "eye", "maia_timepoint", "point_number", "scan_rank",
            "within_scan_sample_rank", "ascan_x_px", "source_volume", "source_image_path",
            "scale_x_mm_per_pixel", "scale_z_mm_per_pixel", "B0_row_px", "B4_row_px",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"Boundary-aligned tile fields missing from {root}: {missing}")
        blocks.append(frame)
    samples = pd.concat(blocks, ignore_index=True)
    rows = []
    for _, frame in samples.groupby(["case_id", "point_number", "scan_rank"], sort=True):
        rows.append(five_centres(frame))
    selected = pd.concat(rows, ignore_index=True)
    visit_folder = {"BL": "OCT_V1", "Y01": "OCT_V2"}
    selected["bscan_path"] = selected.apply(
        lambda row: str(
            private_oct_root / str(row.study_id) / visit_folder[str(row.maia_timepoint)]
            / str(row.eye) / "volume" / f"bscan_{int(row.bscan_index):03d}.png"
        ), axis=1,
    )
    selected.sort_values(["case_id", "point_number", "scan_rank", "encoder_sample_rank"], inplace=True)
    selected.reset_index(drop=True, inplace=True)
    keys = ["case_id", "point_number", "scan_rank", "encoder_sample_rank"]
    if selected[keys].duplicated().any():
        raise RuntimeError("Duplicate tile keys")
    if not selected.groupby(["case_id", "point_number", "scan_rank"]).size().eq(5).all():
        raise RuntimeError("Every point-B-scan must have five tiles")
    if not selected.groupby(["case_id", "point_number"])["scan_rank"].nunique().eq(3).all():
        raise RuntimeError("Every point must have three B-scans")
    forbidden = ("sensitivity", "outcome", "prediction", "residual")
    if any(any(term in column.lower() for term in forbidden) for column in selected.columns):
        raise RuntimeError("Outcome-related field entered label-blind RETFound manifest")
    return selected


def boundary_aligned_tensor(row, cache: dict[str, np.ndarray]) -> np.ndarray:
    path = str(row.bscan_path)
    if path not in cache:
        cache[path] = np.asarray(Image.open(path).convert("L"))
    image = cache[path]
    sx, sz = float(row.scale_x_mm_per_pixel), float(row.scale_z_mm_per_pixel)
    x = int(round(float(row.ascan_x_px)))
    half_width = max(1, int(round(CONTEXT_WIDTH_MM / sx / 2)))
    top = int(np.floor(float(row.B0_row_px) - VERTICAL_MARGIN_MM / sz))
    bottom = int(np.ceil(float(row.B4_row_px) + VERTICAL_MARGIN_MM / sz))
    left, right = x - half_width, x + half_width + 1
    pad_left, pad_right = max(0, -left), max(0, right - image.shape[1])
    pad_top, pad_bottom = max(0, -top), max(0, bottom - image.shape[0])
    crop = image[max(0, top):min(image.shape[0], bottom), max(0, left):min(image.shape[1], right)]
    crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        raise RuntimeError(f"Invalid boundary crop for {row.case_id} point {row.point_number}")
    finite = crop[np.isfinite(crop)]
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        raise RuntimeError("Degenerate OCT intensity range")
    crop = np.clip((crop.astype(np.float32) - low) / (high - low), 0, 1)
    physical_height = crop.shape[0] * sz
    physical_width = crop.shape[1] * sx
    scale = min(224 / max(physical_width, 1e-9), 224 / max(physical_height, 1e-9))
    out_w = max(1, min(224, int(round(physical_width * scale))))
    out_h = max(1, min(224, int(round(physical_height * scale))))
    resized = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((out_w, out_h), Image.Resampling.BILINEAR))
    canvas = np.zeros((224, 224), dtype=np.float32)
    y0, x0 = (224 - out_h) // 2, (224 - out_w) // 2
    canvas[y0:y0 + out_h, x0:x0 + out_w] = resized.astype(np.float32) / 255.0
    mean, sd = float(canvas.mean()), float(canvas.std())
    if sd <= 1e-8:
        raise RuntimeError("Degenerate preprocessed tile")
    canvas = (canvas - mean) / sd
    return np.repeat(canvas[None, :, :], 3, axis=0).astype(np.float32)


def load_model(models_vit_path: Path, checkpoint_dir: Path, token: str):
    import torch
    from huggingface_hub import hf_hub_download

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(hf_hub_download(
        repo_id=REPO_ID, filename=CHECKPOINT_FILENAME, token=token,
        local_dir=checkpoint_dir,
    ))
    if checkpoint_path.stat().st_size != CHECKPOINT_BYTES or sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise RuntimeError("RETFound checkpoint identity mismatch")
    models_vit = load_module(models_vit_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = models_vit.RETFound_mae(img_size=224, num_classes=5, drop_path_rate=0, global_pool=True)
    message = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
    allowed_exact = {"mask_token", "decoder_pos_embed", "norm.weight", "norm.bias"}
    bad_missing = sorted(set(message.missing_keys) - allowed_missing)
    bad_unexpected = sorted(k for k in message.unexpected_keys if k not in allowed_exact and not k.startswith(("decoder_", "decoder_blocks.")))
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: {bad_missing}, {bad_unexpected}")
    model.eval().cuda()
    return model, checkpoint_path


def infer(model, tensors: np.ndarray, batch_size: int) -> np.ndarray:
    import torch
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(tensors[start:start + batch_size]).cuda()
            latent = model.forward_features(batch)
            if latent.ndim == 3:
                latent = latent[:, 0]
            chunks.append(latent.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def run(feature_roots: list[Path], private_oct_root: Path, models_vit_path: Path,
        checkpoint_dir: Path, output_root: Path, staging_root: Path, token: str,
        batch_size: int) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for RETFound extraction")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(feature_roots, private_oct_root)
    manifest.to_csv(output_root / "boundary_aligned_tile_manifest_v1.csv", index=False, lineterminator="\n")
    model, checkpoint_path = load_model(models_vit_path, checkpoint_dir, token)
    case_arrays = []
    for number, (case_id, rows) in enumerate(manifest.groupby("case_id", sort=True), start=1):
        case_path = staging_root / f"{case_id}.npy"
        if case_path.is_file():
            embeddings = np.load(case_path, allow_pickle=False)
            if embeddings.shape != (len(rows), EMBEDDING_DIM) or not np.isfinite(embeddings).all():
                raise RuntimeError(f"Invalid resumable case: {case_id}")
        else:
            cache = {}
            tensors = np.stack([boundary_aligned_tensor(row, cache) for row in rows.itertuples(index=False)])
            embeddings = infer(model, tensors, batch_size)
            np.save(case_path, embeddings)
        case_arrays.append(embeddings)
        print(f"RETFound {number}/{manifest['case_id'].nunique()}: {case_id}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    tiles = np.concatenate(case_arrays).astype(np.float32)
    n_points = manifest.groupby(["case_id", "point_number"]).ngroups
    nested = tiles.reshape(n_points, 3, 5, EMBEDDING_DIM)
    scan_means = nested.mean(axis=2)
    representation = np.concatenate([
        scan_means.mean(axis=1),
        scan_means.std(axis=1, ddof=0),
        nested.std(axis=2, ddof=0).mean(axis=1),
    ], axis=1).astype(np.float32)
    point_rows = manifest.groupby(["case_id", "point_number"], sort=True).first().reset_index()[
        ["case_id", "study_id", "eye", "maia_timepoint", "point_number"]
    ]
    point_rows.insert(0, "embedding_row_index", np.arange(len(point_rows)))
    np.save(output_root / "boundary_aligned_point_representation_3072d_v1.npy", representation)
    point_rows.to_csv(output_root / "boundary_aligned_point_rows_v1.csv", index=False, lineterminator="\n")
    checks = {
        "five_centres_per_scan": bool(manifest.groupby(["case_id", "point_number", "scan_rank"]).size().eq(5).all()),
        "three_scans_per_point": bool(manifest.groupby(["case_id", "point_number"])["scan_rank"].nunique().eq(3).all()),
        "point_rows_match_representation": len(point_rows) == len(representation),
        "representation_3072d": representation.shape[1] == 3072,
        "all_finite": bool(np.isfinite(representation).all()),
        "checkpoint_verified": sha256(checkpoint_path) == CHECKPOINT_SHA256,
        "label_blind": True,
    }
    result = {
        "status": "passed_boundary_aligned_retfound_embeddings_v1" if all(checks.values()) else "failed",
        "cases": int(point_rows["case_id"].nunique()), "points": len(point_rows),
        "tiles": len(manifest), "representation_shape": list(representation.shape),
        "tile_rule": "five native A-scan centres inside the 100 µm footprint on each of three nearest B-scans",
        "crop_rule": "300 µm horizontal context; B0-to-B4 vertical complex plus 50 µm margins; physical aspect ratio preserved",
        "pooling": "mean and SD across scans plus mean within-scan SD",
        "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks),
    }
    (output_root / "BOUNDARY_ALIGNED_RETFOUND_EMBEDDINGS_V1.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"].startswith("failed"):
        raise RuntimeError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, action="append", required=True)
    parser.add_argument("--private-oct-root", type=Path, required=True)
    parser.add_argument("--models-vit", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(
        [p.resolve() for p in args.feature_root],
        args.private_oct_root.resolve(),
        args.models_vit.resolve(), args.checkpoint_dir.resolve(), args.output_root.resolve(),
        args.staging_root.resolve(), args.hf_token, args.batch_size,
    ), indent=2))


if __name__ == "__main__":
    main()
