# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

#!/usr/bin/env python3
"""Extract frozen RETFound embeddings for the accepted label-blind OCT cohort."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


REPO_ID = "YukunZhou/RETFound_mae_natureOCT"
CHECKPOINT_FILENAME = "RETFound_mae_natureOCT.pth"
EXPECTED_CHECKPOINT_SHA256 = "e9ff7864f40334885062953cb3894739d9fff52aaeb22c8b4f47d92270068d18"
EXPECTED_CHECKPOINT_BYTES = 3_952_489_221
EXPECTED_COMMIT = "ae9a9ecf37857cf47b8aa9f87cd6f710d75db287"
EXPECTED_PATCHES = 8_658
EXPECTED_POINTS = 2_886
EMBEDDING_DIMENSIONS = 1_024
PILOT_TOLERANCE = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_tensor(processed_rgb: np.ndarray) -> np.ndarray:
    image = processed_rgb.astype(np.float32) / 255.0
    for channel in range(3):
        mean = float(image[..., channel].mean())
        sd = float(image[..., channel].std())
        if not np.isfinite(sd) or sd <= 1e-8:
            raise RuntimeError("Per-image channel standard deviation is zero or invalid.")
        image[..., channel] = (image[..., channel] - mean) / sd
    return np.transpose(image, (2, 0, 1)).astype(np.float32)


def classify_checkpoint_differences(
    missing_keys: list[str], unexpected_keys: list[str]
) -> tuple[list[str], list[str]]:
    allowed_missing = {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
    allowed_unexpected_exact = {"mask_token", "decoder_pos_embed", "norm.weight", "norm.bias"}
    allowed_unexpected_prefixes = ("decoder_", "decoder_blocks.")
    critical_missing = sorted(set(missing_keys) - allowed_missing)
    critical_unexpected = sorted(
        key for key in unexpected_keys
        if key not in allowed_unexpected_exact and not key.startswith(allowed_unexpected_prefixes)
    )
    return critical_missing, critical_unexpected


def reconstruct_patch(row: Any, project_root: Path, destination: Path, cache: dict[str, np.ndarray]) -> None:
    relative_bscan = str(row.bscan_path)
    if relative_bscan not in cache:
        path = project_root / relative_bscan
        if not path.is_file():
            raise FileNotFoundError(path)
        cache[relative_bscan] = np.asarray(Image.open(path).convert("L"))
    source = cache[relative_bscan]
    x0, x1 = int(row.source_patch_x0_px), int(row.source_patch_x1_exclusive_px)
    if source.shape != (int(row.bscan_height_px), int(row.bscan_width_px)):
        raise RuntimeError(f"B-scan dimensions changed: {relative_bscan}")
    if not (0 <= x0 < x1 <= source.shape[1]):
        raise RuntimeError(f"Unexpected edge padding requirement: {row.case_id}, point {row.point_number}, rank {row.scan_rank}")
    patch = Image.fromarray(source[:, x0:x1]).resize((128, 496), Image.Resampling.BILINEAR)
    if array_sha256(np.asarray(patch)) != row.source_patch_pixel_sha256:
        raise RuntimeError(f"Reconstructed patch pixel hash mismatch: {row.case_id}, point {row.point_number}, rank {row.scan_rank}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    patch.save(destination)


def prepare_tensors(rows: pd.DataFrame, project_root: Path, preprocessing: Any) -> np.ndarray:
    tensors: list[np.ndarray] = []
    cache: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="retfound_reconstructed_", dir="/content" if Path("/content").is_dir() else None) as temp:
        temp_root = Path(temp)
        for index, row in enumerate(rows.itertuples(index=False)):
            patch_path = temp_root / f"patch_{index:04d}.png"
            reconstruct_patch(row, project_root, patch_path, cache)
            processed, metadata = preprocessing.preprocess(patch_path)
            if metadata["processed_rgb_sha256"] != row.processed_rgb_sha256:
                raise RuntimeError(f"Frozen preprocessing hash mismatch: {row.case_id}, point {row.point_number}, rank {row.scan_rank}")
            tensors.append(model_tensor(processed))
    return np.stack(tensors)


def infer(model: Any, tensors: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(tensors[start:start + batch_size]).cuda().float()
            latent = model.forward_features(batch).squeeze(1)
            chunks.append(latent.detach().cpu().numpy().astype(np.float32))
            del batch, latent
    return np.concatenate(chunks, axis=0)


def case_fingerprint(rows: pd.DataFrame) -> str:
    fields = [
        "case_id", "point_number", "scan_rank", "bscan_path", "source_patch_x0_px",
        "source_patch_x1_exclusive_px", "source_patch_sha256", "source_patch_pixel_sha256",
        "processed_rgb_sha256",
    ]
    return text_sha256(rows[fields].to_csv(index=False, lineterminator="\n"))


def load_completed_case(case_id: str, rows: pd.DataFrame, staging_root: Path) -> np.ndarray | None:
    array_path = staging_root / "cases" / f"{case_id}.npy"
    metadata_path = staging_root / "cases" / f"{case_id}.json"
    if not array_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embeddings = np.load(array_path, allow_pickle=False)
    valid = (
        metadata.get("status") == "complete"
        and metadata.get("case_id") == case_id
        and metadata.get("rows") == len(rows)
        and metadata.get("case_input_fingerprint") == case_fingerprint(rows)
        and metadata.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256
        and metadata.get("embedding_sha256") == sha256(array_path)
        and embeddings.shape == (len(rows), EMBEDDING_DIMENSIONS)
        and np.isfinite(embeddings).all()
    )
    if not valid:
        raise RuntimeError(f"Existing staging case is incomplete or incompatible: {case_id}")
    return embeddings


def save_completed_case(case_id: str, rows: pd.DataFrame, embeddings: np.ndarray, staging_root: Path) -> None:
    case_root = staging_root / "cases"
    case_root.mkdir(parents=True, exist_ok=True)
    array_path = case_root / f"{case_id}.npy"
    array_temp = case_root / f".{case_id}.npy.tmp"
    with array_temp.open("wb") as stream:
        np.save(stream, embeddings.astype(np.float32))
    os.replace(array_temp, array_path)
    metadata = {
        "status": "complete", "case_id": case_id, "rows": len(rows),
        "case_input_fingerprint": case_fingerprint(rows),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "embedding_sha256": sha256(array_path),
    }
    metadata_path = case_root / f"{case_id}.json"
    metadata_temp = case_root / f".{case_id}.json.tmp"
    metadata_temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(metadata_temp, metadata_path)


def run(
    bundle_root: Path,
    project_root: Path,
    output_root: Path,
    staging_root: Path,
    weights_dir: Path,
    token: str,
    batch_size: int = 4,
) -> dict[str, Any]:
    if output_root.exists():
        metadata_path = output_root / "RETFOUND_FULL_COHORT_EMBEDDINGS_V1.json"
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("status") == "passed":
                return existing
        raise FileExistsError(f"Output already exists but is not a validated result: {output_root}")
    if not token or len(token.strip()) < 10:
        raise RuntimeError("HF_TOKEN is required through Colab Secrets.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    bundle_metadata = json.loads((bundle_root / "BUNDLE_METADATA_V1.json").read_text(encoding="utf-8"))
    if bundle_metadata.get("retfound_commit") != EXPECTED_COMMIT:
        raise RuntimeError("RETFound commit mismatch.")
    manifest_path = bundle_root / "cohort/FULL_COHORT_PATCH_MANIFEST_LABEL_BLIND_V1.csv"
    manifest = pd.read_csv(manifest_path).sort_values(["case_id", "point_number", "scan_rank"]).reset_index(drop=True)
    forbidden = ("sensitivity", "outcome", "prediction", "residual", "mae", "rmse")
    if any(any(token_ in column.lower() for token_ in forbidden) for column in manifest.columns):
        raise RuntimeError("Outcome or performance field found in label-blind cohort manifest.")
    if len(manifest) != EXPECTED_PATCHES or manifest[["case_id", "point_number", "scan_rank"]].duplicated().any():
        raise RuntimeError("Full-cohort patch manifest coverage is invalid.")
    point_counts = manifest.groupby(["case_id", "point_number"])["scan_rank"].agg(list)
    if len(point_counts) != EXPECTED_POINTS or not point_counts.apply(lambda ranks: ranks == [1, 2, 3]).all():
        raise RuntimeError("Every point must have exactly scan ranks 1, 2, and 3.")

    preprocessing = load_module(bundle_root / "code/prepare_retfound_preprocessing_pilot_v1.py", "frozen_preprocessing")
    models_vit = load_module(bundle_root / "official_retfound/models_vit.py", "frozen_models_vit")
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(hf_hub_download(
        repo_id=REPO_ID, filename=CHECKPOINT_FILENAME, token=token,
        local_dir=weights_dir, local_dir_use_symlinks=False,
    ))
    if checkpoint_path.stat().st_size != EXPECTED_CHECKPOINT_BYTES or sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Official checkpoint hash or size mismatch.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch.manual_seed(20260823)
    torch.cuda.manual_seed_all(20260823)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = models_vit.RETFound_mae(img_size=224, num_classes=5, drop_path_rate=0, global_pool=True)
    message = model.load_state_dict(checkpoint["model"], strict=False)
    critical_missing, critical_unexpected = classify_checkpoint_differences(
        list(message.missing_keys), list(message.unexpected_keys)
    )
    if critical_missing or critical_unexpected:
        raise RuntimeError(f"Critical encoder checkpoint mismatch: {critical_missing}, {critical_unexpected}")
    model.eval().cuda()
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    pilot_rows = pd.read_csv(bundle_root / "pilot/pilot_embedding_rows_v1.csv")
    pilot_expected = np.load(bundle_root / "pilot/pilot_patch_embeddings_v1.npy", allow_pickle=False)
    pilot_keys = pilot_rows[["case_id", "point_number", "scan_rank", "source_patch_sha256", "processed_rgb_sha256"]]
    pilot_manifest = pilot_keys.merge(
        manifest, on=["case_id", "point_number", "scan_rank", "source_patch_sha256", "processed_rgb_sha256"],
        how="left", validate="one_to_one",
    )
    if len(pilot_manifest) != 48 or pilot_manifest["bscan_path"].isna().any():
        raise RuntimeError("Pilot rows do not map exactly to the full-cohort manifest.")
    pilot_observed = infer(model, prepare_tensors(pilot_manifest, project_root, preprocessing), 4)
    pilot_max_difference = float(np.max(np.abs(pilot_expected - pilot_observed)))
    if pilot_max_difference > PILOT_TOLERANCE:
        raise RuntimeError(f"Frozen pilot embeddings did not reproduce: {pilot_max_difference}")
    del pilot_observed

    staging_root.mkdir(parents=True, exist_ok=True)
    case_ids = manifest["case_id"].drop_duplicates().tolist()
    case_arrays: list[np.ndarray] = []
    resumed_cases = 0
    for case_number, case_id in enumerate(case_ids, start=1):
        rows = manifest.loc[manifest["case_id"].eq(case_id)].reset_index(drop=True)
        embeddings = load_completed_case(case_id, rows, staging_root)
        if embeddings is None:
            tensors = prepare_tensors(rows, project_root, preprocessing)
            embeddings = infer(model, tensors, batch_size)
            if embeddings.shape != (len(rows), EMBEDDING_DIMENSIONS) or not np.isfinite(embeddings).all():
                raise RuntimeError(f"Invalid embeddings for {case_id}")
            save_completed_case(case_id, rows, embeddings, staging_root)
            del tensors
            status = "calculated"
        else:
            resumed_cases += 1
            status = "resumed"
        case_arrays.append(embeddings)
        print(f"[{case_number:02d}/{len(case_ids)}] {case_id}: {status}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    patch_embeddings = np.concatenate(case_arrays, axis=0).astype(np.float32)
    if patch_embeddings.shape != (EXPECTED_PATCHES, EMBEDDING_DIMENSIONS):
        raise RuntimeError("Final patch embedding shape mismatch.")
    grouped = manifest.groupby(["case_id", "point_number"], sort=False, observed=True)
    point_rows = grouped.first().reset_index()[["case_id", "study_id", "eye", "maia_timepoint", "point_number"]]
    point_mean = patch_embeddings.reshape(EXPECTED_POINTS, 3, EMBEDDING_DIMENSIONS).mean(axis=1).astype(np.float32)
    point_sd = patch_embeddings.reshape(EXPECTED_POINTS, 3, EMBEDDING_DIMENSIONS).std(axis=1, ddof=0).astype(np.float32)
    point_representation = np.concatenate([point_mean, point_sd], axis=1).astype(np.float32)

    building = Path(tempfile.mkdtemp(prefix=".retfound_full_cohort_", dir=output_root.parent))
    try:
        np.save(building / "patch_embeddings_v1.npy", patch_embeddings)
        np.save(building / "point_embedding_mean_v1.npy", point_mean)
        np.save(building / "point_embedding_sd_v1.npy", point_sd)
        np.save(building / "point_representation_mean_sd_v1.npy", point_representation)
        patch_rows = manifest[["case_id", "study_id", "eye", "maia_timepoint", "point_number", "scan_rank", "source_patch_sha256", "source_patch_pixel_sha256", "processed_rgb_sha256"]].copy()
        patch_rows.insert(0, "embedding_row_index", np.arange(EXPECTED_PATCHES))
        patch_rows.to_csv(building / "patch_embedding_rows_v1.csv", index=False)
        point_rows.insert(0, "point_embedding_row_index", np.arange(EXPECTED_POINTS))
        point_rows.to_csv(building / "point_embedding_rows_v1.csv", index=False)
        checks = {
            "label_blind_manifest": True,
            "exact_patch_rows": len(manifest) == EXPECTED_PATCHES,
            "exact_point_rows": len(point_rows) == EXPECTED_POINTS,
            "three_scans_per_point": True,
            "checkpoint_identity_exact": True,
            "no_critical_encoder_key_mismatch": not critical_missing and not critical_unexpected,
            "pilot_reproduced_within_tolerance": pilot_max_difference <= PILOT_TOLERANCE,
            "patch_embeddings_shape_exact": patch_embeddings.shape == (EXPECTED_PATCHES, EMBEDDING_DIMENSIONS),
            "point_mean_shape_exact": point_mean.shape == (EXPECTED_POINTS, EMBEDDING_DIMENSIONS),
            "point_sd_shape_exact": point_sd.shape == (EXPECTED_POINTS, EMBEDDING_DIMENSIONS),
            "point_representation_shape_exact": point_representation.shape == (EXPECTED_POINTS, 2 * EMBEDDING_DIMENSIONS),
            "all_embeddings_finite": bool(np.isfinite(point_representation).all() and np.isfinite(patch_embeddings).all()),
        }
        checks = {name: bool(value) for name, value in checks.items()}
        result = {
            "status": "passed" if all(checks.values()) else "failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sensitivity_loaded": False, "model_performance_inspected": False,
            "retfound_commit": EXPECTED_COMMIT, "checkpoint_repo_id": REPO_ID,
            "checkpoint_filename": CHECKPOINT_FILENAME, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_bytes": EXPECTED_CHECKPOINT_BYTES, "architecture": "RETFound_mae ViT-Large",
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "batch_size": batch_size,
            "patch_rows": EXPECTED_PATCHES, "point_rows": EXPECTED_POINTS,
            "patch_embedding_shape": list(patch_embeddings.shape),
            "point_mean_shape": list(point_mean.shape), "point_sd_shape": list(point_sd.shape),
            "point_representation_shape": list(point_representation.shape),
            "pilot_reproduction_tolerance": PILOT_TOLERANCE,
            "maximum_pilot_embedding_difference": pilot_max_difference,
            "cases": len(case_ids), "cases_resumed_from_staging": resumed_cases,
            "checkpoint_missing_keys": list(message.missing_keys),
            "checkpoint_unexpected_keys": list(message.unexpected_keys),
            "checkpoint_critical_missing_keys": critical_missing,
            "checkpoint_critical_unexpected_keys": critical_unexpected,
            "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks),
            "next_gate": "independently validate and freeze point-level embeddings before loading sensitivity or fitting PCA/ridge",
        }
        (building / "RETFOUND_FULL_COHORT_EMBEDDINGS_V1.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (building / "README.md").write_text(
            "# RETFound full-cohort frozen embeddings v1\n\n"
            "Label-blind frozen-encoder outputs for 8,658 accepted patches and 2,886 MAIA points. "
            "Point representations concatenate the three-scan embedding mean and population SD. "
            "Sensitivity and model performance were not loaded.\n",
            encoding="utf-8",
        )
        files = [path for path in building.iterdir() if path.is_file()]
        pd.DataFrame([{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(files)]).to_csv(
            building / "OUTPUT_MANIFEST_V1.csv", index=False
        )
        if result["status"] != "passed":
            raise RuntimeError("Full-cohort embedding checks failed.")
        os.replace(building, output_root)
        return result
    except Exception:
        import shutil
        shutil.rmtree(building, ignore_errors=True)
        raise
