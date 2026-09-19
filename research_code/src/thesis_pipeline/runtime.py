# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

"""Hash verification, safe archive restoration and command execution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def require_checks(checks: dict[str, bool], stage: str) -> None:
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise RuntimeError(f"{stage} failed: {failed}")


def safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"Unsafe archive member: {name}")
        archive.extractall(destination)


class FrozenStageStore:
    """Restore only archives that match the frozen registry exactly."""

    def __init__(self, drive_root: Path, workspace_root: Path, registry_path: Path):
        self.drive_root = Path(drive_root)
        self.workspace_root = Path(workspace_root)
        self.registry = json.loads(require_file(registry_path, "asset registry").read_text())

    def available(self, registry_stage: str) -> bool:
        specification = self.registry["frozen_stages"][registry_stage]
        return (self.drive_root / specification["drive_relative_path"]).is_file()

    def restore(self, registry_stage: str, destination: Path | None = None) -> dict:
        specification = self.registry["frozen_stages"][registry_stage]
        archive_path = self.drive_root / specification["drive_relative_path"]
        require_file(archive_path, registry_stage)
        require_checks(
            {
                "archive_size": archive_path.stat().st_size
                == int(specification["archive_bytes"]),
                "archive_sha256": sha256(archive_path)
                == specification["archive_sha256"],
            },
            f"{registry_stage} archive",
        )
        destination = self.workspace_root if destination is None else Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        safe_extract(archive_path, destination)
        failed = []
        for member in specification["members"]:
            path = destination / member["path"]
            if (
                not path.is_file()
                or path.stat().st_size != int(member["bytes"])
                or sha256(path) != member["sha256"]
            ):
                failed.append(member["path"])
        if failed:
            raise RuntimeError(f"Restored member verification failed: {failed[:5]}")
        return {
            "stage": registry_stage,
            "action": "reused_verified_drive_snapshot",
            "members_verified": len(specification["members"]),
            "archive_sha256": specification["archive_sha256"],
        }


class GeneratedStageCache:
    """Hash-verified cache for deterministic stages created by the notebook.

    Each cached stage is a ZIP plus a small JSON sidecar.  The sidecar binds the
    archive hash to the source/input fingerprint supplied by the stage runner,
    preventing stale predictions from being reused after data or code changes.
    """

    def __init__(self, drive_cache_root: Path, workspace_root: Path):
        self.drive_cache_root = Path(drive_cache_root)
        self.workspace_root = Path(workspace_root)
        self.drive_cache_root.mkdir(parents=True, exist_ok=True)

    def paths(self, stage_id: str) -> tuple[Path, Path]:
        return (
            self.drive_cache_root / f"{stage_id}.zip",
            self.drive_cache_root / f"{stage_id}.json",
        )

    def restore(self, stage_id: str, fingerprint: str) -> dict | None:
        archive_path, metadata_path = self.paths(stage_id)
        if not archive_path.is_file() or not metadata_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checks = {
            "stage_id": metadata.get("stage_id") == stage_id,
            "fingerprint": metadata.get("fingerprint") == fingerprint,
            "archive_bytes": archive_path.stat().st_size == metadata.get("archive_bytes"),
            "archive_sha256": sha256(archive_path) == metadata.get("archive_sha256"),
        }
        if not all(checks.values()):
            return None
        safe_extract(archive_path, self.workspace_root)
        return {
            "action": "reused_verified_generated_cache",
            "stage_id": stage_id,
            "fingerprint": fingerprint,
            "archive_sha256": metadata["archive_sha256"],
        }

    def save(self, stage_id: str, fingerprint: str, source_root: Path) -> dict:
        source_root = Path(source_root)
        if not source_root.is_dir():
            raise FileNotFoundError(f"Stage output directory is missing: {source_root}")
        archive_path, metadata_path = self.paths(stage_id)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{stage_id}.", suffix=".zip", dir=self.drive_cache_root
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for path in sorted(source_root.rglob("*")):
                    if path.is_file():
                        archive.write(path, arcname=str(path.relative_to(self.workspace_root)))
            digest = sha256(temporary)
            metadata = {
                "stage_id": stage_id,
                "fingerprint": fingerprint,
                "archive_bytes": temporary.stat().st_size,
                "archive_sha256": digest,
            }
            os.replace(temporary, archive_path)
            metadata_temporary = metadata_path.with_suffix(".json.tmp")
            metadata_temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            os.replace(metadata_temporary, metadata_path)
            return {"action": "rebuilt_validated_and_cached", **metadata}
        finally:
            if temporary.exists():
                temporary.unlink()


def run_script(script: Path, *arguments: object) -> None:
    command = [sys.executable, str(script), *(str(value) for value in arguments)]
    subprocess.run(command, check=True)
