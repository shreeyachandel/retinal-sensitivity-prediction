# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

"""High-level stage calls used by the readable master notebook."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import generate as generate_error_analyses
from .config import PipelineConfig
from .reporting import generate as generate_figures
from .runtime import (
    GeneratedStageCache,
    FrozenStageStore,
    require_checks,
    require_file,
    safe_extract,
    sha256,
)


class ThesisPipeline:
    """A concise interface over the permanent, separately submitted source files."""

    def __init__(self, config: PipelineConfig):
        config.validate()
        self.config = config
        self.root = config.workspace_root
        self.output_root = config.run_output_root
        self.implementation = Path(__file__).resolve().parent / "implementation"
        self.store: FrozenStageStore | None = None
        self.cache: GeneratedStageCache | None = None
        self._manual_qc: dict[str, Path] = {}
        self._restored_frozen_stages: dict[str, dict] = {}
        self.run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        self.manifest = {
            "run_id": self.run_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage_modes": config.stage_modes,
            "stages": {},
        }

    def prepare(self, *, reset_workspace: bool = True) -> dict:
        if reset_workspace and self.root.exists():
            self._make_tree_writable(self.root)
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.store = FrozenStageStore(
            self.config.drive_root,
            self.root,
            self.config.registry_path,
        )
        self.cache = GeneratedStageCache(
            self.config.generated_cache_root,
            self.root,
        )
        if any(
            self.config.stage_modes[name] == "rebuild"
            for name in ("registration", "point_alignment", "point_level_dataset", "folds")
        ):
            self._manual_qc = self._stage_manual_qc_inputs()
        result = {
            "status": "ready",
            "workspace": str(self.root),
            "registry": str(self.config.registry_path),
            "run_id": self.run_id,
        }
        self.manifest["stages"]["setup"] = result
        return result

    def _require_store(self) -> FrozenStageStore:
        if self.store is None:
            raise RuntimeError("Run pipeline.prepare() first")
        return self.store

    def _record(self, stage: str, result: dict) -> dict:
        self.manifest["stages"][stage] = result
        return result

    def _require_cache(self) -> GeneratedStageCache:
        if self.cache is None:
            raise RuntimeError("Run pipeline.prepare() first")
        return self.cache

    def _fingerprint(self, *paths: Path, extra: str = "") -> str:
        digest = hashlib.sha256(extra.encode("utf-8"))
        for path in paths:
            path = require_file(path, "fingerprint input")
            digest.update(str(path.name).encode("utf-8"))
            digest.update(sha256(path).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _make_tree_writable(root: Path) -> None:
        """Allow a previous fresh workspace to be removed safely."""
        for path in sorted(Path(root).rglob("*"), reverse=True):
            path.chmod(0o700 if path.is_dir() else 0o600)
        Path(root).chmod(0o700)

    @staticmethod
    def _make_tree_read_only(root: Path) -> None:
        """Prevent implementation stages from changing a fixed input tree."""
        for path in Path(root).rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        Path(root).chmod(0o555)

    def _restore(self, stage: str, registry_stage: str) -> dict:
        mode = self.config.stage_modes[stage]
        if mode == "rebuild":
            raise RuntimeError(
                f"{stage} raw rebuild was requested. Restore the governed raw acquisition "
                "inputs first; human QC decisions are always reused, never regenerated."
            )
        if registry_stage in self._restored_frozen_stages:
            result = dict(self._restored_frozen_stages[registry_stage])
            result["action"] = "already_restored_and_verified_this_run"
        else:
            result = self._require_store().restore(registry_stage)
            self._restored_frozen_stages[registry_stage] = dict(result)
        return self._record(stage, result)

    def _stage_manual_qc_inputs(self) -> dict[str, Path]:
        """Copy fixed decisions and raw registration inputs into a clean workspace."""
        source = self.config.manual_qc_root / "registration_alignment"
        def first_existing(*candidates: Path) -> Path:
            return next((path for path in candidates if path.exists()), candidates[0])

        required = {
            "working_data": first_existing(
                source / "2026-07-14_deidentified_working_data",
                source / "working_data",
            ),
            "manual_registration": source / "registration_runs",
            "automatic_decisions": first_existing(
                source / "automatic_registration_batch/automatic_visual_review_decisions.csv",
                source / "automatic_visual_review_decisions.csv",
            ),
            "alignment_rules": first_existing(
                source / "post_registration_point_alignment_pilot",
                source / "point_alignment_pilot",
            ),
            "alignment_decisions": first_existing(
                source / "point_alignment_cohort/visual_review_queue_with_sheets.csv",
                source / "visual_review_queue_with_sheets.csv",
                source / "point_alignment_review_decisions.csv",
            ),
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "A clean rebuild requires these fixed manual/QC inputs under "
                f"{source}: "
                f"{', '.join(missing)}"
            )
        destinations = {
            "working_data": self.root / "outputs/2026-07-14_deidentified_working_data",
            "manual_registration": self.root / "outputs/registration_runs",
            "automatic_decisions": self.root / "outputs/automatic_registration_batch/automatic_visual_review_decisions.csv",
            "alignment_rules": self.root / "outputs/post_registration_point_alignment_pilot",
            "alignment_decisions": self.root / "_manual_qc/point_alignment_review_decisions.csv",
        }
        for name in ("working_data", "manual_registration", "alignment_rules"):
            shutil.copytree(required[name], destinations[name], dirs_exist_ok=True)
            self._make_tree_read_only(destinations[name])
        for name in ("automatic_decisions", "alignment_decisions"):
            destinations[name].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(required[name], destinations[name])
            destinations[name].chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return destinations

    def _run(self, command: list[str]) -> None:
        environment = dict(os.environ)
        environment["THESIS_PROJECT_ROOT"] = str(self.root)
        subprocess.run(command, check=True, cwd=self.root, env=environment)

    def _apply_alignment_decisions(self, queue_path: Path) -> Path:
        decisions = pd.read_csv(require_file(
            self._manual_qc["alignment_decisions"], "point-alignment QC decisions"
        ))
        required_columns = {"case_id", "human_decision", "decision_reason"}
        if not required_columns.issubset(decisions.columns):
            raise ValueError("Point-alignment QC decisions have unexpected columns")
        if decisions["case_id"].duplicated().any():
            raise ValueError("Point-alignment QC decisions contain duplicate case IDs")
        queue = pd.read_csv(require_file(queue_path, "generated point-alignment review queue"))
        if set(queue["case_id"]) != set(decisions["case_id"]):
            raise ValueError("Saved point-alignment decisions do not match the regenerated review queue")
        reviewed = queue.drop(columns=["human_decision", "decision_reason"]).merge(
            decisions[["case_id", "human_decision", "decision_reason"]],
            on="case_id", how="left", validate="one_to_one",
        )
        output = queue_path.with_name("visual_review_queue_with_sheets.csv")
        reviewed.to_csv(output, index=False, lineterminator="\n")
        return output

    def _stage1(self, stage: str) -> dict:
        return self._restore(stage, "registration_qc_alignment_and_folds")

    def registration(self) -> dict:
        mode = self.config.stage_modes["registration"]
        output_root = self.root / "outputs/automatic_registration_batch"
        if mode == "rebuild":
            self._run([
                sys.executable, str(self.implementation / "registration_automatic_batch.py"),
                "--output-root", str(output_root), "--overwrite",
            ])
            fingerprint = self._fingerprint(
                self._manual_qc["automatic_decisions"],
                self.root / "outputs/2026-07-14_deidentified_working_data/working_files/registration_manifest_frozen_v1.csv",
                self.implementation / "registration_automatic_batch.py",
                extra="automatic_registration_from_raw_inputs_and_fixed_qc",
            )
            result = self._require_cache().save("registration", fingerprint, output_root)
        else:
            result = self._stage1("registration")
        decisions = pd.read_csv(require_file(
            self.root / "outputs/automatic_registration_batch/automatic_visual_review_decisions.csv",
            "registration decisions",
        ))
        checks = {
            "automatic_decisions_73": len(decisions) == 73,
            "all_decisions_complete": decisions["final_transform"].notna().all(),
            "no_automatic_failures": not decisions["final_transform"].astype(str).eq("").any(),
        }
        require_checks(checks, "registration")
        result.update({"checks": checks, "automatic_cases": len(decisions)})
        return self._record("registration", result)

    def point_alignment(self) -> dict:
        mode = self.config.stage_modes["point_alignment"]
        output_root = self.root / "outputs/point_alignment_cohort"
        if mode == "rebuild":
            if output_root.exists():
                shutil.rmtree(output_root)
            self._run([
                sys.executable, str(self.implementation / "post_registration_point_alignment_cohort.py"),
                "--output-root", str(output_root),
            ])
            self._run([
                sys.executable, str(self.implementation / "validate_point_alignment_cohort.py"),
                "--output-root", str(output_root),
            ])
            self._apply_alignment_decisions(output_root / "visual_review_queue.csv")
            self._run([sys.executable, str(self.implementation / "freeze_point_alignment_cohort.py")])
            self._run([sys.executable, str(self.implementation / "validate_frozen_point_alignment_cohort.py")])
            fingerprint = self._fingerprint(
                self._manual_qc["automatic_decisions"],
                self._manual_qc["alignment_decisions"],
                self._manual_qc["alignment_rules"] / "point_number_mapping.json",
                self._manual_qc["alignment_rules"] / "patch_rule_frozen_pilot.json",
                self.implementation / "post_registration_point_alignment_cohort.py",
                extra="point_alignment_from_raw_inputs_and_fixed_qc",
            )
            result = self._require_cache().save("point_alignment", fingerprint, output_root)
        else:
            result = self._stage1("point_alignment")
        validation = json.loads(require_file(
            self.root / "outputs/point_alignment_cohort/frozen_v1/validation_report.json",
            "point-alignment validation",
        ).read_text())
        counts = validation["actual_counts"]
        checks = {
            "validation_passed": validation["validation_status"] == "passed",
            "eye_visits_78": counts["accepted_eye_visits"] == 78,
            "endpoint_points_2886": counts["endpoint_points"] == 2886,
            "three_scan_rows_8658": counts["patches"] == 8658,
        }
        require_checks(checks, "point alignment")
        result.update({"checks": checks, "counts": counts})
        return self._record("point_alignment", result)

    def point_level_dataset(self) -> dict:
        result = (
            {"action": "rebuilt_with_point_alignment"}
            if self.config.stage_modes["point_level_dataset"] == "rebuild"
            else self._stage1("point_level_dataset")
        )
        freeze = json.loads(require_file(
            self.root / "outputs/point_alignment_cohort/frozen_v1/freeze_metadata.json",
            "point-level dataset freeze",
        ).read_text())
        checks = {
            "frozen_for_modelling": freeze["status"] == "frozen_for_feature_extraction_and_participant_grouped_modelling",
            "points_2886": freeze["included_endpoint_points"] == 2886,
            "scan_mappings_8658": freeze["included_scan_mappings"] == 8658,
            "point0_separate": freeze["included_point0_rows_separate"] == 78,
        }
        require_checks(checks, "point-level dataset")
        result.update({"checks": checks, "counts": {
            "eye_visits": freeze["included_eye_visits"],
            "points": freeze["included_endpoint_points"],
            "point_scan_rows": freeze["included_scan_mappings"],
        }})
        return self._record("point_level_dataset", result)

    def folds(self) -> dict:
        mode = self.config.stage_modes["folds"]
        frozen_root = self.root / "outputs/point_alignment_cohort/frozen_v1"
        fold_root = frozen_root / "folds_v1"
        if mode == "rebuild":
            self._run([
                sys.executable, str(self.implementation / "create_participant_grouped_folds.py"),
                "--frozen-root", str(frozen_root), "--output-root", str(fold_root),
            ])
            self._run([
                sys.executable, str(self.implementation / "validate_participant_grouped_folds.py"),
                "--frozen-root", str(frozen_root), "--fold-root", str(fold_root),
            ])
            fingerprint = self._fingerprint(
                frozen_root / "cohort_case_audit.csv",
                self.implementation / "create_participant_grouped_folds.py",
                extra="participant_grouped_folds_from_rebuilt_cohort",
            )
            result = self._require_cache().save("folds", fingerprint, fold_root)
        else:
            result = self._stage1("folds")
        cohort = pd.read_csv(
            require_file(
                self.root / "outputs/point_alignment_cohort/frozen_v1/cohort_case_audit.csv",
                "accepted cohort audit",
            )
        )
        folds = pd.read_csv(
            require_file(
                self.root / "outputs/point_alignment_cohort/frozen_v1/folds_v1/participant_fold_assignments.csv",
                "participant folds",
            )
        )
        checks = {
            "eye_visits_78": len(cohort) == 78,
            "participants_22": cohort["study_id"].nunique() == 22,
            "five_outer_folds": sorted(folds["outer_fold"].unique()) == [1, 2, 3, 4, 5],
            "participant_in_one_fold": folds.groupby("study_id")["outer_fold"].nunique().eq(1).all(),
        }
        require_checks(checks, "participant folds")
        result.update({"checks": checks, "eye_visits": len(cohort), "participants": cohort["study_id"].nunique()})
        return self._record("folds", result)

    def registration_alignment_folds(self) -> dict:
        """Backward-compatible convenience call; the notebook shows the stages separately."""
        return {
            "registration": self.registration(),
            "point_alignment": self.point_alignment(),
            "point_level_dataset": self.point_level_dataset(),
            "folds": self.folds(),
        }

    def point_footprint_100um(self) -> dict:
        root = self.root / "outputs/point_footprint_sampling_v2"
        mode = self.config.stage_modes["point_footprint_100um"]
        mapping = self.root / "outputs/point_alignment_cohort/frozen_v1/point_scan_mapping_long.csv"
        builder = self.implementation / "build_point_footprint_sampling_v2.py"
        validator = self.implementation / "validate_point_footprint_sampling_v2.py"
        fingerprint_source = mapping
        if not fingerprint_source.is_file():
            seed_spec = self._require_store().registry["frozen_stages"]["point_footprint_100um"]
            fingerprint_source = self.config.drive_root / seed_spec["drive_relative_path"]
        fingerprint = self._fingerprint(
            fingerprint_source, builder, validator,
            extra="point_footprint_sampling_v2;diameter=0.100mm;three_nearest_scans",
        )
        if mode == "rebuild":
            self._restore_private_oct_transport()
            if root.exists():
                shutil.rmtree(root)
            self._run([
                sys.executable, str(builder),
                "--mapping", str(mapping),
                "--output-root", str(root),
                "--exclusions", str(self.root / "outputs/point_alignment_cohort/frozen_v1/excluded_eye_visits.csv"),
            ])
            self._run([sys.executable, str(validator), "--root", str(root)])
            cache = self._require_cache().save("point_footprint_100um", fingerprint, root)
            result = {"action": "rebuilt_validated_and_cached", "cache": cache}
        else:
            result = self._restore("point_footprint_100um", "point_footprint_100um")
        points = pd.read_csv(require_file(root / "point_scan_footprints_v2.csv", "point footprints"))
        samples = pd.read_csv(require_file(root / "along_scan_samples_v2.csv", "native samples"))
        checks = {
            "points_1_to_37": points["point_number"].between(1, 37).all(),
            "three_scans_per_point": points.groupby(["case_id", "point_number"]).size().eq(3).all(),
            "samples_inside_100um": samples["offset_from_registered_centre_mm"].abs().le(0.05 + 1e-12).all(),
        }
        require_checks(checks, "100 µm footprint")
        result.update({"checks": checks, "point_scan_rows": len(points), "native_sample_rows": len(samples)})
        return self._record("point_footprint_100um", result)

    def _restore_private_oct_transport(self) -> Path:
        """Restore the existing label-blind OCT transport used by prior Colab runs."""
        target = self.root / "outputs/2026-07-14_deidentified_working_data/OCT"
        if target.is_dir():
            return target
        specification = self._require_store().registry["existing_large_oct_transport"]
        legacy_root = self.config.drive_root.parent
        archive_path = self.root / specification["archive_name"]
        with archive_path.open("wb") as output:
            for part in specification["parts"]:
                path = require_file(legacy_root / part["name"], "existing OCT transport part")
                require_checks({
                    "part_size": path.stat().st_size == int(part["bytes"]),
                    "part_sha256": sha256(path) == part["sha256"],
                }, f"OCT transport {part['name']}")
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        require_checks({
            "archive_size": archive_path.stat().st_size == int(specification["archive_bytes"]),
            "archive_sha256": sha256(archive_path) == specification["archive_sha256"],
        }, "existing OCT transport")
        extraction = self.root / "_oct_transport"
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extraction)
        archive_path.unlink()
        candidates = list(extraction.rglob("2026-07-14_deidentified_working_data/OCT"))
        if len(candidates) != 1:
            raise RuntimeError("Could not locate private OCT tree in verified transport")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidates[0]), str(target))
        shutil.rmtree(extraction)
        return target

    def combined_boundary_features(self) -> dict:
        """Build or restore the audited B0-B4 features for both visits."""
        output_group = self.root / "outputs/combined_boundary_pipeline_v2"
        mode = self.config.stage_modes["combined_boundary_features"]
        raw_root = self.config.drive_root / "02_raw_boundary_sources"
        source_root = self.config.drive_root / "02_raw_oct_sources"
        required = [
            raw_root / "Visit_1/outputDict_Usher_Visit_1.npy",
            raw_root / "Visit_1/filenames_Usher_Visit_1.npy",
            raw_root / "Visit_1/volume_borders_Usher_Visit_1.npy",
            raw_root / "Visit_3/outputDict_Usher_Visit_2.npy",
            raw_root / "Visit_3/filenames_Usher_Visit_2.npy",
            raw_root / "Visit_3/volume_borders_Usher_Visit_2.npy",
        ]
        fingerprint = self._fingerprint(
            self.implementation / "freeze_heidelberg_boundary_image_mapping_v1.py",
            self.implementation / "audit_heidelberg_boundary_numeric_semantics_v1.py",
            self.implementation / "freeze_private_visit3_boundary_case_crosswalk_v1.py",
            self.implementation / "build_visit3_boundary_footprint_features_v1.py",
            self.implementation / "freeze_boundary_modelling_input_v2.py",
            extra="combined_boundary_pipeline_v2;human_verified_heidelberg_B0_B4;available_mapped_cases_only",
        )
        required_sample_columns = {
            "case_id", "study_id", "eye", "maia_timepoint", "point_number",
            "scan_rank", "within_scan_sample_rank", "ascan_x_px",
            "source_volume", "source_image_path", "scale_x_mm_per_pixel",
            "scale_z_mm_per_pixel", "B0_row_px", "B4_row_px",
        }

        cached = (
            self._require_cache().restore("combined_boundary_features_v2", fingerprint)
            if mode in {"auto", "reuse"}
            else None
        )
        if cached is not None:
            missing_or_incomplete = []
            for visit in ("visit1", "visit3"):
                sample_path = output_group / visit / "features" / "boundary_along_scan_samples_v1.csv"
                if not sample_path.is_file():
                    missing_or_incomplete.append(f"{visit}: sample-level table is absent")
                    continue
                columns = set(pd.read_csv(sample_path, nrows=0).columns)
                if not required_sample_columns.issubset(columns):
                    missing_or_incomplete.append(f"{visit}: required source-mapped columns are absent")
            if missing_or_incomplete:
                raise FileNotFoundError(
                    "The combined-boundary cache is incomplete for the secondary RETFound analysis "
                    f"({'; '.join(missing_or_incomplete)}). Set USE_DRIVE_CACHE=False to rebuild "
                    "the boundary audit and feature tables from the raw source files."
                )

        if cached is None and mode != "rebuild":
            raise FileNotFoundError(
                "No complete verified combined-boundary cache is available. Set "
                "USE_DRIVE_CACHE=False to rebuild the boundary audit and feature tables from "
                "the raw source files."
            )

        if cached is None:
            for path in required:
                require_file(path, "raw boundary rebuild input")
            private_oct = self._restore_private_oct_transport()
            cohort = self.root / "outputs/point_alignment_cohort/frozen_v1/cohort_case_audit.csv"
            footprint = self.root / "outputs/point_footprint_sampling_v2"
            settings = {
                "visit1": {
                    "folder": "Visit_1", "ordinary": "outputDict_Usher_Visit_1.npy",
                    "filenames": "filenames_Usher_Visit_1.npy", "borders": "volume_borders_Usher_Visit_1.npy",
                    "thickness": None, "source": "Visit_1_extracted", "oct": "OCT_V1", "timepoint": "BL",
                },
                "visit3": {
                    "folder": "Visit_3", "ordinary": "outputDict_Usher_Visit_2.npy",
                    "filenames": "filenames_Usher_Visit_2.npy", "borders": "volume_borders_Usher_Visit_2.npy",
                    "thickness": "outputDict_thickness_Usher_Visit_2.npy", "source": "Visit_2_extracted",
                    "oct": "OCT_V2", "timepoint": "Y01",
                },
            }
            for visit, item in settings.items():
                drive_input = raw_root / item["folder"]
                drive_original = source_root / item["source"]
                if not drive_original.is_dir():
                    raise FileNotFoundError(f"Missing original Heidelberg PNG source: {drive_original}")
                staging = self.root / "raw_boundary_staging" / visit
                input_root = staging / item["folder"]
                original = staging / item["source"]
                print(f"Staging {visit} boundary files from Drive to Colab local storage...", flush=True)
                shutil.copytree(drive_input, input_root)
                shutil.copytree(drive_original, original)
                roots = {
                    name: output_group / visit / name
                    for name in ("audit", "mapping", "crosswalk", "semantics", "features", "frozen")
                }
                commands = [
                    [sys.executable, str(self.implementation / "audit_heidelberg_boundary_export_v1.py"), "--input-root", str(input_root), "--output-root", str(roots["audit"]), "--visit-label", visit, "--existing-source-root", str(original), "--cohort-csv", str(cohort), "--cohort-visit", item["oct"], "--replace"],
                    [sys.executable, str(self.implementation / "validate_heidelberg_boundary_export_audit_v1.py"), "--audit-root", str(roots["audit"])],
                    [sys.executable, str(self.implementation / "freeze_heidelberg_boundary_image_mapping_v1.py"), "--dictionary", str(input_root / item["ordinary"]), "--filename-index", str(input_root / item["filenames"]), "--volume-borders", str(input_root / item["borders"]), "--source-root", str(original), "--structural-audit-root", str(roots["audit"]), "--output-root", str(roots["mapping"]), "--replace"],
                    [sys.executable, str(self.implementation / "freeze_private_visit3_boundary_case_crosswalk_v1.py"), "--raw-root", str(original), "--private-oct-root", str(private_oct), "--boundary-mapping-root", str(roots["mapping"]), "--cohort-audit", str(cohort), "--output-root", str(roots["crosswalk"]), "--oct-visit", item["oct"], "--maia-timepoint", item["timepoint"], "--replace"],
                ]
                semantics = [sys.executable, str(self.implementation / "audit_heidelberg_boundary_numeric_semantics_v1.py"), "--ordinary", str(input_root / item["ordinary"]), "--volume-borders", str(input_root / item["borders"]), "--mapping-root", str(roots["mapping"]), "--source-root", str(original), "--output-root", str(roots["semantics"]), "--replace"]
                if item["thickness"] and (input_root / item["thickness"]).is_file():
                    semantics.extend(["--thickness", str(input_root / item["thickness"])])
                commands.extend([
                    semantics,
                    [sys.executable, str(self.implementation / "build_visit3_boundary_footprint_features_v1.py"), "--ordinary", str(input_root / item["ordinary"]), "--volume-borders", str(input_root / item["borders"]), "--boundary-mapping-root", str(roots["mapping"]), "--private-crosswalk-root", str(roots["crosswalk"]), "--footprint-root", str(footprint), "--source-root", str(original), "--semantic-audit-root", str(roots["semantics"]), "--output-root", str(roots["features"]), "--replace"],
                    [sys.executable, str(self.implementation / "freeze_boundary_modelling_input_v2.py"), "--feature-root", str(roots["features"]), "--output-root", str(roots["frozen"]), "--visit-label", visit, "--replace"],
                ])
                for command in commands:
                    self._run(command)
                shutil.rmtree(staging)
            cached = self._require_cache().save("combined_boundary_features_v2", fingerprint, output_group)

        manifests = [json.loads((output_group / v / "frozen/BOUNDARY_MODELLING_INPUT_FREEZE_V2.json").read_text()) for v in ("visit1", "visit3")]
        result = {"status": "passed", "cache": cached, "visits": {m["visit_label"]: m["counts"] for m in manifests}, "same_feature_definition": manifests[0]["feature_columns"] == manifests[1]["feature_columns"]}
        require_checks({"both_visits_passed": all(m["status"] == "frozen_neutral_boundary_modelling_input_v2" for m in manifests), "same_feature_definition": result["same_feature_definition"]}, "combined boundary features")
        return self._record("combined_boundary_features", result)

    def visit1_boundary_audit(self) -> dict:
        registry = self._require_store().registry["raw_boundary_sources"]["visit1"]
        input_root = self.config.drive_root / "02_raw_boundary_sources/Visit_1"
        for role in ("ordinary_dictionary", "filename_index", "volume_border_index"):
            specification = registry[role]
            path = self.config.drive_root / specification["drive_relative_path"]
            require_file(path, f"Visit 1 {role}")
            require_checks(
                {
                    "bytes": path.stat().st_size == int(specification["bytes"]),
                    "sha256": sha256(path) == specification["sha256"],
                },
                f"Visit 1 {role}",
            )
        audit_root = self.root / "outputs/heidelberg_boundary_export_visit1_audit_v1"
        command = [
            sys.executable,
            str(self.implementation / "audit_heidelberg_boundary_export_v1.py"),
            "--input-root", str(input_root),
            "--output-root", str(audit_root),
            "--visit-label", "Visit 1 / baseline",
            "--cohort-csv", str(self.root / "outputs/point_alignment_cohort/frozen_v1/cohort_case_audit.csv"),
            "--cohort-visit", "OCT_V1",
            "--replace",
        ]
        subprocess.run(command, check=True)
        subprocess.run(
            [
                sys.executable,
                str(self.implementation / "validate_heidelberg_boundary_export_audit_v1.py"),
                "--audit-root", str(audit_root),
            ],
            check=True,
        )
        audit = json.loads((audit_root / "BOUNDARY_EXPORT_AUDIT_V1.json").read_text())
        result = {
            "status": audit["status"],
            "checks": audit["checks"],
            "thickness_dictionary_available": audit["thickness_dictionary_available"],
            "audit_root": str(audit_root),
        }
        return self._record("visit1_boundary_audit", result)

    def visit1_boundary_integration(self) -> dict:
        """Gate for the next automatic stage; no manual annotation is requested."""
        audit_path = self.root / "outputs/heidelberg_boundary_export_visit1_audit_v1/BOUNDARY_EXPORT_AUDIT_V1.json"
        audit = json.loads(require_file(audit_path, "Visit 1 audit manifest").read_text())
        accepted = {
            "structural_audit_passed_optional_thickness_crosscheck_pending",
            "structural_audit_passed_semantic_mapping_pending",
        }
        if audit["status"] not in accepted:
            raise RuntimeError(f"Visit 1 structural audit did not pass: {audit['status']}")
        result = {
            "status": "awaiting_automatic_mapping_implementation",
            "manual_review_required": False,
            "reason": (
                "The structural audit is implemented. The generic Visit 1 export-to-source "
                "scan mapping and 100 µm feature builder must be added and validated before "
                "the all-visit endpoint is run."
            ),
        }
        return self._record("visit1_boundary_integration", result)

    def visit3_boundary_features(self) -> dict:
        return self._restore("visit3_boundary_features", "visit3_neutral_boundaries")

    def feature_selection(self) -> dict:
        """Expose the prespecified, label-blind feature block before outcome joining."""
        freeze_root = self.root / "outputs/visit3_boundary_modelling_input_frozen_v1"
        manifest = json.loads(
            require_file(
                freeze_root / "VISIT3_BOUNDARY_MODELLING_INPUT_FREEZE_V1.json",
                "boundary feature freeze manifest",
            ).read_text()
        )
        features = list(manifest["feature_columns"])
        checks = {
            "feature_freeze_passed": manifest["status"] == "frozen_neutral_visit3_boundary_modelling_input_v1",
            "twelve_prespecified_features": len(features) == 12,
            "label_blind": manifest["model_performance_inspected_during_feature_freeze"] is False,
        }
        require_checks(checks, "feature selection")
        return self._record(
            "feature_selection",
            {
                "status": "passed",
                "checks": checks,
                "feature_columns": features,
                "selection_rule": (
                    "Use all prespecified neutral B0-B4 interval summaries; do not select "
                    "features using held-out sensitivity performance."
                ),
            },
        )

    def current_modelling_table(self) -> dict:
        table_root = self.root / "outputs/visit3_boundary_modelling_table_v1"
        store = self._require_store()
        if table_root.is_dir():
            subprocess.run([
                sys.executable, str(self.implementation / "validate_visit3_boundary_modelling_table_v1.py"),
                "--root", str(table_root),
            ], check=True)
            result = {"action": "already_curated_and_revalidated_this_run", "stage": "visit3_modelling_table"}
        elif store.available("visit3_modelling_table"):
            result = self._restore("combined_modelling_table", "visit3_modelling_table")
        else:
            mode = self.config.stage_modes["combined_modelling_table"]
            if mode == "reuse":
                raise FileNotFoundError("Visit 3 modelling table is absent and mode='reuse'")
            curation_root = self.config.drive_root / "02_raw_curation_sources/Visit_3_modelling_table"
            sources = {
                "outcome_join_audit_v1.csv": (263789, "491169e483b220e2919854018d28b0248fe37d4c625737514b18b4f4f2244d8e"),
                "INDEPENDENT_OUTCOME_JOIN_VALIDATION_V1.json": (1078, "79579aa59ce4b9978dc6590cc69f0e7107be717b98aab377ce97291b38c04738"),
                "registered_location_predictors_v1.csv": (286373, "e5c8263c59869afa1d891106d24c1e97b7bf005085d582aa529000647bb69e13"),
            }
            for name, (expected_bytes, expected_hash) in sources.items():
                source = require_file(curation_root / name, f"Visit 3 curation source {name}")
                require_checks({
                    "bytes": source.stat().st_size == expected_bytes,
                    "sha256": sha256(source) == expected_hash,
                }, f"Visit 3 curation source {name}")
            outcome_root = self.root / "outputs/outcome_join_v1"
            location_root = self.root / "outputs/bright_complex_modelling_table_v1"
            outcome_root.mkdir(parents=True, exist_ok=True)
            location_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(curation_root / "outcome_join_audit_v1.csv", outcome_root)
            shutil.copy2(curation_root / "INDEPENDENT_OUTCOME_JOIN_VALIDATION_V1.json", outcome_root)
            location_path = location_root / "registered_location_predictors_v1.csv"
            shutil.copy2(curation_root / location_path.name, location_path)
            subprocess.run([
                sys.executable, str(self.implementation / "build_visit3_boundary_modelling_table_v1.py"),
                "--boundary-root", str(self.root / "outputs/visit3_boundary_modelling_input_frozen_v1"),
                "--outcome-root", str(outcome_root),
                "--fold-root", str(self.root / "outputs/point_alignment_cohort/frozen_v1/folds_v1"),
                "--location-source", str(location_path),
                "--output-root", str(table_root),
            ], check=True)
            subprocess.run([
                sys.executable, str(self.implementation / "validate_visit3_boundary_modelling_table_v1.py"),
                "--root", str(table_root),
            ], check=True)
            result = {"action": "curated_from_verified_drive_sources", "stage": "visit3_modelling_table"}
        table = pd.read_csv(require_file(table_root / "visit3_boundary_modelling_table_v1.csv", "Visit 3 table"))
        checks = {
            "points_1369": len(table) == 1369,
            "eye_visits_37": table["case_id"].nunique() == 37,
            "participants_20": table["study_id"].nunique() == 20,
            "participant_grouped_folds": table.groupby("study_id")["outer_fold"].nunique().eq(1).all(),
        }
        require_checks(checks, "current Visit 3 modelling table")
        result.update({"checks": checks, "scope": "Visit 3 only; not final all-visit endpoint"})
        return self._record("combined_modelling_table", result)

    def model_ladder(self) -> dict:
        table_root = self.root / "outputs/visit3_boundary_modelling_table_v1"
        experiment = self.output_root / "visit3_boundary_experiment"
        protocol_root = experiment / "model_protocol"
        model_root = experiment / "models"
        scripts = self.implementation
        fingerprint = self._fingerprint(
            table_root / "visit3_boundary_modelling_table_v1.csv",
            table_root / "VISIT3_BOUNDARY_MODELLING_TABLE_V1.json",
            scripts / "freeze_visit3_boundary_model_protocol_v1.py",
            scripts / "run_visit3_boundary_models_v1.py",
            scripts / "validate_visit3_boundary_models_v1.py",
            extra=f"seed={self.config.random_seed};model_ladder_v1",
        )
        mode = self.config.stage_modes["model_ladder"]
        cached = self._require_cache().restore("visit3_boundary_model_ladder_v1", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is None and mode == "reuse":
            raise FileNotFoundError("No compatible cached model ladder was found in Drive")
        if cached is None:
            if experiment.exists():
                shutil.rmtree(experiment)
            subprocess.run([sys.executable, str(scripts / "freeze_visit3_boundary_model_protocol_v1.py"), "--table-root", str(table_root), "--output-root", str(protocol_root)], check=True)
            subprocess.run([sys.executable, str(scripts / "run_visit3_boundary_models_v1.py"), "--table-root", str(table_root), "--protocol-root", str(protocol_root), "--output-root", str(model_root)], check=True)
            subprocess.run([sys.executable, str(scripts / "validate_visit3_boundary_models_v1.py"), "--table-root", str(table_root), "--protocol-root", str(protocol_root), "--model-root", str(model_root)], check=True)
            cached = self._require_cache().save("visit3_boundary_model_ladder_v1", fingerprint, experiment)
        subprocess.run([sys.executable, str(scripts / "validate_visit3_boundary_models_v1.py"), "--table-root", str(table_root), "--protocol-root", str(protocol_root), "--model-root", str(model_root)], check=True)
        predictions = pd.read_csv(model_root / "oof_predictions_v1.csv")
        expected_models = [
            "null_training_mean", "null_minus_1db_reference", "location_ridge",
            "boundary_linear", "boundary_ridge", "location_boundary_ridge",
            "location_boundary_random_forest",
        ]
        require_checks({
            "seven_models": predictions["model_id"].nunique() == 7,
            "model_ladder_exact": predictions[["model_position", "model_id"]].drop_duplicates().sort_values("model_position")["model_id"].tolist() == expected_models,
            "one_prediction_per_point_model": not predictions[["model_id", "case_id", "point_number"]].duplicated().any(),
        }, "model ladder")
        result = {"status": "passed", "cache": cached, "models": expected_models, "prediction_rows": len(predictions), "model_root": str(model_root)}
        return self._record("model_ladder", result)

    def evaluation(self) -> dict:
        experiment = self.output_root / "visit3_boundary_experiment"
        model_root = experiment / "models"
        protocol_root = experiment / "evaluation_protocol"
        evaluation_root = experiment / "evaluation"
        scripts = self.implementation
        fingerprint = self._fingerprint(
            model_root / "oof_predictions_v1.csv",
            model_root / "PREDICTION_FREEZE_V1.json",
            scripts / "freeze_visit3_boundary_evaluation_protocol_v1.py",
            scripts / "evaluate_visit3_boundary_models_v1.py",
            scripts / "validate_visit3_boundary_evaluation_v1.py",
            extra="participant_bootstrap_5000;visit3_boundary_evaluation_v1",
        )
        mode = self.config.stage_modes["evaluation"]
        cached = self._require_cache().restore("visit3_boundary_evaluation_v1", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is None and mode == "reuse":
            raise FileNotFoundError("No compatible cached evaluation was found in Drive")
        if cached is None:
            subprocess.run([sys.executable, str(scripts / "freeze_visit3_boundary_evaluation_protocol_v1.py"), "--model-root", str(model_root), "--output-root", str(protocol_root)], check=True)
            subprocess.run([sys.executable, str(scripts / "evaluate_visit3_boundary_models_v1.py"), "--model-root", str(model_root), "--protocol-root", str(protocol_root), "--output-root", str(evaluation_root)], check=True)
            cached = self._require_cache().save("visit3_boundary_evaluation_v1", fingerprint, experiment)
        subprocess.run([sys.executable, str(scripts / "validate_visit3_boundary_evaluation_v1.py"), "--model-root", str(model_root), "--protocol-root", str(protocol_root), "--evaluation-root", str(evaluation_root)], check=True)
        metrics = pd.read_csv(evaluation_root / "all_point_metrics_v1.csv")
        result = {"status": "passed", "cache": cached, "models": len(metrics), "evaluation_root": str(evaluation_root), "primary_metrics": ["MAE", "RMSE", "Bland-Altman bias", "95% limits of agreement"]}
        return self._record("evaluation", result)

    def error_sensitivity_analyses(self) -> dict:
        experiment = self.output_root / "visit3_boundary_experiment"
        prediction_path = experiment / "models/oof_predictions_v1.csv"
        output_root = experiment / "error_sensitivity_analyses"
        fingerprint = self._fingerprint(
            prediction_path,
            Path(__file__).resolve().parent / "analysis.py",
            extra="error_sensitivity_analysis_v1",
        )
        mode = self.config.stage_modes["error_sensitivity_analysis"]
        cached = self._require_cache().restore("visit3_error_sensitivity_analysis_v1", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is None and mode == "reuse":
            raise FileNotFoundError("No compatible cached error/sensitivity analysis was found in Drive")
        if cached is None:
            result = generate_error_analyses(pd.read_csv(prediction_path), output_root)
            cached = self._require_cache().save("visit3_error_sensitivity_analysis_v1", fingerprint, experiment)
        else:
            result = {"status": "passed", "analysis_root": str(output_root)}
        result["cache"] = cached
        return self._record("error_sensitivity_analysis", result)

    def tables_figures(self) -> dict:
        experiment = self.output_root / "visit3_boundary_experiment"
        model_root = experiment / "models"
        evaluation_root = experiment / "evaluation"
        artifact_root = experiment / "thesis_artifacts"
        table_root = artifact_root / "tables"
        figure_root = artifact_root / "figures"
        analysis_root = experiment / "error_sensitivity_analyses"
        fingerprint = self._fingerprint(
            model_root / "oof_predictions_v1.csv",
            evaluation_root / "all_point_metrics_v1.csv",
            analysis_root / "participant_error_metrics.csv",
            analysis_root / "outer_fold_error_metrics.csv",
            analysis_root / "sensitivity_stratum_metrics.csv",
            Path(__file__).resolve().parent / "reporting.py",
            extra="thesis_tables_figures_v1",
        )
        mode = self.config.stage_modes["tables_figures"]
        cached = self._require_cache().restore("visit3_tables_figures_v1", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is not None:
            result = {"status": "passed", "cache": cached, "artifact_root": str(artifact_root)}
            return self._record("tables_figures", result)
        if mode == "reuse":
            raise FileNotFoundError("No compatible cached tables/figures were found in Drive")
        table_root.mkdir(parents=True, exist_ok=True)
        predictions = pd.read_csv(model_root / "oof_predictions_v1.csv")
        metrics = pd.read_csv(evaluation_root / "all_point_metrics_v1.csv")
        ranking = pd.read_csv(evaluation_root / "model_ranking_by_mae_v1.csv")
        metrics.to_csv(table_root / "model_performance.csv", index=False)
        ranking.to_csv(table_root / "model_ranking.csv", index=False)
        copied_analysis_tables = 0
        if analysis_root.is_dir():
            for path in sorted(analysis_root.glob("*.csv")):
                shutil.copy2(path, table_root / path.name)
                copied_analysis_tables += 1
        figures = generate_figures(predictions, metrics, figure_root) if self.config.generate_figures else {}
        cached = self._require_cache().save("visit3_tables_figures_v1", fingerprint, experiment)
        result = {"status": "passed", "cache": cached, "tables": 2 + copied_analysis_tables, "figures": len(figures), "artifact_root": str(artifact_root)}
        return self._record("tables_figures", result)

    def retfound_extension(self) -> dict:
        """Describe and gate the prespecified boundary-aligned RETFound extension."""
        result = {
            "status": "planned_not_part_of_current_seven_model_result",
            "same_cases_and_folds": True,
            "models_to_add": [
                "boundary_aligned_retfound_ridge",
                "location_boundary_retfound_ridge",
            ],
            "requirements": [
                "crop each local OCT tile relative to verified B0-B4 boundaries",
                "pool embeddings across within-footprint samples and three nearest B-scans",
                "fit/tune only inside the same participant-grouped nested folds",
                "evaluate from held-out OOF predictions using the same metrics",
            ],
            "gpu_required_for_embedding_extraction": True,
        }
        return self._record("retfound_extension", result)

    def _primary_embedding_archive(self) -> Path:
        """Rebuild or reuse the label-blind full-cohort RETFound representation."""
        mode = self.config.stage_modes["primary_modelling_table"]
        bundle = (
            self.config.manual_qc_root
            / "primary_retfound"
            / "retfound_full_cohort_colab_bundle_v1.zip"
        )
        if mode == "rebuild":
            return self._rebuild_primary_embeddings(bundle)
        alignment_mapping = (
            self.root
            / "outputs/point_alignment_cohort/frozen_v1/point_scan_mapping_long.csv"
        )
        if bundle.is_file() and alignment_mapping.is_file():
            cached = self._restore_primary_embedding_cache(bundle)
            if cached is not None:
                return cached
        candidates = [
            self.config.drive_root / "01_frozen_assets/full_cohort_retfound_embeddings_v1.zip",
            self.config.drive_root / "03_generated_stage_cache/full_cohort_retfound_embeddings_v1.zip",
            self.config.drive_root / "full_cohort_retfound_embeddings_v1.zip",
            self.config.drive_root.parent.parent / "retfound_full_cohort_embeddings_v1.zip",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "Missing full-cohort RETFound input. Supply the fixed preprocessing "
            "bundle under 01_manual_qc_inputs/primary_retfound/ or a reusable "
            "embedding archive under 01_frozen_assets/."
        )

    def _primary_embedding_fingerprint(self, bundle: Path) -> str:
        return self._fingerprint(
            bundle,
            self.root / "outputs/point_alignment_cohort/frozen_v1/point_scan_mapping_long.csv",
            self.implementation / "calculate_full_cohort_retfound_embeddings_v1.py",
            self.implementation / "prepare_retfound_preprocessing_pilot_v1.py",
            self.implementation / "official_retfound/models_vit.py",
            extra="primary_retfound_from_raw_oct_and_fixed_label_blind_preprocessing",
        )

    def _primary_embedding_zip(self, embedding_root: Path) -> Path:
        archive = self.root / "outputs/retfound_full_cohort_embeddings_v1.zip"
        if archive.exists():
            archive.unlink()
        shutil.make_archive(str(archive.with_suffix("")), "zip", embedding_root)
        return archive

    def _restore_primary_embedding_cache(self, bundle: Path) -> Path | None:
        fingerprint = self._primary_embedding_fingerprint(bundle)
        restored = self._require_cache().restore("primary_retfound_embeddings_v1", fingerprint)
        if restored is None:
            return None
        embedding_root = self.root / "outputs/retfound_full_cohort_embeddings_v1"
        require_file(
            embedding_root / "RETFOUND_FULL_COHORT_EMBEDDINGS_V1.json",
            "cached full-cohort RETFound validation record",
        )
        return self._primary_embedding_zip(embedding_root)

    def _rebuild_primary_embeddings(self, bundle: Path) -> Path:
        """Extract RETFound features from raw OCT using the fixed preprocessing bundle."""
        bundle = require_file(bundle, "fixed RETFound preprocessing bundle")
        fingerprint = self._primary_embedding_fingerprint(bundle)
        output_root = self.root / "outputs/retfound_full_cohort_embeddings_v1"
        staging_root = self.root / "_primary_retfound_staging"
        if output_root.exists():
            shutil.rmtree(output_root)
        if staging_root.exists():
            shutil.rmtree(staging_root)
        bundle_parent = self.root / "_primary_retfound_bundle"
        if bundle_parent.exists():
            shutil.rmtree(bundle_parent)
        bundle_parent.mkdir(parents=True)
        safe_extract(bundle, bundle_parent)
        bundle_roots = list(bundle_parent.rglob("BUNDLE_METADATA_V1.json"))
        if len(bundle_roots) != 1:
            raise RuntimeError("Fixed RETFound bundle must contain one BUNDLE_METADATA_V1.json")
        bundle_root = bundle_roots[0].parent
        replacements = {
            "code/calculate_full_cohort_retfound_embeddings_v1.py": self.implementation / "calculate_full_cohort_retfound_embeddings_v1.py",
            "code/prepare_retfound_preprocessing_pilot_v1.py": self.implementation / "prepare_retfound_preprocessing_pilot_v1.py",
            "official_retfound/models_vit.py": self.implementation / "official_retfound/models_vit.py",
        }
        for relative, source in replacements.items():
            target = bundle_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(require_file(source, "embedded RETFound implementation"), target)
        spec = importlib.util.spec_from_file_location(
            "full_cohort_retfound", bundle_root / "code/calculate_full_cohort_retfound_embeddings_v1.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the embedded RETFound extraction code")
        extractor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extractor)
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from google.colab import userdata  # type: ignore[import-not-found]
                token = userdata.get("HF_TOKEN")
            except ImportError:
                pass
        if not token:
            raise RuntimeError("Set the HF_TOKEN Colab secret before rebuilding RETFound embeddings")
        extractor.run(
            bundle_root=bundle_root,
            project_root=self.root,
            output_root=output_root,
            staging_root=staging_root,
            weights_dir=self.root / "_retfound_weights",
            token=token,
        )
        validation = json.loads(require_file(
            output_root / "RETFOUND_FULL_COHORT_EMBEDDINGS_V1.json",
            "full-cohort RETFound validation record",
        ).read_text())
        require_checks({"embedding_extraction_passed": validation.get("status") == "passed"}, "primary RETFound embeddings")
        self._require_cache().save("primary_retfound_embeddings_v1", fingerprint, output_root)
        return self._primary_embedding_zip(output_root)

    def _primary_curation_source(self, filename: str) -> Path:
        candidates = [
            self.config.drive_root / "02_raw_curation_sources/Visit_3_modelling_table" / filename,
            self.config.drive_root / "visit3_modelling_curation_inputs_v1" / filename,
            self.config.drive_root.parent.parent / "outputs/final_colab_asset_store_v1/visit3_modelling_curation_inputs_v1" / filename,
            self.config.drive_root.parent.parent / "outputs/outcome_join_v1" / filename,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Missing primary curation source: {filename}")

    def full_cohort_primary_modelling_table_v1(self) -> dict:
        """Build or restore the 2,886-row primary table without boundary filtering."""
        output_root = self.root / "outputs/primary_full_cohort_modelling_table_v1"
        embedding_zip = self._primary_embedding_archive()
        outcome = self._primary_curation_source("outcome_join_audit_v1.csv")
        location = self._primary_curation_source("registered_location_predictors_v1.csv")
        builder = self.implementation / "build_primary_full_cohort_modelling_table_v1.py"
        folds = self.root / "outputs/point_alignment_cohort/frozen_v1/folds_v1/case_fold_assignments.csv"
        fingerprint = self._fingerprint(
            embedding_zip, outcome, location, folds, builder,
            extra="primary_full_cohort_modelling_table_v1;all_eligible_points;no_boundary_exclusion",
        )
        mode = self.config.stage_modes["primary_modelling_table"]
        cached = self._require_cache().restore("primary_full_cohort_modelling_table_v1", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is None and mode == "reuse":
            raise FileNotFoundError("No compatible primary full-cohort modelling-table cache exists")
        if cached is None:
            if output_root.exists():
                shutil.rmtree(output_root)
            command = [
                sys.executable, str(builder), "--embedding-zip", str(embedding_zip),
                "--outcome", str(outcome), "--location", str(location),
                "--output-root", str(output_root),
            ]
            if self.config.stage_modes["primary_modelling_table"] == "rebuild":
                command.extend(["--expected-embedding-sha256", ""])
            subprocess.run(command, check=True, cwd=self.root)
            cached = self._require_cache().save("primary_full_cohort_modelling_table_v1", fingerprint, output_root)
        manifest = json.loads(require_file(output_root / "PRIMARY_FULL_COHORT_MODELLING_TABLE_V1.json", "primary table manifest").read_text())
        require_checks({
            "primary_table_passed": manifest["status"] == "passed_primary_full_cohort_modelling_table_v1",
            "all_points_2886": manifest["rows"] == 2886,
            "all_eye_visits_78": manifest["eye_visits"] == 78,
            "participants_22": manifest["participants"] == 22,
        }, "primary full-cohort modelling table")
        manifest["cache"] = cached
        return self._record("primary_modelling_table", manifest)

    def full_cohort_primary_model_ladder_v1(self) -> dict:
        """Fit the all-eligible-point primary five-model ladder."""
        table_root = self.root / "outputs/primary_full_cohort_modelling_table_v1"
        output_root = self.output_root / "primary_full_cohort_model_ladder_v1"
        script = self.implementation / "run_primary_full_cohort_models_v1.py"
        fingerprint = self._fingerprint(
            table_root / "primary_full_cohort_modelling_rows_v1.csv",
            table_root / "point_representation_mean_sd_v1.npy",
            script,
            extra=f"seed={self.config.random_seed};primary_five_models;grouped_nested_validation",
        )
        mode = self.config.stage_modes["primary_model_ladder"]
        cached = self._require_cache().restore("primary_full_cohort_model_ladder_v1", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is None and mode == "reuse":
            raise FileNotFoundError("No compatible primary full-cohort model cache exists")
        if cached is None:
            if output_root.exists():
                shutil.rmtree(output_root)
            subprocess.run([
                sys.executable, str(script),
                "--table", str(table_root / "primary_full_cohort_modelling_rows_v1.csv"),
                "--representations", str(table_root / "point_representation_mean_sd_v1.npy"),
                "--output-root", str(output_root), "--seed", str(self.config.random_seed),
            ], check=True, cwd=self.root)
            cached = self._require_cache().save("primary_full_cohort_model_ladder_v1", fingerprint, output_root)
        manifest = json.loads(require_file(output_root / "PRIMARY_FULL_COHORT_MODEL_RUN_V1.json", "primary model manifest").read_text())
        require_checks({
            "primary_oof_passed": manifest["status"] == "passed_primary_full_cohort_oof_v1",
            "five_models": len(manifest["models"]) == 5,
            "points_2886": manifest["counts"]["points"] == 2886,
        }, "primary model ladder")
        manifest["cache"] = cached
        return self._record("primary_model_ladder", manifest)

    def full_cohort_primary_evaluation_v1(self) -> dict:
        """Calculate primary error, calibration and Bland–Altman agreement metrics."""
        model_root = self.output_root / "primary_full_cohort_model_ladder_v1"
        output_root = self.output_root / "primary_full_cohort_evaluation_v1"
        script = self.implementation / "evaluate_primary_full_cohort_v1.py"
        fingerprint = self._fingerprint(
            model_root / "oof_predictions_primary_v1.csv", script,
            extra="primary_full_cohort_evaluation_v1;participant_bootstrap_5000;error_and_agreement",
        )
        mode = self.config.stage_modes["primary_evaluation"]
        cached = self._require_cache().restore("primary_full_cohort_evaluation_v1", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is None and mode == "reuse":
            raise FileNotFoundError("No compatible primary evaluation cache exists")
        if cached is None:
            if output_root.exists():
                shutil.rmtree(output_root)
            subprocess.run([
                sys.executable, str(script), "--predictions", str(model_root / "oof_predictions_primary_v1.csv"),
                "--output-root", str(output_root), "--bootstrap-repetitions", "5000",
            ], check=True, cwd=self.root)
            cached = self._require_cache().save("primary_full_cohort_evaluation_v1", fingerprint, output_root)
        manifest = json.loads(require_file(output_root / "PRIMARY_FULL_COHORT_EVALUATION_V1.json", "primary evaluation manifest").read_text())
        require_checks({"primary_evaluation_passed": manifest["status"] == "passed_primary_full_cohort_error_agreement_evaluation_v1"}, "primary evaluation")
        manifest["cache"] = cached
        return self._record("primary_evaluation", manifest)

    def full_cohort_primary_reports_v1(self) -> dict:
        """Generate primary tables, figures and frozen-prediction diagnostics."""
        model_root = self.output_root / "primary_full_cohort_model_ladder_v1"
        evaluation_root = self.output_root / "primary_full_cohort_evaluation_v1"
        artifact_root = self.output_root / "primary_full_cohort_reports_v1"
        tables = artifact_root / "tables"
        figures = artifact_root / "figures"
        predictions_path = model_root / "oof_predictions_primary_v1.csv"
        metrics_path = evaluation_root / "all_point_metrics_primary_v1.csv"
        fingerprint = self._fingerprint(
            predictions_path, metrics_path,
            Path(__file__).resolve().parent / "analysis.py",
            Path(__file__).resolve().parent / "reporting.py",
            extra="primary_full_cohort_reports_v1;code_generated_tables_figures",
        )
        mode = self.config.stage_modes["primary_reports"]
        cached = self._require_cache().restore("primary_full_cohort_reports_v1", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is not None:
            return self._record("primary_reports", {"status": "passed", "cache": cached, "artifact_root": str(artifact_root)})
        if mode == "reuse":
            raise FileNotFoundError("No compatible primary report cache exists")
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        tables.mkdir(parents=True, exist_ok=True)
        predictions = pd.read_csv(predictions_path)
        metrics = pd.read_csv(metrics_path)
        shutil.copy2(metrics_path, tables)
        shutil.copy2(evaluation_root / "visit_and_floor_stratum_metrics_primary_v1.csv", tables)
        shutil.copy2(evaluation_root / "model_ranking_by_mae_primary_v1.csv", tables)
        analysis = generate_error_analyses(predictions, artifact_root / "error_sensitivity_analyses")
        figure_outputs = generate_figures(predictions, metrics, figures) if self.config.generate_figures else {}
        cached = self._require_cache().save("primary_full_cohort_reports_v1", fingerprint, artifact_root)
        result = {"status": "passed", "cache": cached, "tables": 3, "figures": len(figure_outputs), "analysis": analysis, "artifact_root": str(artifact_root)}
        return self._record("primary_reports", result)

    def combined_modelling_table_v2(self) -> dict:
        """Join both visits' neutral predictors to outcomes, location and fixed folds."""
        boundary_root = self.root / "outputs/combined_boundary_pipeline_v2"
        output_root = self.root / "outputs/combined_boundary_modelling_table_v2"
        curation = self.config.drive_root / "02_raw_curation_sources/Visit_3_modelling_table"
        outcome = require_file(curation / "outcome_join_audit_v1.csv", "pointwise outcome join")
        location = require_file(curation / "registered_location_predictors_v1.csv", "registered location predictors")
        folds = self.root / "outputs/point_alignment_cohort/frozen_v1/folds_v1/case_fold_assignments.csv"
        fingerprint = self._fingerprint(
            boundary_root / "visit1/frozen/boundary_predictors_frozen_v2.csv",
            boundary_root / "visit1/frozen/BOUNDARY_MODELLING_INPUT_FREEZE_V2.json",
            boundary_root / "visit3/frozen/boundary_predictors_frozen_v2.csv",
            boundary_root / "visit3/frozen/BOUNDARY_MODELLING_INPUT_FREEZE_V2.json",
            outcome, location, folds,
            self.implementation / "build_combined_boundary_modelling_table_v2.py",
            extra="combined_boundary_modelling_table_v2;available_mapped_cases_only",
        )
        mode = self.config.stage_modes["combined_modelling_table"]
        cached = self._require_cache().restore("combined_boundary_modelling_table_v2", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is None and mode == "reuse":
            raise FileNotFoundError("No compatible combined modelling-table cache exists")
        if cached is None:
            command = [
                sys.executable, str(self.implementation / "build_combined_boundary_modelling_table_v2.py"),
                "--boundary-root", str(boundary_root / "visit1/frozen"),
                "--boundary-root", str(boundary_root / "visit3/frozen"),
                "--outcome", str(outcome), "--location", str(location), "--folds", str(folds),
                "--output-root", str(output_root), "--replace",
            ]
            subprocess.run(command, check=True)
            cached = self._require_cache().save("combined_boundary_modelling_table_v2", fingerprint, output_root)
        manifest = json.loads((output_root / "COMBINED_BOUNDARY_MODELLING_TABLE_V2.json").read_text())
        require_checks({"table_passed": manifest["status"] == "frozen_combined_boundary_modelling_table_v2"}, "combined modelling table")
        manifest["cache"] = cached
        return self._record("combined_modelling_table", manifest)

    def boundary_aligned_retfound_v1(self) -> dict:
        """Restore or GPU-extract B0-B4-aligned RETFound point representations."""
        boundary_root = self.root / "outputs/combined_boundary_pipeline_v2"
        embedding_root = self.root / "outputs/boundary_aligned_retfound_embeddings_v1"
        fingerprint = self._fingerprint(
            boundary_root / "visit1/features/boundary_along_scan_samples_v1.csv",
            boundary_root / "visit3/features/boundary_along_scan_samples_v1.csv",
            self.implementation / "calculate_boundary_aligned_retfound_embeddings_v1.py",
            extra="checkpoint=e9ff7864;crop=B0-B4+50um;horizontal=300um;five_by_three",
        )
        mode = self.config.stage_modes["retfound_embeddings"]
        cached = self._require_cache().restore("boundary_aligned_retfound_embeddings_v1", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is None and mode == "reuse":
            raise FileNotFoundError("No compatible boundary-aligned RETFound embedding cache exists")
        if cached is None:
            private_oct = self._restore_private_oct_transport()
            token = os.environ.get("HF_TOKEN", "")
            if not token:
                try:
                    from google.colab import userdata
                    token = userdata.get("HF_TOKEN")
                except Exception:
                    token = ""
            command = [
                sys.executable, str(self.implementation / "calculate_boundary_aligned_retfound_embeddings_v1.py"),
                "--feature-root", str(boundary_root / "visit1/features"),
                "--feature-root", str(boundary_root / "visit3/features"),
                "--private-oct-root", str(private_oct),
                "--models-vit", str(self.implementation / "official_retfound/models_vit.py"),
                "--checkpoint-dir", str(self.config.drive_root / "05_public_model_cache/RETFound"),
                "--output-root", str(embedding_root),
                "--staging-root", str(self.config.drive_root / "03_generated_stage_cache/retfound_case_staging_v1"),
                "--hf-token", token, "--batch-size", "8",
            ]
            subprocess.run(command, check=True)
            cached = self._require_cache().save("boundary_aligned_retfound_embeddings_v1", fingerprint, embedding_root)
        manifest = json.loads((embedding_root / "BOUNDARY_ALIGNED_RETFOUND_EMBEDDINGS_V1.json").read_text())
        require_checks({"embedding_passed": manifest["status"] == "passed_boundary_aligned_retfound_embeddings_v1"}, "boundary-aligned RETFound")
        manifest["cache"] = cached
        return self._record("retfound_embeddings", manifest)

    def combined_model_ladder_v2(self) -> dict:
        table_root = self.root / "outputs/combined_boundary_modelling_table_v2"
        embedding_root = self.root / "outputs/boundary_aligned_retfound_embeddings_v1"
        model_root = self.output_root / "combined_model_ladder_v2"
        fingerprint = self._fingerprint(
            table_root / "combined_boundary_modelling_table_v2.csv",
            embedding_root / "boundary_aligned_point_rows_v1.csv",
            embedding_root / "boundary_aligned_point_representation_3072d_v1.npy",
            self.implementation / "run_combined_model_ladder_v2.py",
            extra=f"seed={self.config.random_seed};nine_models;grouped_nested_validation",
        )
        mode = self.config.stage_modes["model_ladder"]
        cached = self._require_cache().restore("combined_model_ladder_v2", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is None and mode == "reuse":
            raise FileNotFoundError("No compatible combined model cache exists")
        if cached is None:
            subprocess.run([
                sys.executable, str(self.implementation / "run_combined_model_ladder_v2.py"),
                "--table", str(table_root / "combined_boundary_modelling_table_v2.csv"),
                "--embedding-rows", str(embedding_root / "boundary_aligned_point_rows_v1.csv"),
                "--representations", str(embedding_root / "boundary_aligned_point_representation_3072d_v1.npy"),
                "--output-root", str(model_root), "--seed", str(self.config.random_seed),
            ], check=True)
            cached = self._require_cache().save("combined_model_ladder_v2", fingerprint, model_root)
        manifest = json.loads((model_root / "COMBINED_MODEL_RUN_V2.json").read_text())
        require_checks({"nine_model_ladder_passed": manifest["status"] == "passed_combined_nine_model_oof_v2"}, "combined model ladder")
        manifest["cache"] = cached
        return self._record("model_ladder", manifest)

    def combined_evaluation_v2(self) -> dict:
        model_root = self.output_root / "combined_model_ladder_v2"
        evaluation_root = self.output_root / "combined_evaluation_v2"
        script = self.implementation / "evaluate_combined_models_v2.py"
        fingerprint = self._fingerprint(
            model_root / "oof_predictions_v2.csv",
            script,
            extra="combined_evaluation_v2;error_and_agreement;participant_bootstrap",
        )
        mode = self.config.stage_modes["evaluation"]
        cached = self._require_cache().restore("combined_evaluation_v2", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is None and mode == "reuse":
            raise FileNotFoundError("No compatible combined evaluation cache exists")
        if cached is None:
            subprocess.run([
                sys.executable, str(script),
                "--predictions", str(model_root / "oof_predictions_v2.csv"),
                "--output-root", str(evaluation_root),
            ], check=True)
            cached = self._require_cache().save("combined_evaluation_v2", fingerprint, evaluation_root)
        manifest = json.loads((evaluation_root / "COMBINED_EVALUATION_V2.json").read_text())
        require_checks({"evaluation_passed": manifest["status"] == "passed_combined_error_agreement_evaluation_v2"}, "combined evaluation")
        manifest["cache"] = cached
        return self._record("evaluation", manifest)

    def combined_reports_v2(self) -> dict:
        model_root = self.output_root / "combined_model_ladder_v2"
        evaluation_root = self.output_root / "combined_evaluation_v2"
        artifact_root = self.output_root / "thesis_artifacts_v2"
        tables = artifact_root / "tables"; figures = artifact_root / "figures"
        fingerprint = self._fingerprint(
            model_root / "oof_predictions_v2.csv",
            evaluation_root / "all_point_metrics_v2.csv",
            Path(__file__).resolve().parent / "analysis.py",
            Path(__file__).resolve().parent / "reporting.py",
            extra="combined_reports_v2;tables_figures_sensitivity",
        )
        mode = self.config.stage_modes["tables_figures"]
        cached = self._require_cache().restore("combined_reports_v2", fingerprint) if mode in {"auto", "reuse"} else None
        if cached is not None:
            result = {"status": "passed", "cache": cached, "artifact_root": str(artifact_root)}
            return self._record("tables_figures", result)
        if mode == "reuse":
            raise FileNotFoundError("No compatible combined tables/figures cache exists")
        tables.mkdir(parents=True, exist_ok=True)
        predictions = pd.read_csv(model_root / "oof_predictions_v2.csv")
        metrics = pd.read_csv(evaluation_root / "all_point_metrics_v2.csv")
        shutil.copy2(evaluation_root / "all_point_metrics_v2.csv", tables)
        shutil.copy2(evaluation_root / "visit_and_floor_stratum_metrics_v2.csv", tables)
        shutil.copy2(evaluation_root / "model_ranking_by_mae_v2.csv", tables)
        analysis = generate_error_analyses(predictions, artifact_root / "error_sensitivity_analyses")
        figure_outputs = generate_figures(predictions, metrics, figures) if self.config.generate_figures else {}
        cached = self._require_cache().save("combined_reports_v2", fingerprint, artifact_root)
        result = {"status": "passed", "cache": cached, "tables": 3, "figures": len(figure_outputs), "analysis": analysis, "artifact_root": str(artifact_root)}
        return self._record("tables_figures", result)

    def finalise(self) -> dict:
        manifest_path = self.output_root / "RUN_MANIFEST.json"
        # Keep the legacy status string in the manifest for compatibility with
        # older readers, while the current status makes the primary/secondary
        # hierarchy explicit.
        self.manifest["legacy_status"] = "passed_complete_combined_visit_boundary_retfound_pipeline_v2"
        self.manifest["status"] = "passed_complete_primary_full_cohort_secondary_boundary_pipeline_v3"
        self.manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n")
        drive_run_root = self.config.drive_root / "04_final_run_outputs" / self.run_id
        drive_run_root.mkdir(parents=True, exist_ok=False)
        archive_path = drive_run_root / "notebook_run_outputs.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.output_root.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=str(path.relative_to(self.output_root)))
        shutil.copy2(manifest_path, drive_run_root / manifest_path.name)
        return {
            "status": self.manifest["status"],
            "drive_archive": str(archive_path),
            "archive_sha256": sha256(archive_path),
        }
