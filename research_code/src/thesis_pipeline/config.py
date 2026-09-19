# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

"""Configuration used by the stage-oriented Colab notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_STAGE_MODES = {
    "registration": "auto",
    "point_alignment": "auto",
    "point_level_dataset": "auto",
    "folds": "auto",
    "point_footprint_100um": "auto",
    "combined_boundary_features": "auto",
    "combined_modelling_table": "auto",
    "retfound_embeddings": "auto",
    "model_ladder": "auto",
    "evaluation": "auto",
    "error_sensitivity_analysis": "auto",
    "tables_figures": "auto",
    # Primary result: all eligible Visit 1 + Visit 3 points, with no boundary
    # availability exclusion.
    "primary_modelling_table": "auto",
    "primary_model_ladder": "auto",
    "primary_evaluation": "auto",
    "primary_reports": "auto",
}


@dataclass
class PipelineConfig:
    """All paths and stage switches are declared in one visible object."""

    drive_root: Path
    workspace_root: Path = Path("/content/thesis_pipeline_workspace_v1")
    random_seed: int = 20260826
    stage_modes: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_STAGE_MODES)
    )
    generate_tables: bool = True
    generate_figures: bool = True

    def validate(self) -> None:
        allowed = {"auto", "reuse", "rebuild"}
        invalid = {
            name: mode for name, mode in self.stage_modes.items() if mode not in allowed
        }
        if invalid:
            raise ValueError(f"Invalid stage modes: {invalid}")

    @property
    def registry_path(self) -> Path:
        organised = (
            self.drive_root
            / "00_notebook_and_registry"
            / "FINAL_COLAB_ASSET_REGISTRY_V1.json"
        )
        # Local verification uses the prepared asset-store directory, where
        # the registry is kept at its root rather than in Drive's notebook
        # subfolder. Colab always uses the organised path above.
        fallback = self.drive_root / "FINAL_COLAB_ASSET_REGISTRY_V1.json"
        return organised if organised.is_file() else fallback

    @property
    def run_output_root(self) -> Path:
        return self.workspace_root / "outputs" / "final_notebook_run_v1"

    @property
    def generated_cache_root(self) -> Path:
        return self.drive_root / "03_generated_stage_cache"

    @property
    def manual_qc_root(self) -> Path:
        """Fixed human decisions and raw registration/alignment inputs."""
        return self.drive_root / "01_manual_qc_inputs"
