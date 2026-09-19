# Public-release copy extracted from the submitted MSc notebook.
# Study-specific pseudonyms and governed data are intentionally absent.

"""Readable orchestration layer for the thesis implementation."""

from .config import PipelineConfig
from .stages import ThesisPipeline

__all__ = ["PipelineConfig", "ThesisPipeline"]
