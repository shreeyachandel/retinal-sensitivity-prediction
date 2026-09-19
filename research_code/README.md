# Research pipeline

This directory contains the substantive Python implementation embedded in the
submitted MSc reproducibility notebook. It is published for code review and
methodological transparency.

## Included

- MAIA-to-OCT/SLO registration and registration validation;
- frozen-cohort reconciliation and point-alignment checks;
- three-nearest-B-scan and footprint-aware sampling;
- RETFound preprocessing and representation orchestration;
- participant-grouped fold construction and validation;
- primary and secondary model ladders;
- evaluation, agreement analysis and reporting.

The modules remain deliberately close to the submitted implementation. The
small executable package in `../src/retinal_sensitivity/` is the recommended
entry point for running the public synthetic demonstration.

See the [pipeline map](PIPELINE_MAP.md) for a stage-by-stage index from the
submitted notebook to the public source files.

## Public version

Clinical inputs, participant-level outputs and credentials are not included.
Use the repository-level `retinal-demo` command for an end-to-end runnable
example using generated data.

The public research modules are released under the repository's MIT licence.
External tools, models and checkpoints retain their own licences and access
conditions; see `../THIRD_PARTY_NOTICES.md`.
