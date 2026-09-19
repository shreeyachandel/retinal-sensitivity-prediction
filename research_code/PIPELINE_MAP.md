# Research pipeline map

The submitted notebook was self-contained: it embedded each Python module in a
`%%writefile` cell before executing the numbered stages. For public review,
those modules are presented as ordinary source files so they are easier to
navigate, search and diff.

| Notebook stage | Public source |
|---|---|
| Runtime setup and orchestration | [`config.py`](src/thesis_pipeline/config.py), [`runtime.py`](src/thesis_pipeline/runtime.py), [`stages.py`](src/thesis_pipeline/stages.py) |
| Registration | [`registration_landmarks.py`](src/thesis_pipeline/implementation/registration_landmarks.py), [`registration_simpleitk.py`](src/thesis_pipeline/implementation/registration_simpleitk.py), [`registration_automatic.py`](src/thesis_pipeline/implementation/registration_automatic.py), [`registration_automatic_batch.py`](src/thesis_pipeline/implementation/registration_automatic_batch.py) |
| Point alignment and cohort freezing | [`post_registration_point_alignment_cohort.py`](src/thesis_pipeline/implementation/post_registration_point_alignment_cohort.py), [`freeze_point_alignment_cohort.py`](src/thesis_pipeline/implementation/freeze_point_alignment_cohort.py) and validation modules |
| Participant-grouped folds | [`create_participant_grouped_folds.py`](src/thesis_pipeline/implementation/create_participant_grouped_folds.py), [`validate_participant_grouped_folds.py`](src/thesis_pipeline/implementation/validate_participant_grouped_folds.py) |
| Footprint-aware OCT sampling | [`build_point_footprint_sampling_v2.py`](src/thesis_pipeline/implementation/build_point_footprint_sampling_v2.py), [`pool_point_footprint_features_v2.py`](src/thesis_pipeline/implementation/pool_point_footprint_features_v2.py) |
| Primary RETFound representation | [`prepare_retfound_preprocessing_pilot_v1.py`](src/thesis_pipeline/implementation/prepare_retfound_preprocessing_pilot_v1.py), [`calculate_full_cohort_retfound_embeddings_v1.py`](src/thesis_pipeline/implementation/calculate_full_cohort_retfound_embeddings_v1.py) |
| Primary modelling and evaluation | [`run_primary_full_cohort_models_v1.py`](src/thesis_pipeline/implementation/run_primary_full_cohort_models_v1.py), [`evaluate_primary_full_cohort_v1.py`](src/thesis_pipeline/implementation/evaluate_primary_full_cohort_v1.py) |
| Boundary-informed secondary dataset | Heidelberg audit, mapping, footprint and combined-table modules under [`implementation/`](src/thesis_pipeline/implementation/) |
| Secondary modelling and evaluation | [`run_combined_model_ladder_v2.py`](src/thesis_pipeline/implementation/run_combined_model_ladder_v2.py), [`evaluate_combined_models_v2.py`](src/thesis_pipeline/implementation/evaluate_combined_models_v2.py) |
| Reporting | [`analysis.py`](src/thesis_pipeline/analysis.py), [`reporting.py`](src/thesis_pipeline/reporting.py) |

The original notebook's execution cells require governed stage bundles and are
not presented as a misleading one-click public workflow. The repository-level
synthetic notebook is the supported executable route.
