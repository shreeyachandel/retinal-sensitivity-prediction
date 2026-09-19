# Research summary

## Question

Can local structure in optical coherence tomography (OCT) help predict
pointwise retinal sensitivity measured by MAIA microperimetry in Usher
syndrome?

## Research workflow

1. Match OCT and microperimetry examinations at eye-visit level.
2. Register the two imaging coordinate systems and retain human quality-control
   decisions.
3. Map each functional test point to the three nearest OCT B-scans.
4. Extract local image or boundary-derived representations around the mapped
   retinal location.
5. Keep every eye, visit and point from one participant in the same validation
   fold.
6. Evaluate out-of-fold error, agreement and calibration, with uncertainty
   estimated at participant level.

## Submitted-thesis results

The primary cohort contained 22 participants, 78 accepted eye-visits and 2,886
pointwise observations. A ridge model combining retinal location with frozen
RETFound image features achieved the lowest primary MAE: 5.485 dB (95% CI
4.871–6.177), compared with 7.650 dB for location alone. Its RMSE was 6.986 dB,
mean prediction-minus-observation bias was −0.060 dB and Bland–Altman limits of
agreement were −13.754 to 13.634 dB.

The improvement supports a structure–function relationship within this cohort,
but the remaining pointwise spread is too broad to treat OCT predictions as a
replacement for microperimetry. Error was also higher at the device floor.

## Repository contents

This repository includes:

- the full public dissertation, with two retinal QC images replaced by synthetic schematics;
- 37 modules from the submitted registration-to-evaluation pipeline;
- safe aggregate thesis figures and result tables;
- a compact executable implementation of participant-grouped validation; and
- a runnable notebook using generated synthetic data.

Clinical data are not included. The demo illustrates the evaluation pattern
with generated data rather than reproducing the reported thesis results.
