# Third-party notices

## RETFound

The research used RETFound as a frozen retinal-image feature extractor:

- Repository: <https://github.com/rmaphoh/RETFound>
- Paper: Zhou et al., *A foundation model for generalizable disease detection
  from retinal images*, Nature 622, 156-163 (2023).
- The upstream repository currently identifies its licence as Creative Commons
  Attribution-NonCommercial 4.0 International.

This repository does not redistribute RETFound source code, checkpoints or
weights. References to RETFound APIs in the sanitised research pipeline are
integration points only. Anyone using those components must obtain the
upstream implementation and model access separately and follow its licence and
access conditions.

## Scientific Python libraries

The public demonstration uses NumPy, pandas, scikit-learn and Matplotlib under
their respective open-source licences. The historical research pipeline also
references packages including SimpleITK, Pillow, PyTorch and timm; they are not
vendored here.
