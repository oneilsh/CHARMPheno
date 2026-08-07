# Vendored test fixture: `toy_bars_3x3`

These files are copied verbatim from the reference implementation
**dtak/prediction-constrained-topic-models** (MIT License, Copyright (c) 2018 dtak),
`datasets/toy_bars_3x3/`, as published alongside:

> Hughes, Weiner, Hope, McCoy, Perlis, Sudderth, Doshi-Velez.
> "Prediction-constrained semi-supervised topic models." AISTATS 2018.

Used here as the correctness **oracle** for our faithful PC reference
(`analysis/pc/slda_reference.py`): a 3x3 "bars" corpus (V=9) with a single binary
label, plus the authors' published known-good parameter dumps
(`good_loss_{x,pc,y,label_rep}_K4_param_dict.dump`) for each training regime.

The upstream MIT license is preserved as `LICENSE-upstream-dtak-MIT`.
The generator source (`src/`) is intentionally not vendored — only the data.
