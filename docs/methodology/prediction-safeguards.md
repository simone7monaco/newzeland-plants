---
type: Research methodology
title: Prediction safeguards and correction logs
description: Shared inverse-transform safeguards, rejection of invalid model outputs, per-output correction counts and stable error calculations.
tags: [prediction, numerical-stability, baselines, calibration, reproducibility]
status: implemented
generated: {by: codex/gpt-6}
sources:
  - {resource: ../../prediction_bounds.py, title: Shared bounded inverse and correction log}
  - {resource: ../../loader.py, title: Fold-fitted ranges and GNN inversion}
  - {resource: ../../ml_baselines.py, title: ML prediction inversion}
  - {resource: ../../train.py, title: Calibration inversion}
  - {resource: ../../tester.py, title: GNN logs and stable error calculations}
  - {resource: ../../r_baselines_fit.py, title: ML fold log exports}
  - {resource: ../../tests/test_prediction_bounds.py, title: Numerical and logging regression tests}
---

# Prediction safeguards and correction logs

GNN and RF/XGBoost/MLP predictions use `prediction_bounds.bounded_inverse`. Each output is clipped to the transformed training-observed minimum/maximum extended by one standardized unit, inverse-transformed in float64, and re-inverted at the training range itself if the first inverse is nonfinite. Negative results are clipped to zero only for output columns whose fitted observations were all nonnegative. Limits use the observations used to fit the transform; GNN internal-validation targets and all outer-test targets are excluded. Recorded targets are untouched. The existing midpoint projection then orders minimum/maximum pairs.

Raw positive or negative infinity raises `FloatingPointError` **before clipping**. NaNs may represent cells the GNN does not predict, but a NaN on a supplied evaluation/calibration mask raises. A nonfinite inverse after the training-range fallback also raises. Per-fold export checks reject nonfinite evaluated predictions, including any introduced when converting back to a tensor dtype.

## Correction log

Every successful GNN or ML fold writes `prediction_safeguards.csv` beside its predictions. Rows identify `fold`, `stage`, `trait` and `variable`. GNN rows distinguish `calibration` and `test`; ML rows have `test`. The file includes rows with zero corrections. R/statistical baselines do not use this transformed-output policy and do not write this log; their ordering corrections remain in `bounds_order.csv`.

| Column | Meaning |
|---|---|
| `n_predictions` | Finite raw predictions processed |
| `n_missing` | Missing raw predictions preserved as NaN |
| `n_clipped_low`, `n_clipped_high` | Predictions outside the extended transformed range |
| `n_inverse_fallback` | Inverses recomputed at the unextended training range |
| `n_nonnegative_clipped` | Negative inverse results replaced by zero |
| `n_corrected` | Distinct predictions receiving any of these corrections |

Every count also has an `n_eval_*` counterpart, restricted to the supplied test-scoring or calibration mask. Compare these restricted counts across methods: ML models also predict originally missing targets, whereas GNN leave-one-group-out exports retain NaN for unobserved cells. Individual correction counts overlap; `n_corrected` counts their union. Counts precede min/max midpoint projection, which has its own log. Correction counts document numerical interventions, not predictive reliability.

## Error calculations and provenance

Both GNN and baseline per-output metric functions cast predictions and targets to float64 **before subtraction**. RMSE and MAE scale absolute errors by their maximum before squaring or averaging, avoiding intermediate overflow even for large finite errors. Nonfinite evaluated inputs, unrepresentable differences and nonfinite error metrics raise `AnalysisConsistencyError`; outputs with fewer than two evaluated cells retain the existing missing-metric convention. Midpoints use `min / 2 + max / 2` to avoid overflow from adding large finite bounds.

The benchmark hashes `prediction_bounds.py` alongside its other runtime source files. Use a fresh output directory when code or protocol changes. Earlier artifacts retain their original postprocessing and do not acquire these logs retrospectively. Functional validation is recorded in the [log](../log.md); no full benchmark performance claim follows from a unit test or short smoke run.
