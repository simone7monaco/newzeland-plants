---
type: Research methodology
title: Training, masking and prediction
description: Outer and inner splits, target transformations, masked loss and inference visibility.
tags: [training, masking, validation, leakage]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: ../../train.py, title: Model training and validation}
  - {resource: ../../tester.py, title: Full-graph target-masked evaluation}
  - {resource: ../../models.py, title: Masking and per-trait Huber loss}
  - {resource: ../../occlusion.py, title: Paired occlusion groups}
  - {resource: ../../prediction_bounds.py, title: Bounded inverse shared with the ML baselines}
  - {resource: occlusion-protocol.md, title: Fertile/sterile occlusion protocol}
  - {resource: bounds-parameterization.md, title: Ordering of predicted bounds}
  - {resource: "../../Paper/'main.tex", title: Manuscript training objective and validation}
---

# Training, masking and prediction

## Outer and inner data

Current code uses shuffled five-fold species splits. A training-species graph is induced for model fitting; all location nodes remain available. The inner validation mask is selected from the outer-training observations first, and normalization is then fitted on the outer-training traits with the inner-validation cells treated as missing. Neither inner-validation nor outer-test trait values contribute to the fitted transforms; training stops with an error if any trait column keeps fewer than 10 observed cells for fitting (the minimum observed in practice is about 116). The environmental normalizer sees the shared location nodes; this is not a new-location holdout. A 15% internal mask selects checkpoint targets and is also used for residual calibration.

During training the default selects each species–trait group with probability 15% and optimizes masked reconstruction plus `0.2 ×` visible-entry loss, using per-trait Huber reductions by default. Separate reductions are computed for masked and visible entries before addition; the revised manuscript equation describes this implementation. The exact masked fraction per trait can differ because group decisions are shared and observations may already be missing.

## Which targets are visible

The saved historical results use a leave-one-trait-out setting: hide both bounds of the selected trait in test species, retain other observed traits and run the fitted model on the full graph. Training and test species are connected at inference, so test species can receive training-neighbour information; held-out species features and structure are present. This measures transductive reconstruction, not independent geographical transfer. The legacy setup may reveal duplicated fertile/sterile values; the current manuscript describes paired masking.

Current code defaults to fixed grouped occlusion for fertile/sterile stipes and fronds in training, internal validation and evaluation. When one member is a target, both group members are hidden in all test species; only originally observed targets are scored. Single-trait occlusion remains available with `--occlusion_protocol individual`. This is a new implementation, not evidence that the historical benchmark used it. See [update](../updates/reviewer-response.md). Full protocol: [occlusion protocol](occlusion-protocol.md).

## Transformations

Default min and max features are separately Yeo–Johnson transformed and standardized using outer-training observed cells that are not held out for internal validation; missing values are zero-filled with a visibility mask. At evaluation, the GNN's normalized outputs are mapped back with the bounded inverse shared with the ML baselines (`NormalizeFeatures.inverse(bounded=True)`, `prediction_bounds.bounded_inverse`): each output is clipped to its fitted observed normalized range ± 1 SD, inverted, re-clipped to the observed range itself if the inverse is still non-finite, and set to zero if negative for traits whose fitted values were all nonnegative. Recorded targets are never clipped. Until 2026-11-25 the outputs were instead inverted without bounds and passed through softplus, which shifts small positive values upward (softplus(1) ≈ 1.31); results produced before then keep that postprocessing. Reversed (min, max) pairs are then replaced by their midpoint (see [ordering of predicted bounds](bounds-parameterization.md)). Per-fold `predictions_{min,max}.csv` and `predictions_long.csv` contain these original-scale, projected values, scored against the untransformed recorded targets (the same values the baselines use); the raw network outputs are kept only as `predictions_{min,max}_normalized.csv` and are fold-specific. In historical runs `predictions_{min,max}.csv` were the normalized outputs. The Yeo–Johnson inverse is bounded above when the fitted λ is negative, which on the full cohort applies to altitude minimum (λ ≈ −0.45, limit about 1.9 SD above the mean) and spore minimum and maximum (λ ≈ −0.36 and −0.48, limits about 4.4 and 3.8 SD); without bounds, predictions beyond the limit overflow. The current bounded inverse falls back to the observed range or raises on failure; historical results, produced with the unbounded inverse, still require a transform audit.

Both calibration and test inversions now write per-output correction counts to `prediction_safeguards.csv`; raw infinities are rejected before clipping and NaNs on evaluated cells are errors. See [prediction safeguards](prediction-safeguards.md) for the count definitions and float64 error calculations.
