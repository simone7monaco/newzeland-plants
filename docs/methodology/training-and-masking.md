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
  - {resource: "../../Paper/'main.tex", title: Manuscript training objective and validation}
---

# Training, masking and prediction

## Outer and inner data

Current code uses shuffled five-fold species splits. A training-species graph is induced for model fitting; all location nodes remain available. Normalization is fitted on the outer-training graph before the inner validation mask is selected. Thus inner-validation trait values contribute to normalization, though outer-test trait values do not. The environmental normalizer sees the shared location nodes; this is not a new-location holdout. A 15% internal mask selects checkpoint targets and is also used for residual calibration.

During training the default selects each species–trait group with probability 15% and optimizes masked reconstruction plus `0.2 ×` visible-entry loss, using per-trait Huber reductions by default. Separate reductions are computed for masked and visible entries before addition; the revised manuscript equation describes this implementation. The exact masked fraction per trait can differ because group decisions are shared and observations may already be missing.

## Which targets are visible

The saved historical results use a leave-one-trait-out setting: hide both bounds of the selected trait in test species, retain other observed traits and run the fitted model on the full graph. Training and test species are connected at inference, so test species can receive training-neighbour information; held-out species features and structure are present. This measures transductive reconstruction, not independent geographical transfer. The legacy setup may reveal duplicated fertile/sterile values; the current manuscript describes paired masking.

Current code defaults to fixed grouped occlusion for fertile/sterile stipes and fronds in training, internal validation and evaluation. When one member is a target, both group members are hidden in all test species; only originally observed targets are scored. Single-trait occlusion remains available with `--occlusion_protocol individual`. This is a new implementation, not evidence that the historical benchmark used it. See [update](../updates/reviewer-response.md).

## Transformations

Default min and max features are separately Yeo–Johnson transformed and standardized using outer-training observed cells; missing values are zero-filled with a visibility mask. At evaluation, the GNN's normalized outputs are inverse-transformed and passed through softplus on the original scale. The per-fold normalized output CSVs and original-scale CSVs therefore are not interchangeable. The inverse can be numerically unstable for extreme outputs; the historical results require a transform audit.
