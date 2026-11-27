---
type: Research methodology
title: Ordering of predicted minimum and maximum bounds
description: How min <= max is enforced on predicted trait bounds, the alternatives that were tested, and their matched comparison.
tags: [bounds, output-head, post-processing, reviewer-response]
status: draft
generated: {by: claude/opus-5.5}
sources:
  - {resource: ../../tester.py, title: project_ordered_bounds and original-scale evaluation}
  - {resource: ../../train.py, title: Bounds flags, order penalty and calibration}
  - {resource: ../../models.py, title: Independent and min-plus-range output heads}
  - {resource: ../../loader.py, title: Fitted transforms and differentiable inverse}
  - {resource: ../../tests/test_bounds_projection.py, title: Projection tests}
  - {resource: ../../tests/test_order_penalty.py, title: Differentiable inverse tests}
---

# Ordering of predicted minimum and maximum bounds

## Adopted method

The default model keeps two independent output heads, `--bounds_parameterization independent`, with min and max each transformed by its own fold-fitted Yeo–Johnson + z transform. Ordering is enforced after prediction (`--enforce_bounds_order true`, the default). After the inverse transform to the original scale, every reversed pair (min̂ > max̂) is replaced by its midpoint (`tester.project_ordered_bounds`). All other pairs are left unchanged.

This is the Euclidean projection onto the convex set {min ≤ max}. Every observed pair lies in that set, so the projection cannot increase the joint squared error (min, max) of any pair. It can shift the error slightly between the two components. The projection is applied:

- to the test predictions, before metrics, saved CSVs and conformal bounds;
- to the internal-validation predictions used to fit the conformal residual bounds.

The checkpoint-selection loss is computed on normalized outputs and is not affected.

## Evidence

The runs used seeds 42–46, five random species folds, paired occlusion and 1,000 epochs. Training followed the corrected order described in [training and masking](training-and-masking.md), where normalization is fitted after the inner-validation cells are selected and excludes them. Metrics are on the original scale, weighted by n across folds and averaged over six traits × min/max. In the first table, nRMSE is RMSE divided per output by the mean of the independent head.

| Configuration | Spearman | Pearson | nRMSE | min > max |
|---|---|---|---|---|
| Independent heads | 0.580 ± 0.033 | 0.494 ± 0.023 | 1.000 | 0.86% |
| Independent + soft order penalty (λ = 1) | 0.576 ± 0.036 | 0.494 ± 0.023 | 1.001 | 0.78% |
| Min + ReLU range | 0.534 ± 0.043 | 0.480 ± 0.035 | not finite | 0% |
| Min + Softplus range | 0.490 ± 0.023 | 0.406 ± 0.038 | not finite | 0% |

Mean ± sd across seeds. Excluding `Alt max`, the ordered heads have nRMSE 1.046 ± 0.026 (ReLU) and 1.111 ± 0.037 (Softplus).

Projecting the independent predictions after inversion:

- Before projection, 49 of 5,670 test pairs were reversed (0.73% for seeds 42–44, 1.06% for seeds 45–46). All were in `Ferstipe` or `SterileStipe`, about one pair per fold, with min̂ and max̂ nearly equal.
- Pearson and Spearman are unchanged to four decimals. RMSE and MAE change by at most about 0.01% per output.
- The joint squared error was lower or equal in 150 of 150 fold-trait cells.

## Alternatives tested and not adopted

- **Min plus range (`min_plus_range_relu`, `min_plus_range_softplus`).** These heads predict the normalized minimum `a` and a width `b`, and return `a` and `a + relu(b)` or `a + softplus(b)`. Both bounds share the transform fitted on training minima. Order is exact, but accuracy dropped:
  - Softplus was worse than the independent heads in 5 of 5 seeds (Spearman −0.087 ± 0.039).
  - ReLU was worse on average, within seed-to-seed variability.
  - The losses were concentrated in the strongly predicted stipe and frond traits.
  - The min-fitted transform does not fit altitude maxima. The `Alt max` inverse overflowed to infinite RMSE in 12 of 25 (ReLU) and 10 of 25 (Softplus) fold-seeds. Its median RMSE was about 17,000–21,000 m, against 445 m with independent heads. These runs used the unbounded inverse; the current bounded inverse ([training and masking](training-and-masking.md#transformations)) would keep these outputs finite but does not fix the misfit of the shared transform.

  The heads remain available for comparison. `--shared_bounds_normalization true` gives the independent head the same shared transform, to isolate the effect of the constraint.
- **Soft order penalty (`--order_penalty_weight`, default 0).** This adds `λ · mean(relu(min̂ − max̂) / IQR_j)` to the loss. The term is computed on the original scale through the differentiable inverse `NormalizeFeatures.inverse_tensor`. Normalized inputs are clamped to the fitted observed range ± 1 SD, the same window as the evaluation inverse (±6 SD before 2026-11-25 and when no range is fitted), and the power bases are kept inside the Yeo–Johnson domain. The IQR is that of the training maxima, with inner-validation cells excluded. With λ = 1 the change in performance was within seed noise, and reversals fell only from 0.86% to 0.78%. The reversals are rare and tiny, so the penalty gradient is negligible, and the penalty offers no guarantee.

## Usage notes

- The tester counts reversed pairs on the raw predictions, before any projection, whether or not projection is enabled. It writes them to `fold_*/bounds_order.csv` and, summed over folds, to `bounds_order_all_folds.csv`. Columns: per trait, all test pairs, reversed pairs, the same counts restricted to evaluated cells, the largest reversal in original units, and whether the pairs were projected. The total is also printed with the sanity checks.
- Use a fresh output directory per head. The benchmark resume marker tracks the occlusion protocol, not the head parameterization.
- Disable the projection only to reproduce historical outputs, with `--enforce_bounds_order false`.
- Baselines also default to `--enforce_bounds_order true`: reversed finite (min, max) pairs are counted before projection, replaced by their midpoint, and then scored and exported. Per-fold and merged reversal counts are saved for both GNN and baselines; fresh benchmark rates remain to be measured.
