---
type: Experimental protocol
title: Metrics and aggregation
description: Calculation of original-scale errors, IQR normalization and repeated-seed summaries.
tags: [metrics, NRMSE, aggregation, uncertainty]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: ../../tester.py, title: Per-fold metric computation}
  - {resource: ../../dashboard.py, title: IQR scale and aggregation}
  - {resource: ../../notebooks/02_ablation_analysis.ipynb, title: Ablation summary and comparison notebook}
  - {resource: "../../Paper/'main.tex", title: Manuscript validation and Results}
---

# Metrics and aggregation

For one trait, bound, model and fold, RMSE is calculated on observed evaluated cells in the original scale. The dashboard derives the IQR from all observed workbook values of that trait and bound, not from each outer-training fold, then computes `NRMSE-IQR = RMSE / IQR`. A zero IQR produces a missing normalized metric. This scale is a descriptive evaluation denominator, not the model's fitted preprocessing; it nevertheless uses global observed target distribution and must be stated when discussing held-out data.

Across folds, the notebook obtains a per-seed trait/bound RMSE using the square root of the observation-weighted mean of squared fold RMSEs; it then divides by the fixed trait IQR. It averages those 12 output metrics for each seed and summarizes the seed-level means. Therefore the manuscript's **4.511** is the median across seeds of per-seed **mean** NRMSE-IQR across output slices, not the median across the 12 entries in Table 1. See the [audit](../findings/consistency-audit.md).

The notebook contains paired seed-level contrasts, bootstrap and sign-flip calculations and per-slice Wilcoxon tests with a Benjamini–Hochberg adjustment. Multiple seeds reuse species and overlapping partitions. Their spread is not a prediction interval, and seed-level tests must not be interpreted as independent species-level replicates. The manuscript's precise significance claims should be reconciled with the selected test and its assumptions.

Reproduction path: original fold metrics → `dashboard.py`/`ablation_extractor.py` → `benchmark_results/results_ablation.parquet` → `notebooks/02_ablation_analysis.ipynb`. Absolute errors and observed-versus-predicted plots should accompany NRMSE for high-error outputs.
