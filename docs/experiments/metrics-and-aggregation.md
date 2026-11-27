---
type: Experimental protocol
title: Metrics and aggregation
description: Calculation of original-scale errors, IQR normalization and repeated-seed summaries.
tags: [metrics, NRMSE, aggregation, uncertainty]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: ../../tester.py, title: Per-fold metric computation}
  - {resource: ../../analysis_utils.py, title: Global metrics, cohort IQR and common-species filter}
  - {resource: ../../benchmark_analysis.py, title: Historical fold-metric aggregation}
  - {resource: ../../notebooks/01_benchmark_analysis.ipynb, title: Baseline comparison notebook}
  - {resource: ../../notebooks/02_ablation_analysis.ipynb, title: Ablation summary and comparison notebook}
  - {resource: "../../Paper/'main.tex", title: Manuscript validation and Results}
---

# Metrics and aggregation

## Current benchmark analysis

Model comparisons are computed from the original-scale long predictions that every model exports (`predictions_long.csv`), not from fold metric files. For each seed, model run and output (trait × bound), RMSE, MAE, median absolute error, Pearson and Spearman correlations are calculated **globally** from the evaluated cells of all five outer folds pooled together. `NRMSE-IQR = RMSE / IQR`, where the IQR (and range) of each output is computed once from all observed values of the analytical cohort (one value per taxon, 189 taxa, after the loader's validity rules). The denominator is therefore identical for every model, seed and species subset; it uses the global observed distribution and is a descriptive scale, not fitted preprocessing. Seed-level metrics are then summarized as mean and SD across seeds; the SD reflects repeated, overlapping splits and training randomness and is not a prediction interval.

Comparisons against baselines are restricted to the cells common to all compared methods. Because tree-dependent baselines cannot predict the seven taxa absent from the reconstructed tree, these comparisons use the 182 tree-represented taxa for every model. The filter is checked against the saved `ftol_mapping.csv`, and identical scored cells are verified across models. All of this is implemented in `analysis_utils.py` and used by `notebooks/01_benchmark_analysis.ipynb`; see [analysis conventions](analysis-conventions.md).

Per-fold GNN and baseline metric functions now use float64 inputs and scaled RMSE/MAE, and reject nonfinite evaluated inputs or error results before saving metrics. This prevents float32 squaring from turning finite errors into infinite RMSE; see [prediction safeguards](../methodology/prediction-safeguards.md).

## Historical aggregation

For one trait, bound, model and fold, RMSE was calculated on observed evaluated cells in the original scale. `benchmark_analysis.py` derives the IQR from all observed workbook values of that trait and bound, not from each outer-training fold, then computes `NRMSE-IQR = RMSE / IQR`. A zero IQR produces a missing normalized metric. This path is still used by `ablation_extractor.py` to build `results_ablation.parquet` for the GNN ablation notebook.

Across folds, the notebook obtains a per-seed trait/bound RMSE using the square root of the observation-weighted mean of squared fold RMSEs (equal to the pooled RMSE); it then divides by the fixed trait IQR. It averages those 12 output metrics for each seed and summarizes the seed-level means. Therefore the manuscript's **4.511** is the median across seeds of per-seed **mean** NRMSE-IQR across output slices, not the median across the 12 entries in Table 1. See the [audit](../findings/consistency-audit.md).

## Inference

The ablation notebook contains paired seed-level contrasts, bootstrap and sign-flip calculations and per-slice Wilcoxon tests with a Benjamini–Hochberg adjustment. Multiple seeds reuse species and overlapping partitions. Their spread is not a prediction interval, and seed-level tests must not be interpreted as independent species-level replicates. The manuscript's precise significance claims should be reconciled with the selected test and its assumptions. Paired model-versus-baseline differences with intervals are not yet implemented.

Reproduction paths: original-scale long predictions → `analysis_utils.py` → `notebooks/01_benchmark_analysis.ipynb` (baseline comparison); fold metrics → `benchmark_analysis.py`/`ablation_extractor.py` → `benchmark_results/results_ablation.parquet` → `notebooks/02_ablation_analysis.ipynb` (ablation). Absolute errors (RMSE, MAE, median absolute error in original units), IQR denominators and sample sizes are reported with NRMSE. Altitude minimum has an IQR of 10 m with 109 of 185 observed values equal to zero, so modest absolute errors produce large NRMSE-IQR values; spore size contains one extreme taxon (`Pilularia_novae-hollandiae`, maxima about 168,000 against an IQR of 840).
