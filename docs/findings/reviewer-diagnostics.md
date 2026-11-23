---
type: Research finding
title: Reviewer-response diagnostics
description: Saved-prediction bound audit and descriptive spatial-clustering diagnostics.
tags: [results, spatial-clustering, bounds, diagnostics]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: ../../reviewer2_bounds_audit.csv, title: Original-scale bound audit across configurations}
  - {resource: ../../spatial_knee_analysis/cluster_summary.csv, title: Spectral clustering diagnostic summary}
  - {resource: ../../spatial_cluster_knee.py, title: Reproducible spatial clustering diagnostic}
  - {resource: ../../notebooks/02_ablation_analysis.ipynb, title: Bound checks by species and seed}
---

# Reviewer-response diagnostics

## Ordered bounds in the historical full-model predictions

The original-scale GNN CSVs were compared within the same seed, fold, species and trait. Among 60 repeated seed runs, the minimum exceeded the maximum in 307/9,900 fertile-stipe, 320/9,900 sterile-stipe, 14/10,200 fertile-frond and 14/10,200 sterile-frond comparisons; no inversions were observed for spore and altitude. Twenty species were affected by at least one inversion. These counts are repeated species–seed predictions, not 60 independent samples. The notebook's final cell produces a per-species, per-trait count and coverage table. Values were obtained after inverse transformation and the existing softplus clipping, not from normalized network outputs.

## Spatial cluster diagnostic

A descriptive rerun of the loader's spectral clustering used 2,607 occupied cells and 5, 10, 20, 30, 40, 50, 60, 80 and 100 clusters. For 50 clusters, the median cluster contained 48 occupied cells, with a minimum of 2 and maximum of 125. The method emitted connectivity warnings; at some cluster counts partitions were severely unbalanced. The resulting [plot](../../spatial_knee_analysis/spatial_knee.pdf) and summary do not establish an optimal count or assess imputation performance. Coordinate distances are Euclidean in degrees, not geodesic. Recreate covariates and occurrence edges for each partition before a predictive sensitivity study.

## Current implementation

New grouped masking and ordered output heads are documented in [updates](../updates/reviewer-response.md). Short smoke tests of the two ordered heads found no reversed bounds among the 195 observed target pairs in one fold; this is a functional check, not a benchmark or significance test.
