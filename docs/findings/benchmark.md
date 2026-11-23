---
type: Research finding
title: Reported benchmark findings
description: Manuscript results and how the ablation summaries are calculated.
tags: [results, benchmark, ablation]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: "../../Paper/'main.tex", title: Manuscript Results and Table 1}
  - {resource: ../../notebooks/02_ablation_analysis.ipynb, title: Saved-ablation analysis notebook}
  - {resource: ../../benchmark_results/results_ablation.parquet, title: Saved per-fold GNN metric artifact}
---

# Reported benchmark findings

## Reconstruction

Table 1 reports PhyEnv-NN first or second on 11 of 12 outputs, with the best point estimate on five. The paper labels NRMSE-IQR 10.262 as maximum altitude and 18.534 as maximum spore size, but the saved metrics identify them as **minimum** altitude and **minimum** spore size, respectively. The full-model min/max labels are reversed for all six trait pairs; see the [audit](consistency-audit.md). These are reported point estimates, not evidence of statistically significant superiority. Their reliability, potential duplicate-trait shortcut and original-scale data require further checks.

## Ablation

The manuscript reports 4.591 (Base), 4.595 (environment), 4.512 (phylogeny) and 4.511 (full). In the notebook these are medians across 60 seed-level **means across trait/bound slices**, after pooling fold RMSE by observed counts. They are not the medians of the 12 Table 1 entries. The paper reports that adding phylogeny provides the largest aggregate gain and that environment adds a smaller gain; these are configuration contrasts and include the limitations described in [graph and model](../methodology/graph-and-model.md). A 0.002 median difference between full and phylogeny-only should not be interpreted as evidence of equivalence without a suitable design and interval.

## Other findings

The paper presents target-specific feature attributions, local phylogenetic clusters in prediction changes, and poorer normalized performance for small values. Those descriptions rely on IG and proportional errors rather than causal experiments. Reassess small-value and local significance claims under robust error measures and multiple-testing correction; see [analysis protocol](../experiments/explainability-and-phylogeny.md).
