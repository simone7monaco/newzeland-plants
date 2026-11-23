---
type: Experimental protocol
title: Benchmark and ablation protocol
description: Species folds, model configurations, baselines, commands and artifact provenance.
tags: [benchmark, baselines, ablation, reproducibility]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: ../../run_benchmark.py, title: Benchmark orchestration}
  - {resource: ../../r_baselines_fit.py, title: Baseline fitting and evaluation}
  - {resource: ../../r_models.py, title: R MICE and Rphylopars wrappers}
  - {resource: "../../Paper/'main.tex", title: Manuscript validation and Table 1}
  - {resource: ../../benchmark_results/results_ablation.parquet, title: Saved per-fold GNN metrics}
---

# Benchmark and ablation protocol

## Design

The saved ablation artifact contains four configurations, 60 distinct seeds and five outer species folds, with six traits × two bounds per configuration and seed. It corresponds to the historical 170-species workbook; new runs default to 206 taxa with observed traits from the full workbook and cannot be combined with those results. The configurations are `no_env_no_phylo` (Base), `env_only`, `phylo_only` and `env_phylo` (full). Paired comparisons should match seeds, folds, evaluation cells, preprocessing, masks and outputs. These repeated splits overlap and are not independent samples of species.

The paper compares training mean, training median, phylogenetic nearest neighbour and five-neighbour mean, R MICE with CART and Rphylopars with Brownian motion and diagonal phylogenetic covariance. Tree-dependent baselines cannot predict taxa missing from tree tips: current exports retain these rows with missing predictions, and metrics exclude them. The Rphylopars wrapper normalizes quoted Newick tip labels to match the species names consistently with the Python tree reader. Current baseline protocol 1 fits using outer-training species, then masks a target group for all test species and transforms the test data; protocol 2 fits on the training plus masked test table and is transductive. MICE's wrapper calls `mice.mids(newdata=...)`; whether this re-estimates conditional models with test rows needs independent verification before treating it as an immutable trained CART predictor. The paper's baseline comparison does not include an equal-input conventional ML predictor.

## Run current code

Install dependencies as in [README](../../README) and use the R environment for baselines. Example for a fresh exploratory benchmark:

```sh
.venv/bin/python run_benchmark.py 1 --start-seed 42 --occlusion-protocol paired --output-dir benchmark_paired_new
```

This is one seed with five folds, four GNN configurations, selected baselines and XAI; it is not a reproduction of the historical 60-seed result. Pass `--gnn-args` and `--baseline-args` as quoted strings for model-specific settings. The orchestrator enforces a common occlusion protocol and records workbook and tree fingerprints before resuming jobs. Use a **different empty output directory** for each protocol, output head and data version: the markers do not identify every setting. Benchmark defaults turn on XAI and a complete run can be expensive.

## Artifacts

Historical GNN per-fold `predictions_min.csv` and `predictions_max.csv` are normalized outputs; `predictions_min_original.csv`, `predictions_max_original.csv` and `predictions_original_all_folds.csv` contain inverse-transformed, soft-clipped predictions. GNN `per_trait_metrics_original.csv` contains original-scale errors. Baselines write per-fold `predictions_min.csv`, `predictions_max.csv`, and `per_trait_metrics.csv`. The consolidated `results_ablation.parquet` is built by `ablation_extractor.py` using `dashboard.py`. Confirm data, exact code version and parameters for any historical run; folder names alone do not establish provenance.
