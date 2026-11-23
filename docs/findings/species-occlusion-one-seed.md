---
type: Research finding
title: One-seed species coverage and paired occlusion comparison
description: Matched-species results for four GNN conditions and six imputation baselines.
tags: [results, benchmark, occlusion, species-coverage]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: ../../factorial_benchmark_results/seed_42_per_trait.csv, title: Matched-cohort model results}
  - {resource: ../../factorial_benchmark_results/seed_42_summary.csv, title: Matched-cohort model summaries}
  - {resource: ../../factorial_benchmark_results/seed_42_gnn_effects.csv, title: Per-output factorial effects}
  - {resource: ../../factorial_benchmark_results/seed_42_gnn_equal_bounds_subgroups.csv, title: Post-hoc equal-bound subsets}
  - {resource: ../../factorial_benchmark_results/seed_42_complete_population_summary.csv, title: Full-population results}
  - {resource: ../../factorial_benchmark_results/manifest.json, title: Original run settings and fingerprints}
  - {resource: ../../summarize_species_occlusion.py, title: Reproducible summary calculations}
---

# One-seed species coverage and paired occlusion comparison

## Design and limitations

This historical run used a 207-node complete graph, including one taxon without observed traits. The current default excludes that node, so these results are not a rerun of the 206-taxon protocol.

Seed 42, five species folds and 1,000 epochs were used for four conditions: 170 vs 207 available taxa, crossed with individual vs jointly paired fertile/sterile masking. The full graph model had both phylogeny and environment enabled; existing baseline models were also evaluated. All 170 common taxa had the same fold in all conditions, verified against the stable species hash. The GNN comparison below uses the intersection of finite evaluated predictions across all four conditions for each output and normalizes RMSE by the original 170-species workbook's IQR. Thus an output's `n` is held constant across conditions. Lower NRMSE-IQR is better; the aggregate is a simple mean of the twelve output scores, not a pooled RMSE.

The original run was interrupted after 14 of 20 GNN folds. Six folds were rerun individually, starting seed 42 in separate processes, rather than inheriting the advanced random state of the original sequential five-fold process. The fold assignments match, but initialization and random masks are not strictly matched across all conditions. With one seed, these contrasts are descriptive and cannot identify whether a difference is caused by dataset size, the masking intervention or stochastic optimization. A repeated-seed study with consistently seeded folds is needed for reliable inference.

## GNN on the same 170-species cohort

| Dataset | Masking | Mean NRMSE-IQR across 12 outputs | Median across outputs |
|---|---|---:|---:|
| Limited | Individual | 4.622 | 1.574 |
| Limited | Paired | 4.567 | 1.681 |
| Complete | Individual | 4.537 | 1.521 |
| Complete | Paired | 4.676 | 1.694 |

The IQR-normalized spore minima and maxima (roughly 18.5 and 16.3) and altitude minimum (roughly 10.3) dominate the mean. Median and per-output values give complementary summaries. The original full-model seed-42 result is 4.450 on the same workbook/IQR; its random folds and historical implementation differ, so this is descriptive context, not a paired attribution. The original 60-seed artifact's mean of seed-level means is 4.525, and its median is 4.511.

## Factorial contrasts

At limited size, paired vs individual masking changes the mean by **−0.055**; at complete size it changes by **+0.139**. At individual masking, complete vs limited changes the mean by **−0.085**; at paired masking it changes by **+0.109**. These opposing signs depend on trait and mask and are not evidence of statistical significance.

With the complete dataset, paired masking increases NRMSE-IQR from 1.752 to 2.220 for minimum fertile frond and from 1.693 to 2.230 for minimum sterile frond; it increases minimum fertile and sterile stipe scores from 1.509 to 1.663 and from 1.533 to 1.724. In contrast, on the limited dataset the two minimum-frond scores decrease under paired masking, from 2.443 to 1.711 and 1.950 to 1.722. Therefore no uniform paired-masking effect follows from this run. Squared-error changes for the frond minima are strongly influenced by a few taxa, including `Ptisana_salicina` and `Sphaeropteris_medullaris`; this is not a formal influence analysis or proof of mechanism.

As a post-hoc description, the data contain 147 equal-bound stipe pairs and 146 equal-bound frond pairs in the common cohort, with 18 and 24 different-bound pairs respectively. Equal recorded bounds do not by themselves establish monomorphism. See the subgroup CSV for per-condition values and counts.

## Full-population description and baselines

On all evaluable cells in the complete workbook, with that workbook's IQR and the common mask between complete conditions, the GNN mean across 12 outputs is 5.446 for individual masking and 5.584 for paired masking. These are not directly comparable to the 170-species numbers because both the evaluation cohort and IQR denominator differ.

All six baseline models completed. Their results and per-output counts appear in the CSVs, but their aggregate values are **not directly rank-comparable across models** because tree-only baselines have fewer evaluable species. The four complete-dataset Rphylopars taxa with quoted tip-name mismatch were also excluded from that model, and its results must be rerun after fixing the identifier mismatch. Training-mean, median, phylogenetic-neighbour and diagonal Rphylopars scores are identical across masking conditions here; MICE scores differ. See the [audit](consistency-audit.md).

## Reproduction

- [Experiment protocol and recovery provenance](../experiments/species-occlusion-factorial.md)
- Matched-cohort detailed data: `factorial_benchmark_results/seed_42_per_trait.csv`.
- Full population: `factorial_benchmark_results/seed_42_complete_population_per_trait.csv`.
- Contrast and subgroup tables: `seed_42_gnn_effects.csv` and `seed_42_gnn_equal_bounds_subgroups.csv` in the same directory.
- Per-fold predictions and logs reside under `factorial_benchmark_results/seed_42/` and `factorial_benchmark_results/recovery/`. The results directory is ignored by Git and must be preserved externally if needed.
