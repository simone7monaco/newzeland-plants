---
type: Experimental protocol
title: Analysis conventions and consistency checks
description: Canonical names and orderings for all notebooks, original-scale prediction loading, common-species model comparisons and the checks that stop an inconsistent analysis.
tags: [analysis, notebooks, naming, consistency, baselines, metrics]
status: implemented
generated: {by: claude/opus-5.5}
sources:
  - {resource: ../../analysis_utils.py, title: Shared analysis utilities}
  - {resource: ../../tester.py, title: long_predictions_frame (GNN and baseline export)}
  - {resource: ../../r_baselines_fit.py, title: Baseline exports}
  - {resource: ../../loader.py, title: PlantDataset.tree_coverage}
  - {resource: ../../tests/test_analysis_utils.py, title: Unit and notebook tests}
---

# Analysis conventions and consistency checks

`analysis_utils.py` is the single source of truth for how the analysis notebooks (00–04), `benchmark_analysis.py` and `ablation_extractor.py` name, order and load results. Notebooks import names from it instead of defining their own, and every check raises `AnalysisConsistencyError` rather than warning.

## Canonical names and orderings

| Item | Order | Labels |
|---|---|---|
| Traits | `Alt`, `Ferstipe`, `SterileStipe`, `FertileFrond`, `SterileFrond`, `Spore` | `TRAIT_LABELS` (e.g. "Fertile stipe length") |
| Bounds | `min`, `max` | "Minimum value", "Maximum value" |
| Input configurations | `baseline`, `environment`, `phylogeny`, `full` | "No additional inputs" … "Environment + phylogeny"; short `Base`, `Env`, `Phylo`, `Phylo_Env` for the R summaries |
| Models | training mean, training median, Phylo NN, Phylo KNN, Rphylopars (BM), MICE, MissForest, random forest, XGBoost, MLP, PhyEnv-NN | covariate models and the GNN carry their input configuration in the label |
| Environmental features | location covariates of `PlantDataset` | `ENV_FEATURE_LABELS`, domains Climate/Soil/Elevation/Density/Position |

Baseline directory names written by `r_baselines_fit.model_dir_suffix` are parsed strictly by `parse_baseline_dir`. Historical names without an occlusion suffix (e.g. `MICE_prot1`) are accepted for the fold-metric path. Unknown models, unexpected suffixes, environmental/phylogenetic inputs on trait-only methods, or unknown trait, bound, configuration or feature labels raise.

## Loading and checks

`load_benchmark_predictions(results_dir)` reads `runs/seed_*/{baselines,gnn}/*/fold_*/predictions_long.csv` and checks that:

- all predictions are on the original scale (`scale == "original"`); results without long exports must be rerun;
- each model run has five folds and no duplicated cells, and only one directory maps to a model/configuration per seed;
- for every seed, all model runs use the same species and the same species-to-fold assignment;
- observed targets are identical across models (relative tolerance 1e-6);
- evaluated cells are observed and have finite predictions (`allow_nonfinite=True` only for diagnosis). The same condition is enforced earlier, per fold, when exports are written: `tester.long_predictions_frame` raises `AnalysisConsistencyError` for non-finite evaluated cells; `r_baselines_fit.evaluate_and_save` raises for infinite predictions on observed cells, and for missing ones except in tree-dependent methods; `run_benchmark.check_original_scale` repeats the check on each merged `predictions_long_all.parquet`, including resumed runs;
- only tree-dependent methods leave observed cells unscored.

`load_tree_coverage` reads the `ftol_mapping.csv` saved with each run, requires it to be identical across runs and consistent with the FTOL placements.

## Common-species comparisons

`baseline_comparison_cells(predictions, coverage)` keeps, for every model, only the evaluated cells of species scored by all compared model runs. When a tree-dependent method is in the comparison, the removed species must be exactly the taxa absent from the reconstructed tree (seven of 189: `Doodia_milnei`, `Alsophila_dealbata`, `Hypolepis_lactea`, `Hypolepis_rufobarbata`, `Leptolepia_novae-zelandiae`, `Brownseya_serpentina`, `Pellaea_calidirupia`); without one, nothing may be removed. It also verifies that every model is scored on the same number of cells. This applies to comparisons with baselines only; GNN training and GNN-only analyses keep all taxa.

## Metrics

`01_benchmark_analysis.ipynb` opens with an **Experiment status** cell (read-only, matplotlib): a seed × job grid (baselines per input combination and the four GNN configurations) coloured complete/partial/missing, with completed-seed counts per job. The grid comes from `manifest.json` and `run_benchmark.py` (`BASELINES`, `COVARIATE_BASELINES`, `find_baseline_results`); a job is complete when its merged `result_file` exists.

`global_metrics` pools the evaluated cells of the five folds within each seed; `cohort_scales` provides IQR and range per output from all observed cohort values; `summarize_seeds` reports mean and SD across seeds and raises if the number of scored cells changes across seeds. `comparison_table` and `label_output_columns` produce tables in canonical row and column order. See [metrics and aggregation](metrics-and-aggregation.md).

## Validation

`tests/test_analysis_utils.py` builds synthetic benchmark trees and checks naming, the common-species filter, metrics and each failure mode (normalized predictions, mismatched folds, unexpected exclusions, unknown directories). It also executes the code cells of `01_benchmark_analysis.ipynb` on the synthetic results (including the status cell, with a synthetic `manifest.json`). A real-data seed-42/fold-0 smoke run confirmed identical species, folds, observed values and evaluated cells for the GNN and a baseline, that tree-only exclusions belonged to the seven unresolved taxa, and that RMSE recomputed from the GNN long export equals the tester's original-scale RMSE. No benchmark results have been produced with this pipeline yet.
