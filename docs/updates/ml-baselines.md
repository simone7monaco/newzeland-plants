---
type: Project update
title: Non-graph machine-learning baselines
description: Tuned random forest, XGBoost and MLP regressors with selectable trait, categorical, phylogenetic and environmental inputs, run on the existing baseline folds.
tags: [update, baselines, reviewer-response, R2.2, R4.7]
status: draft
generated: {by: claude/opus-5.5}
sources:
  - {resource: ../../ml_baselines.py, title: ML baseline features and tuned regressors}
  - {resource: ../../prediction_bounds.py, title: Bounded inverse shared with the GNN}
  - {resource: ../../r_baselines_fit.py, title: Baseline fitting and evaluation}
  - {resource: ../../tests/test_ml_baselines.py, title: Unit tests}
  - {resource: reviewer-responses.md, title: Reviewer comments R2.2 and R4.7}
---

# Non-graph machine-learning baselines

## Problem

Reviewers 2 (comment 2) and 4 (comment 7) asked whether the gains of PhyEnv-NN come from graph message passing or from the extra information it receives. The existing baselines (training mean/median, phylogenetic k-NN, MICE, Rphylopars) do not include a conventional ML predictor with matching inputs.

## Implementation

`r_baselines_fit.py` accepts three new `--baseline_models`: `random_forest`, `xgboost` and `mlp`. They use the same outer species folds (`data_split`, `--split_strategy`, `--seed`), the same occlusion groups (`--occlusion_protocol`), the same test masking loop, metrics and output files as the other baselines.

For each target trait a separate multi-output regressor predicts minimum and maximum together. It is trained on outer-training species whose target is observed. The inputs are:

| Block | Flag | Content |
|---|---|---|
| Traits | always | min/max of every trait outside the target's occlusion group; Yeo–Johnson + standardization, mean imputation, missingness indicators |
| Categorical | `--use_categorical_features` (default true) | the dummy-encoded attributes of the GNN species input, including family |
| Phylogeny | `--use_phylo_features` | 16 Node2Vec descriptors (zero for the seven unresolved taxa, as in the GNN) |
| Environment | `--use_env_features` | occurrence-weighted mean and SD over the species' location nodes of centroid coordinates and the climate, soil, population and elevation layers, plus log total occurrence and number of occupied locations |

The whole occlusion group is excluded from the inputs (both fertile/sterile members under the default paired protocol), so training matches test-time masking. Feature preprocessing is inside an sklearn pipeline; a `TransformedTargetRegressor` fits the Yeo–Johnson target transformation and standardization separately within every inner-training fold. Inner-validation targets do not contribute to either transform; the selected model is then refitted on all outer-training species. All fitting uses outer-training species only. When no fixed-parameter YAML is supplied, hyperparameters are tuned by randomized search (`--ml_search_iter`, default 30) with inner species-level K-fold CV (`--ml_inner_folds`, default 5) and MSE on validation targets expressed using that inner-training fold’s fitted target transform. Predictions are mapped back with `prediction_bounds.bounded_inverse`, the same rule the GNN uses since 2026-11-25: clip to one transformed SD beyond the training range, invert, fall back to the training range itself if the inverse is non-finite, and clip at zero each target column (min or max) whose training values were all nonnegative. Minimum ≤ maximum is enforced for both baselines and GNN by the default post-hoc midpoint projection, after counting raw reversals. Selected hyperparameters, sample sizes and tuning time are written to `tuning.csv` per fold.

The ML baselines only support `--training_protocol 1`. The same environmental summaries are also added as extra columns for MICE and MissForest with `--use_env_features`, as the phylogenetic descriptors already were with `--use_phylo_features`; these R imputers also impute the summaries of species without occurrences. Training mean/median, phylogenetic k-NN and Rphylopars use trait values (and the tree) only and cannot use either input: when `--use_env_features` or `--use_phylo_features` is set they are skipped with a warning and produce no output. Output directories carry `_env`/`_phylo` only for baselines that use those inputs, plus `_nocat` for ML baselines without categorical inputs.

Example (the four input configurations correspond to the GNN ablations):

```sh
micromamba run -n r-baselines bash -c 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; exec uv run --frozen python r_baselines_fit.py --baseline_models random_forest,xgboost,mlp --k -1'
micromamba run -n r-baselines bash -c 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; exec uv run --frozen python r_baselines_fit.py --baseline_models random_forest,xgboost,mlp --k -1 --use_phylo_features'
micromamba run -n r-baselines bash -c 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; exec uv run --frozen python r_baselines_fit.py --baseline_models random_forest,xgboost,mlp --k -1 --use_env_features'
micromamba run -n r-baselines bash -c 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; exec uv run --frozen python r_baselines_fit.py --baseline_models random_forest,xgboost,mlp --k -1 --use_env_features --use_phylo_features'
```

## Limits

Species-level environmental summaries approximate the location graph and do not reproduce message passing. The models do not use neighbouring species' traits, unlike phylogenetic message passing; the Node2Vec descriptors only encode tree position. Node2Vec and the environmental covariates are computed over all taxa and locations, as for the GNN, but contain no trait values.

## Validation

Unit tests cover the weighted environmental summaries (including missing location values and species without occurrences), the input switches, that the occluded group is never read, and that predictions are finite and nonnegative. A fold-0 smoke run (seed 42, paired occlusion, env + phylo, reduced search of 4 candidates and 3 inner folds) completed for all three models against the cached graph. This was a smoke test only: **no benchmark results yet**. Run all seeds and input configurations, compare with the GNN using paired per-output differences on the same folds, and report computational cost before finalizing reviewer responses R2.2 and R4.7. The manuscript Methods, Table 1 rows (values pending) and an appendix table of search spaces now describe these baselines, and the R2.1, R2.2, R2.3, R4.2 and R4.7 drafts name them.

The standard benchmark now runs MICE, random forest, XGBoost and MLP under all four environmental/phylogenetic input combinations. Trait/tree-only methods run once per seed. The standalone runner accepts `--all_feature_combinations` to enumerate supported combinations (including MissForest when explicitly selected with protocol 2). Covariate results are matched by their actual input suffixes during resume, aggregation and metric loading.

On 2026-11-25, all 30 tests passed in the README R/Python environment, including regression checks that target transforms fit only inner-training rows, input-specific scheduling/resume, species identifiers and original-unit merged exports. Reduced seed-42/fold-0 smoke runs exercised all 21 standard baseline configurations (ML: one search candidate, two inner folds; environmental MICE checks included `m=1, maxit=1`) and produced 12 finite RMSE/MAE rows per configuration with ordered finite bounds. A two-epoch GNN smoke run also completed. These checks establish functionality only; full benchmark results are still pending.

Each ML fold writes `prediction_safeguards.csv`, with per-trait/bound clipping, inverse-fallback and nonnegativity counts, both overall and restricted to evaluated cells. The GNN writes the same counts with separate calibration/test stages. Raw infinities raise before clipping; the shared metric functions use float64 and scaled RMSE/MAE. See [prediction safeguards](../methodology/prediction-safeguards.md).

The benchmark automatically loads `hyperparameters.yaml` from its output directory when present; direct baseline runs accept `--fixed_hyperparameters`. Its `models` mapping supplies parameters per model and trait, and skips inner CV while retaining outer-training preprocessing and fitting. `tuning.csv` records `parameter_selection=fixed`, parameters and elapsed fit time; the inner CV score is empty. Selection from the benchmark evaluation splits makes subsequent results exploratory.
