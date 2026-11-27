---
type: Update
title: Component ablation of family, embeddings, species edges and environment
description: Separate model switches and an incremental/leave-one-out/factorial ablation runner for reviewer comments R2.3, R4.2 and R4.7.
tags: [ablation, phylogeny, environment, reviewer-response]
status: implemented-not-run
generated: {by: claude/opus-5.5}
sources:
  - {resource: ../../run_component_ablation.py, title: Ablation runner}
  - {resource: ../../train.py, title: Training entry point and component flags}
  - {resource: ../../models.py, title: TraitsPredictor}
  - {resource: ../../tests/test_component_ablation.py, title: Unit tests}
---

# Component ablation

## Problem

Reviewers noted that the no-phylogeny configuration keeps family among the categorical inputs and removes Node2Vec descriptors and species edges together, so the phylogenetic and environmental contributions cannot be separated (R2.3, R4.2, R4.7). See [graph and model](../methodology/graph-and-model.md).

## Implementation

`train.py` gains three switches; `--use_env_features` is unchanged:

| Component | Flag | When off |
|---|---|---|
| Family | `--use_family` | `Family_*` dummies are removed from `species_x_gen` before fold normalization; Habitat remains |
| Phylogenetic embeddings | `--use_phylo_embeddings` | Node2Vec descriptors are omitted from the species and bipartite inputs |
| Species edges | `--use_phylo_edges` | The species GNN uses self-loops only, so no message passing between species |
| Environment | `--use_env_features` | No spatial graph or bipartite location-to-species pathway |

The two phylogenetic switches default to `--use_phylo_features`, so earlier commands and the benchmark reproduce their previous behaviour. Run names gain `_no-phylo-emb`, `_no-phylo-edges` and `_no-family` suffixes when relevant.

`run_component_ablation.py` calls `train.py` for every configuration on shared seeds, five folds, occlusion protocol and hyperparameters (`--gnn-args`). Designs:

- `incremental` (default): from the trait-only model, add components in the `--order` (default `family,emb,edges,env`). `--order` can be repeated to compare alternative paths; shared configurations run once.
- `leave-one-out`: the full model minus each component.
- `factorial`: all 16 combinations, for interaction estimates.

Outputs: `runs/seed_<s>/gnn/<config>/` (same layout as `run_benchmark.py`), `ablation_manifest.csv` (flags, command, runtime and status per job), `run_settings.txt` (data hashes, git commit, protocol) and `ablation_predictions_{min,max,long}_all.parquet` (original scale, species as a column) with `configuration`, `seed` and `use_*` columns. `--dry-run` prints the plan without training.

Example: `python run_component_ablation.py 10 --design incremental,leave-one-out --output-dir ablation_results`

## Limits

- Hidden sizes are fixed, so only input-layer width changes between configurations; parameter counts are not recorded yet.
- Without species edges, the species GNN still applies its layers through self-loops, so the model remains a per-species network with the same depth.
- Interpretation depends on the order in incremental designs; use leave-one-out or factorial results to check order effects.
- The non-graph random forest, XGBoost and MLP baselines requested in the same comments are run by `r_baselines_fit.py`/`run_benchmark.py`, not by this runner; see [ML baselines](ml-baselines.md).

## Status

Implemented and unit-tested (`tests/test_component_ablation.py`; forward pass and plans only). No ablation has been run and no results exist.
