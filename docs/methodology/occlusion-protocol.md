---
type: Research methodology
title: Fertile/sterile occlusion protocol
description: Paired masking of fertile and sterile stipe and frond traits in training, inner validation and evaluation.
tags: [occlusion, masking, leakage, reviewer-response]
status: draft
generated: {by: claude/opus-5.5}
sources:
  - {resource: ../../occlusion.py, title: Occlusion groups}
  - {resource: ../../train.py, title: Training and inner validation masks}
  - {resource: ../../tester.py, title: Grouped target occlusion at evaluation}
  - {resource: ../../r_baselines_fit.py, title: Baseline occlusion}
  - {resource: ../../run_benchmark.py, title: Benchmark protocol option}
---

# Fertile/sterile occlusion protocol

The paired protocol groups `Ferstipe` with `SterileStipe` and `FertileFrond` with `SterileFrond`. Other traits form single-member groups. Groups are defined by trait names for every species, not by whether the held-out values are equal or by a monomorphism label. This avoids using the answer to select the mask, and it evaluates the same task for all species.

## Training, validation and testing

- **Training.** One mask decision is sampled per species and group.
- **Inner validation.** Observed members of the selected groups are hidden together. Entries that were already missing stay missing and are not scored. Validation selection keeps at least one training observation per trait when feasible. The inner-validation cells are also excluded from the normalization fit; see [training and masking](training-and-masking.md).
- **Testing.** For each target trait, all members of its group are hidden for every test species, including test nodes where the target itself was already missing. Both bounds and any range input are hidden. Training-species observations remain available as graph context. Metrics are computed only for originally observed target cells.

This is a transductive evaluation, not a test on unseen graph nodes.

## Options

The default `--occlusion_protocol paired` applies to `train.py` and `r_baselines_fit.py`. Use `--occlusion_protocol individual` for the earlier single-trait comparison. In `run_benchmark.py` the equivalent option is `--occlusion-protocol paired` or `individual`.

Use a separate, fresh benchmark output directory per protocol. The benchmark records the protocol and rejects a conflicting or unlabelled existing run. Direct model runs also include the protocol in their output directory names.

When comparing protocols:
- match seeds, folds, representations and other settings;
- retrain the GNN under both settings, rather than only evaluating a previously trained checkpoint.

## Historical source-data audit

The historical reduced workbook in `data/.soft_del/FernMinMax.xlsx` has 170 unique species rows. Among rows with both bounds observed for both members, 147/165 stipe pairs and 146/170 frond pairs have identical minima and maxima. Equality indicates a possible shortcut in the data; it does not prove monomorphism.

These counts describe the historical reduced workbook. The complete source workbook contains 207 taxa. The current analytical workbook, `data/Ferns/FernTraits.xlsx`, contains 189 rows, all with at least one nonempty min/max trait entry. Its taxonomic screening is documented in [study and data](study-and-data.md) and is not a reference to introduce into the paper.

## Remaining analysis

- Run both protocols with identical repeated species folds and all selected baselines.
- Report the paired change in error per output and the evaluated species counts.
- Where labels are available, report independently verified monomorphic and dimorphic taxa separately.
- Record the exact data and checkpoint versions. A one-epoch smoke run is not evidence of predictive performance.

The paired protocol reduces direct fertile/sterile leakage. It does not prevent information transfer through other correlated features or through test-node graph connections.
