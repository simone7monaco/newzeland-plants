---
type: Experimental protocol
title: Species coverage by occlusion factorial benchmark
description: Four matched-fold experiments and common-species outcome analysis.
tags: [benchmark, masking, taxa, factorial]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: ../../run_species_occlusion_experiment.py, title: Factorial benchmark runner}
  - {resource: ../../summarize_species_occlusion.py, title: Common-species outcome analysis}
  - {resource: ../../loader.py, title: Stable species folds}
  - {resource: ../../data/.soft_del/FernMinMax.xlsx, title: Limited species dataset}
  - {resource: ../../data/Ferns/FernMinMax_all.xlsx, title: Complete species dataset}
---

# Species coverage by occlusion factorial benchmark

**Historical experiment:** the completed run used the earlier 207-node configuration, including a taxon without observed traits. The current analytical workbook has 189 rows, all with recorded traits; do not treat the historical results as a rerun of this cohort.

The current standard analytical workbook contains 189 taxa. This historical script explicitly selects its input workbooks, so its runs must be identified by those inputs rather than assumed to use the current standard cohort. They require separate output and do not reproduce the completed 207-node run by default.

Four conditions cross the limited (170 species) and complete (207 species) workbooks with individual-trait and paired fertile/sterile occlusion. The comparison uses one seed (42), 1,000 training epochs, five outer species folds, independent min/max output heads, and the full environment-plus-phylogeny GNN. It also runs the existing six baseline models under the same protocol. Integrated Gradients is disabled because the question concerns predictive errors. The one-seed design is exploratory and does not quantify variability over dataset splits and training randomness.

`--split_strategy stable` assigns each species to a fold using a hash of its name and the seed. The 170 common species thus keep the same test fold in all four conditions even when the dataset changes; train graph, neighbours, preprocessing and the set of other test species can still change, as intended for the dataset intervention. The previous manuscript's random KFold allocation is different, so comparing numerical results to historical results is descriptive rather than a paired counterfactual.

Run `run_species_occlusion_experiment.py` to create isolated conditions and per-job logs and completion markers. It uses source and input fingerprints to protect resume semantics, prepares both dataset caches serially and can execute independent jobs concurrently. Results are incomplete until all required jobs have `success.json`. `summarize_species_occlusion.py` compares original-scale per-trait errors on the intersection of the original 170 species and the valid scored cells in all four conditions, using the limited-workbook IQR for a common metric denominator. Effects are displayed as signed differences in NRMSE-IQR, with negative values indicating improvement. Tree-only baselines cannot predict taxa absent from the tree and therefore may have smaller common evaluation sets. The GNN historical seed-42 comparison uses saved historical output and is explicitly labelled as having different folds.

For each output, the paired-minus-individual contrasts at each dataset size describe the effect of paired masking; the complete-minus-limited contrasts at each masking setting describe the effect of including more species. Summaries are descriptive averages or medians across the twelve outputs and must be read with the per-trait results: high NRMSE outputs can dominate an average.

The summarizer also produces a separate full-population result for the two complete-dataset conditions, using their common evaluable cells and the full workbook's IQR. Its cohort and denominator differ from the common-cohort comparison, so these aggregate scores are not directly comparable to the 170-species scores.

## Resumption after interruption

The first execution completed 14 of 20 GNN folds before interruption. `resume_species_occlusion.py` reruns the remaining six folds separately in `factorial_benchmark_results/recovery/`, preserving the completed folds. A resumed fold starts its random generator from seed 42 as a standalone process, whereas a fold in the original five-fold process inherits the advanced random state. The species-fold allocation remains matched, but initialization and stochastic-mask realizations are not matched counterfactually across all four conditions. Treat differences as exploratory and confounded by training randomness; a fully controlled follow-up should use reproducible fold-specific seeds in every condition and rerun all folds. Baseline predictions from the original four conditions were completed before resumption.
