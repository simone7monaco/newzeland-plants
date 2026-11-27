---
type: Project update
title: Reviewer-response implementation and diagnostics
description: Recent masking, spatial clustering, presentation and ordered-bound changes, with validation and remaining work.
tags: [update, reviewer-response, implementation]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: ../methodology/occlusion-protocol.md, title: Paired occlusion protocol}
  - {resource: ../methodology/bounds-parameterization.md, title: Ordering of predicted bounds}
  - {resource: ../../spatial_cluster_knee.py, title: Spatial clustering diagnostic script}
  - {resource: ../../notebooks/02_ablation_analysis.ipynb, title: Ablation and bound audit notebook}
  - {resource: ../../TODO, title: Outstanding reviewer tasks}
---

# Reviewer-response implementation and diagnostics

## Species coverage

The current analytical workbook is `FernTraits.xlsx`, derived from `FernMinMax_all.xlsx` by excluding the 13 listed lycophytes (including Isoetaceae), four subspecies whose corresponding species is separately represented, and `Hymenophyllum_falklandicum`, which has no observed traits. This documents the taxonomic eligibility rule used to construct the dataset after the paper's study definition; it is not a reference to add to the paper. The resulting 189 rows all have at least one nonempty min/max trait entry. The FTOL 1.10.0 reconstruction matches 166 taxa to original tips (164 binomial and two synonym matches), grafts 16 at genus level, and leaves seven unresolved; the final tree contains 182 taxa. These counts changed after the dataset and FTOL update. The complete source workbook remains at `FernMinMax_all.xlsx`. Historical workbooks have been moved to `data/.soft_del/`; historical predictions retain their original cohort and masking settings. The grafts are documented genus-level approximations, and the seven unresolved species retain no phylogenetic descriptors or interspecies phylogenetic edges.

For reproducibility, the earlier 170-taxon subset omitted 37 taxa from the 207-row source workbook. Twelve of those were Lycopodiaceae and two were Isoetaceae; 18 were absent from the supplied tree and 19 matched its tips. Seventeen had both recorded fertile-stipe bounds equal to zero, compared with 16 of the 170 retained taxa. These are descriptive associations, not a known selection criterion. Version 7 of the loader reused the older occurrence cache; version 8 built edges uniformly for all 207 taxa; version 9 excludes all-missing taxa by default. The Rphylopars tree reader now normalizes Newick quote delimiters around tip names, correcting four tree-present taxa incorrectly omitted by earlier factorial runs; historical results are unchanged.

## Paired occlusion

**Problem.** Fertile and sterile traits can contain duplicated values, letting individual-trait masking expose the answer.

**Change.** Current code jointly hides the fertile/sterile stipe pair and the fertile/sterile frond pair, including both bounds and optional range, throughout training, validation and GNN/baseline evaluation. The default is paired; `individual` preserves the prior comparison. Groups are fixed for all species, independently of the held-out values. See [protocol](../methodology/occlusion-protocol.md).

**Draft response to reviewers 4.1 and 5.1.** We checked the final 189-taxon workbook and found identical minimum and maximum values in 163 of 182 complete fertile--sterile stipe pairs and 160 of 186 complete fertile--sterile frond pairs. We chose paired occlusion throughout training, validation and evaluation to prevent leakage from an identical counterpart, and revised the Methods accordingly. These counts describe equality rather than verified monomorphism. The benchmark rerun and subgroup analysis remain outstanding.

## Spatial clustering and presentation

**Change.** `spatial_cluster_knee.py` varies the number of clusters and saves cluster labels, cluster-size and spatial-compactness curves without modifying existing graph caches. It uses the same spectral clustering defaults as the loader. The ablation heatmap was redrawn with larger labels; manuscript language, caption, heuristic spatial-setting description and bound-violation disclosure were revised.

**Evidence and remaining work.** The 5–100-cluster diagnostic issued graph-connectivity warnings and did not measure predictive sensitivity. Rebuild spatial nodes, occurrence edges and environmental aggregates for each partition before running controlled benchmarks. Full manuscript proofreading and LaTeX compilation remain outstanding.

## Bounds and output heads

**Problem and audit.** Historical full-model outputs sometimes have min > max; see [diagnostics](../findings/reviewer-diagnostics.md). The notebook visualizes affected species/traits and seed counts from saved original-scale per-fold outputs.

**Change (adopted).** The model keeps independent min/max heads. After the inverse transform, reversed pairs are projected onto min ≤ max by replacing both bounds with their midpoint (`--enforce_bounds_order`, default on). The projection is applied to test outputs and to the internal-validation predictions used for conformal calibration. It guarantees ordering and cannot increase the joint squared error of a pair, because every observed pair is ordered. See [ordering of predicted bounds](../methodology/bounds-parameterization.md).

**Evidence.** In five matched seeds, 0.7–1.1% of test pairs were reversed before projection, all in stipe traits and nearly equal. The projection left the correlations unchanged and changed RMSE and MAE by at most about 0.01%. Two alternatives lost accuracy:
- min-plus-range heads (ReLU, Softplus): lower correlations, and overflow of the `Alt max` inverse with the shared min-fitted transform;
- a soft order penalty: no measurable gain, with reversals reduced only from 0.86% to 0.78%.

**Draft response to the reviewer.** We now enforce biologically consistent bounds. The model predicts minimum and maximum with separate heads and transformations. After back-transformation, any pair whose predicted minimum exceeds the predicted maximum is replaced by its midpoint. This is the Euclidean projection onto ordered pairs, so it cannot increase the joint error of any pair whose observed bounds are ordered. Reversals were rare before projection (0.7–1.1% of test pairs in matched runs), all for stipe length and with nearly equal bounds, and the projection left the reported correlations unchanged. We added this step to the Methods and to Table A.x.

*Note.* We also tried to enforce ordering within the network, without success:
- predicting the minimum plus a non-negative width (ReLU or Softplus) under a shared transformation reduced accuracy in all traits with strong signal, and was numerically unstable for maximum altitude;
- adding a soft penalty on reversed bounds to the loss did not change accuracy and removed few reversals.

The post-hoc projection was the only option that guaranteed ordering without cost.

**Remaining.** The final benchmark numbers in the Results must be updated after the pilot and the 60-seed run. Baselines use the same default projection; their reversals are counted before projection in `bounds_order*.csv` and the benchmark's `bounds_order_summary.csv`, but fresh rates have not yet been measured.
