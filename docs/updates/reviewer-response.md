---
type: Project update
title: Reviewer-response implementation and diagnostics
description: Recent masking, spatial clustering, presentation and ordered-bound changes, with validation and remaining work.
tags: [update, reviewer-response, implementation]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: ../../OCCLUSION_PROTOCOL.md, title: Paired occlusion implementation notes}
  - {resource: ../../BOUNDS_PARAMETERIZATION.md, title: Ordered output implementation notes}
  - {resource: ../../spatial_cluster_knee.py, title: Spatial clustering diagnostic script}
  - {resource: ../../notebooks/02_ablation_analysis.ipynb, title: Ablation and bound audit notebook}
  - {resource: ../../TODO, title: Outstanding reviewer tasks}
---

# Reviewer-response implementation and diagnostics

## Species coverage

The historical 170-taxon benchmark used the reduced `FernMinMax.xlsx`, which remains available unchanged. The default min/max loader and baselines now use `FernMinMax_all.xlsx` and, after cleaning, retain the 206 taxa with at least one observed trait. The workbook's remaining taxon, `Hymenophyllum_falklandicum`, is excluded by default and recoverable with `--include_unobserved_species true`; no workbook rows are deleted. Processed graph caches distinguish the workbook, tree and inclusion policy. The loader builds occurrence edges uniformly from the available species rasters. Of the 206 analytical taxa, 184 match tips in the supplied tree; the other 22 have zero phylogenetic embeddings and no interspecies phylogenetic edges. One matching taxon, `Ptisana_salicina`, has no such edges after thresholding, leaving 183 connected species. Supplied-tree membership does not establish original FTOL membership or the provenance of grafts. Tree-only baselines retain absent-tip taxa as rows with missing predictions and exclude them from scored cells. Historical predictions remain associated with their original cohorts and masking settings.

For reproducibility, the earlier 170-taxon subset omitted 37 taxa from the 207-row source workbook. Twelve of those were Lycopodiaceae and two were Isoetaceae; 18 were absent from the supplied tree and 19 matched its tips. Seventeen had both recorded fertile-stipe bounds equal to zero, compared with 16 of the 170 retained taxa. These are descriptive associations, not a known selection criterion. Version 7 of the loader reused the older occurrence cache; version 8 built edges uniformly for all 207 taxa; version 9 excludes all-missing taxa by default. The Rphylopars tree reader now normalizes Newick quote delimiters around tip names, correcting four tree-present taxa incorrectly omitted by earlier factorial runs; historical results are unchanged.

## Paired occlusion

**Problem.** Fertile and sterile traits can contain duplicated values, letting individual-trait masking expose the answer.

**Change.** Current code jointly hides the fertile/sterile stipe pair and the fertile/sterile frond pair, including both bounds and optional range, throughout training, validation and GNN/baseline evaluation. The default is paired; `individual` preserves the prior comparison. Groups are fixed for all species, independently of the held-out values. See [protocol](../../OCCLUSION_PROTOCOL.md).

**Evidence and remaining work.** The workbook contains identical min/max pairs in 147/165 complete stipe pairs and 146/170 complete frond pairs. These are equality counts, not verified monomorphism labels. Automated tests and short functional runs passed. Full matched repeated-seed experiments and subgroup analyses have not been completed.

## Spatial clustering and presentation

**Change.** `spatial_cluster_knee.py` varies the number of clusters and saves cluster labels, cluster-size and spatial-compactness curves without modifying existing graph caches. It uses the same spectral clustering defaults as the loader. The ablation heatmap was redrawn with larger labels; manuscript language, caption, heuristic spatial-setting description and bound-violation disclosure were revised.

**Evidence and remaining work.** The 5–100-cluster diagnostic issued graph-connectivity warnings and did not measure predictive sensitivity. Rebuild spatial nodes, occurrence edges and environmental aggregates for each partition before running controlled benchmarks. Full manuscript proofreading and LaTeX compilation remain outstanding.

## Bounds and output heads

**Problem and audit.** Historical full-model outputs sometimes have min > max; see [diagnostics](../findings/reviewer-diagnostics.md). The notebook now visualizes affected species/traits and seed counts from saved original-scale per-fold outputs.

**Change.** `--bounds_parameterization` selects independent min/max, min plus ReLU range, or min plus Softplus range. Ordered variants share the transformation fitted on training minima and compute the loss on predicted min and max, not on raw width. `--shared_bounds_normalization true` enables an independent-head comparison using the same transform. Range is in transformed coordinates; existing inverse and softplus clipping give output-scale bounds when finite. See [implementation details](../../BOUNDS_PARAMETERIZATION.md).

**Evidence and remaining work.** Automated tests and short one-fold smoke runs passed; both ordered heads had zero inverted bounds among 195 scored pairs in those runs. No matched full benchmark or statistical comparison has been performed. Check inverse-domain numerical stability and use distinct output directories for every head because benchmark resumption does not fully fingerprint run settings.
