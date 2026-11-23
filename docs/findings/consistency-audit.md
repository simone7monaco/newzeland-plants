---
type: Research audit
title: Manuscript and implementation consistency audit
description: Pre-existing manuscript-to-code and manuscript-to-artifact discrepancies and unresolved evidence.
tags: [audit, reproducibility, limitations]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: "../../Paper/'main.tex", title: Current manuscript}
  - {resource: ../../loader.py, title: Data preparation and graph loader}
  - {resource: ../../train.py, title: Training and preprocessing}
  - {resource: ../../models.py, title: Loss and masking}
  - {resource: ../../tester.py, title: Model evaluation}
  - {resource: ../../notebooks/02_ablation_analysis.ipynb, title: Aggregation notebook}
  - {resource: ../../data/Ferns/FernMinMax.xlsx, title: Current workbook}
  - {resource: ../../data/Ferns/grafted_tree.nwk, title: Current tree}
---

# Manuscript and implementation consistency audit

Scope: discrepancies in the pre-existing paper, legacy experimental setup and saved results. Recent changes to occlusion, output heads and spatial diagnostics are documented in [updates](../updates/reviewer-response.md), not treated here as historical manuscript failures. The current source tree cannot establish the exact historical code revision without run provenance.

## Confirmed discrepancies

1. **Sample size and tree coverage — methodological text aligned.** The workbook and species list contain 207 taxa; the standard analytical cohort now contains 206 after excluding the species without observed traits. Of these, 184 match tips in the supplied, potentially grafted tree and 22 do not. Earlier manuscript counts 208 and 174 did not match these artifacts and were removed from the methodological description. The original FTOL coverage and graft provenance remain unresolved. The historical 170-taxon benchmark is preserved and must not be presented as a result of the new 206-taxon protocol.
2. **Aggregate terminology.** Results and Discussion describe 4.511 as the median across trait–variable slices. `02_ablation_analysis.ipynb` first takes the mean across slices within each seed, then the median across seeds. A different statistic—the median of the twelve Table 1 full-model entries—is about 1.535. The number 4.511 is not necessarily wrong; its stated aggregation is incomplete.
3. **Training objective — methodological text aligned.** The earlier manuscript equation used a globally weighted sum divided by `2W`, whereas the current implementation computes separate masked and visible Huber means, by default first averaging per trait, and adds `masked + 0.2 × visible`. The manuscript equation now describes the current default. The exact historical run implementation must still be checked when interpreting its results.
4. **Deterministic-seed wording.** The manuscript says deterministic methods produce the same cross-validation results across different splits. Deterministic fitting is repeatable for the same data and split, but training statistics and predictions can change when species folds change; the current code uses the seed to shuffle folds. The claim needs correction.
5. **Table 1 minimum/maximum labels.** Recomputing the full-model per-output mean NRMSE-IQR across the 60 saved seeds shows that every min/max pair in Table 1 is labelled in the opposite order from the saved metrics. For example, saved `Alt min` is 10.262 and `Alt max` is 0.797; the table labels 10.262 as maximum and 0.797 as minimum. Saved fertile-frond min/max are 1.662/0.530, also reversed in the table. Recheck the source export and regenerate all rows and dependent Results/Discussion claims before correcting the manuscript. Separately, the text attributes 0.787 to training mean, whereas the table assigns 0.787 to training median and 0.790 to training mean; the labels themselves must first be repaired.

## Material uncertainties and setting limitations

- **Tree handling.** Methods now match the loader: taxa absent from the supplied tree have zero embeddings and no interspecies phylogenetic edges. The supplied tree may already contain grafted approximations. Trace tree creation and original FTOL naming before assigning source-FTOL coverage or approximation counts.
- **Validation preprocessing.** Outer-test traits are excluded from fitted trait transforms, but normalization is fitted before selection of the internal validation cells. The internal validation mask is therefore not independent of all fitted preprocessing. All shared spatial nodes participate in fitting environmental normalization; no spatial holdout is performed.
- **Legacy duplicate traits.** In the historical one-trait-at-a-time evaluation, an identical fertile/sterile counterpart could remain visible. The current default and manuscript protocol use paired masking; historical outputs do not. Matched repeated-seed reruns are required.
- **Attribution and tests.** Family remains in the no-phylogeny configuration, and the phylogeny switch jointly changes descriptors and species graph. Attributions are model sensitivity, while proportional errors can be unstable near zero. Local Moran multiplicity handling and the upstream creation of saved summaries were not established from plotting scripts.
- **Preprocessing and coverage provenance.** The source records, units, outlier rules, cache creation and original per-run software/data versions are not sufficiently documented to independently reproduce all paper numbers. Output filenames and current defaults alone cannot prove which version produced historical artifacts.
- **MICE transform semantics.** The wrapper invokes `mice.mids` on new data. The paper's stronger assertion that the fitted CART predictor remains fixed needs verification against the R operation; this audit does not establish that it is incorrect.
- **Rphylopars and quoted tree tips.** `ape::read.tree` retains literal single quotes around four subspecies names in the supplied Newick tree, while Bio.Phylo returns the names without those quotes. The wrapper now removes matching Newick quote delimiters before fitting, so tree-present names match consistently. The earlier full-dataset factorial runs were generated before this correction and contain missing Rphylopars predictions for these four tree-present subspecies in addition to the 22 taxa absent from the supplied tree. The affected names are `Asplenium_appendiculatum_subsp._maritimum`, `Asplenium_flaccidum_subsp._haurakiense`, `Davallia_tasmanii_subsp._cristata`, and `Dicksonia_lanata_subsp._hispida`. Do not mix historical Rphylopars outputs with corrected future runs.
- **Significance and independence.** The paper makes global significance statements and ranks point estimates. Paired repeated splits overlap in species; a clear test, comparison family and appropriate uncertainty analysis are needed before interpreting tiny differences as superiority or equivalence.
