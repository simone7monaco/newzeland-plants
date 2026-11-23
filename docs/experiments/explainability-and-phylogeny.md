---
type: Experimental protocol
title: Attribution and phylogenetic error analyses
description: Integrated Gradients, species-level error changes and phylogenetic diagnostics.
tags: [integrated-gradients, phylogeny, moran, attribution]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: "../../Paper/'main.tex", title: Manuscript attribution and residual analyses}
  - {resource: ../../tester.py, title: Target-masked Integrated Gradients}
  - {resource: ../../notebooks/Lambda signal code.R, title: Lambda plotting source}
  - {resource: ../../notebooks/Morans I plotting code.R, title: LIPA plotting source}
  - {resource: ../../notebooks/Quantile plot.R, title: Quantile plotting source}
---

# Attribution and phylogenetic error analyses

The paper uses Integrated Gradients with a zero input baseline to measure fitted-model sensitivity for each target and reports environmental feature and domain aggregates. The tester masks the active target at inference and computes species and spatial attributions. A feature attribution is conditional on the chosen model, inputs and baseline; it is not causal evidence or proof that the feature improves held-out prediction. Correlation, baseline choice and model-seed stability remain to be assessed.

The manuscript defines a species-level transition as `(absolute error of reference − absolute error of augmented model) / observed value`, with positive values indicating reduced error. Small or zero observed values make this metric unstable or undefined. The paper describes Pagel's lambda and local Moran's I, 999 Monte Carlo permutations and a nominal 0.05 threshold; persistent categories are summarized across seeds. The supplied R scripts inspected here regenerate plots from saved summaries; they do not by themselves establish the upstream generation, exact two-sided permutation implementation, or a multiple-testing correction. Verify the upstream analysis and correct the comparison family before treating local calls as confirmed.

The quantile plots consume `Trait_Quantile_Delta_Mean_SD_Across_Seeds.csv`; lambda and local Moran figures likewise use saved analysis artifacts. Preserve inputs, code, seed and taxon mapping in any rerun. See [reviewer diagnostics](../findings/reviewer-diagnostics.md) and [TODO](../../TODO) for required robustness checks.
