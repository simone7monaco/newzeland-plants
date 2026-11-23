---
type: Research methodology
title: Study and data
description: Trait imputation task, source files, observed coverage and provenance limitations.
tags: [data, traits, phylogeny, geography]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {id: paper, resource: "../../Paper/'main.tex", title: Manuscript, Methods and Results}
  - {id: loader, resource: ../../loader.py, title: Dataset loading and graph construction}
  - {id: traits, resource: ../../data/Ferns/FernMinMax_all.xlsx, title: Full trait workbook}
  - {id: tree, resource: ../../data/Ferns/grafted_tree.nwk, title: Processed phylogenetic tree}
---

# Study and data

## Task

Predict an unobserved species–trait entry using other observed trait values and, depending on the model configuration, phylogenetic and environmental information. Each of six traits has minimum and maximum outputs: fertile and sterile stipe length, fertile and sterile frond length, spore size and altitude. The paper treats recorded bounds as ranges rather than individual-level uncertainty. The current min/max loader requires both bounds, may use a supplied range or compute one, and by default treats invalid bounds as missing.

## Sources and preprocessing

The manuscript describes traits collected from the Flora of New Zealand, NZPCN, Flora online and literature; species names reconciled against POWO and FTOL; screened GBIF and herbarium occurrences; and climate, soil, ecosystem, population and elevation layers. The exact unit conversions, outlier criteria, data-collection provenance and original tree grafting are not fully documented in the executable pipeline. Do not infer them from the workbook. The manuscript records equal fertile/sterile values for monomorphic fronds, a potential shortcut addressed in [training and masking](training-and-masking.md).

The full workbook and species list each contain **207 distinct taxa**. The standard min/max dataset contains **206 taxa with at least one usable observed trait** after cleaning. The excluded taxon, `Hymenophyllum_falklandicum`, has no observed min/max targets; it remains in the unmodified workbook and can be restored for exploratory inference with `--include_unobserved_species true`. Of the 206 analytical taxa, 184 match a tip in the supplied grafted tree and 22 do not. After distance thresholding, 183 taxa have at least one interspecies phylogenetic edge; `Ptisana_salicina` matches a tip but has no such edge. A match to the supplied tree does not establish presence in original FTOL. The former 170-taxon workbook remains available for historical recovery; see the [audit](../findings/consistency-audit.md).

## Data and reproducibility

- Current min/max input: `data/Ferns/FernMinMax_all.xlsx`; the historical benchmark under `benchmark_results/runs/` used `FernMinMax.xlsx` and must not be relabelled as a 206-taxon experiment.
- The loader constructs cached graph and environmental features. Cache files and source data are ignored by Git; record checksums, species list, tree and cache versions before regenerating results.
- The tree graph uses the supplied Newick file, not a reproducible FTOL download in the Python pipeline. Source coverage and name reconciliation require an upstream audit.
