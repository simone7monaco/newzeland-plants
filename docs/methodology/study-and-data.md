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
  - {id: traits, resource: ../../data/Ferns/FernTraits.xlsx, title: Current analytical trait workbook}
  - {id: tree, resource: ../../ftol_tree.R, title: FTOL reconstruction used by the loader}
---

# Study and data

## Task

Predict an unobserved species–trait entry using other observed trait values and, depending on the model configuration, phylogenetic and environmental information. Each of six traits has minimum and maximum outputs: fertile and sterile stipe length, fertile and sterile frond length, spore size and altitude. The paper treats recorded bounds as ranges rather than individual-level uncertainty. The current min/max loader requires both bounds, may use a supplied range or compute one, and by default treats invalid bounds as missing.

## Sources and preprocessing

The manuscript describes traits collected from the Flora of New Zealand, NZPCN, Flora online and literature; species names reconciled against POWO and FTOL; screened GBIF and herbarium occurrences; and climate, soil, ecosystem, population and elevation layers. The exact unit conversions, outlier criteria, data-collection provenance and original tree grafting are not fully documented in the executable pipeline. Do not infer them from the workbook. The manuscript records equal fertile/sterile values for monomorphic fronds, a potential shortcut addressed in [training and masking](training-and-masking.md).

The complete source workbook and species list each contain **207 distinct taxa**. The current analytical workbook, `FernTraits.xlsx`, contains **189 taxa**, after excluding the 13 listed lycophytes (including Isoetaceae), four subspecies where the corresponding species is already represented, and `Hymenophyllum_falklandicum`, which has no observed min/max traits. This taxonomic screening is documented here to make the operational dataset reproducible; it was applied after the paper's study definition and is not a reference to introduce into the paper. All 189 rows have at least one nonempty min/max trait entry. FTOL 1.10.0 provides 166 matched tips (164 binomial matches and two synonym matches); 16 additional taxa are grafted at genus level. The resulting tree contains 182 taxa, while seven remain unresolved and have no phylogenetic edges or embeddings. All 189 taxa are used to train and evaluate the GNN, but comparisons with baselines are scored on the 182 tree-represented taxa, because tree-dependent baselines cannot predict the other seven; see [analysis conventions](../experiments/analysis-conventions.md). Each run saves the placement and tree membership of every taxon as `ftol_mapping.csv`. The complete source workbook is retained as `FernMinMax_all.xlsx`; historical workbooks are archived under `data/.soft_del/`.

The excluded lycophytes are `Isoetes_alpina`, `Isoetes_kirkii`, `Austrolycopodium_fastigiatum`, `Diphasium_scariosum`, `Huperzia_australiana`, `Lateristachys_diffusa`, `Lateristachys_lateralis`, `Palhinhaea_cernua`, `Phlegmariurus_billardierei`, `Phlegmariurus_varius`, `Phylloglossum_drummondii`, `Pseudodiphasium_volubile`, and `Pseudolycopodium_densum`. These are lycophytes, an ancient vascular-plant lineage outside the ferns (monilophytes); they are excluded to keep the cohort fern-only. The four excluded subspecies and their retained species-level representatives are:

| Excluded subspecies | Retained species-level representative |
|---|---|
| `Asplenium_appendiculatum_subsp._maritimum` | `Asplenium_appendiculatum` |
| `Asplenium_flaccidum_subsp._haurakiense` | `Asplenium_flaccidum` |
| `Davallia_tasmanii_subsp._cristata` | `Davallia_tasmanii` |
| `Dicksonia_lanata_subsp._hispida` | `Dicksonia_lanata` |

For these four, retaining both a species-level row and one separately measured subspecies would represent overlapping taxonomic units in the same species-level analysis. This is the operational rationale for dataset membership; it does not claim that subspecies are biologically invalid.

## Data and reproducibility

- Current min/max input: `data/Ferns/FernTraits.xlsx`; its unfiltered source is `data/Ferns/FernMinMax_all.xlsx`. Historical files are archived in `data/.soft_del/` and historical benchmarks must retain their original cohort labels.
- The loader constructs cached graph and environmental features. Cache files and source data are ignored by Git; record checksums, species list, tree and cache versions before regenerating results.
- The default tree is constructed by `ftol_tree.R` from the installed FTOL data via rpy2; the FTOL version, matching and grafting details are recorded in the [reconstruction update](../updates/ftol-reconstruction.md). The Newick fingerprint identifies the generated tree in processed graph caches.
