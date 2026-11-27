---
type: Update
title: FTOL reconstruction and comparison
status: draft
---

# FTOL reconstruction and comparison

`loader.py` builds the phylogeny from `ftol_tree.R` through rpy2 using the taxa in the selected trait workbook. The R environment needs `ftolr`, `ape`, `phytools`, `taxastand` and `pteridocat`. Install `ftolr` and `taxastand` from `https://joelnitta.r-universe.dev` and `pteridocat` from `https://github.com/fernphy/pteridocat`; configure `R_HOME` and `R_LIBS_USER` for the chosen R environment. The verified versions are FTOL data 1.10.0, ftolr 1.10.0, pteridocat 1.2.0, taxastand 2.0.0.9000, ape 5.8.1 and phytools 2.5.2. These versions document the validated run but are not pinned by the environment; package versions and FTOL releases can change the tree; the generated Newick hash participates in graph cache naming. Clade-blocked folds use this same reconstructed tree through `data.tree_newick`; the analysis does not depend on an independently supplied local tree.

For this checkout, the three additional packages were restored from an existing package installation into `.local/r-library`; the loader adds this directory to R's library paths when present. This installation is machine-specific and is not a dependency lockfile. The R runtime and system library path must still be configured for the selected environment.

For `FernTraits.xlsx` (189 taxa), FTOL 1.10.0 yielded 164 exact binomial matches, two synonym matches, 16 genus grafts and seven unresolved taxa. The unresolved taxa are `Doodia_milnei`, `Alsophila_dealbata`, `Hypolepis_lactea`, `Hypolepis_rufobarbata`, `Leptolepia_novae-zelandiae`, `Brownseya_serpentina` and `Pellaea_calidirupia`. The two Hypolepis names resolve to the same FTOL tip and are both left unresolved rather than creating an unsupported duplication. The other five have an absent or non-monophyletic genus. Unresolved taxa remain graph nodes with zero phylogenetic embeddings and no phylogenetic edges.

Earlier documentation compared the reconstructed FTOL tree with a local tree file; that file has since been moved to `data/.soft_del/`. The comparison is historical context only. All current tree-derived features and clade folds use the tree produced by `ftol_tree.R` from the 189-taxon workbook.
