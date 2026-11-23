---
type: Data provenance
title: Historical species coverage and recovery
description: Names omitted from the historical reduced workbook and recovery of the taxon without observed targets.
tags: [species, provenance, recovery]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: ../../data/Ferns/FernMinMax.xlsx, title: Historical reduced trait workbook}
  - {resource: ../../data/Ferns/FernMinMax_all.xlsx, title: Full source trait workbook}
---

# Historical species coverage and recovery

The old 170-taxon workbook and the 207-row source workbook remain intact. The 37 names below occur in the source workbook but not in the reduced workbook. All except `Hymenophyllum_falklandicum` meet the current default requirement of at least one observed target after cleaning. `Hymenophyllum_falklandicum` remains in the source workbook and can be restored as a graph node with `--include_unobserved_species true`; it has no observed target for scoring.

This record documents membership, not a verified reason for historical exclusion. The old workbook can be selected for historical reconstruction using `--traits_filename FernMinMax.xlsx` with the desired explicit masking setting. Historical benchmark results must not be mixed with future runs.

| Species or subspecies | Family |
|---|---|
| `Asplenium_appendiculatum_subsp._maritimum` | Aspleniaceae |
| `Asplenium_decurrens` | Aspleniaceae |
| `Asplenium_flaccidum_subsp._haurakiense` | Aspleniaceae |
| `Asplenium_gracillimum` | Aspleniaceae |
| `Austroblechnum_banksii` | Blechnaceae |
| `Austrolycopodium_fastigiatum` | Lycopodiaceae |
| `Botrychium_australe` | Ophioglossaceae |
| `Botrychium_biforme` | Ophioglossaceae |
| `Brownseya_serpentina` | Lycopodiaceae |
| `Cranfillia_deltoides` | Blechnaceae |
| `Davallia_tasmanii_subsp._cristata` | Davalliaceae |
| `Dicksonia_lanata_subsp._hispida` | Dicksoniaceae |
| `Diphasium_scariosum` | Lycopodiaceae |
| `Huperzia_australiana` | Lycopodiaceae |
| `Hymenophyllum_falklandicum` | Hymenophyllaceae |
| `Hypolepis_amaurorachis` | Dennstaedtiaceae |
| `Hypolepis_lactea` | Dennstaedtiaceae |
| `Hypolepis_rufobarbata` | Dennstaedtiaceae |
| `Isoetes_alpina` | Isoetaceae |
| `Isoetes_kirkii` | Isoetaceae |
| `Lateristachys_diffusa` | Lycopodiaceae |
| `Lateristachys_lateralis` | Lycopodiaceae |
| `Lecanopteris_novae-zealandiae` | Polypodiaceae |
| `Lecanopteris_pustulata` | Polypodiaceae |
| `Lecanopteris_scandens` | Polypodiaceae |
| `Notogrammitis_gunnii` | Polypodiaceae |
| `Palhinhaea_cernua` | Lycopodiaceae |
| `Phlegmariurus_billardierei` | Lycopodiaceae |
| `Phlegmariurus_varius` | Lycopodiaceae |
| `Phylloglossum_drummondii` | Lycopodiaceae |
| `Polystichum_sylvaticum` | Dryopteridaceae |
| `Pseudodiphasium_volubile` | Lycopodiaceae |
| `Pseudolycopodium_densum` | Lycopodiaceae |
| `Pteris_carsei` | Pteridaceae |
| `Pteris_epaleata` | Pteridaceae |
| `Pyrrosia_elaeagnifolia` | Polypodiaceae |
| `Tmesipteris_lanceolata` | Psilotaceae |
