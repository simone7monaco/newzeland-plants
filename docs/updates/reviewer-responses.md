---
okf_version: "0.2"
type: Reviewer response
title: Consolidated reviewer responses
description: Reviewer comments and response drafts for the PhyEnv-NN New Zealand fern trait-imputation study.
tags: [reviewer-response, rebuttal, phyenv-nn]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: "https://app.notion.com/p/41e533e2ce7e80caad8dc0243ace40e8", title: "Reviewer FernStuff: reviewer comments and response drafts"}
  - {resource: ../methodology/study-and-data.md, title: "Current analytical cohort and tree coverage"}
  - {resource: ../findings/consistency-audit.md, title: "Manuscript, artifact and provenance audit"}
---

# Consolidated reviewer responses

## Preamble

This file reorganizes the reviewer questions and response drafts from the linked Notion source into a copyable Markdown structure. Reviewer questions are preserved verbatim. Each answer is a draft; statements about completed analyses or manuscript changes must be checked against the corresponding artifacts before submission.

A later draft is treated as the current version when it updates an earlier response. Where an earlier draft contains a distinct proposed reply, it is retained after the current draft as **Seconda bozza**. Response and similarity-note colors from Notion are represented with inline HTML background/text styles so the visual distinction survives Markdown rendering and can be adapted when pasted back to Notion.

**Cohort-count reconciliation:** Following a dataset and FTOL reconstruction update, the current workbook contains 189 taxa: 166 matched original FTOL tips, 16 were grafted at genus level, and seven remained unresolved. The final tree therefore contains 182 taxa. Benchmark results must be rerun on this cohort and tree before claims about their performance are finalized. Original reviewer wording is preserved as received.

**Color key:** response blocks retain their source background color; “Commenti simili” notes retain the yellow background and orange label styling. “Commenti simili: nessuno” means no parallel reviewer point was identified in the source notes.

## Reviewer 2

### Major comments

1. Novelty and contribution to the state of the art: The manuscript should more clearly establish the methodological novelty of PhyEnv-NN compared with existing approaches for phylogenetically informed trait imputation. Previous studies, including Debastiani et al. (2021), published in Ecological Informatics, have already demonstrated the benefits of incorporating phylogenetic information into machine learning models for missing trait estimation. The authors should clarify which specific limitations of previous approaches are addressed by the proposed heterogeneous graph architecture and explain how this integration provides additional scientific or methodological value.
   > <span style="background-color: #f4cccc">*Risposta/intervento:* We thanked the reviewer for identifying the need for a more precise comparison. We revised the Introduction and Discussion to distinguish the integration of species and location graphs, phylogenetic representations, observed traits and environmental variables from the established use of phylogeny in machine-learning imputation. We compared the inputs, assumptions and scope with Debastiani et al. (2021), and assessed the added value of graph message passing using matched non-graph baselines (random forest, XGBoost and a multilayer perceptron given the same species-level information) and controlled ablations. The revised claims were aligned with those results.</span>
   >
   > <span style="background-color: #f4cccc">*Seconda bozza:* We thank the reviewer for identifying the need for a more precise comparison. We will revise the Introduction and Discussion to distinguish the integration of species and location graphs, phylogenetic representations, observed traits and environmental variables from the broader, established use of phylogeny in machine-learning imputation. We will compare the inputs, assumptions and scope with **Debastiani et al. (2021)**, and assess the added value of graph message passing using matched non-graph baselines and controlled ablations. Any claim of advantage will depend on those results.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: nessuno</span>** — Novità metodologica non discussa direttamente dagli altri.</span>
2. Comparison with alternative machine learning models: Although the proposed framework is compared with several statistical and phylogenetic imputation method, the experimental evaluation does not include a conventional machine learning model using the same input information. I recommend incorporating at least one additional comparison with Random Forest, XGBoost, or a multilayer perceptron using equivalent morphological, phylogenetic, and environmental predictors. This would help determine whether the observed improvements originate from the graph-based architecture or from the additional information available to the proposed model.
   > <span style="background-color: #f4cccc">*Risposta/intervento:* We added three tuned non-graph regressors, a random forest, gradient-boosted trees (XGBoost) and a multilayer perceptron, which received the same available trait values, missingness indicators, categorical variables, phylogenetic descriptors (Node2Vec) and aggregated environmental information (occurrence-weighted means and standard deviations of the location covariates). Hyperparameters were tuned by randomised search with species-level cross-validation within the outer-training species, and the target's occlusion group was never used as input. MICE received the same phylogenetic and environmental summaries. In Table 1 these baselines use both information sources, matching the full model, and they were also fitted with the other three input combinations. All methods used the same species folds, target masks and training-only preprocessing, and were scored on identical cells of the 182 taxa that every method could predict. Because species-level summaries could not reproduce every graph operation, we described the remaining difference in representation in the Methods and reported paired per-output results. The models, inputs and search spaces are described in the Methods and in a new appendix table.</span>
   >
   > <span style="background-color: #f4cccc">*Seconda bozza:* We improved the experimental comparison by adding a tuned random forest, an XGBoost, and a Multilayer perceptron with the same available trait values, missingness indicators, categorical variables, phylogenetic descriptors and aggregated environmental information. All methods use the same species folds, target masks and training-only preprocessing. Because species-level summaries cannot reproduce every graph operation, we will describe the remaining difference in representation explicitly and report paired per-output results.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.7</span>** — Baseline non-graph con gli stessi input.</span>
3. Ablation study and contribution of individual components: The ablation analysis provides useful information regarding the contribution of phylogenetic and environmental variables. However, it is not entirely clear whether removing an information source also removes its corresponding graph connections and message-passing operations. Furthermore, taxonomic family information may retain phylogenetically related information even when explicit phylogenetic descriptors are excluded. The authors should provide a more detailed description of the ablation configurations and, if possible, separately evaluate the contributions of phylogenetic embeddings, graph connectivity, and taxonomic attributes.
   > <span style="background-color: #f4cccc">*Risposta/intervento:* Disabling phylogeny removed the explicit embeddings and replaced species-to-species phylogenetic edges with self-loops; disabling environment removed the spatial and location-to-species pathways. Categorical attributes, including family, remained. We documented this distinction and added controlled configurations separating family, phylogenetic embeddings and species graph connections, as well as the environmental pathways, to assess their incremental contributions. In addition, the non-graph baselines (MICE, random forest, XGBoost and MLP) were fitted with each combination of environmental and phylogenetic inputs, showing the contribution of each information source without graph message passing.</span>
   >
   > <span style="background-color: #f4cccc">*Seconda bozza:* In the current implementation disabling phylogeny removes the explicit embeddings and replaces species-to-species phylogenetic edges with self-loops; disabling environment removes the spatial and location-to-species pathways. Categorical attributes, including family, remain. We will document this distinction and add controlled configurations separating family, phylogenetic embeddings and species graph connections, as well as the environmental pathways, to assess their incremental contributions.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.2</span>** — Separazione dei contributi di filogenesi e graph.</span>
4. Statistical validation of the reported improvements: The manuscript reports that PhyEnv-NN achieves the first or second position for 11 of the 12 predicted outputs. However, ranking among competing methods does not necessarily establish statistically significant superiority. The results in Table 1 should be complemented with measures of variability, confidence intervals, and appropriate paired statistical comparisons. Additionally, the relatively small aggregate improvement observed in the ablation analysis should be discussed in relation to the additional computational complexity of the proposed framework.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We reported per-output errors and paired model differences with uncertainty intervals, using a resampling strategy accounting for repeated, overlapping species splits and a defined multiple-comparison correction. We also reported training and inference costs and discussed the practical magnitude of the differences. Rankings were not treated as evidence of statistical superiority.</span>
   >
   > <span style="background-color: #fff2cc">*Seconda bozza:* We reported per-output errors and paired model differences with uncertainty intervals, using a resampling strategy that accounts for repeated, overlapping species splits, and a defined multiple-comparison correction.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.6, R5.4</span>** — Test statistici e intervalli sui confronti.</span>
5. Generalization and robustness: The evaluation is conducted using 208 fern species and subspecies from New Zealand, providing a geographically and taxonomically specific case study. Although species-level cross-validation is appropriate for evaluating predictions on held-out species, it does not fully demonstrate generalization to independent phylogenetic groups or different geographical regions. I recommend including a taxonomically structured validation experiment, if feasible, or explicitly acknowledging this limitation in the discussion and conclusions.
   > <span style="background-color: #f4cccc">*Risposta/intervento:* Following a dataset update, the analytical workbook contains 189 taxa with observed traits. Species-level evaluation uses held-out taxa within the New Zealand dataset; full-graph evaluation is transductive and can include close relatives of a test taxon. This does not establish generalization to unseen phylogenetic groups or regions. The benchmark must be rerun on the updated cohort.</span>
   >
   > <span style="background-color: #f4cccc">*Seconda bozza:* You are correct. We included a comment to highlght this in the discussion</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.4</span>** — Validazione per gruppi tassonomici distinti.</span>
6. Treatment of missing phylogenetic information: The manuscript indicates that species absent from the phylogenetic tree were approximated using genus- or family-level relationships. However, the graph construction section also describes missing species as isolated nodes with zero phylogenetic descriptors. The authors should clarify how these species were actually represented and whether their inclusion affected the prediction results.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* Following an update to the dataset and FTOL reconstruction, 166 of 189 taxa match original FTOL tips (164 by binomial and two by synonym), 16 are grafted at genus level, and seven remain unresolved. The resulting tree contains 182 taxa. The seven unresolved taxa retain their other available features and spatial connections, with zero phylogenetic descriptors and no interspecies phylogenetic edges. Comparisons with baselines were scored on the 182 tree-represented taxa for every method, because tree-dependent baselines (Phylo NN, Phylo KNN, Rphylopars) cannot predict the seven unresolved taxa; the GNN was still trained and evaluated with all 189 taxa in the other analyses. The effect on prediction requires a matched sensitivity analysis.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.3, R3.2</span>** — Specie senza albero e approssimazioni filogenetiche.</span>
7. Prediction uncertainty and missing-data mechanisms: The manuscript primarily evaluates reconstruction performance using normalized RMSE. While this metric is appropriate for comparing predictive errors, additional information about prediction uncertainty would strengthen the evaluation. The authors should also discuss whether the random masking strategy adequately represents missing-data mechanisms encountered in ecological databases. Evaluating different proportions or patterns of missing data would provide additional evidence regarding model robustness.
   > <span style="background-color: #f4cccc">*Risposta/intervento:* We assessed several masking rates and structured patterns, including jointly missing related traits, on identical held-out cells across methods. We evaluated held-out calibrated prediction intervals, reporting coverage and width by output where sample sizes permitted. We clarified that artificial masking evaluates observed values under the chosen simulation and does not establish performance under an unknown missing-not-at-random mechanism.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: nessuno</span>** — Pattern di dati mancanti: diverso dal leakage dei duplicati.</span>
8. Biological interpretation of model explainability: The use of Integrated Gradients provides an interpretable perspective on the relative contribution of environmental predictors. Nevertheless, feature attribution should not be interpreted as direct evidence of causal ecological relationships. The authors should clarify this distinction and discuss the sensitivity of the attribution results to the selected baseline. The treatment of multiple statistical comparisons in the identification of phylogenetic hotspots and coldspots should also be specified.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We clarified that Integrated Gradients described sensitivity of the fitted model relative to a reference input, rather than causal effects. We compared alternative reference baselines and attribution stability, reported the multiple-testing family and correction used for local phylogenetic tests, and qualified the interpretations in light of these checks.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.9, R4.10, R5.6</span>** — Limiti di explainability e test multipli.</span>

### Minor comments

1. Provide further justification for selecting 50 spatial clusters and six spatial neighbors.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We clarified that 50 clusters represented a pragmatic compromise between spatial resolution and the number of occupied cells per location node, and that six neighbours were a fixed design choice. We examined spatial compactness and cluster-size distributions across alternative cluster counts; these diagnostic curves did not establish an optimal configuration or measure predictive sensitivity.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: nessuno</span>**</span>
2. Consider reporting prediction errors in the original measurement units alongside NRMSE-IQR.
   > <span style="background-color: #d9ead3">*Risposta/intervento:* We added per-output MAE and RMSE in the original units, together with the IQR denominators and sample sizes, alongside NRMSE-IQR.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.6</span>** — Metriche d'errore più interpretabili.</span>
3. Clarify whether the predicted minimum and maximum trait values are constrained to remain biologically consistent.
   > <span style="background-color: #d9ead3">*Risposta/intervento:* We now enforce biologically consistent bounds. The model predicts the minimum and maximum with separate output heads and separately fitted transformations. After back-transformation to the original scale, any pair whose predicted minimum exceeds the predicted maximum is replaced by its midpoint. This is the Euclidean projection onto ordered pairs, so it guarantees minimum ≤ maximum and cannot increase the joint error of any pair whose observed bounds are ordered. The same step is applied to the internal-validation predictions used for calibration. Reversals were rare before projection (0.7–1.1% of test pairs in matched five-seed runs), all for fertile and sterile stipe length, with nearly equal predicted bounds, and the projection left the reported correlations unchanged. We described this step in the Methods and in the hyperparameter table, and replaced the earlier statement that bounds were unconstrained.</span>

   > <span style="background-color: #d9ead3">*Postilla:* We also tried, without success, to impose the ordering inside the network. (i) Predicting the minimum plus a non-negative width (ReLU or Softplus) under a shared transformation guaranteed order but reduced accuracy for the traits with strong signal, and was numerically unstable for maximum altitude. (ii) A soft penalty on reversed bounds added to the training loss did not change accuracy and removed only a small share of the reversals. The post-hoc projection was the only option that guaranteed ordering at no measurable cost.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: nessuno</span>** — Coerenza biologica dei minimi e massimi.</span>
4. Distinguish the ecological meaning of altitudinal range from the morphological traits considered in the study.
   > <span style="background-color: #d9ead3">*Risposta/intervento:* Revisione puntuale del testo, delle figure o delle tabelle.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: nessuno</span>** — Significato ecologico della quota.</span>
5. Improve the readability of figure labels, particularly those presenting multiple model comparisons.
   > <span style="background-color: #d9ead3">*Risposta/intervento:* We enlarged and simplified the labels in the ablation comparison figure, clarified its caption and corrected grammatical and typographical issues in the manuscript.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: nessuno</span>** — Leggibilità delle figure.</span>
6. Review the manuscript for minor grammatical and typographical inconsistencies.
   > <span style="background-color: #d9ead3">*Risposta/intervento:* Accordingly, we checked the text for mistakes and inconsistencies</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: nessuno</span>**</span>

## Reviewer 3

1. The trait data are assembled from the Flora of New Zealand, conservation-network records, and published literature, but the harmonization procedure is not described in sufficient detail. Please clarify unit conversions, the criteria used to exclude outliers when defining the minimum and maximum trait ranges, and the number of records supporting each trait.
   > <span style="background-color: #d9ead3">*Risposta/intervento:* We traced the underlying sources and documented unit conversions, outlier handling and the number of records supporting each trait, including unavailable provenance. We avoided inferring cleaning rules that could not be verified.</span>
   >
   > <span style="background-color: #d9ead3">*Seconda bozza:* All trait data is gathered from Brownsey and Perrie’s treatment of the New Zealand’s fern flora. For trait measurements, the work used all available herbaria specimens and outlier values were clearly listed, allowing us to use the ‘average’ min and max values for each species. We amended the text to make this clear.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: nessuno</span>**</span>
2. Only 174 of the 208 taxa are represented in FTOL, while the remaining taxa are assigned equal relatedness at the genus or family level. This approximation may affect the phylogenetic descriptors and the inferred signal. Please report how many taxa were assigned at each level and, if feasible, provide a sensitivity analysis excluding these taxa.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* Following an update to the dataset and FTOL reconstruction, 166 of 189 taxa match original FTOL tips (164 by binomial and two by synonym), 16 are grafted at the most recent common ancestor of a monophyletic genus, and seven remain unresolved. No family-level grafting is applied. The unresolved taxa have zero phylogenetic descriptors and no interspecies phylogenetic edges. Comparisons with baselines were scored on the 182 tree-represented taxa for every method, because tree-dependent baselines (Phylo NN, Phylo KNN, Rphylopars) cannot predict the seven unresolved taxa; the GNN was still trained and evaluated with all 189 taxa in the other analyses. A sensitivity analysis excluding the grafted taxa remains to be performed.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R2.6, R4.3</span>** — Specie assenti dalla filogenesi.</span>
3. Environmental information is represented by 50 spectral clusters, with each location node connected to its six nearest neighbours and occurrence weights adjusted by population density. Please explain the rationale for these choices more clearly and report whether the main findings are sensitive to the number of clusters or the occurrence-weighting scheme.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We clarified the 50-cluster construction, six-neighbour graph and population-density weighting, and tested prespecified alternatives for spatial resolution and weighting under matched folds. We reported coverage and uncertainty as well as accuracy.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R2.minor1, R4.8</span>** — Scelta e sensibilità del clustering spaziale.</span>
4. The ablation results show that adding environmental variables to the phylogeny-only model changes the median NRMSE-IQR by only 0.002, while environmental information alone does not improve the baseline. Please discuss whether this limited contribution reflects weak environmental signal, coarse spatial aggregation, or limitations in the selected environmental descriptors.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* The small aggregate difference did not identify its cause. We reported output-specific paired effects and sensitivity to spatial resolution, weighting and an alternative environmental representation. We discussed limited signal, aggregation and descriptor choice as possible explanations rather than established causes.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.2, R4.8</span>** — Effetto limitato dei dati ambientali.</span>
5. Prediction errors remain relatively high for altitude range and spore size, for which simple mean or median imputation is competitive or better. Please make the trait-specific reliability of the framework more explicit and clarify how users should interpret imputations for these outputs, rather than implying uniform reconstruction performance across all traits.
   > <span style="background-color: #d9ead3">*Risposta/intervento:* We reported original-unit errors, full distributions and baseline-relative performance for each output, highlighting altitude and spore-size results and avoiding a claim of uniformly reliable imputations.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.6, R5.2</span>** — Affidabilità dei trait con errori elevati.</span>
6. SPG-OD, PyS2CF-Mamba, and edip-net may offer insights for the design of this method.
   > <span style="background-color: #d9d2e9">*Risposta/intervento:* We checked the bibliographic details and methodological relevance of SPG-OD, PyS2CF-Mamba and edip-net, and discussed directly relevant comparisons in the revised manuscript.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: nessuno</span>** — Suggerimenti bibliografici specifici.</span>

## Reviewer 4

1. The validation strategy may suffer from information leakage because identical fertile and sterile frond measurements are assigned to monomorphic species, while only one trait is masked during evaluation. This could artificially improve prediction accuracy. The authors should repeat the evaluation by jointly masking duplicated traits.
   > <span style="background-color: #f4cccc">*Risposta/intervento:* We agreed that masking a trait alone could leave an identical counterpart visible. In the standard protocol, we jointly masked fertile and sterile stipe traits and fertile and sterile frond traits throughout training, validation and evaluation, including both bounds and any derived range. Groups were defined for every taxon independently of its held-out values; only originally observed target cells were scored. We retained individual-trait masking as an optional comparison.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R5.1</span>** — Leakage dei duplicati fertile/sterile.</span>
2. The ablation analysis does not adequately isolate phylogenetic and environmental contributions. Family information remains among the categorical predictors, while phylogenetic message passing can exploit neighboring species' morphological traits. Consequently, the conclusion that phylogeny provides more predictive information than environmental variables is not sufficiently justified. More controlled ablation experiments are required.
   > <span style="background-color: #f4cccc">*Risposta/intervento:* Family remained in the categorical input, and species message passing could use neighbouring species' observed traits when phylogenetic edges were active. We separately evaluated family, embeddings and phylogenetic connectivity, and fitted the non-graph baselines (MICE, random forest, XGBoost and MLP) with each combination of environmental and phylogenetic inputs, so that the contribution of each information source could also be assessed without message passing between neighbouring species. Statements about their relative contributions were revised to reflect those controlled comparisons.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R2.3</span>** — Ablation e influenza residua della filogenesi.</span>
3. **The treatment of species absent from the phylogenetic tree is inconsistent. Section 2.1 describes assigning genus- or family-level relatedness, whereas Section 2.3 represents these species as isolated nodes with zero phylogenetic descriptors. The authors should clarify the actual implementation and explain how phylogenetic embeddings and graph connections are handled during training and testing.**
   > <span style="background-color: #d9ead3">*Risposta/intervento:* Following an update to the dataset and FTOL reconstruction, 166 of 189 taxa match original FTOL tips, 16 are grafted at genus level, and seven remain unresolved. The 182 represented taxa receive tree-derived embeddings and distance-thresholded edges. Unresolved taxa retain their other features and spatial connections, with zero phylogenetic descriptors and no interspecies phylogenetic edges. Training uses the training-species graph; evaluation uses the full graph with test targets masked. Comparisons with baselines were scored on the 182 tree-represented taxa for every method, because tree-dependent baselines (Phylo NN, Phylo KNN, Rphylopars) cannot predict the seven unresolved taxa; the GNN was still trained and evaluated with all 189 taxa in the other analyses.</span>
   >

   

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R2.6, R3.2</span>** — Gestione delle specie senza albero.</span>
4. The random species-level cross-validation does not adequately establish generalization to phylogenetically distinct species, since closely related taxa may appear in both training and testing sets. Additional genus- or clade-blocked validation is necessary to assess the robustness and transferability of the proposed framework.
   > <span style="background-color: #f4cccc">*Risposta/intervento:* We added genus-blocked and feasible clade-blocked evaluation, using common folds and masks across methods, and reported the groups, graph visibility and sample sizes. This tested transfer across held-out groups within the dataset, rather than transfer to a new region.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R2.5</span>** — Validazione bloccata per clade o genere.</span>
5. The reported aggregate performance requires clarification. Table 1 gives a median NRMSE-IQR of approximately 1.535 across the 12 PhyEnv-NN outputs, whereas Section 3.2 reports 4.511 for the full model. The authors should reconcile these values and explain the aggregation procedure.
   > <span style="background-color: #d9ead3">*Risposta/intervento:* We reconstructed the aggregation from the seed-level outputs, calculated both mean and median across the 12 outputs, and corrected the text, table and captions consistently. We added per-output and baseline-relative results and clarified the origin of 4.511.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R5.3</span>** — Incongruenza tra media e mediana aggregate.</span>
6. The claimed predictive superiority lacks sufficient statistical support. Several differences between competing methods are very small, and uncertainty estimates or statistical comparisons are not provided. Furthermore, extremely large NRMSE-IQR values for altitude and spore size raise concerns about practical predictive reliability. The authors should report confidence intervals, additional error metrics, and observed-versus-predicted comparisons.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We added paired uncertainty intervals, original-unit MAE and RMSE, and observed-versus-predicted plots, and investigated extreme normalized values. We moderated superiority claims in accordance with the intervals and effect sizes.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R2.4, R5.4</span>** — Intervalli e significatività dei risultati.</span>
7. The baseline comparison does not establish whether graph message passing provides a meaningful advantage over conventional machine-learning methods. The authors should include competitive non-graph models, such as random forests, gradient boosting, or multilayer perceptrons, using equivalent input features and validation procedures.
   > <span style="background-color: #f4cccc">*Risposta/intervento:* We added three tuned non-graph regressors, a random forest, gradient-boosted trees (XGBoost) and a multilayer perceptron, with matched observed traits, taxonomy, phylogenetic descriptors and aggregated environmental information, using identical splits and target masks and scoring all methods on identical cells of the 182 taxa that every method could predict. Their hyperparameters were tuned with species-level cross-validation within the outer-training species. We acknowledged that aggregation approximated the location-graph input rather than reproducing its message passing, and that these models do not use the traits of related species.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R2.2</span>** — Confronto con baseline non-graph.</span>
8. The environmental analysis relies on only 50 spatial clusters, potentially obscuring important local climatic and ecological variation. Without sensitivity analysis of spatial resolution and alternative environmental representations, the conclusion that environmental information contributes relatively little remains insufficiently supported.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We compared prespecified cluster counts and alternative weighting or representations, reporting per-output paired differences and graph coverage. We qualified conclusions about environmental contribution in light of their sensitivity to the chosen resolution.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R3.3, R5.6</span>** — Risoluzione dei cluster ambientali.</span>
9. The phylogenetic hotspot analysis relies on a proportional error metric that can become unstable for small observed trait values. Moreover, Local Moran's I is applied across numerous species and comparisons without clearly reported multiple-testing corrections. The authors should repeat these analyses using robust error measures and appropriate statistical corrections.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We recomputed error changes using absolute error and a defined log-ratio error for positive values, specifying the treatment of zero values. We repeated the local Moran analysis with a stated multiple-testing family and correction, and reassessed hotspot, coldspot and outlier claims.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R5.5, R2.8</span>** — Errori proporzionali e test multipli.</span>
10. The Integrated Gradients analysis does not adequately establish the importance of individual environmental predictors. Correlated variables, the choice of baseline, and the absence of attribution stability analyses may affect the reported importance rankings. Additional validation using permutation-based importance or grouped feature ablations is recommended.

   > <span style="background-color: #fff2cc">*Risposta/intervento:* We compared IG reference baselines and seed-level rankings and evaluated grouped permutation or feature ablation on held-out predictions. Correlated inputs were interpreted as groups where appropriate, and attribution was described as model sensitivity rather than independent evidence of predictive utility.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R2.8, R5.6</span>** — Robustezza di Integrated Gradients.</span>

## Reviewer 5

1. For monomorphic species, fertile and sterile values are identical (Sec. 2.1). In leave-one-trait-out, the duplicate stays visible, so the benchmark mixes a near-trivial regime with a genuine one. Hold out fertile/sterile pairs jointly, or try to report dimorphic species separately.
   > <span style="background-color: #f4cccc">*Risposta/intervento:* We adopted joint occlusion of fertile/sterile frond and stipe pairs throughout training, validation and testing. Both minimum and maximum components, and any derived range, were hidden together for every test species in the target group. The groups were fixed in advance for all taxa, while evaluation used only originally observed cells. Individual-trait occlusion remained available as an optional comparison.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.1</span>** — Leakage dei duplicati e masking congiunto.</span>
2. NRMSE-IQR is about 10 (max altitude) and 16-18 (spore size) even for the training-mean baseline, which implies extreme values, a unit problem or a back-transformation problem. Fig. A.7 is truncated at the 90th percentile and cannot rule out this possibility. Please show full target and error distributions and check the Yeo-Johnson back-transformation.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We showed untruncated target and residual distributions, verified source units and extreme observations, and independently checked fold-fitted Yeo–Johnson transformations and inverses, including extreme predictions and clipping. We corrected identified issues and recomputed affected results before interpretation.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.6, R3.5</span>** — Errori estremi per quota e spore.</span>
3. The reported "median" NRMSE-IQR for the full model (4.511) cannot be the median of the 12 outputs in Table 1 (about 1.5). It is close to their mean (4.52), which three outputs dominate. Can you please clarify and report per-trait or baseline-relative results?
   > <span style="background-color: #d9ead3">*Risposta/intervento:* We traced the aggregation underlying 4.511, verified the order of operations and reported the mean and median of output-level errors, together with per-output and baseline-relative results. We corrected any mislabeled statistic.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.5</span>** — Media e mediana NRMSE-IQR discordanti.</span>
4. Table 1 gives only point values, despite 60 seeds. Gaps such as 18.527 vs 18.534 do not support rankings, and the "no statistically significant" environmental effect lacks a stated test. Report paired seed-level differences with intervals.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We reported paired differences for matched splits with intervals and a stated comparison procedure, accounting for dependence among repeated splits. We revised the unsupported “no statistically significant” wording in accordance with the documented test and avoided interpreting tiny rank differences as superiority.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R2.4, R4.6</span>** — Differenze accoppiate e intervalli.</span>
5. Dividing by observed trait value inflates errors for small species (axes in the maximum-bound frond panels of the supplementary figures reach -40 to -60). The \*Azolla rubra\* "outlier" and the small-trait quantile result may be artefacts. Use log-ratio errors and recheck all calls.
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We re-evaluated quantile and phylogenetic local analyses with absolute errors and log-ratio errors for strictly positive values, documenting zero handling. We rechecked the Azolla outlier and small-trait claims and retained conclusions supported by the sensitivity analysis.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.9</span>** — Errori proporzionali instabili per valori piccoli.</span>
6. Integrated gradients describe model sensitivity, not useful information, especially for inputs that do not improve out-of-sample accuracy. The null environmental result may reflect the 50-cluster resolution, so a sensitivity analysis is needed here.x
   > <span style="background-color: #fff2cc">*Risposta/intervento:* We clarified that IG measured the fitted model's sensitivity to changes from a selected reference and did not establish that an input improved held-out prediction. We compared reference choices and grouped predictive ablations, evaluated spatial-resolution sensitivity, and aligned the interpretation with the resulting out-of-sample evidence.</span>

   <span style="background-color: #fff2cc">**<span style="color: #e69138">Commenti simili: R4.10, R4.8</span>** — Integrated Gradients e risoluzione ambientale.</span>

## Cross-cutting experimental-setting notes

**Historical single-trait masking could leave an identical fertile/sterile counterpart visible. The current default jointly masks each pair; historical results remain associated with the earlier protocol and require reevaluation.**

**The no-phylogeny ablation retains family information, so it does not remove all taxonomic information. The current phylogeny switch also changes both embeddings and species graph connectivity, preventing attribution of the effect to either component separately.**

**The updated reconstruction distinguishes 166 original FTOL tip matches, 16 genus-level grafts and seven unresolved taxa. Unresolved taxa are retained without phylogenetic edges or descriptors. Comparisons with baselines are scored on the 182 tree-represented taxa for every method; GNN training and GNN-only analyses keep all 189.**

**The paper labels 4.511 a median, while the reviewer reports that the median of Table 1 is approximately 1.535. The calculation and aggregation order must be reproduced before interpreting or correcting the discrepancy.**

**Predicted bounds are now ordered by a post-hoc midpoint projection on the original scale, which has no measurable effect on accuracy. Ordered output heads and a soft penalty were tested and rejected; see [ordering of predicted bounds](../methodology/bounds-parameterization.md).**

**Random species splits and full-graph evaluation can retain relationships between train and test taxa. They estimate performance in that transductive setting and do not establish transfer to unseen clades or regions.**

