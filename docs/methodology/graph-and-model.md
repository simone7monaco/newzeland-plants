---
type: Research methodology
title: Heterogeneous graph and model
description: Spatial and phylogenetic graph construction and message passing.
tags: [graph, model, environment, phylogeny]
status: draft
generated: {by: codex/gpt-6}
sources:
  - {resource: "../../Paper/'main.tex", title: Manuscript graph construction and neural network model}
  - {resource: ../../loader.py, title: PlantDataset and graph construction}
  - {resource: ../../models.py, title: TraitsPredictor and GNN}
---

# Heterogeneous graph and model

## Nodes and edges

Species nodes carry available min/max trait inputs, a visibility mask, categorical features including family, and optional 16-dimensional Node2Vec descriptors. The reconstructed FTOL tree, including internal nodes, is embedded using Node2Vec (walk length 30, 200 walks, window 10; seed 42). Tree-tip pairs are connected when their tree distance is at most the median pairwise distance; distance is an edge attribute. Names unresolved in the reconstructed tree become zero-descriptor nodes without phylogenetic edges. Of 189 taxa, 166 match original FTOL tips, 16 are grafted at genus level, and seven remain unresolved.

Location nodes are spectral clusters of occupied 0.1-degree raster cells, with environmental covariates aggregated by cluster. The loader's default is 50 clusters and six nearest neighbours. A species–location edge carries an occurrence weight. Cached climate, soil and population/elevation aggregates and species–cluster weights can be reused by the loader. These caches must be rebuilt when cluster assignments change. The [spatial diagnostic](../findings/reviewer-diagnostics.md) does not itself rebuild model inputs.

## Computation

With environment enabled, the location graph is encoded and transferred to species by bipartite attention; its representation is concatenated with a species encoding and passed through species graph attention. The default network uses two layers, hidden size 150, GATv2Conv and dropout 0.4; the bipartite layer is GATConv. The standard head outputs normalized min and max independently. The current optional [ordered heads](../updates/reviewer-response.md) instead produce a minimum and nonnegative width under a shared transform.

The no-phylogeny switch removes explicit phylogenetic descriptors and replaces interspecies edges with self-loops; categorical family remains. The no-environment switch omits spatial and bipartite pathways. Consequently a configuration switch changes several components at once and does not isolate family, embeddings and message passing. The legacy switch still behaves this way; the separate switches described in [component ablation](../updates/component-ablation.md) isolate each component. See [benchmark protocol](../experiments/benchmark-protocol.md).
