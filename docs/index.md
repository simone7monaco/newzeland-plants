---
okf_version: "0.2"
title: PhyEnv-NN research knowledge base
description: Methodology, experiments, results, provenance, known discrepancies and updates for the fern trait imputation project.
---

# PhyEnv-NN research knowledge base

This bundle records what the paper describes, what the repository currently implements, and what the saved results support. It follows [Open Knowledge Format 0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md). Concept pages carry YAML metadata and source links; they have not received independent human verification. The sources remain authoritative, and current code is not proof of which code produced an older artifact.

## Start here

- [Study and data](methodology/study-and-data.md): task, data provenance and coverage.
- [Graph and model](methodology/graph-and-model.md): graph construction, features and architecture.
- [Training and masking](methodology/training-and-masking.md): objective, splits and transductive inference.
- [Experimental protocol](experiments/benchmark-protocol.md): configurations, baselines and execution.
- [Metrics and aggregation](experiments/metrics-and-aggregation.md): evaluation, scales and summary statistics.
- [Interpretation analyses](experiments/explainability-and-phylogeny.md): IG, error transitions and local analyses.
- [Species and occlusion comparison](findings/species-occlusion-one-seed.md): one-seed factorial results and limits.
- [Published benchmark findings](findings/benchmark.md): reported results, sources and limitations.
- [Audit of manuscript and implementation](findings/consistency-audit.md): confirmed discrepancies and unresolved provenance.
- [Spatial diagnostic and ordered bounds](findings/reviewer-diagnostics.md): results from the reviewer-response work.
- [Updates](updates/index.md): recent changes, validation and work still outstanding.
- [Log](log.md): chronological bundle history.

Data and benchmark results under `data/` and `benchmark_results/` are ignored by Git in this checkout; availability and exact versions must be checked when reproducing analyses. The [project TODO](../TODO) remains the working task list.
