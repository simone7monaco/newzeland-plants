# Agent instructions

PhyEnv-NN: graph neural network imputation of New Zealand fern traits, benchmarked against statistical, phylogenetic and machine-learning baselines. Manuscript in `Paper/'main.tex`; reviewer-response work tracked in `TODO`.

## Keep the OKF documentation in sync (mandatory)

`docs/` is the project's knowledge base in [Open Knowledge Format 0.2](docs/index.md). It must describe what the code currently does.

**After every change to code, data handling, outputs or analyses, check the OKF and update it in the same task, unless the user explicitly says not to.** This applies even when the request does not mention documentation. If nothing needs changing, say so in your final message, naming the pages you checked.

1. Find the affected pages: search `docs/` for the files, flags, outputs and concepts you touched (each page lists its code in `sources:` frontmatter). Typical owners:
   - data, cohort, tree: `methodology/study-and-data.md`, `updates/ftol-reconstruction.md`
   - model, training, masking, bounds: `methodology/*.md`
   - benchmark runner, baselines, outputs, manifest: `experiments/benchmark-protocol.md`, `updates/ml-baselines.md`
   - metrics, notebooks, names and orderings: `experiments/metrics-and-aggregation.md`, `experiments/analysis-conventions.md`
2. Correct every statement the change makes false (file names, defaults, flags, counts, "not implemented"/"not run" claims), and remove the stale text.
3. New concepts get their own page with YAML frontmatter (`type`, `title`, `description`, `tags`, `status`, `generated`, `sources`), linked from the section `index.md` and, if central, from `docs/index.md`.
4. Add a dated entry to `docs/log.md` (Fix / Update / Decision / Validation / Open): what changed, where, how it was validated.
5. Update `README` when commands, options, outputs or requirements change, and annotate the matching `TODO` item.
6. Separate implemented, smoke-tested and actually run. Never report results that were not produced; one-fold or few-epoch runs are functional checks, not findings.
7. Historical artifacts keep their original provenance; never relabel old results as current.

Change the manuscript or reviewer responses (`docs/updates/reviewer-responses.md`) only when asked, or when a change makes a statement there factually wrong. In that case update it and report the change.

## Code conventions

- Names and orderings of traits, bounds, configurations, models and environmental features come from `analysis_utils.py`. Do not redefine them in notebooks or scripts; extend the canonical definitions instead. Inconsistencies must raise (`AnalysisConsistencyError`), not warn.
- Every model exports original-scale predictions (`predictions_{min,max}.csv`, `predictions_long.csv`); GNN normalized outputs are only `*_normalized.csv`. Comparisons with baselines use `analysis_utils.baseline_comparison_cells` (common species; tree-dependent baselines exclude the taxa absent from the tree).
- GNN and baselines must share folds, occlusion groups, evaluated cells and original-scale targets. A change to one side's evaluation must be mirrored on the other, or documented as a deliberate difference.
- Graph loading needs R through rpy2. Run Python in the R environment:

  ```sh
  micromamba run -n r-baselines bash -c 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; exec .venv/bin/python -m unittest discover -s tests'
  ```

  Run the tests after code changes and report failures with their output.
- Do not commit data, results or caches (`data/`, `benchmark_results/`, `results/`).
