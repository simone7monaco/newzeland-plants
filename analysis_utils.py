"""Shared naming, ordering, loading and consistency checks for the analysis notebooks.

Single source of truth for how traits, bounds, input configurations and models
are named and ordered in every notebook, table and figure, and for how
benchmark predictions are loaded, checked and scored.

Every check raises an exception instead of warning: an unknown model directory,
an unexpected label, normalized predictions, mismatched folds or observed
values, or a species set that differs from the expected tree coverage stop the
analysis before a table can be produced.

Typical use in a notebook::

    from analysis_utils import load_benchmark_predictions, baseline_comparison_cells, global_metrics
    predictions = load_benchmark_predictions(results_dir)
    cells = baseline_comparison_cells(predictions, models=...)
    metrics = global_metrics(cells, scales=cohort_scales(predictions))
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Project location
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmark_results"
DEFAULT_TRAITS_FILE = PROJECT_ROOT / "data" / "Ferns" / "FernTraits.xlsx"


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root from a notebook working directory and make it importable."""
    start = Path.cwd() if start is None else Path(start)
    for parent in (start, *start.parents):
        if (parent / "analysis_utils.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise FileNotFoundError(f"Could not locate analysis_utils.py from {start}")


# ---------------------------------------------------------------------------
# Canonical names and orderings
# ---------------------------------------------------------------------------

TRAIT_ORDER = ("Alt", "Ferstipe", "SterileStipe", "FertileFrond", "SterileFrond", "Spore")
TRAIT_LABELS = {
    "Alt": "Altitude range",
    "Ferstipe": "Fertile stipe length",
    "SterileStipe": "Sterile stipe length",
    "FertileFrond": "Fertile frond length",
    "SterileFrond": "Sterile frond length",
    "Spore": "Spore size",
}

BOUND_ORDER = ("min", "max")
BOUND_LABELS = {"min": "Minimum value", "max": "Maximum value"}
BOUND_SHORT_LABELS = {"min": "min", "max": "max"}

# Benchmark directory name -> configuration id (shared by GNN and covariate baselines).
CONFIGURATION_DIRS = {
    "no_env_no_phylo": "baseline",
    "env_only": "environment",
    "phylo_only": "phylogeny",
    "env_phylo": "full",
}
CONFIGURATION_ORDER = ("baseline", "environment", "phylogeny", "full")
CONFIGURATION_LABELS = {
    "baseline": "No additional inputs",
    "environment": "Environment only",
    "phylogeny": "Phylogeny only",
    "full": "Environment + phylogeny",
}
# Short labels used by the R lambda/LIPA/quantile summaries.
CONFIGURATION_SHORT_LABELS = {
    "baseline": "Base",
    "environment": "Env",
    "phylogeny": "Phylo",
    "full": "Phylo_Env",
}
CONFIGURATION_INPUTS = {
    "baseline": (False, False),
    "environment": (True, False),
    "phylogeny": (False, True),
    "full": (True, True),
}


def configuration_from_inputs(use_env: bool, use_phylo: bool) -> str:
    return {inputs: config for config, inputs in CONFIGURATION_INPUTS.items()}[(bool(use_env), bool(use_phylo))]


def contrast_label(reference: str, treatment: str, short: bool = True) -> str:
    labels = CONFIGURATION_SHORT_LABELS if short else CONFIGURATION_LABELS
    return f"{labels[reference]} → {labels[treatment]}"


_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")
# Location-node covariates (PlantDataset.spatial_global_feature_names) and positional encodings.
ENV_FEATURE_LABELS = {
    "wc2.1_2.5m_bio_1_1": "Annual mean temperature",
    "wc2.1_2.5m_bio_12_1": "Annual precipitation",
    "wc2.1_2.5m_bio_7_1": "Temperature annual range",
    "wc2.1_2.5m_bio_2_1": "Mean diurnal temperature range",
    "wc2.1_2.5m_bio_15_1": "Precipitation seasonality",
    **{f"wc2.1_2.5m_srad_{m:02d}_1": f"Solar radiation - {name}" for m, name in enumerate(_MONTHS, 1)},
    **{f"wc2.1_2.5m_vapr_{m:02d}_1": f"Water vapour pressure - {name}" for m, name in enumerate(_MONTHS, 1)},
    "NZ population density layer_1": "Population density",
    "wc2.1_2.5m_elev_1": "Elevation above sea level",
    "calciumraster_1": "Exchangeable soil calcium",
    "drainageraster_1": "Soil drainage capacity",
    "ecoraster_1": "Ecological raster 1",
    "ecoraster_2": "Ecological raster 2",
    "hardnessraster_1": "Soil induration (hardness)",
    "limiation_to_growthraster_1": "Chemical limitations to plant growth",
    "pHraster_1": "Soil pH",
    "particle_sizeraster_1": "Soil particle size",
    "phosphorousraster_1": "Acid-soluble soil phosphorus",
    "soil_ageraster_1": "Soil age",
    **{f"pos_{i}": f"Position encoding {i}" for i in range(4)},
}
ENV_DOMAIN_ORDER = ("Climate", "Soil", "Elevation", "Density", "Position")
CATEGORICAL_PREFIX_LABELS = {"Family_": "Family", "Habitat_": "Habitat"}


def _env_key(name: str) -> str:
    name = str(name)
    if name in ENV_FEATURE_LABELS:
        return name
    if f"{name}_1" in ENV_FEATURE_LABELS:  # raw raster names without the band suffix
        return f"{name}_1"
    raise AnalysisConsistencyError(f"Unknown environmental feature: {name!r}")


def env_feature_label(name: str) -> str:
    """Publication label of a location covariate; unknown names raise."""
    return ENV_FEATURE_LABELS[_env_key(name)]


def environment_domain(name: str) -> str:
    """Climate, Soil, Elevation, Density or Position for a location covariate."""
    key = _env_key(name)
    if key.startswith("pos_"):
        return "Position"
    if key.startswith("wc2.1_2.5m_elev"):
        return "Elevation"
    if key.startswith("wc2.1_2.5m_"):
        return "Climate"
    if key.startswith("NZ population density"):
        return "Density"
    return "Soil"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    uses_tree: bool = False
    uses_covariates: bool = False


MODEL_SPECS = {spec.key: spec for spec in (
    ModelSpec("training_mean", "Training mean", "reference"),
    ModelSpec("training_median", "Training median", "reference"),
    ModelSpec("phylo_nn", "Phylo NN", "phylogenetic", uses_tree=True),
    ModelSpec("phylo_knn", "Phylo KNN", "phylogenetic", uses_tree=True),
    ModelSpec("rphylopars_bm", "Rphylopars (BM)", "phylogenetic", uses_tree=True),
    ModelSpec("mice", "MICE", "imputation", uses_covariates=True),
    ModelSpec("missforest", "MissForest", "imputation", uses_covariates=True),
    ModelSpec("random_forest", "Random forest", "machine learning", uses_covariates=True),
    ModelSpec("xgboost", "XGBoost", "machine learning", uses_covariates=True),
    ModelSpec("mlp", "MLP", "machine learning", uses_covariates=True),
    ModelSpec("gnn", "PhyEnv-NN", "graph"),
)}
MODEL_ORDER = tuple(MODEL_SPECS)
TREE_ONLY_MODELS = tuple(key for key, spec in MODEL_SPECS.items() if spec.uses_tree)

# FTOL placements for which no tip was added to the reconstructed tree (ftol_tree.R).
UNRESOLVED_PLACEMENTS = (
    "genus absent", "genus non-monophyletic", "shared FTOL tip unresolved",
    "invalid terminal length", "ambiguous", "unmatched",
)


class AnalysisConsistencyError(ValueError):
    """Raised when benchmark outputs violate an invariant required by the analyses."""


@dataclass(frozen=True)
class ModelRun:
    """Parsed identity of one saved model directory."""
    model: str
    configuration_id: str
    training_protocol: int | None = None
    occlusion: str | None = None
    include_unobserved: bool = False
    categorical: bool = True

    @property
    def use_env(self) -> bool:
        return CONFIGURATION_INPUTS[self.configuration_id][0]

    @property
    def use_phylo(self) -> bool:
        return CONFIGURATION_INPUTS[self.configuration_id][1]

    @property
    def label(self) -> str:
        return model_label(self.model, self.configuration_id)


_BASELINE_DIR = re.compile(
    # The occlusion suffix is absent only in historical (pre-paired-protocol) runs.
    r"^(?P<model>[A-Z_]+?)_prot(?P<protocol>\d+)(?:_occlusion-(?P<occlusion>paired|individual))?"
    r"(?P<unobserved>_include-unobserved)?(?P<env>_env)?(?P<phylo>_phylo)?(?P<nocat>_nocat)?$"
)


def parse_baseline_dir(name: str) -> ModelRun:
    """Parse a directory written by r_baselines_fit.model_dir_suffix; unknown names raise."""
    match = _BASELINE_DIR.fullmatch(name)
    if match is None:
        raise AnalysisConsistencyError(f"Unrecognized baseline directory name: {name!r}")
    model = match["model"].lower()
    spec = MODEL_SPECS.get(model)
    if spec is None or model == "gnn":
        raise AnalysisConsistencyError(f"Unknown baseline model {model!r} in directory {name!r}")
    use_env, use_phylo = bool(match["env"]), bool(match["phylo"])
    if (use_env or use_phylo) and not spec.uses_covariates:
        raise AnalysisConsistencyError(f"{spec.label} cannot use environmental/phylogenetic inputs: {name!r}")
    if match["nocat"] and spec.family != "machine learning":
        raise AnalysisConsistencyError(f"Only ML baselines have a no-categorical variant: {name!r}")
    return ModelRun(
        model=model,
        configuration_id=configuration_from_inputs(use_env, use_phylo),
        training_protocol=int(match["protocol"]),
        occlusion=match["occlusion"],
        include_unobserved=bool(match["unobserved"]),
        categorical=not match["nocat"],
    )


def parse_gnn_dir(combination: str) -> ModelRun:
    if combination not in CONFIGURATION_DIRS:
        raise AnalysisConsistencyError(f"Unknown GNN configuration directory: {combination!r}")
    return ModelRun(model="gnn", configuration_id=CONFIGURATION_DIRS[combination])


def model_label(model: str, configuration_id: str | None = None, with_inputs: bool | None = None) -> str:
    """Display label; input configuration is appended for models that can vary it."""
    spec = MODEL_SPECS.get(model)
    if spec is None:
        raise AnalysisConsistencyError(f"Unknown model key: {model!r}")
    if with_inputs is None:
        with_inputs = spec.uses_covariates or model == "gnn"
    if not with_inputs or configuration_id is None:
        return spec.label
    return f"{spec.label} ({CONFIGURATION_LABELS[configuration_id].lower()})"


def output_label(trait: str, variable: str) -> str:
    return f"{TRAIT_LABELS[trait]} · {BOUND_SHORT_LABELS[variable]}"


def require_known(values, allowed, what: str) -> None:
    """Raise if a column contains labels outside the canonical set."""
    unknown = sorted(set(pd.Series(values).dropna().astype(str)) - set(map(str, allowed)))
    if unknown:
        raise AnalysisConsistencyError(f"Unknown {what}: {unknown}; allowed: {list(allowed)}")


def order_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert known identity columns to ordered categoricals (and validate them)."""
    out = frame.copy()
    for column, order in (("trait", TRAIT_ORDER), ("variable", BOUND_ORDER),
                          ("configuration_id", CONFIGURATION_ORDER), ("model", MODEL_ORDER)):
        if column in out:
            require_known(out[column], order, column)
            out[column] = pd.Categorical(out[column].astype(str), categories=list(order), ordered=True)
    return out


def ordered_output_columns(columns: pd.MultiIndex | None = None) -> pd.MultiIndex:
    """(trait, variable) columns in canonical order; validates an existing column set."""
    canonical = pd.MultiIndex.from_product([TRAIT_ORDER, BOUND_ORDER], names=["trait", "variable"])
    if columns is None:
        return canonical
    unknown = set(columns) - set(canonical)
    if unknown:
        raise AnalysisConsistencyError(f"Unknown output columns: {sorted(unknown)}")
    return pd.MultiIndex.from_tuples([c for c in canonical if c in set(columns)], names=columns.names)


# ---------------------------------------------------------------------------
# Loading original-scale predictions
# ---------------------------------------------------------------------------

LONG_FILE = "predictions_long.csv"
TREE_COVERAGE_FILE = "ftol_mapping.csv"
N_FOLDS = 5


def _read_long(fold_dir: Path) -> pd.DataFrame:
    path = fold_dir / LONG_FILE
    if not path.is_file():
        raise AnalysisConsistencyError(
            f"{path} is missing. Results produced before original-scale long exports cannot be "
            "used; rerun the benchmark with the current code."
        )
    return pd.read_csv(path)


def _read_run(model_dir: Path, run: ModelRun, seed: int) -> pd.DataFrame:
    fold_dirs = sorted(path for path in model_dir.glob("fold_*") if path.is_dir())
    if [d.name for d in fold_dirs] != [f"fold_{k}" for k in range(N_FOLDS)]:
        raise AnalysisConsistencyError(f"{model_dir} has folds {[d.name for d in fold_dirs]}, expected {N_FOLDS}.")
    frame = pd.concat([_read_long(d) for d in fold_dirs], ignore_index=True)
    frame.insert(0, "seed", seed)
    frame.insert(1, "model", run.model)
    frame.insert(2, "configuration_id", run.configuration_id)
    frame.insert(3, "model_label", run.label)
    frame.insert(4, "run_dir", model_dir.name)
    return frame


def load_benchmark_predictions(results_dir: Path = DEFAULT_RESULTS_DIR, *, include_gnn: bool = True,
                               include_baselines: bool = True, allow_nonfinite: bool = False) -> pd.DataFrame:
    """Load every model's original-scale long predictions from a run_benchmark.py output directory.

    Returns one row per seed x model run x species x trait x bound and runs
    :func:`check_predictions` before returning.
    """
    runs_dir = Path(results_dir) / "runs"
    if not runs_dir.is_dir():
        raise AnalysisConsistencyError(f"{runs_dir} does not exist; expected run_benchmark.py output.")
    frames = []
    for seed_dir in sorted(runs_dir.glob("seed_*")):
        match = re.fullmatch(r"seed_(\d+)", seed_dir.name)
        if match is None:
            raise AnalysisConsistencyError(f"Unexpected directory in runs/: {seed_dir.name}")
        seed = int(match.group(1))
        if include_baselines:
            for model_dir in sorted(p for p in (seed_dir / "baselines").iterdir() if p.is_dir()):
                frames.append(_read_run(model_dir, parse_baseline_dir(model_dir.name), seed))
        if include_gnn:
            for combination_dir in sorted(p for p in (seed_dir / "gnn").iterdir() if p.is_dir()):
                run = parse_gnn_dir(combination_dir.name)
                experiment_dirs = [p for p in combination_dir.iterdir() if p.is_dir()]
                if len(experiment_dirs) != 1:
                    raise AnalysisConsistencyError(f"Expected one GNN experiment below {combination_dir}, found {experiment_dirs}")
                frames.append(_read_run(experiment_dirs[0], run, seed))
    if not frames:
        raise AnalysisConsistencyError(f"No predictions found below {runs_dir}")
    predictions = pd.concat(frames, ignore_index=True)
    check_predictions(predictions, allow_nonfinite=allow_nonfinite)
    return predictions


def check_predictions(predictions: pd.DataFrame, *, allow_nonfinite: bool = False, rtol: float = 1e-6) -> None:
    """Invariants shared by all model comparisons. Raises AnalysisConsistencyError."""
    required = {"seed", "model", "configuration_id", "run_dir", "species", "fold", "trait",
                "variable", "observed", "predicted", "evaluated", "scale"}
    missing = required - set(predictions.columns)
    if missing:
        raise AnalysisConsistencyError(f"Prediction table lacks columns {sorted(missing)}")
    scales = set(predictions["scale"].unique())
    if scales != {"original"}:
        raise AnalysisConsistencyError(f"Predictions must all be on the original scale, found {sorted(scales)}")
    require_known(predictions["trait"], TRAIT_ORDER, "trait")
    require_known(predictions["variable"], BOUND_ORDER, "bound")
    require_known(predictions["model"], MODEL_ORDER, "model")
    require_known(predictions["configuration_id"], CONFIGURATION_ORDER, "configuration")

    key = ["seed", "run_dir", "model", "configuration_id", "species", "trait", "variable"]
    duplicated = predictions.duplicated(key)
    if duplicated.any():
        raise AnalysisConsistencyError(f"Duplicated prediction rows, e.g. {predictions.loc[duplicated, key].head(3).to_dict('records')}")
    labels = predictions.groupby(["seed", "model", "configuration_id"])["run_dir"].nunique()
    if (labels > 1).any():
        raise AnalysisConsistencyError(f"Several directories map to one model/configuration: {labels[labels > 1].index.tolist()}")

    evaluated = predictions["evaluated"].astype(bool)
    if (evaluated & predictions["observed"].isna()).any():
        raise AnalysisConsistencyError("Some evaluated cells have no observed value.")
    bad = evaluated & ~np.isfinite(predictions["predicted"].to_numpy(dtype=float))
    if bad.any() and not allow_nonfinite:
        examples = predictions.loc[bad, ["seed", "model", "configuration_id", "species", "trait", "variable", "predicted"]]
        raise AnalysisConsistencyError(
            f"{int(bad.sum())} evaluated cells have non-finite predictions (e.g. an overflowing inverse transform):\n"
            f"{examples.head(10).to_string(index=False)}"
        )
    # A model may skip observed cells only when it cannot use the species at all (tree-only methods).
    skipped = predictions.loc[predictions["observed"].notna() & ~evaluated]
    unexpected = skipped.loc[~skipped["model"].isin(TREE_ONLY_MODELS)]
    if not unexpected.empty:
        raise AnalysisConsistencyError(
            f"Non-tree models did not score observed cells:\n{unexpected[['seed', 'model', 'species', 'trait', 'variable']].head(10).to_string(index=False)}"
        )

    for seed, group in predictions.groupby("seed"):
        # Folds: same species -> fold assignment for every model run of a seed.
        folds = group.drop_duplicates(["run_dir", "species"]).pivot(index="species", columns="run_dir", values="fold")
        if folds.isna().any().any():
            raise AnalysisConsistencyError(f"Seed {seed}: model runs cover different species sets.")
        if (folds.nunique(axis=1) > 1).any():
            raise AnalysisConsistencyError(f"Seed {seed}: models use different species folds: {folds.index[folds.nunique(axis=1) > 1].tolist()[:5]}")
        # Observed values: identical recorded targets for every model.
        observed = group.pivot_table(index=["species", "trait", "variable"], columns="run_dir", values="observed", aggfunc="first", dropna=False)
        reference = observed.iloc[:, 0].to_numpy(dtype=float)
        for column in observed.columns[1:]:
            values = observed[column].to_numpy(dtype=float)
            same = (np.isnan(reference) & np.isnan(values)) | np.isclose(reference, values, rtol=rtol, atol=0.0)
            if not same.all():
                raise AnalysisConsistencyError(f"Seed {seed}: observed values differ between {observed.columns[0]} and {column}.")


# ---------------------------------------------------------------------------
# Tree coverage and common-species comparisons
# ---------------------------------------------------------------------------

def load_tree_coverage(results_dir: Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """FTOL placement and tree membership saved by the benchmark; identical across all runs."""
    paths = sorted(Path(results_dir).glob(f"runs/seed_*/baselines/{TREE_COVERAGE_FILE}"))
    paths += sorted(Path(results_dir).glob(f"runs/seed_*/gnn/*/*/{TREE_COVERAGE_FILE}"))
    if not paths:
        raise AnalysisConsistencyError(f"No {TREE_COVERAGE_FILE} found below {results_dir}/runs; rerun with the current code.")
    reference = pd.read_csv(paths[0]).sort_values("species").reset_index(drop=True)
    for path in paths[1:]:
        other = pd.read_csv(path).sort_values("species").reset_index(drop=True)
        if not reference[["species", "placement", "in_tree"]].equals(other[["species", "placement", "in_tree"]]):
            raise AnalysisConsistencyError(f"Tree coverage differs between {paths[0]} and {path}.")
    unresolved = reference["placement"].isin(UNRESOLVED_PLACEMENTS)
    if (unresolved == reference["in_tree"]).any():
        raise AnalysisConsistencyError("FTOL placements and tree membership disagree in the saved coverage table.")
    return reference


def common_species(predictions: pd.DataFrame) -> dict[int, set[str]]:
    """Per seed, the species with every observed cell scored by every model run present."""
    out = {}
    for seed, group in predictions.groupby("seed"):
        observed = group.loc[group["observed"].notna()]
        complete = observed.groupby(["run_dir", "species"])["evaluated"].all().unstack("run_dir")
        out[seed] = set(complete.index[complete.fillna(False).all(axis=1)])
    return out


def baseline_comparison_cells(predictions: pd.DataFrame, coverage: pd.DataFrame | None = None,
                              runs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Restrict a model comparison to the evaluated cells of species common to all compared models.

    Species that a tree-dependent method cannot predict are removed for *every*
    model. When ``coverage`` is given, the removed species must be exactly the
    taxa absent from the reconstructed tree (if any tree-only method is in the
    comparison) or none at all; anything else raises. The GNN's training and
    other analyses are unaffected: this only selects the scored cells.
    """
    if runs is not None:
        predictions = predictions.merge(runs, on=list(runs.columns), how="inner")
    if predictions.empty:
        raise AnalysisConsistencyError("No predictions left for the requested model comparison.")
    all_species = set(predictions["species"])
    shared = common_species(predictions)
    if len({frozenset(species) for species in shared.values()}) != 1:
        raise AnalysisConsistencyError("The common species set differs between seeds.")
    keep = next(iter(shared.values()))
    excluded = all_species - keep
    if coverage is not None:
        absent = set(coverage.loc[~coverage["in_tree"].astype(bool), "species"])
        expected = absent & all_species if predictions["model"].isin(TREE_ONLY_MODELS).any() else set()
        if excluded != expected:
            raise AnalysisConsistencyError(
                f"Common-species filter removed {sorted(excluded)}, but the expected exclusions "
                f"(taxa absent from the tree) are {sorted(expected)}."
            )
    cells = predictions.loc[predictions["species"].isin(keep) & predictions["evaluated"].astype(bool)].copy()
    per_run = cells.groupby(["seed", "run_dir"]).size().unstack("run_dir")
    if (per_run.nunique(axis=1) > 1).any():
        raise AnalysisConsistencyError("Models are scored on different numbers of cells after the common-species filter.")
    cells.attrs["excluded_species"] = sorted(excluded)
    cells.attrs["n_species"] = len(keep)
    return cells


# ---------------------------------------------------------------------------
# Global (fold-pooled) metrics
# ---------------------------------------------------------------------------

def cohort_scales(predictions: pd.DataFrame) -> pd.DataFrame:
    """IQR and range of each output over all observed analytical cells (one value per species).

    Computed from the recorded targets of the full cohort, so the denominator is
    fixed per output and identical for every model, seed and species subset.
    """
    observed = (predictions.loc[predictions["observed"].notna(), ["species", "trait", "variable", "observed"]]
                .drop_duplicates(["species", "trait", "variable"]))
    grouped = observed.groupby(["trait", "variable"])["observed"]
    scales = grouped.agg(
        observed_n="size",
        trait_iqr=lambda v: float(v.quantile(0.75) - v.quantile(0.25)),
        trait_range=lambda v: float(v.max() - v.min()),
    ).reset_index()
    return scales


def _metric_record(group: pd.DataFrame) -> dict[str, float]:
    p = group["predicted"].to_numpy(dtype=float)
    t = group["observed"].to_numpy(dtype=float)
    err = p - t
    n = len(err)
    pearson = float(np.corrcoef(p, t)[0, 1]) if n > 2 and np.std(p) > 1e-12 and np.std(t) > 1e-12 else np.nan
    spearman = float(stats.spearmanr(p, t).statistic) if n > 2 else np.nan
    return {
        "n": n,
        "RMSE": float(np.sqrt(np.mean(err ** 2))) if n else np.nan,
        "MAE": float(np.mean(np.abs(err))) if n else np.nan,
        "MedAE": float(np.median(np.abs(err))) if n else np.nan,
        "Pearson_r": pearson,
        "Spearman_rho": spearman,
    }


def global_metrics(cells: pd.DataFrame, scales: pd.DataFrame,
                   by=("seed", "model", "configuration_id", "model_label")) -> pd.DataFrame:
    """Metrics per seed x model run x output, pooling the evaluated cells of all folds."""
    if not cells["evaluated"].astype(bool).all():
        raise AnalysisConsistencyError("global_metrics expects only evaluated cells.")
    keys = [*by, "trait", "variable"]
    records = [{**dict(zip(keys, key)), **_metric_record(group)} for key, group in cells.groupby(keys, observed=True)]
    metrics = pd.DataFrame(records).merge(scales, on=["trait", "variable"], how="left", validate="many_to_one")
    if metrics["trait_iqr"].isna().any():
        raise AnalysisConsistencyError("Missing IQR denominators for some outputs.")
    iqr = metrics["trait_iqr"].where(metrics["trait_iqr"] > 0)
    metrics["NRMSE_IQR"] = metrics["RMSE"] / iqr
    metrics["NMAE_IQR"] = metrics["MAE"] / iqr
    metrics["NRMSE_range"] = metrics["RMSE"] / metrics["trait_range"].where(metrics["trait_range"] > 0)
    return order_frame(metrics)


def summarize_seeds(metrics: pd.DataFrame, by=("model", "configuration_id", "model_label"),
                    columns=("NRMSE_IQR", "RMSE", "MAE", "MedAE", "Pearson_r", "Spearman_rho")) -> pd.DataFrame:
    """Mean and SD across seeds for each model run and output; SD is not a prediction interval."""
    keys = [*by, "trait", "variable"]
    grouped = metrics.groupby(keys, observed=True)
    summary = grouped[list(columns)].agg(["mean", "std"])
    summary.columns = [f"{c}" if s == "mean" else f"{c}_std" for c, s in summary.columns]
    summary["n_seeds"] = grouped["seed"].nunique()
    summary["n"] = grouped["n"].first()
    if (grouped["n"].nunique() > 1).any():
        raise AnalysisConsistencyError("The number of scored cells changes across seeds for a model/output.")
    summary = summary.reset_index().merge(
        metrics[["trait", "variable", "trait_iqr", "observed_n"]].drop_duplicates(), on=["trait", "variable"])
    return order_frame(summary)


def comparison_table(summary: pd.DataFrame, value: str = "NRMSE_IQR") -> pd.DataFrame:
    """Models x (trait, bound) table in canonical row and column order."""
    frame = order_frame(summary).sort_values(["model", "configuration_id"])
    rows = list(dict.fromkeys(frame["model_label"].astype(str)))
    if frame.duplicated(["model_label", "trait", "variable"]).any():
        raise AnalysisConsistencyError("Several summary rows share a model label and output.")
    flat = frame.assign(model_label=frame["model_label"].astype(str),
                        trait=frame["trait"].astype(str), variable=frame["variable"].astype(str))
    table = flat.pivot(index="model_label", columns=["trait", "variable"], values=value)
    # Every output present in the summary keeps its column, even if all values are NaN.
    outputs = pd.MultiIndex.from_frame(flat[["trait", "variable"]].drop_duplicates())
    return table.reindex(index=rows, columns=ordered_output_columns(outputs))


def label_output_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Replace (trait, bound) keys with publication labels."""
    out = table.copy()
    out.columns = pd.MultiIndex.from_tuples(
        [(TRAIT_LABELS[t], BOUND_SHORT_LABELS[v]) for t, v in out.columns], names=["Trait", "Bound"])
    return out
