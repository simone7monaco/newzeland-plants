"""Shared benchmark metric loading and aggregation used by the notebooks and extractor.

The Streamlit interface is archived under ``.soft_del/``; these scientific data
transformations remain part of the active analysis workflow.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "benchmark_results"
    if (PROJECT_ROOT / "benchmark_results").exists()
    else PROJECT_ROOT / "results"
)

DEFAULT_TRAITS_FILE = PROJECT_ROOT / "data" / "Ferns" / "FernTraits.xlsx"

from analysis_utils import (
    CONFIGURATION_DIRS as BENCHMARK_CONFIG_MAP,
    CONFIGURATION_LABELS as CONFIG_LABELS,
    CONFIGURATION_ORDER,
    AnalysisConsistencyError,
    model_label,
    parse_baseline_dir,
)

# Canonical names live in analysis_utils.py; these aliases keep older callers working.
CONFIG_ORDER = {configuration: index for index, configuration in enumerate(CONFIGURATION_ORDER)}

SOURCE_LABELS = {
    "gnn": "Graph neural network (GNN)",
    "deterministic_baseline": "Deterministic baseline",
    "r_baseline": "Stochastic baseline",
}

EXCLUDED_ATTRIBUTION_COLUMNS = {
    "species", "target_trait", "variable", "configuration_id", "configuration",
    "experiment_dir", "attribution_protocol", "model", "source", "seed",
    "use_env_features", "use_phylo_features", "has_environment", "has_phylogeny",
}

IDENTITY_ATTRIBUTION_COLUMNS = (
    "species", "target_trait", "variable", "configuration_id", "configuration",
    "experiment_dir", "attribution_protocol",
)

ATTRIBUTION_CACHE_SCHEMA_VERSION = 3

FEATURE_CACHE_VERSION = 2

METRIC_OPTIONS = {
    "RMSE / IQR (robust)": ("NRMSE_IQR", True),
    "RMSE / range": ("NRMSE_range", True),
    "RMSE": ("RMSE", True),
    "MAE": ("MAE", True),
    "Pearson r": ("Pearson_r", False),
    "Spearman rho": ("Spearman_rho", False),
    "AIC (accuracy vs complexity)": ("AIC", True),
}

NO_SEED = -1.0

BASELINE_KNN_NEIGHBORS: dict[str, int] = {
    "phylo_nn": 1,
    "phylo_knn": 5,
}

BASELINE_PARAMETER_COUNTS: dict[str, float] = {
    "training_mean": 1.0,  # one fitted constant per trait-variable (the training mean)
    "training_median": 1.0,  # one fitted constant per trait-variable (the training median)
    "rphylopars_bm": 2.0,  # ML Brownian-motion fit: evolutionary rate (sigma^2) + root state, per trait
    "rphylopars_lambda": 3.0,  # BM + Pagel's lambda (not currently run; kept for forward compatibility)
    "rphylopars_kappa": 3.0,  # BM + Pagel's kappa (not currently run; kept for forward compatibility)
    "mice": float("nan"),  # CART ensemble: no fixed parametric parameter count, excluded from AIC
}

def configuration_metadata(name: str) -> dict[str, object]:
    if name in BENCHMARK_CONFIG_MAP:
        configuration_id = BENCHMARK_CONFIG_MAP[name]
        has_environment = configuration_id in ("environment", "full")
        has_phylogeny = configuration_id in ("phylogeny", "full")
    else:
        upper = name.upper()
        has_environment = bool(re.search(r"_ENV(?:_|$)", upper))
        has_phylogeny = bool(re.search(r"_PHYLO(?:_|$)", upper))
        if has_environment and has_phylogeny:
            configuration_id = "full"
        elif has_environment:
            configuration_id = "environment"
        elif has_phylogeny:
            configuration_id = "phylogeny"
        else:
            configuration_id = "baseline"
    return {
        "configuration_id": configuration_id,
        "configuration": CONFIG_LABELS[configuration_id],
        "has_environment": has_environment,
        "has_phylogeny": has_phylogeny,
    }

def model_display_name(model: str) -> str:
    """Canonical display label; unknown baseline directory names raise."""
    if model == "gnn":
        return model_label("gnn", with_inputs=False)
    run = parse_baseline_dir(model)
    return run.label


def baseline_parameter_count(model: str, observed_n: float) -> float:
    base = parse_baseline_dir(model).model
    knn_neighbors = BASELINE_KNN_NEIGHBORS.get(base)
    if knn_neighbors is not None:
        return float(observed_n) / knn_neighbors if pd.notna(observed_n) and observed_n > 0 else float("nan")
    return BASELINE_PARAMETER_COUNTS.get(base, float("nan"))

def _count_checkpoint_parameters(checkpoint_path: Path) -> float:
    if torch is None:
        return float("nan")
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return float("nan")
    return float(sum(tensor.numel() for tensor in state_dict.values() if hasattr(tensor, "numel")))

def discover_gnn_parameter_counts(results_dir: Path) -> dict[str, float]:
    """Count actual GNN weights per configuration; input size (env/phylo) changes the architecture."""
    counts: dict[str, float] = {}
    if is_benchmark_layout(results_dir):
        runs_dir = results_dir / "runs"
        for combination_name, configuration_id in BENCHMARK_CONFIG_MAP.items():
            checkpoint = next(runs_dir.glob(f"seed_*/gnn/{combination_name}/*/best_model_0.pth"), None)
            if checkpoint is not None:
                counts[configuration_id] = _count_checkpoint_parameters(checkpoint)
    else:
        for experiment_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
            checkpoint = experiment_dir / "best_model_0.pth"
            if checkpoint.exists():
                configuration_id = configuration_metadata(experiment_dir.name)["configuration_id"]
                counts[str(configuration_id)] = _count_checkpoint_parameters(checkpoint)
    return counts

def sort_configurations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "configuration_id" not in frame:
        return frame
    output = frame.copy()
    output["_configuration_order"] = output["configuration_id"].map(CONFIG_ORDER).fillna(len(CONFIG_ORDER))
    return output.sort_values("_configuration_order").drop(columns="_configuration_order")

def trait_scales(traits_file: Path) -> pd.DataFrame:
    if not traits_file.exists():
        return pd.DataFrame(columns=["trait", "variable", "observed_n", "trait_range", "trait_iqr"])

    traits = pd.read_excel(traits_file)
    records: list[dict[str, object]] = []
    for column in traits.columns:
        match = re.fullmatch(r"(.+)(Min|Max|Range)", str(column))
        if match is None:
            continue
        values = pd.to_numeric(traits[column], errors="coerce").dropna()
        if values.empty:
            continue
        records.append(
            {
                "trait": match.group(1),
                "variable": match.group(2).lower(),
                "observed_n": int(values.size),
                "trait_range": float(values.max() - values.min()),
                "trait_iqr": float(values.quantile(0.75) - values.quantile(0.25)),
            }
        )
    return pd.DataFrame(records)

def infer_traits_file(results_dir: Path) -> Path:
    """Select a workbook only when a complete saved prediction species set matches it."""
    candidates = [PROJECT_ROOT / 'data/Ferns/FernTraits.xlsx', PROJECT_ROOT / 'data/Ferns/FernMinMax_all.xlsx', PROJECT_ROOT / 'data/.soft_del/FernMinMax.xlsx']
    predictions = sorted(results_dir.glob('runs/seed_*/gnn/*/*/predictions_original_all_folds.csv'))
    for file in predictions[:4]:
        frame = pd.read_csv(file, usecols=['species'])
        species = set(frame['species'].dropna())
        matches = [p for p in candidates if p.exists() and set(pd.read_excel(p, usecols=['Species'])['Species'].dropna()) == species]
        if len(matches) == 1:
            return matches[0]
    raise ValueError('Cannot identify the trait workbook from saved complete species predictions; specify it explicitly.')

def is_benchmark_layout(results_dir: Path) -> bool:
    return (results_dir / "runs").is_dir()

def _read_measured_parameter_count(fold_dir: Path) -> float:
    """Read a fold's actual fitted complexity (e.g. MICE.estimate_parameter_count), if saved."""
    complexity_file = fold_dir / "model_complexity.csv"
    if not complexity_file.exists():
        return float("nan")
    try:
        return float(pd.read_csv(complexity_file)["k_parameters"].iloc[0])
    except (KeyError, IndexError, ValueError, OSError):
        return float("nan")

def _read_metric_files(model_dir: Path, pattern: str, model: str, source: str, seed: float) -> list[pd.DataFrame]:
    rows = []
    for metric_file in sorted(model_dir.glob(pattern)):
        frame = pd.read_csv(metric_file)
        expected = {"trait", "variable", "n", "RMSE", "MAE", "Pearson_r", "Spearman_rho"}
        if not expected.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["fold"] = metric_file.parent.name
        frame["model"] = model
        frame["source"] = source
        frame["seed"] = seed
        frame["measured_k_parameters"] = _read_measured_parameter_count(metric_file.parent)
        rows.append(frame)
    return rows

def discover_benchmark_metric_rows(results_dir: Path) -> pd.DataFrame:
    """Walk run_benchmark.py's runs/ tree: deterministic + stochastic baselines and GNN, per seed."""
    runs_dir = results_dir / "runs"
    gnn_rows: list[pd.DataFrame] = []
    baseline_rows: list[pd.DataFrame] = []

    deterministic_dir = runs_dir / "deterministic_baselines"
    if deterministic_dir.is_dir():
        for model_dir in sorted(path for path in deterministic_dir.iterdir() if path.is_dir()):
            baseline_rows.extend(
                _read_metric_files(model_dir, "fold_*/per_trait_metrics.csv", model_dir.name, "deterministic_baseline", NO_SEED)
            )

    for seed_dir in sorted(runs_dir.glob("seed_*")):
        try:
            seed = float(seed_dir.name.split("_", 1)[1])
        except ValueError:
            continue

        stochastic_dir = seed_dir / "baselines"
        if stochastic_dir.is_dir():
            for model_dir in sorted(path for path in stochastic_dir.iterdir() if path.is_dir()):
                baseline_rows.extend(
                    _read_metric_files(model_dir, "fold_*/per_trait_metrics.csv", model_dir.name, "r_baseline", seed)
                )

        gnn_dir = seed_dir / "gnn"
        if not gnn_dir.is_dir():
            continue
        for combination_dir in sorted(path for path in gnn_dir.iterdir() if path.is_dir()):
            metadata = configuration_metadata(combination_dir.name)
            for experiment_dir in sorted(path for path in combination_dir.iterdir() if path.is_dir()):
                for frame in _read_metric_files(
                    experiment_dir, "fold_*/per_trait_metrics_original.csv", "gnn", "gnn", seed
                ):
                    for key, value in metadata.items():
                        frame[key] = value
                    frame["experiment_dir"] = combination_dir.name
                    gnn_rows.append(frame)

    frames: list[pd.DataFrame] = list(gnn_rows)
    if baseline_rows:
        baseline_frame = pd.concat(baseline_rows, ignore_index=True)
        # Covariate baselines belong only to the inputs they were actually fit
        # with. Trait/tree-only reference methods can be shared across comparisons.
        covariate_model = baseline_frame['model'].str.lower().str.match(
            r'^(mice|missforest|random_forest|xgboost|mlp)(?:_|$)'
        )
        for model, group in baseline_frame.loc[covariate_model].groupby('model', sort=False):
            metadata = configuration_metadata(model)
            group = group.copy()
            for key, value in metadata.items():
                group[key] = value
            group['experiment_dir'] = next(
                name for name, config in BENCHMARK_CONFIG_MAP.items()
                if config == metadata['configuration_id']
            )
            frames.append(group)
        combos = pd.DataFrame(
            [{**configuration_metadata(name), "experiment_dir": name} for name in BENCHMARK_CONFIG_MAP]
        )
        frames.append(baseline_frame.loc[~covariate_model].merge(combos, how="cross"))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def discover_legacy_metric_rows(results_dir: Path) -> pd.DataFrame:
    """Walk the single-run results/ layout (one experiment dir per feature combination, no seeds)."""
    records: list[pd.DataFrame] = []
    for experiment_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        metadata = configuration_metadata(experiment_dir.name)
        for frame in _read_metric_files(
            experiment_dir, "fold_*/per_trait_metrics_original.csv", "gnn", "gnn", NO_SEED
        ):
            for key, value in metadata.items():
                frame[key] = value
            frame["experiment_dir"] = experiment_dir.name
            records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

def load_metric_rows(results_dir: Path, traits_file: Path) -> pd.DataFrame:
    if not results_dir.exists():
        return pd.DataFrame()

    metrics = (
        discover_benchmark_metric_rows(results_dir)
        if is_benchmark_layout(results_dir)
        else discover_legacy_metric_rows(results_dir)
    )
    if metrics.empty:
        return pd.DataFrame()

    numeric_columns = ["n", "RMSE", "MAE", "Pearson_r", "Spearman_rho", "seed"]
    for column in numeric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["display_model"] = metrics["model"].map(model_display_name)
    scales = trait_scales(traits_file)
    metrics = metrics.merge(scales, on=["trait", "variable"], how="left")
    for metric, source_column in (("NRMSE_IQR", "RMSE"), ("NMAE_IQR", "MAE")):
        denominator = metrics["trait_iqr"].where(metrics["trait_iqr"] > 0)
        metrics[metric] = metrics[source_column] / denominator
    for metric, source_column in (("NRMSE_range", "RMSE"), ("NMAE_range", "MAE")):
        denominator = metrics["trait_range"].where(metrics["trait_range"] > 0)
        metrics[metric] = metrics[source_column] / denominator
    gnn_param_counts = discover_gnn_parameter_counts(results_dir)
    estimated_k_parameters = [
        gnn_param_counts.get(configuration_id, float("nan"))
        if model == "gnn"
        else baseline_parameter_count(model, observed_n)
        for model, configuration_id, observed_n in zip(
            metrics["model"], metrics["configuration_id"], metrics["observed_n"]
        )
    ]
    # Prefer a model's actual measured complexity (e.g. MICE's per-fold rpart leaf count) over the
    # analytic estimate, when the baseline fitting script saved one (see model_complexity.csv).
    measured_k_parameters = metrics.get("measured_k_parameters", pd.Series(np.nan, index=metrics.index))
    metrics["k_parameters"] = measured_k_parameters.where(measured_k_parameters.notna(), estimated_k_parameters)
    return sort_configurations(metrics)

def weighted_fisher_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    clipped = values.loc[valid].clip(-0.999999, 0.999999)
    correlation_weights = np.maximum(weights.loc[valid].to_numpy(dtype=float) - 3.0, 1.0)
    return float(np.tanh(np.average(np.arctanh(clipped.to_numpy(dtype=float)), weights=correlation_weights)))

def _aggregate_metric_group(group: pd.DataFrame) -> dict[str, float]:
    weights = group["n"].fillna(0).clip(lower=0)
    total_n = float(weights.sum())
    valid_rmse = group["RMSE"].notna() & (weights > 0)
    valid_mae = group["MAE"].notna() & (weights > 0)
    rmse = (
        float(np.sqrt(np.average(np.square(group.loc[valid_rmse, "RMSE"]), weights=weights.loc[valid_rmse])))
        if valid_rmse.any()
        else float("nan")
    )
    mae = (
        float(np.average(group.loc[valid_mae, "MAE"], weights=weights.loc[valid_mae]))
        if valid_mae.any()
        else float("nan")
    )
    return {
        "n": total_n,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson_r": weighted_fisher_mean(group["Pearson_r"], weights),
        "Spearman_rho": weighted_fisher_mean(group["Spearman_rho"], weights),
    }

def aic_score(rmse: float, n: float, k_parameters: float) -> float:
    """Gaussian-error AIC = n*ln(RMSE^2) + 2k; lower means a better error/complexity trade-off."""
    if not (pd.notna(rmse) and pd.notna(n) and pd.notna(k_parameters)) or rmse <= 0 or n <= 0:
        return float("nan")
    return float(n * np.log(np.square(rmse)) + 2 * k_parameters)

def aggregate_seed_level(metric_rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse folds into one row per (model, configuration, seed, trait, variable)."""
    if metric_rows.empty:
        return pd.DataFrame()

    group_columns = [
        "configuration_id", "configuration", "has_environment", "has_phylogeny",
        "model", "source", "display_model", "seed", "trait", "variable",
    ]
    records: list[dict[str, object]] = []
    for key, group in metric_rows.groupby(group_columns, dropna=False, sort=False):
        aggregated = _aggregate_metric_group(group)
        if aggregated["n"] == 0:
            continue
        trait_iqr = group["trait_iqr"].dropna().iloc[0] if group["trait_iqr"].notna().any() else np.nan
        trait_range = group["trait_range"].dropna().iloc[0] if group["trait_range"].notna().any() else np.nan
        observed_n = group["observed_n"].dropna().iloc[0] if group["observed_n"].notna().any() else np.nan
        k_parameters = group["k_parameters"].mean() if group["k_parameters"].notna().any() else np.nan
        record = {**dict(zip(group_columns, key, strict=True)), **aggregated}
        record["folds"] = int(group["fold"].nunique())
        record["trait_iqr"] = trait_iqr
        record["trait_range"] = trait_range
        record["observed_n"] = observed_n
        record["k_parameters"] = k_parameters
        record["NRMSE_IQR"] = record["RMSE"] / trait_iqr if pd.notna(trait_iqr) and trait_iqr > 0 else np.nan
        record["NRMSE_range"] = record["RMSE"] / trait_range if pd.notna(trait_range) and trait_range > 0 else np.nan
        record["NMAE_IQR"] = record["MAE"] / trait_iqr if pd.notna(trait_iqr) and trait_iqr > 0 else np.nan
        record["AIC"] = aic_score(record["RMSE"], record["n"], k_parameters)
        records.append(record)
    return sort_configurations(pd.DataFrame(records))

def aggregate_across_seeds(seed_level: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated seeds into a mean +/- std per (model, configuration, trait, variable)."""
    if seed_level.empty:
        return pd.DataFrame()

    group_columns = [
        "configuration_id", "configuration", "has_environment", "has_phylogeny",
        "model", "source", "display_model", "trait", "variable",
    ]
    metric_columns = ["RMSE", "MAE", "Pearson_r", "Spearman_rho", "NRMSE_IQR", "NRMSE_range", "NMAE_IQR", "AIC"]
    records: list[dict[str, object]] = []
    for key, group in seed_level.groupby(group_columns, dropna=False, sort=False):
        record = dict(zip(group_columns, key, strict=True))
        record["n_seeds"] = int(group.shape[0])
        record["n"] = int(group["n"].sum())
        record["folds"] = int(group["folds"].sum())
        for column in metric_columns:
            values = group[column].dropna()
            record[column] = float(values.mean()) if not values.empty else np.nan
            record[f"{column}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        record["trait_iqr"] = group["trait_iqr"].dropna().iloc[0] if group["trait_iqr"].notna().any() else np.nan
        record["trait_range"] = group["trait_range"].dropna().iloc[0] if group["trait_range"].notna().any() else np.nan
        record["observed_n"] = group["observed_n"].dropna().iloc[0] if group["observed_n"].notna().any() else np.nan
        record["k_parameters"] = group["k_parameters"].mean() if group["k_parameters"].notna().any() else np.nan
        records.append(record)
    return sort_configurations(pd.DataFrame(records))
