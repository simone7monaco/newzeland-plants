from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "benchmark_results"
    if (PROJECT_ROOT / "benchmark_results").exists()
    else PROJECT_ROOT / "results"
)
DEFAULT_TRAITS_FILE = PROJECT_ROOT / "data" / "Ferns" / "FernMinMax.xlsx"

# Maps run_benchmark.py's feature-combination folder names to the dashboard's configuration ids.
BENCHMARK_CONFIG_MAP = {
    "no_env_no_phylo": "baseline",
    "env_only": "environment",
    "phylo_only": "phylogeny",
    "env_phylo": "full",
}
CONFIG_ORDER = {"baseline": 0, "environment": 1, "phylogeny": 2, "full": 3}
CONFIG_LABELS = {
    "baseline": "Nessun input accessorio",
    "environment": "Solo ambiente",
    "phylogeny": "Solo filogenesi",
    "full": "Ambiente + filogenesi",
}
SOURCE_LABELS = {
    "gnn": "Rete neurale (GNN)",
    "deterministic_baseline": "Baseline deterministica",
    "r_baseline": "Baseline stocastica",
}
# Identifier/metadata columns in attribution files that must never be treated as attributed features.
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
    "RMSE / IQR (robusto)": ("NRMSE_IQR", True),
    "RMSE / range": ("NRMSE_range", True),
    "RMSE": ("RMSE", True),
    "MAE": ("MAE", True),
    "Pearson r": ("Pearson_r", False),
    "Spearman rho": ("Spearman_rho", False),
    "AIC (accuratezza vs complessita)": ("AIC", True),
}
# Sentinel seed for models/runs without seed-to-seed variability (deterministic baselines, legacy runs).
NO_SEED = -1.0
# k-NN baselines fit no likelihood parameters; their model flexibility is instead estimated with the
# classic kNN effective-degrees-of-freedom approximation, k_eff = n_train / k (Hastie, Tibshirani &
# Friedman, "The Elements of Statistical Learning", 2009, sec. 7.6). n_train is proxied by the trait's
# "observed_n" (species with an observed value for that trait), and k is the neighbour count each
# baseline actually uses in build_baseline_model() in r_baselines_fit.py.
BASELINE_KNN_NEIGHBORS: dict[str, int] = {
    "phylo_nn": 1,
    "phylo_knn": 5,
}
# Remaining baselines: exact/standard parameter counts for what they actually fit, used only for the
# AIC diagnostic (GNN parameter counts are instead read exactly from checkpoints).
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
        has_environment = "_ENV_" in upper
        has_phylogeny = "_PHYLO_" in upper
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
    if model == "gnn":
        return "GNN"
    match = re.fullmatch(r"(.+)_(prot\d+)", model)
    base, protocol = match.groups() if match else (model, None)
    label = base.replace("_", " ").title()
    return f"{label} ({protocol})" if protocol else label


def baseline_parameter_count(model: str, observed_n: float) -> float:
    base = re.sub(r"_prot\d+$", "", model).lower()
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
        # Baselines are fit once (per seed) and reused for every feature combination; replicate them
        # across all four configurations so they can be compared against the GNN in each one.
        combos = pd.DataFrame(
            [{**configuration_metadata(name), "experiment_dir": name} for name in BENCHMARK_CONFIG_MAP]
        )
        frames.append(baseline_frame.merge(combos, how="cross"))

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


def representative_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """De-duplicate baseline rows that are identical across configurations (they don't use env/phylo inputs)."""
    if summary.empty:
        return summary
    gnn_rows = summary.loc[summary["model"] == "gnn"]
    baseline_rows = summary.loc[(summary["model"] != "gnn") & (summary["configuration_id"] == "baseline")]
    return sort_configurations(pd.concat([gnn_rows, baseline_rows], ignore_index=True))


def classify_reliability(summary: pd.DataFrame, correlation_floor: float, relative_rmse_limit: float) -> pd.DataFrame:
    output = summary.copy()
    strong = (output["Pearson_r"] >= 0.70) & (output["NRMSE_IQR"] <= relative_rmse_limit * 0.5)
    usable = (output["Pearson_r"] >= correlation_floor) & (output["NRMSE_IQR"] <= relative_rmse_limit)
    output["reliability"] = np.select(
        [strong, usable],
        ["Forte", "Utilizzabile"],
        default="Debole / non interpretabile",
    )
    return output


def _detect_attribution_protocol(metadata_files: list[Path]) -> str:
    if not metadata_files:
        return "legacy_target_visible"
    try:
        payloads = [json.loads(path.read_text()) for path in metadata_files]
    except (OSError, json.JSONDecodeError):
        return "legacy_target_visible"
    if all(item.get("protocol") == "leave_one_trait_out_target_masked" for item in payloads):
        return "leave_one_trait_out_target_masked"
    return "legacy_target_visible"


def discover_benchmark_attribution_rows(results_dir: Path, kind: str) -> pd.DataFrame:
    """Attribution is only produced by the GNN; run_benchmark.py already merges it across seeds."""
    runs_dir = results_dir / "runs"
    records: list[pd.DataFrame] = []
    for combination_name in BENCHMARK_CONFIG_MAP:
        merged_file = results_dir / combination_name / f"attributions_{kind}_all.csv"
        if not merged_file.exists():
            continue
        frame = pd.read_csv(merged_file)
        if not {"species", "target_trait", "variable"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        metadata = configuration_metadata(combination_name)
        for key, value in metadata.items():
            frame[key] = value
        frame["experiment_dir"] = combination_name
        metadata_files = sorted(runs_dir.glob(f"seed_*/gnn/{combination_name}/*/fold_*/attributions_metadata.json"))
        frame["attribution_protocol"] = _detect_attribution_protocol(metadata_files)
        records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def discover_legacy_attribution_rows(results_dir: Path, kind: str) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for experiment_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        merged_file = experiment_dir / f"attributions_{kind}_all.csv"
        attribution_files = [merged_file] if merged_file.exists() else sorted(experiment_dir.glob(f"fold_*/attributions_{kind}.csv"))
        if not attribution_files:
            continue
        metadata = configuration_metadata(experiment_dir.name)
        protocol = _detect_attribution_protocol(sorted(experiment_dir.glob("fold_*/attributions_metadata.json")))
        for attribution_file in attribution_files:
            frame = pd.read_csv(attribution_file)
            if not {"species", "target_trait", "variable"}.issubset(frame.columns):
                continue
            frame = frame.copy()
            frame["experiment_dir"] = experiment_dir.name
            frame["attribution_protocol"] = protocol
            for key, value in metadata.items():
                frame[key] = value
            records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def load_attribution_rows(results_dir: Path, kind: str) -> pd.DataFrame:
    if not results_dir.exists():
        return pd.DataFrame()
    frame = (
        discover_benchmark_attribution_rows(results_dir, kind)
        if is_benchmark_layout(results_dir)
        else discover_legacy_attribution_rows(results_dir, kind)
    )
    return sort_configurations(frame) if not frame.empty else frame


def pretty_environment_name(name: str) -> str:
    labels = {
        "wc2.1_2.5m_bio_1_1": "Temperatura media annuale (BIO1)",
        "wc2.1_2.5m_bio_2_1": "Escursione termica diurna media (BIO2)",
        "wc2.1_2.5m_bio_7_1": "Escursione termica annuale (BIO7)",
        "wc2.1_2.5m_bio_12_1": "Precipitazione annuale (BIO12)",
        "wc2.1_2.5m_bio_15_1": "Stagionalita delle precipitazioni (BIO15)",
    }
    if name in labels:
        return labels[name]
    solar = re.fullmatch(r"wc2\.1_2\.5m_srad_(\d{2})_1", name)
    if solar:
        return f"Radiazione solare, mese {solar.group(1)}"
    vapor = re.fullmatch(r"wc2\.1_2\.5m_vapr_(\d{2})_1", name)
    if vapor:
        return f"Pressione di vapore, mese {vapor.group(1)}"
    return name.replace("_", " ")


def environment_source_columns(project_root: Path) -> list[str]:
    complete_layers = project_root / "data" / "Ferns" / "Complete layers"
    cache_files = [
        complete_layers / f"Climatic layers_space_df_v{FEATURE_CACHE_VERSION}.csv",
        complete_layers / f"population density and elevation layer_space_df_v{FEATURE_CACHE_VERSION}.csv",
        complete_layers / f"Soil NZ layers_space_df_v{FEATURE_CACHE_VERSION}.csv",
    ]
    if all(path.exists() for path in cache_files):
        return [
            column
            for path in cache_files
            for column in pd.read_csv(path, index_col=0).columns.astype(str).tolist()
        ]

    legacy_cache = complete_layers / "{data_path}_space_df.csv"
    if legacy_cache.exists():
        return pd.read_csv(legacy_cache, index_col=0).columns.astype(str).tolist()
    return []


def environment_metadata(project_root: Path, spatial_attributions: pd.DataFrame) -> dict[str, object]:
    if spatial_attributions.empty:
        return {"labels": {}, "environment_columns": [], "source_columns": [], "replication_factor": 0, "replicated": False}
    environment_columns = sorted(
        (column for column in spatial_attributions.columns if re.fullmatch(r"env_\d+", str(column))),
        key=lambda column: int(str(column).split("_")[1]),
    )
    source_columns = environment_source_columns(project_root)
    replication_factor = 0
    replicated = False
    if source_columns and len(environment_columns) % len(source_columns) == 0:
        replication_factor = len(environment_columns) // len(source_columns)
        replicated = replication_factor > 1

    labels: dict[str, dict[str, str]] = {}
    for position, column in enumerate(environment_columns):
        source_name = source_columns[position % len(source_columns)] if source_columns else column
        canonical = pretty_environment_name(source_name)
        block = position // len(source_columns) + 1 if source_columns else 1
        display = f"{canonical} [blocco {block}]" if replicated else canonical
        labels[column] = {"display": display, "canonical": canonical}
    return {
        "labels": labels,
        "environment_columns": environment_columns,
        "source_columns": source_columns,
        "replication_factor": replication_factor,
        "replicated": replicated,
    }


def species_feature_descriptor(feature: str) -> tuple[str, str]:
    trait_input = re.fullmatch(r"(min|max|range)_(.+)", feature)
    if trait_input:
        return "Tratti osservati", f"{trait_input.group(2)} ({trait_input.group(1)})"
    if feature.startswith("gen_"):
        return "Genetica e categorie", feature.removeprefix("gen_").replace("_", ": ", 1)
    if feature.lower() == "phylo" or feature.startswith("phylo_"):
        return "Filogenesi", "Embedding filogenetico" if feature.lower() == "phylo" else feature.replace("_", " ")
    return "Altri input di specie", feature.replace("_", " ")


def spatial_feature_descriptor(feature: str, metadata: dict[str, object]) -> tuple[str, str, str]:
    if feature.startswith("pos_"):
        position = int(feature.split("_")[1]) + 1
        label = f"Codifica posizione {position}"
        return "Posizione", label, label
    labels = metadata.get("labels", {})
    if feature in labels:
        label_data = labels[feature]
        return "Ambiente", label_data["display"], label_data["canonical"]
    label = pretty_environment_name(feature)
    return "Ambiente", label, label


def summarize_attributions(
    attributions: pd.DataFrame,
    kind: str,
    environmental_data: dict[str, object] | None = None,
    collapse_environment_replicas: bool = True,
) -> pd.DataFrame:
    if attributions.empty:
        return pd.DataFrame()

    feature_columns = [column for column in attributions.columns if column not in EXCLUDED_ATTRIBUTION_COLUMNS]
    if not feature_columns:
        return pd.DataFrame()
    identity_columns = [column for column in IDENTITY_ATTRIBUTION_COLUMNS if column in attributions.columns]
    has_seed = "seed" in attributions.columns
    optional_columns = ["seed"] if has_seed else []
    work = attributions[identity_columns + feature_columns + optional_columns].copy()
    work["_row"] = np.arange(work.shape[0])
    long = work.melt(
        id_vars=identity_columns + ["_row"] + optional_columns,
        value_vars=feature_columns,
        var_name="input_feature",
        value_name="signed_ig",
    )
    long["signed_ig"] = pd.to_numeric(long["signed_ig"], errors="coerce").fillna(0.0)

    if kind == "species":
        descriptors = [species_feature_descriptor(str(feature)) for feature in long["input_feature"]]
        long["input_group"] = [descriptor[0] for descriptor in descriptors]
        long["display_feature"] = [descriptor[1] for descriptor in descriptors]
        long["feature_key"] = long["display_feature"]
    else:
        metadata = environmental_data or {}
        descriptors = [spatial_feature_descriptor(str(feature), metadata) for feature in long["input_feature"]]
        long["input_group"] = [descriptor[0] for descriptor in descriptors]
        long["display_feature"] = [descriptor[1] for descriptor in descriptors]
        long["feature_key"] = [descriptor[2] if collapse_environment_replicas and descriptor[0] == "Ambiente" else descriptor[1] for descriptor in descriptors]

    long["absolute_ig"] = long["signed_ig"].abs()
    sample_columns = identity_columns + ["_row", "input_group", "feature_key"]
    per_sample_agg = {"absolute_ig": ("absolute_ig", "sum"), "signed_ig": ("signed_ig", "sum")}
    if has_seed:
        per_sample_agg["seed"] = ("seed", "first")
    per_sample = long.groupby(sample_columns, as_index=False).agg(**per_sample_agg)
    summary_columns = [column for column in identity_columns if column != "species"] + ["input_group", "feature_key"]
    summary_agg = {
        "mean_abs_ig": ("absolute_ig", "mean"),
        "median_abs_ig": ("absolute_ig", "median"),
        "mean_signed_ig": ("signed_ig", "mean"),
        "positive_fraction": ("signed_ig", lambda values: float((values > 0).mean())),
        "samples": ("signed_ig", "size"),
    }
    if has_seed:
        summary_agg["seed_count"] = ("seed", "nunique")
    summary = per_sample.groupby(summary_columns, as_index=False).agg(**summary_agg)
    normalizer = summary.groupby([column for column in summary_columns if column not in {"input_group", "feature_key"}])["mean_abs_ig"].transform("sum")
    summary["importance_share"] = summary["mean_abs_ig"] / normalizer.replace(0, np.nan)
    return sort_configurations(summary.rename(columns={"feature_key": "feature"}))


def grouped_attribution_importance(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    columns = ["configuration_id", "configuration", "attribution_protocol", "target_trait", "variable", "input_group"]
    grouped = summary.groupby(columns, as_index=False)["mean_abs_ig"].sum()
    total = grouped.groupby(columns[:-1])["mean_abs_ig"].transform("sum")
    grouped["importance_share"] = grouped["mean_abs_ig"] / total.replace(0, np.nan)
    return sort_configurations(grouped)


def ablation_deltas(summary: pd.DataFrame, metric: str, lower_is_better: bool) -> pd.DataFrame:
    if summary.empty or metric not in summary:
        return pd.DataFrame()
    index_columns = ["trait", "variable"]
    pivot = summary.pivot_table(index=index_columns, columns="configuration_id", values=metric, aggfunc="first")
    comparisons = [
        ("Ambiente senza filogenesi", "baseline", "environment"),
        ("Ambiente con filogenesi", "phylogeny", "full"),
        ("Filogenesi senza ambiente", "baseline", "phylogeny"),
        ("Filogenesi con ambiente", "environment", "full"),
    ]
    records: list[dict[str, object]] = []
    for label, reference, treatment in comparisons:
        if reference not in pivot or treatment not in pivot:
            continue
        for (trait, variable), values in pivot[[reference, treatment]].dropna().iterrows():
            reference_value = float(values[reference])
            treatment_value = float(values[treatment])
            benefit = reference_value - treatment_value if lower_is_better else treatment_value - reference_value
            relative_benefit = benefit / abs(reference_value) if abs(reference_value) > 1e-12 else np.nan
            records.append(
                {
                    "comparison": label,
                    "trait": trait,
                    "variable": variable,
                    "reference": CONFIG_LABELS[reference],
                    "treatment": CONFIG_LABELS[treatment],
                    "benefit": benefit,
                    "relative_benefit": relative_benefit,
                }
            )
    return pd.DataFrame(records)


def overall_scores(summary: pd.DataFrame, model: str = "gnn") -> pd.DataFrame:
    scoped = summary.loc[summary["model"] == model] if not summary.empty else summary
    if scoped.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for (configuration_id, configuration), group in scoped.groupby(["configuration_id", "configuration"], sort=False):
        records.append(
            {
                "configuration_id": configuration_id,
                "configuration": configuration,
                "median_nrmse_iqr": group["NRMSE_IQR"].median(),
                "median_correlation": group["Pearson_r"].median(),
                "usable_share": float(group["reliability"].isin(["Forte", "Utilizzabile"]).mean()),
                "outputs": int(group.shape[0]),
                "evaluations": int(group["n"].sum()),
                "n_seeds": int(group["n_seeds"].max()) if "n_seeds" in group else 1,
            }
        )
    return sort_configurations(pd.DataFrame(records))


def display_slice_name(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["trait_variable"] = output["trait"].astype(str) + " - " + output["variable"].astype(str)
    return output


def reliability_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "display_model",
        "configuration",
        "trait",
        "variable",
        "n_seeds",
        "folds",
        "n",
        "k_parameters",
        "RMSE",
        "RMSE_std",
        "MAE",
        "MAE_std",
        "NRMSE_IQR",
        "NRMSE_IQR_std",
        "NRMSE_range",
        "Pearson_r",
        "Pearson_r_std",
        "Spearman_rho",
        "AIC",
        "AIC_std",
        "reliability",
    ]
    present = [column for column in columns if column in summary.columns]
    return display_slice_name(summary)[present + ["trait_variable"]].sort_values(["configuration", "display_model", "trait", "variable"])


def render_overview(seed_level: pd.DataFrame, summary: pd.DataFrame, selected_metric: str, selected_models: list[str]) -> None:
    metric, lower_is_better = METRIC_OPTIONS[selected_metric]
    st.subheader("Accuratezza e affidabilita")
    gnn_scores = overall_scores(summary, model="gnn")
    cards = st.columns(max(len(gnn_scores), 1))
    for column, (_, score) in zip(cards, gnn_scores.iterrows(), strict=False):
        column.metric(
            score["configuration"],
            f"{score['median_nrmse_iqr']:.2f} RMSE/IQR",
            f"r mediano {score['median_correlation']:.2f}",
            help="Valori RMSE/IQR piu bassi e correlazioni piu alte sono preferibili. Riferito al modello GNN.",
        )
        column.caption(f"{score['usable_share']:.0%} output utilizzabili o forti su {score['n_seeds']} inizializzazioni")

    scoped_seed_level = seed_level.loc[seed_level["display_model"].isin(selected_models)]
    display = display_slice_name(scoped_seed_level)
    figure = px.box(
        display,
        x="trait_variable",
        y=metric,
        color="display_model",
        facet_col="configuration",
        facet_col_wrap=2,
        points="all",
        category_orders={"configuration": list(CONFIG_LABELS.values())},
        labels={metric: selected_metric, "trait_variable": "Trait e variabile", "display_model": "Modello"},
        title=f"{selected_metric} su CV leave-one-trait-out, ripetuto su piu inizializzazioni",
    )
    figure.update_xaxes(matches=None)
    figure.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=55, b=10), height=680)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Ogni punto e una diversa inizializzazione (seed); il box riassume mediana e dispersione tra inizializzazioni ripetute. "
        "Le baseline deterministiche non variano con il seed e mostrano un solo punto; sono comunque replicate in ogni pannello di configurazione "
        "perche non dipendono da ambiente o filogenesi."
    )

    representative = representative_rows(summary)
    representative = representative.loc[representative["display_model"].isin(selected_models)].copy()
    representative["row_label"] = np.where(
        representative["model"] == "gnn",
        representative["configuration"],
        representative["display_model"] + " (" + representative["source"].map(SOURCE_LABELS).fillna(representative["source"]) + ")",
    )
    heatmap_display = display_slice_name(representative)
    heatmap_source = heatmap_display.pivot_table(index="row_label", columns="trait_variable", values=metric, aggfunc="mean")
    gnn_order = [label for label in CONFIG_LABELS.values() if label in heatmap_source.index]
    other_order = sorted(label for label in heatmap_source.index if label not in gnn_order)
    heatmap_source = heatmap_source.reindex(index=gnn_order + other_order)
    heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_source.to_numpy(),
            x=heatmap_source.columns.tolist(),
            y=heatmap_source.index.tolist(),
            colorscale="RdYlGn_r" if lower_is_better else "RdYlGn",
            colorbar_title=selected_metric,
            hovertemplate="Modello/configurazione: %{y}<br>Trait: %{x}<br>Valore: %{z:.3f}<extra></extra>",
        )
    )
    heatmap.update_layout(
        title="Matrice comparativa (media sulle inizializzazioni)",
        margin=dict(l=10, r=10, t=55, b=10),
        height=max(330, 40 * len(heatmap_source.index)),
    )
    st.plotly_chart(heatmap, width="stretch")

    st.markdown("#### Accuratezza vs complessita del modello (AIC)")
    complexity_source = representative.dropna(subset=["k_parameters", "RMSE"]).copy()
    if complexity_source.empty:
        st.info("Numero di parametri non disponibile per i modelli selezionati (es. MICE, non parametrico).")
    else:
        complexity_agg = complexity_source.groupby(
            ["row_label", "display_model", "configuration", "k_parameters"], as_index=False
        ).agg(RMSE=("RMSE", "median"), AIC=("AIC", "median"))
        complexity_figure = px.scatter(
            complexity_agg,
            x="k_parameters",
            y="RMSE",
            color="display_model",
            symbol="configuration",
            hover_data={"row_label": True, "AIC": ":.1f", "k_parameters": ":,.0f"},
            labels={
                "k_parameters": "Parametri liberi del modello (scala log)",
                "RMSE": "RMSE mediano (trait/variabile)",
                "display_model": "Modello",
            },
            title="Ogni punto e un modello/configurazione: RMSE mediano rispetto al numero di parametri",
            log_x=True,
        )
        complexity_figure.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=55, b=10), height=480)
        st.plotly_chart(complexity_figure, width="stretch")
        st.caption(
            "Punti in basso a sinistra vincono su entrambi i fronti. Un punto piu in basso ma molto piu a destra migliora "
            "l'errore aggiungendo pero molta complessita: l'AIC (colonna nella tabella sotto e nel selettore metrica) penalizza "
            "questo compromesso aggiungendo 2 punti per parametro in piu. I pesi della GNN sono letti dai checkpoint; per le "
            "baseline phylo-kNN il conteggio usa i gradi di liberta effettivi n_train/k; per training mean/median e Rphylopars BM "
            "corrisponde al numero di parametri stimati per massima verosimiglianza; per MICE e il numero di foglie rpart misurato "
            "a ogni fold/seed (se il benchmark e stato rigenerato dopo l'aggiunta di questa misura), mediato sulle ripetizioni; "
            "run precedenti senza questa misura restano esclusi dall'AIC."
        )

    st.dataframe(
        reliability_table(representative),
        width="stretch",
        hide_index=True,
        column_config={
            "display_model": st.column_config.Column("Modello"),
            "n_seeds": st.column_config.NumberColumn("Inizializzazioni", format="%d"),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "RMSE_std": st.column_config.NumberColumn("RMSE (dev.std. seed)", format="%.2f"),
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "MAE_std": st.column_config.NumberColumn("MAE (dev.std. seed)", format="%.2f"),
            "NRMSE_IQR": st.column_config.NumberColumn("RMSE / IQR", format="%.2f"),
            "NRMSE_IQR_std": st.column_config.NumberColumn("RMSE / IQR (dev.std. seed)", format="%.2f"),
            "NRMSE_range": st.column_config.NumberColumn("RMSE / range", format="%.2f"),
            "Pearson_r": st.column_config.NumberColumn("Pearson r", format="%.2f"),
            "Pearson_r_std": st.column_config.NumberColumn("Pearson r (dev.std. seed)", format="%.2f"),
            "Spearman_rho": st.column_config.NumberColumn("Spearman rho", format="%.2f"),
            "k_parameters": st.column_config.NumberColumn("Parametri (k)", format="%.0f"),
            "AIC": st.column_config.NumberColumn("AIC", format="%.1f"),
            "AIC_std": st.column_config.NumberColumn("AIC (dev.std. seed)", format="%.1f"),
        },
    )
    st.caption(
        "RMSE e MAE sono aggregati pesando per il numero di osservazioni entro ogni inizializzazione, poi mediati sulle inizializzazioni. "
        "Le colonne 'dev.std. seed' misurano la variabilita tra inizializzazioni ripetute (0 se ne e disponibile una sola). "
        "RMSE/IQR usa l'IQR delle osservazioni originali e permette confronti tra scale diverse."
    )


def render_ablation(summary: pd.DataFrame, selected_metric: str) -> None:
    metric, lower_is_better = METRIC_OPTIONS[selected_metric]
    gnn_summary = summary.loc[summary["model"] == "gnn"]
    deltas = ablation_deltas(gnn_summary, metric, lower_is_better)
    st.subheader("Effetto marginale degli input accessori (GNN)")
    st.caption("Le baseline non dipendono da ambiente/filogenesi e sono escluse da questo confronto fattoriale.")
    if deltas.empty:
        st.info("Non sono disponibili tutte le quattro configurazioni richieste per il confronto fattoriale.")
        return

    aggregate = deltas.groupby("comparison", as_index=False).agg(
        median_benefit=("benefit", "median"),
        mean_benefit=("benefit", "mean"),
        improved_share=("benefit", lambda values: float((values > 0).mean())),
        slices=("benefit", "size"),
    )
    aggregate["improved_share"] *= 100
    st.dataframe(
        aggregate,
        width="stretch",
        hide_index=True,
        column_config={
            "median_benefit": st.column_config.NumberColumn("Beneficio mediano", format="%.3f"),
            "mean_benefit": st.column_config.NumberColumn("Beneficio medio", format="%.3f"),
            "improved_share": st.column_config.NumberColumn("Slice migliorate (%)", format="%.0f%%"),
        },
    )

    plot_data = display_slice_name(deltas)
    figure = px.strip(
        plot_data,
        x="comparison",
        y="benefit",
        color="comparison",
        hover_data={"trait": True, "variable": True, "reference": True, "treatment": True, "relative_benefit": ":.1%"},
        labels={"comparison": "Intervento", "benefit": f"Beneficio su {selected_metric}"},
        title="Beneficio per trait-variabile: positivo = trattamento migliore",
    )
    figure.add_hline(y=0, line_color="#5f6b73", line_width=1)
    figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10), height=430)
    st.plotly_chart(figure, width="stretch")

    selected_comparison = st.selectbox("Dettaglio confronto", aggregate["comparison"].tolist(), key="ablation_comparison")
    detail = plot_data.loc[plot_data["comparison"] == selected_comparison].sort_values("benefit")
    figure = px.bar(
        detail,
        x="benefit",
        y="trait_variable",
        orientation="h",
        color="benefit",
        color_continuous_scale="RdYlGn",
        labels={"benefit": f"Beneficio su {selected_metric}", "trait_variable": "Trait e variabile"},
        title=selected_comparison,
        hover_data={"reference": True, "treatment": True, "relative_benefit": ":.1%"},
    )
    figure.add_vline(x=0, line_color="#5f6b73", line_width=1)
    figure.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10), height=430)
    st.plotly_chart(figure, width="stretch")
    st.caption("Per metriche di errore, beneficio positivo significa errore piu basso. Per correlazioni, significa correlazione piu alta.")


def attribution_view(
    summary: pd.DataFrame,
    reliability: pd.DataFrame,
    title: str,
    configuration_id: str,
    trait: str,
    variable: str,
) -> None:
    st.markdown(f"#### {title}")
    selected = summary.loc[
        (summary["configuration_id"] == configuration_id)
        & (summary["target_trait"] == trait)
        & (summary["variable"] == variable)
    ].copy()
    if selected.empty:
        st.info("Attribution non disponibile per questa configurazione.")
        return

    selected = selected.sort_values("importance_share", ascending=False)
    group_data = selected.groupby("input_group", as_index=False)["importance_share"].sum().sort_values("importance_share", ascending=True)
    groups, features = st.columns((1, 2))
    group_figure = px.bar(
        group_data,
        x="importance_share",
        y="input_group",
        orientation="h",
        color="input_group",
        labels={"importance_share": "Quota di |IG|", "input_group": "Gruppo di input"},
        title="Quota per gruppo di input",
    )
    group_figure.update_layout(showlegend=False, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=45, b=10), height=360)
    groups.plotly_chart(group_figure, width="stretch")

    top = selected.head(15).sort_values("importance_share", ascending=True)
    feature_figure = px.bar(
        top,
        x="importance_share",
        y="feature",
        orientation="h",
        color="input_group",
        hover_data={"mean_abs_ig": ":.4g", "mean_signed_ig": ":.4g", "positive_fraction": ":.0%", "samples": True},
        labels={"importance_share": "Quota di |IG|", "feature": "Input", "input_group": "Gruppo"},
        title="Input con contributo medio assoluto piu alto",
    )
    feature_figure.update_layout(legend_title_text="", xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=45, b=10), height=460)
    features.plotly_chart(feature_figure, width="stretch")

    display_columns = ["feature", "input_group", "importance_share", "mean_abs_ig", "mean_signed_ig", "positive_fraction", "samples"]
    if "seed_count" in selected.columns:
        display_columns.append("seed_count")
    display = selected[display_columns].copy()
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "importance_share": st.column_config.NumberColumn("Quota |IG|", format="%.2f"),
            "mean_abs_ig": st.column_config.NumberColumn("Media |IG|", format="%.5f"),
            "mean_signed_ig": st.column_config.NumberColumn("IG medio firmato", format="%.5f"),
            "positive_fraction": st.column_config.NumberColumn("IG positivi", format="%.0f%%"),
            "seed_count": st.column_config.NumberColumn("Inizializzazioni", format="%d"),
        },
    )
    if "seed_count" in selected.columns and selected["seed_count"].max(skipna=True) and selected["seed_count"].max() > 1:
        st.caption("Media e quota di importanza calcolate su tutte le specie e tutte le inizializzazioni disponibili (robustezza al seed).")

    quality = reliability.loc[
        (reliability["configuration_id"] == configuration_id)
        & (reliability["trait"] == trait)
        & (reliability["variable"] == variable)
    ]
    if quality.empty:
        st.info("Nessuna metrica di accuratezza corrispondente trovata per questa attribution.")
    else:
        row = quality.iloc[0]
        message = f"Accuratezza della slice: {row['reliability']} (Pearson r={row['Pearson_r']:.2f}, RMSE/IQR={row['NRMSE_IQR']:.2f})."
        if row["reliability"] == "Debole / non interpretabile":
            st.warning(message + " Le attribution restano visibili come diagnostica, ma non sono una base solida per inferenze biologiche.")
        else:
            st.info(message)


def render_attributions(
    species_summary: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    reliability: pd.DataFrame,
    environmental_data: dict[str, object],
) -> None:
    st.subheader("Integrated Gradients: contributori principali")
    available = pd.concat([frame for frame in [species_summary, spatial_summary] if not frame.empty], ignore_index=True) if not species_summary.empty or not spatial_summary.empty else pd.DataFrame()
    if available.empty:
        st.info("Nessun file di attribution IG disponibile.")
        return

    quality = reliability[["configuration_id", "trait", "variable", "reliability"]].rename(columns={"trait": "target_trait"})
    available = available.merge(quality, on=["configuration_id", "target_trait", "variable"], how="left")
    counterfactual = available.loc[available["attribution_protocol"] == "leave_one_trait_out_target_masked"]
    legacy = available.loc[available["attribution_protocol"] == "legacy_target_visible"]
    if counterfactual.empty:
        st.error(
            "Tutte le attribution disponibili sono legacy: il trait target era ancora visibile durante IG. "
            "Non descrivono le sorgenti dell'imputazione leave-one-trait-out; rigenera XAI con il pipeline aggiornato."
        )
        if not st.toggle("Mostra IG legacy solo a scopo diagnostico", value=False, key="show_legacy_attributions"):
            return
        protocol_source = available
    elif legacy.empty:
        protocol_source = counterfactual
    else:
        st.warning("Sono presenti sia IG controfattuali sia IG legacy; per default sono mostrate solo le attribution controfattuali.")
        include_legacy = st.toggle("Includi anche IG legacy diagnostiche", value=False, key="show_legacy_attributions")
        protocol_source = available if include_legacy else counterfactual

    show_reliable_only = st.toggle(
        "Solo slice interpretabili",
        value=True,
        help="Mostra output con accuratezza classificata Forte o Utilizzabile dalle soglie nella barra laterale.",
        key="attribution_reliable_only",
    )
    reliable_available = protocol_source.loc[protocol_source["reliability"].isin(["Forte", "Utilizzabile"])]
    selection_source = reliable_available if show_reliable_only and not reliable_available.empty else protocol_source
    if show_reliable_only and reliable_available.empty:
        st.warning("Nessuna slice soddisfa le soglie attuali: sono mostrate tutte le attribution.")

    configurations = sort_configurations(selection_source[["configuration_id", "configuration"]].drop_duplicates())
    configuration_options = configurations["configuration_id"].tolist()
    if st.session_state.get("attribution_configuration") not in configuration_options:
        st.session_state["attribution_configuration"] = configuration_options[0]
    configuration_id = st.selectbox(
        "Configurazione",
        configuration_options,
        format_func=lambda item: CONFIG_LABELS.get(item, item),
        key="attribution_configuration",
    )
    scoped = selection_source.loc[selection_source["configuration_id"] == configuration_id]
    traits = sorted(scoped["target_trait"].dropna().unique().tolist())
    if st.session_state.get("attribution_trait") not in traits:
        st.session_state["attribution_trait"] = traits[0]
    trait = st.selectbox("Trait predetto", traits, key="attribution_trait")
    variables = sorted(scoped.loc[scoped["target_trait"] == trait, "variable"].dropna().unique().tolist())
    if st.session_state.get("attribution_variable") not in variables:
        st.session_state["attribution_variable"] = variables[0]
    variable = st.selectbox("Variabile predetta", variables, key="attribution_variable")

    allowed_protocols = selection_source["attribution_protocol"].dropna().unique().tolist()
    visible_species = species_summary.loc[species_summary["attribution_protocol"].isin(allowed_protocols)]
    visible_spatial = spatial_summary.loc[spatial_summary["attribution_protocol"].isin(allowed_protocols)]

    species_tab, spatial_tab, comparison_tab = st.tabs(["Specie e filogenesi", "Ambiente e posizione", "Confronto tra configurazioni"])
    with species_tab:
        attribution_view(visible_species, reliability, "Input di specie", configuration_id, trait, variable)
    with spatial_tab:
        attribution_view(visible_spatial, reliability, "Input ambientali e posizionali", configuration_id, trait, variable)
        if environmental_data.get("replicated"):
            st.warning(
                f"Sono state rilevate {len(environmental_data['environment_columns'])} colonne env, cioe {environmental_data['replication_factor']} blocchi di "
                f"{len(environmental_data['source_columns'])} colonne. La vista aggrega i blocchi con lo stesso nome sorgente."
            )
    with comparison_tab:
        grouped_frames = [grouped_attribution_importance(frame) for frame in [visible_species, visible_spatial] if not frame.empty]
        grouped = pd.concat(grouped_frames, ignore_index=True) if grouped_frames else pd.DataFrame()
        comparison = grouped.loc[(grouped["target_trait"] == trait) & (grouped["variable"] == variable)]
        if comparison.empty:
            st.info("Nessuna attribution comparabile disponibile per questa slice.")
        else:
            figure = px.bar(
                comparison,
                x="configuration",
                y="importance_share",
                color="input_group",
                barmode="stack",
                category_orders={"configuration": list(CONFIG_LABELS.values())},
                labels={"configuration": "Configurazione", "importance_share": "Quota di |IG|", "input_group": "Gruppo"},
                title="Come cambia la composizione dei contributi",
            )
            figure.update_layout(yaxis_tickformat=".0%", legend_title_text="", margin=dict(l=10, r=10, t=55, b=10), height=450)
            st.plotly_chart(figure, width="stretch")
            st.caption("Le quote confrontano la distribuzione interna di |IG| in ciascun modello, non un effetto causale ne una grandezza direttamente comparabile tra architetture.")


def render_data_quality(metric_rows: pd.DataFrame, summary: pd.DataFrame, environmental_data: dict[str, object]) -> None:
    st.subheader("Provenienza e limiti dei dati")
    manifest = metric_rows.groupby(
        ["configuration_id", "configuration", "model", "display_model", "source"], as_index=False
    ).agg(
        seeds=("seed", lambda values: int(values.loc[values != NO_SEED].nunique()) or 1),
        folds=("fold", "nunique"),
        metric_rows=("trait", "size"),
        evaluations=("n", "sum"),
    )
    manifest["source"] = manifest["source"].map(SOURCE_LABELS).fillna(manifest["source"])
    st.dataframe(sort_configurations(manifest), width="stretch", hide_index=True)

    weak = summary.loc[
        summary["reliability"] == "Debole / non interpretabile",
        ["display_model", "configuration", "trait", "variable", "Pearson_r", "NRMSE_IQR"],
    ]
    if not weak.empty:
        st.warning(
            f"{len(weak)} delle {len(summary)} combinazioni modello-configurazione-trait-variabile superano le soglie di affidabilita. "
            "Le relative attribution devono essere lette solo come diagnostica del modello."
        )

    if environmental_data.get("replicated"):
        st.error(
            "Le attribution ambientali non sono pienamente interpretabili come layer distinti: gli output contengono tre blocchi uguali per numero "
            "di feature rispetto alla tabella cache disponibile. Il dashboard li aggrega per nome sorgente; non attribuisce contributi a suolo, densita o elevazione."
        )
    else:
        st.info("Non e stata rilevata una ripetizione strutturale delle colonne ambientali nell'output disponibile.")

    export = reliability_table(representative_rows(summary)).to_csv(index=False).encode("utf-8")
    st.download_button("Scarica metriche aggregate CSV", data=export, file_name="cv_metrics_aggregated.csv", mime="text/csv")


@st.cache_data(show_spinner=False)
def cached_metric_rows(results_path: str, traits_path: str) -> pd.DataFrame:
    return load_metric_rows(Path(results_path), Path(traits_path))


@st.cache_data(show_spinner=False)
def cached_attributions(results_path: str, kind: str, schema_version: int) -> pd.DataFrame:
    return load_attribution_rows(Path(results_path), kind)


@st.cache_data(show_spinner=False)
def cached_environment_metadata(project_path: str, spatial_attributions: pd.DataFrame) -> dict[str, object]:
    return environment_metadata(Path(project_path), spatial_attributions)


def main() -> None:
    st.set_page_config(page_title="Fern imputation audit", page_icon="F", layout="wide", initial_sidebar_state="auto")
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(145deg, #f5f8f4 0%, #ffffff 48%, #edf4f0 100%); }
        [data-testid="stSidebar"] { background: #173c36; }
        [data-testid="stSidebar"] * { color: #f5fbf5; }
        [data-testid="stMetric"] { background: rgba(255,255,255,0.72); border: 1px solid #c7d9cf; border-radius: 6px; padding: 0.75rem; }
        h1, h2, h3 { color: #173c36; letter-spacing: 0; }
        input, textarea, select { color: #111111 !important; background-color: #ffffff !important; }
        div[data-baseweb="select"] * { color: #111111 !important; }
        div[data-baseweb="input"] * { color: #111111 !important; }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stTimeInput"] input { color: #111111 !important; background-color: #ffffff !important; }
        [data-testid="stSidebar"] button { background-color: #ffffff !important; }
        [data-testid="stSidebar"] button * { color: #111111 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Audit delle imputazioni dei tratti")
    st.caption(
        "Cross-validation leave-one-trait-out, metriche in scala originale, confronto GNN vs baseline "
        "e attribution Integrated Gradients, ripetuti su piu inizializzazioni"
    )

    with st.sidebar:
        st.header("Controlli")
        results_path = Path(st.text_input("Directory risultati", str(DEFAULT_RESULTS_DIR))).expanduser()
        correlation_floor = st.slider("Correlazione minima per uso interpretativo", 0.0, 0.9, 0.5, 0.05)
        relative_rmse_limit = st.slider("RMSE/IQR massimo per uso interpretativo", 0.25, 2.0, 1.0, 0.05)
        selected_metric = st.selectbox("Metrica di confronto", list(METRIC_OPTIONS))
        if st.button("Ricarica dati", type="secondary"):
            st.cache_data.clear()

    traits_path = results_path.parent / "data" / "Ferns" / "FernMinMax.xlsx"
    if not traits_path.exists():
        traits_path = DEFAULT_TRAITS_FILE
    metric_rows = cached_metric_rows(str(results_path), str(traits_path))
    if metric_rows.empty:
        st.error(f"Nessuna metrica per fold trovata in {results_path}.")
        return

    seed_level = aggregate_seed_level(metric_rows)
    summary = classify_reliability(aggregate_across_seeds(seed_level), correlation_floor, relative_rmse_limit)

    with st.sidebar:
        model_options = sorted(summary["display_model"].unique().tolist())
        selected_models = st.multiselect("Modelli da confrontare", model_options, default=model_options)

    species_attributions = cached_attributions(str(results_path), "species", ATTRIBUTION_CACHE_SCHEMA_VERSION)
    spatial_attributions = cached_attributions(str(results_path), "spatial", ATTRIBUTION_CACHE_SCHEMA_VERSION)
    environmental_data = cached_environment_metadata(str(results_path.parent), spatial_attributions)
    species_summary = summarize_attributions(species_attributions, "species")
    spatial_summary = summarize_attributions(spatial_attributions, "spatial", environmental_data)

    overview_tab, ablation_tab, attribution_tab, quality_tab = st.tabs(["Accuratezza", "Ablation", "Attribution", "Qualita dati"])
    with overview_tab:
        render_overview(seed_level, summary, selected_metric, selected_models)
    with ablation_tab:
        render_ablation(summary, selected_metric)
    with attribution_tab:
        render_attributions(species_summary, spatial_summary, summary, environmental_data)
    with quality_tab:
        render_data_quality(metric_rows, summary, environmental_data)


if __name__ == "__main__":
    main()