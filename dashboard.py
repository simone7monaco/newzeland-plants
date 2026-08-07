from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_TRAITS_FILE = PROJECT_ROOT / "data" / "Ferns" / "FernMinMax.xlsx"
METADATA_COLUMNS = {
    "species",
    "target_trait",
    "variable",
    "configuration_id",
    "configuration",
    "experiment_dir",
    "attribution_protocol",
}
CONFIG_ORDER = {"baseline": 0, "environment": 1, "phylogeny": 2, "full": 3}
CONFIG_LABELS = {
    "baseline": "Nessun input accessorio",
    "environment": "Solo ambiente",
    "phylogeny": "Solo filogenesi",
    "full": "Ambiente + filogenesi",
}
ATTRIBUTION_CACHE_SCHEMA_VERSION = 2
FEATURE_CACHE_VERSION = 2
METRIC_OPTIONS = {
    "RMSE / IQR (robusto)": ("NRMSE_IQR", True),
    "RMSE / range": ("NRMSE_range", True),
    "RMSE": ("RMSE", True),
    "MAE": ("MAE", True),
    "Pearson r": ("Pearson_r", False),
    "Spearman rho": ("Spearman_rho", False),
}


def configuration_metadata(experiment_name: str) -> dict[str, object]:
    name = experiment_name.upper()
    has_environment = "_ENV_" in name
    has_phylogeny = "_PHYLO_" in name
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


def load_metric_rows(results_dir: Path, traits_file: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    if not results_dir.exists():
        return pd.DataFrame()

    for experiment_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        metric_files = sorted(experiment_dir.glob("fold_*/per_trait_metrics_original.csv"))
        if not metric_files:
            continue
        metadata = configuration_metadata(experiment_dir.name)
        for metric_file in metric_files:
            frame = pd.read_csv(metric_file)
            expected = {"trait", "variable", "n", "RMSE", "MAE", "Pearson_r", "Spearman_rho"}
            if not expected.issubset(frame.columns):
                continue
            frame = frame.copy()
            frame["fold"] = metric_file.parent.name
            frame["experiment_dir"] = experiment_dir.name
            for key, value in metadata.items():
                frame[key] = value
            records.append(frame)

    if not records:
        return pd.DataFrame()

    metrics = pd.concat(records, ignore_index=True)
    numeric_columns = ["n", "RMSE", "MAE", "Pearson_r", "Spearman_rho"]
    for column in numeric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    scales = trait_scales(traits_file)
    metrics = metrics.merge(scales, on=["trait", "variable"], how="left")
    metrics["NRMSE_IQR"] = metrics["RMSE"] / metrics["trait_iqr"].replace(0, np.nan)
    metrics["NRMSE_range"] = metrics["RMSE"] / metrics["trait_range"].replace(0, np.nan)
    metrics["NMAE_IQR"] = metrics["MAE"] / metrics["trait_iqr"].replace(0, np.nan)
    return sort_configurations(metrics)


def weighted_fisher_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    clipped = values.loc[valid].clip(-0.999999, 0.999999)
    correlation_weights = np.maximum(weights.loc[valid].to_numpy(dtype=float) - 3.0, 1.0)
    return float(np.tanh(np.average(np.arctanh(clipped.to_numpy(dtype=float)), weights=correlation_weights)))


def aggregate_cv_metrics(metric_rows: pd.DataFrame) -> pd.DataFrame:
    if metric_rows.empty:
        return pd.DataFrame()

    group_columns = [
        "configuration_id",
        "configuration",
        "has_environment",
        "has_phylogeny",
        "trait",
        "variable",
    ]
    records: list[dict[str, object]] = []
    for key, group in metric_rows.groupby(group_columns, dropna=False, sort=False):
        weights = group["n"].fillna(0).clip(lower=0)
        total_n = float(weights.sum())
        if total_n == 0:
            continue
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
        trait_iqr = group["trait_iqr"].dropna().iloc[0] if group["trait_iqr"].notna().any() else np.nan
        trait_range = group["trait_range"].dropna().iloc[0] if group["trait_range"].notna().any() else np.nan
        observed_n = group["observed_n"].dropna().iloc[0] if group["observed_n"].notna().any() else np.nan
        records.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "folds": int(group["fold"].nunique()),
                "n": int(total_n),
                "RMSE": rmse,
                "MAE": mae,
                "Pearson_r": weighted_fisher_mean(group["Pearson_r"], weights),
                "Spearman_rho": weighted_fisher_mean(group["Spearman_rho"], weights),
                "trait_iqr": trait_iqr,
                "trait_range": trait_range,
                "observed_n": observed_n,
                "NRMSE_IQR": rmse / trait_iqr if pd.notna(trait_iqr) and trait_iqr > 0 else np.nan,
                "NRMSE_range": rmse / trait_range if pd.notna(trait_range) and trait_range > 0 else np.nan,
                "NMAE_IQR": mae / trait_iqr if pd.notna(trait_iqr) and trait_iqr > 0 else np.nan,
            }
        )
    return sort_configurations(pd.DataFrame(records))


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


def load_attribution_rows(results_dir: Path, kind: str) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    if not results_dir.exists():
        return pd.DataFrame()

    for experiment_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        merged_file = experiment_dir / f"attributions_{kind}_all.csv"
        attribution_files = [merged_file] if merged_file.exists() else sorted(experiment_dir.glob(f"fold_*/attributions_{kind}.csv"))
        metadata = configuration_metadata(experiment_dir.name)
        protocol_files = sorted(experiment_dir.glob("fold_*/attributions_metadata.json"))
        protocol = "legacy_target_visible"
        if protocol_files:
            try:
                protocol_metadata = [json.loads(path.read_text()) for path in protocol_files]
                if all(item.get("protocol") == "leave_one_trait_out_target_masked" for item in protocol_metadata):
                    protocol = "leave_one_trait_out_target_masked"
            except (OSError, json.JSONDecodeError):
                pass
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
    return sort_configurations(pd.concat(records, ignore_index=True)) if records else pd.DataFrame()


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

    feature_columns = [column for column in attributions.columns if column not in METADATA_COLUMNS and column not in {"has_environment", "has_phylogeny"}]
    if not feature_columns:
        return pd.DataFrame()
    identity_columns = [column for column in METADATA_COLUMNS if column in attributions.columns]
    work = attributions[identity_columns + feature_columns].copy()
    work["_row"] = np.arange(work.shape[0])
    long = work.melt(id_vars=identity_columns + ["_row"], value_vars=feature_columns, var_name="input_feature", value_name="signed_ig")
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
    per_sample = long.groupby(sample_columns, as_index=False).agg(absolute_ig=("absolute_ig", "sum"), signed_ig=("signed_ig", "sum"))
    summary_columns = [column for column in identity_columns if column != "species"] + ["input_group", "feature_key"]
    summary = per_sample.groupby(summary_columns, as_index=False).agg(
        mean_abs_ig=("absolute_ig", "mean"),
        median_abs_ig=("absolute_ig", "median"),
        mean_signed_ig=("signed_ig", "mean"),
        positive_fraction=("signed_ig", lambda values: float((values > 0).mean())),
        samples=("signed_ig", "size"),
    )
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


def overall_scores(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for (configuration_id, configuration), group in summary.groupby(["configuration_id", "configuration"], sort=False):
        records.append(
            {
                "configuration_id": configuration_id,
                "configuration": configuration,
                "median_nrmse_iqr": group["NRMSE_IQR"].median(),
                "median_correlation": group["Pearson_r"].median(),
                "usable_share": float(group["reliability"].isin(["Forte", "Utilizzabile"]).mean()),
                "outputs": int(group.shape[0]),
                "evaluations": int(group["n"].sum()),
            }
        )
    return sort_configurations(pd.DataFrame(records))


def display_slice_name(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["trait_variable"] = output["trait"].astype(str) + " - " + output["variable"].astype(str)
    return output


def reliability_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "configuration",
        "trait",
        "variable",
        "folds",
        "n",
        "RMSE",
        "MAE",
        "NRMSE_IQR",
        "NRMSE_range",
        "Pearson_r",
        "Spearman_rho",
        "reliability",
    ]
    return display_slice_name(summary)[columns + ["trait_variable"]].sort_values(["configuration", "trait", "variable"])


def render_overview(summary: pd.DataFrame, selected_metric: str) -> None:
    metric, lower_is_better = METRIC_OPTIONS[selected_metric]
    st.subheader("Accuratezza e affidabilita")
    scores = overall_scores(summary)
    cards = st.columns(max(len(scores), 1))
    for column, (_, score) in zip(cards, scores.iterrows(), strict=False):
        column.metric(
            score["configuration"],
            f"{score['median_nrmse_iqr']:.2f} RMSE/IQR",
            f"r mediano {score['median_correlation']:.2f}",
            help="Valori RMSE/IQR piu bassi e correlazioni piu alte sono preferibili.",
        )
        column.caption(f"{score['usable_share']:.0%} output classificati utilizzabili o forti")

    display = display_slice_name(summary)
    figure = px.bar(
        display,
        x="trait_variable",
        y=metric,
        color="configuration",
        barmode="group",
        category_orders={"configuration": list(CONFIG_LABELS.values())},
        hover_data={"RMSE": ":.2f", "MAE": ":.2f", "NRMSE_IQR": ":.2f", "Pearson_r": ":.2f", "n": True, "trait_variable": False},
        labels={metric: selected_metric, "trait_variable": "Trait e variabile", "configuration": "Configurazione"},
        title=f"{selected_metric} su CV leave-one-trait-out",
    )
    figure.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=55, b=10), height=460)
    st.plotly_chart(figure, width="stretch")

    heatmap_source = display.pivot(index="configuration", columns="trait_variable", values=metric)
    heatmap_source = heatmap_source.reindex(index=[label for label in CONFIG_LABELS.values() if label in heatmap_source.index])
    heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_source.to_numpy(),
            x=heatmap_source.columns.tolist(),
            y=heatmap_source.index.tolist(),
            colorscale="RdYlGn_r" if lower_is_better else "RdYlGn",
            colorbar_title=selected_metric,
            hovertemplate="Configurazione: %{y}<br>Trait: %{x}<br>Valore: %{z:.3f}<extra></extra>",
        )
    )
    heatmap.update_layout(title="Matrice comparativa", margin=dict(l=10, r=10, t=55, b=10), height=330)
    st.plotly_chart(heatmap, width="stretch")

    st.dataframe(
        reliability_table(summary),
        width="stretch",
        hide_index=True,
        column_config={
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "NRMSE_IQR": st.column_config.NumberColumn("RMSE / IQR", format="%.2f"),
            "NRMSE_range": st.column_config.NumberColumn("RMSE / range", format="%.2f"),
            "Pearson_r": st.column_config.NumberColumn("Pearson r", format="%.2f"),
            "Spearman_rho": st.column_config.NumberColumn("Spearman rho", format="%.2f"),
        },
    )
    st.caption(
        "RMSE e MAE sono aggregati pesando per il numero di osservazioni. Le correlazioni sono medie Fisher-z pesate. "
        "RMSE/IQR usa l'IQR delle osservazioni originali e permette confronti tra scale diverse."
    )


def render_ablation(summary: pd.DataFrame, selected_metric: str) -> None:
    metric, lower_is_better = METRIC_OPTIONS[selected_metric]
    deltas = ablation_deltas(summary, metric, lower_is_better)
    st.subheader("Effetto marginale degli input accessori")
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

    display = selected[["feature", "input_group", "importance_share", "mean_abs_ig", "mean_signed_ig", "positive_fraction", "samples"]].copy()
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "importance_share": st.column_config.NumberColumn("Quota |IG|", format="%.2f"),
            "mean_abs_ig": st.column_config.NumberColumn("Media |IG|", format="%.5f"),
            "mean_signed_ig": st.column_config.NumberColumn("IG medio firmato", format="%.5f"),
            "positive_fraction": st.column_config.NumberColumn("IG positivi", format="%.0f%%"),
        },
    )

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
    manifest = metric_rows.groupby(["configuration_id", "configuration", "experiment_dir"], as_index=False).agg(
        folds=("fold", "nunique"),
        metric_rows=("trait", "size"),
        evaluations=("n", "sum"),
    )
    st.dataframe(sort_configurations(manifest), width="stretch", hide_index=True)

    weak = summary.loc[summary["reliability"] == "Debole / non interpretabile", ["configuration", "trait", "variable", "Pearson_r", "NRMSE_IQR"]]
    if not weak.empty:
        st.warning(
            f"{len(weak)} delle {len(summary)} slice configurazione-trait-variabile superano le soglie di affidabilita. "
            "Le relative attribution devono essere lette solo come diagnostica del modello."
        )

    if environmental_data.get("replicated"):
        st.error(
            "Le attribution ambientali non sono pienamente interpretabili come layer distinti: gli output contengono tre blocchi uguali per numero "
            "di feature rispetto alla tabella cache disponibile. Il dashboard li aggrega per nome sorgente; non attribuisce contributi a suolo, densita o elevazione."
        )
    else:
        st.info("Non e stata rilevata una ripetizione strutturale delle colonne ambientali nell'output disponibile.")

    export = reliability_table(summary).to_csv(index=False).encode("utf-8")
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
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Audit delle imputazioni dei tratti")
    st.caption("Cross-validation leave-one-trait-out, metriche in scala originale e attribution Integrated Gradients")

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

    summary = classify_reliability(aggregate_cv_metrics(metric_rows), correlation_floor, relative_rmse_limit)
    species_attributions = cached_attributions(str(results_path), "species", ATTRIBUTION_CACHE_SCHEMA_VERSION)
    spatial_attributions = cached_attributions(str(results_path), "spatial", ATTRIBUTION_CACHE_SCHEMA_VERSION)
    environmental_data = cached_environment_metadata(str(results_path.parent), spatial_attributions)
    species_summary = summarize_attributions(species_attributions, "species")
    spatial_summary = summarize_attributions(spatial_attributions, "spatial", environmental_data)

    overview_tab, ablation_tab, attribution_tab, quality_tab = st.tabs(["Accuratezza", "Ablation", "Attribution", "Qualita dati"])
    with overview_tab:
        render_overview(summary, selected_metric)
    with ablation_tab:
        render_ablation(summary, selected_metric)
    with attribution_tab:
        render_attributions(species_summary, spatial_summary, summary, environmental_data)
    with quality_tab:
        render_data_quality(metric_rows, summary, environmental_data)


if __name__ == "__main__":
    main()