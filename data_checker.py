"""
Run one benchmark seed with the current preprocessing pipeline and compare it
against the same seed from the previous benchmark_results/.

The new run is always written to a separate, timestamped directory so that
existing benchmark results cannot be overwritten.

Example
-------
python compare_reprocessed_seed.py --seed 42

Optionally:
python compare_reprocessed_seed.py \
    --seed 42 \
    --old-results benchmark_results \
    --output-root benchmark_reprocessing_checks

The script produces:
    <output-root>/seed_<seed>_<timestamp>/
        benchmark/                  # complete new run from run_benchmark.py
        comparison_per_trait.csv    # old vs new seed-level metrics
        comparison_summary.csv      # aggregate model/configuration comparison
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Benchmark layout
# ---------------------------------------------------------------------------

COMBINATION_MAP = {
    "no_env_no_phylo": {
        "configuration_id": "baseline",
        "has_environment": False,
        "has_phylogeny": False,
    },
    "env_only": {
        "configuration_id": "environment",
        "has_environment": True,
        "has_phylogeny": False,
    },
    "phylo_only": {
        "configuration_id": "phylogeny",
        "has_environment": False,
        "has_phylogeny": True,
    },
    "env_phylo": {
        "configuration_id": "full",
        "has_environment": True,
        "has_phylogeny": True,
    },
}

CONFIGURATION_LABELS = {
    "baseline": "No environment / no phylogeny",
    "environment": "Environment only",
    "phylogeny": "Phylogeny only",
    "full": "Environment + phylogeny",
}

REQUIRED_METRIC_COLUMNS = {
    "trait",
    "variable",
    "n",
    "RMSE",
    "MAE",
    "Pearson_r",
    "Spearman_rho",
}

METRICS = (
    "RMSE",
    "MAE",
    "Pearson_r",
    "Spearman_rho",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def run_command(command: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(str(x) for x in command))
    print()

    subprocess.run(command, check=True)


def weighted_fisher_mean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean for correlations, matching the benchmark analysis aggregation."""
    valid = values.notna() & weights.notna() & (weights > 0)

    if not valid.any():
        return float("nan")

    clipped = values.loc[valid].clip(-0.999999, 0.999999)

    correlation_weights = np.maximum(
        weights.loc[valid].to_numpy(dtype=float) - 3.0,
        1.0,
    )

    return float(
        np.tanh(
            np.average(
                np.arctanh(clipped.to_numpy(dtype=float)),
                weights=correlation_weights,
            )
        )
    )


def aggregate_metric_group(group: pd.DataFrame) -> dict[str, float]:
    """
    Collapse fold-level metrics.

    This follows the shared benchmark aggregation:
      * RMSE: sqrt(weighted mean of squared fold RMSE)
      * MAE: weighted mean
      * correlations: weighted Fisher-z mean
    """
    weights = group["n"].fillna(0).clip(lower=0)

    total_n = float(weights.sum())

    valid_rmse = group["RMSE"].notna() & (weights > 0)
    valid_mae = group["MAE"].notna() & (weights > 0)

    if valid_rmse.any():
        rmse = float(
            np.sqrt(
                np.average(
                    np.square(group.loc[valid_rmse, "RMSE"]),
                    weights=weights.loc[valid_rmse],
                )
            )
        )
    else:
        rmse = float("nan")

    if valid_mae.any():
        mae = float(
            np.average(
                group.loc[valid_mae, "MAE"],
                weights=weights.loc[valid_mae],
            )
        )
    else:
        mae = float("nan")

    return {
        "n": total_n,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson_r": weighted_fisher_mean(
            group["Pearson_r"],
            weights,
        ),
        "Spearman_rho": weighted_fisher_mean(
            group["Spearman_rho"],
            weights,
        ),
    }


# ---------------------------------------------------------------------------
# Reading run_benchmark.py outputs
# ---------------------------------------------------------------------------

def read_metric_file(
    path: Path,
    *,
    seed: int,
    model: str,
    source: str,
    combination_name: str,
) -> pd.DataFrame | None:

    frame = pd.read_csv(path)

    if not REQUIRED_METRIC_COLUMNS.issubset(frame.columns):
        return None

    frame = frame.copy()

    metadata = COMBINATION_MAP[combination_name]

    frame["seed"] = seed
    frame["fold"] = path.parent.name
    frame["model"] = model
    frame["source"] = source
    frame["combination"] = combination_name
    frame["configuration_id"] = metadata["configuration_id"]
    frame["configuration"] = CONFIGURATION_LABELS[
        metadata["configuration_id"]
    ]
    frame["has_environment"] = metadata["has_environment"]
    frame["has_phylogeny"] = metadata["has_phylogeny"]

    return frame


def discover_seed_metrics(
    results_dir: Path,
    seed: int,
) -> pd.DataFrame:
    """
    Read fold-level metric outputs generated by run_benchmark.py for one seed.
    """
    seed_dir = results_dir / "runs" / f"seed_{seed}"

    if not seed_dir.is_dir():
        raise FileNotFoundError(
            f"No benchmark results found for seed {seed} at:\n"
            f"  {seed_dir}"
        )

    frames: list[pd.DataFrame] = []

    # ---------------------------------------------------------
    # GNNs
    # ---------------------------------------------------------
    gnn_root = seed_dir / "gnn"

    for combination_name in COMBINATION_MAP:
        combination_dir = gnn_root / combination_name

        if not combination_dir.is_dir():
            continue

        experiment_dirs = [
            p for p in combination_dir.iterdir()
            if p.is_dir()
        ]

        for experiment_dir in experiment_dirs:
            for metric_file in sorted(
                experiment_dir.glob(
                    "fold_*/per_trait_metrics_original.csv"
                )
            ):
                frame = read_metric_file(
                    metric_file,
                    seed=seed,
                    model="gnn",
                    source="gnn",
                    combination_name=combination_name,
                )

                if frame is not None:
                    frames.append(frame)

    # ---------------------------------------------------------
    # Baselines
    #
    # run_benchmark.py computes these once per seed and reuses
    # them for each feature combination. Replicate them across
    # configurations, matching the benchmark analysis logic.
    # ---------------------------------------------------------
    baseline_root = seed_dir / "baselines"

    if baseline_root.is_dir():

        for model_dir in sorted(
            p for p in baseline_root.iterdir()
            if p.is_dir()
        ):
            model_name = model_dir.name

            for metric_file in sorted(
                model_dir.glob("fold_*/per_trait_metrics.csv")
            ):
                for combination_name in COMBINATION_MAP:
                    frame = read_metric_file(
                        metric_file,
                        seed=seed,
                        model=model_name,
                        source="baseline",
                        combination_name=combination_name,
                    )

                    if frame is not None:
                        frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"No metric files were found for seed {seed} under "
            f"{seed_dir}"
        )

    result = pd.concat(frames, ignore_index=True)

    for column in (
        "n",
        "RMSE",
        "MAE",
        "Pearson_r",
        "Spearman_rho",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


# ---------------------------------------------------------------------------
# Fold aggregation
# ---------------------------------------------------------------------------

def aggregate_seed_metrics(
    metric_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one row per:
        model x configuration x trait x variable

    with folds aggregated using the benchmark analysis convention.
    """
    group_columns = [
        "configuration_id",
        "configuration",
        "combination",
        "has_environment",
        "has_phylogeny",
        "model",
        "source",
        "seed",
        "trait",
        "variable",
    ]

    records = []

    for key, group in metric_rows.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        metrics = aggregate_metric_group(group)

        if metrics["n"] <= 0:
            continue

        record = dict(zip(group_columns, key))
        record.update(metrics)
        record["folds"] = group["fold"].nunique()

        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Old vs new comparison
# ---------------------------------------------------------------------------

def compare_seed_results(
    old: pd.DataFrame,
    new: pd.DataFrame,
) -> pd.DataFrame:

    keys = [
        "configuration_id",
        "configuration",
        "combination",
        "has_environment",
        "has_phylogeny",
        "model",
        "trait",
        "variable",
    ]

    value_columns = [
        "n",
        "folds",
        *METRICS,
    ]

    old_subset = old[keys + value_columns].copy()
    new_subset = new[keys + value_columns].copy()

    comparison = old_subset.merge(
        new_subset,
        on=keys,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )

    for metric in METRICS:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_new"]
            - comparison[f"{metric}_old"]
        )

    # Relative error changes are useful for RMSE and MAE.
    for metric in ("RMSE", "MAE"):
        comparison[f"{metric}_pct_change"] = (
            100
            * comparison[f"{metric}_delta"]
            / comparison[f"{metric}_old"]
        )

    # Negative = improvement for error metrics.
    comparison["RMSE_improved"] = (
        comparison["RMSE_delta"] < 0
    )

    # Positive = improvement for correlations.
    comparison["Pearson_improved"] = (
        comparison["Pearson_r_delta"] > 0
    )

    return comparison


# ---------------------------------------------------------------------------
# Overall summary across traits
# ---------------------------------------------------------------------------

def aggregate_across_traits(
    seed_level: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce a compact model/configuration summary across all traits.

    RMSE is pooled as a weighted RMS.
    MAE is a weighted mean.
    Correlations use weighted Fisher-z means.
    """
    group_columns = [
        "configuration_id",
        "configuration",
        "combination",
        "has_environment",
        "has_phylogeny",
        "model",
    ]

    records = []

    for key, group in seed_level.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        metrics = aggregate_metric_group(group)

        record = dict(zip(group_columns, key))
        record.update(metrics)
        record["n_trait_variables"] = len(group)

        records.append(record)

    return pd.DataFrame(records)


def compare_summaries(
    old: pd.DataFrame,
    new: pd.DataFrame,
) -> pd.DataFrame:

    keys = [
        "configuration_id",
        "configuration",
        "combination",
        "has_environment",
        "has_phylogeny",
        "model",
    ]

    comparison = old.merge(
        new,
        on=keys,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )

    for metric in METRICS:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_new"]
            - comparison[f"{metric}_old"]
        )

    for metric in ("RMSE", "MAE"):
        comparison[f"{metric}_pct_change"] = (
            100
            * comparison[f"{metric}_delta"]
            / comparison[f"{metric}_old"]
        )

    return comparison


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_gnn_summary(summary: pd.DataFrame) -> None:
    gnn = summary[
        (summary["model"] == "gnn")
        & (summary["_merge"] == "both")
    ].copy()

    if gnn.empty:
        print("\nNo matched GNN rows found.")
        return

    order = {
        "baseline": 0,
        "phylogeny": 1,
        "environment": 2,
        "full": 3,
    }

    gnn["_order"] = (
        gnn["configuration_id"]
        .map(order)
        .fillna(99)
    )

    gnn = gnn.sort_values("_order")

    columns = [
        "configuration",
        "RMSE_old",
        "RMSE_new",
        "RMSE_delta",
        "RMSE_pct_change",
        "MAE_old",
        "MAE_new",
        "Pearson_r_old",
        "Pearson_r_new",
        "Pearson_r_delta",
    ]

    print("\n" + "=" * 110)
    print("GNN: OLD vs REPROCESSED DATA")
    print("=" * 110)

    print(
        gnn[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\nInterpretation:")
    print("  RMSE_delta < 0       -> new preprocessing improved RMSE")
    print("  RMSE_pct_change < 0  -> new preprocessing improved RMSE")
    print("  Pearson_r_delta > 0  -> new preprocessing improved correlation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one seed with the current preprocessing and compare it "
            "against the same seed in benchmark_results/."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed to rerun and compare.",
    )

    parser.add_argument(
        "--old-results",
        type=Path,
        default=Path("benchmark_results"),
        help="Existing benchmark generated by run_benchmark.py.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark_reprocessing_checks"),
        help=(
            "Parent directory for new isolated runs. "
            "Nothing in benchmark_results/ is modified."
        ),
    )

    parser.add_argument(
        "--gnn-args",
        default="",
        help=(
            "Optional arguments forwarded to run_benchmark.py's "
            "--gnn-args option."
        ),
    )

    parser.add_argument(
        "--baseline-args",
        default="",
        help=(
            "Optional arguments forwarded to run_benchmark.py's "
            "--baseline-args option."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent

    old_results = (
        args.old_results
        if args.old_results.is_absolute()
        else project_root / args.old_results
    ).resolve()

    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else project_root / args.output_root
    ).resolve()

    benchmark_script = project_root / "run_benchmark.py"

    if not benchmark_script.exists():
        raise FileNotFoundError(
            f"run_benchmark.py not found at {benchmark_script}"
        )

    if not old_results.exists():
        raise FileNotFoundError(
            f"Previous benchmark directory not found: {old_results}"
        )

    # ---------------------------------------------------------
    # Check that the requested seed exists BEFORE spending time
    # retraining anything.
    # ---------------------------------------------------------
    old_seed_dir = (
        old_results
        / "runs"
        / f"seed_{args.seed}"
    )

    if not old_seed_dir.is_dir():
        raise FileNotFoundError(
            f"Seed {args.seed} is not present in the old benchmark:\n"
            f"  {old_seed_dir}"
        )

    # ---------------------------------------------------------
    # Unique output directory
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_root = (
        output_root
        / f"seed_{args.seed}_{timestamp}"
    )

    new_benchmark_dir = run_root / "benchmark"

    # Do not silently reuse anything.
    if run_root.exists():
        raise FileExistsError(
            f"Output directory already exists:\n  {run_root}"
        )

    run_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 80)
    print("REPROCESSED DATA BENCHMARK CHECK")
    print("=" * 80)

    print(f"Seed             : {args.seed}")
    print(f"Previous results : {old_results}")
    print(f"New run          : {new_benchmark_dir}")

    # ---------------------------------------------------------
    # Run exactly one seed through the same benchmark pipeline.
    # ---------------------------------------------------------
    command = [
        sys.executable,
        str(benchmark_script),
        "1",
        "--start-seed",
        str(args.seed),
        "--output-dir",
        str(new_benchmark_dir),
    ]

    if args.gnn_args:
        command += [
            "--gnn-args",
            args.gnn_args,
        ]

    if args.baseline_args:
        command += [
            "--baseline-args",
            args.baseline_args,
        ]

    run_command(command)

    # ---------------------------------------------------------
    # Read old and new metrics
    # ---------------------------------------------------------
    print("\nReading previous benchmark metrics...")

    old_fold_metrics = discover_seed_metrics(
        old_results,
        args.seed,
    )

    print("Reading new benchmark metrics...")

    new_fold_metrics = discover_seed_metrics(
        new_benchmark_dir,
        args.seed,
    )

    # ---------------------------------------------------------
    # Aggregate folds
    # ---------------------------------------------------------
    old_seed = aggregate_seed_metrics(
        old_fold_metrics
    )

    new_seed = aggregate_seed_metrics(
        new_fold_metrics
    )

    comparison = compare_seed_results(
        old_seed,
        new_seed,
    )

    per_trait_path = (
        run_root
        / "comparison_per_trait.csv"
    )

    comparison.to_csv(
        per_trait_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Aggregate across traits for concise comparison
    # ---------------------------------------------------------
    old_summary = aggregate_across_traits(
        old_seed
    )

    new_summary = aggregate_across_traits(
        new_seed
    )

    summary = compare_summaries(
        old_summary,
        new_summary,
    )

    summary_path = (
        run_root
        / "comparison_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Console report
    # ---------------------------------------------------------
    print_gnn_summary(summary)

    unmatched = comparison[
        comparison["_merge"] != "both"
    ]

    if not unmatched.empty:
        print(
            "\nWARNING: some old/new metric rows could not be matched."
        )
        print(
            unmatched[
                [
                    "configuration",
                    "model",
                    "trait",
                    "variable",
                    "_merge",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 80)
    print("OUTPUTS")
    print("=" * 80)

    print(f"New benchmark run:")
    print(f"  {new_benchmark_dir}")

    print(f"\nPer-trait comparison:")
    print(f"  {per_trait_path}")

    print(f"\nOverall comparison:")
    print(f"  {summary_path}")

    print(
        "\nThe original benchmark_results/ directory was only read; "
        "it was never passed to run_benchmark.py as an output directory."
    )


if __name__ == "__main__":
    main()
