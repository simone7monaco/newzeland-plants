"""Benchmark all GNN feature combinations and R baselines over repeated seeds."""

from __future__ import annotations

import argparse
import logging
import hashlib
import json
import importlib.metadata
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


FEATURE_COMBINATIONS = (
    (True, True, "env_phylo"),
    (True, False, "env_only"),
    (False, True, "phylo_only"),
    (False, False, "no_env_no_phylo"),
)
COMBINATION_MAP = {name: (use_env, use_phylo) for use_env, use_phylo, name in FEATURE_COMBINATIONS}
ATTRIBUTION_FILES = ("attributions_species_all.parquet", "attributions_spatial_all.parquet")
DETERMINISTIC_BASELINES = (
    "training_mean",
    "training_median",
    "phylo_nn",
    "phylo_knn",
    "rphylopars_bm",
)
STOCHASTIC_BASELINES = ("mice", "random_forest", "xgboost", "mlp")
BASELINES = DETERMINISTIC_BASELINES + STOCHASTIC_BASELINES
COVARIATE_BASELINES = STOCHASTIC_BASELINES
LONG_RESULT_FILE = "predictions_long_all.parquet"
TREE_COVERAGE_FILE = "ftol_mapping.csv"
R_LIBRARY = Path(".local/r-library")
# R packages used by the tree reconstruction (loader.py/ftol_tree.R) and the R baselines.
R_PACKAGES = ("mice", "rpart", "missForest", "Rphylopars", "ape", "phytools", "ftolr", "taxastand", "pteridocat")
LOGGER = logging.getLogger("benchmark")


def r_package_versions() -> dict[str, str]:
    """R, package and FTOL data versions as resolved by the r-baselines environment.

    The project library is put first on the search path, as loader.py does, so
    the recorded versions are the ones actually loaded by the benchmark.
    """
    packages = ", ".join(f"'{name}'" for name in R_PACKAGES)
    script = (
        f"lib <- '{R_LIBRARY}'; if (dir.exists(lib)) .libPaths(c(normalizePath(lib), .libPaths())); "
        "cat('R', as.character(getRversion()), '\\n'); "
        f"for (p in c({packages})) cat(p, tryCatch(as.character(packageVersion(p)), error=function(e) 'MISSING'), '\\n'); "
        "cat('ftol_data', tryCatch(as.character(ftolr::ft_data_ver()), error=function(e) 'MISSING'), '\\n')"
    )
    output = subprocess.run(
        ["micromamba", "run", "-n", "r-baselines", "Rscript", "-e", script],
        check=True, capture_output=True, text=True,
    ).stdout
    versions = dict(line.split(maxsplit=1) for line in output.splitlines() if line.strip())
    versions = {name: version.strip() for name, version in versions.items()}
    missing = sorted(name for name, version in versions.items() if version == "MISSING")
    if missing:
        raise SystemExit(f"R packages required by the benchmark are not installed: {missing}")
    return versions


def r_library_fingerprints() -> str:
    """Hash every package DESCRIPTION in the project R library (name, version, build)."""
    lines = []
    if R_LIBRARY.is_dir():
        for package_dir in sorted(path for path in R_LIBRARY.iterdir() if path.is_dir()):
            description = package_dir / "DESCRIPTION"
            if not description.is_file():
                raise SystemExit(f"{package_dir} is not an installed R package (no DESCRIPTION).")
            lines.append(f"{package_dir} {hashlib.sha256(description.read_bytes()).hexdigest()}\n")
    return "".join(lines)


def check_original_scale(result_dir: Path) -> None:
    """Every model directory must export original-scale long predictions and its tree coverage."""
    long_path = result_dir / LONG_RESULT_FILE
    if not long_path.is_file():
        raise FileNotFoundError(f"{long_path} is missing; rerun with the current code.")
    long = pd.read_parquet(long_path, columns=["scale", "evaluated", "predicted"])
    scales = set(long["scale"].unique())
    if scales != {"original"}:
        raise ValueError(f"{long_path} has prediction scale {sorted(scales)}; expected only 'original'.")
    # Also catches resumed runs written before per-fold finiteness checks existed.
    bad = long["evaluated"].astype(bool) & ~np.isfinite(long["predicted"].to_numpy(dtype=float))
    if bad.any():
        raise ValueError(f"{long_path} has {int(bad.sum())} evaluated cells with non-finite predictions.")


def r_environment_command(command: list[str]) -> list[str]:
    """Both graph loading and R baselines need the README's R runtime."""
    script = (
        'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
        f"exec {shlex.join(command)}"
    )
    return ["micromamba", "run", "-n", "r-baselines", "bash", "-c", script]


class TqdmLoggingHandler(logging.Handler):
    """Write log records without permanently disrupting active progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_seeds", type=int, help="Number of consecutive seeds to run")
    parser.add_argument("--start-seed", type=int, default=42, help="First seed (default: 42)")
    parser.add_argument("--parallel-seeds", type=int, default=1, help="Number of complete seed suites to run concurrently (default: 1)")
    parser.add_argument("--occlusion-protocol", choices=("paired", "individual"), default="paired", help="Shared target masking protocol for all models")
    parser.add_argument("--include-unobserved-species", action="store_true", help="Retain species without observed target traits")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results"),
        help="Benchmark run and aggregate output directory",
    )
    parser.add_argument(
        "--result-file", default="predictions_max_all.parquet",
        help="Per-model merged file to aggregate (default: predictions_max_all.parquet)",
    )
    parser.add_argument('--split-strategy', choices=['random', 'stable', 'louvain', 'genus', 'clade'], default='random')
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
        help="Terminal logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def run(command: list[str], *, label: str) -> None:
    LOGGER.info("Starting %s", label)
    LOGGER.debug("Command: %s", shlex.join(command))
    started_at = time.monotonic()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        LOGGER.exception("%s failed after %.1f minutes", label, (time.monotonic() - started_at) / 60)
        raise
    LOGGER.info("Finished %s in %.1f minutes", label, (time.monotonic() - started_at) / 60)


def read_csv(
    path: Path,
    *,
    seed: int,
    model: str,
    source: str,
    use_env_features: bool | None = None,
    use_phylo_features: bool | None = None,
) -> pd.DataFrame:
    # Support both Parquet and CSV for backward compatibility
    if path.suffix.lower() in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if "species" not in frame.columns and frame.index.name == "species":
        # Baseline parquet files store the species as a named index, which the
        # ignore_index concatenation below would otherwise discard.
        frame = frame.reset_index()
    unnamed_columns = [column for column in frame.columns if column.startswith("Unnamed:")]
    if "species" not in frame.columns and unnamed_columns:
        frame = frame.rename(columns={unnamed_columns[0]: "species"})
    frame.insert(0, "source", source)
    frame.insert(0, "model", model)
    if use_phylo_features is not None:
        frame.insert(0, "use_phylo_features", use_phylo_features)
    if use_env_features is not None:
        frame.insert(0, "use_env_features", use_env_features)
    frame.insert(0, "seed", seed)
    return frame


def find_baseline_results(
    baseline_dir: Path,
    models: tuple[str, ...],
    filename: str,
    use_env: bool = False,
    use_phylo: bool = False,
) -> dict[str, Path]:
    """Return the result for each requested model that has completed in this seed."""
    results: dict[str, Path] = {}
    for model in models:
        model_prefix = f"{model.upper()}_"
        matches = sorted(
            path
            for path in baseline_dir.glob(f"*/{filename}")
            if path.parent.name == model.upper() or path.parent.name.startswith(model_prefix)
        )
        expected = (use_env, use_phylo) if model in COVARIATE_BASELINES else (False, False)
        matches = [path for path in matches if (
            'env' in path.parent.name[len(model):].lower().split('_'),
            'phylo' in path.parent.name[len(model):].lower().split('_'),
        ) == expected]
        if len(matches) > 1:
            raise RuntimeError(
                f"Found multiple {filename} files for baseline '{model}' below {baseline_dir}: "
                f"{matches}"
            )
        if matches:
            results[model] = matches[0]
    return results


def log_bounds_order(model_dir: Path, label: str) -> None:
    """Log how many predicted (min, max) pairs were reversed before projection."""
    path = model_dir / "bounds_order_all_folds.csv"
    if not path.is_file():
        LOGGER.warning("%s: no bounds_order_all_folds.csv found in %s", label, model_dir)
        return
    table = pd.read_csv(path)
    total, reversed_total = int(table.n_pairs.sum()), int(table.n_reversed.sum())
    detail = ""
    if reversed_total:
        worst = table[table.n_reversed > 0]
        detail = " (" + ", ".join(f"{r.trait}: {int(r.n_reversed)}" for r in worst.itertuples()) + ")"
    LOGGER.info("%s: %d/%d test pairs had min > max before projection (%.2f%%)%s%s",
                label, reversed_total, total, 100 * reversed_total / max(total, 1), detail,
                "" if bool(table.projected.all()) else " [NOT projected]")


def collect_bounds_order(seed: int, combination: str, model: str, model_dir: Path) -> pd.DataFrame | None:
    path = model_dir / "bounds_order_all_folds.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    table.insert(0, "model", model)
    table.insert(0, "combination", combination)
    table.insert(0, "seed", seed)
    return table


def run_seed(seed: int, args: argparse.Namespace, shared_args: list[str], progress, progress_lock) -> None:
    """Run and validate every baseline and GNN configuration for one seed."""
    seed_dir = args._runs_dir / f"seed_{seed}"
    baseline_dir = seed_dir / "baselines"

    def update_progress(label: str) -> None:
        with progress_lock:
            progress.set_postfix_str(label, refresh=True)
            progress.update()

    for use_env, use_phylo, combination_name in FEATURE_COMBINATIONS:
        models = BASELINES if not (use_env or use_phylo) else COVARIATE_BASELINES
        baseline_results = find_baseline_results(
            baseline_dir, models, args.result_file, use_env, use_phylo,
        )
        missing_baselines = tuple(model for model in models if model not in baseline_results)
        if missing_baselines:
            baseline_command = [
                "uv", "run", "--frozen", "python", "r_baselines_fit.py", *shared_args,
                "--baseline_models", ",".join(missing_baselines),
                "--use_env_features", str(use_env).lower(),
                "--use_phylo_features", str(use_phylo).lower(),
                "--occlusion_protocol", args.occlusion_protocol,
                "--include_unobserved_species", str(args.include_unobserved_species).lower(),
                "--k", "-1", "--seed", str(seed), "--output_dir", str(baseline_dir),
            ]
            if args._fixed_file is not None:
                baseline_command.extend(['--fixed_hyperparameters', str(args._fixed_file)])
            run(r_environment_command(baseline_command),
                label=f"Baselines {combination_name}: {', '.join(missing_baselines)} (seed {seed})")
            baseline_results = find_baseline_results(
                baseline_dir, models, args.result_file, use_env, use_phylo,
            )
        still_missing = tuple(model for model in models if model not in baseline_results)
        if still_missing:
            raise FileNotFoundError(f"Missing baselines {still_missing} for {combination_name} below {baseline_dir}")
        if not (baseline_dir / TREE_COVERAGE_FILE).is_file():
            raise FileNotFoundError(baseline_dir / TREE_COVERAGE_FILE)
        for result in baseline_results.values():
            for filename in ('predictions_min_all.parquet', 'predictions_max_all.parquet'):
                if not result.with_name(filename).is_file():
                    raise FileNotFoundError(result.with_name(filename))
            check_original_scale(result.parent)
            log_bounds_order(result.parent, f"seed {seed} baseline {result.parent.name}")
        update_progress(f"seed={seed} baselines {combination_name} complete")

    for use_env, use_phylo, combination_name in FEATURE_COMBINATIONS:
        gnn_dir = seed_dir / "gnn" / combination_name
        gnn_results = sorted(gnn_dir.glob(f"*/{args.result_file}"))
        if not gnn_results:
            run(r_environment_command([
                sys.executable, "train.py", *shared_args,
                "--use_env_features", str(use_env).lower(),
                "--use_phylo_features", str(use_phylo).lower(),
                "--occlusion_protocol", args.occlusion_protocol,
                "--include_unobserved_species", str(args.include_unobserved_species).lower(),
                "--compute_xai", "true", "--k", "-1", "--seed", str(seed),
                "--output_dir", str(gnn_dir),
            ]), label=f"GNN {combination_name} (seed {seed})")
            gnn_results = sorted(gnn_dir.glob(f"*/{args.result_file}"))
        if len(gnn_results) != 1:
            raise FileNotFoundError(
                f"Expected one GNN {args.result_file} below {gnn_dir}, found {len(gnn_results)}"
            )
        artifact_dir = gnn_results[0].parent
        for filename in (*ATTRIBUTION_FILES, "predictions_min_all.parquet", "predictions_max_all.parquet",
                         "predictions_original_all_folds.csv", TREE_COVERAGE_FILE):
            if filename == "attributions_spatial_all.parquet" and not use_env:
                continue
            if not (artifact_dir / filename).is_file():
                raise FileNotFoundError(f"GNN output file was not produced: {artifact_dir / filename}")
        check_original_scale(artifact_dir)
        log_bounds_order(artifact_dir, f"seed {seed} GNN {combination_name}")
        update_progress(f"seed={seed} {combination_name} complete")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.n_seeds < 1:
        raise SystemExit("n_seeds must be at least 1")
    if args.parallel_seeds < 1:
        raise SystemExit("--parallel-seeds must be at least 1")

    output_dir = args.output_dir.resolve()
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_args = ['--split_strategy', args.split_strategy]
    fixed_file = output_dir / 'hyperparameters.yaml'
    args._fixed_file = fixed_file if fixed_file.is_file() else None
    source_files = ['run_benchmark.py', 'train.py', 'r_baselines_fit.py', 'loader.py', 'models.py', 'tester.py', 'occlusion.py', 'r_models.py', 'ml_baselines.py', 'prediction_bounds.py']
    manifest = {'schema': 3, 'merged_prediction_scale': 'original', 'start_seed': args.start_seed, 'n_seeds': args.n_seeds,
                'fixed_hyperparameters_sha256': hashlib.sha256(fixed_file.read_bytes()).hexdigest() if args._fixed_file else None,
                'r_versions': r_package_versions(),
                'result_file': args.result_file, 'split_strategy': args.split_strategy, 'occlusion': args.occlusion_protocol,
                'include_unobserved_species': args.include_unobserved_species,
                'feature_combinations': FEATURE_COMBINATIONS, 'baselines': BASELINES,
                'python': sys.version, 'code_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
                'python_packages': sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions() if d.metadata.get('Name'))}
    manifest_path = output_dir / 'manifest.json'
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    if manifest_path.exists() and manifest_path.read_text() != manifest_text:
        raise SystemExit('Benchmark manifest differs; choose a fresh output directory.')
    if not manifest_path.exists() and runs_dir.exists() and any(runs_dir.iterdir()):
        raise SystemExit('Existing runs lack a complete manifest; choose a fresh output directory.')
    manifest_path.write_text(manifest_text)
    marker = output_dir / 'occlusion_protocol.txt'
    if marker.exists() and marker.read_text().strip() != args.occlusion_protocol:
        raise SystemExit('The output directory contains a different occlusion protocol; choose a fresh directory.')
    if not marker.exists() and runs_dir.exists() and any(runs_dir.iterdir()):
        raise SystemExit('Existing benchmark runs have no occlusion protocol marker; choose a fresh output directory.')
    marker.write_text(args.occlusion_protocol + '\n')
    dataset_marker = output_dir / 'dataset_fingerprints.txt'
    inputs = [Path('data/Ferns/FernTraits.xlsx'), Path('ftol_tree.R')]
    fingerprints = ''.join(f'{path} {hashlib.sha256(path.read_bytes()).hexdigest()}\n' for path in inputs)
    fingerprints += r_library_fingerprints()
    fingerprints += f'include_unobserved_species {str(args.include_unobserved_species).lower()}\n'
    if dataset_marker.exists() and dataset_marker.read_text() != fingerprints:
        raise SystemExit('The workbook or phylogenetic tree differs from existing runs; choose a fresh output directory.')
    if not dataset_marker.exists() and runs_dir.exists() and any(runs_dir.iterdir()):
        raise SystemExit('Existing benchmark runs have no dataset fingerprints; choose a fresh output directory.')
    dataset_marker.write_text(fingerprints)
    seeds = range(args.start_seed, args.start_seed + args.n_seeds)
    args._runs_dir = runs_dir
    total_jobs = args.n_seeds * (2 * len(FEATURE_COMBINATIONS))
    LOGGER.info(
        "Benchmark started: %d seeds, %d jobs, up to %d seeds in parallel, output=%s",
        args.n_seeds, total_jobs, args.parallel_seeds, output_dir,
    )

    progress_lock = threading.Lock()
    with tqdm(total=total_jobs, desc="Benchmark", unit="job", dynamic_ncols=True) as progress:
        with ThreadPoolExecutor(max_workers=args.parallel_seeds) as executor:
            futures = {executor.submit(run_seed, seed, args, shared_args, progress, progress_lock): seed for seed in seeds}
            for future in as_completed(futures):
                future.result()

    LOGGER.info("All requested seed suites completed and validated; aggregating all their contributions")

    # Result files to collect: include both min and max by default plus any explicit result-file
    default_result_files = ["predictions_min_all.parquet", "predictions_max_all.parquet", LONG_RESULT_FILE]
    result_files = list(dict.fromkeys([args.result_file] + default_result_files))

    # Prepare aggregation containers per result file and combination
    aggregated_predictions: dict[str, dict[str, list[pd.DataFrame]]] = {
        filename: {name: [] for _, _, name in FEATURE_COMBINATIONS}
        for filename in result_files
    }
    aggregated_attributions: dict[str, dict[str, list[pd.DataFrame]]] = {
        name: {filename: [] for filename in ATTRIBUTION_FILES}
        for _, _, name in FEATURE_COMBINATIONS
    }

    # Per-seed results: detect all seed_* directories and collect per-file
    for seed in seeds:
        seed_dir = runs_dir / f"seed_{seed}"

        baseline_dir = seed_dir / "baselines"
        gnn_root = seed_dir / "gnn"

        for _, _, combination_name in FEATURE_COMBINATIONS:
            use_env, use_phylo = COMBINATION_MAP[combination_name]
            gnn_dir = gnn_root / combination_name

            for filename in result_files:
                gnn_results = sorted(gnn_dir.glob(f"*/{filename}"))
                if not gnn_results:
                    raise FileNotFoundError(f"Missing {filename} below {gnn_dir}")
                gnn_result = gnn_results[0]
                artifact_dir = gnn_result.parent
                aggregated_predictions[filename][combination_name].append(read_csv(
                    gnn_result,
                    seed=seed,
                    model="gnn",
                    source="gnn",
                    use_env_features=use_env,
                    use_phylo_features=use_phylo,
                ))

                # Collect baseline frames matching this result file type
                baseline_results = find_baseline_results(baseline_dir, BASELINES, filename, use_env, use_phylo)
                baseline_frames = [
                    read_csv(
                        result, seed=seed, model=result.parent.name,
                        source=("deterministic_baseline" if model in DETERMINISTIC_BASELINES else "r_baseline"),
                        use_env_features=use_env if model in COVARIATE_BASELINES else False,
                        use_phylo_features=use_phylo if model in COVARIATE_BASELINES else False,
                    )
                    for model, result in baseline_results.items()
                ]
                for baseline_frame in baseline_frames:
                    frame = baseline_frame.copy()
                    aggregated_predictions[filename][combination_name].append(frame)

                # Attributions are shared by the min/max exports, so collect
                # them once rather than once per prediction filename.
                if filename != result_files[0]:
                    continue
                for attr_filename in ATTRIBUTION_FILES:
                    attribution_path = artifact_dir / attr_filename
                    if attr_filename == "attributions_spatial_all.parquet" and not use_env:
                        continue
                    if attribution_path.is_file():
                        aggregated_attributions[combination_name][attr_filename].append(read_csv(
                            attribution_path,
                            seed=seed,
                            model="gnn",
                            source="gnn",
                            use_env_features=use_env,
                            use_phylo_features=use_phylo,
                        ))

    # Write aggregated outputs per combination and per result file
    for _, _, combination_name in FEATURE_COMBINATIONS:
        combination_dir = output_dir / combination_name
        combination_dir.mkdir(parents=True, exist_ok=True)
        for filename in result_files:
            frames = aggregated_predictions[filename][combination_name]
            if frames:
                combined_predictions = pd.concat(frames, ignore_index=True, sort=False)
                prediction_path = combination_dir / filename
                # Write aggregated predictions as Parquet
                combined_predictions.to_parquet(prediction_path, index=False)
                LOGGER.info("Saved %d rows to %s", len(combined_predictions), prediction_path)
            else:
                LOGGER.info("No prediction frames found for %s %s; skipping write", combination_name, filename)

        for filename, frames in aggregated_attributions[combination_name].items():
            if not frames:
                continue
            combined_attributions = pd.concat(frames, ignore_index=True, sort=False)
            attribution_path = combination_dir / filename
            combined_attributions.to_parquet(attribution_path, index=False)
            LOGGER.info("Saved %d rows to %s", len(combined_attributions), attribution_path)

    # Reversed (min > max) predictions before projection, for every seed and model.
    order_tables = []
    for seed in seeds:
        seed_dir = runs_dir / f"seed_{seed}"
        for use_env, use_phylo, combination_name in FEATURE_COMBINATIONS:
            models = COVARIATE_BASELINES if use_env or use_phylo else BASELINES
            for result in find_baseline_results(seed_dir / "baselines", models, args.result_file, use_env, use_phylo).values():
                table = collect_bounds_order(seed, combination_name, result.parent.name, result.parent)
                if table is not None:
                    order_tables.append(table)
        for _, _, combination_name in FEATURE_COMBINATIONS:
            for result in sorted((seed_dir / "gnn" / combination_name).glob(f"*/{args.result_file}")):
                table = collect_bounds_order(seed, combination_name, "gnn", result.parent)
                if table is not None:
                    order_tables.append(table)
    if order_tables:
        summary = pd.concat(order_tables, ignore_index=True)
        summary.to_csv(output_dir / "bounds_order_summary.csv", index=False)
        by_model = summary.groupby(["combination", "model"], sort=False)[["n_reversed", "n_pairs"]].sum()
        for (combination, model), row in by_model.iterrows():
            LOGGER.info("Bounds order, %s %s: %d/%d pairs reversed before projection (%.2f%%)",
                        combination, model, row.n_reversed, row.n_pairs, 100 * row.n_reversed / max(row.n_pairs, 1))
        LOGGER.info("Saved %s", output_dir / "bounds_order_summary.csv")
    LOGGER.info("Benchmark completed successfully")


if __name__ == "__main__":
    main()
