"""Benchmark all GNN feature combinations and R baselines over repeated seeds."""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path

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
STOCHASTIC_BASELINES = ("mice",)
LOGGER = logging.getLogger("benchmark")


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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results"),
        help="Benchmark run and aggregate output directory",
    )
    parser.add_argument(
        "--result-file", default="predictions_max_all.parquet",
        help="Per-model merged file to aggregate (default: predictions_max_all.parquet)",
    )
    parser.add_argument(
        "--gnn-args", default="",
        help="Additional train.py arguments as one shell-style quoted string",
    )
    parser.add_argument(
        "--baseline-args", default="",
        help=(
            "Additional r_baselines_fit.py arguments as one shell-style quoted string; "
            "the benchmark manages --baseline_models itself"
        ),
    )
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


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.n_seeds < 1:
        raise SystemExit("n_seeds must be at least 1")

    output_dir = args.output_dir.resolve()
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    gnn_args = shlex.split(args.gnn_args)
    baseline_args = shlex.split(args.baseline_args)
    predictions = {name: [] for _, _, name in FEATURE_COMBINATIONS}
    attributions = {
        name: {filename: [] for filename in ATTRIBUTION_FILES}
        for _, _, name in FEATURE_COMBINATIONS
    }

    seeds = range(args.start_seed, args.start_seed + args.n_seeds)
    deterministic_dir = runs_dir / "deterministic_baselines"
    deterministic_seed = args.start_seed
    deterministic_results = sorted(deterministic_dir.glob(f"*/{args.result_file}"))
    if len(deterministic_results) == len(DETERMINISTIC_BASELINES):
        LOGGER.info("Found existing deterministic baseline results in %s; skipping run", deterministic_dir)
    else:
        run([
            "bash", "run_baselines.sh", *baseline_args,
            "--baseline_models", ",".join(DETERMINISTIC_BASELINES),
            "--k", "-1", "--seed", str(deterministic_seed),
            "--output_dir", str(deterministic_dir),
        ], label=f"Deterministic baselines (split seed {deterministic_seed})")
        deterministic_results = sorted(deterministic_dir.glob(f"*/{args.result_file}"))
        if len(deterministic_results) != len(DETERMINISTIC_BASELINES):
            raise FileNotFoundError(
                f"Expected {len(DETERMINISTIC_BASELINES)} deterministic baseline results "
                f"below {deterministic_dir}, found {len(deterministic_results)}"
            )
    deterministic_frames = [
        read_csv(
            result,
            seed=deterministic_seed,
            model=result.parent.name,
            source="deterministic_baseline",
        )
        for result in deterministic_results
    ]
    for use_env, use_phylo, combination_name in FEATURE_COMBINATIONS:
        for baseline_frame in deterministic_frames:
            frame = baseline_frame.copy()
            frame.insert(1, "use_env_features", use_env)
            frame.insert(2, "use_phylo_features", use_phylo)
            predictions[combination_name].append(frame)

    total_jobs = 1 + args.n_seeds * (1 + len(FEATURE_COMBINATIONS))
    LOGGER.info(
        "Benchmark started: %d seeds, %d jobs, output=%s",
        args.n_seeds, total_jobs, output_dir,
    )

    with tqdm(total=total_jobs, initial=1, desc="Benchmark", unit="job", dynamic_ncols=True) as progress:
        for seed in seeds:
            seed_dir = runs_dir / f"seed_{seed}"
            baseline_dir = seed_dir / "baselines"
            progress.set_postfix_str(f"seed={seed} stochastic baselines", refresh=True)
            baseline_results = sorted(baseline_dir.glob(f"*/{args.result_file}"))
            if len(baseline_results) == len(STOCHASTIC_BASELINES):
                LOGGER.info("Found existing stochastic baseline results for seed %d; skipping", seed)
            else:
                run([
                    "bash", "run_baselines.sh", *baseline_args,
                    "--baseline_models", ",".join(STOCHASTIC_BASELINES),
                    "--k", "-1", "--seed", str(seed),
                    "--output_dir", str(baseline_dir),
                ], label=f"Stochastic baselines (seed {seed})")
                baseline_results = sorted(baseline_dir.glob(f"*/{args.result_file}"))
                if len(baseline_results) != len(STOCHASTIC_BASELINES):
                    raise FileNotFoundError(
                        f"Expected {len(STOCHASTIC_BASELINES)} stochastic baseline results "
                        f"below {baseline_dir}, found {len(baseline_results)}"
                    )
            baseline_frames = [
                read_csv(result, seed=seed, model=result.parent.name, source="r_baseline")
                for result in baseline_results
            ]
            progress.update()

            for use_env, use_phylo, combination_name in FEATURE_COMBINATIONS:
                progress.set_postfix_str(f"seed={seed} {combination_name}", refresh=True)
                gnn_dir = seed_dir / "gnn" / combination_name
                gnn_results = sorted(gnn_dir.glob(f"*/{args.result_file}"))
                if gnn_results:
                    LOGGER.info("Found existing GNN results for %s seed %d; skipping", combination_name, seed)
                else:
                    run([
                        sys.executable, "train.py", *gnn_args,
                        "--use_env_features", str(use_env).lower(),
                        "--use_phylo_features", str(use_phylo).lower(),
                        "--compute_xai", "true",
                        "--k", "-1", "--seed", str(seed), "--output_dir", str(gnn_dir),
                    ], label=f"GNN {combination_name} (seed {seed})")

                    gnn_results = sorted(gnn_dir.glob(f"*/{args.result_file}"))
                    if len(gnn_results) != 1:
                        raise FileNotFoundError(
                            f"Expected one GNN {args.result_file} below {gnn_dir}, "
                            f"found {len(gnn_results)}"
                        )
                gnn_result = gnn_results[0]
                artifact_dir = gnn_result.parent
                predictions[combination_name].append(read_csv(
                    gnn_result,
                    seed=seed,
                    model="gnn",
                    source="gnn",
                    use_env_features=use_env,
                    use_phylo_features=use_phylo,
                ))
                for baseline_frame in baseline_frames:
                    frame = baseline_frame.copy()
                    frame.insert(1, "use_env_features", use_env)
                    frame.insert(2, "use_phylo_features", use_phylo)
                    predictions[combination_name].append(frame)

                for filename in ATTRIBUTION_FILES:
                    attribution_path = artifact_dir / filename
                    if filename == "attributions_spatial_all.parquet" and not use_env:
                        continue
                    if not attribution_path.is_file():
                        raise FileNotFoundError(f"GNN attribution file was not produced: {attribution_path}")
                    attributions[combination_name][filename].append(read_csv(
                        attribution_path,
                        seed=seed,
                        model="gnn",
                        source="gnn",
                        use_env_features=use_env,
                        use_phylo_features=use_phylo,
                    ))
                progress.update()

    LOGGER.info("All training jobs completed; aggregating CSV files")

    # Result files to collect: include both min and max by default plus any explicit result-file
    default_result_files = ["predictions_min_all.parquet", "predictions_max_all.parquet"]
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

    # Deterministic baselines: replicate available frames for each combination and filename
    for filename in result_files:
        deterministic_results = sorted(deterministic_dir.glob(f"*/{filename}"))
        deterministic_frames = [
            read_csv(result, seed=deterministic_seed, model=result.parent.name, source="deterministic_baseline")
            for result in deterministic_results
        ]
        for use_env, use_phylo, combination_name in FEATURE_COMBINATIONS:
            for baseline_frame in deterministic_frames:
                frame = baseline_frame.copy()
                frame.insert(1, "use_env_features", use_env)
                frame.insert(2, "use_phylo_features", use_phylo)
                aggregated_predictions[filename][combination_name].append(frame)

    # Per-seed results: detect all seed_* directories and collect per-file
    for seed_dir in sorted(runs_dir.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except Exception:
            LOGGER.debug("Skipping non-seed directory: %s", seed_dir)
            continue

        baseline_dir = seed_dir / "baselines"
        gnn_root = seed_dir / "gnn"

        for _, _, combination_name in FEATURE_COMBINATIONS:
            use_env, use_phylo = COMBINATION_MAP[combination_name]
            gnn_dir = gnn_root / combination_name

            for filename in result_files:
                gnn_results = sorted(gnn_dir.glob(f"*/{filename}"))
                if not gnn_results:
                    # No result of this type for this seed/combination; skip
                    continue
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
                baseline_results = sorted(baseline_dir.glob(f"*/{filename}"))
                baseline_frames = [
                    read_csv(result, seed=seed, model=result.parent.name, source="r_baseline")
                    for result in baseline_results
                ]
                for baseline_frame in baseline_frames:
                    frame = baseline_frame.copy()
                    frame.insert(1, "use_env_features", use_env)
                    frame.insert(2, "use_phylo_features", use_phylo)
                    aggregated_predictions[filename][combination_name].append(frame)

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
    LOGGER.info("Benchmark completed successfully")


if __name__ == "__main__":
    main()
