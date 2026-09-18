"""Run repeated GNN trainings with permuted phylogenetic features and aggregate results.

Each run sets `--shuffle_phylo true` in `train.py` so phylogenetic features are shuffled
among species. Outputs are written under `--output-dir` in a `phylo_permutations/perm_<seed>`
subdirectory and are aggregated at the end.
"""

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

LOGGER = logging.getLogger("phylo_permutations")

ATTRIBUTION_FILES = ("attributions_species_all.csv", "attributions_spatial_all.csv")


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_permutations", type=int, help="Number of permutation runs to execute")
    parser.add_argument("--start-seed", type=int, default=42, help="First seed (default: 42)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results"),
        help="Base output directory to store permutation runs and aggregates",
    )
    parser.add_argument(
        "--result-file", default="predictions_max_all.csv",
        help="Per-run merged CSV filename to expect and aggregate (default: predictions_max_all.csv)",
    )
    parser.add_argument(
        "--gnn-args", default="",
        help="Additional train.py arguments as one shell-style quoted string",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run permutations even if results already exist",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true", help="Only aggregate existing results without running new permutations",
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


def read_csv(path: Path, *, seed: int, model: str, source: str, use_env_features: bool | None = None, use_phylo_features: bool | None = None,) -> pd.DataFrame:
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
    if args.n_permutations < 1:
        raise SystemExit("n_permutations must be at least 1")

    output_dir = args.output_dir.resolve()
    runs_dir = output_dir / "runs" / "phylo_permutations"
    output_dir.mkdir(parents=True, exist_ok=True)
    gnn_args = shlex.split(args.gnn_args)

    seeds = range(args.start_seed, args.start_seed + args.n_permutations)

    total_jobs = args.n_permutations
    LOGGER.info("Phylo permutation benchmark: %d runs, output=%s", args.n_permutations, output_dir)

    if not args.aggregate_only:
        with tqdm(total=total_jobs, desc="Permutations", unit="run", dynamic_ncols=True) as progress:
            for seed in seeds:
                progress.set_postfix_str(f"seed={seed}", refresh=True)
                perm_dir = runs_dir / f"perm_{seed}"
                gnn_dir = perm_dir / "gnn" / "env_phylo"
                result_glob = gnn_dir.glob(f"*/{args.result_file}")
                existing = list(result_glob)
                if existing and not args.force:
                    LOGGER.info("Found existing results for perm_%d; skipping", seed)
                    progress.update()
                    continue

                # Ensure directory exists
                gnn_dir.mkdir(parents=True, exist_ok=True)

                run([
                    sys.executable, "train.py", *gnn_args,
                    "--use_env_features", "true",
                    "--use_phylo_features", "true",
                    "--shuffle_phylo", "true",
                    "--compute_xai", "true",
                    "--k", "-1",
                    "--seed", str(seed),
                    "--output_dir", str(gnn_dir),
                ], label=f"Permuted GNN perm_{seed}")
                progress.update()

    LOGGER.info("All permutation runs finished; aggregating outputs")

    # Aggregate both min and max prediction files (plus any explicit --result-file)
    default_result_files = ["predictions_min_all.csv", "predictions_max_all.csv"]
    result_files = list(dict.fromkeys([args.result_file] + default_result_files))

    aggregated_frames: dict[str, list[pd.DataFrame]] = {filename: [] for filename in result_files}
    aggregated_attributions: dict[str, list[pd.DataFrame]] = {filename: [] for filename in ATTRIBUTION_FILES}

    # Scan for all matching result files under the runs directory (perm_* and seed_* etc.)
    runs_root = output_dir / "runs"
    for result_path in sorted(runs_root.rglob(args.result_file)):
        # only consider results that come from a GNN env_phylo run
        parts = result_path.parts
        if "gnn" not in parts or "env_phylo" not in parts:
            continue
        # infer seed or perm label from ancestor folders
        seed = None
        for part in parts:
            if part.startswith("seed_"):
                try:
                    seed = int(part.split("_", 1)[1])
                    break
                except Exception:
                    pass
            if part.startswith("perm_"):
                try:
                    seed = int(part.split("_", 1)[1])
                    break
                except Exception:
                    pass
        seed_val = seed if seed is not None else -1
        aggregated_frames[args.result_file].append(read_csv(result_path, seed=seed_val, model="gnn", source="permute_phylo", use_env_features=True, use_phylo_features=True))
        artifact_dir = result_path.parent
        for attr in ATTRIBUTION_FILES:
            attribution_path = artifact_dir / attr
            if attribution_path.is_file():
                aggregated_attributions[attr].append(read_csv(attribution_path, seed=seed_val, model="gnn", source="permute_phylo", use_env_features=True, use_phylo_features=True))

    # Write aggregated predictions
    perm_out_dir = output_dir / "phylo_permutations"
    perm_out_dir.mkdir(parents=True, exist_ok=True)
    for filename, frames in aggregated_frames.items():
        if frames:
            combined = pd.concat(frames, ignore_index=True, sort=False)
            combined.to_csv(perm_out_dir / filename, index=False)
            LOGGER.info("Saved %d rows to %s", len(combined), perm_out_dir / filename)
        else:
            LOGGER.info("No prediction frames to aggregate for %s", filename)

    for filename, frames in aggregated_attributions.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined.to_csv(perm_out_dir / filename, index=False)
        LOGGER.info("Saved %d rows to %s", len(combined), perm_out_dir / filename)

    LOGGER.info("Permutation benchmark completed")


if __name__ == "__main__":
    main()
