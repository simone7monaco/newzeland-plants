"""Rerun MICE only to produce model_complexity.csv per-fold without touching predictions.

This script locates MICE baseline fold directories under a runs tree (e.g.
`benchmark_results/runs/seed_42/baselines/.../fold_0`) and, where
`model_complexity.csv` is missing (or --force), fits MICE on the original
training split and writes a small CSV with the measured parameter count.

It intentionally does NOT write predictions or metrics so it is safe to run
on an existing benchmark tree.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from r_models import MICE, RBackendUnavailable
from loader import PlantDataset, data_split
from r_baselines_fit import build_species_dataframe

LOGGER = logging.getLogger("run_mice_complexity")


def find_mice_fold_dirs(runs_dir: Path, seeds: Iterable[int]) -> list[Path]:
    found: list[Path] = []
    for seed in seeds:
        # common places where baselines are written by run_benchmark/run_baselines
        candidates = [runs_dir / f"seed_{seed}" / "baselines", runs_dir / f"seed_{seed}"]
        for base in candidates:
            if not base.exists():
                continue
            # find directories whose name contains 'mice' (case-insensitive)
            for d in base.iterdir():
                if not d.is_dir():
                    continue
                if 'mice' in d.name.lower():
                    # collect any fold_* directories under this folder
                    for fold in sorted(d.rglob('fold_*')):
                        if fold.is_dir():
                            found.append(fold)
    return found


def process_fold(fold_dir: Path, seed: int, fold_k: int, use_phylo: bool, force: bool) -> None:
    out_path = fold_dir / 'model_complexity.csv'
    if out_path.exists() and not force:
        LOGGER.info('Skipping existing complexity CSV: %s', out_path)
        return

    LOGGER.info('Computing complexity for %s (seed=%d, fold=%d)', fold_dir, seed, fold_k)

    # Load dataset and reconstruct the same split used by r_baselines_fit.py
    data_path = Path('data/Ferns/')
    dataset = PlantDataset(
        data_path,
        transform=None,
        trait_representation='min_max_range',
        traits_filename='FernTraits.xlsx',
        invalid_bounds_policy='missing',
    )
    raw_data = dataset[0]
    # Perform the same split
    _, test_data = data_split(raw_data.clone(), k=fold_k, seed=seed, split_strategy='random')

    # Build wide dataframes
    trait_feature_keys = ['species_x_min', 'species_x_max']
    trait_names = dataset.trait_names
    full_trait_df, trait_col_names = build_species_dataframe(raw_data, dataset, trait_feature_keys, trait_names)
    variable_names = [key.split('_')[-1] for key in trait_feature_keys]
    trait_columns = [column for variable in variable_names for column in trait_col_names[variable]]
    full_df = pd.concat([full_trait_df, dataset.traits_gen.loc[raw_data.species_names]], axis=1)
    if use_phylo:
        phylo_df = pd.DataFrame(raw_data.species_x_phylo.numpy(), index=raw_data.species_names,
                                columns=[f'phylo_{i}' for i in range(raw_data.species_x_phylo.shape[1])])
        full_df = pd.merge(full_df, phylo_df, left_index=True, right_index=True, how='left')

    # Retrieve train/test using data_split again to get train indices
    train_data, test_data = data_split(raw_data.clone(), k=fold_k, seed=seed, split_strategy='random')
    train_df = full_df.loc[train_data.species_names]

    # Fit MICE on training set only (training_protocol 1 behavior)
    mice = MICE(seed=seed)
    try:
        mice.fit(train_df)
    except RBackendUnavailable as exc:
        raise RuntimeError(
            "R backend is not available. Install R, rpy2 and R packages 'mice' and 'rpart' to run this script."
        ) from exc

    k_parameters = mice.estimate_parameter_count()
    pd.DataFrame({'k_parameters': [int(k_parameters)]}).to_csv(out_path, index=False)
    LOGGER.info('Wrote complexity %d to %s', int(k_parameters), out_path)


def parse_seeds(seed_arg: str, n_seeds: int | None, start_seed: int) -> list[int]:
    if seed_arg:
        parts = [int(x.strip()) for x in seed_arg.split(',') if x.strip()]
        return parts
    if n_seeds is None:
        return [start_seed]
    return list(range(start_seed, start_seed + n_seeds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, default=Path('benchmark_results/runs'), help='Root runs directory')
    parser.add_argument('--start-seed', type=int, default=42)
    parser.add_argument('--n-seeds', type=int, default=5, help='Number of consecutive seeds to scan')
    parser.add_argument('--seeds', type=str, default='', help='Comma-separated list of explicit seeds to process')
    parser.add_argument('--folds', type=str, default='', help='Comma-separated fold indices to process (e.g. 0,1). Empty = auto-detect folds')
    parser.add_argument('--use-phylo', action='store_true', help='Include phylogenetic covariates when refitting MICE')
    parser.add_argument('--force', action='store_true', help='Overwrite existing model_complexity.csv files')
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(message)s')

    seeds = parse_seeds(args.seeds, args.n_seeds, args.start_seed)
    fold_dirs = find_mice_fold_dirs(args.runs_dir, seeds)
    if not fold_dirs:
        LOGGER.warning('No MICE fold directories found under %s for seeds %s', args.runs_dir, seeds)
        return

    # If user provided explicit folds, filter by them
    explicit_folds = [int(x) for x in args.folds.split(',') if x.strip()] if args.folds else None

    for fold_dir in sorted(set(fold_dirs)):
        # infer seed and fold index from the path if possible
        parts = fold_dir.parts
        seed_num = None
        for p in parts:
            if p.startswith('seed_'):
                try:
                    seed_num = int(p.split('_', 1)[1])
                except Exception:
                    pass
        # infer fold
        try:
            fold_name = fold_dir.name
            fold_k = int(fold_name.split('_')[1])
        except Exception:
            LOGGER.warning('Cannot infer fold index from %s; skipping', fold_dir)
            continue
        if explicit_folds is not None and fold_k not in explicit_folds:
            LOGGER.info('Skipping fold %d due to --folds filter', fold_k)
            continue
        if seed_num is None:
            # fall back to provided seeds list first entry
            seed_num = seeds[0]

        try:
            process_fold(fold_dir, seed_num, fold_k, args.use_phylo, args.force)
        except Exception as exc:
            LOGGER.exception('Failed to process %s: %s', fold_dir, exc)


if __name__ == '__main__':
    main()
