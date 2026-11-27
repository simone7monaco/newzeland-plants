import os

# One BLAS thread per process, set before numpy and R load OpenBLAS. The
# benchmark runs several baseline processes in parallel, and mice's
# remove.lindep calls eigen() for every variable and iteration: with threaded
# OpenBLAS on a loaded machine one m=1, maxit=1 fit took 250 s instead of 7 s.
# Export OPENBLAS_NUM_THREADS to override.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
from copy import deepcopy
from typing import cast
import warnings
import yaml
import numpy as np
import torch
from torch_geometric.data import Data
from loader import PlantDataset, data_split
from occlusion import trait_groups, group_for_trait
from tester import compute_metrics_minmax, count_reversed_bounds, long_predictions_frame, project_ordered_bounds
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from Bio import Phylo
from sklearn.impute import SimpleImputer

from r_models import MICE, MissForest, Rphylopars, pool_completions
from analysis_utils import AnalysisConsistencyError, TREE_ONLY_MODELS
from ml_baselines import ML_MODEL_NAMES, TunedTraitRegressor, build_ml_covariates, species_environment_summary
from prediction_bounds import write_safeguard_log

# Baselines that accept non-trait covariates (phylogenetic / environmental columns).
COVARIATE_MODEL_NAMES = ('mice', 'missforest', *ML_MODEL_NAMES)


def uses_covariates(model_name: str) -> bool:
    return model_name.strip().lower() in COVARIATE_MODEL_NAMES


def feature_configurations(args):
    """Run trait-only methods once and supported covariate models in all four settings."""
    if not args.all_feature_combinations:
        yield deepcopy(args)
        return
    names = [name.strip() for name in args.baseline_models.split(',') if name.strip()]
    combinations = [(False, False), (False, True), (True, False), (True, True)]
    for use_env, use_phylo in combinations:
        selected = names if not (use_env or use_phylo) else [name for name in names if uses_covariates(name)]
        if selected:
            local_args = deepcopy(args)
            local_args.use_env_features, local_args.use_phylo_features = use_env, use_phylo
            local_args.baseline_models = ','.join(selected)
            yield local_args


def model_dir_suffix(args, model_name: str) -> str:
    """Directory suffix naming only the inputs that this baseline actually uses."""
    suffix = f"_prot{args.training_protocol}_occlusion-{args.occlusion_protocol}"
    if args.include_unobserved_species:
        suffix += '_include-unobserved'
    if uses_covariates(model_name):
        if args.use_env_features:
            suffix += '_env'
        if args.use_phylo_features:
            suffix += '_phylo'
    if model_name.strip().lower() in ML_MODEL_NAMES and not args.use_categorical_features:
        suffix += '_nocat'
    return suffix


class TrainingStatisticImputer:
    """Column-wise training mean or median imputation."""

    def __init__(self, strategy: str) -> None:
        self.imputer = SimpleImputer(strategy=strategy)
        self.columns: list[str] | None = None

    def fit(self, data: pd.DataFrame, tree_path: Path | None = None) -> "TrainingStatisticImputer":
        self.columns = list(data.columns)
        self.imputer.fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            raise RuntimeError("fit() must be called before transform()")
        if list(data.columns) != self.columns:
            raise ValueError("transform columns must exactly match the fitted columns")
        return pd.DataFrame(
            self.imputer.transform(data), index=data.index, columns=self.columns
        )

    def fit_transform(self, data: pd.DataFrame, tree_path: Path | None = None) -> pd.DataFrame:
        return self.fit(data).transform(data)


class PhylogeneticKNNImputer:
    """Uniformly average the k closest observed training tips per trait."""

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.columns: list[str] | None = None
        self.train_data: pd.DataFrame | None = None
        self.tree = None
        self.tip_labels: set[str] | None = None

    def fit(self, data: pd.DataFrame, tree_path: Path) -> "PhylogeneticKNNImputer":
        tree = Phylo.read(tree_path, 'newick')
        tip_labels = {str(tip.name) for tip in tree.get_terminals()}
        missing = sorted(set(data.index).difference(tip_labels))
        if missing:
            warnings.warn(
                f"Phylogenetic k-NN removed {len(missing)} training species not "
                f"present in the phylogenetic tree: {missing}",
                UserWarning,
                stacklevel=2,
            )
        retained = data.drop(index=missing)
        if retained.empty:
            raise ValueError("No training species are present in the phylogenetic tree")
        all_missing = retained.columns[retained.isna().all()].tolist()
        if all_missing:
            raise ValueError(f"Training traits have no observed values: {all_missing}")
        self.columns = list(retained.columns)
        self.train_data = retained.copy(deep=True)
        self.tree = tree
        self.tip_labels = tip_labels
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None or self.train_data is None or self.tree is None or self.tip_labels is None:
            raise RuntimeError("fit() must be called before transform()")
        if list(data.columns) != self.columns:
            raise ValueError("transform columns must exactly match the fitted columns")
        missing_tips = sorted(set(data.index).difference(self.tip_labels))
        if missing_tips:
            warnings.warn(
                f"Phylogenetic k-NN removed {len(missing_tips)} testing species not "
                f"present in the phylogenetic tree: {missing_tips}",
                UserWarning,
                stacklevel=2,
            )
        result = data.drop(index=missing_tips).copy(deep=True)
        for species in result.index:
            distances = {
                train_species: float(self.tree.distance(species, train_species))
                for train_species in self.train_data.index
            }
            for column in self.columns:
                if pd.notna(result.at[species, column]):
                    continue
                observed = self.train_data[column].dropna()
                neighbours = sorted(
                    observed.index,
                    key=lambda candidate: (distances[candidate], candidate),
                )[:self.k]
                result.at[species, column] = float(observed.loc[neighbours].mean())
        return result

    def fit_transform(self, data: pd.DataFrame, tree_path: Path) -> pd.DataFrame:
        return self.fit(data, tree_path).transform(data)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mask_trait_inputs(data: Data, mask: torch.Tensor, trait_feature_keys: list[str]) -> Data:
    """Hide selected observed trait cells from a graph while retaining their targets separately."""
    masked_data = data.clone()
    masked_data.traits_nanmask = masked_data.traits_nanmask | mask
    for trait_feature_key in trait_feature_keys:
        values = getattr(masked_data, trait_feature_key).clone()
        setattr(masked_data, trait_feature_key, values.masked_fill(mask, 0.0))
    return masked_data


def get_args():
    parser = argparse.ArgumentParser(description='Fit min/max trait imputation baselines')
    parser.add_argument('--output_dir', type=Path, default=Path('results/'), help='Directory to save results and models')
    parser.add_argument('--use_env_features', type=str2bool, nargs='?', const=True, default=False, help='Use species-level environmental summaries (occurrence-weighted mean/SD of location covariates) as extra columns for MICE, MissForest and the ML baselines')
    parser.add_argument('--use_phylo_features', type=str2bool, nargs='?', const=True, default=False, help='Whether to use phylogenetic (Node2Vec) features (MICE, MissForest and ML baselines)')
    parser.add_argument('--use_categorical_features', type=str2bool, nargs='?', const=True, default=True, help='ML baselines: include categorical attributes such as family, as in the GNN species input')
    parser.add_argument('--all_feature_combinations', action='store_true', help='Run each supported baseline with neither input, phylogeny only, environment only and both; trait-only methods run once')
    parser.add_argument('--ml_search_iter', type=int, default=30, help='ML baselines: random-search candidates per target trait')
    parser.add_argument('--ml_inner_folds', type=int, default=5, help='ML baselines: inner species-level CV folds for tuning on outer-training species')
    parser.add_argument('--fixed_hyperparameters', type=Path, default=None, help='YAML file with fixed ML parameters by model and trait')
    parser.add_argument('--n_jobs', type=int, default=-1, help='ML baselines: parallel jobs for hyperparameter search')
    parser.add_argument('--invalid_bounds_policy', type=str, default='missing', choices=['missing', 'error', 'keep'], help='How to handle negative or inconsistent min/max/range records')
    parser.add_argument(
        '--baseline_models',
        type=str,
        default='training_mean,training_median,phylo_nn,phylo_knn,rphylopars_bm,mice,random_forest,xgboost,mlp',
        help=(
            'Comma-separated list of baseline models to run '
            '(options: training_mean, training_median, phylo_nn, phylo_knn, '
            'missforest, mice, rphylopars_bm, random_forest, xgboost, mlp). The Rphylopars '
            'baseline uses BM with a diagonal phylogenetic covariance for '
            'numerical stability. random_forest, xgboost and mlp are tuned '
            'non-graph regressors whose inputs are selected with '
            '--use_env_features, --use_phylo_features and --use_categorical_features.'
        ),
    )
    parser.add_argument('--k', type=int, default=-1, help='Fold index for cross-validation (0-4). Use -1 to perform a complete run over all folds sequentially.')
    parser.add_argument('--split_strategy', type=str, default='random', choices=['random', 'stable', 'louvain', 'genus', 'clade'], help='Outer CV split strategy')
    parser.add_argument('--traits_filename', default='FernTraits.xlsx', help='Trait workbook name within data/Ferns')
    parser.add_argument('--include_unobserved_species', type=str2bool, nargs='?', const=True, default=False, help='Retain species with no observed target traits for exploratory inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used for splitting and model fitting')
    parser.add_argument('--enforce_bounds_order', type=str2bool, nargs='?', const=True, default=True, help='Replace reversed (min, max) predictions by their midpoint before scoring and saving; reversed pairs are always counted first')
    parser.add_argument('--occlusion_protocol', choices=['paired', 'individual'], default='paired', help='Mask fertile/sterile pairs jointly or reproduce individual-trait occlusion')
    parser.add_argument('--validation_mask_ratio', type=float, default=0.15, help='Fraction of observed outer-training cells held out for inner-validation metrics')
    parser.add_argument('--training_protocol', type=int, default=1, choices=[1, 2], help='Training protocol: 1 = fit on outer-training species only; 2 = fit on outer-training + inner-validation species (transductive)') 
    return parser.parse_args()


def build_baseline_model(name: str, categorical_columns: list[str], seed_value: int,
                         ml_options: dict | None = None):
    """Instantiate a baseline imputer by name."""
    name = name.strip().lower()
    if name in ML_MODEL_NAMES:
        if ml_options is None:
            raise ValueError(f"ML baseline '{name}' requires ml_options")
        return TunedTraitRegressor(name, seed=seed_value, **ml_options)
    if name == 'missforest':
        return MissForest(categorical_columns=categorical_columns, seed=seed_value)
    if name == 'mice':
        return MICE(method='cart', seed=seed_value)
    if name == 'training_mean':
        return TrainingStatisticImputer(strategy='mean')
    if name == 'training_median':
        return TrainingStatisticImputer(strategy='median')
    if name == 'phylo_nn':
        return PhylogeneticKNNImputer(k=1)
    if name == 'phylo_knn':
        return PhylogeneticKNNImputer(k=5)
    if name == 'rphylopars_bm':
        # Estimating a fully correlated 12 x 12 phylogenetic covariance on
        # this dataset drives Rphylopars through nearly singular matrices.
        # The diagonal BM model retains phylogenetic reconstruction without
        # the unstable cross-trait covariance optimization.
        return Rphylopars(method='BM', phylopars_kwargs={'phylo_correlated': False})
    raise ValueError(f"Unknown baseline model '{name}'")


def build_species_dataframe(data: Data, dataset: PlantDataset, trait_feature_keys: list[str],
                             trait_names: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Assemble a wide (species x trait-variable) DataFrame with NaNs restored for missing entries."""
    variable_names = [key.split('_')[-1] for key in trait_feature_keys]
    trait_col_names = {
        variable: [f'{c}{variable.title()}' for c in getattr(dataset, f"traits_{variable}").columns]
        for variable in variable_names
    }
    trait_df = pd.concat([
        pd.DataFrame(getattr(data, key).numpy(), index=data.species_names, columns=trait_col_names[variable])
        for key, variable in zip(trait_feature_keys, variable_names)
    ], axis=1)

    nanmask_df = pd.DataFrame(data.traits_nanmask.numpy(), index=data.species_names, columns=trait_names)
    for variable in variable_names:
        for trait, col in zip(trait_names, trait_col_names[variable]):
            trait_df.loc[nanmask_df[trait], col] = np.nan

    return trait_df, trait_col_names


def evaluate_and_save(imputed_df: pd.DataFrame, test_data: Data, trait_names: list[str],
                       trait_col_names: dict[str, list[str]], variable_names: list[str],
                       save_dir: Path,
                       evaluation_species: list[str] | None = None,
                       parameter_count: float | None = None,
                       enforce_bounds_order: bool = True,
                       model_name: str | None = None) -> pd.DataFrame:
    """Score the imputed test-species entries against ground truth and persist CSV outputs."""
    save_dir.mkdir(parents=True, exist_ok=True)
    all_test_species = list(test_data.species_names)
    test_species = all_test_species if evaluation_species is None else evaluation_species
    positions = [all_test_species.index(species) for species in test_species]
    eval_mask = ~test_data.traits_nanmask[positions]

    predictions = {
        variable: torch.from_numpy(
            imputed_df.loc[test_species, trait_col_names[variable]].to_numpy(
                dtype=np.float32, copy=True
            )
        )
        for variable in variable_names
    }
    # Phylogeny-only methods cannot predict species absent from the tree:
    # their rows stay NaN and only available prediction pairs are scored.
    # Any other non-finite prediction on an observed cell is an error, raised
    # here (per fold) rather than silently dropped from the evaluation.
    finite = torch.isfinite(predictions['min']) & torch.isfinite(predictions['max'])
    infinite = torch.isinf(predictions['min']) | torch.isinf(predictions['max'])
    missing = eval_mask & ~finite
    if (eval_mask & infinite).any() or (missing.any() and model_name not in TREE_ONLY_MODELS):
        rows, cols = torch.nonzero(missing, as_tuple=True)
        examples = [(test_species[r], trait_names[c]) for r, c in zip(rows[:5].tolist(), cols[:5].tolist())]
        raise AnalysisConsistencyError(
            f'{model_name or "baseline"} {save_dir}: {int(missing.sum())} observed test cells have non-finite predictions, e.g. {examples}'
        )
    eval_mask = eval_mask & finite
    # Count reversed pairs on the raw predictions, then (optionally) project them.
    bounds_order = count_reversed_bounds(predictions['min'], predictions['max'], eval_mask, trait_names, enforce_bounds_order)
    if enforce_bounds_order:
        predictions['min'], predictions['max'], _ = project_ordered_bounds(predictions['min'], predictions['max'])
    truths = {
        variable: getattr(test_data, f'species_x_{variable}')[positions]
        for variable in variable_names
    }

    metrics = compute_metrics_minmax(
        predictions['min'], predictions['max'], None,
        truths['min'], truths['max'], None,
        eval_mask, trait_names,
    )

    metrics.to_csv(save_dir / 'per_trait_metrics.csv', index=False)
    bounds_order.to_csv(save_dir / 'bounds_order.csv', index=False)
    action = 'set to their midpoint' if enforce_bounds_order else 'left unchanged (projection disabled)'
    print(f"Bounds order: {int(bounds_order.n_reversed.sum())} of {int(bounds_order.n_pairs.sum())} test (min, max) pairs reversed "
          f"({int(bounds_order.n_eval_reversed.sum())} among evaluated cells); {action}.")
    if parameter_count is not None:
        # For AIC-style diagnostics only; see MICE.estimate_parameter_count for how this is derived.
        pd.DataFrame({'k_parameters': [parameter_count]}).to_csv(save_dir / 'model_complexity.csv', index=False)
    for variable in variable_names:
        pd.DataFrame(predictions[variable].numpy(), index=test_species, columns=trait_names).to_csv(
            save_dir / f'predictions_{variable}.csv'
        )
    long_predictions_frame(
        test_species, save_dir.name, trait_names, predictions, truths,
        eval_mask, ~test_data.traits_nanmask[positions],
    ).to_csv(save_dir / 'predictions_long.csv', index=False)
    print(metrics.to_string(index=False, float_format='%.4f'))
    return metrics


def main(args) -> None:
    print(f"---------------\nFitting baselines with args: {args}")

    baseline_model_names = [name.strip() for name in args.baseline_models.split(',') if name.strip()]
    ml_requested = [name for name in baseline_model_names if name.lower() in ML_MODEL_NAMES]
    if ml_requested and args.training_protocol != 1:
        raise SystemExit(
            f"ML baselines {ml_requested} are supervised per-trait regressors fitted on "
            "outer-training species; use --training_protocol 1."
        )

    data_path = Path('data/Ferns/')
    dataset = PlantDataset(
        data_path,
        transform=None,
        trait_representation='min_max_range',
        traits_filename=args.traits_filename,
        invalid_bounds_policy=args.invalid_bounds_policy,
        include_unobserved_species=args.include_unobserved_species,
    )
    trait_names = dataset.trait_names
    occlusion_groups = trait_groups(trait_names, args.occlusion_protocol)
    raw_data = cast(Data, dataset[0])
    data = raw_data.clone()

    trait_feature_keys = ['species_x_min', 'species_x_max']
    if 'species_x_range' in data:
        data.pop('species_x_range')
    variable_names = [key.split('_')[-1] for key in trait_feature_keys]

    full_trait_df, trait_col_names = build_species_dataframe(data, dataset, trait_feature_keys, trait_names)
    trait_columns = [
        column for variable in variable_names for column in trait_col_names[variable]
    ]
    # Species dataframe contains all the species, with the required trait columns to impute, no phylogenetic or categorical variables are added at this point.

    # data_split mutates `data` in place, tagging it with train_mask/test_mask,
    # while also returning the two species-only subgraphs.
    train_data, test_data = data_split(data, k=args.k, seed=args.seed, split_strategy=args.split_strategy)

    categorical_columns = dataset.traits_gen.columns.tolist()
    full_df = pd.concat([full_trait_df, dataset.traits_gen.loc[data.species_names]], axis=1)

    if args.use_phylo_features:
        phylo_df = pd.DataFrame(data.species_x_phylo.numpy(),
                                index=data.species_names,
                                columns=[f'phylo_{i}' for i in range(data.species_x_phylo.shape[1])])
        full_df = pd.merge(full_df, phylo_df, left_index=True, right_index=True, how='left')

    if args.use_env_features:
        # Same species-level summaries as the ML baselines (occurrence-weighted
        # mean/SD of location covariates); MICE and missForest treat them as
        # additional columns, imputing them where a species has no occurrences.
        env_df = species_environment_summary(data)
        full_df = pd.merge(full_df, env_df, left_index=True, right_index=True, how='left')

    train_df = full_df.loc[train_data.species_names]
    test_df = full_df.loc[test_data.species_names]

    ml_train_df = ml_test_df = None
    ml_options = None
    if ml_requested:
        # Species-level inputs only: no fitted statistic is computed here; all
        # scaling, imputation and tuning happen on outer-training species.
        ml_covariates = build_ml_covariates(
            data,
            use_categorical=args.use_categorical_features,
            use_phylo=args.use_phylo_features,
            use_env=args.use_env_features,
        )
        ml_df = pd.concat([full_trait_df.loc[:, trait_columns], ml_covariates.loc[full_trait_df.index]], axis=1)
        ml_train_df = ml_df.loc[train_data.species_names]
        ml_test_df = ml_df.loc[test_data.species_names]
        fixed_by_model = None
        if args.fixed_hyperparameters:
            with args.fixed_hyperparameters.open() as handle:
                fixed_by_model = yaml.safe_load(handle)['models']
            for model_name in baseline_model_names:
                if model_name in ML_MODEL_NAMES and model_name not in fixed_by_model:
                    raise ValueError(f'Missing fixed parameters for {model_name}')
        ml_options = dict(
            trait_names=trait_names,
            trait_col_names=trait_col_names,
            occlusion_groups=occlusion_groups,
            n_iter=args.ml_search_iter,
            inner_folds=args.ml_inner_folds,
            n_jobs=args.n_jobs,
        )
        print(f"ML baseline inputs: {len(trait_columns)} trait columns + {ml_covariates.shape[1]} covariates")

    # Keep tree-based baselines aligned with the FTOL tree reconstructed by
    # PlantDataset; the repository no longer requires a hand-supplied .nwk file.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree_path = args.output_dir / f"ftol_tree_{dataset.tree_fingerprint}.nwk"
    if not tree_path.exists():
        tree_path.write_text(dataset.tree_newick.rstrip() + '\n')
    dataset.tree_coverage().to_csv(args.output_dir / 'ftol_mapping.csv', index=False)
    tree = Phylo.read(tree_path, 'newick')
    tree_tips = {str(tip.name) for tip in tree.get_terminals()}

    for model_name in baseline_model_names:
        print(f"\n=== Fitting baseline '{model_name}' (fold {args.k}) ===")
        model_options = dict(ml_options) if ml_options is not None else None
        if model_name in ML_MODEL_NAMES and fixed_by_model is not None:
            model_options['fixed_params'] = fixed_by_model[model_name]
        model = build_baseline_model(model_name, categorical_columns, args.seed + args.k, model_options)
        model_train_df = train_df
        model_test_df = test_df
        if isinstance(model, TunedTraitRegressor):
            model_train_df = ml_train_df
            model_test_df = ml_test_df
        evaluation_species = list(test_data.species_names)
        if isinstance(model, (Rphylopars, PhylogeneticKNNImputer)):
            missing_train = sorted(set(train_df.index).difference(tree_tips))
            missing_test = sorted(set(test_df.index).difference(tree_tips))
            if missing_train:
                warnings.warn(
                    f"{model_name} removed training species not present in the "
                    f"phylogenetic tree: {missing_train}",
                    UserWarning,
                )
            if missing_test:
                warnings.warn(
                    f"{model_name} cannot predict test species absent from the "
                    f"phylogenetic tree; their output rows remain missing: {missing_test}",
                    UserWarning,
                )
            model_train_df = train_df.drop(index=missing_train)
            model_test_df = test_df.drop(index=missing_test)
            # Keep all held-out taxa in the exported table; taxa absent from
            # the tree remain NaN rather than receiving invented neighbours.

        imputed_df_all = pd.DataFrame(
            np.nan,
            index=evaluation_species,
            columns=[
                column
                for variable in variable_names
                for column in trait_col_names[variable]
            ],
        )
        model_parameter_counts: list[float] = []

        if args.training_protocol == 1:
            if isinstance(model, MissForest):
                raise NotImplementedError("MissForest cannot be evaluated with training protocol 1. Please use MICE or Rphylopars only.")
            elif isinstance(model, MICE):
                model.fit(model_train_df)
                model_parameter_counts.append(model.estimate_parameter_count())
            elif isinstance(model, Rphylopars):  # Rphylopars: phylogeny-only model, no genetic covariates
                model.fit(model_train_df.loc[:, trait_columns], tree_path)
            elif isinstance(model, PhylogeneticKNNImputer):
                model.fit(model_train_df.loc[:, trait_columns], tree_path)
            elif isinstance(model, TrainingStatisticImputer):
                model.fit(model_train_df.loc[:, trait_columns])
            elif isinstance(model, TunedTraitRegressor):
                # One tuned regressor per target trait; its occlusion group is
                # excluded from the inputs, matching the test-time masking.
                model.fit(model_train_df)
            else:
                raise ValueError(f"Unknown model type: {type(model)}")

            for j in range(len(trait_names)):
                columns_to_impute = [trait_col_names[variable][j] for variable in variable_names]
                columns_to_mask = [trait_col_names[variable][idx] for variable in variable_names for idx in group_for_trait(occlusion_groups, j)]
                masked_test_df = model_test_df.copy(deep=True)
                masked_test_df.loc[:, columns_to_mask] = np.nan
                if not isinstance(model, (MICE, TunedTraitRegressor)):
                    masked_test_df = masked_test_df.loc[:, trait_columns]

                if isinstance(model, TunedTraitRegressor):
                    observed = pd.Series(
                        (~test_data.traits_nanmask[:, j]).numpy(), index=test_data.species_names,
                    ).loc[masked_test_df.index].to_numpy()
                    imputed_df = model.transform(masked_test_df, j, evaluation_mask=observed)
                elif isinstance(model, MICE):
                    imputed_df = pool_completions(model.transform(masked_test_df))
                else:
                    imputed_df = model.transform(masked_test_df)
                imputed_df_all.loc[imputed_df.index, columns_to_impute] = imputed_df.loc[:, columns_to_impute].to_numpy()

        else:  # args.training_protocol == 2
            for j in range(len(trait_col_names[variable_names[0]])):
                columns_to_impute = [trait_col_names[variable][j] for variable in variable_names]
                columns_to_mask = [trait_col_names[variable][idx] for variable in variable_names for idx in group_for_trait(occlusion_groups, j)]
                masked_test_df = model_test_df.copy(deep=True)
                masked_test_df.loc[:, columns_to_mask] = np.nan
                masked_df = pd.concat([model_train_df, masked_test_df], axis=0, verify_integrity=True)
                if isinstance(model, MissForest):
                    imputed_df = model.fit_transform(masked_df)
                elif isinstance(model, MICE):
                    imputed_df = pool_completions(model.fit_transform(masked_df))
                    model_parameter_counts.append(model.estimate_parameter_count())
                elif isinstance(model, Rphylopars):
                    imputed_df = model.fit_transform(masked_df.loc[:, trait_columns], tree_path)
                elif isinstance(model, PhylogeneticKNNImputer):
                    imputed_df = model.fit_transform(masked_df.loc[:, trait_columns], tree_path)
                elif isinstance(model, TrainingStatisticImputer):
                    imputed_df = model.fit_transform(masked_df.loc[:, trait_columns])
                else:
                    raise ValueError(f"Unknown model type: {type(model)}")

                imputed_df_all.loc[model_test_df.index, columns_to_impute] = imputed_df.loc[model_test_df.index, columns_to_impute].to_numpy()

        save_dir = args.output_dir / f"{model_name.upper()}{model_dir_suffix(args, model_name)}" / f'fold_{args.k}'
        evaluate_and_save(imputed_df_all, test_data, trait_names, trait_col_names, variable_names,
                           save_dir, evaluation_species,
                           parameter_count=float(np.mean(model_parameter_counts)) if model_parameter_counts else None,
                           enforce_bounds_order=args.enforce_bounds_order, model_name=model_name)
        if isinstance(model, TunedTraitRegressor):
            pd.DataFrame(model.tuning_records).to_csv(save_dir / 'tuning.csv', index=False)
            write_safeguard_log(model.prediction_diagnostics, save_dir / 'prediction_safeguards.csv')


"""
TODO
Aggiungere variante Training test protocol su MissForest (unico protocollo possibile) e MICE

## Fully transductive matrix imputation
For each variable j, fit_transform(train + test_j_masked)
"""


def run_configuration(args):
    seed = args.seed
    pl.seed_everything(seed)
    baseline_model_names = [name.strip() for name in args.baseline_models.split(',') if name.strip()]
    if args.use_env_features or args.use_phylo_features:
        # Trait-only baselines cannot use these inputs: skip them entirely so no
        # output directory suggests results with env/phylo information.
        skipped = [name for name in baseline_model_names if not uses_covariates(name)]
        if skipped:
            warnings.warn(
                f"Skipping {skipped}: they use trait values (and the tree) only, so "
                "--use_env_features/--use_phylo_features do not apply. Run them without these flags.",
                UserWarning,
            )
        baseline_model_names = [name for name in baseline_model_names if uses_covariates(name)]
        args.baseline_models = ','.join(baseline_model_names)
        if not baseline_model_names:
            raise SystemExit('No requested baseline uses environmental or phylogenetic features.')
    baseline_model_list = "\n- " + "\n- ".join(baseline_model_names) if baseline_model_names else "No baseline models specified"

    print(f"""
+------------------------------------------------------
| Starting baselines with models:
| {baseline_model_list}
|
""")


    if args.k != -1:
        print(f"Running fold {args.k + 1}/5")
        main(args)
    else:
        for k in range(5):
            print(f"Running fold {k + 1}/5")
            args.k = k
            main(args)

        variable_names = ['min', 'max']

        for model_name in baseline_model_names:
            model_dir = args.output_dir / f"{model_name.upper()}{model_dir_suffix(args, model_name)}"
            fold_dirs = sorted(model_dir.glob('fold_*'))

            parts: list[pd.DataFrame] = []
            for d in fold_dirs:
                p = d / 'per_trait_metrics.parquet'
                c = d / 'per_trait_metrics.csv'
                if p.exists():
                    parts.append(pd.read_parquet(p))
                elif c.exists():
                    parts.append(pd.read_csv(c))
            if parts:
                metrics_all = pd.concat(parts, ignore_index=True)
                metrics_all.to_parquet(model_dir / 'per_trait_metrics_all.parquet', index=False)

            order_parts = [pd.read_csv(d / 'bounds_order.csv').assign(fold=d.name) for d in fold_dirs if (d / 'bounds_order.csv').exists()]
            if order_parts:
                order_all = pd.concat(order_parts, ignore_index=True).groupby('trait', sort=False).agg(
                    n_pairs=('n_pairs', 'sum'), n_reversed=('n_reversed', 'sum'),
                    n_eval_pairs=('n_eval_pairs', 'sum'), n_eval_reversed=('n_eval_reversed', 'sum'),
                    max_reversal=('max_reversal', 'max'), projected=('projected', 'all'),
                ).reset_index()
                order_all['pct_reversed'] = 100 * order_all.n_reversed / order_all.n_pairs.clip(lower=1)
                order_all.to_csv(model_dir / 'bounds_order_all_folds.csv', index=False)
                print(f"Bounds order for '{model_name}': {int(order_all.n_reversed.sum())}/{int(order_all.n_pairs.sum())} test pairs had min > max before projection.")

            long_parts = [pd.read_csv(d / 'predictions_long.csv') for d in fold_dirs if (d / 'predictions_long.csv').exists()]
            if long_parts:
                pd.concat(long_parts, ignore_index=True).to_parquet(model_dir / 'predictions_long_all.parquet', index=False)

            for variable in variable_names:
                pred_parts: list[pd.DataFrame] = []
                for d in fold_dirs:
                    p = d / f'predictions_{variable}.parquet'
                    c = d / f'predictions_{variable}.csv'
                    if p.exists():
                        pred_parts.append(pd.read_parquet(p))
                    elif c.exists():
                        pred_parts.append(pd.read_csv(c, index_col=0))
                if not pred_parts:
                    continue
                preds_all = pd.concat(pred_parts)
                preds_all.index.name = 'species'
                preds_all.to_parquet(model_dir / f'predictions_{variable}_all.parquet')

            print(f"Saved merged results for '{model_name}' to {model_dir}/")


if __name__ == "__main__":
    args = get_args()
    for configuration in feature_configurations(args):
        run_configuration(configuration)
