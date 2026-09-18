import argparse
from typing import cast
import warnings
import numpy as np
import torch
from torch_geometric.data import Data
from loader import PlantDataset, data_split
from tester import compute_metrics_minmax
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from Bio import Phylo
from sklearn.impute import SimpleImputer

from r_models import MICE, MissForest, Rphylopars


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
    parser.add_argument('--use_env_features', type=str2bool, nargs='?', const=True, default=False, help='Whether to use environmental features')
    parser.add_argument('--use_phylo_features', type=str2bool, nargs='?', const=True, default=False, help='Whether to use phylogenetic features')    
    parser.add_argument('--invalid_bounds_policy', type=str, default='missing', choices=['missing', 'error', 'keep'], help='How to handle negative or inconsistent min/max/range records')
    parser.add_argument(
        '--baseline_models',
        type=str,
        default='training_mean,training_median,phylo_nn,phylo_knn,rphylopars_bm,mice',
        help=(
            'Comma-separated list of baseline models to run '
            '(options: training_mean, training_median, phylo_nn, phylo_knn, '
            'missforest, mice, rphylopars_bm). The Rphylopars '
            'baseline uses BM with a diagonal phylogenetic covariance for '
            'numerical stability.'
        ),
    )
    parser.add_argument('--k', type=int, default=-1, help='Fold index for cross-validation (0-4). Use -1 to perform a complete run over all folds sequentially.')
    parser.add_argument('--split_strategy', type=str, default='random', choices=['random', 'louvain'], help='Outer CV split: random is transductive; louvain requires balanced graph communities')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used for splitting and model fitting')
    parser.add_argument('--validation_mask_ratio', type=float, default=0.15, help='Fraction of observed outer-training cells held out for inner-validation metrics')
    parser.add_argument('--training_protocol', type=int, default=1, choices=[1, 2], help='Training protocol: 1 = fit on outer-training species only; 2 = fit on outer-training + inner-validation species (transductive)') 
    return parser.parse_args()


def build_baseline_model(name: str, categorical_columns: list[str], seed_value: int):
    """Instantiate a baseline imputer by name."""
    name = name.strip().lower()
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
                       parameter_count: float | None = None) -> pd.DataFrame:
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
    if parameter_count is not None:
        # For AIC-style diagnostics only; see MICE.estimate_parameter_count for how this is derived.
        pd.DataFrame({'k_parameters': [parameter_count]}).to_csv(save_dir / 'model_complexity.csv', index=False)
    for variable in variable_names:
        pd.DataFrame(predictions[variable].numpy(), index=test_species, columns=trait_names).to_csv(
            save_dir / f'predictions_{variable}.csv'
        )
    print(metrics.to_string(index=False, float_format='%.4f'))
    return metrics


def main(args) -> None:
    print(f"---------------\nFitting baselines with args: {args}")

    if args.use_env_features:
        warnings.warn("Environmental features are not used by the baseline models. The --use_env_features flag will be ignored.", UserWarning)
        exit(1)

    data_path = Path('data/Ferns/')
    dataset = PlantDataset(
        data_path,
        transform=None,
        trait_representation='min_max_range',
        traits_filename='FernMinMax.xlsx',
        invalid_bounds_policy=args.invalid_bounds_policy,
    )
    trait_names = dataset.trait_names
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

    train_df = full_df.loc[train_data.species_names]
    test_df = full_df.loc[test_data.species_names]

    tree_path = next(data_path.glob('*.nwk'))
    tree = Phylo.read(tree_path, 'newick')
    tree_tips = {str(tip.name) for tip in tree.get_terminals()}

    baseline_model_names = [name.strip() for name in args.baseline_models.split(',') if name.strip()]

    for model_name in baseline_model_names:
        print(f"\n=== Fitting baseline '{model_name}' (fold {args.k}) ===")
        model = build_baseline_model(model_name, categorical_columns, args.seed + args.k)
        model_train_df = train_df
        model_test_df = test_df
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
                    f"{model_name} removed testing and evaluation species not present "
                    f"in the phylogenetic tree: {missing_test}",
                    UserWarning,
                )
            model_train_df = train_df.drop(index=missing_train)
            model_test_df = test_df.drop(index=missing_test)
            evaluation_species = list(model_test_df.index)

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
            else:
                raise ValueError(f"Unknown model type: {type(model)}")

            for j in range(len(trait_names)):
                columns_to_impute = [trait_col_names[variable][j] for variable in variable_names]
                masked_test_df = model_test_df.copy(deep=True)
                masked_test_df.loc[:, columns_to_impute] = np.nan
                if not isinstance(model, MICE):
                    masked_test_df = masked_test_df.loc[:, trait_columns]

                if isinstance(model, MICE):
                    imputed_df = model.transform(masked_test_df)[0]
                else:
                    imputed_df = model.transform(masked_test_df)
                imputed_df_all.loc[imputed_df.index, columns_to_impute] = imputed_df.loc[:, columns_to_impute].to_numpy()

        else:  # args.training_protocol == 2
            for j in range(len(trait_col_names[variable_names[0]])):
                columns_to_impute = [trait_col_names[variable][j] for variable in variable_names]
                masked_test_df = model_test_df.copy(deep=True)
                masked_test_df.loc[:, columns_to_impute] = np.nan
                masked_df = pd.concat([model_train_df, masked_test_df], axis=0, verify_integrity=True)
                if isinstance(model, MissForest):
                    imputed_df = model.fit_transform(masked_df)
                elif isinstance(model, MICE):
                    imputed_df = model.fit_transform(masked_df)[0]
                    model_parameter_counts.append(model.estimate_parameter_count())
                elif isinstance(model, Rphylopars):
                    imputed_df = model.fit_transform(masked_df.loc[:, trait_columns], tree_path)
                elif isinstance(model, PhylogeneticKNNImputer):
                    imputed_df = model.fit_transform(masked_df.loc[:, trait_columns], tree_path)
                elif isinstance(model, TrainingStatisticImputer):
                    imputed_df = model.fit_transform(masked_df.loc[:, trait_columns])
                else:
                    raise ValueError(f"Unknown model type: {type(model)}")

                imputed_df_all.loc[:, columns_to_impute] = imputed_df.loc[model_test_df.index, columns_to_impute].to_numpy()

        save_dir = args.output_dir / f"{model_name.upper()}{args.model_dir_suffix}" / f'fold_{args.k}'
        evaluate_and_save(imputed_df_all, test_data, trait_names, trait_col_names, variable_names,
                           save_dir, evaluation_species,
                           parameter_count=float(np.mean(model_parameter_counts)) if model_parameter_counts else None)


"""
TODO
Aggiungere variante Training test protocol su MissForest (unico protocollo possibile) e MICE

## Fully transductive matrix imputation
For each variable j, fit_transform(train + test_j_masked)
"""


if __name__ == "__main__":
    args = get_args()
    seed = args.seed
    pl.seed_everything(seed)
    baseline_model_names = [name.strip() for name in args.baseline_models.split(',') if name.strip()]
    baseline_model_list = "\n- " + "\n- ".join(baseline_model_names) if baseline_model_names else "No baseline models specified"

    print(f"""
+------------------------------------------------------
| Starting baselines with models:
| {baseline_model_list}
|
""")

    args.model_dir_suffix = f"_prot{args.training_protocol}"
    if args.use_env_features:
        args.model_dir_suffix += "_env"
    if args.use_phylo_features:
        args.model_dir_suffix += "_phylo"

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
            model_dir = args.output_dir / f"{model_name.upper()}{args.model_dir_suffix}"
            fold_dirs = sorted(model_dir.glob('fold_*'))

            metrics_all = pd.concat([pd.read_csv(d / 'per_trait_metrics.csv') for d in fold_dirs], ignore_index=True)
            metrics_all.to_csv(model_dir / 'per_trait_metrics_all.csv', index=False)

            for variable in variable_names:
                preds_all = pd.concat([pd.read_csv(d / f'predictions_{variable}.csv', index_col=0) for d in fold_dirs])
                preds_all.index.name = 'species'
                preds_all.to_csv(model_dir / f'predictions_{variable}_all.csv')

            print(f"Saved merged results for '{model_name}' to {model_dir}/")
