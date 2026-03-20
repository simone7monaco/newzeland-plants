import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from loader import PlantDataset
from models import DeterministicLoss, TraitsPredictor
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler


def compute_reconstruction_mask(species_mean_df, observed_mask_df, mask_ratio=0.15, mask_strategy='random'):
    """Return a pandas DataFrame boolean mask (True = to be masked) matching the model's logic.

    species_mean_df: pandas DataFrame of shape (n_species, n_traits)
    observed_mask_df: pandas DataFrame bool, True where observation exists (not NaN)
    """
    if mask_ratio <= 0:
        return None

    n_species, n_traits = species_mean_df.shape
    if mask_strategy == 'blockwise':
        n_traits_to_mask = max(1, int(n_traits * mask_ratio))
        traits_to_mask = np.random.permutation(n_traits)[:n_traits_to_mask]
        reconstruction = pd.DataFrame(False, index=species_mean_df.index, columns=species_mean_df.columns)
        reconstruction.iloc[:, traits_to_mask] = True
        reconstruction = reconstruction & observed_mask_df
        return reconstruction

    reconstruction = pd.DataFrame(False, index=species_mean_df.index, columns=species_mean_df.columns)
    if mask_strategy == 'balanced':
        for trait in species_mean_df.columns:
            observed_idx = observed_mask_df[trait][observed_mask_df[trait]].index
            n_observed = len(observed_idx)
            if n_observed > 0:
                n_to_mask = max(1, int(n_observed * mask_ratio))
                chosen = np.random.permutation(n_observed)[:n_to_mask]
                reconstruction.loc[observed_idx[chosen], trait] = True
    else:  # random
        # random selection among all entries, but only keep observed ones
        rand_vals = np.random.rand(n_species, n_traits)
        reconstruction = pd.DataFrame(rand_vals < mask_ratio, index=species_mean_df.index, columns=species_mean_df.columns)
        reconstruction = reconstruction & observed_mask_df

    return reconstruction

def compute_correlation(treats_pred, treats_true, mode='per_feature_mean'):
    """Compute Pearson correlation between predicted and true traits.

    Args:
        treats_pred, treats_true: arrays or torch tensors with shape (N, D).
        mode: 'per_feature_mean' (default) computes Pearson per feature and returns the mean
              'global' computes a single Pearson over all valid entries.

    Returns:
        float: correlation (nan if not computable)
    """
    if isinstance(treats_pred, torch.Tensor):
        treats_pred = treats_pred.cpu().numpy()
    if isinstance(treats_true, torch.Tensor):
        treats_true = treats_true.cpu().numpy()

    # mask positions where either pred or true is NaN
    valid_mask = ~np.isnan(treats_true) & ~np.isnan(treats_pred)

    if mode == 'global':
        flat_mask = valid_mask.ravel()
        if np.sum(flat_mask) < 2:
            return np.nan
        a = treats_pred.ravel()[flat_mask]
        b = treats_true.ravel()[flat_mask]
        # guard against zero-variance
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    # default: per-feature mean
    _, n_traits = treats_pred.shape
    corrs = []
    for j in range(n_traits):
        mask = valid_mask[:, j]
        if np.sum(mask) < 2:
            corrs.append(np.nan)
            continue
        a = treats_pred[mask, j]
        b = treats_true[mask, j]
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            corrs.append(np.nan)
            continue
        corr = np.corrcoef(a, b)[0, 1]
        corrs.append(corr)
    return np.nanmean(corrs)


def impute_with_sklearn(imputer_name, df_values, mask_df):
    """Impute entries indicated by `mask` in `df_values` using sklearn imputers.

    imputer_name: 'mean', 'median', 'most_frequent', or 'knn'
    df_values: torch tensor (n, m) with NaNs for original missing
    mask: boolean torch tensor (n, m) True where entries should be imputed
    """
    # df_values expected as pandas DataFrame
    if isinstance(df_values, torch.Tensor):
        df = pd.DataFrame(df_values.cpu().numpy())
    else:
        df = df_values.copy()

    arr_orig = df.values.astype(np.float64)
    mask_np = mask_df.values.astype(bool)

    # set masked positions to NaN (leave existing NaNs as-is)
    arr = arr_orig.copy()
    arr[mask_np] = np.nan

    if imputer_name in ('mean', 'median', 'most_frequent'):
        imp = SimpleImputer(strategy=imputer_name)
    elif imputer_name == 'knn':
        imp = KNNImputer(n_neighbors=5)
    else:
        raise ValueError(f'Unknown imputer {imputer_name}')

    try:
        imputed = imp.fit_transform(pd.DataFrame(arr, columns=df.columns))
        imputed = np.asarray(imputed, dtype=np.float64)
    except Exception:
        # fallback: if imputer fails (e.g., a column all-NaN), fill masked entries with column nanmean
        imputed = arr.copy()
        col_mean = np.nanmean(arr_orig, axis=0)
        col_mean[np.isnan(col_mean)] = 0.0
        for j in range(arr.shape[1]):
            imputed[mask_np[:, j], j] = col_mean[j]

    # after imputation, still may be NaNs if column entirely NaN -> fill with col mean of original or 0
    col_mean = np.nanmean(arr_orig, axis=0)
    col_mean[np.isnan(col_mean)] = 0.0
    inds = np.where(np.isnan(imputed))
    for i, j in zip(*inds):
        imputed[i, j] = col_mean[j]

    imputed_df = pd.DataFrame(imputed, index=df.index, columns=df.columns)
    return imputed_df

normalize_df = lambda df: pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)
def run_baseline(data_path: str, mask_ratio=0.15, mask_strategy='random', imputers=None):
    imputers = imputers or ['mean', 'median']
    dataset = PlantDataset(Path(data_path))
    traits_mean_df = normalize_df(dataset.traits_mean)
    traits_std_df = normalize_df(dataset.traits_std)
    traits_std_df = traits_std_df.loc[traits_mean_df.index, traits_mean_df.columns]

    # Rizomes for std are all nan: convert to zeros
    traits_std_df[traits_std_df.columns[traits_std_df.isna().all(axis=0)]] = 0.0
    # work with DataFrames for imputation and masking
    observed_mask_df = ~traits_mean_df.isna()

    # create reconstruction mask following model's logic (DataFrame boolean)
    reconstruction_mask_df = compute_reconstruction_mask(traits_mean_df, observed_mask_df, mask_ratio=mask_ratio, mask_strategy=mask_strategy)
    if reconstruction_mask_df is None:
        print('No reconstruction mask created (mask_ratio=0). Nothing to do.')
        return
    results = []
    loss_fn = DeterministicLoss()

    for imp in imputers:
        # prepare DataFrame copies (keep NaNs where originally missing)
        pred_mean_df = traits_mean_df.copy()
        pred_std_df = traits_std_df.copy()

        # use sklearn imputers on DataFrames, imputing only entries indicated by reconstruction_mask_df
        pred_mean_imputed_df = impute_with_sklearn(imp, pred_mean_df, reconstruction_mask_df)
        pred_std_imputed_df = impute_with_sklearn(imp, pred_std_df, reconstruction_mask_df)

        # convert to tensors for loss/coverage computation
        pred_mean = torch.tensor(pred_mean_imputed_df.values.astype(np.float32))
        pred_std = torch.tensor(pred_std_imputed_df.values.astype(np.float32))

        target_mean = torch.tensor(traits_mean_df.values.astype(np.float32))
        target_std = torch.tensor(traits_std_df.values.astype(np.float32))

        mask_for_loss = torch.tensor(reconstruction_mask_df.values, dtype=torch.bool)

        loss = loss_fn(pred_mean, pred_std, target_mean, target_std, mask_for_loss)
        mean_rmse = F.mse_loss(pred_mean[mask_for_loss], target_mean[mask_for_loss], reduction='mean').sqrt()

        nanmask = torch.tensor(traits_mean_df.isna().values, dtype=torch.bool)
        coverage = TraitsPredictor.compute_coverage(pred_mean, pred_std, target_mean, confidence_levels=[0.9, 0.95], nanmask=nanmask)

        correlation = compute_correlation(pred_mean, target_mean)
        cov90 = coverage[0.9].item()
        cov95 = coverage[0.95].item()


        results.append({'imputer': imp, 'mask_strategy': mask_strategy, 'mask_ratio': mask_ratio,
                        'loss': loss.item(), 'mean_rmse': mean_rmse.item(),
                        'correlation': correlation})

    
    results = pd.DataFrame(results)
    print('Baseline results:')
    print(results.drop(columns=['mask_strategy', 'mask_ratio']))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/Ferns', help='Path to dataset root')
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--mask_strategy', type=str, default='random', choices=['random', 'blockwise', 'balanced'])
    parser.add_argument('--imputers', type=str, default='mean,median,knn', help='Comma-separated list of imputers')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    imputers = [i.strip() for i in args.imputers.split(',') if i.strip()]
    run_baseline(args.data, mask_ratio=args.mask_ratio, mask_strategy=args.mask_strategy, imputers=imputers)
