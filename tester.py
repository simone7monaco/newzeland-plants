"""
Evaluation pipeline for the TraitsPredictor model.

Part 1 — Transductive Leave-one-trait-out Reconstruction
    For each trait j, hide it *only* for test species, run inference on
    the full graph (training species provide contextual signal), and
    collect predictions.  O(n_traits) forward passes.

    Metrics per trait:
        RMSE, MAE, Pearson r, Spearman rho, Coverage @90%/95%,
        CRPS (Gaussian), Mean Prediction Interval Width.

Part 2 — XAI via Integrated Gradients (Captum)
    Species-side IG  -> importance of each input trait (mean/std),
                        genetic and phylogenetic feature for every
                        output trait.
    Spatial-side IG  -> importance of every environmental / positional
                        feature, disaggregated per test species through
                        the bipartite occurrence edges.

All results are printed, saved to CSV, and plotted to PNG.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import trange
from scipy import stats as sp_stats
import warnings
from torch_geometric.data import Batch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


# =====================================================================
# Part 1 — Leave-one-trait-out reconstruction evaluation
# =====================================================================

@torch.no_grad()
def leave_one_trait_out(model, data, test_indices, device):
    """
    Transductive leave-one-trait-out evaluation on the complete graph.

    For every trait *j* that is observed for a given test species:
      1. Clone the full-graph data.
      2. Zero-out trait *j* for every test species and mark it as missing.
      3. Run a single forward pass on the entire graph (training species
         retain their trait *j* values and provide contextual signal).
      4. Store the prediction for the masked test-species x trait pairs.

    Returns
    -------
    pred_mean  : (n_test, n_traits) predicted means  (NaN where not evaluated)
    pred_std   : (n_test, n_traits) predicted stds
    eval_mask  : (n_test, n_traits) True where trait was masked and evaluated
    """
    model.eval()
    # Support both mean/std and min/max/range representations
    if hasattr(data, 'species_x_mean'):
        trait_mode = 'mean_std'
        n_traits = data.species_x_mean.size(1)
    else:
        trait_mode = 'min_max_range'
        n_traits = data.species_x_min.size(1)
    n_test = len(test_indices)
    observed = ~data.traits_nanmask  # True = originally observed

    pred_mean_out = torch.full((n_test, n_traits), float("nan"), device=device)
    pred_std_out = torch.full((n_test, n_traits), float("nan"), device=device)
    pred_min_out = torch.full((n_test, n_traits), float("nan"), device=device)
    pred_max_out = torch.full((n_test, n_traits), float("nan"), device=device)
    pred_range_out = torch.full((n_test, n_traits), float("nan"), device=device)
    eval_mask = torch.zeros(n_test, n_traits, dtype=torch.bool, device=device)

    for j in trange(n_traits, desc="Leave-one-trait-out"):
        test_has_j = observed[test_indices, j]
        if not test_has_j.any():
            continue

        d = data.clone().to(device)
        mask_global = test_indices[test_has_j]

        # Hide trait j for selected test species
        if trait_mode == 'mean_std':
            d.species_x_mean[mask_global, j] = 0.0
            d.species_x_std[mask_global, j] = 0.0
        else:
            d.species_x_min[mask_global, j] = 0.0
            d.species_x_max[mask_global, j] = 0.0
            d.species_x_range[mask_global, j] = 0.0
        d.traits_nanmask[mask_global, j] = True

        out = model(d)
        # Model may return (mean, std) or (min, max, range)
        if isinstance(out, tuple) and len(out) == 2:
            pm_full, ps_full = out
        elif isinstance(out, tuple) and len(out) == 3:
            pmin_full, pmax_full, pr_full = out
        else:
            raise RuntimeError('Unexpected model output from forward pass')

        local_idx = torch.where(test_has_j)[0]
        if trait_mode == 'mean_std':
            pred_mean_out[local_idx, j] = pm_full[mask_global, j]
            pred_std_out[local_idx, j] = ps_full[mask_global, j]
        else:
            pred_min_out[local_idx, j] = pmin_full[mask_global, j]
            pred_max_out[local_idx, j] = pmax_full[mask_global, j]
            pred_range_out[local_idx, j] = pr_full[mask_global, j]
        eval_mask[local_idx, j] = True

    if trait_mode == 'mean_std':
        return {'mode': trait_mode, 'pred_mean': pred_mean_out, 'pred_std': pred_std_out, 'eval_mask': eval_mask}
    else:
        return {
            'mode': trait_mode,
            'pred_min': pred_min_out,
            'pred_max': pred_max_out,
            'pred_range': pred_range_out,
            'eval_mask': eval_mask,
        }


# -- metrics ---------------------------------------------------------

def _gaussian_crps(mu, sigma, y):
    """Element-wise CRPS for N(mu, sigma^2) evaluated at observation y."""
    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    return sigma * (
        z * (2.0 * sp_stats.norm.cdf(z) - 1.0)
        + 2.0 * sp_stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )


def compute_metrics(pred_mean, pred_std, true_mean, eval_mask, trait_names):
    """
    Per-trait evaluation metrics.

    Returns DataFrame with columns:
        trait, n, RMSE, MAE, Pearson_r, Spearman_rho,
        Coverage_90, Coverage_95, CRPS, MPIW_90, Mean_Pred_Std
    """
    records = []
    n_traits = pred_mean.size(1)

    for j in range(n_traits):
        m = eval_mask[:, j]
        n_eval = int(m.sum().item())
        nan_row = dict(
            trait=trait_names[j], n=n_eval,
            RMSE=np.nan, MAE=np.nan,
            Pearson_r=np.nan, Spearman_rho=np.nan,
            Coverage_90=np.nan, Coverage_95=np.nan,
            CRPS=np.nan, MPIW_90=np.nan, Mean_Pred_Std=np.nan,
        )
        if n_eval < 2:
            records.append(nan_row)
            continue

        p = pred_mean[m, j].cpu().numpy()
        s = pred_std[m, j].cpu().numpy()
        t = true_mean[m, j].cpu().numpy()

        err = p - t
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))

        # Pearson / Spearman
        pr = (
            float(np.corrcoef(p, t)[0, 1])
            if np.std(p) > 1e-12 and np.std(t) > 1e-12
            else np.nan
        )
        sr = float(sp_stats.spearmanr(p, t).statistic) if n_eval > 2 else np.nan # type: ignore

        # Coverage at nominal 90% and 95%
        covs = {}
        for conf, z_val in [(0.90, 1.6449), (0.95, 1.9600)]:
            lo, hi = p - z_val * s, p + z_val * s
            covs[conf] = float(((t >= lo) & (t <= hi)).mean())

        # CRPS and prediction interval width
        crps = float(np.mean(_gaussian_crps(p, s, t)))
        mpiw_90 = float(np.mean(2 * 1.6449 * s))
        mean_s = float(np.mean(s))

        records.append(dict(
            trait=trait_names[j], n=n_eval,
            RMSE=rmse, MAE=mae,
            Pearson_r=pr, Spearman_rho=sr,
            Coverage_90=covs[0.90], Coverage_95=covs[0.95],
            CRPS=crps, MPIW_90=mpiw_90, Mean_Pred_Std=mean_s,
        ))

    return pd.DataFrame(records)


def compute_metrics_minmax(pred_min, pred_max, pred_range, true_min, true_max, true_range, eval_mask, trait_names):
    """
    Compute per-trait, per-variable metrics for min/max/range predictions.
    Returns a DataFrame with rows for each (trait, variable).
    """
    records = []
    n_traits = pred_min.size(1)
    for j in range(n_traits):
        m = eval_mask[:, j]
        n_eval = int(m.sum().item())
        if n_eval < 2:
            for var in ['min', 'max', 'range']:
                if pred_range is None and var == 'range':
                    continue
                records.append(dict(trait=trait_names[j], variable=var, n=n_eval, RMSE=np.nan, MAE=np.nan, Pearson_r=np.nan, Spearman_rho=np.nan))
            continue

        pmin = pred_min[m, j].cpu().numpy()
        pmax = pred_max[m, j].cpu().numpy()
        tmin = true_min[m, j].cpu().numpy()
        tmax = true_max[m, j].cpu().numpy()
        if pred_range is not None:
            pr = pred_range[m, j].cpu().numpy()
            tr = true_range[m, j].cpu().numpy()
            metric_tuples = [('min', pmin, tmin), ('max', pmax, tmax), ('range', pr, tr)]
        else:
            metric_tuples = [('min', pmin, tmin), ('max', pmax, tmax)]

        for name, p, t in metric_tuples:
            err = p - t
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            pr_val = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-12 and np.std(t) > 1e-12 else np.nan
            sr = float(sp_stats.spearmanr(p, t).statistic) if n_eval > 2 else np.nan # type: ignore
            records.append(dict(trait=trait_names[j], variable=name, n=n_eval, RMSE=rmse, MAE=mae, Pearson_r=pr_val, Spearman_rho=sr))

    return pd.DataFrame(records)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample split-conformal quantile for nonconformity scores."""
    if scores.size == 0:
        return float("nan")
    rank = min(int(np.ceil((scores.size + 1) * (1.0 - alpha))), scores.size)
    return float(np.partition(scores, rank - 1)[rank - 1])


def fit_conformal_residual_bounds(
    pred_mean, pred_std, true_mean, calibration_mask, trait_names, alpha=0.10, eps=1e-8,
):
    """Fit trait-wise conformal score bounds on an independent calibration mask."""
    records = []
    for j in range(pred_mean.size(1)):
        mask = calibration_mask[:, j]
        n_calibration = int(mask.sum().item())
        if n_calibration < 2:
            records.append(dict(trait=trait_names[j], alpha=alpha, n_calibration=n_calibration, q_hat=np.nan))
            continue
        score = np.abs(
            (true_mean[mask, j] - pred_mean[mask, j]).detach().cpu().numpy()
        ) / np.maximum(pred_std[mask, j].detach().cpu().numpy(), eps)
        records.append(dict(
            trait=trait_names[j], alpha=alpha, n_calibration=n_calibration,
            q_hat=_conformal_quantile(score, alpha),
        ))
    return pd.DataFrame(records)


def apply_conformal_residual_bounds(
    pred_mean, pred_std, true_mean, eval_mask, trait_names, calibration_bounds, eps=1e-8,
):
    """Apply independently fitted conformal bounds to an evaluation mask."""
    bounds = calibration_bounds.set_index("trait") if calibration_bounds is not None else pd.DataFrame()
    records = []
    reliable_mask = torch.zeros_like(eval_mask, dtype=torch.bool)
    for j in range(pred_mean.size(1)):
        mask = eval_mask[:, j]
        n_eval = int(mask.sum().item())
        bound = bounds.loc[trait_names[j]] if trait_names[j] in bounds.index else None
        q_hat = float(bound["q_hat"]) if bound is not None else float("nan")
        n_calibration = int(bound["n_calibration"]) if bound is not None else 0
        alpha = float(bound["alpha"]) if bound is not None else np.nan
        if n_eval < 1 or not np.isfinite(q_hat):
            records.append(dict(
                trait=trait_names[j], n=n_eval, n_calibration=n_calibration, alpha=alpha,
                q_hat=q_hat, mean_abs_residual=np.nan, empirical_coverage=np.nan,
                calibration_source="inner_validation" if bound is not None else "unavailable",
            ))
            continue
        score = np.abs(
            (true_mean[mask, j] - pred_mean[mask, j]).detach().cpu().numpy()
        ) / np.maximum(pred_std[mask, j].detach().cpu().numpy(), eps)
        within = score <= q_hat
        reliable_mask[mask, j] = torch.as_tensor(within, device=eval_mask.device)
        records.append(dict(
            trait=trait_names[j], n=n_eval, n_calibration=n_calibration, alpha=alpha,
            q_hat=q_hat,
            mean_abs_residual=float(np.mean(np.abs(true_mean[mask, j].detach().cpu().numpy() - pred_mean[mask, j].detach().cpu().numpy()))),
            empirical_coverage=float(np.mean(within)), calibration_source="inner_validation",
        ))
    return pd.DataFrame(records), reliable_mask


def fit_conformal_residual_bounds_minmax(
    pred_min, pred_max, pred_range, true_min, true_max, true_range,
    calibration_mask, trait_names, alpha=0.10,
):
    """Fit absolute-residual conformal bounds for min/max/range on calibration cells."""
    records = []
    for j in range(pred_min.size(1)):
        mask = calibration_mask[:, j]
        n_calibration = int(mask.sum().item())
        for variable, prediction, target in (
            ("min", pred_min, true_min),
            ("max", pred_max, true_max),
            ("range", pred_range, true_range),
        ):
            q_hat = float("nan")
            if n_calibration >= 2:
                scores = np.abs((target[mask, j] - prediction[mask, j]).detach().cpu().numpy())
                q_hat = _conformal_quantile(scores, alpha)
            records.append(dict(
                trait=trait_names[j], variable=variable, alpha=alpha,
                n_calibration=n_calibration, q_hat=q_hat,
            ))
    return pd.DataFrame(records)


def apply_conformal_residual_bounds_minmax(
    pred_min, pred_max, pred_range, true_min, true_max, true_range,
    eval_mask, trait_names, calibration_bounds,
):
    """Apply calibration-derived min/max/range bounds to outer-test cells."""
    bounds = calibration_bounds.set_index(["trait", "variable"]) if calibration_bounds is not None else pd.DataFrame()
    records = []
    reliable_mask = torch.zeros_like(eval_mask, dtype=torch.bool)

    if pred_range is not None:
        target_vars = (
            ("min", pred_min, true_min),
            ("max", pred_max, true_max),
            ("range", pred_range, true_range),
        )
    else:
        target_vars = (
            ("min", pred_min, true_min),
            ("max", pred_max, true_max),
        )

    for j in range(pred_min.size(1)):
        mask = eval_mask[:, j]
        n_eval = int(mask.sum().item())
        within_by_variable = []
        for variable, prediction, target in target_vars:
            key = (trait_names[j], variable)
            bound = bounds.loc[key] if key in bounds.index else None
            q_hat = float(bound["q_hat"]) if bound is not None else float("nan")
            n_calibration = int(bound["n_calibration"]) if bound is not None else 0
            alpha = float(bound["alpha"]) if bound is not None else np.nan
            if n_eval < 1 or not np.isfinite(q_hat):
                within_by_variable.append(np.zeros(n_eval, dtype=bool))
                records.append(dict(
                    trait=trait_names[j], variable=variable, n=n_eval, n_calibration=n_calibration,
                    alpha=alpha, q_hat=q_hat, mean_abs_residual=np.nan, empirical_coverage=np.nan,
                    calibration_source="inner_validation" if bound is not None else "unavailable",
                ))
                continue
            residual = (target[mask, j] - prediction[mask, j]).detach().cpu().numpy()
            within = np.abs(residual) <= q_hat
            within_by_variable.append(within)
            records.append(dict(
                trait=trait_names[j], variable=variable, n=n_eval, n_calibration=n_calibration,
                alpha=alpha, q_hat=q_hat, mean_abs_residual=float(np.mean(np.abs(residual))),
                empirical_coverage=float(np.mean(within)), calibration_source="inner_validation",
            ))
        if n_eval:
            reliable_mask[mask, j] = torch.as_tensor(np.logical_and.reduce(within_by_variable), device=eval_mask.device)
    return pd.DataFrame(records), reliable_mask


# =====================================================================
# Part 2 — XAI: Integrated Gradients
# =====================================================================

def _mask_target_trait_for_imputation(data, test_indices, trait_index, var_names):
    """Apply the same target-trait mask used by leave-one-trait-out evaluation."""

    for var_n in var_names:
        values = getattr(data, f'species_x_{var_n}')
        target_mask = torch.zeros_like(values, dtype=torch.bool)
        target_mask[test_indices, trait_index] = True
        setattr(data, f'species_x_{var_n}', values.masked_fill(target_mask, 0.0))
    traits_nanmask = data.traits_nanmask.clone()
    traits_nanmask[test_indices, trait_index] = True
    data.traits_nanmask = traits_nanmask


def _species_ig(model, data, test_indices, trait_names, device,
                n_steps=50, internal_batch_size=None, gen_col_names=None, var_names=['min', 'max', 'range']):
    """
    IG attributions for **species-side** inputs
    (trait means, trait stds, genetic dummies, phylogenetic embeddings).

    For each output trait *j* the wrapper returns
        pred_mean[test_indices, j]   (shape n_test,)
    and Captum sums over the n_test outputs, so a single IG call yields
    per-input-element attributions aggregated across all test species.

    The target trait is zeroed and marked missing for every test species before
    each IG pass, matching the leave-one-trait-out imputation condition.

    Returns  DataFrame  (n_test x n_traits rows)  x  (input features)
    """
    from captum.attr import IntegratedGradients

    model.eval()

    trait_mode = 'mean_std' if hasattr(data, 'species_x_mean') else 'min_max_range'
    # var_names = ['mean', 'std']

    n_trait_cols = len(var_names)
    pivot_variable = f'species_x_{var_names[0]}'
    n_traits = data[pivot_variable].size(1)
    n_gen = data.species_x_gen.size(1)
    n_phylo = data.species_x_phylo.size(1)

    data_dev = data.to(device)
    n_species_nodes = data.species_num_nodes
    n_spatial_nodes = data.spatial_num_nodes
    n_sp_feat = n_traits * n_trait_cols + n_gen + n_phylo


    for i, x_var in enumerate(var_names):
        assert getattr(data, f"species_x_{x_var}").size(0) == n_species_nodes, \
            f"species_x_{x_var} rows {getattr(data, f'species_x_{x_var}').size(0)} != species_num_nodes {n_species_nodes}"
        if i > 0:
            n_i = data.species_x_std.size(1)
            assert n_traits == n_i, \
                f"trait {var_names[0]}/{var_names[i]} column-count mismatch: {n_traits} vs {n_i}"

    assert data.spatial_x.size(0) == n_spatial_nodes, \
        f"spatial_x rows {data.spatial_x.size(0)} != spatial_num_nodes {n_spatial_nodes}"
    assert data.spatial_global_data.size(0) == n_spatial_nodes, \
        f"spatial_global_data rows {data.spatial_global_data.size(0)} != spatial_num_nodes {n_spatial_nodes}"

    # --- pre-loop: edge-index bounds ---
    if data.species_species_edge_index.numel() > 0:
        assert data.species_species_edge_index.max() < n_species_nodes, \
            "species_species_edge_index out of [0, n_species_nodes)"
    if data.spatial_spatial_edge_index.numel() > 0:
        assert data.spatial_spatial_edge_index.max() < n_spatial_nodes, \
            "spatial_spatial_edge_index out of [0, n_spatial_nodes)"
    assert data.spatial_species_edge_index[0].max() < n_spatial_nodes, \
        "spatial_species_edge_index spatial side out of [0, n_spatial_nodes)"
    assert data.spatial_species_edge_index[1].max() < n_species_nodes, \
        "spatial_species_edge_index species side out of [0, n_species_nodes)"
    assert data.spatial_species_edge_attr.size(0) == data.spatial_species_edge_index.size(1), \
        "spatial_species_edge_attr / edge_index size mismatch"
    assert test_indices.max() < n_species_nodes, \
        f"test_indices max {test_indices.max()} out of [0, n_species_nodes={n_species_nodes})"

    _checked = [False]
    active_target_trait = [0]

    def forward_fn(sp_feat):
        first_call = not _checked[0]
        if first_call:
            assert sp_feat.size(0) % n_species_nodes == 0, (
                f"sp_feat rows {sp_feat.size(0)} not divisible by "
                f"n_species_nodes {n_species_nodes}"
            )
            assert sp_feat.size(1) == n_sp_feat, (
                f"sp_feat cols {sp_feat.size(1)} != expected "
                f"n_mean+n_std+n_gen+n_phylo={n_sp_feat}"
            )

        virtual_batch = sp_feat.size(0) // n_species_nodes

        chunks = sp_feat.split(n_species_nodes, dim=0)
        datas = []
        for i, chunk in enumerate(chunks):
            dtmp = data_dev.clone()
            if trait_mode == 'mean_std':
                dtmp.species_x_mean, dtmp.species_x_std, dtmp.species_x_gen, dtmp.species_x_phylo = torch.split(
                    chunk, [n_traits, n_traits, n_gen, n_phylo], dim=1
                )
            else:
                # min, max, (range,) visibility, gen, phylo
                if 'range' in var_names:
                    dtmp.species_x_min, dtmp.species_x_max, dtmp.species_x_range, dtmp.species_x_gen, dtmp.species_x_phylo = torch.split(
                        chunk, [n_traits, n_traits, n_traits, n_gen, n_phylo], dim=1
                    )
                else:
                    dtmp.species_x_min, dtmp.species_x_max, dtmp.species_x_gen, dtmp.species_x_phylo = torch.split(
                                            chunk, [n_traits, n_traits, n_gen, n_phylo], dim=1
                                        )
            _mask_target_trait_for_imputation(dtmp, test_indices, active_target_trait[0], var_names)
            dtmp.num_nodes = n_species_nodes + data_dev.spatial_num_nodes
            if first_call and i == 0:
                # spatial features must be untouched by species-side perturbation
                assert torch.equal(dtmp.spatial_x.detach(), data_dev.spatial_x.detach()), \
                    "spatial_x was modified inside species IG forward_fn"
                assert torch.equal(
                    dtmp.spatial_global_data.detach(),
                    data_dev.spatial_global_data.detach(),
                ), "spatial_global_data was modified inside species IG forward_fn"
                # all three edge-index tensors must be intact
                assert torch.equal(
                    dtmp.species_species_edge_index,
                    data_dev.species_species_edge_index,
                ), "species_species_edge_index changed inside species IG forward_fn"
                assert torch.equal(
                    dtmp.spatial_spatial_edge_index,
                    data_dev.spatial_spatial_edge_index,
                ), "spatial_spatial_edge_index changed inside species IG forward_fn"
                assert torch.equal(
                    dtmp.spatial_species_edge_index,
                    data_dev.spatial_species_edge_index,
                ), "spatial_species_edge_index changed inside species IG forward_fn"
            datas.append(dtmp)

        _checked[0] = True
        d = Batch.from_data_list(datas)
        out = model(d)
        if isinstance(out, tuple) and len(out) == 2:
            pm = out[0]
            outputs = pm  # shape (total_nodes, n_traits)
        elif isinstance(out, tuple) and len(out) == 3:
            pmin, pmax, pr = out
            # stack variables in order: [min_0..min_N-1, max_0.., range_0..]
            outputs = torch.cat([pmin, pmax, pr], dim=1)
        else:
            raise RuntimeError('Unexpected model output inside species IG forward_fn')
        outputs = outputs.view(virtual_batch, n_species_nodes, -1)[:, test_indices]
        return outputs.reshape(-1, outputs.size(-1))


    sp_input = torch.cat([
        getattr(data, f"species_x_{var_n}") for var_n in var_names
        ] + [data.species_x_gen, data.species_x_phylo],
        dim=1).to(device).requires_grad_(True)

    # TODO: remove here
    # if tait_mode == 'mean_std':
    #     sp_input = torch.cat([
    #         data.species_x_mean, data.species_x_std,
    #         data.species_x_gen, data.species_x_phylo,
    #     ], dim=1).to(device).requires_grad_(True)
    # else:
    #     sp_input = torch.cat([
    #         data.species_x_min, data.species_x_max, data.species_x_range,
    #         data.species_x_gen, data.species_x_phylo,
    #     ], dim=1).to(device).requires_grad_(True)

    baseline = torch.zeros_like(sp_input)
    ig = IntegratedGradients(forward_fn)

    # column names
    g_cols = (
        [f"gen_{c}" for c in gen_col_names]
        if gen_col_names is not None
        else [f"gen_{i}" for i in range(n_gen)]
    )
    phylo_cols = [f"phylo_{i}" for i in range(n_phylo)]

    all_cols = []
    for var_n in var_names:
        all_cols += [f"{var_n}_{t}" for t in trait_names]

    all_cols = all_cols + g_cols + phylo_cols
    species_names = [data.species_names[i] for i in test_indices.cpu().numpy()]

    rows = []
    for var_idx, var_name in enumerate(var_names):
        for j in trange(n_traits, desc=f"Species IG ({var_name})"):
            target = var_idx * n_traits + j
            input_mask = torch.ones_like(sp_input)
            for trait_block in range(n_trait_cols):
                input_mask[test_indices, trait_block * n_traits + j] = 0.0
            active_target_trait[0] = j
            masked_sp_input = (sp_input * input_mask).detach().requires_grad_(True)
            attr = ig.attribute(masked_sp_input, baselines=baseline, target=target, n_steps=n_steps, internal_batch_size=internal_batch_size)
            arr = attr[test_indices].detach().cpu().numpy()
            df_j = pd.DataFrame(arr, index=species_names, columns=all_cols)
            df_j["target_trait"] = trait_names[j]
            df_j["variable"] = var_name
            rows.append(df_j)

    attr_df = pd.concat(rows).reset_index(names="species")

    # collapse phylo embeddings into single mean-|attribution| column
    attr_df["Phylo"] = attr_df[phylo_cols].abs().mean(axis=1)
    attr_df = attr_df.drop(columns=phylo_cols)

    return attr_df


def _spatial_ig(model, data, test_indices, trait_names,
                env_col_names, device, n_steps=30, internal_batch_size=None, var_names=['min', 'max', 'range']):
    """
    IG attributions for **spatial / environmental** features.

    For each output trait *j* the wrapper returns pred_mean[test_indices, j].
    The global spatial IG tensor (n_spatial, n_env) is then disaggregated
    to per-test-species importance by weighting connected spatial nodes
    with their normalised bipartite occurrence edge weights.

    The target trait is zeroed and marked missing for every test species before
    each IG pass, matching the leave-one-trait-out imputation condition.

    Returns  DataFrame  (n_test x n_traits rows)  x  (env features)
    """
    from captum.attr import IntegratedGradients

    model.eval()

    trait_mode = 'mean_std' if hasattr(data, 'species_x_mean') else 'min_max_range'

    n_trait_cols = len(var_names)
    pivot_variable = f'species_x_{var_names[0]}'
    n_traits = data[pivot_variable].size(1)
    n_spatial_x = data.spatial_x.size(1)
    n_global_x = data.spatial_global_data.size(1)

    data_dev = data.to(device)
    n_sp_nodes = data.species_num_nodes
    n_sa_nodes = data.spatial_num_nodes

    # --- pre-loop: node-count consistency ---
    assert data.spatial_x.size(0) == n_sa_nodes, \
        f"spatial_x rows {data.spatial_x.size(0)} != spatial_num_nodes {n_sa_nodes}"
    assert data.spatial_global_data.size(0) == n_sa_nodes, \
        f"spatial_global_data rows {data.spatial_global_data.size(0)} != spatial_num_nodes {n_sa_nodes}"
    assert data[pivot_variable].size(0) == n_sp_nodes, \
        f"{pivot_variable} rows {data[pivot_variable].size(0)} != species_num_nodes {n_sp_nodes}"
    assert data.species_x_gen.size(0) == n_sp_nodes, \
        f"species_x_gen rows {data.species_x_gen.size(0)} != species_num_nodes {n_sp_nodes}"
    assert data.species_x_phylo.size(0) == n_sp_nodes, \
        f"species_x_phylo rows {data.species_x_phylo.size(0)} != species_num_nodes {n_sp_nodes}"
    # --- pre-loop: edge-index bounds ---
    if data.spatial_spatial_edge_index.numel() > 0:
        assert data.spatial_spatial_edge_index.max() < n_sa_nodes, \
            "spatial_spatial_edge_index out of [0, n_sa_nodes)"
    if data.species_species_edge_index.numel() > 0:
        assert data.species_species_edge_index.max() < n_sp_nodes, \
            "species_species_edge_index out of [0, n_sp_nodes)"
    assert data.spatial_species_edge_index[0].max() < n_sa_nodes, \
        "spatial_species_edge_index spatial side out of [0, n_sa_nodes)"
    assert data.spatial_species_edge_index[1].max() < n_sp_nodes, \
        "spatial_species_edge_index species side out of [0, n_sp_nodes)"
    assert data.spatial_species_edge_attr.size(0) == data.spatial_species_edge_index.size(1), \
        "spatial_species_edge_attr / edge_index size mismatch"
    assert test_indices.max() < n_sp_nodes, \
        f"test_indices max {test_indices.max()} out of [0, n_sp_nodes={n_sp_nodes})"

    _checked = [False]
    active_target_trait = [0]

    def forward_fn(spatial_feat):
        first_call = not _checked[0]
        if first_call:
            assert spatial_feat.size(0) % n_sa_nodes == 0, (
                f"spatial_feat rows {spatial_feat.size(0)} not divisible by "
                f"n_sa_nodes {n_sa_nodes}"
            )
            assert spatial_feat.size(1) == n_spatial_x + n_global_x, (
                f"spatial_feat cols {spatial_feat.size(1)} != expected "
                f"n_spatial_x+n_global_x={n_spatial_x + n_global_x}"
            )

        virtual_batch = spatial_feat.size(0) // n_sa_nodes

        chunks = spatial_feat.split(n_sa_nodes, dim=0)
        datas = []
        for i, chunk in enumerate(chunks):
            dtmp = data_dev.clone()
            dtmp.spatial_x, dtmp.spatial_global_data = torch.split(
                chunk, [n_spatial_x, chunk.size(1) - n_spatial_x], dim=1
            )
            _mask_target_trait_for_imputation(dtmp, test_indices, active_target_trait[0], trait_mode)
            dtmp.num_nodes = n_sp_nodes + n_sa_nodes
            if first_call and i == 0:
                expected_pivot = data_dev[pivot_variable].detach().clone()
                expected_pivot[test_indices, active_target_trait[0]] = 0.0
                assert torch.equal(
                    dtmp[pivot_variable].detach(), expected_pivot
                ), f"{pivot_variable} did not preserve the expected target mask"
                assert torch.equal(
                    dtmp.species_x_gen.detach(), data_dev.species_x_gen.detach()
                ), "species_x_gen was modified inside spatial IG forward_fn"
                assert torch.equal(
                    dtmp.species_x_phylo.detach(), data_dev.species_x_phylo.detach()
                ), "species_x_phylo was modified inside spatial IG forward_fn"
                # all three edge-index tensors must be intact
                assert torch.equal(
                    dtmp.species_species_edge_index,
                    data_dev.species_species_edge_index,
                ), "species_species_edge_index changed inside spatial IG forward_fn"
                assert torch.equal(
                    dtmp.spatial_spatial_edge_index,
                    data_dev.spatial_spatial_edge_index,
                ), "spatial_spatial_edge_index changed inside spatial IG forward_fn"
                assert torch.equal(
                    dtmp.spatial_species_edge_index,
                    data_dev.spatial_species_edge_index,
                ), "spatial_species_edge_index changed inside spatial IG forward_fn"
            datas.append(dtmp)

        _checked[0] = True
        d = Batch.from_data_list(datas)
        out = model(d)
        if isinstance(out, tuple) and len(out) == 2:
            pm = out[0]
            outputs = pm
        elif isinstance(out, tuple) and len(out) == 3:
            pmin, pmax, pr = out
            outputs = torch.cat([pmin, pmax, pr], dim=1)
        else:
            raise RuntimeError('Unexpected model output inside spatial IG forward_fn')
        outputs = outputs.view(virtual_batch, n_sp_nodes, -1)[:, test_indices]
        return outputs.reshape(-1, outputs.size(-1))

    spatial_input = torch.cat(
        [data.spatial_x, data.spatial_global_data], dim=1,
    ).to(device).requires_grad_(True)
    baseline = torch.zeros_like(spatial_input)
    ig = IntegratedGradients(forward_fn)

    # build species -> [(spatial_idx, weight)] from bipartite edges
    edge_idx = data.spatial_species_edge_index
    edge_wt = data.spatial_species_edge_attr.squeeze(-1)
    sp2sa: dict[int, list[tuple[int, float]]] = {}
    for e in range(edge_idx.size(1)):
        sa, sp = int(edge_idx[0, e]), int(edge_idx[1, e])
        sp2sa.setdefault(sp, []).append((sa, float(edge_wt[e])))

    species_names = [data.species_names[i] for i in test_indices.cpu().numpy()]
    n_test = len(test_indices)
    n_feat = spatial_input.size(1)

    rows = []
    for var_idx, var_name in enumerate(var_names):
        for j in trange(n_traits, desc=f"Spatial IG ({var_name})"):
            target = var_idx * n_traits + j
            active_target_trait[0] = j
            attr = ig.attribute(spatial_input, baselines=baseline, target=target, n_steps=n_steps, internal_batch_size=internal_batch_size)
            attr_np = attr.detach().cpu().numpy()
            per_sp = np.zeros((n_test, n_feat))
            for li, gi in enumerate(test_indices.cpu().numpy()):
                nbrs = sp2sa.get(int(gi), [])
                if not nbrs:
                    continue
                sa_idx, wts = zip(*nbrs)
                w = np.array(wts)
                w /= w.sum() + 1e-12
                per_sp[li] = (attr_np[list(sa_idx)] * w[:, None]).sum(0)
            df_j = pd.DataFrame(per_sp, index=species_names, columns=env_col_names)
            df_j["target_trait"] = trait_names[j]
            df_j["variable"] = var_name
            rows.append(df_j)

    attr_df = pd.concat(rows).reset_index(names="species")
    return attr_df


# =====================================================================
# Sanity checks
# =====================================================================

def sanity_check_results(
    metrics: pd.DataFrame,
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    true_mean: torch.Tensor,
    eval_mask: torch.Tensor,
    trait_names: list,
    sp_attr: "pd.DataFrame | None" = None,
    sa_attr: "pd.DataFrame | None" = None,
    min_eval_fraction: float = 0.10,
    coverage_tol: float = 0.15,
) -> dict:
    """
    Post-evaluation sanity checks.  Prints colour-coded PASS / WARN / SKIP
    lines and returns a dict with one bool per check (True = passed).

    Checks
    ------
    1.  no_nan_predictions   – no NaN in pred_mean / pred_std at evaluated slots
    2.  beats_mean_baseline  – per-trait RMSE < baseline (predict-zero RMSE in
                               z-normalised space ≈ std(true) ≈ 1.0)
    3.  non_constant_preds   – for each trait, std(pred_mean) > 1e-4
                               (rules out a model that always predicts the same value)
    4.  coverage_calibration – observed 90 / 95 % coverages within ±tol of nominal
    5.  pearson_spearman_sign – Pearson r and Spearman ρ agree in sign for every
                               trait where both are finite
    6.  eval_coverage        – every trait with any observations has at least
                               min_eval_fraction of test species evaluated
    7.  ig_nonzero_species   – (if sp_attr given) mean |IG| > 1e-6 for ≥ 50 %
                               of species feature columns
    8.  ig_target_masked     – (if sp_attr given) all target-trait input
                               attributions are near zero after masking
    9.  ig_nonzero_spatial   – (if sa_attr given) same non-zero check as (7)
    """
    _GREEN  = "\033[92m"
    _YELLOW = "\033[93m"
    _CYAN   = "\033[96m"
    _BOLD   = "\033[1m"
    _RESET  = "\033[0m"

    results: dict[str, bool] = {}

    def _ok(name: str, passed: bool, msg: str = ""):
        results[name] = passed
        colour = _GREEN if passed else _YELLOW
        tag    = "PASS " if passed else "WARN "
        suffix = f"  ({msg})" if msg else ""
        print(f"  {colour}{_BOLD}[{tag}]{_RESET} {name}{suffix}")

    def _skip(name: str, reason: str = ""):
        suffix = f"  ({reason})" if reason else ""
        print(f"  {_CYAN}[SKIP ]{_RESET} {name}{suffix}")

    print(f"\n{_BOLD}====== Sanity checks ======{_RESET}")

    # 1. No NaN at evaluated slots
    pm_np = pred_mean.cpu().numpy()
    ps_np = pred_std.cpu().numpy()
    em_np = eval_mask.cpu().numpy()
    nan_pred = np.isnan(pm_np[em_np]).sum()
    nan_std  = np.isnan(ps_np[em_np]).sum()
    _ok("no_nan_predictions", nan_pred == 0 and nan_std == 0,
        f"{nan_pred} nan means, {nan_std} nan stds in evaluated positions")

    # 2. Beat trivial predict-zero baseline (z-normalised space: baseline RMSE ≈ std(true))
    baseline_fail = []
    for idx, row in metrics.iterrows():
        if np.isnan(row["RMSE"]) or row["n"] < 2:
            continue
        m = em_np[:, idx]
        t = true_mean.cpu().numpy()[m, idx]
        baseline_rmse = float(np.sqrt(np.mean(t ** 2)))  # predict-zero RMSE
        if row["RMSE"] >= baseline_rmse:
            baseline_fail.append(
                f"{row['trait']} (model={row['RMSE']:.3f} vs baseline={baseline_rmse:.3f})"
            )
    _ok("beats_mean_baseline", len(baseline_fail) == 0,
        "; ".join(baseline_fail) if baseline_fail else "")

    # 3. Non-constant predictions
    constant_preds = []
    for idx, row in metrics.iterrows():
        m = em_np[:, idx]
        if m.sum() < 2:
            continue
        p = pm_np[m, idx]
        if np.std(p) < 1e-4:
            constant_preds.append(f"{row['trait']} (std={np.std(p):.2e})")
    _ok("non_constant_preds", len(constant_preds) == 0,
        "; ".join(constant_preds) if constant_preds else "")

    # 5. Coverage calibration within tolerance
    cov_issues = []
    for col, nominal in [("Coverage_90", 0.90), ("Coverage_95", 0.95)]:
        observed = metrics[col].dropna().mean()
        if abs(observed - nominal) > coverage_tol:
            cov_issues.append(
                f"{col}: observed={observed:.3f}, nominal={nominal:.3f}, "
                f"Δ={observed - nominal:+.3f}"
            )
    _ok("coverage_calibration", len(cov_issues) == 0,
        "; ".join(cov_issues) if cov_issues else "")

    # 6. Pearson / Spearman sign agreement
    sign_conflicts = []
    for _, row in metrics.iterrows():
        pr, sr = row["Pearson_r"], row["Spearman_rho"]
        if np.isnan(pr) or np.isnan(sr):
            continue
        if np.sign(pr) != np.sign(sr):
            sign_conflicts.append(
                f"{row['trait']} (Pearson={pr:.3f}, Spearman={sr:.3f})"
            )
    _ok("pearson_spearman_sign", len(sign_conflicts) == 0,
        "; ".join(sign_conflicts) if sign_conflicts else "")

    # 7. Eval coverage per trait
    n_test = em_np.shape[0]
    thin_traits = []
    for idx, row in metrics.iterrows():
        n_obs = int(em_np[:, idx].sum())
        frac = n_obs / max(n_test, 1)
        if n_obs > 0 and frac < min_eval_fraction:
            thin_traits.append(f"{row['trait']} ({n_obs}/{n_test}={frac:.1%})")
    _ok("eval_coverage", len(thin_traits) == 0,
        "; ".join(thin_traits) if thin_traits else "")

    # 8–9. Species-side IG checks
    if sp_attr is not None:
        feat_cols = sp_attr.select_dtypes(include=[np.number]).columns.tolist()

        # 8. Non-zero attributions
        mean_abs = sp_attr[feat_cols].abs().mean()
        nonzero_frac = (mean_abs > 1e-6).mean()
        _ok("ig_nonzero_species", nonzero_frac >= 0.5,
            f"only {nonzero_frac:.1%} of species features have mean |IG| > 1e-6")

        # 9. The target trait must be invisible in leave-one-trait-out IG.
        target_mask_fail = []
        for t in trait_names:
            candidates = [f"mean_{t}", f"std_{t}", f"min_{t}", f"max_{t}", f"range_{t}"]
            subset = sp_attr[sp_attr["target_trait"] == t]
            if subset.empty:
                continue
            for col in candidates:
                if col in subset and float(subset[col].abs().max()) > 1e-6:
                    target_mask_fail.append(f"{t}:{col}")
        _ok("ig_target_masked", len(target_mask_fail) == 0,
            "; ".join(target_mask_fail[:5]) + ("…" if len(target_mask_fail) > 5 else ""))
    else:
        _skip("ig_nonzero_species",  "no species attributions")
        _skip("ig_target_masked", "no species attributions")

    # 10. Spatial IG non-zero
    if sa_attr is not None:
        feat_cols_sa = [c for c in sa_attr.columns if c not in {"species", "target_trait"}]
        mean_abs_sa = sa_attr[feat_cols_sa].abs().mean()
        nonzero_frac_sa = (mean_abs_sa > 1e-6).mean()
        _ok("ig_nonzero_spatial", nonzero_frac_sa >= 0.5,
            f"only {nonzero_frac_sa:.1%} of spatial features have mean |IG| > 1e-6")
    else:
        _skip("ig_nonzero_spatial", "no spatial attributions")

    n_pass = sum(results.values())
    n_total = len(results)
    colour = _GREEN if n_pass == n_total else _YELLOW
    print(f"\n  {colour}{_BOLD}{n_pass}/{n_total} checks passed.{_RESET}")
    return results


def sanity_check_results_minmax(
    metrics: pd.DataFrame,
    pred_min: torch.Tensor,
    pred_max: torch.Tensor,
    pred_range: torch.Tensor,
    true_min: torch.Tensor,
    true_max: torch.Tensor,
    true_range: torch.Tensor,
    eval_mask: torch.Tensor,
    trait_names: list,
    sp_attr: "pd.DataFrame | None" = None,
    sa_attr: "pd.DataFrame | None" = None,
    min_eval_fraction: float = 0.10,
):
    """Simplified sanity checks for min/max/range outputs."""
    _GREEN  = "\033[92m"
    _YELLOW = "\033[93m"
    _CYAN   = "\033[96m"
    _BOLD   = "\033[1m"
    _RESET  = "\033[0m"

    results: dict[str, bool] = {}

    def _ok(name: str, passed: bool, msg: str = ""):
        results[name] = passed
        colour = _GREEN if passed else _YELLOW
        tag    = "PASS " if passed else "WARN "
        suffix = f"  ({msg})" if msg else ""
        print(f"  {colour}{_BOLD}[{tag}]{_RESET} {name}{suffix}")

    def _skip(name: str, reason: str = ""):
        suffix = f"  ({reason})" if reason else ""
        print(f"  {_CYAN}[SKIP ]{_RESET} {name}{suffix}")

    print(f"\n{_BOLD}====== Sanity checks (min/max/range) ======{_RESET}")

    pmin_np = pred_min.cpu().numpy()
    pmax_np = pred_max.cpu().numpy()
    prange_np = pred_range.cpu().numpy()
    em_np = eval_mask.cpu().numpy()
    nan_pred_min = np.isnan(pmin_np[em_np]).sum()
    nan_pred_max = np.isnan(pmax_np[em_np]).sum()
    nan_pred_range = np.isnan(prange_np[em_np]).sum()
    _ok("no_nan_predictions", nan_pred_min == 0 and nan_pred_max == 0 and nan_pred_range == 0,
        f"{nan_pred_min} nan mins, {nan_pred_max} nan maxs, {nan_pred_range} nan ranges in evaluated positions")

    # non-constant predictions per variable
    constant_issues = []
    for j, t in enumerate(trait_names):
        m = em_np[:, j]
        if m.sum() < 2:
            continue
        if np.std(pred_min.cpu().numpy()[m, j]) < 1e-4:
            constant_issues.append(f"{t} min")
        if np.std(pred_max.cpu().numpy()[m, j]) < 1e-4:
            constant_issues.append(f"{t} max")
        if np.std(pred_range.cpu().numpy()[m, j]) < 1e-4:
            constant_issues.append(f"{t} range")
    _ok("non_constant_preds", len(constant_issues) == 0, "; ".join(constant_issues) if constant_issues else "")

    # eval coverage per trait (any variable evaluated)
    n_test = em_np.shape[0]
    thin_traits = []
    for idx, trait in enumerate(trait_names):
        n_obs = int(em_np[:, idx].sum())
        frac = n_obs / max(n_test, 1)
        if n_obs > 0 and frac < min_eval_fraction:
            thin_traits.append(f"{trait} ({n_obs}/{n_test}={frac:.1%})")
    _ok("eval_coverage", len(thin_traits) == 0, "; ".join(thin_traits) if thin_traits else "")

    _skip("coverage_calibration", "not applicable for min/max/range (no predictive std)")
    if sp_attr is not None:
        target_mask_fail = []
        for trait in trait_names:
            subset = sp_attr[sp_attr["target_trait"] == trait]
            for column in (f"min_{trait}", f"max_{trait}", f"range_{trait}"):
                if column in subset and float(subset[column].abs().max()) > 1e-6:
                    target_mask_fail.append(f"{trait}:{column}")
        _ok("ig_target_masked", len(target_mask_fail) == 0,
            "; ".join(target_mask_fail[:5]) + ("…" if len(target_mask_fail) > 5 else ""))
    else:
        _skip("ig_target_masked", "no species attributions")

    n_pass = sum(results.values())
    n_total = len(results)
    colour = _GREEN if n_pass == n_total else _YELLOW
    print(f"\n  {colour}{_BOLD}{n_pass}/{n_total} checks passed.{_RESET}")
    return results


# =====================================================================
# Visualisation helpers
# =====================================================================

def _plot_per_trait_metrics(df, save_path):
    """Grouped bar charts: RMSE, Pearson r, Coverage."""
    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(df) * 0.7), 13))
    x = np.arange(len(df))
    labels = df["trait"].values

    axes[0].bar(x, df["RMSE"], color="steelblue")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Leave-one-trait-out RMSE per trait")

    axes[1].bar(x, df["Pearson_r"], color="coral")
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Leave-one-trait-out Pearson r per trait")
    axes[1].axhline(0, color="k", lw=0.5)

    w = 0.35
    axes[2].bar(x - w / 2, df["Coverage_90"], w, label="90% CI", color="steelblue")
    axes[2].bar(x + w / 2, df["Coverage_95"], w, label="95% CI", color="coral")
    axes[2].axhline(0.90, color="steelblue", ls="--", lw=0.8)
    axes[2].axhline(0.95, color="coral", ls="--", lw=0.8)
    axes[2].set_ylabel("Coverage")
    axes[2].set_title("Prediction interval coverage")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_calibration(df, save_path):
    """Nominal-vs-observed coverage scatter (calibration diagram)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    nominal = [0.90, 0.95]
    observed = [df["Coverage_90"].mean(), df["Coverage_95"].mean()]

    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.4, label="Perfect calibration")
    ax.scatter(nominal, observed, s=90, zorder=5, color="steelblue")
    for n, o in zip(nominal, observed):
        ax.annotate(f"{o:.1%}", (n, o), textcoords="offset points",
                    xytext=(10, -10), fontsize=9)
    ax.set(xlabel="Nominal coverage", ylabel="Observed coverage",
           title="Uncertainty calibration", xlim=(0.5, 1.0), ylim=(0.5, 1.0))
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_scatter_pred_vs_true(pred_mean, true_mean, eval_mask,
                               trait_names, save_path, max_traits=12):
    """Per-trait scatter of predicted vs true values with identity line."""
    n_traits = pred_mean.size(1)
    show_traits = min(n_traits, max_traits)
    cols = min(4, show_traits)
    rows_n = int(np.ceil(show_traits / cols))

    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(4 * cols, 3.5 * rows_n), squeeze=False)
    for j in range(show_traits):
        ax = axes[j // cols, j % cols]
        m = eval_mask[:, j]
        if m.sum() < 2:
            ax.set_title(trait_names[j] + " (n<2)")
            continue
        p = pred_mean[m, j].cpu().numpy()
        t = true_mean[m, j].cpu().numpy()
        ax.scatter(t, p, s=10, alpha=0.6, edgecolors="none")
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.5)
        ax.set(xlabel="True", ylabel="Predicted", title=trait_names[j])

    for idx in range(show_traits, rows_n * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter_pred_vs_true_per_trait(pred_mean, true_mean, eval_mask, trait_names, save_dir):
    """Create one predicted-vs-true scatter figure per trait and save it."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for j, trait_name in enumerate(trait_names):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        m = eval_mask[:, j]
        if m.sum() < 2:
            ax.set_title(trait_name + " (n<2)")
        else:
            p = pred_mean[m, j].cpu().numpy()
            t = true_mean[m, j].cpu().numpy()
            ax.scatter(t, p, s=12, alpha=0.65, edgecolors="none")
            lo = float(min(t.min(), p.min()))
            hi = float(max(t.max(), p.max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.55)
            corr = np.corrcoef(p, t)[0, 1] if np.std(p) > 1e-12 and np.std(t) > 1e-12 else np.nan
            ax.set_title(f"{trait_name} (r={corr:.3f})" if not np.isnan(corr) else trait_name)
        ax.set(xlabel="True", ylabel="Predicted")
        plt.tight_layout()

        path = save_dir / f"scatter_pred_vs_true_{j:03d}_{trait_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def _plot_attribution_heatmap(attr_df, title, save_path, top_k=25):
    """Mean |attribution| heat-map: target_trait x input feature."""
    exclude = {"species", "target_trait", "variable"}
    feat_cols = [c for c in attr_df.columns if c not in exclude]
    grouped = (
        attr_df.groupby("target_trait")[feat_cols]
        .apply(lambda g: g.abs().mean())
    )
    top = grouped.mean(axis=0).nlargest(top_k).index.tolist()
    grouped = grouped[top]

    fig, ax = plt.subplots(
        figsize=(max(10, len(top) * 0.45), max(4, len(grouped) * 0.55))
    )
    sns.heatmap(grouped.astype(float), cmap="viridis", ax=ax, linewidths=0.3)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main entry point
# =====================================================================

class Tester:
    def __init__(self, device):
        self.device = device
        self.metrics = pd.DataFrame()
        self.conformal_bounds = pd.DataFrame()
        self.sp_attr = pd.DataFrame()
        self.sa_attr = pd.DataFrame()
        self.pred_mean = torch.empty(0)
        self.pred_std = torch.empty(0)
        self.true_mean = torch.empty(0)
        self.eval_mask = torch.empty(0)
        self.conformal_reliable_mask = torch.empty(0, dtype=torch.bool)
        self.explainability_eval_mask = torch.empty(0, dtype=torch.bool)
        self.scatterplot_paths: list[Path] = []
        self._orig_mean_dfs: list[pd.DataFrame] = []
        self._orig_std_dfs: list[pd.DataFrame] = []
        # For min/max/range storage
        self._orig_min_dfs: list[pd.DataFrame] = []
        self._orig_max_dfs: list[pd.DataFrame] = []
        self._orig_range_dfs: list[pd.DataFrame] = []

    @torch.no_grad()
    def test_routine(self, model, data, norm_transform, trait_names, device,
                    save_dir="results", compute_xai=True, n_ig_steps=50,
                    ig_internal_batch_size=None,
                    gen_col_names=None, env_col_names=None,
                    conformal_alpha=0.10,
                    conformal_calibration=None,
                    log_wandb=True):
        """
        Full evaluation pipeline (call after training, with best weights loaded).

        Parameters
        ----------
        model          : TraitsPredictor with best weights loaded
        data           : full (unsplit) normalised HeteroData with .test_mask
        norm_transform : NormalizeFeatures  (for inverse normalisation)
        trait_names    : list[str]  trait column names
        device         : torch.device
        save_dir       : str / Path  for all outputs
        compute_xai    : bool  whether to run IG  (Part 2, can be slow)
        n_ig_steps     : int   interpolation steps for Captum IG
        ig_internal_batch_size : int | None  how many IG steps to process per forward pass.
                                None → all n_ig_steps at once (one batch of n_ig_steps full
                                graphs); use a smaller int (e.g. 1) to reduce peak memory
                                at the cost of more forward/backward passes.
        gen_col_names  : list[str] | None  genetic dummy column names
        env_col_names  : list[str] | None  environmental feature column names
        conformal_calibration : pd.DataFrame | None  Bounds fitted on an
                       independent inner-validation mask
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model.to(device).eval()
        data = data.to(device)
        test_indices = torch.where(data.test_mask)[0]
        species_names = [data.species_names[i] for i in test_indices.cpu().numpy()]

        

        # == Part 1: Leave-one-trait-out ==============================
        print("\n====== Part 1: Leave-one-trait-out evaluation ======")
        ret = leave_one_trait_out(model, data, test_indices, device)
        trait_mode = ret['mode']
        self.trait_mode = trait_mode

        if trait_mode == 'mean_std':
            var_names = ['mean', 'std']
        else:
            var_names = ['min', 'max', 'range']
            if not hasattr(data, 'species_x_range'):
                var_names.remove('range')
        

        if trait_mode == 'mean_std':
            pred_mean = ret['pred_mean']
            pred_std = ret['pred_std']
            eval_mask = ret['eval_mask']
            true_mean = data.species_x_mean[test_indices]
            # -- normalised-space metrics --
            metrics = compute_metrics(pred_mean, pred_std, true_mean, eval_mask, trait_names)
        else:
            pred_min = ret['pred_min']
            pred_max = ret['pred_max']
            eval_mask = ret['eval_mask']
            true_min = data.species_x_min[test_indices]
            true_max = data.species_x_max[test_indices]
            if 'range' in var_names:
                pred_range = ret['pred_range']
                true_range = data.species_x_range[test_indices]
            else:
                pred_range = None
                true_range = None
            # -- normalised-space metrics for min/max(/range) --
            metrics = compute_metrics_minmax(pred_min, pred_max, pred_range, true_min, true_max, true_range, eval_mask, trait_names)

        if trait_mode == 'mean_std':
            num_cols = [
                "RMSE", "MAE", "Pearson_r", "Spearman_rho",
                "Coverage_90", "Coverage_95", "CRPS",
            ]
            summary = metrics[num_cols].agg(["mean", "std", "median"])

            print("\nPer-trait metrics (normalised space):")
            print(metrics.to_string(index=False, float_format="%.4f"))
            print("\nAggregate  (mean +/- std):")
            for c in num_cols:
                print(f"  {c:15s}: {summary.loc['mean', c]:.4f} "
                    f"+/- {summary.loc['std', c]:.4f}")

            metrics.to_csv(save_dir / "per_trait_metrics.csv", index=False)
            summary.to_csv(save_dir / "summary_metrics.csv")

            # -- save raw predictions --
            pd.DataFrame(pred_mean.cpu().numpy(), index=species_names, columns=trait_names).to_csv(save_dir / "predictions_mean.csv")
            pd.DataFrame(pred_std.cpu().numpy(), index=species_names, columns=trait_names).to_csv(save_dir / "predictions_std.csv")

            # -- original-space metrics --
            d_tmp = data.clone().cpu()
            d_tmp.species_x_mean = pred_mean.cpu()
            d_tmp.species_x_std = pred_std.cpu()
            d_unnorm = norm_transform.inverse(d_tmp, warn=False, soft_clip=True)
            inv_pred = d_unnorm.species_x_mean
            inv_std = d_unnorm.species_x_std
            inv_true = norm_transform.inverse(data.clone().cpu(), warn=False).species_x_mean[test_indices.cpu()]

            metrics_orig = compute_metrics(inv_pred, inv_std, inv_true, eval_mask.cpu(), trait_names)
            metrics_orig.to_csv(save_dir / "per_trait_metrics_original.csv", index=False)

            conformal_bounds, conformal_reliable_mask = apply_conformal_residual_bounds(
                inv_pred, inv_std, inv_true, eval_mask.cpu(), trait_names, conformal_calibration,
            )
            conformal_bounds.to_csv(save_dir / "conformal_residual_bounds.csv", index=False)
            self.conformal_bounds = conformal_bounds
            self.conformal_reliable_mask = conformal_reliable_mask
            self.explainability_eval_mask = eval_mask.cpu() & conformal_reliable_mask
        else:
            # min_max_range branch: save raw normalized predictions
            pd.DataFrame(pred_min.cpu().numpy(), index=species_names, columns=trait_names).to_csv(save_dir / "predictions_min.csv")
            pd.DataFrame(pred_max.cpu().numpy(), index=species_names, columns=trait_names).to_csv(save_dir / "predictions_max.csv")

            if 'range' in var_names:
                pd.DataFrame(pred_range.cpu().numpy(), index=species_names, columns=trait_names).to_csv(save_dir / "predictions_range.csv")

            # -- original-space metrics --
            d_tmp = data.clone().cpu()
            d_tmp.species_x_min = pred_min.cpu()
            d_tmp.species_x_max = pred_max.cpu()
            if pred_range is not None:
                d_tmp.species_x_range = pred_range.cpu()
            d_unnorm = norm_transform.inverse(d_tmp, warn=False, soft_clip=True)
            inv_min = d_unnorm.species_x_min
            inv_max = d_unnorm.species_x_max
            
            inv_range = d_unnorm.species_x_range if pred_range is not None else None
            data_unnorm = norm_transform.inverse(data.clone().cpu(), warn=False)
            true_min_unn = data_unnorm.species_x_min[test_indices.cpu()]
            true_max_unn = data_unnorm.species_x_max[test_indices.cpu()]
            true_range_unn = data_unnorm.species_x_range[test_indices.cpu()] if pred_range is not None else None


            metrics_orig = compute_metrics_minmax(inv_min, inv_max, inv_range, true_min_unn, true_max_unn, true_range_unn, eval_mask.cpu(), trait_names)
            metrics_orig.to_csv(save_dir / "per_trait_metrics_original.csv", index=False)

            conformal_bounds, conformal_reliable_mask = apply_conformal_residual_bounds_minmax(
                inv_min, inv_max, inv_range, true_min_unn, true_max_unn, true_range_unn,
                eval_mask.cpu(), trait_names, conformal_calibration,
            )
            conformal_bounds.to_csv(save_dir / "conformal_residual_bounds.csv", index=False)
            self.conformal_bounds = conformal_bounds
            self.conformal_reliable_mask = conformal_reliable_mask
            # mark explainability mask where both evaluated and conformally reliable
            self.explainability_eval_mask = eval_mask.cpu() & conformal_reliable_mask

        print("\nOriginal-space metrics:")
        if trait_mode == 'mean_std':
            for _, row in metrics_orig.iterrows():
                print(f"  {row['trait']:25s}:  RMSE={row['RMSE']:.4f}  "
                        f"r={row['Pearson_r']:.4f}")
            print("\nConformal residual bounds (original space):")
            print(conformal_bounds.to_string(index=False, float_format="%.4f"))

            # -- save original-space predictions per fold --
            orig_mean_df = pd.DataFrame(inv_pred.cpu().numpy(), index=species_names, columns=trait_names)
            orig_std_df = pd.DataFrame(inv_std.cpu().numpy(), index=species_names, columns=trait_names)
            orig_mean_df.to_csv(save_dir / "predictions_mean_original.csv")
            orig_std_df.to_csv(save_dir / "predictions_std_original.csv")
            self._orig_mean_dfs.append(orig_mean_df)
            self._orig_std_dfs.append(orig_std_df)
        else:
            for _, row in metrics_orig.iterrows():
                print(f"  {row['trait']:25s} {row['variable']:6s}: RMSE={row['RMSE']:.4f}  r={row['Pearson_r']:.4f}")
            print("\nConformal residual bounds (original space):")
            print(conformal_bounds.to_string(index=False, float_format="%.4f"))

            orig_min_df = pd.DataFrame(inv_min.cpu().numpy(), index=species_names, columns=trait_names)
            orig_max_df = pd.DataFrame(inv_max.cpu().numpy(), index=species_names, columns=trait_names)
            orig_min_df.to_csv(save_dir / "predictions_min_original.csv")
            orig_max_df.to_csv(save_dir / "predictions_max_original.csv")
            self._orig_min_dfs.append(orig_min_df)
            self._orig_max_dfs.append(orig_max_df)
            if inv_range is not None:
                orig_range_df = pd.DataFrame(inv_range.cpu().numpy(), index=species_names, columns=trait_names)
                orig_range_df.to_csv(save_dir / "predictions_range_original.csv")
                self._orig_range_dfs.append(orig_range_df)

        # -- plots (Part 1) --
        if trait_mode == 'mean_std':
            _plot_per_trait_metrics(metrics, save_dir / "metrics_per_trait.png")
            _plot_coverage_calibration(metrics, save_dir / "coverage_calibration.png")
            _plot_scatter_pred_vs_true(pred_mean, true_mean, eval_mask, trait_names, save_dir / "scatter_pred_vs_true.png")
            self.scatterplot_paths = _plot_scatter_pred_vs_true_per_trait(pred_mean, true_mean, eval_mask, trait_names, save_dir / "scatterplots")
        else:
            # For min/max/range, produce variable-specific scatter plots
            _plot_scatter_pred_vs_true(pred_min, true_min, eval_mask, trait_names, save_dir / "scatter_pred_vs_true_min.png")
            _plot_scatter_pred_vs_true(pred_max, true_max, eval_mask, trait_names, save_dir / "scatter_pred_vs_true_max.png")
            paths_min = _plot_scatter_pred_vs_true_per_trait(pred_min, true_min, eval_mask, trait_names, save_dir / "scatterplots_min")
            paths_max = _plot_scatter_pred_vs_true_per_trait(pred_max, true_max, eval_mask, trait_names, save_dir / "scatterplots_max")
            self.scatterplot_paths = paths_min + paths_max 

            if pred_range is not None:
                _plot_scatter_pred_vs_true(pred_range, true_range, eval_mask, trait_names, save_dir / "scatter_pred_vs_true_range.png")
                paths_range = _plot_scatter_pred_vs_true_per_trait(pred_range, true_range, eval_mask, trait_names, save_dir / "scatterplots_range")
                self.scatterplot_paths += paths_range

        if log_wandb and wandb.run is not None:
            wandb_scatter_logs = {}
            if trait_mode == 'mean_std':
                for j, trait_name in enumerate(trait_names):
                    m = eval_mask[:, j]
                    if m.sum() < 2:
                        continue
                    x_values = true_mean[m, j].detach().cpu().numpy().tolist()
                    y_values = pred_mean[m, j].detach().cpu().numpy().tolist()
                    data_t = [[x, y] for x, y in zip(x_values, y_values)]
                    table = wandb.Table(data=data_t, columns=["x", "y"])
                    safe_trait_name = str(trait_name).replace("/", "_")
                    wandb_scatter_logs[f"scatter/{j:03d}_{safe_trait_name}"] = wandb.plot.scatter(table, "x", "y", title=f"{trait_name} predicted vs true")
            else:
                var_targets = [('min', pred_min, true_min), ('max', pred_max, true_max)]
                if 'range' in var_names:
                    var_targets.append(('range', pred_range, true_range))

                for var_name, p, t in var_targets:
                    for j, trait_name in enumerate(trait_names):
                        m = eval_mask[:, j]
                        if m.sum() < 2:
                            continue
                        x_values = t[m, j].detach().cpu().numpy().tolist()
                        y_values = p[m, j].detach().cpu().numpy().tolist()
                        data_t = [[x, y] for x, y in zip(x_values, y_values)]
                        table = wandb.Table(data=data_t, columns=["x", "y"])
                        safe_trait_name = str(trait_name).replace("/", "_")
                        wandb_scatter_logs[f"scatter/{var_name}/{j:03d}_{safe_trait_name}"] = wandb.plot.scatter(table, "x", "y", title=f"{trait_name} {var_name} predicted vs true")

            if wandb_scatter_logs:
                wandb.log(wandb_scatter_logs)

        # == Part 2: Integrated Gradients =============================
        sp_attr: pd.DataFrame | None = None
        sa_attr: pd.DataFrame | None = None

        if not compute_xai:
            print("\nXAI analysis skipped (compute_xai=False).")
        else:
            print("\n====== Part 2: Integrated Gradients attribution ======")

            # --- species-side ---
            sp_attr = _species_ig(
                model, data, test_indices, trait_names, device,
                n_steps=n_ig_steps, internal_batch_size=ig_internal_batch_size, gen_col_names=gen_col_names,
                var_names=var_names
            )
            sp_attr.to_csv(save_dir / "attributions_species.csv", index=False)
            _plot_attribution_heatmap(
                sp_attr,
                "Species-side feature importance  (mean |IG|)",
                save_dir / "heatmap_species_ig.png",
            )
            print(f"  Species attributions saved  ({len(sp_attr)} rows).")

            # --- spatial / environmental side ---
            if model.use_env_features:
                if env_col_names is None:
                    n_pos = data.spatial_x.size(1)
                    n_glob = data.spatial_global_data.size(1)
                    env_col_names = (
                        [f"pos_{i}" for i in range(n_pos)]
                        + [f"env_{i}" for i in range(n_glob)]
                    )
                sa_attr = _spatial_ig(
                    model, data, test_indices, trait_names,
                    env_col_names, device, n_steps=n_ig_steps, internal_batch_size=ig_internal_batch_size,
                    var_names=var_names
                )
                sa_attr.to_csv(save_dir / "attributions_spatial.csv", index=False)
                _plot_attribution_heatmap(
                    sa_attr,
                    "Spatial / Environmental feature importance  (mean |IG|)",
                    save_dir / "heatmap_spatial_ig.png",
                )
                print(f"  Spatial attributions saved  ({len(sa_attr)} rows).")
            else:
                print("  Spatial IG skipped (use_env_features=False).")

            (save_dir / "attributions_metadata.json").write_text(json.dumps({
                "protocol": "leave_one_trait_out_target_masked",
                "target_trait_masked": True,
                "trait_representation": trait_mode,
                "n_ig_steps": n_ig_steps,
            }, indent=2))

        print(f"\nAll results saved to  {save_dir}/")
        if self.metrics.empty:
            self.metrics = metrics
            self.sp_attr = sp_attr
            self.sa_attr = sa_attr
            if trait_mode == 'mean_std':
                self.pred_mean = pred_mean
                self.pred_std = pred_std
                self.true_mean = true_mean
            else:
                self.pred_min = pred_min
                self.pred_max = pred_max
                self.true_min = true_min
                self.true_max = true_max

                if 'range' in var_names:
                    self.pred_range = pred_range
                    self.true_range = true_range
            self.eval_mask = eval_mask
        else:
            self.metrics = pd.concat([self.metrics, metrics], ignore_index=True)
            if sp_attr is not None:
                self.sp_attr = pd.concat([self.sp_attr, sp_attr], ignore_index=True)
            if sa_attr is not None:
                self.sa_attr = pd.concat([self.sa_attr, sa_attr], ignore_index=True)
            if trait_mode == 'mean_std':
                self.pred_mean = torch.cat([self.pred_mean, pred_mean], dim=0) if self.pred_mean is not None else pred_mean
                self.pred_std = torch.cat([self.pred_std, pred_std], dim=0) if self.pred_std is not None else pred_std
                self.true_mean = torch.cat([self.true_mean, true_mean], dim=0) if self.true_mean is not None else true_mean
            else:
                self.pred_min = torch.cat([getattr(self, 'pred_min', torch.empty(0)), pred_min], dim=0) if getattr(self, 'pred_min', None) is not None else pred_min
                self.pred_max = torch.cat([getattr(self, 'pred_max', torch.empty(0)), pred_max], dim=0) if getattr(self, 'pred_max', None) is not None else pred_max
                self.true_min = torch.cat([getattr(self, 'true_min', torch.empty(0)), true_min], dim=0) if getattr(self, 'true_min', None) is not None else true_min
                self.true_max = torch.cat([getattr(self, 'true_max', torch.empty(0)), true_max], dim=0) if getattr(self, 'true_max', None) is not None else true_max
                if pred_range is not None and true_range is not None:
                    self.pred_range = torch.cat([getattr(self, 'pred_range', torch.empty(0)), pred_range], dim=0) if getattr(self, 'pred_range', None) is not None else pred_range
                    self.true_range = torch.cat([getattr(self, 'true_range', torch.empty(0)), true_range], dim=0) if getattr(self, 'true_range', None) is not None else true_range
            self.eval_mask = torch.cat([self.eval_mask, eval_mask], dim=0) if self.eval_mask is not None else eval_mask

        return metrics

    def save_merged_original_predictions(self, save_dir=Path("results")):
        """Merge original-space predictions from all folds and save to CSV."""
        # Support both mean_std and min_max_range stored predictions
        if self._orig_min_dfs:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            merged_min = pd.concat(self._orig_min_dfs)
            merged_max = pd.concat(self._orig_max_dfs)
            merged_min_renamed = merged_min.add_suffix("_min")
            merged_max_renamed = merged_max.add_suffix("_max")
            merged = merged_min_renamed.join(merged_max_renamed)

            if len(self._orig_range_dfs) > 0:
                merged_range = pd.concat(self._orig_range_dfs)
                merged_range_renamed = merged_range.add_suffix("_range")
                merged = merged.join(merged_range_renamed)
            merged.index.name = "species"
            merged.to_csv(save_dir / "predictions_original_all_folds.csv")
            print(f"Merged original-space predictions saved to {save_dir}/predictions_original_all_folds.csv")
            return
        if not self._orig_mean_dfs:
            print("No original-space predictions to merge. Run test_routine() first.")
            return
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        merged_mean = pd.concat(self._orig_mean_dfs)
        merged_std = pd.concat(self._orig_std_dfs)

        # Rename columns to distinguish mean vs std, then join into one file
        merged_mean_renamed = merged_mean.add_suffix("_mean")
        merged_std_renamed = merged_std.add_suffix("_std")
        merged = merged_mean_renamed.join(merged_std_renamed)
        merged.index.name = "species"
        merged.to_csv(save_dir / "predictions_original_all_folds.csv")
        print(f"Merged original-space predictions saved to {save_dir}/predictions_original_all_folds.csv")

    def sanity_checks(self):
        if getattr(self, 'trait_mode', 'mean_std') == 'mean_std':
            self.metrics = self.metrics.groupby("trait").agg({
                "n": "sum",
                "RMSE": "mean",
                "MAE": "mean",
                "Pearson_r": "mean",
                "Spearman_rho": "mean",
                "Coverage_90": "mean",
                "Coverage_95": "mean",
                "CRPS": "mean",
            }).reset_index()

            if self.metrics.empty:
                print("No metrics available for sanity checks. Run test_routine() first.")
                return
            self.sanity_results = sanity_check_results(
                self.metrics, self.pred_mean, self.pred_std, self.true_mean, self.eval_mask,
                self.metrics["trait"].tolist(), sp_attr=self.sp_attr, sa_attr=self.sa_attr,
            )
        else:
            # For min/max/range metrics stored per (trait, variable)
            if self.metrics.empty:
                print("No metrics available for sanity checks. Run test_routine() first.")
                return
            self.sanity_results = sanity_check_results_minmax(
                self.metrics, getattr(self, 'pred_min', torch.empty(0)), getattr(self, 'pred_max', torch.empty(0)), getattr(self, 'pred_range', torch.empty(0)),
                getattr(self, 'true_min', torch.empty(0)), getattr(self, 'true_max', torch.empty(0)), getattr(self, 'true_range', torch.empty(0)),
                self.eval_mask, sorted(set(self.metrics['trait'].tolist())), sp_attr=self.sp_attr, sa_attr=self.sa_attr,
            )
