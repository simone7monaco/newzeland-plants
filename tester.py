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
from pathlib import Path
from tqdm import trange
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


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
    n_traits = data.species_x_mean.size(1)
    n_test = len(test_indices)
    observed = ~data.traits_nanmask  # True = originally observed

    pred_mean_out = torch.full((n_test, n_traits), float("nan"), device=device)
    pred_std_out = torch.full((n_test, n_traits), float("nan"), device=device)
    eval_mask = torch.zeros(n_test, n_traits, dtype=torch.bool, device=device)

    for j in trange(n_traits, desc="Leave-one-trait-out"):
        test_has_j = observed[test_indices, j]
        if not test_has_j.any():
            continue

        d = data.clone().to(device)
        mask_global = test_indices[test_has_j]

        # Hide trait j for selected test species
        d.species_x_mean[mask_global, j] = 0.0
        d.species_x_std[mask_global, j] = 0.0
        d.traits_nanmask[mask_global, j] = True

        pm, ps = model(d)

        local_idx = torch.where(test_has_j)[0]
        pred_mean_out[local_idx, j] = pm[mask_global, j]
        pred_std_out[local_idx, j] = ps[mask_global, j]
        eval_mask[local_idx, j] = True

    return pred_mean_out, pred_std_out, eval_mask


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
        sr = float(sp_stats.spearmanr(p, t).statistic) if n_eval > 2 else np.nan

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


# =====================================================================
# Part 2 — XAI: Integrated Gradients
# =====================================================================

def _species_ig(model, data, test_indices, trait_names, device,
                n_steps=50, gen_col_names=None):
    """
    IG attributions for **species-side** inputs
    (trait means, trait stds, genetic dummies, phylogenetic embeddings).

    For each output trait *j* the wrapper returns
        pred_mean[test_indices, j]   (shape n_test,)
    and Captum sums over the n_test outputs, so a single IG call yields
    per-input-element attributions aggregated across all test species.

    Returns  DataFrame  (n_test x n_traits rows)  x  (input features)
    """
    from captum.attr import IntegratedGradients

    model.eval()
    n_mean = data.species_x_mean.size(1)
    n_std = data.species_x_std.size(1)
    n_gen = data.species_x_gen.size(1)
    n_phylo = data.species_x_phylo.size(1)
    n_traits = n_mean

    data_dev = data.to(device)

    def forward_fn(sp_feat):
        d = data_dev.clone()
        i = 0
        d.species_x_mean = sp_feat[:, i : i + n_mean]; i += n_mean
        d.species_x_std = sp_feat[:, i : i + n_std]; i += n_std
        d.species_x_gen = sp_feat[:, i : i + n_gen]; i += n_gen
        d.species_x_phylo = sp_feat[:, i:]
        pm, _ = model(d)
        return pm[test_indices]  # (n_test, n_traits)

    sp_input = torch.cat([
        data.species_x_mean, data.species_x_std,
        data.species_x_gen, data.species_x_phylo,
    ], dim=1).to(device).requires_grad_(True)

    baseline = torch.zeros_like(sp_input)
    ig = IntegratedGradients(forward_fn)

    # column names
    mean_cols = [f"mean_{t}" for t in trait_names]
    std_cols = [f"std_{t}" for t in trait_names]
    g_cols = (
        [f"gen_{c}" for c in gen_col_names]
        if gen_col_names is not None
        else [f"gen_{i}" for i in range(n_gen)]
    )
    phylo_cols = [f"phylo_{i}" for i in range(n_phylo)]
    all_cols = mean_cols + std_cols + g_cols + phylo_cols

    species_names = [data.species_names[i] for i in test_indices.cpu().numpy()]

    rows = []
    for j in trange(n_traits, desc="Species IG"):
        attr = ig.attribute(
            sp_input, baselines=baseline, target=j, n_steps=n_steps,
        )
        arr = attr[test_indices].detach().cpu().numpy()
        df_j = pd.DataFrame(arr, index=species_names, columns=all_cols)
        df_j["target_trait"] = trait_names[j]
        rows.append(df_j)

    attr_df = pd.concat(rows).reset_index(names="species")

    # collapse phylo embeddings into single mean-|attribution| column
    attr_df["Phylo"] = attr_df[phylo_cols].abs().mean(axis=1)
    attr_df = attr_df.drop(columns=phylo_cols)

    return attr_df


def _spatial_ig(model, data, test_indices, trait_names,
                env_col_names, device, n_steps=30):
    """
    IG attributions for **spatial / environmental** features.

    For each output trait *j* the wrapper returns pred_mean[test_indices, j].
    The global spatial IG tensor (n_spatial, n_env) is then disaggregated
    to per-test-species importance by weighting connected spatial nodes
    with their normalised bipartite occurrence edge weights.

    Returns  DataFrame  (n_test x n_traits rows)  x  (env features)
    """
    from captum.attr import IntegratedGradients

    model.eval()
    n_spatial_x = data.spatial_x.size(1)
    n_traits = data.species_x_mean.size(1)

    data_dev = data.to(device)

    def forward_fn(spatial_feat):
        d = data_dev.clone()
        d.spatial_x = spatial_feat[:, :n_spatial_x]
        d.spatial_global_data = spatial_feat[:, n_spatial_x:]
        pm, _ = model(d)
        return pm[test_indices]

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
    for j in trange(n_traits, desc="Spatial IG"):
        attr = ig.attribute(
            spatial_input, baselines=baseline, target=j, n_steps=n_steps,
        )
        attr_np = attr.detach().cpu().numpy()  # (n_spatial, n_feat)

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
        rows.append(df_j)

    attr_df = pd.concat(rows).reset_index(names="species")
    return attr_df


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


def _plot_attribution_heatmap(attr_df, title, save_path, top_k=25):
    """Mean |attribution| heat-map: target_trait x input feature."""
    exclude = {"species", "target_trait"}
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

@torch.no_grad()
def test_routine(model, data, norm_transform, trait_names, device,
                 save_dir="results", compute_xai=True, n_ig_steps=50,
                 gen_col_names=None, env_col_names=None):
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
    gen_col_names  : list[str] | None  genetic dummy column names
    env_col_names  : list[str] | None  environmental feature column names
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.to(device).eval()
    data = data.to(device)
    test_indices = torch.where(data.test_mask)[0]
    species_names = [data.species_names[i] for i in test_indices.cpu().numpy()]

    # == Part 1: Leave-one-trait-out ==============================
    print("\n====== Part 1: Leave-one-trait-out evaluation ======")
    pred_mean, pred_std, eval_mask = leave_one_trait_out(
        model, data, test_indices, device,
    )

    true_mean = data.species_x_mean[test_indices]

    # -- normalised-space metrics --
    metrics = compute_metrics(
        pred_mean, pred_std, true_mean, eval_mask, trait_names,
    )
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
    pd.DataFrame(
        pred_mean.cpu().numpy(), index=species_names, columns=trait_names,
    ).to_csv(save_dir / "predictions_mean.csv")
    pd.DataFrame(
        pred_std.cpu().numpy(), index=species_names, columns=trait_names,
    ).to_csv(save_dir / "predictions_std.csv")

    # -- original-space metrics --
    d_unnorm = norm_transform.inverse(
        data.clone().to(device)._replace(species_x_mean=pred_mean, species_x_std=pred_std)
    ).species_x_mean.cpu()
    inv_pred = d_unnorm.species_x_mean
    inv_std = d_unnorm.species_x_std
    inv_true = norm_transform.inverse(data.clone().to(device)).species_x_mean.cpu()[test_indices]


    metrics_orig = compute_metrics(
        inv_pred, inv_std, inv_true, eval_mask, trait_names,
    )
    metrics_orig.to_csv(
        save_dir / "per_trait_metrics_original.csv", index=False,
    )
    print("\nOriginal-space metrics:")
    for _, row in metrics_orig.iterrows():
        print(f"  {row['trait']:25s}:  RMSE={row['RMSE']:.4f}  "
                f"r={row['Pearson_r']:.4f}")

    # -- plots (Part 1) --
    _plot_per_trait_metrics(metrics, save_dir / "metrics_per_trait.png")
    _plot_coverage_calibration(metrics, save_dir / "coverage_calibration.png")
    _plot_scatter_pred_vs_true(
        pred_mean, true_mean, eval_mask, trait_names,
        save_dir / "scatter_pred_vs_true.png",
    )

    # == Part 2: Integrated Gradients =============================
    if not compute_xai:
        print("\nXAI analysis skipped (compute_xai=False).")
        return metrics

    print("\n====== Part 2: Integrated Gradients attribution ======")

    # --- species-side ---
    sp_attr = _species_ig(
        model, data, test_indices, trait_names, device,
        n_steps=n_ig_steps, gen_col_names=gen_col_names,
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
            env_col_names, device, n_steps=n_ig_steps,
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

    print(f"\nAll results saved to  {save_dir}/")
    return metrics
