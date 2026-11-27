"""
Circular phylogeny with concentric data rings and embedding-vs-phylogeny panels.

Panel a : fan-shaped tree, branches coloured by major clade, followed by rings
          [clade band | embedding PC1 | embedding PC2 | per-species trait bars]
          and italic tip labels.
Panel b : embedding space (PC1 vs PC2), coloured by clade, 1.5-sigma ellipses.
Panel c : patristic distance vs embedding distance (hexbin + running median),
          with Spearman rho and a Mantel permutation p-value.

Without `embeddings`, only panel a (tree + clade band + optional trait ring) is drawn.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

# Okabe-Ito (colour-blind safe); repeated if more clades are requested.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#7A5195", "#6B6B6B"]
BACKBONE = "#4d4d4d"
MISSING = "#e6e6e6"


# --------------------------------------------------------------------------- #
# Tree utilities
# --------------------------------------------------------------------------- #
def load_pruned_tree(tree_file, species):
    """Read a Newick file, keep only `species`, ladderize. Returns (tree, absent)."""
    tree = Phylo.read(tree_file, "newick")
    species = set(species)
    tree_species = {leaf.name for leaf in tree.get_terminals()}
    included = species & tree_species
    if len(included) < 2:
        raise ValueError("Fewer than two dataset species match the tree tip names.")
    for leaf in list(tree.get_terminals()):
        if leaf.name not in included:
            tree.prune(leaf)
    tree.ladderize()
    return tree, species - tree_species


def ensure_branch_lengths(tree):
    """Fall back to unit-length edges when lengths are incomplete."""
    branches = [c for c in tree.find_clades() if c is not tree.root]
    if any(c.branch_length is None for c in branches):
        for c in branches:
            c.branch_length = 1.0
        return "Topological distance (edges)"
    return "Phylogenetic branch length"


def node_depths(tree):
    depth = {tree.root: 0.0}
    for parent in tree.find_clades(order="preorder"):
        for ch in parent.clades:
            depth[ch] = depth[parent] + (ch.branch_length or 0.0)
    return depth


def split_major_clades(tree, k):
    """Greedily split the largest clade until k groups exist."""
    groups = [tree.root]
    while len(groups) < k:
        splittable = [g for g in groups if not g.is_terminal()]
        if not splittable:
            break
        g = max(splittable, key=lambda c: c.count_terminals())
        groups.remove(g)
        groups.extend(g.clades)
    return groups


def patristic_matrix(tree, tips, depth):
    """All tip-to-tip path lengths in O(n^2) via node depths."""
    n = len(tips)
    idx = {t: i for i, t in enumerate(tips)}
    tipidx = {}
    for c in tree.find_clades(order="postorder"):
        tipidx[c] = (np.array([idx[c]]) if c.is_terminal()
                     else np.concatenate([tipidx[ch] for ch in c.clades]))
    td = np.array([depth[t] for t in tips])
    D = np.zeros((n, n))
    for c in tree.get_nonterminals():
        for a in range(len(c.clades)):
            for b in range(a + 1, len(c.clades)):
                ia, ib = tipidx[c.clades[a]], tipidx[c.clades[b]]
                blk = td[ia][:, None] + td[ib][None, :] - 2 * depth[c]
                D[np.ix_(ia, ib)] = blk
                D[np.ix_(ib, ia)] = blk.T
    return D


# --------------------------------------------------------------------------- #
# Embedding utilities
# --------------------------------------------------------------------------- #
def pca_scores(X, k):
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, len(S))
    return U[:, :k] * S[:k], (S ** 2 / (S ** 2).sum())[:k]


def embedding_distance(X, metric="cosine"):
    if metric == "cosine":
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
        D = 1.0 - Xn @ Xn.T
    elif metric == "euclidean":
        sq = (X ** 2).sum(1)
        D = np.sqrt((sq[:, None] + sq[None, :] - 2 * X @ X.T).clip(0))
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")
    D = D.clip(0)
    np.fill_diagonal(D, 0.0)
    return D


def _rank(x):
    return pd.Series(x).rank().to_numpy()


def mantel_spearman(D1, D2, n_perm=999, rng=None):
    """Spearman rho between two distance matrices + one-sided permutation p."""
    rng = rng or np.random.default_rng(0)
    n = D1.shape[0]
    iu = np.triu_indices(n, 1)
    r1, r2 = _rank(D1[iu]), _rank(D2[iu])
    z1 = (r1 - r1.mean()) / r1.std()
    z2v = (r2 - r2.mean()) / r2.std()
    rho = float((z1 * z2v).mean())
    Z2 = np.zeros((n, n))
    Z2[iu] = z2v
    Z2 = Z2 + Z2.T
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        hits += (z1 * Z2[np.ix_(p, p)][iu]).mean() >= rho
    return rho, (1 + hits) / (n_perm + 1)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _nice_step(x):
    e = 10 ** np.floor(np.log10(x))
    return min((m * e for m in (1, 2, 5, 10) if m * e >= x), default=x)


def _axes_radius_pts(fig, ax):
    pos = ax.get_position()
    w, h = fig.get_size_inches()
    return 0.5 * min(w * pos.width, h * pos.height) * 72


def _to_unit(v, lo=2, hi=98):
    a, b = np.nanpercentile(v, [lo, hi])
    return np.clip((v - a) / (b - a + 1e-12), 0, 1)


def _arc(r, t0, t1, step=0.01):
    tt = np.linspace(t0, t1, max(2, int(np.ceil((t1 - t0) / step))) + 1)
    return list(zip(tt, np.full_like(tt, r)))


# --------------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------------- #
def plot_phylo_figure(
    tree,
    embeddings: pd.DataFrame | None = None,
    trait: pd.Series | None = None,
    *,
    n_clades=6,
    clade_names: dict | None = None,
    trait_label="Trait",
    distance_label="Phylogenetic branch length",
    title=None,
    cmap="viridis",
    n_pcs=2,
    dist_metric="cosine",
    n_perm=999,
    gap_deg=14.0,
    label_tips=True,
    figsize=None,
    seed=0,
    plot_bc=False,
):
    """
    embeddings : DataFrame (species x dims), index = tip names. Missing species are greyed.
    trait      : Series indexed by tip name (scalar per species) -> outer bar ring.
    """
    rng = np.random.default_rng(seed)
    clade_names = clade_names or {}
    n_pcs = max(2, n_pcs)

    tips = tree.get_terminals()
    n = len(tips)
    names = [t.name for t in tips]
    depth = node_depths(tree)
    R = max(depth[t] for t in tips)

    # ---- clades --------------------------------------------------------- #
    tip_pos = {t: i for i, t in enumerate(tips)}
    groups = split_major_clades(tree, n_clades)
    groups.sort(key=lambda g: min(tip_pos[t] for t in g.get_terminals()))
    colours = (PALETTE * (len(groups) // len(PALETTE) + 1))[:len(groups)]
    clade_of = {c: gi for gi, g in enumerate(groups) for c in g.find_clades()}
    tip_group = np.array([clade_of[t] for t in tips])

    # ---- data ----------------------------------------------------------- #
    has_emb = embeddings is not None
    if has_emb:
        X = embeddings.reindex(names).to_numpy(float)
        valid = ~np.isnan(X).any(1)
        if valid.sum() < 3:
            raise ValueError("Fewer than three species have embeddings.")
        scores = np.full((n, n_pcs), np.nan)
        sc, var = pca_scores(X[valid], n_pcs)
        scores[valid, :sc.shape[1]] = sc
    tvals = None
    if trait is not None:
        tvals = trait.reindex(names).to_numpy(float)

    # ---- figure / axes ---------------------------------------------------#
    if has_emb:
        fig = plt.figure(figsize=figsize or (13.5, 7.6))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1.45, 1], hspace=0.34,
                      wspace=0.12, left=0.02, right=0.975, top=0.955, bottom=0.07)
        ax = fig.add_subplot(gs[:, 0], projection="polar")
    else:
        fig = plt.figure(figsize=figsize or (8.6, 8.6))
        ax = fig.add_subplot(111, projection="polar")
        fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.06)

    # ---- radial geometry -------------------------------------------------#
    gap = np.deg2rad(gap_deg)
    theta_tips = np.linspace(gap / 2, 2 * np.pi - gap / 2, n)
    dth = (theta_tips[1] - theta_tips[0]) if n > 1 else 0.1
    theta = {t: theta_tips[i] for i, t in enumerate(tips)}
    for node in tree.find_clades(order="postorder"):
        if not node.is_terminal():
            a = [theta[c] for c in node.clades]
            theta[node] = 0.5 * (min(a) + max(a))

    rings = [("clade", 0.035 * R)]
    if has_emb:
        rings += [(f"PC{k + 1}", 0.055 * R) for k in range(n_pcs)]
    if tvals is not None:
        rings += [(trait_label, 0.10 * R)]
    ring_gap = 0.012 * R
    r_cursor = R * 1.02
    ring_r0 = []
    for _, w in rings:
        ring_r0.append(r_cursor)
        r_cursor += w + ring_gap
    r_lab = r_cursor + 0.01 * R

    P = _axes_radius_pts(fig, ax)
    fs = 6.0
    rmax = r_lab * 1.03
    if label_tips:
        for _ in range(3):
            L = max(len(s) for s in names) * 0.52 * fs
            rmax = r_lab / (1 - min(1.06 * L / P, 0.5))
            circ = 2 * np.pi * P * (r_lab / rmax) * (1 - gap_deg / 360)
            fs = float(np.clip(0.85 * circ / n, 3.0, 8.0))

    # ---- tree branches ---------------------------------------------------#
    lw = float(np.clip(60 / n, 0.35, 1.2))
    segs, cols = [], []
    for node in tree.find_clades(order="preorder"):
        if node.is_terminal():
            continue
        r_n = depth[node]
        th = [theta[c] for c in node.clades]
        if r_n > 0 and len(th) > 1:
            segs.append(_arc(r_n, min(th), max(th)))
            cols.append(colours[clade_of[node]] if node in clade_of else BACKBONE)
        for ch in node.clades:
            segs.append([(theta[ch], r_n), (theta[ch], depth[ch])])
            cols.append(colours[clade_of[ch]] if ch in clade_of else BACKBONE)
    lc = LineCollection(segs, colors=cols, linewidths=lw, zorder=3)
    lc.set_capstyle("round")
    ax.add_collection(lc, autolim=False)

    # guides from tip to ring start + distance grid
    guides = [[(theta[t], depth[t]), (theta[t], R * 1.015)]
              for t in tips if depth[t] < R * 0.995]
    if guides:
        ax.add_collection(LineCollection(guides, colors="#b5b5b5", linewidths=0.3,
                                         linestyles=(0, (1, 2)), zorder=1),
                          autolim=False)
    step = _nice_step(R / 3)
    for tk in np.arange(step, R * 0.999, step):
        ax.add_collection(LineCollection([_arc(tk, theta_tips[0], theta_tips[-1])],
                                         colors="#dcdcdc", linewidths=0.4, zorder=0),
                          autolim=False)
        ax.text(0, tk, f"{tk:g}", rotation=90, ha="center", va="center",
                fontsize=5.5, color="#8a8a8a")

    # ---- rings -----------------------------------------------------------#
    cm = mpl.colormaps[cmap]
    grp_cols = [colours[g] for g in tip_group]
    for (name, w), r0 in zip(rings, ring_r0):
        if name == "clade":
            c = grp_cols
            ax.bar(theta_tips, w, width=dth, bottom=r0, color=c, edgecolor=c,
                   linewidth=0.3, zorder=2)
        elif name.startswith("PC"):
            k = int(name[2:]) - 1
            u = _to_unit(scores[:, k])
            c = [MISSING if np.isnan(x) else cm(x) for x in u]
            ax.bar(theta_tips, w, width=dth, bottom=r0, color=c, edgecolor=c,
                   linewidth=0.3, zorder=2)
        else:  # trait bars
            u = np.nan_to_num(_to_unit(tvals, 0, 100), nan=0.0)
            ax.add_collection(LineCollection([_arc(r0, theta_tips[0], theta_tips[-1])],
                                             colors="#9a9a9a", linewidths=0.4, zorder=2),
                              autolim=False)
            ax.bar(theta_tips, w * (0.04 + 0.96 * u), width=dth * 0.82, bottom=r0,
                   color=grp_cols, edgecolor="none", alpha=0.9, zorder=2)
        ax.text(0, r0 + w / 2, name, rotation=90, ha="center", va="center",
                fontsize=6, color="#333333")

    # ---- tip labels ------------------------------------------------------#
    if label_tips:
        for t, th in zip(tips, theta_tips):
            deg = np.degrees(th)
            flip = 90 < deg < 270
            ax.text(th, r_lab, t.name.replace("_", " "),
                    rotation=deg + 180 if flip else deg, rotation_mode="anchor",
                    ha="right" if flip else "left", va="center",
                    fontsize=fs, fontstyle="italic", color="#222222")

    ax.set_ylim(0, rmax)
    ax.set_axis_off()

    # ---- legend / colourbar / caption -------------------------------------#
    handles = [Line2D([], [], marker="s", ls="", ms=6, color=colours[gi],
                      label=f"{clade_names.get(gi, f'Clade {gi + 1}')} "
                            f"(n={(tip_group == gi).sum()})")
               for gi in range(len(groups))]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, -0.01),
                    frameon=False, fontsize=7, title="Major clades", title_fontsize=7.5,
                    handletextpad=0.3, labelspacing=0.3)
    leg._legend_box.align = "left"
    if has_emb:
        cax = ax.inset_axes([0.79, 0.03, 0.19, 0.012])
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 1), cm),
                          cax=cax, orientation="horizontal")
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["low", "high"], fontsize=6)
        cb.outline.set_linewidth(0.3)
        cax.tick_params(length=2, width=0.3, pad=1)
        cax.set_title("Embedding PC score", fontsize=6.5, pad=2)
    ax.text(0.5, -0.015, f"Grey rings: {distance_label.lower()}", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.5, color="#666666")
    if plot_bc:
        ax.text(0.0, 1.0, "a", transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top")
    if title:
        ax.set_title(title, fontsize=10, pad=2, loc="center")

    info = {"n_tips": n, "groups": groups, "colours": colours}

    # ---- panels b / c ----------------------------------------------------#
    if has_emb and plot_bc:
        axB = fig.add_subplot(gs[0, 1])
        v = valid
        axB.scatter(scores[v, 0], scores[v, 1], c=[grp_cols[i] for i in np.flatnonzero(v)],
                    s=20, edgecolors="white", linewidths=0.4, zorder=3)
        for gi in range(len(groups)):
            m = v & (tip_group == gi)
            if m.sum() < 4:
                continue
            pts = scores[m, :2]
            w_, V = np.linalg.eigh(np.cov(pts.T))
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            axB.add_patch(Ellipse(pts.mean(0), 2 * 1.5 * np.sqrt(w_[1]),
                                  2 * 1.5 * np.sqrt(w_[0]), angle=ang,
                                  facecolor=colours[gi], alpha=0.12, edgecolor=colours[gi],
                                  linewidth=0.6, zorder=1))
        axB.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)", fontsize=8)
        axB.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)", fontsize=8)
        axB.set_title("Embedding space", fontsize=9, loc="left", pad=4)

        depth_t = depth
        Dphy = patristic_matrix(tree, tips, depth_t)[np.ix_(v, v)]
        Demb = embedding_distance(X[v], dist_metric)
        rho, pval = mantel_spearman(Dphy, Demb, n_perm, rng)
        iu = np.triu_indices(Dphy.shape[0], 1)
        x, y = Dphy[iu], Demb[iu]
        axC = fig.add_subplot(gs[1, 1])
        axC.hexbin(x, y, gridsize=38, cmap="Greys", bins="log", mincnt=1,
                   linewidths=0, zorder=1)
        # quantile bins; ties (e.g. ultrametric trees) can collapse bins, so drop empties
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 13)))
        bi = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        keep = [b for b in range(len(edges) - 1) if (bi == b).any()]
        bx = np.array([x[bi == b].mean() for b in keep])
        q = np.array([np.percentile(y[bi == b], [25, 50, 75]) for b in keep])
        axC.fill_between(bx, q[:, 0], q[:, 2], color="#D55E00", alpha=0.2, lw=0, zorder=2)
        axC.plot(bx, q[:, 1], "-o", color="#D55E00", ms=3, lw=1.2, zorder=3)
        ptxt = f"{pval:.3f}" if pval >= 0.001 else "< 0.001"
        axC.text(0.04, 0.95, f"Spearman $\\rho$ = {rho:.2f}\nMantel $p$ {('= ' + ptxt) if pval >= 0.001 else ptxt}",
                 transform=axC.transAxes, va="top", fontsize=8)
        axC.set_xlabel("Patristic distance" + (" (branch length)" if "length" in distance_label
                                               else " (edges)"), fontsize=8)
        axC.set_ylabel(f"Embedding distance ({dist_metric})", fontsize=8)
        axC.set_title("Do embeddings track phylogeny?", fontsize=9, loc="left", pad=4)
        for a_, lab in ((axB, "b"), (axC, "c")):
            a_.spines[["top", "right"]].set_visible(False)
            a_.tick_params(labelsize=7, width=0.5, length=3)
            for s in a_.spines.values():
                s.set_linewidth(0.6)
            a_.text(-0.17, 1.12, lab, transform=a_.transAxes, fontsize=12,
                    fontweight="bold", va="top")
        info.update(rho=rho, mantel_p=pval, pca_var=var)
    return fig, info
