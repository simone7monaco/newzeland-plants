"""Component ablation of PhyEnv-NN (reviewer comments R2.1-3, R4.2, R4.7).

Re-runs the standard ``train.py`` pipeline while switching four components
independently:

* ``family``  - categorical Family dummies among species inputs (``--use_family``)
* ``emb``     - Node2Vec phylogenetic descriptors (``--use_phylo_embeddings``)
* ``edges``   - phylogenetic species-species message passing; self-loops otherwise
                (``--use_phylo_edges``)
* ``env``     - spatial graph and bipartite location-to-species pathway
                (``--use_env_features``)

Designs
-------
``incremental`` (default): starts from the trait-only model (traits, visibility
mask and non-family categorical inputs, self-loops only) and adds one component at a
time following ``--order``; e.g. ``family,emb,edges,env`` gives 5 configurations.
``leave-one-out``: the full model and the full model minus each component.
``factorial``: all 2^4 combinations, needed for interaction estimates.
Several designs can be combined (``--design incremental,leave-one-out``); duplicate
configurations are run once. ``--order`` may be given multiple times to compare
alternative incremental paths.

All configurations share seeds, folds, occlusion protocol, hyperparameters and
training-only preprocessing; only the switched component differs. Hidden sizes are
unchanged, so differences in parameter count come only from input widths.

Layout (compatible with ``run_benchmark.py``)::

    <output-dir>/runs/seed_<s>/gnn/<config>/<TRAIN_EXP_NAME>/...
    <output-dir>/ablation_manifest.csv      # one row per job: flags, runtime, status
    <output-dir>/ablation_<file>.parquet    # aggregated predictions with config columns

Nothing is trained with ``--dry-run``; it only prints the planned commands.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

COMPONENTS = ("family", "emb", "edges", "env")
COMPONENT_FLAGS = {
    "family": "--use_family",
    "emb": "--use_phylo_embeddings",
    "edges": "--use_phylo_edges",
    "env": "--use_env_features",
}
MANAGED_FLAGS = set(COMPONENT_FLAGS.values()) | {
    "--use_phylo_features", "--k", "--seed", "--output_dir", "--occlusion_protocol",
    "--include_unobserved_species",
}
RESULT_FILES = ("predictions_min_all.parquet", "predictions_max_all.parquet", "predictions_long_all.parquet")


def config_name(active: frozenset[str]) -> str:
    if not active:
        return "traits_only"
    return "+".join(c for c in COMPONENTS if c in active)


def parse_order(text: str) -> tuple[str, ...]:
    order = tuple(part.strip() for part in text.split(",") if part.strip())
    if sorted(order) != sorted(set(order)) or not set(order) <= set(COMPONENTS):
        raise argparse.ArgumentTypeError(f"--order must be a permutation of a subset of {COMPONENTS}: {text}")
    return order


def plan_configurations(designs: list[str], orders: list[tuple[str, ...]]) -> list[dict]:
    """Return unique configurations with the designs/steps that requested them."""
    planned: dict[frozenset[str], dict] = {}

    def add(active, tag):
        active = frozenset(active)
        entry = planned.setdefault(active, {"active": active, "name": config_name(active), "tags": []})
        entry["tags"].append(tag)

    full = frozenset(COMPONENTS)
    for design in designs:
        if design == "incremental":
            for order in orders:
                path = "-".join(order)
                for step in range(len(order) + 1):
                    added = order[step - 1] if step else "none"
                    add(order[:step], f"incremental[{path}]:{step}:+{added}")
        elif design == "leave-one-out":
            add(full, "leave-one-out:full")
            for component in COMPONENTS:
                add(full - {component}, f"leave-one-out:-{component}")
        elif design == "factorial":
            for r in range(len(COMPONENTS) + 1):
                for combo in itertools.combinations(COMPONENTS, r):
                    add(combo, "factorial")
        else:
            raise SystemExit(f"Unknown design: {design}")
    return list(planned.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("n_seeds", type=int, help="Number of consecutive seeds")
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--design", default="incremental",
                        help="Comma-separated designs: incremental, leave-one-out, factorial")
    parser.add_argument("--order", type=parse_order, action="append",
                        help="Incremental order (repeatable), default family,emb,edges,env")
    parser.add_argument("--occlusion-protocol", choices=("paired", "individual"), default="paired")
    parser.add_argument("--include-unobserved-species", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("ablation_results"))
    parser.add_argument("--gnn-args", default="", help="Extra train.py arguments, shell-quoted, shared by all configs")
    parser.add_argument("--compute-xai", action="store_true", help="Also compute IG attributions (slow)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and commands without training")
    return parser.parse_args()


def fingerprints(include_unobserved: bool) -> str:
    inputs = [Path("data/Ferns/FernTraits.xlsx"), Path("ftol_tree.R")]
    text = "".join(f"{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in inputs if p.exists())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "train.py", "models.py", "loader.py"],
                                    capture_output=True, text=True).stdout.strip())
        text += f"git {commit}{' (dirty)' if dirty else ''}\n"
    except (OSError, subprocess.CalledProcessError):
        text += "git unavailable\n"
    return text + f"include_unobserved_species {str(include_unobserved).lower()}\n"


def build_command(cfg: dict, seed: int, run_dir: Path, args, gnn_args: list[str]) -> list[str]:
    command = [sys.executable, "train.py", *gnn_args]
    for component, flag in COMPONENT_FLAGS.items():
        command += [flag, str(component in cfg["active"]).lower()]
    command += [
        "--occlusion_protocol", args.occlusion_protocol,
        "--include_unobserved_species", str(args.include_unobserved_species).lower(),
        "--compute_xai", str(args.compute_xai).lower(),
        "--k", "-1", "--seed", str(seed), "--output_dir", str(run_dir),
    ]
    return command


def aggregate(output_dir: Path, configs: list[dict]) -> None:
    by_name = {cfg["name"]: cfg for cfg in configs}
    for filename in RESULT_FILES:
        frames = []
        for path in sorted(output_dir.glob(f"runs/seed_*/gnn/*/*/{filename}")):
            name = path.parent.parent.name
            if name not in by_name:
                continue
            frame = pd.read_parquet(path)
            if "species" not in frame.columns and frame.index.name == "species":
                frame = frame.reset_index()  # wide GNN exports are indexed by species
            unnamed = [c for c in frame.columns if c.startswith("Unnamed:")]
            if "species" not in frame.columns and unnamed:
                frame = frame.rename(columns={unnamed[0]: "species"})
            frame.insert(0, "configuration", name)
            frame.insert(1, "seed", int(path.parents[3].name.split("_", 1)[1]))
            for component in COMPONENTS:
                frame.insert(2, f"use_{component}", component in by_name[name]["active"])
            frames.append(frame)
        if frames:
            out = output_dir / f"ablation_{filename}"
            pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
            print(f"Saved {out} ({len(frames)} runs)")


def main() -> None:
    args = parse_args()
    if args.n_seeds < 1:
        raise SystemExit("n_seeds must be at least 1")
    gnn_args = shlex.split(args.gnn_args)
    clashes = [a for a in gnn_args if a.split("=")[0] in MANAGED_FLAGS]
    if clashes:
        raise SystemExit(f"These flags are managed by the ablation script: {clashes}")
    designs = [d.strip() for d in args.design.split(",") if d.strip()]
    orders = args.order or [COMPONENTS]
    configs = plan_configurations(designs, orders)
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    output_dir = args.output_dir.resolve()

    print(f"{len(configs)} configurations x {len(seeds)} seeds = {len(configs) * len(seeds)} jobs (5 folds each)")
    for cfg in configs:
        print(f"  {cfg['name']:<26} <- {', '.join(cfg['tags'])}")

    if args.dry_run:
        for cfg in configs:
            print(shlex.join(build_command(cfg, seeds[0], output_dir / "runs" / f"seed_{seeds[0]}" / "gnn" / cfg["name"], args, gnn_args)))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "run_settings.txt"
    settings = fingerprints(args.include_unobserved_species) + (
        f"occlusion_protocol {args.occlusion_protocol}\ngnn_args {shlex.join(gnn_args)}\n"
    )
    if marker.exists() and marker.read_text().split("git ")[0] != settings.split("git ")[0]:
        raise SystemExit("Data, protocol or shared arguments differ from existing runs; choose a fresh --output-dir.")
    marker.write_text(settings)

    manifest_path = output_dir / "ablation_manifest.csv"
    manifest = pd.read_csv(manifest_path).to_dict("records") if manifest_path.exists() else []
    for seed in seeds:
        for cfg in configs:
            run_dir = output_dir / "runs" / f"seed_{seed}" / "gnn" / cfg["name"]
            if sorted(run_dir.glob(f"*/{RESULT_FILES[1]}")):
                print(f"[skip] {cfg['name']} seed {seed}: results exist")
                continue
            command = build_command(cfg, seed, run_dir, args, gnn_args)
            print(f"[run] {cfg['name']} seed {seed}\n  {shlex.join(command)}")
            started = time.monotonic()
            status = "ok"
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                status = f"failed ({error.returncode})"
            row = {"configuration": cfg["name"], "seed": seed, "status": status,
                   "runtime_s": round(time.monotonic() - started, 1), "tags": "; ".join(cfg["tags"]),
                   "command": shlex.join(command)}
            row.update({f"use_{c}": c in cfg["active"] for c in COMPONENTS})
            manifest.append(row)
            pd.DataFrame(manifest).to_csv(manifest_path, index=False)
            if status != "ok":
                raise SystemExit(f"{cfg['name']} seed {seed} {status}; see output above")

    aggregate(output_dir, configs)


if __name__ == "__main__":
    main()
