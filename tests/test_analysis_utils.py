"""Consistency checks, naming and common-species metrics of analysis_utils."""
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import analysis_utils as au

ROOT = Path(__file__).resolve().parents[1]
TRAITS = list(au.TRAIT_ORDER)
SPECIES = [f"Genus_sp{i:02d}" for i in range(20)]
ABSENT = {"Genus_sp03", "Genus_sp11"}  # not in the reconstructed tree
BASELINE_DIRS = [
    "TRAINING_MEAN_prot1_occlusion-paired",
    "PHYLO_NN_prot1_occlusion-paired",
    "MICE_prot1_occlusion-paired_env_phylo",
    "RANDOM_FOREST_prot1_occlusion-paired_env_phylo",
]


def observed_value(species, trait, variable):
    rng = np.random.default_rng(abs(hash((species, trait, variable))) % 2**32)
    return np.nan if (species, trait) == ("Genus_sp05", "Spore") else float(rng.uniform(1, 100)) + (variable == "max") * 100


def write_run(model_dir, *, tree_only=False, offset=0.0, scale="original", seed=42, fold_shift=0):
    for k in range(au.N_FOLDS):
        rows = []
        for index, species in enumerate(SPECIES):
            if (index + fold_shift) % au.N_FOLDS != k:
                continue
            for trait in TRAITS:
                for variable in au.BOUND_ORDER:
                    observed = observed_value(species, trait, variable)
                    predictable = not (tree_only and species in ABSENT)
                    predicted = observed + offset + (1.0 if variable == "max" else -1.0) if predictable else np.nan
                    rows.append(dict(species=species, fold=f"fold_{k}", trait=trait, variable=variable,
                                     observed=observed, predicted=predicted,
                                     evaluated=bool(np.isfinite(observed) and predictable), scale=scale))
        fold_dir = model_dir / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(fold_dir / au.LONG_FILE, index=False)


def build_results(root, **overrides):
    seed_dir = root / "runs" / "seed_42"
    coverage = pd.DataFrame({
        "species": SPECIES,
        "ftol_tip": [None if s in ABSENT else s for s in SPECIES],
        "placement": ["genus absent" if s in ABSENT else "exact binomial" for s in SPECIES],
        "in_tree": [s not in ABSENT for s in SPECIES],
    })
    (seed_dir / "baselines").mkdir(parents=True)
    coverage.to_csv(seed_dir / "baselines" / au.TREE_COVERAGE_FILE, index=False)
    for name in BASELINE_DIRS:
        write_run(seed_dir / "baselines" / name, tree_only=name.startswith("PHYLO_NN"),
                  offset=2.0, **overrides.get(name, {}))
    for combination in ("env_phylo", "no_env_no_phylo"):
        gnn_dir = seed_dir / "gnn" / combination / "FERN_RUN"
        write_run(gnn_dir, offset=1.0, **overrides.get("gnn", {}))
        coverage.to_csv(gnn_dir / au.TREE_COVERAGE_FILE, index=False)
    return root


class NamingTests(unittest.TestCase):
    def test_baseline_directories_parse_strictly(self):
        run = au.parse_baseline_dir("XGBOOST_prot1_occlusion-paired_env_phylo")
        self.assertEqual((run.model, run.configuration_id, run.use_env, run.use_phylo), ("xgboost", "full", True, True))
        self.assertEqual(run.label, "XGBoost (environment + phylogeny)")
        self.assertEqual(au.parse_baseline_dir("MICE_prot1").occlusion, None)  # historical name
        for bad in ("TRAINING_MEAN_prot1_occlusion-paired_env", "UNKNOWN_prot1_occlusion-paired", "MICE_prot1_x"):
            with self.assertRaises(au.AnalysisConsistencyError):
                au.parse_baseline_dir(bad)

    def test_orderings_and_unknown_labels(self):
        frame = au.order_frame(pd.DataFrame({"trait": ["Spore", "Alt"], "variable": ["max", "min"]}))
        self.assertEqual(list(frame.sort_values("trait").trait.astype(str)), ["Alt", "Spore"])
        with self.assertRaises(au.AnalysisConsistencyError):
            au.order_frame(pd.DataFrame({"trait": ["Height"]}))
        self.assertEqual(au.contrast_label("baseline", "full"), "Base → Phylo_Env")


class PredictionChecksTests(unittest.TestCase):
    def test_common_species_and_global_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_results(Path(tmp))
            predictions = au.load_benchmark_predictions(root)
            coverage = au.load_tree_coverage(root)
            full = predictions.loc[predictions.configuration_id.eq("full") | predictions.model.isin(["training_mean", "phylo_nn"])]
            cells = au.baseline_comparison_cells(full, coverage=coverage)
            self.assertEqual(cells.attrs["excluded_species"], sorted(ABSENT))
            self.assertFalse(cells.species.isin(ABSENT).any())
            metrics = au.global_metrics(cells, au.cohort_scales(predictions))
            self.assertEqual(metrics.groupby("model_label", observed=True).n.sum().nunique(), 1)  # identical cells
            gnn = metrics.loc[metrics.model.eq("gnn")]
            # GNN offset +1 plus -1 (min) / +1 (max): errors are 0 for minima and 2 for maxima.
            expected = np.where(gnn.variable.astype(str).eq("max"), 2.0, 0.0)
            self.assertTrue(np.allclose(gnn.RMSE, expected))
            self.assertTrue(np.allclose(gnn.NRMSE_IQR, gnn.RMSE / gnn.trait_iqr))
            # Without a tree-only method nothing is excluded.
            no_tree = au.baseline_comparison_cells(full.loc[~full.model.eq("phylo_nn")], coverage=coverage)
            self.assertEqual(no_tree.attrs["excluded_species"], [])

    def test_normalized_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_results(Path(tmp), gnn={"scale": "normalized"})
            with self.assertRaisesRegex(au.AnalysisConsistencyError, "original scale"):
                au.load_benchmark_predictions(root)

    def test_mismatched_folds_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_results(Path(tmp), **{BASELINE_DIRS[0]: {"fold_shift": 1}})
            with self.assertRaisesRegex(au.AnalysisConsistencyError, "different species folds"):
                au.load_benchmark_predictions(root)

    def test_unexpected_exclusions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_results(Path(tmp))
            coverage = au.load_tree_coverage(root)
            coverage.loc[coverage.species.eq("Genus_sp03"), "in_tree"] = True  # claim it is in the tree
            predictions = au.load_benchmark_predictions(root)
            with self.assertRaisesRegex(au.AnalysisConsistencyError, "expected exclusions"):
                au.baseline_comparison_cells(predictions, coverage=coverage)

    def test_unknown_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_results(Path(tmp))
            write_run(root / "runs/seed_42/baselines/MYSTERY_MODEL", offset=0.0)
            with self.assertRaisesRegex(au.AnalysisConsistencyError, "Unrecognized baseline"):
                au.load_benchmark_predictions(root)


class NotebookTests(unittest.TestCase):
    def test_baseline_notebook_runs_on_synthetic_results(self):
        notebook = json.loads((ROOT / "notebooks/01_benchmark_analysis.ipynb").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = build_results(Path(tmp))
            (root / "manifest.json").write_text(json.dumps({
                "start_seed": 42, "n_seeds": 1, "result_file": "predictions_max_all.parquet",
                "feature_combinations": [[True, True, "env_phylo"], [True, False, "env_only"],
                                         [False, True, "phylo_only"], [False, False, "no_env_no_phylo"]],
            }))
            (root / "runs/seed_42/baselines/MICE_prot1_occlusion-paired_env_phylo/predictions_max_all.parquet").touch()
            (root / "runs/seed_42/gnn/env_phylo/FERN_RUN/predictions_max_all.parquet").touch()
            namespace = {"display": lambda *args, **kwargs: None, "__name__": "__notebook__"}
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"]).replace("RESULTS_DIR = DEFAULT_RESULTS_DIR", f"RESULTS_DIR = Path({str(root)!r})")
                exec(compile(source, "notebook-01", "exec"), namespace)
            self.assertIn("tab:performance_traits", namespace["latex"])
            self.assertEqual(namespace["cells"].attrs["excluded_species"], sorted(ABSENT))
            self.assertEqual(namespace["status"].shape, (len(namespace["jobs"]), 1))
            self.assertEqual(int((namespace["status"] == 2).sum()), 2)  # the MICE env+phylo and GNN env_phylo jobs


if __name__ == "__main__":
    unittest.main()
