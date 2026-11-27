from benchmark_analysis import DEFAULT_RESULTS_DIR, infer_traits_file, load_metric_rows
from pathlib import Path
import pandas as pd
import numpy as np

from analysis_utils import CONFIGURATION_LABELS as CONFIG_LABELS, CONFIGURATION_ORDER, require_known

CONFIG_ORDER = list(CONFIGURATION_ORDER)

DEFAULT_RESULTS_PATH = Path("benchmark_results/results_ablation.parquet")

results_path = Path(str(DEFAULT_RESULTS_DIR)).expanduser()
traits_path = infer_traits_file(results_path)

raw = load_metric_rows(Path(results_path), Path(traits_path))

print(f"Shape: {raw.shape[0]:,} rows × {raw.shape[1]} columns")

def normalize_configuration(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    if "configuration_id" in out.columns:
        cfg = out["configuration_id"].astype(str).str.lower().str.strip()
    elif "configuration" in out.columns:
        reverse_labels = {v.lower(): k for k, v in CONFIG_LABELS.items()}
        cfg = out["configuration"].astype(str).str.lower().str.strip().replace(reverse_labels)
    elif {"use_env_features", "use_phylo_features"}.issubset(out.columns):
        env = out["use_env_features"].astype(bool)
        phy = out["use_phylo_features"].astype(bool)
        cfg = pd.Series(
            np.select(
                [~env & ~phy, env & ~phy, ~env & phy, env & phy],
                CONFIG_ORDER,
                default="unknown",
            ),
            index=out.index,
        )
    elif "experiment_dir" in out.columns:
        mapping = {
            "no_env_no_phylo": "baseline",
            "env_only": "environment",
            "phylo_only": "phylogeny",
            "env_phylo": "full",
        }
        cfg = out["experiment_dir"].astype(str).str.lower().map(mapping).fillna("unknown")
    else:
        raise KeyError(
            "No configuration_id/configuration, use_env_features/use_phylo_features flags "
            "or experiment_dir column found. Adapt normalize_configuration()."
        )

    alias = {
        "no_env_no_phylo": "baseline", "none": "baseline", "base": "baseline",
        "env_only": "environment", "environment_only": "environment", "ambiente": "environment",
        "phylo_only": "phylogeny", "phylogeny_only": "phylogeny", "filogenesi": "phylogeny",
        "env_phylo": "full", "environment_phylogeny": "full", "ambiente + filogenesi": "full",
    }
    out["configuration_id"] = cfg.replace(alias)
    require_known(out["configuration_id"], CONFIG_ORDER, "configuration")
    out["configuration"] = out["configuration_id"].map(CONFIG_LABELS)
    return out


df = normalize_configuration(raw)

if "model" in df.columns:
    gnn = df.loc[df["model"].astype(str).str.lower().eq("gnn")].copy()
else:
    gnn = df.copy()

for col in ["seed", "n", "RMSE", "MAE", "Pearson_r", "Spearman_rho", "NRMSE_IQR", "NRMSE_range"]:
    if col in gnn.columns:
        gnn[col] = pd.to_numeric(gnn[col], errors="coerce")

required = {"seed", "configuration_id", "trait", "variable"}
missing = required - set(gnn.columns)
if missing:
    raise KeyError(f"Missing required columns: {sorted(missing)}")

print("Configurations found:", sorted(gnn["configuration_id"].dropna().unique()))
print("Distinct seeds:", gnn["seed"].nunique())
print("GNN rows:", f"{len(gnn):,}")
print(gnn.head())

gnn.to_parquet(DEFAULT_RESULTS_PATH, index=False)
