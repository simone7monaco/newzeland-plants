"""Merge GNN prediction CSVs from all benchmark seeds.

Example:
    python aggregate_gnn_predictions.py

The default input pattern is:
``benchmark_results/runs/seed_*/gnn/*/*/predictions_original_all_folds.csv``.
Each input row receives a ``seed`` column inferred from its ``seed_<number>``
directory before all rows are written to one CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path


LOGGER = logging.getLogger(__name__)
SEED_DIRECTORY = re.compile(r"seed_(\d+)$")
PREDICTION_FILENAME = "predictions_original_all_folds.csv"


def seed_from_path(path: Path) -> int:
    """Return the seed encoded by an ancestor ``seed_<number>`` directory."""
    for parent in path.parents:
        match = SEED_DIRECTORY.fullmatch(parent.name)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not find a seed_<number> directory for {path}")

def exp_from_path(path: Path) -> str:
    """Return the experiment name encoded by an ancestor ``gnn/<experiment>`` directory."""
    for parent in path.parents:
        if parent.name in ['env_only', 'phylo_only', 'no_env_no_phylo', 'env_phylo']:
            exp_raw = parent.name
            return exp_raw
    raise ValueError(f"Could not find a gnn/<experiment> directory for {path}")


def find_prediction_files(runs_dir: Path) -> list[Path]:
    """Return prediction files matching the benchmark GNN directory layout."""
    pattern = f"seed_*/gnn/*/*/{PREDICTION_FILENAME}"
    return sorted(runs_dir.glob(pattern))


def aggregate_predictions(input_files: list[Path], output_path: Path) -> int:
    """Write the rows in *input_files* to *output_path* with their seed."""
    if not input_files:
        raise ValueError("No prediction files were provided.")

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for input_file in input_files:
        exp = exp_from_path(input_file)
        seed = seed_from_path(input_file)
        with input_file.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"Prediction file has no header: {input_file}")

            source_fields = reader.fieldnames
            if "seed" in source_fields:
                raise ValueError(
                    f"Prediction file already has a 'seed' column: {input_file}"
                )
            for fieldname in source_fields:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
            for row in reader:
                row["seed"] = str(seed)
                row["exp"] = exp
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=[*fieldnames, "seed", "exp"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("benchmark_results/runs"),
        help="Directory containing seed_<number> benchmark runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/predictions_original_all_seeds.csv"),
        help="Path for the merged CSV.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    input_files = find_prediction_files(args.runs_dir)
    if not input_files:
        raise FileNotFoundError(
            f"No files matching the GNN prediction layout were found under "
            f"{args.runs_dir}"
        )

    row_count = aggregate_predictions(input_files, args.output)
    LOGGER.info(
        "Merged %d rows from %d files into %s",
        row_count,
        len(input_files),
        args.output,
    )


if __name__ == "__main__":
    main()
