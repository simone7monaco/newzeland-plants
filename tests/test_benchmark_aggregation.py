"""Species names must survive aggregation of baseline and GNN prediction files."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_benchmark import read_csv


class ReadCsvSpeciesTests(unittest.TestCase):
    def test_named_index_becomes_species_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions_max_all.parquet'
            frame = pd.DataFrame({'A': [1.0, 2.0]}, index=pd.Index(['Sp_a', 'Sp_b'], name='species'))
            frame.to_parquet(path)
            result = read_csv(path, seed=1, model='mice', source='r_baseline')
            self.assertEqual(result.species.tolist(), ['Sp_a', 'Sp_b'])

    def test_unnamed_csv_index_still_becomes_species_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions_max.csv'
            pd.DataFrame({'A': [1.0, 2.0]}, index=['Sp_a', 'Sp_b']).to_csv(path)
            result = read_csv(path, seed=1, model='gnn', source='gnn')
            self.assertEqual(result.species.tolist(), ['Sp_a', 'Sp_b'])


if __name__ == '__main__':
    unittest.main()
