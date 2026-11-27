import unittest
import io
from pathlib import Path

import pandas as pd
import torch
from Bio import Phylo

from loader import PlantDataset, data_split, stable_species_folds


ROOT = Path('data/Ferns')


@unittest.skipUnless((ROOT / 'FernMinMax_all.xlsx').exists(), 'Fern dataset unavailable')
class SpeciesCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PlantDataset(ROOT, traits_filename='FernMinMax_all.xlsx')
        cls.graph = cls.dataset[0]
        cls.workbook = pd.read_excel(ROOT / 'FernMinMax_all.xlsx')

    def test_workbook_species_are_retained(self):
        expected = set(self.dataset.traits_min.index)
        self.assertEqual(set(self.graph.species_names), expected)
        self.assertEqual(len(self.graph.species_names), len(expected))
        self.assertEqual(self.graph.species_num_nodes, len(expected))
        self.assertFalse(self.graph.traits_nanmask.all(dim=1).any())
        self.assertEqual(set(self.workbook['Species']) - expected, set(self.dataset.unobserved_species))

    def test_unobserved_species_can_be_recovered(self):
        original = PlantDataset.get_traits_df
        class Reader:
            trait_representation = 'min_max_range'
            traits_filename = 'FernMinMax_all.xlsx'
            pivot_feature = 'traits_min'
            include_unobserved_species = True
        reader = Reader()
        original(reader, ROOT, trait_representation='min_max_range')
        self.assertEqual(set(reader.traits_min.index), set(self.workbook['Species']))
        self.assertEqual(set(reader.unobserved_species), set(self.dataset.unobserved_species))

    def test_absent_tree_species_have_no_phylogenetic_edges_or_embeddings(self):
        tips = {tip.name for tip in Phylo.read(io.StringIO(self.dataset.tree_newick), 'newick').get_terminals()}
        absent = torch.tensor([i for i, name in enumerate(self.graph.species_names) if name not in tips], dtype=torch.long)
        self.assertGreater(absent.numel(), 0)
        self.assertFalse(torch.isin(self.graph.species_species_edge_index, absent).any())
        self.assertFalse((self.graph.species_x_phylo[absent] != 0).any())

    def test_every_species_appears_in_exactly_one_test_fold(self):
        total = torch.zeros(self.graph.species_num_nodes, dtype=torch.int32)
        for k in range(5):
            data_split(self.graph, k=k, seed=42)
            total += self.graph.test_mask.int()
        self.assertTrue((total == 1).all())

    def test_cache_depends_on_workbook(self):
        full = self.dataset.get_processed_file_names()[0]
        self.assertIn(self.dataset.traits_fingerprint, full)
        self.assertIn(self.dataset.tree_fingerprint, full)
        self.assertNotEqual(full, 'heterodata_min_max_range_v6.pt')

    def test_stable_folds_do_not_change_when_species_are_added_or_reordered(self):
        original = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        expanded = ['new', *reversed(original), 'another']
        folds = dict(zip(original, stable_species_folds(original, 42), strict=True))
        expanded_folds = dict(zip(expanded, stable_species_folds(expanded, 42), strict=True))
        self.assertEqual(folds, {name: expanded_folds[name] for name in original})


if __name__ == '__main__':
    unittest.main()
