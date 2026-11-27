import unittest

import torch
from torch_geometric.data import Data

from occlusion import grouped_random_mask, trait_groups
from train import create_validation_mask
from tester import leave_one_trait_out


NAMES = ['Ferstipe', 'SterileStipe', 'FertileFrond', 'SterileFrond', 'Spore']


class OcclusionTests(unittest.TestCase):
    def test_training_and_validation_keep_observed_pairs_together(self):
        groups = trait_groups(NAMES)
        observed = torch.ones((100, 5), dtype=torch.bool)
        observed[0, 1] = False
        for mask in (grouped_random_mask(observed, groups, 0.5),
                     create_validation_mask(observed, 0.2, 42, groups)):
            for a, b in ((0, 1), (2, 3)):
                both = observed[:, a] & observed[:, b]
                self.assertTrue(torch.equal(mask[both, a], mask[both, b]))
            self.assertFalse((mask & ~observed).any())

    def test_inference_hides_pair_for_all_test_nodes_and_scores_observed_only(self):
        class Spy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.occlusion_groups = trait_groups(NAMES)
                self.calls = []

            def forward(self, graph):
                self.calls.append(graph.clone())
                return graph.species_x_min + 1, graph.species_x_max + 1, graph.species_x_range + 1

        values = torch.ones((4, 5))
        missing = torch.zeros((4, 5), dtype=torch.bool)
        missing[2, 0] = True
        graph = Data(species_x_min=values.clone(), species_x_max=values.clone(),
                     species_x_range=values.clone(), traits_nanmask=missing)
        model = Spy()
        result = leave_one_trait_out(model, graph, torch.tensor([1, 2, 3]), 'cpu')
        for feature in ('species_x_min', 'species_x_max', 'species_x_range'):
            self.assertTrue((getattr(model.calls[0], feature)[1:, :2] == 0).all())
            self.assertTrue((getattr(model.calls[0], feature)[0, :2] == 1).all())
        self.assertTrue(model.calls[0].traits_nanmask[1:, :2].all())
        self.assertFalse(result['eval_mask'][1, 0])
        self.assertTrue(result['eval_mask'][1, 1])

    def test_pair_definition_requires_both_columns(self):
        with self.assertRaises(ValueError):
            trait_groups(['Ferstipe', 'Spore'])
        self.assertEqual(trait_groups(['Ferstipe', 'Spore'], 'individual'), [(0,), (1,)])


if __name__ == '__main__':
    unittest.main()
