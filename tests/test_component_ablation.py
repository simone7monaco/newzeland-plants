import itertools
import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from models import TraitsPredictor
from run_component_ablation import plan_configurations
from train import drop_family_features, resolve_component_flags


def toy_data():
    return Data(species_x_min=torch.randn(4, 2), species_x_max=torch.randn(4, 2),
                species_x_gen=torch.ones(4, 3), species_x_phylo=torch.randn(4, 5),
                genetic_feature_names=['Family_A', 'Family_B', 'Habitat_X'],
                traits_nanmask=torch.zeros(4, 2, dtype=torch.bool),
                species_species_edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                species_species_edge_attr=torch.ones(3, 1))


class ComponentAblationTests(unittest.TestCase):
    def test_family_columns_removed(self):
        data = drop_family_features(toy_data())
        self.assertEqual(data.genetic_feature_names, ['Habitat_X'])
        self.assertEqual(data.species_x_gen.shape, (4, 1))

    def test_legacy_flag_expansion(self):
        args = SimpleNamespace(use_phylo_features=False, use_phylo_embeddings=None, use_phylo_edges=True)
        resolve_component_flags(args)
        self.assertEqual((args.use_phylo_embeddings, args.use_phylo_edges, args.use_phylo_features), (False, True, True))

    def test_embedding_and_edge_switches_are_independent(self):
        data = toy_data()
        for emb, edges in itertools.product((False, True), repeat=2):
            model = TraitsPredictor(2, 3, 5, 0, 4, 2, 1, use_env_features=False, mask_ratio=0,
                                    trait_feature_keys=('species_x_min', 'species_x_max'),
                                    use_phylo_embeddings=emb, use_phylo_edges=edges)
            self.assertEqual(model.species_linear.in_features, 3 * 2 + 3 + (5 if emb else 0))
            edge_index, _ = model._species_graph_inputs(data, 4)
            self.assertEqual(edge_index.size(1), 3 if edges else 4)
            low, high = model.eval()(data)
            self.assertEqual(low.shape, (4, 2))

    def test_plans(self):
        self.assertEqual(len(plan_configurations(['incremental'], [('family', 'emb', 'edges', 'env')])), 5)
        self.assertEqual(len(plan_configurations(['leave-one-out'], [])), 5)
        self.assertEqual(len(plan_configurations(['factorial'], [])), 16)


if __name__ == '__main__':
    unittest.main()
