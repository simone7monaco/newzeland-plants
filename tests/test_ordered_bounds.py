import unittest

import torch
from torch_geometric.data import Data

from loader import NormalizeFeatures
from models import MultiTargetDeterministicLoss, TraitsPredictor


class OrderedBoundsTests(unittest.TestCase):
    def test_shared_normalization_and_inverse_preserve_order(self):
        data = Data(species_x_min=torch.tensor([[0., 1.], [1., 2.], [3., 4.], [5., 6.]]),
                    species_x_max=torch.tensor([[1., 3.], [4., 8.], [6., 10.], [9., 12.]]),
                    traits_nanmask=torch.zeros((4, 2), dtype=torch.bool))
        for kind in ('yj', 'z', 'logz'):
            transform = NormalizeFeatures(trait_norm_min=kind, trait_norm_max=kind, shared_bounds=True)
            transform.fit(data)
            normalized = transform(data.clone())
            self.assertTrue((normalized.species_x_min <= normalized.species_x_max).all())
            restored = transform.inverse(normalized, warn=False, bounded=True)
            self.assertTrue(torch.isfinite(restored.species_x_min).all())
            self.assertTrue((restored.species_x_min <= restored.species_x_max).all())

    def test_model_predicts_ordered_bounds_and_gradients(self):
        data = Data(species_x_min=torch.ones(3, 2), species_x_max=torch.ones(3, 2),
                    species_x_gen=torch.ones(3, 1), traits_nanmask=torch.zeros(3, 2, dtype=torch.bool),
                    species_species_edge_index=torch.empty((2, 0), dtype=torch.long),
                    species_species_edge_attr=torch.empty((0, 1)))
        for method in ('min_plus_range_relu', 'min_plus_range_softplus'):
            model = TraitsPredictor(2, 1, 0, 0, 4, 2, 2, use_env_features=False,
                                    use_phylo_features=False, mask_ratio=0,
                                    trait_feature_keys=('species_x_min', 'species_x_max'),
                                    bounds_parameterization=method)
            low, high = model(data)
            self.assertTrue((high >= low).all())
            loss = MultiTargetDeterministicLoss()((low, high), (torch.zeros_like(low), torch.full_like(high, 2.)))
            loss.backward()
            self.assertTrue(torch.isfinite(model.fc.bias.grad).all())
            self.assertGreater(model.fc.bias.grad[2:].abs().sum().item(), 0)

    def test_unsupported_settings_fail(self):
        with self.assertRaises(ValueError):
            NormalizeFeatures(trait_norm_min='yj', trait_norm_max='z', shared_bounds=True)
        with self.assertRaises(ValueError):
            TraitsPredictor(2, 1, 0, 0, 4, 2, 2, use_env_features=False,
                            trait_feature_keys=('species_x_min', 'species_x_max', 'species_x_range'),
                            bounds_parameterization='min_plus_range_relu')


if __name__ == '__main__':
    unittest.main()
