"""Regression test for inner-validation preprocessing leakage."""
import unittest

import torch
from torch_geometric.data import Data

from loader import NormalizeFeatures
from train import create_validation_mask


class InternalValidationIndependenceTests(unittest.TestCase):
    def test_validation_targets_do_not_change_fitted_trait_scaling(self):
        values = torch.arange(1., 21.).reshape(10, 2)
        observed = torch.ones_like(values, dtype=torch.bool)
        held_out = create_validation_mask(observed, 0.2, 42, [[0], [1]])

        def fitted_mean(x):
            # Mirrors train.py: validation cells are marked missing before the fit.
            data = Data(species_x_min=x.clone(), traits_nanmask=~observed | held_out)
            transform = NormalizeFeatures(trait_norm_min='z')
            transform.fit(data)
            return transform.props['species_x_min']['mean']

        perturbed = values.clone()
        perturbed[held_out] += 1000
        torch.testing.assert_close(fitted_mean(values), fitted_mean(perturbed))


if __name__ == '__main__':
    unittest.main()
