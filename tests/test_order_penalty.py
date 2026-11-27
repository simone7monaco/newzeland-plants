"""The differentiable trait inverse must match the reference numpy inverse."""
import unittest

import torch
from torch_geometric.data import Data

from loader import NormalizeFeatures


class InverseTensorTests(unittest.TestCase):
    def test_inverse_tensor_matches_inverse(self):
        torch.manual_seed(0)
        raw = torch.rand(40, 3) * torch.tensor([10., 500., 3000.])
        for mode in ('yj', 'z', 'logz'):
            with self.subTest(mode=mode):
                transform = NormalizeFeatures(trait_norm_min=mode, trait_norm_max=mode)
                data = Data(species_x_min=raw.clone(), species_x_max=raw.clone() * 2,
                            traits_nanmask=torch.zeros_like(raw, dtype=torch.bool))
                normalized = transform(data.clone())
                reference = transform.inverse(normalized, warn=False).species_x_min
                values = normalized.species_x_min.clone().requires_grad_(True)
                recovered = transform.inverse_tensor('species_x_min', values)
                torch.testing.assert_close(recovered, reference, rtol=1e-4, atol=1e-3)
                recovered.sum().backward()
                self.assertTrue(torch.isfinite(values.grad).all())


if __name__ == '__main__':
    unittest.main()
