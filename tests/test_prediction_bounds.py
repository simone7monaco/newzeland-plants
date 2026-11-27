"""Shared original-scale postprocessing (GNN and ML baselines) and MICE pooling."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer
from torch_geometric.data import Data

from analysis_utils import AnalysisConsistencyError
from loader import NormalizeFeatures, _yj_inverse
from prediction_bounds import bounded_inverse, write_safeguard_log
from r_models import pool_completions
from tester import long_predictions_frame, compute_metrics_minmax, project_ordered_bounds


def _normalized(values, nanmask=None):
    values = torch.tensor(values, dtype=torch.float32)
    if nanmask is None:
        nanmask = torch.zeros(values.shape, dtype=torch.bool)
    data = Data(species_x_min=values.clone(), species_x_max=values.clone(), traits_nanmask=nanmask)
    transform = NormalizeFeatures()
    transform.fit(data)
    return transform, transform(data.clone())


class BoundedInverseTests(unittest.TestCase):
    def test_raw_infinities_raise_before_inverse_or_clipping(self):
        for value in (np.inf, -np.inf):
            inverse = Mock()
            with self.assertRaisesRegex(FloatingPointError, 'raw predictions contain infinity'):
                bounded_inverse(inverse, np.array([[value]]), np.array([0.]), np.array([1.]))
            inverse.assert_not_called()
        transform, _ = _normalized([[1.], [2.], [3.]])
        with self.assertRaises(FloatingPointError):
            transform.inverse_tensor('species_x_min', torch.tensor([[float('inf')]]))

    def test_corrections_are_counted_per_output_and_evaluation_mask(self):
        def inverse(z):
            result = z.copy()
            result[:, 0] = np.where(z[:, 0] > 1.5, np.inf, z[:, 0])
            return result

        values = np.array([[-3., -2.], [3., 3.], [1.75, .5], [np.nan, 0.]])
        mask = np.array([[True, False], [True, True], [False, True], [False, True]])
        records = []
        restored = bounded_inverse(inverse, values, np.zeros(2), np.ones(2),
                                   nonnegative=True, diagnostics=records, evaluation_mask=mask)
        np.testing.assert_allclose(restored[:3], [[0., 0.], [1., 2.], [1., .5]])
        self.assertTrue(np.isnan(restored[3, 0]))
        first = records[0]
        self.assertEqual(first['n_predictions'], 3)
        self.assertEqual(first['n_missing'], 1)
        self.assertEqual(first['n_clipped_low'], 1)
        self.assertEqual(first['n_clipped_high'], 1)
        self.assertEqual(first['n_inverse_fallback'], 2)
        self.assertEqual(first['n_nonnegative_clipped'], 1)
        self.assertEqual(first['n_corrected'], 3)
        self.assertEqual(first['n_eval_predictions'], 2)
        self.assertEqual(first['n_eval_inverse_fallback'], 1)
        self.assertEqual(records[1]['n_eval_clipped_low'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fold_0' / 'prediction_safeguards.csv'
            rows = [{k: v for k, v in row.items() if k != 'column'} |
                    {'stage': 'test', 'trait': 'Spore', 'variable': variable}
                    for row, variable in zip(records, ('min', 'max'))]
            write_safeguard_log(rows, path)
            saved = pd.read_csv(path)
            self.assertEqual(saved.fold.tolist(), ['fold_0', 'fold_0'])
            self.assertEqual(saved.n_inverse_fallback.tolist(), [2, 0])

    def test_gnn_log_has_stage_output_names_and_scored_counts(self):
        transform, data = _normalized([[1., 2.], [2., 3.], [3., 5.]])
        data.species_x_min = data.species_x_max = torch.tensor([[1000., -1000.]])
        transform.inverse(data, bounded=True, warn=False, stage='calibration',
                          trait_names=['Spore', 'Alt'], evaluation_mask=np.array([[True, False]]))
        rows = transform.prediction_diagnostics
        self.assertEqual(len(rows), 4)
        self.assertEqual({r['stage'] for r in rows}, {'calibration'})
        self.assertEqual({r['trait'] for r in rows}, {'Spore', 'Alt'})
        self.assertEqual(sum(r['n_eval_predictions'] for r in rows), 2)
        self.assertEqual(sum(r['n_clipped_low'] + r['n_clipped_high'] for r in rows), 4)

    def test_nan_on_an_evaluated_cell_is_rejected(self):
        with self.assertRaisesRegex(FloatingPointError, 'NaN on evaluated cells'):
            bounded_inverse(lambda z: z, np.array([[np.nan]]), np.zeros(1), np.ones(1),
                            evaluation_mask=np.ones((1, 1), dtype=bool))

    def test_raw_yj_inverse_is_infinite_outside_its_domain(self):
        # The reason for the bounded rule: lambda < 0 has no inverse beyond 1/|lambda|.
        with np.errstate(divide='ignore'):
            self.assertTrue(np.isinf(_yj_inverse(np.array([3.0]), -0.5)).all())

    def test_extreme_outputs_stay_finite_and_inside_extended_range(self):
        # Right-skewed positive data: YJ fits lambda < 0, where the raw inverse overflows.
        rng = np.random.default_rng(0)
        transform, normalized = _normalized(rng.lognormal(0, 1.5, size=(60, 2)))
        self.assertTrue((transform.props['species_x_min']['lambdas'] < 0).any())
        normalized.species_x_min = torch.tensor([[1e3, -1e3], [50., -50.]])
        restored = transform.inverse(normalized, warn=False, bounded=True).species_x_min
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue((restored >= 0).all())

    def test_in_range_predictions_are_not_shifted(self):
        # Unlike the previous Softplus, small positive values pass through unchanged.
        values = np.array([[0.05, 2.0], [0.4, 3.0], [1.0, 5.0], [2.5, 9.0], [0.1, 4.0]])
        transform, normalized = _normalized(values)
        restored = transform.inverse(normalized, warn=False, bounded=True).species_x_min.numpy()
        np.testing.assert_allclose(restored, values, rtol=1e-4, atol=1e-5)

    def test_signed_traits_are_not_clipped_at_zero(self):
        values = np.array([[-3.0], [-1.0], [0.5], [2.0], [4.0]])
        transform, normalized = _normalized(values)
        self.assertFalse(bool(transform.props['species_x_min']['nonnegative'][0]))
        restored = transform.inverse(normalized, warn=False, bounded=True).species_x_min.numpy()
        np.testing.assert_allclose(restored, values, rtol=1e-4, atol=1e-5)

    def test_masked_cells_do_not_define_the_range(self):
        values = np.array([[1.0], [2.0], [3.0], [1000.0]])
        nanmask = torch.tensor([[False], [False], [False], [True]])
        transform, _ = _normalized(values, nanmask)
        props = transform.props['species_x_min']
        observed = transform._normalize_values('species_x_min', np.array([[1.0], [3.0]]))
        self.assertAlmostEqual(float(props['observed_min'][0]), observed[0, 0], places=6)
        self.assertAlmostEqual(float(props['observed_max'][0]), observed[1, 0], places=6)

    def test_gnn_and_ml_apply_the_same_rule(self):
        rng = np.random.default_rng(1)
        y = rng.lognormal(0, 1.2, size=(80, 2))
        pt = PowerTransformer(method='yeo-johnson', standardize=True).fit(y)
        y_t = pt.transform(y)
        transform, normalized = _normalized(y)
        # Same transformed-space predictions, including extrapolation, in both pipelines.
        query = np.array([[0.0, 0.5], [4.0, -4.0], [40.0, -40.0]])
        ml = bounded_inverse(pt.inverse_transform, query, y_t.min(0), y_t.max(0), nonnegative=(y >= 0).all(0))
        normalized.species_x_min = torch.tensor(query, dtype=torch.float32)
        gnn = transform.inverse(normalized, warn=False, bounded=True).species_x_min.numpy()
        # Both fit YJ + z on identical data (scipy vs sklearn lambda estimates agree closely).
        np.testing.assert_allclose(gnn, ml, rtol=2e-3)

    def test_inverse_tensor_uses_the_same_window(self):
        rng = np.random.default_rng(2)
        transform, _ = _normalized(rng.lognormal(0, 1.5, size=(60, 2)))
        props = transform.props['species_x_min']
        query = torch.tensor([[100.0, -100.0]], dtype=torch.float32)
        expected = transform._denormalize_values(
            'species_x_min', np.array([[props['observed_max'][0] + 1.0, props['observed_min'][1] - 1.0]]))
        got = transform.inverse_tensor('species_x_min', query).numpy()
        finite = np.isfinite(expected)
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_allclose(got[finite], expected[finite], rtol=1e-3)

    def test_nan_cells_stay_missing(self):
        out = bounded_inverse(lambda z: z, np.array([[np.nan, 1.0]]), np.zeros(2), np.ones(2), nonnegative=True)
        self.assertTrue(np.isnan(out[0, 0]))
        self.assertEqual(out[0, 1], 1.0)


class FailFastTests(unittest.TestCase):
    def test_large_finite_predictions_have_finite_float64_errors(self):
        for dtype, magnitude in [(torch.float32, 1e20), (torch.float64, 1e200)]:
            pred = torch.full((3, 1), magnitude, dtype=dtype)
            truth = torch.zeros_like(pred)
            with np.errstate(over='ignore', invalid='ignore'):
                metrics = compute_metrics_minmax(pred, pred, None, truth, truth, None,
                                                torch.ones_like(pred, dtype=torch.bool), ['Spore'])
            self.assertTrue(np.isfinite(metrics[['RMSE', 'MAE']]).all().all())
            np.testing.assert_allclose(metrics.RMSE, float(pred[0, 0]), rtol=1e-6)
            np.testing.assert_allclose(metrics.MAE, float(pred[0, 0]), rtol=1e-6)

    def test_infinite_errors_are_rejected_before_scoring(self):
        pred = torch.tensor([[float('inf')], [1.]])
        with self.assertRaises(AnalysisConsistencyError):
            compute_metrics_minmax(pred, pred, None, torch.zeros_like(pred), torch.zeros_like(pred), None,
                                   torch.ones_like(pred, dtype=torch.bool), ['Spore'])

    def test_finite_large_bounds_have_finite_midpoint(self):
        pmin, pmax = torch.tensor([[3e38]]), torch.tensor([[2e38]])
        low, high, _ = project_ordered_bounds(pmin, pmax)
        self.assertTrue(torch.isfinite(low).all() and torch.isfinite(high).all())
        torch.testing.assert_close(low, high)

    def test_long_export_rejects_nonfinite_evaluated_cells(self):
        pred = torch.tensor([[1.0, float('inf')]])
        truth = torch.ones(1, 2)
        observed = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaises(AnalysisConsistencyError):
            long_predictions_frame(['sp'], 'fold_0', ['a', 'b'], {'min': pred, 'max': truth},
                                   {'min': truth, 'max': truth}, observed, observed)
        # Non-finite but not evaluated: exported as is.
        eval_mask = torch.tensor([[True, False]])
        frame = long_predictions_frame(['sp'], 'fold_0', ['a', 'b'], {'min': pred, 'max': truth},
                                       {'min': truth, 'max': truth}, eval_mask, observed)
        self.assertEqual(len(frame), 4)


class MicePoolingTests(unittest.TestCase):
    def test_pooled_prediction_is_the_mean_of_completions(self):
        index = ['s1', 's2']
        completions = [
            pd.DataFrame({'aMin': [1.0, 2.0], 'habit': pd.Categorical(['x', 'y'])}, index=index),
            pd.DataFrame({'aMin': [3.0, 6.0], 'habit': pd.Categorical(['x', 'x'])}, index=index),
        ]
        pooled = pool_completions(completions)
        self.assertEqual(pooled['aMin'].tolist(), [2.0, 4.0])
        self.assertEqual(pooled['habit'].tolist(), ['x', 'y'])

    def test_mismatched_completions_raise(self):
        with self.assertRaises(ValueError):
            pool_completions([pd.DataFrame({'a': [1.0]}, index=['s1']), pd.DataFrame({'a': [1.0]}, index=['s2'])])


if __name__ == '__main__':
    unittest.main()
