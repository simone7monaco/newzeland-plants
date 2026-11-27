"""Post-hoc projection of predicted bounds onto min <= max."""
import unittest

import torch

import pandas as pd

from tester import Tester, count_reversed_bounds, project_ordered_bounds


class BoundsProjectionTests(unittest.TestCase):
    def test_projection_orders_pairs_and_keeps_ordered_ones(self):
        pmin = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        pmax = torch.tensor([[2.0, 4.0], [3.0, 8.0]])
        qmin, qmax, reversed_pairs = project_ordered_bounds(pmin, pmax)
        self.assertTrue((qmin <= qmax).all())
        self.assertEqual(reversed_pairs.tolist(), [[False, True], [False, False]])
        torch.testing.assert_close(qmin, torch.tensor([[1.0, 4.5], [3.0, 2.0]]))
        torch.testing.assert_close(qmax, torch.tensor([[2.0, 4.5], [3.0, 8.0]]))

    def test_projection_never_increases_joint_error_for_ordered_truth(self):
        torch.manual_seed(0)
        tmin = torch.rand(500) * 10
        tmax = tmin + torch.rand(500) * 5
        pmin, pmax = tmin + torch.randn(500) * 3, tmax + torch.randn(500) * 3
        qmin, qmax, _ = project_ordered_bounds(pmin, pmax)
        before = (pmin - tmin) ** 2 + (pmax - tmax) ** 2
        after = (qmin - tmin) ** 2 + (qmax - tmax) ** 2
        self.assertTrue((after <= before + 1e-5).all())


    def test_tester_summarizes_reversed_pairs_over_folds(self):
        tester = Tester('cpu')
        for fold, (n_rev, n_eval_rev) in enumerate([(2, 1), (0, 0)]):
            tester._bounds_order_dfs.append(pd.DataFrame({
                'trait': ['A'], 'n_pairs': [10], 'n_reversed': [n_rev], 'n_eval_pairs': [8],
                'n_eval_reversed': [n_eval_rev], 'max_reversal': [0.5 * n_rev], 'projected': [True],
                'fold': [f'fold_{fold}']}))
        summary = tester.bounds_order_summary()
        self.assertEqual(summary.loc[0, 'n_pairs'], 20)
        self.assertEqual(summary.loc[0, 'n_reversed'], 2)
        self.assertEqual(summary.loc[0, 'n_eval_reversed'], 1)
        self.assertAlmostEqual(summary.loc[0, 'pct_reversed'], 10.0)


    def test_count_reversed_bounds_ignores_nan_and_counts_before_projection(self):
        nan = float('nan')
        pmin = torch.tensor([[1.0, 5.0], [4.0, nan], [2.0, 9.0]])
        pmax = torch.tensor([[2.0, 4.0], [3.0, 7.0], [2.0, 8.0]])
        evaluated = torch.tensor([[True, True], [True, True], [False, False]])
        table = count_reversed_bounds(pmin, pmax, evaluated, ['A', 'B'], projected=True)
        self.assertEqual(table.n_pairs.tolist(), [3, 2])          # the NaN pair is not counted
        self.assertEqual(table.n_reversed.tolist(), [1, 2])
        self.assertEqual(table.n_eval_reversed.tolist(), [1, 1])  # the reversed pair in row 2 is not scored
        self.assertEqual(table.max_reversal.tolist(), [1.0, 1.0])
        # counting must not modify its inputs
        self.assertTrue((pmin[1, 0] > pmax[1, 0]).item())


if __name__ == '__main__':
    unittest.main()
