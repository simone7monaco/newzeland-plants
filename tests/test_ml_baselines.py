import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import PowerTransformer

from ml_baselines import TunedTraitRegressor, build_ml_covariates, species_environment_summary
from occlusion import trait_groups


NAMES = ['Ferstipe', 'SterileStipe', 'Spore']
COLS = {v: [f'{n}{v.title()}' for n in NAMES] for v in ('min', 'max')}


def synthetic_table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(3.0, 0.5, size=(n, len(NAMES)))
    frame = pd.DataFrame(
        np.hstack([base, base * 1.5]), columns=COLS['min'] + COLS['max'],
        index=[f'sp{i}' for i in range(n)],
    )
    frame['cov'] = rng.normal(size=n)
    frame.iloc[::7, [2, 5]] = np.nan  # Spore missing for some species
    return frame


class MLBaselineTests(unittest.TestCase):
    def test_fixed_parameters_skip_search_and_fit_predictions(self):
        params = {name: {'max_depth': 5, 'min_samples_leaf': 2, 'max_features': 0.5} for name in NAMES}
        with patch('ml_baselines.RandomizedSearchCV', side_effect=AssertionError('unexpected search')):
            model = TunedTraitRegressor(
                'random_forest', NAMES, COLS, trait_groups(NAMES, 'paired'),
                seed=0, fixed_params=params,
            ).fit(synthetic_table(n=30))
        self.assertEqual(model.tuning_records[0]['parameter_selection'], 'fixed')
        self.assertTrue(np.isfinite(model.transform(synthetic_table(n=4), 0).to_numpy()).all())

    def test_target_transform_is_fitted_inside_each_inner_fold(self):
        table = synthetic_table(n=30).dropna()
        fit_sizes = []
        original_fit = PowerTransformer.fit

        def record_fit(transformer, X, y=None):
            # Targets have two columns; trait predictors have four columns.
            if X.shape[1] == 2:
                fit_sizes.append(len(X))
            return original_fit(transformer, X, y)

        with patch.object(PowerTransformer, 'fit', record_fit):
            model = TunedTraitRegressor(
                'random_forest', NAMES, COLS, trait_groups(NAMES, 'paired'),
                seed=0, n_iter=1, inner_folds=3, n_jobs=1,
            ).fit(table)
        # Each target has three inner fits followed by one outer refit.
        self.assertEqual(len(fit_sizes), 4 * len(NAMES))
        for start in range(0, len(fit_sizes), 4):
            self.assertTrue(all(n < len(table) for n in fit_sizes[start:start + 3]))
            self.assertEqual(fit_sizes[start + 3], len(table))
        self.assertTrue(np.isfinite(model.transform(table.iloc[:3], 0)).all().all())

    def test_environment_summary_is_occurrence_weighted_and_nan_aware(self):
        data = Data(
            species_names=['a', 'b', 'c'], species_num_nodes=3, spatial_num_nodes=2,
            spatial_pos=torch.tensor([[0.0, 0.0], [2.0, 4.0]]),
            spatial_global_data=torch.tensor([[1.0, float('nan')], [3.0, 5.0]]),
            spatial_global_feature_names=['t', 'p'],
            spatial_species_edge_index=torch.tensor([[0, 1, 1], [0, 0, 1]]),  # [location, species]
            spatial_species_edge_attr=torch.tensor([[1.0], [3.0], [2.0]]),
        )
        env = species_environment_summary(data)
        self.assertAlmostEqual(env.loc['a', 'env_mean_t'], 2.5)
        self.assertAlmostEqual(env.loc['a', 'env_mean_centroid_y'], 3.0)
        self.assertAlmostEqual(env.loc['a', 'env_mean_p'], 5.0)  # NaN location excluded
        self.assertAlmostEqual(env.loc['b', 'env_sd_t'], 0.0)
        self.assertTrue(env.loc['c'].filter(like='env_mean').isna().all())
        self.assertEqual(env.loc['c', 'env_n_locations'], 0.0)
        self.assertEqual(env.loc['a', 'env_n_locations'], 2.0)

    def test_covariate_blocks_follow_switches(self):
        data = Data(
            species_names=['a', 'b'], species_x_gen=torch.eye(2), genetic_feature_names=['f1', 'f2'],
            species_x_phylo=torch.zeros(2, 3),
        )
        cov = build_ml_covariates(data, use_categorical=True, use_phylo=True, use_env=False)
        self.assertEqual(list(cov.columns), ['cat_f1', 'cat_f2', 'phylo_0', 'phylo_1', 'phylo_2'])
        self.assertEqual(build_ml_covariates(data, False, False, False).shape, (2, 0))

    def test_occluded_group_is_never_read_and_predictions_are_valid(self):
        table = synthetic_table()
        train, test = table.iloc[:45], table.iloc[45:].copy()
        for name in ('random_forest', 'xgboost', 'mlp'):
            model = TunedTraitRegressor(
                name, NAMES, COLS, trait_groups(NAMES, 'paired'), seed=0,
                n_iter=2, inner_folds=3, n_jobs=1,
            ).fit(train)
            self.assertEqual(len(model.tuning_records), len(NAMES))
            original = model.transform(test, 0)
            scrambled = test.copy()
            # Target and paired counterpart (the occlusion group) are corrupted.
            scrambled[[COLS['min'][0], COLS['max'][0]]] = 1e6
            scrambled[[COLS['min'][1], COLS['max'][1]]] = -1e6
            pd.testing.assert_frame_equal(original, model.transform(scrambled, 0))
            self.assertEqual(list(original.columns), ['FerstipeMin', 'FerstipeMax'])
            self.assertTrue(np.isfinite(original.to_numpy()).all())
            self.assertTrue((original.to_numpy() >= 0).all())
            selected = np.arange(len(test)) % 2 == 0
            with patch.object(model.models[0]['search'].best_estimator_.regressor_, 'predict',
                              return_value=np.full((len(test), 2), 1e6)):
                model.transform(test, 0, evaluation_mask=selected)
            for record in model.prediction_diagnostics[-2:]:
                self.assertEqual(record['stage'], 'test')
                self.assertEqual(record['trait'], 'Ferstipe')
                self.assertEqual(record['n_clipped_high'], len(test))
                self.assertEqual(record['n_eval_clipped_high'], int(selected.sum()))


if __name__ == '__main__':
    unittest.main()
