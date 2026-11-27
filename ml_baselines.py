"""Tuned non-graph regressors for the min/max trait benchmark.

These baselines answer reviewer comments R2.2 and R4.7: conventional machine
learning models that receive species-level versions of the information used by
the GNN, evaluated on the same outer folds and occlusion groups as the other
baselines in ``r_baselines_fit.py``.

For every target trait ``j`` a separate multi-output regressor predicts its
minimum and maximum. Its inputs are:

* observed min/max values of all traits outside the occlusion group of ``j``
  (Yeo-Johnson + standardization, mean imputation and missingness indicators);
* optionally, categorical attributes (the same dummy encoding as the GNN),
  the 16-dimensional Node2Vec phylogenetic descriptors and species-level
  environmental summaries.

Dropping the whole occlusion group from the inputs reproduces the masking used
at test time (paired fertile/sterile groups by default), so training never
learns to rely on a counterpart that is hidden for test species. Every
preprocessing step and the target transformation are fitted on outer-training
species only, and hyperparameters are tuned with an inner species-level CV on
those same species.

Environmental inputs are occurrence-weighted summaries of the location nodes
connected to each species. They approximate, but do not reproduce, the
location graph and its message passing.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from occlusion import group_for_trait
from prediction_bounds import DEFAULT_EXTRAPOLATION_MARGIN, bounded_inverse

ML_MODEL_NAMES = ('random_forest', 'xgboost', 'mlp')


def transformed_target_score(estimator, X, y):
    """Score validation targets on the scale fitted on this inner training fold."""
    truth = estimator.transformer_.transform(y)
    prediction = estimator.regressor_.predict(X)
    return -float(np.mean((prediction - truth) ** 2))


def _phylo_frame(data) -> pd.DataFrame:
    values = data.species_x_phylo.detach().cpu().numpy().astype(np.float64)
    return pd.DataFrame(
        values,
        index=list(data.species_names),
        columns=[f'phylo_{i}' for i in range(values.shape[1])],
    )


def _categorical_frame(data) -> pd.DataFrame:
    values = data.species_x_gen.detach().cpu().numpy().astype(np.float64)
    names = list(getattr(data, 'genetic_feature_names', []))
    if len(names) != values.shape[1]:
        names = [f'{i}' for i in range(values.shape[1])]
    return pd.DataFrame(values, index=list(data.species_names), columns=[f'cat_{n}' for n in names])


def species_environment_summary(data) -> pd.DataFrame:
    """Occurrence-weighted mean and SD of location-node covariates per species.

    Location features are the cluster centroid coordinates and the aggregated
    climate, soil, population and elevation layers. Missing location values are
    excluded from each weighted statistic. Species without occurrence edges get
    missing summaries (imputed downstream with a missingness indicator), a zero
    total occurrence and zero occupied clusters.
    """
    n_species = int(data.species_num_nodes)
    n_locations = int(data.spatial_num_nodes)
    location_features = torch.cat(
        [data.spatial_pos.double(), data.spatial_global_data.double()], dim=1
    ).cpu().numpy()
    # Sanitized names: the same columns are passed to R (MICE, missForest).
    feature_names = ['centroid_x', 'centroid_y'] + [
        re.sub(r'\W+', '_', str(name)).strip('_') for name in getattr(
            data, 'spatial_global_feature_names',
            [f'env_{i}' for i in range(data.spatial_global_data.size(1))],
        )
    ]
    if len(set(feature_names)) != len(feature_names):
        raise ValueError('Environmental feature names collide after sanitization.')

    location_index, species_index = data.spatial_species_edge_index.cpu().numpy()
    weights = data.spatial_species_edge_attr.view(-1).double().cpu().numpy()
    if (weights < 0).any():
        raise ValueError('Occurrence weights must be non-negative.')
    weight_matrix = np.zeros((n_species, n_locations), dtype=np.float64)
    np.add.at(weight_matrix, (species_index, location_index), weights)

    observed = np.isfinite(location_features)
    filled = np.where(observed, location_features, 0.0)
    denominator = weight_matrix @ observed.astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = (weight_matrix @ filled) / denominator
        second_moment = (weight_matrix @ filled ** 2) / denominator
    variance = np.clip(second_moment - mean ** 2, 0.0, None)
    mean[denominator <= 0] = np.nan
    variance[denominator <= 0] = np.nan

    summary = pd.DataFrame(
        np.hstack([mean, np.sqrt(variance)]),
        index=list(data.species_names),
        columns=[f'env_mean_{n}' for n in feature_names] + [f'env_sd_{n}' for n in feature_names],
    )
    summary['env_log_total_occurrence'] = np.log1p(weight_matrix.sum(axis=1))
    summary['env_n_locations'] = (weight_matrix > 0).sum(axis=1).astype(np.float64)
    return summary


def build_ml_covariates(data, use_categorical: bool, use_phylo: bool, use_env: bool) -> pd.DataFrame:
    """Non-trait species covariates for the ML baselines (one row per species)."""
    parts = []
    if use_categorical:
        parts.append(_categorical_frame(data))
    if use_phylo:
        parts.append(_phylo_frame(data))
    if use_env:
        parts.append(species_environment_summary(data))
    if not parts:
        return pd.DataFrame(index=list(data.species_names))
    return pd.concat(parts, axis=1)


def _estimator_and_space(model_name: str, seed: int):
    if model_name == 'random_forest':
        estimator = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=1)
        space = {
            'max_features': [0.2, 0.33, 0.5, 0.75, 1.0],
            'min_samples_leaf': [1, 2, 3, 5, 8],
            'max_depth': [None, 5, 10, 20],
        }
    elif model_name == 'xgboost':
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            objective='reg:squarederror', tree_method='hist', random_state=seed, n_jobs=1,
        )
        space = {
            'n_estimators': [100, 200, 400, 800],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'max_depth': [2, 3, 4, 6],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.7, 1.0],
            'reg_lambda': [0.1, 1.0, 10.0],
            'reg_alpha': [0.0, 0.1, 1.0],
        }
    elif model_name == 'mlp':
        estimator = MLPRegressor(
            max_iter=2000, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=50, random_state=seed,
        )
        space = {
            'hidden_layer_sizes': [(32,), (64,), (128,), (64, 32), (128, 64), (150, 150)],
            'activation': ['relu', 'tanh'],
            'alpha': loguniform(1e-5, 1.0),
            'learning_rate_init': [3e-4, 1e-3, 3e-3],
            'batch_size': [16, 32, 'auto'],
        }
    else:
        raise ValueError(f"Unknown ML baseline '{model_name}'")
    return estimator, {f'model__{key}': value for key, value in space.items()}


class TunedTraitRegressor:
    """One tuned multi-output (min, max) regressor per target trait."""

    def __init__(self, model_name: str, trait_names: list[str],
                 trait_col_names: dict[str, list[str]], occlusion_groups,
                 seed: int, n_iter: int = 30, inner_folds: int = 5, n_jobs: int = -1,
                 extrapolation_margin: float = DEFAULT_EXTRAPOLATION_MARGIN, fixed_params: dict | None = None) -> None:
        if model_name not in ML_MODEL_NAMES:
            raise ValueError(f"Unknown ML baseline '{model_name}'")
        self.model_name = model_name
        self.trait_names = list(trait_names)
        self.trait_col_names = trait_col_names
        self.variables = list(trait_col_names)
        self.trait_columns = [c for v in self.variables for c in trait_col_names[v]]
        self.occlusion_groups = occlusion_groups
        self.seed = seed
        self.n_iter = n_iter
        self.inner_folds = inner_folds
        self.n_jobs = n_jobs
        self.extrapolation_margin = extrapolation_margin
        self.fixed_params = fixed_params
        self.covariate_columns: list[str] | None = None
        self.models: dict[int, dict] = {}
        self.tuning_records: list[dict] = []
        self.prediction_diagnostics: list[dict] = []

    def _target_columns(self, j: int) -> list[str]:
        return [self.trait_col_names[v][j] for v in self.variables]

    def _input_trait_columns(self, j: int) -> list[str]:
        hidden = set(group_for_trait(self.occlusion_groups, j))
        return [
            self.trait_col_names[v][i]
            for v in self.variables for i in range(len(self.trait_names)) if i not in hidden
        ]

    def _pipeline(self, trait_inputs: list[str], estimator) -> Pipeline:
        transformers = []
        if trait_inputs:
            transformers.append(('traits', Pipeline([
                ('yeo_johnson', PowerTransformer(method='yeo-johnson', standardize=True)),
                ('impute', SimpleImputer(strategy='mean', add_indicator=True)),
            ]), trait_inputs))
        if self.covariate_columns:
            transformers.append(('covariates', Pipeline([
                ('impute', SimpleImputer(strategy='median', add_indicator=True)),
                ('scale', StandardScaler()),
            ]), self.covariate_columns))
        if not transformers:
            raise ValueError('The ML baseline has no input features for this target.')
        return Pipeline([
            ('features', ColumnTransformer(transformers, remainder='drop')),
            ('model', estimator),
        ])

    def fit(self, data: pd.DataFrame) -> 'TunedTraitRegressor':
        missing = [c for c in self.trait_columns if c not in data.columns]
        if missing:
            raise ValueError(f'Training data lacks trait columns: {missing}')
        self.covariate_columns = [c for c in data.columns if c not in self.trait_columns]
        self.models.clear()
        self.tuning_records.clear()
        self.prediction_diagnostics.clear()

        for j, trait in enumerate(self.trait_names):
            target_columns = self._target_columns(j)
            trait_inputs = self._input_trait_columns(j)
            rows = data[target_columns].notna().all(axis=1)
            n_train = int(rows.sum())
            if self.fixed_params is None and n_train < self.inner_folds:
                raise ValueError(f"Trait '{trait}' has only {n_train} training targets.")
            X = data.loc[rows, trait_inputs + self.covariate_columns]
            y = data.loc[rows, target_columns].to_numpy(dtype=np.float64)

            estimator, space = _estimator_and_space(self.model_name, self.seed)
            base = TransformedTargetRegressor(
                regressor=self._pipeline(trait_inputs, estimator),
                transformer=PowerTransformer(method='yeo-johnson', standardize=True),
            )
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                warnings.filterwarnings('ignore', message='The total space of parameters')
                if self.fixed_params is None:
                    search = RandomizedSearchCV(
                        base, {f'regressor__{key}': value for key, value in space.items()},
                        n_iter=self.n_iter,
                        cv=KFold(n_splits=self.inner_folds, shuffle=True, random_state=self.seed),
                        scoring=transformed_target_score, random_state=self.seed,
                        n_jobs=self.n_jobs, refit=True, error_score='raise',
                    )
                    search.fit(X, y)
                else:
                    if trait not in self.fixed_params:
                        raise ValueError(f'Missing fixed parameters for {self.model_name}/{trait}')
                    params = dict(self.fixed_params[trait])
                    if 'hidden_layer_sizes' in params:
                        params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
                    prefixed = {f'regressor__model__{key}': value for key, value in params.items()}
                    base.set_params(**prefixed)
                    base.fit(X, y)
                    search = SimpleNamespace(best_estimator_=base, best_params_=prefixed, best_score_=None)
            elapsed = time.perf_counter() - start
            # RandomizedSearchCV refits the winner on all outer-training rows
            # only after tuning with fold-local feature AND target transforms.
            target_transform = search.best_estimator_.transformer_
            y_t = target_transform.transform(y)

            self.models[j] = {
                'search': search,
                'target_transform': target_transform,
                'inputs': trait_inputs + self.covariate_columns,
                'y_t_min': y_t.min(axis=0),
                'y_t_max': y_t.max(axis=0),
                'nonnegative': (y >= 0).all(axis=0),
            }
            n_features = int(search.best_estimator_.regressor_.named_steps['features'].transform(X.iloc[:1]).shape[1])
            self.tuning_records.append({
                'trait': trait,
                'hidden_inputs': ','.join(self.trait_names[i] for i in group_for_trait(self.occlusion_groups, j)),
                'n_train': n_train,
                'n_input_columns': len(trait_inputs) + len(self.covariate_columns),
                'n_model_features': n_features,
                'inner_cv_mse_transformed': float(-search.best_score_) if search.best_score_ is not None else None,
                'parameter_selection': 'fixed' if self.fixed_params is not None else 'inner_cv',
                'best_params': json.dumps(
                    {k.removeprefix('regressor__model__'): v for k, v in search.best_params_.items()}, default=str,
                ),
                'tuning_seconds': elapsed,
            })
        return self

    def transform(self, data: pd.DataFrame, target_index: int, evaluation_mask=None) -> pd.DataFrame:
        """Predict the min/max columns of one trait; occluded inputs are never read."""
        if target_index not in self.models:
            raise RuntimeError('fit() must be called before transform()')
        fitted = self.models[target_index]
        target_columns = self._target_columns(target_index)
        if data.empty:
            return pd.DataFrame(columns=target_columns, index=data.index, dtype=np.float64)
        pred_t = np.asarray(fitted['search'].best_estimator_.regressor_.predict(data.loc[:, fitted['inputs']]), dtype=np.float64)
        pred_t = pred_t.reshape(len(data), len(target_columns))

        # Same original-scale rule as the GNN (NormalizeFeatures.inverse(bounded=True)):
        # bounded extrapolation in the transformed space, finite fallback, then
        # nonnegative clipping for targets whose training values were all >= 0.
        diagnostics = []
        selected = None if evaluation_mask is None else np.repeat(
            np.asarray(evaluation_mask, dtype=bool)[:, None], len(target_columns), axis=1,
        )
        pred = bounded_inverse(
            fitted['target_transform'].inverse_transform, pred_t, fitted['y_t_min'], fitted['y_t_max'],
            margin=self.extrapolation_margin, nonnegative=fitted['nonnegative'],
            diagnostics=diagnostics, evaluation_mask=selected,
        )
        for row in diagnostics:
            column = row.pop('column')
            self.prediction_diagnostics.append({
                'stage': 'test', 'trait': self.trait_names[target_index],
                'variable': self.variables[column], **row,
            })
        return pd.DataFrame(pred, index=data.index, columns=target_columns)
