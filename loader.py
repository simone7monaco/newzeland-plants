import warnings
# torch_geometric (<= 2.8.1) calls the deprecated torch.jit.script at import time; silence only that message.
warnings.filterwarnings('ignore', message='`torch.jit.script` is deprecated', category=FutureWarning)
import sys
from typing import cast
import warnings
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import itertools
import pandas as pd
import re
import hashlib
import io
from Bio import Phylo
import networkx as nx

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"node2vec\.edges",
)
from node2vec import Node2Vec
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import GroupKFold, KFold
from scipy.stats import yeojohnson as _scipy_yeojohnson
from scipy.spatial import cKDTree

import torch
from torch_geometric.transforms import KNNGraph, BaseTransform
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.utils import from_networkx, subgraph, bipartite_subgraph

from prediction_bounds import DEFAULT_EXTRAPOLATION_MARGIN, bounded_inverse

FEATURE_CACHE_VERSION = 3
PROCESSED_DATA_VERSION = 9


def _yj_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Apply Yeo-Johnson transform element-wise with the given lambda."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    lam = float(lmbda)
    if abs(lam) < 1e-10:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1.0, lam) - 1.0) / lam
    if abs(lam - 2.0) < 1e-10:
        out[neg] = -np.log1p(-x[neg])
    else:
        out[neg] = -(np.power(-x[neg] + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def _yj_inverse(y: np.ndarray, lmbda: float) -> np.ndarray:
    """Invert the Yeo-Johnson transform element-wise."""
    out = np.empty_like(y, dtype=np.float64)
    pos = y >= 0
    neg = ~pos
    lam = float(lmbda)
    if abs(lam) < 1e-10:
        out[pos] = np.expm1(y[pos])
    else:
        out[pos] = np.power(np.clip(y[pos] * lam + 1.0, 0.0, None), 1.0 / lam) - 1.0
    if abs(lam - 2.0) < 1e-10:
        out[neg] = 1.0 - np.exp(-y[neg])
    else:
        out[neg] = 1.0 - np.power(np.clip(-(2.0 - lam) * y[neg] + 1.0, 0.0, None), 1.0 / (2.0 - lam))
    return out


class NormalizeFeatures(BaseTransform):
    r"""Normalizes graph attributes; traits can use per-column Yeo-Johnson + z, others use z or sin/cos."""
    def __init__(self, trait_norm_mean='yj', trait_norm_std='z',
                 trait_norm_min='yj', trait_norm_max='yj', trait_norm_range='yj', shared_bounds=False):

        if shared_bounds and trait_norm_min != trait_norm_max:
            raise ValueError('Ordered bounds require the same normalization mode for min and max.')
        self.shared_bounds = shared_bounds

        self.normalizations = {
            'spatial_x': 'sincos', # They are latitude and longitude
            'spatial_global_data': 'z',
            'species_x_gen': 'z',
            'species_x_mean': trait_norm_mean,
            'species_x_std': trait_norm_std,
            'species_x_min': trait_norm_min,
            'species_x_max': trait_norm_max,
            'species_x_range': trait_norm_range,
            'species_x_phylo': None, # x_phylo is a vector of embeddings, so no normalization
            'spatial_spatial_edge_attr': 'z',
            'species_species_edge_attr': 'z',
            'spatial_species_edge_attr': 'z',
        }
        self.props = {}
        self.prediction_diagnostics = []
        self.trait_attributes = {
            'species_x_mean', 'species_x_std', 'species_x_min',
            'species_x_max', 'species_x_range',
        }

    def fit(self, data: Data, eps: float = 1e-6):
        self.prediction_diagnostics.clear()
        if getattr(data, 'props', False):
            print("Warning: Data already has props, overwriting\n")

        for k_norm in self.normalizations:
            if k_norm not in data:
                continue
            if self.normalizations[k_norm] == 'yj':
                # Per-column Yeo-Johnson fit (lambda) + post-transform z-statistics.
                # Masked/missing entries (traits_nanmask) are excluded from fitting.
                data_np = data[k_norm].float().numpy()  # (n_species, n_traits)
                n_traits = data_np.shape[1]
                lambdas = np.ones(n_traits, dtype=np.float64)
                post_mean = np.zeros(n_traits, dtype=np.float64)
                post_std = np.ones(n_traits, dtype=np.float64)
                nan_mask_np = (
                    data.traits_nanmask.numpy()
                    if (hasattr(data, 'traits_nanmask') and k_norm in self.trait_attributes)
                    else np.zeros_like(data_np, dtype=bool)
                )
                for j in range(n_traits):
                    valid = data_np[~nan_mask_np[:, j], j]
                    if len(valid) >= 2:
                        transformed, lam = _scipy_yeojohnson(valid)
                        transformed_array = np.asarray(transformed, dtype=np.float64)
                        lambdas[j] = lam
                        post_mean[j] = np.mean(transformed_array)
                        s = np.std(transformed_array)
                        post_std[j] = s if s > eps else 1.0
                self.props[k_norm] = {
                    'lambdas': torch.tensor(lambdas, dtype=torch.float32),
                    'mean': torch.tensor(post_mean, dtype=torch.float32),
                    'std': torch.tensor(post_std, dtype=torch.float32),
                }
            elif self.normalizations[k_norm] in ['logz', 'z']:
                values = data[k_norm].float()
                is_trait = k_norm in self.trait_attributes and hasattr(data, 'traits_nanmask')
                observed_mask = ~data.traits_nanmask if is_trait else None
                if self.normalizations[k_norm] == 'logz':
                    valid_values = values[observed_mask] if observed_mask is not None else values
                    if (valid_values < 0).any():
                        raise ValueError(
                            f"logz normalization requires non-negative observed values in {k_norm}; "
                            "use yj normalization for traits with negative measurements."
                        )
                    data_k_norm = torch.log(values + eps)
                else:
                    data_k_norm = values

                if data_k_norm.dim() == 1:
                    data_k_norm = data_k_norm.unsqueeze(1)

                mean = torch.zeros(data_k_norm.size(1), dtype=data_k_norm.dtype)
                std = torch.ones(data_k_norm.size(1), dtype=data_k_norm.dtype)
                for j in range(data_k_norm.size(1)):
                    valid = data_k_norm[:, j]
                    if observed_mask is not None:
                        valid = valid[observed_mask[:, j]]
                    valid = valid[torch.isfinite(valid)]
                    if valid.numel() == 0:
                        continue
                    mean[j] = valid.mean()
                    if valid.numel() > 1:
                        value_std = valid.std(unbiased=False)
                        if value_std >= eps:
                            std[j] = value_std
                self.props[k_norm] = {'mean': mean, 'std': std}
            elif self.normalizations[k_norm] in ['sincos', None]:
                pass
            else:
                raise ValueError(f"Unknown normalization: {self.normalizations[k_norm]}")
        if self.shared_bounds:
            if 'species_x_min' not in self.props or 'species_x_max' not in self.props:
                raise ValueError('Ordered bounds require fitted min and max features.')
            # The same monotone mapping is essential: separately fitted maps can
            # reverse the order after inversion even when normalized min <= max.
            self.props['species_x_max'] = {
                name: value.clone() for name, value in self.props['species_x_min'].items()
            }
        self._eps = eps
        self._fit_prediction_ranges(data)

    # Trait outputs that are mapped back to the original scale with the shared
    # bounded-extrapolation rule (prediction_bounds.bounded_inverse).
    BOUNDED_TRAIT_KEYS = ('species_x_mean', 'species_x_min', 'species_x_max', 'species_x_range')

    def _normalize_values(self, k_norm: str, values: np.ndarray) -> np.ndarray:
        """Original -> normalized scale for a (n, k) array (same maps as forward())."""
        mode, props = self.normalizations[k_norm], self.props[k_norm]
        mean, std = props['mean'].double().numpy(), props['std'].double().numpy()
        values = np.asarray(values, dtype=np.float64)
        if mode == 'yj':
            lambdas = props['lambdas'].numpy()
            values = np.column_stack([_yj_transform(values[:, j], lambdas[j]) for j in range(values.shape[1])])
        elif mode == 'logz':
            with np.errstate(divide='ignore', invalid='ignore'):
                values = np.log(values + getattr(self, '_eps', 1e-6))
        elif mode != 'z':
            raise ValueError(f'No trait inverse for normalization {mode!r}.')
        return (values - mean) / std

    def _denormalize_values(self, k_norm: str, values: np.ndarray) -> np.ndarray:
        """Normalized -> original scale for a (n, k) array, without any bounding."""
        mode, props = self.normalizations[k_norm], self.props[k_norm]
        y = np.asarray(values, dtype=np.float64) * props['std'].double().numpy() + props['mean'].double().numpy()
        if mode == 'yj':
            lambdas = props['lambdas'].numpy()
            return np.column_stack([_yj_inverse(y[:, j], lambdas[j]) for j in range(y.shape[1])])
        if mode == 'logz':
            with np.errstate(over='ignore'):
                return np.exp(y) - 1e-6
        if mode == 'z':
            return y
        raise ValueError(f'No trait inverse for normalization {mode!r}.')

    def _fit_prediction_ranges(self, data: Data) -> None:
        """Store, per trait output, the normalized range of the fitted (observed)
        values and whether they were all nonnegative. These parameterize the
        bounded inverse shared with the ML baselines."""
        for k_norm in self.BOUNDED_TRAIT_KEYS:
            if k_norm not in data or k_norm not in self.props or self.normalizations[k_norm] not in ('yj', 'z', 'logz'):
                continue
            values = data[k_norm].double().cpu().numpy()
            if values.ndim == 1:
                values = values[:, None]
            observed = (~data.traits_nanmask.cpu().numpy()) if hasattr(data, 'traits_nanmask') else np.ones(values.shape, dtype=bool)
            observed = observed & np.isfinite(values)
            normalized = np.where(observed, self._normalize_values(k_norm, np.where(observed, values, 0.0)), np.nan)
            has_obs = observed.any(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                low = np.where(has_obs, np.nanmin(normalized, axis=0), 0.0)
                high = np.where(has_obs, np.nanmax(normalized, axis=0), 0.0)
            self.props[k_norm]['observed_min'] = torch.tensor(low, dtype=torch.float64)
            self.props[k_norm]['observed_max'] = torch.tensor(high, dtype=torch.float64)
            self.props[k_norm]['nonnegative'] = torch.tensor((np.where(observed, values, 0.0) >= 0).all(axis=0))
        if self.shared_bounds and all('observed_min' in self.props.get(k, {}) for k in ('species_x_min', 'species_x_max')):
            # min and max share one monotone map; clipping both to the union of
            # their ranges keeps that map (and therefore their order) identical.
            pmin, pmax = self.props['species_x_min'], self.props['species_x_max']
            low = torch.minimum(pmin['observed_min'], pmax['observed_min'])
            high = torch.maximum(pmin['observed_max'], pmax['observed_max'])
            for props in (pmin, pmax):
                props['observed_min'], props['observed_max'] = low.clone(), high.clone()
            
    def transform(self, data: Data, eps: float = 1e-6) -> Data:
        return self.forward(data, eps)

    def fit_transform(self, data: Data, eps: float = 1e-6) -> Data:
        self.fit(data, eps)
        return self.forward(data, eps)

    def forward(self, data: Data, eps: float = 1e-6) -> Data:
        if getattr(data, 'normalized', False):
            raise ValueError("Data is already normalized")

        if not self.props:
            print("Warning: No props found, computing on the fly (this may cause data leakage if done on the whole dataset instead of just the training set)\n")
            self.fit(data, eps)

        setattr(data, 'normalized', True)
        for k_norm in self.normalizations:
            if k_norm not in data:
                continue
            if self.normalizations[k_norm] == 'sincos':
                data[k_norm] = data[k_norm] * np.pi / 180
                data[k_norm] = torch.stack([torch.sin(data[k_norm][:, 0]), torch.cos(data[k_norm][:, 0]),
                                          torch.sin(data[k_norm][:, 1]), torch.cos(data[k_norm][:, 1])], dim=1)
            elif self.normalizations[k_norm] == 'yj':
                props = self.props[k_norm]
                lambdas = props['lambdas'].numpy()
                data_np = data[k_norm].float().cpu().numpy()
                for j in range(data_np.shape[1]):
                    data_np[:, j] = _yj_transform(data_np[:, j], lambdas[j])
                result = torch.tensor(data_np, dtype=data[k_norm].dtype, device=data[k_norm].device)
                data[k_norm] = (result - props['mean'].to(result.device)) / props['std'].to(result.device)
            elif self.normalizations[k_norm] in ['logz', 'z']:
                if self.normalizations[k_norm] == 'logz':
                    data[k_norm] = torch.log(data[k_norm] + eps)
                normalized = (data[k_norm] - self.props[k_norm]['mean']) / self.props[k_norm]['std']
                data[k_norm] = torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
            elif self.normalizations[k_norm] is None:
                pass
            else:
                raise ValueError(f"Unknown normalization: {self.normalizations[k_norm]}")
            if k_norm in self.trait_attributes:
                # Zero out masked entries so they equal the mean (0) in normalized space
                data[k_norm] = data[k_norm] * ~data.traits_nanmask
        return data
    
    def inverse_tensor(self, key: str, values: torch.Tensor, clamp: float = 6.0,
                       margin: float = DEFAULT_EXTRAPOLATION_MARGIN) -> torch.Tensor:
        """Differentiable inverse of a fitted trait transform (yj, z, logz).

        Normalized inputs are clamped to the fitted observed range extended by
        ``margin`` SDs (the same window as ``inverse(bounded=True)`` and the ML
        baselines); without a fitted range, to +-clamp SDs. Power bases are also
        kept inside the Yeo-Johnson inverse domain, so outputs stay finite.
        """
        if torch.isinf(values).any():
            raise FloatingPointError('raw predictions contain infinity; refusing to clip invalid model outputs')
        mode = self.normalizations[key]
        props = self.props[key]
        mean, std = props['mean'].to(values.device), props['std'].to(values.device)
        if 'observed_min' in props:
            low = (props['observed_min'] - margin).to(device=values.device, dtype=values.dtype)
            high = (props['observed_max'] + margin).to(device=values.device, dtype=values.dtype)
            y = torch.maximum(torch.minimum(values, high), low) * std + mean
        else:
            y = values.clamp(-clamp, clamp) * std + mean
        if mode == 'z':
            return y
        if mode == 'logz':
            return torch.exp(y) - 1e-6
        if mode != 'yj':
            raise ValueError(f'No differentiable inverse for normalization {mode!r}.')
        lam = props['lambdas'].to(values.device).expand_as(y)
        pos = y >= 0
        near0, near2 = lam.abs() < 1e-10, (lam - 2.0).abs() < 1e-10
        safe_lam = torch.where(near0, torch.ones_like(lam), lam)
        safe_2ml = torch.where(near2, torch.ones_like(lam), 2.0 - lam)
        # Mask inputs per branch (no inf/NaN in the unused branch) and keep the
        # power bases strictly inside the inverse domain.
        y_pos = torch.where(pos, y, torch.zeros_like(y))
        y_neg = torch.where(pos, torch.zeros_like(y), y)
        pos_base = (y_pos * safe_lam + 1.0).clamp(min=1e-3)
        neg_base = (-safe_2ml * y_neg + 1.0).clamp(min=1e-3)
        pos_val = torch.where(near0, torch.expm1(torch.where(near0, y_pos, torch.zeros_like(y_pos))), pos_base ** (1.0 / safe_lam) - 1.0)
        neg_val = torch.where(near2, 1.0 - torch.exp(-torch.where(near2, y_neg, torch.zeros_like(y_neg))), 1.0 - neg_base ** (1.0 / safe_2ml))
        return torch.where(pos, pos_val, neg_val)

    def inverse(self, data, warn=True, bounded=False, margin=DEFAULT_EXTRAPOLATION_MARGIN,
                stage='prediction', trait_names=None, evaluation_mask=None):
        """Map normalized attributes back to the original scale.

        With ``bounded=True`` (model predictions), the trait outputs in
        ``BOUNDED_TRAIT_KEYS`` follow ``prediction_bounds.bounded_inverse``: clip
        to the fitted observed range +- ``margin`` SDs, fall back to the observed
        range if the inverse is non-finite, and clip at zero the traits whose
        fitted values were all nonnegative -- the same rule as the ML baselines.
        Leave ``bounded=False`` for observed targets, which must not be altered.
        """
        data = data.clone()
        if not getattr(data, 'normalized', False):
            raise ValueError("Data is not normalized")
        data.normalized = False
        if hasattr(data, 'processed'):
            data.processed = True

        for k_norm in self.normalizations:
            if k_norm not in data.keys():
                if warn:
                    print(f"Warning: {k_norm} not in data")
                continue
            mode = self.normalizations[k_norm]
            if mode == 'sincos':
                lat = torch.atan2(data[k_norm][:, 0], data[k_norm][:, 1]) * 180 / np.pi
                lon = torch.atan2(data[k_norm][:, 2], data[k_norm][:, 3]) * 180 / np.pi
                data[k_norm] = torch.stack([lat, lon], dim=1)
            elif mode in ('yj', 'z', 'logz'):
                tensor = data[k_norm]
                values = tensor.detach().double().cpu().numpy()
                squeeze = values.ndim == 1
                if squeeze:
                    values = values[:, None]
                props = self.props[k_norm]
                if bounded and k_norm in self.BOUNDED_TRAIT_KEYS:
                    if 'observed_min' not in props:
                        raise ValueError(f'{k_norm}: no fitted observed range; refit the normalization.')
                    diagnostics = []
                    restored = bounded_inverse(
                        lambda z, k=k_norm: self._denormalize_values(k, z), values,
                        props['observed_min'].numpy(), props['observed_max'].numpy(),
                        margin=margin, nonnegative=props['nonnegative'].numpy(),
                        diagnostics=diagnostics, evaluation_mask=evaluation_mask,
                    )
                    for row in diagnostics:
                        column = row.pop('column')
                        self.prediction_diagnostics.append({
                            'stage': stage,
                            'trait': trait_names[column] if trait_names is not None else str(column),
                            'variable': k_norm.removeprefix('species_x_'), **row,
                        })
                else:
                    restored = self._denormalize_values(k_norm, values)
                if squeeze:
                    restored = restored[:, 0]
                data[k_norm] = torch.tensor(restored, dtype=tensor.dtype, device=tensor.device)
        return data
    

class NZData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'species_species_edge_index':
            return self.species_num_nodes
        elif key == 'spatial_spatial_edge_index':
            return self.spatial_num_nodes
        elif key == 'spatial_species_edge_index':
            return torch.tensor([[self.spatial_num_nodes], [self.species_num_nodes]])
        return super().__inc__(key, value, *args, **kwargs)

if '__main__' in sys.modules:
    setattr(sys.modules['__main__'], 'NZData', NZData)
# Also add to torch safe globals (weights-only safe loading)
torch.serialization.add_safe_globals([NZData])

class PlantDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
                 trait_representation='min_max_range', traits_filename=None,
                 invalid_bounds_policy='missing', include_unobserved_species=False):
        self.trait_representation = trait_representation
        self.pivot_feature = 'traits_mean' if self.trait_representation == 'mean_std' else 'traits_min'
        self.traits_filename = traits_filename or ('../.soft_del/Traits.xlsx' if trait_representation == 'mean_std' else 'FernTraits.xlsx')
        self.invalid_bounds_policy = invalid_bounds_policy
        self.include_unobserved_species = include_unobserved_species
        trait_path = Path(root) / self.traits_filename
        if not trait_path.exists():
            raise FileNotFoundError(f'Trait workbook not found: {trait_path}')
        self.traits_fingerprint = hashlib.sha256(trait_path.read_bytes()).hexdigest()[:16]
        self.get_traits_df(
            root,
            trait_representation=trait_representation,
            invalid_bounds_policy=invalid_bounds_policy,
        )
        self.trait_names = list(getattr(self, self.pivot_feature).columns)
        from rpy2 import robjects
        project_r_lib = Path(__file__).resolve().parent / '.local' / 'r-library'
        if project_r_lib.is_dir():
            robjects.r['.libPaths'](robjects.StrVector([str(project_r_lib), *robjects.r['.libPaths']()]))
        script = Path(__file__).resolve().with_name('ftol_tree.R')
        robjects.r['source'](str(script))
        result = robjects.globalenv['build_ftol_tree'](robjects.StrVector(list(getattr(self, self.pivot_feature).index)))
        self.tree_newick = str(result.rx2('newick')[0])
        mapping = result.rx2('mapping')
        self.ftol_mapping = pd.DataFrame({name: list(mapping.rx2(name)) for name in mapping.names})
        self.ftol_version = str(result.rx2('ftol_version')[0])
        self.tree_fingerprint = hashlib.sha256(self.tree_newick.encode()).hexdigest()[:16]
        unresolved = self.ftol_mapping[self.ftol_mapping['placement'].isin(['genus absent', 'genus non-monophyletic', 'shared FTOL tip unresolved', 'invalid terminal length'])]
        if not unresolved.empty:
            warnings.warn('FTOL unresolved species: ' + ', '.join(unresolved['species']))
        self.y_index = np.arange(len(self.trait_names))
            
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    def tree_coverage(self) -> pd.DataFrame:
        """FTOL placement and tree membership of every analytical taxon.

        ``in_tree`` is read from the reconstructed Newick tips, so it is the
        authoritative set of taxa that tree-dependent methods can predict.
        Saved with each run so analyses can identify unresolved taxa from the
        run's own provenance rather than a hard-coded list.
        """
        tips = {str(tip.name) for tip in Phylo.read(io.StringIO(self.tree_newick), 'newick').get_terminals()}
        species = list(getattr(self, self.pivot_feature).index)
        mapping = self.ftol_mapping.set_index('species').reindex(species)
        coverage = pd.DataFrame({
            'species': species,
            'ftol_tip': mapping['ftol_tip'].to_numpy(),
            'placement': mapping['placement'].to_numpy(),
            'in_tree': [name in tips for name in species],
        })
        coverage['ftol_version'] = self.ftol_version
        coverage['tree_fingerprint'] = self.tree_fingerprint
        return coverage

    @staticmethod
    def load_raster(f, 
                    grid_step=.1, 
                    max_window_pixels=1_000_000,
                    aggregation='sum'
        ):
        """Load and coarsen a raster.  
        Use ``sum`` for occurrence/count rasters and ``mean`` for continuous
        environmental covariates so that coarsening preserves their scale.
        """
        if f.exists():
            raster = xr.open_dataarray(f, engine="rasterio")
        else:
            # search for subspecies to merge
            raster_list = [xr.open_dataarray(subsp_f, engine="rasterio") for subsp_f in f.parent.glob(f"{f.stem.split('_distribution')[0]}*{f.suffix}")]
            if not raster_list:
                raise FileNotFoundError(f"No distribution raster or matching subspecies rasters found for {f}")
            aligned_rasters = xr.align(*raster_list, join="outer", fill_value=0)  # type: ignore
            raster = xr.concat(aligned_rasters, dim="subspecies").sum("subspecies", skipna=True)

        current_step = abs(float((raster.x[1] - raster.x[0]).values))  # degrees
        downsample = max(1, round(grid_step / current_step))

        # Rasterio-backed xarray data is loaded eagerly by coarsen().sum(). Process
        # row windows instead, keeping the largest intermediate array bounded.
        height = raster.sizes['y']
        width = raster.sizes['x']
        trimmed_height = height - height % downsample
        trimmed_width = width - width % downsample
        if not trimmed_height or not trimmed_width:
            raise ValueError(
                f'Raster {f} is smaller than one {downsample}x{downsample} aggregation cell.'
            )

        rows_per_window = max(
            downsample,
            (max_window_pixels // trimmed_width // downsample) * downsample,
        )
        chunks = []
        for start in range(0, trimmed_height, rows_per_window):
            stop = min(start + rows_per_window, trimmed_height)
            window = raster.isel(y=slice(start, stop), x=slice(0, trimmed_width)).load()
            coarsened = window.coarsen(
                     x=downsample, y=downsample, boundary='trim'
                )
            if aggregation == 'sum':
                chunk = coarsened.sum().fillna(0)  # type: ignore
            else:
                # Keep all-missing environmental cells as NaN rather than
                # introducing an artificial zero in physical units.
                chunk = coarsened.mean(skipna=True)  # type: ignore
            chunks.append(chunk)
            
        return xr.concat(chunks, dim='y')
    
    
    def get_traits_df(self, root, dummy_threshold=0.05, drop_threshold=0.7,
                      trait_representation='min_max_range', invalid_bounds_policy='missing'):
        if trait_representation not in {'mean_std', 'min_max_range'}:
            raise ValueError("trait_representation must be 'mean_std' or 'min_max_range'")
        if invalid_bounds_policy not in {'missing', 'error', 'keep'}:
            raise ValueError("invalid_bounds_policy must be 'missing', 'error', or 'keep'")

        traits_path = root / self.traits_filename
        if not traits_path.exists():
            raise RuntimeError(f'{self.traits_filename} not found at {traits_path}')
        sheets = pd.read_excel(traits_path, sheet_name=None)
        candidates = [
            frame for frame in sheets.values()
            if 'Species' in frame.columns and any(
                re.match(r'^.*?(Mean|Std|Var|Min|Max|Range)$', str(column), flags=re.IGNORECASE)
                for column in frame.columns
            )
        ]
        if not candidates:
            raise RuntimeError(
                f'{self.traits_filename} must contain a sheet with a Species column and numerical trait columns '
                'named Varname{Mean|Std|Var|Min|Max|Range}.'
            )
        seen_categorical = set()
        unique_candidates = []
        for frame in candidates:
            columns_to_keep = ['Species']
            for column in frame.columns:
                if column == 'Species':
                    continue
                is_trait_column = re.match(
                    r'^.*?(Mean|Std|Var|Min|Max|Range)$', str(column), flags=re.IGNORECASE
                )
                if is_trait_column or column not in seen_categorical:
                    columns_to_keep.append(column)
                if not is_trait_column:
                    seen_categorical.add(column)
            unique_candidates.append(frame[columns_to_keep])
        trait_values = pd.concat(
            [frame.set_index('Species') for frame in unique_candidates], axis=1
        )
        trait_columns = {}
        categorical_columns = []
        for column in trait_values.columns:
            match = re.match(r'^(.*?)(Mean|Std|Var|Min|Max|Range)$', str(column), flags=re.IGNORECASE)
            if not match:
                categorical_columns.append(column)
                continue
            trait_name, statistic = match.groups()
            trait_name = trait_name.strip()
            statistic = statistic.lower()
            if not trait_name:
                raise RuntimeError(f'Invalid trait column name: {column}')
            if statistic in trait_columns.setdefault(trait_name, {}):
                raise RuntimeError(f'Duplicate {statistic} column for trait {trait_name}')
            trait_columns[trait_name][statistic] = column

        categorical_columns = list(dict.fromkeys(categorical_columns))
        self.traits_gen = trait_values[categorical_columns]
        if trait_representation == 'mean_std':
            selected_traits = {
                trait_name: columns for trait_name, columns in trait_columns.items()
                if 'mean' in columns
            }
            missing_spread = [
                trait_name for trait_name, columns in selected_traits.items()
                if not {'std', 'var'}.intersection(columns)
            ]
            if not selected_traits or missing_spread:
                raise RuntimeError(
                    f'{self.traits_filename} must contain a sheet with a Species column and numerical trait columns '
                    f'named VarnameMean and either VarnameStd or VarnameVar. '
                    f'Missing spread columns: {missing_spread}'
                )
            self.traits_mean = pd.DataFrame({
                trait_name: trait_values[columns['mean']]
                for trait_name, columns in selected_traits.items()
            }, index=trait_values.index)
            self.traits_std = pd.DataFrame({
                trait_name: trait_values[columns['std']]
                if 'std' in columns else np.sqrt(trait_values[columns['var']])
                for trait_name, columns in selected_traits.items()
            }, index=trait_values.index)
        else:
            selected_traits = {
                trait_name: columns for trait_name, columns in trait_columns.items()
                if 'min' in columns or 'max' in columns or 'range' in columns
            }
            missing_bounds = [
                trait_name for trait_name, columns in selected_traits.items()
                if not {'min', 'max'}.issubset(columns)
            ]
            if not selected_traits or missing_bounds:
                raise RuntimeError(
                    f'{self.traits_filename} must contain a sheet with a Species column and numerical trait columns '
                    f'named VarnameMin and VarnameMax; VarnameRange is optional. '
                    f'Missing bound columns: {missing_bounds}'
                )
            self.traits_min = pd.DataFrame({
                trait_name: trait_values[columns['min']]
                for trait_name, columns in selected_traits.items()
            }, index=trait_values.index)
            self.traits_max = pd.DataFrame({
                trait_name: trait_values[columns['max']]
                for trait_name, columns in selected_traits.items()
            }, index=trait_values.index)
            calculated_range = self.traits_max - self.traits_min
            missing_range = [trait_name for trait_name, columns in selected_traits.items() if 'range' not in columns]
            if missing_range:
                print(f"Warning: Range was calculated from Min/Max for: {missing_range}")
            self.traits_range = pd.DataFrame({
                trait_name: trait_values[columns['range']].fillna(calculated_range[trait_name])
                if 'range' in columns else calculated_range[trait_name]
                for trait_name, columns in selected_traits.items()
            }, index=trait_values.index)
            invalid_bounds = (self.traits_min < 0) | (self.traits_max < self.traits_min) | (self.traits_range < 0)
            if invalid_bounds.any().any():
                invalid_rows, invalid_columns = np.where(invalid_bounds.to_numpy())
                invalid_locations = [
                    f"{invalid_bounds.index[row]}:{invalid_bounds.columns[column]}"
                    for row, column in zip(invalid_rows, invalid_columns, strict=True)
                ]
                message = (
                    f"Found {int(invalid_bounds.to_numpy().sum())} invalid min/max/range records "
                    f"({', '.join(invalid_locations[:5])})."
                )
                if invalid_bounds_policy == 'error':
                    raise ValueError(message + " Correct source values or choose invalid_bounds_policy='missing' or 'keep'.")
                if invalid_bounds_policy == 'missing':
                    print(message + " Treating the complete trait record as missing.")
                    for frame in (self.traits_min, self.traits_max, self.traits_range):
                        frame.mask(invalid_bounds, inplace=True)
                else:
                    print(message + " Keeping values as requested.")
            # self.traits_mean = (self.traits_min + self.traits_max) / 2
            # self.traits_std = self.traits_range / np.sqrt(12.0)

        # remove columns with more than drop_threshold missing values
        for df in [getattr(self, attr) for attr in ['traits_mean', 'traits_std', 'traits_min', 'traits_max', 'traits_range'] if hasattr(self, attr)]:
            df.drop(columns=df.columns[(df.isna().sum() / len(df)) > drop_threshold], inplace=True)

        if self.trait_representation == 'min_max_range':
            incomplete_bounds = (
                self.traits_min.isna()
                | self.traits_max.isna()
                | self.traits_range.isna()
            )
            for frame in (self.traits_min, self.traits_max, self.traits_range):
                frame.mask(incomplete_bounds, inplace=True)

        # Decide eligibility after validation and missing-value harmonization;
        # retain the source workbook and names for explicit recovery.
        observed = getattr(self, self.pivot_feature).notna().any(axis=1)
        self.unobserved_species = observed.index[~observed].tolist()
        if not self.include_unobserved_species:
            for attr in ('traits_mean', 'traits_std', 'traits_min', 'traits_max', 'traits_range', 'traits_gen'):
                if hasattr(self, attr):
                    setattr(self, attr, getattr(self, attr).loc[observed].copy())

        for cl_feat in categorical_columns:
            for cl in self.traits_gen[cl_feat].unique(): # type: ignore
                if self.traits_gen[cl_feat].eq(cl).sum() < len(self.traits_gen) * dummy_threshold and not pd.isna(cl):
                    self.traits_gen[cl_feat] = self.traits_gen[cl_feat].replace({cl: 'Other'})
        self.traits_gen_dummy = pd.get_dummies(self.traits_gen, drop_first=True)

        if self.trait_representation == 'mean_std':
            for col in self.traits_mean.columns.difference(self.traits_std.columns):
                self.traits_std[col] = np.nan
            self.traits_std = self.traits_std[self.traits_mean.columns]
            return (self.traits_mean, self.traits_std), self.traits_gen_dummy
        else:
            return (self.traits_min, self.traits_max, self.traits_range), self.traits_gen_dummy

    @property
    def raw_dir(self):
        return Path(self.root) / 'Distribution layers'
    
    def get_raw_file_names(self):
        dist_files = [f'{f}_distribution.tif' for f in getattr(self, self.pivot_feature).index]
        return dist_files

    @property
    def raw_file_names(self):
        return self.get_raw_file_names

    @property
    def raw_paths(self):
        return [str(self.raw_dir / filename) for filename in self.get_raw_file_names()]

    def get_processed_file_names(self):
        coverage = 'all' if self.include_unobserved_species else 'observed'
        return [f'heterodata_{self.trait_representation}_{self.invalid_bounds_policy}_{coverage}_{self.traits_fingerprint}_{self.tree_fingerprint}_v{PROCESSED_DATA_VERSION}.pt']

    @property
    def processed_file_names(self):
        return self.get_processed_file_names

    @property
    def processed_paths(self):
        return [str(Path(self.processed_dir) / filename) for filename in self.get_processed_file_names()]

    def _download(self):
        # A missing distribution raster means that a species has no spatial
        # connections; it must not remove the species from the dataset.
        return

    def load_complete(self, data_path):
        comp_rasters = []
        df_path = Path(self.root) / "Complete layers" / f"{data_path}_space_df_v{FEATURE_CACHE_VERSION}.csv"
        if df_path.exists():
            return pd.read_csv(df_path, index_col=0)
        
        for f in tqdm(sorted((Path(self.root) / "Complete layers" / data_path).rglob('*.tif')), desc=f'Loading {data_path}'):
            raster = self.load_raster(f, aggregation='mean')
            interpolated_raster = raster.interp(x=self.space_df.x.values, y=self.space_df.y.values, method='nearest')
            for band in interpolated_raster.band.values:
                band_raster = interpolated_raster.sel(band=band).to_pandas().reset_index().melt(id_vars=['y'])
                feat_name = f.stem if band == 0 else f"{f.stem}_{band}"
                band_raster = band_raster.rename(columns={'value': feat_name})
                comp_rasters.append(self.space_df.set_index(['x', 'y']).join(band_raster.set_index(['x', 'y'])).groupby(['cluster']).mean()[feat_name])
                if f.stem == 'NZ population density layer':
                    comp_rasters[-1] = np.log1p(comp_rasters[-1])
        df = pd.concat(comp_rasters, axis=1)
        df.to_csv(df_path)
        return df

    def get_species_graph(self):
        tree = Phylo.read(io.StringIO(self.tree_newick), "newick")

        G = nx.Graph()
        for clade in tree.get_nonterminals():  # Internal nodes (non-leaf)
            if clade.name is None:
                clade.name = f"temp_{id(clade)}"
            for child in clade.clades:
                if child.name is None:
                    child.name = f"temp_{id(child)}"
                G.add_edge(clade.name, child.name, weight=child.branch_length)
        
        node2vec = Node2Vec(G, dimensions=16, walk_length=30, num_walks=200, workers=1, seed=42)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Create node embeddings
        node_embeddings = {node: model.wv[node] for node in G.nodes()}

        leaf_nodes = [leaf.name for leaf in tree.get_terminals()]

        G_species = nx.Graph()
        for species1, species2 in itertools.combinations(leaf_nodes, 2):
            dist = tree.distance(species1, species2)  # Compute phylogenetic distance
            G_species.add_edge(species1, species2, weight=dist)

        median = np.median([d["weight"] for u, v, d in G_species.edges(data=True)])
        print(f"Median distance: {median}\n")
        sym_threshold = median

        # Prune the graph by removing edges with weight > sym_threshold
        edges_to_remove = [(u, v) for u, v, d in G_species.edges(data=True) if d["weight"] > sym_threshold]
        G_species.remove_edges_from(edges_to_remove)

        G_species.remove_nodes_from(list(set(G_species.nodes).difference(getattr(self, self.pivot_feature).index)))

        for node in G_species.nodes():
            G_species.nodes[node]["x"] = node_embeddings[node]
        for source, target, attributes in G_species.edges(data=True):
            attributes['weight'] = float(attributes['weight'])
        self.species_graph = from_networkx(G_species)  # Include node attributes
        self.species_graph.node_names = [n for n in G_species.nodes()]  # Map node names to indices

        # Add nodes not present in the ph_tree
        for node in getattr(self, self.pivot_feature).index.difference(self.species_graph.node_names):
            self.species_graph.node_names.append(node)
            self.species_graph.x = torch.cat([self.species_graph.x, torch.zeros(1, self.species_graph.num_features)], dim=0)

        self.species_graph.traits_nanmask = torch.tensor(getattr(self, self.pivot_feature).loc[self.species_graph.node_names].isna().values, dtype=torch.bool)
        for attr in ['traits_mean', 'traits_std', 'traits_min', 'traits_max', 'traits_range']:
            if hasattr(self, attr):
                other_nanmask = torch.tensor(getattr(self, attr).loc[self.species_graph.node_names].isna().values, dtype=torch.bool)
                assert torch.all((~other_nanmask) <= (~self.species_graph.traits_nanmask)), f"Non-nan values in {attr} should not be nan in {self.pivot_feature}"
                setattr(self, attr, getattr(self, attr).loc[self.species_graph.node_names])
                setattr(self.species_graph, attr, torch.tensor(getattr(self, attr).fillna(0).astype(np.float32).values))

        setattr(self, self.pivot_feature, getattr(self, self.pivot_feature).loc[self.species_graph.node_names])
        setattr(self.species_graph, self.pivot_feature, torch.tensor(getattr(self, self.pivot_feature).fillna(0).astype(np.float32).values))
        
        self.traits_gen = self.traits_gen.loc[self.species_graph.node_names]
        self.traits_gen_dummy = self.traits_gen_dummy.loc[self.species_graph.node_names]
        self.species_graph.x_gen = torch.tensor(self.traits_gen_dummy.astype(np.float32).values)

        return self.species_graph

    def get_spatial_graph(self, n_clusters=50, k=6):
        all_occurrences = self.load_raster(Path(self.root)/"Distribution layers/_all_species_distributions.tif", aggregation='sum')
        self.space_df = all_occurrences.isel(band=0).to_pandas().reset_index().melt(id_vars=['y']).rename(columns={'value': 'occurrence'})
        self.space_df.x = self.space_df.x.astype(float)
        self.space_df.y = self.space_df.y.astype(float)
        self.space_df = self.space_df[self.space_df.occurrence > 0].drop(columns='occurrence').reset_index(drop=True)
        
        self.space_clustering = SpectralClustering(n_clusters=n_clusters, random_state=0).fit(self.space_df[['x', 'y']])
        self.space_df['cluster'] = self.space_clustering.labels_
        self.spatial_graph = Data(pos=torch.tensor(self.space_df.groupby('cluster').agg({'x': 'mean', 'y': 'mean'}).values).float())
        
        self.spatial_graph = KNNGraph(k=k)(self.spatial_graph)
        self.spatial_graph.edge_attr = torch.norm(self.spatial_graph.pos[self.spatial_graph.edge_index[0]] - self.spatial_graph.pos[self.spatial_graph.edge_index[1]], dim=1).unsqueeze(1)
        return self.spatial_graph

    def get_distribution(self, dist_path):
        ar = self.load_raster(dist_path, aggregation='sum')

        ar = ar.where(ar > 0, 0)
        ar = ar.where(ar.isnull(), 1)

    def process(self):
        species_graph = self.get_species_graph()
        spatial_graph = self.get_spatial_graph()
        clim_rasters = self.load_complete('Climatic layers')
        population_elevation_rasters = self.load_complete('population density and elevation layer')
        soil_rasters = self.load_complete('Soil NZ layers')

        self.global_data = pd.concat([clim_rasters, population_elevation_rasters, soil_rasters], axis=1)
        coverage = 'all' if self.include_unobserved_species else 'observed'
        index_space_specie_path = Path(self.root) / f'index_space_specie_{coverage}_{self.traits_fingerprint}_v{PROCESSED_DATA_VERSION}.csv'
        if index_space_specie_path.exists():
            index_space_specie = pd.read_csv(index_space_specie_path)
            required_columns = {'cluster', 'species', 'occurrence'}
            if not required_columns.issubset(index_space_specie.columns):
                raise ValueError(f'Invalid species-occurrence cache: {index_space_specie_path}')
            index_space_specie = index_space_specie[['cluster', 'species', 'occurrence']].copy()
        else:
            index_space_specie = pd.DataFrame(columns=['cluster', 'species', 'occurrence'])
            coord_tree = cKDTree(self.space_df[['x', 'y']].to_numpy(dtype=float))
            grouped_occurrences = []
            for species_name in tqdm(species_graph.node_names, desc='Processing species distribution layers'):
                f = self.raw_dir / f'{species_name}_distribution.tif'
                try:
                    raster = self.load_raster(f, aggregation='sum')
                except FileNotFoundError:
                    warnings.warn(f'No distribution raster for {species_name}; retaining it without location edges.')
                    continue
                occurrences = raster.isel(band=0).to_pandas().reset_index().melt(id_vars=['y']).rename(columns={'value': 'occurrence'})
                occurrences = occurrences.loc[occurrences.occurrence > 0].copy()
                if occurrences.empty:
                    continue
                _, indices = coord_tree.query(occurrences[['x', 'y']].to_numpy(dtype=float))
                occurrences['cluster'] = self.space_df.iloc[indices]['cluster'].to_numpy()
                grouped = occurrences.groupby('cluster', as_index=False)['occurrence'].sum()
                grouped['species'] = species_name
                grouped_occurrences.append(grouped)
            if grouped_occurrences:
                index_space_specie = pd.concat(grouped_occurrences, ignore_index=True)

        species_to_index = {species: index for index, species in enumerate(species_graph.node_names)}
        index_space_specie['species_idx'] = index_space_specie.species.map(species_to_index)
        if index_space_specie['species_idx'].isna().any():
            unknown_species = index_space_specie.loc[index_space_specie['species_idx'].isna(), 'species'].unique().tolist()
            raise RuntimeError(f'Distribution index contains species absent from the graph: {unknown_species[:5]}')
        index_space_specie['species_idx'] = index_space_specie['species_idx'].astype(int)
        index_space_specie.to_csv(index_space_specie_path, index=False)
            
        bip_edge_index = torch.tensor(index_space_specie[['cluster', 'species_idx']].values.T, dtype=torch.long)
        bip_edge_attr = torch.tensor(index_space_specie.occurrence.values, dtype=torch.float32).unsqueeze(1)

        if self.trait_representation == 'min_max_range':
            data_all = NZData(
                species_x_min=species_graph.traits_min,
                species_x_max=species_graph.traits_max,
                species_x_range=species_graph.traits_range
            )
        else:
            data_all = NZData(
                species_x_mean=species_graph.traits_mean,
                species_x_std=species_graph.traits_std,
        )
        data_all.species_x_gen = species_graph.x_gen
        data_all.genetic_feature_names = self.traits_gen_dummy.columns.astype(str).tolist()
        data_all.species_names = species_graph.node_names
        data_all.species_x_phylo = species_graph.x
        data_all.traits_nanmask = species_graph.traits_nanmask
        data_all.spatial_x = spatial_graph.pos
        data_all.spatial_pos = spatial_graph.pos # repeated to stay un-normalized
        data_all.spatial_global_data = torch.tensor(self.global_data.values, dtype=torch.float32)
        data_all.spatial_global_feature_names = self.global_data.columns.astype(str).tolist()
        data_all.species_species_edge_index = species_graph.edge_index
        data_all.species_species_edge_attr = species_graph.weight
        data_all.spatial_spatial_edge_index = spatial_graph.edge_index
        data_all.spatial_spatial_edge_attr = spatial_graph.edge_attr
        data_all.spatial_species_edge_index = bip_edge_index
        data_all.spatial_species_edge_attr = bip_edge_attr
        data_all.species_num_nodes = species_graph.num_nodes
        data_all.spatial_num_nodes = spatial_graph.num_nodes
        data_all.num_nodes = species_graph.num_nodes + spatial_graph.num_nodes
        data_all.features_names = getattr(self, self.pivot_feature).columns.tolist()

        self.save([data_all], self.processed_paths[0])
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # data.y = data.x_species[:, self.y_index]*~data.x_species_traits_nanmask
        return data
        

def stable_species_folds(species_names, seed):
    """Return repeatable fold IDs unaffected by ordering or other species."""
    if len(species_names) != len(set(species_names)):
        raise ValueError('Species names must be unique for stable cross-validation.')
    return np.array([
        int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'big') % 5
        for name in species_names
    ])


def data_split(data, test_size=0.3, k=0, seed=42, split_strategy='random'):
    n_species = data.species_num_nodes
    if split_strategy == 'random':
        splitter = KFold(n_splits=5, random_state=seed, shuffle=True)
        splits = list(splitter.split(np.arange(n_species)))
    elif split_strategy == 'stable':
        # A species keeps the same fold when other species are added/removed.
        # This makes the overlap between two taxon sets directly comparable.
        fold_ids = stable_species_folds(data.species_names, seed)
        if any((fold_ids == i).sum() == 0 for i in range(5)):
            raise ValueError('Stable split produced an empty test fold.')
        splits = [(np.flatnonzero(fold_ids != i), np.flatnonzero(fold_ids == i)) for i in range(5)]
    elif split_strategy in ('genus', 'clade'):
        if split_strategy == 'genus':
            groups = [name.split('_')[0] for name in data.species_names]
        else:
            tree = Phylo.read(io.StringIO(data.tree_newick), 'newick')
            names = set(data.species_names)
            clades = [tree.root]
            # Recursively split the largest available clade until at least five
            # nonempty blocks exist; each block remains monophyletic.
            while True:
                candidates = [(len({t.name for t in c.get_terminals()} & names), i)
                              for i, c in enumerate(clades) if c.clades]
                if not candidates:
                    break
                size, i = max(candidates)
                if len(clades) >= 5 and size <= int(np.ceil(n_species / 5)):
                    break
                node = clades.pop(i)
                clades.extend(c for c in node.clades if {t.name for t in c.get_terminals()} & names)
            tip_group = {tip.name: f'clade_{i}' for i, c in enumerate(clades) for tip in c.get_terminals()}
            groups = [tip_group.get(name, f'unresolved_{name}') for name in data.species_names]
        if len(set(groups)) < 5:
            raise ValueError('At least five distinct groups are required for grouped cross-validation.')
        splits = list(GroupKFold(n_splits=5).split(np.arange(n_species), groups=groups))
        if any(len(test) > len(train) for train, test in splits):
            raise ValueError('A group is too large for a valid five-fold split.')
    elif split_strategy == 'louvain':
        graph = nx.Graph()
        graph.add_nodes_from(range(n_species))
        graph.add_edges_from(data.species_species_edge_index.T.numpy())
        communities = list(nx.algorithms.community.louvain_communities(graph, seed=seed))
        groups = np.empty(n_species, dtype=int)
        for group_index, community in enumerate(communities):
            groups[list(community)] = group_index

        largest_community = max(len(community) for community in communities)
        target_fold_size = int(np.ceil(n_species / 5))
        if largest_community > target_fold_size:
            raise ValueError(
                f'Louvain split cannot form balanced 5-fold CV: largest community has {largest_community} species, '
                f'but a fold targets about {target_fold_size}. Use random transductive CV or sparsify the phylogenetic graph first.'
            )
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(np.arange(n_species), groups=groups))
    else:
        raise ValueError("Unknown split strategy.")

    if not 0 <= k < len(splits):
        raise ValueError(f'Fold index {k} is outside [0, {len(splits) - 1}].')
    train_nodes, test_nodes = splits[k]

    train_mask = torch.zeros(data.species_num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.species_num_nodes, dtype=torch.bool)

    train_mask[train_nodes] = True
    test_mask[test_nodes] = True

    if train_mask.sum() < test_mask.sum():
        # print(f'Warning: Swapped train and test masks ({train_mask.sum()} < {test_mask.sum()})')
        train_mask, test_mask = test_mask, train_mask
    data.train_mask = train_mask
    data.test_mask = test_mask

    train_data = data.clone()
    test_data = data.clone()
    for attr in ['x_mean', 'x_std', 'x_min', 'x_max', 'x_range', 'x_gen', 'x_phylo']:
        if f'species_{attr}' not in data:
            continue
        train_data[f'species_{attr}'] = train_data[f'species_{attr}'][train_mask]
        test_data[f'species_{attr}'] = test_data[f'species_{attr}'][test_mask]
    train_data.traits_nanmask = train_data.traits_nanmask[train_mask]
    test_data.traits_nanmask = test_data.traits_nanmask[test_mask]
    train_data.species_names = [name for name, mask in zip(data.species_names, train_mask) if mask]
    test_data.species_names = [name for name, mask in zip(data.species_names, test_mask) if mask]
    train_data.species_num_nodes = train_mask.sum().item()
    test_data.species_num_nodes = test_mask.sum().item()
    train_data.num_nodes = train_data.species_num_nodes + train_data.spatial_num_nodes
    test_data.num_nodes = test_data.species_num_nodes + test_data.spatial_num_nodes
    

    for data_split, mask in zip([train_data, test_data], [train_mask, test_mask]):
        data_split.species_species_edge_index, data_split.species_species_edge_attr = subgraph(
            mask, edge_index=data.species_species_edge_index,
            edge_attr=data.species_species_edge_attr,
            relabel_nodes=True
            )
        data_split.spatial_species_edge_index, data_split.spatial_species_edge_attr = bipartite_subgraph( # type: ignore
            (torch.ones(data.spatial_num_nodes, dtype=torch.bool), mask), 
            edge_index=data.spatial_species_edge_index, edge_attr=data.spatial_species_edge_attr, relabel_nodes=True
            )
    train_data = cast(Data, train_data)
    test_data = cast(Data, test_data)
    return train_data, test_data

if __name__ == '__main__':
    norm_transform = NormalizeFeatures()
    data = PlantDataset(Path('data/Ferns'))[0]
    normed_data = norm_transform(data)
    re_unnormed_data = norm_transform.inverse(normed_data)

    for k in norm_transform.normalizations:
        if not torch.allclose(data[k], re_unnormed_data[k]): # type: ignore
            print(f"Normalization failed for {k} (max diff: {torch.max(torch.abs(data[k] - re_unnormed_data[k]))})")
        # n, k = k.split('/')
        # if len(n.split('-')) > 1:
        #     n = n.split('-')
        # if not torch.allclose(data[n][k], re_unnormed_data[n][k]):
        #     print(f"Normalization failed for {n}: {k} (max diff: {torch.max(torch.abs(data[n][k] - re_unnormed_data[n][k]))})")
