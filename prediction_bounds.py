"""Shared original-scale postprocessing for GNN and ML baseline predictions.

Both model families predict trait targets in a fold-specific transformed space
(Yeo-Johnson + standardization, fitted on outer-training species only) and must
map them back to the original measurement scale under the same rule:

1. clip the transformed prediction to the observed training range extended by
   ``margin`` transformed standard deviations;
2. invert the transform;
3. where the inverse is still non-finite (the Yeo-Johnson inverse is undefined
   beyond ``1/|lambda|`` for lambda < 0, or the analogous limit for lambda > 2),
   fall back to clipping at the observed training range itself, which is always
   inside the inverse domain;
4. clip at zero the columns whose training values were all nonnegative.

Inputs that are NaN (cells a model does not predict) stay NaN.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_EXTRAPOLATION_MARGIN = 1.0
CORRECTION_COUNTS = (
    'n_predictions', 'n_missing', 'n_clipped_low', 'n_clipped_high',
    'n_inverse_fallback', 'n_nonnegative_clipped', 'n_corrected',
)


def write_safeguard_log(records: list[dict], path: Path) -> None:
    """Write per-output counts; evaluated counts use the scoring/calibration mask.

    Counts describe operations before min/max projection (logged separately).
    A cell may contribute to more than one correction count.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['fold', 'stage', 'trait', 'variable', *CORRECTION_COUNTS,
              *(name.replace('n_', 'n_eval_', 1) for name in CORRECTION_COUNTS)]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({'fold': path.parent.name, **row} for row in records)


def bounded_inverse(inverse: Callable[[np.ndarray], np.ndarray], values_t: np.ndarray,
                    observed_min: np.ndarray, observed_max: np.ndarray, *,
                    margin: float = DEFAULT_EXTRAPOLATION_MARGIN,
                    nonnegative: np.ndarray | bool = False,
                    diagnostics: list[dict] | None = None,
                    evaluation_mask: np.ndarray | None = None) -> np.ndarray:
    """Invert column-wise transformed predictions with bounded extrapolation.

    ``values_t`` has shape (n, k); ``observed_min``/``observed_max`` hold the
    per-column training range on the same transformed scale; ``inverse`` maps a
    (n, k) transformed array to the original scale.
    """
    values_t = np.asarray(values_t, dtype=np.float64)
    if values_t.ndim != 2:
        raise ValueError(f'expected a 2-D array of transformed predictions, got shape {values_t.shape}')
    if np.isinf(values_t).any():
        raise FloatingPointError('raw predictions contain infinity; refusing to clip invalid model outputs')
    low = np.asarray(observed_min, dtype=np.float64)
    high = np.asarray(observed_max, dtype=np.float64)
    if low.shape != (values_t.shape[1],) or high.shape != low.shape:
        raise ValueError(f'training range shapes {low.shape}/{high.shape} do not match {values_t.shape[1]} columns')
    if not (np.isfinite(low).all() and np.isfinite(high).all() and (low <= high).all()):
        raise ValueError('training range must be finite with min <= max')
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('margin must be finite and nonnegative')
    selected = np.ones(values_t.shape, dtype=bool) if evaluation_mask is None else np.asarray(evaluation_mask, dtype=bool)
    if selected.shape != values_t.shape:
        raise ValueError('evaluation mask must match the prediction shape')
    if evaluation_mask is not None and (selected & np.isnan(values_t)).any():
        raise FloatingPointError('raw predictions are NaN on evaluated cells')

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        pred = np.asarray(inverse(np.clip(values_t, low - margin, high + margin)), dtype=np.float64)
        if pred.shape != values_t.shape:
            raise ValueError('inverse must preserve the prediction shape')
        invalid = ~np.isfinite(pred) & np.isfinite(values_t)
        if invalid.any():
            fallback = np.asarray(inverse(np.clip(values_t, low, high)), dtype=np.float64)
            pred = np.where(invalid, fallback, pred)
    if (~np.isfinite(pred) & np.isfinite(values_t)).any():
        raise FloatingPointError('inverse transform is non-finite even inside the observed training range')

    nonnegative = np.broadcast_to(np.asarray(nonnegative, dtype=bool), (values_t.shape[1],))
    negative = nonnegative[None, :] & (pred < 0)
    if diagnostics is not None:
        masks = dict(
            n_predictions=np.isfinite(values_t), n_missing=np.isnan(values_t),
            n_clipped_low=values_t < low - margin,
            n_clipped_high=values_t > high + margin,
            n_inverse_fallback=invalid, n_nonnegative_clipped=negative,
        )
        masks['n_corrected'] = (masks['n_clipped_low'] | masks['n_clipped_high'] |
                                invalid | negative)
        for column in range(values_t.shape[1]):
            row = {'column': column}
            for name, mask in masks.items():
                row[name] = int(mask[:, column].sum())
                row[name.replace('n_', 'n_eval_', 1)] = int((mask & selected)[:, column].sum())
            diagnostics.append(row)
    pred = np.where(nonnegative[None, :], np.maximum(pred, 0.0), pred)
    return np.where(np.isnan(values_t), np.nan, pred)
