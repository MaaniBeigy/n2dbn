"""
evaluation/forecast_metrics.py
==============================
The benchmark's own point-forecast errors, and the coherence column beside them.

Which scores appear, and why
----------------------------
The benchmark reports mean absolute error as its primary score and mean squared
error as its secondary one, over rolling origins, and nothing probabilistic.
Those two are what a comparison against its published table has to carry.

Scale
-----
The benchmark's published errors sit between 0.15 and 1.1 while its series run
to hundreds of thousands of counts per epoch, so its errors are computed after
standardizing each series.  The paper does not state the convention, so
`standardize` fixes one here: the mean and the standard deviation come from the
training window alone and are applied to the held-out window unchanged, which is
what keeps an error scale-free without reading the future.

`coherence_error` is the column no baseline of that benchmark carries.  It reads
a forecast at every scale of the ladder and asks whether the children of a block
sum to the parent.  A model that forecasts one resolution has nothing to put in
it, and a model whose coarse scales are sums of its fine scale puts a zero
there by construction.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from ..core.scales import ScaleHierarchy

#: A channel that never moves in the training window has no scale to divide by.
_SD_FLOOR = 1e-8


def train_scaler(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel mean and standard deviation of the training window.

    Args:
        train: `(T, L)` observed values before the origin.

    Returns:
        The `(L,)` mean and the `(L,)` standard deviation, floored so a flat
        channel does not divide by zero.
    """
    train = np.asarray(train, dtype=float)
    return train.mean(axis=0), np.maximum(train.std(axis=0), _SD_FLOOR)


def standardize(values: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    Values on the training window's scale.

    Args:
        values: `(h, L)` array to rescale.
        mean: `(L,)` training mean.
        sd: `(L,)` training standard deviation.

    Returns:
        An `(h, L)` array.
    """
    return (np.asarray(values, dtype=float) - mean[None, :]) / sd[None, :]


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean absolute error over every epoch and channel of one window.

    Args:
        actual: `(h, L)` held-out values.
        predicted: `(h, L)` forecasts.

    Returns:
        The mean.
    """
    return float(
        np.mean(np.abs(np.asarray(actual, float) - np.asarray(predicted, float)))
    )


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean squared error over every epoch and channel of one window.

    Args:
        actual: `(h, L)` held-out values.
        predicted: `(h, L)` forecasts.

    Returns:
        The mean.
    """
    return float(
        np.mean((np.asarray(actual, float) - np.asarray(predicted, float)) ** 2)
    )


def coherence_error(
    forecasts: Sequence[np.ndarray], hierarchy: ScaleHierarchy
) -> Dict[str, float]:
    """
    How far a forecast at every scale is from the sum of the scale below it.

    Args:
        forecasts: one `(H_k, L)` forecast per scale, finest first.
        hierarchy: the ladder the forecasts sit on.

    Returns:
        `coh_k{k}` for every adjacent pair and `coh` for their mean, in counts
        per coarse epoch and channel.
    """
    out: Dict[str, float] = {}
    per_pair = []
    for k in range(1, hierarchy.K):
        parent = np.asarray(hierarchy.parent_maps[k - 1])
        child = np.atleast_2d(np.asarray(forecasts[k - 1], dtype=float))
        above = np.atleast_2d(np.asarray(forecasts[k], dtype=float))
        rolled = np.zeros_like(above)
        np.add.at(rolled, parent, child)
        gap = float(np.mean(np.abs(rolled - above)))
        out[f"coh_k{k}"] = gap
        per_pair.append(gap)
    out["coh"] = float(np.mean(per_pair)) if per_pair else 0.0
    return out


def relative_coherence(
    forecasts: Sequence[np.ndarray], hierarchy: ScaleHierarchy
) -> float:
    """
    The coherence gap divided by the size of the quantity it is a gap in.

    Counts here run to hundreds of thousands per epoch, so an absolute gap is
    unreadable across series and this is what a table can compare.

    Args:
        forecasts: one `(H_k, L)` forecast per scale, finest first.
        hierarchy: the ladder the forecasts sit on.

    Returns:
        The mean absolute gap over the mean absolute coarsest forecast.
    """
    gap = coherence_error(forecasts, hierarchy)["coh"]
    scale = float(np.mean(np.abs(np.asarray(forecasts[-1], dtype=float))))
    return gap / max(scale, _SD_FLOOR)
