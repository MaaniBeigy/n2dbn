"""
model/recency.py
================
Two summaries of the past that stay identified at every interior scale.

The lag axis is the largest source of competing representations of one relation.
A source offering twenty lags offers twenty columns of nearly the same
information, and the search pays for each of them.  Two prefix-measurable
summaries collapse that axis into one column apiece.

`recency` is how long it has been since the label last happened, and
`geometric_lag` is an exponentially weighted total of everything before now.
Both read the strict past alone, so an edge carrying one preserves acyclicity
the same way a lagged read does.

Neither is constant across the children of a parent epoch.  Recency counts up
between events and resets at one, and the geometric weight decays inside any
block long enough to hold two epochs, so both keep a within-block contrast and
stay identified at the scale that carries them.  That is what makes either one a
carrier for a regime effect below the coarsest scale, where an additive regime
level is block constant on every block the path does not switch inside and
contributes nothing.
"""

from __future__ import annotations

import numpy as np


def recency(events: np.ndarray) -> np.ndarray:
    """
    Epochs since the label last happened, read at every epoch of one scale.

    An epoch before the first event carries its distance to the epoch before the
    record, so the column is finite everywhere and rises until the first event.

    Args:
        events: (T,) counts or durations, where a positive entry is an event.

    Raises:
        ValueError: when the input is not one column.
    """
    values = np.asarray(events, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"a recency column reads one series; got {values.shape}")
    out = np.empty(values.size, dtype=float)
    last = -1
    for t in range(values.size):
        out[t] = t - last
        if values[t] > 0.0:
            last = t
    return out


def recency_column(events: np.ndarray) -> np.ndarray:
    """The recency on the log scale, which is the column an edge reads."""
    return np.log1p(recency(events))


def geometric_lag(values: np.ndarray, rho: float) -> np.ndarray:
    """
    An exponentially weighted total of the strict past, `(1-rho) * sum rho^(d-1) z_(t-d)`.

    Computed by the recursion `M_t = rho * M_(t-1) + (1 - rho) * z_(t-1)`, which
    is the same series in one pass and never forms the sum.

    Args:
        rho: the decay, in [0, 1); zero reads the previous epoch alone.

    Raises:
        ValueError: when the input is not one column, or the decay is outside
            the unit interval.
    """
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"a geometric lag reads one series; got {series.shape}")
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"a geometric decay is in [0, 1); got {rho}")
    out = np.zeros(series.size, dtype=float)
    for t in range(1, series.size):
        out[t] = rho * out[t - 1] + (1.0 - rho) * series[t - 1]
    return out


def regime_interaction(carrier: np.ndarray, responsibilities: np.ndarray) -> np.ndarray:
    """
    One column per regime, each the carrier weighted by that regime's posterior.

    A regime enters a finer scale through a feature that varies inside a parent
    block, so the columns a regime contributes are the carrier times each
    regime's responsibility.  A regime whose path does not switch inside a block
    then still contributes nothing there.  That is the invariance acting; no
    modelling choice supplies it.

    Args:
        carrier: (T,) a feature carrying within-block variation.
        responsibilities: (T, M) the posterior over regimes at each epoch.

    Raises:
        ValueError: when the two do not carry the same number of epochs.
    """
    carrier = np.asarray(carrier, dtype=float)
    gamma = np.asarray(responsibilities, dtype=float)
    if gamma.ndim != 2 or gamma.shape[0] != carrier.shape[0]:
        raise ValueError(
            f"a carrier of {carrier.shape[0]} epochs meets a posterior of "
            f"{gamma.shape}, which holds a different number of them"
        )
    return carrier[:, None] * gamma
