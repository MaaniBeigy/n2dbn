"""
evaluation/persistence.py
=========================
Row A11 of the nonstationarity grid, which asks where a series' persistence moves.

A level segmentation reads the mean of a series and a dispersion segmentation
reads its spread, and both hold the ordering inside a segment free.  Persistence
is what that ordering carries: how much of one epoch survives into the next.  A
series whose autocorrelation halves over the record is nonstationary in a way
the three changepoint rows leave untouched, so A11 reads the second-order
structure inside a sliding window and asks whether it holds across the record.

Two readings of that structure run here.  The autocorrelation trajectory is the
lag-1 to lag-L autocorrelation of each window, which states persistence in the
time domain.  The local periodogram is the same window's power on a declared
frequency grid, which states it in the frequency domain; each window is
detrended before its power is taken, which keeps a ramp inside a window out of
the low frequencies.

A change in the spectral shape is read by a CUSUM.  Each window's periodogram is
normalized to a density, so the total power of a window leaves the statistic and
only the shape it puts across frequencies stays.  A window whose detrended
residual sits at the rounding floor of its own magnitude carries no power at all,
which keeps float error out of that density.  The cumulative deviation from
the average density walks away from zero where the shape changes and closes back
at the last window, the window holding the largest absolute deviation is the
estimated change, and a reshuffle of the window order prices it under a null
that assumes nothing about the density.

A caller gets one scalar per participant per series from `persistence_drift`,
the standard deviation of the lag-1 autocorrelation across windows, which is the
magnitude the row carries.  The trajectory, the local periodogram, the CUSUM and
the trajectory RMSE are exported beside it, since the record reports where the
change falls and how sharp it is as well as how large the drift is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import signal

#: A window spans two calendar periods, which is the shortest span carrying a
#: repeat of the cycle inside one window.
WINDOW_PERIODS = 2.0

#: A record holding one spectrum leaves rounding error in the cumulative sum,
#: and a statistic below this floor is read as no change at all.
_FLOOR = 1e-12

#: Detrending a window that is flat, or that is exactly a straight line, leaves a
#: residual at the rounding floor of the window's own magnitude.  A residual
#: below this many multiples of that magnitude is read as no signal at all.
#: Measured on this grid, a straight line lands at 3e-16 of its magnitude and the
#: faintest real rhythm lands at 2e-11, so the floor sits between the two.
_DUST = 64.0 * float(np.finfo(float).eps)


def window_plan(period: float) -> Tuple[int, int, int]:
    """
    The window width, the step and the largest lag one calendar period declares.

    Windows open one calendar period apart, so consecutive windows overlap by
    half and every window opens at the same phase of the cycle.

    Args:
        period: the calendar period of the scale, in epochs of that scale.

    Returns:
        The width in epochs, the step in epochs, and the largest lag.

    Raises:
        ValueError: when the period is not a positive finite number of epochs.
    """
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError(f"a calendar period is positive and finite; got {period}")
    step = max(1, int(round(period)))
    width = max(2, int(round(WINDOW_PERIODS * period)))
    return width, step, min(step, width - 1)


def _finite(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    The values as a float array, refusing one that is not defined.

    Raises:
        ValueError: when the array holds a NaN or an infinity, naming the first
            such value.
    """
    x = np.asarray(values, dtype=float)
    defined = np.isfinite(x)
    if not bool(defined.all()):
        raise ValueError(
            f"{name} holds finite values only; got {x[~defined].ravel()[0]}"
        )
    return x


def _window_starts(n: int, window: int, step: int) -> np.ndarray:
    """
    The opening epoch of every window that fits, empty below two windows.

    A trajectory is read between windows, so a series holding room for one
    window carries no trajectory and comes back empty.

    Raises:
        ValueError: when the width or the step cannot declare a window.
    """
    if window < 2:
        raise ValueError(f"a window holds at least two epochs; got {window}")
    if step < 1:
        raise ValueError(f"a window step is at least one epoch; got {step}")
    starts = np.arange(0, max(n - window + 1, 0), step, dtype=np.int64)
    return starts if starts.size >= 2 else np.zeros(0, dtype=np.int64)


def _autocorrelation(block: np.ndarray, max_lag: int) -> np.ndarray:
    """The lag-1 to lag-L autocorrelation of one window, zeros where it is flat."""
    centred = block - block.mean()
    total = float(centred @ centred)
    # A flat window has no scale to divide by, and its persistence is undefined.
    if total <= 0.0:
        return np.zeros(max_lag, dtype=float)
    return np.array(
        [float(centred[lag:] @ centred[:-lag]) / total for lag in range(1, max_lag + 1)]
    )


def rolling_autocorrelation(
    values: Sequence[float], window: int, step: int, max_lag: int
) -> np.ndarray:
    """
    The lag-1 to lag-L autocorrelation of every sliding window of a series.

    Args:
        values: the series, at one scale and inside one participant.
        window: the width of a window, in epochs.
        step: how far apart two windows open, in epochs.
        max_lag: the largest lag read inside a window.

    Returns:
        An (n_windows, max_lag) array, empty below two windows.

    Raises:
        ValueError: when the series holds a value that is not defined, when the
            width or the step cannot declare a window, or when the largest lag is
            below one epoch or reaches the width of the window, which leaves no
            pair to correlate.
    """
    if max_lag < 1:
        raise ValueError(f"a largest lag is at least one epoch; got {max_lag}")
    x = _finite("a series", values)
    starts = _window_starts(x.size, window, step)
    if window <= max_lag:
        raise ValueError(
            f"a window of {window} epochs carries no lag of {max_lag}; the lag "
            "reaches the width"
        )
    out = np.zeros((starts.size, max_lag), dtype=float)
    for i, start in enumerate(starts):
        out[i] = _autocorrelation(x[start : start + window], max_lag)
    return out


def local_periodogram(
    values: Sequence[float], window: int, step: int, frequencies: Sequence[float]
) -> np.ndarray:
    """
    The power each sliding window puts at every frequency of a declared grid.

    Args:
        values: the series, at one scale and inside one participant.
        window: the width of a window, in epochs.
        step: how far apart two windows open, in epochs.
        frequencies: the grid, in cycles per epoch.

    Returns:
        An (n_windows, n_frequencies) array, empty below two windows.  A window
        that is flat, or that is exactly a straight line, comes back at zero
        power across the grid.

    Raises:
        ValueError: when the series or the grid holds a value that is not
            defined, when the grid is empty or holds a frequency at or below
            zero, or when the width or the step cannot declare a window.
    """
    freqs = np.atleast_1d(_finite("a frequency grid", frequencies))
    if freqs.size == 0:
        raise ValueError("a frequency grid holds at least one frequency; got none")
    if freqs.min() <= 0.0:
        raise ValueError(
            f"a frequency is positive, in cycles per epoch; got {freqs.min()}"
        )
    x = _finite("a series", values)
    starts = _window_starts(x.size, window, step)
    epochs = np.arange(window, dtype=float)
    angle = 2.0 * np.pi * np.outer(epochs, freqs)
    cosine, sine = np.cos(angle), np.sin(angle)
    out = np.zeros((starts.size, freqs.size), dtype=float)
    for i, start in enumerate(starts):
        block = x[start : start + window]
        detrended = signal.detrend(block, type="linear")
        # A window that is flat, or that is a straight line, leaves a residual at
        # the rounding floor of its own magnitude and holds no power at any
        # frequency.  Reading that residual would hand the density below a
        # spectral shape assembled out of float error.
        floor = _DUST * float(np.abs(block).max())
        if float(detrended @ detrended) <= window * floor * floor:
            continue
        out[i] = ((detrended @ cosine) ** 2 + (detrended @ sine) ** 2) / window
    return out


def _densities(power: np.ndarray) -> np.ndarray:
    """Each window's periodogram as a density, uniform where it holds no power."""
    total = power.sum(axis=1, keepdims=True)
    carried = total > 0.0
    share = power / np.where(carried, total, 1.0)
    return np.where(carried, share, 1.0 / power.shape[1])


def _cusum(densities: np.ndarray) -> Tuple[float, int]:
    """The largest absolute cumulative deviation from the average density."""
    # Divided by the root of the window count, so a longer record does not raise
    # the statistic on its length alone.
    walk = np.abs(np.cumsum(densities - densities.mean(axis=0), axis=0))
    walk = walk / math.sqrt(densities.shape[0])
    per_window = walk.max(axis=1)
    return float(per_window.max()), int(np.argmax(per_window))


@dataclass(frozen=True)
class SpectralChange:
    """Where the local spectrum turns, how far it turns, and how often chance does."""

    statistic: float
    window: int
    p_value: float

    def as_row(self) -> Dict[str, float]:
        return {
            "statistic": self.statistic,
            "window": float(self.window),
            "p_value": self.p_value,
        }


def spectral_cusum(
    power: np.ndarray, n_permutations: int = 199, seed: int = 0
) -> SpectralChange:
    """
    Where the shape of the local periodogram changes, and how a reshuffle prices it.

    The window order carries the change and the set of window densities carries
    nothing about it, so a reshuffle of that order is the null of one spectrum
    across the record.  The reshuffle takes the densities as exchangeable, which
    holds for disjoint windows; two windows sharing epochs are correlated, and
    the walk over a correlated order runs above what a reshuffle reaches.

    Args:
        power: an (n_windows, n_frequencies) local periodogram.
        n_permutations: how many reshuffles of the window order the null draws.
        seed: the generator seed of the reshuffle.

    Returns:
        The statistic, the window it falls at, and the permutation p-value.  A
        periodogram of fewer than two windows reports no change at all.

    Raises:
        ValueError: when the periodogram is not an (n_windows, n_frequencies)
            array holding at least one frequency, when it holds a value that is
            not defined, or when the null draws no reshuffle at all, which leaves
            the p-value undefined.
    """
    p = _finite("a local periodogram", power)
    if p.ndim != 2 or p.shape[1] < 1:
        raise ValueError(
            f"a local periodogram is (n_windows, n_frequencies); got shape {p.shape}"
        )
    if n_permutations < 1:
        raise ValueError(
            f"a permutation price draws at least one reshuffle; got {n_permutations}"
        )
    if p.shape[0] < 2:
        return SpectralChange(0.0, 0, 1.0)
    densities = _densities(p)
    statistic, at = _cusum(densities)
    if statistic <= _FLOOR:
        return SpectralChange(0.0, 0, 1.0)
    rng = np.random.default_rng(seed)
    null = np.array(
        [_cusum(rng.permutation(densities, axis=0))[0] for _ in range(n_permutations)]
    )
    p_value = (1 + float(np.sum(null >= statistic))) / (1 + null.size)
    return SpectralChange(statistic, at, p_value)


def persistence_variance(trajectory: np.ndarray) -> float:
    """
    The variance of the lag-1 autocorrelation over windows, which is the magnitude.

    Args:
        trajectory: an (n_windows, n_lags) autocorrelation trajectory.

    Returns:
        The variance across windows, zero below two windows.

    Raises:
        ValueError: when the trajectory is not an (n_windows, n_lags) array
            holding at least one lag, or holds a value that is not defined.
    """
    values = _finite("a trajectory", trajectory)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(
            f"a trajectory is (n_windows, n_lags); got shape {values.shape}"
        )
    if values.shape[0] < 2:
        return 0.0
    return float(values[:, 0].var())


def persistence_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    The root mean squared difference between two autocorrelation trajectories.

    The shorter record sets how many windows are compared, since a real log and
    a synthetic one can carry a different number of them.

    Raises:
        ValueError: when either trajectory is not an (n_windows, n_lags) array,
            when the two carry a different set of lags, or when either holds a
            value that is not defined.
    """
    left = _finite("a trajectory", a)
    right = _finite("a trajectory", b)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError(
            f"two trajectories carry the same lags; got shapes {left.shape} and "
            f"{right.shape}"
        )
    n = min(left.shape[0], right.shape[0])
    if n == 0:
        return 0.0
    return float(np.sqrt(np.mean((left[:n] - right[:n]) ** 2)))


def persistence_drift(values: Sequence[float], period: float) -> float:
    """
    A11: how much the lag-1 autocorrelation of a series moves over the record.

    Args:
        values: the series, at one scale and inside one participant.
        period: the calendar period of that scale, in epochs.

    Returns:
        The standard deviation of the lag-1 autocorrelation across windows, and
        zero for a constant series or one too short to open two windows.

    Raises:
        ValueError: when the period is not a positive finite number of epochs, or
            when the series holds a value that is not defined.
    """
    window, step, max_lag = window_plan(period)
    trajectory = rolling_autocorrelation(values, window, step, max_lag)
    # The autocorrelation of a window of w epochs carries sampling noise of order
    # one over the root of w, so the scalar holds a floor that a steady series
    # reaches and that both sides of the comparison are read under.
    return math.sqrt(persistence_variance(trajectory))
