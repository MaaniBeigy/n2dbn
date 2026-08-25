"""
evaluation/seasonality_drift.py
===============================
Complex demodulation of one series at a declared period, and how far the rhythm
it reads moves across a log.

Row A07 asks whether a discovered rhythm holds still.  Complex demodulation
answers with a trajectory: the series is centred, multiplied by the carrier
`exp(-2i pi t / period)` and low-passed, which leaves one complex number per
epoch whose modulus is half the local amplitude of that rhythm and whose
argument is its local phase.  A rhythm that grows, fades or slides inside the
day is read off those two trajectories directly.

The low-pass is a centred moving average spanning a whole number of periods.  A
boxcar of exactly one period has an exact null at the carrier frequency and at
every harmonic of it, so the sum-frequency image the multiplication leaves at
twice the carrier is removed exactly, and the filter is declared by one integer
with no cutoff to choose.  An even width takes the classical halved end taps,
which keeps the kernel symmetric about an epoch and its nulls exact.  Both
scales of the grid have an integer period, 24 epochs and 7; a fractional period
rounds the width to the nearest whole epoch and leaves a small residue of the
image.  The series is centred before the multiplication, because the mean
arrives at the carrier frequency and the boxcar clears it only where the whole
window fits, and a constant series has to read as zero amplitude at every epoch.
The average itself is read where the whole window fits, so a trajectory opens at
`Demodulation.start` and gives up the epochs the kernel overhangs, `width` of
them at an even width and `width - 1` at an odd one; an average over a partial
window keeps the image it was built to remove, which inflates the amplitude by
half at the two ends the drift summary reads.

A caller gets the demodulate, the amplitude and the phase trajectory, the
unwrapped phase and its slope (the offset between the rhythm in the series and
the declared carrier, in radians per epoch), a tie-corrected Mann-Kendall trend
of the amplitude trajectory written out here so the row stands on numpy and
scipy alone, and the two figures the row is scored on: a relative distance
between two amplitude trajectories and a mean circular distance between two
phase trajectories.  The summary scalar `amplitude_drift` is the relative change
in the amplitude trajectory from its first quarter to its last, and it is zero
for a series holding one level, whose trajectory is silent at every epoch, and
for one shorter than two periods.

Two of those figures ask for care.  The amplitude trajectory is a moving
average, so two of its epochs closer together than one window share most of what
they average, and the normal approximation behind the trend p-value reads that
agreement as evidence; `amplitude_trend` takes the stride between the points it
tests, and `Demodulation.window` is the stride that leaves one point per window.
A relative change is stated against the amplitude a series opens with, so a
participant whose rhythm sits at the level of the noise reports a drift that
moves with that noise, which is why the row reads the drift beside the amplitude
that A05 and A06 report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from scipy import stats

#: How many whole periods the low-pass boxcar spans.
LOWPASS_CYCLES = 1

#: A trajectory is read only where the series covers this many whole periods.
MIN_CYCLES = 2.0

#: The smallest share of the peak amplitude the first quarter is divided by.  A
#: rhythm growing out of silence has a first quarter at zero, and an unfloored
#: ratio reports infinity for it.
AMPLITUDE_FLOOR = 1e-3


def _finite(values: Sequence[float], what: str, dtype: type = float) -> np.ndarray:
    """One series or trajectory as an array, with the first bad value refused."""
    x = np.asarray(values, dtype=dtype)
    if x.size and not np.all(np.isfinite(x)):
        raise ValueError(f"{what} is finite; got {x[~np.isfinite(x)][0]}")
    return x


def lowpass_window(period: float, cycles: int = LOWPASS_CYCLES) -> int:
    """
    How many epochs the centred low-pass spans, a whole number of periods.

    Raises:
        ValueError: when the period is at or below one epoch, which no carrier
            resolves, or when the window holds less than one whole period.
    """
    if not np.isfinite(period) or period <= 1.0:
        raise ValueError(f"a demodulation period is above one epoch; got {period}")
    if cycles < 1:
        raise ValueError(
            f"a low-pass window spans at least one whole period; got {cycles}"
        )
    return max(2, int(round(float(period) * cycles)))


def _lowpass(z: np.ndarray, width: int) -> np.ndarray:
    """The centred moving average of a complex series where the window fits whole."""
    half = width // 2
    kernel = np.ones(2 * half + 1)
    if width % 2 == 0:
        # Halved end taps keep an even kernel symmetric about an epoch, which is
        # what holds its null at the carrier exact.
        kernel[0] = kernel[-1] = 0.5
    return np.convolve(z, kernel / kernel.sum(), mode="valid")


def complex_demodulate(
    values: Sequence[float], period: float, cycles: int = LOWPASS_CYCLES
) -> np.ndarray:
    """
    The low-passed product of the centred series with the carrier at one period.

    The result opens at epoch `width // 2` and covers the span where the average
    sees a whole window, which is `width` epochs shorter than the series at an
    even width and `width - 1` shorter at an odd one.

    Args:
        values: the series, at one scale and inside one participant.
        period: the calendar period of that scale, in epochs.
        cycles: how many whole periods the low-pass spans.

    Raises:
        ValueError: when the series is no longer than the window, which leaves
            no epoch the average can be read at, and when it holds a value that
            is not finite.
    """
    x = _finite(values, "a demodulated series")
    width = lowpass_window(period, cycles)
    if x.size <= width:
        raise ValueError(
            f"a demodulation reads more epochs than its window; got {x.size} "
            f"epochs and a window of {width}"
        )
    t = np.arange(x.size, dtype=float)
    return _lowpass((x - x.mean()) * np.exp(-2j * np.pi * t / period), width)


def amplitude_trajectory(demodulate: np.ndarray) -> np.ndarray:
    """
    The local amplitude of the rhythm at every epoch.

    Raises:
        ValueError: when the demodulate holds a value that is not finite.
    """
    # The product splits a real cosine into two conjugate halves and the
    # low-pass keeps one of them, so the modulus is half the amplitude.
    return 2.0 * np.abs(_finite(demodulate, "a demodulate", dtype=complex))


def phase_trajectory(demodulate: np.ndarray) -> np.ndarray:
    """
    The local phase of the rhythm at every epoch, in radians on (-pi, pi].

    Raises:
        ValueError: when the demodulate holds a value that is not finite.
    """
    return np.angle(_finite(demodulate, "a demodulate", dtype=complex))


def unwrapped_phase(phase: Sequence[float]) -> np.ndarray:
    """
    The phase trajectory with the jumps of its principal branch removed.

    Raises:
        ValueError: when the trajectory holds a value that is not finite.
    """
    return np.unwrap(_finite(phase, "a phase trajectory"))


def phase_slope(unwrapped: Sequence[float]) -> float:
    """
    The frequency offset of a demodulation, in radians per epoch.

    A positive slope says the rhythm in the series runs faster than the declared
    carrier, so its period is the shorter of the two.  A trajectory of fewer than
    two epochs has no slope and reports zero.

    Raises:
        ValueError: when the trajectory holds a value that is not finite.
    """
    p = _finite(unwrapped, "an unwrapped phase")
    if p.size < 2:
        return 0.0
    t = np.arange(p.size, dtype=float)
    t = t - t.mean()
    return float(t @ (p - p.mean()) / (t @ t))


@dataclass(frozen=True)
class Demodulation:
    """What one demodulation reads off one series at one period."""

    period: float
    window: int
    start: int
    defined: bool
    demodulate: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    unwrapped: np.ndarray
    frequency_offset: float


def demodulation(
    values: Sequence[float], period: float, cycles: int = LOWPASS_CYCLES
) -> Demodulation:
    """
    Demodulate one series at one period and read its two trajectories.

    A series covering less than `MIN_CYCLES` periods holds no trajectory, and
    neither does one no longer than the low-pass window.  Both come back flat at
    zero over the whole series with `defined` false, and the caller reads the
    flag.
    """
    x = _finite(values, "a demodulated series")
    width = lowpass_window(period, cycles)
    if x.size < MIN_CYCLES * float(period) or x.size <= width:
        return Demodulation(
            period=float(period),
            window=width,
            start=0,
            defined=False,
            demodulate=np.zeros(x.size, dtype=complex),
            amplitude=np.zeros(x.size),
            phase=np.zeros(x.size),
            unwrapped=np.zeros(x.size),
            frequency_offset=0.0,
        )
    z = complex_demodulate(x, period, cycles)
    phase = phase_trajectory(z)
    unwrapped = unwrapped_phase(phase)
    return Demodulation(
        period=float(period),
        window=width,
        start=width // 2,
        defined=True,
        demodulate=z,
        amplitude=amplitude_trajectory(z),
        phase=phase,
        unwrapped=unwrapped,
        frequency_offset=phase_slope(unwrapped),
    )


@dataclass(frozen=True)
class AmplitudeTrend:
    """The tie-corrected trend of one amplitude trajectory."""

    s: float
    tau: float
    z: float
    p_value: float


def amplitude_trend(amplitude: Sequence[float], stride: int = 1) -> AmplitudeTrend:
    """
    Whether an amplitude trajectory rises or falls, by a Mann-Kendall statistic.

    Args:
        amplitude: the amplitude trajectory of one demodulation.
        stride: how many epochs apart the tested points are.  The trajectory is
            a moving average over one window, so two entries closer than a whole
            window share epochs and the normal approximation below reads their
            agreement as evidence; passing `Demodulation.window` takes one point
            per window, which is the sample the approximation is stated for.

    Returns:
        The statistic, its tau, its normal score and its two-sided p-value.
        Fewer than three tested points and a trajectory of one repeated value
        both leave it undecided, which reports as a p-value of one.

    Raises:
        ValueError: when the stride is below one epoch, and when the trajectory
            holds a value that is not finite.
    """
    if stride < 1:
        raise ValueError(f"a trend stride is at least one epoch; got {stride}")
    a = _finite(amplitude, "an amplitude trajectory")[::stride]
    n = a.size
    if n < 3:
        return AmplitudeTrend(0.0, 0.0, 0.0, 1.0)
    # Only the pairs with i < j: the full sign matrix is antisymmetric and sums
    # to zero whatever the trajectory does.
    s = float(np.triu(np.sign(a[None, :] - a[:, None]), k=1).sum())
    tau = float(s / (0.5 * n * (n - 1.0)))
    _, ties = np.unique(a, return_counts=True)
    correction = float(np.sum(ties * (ties - 1.0) * (2.0 * ties + 5.0)))
    variance = (n * (n - 1.0) * (2.0 * n + 5.0) - correction) / 18.0
    if variance <= 0.0:
        return AmplitudeTrend(s, tau, 0.0, 1.0)
    z = (s - np.sign(s)) / math.sqrt(variance)
    return AmplitudeTrend(s, tau, float(z), float(2.0 * stats.norm.sf(abs(z))))


def _drift(amplitude: np.ndarray) -> float:
    """The relative change from the first quarter of a trajectory to its last."""
    quarter = amplitude.size // 4
    if quarter == 0:
        return 0.0
    peak = float(amplitude.max())
    if peak <= 0.0:
        return 0.0
    first = float(amplitude[:quarter].mean())
    last = float(amplitude[-quarter:].mean())
    return (last - first) / max(first, AMPLITUDE_FLOOR * peak)


def amplitude_drift(values: Sequence[float], period: float) -> float:
    """
    How much the seasonal amplitude evolved, first quarter of the series to last.

    Args:
        values: the series, at one scale and inside one participant.
        period: the calendar period of that scale, in epochs (24.0 at the hour
            scale, 7.0 at the day scale).

    Returns:
        The relative change in the amplitude trajectory.  A rhythm holding still
        reads 0.0, one halving reads -0.5 and one doubling reads 1.0.  A series
        holding one level at every epoch, one shorter than two periods and one
        of fewer than four epochs all read 0.0.  A series carrying only noise
        reads the drift that noise happens to hold, which over the lengths this
        row is scored on runs to a few tenths in either direction.
    """
    return _drift(demodulation(values, period).amplitude)


def amplitude_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """
    The gap between two amplitude trajectories, in units of their own amplitude.

    The gap is the root mean squared difference over the epochs the two share,
    divided by their mean amplitude, so one figure compares participants whose
    rhythms differ in size.  Two silent trajectories are at no distance.

    Raises:
        ValueError: when either trajectory holds a value that is not finite.
    """
    left = _finite(a, "an amplitude trajectory")
    right = _finite(b, "an amplitude trajectory")
    n = min(left.size, right.size)
    if n == 0:
        return 0.0
    gap = float(np.sqrt(np.mean((left[:n] - right[:n]) ** 2)))
    scale = float(np.mean(np.abs(np.concatenate([left[:n], right[:n]]))))
    if scale <= 0.0:
        return 0.0
    return gap / scale


def phase_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """
    The mean circular distance between two phase trajectories, in radians.

    A phase is an angle, so the gap at one epoch is the shorter way round the
    circle and never exceeds pi.  The mean is taken over the epochs the two
    share, and two empty trajectories are at no distance.

    Raises:
        ValueError: when either trajectory holds a value that is not finite.
    """
    left = _finite(a, "a phase trajectory")
    right = _finite(b, "a phase trajectory")
    n = min(left.size, right.size)
    if n == 0:
        return 0.0
    gap = (left[:n] - right[:n] + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.mean(np.abs(gap)))


@dataclass(frozen=True)
class SeasonalityMatch:
    """The four figures the evolving-seasonality row reports on one pair."""

    amplitude_distance: float
    phase_distance: float
    drift_gap: float
    offset_gap: float
    defined: bool

    def as_row(self) -> Dict[str, object]:
        return {
            "amplitude_distance": self.amplitude_distance,
            "phase_distance": self.phase_distance,
            "drift_gap": self.drift_gap,
            "offset_gap": self.offset_gap,
            "defined": self.defined,
        }


def match_seasonality(
    real: Sequence[float],
    synthetic: Sequence[float],
    period: float,
    cycles: int = LOWPASS_CYCLES,
) -> SeasonalityMatch:
    """
    Demodulate two series at one period and compare what each rhythm does.

    Both trajectories come back flat when either series is too short, and the
    four figures then read as agreement, so `defined` says whether the pair was
    read at all.
    """
    left = demodulation(real, period, cycles)
    right = demodulation(synthetic, period, cycles)
    return SeasonalityMatch(
        amplitude_distance=amplitude_distance(left.amplitude, right.amplitude),
        phase_distance=phase_distance(left.phase, right.phase),
        drift_gap=abs(_drift(left.amplitude) - _drift(right.amplitude)),
        offset_gap=abs(left.frequency_offset - right.frequency_offset),
        defined=left.defined and right.defined,
    )
