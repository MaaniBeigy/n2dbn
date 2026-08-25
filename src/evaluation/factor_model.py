"""
evaluation/factor_model.py
==========================
The common component of a panel, estimated by principal components, and how two
panels' factor spaces are compared.

B2 of the nonstationarity grid asks whether the series of a panel move together.
One small set of trajectories carrying most of what every column does is a
common factor, and the row holds when a synthetic panel carries the same number
of them, spanning the same space, on the same shape.

The estimator is the static approximate factor model fitted by principal
components, which is the standard estimator of the common component of a
dynamic factor model.  The panel is standardized by column, so a participant
with many events and one with few weigh the same, and one singular value
decomposition of the standardized panel gives the factors, the loadings, the
common component and the idiosyncratic residual together.

How many factors a panel holds is a choice, so it is made by the Bai and Ng
information criteria IC_p1, IC_p2 and IC_p3 over one declared maximum count.
All three share a fit and differ only in the penalty, so all three are reported
side by side; a count the three agree on is a stronger reading than any one of
them alone.

Two panels are compared on the principal angles between their loading spaces,
which is the comparison of two subspaces that no choice of basis moves, and on
the correlation between two factor trajectories after their signs are aligned,
since a factor and its negative are the same factor.  What a caller gets is the
fit, the criteria with the count each selects, the variance share the kept
factors explain, and those two comparisons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

#: The three criteria, named as they are in the source stated below.
IC_P1 = "IC_p1"
IC_P2 = "IC_p2"
IC_P3 = "IC_p3"
CRITERIA: Tuple[str, ...] = (IC_P1, IC_P2, IC_P3)

#: The largest factor count the criteria choose between.
MAX_FACTORS = 8

#: The criterion a panel comparison reports its factor count from.  IC_p2 prices
#: a factor at the smaller of the two panel dimensions, which is the binding one
#: on a log holding far more epochs than participants.
SELECTION = IC_P2

#: Kept out of a logarithm of a residual variance an exact fit drives to zero.
_FLOOR = 1e-12

#: A column whose spread sits below this holds one value at every epoch.
_SCALE_FLOOR = 1e-12

#: A singular value below this share of the largest carries no direction.
_RANK_TOLERANCE = 1e-10


def _finite(what: str, x: np.ndarray) -> np.ndarray:
    """
    An array once every entry is checked, naming the first entry that is not.

    Raises:
        ValueError: when the array holds a non-finite entry.
    """
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{what} is finite; got {x[~np.isfinite(x)].ravel()[0]}")
    return x


def _whole(what: str, value: float) -> int:
    """
    A count as an integer, refusing what is not a finite whole number.

    A fractional or infinite count has no answer.  Truncating one to the integer
    below it would answer a different question in silence.

    Raises:
        ValueError: when the count is infinite, undefined or fractional.
    """
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{what} is a whole number; got {value}")
    return int(number)


def _panel(values: np.ndarray) -> np.ndarray:
    """
    A panel as a (T, N) float matrix, refusing what a factor model cannot fit.

    Raises:
        ValueError: when the panel is not two-dimensional, holds no epoch or no
            column, or holds a non-finite entry.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"a panel is two-dimensional; got {x.ndim} dimensions")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"a panel holds an epoch and a column; got shape {x.shape}")
    return _finite("a panel", x)


def standardize(panel: np.ndarray) -> np.ndarray:
    """
    A panel centred and scaled by column, a column of no spread left at zero.

    The rule for a constant column is explicit: it is set to zero across every
    epoch, so it carries no direction into the factor space and takes a zero
    loading, and it still counts in the panel width the criteria are stated on.
    """
    x = _panel(panel)
    spread = x.std(axis=0)
    alive = spread > _SCALE_FLOOR
    scales = np.where(alive, spread, 1.0)
    return np.where(alive, (x - x.mean(axis=0)) / scales, 0.0)


@dataclass(frozen=True)
class FactorFit:
    """One principal component fit: the factors, the loadings and the residual."""

    factors: np.ndarray
    loadings: np.ndarray
    common: np.ndarray
    idiosyncratic: np.ndarray
    # Every singular value and every share, so the whole scree is readable
    # beside the factors that were kept.
    singular_values: np.ndarray
    variance_shares: np.ndarray

    @property
    def n_factors(self) -> int:
        return int(self.factors.shape[1])

    @property
    def explained(self) -> float:
        """The share of the standardized panel's variance the kept factors hold."""
        return float(self.variance_shares[: self.n_factors].sum())


def principal_components(panel: np.ndarray, n_factors: int = 1) -> FactorFit:
    """
    The factor model of a panel, by singular value decomposition.

    The factors are scaled so their second moment is the identity, which is the
    normalization the loadings are read against, and the common component is
    then the factors times the loadings whatever the panel's shape.

    Args:
        panel: the (T, N) panel, one column per series.
        n_factors: how many factors to keep.

    Raises:
        ValueError: when the count is not a whole number, or when it is below
            one or above the rank the panel carries, which is the smaller of
            its two dimensions.
    """
    z = standardize(panel)
    epochs, width = z.shape
    limit = min(epochs, width)
    keep = _whole("a factor count", n_factors)
    if keep < 1 or keep > limit:
        raise ValueError(
            f"a panel of shape {z.shape} carries 1 to {limit} factors; got {keep}"
        )
    left, singular, right = np.linalg.svd(z, full_matrices=False)
    energy = singular**2
    total = float(energy.sum())
    shares = energy / total if total > 0.0 else np.zeros_like(energy)
    factors = left[:, :keep] * math.sqrt(epochs)
    loadings = right[:keep].T * singular[:keep] / math.sqrt(epochs)
    common = factors @ loadings.T
    return FactorFit(
        factors=factors,
        loadings=loadings,
        common=common,
        idiosyncratic=z - common,
        singular_values=singular,
        variance_shares=shares,
    )


@dataclass(frozen=True)
class BaiNg:
    """The three criteria over a factor count sweep, and the count each selects."""

    counts: np.ndarray
    residual_variance: np.ndarray
    ic_p1: np.ndarray
    ic_p2: np.ndarray
    ic_p3: np.ndarray
    selected: Dict[str, int]


def bai_ng_criteria(panel: np.ndarray, max_factors: int = MAX_FACTORS) -> BaiNg:
    """
    The Bai-Ng information criteria over a factor count sweep, and their choices.

    The sweep runs from zero factors to the smaller of `max_factors` and the
    rank the panel carries, so a request above that rank is answered with the
    counts the panel admits.

    Raises:
        ValueError: when the maximum factor count is not a whole number or is
            below one, or when the panel is too small for the penalties to
            price a factor at anything.
    """
    ceiling = _whole("a factor count", max_factors)
    if ceiling < 1:
        raise ValueError(f"a sweep reaches at least one factor; got {ceiling}")
    z = standardize(panel)
    epochs, width = z.shape
    # IC_p1 prices a factor at ln(NT / (N + T)), which falls to zero and then
    # below it as (T - 1)(N - 1) drops to one and under.  A criterion that pays
    # nothing for a factor, or pays to take one, runs the sweep to its own top
    # whatever the panel holds, so a panel that small carries no count to read.
    if (epochs - 1) * (width - 1) <= 1:
        raise ValueError(
            f"the criteria need (T - 1)(N - 1) above one; got shape {z.shape}"
        )
    top = min(ceiling, epochs, width)
    energy = np.linalg.svd(z, compute_uv=False) ** 2
    counts = np.arange(top + 1)
    # V(k), the mean squared idiosyncratic residual of a k-factor fit, is what
    # the leading k singular values leave behind.
    left_over = float(energy.sum()) - np.concatenate([[0.0], np.cumsum(energy)])
    residual = np.maximum(left_over[: top + 1] / (epochs * width), _FLOOR)
    # Bai and Ng (2002), Determining the Number of Factors in Approximate Factor
    # Models, Econometrica 70(1), 191-221, criteria PC_p1 to PC_p3 in log form:
    #   IC_p1(k) = ln V(k) + k ((N + T) / (N T)) ln(N T / (N + T))
    #   IC_p2(k) = ln V(k) + k ((N + T) / (N T)) ln(min(N, T))
    #   IC_p3(k) = ln V(k) + k (ln(min(N, T)) / min(N, T))
    ratio = (epochs + width) / (epochs * width)
    smaller = float(min(epochs, width))
    log_v = np.log(residual)
    ic_p1 = log_v + counts * ratio * math.log(epochs * width / (epochs + width))
    ic_p2 = log_v + counts * ratio * math.log(smaller)
    ic_p3 = log_v + counts * math.log(smaller) / smaller
    return BaiNg(
        counts=counts,
        residual_variance=residual,
        ic_p1=ic_p1,
        ic_p2=ic_p2,
        ic_p3=ic_p3,
        selected={
            IC_P1: int(counts[int(np.argmin(ic_p1))]),
            IC_P2: int(counts[int(np.argmin(ic_p2))]),
            IC_P3: int(counts[int(np.argmin(ic_p3))]),
        },
    )


# ---------------------------------------------------------------------------
# How two factor spaces are compared
# ---------------------------------------------------------------------------


def _orthonormal(loadings: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the column space of a loading matrix."""
    left, singular, _ = np.linalg.svd(loadings, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return left[:, :0]
    return left[:, singular > _RANK_TOLERANCE * singular[0]]


def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    The principal angles between two loading spaces, in radians, smallest first.

    Each angle is the arccosine of a singular value of the product of the two
    orthonormal bases.  A zero angle is a direction the two spaces share and
    pi/2 is a direction one holds and the other is blind to, and neither figure
    moves when a space is handed a different basis.

    Raises:
        ValueError: when either argument is not two-dimensional, holds a
            non-finite entry, when the two are stated on different series
            counts, or when either spans nothing.
    """
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(
            f"loadings are two-dimensional; got {left.ndim} and {right.ndim}"
        )
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"two loading spaces stand on one series count; got {left.shape[0]} "
            f"and {right.shape[0]}"
        )
    _finite("a loading space", left)
    _finite("a loading space", right)
    basis_a = _orthonormal(left)
    basis_b = _orthonormal(right)
    if basis_a.shape[1] == 0 or basis_b.shape[1] == 0:
        raise ValueError(
            f"a loading space of rank zero carries no angle; got ranks "
            f"{basis_a.shape[1]} and {basis_b.shape[1]}"
        )
    cosines = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    return np.arccos(np.clip(cosines, 0.0, 1.0))


def factor_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """
    The correlation between two factor trajectories, once their signs are aligned.

    The figure sits in [0, 1].  A trajectory holding one value moves nowhere and
    correlates with nothing, so it scores zero.

    Raises:
        ValueError: when either trajectory holds a non-finite entry, when the
            two are of different lengths, or when they hold fewer than two
            epochs.
    """
    left = np.asarray(a, dtype=float).ravel()
    right = np.asarray(b, dtype=float).ravel()
    if left.size != right.size:
        raise ValueError(
            f"two trajectories are of one length; got {left.size} and {right.size}"
        )
    if left.size < 2:
        raise ValueError(f"a correlation needs two epochs; got {left.size}")
    _finite("a trajectory", left)
    _finite("a trajectory", right)
    u = left - left.mean()
    v = right - right.mean()
    spans = (float(np.abs(u).max()), float(np.abs(v).max()))
    if spans[0] <= 0.0 or spans[1] <= 0.0:
        return 0.0
    # A correlation is unmoved by the scale of either trajectory, and taking the
    # scale out first keeps the squared norm of a large one inside the floats:
    # every entry now sits in [-1, 1], so the inner product and the two norms
    # are bounded by the epoch count.
    u = u / spans[0]
    v = v / spans[1]
    # A factor is identified up to sign, and flipping one trajectory negates the
    # inner product, so the alignment is taken there.
    return abs(float(u @ v)) / float(np.linalg.norm(u) * np.linalg.norm(v))


@dataclass(frozen=True)
class FactorMatch:
    """The four figures the common factor row reports on one pair of panels."""

    count_difference: int
    mean_angle: float
    first_correlation: float
    share_difference: float

    def as_row(self) -> Dict[str, float]:
        return {
            "count_difference": self.count_difference,
            "mean_angle": self.mean_angle,
            "first_correlation": self.first_correlation,
            "share_difference": self.share_difference,
        }


def compare_factors(
    real: np.ndarray,
    synthetic: np.ndarray,
    n_factors: int = 1,
    max_factors: int = MAX_FACTORS,
    criterion: str = SELECTION,
) -> FactorMatch:
    """
    Fit two panels and compare the count, the space and the leading trajectory.

    Raises:
        ValueError: when the criterion is not one of `CRITERIA`, or when the two
            panels are of different shapes, since the angles stand on one series
            count and the correlation on one span.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion {criterion!r}; expected one of {list(CRITERIA)}")
    left = _panel(real)
    right = _panel(synthetic)
    if left.shape != right.shape:
        raise ValueError(
            f"a factor match stands on one panel shape; these are {left.shape} "
            f"and {right.shape}"
        )
    fit_a = principal_components(left, n_factors)
    fit_b = principal_components(right, n_factors)
    counts = (
        bai_ng_criteria(left, max_factors).selected[criterion],
        bai_ng_criteria(right, max_factors).selected[criterion],
    )
    return FactorMatch(
        count_difference=abs(counts[0] - counts[1]),
        mean_angle=float(principal_angles(fit_a.loadings, fit_b.loadings).mean()),
        first_correlation=factor_correlation(fit_a.factors[:, 0], fit_b.factors[:, 0]),
        share_difference=abs(fit_a.explained - fit_b.explained),
    )


def factor_share(values: Sequence[float], period: float) -> float:
    """
    The variance share the leading component of one series' blocks holds.

    One participant's series is one column of the panel B2 is stated on, so the
    scalar is the special case of the row: the series is laid out as whole
    blocks of the calendar period and each position in a block becomes a column.
    Standardizing by column takes the average block out, so what the leading
    component carries is the one shape the blocks depart from that average
    along, and the share is how much of the departure that shape explains.

    A series holding fewer than two whole blocks carries no panel and scores
    zero, and a series whose blocks are identical departs from the average by
    nothing and scores zero, a constant series among them.  Two blocks put the
    standardized panel at rank one and the share at one, which is all two
    cycles can say.

    Args:
        values: one participant's series at one scale.
        period: the calendar period of that scale, in epochs.

    Raises:
        ValueError: when the calendar period is infinite or undefined, when it
            is under two epochs, since a block of one epoch holds no shape to
            share, or when the series holds a non-finite entry.
    """
    if not math.isfinite(period):
        raise ValueError(f"a calendar period is finite; got {period}")
    if period < 2.0:
        raise ValueError(f"a block holds two epochs; got period {period}")
    # The series is checked whole, at every length.  A series too short to lay
    # out as blocks is refused for a non-finite entry on the same terms as one
    # long enough to fit, so the same bad data is answered the same way.
    x = _finite("a series", np.asarray(values, dtype=float).ravel())
    width = int(round(period))
    blocks = x.size // width
    if blocks < 2:
        return 0.0
    panel = x[: blocks * width].reshape(blocks, width)
    return principal_components(panel, 1).explained
