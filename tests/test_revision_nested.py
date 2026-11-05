"""Distribution-level checks for the revised paper model."""

import copy
import itertools

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import nbinom

from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.nested.fit import FitConfig, Fitter
from src.nested.inference import infer, monitor
from src.nested.law import (
    allocation_logpmf,
    coarse_logpmf,
    sample,
    split_logpmf,
    split_sample,
    target_score,
)
from src.nested.model import Parent, Record, initialize


def record(n=24, branching=(3, 2), labels=2, name="subject"):
    h = ScaleHierarchy.from_uniform_branching(n, list(branching))
    data = MultiScaleData(
        h,
        [f"label{i}" for i in range(labels)],
        [np.zeros((t, labels), dtype=np.int64) for t in h.H],
        [np.zeros((t, labels), dtype=np.int64) for t in h.H],
        [np.column_stack([np.sin(np.arange(t))]) for t in h.H],
    )
    return Record(name, data)


def model(rec, **kwargs):
    K = rec.data.K
    return initialize([rec], [1] * K, [0] * K, [0] * K, **kwargs)


def observed(n=36, name="subject", seed=3):
    rec = record(n=n, name=name)
    fitted = model(rec)
    for target in fitted.levels[-1].targets:
        target.intercept[:] = np.log([2, 8])
        target.zero_logit = -8
    return sample(fitted, rec, seed)[0]


def test_censored_tail_and_surplus_are_normalized():
    x = np.arange(5)
    ll = coarse_logpmf(x, x, np.log(8), 0, -1, [2, 3], 4, True)
    assert np.exp(ll).sum() == pytest.approx(1)
    expected_tail = (1 / (1 + np.exp(-1))) * nbinom.sf(3, 2, 2 / 10)
    assert np.exp(ll[-1]) == pytest.approx(expected_tail)
    a = coarse_logpmf(np.array([2]), np.array([7]), 0, np.log(4), -2, [2, 3], 8)
    b = coarse_logpmf(np.array([2]), np.array([2]), 0, 0, -2, [2, 3], 8, True)
    assert a[0] - b[0] == pytest.approx(nbinom.logpmf(5, 3, 3 / 7))
    assert np.isneginf(
        coarse_logpmf(np.array([0]), np.array([1]), 0, 0, -2, [2, 3], 8)[0]
    )


def test_count_only_ignores_duration_values():
    args = (np.array([3]), np.array([0]), 1, 2, -1, [2, 4])
    a = coarse_logpmf(*args, count_only=True)
    b = coarse_logpmf(args[0], np.array([100000]), *args[2:], count_only=True)
    np.testing.assert_array_equal(a, b)


def test_truncated_split_sampler_matches_enumerated_mass():
    support = np.arange(2, 5)
    p = np.exp(split_logpmf(support, 7, 1.2, 3.4, 2, 4))
    assert p.sum() == pytest.approx(1)
    rng = np.random.default_rng(0)
    draws = np.array([split_sample(7, 1.2, 3.4, 2, 4, rng) for _ in range(6000)])
    np.testing.assert_allclose([np.mean(draws == x) for x in support], p, atol=0.025)


@pytest.mark.parametrize("n,lo,hi", [(0, 0, 0), (3, 3, 3), (5, 0, 5)])
def test_forced_final_child(n, lo, hi):
    assert split_sample(n, 1, 0, lo, hi, np.random.default_rng(1)) == n
    assert split_logpmf(np.array([n]), n, 1, 0, lo, hi)[0] == 0


def test_fixed_score_unbounded_chain_matches_dm():
    x = np.array([2, 0, 3])
    alpha = np.array([1.2, 2.3, 0.7])
    n = x.sum()
    sequential = 0.0
    for i in range(3):
        sequential += split_logpmf(
            np.array([x[i]]), n, alpha[i], alpha[i + 1 :].sum(), 0, n
        )[0]
        n -= x[i]
    reference = gammaln(x.sum() + 1) - gammaln(x + 1).sum()
    reference += gammaln(alpha.sum()) - gammaln(alpha.sum() + x.sum())
    reference += (gammaln(alpha + x) - gammaln(alpha)).sum()
    assert sequential == pytest.approx(reference)


def test_within_block_lag_joint_mass_sums_to_one():
    rec = record(n=3, branching=(3,), labels=1)
    fitted = model(rec)
    fitted.levels[0].targets[0].parents[Parent("behavior", 0, 1)] = np.array(
        [1.1, -0.4]
    )
    q = infer(rec, fitted, inner=1)
    total = 0.0
    for x in itertools.product(range(2), repeat=3):
        if sum(x) != 2:
            continue
        active = np.flatnonzero(x)
        for first in range(3):
            y = np.array(x)
            y[active[0]] += first
            y[active[1]] += 2 - first
            rec.data.X[0][:, 0] = x
            rec.data.Y[0][:, 0] = y
            rec.data.X[1][:, 0] = 2
            rec.data.Y[1][:, 0] = 4
            ll = allocation_logpmf(rec, fitted, 0, 0, q[0].gamma, q[0].m_H)
            total += np.exp(ll).sum()
    assert total == pytest.approx(1, abs=1e-9)


def test_draws_preserve_capacity_floor_and_full_durations():
    rec = observed()
    rec.validate()
    assert set(rec.data.X[0].ravel()) <= {0, 1}
    assert np.all(rec.data.Y[0] >= rec.data.X[0])
    assert rec.data.Y[0].max() > 1
    assert rec.data.check_reconciliation() == {"RecErr_X": 0, "RecErr_Y": 0}


def test_support_violation_is_not_repaired():
    rec = record()
    rec.data.X[0][4, 0] = 2
    rec.data.Y[0][4, 0] = 2
    with pytest.raises(ValueError, match="support violation"):
        rec.validate()


def test_masks_and_participant_boundaries():
    rec = record()
    rec.data.observed = np.ones(24, dtype=bool)
    rec.data.observed[6:12] = False
    rec.data._ladder = None
    rows = rec.rows(0, 2)
    assert not rows[:2].any()
    assert not rows[6:14].any()
    assert rows[14:].all()
    assert not record(name="second").rows(0, 2)[:2].any()


def test_positive_lag_graph():
    with pytest.raises(ValueError, match="positive"):
        Parent("behavior", 0, 0)
    assert Parent("context", 0, 0).lag == 0


def test_cohort_penalty_paid_once_and_monitor_is_fresh():
    a, b = observed(name="a"), observed(name="b", seed=4)
    fitted = initialize([a, b], [1, 1, 1], [0, 0, 0], [1, 1, 1])
    fitted.levels[-1].targets[0].parents[Parent("behavior", 0, 1)] = np.array(
        [0.1, 0.2]
    )
    qa, qb = infer(a, fitted, 1), infer(b, fitted, 1)
    combined = monitor([a, b], fitted, {"a": qa, "b": qb})
    separate = monitor([a], fitted, {"a": qa}) + monitor([b], fitted, {"b": qb})
    assert combined == pytest.approx(separate + fitted.penalty())
    fitted.levels[-1].targets[0].intercept[0] += 1
    assert monitor([a, b], fitted, {"a": qa, "b": qb}) != combined


def test_fit_retains_fresh_monotone_monitor():
    rec = observed(n=24)
    fitted = initialize([rec], [1, 1, 2], [0, 0, 0], [1, 1, 1])
    result = Fitter(
        FitConfig(outer=2, inner=1, profile=1, shifts=0, structure=False)
    ).fit([rec], fitted)
    values = [row["retained"] for row in result.trace]
    assert np.all(np.diff(values) >= 0)
    assert values[-1] == pytest.approx(monitor([rec], result.model, result.posteriors))


def test_factor_update_uses_corrected_likelihood():
    rec = observed(n=18)
    fitted = initialize([rec], [1, 1, 2], [0, 0, 1], [1, 1, 1])
    posterior = infer(rec, fitted, inner=2)
    assert posterior[-1].lag_H.shape == (2, 1, 1)
    assert np.isfinite(posterior[-1].m_H).all()
    assert np.isfinite(target_score(rec, fitted, 2, 0, posterior))
