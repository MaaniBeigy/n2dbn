"""
tests/test_phase34_penalty_frontier.py
======================================
Reading the penalty frontier on every scale that carries a true edge.

The frontier used to be scored at the middle scale alone, which read the whole
truth while the middle scale was the only one carrying edges. The grid now
places them at the coarsest scale as well, so a frontier read on one scale can
disagree with the same frontier read on the other and the disagreement is a
property of the reading rather than of the criterion. The summary therefore
reports the agreement per scale and for the mean, and these tests pin that.
"""

from __future__ import annotations

import pytest

from experiments.exp9_selection_and_search import _summary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _penalty_row(lam: float, nll: float, f1_k2: float, f1_k3: float) -> dict:
    return dict(
        table="penalty",
        benchmark="synthetic",
        lam=lam,
        val_nll=nll,
        F1_k2=f1_k2,
        F1_k3=f1_k3,
        F1_mean=(f1_k2 + f1_k3) / 2.0,
        SHD_total=1.0,
    )


@pytest.fixture
def frontier():
    """A frontier whose two scales peak at different penalties."""
    return [
        _penalty_row(0.5, nll=10.0, f1_k2=0.40, f1_k3=0.90),
        _penalty_row(1.0, nll=12.0, f1_k2=0.60, f1_k3=0.70),
        _penalty_row(2.0, nll=14.0, f1_k2=0.80, f1_k3=0.50),
    ]


# ---------------------------------------------------------------------------
# The frontier verdict
# ---------------------------------------------------------------------------


def test_the_verdict_is_reported_for_each_scale_and_for_the_mean(frontier, capsys):
    _summary(frontier, [])
    out = capsys.readouterr().out
    for key in ("F1_k2", "F1_k3", "F1_mean"):
        assert f"{key} peaks at lam=" in out


def test_two_scales_can_disagree_about_the_same_frontier(frontier, capsys):
    # Evidence picks the lowest held-out loss, which is lam 0.5. The coarsest
    # scale agrees there and the middle scale peaks at 2.0, so a frontier read
    # on the middle scale alone reports a disagreement the coarsest does not.
    _summary(frontier, [])
    lines = capsys.readouterr().out.splitlines()
    k2 = next(ln for ln in lines if "F1_k2 peaks" in ln)
    k3 = next(ln for ln in lines if "F1_k3 peaks" in ln)
    assert "agree: False" in k2
    assert "agree: True" in k3


def test_a_scale_the_run_did_not_score_is_skipped_rather_than_defaulted(capsys):
    older = [
        dict(table="penalty", benchmark="synthetic", lam=l, val_nll=n, F1_k2=f)
        for l, n, f in ((0.5, 10.0, 0.4), (2.0, 14.0, 0.8))
    ]
    _summary(older, [])
    out = capsys.readouterr().out
    assert "F1_k2 peaks at lam=" in out
    assert "F1_k3" not in out
    assert "F1_mean" not in out


def test_the_external_frontier_is_still_read_on_its_single_score(capsys):
    ext = [
        dict(table="penalty", benchmark="external", lam=l, val_nll=n, F1=f)
        for l, n, f in ((0.25, 5.0, 0.8), (1.0, 9.0, 0.6))
    ]
    _summary(ext, [])
    out = capsys.readouterr().out
    assert "external" in out and "F1 peaks at lam=0.25" in out
    assert "agree: True" in out
