"""
tests/test_phase55_exp9_scoring_settings.py
===========================================
The held-out score of experiment 9 belongs to the model that was fitted.

Both halves of the experiment fit an estimator and then score the fit on the
validation split.  The two settings that change the answer, the E-step and the
cross-scale coupling, were stated as literals at the scoring call, so a row
fitted under any other pair reported evidence for a model nobody fitted.  Each
row now reads both off the estimator that produced it, which is what these
tests pin, on one small network and one small scenario.

The external half also names its per-scale columns, and the terms they average
have to be the ones the row summed, so the total and the columns are checked
against each other.
"""

from __future__ import annotations

import pytest

import experiments.exp9_selection_and_search as exp9
from src.learning.estep import EStep

NETWORK = ("cancer",)
SCENARIO = ("F2_L5_M2_zil",)
EPOCHS, BLOCK, ITERS = 60, 5, 1


@pytest.fixture
def seen(monkeypatch):
    """The settings every held-out score of a row was taken under."""
    calls = []
    score = exp9.held_out_nll_per_scale

    def spy(data, params, graph, **kwargs):
        calls.append(kwargs)
        return score(data, params, graph, **kwargs)

    monkeypatch.setattr(exp9, "held_out_nll_per_scale", spy)
    monkeypatch.setattr(exp9, "NETWORKS", NETWORK)
    return calls


@pytest.fixture
def uncoupled(monkeypatch):
    """Every estimator of the run fits with the coupling off and its own E-step."""
    build = exp9._sem

    def patched(*args, **kwargs):
        sem = build(*args, **kwargs)
        sem.couple_scales = False
        sem.estep = EStep(max_inner_iter=3)
        return sem

    monkeypatch.setattr(exp9, "_sem", patched)


def test_the_external_row_scores_under_the_estep_the_fit_ran(seen) -> None:
    exp9.external_row(1.0, BLOCK, EPOCHS, ITERS)

    assert len(seen) == len(NETWORK)
    assert all(isinstance(call["estep"], EStep) for call in seen)
    assert all(call["couple_scales"] is True for call in seen)


def test_an_external_fit_with_the_coupling_off_is_scored_with_it_off(
    seen, uncoupled
) -> None:
    exp9.external_row(1.0, BLOCK, EPOCHS, ITERS)

    assert [call["couple_scales"] for call in seen] == [False]
    assert [call["estep"].max_inner_iter for call in seen] == [3]


def test_the_external_row_totals_the_per_scale_terms_it_reports(seen) -> None:
    row = exp9.external_row(1.0, BLOCK, EPOCHS, ITERS)

    assert row["val_nll"] == pytest.approx(row["val_nll_k1"] + row["val_nll_k2"])


def test_the_synthetic_row_scores_under_the_estep_the_fit_ran(seen) -> None:
    exp9.synthetic_row(2.0, ITERS, SCENARIO)

    assert len(seen) == len(SCENARIO)
    assert all(isinstance(call["estep"], EStep) for call in seen)
    assert all(call["couple_scales"] is True for call in seen)


def test_a_synthetic_fit_with_the_coupling_off_is_scored_with_it_off(
    seen, uncoupled
) -> None:
    exp9.synthetic_row(2.0, ITERS, SCENARIO)

    assert [call["couple_scales"] for call in seen] == [False]
    assert [call["estep"].max_inner_iter for call in seen] == [3]
