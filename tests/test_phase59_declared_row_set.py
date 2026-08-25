"""
tests/test_phase59_declared_row_set.py
======================================
The rows a masked fit scores come from the search's declared budget.

Eroding the mask by the lags a model happens to hold would sum a model with a
long reach over fewer rows than one with a short reach, so the wider model
would score higher on the row count alone.  Every guard in the fit compares two
such numbers, so the row set has to be fixed before any model is scored.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.mask import declared_rows
from src.core.typed_graph import COUNT, OBSERVED, TypedEdge, TypedGraph
from src.learning.estep import EStep
from src.learning.evidence import evidence_chain
from src.learning.fine_sem import FineSEM, RegimeChain, fit_rows
from src.model.intensity import FineParams, FineSources


def _gapped(n: int = 12, gap: slice = slice(4, 6)) -> np.ndarray:
    flags = np.ones(n, dtype=bool)
    flags[gap] = False
    return flags


def test_a_fully_observed_scale_needs_no_budget() -> None:
    """A dataset with no gap erodes to itself, so no budget can change it."""
    flags = np.ones(8, dtype=bool)
    np.testing.assert_array_equal(declared_rows(flags, None, "a caller"), flags)
    np.testing.assert_array_equal(declared_rows(flags, 3, "a caller"), flags)


def test_a_gap_without_a_declared_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="no declared lag budget"):
        declared_rows(_gapped(), None, "the E-step")


def test_the_refusal_names_the_caller_and_counts_the_gap() -> None:
    with pytest.raises(ValueError) as excinfo:
        declared_rows(_gapped(), None, "the M-step")
    assert "the M-step" in str(excinfo.value)
    assert "2 unobserved epochs" in str(excinfo.value)


def test_one_budget_gives_one_row_set_whatever_the_model_holds() -> None:
    """The row count is a property of the data and the budget alone."""
    flags = _gapped()
    rows = declared_rows(flags, 2, "the E-step")
    assert int(rows.sum()) == int(declared_rows(flags, 2, "the S-step").sum())
    # A larger budget erodes further, and that is the only thing that moves it.
    assert int(declared_rows(flags, 3, "the E-step").sum()) < int(rows.sum())


def test_an_estep_carries_its_budget_on_the_instance() -> None:
    """The fit sets one budget on the step, so `run` takes no per-call lag."""
    estep = EStep()
    assert estep._budget(1) is None
    estep.lag_budget = [2, 1]
    assert estep._budget(1) == 2
    assert estep._budget(2) == 1


def _sources(n_slots: int = 8, blocks: int = 4) -> FineSources:
    per = n_slots // blocks
    ancestors = [
        np.arange(n_slots) // 1,
        np.arange(n_slots) // per,
    ]
    return FineSources(
        counts=[np.zeros((n_slots, 2)), np.zeros((blocks, 2))],
        durations=[np.zeros((n_slots, 2)), np.zeros((blocks, 2))],
        calendar=[np.zeros((n_slots, 0)), np.zeros((blocks, 0))],
        ancestors=ancestors,
    )


def test_a_masked_fine_fit_without_a_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="declared lag budget"):
        fit_rows(_gapped(8, slice(2, 3)), None, _sources())


def test_an_unmasked_fine_fit_needs_no_budget() -> None:
    assert fit_rows(None, None, _sources()) is None


def test_an_edge_carrying_no_coefficient_does_not_move_the_fine_objective() -> None:
    """The search compares these two numbers, so a row reduction must not pay."""
    n_slots, blocks = 8, 4
    sources = _sources(n_slots, blocks)
    X1 = np.zeros((n_slots, 2), dtype=np.int64)
    Y1 = np.zeros((n_slots, 2), dtype=np.int64)
    X1[1, 0] = X1[5, 0] = 1
    Y1[1, 0] = Y1[5, 0] = 1
    observed = np.ones(n_slots, dtype=bool)
    observed[6] = False

    params = FineParams(n_labels=2)
    params.base[COUNT][:] = -1.5
    chains = [RegimeChain.uniform(scale=2, n_states=2, n_labels=2, n_epochs=blocks)]
    # The budget is the candidate space's, so both graphs are scored on it.
    sem = FineSEM(max_sweeps=1, lag_budget={2: 1})

    bare = TypedGraph(label_names=["target", "source"])
    with_edge = TypedGraph(label_names=["target", "source"])
    with_edge.add_edge(
        TypedEdge(
            src_kind=OBSERVED, src_scale=2, src_label=1, src_quantity=COUNT, lag=1
        ),
        0,
        COUNT,
    )

    plain = sem.objective(X1, Y1, params, bare, sources, chains, 0.0, observed)
    added = sem.objective(X1, Y1, params, with_edge, sources, chains, 0.0, observed)
    assert added == pytest.approx(plain)


def test_the_evidence_chain_sets_the_budget_on_the_step_it_runs() -> None:
    """The chain owns the budget, so a masked score is model-independent."""

    class _Recording(EStep):
        seen: list = []

        def run(self, k, data, params, graph, parent_summaries=None):
            _Recording.seen.append(self._budget(k))
            return _Stub()

    class _Stub:
        log_Z = 0.0

    class _Data:
        K = 2

    _Recording.seen = []
    evidence_chain(_Data(), [None, None], None, estep=_Recording(), lag_budget=[3, 2])
    assert _Recording.seen == [2, 3]
