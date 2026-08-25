"""
The starting point every compared method is given.

A structure learner that begins its search at the generating parent sets is
being told the answer, and greedy forward-backward keeps an edge it starts
with unless deleting it raises the score.  These tests pin the two things that
close that gap: the protocol hands every method the same empty graph and the
same cleared lag coefficients, and the enumerating search returns the same
graph from any starting set, so where it began cannot enter the result.

1.  The protocol clears the generating parent sets before any method sees them
2.  The protocol clears the generating lag coefficients as well
3.  The protocol hands every method a graph with no edges
4.  A method declaring `wants_oracle_start` receives the generating parent sets
5.  A method declaring `wants_oracle_start` receives the generating graph
6.  `OracleParams` and `OracleGraph` are the rows that declare it
7.  The enumerating search reaches the same parent set from either start
8.  The enumerating S-step returns the same graph from either start
9.  Greedy keeps a starting edge that enumeration declines, which is the gap
"""

import pytest

from src.evaluation.baselines import (
    OracleGraph,
    OracleParams,
    _clear_parent_sets,
    _empty_graph,
)
from src.evaluation.protocol import EvaluationProtocol
from src.learning.estep import EStep
from src.learning.evidence import evidence_chain
from src.learning.sstep import ExactSearch, GreedyForwardBackward, SStep
from src.learning.structural_em import TrainingHistory
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig, ScenarioGrid

_CFG = ScenarioConfig(
    name="cold_start_probe",
    K=2,
    H_finest=240,
    branching=[6],
    L=3,
    M_k=[2, 2],
    r_k=[0, 0],
    max_lag=[1, 1],
    max_indegree=[1, 1],
    family=2,
    seed=3,
)


#: The seed `EvaluationProtocol` derives for replicate 0, so the fixture and
#: the protocol read the same generated dataset.
_SEED = _CFG.seed * 1000


@pytest.fixture(scope="module")
def probe_ds():
    gen = make_generator(_CFG, seed=_SEED)
    return gen.generate(seed=_SEED)


class _Recorder:
    """Store the parameters and graph the protocol handed over, then stop."""

    def __init__(self, wants_oracle_start: bool = False) -> None:
        self.wants_oracle_start = wants_oracle_start
        self.estep = None
        self.couple_scales = True
        self.seen_parent_sets = None
        self.seen_lag_keys = None
        self.seen_edges = None

    def fit(self, data, params_list, graph):
        self.seen_parent_sets = [
            {ell: set(p.parent_sets.get(ell, set())) for ell in range(p.L)}
            for p in params_list
        ]
        self.seen_lag_keys = [
            set(p.coarse_count.betaLag if p.is_coarsest else p.fine_count.betaLag)
            for p in params_list
        ]
        self.seen_edges = sum(graph.n_edges(k) for k in range(1, graph.K + 1))
        history = TrainingHistory()
        chain = evidence_chain(data, params_list, graph)
        history.record(chain.summaries, graph)
        return params_list, graph, history


def _run_protocol(recorder):
    grid = ScenarioGrid(scenarios=[_CFG])
    protocol = EvaluationProtocol(
        methods={"probe": recorder}, n_replicates=1, verbose=0, n_jobs=1
    )
    protocol.run(grid)


# ---------------------------------------------------------------------------
# 1-3.  Every compared method starts from the same place
# ---------------------------------------------------------------------------


def test_the_protocol_clears_the_generating_parent_sets(probe_ds):
    """No method sees a parent set it did not propose itself."""
    assert any(v for p in probe_ds.params for v in p.parent_sets.values())
    rec = _Recorder()
    _run_protocol(rec)
    assert all(not s for per_scale in rec.seen_parent_sets for s in per_scale.values())


def test_the_protocol_clears_the_generating_lag_coefficients(probe_ds):
    """A cleared parent set with its coefficient left behind still scores."""
    rec = _Recorder()
    _run_protocol(rec)
    assert all(not keys for keys in rec.seen_lag_keys)


def test_the_protocol_hands_over_a_graph_with_no_edges():
    """The graph is a template for the search, carrying none of the answer."""
    rec = _Recorder()
    _run_protocol(rec)
    assert rec.seen_edges == 0


# ---------------------------------------------------------------------------
# 4-6.  A declared ceiling reads the truth, and says so
# ---------------------------------------------------------------------------


def test_a_declared_oracle_row_receives_the_generating_parent_sets(probe_ds):
    """The ceiling rows are the only ones the generating structure reaches."""
    rec = _Recorder(wants_oracle_start=True)
    _run_protocol(rec)
    truth = [
        {ell: set(p.parent_sets.get(ell, set())) for ell in range(p.L)}
        for p in probe_ds.params
    ]
    assert rec.seen_parent_sets == truth


def test_a_declared_oracle_row_receives_the_generating_graph(probe_ds):
    """Its graph carries one edge per generating parent."""
    rec = _Recorder(wants_oracle_start=True)
    _run_protocol(rec)
    assert rec.seen_edges == sum(
        len(v) for p in probe_ds.params for v in p.parent_sets.values()
    )


def test_the_ceilings_are_the_rows_that_declare_the_oracle_start():
    """A row reading the truth declares it, and no other row does."""
    assert OracleParams.wants_oracle_start is True
    assert OracleGraph.wants_oracle_start is True
    from src.evaluation import baselines
    from src.learning.structural_em import TopDownSEM

    declared = {
        name
        for name in dir(baselines)
        if getattr(getattr(baselines, name), "wants_oracle_start", False)
    }
    assert declared == {"OracleParams", "OracleGraph"}
    assert getattr(TopDownSEM, "wants_oracle_start", False) is False


# ---------------------------------------------------------------------------
# 7-9.  Enumeration cannot carry its starting set into the answer
# ---------------------------------------------------------------------------


def _decreasing_score(parent_set):
    """Every edge is a loss, so the only optimum is the empty set."""
    return -1.0 * len(parent_set)


def test_enumeration_reaches_the_same_set_from_either_start():
    """`ExactSearch.search` takes no starting set, so none can reach it."""
    import inspect

    assert "current_set" not in inspect.signature(ExactSearch.search).parameters
    cands = [(0, 1), (1, 1), (2, 1)]
    exact = ExactSearch(max_indegree=2)
    assert exact.search(_decreasing_score, cands) == set()


def test_greedy_keeps_a_starting_edge_that_enumeration_declines():
    """The gap the enumerating search closes, stated as a measurement."""
    cands = [(0, 1), (1, 1), (2, 1)]

    def plateau(parent_set):
        # Deleting the seeded edge is worth nothing, so greedy declines to.
        return 0.0 if set(parent_set) <= {(0, 1)} else -1.0

    seeded = {(0, 1)}
    greedy = GreedyForwardBackward(max_indegree=2)
    assert greedy.search(seeded, plateau, cands) == seeded
    assert greedy.search(set(), plateau, cands) == set()
    assert ExactSearch(max_indegree=2).search(plateau, cands) == set()


def test_the_enumerating_sstep_returns_the_same_graph_from_either_start(probe_ds):
    """Two S-steps differing only in their starting parent sets agree."""
    from copy import deepcopy

    from src.core.graph import MultiScaleDBNGraph

    data = probe_ds.data
    K = data.K
    truth = MultiScaleDBNGraph(
        hierarchy=probe_ds.hierarchy, label_names=[f"l{i}" for i in range(_CFG.L)]
    )
    for ki, p in enumerate(probe_ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                truth.add_edge(ki + 1, src, lag, dst)

    estep = EStep(max_inner_iter=2, use_laplace=False)
    sstep = SStep(
        lam=2.0,
        search="exact",
        max_lag={k: _CFG.max_lag[k - 1] for k in range(1, K + 1)},
        max_indegree={k: max(_CFG.max_indegree[k - 1], 1) for k in range(1, K + 1)},
        profile_steps=5,
    )

    edges = []
    for params_list, graph in (
        (deepcopy(probe_ds.params), deepcopy(truth)),
        (_clear_parent_sets(probe_ds.params), _empty_graph(truth)),
    ):
        q = estep.run(k=K, data=data, params=params_list[K - 1], graph=graph)
        out_graph, _ = sstep.run(
            k=K,
            q=q,
            params=params_list[K - 1],
            graph=graph,
            data=data,
            hier=probe_ds.hierarchy,
        )
        edges.append(
            sorted(
                (int(src), int(lag), int(dst))
                for dst in range(_CFG.L)
                for (src, lag) in out_graph.get_parents(K, dst)
            )
        )
    assert edges[0] == edges[1]
