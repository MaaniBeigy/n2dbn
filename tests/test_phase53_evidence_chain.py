"""
tests/test_phase53_evidence_chain.py
====================================
The one coarse-to-fine sweep every score is now taken from.

The sweep itself is arithmetic that six copies already agreed on, so the
assertions that matter here are about the two settings the copies stated as
literals.  A model fitted with the cross-scale coupling switched off was scored
with it switched on, and an estimator given `estep_kwargs` was scored by a
default E-step, so two of the numbers the protocol reported described models
nobody had fitted.  Both are pinned below, together with the arithmetic they
sit on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.data.initialize import initialize_from_data
from src.evaluation.baselines import (
    EmptyGraph,
    OracleGraph,
    SingleScaleDBN,
    _clear_parent_sets,
    _scoring_settings,
)
from src.evaluation.metrics import LatentMetrics
from src.evaluation.protocol import EvaluationProtocol
from src.learning import model_selection
from src.learning.estep import EStep
from src.learning.evidence import ChainEvidence, evidence_chain, scoring_settings
from src.learning.model_selection import held_out_nll, held_out_nll_per_scale
from src.learning.structural_em import TopDownSEM, TrainingHistory
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig, ScenarioGrid


def _flat(T=240, L=3, b=8, seed=0):
    """Two scales of counts with no persistent regime, cheap to score."""
    rng = np.random.default_rng(seed)
    n = (T // b) * b
    Xf = rng.poisson(2.0, size=(n, L)).astype(np.int64)
    coarse = Xf.reshape(n // b, b, L).sum(axis=1)
    return MultiScaleData(
        hierarchy=ScaleHierarchy.from_uniform_branching(n, [b]),
        label_names=[f"v{i}" for i in range(L)],
        X=[Xf, coarse],
        Y=[Xf.copy(), coarse.copy()],
        C=[np.zeros((n, 0)), np.zeros((n // b, 0))],
    )


def _one_scale(T=96, L=3, seed=1):
    """A model with a single scale, which has no parent to carry anywhere."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(T, L)).astype(np.int64)
    return MultiScaleData(
        hierarchy=ScaleHierarchy.from_uniform_branching(T, []),
        label_names=[f"v{i}" for i in range(L)],
        X=[X],
        Y=[X.copy()],
        C=[np.zeros((T, 0))],
    )


def _fitted(data, r=0, cross=None):
    """Parameters and an edge-free graph for `data`, index 0 is scale 1."""
    K = data.K
    params = initialize_from_data(
        data, M_k=[2] * K, r_k=[r] * K, max_lag=[1] * K, d_ctx=0, seed=0
    )
    if cross is not None:
        params[0].regime.cross = np.asarray(cross, dtype=float)
    graph = MultiScaleDBNGraph(hierarchy=data.hierarchy, label_names=data.label_names)
    return params, graph


@pytest.fixture(scope="module")
def coupled_world():
    """Two scales whose cross-scale block is strong enough to change the score."""
    data = _flat()
    params, graph = _fitted(data, cross=[[2.0, -2.0], [-2.0, 2.0]])
    return data, params, graph


@pytest.fixture(scope="module")
def factor_world():
    """Two scales carrying a latent factor, so the inner E-step loop bites."""
    data = _flat()
    params, graph = _fitted(data, r=1)
    return data, params, graph


def _by_hand(data, params_list, graph, couple, estep=None):
    """The loop the four call sites used to write out, run once here."""
    run = estep if estep is not None else EStep()
    summaries = [None] * data.K
    for k in range(data.K, 0, -1):
        parent = summaries[k] if (k < data.K and couple) else None
        summaries[k - 1] = run.run(k, data, params_list[k - 1], graph, parent)
    return [float(s.log_Z) for s in summaries]


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def test_the_chain_reproduces_the_hand_written_sweep(coupled_world):
    """Coarsest first, each posterior carried down as the next scale's parent."""
    data, params, graph = coupled_world
    chain = evidence_chain(data, params, graph)
    assert chain.log_Z == pytest.approx(_by_hand(data, params, graph, couple=True))
    assert len(chain.summaries) == data.K


def test_breaking_the_coupling_withholds_the_parent_posterior(coupled_world):
    """No parent summary reaches a finer scale, and the score says so."""
    data, params, graph = coupled_world
    chain = evidence_chain(data, params, graph, couple_scales=False)
    assert chain.log_Z == pytest.approx(_by_hand(data, params, graph, couple=False))
    assert chain.log_Z != pytest.approx(evidence_chain(data, params, graph).log_Z)


def test_the_total_is_the_sum_of_the_breakdown(coupled_world):
    """One number for the model, derived from the per-scale terms."""
    data, params, graph = coupled_world
    chain = evidence_chain(data, params, graph)
    assert chain.total == pytest.approx(sum(chain.log_Z))
    assert isinstance(chain, ChainEvidence)


def test_a_single_scale_model_needs_no_coupling_case():
    """With one scale there is no parent, so both settings give one answer."""
    data = _one_scale()
    params, graph = _fitted(data)
    coupled = evidence_chain(data, params, graph)
    broken = evidence_chain(data, params, graph, couple_scales=False)
    assert len(coupled.log_Z) == 1
    assert coupled.log_Z == pytest.approx(broken.log_Z)


def test_the_supplied_estep_is_the_one_that_runs(factor_world):
    """A configured E-step reaches every scale, and it changes the answer."""
    data, params, graph = factor_world
    estep = EStep(max_inner_iter=1)
    chain = evidence_chain(data, params, graph, estep=estep)
    assert chain.log_Z == pytest.approx(
        _by_hand(data, params, graph, couple=True, estep=estep)
    )
    assert chain.log_Z != pytest.approx(evidence_chain(data, params, graph).log_Z)


# ---------------------------------------------------------------------------
# Reading the two settings off an estimator
# ---------------------------------------------------------------------------


def test_the_settings_come_off_the_estimator():
    """TopDownSEM already answers both questions about itself."""
    sem = TopDownSEM(
        max_outer_iter=1,
        couple_scales=False,
        estep_kwargs=dict(max_inner_iter=2),
        verbose=0,
    )
    estep, couple_scales = scoring_settings(sem)
    assert estep is sem.estep
    assert couple_scales is False


def test_an_estimator_declaring_neither_falls_back_to_the_defaults():
    """The fallback is what TopDownSEM itself is constructed with."""
    estep, couple_scales = scoring_settings(object())
    assert estep is None
    assert couple_scales is True


def test_a_wrapper_derives_the_settings_from_the_kwargs_it_forwards():
    """The keyword arguments a wrapper hands TopDownSEM decide how it is scored."""
    estep, couple_scales = _scoring_settings(
        dict(couple_scales=False, estep_kwargs=dict(max_inner_iter=3, tol=1e-2))
    )
    assert (estep.max_inner_iter, estep.tol) == (3, 1e-2)
    assert couple_scales is False

    plain_estep, plain_couple = _scoring_settings({})
    assert (plain_estep.max_inner_iter, plain_estep.tol) == (10, 1e-4)
    assert plain_couple is True


def test_every_wrapper_reports_the_coupling_it_fits_under():
    """SingleScaleDBN is the one baseline that switches the coupling off."""
    assert SingleScaleDBN(max_outer_iter=1, verbose=0).couple_scales is False
    assert EmptyGraph(max_outer_iter=1, verbose=0).couple_scales is True


# ---------------------------------------------------------------------------
# The two regressions
# ---------------------------------------------------------------------------


def test_single_scale_dbn_is_scored_without_the_parent_posterior(coupled_world):
    """The baseline fitted with no coupling is scored with no coupling."""
    data, params, graph = coupled_world
    method = SingleScaleDBN(max_outer_iter=1, verbose=0)
    got = [float(s.log_Z) for s in method.predict(data, params, graph)]
    assert got == pytest.approx(
        evidence_chain(data, params, graph, couple_scales=False).log_Z
    )
    assert got != pytest.approx(evidence_chain(data, params, graph).log_Z)


def test_a_wrapper_is_scored_by_the_estep_it_was_given(factor_world):
    """estep_kwargs reach the score, so the fit and the score run one E-step."""
    data, params, graph = factor_world
    method = OracleGraph(
        true_graph=graph,
        max_outer_iter=1,
        estep_kwargs=dict(max_inner_iter=1),
        verbose=0,
    )
    got = [float(s.log_Z) for s in method.predict(data, params, graph)]
    assert got == pytest.approx(
        evidence_chain(data, params, graph, estep=EStep(max_inner_iter=1)).log_Z
    )
    assert got != pytest.approx(evidence_chain(data, params, graph).log_Z)


# ---------------------------------------------------------------------------
# The call sites
# ---------------------------------------------------------------------------


def test_the_held_out_score_is_minus_the_chain(coupled_world):
    """Per scale and in total, with the coupling setting carried through."""
    data, params, graph = coupled_world
    chain = evidence_chain(data, params, graph)
    assert held_out_nll_per_scale(data, params, graph) == pytest.approx(
        [-z for z in chain.log_Z]
    )
    assert held_out_nll(data, params, graph) == pytest.approx(-chain.total)

    broken = evidence_chain(data, params, graph, couple_scales=False)
    assert held_out_nll(data, params, graph, couple_scales=False) == pytest.approx(
        -broken.total
    )


def test_selection_scores_a_candidate_under_its_own_estimator(monkeypatch):
    """The settings the candidate was fitted with are the ones that score it."""
    seen = []

    def fake_nll(data, params_list, graph, *, estep=None, couple_scales=True):
        seen.append((estep, couple_scales))
        return 1.0

    monkeypatch.setattr(model_selection, "held_out_nll", fake_nll)
    flat = _flat()
    sem = TopDownSEM(
        max_outer_iter=1,
        couple_scales=False,
        estep_kwargs=dict(max_inner_iter=1),
        sstep_kwargs=dict(max_lag={1: 1, 2: 1}, max_indegree={1: 1, 2: 1}),
        verbose=0,
    )
    model_selection.select_model(
        flat, flat, lambda lam: sem, M_grid=(1,), r_grid=(0,), max_lag=[1, 1], seed=0
    )
    assert seen == [(sem.estep, False)]


class _NoOpFit:
    """An estimator that fits nothing and declares the two settings itself."""

    def __init__(self, estep: EStep, couple_scales: bool) -> None:
        self.estep = estep
        self.couple_scales = couple_scales

    def fit(self, data, params_list, graph):
        return params_list, graph, TrainingHistory()


def test_the_protocol_scores_a_method_the_way_it_was_fitted():
    """The reported val_loss belongs to the model the method actually fitted."""
    cfg = next(s for s in ScenarioGrid.sanity_check() if s.name == "sanity_L3_M2_r0_F2")
    method = _NoOpFit(EStep(), couple_scales=False)
    protocol = EvaluationProtocol(
        methods={"single_scale": method},
        n_replicates=1,
        verbose=0,
        configure_search_space=False,
    )
    (result,) = protocol._run_replicate(cfg, 0)

    gen = make_generator(cfg, seed=cfg.seed * 1000)
    ds = gen.generate(seed=cfg.seed * 1000)
    _, valid, _ = ds.data.split(fracs=(0.7, 0.15, 0.15))
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    params = _clear_parent_sets(ds.params)

    uncoupled = held_out_nll(valid, params, graph, couple_scales=False)
    coupled = held_out_nll(valid, params, graph)
    assert result.metrics["val_loss"] == pytest.approx(uncoupled)
    assert result.metrics["val_loss"] != pytest.approx(coupled)


# ---------------------------------------------------------------------------
# Every number the protocol reports off the sweep
# ---------------------------------------------------------------------------

# The E-step the estimator below declares.  The Laplace pseudo-observation is
# what ties the coarsest-scale factor to the counts, so switching it off leaves
# that factor on its prior path and the factor correlation there moves far
# enough to name the E-step that produced it.
_DECLARED_ESTEP = EStep(use_laplace=False)


def _latent_scores(data, params_list, graph, true_latents, *, estep, couple_scales):
    """The latent block of the protocol's metrics, taken from one sweep."""
    summaries = evidence_chain(
        data, params_list, graph, estep=estep, couple_scales=couple_scales
    ).summaries
    true_S = []
    true_H = []
    for k in range(1, data.K + 1):
        T_k = data.get_X(k).shape[0]
        true_S.append(true_latents.get_S(k)[-T_k:])
        true_H.append(true_latents.get_H(k)[-T_k:])
    return LatentMetrics.compute(
        [s.gamma for s in summaries], true_S, [s.m_H for s in summaries], true_H
    ).to_dict()


@pytest.fixture(scope="module")
def declared_scenario():
    """A scenario carrying both latents, since families I to III pin one to zero."""
    # Family IV keeps the configured M_k and r_k, and with no misspecification
    # type it generates from the model itself.  Both latents have to be present.
    # The E-step setting is read off the factor, and the coupling is read off a
    # regime posterior traveling down to the finer scale.
    return ScenarioConfig(
        name="declared_settings",
        K=2,
        H_finest=240,
        branching=[8],
        L=3,
        M_k=[2, 2],
        r_k=[1, 1],
        max_lag=[1, 1],
        max_indegree=[1, 1],
        family=4,
        seed=0,
        n_replicates=1,
        edge_strength=10.0,
        regime_strength=3.0,
        finest_rate=0.15,
        finest_binary=True,
    )


@pytest.fixture(scope="module")
def declared_world(declared_scenario):
    """One replicate's validation and test halves, its true parameters and graph."""
    cfg = declared_scenario
    gen = make_generator(cfg, seed=cfg.seed * 1000)
    ds = gen.generate(seed=cfg.seed * 1000)
    _, valid, test = ds.data.split(fracs=(0.7, 0.15, 0.15))
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    return valid, test, _clear_parent_sets(ds.params), graph, ds.latents


@pytest.fixture(scope="module")
def declared_metrics(declared_scenario):
    """What the protocol reports for a method declaring neither default."""
    protocol = EvaluationProtocol(
        methods={"declared": _NoOpFit(_DECLARED_ESTEP, couple_scales=False)},
        n_replicates=1,
        verbose=0,
        configure_search_space=False,
    )
    (result,) = protocol._run_replicate(declared_scenario, 0)
    return result.metrics


def test_the_reported_latents_are_inferred_by_the_methods_own_estep(
    declared_world, declared_metrics
):
    """The coarsest-scale factor score belongs to the E-step the method fitted with."""
    _valid, test, params, graph, latents = declared_world
    own = _latent_scores(
        test, params, graph, latents, estep=_DECLARED_ESTEP, couple_scales=False
    )
    default = _latent_scores(
        test, params, graph, latents, estep=None, couple_scales=False
    )
    assert declared_metrics["FactCorr_k2"] == pytest.approx(own["FactCorr_k2"])
    assert own["FactCorr_k2"] != pytest.approx(default["FactCorr_k2"])


def test_the_reported_latents_are_inferred_under_the_coupling_it_fitted_under(
    declared_world, declared_metrics
):
    """No parent posterior reaches the finer scale, and its factor score says so."""
    _valid, test, params, graph, latents = declared_world
    broken = _latent_scores(
        test, params, graph, latents, estep=_DECLARED_ESTEP, couple_scales=False
    )
    coupled = _latent_scores(
        test, params, graph, latents, estep=_DECLARED_ESTEP, couple_scales=True
    )
    assert declared_metrics["FactCorr_k1"] == pytest.approx(broken["FactCorr_k1"])
    assert broken["FactCorr_k1"] != pytest.approx(coupled["FactCorr_k1"])


def test_the_reported_val_loss_is_taken_by_the_methods_own_estep(
    declared_world, declared_metrics
):
    """The held-out score runs the E-step the method fitted with."""
    valid, _test, params, graph, _latents = declared_world
    own = held_out_nll(valid, params, graph, estep=_DECLARED_ESTEP, couple_scales=False)
    default = held_out_nll(valid, params, graph, couple_scales=False)
    assert declared_metrics["val_loss"] == pytest.approx(own)
    assert own != pytest.approx(default)
