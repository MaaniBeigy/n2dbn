# Contributing

For the manuscript revision, `src/nested/` implements the corrected normalized
top-down model and `experiments/revision.py` is its experiment entry point.
The frozen allocation modules remain historical; the new implementation does
not import them. Use `docker compose run --rm revision-tests` for focused
probability and pipeline checks. Historical coverage totals below do not
describe the new module. Full experiment execution is separate from tests;
see [RUN_REVISIONS.md](paper/WSDM/RUN_REVISIONS.md).

Development notes for the N²DBN implementation. The model, the benchmark and
the reported results are described in [README.md](README.md).

## Environment

Development runs through Docker Compose; the only local requirements are
Docker and the compose plugin.

```bash
docker compose build             # cpu and paper images
docker compose run --rm tests    # suite with coverage
docker compose run --rm lint     # isort and black, check only
docker compose run --rm format   # isort and black, in place
docker compose run --rm shell    # interactive, working tree mounted
```

The `tests`, `lint`, `format` and `shell` services mount the working tree over
the copy baked into the image, so an edit takes effect without a rebuild. The
experiment services deliberately run the baked copy, so rebuild before a
benchmark run.

What the images contain is defined by the requirements files.
`requirements.txt` holds what the model and the experiments import, including
pgmpy for nine of the eleven structure learners, pyAgrum for the graph layer,
and networkx-temporal plus teneto (with its `nilearn<0.13` pin) for the
temporal network figures. `requirements-dev.txt` adds black, isort,
pytest-xdist and torch; without torch the eight tests of
`test_phase18b_torch.py` skip and the torch solver goes unmeasured, which is
why the cpu image installs it.

## Test suite

```bash
docker compose run --rm tests                                    # everything
docker compose run --rm tests python -m pytest tests/test_phase22_fixes.py -q
```

```
815 passed, 3 skipped
```

| File                              | Tests | Covers                                                                                                                                                                        |
| --------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_phase01.py`               | 132   | hierarchy, reconciliation invariant, generator, families, scenario grids, binary finest scale                                                                                 |
| `test_phase02.py`               | 24    | ZINB and NB log-likelihoods, allocation log-likelihoods                                                                                                                       |
| `test_phase03.py`               | 28    | forward-backward, Kalman and RTS smoother, episode reconstruction, graph                                                                                                      |
| `test_phase04_05.py`            | 26    | the five parameter-learning blocks, local score, greedy and beam search                                                                                                       |
| `test_phase06.py`               | 14    | outer loop, structure learning, resume from checkpoint                                                                                                                        |
| `test_phase07.py`               | 18    | metrics, all baselines, one protocol run                                                                                                                                      |
| `test_phase08.py`               | 13    | structured logging, plots                                                                                                                                                     |
| `test_phase09_coverage.py`      | 12    | warm start, cross-scale coupling, graph utilities                                                                                                                             |
| `test_phase10_edge_cases.py`    | 7     | factor correlation, greedy regime matching for M > 8, lagged features                                                                                                         |
| `test_phase11_vectorization.py` | 17    | segment reductions, vectorized E-step against the per-epoch reference, Dirichlet-Multinomial likelihood and gradients, structure-learning edge addition, relative convergence |
| `test_phase12_coverage.py`      | 37    | calendar and event log, concentration estimation, monotone parameter blocks, redistribution of excess counts                                                                  |
| `test_phase13_utils.py`         | 17    | logging back ends, every plot, protocol serialization                                                                                                                         |
| `test_phase14_branches.py`      | 18    | per-scale search settings, split guards, cross-scale warm start, penalty modes                                                                                                |
| `test_phase15_final.py`         | 15    | numerical helpers, singular-matrix fallbacks, degenerate configurations                                                                                                       |
| `test_phase16_gaps.py`          | 34    | calibrated penalty, optional-dependency guards, reference lag aggregators, parallel and partial protocol runs, family IV dispatch                                             |
| `test_learning_fixes.py`        | 5     | lag-coefficient updates, cross-scale plumbing,`EmptyGraph` isolation                                                                                                        |
| `test_pyagrum_integration.py`   | 20    | graph conversion, CPT export, scoring backend                                                                                                                                 |
| `test_phase17_resources.py`     | 25    | memory and GPU detection with every fallback, memory-bounded worker counts, thread pinning                                                                                    |
| `test_phase18_classical.py`     | 64    | the lagged design and the encodings, exact recovery by the consistent criteria, search-space and in-degree budgets for all ten entries, the optional-dependency paths          |
| `test_phase18b_torch.py`        | 8     | the torch solver against the numpy one, on the host and on a device                                                                                                           |
| `test_phase19_param_estimation.py` | 20 | the four regime estimators, the mapping onto the parameter block, the three parameter-step modes                                                                              |
| `test_phase20_optional_dependencies.py` | 5 | the import guards for matplotlib and pyAgrum, the beam at a zero in-degree bound, a parent block with no children                                                        |
| `test_phase21_classical_baseline.py` | 16 | the search space reaching the learner, lag-coefficient seeding, the full baseline interface, the experiment script                                                        |
| `test_phase22_fixes.py`         | 20    | the monotone outer guard with backtracking and best-state return, the per-edge shift confirmation with its block-preserving null, the stored learned edges and guard count |
| `test_phase23_temporal_viz.py`  | 16    | edge occurrences, the timeline, snapshot and slice figures, the stored-run orchestrator, the optional-dependency guards for matplotlib, networkx-temporal and teneto       |
| `test_phase24_depletion.py`     | 8     | the block-exclusive running total, the depletion coefficients of the allocation score, the gradient that learns them                                                      |
| `test_phase25_parameter_metrics.py` | 53 | parameter recovery: the directions the data cannot identify score zero, the regime permutation, the factor rotation, transition CPT distance, `edge_auprc` including ties, varsortability and its R-squared counterpart |
| `test_phase26_event_log.py`     | 27    | the exact round trip from a generated event log back to `MultiScaleData`, reconciliation, every raise, padding to whole coarsest epochs                                   |
| `test_phase27_initialize.py`    | 20    | data-driven initialization: shapes against the generator, an end-to-end fit from the result, centered regime effects, no NaN under a degenerate column                     |
| `test_phase28_bnlearn.py`       | 29    | the tiled bnlearn benchmark: every vendored network loads, tiling direction and lag, deterministic sampling, structure and CPT scoring, and the predictive table score that lets learners with different parent sets be compared |
| `test_phase29_search.py`        | 15    | exact and evolutionary parent-set search: both reach a pair of edges that only pays jointly, both respect the in-degree bound, the budget rule that chooses between them |
| `test_phase30_model_selection.py` | 10  | held-out selection of the latent orders and the edge penalty: the argmin is what is returned, one regime is chosen on regime-free data, a candidate that cannot be fitted does not stop the sweep |
| `test_phase31_sequential_allocation.py` | 10 | the sequential in-block draw: the block total is preserved, each child is written before the next is scored, a true self edge leaves a lagged signal in the generated data, reconciliation and the binary cap survive |

`test_finest_scale_binary` asserts strictly, where it once was an expected
failure.
`ScenarioConfig.finest_binary` caps a finest-scale epoch at the number of
wall-clock slots it holds and redistributes any excess to siblings with room
left, so $X^{(1,\ell)}_t \in \{0,1\}$ holds exactly without changing a block
total.

### Coverage

```bash
docker compose run --rm tests
```

The `tests` service runs the suite with coverage enabled. Coverage reads 100%
of all 4999 statements, and every module reads 100%. Keep it there. The two
techniques that get the last statements are worth knowing before adding code
that needs them.

An `except ImportError` guard is reached by blocking the package in
`builtins.__import__`, removing it from `sys.modules`, reloading the module, and
reloading it back in a `finally`, which is what
`test_phase20_optional_dependencies.py` does. Blocking the import alone is not
enough, because a package already in `sys.modules` never reaches the import hook.

A fallback that depends on the machine is reached by monkeypatching the thing it
asks.
`test_phase17_resources.py` covers every branch of the memory and GPU detection
by patching `_read_meminfo`, `os.sysconf`, `shutil.which` and `subprocess.run`.

The torch solver needs torch, which is why `requirements-dev.txt` lists it.
Without torch, `test_phase18b_torch.py` skips itself and its statements go
unmeasured. A CUDA device is not needed: two tests of that file skip without
one, and the branch that routes the solver onto torch is reached through
`_use_torch` on the cpu device, so the 100% figure holds on a machine with no
card. It is checked both ways here, on the host with an H200 and inside the cpu
container without one.

## Formatting

```bash
docker compose run --rm format   # isort and black, in place
docker compose run --rm lint     # the same two, check only
```

Settings live in `pyproject.toml`: black at its default line length of 88, isort
on the black profile. Run `format` before sending a change, not as a separate
pass afterwards.

## The allocation formulation is frozen

Six modules carry the formulation in which a coarse count is drawn and then
split downward among its children: `src/model/allocation.py` and the five under
`src/synthetic/`. Every stored result came from them and they keep reproducing
it, so they stay importable, keep their tests and keep their coverage. They are
not developed further and nothing new is built on them.

`tests/test_frozen_allocation_path.py` enforces that. It holds the frozen set
and the closed list of modules under `src/` that are allowed to import one, and
it fails naming the file and the line when a module outside that list does. A
new entry in `PERMITTED_IMPORTERS` is a new dependency on a path with no future,
so adding one is a decision. It is never a way to quiet a failing test.

## Invariants a change must preserve

1. **Exact reconciliation.** `MultiScaleData.check_reconciliation` returns
   `RecErr_X = RecErr_Y = 0.0` on generated data. Any change to the generator or
   the allocation model has to keep this exact, not approximate. The capacity
   cap of a binary finest scale redistributes counts between siblings and never
   changes a block total, which is what keeps this true.
2. **Vectorized code matches the reference loops.** Anything rewritten as array
   operations is checked in `test_phase11_vectorization.py` against a per-epoch
   implementation of the same mathematics, to a tolerance of 1e-14.
3. **Analytic gradients match finite differences.** New gradients get a central
   difference check, as `test_dirmult_grad_matches_finite_difference` does.
4. **A structure score has to be able to add an edge.** A local score evaluated
   at the incumbent coefficient of a candidate edge scores every addition at
   exactly minus the penalty, so the search can only delete.
   `test_sstep_can_add_edge_from_empty_graph` guards against reintroducing that.
5. **The finer-scale likelihood stays normalized.** An allocation score without
   its block normalizer is unbounded above, which makes the objective, the
   validation loss and every finer-scale structure score meaningless.
   `test_fine_log_like_is_bounded_under_scaling` guards this.
6. **A parameter block does not lower its own objective.** `MStep` evaluates
   `coarse_emission_ll`, `allocation_ll` or `regime_ll` before and after each
   block and keeps the previous values when the block would lower it. A new
   block needs its own objective function and the same check.
7. **A finest-scale slot starts at most one episode of a label** when a scenario
   sets `finest_binary`, which `test_finest_scale_binary` asserts over the whole
   sanity grid.
8. **A graph installed from outside the structure step carries coefficients.**
   The parameter step updates the coefficient of a lagged parent only when a key
   for it already exists in `betaLag` / `alphaLag`, so a baseline that sets
   `parent_sets` and nothing else is scored exactly as the empty graph.
   `baselines._seed_lag_coefficients` creates them, and
   `test_a_learned_edge_changes_the_held_out_objective` guards it.
9. **A parameter score ignores what the data cannot identify.** The allocation
   is a softmax over the children of a block, so a per-label constant added to
   the intercept, to the parent-count coefficient or to the regime effect
   cancels exactly and changes no distribution. A new block of
   `ParameterMetrics` scores the identified quantity, and
   `test_phase25_parameter_metrics.py` proves it by regenerating the data at
   the same seed and requiring array equality before asserting the score is
   zero. A metric that charges for an unidentified direction ranks a worse fit
   above a better one.
10. **A structure learner stays inside the declared space.** Whatever an
   algorithm returns, the parent sets it produces have to sit inside the given
   maximum lag and maximum in-degree, which
   `test_every_algorithm_respects_the_indegree_budget` asserts for every entry of
   `ALGORITHMS`.
11. **The recorded objective trace is monotone.** A sweep that ends below the
    best sweep so far is rejected, retried at a damped learning rate, and the
    fit returns its best state, so `history.elbo` never decreases beyond
    numerical noise and rejected objectives land in `history.rejected`
    instead. `test_the_recorded_trace_is_monotone_on_data` asserts it on data
    and the guard tests of `test_phase22_fixes.py` pin each branch.
12. **Edge confirmation only removes edges.** With `confirm_edges=True` the
    parent sets are a subset of what the plain search returns, never a
    superset, which `test_sstep_confirmation_returns_a_subset_of_the_plain_search`
    asserts.

## Running an experiment during development

Ask `docker compose run --rm cpu-info` how many workers the machine holds
before picking `JOBS`. Every worker keeps its own copy of a 20160-epoch
dataset, and asking for one per core on a machine with little memory per core
breaks the pool with an `ImportError` inside a worker that reads like a code
fault. Worker count changes wall-clock time and nothing else.

The images set `PYTHONHASHSEED=0`, which any run that includes `ges`, `mmhc`
or `tree` needs. Those three take no edge whitelist, so their lagged edges
come from orienting an undirected skeleton, and pgmpy settles that by
iterating over sets of variable names, whose hashes Python randomizes per
process.

```bash
# sanity grid, a few minutes
docker compose run --rm shell python -m experiments.exp1_ideal_recovery \
    --quick --iters 5 --jobs 7 --output-dir results/exp1_quick

# one scenario only
docker compose run --rm shell python - <<'EOF'
from src.synthetic.scenarios import ScenarioGrid
from src.learning.structural_em import TopDownSEM
from src.evaluation.protocol import EvaluationProtocol

full = ScenarioGrid.full_benchmark()
grid = ScenarioGrid(scenarios=[s for s in full.scenarios if s.name == "F2_L5_M2_zil"])
protocol = EvaluationProtocol(
    methods={"n2dbn": TopDownSEM(max_outer_iter=5, verbose=0)},
    n_replicates=1, verbose=2,
)
print(protocol.summary_table(protocol.run(grid), metric_keys=["F1_mean", "AccS_k1"]))
EOF
```

Before running a study on a new scenario, `ScenarioGrid.identifiability_probe`
reports whether the scenario contains what the metrics claim to measure. A
scenario that fails the probe produces numbers that say nothing about the
estimator.

## Profiling

```bash
docker compose run --rm shell sh -c \
    "python -m cProfile -s cumtime -m experiments.exp1_ideal_recovery \
     --quick --iters 1 2>&1 | head -40"
```

The structure-learning step dominates a profile, and within it the profiling of
candidate lag coefficients. Reducing `profile_steps` on `SStep` trades score
accuracy for speed.

## Where things live

| Change                                   | File                                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------- |
| a new emission family                    | `src/model/emissions.py`, then `estep.coarse_ll_tml`                                    |
| a new allocation model                   | `src/model/allocation.py`, `estep.dirmult_ll`, `estep.dirmult_grad`                   |
| a new parameter block                    | `src/learning/mstep.py`, `MStep.run`, plus a block objective for the monotonicity check |
| a new structure score                    | `src/learning/sstep.py`, as a scorer class with `score(parent_set)`                     |
| a new edge penalty                       | `SStep._lam_effective` and, when it needs a null distribution, `sstep._null_threshold`  |
| a new baseline                           | `src/evaluation/baselines.py`, matching the `fit` and `predict` interface             |
| a new metric                             | `src/evaluation/metrics.py` and `aggregate_metrics`                                     |
| a new scenario knob                      | `src/synthetic/scenarios.py`, then thread it through `src/synthetic/families.py`        |
| a new block reduction over parent epochs | `src/core/blocks.py`, as a method of `BlockIndex`                                       |
| anything with a wall-clock meaning       | `src/core/calendar.py`, which owns the origin, the slot length and the event log          |
| a published structure learner            | `src/learning/classical.py`, as an entry in `ALGORITHMS` returning parent sets            |
| a regime parameter estimator             | `src/learning/param_estimation.py`, then a name in `REGIME_ESTIMATORS`                    |
| anything about cores, memory or GPUs     | `src/utils/resources.py`                                                                    |
| a temporal network figure                | `src/utils/temporal_viz.py`, wired into the scripts through `experiment_plots`              |
| a parameter-recovery score               | `src/evaluation/metrics.py`, as a block of `ParameterMetrics`, scoring the identified quantity |
| a reader for observed data               | `src/data/`, returning a `MultiScaleData` whose reconciliation is exact by construction     |
| an external benchmark with a released graph | `src/benchmarks/`, as a module exposing the true edges and a sampler                     |
| a rule for choosing a free quantity from data | `src/learning/model_selection.py`, scored by held-out log evidence and never by a metric against the truth |
| a parent-set search                      | `src/learning/sstep.py`, as a class with `search(...)` returning a parent set, then a branch in `SStep.run` |
| a container service                      | `docker-compose.yml`, and the image it needs in `docker/`                                   |
