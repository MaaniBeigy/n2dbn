# N²DBN: Nested Non-Stationary Dynamic Bayesian Networks

**Manuscript revision:** use the corrected top-down count/duration model in
`src/nested/` and the `revision-*` Docker Compose services. The complete commands,
audits and claim report are documented in
[RUN_REVISIONS.md](paper/WSDM/RUN_REVISIONS.md). Full revised experiments have not
yet been run. The descriptions and `exp*` commands below document historical
implementations and outputs, including the former LifeSnaps intensity model.

Behavioral event logs read differently at every timescale, and a dynamic Bayesian network is defined at just one of them. N²DBN writes a single likelihood over episode starts and durations at the finest resolution, so every count and duration above it is a sum of that one and the scales always agree. Trend, seasonality and lagged dependence share one predictor, each priced at the scale that identifies it.

Reference implementation, benchmarks and evaluation protocol for [paper/WSDM/sigconf.pdf](paper/WSDM/sigconf.pdf).

## Quick start

Everything runs through Docker Compose, so Docker and the compose plugin are the only local requirements.

```bash
docker compose build tests        # n2dbn:cpu
docker compose run --rm cpu-info   # what this machine offers
docker compose run --rm tests      # the suite, with coverage
docker compose run --rm exp-quick  # every budgeted experiment on its quick grid
```

## The experiments the paper reports

Each command is self-contained and writes to `results/<exp>/`.

| Paper          | Measures                                                                                                 | Command                                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| §5.1, Table 1 | Structure recovery on the simulated grid, N²DBN against eight published learners over 540 fits          | `docker compose run --rm exp5`                                                                                       |
| §5.1          | The three sorting baselines, which test whether a variance ordering wins the grid                        | `ALGORITHMS=var-sortnregress,r2-sortnregress,random-sortnregress OUT=exp5_sortnregress docker compose run --rm exp5` |
| §5.2          | Fidelity of a drawn log to the patterns the benchmark declares before any fit                            | `REPLICATES=64 docker compose run --rm exp12`                                                                        |
| §5.3          | What fitting one granularity at a time costs in coherence, nested against per-granularity                | `docker compose run --rm exp17`                                                                                      |
| §5.4, Table 3 | Eight published structure learners on the declared benchmark                                             | `docker compose run --rm exp13`                                                                                      |
| §5.4, Table 3 | The N²DBN row of the same table, with the roster swept from 1 to 30 participants                        | `docker compose run --rm exp11`                                                                                      |
| §5.5, Table 2 | Forecasting two held-out days of QuitoBench traffic, with the coherence column                           | `docker compose run --rm exp19`                                                                                      |
| §6.1          | The LifeSnaps data path: the collapse, the ladder on participant time, the gap audit, the frozen anchors | `docker compose run --rm exp14`                                                                                      |
| §6.2          | The LifeSnaps fit, the falsification battery and the three-way calendar ablation                         | `docker compose run --rm exp15`                                                                                      |
| §6.3, §6.4   | The fit-then-simulate round trip: episode volume, the non-stationarity grid, the changepoint agreement   | `docker compose run --rm exp16`                                                                                      |

Prefix any experiment with `JOBS=7` to fit seven (scenario, replicate) pairs at once.

## Supporting experiments

These are not cited in the submission and are kept for the record.

| Measures                                                            | Command                                                                                    |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Ideal structural and latent recovery on families I to III           | `docker compose run --rm exp1`                                                           |
| Hierarchy benefit against a single-scale DBN                        | `docker compose run --rm exp2`                                                           |
| Ablation: remove the factor, the regime or the graph                | `docker compose run --rm exp3`                                                           |
| Behavior under misspecification                                     | `docker compose run --rm exp4`                                                           |
| Fit an observed event log CSV                                       | `CSV=data/dynamic_patterns/event_log.tsv BRANCHING=60,24,7 docker compose run --rm exp6` |
| Parameter recovery, on a sample-size ladder                         | `docker compose run --rm exp7`                                                           |
| External ground truth: bnlearn networks tiled into 2TBNs            | `docker compose run --rm exp8`                                                           |
| What limits recovery: block length, penalty, stability, sortability | `docker compose run --rm exp9`                                                           |
| The sufficiency ledger, judged before anything is fitted            | `docker compose run --rm exp10`                                                          |

## Data

| Path                                      | Holds                                                                                                           | Shipped                    |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `data/dynamic_patterns/`                | the declared benchmark: 30 simulated participants over 12 weeks, 60 labels, with its manifest                   | yes                        |
| `data/observed_lifesnaps_event_log.csv` | the observed log derived from the LifeSnaps MongoDB dump                                                        | no, rebuild it (below)     |
| `data/quito/`                           | QuitoBench `test_min.parquet` and `test_hour.parquet` ([arXiv:2603.26017](https://arxiv.org/abs/2603.26017)) | no, fetch before `exp19` |

## Rebuilding the LifeSnaps log

The observed log is derived from the published LifeSnaps dump and is not redistributed here. Two services rebuild it, and the restore needs about 12 GiB of disk and runs once.

```bash
docker compose run --rm lifesnaps-restore   # fetch the dump, load it into mongo
docker compose run --rm lifesnaps-build     # write data/observed_lifesnaps_event_log.csv
```

`src/lifesnaps/event_log.py` holds the derivation and `src/lifesnaps/roster.txt` the 60 participants it keeps, in the order that numbers them.

## Building the paper

```bash
docker compose run --rm paper        # paper/WSDM/sigconf.pdf, the submission
```

## Notes on running

- Every `JOBS` worker holds its own copy of the dataset, so memory is the ceiling; `cpu-info` prints what this machine allows.
- `QUICK=--quick` cuts any experiment that declares a budget down to minutes, and `exp-quick` runs that grid over all of them, which is what CI runs.
- The experiment services do not mount the working tree, so they run the source baked into `n2dbn:cpu` and an edit takes effect only after a rebuild. This check names which source a run would execute:

  ```bash
  docker compose run --rm --entrypoint python shell -m src.utils.provenance --check
  GIT_REV=$(git rev-parse --short HEAD) docker compose build tests   # if it says STALE
  ```
- Every run writes `results/<exp>/provenance.json` recording which source actually executed, the git revision, and the versions of the packages that can move a number.
- `EXP=exp3 docker compose run --rm plots` re-renders a finished run's figures from its stored results, with no refit.
- `docker compose run --rm shell` opens an interactive session with the working tree mounted, for a one-off command.

[CONTRIBUTING.md](CONTRIBUTING.md) covers the test suite, the invariants a change has to preserve, and where each kind of change belongs.
