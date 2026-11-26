# N²DBN: Nested Non-Stationary Dynamic Bayesian Networks

The WSDM manuscript reports synthetic structure recovery with a top-down allocation model, QuitoBench point forecasts, and event-log studies with a finest-minute start and duration model. The experiment commands below use the implementations that produced those comparisons.

## Paper experiments

From the repository root on the server, record the input files and build the CPU image once.

```bash
git rev-parse HEAD
sha256sum data/observed_lifesnaps_event_log.csv data/quito/test_min.parquet
docker compose build tests
```

Run the following checks on the server's copies of the data. Both QuitoBench commands refit all methods on the same series, including the original independent deep forecasts. Each output records the parquet SHA-256, training window, forecast origin, per-series errors and coherence. The extended split has a different origin and uses observations from the benchmark test period for training.

```bash
docker compose run --rm lifesnaps-observation-audit
docker compose run --rm latent-matched-recovery
docker compose run --rm quito-forecast-reconciliation
SPLIT=extended docker compose run --rm quito-forecast-reconciliation
```

The LifeSnaps audit compares raw annotation counts with occupied participant-label-minutes on the observation mask used by the event model. Its `audit.json` records the source SHA-256, denominator for each label, simultaneous starts and duration recodings. The paired structure experiment writes `paired.csv` with both arms and their differences for each of 18 scenarios and three seeds. The forecast JSON distinguishes the sum of ten-minute predictions, the independent hourly prediction and its least-squares reconciliation. The summary includes the number of nonfinite errors per metric. New runs write under `results/paper/<service>/`. The archived per-series scores are in [the cutoff forecast output](results/exp19/exp19_min_cutoff.json) and [the extended forecast output](results/exp19/exp19_min_extended.json). The submitted LifeSnaps fit records input SHA-256 `fe8a2b019cec94b78968a40f55245eb8f5a6e4b5919eb6e94155eaafce10fb06`. The separate annotation audit cited in the text records `d2d28bef4db292f0ca838c23d9655485a6e3050fca82344baf6fe45766d1de7d`.

| Paper comparison | Compose service |
| --- | --- |
| Simulated structure recovery, Table 1 | `synthetic-structure-recovery` |
| Ordering baselines and sortability measures | `ordering-baselines`, `benchmark-sortability` |
| Recovery with matched latent orders | `latent-matched-recovery` |
| CalendarBench log fidelity | `calendarbench-log-fidelity` |
| Agreement across resolutions | `cross-resolution-coherence` |
| CalendarBench declared-rule recovery | `calendarbench-pooled-recovery`, `calendarbench-structure-baselines` |
| QuitoBench original forecast comparison | `quito-traffic-forecast` |
| QuitoBench independent, bottom-up and reconciled predictions | `quito-forecast-reconciliation` |
| LifeSnaps coverage, recoding and multiplicity | `lifesnaps-preparation`, `lifesnaps-observation-audit` |
| LifeSnaps graph, falsification and half-panel stability | `lifesnaps-graph-stability` |
| LifeSnaps simulation, volume and changepoints | `lifesnaps-log-fidelity` |

Run a service with `docker compose run --rm SERVICE`. Set `JOBS` for the parallel recovery and CalendarBench services. The two forecast services default to the declared cutoff and 40 series. For a short server check, `LIMIT=1 docker compose run --rm quito-forecast-reconciliation` fits all forecast methods on the first eligible series. A subsequent full run replaces that split's JSON.

## Data

| Path                                      | Holds                                                                                                           | Shipped                    |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `data/dynamic_patterns/`                | the declared benchmark: 30 simulated participants over 12 weeks, 60 labels, with its manifest                   | yes                        |
| `data/observed_lifesnaps_event_log.csv` | the observed log derived from the LifeSnaps MongoDB dump                                                        | no, rebuild it (below)     |
| `data/quito/`                           | QuitoBench `test_min.parquet` ([arXiv:2603.26017](https://arxiv.org/abs/2603.26017)) | no, supply the server's copy |

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

## Running the test suite

```bash
docker compose run --rm tests
```
