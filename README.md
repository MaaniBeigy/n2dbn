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

The LifeSnaps audit compares raw annotation counts with occupied participant-label-minutes on the observation mask used by the event model. Its `audit.json` records the source SHA-256, denominator for each label, simultaneous starts and duration recodings. The paired structure experiment writes `paired.csv` with both arms and their differences for each of 18 scenarios and three seeds. The forecast JSON distinguishes the sum of ten-minute predictions, the independent hourly prediction and its least-squares reconciliation. The summary includes the number of nonfinite errors per metric. Paper experiment containers mount `results/paper/` and each writes to its own service-named subdirectory. Historical `results/revision-parallel-*/` directories belong to a separate experimental implementation. The paper services do not read them. Table 3 uses the [July forecast results](results/paper/quito-forecast-reconciliation/reconciled-traffic_min_cutoff.json) and [August forecast results](results/paper/quito-forecast-reconciliation/reconciled-traffic_min_extended.json). The submitted LifeSnaps fit records input SHA-256 `fe8a2b019cec94b78968a40f55245eb8f5a6e4b5919eb6e94155eaafce10fb06`. The separate annotation audit cited in the text records `e3d372b60d6297413da2616e8e3b66ddd3803fec9afcd864b142cda14089f6b9`.

| Paper comparison | Compose service |
| --- | --- |
| Simulated structure recovery, Table 1 | `synthetic-structure-recovery` |
| Ordering baselines and sortability measures | `ordering-baselines`, `benchmark-sortability` |
| Recovery with matched latent orders | `latent-matched-recovery` |
| CalendarBench log fidelity | `calendarbench-log-fidelity` |
| Agreement across resolutions | `cross-resolution-coherence` |
| CalendarBench declared-rule recovery | `calendarbench-pooled-recovery`, `calendarbench-structure-baselines` |
| QuitoBench original traffic forecast | `quito-traffic-forecast` |
| QuitoBench independent, bottom-up and reconciled predictions | `quito-forecast-reconciliation` |
| LifeSnaps coverage, recoding and multiplicity | `lifesnaps-preparation`, `lifesnaps-observation-audit` |
| LifeSnaps graph, falsification and half-panel stability | `lifesnaps-graph-stability` |
| LifeSnaps simulation, volume and changepoints | `lifesnaps-log-fidelity` |

Run a service with `docker compose run --rm SERVICE`. Set `JOBS` for the parallel recovery and CalendarBench services. Both QuitoBench services default to the declared cutoff and 40 series. For a short server check, `LIMIT=1 docker compose run --rm quito-forecast-reconciliation` fits all forecast methods on the first eligible series. A subsequent full run replaces that split's JSON.

## Forecasting workflow starts and active time

The BPI Challenge 2017 loan-application log records dated workflow transitions. We pair consecutive transitions for the same case and activity when the same recorded resource starts or resumes work and then suspends or completes it. Each positive interval counts as one work segment. This definition excludes pauses and can give several segments for one task. The audit records the number of retained and excluded openings. [Original event log](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b)

Run from the repository root on the server. The published gzip file is about 30 MB.

```bash
mkdir -p data/bpic2017
curl -fL --retry 3 https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c -o data/bpic2017/workflow.xes.gz
echo '183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249  data/bpic2017/workflow.xes.gz' | sha256sum -c -
docker compose build bpic2017-workflow-forecast
docker compose run --rm bpic2017-workflow-forecast
```

At midnight UTC on November 7, 14, 21 and 28, 2016, we fit each method on the preceding 180 days and forecast the next 168 hours. We count starts and sum active seconds by activity in the hour of the start, rounding the hourly sum to whole seconds. Training includes only intervals closed by the forecast origin. We select labels from that training window and report unseen test starts. Hourly starts can exceed one and active time can exceed one hour when workers act concurrently. We use the uncapped allocation model for these aggregate observations.

We fit two regimes and no continuous factor at each of the hourly and daily resolutions, with two candidate hourly lags, seven daily lags, maximum in-degree one and up to six fitting sweeps. We supply daily and weekly calendar terms at their respective resolutions. At each origin we initialize each regime from its fitted terminal marginal and average 64 simulated future paths using observed count and duration lags. We compare with weekly persistence and direct hourly DLinear and PatchTST forecasts using one week of input, one week of output and 25 training epochs. All methods receive the same historical starts and active times. We score hourly forecast MAE separately in starts and active minutes. We retain errors by activity and origin. Each hourly NPZ file contains observations and forecasts, with columns ordered as starts followed by active seconds in the saved activity order.

The service writes `interval-audit.json` and `hourly-forecasts.json` in `results/paper/bpic2017-workflow-forecast/`. The comparison has its own protocol on a public BPM log. The reported QuitoBench results use a different dataset and task.

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
