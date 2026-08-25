# dynamic_patterns

The CalendarBench `dynamic_patterns` run, generated 2026-08-16: 30 people over
12 weeks from 2026-02-16, solved per person per day by Z3 on a 10-minute
placement grid.

## The four files

| file | what it holds |
|---|---|
| `event_log.tsv` | every eventuality, four tab-separated columns: `person_id`, `label`, `start_iso`, `end_iso`. 169,909 rows over 60 labels. |
| `manifest_declared.json` | per label: its kind, its category, the patterns acting on it, the carry-over rules naming it, and whether it is declared stationary. |
| `manifest_resolved.jsonl` | per person per day per label: what the declarations resolved to and what the solver realized. 83,536 lines. |
| `used_configs.json` | the resolved generator configuration, so the run reproduces. |

`event_log.tsv` marks no difference between an event, a routine and a context
episode. All three are named things with a start and an end, and the model must
not see a fact about the generator. The scorer reads the manifests; the fit
reads the log.

## What is declared

| kind | target | scale | label |
|---|---|---|---|
| count trend up | episodes | week | walking |
| count trend down | episodes | week | snacking |
| duration trend up | duration | week | gym |
| duration trend down | duration | week | office_work, on one persona |
| non-monotone count | episodes | week | running |
| weekday seasonality, counts | episodes | weekday | walking, +25 percent Sat and Sun |
| weekday seasonality, duration | duration | weekday | journaling |
| weekday gate | episodes | weekday | office_work, off Sat and Sun |
| time-of-day concentration | placement | day | walking, in 07-10, 12-14 and 17-21 |
| time-of-day drift | placement | week | sleep, onset later each week |
| quarter effect | episodes | quarter | swimming, +50 percent in q2 |
| stationary control | none | | 43 labels carrying `mode: fix` and nothing else |

Six carry-over rules declare lagged dependence: a walking self edge, a sleep
duration threshold at seven hours driving walking counts, running counts
driving yoga counts, office_work duration driving sleep duration, meeting
counts driving journaling duration, and a gym self edge at a seven-day lag.

## What came back, measured on this run

| declared | realized |
|---|---|
| walking count trend, +3 over 12 weeks | 2.54 to 3.74 a day, a rise of 1.46 against a declared 1.86 |
| walking weekend seasonality, +25 percent | 2.99 weekday, 4.43 weekend |
| snacking count trend, -2 | 2.46 to 1.80 a day |
| running non-monotone | 0.9, peak 2.05 at week 5, 1.05 at week 11 |
| duration trend up, gym | 45.5 to 65 minutes |
| swimming quarter effect, +50 percent in q2 | 1.2 to 1.8 a day |
| journaling weekend duration, +50 minutes | 34.6 weekday, 80.5 weekend |
| sleep placement drift, +90 minutes | onset 1385 to 1416, +31 minutes |
| sleep over 7 hours drives walking, +2 | 3.01 after a short night, 3.98 after a long one |

All six carry-over rules fire: sleep to walking 745 times, walking to walking
320, gym to gym 149, meeting to journaling 141, running to yoga 83, office_work
to sleep 72.

Two rows come back attenuated and both are a clamp. Sleep drifts into a window
that ends at midnight, so the last 59 of the declared 90 minutes have nowhere to
go. The carry-over asks walking for +2 on a band whose ceiling is 5.
`manifest_resolved.jsonl` carries the declared value and the realized one on
every person-day label, so the attenuation is readable and does not have to be
inherited. The 43 stationary labels hold a weekly standard deviation between
0.023 and 0.163.

## The two spreads, and why they are there

`target_spread_minutes: 45` and `count_spread_episodes: 1` scatter each day's
realized value around its declared one, symmetrically and seeded per person per
day. Without them the duration objective pins a source to a single value for
the whole horizon: office_work sat at exactly 360 minutes on 568 of 600 days and
meeting at exactly 3 on 588 of 600. A lagged edge out of a constant is
unrecoverable however it is thresholded, so two of the six declared rules never
fired at all. The spreads restore the variance, and the thresholds sit near each
source's own median so an indicator splits the horizon.

## Regenerating it

The configuration lives in CalendarBench at
`src/experiments/persona/dynamic_patterns/`, and its README gives the command.
