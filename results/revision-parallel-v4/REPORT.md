# Revised experiment report

Operational checks for revising claims, not automatic proof of manuscript conclusions.

Old exp* results and smoke checks are excluded.

| Claim | Assessment |
|---|---|
| Nesting improves SHD with latent orders matched | inconclusive |
| Nested latent model beats the empty graph in SHD | inconclusive |
| Latent conditioning improves SHD within nested models | inconclusive |
| CalendarBench typed-rule recovery supports the manuscript claim | inconclusive |
| LifeSnaps grouped-count and summed-mark volume errors are at least four times smaller than the intercept baseline | not_supported |
| LifeSnaps reproduces per-label occupied-start-minute counts | not_supported |
| LifeSnaps preserves every tested nonstationarity form | not_supported |
| Nested forecasts are coherent | inconclusive |
| Nested forecasts match the accuracy of the strongest coherent baseline | inconclusive |
| Scalability sweep is fully measured | supported |

## Run completeness

| Experiment | Complete / planned |
|---|---|
| synthetic | 638 / 640 |
| benchmark | 4 / 10 |
| lifesnaps | 56 / 56 |
| forecast | 478 / 480 |
| forecast-extended | 478 / 480 |
| scalability | 48 / 48 |

## Synthetic recovery (completed runs)

| Method | Completed | F1 | SHD | Empty SHD | Self recall | Cross recall |
| --- | --- | --- | --- | --- | --- | --- |
| dynotears | 40 | 0.472925 | 18.95 | 16 | 0.740625 | 0.31875 |
| empty | 40 | 0 | 16 | 16 | 0 | 0 |
| exhaustive | 40 | 0.118568 | 20.075 | 16 | 0.16875 | 0.028125 |
| ges | 40 | 0.11556 | 19.675 | 16 | 0.16875 | 0.015625 |
| granger | 40 | 0.358642 | 19.775 | 16 | 0.571875 | 0.184375 |
| hc-bdeu | 40 | 0.232938 | 22 | 16 | 0.35 | 0.09375 |
| hc-bic | 40 | 0.118711 | 20.05 | 16 | 0.171875 | 0.025 |
| independent+latent | 40 | 0.146149 | 18.925 | 16 | 0.165625 | 0.11875 |
| independent-no-latent | 40 | 0.267994 | 18.175 | 16 | 0.303125 | 0.23125 |
| nested+latent | 40 | 0.364649 | 12.075 | 16 | 0.275 | 0.296875 |
| nested-no-latent | 40 | 0.467605 | 10.475 | 16 | 0.396875 | 0.390625 |
| pc | 40 | 0.103404 | 19.275 | 16 | 0.140625 | 0.025 |
| r2-sortnregress | 39 | 0.362739 | 19.9744 | 16 | 0.641026 | 0.13141 |
| random-sortnregress | 40 | 0.363716 | 20.175 | 16 | 0.653125 | 0.15 |
| tree | 40 | 0.234591 | 18.625 | 16 | 0.24375 | 0.121875 |
| var-sortnregress | 39 | 0.362827 | 19.9487 | 16 | 0.663462 | 0.13141 |

[Unrounded CSV](tables/recovery.csv)

## forecast (errors summarize finite completed runs)

| Method | Completed | Planned | Failed/missing | Extreme | MAE mean | MAE median | MAE p90 | MAE max | MSE mean | MSE median | Coherence max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DLinear-bottom-up | 40 | 40 | 0 | 0 | 2.89651 | 0.524385 | 1.09303 | 89.0408 | 37504.5 | 0.770378 | 0 |
| DLinear-independent | 40 | 40 | 0 | 0 | 2.97237 | 0.583418 | 1.15956 | 89.0807 | 37504.6 | 0.903466 | 1.53028 |
| DLinear-nnls | 40 | 40 | 0 | 0 | 2.99116 | 0.673404 | 1.30782 | 89.0807 | 37505.1 | 0.972003 | 0 |
| DLinear-ols | 40 | 40 | 0 | 0 | 3.00016 | 0.706934 | 1.31173 | 89.0807 | 37505.1 | 0.981143 | 0 |
| PatchTST-bottom-up | 40 | 40 | 0 | 0 | 2.98217 | 0.701128 | 1.33951 | 89.0446 | 37504.7 | 1.09544 | 0 |
| PatchTST-independent | 40 | 40 | 0 | 0 | 2.92288 | 0.581246 | 1.16489 | 89.0236 | 37504.5 | 0.884754 | 1.19824 |
| PatchTST-nnls | 40 | 40 | 0 | 0 | 2.91005 | 0.606351 | 1.1471 | 89.0293 | 37504.6 | 0.868405 | 0 |
| PatchTST-ols | 40 | 40 | 0 | 0 | 2.91852 | 0.607374 | 1.14752 | 89.0293 | 37504.6 | 0.873398 | 0 |
| naive | 40 | 40 | 0 | 0 | 3.78625 | 1.14995 | 4.24346 | 89.4298 | 37509 | 2.67164 | 0 |
| nested | 39 | 40 | 1 | 0 | 4.75523 | 0.644425 | 2.27891 | 115.69 | 38152.4 | 1.00181 | 1.43911e-16 |
| nested-constrained | 39 | 40 | 1 | 0 | 4.7687 | 0.644386 | 2.27891 | 116.208 | 38145.7 | 1.00181 | 1.77733e-16 |
| seasonal-naive | 40 | 40 | 0 | 0 | 2.91257 | 0.565098 | 1.1828 | 89.0301 | 37504.7 | 1.35864 | 0 |

[Unrounded CSV](tables/forecast.csv)

## forecast-extended (errors summarize finite completed runs)

| Method | Completed | Planned | Failed/missing | Extreme | MAE mean | MAE median | MAE p90 | MAE max | MSE mean | MSE median | Coherence max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DLinear-bottom-up | 40 | 40 | 0 | 0 | 0.376351 | 0.369986 | 0.680534 | 0.939916 | 1.8553 | 0.291488 | 0 |
| DLinear-independent | 40 | 40 | 0 | 0 | 0.403759 | 0.374227 | 0.694471 | 0.924991 | 1.88027 | 0.304246 | 0.487094 |
| DLinear-nnls | 40 | 40 | 0 | 0 | 0.527341 | 0.484713 | 0.880749 | 1.33062 | 2.07067 | 0.420422 | 0 |
| DLinear-ols | 40 | 40 | 0 | 0 | 0.533948 | 0.48487 | 0.880741 | 1.33062 | 2.07544 | 0.420396 | 0 |
| PatchTST-bottom-up | 40 | 40 | 0 | 0 | 0.397048 | 0.373192 | 0.664736 | 1.01959 | 1.92596 | 0.365787 | 0 |
| PatchTST-independent | 40 | 40 | 0 | 0 | 0.384677 | 0.401785 | 0.651824 | 0.991463 | 1.8931 | 0.324835 | 0.529397 |
| PatchTST-nnls | 40 | 40 | 0 | 0 | 0.379932 | 0.37944 | 0.64628 | 0.974248 | 1.87677 | 0.306396 | 0 |
| PatchTST-ols | 40 | 40 | 0 | 0 | 0.384491 | 0.377977 | 0.647604 | 0.997112 | 1.8797 | 0.310473 | 0 |
| naive | 40 | 40 | 0 | 0 | 1.26453 | 0.8468 | 2.3292 | 6.26799 | 7.37822 | 1.17095 | 0 |
| nested | 39 | 40 | 1 | 0 | 0.496651 | 0.477844 | 0.775179 | 1.15694 | 2.419 | 0.501601 | 1.49468e-16 |
| nested-constrained | 39 | 40 | 1 | 0 | 0.483749 | 0.477844 | 0.672437 | 0.970783 | 2.06228 | 0.501722 | 1.49516e-16 |
| seasonal-naive | 40 | 40 | 0 | 0 | 0.353423 | 0.338237 | 0.738194 | 1.12122 | 2.04976 | 0.310096 | 0 |

[Unrounded CSV](tables/forecast-extended.csv)

## CalendarBench observation definition

CalendarBench uses one start per occupied participant-label-minute and the sum of its individual duration marks. Starts at different minutes remain distinct even when durations overlap. `benchmark/annotations.csv` retains the original TSV rows. Fidelity measures these groups; raw manifest episode counts can differ. Manifest rules involving changed count quantities are flagged before interpreting label-level correspondence.

## CalendarBench annotations and modeled groups

| Label | Annotations | Groups | Count reduction | Summed marks |
| --- | --- | --- | --- | --- |
| action_planning | 250 | 250 | 0 | 2319 |
| alone | 4597 | 4597 | 0 | 372258 |
| anxious | 251 | 251 | 0 | 3788 |
| breakfast | 2088 | 2088 | 0 | 30010 |
| calm | 4585 | 4585 | 0 | 94107 |
| childcare | 1967 | 1967 | 0 | 213480 |
| cleaning | 1504 | 1504 | 0 | 109450 |
| cooking | 2643 | 2643 | 0 | 158020 |
| coping_planning | 669 | 669 | 0 | 4837 |
| cycling | 828 | 828 | 0 | 19070 |
| dinner | 2088 | 2088 | 0 | 97200 |
| disposition_to_attend_to_one_s_goals | 840 | 840 | 0 | 7109 |
| drink_water | 14574 | 14574 | 0 | 145740 |
| eldercare | 863 | 863 | 0 | 94470 |
| energetic | 7400 | 7400 | 0 | 836230 |
| external_regulation | 3813 | 3813 | 0 | 99009 |
| gaming | 887 | 887 | 0 | 62600 |
| going_out | 199 | 199 | 0 | 12820 |
| groceries | 1085 | 1085 | 0 | 64270 |
| gym | 249 | 249 | 0 | 16830 |
| habitual_behaviour | 5903 | 5884 | 19 | 219543 |
| happy | 4605 | 4605 | 0 | 100601 |
| home | 10215 | 10215 | 0 | 1180634 |
| hungry | 12198 | 12129 | 69 | 246084 |
| intend_to_walk | 816 | 816 | 0 | 7155 |
| intention_not_to_enact_a_behaviour | 1322 | 1321 | 1 | 17568 |
| intention_to_enact_a_behaviour | 2113 | 2111 | 2 | 32358 |
| intrinsic_motivation | 2568 | 2568 | 0 | 63561 |
| journaling | 655 | 655 | 0 | 33670 |
| laundry | 516 | 516 | 0 | 39120 |
| low_physical_opportunity_to_enact_a_behaviour | 3274 | 3264 | 10 | 255703 |
| low_self_efficacy | 852 | 850 | 2 | 16325 |
| lunch | 2088 | 2088 | 0 | 47590 |
| meeting | 1251 | 1251 | 0 | 67700 |
| napping | 2765 | 2765 | 0 | 47790 |
| need_for_autonomy | 619 | 619 | 0 | 10928 |
| office_work | 1151 | 1151 | 0 | 163820 |
| physical_behavioural_opportunity | 7565 | 7564 | 1 | 855333 |
| psychological_behavioural_capability | 4625 | 4620 | 5 | 313740 |
| public_park | 2287 | 2287 | 0 | 71275 |
| raining_or_snowing | 3724 | 3724 | 0 | 186948 |
| reading | 1903 | 1903 | 0 | 61610 |
| running | 304 | 304 | 0 | 7680 |
| self_efficacy_belief_for_a_behaviour | 756 | 755 | 1 | 51912 |
| shower | 2088 | 2088 | 0 | 29540 |
| sleep | 2088 | 2088 | 0 | 820670 |
| snacking | 5356 | 5356 | 0 | 53560 |
| social_behavioural_opportunity | 6303 | 6302 | 1 | 540221 |
| socialising | 1798 | 1798 | 0 | 165840 |
| stress_episode | 3355 | 3354 | 1 | 61283 |
| study_reading | 930 | 930 | 0 | 85580 |
| studying | 965 | 965 | 0 | 119630 |
| suitable_for_outdoor | 3675 | 3675 | 0 | 308517 |
| swimming | 177 | 177 | 0 | 11640 |
| tired | 5866 | 5866 | 0 | 170534 |
| visit_friends_or_family | 293 | 293 | 0 | 23310 |
| walking | 5180 | 5180 | 0 | 73180 |
| with_family | 5351 | 5351 | 0 | 418077 |
| workplace | 718 | 718 | 0 | 66099 |
| yoga | 311 | 311 | 0 | 11330 |

[Unrounded CSV](tables/benchmark-grouping.csv)

## LifeSnaps observation definition

One modeled start is one occupied participant-label-minute. Its duration mark is the sum of all constituent annotation marks. Counts and mark distributions describe these groups; raw episode multiplicities and individual episode lengths are not modeled. Summed marks are booked duration, not a union of occupied time intervals. The original annotations and their group memberships are retained in `lifesnaps/annotations.csv`.

## LifeSnaps observed annotations and modeled groups

| Label | Annotations | Groups | Count reduction | Summed marks |
| --- | --- | --- | --- | --- |
| gym | 502 | 501 | 1 | 28767 |
| mindfulness | 70 | 68 | 2 | 140 |
| nap | 214 | 214 | 0 | 25903 |
| sleep | 1841 | 1841 | 0 | 835640 |
| survey_response | 10163 | 5229 | 4934 | 10163 |
| walk | 1247 | 1247 | 0 | 43283 |

[Unrounded CSV](tables/lifesnaps-grouping.csv)

## LifeSnaps occupied-start-minute counts

| Label | Observed | Simulated mean | Ratio | Absolute log ratio |
| --- | --- | --- | --- | --- |
| gym | 501 | 529.6 | 1.05709 | 0.0555159 |
| mindfulness | 68 | 533.1 | 7.83971 | 2.0592 |
| nap | 214 | 215.2 | 1.00561 | 0.00559181 |
| sleep | 1841 | 1793.8 | 0.974362 | 0.0259726 |
| survey_response | 5229 | 4979.7 | 0.952324 | 0.0488504 |
| walk | 1247 | 1199.6 | 0.961989 | 0.0387525 |

[Unrounded CSV](tables/lifesnaps-counts.csv)

## LifeSnaps summed annotation-duration marks

| Label | Observed | Simulated mean | Ratio | Absolute log ratio |
| --- | --- | --- | --- | --- |
| gym | 28767 | 51940.1 | 1.80554 | 0.590862 |
| mindfulness | 140 | 572.8 | 4.09143 | 1.40889 |
| nap | 25903 | 38734.8 | 1.49538 | 0.40238 |
| sleep | 835640 | 844026 | 1.01004 | 0.0099854 |
| survey_response | 10163 | 9857.5 | 0.96994 | 0.0305211 |
| walk | 43283 | 43507 | 1.00518 | 0.0051619 |

[Unrounded CSV](tables/lifesnaps-booked_duration.csv)

## LifeSnaps error against the intercept baseline

| Channel | Labels | Full error | Intercept error | Improvement factor |
| --- | --- | --- | --- | --- |
| counts | all_label | 0.372314 | 0.0814998 | 0.218901 |
| counts | four_label | 0.0314582 | 0.0349546 | 1.11114 |
| booked_duration | all_label | 0.407967 | 0.107989 | 0.264699 |
| booked_duration | four_label | 0.252097 | 0.0641399 | 0.254425 |

[Unrounded CSV](tables/lifesnaps-error.csv)

## LifeSnaps accepted graph

| Scale | Target | Source kind | Source | Lag | Count coefficient | Duration coefficient |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | survey_response | context | notification_received | 0 | 3 | 0.371053 |
| 1 | survey_response | context | notification_received | 1 | 1.90526 | 0.129328 |
| 2 | gym | context | hour_cos | 0 | -0.416402 | -0.396093 |
| 2 | gym | context | hour_sin | 0 | -0.967262 | -0.284833 |
| 2 | nap | context | hour_cos | 0 | -0.919376 | 0.132896 |
| 2 | nap | context | hour_sin | 0 | -1.23402 | -0.392141 |
| 2 | sleep | context | hour_cos | 0 | 3 | 0.886433 |
| 2 | sleep | context | hour_sin | 0 | 0.2551 | -0.397438 |
| 2 | survey_response | context | hour_index | 0 | 0.202129 | 0.00807886 |
| 2 | survey_response | context | hour_cos | 0 | -1.44947 | -0.362937 |
| 2 | walk | context | hour_index | 0 | 0.0981732 | -0.000626211 |
| 2 | walk | context | hour_cos | 0 | -0.614386 | -0.00142343 |
| 3 | gym | context | day_index | 0 | -0.0622894 | 0.0425814 |
| 3 | survey_response | context | weekday_index | 0 | -0.047192 | -0.00428874 |

[Unrounded CSV](tables/lifesnaps-graph.csv)

## Criteria and evidence

- **Nesting improves SHD with latent orders matched**: all planned runs complete; four scenarios, >=40 pairs; paired SHD difference CI strictly above zero
- **Nested latent model beats the empty graph in SHD**: all planned runs complete; four scenarios, >=40 pairs; paired SHD difference CI strictly above zero
- **Latent conditioning improves SHD within nested models**: all planned runs complete; four scenarios, >=40 pairs; paired SHD difference CI strictly above zero
- **CalendarBench typed-rule recovery supports the manuscript claim**: inspect manifest reachability, effects of grouping on count quantities, and channel coefficients; label-only correspondence cannot establish typed recovery
- **LifeSnaps grouped-count and summed-mark volume errors are at least four times smaller than the intercept baseline**: intercept/full mean absolute log-volume error >=4 for both channels, using all labels; four-label subset also reported separately
- **LifeSnaps reproduces per-label occupied-start-minute counts**: every count-volume ratio in [0.8, 1.25]; all six panel arms and 50 half fits completed
- **LifeSnaps preserves every tested nonstationarity form**: every applicable form passes its paired equivalence interval; descriptive cells do not count as passes
- **Nested forecasts are coherent**: all planned predictions finite; maximum relative discrepancy <=1e-10
- **Nested forecasts match the accuracy of the strongest coherent baseline**: mean and median standardized hourly MAE no worse than every listed coherent baseline, with no failures; descriptive comparison, not significance
- **Scalability sweep is fully measured**: all four one-factor sweeps completed; timed-out cells remain explicit

Full measurements, failure reasons, label ratios, stability frequencies, paired intervals and forecast tails are in [claims.json](claims.json).

No manuscript numbers are updated automatically. Copy the new measurements only after reviewing the diagnostics and limitations.
