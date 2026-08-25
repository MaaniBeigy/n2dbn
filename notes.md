# Where the manuscript stands, and the order to finish it in

Written 2026-08-24, after the post-repair exp16 and exp15 runs. Root
`MEMORY.md` holds the evidence; `PLAN.md` 39.4.10a item 5 holds the closure.

## What is settled

Arm 1 has a result and it is the one the manuscript needs.

- **The two runs agree.** exp15 and exp16's fitted leg accept the same 23
  edges, edge for edge, both at `4a2a39f` with `runs_image_source: true`.
- **The bidirectional split shows on real data.** All 12 exogenous edges sit at
  the hour scale; all 7 edges at the day and week scales read a label's own
  past; nothing is bought at the minute scale.
- **`cor:model_calendar_home` is confirmed in both directions.** Home beats
  withheld by 2295.64 nats; every-scale costs 33.90 nats against home.
- **The drawn log is faithful on volume.** Four of six labels inside 6 percent
  per replicate; reconciliation error exactly 0.
- **Non-stationarity is preserved at the coarse scales.** Net of the positive
  control the fitted leg passes 4 of the 6 evidence-bearing grid rows, and
  changepoint agreement is monotone in the scale for all six labels, 0.95 to
  0.96 at the week scale for three of them.
- **The volume is what separates the graph from the empty one.** Mean absolute
  log drawn-to-real ratio over the four well-supported labels: fitted 0.034,
  no-calendar 0.061, empty 0.137.

## What is honestly short, and is reported as such

- `survey_response` draws at 0.53 of its rate and `mindfulness` at 1.94; both
  are the labels the panel supports least, in opposite directions.
- The drawn logs leave 5.3 percent of observed days empty against the panel's
  17.0 percent. No grid row reads this.
- A06 and A11 fail at the hour scale. The within-day shape is where the model
  departs.
- Three of four declared controls do not break every row they are declared
  against, which bounds the grid at 46 tested cells of 228 on this roster.
- **The grid does not separate the fitted graph from the empty one.** The only
  separating row, B3, is one of the five that fail their own replicate noise; on
  the six evidence-bearing rows the two legs are identical. Report the volume
  comparison for the separation and the grid for the pattern.
- R7 rejects the model for `gym` on its one testable parent column, and the two
  `nap` self edges carry uninterpretable coefficients that R9 removes.
- **No edge between two distinct behaviour labels is bought at any scale.** The
  23 edges are 12 exogenous and 11 self. The anchor set declares no
  label-to-label positive, so R3 cannot adjudicate it, and the arm has no
  instrument separating "this cohort carries none" from "8 weeks of 51 cannot
  price one". The manuscript reports the count and declines to choose.

## The order to finish in, all five done 2026-08-24

1. `paper/sec_arm1.tex`, input from `sec_results.tex` before the cost
   subsection. Six tables, every number read back from the stored runs by
   `src/evaluation/arm1_report.py` through `paper/_make_arm1_tables.py`.
2. Abstract, intro and synopsis. The abstract carries the volume result, the
   grid's non-separation and the calendar ablation; the abstract and the intro
   said "top-down" where the pass is bidirectional and now say so; the synopsis
   points at the arm.
3. Limitations. A new subsection on what the arm does not settle, and the
   conclusion now establishes five things instead of four.
4. `docker compose run --rm paper` builds 69 pages with no errors and no
   undefined references. The one overfull box is pre-existing.
5. **2753 passed, 4 skipped, 100 percent over 13,348 statements, 0 missed;
   lint clean on 220 files.** Coverage came in at 99 percent first, and the 22
   uncovered statements were the 08-23 repairs' own guard branches, every one
   reachable on the run that caused the repair.
   `tests/test_phase102_guard_callers.py` closes them.

## The submission is `paper/WSDM/sigconf.tex`

`paper/n2dbn.tex` is the long-form record. The submission is the ACM `sigconf`
paper under `paper/WSDM/`, built with

    cd paper/WSDM && docker run --rm -v "$PWD":/w -w /w n2dbn:paper \
        latexmk -pdf -interaction=nonstopmode sigconf.tex

Its three `TODO` sections were written on 2026-08-24 and it builds at 9 pages
with no undefined references or citations. It presents three scales, hour, day
and week, with the minute slot named as the resolution the likelihood and the
episodes are defined at.

## What is left before submission

1. Read the three new sections end to end in the built PDF.
2. Check the LifeSnaps citation added to `paper/WSDM/references.bib`
   (`yfantidou_lifesnaps_2022`, Scientific Data 9:663, 2022) against the real
   entry.
3. Check the declared-benchmark table (Table 3). `results/exp11` is the n2dbn
   run on that benchmark and `results/exp13*` the eight learners; the table now
   carries both. On the three rules every method can express, n2dbn and the four
   best learners each find the same one, and the separation is the stationary
   block: 3 edges for n2dbn against 134 to 165 for the search-based learners, of
   42 labels the manifest declares stationary. Precision against the manifest
   stays unreportable, since the generator applies rules it does not declare. Everything below this line is
optional and none of it changes a claim.

## Optional, in value order

- **39.4.10a item 4b**, splitting the eta-clip counter into (start, mark) x
  (low, high). Code only; the stored runs cannot be re-read for it, and a
  re-run of exp16 costs 4.4 hours.
- **40.4's interaction form for the regime.** The additive homogeneous chain is
  now inert in the fit and costs one grid cell in the draw, which is the
  measurement that makes the interaction the next thing to build. It is a
  model change and it needs a re-run.
- `docs/cook2013casas.pdf` and `docs/klasnja2019heartsteps.pdf` are the wrong
  articles. The bib entries are fine; the vendored files are not.
- exp6 and its compose service still default to `data/events.csv`, retired from
  arm 1.

## Resolved since the last note

`paper/sec_learning.tex` no longer describes a coarsest-to-finest sweep. Its
opening and \S"The outer loop" both state the two passes and cite
`prop:model_block_invariance` for why both are necessary.
