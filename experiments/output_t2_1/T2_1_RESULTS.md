# T2.1 results — the feedback share grows with coupling; within this grid the outcome never depends on it

Issue #48. Answers the open row C-35; durable claim **C-58**. Interpretation was fixed in
`PREREGISTRATION.md`, committed before the run. All numbers below are recomputed from
`t2_1_coupling.jsonl`, not carried from the run log.

## 1. The question and the instrument

At the working point, the feedback-attributable component of the weight change is 12% of
drift, measured as `diff_over_drift` on the recomputed-y pair against a severed-path
floor of exactly zero (C-30, C-31), and the connected and matched-offline arms settle in
the same basin (C-33). T2.1 varied coupling strength one axis at a time and asked two
pre-registered questions: does the share grow, and does any tested setting make the two
arms settle differently?

## 2. Gates, all passed

| gate | result |
|---|---|
| 17/17 matched axes, every cell | passed (10/10 cells, `arms_matched` true) |
| severed recomputed floor exactly 0.0 | passed: `bit_identical` true, `diff_over_drift` 0.000000e+00 |
| base cell reproduces EXP-001 within 2% | passed exactly: 0.1203852102289996 vs EXP-001's 1.203852e-01 |
| no non-finite cells | passed (0/10) |
| ceiling discipline | one cell fired (4x eta); reported as diagnostic, quoted nowhere below |

## 3. Result table

Share = `diff_over_drift`, recomputed pair, float64. Margins in logits beside each basin.

| cell | share | closed basin (margin) | offline basin (margin) | agree | clip |
|---|---|---|---|---|---|
| eta 0.25x | 0.026184 | `prolet` (0.26) | `prolet` (0.26) | yes | 0 |
| eta 0.5x | 0.054883 | `prolet` (0.17) | `prolet` (0.13) | yes | 0 |
| eta 1x (base) | 0.120385 | `comrade` (0.25) | `comrade` (0.20) | yes | 0 |
| eta 2x | 0.272187 | `Divine` (2.38) | `Divine` (2.03) | yes | 0 |
| eta 4x | 0.555575 | `Divine` (2.50) | `【` (0.72) | **no** | **fired, both arms** |
| cadence 2 | 0.054268 | `prolet` (0.17) | `prolet` (0.13) | yes | 0 |
| cadence 4 | 0.025385 | `prolet` (0.26) | `prolet` (0.26) | yes | 0 |
| steps 60 | 0.026649 | `prolet` (0.03) | `prolet` (0.03) | yes | 0 |
| steps 240 | 0.342170 | `Divine` (2.14) | `Divine` (2.30) | yes | 0 |
| severed (gate) | 0.000000 | `the` (0.18) | `the` (0.18) | yes | 0 |

## 4. Answer 1: the share grows, monotonically, on every axis

Within the ceiling-silent cells the share rises monotonically with every coupling knob.
Along the eta ladder it slightly outpaces proportionality (0.25x, 0.5x, 1x, 2x of eta*
give 0.22x, 0.46x, 1x, 2.26x of the base share). Cadence mirrors eta with striking
precision: firing every 2nd step lands within 1.2% of half eta (0.05427 vs 0.05488), and
every 4th step within 3.1% of quarter eta (0.02539 vs 0.02618). Episode length compounds
fastest: the first doubling multiplies the share by 4.52 (0.0266 at 60 steps to 0.1204 at
120) and the second by 2.84 (to 0.3422 at 240). Both doublings exceed a factor of 2, which
is faster than proportional and consistent with feedback feeding on its own accumulated
effect; the two ratios differ, so no single per-doubling constant is claimed.

## 5. Answer 2: within the T2.1 grid, the outcome never depends on feedback

In every ceiling-silent cell the two arms settle in the same basin, including the two
cells where the settled state changes **dynamical class** into the period-2 `Divine`
cycle (2x eta, and 240 steps at the working eta): lag-1 approx 0.67 to 0.69 with lag-2 =
1.0 in **both** arms, margins 2.0 to 2.4 logits, ceiling silent. Whatever the rule alone
does offline, the connected loop does too. So C-33's "feedback changes no visible
outcome" survives throughout this grid, up through 2x eta and 240 steps: the share grows
from 2.5% to 34% of drift without buying a different destination. **This clause is bounded,
not universal**: the follow-up probe T2.1b (same PR, `T2_1B_RESULTS.md`, claim C-59) finds
the first ceiling-silent disagreement at 2.5x eta, outside this grid.

Two observations that travel with this, stated as observations rather than claims:

1. **The `Divine` class change is reachable by a live episode, ceiling silent, and does
   not require feedback.** The dial-up sweep (C-28, provisional) reached `Divine` only at
   1.5x the episode's installed edit; here a 240-step episode at the working step size
   reaches it directly, in both arms. This extends C-28's territory and answers, for the
   episode route, the question of whether the loop's feedback is what finds `Divine`: it
   is not; the rule's accumulated edit is sufficient.
2. **The one arms-disagreement sits in the saturated cell.** At 4x eta both arms hit the
   0.05 drift ceiling exactly and then part ways (closed: `Divine`; offline: a junk token
   outside the 5 baseline families). Under standing rule 2 this is a diagnostic, not a
   result. The pre-registered follow-up T2.1b probed the ceiling-silent gap and found the
   disagreement onset at 2.5x eta (see `T2_1B_RESULTS.md`, C-59).

## 6. Caveats

- Conditional on `y_source="recomputed"` throughout; C-30 and C-31's caveats travel.
- n = 1 prompt (`A01_physics`), 1 site (`blocks.6.mlp`), 1 seed, `hebb` only.
- The severed gate ran at the base point only; the exact-zero floor is re-verified there,
  not at every setting (the 17-axis verification runs at every setting).
- "Monotone" is claimed on a 4-point eta ladder, 2 cadences, 3 lengths; no functional
  form is claimed beyond "faster than proportional in episode length".

## 7. Files

- `PREREGISTRATION.md` — grid, gates and interpretation, committed before the run.
- `t2_1_coupling_sweep.py` (in `experiments/`) — runner; reuses `run_matched_arms`.
- `t2_1_coupling.jsonl` — one record per cell, appended as each completed.
- `meta.json` — configuration and environment.
