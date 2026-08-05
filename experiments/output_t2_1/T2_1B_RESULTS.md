# T2.1b results — feedback changes the outcome at 2.5x eta, ceiling silent

Addendum to T2.1, pre-registered in `PREREGISTRATION_T2_1B.md` (committed before the
run). Answers the question T2.1's saturated 4x cell raised but could not: **is there a
ceiling-silent step size at which the connected and matched-offline arms settle
differently? Yes: 2.5x eta*.** Durable claim **C-59**. All numbers recomputed from
`t2_1b_gap.jsonl`.

## Result table

Share = `diff_over_drift`, recomputed pair, float64. Margins in logits.

| cell | share | closed basin (margin) | offline basin (margin) | agree | ceiling |
|---|---|---|---|---|---|
| eta 2.5x | 0.349645 | `Divine` (2.36) | `【` (2.13) | **no** | **silent, both arms** (drift 0.0403 / 0.0465 vs cap 0.05) |
| eta 3.0x | 0.423137 | `Divine` (2.36) | `【` (0.70) | no | fired, both arms — diagnostic only |
| eta 3.5x | 0.494704 | `Divine` (2.42) | `【` (0.60) | no | fired, both arms — diagnostic only |

All three cells: 17/17 matched axes, no non-finite values, seed 0, 120 steps, cadence 1.

## The admissible finding

At eta = 2.5x the working step size, with the safety ceiling silent on **both** arms
(closed drift 4.03%, offline drift 4.65%, cap 5%) and both margins far above the
pre-registered 0.05-logit threshold:

- **connected loop**: settles into the genuine period-2 `Divine` cycle (lag-1 0.685,
  lag-2 1.000);
- **feedback severed**: ends on a **near-stationary trajectory** (late-window lag-1 0.998)
  reading out `【`, a token belonging to none of the five baseline basin families. A
  late-window lag-1 of 0.998 does **not** establish a fixed point; the severed arm's
  dynamical class is unclassified pending a convergence-tested rerun.

Same rule, same recorded inputs, same seventeen matched axes; the only difference is
whether the loop's own activity feeds back into the drifting weights. This is the first
admissible observation in the project of feedback changing the behavioural outcome. What
is established: the basin-label disagreement, and the connected arm's verified period-2
signature (lag-1 0.685, lag-2 1.000). The stronger class-vs-class reading (cycle with
feedback, fixed point without) awaits a convergence classification of the severed arm's
terminal trajectory, which its late-window lag-1 of 0.998 alone cannot deliver.

Combined with T2.1, the coupling axis now reads: arms agree at 0.25x through 2x
(including an agreed class change into `Divine` at 2x); arms first disagree, ceiling
silent, at 2.5x; every cell at 3x and above saturates the ceiling and is diagnostic
only. The onset of feedback-dependence sits in the interval (2x, 2.5x] at this
resolution.

## Observations that travel (not claims)

1. The severed arm's `【` readout recurs at 3x, 3.5x and (from T2.1, saturated) 4x:
   without feedback, strong raw Hebbian accumulation heads somewhere degenerate, and the
   feedback path is what steers the trajectory onto real attractor structure (`Divine`).
   Stated as an observation; one prompt, one seed.
2. The share at the disagreement onset is 0.3496, essentially the same as the agreeing
   240-step cell's 0.3422. **Share alone does not predict divergence**; the axis matters
   (episode length compounds share without divergence; step size buys divergence at the
   same share).

## Caveats

- Conditional on `y_source="recomputed"` (C-30, C-31 caveats travel).
- n = 1 prompt (`A01_physics`), 1 site, 1 seed, `hebb` only, single grid point of onset;
  the interval (2x, 2.5x) is unprobed below the 0.5x grid step.
- `【`'s basin identity is read from the standard readout; it is not one of the five
  frozen baseline basins, and no claim is made about what it "is" beyond a terminal readout
  outside the census. **It is not called a settled fixed point**, which this caveat used to say and
  which the body of this file correctly denies twice: a late-window lag-1 of 0.998 does not
  establish convergence, no convergence test was run, and the severed arm's dynamical class is
  therefore unclassified.

## Files

- `PREREGISTRATION_T2_1B.md` — grid, gates, interpretation, committed before the run.
- `t2_1b_gap.jsonl` — one record per cell.
- Runner: `experiments/t2_1_coupling_sweep.py --cells b`.
