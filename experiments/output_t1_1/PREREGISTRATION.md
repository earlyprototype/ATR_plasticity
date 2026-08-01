# T1.1 pre-registration (fixed in writing BEFORE the trajectory was observed)

Timestamp (UTC): 2026-08-01T16:22:12Z
Agent: magenta. Issue #45 / ALIGNMENT_REVIEW.md T1.1. Settles CLAIMS C-51, decides C-26.

## The test
Install the working-point edit `W0 + ΔW` as a **static** weight change at site
`blocks.6.mlp` (= `transformer.h.6.mlp.c_proj`, W_out, shape (3072, 768)). Seed the ATR
loop from the model's **original frozen `prolet` settled state** — the state the frozen
(W0) loop settles on for prompt `A01_physics` before any weights moved. Iterate 120 steps
(the episode length) to settling.

ΔW is the working-point edit: `hebb`, eta = 7.065171428571429e-05, cadence 1,
max_delta_frac = 0.05, seed 0, 120-step closed episode on `A01_physics`. Relative weight
change ‖ΔW‖_F / ‖W0‖_F = 0.011239... (≈ 0.0112). Reference ‖W0‖_F = 164.85407309107723.
All norms in float64. ΔW is reproduced in memory (the repo persists no ΔW), and the
reproduction is gated on reaching `comrade` (47998) with 0.0% clip, matching EXP-001.

## Pre-registered two-outcome interpretation (this is the whole point of writing it first)

- **If the loop STAYS at `prolet`** (settled word `prolet`, token 22758): two resting
  states coexist under the edited weights — `prolet` is still a fixed point AND `comrade`
  is a fixed point of the edited map. This is a **created attractor** (issue #25 ladder
  step 4, a bifurcation). => moves C-26 toward `supported`; settles the open row C-51 with
  answer "yes, the original `prolet` state stays put."

- **If the loop MOVES to `comrade`** (settled word `comrade`, token 47998): the edited map
  has a single displaced attractor — the `prolet` fixed point did not survive the edit, it
  flowed to `comrade`. This **refutes** C-26's created-attractor reading; the honest
  reading is a **boundary / displacement move** (ladder step 2: one continuously-relocated
  fixed point that the readout relabels). => C-51 answered "no, it does not stay"; C-26
  stays refuted, not merely not-established.

- **Any other settled word** (not `prolet`, not `comrade`): report it verbatim; it would
  mean the edited map's basin for the `prolet` direction is neither of the two words the
  dispute is about, which is itself a finding and would need its own reading.

## GATE (must pass and be recorded BEFORE interpreting the trajectory)
Zero-step-size control: install `W0 + 0·ΔW` through the same write path as the real edit,
seed from the frozen `prolet` state, iterate. Its trajectory must be **bit-identical**
(max abs difference exactly 0.0 at every position, every step) to the frozen (unedited W0)
loop iterated from the same `prolet` state. If not bit-identical: STOP, report as a finding,
do not interpret T1.1.

## Renormalisation shell (stated so the protocol is checkable)
Primary protocol uses the loop's renormalisation target `initial_norm` = ‖x0‖ from the W0
forward pass on `A01_physics` — the SAME map that produced both the frozen `prolet` state
and the `comrade` episode, and the same map D1 uses. This makes T1.1 the exact dual of
`BASIN_BIFURCATION.md` D1 (D1: W0 map from the `comrade` state; T1.1: W0+ΔW map from the
`prolet` state) and makes the eta=0 gate bit-identical by construction. A robustness variant
re-runs T1.1 on the edited system's own energy shell (`initial_norm` recomputed from a fresh
forward pass under W0+ΔW, D2-style); the verdict must be invariant to this choice.

## No tuning
No parameter is tuned to any basin percentage or to any target outcome. eta, cadence,
prompt, site, seed, step count are all fixed to the working-point values above.
