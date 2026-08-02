# T2.1 pre-registration — does the feedback share grow with coupling, and ever change the outcome?

Written and committed **before** the sweep runs, per standing rule: interpretation is fixed
in writing before each run. Issue #48; answers the open row **C-35**; the durable claim is
**C-58** (claimed on the Identifier registry, discussion #17, before use).

## Question

At the working point, the feedback-attributable component of the weight change is 12% of
drift (`diff_over_drift`, `recomputed`-y arm, severed floor exactly 0.0 — C-30..C-34), and
the connected and matched-offline arms settle in the same basin (C-33: feedback steers the
weights, changes no visible outcome). T2.1 varies coupling strength and asks:

1. Does `diff_over_drift` (recomputed) **grow** as coupling strengthens?
2. Is there a tested setting at which the two arms **settle in different basins**?

## Grid

Base point (identical to EXP-001 / the register's working point): `hebb`,
site `blocks.6.mlp`, prompt `A01_physics`, seed 0, cadence 1, 120 steps,
`max_delta_frac` 0.05, eta* = 7.065171428571429e-05, loop 0→11, severed control 0→3.

Axes varied **one at a time** around the base point (no cross product; each cell names the
one axis it moves):

- **eta ladder**: {0.25×, 0.5×, 1×, 2×, 4×} eta* — five cells. Coupling scales directly
  with step size.
- **cadence**: apply every {2, 4} steps at eta* (base is 1) — two cells. Sparser firing is
  weaker coupling.
- **episode length**: {60, 240} steps at eta* (base is 120) — two cells. Longer episodes
  give feedback more turns of the loop to act.

Nine routed cells total. Plus: one severed cell (loop 0→3) at the base point, as the gate
that the detection floor is still exactly 0.0; and eta=0 gates ride inside every
`run_matched_arms` call via its verification (17/17 axes must match or the cell aborts).

## Measurements per cell

From `run_matched_arms` (y_source="recorded", also_recomputed_y=True), recorded to jsonl
as in EXP-001:

- `diff_over_drift` for the recomputed pair (headline; floor exactly 0.0) and the recorded
  pair (protocol-literal; floor nonzero) — read directly from
  `comparison["weight_recomputed_y"]["diff_over_drift"]` as `compare_weights` defines it
  (‖W_closed − W_off‖_F over the larger arm's own drift), plus `cos_delta` and the norm
  ratio, all float64.
- Behavioural readout: frozen re-run under each arm's final matrix from the same start
  state; basin label **with margin**, lag-1/lag-2 cosines, phase-aware final-state cosine
  between arms.
- Clip state per arm (sticky boolean) — any cell where the ceiling fires is reported but
  **never quoted** in conclusions (standing rule 2).
- Nonfinite flag; n_updates; rel weight change per arm; float64 norms throughout
  (standing rule 5).

## Interpretation, fixed now

- **Growth**: read `diff_over_drift` (recomputed) across the five-cell eta ladder,
  ceiling-silent cells only. "Grows with coupling" requires monotone increase across the
  ceiling-silent ladder; anything else is "does not grow monotonically" with the shape
  reported. Cadence and length cells are reported as secondary, same rule.
- **Outcome divergence**: a cell counts as "arms disagree" only if the two frozen re-runs'
  basin labels differ AND both margins exceed 0.05 logits. A disagreement at a smaller
  margin is reported as "within basin-resolution" (C-07 discipline), not as divergence.
- If any ceiling-silent cell shows arms disagreeing: C-35 is answered "yes — at [setting],
  feedback changes the outcome"; C-58 holds the positive claim.
- If no ceiling-silent cell shows disagreement: C-35 is answered "no, in the tested
  range"; C-58 holds the negative claim with the tested range stated. This is the clean
  negative issue #48 names as worth stating plainly.
- The 12% figure and all shares stay conditional on `y_source="recomputed"` (C-30..C-34
  caveats travel).

## Gates (run aborts or the cell is voided if any fails)

1. `verify_arms_matched` 17/17 on every pair used (raises on mismatch).
2. Severed cell at base point: recomputed severed floor exactly 0.0
   (`bit_identical` True) — the instrument's zero, re-verified in this run.
3. Base-point cell must reproduce the known result: closed arm flips `prolet` → `comrade`,
   ceiling silent, `diff_over_drift` (recomputed) within 2% relative of 1.204e-01.
4. No conclusions from any cell with `clipped` True or `nonfinite` True; such cells are
   kept in the table as diagnostics, marked.

## Compute and resumability

Nine routed cells + one severed; each roughly EXP-001-cell-sized (minutes on CPU), with
the 240-step and 4× -eta cells the slowest. Records are appended to
`t2_1_coupling.jsonl` as they complete; `--resume` skips completed cell ids.
