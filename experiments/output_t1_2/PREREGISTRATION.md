# T1.2 pre-registration (fixed in writing BEFORE any sweep was observed)

Timestamp (UTC): 2026-08-01T17:35:51Z
Agent: magenta. Issue #46 / ALIGNMENT_REVIEW.md T1.2. Settles CLAIMS **C-52**; independent
cross-check on **C-26** (retired by T1.1, #45).

## The test
The **up-then-down α-sweep** for hysteresis. Install `W0 + α·ΔW` as a **static** weight change
at site `blocks.6.mlp` (= `transformer.h.6.mlp.c_proj`, W_out, shape (3072, 768)) for α on a
grid straddling 1.0. Seed each α from the **previous α's settled state** (continuation), sweep
α **up** then **down** over the *same* grid, and read the settled basin at each α both
directions. Whether the up-threshold (`prolet`→`comrade`) and the down-threshold
(`comrade`→`prolet`) land on the same grid point (clean retrace) or are separated by a gap
(hysteresis) is the whole measurement.

ΔW is the working-point edit: `hebb`, eta = 7.065171428571429e-05, cadence 1,
max_delta_frac = 0.05, seed 0, 120-step closed episode on `A01_physics`. Relative weight
change ‖ΔW‖_F / ‖W0‖_F = 0.011239... (≈ 0.0112). Reference ‖W0‖_F = 164.85407309107723. All
norms in float64 (rule 5). ΔW is reproduced in memory (the repo persists no ΔW) and the
reproduction is gated on reaching `comrade` (47998) with 0.0% clip (rule 2), matching EXP-001.
T1.2 edits **no shared source**: `basin_bifurcation.py:401-406` (the from-scratch D2 α-loop)
stays untouched as the disclosed contrast; this runner reuses `bif.*` config + helpers only.

## (a) Grid
α = 0.00, 0.05, 0.10, …, 1.50 — **step 0.05, 31 points**. The up sweep runs this grid
ascending; the down sweep runs the identical grid descending. (Contingency, per the standing
rules: if wall-clock risk means step-0.05 will not complete, the fallback is the **step-0.10
subgrid** — the strict subset α = 0.00, 0.10, …, 1.50, 16 points — run up then down. This
changes only threshold resolution (±0.10 vs ±0.05), not the interpretation, and the run will
state which grid produced the reported numbers.)

## (b) Shell rule
**Primary shell:** per-α `initial_norm` from a **fresh forward pass under `W0 + α·ΔW`** via
`bif.initial_state` (the D2 protocol). This `initial_norm` is a **pure deterministic function
of α** — the weights `W0 + α·ΔW` fully determine the forward pass — so it takes the **identical
value whether α is reached on the up sweep or the down sweep**. The shell therefore *cannot*
fabricate a difference between the up-threshold and the down-threshold: any gap must come from
the continuation **seed** (a path-dependent basin), which is the genuine signature of
coexisting resting states.

**Robustness variant:** a **fixed-W0 shell** — `initial_norm` held at the W0 forward-pass value
(the D1 / T1.1-primary shell, ≈ 1289.226318) for **every α, up and down**. The verdict (gap vs
retrace) **must be invariant** to this choice.

## (c) Seed rule — continuation
- **UP sweep:** α ascending. The first α (α = 0.00) is seeded from the model's **original
  frozen `prolet` settled state** — the state the frozen (W0) loop settles on for `A01_physics`
  before any weights moved. Each subsequent (higher) α is seeded from the **previous, lower α's
  settled state**.
- **DOWN sweep:** α descending from the top of the grid. The first (highest) α is seeded from
  the **top-α settled state carried over from the end of the up sweep**. Each subsequent
  (lower) α is seeded from the **previous, higher α's settled state**.

Continuation is what makes this a hysteresis test: with the shell pinned to a pure function of
α, the seed is the *only* channel through which the sweep direction can matter.

## Pre-registered two-outcome interpretation (the whole point of writing it first)

Define, on the same grid:
- `alpha_up`  = the **first ascending** α whose settled word == `comrade` (22758→47998 flip up);
- `alpha_down`= the **first descending** α whose settled word flips back to `prolet` (47998→22758).
- `gap` = `alpha_up − alpha_down`.

- **A GAP** — `alpha_up` strictly greater than `alpha_down` **by more than one grid step** —
  = **HYSTERESIS**: over the band α ∈ (`alpha_down`, `alpha_up`) the settled basin depends on
  the sweep direction, i.e. **two resting states coexist** for that band of α. => answers C-52
  **"yes, the α-sweep shows hysteresis"**; this would **tension C-26's retirement** — a genuine
  bistable band is exactly the coexistence the created-attractor (step-4) reading needed, and
  would be a real tension worth surfacing.

- **A CLEAN RETRACE** — `alpha_up == alpha_down` (same grid point both ways), or the two differ
  by **≤ one grid step** (grid-localization only) — = **one continuously-moving fixed point**
  that the readout relabels at the same crossing regardless of direction. => answers C-52
  **"no, no hysteresis"**; this **corroborates C-26 `retired`** (one displaced attractor, ladder
  step 2, consistent with T1.1/C-56 and C-27's smooth-logits/discrete-argmax reading).

- **Any settled word neither `prolet` nor `comrade`** at a threshold cell: **report it
  verbatim**; it is its own finding and needs its own reading.

## GATE — α = 0 zero-step-size control (recorded before interpreting anything; rule 2/7)
Install `W0 + 0·ΔW` through the **same** write path as the real edit (`plasticity._site.write`),
seed from the frozen `prolet` state, iterate 30 steps, and compare bit-for-bit against the
frozen (unedited W0) loop from the same state. Must be **bit-identical** (max abs difference
exactly 0.0 at every position, every step; `torch.equal` per step). If not: STOP, report as a
finding, do not interpret T1.2.

## Settling / thresholds (rule 2)
A settled word is the basin held constant over the **last 15** of the 120 steps
(`_settle_step`). Near α ∈ (1.25, 1.50) the loop's lag-1 cosine is known to collapse
(fixed-point → period-2); **unsettled (period-2) cells are never quoted as thresholds**. The
primary threshold target stays in the low-α fixed-point range. `alpha_up` / `alpha_down` are
grid-localized to ±(one grid step).

## Rule 3 (not applicable)
The severed-path 0.0 feedback floor is **not** the null hypothesis here. T1.2 is about
coexistence of resting states under a static weight edit (no plastic feedback during the
sweep), so the severed-path control does not bear on this experiment. Noted as not-applicable.

## No tuning (rule 7)
No parameter is tuned to any basin percentage or target outcome. eta, cadence, prompt, site,
seed, step count, and the ΔW-producing episode are all fixed to the working-point values above;
the sweep loop is bit-exact against the parent loop body (the α = 0 gate proves it).
