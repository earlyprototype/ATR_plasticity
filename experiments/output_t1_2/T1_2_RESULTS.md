# T1.2 — the up-then-down α hysteresis sweep

**Issue #46.** The independent cross-check named by `CLAIMS.md` C-52 (and, through it, C-26).
Pre-registration in `PREREGISTRATION.md`, timestamped before the run. Runner
`t1_2_hysteresis.py`; raw records `t1_2_hysteresis.jsonl`; config/verdict `meta.json`.

Agent: magenta. `gpt2-small`, CPU, float32 activations, **all norms float64**. repo_rev
`fe40683`, transformer-lens `3.6.0`, torch `2.13.0+cpu`. Reuses the repo's own machinery
(`basin_bifurcation`, `plasticity.OjaPlasticity`, `atr_bridge`); the from-scratch D2 α-loop
in `basin_bifurcation.py` is left untouched as the disclosed contrast.

## 1. Pre-registered interpretation (fixed before the run)

- **A gap** between the up-threshold (`prolet`→`comrade`) and the down-threshold
  (`comrade`→`prolet`) on the same grid = **hysteresis** = coexisting resting states →
  answers C-52 "yes", would *tension* C-26's retirement.
- **A clean retrace** (up and down give the same basin map) = one continuously-moving fixed
  point the readout relabels → answers C-52 "no", **corroborates C-26 `retired`**.

## 2. The working point + gates (all passed)

Site `blocks.6.mlp`, `hebb`, eta = 7.065171428571429e-05, cadence 1, max_delta_frac 0.05,
seed 0, prompt `A01_physics`. ΔW is the working-point edit, reproduced in memory.

| gate | result |
|---|---|
| episode reproduces `comrade` (47998) | ok, delta_frac(f64) = 0.011239339962675624, σ₁ = 1.8135175, **0/120 clip** |
| ‖W0‖_F (float64) | 164.85407309107723 (= reference) |
| frozen `prolet` seed restores to W0 | bit-exact (relL2 = 0.0) |
| α=0 bit-identity gate | **bit-identical**, max abs diff 0.000e+00, weight == W0 |

## 3. Result — continuation sweep, step 0.10, α = 0.0 → 1.5 → 0.0

Each α is seeded from the **previous** α's settled state (continuation), not from a fresh
iteration-0. Settled word per grid point:

| direction | `prolet` | `comrade` | threshold |
|---|---|---|---|
| **UP** (0.0→1.5) | α ≤ 0.4 | α ≥ 0.5 | flips at α = **0.5** |
| **DOWN** (1.5→0.0) | α ≤ 0.4 | α ≥ 0.5 | flips back at α = **0.4** |

**The up and down sweeps give the identical α→basin map** — `prolet` for α ≤ 0.4 and
`comrade` for α ≥ 0.5 in *both* directions. Every grid point agrees between the two
directions; there is no α at which the settled basin differs by sweep direction.

- `alpha_up = 0.5`, `alpha_down = 0.4`, **gap = 0.1 = exactly one grid step**. The
  transition straddles a single threshold near α ≈ 0.45; the up-flip lands in the first grid
  cell above it (0.5) and the down-flip in the first cell below (0.4). That 0.1 is grid
  discretisation of one threshold, **not a resolved loop**.
- Every settled state is a fixed point: `final_lag1_cos` = 1.000000 at every α in both
  directions (no period-2 in the 0–1.5 range for this prompt/shell under continuation).
- The crossing is smooth: at α = 0.5 (up) the state settles slowly (settle_step 43, final
  margin 0.020 — sitting on the `prolet`/`comrade` ridge); away from the threshold it settles
  in 1–2 steps. This mirrors T1.1's smooth ridge-crossing.

**Verdict (primary): CLEAN RETRACE (gap ≤ one grid step).**

## 4. Robustness — fixed-W0 shell (D1 shell)

Re-running with the fixed-W0 renormalisation shell instead of the per-α shell gives the
**same** basin map (`prolet` α ≤ 0.4, `comrade` α ≥ 0.5), `alpha_up = 0.5`,
`alpha_down = 0.4`, gap 0.1. `robustness_agrees = true`.

**Verdict (robust): CLEAN RETRACE.** The verdict is invariant to the shell choice.

## 5. Verdict for the register

**C-52 (open): "Does the α-sweep show hysteresis?" — answered NO.** The continuation
up-then-down sweep retraces exactly in both shells; there is no hysteresis loop. This is the
pre-registered "against created-attractor" outcome: consistent with a **single fixed point
that the edit relocates continuously**, not two coexisting resting states.

This **independently corroborates C-26 (`retired`)** and **C-56 (`supported`)** from T1.1 by
the second, different route the register asked for — agreement between the two is worth more
than either alone.

## 6. Caveats that must travel

1. **Grid resolution.** The transition is localised to α ∈ (0.4, 0.5) at step 0.10. Because
   up and down agree at every sampled point, no hysteresis is resolvable — but a loop
   *narrower than one grid step*, entirely inside (0.4, 0.5), is not excluded by this grid. A
   finer sweep around α ∈ [0.4, 0.5] would be needed to exclude a sub-0.10 loop. The data at
   the sampled points show no loop.
2. **No period-2 observed.** All cells settled to fixed points (lag-1 = 1.0) across 0–1.5,
   including the high-α region where the from-scratch D2 sweep reported a lag-1 collapse. The
   difference is plausibly the continuation seeding (each α starts near the previous fixed
   point) or is prompt/shell-dependent; it is noted, not interpreted here.
3. **Scope.** n = 1 prompt (`A01_physics`), 1 site (`blocks.6.mlp`), 1 eta — the working
   point, not a sweep over prompts. Same scope as T1.1.

## 7. Files

- `PREREGISTRATION.md` — the gap-vs-retrace interpretation, timestamped before the run.
- `t1_2_hysteresis.py` — runner (reuses repo machinery; edits no shared source).
- `t1_2_hysteresis.jsonl` — 64 α-cell summaries (primary/robust × up/down × 16) + per-step records.
- `meta.json` — config, gates, primary + robust verdicts, `robustness_agrees`.
