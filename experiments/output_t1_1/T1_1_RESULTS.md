# T1.1 — The coexistence test

**Issue #45.** Named by `CLAIMS.md` C-26 as *"the decisive test"*, by C-51 as its open row,
and by `ALIGNMENT_REVIEW.md` / `BASIN_BIFURCATION.md` as the one unrun experiment that would
settle whether `comrade` is a *created* attractor. This file reports it.

Agent: magenta. Model cache warm. `gpt2-small`, CPU, float32 activations, **all norms
float64**. repo_rev `449bf1a`, transformer-lens `3.6.0`, torch `2.13.0+cpu`.

Reuses the repo's own machinery, reimplementing nothing: `atr_bridge.make_atr_step` /
`initial_state` (the parent loop body, verbatim), `plasticity.OjaPlasticity` (the `hebb` rule
+ ΔW accumulation + ceiling), `baseline_basins.readout_detail` (the exact basin classifier),
and `experiments/basin_bifurcation.py`'s working-point config and helpers. The runner is
`t1_1_coexistence.py` (this directory).

---

## 1. Pre-registered interpretation (fixed in writing BEFORE the trajectory was observed)

Recorded in `PREREGISTRATION.md`, timestamp 2026-08-01T16:22:12Z, before any T1.1 iterate
existed. Verbatim two-outcome map:

- **STAYS at `prolet` (22758)** → two resting states coexist under `W0 + ΔW`: `prolet` is
  still a fixed point *and* `comrade` is a fixed point → **created attractor** (issue #25
  ladder step 4, a bifurcation) → moves C-26 toward `supported`; C-51 answered "yes, stays".
- **MOVES to `comrade` (47998)** → the edited map has a single displaced attractor; the
  `prolet` fixed point did not survive the edit → **boundary / displacement move** (ladder
  step 2, one continuously-relocated fixed point the readout relabels) → **C-26 refuted**;
  C-51 answered "no".
- **Any other settled word** → report verbatim; a finding needing its own reading.

The result below is reported per this map. Nothing was tuned to any basin percentage or
target outcome; every loop parameter is the working point.

## 2. The working point (verbatim, from `basin_bifurcation.py` / EXP-001)

| field | value |
|---|---|
| site | `blocks.6.mlp` = `transformer.h.6.mlp.c_proj` (W_out, (3072, 768)) |
| rule / eta | `hebb`, eta = 7.065171428571429e-05 |
| cadence / max_delta_frac / seed | 1 / 0.05 / 0 |
| prompt | `A01_physics` — "The implications of quantum entanglement suggest that" |
| episode producing ΔW | 120 closed steps, cadence 1, apply after every step |
| T1.1 loop length | 120 steps (per ALIGNMENT_REVIEW T1.1) |

**ΔW reproduction fidelity** (reproduced in memory — the repo persists no ΔW):

| quantity | this run | anchor (EXP-001) | match |
|---|---|---|---|
| ‖W0‖_F (float64) | 164.85407309107723 | 164.85407309107723 | exact |
| ‖ΔW‖_F / ‖W0‖_F | 0.011239339962675624 | 0.011239339962675624 | exact |
| ΔW σ₁ (float64) | 1.8135175021312695 | 1.813517498340477 | ~8 figs |
| episode closed basin | `comrade` (47998) | `comrade` (47998) | match |
| clip / nonfinite | **0/120 clipped, 0 nonfinite** | 0.0% clip | match |

The ΔW-producing episode is a **clean, unclipped** result (clip rate 0/120 = 0.0%,
`clipped=False`, `nonfinite=False`). T1.1's own loop installs ΔW as a **static** edit and
applies no plasticity, so no weight update occurs during the trajectory and clipping is not
reachable there.

## 3. GATE — zero-step-size control (recorded before interpreting anything)

Install `W0 + 0·ΔW` through the **same** write path as the real edit
(`plasticity._site.write`), seed from the frozen `prolet` state, iterate 30 steps, and
compare bit-for-bit against the frozen (unedited W0) loop from the same state.

| gate check | result |
|---|---|
| trajectory bit-identical to frozen loop (30 steps, all positions) | **YES** |
| max abs difference over all steps/positions | **0.0** (exact) |
| every step `torch.equal` | True |
| `W0 + 0·ΔW` weight bit-identical to `W0` | True |

**GATE PASSES.** The installation harness adds nothing; the eta=0 loop is the frozen model.
Interpretation of T1.1 proceeds.

Independent restore check: after reverting the episode, the live matrix returns to `W0` with
relative-L2 = 0.0 (bit-identical). The frozen `prolet` seed state is `prolet` (22758), top-5
`[prolet, bourgeois, Anarch, Marx, comrade]`, ‖state‖ = 4782.779.

## 4. Result — seed the frozen `prolet` state under `W0 + ΔW`, iterate 120 steps

**The loop MOVES to `comrade`.** It does not stay at `prolet`.

| quantity | primary (shell = `init_norm`) | robustness (shell = edited system's own `init_norm`) |
|---|---|---|
| settled word | **`comrade` (47998)** | **`comrade` (47998)** |
| settle step (locks in, holds to 120) | **12** | **12** |
| first leave from `prolet` | iter **12** → `comrade` | iter **12** → `comrade` |
| final lag-1 cosine | **0.9999998807907104** | 1.0000005960464478 |
| final lag-2 cosine | **0.9999999403953552** | 1.0000004768371582 |
| settled margin (top1−top2) | 0.32138 | 0.31965 |
| `comrade` at settled state | **rank 1**, logit 16.3047 | rank 1, logit 16.3069 |
| `prolet` at settled state | rank 3, logit 15.5759 | rank 3, logit 15.5767 |
| settled top-5 | `[comrade, congress, prolet, comrades, proletarian]` | same |

Renorm shell: primary uses `init_norm` = 1289.226318 (the W0 forward-pass norm — the same
map that produced the frozen `prolet` state and the `comrade` episode; makes the gate
bit-identical; the exact dual of `BASIN_BIFURCATION.md` D1). Robustness uses the edited
system's own shell 1289.972778. **The verdict is invariant to this choice** — both settle at
`comrade`, step 12, lag-1 ≈ 1.0.

### The motion is a smooth ridge-crossing, not a jump (the D1 signature, mirrored)

The state's own motion is slow and monotone the whole way: **lag-1 cosine ≥ 0.99990618 at
every one of the 120 steps** (minimum at iter 1; ≥ 0.999992 by iter 4). There is no
discontinuity for the argmax to ride. What crosses is the readout:

| iter | basin | lag-1 | margin | `comrade`/`prolet` |
|---|---|---|---|---|
| 1 | `prolet` | 0.9999062 | 0.2601 | — |
| 5 | `prolet` | 0.9999861 | 0.1686 | — |
| 10 | `prolet` | 0.9999911 | 0.0449 | — |
| 11 | `prolet` | 0.9999921 | 0.0226 | — |
| **12** | **`comrade`** | 0.9999927 | **0.00025** | crossing |
| 13 | `comrade` | 0.9999936 | 0.0237 | — |
| 20 | `comrade` | 0.9999960 | 0.1916 | — |
| 120 | `comrade` | 0.9999999 | 0.3214 | rank 1 / rank 3 |

The margin falls smoothly to essentially zero at iter 12 (0.00025 — the state sitting exactly
on the `prolet`/`comrade` ridge), then rises again as `comrade`. `cos→seed` decays smoothly
1.000000 → 0.990560 and relative-L2 from the seed grows to 0.13910: the state relaxes off the
frozen `prolet` state and onto a **new** fixed point, `comrade`, whose lag-1 = 1.0 confirms it
is a fixed point (not a cycle). The `prolet` label sat on a thin ledge the edited map slides
straight off — the exact mirror of D1, where the *frozen* map slid the `comrade` state off at
iter 4 back into `prolet`.

## 5. Verdict

**C-51 (open):** *Under `W0 + ΔW`, does the original `prolet` state stay put?* — **NO.** It
moves to `comrade`, settling at iteration 12 and holding through 120, with the state creeping
smoothly (lag-1 ≥ 0.99990 throughout). Answered.

**C-26 (`comrade` is a created attractor — step 4, a bifurcation):** **REFUTED by its own
nominated decisive test.** The created-attractor reading required the two resting states to
coexist under `W0 + ΔW` — i.e. the original `prolet` state to stay put. It does not. Under the
edited weights there is **one displaced attractor**, `comrade`, and the frozen `prolet` state
falls into it. This is ladder **step 2**: a single fixed point that the edit relocates
continuously and the readout relabels across a ridge (consistent with C-27's smooth-logits /
discrete-argmax reading and with the α-sweep's continuously-moving settled branch). C-26 moves
from `not-established` to `retired`.

Plain language: **installing the working-point edit does not add a `comrade` basin beside a
surviving `prolet` basin. It moves the one basin.** The 0.0112 edit displaces the settled word
`prolet` → `comrade` by relocating the single attractor, not by creating a second one.

## 6. Caveats that must travel with this result

1. **Scope of "one displaced attractor".** T1.1 tests the exact coexistence claim C-26 rested
   on — that the *original frozen `prolet` state* stays put under `W0 + ΔW` — and refutes it.
   It does not exhaustively prove no other `prolet` fixed point exists elsewhere in state
   space. **T1.2** (the α up-then-down hysteresis sweep, C-52) is the pre-registered
   independent cross-check; retracing → smooth deformation (step 2 confirmed), a hysteresis
   loop → a real transition. T1.1 alone settles the named test; T1.2 audits the wider claim.
2. **`comrade` and `prolet` are close at the settled state** (margin 0.321; `prolet` still
   rank 3 at logit 15.58 vs `comrade` 16.30). The `comrade` fixed point is real (lag-1 = 1.0)
   but sits near `prolet`, consistent with **C-07**'s basin-resolution limit (within-`prolet`
   spread 3.319e-03 ≳ the `Anarch`–`prolet` gap 2.874e-03). "Displaced attractor" is the
   dynamical reading; it does not upgrade the basin taxonomy's resolution.
3. **n = 1 prompt (`A01_physics`), 1 site (`blocks.6.mlp`), 1 eta.** The working point, not a
   sweep. C-54 (the 125-prompt library at the working point) is the breadth test.
4. **Reproduced, not loaded.** ΔW is regenerated from the frozen episode; the weight-space
   anchors match EXP-001 to ~8–12 figures (§2). The verdict is invariant to the renorm-shell
   choice (§4) and the basin label is invariant to the ~0.1%-class state-norm drift noted in
   `BASIN_BIFURCATION.md`.

## 7. Files

- `PREREGISTRATION.md` — the two-outcome map, timestamped before the run.
- `t1_1_coexistence.py` — the runner (reuses repo machinery only).
- `t1_1_trajectory.jsonl` — raw per-step record: episode reproduction, frozen `prolet` state,
  eta=0 gate, and all 240 T1.1 steps (120 primary + 120 robust), with basin, lag-1/lag-2,
  cos→seed, relL2→seed, margin, state norm per step.
- `meta.json` — full config (site, eta, cadence, prompt, step count, seed, ‖ΔW‖_F/‖W0‖_F,
  transformer-lens version via `importlib.metadata`, repo_rev, gate outcome, verdict).
- `PROPOSED_CLAIMS_PATCH.md` — the exact C-51 / C-26 edits for orchestrator review.
