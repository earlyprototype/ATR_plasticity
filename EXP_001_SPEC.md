# EXP-001 — Does the `Divine` period-2 cycle survive plasticity?

**Status:** proposed, not run. Blocked on EXP-000 below.
**Model:** GPT-2 Small, TransformerLens, layers 0→11.
**Cost:** hours on a laptop CPU, not days.

---

## 0. Read this first: the two repos do not currently compose

This was checked, not assumed. On a TransformerLens `HookedTransformer`:

```
candidate_sites(model)                      -> 0 sites
candidate_sites(model, prefix="blocks")     -> 0 sites
OjaPlasticity(m, site="transformer.h.6.mlp.c_proj")
    -> AttributeError: 'HookedTransformer' object has no attribute 'transformer'
OjaPlasticity(m, site="blocks.6.mlp.W_out")
    -> TypeError: no 2-D .weight; not a supported target
```

`plasticity.py` was written against HuggingFace GPT-2 module naming — dotted paths
ending in a module with a 2-D `.weight`. The ATR engine runs TransformerLens, where
the MLP output matrix is a bare `nn.Parameter` named `W_out` hanging directly off
`blocks.{L}.mlp`. There is no module to `register_forward_hook`, and no `.weight`.

There is also no per-step entry point. `atr_engine.run_atr_loop(model, prompt, ...)`
runs the whole loop internally, with its own injection hook; `controls.py` expects
`atr_step(model, r) -> r_next`.

Neither gap is deep. Both must close before any number here means anything.

## EXP-000 — The bridge (do this first)

**A. A TransformerLens site adapter.** The pre- and post-synaptic activity for
`W_out` is exactly recoverable from two TL hook points, verified:

| Quantity | Where | Shape |
|:---|:---|:---|
| `x` (pre-synaptic) | `blocks.{L}.mlp.hook_post` | (seq, 3072) |
| `y` (post-synaptic) | `blocks.{L}.hook_mlp_out` | (seq, 768) |
| `W_out` | `blocks.{L}.mlp.W_out` | (3072, 768) |

`y == x @ W_out + b_out` holds to 3.8e-06 (float32). **`W_out` is already in the
`(n_in, n_out)` convention the learning rules are written in**, so `_hebb_term` and
`_oja_decay` carry over unchanged — the adapter is plumbing, not maths.

**B. A per-step engine.** Factor `atr_step(model, r) -> r_next` out of
`run_atr_loop` without changing its behaviour: inject at `blocks.0.hook_resid_pre`,
read at `blocks.11.hook_resid_post`, rescale to the *initial* norm (`x · ‖x₀‖/‖x‖`),
not to unit norm. Copy it; do not reimplement it. Acceptance test: the factored step,
iterated, reproduces `run_atr_loop`'s trajectory bit-exactly for 20 iterations.

**C. The controls, on that stack.** C0 must pass bit-exactly with the adapter
installed at eta=0 before anything below is run. Note the known caveat: C0 has been
seen to fail intermittently at ~1e-4 on CPU and could not be reproduced in 80
repeats — reproduce against an unhooked control before suspecting the hooks.

---

## 1. The question

Parent finding F9: the `Divine` attractor is not a fixed point and not a wandering
orbit. It is an **exact period-2 limit cycle** — the tensor alternates between two
states A and B, reproduced to machine precision:

| | |
|:---|:---|
| Prompt (`Syntactic`) | `"The cat sat on the mat and then the"` |
| cos(A, B) | 0.6849 |
| L2(A, B) | 1249.43 (against a last-vector norm of 1612) |
| cos(A, f(f(A))) | **1.000000** |
| Phase A readout | `Divine`, p = 0.5046, entropy 3.0500 |
| Phase B readout | `Divine`, p = 0.2252, entropy 4.6175 |

This repo's own H3 says the cycle should be fragile. **EXP-001 asks whether a local
Hebbian update to one weight matrix, driven by the activations of the cycle itself,
destroys it.**

## 2. Why this experiment before the basin sweep

- **The readout is binary and exact.** `cos(A, f(f(A))) = 1.000000` either survives
  or it does not. No statistics, no 125-prompt sweep, no basin-classification
  threshold to argue about. A single number to six decimal places.
- **It is one prompt and one site.** Cheap enough to sweep eta properly.
- **The starting state already exists.** `state_divine.pt` in the parent repo holds
  the iteration-1000 tensor (10, 768) with `initial_norm` 1468.4886474609375 — sitting
  *on* the cycle. No 1000-iteration warm-up.
- **It comes with a matched control object.** `state_prolet.pt` is the `Semantic`
  prompt's fixed point (per-step L2 ~3e-4, the numerical floor) at identical settings.
  Two dynamical objects, one run configuration: an oscillator and a fixed point.
- **The falsifier is interesting either way.** A cycle that survives a modification
  of the very map that generates it is a stronger result than one that breaks.

## 3. Design

- **Site:** `blocks.6.mlp` → `W_out`. Mid-stack, MLP down-projection: this repo's own
  first choice, and the least entangled place to perturb.
- **Start:** the saved iteration-1000 state, not a fresh prompt run.
- **eta ladder:** 0 (C0), then 1e-6, 3e-6, 1e-5, 3e-5, 1e-4. Half-orders.
- **Horizon:** 200 iterations per eta. The cycle is locked in from ~250 and stable to
  1000, so 200 is ample to see it break or hold.
- **Cadence:** see the trap below.
- **Log per iteration:** cos at **lag 1 and lag 2**, L2(A,B), p(top1) and entropy in
  both phases, and `delta_norm` / `delta_frac` / `clipped` / `nonfinite`.

### The cadence trap — the reason this needs saying out loud

F9 was hidden for months because every snapshot schedule sampled even iterations, and
an even-only schedule samples a period-2 orbit at one phase.

**The same trap now exists one level up, in the plasticity schedule.** Applying the
weight update every `k=2` iterations on a period-2 cycle means every update is driven
by phase A's activations and never phase B's. That is not "plasticity on the cycle";
it is plasticity on half of it, and it would bias the weight change along whatever
direction distinguishes the phases.

**Use odd cadence (k=1, or k=3).** If k=2 is run at all, run it as a deliberate
contrast — "updating in phase" versus "updating across phases" is itself a real
question, and F10 says the A↔B flip is a rank-1 self-negating mode, so the two
cadences may pull in opposite directions.

## 4. Outcomes, and what each would mean

| Outcome | Reading |
|:---|:---|
| lag-2 cosine stays 1.000000 | The cycle is robust to modification of the map that generates it. The striking result. |
| Cycle → fixed point | The oscillation is damped out; plasticity acts as a brake. Consolidation, in this repo's H1 sense. |
| Cycle → longer period, or wanders | The update perturbs the flip axis without destroying the orbit. Measure the new period with a lag scan, not a lag-1 gate. |
| Whole trajectory diverges | eta too high, or the ceiling is doing work — check `clipped` before interpreting anything. |

**Kill conditions.** The experiment is void, not interesting, if: C0 fails on the real
stack; `clipped` fires before the cycle changes; or the norm-matched random control
(C2) breaks the cycle the same way Oja does — that last one would mean the finding is
about perturbation magnitude, not about Hebbian learning, and the branch as framed is
dead.

## 5. Controls, in order

1. **C0** at eta=0 on the real stack. Bit-exact. Gate.
2. **C1** revert — the cycle must return to `cos(A, f(f(A))) = 1.000000` after
   `revert()`, from the same saved state.
3. **C2** norm-matched random direction at whatever eta first moves the cycle. **This
   is the one that decides whether the branch is interesting.** Note the known
   subtlety: the norm match is per-update, and accumulated Oja steps are correlated
   where random ones are a walk, so cumulative drift diverges with iteration count —
   compare at matched *cumulative* `delta_frac`, not just matched eta.
4. **The fixed-point control:** the same eta on `state_prolet.pt`. If plasticity
   breaks the cycle *and* destroys the fixed point, the result is "perturbing weights
   changes things". If the cycle breaks while the fixed point holds, the result is
   specific to the oscillator.

## 6. What this does not ask

Nothing here touches basin counts, the 125-prompt library, or the collapse question.
Those need the full sweep and are the natural EXP-002. This is the cheapest sharp
question available, chosen because it has an exact answer and an existing baseline
measured to six decimal places.
