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
- **Cadence: `k=1`. Apply every iteration. Do not sweep it.**
- **Log per iteration:** cos at **lag 1 and lag 2**, L2(A,B), p(top1) and entropy in
  both phases, and `delta_norm` / `delta_frac` / `clipped` / `nonfinite`.

### Why cadence is fixed at 1

There is a real aliasing trap here — the same one that hid F9 for months, one level
up. Applying the update every `k=2` iterations on a period-2 cycle drives every update
from phase A's activations and never phase B's: not plasticity on the cycle, but
plasticity on half of it, biased along whatever direction separates the phases.

The answer is not to reason carefully about which `k` is safe. It is to not create the
problem: **`k=1` cannot alias against anything**, and it is also the tightest coupling
of the two loops, which is the regime this experiment is about. Cadence is a free
parameter with a real biological reading (the ratio of fast to slow dynamics) and it
deserves a sweep — but not here, and not before a single-cadence result exists to
compare a sweep against.

If cadence is ever swept, `k=2` on a period-2 cycle is the one value that must be read
as a phase-locked special case rather than a point on a curve.

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
   is the one that decides whether the branch is interesting.** Two known subtleties,
   both of which have to be handled or C2's verdict is not worth much:

   - The norm match is per-update, and accumulated Oja steps are correlated where
     random ones are a walk, so cumulative drift diverges with iteration count
     (ratio ~1/√2 after two applies, measured). Compare at matched *cumulative*
     `delta_frac`, not just at matched eta.
   - **`c2_random_direction` never plumbs `seed` through** — both arms construct
     `OjaPlasticity` without it, so the random arm always draws with `seed=0`. As
     it stands C2 is a single random sample, not a distribution. For a binary
     outcome like "did the cycle break", one draw is not enough: run the random arm
     over several seeds and report how many broke it. `OjaPlasticity` already takes
     the parameter; the control just does not expose it. Fix before running C2.
4. **The fixed-point control:** the same eta on `state_prolet.pt`. If plasticity
   breaks the cycle *and* destroys the fixed point, the result is "perturbing weights
   changes things". If the cycle breaks while the fixed point holds, the result is
   specific to the oscillator.

## 6. What this does not ask

Nothing here touches basin counts, the 125-prompt library, or the collapse question.
Those need the full sweep and are the natural EXP-002. This is the cheapest sharp
question available, chosen because it has an exact answer and an existing baseline
measured to six decimal places.

---

## 7. The offline arm — a required arm, not an optional control

**Added after the prior-art check.** `PRIOR_ART.md`, "The finding that changes our
experiment": Oja's rule converges to the dominant eigenvector of the second-moment
matrix of whatever activations pass through it, **and it does that with no feedback at
all.** The weight matrix will move and the cycle will be perturbed regardless. So every
outcome in §4 — cycle survives, cycle damps, period lengthens, trajectory diverges — is
consistent with the rule simply doing its job on a fixed activation distribution, and
none of them is on its own evidence about the coupling this project is about.

EXP-001 therefore runs **two arms at every eta on the ladder**, not one.

| Arm | What it is |
|:---|:---|
| Closed loop | Run the loop; every `k=1` iterations apply the rule to the activations flowing through right now. The changed `W_out` shapes the next iterate, which shapes the next update. |
| Offline | Run the same loop **frozen**, recording those activations. Replay the recording through the same rule with no feedback. Install the resulting matrix. Re-run the loop frozen. |

**The result is the difference between the arms.** A closed-loop number reported without
its offline partner is not a finding and should not be written down.

### Implementation

`offline_control.py`. It owns no learning rule and no loop: the rule is `OjaPlasticity`
exactly as `plasticity.py` defines it — same object, same `apply()`, same ceiling — and
the loop is whatever `atr_step` is passed in, i.e. `atr_bridge.make_atr_step`.

```python
from atr_bridge import load_state, make_atr_step_from_state
from offline_control import run_matched_arms

state = load_state(".../state_divine.pt")
step  = make_atr_step_from_state(model, state, layer_start=0, layer_end=11)
res   = run_matched_arms(model, state.tensor, step, site="blocks.6.mlp",
                         n_steps=200, eta=1e-5, apply_every=1)
log(res.summary())
```

`tests/test_offline_control.py` is the acceptance suite.

**One outstanding requirement on `plasticity.py`.** The replayer feeds recorded
activations to `OjaPlasticity._hook` — a private method — because the only alternative is
a second implementation of the accumulation step, and a second implementation is one more
axis on which the arms could silently differ. `OjaPlasticity` should grow a public
`observe(x, y)` that `_hook` itself calls, so the offline arm has a supported entry point.
Nothing about the maths changes; it is a rename and a delegation. Until then, a change to
`_hook`'s signature breaks the offline arm silently, and
`test_a_single_step_replays_to_the_same_update_bit_exactly` is what will catch it.

### The matched-axes requirement

For the difference to be about feedback and nothing else, the arms must match on every
axis in `PRIOR_ART.md`'s "must match" table. That table is mechanised as
`offline_control.MATCHED_AXES`, every row is a first-class field on `ArmConfig` filled in
by reading the rule object after the run, and `verify_arms_matched` checks each one:

| Axis | Field |
|:---|:---|
| eta, and the ceiling | `eta`, `max_delta_frac` |
| Total number of weight updates | `n_updates` |
| Order of the activation samples | `sample_order` — the per-sample iteration index, in consumption order |
| Batching of samples per update | `samples_per_update` |
| Initial weight, and RNG seed | `w0_sha256` (byte hash), `seed`, `rng_state_sha256` |
| Centring, or its deliberate absence | `centring`, read off the rule object rather than asserted |

Plus `site`, `mode`, `transposed` and `dtype`, because a comparison across two different
matrices, rules, layouts or precisions is not a comparison.

**On a mismatch, `run_matched_arms` raises `ArmsMismatchError` and reports no comparison.**
That is deliberate. A mismatched number with a caveat attached is how the caveat gets
lost on the way into a write-up.

Two axes are worth reading the small print on:

- **`sample_order` is not "the arms saw the same values."** They cannot — that difference
  *is* the experiment. It is "the arms consumed their samples in the same order, indexed
  the same way, with nothing dropped and nothing repeated." This is why the recorder
  raises `MemoryError` when a recording exceeds its budget instead of thinning it.
- **`centring` currently reads `absent` in both arms**, because `plasticity.py` offers no
  centring option — neither `x` nor `y` is ever mean-subtracted. The axis passes by shared
  absence, which is the honest state of that check and is written into the recorded string
  rather than hidden behind a `True`. It is read off the object, so the day a `centre`
  flag is added, an arm with it and an arm without stop matching automatically. **If
  centring is ever added it must be added to both arms in the same commit** — PRIOR_ART is
  explicit that applying it to one arm alone moves the fixed point by itself.

### Two paths, and which one "no feedback" means

There are two routes by which a weight change reaches the next update, and they are not
the same route:

- **State feedback.** `W` changes → the loop's next iterate changes → the next `x`
  changes. This is the coupling EXP-001 is about.
- **The rule's own recursion.** `W` changes → `y = x W_out + b` changes → the next update
  changes, with `x` held fixed. This is internal to Oja and is present in ordinary offline
  Oja on a fixed dataset.

Replaying the *recorded* `y` (`y_source="recorded"`, the default, and what PRIOR_ART
specifies literally) freezes both. Replaying with `y` recomputed from the recorded `x` and
the offline arm's own drifting weight (`y_source="recomputed"`) freezes the state feedback
only. `run_matched_arms` measures both by default; they answer different questions and the
gap between them is the floor below which a `recorded`-mode divergence is not about
feedback.

### The detection limit — read this before believing any difference

`tests/test_offline_control.py::test_a_site_the_loop_does_not_route_through` measures the
floor directly. It reads the loop out at `blocks.3.hook_resid_post`, below the site, so
**no state feedback exists at all** and the `x` reaching the rule is bit-identical in both
arms at every step. Whatever the arms still differ by there is noise, and no claim about
feedback can be made underneath it.

| Mode | Floor, feedback severed |
|:---|:---|
| `y_source="recomputed"` | **Zero — bit-identical.** Asserted as a bound (`rel_fro_diff < 1e-8`) *and* as `torch.equal`. |
| `y_source="recorded"` | **6.8% of drift** at eta=1e-5 over 6 steps (`rel_fro_diff` 9.3e-4 against drift 1.3e-2). Not noise: it is Oja's own `y = xW` recursion, frozen by the recording. |

That second row is the one to keep in view. In the default mode, a `diff_over_drift` of a
few percent at a routed site is **inside the floor** and is not evidence of feedback. Use
`weight_recomputed_y` for the feedback claim and `weight` for the literal PRIOR_ART
protocol, and report both.

The zero floor is a defended property, not luck. It holds only because
`offline_control._recompute_y` reproduces the site's **fused `torch.addmm`** bit for bit
rather than merely mathematically — TransformerLens's `batch_addmm` flattens to 2-D and
calls `torch.addmm`, written that way to match HuggingFace's `Conv1D`. The obvious
`x @ W + b` differs in the last bits (max|Δ| 1.9e-06 on `y` at this site), propagates into
every offline update, and compounds: it put a floor of roughly **5e-09 relative Frobenius**
under the entire comparison. That floor was hardware-dependent — 2.9e-09 on one machine and
4.8e-09 on a GitHub Actions runner, same commit and same seed — which is the same class of
trap as `initial_norm` reading 1468.48828125 on CI against 1468.4886474609375 locally.

**Never assert a value for a floating-point floor; assert a bound.** The test does, and
`test_recompute_y_reproduces_the_sites_own_forward_bit_exactly` pins the root cause
separately so a regression names itself.

The eta = 0 identity is a different and stronger kind of exactness — `0 × anything = 0`,
so it is exact by construction on any hardware — and is asserted with `torch.equal`.

### Acceptance, before any eta > 0 number is believed

1. **eta = 0 → the two arms produce bit-identical matrices**, and bit-identical
   trajectories under them. This is C0 for the harness itself. `torch.equal`, not
   `allclose`.
2. **A site the loop does not route through → the `recomputed` arms are bit-identical.**
   With the feedback path severed there is nothing left to differ by.
3. Only then, eta > 0: report `cos_delta` (cosine between the two arms' *changes* — the
   cosine between the full matrices is ~1 by construction and says nothing),
   `rel_fro_diff = ‖W_closed − W_offline‖_F / ‖W_0‖_F`, and `diff_over_drift`, the same
   difference against the larger of the two arms' own drift.

### Reading the outcome

Add a row to §4:

| Outcome | Reading |
|:---|:---|
| The arms agree (difference at or below the no-route floor) | **Feedback contributes nothing detectable at this eta.** This is a real result about this substrate and it is what the reservoir literature in PRIOR_ART would predict. It is not a failed run, and nothing may be tuned to make the number larger. |

That last clause is the point of the arm. `diff_over_drift ≈ 0` at every eta on the ladder
would say the closed-loop result is Oja finding the dominant direction of the frozen
loop's activations, which is what Oja does anywhere. Reporting it is the difference
between a finding and an artefact.
