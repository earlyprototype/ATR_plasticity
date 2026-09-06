# EXP-001 — Does the `Divine` period-2 cycle survive plasticity?

**Status:** the experiment described here is **proposed and still not run**. EXP-000 below is
built and CI-verified, so it is no longer blocked by that.

> **The label `EXP-001` is used for two different experiments, and this file is one of them.**
> The identifier registry claimed EXP-001 for the question in the title, whether a Hebbian
> update destroys the `Divine` period-2 cycle. That experiment has never run. The run written
> up in `EXP_001_RESULTS.md` is a **different** one: the `hebb` basin-flip and offline-arm
> comparison at `blocks.6.mlp` on `A01_physics` and two other prompts, whose results are
> register rows C-20, C-24 and C-30 to C-33. It took the number later.
>
> Nothing here is renamed, because both names are cited across the repository and the register,
> and picking which one keeps the number is the operator's call, recorded as `HANDOVER.md` §6
> item 1. What this notice fixes is the thing that actually misleads: a reader seeing
> "proposed, not run" above a filename whose results file exists would conclude the results
> file reports this spec. It does not.
>
> **This file remains the reference for two things** regardless of how the naming is settled:
> the seventeen matched axes in §7, which every closed-versus-offline comparison in the
> project is checked against, and §0's account of how the two repositories compose. Its §0 and
> §5.3 were corrected during the alignment review and are right either way.
**Model:** GPT-2 Small, TransformerLens, layers 0→11.
**Cost:** estimated single-digit hours on a laptop CPU.

---

## 0. How the two repos compose

They now compose, and CI keeps them composed. Verified on a TransformerLens
`HookedTransformer`:

```
candidate_sites(model)                     -> ['blocks.0.mlp', ..., 'blocks.11.mlp']
OjaPlasticity(model, site="blocks.6.mlp")  -> attaches, via _TransformerLensMLPSite
```

`plasticity.py` was written against HuggingFace GPT-2 module naming — dotted paths
ending in a module with a 2-D `.weight`. The ATR engine runs TransformerLens, where
the MLP output matrix is a bare `nn.Parameter` named `W_out` hanging directly off
`blocks.{L}.mlp`. There is no module to `register_forward_hook`, and no `.weight`.

There is also no per-step entry point. `atr_engine.run_atr_loop(model, prompt, ...)`
runs the whole loop internally, with its own injection hook; `controls.py` expects
`atr_step(model, r) -> r_next`.

Both gaps are now closed. EXP-000 below records how each half was built and what it
was checked against — a record, not a to-do list.

## EXP-000 — The bridge (built, and CI-verified)

**A. The TransformerLens site adapter — `_TransformerLensMLPSite` in `plasticity.py`.**
The pre- and post-synaptic activity for `W_out` is exactly recoverable from two TL
hook points, and the adapter reads it there — there is no module to forward-hook:

| Quantity | Where | Shape |
|:---|:---|:---|
| `x` (pre-synaptic) | `blocks.{L}.mlp.hook_post` | (seq, 3072) |
| `y` (post-synaptic) | `blocks.{L}.hook_mlp_out` | (seq, 768) |
| `W_out` | `blocks.{L}.mlp.W_out` | (3072, 768) |

`y == x @ W_out + b_out` holds to 3.8e-06 (float32). **`W_out` is already in the
`(n_in, n_out)` convention the learning rules are written in**, so `_hebb_term` and
`_oja_decay` carry over unchanged.

**B. The per-step engine — `atr_bridge.make_atr_step`.** `atr_step(model, r) -> r_next`
was factored out of `run_atr_loop` by copying its loop body verbatim — inject at
`blocks.0.hook_resid_pre`, read at `blocks.11.hook_resid_post`, rescale to the
*initial* norm (`x · ‖x₀‖/‖x‖`), not to unit norm — not reimplemented. The acceptance
test `test_step_reproduces_run_atr_loop_bit_exactly` iterates it and requires bit-exact
agreement with `run_atr_loop`; CI checks out the parent repo and runs it under
`ATR_REQUIRE_PARENT=1`, so the guarantee holds on every push, not just the author's
machine.

**C. The controls, on that stack.** With A and B in place, what remained was running
the controls on the real stack — C0 bit-exact with the adapter installed at eta=0 is
the gate. **Correction:** this used to point at `EXP_001_RESULTS.md` for those runs, and that
file records no C0 at all. What actually covers C0 on the real stack is the pytest suite,
which asserts it bit-exactly on 124M real parameters and asserts that it *fails* when handed a
perturbing hook, plus the step-size map's `off` cell. Later experiments do carry their own
eta=0 gates and report them bit-identical, EXP-002's across all twelve plastic matrices. The
known caveat still stands: C0 has been seen to fail
intermittently at ~1e-4 on CPU and could not be reproduced in 80 repeats — reproduce
against an unhooked control before suspecting the hooks.

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
- Both outcomes are informative under the criteria in §4.

## 3. Design

- **Site:** `blocks.6.mlp` → `W_out`. Mid-stack, MLP down-projection: this repo's own
  first choice.
- **Start:** the saved iteration-1000 state, not a fresh prompt run.
- **eta ladder:** 0 (C0), then 1e-6, 3e-6, 1e-5, 3e-5, 1e-4. Half-orders.
- **Horizon:** 200 iterations per eta. The cycle is locked in from ~250 and stable to
  1000; 200 iterations is the pre-registered horizon, and a break not visible within 200
  is recorded as "no break at this horizon".
- **Cadence: `k=1`. Apply every iteration. Do not sweep it.**
- **Log per iteration:** cos at **lag 1 and lag 2**, L2(A,B), p(top1) and entropy in
  both phases, and `delta_norm` / `delta_frac` / `clipped` / `nonfinite`.

### Why cadence is fixed at 1

There is a real aliasing trap here. Applying the update every `k=2` iterations on a
period-2 cycle drives every update from phase A's activations and never phase B's:
not plasticity on the cycle, but plasticity on half of it, biased along whatever
direction separates the phases.

The answer is not to reason carefully about which `k` is safe. It is to not create the
problem: **`k=1` cannot alias against anything**, and it is also the tightest coupling
of the two loops, which is the regime this experiment is about. Cadence is a free
parameter. A sweep is deferred — not here, and not before a single-cadence result
exists to compare a sweep against.

If cadence is ever swept, `k=2` on a period-2 cycle is the one value that must be read
as a phase-locked special case rather than a point on a curve.

## 4. Outcomes, and what each would mean

| Outcome | Reading |
|:---|:---|
| lag-2 cosine stays 1.000000 | The cycle is robust to modification of the map that generates it. |
| Cycle → fixed point | The oscillation is damped out; plasticity acts as a brake. Consolidation, in this repo's H1 sense. |
| Cycle → longer period, or wanders | The update perturbs the flip axis without destroying the orbit. Measure the new period with a lag scan, not a lag-1 gate. |
| Whole trajectory diverges | eta too high, or the ceiling is doing work — check `clipped` before interpreting anything. |

**Kill conditions.** The experiment is void, not interesting, if: C0 fails on the real
stack; `clipped` fires before the cycle changes; or the norm-matched random control
(C2) breaks the cycle the same way Oja does — that last one would mean the finding is
about perturbation magnitude, not about Hebbian learning.

## 5. Controls, in order

1. **C0** at eta=0 on the real stack. Bit-exact. Gate.
2. **C1** revert — the cycle must return to `cos(A, f(f(A))) = 1.000000` after
   `revert()`, from the same saved state.
3. **C2** norm-matched random direction at whatever eta first moves the cycle. Two known
   subtleties — the first to handle when C2 is run, the second now handled in `controls.py`:

   - The norm match is per-update, and accumulated Oja steps are correlated where
     random ones are a walk, so cumulative drift diverges with iteration count
     (ratio ~1/√2 after two applies, measured). Compare at matched *cumulative*
     `delta_frac`, not just at matched eta.
   - **`c2_random_direction` now runs the random arm over multiple seeds.** It takes
     a `seeds` argument (default 0-9), constructs `OjaPlasticity(..., mode="random",
     seed=seed)` once per seed, and reports the spread as a distribution —
     `cos_per_seed`, with `cos_min`/`cos_max` and the mean `cos_oja_vs_random_final`;
     `tests/test_controls.py::test_c2_random_arm_actually_varies_with_the_seed` pins
     that the arm actually varies with the seed. One nuance carries over from
     `EXP_001_RESULTS.md`: `seed` reaches `OjaPlasticity` only through its RNG, drawn
     from in `mode="random"` and nowhere else — deterministic rules like `hebb` give
     bit-identical runs across seeds, so a multi-seed spread is a control for the
     random arm specifically, not for the rule arms.
4. **The fixed-point control:** the same eta on `state_prolet.pt`. If plasticity
   breaks the cycle *and* destroys the fixed point, the result is "perturbing weights
   changes things". If the cycle breaks while the fixed point holds, the result is
   specific to the oscillator.

## 6. What this does not ask

Nothing here touches basin counts, the 125-prompt library, or the collapse question.
Those need the full sweep and are the natural EXP-002. This question was chosen because
it has an exact answer and an existing baseline measured to six decimal places.

---

## 7. The offline arm (required at every eta)

**Added after the prior-art check.** `PRIOR_ART.md`, "The finding that changes our
experiment": Oja's rule converges to the dominant eigenvector of the second-moment
matrix of whatever activations pass through it, **and it does that with no feedback at
all.** Under Oja at eta>0 the weight matrix changes whether or not feedback is present.
Every outcome in §4 — cycle survives, cycle damps, period lengthens, trajectory diverges — is
consistent with the rule simply doing its job on a fixed activation distribution, and
none of them is on its own evidence about the coupling this project is about.

EXP-001 therefore runs **two arms at every eta on the ladder**, not one.

| Arm | What it is |
|:---|:---|
| Closed loop | Run the loop; every `k=1` iterations apply the rule to the activations flowing through right now. The changed `W_out` shapes the next iterate, which shapes the next update. |
| Offline | Run the same loop **frozen**, recording those activations. Replay the recording through the same rule with no feedback. Install the resulting matrix. Re-run the loop frozen. |

**The result is the difference between the arms.** A closed-loop number must be reported
together with its offline partner.

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
| Total number of weight updates | `n_updates`, plus `n_steps` and `apply_every` |
| Order of the activation samples | `sample_order` — the per-sample iteration index, in consumption order |
| Batching of samples per update | `samples_per_update` |
| Initial weight, and RNG seed | `w0_sha256` (byte hash), `seed`, `rng_state_sha256` |
| Centring, or its deliberate absence | `centring`, read off the rule object rather than asserted |

Plus `site`, `mode`, `transposed`, `dtype` and `store_dtype`, because a comparison across
two different matrices, rules, layouts or precisions is not a comparison. The only
`ArmConfig` fields *outside* the table are the three that are supposed to differ — `arm`,
`feedback`, `y_source` — and a test asserts that, so a field added later without a decision
about it surfaces as a red test rather than as an unchecked axis.

`store_dtype` being a matched axis has a consequence worth stating: **a float16 recording
is refused as an offline arm.** Half-precision storage exists and records its own
round-trip error, but a replay carrying rounding the live arm never saw is not matched. The
memory escape hatch that keeps the arms matched is fewer steps, not lower precision.

**On a mismatch, `run_matched_arms` raises `ArmsMismatchError` and reports no comparison.**

Two axes are worth reading the small print on:

- **`sample_order` is not "the arms saw the same values."** They cannot — that difference
  *is* the experiment. It is "the arms consumed their samples in the same order, indexed
  the same way, with nothing dropped and nothing repeated." This is why the recorder
  raises `MemoryError` when a recording exceeds its budget instead of thinning it.
- **`centring` currently reads `absent` in both arms**, because `plasticity.py` offers no
  centring option — neither `x` nor `y` is ever mean-subtracted. The axis passes by shared
  absence, recorded as the literal string `absent` rather than as `True`. It is read off
  the object, so the day a `centre` flag is added, an arm with it and an arm without
  stop matching automatically. **If centring is ever added it must be added to both arms
  in the same commit** — PRIOR_ART is explicit that applying it to one arm alone moves
  the fixed point by itself.

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

### The detection limit

`tests/test_offline_control.py::test_a_site_the_loop_does_not_route_through` measures the
floor directly. It reads the loop out at `blocks.3.hook_resid_post`, below the site, so
**no state feedback exists at all** and the `x` reaching the rule is bit-identical in both
arms at every step. Whatever the arms still differ by there is noise, and no claim about
feedback can be made underneath it.

| Mode | Floor, feedback severed (eta=1e-5, 6 steps, cadence 1) |
|:---|:---|
| `y_source="recomputed"` | **Zero — bit-identical.** Asserted as a bound (`rel_fro_diff < 1e-8`) *and* as `torch.equal`. |
| `y_source="recorded"` | **`diff_over_drift` = 6.77e-02** (`rel_fro_diff` 9.26e-04 against drift 1.37e-02). Not noise: it is Oja's own `y = xW` recursion, frozen by the recording. |

**That second row is larger than the routed signal.** On the full 0→11 loop at the same
eta and step count, `y_source="recorded"` gives `diff_over_drift` = 1.91e-02 — a third of
the no-feedback floor. So in the default mode the arms' difference is dominated by the
frozen-`y` artefact and **is not evidence about feedback at this eta.**

One honest caveat on that comparison: the floor is measured on a shallower loop (0→3), so
its activation statistics and its drift are not identical to the routed loop's. The two
numbers are the same order and the same sign, which is enough to say the default mode
cannot resolve feedback here; they are not exact enough to subtract.

**Consequence for EXP-001: make the feedback claim from `weight_recomputed_y`, whose floor
is exactly zero.** Report `weight` alongside it, because it is the literal PRIOR_ART
protocol, but report it next to its floor and not on its own.

> **Two bounds on "exactly zero", both learned after this section was written.** It is not a
> property of the harness; it is a property of a particular configuration, and outside that
> configuration it is not zero.
>
> - **Whole-matrix sites only.** At a **per-head** site the reconstruction is additive out of a
>   fused twelve-head operation, so the null is a measured float32 bound of about 1e-7, not
>   zero: 2.960e-08 with `bit_identical` False, against a pre-fix 4.9975e-02 (C-45 `retired`,
>   C-57 `supported`).
> - **One plastic site only.** With plasticity at more than one layer, only the **lowest**
>   plastic layer floors at zero. Every layer above it is non-zero, because within a single
>   forward pass a lower layer's drift changes the activations arriving at a higher one, and
>   severing the loop does not cut that path. So a multi-site closed-versus-offline number has
>   **no zero baseline at all** and may not be placed in a series with the single-site shares
>   (C-63, which amends the register's standing rule 3).

The zero floor holds only because `offline_control._recompute_y` reproduces the site's
**fused `torch.addmm`** bit for bit rather than merely mathematically — TransformerLens's
`batch_addmm` flattens to 2-D and
calls `torch.addmm`, written that way to match HuggingFace's `Conv1D`. The obvious
`x @ W + b` differs in the last bits (max|Δ| 1.9e-06 on `y` at this site), propagates into
every offline update, and compounds: it put a floor of roughly **5e-09 relative Frobenius**
under the entire comparison. That floor was hardware-dependent — 2.9e-09 on one machine and
4.8e-09 on a GitHub Actions runner, same commit and same seed — which is the same class of
trap as `initial_norm` reading 1468.48828125 on CI against 1468.4886474609375 locally.

**Floating-point floors in this suite are asserted as bounds, not values, because the
measured floor is hardware-dependent.** The test does so, and
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

### First measurement — the harness's own output, not a result

Recorded so the numbers in this section have provenance. Prompt
`"The cat sat on the mat and then the"`, site `blocks.6.mlp`, layers 0→11, cadence `k=1`,
`max_delta_frac=0.05`, seed 0, mode `oja`, centring absent, arms verified matched on every
axis. **This is a harness shakedown at a short horizon, not EXP-001.** EXP-001 runs 200
iterations from `state_divine.pt`; these are 6 and 20 from a cold prompt.

| eta | steps | drift closed | drift offline | `cos_delta` recorded | `diff_over_drift` recorded | `diff_over_drift` recomputed |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-6 | 20 | 7.06e-04 | 7.09e-04 | 0.9999987 | 5.00e-03 | 4.64e-04 |
| 1e-5 | 6 | 3.38e-03 | 3.44e-03 | 0.9999709 | 1.91e-02 | 2.87e-04 |
| 1e-5 | 20 | 6.63e-03 | 6.93e-03 | 0.9998848 | 4.59e-02 | 4.41e-03 |
| 1e-4 | 20 | 4.31e-02 | 5.00e-02 † | 0.9940700 | 1.72e-01 † | 3.08e-02 |

† **`clipped` fired on the offline arm at eta=1e-4/20 steps** — it hit the 5% ceiling while
the closed arm did not, so that row's arms are no longer matched on effective step size and
its numbers must not be read as a feedback measurement. This is exactly the §4 "check
`clipped` before interpreting anything" kill condition, and it says the eta ladder's top
rung needs either a raised ceiling or a shorter horizon before it is usable.

Reading, stated at the strength the measurement supports: **in `recomputed` mode the
feedback contribution is small but resolvable** — 2.9e-04 to 4.4e-03 of the arms' own drift
at eta=1e-5, against a floor of exactly zero — and it grows with both eta and horizon. In
`recorded` mode nothing is resolvable at all at these settings, because the floor exceeds
the signal. Nothing here was tuned; the eta ladder is §3's, unchanged.

### Reading the outcome

Add a row to §4:

| Outcome | Reading |
|:---|:---|
| The arms agree (difference at or below the no-route floor) | **Feedback contributes nothing detectable at this eta.** Nothing may be tuned to make the number larger. |

`diff_over_drift ≈ 0` at every eta on the ladder would say the closed-loop result is Oja
finding the dominant direction of the frozen loop's activations.
