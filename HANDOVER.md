# Handover

*State of the project as of main `4d3ae9d` (PR #35 merged). Written for someone —
human or model — picking this up cold. Measurements are stated as measurements;
where something is interpretation it is marked as such.*

Read `ORIENTATION.md` first if you have never seen this repo. This file is the
"where we are and what's next" layer on top of it.

---

## 1. What the project is, in one paragraph

The parent project ([ATR](https://github.com/earlyprototype/lucier-gpt2-activ-tensor-reson-experiments))
iterates a **frozen** GPT-2 small on its own residual stream — read at
`blocks.11.hook_resid_post`, rescale to the trajectory's initial norm, re-inject at
`blocks.0.hook_resid_pre`, repeat. Iterated, the same weights define a map from
residual-stream state to residual-stream state, and that map has fixed points,
cycles and basins that single-pass inference never brings into view. **This repo
turns the slow loop on**: one (or many) weight matrices are allowed to change under
a local activation-driven rule while the loop runs. There is no task, no loss and
no target. The narrow question is whether the one channel that persists across
prompts — the weights — can be written to, and whether writing to it changes what
the system does afterwards.

---

## 2. Current state

**Everything below is on `main`.** Suite: **252 tests collected**; 248 pass locally
and 4 skip (the parent-repo bridge tests, which need the parent checked out). CI
checks out the parent and runs all 252 under `ATR_REQUIRE_MODEL=1` and
`ATR_REQUIRE_PARENT=1`, so a missing model or parent is a failure, not a silent skip.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest          # ~2-3 min on CPU; downloads gpt2 (~500MB) once
```

### What has been run

| # | Artifact | What it is |
|---|---|---|
| 1 | `experiments/output_baseline/BASELINE.md` | Frozen 125-prompt basin census — the reference everything is compared against |
| 2 | `STEP_SIZE_MAP.md` | 35 cells: 4 rules × 8 step sizes at one site — where each rule's weights move without the ceiling firing |
| 3 | `EXP_001_RESULTS.md` | The closed-loop vs offline-arm comparison at the one cell that moves the loop |
| 4 | `BASIN_BIFURCATION.md` | Whether the observed basin change is a boundary move or a created attractor |

### What has NOT been run

- **EXP-002 (issue #24), the recorded primary experiment.** See §5.
- Any multi-site / distributed run. The tooling landed in PR #35; nothing has used it.
- Any site other than `blocks.6.mlp`. Every number in the repo is that one matrix.
- Cadence > 1. Every run is cadence 1 (update every iteration).
- Any model other than GPT-2 small.
- The drive/leak/homeostasis experiments in `ISSUE_normalisation_homeostasis.md` (E1–E4).

---

## 3. Results so far

### 3.1 The frozen baseline (the reference)

125 prompts under the frozen loop settle into **5 basins**, named by the top-1 token
of the settled state:

| basin | count | dynamical class |
|---|---|---|
| `prolet` | 55 | fixed point |
| `Divine` | 34 | **period-2 limit cycle** |
| `till` | 19 | fixed point |
| `Anarch` | 16 | fixed point |
| `solidarity` | 1 | fixed point |

`Divine` is the oscillator: lag-1 cosine ≈ 0.68, lag-2 ≈ 1.000000. From
`RESONANCE_NOTE.md`, it is also **attracting and wide** — states displaced by as much
as their own magnitude return to it (5/5 perturbation magnitudes tested), contracting
at ~0.968 per iteration.

### 3.2 The step-size map (`STEP_SIZE_MAP.md`)

One prompt (`A01_physics`), one site (`blocks.6.mlp`), 120 steps, cadence 1,
ceiling 0.05. Only the rule and the step size vary.

| rule | basin behaviour across eta | moves the basin inside the ceiling-silent band? |
|---|---|---|
| `hebb` | `prolet` → **`comrade`** (7.07e-05) → `locality` (at the ceiling) | **yes** |
| `anti_hebb` | `prolet` throughout; → `anarchism` only at 100% clip | no |
| `oja` | `prolet` at every eta, up to and including 5% drift / 100% clip | no |
| `random` | `prolet` at every eta, up to 5% drift | no |

Three facts worth carrying forward:

- **`hebb` at eta ≈ 7.07e-05 is the only cell in the whole sweep that moves the loop
  with the ceiling silent** (1.12% relative weight change, 0.0% clip rate). That is
  the working point every later run uses.
- **Norm-matched `random` never moves the basin**, even at the full 5% drift. `oja`
  reaches *larger* drift (2.9%) than `hebb` needed (1.1%) and also never moves it. So
  the basin change is specific to the update's **direction**, not its magnitude.
- **No spectral collapse anywhere.** Effective rank stays flat (~642 of 768) and
  *rises* under `oja`/`anti_hebb`. The changes are not a hollowing-out artifact.

**How the rules relate**, from `ORIENTATION.md` and the `plasticity.py` header:

```
hebb       dW =   E[x yᵀ]                 no brake
oja        dW =   E[x yᵀ] − W E[y yᵀ]     the second term is the brake
anti_hebb  dW = − E[x yᵀ] − W E[y yᵀ]     reinforcement flipped, brake kept
random     norm-matched noise             the magnitude control
```

`anti_hebb` exists to erode what the loop has settled into; issue #25 calls it the
active ingredient in EXP-002, and it has a bounded fixed point at
`W* = −E[x yᵀ] E[y yᵀ]⁻¹`. Do not implement it as a negative eta — that flips the
brake as well and turns it into an accelerator (issue #27 item 6).

### 3.3 EXP-001 (`EXP_001_RESULTS.md`)

`hebb`, eta 7.065171428571429e-05, `blocks.6.mlp`, 120 steps, cadence 1, ceiling
never fired, ~1.12% relative weight change.

**Basin changes observed** (all three prompts are `prolet` under the frozen loop):

| prompt | frozen → nudged | dynamical class change |
|---|---|---|
| `A01_physics` | `prolet` → **`comrade`** | fixed point → fixed point |
| `A02_medical` | `prolet` → **`comrade`** | fixed point → fixed point |
| `A04_climate` | `prolet` → **`Divine`** | **fixed point → period-2 cycle** (lag-1 0.99999 → 0.661, lag-2 0.99999) |

Two distinct kinds of transition from the same nudge: two prompts reach a terminal
state (`comrade`) that appears **nowhere in the frozen 125-prompt census**, and one
prompt has its *dynamical class* changed — a static fixed point becomes a two-step
oscillation, landing on the pre-existing `Divine` orbit (its nudged lag-1 of 0.661
sits inside the range native `Divine` prompts show frozen: 0.659–0.696).

**The offline arm.** The same update was computed two ways: in situ (weights changing
while the loop runs) and abstractly (the identical rule replayed over activations
recorded from the frozen run, no feedback path). The two agree —
`cos(ΔW_closed, ΔW_offline) = 0.99294`, and the offline arm flips the same basin. A
severed-path control (loop read out at `blocks.3`, below the plastic site, so coupling
is impossible) gives a floor of exactly 0.0 in the zero-floor `recomputed` mode.

That agreement is what the arm is for: it verifies that the Hebbian update is a
function of the activation statistics, so it can be computed abstractly or in situ and
give the same result. Issue #26 separately defines a claim about *coupling* as living
in the difference between the arms; if a coupling claim is ever made, that is the bar
it has to clear.

### 3.4 Basin bifurcation (`BASIN_BIFURCATION.md`) — the newest result

Asks what *kind* of move `prolet → comrade` is, on issue #25's ladder: **step 3**
(the boundary between two existing basins moved) or **step 4** (a new attractor was
created — a bifurcation). Two independent discriminators; they agree.

**D1 — is `comrade` a fixed point of the original frozen map?** Restore W0, seed the
`comrade` state, iterate the **frozen** loop 200 steps. It holds `comrade` for 3
iterations, leaves at **iteration 4**, and settles into the frozen baseline's own
`prolet` fixed point (lag-1 = 1.000000 by ~iter 30), ~9.4% relL2 away. Motion is
smooth and monotone throughout — no discontinuity. **`comrade` is not an attractor of
the original model.**

**D2 — the installable-ΔW α-sweep** (closes issue #32 §5). Install `W0 + α·ΔW`, run
the frozen loop, read the settled basin:

| α | 0.00 | 0.25 | 0.50 | **0.75** | 1.00 | 1.25 | **1.50** |
|---|---|---|---|---|---|---|---|
| basin | `prolet` | `prolet` | `prolet` | **`comrade`** | `comrade` | `comrade` | **`Divine`** |
| lag-1 | 1.000000 | 1.000000 | 1.000000 | 0.999999 | 0.999999 | 0.999998 | **0.733997** |
| comrade − prolet logit gap | −0.657 | −0.404 | −0.114 | +0.203 | +0.525 | +0.816 | +2.735 |

Three things fall out:

1. **Threshold, not smooth bias** — issue #32 §5's answer. The basin holds at `prolet`
   through α ≤ 0.50 and flips discretely at **α\* = 0.75**.
2. **Smooth logits, discrete attractor.** The logit gap is a smooth monotone function
   of α that crosses zero inside (0.50, 0.75]; `comrade`'s rank climbs 5 → 4 → 2 → 1.
   The argmax is discrete; the thing it is the argmax *of* is not. Note the crossing
   is driven mainly by **`prolet`'s logit falling** (16.95 → 16.07), not by `comrade`
   rising (16.29 → 16.27).
3. **A `prolet` → `comrade` → `Divine` cascade.** At α = 1.5 the system tips into the
   period-2 cycle and lag-1 collapses to 0.734. Two bifurcations along one line, and
   the second reproduces the same fixed-point→cycle transition A04 showed in EXP-001.

> *Interpretation, marked as such:* a **single** matrix at ~1% drift creates an
> attractor the original model does not have. That is issue #25's ladder **step 4**,
> which the recorded plan expected to fail first. It did not.

---

## 4. Tooling inventory

| File | What it gives you |
|---|---|
| `plasticity.py` | `OjaPlasticity` — one site, modes `off`/`hebb`/`oja`/`anti_hebb`/`random`, ceiling, bit-exact `revert()`, `report()`. Site adapters for HuggingFace `Conv1D`, TransformerLens `W_out`, and **per-head stripes** on both. `subspace_projector()` for aiming drift. `candidate_sites()` |
| `multi_site.py` | **`MultiSitePlasticity`** — N sites at once (whole matrices and head stripes mixed, heterogeneous mode/eta/ceiling/projector per site). Rejects overlapping footprints at construction, keyed on live parameter identity |
| `atr_bridge.py` | `make_atr_step(model, prompt) -> step(model, r)` — one iteration of the parent loop, **extracted verbatim**, bit-exactness CI-enforced. `initial_state`, `load_state` |
| `controls.py` | C0 (eta=0 identity gate), C1 (revert), C2 (norm-matched random, multi-seed distribution), C3 (Hebb diverges / Oja does not) |
| `offline_control.py` | Matched closed-vs-offline arms, a **17-axis** mechanical match verifier, and the severed-path control |
| `experiments/` | `baseline_basins.py` (the census + the canonical basin readout), `step_size_map.py`, `exp001_hebb.py`, `basin_bifurcation.py` — the last three take `--site` |

**Multi-site usage:**

```python
from multi_site import MultiSitePlasticity, SiteSpec

driver = MultiSitePlasticity(model, [
    SiteSpec("blocks.6.mlp",            mode="hebb",      eta=7.07e-5),
    SiteSpec("blocks.11.attn.head.7",   mode="anti_hebb", eta=1e-6),
])
with driver:
    for i in range(n_iter):
        r = step(model, r)
        driver.apply()          # per-site ceilings, aggregate report()
driver.revert()                 # every touched matrix, bit-exactly
```

Guarantees proven in `tests/test_multi_site.py` (25 tests, real GPT-2, both backends):
disjoint sites move together and revert bit-exactly; **head isolation holds under
simultaneous operation** (running heads 3 and 7 together gives each exactly what it
gives alone, other ten heads bit-identical); the 12 head-instances of one
`attn.c_proj` reconstruct the whole-matrix update bit-for-bit; overlap is rejected.

---

## 5. The recorded plan — what to do next

**The plan lives in GitHub issues #24–#32**, written 2026-07-28 as the record of a
planning conversation (#29 says so explicitly; #27 calls itself "the part of the
walkthrough we never got to"). Read them before proposing anything; they are the
source of truth for intent, and several of them fix interpretations *in advance* on
purpose.

### 5.1 The primary experiment: EXP-002 (issue #24) — NOT YET RUN

Issue #24 states it **"supersedes EXP-001 as the thing to run first."** The sequence:

1. **Collapse.** Run the frozen loop until the state falls into a well. No plasticity
   needed — the frozen loop already collapses 80%+ of the library.
2. **Work the well.** Turn on the **Hebbian / anti-Hebbian balance**. The
   anti-Hebbian direction erodes what the state settled into so it can climb back out.
3. **Stabilise.** Freeze the weights, plasticity off. The drift is now permanent.
4. **Reprompt.** Inject a fresh prompt from a different basin, unseen this session.
   Run frozen.
5. **Measure.** Weight difference (how far, and does its direction align with the
   eroded well) and output (where the fresh prompt lands vs the untouched model).

Steps 3–5 are the persistence test, also written up alone as **issue #29**: the
residual stream is destroyed at an episode boundary, so **the weights are the only
channel** through which episode *n* can reach episode *n+1*. Anything observed in
step 5 came through the weights or came from nowhere.

**Issue #24's own sequencing note:** *"First job is just steps 1 and 3: drive to
collapse and stabilise it. Get that reproducible before adding the balance."*

### 5.2 The direction that matters: escape, not collapse

Issue #27 item 5, stated before any result existed: *"Collapse is already the default,
so 'we caused collapse' is not a finding. The interesting direction is the opposite
one: **escape** — whether a balancing rule can lift the state back out of a well it
has fallen into."* Any framing that reports collapse as the achievement has the
experiment backwards.

### 5.3 The ladder (issue #25) — and where we now are on it

1. Deepen an existing attractor.
2. Shift an existing attractor.
3. **Move a basin boundary** — "the measurable one, and the first real result."
4. **Create an attractor where none existed** — "expect this to fail first."

`BASIN_BIFURCATION.md` puts the `comrade` result at **step 4**. That is further up the
ladder than the plan anticipated, and it should recalibrate expectations for
everything downstream.

### 5.4 Open work not yet started

- **Every site but one.** All numbers are `blocks.6.mlp`. The 12 MLP down-projections,
  12 attention output projections and 144 head stripes are unswept. **eta does not
  transfer across sites** — `‖W0‖_F` and activation scale differ, so the anchoring
  formula must be re-measured per site (`STEP_SIZE_MAP.md` caveats).
- **The distributed regime.** Many sites plastic at once — rung 3 of the repo's own
  ladder, and issue #25's "distributed damping... many small elements each acting on
  its own subspace." Tooling exists; nothing has run.
- **Issue #31 — within-basin spread.** The cheapest measurement in the project, needs
  no plasticity, and it changes how everything else reads: do prompts in one basin
  settle onto the *same* state (erasure — the settled state is just a 5-way label) or
  nearby-but-distinct states (compression)? `RESONANCE_NOTE.md` calls this the first
  thing it would run.
- **Issue #28 — prior-art gaps.** In particular an unverified claim (Chaudhary 2025)
  that transformer plasticity diverges around 8 layers and is stable around 4. **GPT-2
  small is 12.** If it holds, instability is the expected result, not a surprise.
- **Cadence, the drive term β, the leak term α, target-energy** — `DESIGN.md` and
  `ISSUE_normalisation_homeostasis.md` (E1–E4).

---

## 6. Open decisions for the operator

1. **`EXP_001_SPEC.md` is self-inconsistent and it was left that way deliberately.**
   Its title is still *"Does the `Divine` period-2 cycle survive plasticity?"* and its
   header still says **"Status: proposed, not run"**, while `EXP_001_RESULTS.md`
   reports a *different* experiment (the offline-control / basin-flip run). The label
   "EXP-001" got reused. Either retitle the spec to the experiment that ran and give
   the Divine-cycle question its own spec, or flip the status and keep them separate.
   §0 and §5.3 of that file were corrected and are right either way.
2. **Whether to record the α-cascade as a headline finding.** It is committed in
   `BASIN_BIFURCATION.md`; it has not been folded into `README.md` or `ORIENTATION.md`.

---

## 7. Rules this repo runs by — do not relax these

These are the repo's own standards, learned the hard way. They are why the results
above are worth anything.

- **C0 is the gate.** At eta=0 the hooks must not perturb the trajectory by a single
  bit. Nothing downstream is interpretable if it fails.
- **A control that cannot fail is worse than no control.** Every control is tested in
  both directions — it passes clean *and* fails when handed the defect it exists to
  catch. `ATR_REQUIRE_MODEL` / `ATR_REQUIRE_PARENT` turn "the thing under test is
  missing" from a green skip into a failure.
- **Never reimplement the ATR loop.** Import it via `atr_bridge`. A bug in a
  reimplementation is indistinguishable from a plasticity effect.
- **No stand-ins.** The toy model was deleted on purpose: its `Conv1D` was our own
  reimplementation and could disagree with HuggingFace without any test noticing. Two
  assertions that passed on the toy were false on real weights.
- **Report the clipping rate on every result.** A run where the ceiling fired is a
  measurement of the ceiling, not of the rule.
- **Never use an even-only snapshot schedule.** It samples a period-2 orbit at one
  phase and makes oscillation invisible by construction — this hid the parent's F9
  finding for months. Log lag-1 *and* lag-2.
- **Justify any convergence horizon against the contraction factor** (~0.968/iteration
  here, ~71 iterations per decade). A 200-iteration horizon has already produced one
  false "failed to return" in this repo.
- **Use tolerance, not `torch.equal`, for state comparisons.** Two states that are the
  same point of the dynamics differ in float32; exact equality is only for asserting
  bit-identity on purpose (the bridge test, the eta=0 test).
- **Don't tune to match the parent's published basin percentages.** The bridge is
  bit-exact and CI-enforced; a mismatch means something else moved. Find which.
- **Say what you did not rule out.** Issue #27 is the list of failure modes that
  *look like findings*; every write-up should state which it ruled out and how.
- **This is not learning.** No task, no loss, no target. The defensible phrase is that
  the weights **carry a trace of the episode**.

---

## 8. Known risks

- **TransformerLens is deprecating the entry point this repo depends on.** The suite
  emits: `HookedTransformer.from_pretrained is deprecated... use
  TransformerBridge.boot_transformers(...) then enable_compatibility_mode() for
  HookedTransformer-equivalent numerics.` Because every result rests on bit-exact
  reproduction of the parent's loop and saved attractors, a major-version numerics
  shift could invalidate provenance silently. Pin the version; track the v3 migration.
  (Runs so far: 3.5.1 and 3.6.0 — a ~0.1%-class state-norm drift between them is
  already visible in `BASIN_BIFURCATION.md`'s fidelity table, label-invariant.)
- **C0 has flickered on CPU.** Twice, at ~8.6e-05 and 6.3e-05, unreproducible in 80
  controlled repeats and 16 cold processes; an unhooked-vs-unhooked control never
  differed. Best explanation is nondeterministic parallel float reduction order. If a
  `bit_exact` failure appears at that magnitude, reproduce against an
  unhooked-vs-unhooked control before suspecting the hooks.
- **No raw state or dense ΔW is persisted.** `experiments/output_*` holds summaries
  only, so analyses like `basin_bifurcation.py` must *reproduce* an episode rather than
  load it. Worth fixing if ΔW is going to be reused much.
- **Provenance across revisions.** `EXP_001_RESULTS.md` carries a
  `provenance_warning`: its cells were produced under more than one repo revision.

---

## 9. Practical notes

- Everything runs on CPU. The step-size map was 59 CPU-minutes for 35 cells; EXP-001
  was 44 CPU-minutes for 13 cells; `basin_bifurcation.py` is 6.2 minutes. A
  125-prompt sweep is the expensive unit.
- Set `torch.set_num_threads(1)` for determinism; the experiment scripts do.
- The basin readout is `experiments/baseline_basins.py` — final LayerNorm + unembed,
  argmax at the last position. **Reuse it; do not hand-roll a readout.**
- `.claude/hooks/session-start.sh` builds the venv on session start.
- The `board-state` branch is machine-generated agent-coordination state and never
  merges to main.

---

*Last verified against main `4d3ae9d`. If the code and this file disagree, the code
is right and this file is stale — check `git log` since that commit.*
