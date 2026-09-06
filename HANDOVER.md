# Handover

*State of the project as of main `754cc26` (PR #59 merged). Written for someone — human or
model — picking this up cold. Measurements are stated as measurements; where something is
interpretation it is marked as such.*

Read `ORIENTATION.md` first if you have never seen this repo. This file is the
"where we are and what's next" layer on top of it. **`CLAIMS.md` is the authority over
both.** This file describes and sequences; the register decides what any measurement is
allowed to be called, and where the two disagree the register wins and this file is the bug.

> **Correction record.** This file carried a review notice from 2026-07-31 warning that
> `ALIGNMENT_REVIEW.md` had superseded five statements in it. All five are now corrected in
> the body rather than fronted by a warning, so the notice is retired and replaced by this
> record of what changed:
>
> - **§3.3** said the two arms agreeing meant feedback did nothing. That is C-34, `retired`.
>   Corrected in place, with C-31's 12%-of-drift steering, C-33's limit on it, and the later
>   T2.1 and T2.1b bounds.
> - **§3.4 and §5.3** put the `comrade` result at ladder step 4, a created attractor. T1.1
>   refuted it; C-26 is `retired` and the supported reading is step 2, C-56. §3.5 is new and
>   carries the refutation.
> - **§5.4** said issue #31 had not been started. It had; the answer is compression, C-04 –
>   C-06, plus the resolution-limit caveat C-07.
> - **§3.2's** first two carried facts, the "only cell in the whole sweep" undercount and the
>   "direction, not magnitude" inference, were corrected on 2026-08-05 (C-21, C-55).
> - **§5.4's** Chaudhary reading was the superseded one. It now carries C-44 with the two
>   limits that bite: the paper reports no 4-layer result, and nothing in that literature
>   tests 12 layers.
>
> Two of the review's Tier 0 items are still open and are **not** corrections to this file:
> T0.5, margin discipline, and T0.6, committing the prior-art search artifacts.

---

## 1. What the project is, in one paragraph

The parent project ([ATR](https://github.com/earlyprototype/lucier-gpt2-activ-tensor-reson-experiments))
iterates a **frozen** GPT-2 small on its own residual stream — read at
`blocks.11.hook_resid_post`, rescale to the trajectory's initial norm, re-inject at
`blocks.0.hook_resid_pre`, repeat. Iterated, the same weights define a map from
residual-stream state to residual-stream state, and that map has fixed points,
cycles and basins that single-pass inference never brings into view. **This repo
turns the slow loop on**: one (or many) weight matrices are allowed to change under
a local activation-driven rule while the loop runs. There is no task and no target, and no
**externally specified** objective — which is the precise form, because plain Hebb is
gradient ascent on output energy and so is not objective-free (C-11). The narrow question is whether the one channel that persists across
prompts — the weights — can be written to, and whether writing to it changes what
the system does afterwards.

---

## 2. Current state

**Everything below is on `main`.** Suite: **320 tests collected**; 316 pass locally
and 4 skip (the parent-repo bridge tests, which need the parent checked out). CI
checks out the parent and runs all 320 under `ATR_REQUIRE_MODEL=1` and
`ATR_REQUIRE_PARENT=1`, so a missing model or parent is a failure, not a silent skip.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest          # ~2-3 min on CPU; downloads gpt2 (~500MB) once
```

### What has been run

Eleven write-ups, in the order they landed. The register rows each one settles are named,
because `CLAIMS.md` and not this table is what decides what any of them may be called.

| # | Artifact | What it is | Register |
|---|---|---|---|
| 1 | `experiments/output_baseline/BASELINE.md` | Frozen 125-prompt basin census — the reference everything is compared against | C-01 – C-07 |
| 2 | `STEP_SIZE_MAP.md` | 35 cells: 4 rules × 8 step sizes at one site — where each rule's weights move without the ceiling firing | C-13, C-21 |
| 3 | `EXP_001_RESULTS.md` | The closed-loop vs offline-arm comparison at the working point | C-20, C-24, C-30 – C-33 |
| 4 | `BASIN_BIFURCATION.md` | Whether the observed basin change is a boundary move or a created attractor. **Its conclusion was refuted; read entry 5 with it** | C-25, C-27, C-28, and **C-26 `retired`**, C-56, C-68 |
| 5 | `experiments/output_t1_1/T1_1_RESULTS.md` | T1.1, the coexistence test — does the frozen `prolet` state stay put under `W0 + ΔW`? | C-26 `retired`, **C-56** |
| 6 | `experiments/output_t1_2/T1_2_RESULTS.md` | T1.2, the α up-then-down continuation sweep — is there hysteresis? | C-52: no |
| 7 | `experiments/output_rank1_random/T1_4_RESULTS.md` | T1.4, a rank-1 random direction matched on loop displacement | C-22 `retired`, **C-55** |
| 8 | `experiments/output_t1_5/T1_5_RESULTS.md` | T1.5, the head-site `recomputed`-y reconstruction and its fix | C-45 `retired`, **C-57** |
| 9 | `experiments/output_t2_1/T2_1_RESULTS.md`, `T2_1B_RESULTS.md` | T2.1 and T2.1b, the coupling sweep over step size, cadence and episode length | C-35 answered, **C-58, C-59** |
| 10 | `experiments/output_exp002/EXP_002_RESULTS.md` | **EXP-002 (issue #24), the pre-registered primary experiment** — twelve plastic layers, then a reprompt | C-53 answered, **C-60 – C-63** |
| 11 | `experiments/output_exp003/STAGE0_RESULTS.md`, `STAGE1_RESULTS.md`, `STAGE2_RESULTS.md` | EXP-003 stages 0, 1 and 2 — a label-free grid measurement, spectral concentration, and a cadence ladder | **C-65, C-66, C-67.** Entered 2026-08-05; each file previously stated that nothing in it entered the register |

**Two regime boundaries cut across this table and neither may be crossed silently.**
Artifacts 1 – 9 were taken under the 5% drift ceiling; 10 and 11 were taken with the
ceiling **lifted**, by operator decision (C-60). And the exact-zero severed-path floor that
every single-site coupling number is measured against **does not exist** at more than one
plastic layer (C-63), so EXP-002's closed-versus-offline shares may not be put in a series
with C-31's or C-58's.

### What has NOT been run

- **T2.3 (issue #49) — the full 125-prompt library at the working point.** The register
  calls this the largest available credibility gain and it is the only open issue naming an
  experiment. Tracked as C-54. Every basin-flip claim in the repo still rests on 3 prompts
  (C-41), or on 1 for the T1 and T2 series.
- **EXP-003 Stage 3** — injected signal, focal against distributed. Pre-registered, not run;
  Stages 4 and 5 are gated behind it.
- **Any plastic site that is not an MLP output projection.** Twelve of twelve MLP
  down-projections have carried plasticity; **zero of twelve** attention output projections
  and **zero of 144** head stripes have (C-64). That is T3.1, and eta does not transfer
  across sites, so each one has to be re-anchored.
- **Any model other than GPT-2 small** (C-64).
- **T2.4 — why Oja is inert here.** The decay-over-reinforcement ratio of ~110:1 is measured
  at one site and has never been tested as the explanation (C-14, `not-established`).
- **T1.3 — the α interval where the dynamical class actually changes**, (1.25, 1.50), where
  lag-1 collapses. C-28 stays `provisional` until it is resolved.
- **T0.5 margin discipline and T0.6 the prior-art search artifacts.** The first would give
  every basin label a pre-registered ambiguity threshold (C-55 records its absence); the
  second would turn eleven absence claims into a record (C-42, `provisional`).
- The drive/leak/homeostasis experiments in `ISSUE_normalisation_homeostasis.md` (E1–E4).
  Cadence, which used to be listed here, **has** been run: 2 and 4 in T2.1, 4 and 12 in
  EXP-003 Stage 2 (C-58, C-64).

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
as their own magnitude return to it (5/5 perturbation magnitudes tested), with the return
criterion fixed before the run (C-03). The often-quoted ~0.968 per iteration is an
**endpoint slope, not a fit, and contraction here is not a constant**: across the ladder the
per-decade rate ranges 4 to 124. Do not carry the single number without that range.

Two properties hold across **every** basin, not only `Divine`, and both come from
`BASELINE.md`: prompts inside one basin settle onto nearby-but-distinct states rather than
onto a single state, so the settled state carries more than a 5-way label (C-04, mean `1−cos`
3.32e-03 over 2337 pairs), and that within-basin variation is essentially one-dimensional
(C-05). Position uniformity holds for 125 of 125 prompts (C-06), which corrects the parent's
framing that reports it for `Divine` alone.

**The caveat that governs every basin claim in this repo is C-07.** The taxonomy has a
resolution limit and it is comparable to the separation it is resolving: within-basin spread
3.319e-03 against the nearest-basin gap, `Anarch` to `prolet`, of 2.874e-03 — a ratio of
**1.15**. Read C-07's decomposition before applying that pairwise: only `Divine`'s own spread (6.128e-03) exceeds the gap, while `prolet` (2.773e-03), `Anarch` (1.178e-03) and `till` (3.493e-04) all sit below it, so the nearest pair does resolve at its own scale and the pooled ratio is driven by the period-2 basin. As a pooled statement it stands: two basins this project treats as distinct are no further apart than the prompts
inside one of them. Also worth carrying: 69 of the 125 baseline prompts sit at a top1−top2
margin below 0.5, so the label is often close-run at the readout.

### 3.2 The step-size map (`STEP_SIZE_MAP.md`)

One prompt (`A01_physics`), one site (`blocks.6.mlp`), 120 steps, cadence 1,
ceiling 0.05. Only the rule and the step size vary.

| rule | basin behaviour across eta | moves the basin inside the ceiling-silent band? |
|---|---|---|
| `hebb` | `prolet` → **`comrade`** (7.07e-05 **and** 1.18e-04, both ceiling-silent) → `locality` (at the ceiling) | **yes** |
| `anti_hebb` | `prolet` throughout; → `anarchism` only at 100% clip | no |
| `oja` | `prolet` at every eta, up to and including 5% drift / 100% clip | no |
| `random` | `prolet` at every eta, up to 5% drift | no |

Three facts worth carrying forward:

- **`hebb` is the only rule that moves the loop with the ceiling silent, and it does so at
  two step sizes** (C-21): eta ≈ 7.07e-05 at 1.12% relative weight change, and eta ≈
  1.18e-04 at 2.20%, both at 0.0% clip and both to `comrade`. Not an independent
  replication — the two cells share prompt, seed, site and cadence — so what they show is
  that the flip survives a 1.7× change in eta. The first is the working point every later
  run uses.
- **Norm-matched `random` never moves the basin**, even at the full 5% drift. `oja`
  reaches *larger* drift (2.9%) than `hebb` needed (1.1%) and also never moves it. Those
  are measurements and they stand; the inference once drawn from them — that the basin
  change is specific to the update's **direction**, not its magnitude — does not. It is
  **retired twice over**: first by the α-sweep, which produces three basins by scaling one
  fixed direction, and then by T1.4, which flipped the basin with *arbitrary* rank-1
  directions and so took the replacement "magnitude and sign are both required" down with
  it (C-22, `retired`). C-23 records why this control was matched on the wrong quantity:
  isotropic `random` never reaches `hebb`'s operator norm anywhere in the sweep. What
  survives is **C-55** — at matched displacement an arbitrary direction usually *does*
  move the basin, but never to `comrade`, and at 66×–171× `hebb`'s weight cost.
- **No spectral collapse anywhere.** Effective rank stays flat (~642 of 768) and
  *rises* under `oja`/`anti_hebb`. The changes are not a hollowing-out artifact.

**How the rules relate**, from `ORIENTATION.md` and the `plasticity.py` header:

```text
hebb       dW =   E[x yᵀ]                 no brake
oja        dW =   E[x yᵀ] − W E[y yᵀ]     the second term is the brake
anti_hebb  dW = − E[x yᵀ] − W E[y yᵀ]     reinforcement flipped, brake kept
random     norm-matched noise             the magnitude control
```

These five `mode` strings are fixed combinations. A site can also be given
`terms=[TermSpec(...), ...]` — an ordered sum of signed, optionally-projected
terms over the same two primitives — which is what makes the Hebbian /
anti-Hebbian **balance** of #24 step 2 expressible at one site. See §4.

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
recorded from the frozen run, no feedback path). `cos(ΔW_closed, ΔW_offline) = 0.99294`,
and the offline arm flips the same basin. A severed-path control (loop read out at
`blocks.3`, below the plastic site, so coupling is impossible) gives a floor of exactly
0.0 in the zero-floor `recomputed` mode.

**Do not read that closeness as "feedback did nothing".** That reading is register row
**C-34, `retired`**, and the severed control is what retires it: measured against a null
of *exactly zero* rather than against an intuition about how near 1.0 is near enough, the
feedback-attributable component is **12% of total drift** — a **6.81° rotation** of the
update with a **2.8% shortening** (C-31, C-32). Feedback measurably steers ΔW. What it
does not do, at this operating point, is change the outcome: both arms flip to the same
basin on all three prompts (**C-33**, the honest limit that must travel with C-31).

Two later results bound this from both sides. **T2.1** swept the coupling and found the
feedback-attributable share grows monotonically along all three axes — step size, cadence
and episode length — superlinearly in the last, and still never changed the outcome
anywhere inside the grid (**C-58**). **T2.1b** then found the first setting where it does:
at **2.5× this step size**, ceiling silent, the connected loop settles into the period-2
`Divine` cycle while the severed arm ends somewhere else entirely (**C-59**). Everything in
this subsection is conditional on `y_source="recomputed"`; the `recorded` path is not
interpretable as a feedback test and `offline_control.py` says so.

### 3.4 Basin bifurcation (`BASIN_BIFURCATION.md`)

> **Read §3.5 before quoting this subsection.** Its conclusion, that the edit *created* an
> attractor, was refuted by T1.1 and is register row **C-26, `retired`**. The measurements
> below stand as measurements. The reading placed on them does not.

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

1. **The basin holds at `prolet` at every sampled α ≤ 0.50, and the first sampled flip is at
   α = 0.75.** The sweep steps in 0.25, so what it establishes is a **bracket**, that the
   transition lies in (0.50, 0.75], not a threshold at 0.75; the earlier write-up's "α\* =
   0.75" reads a grid point as a measured crossing and is withdrawn. Note also that T1.2's
   finer continuation sweep, stepping in 0.10, places its own crossing near α ≈ 0.45 — a
   different seeding scheme and therefore a different measurement, not a contradiction, but
   the two thresholds are not interchangeable (C-52). This was also written up as "a
   threshold, not a smooth bias", issue #32 §5's answer; that reading is **C-27,
   `not-established`**, because an argmax changes discretely even under perfectly smooth
   motion, so the discreteness is a property of the readout rather than evidence about the
   dynamics.
2. **Smooth logits, discrete attractor.** The logit gap is a smooth monotone function
   of α that crosses zero inside (0.50, 0.75]; `comrade`'s rank climbs 5 → 4 → 2 → 1.
   The argmax is discrete; the thing it is the argmax *of* is not. Note the crossing
   is driven mainly by **`prolet`'s logit falling** (16.95 → 16.07), not by `comrade`
   rising (16.29 → 16.27).
3. **A `prolet` → `comrade` → `Divine` cascade.** At α = 1.5 the system tips into the
   period-2 cycle and lag-1 collapses to 0.734. Two bifurcations along one line, and
   the second reproduces the same fixed-point→cycle transition A04 showed in EXP-001.

### 3.5 T1.1 refuted §3.4's reading: displacement, not creation

`BASIN_BIFURCATION.md` concluded that a single matrix at ~1% drift **creates** an attractor
the original model does not have, which is issue #25's ladder **step 4**. That was recorded
as interpretation, it named the test that would decide it, and the test was run and went
against it.

**T1.1** (`experiments/output_t1_1/T1_1_RESULTS.md`) is the dual of D1 above. D1 seeded the
`comrade` state under the **original** weights and watched it slide back to `prolet`. T1.1
seeds the original frozen **`prolet`** state under `W0 + ΔW` and asks whether it stays.
It does not: the loop moves to `comrade`, settling at iteration 12 and holding through 120,
with step-to-step agreement at or above 0.99990 the whole way and the top1−top2 margin
falling smoothly to 0.00025 at the crossing.

**A created attractor required coexistence — a new `comrade` basin sitting beside a
surviving `prolet` one. There is no coexistence.** There is one fixed point that the edit
relocates continuously, and a readout that relabels it as it crosses a ridge. That is
ladder **step 2**, not step 4, and it is register row **C-56**, `supported`. C-26 is
`retired`. **T1.2** corroborates it independently: an α sweep taken up to 1.5 and back down
retraces exactly, with no hysteresis loop, which is what one continuously-moving fixed
point looks like and not what two coexisting basins look like (C-52). Its own limit: at a
0.10 grid a loop narrower than one step, inside (0.4, 0.5), is not excluded.

**Every measurement in §3.4 survives; only the reading placed on them fell.** D1's slide back
to `prolet` and the whole α table stand as recorded. What they do not support is creation,
because D1 returns the same verdict for *any* ΔW that displaces the fixed point, so it cannot
distinguish displacement from creation on its own.

**One loose thread, found 2026-08-05 and registered as C-68 rather than resolved.** The α
sweep in §3.4 and T1.2's sweep disagree at **α = 0.50**, and nothing in the repository had
noticed. §3.4's D2, starting each α from the prompt's iteration-0 residual tensor, settles on
`prolet` there, at margin 0.1144 with 15 of 15 tail iterations agreeing. T1.2, a continuation
sweep that starts each α from the previous α's settled state, settles on `comrade` at the same
α, in all four of its arms. Same site, same rule, same step size, same prompt, same 120 steps,
same ΔW to sixteen figures, same TransformerLens build. The difference is the initial state,
which means that at α = 0.50 two initial conditions reach two different settled words under
**identical weights**.

Two things follow, and it matters not to overstate either. **C-56 is untouched**: T1.1 tested
α = 1.00 directly, seeding the frozen `prolet` state, and it moved to `comrade`. But **the
broader gloss that the α sweep shows one continuously-moving fixed point at every α is
withdrawn**, and C-52's caveat now says so. It may be genuine bistability at α = 0.50, or it
may be a readout artifact: the D2 margin there is 0.1144 and the `prolet`/`comrade` logit gap
is 0.114, which is inside C-07's resolution limit. The test that would settle it is a
basin-of-attraction probe at that α, seeding several initial states with a convergence
criterion fixed in advance. It has not been run.

The step-3 results are untouched by this and are a different kind of claim. §3.3's
`A04_climate` lands on the **pre-existing** `Divine` orbit, which 34 of the 125 frozen
prompts already occupy, so it needs no new attractor (C-24, C-25), and lag-1 is a property of
the trajectory rather than of the readout, which is why no relabelling artifact can explain
it. §3.4's own class change between α = 1.25 and 1.50 also survives, at `provisional`, and it
too lands in `Divine` (C-28).

**One caveat governs all of it.** `comrade` and `prolet` sit at a settled margin of 0.321
with `prolet` still at rank 3, and the basin taxonomy's resolution limit is comparable to
the separation between basins: **pooled** mean within-basin spread 3.319e-03, over 2337 pairs across all five basins, against the nearest-basin
gap 2.874e-03, a ratio of 1.15 (**C-07**). Two basins this project treats as distinct are
no further apart than the prompts inside one of them.

---

## 4. Tooling inventory

| File | What it gives you |
|---|---|
| `plasticity.py` | `OjaPlasticity` — one site, modes `off`/`hebb`/`oja`/`anti_hebb`/`random`, ceiling, bit-exact `revert()`, `report()`. **`TermSpec` / `terms=`** for composed rules (see below). Site adapters for HuggingFace `Conv1D`, TransformerLens `W_out`, and **per-head stripes** on both. `subspace_projector()` for aiming drift. `candidate_sites()` |
| `multi_site.py` | **`MultiSitePlasticity`** — N sites at once (whole matrices and head stripes mixed, heterogeneous mode/eta/ceiling/projector per site). Rejects overlapping footprints at construction, keyed on live parameter identity |
| `atr_bridge.py` | `make_atr_step(model, prompt) -> step(model, r)` — one iteration of the parent loop, **extracted verbatim**, bit-exactness CI-enforced. `initial_state`, `load_state` |
| `controls.py` | C0 (eta=0 identity gate), C1 (revert), C2 (norm-matched random, multi-seed distribution), C3 (with the ceiling lifted at large eta, Hebb's drift keeps growing where Oja's saturates; the unqualified "Hebb diverges" is retired as C-15) |
| `offline_control.py` | Matched closed-vs-offline arms, a **17-axis** mechanical match verifier, and the severed-path control. At a head site the `recomputed`-y path rebuilds the *shared full* projection output additively, so its no-feedback floor is float32 noise rather than exactly zero (C-57) |
| `mea_grid.py` | The 12 × 12 addressable grid: per-head write lengths and per-block MLP writes at every iteration, and the depth-weighted activity centroid built from them. EXP-003 Stage 0's instrument |
| `mea_stim.py` | Signal injection at a site, as a unit vector scaled by a fraction of the activity already there, with a firing rate and the option of a whole-stream variant. Built for EXP-003 Stage 3, which has not run |
| `experiments/` | `baseline_basins.py` (the census + the canonical basin readout), `step_size_map.py`, `exp001_hebb.py`, `basin_bifurcation.py` — these three take `--site`. Then `exp002_distributed.py` (twelve layers plus the reprompt), `t2_1_coupling_sweep.py`, `rank1_random_control.py`, and `exp003_stage0.py` / `exp003_stage0_analyse.py` / `exp003_stage1.py` / `exp003_stage2.py`. Runners write to `<name>.partial` and rename on success, so an interrupted run cannot leave a half-written artifact behind |

**Composed terms — the #24/#25 balance.** A site's update can be an ordered sum of
signed, optionally-projected terms over the two primitives, instead of one `mode`
string. This is what makes "reinforce inside the target subspace, erode outside it"
(#25) expressible at a single site, and it is what EXP-002 step 2 needs:

```python
P = subspace_projector(basis)                    # (n_out, n_out)
OjaPlasticity(model, site, eta=1e-6, terms=[
    TermSpec("hebb",  +1, P),                    # reinforce inside the direction
    TermSpec("hebb",  -1, torch.eye(768) - P),   # erode around it
    TermSpec("decay", -1),                       # one brake over both
])
```

`terms=None` runs the original `mode` path unchanged; the three closed-form modes are
reproduced bit-exactly by their term spellings. `report()["mode"]` reads `"terms"` on
this path — the schema is unchanged, but anything grouping runs by `mode` will see it.
Covered by `tests/test_terms.py`.

**Reading a composed update — one trap.** At `blocks.6.mlp` the brake is ~110× the
reinforcement term (‖W⟨yyᵀ⟩‖ ≈ 110 × ‖⟨xyᵀ⟩‖), so a sign taken on the *total* update
reads "eroding" in **both** subspaces even when the balance is reinforcing correctly
inside P. Difference against a brake-only arm before concluding the balance is broken.

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

### 5.1 The primary experiment: EXP-002 (issue #24) — RUN, and what it found

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

**It ran** (`experiments/output_exp002/EXP_002_RESULTS.md`, pre-registered before the run,
issue #24 closed), at twelve plastic MLP layers rather than one, and the register carries
its four results at C-60 – C-63. In short:

- **The driven prompt's settled word moves, and the with-feedback and without-feedback runs
  land on different words** (C-60). `hebb` at 1.31% aggregate drift: `prolet` → `Rousse`
  with feedback, `comrade` without. All exact fixed points. The eta=0 gate is bit-identical
  and `revert()` is bit-exact on all twelve matrices.
- **Something does cross the prompt boundary** (C-61). All 31 fresh prompts reproduce the
  committed census before any drift, and after it the settled word changes on 31 of 31 in
  three of the four arms and 30 of 31 in the fourth. That is step 5 of the sequence above,
  and issue #29's argument applies: the residual stream is destroyed at the boundary, so it
  came through the weights or it came from nowhere.
- **What crosses is collapse, not steering** (C-62), and this is the part that matters for
  §5.2 below. The untouched model puts those 31 prompts on 5 distinct words. After `hebb`
  with feedback: 3 words, **27 of 31 on `Rousse`**. Without feedback: 2 words, 30 of 31 on
  `comrade`. `anti_hebb` without feedback: **1 word**, all 31 on `Shiv`. `anti_hebb` with
  feedback gives 19 words but only **4 of 31 at rest at the 120-iteration readout**, so those
  19 are snapshots of trajectories still in motion rather than settled states, and they are
  not surviving structure. **State that as a horizon-bounded result, not as an impossibility.**
  What is measured is that they had not settled by iteration 120, and that the phase-aware
  return test failed at 5 of 5 magnitudes, flooring at ~1e-4 whatever the perturbation size,
  with the best match the *last* iterate every time, which excludes a missed longer-period
  orbit. What is **not** measured is whether a longer run would settle. C-61 may never be
  quoted without C-62.
- **The severed-path control's exact-zero floor is a single-site guarantee and does not
  survive more than one plastic layer** (C-63). Within one forward pass a lower plastic
  layer's drift changes the activations arriving at a higher one, and severing the loop does
  not cut that. So every closed-versus-offline number in EXP-002 has **no zero baseline**.
  This amends standing rule 3 in §7 and was not anticipated by the pre-registration.

**Two regime notes.** The ceiling was **lifted** for this experiment by operator decision,
so nothing in it is continuous with the 5%-capped results above it. And the two arms are not
matched on drift, so rule-to-rule comparison is qualitative only.

Step 2 of the sequence, the Hebbian/anti-Hebbian **balance** at one site via `TermSpec`, is
still the piece that has never been run. EXP-002 ran the two rules separately, not balanced.

### 5.2 The direction that matters: escape, not collapse

Issue #27 item 5, stated before any result existed: *"Collapse is already the default,
so 'we caused collapse' is not a finding. The interesting direction is the opposite
one: **escape** — whether a balancing rule can lift the state back out of a well it
has fallen into."* Any framing that reports collapse as the achievement has the
experiment backwards.

**This is now the sharpest thing in the file, because EXP-002 produced collapse.** C-62
records it: every arm destroys the five-basin census, three of them by leaving one attractor
that swallows nearly everything, and the fourth by leaving most prompts unsettled at the
120-iteration readout. Issue #27 item 5 called that outcome in advance and called it a
non-finding. So the project has demonstrated the direction it said would not count, and has
not yet attempted the direction that would. Escape has never been tested, because step 2 of
§5.1's sequence — the balance — has never been run.

### 5.3 The ladder (issue #25) — and where we now are on it

1. Deepen an existing attractor.
2. Shift an existing attractor.
3. **Move a basin boundary** — "the measurable one, and the first real result."
4. **Create an attractor where none existed** — "expect this to fail first."

`BASIN_BIFURCATION.md` put the `comrade` result at **step 4**, and **T1.1 refuted that**
(§3.5). The supported reading is **step 2**: one fixed point the edit relocates
continuously, which the readout relabels across a ridge (C-56; C-26 `retired`; corroborated
by C-52's clean retrace).

**Where the project actually sits on the ladder:**

| Rung | Status |
|---|---|
| 1. Deepen an existing attractor | Never tested as such |
| 2. Shift an existing attractor | **C-56, `supported`** — the `comrade` result, 1 prompt, 1 site, 1 eta |
| 3. Move a basin boundary | **C-24 / C-25, `supported`** — `A04_climate` lands on the pre-existing `Divine` orbit and changes dynamical class with it. 1 prompt. Issue #25 calls this "the measurable one, and the first real result" |
| 4. Create an attractor where none existed | **Not achieved.** The one claim that reached for it is retired. Issue #25 expected this to fail first, and it did |

So the plan's expectations were **right**, not exceeded, and the earlier note here saying
otherwise was withdrawn. C-24 is the strongest row in the register, because lag-1 is a
property of the trajectory rather than of the readout, so no relabelling artifact can
explain a fixed point becoming a two-step cycle.

### 5.4 Open work not yet started

- **Every site family but one.** Twelve of twelve MLP output projections have carried
  plasticity; the 12 attention output projections and 144 head stripes have not (C-64).
  **eta does not transfer across sites** — `‖W0‖_F` and activation scale differ, so the
  anchoring formula must be re-measured per site (`STEP_SIZE_MAP.md` caveats).
- **The distributed regime has run, and it collapsed the census.** EXP-002 put all twelve
  MLP layers plastic at once, which is issue #25's "distributed damping... many small
  elements each acting on its own subspace". The finding is C-62, and C-63 is the control
  guarantee it cost. What has *not* run is many sites at once **inside** the 5% ceiling, or
  at any site family other than the MLP projections.
- **Issue #31 — within-basin spread. This ran; the answer is compression.** Prompts in one
  basin settle onto nearby-but-distinct states, not the same state, so the settled state is
  more than a 5-way label: mean `1−cos` 3.32e-03 over 2337 pairs, and the variation is
  essentially one-dimensional (participation ratio 1.02–1.29 against a maximum of 15–54).
  Position uniformity holds for 125 of 125 prompts, not only for `Divine`. C-04 – C-06,
  measured in `BASELINE.md`, with the interpretation fixed in advance as the issue required.
  It also produced **C-07**, the resolution-limit caveat that now governs every basin claim
  in the repo.
- **Issue #28 — prior-art gaps.** The Chaudhary 2025 claim has been pinned down and is
  register row C-44, `supported`, but read the caveat before using it: the paper reports **no
  4-layer result at all**, so "stable around 4" is interpolated, and **nothing in that
  literature tests 12 layers with any rule**, so it cannot set an expectation either way for
  GPT-2 small. Use it as corroboration of the family split and never as a prediction for this
  substrate. Also still open: C-42's eleven absence claims have no preserved search artifact
  (T0.6), and C-43, on how this work compares to model editing such as ROME, is unanswered.
- **The drive term β, the leak term α, target-energy** — `DESIGN.md` and
  `ISSUE_normalisation_homeostasis.md` (E1–E4). Cadence has since been run and is no longer
  in this list: 2 and 4 in T2.1, 4 and 12 in EXP-003 Stage 2 (C-58, C-64).
- **EXP-003 stages 3 to 5.** Pre-registered in `experiments/output_exp003/PREREGISTRATION.md`
  and not run. Stage 3 injects a signal at a fraction of the local activity and compares
  focal against distributed placement; Stages 4 and 5 are gated behind it, and Stage 3 also
  needs `mea_stim.py`, which exists and has never been used in a committed run. Two
  measurements registered for Stage 1 were also never implemented and are recorded as not run:
  the depth-weighted centre of the weight change across the twelve sites, and the smallest
  weight change at which each statistic separates the drifted system from the frozen one
  (C-66).

---

## 6. Open decisions for the operator

1. **`EXP_001_SPEC.md` is self-inconsistent and it was left that way deliberately.**
   Its title is still *"Does the `Divine` period-2 cycle survive plasticity?"* and its
   header still says **"Status: proposed, not run"**, while `EXP_001_RESULTS.md`
   reports a *different* experiment (the offline-control / basin-flip run). The label
   "EXP-001" got reused. Either retitle the spec to the experiment that ran and give
   the Divine-cycle question its own spec, or flip the status and keep them separate.
   §0 and §5.3 of that file were corrected and are right either way.
2. **Whether to record the α-cascade as a headline finding.** Partly settled since this was
   written: `README.md` now leads with **editability** rather than with the cascade, and the
   cascade's own class change is C-28, `provisional`, and lands in the pre-existing `Divine`
   basin rather than a new one. So the honest version of this decision is narrower — whether
   C-28 is worth promoting, which needs T1.3 to resolve the α interval first.
3. **CLOSED 2026-08-05: EXP-003 now has register rows.** It previously did not, and this
   entry asked whether it should. It should have, because this register's own first rule is
   that a claim enters it before it enters any prose document, and three stages of
   pre-registered measurement were sitting in committed prose outside it. They are now
   **C-65** (the statistic separates the five end states and **fails** its registered gate on
   dynamical class, so it is blind to the distinction carrying C-24), **C-66** (the
   spectral-concentration mechanism this project proposed for its own collapse is **refuted**
   by its own pre-registered threshold, 0.044% against a 2% floor), and **C-67** (census
   agreement is 0 of 31 at every cadence tested, while the cadence comparison itself is *not*
   established because the drift guard fired at 6.03× against a required factor of 2). What
   stays out of the register, and stays out deliberately, is the cultured-network argument
   the operator had removed; only the borrowed statistics remain, sourced in `MEA_SOURCES.md`.
   **What is left for you here is narrower**: whether Stage 3 should run at all, given that
   Stage 0 showed the instrument it depends on cannot see dynamical class.
4. **Whether `docs/voice.md`'s prohibition on em dashes applies to repository text.** The
   guide says it does, in as many words. In practice every document written since it landed
   still uses them, including all four EXP-003 files, and `CLAIMS.md` uses them throughout.
   Either the guide governs and the register needs a pass, or it is about writing to the
   operator and repository prose is out of its scope. It should not stay ambiguous.

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
- **Report the clip state on every result.** A run where the ceiling fired is a
  measurement of the ceiling, not of the rule. Note that **the library does not give you a
  rate**: `report()` returns `clipped`, a latching boolean that goes true on the first clip
  and clears only on `revert()`, so any rate in this repo is an experiment script's own work
  (C-46, which retired the claim that a rate is recorded).
- **State the ceiling regime, because there are now two.** Everything up to and including
  T2.1b was taken under the 5% cap. EXP-002 and EXP-003 were taken with the cap **lifted**,
  by operator decision. Results from the two regimes may not be quoted as continuous, and a
  reader has to be told which one a number came from (C-60).
- **The severed-path floor is exactly zero at one plastic site and is not zero beyond it.**
  Within a single forward pass a lower plastic layer's drift changes the activations arriving
  at a higher one, which severing the loop does not cut. So a multi-site closed-versus-offline
  number has **no zero baseline** and may not be placed in a series with the single-site
  shares (C-63, which amends the register's standing rule 3; the pre-registration did not
  anticipate this).
- **Never use an even-only snapshot schedule.** It samples a period-2 orbit at one
  phase and makes oscillation invisible by construction — this hid the parent's F9
  finding for months. Log lag-1 *and* lag-2.
- **Justify any convergence horizon against the contraction factor** — and measure that
  factor on the trajectory you are actually running, because **it is not a constant**. The
  often-quoted ~0.968 per iteration, ~71 iterations per decade, is an endpoint slope rather
  than a fit; across the perturbation ladder the per-decade rate ranges **4 to 124**, and
  `EXP_001_RESULTS.md` §6 measures 0.9521, or 47 per decade, on a different trajectory
  (C-03's caveat). A 200-iteration horizon has already produced one false "failed to return"
  in this repo.
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

*Last verified against main `754cc26`, 2026-08-05. If the code and this file disagree, the
code is right and this file is stale — check `git log` since that commit. If `CLAIMS.md` and
this file disagree, the register is right, and that is true whichever is newer.*
