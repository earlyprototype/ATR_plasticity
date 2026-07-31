# Alignment review

*A leadership review of the whole repository — code, artifacts, documents, issues, pull
requests and the agent board — against `main` at `ea4f0c1`. Its job is to say where the
project's claims and its evidence have come apart, and what to do about it.*

**Companion file:** `CLAIMS.md` — the standing claim register this review proposes.
Read this file once; read that one before every write-up.

---

## 1. Verdict

**The measurement layer is excellent. The summary layer has drifted away from it — in
both directions at once — and the project has stopped knowing what it owns.**

The apparatus is better than most published interpretability work: a bit-exact bridge to
the parent engine enforced in CI, 295 tests against real weights, controls tested in both
directions, a 17-axis mechanical match verifier, a severed-path null that is *literally
zero* rather than merely small, and a pre-registered list of failure modes that look like
findings. Three independent audits checked the arithmetic against the committed JSONL.
**No fabricated number was found anywhere.** Essentially every figure in every document
reproduces exactly from the artifacts.

What has gone wrong is entirely above the numbers. Three drifts:

| | Finding |
|---|---|
| **Under-claimed** | The coupling result. The top-level summary tells the next reader feedback did nothing; the evidence says it steers the update, against a null of exactly zero. **F1** |
| **Over-claimed** | The bifurcation. `BASIN_BIFURCATION.md` reports a created attractor; the repo's own baseline shows the `comrade` state sits *well inside* the `prolet` basin's ordinary scatter. **F2** |
| **Unknown to itself** | Issue #31 — "the cheapest measurement in the project", listed as unrun in three places — was run, and is committed in `BASELINE.md`. Its answer changes how every basin result reads. **F3** |

F3 is the one that explains the other two. **No artifact in this repository maps a claim
to its evidence**, so a measurement can be made and lost, and a correction can land in one
file while three others keep the superseded version. §4 proposes the fix; it costs one file.

Separately, a full code audit found the whole-matrix path defended to the last bit and **one
high-severity latent defect on the per-head path** — invisible to all 17 matched axes *and*
to the severed-path control, and sitting directly in front of the next planned experiment.
No published result is affected; every committed artifact is `blocks.6.mlp`. **F11**

Both F1 and F2 entered through pull requests that received **no peer review**, in a project
whose review mechanism had previously caught two defects that changed conclusions (**F9**).

The strategic recommendation (§6) is that the project stop leading with *coupling* and
start leading with *editability of iterated dynamics* — a claim it has already proved,
under better control than it realises (**F4**), and is not making.

---

## 2. What the project actually has

Stated first, because the rest is critical and the asset is real.

**A properly controlled direction-specificity result — better controlled than the repo
claims.** See F4. Three near-rank-1 weight updates at matched Frobenius norm *and* matched
operator norm; one flips the basin, two do not.

**A dynamical-class change.** `A04_climate` goes from a fixed point (lag-1 0.99999) to the
period-2 `Divine` orbit (lag-1 0.661). lag-1 is a property of the trajectory, not of the
readout, so this one cannot be a naming artifact. **It is the strongest single result in
the repository** and it is currently the third thing mentioned in a table.

**A measured coupling contribution**, against a null of exactly zero. See F1.

**A predicted failure mode ruled out with data.** Issue #27 item 11 predicted hollowing-out
— one entry runs away, the normaliser rescales, the matrix collapses while every dial looks
healthy. Fully instrumented, and it did not happen: effective rank 642.64 → 642.57 (0.011%
drop), max-over-mean 33.38 → 33.53. That is worth as much as a positive result and lives
in a JSON file.

**An answer to issue #31** that nobody has noticed they have. See F3.

**Genuine epistemic hygiene.** Pre-registered criteria; interpretations marked as such;
retracted claims recorded rather than deleted; three deterministic seeds reported honestly
as *one run three times* rather than as "n=3, spread 0.000". And an offline control that
undercut the project's own headline, published anyway. That is rare and worth protecting.

---

## 3. Findings

### F1 — The coupling result is under-claimed; the summary layer says the opposite of the evidence

`EXP_001_RESULTS.md` §3, zero-floor `recomputed` mode:

| configuration | `diff_over_drift` | meaning |
|---|---|---|
| routed (loop 0→11, plastic site **inside** the loop) | **0.120** | feedback possible |
| severed (loop 0→3, plastic site **downstream**) | **0.000** | feedback impossible by construction |

The severed arms are **bit-identical** — `torch.equal` returns True, `rel_fro_diff` is
exactly `0.0`. That is the correct and complete null: with no feedback path the two arms
compute the same function of the same inputs and must agree exactly. They do. Any departure
in the routed configuration is attributable to feedback, and the instrument's own round-off
floor (`1 − cos ≈ 1.5e-14`) is ten orders of magnitude below the routed signal:

| prompt | `1 − cos` routed | `1 − cos` severed | ratio |
|---|---|---|---|
| `A01_physics` | 7.06e-03 | 3.9e-13 | 1.8e+10 |
| `A02_medical` | 6.08e-03 | 3.2e-13 | 1.9e+10 |
| `A04_climate` | 6.17e-03 | 2.9e-13 | 2.1e+10 |

And it is a **direction** change, not a scale change. Decomposing the difference against
`ΔW_closed` and `‖ΔW_closed‖`: perpendicular **0.1220**, parallel **0.0213** — **5.73 to 1
in favour of direction.**

> *Correction to `EXP_001_RESULTS.md:133`.* The published figures 0.1153 / 0.0346 (ratio
> 3.33) are arithmetically correct but decomposed against **`ΔW_offline`**, not
> `ΔW_closed` as the prose states. Against the stated reference the ratio is 5.73. The
> direction-dominance is *stronger* than published; only the label is wrong.

**What the summary layer tells a new reader.** `HANDOVER.md` §3.3 — the first file anyone
picks this project up from:

> "The two agree — `cos(ΔW_closed, ΔW_offline) = 0.99294` … if a coupling claim is ever
> made, that is the bar it has to clear."

And the comment on issue #32:

> "near 1 means feedback changed nothing but scale. That is what was measured."

Both are wrong the same way. Issue #26 pre-registered the rubric "near 1 → nothing but
scale" *before the severed control existed to calibrate what "near 1" means*. Once the null
is known to be exactly 1, a cosine of 0.993 is not near 1 — it is 1.8 × 10¹⁰ floors away.
The rubric was applied as written instead of re-read against the control that had since
been built.

**This is a process failure, not a measurement failure.** A peer agent on PR #22 caught
exactly this and said so on the board — *"`EXP_001_RESULTS.md:131` said 'Near 1 means
feedback changed nothing but scale.' Exactly backwards."* The correction landed in
`EXP_001_RESULTS.md` §3–4. It never propagated to `HANDOVER.md`, `ORIENTATION.md`, or the
issue thread.

**The defensible claim:**

> In the closed loop, feedback measurably steers the weight change. Against a severed-path
> null of exactly zero, the feedback-attributable component is 12% of the total drift and
> is predominantly a change of direction rather than magnitude (5.73:1). At this step size
> and horizon that steering does **not** change the behavioural outcome — both arms flip
> to the same basin.

Both halves matter. The first is the coupling result the project has been looking for; the
second is the honest limit, and it makes the next experiment obvious (T1.1).

**Caveats that must travel with the number.** The severed arm runs a shallower loop (0→3)
at 2.6× the drift (2.9% vs 1.1%), so it is not a statistically matched control — the
argument for it as a null is *structural* (no feedback path ⇒ bit-exact agreement) and that
argument holds, but the mismatch belongs beside the number. Independent n is **3 prompts**,
not the 5 cells the tables imply (see F6).

---

### F2 — The bifurcation claim is refuted by the repo's own baseline

`BASIN_BIFURCATION.md` is titled *"`comrade` is a created attractor"* and places the result
at issue #25's ladder **step 4** — "create an attractor where none existed", which the
recorded plan expected to fail first. `HANDOVER.md` §5.3 propagates it: *"further up the
ladder than the plan anticipated… it should recalibrate expectations for everything
downstream."*

It should not. Three independent arguments, in increasing order of severity.

**(a) The α-sweep shows a smoothly deforming fixed point, not a bifurcation.**

| α | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 |
|---|---|---|---|---|---|---|---|
| basin | `prolet` | `prolet` | `prolet` | `comrade` | `comrade` | `comrade` | `Divine` |
| lag-1 | 1.000000 | 1.000000 | 1.000000 | 0.999999 | 0.999999 | 0.999998 | **0.733997** |
| ‖state‖ | 4782.8 | 4794.4 | 4807.3 | 4821.6 | 4836.9 | 4851.8 | 4917.7 |
| Δ‖state‖ | — | +11.6 | +12.9 | +14.3 | +15.3 | +14.9 | **+65.9** |

Across α = 0 → 1.25 there is one fixed point and it moves smoothly: lag-1 pinned at 1.0,
state norm advancing in near-equal increments, no discontinuity anywhere near α\* = 0.75.
What changes discretely at α\* is the **argmax of the readout** — and an argmax always
changes discretely, including under perfectly smooth motion. The document says this and
treats it as corroboration: *"The argmax (the basin) is discrete; the logit it is the argmax
**of** is not. Smooth logits, discrete attractor — the signature the result turns on."*
That is the signature of a **relabelled, continuously-moving fixed point** — ladder step 2.
The crossing is driven mainly by `prolet`'s logit *falling* (16.95 → 16.07) rather than
`comrade`'s rising (16.29 → 16.25): suppression of the incumbent, which is a steering-vector
signature, not a new attractor.

**(b) D1 does not discriminate what it claims to.** For any ΔW ≠ 0 the perturbed map's fixed
point is generically not a fixed point of the unperturbed map. D1 correctly falsifies "the
episode walked into a pre-existing `comrade` basin of W₀" (step 3) — but step 4 is then the
*residual by elimination*, and a norm-matched random edit would pass D1 identically. D1's own
trace argues for step 2: released under W₀ the state relaxes back to `prolet` smoothly and
monotonically, lag-1 above 0.99992 at every sampled iteration.

**(c) The decisive number, from the repo's own baseline.** `BASELINE.md` measures the
`prolet` basin's internal scatter. Set the `comrade` displacement against it:

| quantity | `1 − cos` |
|---|---|
| `comrade` state vs the `prolet` fixed point (D1, iter 200) | **4.39e-03** |
| mean pairwise spread **within** the `prolet` basin (55 prompts) | 2.77e-03 |
| **worst** pair within the `prolet` basin (cos 0.966079) | **3.39e-02** |
| gap between the two *nearest genuine basins* (`Anarch`–`prolet`) | 2.87e-03 |

**The `comrade` state is 7.7× closer to the `prolet` fixed point than the two most distant
members of the `prolet` basin are to each other.** By the project's own basin metric,
`comrade` sits comfortably inside `prolet`'s ordinary scatter — at 13% of that basin's own
diameter. It is 1.5× the gap between two basins the project already treats as distinct,
which means `comrade` and the pre-existing `Anarch`/`prolet` distinction stand or fall
together.

This does not make the result worthless. It **calibrates** it: the basin taxonomy has a
resolution of order `1 − cos` ≈ 3e-3, and the `comrade` displacement is the same order.
A claim of a *created attractor* needs to clear that resolution and does not.

**(d) The file never mentions the offline arm.** `grep -i "offline\|coupling\|feedback"` on
`BASIN_BIFURCATION.md` returns **zero hits**. But the offline arm — no feedback path at all
— flips the same basin, and the commit that produced EXP-001 is titled *"the basin flip is
the rule, not the coupling."* `BASIN_BIFURCATION.md` speaks throughout of what "the episode"
did. Read alone — and it will be, it is the newest result — it supports the conclusion that
the closed loop created an attractor. The repo's own control says the loop was not required.
One sentence fixes this and it must be added.

**The decisive test was not run, and it is cheap.** Under `W0 + ΔW`, seed the *original
frozen* `prolet` state and iterate. Stays `prolet` → two attractors coexist → step 4 stands.
Moves to `comrade` → one displaced attractor → step 2. ~1 CPU-minute. Issue #26 also already
specifies the hysteresis test that would settle it independently — *"Up-then-down eta sweep…
that hysteresis is the cheapest available evidence of a real transition. Almost nobody does
it."* Specified, not run.

**The one genuine qualitative transition is elsewhere and is under-reported.** Between
α = 1.25 and 1.50 lag-1 collapses 0.999998 → 0.734 and the state norm jumps 65.9 against a
running increment of ~15. A fixed point gives way to a period-2 cycle — a real change of
dynamical class. But `Divine` is a **pre-existing** attractor (34 of 125 baseline prompts),
so this is ladder **step 3**, a boundary move, and it sits at 1.5× the ΔW the episode
produced.

**Net effect on the ladder.** The project has a solid **step 3** — `A04_climate` crossing
into the pre-existing `Divine` cycle at α = 1, with a dynamical-class change no readout
artifact can explain. Issue #25 calls step 3 *"the measurable one, and the first real
result."* It is being under-sold in favour of a step-4 claim the data do not carry.
**Swap the emphasis.**

---

### F3 — Issue #31 has already been answered, and the project does not know it

Issue #31 — *within-basin spread: does the attractor compress the prompt, or erase it?* —
is **open**. `HANDOVER.md` §5.4 lists it under "Open work not yet started" and calls it
*"the cheapest measurement in the project… it changes how everything else reads."*
`RESONANCE_NOTE.md` calls it the first thing it would run.

**It was run.** `experiments/output_baseline/BASELINE.md` contains the entire measurement,
committed, with the interpretation fixed in advance exactly as the issue demanded:

| measurement | result |
|---|---|
| within-basin spread, mean `1 − cos` | 3.32e-03 (2337 pairs) |
| between-basin spread, mean | 1.85e-01 (5413 pairs) |
| ratio within/between | 1.80e-02 |
| **ratio against the *nearest* basin pair** | **1.16** |
| within-basin effective dimensionality (participation ratio) | **1.02 – 1.29** across basins |
| position uniformity | **125/125 prompts, every basin** — not a `Divine` property |

**The answer is compression, not erasure** — the issue's "more interesting" outcome, the
one that makes the persistence work worth doing. Prompts in a basin land on nearby-but-
distinct states, and the within-basin variation is essentially **one-dimensional**
(participation ratio 1.02–1.29 against a maximum of n−1 = 15–54).

And the sting: **a ratio of 1.16 against the nearest basin pair means the two closest
basins are no further apart than the prompts inside one of them.** `BASELINE.md` reports
this candidly — it undercuts the basin construct and was published anyway. That number is
the correct lens for every basin result in the repository, and no downstream document cites
it.

**One correction to the plan.** `HANDOVER.md` and issue #31 both assume this can be re-run
"on saved arrays". It cannot: `BASELINE.md` promises 125 settled states in
`experiments/output_baseline/states/`, but `.gitignore:27` excludes `experiments/**/states/`
and the directory does not exist in the repo. The *summary statistics* survive; the raw
states do not. Any extension needs the ~6 CPU-hour baseline re-run. See F8.

---

### F4 — The decisive control the project needs is already in its data, and the one it cites cannot discriminate

`README.md` calls C2 — the norm-matched random arm — *"the one that decides whether this
branch is interesting."* As implemented it **cannot decide anything**, because it is matched
on the wrong quantity.

`random` is matched to Oja's update in **Frobenius norm**. But an isotropic 3072×768 matrix
spreads that norm across ~719 singular directions, while a Hebbian update concentrates it in
one. From the step-size map's own spectra:

| mode | rel ΔW | ‖ΔW‖_F | σ₁ | σ₁/‖ΔW‖_F | basin |
|---|---|---|---|---|---|
| `random` (every eta) | — | — | — | **0.054** | `prolet` |
| `hebb` @ 7.07e-05 | 0.0112 | 1.8529 | 1.8135 | 0.979 | **`comrade`** |

At its largest clean drift (1.84%, above `hebb`'s 1.12%) `random`'s σ₁ is 0.164 — **11×
smaller** than `hebb`'s. Even pinned at the 5% ceiling it reaches σ₁ = 0.447, still **4×
smaller**. **The random arm never reaches `hebb`'s operator norm anywhere in the sweep.**
"Random doesn't move the basin" is therefore consistent with "random is operator-norm tiny",
and the published conclusion — *"the basin change is specific to the update's direction, not
its magnitude"* — is not established by that comparison.

**But the project already has the right control, and has never pointed at it.** `oja` and
`anti_hebb` are also near-rank-1, and at the right eta they match `hebb` on *both* norms:

| mode | eta | rel ΔW | ‖ΔW‖_F | **σ₁** | clip | basin |
|---|---|---|---|---|---|---|
| `hebb` | 7.065e-05 | 0.0112 | 1.8529 | **1.8135** | 0.0% | **`comrade`** ← flips |
| `oja` | 2.944e-06 | 0.0119 | 1.9651 | **1.9288** | 0.0% | `prolet` |
| `anti_hebb` | 2.944e-06 | 0.0116 | 1.9131 | **1.8745** | 0.0% | `prolet` |

Three near-rank-1 updates at the same site, matched on Frobenius norm *and* operator norm
*and* drift, all ceiling-silent, differing **only in direction**. One flips the basin; two do
not. And `oja` at 2.9% drift reaches σ₁ = 4.68 — **2.6× `hebb`'s** — and still does not flip.

**That is the direction-specificity result, properly controlled, already measured, and
requiring no new compute.** It is strictly stronger than the C2 comparison the README leans
on. Promote it; retire the isotropic comparison as the decisive one; and run the genuinely
missing control — a **rank-1 random direction at matched σ₁** — which is cheap and can
falsify the whole thing.

**Related, and it explains everything above.** With the repo's convention (`y = x@W`,
`dW = E[x yᵀ]`), the Hebbian update is exactly

> **ΔW = E[x xᵀ] W = C·W**

— one step of power iteration on the site's input second-moment matrix. Since the attractor
states are position-uniform (F3: 125/125), C is effectively rank-1, so
**ΔW ≈ η·N·λ₁·v₁(v₁ᵀW): a rank-1 weight edit along the site's dominant activation mode.**
The repo's own measurements confirm it (95.8% of ‖ΔW‖²_F in component 1, stable rank 1.04).
Three consequences the write-up must confront:

1. **"No loss" is weaker than stated.** Plain Hebb is exactly gradient ascent on ½E‖y‖²;
   Oja is the same under a norm constraint. There *is* an implicit objective. The defensible
   phrasing is "no **externally specified** objective" — a much smaller gap from unsupervised
   test-time adaptation than "no loss at all".
2. **The offline/closed agreement is predicted, not surprising**, because C barely moves.
3. **Stating the identity makes the result more credible, not less**, and pre-empts the
   reviewer who derives it first.

---

### F5 — The rule the project is built around does nothing; the rule its README dismisses produces every result

| rule | moves the basin? | at what drift |
|---|---|---|
| `hebb` | **yes** | 1.12% and 2.20%, ceiling silent |
| `oja` | no | up to 2.9% clean, 5% at 100% clip |
| `anti_hebb` | no | only at 100% clip |
| `random` | no | up to 1.84% clean, 5% clipped |

`oja` is inert at every step size tested. Every headline number in the repository is a
`hebb` number. The repository is nonetheless organised around Oja: the class is
`OjaPlasticity`; the README carries *"Why Oja rather than Hebb"*; and `PRIOR_ART.md:81`
transfers external risk arguments on the basis that *"We run an Oja rule"* — keyed to an arm
that produces no result.

The README's stated reason — *"Raw Hebbian updates diverge immediately — no fixed point,
unbounded weight growth"* — is **empirically false at the working point**: 120 `hebb` updates
take ‖W‖_F from 164.854 to 164.907, clip rate 0.0%, non-finite count 0. It did not diverge.
Control **C3** ("Does raw Hebb diverge and Oja not?") has a premise the step-size map
contradicts inside the usable band.

This is not a reason to abandon Oja. It is a reason to make **Oja's inertness a stated
finding with a mechanism**. The explanation is already in the repo: at this site the decay
term is ~110× the reinforcement term (`HANDOVER.md` §4), so Oja's update is almost entirely
brake. A rule that is 99% brake tracks the dominant activation direction and goes nowhere
else. That is a publishable observation about Oja inside a pretrained transformer, and it
agrees with the prior-art expectation of saturation.

---

### F6 — Sample size, and three arithmetic errors

**n is smaller than the tables imply.** `EXP_001_RESULTS.md` §3 reports medians over "n=5",
but three of the five cells are bit-identical replicates of `A01_physics` — `hebb` has no
stochastic term, so the seeds are one run reported three times. The document states this
two sections later but does not retract the n=5 medians, and the triplication *forces* the
median onto A01's value, which is also the extreme of the range in three of four rows.
**Real independent n = 3.**

Coverage overall: 3 prompts (1 for the step-size map and the entire bifurcation analysis),
1 site of 168 candidates, 1 eta, cadence 1, 1 model. The ΔW decode and the C1 revert are
each **n = 1**.

**Errors of fact to correct:**

| # | Location | Says | Actually |
|---|---|---|---|
| E1 | `EXP_001_RESULTS.md` intro, `HANDOVER.md:96` | `hebb`@7.07e-05 is "the **only** cell in the whole sweep that moves the loop inside a clean band" | **Two** clean flipping cells: `hebb`@7.07e-05 (1.12% drift) *and* `hebb`@1.18e-04 (2.20% drift), both 0.0% clip, both `comrade`. This is good news — an independent replication — and it is being suppressed by an error |
| E2 | `EXP_001_RESULTS.md` §4 | ΔW effective rank "1.8–3.8 for `oja`" | `oja`'s range is **1.0–2.9**. The 3.8 is `anti_hebb`@9.81e-05, a cell at **60.8% clip** — which by the repo's own rule should not be cited at all |
| E3 | `EXP_001_RESULTS.md:133` | perp/parallel decomposed against `ΔW_closed` | Decomposed against `ΔW_offline`. Against the stated reference: 0.1220 / 0.0213, ratio **5.73** (see F1) |

**Two more citation-hygiene problems of the same class:** `STEP_SIZE_MAP.md` §5's evidence
that effective rank *rises* is taken from `anti_hebb`@9.81e-05 at 60.8% clip; and
`HANDOVER.md:90` cites the `anarchism` flip without noting it occurs at **100% clip**. The
null conclusions stand from the clip-free cells; the specific numbers should not be quoted.

**C2 was never run in EXP-001 at all.** There is no `mode="random"` cell in
`exp001.jsonl`. The "random control #0/#1" columns in §4 are isotropic *decode* directions
for the logit lens, not plasticity arms — easily mistaken for C2 by a reader. The only
random-plasticity data is 8 step-size cells at seed 0, with no cell at the flip's 1.12%
drift (bracketing cells are 0.61% and 1.84%).

**One test has almost no power and is reported as an absence.** §4 concludes the ΔW direction
"carries no measurable preference" for the basin tokens. The null 5–95% bands are
`prolet [7.7, 98.4]`, `comrade [7.1, 98.9]` — a 90% band spanning 91 percentile points cannot
reject anything. "Inside the band" here is uninformative, not a demonstration of absence.

---

### F7 — Document drift

Errors of fact currently live on `main`, beyond those in F6:

| Location | Says | Actually |
|---|---|---|
| `README.md` (×3) | "227 tests" | **295** collected (verified) |
| `README.md` "Why Oja rather than Hebb" | Hebb diverges immediately, unbounded growth | 0% clip, 0 non-finite, ‖W‖_F +0.03% (F5) |
| `README.md:65` | "Rung 1 is the ε→0 limit of rung 4" | Formally **retracted** in `PRIOR_ART.md`; still asserted in README |
| `README.md:165` | "Collapse is the likely default" | Contradicted by the repo's data: effective rank flat at ~642/768 and *rising* |
| `HANDOVER.md:319` | Chaudhary depth figures **unverified**; "instability is the expected result" | `PRIOR_ART.md` records them **verified** and attributes divergence to the *gradient-plastic* variant — the Hebbian family stayed stable and **saturated**. **Two live documents predict opposite failure modes from the same citation** |
| `HANDOVER.md` §3.3 | "The two agree"; coupling bar not cleared | Routed difference entirely above a floor of exactly zero (F1) |
| `HANDOVER.md` §5.3 | Result is at ladder step 4 | Step 2 for `comrade`, step 3 for `A04`/`Divine` (F2) |
| `HANDOVER.md` §5.4 | Issue #31 "not yet started" | Run and committed in `BASELINE.md` (F3) |
| `HANDOVER.md:363` | Contraction 0.968, "~71 iterations per decade", promoted to a repo-wide rule | Not a fit — the per-decade rate ranges 4 to 124 across the perturbation ladder, and its low end returns trivially. `EXP_001_RESULTS.md` §6 measures **0.9521 (47/decade)** and flags the discrepancy |
| `ORIENTATION.md` §"The offline arm" | Coupling claim "lives entirely in the difference" — presented as unmeasured | Measured; the number is 0.120 |
| `EXP_001_SPEC.md` header | "Status: proposed, not run", titled for the `Divine`-cycle question | Describes a **different experiment** from the one `EXP_001_RESULTS.md` reports |
| `EXP_001_SPEC.md` §0 | Points at `EXP_001_RESULTS.md` for the C0 gate | That file contains **no C0 section**. C0 on the real stack was never run in the experiment — only in pytest and the map's `off` cell |
| `PRIOR_ART.md:28` | "every entry has been verified against its source" | Four entries carry no status, two no authors, one no identifier |

The `EXP_001_SPEC.md` collision is logged as open decision #1 in `HANDOVER.md` §6 and was
left deliberately. It should now be closed: the label denotes two different experiments
across two files, and the spec's §5 control ladder reads as satisfied when only C1 was run,
on a different object.

Nothing here is dishonest. Documents were corrected at the point of measurement and the
corrections did not propagate. That is F9.

---

### F8 — Reproducibility and provenance

- **Raw state is not persisted.** `BASELINE.md` promises 125 settled states for exactly the
  within-basin analysis; `.gitignore:27` excludes them and they are absent. Dense ΔW is not
  persisted either, so `basin_bifurcation.py` must **reproduce** the EXP-001 episode rather
  than load it — it matches the weight anchors to ~9 figures but lands 0.12% off on the
  closed-loop state norm. Every future ΔW analysis pays that cost and compounds the drift.
  ΔW is ~9 MB; issue #32 §5 already treats it as a reusable object. **Persist it.**
- **Version skew nobody documents.** `basin_bifurcation` ran under transformer_lens
  **3.6.0**; baseline, step-size map and EXP-001 under **3.5.1**. A ~0.1%-class state-norm
  drift between them is already visible.
- **TransformerLens is deprecating the entry point everything rests on.** Since provenance
  rests on bit-exact reproduction of the parent loop, a v3 numerics shift could invalidate
  it silently. **Pin the version now.**
- **`STEP_SIZE_MAP.md` stamps a single `repo_rev`** while admitting it was built over
  multiple invocations, and its `wall_clock_seconds: 53.2` covers only the 2 refinement
  cells of 35. `EXP_001_RESULTS.md` handles this correctly with a `provenance_warning` and a
  two-revision list; the map should match.
- **C0 has flickered twice on CPU** (~8.6e-05, 6.3e-05), unreproducible in 80 controlled
  repeats and 16 cold processes. Best explanation is non-deterministic float reduction order.
  Keep it visible — C0 is the gate on everything.

---

### F9 — Peer review lapsed exactly when the project started making claims

The board works, and it caught two defects that changed conclusions:

- **PR #8** — `mode="random"` was norm-matched to the raw Hebb term rather than the full Oja
  update, biasing **C2**. Found on the board, fixed.
- **PR #22** — `diff_over_drift` conflating angle with scale, and the "near 1 means nothing
  but scale" reading being backwards. Found by a peer agent, acted on, conclusion changed.

Then it stopped.

| PR | What it shipped | Board participants | Bot reviews |
|---|---|---|---|
| #8 | CI, test suite, 3 fixes | active | yes |
| #22 | prior art, baseline, step-size map, EXP-001 | 2 agents, 3 threads | 8 |
| #33 | measurement/opinion separation | **0** | — |
| #35 | **`BASIN_BIFURCATION.md` — the step-4 claim** | **0** | **0** |
| #37 | **`HANDOVER.md` — the coupling misreading** | **0** | 1 |

The two documents carrying this review's two most significant findings were merged with no
peer review; PR #35 had none at all, bot or agent. Review intensity collapsed at precisely
the transition from building apparatus to making claims.

`PEER_BOARD_SETUP.md` notes the mechanism is advisory — *"nothing blocks a merge"* — and
that upgrading to a blocking gate is *"worth doing only once real threads show flags are
being posted and are accurate."* That condition was met at PR #22. F9 is what the advisory
version cost.

---

### F10 — The novelty claim rests on unpreserved searches and misses six adjacent literatures

`PRIOR_ART.md` is honest that its verdict is *a statement about a search, not about the
literature*. Two structural problems remain.

**The absence claims have no artifact.** Eleven of them carry the novelty — "no work matching
the four-way combination", "16 forward citations of Daydreaming and not one applies it to a
transformer", "no published Oja-family step size inside a pretrained transformer" (which
licenses the bespoke eta anchor, and therefore every number in the repo). None has a
preserved query list, date, endpoint or returned ID. **Committing the search artifacts costs
hours and converts eleven assertions into a record.**

**The search was term-driven, not concept-driven.** The stated inclusion rule — *any work
combining two or more of: local unsupervised rule, pretrained frozen model, closed activation
loop, no objective* — should have returned the following. None appears:

| Area | Collision risk | What it forces |
|---|---|---|
| **Test-time adaptation** — TENT (Wang 2021), TTT, SHOT | **Highest.** TENT is literally: frozen pretrained model, unsupervised weight update at inference driven by the model's own outputs. The only separator is its entropy loss — and per F4 Hebb has an implicit objective too | Must be a named section |
| **Model editing** — ROME (Meng 2022), MEMIT | **Very high.** ROME makes a **rank-1 update to a mid-stack MLP down-projection of GPT-2** and changes the output. Same matrix family, same depth region, same rank as this project's ΔW | Differentiator is real — ROME *solves* for ΔW from a target, this one emerges from activation statistics with none — but it must be stated |
| **Fast weights** — Ba et al. 2016, Schmidhuber 1992, Irie 2022 SRWM | **High.** Ba et al. is a Hebbian outer-product matrix updated from the network's own hidden states inside an iterated inner loop run to a fixed point — three of four ingredients | Cite explicitly; differentiate on decay-to-zero, separate matrix, end-to-end training |
| **Modern Hopfield ≡ attention** — Ramsauer 2020 | **High, and it weakens the best novelty support.** The strongest line in `PRIOR_ART.md` is "16 citers of Daydreaming, none applied it to a transformer". Ramsauer shows attention *is* a modern Hopfield update — so that bridge is one substitution, not a programme | Own it: "the bridge is short and has not been walked" is a more modest claim |
| **Predictive coding / DEQ** — Whittington & Bogacz, Bai et al. | **High.** Canonical PC is: iterate activations to equilibrium on a fixed network, then apply a local pre/post-activity weight update. That is this protocol with an energy function attached | Add a section |
| **Hebbian at scale on pretrained nets** — SoftHebb (Journé 2023), Lagani/Amato | **May kill the eta claim.** These apply hand-written Hebbian/Oja rules to deep and pretrained nets **and publish step sizes** | Check directly — if one exists, the "no published step size" section is wrong |

**Verdict.** The exact four-way conjunction is probably unoccupied, but the way most
arbitrary four-way conjunctions are. Remove any single ingredient and you land somewhere
established: remove "no loss" → predictive coding, TTT; remove "transformer" → Daydreaming
Hopfield (and Ramsauer makes the substitution nearly free); remove "closed loop" → SoftHebb;
remove "hand-written rule" → Chaudhary, Miconi, Najarro & Risi. **"No loss" is the only
load-bearing axis, and per F4 it is softer than claimed.**

**Also flag for human verification:** the Chaudhary quote (arXiv:2510.21908 §4.8) is
agent-fetched and no human has opened the PDF. It is suspicious in shape — one paragraph
that reproduces all three previously-unverified figures verbatim, reassigns every one to the
rule family the project does *not* use, and hands the project's own family a benign outcome
that happens to agree with the other cited work. It also recommends "a practical regime
around 4 layers" while describing experiments at only 2 and 8. That is the shape of a
confabulated reconciliation. **Fortunately nothing depends on it**: the project has run at
12 layers and observed both branches directly — `oja` saturated, `hebb` did not diverge.
Demote the citation to corroboration and mark the quote `UNVERIFIED — agent-fetched`.

---

### F11 — A latent high-severity defect sits directly in the path of the next planned experiment

A full read of `plasticity.py`, `offline_control.py`, `multi_site.py`, `controls.py` and
`atr_bridge.py` against the documented claims found the whole-matrix path **defended to the
last bit** and the per-head path having inherited the scaffolding without inheriting the
tests. Verified correct: all four rules including the critical `anti_hebb` sign
(`plasticity.py:1120` computes `−H − D`, not `−(H − D)`, and the brake is read against the
*live* effective weight so it genuinely contracts); `random` norm-matched to the full Oja
update; the ceiling as a true Euclidean projection onto the Frobenius ball; the 17-axis
verifier as *enforced* (it raises `ArmsMismatchError`) with each axis broken in turn by a
test; the per-head row-slice identity bit-exactly; and `MultiSitePlasticity`'s
no-crosstalk guarantee, which holds structurally because each sub-instance reads its own
stored `W0 + delta` rather than the live matrix.

**The defect.** `offline_control.py:777` fetches the bias from the *whole* projection, and
`:791` then computes `_recompute_y(x_head, W_head, b)` — one head's contribution, not the
shared full output that the entire per-head design requires `y` to be. Shapes are
conformable, so nothing raises. Measured relative error on `y`: **0.838**. End to end:

| site | recomputed `diff_over_drift` | recomputed `cos_delta` | severed-path floor |
|---|---|---|---|
| `attn.c_proj` (whole matrix) | 1.03e-04 | 1.000000 | **0.000e+00** (bit-identical, as documented) |
| `attn.c_proj.head.2` | **6.02e-01** | **0.802** | **3.87e-04** — four orders above the 1e-8 detection limit |

`verify_arms_matched` passes **17/17 in both cases**, because `y_source` is deliberately not
an axis, so the verifier structurally cannot see this. The TransformerLens path is worse:
`_site_bias` looks for `b_out`/`bias` while TL's `Attention` spells it `b_O`, so the bias is
dropped entirely.

**Why it matters now.** `experiments/exp001_hebb.py` documents `--site blocks.11.attn.head.7`
as supported — **the head the parent project found carrying the period-2 cycle**, i.e. the
single most scientifically interesting site in the repo — and calls `run_matched_arms(...,
also_recomputed_y=True)`, the arm the file itself says "is the path the claim is made from".
Running that command produces a large, arms-matched, verifier-blessed, severed-control-passing
"feedback effect" that is **entirely an artifact of reconstructing `y` from one head**.
No test covers `offline_control` at a head site.

**No published result is affected — verified.** Every `site` field in every committed
artifact is `blocks.6.mlp`, and no head-site string appears anywhere in `experiments/output_*`.
This is a latent defect, not a corrupted result. But it is directly in front of T3.1 and
T3.2, so it must be fixed before the site sweep, not after it.

*Fix shape:* refuse `y_source="recomputed"` when the adapter is a head site (loud, one line),
or record the other heads' contribution as a residual and add it back.

**Six lower-severity defects worth tracking:**

1. **`clipped` is a latching boolean the docs call a rate.** It never clears between applies,
   so a naturally-computed clip rate measures "fraction of the run after the first clip".
   `step_size_map.py` works around it by writing to semi-private state from outside;
   `exp001_hebb.py` records only the boolean and annotates the limitation. `ORIENTATION.md`
   ("the clipping rate is recorded on every run") over-claims what the library provides.
2. **Sample accounting is asymmetric between arms.** The closed arm counts *accepted* samples;
   the offline arm appends unconditionally. One non-finite activation makes `n_samples` differ
   by one and raises `ArmsMismatchError` — the operator then hunts a replay bug that is really
   a NaN. Fails safe, for the wrong reason.
3. **`revert()` does not reset the RNG**, though its docstring promises "a clean slate".
   An instance reused across arms gives a C2 random arm not reproducible from its logged seed.
   Nothing in the repo currently trips it — `controls.py` constructs fresh instances.
4. **`last_update_norm` is the pre-clip step norm** (4.09e+04 against a `delta_norm` of
   2.68e-01 on a clipping apply), and retains a stale value when `apply()` returns early on a
   non-finite step.
5. **`_as_projector` checks idempotency but not symmetry**, so an oblique projector is accepted
   and would shear every update while `report()` stayed plausible — the exact failure its own
   docstring says it exists to prevent. Only bites hand-built projectors.
6. **The guard on the guard has a hole.** `PRIOR_ART.md`'s "must match" table has a
   period-detection row that is not in `MATCHED_AXES`; the test whose docstring says "every row
   has to be an axis here" asserts a hardcoded 8-name subset that silently omits exactly that
   row. Also: `centring` is a constant string that can only ever pass, and `rng_state_sha256`
   is digested before any draw so it cannot detect asymmetric randomness consumption.

**Two documentation contradictions in the control definitions.** `ORIENTATION.md:145` defines
C3 as "the plasticity layer must not perturb the loop except through the weight"; `controls.py`
and `README.md` define C3 as the Hebb-diverges/Oja-doesn't demo. `ORIENTATION.md:143` defines
C1 as a weight-equality check; `controls.py` compares *trajectories*. Two of the four gate
controls have two definitions each.

---

## 4. The claim register

F1, F2 and F3 share one cause: **nothing in this repository maps a claim to its evidence and
its status.** Documents record measurements well. Nothing records which claims are live, what
supports each, and which have been superseded — so a corrected conclusion in one file leaves
a stale conclusion in three others, a measurement can be made and forgotten (F3), and nobody
can see the inconsistency without reading everything.

`CLAIMS.md` is included in this branch as that artifact. One row per claim: the claim, its
evidence path, its status (`supported` / `provisional` / `retired` / `not-established`), and
the caveat that must travel with it. Rules:

- A claim enters the register **before** it enters any prose document.
- Every write-up cites row IDs rather than restating claims.
- Changing a row's status is a PR in its own right, with the evidence.
- `HANDOVER.md` §3 becomes a pointer to the register, not a parallel copy.

This is the mechanism that would have caught all three drifts, and it costs one file.

---

## 5. Instructions

### Tier 0 — Free. Do these before anything else; no new compute.

**T0.1 — Recognise that issue #31 is answered.** Close it against `BASELINE.md`, record the
answer (**compression, one-dimensional, position-uniform across all basins, nearest-basin
ratio 1.16**), and propagate it into `HANDOVER.md` §5.4 and `RESONANCE_NOTE.md`. (F3)

**T0.2 — Promote the matched-σ₁ control.** The `hebb`/`oja`/`anti_hebb` triple at matched
Frobenius *and* operator norm is already in `step_size_map.jsonl` and is a strictly better
C2 than the isotropic comparison. Write it up; retire the isotropic arm as decisive. (F4)

**T0.3 — Reset the bifurcation claim.** Retitle `BASIN_BIFURCATION.md`, set the `comrade`
result at ladder step 2 pending T1.1, add the `prolet`-basin-scatter comparison, and add one
sentence noting the offline arm flips the same basin. Correct `HANDOVER.md` §5.3. (F2)

**T0.4 — Fix the errors of fact** in F6 and F7. Two matter beyond hygiene: the "only cell"
error is suppressing an independent replication at `hebb`@1.18e-04, and the Chaudhary
contradiction has two live documents predicting opposite failure modes.

**T0.5 — Adopt margin discipline.** Every basin label gets its top1−top2 margin beside it,
and any flip below a threshold fixed *now, in advance* is reported `readout-ambiguous`.
69 of 125 baseline prompts sit below 0.5. Apply retroactively.

**T0.6 — Commit the prior-art search artifacts.** Query lists, dates, endpoints, returned
IDs. Converts eleven absence claims from assertion to record. (F10)

### Tier 1 — Cheap decisive experiments (CPU-minutes to hours)

**T1.1 — The coexistence test.** Under `W0 + ΔW`, seed the original frozen `prolet` state,
iterate 120 steps. Settles step 2 vs step 4. **~1 CPU-minute.** (F2)

**T1.2 — The α hysteresis sweep** (issue #26 already specifies it). Sweep α up through 1.5
and back down, each α seeded from the *previous* α's settled state. Retracing → smooth
deformation; a hysteresis loop → a real transition. Independent check on T1.1.

**T1.3 — Refine α\* around the real transition,** the interval (1.25, 1.50) where lag-1
collapses — not (0.50, 0.75) where only the argmax moves.

**T1.4 — The rank-matched random control.** A rank-1 random direction at matched σ₁, not
isotropic noise at matched ‖·‖_F. If it also flips the basin, the direction-specificity
conclusion collapses. **This can falsify the project's central result and it is cheap.** (F4)

**T1.5 — Fix `_recompute_y` at head sites, and add a head-site test to
`test_offline_control.py`.** **Blocks T3.1 and T3.2.** No published result is affected, but
the defect is invisible to all 17 axes and to the severed-path control, and it sits on the
path to `blocks.11.attn.head.7` — the site the parent project makes most interesting. Fix
before the site sweep, not after. (F11)

### Tier 2 — The experiments that make this a result

**T2.1 — The coupling-versus-drift curve.** *Highest value in the project.* Sweep eta × step
count; plot `diff_over_drift` and the perpendicular/parallel split against total drift,
routed and severed at every cell. Does the feedback-attributable component grow with coupling
strength, and is there a regime where feedback changes the *outcome* rather than only the
direction? Turns F1's one number into a phenomenon — or into a clean negative result worth
publishing as one.

**T2.2 — Run EXP-002 (issue #24), the pre-registered primary experiment.** Still unrun.
Follow its own sequencing note: *"First job is just steps 1 and 3: drive to collapse and
stabilise it."* The `TermSpec` machinery for the Hebbian/anti-Hebbian balance landed in
PR #37 and has never been used. Mind `HANDOVER.md` §4's trap: the brake is ~110× the
reinforcement term, so a sign read off the total update misreads as "eroding" in both
subspaces — difference against a brake-only arm.

**T2.3 — Get n.** Run the 125-prompt library at both clean `hebb` cells. Converts "3 prompts
flipped" into a distribution. Largest available credibility gain. Budget ~6 CPU-hours.

**T2.4 — Settle Oja.** Test F5's hypothesis directly: measure the brake/reinforcement ratio
across sites and depths, and report Oja's inertness as a finding with a mechanism.

### Tier 3 — Scale (only after Tier 2)

**T3.1 — Site sweep.** Re-anchor eta per site; it does not transfer. **Blocked on T1.5.**
**T3.2 — Multi-site.** Tooling landed in PR #35, never used. Start with two sites — the repo's
own prior art reports per-stage Hebbian modules destabilising where a single module did not.
**Blocked on T1.5.**
**T3.3 — Cadence, drive β, leak α.** `ISSUE_normalisation_homeostasis.md` E1–E4; E1 is cheap
and either outcome strengthens the repo.

### Governance — this week

| | |
|---|---|
| **G1** | Adopt `CLAIMS.md` as the source of truth for what the project asserts |
| **G2** | Reconcile the summary layer with the evidence layer — every row in F7 |
| **G3** | Make peer review a **merge gate** for PRs touching claim documents or `experiments/`. `PEER_BOARD_SETUP.md` documents the upgrade path; PR #22 met its stated precondition; F9 is what waiting cost |
| **G4** | Retire the "EXP-001" label collision; give the `Divine`-cycle question its own number |
| **G5** | Pin `transformer_lens`; open an issue for the v3 migration with a baseline re-run as its acceptance test |
| **G6** | Persist ΔW and the settled states — un-ignore `experiments/**/states/`, or commit compressed. F3's extension and issue #32 §5 both need them |

---

## 6. The strategic reframe

The project has been trying to prove a claim about **coupling**. It has now measured it
(F1), and the honest size is: real, well above a zero null, direction-dominated, and
*behaviourally inert at the operating point tested*. That should be stated — but it will not
carry a project until T2.1 shows the effect grows.

Meanwhile the project has proved something else, cleanly, under better control than it
realises, and is not claiming it:

> **A frozen transformer's iterated-dynamics attractor landscape is editable by a rank-1
> weight perturbation of ~1% derived from the model's own activation statistics — and the
> edit is direction-specific, not magnitude-specific.** Two other near-rank-1 updates at
> matched Frobenius *and* operator norm do not reproduce it; one of them at 2.6× the operator
> norm. In one case the edit changes the *dynamical class* of the trajectory, turning a fixed
> point into a period-2 cycle.

Every clause is measured, controlled and reproducible from committed artifacts. It is a claim
about the **editability of iterated dynamics**, which:

- survives the offline-arm result intact, because it never depended on feedback;
- survives the F2 correction, because it needs no created attractor — a boundary move is
  enough, and issue #25 called that "the first real result";
- makes the coupling number a **refinement** rather than the load-bearing claim: *and 12% of
  that edit is attributable to feedback*;
- and places the work next to model editing, activation steering and test-time adaptation,
  where there is an audience and a comparison class.

**One caution, from F10.** That last point cuts both ways: ROME already makes rank-1 edits to
a mid-stack GPT-2 MLP down-projection. So the novel element is **not** "a rank-1 edit there
changes behaviour" — it is that the edit is derived *with no target*, from the model's own
activation statistics, and is read out on the *iterated map's attractor structure* rather
than on next-token output. The iterated-dynamics readout is the instrument nobody else has.
Say that precisely, or a reviewer will say ROME first.

The framing that survives everything in this review is the one `RESONANCE_NOTE.md` already
reached and then filed away: **a characterisation study.** What does an unconstrained local
rule, given no objective, *do* to a pretrained model's iterated dynamics? That is the null
baseline every loss-driven method above lacks, it is genuinely unoccupied, and it does not
require the loop to be useful for anything.

---

## 7. What this review did not check

- **Experiments were not re-run.** Every number was verified against the committed JSONL
  artifacts rather than regenerated. Three independent audits did this; no fabricated figure
  was found in any document.
- **No external citation was verified against its source.** `PRIOR_ART.md`'s
  verification-status column is the record and issue #28's gaps remain open. See F10 on
  Chaudhary — that one needs a human to open the PDF.
- **The suite was confirmed to collect 295 tests but not run to completion.** CI on `main` is
  green as of 2026-07-31.
- **The code audit ran against a Conv1D-shaped stand-in** for the measured probes, because no
  GPT-2 checkpoint was cached in the review environment. The structural findings in F11 are
  read from source and hold regardless; the specific measured magnitudes (relative error
  0.838, `diff_over_drift` 6.02e-01) should be reproduced on real weights before the fix is
  called complete.
- **Not attempted:** any judgement on whether the parent ATR project's own findings are sound.
  This review takes the bridge's bit-exactness as given, since CI enforces it.

---

*Prepared as a repository leadership review. Every number quoted is reproducible from
`experiments/output_*`. Where this review and the code disagree, the code is right — check
`git log` since `ea4f0c1`.*
