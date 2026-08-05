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
| **Over-claimed** | The bifurcation. `BASIN_BIFURCATION.md` reports a created attractor; the α-sweep shows the observed settled branch moving smoothly while an argmax relabels, and D1 cannot discriminate step 4 from step 2. **F2** |
| **Unknown to itself** | Issue #31 — "the cheapest measurement in the project", listed as unrun in three places — was run, and is committed in `BASELINE.md`. Its answer changes how every basin result reads. **F3** |

F3 is the one that explains the other two. **In the audited state, no artifact in this
repository mapped a claim to its evidence**, so a measurement could be made and lost, and a
correction could land in one file while three others kept the superseded version. `CLAIMS.md`,
added by this review, now provides that mapping; §4 sets out the rules it runs under.

Separately, a full code audit found the whole-matrix path defended to the last bit and **one
high-severity latent defect on the per-head path** — invisible to all 17 matched axes, though
the severed-path control does flag it — sitting directly in front of the next planned
experiment. No published result is affected; every committed artifact is `blocks.6.mlp`. **F11**

> *Update, 2026-08-05 — two clauses in that paragraph are stale.* **The defect is fixed.**
> T1.5 rebuilt the head-site `recomputed`-y path as an additive reconstruction of the shared
> full output (`record.y + x @ delta`, gated on `site.shared_post_activity`); the head-site
> severed floor fell from O(1e-2) to float32 noise **2.960e-08**, `bit_identical` False.
> C-45 is `retired`, C-57 is `supported`, and F11 no longer sits in front of the site sweep —
> see the update under F11 itself. **And "every committed artifact is `blocks.6.mlp`" is
> `retired` with C-40.** All twelve MLP down-projections have since carried plasticity
> (EXP-002, EXP-003 Stage 1), and cadence 2, 4 and 12 have run. The narrower statement that
> holds, and the one this paragraph's conclusion actually needs, is **C-64**: no committed
> experiment applies plasticity at a **head** site — 0 of 12 attention output projections and
> 0 of 144 head stripes — so no published result was ever exposed to the defect.

**This review has itself been adversarially attacked and corrected** — four of its load-bearing
claims were refuted in part, and the withdrawals are marked in place. None of the three findings
above fell. See §7.

Both F1 and F2 entered through pull requests that received **no peer review**, in a project
whose review mechanism had previously caught two defects that changed conclusions (**F9**).

The strategic recommendation (§6) is that the project stop leading with *coupling* and
start leading with *editability of iterated dynamics* — a claim it has already proved,
under better control than it realises (**F4**), and is not making.

---

## 2. What the project actually has

Stated first, because the rest is critical and the asset is real.

**A ceiling-silent, loop-displacement-matched control the repo has never cited.** See F4.
`anti_hebb` at **94%** of `hebb`'s loop-state perturbation, opposite sign on the same axis,
does **not** move the basin — while `hebb` at a quarter of its own perturbation, *same* sign,
also does not. **Both magnitude and sign are required.**

> *Update, 2026-08-05: the last sentence is `retired`, and this asset is smaller than the
> paragraph claims.* **C-22 fell to T1.4**, the test this review itself named as deciding.
> Of the six ARBITRARY rank-1 random directions matched to `hebb`'s loop displacement within
> ±2%, **4 flip the basin** — with no particular sign on the `hebb`/`oja` axis, so the sign
> clause is refuted. The two observations above remain true *as measurements*; what fell is
> the inference that the sign is **required**. The **magnitude** clause survives: Arm A, matched
> on σ₁ = 1.8135 rather than on displacement, gives **0 of 10** flips at displacements 3–5
> orders below `hebb`'s. What survives overall is **C-55**, and it is narrower: no random
> direction reaches `comrade` (**0 of 10 seeds, 0 of all 74 probe evaluations**), and matching
> the displacement costs **66×–171×** `hebb`'s relative weight change (‖ΔW‖_F/‖W0‖_F
> 0.740–1.927 against 0.011239). One prompt (`A01_physics`), one site (`blocks.6.mlp`).
> Evidence: `experiments/output_rank1_random/`.

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
in the routed configuration is attributable to feedback:

| prompt | `1 − cos` routed | severed |
|---|---|---|
| `A01_physics` | 7.06e-03 | **0.0 exactly** (`torch.equal` True) |
| `A02_medical` | 6.08e-03 | **0.0 exactly** |
| `A04_climate` | 6.17e-03 | **0.0 exactly** |

The severed floor is `0.0`, not a small number, so there is no ratio to quote — which is
stronger than any ratio would be. (An earlier draft of this review printed severed values of
~3e-13 and a ratio of ~1.8e+10. Those are float64 round-off in computing `cos(v, v)` on
bit-identical matrices, and the `1 − cos ≈ 1.5e-14` figure quoted alongside them is a *state*
float32 floor from `BASELINE.md`, imported into a *weight* table. Both were wrong to include.)

And it is predominantly a **direction** change. Convention-free: feedback **rotates** the
update by **6.81°** (cos 0.99294; 6.32° and 6.37° on the other two prompts) and **shortens**
it by **2.8%** (`‖ΔW_closed‖ / ‖ΔW_offline‖` = 0.97225).

> **Do not quote a perpendicular:parallel ratio.** It is a deterministic function of those two
> statistics, `r·sinθ / |r·cosθ − 1|`, so it measures nothing they do not. It takes the value
> **3.33** referenced to `ΔW_offline` and **5.73** referenced to `ΔW_closed`; it is
> ill-conditioned here (`d ln ratio / d ln r` = **−47**, with the parallel component's zero
> sitting 2.1% away in norm ratio); and it ranges **5.03–7.67** across the three prompts. At
> this cosine, perpendicular dominance is automatic for any norm ratio in **[0.900, 1.144]** —
> at *equal* norms the ratio would be 16.8, so the measured value is **less** direction-dominated
> than equal norms would give. Report the rotation and the shortening.
>
> *Correction to `EXP_001_RESULTS.md:133`.* The published figures 0.1153 / 0.0346 are
> arithmetically correct, but the prose labels them fractions of `‖ΔW_closed‖`;
> `exp001_hebb.py:1163-1173` normalises by `‖ΔW_offline‖`. **The label is wrong, not the
> numbers.** Note `diff_over_drift` uses that same reference (`offline_control.py:975`,
> `diff / max(drift)`, offline being the larger), so the 0.120 headline above is on the same
> footing; against `‖ΔW_closed‖` it would be 0.124. Fix the label — do not restate the ratio
> against the other reference, which would inflate it without strengthening the effect.

**What the summary layer tells a new reader.** `HANDOVER.md` §3.3 — the first file anyone
picks this project up from:

> "The two agree — `cos(ΔW_closed, ΔW_offline) = 0.99294` … if a coupling claim is ever
> made, that is the bar it has to clear."

And the comment on issue #32:

> "near 1 means feedback changed nothing but scale. That is what was measured."

Both are wrong the same way. Issue #26 pre-registered the rubric "near 1 → nothing but
scale" *before the severed control existed to calibrate what "near 1" means*. Once the null
is known to be **exactly 1** — bit-identical arms, `rel_fro_diff` exactly `0.0` — a cosine of
0.993 is not near 1: it is strictly positive against a null of exactly zero. (No ratio can be
quoted here, and an earlier draft wrongly quoted one: the denominator is zero.)
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
> is predominantly a change of direction: a **6.81° rotation** with a **2.8% shortening**.
> At this step size and horizon that steering does **not** change the behavioural outcome —
> both arms flip to the same basin. The result is conditional on `y_source="recomputed"`;
> in `recorded` mode the severed floor exceeds the routed value and the comparison reverses,
> which is why `offline_control.py` already declares that mode uninterpretable as a feedback
> test.

Both halves matter. The first is the coupling result the project has been looking for; the
second is the honest limit, and it makes the next experiment obvious (T1.1).

**Caveats that must travel with the number.** The severed arm runs a shallower loop (0→3)
at 2.6× the drift (2.9% vs 1.1%), so it is not a statistically matched control — the
argument for it as a null is *structural* (no feedback path ⇒ bit-exact agreement) and that
argument holds, but the mismatch belongs beside the number. Independent n is **3 prompts**,
not the 5 cells the tables imply (see F6).

---

### F2 — The bifurcation claim is not established, and its own α-sweep argues against it

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

Across α = 0 → 1.25 the **observed settled branch** remains a fixed point and moves smoothly:
lag-1 pinned at 1.0,
state norm advancing in near-equal increments, no discontinuity anywhere near α\* = 0.75.
What changes discretely at α\* is the **argmax of the readout** — and an argmax always
changes discretely, including under perfectly smooth motion. The document says this and
treats it as corroboration: *"The argmax (the basin) is discrete; the logit it is the argmax
**of** is not. Smooth logits, discrete attractor — the signature the result turns on."*
That is **consistent with a relabelled, continuously-moving fixed point** — ladder step 2 — though it does not exclude an unobserved coexisting attractor, which is what T1.2 tests.
The crossing is driven entirely by `prolet`'s logit *falling* (16.95 → 16.07, −0.88).
**`comrade`'s logit does not rise — it also falls** (16.29 → 16.27 at the α\* = 0.75 crossing,
−0.02; 16.25 by α = 1.00), monotonically at every α. The argmax changes because the incumbent is suppressed, not because a competitor
grows: a steering-vector signature, not a new attractor.

**(b) D1 does not discriminate what it claims to.** For any ΔW ≠ 0 the perturbed map's fixed
point is generically not a fixed point of the unperturbed map. D1 correctly falsifies "the
episode walked into a pre-existing `comrade` basin of W₀" (step 3) — but step 4 is then the
*residual by elimination* — and D1 returns that same verdict for **any** ΔW that displaces the
fixed point at all, a norm-matched random edit included. (Equal operator norm does not imply
equal trajectories, so a random arm would not reproduce D1's *numbers*; the point is that it
would reach the same *conclusion*, which is what makes D1 non-discriminating.) D1's own
trace argues for step 2: released under W₀ the state relaxes back to `prolet` smoothly and
monotonically, lag-1 above 0.99992 at every sampled iteration.

**(c) The decisive number, from the repo's own baseline.** `BASELINE.md` measures the
`prolet` basin's internal scatter. Set the `comrade` displacement against it:

| quantity | `1 − cos` |
|---|---|
| `comrade` state vs the `prolet` fixed point (D1, iter 200) | **4.39e-03** |
| mean pairwise spread **within** the `prolet` basin (55 prompts) | 2.77e-03 |
| **median** pair within the `prolet` basin (cos 0.999400) | 6.00e-04 |
| **worst** pair within the `prolet` basin (cos 0.966079) | **3.39e-02** |
| gap between the two *nearest genuine basins* (`Anarch`–`prolet`) | 2.87e-03 |

**No argument against step 4 can be built from this displacement's size, in either direction.**
It is *larger* than `prolet`'s mean pairwise spread (1.58×), larger than the nearest-basin
`Anarch`–`prolet` gap (1.53×), and 7.3× larger than the median within-`prolet` pair. It is
smaller than exactly one number — the single worst of 1485 within-`prolet` pairs, which sits at
56× the median — and one extreme order statistic is not a diameter.

Nor would a larger number settle it. Within-basin variation is one-dimensional (**C-05**,
participation ratio 1.29, 87% of variance in a single direction), so the basin is a *segment*,
and a displacement magnitude decides nothing about membership unless its alignment with that
segment is measured — which nothing here does. Against the segment's transverse RMS spread
(chord 0.0265) the `comrade` displacement (chord 0.0937) is 3.5× outside. And membership is
settled by the readout anyway: the `comrade` state's argmax *is* `comrade`. Note also that a
saddle-node bifurcation creates a new attractor at *zero* separation from the old one, so small
separation would not have been evidence against step 4 even if it had been found.

> **An earlier draft of this review got this backwards** and said the `comrade` state is "7.7×
> closer… comfortably inside `prolet`'s ordinary scatter", concluding that a created attractor
> "needs to clear that resolution and does not". That used the worst of 1485 pairs as the
> yardstick; against every other summary of the same distribution the displacement *clears* the
> resolution. The claim is withdrawn. It was the review's own most-flagged weak link, and it
> did not survive.

What the scatter numbers **do** license is narrower and worth keeping. `comrade` sits 1.5× the
`Anarch`–`prolet` gap from `prolet`, so **`comrade` and the pre-existing `Anarch`/`prolet`
distinction stand or fall together**; and at `1 − cos` ~ 3e-03 the basin label is a coarse
instrument — 69 of 125 baseline prompts sit below a 0.5 top1−top2 margin, and `A01_physics`
carries `comrade` at rank 4 in its *frozen* top-5. A basin flip at this scale is a weak signal
whichever way it is read. **The argument against step 4 rests on (a) and (b), not on a distance.**

> **Comparability, checked — it holds.** The two figures do come from different runs: the
> `comrade` number is a D1 relaxation trace (whole-tensor cosine, transformer_lens 3.6.0), the
> scatter numbers are phase-aware **position-mean** `(768,)` cosines over 55 prompts (3.5.1).
> All three licences were verified rather than assumed. **Metric:** position-uniformity (C-06)
> makes the two cosines coincide — across the 91 fixed-point baseline prompts the estimators
> differ by ≤ **1.01e-06**, and on the 34 `Divine` period-2 prompts their `1 − cos` values agree
> to a **median ratio of 0.9997**. **Phase:** a no-op on `prolet`, which is 55 fixed points and
> 0 period-2. **Version:** exactly zero here — baseline `A01_physics` at iteration 120 gives
> `‖state‖` 4782.77880859375 and margin 0.2303314208984375 under 3.5.1, and the α = 0 cell
> reproduces both **bit-for-bit** under 3.6.0.
>
> So the comparison is admissible. It is simply not *decisive*, for the reasons above — which is
> a different and more interesting failure than the one this caveat was written to guard against.
> **C-26 is held at `not-established` because T1.1 is unrun, not because of a measurement doubt.**

> *Update, 2026-08-05: T1.1 has run, and C-26 is no longer held — it is `retired`.* Seeded from
> the original frozen `prolet` state under `W0 + ΔW`, the loop does **not** stay: it moves to
> `comrade`, settling at iteration 12 and holding through 120, lag-1 ≥ 0.99990 at every step and
> the top1−top2 margin falling smoothly to 0.00025 at the crossing. One **displaced** attractor,
> not two coexisting — so the coexistence reading step 4 required fails, and what is measured is
> ladder **step 2**, exactly as (a) and (b) above argued. eta=0 gate bit-identical to the frozen
> loop. Verdict invariant to the renorm shell. The durable claim is **C-56**. Evidence:
> `experiments/output_t1_1/`. *Note the register has since attached a bound this review could not
> have seen:* **C-68** records that at α = 0.50 two committed artifacts reach two different
> settled words from two different initial states under identical weights, so the general "one
> continuously-moving fixed point, never coexistence" gloss is stronger than the evidence at
> that α — but it does not touch T1.1, which tested α = 1.00 directly.

**(d) In the audited state, the file never mentioned the offline arm.**
`grep -i "offline\|coupling\|feedback"` on `BASIN_BIFURCATION.md` at `ea4f0c1` returned **zero
hits**. But the offline arm — no feedback path at all — flips the same basin, and the commit
that produced EXP-001 is titled *"the basin flip is the rule, not the coupling."* The file
speaks throughout of what "the episode" did. Read alone — and it would be, being the newest
result — it supported the conclusion that the closed loop created an attractor, while the
repo's own control says the loop was not required. **This review adds that sentence**, so the
gap is now closed in the file itself; it is recorded here because the omission is what let the
reading spread.

**The decisive test was not run, and it is cheap.** Under `W0 + ΔW`, seed the *original
frozen* `prolet` state and iterate. Stays `prolet` → two attractors coexist → step 4 stands.
Moves to `comrade` → one displaced attractor → step 2. ~1 CPU-minute. Issue #26 also already
specifies the hysteresis test that would settle it independently — *"Up-then-down eta sweep…
that hysteresis is the cheapest available evidence of a real transition. Almost nobody does
it."* Specified, not run.

> *Update, 2026-08-05: both have run.* **T1.1** (the decisive test): the state moves to
> `comrade` — step 2, not step 4 (C-26 `retired`, C-56 `supported`; details in the update two
> paragraphs above). **T1.2** (the hysteresis test): **no hysteresis.** A continuation
> up-then-down α-sweep, step 0.10, 0 → 1.5 → 0, each α seeded from the previous α's settled
> state, **retraces exactly** — both directions give the identical basin map (`prolet` at
> α ≤ 0.4, `comrade` at α ≥ 0.5), with **alpha_up 0.5 and alpha_down 0.4, one grid step apart**,
> i.e. a single threshold near α ≈ 0.45 and no resolved loop. Same verdict under the fixed-W0
> shell; all settled states are fixed points (lag-1 = 1.0). This corroborates T1.1 rather than
> merely repeating it, since the two tests differ in initial condition. Limits: a sub-0.10 loop
> inside (0.4, 0.5) is not excluded at this grid, and n = 1 prompt, 1 site, 1 eta. C-52 is
> answered; evidence `experiments/output_t1_2/`.

**The one genuine qualitative transition is elsewhere and is under-reported.** Between
α = 1.25 and 1.50 lag-1 collapses 0.999998 → 0.734 and the state norm jumps 65.9 against a
running increment of ~15. A fixed point gives way to a period-2 cycle — a real change of
dynamical class. But `Divine` is a **pre-existing** attractor (34 of 125 baseline prompts),
so this is ladder **step 3**, a boundary move, and it sits at 1.5× the ΔW the episode
produced.

**Net effect on the ladder.** The project has a solid **step 3** — `A04_climate` crossing into
the pre-existing `Divine` cycle under the **full applied ΔW**, with a dynamical-class change no
readout artifact can explain. (Not to be confused with the α-sweep above, which is on
`A01_physics`: there `Divine` appears only at α = 1.50, past the 0→1.25 range the smoothness
argument covers, and the `comrade` transition sits at α\* = 0.75.) Issue #25 calls step 3 *"the measurable one, and the first real
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
| **ratio against the *nearest* basin pair** | **1.15** |
| within-basin effective dimensionality (participation ratio) | **1.02 – 1.29** across basins |
| position uniformity | **125/125 prompts, every basin** — not a `Divine` property |

**The answer is compression, not erasure** — the issue's "more interesting" outcome, the
one that makes the persistence work worth doing. Prompts in a basin land on nearby-but-
distinct states, and the within-basin variation is essentially **one-dimensional**
(participation ratio 1.02–1.29 against a maximum of n−1 = 15–54).

And the sting: **a ratio of 1.15 against the nearest basin pair means the two closest
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

**That comparison does not work either, for two reasons — and an earlier draft of this review
promoted it as decisive. It is not.**

**Two directions, not three.** `oja` = `H − D` and `anti_hebb` = `−H − D`
(`plasticity.py:1113-1128`), and at this site the brake dominates the reinforcement term ~110:1
(`HANDOVER.md`), so both collapse to ≈ `−D`. They are one arm sampled twice. Measured in the
loop's own state space, the cosine between their displacements from the `off` cell stays at
**0.9987–0.9999 across the six lowest shared etas** (it drops to 0.991 and 0.939 at the two
heavily-clipped ones), and both sit at cos **−0.95** to `hebb`'s. In the linear regime the three arms
occupy a **single axis**: `hebb` on one sign (`W` grows), `oja`/`anti_hebb` on the other (`W`
shrinks). `step_size_map.py` already encodes this — its `U_REF` calibration constant is
identical (14000.0) for both.

**And matched σ₁ is not matched effect.** The three cells perturb the loop state by
`1 − cos(off)` = **5.000e-03** (`hebb`), **1.197e-04** (`oja`), **2.121e-04** (`anti_hebb`) — a
24–42× gap. `oja` never reaches `hebb`'s perturbation anywhere in the sweep (max 2.724e-03
against the 5.000e-03 that flips). **That is verbatim the defect that disqualified the isotropic
arm**, reappearing in its proposed replacement.

**The control that does work is one row further down the same table.**

| mode | eta | σ₁ | ‖disp‖ | `1 − cos(off)` | cos → flip dir | clip | basin |
|---|---|---|---|---|---|---|---|
| **`hebb`** | **7.065e-05** | 1.8135 | **152.7** | **5.000e-03** | +1.000 | 0.0% | **`comrade`** ← flips |
| `hebb` | 3.925e-05 | 0.9002 | 75.5 | 1.229e-03 | +0.989 | 0.0% | `prolet` |
| **`anti_hebb`** | **2.944e-05** | 6.9517 | **146.8** | **4.713e-03** | −0.924 | **0.0%** | `prolet` |

`anti_hebb`@2.944e-05 is **ceiling-silent**, reaches **94%** of `hebb`'s loop-state perturbation
(`1−cos` 4.713e-03 against 5.000e-03; by displacement norm, 146.8 against 152.7, it is 96%),
points at cos −0.92 to `hebb`'s flip direction — and
**does not flip**. Conversely `hebb`@3.925e-05 points the *same* way at cos +0.99, reaches only
a quarter of the perturbation, and also does not flip.

**So the honest statement is: the flip needs both a sufficient magnitude and the right sign on
this axis. Neither alone.** That is a real, loop-displacement-matched, ceiling-silent control,
already in the repo and never cited — and it is a weaker claim than "direction-specific".

> *Update, 2026-08-05: the sign half of that statement is `retired` (C-22).* T1.4 flips the
> basin with arbitrary rank-1 directions carrying no particular sign on this axis. The magnitude
> half survives. Full accounting in the update three paragraphs below, under "Remaining gaps";
> the surviving claim is **C-55**. The control described here remains a real, correctly-matched
> control — it is the conclusion drawn from two samples of one axis that did not survive
> sampling the space.

**"Direction-specific, not magnitude-specific" must be withdrawn**, because F2's own α-sweep
refutes it two findings earlier: holding this exact ΔW's *direction* fixed and scaling it alone
produces three basins (`prolet` → `comrade` → `Divine`, α\* = 0.75). Within `hebb`, σ₁ 0.9002 →
`prolet` and σ₁ 1.8135 → `comrade`, same direction, both ceiling-silent. Magnitude clearly does
part of the work.

Remaining gaps: only two directions are sampled, and `hebb` grows `W` while `oja`/`anti_hebb`
shrink it, which the loop's L2 renormalisation may treat asymmetrically. **T1.4 — a rank-1
random direction at matched σ₁ *and* matched loop displacement — is still the deciding test,
and it can falsify what is left.**

> *Update, 2026-08-05: T1.4 ran, and it did falsify what was left. `C-22` is `retired`.* The
> two-directions gap named just above was the whole weakness, and sampling arbitrary directions
> closed it against this finding. Of the six of ten seeds matched to `hebb`'s loop displacement
> within ±2%, **4 flip the basin** (`Anarch` ×2, `bourgeois` ×2) — arbitrary rank-1 directions
> with **no particular sign on the `hebb`/`oja` axis**. So "the flip needs the right sign on this
> axis" is refuted; the paragraph three above stands as a pair of measurements but not as the
> inference drawn from them. **The magnitude clause survives**, and survives cleanly: Arm A,
> matched on σ₁ = 1.8135 instead of on displacement, gives **0 of 10** flips at displacements
> 4.0e+03–6.7e+05× smaller than `hebb`'s — which is this finding's own "matched σ₁ is not matched
> effect" point, confirmed by construction. **What replaces C-22 is C-55**, and it is narrower
> than either claim this section considered: an arbitrary rank-1 direction at `hebb`'s
> displacement usually moves the basin, **never to `hebb`'s destination** (`comrade` 0 of 10
> seeds, 0 of all 74 probe evaluations), and only at **66×–171×** `hebb`'s relative weight change
> (‖ΔW‖_F/‖W0‖_F 0.740–1.927 against 0.011239). Four of ten seeds could not be matched — two at
> the search's scale cap, two straddling a displacement discontinuity — so the flip-rate
> denominator is 6, not 10. C-07 travels with the destinations: 6 of the 8 flips land `Anarch`,
> whose gap to `prolet` (2.874e-03) is *below* the **pooled** mean within-basin spread (3.319e-03,
> 2337 pairs across all five basins). `prolet`'s own spread is **2.773e-03**, which that gap
> *exceeds*, so the population has to be named or the comparison reverses — the coarse-instrument
> reading holds on the pooled figure and not on `prolet`'s own scatter (`BASELINE.md`
> "Within-basin spread"). n = 1 prompt
> (`A01_physics`), 1 site, 120-step episode. Evidence: `experiments/output_rank1_random/`.

**The isotropic critique stands unchanged.** Theory for an iid Gaussian (3072, 768) gives
σ₁/‖·‖_F = 0.054127; measured across all 8 `random` cells the mean is 0.054203 — 0.14%
agreement. `random`'s largest σ₁ anywhere is 0.4469 against `hebb`'s flipping 1.8135. Retiring
the isotropic arm as the decisive control is correct.

**Related, and it explains everything above.** At this site the module computes `y = x@W + b_out`
— `x` is `blocks.6.mlp.hook_post`, `y` is `blocks.6.hook_mlp_out` (`plasticity.py:448`) — and the
rule accumulates `ΔW = E[x yᵀ]` (`plasticity.py:105-109`). The exact identity is therefore

> **ΔW = E[x xᵀ]·W + E[x]·b_outᵀ = C·W + x̄ b_outᵀ**

— power iteration on the site's input second-moment matrix, **plus a rank-1 bias term that is
not droppable**: `‖x̄ b_outᵀ‖_F / ‖ΔW‖_F = ‖b_out‖ / ‖ȳ‖ = 3.353 / 19.451 = **17%** at the routed
working point. Position-uniformity *maximises* it, because `E[x]` does not average down. In
augmented form it is still exactly power iteration: with `x̃ = [x; 1]`, `ΔW = E[x x̃ᵀ]·[W; b_outᵀ]`.

Since the attractor states are position-uniform (F3: 125/125) — and injection at
`hook_resid_pre` discards positional embeddings (`atr_bridge.py:37-40`), so uniformity survives
attention and the position-wise MLP exactly — C is effectively rank-1 and

> **ΔW ≈ η·N· x̄ ȳᵀ: a rank-1 weight edit whose input-side factor is the site's dominant
> activation mode, and whose output-side factor is the site's mean output ȳ = Wᵀx̄ + b_out —
> not Wᵀx̄.**
>
> **An earlier draft of this review stated the identity as `ΔW = C·W` "exactly", dropping the
> bias.** The repo's own artifact refutes that form by three to four orders of magnitude:
> `u_right_sign_ref_cos` gives **cos(v₁, ȳ) = 0.999999** in the severed cells (where ΔW is
> exactly rank-1), whereas a bias-free `v₁ ∝ Wᵀx̄` would give **0.9975**. The rank-1 evidence
> originally cited as confirmation (95.8% of `‖ΔW‖²_F` in component 1, stable rank 1.04) does
> **not** discriminate — rank-1-ness follows from position-uniformity alone and holds for
> `C·W + x̄b_outᵀ` equally well.

The rank-1 part is measured: 95.8% of `‖ΔW‖²_F` in component 1 at `routed:A01_physics`,
**100.0%** in the severed cells, stable rank 1.04–1.06 — though only 81–84% in the two episode
cells. **One consequence for the decode:** ~17% of the norm of the direction EXP-001 pushes
through the unembedding is `b_out`, a prompt-independent constant of the layer. That is worth
knowing before reading anything into the token list.
Three consequences the write-up must confront:

1. **"No loss" is weaker than stated.** Plain Hebb is exactly gradient ascent on ½E‖y‖², and
   the bias does not disturb this — `∂/∂W ½E‖xW+b‖² = E[x yᵀ]`, term for term. There *is* an
   implicit objective. The defensible phrasing is "no **externally specified** objective" — a
   much smaller gap from unsupervised test-time adaptation than "no loss at all".
   *Narrower than an earlier draft said:* Oja's **single-unit** rule is the same under a
   unit-norm constraint, but the 768-output subspace rule this repo runs is a constrained
   gradient flow only when `WᵀW = I`, which here it is not (`WᵀW` diagonal mean 35.4, stable
   rank 31.0).
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
`hebb` number.

> *Update, 2026-08-05: "at every step size tested" over-reaches, and the repository's own
> standing prohibition is what withdraws it.* `oja` has **eight** cells in the step-size map.
> **Five** are ceiling-silent at 0.0% clip; the other **three fired the ceiling — at 9.2%, 65.8%
> and 100% clip** — and a ceiling-fired cell measures the ceiling, not the rule, so it may not be
> quoted as evidence here. The licensed statement is the one **C-13** carries: `oja` is inert
> **across the five clean cells**, moving the basin at no step size tested **up to 2.9% drift
> with the ceiling silent**. The three clipped cells also stay `prolet`, and that remains a
> diagnostic note rather than evidence — the same treatment the table row above applies to
> `anti_hebb`'s "only at 100% clip". Nothing in F5's argument depends on the difference: five
> clean cells spanning three decades of eta with the basin never moving carries the finding on
> its own.

The repository is nonetheless organised around Oja: the class is
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
brake, and a rule that is almost all brake would track the dominant activation direction and
go nowhere else. **That is a hypothesis, not an explanation** — the ratio has been measured at
`blocks.6.mlp` and never tested as a cause. `CLAIMS.md` C-14 holds it at `not-established`
and T2.4 is the test. If it survives, it is a publishable observation about Oja inside a
pretrained transformer, and it would agree with the prior-art expectation of saturation.

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

> *Update, 2026-08-05: the site and cadence figures are stale — C-40, the claim that every
> number in the repository is one site and cadence 1, is `retired`.* The current count is
> **C-64**: **twelve of twelve MLP down-projections** have carried plasticity (all of
> `blocks.0.mlp`…`blocks.11.mlp`, in `output_exp002/exp002.jsonl`, `exp002_uncapped.jsonl` and
> `output_exp003/stage1.jsonl`), while **zero of twelve** attention output projections and
> **zero of 144** head stripes have; and **cadence 1, 2, 4 and 12** have all run (1 in EXP-001,
> 2 and 4 in `output_t2_1/t2_1_coupling.jsonl`, 4 and 12 in `output_exp003/stage2.jsonl`). So
> read this paragraph as **12 of 168 candidate sites, one model, one site family** — and the
> scope point it was making survives intact, because that is still the project's weakest
> dimension. **Two boundaries the arithmetic must respect**: the multi-site runs lifted the drift
> ceiling (C-60) and lost the exact-zero severed floor (C-63), so wider coverage cannot be
> assembled by adding them to the single-site series; and cadence > 1 is tested at one site only,
> except for `stage2.jsonl`, whose own registered drift guard fired.

**Errors of fact to correct:**

| # | Location | Says | Actually |
|---|---|---|---|
| E1 | `EXP_001_RESULTS.md` intro, `HANDOVER.md:96` | `hebb`@7.07e-05 is "the **only** cell in the whole sweep that moves the loop inside a clean band" | **Two** clean flipping cells: `hebb`@7.07e-05 (1.12% drift) *and* `hebb`@1.18e-04 (2.20% drift), both 0.0% clip, both `comrade`. Good news — robustness of the flip across a 1.7× change in eta — and it is being suppressed by an error. Not an independent replication: both cells share prompt, seed, site and cadence |
| E2 | `EXP_001_RESULTS.md` §4 | ΔW effective rank "1.8–3.8 for `oja`" | `oja`'s range is **1.0–2.9**. The 3.8 is `anti_hebb`@9.81e-05, a cell at **60.8% clip** — which by the repo's own rule should not be cited at all |
| E3 | `EXP_001_RESULTS.md:133` | perp/parallel decomposed against `ΔW_closed` | Decomposed against `ΔW_offline`. **The label is wrong, not the numbers** — do not restate the ratio against the other reference (see F1); report the 6.81° rotation and the 2.8% shortening instead |

**Two more citation-hygiene problems of the same class:** `STEP_SIZE_MAP.md` §5's evidence
that effective rank *rises* is taken from `anti_hebb`@9.81e-05 at 60.8% clip; and
`HANDOVER.md:90` cites the `anarchism` flip without noting it occurs at **100% clip**. The
null conclusions stand from the clip-free cells; the specific numbers should not be quoted.

> *Update, 2026-08-05: both were adjudicated on 2026-08-01 and the register's standing-prohibitions
> entry now records the outcome — they did not resolve the same way.*
>
> - **`STEP_SIZE_MAP.md` §5 *was* in breach**, exactly as flagged: it bounded the effect with
>   cells at 99.2% and 60.8% clip. It has been **rewritten onto the 23 ceiling-silent cells** —
>   effective rank never below **642.4** against a frozen **642.6**, rising to **646.7** under
>   `anti_hebb` at **0.0% clip**. The conclusion is unchanged and now rests on clean cells. The
>   two withdrawn figures (640.5 and 647.3) are named in place and kept in the full table as
>   diagnostics.
> - **`HANDOVER.md`'s `anarchism` citation is *not* in breach**, and this review's flag on it was
>   wrong. It appears in the step-size summary table with **"only at 100% clip" stated inline**,
>   in a column asking whether the rule moves the basin inside the ceiling-silent band, answered
>   **"no"**. Quoting a ceiling-fired cell as a non-finding with its clip rate attached is
>   precisely what the standing prohibition asks for, not a violation of it. The distinction that
>   matters is whether the cell is being used to *support* a conclusion or to *decline* one.

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
| `README.md` (×2) and `tests/conftest.py` `_unavailable()` | "227 tests" | **295** collected (verified) |
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

> *Update, 2026-08-05: the `PRIOR_ART.md:28` row was re-audited entry by entry and it **still
> stands**, unfixed — all three of its counts reproduce.* Over the seventeen works the file
> cites, **four carry no verification status**: Nellessen & Jan's *Hebbian Natural Abstractions*
> (LessWrong, 2022), which is the load-bearing hit of the whole forum search; **arXiv:2605.04200**,
> the one forward citation of Cazalets & Dambre, given as a bare identifier; an **unnamed bioRxiv
> spiking-network paper**, the one forward citation of Gong et al.; and **Zenke & Gerstner,
> Phil. Trans. R. Soc. B 372:20160259**, named only to record a withdrawn misattribution. **Two
> carry no author list** — *Training-Free Looped Transformers* (arXiv:2605.23872) and *Where to
> Bind Matters* (arXiv:2605.02920), both marked **Verified** against identifier and abstract
> without saying who wrote them. **One carries no identifier of any kind**: the bioRxiv paper,
> dismissed as "nothing near us" on the strength of a one-clause description. *One addition this
> row did not make:* **two further entries are located by venue and year only**, with no arXiv ID,
> DOI, volume or pages — Irie, Csordás & Schmidhuber (ICML 2022) and Lazar, Pipa & Triesch
> (Frontiers in Computational Neuroscience, 2009). `PRIOR_ART.md` now states this position in all
> three places that carried the blanket claim, and the verdict stays `provisional` (C-42).

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

**The Chaudhary citation: checked independently, and `PRIOR_ART.md` was already right.**

An earlier draft of this review flagged the quote (arXiv:2510.21908 §4.8) as unverified and
asked for a human to open the PDF. **That flag was wrong, and it contradicted this review's own
drift table three sections above**, which correctly records `PRIOR_ART.md` as having verified
the figures and identified the gradient-plastic/Hebbian split. The paper has now been fetched
independently for this review. All three questions are resolved:

1. **Does §4.8 contain the passage verbatim? Yes.** §4.8 exists and is titled *"Task-Dependent
   Behaviour and Depth Stress Test"*. The passage `PRIOR_ART.md` quotes matches the source
   word for word, including *"Gradient-plastic Transformers diverge after ∼3000 steps (plastic
   norms >10²; recall below baseline), with the deepest layers showing the largest drift"* and
   *"A practical regime therefore lies around 4 layers: deeper gradient-plastic stacks require
   additional regularization… to prevent instability."*
2. **Does it attribute divergence to the gradient-plastic variant? Yes** — explicitly, and the
   same sentence reports *"Hebbian plasticity remains stable but saturates in performance
   (recall 0.729 ± 0.015)."* `PRIOR_ART.md`'s reading is correct.
3. **Is "around 4 layers" an interpolation? Yes.** The paper's standard experimental models are
   **2 layers**; the depth stress test extends to **8**. No 4-layer result is reported —
   4 is interpolated between the two measured endpoints. A normal thing for a paper to do,
   recorded because it is checkable and because it means "4 layers" is not a measured boundary.

**What this changes for the project — and it is favourable.** The 8-layer divergence belongs to
a rule family this repo does not use. The family it *does* use is the one the paper reports as
**stable at 8 layers**, and this repo's own 12-layer runs agree: `oja` saturated, `hebb` did not
diverge. GPT-2 small is not "past the point where anyone has reported stability" for a Hebbian
rule; it is simply deeper than anyone has tested with any rule. Keep the citation as
corroboration, and add the one thing `PRIOR_ART.md` does not yet note: **nothing in this
literature tests 12 layers**, so the paper cannot set expectations either way for this substrate.

> **This is the third instance of the pattern this review documents, and the first that is the
> review's own.** Issue #31 was recorded as unrun after being run; the coupling result was
> recorded as null after being positive; and this review flagged as unverified a passage the
> repo had already verified and correctly re-read. The failure is the same every time: a
> conclusion reached in one file and not propagated to the file where it is quoted.

---

### F11 — A latent high-severity defect sits directly in the path of the next planned experiment

> *Update, 2026-08-05 — read the whole of F11 in the past tense: **the defect was fixed in
> T1.5** and this finding is no longer a blocker.* The fix is an **additive reconstruction of the
> shared full output**: `replay_offline` branches on `site.shared_post_activity` and rebuilds the
> drifted full output as `record.y + x @ delta`, holding the other heads and `b_O` frozen inside
> the recorded `y`, instead of recomputing one head's contribution from scratch. On real GPT-2
> small at `blocks.11.attn.head.7` the unit reconstruction's relative error to the full output is
> **0.0** at eta = 0, and the head-site `recomputed` severed floor falls from **4.9975e-02** to
> **2.960e-08** with **`bit_identical` False** — float32 noise, **not exactly 0.0**, because a
> fused twelve-head einsum cannot be rebuilt bit-for-bit from one head's additive remainder. Two
> regression tests now cover the head-site path
> (`test_recomputed_y_at_a_head_site_is_the_shared_full_output`,
> `test_a_head_site_the_loop_does_not_route_through`), against a bound of 1e-6. C-45 is
> `retired`, **C-57** is `supported`, the whole-matrix `blocks.6.mlp` path is untouched and still
> bit-exact at 0.0, and **T3.1 / T3.2 are no longer blocked on this**. The structural finding
> below was correct; its magnitudes were not. Evidence: `experiments/output_t1_5/T1_5_RESULTS.md`.

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

All measured magnitudes in this finding (0.838, and this table) come from a **Conv1D-shaped
stand-in, not real GPT-2 weights** — provisional until T1.5 reproduces them (§7, C-45). The
structural defect is read from source and does not depend on them.

> *Update, 2026-08-05: T1.5 reproduced them on real GPT-2 small, and **the stand-in figures did
> not transfer**. Both numbers above are withdrawn.* The relative error on `y` is **≈ 1.00**
> (per-sample 0.9988–1.0021), not 0.838 — reconstructing from one head does not approximate the
> shared full output, it misses it almost entirely. And the head-site `recomputed` severed floor
> is **4.9975e-02**, not 3.87e-04: **two orders larger** than the stand-in predicted, because it
> saturates near the 0.05 `max_delta_frac` ceiling. The stand-in therefore *understated* the
> severity by two orders, which cuts in the finding's favour and against its own caution. Its
> caveat was right to be there. Post-fix the same floor is **2.960e-08** with `bit_identical`
> **False**. *One structural correction the table's own framing needs:* the documented
> **0.000e+00** is a **whole-matrix** property and was never available at a head site — a fused
> twelve-head einsum cannot be rebuilt bit-for-bit from one head's additive remainder — so the
> post-fix null there is a measured float32 bound (test bound 1e-6), not exact zero. The right
> comparison is 4.9975e-02 → 2.960e-08, not 3.87e-04 → 0.0 (C-45, C-57).

`verify_arms_matched` passes **17/17 in both cases**, because `y_source` is deliberately not
an axis, so the verifier structurally cannot see this. The TransformerLens path is worse:
`_site_bias` looks for `b_out`/`bias` while TL's `Attention` spells it `b_O`, so the bias is
dropped entirely.

**Why it matters now.** `experiments/exp001_hebb.py` documents `--site blocks.11.attn.head.7`
as supported — **the head the parent project found carrying the period-2 cycle**, i.e. the
single most scientifically interesting site in the repo — and calls `run_matched_arms(...,
also_recomputed_y=True)`, the arm the file itself says "is the path the claim is made from".
Running that command produces a large, arms-matched, verifier-blessed "feedback effect" that is
**entirely an artifact of reconstructing `y` from one head**. No test covers `offline_control`
at a head site.

**The severed-path control does catch it — which is the good news.** The defect is invisible to
all 17 matched axes, because `y_source` is deliberately not an axis. But the head-site severed
floor comes out at 3.87e-04 against the documented 0.000e+00, four orders above the 1e-8
detection limit the suite asserts. **That is a control failure, and it is the correct
diagnostic**: anyone who runs the severed arm at a head site sees a non-zero floor and knows
something is wrong before trusting the routed number. The danger is only to someone who runs
the routed arm alone. This is the repo's own methodology working — and an argument for making
the severed arm mandatory rather than optional at any new site.

**No published result is affected — verified.** Every `site` field in every committed
artifact is `blocks.6.mlp`, and no head-site string appears anywhere in `experiments/output_*`.
This is a latent defect, not a corrupted result. But it is directly in front of T3.1 and
T3.2, so it must be fixed before the site sweep, not after it.

*Fix shape:* refuse `y_source="recomputed"` when the adapter is a head site (loud, one line),
or record the other heads' contribution as a residual and add it back.

> *Update, 2026-08-05 — the verdict is discharged. **This is no longer a blocker in front of the
> site sweep.*** T1.5 took the **second** of the two fix shapes named above, which is the better
> one: `replay_offline` branches on `site.shared_post_activity` and rebuilds the drifted full
> output additively as `record.y + x @ delta`, other heads and `b_O` frozen in the recorded `y`,
> so the head-site path is repaired rather than refused. The two regression tests this finding
> asked for exist. **T3.1 and T3.2 are unblocked** — the register's list of unrun work no longer
> carries T1.5, and T3.1's remaining blocker is compute, not correctness.
>
> *Two clauses in the paragraph above need restating, and only one of them was wrong.* "Every
> `site` field in every committed artifact is `blocks.6.mlp`" is **`retired` with C-40** — all
> twelve MLP down-projections have since carried plasticity. But **the conclusion it was
> supporting survives, on a narrower and still-true statement (C-64): no committed experiment
> applies plasticity at a *head* site** — 0 of 12 attention output projections and 0 of 144 head
> stripes — so nothing published was ever exposed to this defect. EXP-003 Stage 0 *measures*
> activity at all 144 head sites and applies plasticity at none; the head stripes are exercised
> by the test suite only, which is now where the fix is pinned.

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
ratio 1.15**), and propagate it into `HANDOVER.md` §5.4 and `RESONANCE_NOTE.md`. (F3)

**T0.2 — Promote the loop-displacement-matched control.** `anti_hebb`@2.944e-05 — ceiling-silent,
94% of `hebb`'s loop-state perturbation, opposite sign on the same axis, does not flip — paired
with `hebb`@3.925e-05 (same sign, quarter magnitude, also does not flip). Both are already in
`step_size_map.jsonl` and neither has ever been cited. Write them up as the real control, retire
the isotropic arm as decisive, and **do not** substitute the σ₁-matched triple: matching σ₁
leaves the loop perturbation 24–42× apart, which is the same defect that disqualified the
isotropic arm. The claim they support is "both magnitude and sign are required", not
"direction, not magnitude". (F4)

**T0.3 — Reset the bifurcation claim.** Retitle `BASIN_BIFURCATION.md`, set the `comrade`
result at ladder step 2 pending T1.1, add the `prolet`-basin-scatter comparison, and add one
sentence noting the offline arm flips the same basin. Correct `HANDOVER.md` §5.3. (F2)

**T0.4 — Fix the errors of fact** in F6 and F7. Two matter beyond hygiene: the "only cell"
error is suppressing a second clean flipping cell at `hebb`@1.18e-04, and the Chaudhary
contradiction has two live documents predicting opposite failure modes.

**T0.5 — Adopt margin discipline.** Every basin label gets its top1−top2 margin beside it,
and any flip below a threshold fixed *now, in advance* is reported `readout-ambiguous`.
69 of 125 baseline prompts sit below 0.5. Apply retroactively.

**T0.6 — Commit the prior-art search artifacts.** Query lists, dates, endpoints, returned
IDs. Converts eleven absence claims from assertion to record. (F10)

> **Status note, added 2026-08-05, after this review was written.** Much of the list below has
> since run, and **this list is not the record of it — `CLAIMS.md` is.** Read the per-item text
> below as the review's original reasoning, not as current state. Verified against the register
> and the committed artifacts:
>
> - **Run:** T0.4's "only cell" half (now corrected in place in `EXP_001_RESULTS.md` and
>   `HANDOVER.md` §3.2 — C-21); T1.1 (C-26 `retired`, C-56 `supported`); T1.2 (C-52, no
>   hysteresis); T1.4 (see the update under it — C-22 `retired`, C-55 `supported`); T1.5 (C-45
>   `retired`, C-57 `supported`); T2.1 and T2.1b (C-35 answered, C-58 and C-59 `supported`);
>   **T2.2 — EXP-002 ran and is merged** (C-53 answered; C-60 – C-63), so the "Still unrun" in
>   its own entry below is stale; T3.2 in substance (EXP-002 ran all twelve MLP
>   down-projections, so "tooling landed in PR #35, never used" is also stale — C-64); and
>   T3.3's cadence half (cadence 2 and 4 in T2.1, 4 and 12 in EXP-003 Stage 2 — C-58, C-64).
> - **Superseded:** T0.2. The claim it was to be written up in support of, "both magnitude and
>   sign are required", is itself `retired` — T1.4 took the sign clause down. What survives is
>   C-55.
> - **Still unrun:** T0.5 (margin discipline — C-55 records the gap), T0.6 (search artifacts —
>   C-42), T1.3 (C-28 still `provisional`), T2.3 (C-54, issue #49), T2.4 (C-14
>   `not-established`), T3.1 (0 of 12 attention projections and 0 of 144 head stripes have ever
>   carried plasticity — C-64), and T3.3's drive-β / leak-α half.
> - **Partly propagated:** T0.1 and T0.3. The register carries both answers (C-04 – C-07 for
>   issue #31, C-56 for the ladder step), but `HANDOVER.md` §3.4, §5.3 and §5.4 still read the
>   old way under a top-of-file notice rather than being corrected in place.

### Tier 1 — Cheap decisive experiments (CPU-minutes to hours)

**T1.1 — The coexistence test.** Under `W0 + ΔW`, seed the original frozen `prolet` state,
iterate 120 steps. Settles step 2 vs step 4. **~1 CPU-minute.** (F2)

**T1.2 — The α hysteresis sweep** (issue #26 already specifies it). Sweep α up through 1.5
and back down, each α seeded from the *previous* α's settled state. Retracing → smooth
deformation; a hysteresis loop → a real transition. Independent check on T1.1.

**T1.3 — Refine α\* around the real transition,** the interval (1.25, 1.50) where lag-1
collapses — not (0.50, 0.75) where only the argmax moves.

**T1.4 — The rank-matched random control.** A rank-1 random direction matched to `hebb` on
σ₁ **and** on loop-state displacement (`1 − cos(off)` ≈ 5.0e-03) — not isotropic noise at
matched ‖·‖_F, and not σ₁ alone, since F4 shows σ₁-matching leaves the loop perturbation
24–42× apart. If it also flips the basin, what remains of C-22 collapses and the result is
about magnitude on any rank-1 direction. **This can falsify the project's central claim and
it is cheap.** (F4)

> *Update:* run to completion after this review was written — 10/10 seeds both
> arms, `experiments/output_rank1_random/T1_4_RESULTS.md`. What it changes in
> the register is decided in the C-22 amendment PR (#42), not here.

**T1.5 — Fix `_recompute_y` at head sites, and add a head-site test to
`test_offline_control.py`.** **Blocks T3.1 and T3.2.** No published result is affected, but
the defect is invisible to all 17 axes — though the severed-path control *does* flag it, at a
floor of 3.87e-04 against a documented 0.000e+00 (a stand-in figure; reproducing it on real
weights is part of this task) — and it sits on the
path to `blocks.11.attn.head.7` — the site the parent project makes most interesting. Fix
before the site sweep, not after. (F11)

> *Update: **done**, and it no longer blocks T3.1 or T3.2.* Both halves ran — the fix (additive
> reconstruction of the shared full output, `record.y + x @ delta`, gated on
> `site.shared_post_activity`) and the head-site tests. **The stand-in floor was wrong by two
> orders in the safe direction**: on real GPT-2 small at `blocks.11.attn.head.7` the pre-fix
> severed floor is **4.9975e-02**, not 3.87e-04; post-fix it is **2.960e-08** with
> `bit_identical` **False**, because exact zero was never available at a head site — that is a
> whole-matrix property. C-45 `retired`, C-57 `supported`. Evidence:
> `experiments/output_t1_5/T1_5_RESULTS.md`.

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

> **A frozen transformer's iterated-dynamics attractor landscape is editable by a
> **near-rank-1** weight perturbation of ~1% derived from the model's own activation
> statistics, with no target and no externally specified objective.** A ceiling-silent update
> of **94%** the same loop-state magnitude, in the opposite sign on the same axis, does not
> reproduce it; neither does the same-sign update at a quarter the magnitude. In one case the
> edit changes the *dynamical class* of the trajectory, turning a fixed point into a period-2
> cycle. Measured at one site (`blocks.6.mlp`) of one model (gpt2-small): the basin flip on
> 3 prompts (C-20), the opposite-sign and quarter-magnitude controls on one prompt
> (`A01_physics`), the dynamical-class change on one prompt (`A04_climate`, n = 1).

**"Near-rank-1", not "rank-1".** F4 measures 95.8% of `‖ΔW‖²_F` in the leading component at
`routed:A01_physics` and 100.0% in the severed cells, but only **81–84%** in the two episode
cells. "Rank-1" is a good description of the severed edit and an approximation of the routed
one; it is not exact anywhere the episode actually ran.

**Do not cite the isotropic arm here.** It is Frobenius-matched to **Oja**, not to `hebb`
(`plasticity.py` subtracts the decay term for `"random"` as well as `"oja"` — the fix recorded
in `README.md`). It did not flip, but per C-23 it cannot establish direction-specificity, so
quoting it in the headline would re-import the defect C-23 retires.

**Say "no externally specified objective", not "no loss"** — C-11 establishes that plain Hebb
*is* gradient ascent on ½E‖y‖², so there is an implicit objective and the stronger phrase is
false.

**And the "structured, not generic" clause is `provisional`, not established.** C-22 is
`provisional` and **C-50 / T1.4 is open**: an unrun rank-1 random edit at matched σ₁ *and*
matched loop displacement can still falsify it. Until that runs, the licensed form is
*"current evidence indicates a structured edit; the generic-direction question is open."*
Scope it to the measured prompt and site — **one prompt (`A01_physics`), one seed, one site
(`blocks.6.mlp`)**. Running T1.4 before any write-up is what would let the clause stand
unqualified.

> *Update:* T1.4 has since run to completion (`T1_4_RESULTS.md`) — C-50 is no
> longer open: 4/6 displacement-matched random rank-1 directions flip the basin;
> 0/10 reach `comrade`; 4/10 could not be matched. The paragraph above is left
> as written; what the register says next is decided in PR #42, which proposes
> C-22 → `retired` with the surviving claim entered as C-55.

**Not "direction-specific, not magnitude-specific"** — an earlier draft said that and F2's own
α-sweep refutes it, since scaling this exact ΔW at fixed direction produces three different
basins. Both magnitude and sign matter; that is the claim, and it is the one the data carry.

> *Update, 2026-08-05 — this is the review's closing statement of the claim, and its last
> sentence is `retired`.* **C-22 fell to T1.4.** Magnitude still matters; **sign does not** — 4 of
> the 6 displacement-matched arbitrary rank-1 directions flip the basin with no particular sign
> on the `hebb`/`oja` axis, so the data do not carry the sign half. What the data do carry is
> **C-55**, and the licensed closing form of the editability claim is one clause longer and one
> clause weaker: *an arbitrary rank-1 direction at the same loop displacement usually moves the
> basin too — but never to `hebb`'s destination (`comrade` 0 of 10 seeds, 0 of 74 probe
> evaluations), and only at 66×–171× `hebb`'s weight cost.* So what is specific to the Hebbian
> edit is **its efficiency and its destination**, not the bare fact that the basin moves. Both
> observations the sentence above rests on remain true as measurements; the inference does not.
> The two clauses of the §6 headline claim that are **untouched** are the ones this review
> already bounded: the measured Hebbian flip (C-20) and the `A04` dynamical-class change (C-24),
> which are observations rather than inferences.

Every clause is measured, controlled and reproducible from committed artifacts. It is a claim
about the **editability of iterated dynamics**, which:

- survives the offline-arm result intact, because it never depended on feedback;
- survives the F2 correction, because it needs no created attractor — a boundary move is
  enough, and issue #25 called that "the first real result";
- makes the coupling number a **refinement** rather than the load-bearing claim: *and 12% of
  that edit is attributable to feedback*;
- and places the work next to model editing, activation steering and test-time adaptation,
  where there is an audience and a comparison class.

### Editability and coupling are AND, not either/or — but asymmetrically

Reordering the two is not a choice between them. They answer different halves of one
experiment, and both currently hold:

| | **Editability** | **Coupling** |
|---|---|---|
| Asks | does ΔW change what the model does? | does feedback change how ΔW is produced? |
| Object | the attractor landscape | the weight update itself |
| Rows | C-21, C-22, C-26 | C-30 – C-34 |
| Core number | 1.1% rank-1 edit moves the basin; one case changes dynamical class | 6.81° rotation, 2.8% shortening, 12% of drift, against a null of exactly zero |
| Status | flip measured; "structured, not generic" `provisional` (T1.4) | `supported`, conditional on `y_source="recomputed"` |
| Behavioural consequence | *is* the claim | **none** — both arms flip the same basin (C-33) |
| Falsified by | **T1.4 falsifies only the "structured, not generic" clause** — it cannot touch the measured Hebbian flip or the `A04` dynamical-class change, which are observations, not inferences | T2.1 finding no growth into outcome change |

> *Update, post-T1.4:* the falsification named in the last row happened — the "structured, not
> generic" clause is gone (C-22 `retired`), exactly as bounded above: the measured Hebbian flip
> and the `A04` dynamical-class change are untouched. The editability claim's surviving form is
> C-55's — an arbitrary rank-1 direction at the same loop displacement usually moves the basin,
> never to `hebb`'s destination, at 66×–171× the weight cost. The Status cells above are left
> as written for the record.

> *Update, post-T2.1b — the **Coupling** column's "Behavioural consequence: none" no longer
> holds, and this is the row that changed most.* **C-59: at 2.5× the working step size, ceiling
> silent, feedback does change the outcome.** The connected loop settles into the period-2
> `Divine` cycle (lag-1 0.685, lag-2 1.000, margin 2.36) while the feedback-severed arm ends on a
> near-stationary trajectory reading out **`【`**, a token outside the five baseline basin
> families. Drifts 0.0403 (closed) and 0.0465 (offline) against the 0.05 cap, **0 clip on all
> arms**, 17/17 axes matched, both margins over the pre-registered 0.05-logit threshold by more
> than 40×. This is **the first admissible feedback-changes-outcome observation in the project**,
> and it also discharges the last row's *"T2.1 finding no growth into outcome change"*: the share
> grows monotonically along all three coupling axes **and** crosses into changing the outcome, so
> the two claims merge as this section's "prize" paragraph below anticipated.
>
> **C-33 is not refuted — it is bounded, and the boundary is narrow.** Through the whole T2.1
> grid, **up to 2× eta and 240 steps**, the arms agreed on the settled basin at every
> ceiling-silent cell (**C-58**), including two agreed changes of dynamical class into `Divine`.
> The onset sits in the interval **(2×, 2.5×]** at 0.5× grid resolution. So the honest table cell
> reads: *none up to 2× eta and 240 steps (C-33, C-58); the outcome diverges at 2.5× eta (C-59)*.
> **Share does not predict divergence** — the agreeing 240-step cell carries share 0.3422 against
> the diverging cell's 0.3496; which axis you push matters, not how large the share is. Limits:
> n = 1 prompt, 1 site, 1 seed, `hebb` only; the severed arm's dynamical class is **unclassified**
> (late-window lag-1 0.998 does not establish convergence), so the established contrast is the
> basin-label disagreement plus the connected arm's verified period-2 signature, not a
> class-versus-class statement. Evidence: `experiments/output_t2_1/T2_1B_RESULTS.md`,
> pre-registered in `PREREGISTRATION_T2_1B.md` before the run.

**The dependency runs one way, and that is the whole argument for the ordering.** If
editability fails, coupling is worthless — feedback steering a weight change that changes
nothing is not a result. If coupling fails, editability is untouched. **Editability is
necessary for coupling to matter; coupling is not necessary for editability to matter.** So
editability is the claim, and coupling is a *measure taken on it*.

**Where they genuinely conflict:** C-33 — both arms flip the same basin — cuts both ways. It
*helps* editability (the effect needs no feedback, so the claim is simpler and more robust) and
*embarrasses* coupling (feedback alters ΔW and alters nothing observable). One fact, opposite
signs. That is why the two read as mutually exclusive when they are not.

**Where they combine, and this is the prize:** **T2.1 / C-35**. If the feedback-attributable
component grows with coupling strength and crosses into changing the outcome, the two claims
merge into one that is strictly larger than either — *the landscape is editable by the model's
own activation statistics, **and** the loop steers which edit happens*. That is the project's
founding question, answered properly. It is the highest-value experiment in the register.

**So nothing is dropped.** Lead editability; report coupling in the same breath as C-33's limit
— *"feedback measurably steers the update and does not, at this operating point, change the
outcome"*; run T2.1 to promote coupling from refinement to co-headline.

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
- **No external citation was verified against its source by a human.** One — Chaudhary — was
  verified by agent retrieval, three times independently (F10, C-44); the small remaining gap
  C-44 records is that no human has opened the PDF. For the rest, `PRIOR_ART.md`'s
  verification-status column is the record and issue #28's gaps remain open. (An earlier
  version of this bullet said no citation had been verified at all, contradicting F10 four
  sections above — the propagation failure this review documents, in its own closing section.)
- **The suite was confirmed to collect 295 tests but not run to completion.** CI on `main` is
  green as of 2026-07-31.
- **The code audit ran against a Conv1D-shaped stand-in** for the measured probes, because no
  GPT-2 checkpoint was cached in the review environment. The structural findings in F11 are
  read from source and hold regardless; the specific measured magnitudes (relative error
  0.838, `diff_over_drift` 6.02e-01) should be reproduced on real weights before the fix is
  called complete.
- **Not attempted:** any judgement on whether the parent ATR project's own findings are sound.
  This review takes the bridge's bit-exactness as given, since CI enforces it.

**This review has been through one adversarial pass against itself**, in which four of its
load-bearing claims were attacked by independent skeptics instructed to refute them, and the
verdicts adjudicated against the artifacts. **All four were refuted in part.** The corrections
are marked in place rather than silently applied:

| Claim attacked | Outcome |
|---|---|
| F2c — `comrade` "inside `prolet`'s scatter" | **Withdrawn.** Used the worst of 1485 pairs as the yardstick; against mean, median and nearest-basin gap the displacement is *larger*, not smaller |
| C-22 — "direction-specific, not magnitude-specific" | **Withdrawn.** Refuted by this review's own α-sweep. Replaced by a better control found in the same file: `anti_hebb`@2.944e-05 |
| C-32 — the 5.73:1 direction ratio | **Withdrawn.** Normalisation-dependent, ill-conditioned, and less direction-dominated than equal norms would give. Replaced by a rotation and a shortening |
| C-10 — `ΔW = C·W` "exactly" | **Corrected.** The site has a bias; the omitted term is 17% of `‖ΔW‖_F`, and the repo's own `u_right_sign_ref_cos` refutes the bias-free form by 3–4 orders |

**None of the three headline findings fell.** F1 survives because the severed null really is
`torch.equal`-exact zero. F2 survives on arguments (a) and (b), exactly as it predicted it would
if (c) failed. F3 was not attacked and reproduces directly from `BASELINE.md`.

The pattern is worth naming, because it is the same one the review documents in the project: in
every case the measurement was sound and the sentence written on top of it reached further than
the measurement licensed.

---

*Prepared as a repository leadership review. Every number quoted from the project's own runs is
reproducible from `experiments/output_*`. **Two exceptions, both in F11**: the head-site
magnitudes (relative error 0.838, `diff_over_drift` 6.02e-01, severed floor 3.87e-04) were
measured against a Conv1D-shaped stand-in rather than real GPT-2 weights, and are provisional
until reproduced — the structural finding they illustrate is read from source and does not
depend on them. Where this review and the code disagree, the code is right — check `git log`
since `ea4f0c1`.*
