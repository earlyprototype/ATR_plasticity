# Basin bifurcation — `comrade` is a created attractor **[SUPERSEDED READING]**

*The measurements in this file stand. The title's claim does not — it is **`retired`** in
`CLAIMS.md` C-26, which is materially stronger than the `not-established` this file used to
report: not asserted without sufficient evidence, but **contradicted**, by T1.1, the test this
file itself named as decisive. The surviving reading is **C-56** — one **displaced** attractor,
issue #25 ladder step 2. Retained unrewritten as the record of what the project believed; read
the notice below before quoting anything here.*

> **Review notice — this file's headline is contradicted, and C-26 is `retired`.**
> `ALIGNMENT_REVIEW.md` §F2 found the "created attractor" reading unsupported by evidence
> already in the repo; **T1.1 has since contradicted it directly** (the last paragraph of this
> notice), so the register status is `retired`, not the `not-established` this file used to
> report. The measurements below are sound — the weight-space anchors reproduce EXP-001 to ~9 figures,
> while the closed-loop state norm carries the ~0.1%-class drift this file already reports
> (1.22e-03 relative, from regenerating rather than loading the state). It is the
> *interpretation* placed on them that does not follow. Three points, all from committed
> artifacts:
>
> 1. **No size argument settles this, in either direction.** The `comrade` state sits
>    `1−cos` = 4.39e-03 from the settled `prolet` state — which is *larger* than `prolet`'s
>    mean within-basin spread (2.77e-03), larger than the `Anarch`–`prolet` gap (2.87e-03),
>    and 7.3× larger than the median within-`prolet` pair (6.00e-04). It is smaller only than
>    the single worst of 1485 pairs. And within-basin variation is one-dimensional, so the
>    basin is a segment, not a ball; a saddle-node bifurcation would create a new attractor at
>    *zero* separation anyway. What the numbers do license: `comrade` sits 1.5× the
>    `Anarch`–`prolet` gap away, so **`comrade` and that pre-existing distinction stand or fall
>    together**, and at this scale the basin label is a coarse instrument (69/125 baseline
>    prompts below a 0.5 margin; `A01_physics` carries `comrade` at rank 4 when *frozen*).
> 2. **D2 shows the settled branch moving smoothly, not a bifurcation.** lag-1 stays ≈1.0 across
>    α = 0 → 1.25 and the state norm advances in near-equal increments (+11.6, +12.9, +14.3,
>    +15.3, +14.9). Only the *argmax* changes discretely, inside the bracket (0.50, 0.75] — and
>    an argmax always does, including under perfectly smooth motion. The genuine qualitative
>    transition is between α = 1.25 and 1.50, where lag-1 collapses to 0.734 and the norm jumps
>    +65.9 — and that lands in the **pre-existing** `Divine` basin, i.e. ladder step 3. This
>    describes the branch each α reaches **from its own iteration-0 tensor**; it is not a claim
>    that a single attractor is all there is at every α, which **C-68** withdraws — see
>    *The `alpha` = 0.50 row disagrees with T1.2* below.
> 3. **D1 does not discriminate step 3 or 4 from step 2.** For any ΔW ≠ 0 the perturbed map's
>    fixed point is generically not fixed under the unperturbed map — so **D1 alone cannot
>    distinguish the alternatives**. It returns the same verdict for any ΔW that displaces the
>    fixed point at all, a random edit included (which would not reproduce D1's *numbers* —
>    equal norms do not imply equal trajectories — but would reach the same *conclusion*).
>    D1's own trace — smooth monotone relaxation back to `prolet` — is the signature of a
>    *displaced* fixed point.
>
> **Also missing:** this file never mentions the offline arm. The no-feedback arm flips the
> **same basin** (`EXP_001_RESULTS.md` §1), so the flip is **not uniquely caused by feedback** —
> the rule reproduces it from frozen activation statistics alone. That is not the same as
> "feedback contributes nothing": the measured feedback-attributable component is 12% of total
> drift against a severed-path null of exactly zero (`CLAIMS.md` C-31). Two clauses travel with
> that number and are stated here rather than left to the reader. It is **conditional on
> `y_source="recomputed"`**: in `recorded` mode the severed floor (0.287) *exceeds* the routed
> value (0.248), so the comparison reverses, which is why `offline_control.py` already declares
> that mode uninterpretable as a feedback test (**C-30**). And **at this step size and horizon
> the steering does not change the behavioural outcome** — both arms flip to the same basin
> (**C-33**), which is the honest limit on the 12% and must travel with it. What the offline arm
> rules out is attributing *this basin flip* to the coupling. Read alone, this file implies
> otherwise.
>
> **The decisive test has since run, and it went against this file.** **T1.1**
> (`experiments/output_t1_1/T1_1_RESULTS.md`, `t1_1_trajectory.jsonl`, `meta.json`) seeded the
> original frozen `prolet` state under `W0 + ΔW` and iterated. It does **not** stay: the loop
> moves to `comrade`, settling at iteration **12** and holding through 120, with lag-1 at or
> above **0.99990** at every step and the top1−top2 margin falling smoothly to **0.00025** at
> the crossing. The eta = 0 gate is bit-identical to the frozen loop (max abs diff 0.0). Under
> the pre-registered two-outcome map that is the *moves* branch: **one displaced attractor, not
> two coexisting** — ladder **step 2**, register row **C-56**, `supported` — and **C-26 is
> `retired`**. **T1.2** corroborates it by the second, different route the register asked for:
> an α sweep taken up to 1.5 and back down **retraces**, with no hysteresis loop (**C-52**); its
> own limit is that a loop narrower than the 0.10 grid, entirely inside (0.4, 0.5), is not
> excluded. See `CLAIMS.md` C-26, C-27, C-28, C-52, C-56, C-68.

Issues #25, #32. `hebb`, eta = 7.06517e-05, site `blocks.6.mlp`, 120-step episode, cadence 1, `max_delta_frac` = 0.05, prompt `A01_physics`.

EXP-001 (`EXP_001_RESULTS.md`) showed that one 120-step closed `hebb` episode at this eta takes the basin `prolet` → `comrade` with the norm ceiling silent. It is **not** the only cell in the step-size map where the loop moves: there are **two** ceiling-silent cells that flip the basin — `hebb`@7.07e-05 (1.12% drift, the working point used here) and `hebb`@1.18e-04 (2.20% drift), both at 0.0% clip and both to `comrade` (**C-21**). The two share prompt, seed, site and cadence, so they are **not** an independent replication; what they establish is that the flip is robust across a 1.7× change in eta. This file asks what kind of move it is, on issue #25's ladder: **step 3**, the episode walked the state across a boundary into a `comrade` basin the original frozen map already had (a *latent* attractor), or **step 4**, the episode *created* the `comrade` attractor (a bifurcation). Two independent discriminators are run; they agree — but, as the notice above records, on a reading neither of them can establish, and T1.1 has since contradicted it (C-26 `retired`, C-56 `supported`). The `alpha`-sweep also addresses issue #32 section 5, which asked of the installable ΔW: *smooth bias or a threshold?* — the answer this file gave, *a threshold*, is **C-27** and is `not-established`, so that question is not closed here either.

**Measurement first.** The tables below are measurements; the words *created attractor* and *bifurcation* are an interpretation of them, marked as interpretation where it is made.

## Setup and fidelity anchors

| | |
|---|---|
| model | gpt2-small, frozen, CPU, float32 (norms float64) |
| site | `blocks.6.mlp` (= `transformer.h.6.mlp.c_proj`), (3072, 768) |
| prompt | `A01_physics` — "The implications of quantum entanglement suggest that" |
| rule | `hebb`, eta = 7.06517e-05, cadence 1, `max_delta_frac` = 0.05, seed 0 |
| loop | layers 0→11, episode 120 steps, D1 200 steps, D2 120 steps/alpha |
| alphas | 0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50 |

Nothing is loaded: the repo persists no raw closed-loop state and no ΔW, so the episode is **reproduced** here from the frozen loop and `comrade_state` / `ΔW = W_final − W0` are captured in memory. The anchors are EXP-001's recorded numbers, checked against, not fitted to.

| quantity | this run | EXP-001 recorded | rel. diff |
|---|---|---|---|
| ‖x₀‖ (initial_norm) | 1289.226318 | 1289.226318 | 0.00e+00 |
| ‖W0‖_F | 164.854073 | 164.854073 | 0.00e+00 |
| rel ΔW (`delta_frac`) | 0.011239340 | 0.011239340 | 0.00e+00 |
| ΔW σ₁ | 1.8135175 | 1.8135175 | 2.09e-09 |
| closed basin | `comrade` (47998) | `comrade` (47998) | match |
| frozen baseline basin | `prolet` (22758) | `prolet` (22758) | match |
| closed ‖state‖ (reproduced) | 4830.14307 | 4836.03898 | 1.22e-03 |

The weight-space anchors reproduce EXP-001 to ~9 figures — `delta_frac` at 0.0e+00 relative, ΔW σ₁ at 2.1e-09 — so the ΔW being taken apart below is the EXP-001 ΔW. The closed-loop **state** norm sits 0.12% off its EXP-001 value: the reproduced-not-loaded drift (the state is regenerated from the loop rather than loaded, under transformer_lens 3.6.0 vs the recorded run's), and the basin label survives it. The episode applied 120 updates, `clipped` = False (the ceiling never fired), `nonfinite` = False. After the episode W0 is restored exactly: ‖W − W0‖_F / ‖W0‖_F = 0.0e+00.

## D1 — is `comrade` a fixed point of the original (frozen) map?

Restore W0 and iterate the **frozen** loop starting from the episode's final `comrade` state. A pre-existing basin (step 3) holds the state; a created attractor (step 4) is not present at W0, so the original map carries the state back out — and so does a merely **displaced** fixed point (step 2), which is the alternative this discriminator cannot separate and the reason its verdict below is stated as a measurement only. `cos→c₀` is the cosine to the starting `comrade` state; `lag-1` / `lag-2` are the cosines to the previous one and two iterates — near 1.0 means the state itself is barely moving.

| iter | basin | lag-1 | lag-2 | cos→c₀ | relL2→c₀ |
|---|---|---|---|---|---|
| 1 | `comrade` (47998) | 0.999918 | -- | 0.99992 | 0.0130 |
| 2 | `comrade` (47998) | 0.999968 | 0.999823 | 0.99982 | 0.0188 |
| 3 | `comrade` (47998) | 0.999985 | 0.999915 | 0.99974 | 0.0229 |
| 4 | `prolet` (22758) | 0.999988 | 0.999949 | 0.99965 | 0.0264 |
| 5 | `prolet` (22758) | 0.999992 | 0.999962 | 0.99958 | 0.0292 |
| 10 | `prolet` (22758) | 0.999996 | 0.999981 | 0.99916 | 0.0411 |
| 20 | `prolet` (22758) | 0.999998 | 0.999992 | 0.99825 | 0.0593 |
| 30 | `prolet` (22758) | 1.000000 | 0.999996 | 0.99740 | 0.0722 |
| 50 | `prolet` (22758) | 1.000000 | 1.000000 | 0.99630 | 0.0862 |
| 75 | `prolet` (22758) | 1.000000 | 1.000000 | 0.99580 | 0.0917 |
| 100 | `prolet` (22758) | 1.000000 | 1.000000 | 0.99567 | 0.0931 |
| 150 | `prolet` (22758) | 1.000000 | 1.000000 | 0.99562 | 0.0937 |
| 200 | `prolet` (22758) | 1.000000 | 1.000000 | 0.99561 | 0.0937 |

The frozen map leaves `comrade` at **iteration 4** (into `prolet`) and settles at `prolet` — the frozen baseline's own fixed point — with lag-1 returning to 1.000000. Crucially the state's own motion is smooth and monotone the whole way: lag-1 stays at or above 0.999918 — its minimum, at iteration 1 — at every sampled iteration, and `cos→c₀` decays without a jump. There is no discontinuity for the argmax to ride; the `comrade` label sat on a thin ledge the original map slides straight off, not in a basin that map has. **The measurement: `comrade` is not an attractor of the original frozen map.** That stands as recorded.

What it **cannot** establish is *creation*. D1 returns the same verdict for **any** ΔW that displaces the fixed point at all — for any ΔW ≠ 0 the perturbed map's fixed point is generically not fixed under the unperturbed one — so this trace is equally what a *displaced* fixed point produces, and the smooth monotone relaxation back to `prolet` is that signature rather than evidence against it. Discriminating the two needs the dual experiment, seeding the frozen `prolet` state under `W0 + ΔW`; that is T1.1, it has run, and it found displacement (**C-56**). **[The reading this file originally drew here — a created attractor, issue #25 step 4, a bifurcation — is superseded; C-26 is `retired`.]**

> *Interpretation, in the words it was originally written in — kept as the record of what the
> project believed.* **[SUPERSEDED — C-26 is `retired`, refuted by T1.1; the paragraph above
> gives why D1 cannot carry this reading, and C-56 is what replaces it.]** D1 was taken as the
> definition of step 4 made operational: the attractor the episode ended in is absent from the pre-episode map. The episode did not walk the state to a standing `comrade` basin; the accumulated ΔW is what put a `comrade` fixed point where there was none.

## D2 — the installable ΔW `alpha`-sweep (issue #32 section 5)

Install `W0 + alpha·ΔW` and run the frozen loop under each `alpha`, then read the settled basin. A **smooth bias** would flip the basin somewhere proportional to `alpha` with the state deforming continuously; a **threshold** flips it discretely at some `alpha*` while the underlying logits move smoothly through the crossing.

*Protocol, stated so it is checkable:* each `alpha` is run from its **own** iteration-0 tensor, recomputed by a clean forward pass under `W0 + alpha·ΔW`, so each `alpha` is a self-consistent frozen system. This is not EXP-001 section 1's same-iteration-0 protocol; the `alpha` = 1.0 row is therefore not identical to EXP-001's `closed` re-run.

`comrade rank` is `comrade`'s position in the full 50257-token logit ordering at the settled state (1 = argmax); the logit columns are the raw logits of `comrade` and `prolet`. `gap` is `comrade − prolet`.

| alpha | basin | lag-1 | comrade rank | comrade logit | prolet logit | gap (c−p) | settled |
|---|---|---|---|---|---|---|---|
| 0.00 | `prolet` (22758) | 1.000000 | 5 | 16.2930 | 16.9495 | -0.6565 | yes |
| 0.25 | `prolet` (22758) | 1.000000 | 4 | 16.2847 | 16.6891 | -0.4044 | yes |
| 0.50 | `prolet` (22758) | 1.000000 | 2 | 16.2763 | 16.3908 | -0.1144 | yes |
| 0.75 | `comrade` (47998) | 0.999999 | 1 | 16.2686 | 16.0659 | +0.2027 | yes |
| 1.00 | `comrade` (47998) | 0.999999 | 1 | 16.2523 | 15.7276 | +0.5247 | yes |
| 1.25 | `comrade` (47998) | 0.999998 | 1 | 16.2098 | 15.3940 | +0.8158 | yes |
| 1.50 | `Divine` (13009) | 0.733997 | 1652 | 7.6442 | 4.9095 | +2.7347 | yes |

**A bracket, and not a threshold this sweep establishes.** The basin reads `prolet` at every sampled `alpha` ≤ 0.50 and `comrade` from 0.75 on. The sweep steps in 0.25, so what the grid gives is a **bracket** — the change of label lies in (0.50, 0.75] — and **not** a threshold at 0.75; reading the grid point 0.75 as a measured crossing takes a sampling artefact for a measurement, and is withdrawn. Whether it is a threshold **at all** is **`not-established`** (`CLAIMS.md` **C-27**): the underlying logit gap is smooth and monotone through the crossing and only the argmax is discrete, and an argmax changes discretely even under perfectly smooth motion, so the discreteness is a property of the readout rather than evidence about the dynamics. Note also which side moves: the crossing is driven mainly by `prolet`'s logit **falling** (16.95 → 16.07 across `alpha` = 0.00 → 0.75), not by `comrade`'s rising — `comrade`'s own logit barely moves over the same span, and slightly falls (16.29 → 16.27), so its climb in rank is other tokens falling past it. **[The reading this file originally drew here — "threshold, not smooth bias", issue #32 section 5's answer — is superseded.]**

### Smooth logits, discrete attractor

Underneath the discrete basin flip the logits move smoothly. `comrade`'s rank climbs monotonically with `alpha` — 5th, 4th, 2nd, 1st across alpha = 0.00, 0.25, 0.50, 0.75 — and the `comrade − prolet` logit gap is a smooth, monotone function of `alpha` that crosses zero exactly inside (0.50, 0.75]: 0.00→-0.657, 0.25→-0.404, 0.50→-0.114, 0.75→+0.203, 1.00→+0.525. The argmax (the basin) is discrete; the logit it is the argmax **of** is not. Smooth logits, discrete attractor — the signature the result turns on.

### The `alpha` = 0.50 row disagrees with T1.2, and the disagreement is unresolved (C-68)

The `alpha` = 0.50 row above settles on **`prolet`**, at margin 0.1144, lag-1 0.9999996, with 15 of 15 tail iterations agreeing. T1.2's continuation sweep (`experiments/output_t1_2/t1_2_hysteresis.jsonl`) settles on **`comrade`** at the same `alpha`, in **all four** of its arms — including `robust_up`, whose seed word is `prolet`. Same site, same rule, same eta, same prompt, same 120 steps, the same ΔW (`delta_frac` 0.011239339962675624, σ₁ 1.8135175 in both metas), the same transformer_lens 3.6.0. This is registered as **C-68**, `provisional`, and it is **unresolved**: this file owns one half of it and cannot read it either way.

The **seeding** is the candidate explanation, and it is the one axis the two runs do not share. D2 starts each `alpha` from its own iteration-0 tensor, recomputed under `W0 + alpha·ΔW` (the protocol stated above); T1.2 is a continuation sweep that seeds each `alpha` from the previous `alpha`'s settled state. If that is what it is, then at `alpha` = 0.50 two initial states reach two different settled words under **identical** weights. The alternative is that it is a readout artifact rather than genuine bistability: `comrade` sits at rank 2 here, so the margin **0.1144** *is* the `prolet` − `comrade` logit gap (the table's −0.1144), and that sits inside **C-07**'s resolution limit — the within-`prolet` spread 3.319e-03 against the nearest-basin gap 2.874e-03, ratio 1.15 — so the label is being read at a separation the basin taxonomy cannot resolve. The two runs are not a matched pair, which is why the row is `provisional` and not `supported`.

The deciding test is a **basin-of-attraction probe at `alpha` = 0.50**: install `W0 + 0.5·ΔW` and seed the same map from several initial states — both settled words and the iteration-0 tensor among them — with a convergence criterion **fixed in advance**. It has not been run. Until it does, this sweep may not be described as showing a single attractor at every `alpha`; what each row records is the settled word reached **from that `alpha`'s own iteration-0 tensor**. **C-56 is untouched** by this: T1.1 tested `alpha` = 1.00 directly, seeding the frozen `prolet` state, and it moved to `comrade`.

### The `prolet` → `comrade` → `Divine` cascade

Past `comrade` the sweep changes dynamical class — **[the word "bifurcates" is the
superseded reading; what is measured is a class change, which is real and is ladder step 3]**. At alpha = 1.50 the basin tips into `Divine` and the lag-1 cosine **collapses** from ~1.0000 (a fixed point) to 0.7340 — the fixed point gives way to the period-2 cycle `Divine` sits on in the baseline. So the `alpha`-axis reads `prolet` (fixed point) → `comrade` (fixed point) → `Divine` (period-2): two transitions along one line — the first a relabelling of the readout while the settled branch moves smoothly (ladder step 2, and no longer pending: T1.1 established it, C-26 `retired`, C-56 `supported`, with C-68 the open discrepancy at `alpha` = 0.50), the second a genuine change of dynamical class into the pre-existing `Divine` orbit (step 3, C-28, `provisional`).

## What this establishes, and what it does not

**Establishes:**

- **A bracket on the `alpha`-sweep's change of label.** The basin reads `prolet` at every sampled `alpha` ≤ 0.50 and `comrade` from 0.75 on; the sweep steps in 0.25, so the established statement is the **bracket** (0.50, 0.75], not a threshold at 0.75.
- **D1's measurement.** Seeded at the episode's `comrade` state, the **frozen** W0 map does not hold it: it leaves at iteration 4 and settles at `prolet`, smoothly and monotonically (lag-1 at or above 0.999918 at every sampled iteration). `comrade` is not an attractor of the original frozen map. What that supports is displacement, not creation — see §D1 and C-56.

**Superseded readings, kept in place as the record:**

- **[SUPERSEDED] Closes issue #32 section 5.** This file answered *smooth bias or a threshold?* with **a threshold** at alpha\* = 0.75. That reading is `CLAIMS.md` **C-27**, `not-established`, and is **not quotable**: an argmax changes discretely even under perfectly smooth motion, and here the underlying logit gap is smooth and monotone through the crossing — driven mainly by `prolet`'s logit falling, 16.95 → 16.07, rather than `comrade`'s rising, 16.29 → 16.27. The section is not closed; what survives is the bracket above.
- **[SUPERSEDED] Answers issue #25's step-3-vs-4.** This file concluded `comrade` is a **created attractor** (step 4, a bifurcation) rather than a boundary move (step 3). **C-26 is `retired`** — not merely unsupported but **contradicted**, by T1.1, the test this file named as deciding: seeded from the original frozen `prolet` state, the edited map does not hold it, so the two resting states do not coexist and the edit **displaces** the single attractor (**C-56**, corroborated by T1.2/**C-52**'s hysteresis-free retrace). The `A04`→`Divine` class change *is* step 3 and is not affected. Both discriminators do agree, but not on creation: D1 returns the same verdict for **any** ΔW that displaces the fixed point, and D2's discrete change of label is an argmax crossing a ridge, which is **not** a bifurcation.
- **[SUPERSEDED] The two discriminators are independent and agree.** They agree, but on a reading neither can establish — D1 iterates from the settled state under the *unmodified* map; D2 installs *fractional* ΔW and reads the settled basin from a fresh start. Neither is the other restated.

**Does not / caveats:**

- **Reproduced, not loaded.** No raw closed-loop state or ΔW is persisted in the repo, so this run regenerates them from the frozen episode. The weight-space anchors match EXP-001 to ~9 figures, but the closed-loop state norm carries a ~0.1%-class float drift (reported in the fidelity table); the basin label is invariant to it.
- **The per-`alpha` `initial_state` protocol.** Each `alpha` is run from its own iteration-0 tensor recomputed under `W0 + alpha·ΔW` — a self-consistent frozen system per `alpha`, not the same iteration-0 tensor across `alpha`. Stated plainly so the sweep is checkable and so the `alpha` = 1.0 row is not mistaken for EXP-001's `closed` re-run.
- **The change of label is grid-localized, and "alpha\*" overstates it.** It is pinned only to the sweep grid: it lies in the bracket (0.50, 0.75], not at a sharper edge, and the 0.25 grid does not resolve where inside that interval the crossing sits — nor whether the crossing is a threshold at all (C-27). T1.2's finer continuation sweep, stepping in 0.10, places its own crossing near `alpha` ≈ 0.45; that is a different seeding scheme and therefore a different measurement, not a contradiction, but the two are not interchangeable (C-52, and C-68 for where they disagree outright).
- **A discrepancy this file owns half of.** At `alpha` = 0.50 this sweep and T1.2 reach different settled words from different initial states under identical weights — **C-68**, `provisional`, unresolved. See the subsection above; the named deciding test has not run.
- **One cell.** One prompt (`A01_physics`), one site (`blocks.6.mlp`), one eta, one ceiling, one seed. `hebb` has no stochastic term and the model is frozen and single-threaded, so a seed is a single deterministic run, not a sample; the map's one-prompt-one-site caveats carry over unchanged.
- **No task, no loss, no target.** As with EXP-001, ΔW is what the rule produces on this activation distribution; this file measures the dynamical-systems character of the result, not a trained objective.

## Provenance

Reproduced episode + D1 (200 steps) + D2 (7 alphas × 120 steps), 6.2 CPU-minutes, one torch thread, deterministic.

```json
{
  "alphas": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5
  ],
  "cadence": 1,
  "d1_n_steps": 200,
  "d2_n_steps": 120,
  "device": "cpu",
  "dtype": "float32",
  "episode_ok": true,
  "eta": 7.065171428571429e-05,
  "eta_provenance": "D * ||W0||_F / (N_STEPS * U_ref[hebb]) with D=1.8e-2, U_ref=350, ||W0||_F=164.854 -- the step-size map's anchor, recomputed not rounded (same as EXP-001)",
  "finished": "2026-07-29T21:03:29Z",
  "issues": [
    25,
    32
  ],
  "layer_end": 11,
  "layer_start": 0,
  "max_delta_frac": 0.05,
  "mode": "hebb",
  "model": "gpt2-small",
  "n_episode": 120,
  "norms_dtype": "float64",
  "platform": "Linux 6.18.5 x86_64",
  "prompt": "The implications of quantum entanglement suggest that",
  "prompt_id": "A01_physics",
  "python_version": "3.11.15",
  "repo_rev": "db0183536164ab92e7aadc65071e46222e4adf8b",
  "seed": 0,
  "shards": 1,
  "site": "blocks.6.mlp",
  "started": "2026-07-29T20:57:09Z",
  "torch_threads": 1,
  "torch_version": "2.13.0+cpu",
  "total_seconds": 370.3,
  "transformer_lens_version": "3.6.0",
  "wall_clock_seconds": 370.3
}
```

Raw records — the reproduce/fidelity block, every sampled D1 iterate and every per-`alpha` D2 row: `experiments/output_basin_bifurcation/basin_bifurcation.jsonl`.
