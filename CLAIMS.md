# Claim register

*Every claim this project makes, what supports it, and its current status. This file is the
source of truth for what the repository asserts. Prose documents describe measurements;
**this file decides what those measurements are allowed to be called.***

Introduced by `ALIGNMENT_REVIEW.md`, which explains why: the same measurement was corrected
in one document and left stale in three others, and a completed measurement (C-11) was
recorded as unrun in three places. Nothing here is new evidence — every row cites an artifact
already in the repo.

## How to use this file

- **A claim enters the register before it enters any prose document.** If it is not here, it
  is not something the project asserts.
- **Write-ups cite row IDs**, not restatements. `EXP_001_RESULTS.md` describes what was
  measured; the register says what it means.
- **Changing a row's status is a pull request of its own**, with the evidence, and it goes
  through peer review. A status change is a bigger event than a new measurement.
- **`retired` rows are never deleted.** The record of what the project used to believe is
  part of the evidence.
- **Every row carries its caveat.** If the caveat cannot travel with the claim, the claim is
  not ready to leave this file.

**Two explicit exceptions to "every row cites a committed artifact":**

- **`open` rows cite no evidence, by definition.** They are questions the project intends to
  answer, not things it asserts. Their Evidence column reads `—` and their caveat names the
  experiment that would populate it. C-35 and C-52 – C-54 are the current set; C-50 was
  answered by T1.4 (answer C-55), and C-51 by T1.1 (answer C-56). An `open` row may never be quoted as a claim.
- **A row may cite a gap rather than an artifact** when the claim *is* about an absence — C-43
  ("not yet in `PRIOR_ART.md`") is the example. The evidence for an absence is the search that
  failed to find it, which is why T0.6 exists: commit the search artifacts and these rows get
  real evidence.

**Bootstrap note.** `ALIGNMENT_REVIEW.md` was written before this register existed, so its
findings F1–F11 are numbered independently and do not cite C-rows. The mapping is one-way and
recorded here: F1 → C-30…C-34, F2 → C-26…C-28, F3 → C-04…C-07, F4 → C-10, C-22, C-23,
F5 → C-13…C-15, F6 → C-41, F7 → C-44, F8 → C-47, F10 → C-42, C-43, F11 → C-45, C-46. Every
document written *after* this file cites row IDs directly.

## Status vocabulary

| Status | Means |
|---|---|
| `supported` | Measured, controlled, reproducible from a committed artifact. Quotable with its caveat. |
| `provisional` | Measured, but a named control or replication is missing. Quotable only with the gap stated. |
| `not-established` | Asserted somewhere in the repo without sufficient evidence. **Not quotable.** |
| `retired` | Was asserted, now contradicted or superseded. Kept for the record. |
| `open` | A question the project intends to answer. No claim yet. |

---

## A — The landscape (frozen, no plasticity)

| ID | Claim | Status | Evidence | Caveat that must travel with it |
|---|---|---|---|---|
| **C-01** | 125 prompts settle into 5 basins: `prolet` 55, `Divine` 34, `till` 19, `Anarch` 16, `solidarity` 1 | `supported` | `output_baseline/basins.jsonl`, `BASELINE.md` | Basin = top-1 token of the settled state. 69/125 sit at a top1−top2 margin below 0.5 |
| **C-02** | `Divine` is an exact period-2 limit cycle; all other basins are fixed points | `supported` | `BASELINE.md`; `instrument_validation.json` (lag-1 0.684912, lag-2 1.000000) | "Exact" means indistinguishable from period-2 at the float32 floor (`1−cos` ≈ 1.5e-14), not bit-identical |
| **C-03** | The `Divine` orbit is attracting and wide — 5/5 perturbation magnitudes return | `supported` | `RESONANCE_NOTE.md`; return criterion fixed before running | Contraction is **not** a constant. Per-decade rate ranges 4–124 across the ladder; the quoted 0.968 / 71-per-decade is an endpoint slope, not a fit. `EXP_001_RESULTS.md` §6 measures 0.9521 (47/decade) on a different trajectory |
| **C-04** | Within a basin, prompts settle onto **nearby-but-distinct** states — compression, not erasure | `supported` | `BASELINE.md` "Within-basin spread": mean `1−cos` 3.32e-03 over 2337 pairs | **Answers issue #31.** Interpretation was fixed in advance, as the issue required |
| **C-05** | Within-basin variation is essentially **one-dimensional** | `supported` | `BASELINE.md`: participation ratio 1.02–1.29 against a maximum of n−1 = 15–54 | |
| **C-06** | Position uniformity holds for **every** basin, not only `Divine` | `supported` | `BASELINE.md`: 125/125 prompts fully uniform (all position pairs above cos 0.999) | Corrects the parent's framing, which reports it for `Divine` |
| **C-07** | **The basin taxonomy has a resolution limit, and it is comparable to the basin separation itself** | `supported` | `BASELINE.md`: within-basin spread 3.319e-03 vs nearest-basin gap (`Anarch`–`prolet`) 2.874e-03 — **ratio 1.16** | The load-bearing caveat for every basin result in the repo. Two basins the project treats as distinct are no further apart than the prompts inside one of them |

---

## B — What the rules do

**Convention for C-10 – C-12.** The code stores `W` as `(n_in, n_out)` and computes
`y = x @ W + b` with `x` a **row** vector, so `E[x xᵀ]` and `E[x yᵀ]` below denote the
`(n_in, n_in)` and `(n_in, n_out)` outer-product averages `E[xᵀx]/n` and `E[xᵀy]/n` in code
terms — matching `plasticity.py:105-109`. Equivalent column-vector form: `y = Wᵀx + b`.

| ID | Claim | Status | Evidence | Caveat |
|---|---|---|---|---|
| **C-10** | The Hebbian update is exactly `ΔW = E[x xᵀ]·W + E[x]·b_outᵀ = C·W + x̄ b_outᵀ` — power iteration on the site's input second-moment matrix **plus a rank-1 bias term** | `supported` | `plasticity.py:448` (y is `hook_mlp_out` = `x@W + b_out`), `:105-109` (`ΔW = E[x yᵀ]`). Bias share `‖b_out‖/‖ȳ‖` = 3.353/19.451 = **17%**. Discriminating measurement: `u_right_sign_ref_cos` gives cos(v₁, ȳ) = **0.999999** in the severed cells, where a bias-free `v₁ ∝ Wᵀx̄` would give 0.9975 | **The bias-free form `ΔW = C·W` is retired** — it was refuted by the repo's own artifact by 3–4 orders of magnitude. Consequence: ΔW is a rank-1 edit whose output-side factor is **ȳ**, not `Wᵀx̄`, so ~17% of the direction EXP-001 decodes through the unembedding is a prompt-independent layer constant. Rank-1-ness (95.8% routed, 100.0% severed, 81–84% in the episode cells) follows from position-uniformity alone and does **not** test the `C·W` form |
| **C-11** | "No loss" means **no externally specified objective**, not no objective | `supported` | `∂/∂W ½E‖xW+b‖² = E[x yᵀ]` exactly — plain Hebb is gradient ascent on output energy, and the bias does not disturb the identity | **Oja clause narrowed:** the "same under a norm constraint" equivalence is exact for Oja's *single-unit* rule. The 768-output subspace rule this repo runs is a constrained gradient flow only when `WᵀW = I`, which fails here (`WᵀW` diagonal mean 35.4, stable rank 31.0) |
| **C-12** | `anti_hebb` negates the reinforcement term only and keeps the brake; the brake genuinely contracts | `supported` | `plasticity.py:1120` computes `−H − D`; brake read against the live effective weight; `tests/test_antihebbian.py` holds both failure directions | Bounded fixed point at `W* = −E[x yᵀ]E[y yᵀ]⁻¹` |
| **C-13** | `oja` is **inert** at this site — it moves the basin at no step size tested up to **2.9% drift with the ceiling silent** (0.0% clip) | `supported` | `step_size_map.jsonl`, the 5 `oja` cells at clip rate 0.0% | One site, one prompt. The likely mechanism (C-14) is untested. **Quote only the clean cells**: `oja` also stays `prolet` at 5% drift and 100% clip, but per the standing prohibition below that is a measurement of the ceiling and is a diagnostic note, not evidence |
| **C-14** | Oja is inert **because** the decay term dominates the reinforcement term ~110:1 at this site | `not-established` | Ratio measured at `blocks.6.mlp` only; never tested as an explanation | Plausible and important. See T2.4 |
| **C-15** | Raw Hebb "diverges immediately, unbounded weight growth" | `retired` | Contradicted: at the working point, 120 updates give 0.0% clip, 0 non-finite, ‖W‖_F +0.03% | Still asserted in `README.md` "Why Oja rather than Hebb". Control C3's premise depends on it |

---

## C — The basin-flip result

| ID | Claim | Status | Evidence | Caveat |
|---|---|---|---|---|
| **C-20** | A `hebb` edit of ~1.1% relative Frobenius norm at `blocks.6.mlp`, with the ceiling silent, changes the settled basin | `supported` | `output_exp001/exp001.jsonl`; `step_size_map.jsonl` | 3 independent prompts. `clipped` False on every arm of every cell |
| **C-21** | There are **two** ceiling-silent cells that flip the basin, not one | `supported` | `step_size_map.jsonl`: `hebb`@7.07e-05 (1.12% drift) and `hebb`@1.18e-04 (2.20% drift), both 0.0% clip, both `comrade` | **Corrects** the "only cell in the whole sweep" claim in `EXP_001_RESULTS.md` and `HANDOVER.md:96`. **Not an independent replication** — both cells share prompt, seed, site and cadence, so this is robustness of the flip across a 1.7× change in eta. Currently suppressed by an error |
| **C-22** | The basin flip requires **both** a sufficient magnitude **and** the right sign on the `hebb`/`oja` axis — neither alone | `retired` | **Refuted by T1.4, the test this row itself named as deciding** (`experiments/output_rank1_random/T1_4_RESULTS.md`, `rank1_random.jsonl`, `meta.json`): **4 of the 6** rank-1 random directions matched to `hebb`'s loop displacement within ±2% flip the basin — arbitrary directions, no particular sign on the `hebb`/`oja` axis. The 4 unmatchable seeds (2 at the search's scale cap, 2 straddling a displacement discontinuity) all flipped below target. The magnitude clause survives — Arm A at matched σ₁ gives 0/10 flips at displacements 3–5 orders below `hebb`'s — the sign clause does not | Second retirement on this row: "direction, not magnitude" fell to F2's α-sweep; "magnitude **and** sign" now falls to T1.4. The observations previously cited here (`anti_hebb`@2.944e-05 at 94% displacement not flipping; `hebb` at quarter displacement not flipping) remain true as measurements — what fell is the inference that the sign is *required*. What survives is narrower and lives in **C-55**: no random direction reaches `comrade`, and matching the displacement costs 66×–172× `hebb`'s relative weight change. n = 1 prompt, 1 site |
| **C-23** | The norm-matched isotropic `random` arm establishes direction-specificity | `retired` | `random` holds σ₁/‖ΔW‖_F = **0.054** at every eta against `hebb`'s 0.979 — its operator norm never reaches `hebb`'s anywhere in the sweep (4–11× smaller) | Matched on the wrong quantity. Superseded by C-22. The genuinely missing control was a **rank-1 random direction at matched σ₁** — T1.4 has since run it, and it did falsify C-22 (now `retired`; the surviving claim is C-55) |
| **C-24** | `A04_climate` changes **dynamical class** — fixed point (lag-1 0.99999) → the period-2 `Divine` orbit (lag-1 0.661) | `supported` | `exp001.jsonl`; nudged lag-1 sits inside the frozen `Divine` range 0.659–0.696 | **The strongest result in the repository.** lag-1 is a property of the trajectory, not of the readout, so no relabelling artifact can explain it. n = 1 prompt |
| **C-25** | The `A04` transition is a **boundary move into a pre-existing attractor** — issue #25 ladder step 3 | `supported` | `Divine` holds 34 of the 125 frozen baseline prompts | Issue #25 calls step 3 "the measurable one, and the first real result" |
| **C-26** | `comrade` is a **created attractor** — ladder step 4, a bifurcation | `retired` | **Refuted by T1.1, the test this row named as deciding** (`experiments/output_t1_1/T1_1_RESULTS.md`, `t1_1_trajectory.jsonl`, `meta.json`). Under `W0 + ΔW` seeded from the **original frozen `prolet` state**, the loop does **not** stay: it moves to `comrade`, settling at iteration 12 and holding through 120, with lag-1 ≥ 0.99990 at every step and the top1−top2 margin falling smoothly to 0.00025 at the crossing. One **displaced** attractor, not two coexisting — the frozen `prolet` state falls into the single `comrade` fixed point (lag-1 = 1.0). The created-attractor (step-4) reading required coexistence; it fails. What is measured is ladder **step 2**: a single fixed point the edit relocates continuously and the readout relabels across a ridge (consistent with C-27). eta=0 gate bit-identical to the frozen loop (max abs diff 0.0). Durable finding: **C-56** | **Verdict invariant** to the renorm shell. **Scope:** T1.1 refutes the specific coexistence claim (the original `prolet` state staying put); the independent hysteresis cross-check is **C-52 / T1.2**. `comrade` and `prolet` sit close at settling (margin 0.321, `prolet` rank 3) — within C-07 basin-resolution. n = 1 prompt, 1 site, 1 eta |
| **C-27** | The α-sweep shows a **threshold**, not a smooth bias (issue #32 §5) | `not-established` | The underlying logit gap is smooth and monotone through the crossing; only the argmax is discrete. Driven mainly by `prolet`'s logit *falling* (16.95→16.07), not `comrade`'s rising | An argmax always changes discretely, including under perfectly smooth motion |
| **C-28** | A genuine change of dynamical class occurs between α = 1.25 and 1.50 | `provisional` | lag-1 collapses 0.999998 → 0.734; state norm jumps 65.9 against a running increment of ~15 | Real, and the only qualitative transition in the sweep — but it lands in the **pre-existing** `Divine` basin (step 3, not 4), and sits at 1.5× the ΔW the episode produced |
| **C-55** | At `hebb`'s loop displacement, an **arbitrary** rank-1 direction usually moves the basin — but never to `hebb`'s destination, and only at 66×–172× `hebb`'s weight cost | `supported` | T1.4, 10 seeds per arm (`experiments/output_rank1_random/T1_4_RESULTS.md`, `rank1_random.jsonl`, `meta.json`): Arm B matched within ±2% on 6/10 seeds, **4/6 flip** (`Anarch` ×2, `bourgeois` ×2); destination `comrade` **0/10 seeds and 0 of all 74 probe evaluations**; drift cost ‖ΔW‖_F/‖W0‖_F **0.740–1.927** against `hebb`'s **0.0112**; Arm A (matched σ₁ = 1.8135) **0/10 flips**, displacements 4.0e+03–6.7e+05× smaller than `hebb`'s | n = 1 prompt (`A01_physics`), 1 site, 120-step episode. 4/10 seeds could not be matched — the flip-rate denominator is 6, not 10. **C-07 travels with the destinations**: 6 of the 8 flips land `Anarch`, whose gap to `prolet` (2.874e-03) is *below* the within-`prolet` spread (3.319e-03). Flip margins 0.020–0.684, no pre-registered ambiguity threshold (T0.5 pending). `bourgeois`, like `comrade`, is not among the 5 frozen baseline basins. Position uniformity holds at every recorded Arm B scale (min pairwise cos 0.99999999999993 at 0.39–2.20× ‖W0‖_F), which is what licenses the displacement match itself |
| **C-56** | The working-point edit `W0 + ΔW` (‖ΔW‖_F/‖W0‖_F = 0.0112) **displaces the single settled attractor** `prolet` → `comrade`; it does **not** create a `comrade` attractor beside a surviving `prolet` one | `supported` | T1.1 (`experiments/output_t1_1/T1_1_RESULTS.md`, `t1_1_trajectory.jsonl`, `meta.json`): seeded from the frozen `prolet` state under `W0 + ΔW`, the loop moves to `comrade` (settle iter 12, hold to 120, lag-1 = 1.0 at the settled fixed point; relL2 from the seed → 0.139). eta=0 gate bit-identical to the frozen loop; ΔW reproduced to ‖ΔW‖_F/‖W0‖_F = 0.011239339962675624 and σ₁ = 1.81352, 0/120 clip | **Displacement, not coexistence** — the dual of `BASIN_BIFURCATION.md` D1 (which slid the `comrade` state back to `prolet` under W0). Verdict invariant to renorm shell. Independent audit is C-52/T1.2 (hysteresis). Settled `comrade`/`prolet` within C-07 basin-resolution (margin 0.321, `prolet` rank 3). n = 1 prompt (`A01_physics`), 1 site (`blocks.6.mlp`), 1 eta |

---

## D — The coupling question

| ID | Claim | Status | Evidence | Caveat |
|---|---|---|---|---|
| **C-30** | The severed-path control gives a null of **exactly zero** | `supported` | `exp001.jsonl`: all 5 severed cells `bit_identical: True`, `rel_fro_diff` exactly `0.0`, `diff_over_drift` exactly `0.0` | **Holds in `recomputed` mode only.** In `recorded` mode the severed cells are *not* bit-identical (`rel_fro_diff` 0.0081–0.0084) and the severed floor (0.287) *exceeds* the routed value (0.248) — sign reversed. `offline_control.py:895` already declares `recorded` uninterpretable as a feedback test, and documents why: recorded-y freezes the rule's own y-recursion in one arm only. Structural argument verified: `blocks.3.hook_resid_post` is a function of blocks 0–3 only, all 12 blocks still execute, so the plastic site fires with no causal path to the next iterate |
| **C-31** | **Feedback measurably steers the weight change.** The feedback-attributable component is 12% of total drift, against a null of exactly zero | `supported` | `EXP_001_RESULTS.md` §3: routed `diff_over_drift` 0.120 vs severed 0.000. Instrument round-off floor `1−cos` ≈ 1.5e-14, ten orders below the routed signal | The severed arm runs a shallower loop (0→3) at 2.6× the drift, so it is not a *statistically* matched control — the argument is structural (C-30). Independent n = 3 **Conditional on `y_source="recomputed"`** — see C-30. The two feedback-free offline arms (recorded-y vs recomputed-y) differ from *each other* by `diff_over_drift` 0.238–0.248, roughly twice the feedback-attributable 0.112–0.120; `y_source` is deliberately not one of the 17 axes, and the severed arm is what shows its contribution is exactly zero when the path is cut. The 0.120 headline is normalised by `‖ΔW_offline‖` (`offline_control.py:975`, `diff / max(drift)`); against `‖ΔW_closed‖` it is 0.124 |
| **C-32** | The steering is predominantly a change of **direction**: a **6.81°** rotation with a **2.8%** shortening | `provisional` | cos 0.99294 → 6.81° (A01; 6.32° A02, 6.37° A04); `‖ΔW_closed‖/‖ΔW_offline‖` = 0.97225 | **Do not quote a perpendicular:parallel ratio.** It is the deterministic function `r·sinθ/\|r·cosθ−1\|` of those two numbers, takes 3.33 or 5.73 depending on reference, is ill-conditioned (`d ln ratio/d ln r` = −47, pole 2.1% away), and ranges 5.03–7.67 across prompts. At this cosine perpendicular dominance is automatic for any norm ratio in [0.900, 1.144] — at *equal* norms it would be 16.8, so the measured value is **less** direction-dominated than equal norms would give |
| **C-33** | At this step size and horizon, feedback does **not** change the behavioural outcome | `supported` | Both arms flip to the same basin on all 3 prompts | The honest limit on C-31, and it must travel with it |
| **C-34** | "The two arms agree, so feedback did nothing" | `retired` | Contradicted by C-30 + C-31. Issue #26's rubric ("near 1 → nothing but scale") was written before the severed control existed to calibrate what "near 1" means | Still the reading in `HANDOVER.md` §3.3 and the issue #32 comment |
| **C-35** | Does the feedback-attributable component **grow** with coupling strength, and is there a regime where it changes the outcome? | `open` | — | T2.1. The highest-value experiment in the project; turns C-31 into a phenomenon or into a clean negative result |
| **C-36** | ΔW's dominant direction carries no logit-lens-legible preference for the basin tokens | `not-established` | The null 5–95% bands span 91 percentile points (`prolet` [7.7, 98.4]) — the test cannot reject anything | Reported as an absence in `EXP_001_RESULTS.md` §4. It is an absence of power, not an absence of effect |

---

## E — Scope, novelty and infrastructure

| ID | Claim | Status | Evidence | Caveat |
|---|---|---|---|---|
| **C-40** | Every number in the repository is **one site** (`blocks.6.mlp`), one model, cadence 1 | `supported` | Verified: every `site` field in every committed artifact | 1 site of 168 candidates (12 MLP + 12 attn + 144 head stripes) |
| **C-41** | Independent n is **3 prompts**, not the 5 cells the tables imply | `supported` | `hebb` has no stochastic term, so seeds 0/1/2 are one run three times — spread exactly 0.000e+00 | `EXP_001_RESULTS.md` §3's "n=5" medians are forced onto A01's value, which is the extreme of the range in 3 of 4 rows |
| **C-42** | No work combines a hand-written local rule, a pretrained frozen model, a closed activation loop, and no objective | `provisional` | `PRIOR_ART.md` — and the file is explicit that this is a statement about a search, not about the literature | **11 absence claims have no preserved artifact.** The search was term-driven, not concept-driven, and never queried test-time adaptation (TENT), model editing (ROME/MEMIT), fast weights (Ba et al.), modern Hopfield (Ramsauer), predictive coding/DEQ, or Hebbian-at-scale (SoftHebb). Per C-11, "no loss" is the only load-bearing axis and it is softer than stated |
| **C-43** | ROME already makes rank-1 edits to a mid-stack GPT-2 MLP down-projection | `open` | Not yet in `PRIOR_ART.md` | The novel element is **not** "a rank-1 edit there changes behaviour". It is that the edit is derived with **no target**, from the model's own activation statistics, and read out on the **iterated map's attractor structure**. State it that way or a reviewer will say ROME first |
| **C-44** | Chaudhary 2025 (arXiv:2510.21908) §4.8: Hebbian plasticity stays stable at 8 layers and saturates (recall 0.729 ± 0.015); gradient-plastic diverges after ~3000 steps, deepest layers drifting most | `supported` | Quote recorded in `PRIOR_ART.md` and issue #27, **independently re-fetched during the alignment review** and matching the source word for word, including the §4.8 title *Task-Dependent Behaviour and Depth Stress Test* | **Agent-verified across three independent retrievals; still no human has opened the PDF.** That is the remaining gap, and it is small: `PRIOR_ART.md`'s original fetch, a direct re-fetch during the alignment review, and a third-party web query all return the passage verbatim. **A negative retrieval is weak evidence** — the third-party tool's first query reported that no such section exists, and its second, more specific query returned the section's findings word for word. Absence of a hit is not absence of the text. **Two limits that do bite:** the paper reports no 4-layer result at all (standard models are 2 layers, stress test 8 — "a practical regime around 4 layers" is interpolated), and **nothing in this literature tests 12 layers with any rule**, so it cannot set an expectation either way for GPT-2 small. Use as corroboration of the family split, never as an expectation for this substrate |
| **C-45** | The offline arm's `recomputed`-y path is trustworthy at head sites | `retired` | `offline_control.py:791` reconstructs `y` from one head's contribution rather than the shared full output. Relative error **0.838**; head-site severed floor **3.87e-04** against a documented 0.000e+00 — **all three measured against a Conv1D-shaped stand-in, not real GPT-2 weights, and `provisional` until reproduced (T1.5)**. The *structural* defect is read from source and does not depend on them | **Invisible to all 17 matched axes** (`y_source` is not an axis). The severed-path control **does** flag it — floor 3.87e-04 against a documented 0.000e+00 — so the danger is only to someone who runs the routed arm alone. No published result affected: every artifact is `blocks.6.mlp`. Blocks the site sweep. T1.5 |
| **C-46** | The library records a clipping **rate** | `retired` | `clipped` is a latching boolean that only clears on `revert()`. `step_size_map.py` synthesises a rate by writing to semi-private state; `exp001_hebb.py` records the boolean and documents the limit | `ORIENTATION.md` over-claims; `README.md` is accurate |
| **C-47** | The 125 settled states are saved so within-basin spread can be re-measured with no further model time | `retired` | `.gitignore:27` excludes `experiments/**/states/`; the directory does not exist in the repo | The summary statistics in `BASELINE.md` survive (C-04 – C-07); the raw states do not. Any extension needs the ~6 CPU-hour baseline re-run |

---

## Open questions, tracked

| ID | Question | Where |
|---|---|---|
| **C-35** | Does coupling grow with drift? | T2.1 — highest value |
| **C-50** | Does a rank-1 random edit at matched σ₁ **and matched loop-state displacement** flip the basin? | **Answered by T1.4: yes — 4/6 at matched displacement, never to `comrade`.** C-22 retired; the surviving claim is C-55 |
| **C-51** | Under `W0 + ΔW`, does the original `prolet` state stay put? | **Answered by T1.1: no — it moves to `comrade`, settling at iteration 12 and holding through 120 (lag-1 ≥ 0.99990 the whole way, a smooth ridge-crossing).** One displaced attractor, not two coexisting; C-26 refuted → `retired`; durable claim C-56. Evidence: `experiments/output_t1_1/` |
| **C-52** | Does the α-sweep show hysteresis? | T1.2 — independent check on C-26 |
| **C-53** | Does the persistence claim survive a fresh prompt? (EXP-002, issue #24) | T2.2 — the pre-registered primary experiment, still unrun |
| **C-54** | What does the 125-prompt library do at the working point? | T2.3 — largest credibility gain available |

---

## Standing prohibitions

Carried from `HANDOVER.md` §7, which learned each of these the hard way. They are not
negotiable and they are why the rows above are worth anything.

- **C0 is the gate.** At eta = 0 the hooks must not perturb the trajectory by a single bit.
- **A control that cannot fail is worse than no control.** Every control is tested in both
  directions.
- **Never reimplement the ATR loop.** Import it via `atr_bridge`.
- **Report the clipping rate with every result** — and note C-46, that the library does not
  give you one for free.
- **Never quote a cell where the ceiling fired.** Two such citations are currently live
  (`STEP_SIZE_MAP.md` §5's effective-rank evidence at 60.8% clip; `HANDOVER.md:90`'s
  `anarchism` flip at 100% clip).
- **Never use an even-only snapshot schedule.** Log lag-1 *and* lag-2.
- **Use tolerance, not `torch.equal`,** except where asserting bit-identity on purpose.
- **Don't tune to match the parent's basin percentages.** The bridge is bit-exact and
  CI-enforced; a mismatch means something else moved.
- **Say what you did not rule out.** Issue #27 is the list of failure modes that look like
  findings.
- **This is not learning.** The defensible phrase is that the weights carry a trace of the
  episode — and per C-11, "no externally specified objective" rather than "no objective".

---

*Every row that asserts something cites an artifact already committed — with the two exceptions
declared above: `open` rows cite no evidence by definition, and a row whose claim **is** an
absence cites the search that failed rather than an artifact. Where a row and the code disagree,
the code is right — open a PR and change the row.*
