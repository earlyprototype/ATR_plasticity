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
  experiment that would populate it. C-35 and C-50 – C-54 are the current set. An `open` row
  may never be quoted as a claim.
- **A row may cite a gap rather than an artifact** when the claim *is* about an absence — C-43
  ("not yet in `PRIOR_ART.md`") is the example. The evidence for an absence is the search that
  failed to find it, which is why T0.6 exists: commit the search artifacts and these rows get
  real evidence.

**Bootstrap note.** `ALIGNMENT_REVIEW.md` was written before this register existed, so its
findings F1–F11 are numbered independently and do not cite C-rows. The mapping is one-way and
recorded here: F1 → C-30…C-34, F2 → C-26…C-28, F3 → C-04…C-07, F4 → C-10, C-22, C-23,
F5 → C-13…C-15, F6 → C-41, F7 → C-44, C-46, C-47, F10 → C-42, C-43, F11 → C-45. Every
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

| ID | Claim | Status | Evidence | Caveat |
|---|---|---|---|---|
| **C-10** | The Hebbian update is exactly `ΔW = E[x xᵀ]·W = C·W` — one step of power iteration on the site's input second-moment matrix | `supported` | Analytic, from the repo's own convention; confirmed by measurement — 95.8% of `‖ΔW‖²_F` in component 1, stable rank 1.04 | Since the attractor states are position-uniform (C-06), C is effectively rank-1, so ΔW is a **rank-1 edit along the site's dominant activation mode** |
| **C-11** | "No loss" means **no externally specified objective**, not no objective | `supported` | Follows from C-10: plain Hebb is gradient ascent on ½E‖y‖²; Oja the same under a norm constraint | Supersedes the flat "no task, no loss, no target" phrasing wherever novelty is being argued |
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
| **C-22** | The basin change is specific to the update's **direction**, not its magnitude | `provisional` | Three near-rank-1 updates, all ceiling-silent, σ₁ matched **to within 6%**: `hebb` 1.8135 → **flips**; `oja` 1.9288 → does not; `anti_hebb` 1.8745 → does not. Separately `oja` at σ₁ 4.68 (**2.6× `hebb`'s**) still does not flip — magnitude running the *wrong way* is the stronger half of the argument | **Not "differing only in direction."** The arms sit at different etas (7.065e-05 vs 2.944e-06, a 24× gap) and σ₁ agrees only to 6%; the match is approximate and found on a log-spaced grid, not constructed. Two gaps before this reaches `supported`: (a) whether σ₁ matching across modes at different etas is fair or a coincidence of the grid, and (b) whether `oja`'s update — dominated by the brake, so pointing roughly back along `W` as a shrinkage — differs from `hebb`'s in a way the loop's L2 renormalisation treats differently, which would make "direction-specific" true for a duller reason. **T1.4 (rank-1 random at matched σ₁) is the deciding test.** n = 1 prompt |
| **C-23** | The norm-matched isotropic `random` arm establishes direction-specificity | `retired` | `random` holds σ₁/‖ΔW‖_F = **0.054** at every eta against `hebb`'s 0.979 — its operator norm never reaches `hebb`'s anywhere in the sweep (4–11× smaller) | Matched on the wrong quantity. Superseded by C-22. The genuinely missing control is a **rank-1 random direction at matched σ₁** — see T1.4, which can falsify C-22 |
| **C-24** | `A04_climate` changes **dynamical class** — fixed point (lag-1 0.99999) → the period-2 `Divine` orbit (lag-1 0.661) | `supported` | `exp001.jsonl`; nudged lag-1 sits inside the frozen `Divine` range 0.659–0.696 | **The strongest result in the repository.** lag-1 is a property of the trajectory, not of the readout, so no relabelling artifact can explain it. n = 1 prompt |
| **C-25** | The `A04` transition is a **boundary move into a pre-existing attractor** — issue #25 ladder step 3 | `supported` | `Divine` holds 34 of the 125 frozen baseline prompts | Issue #25 calls step 3 "the measurable one, and the first real result" |
| **C-26** | `comrade` is a **created attractor** — ladder step 4, a bifurcation | `not-established` | Refuted by the repo's own data: the `comrade` state sits `1−cos` = 4.39e-03 from the `prolet` fixed point, against a worst within-`prolet` pair of **3.39e-02** — 7.7× further. It is **inside** `prolet`'s ordinary scatter (C-04, C-07). lag-1 stays ≈1.0 across α = 0→1.25 while the state norm advances in near-equal increments | Currently the **title** of `BASIN_BIFURCATION.md` and a headline in `HANDOVER.md` §5.3. The parsimonious reading is ladder **step 2** — one continuously-moving fixed point whose argmax label changes. **Status is `not-established`, not `retired`**: the scatter comparison needs its comparability documented first — the `comrade` figure is a D1 relaxation trace under transformer_lens 3.6.0, the scatter figures are phase-aware position-mean pairwise statistics under 3.5.1, and position-uniformity (C-06) is what would make them equivalent. If that comparison is thrown out, the rest of F2 (smooth lag-1 across α, near-equal norm increments, D1 non-discriminating) still stands. Decisive test unrun: T1.1 |
| **C-27** | The α-sweep shows a **threshold**, not a smooth bias (issue #32 §5) | `not-established` | The underlying logit gap is smooth and monotone through the crossing; only the argmax is discrete. Driven mainly by `prolet`'s logit *falling* (16.95→16.07), not `comrade`'s rising | An argmax always changes discretely, including under perfectly smooth motion |
| **C-28** | A genuine change of dynamical class occurs between α = 1.25 and 1.50 | `provisional` | lag-1 collapses 0.999998 → 0.734; state norm jumps 65.9 against a running increment of ~15 | Real, and the only qualitative transition in the sweep — but it lands in the **pre-existing** `Divine` basin (step 3, not 4), and sits at 1.5× the ΔW the episode produced |

---

## D — The coupling question

| ID | Claim | Status | Evidence | Caveat |
|---|---|---|---|---|
| **C-30** | The severed-path control gives a null of **exactly zero** | `supported` | `exp001.jsonl`: all 5 severed cells `bit_identical: True`, `rel_fro_diff` exactly `0.0`, `torch.equal` True | Structural, not statistical: with no feedback path the arms compute the same function of the same inputs |
| **C-31** | **Feedback measurably steers the weight change.** The feedback-attributable component is 12% of total drift, against a null of exactly zero | `supported` | `EXP_001_RESULTS.md` §3: routed `diff_over_drift` 0.120 vs severed 0.000. Instrument round-off floor `1−cos` ≈ 1.5e-14, ten orders below the routed signal | The severed arm runs a shallower loop (0→3) at 2.6× the drift, so it is not a *statistically* matched control — the argument is structural (C-30). Independent n = 3 |
| **C-32** | The steering is a change of **direction**, not of scale — 5.73 : 1 | `supported` | Difference decomposed against `ΔW_closed` / `‖ΔW_closed‖`: perpendicular 0.1220, parallel 0.0213 | **Corrects** `EXP_001_RESULTS.md:133`, whose figures (0.1153/0.0346, ratio 3.33) are arithmetically right but decomposed against `ΔW_offline`, not the reference the prose names. The effect is *stronger* than published |
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
| **C-44** | Chaudhary 2025 (arXiv:2510.21908) §4.8: Hebbian plasticity stays stable at 8 layers and saturates; gradient-plastic diverges | `provisional` | Quote recorded in `PRIOR_ART.md` and issue #27 | **Agent-fetched; no human has opened the PDF.** Suspicious in shape — one paragraph resolving three open figures in the project's favour, and recommending "around 4 layers" while describing experiments only at 2 and 8. Nothing depends on it: the project has run at 12 layers and observed both branches directly (C-13, C-15). Mark `UNVERIFIED — agent-fetched` and demote to corroboration. `HANDOVER.md:319` still carries the **superseded** reading and predicts the opposite failure mode |
| **C-45** | The offline arm's `recomputed`-y path is trustworthy at head sites | `retired` | `offline_control.py:791` reconstructs `y` from one head's contribution rather than the shared full output. Measured relative error 0.838; head-site severed floor 3.87e-04 against a documented 0.000e+00 | **Invisible to all 17 matched axes** (`y_source` is not an axis). The severed-path control **does** flag it — floor 3.87e-04 against a documented 0.000e+00 — so the danger is only to someone who runs the routed arm alone. No published result affected: every artifact is `blocks.6.mlp`. Blocks the site sweep. T1.5 |
| **C-46** | The library records a clipping **rate** | `retired` | `clipped` is a latching boolean that only clears on `revert()`. `step_size_map.py` synthesises a rate by writing to semi-private state; `exp001_hebb.py` records the boolean and documents the limit | `ORIENTATION.md` over-claims; `README.md` is accurate |
| **C-47** | The 125 settled states are saved so within-basin spread can be re-measured with no further model time | `retired` | `.gitignore:27` excludes `experiments/**/states/`; the directory does not exist in the repo | The summary statistics in `BASELINE.md` survive (C-04 – C-07); the raw states do not. Any extension needs the ~6 CPU-hour baseline re-run |

---

## Open questions, tracked

| ID | Question | Where |
|---|---|---|
| **C-35** | Does coupling grow with drift? | T2.1 — highest value |
| **C-50** | Does a rank-1 random edit at matched σ₁ flip the basin? | T1.4 — **can falsify C-22** |
| **C-51** | Under `W0 + ΔW`, does the original `prolet` state stay put? | T1.1 — settles C-26 |
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

*Every row cites an artifact already committed. Where a row and the code disagree, the code
is right — open a PR and change the row.*
