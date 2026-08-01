# Proposed CLAIMS.md edits — T1.1 (issue #45)

For orchestrator review and application. **This agent did not edit `CLAIMS.md`.** Evidence:
`experiments/output_t1_1/` (`T1_1_RESULTS.md`, `t1_1_trajectory.jsonl`, `meta.json`,
`PREREGISTRATION.md`). A status change is a PR of its own per the register's own rules; this
file supplies the exact before/after so that PR is mechanical.

Verdict summary: under `W0 + ΔW`, the original frozen `prolet` state **moves to `comrade`**
(settles iter 12, holds to 120; lag-1 ≥ 0.99990 throughout — a smooth ridge-crossing).
=> **one displaced attractor, not two coexisting** => C-26's created-attractor reading is
refuted by the test C-26 itself named decisive.

---

## Edit 1 — C-26  (status `not-established` → `retired`)

**Line 96. Current:**

> | **C-26** | `comrade` is a **created attractor** — ladder step 4, a bifurcation | `not-established` | Not established **by elimination and by the α-sweep — not by a distance argument**. D1 falsifies step 3 but leaves step 4 as a residual, and returns the same verdict for any ΔW that displaces the fixed point. The α-sweep shows the **observed settled branch** moving continuously (it does not exclude a second, unobserved attractor — that is T1.2): lag-1 ≥ 0.999998 across α = 0→1.25, ‖state‖ advancing +11.6/+12.9/+14.3/+15.3/+14.9, `prolet`'s logit falling 0.884 while `comrade`'s *also* falls | **The scatter comparison is withdrawn.** [...] **T1.1 is the decisive test** |

**Proposed replacement:**

> | **C-26** | `comrade` is a **created attractor** — ladder step 4, a bifurcation | `retired` | **Refuted by T1.1, the test this row named as deciding** (`experiments/output_t1_1/T1_1_RESULTS.md`, `t1_1_trajectory.jsonl`, `meta.json`). Under `W0 + ΔW` seeded from the **original frozen `prolet` state**, the loop does **not** stay: it moves to `comrade`, settling at iteration 12 and holding through 120, with lag-1 ≥ 0.99990 at every step (min 0.99991 at iter 1) and the top1−top2 margin falling smoothly to 0.00025 at the crossing before rising again. One **displaced** attractor, not two coexisting — the frozen `prolet` state falls into the single `comrade` fixed point (lag-1 = 1.0). The created-attractor (step-4) reading required coexistence; it fails. What is measured is ladder **step 2**: a single fixed point the edit relocates continuously and the readout relabels across a ridge (consistent with C-27, and with the α-sweep's continuously-moving settled branch). The eta=0 gate is bit-identical to the frozen loop (max abs diff 0.0). | **Verdict invariant** to the renorm shell (`init_norm` and the edited system's own shell both give `comrade`, step 12). **Scope:** T1.1 refutes the specific coexistence claim (the original `prolet` state staying put); the independent hysteresis cross-check is **C-52 / T1.2** — retrace → step 2 confirmed, a loop → a real transition. `comrade` and `prolet` sit close at settling (margin 0.321, `prolet` rank 3) — the dynamical displacement is real but does not beat C-07's basin-resolution limit. n = 1 prompt, 1 site, 1 eta. |

Rationale for `retired` (not `not-established`): the register's vocabulary reserves `retired`
for a claim that "was asserted, now contradicted or superseded." C-26 was asserted (as a
created attractor, throughout `BASIN_BIFURCATION.md`'s original reading) and is now
contradicted by its own nominated decisive test. `retired` rows are never deleted — the record
of the created-attractor reading stays.

---

## Edit 2 — C-51  (open row → answered, following the C-50 → T1.4 pattern)

**Line 138. Current:**

> | **C-51** | Under `W0 + ΔW`, does the original `prolet` state stay put? | T1.1 — settles C-26 |

**Proposed replacement:**

> | **C-51** | Under `W0 + ΔW`, does the original `prolet` state stay put? | **Answered by T1.1: no — it moves to `comrade`, settling at iteration 12 and holding through 120; lag-1 ≥ 0.99990 the whole way (a smooth ridge-crossing, not a jump).** One displaced attractor, not two coexisting; C-26 refuted → `retired`. The durable positive finding is C-56. Evidence: `experiments/output_t1_1/`. |

(Per the register's note on open rows — "C-50 was answered by T1.4 and its answer is C-55" —
an answered open row states its answer and points to the new supported row that carries the
durable claim. C-56 below is that row.)

---

## Edit 3 (recommended) — new row C-56  (the durable positive finding)

The C-50 → C-55 precedent: when an open row is answered, the durable claim enters as its own
`supported` row rather than living only in the answered question. Proposed new row, to be
placed with the F2 / working-point cluster (after C-28) or appended to the register:

> | **C-56** | The working-point edit `W0 + ΔW` (‖ΔW‖_F/‖W0‖_F = 0.0112) **displaces the single settled attractor** `prolet` → `comrade`; it does **not** create a `comrade` attractor beside a surviving `prolet` one | `supported` | T1.1 (`experiments/output_t1_1/T1_1_RESULTS.md`, `t1_1_trajectory.jsonl`, `meta.json`): seeded from the frozen `prolet` state under `W0 + ΔW`, the loop moves to `comrade` (settle iter 12, hold to 120, lag-1 = 1.0 at the settled fixed point; relL2 from the seed → 0.139). eta=0 gate bit-identical to the frozen loop; ΔW reproduced to ‖ΔW‖_F/‖W0‖_F = 0.011239339962675624 and σ₁ = 1.81352, 0/120 clip. | **Displacement, not coexistence** — the dual of `BASIN_BIFURCATION.md` D1 (which slid the `comrade` state back to `prolet` under W0). Verdict invariant to renorm shell. Independent audit is C-52/T1.2 (hysteresis). Settled `comrade`/`prolet` are within C-07's basin-resolution limit (margin 0.321, `prolet` rank 3). n = 1 prompt (`A01_physics`), 1 site (`blocks.6.mlp`), 1 eta. |

If the orchestrator prefers not to mint C-56, fold its content into C-51's answer and drop the
"The durable positive finding is C-56" clause from Edit 2.

---

## Downstream note (not an edit — flag only)

`README.md` still carries the "created attractor" framing that C-26 fed; `BASIN_BIFURCATION.md`
is already marked `[SUPERSEDED READING]` and self-consistently defers to C-26, so retiring C-26
leaves it correct. C-27 (threshold vs smooth bias) is untouched — T1.1 corroborates its
smooth-logits / discrete-argmax reading (margin → 0.00025 at the crossing) rather than
contradicting it. Per repo rules this agent is not editing those shared docs.
