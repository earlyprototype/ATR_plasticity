# Proposed `CLAIMS.md` patch — T1.5 (issue #47)

**PROPOSAL ONLY. Do not apply here.** `CLAIMS.md` is a shared doc; a status/evidence change is "a
pull request of its own, with the evidence, and it goes through peer review." This file records
what magenta proposes and why, backed by `T1_5_RESULTS.md`. Numbers are real GPT-2 small
(`transformer-lens 3.6.0`), float64 norms.

---

## 1. Refresh C-45 evidence (currently line 127, `retired`)

C-45 stays **`retired`** — it is the record of the pre-fix belief that the `recomputed`-y path was
"trustworthy at head sites", which was false. Only its **evidence and caveat** change: the stand-in
numbers are replaced with the reproduced real-weight ones, the "provisional until reproduced (T1.5)"
qualifier is **dropped** (T1.5 has now reproduced them), and a line notes the fix.

### Current row (verbatim)

> | **C-45** | The offline arm's `recomputed`-y path is trustworthy at head sites | `retired` | `offline_control.py:791` reconstructs `y` from one head's contribution rather than the shared full output. Relative error **0.838**; head-site severed floor **3.87e-04** against a documented 0.000e+00 — **all three measured against a Conv1D-shaped stand-in, not real GPT-2 weights, and `provisional` until reproduced (T1.5)**. The *structural* defect is read from source and does not depend on them | **Invisible to all 17 matched axes** (`y_source` is not an axis). The severed-path control **does** flag it — floor 3.87e-04 against a documented 0.000e+00 — so the danger is only to someone who runs the routed arm alone. No published result affected: every artifact is `blocks.6.mlp`. Blocks the site sweep. T1.5 |

### Proposed replacement row

> | **C-45** | The offline arm's `recomputed`-y path was **not** trustworthy at head sites (pre-fix) | `retired` | Pre-fix `offline_control.py:791` reconstructed `y` from one head's contribution (`x @ W_head`) rather than the shared full output. **Reproduced on real GPT-2 small (T1.5, `transformer-lens 3.6.0`):** recomputed-y relative error vs full output **≈ 1.00** (per-sample 0.9988–1.0021), head-site `recomputed` severed floor **4.9975e-02** (saturates near the 0.05 `max_delta_frac` ceiling) against the whole-matrix 0.000e+00. Stand-in figures (0.838 / 3.87e-04) did **not** transfer; the real severed floor is two orders larger. **Fixed in T1.5** by additive reconstruction of the shared full output (`record.y + x @ delta`, gated on `site.shared_post_activity`): the head-site `recomputed` severed floor drops from **O(1e-2) to float32 noise 2.960e-08** (`bit_identical False`) — **NOT exactly 0.0**. | **Invisible to all 17 matched axes** (`y_source` is not an axis). The severed-path control **does** flag it. No published result affected: every committed artifact is `blocks.6.mlp`. **The exact-zero severed floor is a whole-matrix property only** — at a head site `y` is a fused twelve-head einsum reconstructed additively, so its null is a measured float-noise bound (~1e-7 class), not 0.0. Post-fix correctness → proposed C-57. |

**Diff of substance:**
- `0.838` → `≈ 1.00`; `3.87e-04` → `4.9975e-02`; "documented 0.000e+00" kept as the *whole-matrix*
  reference.
- Drop "all three measured against a Conv1D-shaped stand-in … `provisional` until reproduced (T1.5)".
- Add: fix present; head-site severed floor now 2.960e-08, **NOT exactly 0.0**.
- Caveat: exact-zero severed floor holds **only at whole-matrix sites**; head-site null is a
  measured float-noise bound.
- Claim wording tweaked to past tense / "was not trustworthy" so the `retired` row reads correctly
  now that the defect is fixed (optional; evidence refresh is the load-bearing part).

---

## 2. OPTION — new `supported` row for the post-fix head-site path (proposed C-57)

A `supported` row asserting the **fixed** path's correctness. A new row (not un-retiring C-45) is the
right shape: retired rows are never deleted, and this is a new positive assertion. **Needs a fresh
C-number claimed on the registry** — the highest current is C-56, so **C-57** is proposed subject to
confirmation.

Suggested placement: section **E** (infrastructure), directly after C-45.

> | **C-57** | At a per-head site the offline arm's `recomputed`-y path reconstructs the **shared full** projection output, so the severed no-feedback floor is float32 noise, not O(1) | `supported` | `offline_control.py` `replay_offline` branches on `site.shared_post_activity` and rebuilds the drifted full output additively as `record.y + x @ delta` (other heads and `b_O` frozen in the recorded `y`). On real GPT-2 small at `blocks.11.attn.head.7`: unit reconstruction rel error to the full output **0.0** at eta=0 (`test_recomputed_y_at_a_head_site_is_the_shared_full_output`); severed `recomputed` `rel_fro_diff` **2.960e-08**, `bit_identical False` (`test_a_head_site_the_loop_does_not_route_through`), against the pre-fix 4.9975e-02. | The severed floor is **float32 noise (~1e-7 class), NOT exactly 0.0** — a fused twelve-head einsum cannot be reconstructed bit-for-bit from one head's additive remainder, so the null here is a measured bound (test uses `1e-6`), unlike the whole-matrix site's exact 0.0. The `recorded`-y path is unchanged (head-site severed floor 8.525e-05). Whole-matrix `blocks.6.mlp` path untouched and still bit-exact (0.0). |

**Recommendation:** add C-57. It is the only place the register would positively assert the fixed
behaviour, and it carries the whole-matrix-vs-head-site floor distinction that a future site sweep
must not lose.

---

## Notes for the reviewer applying this
- Both edits are to `CLAIMS.md` only; magenta did **not** touch it.
- If C-57 is added, cross-reference lines (`F11 → C-45, C-46`, header at line 39) may want C-57
  appended — reviewer's call.
- Evidence artifacts: `experiments/output_t1_5/T1_5_RESULTS.md`; tests at
  `tests/test_offline_control.py:663,717`.
