# T1.5 — Fix the head-site `recomputed`-y reconstruction (issue #47)

**Agent:** magenta · **Date:** 2026-08-01 · **Backend:** TransformerLens GPT-2 small, `transformer-lens==3.6.0`
**Site under test:** `blocks.11.attn.head.7` (LAYER=11, HEAD=7 — the head the parent's period-2 oscillation lives on).
All norms and relative errors computed in float64. Model calls use `.venv/bin/python` / `.venv/bin/pytest` with `ATR_REQUIRE_MODEL=1`.

---

## 1. Pre-registered null interpretation (rule 6), fixed IN WRITING before running

The quantity is the **severed-path detection floor**: with the state-feedback path cut (the loop
reads at `blocks.3.hook_resid_post`, so `blocks.11`'s `W_O` never feeds back), the `recomputed`-y
offline arm and the closed arm see a bit-identical `x` at every step, so whatever they still differ
by is float noise, not feedback.

**What floor means "correct" at a per-head site:**

- **NOT exactly 0.0.** Exact bit-identity (`bit_identical is True`, `rel_fro_diff == 0.0`) is
  reachable *only at whole-matrix sites*, where the recomputed `y` is the site's own fused
  `addmm` and `_recompute_y` reproduces it bit-for-bit.
- At a head site the post-synaptic `y` is the **shared full projection output** — a fused
  twelve-head einsum (`hook_attn_out`) that all heads write into. The offline arm reconstructs the
  drifted full output **additively** as `record.y + x @ delta` (other heads and `b_O` frozen inside
  the recorded full `y`). Additive reconstruction is arithmetically exact but **not bit-for-bit**
  against a fused reduction, so the correct null is a **measured float32-noise bound**, with
  `bit_identical is False`.
- **Decision rule set before running:** "correct" ⇔ severed `rel_fro_diff` at float-noise level
  (~`1e-7` class) with `bit_identical is False`. A floor at O(1) or O(1e-2) means the single-head
  reconstruction defect is present. A floor of *exactly* 0.0 at a head site would itself be
  suspicious (it would mean `y` had collapsed to an addmm-shaped path).

This is the rule-6 pre-registration, and it deliberately departs from the whole-matrix "exactly
0.0" null. See §7 for the rule-3 tension this creates.

---

## 2. The defect (verified on real GPT-2 weights)

`offline_control.py` `replay_offline`, `recomputed`-y branch (pre-fix line 791):

```python
y = _recompute_y(x, plast._effective_W(), bias)   # x @ W_head  (+ full bias)
```

At a **whole-matrix** site (`blocks.6.mlp`) this is correct — it *is* the site's own forward, matched
bit-for-bit. At a **per-head** site it rebuilds a single head's `(N, 768)` contribution
(`x @ W_head`), which is a quantity the model never forms. The recorded `y` is the full 768-wide
output all twelve heads write into, so the reconstruction lands ≈1.0 relative off it.

**Reproduced on real weights (the stand-in's `0.838` does NOT transfer):**

| quantity | stand-in (pre-T1.5) | **real GPT-2 (measured)** |
|---|---|---|
| recomputed-y rel error vs full output, ATR loop state | 0.838 | **≈ 1.00** (per-sample 0.9988–1.0021; worst 1.002) |
| head-site `recomputed` severed `rel_fro_diff` | 3.87e-04 | **4.9975e-02** (saturates near the 0.05 `max_delta_frac` ceiling) |

The severed floor on real weights is **two orders larger** than the stand-in's `3.87e-04`: the
badly-reconstructed `y` drives the offline arm's `delta` to the 5 % drift ceiling, so
`rel_fro_diff ≈ 0.05` while the closed arm stays well inside it.

---

## 3. The fix (implemented)

**Branch mechanism: a site-type flag, not a runtime shape test.**

- `plasticity.py`: added class attribute `shared_post_activity = True` to **both** head adapters —
  `_HeadSliceSite` (line 607) and `_TransformerLensHeadSite` (line 709). Default absent (falsy)
  everywhere else.
- `offline_control.py` `replay_offline` (recomputed-y branch, lines 810–828): branch on the flag.

```python
if y_source == "recorded":
    y = record.y[j].to(plast.W0.dtype)
elif getattr(plast._site, "shared_post_activity", False):
    # Per-head site: additive-remainder reconstruction of the shared full output.
    y = record.y[j].to(plast.W0.dtype) + x @ plast.delta
else:
    y = _recompute_y(x, plast._effective_W(), bias)   # whole-matrix: fused addmm, unchanged
```

The additive identity `full_y = record.y + x @ delta` holds because every other head and `b_O` are
frozen inside the recorded full `y`, and only this head's block moved (by `delta`); `x` is already
this head's slice, so `x @ delta` is exactly its change to the shared output.

**The matrix path is untouched.** At `blocks.6.mlp` the additive form would reduce to the addmm
mathematically but reintroduce the ~5e-9 unfused-addition floor, breaking
`test_a_site_the_loop_does_not_route_through`'s `bit_identical`. Branching on the flag keeps the
matrix path on `_recompute_y` exactly.

**Docstrings updated** to state that at head sites `y` is the shared full output and the
reconstruction is additive: `replay_offline` (`y_source="recomputed"` description), `_recompute_y`
(a "NOT the path for per-head sites" note), and both adapter classes.

---

## 4. The two new tests — RED before, GREEN after

Both added to `tests/test_offline_control.py`. RED-before was verified by `git stash push
offline_control.py plasticity.py` (source fix removed, tests kept), running, then `git stash pop`.

### Test 1 — unit reconstruction: `test_recomputed_y_at_a_head_site_is_the_shared_full_output` (line 663)

Records at `blocks.11.attn.head.7` on the ATR loop, replays with `y_source='recomputed'` at
**eta=0** (so `delta ≡ 0` and the additive `x @ delta` remainder vanishes — the assertion isolates
the *base-output* choice), spies on every `y` the rule observes, and asserts worst rel error to
`record.y` (the full output) `< 1e-6`. The `x @ delta` remainder's correctness under real drift is
covered by Test 2, at the weight level.

| | measured | verdict |
|---|---|---|
| **RED (pre-fix)** | worst rel err = **1.002e+00** (rebuilt `x @ W_head`, a single head) | fails `< 1e-6` ✗ |
| **GREEN (post-fix)** | worst rel err = **0.000e+00** (sees `record.y` bit-for-bit; `x @ 0` adds nothing) | passes `< 1e-6` ✓ |

Post-fix is *exactly* 0.0, not the plan's `≈1e-7` estimate: at eta=0 the additive base reproduces
`record.y` bit-for-bit. The nonzero float-noise floor shows up in Test 2, where a real `delta`
drives the reconstruction.

### Test 2 — severed head-site floor: `test_a_head_site_the_loop_does_not_route_through` (line 717)

Uses the `shallow_loop` severed setup (reads at `blocks.3`, so `W_O[11]` never feeds back) and
`run_matched_arms(... 'blocks.11.attn.head.7' ... rerun_frozen=False)`; asserts the `recomputed`-y
severed `rel_fro_diff < BOUND`. **Deliberately does NOT assert `bit_identical is True`** (asserts
`is False` instead) — a twelve-head fused einsum cannot be reconstructed bit-for-bit from one head's
additive remainder. Comment cites the einsum-vs-additive reason.

| | measured | verdict |
|---|---|---|
| **RED (pre-fix)** | `recomputed` `rel_fro_diff` = **4.998e-02**, `bit_identical=False` | fails `< 1e-6` ✗ |
| **GREEN (post-fix)** | `recomputed` `rel_fro_diff` = **2.960e-08**, `bit_identical=False` | passes `< 1e-6` ✓ |

**BOUND = 1e-6.** Set from the measured post-fix floor **2.960e-08**: `1e-6` sits ~1.5 orders above
it (headroom for the ~2× hardware variation the module documents on such float-accumulation floors,
e.g. its own 2.9e-9 vs 4.8e-9) and ~4.7 orders below the pre-fix single-head value (~5e-2), so the
test is unambiguously red before and green after. Not tightened below `1e-6`: the floor is
hardware-dependent float-accumulation order and a value would be brittle where a bound is honest.

---

## 5. Reproduced real-weight magnitudes (the numbers C-45 needs)

Measured on real GPT-2 small (`transformer-lens 3.6.0`), float64 norms, ATR working-point loop
(`initial_state`/`make_atr_step`, prompt "The cat sat on the mat and then the", 6 steps, eta=1e-5).

| tag | quantity | value |
|---|---|---|
| **(a)** | buggy `recomputed`-y rel error vs full output, on the ATR loop state | **≈ 1.00** (per-sample 0.9988–1.0021; worst 1.002; sample-0 0.9988) — real analog of stand-in 0.838 |
| **(b)** | buggy head-site `recomputed` severed `rel_fro_diff` | **4.9975e-02** — real analog of stand-in 3.87e-04 (larger: it saturates near the 0.05 ceiling) |
| **(c)** | fixed floor: head-site `recomputed` severed `rel_fro_diff` | **2.960e-08** (`bit_identical=False`); unit reconstruction floor **0.0** at eta=0 |

**Controls (unchanged pre↔post fix, as required):**

- Whole-matrix (`blocks.6.mlp`) `recomputed` severed: `rel_fro_diff = 0.000e+00`, `bit_identical=True` — the matrix path stays bit-exact.
- Head-site `recorded`-y severed: `rel_fro_diff = 8.525e-05` — the `recorded` path is untouched by the fix.

---

## 6. Full-suite result

```
ATR_REQUIRE_MODEL=1 .venv/bin/pytest tests/test_offline_control.py tests/test_head_sites.py
  → 81 passed, 1 warning in 45.29s

ATR_REQUIRE_MODEL=1 .venv/bin/pytest        (whole repo)
  → 293 passed, 4 skipped, 1 warning in 82.49s
```

The matrix-path guards stay GREEN: `test_a_site_the_loop_does_not_route_through`
(`bit_identical` True, floor 0.0), `test_recompute_y_reproduces_the_sites_own_forward_bit_exactly`,
and `test_eta_zero_arms_are_bit_identical` all pass. The two new tests are GREEN post-fix and were
RED pre-fix (§4). No regressions.

---

## 7. The rule-3 / head-site-floor tension (flagged explicitly)

Rule 3, as embodied by the whole-matrix `test_a_site_the_loop_does_not_route_through`, pins the
severed no-feedback floor at **exactly 0.0** (`bit_identical is True`). That exact zero is a
**whole-matrix-only** property: it holds because the recomputed `y` there is the site's own fused
`addmm`, which `_recompute_y` reproduces bit-for-bit. At a **head site** the post-synaptic `y` is the
shared full output — a fused twelve-head einsum — and the offline arm reconstructs it **additively**
(`record.y + x @ delta`). Additive ≠ fused bit-for-bit, so the head-site severed floor is
structurally **nonzero** float32 noise (measured 2.960e-08). Pre-registering "exactly 0.0" as the
head-site null (rule 6) would pre-register an unreachable target and make a *correct* fix read as a
failure; the correct head-site null is therefore a **measured float-noise bound** (BOUND=1e-6, ~1.5
orders above the measured floor). The tension is made assertable in code: Test 2 asserts
`bit_identical is False` and bounds `rel_fro_diff`, with a comment citing the einsum-vs-additive
reason, and the fix's inline comment plus the `replay_offline`/`_recompute_y` docstrings state that
the exact-zero floor is a whole-matrix property only.

---

## 8. Recommendation on a new `supported` C-row

**Recommended: yes, add one** (proposed **C-57**, pending a fresh number on the registry — highest
current is C-56). C-45 stays `retired` as the historical record of the defect (its evidence refreshed
to real weights, §PROPOSED_CLAIMS_PATCH). A new `supported` row is the right home for the post-fix
correctness claim, because "retired rows are never deleted" and the *fixed* path is a new, positive
assertion the register does not yet make.

It would assert: *at a per-head site the offline arm's `recomputed`-y path reconstructs the shared
full projection output additively (`record.y + x @ delta`), so the severed no-feedback floor is
float32 noise (`rel_fro_diff` 2.96e-08, `bit_identical` False), not O(1)* — with the caveat that the
**exact-zero** severed floor is a whole-matrix property only (head-site `y` is a fused twelve-head
einsum reconstructed additively, so its null is a measured float-noise bound). Evidence: the two new
tests + the real-weight magnitudes above.

---

## Files changed (deliverables)

- `plasticity.py` — `shared_post_activity = True` on `_HeadSliceSite` (L607) and
  `_TransformerLensHeadSite` (L709), + docstrings.
- `offline_control.py` — flag-branch in `replay_offline` (L810–828), + `replay_offline` /
  `_recompute_y` docstrings.
- `tests/test_offline_control.py` — `test_recomputed_y_at_a_head_site_is_the_shared_full_output`
  (L663), `test_a_head_site_the_loop_does_not_route_through` (L717), `TL_HEAD_SITE` (L660).
- `experiments/output_t1_5/T1_5_RESULTS.md`, `experiments/output_t1_5/PROPOSED_CLAIMS_PATCH.md`.

No shared docs edited; no git add/commit/push.
