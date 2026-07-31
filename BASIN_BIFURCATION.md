# Basin bifurcation — `comrade` is a created attractor

> **Review notice — this file's headline is not established.** `ALIGNMENT_REVIEW.md` §F2
> finds the "created attractor" reading refuted by evidence already in the repo. The
> measurements below are sound and reproduce exactly; the *interpretation* placed on them
> does not follow. Three points, all from committed artifacts:
>
> 1. **The `comrade` state is inside the `prolet` basin's ordinary scatter.** It sits
>    `1−cos` = 4.39e-03 from the `prolet` fixed point, against a worst within-`prolet` pair
>    of 3.39e-02 (`BASELINE.md`) — 7.7× further. The basin taxonomy's own resolution is
>    ~3e-03, comparable to the gap between `Anarch` and `prolet`.
> 2. **D2 shows one fixed point moving smoothly, not a bifurcation.** lag-1 stays ≈1.0 across
>    α = 0 → 1.25 and the state norm advances in near-equal increments (+11.6, +12.9, +14.3,
>    +15.3, +14.9). Only the *argmax* changes discretely at α\* — and an argmax always does,
>    including under perfectly smooth motion. The genuine qualitative transition is between
>    α = 1.25 and 1.50, where lag-1 collapses to 0.734 and the norm jumps +65.9 — and that
>    lands in the **pre-existing** `Divine` basin, i.e. ladder step 3.
> 3. **D1 does not discriminate step 3 or 4 from step 2.** For any ΔW ≠ 0 the perturbed map's
>    fixed point is generically not fixed under the unperturbed map, so a norm-matched random
>    edit would pass D1 identically. D1's own trace — smooth monotone relaxation back to
>    `prolet` — is the signature of a *displaced* fixed point.
>
> **Also missing:** this file never mentions the offline arm. The no-feedback arm flips the
> same basin (`EXP_001_RESULTS.md` §1), so nothing here is attributable to the loop's
> coupling. Read alone, this file implies otherwise.
>
> The decisive test is unrun and costs ~1 CPU-minute: under `W0 + ΔW`, seed the original
> frozen `prolet` state and iterate. Stays → two attractors coexist → step 4 stands. Moves →
> one displaced attractor → step 2. See `CLAIMS.md` C-26, C-27, C-28 and `ALIGNMENT_REVIEW.md`
> T1.1 – T1.3.

Issues #25, #32. `hebb`, eta = 7.06517e-05, site `blocks.6.mlp`, 120-step episode, cadence 1, `max_delta_frac` = 0.05, prompt `A01_physics`.

EXP-001 (`EXP_001_RESULTS.md`) showed that one 120-step closed `hebb` episode at this eta takes the basin `prolet` → `comrade` with the norm ceiling silent — the one cell in the step-size map where the loop moves. This file asks what kind of move it is, on issue #25's ladder: **step 3**, the episode walked the state across a boundary into a `comrade` basin the original frozen map already had (a *latent* attractor), or **step 4**, the episode *created* the `comrade` attractor (a bifurcation). Two independent discriminators are run; they agree. The `alpha`-sweep also closes issue #32 section 5, which asked of the installable ΔW: *smooth bias or a threshold?*

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

Restore W0 and iterate the **frozen** loop starting from the episode's final `comrade` state. A pre-existing basin (step 3) holds the state; a created attractor (step 4) is not present at W0, so the original map carries the state back out. `cos→c₀` is the cosine to the starting `comrade` state; `lag-1` / `lag-2` are the cosines to the previous one and two iterates — near 1.0 means the state itself is barely moving.

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

The frozen map leaves `comrade` at **iteration 4** (into `prolet`) and settles at `prolet` — the frozen baseline's own fixed point — with lag-1 returning to 1.000000. Crucially the state's own motion is smooth and monotone the whole way: lag-1 stays above 0.99992 at every sampled iteration and `cos→c₀` decays without a jump. There is no discontinuity for the argmax to ride; the `comrade` label sat on a thin ledge the original map slides straight off, not in a basin that map has. **`comrade` is not an attractor of the original frozen map** — a created attractor (issue #25 step 4, a bifurcation).

> *Interpretation.* D1 is the definition of step 4 made operational: the attractor the episode ended in is absent from the pre-episode map. The episode did not walk the state to a standing `comrade` basin; the accumulated ΔW is what put a `comrade` fixed point where there was none.

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

**Threshold, not smooth bias.** The basin is `prolet` for every `alpha` ≤ 0.50 and flips to `comrade` at **alpha\* = 0.75** — a discrete change localized to the interval (0.50, 0.75]. That is issue #32 section 5's answer: **a threshold.**

### Smooth logits, discrete attractor

Underneath the discrete basin flip the logits move smoothly. `comrade`'s rank climbs monotonically with `alpha` — 5th, 4th, 2nd, 1st across alpha = 0.00, 0.25, 0.50, 0.75 — and the `comrade − prolet` logit gap is a smooth, monotone function of `alpha` that crosses zero exactly inside (0.50, 0.75]: 0.00→-0.657, 0.25→-0.404, 0.50→-0.114, 0.75→+0.203, 1.00→+0.525. The argmax (the basin) is discrete; the logit it is the argmax **of** is not. Smooth logits, discrete attractor — the signature the result turns on.

### The `prolet` → `comrade` → `Divine` cascade

Past `comrade` the sweep bifurcates again. At alpha = 1.50 the basin tips into `Divine` and the lag-1 cosine **collapses** from ~1.0000 (a fixed point) to 0.7340 — the fixed point gives way to the period-2 cycle `Divine` sits on in the baseline. So the `alpha`-axis reads `prolet` (fixed point) → `comrade` (fixed point) → `Divine` (period-2): two bifurcations along one line, not one.

## What this establishes, and what it does not

**Establishes:**

- **Closes issue #32 section 5.** The installable-ΔW `alpha`-sweep answer is a **threshold**, not a smooth bias: the basin holds at `prolet` and flips discretely at alpha\* = 0.75.
- **Answers issue #25's step-3-vs-4.** `comrade` is a **created attractor** (step 4, a bifurcation), not a boundary move into a pre-existing basin (step 3). Both discriminators agree: D1 shows the frozen W0 map does not hold the `comrade` state (it leaves at iteration 4 and settles at `prolet`); D2 shows the `comrade` attractor appears **discretely** as `alpha` crosses alpha\*, which is what a bifurcation is.
- **The two discriminators are independent and agree.** D1 iterates from the settled state under the *unmodified* map; D2 installs *fractional* ΔW and reads the settled basin from a fresh start. Neither is the other restated.

**Does not / caveats:**

- **Reproduced, not loaded.** No raw closed-loop state or ΔW is persisted in the repo, so this run regenerates them from the frozen episode. The weight-space anchors match EXP-001 to ~9 figures, but the closed-loop state norm carries a ~0.1%-class float drift (reported in the fidelity table); the basin label is invariant to it.
- **The per-`alpha` `initial_state` protocol.** Each `alpha` is run from its own iteration-0 tensor recomputed under `W0 + alpha·ΔW` — a self-consistent frozen system per `alpha`, not the same iteration-0 tensor across `alpha`. Stated plainly so the sweep is checkable and so the `alpha` = 1.0 row is not mistaken for EXP-001's `closed` re-run.
- **alpha\* is grid-localized.** The flip is pinned only to the sweep grid — it lies in the interval reported above, not to a sharper edge; the grid does not resolve where inside it the crossing sits.
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
