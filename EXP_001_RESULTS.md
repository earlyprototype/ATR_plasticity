# EXP-001 — the offline arm at hebb, eta = 7.06517e-05

Issues #26, #30, #32. `hebb`, eta = 7.06517e-05, site `blocks.6.mlp`, 120 steps, cadence 1, `max_delta_frac` = 0.05.

The step-size map found `hebb` at this eta to be the only cell in the whole sweep that moves the loop inside a clean band: basin `prolet` → `comrade` at 1.12% relative weight change with the norm ceiling silent. `oja`, `anti_hebb` and `random` all have wide usable bands in which nothing whatsoever happens. Every offline-control number previously recorded in this repo was taken at `oja`, eta 1e-5 — inside the dead zone. This is the first time the control has been run where the loop actually moves.

## 1. Does the offline arm flip the basin too?

`A01_physics`, seed 0, layers 0→11. Each arm's final matrix is installed and the loop re-run **frozen** under it from the same iteration-0 tensor, so the four rows differ only in which matrix produced the trajectory.

| arm | basin at iter 120 | top1−top2 logit margin | lag-1 | lag-2 | cos(final, frozen-baseline final) |
|---|---|---|---|---|---|
| frozen baseline (W0) | `prolet` | 0.230 | 1.00000 | 1.00000 | 1.0 (self) |
| closed loop | `comrade` | 0.248 | 1.00000 | 0.99998 | 0.993518 |
| offline, `recomputed` y | `comrade` | 0.200 | 1.00000 | 0.99998 | 0.991624 |
| offline, `recorded` y | `comrade` | 0.291 | 1.00000 | 0.99999 | 0.995913 |

**The offline arm flips too.** Closed loop `prolet` → `comrade`; offline (`recomputed` y, the zero-floor path) `prolet` → `comrade`. Feedback is not required for the basin flip: the flip is a property of the rule applied to this activation distribution.

Closed arm against offline (`recomputed`) arm, final states: phase-aware cos = 0.999842824 (phase `final_prev`), relative L2 1.917e-02. The float32 round-off floor of this instrument is `1 − cos` ≈ 1.5e-14; this pair sits at 1.572e-04, i.e. 1e+10× the floor. `torch.equal` is not used anywhere here: two states that are the same point of the dynamics differ in their last bits.

The frozen baseline sits at a top1−top2 logit margin of 0.230. 69 of the 125 baseline prompts have a margin below 0.5 and this prompt is one of them. The lag-1/lag-2 columns and the final-state cosine are the quantities that do not depend on where an argmax happened to land.

## 2. Arms matched, and the ceiling silent

`verify_arms_matched` passed on **17/17 axes** for every cell (5 routed + 5 severed). 

**The ceiling never fired**, on any arm of any cell — `clipped` False after all 120 updates in every case, as the step-size map's 0.0% clip rate for this cell predicted. The number below is the rule, not `max_delta_frac`.

| cell | rel ΔW closed | rel ΔW offline (recomputed) | rel ΔW offline (recorded) | clipped |
|---|---|---|---|---|
| `routed:A01_physics:seed0` | 1.124e-02 | 1.156e-02 | 8.819e-03 | no |
| `routed:A01_physics:seed1` | 1.124e-02 | 1.156e-02 | 8.819e-03 | no |
| `routed:A01_physics:seed2` | 1.124e-02 | 1.156e-02 | 8.819e-03 | no |
| `routed:A02_medical:seed0` | 1.087e-02 | 1.118e-02 | 8.607e-03 | no |
| `routed:A04_climate:seed0` | 1.080e-02 | 1.103e-02 | 8.523e-03 | no |
| `severed:A01_physics:seed0` | 2.919e-02 | 2.919e-02 | 2.082e-02 | no |
| `severed:A01_physics:seed1` | 2.919e-02 | 2.919e-02 | 2.082e-02 | no |
| `severed:A01_physics:seed2` | 2.919e-02 | 2.919e-02 | 2.082e-02 | no |
| `severed:A02_medical:seed0` | 2.854e-02 | 2.854e-02 | 2.043e-02 | no |
| `severed:A04_climate:seed0` | 2.867e-02 | 2.867e-02 | 2.050e-02 | no |

## 3. Routed against severed

The severed arm reads the loop out at `blocks.3.hook_resid_post`, below the plastic site at layer 6. No state feedback can exist, so the `x` reaching the rule is bit-identical in both arms and **whatever the arms still differ by there is the floor**.

| configuration | y_source | `cos_delta` | `rel_fro_diff` | `diff_over_drift` |
|---|---|---|---|---|
| routed (0→11) | `recomputed` (floor 0) | 9.929e-01 … 9.939e-01 (median 9.929e-01, n=5) | 1.233e-03 … 1.392e-03 (median 1.392e-03, n=5) | 1.118e-01 … 1.204e-01 (median 1.204e-01, n=5) |
| routed (0→11) | `recorded` (floor ≠ 0) | 9.905e-01 … 9.914e-01 (median 9.905e-01, n=5) | 2.597e-03 … 2.782e-03 (median 2.782e-03, n=5) | 2.390e-01 … 2.476e-01 (median 2.476e-01, n=5) |
| severed (0→3) | `recomputed` (floor 0) | 1.000e+00 … 1.000e+00 (median 1.000e+00, n=5) | 0.000e+00 … 0.000e+00 (median 0.000e+00, n=5) | 0.000e+00 … 0.000e+00 (median 0.000e+00, n=5) |
| severed (0→3) | `recorded` (floor ≠ 0) | 1.000e+00 … 1.000e+00 (median 1.000e+00, n=5) | 8.111e-03 … 8.372e-03 (median 8.372e-03, n=5) | 2.842e-01 … 2.868e-01 (median 2.868e-01, n=5) |

In the zero-floor `recomputed` mode the routed cells sit at 1.204e-01 against a severed floor of 0.000e+00. The severed arms come out **bit-identical** — `rel_fro_diff` is exactly 0.0, not small — so the floor is zero in the literal sense and the ratio is not a finite number. The routed difference is therefore entirely above the floor. The usual objection to the severed control — that it runs a shallower loop (0→3) with different activation statistics, so its floor is not exactly the routed loop's — does not bite in this mode: `torch.equal` on the two matrices is True, and zero is zero at any depth. It does bite on the `recorded` row below.

In the default `recorded` mode the same comparison reads 2.476e-01 routed against 2.868e-01 severed. **The severed number is the larger of the two**, i.e. in `recorded` mode the measured difference is larger with the feedback path severed than with it routed.

### Seeds

Seeds [0, 1, 2] on the same prompt give `diff_over_drift` (recomputed) 1.203852e-01, 1.203852e-01, 1.203852e-01 — spread 0.000e+00.

**The spread is exactly zero, and that is a fact about the rule, not a lucky run.** `seed` reaches `OjaPlasticity` only through `self._rng`, which is drawn from in `mode="random"` and nowhere else. `hebb` has no stochastic component, the model is frozen and single-threaded, so three seeds are three bit-identical runs. Reporting them as a three-seed spread would be reporting the same run three times. Variation across prompts is reported below.

Across prompts (all `prolet` under the frozen loop), seed 0:

| prompt | basin frozen → closed | margin frozen | basin offline (recomputed) | `diff_over_drift` recomputed | severed |
|---|---|---|---|---|---|
| `A01_physics` | `prolet` → `comrade` | 0.230 | `comrade` | 1.204e-01 | 0.000e+00 |
| `A02_medical` | `prolet` → `comrade` | 0.280 | `comrade` | 1.122e-01 | 0.000e+00 |
| `A04_climate` | `prolet` → `Divine` | 0.147 | `Divine` | 1.118e-01 | 0.000e+00 |

## 4. Decoding ΔW (issue #32 sections 2 and 3a)

ΔW is (3072, 768) at this site, ‖ΔW‖_F = 1.8529 (1.124% of ‖W0‖_F).

| quantity | closed loop | offline (recomputed) |
|---|---|---|
| σ₁ | 1.8135 | 1.8696 |
| σ₂ | 0.3059 | 0.2991 |
| σ₃ | 0.1744 | 0.1706 |
| fraction of ‖ΔW‖²_F in component 1 | 0.9580 | 0.9624 |
| fraction in the top 5 | 0.9991 | 0.9993 |
| effective rank (participation ratio) | 2.00 | 1.93 |
| stable rank | 1.04 | 1.04 |

The step-size map measured ΔW effective rank 1.8–3.8 for `oja` and 718.8 for the isotropic noise arm. This is the `hebb` number at the cell that moves the loop.

### The dominant direction, decoded

The 768-side factor of ΔW lives in the residual stream's output space; pushed through `W_U` (no `b_U` — a direction is not a state, and the unembedding bias would give every direction, including the controls, the same ordering). The SVD's sign is arbitrary, so it is fixed by requiring a positive inner product with the mean post-synaptic activity at the site over the frozen episode (cos = +0.9938); the same rule is applied to every cell so the cross-basin cosines below are comparable.

| rank | ΔW top | ΔW bottom | random control #0 top | random control #1 top |
|---|---|---|---|---|
| 1 | " Robo" (+0.67) | " Bundy" (-0.66) | "\ufffd\ufffd" (+0.81) | " Shinra" (+0.71) |
| 2 | " 2018" (+0.66) | "\ufffd\ufffd\u6975" (-0.64) | "\ufffd\ufffd\ufffd" (+0.69) | "sole" (+0.69) |
| 3 | " esc" (+0.55) | "ICS" (-0.64) | "ndra" (+0.66) | "sf" (+0.67) |
| 4 | "2018" (+0.55) | " barracks" (-0.61) | "Rog" (+0.66) | "stal" (+0.66) |
| 5 | " preparations" (+0.55) | " tru" (-0.60) | "Air" (+0.62) | "SEA" (+0.66) |
| 6 | "pper" (+0.52) | "\u5973" (-0.60) | "NetMessage" (+0.62) | " dstg" (+0.66) |
| 7 | " Annotations" (+0.52) | "ICLE" (-0.58) | "Kal" (+0.62) | "Constructed" (+0.65) |
| 8 | " Grad" (+0.51) | "justice" (-0.58) | "hiro" (+0.62) | " Ley" (+0.64) |
| 9 | " Dolphin" (+0.50) | " ILCS" (-0.58) | "Pal" (+0.61) | " Nguyen" (+0.62) |
| 10 | "erate" (+0.50) | " Tradable" (-0.57) | "Motor" (+0.61) | " Ply" (+0.62) |
| 11 | "\u200e" (+0.50) | " FAR" (-0.57) | "Susan" (+0.60) | "agall" (+0.61) |
| 12 | " plans" (+0.50) | "AIN" (-0.56) | "months" (+0.59) | "utor" (+0.60) |
| 13 | " mock" (+0.50) | "\ufffd" (-0.55) | " Lung" (+0.59) | " Radiant" (+0.60) |
| 14 | " 2017" (+0.50) | "ById" (-0.55) | " Musk" (+0.59) | "\u5973" (+0.60) |
| 15 | " preview" (+0.49) | "teenth" (-0.55) | "\ufffd\ufffd" (+0.57) | " Colts" (+0.59) |
| 16 | " experiments" (+0.49) | "othing" (-0.54) | " Hyundai" (+0.57) | " Piper" (+0.59) |
| 17 | " Remastered" (+0.49) | "\ufffd\ufffd" (-0.53) | " pal" (+0.57) | "uyomi" (+0.58) |
| 18 | " GIF" (+0.49) | "ciating" (-0.53) | "jong" (+0.57) | " Tamil" (+0.58) |
| 19 | " \u200e" (+0.49) | "assium" (-0.53) | "Sand" (+0.57) | "eri" (+0.58) |
| 20 | " sped" (+0.48) | "bern" (-0.53) | "Chuck" (+0.56) | "ichick" (+0.58) |

The random controls are isotropic directions of **matched norm** (unit, as the singular vector is), decoded identically. Logit spread: ΔW direction σ = 0.1480, max 0.6723; controls σ = 0.1682, 0.1591, 0.1570, max 0.8111, 0.7056, 0.6573.

`W_U` is not isotropic; any direction pushed through it returns a list of tokens that can be narrativised. The control permits comparison of concentration between the ΔW direction and isotropic directions of matched norm.

On the one quantitative comparison available from these columns, the ΔW direction is **not** the more concentrated of the two: its logit spread is 0.1480 against a control median of 0.1591, and its largest logit is 0.6723 against 0.7056. It is flatter than isotropic noise, not sharper.

The measurement below reports where the tokens this experiment is actually about sit in that direction's own ranking of all 50257 of them, against a null of 64 isotropic directions:

| token | percentile under ΔW's dominant direction | null median | null 5–95% |
|---|---|---|---|
| " prolet" | 24.0 | 50.3 | 7.7 – 98.4 |
| " comrade" | 18.9 | 59.7 | 7.1 – 98.9 |
| " Divine" | 82.8 | 57.3 | 16.5 – 88.3 |

The loop went `prolet` → `comrade`. **All 3 of the tokens checked sit inside the null's 5–95% band**, including the one the flip landed on. The dominant direction of ΔW carries no measurable preference for the tokens whose basin it moved the loop into — the basin change is not legible in the weight change by logit lens.

**Issue #32 section 4**: `cos(ΔW_closed, ΔW_offline)` = 0.99294294 (recomputed-y arm), with norm ratio 0.972248.

Splitting the difference between the two updates: the component perpendicular to the closed-loop update is **0.1153** of `||ΔW_closed||`, the component along it is **0.0346** -- a ratio of 3.33 to 1 in favour of the perpendicular part. The cosine and the norm ratio are reported separately for this reason; a single mixed ratio cannot distinguish a direction change from a magnitude change.

## 5. Does the ΔW direction differ by basin? (issue #32 section 3b)

Dominant 768-side ΔW direction, one closed-loop episode per prompt, sign-fixed as above. The basin label is the prompt's **frozen-loop** basin (the attractor the episode ran in), from the 125-prompt baseline.

| | A01_physics<br>`prolet` | A02_medical<br>`prolet` | A04_climate<br>`prolet` | A14_kant<br>`Divine` | A08_linguistics<br>`Divine` |
|---|---|---|---|---|---|
| **A01_physics**<br>`prolet` | -- | +0.998721 | +0.998644 | +0.356371 | +0.466183 |
| **A02_medical**<br>`prolet` | +0.998721 | -- | +0.996716 | +0.363236 | +0.473248 |
| **A04_climate**<br>`prolet` | +0.998644 | +0.996716 | -- | +0.375584 | +0.485904 |
| **A14_kant**<br>`Divine` | +0.356371 | +0.363236 | +0.375584 | -- | +0.938737 |
| **A08_linguistics**<br>`Divine` | +0.466183 | +0.473248 | +0.485904 | +0.938737 | -- |

Within-basin median cosine +0.997680 (4 pairs); between-basin median +0.420884 (6 pairs).

**The directions separate by basin**, so ΔW carries a recoverable trace of which attractor the episode ran in.

Two features of this comparison: First, issue #32's two branches — "ΔW fingerprints the attractor" and "ΔW records the site's activation statistics and nothing about the episode" — are **not mutually exclusive here**, because the attractor is what determines the site's activation statistics. Section 1 showed the offline arm, which sees nothing but the frozen activation statistics, reproduces the closed arm's ΔW to cos = 0.99294. No measurement here isolates an episode-specific component from the site's activation statistics.

Second, prompts in the same basin are not a random sample — `A01`/`A02`/`A04` are all Complex-register academic sentences and the two `Divine` prompts are as well, so prompt similarity and basin membership are confounded in this design. Separating them needs the within/between measurement over many prompts per basin that issue #32 asks for and this run does not have.

## 6. C1 — does the state come back when the weights do?

After the episode, W0 is restored and the loop continues from the closed arm's final state. Horizon 1000 with early stop at `1 − cos` ≤ 1e-12, **not 200**: the measured per-iteration contraction is ~0.968, about 71 iterations per decade of displacement, so returning from a displacement of order 5.0e-03 to the round-off floor needs several hundred iterations and a 200-iteration horizon has already produced one false "failed to return" in this repo.

The reference is iterated in **lockstep**, not held fixed at the episode's last iteration. The frozen loop is itself still settling at iteration 120, so a state that has come all the way back onto the frozen trajectory still reads a nonzero gap against the iteration-120 snapshot. Both are reported: the lockstep gap is the C1 answer, the fixed one is what the naive version of this control would have said.

- start gap `1 − cos` = 5.000e-03
- returned (lockstep reference): **yes**, at iteration 230
- against a **fixed** iteration-120 target, the same run's final gap is 3.189e-05 — which is where a 200-iteration, fixed-target version of this control would have reported a failure to return
- basin after revert: `prolet` (margin 0.227); lockstep reference `prolet` (margin 0.227); fixed iteration-120 target `prolet` (margin 0.230)

Gap curve, lockstep: 4.747e-03 at iteration 1 → 1.322e-12 at 225. Fixed-target: 4.781e-03 → 3.189e-05.

**The contraction measured here is faster than the ~0.968 the horizon was justified against**: 0.9521 per iteration on this trajectory, about 47 iterations per decade of displacement rather than 71. The 1000-iteration horizon was not needed in the end -- it was chosen before the run, from the 0.968 figure.

## 7. What this does and does not establish

- **ΔW ≠ 0 was also produced by the offline arm**, which has no feedback path (section 1).
- **The ΔW magnitude follows from the chosen eta.** eta was chosen from the step-size map.
- **One prompt family, one site, one ceiling, 120 steps, cadence 1.** The map's caveats carry over unchanged.
- **The `recorded`-mode numbers are reported but not claimed from.** Their floor is a frozen-`y` artefact that has been measured larger than the routed signal.
- **No task, no loss, no target.** The design contains none of the three.
- **The offline arm flips too.** Section 3 reports a difference between the two arms' final weight matrices that is above the severed floor.
- **The severed control sets the floor for section 3.** In `recorded` mode the same protocol reports a larger measured difference with the feedback path severed. Only the `recomputed` path has a floor of literal zero.

## Provenance

13 cells, 44 CPU-minutes.

```json
{
  "cadence": 1,
  "device": "cpu",
  "dtype": "float32",
  "eta": 7.065171428571429e-05,
  "eta_provenance": "D * ||W0||_F / (N_STEPS * U_ref[hebb]) with D=1.8e-2, U_ref=350, ||W0||_F=164.854 -- the step-size map's own anchor, recomputed rather than copied from its rounded table entry",
  "finished": "2026-07-29T02:14:51Z",
  "issues": [
    26,
    30,
    32
  ],
  "layer_end": 11,
  "layer_end_severed": 3,
  "layer_start": 0,
  "max_delta_frac": 0.05,
  "mode": "hebb",
  "model": "gpt2-small",
  "n_steps": 120,
  "norms_dtype": "float64",
  "platform": "Linux 6.18.5 x86_64",
  "prompts_divine": [
    "A14_kant",
    "A08_linguistics"
  ],
  "prompts_prolet": [
    "A01_physics",
    "A02_medical",
    "A04_climate"
  ],
  "provenance_warning": "cells in this report were produced under more than one repo revision; see repo_revs",
  "python_version": "3.11.15",
  "repo_rev": "af8082dbcb18c4e856d4d432e5ae703848eced45",
  "repo_revs": [
    "af8082dbcb18c4e856d4d432e5ae703848eced45",
    "bf2719cf2c33e68438e8cbe70fcb1ae6ac59a2eb"
  ],
  "revert_horizon": 1000,
  "seeds": [
    0,
    1,
    2
  ],
  "shards": 2,
  "site": "blocks.6.mlp",
  "started": "2026-07-29T02:10:08Z",
  "torch_threads": 1,
  "torch_version": "2.13.0+cpu",
  "transformer_lens_version": "3.5.1",
  "wall_clock_seconds": 3181.7000000000003,
  "wall_clock_seconds_per_meta": [
    282.9,
    1779.4,
    1119.4
  ],
  "y_sources": [
    "recorded",
    "recomputed"
  ]
}
```

Raw per-cell records, including the full ΔW singular spectra, the 768-component dominant directions and every per-axis match: `experiments/output_exp001/exp001.jsonl`.
