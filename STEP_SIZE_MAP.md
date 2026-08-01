# Step-size map

Issue #30. Where, for each local rule, the weights actually move without the run measuring the norm ceiling instead of the rule.

## Configuration

| | |
|---|---|
| model | gpt2-small, frozen, CPU, float32 |
| site | `blocks.6.mlp` (= `transformer.h.6.mlp.c_proj`), (3072, 768) |
| ‖W0‖_F | 164.854073 (float64) |
| prompt | `A01_physics` — "The implications of quantum entanglement suggest that" |
| loop | layers 0→11, 120 steps, ‖x₀‖ = 1289.226318 |
| plasticity | cadence 1 (update after every step), `max_delta_frac` = 0.05, seed 0 |
| cells | 35 recorded of 33 = 1 frozen reference + 4×8 |
| eta anchor | `eta = D · ‖W0‖_F / (N · U_ref)`, U_ref = {"hebb": 350.0, "oja": 14000.0, "anti_hebb": 14000.0, "random": 14000.0} |
| threads | 1 per process, 2 shard(s) |

Only `mode` and `eta` vary between cells. Prompt, site, step count, seed, ceiling and the iteration-0 state tensor are shared.

## Frozen reference (`mode=off`)

Basin `prolet`, lag-1 1.00000, lag-2 1.00000, ‖W‖_F 164.854073 (unchanged), effective rank 642.64, pre-rescale ‖x‖ 4782.7781 against post-rescale 1289.2263.

`off` accumulates the statistics and applies nothing. Other cells are compared against this row, which is the C0 identity check on the instrument.

## Verdicts

### `hebb`

Geometric middle of the usable band 3.93e-05 … 0.000118 (3 cell(s)): **6.8e-05**. Nearest measured cell: 7.07e-05.

- Nothing happens at or below **3.93e-06** (relative weight change < 0.001).
- Ceiling audible at or above **0.00022** (lowest eta with a clip rate above 2%).
- No run went non-finite at any eta.
- No hollowing-out: effective rank never fell by more than 5% with ‖W‖_F flat.

### `oja`

Geometric middle of the usable band 9.81e-07 … 9.81e-06 (3 cell(s)): **3.1e-06**. Nearest measured cell: 2.94e-06.

- Nothing happens at or below **9.81e-08** (relative weight change < 0.001).
- Ceiling audible at or above **2.94e-05** (lowest eta with a clip rate above 2%).
- No run went non-finite at any eta.
- No hollowing-out: effective rank never fell by more than 5% with ‖W‖_F flat.

### `anti_hebb`

Geometric middle of the usable band 9.81e-07 … 2.94e-05 (4 cell(s)): **5.37e-06**. Nearest measured cell: 2.94e-06.

- Nothing happens at or below **9.81e-08** (relative weight change < 0.001).
- Ceiling audible at or above **9.81e-05** (lowest eta with a clip rate above 2%).
- No run went non-finite at any eta.
- No hollowing-out: effective rank never fell by more than 5% with ‖W‖_F flat.

### `random`

Geometric middle of the usable band 2.94e-06 … 2.94e-05 (3 cell(s)): **9.31e-06**. Nearest measured cell: 9.81e-06.

- Nothing happens at or below **9.81e-07** (relative weight change < 0.001).
- Ceiling audible at or above **9.81e-05** (lowest eta with a clip rate above 2%).
- No run went non-finite at any eta.
- No hollowing-out: effective rank never fell by more than 5% with ‖W‖_F flat.

## What the map says

### 1. Three of the four rules have a wide band, and nothing happens in it

- `oja` at its largest ceiling-silent eta (9.81e-06, clip **0.0%**) moves the weights 2.92% of ‖W0‖_F and the loop does not move: basin `prolet` (frozen: `prolet`), lag-1 1.00000 (frozen 1.00000), cos(final, frozen) = 0.999222.
- `anti_hebb` at its largest ceiling-silent eta (2.94e-05, clip **0.0%**) moves the weights 4.58% of ‖W0‖_F and the loop does not move: basin `prolet` (frozen: `prolet`), lag-1 1.00000 (frozen 1.00000), cos(final, frozen) = 0.995287.
- Pushed all the way to the ceiling — `oja` at 0.000981, clip **100.0%**, the full 5.0% of drift the ceiling allows — the basin is still `prolet` and lag-1 is still 1.00000. That holds at every `oja` eta tested.

### 2. `hebb`: basin change inside the ceiling-silent band

The basin changes at eta 7.07e-05 — `prolet` → `comrade` — at 1.12% relative weight change with the ceiling **silent** (0.0%), cos(final, frozen) = 0.995000.

`hebb` has no decay term. Its band spans a factor of three between the first cell that changes the loop and the first cell that clips, and it is bounded above only by `max_delta_frac`.

### 3. Norm-matched noise control

`random` is norm-matched to what Oja would have applied. Matched not by eta — the noise re-randomises every step and accumulates as a random walk rather than coherently — but by the relative weight change actually reached:

| arm | eta | rel ΔW | clip | loop |
|---|---|---|---|---|
| `random` | 2.94e-05 | 1.84% | 0.0% | no |
| `oja` | 9.81e-06 | 2.92% | 0.0% | cos(final,frozen)=0.999222 |
| `anti_hebb` | 9.81e-06 | 2.71% | 0.0% | cos(final,frozen)=0.998670 |
| `hebb` | 0.000118 | 2.20% | 0.0% | basin 'prolet'->'comrade'; cos(final,frozen)=0.982135 |

At eta 2.94e-05 `random` reaches 1.84% relative weight change with no change in basin, lag-1 or cos(final, frozen), against `oja` 2.92% and `hebb` 2.20% at their own band cells. At `oja`'s 9.81e-06 cell the basin and lag-1 are unchanged and cos(final, frozen) = 0.999222.

### 4. Pre-rescale activation norm across the sweep

Issue #27 item 3's signature is a pre-rescale activation norm that moves while the loop's visible behaviour stays flat. That is not what these cells show.

The frozen loop already runs at pre/post = 3.7098 — the rescaling divides by 3.71 on every step whether or not plasticity is on. Across all 35 cells the plasticity moves that ratio by at most 3.5%.

- `oja` at 9.81e-06: pre-rescale ratio -0.34% against frozen, cos(final, frozen) = 0.999222. Both flat.
- `hebb` at 0.000118: pre-rescale ratio +2.05% against frozen, cos(final, frozen) = 0.982135. Both move.

### 5. No hollowing out anywhere in the sweep

**Corrected 2026-08-01.** This section used to bound the effect with two cells in which the norm ceiling fired, at 99.2% and 60.8% clip. That breaks this repository's standing prohibition against quoting a ceiling-fired cell as evidence, since such a cell measures the ceiling rather than the rule. The conclusion is unchanged and the ceiling-silent cells carry it on their own; the numbers below are those cells only. The two withdrawn figures were an effective rank of 640.5 (`hebb` at eta 0.0393, 99.2% clip) and of 647.3 (`anti_hebb` at eta 9.81e-05, 60.8% clip). Both remain in the full table as diagnostics.

Effective rank starts at 642.6 of a possible 768. Across the 23 cells in which the ceiling never fired, it never falls below 642.4 (`hebb` at eta 1.18e-04, a fall of 0.03%). Under `anti_hebb` it *rises*, reaching 646.7 at eta 2.94e-05 with the ceiling silent at 0.0% clip and drift at 4.58%, so the decay term flattens the spectrum, which is the opposite direction from rank-1 collapse. Max/mean |W| falls from 33.4 to 31.7, the latter also on a ceiling-silent cell.

The ΔW effective-rank and top-0.1%-mass columns are reported for issue #27 item 11. Oja's accumulated update is near rank-1 (effective rank 2.2), while the noise arm's is isotropic (718.8). But Oja's mass is not concentrated in a handful of *entries*: its top 0.1% of entries hold 0.0263 of the total absolute mass against the noise arm's 0.0044.

### What this does and does not rule out

| issue #27 | status |
|---|---|
| 2 — no interesting middle | **Ruled out as a confound.** Every mode has a band where the weights move with the ceiling silent. For `oja`/`anti_hebb`/`random` nothing happens inside it; for `hebb` something does. |
| 3 — we measure the rescaling | **Ruled out here.** The pre-rescale norm is flat wherever the loop is flat, so the homeostat is not absorbing a hidden weight effect. |
| 11 — norm ceiling and rescaling destroy each other | **Not observed on the ceiling-silent cells**, which are the ones this can be read from: effective rank flat or rising across all 23 of them, max entry falling, ΔW mass spread rather than concentrated. The ceiling-fired cells do show small falls (640.5 and 641.4 against a frozen 642.6) and are kept in the full table as diagnostics, not as evidence. See §5. |
| 1 — the rule moves the weights and nothing else happens | **Consistent with, not established.** That claim needs the offline arm (#26); this map only shows the loop-on side. |
| 5 — collapse is already the default | Untouched. This prompt is a fixed point under the frozen loop and stays one. |
| 7 — depth | Untouched. |

### Caveats

One prompt (`A01_physics`), one site (`blocks.6.mlp`), one seed, 120 steps, cadence 1, one ceiling (0.05). The etas quoted above are calibrated for exactly that configuration; a different site has a different ‖W0‖_F and different activation scale, and the anchoring formula has to be re-measured rather than reused. `random` here is a within-cell control, not the full C2. The bands are located to grid resolution — roughly half a decade, and a factor of three for `hebb` after refinement — not to a sharp edge.

## Full table

`clip` is the fraction of the 120 updates the norm ceiling scaled down; it is reported on every row. `erank` is the participation ratio of W's singular values (768 max). `pre/post` is the pre-rescale activation norm over the post-rescale one — the loop's homeostat is the denominator and holds it at ‖x₀‖ exactly.

| mode | D | eta | rel ΔW | ‖W‖_F | clip | nonfin | erank | Δerank | max/mean |W| | pre/post | basin | lag1 | lag2 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `off` | -- | 0 | 0.000e+00 | 164.8541 | 0.0% | 0 | 642.6 | -0.00% | 33.4 | 3.7098 | `prolet` | 1.00000 | 1.00000 | noise floor |
| `hebb` | 1e-04 | 3.93e-07 | 4.906e-05 | 164.8543 | 0.0% | 0 | 642.6 | -0.00% | 33.4 | 3.7099 | `prolet` | 1.00000 | 1.00000 | noise floor |
| `hebb` | 1e-03 | 3.93e-06 | 4.964e-04 | 164.8561 | 0.0% | 0 | 642.6 | -0.00% | 33.4 | 3.7113 | `prolet` | 1.00000 | 1.00000 | noise floor |
| `hebb` | 1e-02 | 3.93e-05 | 5.599e-03 | 164.8785 | 0.0% | 0 | 642.6 | -0.00% | 33.5 | 3.7271 | `prolet` | 1.00000 | 0.99999 | usable |
| `hebb` | 2e-02 | 7.07e-05 | 1.124e-02 | 164.9072 | 0.0% | 0 | 642.6 | -0.01% | 33.5 | 3.7465 | `comrade` | 1.00000 | 0.99999 | usable |
| `hebb` | 3e-02 | 0.000118 | 2.204e-02 | 164.9743 | 0.0% | 0 | 642.4 | -0.03% | 33.6 | 3.7858 | `comrade` | 0.99999 | 0.99997 | usable |
| `hebb` | 6e-02 | 0.00022 | 5.000e-02 | 165.2201 | 7.5% | 0 | 641.7 | -0.15% | 33.8 | 3.8387 | `locality` | 0.99998 | 0.99993 | ceiling |
| `hebb` | 1e-01 | 0.000393 | 5.000e-02 | 165.1871 | 43.3% | 0 | 641.7 | -0.14% | 33.5 | 3.8340 | `locality` | 0.99999 | 0.99997 | ceiling |
| `hebb` | 3e-01 | 0.00118 | 5.000e-02 | 165.1909 | 83.3% | 0 | 641.6 | -0.16% | 33.4 | 3.8102 | `comrade` | 0.99999 | 0.99996 | ceiling |
| `hebb` | 1e+00 | 0.00393 | 5.000e-02 | 165.2062 | 95.8% | 0 | 641.4 | -0.20% | 33.8 | 3.7652 | `comrade` | 0.99999 | 0.99997 | ceiling |
| `hebb` | 1e+01 | 0.0393 | 5.000e-02 | 165.2057 | 99.2% | 0 | 641.4 | -0.20% | 33.7 | 3.7663 | `comrade` | 1.00000 | 1.00000 | ceiling |
| `oja` | 1e-04 | 9.81e-09 | 4.757e-05 | 164.8531 | 0.0% | 0 | 642.6 | +0.00% | 33.4 | 3.7098 | `prolet` | 1.00000 | 1.00000 | noise floor |
| `oja` | 1e-03 | 9.81e-08 | 4.729e-04 | 164.8446 | 0.0% | 0 | 642.7 | +0.01% | 33.4 | 3.7096 | `prolet` | 1.00000 | 1.00000 | noise floor |
| `oja` | 1e-02 | 9.81e-07 | 4.461e-03 | 164.7655 | 0.0% | 0 | 643.1 | +0.07% | 33.1 | 3.7080 | `prolet` | 1.00000 | 1.00000 | usable |
| `oja` | 3e-02 | 2.94e-06 | 1.192e-02 | 164.6204 | 0.0% | 0 | 643.8 | +0.17% | 32.7 | 3.7050 | `prolet` | 1.00000 | 1.00000 | usable |
| `oja` | 1e-01 | 9.81e-06 | 2.925e-02 | 164.2899 | 0.0% | 0 | 645.2 | +0.40% | 31.7 | 3.6974 | `prolet` | 1.00000 | 1.00000 | usable |
| `oja` | 3e-01 | 2.94e-05 | 5.000e-02 | 163.8574 | 9.2% | 0 | 646.8 | +0.65% | 31.7 | 3.6855 | `prolet` | 1.00000 | 1.00000 | ceiling |
| `oja` | 1e+00 | 9.81e-05 | 5.000e-02 | 163.9809 | 65.8% | 0 | 646.4 | +0.59% | 31.8 | 3.6886 | `prolet` | 1.00000 | 0.99999 | ceiling |
| `oja` | 1e+01 | 0.000981 | 5.000e-02 | 164.2742 | 100.0% | 0 | 645.4 | +0.42% | 31.8 | 3.6974 | `prolet` | 1.00000 | 0.99999 | ceiling |
| `anti_hebb` | 1e-04 | 9.81e-09 | 4.800e-05 | 164.8531 | 0.0% | 0 | 642.6 | +0.00% | 33.4 | 3.7098 | `prolet` | 1.00000 | 1.00000 | noise floor |
| `anti_hebb` | 1e-03 | 9.81e-08 | 4.764e-04 | 164.8445 | 0.0% | 0 | 642.7 | +0.01% | 33.4 | 3.7095 | `prolet` | 1.00000 | 1.00000 | noise floor |
| `anti_hebb` | 1e-02 | 9.81e-07 | 4.441e-03 | 164.7656 | 0.0% | 0 | 643.1 | +0.07% | 33.1 | 3.7074 | `prolet` | 1.00000 | 1.00000 | usable |
| `anti_hebb` | 3e-02 | 2.94e-06 | 1.160e-02 | 164.6244 | 0.0% | 0 | 643.7 | +0.17% | 32.7 | 3.7033 | `prolet` | 1.00000 | 1.00000 | usable |
| `anti_hebb` | 1e-01 | 9.81e-06 | 2.706e-02 | 164.3145 | 0.0% | 0 | 645.1 | +0.38% | 31.7 | 3.6933 | `prolet` | 1.00000 | 1.00000 | usable |
| `anti_hebb` | 3e-01 | 2.94e-05 | 4.585e-02 | 163.8705 | 0.0% | 0 | 646.7 | +0.63% | 31.7 | 3.6775 | `prolet` | 1.00000 | 1.00000 | usable |
| `anti_hebb` | 1e+00 | 9.81e-05 | 5.000e-02 | 163.7065 | 60.8% | 0 | 647.3 | +0.72% | 31.7 | 3.6715 | `prolet` | 1.00000 | 1.00000 | ceiling |
| `anti_hebb` | 1e+01 | 0.000981 | 5.000e-02 | 163.8923 | 100.0% | 0 | 646.3 | +0.57% | 31.8 | 3.6584 | `anarchism` | 1.00000 | 0.99999 | ceiling |
| `random` | 1e-04 | 9.81e-09 | 6.116e-06 | 164.8541 | 0.0% | 0 | 642.6 | +0.00% | 33.4 | 3.7098 | `prolet` | 1.00000 | 1.00000 | noise floor |
| `random` | 1e-03 | 9.81e-08 | 6.116e-05 | 164.8541 | 0.0% | 0 | 642.6 | +0.00% | 33.4 | 3.7098 | `prolet` | 1.00000 | 1.00000 | noise floor |
| `random` | 1e-02 | 9.81e-07 | 6.116e-04 | 164.8541 | 0.0% | 0 | 642.6 | +0.00% | 33.4 | 3.7098 | `prolet` | 1.00000 | 1.00000 | noise floor |
| `random` | 3e-02 | 2.94e-06 | 1.835e-03 | 164.8544 | 0.0% | 0 | 642.6 | +0.00% | 33.4 | 3.7098 | `prolet` | 1.00000 | 1.00000 | usable |
| `random` | 1e-01 | 9.81e-06 | 6.118e-03 | 164.8573 | 0.0% | 0 | 642.7 | +0.00% | 33.4 | 3.7099 | `prolet` | 1.00000 | 1.00000 | usable |
| `random` | 3e-01 | 2.94e-05 | 1.836e-02 | 164.8824 | 0.0% | 0 | 642.7 | +0.01% | 33.3 | 3.7100 | `prolet` | 1.00000 | 1.00000 | usable |
| `random` | 1e+00 | 9.81e-05 | 5.000e-02 | 165.0613 | 27.5% | 0 | 643.1 | +0.08% | 33.2 | 3.7103 | `prolet` | 1.00000 | 1.00000 | ceiling |
| `random` | 1e+01 | 0.000981 | 5.000e-02 | 165.0546 | 100.0% | 0 | 643.2 | +0.08% | 33.3 | 3.7102 | `prolet` | 1.00000 | 1.00000 | ceiling |

## Did the loop's behaviour change

Against the frozen `off` cell. `cos(final,frozen)` is between the position-mean residual vectors at the last iteration.

| mode | eta | rel ΔW | clip | changed |
|---|---|---|---|---|
| `hebb` | 3.93e-07 | 4.91e-05 | 0.0% | no |
| `hebb` | 3.93e-06 | 4.96e-04 | 0.0% | cos(final,frozen)=0.999991 |
| `hebb` | 3.93e-05 | 5.60e-03 | 0.0% | cos(final,frozen)=0.998771 |
| `hebb` | 7.07e-05 | 1.12e-02 | 0.0% | basin 'prolet'->'comrade'; cos(final,frozen)=0.995000 |
| `hebb` | 0.000118 | 2.20e-02 | 0.0% | basin 'prolet'->'comrade'; cos(final,frozen)=0.982135 |
| `hebb` | 0.00022 | 5.00e-02 | 7.5% | basin 'prolet'->'locality'; lag1 1.00000->0.99998; cos(final,frozen)=0.936008 |
| `hebb` | 0.000393 | 5.00e-02 | 43.3% | basin 'prolet'->'locality'; cos(final,frozen)=0.932387 |
| `hebb` | 0.00118 | 5.00e-02 | 83.3% | basin 'prolet'->'comrade'; cos(final,frozen)=0.967701 |
| `hebb` | 0.00393 | 5.00e-02 | 95.8% | basin 'prolet'->'comrade'; cos(final,frozen)=0.978233 |
| `hebb` | 0.0393 | 5.00e-02 | 99.2% | basin 'prolet'->'comrade'; cos(final,frozen)=0.979436 |
| `oja` | 9.81e-09 | 4.76e-05 | 0.0% | no |
| `oja` | 9.81e-08 | 4.73e-04 | 0.0% | no |
| `oja` | 9.81e-07 | 4.46e-03 | 0.0% | cos(final,frozen)=0.999984 |
| `oja` | 2.94e-06 | 1.19e-02 | 0.0% | cos(final,frozen)=0.999880 |
| `oja` | 9.81e-06 | 2.92e-02 | 0.0% | cos(final,frozen)=0.999222 |
| `oja` | 2.94e-05 | 5.00e-02 | 9.2% | cos(final,frozen)=0.997276 |
| `oja` | 9.81e-05 | 5.00e-02 | 65.8% | cos(final,frozen)=0.997371 |
| `oja` | 0.000981 | 5.00e-02 | 100.0% | cos(final,frozen)=0.998290 |
| `anti_hebb` | 9.81e-09 | 4.80e-05 | 0.0% | no |
| `anti_hebb` | 9.81e-08 | 4.76e-04 | 0.0% | no |
| `anti_hebb` | 9.81e-07 | 4.44e-03 | 0.0% | cos(final,frozen)=0.999971 |
| `anti_hebb` | 2.94e-06 | 1.16e-02 | 0.0% | cos(final,frozen)=0.999788 |
| `anti_hebb` | 9.81e-06 | 2.71e-02 | 0.0% | cos(final,frozen)=0.998670 |
| `anti_hebb` | 2.94e-05 | 4.58e-02 | 0.0% | cos(final,frozen)=0.995287 |
| `anti_hebb` | 9.81e-05 | 5.00e-02 | 60.8% | cos(final,frozen)=0.993005 |
| `anti_hebb` | 0.000981 | 5.00e-02 | 100.0% | basin 'prolet'->'anarchism'; cos(final,frozen)=0.986590 |
| `random` | 9.81e-09 | 6.12e-06 | 0.0% | no |
| `random` | 9.81e-08 | 6.12e-05 | 0.0% | no |
| `random` | 9.81e-07 | 6.12e-04 | 0.0% | no |
| `random` | 2.94e-06 | 1.83e-03 | 0.0% | no |
| `random` | 9.81e-06 | 6.12e-03 | 0.0% | no |
| `random` | 2.94e-05 | 1.84e-02 | 0.0% | no |
| `random` | 9.81e-05 | 5.00e-02 | 27.5% | cos(final,frozen)=0.999995 |
| `random` | 0.000981 | 5.00e-02 | 100.0% | cos(final,frozen)=0.999997 |

## The homeostat

The ATR loop rescales the state to ‖x₀‖ before every injection, so the post-rescale norm is a constant 1289.2263 on every cell by construction. The pre-rescale norm is what the forward pass actually produced. A pre-rescale norm that moves while the loop's visible behaviour does not is the signature issue #27 item 3 describes.

| mode | eta | pre-rescale (first → last) | max | pre/post at end | loop changed |
|---|---|---|---|---|---|
| `off` | 0 | 3553.29 → 4782.78 | 4927.10 | 3.70980 | no |
| `hebb` | 3.93e-07 | 3553.29 → 4782.96 | 4927.11 | 3.70995 | no |
| `hebb` | 3.93e-06 | 3553.29 → 4784.64 | 4927.20 | 3.71125 | cos(final,frozen)=0.999991 |
| `hebb` | 3.93e-05 | 3553.29 → 4805.09 | 4928.02 | 3.72711 | cos(final,frozen)=0.998771 |
| `hebb` | 7.07e-05 | 3553.29 → 4830.14 | 4930.55 | 3.74654 | basin 'prolet'->'comrade'; cos(final,frozen)=0.995000 |
| `hebb` | 0.000118 | 3553.29 → 4880.69 | 4933.89 | 3.78575 | basin 'prolet'->'comrade'; cos(final,frozen)=0.982135 |
| `hebb` | 0.00022 | 3553.29 → 4949.02 | 4949.30 | 3.83875 | basin 'prolet'->'locality'; lag1 1.00000->0.99998; cos(final,frozen)=0.936008 |
| `hebb` | 0.000393 | 3553.29 → 4942.93 | 4946.30 | 3.83403 | basin 'prolet'->'locality'; cos(final,frozen)=0.932387 |
| `hebb` | 0.00118 | 3553.29 → 4912.26 | 4934.84 | 3.81024 | basin 'prolet'->'comrade'; cos(final,frozen)=0.967701 |
| `hebb` | 0.00393 | 3553.29 → 4854.24 | 5060.93 | 3.76523 | basin 'prolet'->'comrade'; cos(final,frozen)=0.978233 |
| `hebb` | 0.0393 | 3553.29 → 4855.55 | 5028.34 | 3.76625 | basin 'prolet'->'comrade'; cos(final,frozen)=0.979436 |
| `oja` | 9.81e-09 | 3553.29 → 4782.75 | 4927.10 | 3.70979 | no |
| `oja` | 9.81e-08 | 3553.29 → 4782.54 | 4927.08 | 3.70962 | no |
| `oja` | 9.81e-07 | 3553.29 → 4780.49 | 4926.96 | 3.70803 | cos(final,frozen)=0.999984 |
| `oja` | 2.94e-06 | 3553.29 → 4776.55 | 4926.70 | 3.70498 | cos(final,frozen)=0.999880 |
| `oja` | 9.81e-06 | 3553.29 → 4766.75 | 4925.73 | 3.69738 | cos(final,frozen)=0.999222 |
| `oja` | 2.94e-05 | 3553.29 → 4751.48 | 4922.66 | 3.68553 | cos(final,frozen)=0.997276 |
| `oja` | 9.81e-05 | 3553.29 → 4755.46 | 4910.95 | 3.68861 | cos(final,frozen)=0.997371 |
| `oja` | 0.000981 | 3553.29 → 4766.76 | 4909.26 | 3.69738 | cos(final,frozen)=0.998290 |
| `anti_hebb` | 9.81e-09 | 3553.29 → 4782.75 | 4927.10 | 3.70978 | no |
| `anti_hebb` | 9.81e-08 | 3553.29 → 4782.45 | 4927.08 | 3.70955 | no |
| `anti_hebb` | 9.81e-07 | 3553.29 → 4779.65 | 4926.91 | 3.70738 | cos(final,frozen)=0.999971 |
| `anti_hebb` | 2.94e-06 | 3553.29 → 4774.37 | 4926.52 | 3.70328 | cos(final,frozen)=0.999788 |
| `anti_hebb` | 9.81e-06 | 3553.29 → 4761.53 | 4925.03 | 3.69332 | cos(final,frozen)=0.998670 |
| `anti_hebb` | 2.94e-05 | 3553.29 → 4741.17 | 4919.59 | 3.67753 | cos(final,frozen)=0.995287 |
| `anti_hebb` | 9.81e-05 | 3553.29 → 4733.39 | 4911.41 | 3.67150 | cos(final,frozen)=0.993005 |
| `anti_hebb` | 0.000981 | 3553.29 → 4716.47 | 4909.68 | 3.65837 | basin 'prolet'->'anarchism'; cos(final,frozen)=0.986590 |
| `random` | 9.81e-09 | 3553.29 → 4782.78 | 4927.10 | 3.70980 | no |
| `random` | 9.81e-08 | 3553.29 → 4782.78 | 4927.10 | 3.70981 | no |
| `random` | 9.81e-07 | 3553.29 → 4782.79 | 4927.11 | 3.70981 | no |
| `random` | 2.94e-06 | 3553.29 → 4782.80 | 4927.14 | 3.70983 | no |
| `random` | 9.81e-06 | 3553.29 → 4782.86 | 4927.25 | 3.70987 | no |
| `random` | 2.94e-05 | 3553.29 → 4783.04 | 4927.57 | 3.71001 | no |
| `random` | 9.81e-05 | 3553.29 → 4783.44 | 4928.72 | 3.71032 | cos(final,frozen)=0.999995 |
| `random` | 0.000981 | 3553.29 → 4783.33 | 4930.39 | 3.71023 | cos(final,frozen)=0.999997 |

## Hollowing out (issue #27 item 11)

The failure where one entry runs away, the normaliser rescales, and the rest of the matrix is annihilated. ‖W‖_F stays flat and the clipping rate stays low throughout, so neither of those registers it. **Effective rank falling while ‖W‖_F is flat is the signature this section tests for.** `top 0.1% mass` is reported alongside ΔW effective rank so a near-rank-1 update can be distinguished from a handful of entries dominating.

With `max_delta_frac` = 0.05, ‖delta‖_F cannot exceed 5% of ‖W0‖_F, so ‖W‖_F **structurally cannot run away** — its flatness is guaranteed by the same mechanism that would be doing the damage, and it therefore carries no information about whether the damage happened. The flatness threshold here is set at 6% (just above what the ceiling already promises), and effective rank, not ‖W‖_F, is the instrument.

| mode | eta | ‖W‖_F range | erank first → last | min erank | σ₁ energy | max/mean |W| | top 0.1% mass | ΔW erank | ΔW top 0.1% mass |
|---|---|---|---|---|---|---|---|---|---|
| `off` | 0 | 0.00e+00 | 642.6 → 642.6 | 642.6 | 0.0323 | 33.38 | 0.0079 | -- | -- |
| `hebb` | 3.93e-07 | 1.20e-06 | 642.6 → 642.6 | 642.6 | 0.0323 | 33.38 | 0.0079 | 2.2 | 0.0247 |
| `hebb` | 3.93e-06 | 1.22e-05 | 642.6 → 642.6 | 642.6 | 0.0323 | 33.39 | 0.0079 | 2.2 | 0.0248 |
| `hebb` | 3.93e-05 | 1.48e-04 | 642.6 → 642.6 | 642.6 | 0.0323 | 33.46 | 0.0079 | 2.1 | 0.0251 |
| `hebb` | 7.07e-05 | 3.22e-04 | 642.6 → 642.6 | 642.6 | 0.0324 | 33.53 | 0.0079 | 2.0 | 0.0253 |
| `hebb` | 0.000118 | 7.30e-04 | 642.6 → 642.4 | 642.4 | 0.0325 | 33.64 | 0.0079 | 1.9 | 0.0253 |
| `hebb` | 0.00022 | 2.26e-03 | 642.6 → 641.7 | 641.7 | 0.0330 | 33.76 | 0.0079 | 1.7 | 0.0249 |
| `hebb` | 0.000393 | 2.32e-03 | 642.6 → 641.7 | 641.7 | 0.0329 | 33.49 | 0.0079 | 1.3 | 0.0254 |
| `hebb` | 0.00118 | 2.38e-03 | 642.6 → 641.6 | 641.6 | 0.0334 | 33.42 | 0.0080 | 1.0 | 0.0293 |
| `hebb` | 0.00393 | 2.51e-03 | 642.6 → 641.4 | 641.4 | 0.0340 | 33.77 | 0.0080 | 1.0 | 0.0359 |
| `hebb` | 0.0393 | 3.09e-03 | 642.6 → 641.4 | 640.5 | 0.0340 | 33.75 | 0.0080 | 1.0 | 0.0359 |
| `oja` | 9.81e-09 | 5.78e-06 | 642.6 → 642.6 | 642.6 | 0.0323 | 33.38 | 0.0079 | 1.8 | 0.0265 |
| `oja` | 9.81e-08 | 5.74e-05 | 642.6 → 642.7 | 642.6 | 0.0322 | 33.35 | 0.0079 | 1.8 | 0.0265 |
| `oja` | 9.81e-07 | 5.37e-04 | 642.6 → 643.1 | 642.6 | 0.0316 | 33.12 | 0.0079 | 1.8 | 0.0265 |
| `oja` | 2.94e-06 | 1.42e-03 | 642.6 → 643.8 | 642.6 | 0.0305 | 32.66 | 0.0078 | 1.9 | 0.0264 |
| `oja` | 9.81e-06 | 3.42e-03 | 642.6 → 645.2 | 642.6 | 0.0282 | 31.74 | 0.0076 | 2.2 | 0.0263 |
| `oja` | 2.94e-05 | 6.15e-03 | 642.6 → 646.8 | 642.6 | 0.0258 | 31.74 | 0.0073 | 2.9 | 0.0267 |
| `oja` | 9.81e-05 | 8.10e-03 | 642.6 → 646.4 | 642.6 | 0.0262 | 31.75 | 0.0074 | 2.2 | 0.0261 |
| `oja` | 0.000981 | 7.78e-03 | 642.6 → 645.4 | 642.6 | 0.0272 | 31.79 | 0.0075 | 1.0 | 0.0246 |
| `anti_hebb` | 9.81e-09 | 5.84e-06 | 642.6 → 642.6 | 642.6 | 0.0323 | 33.38 | 0.0079 | 1.8 | 0.0264 |
| `anti_hebb` | 9.81e-08 | 5.80e-05 | 642.6 → 642.7 | 642.6 | 0.0322 | 33.35 | 0.0079 | 1.8 | 0.0264 |
| `anti_hebb` | 9.81e-07 | 5.37e-04 | 642.6 → 643.1 | 642.6 | 0.0316 | 33.12 | 0.0079 | 1.8 | 0.0263 |
| `anti_hebb` | 2.94e-06 | 1.39e-03 | 642.6 → 643.7 | 642.6 | 0.0305 | 32.66 | 0.0078 | 2.0 | 0.0262 |
| `anti_hebb` | 9.81e-06 | 3.27e-03 | 642.6 → 645.1 | 642.6 | 0.0284 | 31.75 | 0.0076 | 2.4 | 0.0262 |
| `anti_hebb` | 2.94e-05 | 5.97e-03 | 642.6 → 646.7 | 642.6 | 0.0260 | 31.75 | 0.0074 | 3.2 | 0.0271 |
| `anti_hebb` | 9.81e-05 | 8.38e-03 | 642.6 → 647.3 | 642.6 | 0.0253 | 31.73 | 0.0073 | 3.8 | 0.0283 |
| `anti_hebb` | 0.000981 | 7.80e-03 | 642.6 → 646.3 | 642.6 | 0.0269 | 31.75 | 0.0074 | 3.3 | 0.0234 |
| `random` | 9.81e-09 | 3.33e-09 | 642.6 → 642.6 | 642.6 | 0.0323 | 33.38 | 0.0079 | 718.8 | 0.0044 |
| `random` | 9.81e-08 | 3.24e-08 | 642.6 → 642.6 | 642.6 | 0.0323 | 33.38 | 0.0079 | 718.8 | 0.0044 |
| `random` | 9.81e-07 | 4.17e-07 | 642.6 → 642.6 | 642.6 | 0.0323 | 33.38 | 0.0079 | 718.8 | 0.0044 |
| `random` | 2.94e-06 | 2.17e-06 | 642.6 → 642.6 | 642.6 | 0.0323 | 33.38 | 0.0079 | 718.8 | 0.0044 |
| `random` | 9.81e-06 | 2.02e-05 | 642.6 → 642.7 | 642.6 | 0.0323 | 33.37 | 0.0079 | 718.8 | 0.0044 |
| `random` | 2.94e-05 | 1.72e-04 | 642.6 → 642.7 | 642.6 | 0.0323 | 33.34 | 0.0079 | 718.8 | 0.0044 |
| `random` | 9.81e-05 | 1.27e-03 | 642.6 → 643.1 | 642.6 | 0.0322 | 33.24 | 0.0079 | 718.9 | 0.0044 |
| `random` | 0.000981 | 1.30e-03 | 642.6 → 643.2 | 642.6 | 0.0322 | 33.25 | 0.0079 | 718.8 | 0.0045 |

## Thresholds

Stated so that the verdicts above can be recomputed or disagreed with.

- noise floor: relative weight change < 0.001
- ceiling quiet: clip rate ≤ 0.02; ceiling loud: > 0.05
- hollowing-out flag: effective rank down > 5% while ‖W‖_F range < 6%

## Provenance

35 cells, 59 CPU-minutes total, run as 2 single-threaded shard(s) alongside another sweep on the same 4-core box. `wall_clock_seconds` below is the last invocation's only (the refinement pass), not the whole map's.

```json
{
  "cadence": 1,
  "d_grid": [
    0.0001,
    0.001,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    10.0
  ],
  "device": "cpu",
  "dtype": "float32",
  "finished": "2026-07-29T00:23:14Z",
  "issue": 30,
  "layer_end": 11,
  "layer_start": 0,
  "max_delta_frac": 0.05,
  "model": "gpt2-small",
  "n_steps": 120,
  "norms_dtype": "float64",
  "platform": "Linux 6.18.5 x86_64",
  "prompt_id": "A01_physics",
  "python_version": "3.11.15",
  "repo_rev": "c4f51240b77d1eeebaf048768970df2e506c10d6",
  "seed": 0,
  "shards": 2,
  "site": "blocks.6.mlp",
  "started": "2026-07-29T00:22:21Z",
  "torch_threads": 1,
  "torch_version": "2.13.0+cpu",
  "transformer_lens_version": "3.5.1",
  "u_ref": {
    "anti_hebb": 14000.0,
    "hebb": 350.0,
    "oja": 14000.0,
    "random": 14000.0
  },
  "wall_clock_seconds": 53.2
}
```

Raw per-cell records, including the full ‖W‖_F, delta-frac, update-norm, pre-rescale and singular-spectrum trajectories: `experiments/output_step_size/step_size_map.jsonl`.
