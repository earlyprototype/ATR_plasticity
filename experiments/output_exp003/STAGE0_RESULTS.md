# EXP-003 Stage 0 results

*Measurements only. Thresholds were registered in `PREREGISTRATION.md` before the run.
Run artifact: `stage0.jsonl`. Analysis artifact: `stage0_analysis.json`. Nothing here
enters the claim register.*

## What was run

The 125 committed baseline inputs from `experiments/output_baseline/basins.jsonl`, each
run 120 iterations on the frozen model, with no plasticity and no injected signal.

At every iteration the activity of each of the 144 sites (12 blocks by 12 heads) was
recorded as the length of that head's write into the residual stream, together with each
block's MLP write. From those, the depth-weighted centroid was computed: the mean block
index, weighted by how much each block wrote.

The reported value per input is that centroid averaged over the last 25 iterations.

Run time 70 minutes on CPU.

## Population

| Group | Count |
|---|---|
| `prolet` | 55 |
| `Divine` | 34 |
| `till` | 19 |
| `Anarch` | 16 |
| `solidarity` | 1 |
| fixed point | 91 |
| two-step cycle | 34 |

## The statistic

Between-group spread divided by within-group spread. Registered reference points: 1.0
means groups are not distinguishable from their own internal scatter, and the token
labels score 0.87 on this scale, computed from register row C-07.

## Gates, as registered before the run

| Gate | Registered threshold | Measured | Result |
|---|---|---|---|
| 1. Separates the five end states | above 1.5 | **12.2035** | pass |
| 2. Separates fixed points from two-step cycles | above 1.5 | **1.0036** | **fail** |
| 3. Does not separate random halves of the largest group | below 1.2 in at least 9 of 10 splits | 10 of 10 below | pass |

## Controls

| Control | Registered expectation | Measured | Result |
|---|---|---|---|
| A. Permute block labels | scores below the true statistic | **0.3701** against 12.2035 | pass |
| B. Permute head labels | changes nothing | max deviation **1.78e-15** | pass |

Control B's registered threshold was 1e-9. The measured deviation is six orders of
magnitude below it.

## Three variants of the statistic

Variant 1 is the registered one. Variants 2 and 3 were added after the run and are
labelled as post-registration in `stage0_analysis.json`.

| Variant | End states | Class | Random halves |
|---|---|---|---|
| 1. Depth centroid, as registered | 12.2035 | 1.0036 | 10 of 10 below |
| 2. Same centroid, population mean removed and rescaled | 12.2035 | 1.0036 | 10 of 10 below |
| 3. Twelve-number per-block profile at settle, standardised per block, Euclidean distance | 1.4303 | 0.6195 | 10 of 10 below |

Variant 2 is identical to variant 1 by construction, because the statistic is a ratio and
is unchanged by rescaling a scalar. It is reported so the invariance is visible rather
than assumed.

Variant 3 is a vector rather than a scalar and uses the Euclidean distance measure of
Chao, Bakkum and Potter (2007). It is not their method, which compares whole trajectories
rather than an endpoint summary. The per-iteration activity needed for their method was
not persisted, so it cannot be computed from this run.

## Limits

One model. One loop configuration. No plasticity and no injected signal anywhere in this
stage. 120 iterations against the committed baseline's 300, chosen because that baseline
records agreement between its 120-iteration and 300-iteration readings on 124 of its 125
inputs.

The `solidarity` group has one member, so it contributes to the between-group term and
not to the within-group term.

Gate 2 failed. The registered consequence of a failed gate was that the statistic is
discarded. It was not discarded, and that decision was taken after seeing the result.

## Files

- `stage0.jsonl` — per-input records: the centroid at every iteration, the settled value,
  the per-block and per-site activity profiles at settle, and 10 permutations each of the
  two shuffle controls.
- `stage0_analysis.json` — every gate and control recomputed from those records.
- `experiments/exp003_stage0.py` — the runner.
- `experiments/exp003_stage0_analyse.py` — the analysis.
