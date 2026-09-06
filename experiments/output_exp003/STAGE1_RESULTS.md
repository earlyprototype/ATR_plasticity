# EXP-003 Stage 1 results

*Measurements only. Thresholds were registered in `PREREGISTRATION.md` before the run.
Run artifact: `stage1.jsonl`.*

> **This file used to say that nothing in it entered the claim register. That has changed.**
> This stage's result is register row **C-66**: the mechanism is **refuted** against the
> threshold registered before the run. It refutes one named explanation for the distributed
> collapse and does not explain that collapse or weaken C-62. `CLAIMS.md` is the authority;
> where it and this file disagree, it wins.

## What was run

EXP-002's reinforcing (`hebb`) closed-loop episode, reproduced rather than approximated:
the same driven input `A01_physics`, the same 12 MLP sites, the same 120 adjustments at
every iteration, the same lifted ceiling, and the per-block step sizes taken verbatim from
that experiment's committed record.

Before and after the episode, two quantities were measured on each of the twelve adjusted
matrices:

- **Effective rank**, the participation-ratio measure `experiments/step_size_map.py`
  already uses.
- **Top singular share**, the fraction of total spectral weight held by the largest
  singular value.

## Reproduction gate

| Quantity | EXP-002 committed | This run | Result |
|---|---|---|---|
| Aggregate drift | 0.013111766434820447 | 0.013111766434820447 | exact match |
| Relative miss | tolerance 5% | **0.0** | pass |
| Clipped | — | false | |
| Non-finite | — | false | |

## The measurement

| Quantity | Frozen | After the episode | Change |
|---|---|---|---|
| Mean effective rank | 639.4554 | 639.1740 | **fall of 0.044%** |
| Mean top singular share | 0.007766 | 0.007797 | rise of 0.4% |

## Verdict against the registered thresholds

| Registered rule | Measured | Verdict |
|---|---|---|
| Supported if mean effective rank falls by 10% or more | 0.044% | |
| Refuted if it falls by less than 2% | 0.044% | **refuted** |
| Inconclusive between the two | — | |

The 10% threshold was registered against a stated reference: the largest movement in
effective rank anywhere in the committed step-size map is a rise of about 0.6%, so 10%
would have been more than an order of magnitude beyond anything previously recorded.

## Per-site detail

| Site | Drift | Effective rank fall |
|---|---|---|
| `blocks.0.mlp` | 1.107% | 0.0207% |
| `blocks.1.mlp` | 1.327% | 0.0847% |
| `blocks.2.mlp` | 1.345% | 0.2437% |
| `blocks.3.mlp` | 1.562% | 0.0439% |
| `blocks.4.mlp` | 1.398% | 0.0210% |
| `blocks.5.mlp` | 1.490% | 0.0141% |
| `blocks.6.mlp` | 1.325% | 0.0065% |
| `blocks.7.mlp` | 1.390% | 0.0073% |
| `blocks.8.mlp` | 1.323% | 0.0078% |
| `blocks.9.mlp` | 1.502% | 0.0036% |
| `blocks.10.mlp` | 1.297% | 0.0103% |
| `blocks.11.mlp` | 1.029% | 0.0703% |

The largest fall at any single site is 0.244%, at `blocks.2.mlp`.

## Registered measurements not taken

Two quantities were registered for this stage and were not computed:

- The depth-weighted centre of the weight change across the twelve sites.
- The smallest weight change at which each statistic separates the drifted system from the
  frozen one.

Neither was implemented. They are recorded as not run rather than as absent findings.

## Limits

One driven input, one seed, one rule, twelve sites, all MLP output projections. Ceiling
lifted, so nothing here is continuous with any result taken under the former 5% cap.

Effective rank and top singular share are two specific quantities. They do not measure how
well a matrix separates the states this loop visits.

## Files

- `stage1.jsonl` — the run, including per-site figures.
- `experiments/exp003_stage1.py` — the runner.
