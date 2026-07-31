# T1.4 — rank-1 random control, partial results

> **Superseded.** The sweep is complete: see `T1_4_RESULTS.md` (10/10 both arms).
> This file is kept as the mid-run record. One correction, marked rather than
> silently applied: the "Seed 1001's probe sequence" table below quotes probes at
> scales 85.150 and 357.968 that appear in **no committed version of
> `rank1_random.jsonl`** — they are from the discarded run 1. The committed
> seed-1001 record's probes are listed in `T1_4_RESULTS.md`; the
> distinct-scales-same-displacement observation survives via the committed
> seed-1007 record.

**Status: incomplete.** 10/10 Arm A seeds, 5/10 Arm B seeds. Run stopped at session
end. Resume with:

```bash
.venv/bin/python experiments/rank1_random_control.py --seeds 10 --resume
```

`meta.json` is gitignored and absent until the sweep completes. Source records:
`rank1_random.jsonl` (17 records, including every Arm B probe).

## Setup

| | |
|---|---|
| model | gpt2-small, frozen, CPU, float32 (metrics float64) |
| site | `blocks.6.mlp`, (3072, 768), ‖W0‖_F = 164.854073 |
| prompt | `A01_physics`, 10 positions |
| episode | 120 frozen steps, same as EXP-001 |
| basin | top-1 token at the last position |
| transformer-lens | 3.6.0 |

## Harness validation

| quantity | this run | published |
|---|---|---|
| `hebb` σ₁ | 1.813517502131269 | 1.8135 |
| `hebb` ‖ΔW‖_F | 1.8528509717023924 | 1.8529 |
| `hebb` 1−cos(off) | 5.000281547660079e-03 | 5.000e-03 |
| frozen basin | `prolet` | `prolet` |
| `hebb` basin | `comrade` | `comrade` |

## Arm A — matched operator norm (σ₁ = 1.8135), 10/10 seeds

| | |
|---|---|
| flips | **0 / 10** |
| 1−cos(off) range | 7.4619e-09 – 1.2437e-06 |
| `hebb` 1−cos(off) | 5.0003e-03 |

Ratio of `hebb`'s displacement to this arm's: 4.0e+03 – 6.7e+05.

## Arm B — matched loop displacement (target 1−cos = 5.0003e-03), 5/10 seeds

| seed | scale | ‖ΔW‖_F / ‖W0‖_F | 1−cos(off) | basin | matched (±2%) | flip |
|---|---|---|---|---|---|---|
| 1000 | 121.912 | 0.739 | 5.0182e-03 | `prolet` | yes | no |
| 1001 | 63.718 | 0.386 | 3.1838e-03 | `Anarch` | no | yes |
| 1002 | 317.696 | 1.927 | 5.0251e-03 | `Anarch` | yes | yes |
| 1003 | 184.116 | 1.117 | 5.0249e-03 | `bourgeois` | yes | yes |
| 1004 | 362.704 | 2.200 | 4.9612e-04 | `Anarch` | no | yes |

`hebb` for comparison: scale 1.8135, ‖ΔW‖_F/‖W0‖_F = 0.0112, basin `comrade`.

Counts over the 5 completed seeds:

- flipped: **4 / 5**
- flipped at a matched displacement: **2 / 5** (seeds 1002, 1003)
- matched within ±2%: **3 / 5**
- destinations reached: `Anarch` ×3, `bourgeois` ×1
- destination `comrade` (`hebb`'s): **0 / 5**

## Position uniformity

Minimum pairwise cosine between token positions of the settled state:

| seed | scale | min cos |
|---|---|---|
| 1001 | 63.718 | 0.9999999999999704 |
| 1002 | 317.696 | 0.9999999999999384 |

C-06 records position uniformity for frozen states. These are measured under edits
of 0.39× and 1.93× ‖W0‖_F.

## Unreachable targets

Seeds 1001 and 1004 record `matched=False`: the bracketed search did not span the
target. Seed 1001's probe sequence:

| scale | 1−cos(off) | basin |
|---|---|---|
| 1.814 | 1.2834e-07 | `prolet` |
| 63.718 | 3.1838e-03 | `Anarch` |
| 85.150 | 2.5809e-01 | `Divine` |
| 357.968 | 2.5802e-01 | `Divine` |

Scales 85.150 and 357.968 differ by 4.2× and give the same 1−cos to four figures.

## Bearing on C-22

C-22 currently reads: *the basin flip requires both a sufficient magnitude and the
right sign on the `hebb`/`oja` axis*, status `provisional`, with C-50/T1.4 named as
the falsifying test.

Seeds 1002 and 1003 are rank-1 random directions matched to `hebb`'s loop
displacement within 2%, correctly bracketed, that change the basin.

**C-22 has not been amended.** Per `CLAIMS.md`, a status change is a pull request of
its own with the evidence attached, and this run is half complete.
