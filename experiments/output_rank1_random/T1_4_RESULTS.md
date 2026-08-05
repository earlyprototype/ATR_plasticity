# T1.4 — rank-1 random control, complete

**Status: complete.** 10/10 Arm A seeds, 10/10 Arm B seeds. Supersedes
`T1_4_PARTIAL.md` (kept for the record). Source artifacts: `rank1_random.jsonl`
(22 records, including every Arm B probe) and `meta.json` (committed on
completion, as its `.gitignore` note requires).

The sweep was interrupted three times by the environment and finished under
`--resume`; the resumed process re-validated the harness before touching any
control cell (`prolet` → `comrade` reproduced, same reference values below).

## Setup

| | |
|---|---|
| model | gpt2-small, frozen, CPU, float32 (metrics float64) |
| site | `blocks.6.mlp`, (3072, 768), ‖W0‖_F = 164.854073 |
| prompt | `A01_physics`, 10 positions |
| episode | 120 frozen steps, same as EXP-001 |
| basin | top-1 token at the last position |
| torch / transformer-lens / python | 2.13.0+cpu / 3.6.0 / 3.11.15 |

## Harness validation

| quantity | this run | published |
|---|---|---|
| `hebb` σ₁ | 1.813517502131269 | 1.8135 |
| `hebb` ‖ΔW‖_F | 1.8528509717023924 | 1.8529 |
| `hebb` 1−cos(off) | 5.000281547660079e-03 | 5.000e-03 |
| frozen basin | `prolet` (margin 0.2303) | `prolet` |
| `hebb` basin | `comrade` (margin 0.3237) | `comrade` |

## Arm A — matched operator norm (σ₁ = 1.8135), 10/10 seeds

| | |
|---|---|
| flips | **0 / 10** |
| 1−cos(off) range | 7.4619e-09 – 1.2437e-06 |
| `hebb` 1−cos(off) | 5.0003e-03 |

Ratio of `hebb`'s displacement to this arm's: 4.0e+03 – 6.7e+05. A random
rank-1 direction at `hebb`'s exact operator norm moves the loop state three to
five orders of magnitude less than `hebb`'s direction does.

## Arm B — matched loop displacement (target 1−cos = 5.0003e-03 ± 2%), 10/10 seeds

| seed | scale | ‖ΔW‖_F / ‖W0‖_F | 1−cos(off) | rel err | basin | margin | matched | flip | evals |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 121.912 | 0.740 | 5.0182e-03 | 0.4% | `prolet` | 0.213 | yes | no | 5 |
| 1001 | 63.718 | 0.387 | 3.1838e-03 | 36.3% | `Anarch` | 0.062 | no | yes | 10 |
| 1002 | 317.696 | 1.927 | 5.0251e-03 | 0.5% | `Anarch` | 0.684 | yes | yes | 10 |
| 1003 | 184.116 | 1.117 | 5.0249e-03 | 0.5% | `bourgeois` | 0.383 | yes | yes | 9 |
| 1004 | 362.704 | 2.200 | 4.9612e-04 | 90.1% | `Anarch` | 0.196 | no | yes | 5 |
| 1005 | 214.260 | 1.300 | 5.0713e-03 | 1.4% | `prolet` | 0.084 | yes | no | 8 |
| 1006 | 145.081 | 0.880 | 5.0874e-03 | 1.7% | `Anarch` | 0.417 | yes | yes | 3 |
| 1007 | 81.498 | 0.494 | 4.2927e-03 | 14.2% | `Anarch` | 0.119 | no | yes | 11 |
| 1008 | 362.704 | 2.200 | 4.0490e-03 | 19.0% | `Anarch` | 0.020 | no | yes | 5 |
| 1009 | 124.670 | 0.756 | 4.9529e-03 | 0.9% | `bourgeois` | 0.212 | yes | yes | 8 |

`hebb` for comparison: scale 1.8135, ‖ΔW‖_F/‖W0‖_F = 0.0112, basin `comrade`,
margin 0.324.

Counts over the 10 seeds:

- matched within ±2%: **6 / 10** (seeds 1000, 1002, 1003, 1005, 1006, 1009)
- **flipped at a matched displacement: 4 / 6** (1002, 1003 → 1006, 1009)
- could not be matched: **4 / 10** (1001, 1004, 1007, 1008) — all 4 flipped at
  their closest achieved displacement, which in every case is *below* the target
- flipped at the reported (closest) probe: 8 / 10; flipped at some probe during
  the search: 8 / 10 (same seeds)
- destinations at the reported probe: `Anarch` ×6, `bourgeois` ×2, `prolet` ×2
- destination `comrade` (`hebb`'s): **0 / 10 — and 0 of all 74 Arm B probe
  evaluations.** Basins seen across every probe: `prolet` 37, `Anarch` 15,
  `bourgeois` 14, `Divine` 7, `till` 1, `comrade` 0
- `bourgeois` is not one of the 5 frozen baseline basins (`BASELINE.md`:
  `prolet`, `Divine`, `till`, `Anarch`, `solidarity`) — like `comrade`, it
  appears only under an edit

Margins are reported per T0.5 (`ALIGNMENT_REVIEW.md`); no ambiguity threshold
was fixed in advance for this experiment, so none is applied. For scale: 69 of
125 frozen baseline prompts sit below margin 0.5 (C-01), as do `hebb`'s own
flip (0.324) and 7 of these 8.

## Drift cost

To reach `hebb`'s loop displacement, the matched random directions need
‖ΔW‖_F/‖W0‖_F between **0.740 and 1.927** against `hebb`'s **0.0112** —
66× to 171× the relative weight change for the same loop-state movement.

## Unreachable targets

4/10 seeds record `matched=False`, in two distinct modes, both visible in the
committed probe records:

**Scale ceiling (seeds 1004, 1008).** Both stop at scale 362.704, the search's
hard cap (200 × `hebb` σ₁ = 2.2 × ‖W0‖_F). Displacement at the cap is still
below target: 4.9612e-04 (1004) and 4.0490e-03 (1008). More scale might reach
it; the search does not go there.

**Displacement discontinuity (seeds 1001, 1007).** The displacement jumps past
the target within a sub-percent scale interval, from ~4e-04 to the `Divine`
plateau at ~2.5e-01. Seed 1007's bisection brackets it: scale 81.277 →
4.3550e-04 (`prolet`), 81.498 → 4.2927e-03 (`Anarch`), 81.719 → 2.5969e-01
(`Divine`) — a 0.5% scale interval spanning three decades of displacement.
Seed 1001 is the same shape: 62.806 → 3.9158e-04 (`prolet`), 63.718 →
3.1838e-03 (`Anarch`), 64.642 → 2.6736e-01 (`Divine`). For these directions no
scale within the searched range produces the target displacement; the 5.0e-03
band sits inside the jump.

The saturation the search safeguard exists for is also in the committed
records: seed 1007's probes at scales 86.266, 102.588, 145.081 give 1−cos =
2.4719e-01, 2.4720e-01, 2.4739e-01 — a 1.7× scale range with the displacement
flat to three figures.

Three field caveats, all found in review, none touching a headline count:

1. **`saturation_suspected` reads `false` on seed 1007 despite that plateau.**
   The heuristic that wrote it compared displacements by exact equality after
   rounding to 9 decimals, which the plateau's 2e-05 relative agreement does
   not trigger. Rewritten to a relative-tolerance comparison after the sweep;
   the committed records keep the old rule's output. The probe lists themselves
   are the evidence — read those, not the flag.
2. **The `hebb` reference record's `clip_rate` and `rel_weight_change` are
   `null`** — the script asked `report()` for keys it has never had (fixed to
   `clipped`/`delta_frac` after the sweep). Ceiling-silence for this eta is
   therefore evidenced by `step_size_map.jsonl`'s 0.0%-clip cells (C-20/C-21),
   not by this record.
3. **Seed 1000's Arm B record was produced by the superseded log-log-secant
   search** (pre-`529c0b2`), which is why it lacks the `bracketed`,
   `saturation_suspected` and `position_spread_min_cos` fields the other nine
   carry. Its reported cell is still a plain evaluation of (seed 1000, scale
   121.912) — the search algorithm chose which scales were probed, not what an
   evaluation returns — and it is the best-matched cell of its seed at 0.4%
   relative error.

### Correction to `T1_4_PARTIAL.md`

The partial file's "Seed 1001's probe sequence" table quoted probes at scales
85.150 and 357.968. Those numbers appear in **no committed version of
`rank1_random.jsonl`** — they are from the discarded run 1 (the unsafeguarded
secant search whose defects are described in `rank1_random_control.py`'s
docstring), and issue #41 and the PR #39 body repeat them. The committed
seed-1001 record's probe sequence is the 10-row one summarised above. The
observation the discarded numbers illustrated — distinct scales, same
displacement — is real and is carried by the committed seed-1007 record
instead. The partial file is left as written, with a notice at the top.

## Position uniformity

Minimum pairwise cosine between token positions of the settled state,
recorded per evaluated cell. Over the 9 Arm B records that carry the field
(seed 1000's record predates its introduction in `529c0b2`; Arm A's records
all predate it):

| | |
|---|---|
| minimum over all 9 recorded cells | **0.9999999999999331** |
| scales covered | 0.39× – 2.20× ‖W0‖_F |
| C-06 reference (frozen states) | all pairs above cos 0.999 |

Position uniformity holds at every recorded Arm B scale, ten orders of
magnitude inside C-06's threshold. The position-mean used by `1−cos(off)` is a
faithful summary at these scales, so the Arm B match criterion is valid where
it was measured. Settled-state norm vs the frozen state (`norm_ratio_vs_off`)
stays within 0.984–1.007 across the same records.

## Bearing on C-22

C-22 reads: *the basin flip requires both a sufficient magnitude and the right
sign on the `hebb`/`oja` axis — neither alone*, status `provisional`, with
C-50/T1.4 named as the deciding test. The facts this sweep adds, stated
without amendment:

- 4 of the 6 seeds matched to `hebb`'s loop displacement within 2% change the
  basin. Random rank-1 directions with no particular sign on the `hebb`/`oja`
  axis flip the basin at `hebb`'s displacement.
- 0 of 10 seeds — and 0 of 74 probe evaluations — reach `hebb`'s destination
  `comrade`.
- Matching the displacement costs a random direction 66×–171× `hebb`'s
  relative weight change; at matched operator norm (Arm A) random directions
  move the loop state 3–5 orders of magnitude less and never flip.
- C-07's resolution caveat travels with the destinations: 6 of the 8 flips
  land in `Anarch`, whose distance to `prolet` (1−cos 2.874e-03) is below the
  **pooled** mean within-basin spread (3.319e-03, across all five basins; `prolet`'s own is 2.773e-03, which is *below* this gap, as is `Anarch`'s 1.178e-03 — see C-07's decomposition, which means the coarse-instrument caveat does not bite on these destinations).

**C-22 has not been amended here.** Per `CLAIMS.md`, a status change is a pull
request of its own with the evidence attached. This file is the evidence.
