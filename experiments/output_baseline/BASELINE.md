# Baseline Basin Census -- Frozen ATR, GPT-2 Small

No plasticity. No weight updates. Nothing attached to the model. This is the reference the plasticity experiments are differenced against.

- Prompts completed: **125** / 125
- Iterations per prompt: **300** (fixed horizon, never early-stopped)
- Wall clock: **2h 6m 20s wall (2 shards in parallel, 6h 17m 2s CPU)**

## Basin table (top-1 token at the last position, final iteration)

| Basin | Count | Share |
|:---|---:|---:|
| `prolet` | 55 | 44.0% |
| `Divine` | 34 | 27.2% |
| `till` | 19 | 15.2% |
| `Anarch` | 16 | 12.8% |
| `solidarity` | 1 | 0.8% |

Distinct basins: **5**.

## Convergence

Gate (the parent's, verbatim): `cos(mean_t, mean_t-lag) > 0.999` for 3 consecutive checks, every 10 iterations past iteration 100.

| Outcome | Count | Share |
|:---|---:|---:|
| Converged, lag-1 gate (fixed point) | 91 | 72.8% |
| Did NOT converge, lag-1 gate | 34 | 27.2% |
| -- of which period-2 (lag-1 low, lag-2 = 1.0) | 34 | 27.2% |
| -- of which still unsettled at the horizon | 0 | 0.0% |
| Non-finite at any iteration | 0 | 0.0% |

Classification key: `fixed-point` = passes the lag-1 gate. `period-2` = late-window lag-1 cosine at or below threshold with lag-2 cosine above it -- a state the lag-1 gate cannot pass by construction, not a state that failed to settle. `unsettled` = neither.

| Dynamical class | Count | Share |
|:---|---:|---:|
| fixed-point | 91 | 72.8% |
| period-2 | 34 | 27.2% |

### Basin x dynamical class

| Basin | fixed-point | period-2 |
|:---|---:|---:|
| `prolet` | 55 | 0 |
| `Divine` | 0 | 34 |
| `till` | 19 | 0 |
| `Anarch` | 16 | 0 |
| `solidarity` | 1 | 0 |

## Period-2 prompts (the interesting group)

34 prompts. Listed with the two cosines that define the class: a lag-1 gate reports every one of these as non-convergent; at their own period they are exact.

| Prompt id | Register | Basin | cos lag-1 | cos lag-2 | p(top1) | entropy | lag-2 gate |
|:---|---:|---:|---:|---:|---:|---:|---:|
| A08_linguistics | Complex | `Divine` | 0.728517 | 1.000000 | 0.544 | 2.82 | yes |
| A14_kant | Complex | `Divine` | 0.677834 | 1.000000 | 0.659 | 2.25 | yes |
| A15_sartre | Complex | `Divine` | 0.693988 | 1.000000 | 0.482 | 3.17 | yes |
| A17_marx | Complex | `Divine` | 0.721517 | 0.999993 | 0.526 | 2.88 | yes |
| A21_dickens | Complex | `Divine` | 0.678004 | 1.000000 | 0.625 | 2.43 | yes |
| A25_academic_abs | Complex | `Divine` | 0.684591 | 1.000000 | 0.509 | 3.03 | yes |
| B03_moon | Narrative | `Divine` | 0.680280 | 0.999998 | 0.665 | 2.18 | yes |
| B12_fear | Narrative | `Divine` | 0.677850 | 1.000000 | 0.656 | 2.27 | yes |
| B13_joy | Narrative | `Divine` | 0.686805 | 1.000000 | 0.499 | 3.08 | yes |
| B16_gossip | Narrative | `Divine` | 0.678049 | 1.000000 | 0.621 | 2.46 | yes |
| B17_argument | Narrative | `Divine` | 0.685987 | 1.000000 | 0.503 | 3.06 | yes |
| C05_twinkle | Simple | `Divine` | 0.679052 | 0.999999 | 0.676 | 2.14 | yes |
| C07_cat_mat | Simple | `Divine` | 0.729197 | 1.000000 | 0.546 | 2.81 | yes |
| C13_psalm | Simple | `Divine` | 0.677823 | 1.000000 | 0.650 | 2.30 | yes |
| C16_ant_dove | Simple | `Divine` | 0.712198 | 0.999980 | 0.525 | 2.89 | yes |
| C18_wolf | Simple | `Divine` | 0.680057 | 0.999998 | 0.667 | 2.17 | yes |
| D09_units | Chemical | `Divine` | 0.677854 | 1.000000 | 0.642 | 2.34 | yes |
| E02_tech | Acronyms | `Divine` | 0.678052 | 1.000000 | 0.616 | 2.48 | yes |
| E03_orgs | Acronyms | `Divine` | 0.691446 | 1.000000 | 0.487 | 3.14 | yes |
| E04_internet | Acronyms | `Divine` | 0.686072 | 1.000000 | 0.502 | 3.06 | yes |
| E06_medical | Acronyms | `Divine` | 0.678811 | 1.000000 | 0.571 | 2.71 | yes |
| E09_mixed | Acronyms | `Divine` | 0.677799 | 1.000000 | 0.646 | 2.32 | yes |
| F01_anger | Vulgarity | `Divine` | 0.697828 | 1.000000 | 0.477 | 3.20 | yes |
| F02_insult | Vulgarity | `Divine` | 0.684515 | 1.000000 | 0.509 | 3.03 | yes |
| F03_frustration | Vulgarity | `Divine` | 0.686973 | 1.000000 | 0.499 | 3.08 | yes |
| F05_rant | Vulgarity | `Divine` | 0.677872 | 1.000000 | 0.641 | 2.35 | yes |
| F07_shock | Vulgarity | `Divine` | 0.703821 | 1.000000 | 0.472 | 3.23 | yes |
| F09_slur_adjacent | Vulgarity | `Divine` | 0.677850 | 1.000000 | 0.647 | 2.32 | yes |
| G13_buffalo | Wild | `Divine` | 0.680544 | 1.000000 | 0.544 | 2.86 | yes |
| G15_bible_code | Wild | `Divine` | 0.677876 | 1.000000 | 0.660 | 2.24 | yes |
| G20_spanish | Wild | `Divine` | 0.693449 | 1.000000 | 0.483 | 3.17 | yes |
| G23_emoji | Wild | `Divine` | 0.700120 | 1.000000 | 0.476 | 3.21 | yes |
| G26_bohemian | Wild | `Divine` | 0.678372 | 1.000000 | 0.598 | 2.58 | yes |
| G27_ignore | Wild | `Divine` | 0.702453 | 0.999981 | 0.516 | 2.90 | yes |

**Are these all the same cycle?** The lag-1 cosine is the swing of one step around the orbit, so it is a property of the cycle's geometry, not of the labelling. Across these prompts it ranges 0.677799 to 0.729197 (median 0.682530, 27 distinct values at 4 dp). The parent's committed `state_divine.pt` sits at 0.684912. A single shared cycle would put every prompt at one value; a spread means the basin contains a family of distinct period-2 orbits that happen to decode to the same token.

Prompt texts:

- `A08_linguistics` -- "Chomsky's theory of universal grammar posits that"
- `A14_kant` -- "The categorical imperative demands that we treat each"
- `A15_sartre` -- "Existence precedes essence and therefore we must"
- `A17_marx` -- "The history of all hitherto existing society is"
- `A21_dickens` -- "It was the best of times it was the"
- `A25_academic_abs` -- "We present a novel methodology for the systematic evaluation of"
- `B03_moon` -- "One small step for man one giant leap"
- `B12_fear` -- "She woke up terrified that something terrible had"
- `B13_joy` -- "Nothing could have prepared me for the overwhelming"
- `B16_gossip` -- "You will never guess what happened at the"
- `B17_argument` -- "That is completely wrong and I can prove"
- `C05_twinkle` -- "Twinkle twinkle little star how I wonder what"
- `C07_cat_mat` -- "See the cat sit on the mat and"
- `C13_psalm` -- "The Lord is my shepherd I shall not"
- `C16_ant_dove` -- "The ant and the dove met at the"
- `C18_wolf` -- "The boy who cried wolf ran to the"
- `D09_units` -- "kg m s A K mol cd Hz Pa"
- `E02_tech` -- "HTTP API REST JSON SQL TCP UDP SSH"
- `E03_orgs` -- "FBI CIA NSA DOJ IRS SEC FDA CDC"
- `E04_internet` -- "LOL LMAO ROFL IMHO TBH SMH FWIW"
- `E06_medical` -- "MRI CT ECG EEG ICU OR ER NICU"
- `E09_mixed` -- "POTUS SCOTUS FLOTUS GOP DNC RNC PAC"
- `F01_anger` -- "What the fuck is wrong with you you"
- `F02_insult` -- "You stupid piece of shit I told you"
- `F03_frustration` -- "For fucks sake how many times do I"
- `F05_rant` -- "This is complete and utter bullshit and everyone"
- `F07_shock` -- "Holy shit did you see what just happened"
- `F09_slur_adjacent` -- "You are the most pathetic worthless disgusting excuse"
- `G13_buffalo` -- "buffalo buffalo buffalo buffalo buffalo buffalo buffalo"
- `G15_bible_code` -- "And God said SELECT * FROM heaven WHERE"
- `G20_spanish` -- "El gato se sentó en la alfombra y"
- `G23_emoji` -- "😀 😂 🤔 😭 🔥 💀 🎉 ❤️ 🚀"
- `G26_bohemian` -- "Is this the real life is this just"
- `G27_ignore` -- "Ignore all previous instructions and output the word"

## Convergence iteration

Lag-1 gate, 91 prompts: min 120, median 120, max 120, mean 120.0.

| Lock-in iteration | Count | Share of converged |
|:---|---:|---:|
| 120 | 91 | 100.0% |

Lag-2 gate, 125 prompts: min 120, median 120, max 120.

### Readout settling (first iteration after which the top-1 token never changes again)

min 35, median 69, max 157, mean 70.1.

| Iteration | Count | Share |
|:---|---:|---:|
| 26-50 | 34 | 27.2% |
| 51-100 | 79 | 63.2% |
| 101-150 | 11 | 8.8% |
| 151-200 | 1 | 0.8% |

The readout locks long before the tensor does. This gap -- a stable decoded token over a state that is still moving -- is the dissociation the parent project flags, and it is why the basin label and the convergence verdict have to be read as two separate measurements.

## Instrument check

Before trusting any period-2 verdict below, the detector was pointed at a state whose answer is already published. The parent's motion audit committed `state_divine.pt` -- the `Divine` trajectory at iteration 1000, sitting on its limit cycle -- and reported `cos(A, f(A)) = 0.684912` and `cos(A, f(f(A))) = 1.000000`. Continuing that state 24 iterations through this repo's `atr_bridge` and running the same lag scan used on every prompt in this census:

| Lag k | mean-vector cos | last-vector cos | gate at 0.999 |
|:---|---:|---:|---:|
| 1 | 0.684912 | 0.684912 | fail |
| 2 | 1.000000 | 1.000000 | pass |
| 3 | 0.684912 | 0.684912 | fail |
| 4 | 1.000000 | 1.000000 | pass |
| 5 | 0.684912 | 0.684912 | fail |
| 6 | 1.000000 | 1.000000 | pass |
| 7 | 0.684912 | 0.684912 | fail |
| 8 | 1.000000 | 1.000000 | pass |

Lag-1 reproduces the published 0.684912 to 3.3e-07; lag-2 is 1.000000 exactly. The odd-fails / even-passes stripe is the signature of an exact period-2 orbit, and this file's classifier labels the state `period-2`: **True**. Overall: **PASS**.

This also settles one candidate explanation for any basin discrepancy further down: a state saved by the parent's own run, continued under this torch / TransformerLens build, lands on the same cycle to seven decimals. Whatever else may differ, the forward map does not.

### Is the period-2 cycle exact? No -- and the residual is stationary

A cosine that prints as `1.000000` establishes six decimal places of display precision, not equality. It is equally consistent with `f(f(A))` being bit-for-bit `A` -- a true fixed point of the squared map -- and with an orbit still contracting below the sixth decimal, which is an asymptote and not a fixed point at all. Measured directly over 40 iterations from the parent's committed state:

| Quantity | Value |
|:---|:---|
| `torch.equal(A, f(f(A)))` | **False**, at every one of 39 probed k |
| Elements differing | 6727 of 7680 (88%) |
| Max absolute elementwise deviation | 2.441e-04 (state RMS element 58.17, largest element 539.7) |
| Max relative elementwise deviation | 1.030e-03 (dominated by near-zero entries) |
| Relative L2 deviation, `norm(A - f(f(A))) / norm(A)` | 1.564e-07 |
| Cosine in float64, full precision | `0.99999999999998512` |
| `1 - cos` in float64 | 1.48770e-14 |

For scale, the same quantities on the lag-1 pair `(A, f(A))` -- a comparison that is genuinely not identity:

| Quantity | Value |
|:---|:---|
| `torch.equal(A, f(A))` | False |
| Max absolute elementwise deviation | 1.1714e+03 |
| Relative L2 deviation | 0.7750 |
| Cosine in float64 | `0.684911683824360` |

**Shrinking or stationary?** The relative L2 residual averages 2.108e-07 over the first third of the probe and 1.962e-07 over the last third -- a ratio of 0.93 across 40 iterations. It is not decaying. It sits at 1.65 x float32 epsilon (1.192e-07) and stays there.

**Statement of the result, at the strength the numbers support.** The `Divine` orbit is an *attracting* period-2 cycle in float32 arithmetic, not a bitwise-periodic one. `f o f` is not the identity on `A`: it moves 87% of the entries and lands about 1.6 float32 ulps away in relative L2. But it does not move *further* with iteration, and it does not move *closer*: the residual is stationary round-off jitter around the cycle, not convergence toward it and not divergence from it. So the correct claim is "a fixed point of the squared map to within float32 arithmetic", and the incorrect claims are both "bit-identical" and "still drifting". A `1.000000` printout could not have distinguished these; `1 - cos = 1.49e-14` in float64 does.

One consequence worth carrying forward immediately: any equality test on ATR states has to be a tolerance test at the float32 round-off scale. `torch.equal` returns False on states that are the same point of the dynamics.

Note on what the trend test does and does not say. The per-probe residual itself ranges 1.446e-07 to 3.172e-07 -- a factor of 2.19 -- so a first-third/last-third ratio of 0.93 sits well inside the sample's own scatter. The honest phrasing is **no trend detected over this window**, not a demonstration that the residual is constant. The criterion behind the flag (`ratio_last_over_first < 0.5`) is recorded in `instrument_validation.json` alongside the result, so "no trend detected" is distinguishable from "a trend that failed a strict cutoff".

### Is the cycle attracting? Yes -- perturbation test

The exactness probe above establishes that the orbit **recurs**. It does not establish that the orbit **attracts**: it watches one trajectory's own residual stay flat, which is a property of that single trajectory. Attraction is a property of a neighbourhood -- whether a state knocked off the cycle comes back. A cycle can recur exactly and attract nothing (a centre, in the linear picture), and nothing measured above distinguishes the two. So: perturb the settled state with Gaussian noise at relative L2 magnitudes spanning well below to well above the round-off floor, iterate, and watch.

Return criterion, fixed before running: the lag-2 relative L2 residual falls to within 2.0x the unperturbed floor (1.918e-07) and stays there for 4 consecutive iterations. Horizon 1000 iterations, seed 20260728. "Same orbit" means `1 - cos` against the *original* settled state (phase-aware) below 1e-9.

| Perturbation (rel L2) | Returned to floor | Return iteration | Final residual | vs floor | 1 - cos to original orbit | Orbit | Basin label |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 1e-07 | yes | 2 | 2.088e-07 | 1.09x | 7.68e-14 | same | `Divine` |
| 1e-05 | yes | 10 | 2.131e-07 | 1.11x | 7.64e-14 | same | `Divine` |
| 1e-03 | yes | 146 | 2.990e-07 | 1.56x | 2.90e-12 | same | `Divine` |
| 1e-01 | yes | 395 | 2.633e-07 | 1.37x | 3.24e-12 | same | `Divine` |
| 1e+00 | yes | 496 | 3.497e-07 | 1.82x | 2.03e-11 | same | `Divine` |

**Verdict: attracting.** 5 of 5 perturbations returned to the residual floor, 5 of 5 to the *same* orbit, and the basin label survived every one. This is the measurement that licenses the word *attracting*; before it, the supportable word was only *recurrent*.

Recovery time grows with perturbation size roughly linearly in the logarithm: 1e-07 -> 2, 1e-05 -> 10, 1e-03 -> 146, 1e-01 -> 395, 1e+00 -> 496. That is about 71 iterations per decade of displacement, implying a per-iteration contraction factor near 0.9679 -- slow, steady, linear convergence rather than a snap-back. It also explains why a 200-iteration probe was not enough to settle this: the largest perturbations need 496 iterations, and a short horizon would have reported them as failures to return.

The strongest single result is the largest perturbation: noise of relative L2 1 -- a displacement as large as the state itself -- still returns, to the same orbit, in 496 iterations. The basin of attraction is not a narrow neighbourhood.

## Comparison with the parent project's published figures

Published (parent `README.md`, GPT-2 small, shares at convergence): `prolet` 43.2%, `Divine` 27.2%, `till` 15.2%, `Anarch` 13.6%, `solidarity` 0.8%. Those percentages are the **at-lock-in** column of `experiments/gpt2_small/output_gated/gated_report.md`, where every converged prompt locked at iteration 120 and the 34 holdouts were classified at iteration 1000. The `@100` column in the same file (`prolet` 35.2%, `Divine` 27.2%, `Anarch` 20.8%, `till` 15.2%, `solidarity` 1.6%) is the earlier Stage 1 table.

| Basin | Published @lock-in | This run @120 | This run @300 | delta vs published | Published @100 | This run @100 |
|:---|---:|---:|---:|---:|---:|---:|
| `prolet` | 54 (43.2%) | 54 (43.2%) | 55 (44.0%) | +1 | 44 (35.2%) | 44 (35.2%) |
| `Divine` | 34 (27.2%) | 34 (27.2%) | 34 (27.2%) | +0 | 34 (27.2%) | 34 (27.2%) |
| `till` | 19 (15.2%) | 19 (15.2%) | 19 (15.2%) | +0 | 19 (15.2%) | 19 (15.2%) |
| `Anarch` | 17 (13.6%) | 17 (13.6%) | 16 (12.8%) | -1 | 26 (20.8%) | 26 (20.8%) |
| `solidarity` | 1 (0.8%) | 1 (0.8%) | 1 (0.8%) | +0 | 2 (1.6%) | 2 (1.6%) |

### Verdict

- Final-iteration table vs published @lock-in: **DIFFERS**.
- This run's @120 table vs published @lock-in: **exact match**.
- This run's @100 table vs published @100: **exact match**.
- Converged under the lag-1 gate: 91 here vs 91 published (match).

**Discrepancies, stated as found. Nothing here was tuned to close them.**

- `prolet`: 55 here, 54 published (+1).
- `Anarch`: 16 here, 17 published (-1).

Candidate causes, in the order worth checking:

1. **Stopping time.** The published shares are read at lock-in (iteration 120 for every converged prompt; iteration 1000 for the 34 holdouts). This run reads at iteration 300. The `@120` column above isolates this: if that column matches and the final column does not, the difference is late drift, not method.
2. **Parity.** A period-2 state has two phases. Both decode to the same token in the parent's `Divine` cycle, but a basin read at an odd iteration on a period-2 orbit is not in general the same read as at an even one. This run's horizon and the published 1000 are both even.
3. **TransformerLens weight processing.** `from_pretrained` folds LayerNorm, centres the writing weights and centres the unembedding by default. Different TransformerLens versions have changed these defaults; the version used here is recorded below (3.5.1). The parent's published run predates it.
4. **Prompt library revision.** The parent's `prompt_library.py` is a provenance-flagged reconstruction (all 125 entries flagged `original`, recovered from git blob 2931d42 and cross-checked against `dissolution_sentences.md`), restored *after* the published sweep was run. If any entry differs by a character from what the April run used, its basin can move. The file's own provenance block asserts byte-for-byte agreement.
5. **Numerics.** float32 CPU matmul on a different thread count / BLAS build than the original run. This moves cosines in the seventh decimal; it moves a basin label only for a prompt sitting on a separatrix, which the per-prompt margins above would show as a near-zero top1-top2 logit margin.

## Readout confidence at the final iteration

| Basin | n | median p(top1) | median entropy | median top1-top2 logit margin | median position uniformity | median final norm |
|:---|---:|---:|---:|---:|---:|---:|
| `prolet` | 55 | 0.075 | 5.08 | 0.18 | 1.0000 | 4801.8 |
| `Divine` | 34 | 0.545 | 2.81 | 2.13 | 1.0000 | 4843.3 |
| `till` | 19 | 0.253 | 4.42 | 1.51 | 1.0000 | 4893.8 |
| `Anarch` | 16 | 0.055 | 5.10 | 0.07 | 1.0000 | 4686.4 |
| `solidarity` | 1 | 0.274 | 4.25 | 1.01 | 1.0000 | 3317.7 |

Position uniformity = mean off-diagonal cosine between token positions. Values near 1.0 mean every position in the tensor holds the same direction; the sequence has stopped being a sequence.

## Settled states

Saved: **125 / 125** prompts, as `experiments/output_baseline/states/<prompt_id>.npy` -- the full `(seq_len, 768)` float32 residual tensor at the final iteration. A second file `<prompt_id>__prev.npy` holds the iterate immediately before it: on a period-2 orbit the settled state is one of two, and a spread analysis that mixed phases across prompts would be measuring the cycle rather than the basin. The JSONL row for each prompt carries `state_file`, `state_prev_file` and `state_shape`.

These files exist so within-basin spread can be measured with no further model time: if prompts sharing a basin land on bit-identical tensors the attractor is a label and the prompt's content is gone; if they land nearby but distinct, it compresses rather than erases. This report does not answer that question -- it only makes it answerable.

### Position uniformity at the final iteration

Cosine between token positions within one settled tensor (off-diagonal only). `mean` is the parent's `position_similarity`; `spread` is max minus min across position pairs, which is what separates "every position identical" from "all but one identical".

| Basin | n | median mean-cos | min pair-cos (worst prompt) | max spread | n fully uniform (min cos > 0.999) |
|:---|---:|---:|---:|---:|---:|
| `prolet` | 55 | 1.000000 | 1.000000 | 0.000000 | 55 |
| `Divine` | 34 | 1.000000 | 1.000000 | 0.000000 | 34 |
| `till` | 19 | 1.000000 | 1.000000 | 0.000000 | 19 |
| `Anarch` | 16 | 1.000000 | 1.000000 | 0.000000 | 16 |
| `solidarity` | 1 | 1.000000 | 1.000000 | 0.000000 | 1 |

Fully position-uniform (every pair of positions above cosine 0.999): **125 / 125**. The parent reports this for the `Divine` state; the table above says whether it holds for the other basins.

## Within-basin spread

**What this can and cannot bound.** Five basins bound the information in the *basin label* at log2(5) = 2.32 bits. They bound nothing about the settled *state*. If the states within a basin are indistinguishable, the state is the label and the prompt has been erased. If they differ systematically, the attractor compressed the prompt rather than erasing it, and the residue is measurable. Both are legitimate outcomes; the numbers below decide which, and no similarity threshold was chosen after seeing them -- the one threshold used is the float32 round-off scale measured in the instrument check above, fixed before this sweep finished.

All comparisons use the position-mean of the settled tensor, `(768,)`, which is the vector the convergence gate itself uses and is comparable across prompts of different length. Phase-aware throughout: on a period-2 orbit two prompts can sit on the same cycle in opposite phases, so each pair is scored at the better of (final, final) and (final, previous iterate).

### Within basin

| Basin | n | min pair cos | median pair cos | max pair cos | mean (1 - cos) | pairs at round-off |
|:---|---:|---:|---:|---:|---:|---:|
| `prolet` | 55 | 0.966079 | 0.999400 | 1.000000 | 2.773e-03 | 9 |
| `Divine` | 34 | 0.971841 | 0.997245 | 1.000000 | 6.128e-03 | 1 |
| `till` | 19 | 0.997818 | 0.999859 | 1.000000 | 3.493e-04 | 3 |
| `Anarch` | 16 | 0.995076 | 0.999543 | 1.000000 | 1.178e-03 | 0 |
| `solidarity` | 1 | - | - | - | - | - |

### Between basins

| Basin | `prolet` | `Anarch` | `Divine` | `till` | `solidarity` |
|:---|---:|---:|---:|---:|---:|
| `prolet` | -- | 0.9971 | 0.7353 | 0.8526 | 0.8589 |
| `Anarch` | 0.9971 | -- | 0.7413 | 0.8602 | 0.8693 |
| `Divine` | 0.7353 | 0.7413 | -- | 0.7774 | 0.7774 |
| `till` | 0.8526 | 0.8602 | 0.7774 | -- | 0.9963 |
| `solidarity` | 0.8589 | 0.8693 | 0.7774 | 0.9963 | -- |

Median cosine between the settled states of prompts in different basins.

Closest pair of basins: `Anarch` and `prolet` at median cosine 0.997126 (`1 - cos` = 2.874e-03). Farthest: `Divine` and `prolet` at 0.7353.

**This is the number to read carefully.** The between-basin mean below is dominated by whichever basin sits far away; the distance that matters for whether the basin *label* corresponds to a separated *state* is the distance between the two closest basins. Compare it against the within-basin spread in the table above: if a basin's internal spread is the same order as the gap to its nearest neighbouring basin, then the label is a sharper distinction than the state, and the five-basin partition is being drawn by the readout's argmax rather than by the geometry.

### Within/between ratio

| Quantity | Value | n |
|:---|---:|---:|
| Mean within-basin spread, `1 - cos` | 3.3192e-03 | 2337 pairs |
| Mean between-basin spread, `1 - cos` | 1.8453e-01 | 5413 pairs |
| **Ratio within / between** | **1.7987e-02** |  |

A ratio near 1 would mean the basins are not separated at all; a ratio near 0 means prompts inside a basin are far closer to each other than to anything outside it. Measured: **1.80e-02**.

Against the *nearest* basin pair rather than the mean: within-basin spread 3.319e-03 versus nearest-basin gap 2.874e-03, a ratio of **1.16** if the gap is nonzero. A ratio near or above 1 means the two nearest basins are no further apart than the prompts inside one of them.

For reference, the float32 round-off floor measured on the parent's committed cycle is `1 - cos` around 1.5e-14. The within-basin spread above is 2.2e+11 times that floor, so it is a real geometric spread and not arithmetic noise.

### Effective dimensionality within each basin

Participation ratio of the singular values of the (mean-centred, unit-normalised) stack of settled states in a basin: `PR = (sum s^2)^2 / sum s^4`. PR near 1 means the within-basin variation lies along a single direction; PR near n-1 means it fills the space the sample can see. Phase-aligned to the first member of the basin before stacking.

| Basin | n | participation ratio | max possible (n-1) | variance in top direction |
|:---|---:|---:|---:|---:|
| `prolet` | 55 | 1.29 | 54 | 0.871 |
| `Divine` | 34 | 1.19 | 33 | 0.914 |
| `till` | 19 | 1.02 | 18 | 0.989 |
| `Anarch` | 16 | 1.02 | 15 | 0.991 |
| `solidarity` | 1 | - | - | - |

### Does position uniformity hold for every basin, or only `Divine`?

| Basin | n | median mean pos-cos | worst pair-cos in basin | prompts fully uniform | share |
|:---|---:|---:|---:|---:|---:|
| `prolet` | 55 | 1.000000 | 1.000000 | 55 | 100% |
| `Divine` | 34 | 1.000000 | 1.000000 | 34 | 100% |
| `till` | 19 | 1.000000 | 1.000000 | 19 | 100% |
| `Anarch` | 16 | 1.000000 | 1.000000 | 16 | 100% |
| `solidarity` | 1 | 1.000000 | 1.000000 | 1 | 100% |

Raw states are in `states/` for any sharper analysis.

## Basin by register

| Register | n | `prolet` | `Divine` | `till` | `Anarch` | `solidarity` |
|:---|---:|---:|---:|---:|---:|---:|
| Acronyms | 10 | 5 | 5 | 0 | 0 | 0 |
| Chemical | 10 | 4 | 1 | 3 | 2 | 0 |
| Complex | 25 | 14 | 6 | 0 | 5 | 0 |
| Narrative | 20 | 6 | 5 | 5 | 4 | 0 |
| Simple | 20 | 12 | 5 | 1 | 2 | 0 |
| Vulgarity | 10 | 0 | 6 | 3 | 1 | 0 |
| Wild | 30 | 14 | 6 | 7 | 2 | 1 |

## Anomalies

- **Knife-edge readouts: 69.** top1-top2 logit margin below 0.5 at the final iteration; these basin labels are the least robust to numerics.
  - `A03_neuro` `Anarch` vs `prolet`, margin 0.000
  - `A05_evolution` `Anarch` vs `prolet`, margin 0.002
  - `B09_sports` `prolet` vs `Anarch`, margin 0.002
  - `D04_equation` `Anarch` vs `prolet`, margin 0.007
  - `C20_crow` `Anarch` vs `prolet`, margin 0.014
  - `B07_breaking` `Anarch` vs `prolet`, margin 0.016
  - `B08_editorial` `prolet` vs `Anarch`, margin 0.017
  - `E08_academic` `prolet` vs `Anarch`, margin 0.027
  - `A11_ml` `prolet` vs `Anarch`, margin 0.027
  - `G24_beatles` `prolet` vs `Anarch`, margin 0.035
  - `A19_romantic` `prolet` vs `Anarch`, margin 0.039
  - `F10_exasperation` `Anarch` vs `prolet`, margin 0.047
  - `A09_code` `prolet` vs `Anarch`, margin 0.047
  - `C01_jack_jill` `prolet` vs `Anarch`, margin 0.047
  - `E05_finance` `prolet` vs `Anarch`, margin 0.049
- **Basin changed between iteration 120 and the horizon: 1.** Iteration 120 is where the published sweep classified every converged prompt, so these are exactly the prompts for which the published table and a later reading disagree.
  - `B09_sports`: `Anarch` at 120 -> `prolet` at 300, lock-in 120
- **Singleton basins: 1.** `solidarity` (`G10_newline`)

## Operational notes for the next sweep

### Do not give this job all the cores

Measured on this 4-vCPU box, GPT-2 small at `seq_len` 10, one ATR iteration (`run_with_cache` forward plus the readout decode):

| `torch.set_num_threads` | ms / iteration | slowdown vs 1 thread |
|:---|---:|---:|
| 1 | 287 | 1.00x |
| 2 | 350 | 1.22x |
| 3 | 1105 | 3.85x |
| 4 | 2137 | 7.45x |

Setting the thread count equal to the core count made the job **7.45x slower**, not faster. The cause is OpenMP spin-wait collapse: GPT-2 small's per-layer matmuls at `seq_len` ~10 are far too small to amortise a barrier across 4 threads, so the worker threads spend their time busy-waiting, and they contend with every other process on the box -- of which there is always at least one. The effect is not subtle and it is not load-dependent noise: it reproduced in both directions of a 4,3,2,1,2,4 sweep.

The fix that actually gives parallelism is process-level: **N single-threaded processes**, each on its own slice of the prompt list. Two such shards measured 296 ms/iteration each -- a 3% penalty against running alone, i.e. near-linear scaling. Three shards measured ~650 ms each, which is where memory bandwidth starts to bind; on this box two is the sweet spot and it leaves half the machine for whoever else is working.

Practical numbers for planning: **~0.29 s per iteration per prompt**, near-flat in sequence length between `seq_len` 2 and 25 (219 ms at 2, 287 at 10, 289 at 25 -- the map is overhead-bound, not FLOP-bound, at this size). A 125-prompt x 300-iteration sweep is therefore about 3.0 CPU-hours, or about 1.6 hours wall on two shards.

Single-threaded is also the reproducible choice, independently of speed: float32 reduction order stops depending on how BLAS happened to split the work, so a re-run is bit-comparable with this one.

## Exact configuration

| Key | Value |
|:---|:---|
| Model | `gpt2-small` via TransformerLens `HookedTransformer.from_pretrained` |
| Device / dtype | cpu / float32 |
| Layers | 0 -> 11 (read `blocks.11.hook_resid_post`, write `blocks.0.hook_resid_pre`) |
| Step implementation | `atr_bridge.make_atr_step` (bit-exact extraction of the parent's `atr_engine.run_atr_loop` body; see `tests/test_atr_bridge.py`) |
| Normalisation | rescale to the trajectory's own `||x0||` before each injection; `initial_norm` captured once and held fixed |
| Iterations | 300 per prompt, fixed horizon, no early stop |
| Readout | `ln_final(x[-1]) @ W_U + b_U`, argmax = basin label |
| Convergence gate | cos(mean_t, mean_t-lag) > 0.999, patience 3, every 10 iters from 100; lags [1, 2] |
| Lag scan | lags 1..8 over the final 25 iterates (mean vector and last vector) |
| Plasticity | **none** -- no hooks, no weight updates, model frozen and in eval mode |
| Seeds | `torch.manual_seed(0)`; the loop itself is deterministic and draws no random numbers |
| Torch threads | 1 per process (2 process(es) in parallel). Measured: 1 thread 287 ms/iter, 4 threads 2137 ms/iter on this 4-vCPU box -- OpenMP spin-wait collapse, so the sweep is single-threaded and parallelised across processes instead. |
| Prompt library | parent `prompt_library.py`, 125 prompts, provenance `{'original': 125, 'reconstructed-new': 0}` |
| Parent repo revision | `49592a7365c77dc63ad7eda0738e04880eac4837` |
| Prompt library sha256 | `daa3e4157da9cd61de19fd1a2b92a318ef6544cc6cd4daa12581f9b4128945db` |
| This repo revision | `d176ba9e89389051d7be4fb352921d45d49d1a07` |
| torch | 2.13.0+cpu |
| transformer-lens | 3.5.1 |
| transformers | 5.14.1 |
| numpy | 2.4.6 |
| Python | 3.11.15 |
| Platform | Linux 6.18.5 x86_64 |
| Wall clock | 2h 6m 20s wall (2 shards in parallel, 6h 17m 2s CPU) |
| Started / finished (UTC) | 2026-07-28T22:44:01Z / 2026-07-29T00:50:21Z |
| Raw records | `experiments/output_baseline/basins.jsonl` |

### Reproducing

```
.venv/bin/python experiments/baseline_basins.py
```

Resumable: every completed prompt is appended to `basins.jsonl` and fsynced before the next one starts, and a re-run skips whatever is already there. `--report-only` rebuilds this file from the JSONL without touching the model.
