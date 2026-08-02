# EXP-003 Stage 1 results: this project's explanation of its own collapse is wrong

*Pre-registered in `PREREGISTRATION.md` before the run, with thresholds fixed in
advance. The run is `stage1.jsonl`. Nothing here enters the claim register.*

## The answer first

`MEA_ANALOGUE.md` proposed that EXP-002's collapse happens because repeated rank-one
adjustments concentrate each weight matrix onto a single direction, turning each
adjusted site into something closer to a valve than a map, so that twelve of them in
series leave the loop able to produce only one resting state.

**That is refuted.** Reproducing EXP-002's reinforcing arm exactly, the mean
effective rank of the twelve adjusted matrices falls by **0.044 percent**. The
registered threshold for the explanation to be supported was a fall of 10 percent,
and the threshold for refutation was a fall below 2 percent. The measured value is
227 times smaller than the supporting threshold.

The share of each matrix held by its single strongest direction, which would go to 1
if the valve picture were right, moves from 0.00777 to 0.00780. It goes very
slightly the wrong way.

**The collapse is real and the weights do not concentrate.** Whatever destroys the
model's ability to tell its inputs apart, it is not a loss of capacity in the
matrices.

## The reproduction, which had to pass first

The runner refuses to report anything unless it reproduces EXP-002's episode, on the
grounds that effective ranks from a different episode would say nothing about
EXP-002's collapse. Using that experiment's per-layer step sizes verbatim from its
committed record:

| | |
|---|---|
| Aggregate drift, this run | 0.013111766435 |
| Aggregate drift, EXP-002 | 0.013111766435 |
| Relative miss | 0.0000 |
| Clipped | no |
| Non-finite | none |

An independent check on the measurement itself: this run reads `blocks.6.mlp`'s
frozen effective rank as 642.64, and the committed step-size map records 642.6 for
the same matrix. The instrument agrees with the project's existing number.

## The measurement

| | Frozen | After drift | Change |
|---|---|---|---|
| Mean effective rank over twelve matrices | 639.455 | 639.174 | **−0.044%** |
| Mean share held by the strongest direction | 0.007766 | 0.007797 | +0.4% |

Per site, with each matrix's own drift alongside:

| Site | Effective rank | | Fall | Drift |
|---|---|---|---|---|
| `blocks.0.mlp` | 629.30 | 629.17 | 0.021% | 1.107% |
| `blocks.1.mlp` | 640.60 | 640.05 | 0.085% | 1.327% |
| `blocks.2.mlp` | 624.37 | 622.85 | **0.244%** | 1.345% |
| `blocks.3.mlp` | 608.45 | 608.18 | 0.044% | 1.562% |
| `blocks.4.mlp` | 627.51 | 627.38 | 0.021% | 1.398% |
| `blocks.5.mlp` | 637.21 | 637.12 | 0.014% | 1.490% |
| `blocks.6.mlp` | 642.64 | 642.60 | 0.007% | 1.325% |
| `blocks.7.mlp` | 649.04 | 648.99 | 0.007% | 1.390% |
| `blocks.8.mlp` | 646.81 | 646.76 | 0.008% | 1.323% |
| `blocks.9.mlp` | 652.12 | 652.09 | 0.004% | 1.502% |
| `blocks.10.mlp` | 659.48 | 659.41 | 0.010% | 1.297% |
| `blocks.11.mlp` | 655.94 | 655.48 | 0.070% | 1.029% |

The largest fall anywhere is 0.244 percent, at `blocks.2.mlp`. Every site is far
below even the refutation threshold. There is no site at which the proposed
concentration is happening.

For scale, the committed step-size map's largest movement in this quantity across
every ceiling-silent cell is an increase of about 0.6 percent. So the movement here
is smaller than the movement that map already treats as no movement at all.

## What this costs, stated plainly

**It was predicted.** `MEA_ANALOGUE.md` recorded four objections to its own mechanism
before this ran, and the second was the arithmetic: a matrix with a stable rank of
about 31 out of 768, moved by 1.31 percent, is not going to be dominated by the
addition. That objection was correct.

**One of the two independent routes to the central prediction is gone.** The
distributed-beats-focal prediction had a biological route and a mathematical route,
and the mathematical route ran through this mechanism. It no longer does. The
prediction now rests on the biological measurement alone.

The pre-registration anticipated this and said what follows: the series continues,
because the collapse is a measured fact whatever explains it, and the Stage 3 result
becomes **more** informative rather than less. If spreading the signal out still
beats concentrating it when no mathematical argument predicts it should, that is a
stronger result for the analogy than it would have been with two routes agreeing.

**The claim in `MEA_ANALOGUE.md` is retracted by name rather than edited away**, per
that document's own instruction to say what was said, say it was wrong, and say what
is true instead.

## What the result suggests instead, marked as speculation

This is inference from existing register rows and has not been tested.

The weights come out of the episode with essentially the same spectral character
they went in with, yet the behaviour is destroyed: 27 of 31 inputs that previously
spread over five end states land on one. So the fragility is not in the weights. It
is in how little separates the end states in the first place.

Two committed rows point the same way. Row C-07 records that the spread of states
sharing an end state is *larger* than the gap between the two nearest distinct end
states, a ratio of about 1.16. Row C-55 records that arbitrary rank-one directions,
at matched displacement, usually move the settled end state. Together those say the
landscape sits on a knife edge, and that almost any coherent push of sufficient size
will topple it.

On that reading the collapse is not plasticity degrading the model. It is a barely
separated landscape being displaced wholesale by a 1.3 percent change that leaves the
model's capacity intact. That is a different claim from the one this document
refutes, it is more consistent with what the register already contains, and it makes
a different prediction: the collapse should be reachable by directions that have
nothing to do with the learning rule, which is close to what C-55 already found at a
single site.

Testing it is not part of EXP-003 as registered and is named here as the obvious
follow-up rather than run opportunistically.

## Limits

One episode, one driven input, one rule, one seed, twelve sites, all of them MLP
output projections. Effective rank and top singular share are two summaries of a
spectrum and do not exhaust what "concentration" could mean; a mechanism that
concentrates in some other sense is not excluded by this, only the one that was
stated. The measurement is on the weights alone and says nothing directly about the
loop's dynamics.

## Files

- `stage1.jsonl` — the record, including per-site spectra and the reproduction check.
- `experiments/exp003_stage1.py` — the runner, which refuses to report if the episode
  does not reproduce EXP-002 within five percent.
