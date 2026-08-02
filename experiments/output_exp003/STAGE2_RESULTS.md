# EXP-003 Stage 2 results: the test did not run cleanly, and its own guard said so

*Pre-registered in `PREREGISTRATION.md`, as amended by Amendment 1 there, both
written before this ran. The run is `stage2.jsonl`. Nothing here enters the claim
register.*

## The answer first

The experiment asked whether the collapse is caused by adjusting the weights too
fast, by delivering the same total adjustment in fewer, larger steps.

**It did not deliver the same total adjustment.** The three settings produced
drifts of 1.31, 2.99 and 7.91 percent, a spread of 6.03 times across a ladder that
was supposed to be matched. The pre-registration set a tolerance of two times and
said that beyond it the comparison is qualitative only and the falsifier is not
invoked. That guard fired, so **the falsifier is not invoked and the timescale
question is unanswered.**

The reason is written in the pre-registration, one paragraph above the guard:
multiplying the step size by the cadence holds the total adjustment equal only if
the rule's effect is linear in step size, and EXP-002 already recorded that it is
not. That caution was correct and the correction is now known: the step sizes have
to be re-anchored per cadence to hit a common drift, exactly as EXP-002 had to
anchor per layer.

What can be said without the matching is narrow and is set out below.

## What was run, and what came out

| Cadence | Adjustments | Drift | Census agreement | Settled | Settled words |
|---|---|---|---|---|---|
| reference, frozen | 0 | 0 | **31 / 31** | 31 / 31 | the five committed end states |
| every iteration | 120 | 1.3112% | **0 / 31** | 31 / 31 | `Rousse` 25, `anarchism` 5, `comrade` 1 |
| every 4th | 30 | 2.9909% | **0 / 31** | 31 / 31 | `observer` 31 |
| every 12th | 10 | 7.9112% | **0 / 31** | 31 / 31 | `.` 31 |

Nothing clipped and nothing went non-finite at any setting.

**The reference gate passed at 31 of 31**, which is the check that matters most
here: before any weight moves, the census machinery reproduces the committed
baseline exactly. Every number below is therefore attributable to the drift rather
than to the instrument.

**The every-iteration cell reproduces EXP-002.** Its drift is 0.013111766434820447
against that experiment's committed 0.013111766434820447, and its driven input
settles on `Rousse`, which is what EXP-002 records for the same arm.

## What can be said, and what cannot

**Cannot be said: anything about timescale.** The settings differ in total drift by
six times, so a difference between them is not attributable to how often the
adjustment was applied. This is the whole question the stage exists to answer and
it remains open.

**Can be said, weakly: slowing the adjustment did not rescue anything here.**
Census agreement is zero out of thirty one at every setting, including the slowest.
If slowing the adjustment twelvefold were sufficient on its own to preserve
structure, one would not expect zero at every point. That is weak evidence because
the slowest setting also carries six times the drift, and more drift should destroy
more, so the two effects push the same way and cannot be separated.

**Can be said: the direction of the confounded effect is opposite to the
prediction.** Slower adjustment produced *more* complete collapse, from three
distinct end states down to one, not less. That is fully explained by the larger
drift and is reported as an observation rather than as evidence against the
timescale reading.

**Worth noting for its own sake: every input settled at every setting.** Thirty one
of thirty one reached a fixed point or a two-step cycle in all three cells. Whatever
the reinforcing rule leaves behind, it is a genuine attractor rather than a system
unable to come to rest, which matches EXP-002's return test for the same rule and
distinguishes it sharply from the eroding rule's behaviour there.

## A small discrepancy against EXP-002, recorded rather than smoothed

The every-iteration cell should match EXP-002's closed reinforcing arm and nearly
does, but not exactly. This run gives `Rousse` 25, `anarchism` 5, `comrade` 1; that
experiment reports `Rousse` 27, `anarchism` 3, `comrade` 1. Two inputs differ.

The drift matches to sixteen significant figures, so the episode is identical. The
likely cause is the census draw: both stage runners select thirty one inputs
stratified over the five end states, but they are separate implementations of that
selection and may not pick the same thirty one. This was not checked before the run
and should be. Until it is, the two figures are close but not the same measurement,
and neither is quoted as confirming the other.

## What to do instead

The clean version of this experiment anchors the step size per cadence to hit a
common drift, rather than assuming a linear scaling that the rule does not obey.
That is one calibration pass per setting, of the kind EXP-002 ran across layers, and
it turns a six times spread into a matched comparison. Cost is roughly one extra
short episode per cadence, so about ten minutes added to a run that took just under
fifty.

The re-run should also fix the census draw so it is shared with EXP-002's rather
than reimplemented.

## Limits

One driven input, one seed, one rule, twelve sites, all MLP output projections, no
external signal anywhere in this stage. Three cadence settings spanning a factor of
twelve, which Amendment 1 already records is weaker than the hundredfold originally
registered, so even a matched version of this experiment would not exclude an effect
that needs two orders of magnitude.

## Files

- `stage2.jsonl` — every cell, including per-input settled words and both cosines.
- `experiments/exp003_stage2.py` — the runner.
