# EXP-003 Stage 0 results: the measurement works, with one gate failed and one surprise

*Pre-registered in `PREREGISTRATION.md` before the run. Every threshold below was
written down first. The run is `stage0.jsonl`, 125 inputs at 120 iterations each,
about 52 minutes. Nothing here enters the claim register; the register rows this
work would eventually populate are claimed on the peer board and remain unfilled.*

## The answer first

The label-free measurement adapted from Chao, Bakkum and Potter separates this
model's five end states far better than the token labels the project has been
using, and a control confirms that depth is what carries the information rather
than the reduction being an artefact of squashing many numbers into one.

It scores **12.20** on a scale where 1.0 means the groups cannot be told apart
from their own internal scatter, against **0.87** for the token labels themselves.
Shuffling the block ordering destroys it, dropping the score to **0.37**.

**One registered gate failed**, and it is reported as failed rather than explained
away. The measurement cannot separate the two dynamical classes. The reason is
structural and is given below.

**One expectation was wrong.** The method the source actually used performs
substantially worse here than the simpler reduction I registered. That is reported
because it was measured, not because it was hoped for.

## What was run

Every one of the 125 inputs in the committed baseline census, iterated 120 times
under the frozen model with nothing adjusting. At each iteration the activity of
all 144 sites was recorded, and the centre of activity in depth was computed: the
average block index, weighted by how much each block writes into the model's
internal state. The value reported per input is that quantity averaged over the
last 25 iterations, which is the window the committed census uses for its own
settling statistics.

The gate on the runner is that its trajectory reproduces the project's existing
loop bit for bit, tested and passing at exactly zero deviation, so this measures
the loop the rest of the project studies rather than a near neighbour of it.

## The gates, as registered

| Gate | Registered threshold | Result | |
|---|---|---|---|
| 1. Separates the five end states | above 1.5 | **12.2035** | pass |
| 2. Separates the two dynamical classes | above 1.5 | **1.0036** | **fail** |
| 3. Does not separate random halves of the largest group | below 1.2 in at least 9 of 10 | **10 of 10** | pass |
| A. Shuffling block order destroys it | below the true score | **0.3701** vs 12.2035 | pass |
| B. Shuffling head labels changes nothing | below 1e-9 | **1.78e-15** | pass |

The threshold of 1.5 was set against a stated baseline: the token labels score 0.87
on the same scale, computed from register row C-07, and 1.0 means indistinguishable
from internal scatter. The measurement beats the labels by a factor of fourteen.

Control A is the important one. It is Chao's own test, which they introduced
because their statistic might have worked merely by compressing sixty channels into
two well behaved numbers rather than because electrode position meant anything.
Permuting the block ordering here drops the score from 12.20 to 0.37, which is
below the level at which groups are distinguishable at all. **Depth is carrying the
information.**

Control B is the one whose correct answer was known in advance. Head indices are
arbitrary labels, so permuting them must change nothing. The largest deviation
across every input and every permutation is 1.78e-15, which is float64 rounding.

## What the measurement actually found

| End state | n | mean | spread | range |
|---|---|---|---|---|
| `prolet` | 55 | 7.37316 | 0.00222 | 7.36527 to 7.37700 |
| `Divine` | 34 | 7.41684 | 0.02294 | 7.33701 to 7.46623 |
| `till` | 19 | 7.23757 | 0.00383 | 7.23464 to 7.25136 |
| `Anarch` | 16 | 7.36609 | 0.00321 | 7.35921 to 7.36977 |
| `solidarity` | 1 | 7.25615 | n/a | single member |

The groups are extraordinarily tight. Four of the five have a spread below 0.004 on
a scale running from 0 to 11, while sitting up to 0.18 apart from each other.

**An independent confirmation of an existing claim, which was not designed for.**
Register row C-07 records that the two nearest end states in the project's own
readout are `Anarch` and `prolet`. Every pair of end states here separates above
the registered threshold, and the weakest pair, at 1.81, is `Anarch` against
`prolet`. A completely different measurement, built from activity rather than from
tokens, independently identifies the same pair as the hardest to tell apart.

| Pair | separation |
|---|---|
| `prolet` vs `till` | 33.77 |
| `Anarch` vs `till` | 25.27 |
| `Divine` vs `till` | 8.85 |
| `Divine` vs `prolet` | 3.31 |
| `Divine` vs `Anarch` | 2.41 |
| **`Anarch` vs `prolet`** | **1.81** |

## Why gate 2 failed, stated without softening the failure

The measurement does not separate fixed points from two-step cycles. It scores
1.0036, and 1.0 is the value meaning no separation at all.

The reason is a confound in the census that I should have foreseen and did not.
**All 34 of the two-step cycles are in one end state, `Divine`.** So the registered
test is not comparing dynamical classes at all. It is comparing `Divine` against a
pooled group containing `prolet`, `Anarch`, `till` and `solidarity`, whose members
span 7.238 to 7.373. That pooled spread is four times the distance from `Divine` to
the nearest of them, so the comparison is swamped by the internal structure of a
group that is not internally uniform.

**The consequence for the series, and it is a restriction rather than an excuse.**
The pre-registration says a failed gate means the measurement is discarded and the
series continues with token labels alone. That is too blunt for what happened here,
so the disposition taken is narrower and is stated explicitly so a reader can
disagree with it:

- The measurement **is** validated for separating end states, which is the use the
  rest of EXP-003 needs it for, and it passes that gate by a wide margin with the
  shuffle control confirming the mechanism.
- The measurement **is not** validated for any claim about dynamical class, and no
  such claim may be made from it anywhere in this series.

**This is a post-hoc narrowing of a registered gate and that is the dangerous
direction.** EXP-002 was criticised for a criterion too weak to fail and corrected
it by making it stricter, which is safe. Narrowing scope after seeing a failure is
the opposite move and deserves more scepticism, which is why the failure is left
standing in the table above and the pairwise numbers are labelled post hoc
throughout. The clean fix is a census in which the two classes are not perfectly
confounded with end state, and that does not exist.

## The surprise: the source's own method performs worse here

The pre-registration reduced the grid to one number per input. Verification of the
source afterwards showed that is not what Chao, Bakkum and Potter did: they kept the
whole trajectory, concatenated it into a long vector and compared vectors by plain
Euclidean distance, reaching 22,920 dimensions in simulation.

Both were computed from the same run.

| Measurement | End-state separation |
|---|---|
| The registered single number | **12.2035** |
| The same number, whitened | 12.2035 |
| The full twelve-block profile, standardised, Euclidean | **1.4303** |

The faithful method scores nine times worse and falls below the registered
threshold, though it still beats the token labels. The likely reason is that
standardising each block's contribution separately amplifies blocks whose variation
is mostly noise, and the Euclidean distance then spends most of its dimensions on
that noise, whereas the weighted centroid happens to project along the direction
where the signal lives.

The whitened version is identical to the unwhitened one to five figures, exactly as
predicted in advance: the separation ratio is invariant to any affine rescaling of a
single number, so whitening a scalar cannot change it. It was computed anyway so
the invariance is visible rather than asserted.

**What this means for the port.** Borrowing the apparatus does not mean borrowing
every choice inside it. The idea worth stealing was using the array's geometry at
all, which is what control A confirms is doing the work. The specific reduction is a
free parameter, and on this substrate the cruder one wins. That is worth recording
because the opposite assumption, that faithfulness to the source is always the safer
choice, would have led to adopting the weaker measurement.

## Limits

One model, one loop configuration, no plasticity anywhere in this stage, and the
frozen weights throughout. This validates an instrument; it measures nothing about
the collapse the series exists to study.

The single-member end state `solidarity` contributes a group centre but no internal
scatter, and the separation statistic skips such groups in its denominator rather
than treating them as having zero spread, which would inflate every ratio.

The measurement is a projection from 144 numbers to one and is not information
preserving. The source says the same of theirs: many different activity
distributions map to the same value. A null from this measurement is therefore weak
evidence, and only a positive separation should be read as informative.

## Files

- `stage0.jsonl` — every input's record, including the full per-block profile and the
  144-site grid at settle, so a different analysis needs no further model time.
- `stage0_analysis.json` — all three measurements and every control.
- `experiments/exp003_stage0.py`, `experiments/exp003_stage0_analyse.py` — the runner
  and the analysis, separated so the analysis can be corrected without re-running.
