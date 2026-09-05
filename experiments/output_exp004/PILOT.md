# EXP-004 pilots

*Two scratch runs of the Stage 1 measurement, both made 2026-09-05, both at
`blocks.6.mlp`, one context per class, no loop. Run artifacts: `pilot_v1.json`
(superseded) and `pilot_v2.json`. Script for the second: `experiments/exp004_pilot.py`;
the first's script is at commit 7288f32. The operator has ruled that these runs are to be
given little weight. Nothing here enters the claim register.*

## The first pilot, superseded

The first run tokenised the context and the query separately and joined the token lists,
so for the two contexts ending in a space the reference was the model on a token sequence
GPT-2 would not produce for the joined text. It scored the mean KL divergence over all
query positions. It compared against a Gaussian write matched on Frobenius norm, the
control register row C-23 retired, drawn once from an unseeded generator. Its fact query
asked "Tourists who visit Veltoria usually go first to its capital," and the reference's
most likely next token was ` V`, the country, not the bound city.

Its one positive reading, a fall in mean KL on the topic context from 1.918 to 1.836, was
examined in review position by position: the fall sat at the first query position, which
follows the beginning-of-sequence token in the query-alone run, and the final position
got worse, 1.73 to 1.80. That reading is withdrawn. The file `pilot_v1.json` is kept as
the record of what ran; an independent reproduction of its fact row at eta 0.1 matched it
bit for bit.

## The second pilot

**Limits, stated first.** One site. One forward pass over each context, then one apply.
One context per class. Ten seeded rank-one random writes per cell. No loop. The ceiling
was set to 1.0, so its "clipped" flag is vacuous. Run once on CPU, 85 seconds.

**What was done.** Context and query tokenised jointly, with the context's tokens checked
to be a prefix. Reference: the model on context plus query. Baseline: the query alone
after the beginning-of-sequence token, with the same query token ids. Test: the query
alone after one Hebbian write from a pass over the context, the rule removed before the
query pass and its accumulator checked empty afterwards. Controls at each step size: the
other two contexts' writes rescaled to the own write's Frobenius norm (swap); ten rank-one
random writes matched on the largest singular value (rand, median shown); and the baseline
rescaled in temperature to the test's final-position entropy (temp). Scores at the final
query position only: KL divergence from the reference in nats, and, where the context
binds an answer, the log probability of its first token.

**Screening result.** The fact context fails the screening rule the spec now carries. With
the context present, GPT-2 Small gives the bound token ` Mar` (of "Marrowgate") a log
probability of minus 11.32, below the minus 8.95 it gives with no context at all, and its
most likely next token is ` a`. The model does not perform this binding in context, so
no write could be expected to carry it. The fact rows below are reported for the record
and mean nothing about binding.

**Final-position KL divergence, nats, lower is closer to the reference.**

| Context | Baseline | Drift | Own write | Swap from fact | Swap from format | Swap from topic | Rank-one random, median of 10 | Temperature-matched |
|---|---|---|---|---|---|---|---|---|
| fact | 2.503 | 0.5% | 2.450 | | 2.470 | 2.493 | 2.503 | 2.480 |
| fact | | 1.6% | 2.382 | | 2.404 | 2.461 | 2.500 | 2.456 |
| fact | | 5.2% | 2.333 | | 2.176 | 2.270 | 2.501 | 2.482 |
| fact | | 15.5% | 2.407 | | 2.064 | 1.738 | 2.497 | 2.586 |
| format | 4.978 | 0.4% | 4.949 | 4.964 | | 4.988 | 4.978 | 4.973 |
| format | | 1.2% | 4.881 | 4.937 | | 5.005 | 4.978 | 4.964 |
| format | | 4.1% | 4.597 | 4.844 | | 5.052 | 4.977 | 4.958 |
| format | | 12.2% | 4.001 | 4.514 | | 5.341 | 4.981 | 4.992 |
| topic | 1.839 | 0.4% | 1.826 | 1.838 | 1.845 | | 1.839 | 1.839 |
| topic | | 1.1% | 1.802 | 1.839 | 1.859 | | 1.839 | 1.840 |
| topic | | 3.7% | 1.750 | 1.872 | 1.942 | | 1.839 | 1.841 |
| topic | | 11.1% | 1.829 | 2.197 | 2.382 | | 1.838 | 1.846 |

The two smallest step sizes, at drift below 0.05 percent, moved no score by more than
0.003 and are omitted from the table; they are in the JSON.

**Log probability of the bound token at the final position, higher is better.**

| Context | Reference | Baseline | Drift | Own write | Best swap | Rank-one random, best of 10 | Temperature-matched |
|---|---|---|---|---|---|---|---|
| fact (fails screening) | minus 11.32 | minus 8.95 | 5.2% | minus 9.49 | minus 9.19 | minus 8.93 | minus 8.97 |
| format | minus 1.05 | minus 6.54 | 4.1% | minus 6.44 | minus 6.35 | minus 6.51 | minus 6.55 |
| format | | | 12.2% | minus 6.22 | minus 6.06 | minus 6.50 | minus 6.53 |

**Per-position baseline KL, first query position to last.** Fact: 3.02, 0.46, 1.44, 8.12,
4.75, 1.71, 2.50. Format: 4.34, 1.00, 8.61, 4.98. Topic: 4.48, 1.53, 0.56, 1.84. The first
position is large in every case, which is the reason the mean over positions is no longer
a score.

## Reading, marked as a pilot reading

On the topic context the own write lowers the final-position KL by 0.09 nats at 3.7
percent drift, 1.839 to 1.750, where both swapped writes raise it, the random writes do
not move it, and the temperature-matched baseline does not move it. That is the one
own-context-specific signal in the run, and it is one context, one site, one seed set.
At 11 percent drift the own write holds near baseline while the swapped writes degrade
by 0.4 to 0.5 nats.

On the format context the own write lowers the final-position KL below both swapped
writes at every drift, by 0.5 and 1.3 nats at 12 percent, but the bound token's log
probability rises less under the own write than under the swap from the fact context,
minus 6.22 against minus 6.06. The distribution moves in shape toward the reference; the
bound answer is not specifically recovered. The reference itself has the bound token as
its most likely next word at minus 1.05, so the target was reachable in principle.

On the fact context the write lowers the bound token's probability, and swapped writes
beat the own write on KL; the context failed screening, so this says nothing about
binding.

Any of this could change with a second context per class or a second site.
