# EXP-004 pilots

*Four scratch runs of the Stage 1 measurement, all at `blocks.6.mlp`, one context per
class, no loop. Run artifacts: `pilot_v1.json` (2026-09-05, superseded), `pilot_v2.json`
(2026-09-05, its fact context failed screening), `pilot_v3.json` (2026-09-05, first-token
scoring), `pilot_v4.json` (2026-09-06, the run reported below). Script for the fourth:
`experiments/exp004_pilot.py`; the third's script is at commit 76a4eb9, the second's at
362c3ca, the first's at 7288f32. The measurements the spec cites beside these runs are in
`measurements_v1.json`, from `experiments/exp004_measurements.py`. The operator has ruled
that these runs are to be given little weight. Nothing here enters the claim register.*

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
got worse. `pilot_v1.json` holds no per-position values; the per-position reading was a
reviewer's re-measurement with the first script's logic, not a committed number. That
reading is withdrawn. The file is kept as the record of what ran.

## The second pilot, and the fact context that failed screening

The second run fixed the tokenisation, scored at the final position, recorded the bound
token, and added the swapped-context, rank-one random and temperature-matched controls.
Its fact context read "The capital of the small nation of Veltoria is a city called
Marrowgate. Marrowgate sits on the river Oss and is famous for its glass bridges." With
that context present, GPT-2 Small gave the bound token ` Mar` a log probability of minus
11.32, below the minus 8.95 it gave with no context at all, and predicted ` a` as the next
word. `pilot_v2.json` is kept as the record; its fact rows say nothing about binding.

Two earlier explanations of that failure are withdrawn. The second and third drafts said
the model "does not reproduce a rare multi-token name in context" and quoted a screening
of eight candidates, "2.5 to 13.9 nats" on "seven of seven" against "1.2 nats" for
Marrowgate; that screening was never committed and the 1.2 cannot be reproduced. The
committed screening (`measurements_v1.json`, `screening.facts`) shows the cause was the
phrasing: the same name stated in the spec's verbatim format, "The capital of Veltoria is
Marrowgate.", gets a whole-answer log probability of minus 0.27 with the context present
and is the model's top prediction, while the second pilot's indirect phrasing gets minus
11.3. Eight candidates in the verbatim format with common single-token answers all pass,
with gaps of 8.6 to 13.2 nats over the no-context baseline.

Between the second run and the third, the fact context changed and nothing else. For the
format and topic contexts the own-write, random and temperature-matched arms, and the swap
from the other unchanged context, are bit-identical between `pilot_v2.json` and
`pilot_v3.json`; the swap-from-fact column differs at every step size because it is a
different write. Under the second pilot's fact write, the format context's bound first
token rose by 0.48 nats at the largest step size, more than under the own write; that
column is in `pilot_v2.json` and is one reason the format reading below is stated at the
scale it is.

## The third and fourth pilots

**Limits, stated first.** One site. One forward pass over each context, then one apply.
One context per class. Ten seeded rank-one random writes per cell, the same ten directions
for every context at a given step size. No loop. The ceiling was set to 1.0 and did not
fire at this site. Run once on CPU: 79 seconds for the third, 90 seconds for the fourth,
which adds one forward pass per arm for the whole-answer score and the gate and leak arms.

**What changed between them.** The third run scored the bound answer on its first token.
For the format context the bound answer " GARDEN" is three tokens, ` G`, `ARD`, `EN`, so
the third run's format score was the log probability of ` G`, a prefix shared by 468
vocabulary entries. The fourth run scores the whole answer teacher-forced, one forward
pass on the query followed by the answer's tokens, summing each answer token's log
probability given the gold tokens before it, and records the first token beside it. The
third run's C3 check was an assertion on the rule's private counters placed after
`revert()`, which zeroes those counters, so it could not fail; the fourth run counts
forward hooks at the site's two hook points after every query pass, and runs a
deliberate-leak arm per context that installs the rule during a query pass, shows two
hooks detected and the score bit-identical (the hook only accumulates), then applies and
shows the score move. The fourth run also runs the C0 gate on every context pass (hooks
installed at step size zero, mode off, bit-identical to the pass without hooks at all
210 cached activations and the logits), reports the random arm as percentiles, and
asserts the temperature solver converged. Every arm the two runs compute the same way is
bit-identical between `pilot_v3.json` and `pilot_v4.json` (138 fields compared).

**What was done.** Context and query tokenised jointly, with the context's tokens checked
to be a prefix. Reference: the model on context plus query. Baseline: the query alone
after the beginning-of-sequence token, with the same query token ids. Test: the query
alone after one Hebbian write from a pass over the context, the rule removed before the
query pass. Controls at each step size: the other two contexts' writes rescaled to the own
write's Frobenius norm (swap; the spec now matches on the largest singular value, and each
swap's ratio of largest singular values to the own write is recorded); ten rank-one random
writes matched on the largest singular value (rand); and the baseline rescaled in
temperature to the test's final-position entropy (temp), with a separate temperature solved
at every answer position for the whole-answer score. Scores at the final query position only: KL divergence from
the reference in nats, and, where the context binds an answer, the log probability of the
whole answer and of its first token.

**Contexts.** Fact: "The capital of Veltoria is Oslo. Oslo lies on a fjord and is known
for its museums." Query: "The capital of Veltoria is". Bound answer ` Oslo`, one token,
which the reference makes its most likely next word at log probability minus 0.10 against
minus 11.17 with no context. Format: four lines each giving a word and the same word in
capitals; query "garden ->"; bound answer " GARDEN", three tokens, whole-answer log
probability minus 1.67 in the reference against minus 10.90 with no context, first token
minus 1.05 against minus 6.54. Topic: three sentences about a reactor; query "The
engineers checked the"; no bound answer.

**Log probability of the whole bound answer, higher is better.** For the fact context this
equals the first-token score. Random is the best of the ten draws.

| Context | Drift | Own write | Swap from fact | Swap from format | Swap from topic | Rank-one random, best of 10 | Temperature-matched |
|---|---|---|---|---|---|---|---|
| fact (baseline minus 11.17) | 0.6% | minus 11.43 | | minus 11.12 | minus 11.06 | minus 11.17 | minus 11.22 |
| fact | 1.8% | minus 11.77 | | minus 10.99 | minus 10.83 | minus 11.12 | minus 11.27 |
| fact | 5.9% | minus 11.88 | | minus 10.58 | minus 10.27 | minus 11.13 | minus 11.24 |
| fact | 17.7% | minus 12.60 | | minus 10.53 | minus 11.19 | minus 10.89 | minus 11.15 |
| format (baseline minus 10.90) | 0.4% | minus 10.77 | minus 10.94 | | minus 11.00 | minus 10.90 | minus 10.92 |
| format | 1.2% | minus 10.55 | minus 11.02 | | minus 11.20 | minus 10.88 | minus 10.97 |
| format | 4.1% | minus 10.27 | minus 11.34 | | minus 11.88 | minus 10.88 | minus 11.03 |
| format | 12.2% | minus 12.61 | minus 12.32 | | minus 14.51 | minus 10.81 | minus 11.33 |

**Log probability of the first token of the bound answer, the third run's score, for the
format context.** Per-token values for the own write at 12.2 percent: ` G` minus 6.22,
`ARD` minus 5.21, `EN` minus 1.18, against the baseline's minus 6.54, minus 4.29, minus
0.07: the first-token gain is paid for twice over on the tokens that follow.

| Context | Drift | Own write | Swap from fact | Swap from topic | Rank-one random, best of 10 | Temperature-matched |
|---|---|---|---|---|---|---|
| format (baseline minus 6.54) | 1.2% | minus 6.54 | minus 6.50 | minus 6.58 | minus 6.53 | minus 6.55 |
| format | 4.1% | minus 6.44 | minus 6.46 | minus 6.65 | minus 6.52 | minus 6.55 |
| format | 12.2% | minus 6.22 | minus 6.47 | minus 7.04 | minus 6.50 | minus 6.53 |

**Final-position KL divergence, nats, lower is closer to the reference.** Random is the
median of the ten draws.

| Context | Baseline | Drift | Own write | Swap from fact | Swap from format | Swap from topic | Rank-one random, median of 10 | Temperature-matched |
|---|---|---|---|---|---|---|---|---|
| fact | 10.110 | 0.6% | 10.353 | | 10.054 | 10.000 | 10.112 | 10.153 |
| fact | | 1.8% | 10.672 | | 9.937 | 9.781 | 10.109 | 10.205 |
| fact | | 5.9% | 10.773 | | 9.553 | 9.252 | 10.131 | 10.172 |
| fact | | 17.7% | 11.460 | | 9.532 | 10.143 | 10.072 | 10.088 |
| format | 4.978 | 0.4% | 4.949 | 4.963 | | 4.988 | 4.979 | 4.973 |
| format | | 1.2% | 4.881 | 4.938 | | 5.005 | 4.979 | 4.964 |
| format | | 4.1% | 4.597 | 4.876 | | 5.052 | 4.977 | 4.958 |
| format | | 12.2% | 4.001 | 4.751 | | 5.341 | 4.982 | 4.992 |
| topic | 1.839 | 0.4% | 1.826 | 1.841 | 1.845 | | 1.839 | 1.839 |
| topic | | 1.1% | 1.802 | 1.845 | 1.859 | | 1.839 | 1.840 |
| topic | | 3.7% | 1.750 | 1.876 | 1.942 | | 1.839 | 1.841 |
| topic | | 11.1% | 1.829 | 2.081 | 2.382 | | 1.839 | 1.846 |

The two smallest step sizes, at drift below 0.1 percent, moved no score by more than
0.03 and are omitted from the tables; they are in the JSON. The random medians here are
the proper median of ten (the mean of the fifth and sixth values); the third run's field
of the same name was the fifth smallest, and both are in the fourth run's JSON.

**Per-position baseline KL, first query position to last.** Fact: 2.88, 0.55, 3.89, 8.03,
4.77, 1.73, 10.11. Format: 4.34, 1.00, 8.61, 4.98. Topic: 4.48, 1.53, 0.56, 1.84. The
first position is large in every case, which is the reason the mean over positions is no
longer a score.

**Gate and detector.** C0 bit-identical on all three context passes. The leak arm detected
two hooks during the query pass on all three contexts, with the score unchanged to the
last digit until `apply()` and moved after it (fact: minus 11.1748 to minus 11.1811 at 7.7
percent drift). `c0_gate_bit_identical` and `c3_deliberate_leak` in the JSON.

## Reading, marked as a pilot reading

On the fact context, where the model binds the answer in context with a gap of 11 nats,
the own-context write moves the bound token the wrong way at every drift, from minus
11.17 to minus 11.88 at 5.9 percent and minus 12.60 at 17.7 percent. At drifts up to 5.9
percent the writes from the other two contexts move it the right way, to minus 10.58 and
minus 10.27, and neither the random nor the temperature control moves it by more than
0.04 nats; the same holds on KL, where the own write raises it and the swapped writes
lower it. At 17.7 percent the pattern breaks: the swap from the topic context also moves
the token the wrong way (minus 11.19, KL 10.143 against 10.110) and the best of ten
random writes moves it the right way by 0.29 nats. So at drifts up to 5.9 percent a
single Hebbian write at this site, made from a context the model can bind in context,
carries no binding and is worse for the bound answer than a write made from an unrelated
context. A further measurement on three fact contexts in the same format
(`measurements_v1.json`, `binding_vs_priming`) shows why the entity-query score alone
cannot settle binding: a write from "The capital of Morvane is Tokyo" lifts ` Tokyo` by
1.31 nats on its own query and by 1.59 and 1.46 nats on seven-token queries naming Mozambique
and Tajikistan, whose true answers are not Tokyo.

On the format context the reading depends on the score. On the whole answer, the own
write is the best arm at 0.4, 1.2 and 4.1 percent drift, reaching minus 10.27 at 4.1
percent against the baseline's minus 10.90, the swapped writes' minus 11.34 and minus
11.88, and the best random draw's minus 10.88; at 12.2 percent it falls to minus 12.61,
worse than the baseline and worse than the swap from the fact context. On KL the own
write is below both swapped writes at every drift, by 0.75 and 1.34 nats at 12.2 percent.
On the first token alone, the third run's score, the own write gains 0.31 nats at 12.2
percent; the whole-answer score shows that gain is paid for on the following tokens.
So there is a context-specific movement toward the reference at drifts up to 4.1 percent,
of a size that leaves the whole answer 8.6 nats short of the reference, and it reverses
at 12.2 percent.

On the topic context the own write lowers the final-position KL by 0.09 nats at 3.7
percent drift, where both swapped writes raise it and neither random nor temperature
moves it. At 11 percent the own write holds near baseline while the swapped writes
degrade by 0.2 to 0.5 nats. The spec's pre-registered floor for the topic class is 0.1
nats; this effect is below it.

All of this is one context per class at one site, and the binding-against-priming
measurement shows that a second fact context can turn the sign of the entity-query
score. Any of it could change with a second context or a second site.
