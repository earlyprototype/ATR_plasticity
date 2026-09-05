# EXP-004 pilots

*Three scratch runs of the Stage 1 measurement, all made 2026-09-05, all at
`blocks.6.mlp`, one context per class, no loop. Run artifacts: `pilot_v1.json`
(superseded), `pilot_v2.json` (its fact context failed screening), `pilot_v3.json` (the
run reported below). Script for the second and third: `experiments/exp004_pilot.py`; the
first's script is at commit 7288f32. The operator has ruled that these runs are to be
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

## The second pilot, and the fact context that failed screening

The second run fixed the tokenisation, scored at the final position, recorded the bound
token, and added the swapped-context, rank-one random and temperature-matched controls.
Its fact context bound the invented country Veltoria to the invented city "Marrowgate".
With that context present, GPT-2 Small gave the bound token ` Mar` a log probability of
minus 11.32, below the minus 8.95 it gave with no context at all, and predicted ` a` as
the next word. The model does not reproduce a rare multi-token name in context, so no
write could carry it. `pilot_v2.json` is kept as the record; its fact rows say nothing
about binding, and its format and topic rows are reproduced in the third run below.

Screening eight candidate fact formats on the frozen model gave the answer a log
probability 2.5 to 13.9 nats above the no-context baseline for the seven that bound an
invented entity to a common single-token answer, and 1.2 nats for "Marrowgate". The spec
now requires that format and that screening.

## The third pilot

**Limits, stated first.** One site. One forward pass over each context, then one apply.
One context per class. Ten seeded rank-one random writes per cell. No loop. The ceiling
was set to 1.0, so its "clipped" flag is vacuous. Run once on CPU, 79 seconds.

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

**Contexts.** Fact: "The capital of Veltoria is Oslo. Oslo lies on a fjord and is known
for its museums." Query: "The capital of Veltoria is". Bound token ` Oslo`, which the
reference makes its most likely next word at log probability minus 0.10 against minus
11.17 with no context. Format: four lines each giving a word and the same word in
capitals; query "garden ->"; bound token ` G`, the reference's most likely next word at
minus 1.05 against minus 6.54. Topic: three sentences about a reactor; query "The
engineers checked the"; no bound answer.

**Log probability of the bound token at the final position, higher is better.**

| Context | Drift | Own write | Swap from format | Swap from topic | Swap from fact | Rank-one random, best of 10 | Temperature-matched |
|---|---|---|---|---|---|---|---|
| fact (baseline minus 11.17) | 0.6% | minus 11.43 | minus 11.12 | minus 11.06 | | minus 11.17 | minus 11.22 |
| fact | 1.8% | minus 11.77 | minus 10.99 | minus 10.83 | | minus 11.12 | minus 11.27 |
| fact | 5.9% | minus 11.88 | minus 10.58 | minus 10.27 | | minus 11.13 | minus 11.24 |
| fact | 17.7% | minus 12.60 | minus 10.53 | minus 11.19 | | minus 10.89 | minus 11.15 |
| format (baseline minus 6.54) | 1.2% | minus 6.54 | | minus 6.58 | minus 6.50 | minus 6.53 | minus 6.55 |
| format | 4.1% | minus 6.44 | | minus 6.65 | minus 6.46 | minus 6.51 | minus 6.55 |
| format | 12.2% | minus 6.22 | | minus 7.04 | minus 6.47 | minus 6.50 | minus 6.53 |

**Final-position KL divergence, nats, lower is closer to the reference.**

| Context | Baseline | Drift | Own write | Swap from fact | Swap from format | Swap from topic | Rank-one random, median of 10 | Temperature-matched |
|---|---|---|---|---|---|---|---|---|
| fact | 10.110 | 0.6% | 10.353 | | 10.054 | 10.000 | 10.111 | 10.153 |
| fact | | 1.8% | 10.672 | | 9.937 | 9.781 | 10.104 | 10.205 |
| fact | | 5.9% | 10.773 | | 9.553 | 9.252 | 10.130 | 10.172 |
| fact | | 17.7% | 11.460 | | 9.532 | 10.143 | 10.067 | 10.088 |
| format | 4.978 | 0.4% | 4.949 | 4.963 | | 4.988 | 4.978 | 4.973 |
| format | | 1.2% | 4.881 | 4.938 | | 5.005 | 4.978 | 4.964 |
| format | | 4.1% | 4.597 | 4.876 | | 5.052 | 4.977 | 4.958 |
| format | | 12.2% | 4.001 | 4.751 | | 5.341 | 4.981 | 4.992 |
| topic | 1.839 | 0.4% | 1.826 | 1.841 | 1.845 | | 1.839 | 1.839 |
| topic | | 1.1% | 1.802 | 1.845 | 1.859 | | 1.839 | 1.840 |
| topic | | 3.7% | 1.750 | 1.876 | 1.942 | | 1.839 | 1.841 |
| topic | | 11.1% | 1.829 | 2.081 | 2.382 | | 1.838 | 1.846 |

The two smallest step sizes, at drift below 0.1 percent, moved no score by more than
0.03 and are omitted from the tables; they are in the JSON.

**Per-position baseline KL, first query position to last.** Fact: 2.88, 0.55, 3.89, 8.03,
4.77, 1.73, 10.11. Format: 4.34, 1.00, 8.61, 4.98. Topic: 4.48, 1.53, 0.56, 1.84. The
first position is large in every case, which is the reason the mean over positions is no
longer a score.

## Reading, marked as a pilot reading

On the fact context, where the model binds the answer in context with a gap of 11 nats,
the own-context write moves the bound token the wrong way at every drift: minus 11.17 to
minus 11.88 at 5.9 percent, while the writes from the other two contexts move it the
right way, to minus 10.58 and minus 10.27, and the random and temperature controls do not
move it. The same holds on KL: the own write raises it, the swapped writes lower it. So
a single Hebbian write at this site, made from a context the model can bind in context,
carries no binding and is worse for the bound answer than a write made from an unrelated
context.

On the format context the own write lowers the final-position KL below both swapped
writes at every drift, by 0.75 and 1.34 nats at 12 percent, and raises the bound token's
log probability by 0.32 nats where the swapped writes lower it or leave it. That is a
context-specific movement toward the reference on the one format context tested, of a
size that leaves the bound token 5 nats short of the reference.

On the topic context the own write lowers the final-position KL by 0.09 nats at 3.7
percent drift, where both swapped writes raise it and neither random nor temperature
moves it. At 11 percent the own write holds near baseline while the swapped writes
degrade by 0.2 to 0.5 nats.

Any of this could change with a second context per class or a second site.
