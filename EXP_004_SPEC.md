# EXP-004: Context to weight transfer

**Status:** proposed, not run. Third revision, 2026-09-06, after three independent
fresh-context reviews of the branch (record fidelity, experimental design, pilot code) and
Codex rounds two and three on PR #65; the revision history and what each draft got wrong
are in section 9.
**Model:** GPT-2 Small, TransformerLens, layers 0 to 11.
**Register rows:** C-69, C-70, C-71, entered as `open`. Claimed on the identifier
registry by `agent:ctx-to-weight` on 2026-09-05. Rows C-64 to C-68 are claimed on other
branches and pull requests (EXP-003 on PR #59, the register reconciliation on PR #61),
which is why this experiment's rows begin at C-69.
**Cost:** Stage 1 about four hours of CPU; Stage 1b about three quarters of an hour; Stage 2 about a quarter of an hour; Stage 3 hours, most of it the gradient-trained
arm. Timings are from `experiments/output_exp004/measurements_v1.json` (`timings`): 0.32
seconds per fold on a 24-token context at a feed-forward site, 0.33 at a twelve-stripe
attention site, and 0.28 seconds per seven-token query pass, on the session's CPU with
four threads; the file's earlier runs on the same machine gave 0.34, 0.37 and 0.30. Every cost below is that arithmetic.
**Supporting measurements:** every number quoted in this file that is not from a pilot
cites `experiments/output_exp004/measurements_v1.json`, produced by
`experiments/exp004_measurements.py`; the key in that file is given in brackets.

*What will be run, in what order, and what threshold decides each result. Written before
the runs it governs. Nothing in this file enters the claim register until a stage has run
and its result has been reviewed.*

## 1. The question

Every result in this repository so far feeds the plastic site the model's own free-running
activity and reads out which basin the loop settles in. This experiment feeds the site a
context the model was given, clears the context, and asks whether anything **specific to
that context** survives in the weights, as measured by the model's next-word predictions.
The write channel is unchanged: the same `OjaPlasticity` object, the same rules, the same
`revert()`. Only the input to the channel and the readout differ.

The word "specific" carries the design. A Hebbian write at a feed-forward output site is,
by register row C-10, `E[x xᵀ]·W0 + x̄·b_outᵀ`: an amplification of whatever directions the
context activated, plus a bias term. Such a write can soften or colour the model's
predictions whatever context produced it. A reviewer's re-run of the first pilot with a
swapped-context arm found exactly that: a write made from an unrelated context scored as
well as the write made from the context being tested. So the primary comparison in every
stage is the model's own-context write against writes made from **other contexts of the
same class**, and no hypothesis is decided against random noise alone.

A second word carries the fact and format classes: **binding**. A write from "The capital
of Morvane is Tokyo" raises the token ` Tokyo` after any capital-shaped prompt, because
` Tokyo` is among the directions the context activated. Measured at `blocks.6.mlp` at six
percent drift, that write lifts ` Tokyo` by 1.31 nats on "The capital of Morvane is" and by
1.72 and 1.98 nats on "The capital of Spain is" and "The capital of Egypt is", which
never name Morvane and whose true answers are not Tokyo [`binding_vs_priming`]. A score that reads only the entity query would
count that as binding transferred. It is token priming, and the swapped-context null does
not catch it, because no other context contains the answer token. Every fact and format
score is therefore taken on the entity query **and** on control queries that share the
frame but not the entity, and the decision is on the difference.

## 2. Definitions

**Context, query, bound answer.** A context `C` is a short passage with no trailing
whitespace. A query `Q` is a short prompt beginning with a space or a newline. For the
fact and format classes the context binds an answer, a word the query is meant to elicit,
of one or more tokens.

**Tokenisation.** `C + Q` is tokenised as one string. The context's tokens must be a prefix
of the result, checked by assertion; the query tokens are the remainder. The query-alone
runs use exactly those query token ids after the beginning-of-sequence token, so the
query's tokens are identical in every run and only what precedes them differs.

**Reference.** The model's next-word distribution at each query position when run on
`C + Q` with the original weights.

**Baseline.** The same on the query alone, original weights.

**Test.** The same on the query alone after the context has been folded into the plastic
site and the site's new weight installed.

**Positional confound, stated, and not bounded.** In the reference the query sits after
the context's tokens; in the baseline and test it sits after the beginning-of-sequence
token alone. GPT-2 uses absolute position embeddings, so part of the reference-to-baseline
distance is position, not content, and no write at one site can be expected to close it.
No single control the stack offers decomposes that distance, and the second draft's
claim that a neutral filler "bounds" it was wrong: on the pilot contexts the
length-matched filler sits **further** from the reference than the query-alone baseline
for the fact and topic classes (final-position KL 12.24 against 10.11, and 2.78 against
1.84), and a position-only run, in which the query's tokens take the positions they hold
in the reference with nothing before them but the start token, sits closer for fact and
format (9.02, 4.14) and further for topic (4.72) [`position_control`]. Every stage records
both runs as **additional references**, the filler run and the position-shifted run, with
no bounding interpretation. Transfer is reported against the baseline.

**Scores, at the final query position only.** The first query position sits next to the
beginning-of-sequence token in the query-alone runs and dominated the first pilot's mean;
it is excluded from every decision.

- For the fact and format classes, the primary score is the **teacher-forced log
  probability of the whole bound answer**: the sum, over the answer's tokens, of each
  token's log probability given the query and the answer's preceding tokens, from one
  forward pass. The first token's log probability is recorded beside it. The third pilot
  scored the first token only, and for the three-token answer " GARDEN" that was the log
  probability of " G", a prefix shared by 468 vocabulary entries; scored on the whole
  answer, the write that lifted " G" by 0.31 nats lowered the whole answer by 1.7 nats
  (`pilot_v4.json`, section 9). Fact answers are one token by construction, so the two
  scores coincide there. The secondary score is the KL divergence from the reference to
  the test at the final query position.
- For the topic class, which binds no answer, the primary score is the **KL divergence at
  the final query position**, in nats, from the reference to the test.
- Every record also stores the KL divergence at every query position, so the aggregate can
  be re-derived.

**Control queries, for the fact and format classes.** Each fact context carries two
control queries in the same frame as its query that name a different, real entity **whose
true answer is not the bound answer** ("The capital of Spain is", "The capital of Egypt
is" for a fact bound to Oslo), screened so the frozen model's top prediction on each is
not the bound token. A control whose true answer is the bound token ("The capital of
Norway is" for Oslo, which the previous revision used) would let a write that merely
strengthens the real association lift the token there, and subtracting that lift would
erase genuine entity-specific transfer. Each format context
carries two control queries in the same frame with a different input word ("water ->" for
a query "garden ->"), scored on the original bound answer. A write that raises the bound
answer on the control queries as much as on the entity query has primed a token, not
bound it.

**Transfer.** Baseline score minus test score for KL, test minus baseline for log
probability, so that positive means the write moved the model toward its with-context
behaviour. The same definition applies to every arm, including the temperature-matched
baseline (control C5): its transfer is baseline score minus C5 score for KL and C5 score
minus baseline score for log probability, so that every comparison in this file is
between two transfers.

**Binding transfer, for the fact and format classes.** The write's transfer on the entity
query minus the larger of its transfers on the two control queries. The raw transfer on
the entity query is reported beside it as **priming**. On the three fact contexts measured
at `blocks.6.mlp`, no write, own or swapped, has positive binding transfer; the own
writes' values are minus 1.12, minus 0.71 and minus 0.68 nats [`binding_vs_priming`].
That is a measurement at one site and one drift, and it is why the decision cannot be
made on the entity query alone.

**Specific transfer.** For the topic class, the own-context write's transfer minus the
**maximum** of the transfers under the same-class swapped-context writes (control C4).
For the fact and format classes, the same with binding transfer in place of transfer. A
context **shows specific transfer** when this quantity is positive, that is, when the own
write beats every swapped write. With nine swapped writes, the probability of that under
no effect is exactly one in ten per context; with the three swaps Stage 1 uses for
selection it is one in four. The ten decisions in a class share one pool of ten writes,
so the ten indicators are not independent trials: the null distribution of the
**count** of contexts showing specific transfer is obtained by permuting the write labels
over the ten-by-ten matrix of scores (every context scored under every write) and
counting, over ten thousand permutations, how often seven or more contexts have their
assigned write beat the other nine. The binomial figures quoted below are the
independent-trials approximation and the permutation count is what decides. The second draft's "95th percentile of twenty-nine swaps"
gave a probability that depended on a percentile convention it never named (0.033 for
the maximum, 0.067 for the second largest, 0.078 for linear interpolation), and is
replaced by the maximum, which needs no convention.

**Drift.** `‖ΔW‖_F / ‖W0‖_F`, the site's relative weight change, from `report()`, where
`‖·‖_F` is the Frobenius norm, the square root of the sum of squared entries. For a
twelve-stripe attention site both norms are taken over the stacked stripes. Reported with
every cell with the `clipped` flag. The ceiling is set to 1.0 in every stage. The flag is
**not** vacuous: on the calibration folds below, the `oja` rule reaches the ceiling at
every step size tried at the late attention sites, and `hebb` reaches it at the block 11
stripes at step size 1e-2 [`drift_by_site`]. A clipped fold is never used to solve the
ladder.

**The fold.** With the rule installed at the site, run the model once over `C` alone, so
the rule observes the site's input and output activity at every context position. Remove
the rule. Call `apply()` once. The rule is never installed during a query pass. The
position mean the rule takes **includes the beginning-of-sequence position**, whose
activity at `blocks.6.mlp` is two and a half times the median token's (output norm 60.6
against 19 to 24) and is identical for every context: its projection onto each pilot
write accounts for 4 to 20 percent of the write's squared norm (the projection of the
rest accounts for the remainder, so the two sum to one; a squared ratio of norms, which
the previous push quoted, is not a decomposition because the two parts are not
orthogonal), and the writes' pairwise cosines fall by up to 0.10 when it is removed
[`bos_share`]. Excluding it would need a position mask the adapter does not
have, so this spec records the share per write instead and states that every write shares
that component; the swapped-context null shares it too, which is the point of that null.

**Drift ladder, not step-size ladder.** One apply is linear in eta wherever nothing clips:
the ratio of drifts at step sizes 1e-3 and 1e-2 is 10.000 to seven figures and the cosine
between the two deltas is 1.000, for `hebb` at every site kind and for `oja` where it does
not clip [`linearity`]. `oja`'s first apply is linear because its decay term reads `W0 +
delta` with `delta = 0`. So for each (context, site, rule) one **calibration fold** gives
the drift per unit eta, and the delta at each target drift is the calibration delta
rescaled, with no further fold. Targets: 0.1, 0.3, 1, 3 and 10 percent. The calibration
step size starts at 1e-4 and is divided by ten until the fold's `clipped` flag is false;
if it is still true at 1e-8 the (site, rule) pair is dropped and recorded. Drift at one
step size varies more than twenty-fold across sites: for `hebb` at 1e-2, 0.59 percent at
`blocks.6.mlp` against 10.2 percent at `blocks.11.attn.head.7` and 42 percent, clipped,
at the twelve block 11 stripes; for `oja` at 1e-4, 0.46 percent at `blocks.6.mlp` against
5.6 percent at `blocks.11.mlp` and the ceiling at every block 11 attention site
[`drift_by_site`]. The calibration fold's report is written to the record.

**The fact format, and class screening.** A fact context names an invented entity and
binds it to a **common, single-token** answer, stated verbatim ("The capital of Veltoria is
Oslo."), and its query restates the entity ("The capital of Veltoria is"). On the frozen
model, eight candidates in this format give the answer a log probability of minus 0.30 or
better with the context present, 8.6 to 13.2 nats above the no-context baseline, eight of
eight [`screening.facts`]. The second pilot's fact context introduced its entity
indirectly ("The capital of the small nation of Veltoria is a city called Marrowgate");
with it present the model gives the three-token name a whole-answer log probability of
minus 11.3, against minus 22.4 without: an eleven-nat lift to an answer the model still
does not predict, its most likely next token being ` a`. The same name stated in the
verbatim format gets minus 0.27 and is the model's top prediction, so the second pilot's
failure was the phrasing, not the name's rarity or length; the earlier drafts'
explanation, and their "1.2 nats" for that context, cited no artifact and are withdrawn.
That case is also why a gap alone cannot be the screen. **Screening rule:** a fact or
format context enters its class only if the reference gives the whole bound answer a log
probability at least 2 nats above the baseline's **and** makes the answer's first token
its most likely next token. The one-token rule for fact answers is kept so that the
first-token and whole-answer scores coincide and the control queries score one token. A
format context's answer may be several tokens; it is scored whole. On four format
candidates the gaps were 9.2, 7.5, 6.1 and, for a reverse-the-letters format, 1.1, which
fails [`screening.formats`].

**Candidate pool and split.** A pool of at least forty candidates per class is written
before any run and screened on the frozen model. The first twenty that pass, in pool
order, form the class; the pool, every pass or fail and every gap go to the run record.
Each class is then split by a fixed seed into ten **development** contexts and ten
**held-out** contexts. Topic contexts are written to one fixed token count, checked at
screening, because the loop keeps a context's length (Stage 2).

**Context classes.**

- **Fact.** As above.
- **Format.** Four demonstrations of a mechanical transformation, followed by a fifth
  input; two control queries with different fifth inputs.
- **Topic.** Three sentences on a subject, at the fixed token count, followed by a query
  that continues it.

## 3. Controls

**C0, the gate.** With `eta = 0` and `mode = "off"`, the context pass with the rule's hooks
installed must be bit-identical to the same pass without them at every entry of
`run_with_cache` and at the logits. Measured on the pilot fact context: identical at all
210 cached activations and the logits, at `blocks.6.mlp` and at the twelve block 11
stripes [`c0_gate`]; the fourth pilot runs it on every context pass. A version of C0
that compares only the query-alone runs cannot fail, because the rule is removed for
those. This is not the parent's `controls.c0_identity`, which compares loop states.

**C1, revert.** After `revert()`, the baseline must be reproduced bit-identically.

**C2, magnitude null.** At the decision cell of each stage, one hundred random writes,
each a rank-one matrix with the largest singular value matched to the own-context write's
largest singular value, from a fixed seed; for a stripe site, on the stacked matrix.
Register row C-23 retired the isotropic Frobenius-matched control for matching on the
wrong quantity. Reported as percentiles; no hypothesis is decided against it.

**C3, no rule on the query, with a detector.** The detector is the count of forward hooks
at the site's hook points (`blocks.{L}.mlp.hook_post` and `blocks.{L}.hook_mlp_out` for
a feed-forward site), read after every query pass and required to equal the count at
rest, together with the rule's `n_applied` unchanged. The rule object's own counters are
not the detector: they read zero whenever the object is not installed, and the third
pilot's assertion on them sat after `revert()`, which zeroes them, so it could not fail.
One **deliberate-leak arm** per stage installs the rule during a query pass and shows the
detector at the installed count, two hooks per stripe (two for a feed-forward site,
twenty-four for a twelve-stripe attention site), with the score bit-identical, because
the hook only accumulates, then calls `apply()` and shows the score move. The fourth pilot runs this arm
(`pilot_v4.json`, `c3_deliberate_leak`).

**C4, swapped-context writes.** The primary null. For a query from context `i`, install
the write made from context `j ≠ i` **of the same class**, rescaled to the own write's
largest singular value, with its Frobenius norm recorded, and score it on the same
queries. In Stage 1, three fixed same-class swaps per cell. In Stages 1b, 2 and 3, all
nine other same-class contexts in the held-out set. Same-class only, because a null drawn
from other classes tests class membership, not context specificity: the pilot writes'
cosines are 0.10 to 0.41 across classes and 0.82 between the two fact contexts
[`sigma1_share`]. Cross-class swaps are run in Stage 1b and reported descriptively.
Matching on the largest singular value rather than the Frobenius norm follows C-23; the
two differ by up to 1.38 times across the pilot writes, whose largest singular value
carries 65 to 89 percent of their Frobenius norm [`sigma1_share`].

**C5, temperature-matched baseline.** Scale the baseline's final-position logits so its
entropy matches the test's; for the whole-answer score, the same temperature at every
answer position. A write that only flattens or sharpens the distribution scores no better
than this. Its transfer is defined in section 2 and reported beside every cell.

**C6, control queries.** Defined in section 2; they turn transfer into binding transfer
for the fact and format classes. Not a null: a definition of the score.

## 4. Stage 1: the map, on the development set

**What runs.** Two rules (`hebb`, `oja`; `anti_hebb` is dropped, `random` is replaced by
C2). Sixteen sites: the twelve feed-forward output matrices `blocks.{0..11}.mlp`, and the
attention output matrices of blocks 2, 5, 8 and 11, each addressed as its twelve head
stripes `blocks.{L}.attn.head.{0..11}` driven together through `MultiSitePlasticity`,
which is the whole matrix (a whole-matrix attention site is not constructible on
TransformerLens; the stripe form constructs, and the block 2 and block 11 sets were
exercised for the measurements file). Five target drifts from one calibration fold per
(context, site, rule). Thirty development contexts. Three same-class swapped writes per
cell. Two control queries per fact and format context.

**Cost.** 960 calibration folds, plus re-calibrations where `oja` clips. 4,800 cells (16
sites, 2 rules, 5 drifts, 30 contexts). Per cell, four arms (own and three swaps), each
scored on one query for topic and on three queries for fact and format: 44,800 query
passes. About four hours.

**Output.** For each class, the single cell (site, rule, drift) with the highest median
specific transfer over its ten development contexts. Nothing is decided here. Stage 1 is
a map and a selection; the multiplicity of 80 cells per rule, 160 in total, means any
threshold applied here would be met by noise, as the review computed.

## 5. Stage 1b: confirmation, on the held-out set

**What runs.** At each class's selected cell only: the own-context write, the nine
same-class swapped writes, the twenty cross-class swapped writes (descriptive), one
hundred C2 writes, and C5, for each of the ten held-out contexts, on the entity query and
the two control queries for fact and format.

**Cost.** Ninety folds and about 9,200 query passes: about three quarters of an hour.

**Decision rule.** A held-out context **shows specific transfer** if its own-context
transfer (binding transfer for fact and format) exceeds all nine same-class swapped
transfers and exceeds the C5 transfer on the same scale (the C5 binding transfer for fact
and format, computed from C5 on the entity query and on the control queries by the same
subtraction), and, for the topic class, is at least 0.1 nats. Under no effect a context
does the first with probability one in ten; seven of ten by chance has probability about
nine in a million under independence, and the permutation count of section 2 is what
decides; three classes are tested. The 0.1-nat floor for topic is a pre-registered
effect size: the pilot's topic effect was 0.09 nats on a reference-to-baseline gap of
1.84, and a sign test without a floor would let an effect of that size count.

**Three outcomes, not two.** Missing seven of ten does not establish absence: six
swap-beating contexts would mean most held-out contexts transferred. Each hypothesis
below therefore has a refuting count, a supporting count with an equivalence bound, and
a zone between them recorded as **not established**.

- **H4.1, binding does not transfer.** For the fact class and for the format class.
  **Refuted** if seven or more of ten held-out contexts show specific transfer at the
  selected cell. **Supported** if at most two of ten do and the median binding transfer
  over the ten is below 0.1 nats, the equivalence bound. Three to six of ten, or a median
  at or above the bound, is not established. Register row C-69.
- **H4.2, colouring transfers.** For the topic class. **Supported** if seven or more of
  ten held-out contexts show specific transfer at the selected cell. **Refuted** if at
  most two do and the median specific transfer is below 0.1 nats. Otherwise not
  established. Register row C-69.

The Stage 1 map also reports, descriptively and without a hypothesis, whether the selected
cells fall on feed-forward or attention sites, and the priming lifts beside every binding
transfer.

**What enters the register.** C-69 takes H4.1 and H4.2 together as one row: what a
single-site, single-pass write carries that is specific to its context, by class, at the
selected cell, against same-class swapped-context writes, with binding separated from
priming.

## 6. Stage 2: the loop before the write

**What runs.** At the topic class's selected cell, on the ten held-out topic contexts,
which share one token count. The parent's loop is run over the state produced by the
context pass for `N` iterations, with `N` in 1, 10 and 100, through
`atr_bridge.make_atr_step`, never a reimplementation, as three separate trajectories.
Two write arms at each `N`: the **context write** (the fold as in Stage 1, `N = 0`), and
the **loop write**, the rule's accumulator over the `N` loop iterations only, applied
once. The two are not averaged together, because `apply()` averages over batches and an
average would dilute a context write by `1/(N+1)`. Swapped-context loop writes at every
`N` from the nine other held-out topic contexts, rescaled as in C4.

**What the loop does to the context, measured.** The injection overwrites the first
block's input wholesale, so the context's **tokens** are never read again after the
context pass, and the prompt string survives only as a sequence length (`atr_bridge.py`,
citing the parent's `docs/TECHNICAL.md`). The **state** the loop starts from is the
context's own residual stream, so whatever the loop keeps of that state is context
information, and how much of it survives is exactly what this stage measures; the
previous revision's "survives only as a sequence length" would have read a surviving
effect as impossible. Measured at
`blocks.6.mlp` with `hebb` at step size 1e-3 on two topic contexts of different length:
the loop writes from the two contexts have cosine 0.72 at one iteration and 0.96 at ten,
against 0.41 for their context writes; each loop write's cosine with its own context write
is minus 0.22 and minus 0.04 at one iteration and within 0.03 of zero at ten. At a hundred
iterations the cosine between the two loop writes falls to 0.32 [`loop_writes`]. The
second draft quoted 0.98 and cited nothing; these are the committed values. Length is
held fixed in this stage so the swapped null cannot be beaten on length alone.

- **H4.4, nothing context-specific survives the loop, at the selected cell.** At `N = 10`
  and at `N = 100`, against swapped-context loop writes at the same `N`. **Refuted** if
  seven or more of ten held-out contexts show specific transfer under the loop write at
  either `N`. **Supported** if at most two do at both `N` and the median specific transfer
  is below 0.1 nats at both. Otherwise not established. Register row C-70, which is
  restricted to this one cell:
  a cell that best writes a context directly need not be the cell that best keeps it
  through the loop, and selecting loop cells on the development set would cost a second
  map. The restriction is stated in the row.

**Why this stage exists.** Lee and colleagues fold context into weights after an offline
recurrent pass over the stored context. This project's loop runs with no stored context.
Whether a free-running loop keeps anything of the context that seeded it is a question no
published work has asked, and either answer is a result.

**Cost.** Ten contexts, three trajectories of 1, 10 and 100 iterations each, thirty loop
writes, three hundred own and swapped query passes, and C2's one hundred random writes at
each of the two deciding cells (`N = 10` and `N = 100`) for each context, two thousand
more passes, with C5 beside each: about half an hour.

## 7. Stage 3: the fidelity ladder

**What runs.** At each class's selected cell, on the held-out contexts, three write
methods at matched size, each scored by the same specific transfer:

1. **Best target-free write.** Whichever of `hebb` and `oja` Stage 1 selected. The two
   see the same activity and differ only in a brake, so they are one rung.
2. **Ridge least squares, query-blind.** Twenty probe queries per context, drawn from a
   fixed pool disjoint from the scored query. At the site, with `y = x·W + b`: regress
   the with-context output `y_ctx` on the probe-alone input `x_probe` to solve
   `ΔW = argmin ‖x_probe·(W0 + ΔW) + b − y_ctx‖² + λ‖ΔW‖²`, with the ridge weight `λ`
   chosen so that `‖ΔW‖_F` equals rung 1's. The system is about forty rows (the probe
   queries' token positions) against 3,072 inputs, which is why ridge rather than plain
   least squares. The solution's norm rises as `λ` falls and is bounded above by the
   minimum-norm solution; at `blocks.6.mlp` with twenty probes that bound was measured in
   review at about 8 percent drift, so the 10 percent rung is out of reach there. **If
   rung 1's norm exceeds the minimum-norm solution's, rung 2 is scored at the minimum-norm
   solution, the shortfall is recorded, and H4.5 is not decided for that context.** The
   write sees probe queries and never the scored one. This rung matches site outputs, not
   next-word predictions.
3. **Gradient-trained low-rank adapter, query-blind.** A rank-8 adapter on the same site,
   trained to minimise the final-position KL divergence to the reference on the twenty
   probe queries, with `ΔW` rescaled to rung 1's Frobenius norm after every step, then
   scored on the held-out query. This rung matches next-word predictions, on probe
   queries drawn from the same pool as the scored one, so rung 3 exceeding rung 1 is the
   expected price of writing with no target, not a finding; the finding is the size of
   the gap.

**Matched size, stated precisely.** All three rungs share `‖ΔW‖_F`. Their largest singular
values differ by construction (a rank-8 write at matched Frobenius norm has a smaller
largest singular value than a near-rank-one write), so `σ₁`, the largest singular value,
is reported beside every rung and the C2 distribution is run at each rung's own `σ₁`.
Reporting `σ₁` records the mismatch; it does not remove it, and C-71 says so.

- **H4.5, transfer is ordered by what the method knows.** Rung 3 exceeds rung 2, and rung
  2 exceeds rung 1, in specific transfer on seven or more of ten decidable held-out
  contexts, per class. **Refuted** if either ordering fails on seven or more. Between
  those counts the row records "not decided" with the counts. Register row C-71. The row
  records the ordering and the size of each gap; the gap between rung 1 and rung 3 is the
  price of writing with no target.

**Cost.** Rung 1 is free from Stage 1b. Rung 2 is one ridge solve per context. Rung 3 is
minutes of training per context on CPU, so hours in total.

## 8. Reporting

Every cell writes one record: context id, class and split, site, rule, calibration step
size and the calibration fold's report, target and realised drift, `clipped`, non-finite,
`N`, the full per-position KL for the baseline, the filler reference, the position-shifted
reference and the test, the primary and secondary scores with the first-token and
whole-answer log probabilities where an answer is bound, the transfers on the control
queries and the binding transfer, every swapped write's transfer with its `σ₁` and
Frobenius norm, the C2 percentiles, the C5 transfer, the beginning-of-sequence share of
the write, and the C3 detector reading. The run record is
`experiments/output_exp004/exp004.jsonl`. The results file states measurements only, in
the form of the EXP-003 stage files, and cites the record. The claim register is updated
in a separate pull request, per the register's rule that a status change is a bigger
event than a measurement.

## 9. Pilots and revision history

**First pilot, 2026-09-05, superseded.** One scratch run at `blocks.6.mlp`, three contexts,
mean KL over all query positions, an isotropic random control with one unseeded draw.
Recorded as `experiments/output_exp004/pilot_v1.json`; its script is at commit 7288f32.
Two independent reviews found that it tokenised the context and query separately, so the
reference was not the model on the joined text; that its one positive result, a fall in
mean KL on the topic context, sat entirely at the first query position and reversed at
the final one; that its fact query asked for the country rather than the bound word; and
that its random control was the one C-23 retired. The topic result is **withdrawn**.

**Second and third pilots, 2026-09-05.** The same measurement with joint tokenisation,
scores at the final position, the bound token recorded, seeded rank-one random writes
matched on `σ₁`, swapped-context writes, and the temperature-matched baseline. The second
run's fact context introduced its entity indirectly and the model did not produce the
name on the entity query; it is recorded as `pilot_v2.json` and its fact rows say nothing
about binding. The third run replaced that context with one in the fact format above and
is recorded as `pilot_v3.json`. They are pilots: one site, one seed set, one context per
class, and the operator has ruled they are to be given little weight.

**Fourth pilot, 2026-09-06.** The third run's measurement with the whole bound answer
scored teacher-forced beside its first token, the C3 detector and deliberate-leak arm of
section 3, the C0 gate on every context pass, the random arm as percentiles, and a
convergence assertion on the temperature solver; the own-write, random and
temperature-matched arms at the final position are computed exactly as in the third run.
Recorded as `pilot_v4.json`, script `experiments/exp004_pilot.py`, results and limits in
`PILOT.md`. Its whole-answer scores are what retired the first-token metric.

**What the first draft of this file got wrong**, for the record: it decided hypotheses on
a 95th percentile of ten random draws across 384 scanned cells, which the review computed
would refute H4.1 and support H4.2 from noise alone; it named attention sites the adapter
cannot construct; it selected and tested Stage 2 on the same contexts; it promised matched
drift across rules without saying how; it undercounted the control cost five-fold; and it
attributed to the parent a "position-uniform by iteration 10" finding that is not in the
parent's record.

**What the third revision's first push got wrong**, found by Codex round four on
2026-09-06 and fixed the same day: its control queries could name the real entity whose
capital is the bound answer, so a strengthened real association would have been
subtracted as priming; its C5 comparison for fact and format put a raw transfer against
a binding transfer; its hypotheses read a miss of seven of ten as support for absence; its
seven-of-ten tail assumed ten independent trials where the ten decisions share one write
pool; its leak detector expected two hooks at a twelve-stripe site that installs
twenty-four; its Stage 2 cost omitted C2; its beginning-of-sequence share was a squared
ratio that is not a decomposition; and it said the context "survives only as a sequence
length" through the loop, which would have read a surviving effect as impossible.

**What the second draft got wrong**, found by three independent reviews and Codex rounds
two and three: its fact-class score could not tell binding from token priming, and on one
of three contexts in its own format a write that binds nothing would have passed its
rule; its primary null drew most of its swaps from other classes, so it tested class
membership; it rescaled swaps on the Frobenius norm, the matching C-23 retired; its null
probability depended on an unnamed percentile convention; it scored the first token of a
multi-token answer, and its one-token screening rule would have emptied the format class
as the pilot defined it; its C3 check could not fail; its "clipped cannot fire" was false
for the calibration folds, and its `oja` calibration at 1e-3 would have clipped at the
late sites; its ridge rung's top drift was unreachable and its row count was ten times too
high; it called a neutral filler a bound on the positional confound, which it measurably
is not; it counted "160 cells per rule" where there are 80, "thirty folds" in Stage 1b
where there are ninety, and "111 loop iterations" as an hour; and it quoted six numbers
"measured in review" that cited no artifact, of which one, the "1.2 nats for Marrowgate",
could not be reproduced and is withdrawn, and the rest are now in `measurements_v1.json`.
