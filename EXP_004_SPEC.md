# EXP-004: Context to weight transfer

**Status:** proposed, not run. Revised 2026-09-05 after two independent reviews and a
Codex review of the first draft (PR #65); the revision history is in section 9.
**Model:** GPT-2 Small, TransformerLens, layers 0 to 11.
**Register rows:** C-69, C-70, C-71, entered as `open`. Claimed on the identifier
registry by `agent:ctx-to-weight` on 2026-09-05. Rows C-64 to C-68 are claimed on other
branches and pull requests (EXP-003 on PR #59, the register reconciliation on PR #61),
which is why this experiment's rows begin at C-69.
**Cost:** Stage 1 about three hours of CPU; Stage 1b about half an hour; Stage 2 about an
hour; Stage 3 hours, most of it the gradient-trained arm. Timings measured in this
session: 0.66 seconds per fold (one context pass with the rule observing, one apply) and
0.3 seconds per query pass, on this session's CPU.

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
predictions whatever context produced it. The first pilot of this measurement (section 9)
found exactly that: a write made from an unrelated context scored as well as the write
made from the context being tested. So the primary comparison in every stage is the
model's own-context write against writes made from **other** contexts, and no hypothesis
is decided against random noise alone.

## 2. Definitions

**Context, query, bound answer.** A context `C` is a short passage with no trailing
whitespace. A query `Q` is a short prompt beginning with a space or a newline. For the
fact and format classes the context binds an answer, a word the query is meant to elicit;
its first token is the **bound token**.

**Tokenisation.** `C + Q` is tokenised as one string. The context's tokens must be a prefix
of the result, checked by assertion; the query tokens are the remainder. The query-alone
runs use exactly those query token ids after the beginning-of-sequence token, so the
query's tokens are identical in every run and only what precedes them differs.

**Reference.** The model's next-word distribution at each query position when run on
`C + Q` with the original weights.

**Baseline.** The same on the query alone, original weights.

**Test.** The same on the query alone after the context has been folded into the plastic
site and the site's new weight installed.

**Positional confound, stated.** In the reference the query sits after the context's
tokens; in the baseline and test it sits after the beginning-of-sequence token alone.
GPT-2 uses absolute position embeddings, so part of the reference-to-baseline distance is
position, not content, and no write at one site can be expected to close it. The
achievable floor is unknown. Every stage therefore also records a **length-matched neutral
reference**: the model run on a neutral filler passage of the same token length as `C`
followed by `Q`. Transfer is reported against the baseline; the neutral reference bounds
how much of the baseline distance is not content.

**Scores, at the final query position only.** The first query position sits next to the
beginning-of-sequence token in the query-alone runs and dominated the first pilot's mean;
it is excluded from every decision.

- For the fact and format classes, the primary score is the **log probability of the bound
  token** at the final query position. The secondary score is the KL divergence from the
  reference to the test at that position.
- For the topic class, which binds no answer, the primary score is the **KL divergence at
  the final query position**, in nats, from the reference to the test.
- Every record also stores the KL divergence at every query position, so the aggregate can
  be re-derived.

**Transfer.** Baseline score minus test score for KL, test minus baseline for log
probability, so that positive means the write moved the model toward its with-context
behaviour.

**Specific transfer.** The own-context write's transfer minus the 95th percentile of the
transfer distribution under swapped-context writes (control C4 below). This is the
quantity every hypothesis is stated in.

**Drift.** `‖ΔW‖_F / ‖W0‖_F`, the site's relative weight change, from `report()`, where
`‖·‖_F` is the Frobenius norm, the square root of the sum of squared entries. Reported
with every cell with the `clipped` flag. The ceiling is set to 1.0 in every stage, so
"clipped" cannot fire and is reported as vacuous, per the standing prohibition.

**The fold.** With the rule installed at the site, run the model once over `C` alone, so
the rule observes the site's input and output activity at every context position. Remove
the rule. Call `apply()` once. The rule is never installed during a query pass.

**Drift ladder, not step-size ladder.** One apply is linear in eta, so for each (context,
site, rule) one observation at eta = 1e-3 gives the drift per unit eta, and the eta that
lands each target drift is solved from it. Targets: 0.1, 0.3, 1, 3 and 10 percent. Drift
at the same eta varies more than twenty-fold across sites (measured in review: 0.52
percent at `blocks.6.mlp` against 10.96 percent at `blocks.11.attn.head.7` at eta 1e-2),
so a shared eta ladder would put some sites out of range in both directions.

**The fact format, and class screening.** A fact context names an invented entity and binds
it to a **common, single-token** answer, stated verbatim ("The capital of Veltoria is
Oslo."), and its query restates the entity ("The capital of Veltoria is"). GPT-2 Small
binds this format in context: on the frozen model, screening eight candidates gave the
answer a log probability 2.5 to 13.9 nats above the no-context baseline for the seven
that used a common single-token answer, and 1.2 nats for the one that used a rare
multi-token name ("Marrowgate"), which the model does not reproduce even with the
context present. So every fact or format context enters its class only if the reference
gives the bound token a log probability at least 2 nats above the baseline's at the final
query position, and the bound answer must be one token. Screening runs on the frozen
model before any fold, and the screened set with its gaps is written into the run record.

**Context classes.** Three, twenty contexts each, of which ten are the **development set**
and ten the **held-out set**, split by a fixed seed before any run:

- **Fact.** An invented entity bound to a common single-token attribute, in the format above, followed by a query that restates the entity.
- **Format.** Four demonstrations of a mechanical transformation, followed by a fifth
  input.
- **Topic.** Three sentences on a subject, followed by a query that continues it.

## 3. Controls

**C0, the gate.** With `eta = 0` and `mode = "off"`, the context pass with the rule's hooks
installed must produce activations bit-identical to the same pass without them, at every
hook point the parent's C0 checks. A version of C0 that only compares the query-alone runs
cannot fail, because the rule is removed for those.

**C1, revert.** After `revert()`, the baseline must be reproduced bit-identically.

**C2, magnitude null.** At the decision cell of each stage, one hundred random writes,
each a rank-one matrix with the largest singular value matched to the own-context write's
largest singular value, from a fixed seed. Register row C-23 retired the isotropic
Frobenius-matched control for matching on the wrong quantity, and the first pilot
confirmed it barely moves the model at any drift. Reported as a distribution; no
hypothesis is decided against it.

**C3, no rule on the query, with a detector.** After every query pass, the rule's batch
counter must read zero and its applied count must be unchanged. One deliberate-leak arm
per stage installs the rule during a query pass and shows the score moves, so the
detector is known to fire.

**C4, swapped-context writes.** The primary null. For a query from context `i`, install
the write made from context `j ≠ i`, rescaled to the own write's Frobenius norm, and
score. In Stage 1 five fixed swaps per cell (two from the same class, three from the other
classes). In Stages 1b, 2 and 3, all other contexts in the held-out set. Specific transfer
is defined against this distribution's 95th percentile.

**C5, temperature-matched baseline.** Scale the baseline's final-position logits so its
entropy matches the test's. A write that only flattens or sharpens the distribution
scores no better than this, and it is reported beside every cell.

## 4. Stage 1: the map, on the development set

**What runs.** Two rules (`hebb`, `oja`; `anti_hebb` is dropped, `random` is replaced by
C2). Sixteen sites: the twelve feed-forward output matrices `blocks.{0..11}.mlp`, and the
attention output matrices of blocks 2, 5, 8 and 11, each addressed as its twelve head
stripes `blocks.{L}.attn.head.{0..11}` driven together through `MultiSitePlasticity`,
which is the whole matrix (a whole-matrix attention site is not constructible on
TransformerLens; the stripe form was verified to construct in review). Five target drifts.
Thirty development contexts. One fold per cell, five swapped-context writes per cell.

**Cost.** 4,800 folds and 24,000 swapped query passes: about three hours.

**Output.** For each class, the single cell (site, rule, drift) with the highest median
specific transfer over its ten development contexts. Nothing is decided here. Stage 1 is
a map and a selection; the multiplicity of 160 cells per rule means any threshold applied
here would be met by noise, as the review computed.

## 5. Stage 1b: confirmation, on the held-out set

**What runs.** At each class's selected cell only: the own-context write, all twenty-nine
swapped-context writes from the held-out set, one hundred C2 writes, and C5, for each of
the ten held-out contexts.

**Cost.** Thirty folds and about four thousand query passes: about half an hour.

**Decision rule.** A held-out context **shows specific transfer** if its own-context
transfer exceeds the 95th percentile of its twenty-nine swapped-context transfers and
exceeds C5. Under no effect, a context does this with probability about 0.07; seven of ten
by chance has probability below one in a million, and three classes are tested.

- **H4.1, binding does not transfer.** For the fact class and for the format class, fewer
  than seven of ten held-out contexts show specific transfer at the selected cell.
  **Refuted** if seven or more do in either class. Register row C-69.
- **H4.2, colouring transfers.** For the topic class, seven or more of ten held-out
  contexts show specific transfer at the selected cell. **Refuted** if fewer do. Register
  row C-69.

The Stage 1 map also reports, descriptively and without a hypothesis, whether the selected
cells fall on feed-forward or attention sites.

**What enters the register.** C-69 takes H4.1 and H4.2 together as one row: what a
single-site, single-pass write carries that is specific to its context, by class, at the
selected cell, against swapped-context writes.

## 6. Stage 2: the loop before the write

**What runs.** At the topic class's selected cell, on the ten held-out topic contexts. The
parent's loop is run over the state produced by the context pass for `N` iterations, with
`N` in 1, 10 and 100, through `atr_bridge.make_atr_step`, never a reimplementation. Two
write arms at each `N`: the **context write** (the fold as in Stage 1, `N = 0`), and the
**loop write**, the rule's accumulator over the `N` loop iterations only, applied once.
The two are not averaged together, because `apply()` averages over batches and an average
would dilute a context write by `1/(N+1)`. Swapped-context loop writes at every `N`, from
all nine other held-out topic contexts.

**What the loop does to the context, established from the record.** The injection
overwrites the first block's input wholesale, so from the first iteration the context
survives only as a sequence length (parent `docs/TECHNICAL.md`; `atr_bridge.py`). In
review, the loop write after ten iterations from two different contexts had cosine 0.98
with each other, against 0.40 for their context writes, and the loop write's cosine with
its own context write fell from minus 0.2 at one iteration to about zero at ten.

- **H4.4, nothing context-specific survives the loop.** At `N = 10` and at `N = 100`, fewer
  than seven of ten held-out contexts show specific transfer under the loop write, against
  swapped-context loop writes at the same `N`. **Refuted** if seven or more do at either
  `N`, which would mean the loop consolidated something specific to the context it
  started from. Register row C-70.

**Why this stage exists.** Lee and colleagues fold context into weights after an offline
recurrent pass over the stored context. This project's loop runs with no stored context.
Whether a free-running loop keeps anything of the context that seeded it is a question no
published work has asked, and either answer is a result.

**Cost.** Ten contexts, 111 loop iterations each, plus folds and swaps: about an hour.

## 7. Stage 3: the fidelity ladder

**What runs.** At each class's selected cell, on the held-out contexts, three write
methods at matched size, each scored by the same specific transfer:

1. **Best target-free write.** Whichever of `hebb` and `oja` Stage 1 selected. The two
   see the same activity and differ only in a brake, so they are one rung.
2. **Ridge least squares, query-blind.** Twenty probe queries per context, drawn from a
   fixed pool disjoint from the scored query. At the site, with `y = x·W + b`: regress
   the with-context output `y_ctx` on the probe-alone input `x_probe` to solve
   `ΔW = argmin ‖x_probe·(W0 + ΔW) + b − y_ctx‖² + λ‖ΔW‖²`, with the ridge weight `λ`
   chosen so that `‖ΔW‖_F` equals rung 1's. The system is under-determined (a few hundred
   rows against 3,072 inputs), which is why ridge rather than plain least squares. The
   write sees probe queries and never the scored one. This rung matches site outputs, not
   next-word predictions.
3. **Gradient-trained low-rank adapter, query-blind.** A rank-8 adapter on the same site,
   trained to minimise the final-position KL divergence to the reference on the twenty
   probe queries, with `ΔW` rescaled to rung 1's Frobenius norm after every step, then
   scored on the held-out query. This rung matches next-word predictions.

**Matched size, stated precisely.** All three rungs share `‖ΔW‖_F`. Their largest singular
values differ by construction (a rank-8 write at matched Frobenius norm has a smaller
largest singular value than a near-rank-one write), so `σ₁`, the largest singular value,
is reported beside every rung and the C2 distribution is run at each rung's own `σ₁`.

- **H4.5, transfer is ordered by what the method knows.** Rung 3 exceeds rung 2, and rung
  2 exceeds rung 1, in specific transfer on seven or more of ten held-out contexts, per
  class. **Refuted** if either ordering fails on seven or more. Register row C-71. The
  row records the ordering and the size of each gap; the gap between rung 1 and rung 3
  is the price of writing with no target.

**Cost.** Rung 1 is free from Stage 1b. Rung 2 is one ridge solve per context. Rung 3 is
minutes of training per context on CPU, so hours in total.

## 8. Reporting

Every cell writes one record: context id, class and split, site, rule, target and
realised drift, `clipped` (vacuous), non-finite, `N`, the full per-position KL for the
baseline and the test, the primary and secondary scores, the swapped-context
distribution's percentiles, the C2 distribution's percentiles, the C5 score, `σ₁`, and the
C3 detector reading. The run record is `experiments/output_exp004/exp004.jsonl`. The
results file states measurements only, in the form of the EXP-003 stage files, and cites
the record. The claim register is updated in a separate pull request, per the register's
rule that a status change is a bigger event than a measurement.

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
run's fact context bound an invented country to a rare multi-token name the model does
not reproduce in context; it is recorded as `pilot_v2.json` and its fact rows mean
nothing. The third run replaced that context with one in the fact format above and is
recorded as `pilot_v3.json`, script `experiments/exp004_pilot.py`, results and limits in
`PILOT.md`. Together they motivated the fact format, the screening rule, the
swapped-context primary null, and the final-position score. They are pilots: one site,
one seed set, one context per class, and the operator has ruled they are to be given
little weight.

**What the first draft of this file got wrong**, for the record: it decided hypotheses on
a 95th percentile of ten random draws across 384 scanned cells, which the review computed
would refute H4.1 and support H4.2 from noise alone; it named attention sites the adapter
cannot construct; it selected and tested Stage 2 on the same contexts; it promised matched
drift across rules without saying how; it undercounted the control cost five-fold and
assumed a forward pass five times faster than measured; and it attributed to the parent a
"position-uniform by iteration 10" finding that is not in the parent's record.
