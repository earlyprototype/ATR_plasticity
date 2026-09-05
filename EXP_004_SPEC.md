# EXP-004: Context to weight transfer

**Status:** proposed, not run. Awaiting the operator's reading of
`docs/CONTEXT_TO_WEIGHT_NOTE_2026-09-05.md`, which argues for this protocol.
**Model:** GPT-2 Small, TransformerLens, layers 0 to 11.
**Register rows:** C-69, C-70, C-71, entered as `open`. Claimed on the identifier
registry by `agent:ctx-to-weight` on 2026-09-05.
**Cost:** Stage 1 single-digit hours of CPU; Stage 2 about an hour; Stage 3 hours, most of
it the gradient-trained arm.

*What will be run, in what order, and what threshold decides each result. Written before
the runs it governs. Nothing in this file enters the claim register until a stage has run
and its result has been reviewed.*

## The question

Every result in this repository so far feeds the plastic site the model's own free-running
activity and reads out which basin the loop settles in. This experiment feeds the site a
context the model was given, clears the context, and asks how much of the context's effect
on the model's next-word predictions survives in the weights. The write channel is
unchanged: the same `OjaPlasticity` object, the same rules, the same ceiling, the same
`revert()`. Only the input to the channel and the readout differ.

## Definitions

**Context, query, reference.** A context `C` is a short passage. A query `Q` is a short
prompt that follows it. The reference is the model's next-word distribution at each of the
query's token positions when run on `C` followed by `Q`, with the original weights.

**Baseline.** The model's next-word distribution at the query positions when run on `Q`
alone, with the original weights. This is what "no context at all" scores.

**Test.** The model's next-word distribution at the query positions when run on `Q` alone,
after the context has been folded into the plastic site and the site's new weight installed.

**Score.** The KL divergence from the reference to the test, in nats, averaged over the
query positions. Computed in float64 from the logits. Zero means the test predicts exactly
as the reference did. A second score, the log probability the test assigns to the
reference's most likely final token, is recorded alongside and is not used for any decision.

**Transfer.** The baseline score minus the test score, in nats. Positive means the write
moved the model toward its with-context behaviour. This is the quantity every hypothesis
below is stated in.

**Drift.** `‖ΔW‖_F / ‖W0‖_F`, the site's relative weight change, from `report()`. Reported
with every cell, with the ceiling's `clipped` flag, per the standing prohibition.

**The fold.** With the rule installed at the site, run the model once over `C` alone, so
the rule observes the site's input and output activity at every context position, then
call `apply()` once. One pass, one write. Stage 2 changes what runs before the write and
nothing else.

**Context classes.** Three, ten contexts each, thirty in total, fixed before the run and
written into the run record:

- **Fact.** An invented entity and one attribute of it, followed by a query that asks for
  the attribute. Invented, so the model cannot know it from training.
- **Format.** Four demonstrations of a mechanical transformation, followed by a fifth
  input. The model can know the transformation from training; what the context supplies
  is which transformation.
- **Topic.** Three sentences on a subject, followed by a query that continues it. What the
  context supplies is a vocabulary and register.

The three classes are chosen because the softmax argument predicts a different result for
each: a fact needs a sharp lookup, a format needs a rule, a topic needs only a colouring.

## Controls

**C0, the gate.** With `eta = 0` and `mode = "off"`, the test must be bit-identical to the
baseline. If it is not, the instrument perturbs the model and nothing downstream is
interpretable.

**C1, revert.** After `revert()`, the baseline must be reproduced bit-identically.

**C2, matched random.** For every `(context, site, eta)` cell, ten random matrices of the
same Frobenius norm as that cell's `ΔW`, each written to the site in turn. The transfer
under random is reported as a distribution, and every hypothesis compares against its 95th
percentile. Register row C-23 retired the single-seed norm-matched control for matching on
the wrong quantity, so a second random arm matches on `σ₁`, the largest singular value, by
writing a random rank-one matrix of that size, per T1.4. Both arms are reported.

**C3, no rule on the query.** The rule observes only the context pass. The query pass runs
with the rule removed. A leaked hook on the query pass would let the write see the query,
which is the thing this experiment is not allowed to do.

## Stage 1: the single-pass map

**What runs.** Four rules (`hebb`, `oja`, `anti_hebb`, `random`), twenty-four sites (the
twelve feed-forward output matrices `blocks.{0..11}.mlp` and the twelve attention output
matrices `blocks.{0..11}.attn`), eight step sizes spaced by half-decades from 1e-4 to 3e-1,
thirty contexts. Ceiling lifted to 1.0 so the rule is measured rather than the ceiling,
with drift recorded per cell and any cell above 15 percent drift excluded from the
summary. One fold per cell.

**Cost.** About 23,000 folds plus 24,000 random writes, each followed by one short forward
pass. Single-digit hours on this session's CPU.

**Hypotheses, with what decides each.**

- **H4.1, binding does not transfer.** For every fact-class and format-class context, no
  `hebb` or `oja` cell at any site or step size has transfer above the 95th percentile of
  its matched-random distribution. Refuted if any such cell exists with drift below 15
  percent and the ceiling silent, on at least three of the ten contexts in that class at
  the same site and step size. Register row C-69.
- **H4.2, colouring transfers.** For topic-class contexts, at least one `hebb` cell has
  transfer above the 95th percentile of its matched-random distribution, on at least three
  of ten contexts at the same site and step size. Refuted if no such cell exists. Register
  row C-69.
- **H4.3, the attention sites do better than the feed-forward sites.** The best
  ceiling-silent transfer over attention sites exceeds the best over feed-forward sites on
  the topic class, by more than the random distribution's spread at that cell. Refuted if
  it does not. Not registered; informs the site choice for later stages.

**What enters the register.** C-69 takes the answer to H4.1 and H4.2 together, as one row:
what a single-site, single-pass Hebbian write carries, by context class, against matched
random. If both hypotheses hold, the row says the write carries colouring and not binding.
If H4.1 is refuted, the row says where and at what cost binding transferred.

## Stage 2: the loop as consolidation

**What runs.** The feed-forward site and the step size that gave the best ceiling-silent
topic-class transfer in Stage 1, `hebb` only, all thirty contexts. Before the fold, the
parent's loop is run over the context's state for `N` iterations, with `N` in 0, 1, 10 and
100, through `atr_bridge.make_atr_step`, never a reimplementation. `N = 0` is the Stage 1
fold, reproduced. The rule observes every loop iteration and applies once at the end, so
the write is the average over what the loop produced.

**Cost.** Thirty contexts times 111 loop iterations plus folds and random controls. About
an hour on CPU.

**Hypothesis, with what decides it.**

- **H4.4, the loop erases.** Transfer at `N = 100` is below transfer at `N = 0` on the
  topic class, by more than the matched-random spread, on at least seven of ten contexts.
  This is the prediction from the parent's finding that the loop's state becomes
  position-uniform by about iteration 10 and that 125 prompts collapse into 5 basins.
  Refuted if transfer at `N = 100` exceeds transfer at `N = 0` by more than the spread on
  at least seven of ten contexts, which would mean the loop consolidates. Register row
  C-70.

**Why this stage exists.** Lee and colleagues fold context into weights after an offline
recurrent pass over the stored context. This project's loop runs with no stored context.
Whether a free-running loop helps or hurts a subsequent fold is a question only this
project can ask, and either answer is a result.

## Stage 3: the fidelity ladder

**What runs.** At the site and step size chosen in Stage 2, on all thirty contexts, four
write methods at matched drift, each scored by the same transfer:

1. **`hebb`**, target-free, one pass. From Stage 1.
2. **`oja`**, target-free with a brake, one pass. From Stage 1.
3. **Least squares, query-blind.** Twenty probe queries per context, disjoint from the
   scored query, drawn from a fixed pool. Solve for the `ΔW` at the site that best maps
   the site's input on the probe queries alone to its output on the probe queries with the
   context present. Rank-limited to match the Hebbian write's effective rank. The write
   sees probe queries and never the scored one.
4. **Gradient-trained low-rank adapter, query-blind.** A rank-8 adapter on the same site,
   trained to minimise the score on the twenty probe queries, then scored on the held-out
   query. The context-distillation rung.

**Cost.** Rungs 1 and 2 are free from Stage 1. Rung 3 is one linear solve per context. Rung
4 is minutes of training per context on CPU, so hours in total.

**Hypothesis, with what decides it.**

- **H4.5, transfer is ordered by how much the method knows.** Transfer increases
  monotonically down the ladder on each context class, at matched drift. Refuted if any
  lower rung beats a higher one on a class by more than the matched-random spread on at
  least seven of ten contexts. Register row C-71. The row records the ordering and the size
  of each gap; the gap between rung 1 and rung 4 is the price of writing with no target.

## Stage 4, optional: many sites at once

`MultiSitePlasticity` drives every feed-forward site together. EXP-002 found that this
collapses the loop's landscape. Whether it carries more of a context than one site does is
a one-cell question at the Stage 1 step size, run only if Stage 1 leaves the best single
site below the topic class's baseline by more than half. Not registered.

## Reporting

Every cell writes one record: context id and class, site, rule, eta, `N`, drift, clipped,
non-finite, baseline score, test score, transfer, the random distribution's percentiles,
and the final-token log probability. The run record is `experiments/output_exp004/
exp004.jsonl`. The results file states measurements only, in the form of the EXP-003
stage files, and cites the record. The claim register is updated in a separate pull
request, per the register's rule that a status change is a bigger event than a
measurement.

## Pilot

One scratch run of the Stage 1 measurement, at one site, on three contexts, with one
random seed and no loop, was made on 2026-09-05 before this protocol was written. It is
recorded in `experiments/output_exp004/PILOT.md` and `pilot.json`, with its script at
`experiments/exp004_pilot.py`. The operator has ruled it is to be given little weight. It
informed the choice of context classes and the step-size range, and nothing else in this
file depends on it.
