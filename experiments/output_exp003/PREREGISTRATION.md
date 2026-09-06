# EXP-003 protocol

*What will be run, in what order, and what threshold decides each result. Written before
the runs it governs. Source material for the borrowed measurements is in `MEA_SOURCES.md`.
Nothing in this file enters the claim register; the **results** of stages 0, 1 and 2 now do,
as register rows C-65, C-66 and C-67.*

**History.** An earlier version of this file interleaved the protocol with an argument
about cultured neural networks and with proposed mechanisms for EXP-002's collapse. That
argument was written by an agent, was not requested, and has been removed at the
operator's instruction. The protocol, the thresholds and the amendments are unchanged.
Stages 0, 1 and 2 have run; their results are in the three `STAGE*_RESULTS.md` files.

## Regime

**No drift ceiling**, by operator decision, for this entire series, as for EXP-002. Every
result in this repository before EXP-002 was taken with that cap at 5%, so nothing
produced here is continuous with those without saying so.

## Definitions

**The loop.** The frozen model's state is read from the last block, rescaled to a fixed
size, and written back into the first block. One pass is one iteration. With `f` the model
and `N` the rescaling: `r` becomes `f(N(r))`.

**The grid.** 12 blocks by 12 heads, 144 addressable sites. Sites both carry the activity
measurement and receive the injected signal.

**Strength.** A unit vector added at a site, scaled by `beta` times the length of the
activity already present there, so `beta` is a fraction of local activity. Ladder: 0.003,
0.010, 0.032, 0.100, 0.316.

**Rate.** The signal is injected on one iteration in every `m`. Ladder: `m` in 1, 2, 4, 8,
16.

**Where current enters.** A site is a block and a head. The signal is added at
`blocks.{L}.attn.hook_z` indexed by that head. A whole-stream variant at
`blocks.{L}.hook_resid_pre` is also implemented.

**Stimulated sites are excluded from the activity measurement** on the iterations they
fire, and the excluded count is reported.

**Settled.** A fixed point if consecutive iterations agree above 0.999 on three checks
every ten iterations after iteration 100; a two-step cycle if that fails and the same test
on iterations two apart passes; unsettled otherwise. **Settled means either of the first
two.** This is the committed baseline's classifier, used unchanged.

A criterion built on consecutive steps alone would score the census's eight `Divine`
inputs as unsettled, because that end state is a two-step cycle with consecutive-step
agreement near 0.68, and would set the baseline at 23 of 31 rather than 31 of 31.

**Census agreement.** Of 31 fresh inputs, how many settle where the frozen model puts
them, counting only those that settled. Frozen baseline: 31 of 31.

Counting distinct end states is not used as a measurement. EXP-002 records an arm with 19
distinct end states and only 4 of 31 settled.

**Separation statistic.** Between-group spread over within-group spread. 1.0 means groups
are indistinguishable from their own internal scatter. Token labels score 0.87, from
register row C-07. Reported alongside every count.

## Arm assignments

- **Distributed:** 24 of the 144 sites, drawn without replacement by
  `torch.Generator().manual_seed(20260802)` over sites ordered `(block, head)` with block
  varying slowest. The realised list is written into the run record.
- **Focal, primary:** `(block 6, head 8)`. Block 6 is where every committed single-site
  result was measured. Head 8 is arbitrary and pinned.
- **Focal, second arm:** the lower median of the realised distributed draw, defined as
  element 11 after sorting the 24 sites by `(block, head)`.

**How the two focal arms combine.** At each setting, the focal result is the **maximum**
of the two arms' census agreement. Maximum rather than mean, because the comparison asks
whether concentrating the signal can do as well as spreading it, so the strongest focal
result is the fair opponent. If the two arms differ by more than 5 of 31, the focal
condition is reported as **site-dependent** at that setting and **no
distributed-versus-focal claim is made there**, whichever arm is higher. The same rule and
the same maximum apply to Stage 4's selection.

Sites are chosen by position, not by depth: the forward direction is ignored in the design
and measured in the result.

## Stages

### Stage 0 — validate the measurement. RUN.

Depth-weighted centroid over the grid, on the frozen model, against the committed 125-input
baseline.

| Gate | Threshold |
|---|---|
| Separates the five end states | above 1.5 |
| Separates fixed points from two-step cycles | above 1.5 |
| Does not separate random halves of the largest group | below 1.2 in at least 9 of 10 splits |
| Control A, permute block labels | must score below the true statistic |
| Control B, permute head labels | the **depth** centroid must change nothing, threshold 1e-9 |

Control B is a depth-only control: it permutes head labels and checks that `ca_depth` is
unmoved. It does not test the head component of the two-number centroid, which is not used
by any gate.

Registered consequence of a failed gate: the statistic is discarded and later stages use
labels alone with C-07's limitation attached.

Results: `STAGE0_RESULTS.md`.

### Stage 1 — spectral concentration. RUN.

Participation-ratio effective rank on the twelve adjusted matrices, before and after
EXP-002's reinforcing episode reproduced exactly.

| Verdict | Threshold |
|---|---|
| Supported | mean effective rank falls 10% or more |
| Refuted | falls less than 2% |
| Inconclusive | between the two |

10% was set against the largest movement in the committed step-size map, a rise of about
0.6%.

Also registered and **not implemented**: the depth-weighted centre of the weight change,
and the smallest detectable change per statistic.

Results: `STAGE1_RESULTS.md`.

### Stage 2 — adjustment cadence. RUN.

Cadence `k` in 1, 4, 12 at a fixed 120-iteration episode, step size multiplied by `k`.

| Rule | Threshold |
|---|---|
| Drift must match across the ladder | within a factor of 2 |
| If unmatched | comparison is qualitative only, falsifier not invoked |
| Falsifier | agreement at `k`=12 exceeds `k`=1 by 5 or more |

Results: `STAGE2_RESULTS.md`.

### Stage 3 — injected signal, focal against distributed. NOT RUN.

Reinforcing rule at all twelve blocks, loop closed, signal injected during the adjustment.

**Grid:** the rate ladder at fixed strength 0.032, and the strength ladder at fixed rate
`m`=1, both shapes. Eighteen distinct settings.

**Two censuses per setting**, one with the signal still applied and one with it removed.
Both run from a single frozen copy of the weights with all adjustment disabled, so neither
can affect the other and their order cannot matter.

**Matching between shapes:** sum of squares of the per-site strengths held equal, so 24
sites each receive `beta` divided by the square root of 24. The plain-sum alternative is
reported as a sensitivity check.

| Threshold | Value |
|---|---|
| Distributed advantage claimed | distributed exceeds best focal by 10 or more of 31, on the signal-on census |
| Crossing claimed | focal exceeds distributed by 5 or more at `m`=16 **and** distributed exceeds focal by 10 or more at `m`=1 |

A crossing on the strength ladder is recorded as a separate observation about this system.

**Controls:**

- *No signal*: strength zero, must reproduce EXP-002's collapse.
- *Signal without adjustment*: frozen model plus signal, must leave census agreement at 28
  of 31 or above. Below that, the setting is excluded and reported as excluded.
- *Reflection*: at each setting, take pairwise distances across the 31 settled states as
  one minus cosine, and the same for the injected vectors and for the frozen model's
  settled states. Compute Spearman rank correlation over the 465 pairs, giving `s_signal`
  and `s_frozen`. **If `s_signal` exceeds `s_frozen` by more than 0.10, that setting and
  every stronger one on the same ladder are discarded.** Ties discard. Computed on the
  signal-on census; for a two-step cycle the representative state is the last iterate.

  **"Stronger" is defined per ladder**, because the two run in opposite directions.
  On the strength ladder, stronger means larger `beta`, so discarding proceeds upward
  through 0.003, 0.010, 0.032, 0.100, 0.316. On the rate ladder, stronger means **more
  frequent**, so it is smaller `m`, and discarding proceeds downward through 16, 8, 4, 2,
  1. Discarding is applied independently on each ladder.
- *Zero strength*: bit-identical to the existing loop, and one injection point bit-identical
  to single-point injection.

If reflection leaves fewer than three admissible settings on a ladder, the crossing test on
that ladder is inconclusive rather than null.

Cost: about 55 hours.

### Stage 4 — signal during against signal after. NOT RUN.

Two orders at the setting with the highest signal-on census agreement from Stage 3,
**selected only from settings that survived every Stage 3 control**: not reflection
discarded, not excluded by the signal-without-adjustment floor, and not marked
site-dependent. Ties to the lower strength, then the larger `m`, then the distributed
shape. If no admissible setting remains, Stage 4 does not run.

Falsifier: if applying the signal afterwards recovers to within 5 of 31 of applying it
during, the account of where the change is held is wrong.

Does not run if no Stage 3 setting exceeded its no-signal arm by 5 or more.

### Stage 5 — site count. NOT RUN, NON-FALSIFYING.

Compares EXP-002's twelve-site run against work using the 144 mixing units. **No falsifier
is offered.** The two differ in site count and site type at once, so neither outcome is
attributable to either. Reported as an observation.

What would make it a test: hold site type fixed and vary only the count.

## Matched conditions

The project checks seventeen conditions before comparing connected against disconnected
runs. Signal strength and signal shape become the eighteenth and nineteenth. No comparison
is reported unless all nineteen agree.

Register row C-63 records that the no-feedback baseline is exactly zero only at a single
**plastic** site. **Every comparison in this series adjusts twelve plastic sites**, so none
has a zero baseline and no quantity here may be placed in a series with rows C-31 or C-58.
That statement is about plastic sites only. It is unrelated to the number of sites that
receive the injected signal, which is 24 for the distributed arm, 1 for focal, and 0 in
Stages 0 to 2.

## What is not claimed

No claim that the system is learning. No claim about what any end state means beyond being
the settled top-1 token. No claim that any process here is the process occurring in tissue.

## Known limits

One driven input, one seed, one site family, one model, one episode length. The census uses
31 inputs, not the full 125. Register rows C-64 and C-41 already record this as the
project's weakest dimension. (This cited C-40 until that row was retired for staleness —
it asserted a single site and cadence 1, which this experiment's own Stage 1 and Stage 2
contradict. C-64 carries the same scope point with the corrected numbers.)

The ATR loop's own injection at `blocks.{layer_start}.hook_resid_pre` rescales the carried
state and writes it back wholesale; that is the loop, not the stimulation. The **stimulation**
vectors are separate: one fixed unit direction per site, drawn once from a seeded generator
and reused on every firing, added to what is already at the site. Fresh noise per step is not
tested and is the obvious follow-up.

Rule-to-rule comparison is qualitative only, following EXP-002, because the two rules
cannot be matched on drift.

---

## Amendment 1: cadence ladder, before Stage 2

Registered as `k` in 1, 10, 100 with the episode lengthened so the slowest setting still
applied 120 adjustments. That makes the `k`=100 cell 12,000 iterations, about 90 minutes at
the measured 0.44 seconds per iteration.

Changed to `k` in 1, 4, 12 at a fixed 120-iteration episode, step size multiplied by `k`.

Consequence: the extremes differ by 12 rather than 100, so a null is "not supported at this
resolution" and not a refutation.

## Amendment 2: three gaps closed before Stage 3

1. Both arms' sites are now pinned, above. Neither was.
2. The reflection control's decision rule is now executable, above. It was a description.
3. Stage 1 registered two measurements that were never implemented; they are recorded as
   not run in `STAGE1_RESULTS.md`.
