# Orientation

*What this project is, where it came from, and why the experiment is shaped the way
it is. Written for someone arriving cold. **No findings here**, deliberately: this file
explains the apparatus and the two regime boundaries that govern how any number in this
repository has to be read, and the reading order at the bottom says where the findings live.*

> **Before you quote anything from any document in this repository, know that
> [`CLAIMS.md`](CLAIMS.md) exists and outranks all of them.** Prose files here describe what
> was measured. The register decides what those measurements are allowed to be *called*, it
> carries the caveat that has to travel with each one, and it keeps retired claims on the page
> rather than deleting them. It exists because the same correction was once made in one
> document and left stale in three others. Where a prose file and the register disagree, the
> register is right and the prose file is the bug. This file is prose.

## The starting point: the ATR loop

The parent project (`lucier-gpt2-activ-tensor-reson-experiments`) runs GPT-2 small in a
closed loop:

1. Run a prompt through the model.
2. Read the residual stream at the last block, `blocks.11.hook_resid_post`.
3. Rescale it to the norm the stream had at the *start* of the trajectory.
4. Write it back in at the first block, `blocks.0.hook_resid_pre`.
5. Repeat, for hundreds of iterations.

The weights never change; only the residual-stream state carried between iterations does.

Run normally, a transformer is a single pass — you put a prompt in, you get logits out,
nothing persists. Iterated, the same weights define a map from a residual-stream state to
a residual-stream state, and that map can be applied to its own output. Fixed points,
cycles and basins are properties of that map. They are determined by the weights, and
ordinary inference never brings them into view because ordinary inference applies the map
once.

The parent project ran this loop over 125 prompts: the trajectories converge to a small
number of end states, one of which is a two-step cycle rather than a fixed point.

## Why add plasticity

When a new prompt arrives, the residual stream is rebuilt from scratch. Whatever the loop
reached — the basin, the cycle, the state it took hundreds of iterations to find — is
gone.

**The weights are the only thing that persists across prompts.** They are the only place
an episode can leave a mark that a later episode could read.

That is the whole reason this repo exists. Not a claim about learning: there is no task and
no target anywhere in this design. The narrow question is whether the one persistent channel
can be written to, and whether writing to it changes what the system does afterwards.

**Be precise about "no loss", because the repo has been caught being imprecise about it.**
What is true is that there is no *externally specified* objective. It is not true that there
is no objective at all: plain Hebb is exactly gradient ascent on output energy, since
`∂/∂W ½E‖xW+b‖² = E[x yᵀ]`, which is the update. Register row **C-11** holds this, and it
matters because "no objective" is the axis the novelty claim leans on, and it is softer than
it first looks.

## The design, and why each piece is there

### One weight matrix, and later twelve

`OjaPlasticity` attaches to a single matrix — by default the MLP output projection in
block 6. It watches the activations flowing through that matrix and, on request, nudges
it.

The module knows nothing about the ATR loop. It installs hooks, accumulates an update,
and applies it when told. The loop is imported from the parent, unchanged, so the only
new code in this repo is the plasticity layer.

That single-site design is where the project started and where most of its numbers come
from. `MultiSitePlasticity` in `multi_site.py` later generalised it to N sites at once, and
EXP-002 used it to make all twelve MLP output projections plastic simultaneously. **Read the
two as separate regimes**, because the multi-site work also runs with the drift ceiling
lifted and loses a control guarantee the single-site work depends on. Both differences are
spelled out further down, under the severed-path control and under "the second regime".

### Four rules

Written in the convention `W` is `(n_in, n_out)` and the module computes `y = x @ W`:

```
hebb       dW =   E[x yᵀ]
oja        dW =   E[x yᵀ] − W E[y yᵀ]
anti_hebb  dW = − E[x yᵀ] − W E[y yᵀ]
random     norm-matched noise
```

`x` is the pre-synaptic activity, `y` the post-synaptic. The second term in Oja is a
brake: it opposes growth and keeps the weight bounded. Hebb has no brake.

Anti-Hebbian negates the reinforcement term **only**. Negating the learning rate instead
would flip the brake as well, turning it into an accelerator.

`random` is the control arm: an update of the same magnitude with no structure. If a
result appears under `random` too, that indicates the result follows from perturbing the
matrix rather than from the rule's structure.

### A step size and a cadence

`eta` scales the update. Cadence is how many loop iterations pass between applications.

A ceiling, `max_delta_frac`, caps total drift from the starting weight, 5% by default.
Hebb has no brake of its own, so the ceiling is what bounds it at large step sizes. Note
that at the step sizes these experiments run at, Hebb does not in fact run away: at the
working point it records zero non-finite values, the ceiling never fires, and the weight
norm moves 0.03%. The claim that raw Hebb diverges immediately is retired as register row
C-15. What survives is narrower: at large step size with the ceiling lifted, Hebb's drift
keeps growing over the applications control C3 measures, while Oja's settles. C3 runs a
fixed, small number of applications, so that is continued growth in the regime tested and
not a demonstration that the growth is unbounded.

The ceiling is a chosen number, and a run in which it is firing is a measurement of the
ceiling rather than of the rule, so the clip state has to be reported with every result.
**The library does not give you a clipping rate.** `report()` returns `clipped`, a single
latching boolean that goes true on the first clip and clears only on `revert()`. Register
row C-46 retires the claim that a rate is recorded. The experiment scripts work around
this: `step_size_map.py` synthesises a per-cell rate itself, and `exp001_hebb.py` records
the boolean and documents the limitation.

### Two backends

The parent's loop runs on TransformerLens, where the target is a bare `nn.Parameter`
called `W_out` with no owning module whose forward is that matmul. HuggingFace stores the
same matrix as a `Conv1D` with a `.weight`.

Site adapters supply four things — read the weight, write it, start capturing activations,
stop — so the rules never see the difference. Adding a backend means writing those four
methods.

### Per-head sites

Attention output projections can be targeted one head at a time. The parent found the
two-step cycle carried by a single attention head in block 11. Head-level targeting exists
so that site can be addressed directly.

`x` is that head's slice of the input; `y` is the full projection output, since the heads
share one post-synaptic activity. That choice makes a head update exactly the row slice of
the whole-matrix update.

## The offline arm: connected against disconnected

**Oja's rule moves the weight matrix whether or not there is any feedback.** Feed it a
recording of activations and it will still converge toward the dominant direction of their
second-moment matrix. So "we ran the loop with plasticity on and the behaviour changed" is
not evidence about the loop.

The claim this project can make is about *coupling* — weights changing while the thing
they change feeds back into them. That claim lives entirely in the difference between two
runs:

| | Connected | Disconnected |
|---|---|---|
| Loop runs | yes | yes, once, frozen |
| Activations | live, changing as weights change | recorded from the frozen run, fixed |
| Rule applied | to live activations | replayed over the recording |
| Feedback | present | absent by construction |

Everything else must match: step size, ceiling, number of updates, order of samples,
batching, starting weight, seed, and whether centring is applied. Seventeen such axes are
checked mechanically before a comparison is reported; a mismatch on any of them means the
difference is not evidence about feedback.

`offline_control.py` implements both arms and the verifier.

### The severed-path control

One further check, because a matched comparison can still be measuring the wrong thing:
run the same comparison with the feedback path **physically cut** — read the loop out at
an early block, put the plastic site downstream of it, so coupling is impossible.

Any effect claimed in the connected configuration has to exceed what this reports.

**This control's floor is exactly zero at one plastic site, and it is not zero beyond one.**
That distinction is register row **C-63** and it is the most important thing to understand
before reading any multi-site number. Severing the loop cuts a plastic layer's path into the
*next* iterate, which for a single site is the only path there is. With several plastic
layers, a lower one's drift changes the activations arriving at a higher one **within a single
forward pass**, and no amount of severing the loop cuts that. So a multi-site
connected-versus-offline difference measures loop feedback *and* within-pass layer
interaction, with no zero baseline to measure it against, and it may not be put in a series
with the single-site numbers. EXP-002's own severed gate shows this directly: the lowest
plastic layer gives exactly 0.0, bit-identical, and every layer above it does not.

There is a second, smaller boundary. At a **per-head** site the reconstruction is additive
out of a fused twelve-head operation, so the floor is float32 noise, around 1e-7, rather than
exactly zero (C-57). The exact zero is a whole-matrix property.

## Controls

- **C0** — with `eta = 0`, the model must be bit-identical to the frozen one. If this
  fails, nothing downstream is interpretable.
- **C1** — after `revert()`, the weight must return exactly to its starting value.
- **C2** — the `random` arm, norm-matched to what the rule would have applied.
- **C3** — the plasticity layer must not perturb the loop except through the weight.

An autouse fixture checks that no test leaves a watched matrix modified, since the model
is session-scoped and a leak would contaminate every test after it.

## What gets measured

From the loop: which basin a trajectory settles into, the lag-1 and lag-2 cosines that
distinguish a fixed point from a two-step cycle, position uniformity, and the margin
between the top two tokens at the readout.

From the weights: Frobenius norm over time, relative change from the start, whether the
ceiling clipped (a boolean from the library, or a rate the experiment script computes for
itself, per C-46 above), non-finite count, largest entry, and effective rank.

Effective rank is there for a specific failure: one entry runs away, the normaliser
rescales, and the rest of the matrix is flattened, while the norm stays constant and the
ceiling stays quiet. It is tracked so that this failure mode is visible when the norm and
the clip state are not showing it.

## The second regime: the ceiling comes off

Everything described above runs with `max_delta_frac` capping total drift at 5%. **EXP-002 and
EXP-003 do not.** The ceiling was lifted for that series by operator decision, so those runs
reach drift of 1.3% to 7.9% with nothing clipping, and **no number from them is continuous
with a number from the capped runs.** If you are comparing two figures in this repository, the
first thing to check is whether they came from the same regime. Register row C-60 carries this.

## Reading order

**Read the register first if you intend to quote anything.** The rest is background and
sequencing.

| # | File | What it is for |
|---|---|---|
| 1 | This file | The apparatus, and why each piece of it is shaped that way |
| 2 | [`CLAIMS.md`](CLAIMS.md) | **The authority.** Every claim the project makes, its status, its evidence and the caveat that must travel with it. Retired claims stay on the page |
| 3 | [`HANDOVER.md`](HANDOVER.md) | Where the project is, what has run, what has not, and what the recorded plan says to do next |
| 4 | `PRIOR_ART.md` | What exists in the literature, with each entry's verification status. Read C-42's caveat with it: the novelty claim rests on a search, and eleven of its absence claims have no preserved artifact |
| 5 | `experiments/output_baseline/BASELINE.md` | The frozen reference every later result is compared against, and the source of the resolution-limit caveat C-07 |
| 6 | `STEP_SIZE_MAP.md` | Which step sizes do anything, at one site |
| 7 | `EXP_001_RESULTS.md` | The connected-versus-offline comparison at the working point |
| 8 | `BASIN_BIFURCATION.md`, then `experiments/output_t1_1/T1_1_RESULTS.md` | In that order, and do not stop after the first: the second refutes the first's conclusion. The edit **displaces** one attractor rather than creating a second (C-26 `retired`, C-56 `supported`) |
| 9 | `experiments/output_exp002/EXP_002_RESULTS.md` | Twelve plastic layers and a reprompt. Something does cross the prompt boundary, and what crosses is collapse rather than steering (C-61 with C-62, never apart) |
| 10 | `experiments/output_exp003/` | Three stages run against pre-registered thresholds, plus `MEA_SOURCES.md` for the borrowed measurements. **Nothing here enters the register**, by design, so nothing in it is quotable as a claim |
| 11 | `RESONANCE_NOTE.md` | The open question about what the loop is, kept separate from the measurements on purpose |
| 12 | `ALIGNMENT_REVIEW.md` | How the claim layer and the evidence layer came apart once, and the task list that came out of it. Its status note says which items have since run |

`EXP_001_SPEC.md` is deliberately left off this list. It still carries the title and the
"proposed, not run" status of a *different* experiment from the one `EXP_001_RESULTS.md`
reports, because the label "EXP-001" was reused. Resolving that is an open decision for the
operator, recorded as `HANDOVER.md` §6 item 1. Its matched-axes table is still the reference
for the seventeen axes, and its §0 and §5.3 were corrected and are right either way.
