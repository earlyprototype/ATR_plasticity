# Orientation

*What this project is, where it came from, and why the experiment is shaped the way
it is. Written for someone arriving cold. No results here — see `EXP_001_RESULTS.md`,
`STEP_SIZE_MAP.md` and `experiments/output_baseline/BASELINE.md` for those.*

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

That is the whole reason this repo exists. Not a claim about learning: there is no task,
no loss, and no target anywhere in this design. The narrow question is whether the one
persistent channel can be written to, and whether writing to it changes what the system
does afterwards.

## The design, and why each piece is there

### One weight matrix

`OjaPlasticity` attaches to a single matrix — by default the MLP output projection in
block 6. It watches the activations flowing through that matrix and, on request, nudges
it.

The module knows nothing about the ATR loop. It installs hooks, accumulates an update,
and applies it when told. The loop is imported from the parent, unchanged, so the only
new code in this repo is the plasticity layer.

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
C-15, and it holds only at large step size with the ceiling lifted.

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

## Reading order

1. This file.
2. `PRIOR_ART.md` — what exists in the literature, with each entry's verification status.
3. `EXP_001_SPEC.md` — the protocol, including the matched-axes table.
4. `experiments/output_baseline/BASELINE.md` — the frozen reference.
5. `STEP_SIZE_MAP.md` — which step sizes do anything.
6. `EXP_001_RESULTS.md` — the comparison.
7. `RESONANCE_NOTE.md` — the open question about what the loop is, kept separate from the
   measurements on purpose.
