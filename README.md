# Hebbian ATR — plastic-weight iterated dynamics

*What happens to a model's attractor landscape when the weights are allowed to
change under the loop?*

> **Status: the plasticity scaffold now composes with the parent ATR engine, and
> the first experiment has been run.** A TransformerLens site adapter
> (`_TransformerLensMLPSite` in [plasticity.py](plasticity.py)) attaches the rule
> to the parent's bare `W_out` parameter, and a per-step bridge
> ([atr_bridge.py](atr_bridge.py)) hands `controls.py` the `atr_step(model, r)` it
> expects by extracting one iteration of the parent loop verbatim. CI checks out
> the parent repo and runs the bridge's acceptance test under `ATR_REQUIRE_PARENT=1`
> ([.github/workflows/tests.yml](.github/workflows/tests.yml)), so the extracted
> step is held to reproducing `run_atr_loop` bit-exactly rather than merely
> importing. The first experiment has since been run: results in
> [EXP_001_RESULTS.md](EXP_001_RESULTS.md), the frozen baseline in
> [BASELINE.md](experiments/output_baseline/BASELINE.md), the step-size map in
> [STEP_SIZE_MAP.md](STEP_SIZE_MAP.md). Arriving cold? Read
> [ORIENTATION.md](ORIENTATION.md) first, then [EXP_001_SPEC.md](EXP_001_SPEC.md).
>
> The suite found three defects, all since fixed and each now held by a test that
> fails if it returns:
>
> 1. `transposed=True` never transposed — `_hook`'s branch was a bare `pass`, so
>    non-square `nn.Linear` raised in `apply()` and square `nn.Linear` silently
>    learned the update transposed. The flip now happens in `apply()`.
> 2. `mode="random"` was norm-matched to the raw Hebb term rather than to Oja,
>    which **biased C2** — worst at the large eta C2 is actually run at. The
>    decay term is now subtracted for `"random"` as well as `"oja"`.
> 3. C1, C2 and C3 leaked their forward hook and left the weights drifted if
>    `atr_step` raised, where C0 used a `with` block and did not. All three now
>    remove the hook and revert the weights on the way out.
>
> A green suite is not C0 passing. Control C0 must still pass bit-exactly
> against real weights for anything here to count as a result worth recording.

## The question

The parent project ([ATR](https://github.com/earlyprototype/lucier-gpt2-activ-tensor-reson-experiments))
iterates a frozen transformer's forward map and characterises the attractors.
The weights never move. That is the fast loop of a neural system studied in
isolation, with the slow loop switched off.

This repo turns the slow loop back on, minimally: **one weight matrix, one local
learning rule, driven by the activations passing through it during iteration.**

The specific question is whether the landscape's structure is a static property
of a trained artifact or something the loop can reshape:

- Do basins **deepen** — wider basins of attraction, i.e. consolidation?
- Or does the landscape **collapse further** — fewer basins, the model-collapse
  direction?
- Does the `Divine` period-2 limit cycle become a fixed point, or survive?
- Does GPT-2 Medium's single `D` funnel open, or tighten?

## What the project can say so far

This section states the claim the project leads with, and the frame around it. It
cites the claim register, [CLAIMS.md](CLAIMS.md), which is the file that decides what
each measurement is allowed to be called. The reasoning behind the choice of frame is
in [ALIGNMENT_REVIEW.md](ALIGNMENT_REVIEW.md) section 6, which is marked there as
interpretation rather than measurement.

**The project is a characterisation study.** The question it leads with is: what does
an unconstrained local learning rule, given no task and no target, do to a pretrained
model's iterated dynamics? "Iterated dynamics" is the behaviour of the loop that feeds
the model's output back in as its next input, hundreds of times, until the internal
state settles. That question carries no externally specified objective, unlike the
fine-tuning and model-editing work it sits next to. Whether that makes it unoccupied
ground is **not** settled here: the register holds the novelty claim at `provisional`
(C-42), because it rests on a literature search rather than on the literature, and
eleven of its absence claims have no preserved search artifact. C-43, on how the work
compares to model editing, is still `open`. Treat this as a description of the setup,
not as a claim to priority.

**The standing result is about editability.** A frozen transformer's settled-state
landscape is editable by a near-rank-1 weight change of about 1% of the matrix's own
size, derived from the model's own activation statistics, with no target and no
externally specified objective. "Near-rank-1" means the change is dominated by a single
direction: 96% to 100% of it in the routed and severed control cells, and 81% to 84% in
the two live-episode cells, so "rank-1" is an approximation rather than an exact
description. At one site (`blocks.6.mlp`) of one model (GPT-2 small), that edit changes
which word the loop settles on (claim C-20, three prompts), and in one case it changes
the trajectory's dynamical class, turning a fixed point into a two-step cycle (claim
C-24, one prompt). These are the supported rows, and each carries its sample size, which
is small, and its single site.

**What the random-direction control settled.** An arbitrary near-rank-1 direction of the
same size usually moves the settled word too, in 4 of the 6 cases that could be
size-matched, so the edit is not special merely for having structure (claim C-55, which
retired the earlier "the right sign is required" claim, C-22). But no random direction
ever reaches the Hebbian rule's own destination, 0 of 10 random seeds, and matching its
effect costs 66 to 172 times as much weight change. So direction does not decide whether
the landscape moves; it decides where it moves to, and how cheaply.

**Coupling is reported as a refinement, not the headline.** The project's founding
interest is coupling: the weights changing while the activations they change feed back
into them. That has been measured (claim C-31): the feedback-attributable part of the
weight change is 12% of the total, against a severed-path control whose floor is exactly
zero, not a small number. But at the operating point tested, feedback changes the weights
without changing the outcome, since the connected and disconnected runs settle on the
same word (claim C-33). The honest one-line form is that feedback measurably steers the
update and does not, at this operating point, change the result. Whether that holds as
coupling grows is the highest-value open experiment, T2.1 (issue #48).

**Two honest limits on the lead.** First, on what the edit actually does: the
"created attractor" reading is retired (claim C-26). The coexistence test (T1.1, issue
#45) shows the edit displaces the single settled state rather than creating a second one
beside it, and the hysteresis sweep (T1.2, issue #46) retraces cleanly, corroborating
the same reading by an independent route (claim C-56). The edit relocates the one
settled state; it does not add another.
Second, on novelty: model editing, in particular ROME, already makes near-rank-1 edits to
a mid-stack GPT-2 projection, so the new element is not "a rank-1 edit changes
behaviour." It is that this edit is derived with no target, from the model's own
activity, and is read out on the iterated map's settled-state structure rather than on
next-token output. Stated any other way, the comparison to prior work is lost.

## Where this sits

| Rung | Weights | Status in the literature |
|:---|:---|:---|
| 1 | Frozen | Done — the parent ATR project |
| 2 | **Oja rule on one site** | **Empty — this repo** |
| 3 | **Local rules on every head** | **Empty** |
| 4 | Gradient fine-tuning on own output | Occupied — recursive training / model collapse |

Rung 1 is the ε→0 limit of rung 4. The interesting unexplored region is the
middle, and it is cheap to reach.

## Why Oja rather than Hebb

**Corrected 2026-08-01.** This section used to open by saying that raw Hebbian
updates diverge immediately, with no fixed point and unbounded weight growth.
That was wrong at the step size this project actually runs at, and the register
retires it as row C-15. What the committed sweep records is below.

Raw Hebb is bounded and finite everywhere it has been measured here. At the
working point (step size 7.07e-05, 120 updates, one update per iteration) it
gives zero non-finite values, a clip rate of 0.0% meaning the safety ceiling
never fired, and a weight-norm change of +0.03%. Across all ten `hebb` cells of
the step-size map the non-finite count is zero in every one
([STEP_SIZE_MAP.md](STEP_SIZE_MAP.md)). At the larger step sizes total drift
stops at the 5% ceiling with the clip rate climbing to 99.2%, so those cells
measure the ceiling and say nothing either way about unbounded growth.

The divergence claim survives only in a narrower regime, and there it is real:
with the ceiling lifted (control C3 runs at `max_delta_frac=1e9`) and a step
size of 1e-3, roughly fourteen times the working point, the Hebb drift trace
grows monotonically and super-linearly over a few applications while the Oja
trace saturates. Note the limit of that evidence: C3 runs a fixed, small number
of applications, so what it records is continued growth over the run measured,
not an unbounded limit. Nothing here shows Hebb never levels off. The
defensible statement is that at large step size with the ceiling removed,
Hebb's drift keeps growing over the applications measured while Oja's settles,
not that Hebb diverges immediately or without bound.

One further correction from the same measurement, because it inverts the
intuition: at every stable step size Oja's own update is about a hundred times
**larger** than Hebb's in absolute terms, since at a real weight scale of
`‖W‖_F = 164.9` the decay term dominates the reinforcement term rather than
correcting it. "Hebb diverges, Oja does not" is a claim about growth across a
run. It is false as a claim about which update is bigger at any single step.

Oja's rule adds a decay term proportional to post-synaptic activity:

```
Hebb:  dW = <x yᵀ>
Oja:   dW = <x yᵀ> − W <y yᵀ>
```

The decay is not a convenience. **Oja's rule is Hebbian learning with
normalisation built in, and it performs power iteration on the input correlation
structure.** ATR is nonlinear power iteration on activations. So the activation
loop and the weight loop are the same mathematics one level apart, which means
the normalisation question in the parent repo and the learning rule here are the
same question asked twice. That's the theoretical spine of this branch, and it
is worth stating in any write-up.

`mode="hebb"` is included so you can produce the divergence figure (control C3)
rather than asserting it.

**And the section title has not survived the experiments.** The theoretical
argument above still stands, but it did not predict what happened. Oja's rule
was run at eight step sizes spanning five orders of magnitude and moved the
loop's settled word at none of them, including the ceiling-silent cells up to
2.9% drift (register row C-13). Every result in which the loop's behaviour
changed came from `hebb`, the mode included as a foil. The other rules are not
absent from the record: the step-size map and the C3 traces record them, and
what they record is that nothing moved. Why Oja is inert here is **not**
established: the leading explanation, that the brake outweighs the
reinforcement term by roughly 110 to 1 at this site, is row C-14 and has never
been tested as an explanation. Read this section as the reason the project
started with Oja, not as a finding.

## Hardware

A laptop. GPT-2 Small is 124M parameters, ~500MB in fp32; the parent project has
already run full sweeps on a CPU cloud container. A Hebbian update is an outer
product. No backpropagation, no optimiser state, no dataset, no cluster.

## Architecture

**This repo does not contain an ATR loop, on purpose.** `plasticity.py` installs
hooks on one weight matrix and applies a learning rule on request. You wrap it
around the tested engine from the parent repo:

```python
from plasticity import OjaPlasticity
from atr_engine import atr_step        # from the parent project

with OjaPlasticity(model, site="transformer.h.6.mlp.c_proj",
                   eta=1e-6, mode="oja") as plast:
    r = initial_tensor
    for i in range(n_iter):
        r = atr_step(model, r)
        if (i + 1) % 4 == 0:
            plast.apply()
        log(i, decode(r), plast.report())
```

The plasticity layer is the only new, untested component. Keep it that way —
if you reimplement the loop here, a bug in the reimplementation becomes
indistinguishable from a plasticity effect.

## Choosing a site

`candidate_sites(model)` enumerates the options. Preferred order:

1. **`mlp.c_proj`** — the MLP down-projection. Pre- and post-synaptic activity
   both cleanly defined, least entangled place to perturb. **Start here.**
2. `attn.c_proj` — the OV output circuit.
3. `mlp.c_fc` — pre-nonlinearity, so "post-synaptic activity" is murkier.

Avoid `attn.c_attn` initially: it packs Q, K and V into one matrix, so a
Hebbian update there is three experiments at once.

Layer choice: mid-stack (6 of 12 in GPT-2 Small) is the conventional starting
point, but this is an assumption worth sweeping once C0–C2 pass.

## Running the controls

In order. Do not skip.

| ID | Question | Failure means |
|:---|:---|:---|
| **C0** | At `eta=0`, do the hooks perturb the trajectory? | **Stop.** Contamination; nothing downstream is interpretable |
| **C1** | Does `revert()` restore the original trajectory? | Hidden state accumulating somewhere |
| **C2** | Does a random update of matched norm do the same thing? | You've measured "perturbing weights changes things", not Hebbian learning |
| **C3** | With the ceiling lifted and a large step size, does Hebb's drift keep growing where Oja's saturates? | The decay term isn't doing its job |

```python
from controls import c0_identity, c1_revert, c2_random_direction, c3_divergence_demo

print(c0_identity(model, r0, atr_step, site="transformer.h.6.mlp.c_proj"))
```

**C2 is the one that decides whether this branch is interesting.** If Oja and a
norm-matched random matrix produce the same landscape change, the finding is
about perturbation magnitude, not about learning.

## Learning rate

There is no gradient and no optimiser here. Start at `eta=1e-6` and increase by
half-orders. `max_delta_frac` (default 0.05) caps total drift from the original
weights at 5% Frobenius norm and flags `clipped` when it bites. If you're
clipping, either eta is too high or you have found the interesting regime — check
which by looking at whether the landscape changed before the clip fired.

## What to expect (and the reason it's worth doing anyway)

From the biological analogue: **collapse is the likely default.** Dissociated
cortical cultures of locally-plastic units, with no external drive and no body,
reliably fall into globally synchronised bursting — one low-dimensional attractor
that destroys the network's capacity to express anything else. The Potter lab
spent fifteen years on this, and their remedy was continuous distributed
stimulation to hold the network off its attractor (Wagenaar, Madhavan, Pine &
Potter 2005, *J Neurosci* 25:680).

So expect plasticity alone to make the landscape worse, and treat an external
drive term as the parameter that buys you interesting dynamics:

```
r_next = normalise( f(r) + β·e_prompt )
```

That is experiment E3 in the parent repo's normalisation issue. **The homeostasis
experiment and the plasticity experiment are the same experiment, run in the
other order** — which is the strongest reason to keep both repos in view at once.

## Running the test suite

The controls above are the gate on *results*. The test suite is the gate on the
code that runs them. **Every test runs against real GPT-2 small.**

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest                 # 295 tests (4 skip without the parent repo), ~2-3 min on CPU; downloads gpt2 (~500MB) once
```

A Claude Code session does this for you: `.claude/hooks/session-start.sh` builds
the venv on session start and exits early once it is there.

There is no toy model and no fast/slow split. There was a toy, and it was removed
on purpose: its `Conv1D` was **our own reimplementation**, so any way in which it
diverged from HuggingFace's would have passed every test in the suite. A stand-in
that can quietly disagree with the thing it stands in for is not a test of the
thing. Two assertions that had passed against the toy turned out to be false on
real weights — see the Hebb/Oja note below.

What the suite checks: that `transformer.h.6.mlp.c_proj` really is a `Conv1D` of
shape `(3072, 768)` in the `(n_in, n_out)` convention the rules assume; that the
learning rules match a closed form reconstructed from real activations; that C0
holds bit-exactly on 124M real parameters and *fails* when handed a hook that
perturbs; that `revert()` restores a real matrix bit-exactly; and that the C2
norm-match holds at real activation scale.

One `nn.Linear` fixture survives, clearly labelled as a code-path fixture rather
than a model: GPT-2 is Conv1D at all 48 of its candidate sites, so nothing real
exercises `transposed=True`.

CI runs the whole suite on every push, with the checkpoint cached. It sets
`ATR_REQUIRE_MODEL=1`, which turns "GPT-2 unavailable" from a skip into a
failure — without it a runner missing the model skips all 295 tests and exits 0,
and a check that cannot fail is worse than no check.

**Hebb and Oja invert between the toy and the real model.** At the real weight
scale `‖W‖_F = 164.9`, the Oja decay term `W<y yᵀ>` *dominates* the Hebb term
`<x yᵀ>` — mean update norms 34,685 against 331, a factor of ~100 — where on the
toy the two were within a percent. "Hebb diverges, Oja does not" is a claim about
growth across a run, not about which update is larger at any step. An eta
calibrated from a Hebb trace is orders of magnitude wrong for Oja at the same
site.

**One observation worth recording.** C0 on real GPT-2 has been seen to fail
intermittently on CPU, twice, at deviations of 8.6e-05 and 6.3e-05 — and could
not be reproduced in 80 controlled repeats, quiet and under load, nor in 16 cold
processes. An unhooked-vs-unhooked control never differed. The hooks do not
mutate anything, so the likely cause is nondeterministic parallel reduction
order rather than contamination. If a `bit_exact` failure appears at that
magnitude, suspect this before suspecting the hook — and note that DESIGN.md
already lists nondeterminism as a way to get a wrong answer here.

**A passing suite is not C0 passing.** The tests check that the learning rules
compute what they claim, that the ceiling and `revert()` hold, and that each
control can fail when handed the defect it exists to catch. They say nothing about
whether the hooks perturb a real transformer's trajectory. That is still C0's job,
against real weights, and it is still the first thing to run.

## Files

```
plasticity.py   OjaPlasticity: hooks, Oja/Hebb/random rules, delta tracking, revert
controls.py     C0-C3, each taking your atr_step as an argument
tests/          pytest suite, all of it against real GPT-2 small
EXP_001_SPEC.md the proposed first experiment, and the bridge it needs first
DESIGN.md       measurement plan, failure modes, what would falsify what
requirements.txt        torch, transformers -- the experiment
requirements-dev.txt    pytest and transformers -- the suite
pyproject.toml          pytest configuration
```

## License

Match the parent project.
