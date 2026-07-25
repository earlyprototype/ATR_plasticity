# Hebbian ATR — plastic-weight iterated dynamics

*What happens to a model's attractor landscape when the weights are allowed to
change under the loop?*

> **Status: scaffold. Nothing here has been executed against real weights.**
> There is now a test suite (`pytest`, no download required) covering the
> learning rules and the controls against a toy network. It found three defects,
> all since fixed and each now held by a test that fails if it returns:
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
> against real weights before anything here produces a result worth recording.

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

Raw Hebbian updates diverge immediately — no fixed point, unbounded weight
growth. Oja's rule adds a decay term proportional to post-synaptic activity:

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
| **C3** | Does raw Hebb diverge and Oja not? | The decay term isn't doing its job |

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

The controls above are the gate on *results*. The test suite below is the gate on
the code that runs them — it does not need a real model, a download, or a GPU.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

A Claude Code session does this for you: `.claude/hooks/session-start.sh` builds
the venv on session start and exits early once it is there.

Everything runs against a toy network in `tests/conftest.py` whose module tree
mirrors the parts of GPT-2 the plasticity layer reaches for — Conv1D weights of
shape `(n_in, n_out)`, dotted paths like `transformer.h.1.mlp.c_proj`. That keeps
the default suite offline and fast. Tests that would need real GPT-2 weights are
marked `slow` and excluded from a bare `pytest`; run them with `pytest -m slow`.

**A passing suite is not C0 passing.** The tests check that the learning rules
compute what they claim, that the ceiling and `revert()` hold, and that each
control can fail when handed the defect it exists to catch. They say nothing about
whether the hooks perturb a real transformer's trajectory. That is still C0's job,
against real weights, and it is still the first thing to run.

## Files

```
plasticity.py   OjaPlasticity: hooks, Oja/Hebb/random rules, delta tracking, revert
controls.py     C0-C3, each taking your atr_step as an argument
tests/          pytest suite: toy-model fixtures, rule correctness, control gates
DESIGN.md       measurement plan, failure modes, what would falsify what
requirements.txt        torch, transformers -- the experiment
requirements-dev.txt    pytest -- the suite
pyproject.toml          pytest configuration
```

## License

Match the parent project.
