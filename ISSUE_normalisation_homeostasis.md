# Normalisation as absorption: what the L2 rescale does and doesn't do

**Type:** method / open question / experiment design
**Related:** #DYNSYS (dynamical-systems interpretability positioning — replace
with issue number once created)

## Summary

The ATR loop rescales the activation tensor to its initial L2 energy at every
iteration. This was introduced as an analogue of room-surface absorption: a
crude coefficient preventing runaway growth. Interrogating that choice turns up
one correction to the stated isomorphism, one claim we can defend more strongly
than before, and four cheap experiments — each of which suggests a current
finding is a slice through an unvaried parameter.

## Correction to the isomorphism

The current framing implies the L2 rescale plays the role of absorption. It
doesn't, and the honest version is stronger.

Room resonance arises because iterated convolution with the room's impulse
response repeatedly multiplies the signal by one operator; whatever that
operator has **highest gain** on survives. The unequal gain across frequencies
comes from standing-wave geometry *and* frequency-selective absorption together.
Lucier's room tone is the dominant eigenmode winning by repeated
multiplication — power iteration, in air.

Our L2 rescale is **frequency-flat**: a single scalar applied to every
component. It cannot select modes. The mode selection in ATR is done entirely by
the **directional anisotropy of the transformer forward map** — unequal gain
across directions in activation space, which lives in the weights.

Corrected correspondence:

| Room | ATR |
|:---|:---|
| impulse response | forward map `f` |
| frequency-selective structure (geometry + selective absorption) | directional anisotropy of `f` |
| broadband level control | L2 gain control (what we actually have) |
| dominant surviving mode | attractor |

So: **the normalisation is gain control, not absorption, and the
frequency-dependence lives inside `f`.** This makes the isomorphism more
defensible, not less — we are no longer claiming the normalisation does
something it can't.

Action: update ISOMORPHISM.md. This is the section that most needs it.

## What the rescale can and cannot do

**Exactly, instantaneously:** scaling multiplies every component by the same
scalar, changing magnitude only. The direction is unchanged — a purely radial
move along the same ray. Therefore the rescale **cannot rotate the state**, and
since cosine similarity is computed on normalised vectors and ignores magnitude
entirely, **the convergence gate cannot see the rescale at the moment it is
applied.**

**Indirectly, at the next step:** `f` is nonlinear, so `f(c·r) ≠ c·f(r)` in
general. The magnitude fed in can change the direction that comes out next
iteration. The rescale doesn't rotate the state; it changes what the map does
with it afterwards.

**How much is an open empirical question**, because GPT-2 applies LayerNorm at
the input of every block and LayerNorm largely discards input scale. It is
possible the rescale is very nearly inert. See E1.

## Three axes on which this differs from biological homeostasis

Relevant if the loop is ever framed as a homeostatic mechanism rather than
merely a numerical guard:

1. **Global vs local.** One scalar for the whole tensor, vs per-unit set-points.
2. **Instantaneous vs slow.** Exact reset every step, vs integration over
   minutes to hours with lag and overshoot.
3. **Direction-preserving vs direction-shaping.** Magnitude-only, vs
   multiplicative synaptic scaling which adjusts individual weights and
   therefore *does* change direction.

Axis 3 is the one that matters for interpretation. **[SUPERSEDED — no register
row supports this, and this file contradicts it two sections up.]** *Because our
rule is direction-preserving, all observed basin structure is attributable to the
map rather than to our normalisation choice.* That does not follow, and "What the
rescale can and cannot do" above already says why.

The defensible version: the rescale **cannot itself rotate the state
instantaneously** — it is a single scalar, a purely radial move along the same
ray. But `f` is **nonlinear**, so the magnitude the rescale sets can change the
**next** iterate's direction, and a landscape read off an iterated map is a
property of the composition, not of one step of it. **Whether the coefficient is
load-bearing is an open empirical question, and E1 is the unrun test.** Until it
runs, every basin result in the repo is a slice at one arbitrary target energy,
and no document may say the normalisation has been ruled out as a contributor.

The direction-preserving property is still the right thing to relax if we want to
study homeostasis rather than merely impose it — that part of the paragraph
survives.

## Experiments

### E1 — Target-energy sweep (cheapest; do first)

Sweep the target energy `E` over 2–3 orders of magnitude, 125-prompt sweep at
each. Does the basin assignment move?

- **Invariant** → LayerNorm absorbs the rescale; the coefficient is inert; we
  can state "the landscape is invariant to the absorption coefficient", which
  substantially strengthens every existing claim.
- **Moves** → scale is load-bearing and the choice of `E` needs justification;
  every existing result is a slice at one arbitrary `E`.

Either outcome is worth reporting. Effort: hours.

### E2 — Leak term / damping (resolves an F9 ambiguity)

Replace `r_next = normalise(f(r))` with:

```
r_next = normalise( (1−α)·r + α·f(r) )
```

Current runs are α=1. Sweep α ∈ (0, 1].

Rationale: damping can convert a limit cycle into a fixed point. Toy case:
`f(x) = 1 − x` from x=0 oscillates 0,1,0,1 forever; at α=0.5,
`r_next = 0.5x + 0.5(1−x) = 0.5` — a fixed point in one step. Averaging pulls
the two phases of the oscillation together until they merge.

**Implication for F9:** whether `Divine` is a period-2 limit cycle or a fixed
point may be a property of α, not of the model. F9 should be re-read as "at
α=1". Also possibly relevant to the 34 prompts blocked on gate re-design (#9),
and to Pythia-410m's non-consolidation.

### E3 — External drive (the strongest experiment; from the MEA literature)

```
r_next = normalise( f(r) + β·e_prompt )
```

Sweep β from 0. Variant: fresh calibrated noise instead of the prompt embedding
at each step.

Rationale, from the biological analogue: dissociated cortical cultures collapse
*spontaneously* into globally synchronised bursting — thousands of spikes in
0.1–2 s recruiting essentially the whole network. That is dynamical collapse
onto a single low-dimensional attractor, and it destroyed the Potter lab's
ability to observe anything else. One remedy was **continuous distributed
electrical stimulation to quiet bursts** (Wagenaar, Madhavan, Pine & Potter
2005, J Neurosci 25:680): external drive holding the network off its attractor.
That paper's own result — multi-electrode stimulation quiets bursting, and
bursting resumes when it stops — is verified independently and stands
(`MEA_SOURCES.md`).

**Correction, and it goes the other way for this experiment.** This paragraph
used to add that *burst-quieting was the one protocol in **Searching for
Plasticity** (2006) where a real effect became detectable*, immediately after
crediting distributed electrical stimulation, which reads as crediting the
electrical protocol. **The published correction to that paper — Journal of
Negative Results in Biomedicine 6:3 (2007), against the original 5:16 —
swaps the two protocol labels throughout.** The protocol that worked suppressed
bursting **with added magnesium**, not with distributed electrical stimulation,
and the correction instructs that the discussion paragraph crediting distributed
electrical stimulation be dropped entirely. The uncorrected text is still
publicly served, which is how this got in; see `MEA_SOURCES.md`, which records
that an earlier draft of that file made the same mistake.

What survives is the weaker and more general point, and it is the one E3 should
lean on: **quieting bursts aids the induction and detection of lasting
plasticity** (Potter, 2008 review chapter, quoted in `MEA_SOURCES.md`). What does
*not* survive is the specific claim that **external electrical drive** was the
means — in the one protocol that worked, the quieting was pharmacological. E3
imposes drive, which is the analogue of the label the correction removed, not of
the one it installed. That does not sink E3, but the biological precedent is now
for *quieting by any means*, and this file should not imply the literature
already picked drive.

Compare also Chao, Bakkum, Wagenaar & Potter (2005) on random background
stimulation stabilising networks after tetanization — noting from
`MEA_SOURCES.md` that this one is a **simulation** of 1000 integrate-and-fire
neurons, not a living-culture measurement.

**Hypothesis:** GPT-2 Medium's collapse to `D` by iteration 10 is the analogue
of runaway global bursting. Predict that Medium's single funnel opens into
multiple basins at some β > 0.

**If confirmed**, "number of basins" is not a model property but a function of
drive amplitude, and the Act II cross-model table becomes a slice at β=0. This
is the experiment most likely to reframe the series. Effort: days.

### E4 — Principled homeostasis (later; do after E1–E3)

Replace the instantaneous rescale with a proportional-integral controller on the
norm: a target set-point, a gain, an integration time constant. Prior art with
the stability analysis already done: **Newman et al. (2015), "Optogenetic
feedback control of neural activity", eLife 4:e07192** — a real-time PI
controller clamping network firing rate via optogenetic drive. The controller
design section is directly transferable, including the failure modes.

Also worth trying: per-layer or per-head normalisation (local set-points, which
introduce direction changes), and scaling `W_V`/`W_O` per head to hit a target
activation norm — a direct analogue of multiplicative synaptic scaling.

## Tasks

- [ ] Update ISOMORPHISM.md with the corrected absorption/anisotropy mapping
- [ ] Add the direction-preserving property to TECHNICAL.md — the property only,
      **not** the "so the basin structure is all the map's" consequence, which is
      withdrawn above and awaits E1
- [ ] Run `spectral_resonance.ipynb` (currently scaffold-only) — SVD-predicted
      per-head resonance is the direct test of whether `f`'s anisotropy predicts
      the basins, and is now the missing link in the corrected isomorphism
- [ ] E1: target-energy sweep
- [ ] E2: leak-term sweep; re-read F9 as "at α=1"
- [ ] E3: external-drive sweep on GPT-2 Medium
- [ ] E4: PI-controller normalisation, after E1–E3
