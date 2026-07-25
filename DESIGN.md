# Design

## Hypotheses, and what would kill each

**H1 — Consolidation.** Activation-driven Oja updates deepen existing basins:
the basin of attraction widens, convergence happens in fewer iterations, and the
basin assignment for borderline prompts becomes more stable across reruns.

*Killed by:* no change in basin assignment or convergence iteration beyond what
C2's random-direction control produces.

**H2 — Collapse.** Plasticity reduces the number of distinct basins, driving the
landscape toward the single-funnel regime already observed in GPT-2 Medium and
Pythia-160m without any plasticity at all.

*Killed by:* basin count stable or increasing under plasticity.

H1 and H2 are not exclusive — the plausible outcome is consolidation at small
eta followed by collapse past a threshold, in which case the interesting result
is **where the threshold is and what it depends on**.

**H3 — The cycle is fragile.** The `Divine` period-2 limit cycle (parent repo
F9) does not survive plasticity: the weight change breaks the exact
`cos(A, f(f(A))) = 1.000000` relation.

*Killed by:* the cycle persisting to machine precision under updates. That would
be a striking result in its own right — a dynamical object robust to
modification of the map that generates it.

**H4 — Drive dependence.** Whatever happens under H1/H2 depends on an external
drive term β, and the plasticity outcome at β=0 is the degenerate case.

*Killed by:* landscape changes under plasticity being invariant to β.

## Measurement plan

Every run logs, per iteration:

- decoded top-1 token, and full-distribution entropy / logit margin (reuse the
  parent repo's readout-confidence code — do not rewrite it)
- consecutive-iteration cosine at **lag 1 and lag 2** (F9's lesson: a lag-1 gate
  cannot pass a period-2 cycle by construction)
- `delta_norm`, `delta_frac`, `clipped`, `nonfinite` from `report()`

Per run:

- terminal attractor and iteration of lock-in
- basin assignment over the 125-prompt library, for comparison against the
  parent repo's convergence matrix
- the delta matrix itself, saved — its spectrum is a result, not just a
  diagnostic. If the update is dominated by one or two singular directions,
  that connects to the anisotropy story in the parent repo's isomorphism.

## Snapshot schedule — read this before running anything

**Do not use an even-only snapshot schedule.** F9 in the parent repo was hidden
for months because every schedule sampled at even iterations, and an even-only
schedule samples a period-2 orbit at a single phase, making the oscillation
invisible by construction.

Use consecutive-iteration snapshots in any late band you care about, or at
minimum a schedule with both parities. Aliasing is the single most likely way to
get a wrong answer here, and adding plasticity gives the system more ways to be
periodic, not fewer.

## Failure modes to watch for

| Symptom | Likely cause |
|:---|:---|
| C0 fails | Hook has a side effect; dtype cast leaking; in-place op on a captured tensor |
| `nonfinite` fires early | eta too high, or `mode="hebb"` (expected) |
| `clipped` fires immediately | eta far too high; the ceiling is doing its job |
| Landscape identical to baseline at every eta | Site is not load-bearing for the dynamics; try a different layer or `attn.c_proj` |
| Oja and random controls agree (C2) | The result is about perturbation magnitude; the branch as framed is dead and needs rethinking |
| Basins change but so does the eta=0 rerun | Nondeterminism somewhere; check the parent repo's reproducibility gate still passes on this machine |

## Sweep order

1. C0, C1 on a single prompt, 50 iterations. Gate.
2. C3 divergence demo — one figure, establishes the rule choice.
3. Single prompt, eta ∈ {1e-6, 3e-6, 1e-5, 3e-5, 1e-4}, 500 iterations. Find
   the eta where anything happens at all.
4. C2 at that eta. **Gate — if this fails, stop and rethink.**
5. Five-prompt piece (the original Act I set) at that eta. Compare terminal
   attractors and dissolution waypoints against the known baseline.
6. 125-prompt sweep at 2–3 eta values. Convergence matrix, compared block-for-block
   against the parent repo's.
7. Repeat step 5 on GPT-2 Medium — does `D` open up?
8. Add the drive term β; 2-D sweep over (eta, β).

Steps 1–5 are days. Steps 6–8 depend on how long a 125-prompt sweep takes on
your hardware, which the parent repo already knows.

## Open design questions

- **Cadence.** Applying an update every iteration couples the two loops
  tightly; applying every k iterations approximates a timescale separation.
  k is a free parameter with a biological interpretation (the ratio of fast to
  slow dynamics) and should eventually be swept, not fixed.
- **One site or many.** Rung 3 of the ladder ("each head has a looping training
  function") requires updating every head simultaneously with a local rule. The
  scaffold supports one site. Extending to many is mechanically easy and
  interpretively much harder — do it only after single-site results exist.
- **Whether to renormalise the weights.** Oja's decay term stabilises the
  update, but nothing constrains the weight matrix's overall scale over long
  runs. A separate weight-norm homeostasis may be needed, which would be a
  second homeostatic mechanism layered on the first — worth noticing as an
  echo of the parent repo's normalisation question rather than fixing
  reflexively.
