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
- the delta matrix's spectrum — a result, not just a diagnostic. If the update
  is dominated by one or two singular directions, that connects to the
  anisotropy story in the parent repo's isomorphism.

  **This line used to say the delta matrix itself is saved. It is not, and
  nothing else is either.** `experiments/output_*` holds summary statistics
  only: no dense ΔW and no raw state is persisted anywhere in the repository
  (`.gitignore:27-29` excludes the state directories and both array formats,
  and no runner writes a ΔW at all). The consequence is a standing cost, not a
  tidiness point — every downstream analysis has to **regenerate** the episode
  rather than load it, and pays a float drift of roughly **0.1%** for doing so:
  `basin_bifurcation.py` matches the weight anchors to about nine figures and
  still lands 0.12% off on the closed-loop state norm, and each such
  regeneration compounds on the last. Cross-reference **C-47** (`retired`),
  which is the identical failure for the settled states — `BASELINE.md`
  promised 125 saved states for exactly the within-basin re-analysis,
  `.gitignore:27` excludes them, the directory does not exist, and the summary
  statistics survive where the raw objects do not. Re-deriving those costs the
  ~6 CPU-hour baseline re-run. If ΔW is going to be reused as an object, which
  issue #32 §5 already assumes, it has to be persisted first.

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
| `nonfinite` fires early | eta too high. **Not** "expected under `mode="hebb"`": across `hebb`'s **five ceiling-silent cells** in the step-size map, drift up to **2.20% at 0.0% clip**, it never once went non-finite, and at the working point ‖W‖_F moved **+0.03%** (register row C-15) |
| `clipped` fires immediately | eta far too high; the ceiling is doing its job |
| Landscape identical to baseline at every eta | Site is not load-bearing for the dynamics; try a different layer or `attn.c_proj` |
| Oja and random controls agree (C2) | The result is about perturbation magnitude; the branch as framed is dead and needs rethinking |
| Basins change but so does the eta=0 rerun | Nondeterminism somewhere; check the parent repo's reproducibility gate still passes on this machine |

**The `nonfinite` row is rescoped, 2026-08-05.** It used to rest the no-divergence
claim on "all 35 cells of the step-size map". That is the wrong evidence twice
over. Only 10 of the 35 cells run `hebb` at all, and **five of those ten fired the
norm ceiling** (7.5%, 43.3%, 83.3%, 95.8% and 99.2% clip) — a clipped cell has its
drift bounded by `max_delta_frac` by construction, so it cannot bear on whether the
rule itself diverges; it measures the ceiling. The claim is true and now rests only
on what can carry it: `hebb`'s five ceiling-silent cells. The clipped cells stay in
`STEP_SIZE_MAP.md`'s table as diagnostics. Step 2 of the sweep order below states the
other half of C-15's scope, which is that nothing here licenses "without bound".

## Sweep order

1. C0, C1 on a single prompt, 50 iterations. Gate.
2. C3 divergence demo, one figure. **It did not establish the rule choice, and
   the sweep since went the other way.** C3 shows Hebb's drift growing where
   Oja's saturates only with the ceiling lifted at a large eta; at the step
   sizes actually used, Hebb is bounded and finite (C-15), and C3's evidence is
   continued growth over the fixed small number of applications it runs, not an
   unbounded limit. Oja then moved the basin at none of the **five** step sizes
   tested with the ceiling silent — up to **2.9% drift at 0.0% clip**, which is
   what C-13 rests on. The remaining three of its eight cells fired the ceiling
   (9.2%, 65.8% and 100% clip) and are **diagnostic only**: Oja stays `prolet`
   in those too, but a clipped cell measures `max_delta_frac` rather than the
   rule, so it is a note and not evidence. Every result in which the loop's
   behaviour changed comes from `hebb`. The other rules are recorded throughout
   the step-size map; what they record is that nothing moved.
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

- **Cadence — swept, and this question is answered.** Applying an update every
  iteration couples the two loops tightly; applying every k iterations
  approximates a timescale separation. k is a free parameter with a biological
  interpretation (the ratio of fast to slow dynamics), and this line's "should
  eventually be swept, not fixed" has since been done: **k = 1, 2, 4 and 12**
  have run (C-64) — 2 and 4 in T2.1 alongside its cadence-1 cells, and 1, 4 and
  12 in EXP-003 Stage 2 (`stage2.jsonl` field `k`, `ladder: [1, 4, 12]`).
  Loosening the coupling does what the framing predicts. The feedback-attributable share of
  the weight change falls as k rises, **0.1204 at k=1, 0.0543 at k=2, 0.0254 at
  k=4** (C-58), which is the same coupling-strength monotonicity read along this
  axis rather than a separate effect. Two limits travel with that. Cadence above
  1 has only ever run at **one site** (`t2_1_coupling.jsonl`) apart from Stage 2;
  and **Stage 2's own registered drift guard fired** — its runner scaled eta by k
  on the assumption that drift is linear in eta, it is not, and the drift spread
  across that ladder came out **6.03×** against a required factor of 2, so
  `drift_matched` is false, cadence there is confounded with drift, and nothing
  in it separates a slow-timescale effect from a larger-drift one (C-67).
- **One site or many — answered, and the interpretive half of the warning was
  right.** Rung 3 of the ladder ("each head has a looping training function")
  requires updating every head simultaneously with a local rule. This line used
  to say the scaffold supports one site and that extending to many should wait
  for single-site results; both halves are now stale. `multi_site.py` is in the
  repository, and EXP-002 carried plasticity at **all twelve MLP
  down-projections at once** (C-60), after the single-site series had run. What
  survives is the warning that many sites are interpretively much harder — and
  it is no longer a worry but registered fact. A multi-site run **lifts the drift
  ceiling** (C-60, `max_delta_frac` 1e9, so it is a different regime from every
  result taken under the 5% cap) and **loses the exact-zero severed-path floor**
  (C-63: only the *lowest* plastic layer floors at zero, because a lower layer's
  drift reaches a higher one inside a single forward pass and severing the loop
  does not cut that). Multi-site closed-versus-offline numbers therefore have no
  zero baseline, and **may not be placed in one series with the single-site
  shares** (C-64) — wider coverage cannot be assembled by adding them up. Rung 3
  itself is still untouched: **0 of 144 head stripes** have carried plasticity in
  a committed experiment (C-64).
- **Whether to renormalise the weights.** Oja's decay term stabilises the
  update, but nothing constrains the weight matrix's overall scale over long
  runs. A separate weight-norm homeostasis may be needed, which would be a
  second homeostatic mechanism layered on the first — worth noticing as an
  echo of the parent repo's normalisation question rather than fixing
  reflexively.
