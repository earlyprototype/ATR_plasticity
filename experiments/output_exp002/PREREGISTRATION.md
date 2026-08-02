# EXP-002 pre-registration — distributed plasticity, then a fresh prompt

Written and committed **before** anything runs. Issue #24 (the sequence), issue #26 (the
controls that make it count), issue #29 (the persistence question step 4 reaches).
Durable claims **C-60, C-61, C-62, C-63**, all claimed on the Identifier registry
(discussion #17) before use.

Operator mandate: run the whole sequence, not the cautious first two steps, and drive
**every layer** rather than the single site every prior result in this repository used.

## What is new here, and what is borrowed

Every committed result so far moved **one** weight matrix, `blocks.6.mlp`. This runs
plasticity at **all twelve MLP sites at once** (`blocks.0.mlp` … `blocks.11.mlp`) using
the existing `MultiSitePlasticity` driver, which has tests but has never produced a
recorded result. The learning rules, the ceiling, the revert guarantee and the offline
replay are all borrowed unchanged from the single-site path.

## The sequence (issue #24)

1. **Collapse.** Frozen loop on prompt `A01_physics`, no plasticity, until settled. This
   is the starting condition, not a result.
2. **Work the well.** Plasticity on at all twelve sites, closed loop, 120 updates.
3. **Stabilise.** Plasticity off, weights left where they drifted.
4. **Reprompt.** Fresh prompts the drifting never saw, run frozen under the drifted
   weights.
5. **Measure.** Per-site weight movement, and where the fresh prompts land.

## Arms

Two rule arms, both at all twelve sites, both with the full control set:

- **`anti_hebb`** — issue #24's literal mechanism: erode what the state settled into so
  it can climb out of the well.
- **`hebb`** — the only rule that has ever moved this repository's settled state (C-13
  records that `oja` is inert at every step size tested, and every behaviour-changing
  result came from `hebb`). Included because the issue's mechanism and the repository's
  evidence point at different rules, and guessing between them before the run is exactly
  what a pre-registration is supposed to prevent.

## The offline arm, and how it is built

Issue #26 makes this blocking: **without it the result means nothing**, because Oja-family
rules move weights with no feedback at all, so a closed-loop change on its own proves
only that the rule ran.

`offline_control.py` is single-site by signature. The multi-site arm is therefore built
from the tested single-site path rather than from new capture code:

1. The frozen loop is deterministic, so recording site *k* in its own pass produces the
   same data as recording all twelve in one pass. Each site is recorded with the tested
   `record_frozen_activations`.
2. Offline replay has no feedback, so each site's offline trajectory is independent of
   every other site's. Each is replayed with the tested `replay_offline`.
3. The twelve resulting matrices are installed together, and the loop is re-run frozen.

**What the difference between arms measures here is broader than in the single-site
case, and this is stated in advance so no one reads it as the same quantity.** With one
plastic site, the closed-versus-offline difference is state feedback alone. With twelve,
it is state feedback *plus* the cross-site interaction: in the closed loop the
activations arriving at layer 7 have already passed through the drifting layers below
it, and in the offline replay they have not. Both are consequences of running the loop
closed, so both belong in the feedback-attributable quantity, but the two cannot be
separated by this design and no attempt is made to separate them.

## The severed control, and an honest limit on it

The project's standing rule 3 says the severed-path control is the baseline for any
claim that feedback did something, and that its floor is exactly zero. The severed
control works by reading the loop **below** the plastic site, so the site cannot reach
the next iterate.

**With plasticity at all twelve layers there is no such readout point**, because
`blocks.0.mlp` sits below any layer the loop could be read at. The exact-zero floor
therefore cannot be established for the full twelve-site configuration. This is a
property of the configuration, not an oversight, and it is recorded as a limitation on
every number this experiment produces.

What is run instead, so the floor is not simply abandoned: a **severable
sub-configuration** with plasticity at `blocks.4.mlp` … `blocks.11.mlp` (eight sites) and
the loop read at `blocks.3.hook_resid_post`. Its recomputed-y floor must come out exactly
0.0, bit-identical, as it does at one site. That establishes the instrument's null for
the multi-site machinery; it does not establish it for the twelve-site runs, and the
distinction travels with the result.

## Learning rate

The working step size was anchored to `blocks.6.mlp`. Other layers have different
activation scales, so the same value may drive some sites into the ceiling. A
**calibration pass** runs first: one short episode per arm at a candidate step size,
reading per-site drift and clip status only. Its output selects the step size for the
main run and is **declared calibration, not evidence** — no claim cites it. The selection
rule is fixed now: take the largest candidate at which **no site clips**, from the ladder
{1x, 0.5x, 0.25x, 0.1x, 0.05x} of the working value.

## Controls and gates (issue #26)

| Gate | Requirement |
|---|---|
| C0, step size zero | Closed loop at eta 0 must be bit-identical to the frozen loop, and every matrix bit-identical to its start |
| C1, revert | `revert()` must restore all twelve matrices bit-exactly |
| Ceiling | Per-site clip flag recorded; **any run with a clipped site is diagnostic only** and is quoted in no conclusion (standing rule 2) |
| Non-finite | Recorded per site and per arm; any non-finite run is void |
| Severed floor | Exactly 0.0 and bit-identical, on the severable eight-site sub-configuration |
| Matched axes | Step size, ceiling, update count, sample order, batching, initial weight and seed identical between closed and offline; recorded alongside every number |

## Measurements

**Primary (behavioural, per issue #26):** basin membership. A stratified set of **30
prompts** drawn from the frozen 125-prompt census covering all five basins in proportion
(`prolet`, `Divine`, `till`, `Anarch`, and the single `solidarity` prompt), each run
frozen under three weight sets: original, closed-loop-drifted, offline-drifted. The
driven prompt `A01_physics` is excluded from this set so every reprompt is genuinely
fresh. Every basin label is reported with its top-1 minus top-2 logit margin.

**The steering-versus-collapse discriminator (issue #26, both halves required):**

1. **Does it return?** Perturb the settled state under the drifted weights at five
   magnitudes and iterate. Criterion fixed now, matching `RESONANCE_NOTE.md`: returned
   means the state comes back to within `1-cos < 1e-9` of the settled state,
   phase-aware, inside 40 iterations.
2. **Did the rest survive?** Across the 30 reprompts, count how many distinct basins
   remain occupied under the drifted weights.

**Supporting:** per-site drift and singular-value structure in float64; effective rank of
each drifted matrix against its original; the settled state's lag-1 and lag-2 cosines,
which say whether it rests or cycles.

## Interpretation, fixed now

- **C-60 (the driven prompt).** If the settled word under the drifted weights differs
  from the frozen settled word, at a margin above 0.05 logits, distributed plasticity
  moved the driven prompt's outcome. If not, it did not. Either way the per-site drift is
  reported so a null is distinguishable from "nothing moved".
- **C-61 (persistence, the point of the whole design).** For each fresh prompt, compare
  its basin under the original weights against its basin under the closed-drifted
  weights. "Something carried across the prompt boundary" requires **at least one fresh
  prompt changing basin at a margin above 0.05 logits in both readings**. If no fresh
  prompt changes, the honest result is that the weight change did not reach anywhere the
  fresh prompts use, and that is a clean negative worth stating.
- **C-62 (steering or collapse).** Steering requires **both**: the perturbed state
  returns, **and** more than one basin remains occupied across the 30 reprompts. If
  every reprompt lands in one place, that is collapse, and it is reported as collapse
  however dramatic it looks. If the state does not return, whatever moved is a bias and
  not an attractor.
- **C-63 (what feedback contributed).** The closed-versus-offline difference over total
  drift, reported per site and in aggregate, carrying the cross-site caveat above and the
  severed-floor limitation. If the two arms produce the same basins everywhere, feedback
  changed no outcome in this configuration and the framing rests on the weight difference
  alone.
- The word **learning** is not used for any outcome. Issue #29 fixes the honest phrasing:
  a measured difference in behaviour on a fresh prompt, nothing stronger.
- A trajectory that neither settles nor diverges is a **result**, not a failed run
  (issue #26's fourth outcome), and is reported as wandering with structure.

## Compute and resumability

Roughly one and a half to two hours on CPU. Records append per unit of work to
`exp002.jsonl`; `--resume` skips completed units, since the environment has killed long
runs in this repository before.
