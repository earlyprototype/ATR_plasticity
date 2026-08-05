# EXP-002 results — distributed plasticity collapses the landscape; feedback picks where

Issue #24 (the sequence), #26 (the controls), #29 (the persistence question). Durable
claims **C-60, C-61, C-62, C-63**, claimed on the registry before use. Interpretation
fixed in `PREREGISTRATION.md` and its two amendments, all committed before the run.
Every number below is recomputed from `exp002_uncapped.jsonl`.

**Regime note, and it must travel with every number here.** This run was made with the
drift ceiling **lifted** (Amendment 2, operator decision): `max_delta_frac` = 1e9, so no
update was ever scaled down. Every prior result in this repository was taken under a 5%
ceiling. Nothing here may be quoted as continuous with those without saying so.

## 1. What was run

All twelve MLP sites plastic at once (`blocks.0.mlp` … `blocks.11.mlp`), driven on
`A01_physics` for 120 updates, weights then frozen, then 31 fresh prompts run under the
resulting weights. Two rule arms, `hebb` and `anti_hebb`, each with a matched
no-feedback arm built from the tested single-site recording and replay.

Per-site step sizes were anchored so every layer travels a comparable relative distance
(Amendment 1). At one shared step size the layers differ by **136x** for `hebb` and **56x** for `anti_hebb` (`calibration.probe_per_site[].delta_frac` at the shared probe eta; an earlier "more than 200x" here pooled the two rules, which is not a span across layers), so a shared
value cannot drive them all. Anchoring worked well for `hebb` (achieved drift 1.03%–1.56%
against a 1.12% target) and poorly for `anti_hebb` (0.92%–11.18%, a 12x spread), whose
drift is strongly non-linear in step size. **Every `anti_hebb` number carries that
imbalance**, and the two arms are not matched on total drift: 1.31% against 5.38%.

## 2. Gates

| Gate | Result |
|---|---|
| Step size zero | Bit-identical to the frozen loop, max abs diff **0.0** |
| Revert | All twelve matrices restored **bit-exactly** |
| Non-finite | None, any arm, any site |
| Collapse step reproduces | Driven prompt settles `prolet`, matching the committed baseline |
| Reference reprompts reproduce | **31/31** fresh prompts settle exactly where the baseline census says |
| Severed floor | **FAILED to be zero — see §3, this is a finding** |

The reference check matters: before any drift, the reprompt machinery reproduces the
committed 125-prompt census exactly on all 31 prompts, so every change below is
attributable to the drift and not to the instrument.

## 3. The severed floor is not zero beyond one plastic layer

Run on the severable eight-layer configuration (`blocks.4.mlp` … `blocks.11.mlp`, loop
read at `blocks.3.hook_resid_post`), where standing rule 3 says the closed and offline
arms must come out bit-identical:

| Site | closed−offline difference | difference / drift | bit-identical |
|---|---|---|---|
| `blocks.4.mlp` | **0.000000** | 0.000000 | **True** |
| `blocks.5.mlp` | 0.434388 | 0.033 | False |
| `blocks.6.mlp` | 0.777322 | 0.143 | False |
| `blocks.7.mlp` | 0.326417 | 0.152 | False |
| `blocks.8.mlp` | 0.176118 | 0.170 | False |
| `blocks.9.mlp` | 0.075992 | 0.186 | False |
| `blocks.10.mlp` | 0.053070 | 0.159 | False |
| `blocks.11.mlp` | 0.213386 | 0.245 | False |

The lowest plastic layer gives exactly zero, bit-identical; every layer above it is
non-zero and the ratio climbs with height. That is a signature, not noise. The severed
control cuts the path from a plastic layer into the **next loop iterate**. For the lowest
plastic layer that is the only path, so the control delivers its promised zero. Every
layer above has a second path the control cannot cut: **within a single forward pass, a
lower plastic layer's drift changes the activations arriving at a higher one.** That path
does not involve the loop and survives severing it.

**Consequence.** Standing rule 3's exact-zero severed floor is a **single-site
guarantee**. It does not extend to plasticity at more than one layer. The
pre-registration anticipated that loop feedback and cross-site interaction could not be
separated here; it did not anticipate that the floor itself stops being zero, and that
is recorded as a correction rather than folded away.

**What this costs.** Every closed-versus-offline number in this experiment measures loop
feedback **and/or** within-pass interaction between layers, with no zero baseline. The
share figures below are reported for completeness and are **not comparable** to the
single-site shares in C-31 or C-58.

## 4. The driven prompt (C-60)

| Arm | untouched | with feedback | without feedback |
|---|---|---|---|
| `hebb` (drift 1.31%) | `prolet` (margin 0.230) | **`Rousse`** (0.582) | **`comrade`** (0.307) |
| `anti_hebb` (drift 5.38%) | `prolet` (0.230) | **`arcane`** (lag-1 0.999033, not at rest) | **`Shiv`** (lag-1 0.999565) |

Distributed plasticity moves the driven prompt's settled word in both arms, at margins
far above the pre-registered 0.05 threshold. In both arms the with-feedback and
without-feedback runs settle on **different words**, so feedback (or within-pass layer
interaction, per §3) decides the destination.

## 5. Persistence, and the discriminator (C-61, C-62)

The point of the whole design: the residual stream is destroyed when a new prompt
arrives, so the weights are the only channel that can carry anything across.

| Condition | changed | distinct words | at rest | distribution |
|---|---|---|---|---|
| untouched (reference) | — | **5** | 31/31 | prolet 13, Divine 8, till 5, Anarch 4, solidarity 1 |
| `hebb`, with feedback | **31 / 31** | 3 | 31/31 | **Rousse 27**, anarchism 3, comrade 1 |
| `hebb`, without feedback | **30 / 31** | 2 | 31/31 | **comrade 30**, prolet 1 |
| `anti_hebb`, with feedback | **31 / 31** | 19 | **4/31** | no word above 3; 9 words seen once |
| `anti_hebb`, without feedback | **31 / 31** | **1** | 30/31 | **Shiv 31** |

"At rest" counts prompts whose trajectory has settled, meaning a fixed point
(step-to-step agreement above 0.9999) **or a two-step cycle** (agreement above 0.9999
between iterates two apart). It is the column that makes the `anti_hebb` rows readable.

**A correction to this sentence, made after the merge and changing no number.** It
originally described the column as counting fixed points by step-to-step agreement
alone. That does not match the reference row, because `Divine` is a two-step cycle:
recomputed from `exp002_uncapped.jsonl`, the eight `Divine` inputs in the census sit
at step-to-step agreement 0.658 to 0.702 with two-step agreement above 0.99999, so a
fixed-point-only reading gives 23 of 31 rather than the 31 of 31 reported. The
reported figure is correct and the phase-aware definition above is the one that
produces it.

**The `anti_hebb` reading is unaffected**, which is the part that matters: its closed
arm counts 4 of 31 under both definitions, so the twenty-seven trajectories still in
motion are genuinely not two-step cycles being missed, and the claim that nineteen
distinct words is non-convergence rather than surviving structure stands.

**Something carries across the prompt boundary, decisively.** Every fresh prompt changed
under both closed arms and under `anti_hebb` without feedback; `hebb` without feedback
changed 30 of 31, the exception being `A16_wittgenstein`, which stayed `prolet`. The
weights are the only channel that survives a new prompt, and the drift reached far enough
to change essentially every one of them.

**In the `hebb` arm it is collapse, not steering.** The untouched model spreads these
prompts over five words; afterwards 27 of 31 land on one, and that one is the word the
driving episode settled on. Collapse does not require feedback: the no-feedback arm
collapses just as hard, to a different word (`comrade`, 30 of 31). What feedback changes
is the destination.

**The `anti_hebb` arm is destructive in a different way, and the two conditions are
opposites.** Without feedback it is the most complete collapse in the experiment: every
one of the 31 prompts lands on the single word `Shiv`, and 30 of them settle. With
feedback it does not collapse — 19 distinct words, none appearing more than three times —
**but almost nothing settles**: only 4 of 31 trajectories are fixed points. So that
diversity is not preserved basin structure. It is 31 trajectories still in motion, read
at iteration 120 wherever they happened to be. Nineteen words is a symptom of
non-convergence, not of surviving landscape, and it must not be read as feedback
protecting the model's structure.

Stated plainly: at these drifts every arm destroys the basin census. `hebb` destroys it
by pulling everything into one new attractor. `anti_hebb` without feedback destroys it by
pulling everything into one word. `anti_hebb` with feedback destroys it by stopping the
system settling at all.

**A correction to the pre-registered criterion.** The registered rule said steering
requires "more than one basin remains occupied". Two and three words remain occupied, so
a literal reading scores this a pass. **That reading is wrong and is not taken.** 27 of
31 prompts on a single word is collapse in any honest sense, and issue #26 says so
directly: "Collapse means everything now lands in the same place." The criterion was
badly specified — it should have compared the distribution against the baseline rather
than counting how many words survive at all — and the fault is the criterion's, not the
result's.

### Does it return?

Phase-aware over the last two iterates, five independent perturbation directions, as the
pre-registration requires. (The first implementation was single-phase and reused one
direction; both were corrected and the test re-run. Neither verdict changed.)

| Arm | returned | detail |
|---|---|---|
| `hebb` | **5 / 5** | returns from every magnitude, in 1, 1, 10, 19 and 30 iterations |
| `anti_hebb` | **0 / 5** | approaches to ~1e-4 and no closer, at every magnitude |

The two rules leave qualitatively different objects. `hebb` leaves a **genuine
attractor**: exact fixed points (lag-1 and lag-2 both 1.000000) that pull perturbed
states back, with return time growing sensibly with perturbation size. It is a real
attractor that happens to swallow nearly everything.

`anti_hebb` leaves something that is **not a resting state of either kind**. The
phase-aware test rules out a two-step cycle being missed: its best match was the *last*
iterate at all five magnitudes, never the previous one, so there is no opposite phase it
was returning to. What it leaves: lag-1 0.999033 and
lag-2 0.996163 (the two-step agreement being worse than the one-step rules out a
period-2 cycle), and the return test floors at ~1e-4 *independent of how far the state
was perturbed*, from 1e-7 to 1e-1. There was no resting point to return to; the
trajectory is still moving at that scale. This is issue #26's **fourth outcome** —
approaching, never settling, never diverging — which that issue states in advance is a
result and not a failed run.

So `anti_hebb` scoring 0/5 is not "no attraction": from a perturbation of 0.1 it comes
back only to within 0.00068, about 2.2 orders closer rather than the three once stated here. It is "no fixed point at the precision the
criterion demands".

The reprompt table confirms this is systemic rather than a property of the driven prompt
alone: under `anti_hebb` with feedback only 4 of 31 fresh prompts reach a fixed point,
against 31 of 31 under both `hebb` conditions and 30 of 31 under `anti_hebb` without
feedback. Non-settling is **associated with the closed condition** under the eroding rule. It is
not attributed to loop feedback alone: §3 shows the closed-versus-offline contrast here
also carries within-pass interaction between layers, and this design cannot separate the
two.

## 6. What feedback contributed (C-63)

Closed against matched offline, stacked over the twelve layers, float64, **with no zero
floor available (§3)**: `hebb` 0.4373, `anti_hebb` 0.2576. Per layer for `hebb` the
ratio runs 0.27–0.55, peaking in the middle layers (4–8) rather than at either end.

These are reported as measured quantities with their limitation attached. They are not
feedback-attributable shares in the sense C-31 and C-58 use, and must not be placed in a
series with them.

## 7. Caveats

- Ceiling lifted; every prior result was capped. Regime difference, stated everywhere.
- No zero floor for any closed-versus-offline number here (§3).
- The two arms are not matched on drift (1.31% against 5.38%) and `anti_hebb`'s per-layer
  anchoring spans 12x, so arm-to-arm comparison is qualitative only.
- One driven prompt, one seed, one site family (MLP down-projections), one episode length.
- 31 reprompts, stratified over the five baseline basins, not the full 125-prompt census.
- `Rousse`, `arcane`, `Shiv` and `anarchism` are readout labels; no claim is made about
  what they mean beyond being the settled top-1 token.

## 8. Files

- `PREREGISTRATION.md` — design, gates, interpretation, plus Amendment 1 (per-layer
  anchored step sizes) and Amendment 2 (ceiling lifted), each committed before the run
  it governs.
- `exp002_distributed.py` (in `experiments/`) — runner.
- `exp002_uncapped.jsonl` — every unit of this run.
- `exp002.jsonl` — the earlier capped attempt, kept whole and not mixed in.
- `meta.json` — configuration and environment.
