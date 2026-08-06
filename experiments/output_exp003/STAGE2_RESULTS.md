# EXP-003 Stage 2 results

*Measurements only. Registered in `PREREGISTRATION.md` and amended by Amendment 1 there,
both before this run. Run artifact: `stage2.jsonl`.*

> **This file used to say that nothing in it entered the claim register. That has changed.**
> The **measurement** is register row **C-67**: census agreement is 0 of 31 at all three
> cadences. The **cadence comparison** this stage was built to make is not in the register and
> is not established, because the registered drift guard fired at a spread of 6.03× against a
> required factor of 2, so cadence is confounded with drift here. `CLAIMS.md` is the
> authority; where it and this file disagree, it wins.

## What was run

Three settings of how often the weights are adjusted, at a fixed 120-iteration episode,
with the step size multiplied by the cadence so that the total adjustment would be equal.
Driven input `A01_physics`, 12 MLP sites, reinforcing rule, ceiling lifted, per-block step
sizes from EXP-002's committed record.

After each episode the weights were frozen and 31 fresh inputs were run, stratified over
the five committed end states.

**Census agreement** counts how many of those 31 settle where the frozen model puts them,
counting only inputs that settled. **Settled** means a fixed point or a two-step cycle, by
the committed baseline's classifier and thresholds.

## Results

| Cadence | Adjustments | Drift | Census agreement | Settled | Settled words |
|---|---|---|---|---|---|
| reference, frozen | 0 | 0 | **31 / 31** | 31 / 31 | the five committed end states |
| every iteration | 120 | 1.3112% | **0 / 31** | 31 / 31 | `Rousse` 25, `anarchism` 5, `comrade` 1 |
| every 4th | 30 | 2.9909% | **0 / 31** | 31 / 31 | `observer` 31 |
| every 12th | 10 | 7.9112% | **0 / 31** | 31 / 31 | `.` 31 |

Nothing clipped at any setting. The non-finite flag was not recorded in this run, because
the runner gained that field afterwards, so no claim is made about it.

## The registered guard fired

| Registered rule | Measured | Result |
|---|---|---|
| Drift across the ladder must match within a factor of 2 | spread **6.03x** (1.3112% to 7.9112%) | **not matched** |
| If not matched, the comparison is qualitative only and the falsifier is not invoked | — | falsifier not invoked |
| Falsifier, had it been invoked: agreement at cadence 12 must exceed cadence 1 by 5 or more | difference **0** | not evaluated |

The step size was multiplied by the cadence on the assumption that drift scales linearly
with step size. It did not. EXP-002's committed record already states that this rule's
drift is non-linear in step size.

## Reference gate

The 31 fresh inputs reproduce the committed baseline at 31 of 31 before any weight moves.

## Reproduction of EXP-002

The every-iteration cell matches EXP-002's committed drift of 0.013111766434820447 and its
driven input settles on `Rousse`, as that experiment records for the same arm.

Its fresh-input distribution differs: `Rousse` 25, `anarchism` 5, `comrade` 1 here against
`Rousse` 27, `anarchism` 3, `comrade` 1 there. Two inputs differ. The drift matches to
sixteen significant figures. The two runners select their 31 inputs by separate
implementations of the same stratified rule and may not select the same 31; this was not
checked before the run.

## Limits

One driven input, one seed, one rule, twelve sites, all MLP output projections, no
injected signal anywhere in this stage. Ceiling lifted, so nothing here is continuous with
results taken under the former 5% cap.

Three cadence settings spanning a factor of 12. Amendment 1 records that this is weaker
than the factor of 100 originally registered, which was dropped because the slowest
setting would have required 12,000 iterations.

The settled classification is taken on unperturbed trajectories. No state was perturbed
and no return was measured, so these results say the trajectories came to rest and say
nothing about whether the resting states attract.

## Files

- `stage2.jsonl` — every cell, including per-input settled words and both cosines.
- `experiments/exp003_stage2.py` — the runner.
