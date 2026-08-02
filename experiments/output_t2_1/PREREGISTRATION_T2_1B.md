# T2.1b pre-registration — the ceiling-silent gap between 2x and 4x eta

Written and committed **before** the probe runs. Operator-approved follow-up to T2.1
(same PR); the durable claim is **C-59** (claimed on the Identifier registry before use).
This is an addendum: T2.1's registered grid, gates and results are frozen and unchanged.

## Question

T2.1 found the connected and matched-offline arms agreeing at every ceiling-silent
setting, and disagreeing only at 4x eta*, where **both arms saturated the 0.05 drift
ceiling**, making that cell inadmissible under standing rule 2. T2.1b asks the one
question that cell raised but could not answer: **is there a ceiling-silent step size at
which the two arms settle differently?**

## Grid

Three cells on the untested interval, everything else pinned to the T2.1 base point
(`hebb`, `blocks.6.mlp`, `A01_physics`, seed 0, cadence 1, 120 steps, `max_delta_frac`
0.05): eta = {2.5x, 3.0x, 3.5x} of eta* = 7.065171428571429e-05.

Expectation stated in advance: closed-arm drift was 0.0286 at 2x, so 3.5x may reach the
0.05 ceiling. Any cell that clips is reported in the table, marked, and quoted in no
conclusion; the answer is read from the ceiling-silent cells only.

Records go to a **separate** file, `t2_1b_gap.jsonl`, leaving the registered T2.1
artifact untouched.

## Measurements and gates

Identical to T2.1's pre-registration: `diff_over_drift` on the recomputed pair, basin
labels with margins from frozen re-runs under each arm's final matrix, 17/17 axis
verification aborting any mismatched cell, float64 norms, sticky clip flag per arm. The
severed floor and base-point reproduction gates were established in T2.1 on this same
instrument this same day and are not re-run.

## Interpretation, fixed now

- A cell counts as "arms disagree" only if the basin labels differ AND both margins
  exceed 0.05 logits (T2.1's rule, unchanged).
- If any ceiling-silent cell disagrees: **C-59 holds the positive claim** that feedback
  changes the behavioural outcome at that coupling, the first such observation, and
  C-58's "never differs" clause gets that boundary attached.
- If every ceiling-silent cell agrees: **C-59 holds the negative claim** that arm
  agreement extends through the largest ceiling-silent eta tested; the feedback-outcome
  question then has no admissible positive anywhere in the instrument's safe range, and
  the only route left is raising the ceiling, which is a different experiment and a
  different safety argument.
- Where each arm lands (`comrade`, `Divine`, or other) is reported with margins either
  way; basin identity is not part of the disagreement criterion.
