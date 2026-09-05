# EXP-004 pilot

*One scratch run of the Stage 1 measurement, made 2026-09-05 before `EXP_004_SPEC.md` was
written. Run artifact: `pilot.json`. Script: `experiments/exp004_pilot.py`. The operator
has ruled that this run is to be given little weight. Nothing here enters the claim
register.*

## Limits, stated first

One site, `blocks.6.mlp`. One forward pass over each context, then one `apply()`. Three
contexts, one per class. One random seed for the control, matched on Frobenius norm only,
which register row C-23 records as the wrong quantity to match on. No loop. Ceiling lifted
to 1.0. Run once on CPU in about twenty seconds. The step sizes are half-decades from 1e-4
to 3e-1, chosen without a map.

## What was run

For each context, the reference is the model's next-word distribution at the query
positions when run on the context followed by the query. The baseline is the same on the
query alone. The rule (`hebb`) was installed at the site, the model was run once over the
context alone, the update was applied at each step size, and the query was run alone
again. The control writes a Gaussian matrix of the same Frobenius norm to the site. The
score is the mean KL divergence from the reference over the query positions, in nats.

## What came out

| Context | Baseline KL | Step size | Drift | KL after `hebb` | KL after random | Log-prob of reference top-1, `hebb` | same, random |
|---|---|---|---|---|---|---|---|
| fact | 1.449 | 1e-4 | 0.0001 | 1.449 | 1.449 | -1.94 | -1.94 |
| fact | | 1e-3 | 0.0005 | 1.450 | 1.449 | -1.94 | -1.94 |
| fact | | 1e-2 | 0.0051 | 1.457 | 1.449 | -1.88 | -1.94 |
| fact | | 3e-2 | 0.0153 | 1.477 | 1.446 | -1.74 | -1.95 |
| fact | | 1e-1 | 0.0509 | 1.589 | 1.453 | -1.39 | -1.93 |
| fact | | 3e-1 | 0.1527 | 2.105 | 1.461 | -2.11 | -2.01 |
| format | 4.654 | 1e-4 | 0.0000 | 4.654 | 4.654 | -6.61 | -6.61 |
| format | | 1e-3 | 0.0004 | 4.655 | 4.654 | -6.61 | -6.61 |
| format | | 1e-2 | 0.0040 | 4.659 | 4.655 | -6.61 | -6.61 |
| format | | 3e-2 | 0.0121 | 4.676 | 4.652 | -6.59 | -6.61 |
| format | | 1e-1 | 0.0402 | 4.825 | 4.645 | -6.46 | -6.60 |
| format | | 3e-1 | 0.1207 | 5.879 | 4.636 | -6.30 | -6.55 |
| topic | 1.918 | 1e-4 | 0.0000 | 1.918 | 1.918 | -7.22 | -7.22 |
| topic | | 1e-3 | 0.0004 | 1.918 | 1.918 | -7.22 | -7.22 |
| topic | | 1e-2 | 0.0037 | 1.912 | 1.918 | -7.21 | -7.22 |
| topic | | 3e-2 | 0.0110 | 1.902 | 1.917 | -7.19 | -7.22 |
| topic | | 1e-1 | 0.0366 | 1.871 | 1.920 | -7.20 | -7.23 |
| topic | | 3e-1 | 0.1098 | 1.836 | 1.939 | -7.49 | -7.27 |

The reference's most likely final token was ` V` for the fact context, ` G` for the
format context and ` reactor` for the topic context. The ceiling never fired.

## Reading, marked as a pilot reading

On the fact and format contexts the KL never fell below baseline at any step size and
rose above the random control at every step size where the write was large enough to
register. On the topic context the KL fell from 1.918 to 1.836 at 11 percent drift while
the random control rose to 1.939. The reference's top token gained probability under
`hebb` on the fact context at 5 percent drift and lost it at 15 percent. A second seed or
a second context per class could change any of this.
