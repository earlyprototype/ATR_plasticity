# Is "resonant LLM" a real architecture, or a regime?

*A note on the idea, written before we had the results that would settle it. Plain
language, honest about which parts are strong.*

> **Notice — the measurement this note was waiting on has run.** I wrote this with the
> within-basin question open and treated it as the thing that would decide "different
> animal". It has since been made, and it is committed in
> `experiments/output_baseline/BASELINE.md`: the answer is **compression, not erasure**
> (C-04, C-05); position uniformity holds for **every** basin, not only `Divine` (C-06); and
> the basin construct itself now carries a resolution limit (C-07). The passages that turned
> on the open question are corrected in place below and marked as superseded rather than
> deleted, because what I believed at the time is part of the record. Every measured number I
> quoted stands as a measurement — but the contraction figures needed a caveat they did not
> have (C-03), and they now carry it.

## The short version

The strong part of the idea is real and I think it is being undersold. The part that
sounds most exciting — that a resonant model is "a different animal" — is the part that
had the least evidence when I wrote this, **[SUPERSEDED — see C-04.]** *and there is a
specific measured number in the parent project that argues against it.* That number was the
basin count, and it does not argue against it. It bounds the basin *label*; the settled state
has since been measured and carries more than its label does.

I'll take them separately.

---

## What "resonant" actually means here

Worth pinning down, because the word does a lot of work and it comes from a different
field.

In its ordinary engineering sense, a resonant system has a preferred frequency: feed it
something near that frequency and it amplifies; feed it something else and it doesn't.
That is not quite what we have.

What we actually have, measured: run the residual stream back into the model repeatedly
and it stops moving. Most prompts settle onto a fixed point. One basin — Divine — settles
onto a two-step cycle: the state at step *n* matches the state at step *n+2*, with a
reported cosine of 1.000000.

That number needed checking, because 1.000000 establishes display precision and not
equality. It has now been measured, on the parent's own committed `state_divine.pt`, over
40 iterations, in float64. **The answer is neither of the two options I expected.**

| | A vs f(f(A)) | A vs f(A), for scale |
|---|---|---|
| `torch.equal` | False, at every probed step | False |
| Elements differing | 6727 / 7680 (88%) | — |
| Relative L2, ‖d‖/‖A‖ | 1.56e-07 | 0.775 |
| Cosine, float64, full precision | 0.99999999999998512 | 0.684911683824360 |
| 1 − cos | 1.49e-14 | 0.315 |

So it is not bit-identical: applying the map twice moves 88% of the entries. The residual
also shows no clear trend over the probe — it averages 2.11e-07 over the first third and
1.96e-07 over the last third, a ratio of 0.93, against a per-probe spread of 1.45e-07 to
3.17e-07. A 7% drift inside that much scatter is not a trend one way or the other.

On its own that supports near period-2 recurrence, and no more. **Attraction is a separate
claim** — it is about what happens to *nearby* states, not about one orbit's own residual —
so I had no basis for the word until it was tested. It has now been tested.

Perturb the settled state by Gaussian noise at five magnitudes, iterate, and see whether it
comes back. Criterion fixed before running: returned means the residual sits within 2× the
floor for four consecutive iterations; same orbit means 1−cos < 1e-9 against the original
settled state, compared phase-aware.

| Perturbation (rel L2) | Returned | Iterations | 1−cos to original orbit | Basin |
|---|---|---|---|---|
| 1e-07 | yes | 2 | 7.7e-14 | Divine |
| 1e-05 | yes | 10 | 7.6e-14 | Divine |
| 1e-03 | yes | 146 | 2.9e-12 | Divine |
| 1e-01 | yes | 395 | 3.2e-12 | Divine |
| 1e+00 | yes | 496 | 2.0e-11 | Divine |

**Five out of five return, all to the same orbit, and the basin label survives every time.**
So the word **attracting** is earned, and the three possibilities are cleanly separated: it
returns, it does not fall into a different attractor, and it does not wander.

Two details worth more than the verdict. Recovery takes about **71 iterations per decade of
displacement**, a per-iteration contraction factor near 0.968 — slow, steady linear
convergence rather than a snap-back.

**That pair of numbers needs a caveat I did not give it, and C-03 carries it: contraction
here is not a constant, and 0.968 / 71-per-decade is an endpoint slope rather than a fit.**
Across the very ladder in the table above the per-decade rate ranges **4 to 124**, and its
low end returns trivially; `EXP_001_RESULTS.md` §6 measures **0.9521, about 47 iterations per
decade**, on a different trajectory. I wrote the single figure as though it were a property
of the orbit. It is a property of one end of one ladder, and it should not be quoted without
the range.

And the largest case is a displacement *as large as the state itself*, and it still comes
home: the basin is not a narrow neighbourhood around the cycle.

A caution that survives all this: the first run of this test used a 200-iteration horizon
and reported the largest perturbation as a failure to return. It was a horizon artifact. At
these contraction rates, **any convergence test in this repo needs a horizon justified
against the contraction factor**, or it will report false negatives at exactly the
displacements that matter most.

One practical consequence does follow from what was measured: **any state-equality test in
this repo has to be a tolerance test at float32 round-off.** `torch.equal` returns False on
two states that are the same point of the dynamics, so an exact-equality assertion here
tests the arithmetic rather than the system.

So the honest word is **attractor**, not resonance. The system has a handful of states it
falls into and stays in: 125 prompts across the parent's library land in 5 basins. That is
an attractor result. Calling it resonance imports promises about frequency response that
we have not measured and probably cannot.

I'd keep the word "resonant" for the intuition and "attractor" for anything written
down.

---

## The strong claim, which I think is right

**A transformer is normally executed as a single pass. Applied iteratively to its own
output, the same weights define a dynamical system — and that system has structure which
single-pass execution never brings into view.**

I had this stated more strongly and it was wrong, so here is the careful version.

The map is always there. GPT-2's weights define a function from a residual-stream state to
a residual-stream state, and any such function can be composed with itself. Its fixed
points, cycles and basins are fully determined by the weights and exist as mathematical
objects whether or not anyone ever iterates it. They are not created by closing the loop.

What closing the loop changes is that those properties become **an actual runtime process,
and therefore measurable**. Under ordinary inference the map is applied once. Nothing
settles, because nothing is iterated; "where does it settle" is a question about a process
that is not being run. Iterate it and the question has an answer you can record.

And critically: **you did not add a single weight to get it.** The basins belong to the
GPT-2 that already existed. The honest statement is not that we brought them into being —
it is that nobody had looked, because looking requires running the model in a way nobody
runs it.

That is the genuinely good idea, and it does not need any grand theory to be interesting.
It says: *the weights of a trained transformer implicitly specify a dynamical system, and
that system has structure.* Five basins. A period-2 cycle. An oscillation carried by a
single attention head in block 11.

So ATR is best understood as **a measurement instrument** — it reveals structure in the
weights that ordinary inference cannot see. That framing survives every criticism I can
think of, including all the ones below. It does not depend on the loop being useful for
anything.

---

## The weak claim, and the number I thought argued against it

The exciting version goes: once the model is resonant, and once we can bias which state
it resonates in, it is a completely different animal — same weights, different behaviour.

Here is the problem. **The settled states are position-uniform.** Every token position holds
essentially the same vector. And 125 prompts collapse into 5 basins.

**[SUPERSEDED — see C-06.]** *I wrote that as a `Divine` property*, following the parent,
which reports it there. It is not one: `BASELINE.md` finds every pair of token positions above
cosine 0.999 on **125 of 125 prompts, in every basin** — worst pair-cosine 1.000000 and
maximum spread 0.000000 in all five. So this corrects the parent's framing rather than
qualifying it, and the problem below is a problem about the whole landscape, not about one
basin.

Put those together and ask what the settled state actually knows.

Here I have to be careful about what the counting argument does and does not bound. Five
basins means the **basin label** carries at most log2(5) ≈ 2.32 bits about the prompt. It
does **not** bound the settled state itself. If prompts inside a basin settle onto
distinct states, the state can carry a great deal more than its label does.

So the correct statement was conditional, and it split exactly along the measurement in
the within-basin issue — which has since been made:

- If within-basin states are indistinguishable, the settled state *is* the label, and 2.32
  bits is the whole of it. The attractor has erased the prompt.
- If within-basin states differ systematically, the attractor has compressed rather than
  erased, and the bound does not apply to the state at all.

**[SUPERSEDED — it has been measured. See C-04, C-05.]** *I don't know which, and neither
does anyone else yet.* It is the **second** branch. Prompts sharing a basin settle onto
nearby-but-distinct states: mean `1−cos` of **3.32e-03** over 2337 pairs, which is eleven
orders of magnitude above the float32 round-off floor measured on the parent's own cycle —
of order 1e-14, and C-02 carries the probe range — so this is a real difference and not
arithmetic. The attractor **compresses rather than erases**, and the 2.32-bit bound does not
apply to the settled state at all.

What it compresses onto is thin, and I would rather say so than oversell it: the within-basin
variation is essentially **one-dimensional** — participation ratio **1.02 to 1.29** against a
maximum of n−1 = **15 to 54**, with 87% to 99% of the variance in the top direction. A single
direction of spread is not much room for a prompt to live in. But it is not zero, and zero is
what the erasure reading needed.

I previously wrote that destroying information is what an attractor does *by definition*, and
that was too strong — basins merge trajectories that started in the same basin, which
constrains how much can survive but does not force it to zero. The amount actually destroyed
was an empirical quantity, and it has now been measured rather than argued.

The same measurement handed back something I did not ask for, and it cuts the other way.
**The basin taxonomy has a resolution limit comparable to the basin separation itself**
(C-07): the within-basin spread of 3.319e-03 is *larger* than the gap between the two nearest
basins, `Anarch` and `prolet`, at 2.874e-03 — a ratio of **1.15**. Two basins this project
treats as distinct are no further apart than the prompts inside one of them. That caveat has
to travel with every basin number in this note, including the five-basin count the argument
above is built on.

So "a different animal" needs care, though not the care I expected. The erasure reading is
out, so the struck-bell analogy — doing something, doing it stably, and doing it nearly
independently of how you struck it — is the wrong picture. The compression reading is the one
that holds, and it is the more interesting of the two. What the claim now rests on is how much
that one-dimensional spread actually carries about the prompt, and whether any of it reaches
the output. Neither has been measured.

There is a second number often quoted alongside this — the readout invisibility ratio of
0.295, taken to mean that a large share of what happens in the residual stream never
reaches the logits. **I am not going to use it as evidence here, because we cannot define
it.** Its numerator, denominator, and whether it aggregates over layers or token positions
are all unknown to this repo; it was inherited from the parent and never checked. A number
whose measurement you cannot state is not evidence, and leaning on it while admitting that
would be exactly the kind of move this file is supposed to catch.

It stands as an **unvalidated lead**: worth pulling across and defining, and worth
re-measuring ourselves, at which point it may well support the cautious reading. Until
then the argument above rests on the basin counting alone.

**I don't think this kills the idea. I think it tells us exactly what the idea has to
show to survive.**

---

## What would have to be true

For the strong version of "different animal" to hold, three things all have to be true,
and they are separable and independently testable.

**1. The settled state has to carry more than a basin label. — Measured. It does (C-04,
C-05).**

Test: within a single basin, do different prompts settle onto *exactly* the same state, or
onto nearby-but-distinct states? If distinct, how much does the spread encode about the
prompt? **[SUPERSEDED — see C-47, `retired`.]** *This is cheap — we already produce the
states.* We do not keep them, and the difference matters. `BASELINE.md` promises the 125
settled tensors in `experiments/output_baseline/states/`, but `.gitignore` excludes
`experiments/**/states/` and the directory is not in the repo. The **summary statistics
survive**, which is why the first half of this test is answered; the raw states do not, so
anything sharper than what `BASELINE.md` already reports costs the ~6 CPU-hour baseline
re-run. Not expensive, but not free, and not what I said.

Answered, on the first half: prompts within a basin **are** distinguishable, so the attractor
is a compression rather than an erasure, and everything downstream does get more interesting.
They are not bit-identical, so the state is not merely a label. **Still open, and it is the
half that carries the weight: *how much* the spread encodes about the prompt.** One
dimension of variation is measured; what varies along it is not.

**2. The state has to reach the output.**

The 0.295 invisibility figure means we should not assume it does. Measure the settled
state's effect on next-token distributions directly, and separately from measuring the
state itself.

**3. Something has to persist across prompts.**

The residual stream is destroyed when a new prompt arrives. So whatever the loop achieved
is gone unless it was written somewhere. The weights are the only place it can be written.
That is the whole reason plasticity is in this project, and it is covered in the
persistence issue.

---

## "Biasing which state it falls into" — this part is concrete and I like it

Set the grand framing aside; there is a specific, modest, testable version of the idea
here.

If the basins are a property of the weights, then changing the weights changes the basins.
Not metaphorically — this is close to mechanical. Move the weights, and the boundaries
between basins move, and prompts near a boundary switch sides.

That gives a clean experiment with a binary outcome: take a prompt that reliably lands in
basin A, apply plasticity, and see whether it lands in basin B. Basin membership is
already measured and discrete, so there is no metric to argue about. Either it switched or
it didn't.

And it comes with an obvious trap, which is why the offline control exists: Oja's rule
moves the weights whether or not there is any feedback. So "we applied plasticity and the
basin changed" proves nothing on its own. It proves something only against the offline arm
— same rule, same activations, same number of updates, replayed with no feedback. The
claim is the *difference* between those two runs and nothing else.

---

## On "missing link"

Missing link to what, specifically? I want to separate two versions, because one is
defensible and one isn't yet.

**Defensible:** it is a missing link between how we normally describe transformers
(single-pass, stateless) and the large body of theory about recurrent dynamical systems —
attractors, basins, limit cycles, bifurcations, stability analysis. That theory is mature,
and our prior-art search did not find it applied to a pretrained language model's weights
in this way.

Note the scope of that last sentence, deliberately. **It is a statement about our search,
not about the literature** — the same standard `PRIOR_ART.md` holds itself to, and it
applies here too. I originally wrote "essentially never been pointed at", which claims far
more than we can support, and added a causal explanation for the absence on top of it.
Neither is earned. The searched-and-not-found version is enough to justify the work and is
the only version that should reach a write-up. The coverage gaps in `PRIOR_ART.md` — the
forums in particular — are the reason to keep it hedged.

**Not yet defensible:** that it is a missing link to anything about cognition, or to why
these models do what they do in normal use. Ordinary inference is one forward pass. Nothing
in normal operation runs this loop. So whatever we find here is a property of the weights,
not an explanation of the model's ordinary behaviour. It could become relevant to that —
if, for instance, the basins turn out to correspond to something we can identify in the
model's normal outputs — but that would be a further finding, not this one.

I'd hold the line there hard, because the second version is much more fun to say and would
be the first thing a reviewer went after.

---

## Where I actually land

Three sentences — and the corrections one of them has since earned.

The reframing from feedforward function to dynamical system is correct, is not a metaphor,
and is enough on its own to justify the project. **[SUPERSEDED — the measurement has been
made, and it says the opposite. See C-04, C-05.]** *The evidence so far says these attractors
are information-destroying rather than information-preserving, which makes "different animal"
a claim that needs the within-basin spread measurement before anyone says it out loud.* The
attractors **compress rather than erase**: prompts in one basin settle onto nearby-but-distinct
states, mean `1−cos` 3.32e-03 over 2337 pairs, and the variation is essentially
one-dimensional, participation ratio 1.02 to 1.29 against a maximum of 15 to 54. The ~2.32-bit
counting bound applies to the basin **label**, and never to the settled state — I let those two
run together, and that is the error underneath the sentence above. And the one modest version
of the idea — that moving the weights moves the basin boundaries, measurably, in a way that is
not explained by the rule simply drifting — is both the cheapest thing to test and the thing
that would make everything else worth doing.

**[SUPERSEDED — it ran. See C-04 to C-07.]** *The first measurement I would run is the
within-basin one, and it needs no plasticity at all.* It ran, and it is `BASELINE.md`. Two
things go in its place. What "different animal" needs now is one step further out: how much of
the prompt that one-dimensional spread actually carries, and whether any of it reaches the
output. And the largest credibility gain available to the project is the **125-prompt library
at the working point** — T2.3, register row C-54, issue #49 — which is what would move every
basin-flip claim here off the 3 prompts it rests on. Read either against C-07 first:
within-basin spread 3.319e-03 against a nearest-basin gap of 2.874e-03, ratio 1.15, which is
the load-bearing caveat on anything in this note that uses the word basin.

---

*Related: the persistence issue, the failure-modes issue, the collapse-and-stabilise
experiment, and `PRIOR_ART.md` on what the rule does with no feedback.*
