# Is "resonant LLM" a real architecture, or a regime?

*A note on the idea, written before we have the results that would settle it. Plain
language, honest about which parts are strong.*

## The short version

The strong part of the idea is real and I think it is being undersold. The part that
sounds most exciting — that a resonant model is "a different animal" — is the part that
currently has the least evidence, and there is a specific measured number in the parent
project that argues against it.

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

**What this supports, stated precisely: near period-2 recurrence, with the residual
sitting at about 1.65 times float32 epsilon and no detected trend over 40 iterations.**

**What it does NOT support, and I had this wrong:** that the cycle is *attracting*. I
wrote that, and it does not follow from this measurement. Attraction is a claim about what
happens to *nearby* states — perturb one off the cycle and see whether it comes back. We
measured a single orbit's own residual. That tells us the orbit is stable *in the sense of
persisting*, not that it pulls anything toward it. I also asserted that a bitwise-exact
cycle would be "a knife-edge"; that was invented. Bitwise exactness and attraction are
independent properties and neither implies anything about the other.

The perturbation test is the missing measurement and it is cheap: take the settled state,
add noise at several magnitudes, iterate, and record whether the residual returns to the
1.65-epsilon floor and how fast. Until that is run, the honest word is **recurrent**, not
attracting. It is queued.

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

## The weak claim, and the number that argues against it

The exciting version goes: once the model is resonant, and once we can bias which state
it resonates in, it is a completely different animal — same weights, different behaviour.

Here is the problem. **The Divine state is position-uniform.** Every token position holds
essentially the same vector. And 125 prompts collapse into 5 basins.

Put those together and ask what the settled state actually knows.

Here I have to be careful about what the counting argument does and does not bound. Five
basins means the **basin label** carries at most log2(5) ≈ 2.32 bits about the prompt. It
does **not** bound the settled state itself. If prompts inside a basin settle onto
distinct states, the state can carry a great deal more than its label does.

So the correct statement is conditional, and it splits exactly along the measurement in
the within-basin issue:

- If within-basin states are indistinguishable, the settled state *is* the label, and 2.32
  bits is the whole of it. The attractor has erased the prompt.
- If within-basin states differ systematically, the attractor has compressed rather than
  erased, and the bound does not apply to the state at all.

I don't know which, and neither does anyone else yet. I previously wrote that destroying
information is what an attractor does *by definition*, and that is too strong — basins
merge trajectories that started in the same basin, which constrains how much can survive
but does not force it to zero. The amount actually destroyed is an empirical quantity, and
it is the thing to go and measure.

So "a different animal" needs care. *If* the erasure reading holds, the system is a
different animal in the way that a struck bell is: it is doing something, doing it stably,
and what it does is nearly independent of how you struck it. If the compression reading
holds, that analogy is wrong and the picture is much more interesting.

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

**1. The settled state has to carry more than a basin label.**

Test: within a single basin, do different prompts settle onto *exactly* the same state, or
onto nearby-but-distinct states? If distinct, how much does the spread encode about the
prompt? This is cheap — we already produce the states. If prompts within a basin are
distinguishable, the attractor is a compression rather than an erasure, and everything
downstream gets more interesting. If they are bit-identical, the state is a label and we
should say so.

This is the measurement I would run first, and it does not need plasticity at all.

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

Three sentences.

The reframing from feedforward function to dynamical system is correct, is not a metaphor,
and is enough on its own to justify the project. The evidence so far says these attractors
are information-destroying rather than information-preserving, which makes "different
animal" a claim that needs the within-basin spread measurement before anyone says it out
loud. And the one modest version of the idea — that moving the weights moves the basin
boundaries, measurably, in a way that is not explained by the rule simply drifting — is
both the cheapest thing to test and the thing that would make everything else worth doing.

The first measurement I would run is the within-basin one, and it needs no plasticity at
all.

---

*Related: the persistence issue, the failure-modes issue, the collapse-and-stabilise
experiment, and `PRIOR_ART.md` on what the rule does with no feedback.*
