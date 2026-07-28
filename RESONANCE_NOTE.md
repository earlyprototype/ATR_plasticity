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
onto an exact two-step cycle: the state at step *n* equals the state at step *n+2*, with
a cosine of 1.000000. Not approximately. Exactly.

So the honest word is **attractor**, not resonance. The system has a small number of
states it falls into and stays in. That five prompts out of a hundred and twenty-five
land somewhere and never leave is an attractor result. Calling it resonance imports
promises about frequency response that we have not measured and probably cannot.

I'd keep the word "resonant" for the intuition and "attractor" for anything written
down.

---

## The strong claim, which I think is right

**A transformer run normally is a feedforward function. Run in this loop, it is a
dynamical system. Those are different mathematical objects, and the second one has
properties the first cannot have.**

A feedforward function has no state. You put a prompt in, you get logits out, nothing
persists, and the concept of "where does it settle" is not defined — there is nowhere for
it to settle *to*. Fixed points, basins, limit cycles, periods: none of these are
properties a feedforward network can have. They are not hidden in there waiting to be
found. They do not exist until you close the loop.

Close the loop and they exist. And critically: **you did not add any weights to make them
exist.** The basins are a property of the weights GPT-2 already had. They were there the
whole time, in the sense that they are fully determined by the weights — and they were
never there, in the sense that nothing about ordinary inference brings them into being.

That is the genuinely good idea, and it does not need any grand theory to be interesting.
It says: *the weights of a trained transformer implicitly specify a dynamical system that
nobody has ever run, and that system has structure.* Five basins. An exact period-2 cycle.
An oscillation carried by a single attention head in block 11.

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

Put those together and ask what the settled state actually knows. If a hundred and
twenty-five different prompts end up in one of five states, then the settled state carries
at most about two and a bit bits of information about the prompt that produced it. It
knows *which basin*. It does not appear to know what you said.

That is what an attractor does — it destroys information. That is the definition. A basin
is precisely a set of starting points that become indistinguishable.

So "a different animal" needs care. A system in a low-information attractor is a different
animal in the way that a struck bell is a different animal from a bell: it is doing
something, it is doing it stably, and what it is doing is nearly independent of how you
struck it.

The second number in the same direction: the readout invisibility ratio is 0.295. A large
share of what happens in the residual stream never reaches the logits at all. So even
where the state does differ, the output may not.

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

**Defensible:** it is a missing link between how we describe transformers (feedforward,
stateless) and the large body of theory about recurrent dynamical systems — attractors,
basins, limit cycles, bifurcations, stability analysis. That theory is mature and has
essentially never been pointed at a pretrained language model's weights, because there was
no dynamical system to point it at. Now there is. That is a real gap and we are standing
in it.

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
