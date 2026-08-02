# EXP-003 pre-registration: does an external signal prevent the collapse?

*Written and committed before anything runs. The reasoning behind this experiment, and
the literature it copies, are in `MEA_ANALOGUE.md`. This file is the protocol: what will
be run, in what order, what each stage would have to produce to count as a success, and
what would stop the series. Identifiers are claimed on the peer board before use.*

## The answer this experiment is trying to reach

Experiment EXP-002 found that making twelve parts of the model adjustable at once
destroys the system's ability to tell its inputs apart. Thirty one test inputs that
previously settled into five distinguishable end states afterwards settled into one, or
two, or three. The register records this as row C-62 under the summary "collapse, not
steering".

This experiment asks whether feeding a small signal into the loop from outside prevents
that, and if so, what shape the signal has to have. The reason to expect it might is that
a laboratory at Georgia Tech spent years on the same failure in living tissue and found a
remedy with a specific and copyable shape: a weak signal, spread across many places at
once, delivered while the damage would otherwise be happening, and effective only for as
long as it is applied.

Two corrections to that summary are recorded in `MEA_ANALOGUE.md` and are carried here
because they change what this experiment measures. Spreading the signal out is not
uniformly better than concentrating it: the biological measurement found concentrated
slightly better below a threshold rate and spread out much better above it, so the shape
to look for is a crossing rather than a ranking. And the signal left nothing behind, with
the pathology returning the moment stimulation stopped, which is why every setting below
is measured twice, once with the signal still applied and once without.

## The regime, which must travel with every number this produces

**There is no drift ceiling.** By operator decision the cap on how far weights may move
is removed for this entire series, as it was for EXP-002. Every result in this repository
before EXP-002 was taken with that cap set at five percent, so nothing produced here may
be described as continuous with those earlier results without saying so in the same
sentence.

The reason for removing it is not convenience. A cap on how far a weight may move is
itself a stabilising mechanism imposed from outside, and the absence of such a mechanism
is precisely the condition under study. Leaving the cap in place would have quietly
supplied part of the answer the experiment is asking for.

## Definitions used throughout

**The loop.** The frozen model is run so that its internal state is read out of the last
layer, rescaled to a fixed size, and written back into the first layer, repeatedly. One
pass through the model is one iteration. Written as a formula, with `f` standing for the
model and `N` for the rescaling, the loop is `r` becomes `f(N(r))`.

**The signal, and how it enters.** The proposed change adds a fixed vector `e`, scaled by
a number `beta`, at the point where the state is written back in. The loop becomes `r`
becomes `f(N(r) + beta e)`, with `e` scaled to size one, so that `beta` is the strength of
the injected signal measured against the size of the system's own activity. A `beta` of
0.01 means the injected signal is one percent as large as the state it is added to. This
is a change of a single line at `atr_bridge.py:194`, where the tensor to be injected is
prepared, and it does not reimplement the loop, which the project's standing rules forbid.

**Distributed versus focal.** Focal means the whole signal enters at one place, the point
where the loop closes. Distributed means the same total signal is divided among several
depths of the model, entering at several points along one forward pass. The biological
protocol being copied used ten to twenty electrodes out of sixty, so the corresponding
fraction here is roughly one sixth to one third of the twelve available blocks, which is
two to four injection points.

**At rest.** A trajectory counts as at rest if consecutive iterations agree above 0.9999
on a scale where one means identical. This is the definition EXP-002 used and it is
carried unchanged.

## The primary measurement, and why it is not diversity

**The primary measurement is census agreement:** of the thirty one test inputs, how many
settle into the same end state the frozen model puts them in, counting only those that are
at rest.

The frozen baseline is thirty one out of thirty one, and this is already an established
gate rather than a new quantity: EXP-002 verified that before any weights moved, all
thirty one inputs reproduced the committed reference exactly. In the collapsed runs the
figure is at most thirteen and in practice near zero, because the end states the collapsed
runs produce are not among the five the frozen model produces at all.

**Counting how many different end states occur is explicitly rejected as the primary
measurement**, and the reason is a result this repository already has. One arm of EXP-002
produced nineteen different end states, which looks like diversity preserved, but only
four of the thirty one trajectories had come to rest. The other twenty seven were still
moving. Nineteen different answers were nineteen snapshots of motion.

The same trap waits for this experiment in a worse form. If the injected signal is made
strong enough, every input will produce a different end state, because each will simply
return its own injected signal. That is the signal being reflected, not the model
expressing anything, and a count of distinct outcomes cannot tell the two apart. Census
agreement can, because a reflected signal has no reason to reproduce the frozen model's
particular assignment of which input goes where.

**The corroborating measurement is label free**, and is defined in Stage 0 below. It is
needed because of register row C-07, which records that the labels naming the end states
can barely tell them apart: states sharing a label are spread further than the gap between
the two nearest distinct labels, by a ratio of about 1.16 where 1.0 would mean the labels
carry no information at all. Any result resting only on labels is exposed to that.

**A note on C-07, because the same laboratory formalised the problem it describes and
this project should adopt their framing rather than inventing a second one.** Faced with
cultures whose connectivity drifted continuously on its own, Chao, Bakkum and Potter
defined a ratio: the distance from one measurement period to the centre of another,
divided by the scatter of that other period about its own centre. They state that a value
near 1 means no detectable change, because the difference between the two periods is
indistinguishable from the ordinary scatter inside one of them.

Row C-07 is that quantity, arrived at independently. The gap between the two nearest end
state labels is 2.874e-03 and the spread within one label is 3.319e-03, a ratio of 0.87,
or 1.16 the other way up. Read through their framework, that is a system whose labelled
categories sit right at the edge of being distinguishable from their own internal scatter.

Two consequences follow and both are adopted here. First, every result in this experiment
reports this ratio alongside the count, so that a change in the count is accompanied by
whether the categories were separable at all. Second, the same ratio is the measure of
whether any effect of the injected signal exceeds the drift the system shows without it,
which is the control Potter describes as having been necessary in the wetware case: "It
was necessary to develop analytical tools to deal with the large ongoing drift in
functional connectivity of cultured networks, to reveal changes induced by external
stimuli."

## The stages, in the order they will run

The order is chosen so that the cheapest result that could stop the series comes first.
Each stage has a gate. If a gate fails, the series stops there and the failure is reported.

### Stage 0: build a measurement that does not use the labels, and prove it works

**What it is.** A quantity adapted from the centre of activity measurement used by Chao,
Bakkum and Potter, which reduced the activity of a whole electrode array to a single
moving point by taking the average position of the electrodes, weighted by how strongly
each was firing. Their array had real physical geometry, which is what made an average
position meaningful.

A model's internal state has no comparable geometry across its width, and taking an
average across it would be meaningless. But depth is real geometry: the blocks are
ordered, and activity genuinely passes along them. So the adapted quantity is the average
block index, weighted by how much each block writes into the state:

    A(t) = sum over L of L times w_L(t), divided by sum over L of w_L(t)

where `w_L(t)` is the size of what block `L` adds to the state at iteration `t`, measured
as the length of the difference between the state after that block and the state before
it. The result is a single number between 0 and 11 saying where in the depth of the model
the work is being done, and following it across iterations gives a trajectory.

This is basis free, meaning it does not depend on any arbitrary choice of coordinates, it
is unaffected by the rescaling step, and it never consults a label.

**The gate, which this stage must pass before any of it is used.** Measured on the frozen
model, where the answer is already known, the quantity must separate the five known end
states, and it must distinguish the one end state known to be a two step cycle from the
four known to be resting points. It must also be shown to fail where it should fail: on
inputs the register records as sharing an end state, it must not separate them.

**The control that decides whether depth is doing the work, taken directly from the
source.** Chao, Bakkum and Potter faced the identical objection, that their statistic
might work merely because it compressed sixty channels into two well behaved numbers
rather than because electrode position meant anything. Their answer was to shuffle the
electrode positions at random and recompute. Their reported result is that the smallest
change the statistic could detect roughly doubled under shuffling, from 4.68 percent to
10.8 percent, that sensitivity fell from 88.7 percent to 35.4 percent, and that in living
cultures the induced change exceeded ordinary drift for the true statistic at a
significance below one in ten thousand but not for the shuffled version, which reached
only 0.19 and is therefore indistinguishable from drift.

The port is exact and cheap. The same quantity is recomputed with the block indices
randomly permuted, ten permutations. If the shuffled version separates the five known end
states as well as the true one does, then the ordering of the blocks is not carrying the
information, the statistic is only a summary of activity size, and it is discarded no
matter how well it performed on the gate above.

**If the gate or the shuffle control fails**, the quantity is discarded and the series
continues using labels alone, with row C-07's limitation attached to every subsequent
result. This is recorded as a real possible outcome, not a formality.

**A second quantity, free, computed on weights rather than activity.** The same source
defines a companion measure over connection strengths rather than over activity, on the
grounds that one describes how signals propagate and the other describes how the
connections themselves are moving. Its port here is the depth weighted centre of the
weight change: each block's index weighted by how far that block's matrix has moved,
divided by the total movement. It says where in the depth of the model the adjustment is
concentrating.

This needs no new model time at all, because EXP-002 already records how far each of its
twelve matrices moved. It is computed and reported in Stage 1.

**Cost.** No new model time beyond one pass over the existing baseline states.

### Stage 1: check the proposed explanation against runs that already exist

**What it is.** The explanation offered in `MEA_ANALOGUE.md` for why collapse happens is
that each adjustable site becomes concentrated on a single direction, so that it emits
roughly the same output whatever arrives. This repository already measures how
concentrated a matrix has become, and EXP-002's runs are already committed.

**The prediction.** In the arms that collapsed, the concentration measure should have
moved sharply from its frozen value. In the arm that did not collapse but failed to
settle, it should behave differently.

**The gate.** If concentration has not moved appreciably in the collapsed runs, the
explanation is wrong. That does not stop the series, because the collapse is a measured
fact whatever explains it, but it removes the mathematical route to the distributed
versus focal prediction and that prediction then rests on the biology alone. This must be
recorded if it happens, because it changes how much weight the analogy is carrying.

**Also computed here, at no extra cost.** The depth weighted centre of the weight change,
defined in Stage 0, is reported for each arm of EXP-002 from that experiment's committed
per layer movement figures. In the source this measure was used to confirm that stimulating
at different places moved the connection strengths in different directions, which is what
established that the conditions being compared were genuinely distinct rather than
variations on one thing. The corresponding question here is whether the two rules, and the
connected and disconnected conditions, concentrate their adjustment at different depths.
No prediction is registered for this, because none is held. It is reported because it is
free and because a later stage will want the baseline.

**How sensitivity is quoted, following the source rather than inventing a convention.**
Chao, Bakkum and Potter judged competing statistics by the smallest true change each could
detect, having manufactured a situation where the true change was known. That situation
exists here without any simulation, because the weight change is known exactly. Any
statistic proposed in this experiment is therefore quoted with the smallest weight change
at which it separates the drifted system from the frozen one, and statistics are compared
on that number rather than on whether each individually reached significance.

**Cost.** No new model time. This is a computation on committed artifacts.

### Stage 2: is the collapse caused by adjusting too fast?

**What it is.** In living tissue, connections change far more slowly than activity passes
through them. Here they change at comparable speed: EXP-002 adjusted the weights on every
iteration. This stage slows the adjustment down, holding the total amount of adjustment
fixed, by applying larger changes less often. The project already supports this as a
setting and has previously used values of one, two and four.

**The prediction.** Collapse should recede as the adjustment is slowed. Census agreement
should rise.

**The falsifier, stated numerically.** If census agreement stays below five out of thirty
one across a hundredfold change in how often adjustments are applied, the timescale
reading is dead and will be recorded as dead.

**Why this is second.** It is the cheapest experiment that could remove a whole branch of
the argument, and it needs no new code beyond what exists.

**Cost.** Roughly three minutes per adjustment episode and roughly ninety minutes per
census of thirty one inputs, on the processor available. Four settings, one rule, gives
about six hours.

### Stage 3: the signal, focal against distributed

**What it is.** The main experiment. Under the reinforcing rule at all twelve blocks,
with the loop closed, the signal is injected during the adjustment. Two shapes are
compared at matched total strength: all of it at the point where the loop closes, against
the same total divided among three points spread through the depth of the model.

The strength is swept, because the biological protocol was deliberately weak and the
correct strength here is unknown. The sweep runs in half order of magnitude steps from a
strength of 0.003, meaning three tenths of one percent of the system's own activity, up
to 0.3, meaning thirty percent, or until the reflection control below fires.

**The census is taken twice at every setting, once with the signal still being applied and
once with it removed.** This is not a robustness check, it is the experiment. The
biological measurement being copied reports that "bursting resumed as soon as stimulation
was stopped", and a companion paper by the same authors says of that work that "we saw no
plastic effects from burst quieting". So the prediction is that recovery appears in the
census taken with the signal on and disappears in the census taken with it off. An
experiment that measured only the second would be predicted by the very source it is
copying to find nothing, and reporting that as a refutation would be an error. This
paragraph exists so that the error is not available.

**The predictions.** With the signal still applied, distributed injection should recover
census agreement. Focal injection at matched total strength should recover substantially
less at high strength, because a signal entering before the first concentrated site is
converted to that site's fixed direction immediately.

**The prediction that would be hardest to produce by accident, and is therefore the one
worth the most.** The biological measurement does not say spread out is simply better. It
records a crossing point: below roughly ten stimuli per second, spreading the signal
across many electrodes gave slightly worse burst reduction than concentrating it in one,
and above that rate spreading it gave much better reduction. The corresponding prediction
here is that focal injection wins at low strength and distributed injection wins at high
strength, with a crossing in between. Nothing in the mathematical account predicts a
crossing, so finding one would be an agreement that no part of this design was built to
produce.

**The falsifiers, stated numerically.**

The distributed advantage at high strength is claimed only if distributed injection
reaches a census agreement at least ten out of thirty one higher than the best focal
result at the same total strength. Ten is roughly a third of the census and is well above
the one point difference that separated the two collapsed arms of EXP-002. If focal
matches distributed within that margin at every strength tested, the central prediction
of `MEA_ANALOGUE.md` has failed and that document is withdrawn.

The crossing is claimed only if focal exceeds distributed by at least five out of thirty
one at the lowest strength that produces any recovery, and distributed exceeds focal by at
least ten at the highest admissible strength. Absence of a crossing is not a refutation,
because the crossing was not predicted by both routes, but its presence is strong
confirmation and its absence must be reported as the weaker outcome rather than passed
over.

Recovery that survives removal of the signal refutes the mechanism claimed in
`MEA_ANALOGUE.md`, which holds that the signal props the system up rather than repairing
it. This is recorded as a prediction that can fail in an interesting direction: it would
mean this system does something the cultures did not.

**The controls, each of which can fail.**

The *no signal* arm is the same run with strength zero, and it must reproduce EXP-002's
collapse. If it does not, something other than the signal is being measured.

The *signal without adjustment* arm applies the signal to the frozen model with nothing
adjusting. It must leave the census at or near thirty one out of thirty one. If injecting
the signal changes the frozen model's behaviour on its own, then any recovery seen in the
main arm may be the signal reorganising the system rather than protecting it, and the
result cannot be read as intended.

The *reflection* control is the one that catches the trap described earlier. At each
strength, the settled states are compared against the injected vectors. If the settled
states are predicted by the injected signals better than they are predicted by the frozen
model's own arrangement, the system is reflecting rather than expressing, and every
result at that strength and above is discarded. This control exists specifically because a
strong enough signal will always produce a diverse looking result.

The *zero strength* gate is the same shape as the project's existing first control: at
strength exactly zero the trajectory must be identical to the existing loop to the last
bit. If it is not, the injection has changed something it should not have.

**Cost.** Two shapes, roughly five strengths, one rule, with two censuses at each because
of the signal on and signal off requirement above: on the order of eighty hours. This is
by far the expensive stage, which is why three cheaper ones precede it, and it is the
stage most likely to need cutting down. If it must be cut, the strengths are thinned
rather than the two censuses, because dropping the signal on census would remove the only
condition the biology predicts will work.

### Stage 4: does the signal have to arrive during the damage?

**What it is.** Two orders, compared. In the first the signal is present while the weights
are adjusting. In the second the weights are adjusted with no signal, which produces the
collapse, and the signal is applied only afterwards, during the census.

**The prediction.** The first should recover census agreement and the second should not,
because the damage is recorded in the weights and a later signal does not reach it.

**Why it is worth running even though the answer seems obvious.** Register row C-52
records that the frozen system shows no history dependence whatever, retracing its own
path exactly when a setting is swept up and then back down. If the adjusting system does
show history dependence, that is a real difference between the frozen and adjusting
systems rather than a restatement, and it is worth having on the record.

**The falsifier.** If applying the signal afterwards recovers census agreement as well as
applying it during, then the collapse is not carried in the weights in the way claimed,
and the account in `MEA_ANALOGUE.md` is wrong about the mechanism.

**Cost.** Two conditions at the single best strength from Stage 3, about four hours.

### Stage 5, conditional: does density decide how fast collapse arrives?

**What it is.** Not a new run. Work already in progress makes the model's one hundred and
forty four small internal mixing units adjustable, rather than the twelve blocks EXP-002
used. The biological finding is that denser cultures began bursting earlier in
development than sparse ones.

**The prediction.** Collapse should arrive in fewer adjustment steps with one hundred and
forty four adjustable sites than with twelve.

**The caution that must be attached.** The two runs differ in what kind of part is being
adjusted, not only in how many, so a difference in speed has an alternative explanation
and this comparison cannot settle it alone. It is recorded as suggestive at best.

**The falsifier.** If collapse arrives no sooner with one hundred and forty four sites
than with twelve, the density reading is wrong.

## What is not being claimed

No claim is made that the system here is learning, and the word does not appear as a
description of any result. The project's standing rules are explicit that the defensible
statement is that the weights carry a trace of what happened, and that there is no
externally specified objective rather than no objective at all.

No claim is made about what any end state means. The end states are the labels the
readout produces and nothing more, following EXP-002's caveat.

No claim is made that the process occurring here is the process occurring in tissue. The
resemblance is in the shape of the failure, and `MEA_ANALOGUE.md` states at length where
it stops.

## Matched conditions

The project checks seventeen conditions mechanically before comparing a connected run
against a disconnected one. Signal strength and signal shape become new conditions that
must match, bringing the count to nineteen, and no comparison may be reported unless all
nineteen agree.

Row C-63 records that the no feedback baseline is exactly zero only when a single site is
adjustable, and is not zero beyond that. Every comparison in this experiment involves
twelve adjustable sites, so no comparison here has a zero baseline, and no quantity
produced by this experiment may be placed in a series with the single site numbers in
rows C-31 or C-58.

## Known limits, stated before the results exist

One input drives the adjustment, one seed, one family of sites, one model, one episode
length. The census uses thirty one inputs rather than the full one hundred and twenty
five. The register already says of itself, in rows C-40 and C-41, that this is the
project's weakest dimension, and this experiment does not fix it.

The injected vector is the input's own starting state. Whether a different choice, such as
fresh noise at every step, behaves differently is not tested here and is named as the
obvious follow up. The biological protocol's signal was not the animal's own prior
activity, so this choice is a departure from the thing being copied and is flagged as one.

The comparison between rules is qualitative only, following EXP-002, because the two rules
cannot be matched on how far they move the weights.
