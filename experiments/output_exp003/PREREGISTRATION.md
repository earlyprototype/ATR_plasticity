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

**The signal has two independent settings, and confusing them was an error in the first
draft of this document.** The biological protocol varies both how strong each pulse is,
set by a voltage, and how often pulses arrive, set in stimuli per second. These are
separate knobs and the finding that matters most is expressed in the second of them. An
earlier version of this file swept only strength and then claimed the biological result
about rate as its prediction. That is a category error and it is corrected here.

**Strength.** A fixed vector `e` scaled to size one is added, multiplied by a number
`beta`, at the point where the state is written back in. The loop becomes `r` becomes
`f(N(r) + beta e)`. A `beta` of 0.01 means the injected signal is one percent as large as
the state it is added to. The strength sweep is the fixed ladder
0.003, 0.010, 0.032, 0.100, 0.316, which is five values in half order of magnitude steps.

**Rate.** The signal is injected on one iteration in every `m`, and is absent on the
others. `m` equal to 1 means every iteration. The rate ladder is `m` in 1, 2, 4, 8, 16.
**This is the axis carrying the biological crossing**, because the measurement it copies
is stated in stimuli per second, and a rate is what stimuli per second is.

**Which axis carries which prediction.** The crossing between concentrated and spread out
injection is predicted on the rate axis, because that is where the source measured it. Any
crossing found on the strength axis is a new hypothesis belonging to this system, is
labelled as such, and may not be quoted as agreement with the biology.

## The grid, which is where both the stimulation and the measurement live

**GPT-2 small is twelve blocks of twelve attention heads, which is a twelve by twelve grid
of 144 addressable sites.** Potter's array was sixty electrodes on an eight by eight grid.
So the grid is not a figure of speech in this experiment, it is the same kind of object at
a comparable and slightly denser scale, and it replaces the twelve injection points an
earlier version of this document assumed were all that was available.

**Operator instruction, adopted: for choosing where to stimulate, the forward direction is
ignored.** The 144 sites are treated as a spatial array to sample from rather than as a
causal chain to reason about. This copies the constraint the original experiments worked
under, since an experimenter choosing electrodes on a plate cannot choose which way signal
propagates through the tissue either, and it has a consequence worth stating plainly: it
removes the temptation to design the stimulation pattern around the mechanism this project
has hypothesised. An earlier draft chose injection points by arguing about what sits before
and after the concentrated sites. That built the conclusion into the design. Choosing by
position instead leaves the depth asymmetry to appear in the results, where it can be
measured, and where the measurement is the effective depth of the stimulation against the
depth it was aimed at.

**The same sites record and stimulate, and that inherits a problem with a known solution.**
In an electrode array the electrode that delivers current is also the electrode that
records, which is why Wagenaar and Potter had to publish a method for suppressing the
artefact of recording through an electrode just stimulated. The same hazard is exact here:
reading the activity of a site on an iteration when current was delivered to it would be
reading the injection back. **Any site stimulated on an iteration is excluded from the
activity measurement for that iteration**, and the number excluded is reported so that a
reader can see how much of the grid the measurement was blind to.

**Where current enters.** A site is a pair of a block and a head. The signal is added to
that head's output before the output projection, at `blocks.{L}.attn.hook_z` indexed by the
head, which is the faithful analogue of current delivered at one location. A whole stream
variant, adding to `blocks.{L}.hook_resid_pre`, is also implemented for comparison with the
earlier design. Strength is always relative to what is already at the site on that pass, so
a `beta` of 0.01 means one percent of local activity wherever it is delivered. That
normalisation matters because EXP-002 found more than a two hundredfold spread of
activation scale across blocks, and a single absolute scale would silently mean different
things at different depths.

**Distributed versus focal, with the fraction taken from the source.** Wagenaar's protocol
used ten to twenty electrodes of sixty, which is a sixth to a third of the array. The same
fraction of 144 is **24 to 48 sites**, and the registered distributed condition is **24
sites drawn at random across the grid** under a fixed seed. Focal is **one site**. That is a
far sharper contrast than the three points out of twelve an earlier version proposed.

**What has to be built, stated because an earlier draft implied it was already there.**
`atr_bridge.make_atr_step` installs one injection hook at one block, fixed at construction,
so grid stimulation needs its own path. It is built in `mea_stim.py` as a step that adds
hooks alongside the loop's existing injection rather than reimplementing the loop, which
the standing rules forbid. **Its gates**: with no plan, or with strength exactly zero, the
trajectory must be bit-identical to the bridge's, and a site colliding with the loop's own
injection point must raise rather than be silently overwritten.

The quantity held equal between the two shapes is **the total injected length, measured in
units of the receiving state's own length at each injection point.** Concretely, each
injection point `j` receives `beta_j` times a unit vector, and the two arms are matched
when the sum of squares of the `beta_j` is equal, so distributed injection at 24 sites
uses `beta` divided by the square root of 24, which is 4.9, at each. Sum of squares rather than plain
sum is chosen because the injected vectors at different depths are close to orthogonal,
being unrelated directions in a high dimensional space, so their combined length grows as
the square root rather than linearly. **This choice is registered here because it decides
the comparison**, and an alternative matching on the plain sum is reported alongside as a
sensitivity check rather than being selected after seeing which one favours the
prediction.

The normalisation of `beta_j` is against the length of the activity already present at
site `j` on the same forward pass, so that strength always means the same thing relative
to local activity regardless of depth. Blocks differ substantially in activation scale,
which is why EXP-002 had to anchor its step sizes per layer, and a shared absolute scale
would repeat that problem.

**Settled, and why the obvious definition is wrong here.** A first draft of this document
defined a trajectory as at rest if consecutive iterations agreed above 0.9999 on a scale
where one means identical, following the wording EXP-002 uses. **That definition is
unusable for this experiment and would have silently destroyed the primary measurement.**

The reason is that one of the five end states in the census is not a resting point. The
committed baseline classifies 34 of its 125 inputs as a two step cycle rather than a fixed
point, all of them in the end state labelled `Divine`, and their consecutive step agreement
sits near 0.68 rather than near 1. Their two step agreement is what reaches 1. The thirty
one input census contains eight such inputs. A criterion built on consecutive steps would
therefore have marked all eight as not settled, and the baseline for the primary
measurement would have been twenty three out of thirty one rather than thirty one out of
thirty one, with the eight discarded inputs being precisely the ones whose behaviour is
most distinctive.

**The definition used here is the committed baseline's, which classifies in two stages.**
A trajectory is a fixed point if the agreement between successive iterations exceeds 0.999
on three consecutive checks taken every ten iterations after iteration 100. It is a two
step cycle if that test fails while the same test on iterations two apart passes. It is
unsettled if neither passes. **Settled means either of the first two.** Both are legitimate
end states, both count toward census agreement, and a change from one class to the other
is itself reported, since register row C-24 treats exactly that change as the strongest
result in the repository.

The thresholds above are the parent's and are used unchanged rather than tightened, so
that a number produced here is comparable with the committed census.

**A discrepancy in EXP-002 that this definition exposes, raised on that experiment's own
review thread rather than resolved here.** EXP-002's reference row reports all thirty one
untouched inputs as at rest while also reporting eight of them in the `Divine` end state,
and its prose defines at rest by agreement between successive steps. Those two statements
cannot both hold under the committed baseline, which puts `Divine` at a consecutive step
agreement near 0.68. Either that experiment's measurement is phase aware and its prose
describes it wrongly, or the count is wrong. This experiment does not depend on which,
because it fixes its own definition above, but the answer changes a number inside claim
C-62 and so belongs with that claim.

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

**The invariance claim, stated precisely rather than loosely.** An earlier version of this
file called the quantity basis free, which is too strong. What is true is narrower. The
lengths are ordinary Euclidean lengths in the model's own coordinates, so the quantity is
unchanged under rotations and reflections of those coordinates but not under an arbitrary
change of coordinates that stretches some directions more than others. That restricted
invariance is the relevant one here, because the objection it answers is that the residual
stream has no privileged set of coordinates, and a rotation is exactly the ambiguity that
objection refers to.

The claim about the rescaling step is also narrower than stated earlier. The quantity is a
ratio of lengths measured within one forward pass, so multiplying the injected state by a
constant leaves it unchanged to the extent that the model's blocks respond proportionally
to their input. They do not respond exactly proportionally, because of the nonlinearities
and the layer normalisation inside each block. **The registered form of the claim is
therefore that the quantity is invariant under rotation exactly, and insensitive to overall
rescaling only approximately**, with the size of that insensitivity measured rather than
assumed: it is checked by recomputing the quantity on states scaled by one half and by two
and reporting how far it moves. If it moves by more than five percent of its range across
that fourfold change in scale, the quantity is reported as scale dependent and is used only
within runs that share a scale.

The quantity never consults a label, which is the property that matters most and which does
hold without qualification.

**The gate, which this stage must pass before any of it is used, stated as a number rather
than as a word.** Measured on the frozen model, where the answer is already known, compute
the quantity for all 125 committed baseline inputs. The test statistic is the ratio of the
spread of the quantity between end states to its spread within them, in the form the
project already uses for exactly this purpose: the mean distance between the group centres
divided by the mean distance of members to their own centre. **The gate is passed if that
ratio exceeds 1.5.**

The threshold of 1.5 is registered here and is chosen against a stated baseline: the token
labels themselves score 0.87 on the same scale, from the numbers in row C-07, and a value
of 1.0 means the groups are indistinguishable from their own internal scatter. A quantity
that does not beat the labels is not worth adopting, and 1.5 asks it to beat them by a
clear margin rather than a marginal one.

**A second gate on dynamical class.** The quantity must separate the 34 inputs the baseline
classifies as a two step cycle from the 91 it classifies as fixed points, with the same
ratio exceeding 1.5.

**The failure direction, which must also be demonstrated.** Within the largest end state
group, the quantity must **not** separate inputs into subgroups: the same ratio computed on
a random split of that group in half must fall below 1.2 in at least nine of ten random
splits. A quantity that separates arbitrary halves of a homogeneous group is finding
structure that is not there.

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

**The gate, stated as a number.** The measure is the participation ratio effective rank
that `experiments/step_size_map.py` already computes, applied to each of the twelve
adjusted matrices. The frozen reference for the single site the project has measured most
is 642.6 out of a possible 768. **The explanation is supported if the mean effective rank
across the twelve adjusted matrices falls by at least ten percent from its frozen value in
the arms that collapsed, and is not supported if it falls by less than two percent.** A
fall between two and ten percent is reported as inconclusive rather than being argued
either way.

Ten percent is registered against a stated baseline: the largest movement in effective rank
anywhere in the committed step size map is an increase of about 0.6 percent, from 642.6 to
646.7, across every ceiling silent cell. So a ten percent fall would be more than an order
of magnitude larger than anything this project has yet seen, which is what the explanation
requires, and two percent is comfortably above the noise that map establishes.

**Stop behaviour.** A failure here does not stop the series, and this is deliberate rather
than an oversight. The collapse is a measured fact whatever explains it, so the later
stages remain meaningful. What a failure removes is the mathematical route to the
distributed versus focal prediction, leaving that prediction resting on the biology alone.
That is recorded prominently if it happens, because it changes how much weight the analogy
is carrying, and it makes the Stage 3 result correspondingly more informative rather than
less.

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
fixed, by applying proportionally larger changes proportionally less often.

**The settings, enumerated.** Adjustments are applied once every `k` iterations for `k` in
1, 10 and 100, with the step size multiplied by `k` so that the total adjustment over the
episode is held equal. The episode is lengthened so that the largest setting still applies
one hundred and twenty adjustments, matching EXP-002. The ratio between the extreme
settings is one hundred, which is what the falsifier below refers to. The values 2 and 4
that the project has used before are not included, because they span a factor of two and
the question here is whether two orders of magnitude changes anything.

**The prediction.** Collapse should recede as the adjustment is slowed. Census agreement
should rise monotonically with `k`.

**The falsifier, stated numerically.** If census agreement at `k` equal to 100 does not
exceed census agreement at `k` equal to 1 by at least five out of thirty one, the
timescale reading is dead and will be recorded as dead. Five is chosen because it is the
smallest difference that exceeds the count of any single minority end state in the
untouched census apart from the largest two, so a difference below it could be produced by
one end state moving.

**The confound that must be reported with the result.** Multiplying the step size by `k`
holds the total adjustment equal only if the rule's effect is linear in step size, and
EXP-002 recorded that it is not for the eroding rule, whose drift spanned twelve times
across layers at one anchored setting. The achieved drift at each `k` is therefore
reported, and if it varies by more than a factor of two across the three settings, the
comparison is qualitative only and the falsifier above is not invoked.

**Why this is second.** It is the cheapest experiment that could remove a whole branch of
the argument, and it needs no new code beyond what exists.

**Cost.** Roughly three minutes per adjustment episode and roughly ninety minutes per
census of thirty one inputs, on the processor available. Four settings, one rule, gives
about six hours.

### Stage 3: the signal, focal against distributed

**What it is.** The main experiment. Under the reinforcing rule at all twelve blocks,
with the loop closed, the signal is injected during the adjustment. Two shapes are
compared at matched total strength: all of it at a single site, against the same total
divided among 24 sites drawn at random across the 144-site grid under a fixed seed. Sites
are chosen by position and not by depth, per the operator instruction recorded above, so
the forward direction is ignored in the design and measured in the result.

**The grid.** The rate ladder `m` in 1, 2, 4, 8, 16 is run at the fixed strength 0.032,
which is the middle of the strength ladder. The strength ladder 0.003, 0.010, 0.032,
0.100, 0.316 is run at the fixed rate `m` equal to 1. Both ladders are run for both
shapes. This is a cross rather than a full grid, twenty settings rather than fifty, and
the cross is centred on the setting the two ladders share so that the two axes are
anchored to a common point.

**What happens when the reflection control fires part way up a ladder.** Every setting at
or above the firing point is discarded. If that leaves fewer than three admissible
settings on the strength ladder, the crossing test on that ladder is reported as
inconclusive rather than as a null, because a crossing cannot be located in two points.
The rate ladder is unaffected by strength driven reflection, since strength is fixed
along it, and this is one reason the rate axis carries the primary prediction.

**The census is taken twice at every setting, once with the signal still being applied and
once with it removed.** This is not a robustness check, it is the experiment. The
biological measurement being copied reports that "bursting resumed as soon as stimulation
was stopped", and a companion paper by the same authors says of that work that "we saw no
plastic effects from burst quieting". So the prediction is that recovery appears in the
census taken with the signal on and disappears in the census taken with it off. An
experiment that measured only the second would be predicted by the very source it is
copying to find nothing, and reporting that as a refutation would be an error. This
paragraph exists so that the error is not available.

**The exact procedure for the two censuses, because the order could otherwise contaminate
the comparison.** One adjustment episode is run per setting with the signal applied. At
the end of that episode the weights are frozen and copied, and **all adjustment is
disabled for everything that follows**. Both censuses then run from that single frozen copy
of the weights, so neither can affect the other and the order they are run in cannot
matter. The only difference between them is whether the signal is injected during the
census itself. Each census input starts from its own clean initial state, exactly as the
committed reference census does, and is run for the same one hundred and twenty
iterations. No re-equilibration period is used, and none is needed, because nothing is
adapting during either census.

**The predictions.** With the signal still applied, distributed injection should recover
census agreement. Focal injection at matched total strength should recover substantially
less at high strength, because a signal entering before the first concentrated site is
converted to that site's fixed direction immediately.

**The prediction that would be hardest to produce by accident, and is therefore the one
worth the most.** The biological measurement does not say spread out is simply better. It
records a crossing point: below roughly ten stimuli per second, spreading the signal
across many electrodes gave slightly worse burst reduction than concentrating it in one,
and above that rate spreading it gave much better reduction.

**That crossing is in rate, so the prediction is registered on the rate ladder and not the
strength ladder.** Concentrated injection should give higher census agreement than spread
out injection at low rate, meaning large `m`, and spread out injection should give higher
agreement at high rate, meaning `m` equal to 1, with a crossing between. Nothing in the
mathematical account predicts a crossing, so finding one would be an agreement that no
part of this design was built to produce.

Any crossing observed on the strength ladder is recorded as a separate and new hypothesis
about this system. It is not agreement with the biology, because the biological
measurement is not about strength, and it may not be quoted as such.

**The falsifiers, stated numerically.**

The distributed advantage is claimed only if distributed injection reaches a census
agreement at least ten out of thirty one higher than the best focal result at the same
setting, on the signal on census. Ten is roughly a third of the census and is well above
the one point difference that separated the two collapsed arms of EXP-002. If focal
matches distributed within that margin at every admissible setting on both ladders, the
central prediction of `MEA_ANALOGUE.md` has failed and that document is withdrawn.

The crossing is claimed only if, on the rate ladder, focal exceeds distributed by at least
five out of thirty one at `m` equal to 16 and distributed exceeds focal by at least ten at
`m` equal to 1. Absence of a crossing is not a refutation, because the crossing was not
predicted by both routes, but its presence is strong
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
adjusting. **It must leave census agreement at twenty eight out of thirty one or above.**
Twenty eight allows three inputs to move, which is a tenth of the census, against a
baseline of thirty one out of thirty one. If a setting drops this arm below twenty eight,
then at that setting the signal is reorganising the frozen model on its own, any recovery
seen in the main arm at that setting cannot be read as protection, and **that setting is
excluded from the main analysis and reported as excluded.**

The *reflection* control is the one that catches the trap described earlier, and its rule
is written out in full here because it decides which data are analysed and a vague rule
would let that decision be made after seeing the results.

For each setting, form two matrices of pairwise distances across the thirty one census
inputs. The first holds the distances between the settled states, measured as one minus
the cosine between them. The second holds the same distances between the injected vectors.
A third matrix holds the distances between the settled states of the frozen model, which is
the committed reference. Compute the Spearman rank correlation between the settled state
distances and each of the other two, using the 465 unordered pairs. Call these `s_signal`
and `s_frozen`.

**The rule: if `s_signal` exceeds `s_frozen` by more than 0.10, the setting is judged
reflection dominated, and that setting and every stronger one on the same ladder are
discarded.** The threshold of 0.10 on a correlation scale running from minus one to one is
registered here rather than chosen later. Rank correlation is used rather than a
correlation on the raw values because only the ordering of the distances is meaningful.
Ties are broken toward discarding, so an exact difference of 0.10 discards.

The *zero strength* gate is the same shape as the project's existing first control: at
strength exactly zero the trajectory must be identical to the existing loop to the last
bit, and the spread out injection path with one injection point must be identical to the
single point path to the last bit. If either fails, the injection has changed something it
should not have.

**Cost, corrected.** The cross described above is twenty settings, being two shapes times
ten ladder points, less the two shared centre points, so eighteen distinct settings. Each
needs one adjustment episode of about three minutes and two censuses of about ninety
minutes, giving about three hours per setting and **about fifty five hours in total**.

An earlier version of this file said eighty hours, which was wrong: it was an estimate made
before the grid was reduced from a full grid to a cross, and it was not recomputed when the
grid changed. This is by far the most expensive stage, which is why three cheaper ones
precede it. If it must be cut, the strength ladder is thinned first, because the primary
prediction lives on the rate ladder, and the two censuses are never cut, because dropping
the signal on census would remove the only condition the biology predicts will work.

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

**The falsifier.** If applying the signal afterwards recovers census agreement to within
five out of thirty one of what applying it during achieves, then the collapse is not
carried in the weights in the way claimed, and the account in `MEA_ANALOGUE.md` is wrong
about the mechanism.

**The setting used, fixed in advance so it is not chosen to suit the result.** Both
conditions run at the shape and setting that gave the highest signal on census agreement in
Stage 3, with ties broken toward the lower strength and then the larger `m`. If Stage 3
produced no setting whose signal on census agreement exceeded its no signal arm by at least
five out of thirty one, Stage 4 does not run at all, because there is no recovery whose
timing could be tested.

**Cost.** Two conditions, about six hours.

### Stage 5, conditional: does density decide how fast collapse arrives?

**What it is.** Not a new run. Work already in progress makes the model's one hundred and
forty four small internal mixing units adjustable, rather than the twelve blocks EXP-002
used. The biological finding is that denser cultures began bursting earlier in
development than sparse ones.

**The prediction.** Collapse should arrive in fewer adjustment steps with one hundred and
forty four adjustable sites than with twelve.

**This stage carries no falsifier, and an earlier version of this file wrongly gave it
one.** The two runs differ in two ways at once, in how many sites are adjustable and in
what kind of part each site is, so neither outcome can be attributed to density. If
collapse arrives sooner with one hundred and forty four sites, that is consistent with
density and equally consistent with the mixing units simply being more consequential
individually. If it does not arrive sooner, that is consistent with density being wrong and
equally consistent with a density effect being masked by a site type effect in the opposite
direction. **A comparison that cannot distinguish its own alternatives is not a test**, and
this project's standing rules say that a control which cannot fail is worse than none.

The stage is therefore reported as an observation and is explicitly non falsifying. The
earlier version stated the confound in one paragraph and then invoked a falsifier in the
next, which was a contradiction inside a single section.

**What would make it a test**, recorded so that it is available later: hold the site type
constant and vary only the count, by running the twelve block configuration against a
random subset of four blocks and against all twelve, at matched total adjustment. That is a
new run and is not part of this experiment.

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

---

## Amendment 1: the cadence ladder, written before Stage 2 runs

**What changes.** Stage 2 was registered above as applying adjustments once every
`k` iterations for `k` in 1, 10 and 100, with the episode lengthened so that the
largest setting still applied one hundred and twenty adjustments.

**Why that is not runnable.** Holding the adjustment count at one hundred and
twenty means the `k` equal to 100 cell needs twelve thousand iterations. At the
measured rate of roughly 0.44 seconds per iteration on the processor available,
that single cell is about ninety minutes, and the ladder as registered costs more
than the whole rest of the experiment. The figure comes from the committed
baseline, which records 132 seconds for a three hundred iteration run.

**What it becomes.** The episode is held at one hundred and twenty iterations and
the ladder is `k` in 1, 4 and 12, giving one hundred and twenty, thirty and ten
adjustments. The step size is multiplied by `k` so that the total adjustment is
held equal across the ladder, which is what the comparison needs.

**What this costs in what can be concluded, stated plainly.** The extreme settings
now differ by a factor of twelve rather than one hundred, so the falsifier
registered above is restated: **if census agreement at `k` equal to 12 does not
exceed census agreement at `k` equal to 1 by at least five out of thirty one, the
timescale reading is not supported at this resolution.** That is a weaker statement
than the original. A twelvefold change failing to move anything does not exclude an
effect that needs two orders of magnitude, and the result must say so rather than
being reported as a clean refutation.

**A second consequence, which is a gain rather than a loss.** At `k` equal to 12
the episode applies ten adjustments rather than one hundred and twenty, so the
ladder now spans from many small increments to few large ones at equal total
change. That is the timescale separation question in its clearest form: the fast
variable has twelve times longer to settle between moves of the slow one. It also
places the far end of the ladder near the single large edit that register rows C-56
and T1.1 already characterise, which gives the result an existing point of contact.

**The registered comparison is unchanged otherwise**: same driven prompt, same
twelve sites, same reinforcing rule, same lifted ceiling, and the achieved drift is
reported at every setting so that a spread of more than a factor of two across the
ladder demotes the comparison to qualitative, as registered above.

---

## Amendment 2: three gaps a review found, closed before Stage 3 runs

All three are places where this file left a choice open that could otherwise have been
made after seeing results. Recorded here rather than edited in above, so the change is
visible.

**The reflection control did not say which census it reads.** Every setting produces two
censuses, one with the signal applied and one without, and the control discards data, so
reading a different one could discard different points. **It reads the signal-on census
only.** That is the census the recovery prediction lives in, so it is the one where a
reflected signal would masquerade as a result. The signal-off census is never used to
discard.

For a settled state that is a two-step cycle rather than a fixed point, the representative
state is **the mean of the last two iterates**. Taking either single iterate would make the
value depend on which phase the run happened to stop in, which is the aliasing error the
project's standing prohibitions already name.

**Neither arm's sites were pinned.** The distributed arm said 24 sites under a fixed seed
without giving the seed, and the focal arm said one site without saying which. Site depth
and local activity both matter, so both are now fixed:

- **Distributed:** 24 of the 144 sites, drawn without replacement by
  `torch.Generator().manual_seed(20260802)` over the sites ordered `(block, head)` with
  block varying slowest. The realised list is written into the run record so it can be
  checked rather than trusted.
- **Focal, primary:** the single site `(block 6, head 8)`. Block 6 is where every
  committed single-site result in this repository was measured, which makes it the least
  arbitrary choice available. Head 8 is a fixed choice with no further justification. An
  earlier version of this line called it the median head index, which is simply wrong:
  with twelve heads indexed 0 to 11 there is no single median, and 8 is not it under
  either convention. The claim is withdrawn and the coordinate stands as an arbitrary but
  pinned one.
- **Focal, second arm:** the lower median of the realised distributed draw, defined
  exactly so it is reproducible. Sort the 24 drawn sites ascending by `(block, head)` and
  take element 11, the lower of the two middle entries of an even-sized list. This arm
  exists so that a focal result cannot be dismissed as an unlucky site. If the two focal
  arms disagree by more than five out of thirty one on census agreement, the focal
  condition is reported as site-dependent and no distributed-versus-focal claim is made at
  that setting.

**Stage 1's scope was overstated here.** This file described Stage 1 as also reporting the
depth-weighted centre of the weight change and a smallest-detectable-change sensitivity.
The runner does neither: it reproduces EXP-002's episode and measures spectral
concentration on the twelve matrices, which is the concentration gate and nothing more.
Those two further measurements are **not run** and are not reported. The gate that was
registered is the one that was tested, and its result stands; the additional measurements
move to the follow-up list.

**One consequence of the reproduction requirement, made explicit.** If the episode fails
to reproduce EXP-002 within five percent, the concentration verdict describes a different
episode and is **invalid rather than merely caveated**. The runner now prints that
alongside the verdict so it cannot be read out of context from the console.
