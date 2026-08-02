# The cultured network analogue

*Positioning, not measurement. This document argues that the experiment this repository
has been running is a computational version of an experiment that was run in living
tissue for roughly fifteen years, and that the older experiment's findings are therefore
predictions here rather than decorations. Every measurement it cites lives in
`CLAIMS.md` and was made before this document existed. Nothing here enters the claim
register. This document answers the open task recorded at the foot of
`ISSUE_dynamical_systems_interp.md`, which asked whether the biological framing belongs
in the repository at all.*

## The answer first

When a network of locally adjustable units is closed on itself, with no input arriving
from outside, it collapses. Every input it is later given produces the same output. The
network stops being able to express more than one thing.

That is what experiment EXP-002 found in this repository last week, and it is what Steve
Potter's laboratory at Georgia Tech found in living cortical tissue, repeatedly, from the
late nineteen nineties onward. The two systems are made of entirely different material
and the measurements are of entirely different quantities, but the structure of the
setup is the same and so is the failure.

The reason this matters is not that a resemblance is pleasing. It is that the
Potter laboratory did not stop at the failure. They found what prevented it, and they
found it in a form specific enough to copy: a small, continuous, deliberately thin
signal fed into many places at once. That finding has not been tested here. It is
therefore a prediction, and it is falsifiable, which is the only thing that makes an
analogy worth writing down.

## What a cultured network is, and what it does

A dissociated cortical culture is brain tissue taken apart into individual cells and
grown on a glass plate that has electrodes built into it. The electrodes let an
experimenter both listen to the cells and stimulate them. The cells regrow connections
to each other over days and weeks, so the result is a real neural network, made of real
neurons, that can be watched continuously. What it does not have is a body. No senses
feed into it and no muscles lead out of it. It is connected only to itself.

Such a culture reliably develops a pathology. Instead of producing varied patterns of
activity, it falls into whole network bursts: almost every cell firing at once, in short
violent episodes, over and over. Wagenaar, Pine and Potter followed fifty eight cultures
through their first five weeks and reported that the share of activity taken up by these
whole network bursts kept increasing as the cultures matured, and that denser cultures,
meaning ones with more cells and therefore more connections, began bursting earlier in
development than sparse ones.

The important point is what the bursting costs. A network that is bursting is not
expressing anything else. Whatever variety of patterns it might have been capable of is
replaced by one pattern, repeated. Potter's own framing, in a review of this work, is
that population bursting in culture is worth studying as a model of *deafferentation
syndromes*, meaning conditions caused by a loss of incoming signal, and he names chronic
pain and epilepsy as the clinical cases. The culture bursts because it has been cut off
from input.

That last sentence is the joint on which this entire document turns, and it is Potter's
account rather than an interpretation added here.

## What this repository built, without meaning to

The loop this project studies takes a frozen language model, reads the internal state
out of the last layer, rescales it, and writes it back into the first layer, hundreds of
times. Nothing enters from outside once the loop has started. The system is connected
only to itself. In the vocabulary above, it is deafferented by construction.

For most of this project's life that did not make it a cultured network, because only
one small part of the model was allowed to change. One adjustable site in a frozen model
is a single electrode, and a single electrode is not a culture. The comparison would
have been empty.

That changed with EXP-002, which made all twelve of the model's mid layer processing
blocks adjustable at once, each one adjusting itself from the activity passing through
it, with no target and no instruction. At that point the setup has the property that
matters: many locally adjustable elements, closed on themselves, with nothing coming in.

It then did the thing the cultures do.

**A note on the status of that experiment, because two documents in this repository
currently disagree about it.** `HANDOVER.md` lists EXP-002 under work that has not been
run, and that file is correct about the state of the main branch. The experiment has been
run and its results and register rows are committed, but on an unmerged branch under review
as pull request 57. Everything this document says about EXP-002 therefore depends on that
review concluding and those rows entering the register. Until it does, the collapse result
should be read as measured and committed but not yet accepted, and if the review changes
the numbers this document changes with them. The experiment plan that accompanies this file
does not begin until that pull request lands, for the same reason.

Before any adjustment, thirty one test inputs settled into five distinguishable
end states, distributed as thirteen, eight, five, four and one. After the adjustment,
under the reinforcing rule with the loop closed, twenty seven of the thirty one settled
into a single end state. Under the same rule with the loop cut, thirty of thirty one
settled into a single end state. Under the eroding rule with the loop cut, all thirty
one settled into one. The register records this as claim C-62, whose own summary is
"collapse, not steering".

For scale: five distinct end states falling to one is the entire measured structure of
this system disappearing. The baseline is not "a few inputs moved". The baseline is that
before the change, the system distinguished its inputs, and afterwards it did not.

## The array, which turns out to be the closest correspondence of all

Everything above compares what the two systems *do*. There is a closer
correspondence in what they *are*, and it was pointed out by the operator rather
than found here.

A cultured network is studied through an array of electrodes laid out on a grid.
Potter's was sixty electrodes arranged eight by eight. The electrodes are how the
network is both listened to and stimulated, and their positions are the only
geometry the experiment has.

GPT-2 small is twelve blocks of twelve attention heads. That is a twelve by twelve
grid of 144 sites, slightly denser than the array and of the same order. Each site
can be read, because a head's contribution to the model's internal state is a
well defined quantity, and each site can be written to, because a signal can be
added to that head's output. So the grid is not a way of speaking about the model.
It is the same kind of object the biological work was built around.

Two consequences follow, and the second is the useful one.

The first is that the measurement described later in this document can use the
original equation rather than a reduction of it, because that equation was written
for a two dimensional array and now there is one.

The second is a problem inherited along with the apparatus. In an electrode array
the electrode that delivers current is the same electrode that records, which is
why the same laboratory had to develop a method for suppressing the artefact of
recording through an electrode just stimulated. The identical hazard exists here:
reading a site's activity on an iteration when a signal was injected into it would
be reading the injection back rather than the network's response. That has to be
handled, it is handled by excluding stimulated sites from the measurement on the
iterations they fire, and it is the sort of thing that would have been discovered
late and expensively if the grid had not made it obvious early.

**One limit, because the correspondence is not perfect and pretending otherwise
would be the failure this document keeps warning about.** In the array, position
means the same thing everywhere: two electrodes two hundred micrometres apart are
two hundred micrometres apart wherever they sit. In the grid only one axis is like
that. Block index is ordered and meaningful, and activity really does flow along
it. Head index is a label with no canonical order, and permuting the labels changes
nothing about the model. So the grid is a genuine array in one direction and an
unordered set in the other.

That is worked around rather than hidden. Choosing where to stimulate needs only a
set to sample from, so the grid serves fully for that. The measurement uses only
the axis that has a metric. And the arbitrariness of the other axis becomes the
most reliable control available here, because permuting head labels must leave
every result exactly unchanged, which is a control whose correct answer is known
before it is run.

## The one result that is not collapse, and why it is the more important one

One arm of EXP-002 produced nineteen distinct end states rather than one, which reads at
first glance as diversity surviving. It was not. Only four of the thirty one trajectories
had come to rest by the end of the run. The other twenty seven were still moving when the
measurement was taken, so the nineteen different answers were nineteen snapshots of
motion, not nineteen stable states. The results document says so directly and refuses the
optimistic reading.

This matters beyond EXP-002, because it is the same trap that any experiment adding an
external signal will walk into. A system being pushed hard from outside will also produce
many different answers, and those answers will also not be the system expressing
anything. They will be the push, reflected. Counting how many different outcomes occur is
therefore not a measurement of health, and this repository has already proved that on its
own data.

The correct measurement is given below.

## Why the collapse happens: the explanation this document offered, and its refutation

**RETRACTED, 2026-08-02, by the test this section itself named as deciding.** The
mechanism set out below was measured in EXP-003 Stage 1 and it is wrong. It is kept
in place rather than deleted, because this repository keeps what it used to believe,
and because the reasoning that follows is still the clearest statement of what was
tested.

What was claimed: that repeated rank-one adjustments concentrate each weight matrix
onto a single direction, so that twelve adjusted sites in series leave the loop able
to produce only one resting state.

What was measured, reproducing EXP-002's episode exactly to twelve significant
figures: the mean effective rank of the twelve matrices falls by **0.044 percent**.
The threshold registered in advance for the explanation to be supported was a fall of
10 percent, and for refutation a fall below 2 percent. The share of each matrix held
by its strongest direction, which the valve picture requires to grow toward 1, moves
from 0.00777 to 0.00780, very slightly the wrong way. No site shows the effect; the
largest fall anywhere is 0.244 percent.

**So the weights come out of the collapse with essentially the character they went in
with, on the two measures that were taken.** Whatever destroys this model's ability to
tell its inputs apart, it is not **the spectral concentration this stage tested**.

**That is narrower than an earlier version of this sentence claimed**, which said the
collapse was not a loss of capacity in the matrices at all. Effective rank and the
dominant direction's share are two specific quantities. They do not measure how well a
matrix separates the inputs it actually receives, and a matrix can lose the ability to
distinguish the states this loop visits while its spectrum barely moves. Other
capacity-loss mechanisms are untested, and the honest statement is that the proposed
one is refuted rather than that the whole family is. Full numbers in
`experiments/output_exp003/STAGE1_RESULTS.md`.

**What this costs the argument, stated rather than absorbed.** The
distributed-beats-focal prediction had two independent routes, one from the biology
and one from this mechanism. The second is gone, and the prediction now rests on the
biological measurement alone. The pre-registration anticipated exactly this and said
the series continues, because the collapse is a measured fact whatever explains it,
and because a spread-out signal still beating a concentrated one when no mathematical
argument predicts it would be a stronger result for the analogy rather than a weaker
one.

**What the refutation suggests instead, marked as speculation and untested.** The
weights barely change in character while the behaviour is destroyed entirely, which
points at the landscape rather than the matrices. Row C-07 records that the spread of
states sharing an end state exceeds the gap between the two nearest distinct end
states, and row C-55 records that arbitrary directions at matched displacement
usually move the settled end state. Together they suggest a landscape on a knife edge
that almost any coherent push of sufficient size will topple, in which case the
collapse is not plasticity degrading the model but a barely separated structure being
displaced wholesale by a small change that leaves the matrices spectrally intact.

---

*The original argument, kept for the record. It was inference from results already in
the register, not a new measurement, and it was marked as such at the time.*

Register row C-10 establishes what the reinforcing rule actually computes. Each time it
fires, it adds to the weight matrix a quantity equal to the matrix multiplied by the
input correlation structure, plus a smaller fixed term. The row also records that each
individual adjustment is of the simplest possible shape, technically rank one, meaning it
writes along a single direction rather than adjusting many independent things.

Repeat that. An adjustment that always writes along a direction determined by the
activity, applied to a matrix that then shapes the activity, is a process that
concentrates. Its effect is to make one direction inside the matrix grow relative to all
the others. If that goes far enough, the site stops performing a varied transformation on
what arrives, and instead emits roughly the same direction regardless of input, scaled by
how much the input happens to overlap with one particular pattern. A site behaving that way
is closer to a valve than to a map, and a series of such valves would confine what the loop
can produce toward a line, which can support only one resting point.

**That chain is a hypothesis with a substantial obstacle in front of it, and an earlier
version of this document presented it far too confidently.** Four things have to be said
against it, and the fourth is serious.

The first is that a rank one *adjustment* does not make the *matrix* rank one. The matrix
being adjusted starts full, and register row C-11 records its stable rank as about 31 on a
scale where 768 would be the maximum, so it has real structure to begin with. What the
adjustments add is small by comparison.

The second is how small. EXP-002's reinforcing arm moved the weights by 1.31 percent of
their own size in aggregate. **A one percent addition does not dominate a matrix**, and the
valve picture requires domination, not presence. On the face of it the arithmetic does not
support the story.

The third is that the argument treats the model as though it were linear. Each block
contains nonlinearities and a normalisation step, and the loop applies a further
normalisation on every iteration. A fixed point argument that ignores all of these is a
sketch of a mechanism rather than a derivation.

The fourth is that the successive adjustments need not point the same way. The
concentration story requires them to align, so that they accumulate rather than cancelling.
Whether they do is a measurable property of the run and has not been measured.

**So the honest position is that this is the leading candidate explanation and it is
currently in tension with its own arithmetic.** Marking the status plainly: C-10 is measured
and supported, C-62 is measured and supported, and the connection between them proposed
here is untested inference which the numbers do not obviously favour.

That tension is the reason Stage 1 of the experiment is worth running first rather than
being a formality. This repository already measures how concentrated a matrix has become,
and the prediction is specific: the concentration measure should fall by at least ten
percent in the runs that collapsed, where the largest movement anywhere in the committed
step size map is an increase of about 0.6 percent. If it does not move by even two percent,
this explanation is wrong, the collapse needs a different account, and one of the two
independent routes to the main prediction of this document disappears. The cost of finding
out is zero, because the runs already exist.

## What the Potter laboratory did about it

Two findings, both verified against the sources during the preparation of this document,
and both quoted rather than paraphrased because the details are the useful part.

The first is the remedy. In a review of the laboratory's programme, Potter describes the
background signal they fed to their cultures as "distributed, low frequency stimulation
(~1 Hz/electrode across 10-20 electrodes)", and says this "allows the networks to continue
to respond to meaningful sensory input". The paper this refers to is Wagenaar, Madhavan,
Pine and Potter, published in the Journal of Neuroscience in 2005, and its title states
the method: *Controlling bursting in cortical cultures with closed-loop multi-electrode
stimulation*.

Three details in that description are doing real work and none of them is decorative.
The signal is spread across many electrodes rather than concentrated in one. The rate at
each electrode is low, about one pulse per second, rather than strong. And the loop is
closed, meaning the stimulation responds to what the network is doing rather than running
to a fixed schedule.

**Spreading the signal out is not simply better, and an earlier draft of this document
had that wrong.** The 2005 paper compares stimulating through one electrode against
stimulating through many, and the result depends on the overall rate. In the paper's own
words: "At intermediate frequencies (2-10 stim/sec), this protocol resulted in somewhat
higher burstiness than single-electrode stimulation, but at frequencies above 10 stim/sec,
multi-electrode stimulation resulted in greatly improved burst reduction." Below about ten
stimuli per second, spreading the signal out made the bursting slightly worse than
concentrating it. Above that rate, spreading it out was much better. There is a crossing
point, and which side of it you are on decides the answer.

A second caution belongs with that one. The best spread out condition in the paper used
both more electrodes and five times the overall rate of the best concentrated condition,
so those two conditions differ in two ways at once and the comparison cannot separate
them. The only comparison in the paper that holds the rate fixed is the one that also
closed the loop, adjusting the strength at each electrode according to how the network
was responding, and that condition won. What the paper supports at matched rate is
therefore narrower than "spread out beats concentrated". It is that adjusting the signal
in response to the network beats a fixed spread out signal, which in turn beats a fixed
concentrated one.

The second finding is why any of this is worth the trouble. Potter, in the same review,
writes that the laboratory had "recently demonstrated that quieting bursts aids the
induction and detection of lasting functional plasticity".

That sentence reframes the whole enterprise. The external signal is not itself the
interesting result. It is the precondition that makes anything else observable. A network
that is collapsed cannot be shown to have learned anything, because it responds
identically to everything, and so no change in it can be detected. Quieting the collapse
is what makes the measurement possible.

The corroborating evidence for that reading is a separate paper by the same laboratory,
Wagenaar, Pine and Potter, published in the Journal of Negative Results in Biomedicine in
2006 under the title *Searching for plasticity in dissociated cortical cultures on
multi-electrode arrays*. It is, as the venue implies, a report of failure: a series of
protocols intended to induce lasting change, most of which did not work. The paper's own
words for the exception are "One successful protocol with burst quieting in Series III".

**What quieted the bursts in that one successful protocol was a chemical, not a signal,
and this correction matters enough to state loudly.** The paper as originally published
was wrong about which of its protocols succeeded. A correction issued by the same authors
in the same journal in 2007 swapped the labels of two protocols throughout, with the
consequence that the successful one is the protocol that suppressed bursting by adding
magnesium to the growth medium, not the protocol that suppressed it with distributed
electrical stimulation. Both of the protocols that used distributed electrical stimulation
to quiet bursts failed. The correction also instructed that the paragraph in the original
discussion crediting distributed electrical stimulation "should be dropped entirely".

The uncorrected version of that paper is still the one served by at least two public
archives, and an earlier draft of this document quoted it and drew the wrong conclusion.
The corrected reading is weaker in an important way: it supports the claim that quieting
the collapse is what makes plasticity detectable, and it does **not** support the claim
that electrical stimulation is what should do the quieting.

**A third finding, which constrains what the remedy can be expected to do.** The 2005
paper reports that "bursting resumed as soon as stimulation was stopped", and the 2006
paper says of that earlier work that "we saw no plastic effects from burst quieting". The
external signal does not repair the network and does not leave anything behind. It holds
the network off its attractor for exactly as long as it is applied.

That is a real constraint on the experiment rather than a detail, and it is discussed
again in the predictions below, because it means a test that removes the signal before
measuring is testing something the biology says will not work.

## What the analogy predicts here

These are stated as predictions, before the experiments, so that they can fail. The
detailed forms, with the numerical thresholds that decide each one, are in the
pre-registration for experiment EXP-003. What follows is the reasoning.

**The remedy should work only while it is applied.** Feeding a small external signal into
the loop, while the adjustable sites are adjusting, should preserve the ability to
distinguish inputs that is otherwise destroyed, and that ability should disappear again
when the signal is removed. The second half of this prediction is as important as the
first and comes directly from the correction recorded above: the cultures resumed bursting
as soon as the stimulation stopped. An experiment that applies the signal, then takes it
away, then measures, is predicted to see nothing. If the signal turns out to leave lasting
structure behind, the analogy is wrong in an interesting direction rather than a dull one,
because that would be a difference between the two systems rather than a failure of the
comparison.

**Spreading the signal out should beat concentrating it above some strength, and lose to
it below.** This is the prediction worth the most, and it is sharper than the version an
earlier draft of this document contained, which simply said spread out should win. The
biological result has a crossing point: below roughly ten stimuli per second, spreading
the signal across many electrodes was slightly worse than concentrating it in one, and
above that rate it was much better.

Two independent routes arrive at the upper half of that prediction. The biological route
is the measurement just quoted. The mathematical route is the argument given above, with
the obstacles recorded there attached to it: if each adjustable site has become a valve
emitting a fixed direction, then a signal injected before the first site is converted into
that fixed direction immediately and contributes only a change of scale, whereas a signal
entering between sites is not. Neither route is obvious from the other, and they agree
about the strong signal case.

**The crossing is in rate, not in strength, and getting that wrong would have invalidated
the prediction.** The biological measurement is expressed in stimuli per second, which is
how often pulses arrive, and not in the voltage of each pulse. Those are two separate
settings in that laboratory's protocol and both were varied. An earlier version of this
document proposed sweeping only the strength of the injected signal and then claimed the
crossing as its prediction, which compares a result about rate against an experiment about
amplitude. The corrected experiment sweeps how often the signal is injected as a separate
setting, and the crossing prediction is registered on that setting alone. A crossing found
in strength would be a new observation about this system rather than agreement with the
biology, and the pre-registration says so.

Neither route predicts the crossing point, and that is what makes it the most valuable
thing to look for. A crossing point is a specific and unlikely shape. If the experiment
here shows spread out injection losing at low rate and winning at high rate, that is a
quantitative agreement nobody designed for. If it shows spread out injection simply winning
everywhere, the agreement is weaker and more easily explained by the mathematics alone.

**The signal should have to respond to the system, not merely be present.** The only
comparison in the 2005 paper that holds the overall rate fixed found that adjusting the
signal according to how the network was responding beat a fixed signal of the same
strength. This is the least tested of the predictions here and the most likely to be
beyond what this project can currently build, so it is recorded as a direction rather than
as a stage of the experiment that follows.

**The signal should have to arrive during the adjustment, not after it.** The damage is
recorded in the weights, and the weights are what a later input meets. So a signal
applied afterwards should not repair anything. This one is sharpened by an existing
result: register row C-52 records that the frozen system shows no history dependence at
all, retracing its own path exactly when a parameter is swept up and then down again. If
the adjusting system does show history dependence, that is a genuine difference between
the two, not a restatement of something already known.

**More adjustable sites may collapse sooner, and this one cannot be a test.** The
developmental finding is that denser cultures, with more cells and more connections, began
bursting earlier than sparse ones. The corresponding quantity here is the number of sites
allowed to adjust: EXP-002 used twelve, and work in progress uses the model's small
internal mixing units instead, of which there are one hundred and forty four.

**No falsifier is offered, and an earlier version of this document wrongly gave one.**
Those two runs differ in how many sites adjust and in what kind of part each site is, both
at once, so neither outcome can be attributed to density. Collapsing sooner is equally
consistent with the mixing units simply mattering more individually; not collapsing sooner
is equally consistent with a density effect masked by a site-type effect pulling the other
way. The pre-registration struck this falsifier for that reason and this paragraph now
matches it. What would make it a test is holding the site type fixed and varying only the
count, which is a run nobody has done.

## What would falsify the analogy

Stated plainly and in advance, because an analogy that cannot fail is worthless and this
project's standing rules say so in those words.

The analogy is refuted if a signal injected at a single place restores the ability to
distinguish inputs just as well as the same total signal spread across several places, at
every strength tested. That would mean the specific thing the Potter laboratory discovered
does not transfer, and what remains is only the generic observation that pushing a system
from outside disturbs it.

The analogy is strongly confirmed, in a way that would be hard to explain by accident, if
the two shapes of signal cross over: concentrated injection winning at low strength and
spread out injection winning at high strength. That is the pattern the 2005 measurements
show, and no part of the mathematical argument predicts it. A crossing point is the
specific result to look for.

The analogy is refuted in its account of the mechanism if the recovery survives removal of
the signal. The cultures resumed bursting the moment stimulation stopped, so a signal that
leaves lasting structure behind here is doing something the biological one did not.

The analogy is substantially weakened if the collapse turns out to be an artifact of how
the end state is read. This repository's own row C-07 records that the labels used to
name end states are barely able to tell the states apart: the spread of states sharing a
label is larger than the gap between the two nearest distinct labels, by a ratio of
about one point one six. If the states in the collapsed runs are in fact as far apart as
they ever were, and only the naming has become uniform, then nothing dynamical collapsed
and the resemblance to bursting is empty. Testing this requires a measurement that does
not use the labels at all, which is why EXP-003 begins by building one.

The analogy is weakened, though not refuted, if the collapse is insensitive to how often
the adjustments are applied. In living tissue the adjustment of connections is far slower
than the activity passing through them. Here the two happen at comparable rates, and it
is possible that the collapse is caused by that mismatch rather than by anything the
biology would recognise. If slowing the adjustments by a factor of one hundred leaves the
collapse unchanged, then the timescale reading is dead and should be recorded as such.

The analogy is worth abandoning entirely if it never generates a prediction that plain
mathematics did not already supply. That is the honest risk here and it is not small. The
argument in the section above on why collapse happens is a mathematical argument, and a
critic is entitled to say that everything predicted below follows from it without any
need for cultured neurons. The reply this document offers is the second prediction, where
the biology and the mathematics arrive independently at the same non obvious protocol,
and the fourth, where the biology supplies a prediction about density that the
mathematics does not. If both of those fail, the framing was decoration and should go.

## Where the analogy stops holding

An analogy pushed past its limit is a lie with good manners, so the limits are stated
here rather than left to be found.

The two systems measure different things. A cultured network is measured in spikes,
which are events in time, and its bursting is a temporal phenomenon: cells firing
together within milliseconds. The system here is measured in settled states, which are
the resting points of a deterministic process, with no time axis of that kind at all.
Calling both of them collapse is a claim about the shape of the failure, meaning many
possible behaviours reduced to one, and it is not a claim that the same physical process
is occurring. Anyone who reads it as the latter has been misled, and this paragraph
exists so that they cannot say they were not warned.

A cultured network is noisy, wet and variable between preparations. The findings above
rest on fifty eight cultures. This system is deterministic, and its results so far rest
on one input, one site and one seed, a limitation the register states about itself in
rows C-40 and C-41. Where the biology says an effect is real because it survived across
many preparations, nothing here has yet earned the same status, and the sample size
question is not solved by the analogy.

The adjustment rules used here were chosen because they are simple and local, not because
anyone established that cortical tissue uses them. The resemblance is at the level of
"local rule with no external instruction", and no further.

## What this document commits the project to

It does not commit the project to any new claim. No row of the register changes on
account of anything written here, and no measurement is reported here that was not
already committed elsewhere.

It commits the project to running EXP-003 and reporting the result whichever way it
comes out, including the outcome in which the framing is refuted and this file is
retired. The repository keeps retired material rather than deleting it, and this document
should be treated the same way.

## Sources

Verified during the preparation of this document. Where a passage is quoted, it was
quoted from the retrieved text rather than from memory. The status column is honest about
what that verification consisted of, following the precedent set by register row C-44,
which distinguishes a source an agent retrieved from a source a person has read.

| Source | Used for | Verification status |
|---|---|---|
| Wagenaar, Madhavan, Pine and Potter (2005), *Controlling bursting in cortical cultures with closed-loop multi-electrode stimulation*, Journal of Neuroscience 25(3):680-688 | The remedy, the crossing point between concentrated and spread out signals, the rate matched comparison, and that bursting resumed when stimulation stopped | Methods and results read from the authors' own reprint, whose abstract matches the published one. The publisher's site refused access, so this is a copy hosted by an author rather than the journal's version of record |
| Potter, review chapter for the 6th International Meeting on Substrate-Integrated Microelectrodes (2008) | The quoted stimulation parameters, the quoted claim that quieting bursts aids induction and detection, and the deafferentation framing | Passages quoted directly from retrieved text |
| Wagenaar, Pine and Potter (2006), *Searching for plasticity in dissociated cortical cultures on multi-electrode arrays*, Journal of Negative Results in Biomedicine 5:16, **together with its correction, Journal of Negative Results in Biomedicine 6:3 (2007)** | That the induction attempts largely failed, and that the one exception quieted bursts chemically rather than electrically | Both read. **The correction is essential and the uncorrected text is still publicly served.** It swaps two protocol labels throughout, with the effect that the successful protocol is the one using added magnesium, and instructs that a discussion paragraph crediting distributed electrical stimulation be dropped entirely. An earlier draft of this document relied on the uncorrected text and stated the opposite |
| Wagenaar, Pine and Potter (2006), *An extremely rich repertoire of bursting patterns during the development of cortical cultures*, BMC Neuroscience 7:11 | The developmental course, the fifty eight cultures, and that denser cultures burst earlier | Passages quoted from retrieved text |
| Chao, Bakkum and Potter (2007), Journal of Neural Engineering 4(3):294-308 | The centre of activity measurement and its trajectory, the shuffled position control, the change to drift ratio, the companion measure over connection strengths, and the practice of judging a statistic by the smallest true change it detects. EXP-003 adapts all five | Defining equations, the shuffle control and its reported sensitivities, and the change to drift definition all quoted from the retrieved reprint. This is the most heavily used source in the experiment design |
| Wagenaar, Nadasdy and Potter (2006), Physical Review E 73:051907 | Cited for context only, not relied upon for any claim here | Title, authors and journal retrieved. Contents not used |

**One gap, stated because the register's own convention requires it.** No person on this
project has opened any of these papers. Every passage above was retrieved and quoted by
an agent. Row C-44 of the register treats that as a real limitation rather than a
formality, and the same caveat applies here with more force, because this document rests
on the sources more heavily than that row does.
