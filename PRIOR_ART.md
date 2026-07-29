# Prior art

*Search run 2026-07-27. Extended, verified and review-corrected 2026-07-28/29.
What exists, in plain terms, and how close it gets.*

## The claim, stated at the strength the search supports

**No work matching this combination was found: a hand-written local rule, on a
pretrained frozen model, driven by a closed activation loop, with no task and no
loss.** Every ingredient separately is well studied; the combination did not appear in
what was searched.

That is still a statement about a search, not about the literature. **It is not "nobody
has done this."** It is now a considerably larger search: the forum gap and the
citation-graph gap have been closed and turned up nothing, and every entry below has been
verified against its source. The remaining coverage limits are recorded at the bottom and
they are still real. Treat the claim as provisional, but less provisional than it was.

## How the search was done

| | |
|---|---|
| Method | ~20 web searches by an automated agent (27 Jul), then a targeted verification and gap-closing pass (28 Jul), then two review passes correcting over-claims (28-29 Jul) |
| Dates | 2026-07-27, extended 2026-07-28, corrected 2026-07-29 |
| Databases | Open web, arXiv API, Semantic Scholar (metadata + forward citations), PubMed Central, GitHub code and repository search |
| Forums | LessWrong and the Alignment Forum searched directly through the site's own search API — 16 phrasings across posts, comments and wikitags. Plus general web and GitHub |
| Inclusion | Any work combining two or more of: local unsupervised rule, pretrained frozen model, closed activation loop, no objective |
| Verification | Every entry below carries a status, and all are verified against the source, most with a quote |

## Verification status

- **Daydreaming Hopfield networks — verified.** Title, authors, journal and substance
  confirmed. Full body also read, which corrected our description of their stabilisers
  (see the stabilisers section — the per-entry bound is called J_max, and it exists for a
  different reason than this file previously gave).
- **Chaudhary 2025 — verified, including the depth figures, but they say something
  different from what was recorded here.** See the correction below. This is the most
  important change in this revision.
- **Cazalets & Dambre — verified.** Identifier, journal, volume and the quoted claim all
  confirmed. One correction: the rule they compare against is *Anti-Oja*, not plain Oja.
- **Lee, McLeish, Goldstein & Fanti — identifier verified, our summary was wrong.**
  Corrected below.
- **Gong, Chen & Ching — verified** with quotes from the abstract.
- **Hopfield/Feinstein/Palmer 1983, Schlag et al. 2021, Irie et al. 2022, Shumailov et
  al. 2024, Lazar/Pipa/Triesch 2009 — verified**, identifiers and substance confirmed.
- **Zenke, Gerstner & Ganguli 2017 — verified, including the "10-20 seconds" figure.**
  An earlier revision of this file marked the number unsourced and guessed it belonged to
  a companion paper. That guess was wrong: the sentence is in this paper. Quoted in the
  stabilisers section.

---

## The correction that matters most

`PRIOR_ART.md` previously recorded three unverified figures from Chaudhary 2025 —
divergence at 8 layers, stability around 4 layers, deepest layers drifting most — and
warned that if they held, GPT-2 small at 12 layers would sit past the point where anyone
has reported stability.

**The figures are real. They are in the paper. But they are about the gradient-based
rule, not the Hebbian one.** The full passage, from Section 4.8 (*Task-Dependent
Behaviour and Depth Stress Test*), quoted in full:

> "Extending the copying task to 8-layer models exposes stability limits.
> Gradient-plastic Transformers diverge after ∼3000 steps (plastic norms >10²; recall
> below baseline), with the deepest layers showing the largest drift. Hebbian plasticity
> remains stable but saturates in performance (recall 0.729 ± 0.015). A practical regime
> therefore lies around 4 layers: deeper gradient-plastic stacks require additional
> regularization (e.g., gradient clipping or frozen upper layers) to prevent
> instability."

Read carefully, that sentence splits in two:

- **Divergence at 8 layers, and deepest-layers-drift-most: the gradient-plastic
  variant.** Not the Hebbian one.
- **The Hebbian variant at 8 layers: stable, but it stops doing anything useful.** It
  saturates.
- **"A practical regime around 4 layers" is explicitly qualified** — the very next clause
  says "deeper *gradient-plastic* stacks require additional regularization". It is not a
  general depth limit for plasticity.

**What this changes for us.** We run an Oja rule, which is in the Hebbian family, not the
gradient family. The one published depth stress test says that at 8 layers the Hebbian
side does *not* blow up — it goes quiet. So the expected failure mode this paper points
at is **saturation, not divergence.** That is the same direction the reservoir work
points (below): the most likely outcome is that very little happens.

**Two limits on how far this transfers.** Their deepest test is 8 layers; GPT-2 small is
12, so we are still past anything anyone has measured. And their models are trained from
scratch with the plastic module in the loop, so "stable" there means stable during
training, not stable in a closed activation loop on a frozen pretrained network. This
is a hint about which way to expect things to fail, not a prediction.

The earlier note in this file — "if the depth figures hold, they matter" — was right that
they matter. It was wrong about which way.

---

## Two claims this repo has made that are wrong

Recorded here so they do not reach a write-up.

**1. "Iterating a frozen model is unexplored" — false.** It has been done in
autoencoders and in recent looped / latent-reasoning language-model work, and some of
that literature reports orbit-like trajectories. There is now a direct example:
*Training-Free Looped Transformers* (arXiv:2605.23872, 2026) loops a contiguous block of
layers of a **frozen pretrained checkpoint** at inference time with no fine-tuning at
all, across seven model families. **Verified** — abstract fetched. It confirms the point
squarely: iterating a frozen pretrained model is an active, published area. It also
reports that naive re-application of a block usually makes things worse, which is worth
knowing. The parent project's *specific* protocol and results may well be new; the
general idea is not.

**2. "Frozen weights is just self-training with the learning rate at zero" — false.**
Gradient descent on a loss and a local correlation rule are different processes, not one
process at two gains. Turning eta to zero recovers frozen weights, but there is no
continuous path from Oja to backpropagation. The defensible framing is two separate axes
-- *what kind of rule* and *how strong* -- with the local-rule axis unswept at every
strength.

---

## The four closest pieces of work

### Daydreaming Hopfield networks

Serricchio, Bocchi, Chilin, Marino, Negri, Cammarota & Ricci-Tersenghi.
*Daydreaming Hopfield Networks and their surprising effectiveness on correlated data.*
arXiv:2405.08777 · Neural Networks 186, 107216 (2025). **Verified** — arXiv record,
journal reference and author list all confirmed.

**What they were trying to do.** Fix a known problem with associative memory networks:
they invent false memories -- stable states nobody stored. The idea was to let the
network dream. Run it until it settles into a state on its own, then adjust the weights
to erase whatever it just settled into, while reinforcing the real stored patterns.

**What they found.** It works. Spurious memories are erased, real ones end up with
larger basins of attraction, and storage capacity improves -- notably on correlated
data. The procedure converges to a stationary retrieval map without destroying what was
stored.

**How close.** Closest mechanism in existence: the weight update is driven by a state
the network generated itself, with no gradients and no loss. **Not us** because it is a
Hopfield network rather than a transformer, and the update still refers to stored
patterns -- there is a target, just not a loss function.

Its ancestor -- Hopfield, Feinstein & Palmer, *'Unlearning' has a stabilizing effect in
collective memories*, Nature 304:158-159 (1983) -- **is** genuinely target-free: run the
network, see where it lands, weaken that. **Verified** — Nature record and page range
confirmed. 43 years old, and the true precedent.

**Forward citations checked.** 16 papers cite the Daydreaming work. Every one of them
stays inside the Hopfield / statistical-physics world -- unlearning analyses, dreaming
with bounded synapses, dense associative memory, biased patterns. **Not one applies the
idea to a transformer, pretrained or otherwise.** That is a meaningful negative result:
the most obvious bridge from this literature to ours has not been walked.

### Hebbian and gradient-based plasticity in transformers

Siddharth Chaudhary. *Enabling Robust In-Context Memory and Rapid Task Adaptation in
Transformers with Hebbian and Gradient-Based Plasticity.* arXiv:2510.21908 (2025).
**Verified** — abstract and full HTML body fetched and read.

**What they were trying to do.** Give transformers a fast-changing memory that updates
during use rather than during training, and compare a Hebbian version against a
gradient-based one.

**What they found.** From the abstract: "Hebbian plasticity consistently achieves lower
loss and stronger few-shot generalization, while gradient-based updates perform best on
long-horizon credit assignment." Also, usefully: "When associations are short and
linearly separable, static weights suffice, defining a clear boundary condition for when
plasticity helps."

**The depth results.** Real, and quoted in full in the correction section above. In
short: the 8-layer divergence and the deepest-layer drift belong to the **gradient**
rule; the **Hebbian** rule stayed stable at 8 layers and saturated instead; and the
"practical regime around 4 layers" is stated about gradient-plastic stacks specifically.

**Scale.** Their main experiments use **2-layer** transformers (d_model 128-256, 4
heads). The depth stress test goes to 8. GPT-2 small is 12.

**How close.** The nearest thing anyone will cite at us. **Not us** because the
plasticity rule is *learned by gradient descent on a task*, the models are trained from
scratch rather than pretrained and frozen, and updates are driven by external data
rather than a closed loop.

**Forward citations checked.** Exactly one paper cites it: *Where to Bind Matters:
Hebbian Fast Weights in Vision Transformers for Few-Shot Character Recognition*
(arXiv:2605.02920, 2026). **Verified** — abstract fetched. Vision transformers, Omniglot,
a meta-learning loss. Its one transferable finding is that *where* you put the Hebbian
module matters a lot, and that putting a separate Hebbian module at every stage caused
training instability while a single module at the last stage did not. Same shape as
Chaudhary's depth result, from a different direction: spread the plasticity through the
depth and it goes wrong; concentrate it near the end and it does not.

### Reshaping reservoirs with unsupervised Hebbian adaptation

Cazalets & Dambre. *Reshaping reservoirs with unsupervised Hebbian adaptation.* Nature
Communications 17, 450 (2026). DOI 10.1038/s41467-025-67137-1. **Verified** —
identifier, volume, article number and the quoted claim all confirmed against the open
full text.

**What they were trying to do.** Take a randomly wired recurrent network, improve it
with local unsupervised rules and no gradient steps, then test it on downstream tasks.
Their own rule is called HAG (Hebbian Architecture Generation), and it *grows*
connections between neurons that co-activate rather than just reweighting them.

**What they found.** Their own rule helped, a lot. The baseline rules did not. Verbatim:

> "Networks trained only with Intrinsic Plasticity or Anti-Oja rules never surpass HAG
> and seldom exceed even the static ESN; short-term weight updates do not reorganize the
> latent space as effectively as the long-horizon Hebbian growth employed by HAG."

**Two corrections to what this file said before.** The rule they benchmark is
**Anti-Oja**, not plain Oja — Oja's rule with the sign flipped. And that sentence sits in
their analysis of *latent-space separability*, not raw task accuracy; in the accuracy
tables Anti-Oja does occasionally edge past a static reservoir. The finding is real, it
is just narrower than "Oja does nothing".

**How close.** Not close in substrate, and this is worth being strict about. A result
about tanh units in a randomly wired reservoir **cannot establish anything** about what
an Oja rule does to a pretrained transformer in a closed activation loop. Different
units, different connectivity, different input regime, different rule sign.

What it is good for is **motivating a null hypothesis**, not supplying evidence for one:

> **H₀: the closed-loop arm is indistinguishable from the offline arm.** Local
> unsupervised weight updates move the network somewhere, but the feedback coupling adds
> nothing on top of what the same rule does to a fixed recording of the same
> activations.

Cazalets & Dambre make that null *plausible* rather than *supported* — two independent
lines of work (theirs, and Chaudhary's Hebbian-at-8-layers saturation) both found that
local rules on their own tend to do less than expected. Neither is about our substrate.
Stating the null this way is useful because it is the thing our experiment is actually
built to reject.

**What would test it.** The offline control specified later in this file, run as a
**paired, pre-specified comparison**. An earlier revision of this file said "reject H₀ if
the between-arm difference exceeds the within-arm seed spread". That is a heuristic, not a
test — it pools the arms independently, has no stated uncertainty, and has no rule for
what counts as *no* effect. Replaced with:

1. **Run the arms matched and paired.** For each seed s, run both arms from the same
   initial weight and the same seed, matched on every axis except feedback (see the table
   below). This gives one **paired difference** per seed, d(s) = closed-loop(s) −
   offline(s). Paired per-run differences, not two independent pools — the seeds are a
   blocking factor, and pooling throws that away.
2. **Pre-specify one primary metric** before looking at any result, and say so in writing.
   Everything else is secondary and reported as exploratory. The natural primary is the
   relative Frobenius distance between the two arms' final weight matrices; whatever is
   chosen, it is chosen first. Any secondary metric describing the loop's attractor
   structure must be **period-aware** — detected period k and the residual at that period,
   plus basin sizes defined over periodic orbits, not over fixed points alone (see
   "Convergence checks must allow for periodic orbits"). A metric that can only see fixed
   points would score this project's own `Divine` result as a non-convergence.
3. **Report an uncertainty interval on the mean paired difference**, from a permutation
   test over seeds (randomly flip the sign of each d(s); the null distribution is the
   distribution of the mean under exchangeability) or a bootstrap over seeds. Report the
   interval, not just a point estimate, and not just a p-value.
4. **Check it against the harness noise floor first.** This harness's floor for
   arm-to-arm agreement is **below 1e-8 relative Frobenius** — measured at **2.898e-09**
   on this box and **4.757e-09** on the CI runner. A difference smaller than that bound is
   numerical noise and **is not evidence of anything**, no matter what a significance test
   says about it. The noise floor gates the test; it does not compete with it.
5. **To claim the arms are the same, state an equivalence margin.** Pick a margin δ that
   would be too small to matter scientifically, decided in advance, and show the
   uncertainty interval on the paired difference lies entirely inside ±δ. Without that,
   the only honest phrasing for a null result is **"no detectable effect at this sample
   size"**. Never "the null was accepted", never "the arms are identical".

This also fixes the direction of inference. The reservoir work does not tell us what we
will find; it tells us to design for the possibility that the answer is small, and to be
able to tell "nothing happened" apart from "we could not tell".

**Forward citations checked.** One citing paper, on free-energy-principle / neural
manifold theory (arXiv:2605.04200). Nothing near us.

### Do language models need sleep?

Lee, McLeish, Goldstein & Fanti. *Do Language Models Need Sleep? Offline Recurrence for
Improved Online Inference.* arXiv:2605.26099 (submitted 25 May 2026). **Identifier and
authors verified; our previous summary of it was wrong and is corrected here.**

**What they were trying to do.** Attention gets expensive as context grows. So: let the
model periodically "sleep" -- pause, fold the recent context down into persistent fast
weights, and throw away the key-value cache. During sleep the model makes N offline
recurrent passes over the accumulated context and updates fast weights in its
state-space-model blocks through a learned local rule.

**What they found.** It works on the tasks measured -- cellular automata, multi-hop graph
retrieval, and a maths reasoning task where a plain transformer and hybrid models fail.
Longer sleep (larger N) gives better results, with the biggest gains where deeper
reasoning is needed.

**What we had wrong.** This file previously said the model "runs forward with no external
input at all". It does not. The offline phase runs recurrent passes **over the
accumulated context** -- there is input, it is just internal rather than newly arriving.
It is also an SSM-block architecture, not a plain transformer, and the whole point is
inference latency, not spontaneous dynamics.

**How close.** Still the closest *loop shape* in the literature: an offline phase, no new
tokens arriving, weights changing. **Not us** because the rule is learned rather than
hand-written, the model is trained for the procedure, the offline pass is driven by
stored context rather than by the model's own free-running activations, and success is
defined by downstream task scores.

**Forward citations checked.** Zero citations as of 2026-07-28. Too recent.

---

## Is anyone else doing this? The collision search

**No collision found.** This is the gap that mattered most, and it is now properly
searched rather than barely searched. What was done:

- **LessWrong and the Alignment Forum**, through the site's own search endpoint rather
  than a general web engine — 16 phrasings, covering posts, comments and wikitags:
  Hebbian frozen transformer; Oja rule language model; activation feedback loop GPT-2;
  iterated forward pass fixed point; self-generated activations weight update; plasticity
  frozen LLM no loss; residual stream attractor plasticity; recurrent injection weights
  change; local learning rule no gradient transformer; weights change at inference no
  backprop; feeding model output back into itself; online weight updates during
  inference; self-referential loop weights drift; and others.
- **The single sharpest result:** the word **"Oja" appears in exactly two LessWrong posts
  and one comment, in the entire site history.** The two posts are Nellessen & Jan's
  *[Hebbian Natural Abstractions]* sequence (2022), which uses Oja-with-decay to argue
  that biological brains extract principal components — a theory argument about brains,
  with no transformer and no experiment. The third hit is an unrelated 2015 media-thread
  comment. Nobody on LessWrong has run this.
- **GitHub**, repository and code search, for Hebbian/Oja plasticity applied to GPT-2 or
  frozen pretrained transformers. Nothing. The repository hits are memory systems for
  agents, spiking-network paper lists, and continual-learning reading lists — none of
  them apply a local rule to a frozen pretrained model's weights.
- **arXiv full API**, several term combinations, sorted newest first.
- **General web**, including searches aimed specifically at blog posts, notebooks and
  weekend writeups rather than papers.

If someone has done this, it is not indexed anywhere the above would reach. That is not
proof of absence, but it is a real search now rather than a gesture at one.

---

## One published claim we can test cheaply

Gong, Chen & Ching, *Strong anti-Hebbian plasticity alters the convexity of network
attractor landscapes*, arXiv:2312.14896 (2023). **Verified** — title, authors and the
quoted claims confirmed against the abstract.

They report that moving from Hebbian to anti-Hebbian learning produces "a pitchfork
bifurcation that destroys convexity in the network attractor landscape", and --
counterintuitively -- that "attractor landscapes are more sensitive to slower learning
rates than faster ones."

Our eta sweep can test whether that trend **transfers to this substrate**. It cannot
falsify their result: a different model, different metrics and different assumptions
mean a non-replication here says nothing about their setting. Report it as "replicates"
or "does not reproduce under this setup", never as falsification.

*Forward citations checked:* one, a bioRxiv spiking-network paper. Nothing near us.

---

## The finding that changes our experiment

An earlier version of this file said flatly that **Oja's rule converges to the dominant
eigenvector of the second-moment matrix E[x xᵀ] of whatever activations flow through
it.** That is a theorem, and like every theorem it has preconditions. Ours are not met in
both arms of the experiment, and the difference between the two arms is exactly where the
preconditions break. Getting this right sharpens the case for the offline control rather
than weakening anything.

### Lead with the property we actually rely on

Before any theory, here is the distinction the whole experiment rests on, and it needs no
theorem at all:

> **In the offline arm the input sequence is unaffected by the weight update. In the
> closed loop it is not.**

That is a structural fact about the two arms, it is true by how they are built, and it is
the only property the comparison requires. Everything below is about what we may and may
not additionally claim on top of it. **Neither arm gets a convergence guarantee.**

### What the theorem actually requires

Oja's convergence result is a stochastic-approximation result. It needs at least:

1. **A stationary stochastic input process** with a well-defined second-moment matrix —
   samples drawn from a fixed distribution that does not change as the weights change.
2. **A decaying step size** satisfying the usual Robbins–Monro conditions: the step sizes
   must sum to infinity (so the process can travel any distance) while their squares sum
   to something finite (so the noise averages away). A constant eta fails the second.
3. **A simple dominant eigenvalue** — Oja's original analysis assumes the largest
   eigenvalue of the second-moment matrix is non-degenerate. If the top two eigenvalues
   are equal or near-equal, the "dominant eigenvector" is not unique and the rule has no
   single direction to converge to. This file previously recorded it as never having been
   checked. A first check is now in hand, and it is only a first check:

   | Matrix, ordinary text at the default site | λ₁ | λ₂ | λ₂/λ₁ |
   |---|---|---|---|
   | Second moment E[x xᵀ] | 21.749 | 7.436 | **0.342** |
   | Covariance | 7.515 | 4.804 | **0.639** |

   On the raw second moment the gap is comfortable, so the precondition looks satisfied
   for that matrix on ordinary input. **On the centred matrix the gap is much weaker**,
   which is a second reason centring is not a free choice. **Still unverified for the
   closed loop**, where the matrix moves as the weights move and the gap can close at any
   point — a spectral gap measured once at the start says nothing about step 5000. Track
   λ₂/λ₁ across the run rather than assuming it.

Under those conditions the weight vector converges to the dominant eigenvector of
E[x xᵀ]. That is *ordinary PCA* only when the activations are zero-mean; on non-centred
inputs the mean itself contributes and the result is biased relative to covariance PCA.

### The non-zero-mean claim, measured

This file previously asserted "post-GELU activation is not zero-mean" without a number.
It is now measured, on this box, at the default site — `blocks.6.mlp.hook_post`, which is
the x that Oja consumes at `blocks.6.mlp.W_out`. GPT-2 small, float32, weights frozen.

| | Ordinary text (107 positions, 3 prompts) | `Divine` state (10 positions, renormalised to ‖x₀‖ = 1468.489) |
|---|---|---|
| Scalar mean of x | **−0.047466** | **−0.025227** |
| Std of x | 0.182791 | 0.159973 |
| Mean / std | −0.2597 | −0.1577 |
| Fraction of entries negative | 0.8493 | 0.8229 |
| min / max | −0.1700 / +3.9514 | −0.1700 / +2.5222 |
| ‖E[x]‖ | 4.4969 | 8.9760 |
| ‖E[x]E[x]ᵀ‖_F ⁄ ‖E[x xᵀ]‖_F | **0.7878** | **1.0000** |
| ‖cov‖_F ⁄ ‖E[x xᵀ]‖_F | 0.5468 | **0.0000** |

The claim holds and is not marginal. The mean is negative, roughly a quarter of a standard
deviation, and **85% of post-GELU entries are negative** — GELU's floor at −0.1700 shows up
exactly as expected. So **Oja here targets the raw second moment, not the covariance**,
unless centring is explicitly applied. Describe it that way.

**Two things worth more than the headline number.** First, on ordinary text the mean term
carries **79% of the second-moment matrix's Frobenius norm**. Centring is not a
refinement; it changes most of the matrix.

Second, and this one is operational: **on the `Divine` state the covariance is
numerically zero.** The `Divine` attractor is position-uniform — every token position
holds nearly the same vector — so E[x xᵀ] is effectively rank-1 and equal to E[x]E[x]ᵀ.
**If centring is ever switched on while running from `Divine`, Oja is left with nothing to
work with.** That is not a subtlety, it is the difference between a rank-1 matrix and an
empty one, and it makes the "centring, or the deliberate absence of it" row of the
offline-control table load-bearing rather than pedantic.

*Status: measured on this box at one site and one layer, over three ordinary prompts plus
the committed `Divine` state. Not swept across sites, layers or seeds — treat the exact
figures as indicative and the sign and order of magnitude as solid.*

### The offline arm: a fixed-input replay baseline, not a converging one

The offline arm replays a **fixed, finite, deterministic recording**. An earlier revision
of this file called that "stationary by construction" and concluded the theorem applies.
**That was wrong.** A fixed recording is reproducible, but it is not a stationary
stochastic process with a well-defined covariance — it is one finite sample path. Combined
with a constant eta and finitely many updates, the offline arm does **not** earn the
convergence result either.

What the offline arm is, precisely, is an **empirical fixed-input replay baseline**: the
arm in which the activation sequence driving the update does not respond to the update.
That is a much weaker claim than convergence, and it is the only claim the comparison
needs.

**To earn any convergence language for this arm we would have to add assumptions and
measure them**: treat the recording as a sample from some notional stationary source,
check that the empirical second-moment matrix has a simple dominant eigenvalue with a
usable spectral gap, and then *show* the weight vector settling — its direction stabilising
across replay passes, to a tolerance, reproducibly across seeds. Until that is done, say
"fixed-input replay", not "converges to the dominant eigenvector".

### The closed-loop arm: further still from any theorem

The closed loop breaks the same preconditions, and breaks the first one *by construction*:

- **The activations are non-stationary on purpose.** They change *because* the weights
  change. That is not a nuisance in our setup, it is the entire object of study. The input
  distribution at step t depends on the weights at step t, which depend on the activations
  at step t−1. There is no fixed E[x xᵀ] for the rule to find, not even a notional one.
- **Fixed step size, finite number of updates.** No decaying schedule, so no Robbins–Monro
  guarantee.
- **A norm ceiling that clips.** Once clipping is active the update is no longer Oja's
  rule; it is Oja's rule composed with a projection. Convergence results for the
  unprojected rule say nothing about it.

So for the closed-loop arm, **convergence is not guaranteed and must not be assumed.**
The correct description until we show otherwise is **coupling-induced drift**: the weights
move, the activations move because the weights moved, and where that lands is an empirical
question. It could converge, settle onto a periodic orbit, wander, or saturate. We do not
get to assert any of those from theory.

**If we want to claim the closed-loop arm converged, we have to demonstrate it
empirically** — and the demonstration has to allow for periodic orbits, not just fixed
points (see below). A single run that stops changing is not a demonstration of
convergence.

### Why this makes the offline control essential rather than optional

The point that survives unchanged is this: **the weight matrix will move and the
attractors will shift with no feedback whatsoever.** That is simply what the rule does to
any activation stream. So a closed-loop result proves nothing on its own -- our claim is
about the *coupling*, weights changing while the thing they are changing feeds back into
them.

The two arms are cleanly separated by one property and one only: whether the input
sequence responds to the update. Neither arm comes with a predicted endpoint. **The
difference between the two arms is the only place our claim can live**, and it has to be
established by measurement rather than inherited from theory.

### Convergence checks must allow for periodic orbits

This repo's own headline finding is a **period-2 limit cycle** — the parent project's
`Divine` basin, measured as near period-2 recurrence on the committed `state_divine.pt`.
So "did it converge?" cannot be answered by testing whether successive states are equal.

**A convergence check that only looks for fixed points will score a stable period-2 orbit
as non-convergence.** That is the exact error to avoid, and it has already bitten this
project once in the form of an even-only snapshot schedule that aliased a period-2 orbit
into a fixed point.

Every convergence and settling check in both arms must therefore be a **lag scan, not a
lag-1 equality test**:

- Test recurrence at lag k for k = 1 … K, with a small bound (K = 8 is ample; K ≥ 2 is
  mandatory), against a stated tolerance rather than exact equality.
- Report the **smallest** k that meets tolerance as the detected period. k = 1 is a fixed
  point; k = 2 is the `Divine` case; no k meeting tolerance within K is "no periodic
  attractor detected at period ≤ K", which is *not* the same as divergence.
- Report the residual at the detected period alongside k. A period found at loose
  tolerance and one found at tight tolerance are different results.
- Sample at a stride coprime with the periods being tested, or at every step. An even-only
  schedule cannot see period 2.

The same applies to the weight matrix itself: it may settle onto a cycle rather than a
point, and the check for that is the same lag scan.

### The offline control, specified

Record the activations from the frozen loop, run the same rule over that recording with
no feedback, install the resulting matrix, re-run the loop frozen, and compare against
the closed-loop run. **The claim lives entirely in the difference between those two.**

For the comparison to mean anything, the offline arm must match the closed-loop arm on
every axis except feedback:

| Must match | Why |
|---|---|
| eta, and the ceiling | Different step sizes give different drift, not different mechanisms |
| Total number of weight updates | The arms must travel the same number of steps |
| Order of the activation samples | Oja is sequential; a reshuffled replay is a different trajectory |
| Batching of samples per update | Averaging over a different batch changes the update |
| Initial weight, and RNG seed | Same starting point, same draws |
| Centring, or the deliberate absence of it | Applied to one arm and not the other, this alone changes what the rule targets — and at the `Divine` state, centring leaves a numerically empty matrix (measured above) |
| Period-detection settings: K, tolerance, sampling stride | A period-2 orbit read with an even-only stride looks like a fixed point. Differing schedules make the arms incomparable |

Record all of these alongside the result. If the two arms differ on any of them, the
difference between them is not evidence about feedback.

**Basins must be defined over attractors, not over fixed points.** The measurement here is
"which attractor does this initial state fall into, and how large is its basin" — and an
attractor may be a fixed point (period 1) or a periodic orbit (period k ≥ 2). Concretely:

- Run the loop from each initial state, then classify the endpoint by the lag scan above:
  detected period k plus the residual at that period.
- Two initial states are **in the same basin** if they land on the same orbit — matched as
  a *set* of states up to cyclic rotation and tolerance, not by comparing single snapshots.
  Comparing one snapshot to one snapshot will split a single period-2 basin into two
  phantom basins, or merge two genuinely different ones, depending on the phase each run
  happened to be sampled at.
- Basin size is then the fraction of initial states landing on that orbit. Report basin
  sizes with the period alongside, so a change from "period 2, basin 0.4" to "period 1,
  basin 0.4" is visible as the qualitative change it is.

This matters concretely for us: the headline comparison is whether the `Divine` period-2
cycle survives plasticity, damps to a fixed point, lengthens, or breaks up. **All four
outcomes are indistinguishable to a fixed-point-only basin measurement**, and three of the
four would be misreported as "did not converge".

---

## Two framings we inherit whether we like them or not

**Linear attention is already an outer-product Hebbian update.** Schlag, Irie &
Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021
(arXiv:2102.11174); Irie, Csordás & Schmidhuber, *The Dual Form of Neural Networks
Revisited*, ICML 2022. **Verified** — both venues and titles confirmed. If we ever drift
an attention matrix we are formally running a fast-weight programmer whose keys and
values are self-generated. That literature's analysis and its known interference failure
modes come with it, and we would have to say why ours differs.

**The model-collapse literature is our high-gain neighbour.** Shumailov, Shumaylov, Zhao,
Papernot, Anderson & Gal, *AI models collapse when trained on recursively generated
data*, Nature 631:755-759 (2024). **Verified** — volume and page range confirmed; note
there is also a 2025 author correction on record. Models retrained on their own output
degrade in a documented way: early collapse where distributional errors accumulate and
the model drifts from the true distribution, late collapse where low-frequency events
disappear permanently and the distribution narrows toward a point.

Ours is the same *shape* -- a system consuming its own output -- at the activation level,
on a timescale of seconds instead of generations. But **their measurements do not
transfer automatically**, and it would be sloppy to say they do. They retrain a model
from scratch each generation on a finite sample of text generated by the previous one;
their damage comes from **sampling error compounding across generations**. We update one
model's weights in place from its own activation stream, with no sampling of a dataset,
no retraining, and no generational boundary. Given a seed our drift is deterministic;
theirs is not.

What transfers is the **shape of the diagnostics**, as candidates to try:

| Their finding | Our analogue | How to measure it |
|---|---|---|
| Early collapse: tails of the distribution are lost first | The loop's activation distribution narrows as weights drift | Track variance and entropy of the activation distribution at the update site, per loop step, closed-loop arm vs offline arm |
| Late collapse: low-frequency events vanish permanently | Rare tokens stop being reachable from the drifted model | Compare next-token probability mass assigned to rare tokens, drifted model vs frozen model, on a fixed held-out prompt set |
| Drift away from the true distribution | Drift away from the frozen model's own behaviour | Perplexity of the drifted model on a held-out real corpus, against the frozen model as the reference point |
| Collapse toward a single low-variance mode | The loop falls into one attractor and stops exploring | Count distinct fixed points reached from a fixed set of initial states, and their basin sizes, before and after drift |

Each of these is a **candidate diagnostic borrowed by analogy**, not an imported result.
None of their quantitative rates or their generation-count axis carries over, and a
finding of ours should never be reported as replicating or contradicting theirs.

---

## Stabilisers other people needed

Nobody in the surveyed work got away with a single mechanism.

- **Daydreaming Hopfield: verified from the paper body**, and the earlier description
  here was imprecise. Two separate mechanisms:
  - **Periodic L2 normalisation of the whole coupling matrix.** The update rule is
    invariant under global rescaling of J, but they normalise anyway — "we prefer to keep
    it well bounded, and so we normalize it every N steps", implemented in their
    pseudocode as `J_ij ← J_ij / ||J||₂` once per epoch.
  - **A per-entry magnitude threshold, J_max**, on the absolute value any single coupling
    may take. Verbatim: "We have solved this problem by just introducing a threshold
    J_max on the maximum absolute value that a single coupling can assume."
  - **The reason for J_max is more interesting than "high load", which is what this file
    said before.** On strongly correlated real data (MNIST, where background pixels are
    perfectly correlated) the algorithm correctly tries to drive those couplings to
    infinity. Then the global normalisation step divides everything by that huge norm and
    **every other entry vanishes, erasing the information stored in the matrix.** J_max
    exists to stop one runaway entry from wiping out the rest.
  - Load enters separately: with J_max in place, they had to change *how* they normalise
    so that the norm of J does not depend on the load α.
  - **Directly relevant to us.** We have both mechanisms in the ATR loop — a norm ceiling
    and a rescaling — and this is a documented case of the two interacting badly. Worth
    checking whether a small number of entries in our drifted matrix are absorbing the
    ceiling and flattening everything else.
- **The reservoir work (Cazalets & Dambre): verified from the paper.** Mean-HAG holds a
  target mean firing rate with a permissible deviation band. Variance-HAG holds a target
  standard deviation *and* adds an explicit safeguard: if any neuron's state exceeds a
  saturation threshold, its synaptic weights are scaled down by a fixed factor. They say
  plainly that this "keeps the network in a balanced regime, promotes stability in
  practice and prevents blow-up", and that "formal convergence is not guaranteed".
- **Self-organising recurrent networks** (Lazar, Pipa & Triesch, *SORN: a self-organizing
  recurrent neural network*, Frontiers in Computational Neuroscience, 2009 —
  **verified**): three mechanisms at once -- spike-timing plasticity, intrinsic plasticity
  and synaptic normalisation -- to keep the dynamics "in a healthy regime suitable for
  learning". Remove any one and the healthy regime degrades.
- **Zenke, Gerstner & Ganguli**, *The temporal paradox of Hebbian learning and homeostatic
  plasticity*, Current Opinion in Neurobiology 43:166-176 (2017). **Verified from the
  published text**, including the number this file previously could not source. Verbatim:

  > "Because of these slow stabilization dynamics, the fast interplay between Hebbian
  > plasticity and recurrent network dynamics leads to rapid population firing rate
  > destabilization within 10–20 s for both learning rules."

  And the separation that creates the paradox, also verbatim: "forms of Hebbian plasticity
  can be induced on the timescale of seconds to minutes, whilst most forms of homeostatic
  synaptic plasticity operate over hours or days." Modelling studies that tried to
  stabilise Hebbian learning with homeostasis "were typically required to speed up
  homeostatic plasticity to timescales that are orders of magnitude faster than those
  observed in experiments". The conclusion is that compensatory mechanisms "must act on
  similar or even faster timescales than Hebbian plasticity itself" — necessary, not
  optional.

  **Correction to the previous revision of this file:** the "10-20 seconds" figure was
  marked unsourced and speculatively attributed to the companion paper (Zenke & Gerstner,
  Phil. Trans. R. Soc. B 372:20160259). That was wrong. The sentence is in the Current
  Opinion paper, and it is safe to quote with that attribution.

**Where we stand:** the activation rescaling in the ATR loop is already a fast homeostat,
but it acts on activations, not weights. Oja's decay term is a weight-side one. Whether
those two suffice is open, and the honest answer is that everyone else needed more.

---

## Published step sizes: still nothing

**Searched again. Still could not find any published learning rate for an Oja-family rule
inside a pretrained transformer, because the experiment does not appear to exist.** That
now stands as a finding rather than a gap in the search.

The closest published number is Cazalets & Dambre's Anti-Oja learning rate, η_oja, which
they sweep as a hyperparameter for an echo state network. Even that is not usable
directly: the numeric grid lives in their Appendix B.1, which is in supplementary
material we could not retrieve, and it is a reservoir of tanh units, not a post-GELU
site inside a 12-layer pretrained transformer. It would not transfer.

Our sweep would be the first data point -- a good sign for the gap, a bad sign for the
compute budget.

---

## Coverage gaps

These are the reasons the verdict at the top is still provisional. It is a shorter list
than it was.

**Closed since the last revision:**

- ~~Blogs, LessWrong, the Alignment Forum: one query only.~~ **Closed.** Searched
  directly through the forums' own search endpoint, 16 phrasings, posts and comments.
  Nothing found. "Oja" appears in two posts and one comment site-wide.
- ~~No citation-graph sweep.~~ **Closed.** Forward citations traced through Semantic
  Scholar for all four closest entries plus Gong et al. Daydreaming has 16 citations, all
  in statistical physics; Chaudhary has 1, in vision transformers; Cazalets & Dambre has
  1, in theoretical neuroscience; Lee et al. has 0. No citing paper is closer to us than
  the paper it cites.
- ~~Most entries unverified.~~ **Closed.** Every entry now carries a status, and all are
  verified against the source.
- ~~The Chaudhary depth figures.~~ **Closed, and they say something different from what
  was recorded.** See the correction near the top of this file.
- ~~The "10-20 seconds" homeostasis figure is not pinned to a source.~~ **Closed.** It is
  in Zenke, Gerstner & Ganguli 2017, quoted verbatim in the stabilisers section. The
  earlier guess that it belonged to the companion paper was wrong.
- ~~"Post-GELU activation is not zero-mean" asserted without a number.~~ **Closed.**
  Measured at the default site: mean −0.047 on ordinary text, 85% of entries negative, and
  the mean term carrying 79% of the second-moment matrix's Frobenius norm. Table in "The
  non-zero-mean claim, measured".

**Still open:**

- **Nothing has been tested past 8 layers.** Chaudhary's stress test stops there; GPT-2
  small is 12. Whatever the Hebbian side does at 12 layers in a closed loop on frozen
  pretrained weights is unmeasured by anyone, including in the direction of "quietly does
  nothing".
- **Non-English and non-indexed venues were not searched**, nor were workshop
  proceedings that do not appear on arXiv, closed Discord/Slack research communities, or
  university theses. A collision could still live in any of those.
- **Semantic Scholar's citation graph lags.** Anything published in the last few weeks
  that cites the four closest papers will not have shown up. Worth re-running the forward
  sweep immediately before any write-up.
- **No published step size exists to anchor our eta sweep.** Not a search gap any more —
  a real one. See the section above.
- **Neither arm has a convergence guarantee, and neither has been shown to converge.**
  The closed loop breaks stationarity by construction, and the offline arm — despite
  replaying a fixed recording — is a single finite sample path under constant eta, which
  does not satisfy the theorem either. Both are empirical questions. See "The finding that
  changes our experiment".
- **The simple-dominant-eigenvalue precondition is checked only once, on ordinary text.**
  λ₂/λ₁ = 0.342 on the raw second moment (comfortable) and 0.639 centred (weak). Unchecked
  inside the closed loop, where the matrix moves and the gap can close mid-run. Track it
  across the run.
- **The post-GELU measurements are one site, one layer, one box.** `blocks.6.mlp`, three
  ordinary prompts plus the `Divine` state, no sweep over sites, layers or seeds. Sign and
  order of magnitude are solid; the exact figures are indicative.
- **We have not checked whether our norm ceiling and our rescaling interact badly.** The
  Daydreaming authors hit exactly that failure — a few runaway entries plus global
  normalisation flattening everything else — and needed a per-entry bound (J_max) to fix
  it. Whether our matrix does the same thing is unmeasured.
- **No equivalence margin has been chosen.** Until someone picks a δ in advance, this
  experiment can report "no detectable effect at this sample size" but cannot report that
  the two arms are the same.
