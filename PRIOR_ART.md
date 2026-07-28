# Prior art

*Search run 2026-07-27. Extended and verified 2026-07-28. What exists, in plain
terms, and how close it gets.*

## The claim, stated at the strength the search supports

**No work matching this combination was found: a hand-written local rule, on a
pretrained frozen model, driven by a closed activation loop, with no task and no
loss.** Every ingredient separately is well studied; the combination did not appear in
what was searched.

That is still a statement about a search, not about the literature. **It is not "nobody
has done this."** It is now a considerably larger search: the forum gap and the
citation-graph gap have been closed and turned up nothing, and most entries below have
been verified against their sources. The remaining coverage limits are recorded at the
bottom and they are still real. Treat the claim as provisional, but less provisional
than it was.

## How the search was done

| | |
|---|---|
| Method | ~20 web searches by an automated agent (27 Jul), then a targeted verification and gap-closing pass (28 Jul) |
| Dates | 2026-07-27, extended 2026-07-28 |
| Databases | Open web, arXiv API, Semantic Scholar (metadata + forward citations), PubMed Central, GitHub code and repository search |
| Forums | LessWrong and the Alignment Forum searched directly through the site's own search API — 16 phrasings across posts, comments and wikitags. Plus general web and GitHub |
| Inclusion | Any work combining two or more of: local unsupervised rule, pretrained frozen model, closed activation loop, no objective |
| Verification | Every entry below now carries a status. Most are verified against the source with a quote |

## Verification status

- **Daydreaming Hopfield networks — verified.** Title, authors, journal and substance
  confirmed.
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
- **Zenke, Gerstner & Ganguli 2017 — partly verified.** The paper and its argument are
  confirmed; the specific "10-20 seconds" number may belong to the companion paper. See
  the note in the stabilisers section.

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
two-arm comparison with a noise floor:

1. Run both arms matched on every axis except feedback (see the table below).
2. Repeat each arm across seeds. The seed-to-seed spread **within** an arm is the noise
   floor.
3. Measure the **between-arm** difference on the same quantities — distance between the
   final weight matrices, and the change in the loop's attractor structure (which fixed
   points the loop reaches, and the size of their basins).
4. **Reject H₀ only if the between-arm difference exceeds the within-arm seed spread.**
   If it does not, the honest report is "no detectable effect of coupling at this eta and
   this number of steps" — which is a result, not a failure.

This also fixes the direction of inference. The reservoir work does not tell us what we
will find; it tells us we need enough seeds to distinguish "nothing happened" from "we
could not tell".

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

**Oja's rule converges to the dominant eigenvector of the second-moment matrix
E[x xᵀ] of whatever activations flow through it.** That is *ordinary PCA* only when
those activations are zero-mean; on non-centred inputs the mean itself contributes and
the result is biased relative to covariance PCA.

This matters concretely here. At the current default site, x is post-GELU activation,
which is **not** zero-mean, so the honest statement is "Oja finds the dominant direction
of the second-moment matrix", not "Oja does PCA".

Either way the point stands: **the weight matrix will move and the attractors will
shift with no feedback whatsoever.** That is what the rule does. So a closed-loop result
proves nothing on its own -- our claim is about the *coupling*, weights changing while
the thing they are changing feeds back into them.

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
| Centring, or the deliberate absence of it | Applied to one arm and not the other, this alone changes the fixed point |

Record all of these alongside the result. If the two arms differ on any of them, the
difference between them is not evidence about feedback.

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
degrade in a documented way: early collapse where distributional errors accumulate, late
collapse where low-frequency events disappear permanently. Ours is the same shape at the
activation level, on a timescale of seconds instead of generations. Their measurements
transfer.

---

## Stabilisers other people needed

Nobody in the surveyed work got away with a single mechanism.

- **Daydreaming Hopfield:** periodic renormalisation of the weight matrix, plus hard
  clamping on individual entries, which becomes *necessary* at high load.
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
  plasticity*, Current Opinion in Neurobiology 43:166-176 (2017). **Partly verified.**
  The paper and its central argument are confirmed: Hebbian plasticity alone is unstable
  and runs away, it acts on seconds to minutes, most measured homeostatic plasticity acts
  over hours or days, and the gap means fast compensating mechanisms are mathematically
  necessary rather than optional. The specific **"10-20 seconds"** figure quoted here
  could not be pinned to this paper — it appears to come from the companion paper Zenke &
  Gerstner, *Hebbian plasticity requires compensatory processes on multiple timescales*,
  Phil. Trans. R. Soc. B 372:20160259 (2017). Neither full text could be opened through
  the proxy. Attribute it to the pair, or check it before quoting the number.

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
- ~~Most entries unverified.~~ **Closed.** Every entry now carries a status, and all but
  one are verified against the source.
- ~~The Chaudhary depth figures.~~ **Closed, and they say something different from what
  was recorded.** See the correction near the top of this file.

**Still open:**

- **Nothing has been tested past 8 layers.** Chaudhary's stress test stops there; GPT-2
  small is 12. Whatever the Hebbian side does at 12 layers in a closed loop on frozen
  pretrained weights is unmeasured by anyone, including in the direction of "quietly does
  nothing".
- **The "10-20 seconds" homeostasis figure is not pinned to a source.** Do not quote the
  number until someone opens one of the two Zenke papers. The qualitative argument is
  safe.
- **Non-English and non-indexed venues were not searched**, nor were workshop
  proceedings that do not appear on arXiv, closed Discord/Slack research communities, or
  university theses. A collision could still live in any of those.
- **Semantic Scholar's citation graph lags.** Anything published in the last few weeks
  that cites the four closest papers will not have shown up. Worth re-running the forward
  sweep immediately before any write-up.
- **No published step size exists to anchor our eta sweep.** Not a search gap any more —
  a real one. See the section above.
