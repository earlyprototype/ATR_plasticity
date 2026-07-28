# Prior art

*Searched 2026-07-27. What exists, in plain terms, and how close it gets.*

**Verdict: the specific combination here is unoccupied.** A hand-written local rule,
on a pretrained frozen model, driven by a closed activation loop, with no task and no
loss. Every ingredient separately is well studied. The four-way combination did not
turn up anywhere.

That is a narrower claim than "nobody has done Hebbian learning in a transformer",
and it is the one to make.

---

## Two things this repo has claimed that are wrong

Recorded here so they do not reach a write-up.

**1. "Iterating a frozen model is unexplored" — false.** It has been done in
autoencoders and in recent looped / latent-reasoning language-model work, and some of
that literature reports orbit-like trajectories. So someone has already seen limit
cycles under iteration, in a different setting. The parent project's *specific*
protocol and results may well be new; the general idea is not.

**2. "Frozen weights is just self-training with the learning rate at zero" — false.**
Gradient descent on a loss and a local correlation rule are different processes, not
one process at two gains. Turning eta to zero recovers frozen weights, but there is no
continuous path from Oja to backpropagation. The defensible framing is two separate
axes -- *what kind of rule* and *how strong* -- with the local-rule axis unswept at
every strength.

---

## The four closest pieces of work

### Daydreaming Hopfield networks (2024/25)

**What they were trying to do.** Fix an old problem with associative memory networks:
they invent false memories -- stable states nobody stored. The idea was to let the
network dream. Run it until it settles into a state on its own, then adjust the
weights to weaken whatever it just settled into, while reinforcing real stored
patterns.

**What they found.** It works. The false memories get erased, and the real ones end up
with much larger basins -- they become easier to fall into from a wider range of
starting points. Capacity improves substantially.

**How close.** Closest mechanism in existence. The weight update is driven by a state
the network generated itself, with no gradients and no loss. **Not us** because it is a
Hopfield network rather than a transformer, and because the update still refers to
stored patterns -- there is a target, just not a loss function.

Its ancestor, Crick's "unlearning" (1983), *is* genuinely target-free: run the network,
see where it lands, weaken that. That is 43 years old and is the true precedent for
this idea.

### Hebbian plasticity in transformers (2025)

**What they were trying to do.** Give transformers a fast-changing memory that updates
during use rather than during training, and compare a Hebbian version against a
gradient-based one.

**What they found.** It runs, but fragilely. At 8 layers it diverges after around 3000
steps. Stable at roughly 4 layers. The deepest layers drift most. Their Hebbian variant
is described as stable but saturating -- it does not explode, it just stops doing
anything.

**How close.** This is the nearest thing anyone will cite at us. **Not us** because the
plasticity rule itself is *learned by gradient descent on a task*, the models are
trained from scratch rather than pretrained and frozen, and the updates are driven by
external data rather than a closed loop.

**Directly relevant warning: GPT-2 small is 12 layers.**

### Reshaping reservoirs with unsupervised Hebbian adaptation (Nature Communications, 2026)

**What they were trying to do.** Take a randomly wired recurrent network and improve it
using only local unsupervised rules -- no gradient steps at all -- then check whether it
got better at downstream tasks.

**What they found.** Their own rule helped. But plain Oja-family rules and intrinsic
plasticity "never surpass" it and "seldom exceed even" the untouched random network.

**How close.** Not close in substrate -- random reservoir, external data, judged on task
accuracy. Valuable for two reasons: it is the best template available for *what to
measure*, and it is the strongest evidence that **the most likely outcome of our
experiment is that nothing much happens.**

### Do language models need sleep? (2026)

**What they were trying to do.** Let a model run forward for a while with no external
input at all -- an offline phase -- updating fast weights by a local rule, then check
whether it performs better afterwards.

**What they found.** The offline phase helps on the tasks they measured.

**How close.** Closest *loop shape*: a closed loop with no input, weights changing. **Not
us** because the local rule is learned rather than hand-written, the model is trained for
the procedure, and success is defined by downstream task scores.

---

## One result that makes a prediction we can test cheaply

Work on anti-Hebbian plasticity in attractor networks (2023) reports that the landscape
changes shape through a bifurcation as the rule is varied -- and, counterintuitively,
that **landscapes are more sensitive to slower learning rates than faster ones.**

Our eta sweep either reproduces that in a new substrate or falsifies a published claim.
Either is a result, and it costs nothing extra.

---

## The finding that changes our experiment

**Oja's rule on a fixed input distribution is PCA.** Point it at any stream of
activations and it rotates toward that stream's dominant direction. That happens with
no feedback whatsoever.

So the weight matrix will change, and the attractors will move, *and this proves
nothing on its own*. Our claim is about the coupling -- weights changing while the thing
they are changing feeds back into them.

**Required control.** Record the activations from the frozen loop; run Oja over that
recording offline with no feedback; install the resulting matrix; re-run the loop
frozen. Compare against the closed-loop run. **The claim lives entirely in the
difference between those two.** Without it, every result is attributable to "Oja did
PCA" and a reviewer will say so.

---

## Two framings we inherit whether we like them or not

**Linear attention is already an outer-product Hebbian update.** If we ever drift an
attention matrix, we are formally running a fast-weight programmer whose keys and values
are self-generated. That literature's analysis and its known interference failure modes
come with it, and we would have to say why ours is different.

**The model-collapse literature is our high-gain neighbour.** Models retrained on their
own output degrade in a documented way. Ours is the same shape at the activation level,
on a timescale of seconds instead of generations. Their measurements transfer.

---

## Stabilisers other people needed

Nobody in this space got away with a single mechanism.

- Daydreaming Hopfield: renormalise the weight matrix periodically, plus hard clamping
  on individual entries, which becomes *necessary* at high load.
- The reservoir work: a homeostatic target with bounds, plus a scaling safeguard.
- Self-organising recurrent networks: three mechanisms at once -- spike-timing
  plasticity, intrinsic plasticity and synaptic normalisation. Remove any one and the
  healthy regime degrades.
- Computational neuroscience: Hebbian plasticity destabilises population rates within
  10-20 seconds, while biological homeostasis acts over hours -- far too slow. Fast
  compensating mechanisms are mathematically necessary, not optional.

**Where we stand on this:** the activation rescaling in the ATR loop is already a fast
homeostat, but it acts on activations, not weights. Oja's decay term is a weight-side
one. Whether those two are sufficient is an open question, and the honest answer is
that everyone else needed more.

---

## Where the search was thin

- **Blogs, LessWrong, the Alignment Forum: one query only.** This is exactly the kind of
  experiment that lives as an unpublished notebook or a workshop poster. **Most likely
  place a collision is hiding.** Worth an hour before writing an introduction.
- **No citation-graph sweep.** No forward-citation trace from the closest papers, which
  is the systematic way to close the remaining gap.
- **Could not determine** any published learning rate for an Oja-family rule inside a
  pretrained transformer, because no such experiment was found. Our sweep would be the
  first data point -- good sign for the gap, bad sign for the compute budget.
