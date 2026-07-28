# Prior art

*Search run 2026-07-27. What exists, in plain terms, and how close it gets.*

## The claim, stated at the strength the search supports

**No work matching this combination was found: a hand-written local rule, on a
pretrained frozen model, driven by a closed activation loop, with no task and no
loss.** Every ingredient separately is well studied; the combination did not appear in
what was searched.

That is a statement about a search, not about the literature. **It is not "nobody has
done this."** The coverage limits are recorded at the bottom of this file and they are
real — in particular, the venues where an unpublished version of this idea would most
likely sit were barely searched. Treat the claim as provisional until those gaps are
closed.

## How the search was done

| | |
|---|---|
| Method | ~20 web searches by an automated agent, following leads, fetching the closest papers |
| Date | 2026-07-27 |
| Databases | Open web and arXiv. **No** Semantic Scholar or citation-graph traversal |
| Forums | One query across blogs / LessWrong / Alignment Forum |
| Inclusion | Any work combining two or more of: local unsupervised rule, pretrained frozen model, closed activation loop, no objective |
| Verification | Two entries fetched and checked by hand (below). The rest are **as reported by the agent and not independently verified** |

## Verification status

- **Daydreaming Hopfield networks — verified.** Title, authors and substance confirmed
  against the arXiv record.
- **Chaudhary 2025 — partly verified.** Title and author confirmed. **The depth figures
  quoted below (divergence at 8 layers, stability around 4) were NOT found in the
  abstract and remain unverified** — they need checking against the paper body before
  anyone relies on them.
- **Everything else — unverified.** Identifiers are recorded so they can be checked.

---

## Two claims this repo has made that are wrong

Recorded here so they do not reach a write-up.

**1. "Iterating a frozen model is unexplored" — false.** It has been done in
autoencoders and in recent looped / latent-reasoning language-model work, and some of
that literature reports orbit-like trajectories. The parent project's *specific*
protocol and results may well be new; the general idea is not.

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
arXiv:2405.08777 · Neural Networks 186 (2025). **Verified.**

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

Its ancestor -- Hopfield, Feinstein & Palmer, *Unlearning has a stabilizing effect in
collective memories*, Nature 304:158 (1983) -- **is** genuinely target-free: run the
network, see where it lands, weaken that. 43 years old, and the true precedent.

### Hebbian and gradient-based plasticity in transformers

Siddharth Chaudhary. *Enabling Robust In-Context Memory and Rapid Task Adaptation in
Transformers with Hebbian and Gradient-Based Plasticity.* arXiv:2510.21908 (2025).
**Title and author verified; depth figures below unverified.**

**What they were trying to do.** Give transformers a fast-changing memory that updates
during use rather than during training, and compare a Hebbian version against a
gradient-based one.

**What they found.** Hebbian plasticity gives lower loss and stronger few-shot
generalisation; gradient-based updates do better on long-horizon credit assignment.
*Reported but not yet verified:* divergence at 8 layers, stability around 4, deepest
layers drifting most, and the Hebbian variant being stable but saturating.

**How close.** The nearest thing anyone will cite at us. **Not us** because the
plasticity rule is *learned by gradient descent on a task*, the models are trained from
scratch rather than pretrained and frozen, and updates are driven by external data
rather than a closed loop.

**If the depth figures hold, they matter: GPT-2 small is 12 layers.** Check before
relying on this.

### Reshaping reservoirs with unsupervised Hebbian adaptation

Cazalets & Dambre. Nature Communications 17:450 (2026). *Unverified.*

**What they were trying to do.** Take a randomly wired recurrent network, improve it
with local unsupervised rules and no gradient steps, then test it on downstream tasks.

**What they found.** Their own rule helped. Plain Oja-family rules and intrinsic
plasticity reportedly "never surpass" it and "seldom exceed even" the untouched random
network.

**How close.** Not close in substrate. Valuable for two reasons: the best available
template for *what to measure*, and the strongest evidence that **the most likely
outcome of our experiment is that very little happens.**

### Do language models need sleep?

Lee, McLeish, Goldstein & Fanti. arXiv:2605.26099 (2026). *Unverified.*

**What they were trying to do.** Let a model run forward with no external input at all
-- an offline phase -- updating fast weights by a local rule, then check whether it
performs better afterwards.

**What they found.** The offline phase helps on the tasks measured.

**How close.** Closest *loop shape*: closed loop, no input, weights changing. **Not us**
because the rule is learned rather than hand-written, the model is trained for the
procedure, and success is defined by downstream task scores.

---

## One published claim we can test cheaply

Gong, Chen & Ching, arXiv:2312.14896 (2023), report that anti-Hebbian plasticity alters
the convexity of attractor landscapes through a bifurcation, and -- counterintuitively
-- that landscapes are **more** sensitive to slower learning rates than faster ones.

Our eta sweep can test whether that trend **transfers to this substrate**. It cannot
falsify their result: a different model, different metrics and different assumptions
mean a non-replication here says nothing about their setting. Report it as "replicates"
or "does not reproduce under this setup", never as falsification.

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

**Linear attention is already an outer-product Hebbian update** (Schlag, Irie &
Schmidhuber, ICML 2021; Irie et al., ICML 2022). If we ever drift an attention matrix we
are formally running a fast-weight programmer whose keys and values are self-generated.
That literature's analysis and its known interference failure modes come with it, and we
would have to say why ours differs.

**The model-collapse literature is our high-gain neighbour** (Shumailov et al., Nature
2024). Models retrained on their own output degrade in a documented way. Ours is the
same shape at the activation level, on a timescale of seconds instead of generations.
Their measurements transfer.

---

## Stabilisers other people needed

Nobody in the surveyed work got away with a single mechanism.

- Daydreaming Hopfield: periodic renormalisation of the weight matrix, plus hard
  clamping on individual entries, which becomes *necessary* at high load.
- The reservoir work: a homeostatic target with bounds, plus a scaling safeguard.
- Self-organising recurrent networks (Lazar, Pipa & Triesch, 2009): three mechanisms at
  once -- spike-timing plasticity, intrinsic plasticity and synaptic normalisation.
  Remove any one and the healthy regime degrades.
- Zenke, Gerstner & Ganguli (2017): Hebbian plasticity destabilises population rates
  within 10-20 seconds while biological homeostasis acts over hours -- far too slow, so
  fast compensating mechanisms are mathematically necessary rather than optional.

**Where we stand:** the activation rescaling in the ATR loop is already a fast homeostat,
but it acts on activations, not weights. Oja's decay term is a weight-side one. Whether
those two suffice is open, and the honest answer is that everyone else needed more.

---

## Coverage gaps

These are the reasons the verdict at the top is provisional.

- **Blogs, LessWrong, the Alignment Forum: one query only.** This is exactly where an
  unpublished notebook or workshop poster doing this would live. **The most likely place
  a collision is hiding.** Worth an hour before writing any introduction.
- **No citation-graph sweep.** No forward-citation trace from the closest papers, which
  is the systematic way to close the remaining gap.
- **Most entries unverified**, as marked above.
- **Could not determine** any published learning rate for an Oja-family rule inside a
  pretrained transformer, because no such experiment was found. Our sweep would be the
  first data point -- a good sign for the gap, a bad sign for the compute budget.
