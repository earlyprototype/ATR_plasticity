# Position ATR within dynamical-systems interpretability

**Type:** research / literature
**Related:** #NORM (normalisation & homeostasis — replace with issue number once created)

## Why this issue exists

ATR is an iterated-map experiment on a frozen transformer. The mechanistic
interpretability mainstream is circuits and sparse autoencoders — static,
structural methods. There is a separate, much smaller dynamical-systems thread,
and ATR sits inside it without currently citing it. This issue captures that
literature, identifies which parts bear on our open questions, and records the
positioning claim we can defend.

## The defensible positioning claim

> Dynamical-systems interpretability of transformers is nascent and largely
> theoretical. Where it is empirical, it treats **depth as the time axis** (one
> pass through the stack) or operates at the **text level** (iterated
> generate-and-resubmit). ATR occupies a third position: repeated application of
> the **whole forward map** with **tensor-level** re-injection. That niche is
> close to empty.

Worth stating carefully — "empty niche" claims age badly. Re-verify before any
write-up.

## Literature to read and place

### The mature lineage: RNNs (read first — closest methodology)

- **Sussillo & Barak (2013)** — the founding method: locate fixed points and
  slow points by optimisation, linearise around each, assemble the phase-space
  flow piecewise, read stability off the eigenspectrum.
- **Maheswaranathan et al. (2019)** — reverse-engineering RNN dynamics via
  linearisation; line attractors for sentiment analysis.
- **Marschall & Savin (2023)** — more recent treatment.
- **Known limitation, directly relevant to us:** the linearise-around-fixed-points
  approach becomes intractable when many dimensions behave non-convergently.
  **This is our Pythia-410m case.** That literature has already met our hardest
  problem and not solved it. Cite this rather than presenting 410m as a novel
  difficulty.

### Depth-as-time for transformers

- **"Transformer Dynamics: A neuroscientific approach to interpretability of
  large language models"** (arXiv 2502.12131) — residual stream as a dynamical
  system evolving across layers. Reports continuity of individual unit
  activations despite the non-privileged basis, activations accelerating and
  densifying with depth, individual units on unstable periodic orbits,
  attractor-like dynamics in lower layers in reduced dimensions, and
  self-correction after perturbation ("pseudo-attractor"). Also the source of the
  "nascent and largely theoretical" characterisation.
  - **Critical distinction to make explicit in any write-up:** this is a
    *different time axis from ours*. Depth-as-time is one pass through the
    stack. ATR is repeated application of the entire stack. Two dynamical
    systems extracted from one model.
  - **This bears directly on our open question** "does the landscape depend on
    where the loop is cut (layer window / depth)?" — that question is precisely
    about how these two time axes interact.
- **Sander et al. (2022)** — residual connections as Euler discretisation of a
  continuous flow; depth as integration time.
- **Geshkovski et al. (2023)** — self-attention as an interacting particle
  system; mean-field limits and clustering dynamics.
- **Wright & Gonzalez (2021)** — transformers as flows on sequence space with
  conservation laws.

### Text-level iteration (nearest neighbour to our result)

- **"Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems
  View of Successive Paraphrasing"** (arXiv 2502.15208) — treats an LLM as an
  iterative text-to-text map, characterises long-term behaviour as attractors
  including fixed points **and limit cycles**, using successive paraphrasing as
  the testbed.
  - **They find limit cycles at the text level. We found an exact period-2 limit
    cycle at the tensor level (F9).** Convergent evidence from an independent
    direction and a different level of description. This is the single most
    important citation for the `Divine` result and should be added to
    FINDINGS.md F9.

### Adjacent formalism

- **Deep equilibrium models (Bai et al.)** — explicitly solve for the fixed
  point of a repeatedly-applied layer. Same mathematics as ATR, deployed at
  training time rather than as a probe. Useful for framing ATR as "DEQ analysis
  applied post hoc to a model that wasn't trained as one."
- **Hopfield (1982)** — the ancestor of all of this; attractor networks.
- **Block-recurrent dynamics in vision transformers** (arXiv 2512.19941) —
  "dynamical interpretability" as an explicit term; directional convergence on
  the unit sphere, class-dependent basins, angular attractors. Note the
  **directional** framing: they measure cosine to final representation and find
  S-shaped saturation. Methodologically very close to our convergence gate.

## Tasks

- [ ] Read Sussillo & Barak (2013) and write 3 sentences on whether
      fixed-point-finding-by-optimisation is applicable to our setting, or
      whether the dimensionality forecloses it
- [ ] Read arXiv 2502.12131; add the depth-as-time vs whole-map-iteration
      distinction to TECHNICAL.md
- [ ] Read arXiv 2502.15208; add to FINDINGS.md F9 as convergent evidence
- [ ] Add a "Related dynamical-systems work" section to
      ATR_METHOD_COMPARISON.md (currently frames ATR against circuits/SAEs only)
- [ ] Re-verify the "empty niche" claim with a fresh literature search before
      any write-up
- [ ] Decide whether the biological-analogue framing (Potter lab / MEA cultures)
      belongs in the repo or stays as background reading

## Biological analogue (background, may not belong in repo)

Potter lab (Georgia Tech / Caltech, ~1997–2015) ran closed-loop stimulation on
living cortical cultures on multi-electrode arrays. Directly relevant papers:

- **Wagenaar, Nadasdy & Potter (2006)**, *Persistent dynamic attractors in
  activity patterns of cultured neuronal networks*, Phys Rev E 73:051907 —
  attractor characterisation in a living network.
- **Potter (2008)**, *How Should We Think About Bursts?* — measurement
  conventions manufacture the phenomenon. Our aliasing failure (F9) is the same
  error class: an even-only snapshot schedule cannot see a period-2 orbit.
- **Chao, Bakkum & Potter (2007)**, *Comparison of the Center of Activity
  Trajectory (CAT) with other statistics*, J Neural Eng 4:294 — head-to-head
  comparison of summary statistics for detecting change in a high-dimensional
  population state, in both simulated and living networks. Relevant to gate
  design (#9).
- **Wagenaar, Pine & Potter (2006)**, *Searching for plasticity...*, J Negat
  Results BioMed 5:16 — the methodological model for our own refutation:
  positive controls established first, every change measured against
  spontaneous drift, and the false-positive mechanism diagnosed rather than
  merely reported.
