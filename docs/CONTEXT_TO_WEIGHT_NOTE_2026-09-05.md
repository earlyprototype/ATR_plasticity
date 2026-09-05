# Context to weight transfer: the idea, its parts, and the work around it

*A reading note written 2026-09-05 for TC, the operator, in answer to three requests made in session: give an overview of the idea raised that day, that a context could be folded into the model's weights by the write channel this project built; explain its technical elements in a readable form; and describe, with links, each piece of prior or related work the discussion touched. It sits beside [ORIENTATION.md](../ORIENTATION.md), which explains the loop and the plasticity rule, and beside [EXP_004_SPEC.md](../EXP_004_SPEC.md), the protocol that would test the idea, and does not repeat either.*

> **Provenance.** Facts about this project were read from the committed record: `ORIENTATION.md`, `CLAIMS.md`, `EXP_001_RESULTS.md`, `HANDOVER.md`, `RESONANCE_NOTE.md`, `ALIGNMENT_REVIEW.md` and `PRIOR_ART.md` in this repository; `README.md` and `docs/LATENT_CONTEXT_NOTE_2026-09-04.md` in the parent repository `lucier-gpt2-activ-tensor-reson-experiments`; and `OPERATOR_REPORT_2026-07-31.md` in `ATR_research`. Facts about papers were read from their entries in `PRIOR_ART.md` where an entry exists, and are otherwise recalled from training; section 4 marks each one, and no paper was re-read for this note. Model sizes were read from the TransformerLens configuration of `gpt2` loaded in this session. One thing was run: a scratch pilot of the proposed measurement on GPT-2 Small, on this session's CPU, in about twenty seconds; its script and output are committed as `experiments/exp004_pilot.py` and `experiments/output_exp004/`, and section 5 reports it with its limits first. Nothing from it enters the claim register. Each claim is marked as established (read from a record or a paper), inferred (reasoned from established facts), or speculation.

---

## 1. The answers in brief

**The idea is that the write channel this project built can carry a context the model was given, so that the context can then be cleared.** The project showed that a Hebbian write, driven by the model's own activity, changes one weight matrix in a way that changes later behaviour. The same arithmetic is how a linear-attention model stores its context. So the channel should be able to store a context, and a context stored in weights no longer takes up room in the context window. That last consequence is where the session started; the reasoning that reached it is the project's own results. Section 2 has the idea in one page.

**The technical elements are a context window, a key-value cache, a weight matrix, an outer product, a softmax, and one measure of distance.** Each is explained in ordinary words in section 3, with a table of the sizes involved, and the whole idea reduces to: sum outer products of activity over the context into a matrix, install the matrix, and measure how far the model's predictions now sit from what they would have been with the context present.

**Nineteen pieces of published work sit near this, and this project differs from all of them in the same way.** Every one either trains the model to be written into, or computes the write with knowledge of what will later be asked. This project writes into a frozen, pretrained model with a fixed rule that never sees the query. Section 4 describes each work in a few sentences with a link.

## 2. The idea, in one page

**What the project has.** The parent project runs GPT-2 Small in a closed loop: it reads the model's residual stream, the running vector each layer reads from and adds to, at the last block, rescales it, and writes it back into the first block, hundreds of times. This repository lets one weight matrix change while that loop runs, under a Hebbian rule: the matrix is nudged in proportion to the product of the activity entering it and the activity leaving it. EXP-001 measured that at the sixth block's feed-forward output matrix this write, at 1.12 percent of the matrix's size, moves the loop's settled state from one basin to another, where a random write of the same size moves it at no step size tested (register rows C-21 and C-55). That is established.

**What the prior-art file already says about that write.** The entry for Schlag, Irie and Schmidhuber's 2021 paper records that a linear-attention model stores its context by summing, over every context token, the outer product of a key vector and a value vector into a matrix, and that if this project ever drifts an attention matrix it is formally running the same machine with self-generated keys and values. That is established.

**The deduction.** If the write is the operation by which a linear-attention model stores its context, then the write should be able to store a context. The project has only ever fed the channel the model's own free-running activity. Feeding it a context the model was given is a different input to the same channel. Whether anything survives the transfer is an open question. That is an inference from the two established facts above.

**The consequence.** A context held in a model's context window costs memory that grows with every token, in a structure called the key-value cache. A weight change costs a fixed amount however long the context was. A channel that moved context out of the window and into the weights would therefore be a memory-management step for any long-running use of a model, and that is the practical reason the idea matters. That is an inference; section 3 gives the sizes.

**What it changes for this project.** It supplies a target without adding a task. The reference is the model's own behaviour with the context present, and the score is how far the model with the context removed and the write installed sits from that reference. That score is a continuous number comparable across sites, rules, step sizes, loop lengths and models, where every result here so far is a discrete basin label. It also states the project's difference from weight-editing methods, which `ALIGNMENT_REVIEW.md` section 6 warns a reviewer will raise first, as a measurement: the editing methods compute their write knowing what will be asked, and this write does not, so the cost of that blindness can be put in numbers. Both are inferences about method.

**Where the room analogy holds and where it stops.** In the project's founding image, Lucier's recording played back into a room until only the room's tone remains, the room is the weights and the recording is the activity. This idea asks whether a sound played into the room once can change the room's shape, so that a later, different recording comes out coloured by the first. The analogy carries that far. It stops at what is carried: a room's shape can hold a resonance, a general colouring, but it cannot hold a sentence. Whether a weight change is like that, holding a colouring but not a fact, is what the experiment measures, and the analogy must not be allowed to answer it.

## 3. The technical elements, in ordinary words

**The context window and the key-value cache.** A language model predicts the next word from the words before it. The words it can see at once are its context, and the fixed maximum is the context window. To avoid recomputing, the model keeps, for every context token and every layer, two vectors called a key and a value; the store of these is the key-value cache, and it grows by a fixed amount per token. When the window is full, something has to be discarded or compressed. That is established.

**Weights against activations.** The model has two kinds of numbers. The weights are the fixed numbers learned during training; they define what the model is. The activations are the numbers that flow through it for one input; they are rebuilt from nothing for every new input. Learning from a context, in an ordinary model, changes activations only. The weights never move. This project is the exception: it moves one weight matrix. That is established.

**The Hebbian write.** At the chosen matrix, the rule watches the vector entering, `x`, and the vector leaving, `y`, and accumulates the outer product `x yᵀ`, a matrix in which each entry is one entering number times one leaving number. Averaged over the tokens it saw and scaled by a step size, that matrix is added to the weights. The rule has no target, no error signal and no knowledge of what will be asked later. That is established from `plasticity.py`.

**Why the write is the same arithmetic as linear attention's memory.** Ordinary attention lets each position look at every context position, weigh them by how well its query matches their keys, and take the weighted sum of their values. If the weighting is made linear, the whole context collapses into one matrix, the sum over context tokens of each value times each key, and every later query just multiplies by that matrix. That sum is an accumulated outer product, the same object the Hebbian rule builds. So a linear-attention model's memory of its context is a Hebbian matrix. That is established for those architectures.

**What the softmax adds, and why it is the loss.** Ordinary attention does not weigh linearly. It passes the query-key matches through a softmax, a step that turns a list of scores into a sharp set of weights that can pick out one exact token. That sharpness is what lets a transformer look up a specific fact from its context. A summed outer product has no such step; it can only add a fixed matrix. So when a context is folded into a matrix, whatever needed the sharp lookup is smeared. This is the established reason linear attention is worse than ordinary attention at exact recall, and it predicts that a Hebbian write will carry the general colouring of a context better than a specific fact in it. That prediction is an inference.

**How transfer is measured.** Run the model on the context followed by a query and record its next-word probabilities at the query positions; that is the reference. Run the query alone with the original weights; that is the baseline. Fold the context into the weights, clear it, and run the query alone again; that is the test. The score is the KL divergence, a standard measure of how much one probability distribution differs from another, in units called nats, from the reference to the test, averaged over the query positions. Zero means the folded model predicts exactly as the with-context model did; the baseline's own score says how far away "no context at all" sits. A control writes random numbers of the same size into the same matrix, so that any improvement can be attributed to the direction of the write and not its mere size. That is a method, and the pilot in section 5 ran it once.

**The loop as an optional consolidation step.** The parent's loop iterates the model on its own state with no input. One published method, described in section 4, folds context into weights after an offline pass over the stored context. This project's loop can run before the fold with no stored context at all. Comparing a fold after one pass with a fold after many loop iterations asks whether the loop consolidates the context or erases it. This is the one comparison only this project can make, and it is an inference that it is informative rather than a measurement.

**The sizes involved.** Arithmetic on published shapes, established for GPT-2 Small from the configuration loaded in this session and recalled for the larger model.

| Model and context | Key-value cache the context occupies | One weight change at a single feed-forward output matrix |
|---|---|---|
| GPT-2 Small, per token of context | 18,432 numbers: 12 layers, a key and a value each, 768 numbers wide | 2,359,296 numbers for the full matrix, or 3,840 for a rank-one change, the shape EXP-001 measured the write to have (95.8 percent of its squared size in one direction, `EXP_001_RESULTS.md` section 4) |
| GPT-2 Small, a 1,000-token context | 18.4 million numbers | as above; the weight change does not grow with the context |
| A 7-billion-parameter model of the Llama 2 shape, a 100,000-token context | about 26 billion numbers, about 52 gigabytes at 16-bit precision | tens of megabytes as a low-rank adapter, the usual way such a change is stored |

## 4. Prior and related work, each in a few sentences

Each entry says what the work does, how it relates to the idea, and whether it was read from `PRIOR_ART.md` (verified there) or is recalled from training with a link not re-checked in this session. The sameness across all of them, an inference from the descriptions, is that each either trains the model to be written into or computes the write knowing the query; this project does neither.

**Fast weights and the outer-product memory.**

- **Hopfield 1982, "Neural networks and physical systems with emergent collective computational abilities."** The origin of storing patterns as a sum of outer products in a weight matrix and reading them back by settling into an attractor. This project's basins are the same kind of object, and its write rule is the same kind of storage. Recalled. https://doi.org/10.1073/pnas.79.8.2554
- **Ba, Hinton, Mnih, Leibo and Ionescu 2016, "Using Fast Weights to Attend to the Recent Past."** Adds to a recurrent network a second, fast-changing weight matrix updated by an outer product of recent activity, as a short-term memory beside the slow trained weights. The two-timescale picture, fast weights holding the recent past and slow weights holding what was learned, is the picture this idea sits in. Recalled; `ALIGNMENT_REVIEW.md` finding F10 names it as missing from `PRIOR_ART.md`. https://arxiv.org/abs/1610.06258
- **Schlag, Irie and Schmidhuber 2021, "Linear Transformers Are Secretly Fast Weight Programmers."** Shows that attention with a linear weighting is exactly a fast-weight memory: the context becomes a matrix built from outer products of keys and values. This is the entry that makes this project's write channel a context-storage channel on paper. Verified in `PRIOR_ART.md`. https://arxiv.org/abs/2102.11174
- **Irie, Csordás and Schmidhuber 2022, "The Dual Form of Neural Networks Revisited."** Makes the same equivalence general: any linear layer trained by gradient descent can be written as attention over its training examples, so weights and stored examples are two views of one thing. Verified in `PRIOR_ART.md`. https://arxiv.org/abs/2202.05798
- **Yang, Kautz and Hatamizadeh 2024, "Gated Delta Networks."** A linear-attention variant whose memory matrix is updated by a delta rule, an outer-product write with a forgetting term that is, in form, Oja's rule from this repository. The current state of the art in that family. Recalled. https://arxiv.org/abs/2412.06464

**In-context learning as a weight update in disguise.**

- **von Oswald and colleagues 2023, "Transformers Learn In-Context by Gradient Descent."** Constructs transformers whose forward pass over a context is one step of gradient descent on the context, so that learning from context and learning into weights are the same computation. Supports the claim that a context has a weight-delta equivalent. Recalled. https://arxiv.org/abs/2212.07677
- **Dai and colleagues 2023, "Why Can GPT Learn In-Context?"** Argues that a pretrained transformer's attention over context produces an implicit weight update, a "meta-gradient", and measures that it behaves like fine-tuning. Same claim, on real models. Recalled. https://arxiv.org/abs/2212.10559
- **Todd and colleagues 2023, "Function Vectors in Large Language Models."** Finds that the effect of a task-demonstrating context can be captured as a single vector in the residual stream and added into a later run with no context, transferring the task. This is context-to-activation transfer, one step short of context-to-weight, and its success on tasks and failure on specifics matches the pilot in section 5. Recalled. https://arxiv.org/abs/2310.15213
- **Hendel, Geva and Globerson 2023, "In-Context Learning Creates Task Vectors."** The same finding from a second group at the same time. Recalled. https://arxiv.org/abs/2310.15916

**Architectures that update weights while reading.**

- **Sun and colleagues 2024, "Learning to (Learn at Test Time): RNNs with Expressive Hidden States."** Makes the hidden state of a sequence model a small weight matrix, updated by a gradient step on each token as it is read. Context-to-weight transfer as the design of the layer, trained end to end. Recalled. https://arxiv.org/abs/2407.04620
- **Behrouz, Zhong and Mirrokni 2025, "Titans: Learning to Memorize at Test Time."** A neural memory module updated at inference by a surprise-weighted rule, with forgetting, alongside ordinary attention. The same idea at production scale. Recalled. https://arxiv.org/abs/2501.00663
- **Chaudhary 2025, "Enabling Robust In-Context Memory and Rapid Task Adaptation in Transformers with Hebbian and Gradient-Based Plasticity."** Compares a Hebbian fast-weight rule against a gradient rule inside small transformers trained from scratch, finding the Hebbian rule better at few-shot memory. The nearest work on rule family; not this project because the rule is learned and the model is trained for it. Verified in `PRIOR_ART.md`. https://arxiv.org/abs/2510.21908

**Folding context into weights as a procedure.**

- **Snell, Klein and Zhong 2022, "Learning by Distilling Context."** Fine-tunes a model without the context to match the same model's outputs with the context, so the context's effect moves into the weights by gradient descent. The gradient-based rung of the fidelity ladder EXP-004 proposes. Recalled. https://arxiv.org/abs/2209.15189
- **Eyuboglu and colleagues 2025, "Cartridges: Lightweight and General-Purpose Long Context via Self-Study."** Trains a small key-value cache offline, by having the model quiz itself on a long document, so the document can be served without the full cache. Context compression by training, aimed at the same memory cost. Recalled. https://arxiv.org/abs/2506.06266
- **Lee, McLeish, Goldstein and Fanti 2026, "Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference."** Periodically pauses, runs offline recurrent passes over the accumulated context, folds it into fast weights with a learned rule, and discards the key-value cache. The closest published shape to the idea; not this project because the rule is learned, the model is trained for the procedure, and the offline pass reads stored context where this project's loop reads nothing. Verified in `PRIOR_ART.md`. https://arxiv.org/abs/2605.26099

**Editing weights directly, with knowledge of the query.**

- **Meng, Bau, Andonian and Belinkov 2022, "Locating and Editing Factual Associations in GPT" (ROME).** Writes a single fact into a mid-stack feed-forward output matrix of GPT-2, the same site class this project uses, by a closed-form rank-one edit computed from the fact's subject. Register row C-43 records that it is not yet in `PRIOR_ART.md` and that the difference is the target: ROME knows the query, the Hebbian write does not. Recalled. https://arxiv.org/abs/2202.05262
- **Meng, Sharma, Andonian, Belinkov and Bau 2023, "Mass-Editing Memory in a Transformer" (MEMIT).** Extends ROME to thousands of facts across several layers by least squares. The least-squares rung of the fidelity ladder. Recalled. https://arxiv.org/abs/2210.07229

**Latent reasoning, for where the parent's loop sits.**

- **Hao and colleagues 2024, "Training Large Language Models to Reason in a Continuous Latent Space" (Coconut).** Feeds the model's final hidden state back in as its next input and trains it to reason that way. The parent's note of 2026-09-04 identifies the ATR loop as Coconut's latent mode with the training removed. Recalled. https://arxiv.org/abs/2412.06769

**The biological frame.**

- **McClelland, McNaughton and O'Reilly 1995, "Why there are complementary learning systems in the hippocampus and neocortex."** The theory that the brain keeps a fast store for episodes and a slow store for structure, and consolidates from one to the other offline, during sleep. The fast-and-slow, context-and-weights picture in machine terms is this picture, and the sleep paper above takes its name from it. Recalled. https://doi.org/10.1037/0033-295X.102.3.419

## 5. The pilot, with its limits first

**Limits.** One site, `blocks.6.mlp`, the sixth block's feed-forward output matrix. One forward pass over each context, so one write. Three contexts, one per class. One random seed for the control. No loop. The drift ceiling was lifted so the rule could be measured up to 15 percent of the matrix's size. Scratch code, run once. The operator has ruled that this is to be given little weight, and nothing here enters the register.

**What came out.** The score is the KL divergence from the with-context reference over the query positions, in nats, lower meaning more of the context survived, with the no-context baseline as the comparison. Best means the lowest score over the six step sizes tried.

| Context class | Baseline, no context | Best after the Hebbian write | Random noise of the same size at that drift | Log probability of the reference's top word, no context, then best after the write |
|---|---|---|---|---|
| A fact: an invented country and its capital | 1.449 | 1.449; never improved, and 2.105 at 15 percent drift | 1.449 to 1.461 | minus 1.94, then minus 1.39 at 5.1 percent drift |
| A format: four lines each giving a word and then the same word in capitals | 4.654 | 4.654; never improved, and 5.879 at 12 percent drift | 4.654 to 4.636 | minus 6.61, then minus 6.46 |
| A topic: three sentences about a reactor | 1.918 | 1.836 at 11 percent drift | 1.939 | minus 7.22, then minus 7.20 |

**The pilot reading.** The write moved the model toward the words in the context and did not store which word goes with which. On the fact, the correct word became more likely while the whole distribution moved further from the reference, and further than random noise of the same size moved it. On the format, nothing transferred. On the topic, the score fell by 4 percent, 1.918 to 1.836, against a random control that rose. This is consistent with a write that is close to rank one, a single direction added everywhere, which is what EXP-001 measured, and with the softmax argument in section 3. It is a pilot reading that a second seed or a second context could overturn.

## 6. What remains, and what needs the operator's decision

What happened: this note recorded the idea, traced it to two results in the record and one entry in the prior-art file, explained its parts, described nineteen related works with links, and reported one pilot run. What it means: the project's write channel is, by the same arithmetic, a context-storage channel; whether it can store a given context is an open, cheap, measurable question; and the pilot, given little weight, suggests a single write at one site stores a colouring and not a fact. Both readings are inferences.

What remains, in order of information per unit cost, none of it started:

1. **Read and amend `EXP_004_SPEC.md`**, which sits beside this note with register rows C-69 to C-71 entered as open. Cost: a reading.
2. **Stage 1 of EXP-004, the single-pass map.** Four rules, twelve feed-forward and twelve attention output sites, eight step sizes, thirty contexts in three classes, a ten-seed random control. Cost: single-digit hours of CPU on GPT-2 Small.
3. **Stage 2, the loop as consolidation.** The write after zero, one, ten and one hundred loop iterations. The one stage no other group can run. Cost: about an hour of CPU.
4. **Stage 3, the fidelity ladder.** Hebb, Oja, a query-blind least-squares fit, and a gradient-trained low-rank adapter at matched drift. Cost: hours, most of it the adapter.
5. **The prior-art additions in section 4.** Fifteen of the nineteen entries are not in `PRIOR_ART.md`. Cost: a reading session and a verification pass, no model time.

What needs the operator's decision:

- **Whether EXP-004 runs, and which stages.** The recommendation offered is Stages 1 and 2 together: Stage 2 is the result only this project can produce, and Stage 1 is the map it must be read against.
- **Whether the pilot script and output stay in the repository.** They are committed so the pilot's numbers cite an artifact, as the register requires of any number quoted in prose. The recommendation offered is to keep them, labelled as they are.
- **Which site class to lead with.** The feed-forward output matrix is where every committed result lives; the attention output matrix is where the fast-weight equivalence is closest. The recommendation offered is to run both in Stage 1.

## Sources

- `ORIENTATION.md`, `CLAIMS.md`, `EXP_001_RESULTS.md`, `HANDOVER.md`, `RESONANCE_NOTE.md`, `ALIGNMENT_REVIEW.md`, `PRIOR_ART.md`, `plasticity.py`, this repository.
- `docs/LATENT_CONTEXT_NOTE_2026-09-04.md` and `README.md`, in `earlyprototype/lucier-gpt2-activ-tensor-reson-experiments`.
- `OPERATOR_REPORT_2026-07-31.md`, in `earlyprototype/ATR_research`.
- `experiments/exp004_pilot.py` and `experiments/output_exp004/pilot.json`, this repository, the pilot's script and output.
- The nineteen papers listed in section 4, each with its link there.
