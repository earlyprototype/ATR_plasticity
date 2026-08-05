# Multi-electrode array literature: verified sources

*Reference material only. This file records what was retrieved from the cultured-network
literature, what was verified, and what EXP-003 borrows from it. It contains no argument
about whether that literature applies to this project, and no interpretation of this
project's results. Nothing in this file enters the claim register: it is source material, and
the register cites it only through the rows that record what EXP-003 measured, C-65 to C-67, and through **C-44**, which records the limitation noted at the end of this file that no person on this project has opened any of these papers.*

**History.** This file was previously `MEA_ANALOGUE.md` and argued that the ATR loop is a
computational analogue of a dissociated cortical culture. That argument was written by an
agent, was not requested, and has been removed at the operator's instruction. What remains
is the source material, which is verifiable independently of any reading placed on it.

## What EXP-003 borrows

| Borrowed | From | Where it is used |
|---|---|---|
| Centre of activity: a firing-rate-weighted centroid over recording-site positions | Chao, Bakkum and Potter (2007) equation 1 | Stage 0, with block index in place of electrode position |
| Shuffled-position control: permute the site positions and recompute | Chao, Bakkum and Potter (2007) supplement S2 | Stage 0 controls A and B |
| Change-to-drift ratio: distance between periods over internal scatter, with 1.0 meaning no detectable change | Chao, Bakkum and Potter (2007) | Stage 0's separation statistic |
| Judging a statistic by the smallest true change it detects | Chao, Bakkum and Potter (2007) | Registered for Stage 1, not implemented |
| Distributed low-rate stimulation, roughly 1 Hz across 10 to 20 of 60 electrodes | Potter (2008) review, quoting Wagenaar et al. (2005) | Stage 3 site fraction, not yet run |
| Stimulation rate as a separate parameter from stimulation amplitude | Wagenaar et al. (2005) | Stage 3 rate ladder, not yet run |

## Verified sources

Where a passage is quoted, it was quoted from retrieved text rather than from memory.

| Source | What was verified | Verification status |
|---|---|---|
| Wagenaar, Madhavan, Pine and Potter (2005), *Controlling bursting in cortical cultures with closed-loop multi-electrode stimulation*, Journal of Neuroscience 25(3):680-688 | Title, authors, journal, volume, pages. Methods and results including the comparison between single-electrode and multi-electrode stimulation, and that bursting resumed when stimulation stopped | Read from the authors' own reprint, whose abstract matches the published one. The publisher's site refused access, so this is an author-hosted copy rather than the version of record |
| Potter, review chapter for the 6th International Meeting on Substrate-Integrated Microelectrodes (2008) | Stimulation parameters quoted verbatim; the statement that quieting bursts aids the induction and detection of lasting plasticity; the framing of population bursting as a model of deafferentation | Passages quoted directly from retrieved text |
| Wagenaar, Pine and Potter (2006), *Searching for plasticity in dissociated cortical cultures on multi-electrode arrays*, Journal of Negative Results in Biomedicine 5:16, **with its correction, Journal of Negative Results in Biomedicine 6:3 (2007)** | That most induction protocols failed, and which one did not | **The correction is essential and the uncorrected text is still publicly served.** It swaps two protocol labels throughout, so the successful protocol is the one that suppressed bursting with added magnesium rather than with distributed electrical stimulation, and it instructs that a discussion paragraph crediting distributed electrical stimulation be dropped entirely. An earlier draft of this file relied on the uncorrected text and stated the opposite |
| Wagenaar, Pine and Potter (2006), *An extremely rich repertoire of bursting patterns during the development of cortical cultures*, BMC Neuroscience 7:11 | 58 cultures followed over five weeks; burst share rising with culture age; denser cultures bursting earlier than sparse ones | Passages quoted from retrieved text |
| Chao, Bakkum and Potter (2007), *Region-specific network plasticity in simulated and living cortical networks: comparison of the center of activity trajectory (CAT) with other statistics*, Journal of Neural Engineering 4(3):294-308, doi:10.1088/1741-2560/4/3/015 | Defining equations for the centre of activity and its trajectory; the shuffled-position control and its reported sensitivities (detectable change 4.68% against 10.8% shuffled, sensitivity 88.7% against 35.4%); the change-to-drift definition; the companion measure over connection strengths | Quoted from the retrieved reprint. This is the source EXP-003 draws on most |
| Chao, Bakkum, Wagenaar and Potter (2005), *Effects of random external background stimulation on network synaptic stability after tetanization: a modeling study*, Neuroinformatics 3(3):263-280 | That this result is a **simulation** of 1000 integrate-and-fire neurons, not a living-culture measurement | Retrieved. Not relied on by any part of EXP-003 |
| Wagenaar, Nadasdy and Potter (2006), *Persistent dynamic attractors in activity patterns of cultured neuronal networks*, Physical Review E 73:051907 | Title, authors, journal | Contents not used |
| Bakkum, Chao and Potter (2008), *Spatio-temporal electrical stimuli shape behavior of an embodied cortical network in a goal-directed learning task*, Journal of Neural Engineering 5(3):310-323, doi:10.1088/1741-2560/5/3/004, PMCID PMC2559979 | The whitening transform applied to the centre of activity, cited in the limitations below | Retrieved from PMC in a single fetch, not cross-checked against the PDF. Not relied on by any measurement in EXP-003 |

## Recorded limitations of the source measurements

Stated by the authors, and carried here because EXP-003 adapts these measurements.

The centre of activity reduces 60 channels to 2 numbers and is not information-preserving:
many different activity distributions map to the same value. It is normalised by total
firing rate and is therefore blind to overall activity level. Bakkum, Chao and Potter
(2008) required a whitening transform to remove a directional bias caused by uneven
distribution of cells across the array.

## One gap

**No person on this project has opened any of these papers.** Every passage was retrieved
and quoted by an agent. Register row C-44 treats that as a real limitation rather than a
formality.
