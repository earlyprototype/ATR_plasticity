"""
Basin bifurcation (issues #25, #32): what kind of move the EXP-001
`prolet` -> `comrade` flip is. D1 and D2 below, and the numbers they produce,
stand as recorded. The reading this file was built to argue does not.

  RETIRED READING -- read this before the rest of the docstring. This file was
  written to conclude that the flip is a *created* attractor, issue #25 ladder
  step 4, a bifurcation, rather than a boundary move into a pre-existing basin.
  That conclusion is `CLAIMS.md` C-26 and it is **`retired`** -- which is
  materially stronger than `not-established`: not asserted without sufficient
  evidence, but *contradicted*, by T1.1, the test this file itself named as
  deciding. Seeded from the original frozen `prolet` state under `W0 + ΔW` the
  loop does NOT stay: it moves to `comrade`, settling at iteration 12 and
  holding through 120. The two resting states therefore do not coexist, and
  coexistence is what step 4 required. What the edit does is *displace* a single
  attractor -- ladder **step 2**, register row **C-56**, `supported`,
  corroborated by T1.2's clean retrace with no hysteresis (**C-52**). Nothing
  measured here changed; what changed is what the measurements are allowed to be
  called. The report this file writes keeps the original wording in place marked
  as superseded, which is this repo's convention. See C-26, C-27, C-28, C-52,
  C-56 and C-68.

EXP-001 (`EXP_001_RESULTS.md`) established that at `hebb`, eta = 7.065e-05, one
120-step closed loop on `A01_physics` takes the basin from `prolet` to `comrade`
with the norm ceiling silent. It is NOT the only cell in the step-size map where
the loop moves: there are **two** ceiling-silent cells that flip the basin --
`hebb`@7.07e-05 (1.12% drift, the working point reproduced here) and
`hebb`@1.18e-04 (2.20% drift), both at 0.0% clip and both to `comrade` (C-21).
The two share prompt, seed, site and cadence, so they are NOT an independent
replication; what they establish is that the flip is robust across a 1.7x change
in eta. That result leaves one question open, and it is the question on issue
#25's ladder:

  step 2   the episode displaced a fixed point the map already had, moving it
           far enough that the readout relabels it across a ridge. One
           attractor, relocated. (Not one of the two this file set out to
           separate -- and, per T1.1/C-56, the one that survives.)
  step 3   the episode walked the state across a boundary that was already there,
           into a `comrade` basin the original frozen map already had. A *latent*
           attractor; the weights only nudged the state past a standing ridge.
  step 4   the episode *created* the `comrade` attractor. A bifurcation: at W0
           there is no `comrade` fixed point for the loop to fall into, and the
           accumulated ΔW brought one into existence.

This file runs two independent discriminators and they do answer the same way --
but on a reading neither of them can establish, because neither separates step 3
or step 4 from **step 2**. It also *addresses*, and does not close, issue #32
section 5, which asked of the installable-ΔW alpha-sweep: *smooth bias or a
threshold?* The answer this file gave, *a threshold*, is C-27 and is
`not-established`: an argmax changes discretely even under perfectly smooth
motion, so its discreteness is a property of the readout rather than evidence
about the dynamics. The sweep steps in 0.25, so what the grid supports is a
BRACKET -- the change of label lies in (0.50, 0.75] -- and not a threshold at
0.75.

## The two discriminators

  D1   Is `comrade` a fixed point of the ORIGINAL (frozen, W0) map? Restore W0 and
       iterate the frozen loop starting FROM the episode's final `comrade` state.
       If `comrade` were a pre-existing basin (step 3), the state stays in it; a
       created attractor (step 4) is not there at W0, so the original map carries
       the state back out. The state's own motion is slow and smooth -- lag-1
       cosine ~1.0 the whole way -- so any basin change is the argmax crossing a
       ridge, not a discontinuity.
       What D1 CANNOT do is establish creation, and its recorded verdict is
       therefore a measurement and nothing more. A merely *displaced* fixed point
       (step 2) carries the state back out too: for any ΔW != 0 the perturbed
       map's fixed point is generically not fixed under the unperturbed one, so
       D1 returns this same verdict for ANY ΔW that displaces it, a random edit
       included. The dual experiment is what discriminates -- seed the frozen
       `prolet` state under `W0 + ΔW` -- and that is T1.1, which found
       displacement (C-56).

  D2   The installable ΔW alpha-sweep (issue #32 section 5). Install `W0 + alpha·ΔW`
       for alpha in a grid straddling 1.0 and run the frozen loop under each. A
       *smooth bias* would slide the basin's logits and flip the basin somewhere
       proportional to alpha with the state deforming continuously; a *threshold*
       flips the basin discretely at some alpha* while the underlying logits move
       smoothly through the crossing.
       This grid does not settle that. A discrete change of argmax is not a
       bifurcation -- an argmax changes discretely under perfectly smooth motion
       too -- and at 0.25 spacing the flip is bracketed, not located. The one
       genuine qualitative transition the sweep finds is the lag-1 collapse
       between alpha 1.25 and 1.50, and it lands in the PRE-EXISTING `Divine`
       basin, so it is ladder step 3 rather than a created attractor (C-28).

## What this reproduces, and what is loaded

Nothing is loaded. The repo does not persist the EXP-001 closed-loop final state
or its ΔW (only the JSONL summaries), so this file *reproduces* the episode from
the frozen loop -- 120 `hebb` steps, cadence 1, applying after every step -- and
captures `comrade_state` and `ΔW = W_final − W0` in memory. The weight-space
anchors (`delta_frac`, ΔW σ₁) reproduce EXP-001 to ~9 figures; the closed-loop
state norm sits ~0.1% off its EXP-001 value, the reproduced-not-loaded drift, and
that gap is reported rather than hidden.

## The protocol choice D2 makes, stated up front

Each alpha is run from its OWN iteration-0 tensor, recomputed by a clean forward
pass under `W0 + alpha·ΔW`. So each alpha is a self-consistent frozen system: the
modified weights produce both the trajectory and the state it starts from. This
is NOT EXP-001 section 1's protocol, which installs each arm's matrix and re-runs
from the *same* iteration-0 tensor. Stated here so the sweep is checkable; the
alpha=1.0 row is therefore not identical to EXP-001's `closed` re-run row.

The choice is load-bearing, and it is half of a live discrepancy. At alpha = 0.50
this sweep settles on `prolet` from the prompt's iteration-0 tensor, while T1.2's
continuation sweep -- same site, rule, eta, prompt, steps and ΔW -- settles on
`comrade` from a settled state, in all four of its arms. That is C-68,
`provisional` and **unresolved**; the seeding is the candidate explanation and the
deciding test (a basin-of-attraction probe at alpha = 0.50, several initial states,
convergence criterion fixed in advance) has not run. Until it does, what each row
records is the settled word reached from THAT alpha's own iteration-0 tensor, and
this sweep may not be described as showing a single attractor at every alpha.

## Machinery

The repo's own, reimplementing nothing: `atr_bridge.initial_state` /
`make_atr_step` for the loop body, `plasticity.OjaPlasticity` for the rule and the
ΔW accumulation, and `baseline_basins.readout_detail` for the exact basin readout
every other experiment here uses.

## Running it

    python experiments/basin_bifurcation.py            # run + write report
    python experiments/basin_bifurcation.py --report-only

One torch thread, deterministic: `hebb` has no stochastic term, the model is
frozen, so a seed is a single reproducible run and there is nothing to shard.
About 6 CPU-minutes. If the reproduced episode does not reach `comrade` the run
stops before the discriminators and says so -- a reproduction that differs is
itself the finding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

# Set before torch imports anything that reads them.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atr_bridge import initial_state, make_atr_step            # noqa: E402
from plasticity import OjaPlasticity                           # noqa: E402
import baseline_basins as bb                                   # noqa: E402


# ---------------------------------------------------------------------------
# Config. Every value is echoed into the write-up; the same cell EXP-001 ran,
# because this file's whole premise is that it reproduces that episode before it
# takes it apart.
# ---------------------------------------------------------------------------

MODEL_NAME = "gpt2-small"          # TransformerLens alias of HF "gpt2"; identical weights.

# The plastic site: `blocks.6.mlp`'s W_out, (3072, 768), already in the rules'
# (n_in, n_out) convention. Same matrix EXP-001 and the step-size map moved.
SITE = "blocks.6.mlp"

# A01_physics: the step-size map's prompt and EXP-001's. Frozen basin `prolet`,
# a fixed point (lag-1 ~1.0), so any departure is visible against a still baseline.
PROMPT = "The implications of quantum entanglement suggest that"
PROMPT_ID = "A01_physics"

# The one eta where the loop moves. Recomputed by the step-size map's own anchor
# (D·‖W0‖_F / (N·U_ref[hebb]), D=1.8e-2), not rounded -- see EXP_001_RESULTS.md.
ETA = 7.065171428571429e-05
MODE = "hebb"
CADENCE = 1                        # apply after every loop step
MAX_DELTA_FRAC = 0.05              # the drift ceiling; silent throughout this cell
SEED = 0

LAYER_START = 0
LAYER_END = 11

# Episode length: EXP-001's published lock-in count. The frozen loop for this
# prompt is already settled at 120, so the captured `comrade` state is a settled
# state, not a mid-transient one.
N_EPISODE = 120

# D1: how far to iterate the frozen map from the comrade state. 200 is ample --
# the measured per-iteration contraction here is ~0.9995 on lag-1 and the basin
# resolves in the first handful of steps -- but the tail is recorded so the
# settled fixed point is visible, not assumed.
D1_N = 200
D1_SAMPLE_ITERS = (1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100, 150, 200)

# D2: the alpha grid, straddling 1.0 (the full ΔW). 120 frozen steps per alpha,
# matching the episode length. The last D2_TAIL basins are checked identical so a
# reported basin is a settled one, not the state passing through.
D2_N = 120
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
D2_TAIL = 15

# EXP-001's recorded numbers, for the fidelity cross-check. These are anchors to
# check the reproduction against, NOT targets to fit; the run reports its own.
# `closed_state_norm` is the reproduced-not-loaded one and is expected to drift.
ANCHOR = {
    "initial_norm": 1289.226318359375,
    "W0_norm": 164.85407309107723,
    "delta_frac": 0.011239339962675624,
    "dW_sigma1": 1.813517498340477,
    "closed_basin": "comrade",
    "closed_basin_token_id": 47998,
    "closed_state_norm": 4836.03897896114,
    "frozen_basin": "prolet",
    "frozen_basin_token_id": 22758,
}

TORCH_THREADS = 1

OUT_DIR = REPO_ROOT / "experiments" / "output_basin_bifurcation"
REPORT_PATH = REPO_ROOT / "BASIN_BIFURCATION.md"


# ---------------------------------------------------------------------------
# Readout helpers -- the repo's exact basin classifier, not a reimplementation.
# ---------------------------------------------------------------------------

def basin_of(model, state: torch.Tensor) -> dict:
    """The repo's exact readout (`baseline_basins.readout_detail`) on the last
    position -- the same one the baseline sweep, the step-size map and EXP-001
    label basins with."""
    d = bb.readout_detail(model, state[-1, :], k=5)
    return {
        "basin": d["top_token_strings"][0].strip(),
        "basin_raw": d["top_token_strings"][0],
        "basin_token_id": d["top_token_ids"][0],
        "top5": [t.strip() for t in d["top_token_strings"]],
        "top5_token_ids": d["top_token_ids"],
        "top5_logits": d["top_logits"],
        "margin": d["top_logit_margin"],
    }


def token_rank_logit(model, state: torch.Tensor, token_id: int) -> dict:
    """Where one specific token sits in the full-vocabulary logit ordering at the
    last position. `rank` is 1-based (1 == argmax). This is the instrument for
    the "smooth logits under a discrete basin flip" reading in D2: the argmax
    (the basin) is discrete, this is not."""
    logits = bb._logits(model, state[-1, :])
    lg = float(logits[token_id])
    rank = int((logits > logits[token_id]).sum().item()) + 1
    return {"rank": rank, "logit": lg}


def cosflat(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0))


def relL2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def _single_token_id(model, s: str) -> int:
    """Last BPE token of `s` -- the TransformerLens tokenizer prepends BOS, so
    encode(' Divine') is [50256, 13009] and the real id is the tail."""
    return int(model.tokenizer.encode(s)[-1])


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(model, out_dir: Path, meta: dict) -> tuple[list, dict]:
    """Reproduce the episode, then D1 and D2. Returns (records, meta).

    Weight hygiene: the plastic delta is captured before any revert, and the live
    matrix is put back to W0 at the end of every phase so no phase runs on another
    phase's weights."""
    t0 = time.time()
    records: list = []

    # -- Phase 0: iteration-0 state at W0 --------------------------------------
    st0 = initial_state(model, PROMPT, layer_end=LAYER_END)
    init_norm = st0.initial_norm
    print(f"[setup] initial_norm = {init_norm:.6f} "
          f"(anchor {ANCHOR['initial_norm']})", flush=True)

    # -- Phase 1: reproduce the 120-step closed hebb episode -------------------
    plast = OjaPlasticity(model, SITE, eta=ETA, mode=MODE, cadence=CADENCE,
                          max_delta_frac=MAX_DELTA_FRAC, seed=SEED)
    W0 = plast.W0.clone()
    W0_norm = plast.W0_norm

    # One step closure, reused for the episode, the frozen baseline and D1: all
    # three run layers 0->11 at this initial_norm, so they are the same map; only
    # the live weight and the starting state differ between them.
    step = make_atr_step(model, PROMPT, layer_start=LAYER_START,
                         layer_end=LAYER_END, initial_norm=init_norm)

    plast.install()
    r = st0.tensor.clone()
    t_epi = time.time()
    for _ in range(N_EPISODE):
        r = step(model, r)
        plast.apply()
    comrade_state = r.clone()
    dW = plast.delta.clone()                 # W_final − W0, exactly
    rep = plast.report()
    plast.remove()                           # detaches the hook; live weight is still W0+ΔW

    cb = basin_of(model, comrade_state)
    # Spectral cross-check of ΔW, in float64 like every norm in this repo.
    sv = torch.linalg.svdvals(dW.double())
    dW_sigma1 = float(sv[0])
    dW_fro = float(dW.double().norm())

    episode_ok = (cb["basin"] == "comrade" and cb["basin_token_id"] == 47998)
    print(f"[reproduce] n_applied={rep['n_applied']} "
          f"delta_frac={rep['delta_frac']:.12f} clipped={rep['clipped']} "
          f"nonfinite={rep['nonfinite']}", flush=True)
    print(f"[reproduce] closed basin={cb['basin']!r} ({cb['basin_token_id']}) "
          f"‖state‖={float(comrade_state.norm()):.5f}  ΔW σ1={dW_sigma1:.9f}  "
          f"top5={cb['top5']}", flush=True)

    comrade_id = cb["basin_token_id"]

    # Restore W0 exactly, and check it.
    plast.revert()
    restore = {"cos_W_vs_W0": cosflat(plast._site.weight, W0),
               "relL2_W_vs_W0": relL2(plast._site.weight, W0)}

    # -- Phase 1b: frozen baseline confirms prolet -----------------------------
    r = st0.tensor.clone()
    for _ in range(N_EPISODE):
        r = step(model, r)
    fb = basin_of(model, r)
    frozen_state_norm = float(r.norm())
    prolet_id = fb["basin_token_id"]
    divine_id = _single_token_id(model, " Divine")
    print(f"[frozen-baseline] basin={fb['basin']!r} ({prolet_id}) "
          f"‖state‖={frozen_state_norm:.4f}  top5={fb['top5']}", flush=True)

    reproduce_rec = {
        "kind": "reproduce",
        "episode_ok": episode_ok,
        "n_applied": rep["n_applied"],
        "delta_frac": rep["delta_frac"],
        "clipped": rep["clipped"],
        "nonfinite": rep["nonfinite"],
        "initial_norm": init_norm,
        "W0_norm": W0_norm,
        "seq_len": list(comrade_state.shape),
        "dW_sigma1": dW_sigma1,
        "dW_fro": dW_fro,
        "closed_state_norm": float(comrade_state.norm()),
        "closed_basin": cb["basin"],
        "closed_basin_token_id": cb["basin_token_id"],
        "closed_top5": cb["top5"],
        "closed_top5_token_ids": cb["top5_token_ids"],
        "closed_margin": cb["margin"],
        "frozen_basin": fb["basin"],
        "frozen_basin_token_id": fb["basin_token_id"],
        "frozen_top5": fb["top5"],
        "frozen_margin": fb["margin"],
        "frozen_state_norm": frozen_state_norm,
        "restore_check": restore,
        "comrade_token_id": comrade_id,
        "prolet_token_id": prolet_id,
        "divine_token_id": divine_id,
        "anchor": ANCHOR,
        "seconds": round(time.time() - t_epi, 2),
    }
    records.append(reproduce_rec)

    if not episode_ok:
        print("[reproduce] *** episode did NOT reach comrade -- stopping before "
              "the discriminators (a reproduction that differs is the finding) ***",
              flush=True)
        meta["episode_ok"] = False
        meta["stopped_reason"] = (
            f"reproduced closed basin {cb['basin']!r} ({cb['basin_token_id']}), "
            f"expected comrade (47998)")
        meta["total_seconds"] = round(time.time() - t0, 1)
        return records, meta
    meta["episode_ok"] = True

    # -- Discriminator 1: is comrade a fixed point of the frozen (W0) map? ------
    # Live weight is W0 (reverted above). Iterate the frozen map FROM the comrade
    # state; if comrade is not an attractor at W0 the map carries the state out.
    # That is all this measures. It does NOT show creation: a merely *displaced*
    # fixed point (ladder step 2) produces the same trace, because for any dW != 0
    # the perturbed map's fixed point is generically not fixed under the
    # unperturbed one -- so D1 returns this verdict for ANY dW that displaces it.
    # C-26, the created-attractor reading, is `retired`; T1.1 ran the dual
    # experiment and found displacement (C-56). Keep the verdict string below a
    # measurement: it is written into the committed JSONL.
    print(f"[D1] iterating the frozen map from the comrade state, {D1_N} steps ...",
          flush=True)
    r = comrade_state.clone()
    prev1 = comrade_state.clone()            # iter-0 reference for lag-1
    prev2 = None
    first_leave = None
    sample = set(D1_SAMPLE_ITERS)
    d1_rows = []
    for i in range(1, D1_N + 1):
        r = step(model, r)
        b = basin_of(model, r)
        lag1 = cosflat(r, prev1)
        lag2 = cosflat(r, prev2) if prev2 is not None else None
        cos_c = cosflat(r, comrade_state)
        rl2_c = relL2(r, comrade_state)
        if b["basin"] != "comrade" and first_leave is None:
            first_leave = {"iter": i, "basin": b["basin"],
                           "token_id": b["basin_token_id"]}
        if i in sample or (first_leave and first_leave["iter"] == i):
            row = {"kind": "d1", "iter": i, "basin": b["basin"],
                   "basin_token_id": b["basin_token_id"], "top5": b["top5"],
                   "margin": b["margin"], "lag1_cos": lag1, "lag2_cos": lag2,
                   "cos_to_comrade0": cos_c, "relL2_to_comrade0": rl2_c}
            d1_rows.append(row)
            records.append(row)
            blabel = f"{b['basin']}({b['basin_token_id']})"
            print(f"[D1] it={i:3d} basin={blabel:<14} "
                  f"lag1={lag1:.6f} cos->c0={cos_c:.5f} "
                  f"relL2->c0={rl2_c:.4f}", flush=True)
        prev2 = prev1
        prev1 = r.clone()

    last = basin_of(model, r)
    # `first_leave is None` is the horizon statement; the final label alone is not,
    # because a trajectory can leave `comrade` and come back and the final label
    # would not show it. Either way this is a statement about the READOUT over
    # D1_N steps, not about the state being fixed, which a finite run of identical
    # labels cannot establish.
    stays = (first_leave is None and last["basin"] == "comrade")
    d1_summary = {
        "kind": "d1_summary",
        "n_steps": D1_N,
        "stays_comrade": stays,
        "first_leave": first_leave,
        "final_basin": last["basin"],
        "final_basin_token_id": last["basin_token_id"],
        "final_top5": last["top5"],
        "final_lag1_cos": cosflat(r, prev2),
        # MEASUREMENT ONLY -- no ladder step, no bifurcation. This field lands in
        # `basin_bifurcation.jsonl`, so a reading written here becomes a reading
        # sitting in a committed data artifact. It used to read "created (issue
        # #25 step 4, a bifurcation)" / "latent (issue #25 step 3)"; both are
        # interpretations D1 alone cannot license (see the comment above the
        # loop), and the step-4 one is C-26, `retired`. What the ladder step is
        # belongs to CLAIMS.md -- C-56 -- not to this record.
        "verdict": (
            f"comrade IS a fixed point of the original frozen map: the frozen "
            f"loop holds it for all {D1_N} steps"
            if stays else
            f"comrade is NOT an attractor of the original frozen map: the frozen "
            f"loop leaves it at iteration {first_leave['iter']} and settles at "
            f"{last['basin']}"),
    }
    records.append(d1_summary)
    print(f"[D1] VERDICT: {d1_summary['verdict']}", flush=True)
    print(f"[D1] first_leave={first_leave} final_basin={last['basin']!r} "
          f"({last['basin_token_id']})", flush=True)

    # -- Discriminator 2: installable ΔW alpha-sweep (issue #32 section 5) -------
    # Install W0 + alpha·ΔW directly on the live matrix (bypassing the rule's
    # accumulator -- ΔW is already captured), recompute iteration-0 UNDER those
    # weights, run the frozen loop, read the settled basin. Restore W0 after each.
    print("[D2] alpha-sweep ...", flush=True)
    d2_rows = []
    for a in ALPHAS:
        plast._site.write(W0 + a * dW)
        st = initial_state(model, PROMPT, layer_end=LAYER_END)     # iter 0 under W0+alpha·ΔW
        stepa = make_atr_step(model, PROMPT, layer_start=LAYER_START,
                              layer_end=LAYER_END, initial_norm=st.initial_norm)
        r = st.tensor.clone()
        prev1 = r.clone()
        prev_for_lag = prev1
        tail = []
        for i in range(1, D2_N + 1):
            r = stepa(model, r)
            if i > D2_N - D2_TAIL:
                tail.append(basin_of(model, r)["basin"])
            prev_for_lag = prev1
            prev1 = r.clone()
        b = basin_of(model, r)
        lag1 = cosflat(r, prev_for_lag)
        row = {
            "kind": "d2", "alpha": a, "basin": b["basin"],
            "basin_token_id": b["basin_token_id"], "top5": b["top5"],
            "top5_token_ids": b["top5_token_ids"], "margin": b["margin"],
            "final_lag1_cos": lag1, "final_norm": float(r.norm()),
            "init_norm": st.initial_norm,
            "settled_last15": len(set(tail)) == 1, "tail_basins": tail,
            "comrade": token_rank_logit(model, r, comrade_id),
            "prolet": token_rank_logit(model, r, prolet_id),
            "divine": token_rank_logit(model, r, divine_id),
        }
        d2_rows.append(row)
        records.append(row)
        plast._site.write(W0)                # restore before the next alpha
        blabel = f"{b['basin']}({b['basin_token_id']})"
        print(f"[D2] alpha={a:.2f} basin={blabel:<14} "
              f"lag1={lag1:.6f} comrade(rank={row['comrade']['rank']},"
              f"logit={row['comrade']['logit']:.3f}) "
              f"prolet_logit={row['prolet']['logit']:.3f} settled={row['settled_last15']} "
              f"top5={b['top5'][:4]}", flush=True)

    base_basin = d2_rows[0]["basin"]
    alpha_star = next((row["alpha"] for row in d2_rows
                       if row["basin"] != base_basin), None)
    lag1s = [row["final_lag1_cos"] for row in d2_rows]
    d2_summary = {
        "kind": "d2_summary",
        "n_steps": D2_N,
        "alphas": list(ALPHAS),
        "base_basin": base_basin,
        "alpha_star_first_change": alpha_star,
        "lag1_min": min(lag1s),
        "lag1_max": max(lag1s),
        "basins_by_alpha": {str(row["alpha"]): row["basin"] for row in d2_rows},
    }
    records.append(d2_summary)
    print(f"[D2] base_basin(alpha=0)={base_basin!r} "
          f"alpha_star(first change)={alpha_star} "
          f"lag1 range [{min(lag1s):.6f}, {max(lag1s):.6f}]", flush=True)

    meta["total_seconds"] = round(time.time() - t0, 1)
    return records, meta


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x, spec=".4g", none="--"):
    if x is None:
        return none
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return format(x, spec)


def _by_kind(records: list, kind: str) -> list:
    return [r for r in records if r.get("kind") == kind]


def build_report(records: list, meta: dict) -> str:
    rep = next(iter(_by_kind(records, "reproduce")), None)
    d1 = _by_kind(records, "d1")
    d1s = next(iter(_by_kind(records, "d1_summary")), None)
    d2 = _by_kind(records, "d2")
    d2s = next(iter(_by_kind(records, "d2_summary")), None)

    L: list = []
    A = L.append

    # The title is kept in the words it was written in and marked superseded --
    # this repo's convention for a retired reading -- with the note below carrying
    # the correction, so a regenerated report cannot assert C-26 unqualified.
    A("# Basin bifurcation — `comrade` is a created attractor "
      "**[SUPERSEDED READING]**")
    A("")
    A("*The measurements in this file stand. The title's claim does not — it is "
      "**`retired`** in `CLAIMS.md` **C-26**, which is materially stronger than "
      "`not-established`: not asserted without sufficient evidence, but "
      "**contradicted**, by T1.1, the test this file itself named as decisive. "
      "T1.1 seeded the original frozen `prolet` state under `W0 + ΔW` and it did "
      "not stay — it moved to `comrade`, settling at iteration 12 and holding "
      "through 120 — so the two resting states do not coexist and the surviving "
      "reading is **C-56**: one **displaced** attractor, issue #25 ladder step "
      "2, corroborated by T1.2's hysteresis-free retrace (**C-52**). The title "
      "is retained as the record of what the project believed. Read C-26, C-27, "
      "C-28, C-52, C-56 and C-68 before quoting anything below.*")
    A("")
    A(f"Issues #25, #32. `{MODE}`, eta = {ETA:.6g}, site `{SITE}`, "
      f"{N_EPISODE}-step episode, cadence {CADENCE}, "
      f"`max_delta_frac` = {MAX_DELTA_FRAC}, prompt `{PROMPT_ID}`.")
    A("")
    A("EXP-001 (`EXP_001_RESULTS.md`) showed that one 120-step closed `hebb` "
      "episode at this eta takes the basin `prolet` → `comrade` with the norm "
      "ceiling silent. It is **not** the only cell in the step-size map where "
      "the loop moves: there are **two** ceiling-silent cells that flip the "
      "basin — `hebb`@7.07e-05 (1.12% drift, the working point used here) and "
      "`hebb`@1.18e-04 (2.20% drift), both at 0.0% clip and both to `comrade` "
      "(**C-21**). The two share prompt, seed, site and cadence, so they are "
      "**not** an independent replication; what they establish is that the flip "
      "is robust across a 1.7× change in eta. This file asks what kind of move "
      "it is, on issue #25's ladder: **step 3**, the episode walked the state "
      "across a boundary into a `comrade` basin the original frozen map already "
      "had (a *latent* attractor), or **step 4**, the episode *created* the "
      "`comrade` attractor (a bifurcation). Two independent discriminators are "
      "run; they agree — but, as the note above records, on a reading neither of "
      "them can establish, and T1.1 has since contradicted it (C-26 `retired`, "
      "C-56 `supported`). The `alpha`-sweep also **addresses** issue #32 section "
      "5, which asked of the installable ΔW: *smooth bias or a threshold?* — the "
      "answer this file gave, *a threshold*, is **C-27** and is "
      "`not-established`, so that question is not closed here either.")
    A("")
    A("**Measurement first.** The tables below are measurements; the words "
      "*created attractor* and *bifurcation* are an interpretation of them, "
      "marked as interpretation where it is made.")
    A("")

    if rep is None:
        A("No records.")
        return "\n".join(L)

    # --- setup + fidelity -------------------------------------------------
    A("## Setup and fidelity anchors")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| model | {MODEL_NAME}, frozen, CPU, float32 (norms float64) |")
    A(f"| site | `{SITE}` (= `transformer.h.6.mlp.c_proj`), (3072, 768) |")
    A(f"| prompt | `{PROMPT_ID}` — {json.dumps(PROMPT)} |")
    A(f"| rule | `{MODE}`, eta = {ETA:.6g}, cadence {CADENCE}, "
      f"`max_delta_frac` = {MAX_DELTA_FRAC}, seed {SEED} |")
    A(f"| loop | layers {LAYER_START}→{LAYER_END}, episode {N_EPISODE} steps, "
      f"D1 {D1_N} steps, D2 {D2_N} steps/alpha |")
    A(f"| alphas | {', '.join(_fmt(a, '.2f') for a in ALPHAS)} |")
    A("")
    A("Nothing is loaded: the repo persists no raw closed-loop state and no ΔW, "
      "so the episode is **reproduced** here from the frozen loop and "
      "`comrade_state` / `ΔW = W_final − W0` are captured in memory. The anchors "
      "are EXP-001's recorded numbers, checked against, not fitted to.")
    A("")
    an = rep["anchor"]
    A("| quantity | this run | EXP-001 recorded | rel. diff |")
    A("|---|---|---|---|")

    def _rel(a, b):
        return abs(a - b) / abs(b) if b else float("nan")
    A(f"| ‖x₀‖ (initial_norm) | {_fmt(rep['initial_norm'], '.6f')} | "
      f"{_fmt(an['initial_norm'], '.6f')} | "
      f"{_rel(rep['initial_norm'], an['initial_norm']):.2e} |")
    A(f"| ‖W0‖_F | {_fmt(rep['W0_norm'], '.6f')} | {_fmt(an['W0_norm'], '.6f')} | "
      f"{_rel(rep['W0_norm'], an['W0_norm']):.2e} |")
    A(f"| rel ΔW (`delta_frac`) | {_fmt(rep['delta_frac'], '.9f')} | "
      f"{_fmt(an['delta_frac'], '.9f')} | "
      f"{_rel(rep['delta_frac'], an['delta_frac']):.2e} |")
    A(f"| ΔW σ₁ | {_fmt(rep['dW_sigma1'], '.7f')} | {_fmt(an['dW_sigma1'], '.7f')} | "
      f"{_rel(rep['dW_sigma1'], an['dW_sigma1']):.2e} |")
    A(f"| closed basin | `{rep['closed_basin']}` ({rep['closed_basin_token_id']}) | "
      f"`{an['closed_basin']}` ({an['closed_basin_token_id']}) | "
      f"{'match' if rep['closed_basin'] == an['closed_basin'] else '**DIFFER**'} |")
    A(f"| frozen baseline basin | `{rep['frozen_basin']}` "
      f"({rep['frozen_basin_token_id']}) | `{an['frozen_basin']}` "
      f"({an['frozen_basin_token_id']}) | "
      f"{'match' if rep['frozen_basin'] == an['frozen_basin'] else '**DIFFER**'} |")
    A(f"| closed ‖state‖ (reproduced) | {_fmt(rep['closed_state_norm'], '.5f')} | "
      f"{_fmt(an['closed_state_norm'], '.5f')} | "
      f"{_rel(rep['closed_state_norm'], an['closed_state_norm']):.2e} |")
    A("")
    A(f"The weight-space anchors reproduce EXP-001 to ~9 figures — `delta_frac` "
      f"at {_rel(rep['delta_frac'], an['delta_frac']):.1e} relative, ΔW σ₁ at "
      f"{_rel(rep['dW_sigma1'], an['dW_sigma1']):.1e} — so the ΔW being taken "
      f"apart below is the EXP-001 ΔW. The closed-loop **state** norm sits "
      f"{_rel(rep['closed_state_norm'], an['closed_state_norm']):.2%} off its "
      f"EXP-001 value: the reproduced-not-loaded drift (the state is regenerated "
      f"from the loop rather than loaded, under transformer_lens "
      f"{meta.get('transformer_lens_version', '?')} vs the recorded run's), and "
      f"the basin label survives it. The episode applied {rep['n_applied']} "
      f"updates, `clipped` = {rep['clipped']} (the ceiling never fired), "
      f"`nonfinite` = {rep['nonfinite']}. After the episode W0 is restored "
      f"exactly: ‖W − W0‖_F / ‖W0‖_F = "
      f"{_fmt(rep['restore_check']['relL2_W_vs_W0'], '.1e')}.")
    A("")

    # --- D1 ---------------------------------------------------------------
    A("## D1 — is `comrade` a fixed point of the original (frozen) map?")
    A("")
    A("Restore W0 and iterate the **frozen** loop starting from the episode's "
      "final `comrade` state. A pre-existing basin (step 3) holds the state; a "
      "created attractor (step 4) is not present at W0, so the original map "
      "carries the state back out — and so does a merely **displaced** fixed "
      "point (step 2), which is the alternative this discriminator cannot "
      "separate and the reason its verdict below is stated as a measurement "
      "only. `cos→c₀` is the cosine to the starting "
      "`comrade` state; `lag-1` / `lag-2` are the cosines to the previous one and "
      "two iterates — near 1.0 means the state itself is barely moving.")
    A("")
    A("| iter | basin | lag-1 | lag-2 | cos→c₀ | relL2→c₀ |")
    A("|---|---|---|---|---|---|")
    for row in d1:
        A(f"| {row['iter']} | `{row['basin']}` ({row['basin_token_id']}) | "
          f"{_fmt(row['lag1_cos'], '.6f')} | {_fmt(row['lag2_cos'], '.6f')} | "
          f"{_fmt(row['cos_to_comrade0'], '.5f')} | "
          f"{_fmt(row['relL2_to_comrade0'], '.4f')} |")
    A("")
    if d1s:
        fl = d1s["first_leave"]
        if d1s["stays_comrade"]:
            A(f"The state stays in `comrade` for all {d1s['n_steps']} frozen "
              f"steps. **The measurement: `comrade` is a fixed point of the "
              f"original map.** Which ladder step that makes it is not D1's to "
              f"say — see `CLAIMS.md` C-26 and C-56.")
        else:
            A(f"The frozen map leaves `comrade` at **iteration "
              f"{fl['iter']}** (into `{fl['basin']}`) and settles at "
              f"`{d1s['final_basin']}` — the frozen baseline's own fixed point — "
              f"with lag-1 returning to {_fmt(d1s['final_lag1_cos'], '.6f')}. "
              f"Crucially the state's own motion is smooth and monotone the whole "
              f"way: lag-1 stays at or above "
              # '.6f', not '.5f': at 5 decimals the committed minimum 0.9999184
              # displays as 0.99992, which is *larger* than the value, so the
              # sentence was false as printed. The audit that caught it is
              # recorded in BASIN_BIFURCATION.md.
              f"{_fmt(min(r['lag1_cos'] for r in d1), '.6f')} — its minimum over "
              f"the sampled iterations — and `cos→c₀` decays without a jump. "
              f"There is no "
              f"discontinuity for the argmax to ride; the `comrade` label sat on "
              f"a thin ledge the original map slides straight off, not in a basin "
              f"that map has. **The measurement: `comrade` is not an attractor of "
              f"the original frozen map.** That stands as recorded.")
            A("")
            A(f"What it **cannot** establish is *creation*. D1 returns the same "
              f"verdict for **any** ΔW that displaces the fixed point at all — "
              f"for any ΔW ≠ 0 the perturbed map's fixed point is generically not "
              f"fixed under the unperturbed one — so this trace is equally what a "
              f"*displaced* fixed point produces, and the smooth monotone "
              f"relaxation back to `{d1s['final_basin']}` is that signature "
              f"rather than evidence against it. Discriminating the two needs the "
              f"dual experiment, seeding the frozen `{d1s['final_basin']}` state "
              f"under `W0 + ΔW`; that is T1.1, it has run, and it found "
              f"displacement (**C-56**). **[The reading this file originally drew "
              f"here — a created attractor, issue #25 step 4, a bifurcation — is "
              f"superseded; C-26 is `retired`.]**")
        A("")
        # Kept in the words it was written in, marked superseded: this repo's
        # convention. Do not delete it and do not un-mark it.
        A(f"> *Interpretation, in the words it was originally written in — kept "
          f"as the record of what the project believed.* **[SUPERSEDED — C-26 is "
          f"`retired`, refuted by T1.1; the paragraph above gives why D1 cannot "
          f"carry this reading, and C-56 is what replaces it.]** D1 was taken as "
          f"the definition of step 4 made operational: "
          f"the attractor the episode ended in is absent from the pre-episode "
          f"map. The episode did not walk the state to a standing `comrade` "
          f"basin; the accumulated ΔW is what put a `comrade` fixed point where "
          f"there was none.")
        A("")

    # --- D2 ---------------------------------------------------------------
    A("## D2 — the installable ΔW `alpha`-sweep (issue #32 section 5)")
    A("")
    A("Install `W0 + alpha·ΔW` and run the frozen loop under each `alpha`, then "
      "read the settled basin. A **smooth bias** would flip the basin somewhere "
      "proportional to `alpha` with the state deforming continuously; a "
      "**threshold** flips it discretely at some `alpha*` while the underlying "
      "logits move smoothly through the crossing.")
    A("")
    A("*Protocol, stated so it is checkable:* each `alpha` is run from its **own** "
      "iteration-0 tensor, recomputed by a clean forward pass under "
      "`W0 + alpha·ΔW`, so each `alpha` is a self-consistent frozen system. This "
      "is not EXP-001 section 1's same-iteration-0 protocol; the `alpha` = 1.0 "
      "row is therefore not identical to EXP-001's `closed` re-run.")
    A("")
    A("`comrade rank` is `comrade`'s position in the full 50257-token logit "
      "ordering at the settled state (1 = argmax); the logit columns are the raw "
      "logits of `comrade` and `prolet`. `gap` is `comrade − prolet`.")
    A("")
    A("| alpha | basin | lag-1 | comrade rank | comrade logit | prolet logit | "
      "gap (c−p) | settled |")
    A("|---|---|---|---|---|---|---|---|")
    for row in d2:
        gap = row["comrade"]["logit"] - row["prolet"]["logit"]
        A(f"| {row['alpha']:.2f} | `{row['basin']}` ({row['basin_token_id']}) | "
          f"{_fmt(row['final_lag1_cos'], '.6f')} | {row['comrade']['rank']} | "
          f"{_fmt(row['comrade']['logit'], '.4f')} | "
          f"{_fmt(row['prolet']['logit'], '.4f')} | {gap:+.4f} | "
          f"{'yes' if row['settled_last15'] else 'no'} |")
    A("")
    if d2s:
        astar = d2s["alpha_star_first_change"]
        gaps = [(row["alpha"], row["comrade"]["logit"] - row["prolet"]["logit"],
                 row["comrade"]["rank"]) for row in d2]

        def _ord(r):
            return {1: "1st", 2: "2nd", 3: "3rd"}.get(r, f"{r}th")

        if astar is not None:
            # the alpha just below the flip, for the interval statement
            below = None
            flip_basin = "comrade"
            for row in d2:
                if row["alpha"] == astar:
                    flip_basin = row["basin"]
                    break
                below = row["alpha"]
            A(f"**A bracket, and not a threshold this sweep establishes.** The "
              f"basin reads `{d2s['base_basin']}` at every sampled `alpha` ≤ "
              f"{_fmt(below, '.2f')} and `{flip_basin}` from "
              f"{_fmt(astar, '.2f')} on. The sweep steps in 0.25, so what the "
              f"grid gives is a **bracket** — the change of label lies in "
              f"({_fmt(below, '.2f')}, {_fmt(astar, '.2f')}] — and **not** a "
              f"threshold at {_fmt(astar, '.2f')}; reading the grid point "
              f"{_fmt(astar, '.2f')} as a measured crossing takes a sampling "
              f"artefact for a measurement, and is withdrawn. Whether it is a "
              f"threshold **at all** is **`not-established`** (`CLAIMS.md` "
              f"**C-27**): the underlying logit gap is smooth and monotone "
              f"through the crossing and only the argmax is discrete, and an "
              f"argmax changes discretely even under perfectly smooth motion, so "
              f"the discreteness is a property of the readout rather than "
              f"evidence about the dynamics.")
            A("")
            # Which side of the gap moves. The generator cannot assert the
            # direction for an arbitrary rerun, so it points at its own columns
            # and attributes the figures to the committed run they come from.
            A(f"Note also which side moves, and check it against the two logit "
              f"columns above: the crossing is driven mainly by `prolet`'s logit "
              f"**falling**, not by `comrade`'s rising — `comrade`'s own logit "
              f"barely moves across the bracket and slightly *falls*, so its "
              f"climb in rank is other tokens falling past it. In the committed "
              f"sweep that is `prolet` 16.95 → 16.07 across `alpha` = 0.00 → "
              f"0.75 against `comrade` 16.29 → 16.27. **[The reading this file "
              f"originally drew here — \"threshold, not smooth bias\", issue #32 "
              f"section 5's answer — is superseded.]**")
            A("")

            # smooth logits under the discrete flip
            ranks = ", ".join(_ord(r) for a, _, r in gaps if a <= astar)
            A("### Smooth logits, discrete attractor")
            A("")
            A(f"Underneath the discrete basin flip the logits move smoothly. "
              f"`comrade`'s rank climbs monotonically with `alpha` — "
              f"{ranks} across alpha = "
              f"{', '.join(_fmt(a, '.2f') for a, _, r in gaps if a <= astar)} — "
              f"and the `comrade − prolet` logit gap is a smooth, monotone "
              f"function of `alpha` that crosses zero exactly inside "
              f"({_fmt(below, '.2f')}, {_fmt(astar, '.2f')}]: "
              + ", ".join(f"{a:.2f}→{g:+.3f}" for a, g, r in gaps
                          if a <= max(astar, 1.0)) + ". "
              f"The argmax (the basin) is discrete; the logit it is the argmax "
              f"**of** is not. Smooth logits, discrete attractor — the signature "
              f"the result turns on.")
            A("")
        else:
            A(f"No basin flip occurred across the swept grid: the basin stays "
              f"`{d2s['base_basin']}` at every `alpha`. The `comrade − prolet` "
              f"logit gap moved "
              + ", ".join(f"{a:.2f}→{g:+.3f}" for a, g, r in gaps) + ".")
            A("")

        # the cascade
        cyc = [row for row in d2 if row["final_lag1_cos"] < 0.9]
        c0 = cyc[0] if cyc else None
        tail_basin = c0["basin"] if c0 else "—"
        A(f"### The `{d2s['base_basin']}` → `comrade` → `{tail_basin}` cascade")
        A("")
        if c0:
            A(f"Past `comrade` the sweep changes dynamical class — **[the word "
              f"\"bifurcates\" is the superseded reading; what is measured is a "
              f"class change, which is real and is ladder step 3]**. At alpha = "
              f"{c0['alpha']:.2f} the basin tips into `{c0['basin']}` and the "
              f"lag-1 cosine **collapses** from ~"
              f"{_fmt(max(r['final_lag1_cos'] for r in d2), '.4f')} (a fixed "
              f"point) to {_fmt(c0['final_lag1_cos'], '.4f')} — the fixed point "
              f"gives way to the period-2 cycle `{c0['basin']}` sits on in the "
              f"baseline. So the `alpha`-axis reads `{d2s['base_basin']}` (fixed "
              f"point) → `comrade` (fixed point) → `{c0['basin']}` (period-2): "
              f"**two transitions along one line, and neither is a created "
              f"attractor.** The first is a relabelling of the readout while the "
              f"settled branch moves smoothly — ladder step 2, and no longer "
              f"pending: T1.1 established it (C-26 `retired`, C-56 `supported`), "
              f"with C-68 the open discrepancy at `alpha` = 0.50. The second is a "
              f"genuine change of dynamical class into the **pre-existing** "
              f"`{c0['basin']}` orbit, which `{c0['basin']}` already holds in the "
              f"frozen baseline — step 3, **C-28**, `provisional` — and it sits "
              f"at 1.5× the ΔW the episode produced.")
        else:
            A(f"Across this grid the lag-1 cosine stays near 1.0 at every alpha "
              f"(min {_fmt(d2s['lag1_min'], '.4f')}); no period-2 cycle appears "
              f"in the swept range.")
        A("")

    # --- what this does / does not ----------------------------------------
    A("## What this establishes, and what it does not")
    A("")
    A("**Establishes:**")
    A("")
    if d2s and d1s:
        # Branch on the recorded outcome rather than assuming it. A sweep in which
        # no label changes leaves `alpha_star_first_change` None, and a sweep whose
        # seeded state never leaves leaves `stays_comrade` True -- both are real
        # possible results of this experiment, and asserting the outcome we happened
        # to get would make the generator lie on a rerun that got the other one.
        a_star = d2s.get("alpha_star_first_change")
        if a_star is None:
            A(f"- **No change of label anywhere in the `alpha`-sweep.** The basin "
              f"holds at `{d2s['base_basin']}` at every sampled alpha, so this run "
              f"establishes no bracket and nothing about a threshold.")
        else:
            A(f"- **A bracket on the `alpha`-sweep's change of label.** The basin "
              f"holds at `{d2s['base_basin']}` and reads `comrade` by alpha\\* = "
              f"{_fmt(a_star, '.2f')}; the sweep steps in "
              f"0.25, so what is established is the **bracket** — the single grid "
              f"step ending at {_fmt(a_star, '.2f')} — and "
              f"not a threshold at "
              f"{_fmt(a_star, '.2f')} itself.")
        if d1s.get("stays_comrade"):
            A(f"- **D1's measurement.** Seeded at the episode's `comrade` state, the "
              f"**frozen** W0 map's readout does not leave `comrade` within "
              f"{d1s['n_steps']} steps. That is a statement about the readout over a "
              f"finite horizon and not a demonstration that the state is fixed — see "
              f"§D1 and **C-56**.")
        else:
            A(f"- **D1's measurement.** Seeded at the episode's `comrade` state, the "
              f"**frozen** W0 map does not hold it: the readout leaves at iteration "
              f"{d1s['first_leave']['iter'] if d1s['first_leave'] else '--'} "
              f"and settles at `{d1s['final_basin']}`, smoothly and monotonically. "
              f"`comrade` is not an attractor of the original frozen map. What that "
              f"supports is **displacement**, not creation — see §D1 and **C-56**.")
    A("")
    # These three bullets are kept in place, marked superseded, rather than
    # deleted: the repo's convention is that the record of what it used to
    # believe is part of the evidence (CLAIMS.md, "retired rows are never
    # deleted"). Do not restore them to the Establishes list.
    A("**Superseded readings, kept in place as the record:**")
    A("")
    if d2s and d1s:
        A(f"- **[SUPERSEDED] Closes issue #32 section 5.** This file answered "
          f"*smooth bias or a threshold?* with **a threshold** at alpha\\* = "
          f"{_fmt(d2s['alpha_star_first_change'], '.2f')}. That reading is "
          f"`CLAIMS.md` **C-27**, `not-established`, and is **not quotable**: an "
          f"argmax changes discretely even under perfectly smooth motion, and "
          f"here the underlying logit gap is smooth and monotone through the "
          f"crossing — driven mainly by `prolet`'s logit falling, 16.95 → 16.07, "
          f"rather than `comrade`'s rising, 16.29 → 16.27. The section is not "
          f"closed; what survives is the bracket above.")
        A(f"- **[SUPERSEDED] Answers issue #25's step-3-vs-4.** This file "
          f"concluded `comrade` is a **created attractor** (step 4, a "
          f"bifurcation) rather than a boundary move (step 3). **C-26 is "
          f"`retired`** — not merely unsupported but **contradicted**, by T1.1, "
          f"the test this file named as deciding: seeded from the original frozen "
          f"`{d1s['final_basin']}` state, the edited map does not hold it, so the "
          f"two resting states do not coexist and the edit **displaces** the "
          f"single attractor (**C-56**, corroborated by T1.2/**C-52**'s "
          f"hysteresis-free retrace). The `A04`→`Divine` class change *is* step 3 "
          f"and is not affected. Both discriminators do agree, but not on "
          f"creation: D1 returns the same verdict for **any** ΔW that displaces "
          f"the fixed point, and D2's discrete change of label is an argmax "
          f"crossing a ridge, which is **not** a bifurcation.")
        A("- **[SUPERSEDED] The two discriminators are independent and agree.** "
          "They agree, but on a reading neither can establish — D1 iterates "
          "from the settled state under the *unmodified* map; D2 installs "
          "*fractional* ΔW and reads the settled basin from a fresh start. "
          "Neither is the other restated.")
    A("")
    A("**Does not / caveats:**")
    A("")
    A("- **Reproduced, not loaded.** No raw closed-loop state or ΔW is persisted "
      "in the repo, so this run regenerates them from the frozen episode. The "
      "weight-space anchors match EXP-001 to ~9 figures, but the closed-loop "
      "state norm carries a ~0.1%-class float drift (reported in the fidelity "
      "table); the basin label is invariant to it.")
    A("- **The per-`alpha` `initial_state` protocol.** Each `alpha` is run from "
      "its own iteration-0 tensor recomputed under `W0 + alpha·ΔW` — a "
      "self-consistent frozen system per `alpha`, not the same iteration-0 tensor "
      "across `alpha`. Stated plainly so the sweep is checkable and so the "
      "`alpha` = 1.0 row is not mistaken for EXP-001's `closed` re-run.")
    if d2s:
        A(f"- **The change of label is grid-localized, and \"alpha\\*\" overstates "
          f"it.** It is pinned only to the sweep grid — it lies in the bracket "
          f"reported above, not at a sharper edge — and the 0.25 grid does not "
          f"resolve where inside that interval the crossing sits, nor whether the "
          f"crossing is a threshold at all (**C-27**, `not-established`). T1.2's "
          f"finer continuation sweep, stepping in 0.10, places its own crossing "
          f"near `alpha` ≈ 0.45; that is a different seeding scheme and therefore "
          f"a different measurement, not a contradiction, but the two are not "
          f"interchangeable (**C-52**, and **C-68** for where they disagree "
          f"outright).")
        A(f"- **A discrepancy this file owns half of.** At `alpha` = 0.50 this "
          f"sweep and T1.2 reach **different settled words from different "
          f"initial states under identical weights** — this sweep from that "
          f"`alpha`'s own iteration-0 tensor (`prolet` in the committed run), "
          f"T1.2's continuation sweep from a settled state, where it reaches "
          f"`comrade` in all four of its arms, `robust_up` included. Same site, "
          f"rule, eta, prompt, steps and ΔW. That is **C-68**, `provisional` and "
          f"**unresolved**; the seeding is the candidate explanation and the "
          f"deciding test (a basin-of-attraction probe at `alpha` = 0.50, several "
          f"initial states, convergence criterion fixed in advance) has not run. "
          f"Until it does, what each row records is the settled word reached from "
          f"**that `alpha`'s own iteration-0 tensor**, and this sweep may not be "
          f"described as showing a single attractor at every `alpha`. **C-56 is "
          f"untouched** by it: T1.1 tested `alpha` = 1.00 directly.")
    A("- **One cell.** One prompt (`A01_physics`), one site (`blocks.6.mlp`), one "
      "eta, one ceiling, one seed. `hebb` has no stochastic term and the model is "
      "frozen and single-threaded, so a seed is a single deterministic run, not a "
      "sample; the map's one-prompt-one-site caveats carry over unchanged.")
    A("- **No task, no loss, no target.** As with EXP-001, ΔW is what the rule "
      "produces on this activation distribution; this file measures the "
      "dynamical-systems character of the result, not a trained objective.")
    A("")

    # --- provenance -------------------------------------------------------
    A("## Provenance")
    A("")
    A(f"Reproduced episode + D1 ({D1_N} steps) + D2 ({len(ALPHAS)} alphas × "
      f"{D2_N} steps), {meta.get('total_seconds', 0) / 60:.1f} CPU-minutes, one "
      f"torch thread, deterministic.")
    A("")
    A("```json")
    A(json.dumps(meta, indent=2, sort_keys=True))
    A("```")
    A("")
    A("Raw records — the reproduce/fidelity block, every sampled D1 iterate and "
      "every per-`alpha` D2 row: "
      "`experiments/output_basin_bifurcation/basin_bifurcation.jsonl`.")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _build_meta(args) -> dict:
    meta = {
        "issues": [25, 32],
        "model": MODEL_NAME,
        "site": SITE,
        "prompt_id": PROMPT_ID,
        "prompt": PROMPT,
        "mode": MODE,
        "eta": ETA,
        "eta_provenance": ("D * ||W0||_F / (N_STEPS * U_ref[hebb]) with D=1.8e-2, "
                           "U_ref=350, ||W0||_F=164.854 -- the step-size map's "
                           "anchor, recomputed not rounded (same as EXP-001)"),
        "cadence": CADENCE,
        "max_delta_frac": MAX_DELTA_FRAC,
        "seed": SEED,
        "layer_start": LAYER_START,
        "layer_end": LAYER_END,
        "n_episode": N_EPISODE,
        "d1_n_steps": D1_N,
        "d2_n_steps": D2_N,
        "alphas": list(ALPHAS),
        "device": "cpu",
        "dtype": "float32",
        "norms_dtype": "float64",
        "torch_threads": args.threads,
        "shards": 1,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "repo_rev": bb._git_rev(REPO_ROOT),
    }
    try:
        from importlib.metadata import version as _v
        meta["transformer_lens_version"] = _v("transformer-lens")
    except Exception:
        meta["transformer_lens_version"] = "unknown"
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=TORCH_THREADS)
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--report", type=str, default=str(REPORT_PATH))
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the markdown from the recorded JSONL + meta, no run")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "basin_bifurcation.jsonl"
    meta_path = out_dir / "meta.json"

    if args.report_only:
        records = bb.read_jsonl(jsonl)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        Path(args.report).write_text(build_report(records, meta), encoding="utf-8")
        print(f"[report] {args.report} ({len(records)} records)")
        return 0

    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    meta = _build_meta(args)
    meta["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    from transformer_lens import HookedTransformer
    t_load = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    model.eval()
    model.requires_grad_(False)
    print(f"[model] {MODEL_NAME} loaded in {time.time() - t_load:.1f}s "
          f"(threads={args.threads})", flush=True)
    assert model.cfg.n_layers - 1 == LAYER_END

    records, meta = run(model, out_dir, meta)

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["wall_clock_seconds"] = meta.get("total_seconds")

    bb.write_jsonl(jsonl, records)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    Path(args.report).write_text(build_report(records, meta), encoding="utf-8")

    print(f"[done] {len(records)} records → {jsonl.name}, meta.json; "
          f"report → {Path(args.report).name}; "
          f"{meta.get('total_seconds', 0) / 60:.1f} min "
          f"(episode_ok={meta.get('episode_ok')})", flush=True)
    return 0 if meta.get("episode_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
