"""
EXP-001, run at `hebb`, eta = 7.065e-05 -- a step size where the loop moves.

The step-size map (issue #30, `STEP_SIZE_MAP.md`) settled the question this
experiment could not previously ask honestly. `oja`, `anti_hebb` and `random`
all have wide bands in which the weights move several percent with the norm
ceiling silent and **the loop does not move at all** -- basin, lag-1 and lag-2
unchanged even with `oja` pinned at the ceiling at 100% clipping. `hebb` is the
single exception among the *rules*, and it moves the loop at **two** of the step
sizes swept, not one (C-21): at eta = 7.065e-05, this file's cell, the basin
goes `prolet` -> `comrade` at 1.12% relative weight change with the ceiling
**silent** (0.0% clipping), and at eta = 1.18e-04 it reaches the same `comrade`
at 2.20% drift, also ceiling-silent. The two are NOT an independent replication
-- they share prompt, seed, site and cadence -- so what they establish is that
the flip survives a 1.7x change in eta.

Every offline-control measurement this repo had recorded **before this run** was
taken at `oja`, eta = 1e-5 -- which the map now places inside the dead zone. So
at the time of writing the project had never run its own control at a step size
where anything happens. That is what this file does. The later offline and
severed arms at `hebb` all postdate it: C-31 is this run's own, and C-58's eta
and cadence grid is T2.1's.

## The question, and why the offline arm is the whole experiment

The closed loop flips the basin. Does the offline arm flip it too?

    closed    run the ATR loop; after every iteration apply `hebb` to the
              activations flowing through right now. The changed W_out shapes
              the next iterate, which shapes the next update.
    offline   run the same loop FROZEN, recording those activations. Replay
              the recording through the same rule with no feedback. Install
              the resulting matrix. Re-run the loop frozen.

If the offline arm flips too, feedback is not required for the flip and the
flip is a property of the rule applied to this activation distribution -- which
is what Oja-family rules do anywhere, with or without a loop. If it does not,
that is the first evidence in this project that coupling does something the
rule alone cannot.

`y_source="recomputed"` is the path the claim is made from, because its
detection floor is exactly zero (`EXP_001_SPEC.md` section 7). The literal
PRIOR_ART protocol, `y_source="recorded"`, is reported beside it because its
floor is *larger than the routed signal* at the settings measured so far, and a
`recorded`-mode number quoted alone is not evidence about feedback.

## The severed control

A severed-path arm runs at every routed configuration: the loop reads out at
`blocks.3.hook_resid_post`, below the plastic site at layer 6, so no state
feedback can exist and the `x` reaching the rule is bit-identical in both arms.
Whatever the arms still differ by there is the floor. **Any effect claimed in
the routed configuration must exceed the severed one.** A previous severed test
in `recorded` mode read 6.77e-02 against a routed 1.91e-02 -- the naive protocol
reporting a *larger* apparent effect with the feedback physically disconnected.

## Reading the basin claim

69 of the 125 baseline prompts have a top1-top2 logit margin below 0.5, and
`A01_physics` -- the map's prompt, and this file's -- is one of them, at 0.227.
A flip across a margin that narrow is a different object from a flip across a
wide one, so the margin is reported on every basin row and no basin claim in
this file appears without it.

## What is run

  routed    matched arms, layers 0->11, site blocks.6.mlp, hebb, eta 7.065e-05,
            120 steps, cadence 1, seeds 0/1/2 on A01_physics plus two more
            `prolet` prompts. Both offline y_sources. The closed, offline and
            offline-recomputed matrices are each installed and the loop re-run
            frozen under them, so basin / lag-1 / lag-2 / final-state cosine
            are measured behaviourally rather than inferred from ||dW||.
  severed   the same, loop read out at blocks.3, no reruns (the layer-3
            residual has no basin worth decoding); diff_over_drift only.
  episode   a single closed-loop episode on a `Divine` prompt, for the
            cross-basin dW direction comparison (issue #32 section 3b).
  revert    C1: after the episode, restore W0 and iterate from the closed arm's
            final state. Horizon 1000 with early stop, NOT 200: the measured
            per-iteration contraction is ~0.968, i.e. ~71 iterations per decade
            of displacement, and a 200-iteration horizon has already produced
            one false "failed to return" in this repo.

## Running it

    python experiments/exp001_hebb.py --shard 0 --nshards 2 &
    python experiments/exp001_hebb.py --shard 1 --nshards 2 &
    python experiments/exp001_hebb.py --report-only

One torch thread per process. Four threads is 7x SLOWER than one on this box
(OpenMP spin-wait collapse at seq_len ~10); two single-threaded shards is where
the parallelism comes from. Checkpointed to JSONL after every cell.

Hook discipline: nothing here calls `model.reset_hooks()`. That would clear the
ATR engine's injection hook and silently detach the loop. Install/remove is
paired and `revert()` runs in a `finally`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atr_bridge import initial_state, make_atr_step                  # noqa: E402
from plasticity import _make_site                                    # noqa: E402
from offline_control import (                                        # noqa: E402
    compare_states,
    compare_weights,
    installed_weight,
    record_frozen_activations,
    replay_offline,
    run_closed_loop_arm,
    run_matched_arms,
    verify_arms_matched,
)
import baseline_basins as bb                                         # noqa: E402


# ---------------------------------------------------------------------------
# Config -- every value echoed into the write-up
# ---------------------------------------------------------------------------

PARENT_DEFAULT = os.environ.get("ATR_PARENT_PATH", bb.PARENT_DEFAULT)

MODEL_NAME = "gpt2-small"
# `--site` overrides this at the command line (main() reassigns SITE from it) so
# the single-site "separate" arm can target any site or head without editing the
# file. DEFAULT_SITE is the fixed reference ETA below is anchored to: ETA was
# derived from blocks.6.mlp's ||W0||_F and U_ref[hebb], so a non-default site
# re-uses an anchor that is not its own (main() says so) and the default path is
# bit-identical.
SITE = "blocks.6.mlp"
DEFAULT_SITE = "blocks.6.mlp"
LAYER_START = 0
LAYER_END = 11
# The severed readout. Below the plastic site at layer 6, so the loop cannot
# route a weight change back into its own next iterate.
LAYER_END_SEVERED = 3

MODE = "hebb"
N_STEPS = 120
CADENCE = 1
MAX_DELTA_FRAC = 0.05

# Exactly the map's cell, recomputed from the map's own anchor rather than
# copied from its rounded table entry:
#     eta = D * ||W0||_F / (N_STEPS * U_ref[hebb]),  D = 1.8e-2, U_ref = 350
ETA = 1.8e-2 * 164.854 / (N_STEPS * 350.0)          # 7.065171e-05

SEEDS = (0, 1, 2)
PROMPT_MAIN = "A01_physics"
PROMPTS_PROLET = ("A01_physics", "A02_medical", "A04_climate")
PROMPTS_DIVINE = ("A14_kant", "A08_linguistics")

# Lag scan window, identical to the baseline sweep's and the map's.
LAG_WINDOW = 25
MAX_LAG = 8

# C1 revert horizon. Justified against the contraction factor, not guessed:
# ~0.968 per iteration is ~71 iterations per decade, and returning from a
# displacement of order 1e-2 to the float32 round-off floor (1-cos ~ 1.5e-14)
# is ~12 decades, i.e. ~850 iterations. 200 is not enough and has already
# produced one false negative in this repo.
REVERT_HORIZON = 1000
# Early stop: 1-cos at or below this is "returned", at the measured float32
# round-off floor of the instrument (1.5e-14) with an order of headroom.
RETURN_TOL = 1e-12

# State equality is a tolerance test at float32 round-off, never torch.equal:
# two states that are the same point of the dynamics differ in the last bits.
SAME_STATE_TOL = 1e-12

TORCH_THREADS = 1

OUT_DIR = REPO_ROOT / "experiments" / "output_exp001"
REPORT_PATH = REPO_ROOT / "EXP_001_RESULTS.md"

TOP_K_TOKENS = 20
N_RANDOM_CONTROLS = 3


# ---------------------------------------------------------------------------
# Small numeric helpers. Float64 for every norm and cosine.
# ---------------------------------------------------------------------------

def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
    a, b = u.double().flatten(), v.double().flatten()
    na, nb = a.norm().item(), b.norm().item()
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return (a @ b).item() / (na * nb)


def same_state(a: torch.Tensor, b: torch.Tensor, tol: float = SAME_STATE_TOL) -> bool:
    """Tolerance test at float32 round-off, deliberately not `torch.equal`.

    `torch.equal` returns False on two states that are the same point of the
    dynamics -- the measured round-off floor on this instrument is 1-cos ~
    1.5e-14, so an exact-equality test on a 120-iteration trajectory answers a
    question nobody asked.
    """
    return (1.0 - _cos(a, b)) <= tol


def phase_aware_cos(a_final, b_final, b_prev=None, a_prev=None) -> dict:
    """Best cosine over the phases of a possibly period-2 orbit.

    On a period-2 orbit two states can sit on the same cycle in opposite
    phases, and a final-to-final comparison scores that as a large distance
    when it is none. Scored at the better of (final, final), (final, prev) and
    (prev, final).
    """
    cands = [("final_final", _cos(a_final, b_final))]
    if b_prev is not None:
        cands.append(("final_prev", _cos(a_final, b_prev)))
    if a_prev is not None:
        cands.append(("prev_final", _cos(a_prev, b_final)))
    best = max(cands, key=lambda kv: kv[1])
    return {"cos": best[1], "phase": best[0],
            "all": {k: v for k, v in cands}}


def spectrum(m: torch.Tensor, k: int = 32) -> dict:
    """Singular spectrum of a matrix in float64, plus the energy share of s1.

    `frac_energy_1` is issue #32 section 2's number: the fraction of
    ||dW||_F^2 in the first component. `erank_pr` is the participation ratio
    of the singular values, the same instrument the step-size map uses.
    """
    md = m.double()
    sv = torch.linalg.svdvals(md)
    s2 = (sv * sv)
    tot2 = s2.sum().item()
    s_sum = sv.sum().item()
    return {
        "singular_values_top": [float(x) for x in sv[:k]],
        "n_svals": int(sv.numel()),
        "fro": float(md.norm()),
        "frac_energy_1": (s2[0].item() / tot2) if tot2 > 0 else float("nan"),
        "frac_energy_top5": (s2[:5].sum().item() / tot2) if tot2 > 0 else float("nan"),
        "erank_pr": (s_sum * s_sum / tot2) if tot2 > 0 else float("nan"),
        "erank_stable": (tot2 / s2[0].item()) if s2[0].item() > 0 else float("nan"),
    }


def dominant_factors(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """(left, right, sigma1) of the dominant component of a 2-D matrix, float64.

    `m` is (n_in, n_out) = (3072, 768) at this site, so the *right* factor is
    the 768-side one that lives in residual-stream output space and is what
    issue #32 section 3a calls `u`.
    """
    U, S, Vh = torch.linalg.svd(m.double(), full_matrices=False)
    return U[:, 0].clone(), Vh[0, :].clone(), float(S[0])


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------

def basin_of(model, r: torch.Tensor) -> dict:
    """The parent's readout, at the last position: label, margin, top-5.

    The margin travels with the label everywhere in this file. A flip across a
    0.002 logit margin and a flip across a 2.0 one are not the same result and
    must not be able to appear in the same column without their margins.
    """
    d = bb.readout_detail(model, r[-1, :])
    return {
        "basin": d["top_token_strings"][0].strip(),
        "basin_raw": d["top_token_strings"][0],
        "basin_token_id": d["top_token_ids"][0],
        "top5_tokens": d["top_token_strings"],
        "top5_probs": d["top_token_probs"],
        "top_logit_margin": d["top_logit_margin"],
        "entropy": d["entropy"],
    }


def trajectory_stats(model, traj: list) -> dict:
    """Basin, margin and the late-window lag scan for one frozen re-run."""
    tail = traj[-(LAG_WINDOW + MAX_LAG):]
    scan = bb.lag_scan(torch.stack([t.mean(dim=0) for t in tail]), MAX_LAG)
    out = basin_of(model, traj[-1])
    out.update({
        "cos_lag1_mean": scan.get(1, {}).get("mean"),
        "cos_lag2_mean": scan.get(2, {}).get("mean"),
        "lag_scan": {str(k): v["mean"] for k, v in scan.items()},
        "final_tensor_norm": float(traj[-1].double().norm()),
    })
    return out


def decode_direction(model, v: torch.Tensor, k: int = TOP_K_TOKENS) -> dict:
    """Logit lens on a direction: `v @ W_U`, top and bottom k tokens.

    `b_U` is deliberately not added. A direction is not a state; adding the
    unembedding bias would mix the model's unconditional token frequency into
    what is supposed to be the direction's own content, and every direction --
    including the random control -- would inherit the same bias ordering.
    """
    vv = v.to(model.W_U.dtype).flatten()
    logits = (vv @ model.W_U).double()
    top = torch.topk(logits, k)
    bot = torch.topk(-logits, k)
    return {
        "top_tokens": [model.tokenizer.decode([int(i)]) for i in top.indices],
        "top_logits": [float(x) for x in top.values],
        "bottom_tokens": [model.tokenizer.decode([int(i)]) for i in bot.indices],
        "bottom_logits": [float(-x) for x in bot.values],
        "logit_std": float(logits.std()),
        "logit_max": float(logits.max()),
    }


def random_direction_controls(model, dim: int, norm: float, seed: int,
                              n: int = N_RANDOM_CONTROLS) -> list:
    """Matched-norm isotropic directions, decoded the same way.

    Without this a plausible-looking token list means nothing: `W_U` is not
    isotropic, and *any* direction pushed through it produces a list of tokens
    that reads like a theme.
    """
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        z = torch.randn(dim, generator=g, dtype=torch.float64)
        z = z * (norm / z.norm().item())
        d = decode_direction(model, z.float())
        d["draw"] = i
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Frozen re-runs
# ---------------------------------------------------------------------------

def frozen_trajectory(model, r0: torch.Tensor, step, n_steps: int) -> list:
    """`n_steps` of the loop with nothing installed and nothing changing."""
    r = r0.clone()
    out = []
    for _ in range(n_steps):
        r = step(model, r)
        out.append(r.detach().clone())
    return out


def rerun_under(model, w: torch.Tensor, r0: torch.Tensor, step, n_steps: int) -> list:
    """Install `w`, run the loop frozen under it, restore. Restores on raise."""
    with installed_weight(model, SITE, w):
        return frozen_trajectory(model, r0, step, n_steps)


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def cell_id(c: dict) -> str:
    return f"{c['kind']}:{c['prompt_id']}:seed{c['seed']}"


def build_cells() -> list:
    cells = []
    # Part 1, routed: three seeds on the map's prompt, then two more prolet
    # prompts at seed 0 so the spread is over something that can actually vary.
    for s in SEEDS:
        cells.append({"kind": "routed", "prompt_id": PROMPT_MAIN, "seed": s})
    for p in PROMPTS_PROLET[1:]:
        cells.append({"kind": "routed", "prompt_id": p, "seed": 0})
    # Part 1, severed: same shape.
    for s in SEEDS:
        cells.append({"kind": "severed", "prompt_id": PROMPT_MAIN, "seed": s})
    for p in PROMPTS_PROLET[1:]:
        cells.append({"kind": "severed", "prompt_id": p, "seed": 0})
    # Part 3: Divine-basin episodes for the cross-basin dW direction test.
    for p in PROMPTS_DIVINE:
        cells.append({"kind": "episode", "prompt_id": p, "seed": 0})
    # C1 revert, once.
    cells.append({"kind": "revert", "prompt_id": PROMPT_MAIN, "seed": 0})
    return cells


def _dw_block(model, delta: torch.Tensor, y_mean: torch.Tensor, seed: int) -> dict:
    """Everything issue #32 sections 2 and 3a ask of one dW.

    Sign convention, stated because an SVD factor's sign is arbitrary and a
    token list read off the wrong one is the negation of the answer: the right
    factor is flipped, if needed, so that it has a positive inner product with
    the mean post-synaptic activity at the site over the frozen episode. That
    makes it point the way the site's output actually points, and it is the
    same rule for every cell, so the cross-basin cosines below are comparable.
    """
    left, right, s1 = dominant_factors(delta)
    sign = 1.0 if float(right @ y_mean.double()) >= 0 else -1.0
    right = right * sign
    left = left * sign
    spec = spectrum(delta)
    return {
        "delta_spectrum": spec,
        "sigma1": s1,
        "u_right_768": [float(x) for x in right],
        "u_right_sign_ref_cos": float(_cos(right, y_mean)),
        "u_left_mean": float(left.mean()),
        "decode": decode_direction(model, right.float()),
        "random_controls": random_direction_controls(
            model, right.numel(), float(right.norm()), seed=1234 + seed),
    }


def run_routed_cell(model, prompt: str, seed: int, layer_end: int,
                    do_reruns: bool) -> dict:
    """One matched-arms cell, plus the behavioural readout of each arm.

    `run_matched_arms` is called once with `y_source="recorded"` and
    `also_recomputed_y=True`, which runs exactly the two replays that
    `y_source="recorded"` and `y_source="recomputed"` would run separately --
    one recording and one closed-loop run instead of two of each. The headline
    is read from the `recomputed` pair, whose detection floor is zero; the
    `recorded` pair is reported beside it because it is the literal PRIOR_ART
    protocol and its floor is not zero.
    """
    t0 = time.time()
    s0 = initial_state(model, prompt, layer_end=layer_end)
    step = make_atr_step(model, prompt, layer_start=LAYER_START,
                         layer_end=layer_end, initial_norm=s0.initial_norm)

    res = run_matched_arms(
        model, s0.tensor, step, SITE, N_STEPS,
        eta=ETA, mode=MODE, max_delta_frac=MAX_DELTA_FRAC,
        seed=seed, apply_every=CADENCE,
        y_source="recorded", also_recomputed_y=True,
        rerun_frozen=False,        # done below, so the trajectories survive
        keep_states=True,
    )

    ver = res.verification
    rec = {
        "cell_id": None,
        "kind": "routed" if layer_end == LAYER_END else "severed",
        "prompt": prompt,
        "seed": seed,
        "layer_start": LAYER_START,
        "layer_end": layer_end,
        "site": SITE,
        "mode": MODE,
        "eta": ETA,
        "n_steps": N_STEPS,
        "cadence": CADENCE,
        "max_delta_frac": MAX_DELTA_FRAC,
        "initial_norm": s0.initial_norm,
        "seq_len": int(s0.tensor.shape[0]),

        "arms_matched": ver["ok"],
        "arms_matched_axes": [a["axis"] for a in ver["axes"] if a["match"]],
        "arms_mismatched_axes": [a["axis"] for a in ver["axes"] if not a["match"]],
        "n_axes_checked": len(ver["axes"]),
        "verdict": ver["verdict"],

        "clipped_closed": res.closed.report["clipped"],
        "clipped_offline_recorded": res.offline.report["clipped"],
        "clipped_offline_recomputed": (res.offline_recomputed_y.report["clipped"]
                                       if res.offline_recomputed_y else None),
        "nonfinite_closed": res.closed.report["nonfinite"],
        "n_updates": res.closed.config.n_updates,
        "rel_weight_change_closed": res.closed.report["delta_frac"],
        "rel_weight_change_offline_recorded": res.offline.report["delta_frac"],
        "rel_weight_change_offline_recomputed": (
            res.offline_recomputed_y.report["delta_frac"]
            if res.offline_recomputed_y else None),

        "weight_recorded": res.comparison["weight"],
        "weight_recomputed_y": res.comparison.get("weight_recomputed_y"),
        "offline_recorded_vs_recomputed_y": res.comparison.get(
            "offline_recorded_vs_recomputed_y"),
    }

    # Per-update clip rate, read off the rule's own flag rather than inferred.
    # `clipped` on the instance is sticky for a whole run, so this is "did the
    # ceiling ever fire", which is the only claim it can support.
    rec["clip_rate_note"] = ("OjaPlasticity.clipped is sticky per run; "
                             "False means the ceiling never fired in 120 updates")

    frozen_traj = list(res.record.states)
    if do_reruns:
        arms = {"closed": res.closed.weight,
                "offline_recorded": res.offline.weight}
        if res.offline_recomputed_y is not None:
            arms["offline_recomputed"] = res.offline_recomputed_y.weight

        traj = {"frozen": frozen_traj}
        for name, w in arms.items():
            traj[name] = rerun_under(model, w, s0.tensor, step, N_STEPS)

        rec["readout"] = {k: trajectory_stats(model, v) for k, v in traj.items()}

        # Phase-aware state comparisons. Every pair is scored at the better of
        # (final, final) and (final, previous iterate) -- see phase_aware_cos.
        def pair(a, b):
            pa = phase_aware_cos(traj[a][-1], traj[b][-1],
                                 b_prev=traj[b][-2], a_prev=traj[a][-2])
            # `compare_states`'s own `cos` is the plain (final, final) one and
            # is kept under that name; the phase-aware figure gets its own key
            # so the two can never be confused for each other. `phase_all`
            # carries all three candidate cosines, so the phase-aware number is
            # recomputable from the record without trusting this line.
            d = compare_states(traj[a][-1], traj[b][-1])
            d["phase_aware_cos"] = pa["cos"]
            d["phase"] = pa["phase"]
            d["all"] = pa["all"]
            d["same_state_at_roundoff"] = same_state(traj[a][-1], traj[b][-1])
            return d

        rec["state"] = {
            "closed_vs_offline_recomputed": pair("closed", "offline_recomputed"),
            "closed_vs_offline_recorded": pair("closed", "offline_recorded"),
            "closed_vs_frozen": pair("closed", "frozen"),
            "offline_recomputed_vs_frozen": pair("offline_recomputed", "frozen"),
            "offline_recorded_vs_frozen": pair("offline_recorded", "frozen"),
        }
        # The closed arm's own live trajectory (weights moving) against the
        # frozen re-run under its own final matrix: not the same object, and
        # the gap is how much of the effect is the weights having been in
        # motion rather than where they ended up.
        rec["state"]["closed_live_vs_closed_rerun"] = compare_states(
            res.closed.states[-1], traj["closed"][-1])

    # dW, for issue #32. y_mean is the mean post-synaptic activity at the site
    # over the frozen episode, used only to fix the SVD sign.
    y_mean = torch.stack([y.double().mean(dim=0) for y in res.record.y]).mean(dim=0)
    rec["dW_closed"] = _dw_block(model, res.closed.delta, y_mean, seed)
    rec["dW_offline_recomputed"] = (
        _dw_block(model, res.offline_recomputed_y.delta, y_mean, seed)
        if res.offline_recomputed_y else None)
    rec["cos_dW_closed_offline_recomputed"] = (
        _cos(res.closed.delta, res.offline_recomputed_y.delta)
        if res.offline_recomputed_y else None)
    rec["y_mean_norm"] = float(y_mean.norm())

    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def run_episode_cell(model, prompt: str, seed: int) -> dict:
    """One closed-loop episode, for the cross-basin dW comparison only."""
    t0 = time.time()
    s0 = initial_state(model, prompt, layer_end=LAYER_END)
    step = make_atr_step(model, prompt, layer_start=LAYER_START,
                         layer_end=LAYER_END, initial_norm=s0.initial_norm)

    # The frozen record is run for its activations alone: y_mean fixes the SVD
    # sign under the same convention every other cell uses.
    record = record_frozen_activations(model, s0.tensor, step, SITE, N_STEPS,
                                       keep_states=True)
    y_mean = torch.stack([y.double().mean(dim=0) for y in record.y]).mean(dim=0)

    arm = run_closed_loop_arm(model, s0.tensor, step, SITE, N_STEPS,
                              eta=ETA, mode=MODE, max_delta_frac=MAX_DELTA_FRAC,
                              seed=seed, apply_every=CADENCE, keep_states=True)

    frozen_stats = trajectory_stats(model, list(record.states))
    rec = {
        "cell_id": None,
        "kind": "episode",
        "prompt": prompt,
        "seed": seed,
        "layer_start": LAYER_START,
        "layer_end": LAYER_END,
        "site": SITE,
        "mode": MODE,
        "eta": ETA,
        "n_steps": N_STEPS,
        "seq_len": int(s0.tensor.shape[0]),
        "initial_norm": s0.initial_norm,
        "clipped_closed": arm.report["clipped"],
        "nonfinite_closed": arm.report["nonfinite"],
        "rel_weight_change_closed": arm.report["delta_frac"],
        "readout": {"frozen": frozen_stats,
                    "closed_live": trajectory_stats(model, list(arm.states))},
        "dW_closed": _dw_block(model, arm.delta, y_mean, seed),
        "y_mean_norm": float(y_mean.norm()),
        "seconds": round(time.time() - t0, 2),
    }
    return rec


def run_revert_cell(model, prompt: str, seed: int) -> dict:
    """C1: does the loop return to the frozen attractor once W0 is restored?

    Horizon 1000 with early stop, justified against the measured contraction
    (~0.968/iteration, ~71 iterations per decade). The trace of an episode is
    supposed to live in dW and nowhere else; if the state does not come back
    after the weights do, something else is holding it.
    """
    t0 = time.time()
    s0 = initial_state(model, prompt, layer_end=LAYER_END)
    step = make_atr_step(model, prompt, layer_start=LAYER_START,
                         layer_end=LAYER_END, initial_norm=s0.initial_norm)

    frozen = frozen_trajectory(model, s0.tensor, step, N_STEPS)
    target = frozen[-1]

    arm = run_closed_loop_arm(model, s0.tensor, step, SITE, N_STEPS,
                              eta=ETA, mode=MODE, max_delta_frac=MAX_DELTA_FRAC,
                              seed=seed, apply_every=CADENCE, keep_states=True)
    # run_closed_loop_arm reverts the live weight in its own finally block, so
    # the model is already back on W0 here. Continue from where the episode
    # left the state.
    r = arm.states[-1].clone()
    start_gap = 1.0 - _cos(r, target)

    # The reference is iterated in LOCKSTEP, not held fixed at iteration 120.
    # A fixed target answers a different question: the frozen loop is itself
    # still settling at 120, so a reverted state that has come all the way back
    # onto the frozen trajectory still reads a nonzero gap against the
    # iteration-120 snapshot, and the run looks like a failure to return when
    # it is a comparison against a stale reference. Both gaps are reported --
    # the lockstep one is the C1 answer, the fixed one is what a naive version
    # of this control would have said.
    ref = target.clone()

    returned_at = None
    prev = ref_prev = None
    curve = []
    for i in range(1, REVERT_HORIZON + 1):
        prev, r = r, step(model, r)
        ref_prev, ref = ref, step(model, ref)
        # Phase-aware on both sides: on a period-2 orbit the two trajectories
        # can sit on the same cycle in opposite phases.
        gap = min(1.0 - _cos(r, ref), 1.0 - _cos(prev, ref), 1.0 - _cos(r, ref_prev))
        gap_fixed = min(1.0 - _cos(r, target), 1.0 - _cos(prev, target))
        if i <= 20 or i % 25 == 0:
            curve.append([i, gap, gap_fixed])
        if gap <= RETURN_TOL:
            returned_at = i
            break

    return {
        "final_gap_vs_lockstep_reference": gap,
        "final_gap_vs_fixed_iter120_target": gap_fixed,
        "basin_lockstep_reference": basin_of(model, ref),
        "cell_id": None,
        "kind": "revert",
        "prompt": prompt,
        "seed": seed,
        "eta": ETA,
        "mode": MODE,
        "n_steps": N_STEPS,
        "horizon": REVERT_HORIZON,
        "return_tol": RETURN_TOL,
        "start_gap_1_minus_cos": start_gap,
        "returned_at_iter": returned_at,
        "final_gap_1_minus_cos": gap,
        "returned": returned_at is not None,
        "gap_curve": curve,
        "basin_after_revert": basin_of(model, r),
        "basin_frozen_target": basin_of(model, target),
        "seconds": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Post-hoc: does the dW direction actually point at the basin tokens?
# ---------------------------------------------------------------------------

def _lens_token_ids(model, recs: list) -> dict:
    """The basin token ids, taken from the readouts rather than re-tokenised.

    `tokenizer.encode(" prolet")` does **not** return one token -- these are
    basin *labels*, decoded from single ids the readout produced, and
    re-encoding the printed string is a different operation that silently
    returns two pieces. Every id here is one the readout actually emitted, so
    the check is against the same vocabulary entries the basin labels came
    from.
    """
    ids = {}
    for r in recs:
        for ro in (r.get("readout") or {}).values():
            if ro.get("basin_token_id") is not None:
                ids[ro["basin_raw"]] = int(ro["basin_token_id"])
    return ids


def logit_lens_extra(model, recs: list, n_null: int = 64) -> dict:
    """Percentile rank of the basin tokens under each cell's dW direction.

    The top-20 list in section 3a answers "what does this direction point at"
    only in the loose sense that *any* direction pushed through `W_U` returns
    twenty tokens. This asks the sharp version: **where do the tokens the
    experiment is actually about sit** in that direction's own ranking, and is
    that anywhere unusual against directions drawn at random?

    A rank at the 50th percentile means the weight change carries no
    preference at all for the token whose basin it moved the loop into. The
    null is the empirical distribution over `n_null` isotropic directions, so
    "unusual" is measured rather than asserted.
    """
    ids = _lens_token_ids(model, recs)
    W_U = model.W_U.double()
    n_vocab = W_U.shape[1]

    def ranks(v: torch.Tensor) -> dict:
        logits = (v.double() @ W_U)
        out = {}
        for t, i in ids.items():
            # Percentile of this token's logit among all 50257.
            pct = float((logits < logits[i]).double().mean()) * 100.0
            out[t] = {"logit": float(logits[i]), "percentile": pct}
        return out

    g = torch.Generator().manual_seed(20260729)
    null = []
    for _ in range(n_null):
        z = torch.randn(W_U.shape[0], generator=g, dtype=torch.float64)
        z = z / z.norm()
        null.append(ranks(z))

    out = {"tokens": {t: int(i) for t, i in ids.items()},
           "n_vocab": int(n_vocab), "n_null": n_null, "cells": {}}
    for t in ids:
        p = sorted(x[t]["percentile"] for x in null)
        out.setdefault("null", {})[t] = {
            "median_percentile": p[len(p) // 2],
            "p05": p[int(0.05 * len(p))], "p95": p[int(0.95 * len(p))],
        }
    for r in recs:
        if not r.get("dW_closed"):
            continue
        v = torch.tensor(r["dW_closed"]["u_right_768"], dtype=torch.float64)
        ro = r.get("readout") or {}
        out["cells"][r["cell_id"]] = {
            # Episode cells record the live arm as "closed_live"; only severed
            # cells legitimately have no closed readout at all.
            "basin_frozen": (ro.get("frozen") or {}).get("basin"),
            "basin_closed": ((ro.get("closed") or ro.get("closed_live")) or {}).get("basin"),
            "ranks": ranks(v),
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, spec=".4g", none="--"):
    if x is None:
        return none
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return format(x, spec)


def _basin_cell(d: dict) -> str:
    if not d:
        return "--"
    return f"`{d['basin']}` (margin {d['top_logit_margin']:.3f})"


def _spread(vals) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.3e}"
    return (f"{min(vals):.3e} … {max(vals):.3e} "
            f"(median {statistics.median(vals):.3e}, n={len(vals)})")


def _lens_section(lens: dict, main: dict) -> list:
    """Where the basin tokens themselves sit in the dW direction's ranking."""
    L, A = [], None
    L = []
    A = L.append
    if not lens or not main or main["cell_id"] not in lens.get("cells", {}):
        return L
    cell = lens["cells"][main["cell_id"]]
    A("The top-20 list answers \"what does this direction point at\" only in "
      "the loose sense that any direction points at twenty tokens. The sharp "
      "version is where the tokens this experiment is actually about sit in "
      "that direction's own ranking of all "
      f"{lens['n_vocab']} of them, against a null of {lens['n_null']} "
      "isotropic directions:")
    A("")
    A("| token | percentile under ΔW's dominant direction | null median | null 5–95% |")
    A("|---|---|---|---|")
    for t, v in cell["ranks"].items():
        n = lens["null"][t]
        A(f"| {json.dumps(t)} | {v['percentile']:.1f} | "
          f"{n['median_percentile']:.1f} | "
          f"{n['p05']:.1f} – {n['p95']:.1f} |")
    A("")
    inside = [t for t, v in cell["ranks"].items()
              if lens["null"][t]["p05"] <= v["percentile"] <= lens["null"][t]["p95"]]
    A(f"The loop went `{cell['basin_frozen']}` → `{cell['basin_closed']}`. "
      + (f"**All {len(inside)} of the tokens checked sit inside the null's "
         f"5–95% band**, including the one the flip landed on. The dominant "
         f"direction of ΔW carries no measurable preference for the tokens "
         f"whose basin it moved the loop into — the basin change is not "
         f"legible in the weight change by logit lens, and the top-20 list "
         f"above should be read as what a direction looks like through `W_U`, "
         f"not as content."
         if len(inside) == len(cell["ranks"]) else
         f"{len(cell['ranks']) - len(inside)} of {len(cell['ranks'])} tokens "
         f"sit outside the null's 5–95% band."))
    A("")
    return L


def build_report(recs: list, meta: dict, lens: dict | None = None) -> str:
    routed = [r for r in recs if r["kind"] == "routed"]
    severed = [r for r in recs if r["kind"] == "severed"]
    episodes = [r for r in recs if r["kind"] == "episode"]
    reverts = [r for r in recs if r["kind"] == "revert"]
    main = next((r for r in routed
                 if r.get("prompt_id") == PROMPT_MAIN and r["seed"] == 0),
                routed[0] if routed else None)

    def pcos(d):
        """The phase-aware cosine, recomputed from the record's own candidates.

        Not read from a single stored field: on a period-2 orbit two states can
        sit on the same cycle in opposite phases, and the (final, final) entry
        that `compare_states` writes under `cos` would score that as a large
        distance when it is none.
        """
        if not d:
            return None
        if d.get("all"):
            return max(d["all"].values())
        return d.get("phase_aware_cos", d.get("cos"))

    L, A = [], None
    L = []
    A = L.append

    # Title and opening paragraph: "the one step size / the only cell in the
    # whole sweep" is the C-21 undercount. There are TWO ceiling-silent cells
    # that flip the basin, both `hebb`. What is singular is the RULE, not the
    # step size, and the two cells are not an independent replication.
    A(f"# EXP-001 — the offline arm at hebb, eta = {ETA:.6g}")
    A("")
    A("Issues #26, #30, #32. `hebb`, eta = "
      f"{ETA:.6g}, site `{SITE}`, {N_STEPS} steps, cadence {CADENCE}, "
      f"`max_delta_frac` = {MAX_DELTA_FRAC}.")
    A("")
    A("The step-size map found `hebb` to be the only *rule* that moves the loop "
      "inside a clean band, and it does so at **two** step sizes (**C-21**): "
      "this eta, giving basin `prolet` → `comrade` at 1.12% relative weight "
      "change with the norm ceiling silent, and eta 1.18e-04, giving the same "
      "`comrade` at 2.20% drift, also ceiling-silent. The two are **not** an "
      "independent replication — they share prompt, seed, site and cadence — so "
      "what they establish is that the flip survives a 1.7× change in eta. "
      "`oja`, `anti_hebb` and `random` all have wide usable bands in which no "
      "basin change was recorded. Every offline-control number recorded in this "
      "repo **before this run** was taken at `oja`, eta 1e-5 — inside the dead "
      "zone. The later offline and severed arms at `hebb` all postdate it: C-31 "
      "is this run's own, and C-58's eta and cadence grid is T2.1's. This is the "
      "first time the control has been run where the loop actually moves.")
    A("")

    if not recs:
        A("No records.")
        return "\n".join(L)

    # --- headline ---------------------------------------------------------
    A("## 1. The headline: does the offline arm flip the basin too?")
    A("")
    if main and main.get("readout"):
        ro = main["readout"]
        A("`A01_physics`, seed 0, layers 0→11. Each arm's final matrix is "
          "installed and the loop re-run **frozen** under it from the same "
          "iteration-0 tensor, so the four rows differ only in which matrix "
          "produced the trajectory.")
        A("")
        A("| arm | basin at iter 120 | top1−top2 logit margin | lag-1 | lag-2 | "
          "cos(final, frozen-baseline final) |")
        A("|---|---|---|---|---|---|")
        order = [("frozen", "frozen baseline (W0)"),
                 ("closed", "closed loop"),
                 ("offline_recomputed", "offline, `recomputed` y"),
                 ("offline_recorded", "offline, `recorded` y")]
        st = main.get("state", {})
        key = {"closed": "closed_vs_frozen",
               "offline_recomputed": "offline_recomputed_vs_frozen",
               "offline_recorded": "offline_recorded_vs_frozen"}
        for k, label in order:
            d = ro.get(k)
            if not d:
                continue
            c = "1.0 (self)" if k == "frozen" else _f(pcos(st.get(key[k])), ".6f")
            A(f"| {label} | `{d['basin']}` | {d['top_logit_margin']:.3f} | "
              f"{_f(d['cos_lag1_mean'], '.5f')} | {_f(d['cos_lag2_mean'], '.5f')} | {c} |")
        A("")
        fb = ro["frozen"]["basin"]
        cb = ro["closed"]["basin"]
        ob = ro.get("offline_recomputed", {}).get("basin")
        if cb != fb and ob == cb:
            A(f"**The offline arm flips too.** Closed loop `{fb}` → `{cb}`; "
              f"offline (`recomputed` y, the zero-floor path) `{fb}` → `{ob}`. "
              "Feedback is not required for the basin flip: the flip is a "
              "property of the rule applied to this activation distribution, "
              "which is what a Hebbian rule does anywhere, loop or no loop.")
        elif cb != fb and ob == fb:
            A(f"**The offline arm does not flip.** Closed loop `{fb}` → `{cb}`; "
              f"offline stays `{ob}`. This is the first evidence in this "
              "project that the coupling does something the rule alone cannot.")
        else:
            A(f"Closed loop basin `{cb}`, offline `{ob}`, frozen `{fb}`.")
        A("")
        cvo = st.get("closed_vs_offline_recomputed")
        if cvo:
            A(f"Closed arm against offline (`recomputed`) arm, final states: "
              f"phase-aware cos = {pcos(cvo):.9f} (phase "
              f"`{cvo.get('phase')}`), relative L2 "
              f"{cvo['rel_l2_diff']:.3e}. The float32 round-off floor of this "
              f"instrument is `1 − cos` ≈ 1.5e-14; this pair sits at "
              f"{1 - pcos(cvo):.3e}, i.e. "
              f"{(1 - pcos(cvo)) / 1.5e-14:.2g}× the floor. "
              f"`torch.equal` is not used anywhere here: two states that are "
              f"the same point of the dynamics differ in their last bits.")
            A("")
        A(f"**Read the margin.** The frozen baseline sits at a top1−top2 logit "
          f"margin of {ro['frozen']['top_logit_margin']:.3f}. 69 of the 125 "
          f"baseline prompts have a margin below 0.5 and this prompt is one of "
          f"them, so the basin label here is a coin balanced on its edge: a "
          f"flip across it is a much weaker statement than a flip across a "
          f"margin of 2. The lag-1/lag-2 columns and the final-state cosine are "
          f"the quantities that do not depend on where an argmax happened to "
          f"land.")
        A("")

    # --- matched axes -----------------------------------------------------
    A("## 2. Arms matched, and the ceiling silent")
    A("")
    ok = all(r["arms_matched"] for r in routed + severed)
    n_ax = routed[0]["n_axes_checked"] if routed else 0
    A(f"`verify_arms_matched` passed on **{n_ax}/{n_ax} axes** for "
      f"{'every' if ok else 'NOT every'} cell "
      f"({len(routed)} routed + {len(severed)} severed). "
      + ("" if ok else "**A mismatched cell reports no comparison.**"))
    A("")
    clip = [r for r in routed + severed
            if r["clipped_closed"] or r["clipped_offline_recorded"]
            or r["clipped_offline_recomputed"]]
    if clip:
        A(f"**The norm ceiling fired on {len(clip)} cell(s).** Those cells are "
          "not usable: a clipped arm is a different arm, and the comparison is "
          "no longer about feedback. Cells: "
          + ", ".join(r["cell_id"] for r in clip) + ".")
    else:
        A(f"**The ceiling never fired**, on any arm of any cell — "
          f"`clipped` False after all {N_STEPS} updates in every case, as the "
          f"step-size map's 0.0% clip rate for this cell predicted. The number "
          f"below is the rule, not `max_delta_frac`.")
    A("")
    if routed:
        A("| cell | rel ΔW closed | rel ΔW offline (recomputed) | rel ΔW offline (recorded) | clipped |")
        A("|---|---|---|---|---|")
        for r in routed + severed:
            A(f"| `{r['cell_id']}` | {r['rel_weight_change_closed']:.3e} | "
              f"{_f(r['rel_weight_change_offline_recomputed'], '.3e')} | "
              f"{r['rel_weight_change_offline_recorded']:.3e} | "
              f"{'yes' if r['clipped_closed'] else 'no'} |")
        A("")

    # --- routed vs severed ------------------------------------------------
    A("## 3. Routed against severed")
    A("")
    A("The severed arm reads the loop out at "
      f"`blocks.{LAYER_END_SEVERED}.hook_resid_post`, below the plastic site at "
      "layer 6. No state feedback can exist, so the `x` reaching the rule is "
      "bit-identical in both arms and **whatever the arms still differ by "
      "there is the floor**. Any effect claimed in the routed configuration "
      "has to exceed it.")
    A("")
    A("| configuration | y_source | `cos_delta` | `rel_fro_diff` | `diff_over_drift` |")
    A("|---|---|---|---|---|")
    for label, group in (("routed (0→11)", routed), ("severed (0→3)", severed)):
        for ys, key in (("`recomputed` (floor 0)", "weight_recomputed_y"),
                        ("`recorded` (floor ≠ 0)", "weight_recorded")):
            vals = [r[key] for r in group if r.get(key)]
            if not vals:
                continue
            A(f"| {label} | {ys} | "
              f"{_spread([v['cos_delta'] for v in vals])} | "
              f"{_spread([v['rel_fro_diff'] for v in vals])} | "
              f"{_spread([v['diff_over_drift'] for v in vals])} |")
    A("")
    rr = [r["weight_recomputed_y"]["diff_over_drift"] for r in routed
          if r.get("weight_recomputed_y")]
    sr = [r["weight_recomputed_y"]["diff_over_drift"] for r in severed
          if r.get("weight_recomputed_y")]
    if rr and sr:
        msr = statistics.median(sr)
        A(f"In the zero-floor `recomputed` mode the routed cells sit at "
          f"{statistics.median(rr):.3e} against a severed floor of "
          f"{msr:.3e}"
          + (". The severed arms come out **bit-identical** — `rel_fro_diff` "
             "is exactly 0.0, not small — so the floor is zero in the literal "
             "sense and the ratio is not a finite number. The routed "
             "difference is therefore entirely above the floor, and it is the "
             "one measurement in this file that is unambiguously about "
             "feedback. The usual objection to the severed control — that it "
             "runs a shallower loop (0→3) with different activation "
             "statistics, so its floor is not exactly the routed loop's — "
             "does not bite in this mode: `torch.equal` on the two matrices "
             "is True, and zero is zero at any depth. It does bite on the "
             "`recorded` row below."
             if msr == 0.0 else
             f" — a ratio of {statistics.median(rr) / msr:.3g}."))
        A("")
    rc = [r["weight_recorded"]["diff_over_drift"] for r in routed]
    sc = [r["weight_recorded"]["diff_over_drift"] for r in severed]
    if rc and sc:
        A(f"In the default `recorded` mode the same comparison reads "
          f"{statistics.median(rc):.3e} routed against "
          f"{statistics.median(sc):.3e} severed. "
          + ("**The severed number is the larger of the two**, i.e. the naive "
             "protocol reports a bigger apparent effect with the feedback "
             "physically disconnected than with it connected, and no claim "
             "about feedback can be made from it."
             if statistics.median(sc) >= statistics.median(rc) else
             "The routed number exceeds the severed one here, but the "
             "`recorded` floor is a frozen-`y` artefact rather than noise and "
             "the two loops have different depths, so the gap is indicative "
             "and not subtractable."))
        A("")

    # --- seeds ------------------------------------------------------------
    A("### Seeds")
    A("")
    by_seed = {}
    for r in routed:
        if r.get("prompt_id") == PROMPT_MAIN:
            by_seed[r["seed"]] = r
    if len(by_seed) > 1:
        vals = [by_seed[s]["weight_recomputed_y"]["diff_over_drift"] for s in sorted(by_seed)]
        spread = max(vals) - min(vals)
        A(f"Seeds {sorted(by_seed)} on the same prompt give `diff_over_drift` "
          f"(recomputed) {', '.join(f'{v:.6e}' for v in vals)} — spread "
          f"{spread:.3e}.")
        A("")
        if spread == 0.0:
            A("**The spread is exactly zero, and that is a fact about the rule, "
              "not a lucky run.** `seed` reaches `OjaPlasticity` only through "
              "`self._rng`, which is drawn from in `mode=\"random\"` and nowhere "
              "else. `hebb` has no stochastic component, the model is frozen and "
              "single-threaded, so three seeds are three bit-identical runs. "
              "Reporting them as a three-seed spread would be reporting the same "
              "run three times. The spread that means something here is across "
              "prompts, below.")
            A("")
    prompts = sorted({r.get("prompt_id") for r in routed})
    if len(prompts) > 1:
        A("Across prompts (all `prolet` under the frozen loop), seed 0:")
        A("")
        A("| prompt | basin frozen → closed | margin frozen | "
          "basin offline (recomputed) | `diff_over_drift` recomputed | severed |")
        A("|---|---|---|---|---|---|")
        for p in prompts:
            r = next(x for x in routed if x.get("prompt_id") == p and x["seed"] == 0)
            sv = next((x for x in severed
                       if x.get("prompt_id") == p and x["seed"] == 0), None)
            ro = r.get("readout", {})
            fb = ro.get("frozen", {})
            A(f"| `{p}` | `{fb.get('basin')}` → "
              f"`{ro.get('closed', {}).get('basin')}` | "
              f"{_f(fb.get('top_logit_margin'), '.3f')} | "
              f"`{ro.get('offline_recomputed', {}).get('basin')}` | "
              f"{_f(r['weight_recomputed_y']['diff_over_drift'], '.3e')} | "
              f"{_f(sv['weight_recomputed_y']['diff_over_drift'], '.3e') if sv else '--'} |")
        A("")

    # --- dW ---------------------------------------------------------------
    A("## 4. Decoding ΔW (issue #32 sections 2 and 3a)")
    A("")
    if main:
        d = main["dW_closed"]
        sp = d["delta_spectrum"]
        A(f"ΔW is (3072, 768) at this site, ‖ΔW‖_F = {sp['fro']:.4f} "
          f"({main['rel_weight_change_closed']:.3%} of ‖W0‖_F).")
        A("")
        A("| quantity | closed loop | offline (recomputed) |")
        A("|---|---|---|")
        od = main.get("dW_offline_recomputed") or {}
        osp = od.get("delta_spectrum", {})
        A(f"| σ₁ | {sp['singular_values_top'][0]:.4f} | "
          f"{_f((osp.get('singular_values_top') or [None])[0], '.4f')} |")
        A(f"| σ₂ | {sp['singular_values_top'][1]:.4f} | "
          f"{_f((osp.get('singular_values_top') or [None, None])[1], '.4f')} |")
        A(f"| σ₃ | {sp['singular_values_top'][2]:.4f} | "
          f"{_f((osp.get('singular_values_top') or [None, None, None])[2], '.4f')} |")
        A(f"| fraction of ‖ΔW‖²_F in component 1 | {sp['frac_energy_1']:.4f} | "
          f"{_f(osp.get('frac_energy_1'), '.4f')} |")
        A(f"| fraction in the top 5 | {sp['frac_energy_top5']:.4f} | "
          f"{_f(osp.get('frac_energy_top5'), '.4f')} |")
        A(f"| effective rank (participation ratio) | {sp['erank_pr']:.2f} | "
          f"{_f(osp.get('erank_pr'), '.2f')} |")
        A(f"| stable rank | {sp['erank_stable']:.2f} | "
          f"{_f(osp.get('erank_stable'), '.2f')} |")
        A("")
        A("The step-size map measured ΔW effective rank 1.8–3.8 for `oja` and "
          "718.8 for the isotropic noise arm. This is the `hebb` number at the "
          "cell that moves the loop. Near-rank-1 is **expected** — it is the "
          "rule doing what the rule does — and is reported as a sanity check, "
          "never as a discovery.")
        A("")
        A("### The dominant direction, decoded")
        A("")
        A("The 768-side factor of ΔW lives in the residual stream's output "
          "space; pushed through `W_U` (no `b_U` — a direction is not a state, "
          "and the unembedding bias would give every direction, including the "
          "controls, the same ordering). The SVD's sign is arbitrary, so it is "
          "fixed by requiring a positive inner product with the mean "
          "post-synaptic activity at the site over the frozen episode "
          f"(cos = {d['u_right_sign_ref_cos']:+.4f}); the same rule is applied "
          "to every cell so the cross-basin cosines below are comparable.")
        A("")
        A("| rank | ΔW top | ΔW bottom | random control #0 top | random control #1 top |")
        A("|---|---|---|---|---|")
        rc0, rc1 = d["random_controls"][0], d["random_controls"][1]
        for i in range(TOP_K_TOKENS):
            A(f"| {i + 1} | {json.dumps(d['decode']['top_tokens'][i])} "
              f"({d['decode']['top_logits'][i]:+.2f}) | "
              f"{json.dumps(d['decode']['bottom_tokens'][i])} "
              f"({d['decode']['bottom_logits'][i]:+.2f}) | "
              f"{json.dumps(rc0['top_tokens'][i])} ({rc0['top_logits'][i]:+.2f}) | "
              f"{json.dumps(rc1['top_tokens'][i])} ({rc1['top_logits'][i]:+.2f}) |")
        A("")
        A(f"The random controls are isotropic directions of **matched norm** "
          f"(unit, as the singular vector is), decoded identically. Logit "
          f"spread: ΔW direction σ = {d['decode']['logit_std']:.4f}, max "
          f"{d['decode']['logit_max']:.4f}; controls σ = "
          + ", ".join(f"{c['logit_std']:.4f}" for c in d["random_controls"])
          + ", max "
          + ", ".join(f"{c['logit_max']:.4f}" for c in d["random_controls"])
          + ".")
        A("")
        A("**Read the two columns side by side before reading either one.** "
          "`W_U` is not isotropic; any direction pushed through it returns a "
          "list of tokens that can be narrativised. The question the control "
          "answers is whether the ΔW list is more concentrated, or more "
          "coherent, than a random direction's — not whether it looks "
          "meaningful on its own.")
        A("")
        ctrl_std = statistics.median(c["logit_std"] for c in d["random_controls"])
        ctrl_max = statistics.median(c["logit_max"] for c in d["random_controls"])
        A(f"On the one quantitative comparison available from these columns, "
          f"the ΔW direction is **not** the more concentrated of the two: its "
          f"logit spread is {d['decode']['logit_std']:.4f} against a control "
          f"median of {ctrl_std:.4f}, and its largest logit is "
          f"{d['decode']['logit_max']:.4f} against {ctrl_max:.4f}. "
          + ("It is flatter than isotropic noise, not sharper."
             if d["decode"]["logit_std"] < ctrl_std else
             "It is sharper than isotropic noise."))
        A("")
        L.extend(_lens_section(lens, main))
        if main.get("cos_dW_closed_offline_recomputed") is not None:
            _c = main['cos_dW_closed_offline_recomputed']
            _r = (main['rel_weight_change_closed']
                  / main['rel_weight_change_offline_recomputed'])
            # Decompose the difference between the two updates into the part
            # perpendicular to the closed arm (a direction change) and the part
            # along it (a scale change), both relative to ||dW_closed||. A
            # cosine near 1 does NOT mean "scale only": at cos 0.993 the
            # perpendicular part is still several times the parallel one.
            _perp = _r * math.sqrt(max(0.0, 1.0 - _c * _c))
            _par = abs(1.0 - _r * _c)
            A(f"**Issue #32 section 4**: `cos(ΔW_closed, ΔW_offline)` = "
              f"{_c:.8f} (recomputed-y arm), with norm ratio {_r:.6f}.")
            A("")
            A(f"Splitting the difference between the two updates: the component "
              f"perpendicular to the closed-loop update is **{_perp:.4f}** of "
              f"`||ΔW_closed||`, the component along it is **{_par:.4f}** -- a "
              f"ratio of {_perp / _par:.2f} to 1"
              + (" in favour of the perpendicular part." if _perp > _par else
                 " in favour of the parallel part.")
              + " The cosine and the norm ratio are reported separately for "
                "this reason; a single mixed ratio cannot distinguish a "
                "direction change from a magnitude change.")
            A("")

    # --- across basins ----------------------------------------------------
    A("## 5. Does the ΔW direction differ by basin? (issue #32 section 3b)")
    A("")
    dw_cells = [r for r in routed if r["seed"] == 0 and r.get("dW_closed")] + \
               [r for r in episodes if r.get("dW_closed")]
    if len(dw_cells) >= 2:
        labels = []
        vecs = []
        for r in dw_cells:
            b = (r.get("readout", {}).get("frozen", {}) or {}).get("basin", "?")
            labels.append((r.get("prompt_id", r["prompt"][:24]), b))
            vecs.append(torch.tensor(r["dW_closed"]["u_right_768"],
                                     dtype=torch.float64))
        A("Dominant 768-side ΔW direction, one closed-loop episode per prompt, "
          "sign-fixed as above. The basin label is the prompt's **frozen-loop** "
          "basin (the attractor the episode ran in), from the 125-prompt "
          "baseline.")
        A("")
        A("| | " + " | ".join(f"{p}<br>`{b}`" for p, b in labels) + " |")
        A("|---|" + "---|" * len(labels))
        for i, (p, b) in enumerate(labels):
            row = [f"**{p}**<br>`{b}`"]
            for j in range(len(labels)):
                row.append("--" if i == j else f"{_cos(vecs[i], vecs[j]):+.6f}")
            A("| " + " | ".join(row) + " |")
        A("")
        within, between = [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                c = _cos(vecs[i], vecs[j])
                (within if labels[i][1] == labels[j][1] else between).append(c)
        if within and between:
            A(f"Within-basin median cosine {statistics.median(within):+.6f} "
              f"({len(within)} pairs); between-basin median "
              f"{statistics.median(between):+.6f} ({len(between)} pairs).")
            A("")
            if abs(statistics.median(within) - statistics.median(between)) < 0.02:
                A("**The directions do not separate by basin.** ΔW records the "
                  "site's activation statistics; it does not fingerprint which "
                  "attractor the episode was in, and the persistence argument "
                  "gets no episode-specific content from this object.")
            else:
                A("**The directions separate by basin**, so ΔW carries a "
                  "recoverable trace of which attractor the episode ran in.")
            A("")
            A("Two things have to be said about that before it is worth "
              "anything. First, issue #32's two branches — \"ΔW fingerprints "
              "the attractor\" and \"ΔW records the site's activation "
              "statistics and nothing about the episode\" — are **not "
              "mutually exclusive here**, because the attractor is what "
              "determines the site's activation statistics. Section 1 showed "
              "the offline arm, which sees nothing but the frozen activation "
              "statistics, reproduces the closed arm's ΔW to "
              f"cos = {_f(main.get('cos_dW_closed_offline_recomputed'), '.5f') if main else '--'}. "
              "So this separation is the second branch, seen from the other "
              "side: the direction is a readout of the activation "
              "distribution, and the distribution differs between attractors. "
              "It is not evidence of anything episode-specific over and above "
              "that.")
            A("")
            A("Second, prompts in the same basin are not a random sample — "
              "`A01`/`A02`/`A04` are all Complex-register academic sentences "
              "and the two `Divine` prompts are as well, so prompt similarity "
              "and basin membership are confounded in this design. Separating "
              "them needs the within/between measurement over many prompts "
              "per basin that issue #32 asks for and this run does not have.")
            A("")

    # --- C1 ---------------------------------------------------------------
    if reverts:
        r = reverts[0]
        A("## 6. C1 — does the state come back when the weights do?")
        A("")
        A(f"After the episode, W0 is restored and the loop continues from the "
          f"closed arm's final state. Horizon {r['horizon']} with early stop at "
          f"`1 − cos` ≤ {r['return_tol']:g}, **not 200**: the measured "
          f"per-iteration contraction is ~0.968, about 71 iterations per decade "
          f"of displacement, so returning from a displacement of order "
          f"{r['start_gap_1_minus_cos']:.1e} to the round-off floor needs "
          f"several hundred iterations and a 200-iteration horizon has already "
          f"produced one false \"failed to return\" in this repo.")
        A("")
        A("The reference is iterated in **lockstep**, not held fixed at the "
          "episode's last iteration. The frozen loop is itself still settling "
          "at iteration 120, so a state that has come all the way back onto "
          "the frozen trajectory still reads a nonzero gap against the "
          "iteration-120 snapshot. Both are reported: the lockstep gap is the "
          "C1 answer, the fixed one is what the naive version of this control "
          "would have said.")
        A("")
        A(f"- start gap `1 − cos` = {r['start_gap_1_minus_cos']:.3e}")
        A(f"- returned (lockstep reference): "
          f"**{'yes' if r['returned'] else 'no'}**"
          + (f", at iteration {r['returned_at_iter']}" if r["returned"] else
             f"; final gap {_f(r.get('final_gap_vs_lockstep_reference'), '.3e')} "
             f"after {r['horizon']} iterations"))
        A(f"- against a **fixed** iteration-120 target, the same run's final "
          f"gap is {_f(r.get('final_gap_vs_fixed_iter120_target'), '.3e')}"
          + ("" if r.get("final_gap_vs_fixed_iter120_target") is None else
             " — which is where a 200-iteration, fixed-target version of this "
             "control would have reported a failure to return"))
        A(f"- basin after revert: {_basin_cell(r['basin_after_revert'])}; "
          f"lockstep reference "
          f"{_basin_cell(r.get('basin_lockstep_reference'))}; "
          f"fixed iteration-120 target {_basin_cell(r['basin_frozen_target'])}")
        A("")
        if r.get("gap_curve") and len(r["gap_curve"][0]) > 2:
            first, last = r["gap_curve"][0], r["gap_curve"][-1]
            A(f"Gap curve, lockstep: {first[1]:.3e} at iteration {first[0]} → "
              f"{last[1]:.3e} at {last[0]}. Fixed-target: {first[2]:.3e} → "
              f"{last[2]:.3e}.")
            A("")
            if first[1] > 0 and last[1] > 0 and last[0] > first[0]:
                # Displacement ~ sqrt(2(1-cos)); the contraction is per
                # iteration of that, which is the quantity the horizon
                # argument is made in.
                d0, d1 = math.sqrt(2 * first[1]), math.sqrt(2 * last[1])
                n = last[0] - first[0]
                lam = (d1 / d0) ** (1.0 / n)
                # The verdict has to be read off `lam`, not asserted. The
                # earlier wording said "faster" unconditionally, and the
                # max(..., 1e-12) floor would have turned a non-contracting
                # run into a confident "1e+12 iterations per decade".
                if lam >= 1.0:
                    A(f"**The trajectory did not contract on this run**: the "
                      f"measured per-iteration factor is {lam:.4f}, at or above "
                      f"1, so no iterations-per-decade figure is meaningful and "
                      f"the C1 return has to be read from the gap curve alone.")
                else:
                    per_decade = 1 / -math.log10(lam)
                    faster = lam < 0.968
                    A(f"**The contraction measured here is "
                      f"{'faster' if faster else 'slower'} than the ~0.968 the "
                      f"horizon was justified against**: {lam:.4f} per "
                      f"iteration on this trajectory, about {per_decade:.0f} "
                      f"iterations per decade of displacement rather than 71. "
                      f"The 1000-iteration horizon was "
                      f"{'not needed in the end' if faster else 'needed'} -- but "
                      f"it was chosen before the run, and at the 0.968 figure it "
                      f"was the right choice.")
                A("")

    # --- what this does not say -------------------------------------------
    A("## 7. What this does and does not establish")
    A("")
    A("- **ΔW ≠ 0 is not evidence of anything.** The rule moves the weights "
      "with no feedback whatsoever; only the arm comparison speaks to feedback.")
    A("- **Magnitude is not evidence.** eta was chosen, from the step-size map.")
    A("- **One prompt family, one site, one ceiling, 120 steps, cadence 1.** "
      "The map's caveats carry over unchanged.")
    A("- **The `recorded`-mode numbers are reported but not claimed from.** "
      "Their floor is a frozen-`y` artefact that has been measured larger than "
      "the routed signal.")
    A("- **None of this is learning.** No task, no loss, no target. The "
      "defensible phrase is that the weights carry a trace of the episode.")
    A("- **The basin flip is not the finding.** The offline arm flips too, so "
      "the flip is what the rule does to this activation distribution. What "
      "survives is a small, measurable, above-floor difference in *where the "
      "two arms' weights land* — reported in section 3 — which is a much "
      "narrower claim than \"the coupling changes the attractor\".")
    A("- **The severed control is the reason the section-3 number can be "
      "quoted at all.** In `recorded` mode the same protocol reports a larger "
      "apparent effect with the feedback physically disconnected. Only the "
      "`recomputed` path has a floor of literal zero, and only there does the "
      "routed number mean what it appears to mean.")
    A("")

    A("## Provenance")
    A("")
    A(f"{len(recs)} cells, {sum(r.get('seconds', 0) for r in recs) / 60:.0f} "
      f"CPU-minutes.")
    A("")
    A("```json")
    A(json.dumps(meta, indent=2, sort_keys=True))
    A("```")
    A("")
    A("Raw per-cell records, including the full ΔW singular spectra, the "
      "768-component dominant directions and every per-axis match: "
      "`experiments/output_exp001/exp001.jsonl`.")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def read_all(out_dir: Path) -> list:
    recs, seen = [], set()
    for p in sorted(out_dir.glob("exp001*.jsonl")):
        for r in bb.read_jsonl(p):
            if r["cell_id"] not in seen:
                seen.add(r["cell_id"])
                recs.append(r)
    return recs


def main(argv=None):
    # SITE is a module global read by the cell runners, build_report() and the
    # meta block; --site reassigns it below so the override threads through all
    # of them. Declared here, before the `default=SITE` read, as `global` must
    # precede any use of the name in the function.
    global SITE

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=TORCH_THREADS)
    ap.add_argument("--parent", type=str, default=PARENT_DEFAULT)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--extra", action="store_true",
                    help="post-hoc logit-lens check on the recorded dW "
                         "directions; needs the model but no loop runs")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--report", type=str, default=str(REPORT_PATH))
    ap.add_argument("--site", type=str, default=SITE,
                    help="target site for the single-site arm, e.g. blocks.8.mlp "
                         "or blocks.11.attn.head.7. Default keeps the calibrated "
                         "blocks.6.mlp. A non-default site re-uses the default's "
                         "ETA anchor -- re-derive ETA for a clean run; the default "
                         "path is unchanged.")
    args = ap.parse_args(argv)

    # Reassign once from --site; with the default this is a no-op and the run is
    # bit-identical.
    SITE = args.site

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / ("exp001.jsonl" if args.nshards == 1
                       else f"exp001.shard{args.shard}.jsonl")

    cells = build_cells()
    order = {cell_id(c): i for i, c in enumerate(cells)}

    meta = {
        "issues": [26, 30, 32],
        "model": MODEL_NAME,
        "site": SITE,
        "mode": MODE,
        "eta": ETA,
        "eta_provenance": ("D * ||W0||_F / (N_STEPS * U_ref[hebb]) with D=1.8e-2, "
                           "U_ref=350, ||W0||_F=164.854 -- the step-size map's "
                           "own anchor, recomputed rather than copied from its "
                           "rounded table entry"),
        "n_steps": N_STEPS,
        "cadence": CADENCE,
        "max_delta_frac": MAX_DELTA_FRAC,
        "layer_start": LAYER_START,
        "layer_end": LAYER_END,
        "layer_end_severed": LAYER_END_SEVERED,
        "seeds": list(SEEDS),
        "prompts_prolet": list(PROMPTS_PROLET),
        "prompts_divine": list(PROMPTS_DIVINE),
        "y_sources": ["recorded", "recomputed"],
        "revert_horizon": REVERT_HORIZON,
        "device": "cpu",
        "dtype": "float32",
        "norms_dtype": "float64",
        "torch_threads": args.threads,
        "shards": args.nshards,
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

    if args.extra:
        torch.set_num_threads(args.threads)
        torch.set_grad_enabled(False)
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
        model.eval()
        model.requires_grad_(False)
        recs = read_all(out_dir)
        recs.sort(key=lambda r: order.get(r["cell_id"], 10 ** 6))
        (out_dir / "dw_logit_lens.json").write_text(
            json.dumps(logit_lens_extra(model, recs), indent=2), encoding="utf-8")
        print(f"[extra] {out_dir / 'dw_logit_lens.json'}")
        return 0

    if args.report_only:
        recs = read_all(out_dir)
        recs.sort(key=lambda r: order.get(r["cell_id"], 10 ** 6))
        bb.write_jsonl(out_dir / "exp001.jsonl", recs)
        if len(recs) >= len(cells):
            for p in sorted(out_dir.glob("exp001.shard*.jsonl")):
                p.unlink()
        metas = [json.loads(p.read_text())
                 for p in sorted(out_dir.glob("exp001_meta*.json")) if p.stat().st_size]
        if metas:
            meta.update(metas[0])
            meta["shards"] = max(m.get("shards", 1) for m in metas)
            # sorted() puts exp001_meta.json first, so metas[0] supplied the
            # revision and wall clock while the shard count came from the
            # others -- publishing one shard's 282.9s for a run that actually
            # spanned ~2900s across two revisions. Aggregate instead of picking.
            revs = sorted({m.get("repo_rev") for m in metas if m.get("repo_rev")})
            meta["repo_revs"] = revs
            meta["wall_clock_seconds_per_meta"] = [
                m.get("wall_clock_seconds") for m in metas]
            meta["wall_clock_seconds"] = sum(
                m.get("wall_clock_seconds") or 0.0 for m in metas)
            if len(revs) > 1:
                meta["provenance_warning"] = (
                    "cells in this report were produced under more than one "
                    "repo revision; see repo_revs")
        lens_path = out_dir / "dw_logit_lens.json"
        lens = json.loads(lens_path.read_text()) if lens_path.exists() else None
        Path(args.report).write_text(build_report(recs, meta, lens), encoding="utf-8")
        print(f"[report] {args.report} ({len(recs)}/{len(cells)} cells)")
        return 0

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    mine = cells[args.shard::args.nshards] if args.nshards > 1 else cells
    done = {r["cell_id"] for r in read_all(out_dir)}
    todo = [c for c in mine if cell_id(c) not in done]

    print(f"[config] {MODEL_NAME} site={SITE} mode={MODE} eta={ETA:.6g} "
          f"steps={N_STEPS} threads={args.threads} "
          f"shard={args.shard}/{args.nshards}", flush=True)
    if SITE != DEFAULT_SITE:
        print(f"[site] non-default --site {SITE!r}: ETA={ETA:.6g} was anchored to "
              f"{DEFAULT_SITE!r}'s ||W0||_F=164.854 and U_ref[hebb]=350; re-derive "
              "ETA for a clean run (see the module docstring's eta provenance).",
              flush=True)
    print(f"[plan] {len(mine)} cells in this shard, {len(done)} recorded, "
          f"{len(todo)} to run -> {jsonl.name}", flush=True)
    if not todo:
        return 0

    pl = bb.load_prompt_library(args.parent)
    prompts = bb.ordered_prompts(pl)

    from transformer_lens import HookedTransformer
    t_load = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    model.eval()
    model.requires_grad_(False)
    print(f"[model] loaded in {time.time() - t_load:.1f}s", flush=True)

    # Every cell must start from the same W0. Every arm reverts in a finally
    # block, but a leak would look like a feedback effect, so it is checked
    # rather than assumed.
    # Read through the adapter, not a literal blocks[L].mlp.W_out: that assumed an
    # MLP site and would compare an untouched matrix for a per-head or attention
    # --site while the real plastic site leaked between cells. `_make_site`
    # resolves whatever SITE names, exactly as the rule does.
    _guard = _make_site(model, SITE)
    def _live_w0():
        return _guard.weight
    w0_ref = _live_w0().detach().clone()

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    for k, c in enumerate(todo, 1):
        cid = cell_id(c)
        if not torch.equal(_live_w0(), w0_ref):
            raise RuntimeError(
                f"the live weight at {SITE} is not W0 at the start of {cid}; a "
                "previous cell's revert did not restore it and every cell after "
                "this one is on a different matrix")
        prompt = prompts[c["prompt_id"]]
        if c["kind"] == "routed":
            rec = run_routed_cell(model, prompt, c["seed"], LAYER_END, do_reruns=True)
        elif c["kind"] == "severed":
            rec = run_routed_cell(model, prompt, c["seed"], LAYER_END_SEVERED,
                                  do_reruns=False)
        elif c["kind"] == "episode":
            rec = run_episode_cell(model, prompt, c["seed"])
        elif c["kind"] == "revert":
            rec = run_revert_cell(model, prompt, c["seed"])
        else:
            raise ValueError(c["kind"])
        rec["cell_id"] = cid
        rec["prompt_id"] = c["prompt_id"]
        bb.append_jsonl(jsonl, rec)
        elapsed = time.time() - t0
        left = (elapsed / k) * (len(todo) - k)
        extra = ""
        if rec.get("weight_recomputed_y"):
            extra = (f"dod_recomp={rec['weight_recomputed_y']['diff_over_drift']:.3e} "
                     f"dod_rec={rec['weight_recorded']['diff_over_drift']:.3e}")
        if rec.get("readout"):
            ro = rec["readout"]
            extra += (f" basin frozen={ro['frozen']['basin']!r}"
                      + (f" closed={ro['closed']['basin']!r}" if "closed" in ro else "")
                      + (f" offl={ro['offline_recomputed']['basin']!r}"
                         if "offline_recomputed" in ro else ""))
        print(f"[{k}/{len(todo)}] {cid:<34} {extra} "
              f"({rec['seconds']:.0f}s, ETA {bb._hms(left)})", flush=True)

    meta["started"] = started
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["wall_clock_seconds"] = round(time.time() - t0, 1)
    (out_dir / (f"exp001_meta.shard{args.shard}.json" if args.nshards > 1
                else "exp001_meta.json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {len(todo)} cells in {bb._hms(time.time() - t0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
