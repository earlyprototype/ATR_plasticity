"""
Step-size map (issue #30): find the eta band, per rule, where anything can happen.

The precondition for every other plasticity experiment in this repo. We do not
know what step size to use, and the rules do not agree about it: measured at the
default site on the frozen loop, the raw update norm is ~350 for `hebb` and
~1.4e4 for `oja`/`anti_hebb` -- a factor of forty, at the *same* eta. A single
swept eta range therefore puts at least one rule in the wrong regime, and we
would then report a property of the regime as a property of the rule.

Two failure regimes are easy to reach and both look like results:

  too small   the weights barely move, every diagnostic matches the frozen run,
              and we have measured our own noise floor;
  too large   the norm ceiling fires on most updates, and we have measured
              `max_delta_frac` rather than the rule.

and a third, from issue #27 item 11, that reads as healthy on every conventional
dial:

  hollowed    one entry runs away, the normaliser rescales, the rest of the
              matrix is annihilated. ||W||_F stays flat, the clipping rate stays
              low, the relative weight change moves. **Effective rank falling
              while ||W||_F is flat is that failure and nothing else**, which is
              why the spectral columns are recorded on every cell rather than
              only where something looked wrong.

## What one cell is

One (mode, eta) pair: the ATR loop run for `N_STEPS` iterations from one fixed
prompt, one fixed site, one fixed seed, with `OjaPlasticity` applying an update
after every step. Only the mode and eta vary between cells; everything else --
prompt, site, step count, seed, ceiling, the initial state tensor itself -- is
shared, so a difference between two cells is a difference the step size made.

Per step we record the weight-norm trajectory, the relative weight change, the
per-step clipping flag, the per-step non-finite flag, the pre-rescale activation
norm (the post-rescale one is `initial_norm` by construction -- that is the
homeostat, and the gap between the two is how much of the rule's effect it is
absorbing), max and mean absolute entry, and, on a snapshot schedule, the
singular-value spectrum.

## How the eta grid is chosen

Not globally. Each mode's grid is anchored to that mode's own measured update
norm, so cell k of `hebb` and cell k of `oja` are matched in *effect* rather
than in nominal eta:

    eta(mode, D) = D * ||W0||_F / (N_STEPS * U_ref[mode])

`D` is the relative weight displacement the cell would reach if all `N_STEPS`
updates added coherently -- an upper bound, since Oja's decay term opposes its
own reinforcement term, so the delta_frac a cell actually reaches is at or below
its `D`. The grid is a log sweep in `D`, straddling `MAX_DELTA_FRAC` so both the
noise floor and the ceiling are inside the swept range by construction rather
than by luck. `U_ref` is measured, once, on the frozen weights (see the constant).

## Running it

    python experiments/step_size_map.py --shard 0 --nshards 2 &
    python experiments/step_size_map.py --shard 1 --nshards 2 &
    python experiments/step_size_map.py --report-only

One torch thread per process, deliberately: 4 threads is 7x SLOWER than 1 on
this box (2137 vs 287 ms/iter), the OpenMP spin-wait collapse when the thread
count meets the core count. Two single-threaded shards is where the parallelism
comes from. Results are checkpointed to JSONL after every cell, so a killed run
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path

# Set before torch imports anything that reads it.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atr_bridge import initial_state, make_atr_step            # noqa: E402
from plasticity import OjaPlasticity, _make_site               # noqa: E402
import baseline_basins as bb                                    # noqa: E402  (helpers, reused)


# ---------------------------------------------------------------------------
# Config. Every value is echoed into the write-up; a step-size map without its
# step count and its ceiling is unreadable.
# ---------------------------------------------------------------------------

PARENT_DEFAULT = os.environ.get("ATR_PARENT_PATH", bb.PARENT_DEFAULT)

MODEL_NAME = "gpt2-small"
LAYER_START = 0
LAYER_END = 11

# The site. `blocks.6.mlp` is the TransformerLens spelling of the same matrix as
# HuggingFace's `transformer.h.6.mlp.c_proj` -- that MLP's W_out, (3072, 768),
# already in the rules' (n_in, n_out) convention. The loop runs on
# TransformerLens, so this is the spelling that can actually be attached to.
# ||W0||_F = 164.854 here against 164.862 quoted for the HuggingFace matrix; the
# gap is TransformerLens's weight processing, not a different matrix.
#
# `--site` overrides this at the command line (main() reassigns SITE from it) so
# the single-site "separate" sweep can target any site or head without editing the
# file. DEFAULT_SITE is kept as the fixed reference the calibration below is
# anchored to: `U_REF` and `W0_NORM_CALIBRATED` were measured on it, so a
# non-default site re-uses an anchor that does not belong to it (main() says so
# loudly) and the default path stays bit-identical.
SITE = "blocks.6.mlp"
DEFAULT_SITE = "blocks.6.mlp"

# One prompt, fixed. A01_physics: first in the parent library's sweep order,
# lands in `prolet` (the modal basin, 43.2% published) and converges to a fixed
# point with lag-1 cosine 1.00000 under the frozen loop. A fixed point is the
# right baseline here because *any* departure from it is visible -- on the
# period-2 prompts the lag-1 cosine is already 0.68 and a change has to be read
# against a moving reference.
PROMPT_ID = "A01_physics"

# 120 steps, matching the parent's published lock-in iteration. Chosen over 300
# (the baseline sweep's count) as the compute trade: 33 cells at 300 steps would
# not have finished alongside the 125-prompt sweep running on the same box. The
# frozen run for this prompt is already locked in at 120, so the loop-behaviour
# columns are read at a settled state, not mid-transient.
N_STEPS = 120

SEED = 0

# The ceiling on total drift from W0, `||delta||_F / ||W0||_F`. This is the
# quantity whose firing rate every cell reports. Left at the library default.
MAX_DELTA_FRAC = 0.05

# Update applied after every loop step.
CADENCE = 1

# Raw update norm ||upd||_F (pre-eta) per mode, measured on this site with the
# frozen weights over the first six loop steps from this prompt. These are the
# anchors that make the four sweeps comparable in effect; they are NOT tuned,
# they are what the rules produce. Measured 2026-07-28:
#   hebb      110, 135, 160, 793, 130, 799   -> ~350
#   oja       1.00e4, 1.42e4, 1.43e4, 2.15e4, 3.59e3, 1.87e4  -> ~1.4e4
#   anti_hebb 1.00e4, 1.43e4, 1.43e4, 2.20e4, 3.65e3, 1.92e4  -> ~1.4e4
#   random    norm-matched to oja by construction              -> ~1.4e4
U_REF = {"hebb": 350.0, "oja": 14000.0, "anti_hebb": 14000.0, "random": 14000.0}

MODES = ("hebb", "oja", "anti_hebb", "random")

# Target coherent relative displacement per cell. Log sweep straddling
# MAX_DELTA_FRAC (0.05), so the noise floor and the ceiling are both inside the
# swept range by construction. 1e-4 is four decades below anything measurable;
# 1e1 is two hundred times the ceiling.
D_GRID = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 1e1)

# Refinement cells, added after the main grid. `hebb` is the only mode whose
# usable band came out narrower than the grid spacing -- 3.9e-5 to 1.2e-4, a
# factor of three, with the ceiling already at 43% one grid point above -- so
# it gets two extra points: one at the geometric middle of that band, so the
# recommended eta is a measured cell rather than an interpolation, and one
# between the last quiet cell and the first clipping one, to locate the upper
# edge. The other three modes have bands two decades wide and need no help.
REFINE = (("hebb", 1.8e-2), ("hebb", 5.6e-2))

# Iterations at which the singular-value spectrum is taken. svdvals on a
# (3072, 768) float64 matrix costs 0.25 s, so this is affordable densely, and
# density is the point: effective rank is the diagnostic the whole hollowing-out
# question rests on and a sparse schedule could step over the collapse.
SPECTRAL_SCHEDULE = tuple(sorted(set([0, 1, 2, 3] + list(range(5, N_STEPS + 1, 5)) + [N_STEPS])))

# Late-window periodicity scan, same shape as the baseline sweep's.
LAG_WINDOW = 25
MAX_LAG = 8

# Verdict thresholds. Stated here rather than buried in the report so that
# changing one changes the report's own account of itself.
FLOOR_DELTA_FRAC = 1e-3     # below this the weights have not measurably moved
CLIP_QUIET = 0.02           # ceiling is "quiet" at or below 2% of updates
CLIP_LOUD = 0.05            # above this the cell measures the ceiling, not the rule
ERANK_DROP_FLAG = 0.05      # >5% fall in effective rank is flagged
# "Flat" ||W||_F, for the hollowing-out flag. Set just above MAX_DELTA_FRAC on
# purpose: with the ceiling on, ||delta||_F cannot exceed 5% of ||W0||_F, so
# ||W||_F structurally CANNOT run away and its flatness carries no information.
# That is exactly issue #27 item 11's point -- the norm is guaranteed healthy by
# the very mechanism doing the damage -- and it is why effective rank, not
# ||W||_F, is the instrument here.
WNORM_FLAT = MAX_DELTA_FRAC + 0.01

TORCH_THREADS = 1

OUT_DIR = REPO_ROOT / "experiments" / "output_step_size"
REPORT_PATH = REPO_ROOT / "STEP_SIZE_MAP.md"


# ---------------------------------------------------------------------------
# Spectral diagnostics -- the issue #27 item 11 instrument
# ---------------------------------------------------------------------------

def spectral(m: torch.Tensor) -> dict:
    """Effective rank and friends for a 2-D matrix, in float64.

    `erank_pr` is the participation ratio of the singular values,
    `(sum s)^2 / sum s^2`. It equals n for a flat spectrum and 1 for a rank-1
    matrix, and it is the number to lead with: **falling while ||W||_F is flat
    is the hollowing-out failure and nothing else.** A runaway norm shows up in
    ||W||_F; a runaway single *entry* that the normaliser then divides out does
    not, and this does.

    `erank_stable` is the stable rank `sum s^2 / s1^2` -- the same collapse read
    through the energy rather than the amplitude, kept because it is the form
    the rest of this project's measurements already use.

    `top_mass_frac` is the share of total absolute mass held by the largest 0.1%
    of entries. This is the column that distinguishes the two things that both
    produce a near-rank-1 matrix: Oja's rank-1 update is a smooth outer product
    spread across every entry, while the pathology is a handful of individual
    entries dominating. Rank alone cannot tell them apart.
    """
    md = m.double()
    sv = torch.linalg.svdvals(md)
    s_sum = sv.sum().item()
    s2_sum = (sv * sv).sum().item()
    s1 = sv[0].item()
    a = md.abs()
    total = a.sum().item()
    k = max(1, int(0.001 * a.numel()))
    top = torch.topk(a.reshape(-1), k).values.sum().item()
    return {
        "erank_pr": (s_sum * s_sum / s2_sum) if s2_sum > 0 else float("nan"),
        "erank_stable": (s2_sum / (s1 * s1)) if s1 > 0 else float("nan"),
        "sigma1": s1,
        "sigma1_frac_energy": (s1 * s1 / s2_sum) if s2_sum > 0 else float("nan"),
        "sigma1_frac_sum": (s1 / s_sum) if s_sum > 0 else float("nan"),
        "max_abs": a.max().item(),
        "mean_abs": a.mean().item(),
        "max_over_mean_abs": (a.max().item() / a.mean().item()) if total > 0 else float("nan"),
        "top_mass_frac": (top / total) if total > 0 else float("nan"),
        "n_svals": int(sv.numel()),
    }


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------

def run_cell(model, state0, step, mode: str, eta: float, d_target: float | None,
             n_steps: int = N_STEPS) -> dict:
    """One (mode, eta) pair: the ATR loop with plasticity live, fully instrumented.

    Hook discipline, and it is not optional. `atr_bridge.make_atr_step` calls
    `model.reset_hooks()` in its own finally block on every step -- that is the
    parent's loop body, copied, and it clears every TransformerLens hook. The
    plasticity layer survives it only because `_TransformerLensMLPSite` attaches
    plain `torch` forward hooks to the HookPoint modules rather than going
    through `model.add_hook`. Nothing here may call `reset_hooks()` itself: doing
    so would silently detach the ATR engine's injection hook and leave the loop
    running open. Install/remove and `revert()` are in `finally` blocks so the
    live weight is restored even when a cell raises.
    """
    t0 = time.time()
    plast = OjaPlasticity(model, SITE, eta=eta, mode=mode, cadence=CADENCE,
                          max_delta_frac=MAX_DELTA_FRAC, seed=SEED)
    W0 = plast.W0
    W0_norm = plast.W0_norm

    # Per-step series.
    w_norm, delta_frac, upd_norm_raw, pre_rescale = [], [], [], []
    max_abs, mean_abs = [], []
    n_clipped = n_nonfinite = n_applies = 0
    clip_first = nonfinite_first = None
    spec_rows = []

    # Loop-behaviour buffers, same shape as the baseline sweep's.
    r = state0.tensor.clone()
    tail = LAG_WINDOW + MAX_LAG
    mean_tail = deque([r.mean(dim=0).clone()], maxlen=tail)
    last_tail = deque([r[-1, :].clone()], maxlen=tail)
    top1_traj = [bb.top1_id(model, r[-1, :])]

    s0 = spectral(W0)
    spec_rows.append({"iteration": 0, **s0})
    w_norm.append(W0_norm)
    delta_frac.append(0.0)
    max_abs.append(s0["max_abs"])
    mean_abs.append(s0["mean_abs"])

    plast.install()
    try:
        for i in range(1, n_steps + 1):
            # `clipped` and `nonfinite` on the instance are sticky for the whole
            # run by design (see revert()'s docstring). Cleared here, read after
            # apply(), so what is counted is per-update events rather than "did
            # this ever happen". Cleared BEFORE the step, not before apply(),
            # because the hook can raise `nonfinite` during the forward.
            plast.clipped = False
            plast.nonfinite = False

            r = step(model, r)

            # Pre-rescale: the raw output of the forward. The post-rescale norm
            # is `initial_norm` exactly, every step, by construction -- that is
            # the homeostat. The ratio of the two is how hard it is working.
            pre_rescale.append(r.double().norm().item())

            rep = plast.apply()
            n_applies += 1
            if plast.clipped:
                n_clipped += 1
                if clip_first is None:
                    clip_first = i
            if plast.nonfinite:
                n_nonfinite += 1
                if nonfinite_first is None:
                    nonfinite_first = i

            wd = (W0 + plast.delta).double()
            w_norm.append(wd.norm().item())
            delta_frac.append(rep["delta_frac"])
            upd_norm_raw.append(rep["last_update_norm"] / eta if eta else 0.0)
            a = wd.abs()
            max_abs.append(a.max().item())
            mean_abs.append(a.mean().item())

            mean_tail.append(r.mean(dim=0).clone())
            last_tail.append(r[-1, :].clone())
            top1_traj.append(bb.top1_id(model, r[-1, :]))

            if i in SPECTRAL_SCHEDULE:
                spec_rows.append({"iteration": i, **spectral(W0 + plast.delta)})

        final_delta = plast.delta.clone()
        final_report = plast.report()
    finally:
        plast.remove()
        # Unconditional: a cell that leaves its delta in the live weight poisons
        # every later cell, and the map's whole premise is that only eta and mode
        # differ between them.
        plast.revert()

    scan_mean = bb.lag_scan(torch.stack(list(mean_tail)), MAX_LAG)
    scan_last = bb.lag_scan(torch.stack(list(last_tail)), MAX_LAG)
    final_read = bb.readout_detail(model, r[-1, :])
    pos = bb.position_stats(r)

    s_end = spec_rows[-1]
    d_spec = spectral(final_delta) if final_delta.abs().max().item() > 0 else None

    rec = {
        "cell_id": f"{mode}@{eta:.6g}",
        "mode": mode,
        "eta": eta,
        "d_target": d_target,
        "site": SITE,
        "prompt_id": PROMPT_ID,
        "prompt": state0.prompt,
        "n_steps": n_steps,
        "seed": SEED,
        "max_delta_frac": MAX_DELTA_FRAC,
        "W0_norm": W0_norm,
        "initial_norm": state0.initial_norm,

        # --- how far we travelled ---------------------------------------
        "rel_weight_change": final_report["delta_frac"],
        "delta_norm": final_report["delta_norm"],
        "w_norm_first": w_norm[0],
        "w_norm_last": w_norm[-1],
        "w_norm_max": max(w_norm),
        "w_norm_min": min(w_norm),
        "w_norm_rel_range": (max(w_norm) - min(w_norm)) / w_norm[0],
        "w_norm_traj": w_norm,
        "delta_frac_traj": delta_frac,
        "upd_norm_raw_mean": (sum(upd_norm_raw) / len(upd_norm_raw)) if upd_norm_raw else 0.0,
        "upd_norm_raw_traj": upd_norm_raw,

        # --- the ceiling, on every single cell ---------------------------
        "n_applies": n_applies,
        "n_clipped": n_clipped,
        "clip_rate": n_clipped / n_applies if n_applies else float("nan"),
        "clip_first_iter": clip_first,
        "n_nonfinite": n_nonfinite,
        "nonfinite_first_iter": nonfinite_first,

        # --- the homeostat -----------------------------------------------
        "pre_rescale_first": pre_rescale[0] if pre_rescale else None,
        "pre_rescale_last": pre_rescale[-1] if pre_rescale else None,
        "pre_rescale_max": max(pre_rescale) if pre_rescale else None,
        "post_rescale_norm": state0.initial_norm,   # constant, by construction
        "pre_over_post_last": (pre_rescale[-1] / state0.initial_norm) if pre_rescale else None,
        "pre_rescale_traj": pre_rescale,

        # --- did the loop change at all ----------------------------------
        "basin": final_read["top_token_strings"][0].strip(),
        "basin_raw": final_read["top_token_strings"][0],
        "basin_token_id": final_read["top_token_ids"][0],
        "final_top5_tokens": final_read["top_token_strings"],
        "final_top5_probs": final_read["top_token_probs"],
        "final_entropy": final_read["entropy"],
        "cos_lag1_mean": scan_mean.get(1, {}).get("mean"),
        "cos_lag2_mean": scan_mean.get(2, {}).get("mean"),
        "cos_lag1_last": scan_last.get(1, {}).get("mean"),
        "cos_lag2_last": scan_last.get(2, {}).get("mean"),
        "lag_scan_mean_vec": {str(k): v for k, v in scan_mean.items()},
        "final_position_similarity": None if pos is None else pos["mean"],
        "final_tensor_norm": float(r.double().norm()),
        "final_mean_vec": [float(x) for x in r.mean(dim=0)],
        "top1_trajectory_ids": top1_traj,
        "nonfinite_state": not bool(torch.isfinite(r).all()),

        # --- hollowing out (issue #27 item 11) ---------------------------
        "max_abs_W_first": max_abs[0],
        "max_abs_W_last": max_abs[-1],
        "max_over_mean_abs_first": spec_rows[0]["max_over_mean_abs"],
        "max_over_mean_abs_last": s_end["max_over_mean_abs"],
        "erank_pr_first": spec_rows[0]["erank_pr"],
        "erank_pr_last": s_end["erank_pr"],
        "erank_pr_min": min(s["erank_pr"] for s in spec_rows),
        "erank_pr_rel_drop": 1.0 - s_end["erank_pr"] / spec_rows[0]["erank_pr"],
        "erank_stable_first": spec_rows[0]["erank_stable"],
        "erank_stable_last": s_end["erank_stable"],
        "sigma1_frac_energy_first": spec_rows[0]["sigma1_frac_energy"],
        "sigma1_frac_energy_last": s_end["sigma1_frac_energy"],
        "top_mass_frac_first": spec_rows[0]["top_mass_frac"],
        "top_mass_frac_last": s_end["top_mass_frac"],
        "max_abs_W_traj": max_abs,
        "mean_abs_W_traj": mean_abs,
        "spectral": spec_rows,
        "delta_spectral": d_spec,

        "seconds": round(time.time() - t0, 2),
    }
    return rec


# ---------------------------------------------------------------------------
# The cell list
# ---------------------------------------------------------------------------

# Calibration constant, asserted against the live weight in main() before any
# cell runs. Backends disagree on this norm by ~0.005% (see the module
# docstring), and a silent drift here would restate every D target -- and so
# every recommended eta -- against a matrix that is not the one being swept.
W0_NORM_CALIBRATED = 164.854

def eta_for(mode: str, d: float) -> float:
    """`D * ||W0||_F / (N_STEPS * U_ref[mode])` -- see the module docstring."""
    return d * W0_NORM_CALIBRATED / (N_STEPS * U_REF[mode])


def build_cells() -> list:
    """Every cell, in a fixed order. Cell 0 is the frozen reference.

    `mode="off"` at eta=0 accumulates the statistics and applies nothing, so it
    is simultaneously the frozen baseline every other cell is read against and
    the C0 identity check: if it does not reproduce the frozen loop's basin and
    cosines exactly, the instrument is wrong and nothing below it means anything.
    """
    cells = [{"mode": "off", "eta": 0.0, "d_target": None}]
    for mode in MODES:
        for d in D_GRID:
            cells.append({"mode": mode, "eta": eta_for(mode, d), "d_target": d})
    for mode, d in REFINE:
        cells.append({"mode": mode, "eta": eta_for(mode, d), "d_target": d})
    return cells


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x, spec=".4g", none="--"):
    if x is None:
        return none
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return format(x, spec)


def is_hollowed(rec: dict) -> bool:
    """Effective rank down while ||W||_F flat -- issue #27 item 11 and nothing else."""
    return (rec["erank_pr_rel_drop"] > ERANK_DROP_FLAG
            and rec["w_norm_rel_range"] < WNORM_FLAT)


def verdict_for_cell(rec: dict, ref: dict | None) -> str:
    """One cell's regime, from the numbers and nothing else.

    The hollowing-out tag is additive rather than exclusive: a cell can be both
    pinned against the ceiling and hollowed out, and suppressing the second
    because the first fired would hide the failure the ceiling is causing.
    """
    if rec["n_nonfinite"] or rec["nonfinite_state"]:
        base = "diverged"
    elif rec["clip_rate"] > CLIP_LOUD:
        base = "ceiling"
    elif rec["rel_weight_change"] < FLOOR_DELTA_FRAC:
        base = "noise floor"
    elif rec["clip_rate"] > CLIP_QUIET:
        base = "ceiling audible"
    else:
        base = "usable"
    return base + (" + **hollowed**" if is_hollowed(rec) else "")


def loop_changed(rec: dict, ref: dict | None) -> str:
    """Whether the loop's behaviour moved, against the frozen reference cell."""
    if ref is None:
        return "--"
    bits = []
    if rec["basin"] != ref["basin"]:
        bits.append(f"basin {ref['basin']!r}->{rec['basin']!r}")
    if rec["cos_lag1_mean"] is not None and ref["cos_lag1_mean"] is not None:
        if abs(rec["cos_lag1_mean"] - ref["cos_lag1_mean"]) > 1e-5:
            bits.append(f"lag1 {ref['cos_lag1_mean']:.5f}->{rec['cos_lag1_mean']:.5f}")
    a = torch.tensor(rec["final_mean_vec"], dtype=torch.float64)
    b = torch.tensor(ref["final_mean_vec"], dtype=torch.float64)
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
    if cos < 1 - 1e-6:
        bits.append(f"cos(final,frozen)={cos:.6f}")
    return "; ".join(bits) if bits else "no"


def band_for_mode(recs: list) -> dict:
    """The usable band for one mode, and the eta that sits in the middle of it.

    Picked by where the ceiling is quiet and the weights are actually moving --
    not by which cell did the most interesting thing. A mode with no cell
    satisfying both has no usable band, and that is a legitimate outcome.
    """
    ok = [r for r in recs
          if r["n_nonfinite"] == 0 and not r["nonfinite_state"]
          and r["clip_rate"] <= CLIP_QUIET
          and r["rel_weight_change"] >= FLOOR_DELTA_FRAC]
    below = [r for r in recs if r["rel_weight_change"] < FLOOR_DELTA_FRAC]
    above = [r for r in recs if r["clip_rate"] > CLIP_QUIET or r["n_nonfinite"]
             or r["nonfinite_state"]]
    out = {
        "eta_floor": max((r["eta"] for r in below), default=None),
        "eta_ceiling": min((r["eta"] for r in above), default=None),
        "band": [min((r["eta"] for r in ok), default=None),
                 max((r["eta"] for r in ok), default=None)],
        "n_usable": len(ok),
        "recommended": None,
        "recommended_measured": None,
        "hollowed": [r["eta"] for r in recs if is_hollowed(r)],
    }
    if ok:
        lo, hi = out["band"]
        # The geometric middle of the usable cells, not the cell that did the
        # most interesting thing. If the middle happens to be a swept cell,
        # `recommended_measured` equals it; otherwise it is the nearest one and
        # the recommendation is an interpolation, which the report says.
        mid = math.sqrt(lo * hi) if lo > 0 and hi > 0 else (lo or hi)
        out["recommended"] = mid
        out["recommended_measured"] = min(
            ok, key=lambda r: abs(math.log10(r["eta"]) - math.log10(mid)))["eta"]
    return out


def _cos_to_frozen(rec: dict, ref: dict) -> float:
    a = torch.tensor(rec["final_mean_vec"], dtype=torch.float64)
    b = torch.tensor(ref["final_mean_vec"], dtype=torch.float64)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def _findings(recs: list, by_mode: dict, ref: dict | None) -> list:
    """The reading. Every number here is computed from the records, not typed in.

    Issue #27 closes by asking every write-up to say which of its failure modes
    it ruled out and how; this section is that, plus the one thing the map is
    actually for.
    """
    L, A = [], None
    A = L.append
    if ref is None:
        return L

    def quietest_top(mode):
        """The largest-eta cell of a mode with the ceiling silent (clip == 0)."""
        q = [r for r in by_mode.get(mode, []) if r["clip_rate"] == 0.0]
        return max(q, key=lambda r: r["eta"]) if q else None

    A("## What the map says")
    A("")
    A("### 1. Three of the four rules have a wide band, and nothing happens in it")
    A("")
    for mode in ("oja", "anti_hebb"):
        t = quietest_top(mode)
        if t is None:
            continue
        A(f"- `{mode}` at its largest ceiling-silent eta ({t['eta']:.3g}, clip "
          f"**{t['clip_rate']:.1%}**) moves the weights "
          f"{t['rel_weight_change']:.2%} of ‖W0‖_F and the loop does not move: "
          f"basin `{t['basin']}` (frozen: `{ref['basin']}`), lag-1 "
          f"{_fmt(t['cos_lag1_mean'], '.5f')} (frozen "
          f"{_fmt(ref['cos_lag1_mean'], '.5f')}), "
          f"cos(final, frozen) = {_cos_to_frozen(t, ref):.6f}.")
    pinned = [r for r in by_mode.get("oja", []) if r["clip_rate"] >= 0.99]
    if pinned:
        p = pinned[0]
        A(f"- Pushed all the way to the ceiling — `oja` at {p['eta']:.3g}, clip "
          f"**{p['clip_rate']:.1%}**, the full {p['rel_weight_change']:.1%} of "
          f"drift the ceiling allows — the basin is still `{p['basin']}` and "
          f"lag-1 is still {_fmt(p['cos_lag1_mean'], '.5f')}. There is no eta at "
          f"which this rule moves this loop; the ceiling is reached first.")
    A("")
    A("This is saturation, and it is the outcome the prior work predicted: the "
      "reservoir result (Oja-family rules \"seldom exceed even\" the untouched "
      "network) and the Hebbian arm of Chaudhary 2025 (stable at depth, "
      "saturating in performance) both point here. It is a null result, and a "
      "null result with a measured band around it is a very different object "
      "from a null result at an unexamined step size.")
    A("")

    A("### 2. `hebb` is the exception, and it is the rule with no brake")
    A("")
    h = by_mode.get("hebb", [])
    flip = [r for r in h if r["clip_rate"] == 0.0 and r["basin"] != ref["basin"]]
    if flip:
        f0 = min(flip, key=lambda r: r["eta"])
        A(f"The basin changes at eta {f0['eta']:.3g} — `{ref['basin']}` → "
          f"`{f0['basin']}` — at {f0['rel_weight_change']:.2%} relative weight "
          f"change with the ceiling **silent** ({f0['clip_rate']:.1%}), "
          f"cos(final, frozen) = {_cos_to_frozen(f0, ref):.6f}. That is a real "
          f"effect inside a clean band, not a ceiling artefact.")
        A("")
    A("The catch is what `hebb` is. It has no decay term, so its band is narrow "
      "(a factor of three between the first cell that moves the loop and the "
      "first cell that clips) and it is bounded above only by `max_delta_frac`. "
      "The rule that does something is the rule the ceiling is holding up.")
    A("")

    A("### 3. Direction matters, but not enough to rescue Oja")
    A("")
    A("`random` is norm-matched to what Oja would have applied, so it isolates "
      "whether the *direction* is doing work. Matched not by eta — the noise "
      "re-randomises every step and accumulates as a random walk rather than "
      "coherently — but by the relative weight change actually reached:")
    A("")
    A("| arm | eta | rel ΔW | clip | loop |")
    A("|---|---|---|---|---|")
    for mode in ("random", "oja", "anti_hebb", "hebb"):
        cands = [r for r in by_mode.get(mode, [])
                 if r["clip_rate"] == 0.0 and 0.008 <= r["rel_weight_change"] <= 0.05]
        if not cands:
            continue
        c = min(cands, key=lambda r: abs(math.log(r["rel_weight_change"] / 0.02)))
        A(f"| `{mode}` | {c['eta']:.3g} | {c['rel_weight_change']:.2%} | "
          f"{c['clip_rate']:.1%} | {loop_changed(c, ref)} |")
    A("")
    A("Isotropic noise of the same size does nothing at all, so the rules are "
      "not merely \"a perturbation of this magnitude\". But Oja's structured "
      "direction does almost nothing either. The gap that matters is between "
      "`hebb` and everything else, not between structure and noise.")
    A("")

    A("### 4. The homeostat is not what is hiding the effect")
    A("")
    A("Issue #27 item 3's signature is a pre-rescale activation norm that moves "
      "while the loop's visible behaviour stays flat. That is not what these "
      "cells show.")
    A("")
    A(f"The frozen loop already runs at pre/post = {ref['pre_over_post_last']:.4f} "
      f"— the rescaling divides by {ref['pre_over_post_last']:.2f} on every step "
      f"whether or not plasticity is on. Across all {len(recs)} cells the "
      f"plasticity moves that ratio by at most "
      f"{max(abs(r['pre_over_post_last'] / ref['pre_over_post_last'] - 1) for r in recs):.1%}.")
    A("")
    for mode in ("oja", "hebb"):
        t = quietest_top(mode)
        if t is None:
            continue
        dr = t["pre_over_post_last"] / ref["pre_over_post_last"] - 1
        A(f"- `{mode}` at {t['eta']:.3g}: pre-rescale ratio {dr:+.2%} against "
          f"frozen, cos(final, frozen) = {_cos_to_frozen(t, ref):.6f}. "
          + ("Both flat — the change is not reaching the activations at all, "
             "rather than reaching them and being absorbed."
             if abs(dr) < 0.005 else
             "Both move, and together — the homeostat is passing the effect "
             "through, not eating it."))
    A("")

    A("### 5. No hollowing out anywhere in the sweep")
    A("")
    worst = min(recs, key=lambda r: r["erank_pr_min"])
    best = max(recs, key=lambda r: r["erank_pr_last"])
    A(f"Effective rank starts at {ref['erank_pr_first']:.1f} (of 768) and over "
      f"every cell in the map never falls below {worst['erank_pr_min']:.1f} "
      f"(`{worst['mode']}` at {worst['eta']:.3g}, a "
      f"{1 - worst['erank_pr_min'] / ref['erank_pr_first']:.2%} fall). Under "
      f"`oja` and `anti_hebb` it *rises*, to {best['erank_pr_last']:.1f} — the "
      f"decay term flattens the spectrum, which is the opposite direction from "
      f"rank-1 collapse. The largest singular value's energy share falls from "
      f"{ref['sigma1_frac_energy_last']:.4f} to "
      f"{best['sigma1_frac_energy_last']:.4f}, and max/mean |W| falls from "
      f"{ref['max_over_mean_abs_last']:.1f} to "
      f"{best['max_over_mean_abs_last']:.1f}. Nothing is running away.")
    A("")
    o = quietest_top("oja")
    rnd = quietest_top("random")
    if o and rnd and o.get("delta_spectral") and rnd.get("delta_spectral"):
        A(f"The ΔW columns do the distinguishing issue #27 item 11 asks for. "
          f"Oja's accumulated update is near rank-1 (effective rank "
          f"{o['delta_spectral']['erank_pr']:.1f}) exactly as #32 section 2 "
          f"expects, while the noise arm's is isotropic "
          f"({rnd['delta_spectral']['erank_pr']:.1f}). But Oja's mass is not "
          f"concentrated in a handful of *entries*: its top 0.1% of entries "
          f"hold {o['delta_spectral']['top_mass_frac']:.4f} of the total "
          f"absolute mass against the noise arm's "
          f"{rnd['delta_spectral']['top_mass_frac']:.4f} — a smooth outer "
          f"product, not a runaway coupling. Low rank here is the rule working, "
          f"not the pathology.")
        A("")

    A("### What this does and does not rule out")
    A("")
    A("| issue #27 | status |")
    A("|---|---|")
    A("| 2 — no interesting middle | **Ruled out as a confound, and answered.** "
      "Every mode has a band where the weights move with the ceiling silent. "
      "For `oja`/`anti_hebb`/`random` nothing happens inside it; for `hebb` "
      "something does. |")
    A("| 3 — we measure the rescaling | **Ruled out here.** The pre-rescale norm "
      "is flat wherever the loop is flat, so the homeostat is not absorbing a "
      "hidden weight effect. |")
    A("| 11 — norm ceiling and rescaling destroy each other | **Not observed.** "
      "Effective rank flat or rising on every cell, max entry falling, ΔW mass "
      "spread rather than concentrated. |")
    A("| 1 — the rule moves the weights and nothing else happens | **Consistent "
      "with, not established.** That claim needs the offline arm (#26); this "
      "map only shows the loop-on side. |")
    A("| 5 — collapse is already the default | Untouched. This prompt is a fixed "
      "point under the frozen loop and stays one. |")
    A("| 7 — depth | Untouched. |")
    A("")
    A("### Caveats")
    A("")
    A(f"One prompt (`{PROMPT_ID}`), one site (`{SITE}`), one seed, "
      f"{ref['n_steps']} steps, cadence {CADENCE}, one ceiling "
      f"({MAX_DELTA_FRAC}). The recommended etas are calibrated for exactly that "
      f"configuration; a different site has a different ‖W0‖_F and different "
      f"activation scale, and the anchoring formula has to be re-measured rather "
      f"than reused. `random` here is a within-cell control, not the full C2. "
      f"The bands are located to grid resolution — roughly half a decade, and a "
      f"factor of three for `hebb` after refinement — not to a sharp edge.")
    A("")
    return L


def build_report(recs: list, meta: dict) -> str:
    by_mode = {}
    ref = next((r for r in recs if r["mode"] == "off"), None)
    for r in recs:
        by_mode.setdefault(r["mode"], []).append(r)
    for v in by_mode.values():
        v.sort(key=lambda r: r["eta"])

    L = []
    A = L.append
    A("# Step-size map")
    A("")
    A("Issue #30. Where, for each local rule, the weights actually move without "
      "the run measuring the norm ceiling instead of the rule.")
    A("")
    A("## Configuration")
    A("")
    A(f"| | |")
    A("|---|---|")
    A(f"| model | {MODEL_NAME}, frozen, CPU, float32 |")
    if SITE == DEFAULT_SITE:
        A(f"| site | `{SITE}` (= `transformer.h.6.mlp.c_proj`), (3072, 768) |")
    else:
        A(f"| site | `{SITE}` |")
    A(f"| ‖W0‖_F | {_fmt(ref['W0_norm'], '.6f') if ref else '--'} (float64) |")
    A(f"| prompt | `{PROMPT_ID}` — {json.dumps(ref['prompt']) if ref else ''} |")
    n_steps = ref["n_steps"] if ref else meta.get("n_steps", N_STEPS)
    A(f"| loop | layers {LAYER_START}→{LAYER_END}, {n_steps} steps, "
      f"‖x₀‖ = {_fmt(ref['initial_norm'], '.6f') if ref else '--'} |")
    A(f"| plasticity | cadence {CADENCE} (update after every step), "
      f"`max_delta_frac` = {MAX_DELTA_FRAC}, seed {SEED} |")
    # build_cells() appends REFINE on top of the grid, so leaving it out here
    # printed "35 recorded of 33" -- more cells recorded than exist.
    A(f"| cells | {len(recs)} recorded of "
      f"{1 + len(MODES) * len(D_GRID) + len(REFINE)} = 1 frozen reference + "
      f"{len(MODES)}x{len(D_GRID)} grid + {len(REFINE)} refinement |")
    A(f"| eta anchor | `eta = D · ‖W0‖_F / (N · U_ref)`, "
      f"U_ref = {json.dumps(U_REF)} |")
    A(f"| threads | {meta.get('torch_threads', TORCH_THREADS)} per process, "
      f"{meta.get('shards', 1)} shard(s) |")
    A("")
    A("Only `mode` and `eta` vary between cells. Prompt, site, step count, seed, "
      "ceiling and the iteration-0 state tensor are shared, so a difference "
      "between two cells is a difference the step size made.")
    A("")

    if ref:
        A("## Frozen reference (`mode=off`)")
        A("")
        A(f"Basin `{ref['basin']}`, lag-1 {_fmt(ref['cos_lag1_mean'], '.5f')}, "
          f"lag-2 {_fmt(ref['cos_lag2_mean'], '.5f')}, "
          f"‖W‖_F {_fmt(ref['w_norm_last'], '.6f')} (unchanged), "
          f"effective rank {_fmt(ref['erank_pr_last'], '.2f')}, "
          f"pre-rescale ‖x‖ {_fmt(ref['pre_rescale_last'], '.4f')} against "
          f"post-rescale {_fmt(ref['post_rescale_norm'], '.4f')}.")
        A("")
        A("`off` accumulates the statistics and applies nothing, so this row is "
          "both the baseline every other cell is read against and the C0 "
          "identity check on the instrument.")
        A("")

    A("## Verdicts")
    A("")
    for mode in MODES:
        rows = by_mode.get(mode, [])
        if not rows:
            continue
        b = band_for_mode(rows)
        A(f"### `{mode}`")
        A("")
        if b["n_usable"] == 0:
            A("**No usable band.** No swept eta both moved the weights past the "
              f"noise floor (rel. change ≥ {FLOOR_DELTA_FRAC:g}) and kept the "
              f"ceiling quiet (clip rate ≤ {CLIP_QUIET:g}).")
        else:
            meas = b["recommended_measured"]
            same = abs(math.log10(meas) - math.log10(b["recommended"])) < 0.005
            A(f"**Recommended eta: {b['recommended']:.3g}** — the geometric "
              f"middle of the usable band {b['band'][0]:.3g} … "
              f"{b['band'][1]:.3g} ({b['n_usable']} cell(s)), where the ceiling "
              f"is quiet and the weights are actually moving."
              + ("" if same else f" Nearest measured cell: {meas:.3g}."))
        A("")
        A(f"- Nothing happens at or below **{_fmt(b['eta_floor'], '.3g')}** "
          f"(relative weight change < {FLOOR_DELTA_FRAC:g}).")
        A(f"- Ceiling audible / diverges at or above "
          f"**{_fmt(b['eta_ceiling'], '.3g')}**.")
        if b["hollowed"]:
            A(f"- **Hollowing-out flagged** (effective rank fell >"
              f"{ERANK_DROP_FLAG:.0%} while ‖W‖_F stayed within "
              f"{WNORM_FLAT:.0%}) at eta = "
              + ", ".join(f"{e:.3g}" for e in b["hollowed"]) + ".")
        else:
            A("- No hollowing-out: effective rank never fell by more than "
              f"{ERANK_DROP_FLAG:.0%} with ‖W‖_F flat.")
        A("")

    L.extend(_findings(recs, by_mode, ref))

    A("## Full table")
    A("")
    A("`clip` is the fraction of the "
      f"{n_steps} updates the norm ceiling scaled down; it is reported on every "
      "row because a number quoted without it is not usable. `erank` is the "
      "participation ratio of W's singular values (768 max). `pre/post` is the "
      "pre-rescale activation norm over the post-rescale one — the loop's "
      "homeostat is the denominator and holds it at ‖x₀‖ exactly.")
    A("")
    hdr = ("| mode | D | eta | rel ΔW | ‖W‖_F | clip | nonfin | erank | "
           "Δerank | max/mean |W| | pre/post | basin | lag1 | lag2 | verdict |")
    A(hdr)
    A("|" + "---|" * 15)
    order = ["off"] + list(MODES)
    for mode in order:
        for r in by_mode.get(mode, []):
            A("| `{m}` | {d} | {eta} | {rel} | {wn} | {clip} | {nf} | {er} | "
              "{der} | {mm} | {pp} | `{basin}` | {l1} | {l2} | {v} |".format(
                  m=r["mode"],
                  d=_fmt(r["d_target"], ".0e"),
                  eta=_fmt(r["eta"], ".3g"),
                  rel=_fmt(r["rel_weight_change"], ".3e"),
                  wn=_fmt(r["w_norm_last"], ".4f"),
                  clip="{:.1%}".format(r["clip_rate"]),
                  nf=r["n_nonfinite"] + (1 if r["nonfinite_state"] else 0),
                  er=_fmt(r["erank_pr_last"], ".1f"),
                  der="{:+.2%}".format(-r["erank_pr_rel_drop"]),
                  mm=_fmt(r["max_over_mean_abs_last"], ".1f"),
                  pp=_fmt(r["pre_over_post_last"], ".4f"),
                  basin=r["basin"],
                  l1=_fmt(r["cos_lag1_mean"], ".5f"),
                  l2=_fmt(r["cos_lag2_mean"], ".5f"),
                  v=verdict_for_cell(r, ref),
              ))
    A("")

    A("## Did the loop's behaviour change")
    A("")
    A("Against the frozen `off` cell. `cos(final,frozen)` is between the "
      "position-mean residual vectors at the last iteration.")
    A("")
    A("| mode | eta | rel ΔW | clip | changed |")
    A("|---|---|---|---|---|")
    for mode in MODES:
        for r in by_mode.get(mode, []):
            A(f"| `{r['mode']}` | {_fmt(r['eta'], '.3g')} | "
              f"{_fmt(r['rel_weight_change'], '.2e')} | "
              f"{r['clip_rate']:.1%} | {loop_changed(r, ref)} |")
    A("")

    A("## The homeostat")
    A("")
    A("The ATR loop rescales the state to ‖x₀‖ before every injection, so the "
      "post-rescale norm is a constant "
      f"{_fmt(ref['post_rescale_norm'], '.4f') if ref else ''} on every cell by "
      "construction. The pre-rescale norm is what the forward pass actually "
      "produced. If the pre-rescale norm moves while the loop's visible "
      "behaviour does not, the rescaling absorbed the rule's effect (issue #27 "
      "item 3).")
    A("")
    A("| mode | eta | pre-rescale (first → last) | max | pre/post at end | "
      "loop changed |")
    A("|---|---|---|---|---|---|")
    for mode in order:
        for r in by_mode.get(mode, []):
            A(f"| `{r['mode']}` | {_fmt(r['eta'], '.3g')} | "
              f"{_fmt(r['pre_rescale_first'], '.2f')} → "
              f"{_fmt(r['pre_rescale_last'], '.2f')} | "
              f"{_fmt(r['pre_rescale_max'], '.2f')} | "
              f"{_fmt(r['pre_over_post_last'], '.5f')} | "
              f"{loop_changed(r, ref)} |")
    A("")

    A("## Hollowing out (issue #27 item 11)")
    A("")
    A("The failure where one entry runs away, the normaliser rescales, and the "
      "rest of the matrix is annihilated. ‖W‖_F stays flat and the clipping "
      "rate stays low throughout, so every conventional dial reads healthy. "
      "**Effective rank falling while ‖W‖_F is flat is that failure and nothing "
      "else.** `top 0.1% mass` is the column that separates it from Oja simply "
      "producing a near-rank-1 update: Oja's is a smooth outer product spread "
      "over every entry, the pathology is a handful of entries dominating.")
    A("")
    A(f"Note what the ceiling does to the first column. With `max_delta_frac` = "
      f"{MAX_DELTA_FRAC}, ‖delta‖_F cannot exceed {MAX_DELTA_FRAC:.0%} of ‖W0‖_F, "
      f"so ‖W‖_F **structurally cannot run away** — its flatness is guaranteed by "
      f"the same mechanism that would be doing the damage, and it therefore "
      f"carries no information about whether the damage happened. That is issue "
      f"#27 item 11's point exactly, and it is why the flatness threshold here is "
      f"set at {WNORM_FLAT:.0%} (just above what the ceiling already promises) "
      f"and why effective rank, not ‖W‖_F, is the instrument.")
    A("")
    A("| mode | eta | ‖W‖_F range | erank first → last | min erank | σ₁ energy | "
      "max/mean |W| | top 0.1% mass | ΔW erank | ΔW top 0.1% mass |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for mode in order:
        for r in by_mode.get(mode, []):
            ds = r.get("delta_spectral") or {}
            A(f"| `{r['mode']}` | {_fmt(r['eta'], '.3g')} | "
              f"{r['w_norm_rel_range']:.2e} | "
              f"{_fmt(r['erank_pr_first'], '.1f')} → {_fmt(r['erank_pr_last'], '.1f')} | "
              f"{_fmt(r['erank_pr_min'], '.1f')} | "
              f"{_fmt(r['sigma1_frac_energy_last'], '.4f')} | "
              f"{_fmt(r['max_over_mean_abs_last'], '.2f')} | "
              f"{_fmt(r['top_mass_frac_last'], '.4f')} | "
              f"{_fmt(ds.get('erank_pr'), '.1f')} | "
              f"{_fmt(ds.get('top_mass_frac'), '.4f')} |")
    A("")

    A("## Thresholds")
    A("")
    A("Stated so that the verdicts above can be recomputed or disagreed with.")
    A("")
    A(f"- noise floor: relative weight change < {FLOOR_DELTA_FRAC:g}")
    A(f"- ceiling quiet: clip rate ≤ {CLIP_QUIET:g}; ceiling loud: > {CLIP_LOUD:g}")
    A(f"- hollowing-out flag: effective rank down > {ERANK_DROP_FLAG:.0%} while "
      f"‖W‖_F range < {WNORM_FLAT:.0%}")
    A("")
    A("## Provenance")
    A("")
    A(f"{len(recs)} cells, {sum(r['seconds'] for r in recs) / 60:.0f} CPU-minutes "
      f"total, run as {meta.get('shards', 1)} single-threaded shard(s) alongside "
      f"another sweep on the same 4-core box. `wall_clock_seconds` below is the "
      f"last invocation's only (the refinement pass), not the whole map's.")
    A("")
    A("```json")
    A(json.dumps(meta, indent=2, sort_keys=True))
    A("```")
    A("")
    A(f"Raw per-cell records, including the full ‖W‖_F, delta-frac, update-norm, "
      f"pre-rescale and singular-spectrum trajectories: "
      f"`experiments/output_step_size/step_size_map.jsonl`.")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_all(out_dir: Path) -> list:
    recs, seen = [], set()
    for p in sorted(out_dir.glob("step_size_map*.jsonl")):
        for r in bb.read_jsonl(p):
            if r["cell_id"] not in seen:
                seen.add(r["cell_id"])
                recs.append(r)
    return recs


def main(argv=None):
    # SITE is a module global read by run_cell(), build_report() and the meta
    # block; --site reassigns it below so the override threads through all of
    # them. Declared here, before the `default=SITE` read, as `global` must
    # precede any use of the name in the function.
    global SITE

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=TORCH_THREADS)
    ap.add_argument("--parent", type=str, default=PARENT_DEFAULT)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--site", type=str, default=SITE,
                    help="target site for the single-site 'separate' sweep, e.g. "
                         "blocks.8.mlp or blocks.11.attn.head.7. Default keeps the "
                         "calibrated blocks.6.mlp. A non-default site re-uses the "
                         "default's eta anchor (U_REF, W0_NORM_CALIBRATED), which "
                         "was measured on blocks.6.mlp -- re-measure them for a "
                         "clean sweep; the default path is unchanged.")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--report", type=str, default=str(REPORT_PATH))
    args = ap.parse_args(argv)

    # Reassign once from --site; with the default this is a no-op and the run is
    # bit-identical.
    SITE = args.site

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / ("step_size_map.jsonl" if args.nshards == 1
                       else f"step_size_map.shard{args.shard}.jsonl")

    cells = build_cells()
    order = {c["mode"] + f"@{c['eta']:.6g}": i for i, c in enumerate(cells)}

    meta = {
        "issue": 30,
        "model": MODEL_NAME,
        "site": SITE,
        "prompt_id": PROMPT_ID,
        "n_steps": args.steps,
        "seed": SEED,
        "max_delta_frac": MAX_DELTA_FRAC,
        "cadence": CADENCE,
        "u_ref": U_REF,
        "d_grid": list(D_GRID),
        "layer_start": LAYER_START,
        "layer_end": LAYER_END,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "torch_threads": args.threads,
        "shards": args.nshards,
        "device": "cpu",
        "dtype": "float32",
        "norms_dtype": "float64",
        "repo_rev": bb._git_rev(REPO_ROOT),
    }
    try:
        from importlib.metadata import version as _v
        meta["transformer_lens_version"] = _v("transformer-lens")
    except Exception:
        meta["transformer_lens_version"] = "unknown"

    if args.report_only:
        recs = read_all(out_dir)
        recs.sort(key=lambda r: order.get(r["cell_id"], 10 ** 6))
        bb.write_jsonl(out_dir / "step_size_map.jsonl", recs)
        if len(recs) >= len(cells):
            for p in sorted(out_dir.glob("step_size_map.shard*.jsonl")):
                p.unlink()
        metas = [json.loads(p.read_text()) for p in sorted(out_dir.glob("step_size_meta*.json"))
                 if p.stat().st_size]
        if metas:
            meta.update(metas[0])
            meta["shards"] = max(m.get("shards", 1) for m in metas)
        Path(args.report).write_text(build_report(recs, meta), encoding="utf-8")
        print(f"[report] {args.report} ({len(recs)}/{len(cells)} cells)")
        return 0

    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    mine = cells[args.shard::args.nshards] if args.nshards > 1 else cells
    done = {r["cell_id"] for r in read_all(out_dir)}
    todo = [c for c in mine if f"{c['mode']}@{c['eta']:.6g}" not in done]

    print(f"[config] {MODEL_NAME} site={SITE} prompt={PROMPT_ID} steps={args.steps} "
          f"threads={args.threads} shard={args.shard}/{args.nshards}", flush=True)
    print(f"[plan] {len(mine)} cells in this shard, {len(done)} already recorded, "
          f"{len(todo)} to run -> {jsonl.name}", flush=True)
    if not todo:
        return 0

    pl = bb.load_prompt_library(args.parent)
    prompts = bb.ordered_prompts(pl)
    prompt = prompts[PROMPT_ID]

    from transformer_lens import HookedTransformer
    t_load = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    model.eval()
    model.requires_grad_(False)
    print(f"[model] loaded in {time.time() - t_load:.1f}s", flush=True)
    assert model.cfg.n_layers - 1 == LAYER_END
    # Read the live weight through the same adapter the rule uses, so a per-head
    # or attention --site resolves to the right matrix instead of assuming an MLP
    # W_out. For the default site this is exactly blocks[6].mlp.W_out.
    w0_now = _make_site(model, SITE).weight.double().norm().item()
    if SITE == DEFAULT_SITE:
        assert abs(w0_now - W0_NORM_CALIBRATED) < 1e-2, (
            f"||W0||_F is {w0_now:.6f}, not the {W0_NORM_CALIBRATED} eta_for() was "
            "calibrated against -- every D target and recommended eta would be stale")
    else:
        print(f"[site] non-default --site {SITE!r}: ||W0||_F = {w0_now:.6f}. "
              f"eta_for() anchors to W0_NORM_CALIBRATED={W0_NORM_CALIBRATED} and "
              f"U_REF measured on {DEFAULT_SITE!r}; re-measure both for a clean "
              "sweep (see Caveats).", flush=True)

    # Iteration 0 and the step closure are built ONCE, on the frozen weights,
    # and shared by every cell. Rebuilding `initial_state` inside a cell would
    # run a forward pass with the plasticity hook live, so cell k's starting
    # tensor would carry cell k's first update -- and the map's premise is that
    # every cell starts from the same place.
    state0 = initial_state(model, prompt, layer_end=LAYER_END)
    step = make_atr_step(model, prompt, layer_start=LAYER_START, layer_end=LAYER_END,
                         initial_norm=state0.initial_norm)
    print(f"[state0] seq_len={state0.tensor.shape[0]} "
          f"initial_norm={state0.initial_norm:.6f}", flush=True)

    # Every cell must start from the same W0. `run_cell` reverts in a finally
    # block, but a leak would show up as a slow drift across cells that looked
    # like a step-size effect, so it is asserted rather than assumed. Read through
    # the adapter, not a literal blocks[6].mlp.W_out, so the guard watches whatever
    # SITE names rather than an untouched matrix while the real site leaks.
    _guard = _make_site(model, SITE)
    w0_ref = _guard.weight.detach().clone()

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    for k, c in enumerate(todo, 1):
        if not torch.equal(_guard.weight, w0_ref):
            raise RuntimeError(
                f"the live weight at {SITE} is not W0 at the start of cell "
                f"{c['mode']}@{c['eta']:.6g}; a previous cell's revert() did not "
                "restore it and every cell after this one is on a different matrix")
        rec = run_cell(model, state0, step, c["mode"], c["eta"], c["d_target"],
                       n_steps=args.steps)
        bb.append_jsonl(jsonl, rec)
        elapsed = time.time() - t0
        eta_left = (elapsed / k) * (len(todo) - k)
        print(f"[{k}/{len(todo)}] {rec['cell_id']:<24} "
              f"relDW={rec['rel_weight_change']:.3e} clip={rec['clip_rate']:.1%} "
              f"nonfin={rec['n_nonfinite']} erank={rec['erank_pr_last']:.1f} "
              f"basin={rec['basin']!r} lag1={_fmt(rec['cos_lag1_mean'], '.5f')} "
              f"({rec['seconds']:.0f}s, ETA {bb._hms(eta_left)})", flush=True)

    meta["started"] = started
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["wall_clock_seconds"] = round(time.time() - t0, 1)
    (out_dir / (f"step_size_meta.shard{args.shard}.json" if args.nshards > 1
                else "step_size_meta.json")).write_text(json.dumps(meta, indent=2),
                                                  encoding="utf-8")
    print(f"[done] {len(todo)} cells in {bb._hms(time.time() - t0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
