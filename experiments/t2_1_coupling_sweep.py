"""T2.1 -- does the feedback share grow with coupling, and ever change the outcome?

Issue #48. Answers the open row C-35; the durable claim is C-58. The
interpretation is fixed in experiments/output_t2_1/PREREGISTRATION.md, which
was committed before this ran.

One matched-arms cell per coupling setting, varied one axis at a time around
the EXP-001 working point (hebb, blocks.6.mlp, A01_physics, seed 0, cadence 1,
120 steps, eta* = 7.065171428571429e-05):

  eta ladder   {0.25x, 0.5x, 1x, 2x, 4x} eta*
  cadence      apply every {2, 4} at eta*
  length       {60, 240} steps at eta*

plus one severed cell (loop 0->3) at the base point re-verifying the exact-zero
recomputed floor. Records append to t2_1_coupling.jsonl as they complete;
--resume skips finished cell ids.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

from atr_bridge import initial_state, make_atr_step                  # noqa: E402
from offline_control import (                                        # noqa: E402
    compare_states,
    run_matched_arms,
)
sys.path.insert(0, _here)
import baseline_basins as bb                                         # noqa: E402

# The working point, byte-for-byte the EXP-001 constants.
SITE = "blocks.6.mlp"
LAYER_START = 0
LAYER_END = 11
LAYER_END_SEVERED = 3
MODE = "hebb"
MAX_DELTA_FRAC = 0.05
BASE_N_STEPS = 120
BASE_CADENCE = 1
ETA_STAR = 1.8e-2 * 164.854 / (BASE_N_STEPS * 350.0)   # 7.065171428571429e-05
SEED = 0
PROMPT_ID = "A01_physics"
PROMPT = "The implications of quantum entanglement suggest that"

LAG_WINDOW = 12
MAX_LAG = 4

OUT_DIR = os.path.join(_here, "output_t2_1")
JSONL = os.path.join(OUT_DIR, "t2_1_coupling.jsonl")
META = os.path.join(OUT_DIR, "meta.json")

# The pre-registered grid: one axis moved per cell, base point included once.
CELLS: list[dict] = (
    [{"cell_id": f"eta_{m}x", "axis": "eta", "eta": m * ETA_STAR,
      "cadence": BASE_CADENCE, "n_steps": BASE_N_STEPS,
      "layer_end": LAYER_END}
     for m in (0.25, 0.5, 1.0, 2.0, 4.0)]
    + [{"cell_id": f"cadence_{k}", "axis": "cadence", "eta": ETA_STAR,
        "cadence": k, "n_steps": BASE_N_STEPS, "layer_end": LAYER_END}
       for k in (2, 4)]
    + [{"cell_id": f"steps_{n}", "axis": "n_steps", "eta": ETA_STAR,
        "cadence": BASE_CADENCE, "n_steps": n, "layer_end": LAYER_END}
       for n in (60, 240)]
    + [{"cell_id": "severed_base", "axis": "severed", "eta": ETA_STAR,
        "cadence": BASE_CADENCE, "n_steps": BASE_N_STEPS,
        "layer_end": LAYER_END_SEVERED}]
)


def basin_of(model, r: torch.Tensor) -> dict:
    """The parent's readout at the last position: label, margin, top-5."""
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
        "final_tensor_norm": float(traj[-1].double().norm()),
    })
    return out


def rerun_under(model, w: torch.Tensor, r0: torch.Tensor, step, n_steps: int):
    """The loop run frozen under an installed weight, trajectory kept."""
    from offline_control import installed_weight, _frozen_trajectory
    with installed_weight(model, SITE, w):
        return _frozen_trajectory(model, r0, step, n_steps)


def run_cell(model, cell: dict) -> dict:
    t0 = time.time()
    s0 = initial_state(model, PROMPT, layer_end=cell["layer_end"])
    step = make_atr_step(model, PROMPT, layer_start=LAYER_START,
                         layer_end=cell["layer_end"],
                         initial_norm=s0.initial_norm)

    res = run_matched_arms(
        model, s0.tensor, step, SITE, cell["n_steps"],
        eta=cell["eta"], mode=MODE, max_delta_frac=MAX_DELTA_FRAC,
        seed=SEED, apply_every=cell["cadence"],
        y_source="recorded", also_recomputed_y=True,
        rerun_frozen=False,
        keep_states=True,
    )

    ver = res.verification
    rec: dict = {
        "cell_id": cell["cell_id"],
        "axis": cell["axis"],
        "kind": "severed" if cell["layer_end"] == LAYER_END_SEVERED else "routed",
        "prompt_id": PROMPT_ID,
        "prompt": PROMPT,
        "seed": SEED,
        "site": SITE,
        "mode": MODE,
        "eta": cell["eta"],
        "eta_over_eta_star": cell["eta"] / ETA_STAR,
        "cadence": cell["cadence"],
        "n_steps": cell["n_steps"],
        "layer_start": LAYER_START,
        "layer_end": cell["layer_end"],
        "max_delta_frac": MAX_DELTA_FRAC,
        "initial_norm": s0.initial_norm,

        "arms_matched": ver["ok"],
        "n_axes_checked": len(ver["axes"]),
        "arms_mismatched_axes": [a["axis"] for a in ver["axes"] if not a["match"]],

        "clipped_closed": res.closed.report["clipped"],
        "clipped_offline_recorded": res.offline.report["clipped"],
        "clipped_offline_recomputed": res.offline_recomputed_y.report["clipped"],
        "nonfinite_closed": res.closed.report["nonfinite"],
        "n_updates": res.closed.config.n_updates,
        "rel_weight_change_closed": res.closed.report["delta_frac"],
        "rel_weight_change_offline_recomputed":
            res.offline_recomputed_y.report["delta_frac"],

        "weight_recorded": res.comparison["weight"],
        "weight_recomputed_y": res.comparison["weight_recomputed_y"],
    }

    # Behavioural readout: the loop run frozen under each arm's final matrix
    # from the same start state, exactly EXP-001's protocol.
    traj = {"frozen": list(res.record.states)}
    traj["closed"] = rerun_under(model, res.closed.weight, s0.tensor, step,
                                 cell["n_steps"])
    traj["offline_recomputed"] = rerun_under(
        model, res.offline_recomputed_y.weight, s0.tensor, step,
        cell["n_steps"])

    rec["readout"] = {k: trajectory_stats(model, v) for k, v in traj.items()}
    rec["state_closed_vs_offline_recomputed"] = compare_states(
        traj["closed"][-1], traj["offline_recomputed"][-1])

    ro_c = rec["readout"]["closed"]
    ro_o = rec["readout"]["offline_recomputed"]
    rec["arms_basin_agree"] = ro_c["basin"] == ro_o["basin"]
    rec["arms_basin_margins"] = [ro_c["top_logit_margin"],
                                 ro_o["top_logit_margin"]]

    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    done: set[str] = set()
    if args.resume and os.path.exists(JSONL):
        with open(JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["cell_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[resume] {len(done)} cells already recorded: {sorted(done)}")

    from transformer_lens import HookedTransformer
    import importlib.metadata
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()

    if not (args.resume and os.path.exists(META)):
        meta = {
            "experiment": "T2.1 coupling sweep",
            "issue": 48,
            "answers": ["C-35"],
            "durable_claim": "C-58",
            "model": "gpt2-small",
            "site": SITE,
            "prompt_id": PROMPT_ID,
            "mode": MODE,
            "eta_star": ETA_STAR,
            "max_delta_frac": MAX_DELTA_FRAC,
            "seed": SEED,
            "n_cells": len(CELLS),
            "cells": [c["cell_id"] for c in CELLS],
            "device": "cpu",
            "dtype": "float32",
            "norms_dtype": "float64",
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "transformer_lens_version":
                importlib.metadata.version("transformer-lens"),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(META, "w") as f:
            json.dump(meta, f, indent=1)

    for cell in CELLS:
        if cell["cell_id"] in done:
            continue
        print(f"[cell] {cell['cell_id']} starting", flush=True)
        rec = run_cell(model, cell)
        with open(JSONL, "a") as f:
            f.write(json.dumps(rec) + "\n")
        w = rec["weight_recomputed_y"]
        print(f"[cell] {cell['cell_id']} done in {rec['seconds']}s: "
              f"diff_over_drift(recomputed)={w['diff_over_drift']:.6e} "
              f"basins closed={rec['readout']['closed']['basin']!r} "
              f"offline={rec['readout']['offline_recomputed']['basin']!r} "
              f"agree={rec['arms_basin_agree']} "
              f"clipped={rec['clipped_closed']}", flush=True)

    print("[sweep] complete", flush=True)


if __name__ == "__main__":
    main()
