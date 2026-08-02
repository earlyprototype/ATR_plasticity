"""EXP-003 Stage 0: validation of the grid measurement.

Protocol and thresholds: `experiments/output_exp003/PREREGISTRATION.md`.
Results: `experiments/output_exp003/STAGE0_RESULTS.md`.

WHAT IT MEASURES. The depth-weighted centroid over the 144-site grid, on the
frozen model, against the committed 125-input baseline where the classifications
are already recorded. No plasticity and no injected signal.

REGISTERED GATES AND THRESHOLDS.

  Gate 1  separates the five committed end states, ratio above 1.5.
  Gate 2  separates the 34 two-step cycles from the 91 fixed points, above 1.5.
  Gate 3  does NOT separate random halves of the largest group: below 1.2 in at
          least nine of ten splits.

The 1.5 figure was set against 0.87, which is what the token labels score on the
same scale, computed from register row C-07. A value of 1.0 means groups are
indistinguishable from their own internal scatter.

CONTROLS.

  A  permuting block labels must score below the true statistic.
  B  permuting head labels must change nothing, threshold 1e-9. Head indices are
     arbitrary, so this control has a known correct answer.

Every record is written to `output_exp003/stage0.jsonl` as it is produced.

Usage:
    .venv/bin/python experiments/exp003_stage0.py --limit 8      # smoke
    .venv/bin/python experiments/exp003_stage0.py                # whole census
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
import os
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import atr_bridge                                          # noqa: E402
import mea_grid                                            # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "experiments" / "output_baseline" / "basins.jsonl"
OUT_DIR = ROOT / "experiments" / "output_exp003"

# Matches the committed census: it reports basin_at_120 alongside the 300-iteration
# basin and they agree on 124 of 125, so 120 is where the parent's own numbers say
# the trajectory has settled. Running 300 would triple the cost for one prompt's
# worth of difference.
N_ITER = 120

# The committed census uses a 25-iteration window for its lag statistics. The
# settled value of the statistic is averaged over the same window rather than read
# off the last iterate alone, because a period-2 orbit's last iterate depends on
# which phase the run happened to stop in.
SETTLE_WINDOW = 25

GATE_SEPARATION = 1.5
GATE_FAILURE_DIRECTION = 1.2
N_SHUFFLES = 10
N_RANDOM_SPLITS = 10
SEED = 20260802


def load_census() -> list[dict]:
    """The committed 125-input baseline census, one record per input."""
    with open(BASELINE) as f:
        return [json.loads(line) for line in f]


def dynamical_class(row: dict) -> str:
    """The committed baseline's own classification, not a new one.

    `converged_lag1` true means a fixed point. False with `converged_lag2` true
    means a two-step cycle. The census reports 91 and 34 of these respectively and
    zero of neither.
    """
    if row.get("converged_lag1"):
        return "fixed-point"
    if row.get("converged_lag2"):
        return "period-2"
    return "unsettled"


def run_one(model, row: dict) -> dict:
    """One prompt: iterate, and record the centre of activity at every step."""
    prompt = row["prompt"]
    t0 = time.time()

    s0 = atr_bridge.initial_state(model, prompt)
    step = mea_grid.grid_step(model, prompt, initial_norm=s0.initial_norm)

    r = s0.tensor.clone()
    depth_trace, head_trace = [], []
    settled_activity = []

    for i in range(N_ITER):
        r, activity = step(model, r)
        depth_trace.append(mea_grid.ca_depth(activity))
        head_trace.append(mea_grid.ca_head(activity))
        if i >= N_ITER - SETTLE_WINDOW:
            settled_activity.append(activity)

    # The settled value: the statistic averaged over the settle window. Kept
    # alongside the raw trace so a later reader can recompute it differently.
    settled_depth = statistics.fmean(depth_trace[-SETTLE_WINDOW:])

    # The shuffled versions are computed from the SAME activity tensors, so the
    # comparison isolates the permutation and not a second forward pass.
    # A stable digest, not Python's `hash`. String hashing is randomised per
    # process unless PYTHONHASHSEED is fixed, so the shuffle draws would not be
    # reproducible from SEED alone.
    digest = int.from_bytes(hashlib.sha256(row["prompt_id"].encode()).digest()[:4], "big")
    gen = torch.Generator().manual_seed((SEED + digest) % (2**31))
    layer_shuffled, head_shuffled = [], []
    for _ in range(N_SHUFFLES):
        layer_shuffled.append(statistics.fmean(
            mea_grid.ca_depth(mea_grid.shuffle_layers(a, gen)) for a in settled_activity
        ))
        head_shuffled.append(statistics.fmean(
            mea_grid.ca_depth(mea_grid.shuffle_heads(a, gen)) for a in settled_activity
        ))

    # The per-layer mass profile at settle, so a later reader can do more than the
    # scalar centroid without paying for the run again. Twelve numbers per prompt.
    mass_profile = torch.stack([
        a.heads.sum(dim=1) + a.mlp for a in settled_activity
    ]).mean(dim=0)
    head_profile = torch.stack([a.heads for a in settled_activity]).mean(dim=0)

    return {
        "prompt_id": row["prompt_id"],
        "basin": row["basin"],
        "dyn_class": dynamical_class(row),
        "settled_ca_depth": settled_depth,
        "settled_mass_per_layer": [float(x) for x in mass_profile],
        "settled_mass_grid": [[float(x) for x in r_] for r_ in head_profile],
        "settled_ca_head": statistics.fmean(head_trace[-SETTLE_WINDOW:]),
        "ca_depth_trace": depth_trace,
        "ca_depth_layer_shuffled": layer_shuffled,
        "ca_depth_head_shuffled": head_shuffled,
        "seconds": round(time.time() - t0, 2),
    }


def group(rows: list[dict], key: str, value: str = "settled_ca_depth") -> dict[str, list[float]]:
    """Collect one measured value per input, bucketed by the named field."""
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r[value])
    return out


def analyse(rows: list[dict]) -> dict:
    """Every gate and control, computed from the per-prompt records."""
    by_basin = group(rows, "basin")
    by_class = group(rows, "dyn_class")

    basin_ratio = mea_grid.separation_ratio(by_basin)
    class_ratio = mea_grid.separation_ratio(by_class)

    # Control A: the layer shuffle, one ratio per shuffle index, so the spread
    # across shuffles is visible rather than averaged away.
    layer_shuffled_ratios = []
    for j in range(N_SHUFFLES):
        shuffled = {}
        for r in rows:
            shuffled.setdefault(r["basin"], []).append(r["ca_depth_layer_shuffled"][j])
        layer_shuffled_ratios.append(mea_grid.separation_ratio(shuffled))

    # Control B: the head shuffle. Correct answer is that it changes nothing, so
    # the reported quantity is the largest deviation from the true value anywhere.
    head_shuffle_max_dev = max(
        abs(h - r["settled_ca_depth"])
        for r in rows for h in r["ca_depth_head_shuffled"]
    )

    # Gate 3: random halves of the largest basin must not look separated.
    largest = max(by_basin.items(), key=lambda kv: len(kv[1]))[0]
    pool = [r["settled_ca_depth"] for r in rows if r["basin"] == largest]
    rng = random.Random(SEED)
    split_ratios = []
    for _ in range(N_RANDOM_SPLITS):
        shuffled_pool = pool[:]
        rng.shuffle(shuffled_pool)
        half = len(shuffled_pool) // 2
        split_ratios.append(mea_grid.separation_ratio(
            {"a": shuffled_pool[:half], "b": shuffled_pool[half:]}
        ))
    n_below = sum(1 for s in split_ratios if s < GATE_FAILURE_DIRECTION)

    return {
        "kind": "analysis",
        "n_prompts": len(rows),
        "basin_counts": {k: len(v) for k, v in by_basin.items()},
        "class_counts": {k: len(v) for k, v in by_class.items()},
        "basin_means": {k: statistics.fmean(v) for k, v in by_basin.items()},
        "class_means": {k: statistics.fmean(v) for k, v in by_class.items()},

        "gate1_basin_separation": basin_ratio,
        "gate1_pass": bool(basin_ratio > GATE_SEPARATION),
        "gate2_class_separation": class_ratio,
        "gate2_pass": bool(class_ratio > GATE_SEPARATION),
        "gate3_random_split_ratios": split_ratios,
        "gate3_n_below_threshold": n_below,
        "gate3_pass": bool(n_below >= 9),

        "controlA_layer_shuffled_ratios": layer_shuffled_ratios,
        "controlA_layer_shuffled_mean": statistics.fmean(layer_shuffled_ratios),
        "controlA_pass": bool(statistics.fmean(layer_shuffled_ratios) < basin_ratio),

        "controlB_head_shuffle_max_deviation": head_shuffle_max_dev,
        "controlB_pass": bool(head_shuffle_max_dev < 1e-9),

        "thresholds": {
            "separation": GATE_SEPARATION,
            "failure_direction": GATE_FAILURE_DIRECTION,
            "token_label_baseline_from_C07": 0.87,
        },
        "largest_basin": largest,
    }


def main() -> int:
    """Run the census, write every record as it is produced, then report the gates."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run a stratified sample of N prompts drawn across all end states")
    ap.add_argument("--out", default=str(OUT_DIR / "stage0.jsonl"))
    args = ap.parse_args()
    if args.limit < 0:
        # A negative limit silently produces an empty selection and a run over zero
        # inputs, which would report gates computed from nothing.
        ap.error("--limit must be 0 (the whole census) or a positive count")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from transformer_lens import HookedTransformer
    print("[setup] loading gpt2 small on cpu", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    census = load_census()
    if args.limit:
        # Stratified so a smoke run still contains every basin and both classes.
        by_basin: dict[str, list[dict]] = {}
        for r in census:
            by_basin.setdefault(r["basin"], []).append(r)
        picked, i = [], 0
        while len(picked) < args.limit:
            added = False
            for b in sorted(by_basin):
                if i < len(by_basin[b]) and len(picked) < args.limit:
                    picked.append(by_basin[b][i]); added = True
            if not added:
                break
            i += 1
        census = picked

    print(f"[setup] {len(census)} prompts, {N_ITER} iterations each", flush=True)

    rows = []
    # PARTIAL SUFFIX: write to a scratch path and rename only on success, so an
    # in-flight artifact can never be mistaken for, or committed as, a finished one.
    partial = args.out + ".partial"
    with open(partial, "w") as f:
        f.write(json.dumps({
            "kind": "meta",
            "n_iter": N_ITER,
            "settle_window": SETTLE_WINDOW,
            "n_prompts": len(census),
            "seed": SEED,
            "torch": torch.__version__,
            "gates": {
                "separation": GATE_SEPARATION,
                "failure_direction": GATE_FAILURE_DIRECTION,
            },
        }) + "\n")
        f.flush()

        t_start = time.time()
        for n, row in enumerate(census, 1):
            rec = run_one(model, row)
            rec["kind"] = "prompt"
            rows.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            elapsed = time.time() - t_start
            rate = elapsed / n
            print(f"[{n}/{len(census)}] {rec['prompt_id']:22s} "
                  f"{rec['basin']:12s} {rec['dyn_class']:12s} "
                  f"ca={rec['settled_ca_depth']:.4f} "
                  f"{rec['seconds']:.1f}s  eta {(len(census)-n)*rate/60:.1f}min",
                  flush=True)

        result = analyse(rows)
        f.write(json.dumps(result) + "\n")

    os.replace(partial, args.out)

    print("\n=== STAGE 0 ===", flush=True)
    print(f"gate 1 basins       {result['gate1_basin_separation']:.4f} "
          f"(need > {GATE_SEPARATION}, labels score 0.87)  "
          f"{'PASS' if result['gate1_pass'] else 'FAIL'}")
    print(f"gate 2 class        {result['gate2_class_separation']:.4f}  "
          f"{'PASS' if result['gate2_pass'] else 'FAIL'}")
    print(f"gate 3 failure dir  {result['gate3_n_below_threshold']}/10 below "
          f"{GATE_FAILURE_DIRECTION}  {'PASS' if result['gate3_pass'] else 'FAIL'}")
    print(f"control A shuffle   {result['controlA_layer_shuffled_mean']:.4f} "
          f"vs true {result['gate1_basin_separation']:.4f}  "
          f"{'PASS' if result['controlA_pass'] else 'FAIL'}")
    print(f"control B heads     max dev {result['controlB_head_shuffle_max_deviation']:.2e}  "
          f"{'PASS' if result['controlB_pass'] else 'FAIL'}")
    print(f"\nbasin means: {result['basin_means']}")
    print(f"class means: {result['class_means']}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
