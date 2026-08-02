"""EXP-003 Stage 2: adjustment cadence.

Protocol and thresholds: `experiments/output_exp003/PREREGISTRATION.md`, as
amended by Amendment 1 there. Results:
`experiments/output_exp003/STAGE2_RESULTS.md`.

WHAT IT VARIES. How often the weights are adjusted, at a fixed 120-iteration
episode, with the step size multiplied by the cadence. Cadences 1, 4 and 12 give
120, 30 and 10 adjustments.

WHAT IS HELD EQUAL, AND THE GUARD ON IT. The intent is equal total adjustment with
only its granularity changing. The achieved drift is reported at every setting,
and if it varies by more than a factor of two across the ladder the comparison is
qualitative only and the falsifier is not invoked.

THE MEASUREMENT. Census agreement: of 31 fresh inputs, how many settle where the
frozen model puts them, counting only those that settled.

SETTLED MEANS FIXED POINT OR TWO-STEP CYCLE, by the committed baseline's
classifier. A criterion built on consecutive steps alone would score the census's
eight `Divine` inputs as unsettled, because that end state is a two-step cycle
with consecutive-step agreement near 0.68, and would set the baseline at 23 of 31
rather than 31 of 31.

THE FALSIFIER, as restated by Amendment 1. If census agreement at cadence 12 does
not exceed cadence 1 by at least 5 of 31, the result is recorded as not supported
at this resolution rather than as a refutation, because the ladder spans a factor
of 12 rather than the 100 originally registered.

Usage:
    .venv/bin/python experiments/exp003_stage2.py --reprompts 31
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import atr_bridge                                          # noqa: E402
from multi_site import MultiSitePlasticity, SiteSpec       # noqa: E402
# The readout and the lag scan are imported rather than reimplemented, for the
# same reason the ATR loop is: a second implementation of either would be a second
# definition of what a basin is, and the reference gate would stop meaning anything.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_basins as bb                               # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "output_exp003"
BASELINE = ROOT / "experiments" / "output_baseline" / "basins.jsonl"

DRIVEN_PROMPT_ID = "A01_physics"
N_ITER = 120
CEILING = 1e9
CADENCE_LADDER = (1, 4, 12)          # Amendment 1
SETTLE_GATE = 0.999                  # the parent's gate, used unchanged
LAG_WINDOW = 25

# EXP-002's per-layer anchored step sizes, verbatim from its committed record.
BASE_ETAS = [
    5.665036555555215e-07, 6.939986736112205e-06, 3.452525056779305e-06,
    4.588478481371286e-05, 6.915565746725232e-05, 7.631763587858851e-05,
    7.716275812045379e-05, 6.910422786183772e-05, 5.397873822782322e-05,
    3.91104498477202e-05, 1.6096948643176555e-05, 1.1061980757811969e-06,
]
SITES = [f"blocks.{i}.mlp" for i in range(12)]

FALSIFIER_MARGIN = 5


def load_census() -> list[dict]:
    """The committed 125-input baseline census, one record per input."""
    with open(BASELINE) as f:
        return [json.loads(line) for line in f]


def pick_reprompts(rows: list[dict], n: int) -> list[dict]:
    """A stratified draw over the five frozen end states, driven prompt excluded.

    Same construction as EXP-002's so the two are comparable: every end state gets
    at least one input, order inside a group taken by input id so the draw is
    reproducible without a seed.
    """
    pool = [r for r in rows if r["prompt_id"] != DRIVEN_PROMPT_ID]
    by_basin: dict[str, list[dict]] = {}
    for r in sorted(pool, key=lambda r: r["prompt_id"]):
        by_basin.setdefault(r["basin"], []).append(r)
    picked, i = [], 0
    while len(picked) < n:
        added = False
        for b in sorted(by_basin):
            if i < len(by_basin[b]) and len(picked) < n:
                picked.append(by_basin[b][i])
                added = True
        if not added:
            break
        i += 1
    return picked


def classify(traj: list[torch.Tensor]) -> tuple[str, float, float]:
    """Fixed point, two-step cycle, or unsettled, by the committed baseline's rule.

    Uses the baseline's own `lag_scan` over the mean vector, its 25-iteration
    window and its 0.999 gate, so the classes here are the classes in
    `basins.jsonl` rather than a second opinion about them.
    """
    tail = traj[-(LAG_WINDOW + 2):]
    stack = torch.stack([x.mean(dim=0) for x in tail])
    scan = bb.lag_scan(stack, 2)
    c1 = scan.get(1, {}).get("mean", float("nan"))
    c2 = scan.get(2, {}).get("mean", float("nan"))
    if c1 > SETTLE_GATE:
        return "fixed-point", c1, c2
    if c2 > SETTLE_GATE:
        return "period-2", c1, c2
    return "unsettled", c1, c2


def basin_of(model, state: torch.Tensor) -> str:
    """The committed readout, imported. `state` is (seq, d_model); the baseline
    reads the last position."""
    return bb.readout_detail(model, state[-1, :])["top_token_strings"][0].strip()


def run_census(model, reprompts: list[dict]) -> dict:
    """Run every fresh input under whatever weights are currently installed."""
    agree = 0
    settled = 0
    rows = []
    for r in reprompts:
        s0 = atr_bridge.initial_state(model, r["prompt"])
        step = atr_bridge.make_atr_step(model, r["prompt"], initial_norm=s0.initial_norm)
        x = s0.tensor.clone()
        traj = [x]
        for _ in range(N_ITER):
            x = step(model, x)
            traj.append(x)
        cls, c1, c2 = classify(traj)
        word = basin_of(model, x)
        is_settled = cls in ("fixed-point", "period-2")
        matches = is_settled and word.strip() == r["basin"].strip()
        settled += int(is_settled)
        agree += int(matches)
        rows.append({"prompt_id": r["prompt_id"], "frozen_basin": r["basin"],
                     "word": word, "class": cls, "cos_lag1": c1, "cos_lag2": c2,
                     "settled": is_settled, "agrees": matches})
    return {"census_agreement": agree, "n_settled": settled,
            "n": len(reprompts), "rows": rows}


def main() -> int:
    """Run the reference gate, then each cadence cell, then report whether drift matched."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--reprompts", type=int, default=31)
    ap.add_argument("--out", default=str(OUT_DIR / "stage2.jsonl"))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from transformer_lens import HookedTransformer
    print("[setup] loading gpt2 small on cpu", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = load_census()
    by_id = {r["prompt_id"]: r for r in rows}
    driven = by_id[DRIVEN_PROMPT_ID]
    reprompts = pick_reprompts(rows, args.reprompts)
    print(f"[setup] {len(reprompts)} fresh inputs over "
          f"{len(set(r['basin'] for r in reprompts))} end states", flush=True)

    # PARTIAL SUFFIX: see exp003_stage0.py. Rename on success only.
    partial = args.out + ".partial"
    out = open(partial, "w")
    out.write(json.dumps({"kind": "meta", "ladder": list(CADENCE_LADDER),
                          "n_iter": N_ITER, "amendment": 1,
                          "falsifier_margin": FALSIFIER_MARGIN}) + "\n")
    out.flush()

    # --- reference gate: the census must reproduce itself before anything moves
    t0 = time.time()
    ref = run_census(model, reprompts)
    ref["kind"] = "reference"
    out.write(json.dumps(ref) + "\n")
    out.flush()
    print(f"[gate] reference census agreement {ref['census_agreement']}/{ref['n']}, "
          f"settled {ref['n_settled']}/{ref['n']}  {time.time()-t0:.0f}s", flush=True)
    if ref["census_agreement"] != ref["n"]:
        print("[gate] WARNING reference census does not reproduce itself; every "
              "number below is suspect", flush=True)

    results = {}
    for k in CADENCE_LADDER:
        tk = time.time()
        driver = MultiSitePlasticity(model, [
            SiteSpec(s, mode="hebb", eta=e * k, max_delta_frac=CEILING)
            for s, e in zip(SITES, BASE_ETAS, strict=True)
        ])
        s0 = atr_bridge.initial_state(model, driven["prompt"])
        step = atr_bridge.make_atr_step(model, driven["prompt"],
                                        initial_norm=s0.initial_norm)
        x = s0.tensor.clone()
        with driver:
            n_applied = 0
            for i in range(N_ITER):
                x = step(model, x)
                if (i + 1) % k == 0:
                    driver.apply()
                    n_applied += 1
            rep = driver.report()
            drift = float(rep.get("delta_frac", float("nan")))
            clipped = bool(rep.get("clipped"))
            nonfinite = bool(rep.get("nonfinite"))
            # Weights stay where they drifted for the census, then are reverted.
            cen = run_census(model, reprompts)
        # `with` exit reverts every matrix bit-exactly.

        rec = {"kind": "cell", "k": k, "n_applied": n_applied,
               "aggregate_drift": drift, "clipped": clipped,
               "nonfinite": nonfinite,
               "driven_word": basin_of(model, x),
               "census_agreement": cen["census_agreement"],
               "n_settled": cen["n_settled"], "n": cen["n"],
               "rows": cen["rows"], "seconds": round(time.time() - tk, 1)}
        results[k] = rec
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[k={k:3d}] applied {n_applied:3d}  drift {drift:.6f}  "
              f"clip={clipped}  agreement {cen['census_agreement']}/{cen['n']}  "
              f"settled {cen['n_settled']}/{cen['n']}  "
              f"{time.time()-tk:.0f}s", flush=True)

    drifts = [results[k]["aggregate_drift"] for k in CADENCE_LADDER]
    spread = max(drifts) / min(drifts) if min(drifts) > 0 else float("inf")
    lo, hi = CADENCE_LADDER[0], CADENCE_LADDER[-1]
    delta = results[hi]["census_agreement"] - results[lo]["census_agreement"]

    analysis = {
        "kind": "analysis",
        "reference_agreement": ref["census_agreement"],
        "agreement_by_k": {str(k): results[k]["census_agreement"] for k in CADENCE_LADDER},
        "settled_by_k": {str(k): results[k]["n_settled"] for k in CADENCE_LADDER},
        "drift_by_k": {str(k): results[k]["aggregate_drift"] for k in CADENCE_LADDER},
        "drift_spread": spread,
        "drift_matched": bool(spread <= 2.0),
        "delta_agreement_hi_minus_lo": delta,
        "falsifier_margin": FALSIFIER_MARGIN,
        "timescale_supported": bool(delta >= FALSIFIER_MARGIN and spread <= 2.0),
        "qualitative_only": bool(spread > 2.0),
    }
    out.write(json.dumps(analysis) + "\n")
    out.close()
    os.replace(partial, args.out)

    print("\n=== STAGE 2 ===")
    print(f"reference agreement      {ref['census_agreement']}/{ref['n']}")
    for k in CADENCE_LADDER:
        r = results[k]
        print(f"k={k:3d}  applied {r['n_applied']:3d}  drift {r['aggregate_drift']:.6f}  "
              f"agreement {r['census_agreement']}/{r['n']}  settled {r['n_settled']}/{r['n']}")
    print(f"drift spread across ladder {spread:.3f}x "
          f"({'matched' if spread <= 2.0 else 'NOT matched, comparison qualitative only'})")
    print(f"agreement at k={hi} minus k={lo}: {delta:+d} "
          f"(need >= {FALSIFIER_MARGIN})")
    print(f"TIMESCALE READING: {'SUPPORTED' if analysis['timescale_supported'] else 'NOT SUPPORTED at this resolution'}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
