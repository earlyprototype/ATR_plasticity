"""EXP-003 Stage 1: spectral concentration of the adjusted matrices.

Protocol and thresholds: `experiments/output_exp003/PREREGISTRATION.md`.
Results: `experiments/output_exp003/STAGE1_RESULTS.md`.

WHAT IT MEASURES. Participation-ratio effective rank and top singular share, on
each of the twelve adjusted matrices, before and after EXP-002's reinforcing
episode. The effective rank measure is the one `experiments/step_size_map.py`
already uses, so the number is comparable with what the project has recorded.

REGISTERED THRESHOLDS.

  Supported     mean effective rank falls by >= 10%.
  Refuted       it falls by < 2%.
  Inconclusive  anything between.

The 10% figure was set against the largest movement anywhere in the committed
step-size map, a rise of about 0.6 percent from 642.6 to 646.7.

REPRODUCTION RATHER THAN APPROXIMATION. The per-block step sizes below are taken
verbatim from EXP-002's committed record (`output_exp002/exp002_uncapped.jsonl`,
the `step2_episode` unit for `hebb`), as are the driven input, the step count and
the lifted ceiling. If the aggregate drift does not match EXP-002's
0.013111766434820447 within 5 percent, the run reports a failed reproduction
rather than reporting figures from a different episode.

Usage:
    .venv/bin/python experiments/exp003_stage1.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import atr_bridge                                          # noqa: E402
from multi_site import MultiSitePlasticity, SiteSpec       # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "output_exp003"

DRIVEN_PROMPT_ID = "A01_physics"
N_STEPS = 120
CADENCE = 1
CEILING = 1e9                    # lifted, per the operator decision governing EXP-002

# Verbatim from EXP-002's committed step2_episode record for `hebb`.
ETAS = [
    5.665036555555215e-07,
    6.939986736112205e-06,
    3.452525056779305e-06,
    4.588478481371286e-05,
    6.915565746725232e-05,
    7.631763587858851e-05,
    7.716275812045379e-05,
    6.910422786183772e-05,
    5.397873822782322e-05,
    3.91104498477202e-05,
    1.6096948643176555e-05,
    1.1061980757811969e-06,
]
SITES = [f"blocks.{i}.mlp" for i in range(12)]

EXP002_AGGREGATE_DRIFT = 0.013111766434820447
REPRODUCTION_TOLERANCE = 0.05    # relative; a 5% miss means a different episode

GATE_SUPPORTED_FALL = 0.10
GATE_REFUTED_FALL = 0.02


def erank(m: torch.Tensor) -> float:
    """Participation-ratio effective rank, float64.

    Copied in form from `experiments/exp002_distributed.py` and
    `experiments/step_size_map.py` so the number means the same thing here as in
    the committed step-size map, whose ceiling-silent cells never fall below 642.4
    against a frozen 642.6.
    """
    sv = torch.linalg.svdvals(m.double())
    s2 = sv * sv
    tot = s2.sum().item()
    return (sv.sum().item() ** 2 / tot) if tot > 0 else float("nan")


def top_singular_share(m: torch.Tensor) -> float:
    """The share of the matrix's total spectral weight held by its top direction.

    A more direct reading of "has this become a valve" than effective rank: it goes
    to 1 for a rank-one matrix. Reported alongside because the mechanism is stated
    in terms of one direction dominating, and effective rank answers a slightly
    broader question.
    """
    sv = torch.linalg.svdvals(m.double())
    return float(sv[0] / sv.sum())


def main() -> int:
    """Reproduce EXP-002's episode, then measure spectral concentration on the twelve matrices."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "stage1.jsonl"

    from transformer_lens import HookedTransformer
    print("[setup] loading gpt2 small on cpu", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with open(ROOT / "experiments" / "output_baseline" / "basins.jsonl") as f:
        rows = {r["prompt_id"]: r for r in (json.loads(line) for line in f)}
    prompt = rows[DRIVEN_PROMPT_ID]["prompt"]

    records = []
    t0 = time.time()

    # --- frozen reference, before anything moves -------------------------------
    driver = MultiSitePlasticity(model, [
        SiteSpec(s, mode="hebb", eta=e, max_delta_frac=CEILING)
        for s, e in zip(SITES, ETAS, strict=True)
    ])
    # `W0` is each sub-instance's own frozen copy, taken at construction. Using it
    # rather than re-reading the live weight means the reference is the same object
    # the ceiling and the revert guarantee are measured against.
    frozen_by_site = {sub.site: sub.W0.detach().clone() for sub in driver}
    assert set(frozen_by_site) == set(SITES), "driver sites do not match the spec list"

    frozen_stats = {
        site: {"erank": erank(w), "top_share": top_singular_share(w),
               "fro": float(w.double().norm())}
        for site, w in frozen_by_site.items()
    }
    print("[frozen] mean effective rank "
          f"{statistics.fmean(v['erank'] for v in frozen_stats.values()):.2f}", flush=True)

    # --- the episode, reproducing EXP-002's hebb closed arm --------------------
    s0 = atr_bridge.initial_state(model, prompt)
    step = atr_bridge.make_atr_step(model, prompt, initial_norm=s0.initial_norm)

    r = s0.tensor.clone()
    with driver:
        for i in range(N_STEPS):
            r = step(model, r)
            if (i + 1) % CADENCE == 0:
                driver.apply()
            if (i + 1) % 30 == 0:
                rep = driver.report()
                print(f"  [{i+1}/{N_STEPS}] drift {rep.get('delta_frac', float('nan')):.6f} "
                      f"clipped={rep.get('clipped')} {time.time()-t0:.0f}s", flush=True)

        report = driver.report()
        drifted_by_site = {
            sub.site: sub._site.weight.detach().clone() for sub in driver
        }

    aggregate = float(report.get("delta_frac", float("nan")))
    rel_miss = abs(aggregate - EXP002_AGGREGATE_DRIFT) / EXP002_AGGREGATE_DRIFT
    reproduced = rel_miss < REPRODUCTION_TOLERANCE
    print(f"[episode] aggregate drift {aggregate:.12f} vs EXP-002 "
          f"{EXP002_AGGREGATE_DRIFT:.12f}  rel miss {rel_miss:.4f}  "
          f"{'REPRODUCED' if reproduced else 'DID NOT REPRODUCE'}", flush=True)

    drifted_stats = {
        site: {"erank": erank(w), "top_share": top_singular_share(w),
               "fro": float(w.double().norm())}
        for site, w in drifted_by_site.items()
    }

    # --- the gate --------------------------------------------------------------
    frozen_mean = statistics.fmean(v["erank"] for v in frozen_stats.values())
    drifted_mean = statistics.fmean(v["erank"] for v in drifted_stats.values())
    fall = (frozen_mean - drifted_mean) / frozen_mean

    if fall >= GATE_SUPPORTED_FALL:
        verdict = "supported"
    elif fall < GATE_REFUTED_FALL:
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    top_frozen = statistics.fmean(v["top_share"] for v in frozen_stats.values())
    top_drifted = statistics.fmean(v["top_share"] for v in drifted_stats.values())

    result = {
        "kind": "analysis",
        "reproduced_exp002": reproduced,
        "aggregate_drift": aggregate,
        "exp002_aggregate_drift": EXP002_AGGREGATE_DRIFT,
        "relative_miss": rel_miss,
        "clipped": bool(report.get("clipped")),
        "nonfinite": bool(report.get("nonfinite")),
        "frozen_mean_erank": frozen_mean,
        "drifted_mean_erank": drifted_mean,
        "fractional_fall": fall,
        "verdict": verdict,
        "frozen_mean_top_singular_share": top_frozen,
        "drifted_mean_top_singular_share": top_drifted,
        "per_site": [
            {
                "site": s,
                "erank_frozen": frozen_stats[s]["erank"],
                "erank_drifted": drifted_stats[s]["erank"],
                "erank_fall": (frozen_stats[s]["erank"] - drifted_stats[s]["erank"])
                              / frozen_stats[s]["erank"],
                "top_share_frozen": frozen_stats[s]["top_share"],
                "top_share_drifted": drifted_stats[s]["top_share"],
                "delta_frac": (float((drifted_by_site[s] - frozen_by_site[s]).double().norm())
                               / frozen_stats[s]["fro"]),
            }
            for s in SITES
        ],
        "gates": {
            "supported_fall": GATE_SUPPORTED_FALL,
            "refuted_fall": GATE_REFUTED_FALL,
            "step_size_map_largest_movement": 0.006,
        },
        "seconds": round(time.time() - t0, 1),
    }
    records.append(result)

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print("\n=== STAGE 1 ===")
    print(f"reproduces EXP-002   {reproduced}  (rel miss {rel_miss:.4f})")
    print(f"frozen  mean erank   {frozen_mean:.3f}")
    print(f"drifted mean erank   {drifted_mean:.3f}")
    print(f"fractional fall      {fall:+.5f}   "
          f"(supported >= {GATE_SUPPORTED_FALL}, refuted < {GATE_REFUTED_FALL})")
    if reproduced:
        print(f"VERDICT              {verdict.upper()}")
    else:
        print(f"VERDICT              {verdict.upper()}  <-- NOT VALID")
        print("                     the episode did not reproduce EXP-002, so these "
              "effective ranks\n                     describe a different episode and "
              "the verdict must not be quoted")
    print(f"top singular share   {top_frozen:.5f} -> {top_drifted:.5f}")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
