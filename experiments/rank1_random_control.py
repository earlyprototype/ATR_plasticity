#!/usr/bin/env python3
"""
T1.4 -- the rank-matched random control. Can falsify C-22.

WHAT THIS ASKS
--------------
`hebb` at eta 7.065e-05 flips `A01_physics` from `prolet` to `comrade` with the
norm ceiling silent. `CLAIMS.md` C-22 says the flip needs both a sufficient
magnitude AND the right sign on the hebb/oja axis -- i.e. it is not reproduced
by an arbitrary edit of the same size.

The isotropic `random` arm cannot test that: it spreads its Frobenius norm over
~719 singular directions, so it never reaches hebb's operator norm anywhere in
the sweep (max sigma1 0.4469 against hebb's 1.8135). "Random doesn't flip" is
therefore consistent with "random is operator-norm tiny", and decides nothing.

This file runs the control that does decide it: a **rank-1 random direction**,
matched to hebb on the two quantities that actually characterise the edit.

  Arm A -- matched operator norm.  dW = c * u v^T, u,v random unit vectors,
           c chosen so sigma1 equals hebb's exactly. For a rank-1 matrix with
           unit factors, sigma1 = ||.||_F = c, so this also lands just under
           hebb's Frobenius norm (hebb is 95.8% rank-1, not exactly rank-1).

  Arm B -- matched loop displacement. The review's F4 shows sigma1-matching
           leaves the loop perturbation 24-42x apart between modes, so sigma1
           alone is not a sufficient match. Arm B binary-searches c until the
           settled state moves as far from the frozen state as hebb's does
           (1 - cos ~ 5.0e-03), then asks the same question.

READING THE RESULT
------------------
  Any seed flips to `comrade`  ->  C-22 falsifiable claim is DEAD. The basin
                                   change follows from a rank-1 edit of
                                   sufficient magnitude in ANY direction, and
                                   "structured, not generic" must be retired.
  No seed flips, both arms      ->  C-22 survives this test. Direction (or at
                                   least non-genericity) is doing real work.
  Mixed                         ->  report the flip rate; it is a probability,
                                   not a yes/no, and C-22 becomes a statement
                                   about how much of the sphere flips.

Fixed in advance, per the register's own rule: the verdict above is written
before the run, and the flip criterion is the basin label at the end of the
same 120-step episode hebb used. No threshold is chosen after seeing numbers.

HARNESS
-------
Deliberately the same objects `basin_bifurcation.py` uses -- `initial_state`,
`make_atr_step`, `basin_of`, `OjaPlasticity` for W0/site access -- so a
difference between this file's hebb row and the published one is a harness bug,
not a finding. The hebb row is re-run here for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atr_bridge import initial_state, make_atr_step            # noqa: E402
from plasticity import OjaPlasticity                           # noqa: E402

# --- Config: every value matches basin_bifurcation.py / the step-size map ----
MODEL_NAME = "gpt2-small"
SITE = "blocks.6.mlp"
PROMPT = "The implications of quantum entanglement suggest that"
PROMPT_ID = "A01_physics"
ETA = 7.065171428571429e-05
MODE = "hebb"
CADENCE = 1
MAX_DELTA_FRAC = 0.05
SEED = 0
LAYER_START = 0
LAYER_END = 11
N_EPISODE = 120

# Arm B search: stop when the loop displacement is within this relative
# tolerance of hebb's, or after this many bisection steps.
DISP_TOL = 0.02
MAX_BISECT = 8


def basin_of(model, state: torch.Tensor) -> dict:
    """Top-1 token of the settled state at the last position -- the project's
    basin definition (`baseline_basins.py:401-403`)."""
    with torch.no_grad():
        logits = model.unembed(model.ln_final(state.unsqueeze(0)))[0, -1]
        top = torch.topk(logits, 5)
    toks = [model.to_string(int(i)).strip() for i in top.indices]
    return {
        "basin": toks[0],
        "top5": toks,
        "margin": float(top.values[0] - top.values[1]),
    }


def posmean(state: torch.Tensor) -> torch.Tensor:
    """Position-mean (768,) -- the representation BASELINE.md's scatter uses."""
    return state.mean(dim=0)


def one_minus_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - F.cosine_similarity(a.reshape(1, -1).double(),
                                           b.reshape(1, -1).double()).item())


def run_frozen(model, step, st0_tensor: torch.Tensor, n: int) -> torch.Tensor:
    """Iterate the frozen map n times under whatever weight is currently live."""
    r = st0_tensor.clone()
    for _ in range(n):
        r = step(model, r)
    return r


def rank1_random(g: torch.Generator, n_in: int, n_out: int,
                 scale: float, device, dtype) -> torch.Tensor:
    """c * u v^T with u, v drawn uniform on their unit spheres.

    Exactly rank 1, so sigma1 == ||.||_F == scale. Drawing Gaussian and
    normalising is the standard uniform-on-sphere construction."""
    u = torch.randn(n_in, generator=g, device=device, dtype=torch.float32)
    v = torch.randn(n_out, generator=g, device=device, dtype=torch.float32)
    u = u / u.norm()
    v = v / v.norm()
    return (scale * torch.outer(u, v)).to(dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of random rank-1 directions per arm")
    ap.add_argument("--out", default="experiments/output_rank1_random")
    ap.add_argument("--skip-arm-b", action="store_true",
                    help="run only the matched-sigma1 arm (much faster)")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    from transformer_lens import HookedTransformer
    import transformer_lens
    print(f"[setup] loading {MODEL_NAME} ...", flush=True)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    st0 = initial_state(model, PROMPT, layer_end=LAYER_END)
    step = make_atr_step(model, PROMPT, layer_start=LAYER_START,
                         layer_end=LAYER_END, initial_norm=st0.initial_norm)

    plast = OjaPlasticity(model, SITE, eta=ETA, mode=MODE, cadence=CADENCE,
                          max_delta_frac=MAX_DELTA_FRAC, seed=SEED)
    W0 = plast.W0.clone()
    n_in, n_out = W0.shape
    print(f"[setup] site {SITE} W0 {tuple(W0.shape)} "
          f"||W0||_F {W0.norm().item():.4f}", flush=True)

    # -- Reference 0: the frozen loop under W0 --------------------------------
    t = time.time()
    off_state = run_frozen(model, step, st0.tensor, N_EPISODE)
    off_b = basin_of(model, off_state)
    off_pm = posmean(off_state)
    dt = time.time() - t
    print(f"[off ] basin={off_b['basin']!r} margin={off_b['margin']:.4f} "
          f"({dt:.1f}s for {N_EPISODE} steps)", flush=True)
    records.append({"arm": "off", "basin": off_b["basin"], "top5": off_b["top5"],
                    "margin": off_b["margin"], "seconds": dt})

    # -- Reference 1: hebb, re-run here so a harness bug is visible ------------
    plast.install()
    r = st0.tensor.clone()
    for _ in range(N_EPISODE):
        r = step(model, r)
        plast.apply()
    hebb_state = r.clone()
    dW_hebb = plast.delta.clone()
    rep = plast.report()
    plast.remove()
    plast._site.write(W0)                      # restore before anything else

    sv = torch.linalg.svdvals(dW_hebb.double())
    hebb_sigma1 = float(sv[0])
    hebb_fro = float(dW_hebb.double().norm())
    hebb_disp = one_minus_cos(posmean(hebb_state), off_pm)
    hebb_b = basin_of(model, hebb_state)
    print(f"[hebb] basin={hebb_b['basin']!r} sigma1={hebb_sigma1:.4f} "
          f"||dW||_F={hebb_fro:.4f} 1-cos(off)={hebb_disp:.4e} "
          f"clip={rep.get('clip_rate', float('nan')):.3f}", flush=True)
    records.append({
        "arm": "hebb", "eta": ETA, "basin": hebb_b["basin"], "top5": hebb_b["top5"],
        "margin": hebb_b["margin"], "sigma1": hebb_sigma1, "fro": hebb_fro,
        "frac_energy_1": float(sv[0] ** 2 / (sv ** 2).sum()),
        "disp_1mcos": hebb_disp, "clip_rate": rep.get("clip_rate"),
        "rel_weight_change": rep.get("rel_change"),
    })

    HARNESS_OK = hebb_b["basin"] == "comrade" and off_b["basin"] == "prolet"
    print(f"[chk ] harness reproduces published prolet->comrade: {HARNESS_OK}",
          flush=True)

    def eval_scale(g_seed: int, scale: float) -> dict:
        """Install W0 + rank-1 random at this scale, run frozen, measure."""
        g = torch.Generator(device="cpu").manual_seed(g_seed)
        dW = rank1_random(g, n_in, n_out, scale, W0.device, W0.dtype)
        plast._site.write(W0 + dW)
        try:
            st = run_frozen(model, step, st0.tensor, N_EPISODE)
        finally:
            plast._site.write(W0)
        b = basin_of(model, st)
        return {
            "scale": scale, "basin": b["basin"], "top5": b["top5"],
            "margin": b["margin"],
            "disp_1mcos": one_minus_cos(posmean(st), off_pm),
            "sigma1": scale, "fro": scale,
        }

    # -- Arm A: matched operator norm -----------------------------------------
    print(f"\n[armA] matched sigma1 = {hebb_sigma1:.4f}, {args.seeds} seeds",
          flush=True)
    for s in range(args.seeds):
        t = time.time()
        rec = eval_scale(1000 + s, hebb_sigma1)
        rec.update({"arm": "rank1_random_matched_sigma1", "seed": 1000 + s,
                    "seconds": time.time() - t,
                    "flipped": rec["basin"] != off_b["basin"]})
        records.append(rec)
        print(f"[armA] seed={1000+s} basin={rec['basin']!r:<14} "
              f"1-cos(off)={rec['disp_1mcos']:.4e} "
              f"(hebb {hebb_disp:.4e}) flip={rec['flipped']}", flush=True)

    # -- Arm B: matched loop displacement -------------------------------------
    if not args.skip_arm_b:
        print(f"\n[armB] matched loop displacement = {hebb_disp:.4e}, "
              f"{args.seeds} seeds", flush=True)
        for s in range(args.seeds):
            lo, hi = hebb_sigma1, hebb_sigma1
            # Expand upward until displacement brackets hebb's.
            probe = eval_scale(1000 + s, hi)
            n_eval = 1
            while probe["disp_1mcos"] < hebb_disp and hi < 200 * hebb_sigma1:
                lo, hi = hi, hi * 2.0
                probe = eval_scale(1000 + s, hi)
                n_eval += 1
            best = probe
            for _ in range(MAX_BISECT):
                if abs(best["disp_1mcos"] - hebb_disp) / hebb_disp <= DISP_TOL:
                    break
                mid = 0.5 * (lo + hi)
                probe = eval_scale(1000 + s, mid)
                n_eval += 1
                if probe["disp_1mcos"] < hebb_disp:
                    lo = mid
                else:
                    hi = mid
                best = probe
            best.update({"arm": "rank1_random_matched_disp", "seed": 1000 + s,
                         "n_evals": n_eval,
                         "flipped": best["basin"] != off_b["basin"]})
            records.append(best)
            print(f"[armB] seed={1000+s} scale={best['scale']:.4f} "
                  f"basin={best['basin']!r:<14} "
                  f"1-cos(off)={best['disp_1mcos']:.4e} "
                  f"flip={best['flipped']} ({n_eval} evals)", flush=True)

    # -- Verdict ---------------------------------------------------------------
    armA = [r for r in records if r.get("arm") == "rank1_random_matched_sigma1"]
    armB = [r for r in records if r.get("arm") == "rank1_random_matched_disp"]
    flips_A = sum(1 for r in armA if r["flipped"])
    flips_B = sum(1 for r in armB if r["flipped"])

    meta = {
        "experiment": "T1.4 rank-1 random control",
        "model": MODEL_NAME, "site": SITE, "prompt_id": PROMPT_ID,
        "n_episode": N_EPISODE, "seeds": args.seeds,
        "hebb": {"eta": ETA, "sigma1": hebb_sigma1, "fro": hebb_fro,
                 "disp_1mcos": hebb_disp, "basin": hebb_b["basin"]},
        "off_basin": off_b["basin"],
        "harness_reproduces_published": HARNESS_OK,
        "flips_matched_sigma1": f"{flips_A}/{len(armA)}",
        "flips_matched_disp": f"{flips_B}/{len(armB)}" if armB else "not run",
        "torch": torch.__version__,
        "transformer_lens": getattr(transformer_lens, "__version__", "unknown"),
        "python": platform.python_version(),
    }

    (out_dir / "rank1_random.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("\n" + "=" * 66)
    print(f"harness reproduces published prolet->comrade : {HARNESS_OK}")
    print(f"matched-sigma1 arm flipped                   : {flips_A}/{len(armA)}")
    if armB:
        print(f"matched-displacement arm flipped             : {flips_B}/{len(armB)}")
    print("=" * 66)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
