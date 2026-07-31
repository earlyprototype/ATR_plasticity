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
Two DIFFERENT questions, which the first revision of this file conflated:

  Q1  Does a random rank-1 edit move the basin AT ALL?
  Q2  Does it reproduce hebb's specific destination, `comrade`?

  Q1 yes  ->  "an arbitrary rank-1 edit cannot move the basin" is dead. What
              survives is at most a claim about WHICH basin you land in.
  Q2 yes  ->  C-22 is dead outright: hebb's direction is doing no work the
              magnitude was not already doing.
  Both no ->  C-22 survives this test.
  Mixed   ->  report both as rates over seeds. C-22 becomes a statement about
              how much of the sphere does what, not a yes/no.

PRE-REGISTRATION, AND AN AMENDMENT DECLARED AFTER SEEING DATA
-------------------------------------------------------------
The first revision fixed one criterion in advance: "any seed flips to
`comrade` -> C-22 is dead." That is Q2 only, and it was written believing the
interesting failure mode was reproducing hebb's destination.

Run 1 (10 seeds, arm A; 2 seeds, arm B) produced seed 1001: a random rank-1
direction that flipped the basin to **`Anarch`** -- not `comrade` -- at a
displacement BELOW hebb's. Under the letter of the original criterion that is
not a falsification. Under any honest reading it is a material result, because
"an arbitrary edit of this size cannot move the basin" is what the isotropic
control was always taken to show.

So the criterion is split into Q1/Q2 above. **This amendment was made after
seeing that data point and is declared here rather than applied silently.**
Run 1's arm B is discarded for two code defects found afterwards (the
bisection reported the last probe rather than the closest, and expansion
probes were dropped from the record); its seed-1001 observation is what
prompted this split and is re-derived from scratch here, same seeds.

The flip criterion itself is unchanged: the basin label at the end of the same
120-step episode hebb used. No threshold is chosen after seeing numbers.

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
    """`1 - cos(a, b)` in float64, the repo's standard state-distance metric.

    float64 on purpose: the quantities compared here run down to ~1e-09, and a
    float32 reduction over 2.36M elements carries ~5e-05 relative error -- four
    orders above the signal. Every derived number in this file is float64 for
    the same reason."""
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
    """Run the reference rows, then both random arms, and write the verdict.

    Order matters. The frozen (`off`) and `hebb` rows run first and the script
    aborts if they do not reproduce `prolet` -> `comrade`, because every number
    after that point is only meaningful against a harness known to reproduce
    the published episode. Writes `rank1_random.jsonl` (one record per arm per
    seed, including every Arm B probe) and `meta.json` (provenance and the
    flip rates) to --out."""
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

    # Resolve the installed distribution version from packaging metadata, not
    # from `transformer_lens.__version__` (absent in some builds -- it raised
    # AttributeError here, which an earlier revision silently recorded as
    # "unknown"). An artifact whose provenance says "unknown" cannot support a
    # reproducibility claim, so an unresolvable version is a hard failure.
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        TL_VERSION = _pkg_version("transformer-lens")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "transformer-lens distribution metadata unavailable; refusing to "
            "write an artifact with unknown provenance") from exc

    print(f"[setup] loading {MODEL_NAME} (transformer-lens {TL_VERSION}) ...",
          flush=True)
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
    # float64. A float32 .norm() over 2.36M elements accumulates ~4.7e-05
    # relative error here (164.8464 against the true 164.854073) -- enough to
    # look like a version discrepancy against the repo's canonical constant when
    # it is only the reduction's own round-off. Every derived quantity in this
    # file is computed in float64 for the same reason.
    w0n = float(W0.double().norm())
    print(f"[setup] site {SITE} W0 {tuple(W0.shape)} "
          f"||W0||_F {w0n:.6f} (float64; canonical 164.854073)", flush=True)

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
    if not HARNESS_OK:
        # A harness that does not reproduce the reference transition is a
        # defect, not a result. Producing random-control records under it would
        # give a verdict about the wrong map -- fail loudly instead.
        raise RuntimeError(
            f"Harness did not reproduce the reference basin transition: "
            f"off={off_b['basin']!r} (expected 'prolet'), "
            f"hebb={hebb_b['basin']!r} (expected 'comrade'). "
            f"No control records written.")

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
            # Expand upward until displacement brackets hebb's. EVERY probe is
            # kept: a basin flip seen during expansion is a real observation
            # about this direction and must not be discarded just because the
            # scale was not the one the search finally settled on.
            probe = eval_scale(1000 + s, hi)
            n_eval = 1
            all_probes = [probe]
            while probe["disp_1mcos"] < hebb_disp and hi < 200 * hebb_sigma1:
                lo, hi = hi, hi * 2.0
                probe = eval_scale(1000 + s, hi)
                n_eval += 1
                all_probes.append(probe)
            # Keep the probe CLOSEST to the target, not the last one evaluated.
            # Bisection's final probe can sit further from the target than an
            # earlier one, which would report an unmatched point as "matched".
            def err(p):
                """Relative distance from hebb's loop displacement, for
                choosing which probe is the best-matched one."""
                return abs(p["disp_1mcos"] - hebb_disp) / hebb_disp

            best = min(all_probes, key=err)
            for _ in range(MAX_BISECT):
                if err(best) <= DISP_TOL:
                    break
                mid = 0.5 * (lo + hi)
                probe = eval_scale(1000 + s, mid)
                n_eval += 1
                all_probes.append(probe)
                if probe["disp_1mcos"] < hebb_disp:
                    lo = mid
                else:
                    hi = mid
                best = min(all_probes, key=err)
            # A flip anywhere in the search is reported, not only at `best`.
            flips_seen = sorted(
                ({"scale": p["scale"], "basin": p["basin"],
                  "disp_1mcos": p["disp_1mcos"]}
                 for p in all_probes if p["basin"] != off_b["basin"]),
                key=lambda p: p["scale"])
            best.update({"arm": "rank1_random_matched_disp", "seed": 1000 + s,
                         "n_evals": n_eval,
                         "disp_rel_err": err(best),
                         "matched": err(best) <= DISP_TOL,
                         "flipped": best["basin"] != off_b["basin"],
                         "flipped_anywhere": bool(flips_seen),
                         "flips_seen": flips_seen,
                         # every probe for this seed, so a flip seen at ANY
                         # scale during the search is not silently discarded
                         "probes": [{"scale": p["scale"], "basin": p["basin"],
                                     "disp_1mcos": p["disp_1mcos"]}
                                    for p in all_probes]})
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
    # Separate from flips_B on purpose: a direction that flips at SOME scale
    # during the search but not at the matched point is a different fact from
    # one that never flips at all, and C-22's wording depends on which it is.
    flips_B_anywhere = sum(1 for r in armB if r.get("flipped_anywhere"))
    matched_B = sum(1 for r in armB if r.get("matched"))

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
        "flips_any_scale_disp": f"{flips_B_anywhere}/{len(armB)}" if armB else "not run",
        "matched_within_tol": f"{matched_B}/{len(armB)}" if armB else "not run",
        "disp_tol": DISP_TOL,
        "torch": torch.__version__,
        "transformer_lens": TL_VERSION,
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
        print(f"  ...flipped at ANY scale during search      : {flips_B_anywhere}/{len(armB)}")
        print(f"  ...actually matched within {DISP_TOL:.0%}            : {matched_B}/{len(armB)}")
    print("=" * 66)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
