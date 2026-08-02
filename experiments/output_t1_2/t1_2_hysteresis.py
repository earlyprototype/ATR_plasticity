"""
T1.2 — the hysteresis test (issue #46; ALIGNMENT_REVIEW.md T1.2; settles CLAIMS C-52,
independent cross-check on C-26 which T1.1/#45 retired).

The up-then-down alpha-sweep. Install the working-point edit `W0 + alpha*ΔW` as a STATIC
weight change at site `blocks.6.mlp`, seed each alpha from the PREVIOUS alpha's SETTLED state
(continuation), sweep alpha UP then DOWN over the same grid, and read the settled basin both
directions.

  GAP  (up-threshold prolet->comrade at a HIGHER alpha than the down-threshold
        comrade->prolet, by more than one grid step)  -> HYSTERESIS -> two resting states
        coexist over a band of alpha -> C-52 "yes"; tensions C-26's retirement.
  CLEAN RETRACE (same grid point both ways, gap <= one grid step)                  -> one
        continuously-moving fixed point the readout relabels -> C-52 "no"; corroborates
        C-26 `retired`.

Reimplements nothing; edits no shared source. Reuses via `import basin_bifurcation as bif`:
  - atr_bridge.initial_state / make_atr_step        (the loop body, verbatim from the parent)
  - plasticity.OjaPlasticity                        (the hebb rule + ΔW accumulation + ceiling)
  - baseline_basins.readout_detail                  (the exact basin readout every experiment uses)
  - bif config + helpers (basin_of, token_rank_logit, cosflat, relL2, _single_token_id,
    initial_state)                                  (the working-point cell, verbatim)
`_settle_step` is copied from the T1.1 runner (per method). The from-scratch D2 alpha-loop in
`basin_bifurcation.py:401-406` is NOT touched — it stays as the disclosed contrast.

Working point (basin_bifurcation.py, EXP-001): site blocks.6.mlp, hebb,
eta = 7.065171428571429e-05, cadence 1, max_delta_frac = 0.05, seed 0, prompt A01_physics,
120-step episode. ‖ΔW‖_F/‖W0‖_F = 0.011239..., ‖W0‖_F = 164.85407309107723 (float64).

Run:  .venv/bin/python experiments/output_t1_2/t1_2_hysteresis.py [--step 0.05|0.10]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

# This file lives at experiments/output_t1_2/ -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments"
for p in (str(REPO_ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from atr_bridge import initial_state, make_atr_step          # noqa: E402
from plasticity import OjaPlasticity                         # noqa: E402
import baseline_basins as bb                                 # noqa: E402
import basin_bifurcation as bif                              # noqa: E402

# Working-point config, taken verbatim from basin_bifurcation (do NOT redefine).
SITE = bif.SITE
PROMPT = bif.PROMPT
PROMPT_ID = bif.PROMPT_ID
ETA = bif.ETA
MODE = bif.MODE
CADENCE = bif.CADENCE
MAX_DELTA_FRAC = bif.MAX_DELTA_FRAC
SEED = bif.SEED
LAYER_START = bif.LAYER_START
LAYER_END = bif.LAYER_END
N_EPISODE = bif.N_EPISODE
MODEL_NAME = bif.MODEL_NAME
ANCHOR = bif.ANCHOR

# Per-alpha loop length: the episode length (rule 7 — fixed to the working point).
SWEEP_N = 120
GATE_N = 30           # zero-step-size bit-identity gate.
SETTLE_TAIL = 15      # settled = basin constant over the last SETTLE_TAIL steps.

OUT_DIR = REPO_ROOT / "experiments" / "output_t1_2"
JSONL_PATH = OUT_DIR / "t1_2_hysteresis.jsonl"
META_PATH = OUT_DIR / "meta.json"

basin_of = bif.basin_of
cosflat = bif.cosflat
relL2 = bif.relL2
token_rank_logit = bif.token_rank_logit

COMRADE_ID = 47998
PROLET_ID = 22758


def _settle_step(basins: list, tail: int):
    """Settled word = the basin that holds for the final `tail` steps (constant).
    Returns (settled_word, first_step_it_locks_in, is_settled). Steps are 1-based.
    (Copied verbatim from the T1.1 runner, per method.)"""
    if len(basins) < tail:
        return None, None, False
    final = basins[-1]
    settled = all(b == final for b in basins[-tail:])
    if not settled:
        return None, None, False
    lock = len(basins)
    for i in range(len(basins) - 1, -1, -1):
        if basins[i] == final:
            lock = i
        else:
            break
    return final, lock + 1, True  # +1 -> 1-based iteration number


def make_grid(step: float):
    """alpha = 0.00 .. 1.50 inclusive, given step. Rounded to avoid fp drift."""
    n = int(round(1.50 / step))
    return [round(k * step, 4) for k in range(n + 1)]


# Incremental, torn-line-resistant JSONL writer: truncate once, then append a
# block per alpha and flush. A killed run keeps every completed alpha.
def jsonl_truncate(path: Path):
    with path.open("w", encoding="utf-8"):
        pass


def jsonl_append(path: Path, recs: list):
    with path.open("a", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_alpha(model, plast, W0, dW, alpha, seed_state, initial_norm, tag):
    """Install W0+alpha*dW, seed from `seed_state`, iterate SWEEP_N steps on the
    given renorm shell. Returns (per_step_rows, summary, settled_state)."""
    plast._site.write(W0 + alpha * dW)
    stepx = make_atr_step(model, PROMPT, layer_start=LAYER_START,
                          layer_end=LAYER_END, initial_norm=initial_norm)
    r = seed_state.clone()
    prev1 = seed_state.clone()   # iter-0 reference for lag-1
    prev2 = None
    rows = []
    basins = []
    first_flip = None
    seed_word = basin_of(model, seed_state)["basin"]
    for i in range(1, SWEEP_N + 1):
        r = stepx(model, r)
        b = basin_of(model, r)
        lag1 = cosflat(r, prev1)
        lag2 = cosflat(r, prev2) if prev2 is not None else None
        basins.append(b["basin"])
        if b["basin"] != seed_word and first_flip is None:
            first_flip = {"iter": i, "from": seed_word, "to": b["basin"],
                          "token_id": b["basin_token_id"]}
        rows.append({
            "kind": f"{tag}_step", "alpha": alpha, "iter": i,
            "basin": b["basin"], "basin_token_id": b["basin_token_id"],
            "top5": b["top5"], "margin": b["margin"],
            "lag1_cos": lag1, "lag2_cos": lag2,
            "cos_to_seed": cosflat(r, seed_state), "relL2_to_seed": relL2(r, seed_state),
            "state_norm": float(r.norm()),
        })
        prev2 = prev1
        prev1 = r.clone()
    plast._site.write(W0)
    settled_word, settle_it, is_settled = _settle_step(basins, SETTLE_TAIL)
    settled_state = r.clone()
    summary = {
        "kind": f"{tag}_summary", "alpha": alpha, "tag": tag,
        "initial_norm": initial_norm, "n_steps": SWEEP_N,
        "seed_word": seed_word,
        "settled_word": settled_word, "settled_token_id": rows[-1]["basin_token_id"],
        "settle_step": settle_it, "is_settled": is_settled,
        "final_basin": basins[-1], "first_flip_from_seed": first_flip,
        "final_lag1_cos": rows[-1]["lag1_cos"], "final_lag2_cos": rows[-1]["lag2_cos"],
        "final_margin": rows[-1]["margin"], "final_top5": rows[-1]["top5"],
        "final_norm": float(r.norm()),
        "comrade_probe": token_rank_logit(model, r, COMRADE_ID),
        "prolet_probe": token_rank_logit(model, r, PROLET_ID),
    }
    return rows, summary, settled_state


def sweep(model, plast, W0, dW, grid, seed_state, shell, W0_init_norm, direction, variant):
    """One directional sweep. `shell` in {'per_alpha','fixed_W0'}. `direction` in
    {'up','down'} (grid already ordered). Returns (list_of_summaries, end_state)."""
    tag = f"{variant}_{direction}"
    carried = seed_state
    summaries = []
    for alpha in grid:
        if shell == "per_alpha":
            plast._site.write(W0 + alpha * dW)
            inorm = initial_state(model, PROMPT, layer_end=LAYER_END).initial_norm
            plast._site.write(W0)
        else:
            inorm = W0_init_norm
        rows, summ, carried = run_alpha(model, plast, W0, dW, alpha, carried, inorm, tag)
        summ["shell"] = shell
        summ["direction"] = direction
        summ["variant"] = variant
        summaries.append(summ)
        jsonl_append(JSONL_PATH, rows + [summ])
        sw = summ["settled_word"]
        print(f"[{tag}] a={alpha:.2f} shell={shell} settled={sw!r} "
              f"is_settled={summ['is_settled']} settle_step={summ['settle_step']} "
              f"lag1={summ['final_lag1_cos']:.6f} "
              f"comrade(r{summ['comrade_probe']['rank']},{summ['comrade_probe']['logit']:.3f}) "
              f"prolet(r{summ['prolet_probe']['rank']},{summ['prolet_probe']['logit']:.3f})",
              flush=True)
    return summaries, carried


def find_alpha_up(up_summaries):
    """First ASCENDING alpha whose settled word == comrade (settled cells only)."""
    for s in up_summaries:              # up_summaries are in ascending order
        if s["is_settled"] and s["settled_word"] == "comrade":
            return s["alpha"]
    return None


def find_alpha_down(down_summaries):
    """First DESCENDING alpha whose settled word == prolet (settled cells only)."""
    for s in down_summaries:            # down_summaries are in descending order
        if s["is_settled"] and s["settled_word"] == "prolet":
            return s["alpha"]
    return None


def run(model, step_size: float):
    records_meta: dict = {}
    t0 = time.time()
    jsonl_truncate(JSONL_PATH)

    grid_up = make_grid(step_size)          # ascending 0.00 .. 1.50
    grid_down = list(reversed(grid_up))     # descending 1.50 .. 0.00
    print(f"[grid] step={step_size} n={len(grid_up)} "
          f"[{grid_up[0]} .. {grid_up[-1]}]", flush=True)

    # -- Phase 0: iteration-0 state at W0 --------------------------------------
    st0 = initial_state(model, PROMPT, layer_end=LAYER_END)
    init_norm = st0.initial_norm

    # -- Phase 1: reproduce the working-point episode -> capture ΔW ------------
    plast = OjaPlasticity(model, SITE, eta=ETA, mode=MODE, cadence=CADENCE,
                          max_delta_frac=MAX_DELTA_FRAC, seed=SEED)
    W0 = plast.W0.clone()
    W0_norm = plast.W0_norm  # float64 Frobenius norm of W0

    step = make_atr_step(model, PROMPT, layer_start=LAYER_START,
                         layer_end=LAYER_END, initial_norm=init_norm)

    plast.install()
    r = st0.tensor.clone()
    for _ in range(N_EPISODE):
        r = step(model, r)
        plast.apply()
    comrade_state = r.clone()
    dW = plast.delta.clone()          # W_final - W0, exactly
    rep = plast.report()
    plast.remove()                    # detach hook; live weight is W0+ΔW here

    cb = basin_of(model, comrade_state)
    sv = torch.linalg.svdvals(dW.double())
    dW_sigma1 = float(sv[0])
    dW_fro = float(dW.double().norm())
    delta_frac = dW_fro / W0_norm     # float64
    episode_ok = (cb["basin"] == "comrade" and cb["basin_token_id"] == COMRADE_ID)
    episode_clipped = bool(rep["clipped"])
    epi_rec = {
        "kind": "reproduce_episode",
        "episode_ok": episode_ok, "n_applied": rep["n_applied"],
        "delta_frac": rep["delta_frac"], "delta_frac_f64": delta_frac,
        "clipped": episode_clipped,
        "clip_rate": 0.0 if not episode_clipped else "SEE_FINDING",
        "nonfinite": bool(rep["nonfinite"]),
        "init_norm": init_norm, "W0_norm": W0_norm,
        "dW_sigma1": dW_sigma1, "dW_fro": dW_fro,
        "closed_basin": cb["basin"], "closed_basin_token_id": cb["basin_token_id"],
        "closed_top5": cb["top5"],
        "anchor_delta_frac": ANCHOR["delta_frac"], "anchor_dW_sigma1": ANCHOR["dW_sigma1"],
    }
    jsonl_append(JSONL_PATH, [epi_rec])
    print(f"[episode] ok={episode_ok} basin={cb['basin']!r}({cb['basin_token_id']}) "
          f"delta_frac(f64)={delta_frac:.12f} sigma1={dW_sigma1:.9f} "
          f"clipped={episode_clipped} n_applied={rep['n_applied']}", flush=True)

    if not episode_ok:
        records_meta["blocker"] = (f"episode reproduced basin {cb['basin']!r} "
                                   f"({cb['basin_token_id']}), expected comrade({COMRADE_ID})")
        records_meta["total_seconds"] = round(time.time() - t0, 1)
        return records_meta

    # -- Phase 1b: frozen baseline -> the ORIGINAL FROZEN prolet settled state -
    plast.revert()                    # live weight back to W0 exactly
    restore_relL2 = relL2(plast._site.weight, W0)
    r = st0.tensor.clone()
    for _ in range(N_EPISODE):
        r = step(model, r)
    prolet_state = r.clone()          # <-- the up-sweep seed
    fb = basin_of(model, prolet_state)
    prolet_id = fb["basin_token_id"]
    frozen_prolet_ok = (fb["basin"] == "prolet" and prolet_id == PROLET_ID)
    fp_rec = {
        "kind": "frozen_prolet_state",
        "basin": fb["basin"], "basin_token_id": prolet_id, "top5": fb["top5"],
        "margin": fb["margin"], "state_norm": float(prolet_state.norm()),
        "restore_relL2_W_vs_W0": restore_relL2, "frozen_prolet_ok": frozen_prolet_ok,
    }
    jsonl_append(JSONL_PATH, [fp_rec])
    print(f"[frozen] prolet state basin={fb['basin']!r}({prolet_id}) "
          f"ok={frozen_prolet_ok} ‖state‖={float(prolet_state.norm()):.5f} "
          f"restore_relL2={restore_relL2:.2e}", flush=True)

    # -- GATE: alpha=0 bit-identity vs the frozen loop -------------------------
    ref_states = []
    r = prolet_state.clone()
    for _ in range(GATE_N):
        r = step(model, r)
        ref_states.append(r.clone())
    plast._site.write(W0 + 0.0 * dW)
    ctrl_states = []
    r = prolet_state.clone()
    for _ in range(GATE_N):
        r = step(model, r)
        ctrl_states.append(r.clone())
    plast._site.write(W0)
    max_abs = 0.0
    all_equal = True
    for a, b in zip(ref_states, ctrl_states):
        d = float((a - b).abs().max())
        max_abs = max(max_abs, d)
        if not torch.equal(a, b):
            all_equal = False
    gate_ok = all_equal and (max_abs == 0.0)
    plast._site.write(W0 + 0.0 * dW)
    weight_bit_identical = bool(torch.equal(plast._site.weight, W0))
    plast._site.write(W0)
    gate_rec = {
        "kind": "gate_alpha0", "n_steps": GATE_N, "bit_identical": gate_ok,
        "max_abs_diff": max_abs, "all_torch_equal": all_equal,
        "zero_edit_weight_bit_identical_to_W0": weight_bit_identical,
    }
    jsonl_append(JSONL_PATH, [gate_rec])
    print(f"[GATE] alpha=0 bit_identical={gate_ok} max_abs_diff={max_abs:.3e} "
          f"weight==W0:{weight_bit_identical}", flush=True)
    records_meta["gate_bit_identical"] = gate_ok
    if not gate_ok:
        records_meta["blocker"] = (f"alpha=0 gate NOT bit-identical: max_abs_diff={max_abs:.3e}")
        records_meta["total_seconds"] = round(time.time() - t0, 1)
        return records_meta

    common = dict(episode_ok=episode_ok, frozen_prolet_ok=frozen_prolet_ok,
                  delta_frac_f64=delta_frac, W0_norm_f64=W0_norm, dW_sigma1=dW_sigma1,
                  init_norm_W0=init_norm, comrade_token_id=COMRADE_ID,
                  prolet_token_id=prolet_id, grid_step=step_size,
                  grid_n=len(grid_up))
    records_meta.update(common)

    # ================= PRIMARY: per-alpha shell (pure function of alpha) =======
    print("\n[PRIMARY] shell = per-alpha initial_norm under W0+alpha*ΔW\n", flush=True)
    up_p, top_state_p = sweep(model, plast, W0, dW, grid_up, prolet_state,
                              "per_alpha", init_norm, "up", "primary")
    down_p, _ = sweep(model, plast, W0, dW, grid_down, top_state_p,
                      "per_alpha", init_norm, "down", "primary")
    alpha_up_p = find_alpha_up(up_p)
    alpha_down_p = find_alpha_down(down_p)
    gap_p = (round(alpha_up_p - alpha_down_p, 4)
             if (alpha_up_p is not None and alpha_down_p is not None) else None)
    verdict_p = _verdict(alpha_up_p, alpha_down_p, gap_p, step_size)
    prim_verdict = {
        "kind": "primary_verdict", "shell": "per_alpha",
        "alpha_up": alpha_up_p, "alpha_down": alpha_down_p, "gap": gap_p,
        "grid_step": step_size, "verdict": verdict_p,
    }
    jsonl_append(JSONL_PATH, [prim_verdict])
    print(f"\n[PRIMARY VERDICT] alpha_up={alpha_up_p} alpha_down={alpha_down_p} "
          f"gap={gap_p} -> {verdict_p}\n", flush=True)
    records_meta.update(primary_alpha_up=alpha_up_p, primary_alpha_down=alpha_down_p,
                        primary_gap=gap_p, primary_verdict=verdict_p)
    _write_partial_meta(records_meta, t0)

    # ================= ROBUSTNESS: fixed-W0 shell ==============================
    print("\n[ROBUST] shell = fixed W0 initial_norm (D1 shell), all alpha\n", flush=True)
    up_r, top_state_r = sweep(model, plast, W0, dW, grid_up, prolet_state,
                              "fixed_W0", init_norm, "up", "robust")
    down_r, _ = sweep(model, plast, W0, dW, grid_down, top_state_r,
                      "fixed_W0", init_norm, "down", "robust")
    alpha_up_r = find_alpha_up(up_r)
    alpha_down_r = find_alpha_down(down_r)
    gap_r = (round(alpha_up_r - alpha_down_r, 4)
             if (alpha_up_r is not None and alpha_down_r is not None) else None)
    verdict_r = _verdict(alpha_up_r, alpha_down_r, gap_r, step_size)
    rob_verdict = {
        "kind": "robust_verdict", "shell": "fixed_W0",
        "alpha_up": alpha_up_r, "alpha_down": alpha_down_r, "gap": gap_r,
        "grid_step": step_size, "verdict": verdict_r,
    }
    jsonl_append(JSONL_PATH, [rob_verdict])
    print(f"\n[ROBUST VERDICT] alpha_up={alpha_up_r} alpha_down={alpha_down_r} "
          f"gap={gap_r} -> {verdict_r}\n", flush=True)
    records_meta.update(robust_alpha_up=alpha_up_r, robust_alpha_down=alpha_down_r,
                        robust_gap=gap_r, robust_verdict=verdict_r,
                        robustness_agrees=(verdict_p == verdict_r))
    records_meta["total_seconds"] = round(time.time() - t0, 1)
    return records_meta


def _verdict(alpha_up, alpha_down, gap, step_size):
    if alpha_up is None or alpha_down is None:
        return f"INCONCLUSIVE (alpha_up={alpha_up}, alpha_down={alpha_down})"
    if gap > step_size + 1e-9:
        return "HYSTERESIS (gap > one grid step)"
    return "CLEAN RETRACE (gap <= one grid step)"


def build_meta(base: dict) -> dict:
    m = {
        "experiment": "T1.2 hysteresis test (up-then-down alpha-sweep)",
        "issue": 46, "settles": ["C-52"], "cross_checks": ["C-26"],
        "model": MODEL_NAME, "site": SITE,
        "site_alias": "transformer.h.6.mlp.c_proj (W_out, (3072,768))",
        "prompt_id": PROMPT_ID, "prompt": PROMPT,
        "mode": MODE, "eta": ETA, "cadence": CADENCE,
        "max_delta_frac": MAX_DELTA_FRAC, "seed": SEED,
        "layer_start": LAYER_START, "layer_end": LAYER_END,
        "n_episode_for_dW": N_EPISODE,
        "sweep_n_steps": SWEEP_N, "gate_n_steps": GATE_N, "settle_tail": SETTLE_TAIL,
        "reference_W0_norm_F": 164.85407309107723,
        "shell_primary": "per-alpha initial_norm from fresh forward pass under W0+alpha*dW",
        "shell_robust": "fixed W0 initial_norm (D1 shell)",
        "seed_rule": "continuation (each alpha seeded from previous alpha's settled state)",
        "device": "cpu", "dtype": "float32", "norms_dtype": "float64",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "repo_rev": bb._git_rev(REPO_ROOT),
    }
    try:
        from importlib.metadata import version as _v
        m["transformer_lens_version"] = _v("transformer-lens")
    except Exception as e:
        m["transformer_lens_version"] = f"unknown ({e})"
    m.update(base)
    return m


def _write_partial_meta(base: dict, t0: float):
    m = build_meta(dict(base))
    m["partial"] = True
    m["total_seconds"] = round(time.time() - t0, 1)
    m["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    META_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.05,
                    help="alpha grid step (0.05 = 31 pts, 0.10 = 16 pts)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    from transformer_lens import HookedTransformer
    t = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    model.eval(); model.requires_grad_(False)
    print(f"[model] {MODEL_NAME} loaded in {time.time()-t:.1f}s", flush=True)
    assert model.cfg.n_layers - 1 == LAYER_END

    base_meta = run(model, args.step)
    meta = build_meta(base_meta)
    meta["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] wrote t1_2_hysteresis.jsonl; meta.json; "
          f"{meta.get('total_seconds',0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
