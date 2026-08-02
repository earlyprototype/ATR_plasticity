"""
T1.1 — the coexistence test (issue #45; ALIGNMENT_REVIEW.md T1.1; settles CLAIMS C-51,
decides C-26).

Under the working-point edit `W0 + ΔW` installed as a STATIC weight change, seed the ATR
loop from the model's ORIGINAL FROZEN `prolet` settled state and iterate to settling.

  STAYS at prolet  -> two attractors coexist -> created attractor (step 4) -> C-26 toward
                      supported; C-51 = "yes, stays".
  MOVES to comrade -> one displaced attractor -> boundary/displacement move (step 2) ->
                      C-26 refuted; C-51 = "no".

Reimplements nothing. Reuses:
  - atr_bridge.initial_state / make_atr_step        (the loop body, verbatim from the parent)
  - plasticity.OjaPlasticity                        (the hebb rule + ΔW accumulation + ceiling)
  - baseline_basins.readout_detail                  (the exact basin readout every experiment uses)
  - experiments.basin_bifurcation config + helpers  (the working-point cell, verbatim)

Working point (basin_bifurcation.py, EXP-001): site blocks.6.mlp, hebb,
eta = 7.065171428571429e-05, cadence 1, max_delta_frac = 0.05, seed 0, prompt A01_physics,
120-step episode. ‖ΔW‖_F/‖W0‖_F = 0.011239..., ‖W0‖_F = 164.85407309107723 (float64).

Run:  .venv/bin/python experiments/output_t1_1/t1_1_coexistence.py
"""

from __future__ import annotations

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

# This file lives at experiments/output_t1_1/ -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments"
# Mirror how the repo's own scripts resolve imports: repo root gives atr_bridge /
# plasticity; experiments/ gives baseline_basins and basin_bifurcation (no package
# __init__ exists, and basin_bifurcation does a top-level `import baseline_basins`).
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

# T1.1 loop length: the episode length (ALIGNMENT_REVIEW T1.1: "iterate 120 steps").
T11_N = 120
GATE_N = 30           # zero-step-size bit-identity gate: 30 steps is ample for a fixed point.
SETTLE_TAIL = 15      # settled = basin constant over the last SETTLE_TAIL steps.

OUT_DIR = REPO_ROOT / "experiments" / "output_t1_1"

basin_of = bif.basin_of
cosflat = bif.cosflat
relL2 = bif.relL2
token_rank_logit = bif.token_rank_logit


def _settle_step(basins: list[str], tail: int):
    """Settled word = the basin that holds for the final `tail` steps (constant).
    Returns (settled_word, first_step_it_locks_in, is_settled). Steps are 1-based."""
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


def run(model):
    records: list = []
    meta: dict = {}
    t0 = time.time()

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
    episode_ok = (cb["basin"] == "comrade" and cb["basin_token_id"] == 47998)
    comrade_id = cb["basin_token_id"]

    episode_clipped = bool(rep["clipped"])
    records.append({
        "kind": "reproduce_episode",
        "episode_ok": episode_ok,
        "n_applied": rep["n_applied"],
        "delta_frac": rep["delta_frac"],
        "delta_frac_f64": delta_frac,
        "clipped": episode_clipped,
        "clip_rate": 0.0 if not episode_clipped else "SEE_FINDING",
        "nonfinite": bool(rep["nonfinite"]),
        "init_norm": init_norm,
        "W0_norm": W0_norm,
        "dW_sigma1": dW_sigma1,
        "dW_fro": dW_fro,
        "closed_basin": cb["basin"],
        "closed_basin_token_id": cb["basin_token_id"],
        "closed_top5": cb["top5"],
        "anchor_delta_frac": ANCHOR["delta_frac"],
        "anchor_dW_sigma1": ANCHOR["dW_sigma1"],
    })
    print(f"[episode] ok={episode_ok} basin={cb['basin']!r}({comrade_id}) "
          f"delta_frac(f64)={delta_frac:.12f} sigma1={dW_sigma1:.9f} "
          f"clipped={episode_clipped} n_applied={rep['n_applied']}", flush=True)

    if not episode_ok:
        meta["blocker"] = (f"episode reproduced basin {cb['basin']!r} "
                           f"({cb['basin_token_id']}), expected comrade(47998) — "
                           f"cannot run T1.1")
        meta["total_seconds"] = round(time.time() - t0, 1)
        return records, meta

    # -- Phase 1b: frozen baseline -> the ORIGINAL FROZEN prolet settled state -
    plast.revert()                    # live weight back to W0 exactly
    restore_cos = cosflat(plast._site.weight, W0)
    restore_rl2 = relL2(plast._site.weight, W0)
    r = st0.tensor.clone()
    for _ in range(N_EPISODE):
        r = step(model, r)
    prolet_state = r.clone()          # <-- the state T1.1 seeds from
    fb = basin_of(model, prolet_state)
    prolet_id = fb["basin_token_id"]
    divine_id = bif._single_token_id(model, " Divine")
    frozen_prolet_ok = (fb["basin"] == "prolet" and prolet_id == 22758)
    records.append({
        "kind": "frozen_prolet_state",
        "basin": fb["basin"], "basin_token_id": prolet_id,
        "top5": fb["top5"], "margin": fb["margin"],
        "state_norm": float(prolet_state.norm()),
        "restore_cos_W_vs_W0": restore_cos, "restore_relL2_W_vs_W0": restore_rl2,
        "frozen_prolet_ok": frozen_prolet_ok,
    })
    print(f"[frozen] prolet state basin={fb['basin']!r}({prolet_id}) "
          f"ok={frozen_prolet_ok} ‖state‖={float(prolet_state.norm()):.5f} "
          f"restore_relL2={restore_rl2:.2e}", flush=True)

    # -- GATE: zero-step-size control must be bit-identical to the frozen loop --
    ref_states = []
    r = prolet_state.clone()
    for _ in range(GATE_N):
        r = step(model, r)
        ref_states.append(r.clone())

    plast._site.write(W0 + 0.0 * dW)  # same write path as the real edit, zero magnitude
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
    records.append({
        "kind": "gate_eta0",
        "n_steps": GATE_N,
        "bit_identical": gate_ok,
        "max_abs_diff": max_abs,
        "all_torch_equal": all_equal,
        "zero_edit_weight_bit_identical_to_W0": weight_bit_identical,
        "gate_seed_basin": basin_of(model, ctrl_states[-1])["basin"],
    })
    print(f"[GATE] eta=0 bit_identical={gate_ok} max_abs_diff={max_abs:.3e} "
          f"weight==W0:{weight_bit_identical}", flush=True)

    meta["gate_bit_identical"] = gate_ok
    if not gate_ok:
        meta["blocker"] = (f"eta=0 gate NOT bit-identical: max_abs_diff={max_abs:.3e}. "
                           f"STOP — do not interpret T1.1.")
        meta["total_seconds"] = round(time.time() - t0, 1)
        return records, meta

    # -- Seeded run helper -----------------------------------------------------
    def run_seeded(live_weight, seed_state, n, initial_norm, tag):
        plast._site.write(live_weight)
        stepx = make_atr_step(model, PROMPT, layer_start=LAYER_START,
                              layer_end=LAYER_END, initial_norm=initial_norm)
        r = seed_state.clone()
        prev1 = seed_state.clone()   # iter-0 reference for lag-1
        prev2 = None
        rows = []
        basins = []
        first_leave = None
        for i in range(1, n + 1):
            r = stepx(model, r)
            b = basin_of(model, r)
            lag1 = cosflat(r, prev1)
            lag2 = cosflat(r, prev2) if prev2 is not None else None
            cos_seed = cosflat(r, seed_state)
            rl2_seed = relL2(r, seed_state)
            basins.append(b["basin"])
            if b["basin"] != "prolet" and first_leave is None:
                first_leave = {"iter": i, "basin": b["basin"],
                               "token_id": b["basin_token_id"]}
            rows.append({
                "kind": f"t11_{tag}_step", "iter": i,
                "basin": b["basin"], "basin_token_id": b["basin_token_id"],
                "top5": b["top5"], "margin": b["margin"],
                "lag1_cos": lag1, "lag2_cos": lag2,
                "cos_to_seed": cos_seed, "relL2_to_seed": rl2_seed,
                "state_norm": float(r.norm()),
            })
            prev2 = prev1
            prev1 = r.clone()
        plast._site.write(W0)
        settled_word, settle_it, is_settled = _settle_step(basins, SETTLE_TAIL)
        return rows, {
            "tag": tag, "initial_norm": initial_norm, "n_steps": n,
            "settled_word": settled_word, "settled_token_id": rows[-1]["basin_token_id"],
            "settle_step": settle_it, "is_settled": is_settled,
            "final_basin": basins[-1], "first_leave_from_prolet": first_leave,
            "final_lag1_cos": rows[-1]["lag1_cos"], "final_lag2_cos": rows[-1]["lag2_cos"],
            "final_margin": rows[-1]["margin"], "final_top5": rows[-1]["top5"],
            "comrade_probe": token_rank_logit(model, r, comrade_id),
            "prolet_probe": token_rank_logit(model, r, prolet_id),
        }

    # -- T1.1 primary (init_norm shell, dual of D1) ----------------------------
    print("[T1.1 primary] W0+ΔW, seed=frozen prolet state, shell=init_norm ...", flush=True)
    prim_rows, prim_sum = run_seeded(W0 + dW, prolet_state, T11_N, init_norm, "primary")
    records.extend(prim_rows)
    prim_sum["kind"] = "t11_primary_summary"
    records.append(prim_sum)
    print(f"[T1.1 primary] settled_word={prim_sum['settled_word']!r} "
          f"settle_step={prim_sum['settle_step']} lag1={prim_sum['final_lag1_cos']:.6f} "
          f"lag2={prim_sum['final_lag2_cos']:.6f} first_leave={prim_sum['first_leave_from_prolet']}",
          flush=True)

    # -- T1.1 robustness (edited system's own energy shell, D2-style) ----------
    plast._site.write(W0 + dW)
    st_edit = initial_state(model, PROMPT, layer_end=LAYER_END)
    edit_norm = st_edit.initial_norm
    plast._site.write(W0)
    print(f"[T1.1 robust] fresh init_norm under W0+ΔW = {edit_norm:.6f} "
          f"(W0 init_norm = {init_norm:.6f})", flush=True)
    rob_rows, rob_sum = run_seeded(W0 + dW, prolet_state, T11_N, edit_norm, "robust")
    records.extend(rob_rows)
    rob_sum["kind"] = "t11_robust_summary"
    rob_sum["edit_init_norm"] = edit_norm
    records.append(rob_sum)
    print(f"[T1.1 robust]  settled_word={rob_sum['settled_word']!r} "
          f"settle_step={rob_sum['settle_step']} lag1={rob_sum['final_lag1_cos']:.6f}",
          flush=True)

    meta.update({
        "episode_ok": episode_ok,
        "frozen_prolet_ok": frozen_prolet_ok,
        "delta_frac_f64": delta_frac,
        "W0_norm_f64": W0_norm,
        "dW_sigma1": dW_sigma1,
        "comrade_token_id": comrade_id,
        "prolet_token_id": prolet_id,
        "primary_settled_word": prim_sum["settled_word"],
        "robust_settled_word": rob_sum["settled_word"],
        "verdict_agrees": prim_sum["settled_word"] == rob_sum["settled_word"],
        "total_seconds": round(time.time() - t0, 1),
    })
    return records, meta


def build_meta(base: dict) -> dict:
    m = {
        "experiment": "T1.1 coexistence test",
        "issue": 45,
        "settles": ["C-51"], "decides": ["C-26"],
        "model": MODEL_NAME,
        "site": SITE,
        "site_alias": "transformer.h.6.mlp.c_proj (W_out, (3072,768))",
        "prompt_id": PROMPT_ID, "prompt": PROMPT,
        "mode": MODE, "eta": ETA, "cadence": CADENCE,
        "max_delta_frac": MAX_DELTA_FRAC, "seed": SEED,
        "layer_start": LAYER_START, "layer_end": LAYER_END,
        "n_episode_for_dW": N_EPISODE,
        "t11_n_steps": T11_N, "gate_n_steps": GATE_N, "settle_tail": SETTLE_TAIL,
        "reference_W0_norm_F": 164.85407309107723,
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


def main():
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

    records, base_meta = run(model)
    meta = build_meta(base_meta)
    meta["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    bb.write_jsonl(OUT_DIR / "t1_1_trajectory.jsonl", records)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {len(records)} records -> t1_1_trajectory.jsonl; meta.json; "
          f"{meta.get('total_seconds',0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
