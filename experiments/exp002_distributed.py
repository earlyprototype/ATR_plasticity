"""EXP-002 -- distributed plasticity, then a fresh prompt (issue #24).

The full sequence, at all twelve MLP sites rather than the single site every
prior committed result used: collapse, work the well, stabilise, reprompt,
measure. Controls per issue #26; persistence question per issue #29.
Interpretation fixed in output_exp002/PREREGISTRATION.md, committed first.

Durable claims C-60 (driven prompt), C-61 (persistence), C-62 (steering vs
collapse), C-63 (what feedback contributed), all claimed on the registry first.

The offline arm is built from the tested single-site path: the frozen loop is
deterministic, so recording each site in its own pass gives the same data as one
combined pass, and offline replay has no feedback, so each site's replay is
independent of the others. No new capture code runs.

Units of work append to exp002.jsonl as they finish; --resume skips finished
units and recomputes only the deterministic prerequisites.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
from typing import Optional

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

from atr_bridge import initial_state, make_atr_step                  # noqa: E402
from multi_site import MultiSitePlasticity, SiteSpec                 # noqa: E402
from offline_control import (                                        # noqa: E402
    _frozen_trajectory,
    _make_site,
    installed_weight,
    record_frozen_activations,
    replay_offline,
)
sys.path.insert(0, _here)
import baseline_basins as bb                                         # noqa: E402

SITES_ALL = [f"blocks.{i}.mlp" for i in range(12)]
# The severable sub-configuration: every plastic site sits above the readout, so
# the loop cannot carry their drift into the next iterate. blocks.0-3 are
# excluded precisely because nothing can be read below blocks.0.
SITES_SEVERABLE = [f"blocks.{i}.mlp" for i in range(4, 12)]
LAYER_START = 0
LAYER_END = 11
LAYER_END_SEVERED = 3

ETA_STAR = 1.8e-2 * 164.854 / (120 * 350.0)       # 7.065171428571429e-05
# Amendment 1: probe low and shared, then scale each site to a common target.
PROBE_MULTS = [0.01, 0.002, 0.0005]
TARGET_DRIFT = 0.0112   # the drift the single-site working point produced (C-21)
MAX_SCALE = 2000.0      # ceiling on the per-site scale-up, so an inert site
                        # cannot demand an absurd step size
MAX_DELTA_FRAC = 0.05
SEED = 0
MODES = ["hebb", "anti_hebb"]

N_EPISODE = 120
# 120 iterations for the reprompt reads. Justified from the committed baseline,
# not chosen for convenience: basin_at_120 equals the 300-iteration settled basin
# on 124 of its 125 prompts (basin_at_100 manages only 113), so 120 is where the
# label is already the label, at 40% of the cost.
N_REPROMPT = 120
LAG_WINDOW = 12
MAX_LAG = 4
N_REPROMPTS_WANTED = 30
DRIVEN_PROMPT_ID = "A01_physics"

# Return test (issue #26 discriminator 1), criterion fixed in the prereg.
PERTURB_MAGS = [1e-7, 1e-5, 1e-3, 1e-2, 1e-1]
RETURN_ITERS = 40
RETURN_TOL = 1e-9

OUT_DIR = os.path.join(_here, "output_exp002")
JSONL = os.path.join(OUT_DIR, "exp002.jsonl")
META = os.path.join(OUT_DIR, "meta.json")
BASELINE = os.path.join(_here, "output_baseline", "basins.jsonl")


# ----------------------------------------------------------------- readouts

def basin_of(model, r: torch.Tensor) -> dict:
    d = bb.readout_detail(model, r[-1, :])
    return {
        "basin": d["top_token_strings"][0].strip(),
        "basin_token_id": d["top_token_ids"][0],
        "top5_tokens": d["top_token_strings"],
        "top_logit_margin": d["top_logit_margin"],
        "entropy": d["entropy"],
    }


def trajectory_stats(model, traj: list) -> dict:
    tail = traj[-(LAG_WINDOW + MAX_LAG):]
    scan = bb.lag_scan(torch.stack([t.mean(dim=0) for t in tail]), MAX_LAG)
    out = basin_of(model, traj[-1])
    out.update({
        "cos_lag1_mean": scan.get(1, {}).get("mean"),
        "cos_lag2_mean": scan.get(2, {}).get("mean"),
        "final_norm": float(traj[-1].double().norm()),
    })
    return out


def erank(m: torch.Tensor) -> float:
    """Participation-ratio effective rank, float64, as the step-size map uses."""
    sv = torch.linalg.svdvals(m.double())
    s2 = sv * sv
    tot = s2.sum().item()
    return (sv.sum().item() ** 2 / tot) if tot > 0 else float("nan")


@contextlib.contextmanager
def installed_weights(model, pairs):
    """Install many site weights at once; every one restored on the way out."""
    with contextlib.ExitStack() as stack:
        for site, w in pairs:
            stack.enter_context(installed_weight(model, site, w))
        yield


# ----------------------------------------------------------------- the arms

def build_driver(model, sites, mode: str, eta) -> MultiSitePlasticity:
    """`eta` is one value shared by every site, or one value per site.

    Per-site values are the amended design: layers respond so differently to the
    same step size (over 200x spread, measured) that one shared value cannot
    drive them all without either clipping the fastest or leaving the slowest
    inert.
    """
    etas = [eta] * len(sites) if isinstance(eta, (int, float)) else list(eta)
    if len(etas) != len(sites):
        raise ValueError(f"{len(etas)} step sizes for {len(sites)} sites")
    return MultiSitePlasticity(model, [
        SiteSpec(s, mode=mode, eta=e, cadence=1,
                 max_delta_frac=MAX_DELTA_FRAC, seed=SEED)
        for s, e in zip(sites, etas)
    ])


def run_closed_episode(model, r0, step, sites, mode: str, eta: float,
                       n_steps: int, keep_states: bool = True):
    """Plasticity live at every site; the changed weights drive the next step.

    Returns (report, per-site effective weights, trajectory). The matrices are
    restored on the model before returning: the drifted ones come back as values.
    """
    driver = build_driver(model, sites, mode, eta)
    states = []
    driver.install()
    try:
        r = r0.clone()
        for _ in range(n_steps):
            r = step(model, r)
            if keep_states:
                states.append(r.detach().clone())
            driver.apply()
        report = driver.report()
        weights = [(s, p._effective_W().detach().clone())
                   for s, p in zip(sites, driver)]
    finally:
        driver.revert()
        driver.remove()
    return report, weights, states


def offline_weights(model, r0, step, sites, mode, eta, n_steps: int):
    """The blocking control (issue #26), built from the tested single-site path.

    One deterministic frozen recording per site, then one independent offline
    replay per site. No feedback reaches any of them: every recording is of the
    untouched model, and no replay can see another replay's drift.
    """
    etas = [eta] * len(sites) if isinstance(eta, (int, float)) else list(eta)
    out = []
    diag = []
    for site, eta_i in zip(sites, etas):
        rec = record_frozen_activations(model, r0, step, site, n_steps,
                                        keep_states=False)
        arm = replay_offline(model, rec, eta=eta_i, mode=mode,
                             max_delta_frac=MAX_DELTA_FRAC, seed=SEED,
                             apply_every=1, y_source="recomputed")
        out.append((site, arm.weight.detach().clone()))
        diag.append({
            "site": site,
            "eta": eta_i,
            "delta_frac": arm.report()["delta_frac"],
            "clipped": arm.report()["clipped"],
            "nonfinite": arm.report()["nonfinite"],
            "n_updates": arm.config.n_updates,
        })
    return out, diag


def compare_weightsets(closed, offline, w0s) -> dict:
    """Closed against offline, stacked over sites, everything float64.

    The same quantity `compare_weights` computes at one site, extended over the
    disjoint block: the difference between the arms against the larger arm's own
    drift. Reported per site as well, since an aggregate can hide one site doing
    all the work.
    """
    num2 = da2 = db2 = w02 = 0.0
    per = []
    for (s, wc), (_, wo), (_, w0) in zip(closed, offline, w0s):
        a, b, z = wc.double(), wo.double(), w0.double()
        d = (a - b).norm().item()
        na, nb = (a - z).norm().item(), (b - z).norm().item()
        num2 += d * d
        da2 += na * na
        db2 += nb * nb
        w02 += z.norm().item() ** 2
        per.append({
            "site": s,
            "diff": d,
            "drift_closed": na,
            "drift_offline": nb,
            "diff_over_drift": d / max(na, nb) if max(na, nb) > 0 else float("nan"),
            "bit_identical": bool(torch.equal(wc, wo)),
        })
    num, da, db, w0n = num2 ** 0.5, da2 ** 0.5, db2 ** 0.5, w02 ** 0.5
    return {
        "diff_fro": num,
        "rel_fro_diff": num / w0n if w0n else float("nan"),
        "drift_closed_rel": da / w0n if w0n else float("nan"),
        "drift_offline_rel": db / w0n if w0n else float("nan"),
        "diff_over_drift": num / max(da, db) if max(da, db) > 0 else float("nan"),
        "bit_identical": all(p["bit_identical"] for p in per),
        "per_site": per,
    }


# ----------------------------------------------------------------- prompts

def pick_reprompts(n: int) -> list[dict]:
    """A stratified draw over the five frozen basins, driven prompt excluded.

    Proportional to the census (prolet 55, Divine 34, till 19, Anarch 16,
    solidarity 1), with every basin guaranteed at least one prompt so the
    survival count in the discriminator can actually see each of them, and the
    order inside a basin taken by prompt id so the draw is reproducible without
    an RNG.
    """
    rows = [json.loads(l) for l in open(BASELINE)]
    rows = [r for r in rows if r["prompt_id"] != DRIVEN_PROMPT_ID]
    by: dict[str, list] = {}
    for r in sorted(rows, key=lambda r: r["prompt_id"]):
        by.setdefault(r["basin"], []).append(r)
    total = sum(len(v) for v in by.values())
    quota = {b: max(1, round(n * len(v) / total)) for b, v in by.items()}
    out = []
    for b in sorted(by):
        out.extend(by[b][:quota[b]])
    return out


# ----------------------------------------------------------------- units

def unit_done(done: set, key: str) -> bool:
    return key in done


def append(rec: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    done: set = set()
    if args.resume and os.path.exists(JSONL):
        with open(JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["unit"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[resume] {len(done)} units already recorded", flush=True)

    from transformer_lens import HookedTransformer
    import importlib.metadata
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()

    rows = {r["prompt_id"]: r for r in (json.loads(l) for l in open(BASELINE))}
    driven = rows[DRIVEN_PROMPT_ID]
    prompt = driven["prompt"]
    s0 = initial_state(model, prompt, layer_end=LAYER_END)
    step = make_atr_step(model, prompt, layer_start=LAYER_START,
                         layer_end=LAYER_END, initial_norm=s0.initial_norm)

    reprompts = pick_reprompts(N_REPROMPTS_WANTED)
    print(f"[setup] {len(reprompts)} fresh prompts across "
          f"{len(set(r['basin'] for r in reprompts))} basins", flush=True)

    if not (args.resume and os.path.exists(META)):
        with open(META, "w") as f:
            json.dump({
                "experiment": "EXP-002 distributed plasticity",
                "issues": [24, 26, 29],
                "claims": ["C-60", "C-61", "C-62", "C-63"],
                "sites": SITES_ALL,
                "sites_severable": SITES_SEVERABLE,
                "modes": MODES,
                "eta_star": ETA_STAR,
                "max_delta_frac": MAX_DELTA_FRAC,
                "seed": SEED,
                "n_episode": N_EPISODE,
                "n_reprompt": N_REPROMPT,
                "driven_prompt_id": DRIVEN_PROMPT_ID,
                "reprompt_ids": [r["prompt_id"] for r in reprompts],
                "model": "gpt2-small", "device": "cpu", "dtype": "float32",
                "norms_dtype": "float64",
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "transformer_lens_version":
                    importlib.metadata.version("transformer-lens"),
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, f, indent=1)

    # ---------------------------------------------------------- step 1: collapse
    t0 = time.time()
    frozen_traj = _frozen_trajectory(model, s0.tensor, step, N_EPISODE)
    frozen_stats = trajectory_stats(model, frozen_traj)
    if not unit_done(done, "collapse"):
        append({"unit": "collapse", "kind": "step1_collapse",
                "prompt_id": DRIVEN_PROMPT_ID, "n_steps": N_EPISODE,
                "readout": frozen_stats,
                "baseline_basin": driven["basin"],
                "matches_baseline": frozen_stats["basin"] == driven["basin"],
                "seconds": round(time.time() - t0, 1)})
        print(f"[collapse] settles {frozen_stats['basin']!r} "
              f"(baseline says {driven['basin']!r})", flush=True)

    # ---------------------------------------------------------- gate: eta = 0
    if not unit_done(done, "gate:eta0"):
        t0 = time.time()
        rep0, w0set, traj0 = run_closed_episode(
            model, s0.tensor, step, SITES_ALL, "hebb", 0.0, 30)
        base = _frozen_trajectory(model, s0.tensor, step, 30)
        same = all(torch.equal(a, b) for a, b in zip(traj0, base))
        maxdiff = max(float((a.double() - b.double()).abs().max())
                      for a, b in zip(traj0, base))
        append({"unit": "gate:eta0", "kind": "gate",
                "bit_identical_trajectory": bool(same),
                "max_abs_diff": maxdiff,
                "delta_frac": rep0["delta_frac"],
                "clipped": rep0["clipped"], "nonfinite": rep0["nonfinite"],
                "seconds": round(time.time() - t0, 1)})
        print(f"[gate eta0] bit-identical={same} max_abs_diff={maxdiff}", flush=True)
        if not same:
            raise RuntimeError("eta=0 gate failed: plasticity at eta 0 changed the loop")

    # ---------------------------------------------------------- gate: revert
    if not unit_done(done, "gate:revert"):
        t0 = time.time()
        driver = build_driver(model, SITES_ALL, "hebb", ETA_STAR)
        before = [p._site.weight.detach().clone() for p in driver]
        driver.install()
        r = s0.tensor.clone()
        for _ in range(10):
            r = step(model, r)
            driver.apply()
        moved = any(not torch.equal(p._site.weight, b) for p, b in zip(driver, before))
        driver.revert()
        restored = all(torch.equal(p._site.weight, b) for p, b in zip(driver, before))
        driver.remove()
        append({"unit": "gate:revert", "kind": "gate",
                "weights_moved_while_live": bool(moved),
                "revert_bit_exact_all_sites": bool(restored),
                "n_sites": len(SITES_ALL),
                "seconds": round(time.time() - t0, 1)})
        print(f"[gate revert] moved={moved} restored_bit_exact={restored}", flush=True)
        if not restored:
            raise RuntimeError("revert gate failed: a matrix was not restored bit-exactly")

    # ---------------------------------------------------------- calibration
    # Amendment 1: per-site step sizes anchored to a common target drift. The
    # probe measures how far each site travels at a low shared step size over a
    # full episode; each site's step size is then scaled to reach TARGET_DRIFT.
    # Calibration only -- declared non-evidence in the pre-registration.
    def calibrate(mode: str) -> tuple[list, dict]:
        key = f"calib:{mode}"
        for line in open(JSONL):
            d = json.loads(line)
            if d.get("unit") == key:
                return d["etas"], d
        probe_eta = None
        probe_per = None
        for mult in PROBE_MULTS:
            rep, _, _ = run_closed_episode(model, s0.tensor, step, SITES_ALL,
                                           mode, mult * ETA_STAR, N_EPISODE,
                                           keep_states=False)
            print(f"[calib {mode}] probe {mult}x: clip={rep['clipped']} "
                  f"agg={rep['delta_frac']:.5f}", flush=True)
            if not rep["clipped"] and not rep["nonfinite"]:
                probe_eta = mult * ETA_STAR
                probe_mult = mult
                probe_per = [{"site": r["site"], "delta_frac": r["delta_frac"]}
                             for r in rep["per_site"]]
                break
        if probe_eta is None:
            raise RuntimeError(f"[{mode}] every probe step size clipped a site")

        # Scale each site to the target. Drift is close to linear in the step
        # size at fixed step count, so this is one Newton step from the probe,
        # and the achieved drift is measured and recorded rather than assumed.
        etas = []
        for rec in probe_per:
            d = rec["delta_frac"]
            if not (d > 0):
                etas.append(probe_eta * MAX_SCALE)
                continue
            scale = min(TARGET_DRIFT / d, MAX_SCALE)
            etas.append(probe_eta * scale)

        rep2, _, _ = run_closed_episode(model, s0.tensor, step, SITES_ALL, mode,
                                        etas, N_EPISODE, keep_states=False)
        achieved = [{"site": r["site"], "eta": e, "delta_frac": r["delta_frac"],
                     "clipped": r["clipped"], "nonfinite": r["nonfinite"]}
                    for r, e in zip(rep2["per_site"], etas)]
        rec = {"unit": key, "kind": "calibration", "mode": mode,
               "probe_eta": probe_eta, "probe_mult": probe_mult,
               "probe_per_site": probe_per,
               "target_drift": TARGET_DRIFT, "max_scale": MAX_SCALE,
               "etas": etas, "achieved": achieved,
               "any_clipped": bool(rep2["clipped"]),
               "nonfinite": bool(rep2["nonfinite"]),
               "aggregate_delta_frac": rep2["delta_frac"],
               "note": "calibration only; declared non-evidence in the prereg"}
        append(rec)
        print(f"[calib {mode}] anchored: achieved drift "
              f"{[round(a['delta_frac'], 4) for a in achieved]} "
              f"clip={rep2['clipped']}", flush=True)
        return etas, rec

    ETAS = {}
    for mode in MODES:
        ETAS[mode], _ = calibrate(mode)
    # The severed gate drives only blocks.4..11, so it takes that slice of the
    # calibrated step sizes -- the same value each of those sites gets in the
    # main run, so the floor is measured on the configuration that is used.
    severed_etas = [ETAS["hebb"][SITES_ALL.index(s_)] for s_ in SITES_SEVERABLE]

    # ---------------------------------------------------------- gate: severed
    if not unit_done(done, "gate:severed"):
        t0 = time.time()
        s0s = initial_state(model, prompt, layer_end=LAYER_END_SEVERED)
        steps = make_atr_step(model, prompt, layer_start=LAYER_START,
                              layer_end=LAYER_END_SEVERED,
                              initial_norm=s0s.initial_norm)
        rep, wc, _ = run_closed_episode(model, s0s.tensor, steps,
                                        SITES_SEVERABLE, "hebb", severed_etas,
                                        N_EPISODE, keep_states=False)
        wo, _ = offline_weights(model, s0s.tensor, steps, SITES_SEVERABLE,
                                "hebb", severed_etas, N_EPISODE)
        # W0 read straight off the untouched model, never reconstructed
        w0s = [(s, _make_site(model, s).weight.detach().clone())
               for s in SITES_SEVERABLE]
        cmp = compare_weightsets(wc, wo, w0s)
        append({"unit": "gate:severed", "kind": "gate", "sites": SITES_SEVERABLE,
                "read_at": f"blocks.{LAYER_END_SEVERED}.hook_resid_post",
                "etas": severed_etas, "n_steps": N_EPISODE,
                "clipped": rep["clipped"], "nonfinite": rep["nonfinite"],
                "comparison": cmp,
                "floor_is_exactly_zero": cmp["diff_fro"] == 0.0,
                "bit_identical": cmp["bit_identical"],
                "seconds": round(time.time() - t0, 1)})
        print(f"[gate severed] floor diff_fro={cmp['diff_fro']!r} "
              f"bit_identical={cmp['bit_identical']}", flush=True)

    # ---------------------------------------------------------- the two arms
    W0S = [(s, _make_site(model, s).weight.detach().clone()) for s in SITES_ALL]

    # reprompt under the untouched model, once, shared by both arms
    for row in reprompts:
        key = f"reprompt:original:{row['prompt_id']}"
        if unit_done(done, key):
            continue
        t0 = time.time()
        si = initial_state(model, row["prompt"], layer_end=LAYER_END)
        st = make_atr_step(model, row["prompt"], layer_start=LAYER_START,
                           layer_end=LAYER_END, initial_norm=si.initial_norm)
        tr = _frozen_trajectory(model, si.tensor, st, N_REPROMPT)
        rd = trajectory_stats(model, tr)
        append({"unit": key, "kind": "reprompt", "condition": "original",
                "prompt_id": row["prompt_id"], "baseline_basin": row["basin"],
                "readout": rd, "seconds": round(time.time() - t0, 1)})
        print(f"[reprompt original] {row['prompt_id']}: {rd['basin']!r}", flush=True)

    for mode in MODES:
        print(f"[arm] {mode} starting", flush=True)
        t0 = time.time()
        rep, wc, traj = run_closed_episode(model, s0.tensor, step, SITES_ALL,
                                           mode, ETAS[mode], N_EPISODE)
        closed_stats = trajectory_stats(model, traj)
        wo, odiag = offline_weights(model, s0.tensor, step, SITES_ALL, mode,
                                    ETAS[mode], N_EPISODE)
        cmp = compare_weightsets(wc, wo, W0S)

        with installed_weights(model, wo):
            otraj = _frozen_trajectory(model, s0.tensor, step, N_REPROMPT)
            offline_stats = trajectory_stats(model, otraj)
        with installed_weights(model, wc):
            ctraj = _frozen_trajectory(model, s0.tensor, step, N_REPROMPT)
            closed_rerun_stats = trajectory_stats(model, ctraj)

        if not unit_done(done, f"episode:{mode}"):
            append({
                "unit": f"episode:{mode}", "kind": "step2_episode", "mode": mode,
                "etas": ETAS[mode], "target_drift": TARGET_DRIFT,
                "n_steps": N_EPISODE,
                "sites": SITES_ALL,
                "clipped": rep["clipped"], "nonfinite": rep["nonfinite"],
                "aggregate_delta_frac": rep["delta_frac"],
                "per_site": [{"site": r["site"], "delta_frac": r["delta_frac"],
                              "clipped": r["clipped"], "nonfinite": r["nonfinite"]}
                             for r in rep["per_site"]],
                "offline_per_site": odiag,
                "comparison_closed_vs_offline": cmp,
                "readout_live_closed": closed_stats,
                "readout_frozen_under_closed": closed_rerun_stats,
                "readout_frozen_under_offline": offline_stats,
                "readout_frozen_original": frozen_stats,
                "erank_change": [
                    {"site": s, "erank_w0": erank(w0), "erank_closed": erank(w)}
                    for (s, w), (_, w0) in zip(wc, W0S)],
                "seconds": round(time.time() - t0, 1),
            })
            print(f"[episode {mode}] agg drift {rep['delta_frac']:.5f} "
                  f"clip={rep['clipped']} | driven settles "
                  f"closed={closed_rerun_stats['basin']!r} "
                  f"offline={offline_stats['basin']!r} "
                  f"frozen={frozen_stats['basin']!r} | "
                  f"feedback share {cmp['diff_over_drift']:.4f}", flush=True)

        # ------------------------------------------------ return test (C-62.1)
        if not unit_done(done, f"return:{mode}"):
            t1 = time.time()
            settled = ctraj[-1]
            results = []
            with installed_weights(model, wc):
                for mag in PERTURB_MAGS:
                    g = torch.randn(settled.shape, generator=torch.Generator().manual_seed(SEED))
                    p = settled + g * (mag * settled.norm() / g.norm())
                    r = p.clone()
                    best, ret_at = 1.0, None
                    for i in range(RETURN_ITERS):
                        r = step(model, r)
                        a, b = r.double().flatten(), settled.double().flatten()
                        c = 1.0 - float((a @ b) / (a.norm() * b.norm()))
                        best = min(best, c)
                        if c < RETURN_TOL and ret_at is None:
                            ret_at = i + 1
                            break
                    results.append({"magnitude": mag, "returned": ret_at is not None,
                                    "iterations": ret_at, "best_one_minus_cos": best})
            append({"unit": f"return:{mode}", "kind": "return_test", "mode": mode,
                    "tolerance": RETURN_TOL, "max_iters": RETURN_ITERS,
                    "results": results,
                    "n_returned": sum(1 for r in results if r["returned"]),
                    "seconds": round(time.time() - t1, 1)})
            print(f"[return {mode}] "
                  f"{sum(1 for r in results if r['returned'])}/{len(results)} returned",
                  flush=True)

        # ------------------------------------------------ reprompts (C-61)
        for cond, wset in (("closed", wc), ("offline", wo)):
            for row in reprompts:
                key = f"reprompt:{cond}_{mode}:{row['prompt_id']}"
                if unit_done(done, key):
                    continue
                t1 = time.time()
                si = initial_state(model, row["prompt"], layer_end=LAYER_END)
                st = make_atr_step(model, row["prompt"], layer_start=LAYER_START,
                                   layer_end=LAYER_END,
                                   initial_norm=si.initial_norm)
                with installed_weights(model, wset):
                    tr = _frozen_trajectory(model, si.tensor, st, N_REPROMPT)
                    rd = trajectory_stats(model, tr)
                append({"unit": key, "kind": "reprompt",
                        "condition": f"{cond}_{mode}", "mode": mode,
                        "prompt_id": row["prompt_id"],
                        "baseline_basin": row["basin"],
                        "readout": rd, "seconds": round(time.time() - t1, 1)})
                print(f"[reprompt {cond}_{mode}] {row['prompt_id']}: "
                      f"{rd['basin']!r}", flush=True)

        print(f"[arm] {mode} complete in {round(time.time()-t0)}s", flush=True)

    print("[exp002] complete", flush=True)


if __name__ == "__main__":
    main()
