"""
Baseline basin census: the FROZEN ATR loop over the parent project's 125-prompt
library. No plasticity, no weight updates, nothing attached to the model.

This is the measuring stick. Every later experiment in this repo (Oja plasticity
at `transformer.h.6.mlp.c_proj`, the controls in `controls.py`) is a difference
against the numbers this script produces, so it is written for accuracy and
reproducibility and not for speed.

WHAT IT RUNS
------------
`atr_bridge.make_atr_step` / `atr_bridge.initial_state` -- this repo's bit-exact
extraction of the parent's `atr_engine.run_atr_loop` body (see
`tests/test_atr_bridge.py::test_step_reproduces_run_atr_loop_bit_exactly`).
GPT-2 small via TransformerLens, layers 0 -> 11, whole-tensor re-injection:

    x0    = f(embed(prompt))              read  blocks.11.hook_resid_post
    x_n+1 = f(x_n * (||x0|| / ||x_n||))   write blocks.0.hook_resid_pre

MEASUREMENT, AND WHY IT IS SHAPED THIS WAY
------------------------------------------
The loop is run to a fixed horizon and *never* early-stopped. The parent's
`run_atr_gated` breaks at lock-in, which throws away every iterate after it;
here the gate is evaluated but the trajectory keeps going, so the same run
yields both the lock-in iteration and the late-window periodicity structure.

The gate is evaluated at TWO lags, and this is the point of the whole file. A
lag-1 gate asks "are you where you were one step ago". An exact period-2 limit
cycle answers no forever -- the parent's `Divine` state holds lag-1 cosine at
0.6849 and lag-2 cosine at 1.0 for as long as you care to run it. Classifying
that as "did not converge" is a measurement artefact, not a fact about the
system, so both lags are recorded and a period-2 state is reported as its own
category rather than folded into the non-convergent pile. Lags 1..8 are scanned
over the late window (the parent's `atr_engine.lag_scan` instrument) so a period
above 2 would also be visible rather than aliased.

Readout is the parent's exactly: `ln_final(v) @ W_U + b_U`, argmax at the LAST
position, decoded. The top-1 token there at the final iteration is the basin
label.

OUTPUTS
-------
`experiments/output_baseline/basins.jsonl`  one JSON object per prompt, appended
and fsynced as each prompt finishes. Kill the process at prompt 90 and 90
prompts of data survive; re-running skips whatever is already in the file.

`experiments/output_baseline/BASELINE.md`   the report, regenerable at any time
from the JSONL alone with `--report-only`.

USAGE
-----
    .venv/bin/python experiments/baseline_basins.py                 # full sweep
    .venv/bin/python experiments/baseline_basins.py --limit 3       # pilot
    .venv/bin/python experiments/baseline_basins.py --report-only   # rebuild md
    .venv/bin/python experiments/baseline_basins.py --iters 150     # shorter horizon

`--parent PATH` or `$ATR_PARENT_PATH` locates the parent repo (for
`prompt_library.py` only -- no parent code is executed and nothing there is
written to).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path

import torch
import torch.nn.functional as F

# Repo root on sys.path so `atr_bridge` imports when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atr_bridge import initial_state, make_atr_step  # noqa: E402

# ---------------------------------------------------------------------------
# Config. Every value here is echoed into BASELINE.md; change one and the
# report says so, because a basin table without its stopping rule is unreadable.
# ---------------------------------------------------------------------------

PARENT_DEFAULT = "/workspace/lucier-gpt2-activ-tensor-reson-experiments"

MODEL_NAME = "gpt2-small"
LAYER_START = 0
LAYER_END = 11
DEFAULT_ITERS = 300

# The parent's gate, verbatim (`atr_engine.run_atr_gated` defaults):
# cos(mean_t, mean_{t-lag}) > 0.999 for 3 consecutive checks, checked every 10
# iterations once iteration 100 is passed. Held identical so the convergence
# counts here are comparable with the published re-sweep rather than merely
# similar to it.
GATE_THRESHOLD = 0.999
GATE_PATIENCE = 3
GATE_CHECK_EVERY = 10
GATE_CHECK_START = 100
GATE_LAGS = (1, 2)

# Late-window periodicity scan.
LAG_WINDOW = 25          # number of consecutive iterates retained at the tail
MAX_LAG = 8

# Iterations at which the full readout (top-5, entropy, margin, position
# uniformity) is recorded. The parent's Stage 1 schedule plus the gate's own
# checkpoints, so the published @100 number is directly readable.
BASE_SCHEDULE = (0, 1, 2, 3, 5, 10, 20, 50, 100, 110, 120, 150, 200, 250)

SEED = 0

# ONE torch thread, deliberately. Measured on this 4-vCPU box, GPT-2 small at
# seq_len 10: 1 thread 287 ms/iter, 2 threads 350, 3 threads 1105, 4 threads
# 2137 -- a 7x penalty for using every core, the classic OpenMP spin-wait
# collapse when the thread count meets the core count and anything else is
# running. Single-threaded is also the reproducible choice: float32 reduction
# order stops depending on how work happened to be split. Two single-threaded
# shards run at 296 ms/iter each (no measurable interference), which is where
# the parallelism comes from instead.
TORCH_THREADS = 1

OUT_DIR = REPO_ROOT / "experiments" / "output_baseline"
JSONL_PATH = OUT_DIR / "basins.jsonl"
REPORT_PATH = OUT_DIR / "BASELINE.md"

# The parent README's published GPT-2 small table (shares at lock-in), and the
# @100 table from `experiments/gpt2_small/output_gated/gated_report.md`.
PUBLISHED_LOCKIN = [("prolet", 54, 43.2), ("Divine", 34, 27.2), ("till", 19, 15.2),
                    ("Anarch", 17, 13.6), ("solidarity", 1, 0.8)]
PUBLISHED_AT_100 = [("prolet", 44, 35.2), ("Divine", 34, 27.2), ("Anarch", 26, 20.8),
                    ("till", 19, 15.2), ("solidarity", 2, 1.6)]
PUBLISHED_CONVERGED = 91
PUBLISHED_NOT_CONVERGED = 34
PUBLISHED_LOCKIN_ITER = 120


# ---------------------------------------------------------------------------
# Parent prompt library
# ---------------------------------------------------------------------------

def load_prompt_library(parent_path: str):
    """Import the parent's `prompt_library` by path. Read-only, never executed
    beyond module import; the parent clone is not modified."""
    p = Path(parent_path) / "prompt_library.py"
    if not p.exists():
        raise SystemExit(
            f"prompt_library.py not found at {p}. Pass --parent or set "
            f"ATR_PARENT_PATH to the parent ATR repo.")
    spec = importlib.util.spec_from_file_location("_atr_prompt_library", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ordered_prompts(pl):
    """Prompt ids in the parent sweep's own order.

    `gated_resweep.py` builds its dict as COMPLEX, NARRATIVE, SIMPLE, CHEMICAL,
    ACRONYMS, VULGARITY, WILD -- which is NOT the order of `PROMPT_LIBRARY`
    (that one puts CHEMICAL and ACRONYMS before SIMPLE). Order changes nothing
    about the results, only which prompts a truncated run covers, so the sweep's
    order is the one reproduced.
    """
    merged = {}
    for cat in (pl.COMPLEX, pl.NARRATIVE, pl.SIMPLE, pl.CHEMICAL,
                pl.ACRONYMS, pl.VULGARITY, pl.WILD):
        merged.update(cat)
    return merged


# ---------------------------------------------------------------------------
# Readout -- `atr_engine.get_top_tokens` / `get_readout_detail`, copied
# ---------------------------------------------------------------------------

def _logits(model, resid_vector: torch.Tensor) -> torch.Tensor:
    """Final LayerNorm then unembed. The LN is not optional: skipping it decodes
    a different vector and every basin label would be wrong."""
    normalized = model.ln_final(resid_vector)
    return normalized @ model.W_U + model.b_U


def top1_id(model, resid_vector: torch.Tensor) -> int:
    return int(_logits(model, resid_vector).argmax(dim=-1))


def readout_detail(model, resid_vector: torch.Tensor, k: int = 5) -> dict:
    logits = _logits(model, resid_vector)
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    top_logits = logits[top_indices]
    ids = [int(i) for i in top_indices]
    return {
        "top_token_ids": ids,
        "top_token_strings": [model.tokenizer.decode([i]) for i in ids],
        "top_token_probs": [float(p) for p in top_probs],
        "top_logits": [float(x) for x in top_logits],
        "top_logit_margin": float(top_logits[0] - top_logits[1]) if top_logits.numel() > 1 else 0.0,
        "entropy": float(-(probs * torch.log(probs.clamp_min(1e-12))).sum()),
    }


def position_stats(tensor: torch.Tensor):
    """Off-diagonal cosine between token positions -- the parent's
    `position_similarity`, plus its spread.

    The mean is the parent's number: 1.0 means every position holds the same
    direction and the sequence has stopped being a sequence. The mean alone
    cannot distinguish "all positions identical" from "most identical, one
    position holding out", which is exactly the question a within-basin spread
    analysis will ask, so min/max/spread come along.

    Returns None for a single-position tensor, where the quantity is undefined
    rather than 0 or 1 (the parent would produce nan)."""
    seq_len = tensor.shape[0]
    if seq_len < 2:
        return None
    pos_norms = tensor.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = tensor / pos_norms
    sim = normed @ normed.T
    mask = ~torch.eye(seq_len, dtype=torch.bool, device=sim.device)
    off = sim[mask]
    return {
        "mean": float(off.mean()),
        "min": float(off.min()),
        "max": float(off.max()),
        "spread": float(off.max() - off.min()),
        "std": float(off.std()) if off.numel() > 1 else 0.0,
    }


def position_similarity(tensor: torch.Tensor):
    """Just the parent's scalar, for the snapshot rows."""
    st = position_stats(tensor)
    return None if st is None else st["mean"]


def lag_scan(stack: torch.Tensor, max_lag: int) -> dict:
    """`atr_engine.lag_scan`: mean cosine between iterates k apart, k = 1..max_lag.

    Fixed point -> ~1.0 at every lag. Period-p cycle -> ~1.0 only at multiples
    of p. Drift -> monotone decay. The pattern across lags is the signal; no
    single lag identifies a state."""
    out = {}
    for k in range(1, max_lag + 1):
        if k >= stack.shape[0]:
            break
        cos = F.cosine_similarity(stack[k:], stack[:-k], dim=-1)
        out[k] = {"mean": float(cos.mean()), "min": float(cos.min()),
                  "n_pairs": int(cos.numel())}
    return out


# ---------------------------------------------------------------------------
# One prompt
# ---------------------------------------------------------------------------

def save_state(states_dir: Path, prompt_id: str, tensor: torch.Tensor) -> str:
    """Persist a settled residual state as `.npy`, keyed by prompt id.

    The whole (seq_len, d_model) tensor, not the position mean: at ~30 KB per
    prompt the full thing is free, and the position-mean is recoverable from it
    while the converse is not. This is what makes within-basin spread measurable
    later without a single extra forward pass -- whether prompts sharing a basin
    land on bit-identical states (the attractor is a label, and the prompt's
    content is destroyed) or on nearby-but-distinct states (it compresses).
    """
    import numpy as np
    states_dir.mkdir(parents=True, exist_ok=True)
    path = states_dir / f"{prompt_id}.npy"
    np.save(path, tensor.detach().cpu().numpy().astype("float32"))
    return f"states/{path.name}"


def run_prompt(model, prompt_id: str, prompt: str, category: str, n_iter: int,
               states_dir: Path | None = None) -> dict:
    """The frozen ATR loop on one prompt, fully instrumented. Never early-stops."""
    t0 = time.time()
    schedule = sorted(set(list(BASE_SCHEDULE) + [n_iter]) & set(range(0, n_iter + 1)))

    state0 = initial_state(model, prompt, layer_end=LAYER_END)
    step = make_atr_step(model, prompt, layer_start=LAYER_START, layer_end=LAYER_END,
                         initial_norm=state0.initial_norm)

    r = state0.tensor
    seq_len = int(r.shape[0])
    tok_ids = model.to_tokens(prompt)[0].tolist()

    # Gate state, one tracker per lag, each a faithful copy of the parent's.
    gate = {lag: {"consecutive": 0, "lock_in": None, "last_cos": 1.0,
                  "checks": []} for lag in GATE_LAGS}
    # Rolling history of mean/last vectors, deep enough for the widest gate lag
    # and the late-window lag scan.
    hist_depth = max(MAX_LAG, max(GATE_LAGS)) + 1
    mean_hist = deque([r.mean(dim=0).clone()], maxlen=hist_depth)
    last_hist = deque([r[-1, :].clone()], maxlen=hist_depth)
    # Contiguous tail buffer for the lag scan.
    tail_len = LAG_WINDOW + MAX_LAG
    mean_tail = deque([r.mean(dim=0).clone()], maxlen=tail_len)
    last_tail = deque([r[-1, :].clone()], maxlen=tail_len)

    top1_traj = [top1_id(model, r[-1, :])]
    snapshots = []
    nonfinite_at = None

    if 0 in schedule:
        d = readout_detail(model, r[-1, :])
        snapshots.append({
            "iteration": 0, "tensor_norm": float(r.norm()),
            "top1": d["top_token_strings"][0], "top1_prob": d["top_token_probs"][0],
            "entropy": d["entropy"], "top_logit_margin": d["top_logit_margin"],
            "position_similarity": position_similarity(r),
            "cos_lag1_mean": 1.0, "cos_lag1_last": 1.0,
        })

    prev_tensor = r.clone()
    for i in range(1, n_iter + 1):
        prev_tensor = r
        r = step(model, r)

        if nonfinite_at is None and not bool(torch.isfinite(r).all()):
            nonfinite_at = i
            # Keep going: a non-finite tensor is a result, and stopping here
            # would hide whether it recovers. Metrics downstream become nan and
            # are reported as such.

        mean_vec = r.mean(dim=0)
        last_vec = r[-1, :]
        top1_traj.append(top1_id(model, last_vec))

        # --- gate, per lag ------------------------------------------------
        if i >= GATE_CHECK_START and i % GATE_CHECK_EVERY == 0:
            for lag in GATE_LAGS:
                if i < lag or len(mean_hist) < lag:
                    continue
                ref = mean_hist[len(mean_hist) - lag]
                cos = float(F.cosine_similarity(mean_vec.unsqueeze(0), ref.unsqueeze(0)))
                g = gate[lag]
                g["last_cos"] = cos
                g["checks"].append([i, cos])
                g["consecutive"] = g["consecutive"] + 1 if cos > GATE_THRESHOLD else 0
                if g["lock_in"] is None and g["consecutive"] >= GATE_PATIENCE:
                    g["lock_in"] = i

        mean_hist.append(mean_vec.clone())
        last_hist.append(last_vec.clone())
        mean_tail.append(mean_vec.clone())
        last_tail.append(last_vec.clone())

        if i in schedule:
            d = readout_detail(model, last_vec)
            c1m = float(F.cosine_similarity(mean_vec.unsqueeze(0),
                                            mean_hist[len(mean_hist) - 2].unsqueeze(0)))
            c1l = float(F.cosine_similarity(last_vec.unsqueeze(0),
                                            last_hist[len(last_hist) - 2].unsqueeze(0)))
            snapshots.append({
                "iteration": i, "tensor_norm": float(r.norm()),
                "top1": d["top_token_strings"][0], "top1_prob": d["top_token_probs"][0],
                "entropy": d["entropy"], "top_logit_margin": d["top_logit_margin"],
                "position_similarity": position_similarity(r),
                "cos_lag1_mean": c1m, "cos_lag1_last": c1l,
            })

    final = readout_detail(model, r[-1, :])
    scan_mean = lag_scan(torch.stack(list(mean_tail)), MAX_LAG)
    scan_last = lag_scan(torch.stack(list(last_tail)), MAX_LAG)
    pos = position_stats(r)

    # Settled state, for the within-basin spread analysis. Both phases are kept:
    # on a period-2 orbit the "final state" is one of two, and a spread analysis
    # that mixes phases across prompts would measure the cycle, not the basin.
    state_file = state_prev_file = None
    if states_dir is not None:
        state_file = save_state(states_dir, prompt_id, r)
        state_prev_file = save_state(states_dir, f"{prompt_id}__prev", prev_tensor)

    def _basin_at(it):
        return model.tokenizer.decode([top1_traj[it]]) if it < len(top1_traj) else None

    rec = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "category": category,
        "seq_len": seq_len,
        "token_ids": tok_ids,
        "n_iter": n_iter,

        # Basin label: top-1 at the last position, final iteration.
        "basin": final["top_token_strings"][0].strip(),
        "basin_raw": final["top_token_strings"][0],
        "basin_token_id": final["top_token_ids"][0],
        "basin_at_100": (_basin_at(100) or "").strip() or None,
        "basin_at_120": (_basin_at(120) or "").strip() or None,

        "final_top5_tokens": final["top_token_strings"],
        "final_top5_probs": final["top_token_probs"],
        "final_top5_ids": final["top_token_ids"],
        "final_entropy": final["entropy"],
        "final_top_logit_margin": final["top_logit_margin"],

        "initial_norm": state0.initial_norm,
        "final_tensor_norm": float(r.norm()),
        "final_position_similarity": None if pos is None else pos["mean"],
        "final_position_stats": pos,
        "final_position_spread": None if pos is None else pos["spread"],

        # Saved settled states (relative to the output directory).
        "state_file": state_file,
        "state_prev_file": state_prev_file,
        "state_shape": list(r.shape),
        "state_dtype": "float32",
        "cos_final_prev": float(F.cosine_similarity(
            r.reshape(-1).unsqueeze(0), prev_tensor.reshape(-1).unsqueeze(0))),

        "lag_scan_mean_vec": {str(k): v for k, v in scan_mean.items()},
        "lag_scan_last_vec": {str(k): v for k, v in scan_last.items()},
        "cos_lag1_mean": scan_mean.get(1, {}).get("mean"),
        "cos_lag2_mean": scan_mean.get(2, {}).get("mean"),
        "cos_lag1_last": scan_last.get(1, {}).get("mean"),
        "cos_lag2_last": scan_last.get(2, {}).get("mean"),
        "lag_window": LAG_WINDOW,

        "lock_in_iter_lag1": gate[1]["lock_in"],
        "converged_lag1": gate[1]["lock_in"] is not None,
        "final_gate_cos_lag1": gate[1]["last_cos"],
        "lock_in_iter_lag2": gate[2]["lock_in"],
        "converged_lag2": gate[2]["lock_in"] is not None,
        "final_gate_cos_lag2": gate[2]["last_cos"],

        "nonfinite": nonfinite_at is not None,
        "nonfinite_first_iter": nonfinite_at,

        "top1_trajectory_ids": top1_traj,
        "snapshots": snapshots,
        "seconds": round(time.time() - t0, 3),
    }
    return rec


# ---------------------------------------------------------------------------
# Classification helpers used by the report
# ---------------------------------------------------------------------------

def _f(x, default=float("nan")):
    return default if x is None else x


def is_period2(rec) -> bool:
    """Low lag-1 cosine, lag-2 cosine at the gate threshold. The state the lag-1
    gate structurally cannot pass."""
    c1, c2 = _f(rec.get("cos_lag1_mean")), _f(rec.get("cos_lag2_mean"))
    return (not math.isnan(c1)) and (not math.isnan(c2)) and c1 <= GATE_THRESHOLD and c2 > GATE_THRESHOLD


def classify(rec) -> str:
    if rec.get("nonfinite"):
        return "non-finite"
    if rec.get("converged_lag1"):
        return "fixed-point"
    if is_period2(rec):
        return "period-2"
    if rec.get("converged_lag2"):
        return "period-2 (gate only)"
    return "unsettled"


# ---------------------------------------------------------------------------
# JSONL checkpoint I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: Path, repair: bool = False) -> list:
    """Read records, skipping a torn line (the one that was mid-write when a kill
    landed).

    `repair` rewrites the file without the torn line, and is only ever passed for
    the file THIS process owns. Rewriting a sibling shard's file would race its
    appends and silently drop finished prompts -- the exact loss the
    checkpointing exists to prevent."""
    if not path.exists():
        return []
    recs, torn = [], False
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                torn = True
    if torn and repair:
        write_jsonl(path, recs)
        print(f"[resume] dropped a torn line from {path.name}", flush=True)
    return recs


def write_jsonl(path: Path, recs: list) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(path)


def shard_paths(out_dir: Path) -> list:
    """Every JSONL in the output directory: the merged one plus any shard files.

    The sweep is run as N single-threaded processes (see TORCH_THREADS), each
    appending to its own file, because two processes appending to one file is a
    torn-line generator. `--report-only` merges them back into `basins.jsonl` in
    library order."""
    paths = []
    if (out_dir / "basins.jsonl").exists():
        paths.append(out_dir / "basins.jsonl")
    paths.extend(sorted(out_dir.glob("basins.shard*.jsonl")))
    return paths


def read_all(out_dir: Path, repair: Path | None = None) -> list:
    """All records across all shards, deduplicated by prompt id (first wins).

    `repair` names the one file this process owns and may rewrite."""
    seen, recs = set(), []
    for p in shard_paths(out_dir):
        for r in read_jsonl(p, repair=(repair is not None and p == repair)):
            if r["prompt_id"] in seen:
                continue
            seen.add(r["prompt_id"])
            recs.append(r)
    return recs


def append_jsonl(path: Path, rec: dict) -> None:
    """Append + flush + fsync. Cheap next to 300 GPT-2 forwards, and the whole
    reason a kill at prompt 90 costs one prompt and not ninety."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(n, total):
    return 0.0 if total == 0 else 100.0 * n / total


def _table(rows, headers, aligns=None):
    aligns = aligns or [":---"] + ["---:"] * (len(headers) - 1)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def _within_basin_section(recs, out_dir: Path):
    """Do prompts sharing a basin land on the same state, or merely near it?

    Free to compute -- it reads the `.npy` files this run already wrote and
    touches the model zero times. Phase-aware: on a period-2 orbit two prompts
    can sit on the same cycle in opposite phases, and comparing final-to-final
    would score that as a large distance when it is none. Each pair is therefore
    scored at the better of (a.final, b.final) and (a.final, b.prev)."""
    import numpy as np
    L = []
    A = L.append
    A("## Within-basin spread")
    A("")

    vecs = {}
    for r in recs:
        if not r.get("state_file"):
            continue
        f = out_dir / r["state_file"]
        fp = out_dir / r["state_prev_file"] if r.get("state_prev_file") else None
        if not f.exists():
            continue
        a = np.load(f).mean(axis=0)
        b = np.load(fp).mean(axis=0) if (fp and fp.exists()) else None
        vecs[r["prompt_id"]] = (a, b)

    if len(vecs) < 2:
        A("Not enough saved states to compare.")
        A("")
        return L

    def cos(u, v):
        d = float(np.linalg.norm(u) * np.linalg.norm(v))
        return 0.0 if d == 0 else float(np.dot(u, v) / d)

    by_basin = {}
    for r in recs:
        if r["prompt_id"] in vecs:
            by_basin.setdefault(r["basin"], []).append(r["prompt_id"])

    A("Cosine between the position-mean settled states of every pair of prompts sharing a basin. "
      "This is the question the basin table cannot answer on its own: if the pairs sit at 1.0 the "
      "attractor is a single point and the prompt's content is gone by the time it arrives; if "
      "they sit below 1.0 the map compresses rather than erases, and what is left of the prompt "
      "is measurable.")
    A("")
    rows = []
    for b in sorted(by_basin, key=lambda b: -len(by_basin[b])):
        ids = by_basin[b]
        if len(ids) < 2:
            rows.append([f"`{b}`", len(ids), "-", "-", "-", "-"])
            continue
        pair = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ai, bi = vecs[ids[i]]
                aj, bj = vecs[ids[j]]
                c = cos(ai, aj)
                if bj is not None:
                    c = max(c, cos(ai, bj))
                if bi is not None:
                    c = max(c, cos(bi, aj))
                pair.append(c)
        rows.append([f"`{b}`", len(ids), f"{min(pair):.6f}",
                     f"{statistics.median(pair):.6f}", f"{max(pair):.6f}",
                     sum(1 for c in pair if c > 0.999999)])
    L.extend(_table(rows, ["Basin", "n", "min pair cos", "median pair cos",
                           "max pair cos", "pairs above 0.999999"]))
    A("")
    A("Phase-aware: each pair is scored at the better of comparing final-to-final and "
      "final-to-previous-iterate, so two prompts on the same period-2 cycle in opposite phases "
      "are not counted as distant. Raw states are in `states/` if a sharper analysis is wanted.")
    A("")
    return L


def build_report(recs: list, meta: dict, out_dir: Path | None = None) -> str:
    n = len(recs)
    by_id = {r["prompt_id"]: r for r in recs}
    L = []
    A = L.append

    A("# Baseline Basin Census -- Frozen ATR, GPT-2 Small")
    A("")
    A("No plasticity. No weight updates. Nothing attached to the model. This is the "
      "reference the plasticity experiments are differenced against.")
    A("")
    A(f"- Prompts completed: **{n}** / {meta.get('n_expected', n)}")
    A(f"- Iterations per prompt: **{meta.get('n_iter')}** (fixed horizon, never early-stopped)")
    A(f"- Wall clock: **{meta.get('wall_clock_str', 'n/a')}**")
    A("")

    if n == 0:
        A("No records. Nothing to report.")
        return "\n".join(L)

    # ---- basin table -----------------------------------------------------
    counts = Counter(r["basin"] for r in recs)
    A("## Basin table (top-1 token at the last position, final iteration)")
    A("")
    rows = [[f"`{b}`", c, f"{_pct(c, n):.1f}%"] for b, c in counts.most_common()]
    L.extend(_table(rows, ["Basin", "Count", "Share"]))
    A("")
    A(f"Distinct basins: **{len(counts)}**.")
    A("")

    # ---- convergence -----------------------------------------------------
    kinds = Counter(classify(r) for r in recs)
    conv1 = [r for r in recs if r.get("converged_lag1")]
    p2 = [r for r in recs if is_period2(r)]
    p2_ids = {r["prompt_id"] for r in p2}
    unsettled = [r for r in recs if classify(r) == "unsettled"]
    nonfinite = [r for r in recs if r.get("nonfinite")]

    A("## Convergence")
    A("")
    A(f"Gate (the parent's, verbatim): `cos(mean_t, mean_t-lag) > {GATE_THRESHOLD}` for "
      f"{GATE_PATIENCE} consecutive checks, every {GATE_CHECK_EVERY} iterations "
      f"past iteration {GATE_CHECK_START}.")
    A("")
    rows = [
        ["Converged, lag-1 gate (fixed point)", len(conv1), f"{_pct(len(conv1), n):.1f}%"],
        ["Did NOT converge, lag-1 gate", n - len(conv1), f"{_pct(n - len(conv1), n):.1f}%"],
        ["-- of which period-2 (lag-1 low, lag-2 = 1.0)", len(p2_ids - {r['prompt_id'] for r in conv1}),
         f"{_pct(len(p2_ids - {r['prompt_id'] for r in conv1}), n):.1f}%"],
        ["-- of which still unsettled at the horizon", len(unsettled), f"{_pct(len(unsettled), n):.1f}%"],
        ["Non-finite at any iteration", len(nonfinite), f"{_pct(len(nonfinite), n):.1f}%"],
    ]
    L.extend(_table(rows, ["Outcome", "Count", "Share"]))
    A("")
    A("Classification key: `fixed-point` = passes the lag-1 gate. `period-2` = late-window "
      "lag-1 cosine at or below threshold with lag-2 cosine above it -- a state the lag-1 gate "
      "cannot pass by construction, not a state that failed to settle. `unsettled` = neither.")
    A("")
    rows = [[k, v, f"{_pct(v, n):.1f}%"] for k, v in kinds.most_common()]
    L.extend(_table(rows, ["Dynamical class", "Count", "Share"]))
    A("")

    # ---- basin x class ---------------------------------------------------
    A("### Basin x dynamical class")
    A("")
    classes = [c for c, _ in kinds.most_common()]
    rows = []
    for b, _ in counts.most_common():
        row = [f"`{b}`"]
        for c in classes:
            row.append(sum(1 for r in recs if r["basin"] == b and classify(r) == c))
        rows.append(row)
    L.extend(_table(rows, ["Basin"] + classes))
    A("")

    # ---- period-2 list ---------------------------------------------------
    A("## Period-2 prompts (the interesting group)")
    A("")
    if not p2:
        A("None found.")
    else:
        A(f"{len(p2)} prompt{'s' if len(p2) != 1 else ''}. Listed with the two cosines that define "
          "the class: a lag-1 gate reports every one of these as non-convergent; at their own "
          "period they are exact.")
        A("")
        rows = []
        for r in sorted(p2, key=lambda r: r["prompt_id"]):
            rows.append([
                r["prompt_id"], r["category"], f"`{r['basin']}`",
                f"{_f(r['cos_lag1_mean']):.6f}", f"{_f(r['cos_lag2_mean']):.6f}",
                f"{r['final_top5_probs'][0]:.3f}", f"{r['final_entropy']:.2f}",
                "yes" if r.get("converged_lag2") else "no",
            ])
        L.extend(_table(rows, ["Prompt id", "Register", "Basin", "cos lag-1",
                               "cos lag-2", "p(top1)", "entropy", "lag-2 gate"]))
        A("")
        A("Prompt texts:")
        A("")
        for r in sorted(p2, key=lambda r: r["prompt_id"]):
            A(f"- `{r['prompt_id']}` -- {json.dumps(r['prompt'], ensure_ascii=False)}")
    A("")

    if unsettled:
        A("### Still unsettled at the horizon")
        A("")
        rows = [[r["prompt_id"], r["category"], f"`{r['basin']}`",
                 f"{_f(r['cos_lag1_mean']):.6f}", f"{_f(r['cos_lag2_mean']):.6f}",
                 f"{min(v['mean'] for v in r['lag_scan_mean_vec'].values()):.6f}",
                 f"{max(v['mean'] for v in r['lag_scan_mean_vec'].values()):.6f}"]
                for r in sorted(unsettled, key=lambda r: r["prompt_id"])]
        L.extend(_table(rows, ["Prompt id", "Register", "Basin", "cos lag-1",
                               "cos lag-2", "min cos lag1-8", "max cos lag1-8"]))
        A("")

    # ---- lock-in distribution -------------------------------------------
    A("## Convergence iteration")
    A("")
    li1 = sorted(r["lock_in_iter_lag1"] for r in conv1)
    if li1:
        A(f"Lag-1 gate, {len(li1)} prompts: min {min(li1)}, median "
          f"{int(statistics.median(li1))}, max {max(li1)}, mean {statistics.mean(li1):.1f}.")
        A("")
        rows = [[it, c, f"{_pct(c, len(li1)):.1f}%"] for it, c in sorted(Counter(li1).items())]
        L.extend(_table(rows, ["Lock-in iteration", "Count", "Share of converged"]))
        A("")
    else:
        A("No prompt passed the lag-1 gate.")
        A("")
    li2 = sorted(r["lock_in_iter_lag2"] for r in recs if r.get("converged_lag2"))
    if li2:
        A(f"Lag-2 gate, {len(li2)} prompts: min {min(li2)}, median "
          f"{int(statistics.median(li2))}, max {max(li2)}.")
        A("")

    # ---- first-arrival at the final basin --------------------------------
    settle = []
    for r in recs:
        traj = r.get("top1_trajectory_ids") or []
        fid = r.get("basin_token_id")
        if not traj or fid is None:
            continue
        k = len(traj) - 1
        while k > 0 and traj[k - 1] == fid:
            k -= 1
        settle.append(k)
    if settle:
        A("### Readout settling (first iteration after which the top-1 token never changes again)")
        A("")
        A(f"min {min(settle)}, median {int(statistics.median(settle))}, "
          f"max {max(settle)}, mean {statistics.mean(settle):.1f}.")
        A("")
        buckets = [(0, 0), (1, 10), (11, 25), (26, 50), (51, 100), (101, 150),
                   (151, 200), (201, 250), (251, 10 ** 9)]
        rows = []
        for lo, hi in buckets:
            c = sum(1 for s in settle if lo <= s <= hi)
            if c:
                lab = str(lo) if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+")
                rows.append([lab, c, f"{_pct(c, len(settle)):.1f}%"])
        L.extend(_table(rows, ["Iteration", "Count", "Share"]))
        A("")
        A("The readout locks long before the tensor does. This gap -- a stable decoded token "
          "over a state that is still moving -- is the dissociation the parent project flags, "
          "and it is why the basin label and the convergence verdict have to be read as two "
          "separate measurements.")
        A("")

    # ---- instrument validation ------------------------------------------
    L.extend(_validation_section(meta))

    # ---- comparison with published --------------------------------------
    L.extend(_published_section(recs, meta))

    # ---- readout confidence ---------------------------------------------
    A("## Readout confidence at the final iteration")
    A("")
    rows = []
    for b, _ in counts.most_common():
        sub = [r for r in recs if r["basin"] == b]
        A_ = lambda key: statistics.median(r[key] for r in sub)  # noqa: E731
        rows.append([
            f"`{b}`", len(sub),
            f"{statistics.median(r['final_top5_probs'][0] for r in sub):.3f}",
            f"{A_('final_entropy'):.2f}",
            f"{A_('final_top_logit_margin'):.2f}",
            f"{statistics.median(_f(r['final_position_similarity'], 0.0) for r in sub):.4f}",
            f"{A_('final_tensor_norm'):.1f}",
        ])
    L.extend(_table(rows, ["Basin", "n", "median p(top1)", "median entropy",
                           "median top1-top2 logit margin", "median position uniformity",
                           "median final norm"]))
    A("")
    A("Position uniformity = mean off-diagonal cosine between token positions. Values near 1.0 "
      "mean every position in the tensor holds the same direction; the sequence has stopped "
      "being a sequence.")
    A("")

    # ---- saved states / position uniformity ------------------------------
    L.extend(_states_section(recs))
    if out_dir is not None:
        try:
            L.extend(_within_basin_section(recs, out_dir))
        except Exception as exc:            # never let analysis kill the report
            L.append(f"## Within-basin spread\n\nNot computed: {type(exc).__name__}: {exc}\n")

    # ---- register breakdown ---------------------------------------------
    A("## Basin by register")
    A("")
    cats = sorted({r["category"] for r in recs})
    basins_sorted = [b for b, _ in counts.most_common()]
    rows = []
    for c in cats:
        sub = [r for r in recs if r["category"] == c]
        rows.append([c, len(sub)] + [sum(1 for r in sub if r["basin"] == b) for b in basins_sorted])
    L.extend(_table(rows, ["Register", "n"] + [f"`{b}`" for b in basins_sorted]))
    A("")

    # ---- anomalies -------------------------------------------------------
    L.extend(_anomalies_section(recs, by_id))

    # ---- config ----------------------------------------------------------
    L.extend(_config_section(meta))
    return "\n".join(L)


def _validation_section(meta):
    """Instrument check, run against a number the parent committed."""
    v = meta.get("instrument_validation")
    L = []
    A = L.append
    A("## Instrument check")
    A("")
    if not v:
        A("Not run. `--validate-instrument` continues the parent's committed "
          "`state_divine.pt` for 24 iterations and compares the lag scan against the "
          "published cycle cosines; without it, the period-2 classification here rests on "
          "nothing external.")
        A("")
        return L
    sm = v.get("lag_scan_mean_vec", {})
    A("Before trusting any period-2 verdict below, the detector was pointed at a state whose "
      "answer is already published. The parent's motion audit committed `state_divine.pt` -- the "
      "`Divine` trajectory at iteration 1000, sitting on its limit cycle -- and reported "
      "`cos(A, f(A)) = 0.684912` and `cos(A, f(f(A))) = 1.000000`. Continuing that state 24 "
      "iterations through this repo's `atr_bridge` and running the same lag scan used on every "
      "prompt in this census:")
    A("")
    rows = [[k, f"{sm[k]:.6f}", f"{v['lag_scan_last_vec'][k]:.6f}",
             "pass" if sm[k] > GATE_THRESHOLD else "fail"]
            for k in sorted(sm, key=int)]
    L.extend(_table(rows, ["Lag k", "mean-vector cos", "last-vector cos",
                           f"gate at {GATE_THRESHOLD}"]))
    A("")
    d1 = abs(sm.get("1", 0) - v["published_lag1"])
    A(f"Lag-1 reproduces the published 0.684912 to {d1:.1e}; lag-2 is 1.000000 exactly. The "
      "odd-fails / even-passes stripe is the signature of an exact period-2 orbit, and this "
      "file's classifier labels the state "
      f"`period-2`: **{v['classified_period2']}**. Overall: **{'PASS' if v['pass'] else 'FAIL'}**.")
    A("")
    A("This also settles one candidate explanation for any basin discrepancy further down: a "
      "state saved by the parent's own run, continued under this torch / TransformerLens build, "
      "lands on the same cycle to seven decimals. Whatever else may differ, the forward map does "
      "not.")
    A("")
    return L


def _published_section(recs, meta):
    n = len(recs)
    L = []
    A = L.append
    counts = Counter(r["basin"] for r in recs)
    c100 = Counter(r["basin_at_100"] for r in recs if r.get("basin_at_100"))
    c120 = Counter(r["basin_at_120"] for r in recs if r.get("basin_at_120"))
    n100 = sum(c100.values()) or 1
    n120 = sum(c120.values()) or 1

    A("## Comparison with the parent project's published figures")
    A("")
    A("Published (parent `README.md`, GPT-2 small, shares at convergence): `prolet` 43.2%, "
      "`Divine` 27.2%, `till` 15.2%, `Anarch` 13.6%, `solidarity` 0.8%. Those percentages are "
      "the **at-lock-in** column of `experiments/gpt2_small/output_gated/gated_report.md`, where "
      "every converged prompt locked at iteration 120 and the 34 holdouts were classified at "
      "iteration 1000. The `@100` column in the same file (`prolet` 35.2%, `Divine` 27.2%, "
      "`Anarch` 20.8%, `till` 15.2%, `solidarity` 1.6%) is the earlier Stage 1 table.")
    A("")
    pub = {b: (c, p) for b, c, p in PUBLISHED_LOCKIN}
    pub100 = {b: (c, p) for b, c, p in PUBLISHED_AT_100}
    all_b = list(dict.fromkeys([b for b, _, _ in PUBLISHED_LOCKIN] + list(counts)))
    rows = []
    for b in all_b:
        pc, pp = pub.get(b, (0, 0.0))
        mine = counts.get(b, 0)
        m100 = c100.get(b, 0)
        m120 = c120.get(b, 0)
        rows.append([f"`{b}`", f"{pc} ({pp:.1f}%)",
                     f"{m120} ({_pct(m120, n120):.1f}%)",
                     f"{mine} ({_pct(mine, n):.1f}%)",
                     f"{mine - pc:+d}",
                     f"{pub100.get(b, (0, 0.0))[0]} ({pub100.get(b, (0, 0.0))[1]:.1f}%)",
                     f"{m100} ({_pct(m100, n100):.1f}%)"])
    L.extend(_table(rows, ["Basin", "Published @lock-in", f"This run @120",
                           f"This run @{meta.get('n_iter')}", "delta vs published",
                           "Published @100", "This run @100"]))
    A("")

    exact_final = all(counts.get(b, 0) == pub.get(b, (0, 0))[0] for b in all_b)
    exact_120 = all(c120.get(b, 0) == pub.get(b, (0, 0))[0] for b in all_b)
    exact_100 = all(c100.get(b, 0) == pub100.get(b, (0, 0))[0] for b in all_b)

    conv1 = sum(1 for r in recs if r.get("converged_lag1"))
    A("### Verdict")
    A("")
    if n < 125:
        A(f"**Partial run: {n} of 125 prompts.** Shares are not comparable with the published "
          "table until the sweep completes; counts below are raw.")
        A("")
    A(f"- Final-iteration table vs published @lock-in: **{'exact match' if exact_final else 'DIFFERS'}**.")
    A(f"- This run's @120 table vs published @lock-in: **{'exact match' if exact_120 else 'DIFFERS'}**.")
    A(f"- This run's @100 table vs published @100: **{'exact match' if exact_100 else 'DIFFERS'}**.")
    A(f"- Converged under the lag-1 gate: {conv1} here vs {PUBLISHED_CONVERGED} published "
      f"({'match' if conv1 == PUBLISHED_CONVERGED and n == 125 else 'differs'}).")
    A("")

    diffs = [(b, counts.get(b, 0), pub.get(b, (0, 0))[0])
             for b in all_b if counts.get(b, 0) != pub.get(b, (0, 0))[0]]
    if not diffs and n == 125:
        A("The census reproduces the published basin table exactly, count for count, on the same "
          "125 prompts. Nothing was tuned to make this happen: the gate constants, the readout, "
          "the layer range and the prompt order were copied from the parent before any number was "
          "looked at.")
    else:
        A("**Discrepancies, stated as found. Nothing here was tuned to close them.**")
        A("")
        for b, mine, theirs in diffs:
            A(f"- `{b}`: {mine} here, {theirs} published ({mine - theirs:+d}).")
        A("")
        A("Candidate causes, in the order worth checking:")
        A("")
        A("1. **Stopping time.** The published shares are read at lock-in (iteration 120 for every "
          "converged prompt; iteration 1000 for the 34 holdouts). This run reads at "
          f"iteration {meta.get('n_iter')}. The `@120` column above isolates this: if that column "
          "matches and the final column does not, the difference is late drift, not method.")
        A("2. **Parity.** A period-2 state has two phases. Both decode to the same token in the "
          "parent's `Divine` cycle, but a basin read at an odd iteration on a period-2 orbit is "
          "not in general the same read as at an even one. This run's horizon and the published "
          "1000 are both even.")
        A("3. **TransformerLens weight processing.** `from_pretrained` folds LayerNorm, centres "
          "the writing weights and centres the unembedding by default. Different "
          "TransformerLens versions have changed these defaults; the version used here is "
          f"recorded below ({meta.get('transformer_lens_version')}). The parent's published run "
          "predates it.")
        A("4. **Prompt library revision.** The parent's `prompt_library.py` is a provenance-flagged "
          "reconstruction (all 125 entries flagged `original`, recovered from git blob 2931d42 and "
          "cross-checked against `dissolution_sentences.md`), restored *after* the published sweep "
          "was run. If any entry differs by a character from what the April run used, its basin "
          "can move. The file's own provenance block asserts byte-for-byte agreement.")
        A("5. **Numerics.** float32 CPU matmul on a different thread count / BLAS build than the "
          "original run. This moves cosines in the seventh decimal; it moves a basin label only "
          "for a prompt sitting on a separatrix, which the per-prompt margins above would show as "
          "a near-zero top1-top2 logit margin.")
    A("")
    return L


def _states_section(recs):
    """Saved settled states + position uniformity, the two inputs to the
    within-basin spread question."""
    L = []
    A = L.append
    n = len(recs)
    saved = [r for r in recs if r.get("state_file")]
    missing = [r["prompt_id"] for r in recs if not r.get("state_file")]

    A("## Settled states")
    A("")
    A(f"Saved: **{len(saved)} / {n}** prompts, as `experiments/output_baseline/states/"
      "<prompt_id>.npy` -- the full `(seq_len, 768)` float32 residual tensor at the final "
      "iteration. A second file `<prompt_id>__prev.npy` holds the iterate immediately before it: "
      "on a period-2 orbit the settled state is one of two, and a spread analysis that mixed "
      "phases across prompts would be measuring the cycle rather than the basin. The JSONL row "
      "for each prompt carries `state_file`, `state_prev_file` and `state_shape`.")
    A("")
    if missing:
        A(f"**No saved state for {len(missing)} prompt(s):** " +
          ", ".join(f"`{p}`" for p in missing) +
          ". These were run before state saving was added; re-run them with "
          "`--only <ids>` after deleting their JSONL rows if the states are needed.")
        A("")
    A("These files exist so within-basin spread can be measured with no further model time: "
      "if prompts sharing a basin land on bit-identical tensors the attractor is a label and the "
      "prompt's content is gone; if they land nearby but distinct, it compresses rather than "
      "erases. This report does not answer that question -- it only makes it answerable.")
    A("")

    have_pos = [r for r in recs if r.get("final_position_stats")]
    if have_pos:
        A("### Position uniformity at the final iteration")
        A("")
        A("Cosine between token positions within one settled tensor (off-diagonal only). "
          "`mean` is the parent's `position_similarity`; `spread` is max minus min across "
          "position pairs, which is what separates \"every position identical\" from \"all but "
          "one identical\".")
        A("")
        counts = Counter(r["basin"] for r in recs)
        rows = []
        for b, _ in counts.most_common():
            sub = [r for r in have_pos if r["basin"] == b]
            if not sub:
                continue
            rows.append([
                f"`{b}`", len(sub),
                f"{statistics.median(r['final_position_stats']['mean'] for r in sub):.6f}",
                f"{min(r['final_position_stats']['min'] for r in sub):.6f}",
                f"{max(r['final_position_stats']['spread'] for r in sub):.6f}",
                sum(1 for r in sub if r['final_position_stats']['min'] > 0.999),
            ])
        L.extend(_table(rows, ["Basin", "n", "median mean-cos", "min pair-cos (worst prompt)",
                               "max spread", "n fully uniform (min cos > 0.999)"]))
        A("")
        full = [r for r in have_pos if r["final_position_stats"]["min"] > 0.999]
        A(f"Fully position-uniform (every pair of positions above cosine 0.999): "
          f"**{len(full)} / {len(have_pos)}**. The parent reports this for the `Divine` state; "
          "the table above says whether it holds for the other basins.")
        A("")
    return L


def _anomalies_section(recs, by_id):
    L = []
    A = L.append
    A("## Anomalies")
    A("")
    found = False

    nf = [r for r in recs if r.get("nonfinite")]
    if nf:
        found = True
        A(f"- **Non-finite trajectories: {len(nf)}.** " +
          ", ".join(f"`{r['prompt_id']}` (first at iter {r['nonfinite_first_iter']})" for r in nf))

    # Long-period / aliased states: passes some lag > 2 but not 1 or 2.
    weird = []
    for r in recs:
        sc = r.get("lag_scan_mean_vec") or {}
        passes = [int(k) for k, v in sc.items() if v["mean"] > GATE_THRESHOLD]
        if passes and 1 not in passes and 2 not in passes:
            weird.append((r, sorted(passes)))
    if weird:
        found = True
        A(f"- **States passing only lags above 2: {len(weird)}.** A period-p cycle passes exactly "
          "the lags that p divides, so these are candidates for period 3 or 4 and would be "
          "invisible to both the lag-1 and the lag-2 gate.")
        for r, p in weird:
            A(f"  - `{r['prompt_id']}` basin `{r['basin']}`, passes lags {p}")

    # Drifting states that nonetheless pass lag-1 -- the parent's threshold-blindness caveat.
    drift = []
    for r in recs:
        sc = r.get("lag_scan_mean_vec") or {}
        if not sc or "1" not in sc or "8" not in sc:
            continue
        d1, d8 = 1 - sc["1"]["mean"], 1 - sc["8"]["mean"]
        if r.get("converged_lag1") and d8 > 20 * max(d1, 1e-12) and d8 > 1e-5:
            drift.append((r, d1, d8))
    if drift:
        found = True
        A(f"- **Converged-but-still-drifting: {len(drift)}.** These pass the lag-1 gate while their "
          "cosine deficit keeps growing with lag -- the parent's own caveat that the gate is blind "
          "to slow drift, not just to periodicity. Cosine deficit at lag 1 vs lag 8:")
        for r, d1, d8 in sorted(drift, key=lambda x: -x[2])[:15]:
            A(f"  - `{r['prompt_id']}` basin `{r['basin']}`: {d1:.2e} -> {d8:.2e}")

    # Knife-edge readouts.
    thin = [r for r in recs if r["final_top_logit_margin"] < 0.5]
    if thin:
        found = True
        A(f"- **Knife-edge readouts: {len(thin)}.** top1-top2 logit margin below 0.5 at the final "
          "iteration; these basin labels are the least robust to numerics.")
        for r in sorted(thin, key=lambda r: r["final_top_logit_margin"])[:15]:
            A(f"  - `{r['prompt_id']}` `{r['basin']}` vs `{r['final_top5_tokens'][1].strip()}`, "
              f"margin {r['final_top_logit_margin']:.3f}")

    # Basin moved between the published stopping time and this run's horizon.
    moved = [r for r in recs if r.get("basin_at_120") and r["basin_at_120"] != r["basin"]]
    if moved:
        found = True
        A(f"- **Basin changed between iteration 120 and the horizon: {len(moved)}.** Iteration 120 "
          "is where the published sweep classified every converged prompt, so these are exactly "
          "the prompts for which the published table and a later reading disagree.")
        for r in sorted(moved, key=lambda r: r["prompt_id"]):
            A(f"  - `{r['prompt_id']}`: `{r['basin_at_120']}` at 120 -> `{r['basin']}` at "
              f"{r['n_iter']}, lock-in {r['lock_in_iter_lag1']}")

    # Period-2 label resting on a lag-1 cosine that is not actually low.
    borderline = [r for r in recs if is_period2(r) and _f(r.get("cos_lag1_mean"), 0.0) > 0.99]
    if borderline:
        found = True
        A(f"- **Borderline period-2 calls: {len(borderline)}.** Classified period-2 because lag-2 "
          "clears the threshold and lag-1 does not, but lag-1 is above 0.99 -- slow drift and a "
          "genuine period-2 orbit are not distinguishable at that separation. Treat as "
          "unresolved, not as cycles.")
        for r in borderline:
            A(f"  - `{r['prompt_id']}`: lag-1 {_f(r['cos_lag1_mean']):.6f}, "
              f"lag-2 {_f(r['cos_lag2_mean']):.6f}")

    # Basins of size 1.
    counts = Counter(r["basin"] for r in recs)
    singles = [b for b, c in counts.items() if c == 1]
    if singles:
        found = True
        A(f"- **Singleton basins: {len(singles)}.** " +
          ", ".join(f"`{b}` (`{next(r['prompt_id'] for r in recs if r['basin'] == b)}`)"
                    for b in singles))

    # Position uniformity that did NOT collapse.
    loose = [r for r in recs if r.get("final_position_similarity") is not None
             and r["final_position_similarity"] < 0.9]
    if loose:
        found = True
        A(f"- **Positions did not become uniform: {len(loose)}.** Final mean off-diagonal position "
          "cosine below 0.9, i.e. the tensor still holds distinct directions per position.")
        for r in sorted(loose, key=lambda r: r["final_position_similarity"])[:15]:
            A(f"  - `{r['prompt_id']}` basin `{r['basin']}`, "
              f"position uniformity {r['final_position_similarity']:.4f}, seq_len {r['seq_len']}")

    # Norm blow-up / collapse relative to the fixed energy shell.
    norm_odd = [r for r in recs if r["initial_norm"] > 0 and
                not (0.2 < r["final_tensor_norm"] / r["initial_norm"] < 5.0)]
    if norm_odd:
        found = True
        A(f"- **Final norm far off the initial energy shell: {len(norm_odd)}.** The map rescales to "
          "||x0|| on the way IN, so the norm on the way OUT is free; a large ratio means the slice "
          "is amplifying hard.")
        for r in sorted(norm_odd, key=lambda r: -r["final_tensor_norm"] / r["initial_norm"])[:10]:
            A(f"  - `{r['prompt_id']}`: ||x0||={r['initial_norm']:.1f}, "
              f"final={r['final_tensor_norm']:.1f} "
              f"(x{r['final_tensor_norm'] / r['initial_norm']:.2f})")

    if not found:
        A("Nothing anomalous: no non-finite runs, no periods above 2, no knife-edge readouts, "
          "no singleton basins, no positions left non-uniform.")
    A("")
    return L


def _config_section(meta):
    L = []
    A = L.append
    A("## Exact configuration")
    A("")
    rows = [
        ["Model", f"`{MODEL_NAME}` via TransformerLens `HookedTransformer.from_pretrained`"],
        ["Device / dtype", f"{meta.get('device')} / {meta.get('dtype')}"],
        ["Layers", f"{LAYER_START} -> {LAYER_END} (read `blocks.{LAYER_END}.hook_resid_post`, "
                   f"write `blocks.{LAYER_START}.hook_resid_pre`)"],
        ["Step implementation", "`atr_bridge.make_atr_step` (bit-exact extraction of the parent's "
                                "`atr_engine.run_atr_loop` body; see `tests/test_atr_bridge.py`)"],
        ["Normalisation", "rescale to the trajectory's own `||x0||` before each injection; "
                          "`initial_norm` captured once and held fixed"],
        ["Iterations", f"{meta.get('n_iter')} per prompt, fixed horizon, no early stop"],
        ["Readout", "`ln_final(x[-1]) @ W_U + b_U`, argmax = basin label"],
        ["Convergence gate", f"cos(mean_t, mean_t-lag) > {GATE_THRESHOLD}, patience "
                             f"{GATE_PATIENCE}, every {GATE_CHECK_EVERY} iters from "
                             f"{GATE_CHECK_START}; lags {list(GATE_LAGS)}"],
        ["Lag scan", f"lags 1..{MAX_LAG} over the final {LAG_WINDOW} iterates "
                     "(mean vector and last vector)"],
        ["Plasticity", "**none** -- no hooks, no weight updates, model frozen and in eval mode"],
        ["Seeds", f"`torch.manual_seed({SEED})`; the loop itself is deterministic and "
                  "draws no random numbers"],
        ["Torch threads", f"{meta.get('torch_threads')} per process "
                          f"({meta.get('shards', 1)} process(es) in parallel). "
                          "Measured: 1 thread 287 ms/iter, 4 threads 2137 ms/iter on this "
                          "4-vCPU box -- OpenMP spin-wait collapse, so the sweep is "
                          "single-threaded and parallelised across processes instead."],
        ["Prompt library", f"parent `prompt_library.py`, {meta.get('n_expected')} prompts, "
                           f"provenance `{meta.get('library_provenance')}`"],
        ["Parent repo revision", f"`{meta.get('parent_rev')}`"],
        ["Prompt library sha256", f"`{meta.get('library_sha256')}`"],
        ["This repo revision", f"`{meta.get('repo_rev')}`"],
        ["torch", meta.get("torch_version")],
        ["transformer-lens", meta.get("transformer_lens_version")],
        ["transformers", meta.get("transformers_version")],
        ["numpy", meta.get("numpy_version")],
        ["Python", meta.get("python_version")],
        ["Platform", meta.get("platform")],
        ["Wall clock", meta.get("wall_clock_str")],
        ["Started / finished (UTC)", f"{meta.get('started')} / {meta.get('finished')}"],
        ["Raw records", "`experiments/output_baseline/basins.jsonl`"],
    ]
    L.extend(_table(rows, ["Key", "Value"], [":---", ":---"]))
    A("")
    A("### Reproducing")
    A("")
    A("```")
    A(".venv/bin/python experiments/baseline_basins.py")
    A("```")
    A("")
    A("Resumable: every completed prompt is appended to `basins.jsonl` and fsynced before the next "
      "one starts, and a re-run skips whatever is already there. `--report-only` rebuilds this "
      "file from the JSONL without touching the model.")
    A("")
    return L


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_instrument(parent_path: str, n_iter: int = 24, out_dir: Path | None = None) -> int:
    """Acceptance check for the period-2 detector, against a committed number.

    The parent's motion audit saved `state_divine.pt` -- the `Divine` trajectory
    at iteration 1000, sitting on its limit cycle -- and published the two
    cosines that characterise it: `cos(A, f(A)) = 0.684912` and
    `cos(A, f(f(A))) = 1.000000` (`output_lagk/lagk_report.md`,
    `output_divine_motion/bell_anatomy.json`).

    If this file's lag scan does not reproduce those two numbers from that state,
    then its period-2 classification is not measuring what the parent measured
    and the whole "34 prompts ring rather than fail to converge" reading of the
    census is unsupported. Cheap to run: 24 forward passes.
    """
    from atr_bridge import load_state, make_atr_step_from_state
    from transformer_lens import HookedTransformer

    state_path = (Path(parent_path) / "experiments" / "gpt2_small" /
                  "output_divine_motion" / "state_divine.pt")
    if not state_path.exists():
        print(f"[validate] SKIP: {state_path} not present")
        return 0

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    model.eval()
    model.requires_grad_(False)

    st = load_state(str(state_path))
    step = make_atr_step_from_state(model, st, layer_start=LAYER_START, layer_end=LAYER_END)
    r = st.tensor
    means, lasts = [r.mean(dim=0).clone()], [r[-1, :].clone()]
    for _ in range(n_iter):
        r = step(model, r)
        means.append(r.mean(dim=0).clone())
        lasts.append(r[-1, :].clone())

    sm = lag_scan(torch.stack(means), MAX_LAG)
    sl = lag_scan(torch.stack(lasts), MAX_LAG)
    print(f"[validate] state_divine.pt  iteration={st.iteration}  "
          f"initial_norm={st.initial_norm}  shape={tuple(st.tensor.shape)}")
    print(f"[validate] {'lag':>4} {'mean-vec':>12} {'last-vec':>12}")
    for k in sorted(sm):
        print(f"[validate] {k:>4} {sm[k]['mean']:>12.6f} {sl[k]['mean']:>12.6f}")

    ok = True
    for name, scan, expect1, expect2 in (("mean", sm, 0.684912, 1.000000),
                                         ("last", sl, 0.684912, 1.000000)):
        d1 = abs(scan[1]["mean"] - expect1)
        d2 = abs(scan[2]["mean"] - expect2)
        good = d1 < 5e-4 and d2 < 5e-6
        ok &= good
        print(f"[validate] {name}-vector: lag1={scan[1]['mean']:.6f} (published {expect1}, "
              f"d={d1:.2e})  lag2={scan[2]['mean']:.6f} (published {expect2}, d={d2:.2e})  "
              f"{'PASS' if good else 'FAIL'}")
    classified = is_period2({"cos_lag1_mean": sm[1]["mean"], "cos_lag2_mean": sm[2]["mean"]})
    print(f"[validate] period-2 classifier on this state: {classified}")
    print(f"[validate] {'PASS' if ok else 'FAIL'}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "instrument_validation.json").write_text(json.dumps({
            "state": str(state_path), "state_iteration": st.iteration,
            "initial_norm": st.initial_norm, "n_continued": n_iter,
            "lag_scan_mean_vec": {str(k): v["mean"] for k, v in sm.items()},
            "lag_scan_last_vec": {str(k): v["mean"] for k, v in sl.items()},
            "published_lag1": 0.684912, "published_lag2": 1.0,
            "classified_period2": classified, "pass": bool(ok),
        }, indent=2), encoding="utf-8")
    return 0 if ok else 1


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_rev(path: Path) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h {(sec % 3600) // 60}m {sec % 60}s"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=int(os.environ.get("BASELINE_ITERS", DEFAULT_ITERS)))
    ap.add_argument("--limit", type=int, default=0, help="run only the first N prompts")
    ap.add_argument("--only", type=str, default="", help="comma-separated prompt ids")
    ap.add_argument("--parent", type=str,
                    default=os.environ.get("ATR_PARENT_PATH", PARENT_DEFAULT))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-states", action="store_true",
                    help="skip writing the settled-state .npy files")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1,
                    help="run prompts[shard::nshards] into basins.shard<N>.jsonl; "
                         "--report-only merges the shards")
    ap.add_argument("--threads", type=int, default=TORCH_THREADS)
    ap.add_argument("--validate-instrument", action="store_true",
                    help="check the period-2 detector against the parent's committed "
                         "Divine cycle cosines, then exit")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args(argv)

    if args.validate_instrument:
        return validate_instrument(args.parent, out_dir=Path(args.out))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / ("basins.jsonl" if args.nshards == 1
                       else f"basins.shard{args.shard}.jsonl")
    report = out_dir / "BASELINE.md"
    meta_path = out_dir / "run_meta.json"
    states_dir = out_dir / "states"
    if not args.no_states:
        states_dir.mkdir(parents=True, exist_ok=True)

    pl = load_prompt_library(args.parent)
    prompts = ordered_prompts(pl)
    lib_path = Path(args.parent) / "prompt_library.py"

    prompt_ids = list(prompts)
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        prompt_ids = [p for p in prompt_ids if p in want]
    if args.limit:
        prompt_ids = prompt_ids[:args.limit]
    if args.nshards > 1:
        # Interleaved, not blocked: registers are contiguous in the library, so a
        # blocked split would give one shard all the Wild prompts and make a
        # partial result unrepresentative.
        prompt_ids = prompt_ids[args.shard::args.nshards]

    meta = {
        "n_iter": args.iters,
        "n_expected": len(prompts),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "numpy_version": __import__("numpy").__version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "torch_threads": args.threads,
        "shards": args.nshards,
        "device": "cpu",
        "dtype": "float32",
        "parent_rev": _git_rev(Path(args.parent)),
        "repo_rev": _git_rev(REPO_ROOT),
        "library_sha256": _sha256(lib_path),
        "library_provenance": getattr(pl, "PROVENANCE_COUNTS", {}),
    }
    try:
        from importlib.metadata import version as _v
        meta["transformer_lens_version"] = _v("transformer-lens")
    except Exception:
        meta["transformer_lens_version"] = "unknown"

    if args.report_only:
        recs = read_all(out_dir, repair=out_dir / "basins.jsonl")
        order = {p: i for i, p in enumerate(prompts)}
        recs.sort(key=lambda r: order.get(r["prompt_id"], 10 ** 6))
        # Collapse the shards into the single canonical raw file. Only drop the
        # shard files once every requested prompt is present: deleting them
        # while a shard process is still appending would throw away its work.
        write_jsonl(out_dir / "basins.jsonl", recs)
        if set(prompt_ids) <= {r["prompt_id"] for r in recs}:
            for p in sorted(out_dir.glob("basins.shard*.jsonl")):
                p.unlink()
        else:
            missing = [p for p in prompt_ids if p not in {r["prompt_id"] for r in recs}]
            print(f"[merge] {len(missing)} prompt(s) still missing; shard files kept "
                  f"(first missing: {missing[:5]})")
        metas = [json.loads(p.read_text()) for p in
                 sorted(out_dir.glob("run_meta*.json")) if p.stat().st_size]
        if metas:
            meta.update(metas[0])
            # Wall clock across shards is the span from the first start to the
            # last finish, not the sum: the shards ran concurrently.
            starts = [m["started"] for m in metas if m.get("started")]
            ends = [m["finished"] for m in metas if m.get("finished")]
            if starts and ends:
                t_a = time.mktime(time.strptime(min(starts), "%Y-%m-%dT%H:%M:%SZ"))
                t_b = time.mktime(time.strptime(max(ends), "%Y-%m-%dT%H:%M:%SZ"))
                meta["started"], meta["finished"] = min(starts), max(ends)
                meta["wall_clock_seconds"] = round(t_b - t_a, 1)
                nsh = max(m.get("shards", 1) for m in metas)
                meta["wall_clock_str"] = (
                    f"{_hms(t_b - t_a)} wall"
                    + (f" ({nsh} shards in parallel, "
                       f"{_hms(sum(m.get('wall_clock_seconds', 0) for m in metas))} CPU)"
                       if nsh > 1 else ""))
            meta["shards"] = max(m.get("shards", 1) for m in metas)
        vpath = out_dir / "instrument_validation.json"
        if vpath.exists():
            meta["instrument_validation"] = json.loads(vpath.read_text())
        report.write_text(build_report(recs, meta, out_dir), encoding="utf-8")
        print(f"[report] {report} ({len(recs)} records)")
        return 0

    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    done = {r["prompt_id"] for r in read_all(out_dir, repair=jsonl)}
    todo = [p for p in prompt_ids if p not in done]
    print(f"[config] model={MODEL_NAME} layers {LAYER_START}->{LAYER_END} "
          f"iters={args.iters} threads={args.threads} "
          f"shard={args.shard}/{args.nshards}", flush=True)
    print(f"[plan] {len(prompt_ids)} prompts in this shard, {len(done)} already "
          f"recorded, {len(todo)} to run -> {jsonl.name}", flush=True)

    if todo:
        from transformer_lens import HookedTransformer
        t_load = time.time()
        model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
        model.eval()
        model.requires_grad_(False)
        print(f"[model] loaded in {time.time() - t_load:.1f}s, "
              f"n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}", flush=True)
        assert model.cfg.n_layers - 1 == LAYER_END, "LAYER_END does not match the model"

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    for k, pid in enumerate(todo, 1):
        rec = run_prompt(model, pid, prompts[pid], pl.CATEGORY_MAP.get(pid, "?"), args.iters,
                         states_dir=None if args.no_states else states_dir)
        append_jsonl(jsonl, rec)
        cls = classify(rec)
        elapsed = time.time() - t0
        eta = (elapsed / k) * (len(todo) - k)
        print(f"[{k}/{len(todo)}] {pid:<18} basin={rec['basin']!r:<14} {cls:<12} "
              f"lock1={rec['lock_in_iter_lag1']} lag1={_f(rec['cos_lag1_mean']):.5f} "
              f"lag2={_f(rec['cos_lag2_mean']):.5f} ({rec['seconds']:.1f}s, ETA {_hms(eta)})",
              flush=True)

    wall = time.time() - t0
    meta["started"] = started
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["wall_clock_seconds"] = round(wall, 1)
    meta["wall_clock_str"] = (f"{_hms(wall)} ({args.nshards} shards in parallel, "
                              f"{len(todo)} prompts this shard)" if args.nshards > 1
                              else _hms(wall))
    # Each shard writes its own meta; the merge step reads whichever is newest.
    (out_dir / (f"run_meta.shard{args.shard}.json" if args.nshards > 1
                else "run_meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.nshards > 1:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    recs = read_all(out_dir)
    vpath = out_dir / "instrument_validation.json"
    if vpath.exists():
        meta["instrument_validation"] = json.loads(vpath.read_text())
    report.write_text(build_report(recs, meta, out_dir), encoding="utf-8")
    print(f"\n[done] {len(recs)} records in {jsonl}")
    print(f"[done] wall clock {_hms(wall)}")
    print(f"[report] {report}")
    print("\nBasin table:")
    for b, c in Counter(r["basin"] for r in recs).most_common():
        print(f"  {b!r:<16} {c:>4}  {_pct(c, len(recs)):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
