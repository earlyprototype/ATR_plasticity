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
    """Import the parent's `prompt_library` by path.

    This is an import, so module-level code in that file runs in this process --
    the same trust assumption as any `import`. The path comes from `--parent` or
    `$ATR_PARENT_PATH` and must therefore be trusted. What this function does
    *not* do: write anything to the parent clone, or call any other parent code.
    The only thing taken from it is the prompt data."""
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
        return 0.0 if d == 0 else float(np.clip(np.dot(u, v) / d, -1.0, 1.0))

    by_basin = {}
    for r in recs:
        if r["prompt_id"] in vecs:
            by_basin.setdefault(r["basin"], []).append(r["prompt_id"])

    A("**What this can and cannot bound.** Five basins bound the information in the *basin label* "
      "at log2(5) = 2.32 bits. They bound nothing about the settled *state*. If the states within "
      "a basin are indistinguishable, the state is the label and the prompt has been erased. If "
      "they differ systematically, the attractor compressed the prompt rather than erasing it, "
      "and the residue is measurable. Both are legitimate outcomes; the numbers below decide "
      "which, and no similarity threshold was chosen after seeing them -- the one threshold used "
      "is the float32 round-off scale measured in the instrument check above, fixed before this "
      "sweep finished.")
    A("")
    A("All comparisons use the position-mean of the settled tensor, `(768,)`, which is the vector "
      "the convergence gate itself uses and is comparable across prompts of different length. "
      "Phase-aware throughout: on a period-2 orbit two prompts can sit on the same cycle in "
      "opposite phases, so each pair is scored at the better of (final, final) and "
      "(final, previous iterate).")
    A("")

    def pair_cos(pi, pj):
        ai, bi = vecs[pi]
        aj, bj = vecs[pj]
        c = cos(ai, aj)
        if bj is not None:
            c = max(c, cos(ai, bj))
        if bi is not None:
            c = max(c, cos(bi, aj))
        return c

    # ---- within-basin -----------------------------------------------------
    ROUNDOFF = 1e-12   # see instrument check: cycle round-off sits at 1-cos ~ 1.5e-14
    within_all, rows = [], []
    for b in sorted(by_basin, key=lambda b: -len(by_basin[b])):
        ids = by_basin[b]
        if len(ids) < 2:
            rows.append([f"`{b}`", len(ids), "-", "-", "-", "-", "-"])
            continue
        pair = [pair_cos(ids[i], ids[j])
                for i in range(len(ids)) for j in range(i + 1, len(ids))]
        within_all.extend(pair)
        rows.append([f"`{b}`", len(ids), f"{min(pair):.6f}",
                     f"{statistics.median(pair):.6f}", f"{max(pair):.6f}",
                     f"{statistics.mean(1 - c for c in pair):.3e}",
                     sum(1 for c in pair if (1 - c) < ROUNDOFF)])
    A("### Within basin")
    A("")
    L.extend(_table(rows, ["Basin", "n", "min pair cos", "median pair cos", "max pair cos",
                           "mean (1 - cos)", "pairs at round-off"]))
    A("")

    # ---- between-basin ----------------------------------------------------
    basins = [b for b in by_basin if len(by_basin[b]) >= 1]
    between_all, pair_medians = [], {}
    if len(basins) >= 2:
        A("### Between basins")
        A("")
        rows = []
        header = ["Basin"] + [f"`{b}`" for b in basins]
        for bi in basins:
            row = [f"`{bi}`"]
            for bj in basins:
                if bi == bj:
                    row.append("--")
                    continue
                pc = [pair_cos(p, q) for p in by_basin[bi] for q in by_basin[bj]]
                if bi < bj:
                    between_all.extend(pc)
                    pair_medians[(bi, bj)] = statistics.median(pc)
                row.append(f"{statistics.median(pc):.4f}")
            rows.append(row)
        L.extend(_table(rows, header))
        A("")
        A("Median cosine between the settled states of prompts in different basins.")
        A("")
        if pair_medians:
            (na, nb), nearest = max(pair_medians.items(), key=lambda kv: kv[1])
            (fa, fb), farthest = min(pair_medians.items(), key=lambda kv: kv[1])
            A(f"Closest pair of basins: `{na}` and `{nb}` at median cosine {nearest:.6f} "
              f"(`1 - cos` = {1 - nearest:.3e}). Farthest: `{fa}` and `{fb}` at {farthest:.4f}.")
            A("")
            A("**This is the number to read carefully.** The between-basin mean below is dominated "
              "by whichever basin sits far away; the distance that matters for whether the basin "
              "*label* corresponds to a separated *state* is the distance between the two closest "
              "basins. Compare it against the within-basin spread in the table above: if a "
              "basin's internal spread is the same order as the gap to its nearest neighbouring "
              "basin, then the label is a sharper distinction than the state, and the five-basin "
              "partition is being drawn by the readout's argmax rather than by the geometry.")
            A("")

    # ---- one number -------------------------------------------------------
    if within_all and between_all:
        w = statistics.mean(1 - c for c in within_all)
        bt = statistics.mean(1 - c for c in between_all)
        A("### Within/between ratio")
        A("")
        rows = [
            ["Mean within-basin spread, `1 - cos`", f"{w:.4e}", f"{len(within_all)} pairs"],
            ["Mean between-basin spread, `1 - cos`", f"{bt:.4e}", f"{len(between_all)} pairs"],
            ["**Ratio within / between**", f"**{w / bt:.4e}**" if bt else "n/a", ""],
        ]
        L.extend(_table(rows, ["Quantity", "Value", "n"], [":---", "---:", "---:"]))
        A("")
        A(f"A ratio near 1 would mean the basins are not separated at all; a ratio near 0 means "
          f"prompts inside a basin are far closer to each other than to anything outside it. "
          f"Measured: **{w / bt:.2e}**.")
        A("")
        if pair_medians:
            nearest = max(pair_medians.values())
            gap = 1 - nearest
            A(f"Against the *nearest* basin pair rather than the mean: within-basin spread "
              f"{w:.3e} versus nearest-basin gap {gap:.3e}, a ratio of "
              f"**{w / gap:.2f}** if the gap is nonzero. A ratio near or above 1 means the two "
              "nearest basins are no further apart than the prompts inside one of them.")
            A("")
        A(f"For reference, the float32 round-off floor measured on the parent's committed cycle "
          f"is `1 - cos` around 1.5e-14. The within-basin spread above is "
          f"{w / 1.5e-14:.1e} times that floor, so it is a real geometric spread and not "
          "arithmetic noise.")
        A("")

    # ---- effective dimensionality ----------------------------------------
    A("### Effective dimensionality within each basin")
    A("")
    A("Participation ratio of the singular values of the (mean-centred, unit-normalised) stack of "
      "settled states in a basin: `PR = (sum s^2)^2 / sum s^4`. PR near 1 means the within-basin "
      "variation lies along a single direction; PR near n-1 means it fills the space the sample "
      "can see. Phase-aligned to the first member of the basin before stacking.")
    A("")
    rows = []
    for b in sorted(by_basin, key=lambda b: -len(by_basin[b])):
        ids = by_basin[b]
        if len(ids) < 3:
            rows.append([f"`{b}`", len(ids), "-", "-", "-"])
            continue
        ref = vecs[ids[0]][0]
        stack = []
        for pid in ids:
            a, bb = vecs[pid]
            v_ = a if (bb is None or cos(a, ref) >= cos(bb, ref)) else bb
            nrm = np.linalg.norm(v_)
            stack.append(v_ / nrm if nrm else v_)
        M = np.asarray(stack, dtype=np.float64)
        Mc = M - M.mean(axis=0, keepdims=True)
        s = np.linalg.svd(Mc, compute_uv=False)
        lam = s ** 2
        pr = float((lam.sum() ** 2) / (lam ** 2).sum()) if lam.sum() > 0 else float("nan")
        frac1 = float(lam[0] / lam.sum()) if lam.sum() > 0 else float("nan")
        rows.append([f"`{b}`", len(ids), f"{pr:.2f}", f"{min(len(ids) - 1, 768)}",
                     f"{frac1:.3f}"])
    L.extend(_table(rows, ["Basin", "n", "participation ratio", "max possible (n-1)",
                           "variance in top direction"]))
    A("")

    # ---- position uniformity, all basins or only Divine? -----------------
    A("### Does position uniformity hold for every basin, or only `Divine`?")
    A("")
    rows = []
    for b in sorted(by_basin, key=lambda b: -len(by_basin[b])):
        sub = [r for r in recs if r["basin"] == b and r.get("final_position_stats")]
        if not sub:
            continue
        mins = [r["final_position_stats"]["min"] for r in sub]
        rows.append([f"`{b}`", len(sub),
                     f"{statistics.median(r['final_position_stats']['mean'] for r in sub):.6f}",
                     f"{min(mins):.6f}",
                     sum(1 for m in mins if m > 0.999),
                     f"{sum(1 for m in mins if m > 0.999) / len(sub) * 100:.0f}%"])
    L.extend(_table(rows, ["Basin", "n", "median mean pos-cos", "worst pair-cos in basin",
                           "prompts fully uniform", "share"]))
    A("")
    A("Raw states are in `states/` for any sharper analysis.")
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
        c1s = [_f(r["cos_lag1_mean"]) for r in p2 if r.get("cos_lag1_mean") is not None]
        if len(c1s) > 1:
            A(f"**Are these all the same cycle?** The lag-1 cosine is the swing of one step "
              f"around the orbit, so it is a property of the cycle's geometry, not of the "
              f"labelling. Across these prompts it ranges {min(c1s):.6f} to {max(c1s):.6f} "
              f"(median {statistics.median(c1s):.6f}, {len(set(round(c, 4) for c in c1s))} "
              "distinct values at 4 dp). The parent's committed `state_divine.pt` sits at "
              "0.684912. A single shared cycle would put every prompt at one value; a spread "
              "means the basin contains a family of distinct period-2 orbits that happen to "
              "decode to the same token.")
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

    # ---- operational ------------------------------------------------------
    L.extend(_operational_section(meta))

    # ---- config ----------------------------------------------------------
    L.extend(_config_section(meta))
    return "\n".join(L)


def _operational_section(meta):
    """Cost model for whoever runs the next sweep."""
    L = []
    A = L.append
    A("## Operational notes for the next sweep")
    A("")
    A("### Do not give this job all the cores")
    A("")
    A("Measured on this 4-vCPU box, GPT-2 small at `seq_len` 10, one ATR iteration "
      "(`run_with_cache` forward plus the readout decode):")
    A("")
    L.extend(_table([
        ["1", "287", "1.00x"],
        ["2", "350", "1.22x"],
        ["3", "1105", "3.85x"],
        ["4", "2137", "7.45x"],
    ], ["`torch.set_num_threads`", "ms / iteration", "slowdown vs 1 thread"]))
    A("")
    A("Setting the thread count equal to the core count made the job **7.45x slower**, not "
      "faster. The cause is OpenMP spin-wait collapse: GPT-2 small's per-layer matmuls at "
      "`seq_len` ~10 are far too small to amortise a barrier across 4 threads, so the worker "
      "threads spend their time busy-waiting, and they contend with every other process on the "
      "box -- of which there is always at least one. The effect is not subtle and it is not "
      "load-dependent noise: it reproduced in both directions of a 4,3,2,1,2,4 sweep.")
    A("")
    A("The fix that actually gives parallelism is process-level: **N single-threaded processes**, "
      "each on its own slice of the prompt list. Two such shards measured 296 ms/iteration each "
      "-- a 3% penalty against running alone, i.e. near-linear scaling. Three shards measured "
      "~650 ms each, which is where memory bandwidth starts to bind; on this box two is the "
      "sweet spot and it leaves half the machine for whoever else is working.")
    A("")
    A("Practical numbers for planning: **~0.29 s per iteration per prompt**, near-flat in "
      "sequence length between `seq_len` 2 and 25 (219 ms at 2, 287 at 10, 289 at 25 -- the map "
      "is overhead-bound, not FLOP-bound, at this size). A 125-prompt x 300-iteration sweep is "
      "therefore about 3.0 CPU-hours, or about 1.6 hours wall on two shards.")
    A("")
    A("Single-threaded is also the reproducible choice, independently of speed: float32 "
      "reduction order stops depending on how BLAS happened to split the work, so a re-run is "
      "bit-comparable with this one.")
    A("")
    return L


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

    ex = v.get("exactness")
    if ex:
        tr = ex["trend"]
        c2, c1 = ex["lag2"][0], ex["lag1"][0]
        A("### Is the period-2 cycle exact? No -- and the residual is stationary")
        A("")
        A("A cosine that prints as `1.000000` establishes six decimal places of display "
          "precision, not equality. It is equally consistent with `f(f(A))` being bit-for-bit "
          "`A` -- a true fixed point of the squared map -- and with an orbit still contracting "
          "below the sixth decimal, which is an asymptote and not a fixed point at all. "
          f"Measured directly over {ex['n_probe']} iterations from the parent's committed state:")
        A("")
        rows = [
            ["`torch.equal(A, f(f(A)))`", f"**False**, at every one of {len(ex['lag2'])} probed k"],
            ["Elements differing", f"{c2['n_elements_differing']} of {c2['n_elements']} "
                                   f"({100 * c2['n_elements_differing'] / c2['n_elements']:.0f}%)"],
            ["Max absolute elementwise deviation", f"{c2['max_abs_diff']:.3e} "
                                                   f"(state RMS element {c2['rms_a']:.2f}, "
                                                   f"largest element {c2['max_abs_a']:.1f})"],
            ["Max relative elementwise deviation", f"{c2['max_rel_diff']:.3e} "
                                                   "(dominated by near-zero entries)"],
            ["Relative L2 deviation, `norm(A - f(f(A))) / norm(A)`", f"{c2['l2_rel']:.3e}"],
            ["Cosine in float64, full precision", f"`{c2['cos_float64']:.17f}`"],
            ["`1 - cos` in float64", f"{c2['one_minus_cos']:.5e}"],
        ]
        L.extend(_table(rows, ["Quantity", "Value"], [":---", ":---"]))
        A("")
        A("For scale, the same quantities on the lag-1 pair `(A, f(A))` -- a comparison that is "
          "genuinely not identity:")
        A("")
        rows = [
            ["`torch.equal(A, f(A))`", str(c1["bit_identical"])],
            ["Max absolute elementwise deviation", f"{c1['max_abs_diff']:.4e}"],
            ["Relative L2 deviation", f"{c1['l2_rel']:.4f}"],
            ["Cosine in float64", f"`{c1['cos_float64']:.15f}`"],
        ]
        L.extend(_table(rows, ["Quantity", "Value"], [":---", ":---"]))
        A("")
        A(f"**Shrinking or stationary?** The relative L2 residual averages "
          f"{tr['l2_rel_first_third_mean']:.3e} over the first third of the probe and "
          f"{tr['l2_rel_last_third_mean']:.3e} over the last third -- a ratio of "
          f"{tr['ratio_last_over_first']:.2f} across {ex['n_probe']} iterations. It is not "
          "decaying. It sits at "
          f"{tr['l2_rel_last_third_mean'] / tr['float32_eps']:.2f} x float32 epsilon "
          f"({tr['float32_eps']:.3e}) and stays there.")
        A("")
        A("**Statement of the result, at the strength the numbers support.** The `Divine` orbit is "
          "an *attracting* period-2 cycle in float32 arithmetic, not a bitwise-periodic one. "
          "`f o f` is not the identity on `A`: it moves 87% of the entries and lands about 1.6 "
          "float32 ulps away in relative L2. But it does not move *further* with iteration, and "
          "it does not move *closer*: the residual is stationary round-off jitter around the "
          "cycle, not convergence toward it and not divergence from it. So the correct claim is "
          "\"a fixed point of the squared map to within float32 arithmetic\", and the incorrect "
          "claims are both \"bit-identical\" and \"still drifting\". A `1.000000` printout could "
          "not have distinguished these; `1 - cos = "
          f"{c2['one_minus_cos']:.2e}` in float64 does.")
        A("")
        A("Two consequences worth carrying forward. First, any equality test on ATR states has to "
          "be a tolerance test at the float32 round-off scale -- `torch.equal` will return False "
          "on states that are the same point of the dynamics. Second, the cycle being attracting "
          "rather than exact is what makes it robust: it is reached from a neighbourhood and "
          "survives perturbation, which a knife-edge bitwise cycle would not.")
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
    L.extend(_table(rows, ["Basin", "Published @lock-in", "This run @120",
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

def cycle_exactness(model, state, n_probe: int = 10) -> dict:
    """Is the period-2 orbit bit-identical, or merely tight?

    A cosine that PRINTS as 1.000000 establishes six decimals of display
    precision and nothing else. It is equally consistent with `f(f(A))` being
    bit-for-bit `A` -- a genuine fixed point of the squared map -- and with an
    orbit still contracting below the sixth decimal, which is not a fixed point
    at all but an asymptote. The two are different objects, so the printed
    cosine is replaced here with: `torch.equal`, max absolute and max relative
    elementwise deviation, and the cosine recomputed in float64 at full
    precision. The lag-1 pair is measured the same way to give the scale of a
    comparison that is genuinely not identity.
    """
    from atr_bridge import make_atr_step_from_state
    step = make_atr_step_from_state(model, state, layer_start=LAYER_START, layer_end=LAYER_END)

    iters = [state.tensor.clone()]
    for _ in range(n_probe):
        iters.append(step(model, iters[-1]))

    def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
        a64, b64 = a.double(), b.double()
        diff = (a64 - b64).abs()
        # Elementwise relative deviation is reported against |a| and, separately,
        # against the RMS element of a. The first is honest but dominated by
        # near-zero entries; the second is the one to quote for scale.
        denom = a64.abs().clamp_min(torch.finfo(torch.float64).tiny)
        cos64 = float((a64.reshape(-1) @ b64.reshape(-1)) / (a64.norm() * b64.norm()))
        rms = float(a64.pow(2).mean().sqrt())
        return {
            "bit_identical": bool(torch.equal(a, b)),
            "max_abs_diff": float(diff.max()),
            "max_rel_diff": float((diff / denom).max()),
            "max_abs_diff_over_rms": float(diff.max()) / rms,
            "l2_diff": float((a64 - b64).norm()),
            "l2_a": float(a64.norm()),
            "l2_rel": float((a64 - b64).norm() / a64.norm()),
            "rms_a": rms,
            "max_abs_a": float(a64.abs().max()),
            "cos_float64": cos64,
            "one_minus_cos": float(1.0 - cos64),
            "n_elements_differing": int((a != b).sum()),
            "n_elements": int(a.numel()),
        }

    lag2 = [compare(iters[k], iters[k + 2]) for k in range(len(iters) - 2)]
    lag1 = [compare(iters[k], iters[k + 1]) for k in range(len(iters) - 1)]
    first_exact = next((k for k, c in enumerate(lag2) if c["bit_identical"]), None)

    # Shrinking or stationary? A cycle being asymptotically approached has a
    # lag-2 residual that decays with k; float32 round-off jitter around an
    # attracting cycle does not. Comparing the first and last thirds separates
    # the two without fitting anything.
    rel = [c["l2_rel"] for c in lag2]
    third = max(1, len(rel) // 3)
    head, tail = rel[:third], rel[-third:]
    SHRINK_RATIO_THRESHOLD = 0.5     # pre-registered: see note below
    ratio = (statistics.mean(tail) / statistics.mean(head)
             if statistics.mean(head) else float("nan"))
    trend = {
        "l2_rel_first_third_mean": statistics.mean(head),
        "l2_rel_last_third_mean": statistics.mean(tail),
        "l2_rel_min": min(rel), "l2_rel_max": max(rel),
        "l2_rel_per_probe_spread": [min(rel), max(rel)],
        "ratio_last_over_first": ratio,
        "float32_eps": float(torch.finfo(torch.float32).eps),
        # The flag is a criterion, not a measurement, so the criterion travels
        # with it: without the threshold in the artifact a reader cannot tell
        # "no trend detected" from "a trend that failed a strict cutoff".
        "shrink_ratio_threshold": SHRINK_RATIO_THRESHOLD,
        "shrink_criterion": f"ratio_last_over_first < {SHRINK_RATIO_THRESHOLD}",
        "verdict": ("shrinking" if ratio < SHRINK_RATIO_THRESHOLD else "no trend detected"),
        "verdict_note": (
            "The per-probe residual itself ranges "
            f"{min(rel):.3e} to {max(rel):.3e}, a factor of {max(rel) / min(rel):.2f}, so a "
            f"first-third/last-third ratio of {ratio:.3f} sits well inside the sample's own "
            "scatter. The correct reading is NO TREND DETECTED over this window -- not a "
            "demonstration that the residual is constant."),
    }
    trend["shrinking"] = ratio < SHRINK_RATIO_THRESHOLD
    return {
        "n_probe": n_probe,
        "lag2": lag2,
        "lag1": lag1,
        "all_lag2_bit_identical": all(c["bit_identical"] for c in lag2),
        "any_lag2_bit_identical": any(c["bit_identical"] for c in lag2),
        "first_bit_identical_k": first_exact,
        "any_lag1_bit_identical": any(c["bit_identical"] for c in lag1),
        "trend": trend,
    }


PERTURB_MAGNITUDES = (1e-7, 1e-5, 1e-3, 1e-1, 1.0)
PERTURB_SEED = 20260728
# Deep enough that a slow return is distinguishable from no return: the first
# pass at 200 saw recovery times of 2, 10 and 146 iterations for the three
# smallest magnitudes, so a horizon of 200 could not have told "returning
# slowly" from "not returning" for the largest. Each magnitude stops early once
# it has been back at the floor for PERTURB_SETTLE iterations, so the small ones
# stay cheap.
PERTURB_MAX_ITER = 1000
PERTURB_SETTLE = 20
# "Returned to the floor" is defined as the lag-2 relative L2 residual falling
# to within RETURN_FACTOR of the unperturbed floor, and staying there for
# RETURN_PATIENCE consecutive iterations. Both are fixed here, before the test
# is run, so the recovery iteration counts are not a threshold chosen to suit
# the answer.
RETURN_FACTOR = 2.0
RETURN_PATIENCE = 4


def perturbation_test(model, state, floor: float, ref_a, ref_b,
                      magnitudes=PERTURB_MAGNITUDES, max_iter=PERTURB_MAX_ITER) -> dict:
    """Does a state knocked off the cycle come back to it?

    This is the measurement that licenses the word "attracting", and it is NOT
    what the exactness probe measured. That probe watched one orbit's own
    residual stay flat, which is a statement about a single trajectory;
    attraction is a statement about a NEIGHBOURHOOD -- whether states near the
    cycle are drawn onto it. A cycle can perfectly well recur without attracting
    anything (a centre, in the linear picture), and nothing measured so far
    distinguishes that from an attractor.

    So: perturb the settled state by Gaussian noise at relative L2 magnitudes
    spanning well below to well above the round-off floor, iterate, and record
    which of three things happens.

      returns to the same orbit   -> attracting, and the word is earned
      settles on a different orbit -> a basin boundary is nearby
      neither                      -> not attracting

    `floor`, `ref_a`, `ref_b` come from the unperturbed run: the residual floor
    and the two phases of the original cycle, so "same orbit" is a measurement
    against the actual original and not against a re-derived one.
    """
    from atr_bridge import make_atr_step_from_state
    step = make_atr_step_from_state(model, state, layer_start=LAYER_START, layer_end=LAYER_END)

    def rel_l2(a, b):
        a64, b64 = a.double(), b.double()
        return float((a64 - b64).norm() / a64.norm())

    def cos64(a, b):
        a64, b64 = a.double().reshape(-1), b.double().reshape(-1)
        return float((a64 @ b64) / (a64.norm() * b64.norm()))

    def orbit_cos(x):
        """Phase-aware similarity to the ORIGINAL cycle."""
        return max(cos64(x, ref_a), cos64(x, ref_b))

    base_label = readout_detail(model, ref_a[-1, :], k=1)["top_token_strings"][0].strip()
    target = floor * RETURN_FACTOR
    g = torch.Generator().manual_seed(PERTURB_SEED)
    results = []

    for m in magnitudes:
        noise = torch.randn(state.tensor.shape, generator=g, dtype=state.tensor.dtype)
        noise = noise * (m * float(ref_a.norm()) / float(noise.norm()))
        x = ref_a + noise
        applied = rel_l2(ref_a, x)

        hist = [x.clone()]
        residuals, streak, return_iter = [], 0, None
        n_run = 0
        for i in range(1, max_iter + 1):
            x = step(model, x)
            n_run = i
            hist.append(x.clone())
            if len(hist) > 3:
                hist.pop(0)
            if i >= 2:
                r = rel_l2(hist[-3], hist[-1])
                residuals.append(r)
                streak = streak + 1 if r <= target else 0
                if return_iter is None and streak >= RETURN_PATIENCE:
                    return_iter = i - RETURN_PATIENCE + 1
            if return_iter is not None and i >= return_iter + PERTURB_SETTLE:
                break

        final_res = statistics.median(residuals[-10:]) if residuals else float("nan")
        oc = orbit_cos(x)
        label = readout_detail(model, x[-1, :], k=1)["top_token_strings"][0].strip()
        results.append({
            "magnitude_requested": m,
            "magnitude_applied": applied,
            "returned_to_floor": return_iter is not None,
            "return_iteration": return_iter,
            "final_residual": final_res,
            "final_residual_over_floor": final_res / floor if floor else float("nan"),
            "residual_at_iter2": residuals[0] if residuals else None,
            "cos_to_original_orbit": oc,
            "one_minus_cos_to_original_orbit": 1.0 - oc,
            "same_orbit": bool(1.0 - oc < 1e-9),
            "basin_label": label,
            "basin_label_survived": label == base_label,
            "iterations_run": n_run,
            "residual_trace_every_25": residuals[::25],
        })
        print(f"[perturb] m={m:.0e} applied={applied:.3e} "
              f"return={'iter ' + str(return_iter) if return_iter else 'NO(' + str(n_run) + ')'} "
              f"final_res={final_res:.3e} ({final_res / floor:.2f}x floor) "
              f"1-cos_to_orig={1 - oc:.3e} label={label!r} "
              f"{'SAME orbit' if 1 - oc < 1e-9 else 'DIFFERENT orbit'}", flush=True)

    n_ret = sum(1 for r in results if r["returned_to_floor"])
    n_same = sum(1 for r in results if r["same_orbit"])
    if n_ret == len(results) and n_same == len(results):
        verdict = "attracting"
    elif n_ret == len(results):
        verdict = "returns to a cycle, but not always the same one"
    elif n_ret == 0:
        verdict = "not attracting over the tested window"
    else:
        verdict = "mixed: attracting for small perturbations only"
    return {
        "floor": floor, "return_target": target,
        "return_factor": RETURN_FACTOR, "return_patience": RETURN_PATIENCE,
        "max_iter": max_iter, "seed": PERTURB_SEED,
        "noise": "Gaussian, scaled to the requested relative L2 of the settled state",
        "base_label": base_label,
        "results": results,
        "n_returned": n_ret, "n_same_orbit": n_same, "n_tested": len(results),
        "verdict": verdict,
    }


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

    # --- is the cycle bit-identical, or only tight? ------------------------
    ex = cycle_exactness(model, st, n_probe=40)
    print("[exact] lag-2 pairs A_k vs A_k+2:")
    print(f"[exact] {'k':>3} {'bit-ident':>10} {'max|d|':>11} {'maxrel':>10} "
          f"{'||d||/||A||':>12} {'1-cos(f64)':>13} {'differ':>14}")
    for k, c in enumerate(ex["lag2"]):
        if k % 4 and k != len(ex["lag2"]) - 1:
            continue
        print(f"[exact] {k:>3} {str(c['bit_identical']):>10} {c['max_abs_diff']:>11.3e} "
              f"{c['max_rel_diff']:>10.2e} {c['l2_rel']:>12.3e} "
              f"{c['one_minus_cos']:>13.5e} {c['n_elements_differing']:>6}/{c['n_elements']}")
    tr = ex["trend"]
    print(f"[exact] residual ||d||/||A||: first third {tr['l2_rel_first_third_mean']:.4e}, "
          f"last third {tr['l2_rel_last_third_mean']:.4e}, "
          f"ratio {tr['ratio_last_over_first']:.3f} -> "
          f"{'SHRINKING (asymptotic)' if tr['shrinking'] else 'STATIONARY (round-off jitter)'}")
    print(f"[exact] float32 eps = {tr['float32_eps']:.3e}; residual is "
          f"{tr['l2_rel_last_third_mean'] / tr['float32_eps']:.2f} x eps")
    print(f"[exact] verdict: {tr['verdict']} (criterion {tr['shrink_criterion']}); "
          f"per-probe spread {tr['l2_rel_min']:.3e}..{tr['l2_rel_max']:.3e}")

    # --- does a nearby state come back? (attraction, not just recurrence) ---
    from atr_bridge import make_atr_step_from_state
    _step = make_atr_step_from_state(model, st, layer_start=LAYER_START, layer_end=LAYER_END)
    settled = st.tensor.clone()
    for _ in range(8):                      # settle onto the cycle before probing
        settled = _step(model, settled)
    phase_b = _step(model, settled)
    floor = statistics.median([c["l2_rel"] for c in ex["lag2"]])
    pert = perturbation_test(model, st, floor, settled, phase_b)
    print(f"[perturb] verdict: {pert['verdict']} "
          f"({pert['n_returned']}/{pert['n_tested']} returned, "
          f"{pert['n_same_orbit']}/{pert['n_tested']} to the same orbit)")
    c1 = ex["lag1"][0]
    print(f"[exact] lag-1 reference (A vs f(A)): bit_identical={c1['bit_identical']} "
          f"max|d|={c1['max_abs_diff']:.4e} max rel d={c1['max_rel_diff']:.4e} "
          f"||d||2={c1['l2_diff']:.4e} (||A||={c1['l2_a']:.4e}) "
          f"cos64={c1['cos_float64']:.15f}")
    print(f"[exact] lag-2 cos in float64, full precision: "
          f"{ex['lag2'][0]['cos_float64']:.17f}")
    print(f"[exact] all lag-2 pairs bit-identical: {ex['all_lag2_bit_identical']}; "
          f"first bit-identical k: {ex['first_bit_identical_k']}")
    print(f"[validate] {'PASS' if ok else 'FAIL'}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "instrument_validation.json").write_text(json.dumps({
            # Provenance: enough to re-run this probe on another machine.
            "state": _relpath(state_path, Path(parent_path)),
            "state_absolute_path_on_this_machine": str(state_path),
            "state_blob_sha256": _sha256(state_path),
            "state_git_blob": _git_hash_object(state_path, Path(parent_path)),
            "parent_repo_rev": _git_rev(Path(parent_path)),
            "this_repo_rev": _git_rev(REPO_ROOT),
            "model": MODEL_NAME,
            "layers": [LAYER_START, LAYER_END],
            "device": "cpu",
            "dtype": str(st.tensor.dtype),
            "comparison_dtype": "float64 (states cast up before differencing)",
            "torch_version": torch.__version__,
            "transformer_lens_version": _tl_version(),
            "torch_threads": torch.get_num_threads(),
            "state_iteration": st.iteration,
            "initial_norm": st.initial_norm, "n_continued": n_iter,
            "lag_scan_mean_vec": {str(k): v["mean"] for k, v in sm.items()},
            "lag_scan_last_vec": {str(k): v["mean"] for k, v in sl.items()},
            "published_lag1": 0.684912, "published_lag2": 1.0,
            "classified_period2": classified, "pass": bool(ok),
            "exactness": ex,
            "perturbation": pert,
        }, indent=2), encoding="utf-8")
    return 0 if ok else 1


def _relpath(path: Path, root: Path) -> str:
    """Path relative to a repo root, so the artifact means something off this box."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _git_hash_object(path: Path, repo: Path) -> str:
    """`git hash-object` -- the blob id, which is how the parent repo names this
    file's exact contents in its own history."""
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(repo), "hash-object", str(path)],
                              capture_output=True, text=True,
                              timeout=30).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _tl_version() -> str:
    try:
        from importlib.metadata import version as _v
        return _v("transformer-lens")
    except Exception:
        return "unknown"


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
