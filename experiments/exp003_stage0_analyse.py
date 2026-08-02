"""Stage 0 analysis, including the form the source actually used.

Reads `output_exp003/stage0.jsonl` and re-derives every gate from the saved
per-prompt records, so the analysis can be corrected without paying for the run
again.

WHY THIS EXISTS SEPARATELY, AND A CORRECTION TO THE RUNNER'S OWN ANALYSIS

The pre-registration and `exp003_stage0.py` both reduce the grid to a single
number per prompt: the depth-weighted centroid. That is a reduction I made, and
verification of the source afterwards showed it is not what Chao, Bakkum and
Potter did.

They did not compare scalars. Their centre of activity is two numbers per time
bin, their trajectory is the sequence of those over 191 bins, and they compared
whole trajectories by concatenating them into what they call a whole-input-output
vector and taking plain Euclidean distance in that space. No bespoke trajectory
metric, no curve registration; principal components were used for pictures only.
In simulation that vector ran to 22,920 dimensions.

A faithful port would keep the whole trajectory rather than collapsing it. **What
this script computes is not that**, and the label is corrected here: it uses the
twelve-number per-layer profile at settle, which is a summary of the trajectory's
endpoint rather than the trajectory. Persisting the full per-iteration activity for
125 inputs was not done, so the full-trajectory comparison cannot be computed from
this run and is not claimed. What follows is a source-inspired variant, and it
matters for a specific reason the same authors ran into. Bakkum, Chao and Potter
(2008) had to apply a whitening transform to the centre of activity because an
uneven distribution of cells across the array biased it in a fixed direction for
every preparation. The identical bias exists in this substrate and is larger:
deeper blocks systematically write more into the residual stream, so the depth
centroid is dragged the same way for every input, and the between-input variation
that carries the signal is a small residue on top of a large common offset.

This script therefore reports three measurements, and says which is which:

  1. `scalar`   the pre-registered depth centroid. Reported because it was
                registered, whatever it says.
  2. `whitened` the same centroid after removing the population mean and scaling
                by the population spread. Note in advance that this cannot change
                the separation ratio, because that ratio is invariant to any
                affine rescaling of a scalar. It is computed and reported anyway
                so that the invariance is visible rather than assumed, and so no
                later reader thinks whitening was skipped.
  3. `profile`  the twelve-number per-layer mass profile at settle, standardised
                per layer across the population, compared by Euclidean distance.
                This is a source-INSPIRED variant, not the source's method: it keeps
                a vector rather than a scalar and uses their distance measure, but
                over an endpoint summary rather than over the whole trajectory. It
                is not privileged over measurement 1 on grounds of faithfulness,
                because it is not faithful.

Measurement 3 is a departure from the registered protocol and is labelled a
post-registration addition wherever it appears. It is not a replacement for
measurement 1, which stands as registered and is reported first.

Usage:
    .venv/bin/python experiments/exp003_stage0_analyse.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mea_grid import separation_ratio                      # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
STAGE0 = ROOT / "experiments" / "output_exp003" / "stage0.jsonl"
OUT = ROOT / "experiments" / "output_exp003" / "stage0_analysis.json"

GATE_SEPARATION = 1.5
GATE_FAILURE_DIRECTION = 1.2
TOKEN_LABEL_BASELINE = 0.87          # register row C-07, read as a change-to-drift ratio
N_RANDOM_SPLITS = 10
SEED = 20260802


def load(path: Path) -> list[dict]:
    """Every JSON record from a run file, in order."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def euclidean_separation(vectors: dict[str, list[list[float]]]) -> float:
    """Between-group over within-group spread, for vector-valued features.

    The same change-to-drift logic as the scalar version, in the form Chao,
    Bakkum and Potter used it: distance from each group's centre to the grand
    centre, over the mean distance of members to their own group centre. A value
    near 1 means groups are indistinguishable from their own internal scatter.
    """
    def centre(vs):
        """The componentwise mean of a set of vectors."""
        return [statistics.fmean(c) for c in zip(*vs, strict=True)]

    def dist(a, b):
        """Euclidean distance between two vectors."""
        return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5

    centres, within = {}, []
    for name, vs in vectors.items():
        if not vs:
            continue
        centres[name] = centre(vs)
        if len(vs) >= 2:
            within.extend(dist(v, centres[name]) for v in vs)

    if len(centres) < 2 or not within:
        return float("nan")

    grand = centre(list(centres.values()))
    between = statistics.fmean(dist(c, grand) for c in centres.values())
    scatter = statistics.fmean(within)
    return between / scatter if scatter > 0 else float("inf")


def standardise(rows: list[dict], key: str) -> dict[str, list[float]]:
    """Per-dimension z-scores across the whole population.

    This is the analogue of the whitening Bakkum, Chao and Potter applied to
    remove a fixed directional bias. Here the bias is that deeper blocks write
    more, which shifts every input's profile the same way; standardising per layer
    removes that common component so what remains is how each input differs.
    """
    mats = [r[key] for r in rows]
    cols = list(zip(*mats, strict=True))
    means = [statistics.fmean(c) for c in cols]
    sds = [statistics.pstdev(c) or 1.0 for c in cols]
    return {
        r["prompt_id"]: [(v - m) / s for v, m, s in zip(r[key], means, sds, strict=True)]
        for r in rows
    }


def gates(values_by_group: dict, sep_fn, rows: list[dict], label: str) -> dict:
    """Score one measurement against all three registered gates and return the verdicts."""
    basin_ratio = sep_fn(values_by_group["basin"])
    class_ratio = sep_fn(values_by_group["class"])

    # Failure direction: random halves of the largest group must not separate.
    largest = max(values_by_group["basin"].items(), key=lambda kv: len(kv[1]))[0]
    pool = list(values_by_group["basin"][largest])
    rng = random.Random(SEED)
    splits = []
    for _ in range(N_RANDOM_SPLITS):
        p = pool[:]
        rng.shuffle(p)
        h = len(p) // 2
        splits.append(sep_fn({"a": p[:h], "b": p[h:]}))
    n_below = sum(1 for s in splits if s == s and s < GATE_FAILURE_DIRECTION)

    return {
        "measurement": label,
        "basin_separation": basin_ratio,
        "basin_pass": bool(basin_ratio == basin_ratio and basin_ratio > GATE_SEPARATION),
        "class_separation": class_ratio,
        "class_pass": bool(class_ratio == class_ratio and class_ratio > GATE_SEPARATION),
        "random_split_ratios": splits,
        "n_splits_below": n_below,
        "failure_direction_pass": bool(n_below >= 9),
        "largest_group": largest,
        "beats_token_labels": bool(basin_ratio == basin_ratio
                                   and basin_ratio > TOKEN_LABEL_BASELINE),
    }


def main() -> int:
    """Re-derive every gate from the saved records, for all three measurements."""
    recs = load(STAGE0)
    rows = [r for r in recs if r.get("kind") == "prompt"]
    if not rows:
        print("no prompt records in stage0.jsonl; has the run finished?")
        return 1
    print(f"[analysis] {len(rows)} prompts")

    # ---- 1. the registered scalar -----------------------------------------
    scalar_basin, scalar_class = {}, {}
    for r in rows:
        scalar_basin.setdefault(r["basin"], []).append(r["settled_ca_depth"])
        scalar_class.setdefault(r["dyn_class"], []).append(r["settled_ca_depth"])
    g_scalar = gates({"basin": scalar_basin, "class": scalar_class},
                     separation_ratio, rows, "scalar (registered)")

    # ---- 2. whitened scalar, reported to show the invariance ---------------
    vals = [r["settled_ca_depth"] for r in rows]
    mu, sd = statistics.fmean(vals), (statistics.pstdev(vals) or 1.0)
    wb, wc = {}, {}
    for r in rows:
        z = (r["settled_ca_depth"] - mu) / sd
        wb.setdefault(r["basin"], []).append(z)
        wc.setdefault(r["dyn_class"], []).append(z)
    g_white = gates({"basin": wb, "class": wc}, separation_ratio, rows,
                    "whitened scalar")

    # ---- 3. source-inspired variant: the settle profile, Euclidean ---------
    # Checked across every record, not just the first: a mixed schema would
    # otherwise either raise part way through or silently skip the measurement
    # depending on which record happened to be written first.
    presence = ["settled_mass_per_layer" in r for r in rows]
    if any(presence) and not all(presence):
        raise ValueError(
            f"settled_mass_per_layer present in {sum(presence)} of {len(rows)} "
            f"records; the run file has a mixed schema"
        )
    have_profile = all(presence)
    if have_profile:
        widths = {len(r["settled_mass_per_layer"]) for r in rows}
        if len(widths) != 1:
            raise ValueError(f"per-layer profiles have mixed widths: {sorted(widths)}")
    g_profile = None
    if have_profile:
        z = standardise(rows, "settled_mass_per_layer")
        pb, pc = {}, {}
        for r in rows:
            pb.setdefault(r["basin"], []).append(z[r["prompt_id"]])
            pc.setdefault(r["dyn_class"], []).append(z[r["prompt_id"]])
        g_profile = gates({"basin": pb, "class": pc}, euclidean_separation, rows,
                          "settle profile, source-inspired (post-registration)")

    # ---- controls ----------------------------------------------------------
    n_shuf = len(rows[0]["ca_depth_layer_shuffled"])
    layer_ratios = []
    for j in range(n_shuf):
        d = {}
        for r in rows:
            d.setdefault(r["basin"], []).append(r["ca_depth_layer_shuffled"][j])
        layer_ratios.append(separation_ratio(d))
    head_dev = max(abs(h - r["settled_ca_depth"])
                   for r in rows for h in r["ca_depth_head_shuffled"])

    result = {
        "n_prompts": len(rows),
        "basin_counts": {k: len(v) for k, v in scalar_basin.items()},
        "class_counts": {k: len(v) for k, v in scalar_class.items()},
        "scalar": g_scalar,
        "whitened_scalar": g_white,
        "profile": g_profile,
        "control_layer_shuffle_ratios": layer_ratios,
        "control_layer_shuffle_mean": statistics.fmean(layer_ratios),
        "control_layer_shuffle_pass": bool(
            statistics.fmean(layer_ratios) < g_scalar["basin_separation"]),
        "control_head_shuffle_max_deviation": head_dev,
        "control_head_shuffle_pass": bool(head_dev < 1e-9),
        "thresholds": {
            "separation": GATE_SEPARATION,
            "failure_direction": GATE_FAILURE_DIRECTION,
            "token_labels_from_C07": TOKEN_LABEL_BASELINE,
        },
    }
    OUT.write_text(json.dumps(result, indent=1))

    def show(g):
        """Print one measurement's gate results, or note that it was not computed."""
        if g is None:
            print("  (per-layer profile not saved in this run)")
            return
        print(f"  {g['measurement']}")
        print(f"    basins {g['basin_separation']:.4f}  "
              f"{'PASS' if g['basin_pass'] else 'FAIL'}   "
              f"(token labels score {TOKEN_LABEL_BASELINE}, "
              f"beats them: {g['beats_token_labels']})")
        print(f"    class  {g['class_separation']:.4f}  "
              f"{'PASS' if g['class_pass'] else 'FAIL'}")
        print(f"    failure direction {g['n_splits_below']}/10 below "
              f"{GATE_FAILURE_DIRECTION}  "
              f"{'PASS' if g['failure_direction_pass'] else 'FAIL'}")

    print("\n=== STAGE 0 ANALYSIS ===")
    print(f"{len(rows)} prompts  basins {result['basin_counts']}  "
          f"classes {result['class_counts']}\n")
    show(g_scalar)
    show(g_white)
    show(g_profile)
    print(f"\n  control A layer shuffle  mean {result['control_layer_shuffle_mean']:.4f} "
          f"vs true {g_scalar['basin_separation']:.4f}  "
          f"{'PASS' if result['control_layer_shuffle_pass'] else 'FAIL'}")
    print(f"  control B head shuffle   max deviation "
          f"{result['control_head_shuffle_max_deviation']:.2e}  "
          f"{'PASS' if result['control_head_shuffle_pass'] else 'FAIL'}")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
