"""
EXP-004 supporting measurements: every number the spec or the reading note quotes
that was previously "measured in review" and cited no artifact, re-measured here
and written to one JSON file so each can be cited.

Run from the repository root:

    .venv/bin/python experiments/exp004_measurements.py experiments/output_exp004/measurements_v1.json

Sections of the output, each a top-level key:

- ``screening``: frozen-model screening of candidate fact contexts (bound token's
  log probability with the context against without) and of format contexts
  (teacher-forced log probability of the whole bound answer).
- ``binding_vs_priming``: for three fact contexts, the own-context Hebbian write's
  effect on the bound token on the entity query and on two control queries that
  never name the entity, against swapped-context writes.
- ``sigma1_share``: how much of each context write's squared Frobenius norm sits in
  its largest singular direction.
- ``drift_by_site``: realised drift of one fold at fixed step sizes across the site
  kinds the spec names, for ``hebb`` and ``oja``, with the ``clipped`` flag.
- ``linearity``: whether one apply is linear in eta, by the ratio of drifts and the
  cosine between deltas at two step sizes.
- ``c0_gate``: the context pass with hooks installed at eta 0, mode off, against the
  pass without hooks, compared at every cached activation and the logits.
- ``bos_share``: the share of each write carried by the beginning-of-sequence position.
- ``position_control``: the final-position KL of the query-alone baseline, of a
  position-shifted run, and of a length-matched neutral filler run.
- ``loop_writes``: the rule's accumulator over N loop iterations from two topic
  contexts, compared with each other and with the context writes.
- ``timings``: seconds per fold and per query pass on this machine.

Nothing here enters the claim register. Nothing here is a stage of EXP-004.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atr_bridge import initial_state, make_atr_step  # noqa: E402
from multi_site import MultiSitePlasticity, SiteSpec  # noqa: E402
from plasticity import OjaPlasticity  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

torch.manual_seed(0)
model = HookedTransformer.from_pretrained("gpt2", device="cpu")
model.eval()
tok = model.tokenizer
BOS = tok.bos_token_id
SITE = "blocks.6.mlp"
R = {"model": "gpt2", "site": SITE, "torch": torch.__version__, "threads": torch.get_num_threads()}
T_START = time.time()


def ids(text):
    return tok.encode(text)[1:]


def seq(token_ids):
    return torch.tensor([[BOS] + list(token_ids)])


@torch.no_grad()
def logits(tokens):
    return model(tokens)[0].double()


def lp_final(tokens, token_id):
    return torch.log_softmax(logits(tokens)[-1], -1)[token_id].item()


def full_answer_lp(prefix_ids, ans_ids):
    """Teacher-forced log probability of the whole answer after the prefix: one
    forward pass on prefix + answer[:-1], summing log p of each answer token at
    the position that predicts it."""
    lg = logits(seq(list(prefix_ids) + list(ans_ids[:-1])))
    n = len(ans_ids)
    out = []
    for i, a in enumerate(ans_ids):
        out.append(torch.log_softmax(lg[len(prefix_ids) + i], -1)[a].item())
    return sum(out), out


def split(C, Q):
    c, cq = ids(C), ids(C + Q)
    assert cq[: len(c)] == c
    return c, cq[len(c):]


def fold(site, mode, eta, ctx_tokens):
    """One context pass with the rule observing, one apply. Returns the delta
    (stacked over stripes for a multi-site set), the report, and the batch
    count seen before apply."""
    if isinstance(site, list):
        d = MultiSitePlasticity(model, [SiteSpec(s, mode=mode, eta=eta, max_delta_frac=1.0) for s in site])
    else:
        d = OjaPlasticity(model, site, eta=eta, mode=mode, max_delta_frac=1.0)
    d.install()
    with torch.no_grad():
        model(ctx_tokens)
    d.remove()
    rep = d.apply()
    if isinstance(site, list):
        delta = torch.cat([p.delta.clone() for p in d], 0)
        w0n = torch.sqrt(sum(p.W0.double().norm() ** 2 for p in d)).item()
        rep = {"delta_frac": delta.double().norm().item() / w0n,
               "clipped": any(p.clipped for p in d), "nonfinite": any(p.nonfinite for p in d),
               "per_stripe_delta_frac": [p.report()["delta_frac"] for p in d]}
    else:
        delta = d.delta.clone()
        rep = {k: rep[k] for k in ("delta_frac", "clipped", "nonfinite")}
    d.revert()
    return delta, rep


probe = OjaPlasticity(model, SITE, eta=0.0, mode="off")
site6, W0 = probe._site, probe.W0.clone()


def with_delta(delta, fn):
    site6.write(W0 + delta)
    try:
        return fn()
    finally:
        site6.write(W0)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().double()[None], b.flatten().double()[None]).item()


# ---------------------------------------------------------------- screening
FACT_CANDIDATES = {
    "Veltoria/Oslo": ("The capital of Veltoria is Oslo. Oslo lies on a fjord and is known for its museums.", " The capital of Veltoria is", " Oslo"),
    "Brannock/Paris": ("The capital of Brannock is Paris. Paris is famous for its river and its cafes.", " The capital of Brannock is", " Paris"),
    "Morvane/Tokyo": ("The capital of Morvane is Tokyo. Tokyo is a large city with many trains.", " The capital of Morvane is", " Tokyo"),
    "Kelvarn/Rome": ("The capital of Kelvarn is Rome. Rome has stood for more than two thousand years.", " The capital of Kelvarn is", " Rome"),
    "Ostrelia/Cairo": ("The capital of Ostrelia is Cairo. Cairo sits beside a wide river in the desert.", " The capital of Ostrelia is", " Cairo"),
    "Dranmoor/Lima": ("The capital of Dranmoor is Lima. Lima is a coastal city with a mild climate.", " The capital of Dranmoor is", " Lima"),
    "Quillane/Berlin": ("The capital of Quillane is Berlin. Berlin is known for its music and its winters.", " The capital of Quillane is", " Berlin"),
    "Tessaro/Madrid": ("The capital of Tessaro is Madrid. Madrid lies on a high plain in the centre of the country.", " The capital of Tessaro is", " Madrid"),
    # The second pilot's fact context, verbatim (commit 362c3ca): the entity is
    # introduced indirectly ("a city called") rather than in the spec's format.
    "Veltoria/Marrowgate (pilot_v2 wording)": ("The capital of the small nation of Veltoria is a city called Marrowgate. "
                                               "Marrowgate sits on the river Oss and is famous for its glass bridges.",
                                               " The capital of Veltoria is", " Marrowgate"),
    # The same invented name in the spec's verbatim format, to separate the
    # effect of the name from the effect of the phrasing.
    "Veltoria/Marrowgate (spec format)": ("The capital of Veltoria is Marrowgate. Marrowgate lies on a fjord and is known for its museums.",
                                          " The capital of Veltoria is", " Marrowgate"),
}
FORMAT_CANDIDATES = {
    "capitals/garden (pilot_v3)": ("apple -> APPLE\nriver -> RIVER\nstone -> STONE\ncloud -> CLOUD", "\ngarden ->", " GARDEN"),
    "capitals/water": ("apple -> APPLE\nriver -> RIVER\nstone -> STONE\ncloud -> CLOUD", "\nwater ->", " WATER"),
    "reverse pairs/dog": ("cat -> tac\nsun -> nus\nbed -> deb\ntop -> pot", "\ndog ->", " god"),
    "plural/cat": ("dog -> dogs\nbook -> books\ntree -> trees\ncar -> cars", "\ncat ->", " cats"),
}
screen = {"facts": {}, "formats": {}, "rule": "a context passes if the whole bound answer's log probability with the "
          "context exceeds its log probability without by at least 2 nats AND the answer's first token is the "
          "reference's most likely next token AND every token of the answer has a teacher-forced log "
          "probability of at least -2 in the reference; fact answers must also be one token; format answers "
          "may be several tokens and are scored teacher-forced over all of them"}


def screen_one(C, Q, A, one_token_required):
    c, q = split(C, Q)
    a = ids(A)
    ref, base = full_answer_lp(c + q, a), full_answer_lp(q, a)
    top1 = logits(seq(c + q))[-1].argmax().item()
    every_token_likely = all(v >= -2.0 for v in ref[1])
    return {"answer_tokens": [tok.decode([t]) for t in a], "n_answer_tokens": len(a),
            "reference_lp": ref[0], "baseline_lp": base[0], "gap": ref[0] - base[0],
            "reference_per_token": ref[1], "baseline_per_token": base[1],
            "reference_top1": tok.decode([top1]), "reference_top1_is_answer_start": top1 == a[0],
            "every_reference_token_at_least_minus_2": every_token_likely,
            "passes": (ref[0] - base[0] >= 2.0) and top1 == a[0] and every_token_likely
            and (len(a) == 1 or not one_token_required)}


for name, (C, Q, A) in FACT_CANDIDATES.items():
    screen["facts"][name] = screen_one(C, Q, A, one_token_required=True)
for name, (C, Q, A) in FORMAT_CANDIDATES.items():
    screen["formats"][name] = screen_one(C, Q, A, one_token_required=False)
R["screening"] = screen
print("screening done", flush=True)

# ------------------------------------------------------- binding vs priming
ETA = 0.1
# Control queries keep the frame but name real entities whose true capital is
# not any of the three bound answers (Oslo, Paris, Tokyo), so a write that
# strengthens a real association cannot raise the bound token on a control;
# and each control tokenises to the same number of tokens as the entity query
# it controls for, so the answer is predicted at the same absolute position.
CONTROL_POOL = ["Spain", "Egypt", "Portugal", "Argentina", "Australia", "Indonesia", "Nigeria", "Kazakhstan",
                "Venezuela", "Slovakia", "Uzbekistan", "Mozambique", "Tajikistan", "Azerbaijan", "Guatemala",
                "Bangladesh", "Cambodia", "Lithuania", "Mauritania", "Kyrgyzstan", "Madagascar", "Zimbabwe"]
BOUND_ANSWERS = (" Oslo", " Paris", " Tokyo")
facts3 = ["Veltoria/Oslo", "Brannock/Paris", "Morvane/Tokyo"]


def pick_controls(entity_query, n=2):
    """The first n countries in the pool whose control query has the same token
    count as the entity query and whose frozen-model top prediction is not a
    bound answer. None of the pool's capitals is a bound answer."""
    target = len(ids(entity_query))
    out = []
    for country in CONTROL_POOL:
        cq = f" The capital of {country} is"
        if len(ids(cq)) != target:
            continue
        top = tok.decode([logits(seq(ids(cq)))[-1].argmax().item()])
        if top in BOUND_ANSWERS:
            continue
        out.append((cq, top))
        if len(out) == n:
            break
    assert len(out) == n, f"no {n} controls of {target} tokens for {entity_query!r}"
    return out
writes = {}
for name in facts3:
    C, Q, A = FACT_CANDIDATES[name]
    c, _ = split(C, Q)
    writes[name] = fold(SITE, "hebb", ETA, seq(c))
bvp = {"eta": ETA, "rescaling": "each foreign write rescaled to the own write's Frobenius norm, as in the pilot",
       "control_rule": "control entities are real, their true capital is none of the bound answers, the "
                       "frozen model's top prediction on each control query is not a bound answer, and each "
                       "control query has the same token count as the entity query it controls for",
       "control_pool": CONTROL_POOL, "rows": {}}
for name in facts3:
    C, Q, A = FACT_CANDIDATES[name]
    c, q = split(C, Q)
    a = ids(A)[0]
    own, own_rep = writes[name]
    controls = pick_controls(Q)
    control_queries = [cq for cq, _ in controls]
    row = {"drift": own_rep["delta_frac"], "entity_query_tokens": len(q),
           "control_queries": {cq: {"tokens": len(ids(cq)), "frozen_top1": top} for cq, top in controls},
           "baseline_entity_query": lp_final(seq(q), a),
           "baseline_control_queries": {cq: lp_final(seq(ids(cq)), a) for cq in control_queries}, "writes": {}}
    for wname in facts3:
        dW = writes[wname][0]
        dW = dW * (own.norm() / dW.norm())
        ent = with_delta(dW, lambda: lp_final(seq(q), a)) - row["baseline_entity_query"]
        ctl = {cq: with_delta(dW, lambda: lp_final(seq(ids(cq)), a)) - row["baseline_control_queries"][cq] for cq in control_queries}
        row["writes"][wname] = {"lift_entity_query": ent, "lift_control_queries": ctl,
                                "binding_transfer": ent - max(ctl.values())}
    bvp["rows"][name] = row
R["binding_vs_priming"] = bvp
print("binding vs priming done", flush=True)

# --------------------------------------------------------------- sigma1 share
PILOT_CONTEXTS = {
    "fact (pilot_v3)": FACT_CANDIDATES["Veltoria/Oslo"][0],
    "fact (pilot_v2, Marrowgate)": FACT_CANDIDATES["Veltoria/Marrowgate (pilot_v2 wording)"][0],
    "format (pilot_v2, v3)": FORMAT_CANDIDATES["capitals/garden (pilot_v3)"][0],
    "topic (pilot_v2, v3)": "The reactor core is cooled by pressurised water. Control rods of boron "
                            "carbide absorb neutrons, and the turbine hall converts steam to power.",
}
ctx_writes = {}
share = {"eta": ETA, "site": SITE, "rule": "hebb", "rows": {}}
for name, C in PILOT_CONTEXTS.items():
    dW, rep = fold(SITE, "hebb", ETA, seq(ids(C)))
    ctx_writes[name] = dW
    s = torch.linalg.svdvals(dW.double())
    share["rows"][name] = {"drift": rep["delta_frac"], "sigma1_over_frobenius": (s[0] / s.norm()).item(),
                           "sigma1_squared_share": (s[0] ** 2 / (s ** 2).sum()).item()}
names = list(PILOT_CONTEXTS)
share["cosines_between_writes"] = {f"{a} | {b}": cos(ctx_writes[a], ctx_writes[b])
                                   for i, a in enumerate(names) for b in names[i + 1:]}
R["sigma1_share"] = share
print("sigma1 share done", flush=True)

# --------------------------------------------------------- drift by site, linearity
SITES = {"blocks.6.mlp": "blocks.6.mlp", "blocks.11.mlp": "blocks.11.mlp",
         "blocks.11.attn.head.7": "blocks.11.attn.head.7",
         "blocks.2.attn stripes": [f"blocks.2.attn.head.{h}" for h in range(12)],
         "blocks.11.attn stripes": [f"blocks.11.attn.head.{h}" for h in range(12)]}
fact_ctx = seq(ids(FACT_CANDIDATES["Veltoria/Oslo"][0]))
drift = {"context": "Veltoria/Oslo", "rows": {}}
lin = {"context": "Veltoria/Oslo", "rows": {}}
for sname, site in SITES.items():
    for mode in ("hebb", "oja"):
        etas = [1e-4, 1e-3, 1e-2] if mode == "oja" else [1e-3, 1e-2]
        deltas, reps = {}, {}
        for eta in etas:
            d, rep = fold(site, mode, eta, fact_ctx)
            deltas[eta], reps[eta] = d, rep
        drift["rows"][f"{sname} / {mode}"] = {str(e): reps[e] for e in etas}
        e1, e2 = etas[-2], etas[-1]
        lin["rows"][f"{sname} / {mode}"] = {
            "etas": [e1, e2], "drift_ratio": reps[e2]["delta_frac"] / reps[e1]["delta_frac"],
            "expected_ratio": e2 / e1, "cosine": cos(deltas[e1], deltas[e2]),
            "clipped": [reps[e1]["clipped"], reps[e2]["clipped"]]}
        print("drift", sname, mode, {str(e): round(reps[e]["delta_frac"], 5) for e in etas}, flush=True)
R["drift_by_site"] = drift
R["linearity"] = lin

# ------------------------------------------------------------------ C0 gate
c0 = {}
for sname, site in {"blocks.6.mlp": "blocks.6.mlp", "blocks.11.attn stripes": SITES["blocks.11.attn stripes"]}.items():
    with torch.no_grad():
        lg0, cache0 = model.run_with_cache(fact_ctx)
    if isinstance(site, list):
        d = MultiSitePlasticity(model, [SiteSpec(s, mode="off", eta=0.0) for s in site])
    else:
        d = OjaPlasticity(model, site, eta=0.0, mode="off")
    d.install()
    with torch.no_grad():
        lg1, cache1 = model.run_with_cache(fact_ctx)
    d.remove()
    d.revert()
    same = all(torch.equal(cache0[k], cache1[k]) for k in cache0) and torch.equal(lg0, lg1)
    c0[sname] = {"bit_identical": bool(same), "n_cache_entries": len(cache0)}
R["c0_gate"] = c0
print("c0", c0, flush=True)

# ---------------------------------------------------------------- BOS share
bos = {"site": SITE, "rows": {}, "note": "x is blocks.6.mlp.hook_post, y is blocks.6.hook_mlp_out; the Hebbian "
       "term is the position mean of x y^T; the BOS row is position 0. The write is full = bos + rest. "
       "'bos_projection_share' is <bos, full> / |full|^2, which with rest's projection sums to one and is "
       "the decomposition; 'bos_relative_magnitude' is |bos| / |full|; 'bos_term_squared_share' is "
       "|bos|^2 / |full|^2 and is NOT a decomposition because bos and rest are not orthogonal"}
bos_terms, full_terms = {}, {}
for name, C in PILOT_CONTEXTS.items():
    with torch.no_grad():
        _, cache = model.run_with_cache(seq(ids(C)), names_filter=lambda n: n in ("blocks.6.mlp.hook_post", "blocks.6.hook_mlp_out"))
    x, y = cache["blocks.6.mlp.hook_post"][0].double(), cache["blocks.6.hook_mlp_out"][0].double()
    full = x.T @ y / x.shape[0]
    b = torch.outer(x[0], y[0]) / x.shape[0]
    rest = full - b
    bos_terms[name], full_terms[name] = b, full
    bos["rows"][name] = {"n_positions": x.shape[0], "y_norm_bos": y[0].norm().item(),
                         "y_norm_median_others": y[1:].norm(dim=1).median().item(),
                         "bos_term_squared_share": (b.norm() ** 2 / full.norm() ** 2).item(),
                         "bos_relative_magnitude": (b.norm() / full.norm()).item(),
                         "bos_projection_share": ((b * full).sum() / full.norm() ** 2).item(),
                         "rest_projection_share": ((rest * full).sum() / full.norm() ** 2).item(),
                         "cosine_full_vs_without_bos": cos(full, rest)}
bos["cosines_between_bos_terms"] = {f"{a} | {b_}": cos(bos_terms[a], bos_terms[b_]) for i, a in enumerate(names) for b_ in names[i + 1:]}
bos["cosines_between_writes_without_bos"] = {f"{a} | {b_}": cos(full_terms[a] - bos_terms[a], full_terms[b_] - bos_terms[b_])
                                             for i, a in enumerate(names) for b_ in names[i + 1:]}
R["bos_share"] = bos
print("bos done", flush=True)

# --------------------------------------------------------- position control
FILLER = ("It was an ordinary afternoon and nothing in particular was happening anywhere, which is how most "
          "afternoons go, and people went about their usual business as they always do without much thought "
          "about any of it at all, one thing after another.")
PILOT_CASES = {
    "fact": (FACT_CANDIDATES["Veltoria/Oslo"][0], FACT_CANDIDATES["Veltoria/Oslo"][1]),
    "format": (FORMAT_CANDIDATES["capitals/garden (pilot_v3)"][0], FORMAT_CANDIDATES["capitals/garden (pilot_v3)"][1]),
    "topic": (PILOT_CONTEXTS["topic (pilot_v2, v3)"], " The engineers checked the"),
}


def kl(t, s):
    lt, ls = torch.log_softmax(t, -1), torch.log_softmax(s, -1)
    return (lt.exp() * (lt - ls)).sum(-1)


pos = {"filler": FILLER, "position_shift": "hook_pos_embed replaced so the query's tokens take the positions "
       "they have in the reference while nothing precedes them but BOS", "rows": {}}
for name, (C, Q) in PILOT_CASES.items():
    c, q = split(C, Q)
    nq, nc = len(q), len(c)
    teacher, base = logits(seq(c + q))[-nq:], logits(seq(q))[-nq:]
    idx = torch.tensor([0] + list(range(nc + 1, nc + 1 + nq)))

    def shift(t, hook, idx=idx):
        return model.W_pos[idx][None]
    with torch.no_grad():
        shifted = model.run_with_hooks(seq(q), fwd_hooks=[("hook_pos_embed", shift)])[0, -nq:].double()
        fill = ids(FILLER)[:nc]
        assert len(fill) == nc
        filler = model(seq(fill + q))[0, -nq:].double()
    pos["rows"][name] = {"kl_final_baseline": kl(teacher[-1], base[-1]).item(),
                         "kl_final_position_shifted": kl(teacher[-1], shifted[-1]).item(),
                         "kl_final_neutral_filler": kl(teacher[-1], filler[-1]).item()}
R["position_control"] = pos
print("position done", flush=True)

# ---------------------------------------------------------------- loop writes
TOPIC_B = ("The orchestra tuned before the concert. The strings followed the oboe's note, and the "
           "conductor waited until the hall was silent.")
loop = {"site": SITE, "rule": "hebb", "eta": 1e-3, "layer_start": 0, "layer_end": 11,
        "definition": "the rule observes every loop iteration and is applied once at the end; the loop starts "
                      "from the state the context pass produced, per atr_bridge.initial_state", "rows": {}}
loop_deltas = {}
ctx_for_loop = {"topic A (pilot)": PILOT_CONTEXTS["topic (pilot_v2, v3)"], "topic B": TOPIC_B}
for name, C in ctx_for_loop.items():
    cw, _ = fold(SITE, "hebb", loop["eta"], seq(ids(C)))
    loop_deltas[(name, 0)] = cw
    st = initial_state(model, C, layer_end=11)
    step = make_atr_step(model, C, layer_start=0, layer_end=11, initial_norm=st.initial_norm)
    for N in (1, 10, 100):
        p = OjaPlasticity(model, SITE, eta=loop["eta"], mode="hebb", max_delta_frac=1.0)
        p.install()
        r = st.tensor.clone()
        for _ in range(N):
            r = step(model, r)
        p.remove()
        rep = p.apply()
        loop_deltas[(name, N)] = p.delta.clone()
        p.revert()
        loop["rows"][f"{name} / N={N}"] = {"drift": rep["delta_frac"], "cos_with_own_context_write": cos(loop_deltas[(name, N)], cw)}
    assert torch.equal(site6.weight.detach(), W0)
loop["cos_between_contexts"] = {f"N={N}": cos(loop_deltas[("topic A (pilot)", N)], loop_deltas[("topic B", N)]) for N in (0, 1, 10, 100)}
R["loop_writes"] = loop
print("loop done", loop["cos_between_contexts"], flush=True)

# -------------------------------------------------------------------- timings
t = time.time()
for _ in range(5):
    fold(SITE, "hebb", 1e-3, fact_ctx)
per_fold = (time.time() - t) / 5
q7 = seq(split(*PILOT_CASES["fact"])[1])
t = time.time()
for _ in range(20):
    logits(q7)
per_query = (time.time() - t) / 20
t = time.time()
for _ in range(3):
    fold(SITES["blocks.11.attn stripes"], "hebb", 1e-3, fact_ctx)
per_fold_stripes = (time.time() - t) / 3
R["timings"] = {"seconds_per_fold_mlp": per_fold, "seconds_per_fold_12_stripes": per_fold_stripes,
                "seconds_per_query_pass_7_tokens": per_query, "context_tokens": fact_ctx.shape[1]}
R["elapsed_s"] = time.time() - T_START
assert torch.equal(site6.weight.detach(), W0), "weight not restored"
out = sys.argv[1] if len(sys.argv) > 1 else "/dev/null"
json.dump(R, open(out, "w"), indent=1)
print(json.dumps(R["timings"]), f"elapsed {R['elapsed_s']:.0f}s")
