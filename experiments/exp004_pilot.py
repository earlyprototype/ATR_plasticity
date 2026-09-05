"""
EXP-004 pilot, second version: context-to-weight transfer probe, single pass, no
loop, with the controls the first version lacked.

The first version (commit 7288f32, output ``output_exp004/pilot_v1.json``) built
its reference from a context and a query tokenised separately, so the boundary
token was not the one GPT-2 would produce for the joined text; scored the mean
KL over all query positions, which let the first query token (adjacent to BOS in
the query-alone run) dominate; asked a fact query whose answer was the country,
not the bound attribute; and compared only against an isotropic random write,
the control register row C-23 retired. This version fixes each of those and adds
the swapped-context control. It remains a pilot: three contexts, one site, no
loop. Nothing from it enters the claim register.

Run from the repository root:

    .venv/bin/python experiments/exp004_pilot.py experiments/output_exp004/pilot_v2.json
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plasticity import OjaPlasticity  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

torch.manual_seed(0)
model = HookedTransformer.from_pretrained("gpt2", device="cpu")
model.eval()
tok = model.tokenizer
SITE = "blocks.6.mlp"
ETAS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
N_RANDOM = 10

# Contexts carry no trailing whitespace; queries begin with the separator, so
# joint tokenisation of context + query has the context's tokens as a prefix.
# ``bound`` is the answer the context binds, whose first token is scored.
CASES = {
    "fact": dict(
        C="The capital of the small nation of Veltoria is a city called Marrowgate. "
          "Marrowgate sits on the river Oss and is famous for its glass bridges.",
        Q=" The capital of Veltoria is",
        bound=" Marrowgate",
    ),
    "format": dict(
        C="apple -> APPLE\nriver -> RIVER\nstone -> STONE\ncloud -> CLOUD",
        Q="\ngarden ->",
        bound=" GARDEN",
    ),
    "topic": dict(
        C="The reactor core is cooled by pressurised water. Control rods of boron "
          "carbide absorb neutrons, and the turbine hall converts steam to power.",
        Q=" The engineers checked the",
        bound=None,
    ),
}


def ids(text):
    return tok.encode(text)[1:]  # drop the BOS the tokenizer prepends


def build(case):
    c, cq = ids(case["C"]), ids(case["C"] + case["Q"])
    assert cq[: len(c)] == c, "context tokens must be a prefix of the joint tokenisation"
    q = cq[len(c):]
    bos = tok.bos_token_id
    ref_tokens = torch.tensor([[bos] + cq])
    q_tokens = torch.tensor([[bos] + q])
    ctx_tokens = torch.tensor([[bos] + c])
    bound = ids(case["bound"])[0] if case["bound"] else None
    return ref_tokens, q_tokens, ctx_tokens, len(q), bound


@torch.no_grad()
def q_logits(tokens, n_q):
    return model(tokens)[0, -n_q:].double()


def kl_per_pos(teacher, student):
    lt, ls = torch.log_softmax(teacher, -1), torch.log_softmax(student, -1)
    return (lt.exp() * (lt - ls)).sum(-1)


def scores(teacher, student, bound):
    k = kl_per_pos(teacher, student)
    out = {
        "kl_final": k[-1].item(),
        "kl_mean_from_pos2": k[1:].mean().item() if len(k) > 1 else float("nan"),
        "kl_per_position": [round(v, 6) for v in k.tolist()],
    }
    if bound is not None:
        out["bound_logprob"] = torch.log_softmax(student[-1], -1)[bound].item()
    return out


def entropy(logits):
    p = torch.log_softmax(logits, -1)
    return -(p.exp() * p).sum().item()


def temperature_matched(baseline_final, target_entropy):
    """Scale the baseline's final-position logits by 1/T so its entropy matches
    the test's. A write that only flattens the distribution scores no better
    than this."""
    lo, hi = 0.25, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if entropy(baseline_final / mid) < target_entropy:
            lo = mid
        else:
            hi = mid
    return baseline_final / mid, mid


def hebb_delta(ctx_tokens, eta):
    """One context pass with the rule observing, then one apply. The rule is
    removed before any query pass, and C3 is checked: nothing accumulates after."""
    p = OjaPlasticity(model, SITE, eta=eta, mode="hebb", max_delta_frac=1.0)
    p.install()
    with torch.no_grad():
        model(ctx_tokens)
    p.remove()
    rep = p.apply()
    d = p.delta.clone()
    p.revert()
    assert p._n_batches == 0 and p._acc is None
    return d, rep


def with_delta(p_site, w0, delta, fn):
    p_site.write(w0 + delta)
    try:
        return fn()
    finally:
        p_site.write(w0)


results = {"site": SITE, "etas": ETAS, "n_random": N_RANDOM, "cases": {}}
t0 = time.time()
probe = OjaPlasticity(model, SITE, eta=0.0, mode="off")
site, W0 = probe._site, probe.W0.clone()
w_sha_before = torch.sum(W0.double()).item()

built = {name: build(case) for name, case in CASES.items()}
for name, (ref_t, q_t, ctx_t, n_q, bound) in built.items():
    teacher = q_logits(ref_t, n_q)
    baseline = q_logits(q_t, n_q)
    row = {
        "n_query_tokens": n_q,
        "query_tokens": [tok.decode([i]) for i in q_t[0, 1:].tolist()],
        "bound_token": tok.decode([bound]) if bound is not None else None,
        "teacher_top1_final": tok.decode([teacher[-1].argmax().item()]),
        "teacher": {"bound_logprob": torch.log_softmax(teacher[-1], -1)[bound].item()} if bound is not None else {},
        "baseline": scores(teacher, baseline, bound),
        "sweep": [],
    }
    for eta in ETAS:
        dW, rep = hebb_delta(ctx_t, eta)
        test = with_delta(site, W0, dW, lambda: q_logits(q_t, n_q))
        cell = {"eta": eta, "delta_frac": rep["delta_frac"], "clipped": rep["clipped"],
                "hebb": scores(teacher, test, bound)}
        # C4: swapped-context writes at the same eta.
        cell["swap"] = {}
        for other, (_, _, octx, _, _) in built.items():
            if other == name:
                continue
            odW, _ = hebb_delta(octx, eta)
            odW = odW * (dW.norm() / odW.norm())  # same Frobenius norm as the own write
            cell["swap"][other] = scores(teacher, with_delta(site, W0, odW, lambda: q_logits(q_t, n_q)), bound)
        # C2: rank-one random writes matched on the largest singular value.
        s1 = torch.linalg.svdvals(dW.double())[0].item()
        g = torch.Generator().manual_seed(1000 + int(round(eta * 1e6)))
        rand = []
        for _ in range(N_RANDOM):
            u = torch.randn(dW.shape[0], generator=g); v = torch.randn(dW.shape[1], generator=g)
            R = torch.outer(u / u.norm(), v / v.norm()).to(dW.dtype) * s1
            rand.append(scores(teacher, with_delta(site, W0, R, lambda: q_logits(q_t, n_q)), bound))
        cell["random_rank1"] = {
            "kl_final_median": float(torch.tensor([r["kl_final"] for r in rand]).median()),
            "kl_final_min": min(r["kl_final"] for r in rand),
            "bound_logprob_max": max(r["bound_logprob"] for r in rand) if bound is not None else None,
        }
        # C5: temperature-matched baseline at the final position.
        tm, T = temperature_matched(baseline[-1], entropy(test[-1]))
        cell["temperature_matched"] = {"T": T, "kl_final": kl_per_pos(teacher[-1:], tm[None])[0].item()}
        if bound is not None:
            cell["temperature_matched"]["bound_logprob"] = torch.log_softmax(tm, -1)[bound].item()
        row["sweep"].append(cell)
    results["cases"][name] = row

assert torch.equal(site.weight.detach(), W0), "weight not restored"
results["elapsed_s"] = time.time() - t0
out = sys.argv[1] if len(sys.argv) > 1 else "/dev/null"
json.dump(results, open(out, "w"), indent=1)

for name, row in results["cases"].items():
    b = row["baseline"]
    print(f"== {name}: query {row['query_tokens']} bound={row['bound_token']!r} teacher top1={row['teacher_top1_final']!r}")
    print(f"   baseline: kl_final={b['kl_final']:.3f} kl_mean_pos2+={b['kl_mean_from_pos2']:.3f} "
          f"bound_lp={b.get('bound_logprob', float('nan')):.2f} (teacher bound_lp={row['teacher'].get('bound_logprob', float('nan')):.2f})")
    print(f"   {'eta':>6} {'dW/W':>7} | {'hebb':>6} {'swapA':>6} {'swapB':>6} {'rand':>6} {'tempT':>6} | {'hebb':>6} {'swapA':>6} {'swapB':>6} {'rand':>6} {'tempT':>6}   [kl_final | bound_lp]")
    for c in row["sweep"]:
        sw = list(c["swap"].values())
        blp = lambda d: d.get("bound_logprob", float("nan"))
        print(f"   {c['eta']:6.0e} {c['delta_frac']:7.4f} | {c['hebb']['kl_final']:6.3f} {sw[0]['kl_final']:6.3f} {sw[1]['kl_final']:6.3f} "
              f"{c['random_rank1']['kl_final_median']:6.3f} {c['temperature_matched']['kl_final']:6.3f} | "
              f"{blp(c['hebb']):6.2f} {blp(sw[0]):6.2f} {blp(sw[1]):6.2f} "
              f"{(c['random_rank1']['bound_logprob_max'] if c['random_rank1']['bound_logprob_max'] is not None else float('nan')):6.2f} "
              f"{c['temperature_matched'].get('bound_logprob', float('nan')):6.2f}")
    print(f"   per-position baseline KL: {b['kl_per_position']}")
print(f"elapsed {results['elapsed_s']:.1f}s")
