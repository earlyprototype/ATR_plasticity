"""
EXP-004 pilot, fourth run: context-to-weight transfer probe, single pass, no
loop, with the controls the earlier versions lacked or got wrong.

History. The first version (commit 7288f32, ``output_exp004/pilot_v1.json``)
tokenised the context and query separately, averaged the score over all query
positions, asked for the country rather than the bound word, and compared only
against an isotropic random write. The second and third runs (``pilot_v2.json``,
``pilot_v3.json``) fixed those and added the swapped-context, rank-one random
and temperature-matched controls; the second's fact context failed screening
and the third replaced it. Review of the third run found two more defects,
fixed here: the bound answer was scored on its first token only, which for a
three-token answer such as " GARDEN" is the log probability of " G"; and the C3
check was an assertion placed after ``revert()``, which zeroes the fields it
asserted on, so it could not fail. This run scores the whole bound answer
teacher-forced as well as its first token, checks C3 by counting hooks at the
site after every query pass and by a deliberate-leak arm that installs the rule
during a query pass and then applies it, runs the C0 gate on every context
pass, reports the random arm as percentiles, and asserts the temperature
solver converged. The own-write, random and temperature arms at the final
position are computed exactly as in the third run, so those numbers should be
bit-identical to ``pilot_v3.json`` on the same machine.

It remains a pilot: three contexts, one site, one seed set, no loop. Nothing
from it enters the claim register.

Run from the repository root:

    .venv/bin/python experiments/exp004_pilot.py experiments/output_exp004/pilot_v4.json
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
HOOK_POINTS = ("blocks.6.mlp.hook_post", "blocks.6.hook_mlp_out")  # where the site's rule attaches
ETAS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
N_RANDOM = 10

# Contexts carry no trailing whitespace; queries begin with the separator, so
# joint tokenisation of context + query has the context's tokens as a prefix.
# ``bound`` is the answer the context binds; its first token and its whole
# token sequence are both scored.
CASES = {
    "fact": dict(
        # An invented entity bound to a common single-token answer, the fact
        # format EXP_004_SPEC.md defines. The earlier "Marrowgate" context
        # (pilot_v2.json) failed the screening rule: a rare multi-token answer
        # GPT-2 Small does not reproduce even with the context present.
        C="The capital of Veltoria is Oslo. Oslo lies on a fjord and is known for its museums.",
        Q=" The capital of Veltoria is",
        bound=" Oslo",
    ),
    "format": dict(
        # The bound answer is three tokens (" G", "ARD", "EN"); kept from the
        # third run so the first-token and whole-answer scores can be compared
        # on the same context.
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
    bound = ids(case["bound"]) if case["bound"] else None
    return ref_tokens, q_tokens, ctx_tokens, len(q), bound


@torch.no_grad()
def all_logits(tokens):
    return model(tokens)[0].double()


def q_logits(tokens, n_q):
    return all_logits(tokens)[-n_q:]


def kl_per_pos(teacher, student):
    lt, ls = torch.log_softmax(teacher, -1), torch.log_softmax(student, -1)
    return (lt.exp() * (lt - ls)).sum(-1)


def full_answer(prefix_tokens, bound, temperature=1.0):
    """Teacher-forced log probability of every token of the bound answer after
    the prefix: one forward pass on prefix + answer[:-1]; the logits at the
    position that predicts each answer token are read, optionally divided by a
    temperature, and the answer token's log probability summed."""
    n_p = prefix_tokens.shape[1]
    toks = torch.cat([prefix_tokens, torch.tensor([bound[:-1]], dtype=torch.long)], 1)
    lg = all_logits(toks)
    per = [torch.log_softmax(lg[n_p - 1 + i] / temperature, -1)[a].item() for i, a in enumerate(bound)]
    return sum(per), per


def scores(teacher, student, bound, prefix_tokens=None, temperature=1.0):
    k = kl_per_pos(teacher, student)
    out = {
        "kl_final": k[-1].item(),
        "kl_mean_from_pos2": k[1:].mean().item() if len(k) > 1 else float("nan"),
        "kl_per_position": [round(v, 6) for v in k.tolist()],
    }
    if bound is not None:
        out["bound_logprob"] = torch.log_softmax(student[-1] / temperature, -1)[bound[0]].item()
        if prefix_tokens is not None:
            total, per = full_answer(prefix_tokens, bound, temperature)
            out["bound_logprob_full"] = total
            out["bound_logprob_per_token"] = per
    return out


def entropy(logits):
    p = torch.log_softmax(logits, -1)
    return -(p.exp() * p).sum().item()


def temperature_matched(baseline_final, target_entropy):
    """Scale the baseline's final-position logits by 1/T so its entropy matches
    the test's. A write that only flattens the distribution scores no better
    than this. The bisection is checked to have converged."""
    lo, hi = 0.25, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if entropy(baseline_final / mid) < target_entropy:
            lo = mid
        else:
            hi = mid
    assert abs(entropy(baseline_final / mid) - target_entropy) < 1e-6, "temperature match did not converge"
    return baseline_final / mid, mid


def n_hooks():
    return sum(len(model.hook_dict[h]._forward_hooks) for h in HOOK_POINTS)


HOOKS_AT_REST = n_hooks()


def c0_gate(ctx_tokens):
    """The context pass with the rule's hooks installed at eta 0, mode off,
    against the same pass without them, at every cached activation and the
    logits. Must be bit-identical."""
    with torch.no_grad():
        lg0, c0 = model.run_with_cache(ctx_tokens)
    p = OjaPlasticity(model, SITE, eta=0.0, mode="off")
    p.install()
    with torch.no_grad():
        lg1, c1 = model.run_with_cache(ctx_tokens)
    p.remove()
    p.revert()
    return bool(all(torch.equal(c0[k], c1[k]) for k in c0) and torch.equal(lg0, lg1))


def hebb_delta(ctx_tokens, eta):
    """One context pass with the rule observing, then one apply. The rule is
    removed before any query pass; the hook count at the site is checked to be
    back at rest."""
    p = OjaPlasticity(model, SITE, eta=eta, mode="hebb", max_delta_frac=1.0)
    p.install()
    with torch.no_grad():
        model(ctx_tokens)
    p.remove()
    rep = p.apply()
    d = p.delta.clone()
    p.revert()
    assert n_hooks() == HOOKS_AT_REST, "rule hooks still installed after the fold"
    return d, rep


def with_delta(p_site, w0, delta, fn):
    """Install a delta, run fn, restore. C3: no rule hook may be present
    during fn, checked before and after."""
    assert n_hooks() == HOOKS_AT_REST
    p_site.write(w0 + delta)
    try:
        return fn()
    finally:
        p_site.write(w0)
        assert n_hooks() == HOOKS_AT_REST, "a rule hook was installed during a query pass"


def deliberate_leak(q_tokens, n_q, bound, eta=1e-1):
    """C3 fail direction: install the rule during a query pass. The hook count
    detector must read the two hooks, the score must not move until apply(),
    and must move after it."""
    before = q_logits(q_tokens, n_q)[-1]
    p = OjaPlasticity(model, SITE, eta=eta, mode="hebb", max_delta_frac=1.0)
    p.install()
    during = q_logits(q_tokens, n_q)[-1]
    detector = n_hooks() - HOOKS_AT_REST
    p.remove()
    rep = p.apply()
    after = q_logits(q_tokens, n_q)[-1]
    p.revert()
    key = (lambda l: torch.log_softmax(l, -1)[bound[0]].item()) if bound else (lambda l: l.max().item())
    return {"eta": eta, "hooks_detected_during_query": detector, "score_before": key(before),
            "score_with_rule_installed_no_apply": key(during), "score_after_apply": key(after),
            "drift_after_apply": rep["delta_frac"],
            "detector_fired": detector == 2, "score_moved_only_after_apply":
                bool(torch.equal(before, during) and not torch.equal(before, after))}


results = {"site": SITE, "etas": ETAS, "n_random": N_RANDOM, "cases": {},
           "notes": {"random_directions": "the ten rank-one directions are seeded by eta only, so the three "
                                          "contexts see the same ten directions at each eta, scaled to each "
                                          "own write's largest singular value",
                     "temperature_matched_full": "for the whole-answer score the same temperature is applied "
                                                 "to the baseline's logits at every answer position"}}
t0 = time.time()
probe = OjaPlasticity(model, SITE, eta=0.0, mode="off")
site, W0 = probe._site, probe.W0.clone()

built = {name: build(case) for name, case in CASES.items()}
for name, (ref_t, q_t, ctx_t, n_q, bound) in built.items():
    teacher = q_logits(ref_t, n_q)
    baseline = q_logits(q_t, n_q)
    row = {
        "n_query_tokens": n_q,
        "query_tokens": [tok.decode([i]) for i in q_t[0, 1:].tolist()],
        "bound_token": tok.decode([bound[0]]) if bound is not None else None,
        "bound_tokens": [tok.decode([i]) for i in bound] if bound is not None else None,
        "teacher_top1_final": tok.decode([teacher[-1].argmax().item()]),
        "c0_gate_bit_identical": c0_gate(ctx_t),
        "teacher": {},
        "baseline": scores(teacher, baseline, bound, q_t),
        "c3_deliberate_leak": deliberate_leak(q_t, n_q, bound),
        "sweep": [],
    }
    if bound is not None:
        row["teacher"]["bound_logprob"] = torch.log_softmax(teacher[-1], -1)[bound[0]].item()
        row["teacher"]["bound_logprob_full"], row["teacher"]["bound_logprob_per_token"] = full_answer(ref_t, bound)
    for eta in ETAS:
        dW, rep = hebb_delta(ctx_t, eta)
        test = with_delta(site, W0, dW, lambda: q_logits(q_t, n_q))
        cell = {"eta": eta, "delta_frac": rep["delta_frac"], "clipped": rep["clipped"],
                "hebb": with_delta(site, W0, dW, lambda: scores(teacher, test, bound, q_t))}
        # C4: swapped-context writes at the same eta, rescaled to the own write's
        # Frobenius norm (the third run's convention; the spec now matches on the
        # largest singular value, and both norms are recorded here).
        cell["swap"] = {}
        for other, (_, _, octx, _, _) in built.items():
            if other == name:
                continue
            odW, _ = hebb_delta(octx, eta)
            odW = odW * (dW.norm() / odW.norm())
            cell["swap"][other] = with_delta(site, W0, odW, lambda: scores(teacher, q_logits(q_t, n_q), bound, q_t))
            cell["swap"][other]["sigma1_ratio_to_own"] = (torch.linalg.svdvals(odW.double())[0] / torch.linalg.svdvals(dW.double())[0]).item()
        # C2: rank-one random writes matched on the largest singular value.
        s1 = torch.linalg.svdvals(dW.double())[0].item()
        g = torch.Generator().manual_seed(1000 + int(round(eta * 1e6)))
        rand = []
        for _ in range(N_RANDOM):
            u = torch.randn(dW.shape[0], generator=g); v = torch.randn(dW.shape[1], generator=g)
            Rm = torch.outer(u / u.norm(), v / v.norm()).to(dW.dtype) * s1
            rand.append(with_delta(site, W0, Rm, lambda: scores(teacher, q_logits(q_t, n_q), bound, q_t)))

        def pct(key):
            x = torch.tensor([r[key] for r in rand], dtype=torch.float64)
            return {"p50": torch.quantile(x, 0.5).item(), "p95": torch.quantile(x, 0.95).item(),
                    "min": x.min().item(), "max": x.max().item()}
        cell["random_rank1"] = {"sigma1": s1, "kl_final": pct("kl_final"),
                               # kept for continuity with pilot_v3.json's fields
                               "kl_final_min": min(r["kl_final"] for r in rand),
                               "bound_logprob_max": max(r["bound_logprob"] for r in rand) if bound is not None else None}
        if bound is not None:
            cell["random_rank1"]["bound_logprob"] = pct("bound_logprob")
            cell["random_rank1"]["bound_logprob_full"] = pct("bound_logprob_full")
        # C5: temperature-matched baseline at the final position; for the
        # whole-answer score the same temperature at every answer position.
        tm, T = temperature_matched(baseline[-1], entropy(test[-1]))
        cell["temperature_matched"] = {"T": T, "kl_final": kl_per_pos(teacher[-1:], tm[None])[0].item()}
        if bound is not None:
            cell["temperature_matched"]["bound_logprob"] = torch.log_softmax(tm, -1)[bound[0]].item()
            cell["temperature_matched"]["bound_logprob_full"], _ = full_answer(q_t, bound, T)
        row["sweep"].append(cell)
    results["cases"][name] = row

assert torch.equal(site.weight.detach(), W0), "weight not restored"
assert n_hooks() == HOOKS_AT_REST
results["elapsed_s"] = time.time() - t0
out = sys.argv[1] if len(sys.argv) > 1 else "/dev/null"
json.dump(results, open(out, "w"), indent=1)

for name, row in results["cases"].items():
    b = row["baseline"]
    print(f"== {name}: query {row['query_tokens']} bound={row['bound_tokens']!r} teacher top1={row['teacher_top1_final']!r} "
          f"C0={row['c0_gate_bit_identical']} leak detector={row['c3_deliberate_leak']['detector_fired']} "
          f"moved only after apply={row['c3_deliberate_leak']['score_moved_only_after_apply']}")
    print(f"   baseline: kl_final={b['kl_final']:.3f} bound_lp={b.get('bound_logprob', float('nan')):.2f} "
          f"full={b.get('bound_logprob_full', float('nan')):.2f} (teacher bound_lp={row['teacher'].get('bound_logprob', float('nan')):.2f} "
          f"full={row['teacher'].get('bound_logprob_full', float('nan')):.2f})")
    print(f"   {'eta':>6} {'dW/W':>7} | {'hebb':>6} {'swapA':>6} {'swapB':>6} {'rand':>6} {'tempT':>6} | "
          f"{'hebb':>6} {'swapA':>6} {'swapB':>6} {'rand95':>6} {'tempT':>6} | {'hebb':>6} {'swapA':>6} {'swapB':>6} {'rand95':>6} {'tempT':>6}"
          f"   [kl_final | bound_lp first | bound_lp full]")
    for c in row["sweep"]:
        sw = list(c["swap"].values())
        f = lambda d, k: d.get(k, float("nan"))
        r = c["random_rank1"]
        print(f"   {c['eta']:6.0e} {c['delta_frac']:7.4f} | {c['hebb']['kl_final']:6.3f} {sw[0]['kl_final']:6.3f} {sw[1]['kl_final']:6.3f} "
              f"{r['kl_final']['p50']:6.3f} {c['temperature_matched']['kl_final']:6.3f} | "
              f"{f(c['hebb'], 'bound_logprob'):6.2f} {f(sw[0], 'bound_logprob'):6.2f} {f(sw[1], 'bound_logprob'):6.2f} "
              f"{(r['bound_logprob']['p95'] if 'bound_logprob' in r else float('nan')):6.2f} {f(c['temperature_matched'], 'bound_logprob'):6.2f} | "
              f"{f(c['hebb'], 'bound_logprob_full'):6.2f} {f(sw[0], 'bound_logprob_full'):6.2f} {f(sw[1], 'bound_logprob_full'):6.2f} "
              f"{(r['bound_logprob_full']['p95'] if 'bound_logprob_full' in r else float('nan')):6.2f} {f(c['temperature_matched'], 'bound_logprob_full'):6.2f}")
    print(f"   per-position baseline KL: {b['kl_per_position']}")
print(f"elapsed {results['elapsed_s']:.1f}s")
