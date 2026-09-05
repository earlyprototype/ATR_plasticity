"""
EXP-004 pilot: context-to-weight transfer probe, single pass, no ATR loop.

A scratch run made 2026-09-05 before EXP_004_SPEC.md was written. Recorded in
output_exp004/PILOT.md with its limits; the operator has ruled it is to be given
little weight, and nothing from it enters the claim register.

Run from the repository root:  .venv/bin/python experiments/exp004_pilot.py out.json

  teacher   = model(C + Q)          logits at the Q positions
  baseline  = model(Q)              no context, no weight change
  transfer  = model_{W0+dW}(Q)      context folded into blocks.6.mlp W_out by
                                    one Hebbian write over C's activations
  control   = model_{W0+R}(Q)       R gaussian, Frobenius-norm-matched to dW

Score: mean KL(teacher || student) over Q positions, and the teacher's top-1
log-prob at the final position. Lower KL = more of the context survived.
"""
import sys, json, time, os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer_lens import HookedTransformer
from plasticity import OjaPlasticity

torch.manual_seed(0)
model = HookedTransformer.from_pretrained("gpt2", device="cpu")
model.eval()
SITE = "blocks.6.mlp"

CASES = {
    "fact": (
        "The capital of the small nation of Veltoria is a city called Marrowgate. "
        "Marrowgate sits on the river Oss and is famous for its glass bridges. ",
        "Tourists who visit Veltoria usually go first to its capital,",
    ),
    "format": (
        "apple -> APPLE\nriver -> RIVER\nstone -> STONE\ncloud -> CLOUD\n",
        "garden ->",
    ),
    "topic": (
        "The reactor core is cooled by pressurised water. Control rods of boron "
        "carbide absorb neutrons, and the turbine hall converts steam to power. ",
        "The engineers checked the",
    ),
}
ETAS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]

def toks(s):
    return model.to_tokens(s)  # prepends BOS

@torch.no_grad()
def logits_at_q(text_tokens, n_q):
    lg = model(text_tokens)[0]            # (T, vocab)
    return lg[-n_q:].double()

def kl(teacher, student):
    lt = torch.log_softmax(teacher, -1)
    ls = torch.log_softmax(student, -1)
    return (lt.exp() * (lt - ls)).sum(-1).mean().item()

def top1_lp(teacher, student):
    t = teacher[-1].argmax().item()
    return torch.log_softmax(student[-1], -1)[t].item(), model.tokenizer.decode([t])

results = {}
t0 = time.time()
for name, (C, Q) in CASES.items():
    q_tok = toks(Q)[:, 1:]                 # drop BOS from Q for concatenation
    n_q = q_tok.shape[1]
    cq = torch.cat([toks(C), q_tok], dim=1)
    teacher = logits_at_q(cq, n_q)
    base = logits_at_q(toks(Q), n_q)
    row = {"baseline_kl": kl(teacher, base),
           "baseline_top1_lp": top1_lp(teacher, base)[0],
           "teacher_top1": top1_lp(teacher, teacher)[1],
           "teacher_top1_lp": top1_lp(teacher, teacher)[0],
           "sweep": []}
    for eta in ETAS:
        p = OjaPlasticity(model, SITE, eta=eta, mode="hebb", max_delta_frac=1.0)
        with p:
            with torch.no_grad():
                model(toks(C))               # one pass over the context, hooks observe
            rep = p.apply()
            tr = logits_at_q(toks(Q), n_q)
            dW = p.delta.clone()
            cell = {"eta": eta, "delta_frac": rep["delta_frac"], "clipped": rep["clipped"],
                    "hebb_kl": kl(teacher, tr), "hebb_top1_lp": top1_lp(teacher, tr)[0]}
            p.revert()
            # norm-matched random control, same site, same magnitude
            R = torch.randn_like(dW); R *= dW.norm() / R.norm()
            p._site.write(p.W0 + R)
            rc = logits_at_q(toks(Q), n_q)
            p._site.write(p.W0)
            cell["rand_kl"] = kl(teacher, rc)
            cell["rand_top1_lp"] = top1_lp(teacher, rc)[0]
        row["sweep"].append(cell)
    results[name] = row

print(f"elapsed {time.time()-t0:.1f}s\n")
for name, row in results.items():
    print(f"== {name}: teacher top-1 {row['teacher_top1']!r} lp={row['teacher_top1_lp']:.2f} | "
          f"no-context KL={row['baseline_kl']:.3f} top1_lp={row['baseline_top1_lp']:.2f}")
    print(f"{'eta':>7} {'dW/W':>8} {'hebb KL':>8} {'rand KL':>8} {'hebb lp':>8} {'rand lp':>8}")
    for c in row["sweep"]:
        print(f"{c['eta']:7.0e} {c['delta_frac']:8.4f} {c['hebb_kl']:8.3f} {c['rand_kl']:8.3f} "
              f"{c['hebb_top1_lp']:8.2f} {c['rand_top1_lp']:8.2f}")
    print()
json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1)
