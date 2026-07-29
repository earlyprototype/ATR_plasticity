"""
Tests for `terms=`: composed, signed, optionally-projected update rules.

Issue #25's knob table ends on a row the code could not express: "Sign, per
subspace -- reinforce inside the target subspace, erode outside it", and its
summary, "pick a direction, restrict drift to it, reinforce it, erode around
it". Issue #24 step 2 -- the recorded primary experiment -- is "turn on the
Hebbian / anti-Hebbian balance". Neither was reachable. `mode` is one string and
`_hook` takes one branch, so a site carried reinforcement OR erosion; and the
two-instance workaround is closed by `MultiSitePlasticity._reject_overlap`,
which refuses two specs on the same rows for reasons that remain good. `terms=`
opens the rule up instead: a list of (primitive, sign, projector, scale), summed
per firing.

What this file has to establish, in order of what a wrong answer would cost:

  1. EQUIVALENCE. A `terms` list spelling out an existing mode reproduces that
     mode's delta BIT-EXACTLY. This is the load-bearing claim: it is what makes
     `terms` a generalisation of the rules rather than a second, subtly
     different implementation of them, and it is the only cheap check that the
     composed path computes the primitives the docstring names.
  2. THE BALANCE IS EXPRESSIBLE AND BEHAVES. `[+H P, -H (I-P), -D]` reinforces
     inside P, erodes outside it, and is still braked. The sign claim is made
     against the reinforcement direction the update was built from, and the
     brake claim the way `test_antihebbian.py` makes it -- on the weight norm
     over a run, with the failure arm alongside.
  3. DEGENERATE LIMITS. At P = I the balance IS the all-reinforce rule; at P = 0
     it IS the all-erode rule. These pin the semantics at both ends, and they
     are bit-exact for the same reason (1) is.
  4. GUARDS. Every way of writing a composed rule wrong fails loudly.
  5. THE SCAFFOLD SURVIVES. Ceiling, revert, nonfinite, report schema, and the
     driver, on the composed path.

BIT-EXACTNESS, and how it is measured. `terms=[+H]` and `mode="hebb"` differ
only by a multiplication by 1.0 and an addition of terms in the same order, so
they are bit-identical *given identical activations* -- which is a claim about
the rule, not about the model. Where a test asserts `torch.equal` it feeds both
arms THE SAME captured (x, y) pairs through `observe()`, the replay entry point
`offline_control.py` already uses, so the model's forward is not in the
comparison at all. One test drives both arms live as well, to show the hook
branch is wired to the same arithmetic; that one is measured to tolerance,
because two separate drives of a 124M-parameter model are only reproducible to
the precision of its own matmuls.

Weight hygiene as elsewhere: the model is session-scoped, every mutating test
reverts in a `finally`, and `conftest._target_weight_unchanged` is the backstop.
"""

from __future__ import annotations

import pytest
import torch

from conftest import D_MLP, D_MODEL, REPORT_TYPES
from multi_site import MultiSitePlasticity, SiteSpec
from plasticity import OjaPlasticity, TermSpec, subspace_projector
from test_plasticity import capture, drive


# Same eta as `test_antihebbian.TestAntiHebbRule`: large enough to resolve the
# delta in float32 at GPT-2's weight scale, small enough not to reach the
# ceiling at this site.
ETA = 1e-5

# Layer 11's attention output projection -- the block the parent found carrying
# the period-2 oscillation, and where the per-head sites the driver test uses
# live. Watched by `conftest._target_weight_unchanged`.
HF_ATTN = "transformer.h.11.attn.c_proj"


# --------------------------------------------------------------------------
# Local helpers -- deliberately independent of the code under test
# --------------------------------------------------------------------------

def hebb_term(x, y):
    return (x.transpose(0, 1) @ y) / x.shape[0]


def decay_term(y, w):
    return w @ ((y.transpose(0, 1) @ y) / y.shape[0])


def mean_over_firings(seen, fn):
    """Mean of `fn(x, y)` over hook firings, summed in firing order."""
    total = None
    for x, y in seen:
        term = fn(x, y)
        total = term if total is None else total + term
    return total / len(seen)


def rel(a, b) -> float:
    """||a - b||_F / ||b||_F in float64 -- the scale-free way to compare two
    updates, and the only honest one when differencing arms whose norms are two
    orders of magnitude apart."""
    return ((a - b).double().norm() / b.double().norm()).item()


def inner(a, b) -> float:
    """<a, b>_F in float64: positive means the two point the same way."""
    return (a.double() * b.double()).sum().item()


def w_at(gpt2, path: str) -> torch.Tensor:
    """The live weight at a dotted path, resolved without asking the code under
    test where it lives."""
    obj = gpt2
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj.weight


def directions(gpt2, k=1) -> torch.Tensor:
    """k unembedding rows -- one of the three things issue #25 names as a
    direction worth aiming at, so the projector is built from a real one rather
    than from noise that would make the test easier and the claim weaker."""
    return gpt2.lm_head.weight[:k].detach().clone()


def record(gpt2, site, r0, atr_step, n=2):
    """One drive's worth of (x, y) pairs at `site`, captured once so that every
    arm compared against it sees byte-identical inputs."""
    p = OjaPlasticity(gpt2, site=site, eta=0.0, mode="off")
    with capture(p.module) as seen, p:
        drive(gpt2, r0, atr_step, n=n)
    assert len(seen) == n
    return seen


def replay_delta(gpt2, site, seen, **kwargs):
    """Push a recording through the rule with no forward pass, via `observe()`.

    Nothing is installed and nothing is driven, so the ONLY thing that can
    differ between two calls is the rule under test. The weight is still written
    by `apply()`, so this reverts like any other arm.
    """
    p = OjaPlasticity(gpt2, site=site, eta=ETA, **kwargs)
    try:
        for x, y in seen:
            p.observe(x, y)
        rep = p.apply()
        return p.delta.clone(), rep
    finally:
        p.revert()


def driven_delta(gpt2, site, r0, atr_step, n=2, eta=ETA, **kwargs):
    """One live accumulate-and-apply cycle; the weight is handed back untouched."""
    p = OjaPlasticity(gpt2, site=site, eta=eta, **kwargs)
    try:
        with p:
            drive(gpt2, r0, atr_step, n=n)
        rep = p.apply()
        return p.delta.clone(), rep
    finally:
        p.revert()


# The composed spellings of the three modes that have a closed form. "random"
# and "off" are not rules over the primitives -- one is a noise arm built in
# apply(), the other is the absence of a rule -- so they have no term list and
# are deliberately absent here.
MODE_AS_TERMS = {
    "hebb": [TermSpec("hebb", +1)],
    "oja": [TermSpec("hebb", +1), TermSpec("decay", -1)],
    "anti_hebb": [TermSpec("hebb", -1), TermSpec("decay", -1)],
}


# --------------------------------------------------------------------------
# 1. Equivalence with the existing modes
# --------------------------------------------------------------------------

class TestEquivalenceWithTheModes:
    """
    If a composed rule that spells out `oja` is not `oja`, then `terms` computes
    something nobody wrote down, and every EXP-002 balance result is about an
    unnamed rule. This is the cheapest possible check of that and the strictest.
    """

    @pytest.mark.parametrize("mode", sorted(MODE_AS_TERMS))
    def test_terms_reproduce_the_mode_bit_exactly(self, gpt2, site, r0, atr_step, mode):
        """
        BIT-EXACT, not to tolerance, and by construction rather than by luck.

        `terms` computes `sum_i (sign_i * scale_i) * T_i`; at sign=+1, scale=1
        that is a multiplication by 1.0, which IEEE-754 leaves every float
        unchanged, and `a + (-b)` is defined as `a - b`. So the composed path
        performs the same operations on the same values in the same order as the
        mode branch, and the only way the two could differ is if they were
        handed different activations -- which is why both arms are replayed from
        ONE recording rather than driven twice.

        `torch.equal` and not `allclose` deliberately: a tolerance here would
        pass for a composed path that had picked up an extra rounding step, and
        that extra step is precisely the thing that would make `terms` a second
        implementation of the rules rather than the same one.
        """
        seen = record(gpt2, site, r0, atr_step)

        by_mode, rep_m = replay_delta(gpt2, site, seen, mode=mode)
        by_terms, rep_t = replay_delta(gpt2, site, seen, terms=MODE_AS_TERMS[mode])

        assert by_mode.double().norm().item() > 0        # or equality is vacuous
        assert rep_m["clipped"] is False                 # or both are a rescale
        assert rep_t["clipped"] is False
        assert torch.equal(by_terms, by_mode), (
            f"composed {mode} differs from mode={mode!r} by "
            f"{(by_terms - by_mode).abs().max().item():.3e}"
        )

    def test_the_hook_path_composes_the_same_rule_as_the_replay_path(
        self, gpt2, site, r0, atr_step
    ):
        """
        The equivalence above is asserted through `observe()`. This one drives
        the real forward hook, because `_hook` is where a composed run actually
        gets its activations and a branch wired only into `observe` would pass
        every test above and accumulate nothing in a real loop.

        To tolerance, not bit-exactly, and the tolerance is the model's: two
        separate drives of GPT-2 small agree to about float32 epsilon on the
        activations, so the deltas do too. The bit-exact claim about the
        arithmetic is made above, where the model is held fixed.
        """
        live, rep = driven_delta(gpt2, site, r0, atr_step, terms=MODE_AS_TERMS["oja"])
        by_mode, _ = driven_delta(gpt2, site, r0, atr_step, mode="oja")

        assert rep["n_applied"] == 1
        assert rep["clipped"] is False
        assert live.double().norm().item() > 0
        assert rel(live, by_mode) < 1e-6

    def test_a_scale_is_a_coefficient_on_one_term_and_nothing_else(
        self, gpt2, site, r0, atr_step
    ):
        """
        `scale` has to multiply its own term, not the update. Scaling the whole
        rule is what `eta` does, and a `scale` that leaked into the other terms
        would be an undocumented second learning rate -- the run would be at a
        different eta from the one the log records, which is README defect 1's
        shape exactly.

        Checked against the closed form: halving the Hebb term's scale must move
        the delta by precisely `-eta * <x y^T> / 2` and leave the decay term
        alone.
        """
        seen = record(gpt2, site, r0, atr_step)
        hebb = mean_over_firings(seen, hebb_term)

        full, _ = replay_delta(gpt2, site, seen, terms=MODE_AS_TERMS["oja"])
        halved, _ = replay_delta(gpt2, site, seen, terms=[
            TermSpec("hebb", +1, scale=0.5), TermSpec("decay", -1),
        ])

        assert rel(full - halved, ETA * 0.5 * hebb) < 1e-4
        # ...and it is a real change, not a rounding one.
        assert (full - halved).double().norm().item() > 0.0

    def test_the_default_path_is_untouched_by_the_new_argument(
        self, gpt2, site, r0, atr_step
    ):
        """
        `terms=None` must be the previous behaviour bit-for-bit, or every number
        already recorded in this repo is invalidated by an argument nobody
        passed. The same claim `test_plasticity.py` makes for `project=None`.
        """
        seen = record(gpt2, site, r0, atr_step)
        a, _ = replay_delta(gpt2, site, seen, mode="oja")
        b, _ = replay_delta(gpt2, site, seen, mode="oja", terms=None)
        assert torch.equal(a, b)

    def test_report_names_the_composed_rule_rather_than_a_mode_it_is_not(
        self, gpt2, site
    ):
        """
        A composed run is none of the five named rules, so a log row saying
        "oja" would be false -- and `mode` is the field every sweep groups by.
        `mode="oja"` is still what a caller may write alongside `terms` (it is
        indistinguishable from the default), so the object has to disagree with
        its own argument here, on purpose.
        """
        p = OjaPlasticity(gpt2, site=site, mode="oja", terms=MODE_AS_TERMS["oja"])
        assert p.mode == "terms"
        assert p.report()["mode"] == "terms"
        assert "terms" not in OjaPlasticity.VALID_MODES     # not a user-settable mode
        with pytest.raises(ValueError, match="mode must be one of"):
            OjaPlasticity(gpt2, site=site, mode="terms")


# --------------------------------------------------------------------------
# 2. The balance issue #25 asks for
# --------------------------------------------------------------------------

class TestTheBalanceIsExpressible:
    """
    Issue #25: "pick a direction, restrict drift to it, reinforce it, erode
    around it". Issue #24 step 2: "turn on the Hebbian / anti-Hebbian balance".
    Written out, that is

        [ +H P,  -H (I - P),  -D ]

    with the complement built explicitly rather than assumed. This class is the
    reason the parameter exists.
    """

    @staticmethod
    def balance(P, I):
        return [TermSpec("hebb", +1, P),
                TermSpec("hebb", -1, I - P),
                TermSpec("decay", -1)]

    def test_the_complement_of_a_projector_is_itself_a_projector(self, gpt2):
        """
        `I - P` carries the whole "erode outside it" half of the rule, and the
        constructor will reject anything that is not idempotent -- so if the
        complement were not a projection the balance would be unwritable. Cheap
        to state, and it is the step a reader is most likely to take on trust.
        """
        P = subspace_projector(directions(gpt2, k=2))
        Q = torch.eye(D_MODEL) - P

        assert torch.allclose(Q @ Q, Q, atol=1e-5)
        assert torch.allclose(P @ Q, torch.zeros_like(Q), atol=1e-5)   # disjoint
        assert torch.linalg.matrix_rank(Q.double(), rtol=1e-6).item() == D_MODEL - 2
        OjaPlasticity(gpt2, site="transformer.h.6.mlp.c_proj",
                      terms=[TermSpec("hebb", -1, Q)])                 # accepted

    def test_the_update_reinforces_inside_P_and_erodes_outside_it(
        self, gpt2, site, r0, atr_step
    ):
        """
        The claim, as a sign, measured against the direction the update was
        built from.

        The Hebbian half is isolated by differencing the balance against a
        brake-only arm driven identically: `[+H P, -H Q, -D] - [-D]` is exactly
        `eta (H P - H Q)`, and the decay term cancels because both arms see the
        same activations and the same effective weight (one apply, so W is W0
        throughout). Isolating it is not optional -- at this site `||W <y y^T>||`
        is ~110x `||<x y^T>||`, so the brake dominates the raw update in BOTH
        subspaces and a sign taken on the total would read "erode" everywhere,
        for a rule that is reinforcing inside P perfectly well.
        """
        P = subspace_projector(directions(gpt2, k=1))
        I = torch.eye(D_MODEL)
        Q = I - P
        seen = record(gpt2, site, r0, atr_step)
        hebb = mean_over_firings(seen, hebb_term)

        bal, rep = replay_delta(gpt2, site, seen, terms=self.balance(P, I))
        brake, _ = replay_delta(gpt2, site, seen, terms=[TermSpec("decay", -1)])
        reinforcement = bal - brake

        assert rep["clipped"] is False       # or the delta is a rescale, not the rule
        assert rep["nonfinite"] is False

        # THE CLAIM: opposite signs against the same reference direction.
        assert inner(reinforcement @ P, hebb @ P) > 0, "not reinforcing inside P"
        assert inner(reinforcement @ Q, hebb @ Q) < 0, "not eroding outside P"

        # ...and not merely opposite in sign: each half IS the Hebb term, signed.
        assert rel(reinforcement @ P, ETA * (hebb @ P)) < 1e-4
        assert rel(reinforcement @ Q, -ETA * (hebb @ Q)) < 1e-4
        # Both halves are actually present. Without this the test would pass for
        # a rule that had quietly dropped one of them into the numerical floor.
        assert (reinforcement @ P).double().norm().item() > 0
        assert (reinforcement @ Q).double().norm().item() > 0

    def test_the_brake_is_still_in_the_composed_update(self, gpt2, site, r0, atr_step):
        """
        The decay term must survive composition, exactly and identifiably.
        Dropping it is the silent failure: the rule still runs, still reports,
        and is now unbraked -- and at this site the brake is 110x the term it
        brakes, so its absence is not something a delta_frac would reveal on a
        short run.
        """
        P = subspace_projector(directions(gpt2, k=1))
        I = torch.eye(D_MODEL)
        seen = record(gpt2, site, r0, atr_step)
        # W0, snapshotted before any arm runs: no apply has happened during the
        # recording, so this is what a first-apply decay term is taken against.
        w0 = w_at(gpt2, site).detach().clone()
        decay = mean_over_firings(seen, lambda x, y: decay_term(y, w0))
        hebb = mean_over_firings(seen, hebb_term)

        with_brake, _ = replay_delta(gpt2, site, seen, terms=self.balance(P, I))
        without, _ = replay_delta(gpt2, site, seen, terms=self.balance(P, I)[:2])

        # The term the brake contributes is exactly `-eta W <y y^T>`, and it is
        # the whole difference between the two arms.
        assert rel(with_brake - without, -ETA * decay) < 1e-4
        assert (with_brake - without).double().norm().item() > 0.0
        # The brake dominates the term it brakes at this site (measured ~110x),
        # which is why (a) the reinforcement has to be isolated to read its sign
        # in the test above and (b) losing the brake would not show up as an
        # implausible delta_frac on a short run. Same discriminator as
        # `test_antihebbian.py` uses for the same reason.
        assert decay.double().norm().item() > 10 * hebb.double().norm().item()

    def test_the_balance_stays_bounded_where_a_flipped_brake_diverges(
        self, gpt2, site, r0, atr_step
    ):
        """
        Issue #25's explicit demand, for the composed rule: "the anti-Hebbian
        mode specifically needs a test that the brake is still braking -- that
        the weight norm stays bounded".

        Both directions, as `test_antihebbian.TestBoundedness` does it. `terms`
        hands the sign of the decay term to the caller, which is the point and
        also a new way to get it wrong: `TermSpec("decay", +1)` is the mistake
        `mode="anti_hebb"` exists to prevent, now writable on purpose. It must
        diverge, or "the brake is braking" is a statement about a small eta.

        The ceiling is lifted on both arms for the same reason
        `c3_divergence_demo` lifts it: at 5% it binds on both and hides exactly
        the difference being measured.
        """
        P = subspace_projector(directions(gpt2, k=1))
        I = torch.eye(D_MODEL)
        N, eta = 16, 3e-5

        def norm_trace(terms):
            p = OjaPlasticity(gpt2, site=site, eta=eta, terms=terms,
                              max_delta_frac=1e9)
            trace = []
            try:
                with p:
                    r = r0
                    for _ in range(N):
                        r = atr_step(gpt2, r)
                        p.apply()
                        trace.append(p.module.weight.double().norm().item())
                return trace, p.report()
            finally:
                p.revert()

        w0 = w_at(gpt2, site).double().norm().item()
        braked, rep_b = norm_trace(self.balance(P, I))
        flipped, rep_f = norm_trace(self.balance(P, I)[:2] + [TermSpec("decay", +1)])

        assert rep_b["nonfinite"] is False and rep_f["nonfinite"] is False
        assert rep_b["n_applied"] == rep_f["n_applied"] == N

        # Braked: never grows, at any point in the run, and stays percent-scale.
        assert max(braked) <= w0, f"the balance grew: {max(braked):.6g} > {w0:.6g}"
        assert braked[-1] < w0
        assert rep_b["delta_frac"] < 0.1

        # Flipped: grows without bound, orders of magnitude, not percent.
        assert flipped[-1] > 2.0 * w0
        assert all(flipped[i] < flipped[i + 1] for i in range(len(flipped) - 1))
        assert rep_f["delta_frac"] > 1.0
        assert flipped[-1] > 10.0 * braked[-1]


# --------------------------------------------------------------------------
# 3. Degenerate limits
# --------------------------------------------------------------------------

class TestDegenerateLimits:
    """
    The balance is a one-parameter family in P, and its two endpoints are rules
    that already exist. Pinning them is what makes "reinforce inside, erode
    outside" a statement about P rather than a description of whatever the code
    happens to do: at P = I there is no outside and the rule must BE `oja`; at
    P = 0 there is no inside and it must BE `anti_hebb`.

    Bit-exact for the same reason the equivalence tests are, with one extra
    step: a matmul by `I` or by `0` introduces no rounding, because every output
    element is a sum of exact zeros plus at most one exact product.
    """

    @pytest.mark.parametrize(
        "P_is, expect",
        [("identity", "oja"),         # no outside: all-reinforce
         ("zero", "anti_hebb")],      # no inside:  all-erode
    )
    def test_the_endpoints_are_the_rules_they_should_be(
        self, gpt2, site, r0, atr_step, P_is, expect
    ):
        I = torch.eye(D_MODEL)
        P = I if P_is == "identity" else torch.zeros(D_MODEL, D_MODEL)
        seen = record(gpt2, site, r0, atr_step)

        balanced, rep = replay_delta(gpt2, site, seen, terms=[
            TermSpec("hebb", +1, P),
            TermSpec("hebb", -1, I - P),
            TermSpec("decay", -1),
        ])
        reference, _ = replay_delta(gpt2, site, seen, mode=expect)

        assert reference.double().norm().item() > 0
        assert rep["clipped"] is False
        assert torch.equal(balanced, reference), (
            f"P={P_is} balance is not {expect}: max difference "
            f"{(balanced - reference).abs().max().item():.3e}"
        )

    def test_the_two_endpoints_are_not_the_same_rule(self, gpt2, site, r0, atr_step):
        """
        Both limits pass trivially if the projectors were being ignored, so the
        family has to be shown to have two distinct ends. It is the Hebb term,
        twice, that separates them -- the brake is identical in both.
        """
        I = torch.eye(D_MODEL)
        Z = torch.zeros(D_MODEL, D_MODEL)
        seen = record(gpt2, site, r0, atr_step)
        hebb = mean_over_firings(seen, hebb_term)

        at_I, _ = replay_delta(gpt2, site, seen, terms=[
            TermSpec("hebb", +1, I), TermSpec("hebb", -1, I - I),
            TermSpec("decay", -1)])
        at_0, _ = replay_delta(gpt2, site, seen, terms=[
            TermSpec("hebb", +1, Z), TermSpec("hebb", -1, I - Z),
            TermSpec("decay", -1)])

        assert not torch.equal(at_I, at_0)
        assert rel(at_I - at_0, 2.0 * ETA * hebb) < 1e-4

    def test_a_projector_on_every_term_composes_with_the_whole_update_one(
        self, gpt2, site, r0, atr_step
    ):
        """
        Per-term and whole-update projection are ALLOWED TOGETHER, and this is
        what that means: the per-term projectors shape each term as it is
        accumulated, the whole-update one is applied to the averaged sum in
        `apply()`, so the result is `(sum_i s_i (T_i P_i)) P`. Both are linear,
        so the composition is the obvious one -- but "obvious" is how an
        ordering bug survives, and the two projections are applied in different
        methods.

        Stated as an identity a reader can check: with the SAME projector in
        both places, the whole-update one is idempotent on the result and
        changes nothing.
        """
        P = subspace_projector(directions(gpt2, k=2))
        seen = record(gpt2, site, r0, atr_step)
        terms = [TermSpec("hebb", +1, P), TermSpec("decay", -1, P)]

        per_term, _ = replay_delta(gpt2, site, seen, terms=terms)
        both, _ = replay_delta(gpt2, site, seen, terms=terms, project=P)

        assert per_term.double().norm().item() > 0
        assert rel(both, per_term) < 1e-5
        # And it really was a restriction: the update lives inside P either way.
        assert rel(per_term @ P, per_term) < 1e-5

        # The other order is a real composition, not a no-op: projecting the
        # whole update onto a DIFFERENT subspace narrows it further.
        other = subspace_projector(gpt2.lm_head.weight[5:6].detach().clone())
        narrowed, _ = replay_delta(gpt2, site, seen, terms=terms, project=other)
        assert narrowed.double().norm().item() < per_term.double().norm().item()
        assert rel(narrowed, per_term) > 0.5


# --------------------------------------------------------------------------
# 4. Guards
# --------------------------------------------------------------------------

class TestGuards:
    """
    Every one of these is a composed rule that would otherwise run and report
    plausibly while computing something the caller did not ask for. They fail at
    construction, before a hook is installed, so a sweep dies on the offending
    cell rather than an hour into the run.
    """

    def test_a_wrongly_shaped_per_term_projector_is_refused(self, gpt2, site):
        """The input axis, not the output one -- conformable at this site only
        by accident of the widths, and silently a different experiment."""
        with pytest.raises(ValueError, match=r"terms\[0\]\.project must be"):
            OjaPlasticity(gpt2, site=site,
                          terms=[TermSpec("hebb", +1, torch.eye(D_MLP))])
        # The index names the offending term, not just "a projector".
        with pytest.raises(ValueError, match=r"terms\[1\]\.project must be"):
            OjaPlasticity(gpt2, site=site, terms=[
                TermSpec("hebb", +1),
                TermSpec("decay", -1, torch.eye(D_MODEL + 1)),
            ])

    @pytest.mark.parametrize(
        "bad",
        [torch.eye(D_MODEL) * 2.0,              # scaling, not projecting
         torch.ones(D_MODEL, D_MODEL)],
    )
    def test_a_per_term_matrix_that_is_not_a_projection_is_refused(
        self, gpt2, site, bad
    ):
        """
        The same check the whole-update projector already gets, because a
        per-term one has the same failure: `2I` is symmetric, square, correctly
        shaped and doubles its term, and nothing downstream would notice -- the
        run would simply weight that term twice as heavily as the log says.
        """
        with pytest.raises(ValueError, match="not idempotent"):
            OjaPlasticity(gpt2, site=site, terms=[TermSpec("hebb", +1, bad)])

    def test_an_empty_terms_list_is_refused(self, gpt2, site):
        """It would accumulate zero on every firing and report a run that did
        nothing -- indistinguishable in a log from `mode="off"` and from a site
        that never fired."""
        with pytest.raises(ValueError, match="terms= is empty"):
            OjaPlasticity(gpt2, site=site, terms=[])

    @pytest.mark.parametrize("mode", ["hebb", "anti_hebb", "random", "off"])
    def test_a_non_default_mode_alongside_terms_is_refused(self, gpt2, site, mode):
        """
        The ambiguity, failed loudly. Letting `terms` win silently would run a
        rule that the caller's own `mode=` argument contradicts, and `mode` is
        the field a sweep groups its results by -- the run would be filed under
        a rule it did not use.
        """
        with pytest.raises(ValueError, match="both specify the update rule"):
            OjaPlasticity(gpt2, site=site, mode=mode, terms=MODE_AS_TERMS["oja"])

    def test_the_default_mode_alongside_terms_is_the_one_accepted_case(
        self, gpt2, site
    ):
        """
        `mode="oja"` written out is indistinguishable from the default, so it
        cannot be rejected without rejecting `SiteSpec`'s own defaults -- which
        pass `mode` explicitly on every build. Documented as the precedence
        rule: terms win, and `report()["mode"]` says so.
        """
        p = OjaPlasticity(gpt2, site=site, mode="oja", terms=MODE_AS_TERMS["hebb"])
        assert p.mode == "terms"
        assert len(p.terms) == 1

    @pytest.mark.parametrize(
        "kwargs, match",
        [(dict(primitive="hebbian"), "primitive must be one of"),
         (dict(primitive="oja"), "primitive must be one of"),      # a mode, not a term
         (dict(primitive="hebb", sign=0), "sign must be exactly"),
         (dict(primitive="hebb", sign=2), "sign must be exactly"),
         (dict(primitive="hebb", sign=-0.5), "sign must be exactly"),
         (dict(primitive="hebb", scale=0.0), "scale must be a finite positive"),
         (dict(primitive="hebb", scale=-1.0), "scale must be a finite positive"),
         (dict(primitive="hebb", scale=float("inf")), "scale must be a finite positive")],
    )
    def test_a_malformed_term_is_refused_where_it_is_written(self, kwargs, match):
        """
        These need no model, so they fail at the point the spec is written
        rather than when a driver builds it. `sign=-0.5` is the dangerous one:
        it would work, and it would put a magnitude in the field the log reads
        as a direction.
        """
        with pytest.raises(ValueError, match=match):
            TermSpec(**kwargs)

    def test_terms_may_be_dicts_and_a_bad_entry_is_named(self, gpt2, site):
        """The experiment scripts build cells as dicts, exactly as `SiteSpec`
        already accepts; an unknown key must be a TypeError rather than a
        silently ignored field, or a typo'd `scal=` runs at scale 1."""
        p = OjaPlasticity(gpt2, site=site, terms=[
            {"primitive": "hebb", "sign": +1},
            {"primitive": "decay", "sign": -1},
        ])
        assert [t.primitive for t in p.terms] == ["hebb", "decay"]

        with pytest.raises(TypeError, match="each term must be a TermSpec"):
            OjaPlasticity(gpt2, site=site, terms=["hebb"])
        with pytest.raises(TypeError):
            OjaPlasticity(gpt2, site=site, terms=[{"primitive": "hebb", "scal": 2}])


# --------------------------------------------------------------------------
# 5. The scaffold, on the composed path
# --------------------------------------------------------------------------

class TestComposedRulesHonourTheScaffold:
    """
    A new rule shape that skips the ceiling, or leaves a residue after
    `revert()`, or reports a run that did not happen, is a new way to lose a
    sweep. These are the claims `test_antihebbian.py` makes for `anti_hebb`,
    asked of a composed rule -- and they should hold trivially, because the
    composed path rejoins the shared machinery at the accumulator.
    """

    @staticmethod
    def balance(gpt2):
        P = subspace_projector(directions(gpt2, k=1))
        I = torch.eye(D_MODEL)
        return [TermSpec("hebb", +1, P), TermSpec("hebb", -1, I - P),
                TermSpec("decay", -1)]

    def test_the_ceiling_binds(self, gpt2, site, r0, atr_step):
        """The guard against silently destroying the model is rule-independent,
        and the composed path is downstream of it."""
        p = OjaPlasticity(gpt2, site=site, eta=1e3, terms=self.balance(gpt2),
                          max_delta_frac=0.05)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
            rep = p.apply()

            assert rep["clipped"] is True
            assert rep["nonfinite"] is False
            assert rep["delta_frac"] <= 0.05 + 1e-6
            assert rep["delta_frac"] == pytest.approx(0.05, rel=1e-4)
            assert torch.isfinite(p.module.weight).all()
        finally:
            p.revert()

    def test_revert_restores_bit_exactly_and_clears_the_diagnostics(
        self, gpt2, site, r0, atr_step
    ):
        """
        Control C1 for the composed path. EXP-002 alternates arms; if one leaves
        either the matrix or the diagnostics dirty, every later arm is measured
        against a model the controls never gated.
        """
        w_before = w_at(gpt2, site).detach().clone()

        p = OjaPlasticity(gpt2, site=site, eta=1e3, terms=self.balance(gpt2),
                          max_delta_frac=1e-4)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
                dirty = p.apply()
            assert dirty["clipped"] is True          # the flags really were set
            assert dirty["n_applied"] == 1
            assert dirty["delta_norm"] > 0.0
            assert not torch.equal(p.module.weight, w_before)
        finally:
            p.revert()

        assert torch.equal(p.module.weight, w_before)
        clean = p.report()
        assert clean["clipped"] is False
        assert clean["nonfinite"] is False
        assert clean["n_applied"] == 0
        assert clean["delta_norm"] == 0.0
        assert clean["last_update_norm"] == 0.0

    def test_report_schema_and_diagnostics(self, gpt2, site, r0, atr_step):
        """
        The per-iteration log schema DESIGN.md specifies, unchanged by the new
        rule shape: same keys, same types, `mode` carrying the composed rule's
        name. A missing or wrongly-typed key breaks the log for a run that may
        take days.
        """
        p = OjaPlasticity(gpt2, site=site, eta=ETA, terms=self.balance(gpt2))
        try:
            with p:
                drive(gpt2, r0, atr_step, n=3)
            rep = p.apply()

            assert set(rep) == set(REPORT_TYPES)
            for key, typ in REPORT_TYPES.items():
                assert isinstance(rep[key], typ), f"{key}: {type(rep[key])}"
            assert rep["mode"] == "terms"
            assert rep["n_applied"] == 1
            assert rep["clipped"] is False
            assert rep["nonfinite"] is False
            assert rep["delta_norm"] > 0.0
            assert rep["delta_frac"] == pytest.approx(rep["delta_norm"] / p.W0_norm,
                                                      rel=1e-9)
            assert rep["last_update_norm"] == pytest.approx(rep["delta_norm"], rel=1e-5)
        finally:
            p.revert()

    def test_collection_does_not_touch_the_weight(self, gpt2, site, r0, atr_step):
        """
        Control C0 for the composed path: watching must not perturb. The rule
        branch runs inside the hook, so a composed rule that wrote from there
        rather than from `apply()` would break C0 for composed runs alone.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e-3, terms=self.balance(gpt2))
        w_before = p.module.weight.detach().clone()
        with p:
            drive(gpt2, r0, atr_step, n=3)
        assert torch.equal(p.module.weight, w_before)
        assert p._n_batches == 3

    def test_nonfinite_activations_are_rejected_not_absorbed(
        self, gpt2, site, r0, atr_step
    ):
        """
        The guard sits before the rule branch, so it has to hold for a composed
        rule too -- and a balance driven toward a degenerate state is exactly
        where non-finite activations show up.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1.0, terms=self.balance(gpt2))
        bad = r0.clone()
        bad[0, 1, 2] = float("nan")
        w_before = p.module.weight.detach().clone()
        try:
            with p:
                atr_step(gpt2, bad)
            assert p.nonfinite is True
            assert p._acc is None
            assert p._n_batches == 0
            rep = p.apply()
            assert rep["nonfinite"] is True
            assert torch.equal(p.module.weight, w_before)
        finally:
            p.revert()

    def test_the_brake_tracks_the_live_weight(self, gpt2, site, r0, atr_step):
        """
        A composed decay term must read `W0 + delta`, not `W0` -- the same
        property `test_antihebbian.py` pins for `anti_hebb`, and the reason a
        composed erosion rule is bounded rather than an affine drift with no
        feedback. Composition per firing is what buys this, and composition per
        apply() would silently lose it.
        """
        terms = [TermSpec("hebb", -1), TermSpec("decay", -1)]
        p = OjaPlasticity(gpt2, site=site, eta=ETA, terms=terms)
        try:
            with capture(p.module) as seen, p:
                r = drive(gpt2, r0, atr_step, n=2)
                p.apply()
                w_eff = p.W0 + p.delta         # what round two must decay against
                assert p.delta.norm().item() > 0
                seen.clear()
                after_first = p.delta.clone()
                drive(gpt2, r, atr_step, n=2)

            expected = mean_over_firings(
                seen, lambda x, y: -hebb_term(x, y) - decay_term(y, w_eff))
            rep = p.apply()
            assert rep["clipped"] is False     # or the comparison is against a rescale
            assert rel(p.delta - after_first, ETA * expected) < 1e-4
        finally:
            p.revert()

    def test_a_composed_rule_runs_inside_the_multi_site_driver(
        self, gpt2, r0, atr_step
    ):
        """
        `SiteSpec` has to carry `terms` through, or the balance is unreachable
        from the driver that EXP-002 will actually run -- and the driver is the
        reason it has to be per-site in the first place: `_reject_overlap`
        refuses the two-instance workaround, so ONE spec has to be able to
        reinforce and erode at once.

        Two sites, both composed, one a whole matrix and one a head stripe, on
        the matrices `conftest` watches.
        """
        P = subspace_projector(directions(gpt2, k=1))
        I = torch.eye(D_MODEL)
        balance = (TermSpec("hebb", +1, P), TermSpec("hebb", -1, I - P),
                   TermSpec("decay", -1))

        before = {
            "mlp": gpt2.transformer.h[6].mlp.c_proj.weight.detach().clone(),
            "attn": gpt2.transformer.h[11].attn.c_proj.weight.detach().clone(),
        }
        d = MultiSitePlasticity(gpt2, [
            SiteSpec("transformer.h.6.mlp.c_proj", eta=1e-6, terms=balance),
            SiteSpec(f"{HF_ATTN}.head.7", eta=1e-8, terms=balance),
        ])
        try:
            with d:
                drive(gpt2, r0, atr_step, n=2)
            rep = d.apply()

            assert rep["n_applied"] == 1
            assert [s["mode"] for s in rep["per_site"]] == ["terms", "terms"]
            assert rep["delta_norm"] > 0.0
            assert rep["clipped"] is False
            assert rep["nonfinite"] is False
            assert all(p.terms is not None and len(p.terms) == 3 for p in d)
            # Both matrices actually moved, and the head site moved only its own
            # 64 rows -- the disjointness the driver rests on, under a composed
            # rule as under a mode.
            assert not torch.equal(gpt2.transformer.h[6].mlp.c_proj.weight,
                                   before["mlp"])
            attn_now = gpt2.transformer.h[11].attn.c_proj.weight
            assert not torch.equal(attn_now[7 * 64:8 * 64], before["attn"][7 * 64:8 * 64])
            assert torch.equal(attn_now[:7 * 64], before["attn"][:7 * 64])
            assert torch.equal(attn_now[8 * 64:], before["attn"][8 * 64:])
        finally:
            d.revert()

        assert torch.equal(gpt2.transformer.h[6].mlp.c_proj.weight, before["mlp"])
        assert torch.equal(gpt2.transformer.h[11].attn.c_proj.weight, before["attn"])

    def test_the_driver_refuses_a_spec_whose_mode_and_terms_disagree(self, gpt2):
        """`SiteSpec` passes both through and lets `OjaPlasticity` be the one
        place the ambiguity is decided, so the driver fails at construction
        naming the offending site rather than dropping it."""
        with pytest.raises(ValueError, match="both specify the update rule"):
            MultiSitePlasticity(gpt2, [
                SiteSpec("transformer.h.6.mlp.c_proj", mode="anti_hebb",
                         terms=MODE_AS_TERMS["oja"]),
            ])
