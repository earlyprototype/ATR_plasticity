"""
Tests for `mode="anti_hebb"`, against real GPT-2 small.

The mode exists to *erode* what the loop has settled into (issue #25). The one
thing that can go wrong here and still look right is the sign of the decay term.
The rule is

    anti-Hebb   dW = -<x y^T> - W <y y^T>

and the naive spelling of the same intent -- `mode="oja"` with `eta < 0` --
computes

    negative eta  dW = -<x y^T> + W <y y^T>

which decorrelates, as wanted, and simultaneously deletes the only thing holding
the weight norm finite: `<y y^T>` is positive semi-definite, so `+W <y y^T>`
points along W. The two differ by `2 eta W <y y^T>` and by nothing else. They
have the same shape, the same report schema, the same cost, and on a short run
they look alike -- which is why the divergence is demonstrated here on real
weights rather than argued from the algebra alone.

`TestBoundedness` is the test this module exists for. The rest checks that the
new mode carries the guarantees the older modes already had: the ceiling, a
bit-exact `revert()`, the diagnostics, and the `transposed=True` flip.

Weight hygiene as elsewhere: the model is session-scoped, every mutating test
reverts in a `finally`, and `conftest._target_weight_unchanged` is the backstop.
"""

from __future__ import annotations

import pytest
import torch

from conftest import D_MLP, D_MODEL, REPORT_TYPES
from plasticity import OjaPlasticity
from test_plasticity import LinearProbe, capture, drive


# --------------------------------------------------------------------------
# Local helpers -- deliberately independent of the code under test
# --------------------------------------------------------------------------

def hebb_term(x, y):
    return (x.transpose(0, 1) @ y) / x.shape[0]


def decay_term(x, y, w):
    return w @ ((y.transpose(0, 1) @ y) / y.shape[0])


def anti_hebb_term(x, y, w):
    """The rule under test, written out once, in full, by hand."""
    return -hebb_term(x, y) - decay_term(x, y, w)


def mean_over_firings(seen, fn):
    """Mean of `fn(x, y)` over hook firings, summed in firing order."""
    total = None
    for x, y in seen:
        term = fn(x, y)
        total = term if total is None else total + term
    return total / len(seen)


def _weight_at(model, path: str) -> torch.Tensor:
    """The live weight at a dotted path, resolved without asking the module."""
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj.weight


def w_norm(model, path: str) -> float:
    """||W||_F in float64, the quantity the boundedness claim is about."""
    return _weight_at(model, path).double().norm().item()


def resolve_weight(model, path: str) -> torch.Tensor:
    return _weight_at(model, path).detach().clone()


# --------------------------------------------------------------------------
# 1. The rule
# --------------------------------------------------------------------------

class TestAntiHebbRule:

    # Same eta as TestLearningRule: large enough to resolve the delta in
    # float32 at GPT-2's weight scale, small enough not to reach the ceiling
    # (measured delta_frac 0.011 at this site).
    ETA = 5e-5

    def test_anti_hebb_update_matches_closed_form(self, gpt2, site, r0, atr_step):
        """
        dW = -<x y^T> - W <y y^T>, reconstructed from activations this test
        captured itself. A mismatch means the mode is some other rule, and every
        EXP-002 erosion result is about something nobody wrote down.
        """
        p = OjaPlasticity(gpt2, site=site, eta=self.ETA, mode="anti_hebb")
        try:
            with capture(p.module) as seen, p:
                drive(gpt2, r0, atr_step, n=3)

            # No apply yet, so the live weight is still W0 and that is what the
            # decay term must have been taken against.
            expected = mean_over_firings(seen, lambda x, y: anti_hebb_term(x, y, p.W0))
            rep = p.apply()

            assert len(seen) == 3
            assert torch.allclose(p.delta, self.ETA * expected, rtol=1e-5, atol=1e-9)
            assert torch.allclose(p.module.weight, p.W0 + self.ETA * expected,
                                  rtol=1e-6, atol=1e-9)
            assert not rep["clipped"]
            assert rep["mode"] == "anti_hebb"
            assert rep["n_applied"] == 1
        finally:
            p.revert()

    def test_anti_hebb_flips_the_reinforcement_term_and_only_that(
        self, gpt2, site, r0, atr_step
    ):
        """
        The claim of issue #25, stated as an identity between the two arms.

        anti-Hebb at +eta  =  eta * (-H - D)
        Oja at -eta        =  eta * (-H + D)

        so their difference is exactly `-2 eta D` -- the decay term, twice, and
        nothing else. If the mode were implemented as a negative learning rate
        the difference would be zero; if it flipped the decay term as well it
        would be `-2 eta H`, which at this site is ~100x smaller and would pass
        any tolerance loose enough to be safe.
        """
        p_a = OjaPlasticity(gpt2, site=site, eta=self.ETA, mode="anti_hebb")
        try:
            with capture(p_a.module) as seen, p_a:
                drive(gpt2, r0, atr_step, n=3)
            p_a.apply()
            delta_a = p_a.delta.clone()
        finally:
            p_a.revert()

        p_n = OjaPlasticity(gpt2, site=site, eta=-self.ETA, mode="oja")
        try:
            with p_n:
                drive(gpt2, r0, atr_step, n=3)
            p_n.apply()
            delta_n = p_n.delta.clone()
        finally:
            p_n.revert()

        decay = mean_over_firings(seen, lambda x, y: decay_term(x, y, p_a.W0))
        hebb = mean_over_firings(seen, hebb_term)

        assert torch.allclose(delta_a - delta_n, -2.0 * self.ETA * decay,
                              rtol=1e-4, atol=1e-9)
        assert not torch.allclose(delta_a, delta_n, rtol=1e-2, atol=1e-9)
        # The two failure directions this test discriminates, pinned as
        # magnitudes rather than assumed: the decay term dominates the Hebb term
        # at real weight scale, so "the difference is 2*eta*D" and "the
        # difference is 2*eta*H" are not confusable at any sane tolerance.
        assert decay.norm().item() > 10 * hebb.norm().item()

    def test_the_brake_tracks_the_live_weight(self, gpt2, site, r0, atr_step):
        """
        The decay term must be taken against W0 + delta, not against W0.

        This is *why* anti-Hebb is bounded rather than merely negative: the
        brake is proportional to the current weight, so it grows as the weight
        does and shrinks as it erodes. Frozen at W0 the rule is an affine drift
        with no feedback, and the boundedness test below would be measuring
        nothing but a small eta.
        """
        p = OjaPlasticity(gpt2, site=site, eta=self.ETA, mode="anti_hebb")
        try:
            with capture(p.module) as seen, p:
                r = drive(gpt2, r0, atr_step, n=2)
                p.apply()
                w_eff = p.W0 + p.delta          # what round two must decay against
                assert p.delta.norm().item() > 0
                seen.clear()
                delta_after_first = p.delta.clone()
                drive(gpt2, r, atr_step, n=2)

            expected = mean_over_firings(seen, lambda x, y: anti_hebb_term(x, y, w_eff))
            rep = p.apply()
            assert not rep["clipped"]           # or the comparison is against a rescale
            assert torch.allclose(p.delta - delta_after_first, self.ETA * expected,
                                  rtol=1e-4, atol=1e-9)
        finally:
            p.revert()


# --------------------------------------------------------------------------
# 2. Boundedness -- the reason the mode is not a negative eta
# --------------------------------------------------------------------------

class TestBoundedness:

    # Measured on GPT-2 small at transformer.h.6.mlp.c_proj, ||W0||_F = 164.86,
    # one forward per apply, ceiling lifted:
    #
    #   step             1       5      10      13      14      15      16
    #   anti_hebb   164.68  164.30  164.22  164.16  164.13  164.11  164.10
    #   oja, -eta   165.05  165.99  170.09  192.26  253.87  893.07  6.7e+04
    #
    # The negative-eta arm passes 1e10 by step 17 and is only stopped by the
    # (deliberately lifted) ceiling; anti-Hebb decays monotonically and is still
    # decaying at step 50 (163.6). N=16 is the first step at which the two are
    # separated by orders of magnitude rather than percent.
    ETA = 3e-5
    N = 16

    def _norm_trace(self, gpt2, site, r0, atr_step, eta, mode):
        """||W||_F after each of N applies, weights handed back untouched."""
        p = OjaPlasticity(gpt2, site=site, eta=eta, mode=mode,
                          # The ceiling would bind on both arms and hide exactly
                          # the difference being measured -- the same reason
                          # c3_divergence_demo lifts it.
                          max_delta_frac=1e9)
        trace = []
        try:
            with p:
                r = r0
                for _ in range(self.N):
                    r = atr_step(gpt2, r)
                    p.apply()
                    trace.append(w_norm(gpt2, site))
            return trace, p.report()
        finally:
            p.revert()

    def test_negative_eta_diverges_where_anti_hebb_stays_bounded(
        self, gpt2, site, r0, atr_step
    ):
        """
        The whole point of the mode, on real weights.

        Two arms at matched |eta|, one per spelling of "learn the other way".
        The naive spelling grows the weight without bound; the implemented one
        does not. If this ever fails in the direction of "both bounded", the
        anti-Hebbian mode has quietly become a negative learning rate and
        EXP-002's erosion runs are a divergence with a ceiling on it.
        """
        w0 = w_norm(gpt2, site)

        anti, rep_a = self._norm_trace(gpt2, site, r0, atr_step,
                                       eta=self.ETA, mode="anti_hebb")
        neg, rep_n = self._norm_trace(gpt2, site, r0, atr_step,
                                      eta=-self.ETA, mode="oja")

        # Neither arm fell over in a way that would make the comparison vacuous.
        assert rep_a["nonfinite"] is False
        assert rep_n["nonfinite"] is False
        assert rep_a["n_applied"] == rep_n["n_applied"] == self.N

        # The naive spelling grows, and not marginally.
        assert neg[-1] > 2.0 * w0, f"negative eta did not diverge: {neg[-1]:.4g}"
        assert all(neg[i] < neg[i + 1] for i in range(len(neg) - 1)), \
            "negative-eta arm did not grow monotonically"

        # The implemented one never grows at all, at any point in the run.
        assert max(anti) <= w0, f"anti_hebb grew: max {max(anti):.6g} > {w0:.6g}"
        assert anti[-1] < w0
        assert all(anti[i] > anti[i + 1] for i in range(len(anti) - 1)), \
            "anti_hebb arm is not monotonically eroding"

        # Bounded, not merely smaller: the drift is percent-scale where the
        # other arm is orders of magnitude.
        assert rep_a["delta_frac"] < 0.1
        assert rep_n["delta_frac"] > 1.0
        assert neg[-1] > 10.0 * anti[-1]


# --------------------------------------------------------------------------
# 3. The guarantees the other modes already carry
# --------------------------------------------------------------------------

class TestAntiHebbHonoursTheScaffold:
    """
    A new mode that skips the ceiling, or leaves a residue after revert(), or
    reports a run that did not happen, is a new way to lose a sweep. These are
    the same claims `test_plasticity.py` makes for "oja", asked of "anti_hebb".
    """

    def test_the_ceiling_binds(self, gpt2, site, r0, atr_step):
        """The guard against silently destroying the model is mode-independent."""
        p = OjaPlasticity(gpt2, site=site, eta=1e3, mode="anti_hebb",
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
        Control C1 for the new mode. EXP-002 alternates reinforcement and
        erosion arms; if the erosion arm leaves either the matrix or the
        diagnostics dirty, every later arm is measured against a model the
        controls never gated.
        """
        w_before = resolve_weight(gpt2, site)

        p = OjaPlasticity(gpt2, site=site, eta=1e3, mode="anti_hebb",
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
        The per-iteration log schema DESIGN.md specifies, for a mode that did
        not exist when it was written. A missing or wrongly-typed key breaks the
        log for a run that may take days.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e-5, mode="anti_hebb")
        try:
            with p:
                drive(gpt2, r0, atr_step, n=3)
            rep = p.apply()

            assert set(rep) == set(REPORT_TYPES)
            for key, typ in REPORT_TYPES.items():
                assert isinstance(rep[key], typ), f"{key}: {type(rep[key])}"
            assert rep["mode"] == "anti_hebb"
            assert rep["n_applied"] == 1
            assert rep["clipped"] is False
            assert rep["nonfinite"] is False
            assert rep["delta_norm"] > 0.0
            assert rep["delta_frac"] == pytest.approx(rep["delta_norm"] / p.W0_norm,
                                                      rel=1e-9)
            # First apply, so the accumulated delta is the single step.
            assert rep["last_update_norm"] == pytest.approx(rep["delta_norm"], rel=1e-5)
        finally:
            p.revert()

    def test_nonfinite_activations_are_rejected_not_absorbed(
        self, gpt2, site, r0, atr_step
    ):
        """
        A nan reaching the accumulator destroys the weight while the report
        still looks healthy. The guard sits before the rule branch, so it has to
        hold for this mode too -- and an erosion run driven toward a degenerate
        state is where non-finite activations are most likely to appear.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1.0, mode="anti_hebb")
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

    def test_off_still_beats_anti_hebb_to_the_weight(self, gpt2, site, r0, atr_step):
        """
        mode="off" is the C0/C1 instrument and must stay inert whatever else was
        added. Cheap to check, and the failure it catches -- a new branch that
        writes before the "off" early return -- is silent.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1.0, mode="off")
        w_before = p.module.weight.detach().clone()
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
                rep = p.apply()
            assert torch.equal(p.module.weight, w_before)
            assert rep["n_applied"] == 0
        finally:
            p.revert()


# --------------------------------------------------------------------------
# 4. transposed=True
# --------------------------------------------------------------------------
# No GPT-2 site is an nn.Linear, so this uses `test_plasticity.LinearProbe` --
# a code-path fixture, not a model, and non-square precisely so a missing
# transpose cannot hide. See the note above it there.

def test_transposed_anti_hebb_is_written_in_the_weight_layout(r0):
    """
    The flip in apply() is downstream of the rule branch, so it either works for
    every mode or the branch order is wrong. Against a square nn.Linear an
    unflipped write is silent -- it learns a scrambled update and raises
    nothing -- which is README defect 1 in the shape it took before it was fixed.
    """
    torch.manual_seed(1234)
    probe = LinearProbe()
    probe.eval()
    probe.requires_grad_(False)

    def step(model, r):
        with torch.no_grad():
            out = model(r)
        return out / (out.norm() + 1e-12)

    p = OjaPlasticity(probe, site="proj", eta=1.0, mode="anti_hebb",
                      transposed=True)
    w_before = p.module.weight.detach().clone()
    with capture(p.module) as seen, p:
        drive(probe, r0, step, n=3)

    # The rules run in (n_in, n_out); `transposed=True` stores (n_out, n_in),
    # so the decay term sees the flipped view and the write is flipped back.
    expected = mean_over_firings(
        seen, lambda x, y: anti_hebb_term(x, y, p.W0.transpose(0, 1))
    )
    rep = p.apply()

    assert p.W0.shape == (D_MLP, D_MODEL)
    assert rep["clipped"] is False      # or delta is a rescale, not the update
    assert torch.allclose(p.delta, expected.transpose(0, 1), rtol=1e-4, atol=1e-9)
    assert p.module.weight.shape == w_before.shape
    assert torch.isfinite(p.module.weight).all()

    p.revert()
    assert torch.equal(p.module.weight, w_before)
