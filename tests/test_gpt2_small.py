"""
The real-GPT-2 layer of the suite.

Everything else in `tests/` runs against the toy network in `conftest.py`. The
toy's `Conv1D` is our own reimplementation of HuggingFace's, which means the one
thing the toy structurally cannot check is whether that reimplementation is
right. If HF's `Conv1D` stored `(n_out, n_in)`, or applied `W.T`, every existing
test would still pass and every learning rule in `plasticity.py` would be
transposed against real weights. These tests close that gap, and then re-run the
repo's own gates (C0-C3) on the 124M-parameter model the experiment is actually
for.

Marked `slow` in full: a bare `pytest` never reaches this file, never imports
`transformers`, and never touches the HuggingFace cache. `transformers` is
imported inside fixtures for that reason, not at module scope.

The model fixture is SESSION-scoped -- 124M parameters is not something to
reload per test. That makes weight hygiene a correctness requirement rather than
a courtesy: a test that leaves `transformer.h.6.mlp.c_proj` modified silently
re-runs every later test against a different model. `_target_weight_unchanged`
below is autouse and fails the test that does it.

Run with:  .venv/bin/python -m pytest tests/test_gpt2_small.py -m slow -q
"""

from __future__ import annotations

import math

import pytest
import torch

from conftest import Conv1D as ToyConv1D
from controls import c0_identity, c1_revert, c2_random_direction, c3_divergence_demo
from plasticity import OjaPlasticity, candidate_sites

pytestmark = pytest.mark.slow


# The site the README nominates as the first target: MLP down-projection,
# mid-stack (layer 6 of 12).
SITE = "transformer.h.6.mlp.c_proj"

N_LAYER = 12
D_MODEL = 768
D_MLP = 4 * D_MODEL      # 3072

REPORT_TYPES = {
    "site": str,
    "mode": str,
    "eta": float,
    "n_applied": int,
    "delta_norm": float,
    "delta_frac": float,
    "last_update_norm": float,
    "clipped": bool,
    "nonfinite": bool,
}


def resolve(model, path: str):
    """Resolve a dotted site path independently of `OjaPlasticity._resolve`."""
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
# These deliberately shadow the toy-scale `site`, `r0` and `atr_step` fixtures
# in conftest.py: same names, same contracts, real dimensions.


@pytest.fixture(scope="session")
def gpt2():
    """
    GPT-2 small, loaded once for the whole session, frozen and in eval mode.

    Skips rather than fails when `transformers` is absent or the weights cannot
    be materialised (no network, cold cache). A red suite on a machine that was
    never going to have the model tells you nothing about the code.
    """
    # exc_type=ImportError so a broken install (present, but its own imports
    # fail) skips like an absent one rather than erroring the whole class.
    transformers = pytest.importorskip("transformers", exc_type=ImportError)
    try:
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    except (OSError, ImportError) as exc:      # incl. hub HTTP errors (OSError)
        pytest.skip(f"GPT-2 small not loadable offline: {type(exc).__name__}: {exc}")
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="session")
def hf_conv1d():
    """`transformers.pytorch_utils.Conv1D` -- the class conftest's toy mirrors."""
    pytorch_utils = pytest.importorskip("transformers.pytorch_utils", exc_type=ImportError)
    return pytorch_utils.Conv1D


@pytest.fixture(scope="session")
def site() -> str:
    return SITE


@pytest.fixture(scope="session")
def r0() -> torch.Tensor:
    """A residual-stream state at GPT-2 small's real width: (1, 4, 768)."""
    g = torch.Generator().manual_seed(7)
    r = torch.randn(1, 4, D_MODEL, generator=g)
    return r / r.norm()


@pytest.fixture(scope="session")
def atr_step():
    """
    TEST DOUBLE for the parent project's engine, not an ATR implementation.

    `plasticity.py` and `README` are both explicit that the real loop must be
    imported from the ATR repo and never reimplemented here; this is a
    deterministic, side-effect-free one-step map with the required signature
    `atr_step(model, r) -> r_next`, and nothing more. Any trajectory difference
    a test sees therefore comes from the plasticity layer alone.
    """

    def _step(model, r: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = model.transformer(inputs_embeds=r).last_hidden_state
        return out / (out.norm() + 1e-12)

    return _step


@pytest.fixture(autouse=True)
def _target_weight_unchanged(request):
    """
    Session-scoped model plus mutating tests equals cross-test contamination.

    Every test in this file must hand the model back exactly as it found it. A
    failure here is not a failure of the test that trips it in isolation -- it
    means that test was corrupting every test that ran after it.

    Guards only tests that actually asked for the model, so that the tests
    which need nothing but the `transformers` package -- the toy-Conv1D
    equivalence check in particular -- still run on a cold cache.
    """
    if "gpt2" not in request.fixturenames:
        yield
        return
    model = request.getfixturevalue("gpt2")
    site = request.getfixturevalue("site")
    w = resolve(model, site).weight
    before = w.detach().clone()
    yield
    assert torch.equal(w, before), f"{site} left modified for subsequent tests"


# --------------------------------------------------------------------------
# 1. The convention the toy asserts, checked against the real class
# --------------------------------------------------------------------------

class TestConv1DConvention:
    """
    `plasticity.py`'s rules are written for `y = x @ W` with W of shape
    `(n_in, n_out)`. That is HuggingFace's Conv1D and the transpose of
    `nn.Linear`. Everything downstream rests on it.
    """

    def test_real_site_is_a_conv1d_with_n_in_by_n_out_weight(self, gpt2, site, hf_conv1d):
        """
        The premise of the whole repo. If GPT-2's c_proj were `(n_out, n_in)`,
        `transposed=False` would be wrong at every site and the Oja decay term
        `W <y y^T>` would not even be conformable.
        """
        mod = resolve(gpt2, site)
        assert isinstance(mod, hf_conv1d)
        assert mod.weight.shape == (D_MLP, D_MODEL)     # (n_in, n_out)
        assert mod.bias.shape == (D_MODEL,)
        assert mod.weight.dtype == torch.float32
        # Non-square, so the two conventions are distinguishable by shape.
        assert mod.weight.shape[0] != mod.weight.shape[1]

    def test_forward_is_x_at_w_plus_b(self, gpt2, site, r0, atr_step):
        """
        The hook reads `inputs[0]` as x and `output` as y and forms `<x y^T>`.
        That is only the outer product of pre- and post-synaptic activity if the
        module really computes `x @ W + b` in this layout. Checked on activations
        from a real forward pass, not synthetic ones.
        """
        mod = resolve(gpt2, site)
        seen = []
        handle = mod.register_forward_hook(
            lambda _m, i, o: seen.append((i[0].detach().clone(), o.detach().clone()))
        )
        try:
            atr_step(gpt2, r0)
        finally:
            handle.remove()

        assert len(seen) == 1
        x, y = seen[0]
        assert x.shape[-1] == D_MLP and y.shape[-1] == D_MODEL
        manual = x @ mod.weight + mod.bias
        assert torch.allclose(manual, y, rtol=1e-5, atol=1e-5)

    def test_toy_conv1d_matches_huggingface_exactly(self, hf_conv1d):
        """
        THE TEST THAT JUSTIFIES THE TOY MODEL.

        `tests/conftest.py` reimplements Conv1D so the default suite stays
        offline. That reimplementation is an unchecked assumption in every other
        test file: if it diverged from HuggingFace's -- transposed weight,
        `W.T @ x`, bias on the wrong axis -- the toy suite would stay green
        while `plasticity.py` was systematically wrong on real GPT-2. Given the
        same weights and the same input the two forwards must agree bit for bit.

        Deliberately non-square (5 -> 8) so a transpose cannot hide.
        """
        n_in, n_out = 5, 8
        toy = ToyConv1D(n_out, n_in)
        real = hf_conv1d(n_out, n_in)

        # Same constructor argument order, same stored layout.
        assert toy.weight.shape == real.weight.shape == (n_in, n_out)
        assert toy.bias.shape == real.bias.shape == (n_out,)

        g = torch.Generator().manual_seed(11)
        w = torch.randn(n_in, n_out, generator=g)
        b = torch.randn(n_out, generator=g)
        with torch.no_grad():
            for m in (toy, real):
                m.weight.copy_(w)
                m.bias.copy_(b)

        x = torch.randn(2, 3, n_in, generator=g)
        with torch.no_grad():
            # Bit-exact is the right bar here: both run the same torch.addmm
            # on the same shapes, so any difference is a real divergence.
            assert torch.equal(toy(x), real(x))
            # ...and both are the closed form the learning rules assume. This
            # one is allclose, not equal: addmm is a fused kernel and
            # `x @ w + b` is matmul-then-add, so the two are not guaranteed
            # bit-identical across backends or thread counts -- the same
            # nondeterminism class as the C0 note above.
            assert torch.allclose(real(x), (x @ w + b), rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# 2. Site machinery against the real module tree
# --------------------------------------------------------------------------

class TestCandidateSites:

    def test_returns_forty_eight_sites_on_gpt2_small(self, gpt2, site):
        """
        12 layers x {attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj}. A count
        other than 48 means the enumerator is either missing targets the
        operator is told to choose from, or offering ones that are not
        plasticity targets at all.
        """
        sites = candidate_sites(gpt2)
        assert len(sites) == 48
        assert len(set(sites)) == 48
        assert site in sites
        expected = [
            f"transformer.h.{i}.{m}"
            for i in range(N_LAYER)
            for m in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
        ]
        assert sites == expected

    def test_every_site_resolves_to_a_two_d_weight(self, gpt2):
        """
        README sends the operator to this list to pick a site. An entry that
        does not survive `OjaPlasticity`'s own 2-D check is a TypeError waiting
        at construction time.
        """
        for s in candidate_sites(gpt2):
            mod = resolve(gpt2, s)
            assert mod.weight.dim() == 2
            assert OjaPlasticity(gpt2, site=s).W0.shape == mod.weight.shape

    def test_layernorms_embeddings_and_one_d_parameters_are_excluded(self, gpt2):
        """
        A LayerNorm gain is 1-D: `<x y^T>` against it is not a matrix and the
        rule is meaningless. The embeddings are 2-D and would pass a naive
        filter, but they sit outside the block stack.
        """
        sites = candidate_sites(gpt2)
        assert not any(".ln_" in s for s in sites)
        assert "transformer.ln_f" not in sites
        assert "transformer.wte" not in sites and "transformer.wpe" not in sites
        assert not any(s == "lm_head" for s in sites)
        for name, mod in gpt2.named_modules():
            w = getattr(mod, "weight", None)
            if torch.is_tensor(w) and w.dim() != 2:
                assert name not in sites
        # Biases are parameters, not modules, so they cannot appear at all.
        assert not any(s.endswith(".bias") for s in sites)

    def test_c_attn_is_offered_but_documented_as_the_one_to_avoid(self, gpt2):
        """
        `attn.c_attn` packs Q, K and V into one (768, 2304) matrix, so a
        Hebbian update there is three experiments at once. It is a legitimate
        2-D site and is listed; the warning has to live in the documentation,
        and if that warning is ever deleted the operator loses the only signal
        that the default-looking choice is the wrong one.
        """
        sites = candidate_sites(gpt2)
        c_attn = [s for s in sites if s.endswith("attn.c_attn")]
        assert len(c_attn) == N_LAYER
        assert resolve(gpt2, c_attn[0]).weight.shape == (D_MODEL, 3 * D_MODEL)
        doc = candidate_sites.__doc__ or ""
        assert "attn.c_attn" in doc and "Avoid" in doc

    def test_site_resolves_through_the_numeric_modulelist_index(self, gpt2, site):
        """
        `transformer.h.6` indexes an `nn.ModuleList`. If the numeric part were
        treated as an attribute, the entire site-addressing scheme would be
        unusable on the real model regardless of what the toy does.
        """
        p = OjaPlasticity(gpt2, site=site)
        assert p.module is gpt2.transformer.h[6].mlp.c_proj
        assert p.W0.shape == (D_MLP, D_MODEL)
        assert p.W0.data_ptr() != p.module.weight.data_ptr()
        assert p.W0_norm > 0


# --------------------------------------------------------------------------
# 3. C0 -- the gate, on real weights
# --------------------------------------------------------------------------

class TestC0OnRealWeights:

    def test_c0_identity_is_bit_exact(self, gpt2, r0, atr_step, site):
        """
        THE GATE. README: "A green suite is not C0 passing. Control C0 must
        still pass bit-exactly against real weights before anything here
        produces a result worth recording." This is that run.

        Failure means installing the hooks perturbs the trajectory at eta=0,
        i.e. every plasticity result is confounded with an instrumentation
        artefact and nothing downstream is interpretable.

        KNOWN INTERMITTENCY, recorded so a future flake is recognised rather
        than re-investigated from scratch: this assertion has been observed to
        fail twice on the development machine, with max deviations of 8.6e-05
        and 6.3e-05. It could not be reproduced in 80 controlled repeats (0/20
        hooked, 0/20 unhooked, on a quiet machine and under CPU load, plus 16
        cold fresh processes), and an unhooked-vs-unhooked control never
        differed -- so the hooks provably do not mutate anything. Best current
        explanation is nondeterministic parallel float reduction order in the
        CPU BLAS, not contamination by this repo. The documented contract is
        bit-exactness, so the assertion stays strict and there is no retry: if
        it fires, reproduce it against an unhooked-vs-unhooked control before
        suspecting `plasticity.py`.
        """
        res = c0_identity(gpt2, r0, atr_step, site=site, n_iter=3)
        assert res["control"] == "C0_identity"
        assert res["max_abs_deviation"] == 0.0
        assert res["bit_exact"] is True
        assert res["verdict"] == "PASS"


# --------------------------------------------------------------------------
# 4. apply() / revert() on real weights
# --------------------------------------------------------------------------

class TestApplyRevert:

    def test_apply_moves_the_weight_and_revert_restores_it_bit_exactly(
        self, gpt2, r0, atr_step, site
    ):
        """
        The read/write half of the scaffold on a 3072x768 matrix that took real
        compute to produce. `apply()` must actually move it, `report()` must
        describe the move, and `revert()` must put back the original bits --
        not something within float tolerance of them. Anything less and the
        session-scoped model is quietly different for every later test, and in
        a real sweep the C2 arms would not start from the same weights.
        """
        w_before = resolve(gpt2, site).weight.detach().clone()

        # try/finally, not a trailing revert(): the model fixture is
        # session-scoped, so an assertion that fails partway would otherwise
        # hand every later test in this file a modified GPT-2.
        p = OjaPlasticity(gpt2, site=site, eta=1e-5, mode="oja")
        try:
            with p:
                r = r0
                for _ in range(3):
                    r = atr_step(gpt2, r)
                rep = p.apply()

            assert set(rep) == set(REPORT_TYPES)
            for key, typ in REPORT_TYPES.items():
                assert isinstance(rep[key], typ), f"{key}: {type(rep[key])}"
            assert rep["site"] == site
            assert rep["mode"] == "oja"
            assert rep["n_applied"] == 1
            assert rep["nonfinite"] is False
            assert rep["clipped"] is False
            assert 0.0 < rep["delta_frac"] < 0.05
            assert math.isfinite(rep["delta_norm"]) and rep["delta_norm"] > 0
            assert rep["delta_frac"] == pytest.approx(
                rep["delta_norm"] / p.W0_norm, rel=1e-9
            )

            w_after = resolve(gpt2, site).weight
            assert not torch.equal(w_after, w_before)
            assert torch.isfinite(w_after).all()
        finally:
            p.revert()

        assert torch.equal(resolve(gpt2, site).weight, w_before)
        assert torch.equal(p.delta, torch.zeros_like(p.W0))
        assert p.report()["delta_norm"] == 0.0

    def test_trajectory_returns_to_baseline_after_revert(self, gpt2, r0, atr_step, site):
        """
        Restoring the weight is necessary but not sufficient: C1's claim is
        about the *trajectory*. If the iterated map does not come back, state
        is accumulating somewhere other than the weight and no A/B comparison
        in this repo is trustworthy.
        """
        def run(n=2):
            r = r0
            for _ in range(n):
                r = atr_step(gpt2, r)
            return r

        baseline = run()

        p = OjaPlasticity(gpt2, site=site, eta=1e-4, mode="oja")
        try:
            with p:
                r = r0
                for _ in range(2):
                    r = atr_step(gpt2, r)
                    p.apply()
            perturbed = run()
            assert not torch.equal(perturbed, baseline)  # the run was non-vacuous
        finally:
            # Same reason as above: a failed assertion must not leave the
            # session-scoped model perturbed for whatever runs next.
            p.revert()

        assert torch.equal(run(), baseline)


# --------------------------------------------------------------------------
# 5. C2's norm match at real activation scale
# --------------------------------------------------------------------------

class TestC2NormMatch:

    def test_random_is_norm_matched_to_oja_on_real_weights(self, gpt2, r0, atr_step, site):
        """
        README defect 2: `mode="random"` was norm-matched to the raw Hebb term
        rather than to Oja, and the bias grew with weight and activation
        magnitude -- so real GPT-2 scale is exactly where it mattered and the
        toy is exactly where it hides. At this site the Oja decay term is not a
        small correction to the Hebb term, it dominates it (see TestC3), so a
        regression would mismatch the two arms by orders of magnitude, not
        percent.

        C2 is the control README calls decisive: if the arms differ in
        magnitude, the comparison of Oja against a random direction is void.
        """
        norms = {}
        for mode in ("oja", "random"):
            p = OjaPlasticity(gpt2, site=site, eta=1e-6, mode=mode, seed=0)
            with p:
                r = r0
                for _ in range(3):
                    r = atr_step(gpt2, r)
                norms[mode] = p.apply()["delta_norm"]
            p.revert()

        assert norms["oja"] > 0
        assert norms["random"] == pytest.approx(norms["oja"], rel=1e-5)

    def test_cumulative_delta_frac_of_the_two_arms_is_not_norm_matched(
        self, gpt2, r0, atr_step, site
    ):
        """
        The norm match above is PER APPLY. `c2_random_direction` applies every
        iteration, so the two arms' *cumulative* delta_frac must not match: Oja
        steps point the same way and add linearly, independent random steps add
        in quadrature, giving random/oja ~ 1/sqrt(n_applied).

        Recorded as a test because the gap looks exactly like the defect README
        lists as #2 and is not one. Deleting the fix would show up in the
        single-apply test, not here; "fixing" this one would break the control.
        """
        w_before = resolve(gpt2, site).weight.detach().clone()
        res = c2_random_direction(gpt2, r0, atr_step, site=site, eta=1e-6, n_iter=2)

        assert res["control"] == "C2_random_direction"
        assert res["delta_frac_oja"] > 0 and res["delta_frac_random"] > 0
        assert math.isfinite(res["cos_oja_vs_random_final"])
        ratio = res["delta_frac_random"] / res["delta_frac_oja"]
        assert 0.6 < ratio < 0.85          # 1/sqrt(2) = 0.707; measured 0.717

        # Both arms must hand the weights back -- the second arm has to start
        # from the original matrix or the comparison is between two conditions
        # that differ by more than the update direction.
        assert torch.equal(resolve(gpt2, site).weight, w_before)


# --------------------------------------------------------------------------
# 6. The ceiling on a real matrix
# --------------------------------------------------------------------------

class TestCeiling:
    """
    Real weight norms are orders of magnitude away from the toy's, and the
    ceiling is a *fraction*, so it is worth confirming the guard binds here too.
    """

    def test_large_eta_is_clipped_at_the_ceiling(self, gpt2, r0, atr_step, site):
        """
        The ceiling "is the guard against silently destroying the model". An
        unenforced ceiling on a 124M-parameter model is a wrecked matrix and a
        report that still looks healthy.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e3, mode="oja", max_delta_frac=0.05)
        try:
            with p:
                r = r0
                for _ in range(2):
                    r = atr_step(gpt2, r)
                rep = p.apply()

            assert rep["clipped"] is True
            assert rep["nonfinite"] is False
            # Held to float32 precision, not exactly. Rescaling a 3072x768
            # float32 matrix by a float64 factor leaves the norm ~2.7e-8
            # relative above the ceiling, and correcting again is a no-op --
            # the correction factor rounds to 1.0 in float32. Same tolerance
            # the toy suite uses, so neither test passes by luck of matrix size.
            assert rep["delta_frac"] <= 0.05 + 1e-6
            assert rep["delta_frac"] == pytest.approx(0.05, rel=1e-4)
            assert torch.isfinite(p.module.weight).all()
        finally:
            p.revert()

    def test_tiny_eta_is_not_clipped(self, gpt2, r0, atr_step, site):
        """A ceiling that fires in the safe regime sends the operator chasing a
        regime that is not there, and caps an eta sweep before it begins."""
        p = OjaPlasticity(gpt2, site=site, eta=1e-9, mode="oja", max_delta_frac=0.05)
        try:
            with p:
                r = r0
                for _ in range(2):
                    r = atr_step(gpt2, r)
                rep = p.apply()

            assert rep["clipped"] is False
            assert 0.0 < rep["delta_frac"] < 0.05
        finally:
            p.revert()


# --------------------------------------------------------------------------
# 7. C1 on real weights
# --------------------------------------------------------------------------

class TestC1OnRealWeights:

    def test_c1_revert_is_bit_exact_and_non_vacuous(self, gpt2, r0, atr_step, site):
        """
        C1 as the control module runs it, end to end on real weights. The
        `delta_frac_before_revert > 0` assertion is what stops this passing
        vacuously: a run that never moved the weight would report bit-exact
        restoration while testing nothing.
        """
        res = c1_revert(gpt2, r0, atr_step, site=site, eta=1e-4, n_iter=3)
        assert res["control"] == "C1_revert"
        assert res["delta_frac_before_revert"] > 0.0
        assert res["max_abs_deviation_after_revert"] == 0.0
        assert res["bit_exact"] is True


# --------------------------------------------------------------------------
# 8. C3 -- Hebb vs Oja on real weights
# --------------------------------------------------------------------------
# Measured on GPT-2 small at this site, eta from 1e-6 to 1e-3, n_iter 3-8: the
# Hebb trace grows monotonically (super-linearly once eta >= 1e-3), while the
# Oja trace saturates -- at eta=1e-3 it *falls* after the first apply and then
# sits at ~0.17, which is the decay term pulling the weight to Oja's fixed
# point. That is the C3 claim.
#
# Two things the toy did not predict, both consequences of real weight scale:
#   * Oja's delta_frac is ~20-100x LARGER than Hebb's in absolute terms at
#     every stable eta. ||W||_F ~ 30 here, so W<y y^T> dominates <x y^T>
#     instead of correcting it. "Hebb diverges, Oja does not" is a statement
#     about growth over the run, not about which update is bigger.
#   * The claim has an upper eta bound. At eta >= 3e-3 the Oja arm itself
#     explodes (delta_frac 1e9 within three applies) because eta*lambda_max of
#     <y y^T> leaves the stable range. eta=1e-3 is inside the window; 3e-3 is
#     not.


class TestC3OnRealWeights:

    ETA = 1e-3
    N_ITER = 4

    def test_hebb_grows_while_oja_saturates(self, gpt2, r0, atr_step, site):
        """
        README: "`mode='hebb'` is included so you can produce the divergence
        figure (control C3) rather than asserting it." This is the figure, in
        assertion form, on real weights. Failure means the decay term is not
        doing its job -- the rule this repo argues for is not distinguishable
        from the one it is supposed to improve on, and the Oja-vs-Hebb framing
        has no support.
        """
        res = c3_divergence_demo(
            gpt2, r0, atr_step, site=site, eta=self.ETA, n_iter=self.N_ITER
        )
        assert res["control"] == "C3_divergence_demo"
        hebb = res["delta_frac_traces"]["hebb"]
        oja = res["delta_frac_traces"]["oja"]
        assert len(hebb) == len(oja) == self.N_ITER
        assert all(v > 0 for v in hebb) and all(v > 0 for v in oja)

        # Hebb: monotone and compounding -- no fixed point.
        assert all(hebb[i] < hebb[i + 1] for i in range(len(hebb) - 1))
        hebb_growth = hebb[-1] / hebb[0]
        assert hebb_growth > self.N_ITER - 1     # super-linear accumulation

        # Oja: bounded. The decay term pulls it back after the first apply.
        oja_growth = oja[-1] / oja[0]
        assert oja_growth < 1.0
        assert max(oja) / oja[0] <= 1.0 + 1e-6

        assert hebb_growth > 4 * oja_growth

    def test_c3_restores_the_weights_despite_running_without_a_ceiling(
        self, gpt2, r0, atr_step, site
    ):
        """
        C3 sets `max_delta_frac=1e9` deliberately. An un-reverted arm therefore
        hands the next arm -- and, with a session-scoped model, every later
        test -- a wrecked matrix. `_target_weight_unchanged` guards this file
        generally; this asserts it for the one control that runs unguarded.
        """
        w_before = resolve(gpt2, site).weight.detach().clone()
        c3_divergence_demo(
            gpt2, r0, atr_step, site=site, eta=self.ETA, n_iter=self.N_ITER
        )
        assert torch.equal(resolve(gpt2, site).weight, w_before)

    def test_oja_update_is_larger_than_hebb_at_real_weight_scale(
        self, gpt2, r0, atr_step, site
    ):
        """
        Recorded because it inverts the intuition the toy model gives. At the
        toy's init scale (std 0.02) the Oja decay contributes under 1% of the
        update norm; on real GPT-2 weights it dominates, so the Oja delta is
        the larger of the two from the first apply onward. Anyone reading a
        delta_frac trace and expecting "Oja = Hebb minus a small correction"
        will misread the run, and any eta chosen for Hebb is far too large for
        Oja at the same site.
        """
        norms = {}
        for mode in ("hebb", "oja"):
            p = OjaPlasticity(gpt2, site=site, eta=1e-6, mode=mode)
            with p:
                r = r0
                for _ in range(2):
                    r = atr_step(gpt2, r)
                norms[mode] = p.apply()["delta_norm"]
            p.revert()
        assert norms["oja"] > 10 * norms["hebb"]
