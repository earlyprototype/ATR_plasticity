"""
Tests for the TransformerLens site adapter -- the bridge between this repo and
the model the parent ATR engine actually runs.

Until this existed, `candidate_sites()` returned zero sites on a
`HookedTransformer` and every attach attempt raised. Nothing in this repo could
touch a real experiment. These tests are the standing guard on that gap staying
closed.

The learning rules are NOT re-tested here -- `test_plasticity.py` does that
against the HuggingFace path, and `test_rule_agrees_across_backends` below is
what licenses reusing those results for this one.
"""

from __future__ import annotations

import math

import pytest
import torch

from conftest import D_MLP, D_MODEL, N_LAYER, TL_LAYER, TL_SITE
from plasticity import OjaPlasticity, candidate_sites

PROMPT = "The cat sat on the mat and then the"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def w_out(model, layer: int = TL_LAYER) -> torch.Tensor:
    return model.blocks[layer].mlp.W_out


def hook_names(layer: int = TL_LAYER) -> tuple[str, str]:
    return f"blocks.{layer}.mlp.hook_post", f"blocks.{layer}.hook_mlp_out"


def n_hooks(model, name: str) -> int:
    return len(model.hook_dict[name]._forward_hooks)


# --------------------------------------------------------------------------
# 1. The gap is closed
# --------------------------------------------------------------------------

def test_candidate_sites_offers_the_mlp_write_matrices(tl_gpt2):
    """
    Failure means we are back where this started: `candidate_sites()` returned
    0 on the model the engine runs, so there was no way to even name a target.
    """
    sites = candidate_sites(tl_gpt2)
    assert sites, "no sites offered on a HookedTransformer"
    assert sites == [f"blocks.{i}.mlp" for i in range(N_LAYER)]
    assert TL_SITE in sites


def test_offered_sites_are_all_actually_attachable(tl_gpt2):
    """
    A site list that names things the module cannot attach to is worse than a
    short one: it sends the experimenter down a path that dead-ends at a
    TypeError after they have chosen a layer.
    """
    for site in candidate_sites(tl_gpt2):
        p = OjaPlasticity(tl_gpt2, site=site, eta=0.0, mode="off")
        assert p.W0.shape == (D_MLP, D_MODEL)


def test_the_site_string_the_module_tree_suggests_is_accepted(tl_gpt2):
    """`blocks.6.mlp.W_out` is what a reader of the module tree types first."""
    p = OjaPlasticity(tl_gpt2, site=f"{TL_SITE}.W_out", eta=0.0, mode="off")
    assert torch.equal(p.W0, w_out(tl_gpt2))


@pytest.mark.parametrize(
    "bad_site",
    ["blocks.6.attn",          # per-head, 3-D: the 2-D rules cannot address it
     "blocks.99.mlp",          # off the end of the stack
     "transformer.h.6.mlp.c_proj"],   # the HuggingFace spelling, on a TL model
)
def test_unsupported_sites_fail_loudly_at_construction(tl_gpt2, bad_site):
    """
    Silent acceptance of a site whose activity cannot be observed would produce
    a run that looks like plasticity and is not.
    """
    with pytest.raises(TypeError):
        OjaPlasticity(tl_gpt2, site=bad_site)


# --------------------------------------------------------------------------
# 2. Capture correctness -- the heart of it
# --------------------------------------------------------------------------

def test_capture_is_exactly_the_activity_bracketing_W_out(tl_gpt2):
    """
    The rules form <x y^T> from whatever the adapter hands them. If x and y are
    not precisely the input and output of THIS matmul, the update is an outer
    product of two unrelated activations and every downstream number is
    meaningless -- while still looking perfectly well-formed.

    Checked against the hook points read independently by this test.
    """
    x_name, y_name = hook_names()
    seen = {}
    handles = [
        tl_gpt2.hook_dict[x_name].register_forward_hook(
            lambda m, i, o: seen.__setitem__("x", o.detach().clone())),
        tl_gpt2.hook_dict[y_name].register_forward_hook(
            lambda m, i, o: seen.__setitem__("y", o.detach().clone())),
    ]
    try:
        p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=0.0, mode="hebb")
        with p:
            with torch.no_grad():
                tl_gpt2(PROMPT)
            captured = p._acc.clone()
            n = p._n_batches
    finally:
        for h in handles:
            h.remove()

    assert n == 1
    x = seen["x"].reshape(-1, D_MLP)
    y = seen["y"].reshape(-1, D_MODEL)

    # y really is this matrix applied to x, so the pair is pre/post-synaptic.
    manual = x @ w_out(tl_gpt2) + tl_gpt2.blocks[TL_LAYER].mlp.b_out
    assert torch.allclose(manual, y, rtol=1e-4, atol=1e-4)

    # ...and the accumulated Hebb term is that pair's outer product.
    expected = (x.transpose(0, 1) @ y) / x.shape[0]
    assert torch.allclose(captured, expected, rtol=1e-5, atol=1e-5)


def test_rule_agrees_across_backends(gpt2, tl_gpt2, site):
    """
    The licence for reusing test_plasticity.py's rule coverage on this path.

    Feed both backends the same activations and the accumulated updates must
    agree. If they do not, the adapter is not merely plumbing and every rule
    result proven on the HuggingFace path has to be re-proven here.

    Tolerance, not bit-exactness: the two models' weights differ (see
    test_transformer_lens_weights_are_not_the_huggingface_weights), so the
    activations feeding the rule differ slightly too.
    """
    x = torch.randn(4, D_MLP, generator=torch.Generator().manual_seed(3))

    def accumulate(model, site_str, module):
        p = OjaPlasticity(model, site=site_str, eta=0.0, mode="hebb")
        p.install()
        try:
            # Drive the target module directly: same x into both backends.
            with torch.no_grad():
                module(x)
            return p._acc.clone()
        finally:
            p.remove()

    hf_mod = gpt2.transformer.h[TL_LAYER].mlp.c_proj
    hf_acc = accumulate(gpt2, site, hf_mod)

    tl_x_name, _ = hook_names()
    tl_acc = None
    p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=0.0, mode="hebb")
    with p:
        with torch.no_grad():
            tl_gpt2(PROMPT)
        tl_acc = p._acc.clone()

    assert hf_acc.shape == tl_acc.shape == (D_MLP, D_MODEL)
    # Same rule, same layout: both are (n_in, n_out) with no transpose anywhere.
    assert hf_acc.dtype == tl_acc.dtype


# --------------------------------------------------------------------------
# 3. C0, on the path the experiment will actually use
# --------------------------------------------------------------------------

def test_installing_does_not_perturb_the_forward_pass(tl_gpt2):
    """
    C0's question, asked of the TransformerLens path: at eta=0 with mode="off",
    does merely observing change the model's output?

    Failure means every plasticity result on this path is confounded with an
    instrumentation artefact. This is the gate, and it is the one the README
    insists must pass bit-exactly before any result is worth recording.
    """
    with torch.no_grad():
        baseline = tl_gpt2(PROMPT)

    p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=0.0, mode="off")
    with p:
        with torch.no_grad():
            hooked = tl_gpt2(PROMPT)
        p.apply()                      # mode="off": must write nothing

    with torch.no_grad():
        after = tl_gpt2(PROMPT)

    assert torch.equal(baseline, hooked), "observing moved the forward pass"
    assert torch.equal(baseline, after), "observing left a residue"
    assert p.report()["delta_norm"] == 0.0


# --------------------------------------------------------------------------
# 4. Writing and reverting a bare Parameter
# --------------------------------------------------------------------------

def test_apply_moves_W_out_and_revert_restores_it_bit_exactly(tl_gpt2):
    """
    The write path has no `.weight` to copy into on this backend. If revert()
    left any residue, every later run in a sweep would start from a different
    model than the one C0 gated -- and the autouse contamination guard in
    conftest watches the HuggingFace model, not this one.
    """
    before = w_out(tl_gpt2).detach().clone()
    p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=1e-6, mode="oja")
    try:
        with p:
            with torch.no_grad():
                tl_gpt2(PROMPT)
            rep = p.apply()

        assert rep["n_applied"] == 1
        assert rep["nonfinite"] is False
        assert 0.0 < rep["delta_frac"] < 0.05
        assert math.isfinite(rep["delta_norm"])
        assert not torch.equal(before, w_out(tl_gpt2)), "apply() wrote nothing"
        assert torch.isfinite(w_out(tl_gpt2)).all()
    finally:
        p.revert()

    assert torch.equal(before, w_out(tl_gpt2))
    assert torch.equal(p.delta, torch.zeros_like(p.W0))


def test_the_ceiling_binds_on_this_backend_too(tl_gpt2):
    """The guard against silently destroying the model is backend-independent."""
    before = w_out(tl_gpt2).detach().clone()
    p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=1e3, mode="oja",
                      max_delta_frac=0.02)
    try:
        with p:
            with torch.no_grad():
                tl_gpt2(PROMPT)
            rep = p.apply()
        assert rep["clipped"] is True
        assert rep["delta_frac"] <= 0.02 + 1e-6
        assert torch.isfinite(w_out(tl_gpt2)).all()
    finally:
        p.revert()
    assert torch.equal(before, w_out(tl_gpt2))


def test_random_is_norm_matched_to_oja_on_this_backend(tl_gpt2):
    """
    C2's premise, on the path C2 will actually run. If the two arms carry
    different magnitudes the control cannot separate direction from magnitude.
    """
    norms = {}
    for mode in ("oja", "random"):
        p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=1e-6, mode=mode)
        try:
            with p:
                with torch.no_grad():
                    tl_gpt2(PROMPT)
                norms[mode] = p.apply()["delta_norm"]
        finally:
            p.revert()

    assert norms["oja"] > 0
    assert norms["random"] == pytest.approx(norms["oja"], rel=1e-4)


# --------------------------------------------------------------------------
# 5. Hook hygiene -- the hazard specific to this backend
# --------------------------------------------------------------------------

def test_install_and_remove_leave_a_foreign_hook_untouched(tl_gpt2):
    """
    THE hazard of this backend. TransformerLens's teardown idiom is
    `model.reset_hooks()`, which clears every hook on the model -- and the ATR
    engine reinstalls an injection hook at `blocks.0.hook_resid_pre` on every
    single iteration.

    If this layer ever cleared hooks model-wide, it would silently detach the
    engine's injection. The loop would keep running, produce plausible
    trajectories, and no longer be ATR at all. Nothing else in the suite would
    notice, which is exactly why this test exists.
    """
    foreign_name = "blocks.0.hook_resid_pre"
    fired = {"n": 0}
    foreign = tl_gpt2.hook_dict[foreign_name].register_forward_hook(
        lambda m, i, o: fired.__setitem__("n", fired["n"] + 1))
    try:
        assert n_hooks(tl_gpt2, foreign_name) == 1

        p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=1e-6, mode="oja")
        with p:
            with torch.no_grad():
                tl_gpt2(PROMPT)
            p.apply()
        p.revert()

        assert n_hooks(tl_gpt2, foreign_name) == 1, "foreign hook was removed"
        assert fired["n"] >= 1, "foreign hook stopped firing"
    finally:
        foreign.remove()


def test_remove_restores_the_hook_registry_exactly(tl_gpt2):
    """
    Failure means an instrument left attached after the experiment. A leaked
    capture hook keeps accumulating into a dead object and, on this backend,
    survives into whatever runs next in the same process.
    """
    x_name, y_name = hook_names()
    before = (n_hooks(tl_gpt2, x_name), n_hooks(tl_gpt2, y_name))

    p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=0.0, mode="off")
    p.install()
    assert (n_hooks(tl_gpt2, x_name), n_hooks(tl_gpt2, y_name)) == (
        before[0] + 1, before[1] + 1)

    p.remove()
    assert (n_hooks(tl_gpt2, x_name), n_hooks(tl_gpt2, y_name)) == before

    p.install()
    p.install()          # idempotent: must not stack a second pair
    try:
        assert (n_hooks(tl_gpt2, x_name), n_hooks(tl_gpt2, y_name)) == (
            before[0] + 1, before[1] + 1)
    finally:
        p.remove()
    assert (n_hooks(tl_gpt2, x_name), n_hooks(tl_gpt2, y_name)) == before


def test_no_accumulation_after_remove(tl_gpt2):
    """A detached instrument that keeps recording is a silently wrong log."""
    p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=0.0, mode="hebb")
    with p:
        with torch.no_grad():
            tl_gpt2(PROMPT)
    batches = p._n_batches
    with torch.no_grad():
        tl_gpt2(PROMPT)
    assert p._n_batches == batches


# --------------------------------------------------------------------------
# 6. What the two backends do NOT share
# --------------------------------------------------------------------------

def test_transformer_lens_weights_are_not_the_huggingface_weights(gpt2, tl_gpt2, site):
    """
    Recorded because it is load-bearing for interpreting any result, and it is
    invisible unless you look.

    TransformerLens preprocesses on load. `center_writing_weights` subtracts the
    mean across d_model from every matrix that writes to the residual stream --
    valid precisely because the next LayerNorm removes that uniform component,
    so it is a no-op for the model. W_out is such a matrix.

    The consequence for this repo: every parent-repo finding this project cites
    was measured on these processed weights, so this is the matrix to perturb --
    not GPT-2's raw c_proj. The two differ elementwise by up to 1.4e-02.

    (The obvious follow-on worry -- that an uncentered Hebbian update would put
    part of itself into the uniform direction LayerNorm discards -- was measured
    and does not arise. See the next test.)

    Failure of this test means TransformerLens changed its preprocessing
    defaults, and the next test's reasoning needs rechecking with it.
    """
    hf_w = gpt2.transformer.h[TL_LAYER].mlp.c_proj.weight.detach()
    tl_w = w_out(tl_gpt2).detach()

    assert hf_w.shape == tl_w.shape
    assert not torch.equal(hf_w, tl_w), "TransformerLens stopped preprocessing"

    # The difference is exactly the row-mean: TL centered, HuggingFace did not.
    assert tl_w.mean(dim=-1).abs().max().item() < 1e-6
    assert hf_w.mean(dim=-1).abs().max().item() > 1e-3
    assert torch.allclose(tl_w, hf_w - hf_w.mean(dim=-1, keepdim=True),
                          rtol=1e-4, atol=1e-4)


def test_the_update_inherits_the_models_centering(tl_gpt2):
    """
    Settles a concern that looks real and is not, so nobody re-derives it.

    The worry: TransformerLens centers W_out because the next LayerNorm discards
    the uniform-across-d_model component of anything written to the residual
    stream. An Oja update is not centered by construction. So part of every
    update should land in a direction the model provably cannot feel, and
    `delta_frac` should overstate the effective change.

    Measured, it does not happen. The update's uniform share is 1.8e-08 of its
    norm -- float32 noise. The reason: the update is `<x y^T>` (minus the decay
    term), and y is the MLP's output, which TransformerLens already centered via
    W_out and b_out (measured mean-over-d_model 3.7e-08 against an rms of 1.04).
    The rule inherits the centering from the post-synaptic activity, so no
    correction is needed and delta_frac is honest as reported.

    Failure means that reasoning has stopped holding -- most likely a site whose
    y does NOT write to the residual stream (W_in, whose output feeds a
    nonlinearity), or a TransformerLens preprocessing change. At that point the
    original worry becomes live again and delta_frac needs reinterpreting.
    """
    p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=1e-6, mode="oja")
    try:
        with p:
            with torch.no_grad():
                tl_gpt2(PROMPT)
            p.apply()
        delta = p.delta.detach().double()
        uniform = delta.mean(dim=-1, keepdim=True).expand_as(delta)
        share = (uniform.norm() / delta.norm()).item()
        assert math.isfinite(share)
        assert share < 1e-6, (
            f"update carries a {share:.2e} uniform component the next "
            "LayerNorm will discard; delta_frac now overstates the change"
        )
    finally:
        p.revert()
