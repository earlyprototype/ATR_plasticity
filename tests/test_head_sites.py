"""
Tests for per-head sites: one attention head's stripe of a block's output
projection, on both backends.

Issue #25 asks for head granularity because the object of study is a head --
the parent project's period-2 oscillation is carried by a single head in block
11 -- and the plasticity layer could previously only move whole matrices.

A head is not a parameter. HuggingFace stores all twelve of GPT-2's heads in one
(768, 768) Conv1D, where head h owns rows [64h : 64h+64]; TransformerLens stores
the same matrix as (n_heads, d_head, d_model), already per head. So the
HuggingFace path is arithmetic that can be wrong by exactly 64 rows and still
produce a plausible run, and the TransformerLens path is the same experiment
with the arithmetic already done. Both are tested, and
`test_head_stripes_are_the_same_rows_on_both_backends` is what says the
arithmetic agrees with the model that did not have to do any.

The two claims worth stating as claims:

  1. ISOLATION. Updating head 7 leaves the other eleven heads' rows bit-for-bit
     identical -- not equal to float tolerance, identical. A distributed
     experiment over heads is only interpretable if the sites are disjoint.
  2. RESTRICTION, NOT REPLACEMENT. The update a head site produces is exactly
     the row slice of the update the whole-matrix site would have produced.
     Both rules are row-wise in the (n_in, n_out) convention, so this is a
     property of the rules and not a coincidence -- and it is what licenses
     reading the whole-matrix rule tests as covering these sites too.

Weight hygiene: these tests touch `transformer.h.11.attn.c_proj` and block 11's
W_O, neither of which `conftest._target_weight_unchanged` watches. Every
mutating test therefore reverts in a `finally` AND asserts the restore itself.
"""

from __future__ import annotations

import pytest
import torch

from conftest import D_MODEL, N_LAYER
from plasticity import OjaPlasticity, candidate_sites
from test_plasticity import capture, drive

# Block 11, head 7: the last block because that is where the parent project's
# oscillation lives, and a middle head because head 0 and head 11 are the two
# an off-by-one in the row arithmetic would land on.
LAYER = 11
HEAD = 7
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS                      # 64

HF_MATRIX = f"transformer.h.{LAYER}.attn.c_proj"
HF_SITE = f"{HF_MATRIX}.head.{HEAD}"
TL_SITE = f"blocks.{LAYER}.attn.head.{HEAD}"
ROWS = slice(HEAD * D_HEAD, (HEAD + 1) * D_HEAD)

PROMPT = "The cat sat on the mat and then the"


# --------------------------------------------------------------------------
# Local helpers
# --------------------------------------------------------------------------

def hf_matrix(gpt2) -> torch.Tensor:
    return gpt2.transformer.h[LAYER].attn.c_proj.weight


def tl_W_O(tl_gpt2) -> torch.Tensor:
    return tl_gpt2.blocks[LAYER].attn.W_O


def run_tl(tl_gpt2):
    with torch.no_grad():
        tl_gpt2(PROMPT)


# --------------------------------------------------------------------------
# 1. The layout premise the row arithmetic rests on
# --------------------------------------------------------------------------

def test_head_stripes_are_the_same_rows_on_both_backends(gpt2, tl_gpt2):
    """
    The premise of `_HeadSliceSite`: head h owns rows [64h : 64h+64] of
    HuggingFace's packed c_proj.

    Checked against TransformerLens, which stores W_O as (n_heads, d_head,
    d_model) and therefore never had to do this arithmetic. If the two ever
    disagree, one of them is addressing a different head -- and nothing about
    the shapes, the report, or the run would say so, because every stripe of
    this matrix is a valid (64, 768) matrix.

    The residual difference is TransformerLens's `center_writing_weights`, which
    subtracts the mean over d_model from anything that writes to the residual
    stream (see test_tl_site.py). Row-centering commutes with taking rows, so it
    cancels exactly here rather than to a tolerance.
    """
    hf_w = hf_matrix(gpt2).detach()
    tl_w = tl_W_O(tl_gpt2).detach()

    assert hf_w.shape == (N_HEADS * D_HEAD, D_MODEL)
    assert tl_w.shape == (N_HEADS, D_HEAD, D_MODEL)

    for h in range(N_HEADS):
        stripe = hf_w[h * D_HEAD:(h + 1) * D_HEAD]
        centered = stripe - stripe.mean(dim=-1, keepdim=True)
        assert torch.equal(centered, tl_w[h]), f"head {h} is not the same rows"

    # ...and the two site spellings pick up exactly those stripes.
    p_hf = OjaPlasticity(gpt2, site=HF_SITE, eta=0.0, mode="off")
    p_tl = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=0.0, mode="off")
    assert p_hf.W0.shape == p_tl.W0.shape == (D_HEAD, D_MODEL)
    assert torch.equal(p_hf.W0, hf_w[ROWS])
    assert torch.equal(p_tl.W0, tl_w[HEAD])


# --------------------------------------------------------------------------
# 2. Naming and construction
# --------------------------------------------------------------------------

class TestHeadSiteNaming:

    def test_candidate_sites_offers_heads_only_when_asked(self, gpt2, tl_gpt2):
        """
        Head sites are a different granularity of experiment, not a longer list
        of the same one -- `max_delta_frac` is a fraction of whichever W0 the
        site names, so a whole-matrix arm and a per-head arm at one eta are not
        comparable. Defaulting them on would silently change what every existing
        caller of `candidate_sites()` is offered.
        """
        assert candidate_sites(gpt2) == candidate_sites(gpt2, heads=False)
        assert len(candidate_sites(gpt2)) == 48
        assert len(candidate_sites(tl_gpt2)) == N_LAYER

        hf = candidate_sites(gpt2, heads=True)
        assert len(hf) == 48 + N_LAYER * N_HEADS
        assert hf[:48] == candidate_sites(gpt2)
        assert HF_SITE in hf
        # Only the output projection is offered per head: c_attn's head stripes
        # run along the output axis, three times over, so the same suffix there
        # would mean something else.
        assert not any(".c_attn.head." in s for s in hf)
        assert not any(".mlp." in s and ".head." in s for s in hf)

        tl = candidate_sites(tl_gpt2, heads=True)
        assert len(tl) == N_LAYER + N_LAYER * N_HEADS
        assert tl[:N_LAYER] == candidate_sites(tl_gpt2)
        assert TL_SITE in tl

    def test_every_offered_head_site_is_attachable(self, gpt2, tl_gpt2):
        """
        A list that names things the module cannot attach to sends the
        experimenter down a path that dead-ends at a TypeError after they have
        chosen a head. 144 of them per backend, so the failure would be found
        late and by hand.
        """
        for s in candidate_sites(gpt2, heads=True)[48:]:
            assert OjaPlasticity(gpt2, site=s, eta=0.0, mode="off").W0.shape == \
                (D_HEAD, D_MODEL)
        for s in candidate_sites(tl_gpt2, heads=True)[N_LAYER:]:
            assert OjaPlasticity(tl_gpt2, site=s, eta=0.0, mode="off").W0.shape == \
                (D_HEAD, D_MODEL)

    @pytest.mark.parametrize(
        "bad_site",
        [f"{HF_MATRIX}.head.12",        # off the end: 12 heads, 0..11
         f"{HF_MATRIX}.head.-1",        # not a head index at all
         f"{HF_MATRIX}.head.x",         # not a number
         f"{HF_MATRIX}.head.",          # empty index
         f"transformer.h.{LAYER}.mlp.c_proj.head.0",   # not an attention matrix
         f"transformer.h.{LAYER}.ln_1.head.0"],        # not a matrix at all
    )
    def test_unattachable_head_sites_fail_loudly(self, gpt2, bad_site):
        """
        Every one of these has a plausible reading and a wrong answer available.
        `mlp.c_proj.head.0` is the dangerous one: it is a real 2-D weight and
        rows 0..63 of it are a real (64, 768) matrix, so slicing it would
        succeed and produce a run about a stripe of an MLP that no head owns.
        The module that owns the matrix is asked for its head count and there
        isn't one, which is the check that stops it.
        """
        with pytest.raises(TypeError):
            OjaPlasticity(gpt2, site=bad_site)

    @pytest.mark.parametrize(
        "bad_site",
        [f"blocks.{LAYER}.attn.head.12",
         f"blocks.{LAYER}.mlp.head.0",
         "blocks.99.attn.head.0",
         f"{HF_MATRIX}.head.{HEAD}"],     # HuggingFace spelling on a TL model
    )
    def test_unattachable_head_sites_fail_loudly_on_transformer_lens(
        self, tl_gpt2, bad_site
    ):
        with pytest.raises(TypeError):
            OjaPlasticity(tl_gpt2, site=bad_site)

    def test_transposed_is_refused_rather_than_approximated(self, gpt2):
        """
        A head owns rows of an (n_in, n_out) matrix and columns of an
        (n_out, n_in) one. The adapter slices rows unconditionally, so
        `transposed=True` would take 64 wrong numbers and report nothing --
        the same class of silent defect as README defect 1. Refused at
        construction instead.
        """
        with pytest.raises(ValueError, match="transposed=True is not supported"):
            OjaPlasticity(gpt2, site=HF_SITE, transposed=True)
        # ...and the whole-matrix site is unaffected by the new check.
        assert OjaPlasticity(gpt2, site=HF_MATRIX, transposed=True).transposed


# --------------------------------------------------------------------------
# 3. Isolation -- claim 1
# --------------------------------------------------------------------------

class TestIsolation:

    ETA = 1e-8    # far below the ceiling: a clipped update is a rescale, not the rule

    @pytest.mark.parametrize("mode", ["oja", "hebb", "anti_hebb", "random"])
    def test_only_the_named_head_moves_on_huggingface(
        self, gpt2, r0, atr_step, mode
    ):
        """
        The claim per head, on the packed matrix where it can fail.

        Bit-for-bit on the other eleven, deliberately: writing the whole matrix
        back with the untouched rows unchanged in value would pass an
        `allclose`, and would still mean the site is the matrix rather than the
        head -- every "one head" result would then be a whole-matrix result
        wearing a label, and the distributed-damping experiment issue #25 wants
        would be untestable.
        """
        before = hf_matrix(gpt2).detach().clone()
        p = OjaPlasticity(gpt2, site=HF_SITE, eta=self.ETA, mode=mode)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=1)
            rep = p.apply()

            after = hf_matrix(gpt2)
            assert rep["clipped"] is False
            assert rep["delta_norm"] > 0.0
            assert not torch.equal(after[ROWS], before[ROWS]), "the named head did not move"
            for h in range(N_HEADS):
                if h == HEAD:
                    continue
                rows = slice(h * D_HEAD, (h + 1) * D_HEAD)
                assert torch.equal(after[rows], before[rows]), f"head {h} was modified"
        finally:
            p.revert()
        assert torch.equal(hf_matrix(gpt2), before)

    def test_only_the_named_head_moves_on_transformer_lens(self, tl_gpt2):
        """The same claim on the backend the ATR engine runs."""
        before = tl_W_O(tl_gpt2).detach().clone()
        p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=1e-9, mode="oja")
        try:
            with p:
                run_tl(tl_gpt2)
            rep = p.apply()

            after = tl_W_O(tl_gpt2)
            assert rep["clipped"] is False
            assert not torch.equal(after[HEAD], before[HEAD])
            for h in range(N_HEADS):
                if h != HEAD:
                    assert torch.equal(after[h], before[h]), f"head {h} was modified"
        finally:
            p.revert()
        assert torch.equal(tl_W_O(tl_gpt2), before)

    def test_two_heads_can_be_driven_at_once_without_crosstalk(
        self, gpt2, r0, atr_step
    ):
        """
        Issue #25's actual programme is distributed damping: many small elements
        each acting on its own subspace. That needs two instances on the same
        matrix to be independent, and the failure mode is specific -- each
        instance holds its own W0 and writes `W0 + delta`, so if either wrote
        the whole matrix it would silently revert the other's work while
        reporting its own delta as applied.
        """
        before = hf_matrix(gpt2).detach().clone()
        other = 3
        p_a = OjaPlasticity(gpt2, site=HF_SITE, eta=self.ETA, mode="oja")
        p_b = OjaPlasticity(gpt2, site=f"{HF_MATRIX}.head.{other}",
                            eta=self.ETA, mode="oja")
        try:
            with p_a, p_b:
                drive(gpt2, r0, atr_step, n=1)
                p_a.apply()
                p_b.apply()

            after = hf_matrix(gpt2)
            rows_b = slice(other * D_HEAD, (other + 1) * D_HEAD)
            assert torch.equal(after[ROWS], before[ROWS] + p_a.delta)
            assert torch.equal(after[rows_b], before[rows_b] + p_b.delta)
            for h in range(N_HEADS):
                if h in (HEAD, other):
                    continue
                rows = slice(h * D_HEAD, (h + 1) * D_HEAD)
                assert torch.equal(after[rows], before[rows])
        finally:
            p_a.revert()
            p_b.revert()
        assert torch.equal(hf_matrix(gpt2), before)


# --------------------------------------------------------------------------
# 4. Restriction, not replacement -- claim 2
# --------------------------------------------------------------------------

class TestTheUpdateIsTheSameRule:

    ETA = 1e-9

    @pytest.mark.parametrize("mode", ["oja", "hebb", "anti_hebb"])
    def test_the_head_update_is_the_row_slice_of_the_whole_matrix_update(
        self, gpt2, r0, atr_step, mode
    ):
        """
        Bit-for-bit, which is stronger than it looks and is the point.

        `<x y^T>` row i depends only on x_i, and `W <y y^T>` row i only on row i
        of W, so restricting the input to a head's rows restricts the update to
        those rows and changes nothing else. Slicing x -- rather than, say,
        slicing y or using the head's isolated contribution to the output -- is
        the only choice that has this property, and it is the choice that makes
        `test_plasticity.py`'s rule tests cover these sites too.

        A failure here means a per-head run and a whole-matrix run are not
        measuring the same rule, so no per-head result can be read against the
        whole-matrix baseline the repo already has.
        """
        p_f = OjaPlasticity(gpt2, site=HF_MATRIX, eta=self.ETA, mode=mode)
        try:
            with p_f:
                drive(gpt2, r0, atr_step, n=1)
            rep_f = p_f.apply()
            slice_of_full = p_f.delta[ROWS].clone()
        finally:
            p_f.revert()

        p_h = OjaPlasticity(gpt2, site=HF_SITE, eta=self.ETA, mode=mode)
        try:
            with p_h:
                drive(gpt2, r0, atr_step, n=1)
            rep_h = p_h.apply()
            head_delta = p_h.delta.clone()
        finally:
            p_h.revert()

        assert rep_f["clipped"] is False and rep_h["clipped"] is False
        assert head_delta.norm().item() > 0
        assert torch.equal(head_delta, slice_of_full)

    def test_the_ceiling_is_a_fraction_of_the_head_not_of_the_matrix(
        self, gpt2, r0, atr_step
    ):
        """
        `max_delta_frac` is drift relative to the site's own W0, and for a head
        site that is ||W_head||_F (35.58 here), not the whole matrix's 139.72.
        Read the other way round, a 5% cap measured against the matrix would be
        a 20% cap on the head -- four times the drift the operator asked for, on
        the one experiment where the point is to perturb a head gently.
        """
        p = OjaPlasticity(gpt2, site=HF_SITE, eta=1e3, mode="oja",
                          max_delta_frac=0.05)
        before = hf_matrix(gpt2).detach().clone()
        try:
            head_norm = p.W0_norm
            matrix_norm = before.double().norm().item()
            assert head_norm < matrix_norm            # the two are distinguishable
            assert head_norm == pytest.approx(before[ROWS].double().norm().item())

            with p:
                drive(gpt2, r0, atr_step, n=1)
            rep = p.apply()

            assert rep["clipped"] is True
            assert rep["delta_frac"] == pytest.approx(0.05, rel=1e-4)
            assert rep["delta_norm"] == pytest.approx(0.05 * head_norm, rel=1e-4)
            assert torch.isfinite(hf_matrix(gpt2)).all()
        finally:
            p.revert()
        assert torch.equal(hf_matrix(gpt2), before)


# --------------------------------------------------------------------------
# 5. Capture -- what x and y actually are
# --------------------------------------------------------------------------

def test_x_is_the_heads_own_rows_and_y_is_the_shared_output(gpt2, r0, atr_step):
    """
    The pair the rule is formed from, read off the model by this test rather
    than by the module under test.

    x must be the head's 64 columns of the c_proj input and y the full 768-wide
    output. Slicing y as well would be the tempting mistake: it would make the
    update the outer product of a head's input with an arbitrary 64 of the
    output units, which is not a rule anyone has proposed and would look
    entirely healthy in the report.
    """
    p = OjaPlasticity(gpt2, site=HF_SITE, eta=0.0, mode="hebb")
    with capture(p.module) as seen, p:
        drive(gpt2, r0, atr_step, n=1)
        acc = p._acc.clone()
        n = p._n_batches

    assert n == 1
    x_full, y = seen[0]
    assert x_full.shape[-1] == D_MODEL and y.shape[-1] == D_MODEL
    x = x_full[:, ROWS]
    expected = (x.transpose(0, 1) @ y) / x.shape[0]

    assert acc.shape == (D_HEAD, D_MODEL)
    assert torch.allclose(acc, expected, rtol=1e-5, atol=1e-6)


def test_transformer_lens_capture_is_z_of_this_head_and_the_attention_output(tl_gpt2):
    """
    Same claim on the other backend, where x and y come from two hook points
    rather than from one module's forward. `hook_z` is (batch, pos, n_heads,
    d_head) and the head axis is the third, not the last -- indexing the wrong
    axis there is an error that stays in range for GPT-2 small, since d_head
    (64) and n_heads (12) are both valid indices into neither.
    """
    x_name = f"blocks.{LAYER}.attn.hook_z"
    y_name = f"blocks.{LAYER}.hook_attn_out"
    seen = {}
    handles = [
        tl_gpt2.hook_dict[x_name].register_forward_hook(
            lambda m, i, o: seen.__setitem__("z", o.detach().clone())),
        tl_gpt2.hook_dict[y_name].register_forward_hook(
            lambda m, i, o: seen.__setitem__("a", o.detach().clone())),
    ]
    try:
        p = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=0.0, mode="hebb")
        with p:
            run_tl(tl_gpt2)
        acc = p._acc.clone()
        n = p._n_batches
    finally:
        for h in handles:
            h.remove()

    z, a = seen["z"], seen["a"]
    assert n == 1
    assert z.shape[-2:] == (N_HEADS, D_HEAD)
    x = z[..., HEAD, :].reshape(-1, D_HEAD)
    y = a.reshape(-1, D_MODEL)
    expected = (x.transpose(0, 1) @ y) / x.shape[0]

    assert acc.shape == (D_HEAD, D_MODEL)
    assert torch.allclose(acc, expected, rtol=1e-5, atol=1e-5)
    # The head axis, not some other axis of the right size: any other head's z
    # would give a different accumulator.
    for other in (0, HEAD + 1):
        wrong = z[..., other, :].reshape(-1, D_HEAD)
        assert not torch.allclose(acc, (wrong.transpose(0, 1) @ y) / wrong.shape[0],
                                  rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------
# 6. The scaffold, on the new site type
# --------------------------------------------------------------------------

def test_revert_restores_the_stripe_bit_exactly(gpt2, r0, atr_step):
    """
    Control C1 on a head site. `conftest._target_weight_unchanged` watches the
    mid-stack MLP, not this matrix, so a residue here would contaminate every
    later test in the session with nothing to report it.
    """
    before = hf_matrix(gpt2).detach().clone()
    p = OjaPlasticity(gpt2, site=HF_SITE, eta=1e-6, mode="anti_hebb")
    try:
        with p:
            drive(gpt2, r0, atr_step, n=2)
            p.apply()
            assert not torch.equal(hf_matrix(gpt2), before)
            drive(gpt2, r0, atr_step, n=1)      # leave something pending too
    finally:
        p.revert()

    assert torch.equal(hf_matrix(gpt2), before)
    assert torch.equal(p.delta, torch.zeros_like(p.W0))
    assert p.report()["delta_norm"] == 0.0


def test_hooks_are_removed_from_both_backends(gpt2, tl_gpt2):
    """
    A leaked hook on a head site keeps slicing activations into a dead object,
    and on the TransformerLens path it survives into whatever runs next in the
    process. `install()` must be exactly invertible here as everywhere else.
    """
    mod = gpt2.transformer.h[LAYER].attn.c_proj
    before = len(mod._forward_hooks)
    p = OjaPlasticity(gpt2, site=HF_SITE, eta=0.0, mode="off")
    p.install()
    assert len(mod._forward_hooks) == before + 1
    p.install()                                  # idempotent
    assert len(mod._forward_hooks) == before + 1
    p.remove()
    assert len(mod._forward_hooks) == before
    p.remove()                                   # second remove must not raise

    names = [f"blocks.{LAYER}.attn.hook_z", f"blocks.{LAYER}.hook_attn_out"]
    counts = [len(tl_gpt2.hook_dict[n]._forward_hooks) for n in names]
    q = OjaPlasticity(tl_gpt2, site=TL_SITE, eta=0.0, mode="off")
    with q:
        assert [len(tl_gpt2.hook_dict[n]._forward_hooks) for n in names] == \
            [c + 1 for c in counts]
    assert [len(tl_gpt2.hook_dict[n]._forward_hooks) for n in names] == counts
