"""
Tests for `multi_site.py`: simultaneous plasticity on many sites at once.

Issue #25 item 2 and DESIGN.md's "one site or many" ask for distributed damping
-- many small local rules acting together, whole matrices and per-head stripes
mixed, each on its own subspace. `MultiSitePlasticity` composes N `OjaPlasticity`
instances to do that. It reimplements no rule, so the learning-rule correctness
tested in `test_plasticity.py` and `test_head_sites.py` carries over to each site
unchanged; what is new here, and all this file tests, is the *composition*:

  1. DISJOINT WRITES. Two per-head instances on two heads of ONE `attn.c_proj`
     must not clobber each other under simultaneous apply(). This is the property
     that makes a distributed head experiment interpretable -- it is
     `test_head_sites.py`'s ISOLATION claim, now under the driver and at twelve
     heads rather than two. It holds because the head adapters write row-confined;
     the driver's job is to preserve it and to REJECT the overlaps that would
     break it.
  2. COMPOSITION EQUALS THE WHOLE. The twelve head-instances of one matrix, run
     together, reconstruct the single whole-matrix instance's update bit-for-bit
     -- issue #25's "restriction, not replacement", composed.
  3. Every single-site guarantee -- per-site ceiling, bit-exact revert of every
     touched matrix, the nonfinite flag, the report schema -- survives being one
     of many.

Weight hygiene: several sites here (mid-stack MLPs, layer-11 attention) are not
all watched by `conftest._target_weight_unchanged`, so every mutating test
reverts in a `finally` AND asserts the restore itself, exactly as
`test_head_sites.py` does.
"""

from __future__ import annotations

import pytest
import torch

from conftest import D_MODEL, N_LAYER, REPORT_TYPES
from plasticity import OjaPlasticity
from multi_site import MultiSitePlasticity, SiteSpec
from test_plasticity import drive


# Layer 11's attention output projection is where the per-head sites live -- the
# block the parent found carrying the period-2 oscillation, and the one
# `conftest` and `test_head_sites` already reach for.
LAYER = 11
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS                       # 64
HF_ATTN = f"transformer.h.{LAYER}.attn.c_proj"
TL_ATTN_HEAD = f"blocks.{LAYER}.attn.head"

# Two disjoint whole matrices, neither the attention projection above.
HF_MLP_A = "transformer.h.5.mlp.c_proj"
HF_MLP_B = "transformer.h.8.mlp.c_proj"
TL_MLP = "blocks.5.mlp"

PROMPT = "The cat sat on the mat and then the"


# --------------------------------------------------------------------------
# Local helpers, independent of the code under test
# --------------------------------------------------------------------------

def hf_attn(gpt2) -> torch.Tensor:
    return gpt2.transformer.h[LAYER].attn.c_proj.weight


def hf_weight(gpt2, path: str) -> torch.Tensor:
    obj = gpt2
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj.weight


def tl_W_O(tl_gpt2) -> torch.Tensor:
    return tl_gpt2.blocks[LAYER].attn.W_O


def tl_W_out(tl_gpt2, layer: int) -> torch.Tensor:
    return tl_gpt2.blocks[layer].mlp.W_out


def rows(h: int) -> slice:
    return slice(h * D_HEAD, (h + 1) * D_HEAD)


def run_tl(tl_gpt2) -> None:
    with torch.no_grad():
        tl_gpt2(PROMPT)


def head(matrix: str, h: int) -> str:
    return f"{matrix}.head.{h}"


# --------------------------------------------------------------------------
# 1. Construction and configuration
# --------------------------------------------------------------------------

class TestConstruction:

    def test_specs_may_be_dicts_or_SiteSpecs(self, gpt2):
        """The experiment scripts build cells as dicts; a driver that only took a
        bespoke record would force every caller to convert. Both are accepted and
        normalise to the same thing."""
        a = MultiSitePlasticity(gpt2, [
            {"site": HF_MLP_A, "mode": "oja", "eta": 1e-7},
            SiteSpec(HF_MLP_B, mode="hebb", eta=1e-6),
        ])
        assert len(a) == 2
        assert a.sites == [HF_MLP_A, HF_MLP_B]
        assert a[0].mode == "oja" and a[1].mode == "hebb"
        assert a[0].eta == 1e-7 and a[1].eta == 1e-6

    def test_heterogeneous_config_is_carried_through_per_site(self, gpt2):
        """Issue #25's whole point is many small *heterogeneous* elements. Mode,
        eta and ceiling must reach each underlying instance intact, or the driver
        is silently running one experiment where the caller asked for several."""
        d = MultiSitePlasticity(gpt2, [
            SiteSpec(HF_MLP_A, mode="oja", eta=1e-7, max_delta_frac=0.05),
            SiteSpec(head(HF_ATTN, 3), mode="anti_hebb", eta=2e-8, max_delta_frac=0.01),
        ])
        assert d[0].mode == "oja" and d[0].eta == 1e-7 and d[0].max_delta_frac == 0.05
        assert d[1].mode == "anti_hebb" and d[1].eta == 2e-8
        assert d[1].max_delta_frac == 0.01
        # The per-head instance really is the stripe, not the matrix.
        assert d[1].W0.shape == (D_HEAD, D_MODEL)

    def test_empty_driver_is_refused(self, gpt2):
        """An empty driver would install nothing and no-op every apply() while
        looking like a live instrument -- the failure a caller would notice only
        by the weights never moving."""
        with pytest.raises(ValueError, match="at least one site"):
            MultiSitePlasticity(gpt2, [])

    def test_a_bad_site_raises_at_construction_naming_the_site(self, gpt2):
        """A bad site among good ones must fail loud, and as the same TypeError a
        hand-built OjaPlasticity would raise -- not be silently dropped, leaving a
        driver quietly smaller than the experiment asked for."""
        with pytest.raises(TypeError):
            MultiSitePlasticity(gpt2, [
                SiteSpec(HF_MLP_A),
                SiteSpec("transformer.h.6.mlp"),      # a container, no 2-D weight
            ])

    def test_coerce_rejects_a_non_spec(self, gpt2):
        with pytest.raises(TypeError, match="SiteSpec or a dict"):
            MultiSitePlasticity(gpt2, [HF_MLP_A])     # a bare string, not a spec


# --------------------------------------------------------------------------
# 2. Overlap rejection -- the failure-injection direction
# --------------------------------------------------------------------------

class TestOverlapRejected:
    """
    The disjoint-write property is a precondition, so the driver enforces it. The
    both-directions treatment: it rejects every way two sites can write the same
    rows, and it permits the disjoint cases those rejections must not catch by
    mistake.
    """

    def test_a_whole_matrix_and_a_head_of_it_collide(self, gpt2):
        """The whole `attn.c_proj` writes all 768 rows; head 3 writes 64 of them.
        Applied together, the whole-matrix write would reset head 3's rows to W0
        after head 3 moved them -- the exact clobber the property forbids."""
        with pytest.raises(ValueError, match="overlapping rows"):
            MultiSitePlasticity(gpt2, [
                SiteSpec(HF_ATTN, mode="oja", eta=1e-8),
                SiteSpec(head(HF_ATTN, 3), mode="oja", eta=1e-8),
            ])

    def test_the_same_head_twice_collides(self, gpt2):
        with pytest.raises(ValueError, match="overlapping rows"):
            MultiSitePlasticity(gpt2, [
                SiteSpec(head(HF_ATTN, 7), mode="oja", eta=1e-8),
                SiteSpec(head(HF_ATTN, 7), mode="hebb", eta=1e-8),
            ])

    def test_the_same_matrix_under_two_spellings_collides_on_transformer_lens(
        self, tl_gpt2
    ):
        """`blocks.6.mlp` and `blocks.6.mlp.W_out` are the same parameter under
        two names. Overlap is checked on the live storage, not the string, so the
        alias is caught -- a string comparison would have missed it and let two
        whole-matrix instances clobber each other."""
        with pytest.raises(ValueError, match="overlapping rows"):
            MultiSitePlasticity(tl_gpt2, [
                SiteSpec("blocks.6.mlp", mode="oja", eta=1e-8),
                SiteSpec("blocks.6.mlp.W_out", mode="oja", eta=1e-8),
            ])

    def test_different_heads_of_one_matrix_are_allowed(self, gpt2):
        """The case the rejections above must NOT catch: two different heads of one
        matrix are disjoint and are the whole point of the class."""
        d = MultiSitePlasticity(gpt2, [
            SiteSpec(head(HF_ATTN, 3), mode="oja", eta=1e-8),
            SiteSpec(head(HF_ATTN, 7), mode="oja", eta=1e-8),
        ])
        assert len(d) == 2

    def test_heads_of_different_matrices_and_whole_matrices_are_allowed(
        self, gpt2
    ):
        """Different matrices never share storage, so a whole MLP, a whole
        attention projection on another layer, and a head of a third are all
        disjoint -- the mixed distributed run issue #25 describes."""
        d = MultiSitePlasticity(gpt2, [
            SiteSpec(HF_MLP_A, mode="oja", eta=1e-7),
            SiteSpec(HF_MLP_B, mode="anti_hebb", eta=1e-7),
            SiteSpec(head(HF_ATTN, 5), mode="hebb", eta=1e-5),
        ])
        assert len(d) == 3


# --------------------------------------------------------------------------
# 3. Many disjoint sites move together and revert cleanly
# --------------------------------------------------------------------------

class TestDisjointSimultaneous:

    def test_four_disjoint_sites_move_together_and_revert_bit_exactly(
        self, gpt2, r0, atr_step
    ):
        """
        Two whole MLPs on different layers and two heads of one attention
        projection, heterogeneous in mode and eta, driven and applied in one loop.
        Every touched matrix must move, the untouched heads of the attention
        projection must not, and revert() must restore all three matrices to their
        exact original bits -- on a session-scoped model, a residue in the
        unwatched mid-stack MLPs would silently corrupt every later test.
        """
        specs = [
            SiteSpec(HF_MLP_A, mode="oja", eta=1e-6),
            SiteSpec(HF_MLP_B, mode="anti_hebb", eta=1e-6),
            SiteSpec(head(HF_ATTN, 3), mode="hebb", eta=1e-4),
            SiteSpec(head(HF_ATTN, 7), mode="oja", eta=1e-8),
        ]
        before_a = hf_weight(gpt2, HF_MLP_A).detach().clone()
        before_b = hf_weight(gpt2, HF_MLP_B).detach().clone()
        before_attn = hf_attn(gpt2).detach().clone()

        driver = MultiSitePlasticity(gpt2, specs)
        try:
            with driver:
                drive(gpt2, r0, atr_step, n=1)
            rep = driver.apply()

            assert rep["n_sites"] == 4
            assert rep["n_applied"] == 1
            assert rep["clipped"] is False
            assert rep["nonfinite"] is False
            assert rep["delta_norm"] > 0.0
            assert [r["site"] for r in rep["per_site"]] == [s.site for s in specs]
            for r in rep["per_site"]:
                assert r["delta_norm"] > 0.0, f"{r['site']} did not move"

            assert not torch.equal(hf_weight(gpt2, HF_MLP_A), before_a)
            assert not torch.equal(hf_weight(gpt2, HF_MLP_B), before_b)
            after = hf_attn(gpt2)
            assert not torch.equal(after[rows(3)], before_attn[rows(3)])
            assert not torch.equal(after[rows(7)], before_attn[rows(7)])
            for h in range(N_HEADS):
                if h in (3, 7):
                    continue
                assert torch.equal(after[rows(h)], before_attn[rows(h)]), \
                    f"head {h} moved but was not a site"
        finally:
            driver.revert()

        assert torch.equal(hf_weight(gpt2, HF_MLP_A), before_a)
        assert torch.equal(hf_weight(gpt2, HF_MLP_B), before_b)
        assert torch.equal(hf_attn(gpt2), before_attn)
        assert driver.report()["delta_norm"] == 0.0
        assert driver.report()["n_applied"] == 0

    def test_disjoint_sites_move_together_on_transformer_lens(self, tl_gpt2):
        """The same claim on the backend the ATR engine runs: one MLP W_out and
        two heads of one attention W_O, the head writes indexed rather than
        sliced. The mid-stack MLP is not watched by `conftest`, so it is reverted
        and the restore asserted here."""
        before_mlp = tl_W_out(tl_gpt2, 5).detach().clone()
        before_wo = tl_W_O(tl_gpt2).detach().clone()

        driver = MultiSitePlasticity(tl_gpt2, [
            SiteSpec(TL_MLP, mode="oja", eta=1e-7),
            SiteSpec(f"{TL_ATTN_HEAD}.3", mode="oja", eta=1e-8),
            SiteSpec(f"{TL_ATTN_HEAD}.7", mode="oja", eta=1e-8),
        ])
        try:
            with driver:
                run_tl(tl_gpt2)
            rep = driver.apply()

            assert rep["clipped"] is False
            assert not torch.equal(tl_W_out(tl_gpt2, 5), before_mlp)
            after = tl_W_O(tl_gpt2)
            assert not torch.equal(after[3], before_wo[3])
            assert not torch.equal(after[7], before_wo[7])
            for h in range(N_HEADS):
                if h not in (3, 7):
                    assert torch.equal(after[h], before_wo[h]), f"head {h} moved"
        finally:
            driver.revert()

        assert torch.equal(tl_W_out(tl_gpt2, 5), before_mlp)
        assert torch.equal(tl_W_O(tl_gpt2), before_wo)


# --------------------------------------------------------------------------
# 4. Isolation under simultaneous operation -- claim 1, at the driver
# --------------------------------------------------------------------------

class TestIsolationUnderSimultaneousOperation:

    ETA = 1e-8       # far below the ceiling: a clipped update is a rescale, not the rule

    @pytest.mark.parametrize("mode", ["oja", "hebb", "anti_hebb"])
    def test_two_heads_together_equal_each_head_run_alone(
        self, gpt2, r0, atr_step, mode
    ):
        """
        Head 3 and head 7 of one `attn.c_proj`, driven together in one driver,
        produce on each head's rows exactly the delta that head produces run
        alone -- and leave the other ten heads' rows bit-identical. Bit-for-bit,
        because that is the isolation claim: an `allclose` would also pass a driver
        that quietly wrote whole matrices and happened to restore the other rows
        to the same values, which is the failure mode that makes a distributed
        head experiment uninterpretable.
        """
        before = hf_attn(gpt2).detach().clone()

        driver = MultiSitePlasticity(gpt2, [
            SiteSpec(head(HF_ATTN, 3), mode=mode, eta=self.ETA),
            SiteSpec(head(HF_ATTN, 7), mode=mode, eta=self.ETA),
        ])
        try:
            with driver:
                drive(gpt2, r0, atr_step, n=1)
            rep = driver.apply()
            assert rep["clipped"] is False
            d3_together = driver[0].delta.clone()
            d7_together = driver[1].delta.clone()

            after = hf_attn(gpt2)
            for h in range(N_HEADS):
                if h in (3, 7):
                    continue
                assert torch.equal(after[rows(h)], before[rows(h)]), \
                    f"head {h} was modified by a run that did not name it"
        finally:
            driver.revert()
        assert torch.equal(hf_attn(gpt2), before)

        d3_alone = self._delta_alone(gpt2, r0, atr_step, head(HF_ATTN, 3), mode)
        d7_alone = self._delta_alone(gpt2, r0, atr_step, head(HF_ATTN, 7), mode)

        assert d3_alone.norm().item() > 0 and d7_alone.norm().item() > 0
        assert torch.equal(d3_together, d3_alone), "head 3 differed run together vs alone"
        assert torch.equal(d7_together, d7_alone), "head 7 differed run together vs alone"

    def _delta_alone(self, gpt2, r0, atr_step, site, mode) -> torch.Tensor:
        """One head's delta from a single-site run, weights handed back untouched."""
        p = OjaPlasticity(gpt2, site=site, eta=self.ETA, mode=mode)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=1)
            p.apply()
            return p.delta.clone()
        finally:
            p.revert()


# --------------------------------------------------------------------------
# 5. Composition equals the whole matrix -- claim 2, composed
# --------------------------------------------------------------------------

class TestCompositionEqualsWhole:

    ETA = 1e-9

    @pytest.mark.parametrize("mode", ["oja", "hebb", "anti_hebb"])
    def test_twelve_heads_together_reconstruct_the_whole_matrix_update(
        self, gpt2, r0, atr_step, mode
    ):
        """
        All twelve heads of one `attn.c_proj`, driven together, compose to exactly
        the update the single whole-matrix instance produces. `test_head_sites.py`
        proves each head's delta is the bit-for-bit row slice of the whole-matrix
        delta; this is that property summed over every head -- issue #25's
        "restriction, not replacement", made whole again by running all the
        restrictions at once. A failure means the twelve-head distributed run and
        the whole-matrix run are not the same rule, so no per-head result could be
        read against the whole-matrix baseline the repo already has.
        """
        before = hf_attn(gpt2).detach().clone()

        driver = MultiSitePlasticity(gpt2, [
            SiteSpec(head(HF_ATTN, h), mode=mode, eta=self.ETA)
            for h in range(N_HEADS)
        ])
        try:
            with driver:
                drive(gpt2, r0, atr_step, n=1)
            rep = driver.apply()
            assert rep["clipped"] is False
            # Assemble the composed delta from each head's own delta attribute
            # (bit-exact), not from a weight read-back (which would fold in a
            # float32 add/sub round-trip).
            composed = torch.empty_like(before)
            for h in range(N_HEADS):
                composed[rows(h)] = driver[h].delta
            # ...and the live matrix really is W0 + that composed delta.
            assert torch.equal(hf_attn(gpt2), before + composed)
        finally:
            driver.revert()
        assert torch.equal(hf_attn(gpt2), before)

        whole_delta = self._whole_delta(gpt2, r0, atr_step, mode)

        assert whole_delta.norm().item() > 0
        assert torch.equal(composed, whole_delta)
        # The aggregate drift fraction the driver reports coincides with the whole
        # matrix's, because the head norms partition the matrix norm.
        pw_frac = (whole_delta.double().norm() / before.double().norm()).item()
        assert rep["delta_frac"] == pytest.approx(pw_frac, rel=1e-6)

    def _whole_delta(self, gpt2, r0, atr_step, mode) -> torch.Tensor:
        p = OjaPlasticity(gpt2, site=HF_ATTN, eta=self.ETA, mode=mode)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=1)
            rep = p.apply()
            assert rep["clipped"] is False
            return p.delta.clone()
        finally:
            p.revert()


# --------------------------------------------------------------------------
# 6. The single-site guarantees, preserved for one of many
# --------------------------------------------------------------------------

class TestScaffoldGuaranteesSurvive:

    def test_hooks_install_and_remove_across_every_site(self, gpt2):
        """
        Two heads of one c_proj plus a separate MLP: the two head hooks land on
        the *same* module (each head hooks the whole projection), which is exactly
        why disjoint writes are needed downstream, and both must be removed. A
        leaked hook keeps slicing activations into a dead object and contaminates
        the next run.
        """
        cproj = gpt2.transformer.h[LAYER].attn.c_proj
        mlp = gpt2.transformer.h[5].mlp.c_proj
        c0, m0 = len(cproj._forward_hooks), len(mlp._forward_hooks)

        driver = MultiSitePlasticity(gpt2, [
            SiteSpec(head(HF_ATTN, 3), mode="off", eta=0.0),
            SiteSpec(head(HF_ATTN, 7), mode="off", eta=0.0),
            SiteSpec(HF_MLP_A, mode="off", eta=0.0),
        ])
        driver.install()
        assert len(cproj._forward_hooks) == c0 + 2      # two heads, one module
        assert len(mlp._forward_hooks) == m0 + 1
        driver.install()                                # idempotent
        assert len(cproj._forward_hooks) == c0 + 2
        driver.remove()
        assert len(cproj._forward_hooks) == c0
        assert len(mlp._forward_hooks) == m0
        driver.remove()                                 # second remove must not raise

    def test_context_manager_removes_on_exception(self, gpt2):
        """The documented usage is `with driver`; an engine that raises mid-loop
        must not leave any of the sites instrumented."""
        cproj = gpt2.transformer.h[LAYER].attn.c_proj
        base = len(cproj._forward_hooks)
        with pytest.raises(RuntimeError):
            with MultiSitePlasticity(gpt2, [
                SiteSpec(head(HF_ATTN, 3), mode="off", eta=0.0),
                SiteSpec(head(HF_ATTN, 7), mode="off", eta=0.0),
            ]):
                raise RuntimeError("boom")
        assert len(cproj._forward_hooks) == base

    def test_the_per_site_ceiling_is_per_site(self, gpt2, r0, atr_step):
        """
        `max_delta_frac` is a per-site guarantee, so one site clipping at a tiny
        ceiling must not scale down another site at a generous one -- and the
        aggregate must report that a site clipped rather than hide it behind the
        healthy one. Two disjoint MLPs, one eta far too high under a 1e-4 ceiling,
        the other tiny.
        """
        before_a = hf_weight(gpt2, HF_MLP_A).detach().clone()
        before_b = hf_weight(gpt2, HF_MLP_B).detach().clone()

        driver = MultiSitePlasticity(gpt2, [
            SiteSpec(HF_MLP_A, mode="oja", eta=1e3, max_delta_frac=1e-4),
            SiteSpec(HF_MLP_B, mode="oja", eta=1e-9, max_delta_frac=0.05),
        ])
        try:
            with driver:
                drive(gpt2, r0, atr_step, n=1)
            rep = driver.apply()

            clip_site, safe_site = rep["per_site"]
            assert clip_site["clipped"] is True
            assert clip_site["delta_frac"] == pytest.approx(1e-4, rel=1e-4)
            assert safe_site["clipped"] is False
            assert 0.0 < safe_site["delta_frac"] < 0.05
            assert rep["clipped"] is True                # any() over the sites
            assert torch.isfinite(hf_weight(gpt2, HF_MLP_A)).all()
        finally:
            driver.revert()
        assert torch.equal(hf_weight(gpt2, HF_MLP_A), before_a)
        assert torch.equal(hf_weight(gpt2, HF_MLP_B), before_b)

    def test_nonfinite_at_one_site_is_flagged_and_writes_nothing(
        self, gpt2, r0, atr_step
    ):
        """
        A non-finite activation must set the aggregate `nonfinite` flag and leave
        every matrix untouched -- the flag is a diagnostic in DESIGN's failure
        table, and a nan reaching one site's accumulator while the report read
        healthy is the worst outcome on a session-scoped model.
        """
        before_mlp = hf_weight(gpt2, HF_MLP_A).detach().clone()
        before_attn = hf_attn(gpt2).detach().clone()
        bad = r0.clone()
        bad[0, 0, 0] = float("inf")

        driver = MultiSitePlasticity(gpt2, [
            SiteSpec(HF_MLP_A, mode="oja", eta=1.0),
            SiteSpec(head(HF_ATTN, 3), mode="oja", eta=1.0),
        ])
        try:
            with driver:
                atr_step(gpt2, bad)
            rep = driver.apply()

            assert rep["nonfinite"] is True
            assert rep["n_applied"] == 0
            assert rep["delta_norm"] == 0.0
            assert torch.equal(hf_weight(gpt2, HF_MLP_A), before_mlp)
            assert torch.equal(hf_attn(gpt2), before_attn)
        finally:
            driver.revert()

    def test_revert_clears_the_aggregate_diagnostics(self, gpt2, r0, atr_step):
        """
        report() is the per-iteration log schema. After a clipping run, revert()
        must return the aggregate to a clean slate, or a driver reused across an
        eta sweep reports every later eta as clipped once the largest one was --
        the same trap `test_plasticity.py` guards for a single instance, now at
        the aggregate.
        """
        driver = MultiSitePlasticity(gpt2, [
            SiteSpec(HF_MLP_A, mode="oja", eta=1e3, max_delta_frac=1e-4),
            SiteSpec(HF_MLP_B, mode="oja", eta=1e3, max_delta_frac=1e-4),
        ])
        try:
            with driver:
                drive(gpt2, r0, atr_step, n=1)
            dirty = driver.apply()
            assert dirty["clipped"] is True
            assert dirty["n_applied"] == 1
            assert dirty["delta_norm"] > 0.0
        finally:
            driver.revert()

        clean = driver.report()
        assert clean["clipped"] is False
        assert clean["nonfinite"] is False
        assert clean["n_applied"] == 0
        assert clean["delta_norm"] == 0.0
        assert clean["delta_frac"] == 0.0

    def test_report_is_per_site_plus_a_consistent_aggregate(
        self, gpt2, r0, atr_step
    ):
        """
        The report contract: a `per_site` list of ordinary single-site reports,
        each in the schema DESIGN specifies, plus an aggregate whose norms are the
        float64 Frobenius norms of the stacked deltas. If the aggregate is not
        exactly `sqrt(sum of squares)`, the one number a distributed run logs is
        not the drift it claims.
        """
        specs = [
            SiteSpec(HF_MLP_A, mode="oja", eta=1e-6),
            SiteSpec(head(HF_ATTN, 3), mode="hebb", eta=1e-5),
        ]
        driver = MultiSitePlasticity(gpt2, specs)
        try:
            with driver:
                drive(gpt2, r0, atr_step, n=2)
            rep = driver.apply()

            assert set(rep) == {
                "n_sites", "sites", "n_applied", "delta_norm", "delta_frac",
                "last_update_norm", "clipped", "nonfinite", "per_site",
            }
            assert rep["n_sites"] == 2
            assert rep["sites"] == [HF_MLP_A, head(HF_ATTN, 3)]
            assert len(rep["per_site"]) == 2
            for r in rep["per_site"]:
                assert set(r) == set(REPORT_TYPES)
                for key, typ in REPORT_TYPES.items():
                    assert isinstance(r[key], typ), f"{key}: {type(r[key])}"

            # Aggregate norms are sqrt(sum of squares), recomputed independently.
            import math
            dn = math.sqrt(sum(r["delta_norm"] ** 2 for r in rep["per_site"]))
            assert rep["delta_norm"] == pytest.approx(dn, rel=1e-12)
            w0 = math.sqrt(sum(p.W0_norm ** 2 for p in driver))
            assert rep["delta_frac"] == pytest.approx(dn / w0, rel=1e-12)
            assert rep["clipped"] is False and rep["nonfinite"] is False
        finally:
            driver.revert()

    def test_repr_summarises_the_driver(self, gpt2):
        text = repr(MultiSitePlasticity(gpt2, [
            SiteSpec(HF_MLP_A), SiteSpec(HF_MLP_B), SiteSpec(head(HF_ATTN, 3)),
        ]))
        assert "MultiSitePlasticity" in text
        assert "n_sites=3" in text
