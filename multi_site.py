"""
Simultaneous multi-site plasticity: drive many local rules in one ATR loop.

`OjaPlasticity` moves ONE weight matrix, or one head's stripe of one. Issue #25
item 2 and DESIGN.md's "one site or many" ask for the other thing -- distributed
damping, "many small elements each acting on its own subspace": every head
plastic at once, whole matrices and per-head stripes mixed in the same loop, each
with its own rule, learning rate and ceiling. Rung 3 of issue #25's ladder ("each
head has a looping training function") is this class pointed at every head of a
block.

    from multi_site import MultiSitePlasticity, SiteSpec
    from plasticity import TermSpec

    driver = MultiSitePlasticity(model, [
        SiteSpec("blocks.5.mlp",          mode="oja",       eta=1e-6),
        SiteSpec("blocks.6.mlp",          mode="anti_hebb", eta=3e-7),
        SiteSpec("blocks.11.attn.head.3", mode="oja",       eta=1e-8),
        SiteSpec("blocks.11.attn.head.7", mode="oja",       eta=1e-8),
    ])
    with driver:                              # installs every capture hook
        r = initial_tensor
        for i in range(n_iter):
            r = my_atr_step(model, r)         # YOUR existing engine, unchanged
            if (i + 1) % 4 == 0:
                driver.apply()                # commits every site's update
            log(driver.report())              # per-site + aggregate
    # __exit__ removes every hook; call driver.revert() to restore every matrix.

This class does NOT reimplement the learning rules. It composes N `OjaPlasticity`
instances and orchestrates them: `install()` all their capture hooks, `apply()`
all their updates on the caller's cadence, `revert()` all of them, `report()`
collectively. Everything a single-site run guarantees -- the per-site
`max_delta_frac` ceiling, bit-exact `revert()` of every touched matrix, the
`nonfinite` flag, the report schema -- each sub-instance still guarantees for its
own site, because each sub-instance IS an ordinary single-site run. Heterogeneous
config per site is the point, not an afterthought: mode, eta and ceiling are
per-`SiteSpec`, so a block can reinforce inside one head and erode in another in
the same pass (issue #25's "sign, per subspace"), and a per-site `project`
carries the subspace knob through unchanged. A per-site `terms` carries the
composed rule through as well, which is the finer grain of the same idea: with
`mode` a site is reinforcing or eroding, and the two can only be separated by
naming two sites; with `terms` ONE site does both, split by subspace rather than
by matrix.

    SiteSpec("blocks.11.attn.head.7", eta=1e-8, terms=[
        TermSpec("hebb",  +1, P),          # reinforce inside the direction
        TermSpec("hebb",  -1, I - P),      # erode around it
        TermSpec("decay", -1),             # one brake over both
    ])

WHY DISJOINT WRITES ARE CORRECT HERE -- the one property that makes a distributed
head experiment interpretable (issue #25, and `test_head_sites.py`'s ISOLATION
claim). Each `OjaPlasticity` holds its own frozen `W0` and, on `apply()`, writes
`W0 + delta` back through its site adapter. If that write covered the whole
matrix, two instances on two heads of ONE `attn.c_proj` would clobber: the second
write would reset the first head's rows to their `W0`, because the second
instance's delta is zero on those rows. It does not clobber, because the per-head
adapters write ROW-CONFINED -- `_HeadSliceSite.write` does `weight[lo:hi].copy_`
and `_TransformerLensHeadSite.write` does `W_O[head].copy_`, each touching only
its own head's rows and leaving the other eleven untouched (not rewritten to the
same value through float32: untouched). Two disjoint heads of one matrix
therefore already compose without crosstalk, which is exactly what
`test_head_sites.py::test_two_heads_can_be_driven_at_once_without_crosstalk`
established for two instances and this class relies on for as many as twelve.

The only way instances can clobber is if their write footprints OVERLAP: a whole
matrix and a head of it, or the same matrix named twice. There is no disjoint
sub-matrix write that is not already row-confined -- the whole-matrix adapters
always cover the whole matrix, and the only sub-matrix adapters are the per-head
ones, which are row-confined -- so composing deltas by hand and writing each
matrix once buys nothing that rejecting overlap does not. This class therefore
REJECTS overlapping footprints at construction and leans on the existing
row-confined head writes for everything else. Overlap is computed from the live
parameter's storage, not the site string, so the two spellings of one matrix
(`blocks.6.mlp` and its `blocks.6.mlp.W_out` alias) collapse to one key and a
whole-vs-head overlap is caught even across spellings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn

# The driver is part of the plasticity subsystem, so it is allowed to know the
# subsystem's adapters -- `offline_control.py` imports `_make_site` from here for
# the same reason. The four adapter classes are needed to read each instance's
# write footprint (which matrix, which rows) for the overlap check; that footprint
# is not on `OjaPlasticity`'s public surface because a single-site run never needs
# it. Importing them, rather than re-parsing the site string, is what makes the
# check alias-proof and layout-correct.
from plasticity import (
    OjaPlasticity,
    TermSpec,
    _HeadSliceSite,
    _TransformerLensHeadSite,
    _TransformerLensMLPSite,
    _WeightModuleSite,
)


# --------------------------------------------------------------------------
# Per-site configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteSpec:
    """
    One site's plasticity configuration -- everything `OjaPlasticity` takes but
    the model, as a first-class record so a list of them reads as the experiment.

    The fields and their defaults mirror `OjaPlasticity`'s constructor exactly,
    so a `SiteSpec` is nothing more than "these kwargs, at this site" and a
    single-site run and this site inside a driver are the same object twice. That
    parity is deliberate: it is what lets `test_head_sites.py`'s single-site rule
    tests stand as coverage of each site the driver drives.

    `mode`, `eta` and `max_delta_frac` are meant to differ between sites -- issue
    #25's whole point is many small heterogeneous elements -- and `project`
    carries the per-site subspace knob (an (n_out, n_out) orthogonal projector
    from `subspace_projector`) through untouched.

    `terms` carries the composed rule through the same way: a list of `TermSpec`
    instead of a `mode`, so a site can reinforce inside a subspace and erode
    outside it in one pass rather than needing two instances on the same rows
    (which `_reject_overlap` refuses, and rightly). It is mutually exclusive
    with a non-default `mode`, and `OjaPlasticity` is what enforces that -- the
    spec passes both through and lets the site that is wrong be the one named in
    the error. Pass it as a TUPLE if the spec is ever hashed: this dataclass is
    frozen, so a list field would make `hash(spec)` raise where a tuple will not.
    """

    site: str
    mode: str = "oja"
    eta: float = 1e-6
    cadence: int = 1
    max_delta_frac: float = 0.05
    transposed: bool = False
    seed: Optional[int] = 0
    project: Optional[torch.Tensor] = None
    terms: Optional[Sequence[Union[TermSpec, dict]]] = None

    def build(self, model: nn.Module) -> OjaPlasticity:
        """Construct the single-site instance this spec describes.

        Any bad field -- an unknown mode, an unattachable site, a `transposed`
        head, a non-idempotent projector, a `mode` and `terms` that disagree --
        raises here exactly as it would for a hand-built `OjaPlasticity`, so the
        driver fails loud at construction on the offending site rather than
        silently dropping it.
        """
        return OjaPlasticity(
            model,
            site=self.site,
            eta=self.eta,
            mode=self.mode,
            cadence=self.cadence,
            max_delta_frac=self.max_delta_frac,
            transposed=self.transposed,
            seed=self.seed,
            project=self.project,
            terms=self.terms,
        )

    @classmethod
    def coerce(cls, spec: Union["SiteSpec", dict]) -> "SiteSpec":
        """Accept a `SiteSpec` as-is, or a plain dict -- the shape the experiment
        scripts already build their cells in. An unknown key is a `TypeError`
        from the constructor, not a silently ignored field."""
        if isinstance(spec, SiteSpec):
            return spec
        if isinstance(spec, dict):
            return cls(**spec)
        raise TypeError(
            f"each site spec must be a SiteSpec or a dict of its fields, got "
            f"{type(spec).__name__}"
        )


# --------------------------------------------------------------------------
# Write footprints -- which rows of which live matrix an instance touches
# --------------------------------------------------------------------------

def _footprint(plast: OjaPlasticity) -> tuple[int, int, int]:
    """
    `(matrix_key, lo, hi)`: the half-open row interval `[lo, hi)` this instance
    writes, on the live matrix identified by `matrix_key`.

    `matrix_key` is the storage address of the underlying parameter, so two heads
    of one packed `c_proj` share a key while two MLPs on different layers do not,
    and the two spellings of one matrix collapse to the same key because they
    resolve to the same tensor. `[lo, hi)` is on the parameter's primary axis: a
    whole-matrix site spans it entirely, a per-head site spans only its stripe --
    contiguous input rows on HuggingFace, the head index on TransformerLens, both
    of which are the axis the head owns and the axis the adapter's `write`
    confines itself to.

    The intervals of two head sites are only ever compared when their keys match,
    i.e. on the same tensor's same axis, so the HuggingFace row scale (0..768) and
    the TransformerLens head-index scale (0..12) never cross.
    """
    site = plast._site
    if isinstance(site, _HeadSliceSite):
        base = site.module.weight            # the full packed Conv1D, shared by heads
        return base.data_ptr(), site._lo, site._hi
    if isinstance(site, _TransformerLensHeadSite):
        base = site.module.W_O               # (n_heads, d_head, d_model), shared by heads
        return base.data_ptr(), site.head, site.head + 1
    if isinstance(site, _WeightModuleSite):
        base = site.module.weight
        return base.data_ptr(), 0, base.shape[0]
    if isinstance(site, _TransformerLensMLPSite):
        base = site.module.W_out
        return base.data_ptr(), 0, base.shape[0]
    raise TypeError(
        f"{plast.site!r}: unrecognised site adapter {type(site).__name__}; "
        "its write footprint is unknown, so overlap cannot be checked"
    )


def _overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    """Same matrix, and the two row intervals intersect."""
    (key_a, lo_a, hi_a), (key_b, lo_b, hi_b) = a, b
    return key_a == key_b and lo_a < hi_b and lo_b < hi_a


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------

class MultiSitePlasticity:
    """
    Local plasticity on many sites at once, composed from N `OjaPlasticity`.

    Parameters
    ----------
    model : nn.Module
        The transformer. Left otherwise untouched, exactly as for a single site.
    specs : iterable of SiteSpec or dict
        One configuration per site. Whole-matrix and per-head sites may be mixed,
        the modes/etas/ceilings may all differ, and per-head sites of the same
        matrix are allowed as long as they name different heads. Passing the same
        matrix twice, or a whole matrix together with a head of it, is rejected --
        see the class of `ValueError` below.

    Raises
    ------
    ValueError
        If two specs write overlapping rows of the same matrix. That is the
        precondition the whole distributed picture rests on: with overlap, one
        instance's `apply()` would silently undo another's (whole-matrix write) or
        two rules would fight over the same rows, and "many small elements each on
        its own subspace" would no longer be true. Rejected at construction, with
        both offending sites named, rather than discovered as an un-reproducible
        result.
    """

    def __init__(
        self,
        model: nn.Module,
        specs: Iterable[Union[SiteSpec, dict]],
    ):
        specs = [SiteSpec.coerce(s) for s in specs]
        if not specs:
            raise ValueError(
                "MultiSitePlasticity needs at least one site; an empty driver "
                "would install nothing and silently no-op every apply()"
            )

        self.model = model
        self.specs: tuple[SiteSpec, ...] = tuple(specs)
        # Build every instance first, so a bad site raises before any overlap
        # arithmetic runs and the error names the site, not a footprint.
        self._plasts: tuple[OjaPlasticity, ...] = tuple(s.build(model) for s in specs)

        self._reject_overlap()

        # Aggregate reference norm: the Frobenius norm of every touched matrix
        # stacked, in float64, `sqrt(sum ||W0_i||_F^2)`. Because the sites are
        # disjoint, this is the norm of the block the driver as a whole can move,
        # and it is what the aggregate `delta_frac` is measured against. Computed
        # once, like each instance's own `W0_norm`.
        self._w0_norm = math.sqrt(sum(p.W0_norm ** 2 for p in self._plasts))

        self._installed = False
        self.n_applied = 0

    # -------------------------------------------------------------- overlap
    def _reject_overlap(self) -> None:
        prints = [(_footprint(p), p.site) for p in self._plasts]
        for i in range(len(prints)):
            fp_i, site_i = prints[i]
            for j in range(i):
                fp_j, site_j = prints[j]
                if _overlap(fp_i, fp_j):
                    raise ValueError(
                        f"sites {site_j!r} and {site_i!r} write overlapping rows "
                        f"of the same matrix. Disjoint writes are the property "
                        f"that makes a distributed run interpretable (issue #25): "
                        f"with overlap, one apply() would clobber the other. Name "
                        f"different heads of a matrix, or drop the whole-matrix "
                        f"site when a head of it is present."
                    )

    # -------------------------------------------------------------- setup
    def install(self) -> "MultiSitePlasticity":
        """Install every site's capture hook. Idempotent, because each
        `OjaPlasticity.install()` is."""
        for p in self._plasts:
            p.install()
        self._installed = True
        return self

    def remove(self) -> None:
        """Remove every hook this driver installed, and nothing else -- each
        instance removes precisely its own handles (the hook-hygiene note on
        `_TransformerLensMLPSite`), never a model-wide `reset_hooks()`."""
        for p in self._plasts:
            p.remove()
        self._installed = False

    def __enter__(self) -> "MultiSitePlasticity":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()

    # -------------------------------------------------------------- apply
    def apply(self) -> dict:
        """
        Commit every site's accumulated update. Returns `report()`.

        The instances are applied in construction order, but the order does not
        matter: their write footprints are disjoint (enforced at construction), so
        no instance's write touches a row another instance owns. Each `apply()`
        no-ops safely if its site accumulated nothing or is in `mode="off"`, so a
        cadence tick that lands before some site has seen a forward pass is
        harmless.
        """
        committed = False
        for p in self._plasts:
            before = p.n_applied
            p.apply()
            committed = committed or p.n_applied > before
        if committed:
            self.n_applied += 1
        return self.report()

    def revert(self) -> None:
        """
        Restore every touched matrix to its `W0`, bit-exactly, and reset every
        site's diagnostics -- each instance reverts its own site, and because the
        writes are row-confined and disjoint, reverting all of them restores every
        matrix (whole or striped) to the bits it started with. The driver's own
        `n_applied` resets too, matching every field `report()` prints.
        """
        for p in self._plasts:
            p.revert()
        self.n_applied = 0

    # -------------------------------------------------------------- report
    def report(self) -> dict:
        """
        Per-site reports plus an aggregate over all of them.

        `per_site` is the list of ordinary single-site `report()` dicts, unchanged
        and in construction order. The aggregate collapses them the way the
        distributed picture reads them:

        - `delta_norm` / `last_update_norm` are the float64 Frobenius norms of the
          stacked deltas / updates, `sqrt(sum n_i^2)` -- the norm of the whole
          disjoint block the driver moved.
        - `delta_frac` is that against `sqrt(sum ||W0_i||_F^2)`, so for the twelve
          heads of one `attn.c_proj` it coincides with the whole matrix's
          `delta_frac` (the head norms partition the matrix norm).
        - `clipped` / `nonfinite` are `any` over the sites: one clipped or
          poisoned site is a fact about the run the operator has to see, and an
          aggregate that hid it behind eleven healthy sites would be the worst
          kind of green.
        """
        per = [p.report() for p in self._plasts]
        delta_norm = math.sqrt(sum(r["delta_norm"] ** 2 for r in per))
        last_update_norm = math.sqrt(sum(r["last_update_norm"] ** 2 for r in per))
        return {
            "n_sites": len(per),
            "sites": [r["site"] for r in per],
            "n_applied": self.n_applied,
            "delta_norm": delta_norm,
            "delta_frac": delta_norm / self._w0_norm if self._w0_norm else float("nan"),
            "last_update_norm": last_update_norm,
            "clipped": any(r["clipped"] for r in per),
            "nonfinite": any(r["nonfinite"] for r in per),
            "per_site": per,
        }

    # -------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self._plasts)

    def __getitem__(self, i: int) -> OjaPlasticity:
        """The site's underlying `OjaPlasticity`, for reading its own `delta`,
        `report()` or `W0` -- the driver orchestrates them, it does not hide
        them."""
        return self._plasts[i]

    def __iter__(self):
        return iter(self._plasts)

    @property
    def sites(self) -> list[str]:
        return [p.site for p in self._plasts]

    def __repr__(self) -> str:
        r = self.report()
        return (
            f"MultiSitePlasticity(n_sites={r['n_sites']}, applied={r['n_applied']}, "
            f"delta_frac={r['delta_frac']:.3e}, clipped={r['clipped']}, "
            f"nonfinite={r['nonfinite']})"
        )
