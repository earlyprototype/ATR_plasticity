"""
The offline arm -- the control that decides whether a closed-loop result means
anything.

Oja's rule converges to the dominant eigenvector of the second-moment matrix of
whatever activations pass through it, and it does that **with no feedback at
all**. The weight matrix will move and the attractors will shift regardless. So
a closed-loop run on its own measures "the rule did its job", not "the coupling
did something". The claim this project makes lives entirely in the difference
between two arms:

    closed   run the ATR loop; every k steps apply the rule to the activations
             flowing through right now. The changed weights shape the next
             activations, which shape the next update.
    offline  run the same loop FROZEN, recording those activations. Replay the
             recording through the same rule with no feedback. Install the
             resulting matrix. Re-run the loop frozen. Compare.

For that difference to be about feedback and nothing else, the arms have to
match on every other axis. `PRIOR_ART.md` ("The offline control, specified")
lists them; `MATCHED_AXES` below is that table, mechanised. `verify_arms_matched`
checks each one, and `run_matched_arms` **refuses to report a comparison** when
any of them differs -- a mismatch is not a caveat, it is a dead result.

    from atr_bridge import initial_state, make_atr_step
    from offline_control import run_matched_arms

    s = initial_state(model, prompt)
    step = make_atr_step(model, prompt, initial_norm=s.initial_norm)
    res = run_matched_arms(model, s.tensor, step, site="blocks.6.mlp",
                           n_steps=200, eta=1e-5)
    print(res.comparison["weight"]["rel_fro_diff"])

This module owns no learning rule and no loop. The rule is `OjaPlasticity`
exactly as `plasticity.py` defines it -- the same object, the same `apply()`,
the same ceiling -- and the loop is whatever `atr_step` the caller hands in.
The only thing invented here is the bookkeeping that proves the two arms are
comparable.

**A near-zero difference is a result, not a failure.** It would mean feedback
contributes nothing detectable at that eta, which is a real and publishable
finding about this substrate. Nothing here should ever be tuned to make the
number larger.

WHAT "NO FEEDBACK" MEANS, PRECISELY. There are two paths by which a weight
change can reach the next update, and they are not the same path:

    state feedback   W changes -> the loop's next state changes -> the next x
                     changes. This is the coupling the project is about.
    local recursion  W changes -> y = xW changes -> the next update changes,
                     even with x held fixed. This is internal to Oja's rule and
                     is present in ordinary offline Oja on a fixed dataset.

Replaying the *recorded* y (`y_source="recorded"`, the default, and what
`PRIOR_ART.md` literally specifies) freezes both. Replaying with y recomputed
from the recorded x and the offline arm's own drifting weight
(`y_source="recomputed"`) freezes only the state feedback and leaves the rule's
own recursion intact in both arms. `run_matched_arms` measures both, because
which one you mean changes what the difference is evidence for -- on GPT-2 small
at eta=1e-5 the two answers differ by two orders of magnitude.

THE DETECTION LIMIT, and it is not the same for the two modes. Sever the
feedback path entirely -- run the loop reading out below the site -- and:

  recomputed   the arms come out **bit-identical**. Floor zero. This is where
               a claim about feedback can live.
  recorded     the arms still differ by `diff_over_drift` 6.8e-02, with no
               feedback anywhere in the system, because the recording froze the
               rule's own recursion. That floor is *larger* than the routed
               signal at the same eta (1.9e-02). In the default mode, at this
               eta, the difference between the arms is the frozen y and not
               the coupling.

The zero floor is defended, not lucky: it holds only because `_recompute_y`
reproduces the site's fused addmm bit for bit instead of merely mathematically.
The obvious `x @ W + b` put ~5e-09 relative Frobenius under everything --
hardware-dependent (2.9e-09 and 4.8e-09 on two machines), harmless at any usable
eta, and entirely self-inflicted. `test_a_site_the_loop_does_not_route_through`
holds the line, and asserts a bound as well as the identity, because a
floating-point floor is a bound and never a value.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

# Private, deliberately: the point of this module is that the offline arm runs
# the SAME site adapter and the SAME rule object as the closed-loop arm. Any
# reimplementation here -- a second way of finding x and y, a second copy of the
# update -- would be one more axis on which the arms could silently differ.
from plasticity import OjaPlasticity, _make_site

AtrStep = Callable[[object, torch.Tensor], torch.Tensor]

__all__ = [
    "MATCHED_AXES",
    "ArmsMismatchError",
    "ActivationRecord",
    "ArmConfig",
    "ArmResult",
    "MatchedArmsResult",
    "record_frozen_activations",
    "replay_offline",
    "run_closed_loop_arm",
    "verify_arms_matched",
    "compare_weights",
    "compare_states",
    "run_matched_arms",
    "installed_weight",
]


# 2 GiB. Not a tuned number -- a wall, so that "N steps of (seq, 3072) float32"
# fails loudly at construction instead of swapping the machine to death an hour
# into a sweep. `record_frozen_activations` raises with the arithmetic in the
# message and tells you the two honest ways out (fewer steps, or float16 with
# the precision loss measured and recorded). It never subsamples: a recording
# that quietly dropped every other step is a different trajectory, and the
# arms would no longer be matched on sample order.
DEFAULT_MEMORY_BUDGET = 2 * 1024 ** 3

VALID_Y_SOURCES = ("recorded", "recomputed")


# --------------------------------------------------------------------------
# The matched-axes table
# --------------------------------------------------------------------------
# PRIOR_ART.md, "The offline control, specified". One entry per row, plus the
# three (site, mode, dtype) that are implicit there because a comparison across
# different matrices, different rules or different precisions is not a
# comparison at all.
#
# `sample_order` is the axis most likely to be misread. It is NOT "the two arms
# saw the same activation values" -- they cannot, that difference IS the
# experiment. It is "the two arms consumed their samples in the same order,
# indexed the same way", i.e. step 0's sample first, then step 1's, with nothing
# dropped and nothing repeated. A replay that subsampled or shuffled would trip
# it.

MATCHED_AXES: tuple[tuple[str, str], ...] = (
    ("site", "a comparison across two different weight matrices is not a comparison"),
    ("mode", "two different learning rules are two different experiments"),
    ("eta", "different step sizes give different drift, not different mechanisms"),
    ("max_delta_frac", "the ceiling is part of the rule; a clipped arm is a different arm"),
    ("transposed", "the layout the rule writes in must be the same in both arms"),
    ("dtype", "a float64 arm and a float32 arm differ by more than feedback"),
    ("store_dtype", "a lossily stored replay carries rounding the live arm does not"),
    ("n_steps", "the arms must travel the same number of loop iterations"),
    ("apply_every", "the cadence sets which activations end up in which update"),
    ("n_updates", "the arms must travel the same number of steps"),
    ("n_samples", "the same activations must reach the rule, in the same quantity"),
    ("samples_per_update", "averaging over a different batch changes the update"),
    ("sample_order", "Oja is sequential; a reshuffled replay is a different trajectory"),
    ("w0_sha256", "same starting point"),
    ("seed", "same draws"),
    ("rng_state_sha256", "same draws, checked at the generator rather than the argument"),
    ("centring", "applied to one arm and not the other, this alone moves the fixed point"),
)


class ArmsMismatchError(RuntimeError):
    """The two arms are not comparable, so no comparison is reported.

    Carries the full per-axis verification dict on `.verification`. This is
    raised rather than returned on purpose: a mismatched comparison with a
    warning attached is exactly the kind of number that survives into a
    write-up with the warning stripped off.
    """

    def __init__(self, verification: dict):
        self.verification = verification
        bad = ", ".join(a["axis"] for a in verification["mismatched"])
        super().__init__(
            f"closed-loop and offline arms differ on: {bad}. "
            "The difference between them is not evidence about feedback, so no "
            "comparison is reported. See .verification for the per-axis detail."
        )


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------

def _sha256(t: torch.Tensor) -> str:
    """Content hash of a tensor's raw bytes, for the initial-weight axis.

    Bytes, not values: two matrices that agree to float32 print precision but
    differ in the last bit are not the same starting point, and the whole reason
    this axis is checked is that "same starting point" has to mean exactly that.
    """
    a = t.detach().cpu().contiguous()
    try:
        buf = a.numpy().tobytes()
    except TypeError:           # dtypes numpy has no view of, e.g. bfloat16
        buf = a.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _centring_of(plast: OjaPlasticity) -> str:
    """
    Read the rule's centring setting off the object, rather than asserting it.

    `plasticity.py` currently offers no centring option at all -- neither x nor
    y is mean-subtracted anywhere -- so today this returns "absent" for both
    arms and the axis passes by shared absence. That is the honest state of the
    check and it is written down in the returned string rather than hidden
    behind a `True`. It is read from the object so that the day someone adds a
    `centre` flag to `OjaPlasticity`, an arm configured with it and an arm
    without it stop matching *automatically*, instead of both continuing to
    report a hardcoded constant.
    """
    for name in ("centre", "center", "centring", "centering"):
        if hasattr(plast, name):
            return f"{name}={getattr(plast, name)!r}"
    return "absent (plasticity.py offers no centring option)"


def _rng_digest(plast: OjaPlasticity) -> str:
    """Digest of the rule's generator state, read immediately after construction."""
    rng = getattr(plast, "_rng", None)
    if rng is None:
        return "no generator"
    return _sha256(rng.get_state())


def _site_bias(adapter) -> Optional[torch.Tensor]:
    """
    The bias added after the target matmul, if the site has one.

    Only `y_source="recomputed"` needs it. TransformerLens spells it `b_out` on
    the MLP; HuggingFace's Conv1D and `nn.Linear` both spell it `bias`.
    """
    for name in ("b_out", "bias"):
        b = getattr(adapter.module, name, None)
        if torch.is_tensor(b) and b.dim() == 1:
            return b.detach()
    return None


def _recompute_y(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """
    `y` from `x` and the current weight, matching the site's own arithmetic bit
    for bit -- which is not the same thing as matching it mathematically.

    `torch.addmm(b, x, W)`, not `x @ W + b`. The two agree to float32 rounding
    and differ in the last bits, because the fused kernel accumulates the bias
    inside the reduction and the unfused pair does not. Measured on GPT-2's
    blocks.6.mlp: max |diff| 1.9e-06 on y, which propagates into the offline
    arm's updates and compounds -- it put a ~5e-09 relative-Frobenius floor
    under the whole comparison until this was fixed, which was within an order
    of magnitude of nothing and about eight orders below the eta=1e-5 signal,
    but it was a floor that did not need to exist.

    Both backends this module can attach to a real model are addmm underneath,
    which is why matching it is exact rather than merely closer:
    TransformerLens's `batch_addmm` flattens to 2-D and calls `torch.addmm`, and
    it is written that way precisely to match HuggingFace's `Conv1D`, which does
    the same. Verified bit-exact against `blocks.6.mlp` on GPT-2 small.

    The `transposed=True` path is `nn.Linear`, whose `F.linear` is also addmm
    but against a transposed weight -- which is the view `_effective_W()` hands
    over, so the shapes line up. **Bit-exactness there is untested**, because
    GPT-2 is Conv1D throughout and this repo has no real-model `nn.Linear` site.
    Do not assume the zero floor transfers to that path without measuring it.

    If a future site is not addmm-shaped, this is the function to change, and
    `test_a_site_the_loop_does_not_route_through` is what will tell you: it
    asserts bit-identity with the feedback path severed, and that assertion is
    only reachable while this matches.
    """
    if bias is None:
        return x @ w
    return torch.addmm(bias, x, w)


@contextmanager
def installed_weight(model, site: str, w: torch.Tensor):
    """
    Temporarily write `w` into the site's matrix, restoring on the way out.

    Used for the frozen re-runs: "install the resulting matrix, re-run the loop
    frozen". Restores in a `finally`, because a re-run that raises with an
    experimental matrix installed silently poisons every later measurement on a
    session-scoped model.
    """
    adapter = _make_site(model, site)
    before = adapter.weight.detach().clone()
    adapter.write(w)
    try:
        yield adapter
    finally:
        adapter.write(before)


def _frozen_trajectory(model, r0: torch.Tensor, atr_step: AtrStep, n_steps: int) -> list[torch.Tensor]:
    """`n_steps` of the loop with nothing installed and nothing changing."""
    r = r0.clone()
    out = []
    for _ in range(n_steps):
        r = atr_step(model, r)
        out.append(r.detach().clone())
    return out


# --------------------------------------------------------------------------
# 1. The recorder
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ActivationRecord:
    """
    Every (x, y) pair the rule would have consumed, in order, from a frozen run.

    `x` and `y` are already in the shape and layout `OjaPlasticity._hook`
    reduces them to -- flattened over batch and position to (N, n_in) and
    (N, n_out) -- so a replay hands the rule exactly what a live forward would
    have. Nothing is averaged, nothing is dropped.

    `step_index[i]` is the loop iteration that produced sample `i`. It is stored
    per sample rather than per step because one iteration is not guaranteed to
    fire the site exactly once, and the sample-order axis has to survive the
    case where it does not.
    """

    site: str
    step_index: tuple[int, ...]
    x: tuple[torch.Tensor, ...]
    y: tuple[torch.Tensor, ...]
    n_steps: int
    store_dtype: torch.dtype
    weight_dtype: torch.dtype
    bytes_stored: int
    # Empty when store_dtype is the weight dtype (nothing was rounded).
    # Otherwise the worst relative Frobenius error the round-trip introduced,
    # measured on the samples themselves rather than assumed from the format.
    precision: dict = field(default_factory=dict)
    states: tuple[torch.Tensor, ...] = ()

    @property
    def n_samples(self) -> int:
        return len(self.x)

    def samples_for_step(self, i: int) -> list[int]:
        return [j for j, s in enumerate(self.step_index) if s == i]

    def __repr__(self) -> str:
        mb = self.bytes_stored / 1024 ** 2
        return (
            f"ActivationRecord(site={self.site!r}, steps={self.n_steps}, "
            f"samples={self.n_samples}, dtype={self.store_dtype}, {mb:.1f} MiB)"
        )


def record_frozen_activations(
    model,
    r0: torch.Tensor,
    atr_step: AtrStep,
    site: str,
    n_steps: int,
    *,
    store_dtype: Optional[torch.dtype] = None,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET,
    keep_states: bool = True,
) -> ActivationRecord:
    """
    Run the loop FROZEN for `n_steps` and capture what the rule would have seen.

    Capture goes through the same `_SiteAdapter` the rule installs on, so x and
    y are read at the same two points and paired by the same logic. The weight
    is never written: this is the frozen trajectory, and `states` is it.

    Memory is the one thing to think about here. A (10, 3072) x plus a
    (10, 768) y is 150 KiB per step in float32; at seq 512 it is 7.5 MiB per
    step and 200 steps is 1.5 GiB. The projected total is computed from the
    first step and checked against `memory_budget_bytes` **before** the second
    step runs, so an over-budget recording fails in seconds rather than at the
    end.

    Two ways out, and they are not equivalent. **Fewer steps keeps the arms
    matched; `store_dtype=torch.float16` does not.** Half-precision storage is
    offered, and its measured round-trip error lands in `.precision` and travels
    with the record -- but `store_dtype` is a matched axis, because a replay
    carrying float16 rounding that the live arm never saw differs from the
    closed-loop arm by more than feedback. `run_matched_arms` will refuse the
    comparison. Use it to record for inspection or for replay experiments, not
    to shrink the control arm of a real measurement.

    Subsampling is not offered at all: a thinned recording is a different sample
    order, and the arms would stop being matched on an axis nothing can see
    afterwards.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    adapter = _make_site(model, site)
    weight_dtype = adapter.weight.dtype
    store_dtype = weight_dtype if store_dtype is None else store_dtype

    w_before = adapter.weight.detach().clone()

    pending: list[tuple[torch.Tensor, torch.Tensor]] = []

    def sink(_module, inputs, output):
        # Deliberately the same reduction `OjaPlasticity._hook` performs, and
        # in the same order: detach, flatten to 2-D, cast to the weight's dtype.
        # Capturing anything else would mean the replay feeds the rule something
        # the live hook never would have.
        x, y = inputs[0], output
        if not (torch.is_tensor(x) and torch.is_tensor(y)):
            return
        pending.append((
            x.detach().reshape(-1, x.shape[-1]).to(weight_dtype),
            y.detach().reshape(-1, y.shape[-1]).to(weight_dtype),
        ))

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    idx: list[int] = []
    states: list[torch.Tensor] = []
    n_bytes = 0
    worst_rel = 0.0
    checked_budget = False

    handle = adapter.install(sink)
    try:
        r = r0.clone()
        for i in range(n_steps):
            pending.clear()
            r = atr_step(model, r)
            if keep_states:
                states.append(r.detach().clone())
            for x, y in pending:
                # `.clone()` because `.to()` and `.cpu()` are both no-ops when
                # the dtype and device already match, and would hand back a
                # view of the forward pass's own buffer. Nothing mutates those
                # today; a recording that silently aliased live activations
                # would be a bad thing to find out about later.
                xs_i = x.to(store_dtype).cpu().clone()
                ys_i = y.to(store_dtype).cpu().clone()
                if store_dtype != weight_dtype:
                    worst_rel = max(
                        worst_rel,
                        _round_trip_rel(x, xs_i, weight_dtype),
                        _round_trip_rel(y, ys_i, weight_dtype),
                    )
                xs.append(xs_i)
                ys.append(ys_i)
                idx.append(i)
                n_bytes += xs_i.numel() * xs_i.element_size()
                n_bytes += ys_i.numel() * ys_i.element_size()
            # Gated on the first step that actually produced samples, NOT on
            # step 0. A site that does not fire on the first iteration -- one
            # that starts firing from step 1, or a loop whose first iteration
            # short-circuits -- would otherwise skip the wall entirely, which
            # is exactly the case the wall exists for. The projection is over
            # the steps still to come, not all of them, because `n_bytes`
            # already counts what is held.
            if n_bytes and not checked_budget:
                checked_budget = True
                projected = n_bytes * (n_steps - i)
                if projected > memory_budget_bytes:
                    raise MemoryError(
                        f"recording {n_steps} steps at {site} in {store_dtype} would "
                        f"hold ~{projected / 1024 ** 3:.2f} GiB "
                        f"({n_bytes / 1024 ** 2:.2f} MiB/step from step {i}), over the "
                        f"{memory_budget_bytes / 1024 ** 3:.2f} GiB budget. Lower "
                        "n_steps (the only way out that keeps the arms matched), "
                        "or raise memory_budget_bytes deliberately. "
                        "store_dtype=torch.float16 halves it and records its "
                        "round-trip error into .precision, but store_dtype is a "
                        "matched axis and run_matched_arms will refuse the "
                        "comparison. This never subsamples: a thinned recording "
                        "is a different sample order, and the arms would stop "
                        "being matched on an axis nothing can see afterwards."
                    )
    finally:
        # Exactly the inverse of install(), never a model-wide reset -- see the
        # hook-hygiene note in plasticity.py. The ATR engine's own injection
        # hook must survive this.
        adapter.remove(handle)

    # RuntimeError, not `assert`: `python -O` strips asserts, and this is the
    # invariant the entire offline arm rests on. A check whose failure would
    # invalidate every number downstream must not be optional at runtime.
    if not torch.equal(adapter.weight, w_before):
        raise RuntimeError(
            f"the recording run modified {site}; it must be frozen. The "
            "recording is of a loop the closed-loop arm never ran, so the two "
            "arms would differ by the instrument as well as by feedback."
        )

    precision: dict = {}
    if store_dtype != weight_dtype:
        precision = {
            "store_dtype": str(store_dtype),
            "weight_dtype": str(weight_dtype),
            "max_rel_round_trip_error": worst_rel,
            "note": (
                "samples were stored lossily and cast back to the weight dtype "
                "on replay; the offline arm's updates carry this error and the "
                "closed-loop arm's do not"
            ),
        }

    return ActivationRecord(
        site=site,
        step_index=tuple(idx),
        x=tuple(xs),
        y=tuple(ys),
        n_steps=n_steps,
        store_dtype=store_dtype,
        weight_dtype=weight_dtype,
        bytes_stored=n_bytes,
        precision=precision,
        states=tuple(states),
    )


def _round_trip_rel(original: torch.Tensor, stored: torch.Tensor, dtype: torch.dtype) -> float:
    """Relative Frobenius error of storing `original` as `stored`, in float64."""
    o = original.double()
    n = o.norm().item()
    if n == 0.0:
        return 0.0
    return (o - stored.to(dtype).double()).norm().item() / n


# --------------------------------------------------------------------------
# 2 & 3. The two arms
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmConfig:
    """
    Everything the two arms have to agree on, as first-class fields.

    Every name in `MATCHED_AXES` is an attribute here, and `axis_values()`
    returns exactly that subset. The three fields NOT in `MATCHED_AXES` --
    `arm`, `feedback` and `y_source` -- are the ones that are *expected* to
    differ, because they are what distinguishes the arms. Everything else on
    this class is checked.
    """

    arm: str
    feedback: bool
    site: str
    mode: str
    eta: float
    max_delta_frac: float
    transposed: bool
    dtype: str
    seed: Optional[int]
    rng_state_sha256: str
    w0_sha256: str
    centring: str
    n_steps: int
    apply_every: int
    n_updates: int
    n_samples: int
    samples_per_update: tuple[int, ...]
    sample_order: tuple[int, ...]
    y_source: str
    store_dtype: str

    def axis_values(self) -> dict:
        return {name: getattr(self, name) for name, _ in MATCHED_AXES}


@dataclass(frozen=True)
class ArmResult:
    """One arm's outcome: its config, its final matrix, and the rule's own log."""

    config: ArmConfig
    weight: torch.Tensor
    w0: torch.Tensor
    report: dict
    states: tuple[torch.Tensor, ...] = ()

    @property
    def delta(self) -> torch.Tensor:
        return self.weight - self.w0


def _arm_config(
    plast: OjaPlasticity,
    *,
    arm: str,
    feedback: bool,
    seed: Optional[int],
    rng_digest: str,
    w0_sha: str,
    centring: str,
    n_steps: int,
    apply_every: int,
    samples_per_update: Sequence[int],
    sample_order: Sequence[int],
    y_source: str,
    store_dtype: str,
) -> ArmConfig:
    return ArmConfig(
        arm=arm,
        feedback=feedback,
        site=plast.site,
        mode=plast.mode,
        eta=plast.eta,
        max_delta_frac=plast.max_delta_frac,
        transposed=plast.transposed,
        dtype=str(plast.W0.dtype),
        seed=seed,
        rng_state_sha256=rng_digest,
        w0_sha256=w0_sha,
        centring=centring,
        n_steps=n_steps,
        apply_every=apply_every,
        n_updates=plast.n_applied,
        n_samples=len(sample_order),
        samples_per_update=tuple(samples_per_update),
        sample_order=tuple(sample_order),
        y_source=y_source,
        store_dtype=store_dtype,
    )


def run_closed_loop_arm(
    model,
    r0: torch.Tensor,
    atr_step: AtrStep,
    site: str,
    n_steps: int,
    *,
    eta: float,
    mode: str = "oja",
    max_delta_frac: float = 0.05,
    transposed: bool = False,
    seed: Optional[int] = 0,
    apply_every: int = 1,
    keep_states: bool = True,
) -> ArmResult:
    """
    The experimental arm: plasticity live inside the loop.

    Every `apply_every` iterations the accumulated update is committed, and the
    changed matrix is what the next iteration runs through. The weight is
    restored before returning -- the final matrix comes back in the result, not
    left installed on the model.
    """
    plast = OjaPlasticity(
        model,
        site=site,
        eta=eta,
        mode=mode,
        cadence=apply_every,
        max_delta_frac=max_delta_frac,
        transposed=transposed,
        seed=seed,
    )
    w0_sha = _sha256(plast.W0)
    rng_digest = _rng_digest(plast)
    centring = _centring_of(plast)

    samples_per_update: list[int] = []
    sample_order: list[int] = []
    states: list[torch.Tensor] = []

    plast.install()
    try:
        r = r0.clone()
        for i in range(n_steps):
            before = plast._n_batches
            r = atr_step(model, r)
            # How many times the site actually fired this iteration, read off
            # the rule's own counter rather than assumed to be one. If a loop
            # ever runs the model twice per step this stays honest and the
            # sample-order axis keeps meaning what it says.
            sample_order.extend([i] * (plast._n_batches - before))
            if keep_states:
                states.append(r.detach().clone())
            if (i + 1) % apply_every == 0:
                batched = plast._n_batches
                plast.apply()
                if batched:
                    samples_per_update.append(batched)
        weight = (plast.W0 + plast.delta).detach().clone()
        report = plast.report()
        config = _arm_config(
            plast,
            arm="closed_loop",
            feedback=True,
            seed=seed,
            rng_digest=rng_digest,
            w0_sha=w0_sha,
            centring=centring,
            n_steps=n_steps,
            apply_every=apply_every,
            samples_per_update=samples_per_update,
            sample_order=sample_order,
            y_source="live",
            store_dtype=str(plast.W0.dtype),
        )
        w0 = plast.W0.detach().clone()
    finally:
        # Order matters: remove the hooks, then put the matrix back. Both
        # unconditional -- a raised step must not leave a modified GPT-2 behind
        # for whatever runs next.
        plast.remove()
        plast.revert()

    return ArmResult(config=config, weight=weight, w0=w0, report=report,
                     states=tuple(states))


def replay_offline(
    model,
    record: ActivationRecord,
    *,
    eta: float,
    mode: str = "oja",
    max_delta_frac: float = 0.05,
    transposed: bool = False,
    seed: Optional[int] = 0,
    apply_every: int = 1,
    y_source: str = "recorded",
) -> ArmResult:
    """
    The control arm: the same rule, the same samples, the same order, no loop.

    No forward pass happens here. The recorded (x, y) pairs are handed straight
    to `OjaPlasticity._hook` -- the same entry point a live forward hook calls,
    with the same arguments -- and `apply()` fires on the same cadence, over the
    same number of samples per update, as the closed-loop arm.

    (`_hook` is private. It is used anyway, because the alternative is a second
    implementation of the accumulation step, and a second implementation is a
    one more axis on which the arms could silently differ, and one the verifier
    could not see. The requirement this implies for `plasticity.py` -- a public
    `observe(x, y)` that `_hook` itself calls -- is recorded in
    `EXP_001_SPEC.md`. Until it exists, a change to `_hook`'s signature breaks
    the offline arm silently, and
    `test_a_single_step_replays_to_the_same_update_bit_exactly` is what catches
    it.)

    `y_source`:
      "recorded"   replay y as captured from the frozen loop. Freezes both the
                   state feedback and Oja's own y = xW recursion. This is what
                   PRIOR_ART.md specifies literally.
      "recomputed" recompute y from the recorded x and this arm's current
                   weight, via `_recompute_y`, which matches the site's fused
                   addmm bit for bit rather than merely mathematically. Freezes
                   the state feedback only, leaving the rule's internal
                   recursion live in both arms, which is the stricter isolation
                   of the feedback path -- and the one under which a loop that
                   does not route through the site gives bit-identical arms.
    """
    if y_source not in VALID_Y_SOURCES:
        raise ValueError(f"y_source must be one of {VALID_Y_SOURCES}, got {y_source!r}")

    plast = OjaPlasticity(
        model,
        site=record.site,
        eta=eta,
        mode=mode,
        cadence=apply_every,
        max_delta_frac=max_delta_frac,
        transposed=transposed,
        seed=seed,
    )
    w0_sha = _sha256(plast.W0)
    rng_digest = _rng_digest(plast)
    centring = _centring_of(plast)

    bias = _site_bias(plast._site) if y_source == "recomputed" else None

    samples_per_update: list[int] = []
    sample_order: list[int] = []

    # No hooks are installed: nothing runs forward. The weight still gets
    # written by apply(), so the revert in the finally is not optional.
    try:
        for i in range(record.n_steps):
            for j in record.samples_for_step(i):
                x = record.x[j].to(plast.W0.dtype)
                if y_source == "recorded":
                    y = record.y[j].to(plast.W0.dtype)
                else:
                    y = _recompute_y(x, plast._effective_W(), bias)
                plast._hook(plast.module, (x,), y)
                sample_order.append(i)
            if (i + 1) % apply_every == 0:
                batched = plast._n_batches
                plast.apply()
                if batched:
                    samples_per_update.append(batched)
        weight = (plast.W0 + plast.delta).detach().clone()
        report = plast.report()
        config = _arm_config(
            plast,
            arm="offline",
            feedback=False,
            seed=seed,
            rng_digest=rng_digest,
            w0_sha=w0_sha,
            centring=centring,
            n_steps=record.n_steps,
            apply_every=apply_every,
            samples_per_update=samples_per_update,
            sample_order=sample_order,
            y_source=y_source,
            store_dtype=str(record.store_dtype),
        )
        w0 = plast.W0.detach().clone()
    finally:
        plast.revert()

    return ArmResult(config=config, weight=weight, w0=w0, report=report)


# --------------------------------------------------------------------------
# 4. The verifier
# --------------------------------------------------------------------------

def _brief(value) -> str:
    """A comparable field, short enough to print in a verdict table."""
    if isinstance(value, tuple) and len(value) > 8:
        head = ", ".join(repr(v) for v in value[:4])
        return f"<{len(value)} items: {head}, ... sha256={_sha256_bytes(repr(value).encode())[:12]}>"
    return repr(value)


def verify_arms_matched(closed: ArmConfig, offline: ArmConfig) -> dict:
    """
    Hard pass/fail on whether the two arms differ by feedback and nothing else.

    Checks every row of `MATCHED_AXES` mechanically, by reading the two configs
    -- not by trusting that they were constructed from the same arguments. The
    returned dict is the audit trail: one entry per axis, with both values and
    the reason the axis is on the list.

    `ok` False means the difference between the arms is not evidence about
    feedback. `run_matched_arms` turns that into an exception rather than a
    footnote.
    """
    axes = []
    for name, why in MATCHED_AXES:
        a = getattr(closed, name)
        b = getattr(offline, name)
        axes.append({
            "axis": name,
            "closed": _brief(a),
            "offline": _brief(b),
            "match": bool(a == b),
            "why": why,
        })

    mismatched = [a for a in axes if not a["match"]]

    # The arms must also actually BE the two arms. A "comparison" of a closed
    # arm against a second closed arm passes every axis above and measures
    # nothing.
    structural = []
    if closed.feedback is not True or offline.feedback is not False:
        structural.append(
            f"arms are not one-with-feedback and one-without "
            f"(closed.feedback={closed.feedback}, offline.feedback={offline.feedback})"
        )
    if closed.arm == offline.arm:
        structural.append(f"both arms are labelled {closed.arm!r}")

    ok = not mismatched and not structural
    return {
        "ok": ok,
        "axes": axes,
        "mismatched": mismatched,
        "structural_problems": structural,
        "verdict": (
            "MATCHED -- the arms differ by feedback alone"
            if ok else
            "MISMATCHED -- the difference between these arms is not about feedback"
        ),
    }


# --------------------------------------------------------------------------
# Comparison metrics
# --------------------------------------------------------------------------

def compare_weights(closed: torch.Tensor, offline: torch.Tensor, w0: torch.Tensor) -> dict:
    """
    How far apart the two arms' matrices ended up. Everything in float64.

    Read `cos_delta` and `rel_fro_diff`, in that order, and read them against
    the drifts:

      cos_weight       cosine between the two full matrices. Near 1.0 by
                       construction whenever the drift is small -- both are W0
                       plus something tiny -- and therefore close to
                       uninformative. Reported because it is the number people
                       ask for.
      cos_delta        cosine between the two *changes*. This is the one that
                       says whether the arms moved in the same direction.
      rel_fro_diff     ||W_closed - W_offline||_F / ||W_0||_F. The headline.
      diff_over_drift  the same difference against the larger of the two arms'
                       own drifts. ~0 means feedback contributed nothing
                       detectable; ~1 means the arms went somewhere different.
    """
    a = closed.detach().double()
    b = offline.detach().double()
    z = w0.detach().double()
    da, db = a - z, b - z

    w0_norm = z.norm().item()
    diff = (a - b).norm().item()
    da_n, db_n = da.norm().item(), db.norm().item()

    def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
        nu, nv = u.norm().item(), v.norm().item()
        if nu == 0.0 or nv == 0.0:
            return float("nan")
        return (u.flatten() @ v.flatten()).item() / (nu * nv)

    return {
        "bit_identical": bool(torch.equal(closed, offline)),
        "cos_weight": _cos(a, b),
        "cos_delta": _cos(da, db),
        "fro_diff": diff,
        "rel_fro_diff": diff / w0_norm if w0_norm else float("nan"),
        "drift_closed_rel": da_n / w0_norm if w0_norm else float("nan"),
        "drift_offline_rel": db_n / w0_norm if w0_norm else float("nan"),
        "diff_over_drift": diff / max(da_n, db_n) if max(da_n, db_n) > 0 else float("nan"),
        "w0_fro": w0_norm,
    }


def compare_states(closed: torch.Tensor, offline: torch.Tensor) -> dict:
    """Cosine and relative L2 between two loop states, in float64."""
    a = closed.detach().double().flatten()
    b = offline.detach().double().flatten()
    na, nb = a.norm().item(), b.norm().item()
    d = (a - b).norm().item()
    return {
        "bit_identical": bool(torch.equal(closed, offline)),
        "cos": (a @ b).item() / (na * nb) if na and nb else float("nan"),
        "l2_diff": d,
        "rel_l2_diff": d / na if na else float("nan"),
    }


# --------------------------------------------------------------------------
# 3. The matched-arms runner
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchedArmsResult:
    """Both arms, the verification, and -- only if it passed -- the comparison."""

    closed: ArmResult
    offline: ArmResult
    offline_recomputed_y: Optional[ArmResult]
    verification: dict
    comparison: dict
    record: ActivationRecord

    def summary(self) -> dict:
        """
        The flat dict worth writing to a log line.

        The unsuffixed metrics are `y_source`'s -- the literal PRIOR_ART
        protocol by default. The `_recomputed_y` pair is the one whose floor is
        zero, and is therefore the pair a claim about feedback should be read
        from; it is None when `also_recomputed_y` was off. Both are carried
        because reporting either alone is misleading in a different direction.
        """
        w = self.comparison["weight"]
        r = self.comparison.get("weight_recomputed_y")
        return {
            "site": self.closed.config.site,
            "mode": self.closed.config.mode,
            "eta": self.closed.config.eta,
            "n_steps": self.closed.config.n_steps,
            "n_updates": self.closed.config.n_updates,
            "arms_matched": self.verification["ok"],
            "bit_identical": w["bit_identical"],
            "cos_weight": w["cos_weight"],
            "cos_delta": w["cos_delta"],
            "rel_fro_diff": w["rel_fro_diff"],
            "diff_over_drift": w["diff_over_drift"],
            "drift_closed_rel": w["drift_closed_rel"],
            "drift_offline_rel": w["drift_offline_rel"],
            "cos_delta_recomputed_y": r["cos_delta"] if r else None,
            "rel_fro_diff_recomputed_y": r["rel_fro_diff"] if r else None,
            "diff_over_drift_recomputed_y": r["diff_over_drift"] if r else None,
            "clipped_closed": self.closed.report["clipped"],
            "clipped_offline": self.offline.report["clipped"],
        }


def run_matched_arms(
    model,
    r0: torch.Tensor,
    atr_step: AtrStep,
    site: str,
    n_steps: int,
    *,
    eta: float,
    mode: str = "oja",
    max_delta_frac: float = 0.05,
    transposed: bool = False,
    seed: Optional[int] = 0,
    apply_every: int = 1,
    y_source: str = "recorded",
    also_recomputed_y: bool = True,
    store_dtype: Optional[torch.dtype] = None,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET,
    rerun_frozen: bool = True,
    keep_states: bool = False,
) -> MatchedArmsResult:
    """
    Run both arms and compare them -- or refuse to, if they are not matched.

    Order, and why:

      1. record   the frozen loop, capturing what the rule would have consumed.
                  First, because it must see the untouched weights.
      2. closed   the same loop with plasticity live.
      3. offline  the recording replayed through the same rule, no feedback.
      4. verify   every row of the matched-axes table, mechanically.
      5. compare  the two matrices, and -- with `rerun_frozen` -- the loop run
                  frozen under each of them, which is the behavioural readout
                  PRIOR_ART.md actually asks for.

    Raises `ArmsMismatchError` at step 4 rather than returning a caveated
    number. Both arms have already run by then; that cost is the price of
    checking the axes that only exist after a run (update counts, batching,
    sample order) instead of trusting the arguments.

    `also_recomputed_y` runs a second offline replay with y recomputed from the
    drifting offline weight, which isolates the state-feedback path from Oja's
    own y = xW recursion. It costs no forward passes and it is the difference
    between "feedback did nothing" and "the recording froze more than feedback".
    """
    record = record_frozen_activations(
        model, r0, atr_step, site, n_steps,
        store_dtype=store_dtype,
        memory_budget_bytes=memory_budget_bytes,
        keep_states=True,
    )

    closed = run_closed_loop_arm(
        model, r0, atr_step, site, n_steps,
        eta=eta, mode=mode, max_delta_frac=max_delta_frac,
        transposed=transposed, seed=seed, apply_every=apply_every,
        keep_states=True,
    )

    offline = replay_offline(
        model, record,
        eta=eta, mode=mode, max_delta_frac=max_delta_frac,
        transposed=transposed, seed=seed, apply_every=apply_every,
        y_source=y_source,
    )

    verification = verify_arms_matched(closed.config, offline.config)
    if not verification["ok"]:
        raise ArmsMismatchError(verification)

    offline_recomputed = None
    if also_recomputed_y and y_source != "recomputed":
        offline_recomputed = replay_offline(
            model, record,
            eta=eta, mode=mode, max_delta_frac=max_delta_frac,
            transposed=transposed, seed=seed, apply_every=apply_every,
            y_source="recomputed",
        )
        v2 = verify_arms_matched(closed.config, offline_recomputed.config)
        if not v2["ok"]:
            raise ArmsMismatchError(v2)

    comparison: dict = {
        "weight": compare_weights(closed.weight, offline.weight, closed.w0),
        "frozen_baseline_final_state": None,
        "state": None,
        "closed_trajectory_vs_offline_rerun": None,
    }
    if offline_recomputed is not None:
        comparison["weight_recomputed_y"] = compare_weights(
            closed.weight, offline_recomputed.weight, closed.w0
        )
        comparison["offline_recorded_vs_recomputed_y"] = compare_weights(
            offline.weight, offline_recomputed.weight, closed.w0
        )

    if rerun_frozen:
        # "install the resulting matrix, re-run the loop frozen, and compare".
        # Both arms get the same treatment: the loop is run frozen under each
        # final matrix from the same r0, so the states being compared differ
        # only in which matrix produced them -- not in whether the weights were
        # moving while the trajectory ran.
        with installed_weight(model, site, closed.weight):
            rerun_closed = _frozen_trajectory(model, r0, atr_step, n_steps)
        with installed_weight(model, site, offline.weight):
            rerun_offline = _frozen_trajectory(model, r0, atr_step, n_steps)

        base_final = record.states[-1]
        comparison["state"] = compare_states(rerun_closed[-1], rerun_offline[-1])
        comparison["state_closed_vs_frozen_baseline"] = compare_states(
            rerun_closed[-1], base_final
        )
        comparison["state_offline_vs_frozen_baseline"] = compare_states(
            rerun_offline[-1], base_final
        )
        comparison["closed_trajectory_vs_offline_rerun"] = compare_states(
            closed.states[-1], rerun_offline[-1]
        )
        comparison["frozen_baseline_final_state"] = "record.states[-1]"

    return MatchedArmsResult(
        closed=closed if keep_states else ArmResult(
            config=closed.config, weight=closed.weight, w0=closed.w0,
            report=closed.report, states=(),
        ),
        offline=offline,
        offline_recomputed_y=offline_recomputed,
        verification=verification,
        comparison=comparison,
        record=record if keep_states else ActivationRecord(
            site=record.site, step_index=record.step_index, x=(), y=(),
            n_steps=record.n_steps, store_dtype=record.store_dtype,
            weight_dtype=record.weight_dtype, bytes_stored=0,
            precision=record.precision, states=record.states,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(
        "Import run_matched_arms and pass in your own atr_step; there is no "
        "default loop here on purpose. See EXP_001_SPEC.md, 'The offline arm'."
    )
