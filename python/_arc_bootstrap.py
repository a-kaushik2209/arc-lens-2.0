"""
ARC Lens — instrumentation bootstrap
====================================

Imported as the *first line* of the user's training script by ``runner.py``.
Everything the harness needs lives in this module rather than in a text header
prepended to the user's source, so a traceback in ``train.py`` still reports the
line number that is actually in ``train.py``.

What it hooks and why
---------------------

The measurement anchor is ``Optimizer.step``, not ``Tensor.backward``:

* ``self`` is definitionally the optimizer being stepped, so there is no guessing.
* ``self.param_groups`` are definitionally the parameters being updated, so the
  gradient norm is computed over the right tensors.
* one call is one *weight update*, which is what a training step means — this is
  what makes gradient accumulation and multi-optimizer (GAN) setups correct
  rather than merely not-crashing.

``Tensor.backward`` and ``GradScaler.scale`` are still patched, but only to
*record* which loss belongs to the update that is about to happen. They never
emit. Under AMP, ``GradScaler.scale(loss)`` hands us the **unscaled** loss
directly, which is the only place that value is available without guessing at
the scale factor.

Concrete optimizers (``Adam``, ``SGD``, …) each define their own ``step``, so
patching ``Optimizer.step`` on the base class would miss all of them. Instead
``Optimizer.__init__`` is patched and each *instance* gets its ``step`` wrapped.
That covers every subclass, including third-party and custom optimizers.

The wrapper is installed with ``types.MethodType``, and that detail is load
bearing. ``LRScheduler.__init__`` reaches into ``optimizer.step.__func__`` to
install its own bookkeeping; a plain function has no ``__func__``, so assigning
one made *constructing any scheduler* raise ``AttributeError`` before training
started — on the majority of real scripts. Binding it as a method keeps the
wrapper indistinguishable from the method it replaced. (An earlier version of
this docstring claimed instance-wrapping was "the same technique PyTorch's own
LR schedulers use". It is the technique those schedulers reject.)
"""

from __future__ import annotations

import json
import math
import statistics
import os
import sys
import time
import types
from collections import OrderedDict, deque

# ─────────────────────────────────────────────────────────────────────────────
# Transport — newline-delimited JSON on stdout, read by the extension host
# ─────────────────────────────────────────────────────────────────────────────

_stdout = sys.stdout


_transport_open = True


def emit(event: dict) -> None:
    """Write one event. Never raises into the training loop.

    If the extension host closes its read end while the run is still going —
    the panel is disposed, the window reloads, the user pipes stdout to `head` —
    the next write raises BrokenPipeError. That used to escape the step hook and
    kill the user's training, and the error handler could not report it either,
    because reporting goes through this same function. A monitor must not be able
    to take down the run it is monitoring, least of all while shutting down.
    """
    global _transport_open
    if not _transport_open:
        return
    if event.get("type") == "metric" and _finished:
        # finish() has already sent the final run_summary/status pair. A
        # straggling optimizer.step() firing after that (e.g. from code that
        # keeps training past the user's own finish() call) must not resume
        # the metric stream once the run has been declared over.
        return
    try:
        print(json.dumps(event), file=_stdout, flush=True)
    except (BrokenPipeError, OSError, ValueError):
        # Nothing is listening any more. Go quiet for the rest of the run rather
        # than raising once per step.
        _transport_open = False


def emit_log(message: str, level: str = "info") -> None:
    emit({"type": "log", "level": level, "message": message})


# ─────────────────────────────────────────────────────────────────────────────
# Degradation — a broken collector must never look like a healthy run
# ─────────────────────────────────────────────────────────────────────────────

_warned: set[str] = set()
_degraded: list[str] = []


def warn_once(key: str, message: str) -> None:
    """Report the first occurrence of each distinct instrumentation failure.

    Repeats are suppressed so a per-step exception cannot flood the transport,
    but the first one is always surfaced and the run is flagged DEGRADED in the
    dashboard header. Silent degradation is the one failure mode a reliability
    tool must not have: it turns "ARC is broken" into "your training looks fine".
    """
    if key in _warned:
        return
    _warned.add(key)
    _degraded.append(key)
    emit_log(message, "warning")
    emit({"type": "degraded", "component": key, "message": message})


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (environment, set by the extension host)
# ─────────────────────────────────────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# "baseline" suppresses every intervention while still emitting full telemetry.
# This is what makes an honest A/B possible: same script, same seed, same
# instrumentation, interventions on vs off.
MODE = os.environ.get("ARC_MODE", "active").lower()
INTERVENTIONS_ENABLED = MODE != "baseline"

STEP_DELAY = _env_float("ARC_STEP_DELAY", 0.0)
CHECKPOINT_EVERY = _env_int("ARC_CHECKPOINT_EVERY", 25)
MAX_CHECKPOINTS = _env_int("ARC_MAX_CHECKPOINTS", 3)
# Storage budget for the checkpoint ring buffer, independent of the count-based
# MAX_CHECKPOINTS above. Whichever cap is tighter wins — see
# CheckpointStore._recompute_effective_max.
MAX_CHECKPOINT_MB = _env_float("ARC_MAX_CHECKPOINT_MB", 512.0)
# Effective rank needs an SVD per layer. That is far too expensive to run every
# step, and instability signatures develop over tens of steps anyway, so the
# expensive collectors are sampled. Sampling goes dense automatically while risk
# is elevated (see _should_sample_advanced).
# Measured on an RTX 3050 with a 2.79M-parameter CNN (see
# python/benchmark_overhead.py and docs/benchmark_overhead.json): sampling the
# expensive collectors every 25 steps costs 8.4% wall-clock, while sampling them
# every step costs 170%. Instability signatures develop over tens of steps, so
# the dense setting buys nothing but cost.
ADVANCED_EVERY = _env_int("ARC_ADVANCED_EVERY", 25)
# The rate used while risk is elevated. Five times denser than normal, not
# every step — see _should_sample_advanced for why that distinction matters.
DENSE_ADVANCED_EVERY = max(1, _env_int("ARC_ADVANCED_EVERY_DENSE", max(1, ADVANCED_EVERY // 5)))
# Same coalesce-while-healthy / densify-while-risky policy, applied to the
# per-step `metric` event instead of the expensive collectors. Defaults to 1
# (emit every step) so existing users see no change unless they opt in.
METRIC_EVERY = _env_int("ARC_METRIC_EVERY", 1)
DENSE_METRIC_EVERY = max(1, _env_int("ARC_METRIC_EVERY_DENSE", 1))
LOSS_EXPLOSION_THRESHOLD = _env_float("ARC_LOSS_EXPLOSION", 1e6)
COOLDOWN_STEPS = _env_int("ARC_COOLDOWN_STEPS", 15)

# Silent-failure detection. Thresholds are fractions of the run's own baseline,
# not absolutes, because the scale of entropy and effective rank is entirely
# architecture-dependent.
STRUCTURAL_BASELINE_SAMPLES = _env_int("ARC_BASELINE_SAMPLES", 3)
STRUCTURAL_SUSTAIN = _env_int("ARC_STRUCTURAL_SUSTAIN", 3)
RANK_COLLAPSE_FRACTION = _env_float("ARC_RANK_COLLAPSE", 0.5)
# Structural signals move by orders of magnitude while a model goes from
# random to structured. Nothing is judged until the run is past that.
STRUCTURAL_WARMUP_STEPS = _env_int("ARC_STRUCTURAL_WARMUP", 200)

# Loss plateau — the rule that actually catches a silent death.
#
# Measured on this workload (CIFAR-10, 9-layer CNN, seed 1234, 780 steps), by
# replaying both arms' per-batch losses through the counter below:
#
#     arm                  max consecutive steps without improvement
#     lr=0.03  73.6%                      82        <- healthy
#     lr=0.50  10.0%                     764        <- dead, loss pinned at ln(10)
#
# A 9.3x gap. That is the separation the two deleted rules never had, and the
# separation the surviving rank rule could not reach: on the same pair of runs
# effective rank fell only to 87.4% of its step-1 value on the dead arm against
# 97.2% on the healthy one, nowhere near the 50% its trigger required.
#
# Patience sits 3.7x above the measured healthy maximum rather than just above
# it, because 82 is one seed on one workload and a task with noisier batches
# will stall longer while training perfectly well. The cost of being late is a
# few hundred wasted steps; the cost of being early is the entropy rule, which
# took a healthy run from 87.4% to chance.
LOSS_PLATEAU_PATIENCE = _env_int("ARC_PLATEAU_PATIENCE", 300)
# A new best must beat the old one by this much to count. Guards against a
# monotonically-but-uselessly improving loss ratcheting the counter forever.
LOSS_PLATEAU_MIN_DELTA = _env_float("ARC_PLATEAU_DELTA", 0.001)
# A plateau only counts as a failure if the run never got anywhere. Fires only
# when best_loss / first_loss is *above* this — i.e. the run has improved by
# less than 40% from its opening loss. Measured: a healthy run reaches 0.271, a
# dead one 0.888. Set at 0.60, roughly midway in log terms and far from both.
#
# Without this guard the rule fires on convergence, because the best-ever-loss
# counter makes stalls grow without bound as a run succeeds. A 10-epoch A/B
# caught it firing twice on a run that reached 87.5%.
LOSS_PLATEAU_PROGRESS_RATIO = _env_float("ARC_PLATEAU_PROGRESS", 0.60)
# The opening reference is the median of this many early losses, not the single
# first one. One batch is a sample of size one: an unlucky first batch reads high,
# which would make a dead run look like it had improved and suppress a true
# positive. The median of ten is stable against that at no meaningful cost.
LOSS_PLATEAU_OPENING_SAMPLES = _env_int("ARC_PLATEAU_OPENING", 10)

# Kinds ARC reports but never acts on.
#
# `loss_plateau` detects correctly and its only available response makes things
# worse, so it is allowed to speak and not to touch the run. The four-LR A/B
# measured the pair at lr=0.5 (seed 1234, 10 epochs, identical data order):
#
#     epoch    baseline (no action)     active (3x reduce_lr at step 316)
#         1    10.00%  lr=4.91e-01      10.00%  lr=2.45e-01
#         4     9.76%  lr=3.34e-01      10.00%  lr=4.18e-02
#         5    26.73%  lr=2.56e-01      10.00%  lr=3.20e-02
#        10    73.19%                   10.00%  loss 2.3026 = ln(10)
#
# The detection was right — the run really was pinned at chance for four
# epochs. But the control arm escaped on its own once cosine decay walked the
# LR down, and the arm ARC "helped" never did. Large steps were the only thing
# that could carry the weights out of the dead region; cutting the LR at the
# moment of the plateau removed them and locked the collapse in. Interventions
# turned a run that recovered to 73% into one that finished at chance.
#
# So the rule reports. Acting again needs a measured trajectory showing some
# action actually helps a plateaued run — the same bar every other rule here
# has to clear. See docs/EXPERIMENT_RESULTS.md.
#
# `representation_collapse` is here for the same reason, found by the sweep that
# confirmed the plateau fix. It was the last structural rule still allowed to
# act, and it reproduced the defect exactly. At lr=0.5 (seed 1234, 10 epochs):
#
#     epoch    baseline (no action)     active (3x rollback_and_reduce_lr)
#         3    10.00%  lr=4.04e-01      10.00%  lr=4.04e-01
#         4    19.35%  lr=3.34e-01      10.00%  lr=3.34e-01
#         5    28.32%  lr=2.56e-01      21.44%  lr=3.20e-02
#        10    75.18%                   30.84%
#
# -44.34pp. Same mechanism as the plateau: the control arm escaped the dead
# region once cosine decay lowered the learning rate by itself, and the arm ARC
# cut sat an order of magnitude below that and never caught up. Rolling back
# adds nothing either — the checkpoints in the ring are from the collapsed
# region, which is where the model already is.
#
# The consequence is worth stating plainly: no structural rule is allowed to act
# any more. Every one that was given the power either fired on healthy runs or
# damaged failing ones. What still acts is numerical divergence and gradient
# clipping, both of which are verified.
REPORT_ONLY_KINDS = frozenset({"loss_plateau", "representation_collapse"})

MAX_ATTEMPTS_PER_KIND = _env_int("ARC_MAX_ATTEMPTS", 3)
# Upper bound on simultaneously tracked optimizers. Enough for a GAN or an
# actor-critic; small enough that a sweep loop cannot accumulate checkpoints for
# every model it has ever built. See _evict_stale_monitors.
MAX_TRACKED_OPTIMIZERS = _env_int("ARC_MAX_TRACKED_OPTIMIZERS", 4)


# ─────────────────────────────────────────────────────────────────────────────
# Torch
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a hard requirement in practice
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    import arc as _arc_pkg
    HAS_ARC = True
except ImportError:
    _arc_pkg = None
    HAS_ARC = False


# ─────────────────────────────────────────────────────────────────────────────
# Run state
# ─────────────────────────────────────────────────────────────────────────────

class _RunState:
    def __init__(self) -> None:
        self.step = 0
        self.epoch = 0
        self.loss_history: deque[float] = deque(maxlen=50)
        self.intervention_count = 0
        self.risk = 0.0
        # loss recorded by backward()/scale(), consumed by the next step()
        self.pending_loss = None          # torch.Tensor, always UNSCALED
        self.pending_is_scaled = False
        self.active_scaler = None
        self.unscaled_optimizers: set[int] = set()
        self.backward_calls = 0
        self.instrument_seconds = 0.0
        self.train_start = time.time()
        self.clip_max_norm = None
        self.grad_clip_enabled = False
        self.last_intervention_step = -10**9
        self.attempts_by_kind: dict[str, int] = {}
        self.abandoned_kinds: set[str] = set()
        # Report-only kinds that have said their piece and stopped repeating.
        # Kept apart from `abandoned_kinds` because that set is what the run
        # summary publishes as "unrecoverable", and a report-only kind has no
        # standing to make that claim — ARC never acted, so it has no evidence
        # the run is beyond saving. The lr=0.5 control arm is the proof: it
        # reported a plateau, took no action, and recovered to 73.19%.
        self.silenced_kinds: set[str] = set()
        # Re-entrancy guard for nested optimizers; see the step wrapper.
        self.in_step = False


STATE = _RunState()


# ─────────────────────────────────────────────────────────────────────────────
# Gradient norm — one fused kernel, one device sync
# ─────────────────────────────────────────────────────────────────────────────

def _grad_norm_tensor(params):
    """Total L2 norm over `params`, as a 0-dim tensor (no sync performed here).

    The naive implementation calls ``.item()`` once per parameter tensor, which
    is a device synchronisation each time — 161 of them per step on ResNet-50,
    scaling with parameter *count* rather than size. ``torch._foreach_norm`` is
    the same fused kernel ``clip_grad_norm_`` itself uses: one launch, and the
    caller decides when (or whether) to sync.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return None
    if len(grads) == 1:
        return torch.linalg.vector_norm(grads[0].detach(), 2)
    norms = torch._foreach_norm(grads, 2.0)
    return torch.linalg.vector_norm(torch.stack(norms), 2)


def _params_of(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            yield p


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint store — CPU-resident, bounded, RNG-preserving
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointStore:
    """Rolling snapshots of model + optimizer state, held in CPU memory.

    Deep-copying ``state_dict()`` in place keeps every checkpoint on the GPU.
    With Adam that is roughly ``max_checkpoints x 3 x model size`` of VRAM
    (weights plus two moment buffers), which is how a tool that exists to warn
    about OOM ends up causing one. Copies land on the host instead; the transfer
    cost is paid once every ``CHECKPOINT_EVERY`` steps and restore is rare.

    RNG state travels with the checkpoint so that a rollback resumes the same
    data order and the same dropout masks — which is also what makes the
    baseline-vs-active A/B comparison a fair one.
    """

    def __init__(self, model, optimizer, max_checkpoints: int = 3) -> None:
        self.model = model
        self.optimizer = optimizer
        self.max_checkpoints = max(1, max_checkpoints)
        # The effective bound, tightened once bytes_per_checkpoint is known and
        # MAX_CHECKPOINT_MB turns out to be the stricter of the two caps. A
        # plain list rather than deque(maxlen=...) because that bound has to
        # move at runtime, and a deque's maxlen is fixed at construction.
        self.effective_max = self.max_checkpoints
        self.snapshots: list = []
        self.bytes_per_checkpoint = 0
        self.pruned_count = 0
        self._warned_size = False

    @staticmethod
    def _to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: CheckpointStore._to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(CheckpointStore._to_cpu(v) for v in obj)
        return obj

    def _rng_state(self) -> dict:
        state = {"torch": torch.get_rng_state(), "cuda": None, "numpy": None}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        try:
            import numpy as np
            state["numpy"] = np.random.get_state()
        except ImportError:
            pass
        return state

    def estimate_bytes(self) -> int:
        total = 0
        for p in self.model.parameters():
            total += p.numel() * p.element_size()
        for state in self.optimizer.state.values():
            if isinstance(state, dict):
                for v in state.values():
                    if torch.is_tensor(v):
                        total += v.numel() * v.element_size()
        return total

    def save(self, step: int) -> None:
        try:
            # Re-estimated rather than measured once.
            #
            # The first save happens at monitor registration, before the
            # optimizer has ever stepped, so `optimizer.state` is still empty and
            # the moment buffers are not counted. Reporting that figure and never
            # revisiting it understated the true cost by ~3x for Adam — the exact
            # "3 x model size" factor this class exists to avoid paying in VRAM.
            estimate = self.estimate_bytes()
            grew = estimate > self.bytes_per_checkpoint
            if grew:
                self.bytes_per_checkpoint = estimate
                self._recompute_effective_max()
            self.snapshots.append({
                "step": step,
                "time": time.time(),
                "model": self._to_cpu(self.model.state_dict()),
                "optimizer": self._to_cpu(self.optimizer.state_dict()),
                "rng": self._rng_state(),
            })
            if grew:
                self._report_budget()
            self._trim()
            self.failed = False
        except Exception as exc:
            # Say what is actually true. This used to report "Checkpointing
            # disabled" and then silently retry every subsequent save, leaving
            # the user believing checkpointing was off while a stale last-good
            # snapshot sat in the ring — and a rollback to it still reported
            # success.
            self.failed = True
            warn_once(
                "checkpoint_save",
                f"Checkpoint failed ({exc}). ARC will keep trying; if this persists, "
                "rollback will restore an increasingly stale snapshot.",
            )

    def release(self) -> None:
        """Drop retained state. Used when a monitor is evicted."""
        self.snapshots.clear()
        self.collector = None

    def _recompute_effective_max(self) -> None:
        """Tighten the ring-buffer bound to whichever cap is stricter.

        Called whenever bytes_per_checkpoint grows (it is only ever
        re-estimated upward — see the comment in save()). MAX_CHECKPOINTS is a
        count; MAX_CHECKPOINT_MB is a byte budget. The effective bound is
        whichever produces fewer snapshots, and it never drops below 1 — a
        single checkpoint beats none, even if that one checkpoint alone
        exceeds the configured MB budget.
        """
        if self.bytes_per_checkpoint <= 0:
            return
        budget_count = int((MAX_CHECKPOINT_MB * 1_000_000) // self.bytes_per_checkpoint)
        effective = max(1, min(self.max_checkpoints, budget_count))
        if effective == 1 and budget_count < 1:
            warn_once(
                "checkpoint_budget_tight",
                f"ARC_MAX_CHECKPOINT_MB={MAX_CHECKPOINT_MB:.0f} fits less than one "
                f"{self.bytes_per_checkpoint / 1e6:.1f} MB checkpoint — rollback "
                "capacity reduced to a single checkpoint.",
            )
        self.effective_max = effective

    def _trim(self) -> None:
        while len(self.snapshots) > self.effective_max:
            self.snapshots.pop(0)
            # Only count evictions the MB budget caused beyond what the
            # count-based MAX_CHECKPOINTS ring buffer would have done on its
            # own, so pruned_count reads as "what the byte budget cost you".
            if self.effective_max < self.max_checkpoints:
                self.pruned_count += 1

    def _report_budget(self) -> None:
        total = self.bytes_per_checkpoint * self.effective_max
        emit({
            "type": "checkpoint_budget",
            "bytes_per_checkpoint": self.bytes_per_checkpoint,
            "max_checkpoints": self.effective_max,
            "total_bytes": total,
            "budget_mb": MAX_CHECKPOINT_MB,
            "pruned_count": self.pruned_count,
            "location": "cpu",
        })
        emit_log(
            f"Checkpoint budget: {total / 1e6:.1f} MB host RAM "
            f"({self.effective_max} x {self.bytes_per_checkpoint / 1e6:.1f} MB, CPU-resident).",
        )

    def restore(self, index: int = -1) -> int:
        """Restore a snapshot. Returns how many steps were rewound."""
        if not self.snapshots:
            return 0
        cp = self.snapshots[index]
        self.model.load_state_dict(cp["model"])
        self.optimizer.load_state_dict(cp["optimizer"])
        # load_state_dict restores the checkpointed LR too; the caller scales it
        # afterwards, which is what stops a rollback resurrecting the LR that
        # caused the divergence in the first place.
        rng = cp.get("rng") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            try:
                import numpy as np
                np.random.set_state(rng["numpy"])
            except ImportError:
                pass
        return max(0, STATE.step - cp["step"])


# ─────────────────────────────────────────────────────────────────────────────
# Per-optimizer monitor
# ─────────────────────────────────────────────────────────────────────────────

class OptimizerMonitor:
    """Everything ARC tracks for one optimizer.

    Kept per-optimizer rather than globally so a GAN's generator and
    discriminator each get their own checkpoints, their own gradient history and
    their own recovery, instead of one overwriting the other.
    """

    def __init__(self, optimizer, model, name: str) -> None:
        self.optimizer = optimizer
        self.model = model
        self.name = name
        self.store = CheckpointStore(model, optimizer, MAX_CHECKPOINTS)
        self.store.failed = False
        self.collector = None
        self.collector_ready = False
        self.local_step = 0
        # Cumulative learning-rate factor ARC has applied, and the LR values ARC
        # last wrote. See enforce_lr for why both are needed.
        self.lr_factor = 1.0
        self.last_lr: list[float] | None = None
        self.baseline: dict[str, float] = {}
        self.baseline_samples = 0
        self.structural_strikes = 0
        self.structural_kind: str | None = None
        # Loss-plateau state. Tracked per optimizer, not globally, so a GAN's
        # two optimizers plateau independently.
        self.best_loss = float("inf")
        self.opening_losses: list[float] = []
        self.steps_without_improvement = 0

    # -- silent death ---------------------------------------------------------
    def check_plateau(self, loss_val: float) -> tuple[str, str] | None:
        """Detect a run that has stopped learning while its loss stays finite.

        This is the failure mode the structural tier was built for and did not
        catch. A real run at lr=0.5 finished at 10.00% validation accuracy —
        chance — with its loss pinned at ln(10) for 750+ consecutive steps. It
        produced no NaN, no gradient spike, and no rank collapse, so every rule
        ARC had was silent for all 780 steps.

        The counter is the whole rule: how long since the loss last set a new
        best. It needs no collector, works without ``arc-training`` installed,
        and is the only signal measured on this workload that separates the two
        arms by more than a factor of two. See LOSS_PLATEAU_PATIENCE.

        Deliberately keyed on the *best-ever* batch loss rather than a moving
        average. A dead run's mean is stable, so a mean-based rule has to pick a
        tolerance; a best-ever counter needs no tolerance at all, because a run
        that is still learning keeps producing new minima and a dead one never
        produces another.
        """
        # An unobservable loss arrives here as NaN — torch.autograd.backward(),
        # a non-scalar backward, LBFGS. Counting that as "no improvement" would
        # trip the rule on a run that is training perfectly well and simply
        # cannot be watched. Absent is not stalled.
        if math.isnan(loss_val) or math.isinf(loss_val):
            return None

        # The reference for "did this run ever get anywhere", captured from the
        # opening steps and then frozen. A median rather than the single first
        # loss: one batch is a sample of size one, and an unlucky high first
        # batch would make a dead run look like it had improved.
        if len(self.opening_losses) < LOSS_PLATEAU_OPENING_SAMPLES:
            self.opening_losses.append(loss_val)

        if loss_val < self.best_loss - LOSS_PLATEAU_MIN_DELTA:
            self.best_loss = loss_val
            self.steps_without_improvement = 0
            return None

        self.steps_without_improvement += 1
        if self.steps_without_improvement < LOSS_PLATEAU_PATIENCE:
            return None

        stalled = self.steps_without_improvement
        # Reset before returning, on every path. Without this the rule re-fires
        # on every subsequent step and the agent walks the learning rate to zero.
        self.steps_without_improvement = 0

        # A plateau on its own is not a failure — it is what convergence looks
        # like. This guard is the whole difference between the two.
        #
        # The rule shipped without it and a 10-epoch A/B caught that immediately:
        # a run that reached 87.5% validation accuracy tripped it twice. The
        # counter keys off the *best-ever* batch loss, so as a run converges its
        # own record gets harder to beat and the stalls grow without bound. On a
        # 780-step run the healthy maximum was 82 steps; over 3900 steps a
        # perfectly healthy run blows through any fixed patience. Raising the
        # patience cannot fix that — the stall length is unbounded on a
        # successful run, so there is no value that separates them.
        #
        # What separates them is whether the run ever got anywhere. Measured on
        # the pair of runs this rule was built from:
        #
        #     arm                  first loss   best loss   best/first
        #     lr=0.03  healthy        2.3018      0.6233       0.271
        #     lr=0.50  dead           2.3221      2.0632       0.888
        #
        # A dead run stalls *having never improved*; a converged run stalls
        # having improved enormously. That ratio is architecture- and
        # task-independent in a way "loss near ln(num_classes)" is not — it needs
        # no knowledge of the class count, or even that this is classification.
        #
        # Known limitation, stated rather than discovered: this assumes a run has
        # substantial headroom from its opening loss. Fine-tuning a pretrained
        # model does not — it can legitimately start at a low loss and converge
        # after a 10% improvement, which reads as "never got anywhere" here and
        # would trip the rule on a perfectly good run. Training from scratch is
        # the case this is calibrated for. If ARC starts being used on
        # fine-tuning workloads, this needs a different reference than the first
        # observed loss, and it needs measuring on that workload rather than
        # reasoning about it — which is the whole lesson of the two deleted rules
        # and of this rule's own first version.
        # ponytail: first-loss reference, revisit if fine-tuning runs appear.
        opening = statistics.median(self.opening_losses) if self.opening_losses else 0.0

        # A *negative* opening reference is not degenerate — it is ordinary for a
        # WGAN critic, a continuous-distribution NLL, or Dice-minus-one. But the
        # `best / opening < RATIO` test inverts under a sign flip, so the progress
        # guard this rule depends on silently stops working, and a run that
        # improved from -0.5 to -5.0 (10x) and then sat at its best for the
        # patience window — routine on a converged run, per the comments above —
        # was reported as a silent death. That is exactly the false positive
        # LOSS_PLATEAU_PROGRESS_RATIO exists to prevent, and the failure mode both
        # deleted structural rules were deleted for. The rule cannot distinguish
        # the two cases here, so it says nothing: every other guard in this file
        # fails closed.
        if opening < 0:
            return None

        # An opening of exactly 0.0 is a different case, and firing is deliberate.
        # A loss pinned at exactly zero from the opening samples onward has not
        # converged to zero, it has never moved — normally a broken objective. The
        # ratio is undefined rather than inverted, so the message below omits the
        # progress figure rather than dividing by zero, which would throw inside
        # the path that runs while a failure is being handled.
        if opening == 0:
            return (
                "loss_plateau",
                f"loss has failed to improve on its best value of {self.best_loss:.4f} "
                f"for {stalled} consecutive steps",
            )

        if self.best_loss / opening < LOSS_PLATEAU_PROGRESS_RATIO:
            # This run has made real progress. Stalling now is convergence, and
            # cutting the learning rate of a converged run is exactly the
            # unhelpful intervention the deleted rules were deleted for.
            return None

        return (
            "loss_plateau",
            f"loss has failed to improve on its best value of {self.best_loss:.4f} "
            f"for {stalled} consecutive steps, having improved only "
            f"{(1 - self.best_loss / opening) * 100:.1f}% from its opening loss",
        )

    # -- learning rate --------------------------------------------------------
    def enforce_lr(self) -> None:
        """Re-apply ARC's learning-rate reduction on top of whatever the user's
        scheduler just wrote.

        A scheduler recomputes ``group['lr']`` from its own base every step. So
        an intervention that only divides ``group['lr']`` is erased on the very
        next iteration and the run walks straight back into the divergence it
        was just rescued from — silently, because the intervention was logged as
        successful. Most real runs use a scheduler, so this is the common case,
        not the corner case.

        Rather than trying to discover and rewrite every scheduler's internal
        base (they differ, and third-party ones are unbounded), ARC treats its
        reduction as a multiplier it re-asserts each step. A group whose LR still
        holds the value ARC last wrote was not touched from outside and is left
        alone, so a run with no scheduler never compounds.
        """
        if self.lr_factor == 1.0:
            self.last_lr = [g["lr"] for g in self.optimizer.param_groups]
            return
        groups = self.optimizer.param_groups
        if self.last_lr is not None and len(self.last_lr) == len(groups):
            for i, group in enumerate(groups):
                if group["lr"] != self.last_lr[i]:
                    group["lr"] *= self.lr_factor
        self.last_lr = [g["lr"] for g in groups]

    def scale_lr(self, factor: float) -> tuple[float, float]:
        """Scale the live LR, the stored checkpoints, and all future steps."""
        groups = self.optimizer.param_groups
        old_lr = groups[0]["lr"] if groups else 0.0
        for group in groups:
            group["lr"] *= factor
            # Keep the standard PyTorch scheduler anchor in step as well, so a
            # scheduler that reads initial_lr starts from the corrected value.
            if "initial_lr" in group:
                group["initial_lr"] *= factor
        self.lr_factor *= factor
        self.last_lr = [g["lr"] for g in groups]

        # Scaling only the optimizer would let the next rollback restore the
        # exact learning rate that caused the divergence.
        for snapshot in self.store.snapshots:
            opt_state = snapshot.get("optimizer") or {}
            for group in opt_state.get("param_groups", []) or []:
                if "lr" in group:
                    group["lr"] *= factor
                if "initial_lr" in group:
                    group["initial_lr"] *= factor
        return old_lr, (groups[0]["lr"] if groups else 0.0)

    # -- arc signal collectors ------------------------------------------------
    def attach_collector(self) -> None:
        if not HAS_ARC or self.collector_ready:
            return
        self.collector_ready = True
        try:
            from arc.signals import CompositeCollector, GradientCollector, WeightCollector
            self.collector = CompositeCollector([GradientCollector(), WeightCollector()])
            self.collector.attach(self.model, self.optimizer)
            emit_log(f"ARC signal collectors attached ({self.name}).")
        except Exception as exc:
            self.collector = None
            warn_once(
                "collector_attach",
                f"Advanced diagnostics unavailable — collector attach failed: {exc}",
            )

    def collect_advanced(self) -> dict | None:
        if self.collector is None:
            return None
        try:
            self.collector.step()
            signals = self.collector.collect().signals
            grad_g = signals.get("gradient.global", {}) or {}
            weight_g = signals.get("weight.global", {}) or {}
            advanced: dict = {}
            mapping = (
                ("grad_norm_l2", grad_g, "total_grad_norm_l2"),
                ("gradient_entropy", grad_g, "total_grad_entropy"),
                ("grad_flow_ratio", grad_g, "grad_flow_ratio"),
                ("weight_norm", weight_g, "total_weight_norm"),
                ("effective_rank", weight_g, "mean_effective_rank"),
                ("weight_update_ratio", weight_g, "mean_update_ratio"),
            )
            diverged: list[str] = []
            for out_key, source, key in mapping:
                if key not in source:
                    continue
                value = _finite(source[key])
                if value is not None:
                    advanced[out_key] = value
                else:
                    # The value exists and is not representable — inf or NaN.
                    #
                    # Refusing to put it in the JSON is right; silently dropping
                    # the *fact* that it went non-finite is not. For
                    # `grad_flow_ratio` the divergence is the diagnosis: the
                    # upstream ratio is late-layer gradient norm over early, and
                    # it returns inf exactly when the early layers stop receiving
                    # gradient at all. Measured on a real pair of runs, the
                    # healthy arm stayed inside 1.36-3.10 for every sample while
                    # the dead arm hit 50.11 at step 50 and went non-finite from
                    # step 75 onward — so the chart gap was the loudest signal in
                    # the run, and it was being thrown away as unserialisable.
                    #
                    # Reported as a flag rather than a sentinel number, because a
                    # stand-in value would be indistinguishable from a real
                    # measurement on the chart.
                    diverged.append(out_key)
            if diverged:
                advanced["nonfinite"] = diverged
            return advanced or None
        except Exception as exc:
            warn_once("collector_step", f"Advanced diagnostics degraded: {exc}")
            self.collector = None
            return None

    # -- silent-failure detection --------------------------------------------
    #
    # There was a `loss_is_improving()` helper here, used to qualify the
    # update-ratio rule. Both are gone. It tried to read a real improvement of
    # roughly 1% per 60 steps out of a per-batch loss whose noise is around 40%
    # of its own mean — averaging 20 samples still leaves ~14% noise, so the
    # verdict was close to a coin flip whichever window it used. Two A/B runs
    # measured the cost of those flips at 1.74 and 0.78 points of validation
    # accuracy on runs that needed no intervention.
    #
    # The per-monitor `recent_losses` deque went with it. It was written and
    # cleared on every step but never read by anything — the `recent_losses`
    # the agent shows in its snapshot comes from `STATE.loss_history`, which is
    # a different collection. A comment here claimed otherwise; it was wrong.

    def check_structural(self, advanced: dict | None) -> tuple[str, str] | None:
        """Detect pathologies that never produce a NaN.

        A run can be completely dead while its loss stays finite and its
        gradient norm stays small — the loss just sits at ``ln(num_classes)``
        forever, and nothing in the numerical-failure path sees it. Catching that
        is what these signals are collected for.

        **One rule survives here, and the history is the point.** Three were
        written; two were deleted after an A/B measured each one intervening on
        healthy runs, at a cost of 1.74 and 0.78 accuracy points in one case and
        an entire run — 87.4% to chance — in the other. Both are documented in
        full below, because the failure mode they share is the thing to know
        before adding a third: every one of these signals moves by orders of
        magnitude in a run's opening steps purely because the model goes from
        random to structured, so a rule written against that transient fires on
        healthy training.

        Three guards follow from that, and a new rule needs all of them:

        * nothing is judged until ``STRUCTURAL_WARMUP_STEPS`` have passed, so the
          baseline is measured on a settled run rather than during the transient;
        * thresholds are fractions of the run's *own* baseline, because the
          absolute scale of these quantities is entirely architecture-dependent —
          a value tuned on a small MLP means nothing on a CNN 256 units wide;
        * a rule must hold for several consecutive samples. One bad sample is
          noise; a sustained one is a diagnosis.

        And a rule needs evidence: a measured trajectory showing the statistic
        actually separates a healthy run from a failing one. Both deleted rules
        looked entirely reasonable as arguments. Neither survived measurement.
        """
        if not advanced:
            return None

        rank = advanced.get("effective_rank")

        # The baseline is captured after a warmup, not from step 1.
        #
        # Every one of these signals moves by orders of magnitude during the
        # first few dozen steps, on healthy and failing runs alike, simply
        # because the model goes from random to structured. A baseline taken
        # inside that transient makes normal early learning look like collapse —
        # which is exactly how the entropy rule came to kill a healthy run at
        # step 125.
        # Capture the baseline from the run's *opening* samples, but do not
        # judge against it until the warmup has passed.
        #
        # These were one step before, and collapsing them was a real bug. The
        # baseline was taken from the first samples after step 200, which means
        # a run that died before step 200 had its "healthy" reference measured
        # on the corpse. Ratios are then computed dead-against-dead and sit at
        # ~1.0 forever. Measured on a real pair of runs, that made the dead arm
        # score *better* than the healthy one:
        #
        #     arm                baseline   floor    floor / baseline
        #     lr=0.03  73.6%      28.312    27.940        98.69%
        #     lr=0.50  10.0%      25.207    25.136        99.72%   <- dead
        #
        # The dead run collapsed by step 45 and flatlined by step 100, so its
        # post-warmup baseline was simply its collapsed value. Taking the
        # baseline early fixes that: against a step-1 reference the dead arm
        # reads 87.4% and the healthy arm 97.2%.
        #
        # Judging still waits for the warmup, because that guard is what stops a
        # rule firing on the random-to-structured transient. The two concerns
        # were conflated; they are separate and only one of them needs to wait.
        if rank is not None and rank > 0 and self.baseline_samples < STRUCTURAL_BASELINE_SAMPLES:
            self.baseline["effective_rank"] = max(self.baseline.get("effective_rank", 0.0), rank)
            self.baseline_samples += 1
        if STATE.step < STRUCTURAL_WARMUP_STEPS:
            return None

        triggered: tuple[str, str] | None = None

        # There is deliberately no rule on `weight_update_ratio`.
        #
        # One existed, firing above an absolute ceiling of 0.05. It was removed
        # after measuring the statistic across four learning rates on this
        # workload (3 epochs each, sampled every 10 steps):
        #
        #     lr     median     p90      max   longest run >0.05
        #     0.03   0.0240   0.0894   0.2847   31    <- healthy, 87% accuracy
        #     0.10   0.0374   0.0861   0.2943   37    <- healthy, 88% accuracy
        #     0.25   0.0456   0.0882   0.3224   26    <- damaged, 76% accuracy
        #     0.50   0.0629   0.1008   0.3997   61    <- dead, 10% accuracy
        #
        # The distributions overlap almost completely. A perfectly healthy run
        # peaks higher than the damaged one and sustains a *longer* consecutive
        # breach, so neither a magnitude threshold nor a persistence requirement
        # can separate them. The rule was a proxy for "the learning rate is
        # large", not for "training is failing" — which is why cutting the LR on
        # it helped when the rate was genuinely too high and hurt when it wasn't.
        #
        # It was not a theoretical concern: in back-to-back A/B runs it cost
        # 1.74 and 0.78 points of validation accuracy on runs that needed no
        # help at all. The ratio is still collected and charted, because it is
        # informative to a human reading the run. It just does not get to act.
        #
        # The loss-trend guard that once qualified this rule went with it: a
        # per-batch loss on this task is noisy at ~40% of its mean against a real
        # improvement of ~1% per 60 steps, so no window recovers the signal, and
        # a guard that is wrong a third of the time is worse than no rule.

        base_rank = self.baseline.get("effective_rank")
        if (
            triggered is None
            and base_rank
            and rank is not None
            and rank < base_rank * RANK_COLLAPSE_FRACTION
        ):
            # Ungated, because the threshold is conservative by a wide margin.
            #
            # Know what that margin costs. The original note here read "nothing
            # plausible reaches 50% except a real collapse" — checked only in
            # the false-positive direction. Measured in the other direction, a
            # genuinely dead network does not reach it either: a CIFAR run that
            # finished at chance accuracy bottomed at 87.4% of its step-1 rank,
            # against 97.2% for a healthy one. Real separation, roughly ten
            # points of it, and a trigger sitting four times further away.
            #
            # That is not a threshold to tune. `effective_rank` is the SVD
            # entropy of the *weight* matrices, and a network can emit a nearly
            # constant output while every weight matrix stays well-conditioned —
            # which is exactly what happened, weight norm halving from 147.9 to
            # 71.4 while rank held. Weight conditioning is not representational
            # rank; they diverge precisely in this failure mode.
            #
            # So this rule is left conservative and `loss_plateau` covers the
            # silent-death case instead, on a signal that separates the same two
            # runs by 9.3x rather than 1.1x. Moving this threshold up to catch
            # what plateau already catches would buy nothing and reintroduce the
            # knife-edge that got two previous rules deleted.
            triggered = (
                "representation_collapse",
                f"mean effective rank {rank:.2f} has fallen below "
                f"{RANK_COLLAPSE_FRACTION:.0%} of its baseline {base_rank:.2f}",
            )

        # There is deliberately no rule on `gradient_entropy` either, and this
        # one was removed only after it did real damage.
        #
        # It fired below 1% of an opening baseline. On a healthy CIFAR run at
        # lr=0.25 it fired at step 125 and three rollbacks took the model from
        # 87.4% to 10% — chance — in the first epoch, while the control arm went
        # on to 88.5%. ARC destroyed the run it was protecting and then correctly
        # reported it unrecoverable.
        #
        # Measuring the trajectory afterwards showed no threshold could have
        # saved the rule. Sampled every 10 steps over two epochs:
        #
        #     step   lr=0.25 (healthy, 87.4%)   lr=0.50 (dead, 10%)
        #        1              2.95e-01              2.95e-01
        #       30              1.09e-02              1.26e-01
        #       70              1.44e-05              1.45e-05
        #      200+             1.44e-05              1.29e-05
        #
        # The healthy run and the dead run settle to *the same value*. The signal
        # carries no information about run health after roughly step 70.
        #
        # The reason is upstream: `GradientCollector._compute_entropy` bins a
        # heavy-tailed gradient distribution with `torch.histc` on a linear
        # scale. A few outliers set the range, essentially all mass lands in one
        # bin, and the normalised entropy saturates near zero — for any run. It
        # is measuring outlier spread, not information content. Making it
        # informative needs log-magnitude binning and a fresh validation that the
        # result actually separates the cases; until someone does that, this is a
        # chart line, not a trigger.

        if triggered is None:
            self.structural_strikes = 0
            self.structural_kind = None
            return None

        kind, _ = triggered
        if kind != self.structural_kind:
            self.structural_kind = kind
            self.structural_strikes = 0
        self.structural_strikes += 1
        if self.structural_strikes < STRUCTURAL_SUSTAIN:
            return None
        self.structural_strikes = 0
        return triggered


def _finite(value):
    """JSON has no NaN or Infinity. Absent beats fabricated, so unrepresentable
    values become None and render as a gap in the chart."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


# Insertion-ordered so the least recently used entry is the eviction victim.
MONITORS: "OrderedDict[int, OptimizerMonitor]" = OrderedDict()


# ─────────────────────────────────────────────────────────────────────────────
# Model resolution
# ─────────────────────────────────────────────────────────────────────────────

_EXPLICIT_MODELS: list = []


def watch(model, optimizer=None):
    """Explicit escape hatch: ``model, optimizer = arc_watch(model, optimizer)``.

    Automatic resolution matches an optimizer to its model by parameter
    identity, which is correct for every setup encountered in practice. This
    exists for the ones it is not — a model whose parameters are assembled from
    several modules, or an optimizer constructed before its model is reachable.
    """
    if model is not None and model not in _EXPLICIT_MODELS:
        _EXPLICIT_MODELS.append(model)
    if optimizer is not None:
        _resolve_monitor(optimizer)
    return (model, optimizer) if optimizer is not None else model


def _find_owning_model(optimizer):
    """Find the module that actually owns this optimizer's parameters.

    The frame walk on its own is what misattributes GANs: it takes the *first*
    ``nn.Module`` in ``locals()`` and hopes. Here the walk only produces
    candidates — the winner is decided by parameter identity, so the
    discriminator's optimizer cannot be matched to the generator.
    """
    target = {id(p) for p in _params_of(optimizer)}
    if not target:
        return None

    def score(module) -> float:
        owned = {id(p) for p in module.parameters()}
        if not owned:
            return 0.0
        return len(target & owned) / len(target)

    roots: list = list(_EXPLICIT_MODELS)

    import inspect
    frame = inspect.currentframe()
    try:
        depth = 0
        while frame is not None and depth < 12:
            for value in list(frame.f_locals.values()):
                if isinstance(value, nn.Module) and not any(value is r for r in roots):
                    roots.append(value)
            frame = frame.f_back
            depth += 1
    finally:
        del frame

    # Expand each root into itself plus its submodules.
    #
    # Only *locals* reach the frame walk, so a model held as an attribute —
    # `self.discriminator`, or `pair.disc` — is invisible to it, and the
    # enclosing container would win instead by virtue of owning every parameter.
    # Rolling that container back would discard the other network's progress,
    # which is the same misattribution bug in a different disguise.
    candidates: list = []
    for root in roots:
        for module in root.modules():
            if not any(module is c for c in candidates):
                candidates.append(module)

    best, best_score, best_size = None, 0.0, 0
    for candidate in candidates:
        s = score(candidate)
        if s <= 0.0:
            continue
        size = sum(p.numel() for p in candidate.parameters())
        # Prefer the highest coverage; among equally complete owners prefer the
        # smallest, so the specific submodule this optimizer updates beats any
        # wrapper that merely contains it.
        if s > best_score or (s == best_score and best is not None and size < best_size):
            best, best_score, best_size = candidate, s, size

    return best if best_score > 0.0 else None


def _evict_stale_monitors() -> None:
    """Bound the monitor table.

    `MONITORS` is keyed by `id(optimizer)` and held strongly, so nothing ever
    left it. A cross-validation or hyperparameter sweep — `for fold in ...:
    model = Net(); opt = Adam(...)` — accumulates one monitor per iteration, each
    pinning a model, an optimizer and up to `MAX_CHECKPOINTS` host-resident state
    dicts. Five folds of a 1 GB model retained roughly 15 GB of host RAM that
    could never be reclaimed, from the class whose entire purpose is to keep
    checkpointing from exhausting memory.

    Keeping the most recent few is the right trade: a sweep only ever trains one
    model at a time, while a GAN or actor-critic needs a handful live at once.
    """
    while len(MONITORS) > MAX_TRACKED_OPTIMIZERS:
        _key, victim = next(iter(MONITORS.items()))
        MONITORS.pop(_key, None)
        try:
            # `release()` lives on CheckpointStore, not on the monitor — MONITORS
            # holds monitors, so `victim.release()` raised AttributeError every
            # time and the bare except swallowed it. The cleanup this eviction
            # exists to perform, including detaching the arc collector's hooks
            # from the evicted model, had therefore never once run.
            victim.store.release()
        except Exception:
            pass


def _resolve_monitor(optimizer):
    key = id(optimizer)
    monitor = MONITORS.get(key)
    if monitor is not None:
        # Refresh recency so the active optimizer is never the eviction victim.
        MONITORS.move_to_end(key)
        return monitor
    model = _find_owning_model(optimizer)
    if model is None:
        warn_once(
            "model_unresolved",
            "Could not match an nn.Module to this optimizer — checkpointing and "
            "rollback are disabled. Add `model, optimizer = arc_watch(model, optimizer)` "
            "after constructing them to resolve this explicitly.",
        )
        return None
    name = type(model).__name__ + "/" + type(optimizer).__name__
    monitor = OptimizerMonitor(optimizer, model, name)
    MONITORS[key] = monitor
    _evict_stale_monitors()
    monitor.attach_collector()
    monitor.store.save(STATE.step)
    emit_log(f"ARC monitoring {name} ({sum(p.numel() for p in model.parameters()):,} params).")
    return monitor


# ─────────────────────────────────────────────────────────────────────────────
# Risk heuristic
# ─────────────────────────────────────────────────────────────────────────────

def _risk_score(losses, grad_norm: float, is_bad: bool) -> tuple[float, str]:
    if is_bad:
        return 1.0, "CRITICAL"
    risk = 0.0
    if len(losses) >= 5 and losses[0] > 0 and losses[-1] > losses[0] * 2.0:
        risk += 0.4
    if grad_norm > 10.0:
        risk += 0.4
    if grad_norm > 50.0:
        risk += 0.2
    risk = min(risk, 1.0)
    label = "CRITICAL" if risk > 0.8 else "HIGH" if risk > 0.5 else "MEDIUM" if risk > 0.25 else "LOW"
    return risk, label


def _should_sample_advanced() -> bool:
    """Sample expensive signals densely while unstable, sparsely while calm.

    "Densely" is deliberately not "every step". Sampling every step costs 170%
    wall-clock against 8.4% at the normal rate (see benchmark_overhead.py), so a
    run that sits at elevated risk for a long stretch — exactly what an unstable
    run does — would be paying the worst possible rate for the longest time. The
    dense rate is a fixed multiple of the normal one instead, which is enough to
    resolve a developing pathology without tripling the cost of the run it is
    trying to save.
    """
    if ADVANCED_EVERY <= 1 or STATE.step == 1:
        return True
    if STATE.risk >= 0.4:
        return STATE.step % DENSE_ADVANCED_EVERY == 0
    return STATE.step % ADVANCED_EVERY == 0


def _should_emit_metric() -> bool:
    """Coalesce the per-step `metric` event while healthy, densify under risk.

    Mirrors `_should_sample_advanced` exactly. `METRIC_EVERY` defaults to 1,
    which reproduces the previous behaviour of emitting every step. This gates
    only the emit call — loss history and the risk score are computed every
    step regardless, so risk detection is never delayed by coalescing.
    """
    if METRIC_EVERY <= 1 or STATE.step == 1:
        return True
    if STATE.risk >= 0.4:
        return STATE.step % DENSE_METRIC_EVERY == 0
    return STATE.step % METRIC_EVERY == 0


# ─────────────────────────────────────────────────────────────────────────────
# The measurement point
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_mem_mb() -> float:
    if HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1048576.0
    return 0.0


def on_optimizer_step(optimizer) -> None:
    """Called once per weight update, before the update is applied."""
    started = time.perf_counter()
    try:
        _on_optimizer_step_inner(optimizer)
    except Exception as exc:
        warn_once("step_hook", f"ARC step hook degraded: {exc}")
    finally:
        STATE.instrument_seconds += time.perf_counter() - started


def _on_optimizer_step_inner(optimizer) -> None:
    monitor = _resolve_monitor(optimizer)
    if monitor is not None:
        # Runs before the LR is read and before the update is applied, so an
        # earlier intervention survives the user's scheduler.
        monitor.enforce_lr()
    STATE.step += 1
    step = STATE.step

    params = list(_params_of(optimizer))
    norm_tensor = _grad_norm_tensor(params)

    loss_tensor = STATE.pending_loss
    STATE.pending_loss = None
    # An *absent* loss is not a bad loss.
    #
    # `pending_loss` is only set by the patched `Tensor.backward` and
    # `GradScaler.scale`. It is legitimately None whenever the user reaches the
    # backward pass another way — `torch.autograd.backward(loss)` (the function
    # form, which is not a Tensor method), a non-scalar backward, or a
    # closure-driven optimizer like LBFGS that runs backward *inside* step().
    #
    # Treating that as NaN made ARC diagnose a failure on every step of a
    # perfectly healthy run, roll back, cut the learning rate, and zero the
    # gradients before the real update — freezing the model completely while
    # the log filled with successful-looking interventions.
    loss_known = loss_tensor is not None

    # One synchronisation for both scalars instead of one each.
    if loss_known and norm_tensor is not None:
        both = torch.stack([loss_tensor.detach().reshape(()).float(), norm_tensor.float()])
        loss_val, grad_norm = (float(x) for x in both.tolist())
    else:
        loss_val = float(loss_tensor.detach().item()) if loss_known else float("nan")
        grad_norm = float(norm_tensor.item()) if norm_tensor is not None else 0.0

    if not loss_known:
        warn_once(
            "loss_unavailable",
            "Could not observe the loss for this optimizer — ARC is reporting gradient "
            "and learning-rate telemetry only, and will not act on loss-based failures. "
            "This happens with torch.autograd.backward(), a non-scalar backward, or a "
            "closure-based optimizer such as LBFGS.",
        )

    # Under AMP the gradients are scaled unless the scaler already unscaled them
    # for this optimizer. Reporting a norm inflated by ~65536x would make every
    # gradient number meaningless on exactly the setups this tool targets.
    scaler = STATE.active_scaler
    if scaler is not None and id(optimizer) not in STATE.unscaled_optimizers:
        try:
            scale = float(scaler.get_scale())
            if scale > 0:
                grad_norm /= scale
        except Exception:
            warn_once("amp_scale", "Could not read GradScaler scale; gradient norms may be scaled.")

    lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
    is_bad = loss_known and (
        math.isnan(loss_val) or math.isinf(loss_val) or abs(loss_val) > LOSS_EXPLOSION_THRESHOLD
    )

    if not is_bad:
        STATE.loss_history.append(loss_val)

    advanced = None
    if monitor is not None:
        if _should_sample_advanced():
            advanced = monitor.collect_advanced()
        if not is_bad:
            monitor.local_step += 1
            if monitor.local_step % max(1, CHECKPOINT_EVERY) == 0:
                monitor.store.save(step)

    risk, label = _risk_score(list(STATE.loss_history)[-5:], grad_norm, is_bad)
    STATE.risk = risk
    emit({"type": "risk", "score": round(risk, 4), "label": label})

    if _should_emit_metric():
        emit({
            "type": "metric",
            "step": step,
            "epoch": STATE.epoch,
            "loss": _finite(loss_val),
            "grad_norm": round(grad_norm, 6),
            "lr": lr,
            "gpu_mem_mb": round(_gpu_mem_mb(), 1),
            "optimizer": monitor.name if monitor else type(optimizer).__name__,
            "timestamp": time.time(),
            "advanced": advanced,
        })

    # Clipping runs after the metric is emitted, so the reported gradient norm is
    # the true pre-clip value — clipping is an intervention, and hiding its effect
    # behind a clamped number would make the chart stop showing why it fired.
    #
    # Skipped while gradients are still scaled: clip_grad_norm_ compares against
    # an absolute threshold, so applying max_norm=1.0 to gradients inflated by
    # ~65536x would scale every update to nothing. Under scaler.step() the
    # gradients are already unscaled by the time this runs, which is the normal
    # AMP path; the guard covers a raw optimizer.step() under a live scaler.
    grads_are_scaled = (
        STATE.active_scaler is not None and id(optimizer) not in STATE.unscaled_optimizers
    )
    if STATE.grad_clip_enabled and STATE.clip_max_norm and params and not grads_are_scaled:
        try:
            torch.nn.utils.clip_grad_norm_(params, STATE.clip_max_norm)
        except Exception as exc:
            warn_once("grad_clip", f"Gradient clipping failed: {exc}")

    if is_bad:
        _handle_failure(optimizer, monitor, step, loss_val, grad_norm, lr, advanced,
                        kind="numerical", reason="loss is non-finite or exploded")
    elif monitor is not None:
        # Checked every step, not on the advanced-sampling cadence: the plateau
        # counter needs every loss to be correct, and it costs two float
        # comparisons. It also runs without `arc-training` installed, which the
        # collector-based checks below cannot.
        structural = monitor.check_plateau(loss_val) if loss_known else None
        if structural is None and advanced:
            structural = monitor.check_structural(advanced)
        if (structural is not None
                and structural[0] not in STATE.abandoned_kinds
                and structural[0] not in STATE.silenced_kinds):
            # Once a kind has been declared unrecoverable it has been reported
            # in full; re-announcing the same diagnosis every sample would bury
            # the log without adding information.
            kind, reason = structural
            _handle_failure(optimizer, monitor, step, loss_val, grad_norm, lr, advanced,
                            kind=kind, reason=reason)

    if STEP_DELAY > 0:
        time.sleep(STEP_DELAY)


def _handle_failure(optimizer, monitor, step, loss_val, grad_norm, lr, advanced,
                    kind: str = "numerical", reason: str = "") -> None:
    emit({
        "type": "failure_detected",
        "step": step,
        "kind": kind,
        "reason": reason,
        "loss": "NaN/Inf" if (math.isnan(loss_val) or math.isinf(loss_val)) else _finite(loss_val),
        "grad_norm": round(grad_norm, 6),
    })

    # A rollback needs a few steps to show whether it worked. Without a cooldown
    # the very next step re-detects the same failure, rolls back to the same
    # checkpoint and halves the LR again — the run collapses into an intervention
    # loop that looks busy in the log while making no progress.
    # Futility circuit-breaker.
    #
    # Some runs are past saving: once a network has genuinely collapsed, every
    # checkpoint still in the ring buffer is collapsed too, so rolling back and
    # halving the learning rate again just produces a slower dead run. Repeating
    # it eleven times fills the log with successful-looking interventions while
    # the model sits at chance accuracy.
    #
    # Admitting the run is unrecoverable is more useful than pretending
    # otherwise: the user's real decision at that point is whether to keep
    # paying for the GPU, and telling them to stop at step 70 is worth far more
    # than a twelfth rollback.
    # The three early exits below all decline to act. None of them may touch the
    # gradients on the way out.
    #
    # They used to call `optimizer.zero_grad()` before returning, which silently
    # discarded the user's update. That is an intervention — and in the case of a
    # diverging run, the single most consequential one available, since dropping
    # every bad update is most of what a rollback achieves. It made the log's
    # claims false in three separate ways: "left as-is" froze the model instead,
    # "letting the previous recovery take effect" suppressed up to 15 healthy
    # updates whose gradients were never bad, and "interventions suppressed"
    # quietly protected the control arm of the very experiment built to measure
    # what interventions are worth.
    attempts = STATE.attempts_by_kind.get(kind, 0)
    if attempts >= MAX_ATTEMPTS_PER_KIND:
        report_only = kind in REPORT_ONLY_KINDS
        seen = STATE.silenced_kinds if report_only else STATE.abandoned_kinds
        if kind not in seen:
            seen.add(kind)
            last_snapshot = monitor.store.snapshots[-1] if (monitor is not None and monitor.store.snapshots) else None
            if last_snapshot is not None:
                elapsed_seconds = time.time() - last_snapshot["time"]
                last_checkpoint_step = last_snapshot["step"]
            else:
                # No checkpoint exists yet (e.g. the model was never resolved).
                # This is time since the run *started*, not since the last
                # checkpoint — there isn't one — so it understates or overstates
                # the true waste depending on how CHECKPOINT_EVERY compares to
                # how long the run has been going. Best available fallback.
                elapsed_seconds = time.time() - STATE.train_start
                last_checkpoint_step = None
            # A report-only kind gets its own event type. Emitting "unrecoverable"
            # here would be a claim ARC cannot support: it never acted, so it has
            # no evidence the run is past saving — and on the measured lr=0.5 pair
            # that claim would have been flatly false, because the arm that was
            # left alone escaped the plateau and finished at 73.19%. "I have
            # nothing further to add" and "this run is dead" are different
            # statements and only one of them is earned.
            emit({
                "type": "detection_silenced" if report_only else "unrecoverable",
                "step": step,
                "kind": kind,
                "attempts": attempts,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "last_checkpoint_step": last_checkpoint_step,
                "message": (
                    f"'{kind}' has now been reported {attempts} times and the run has not "
                    "resumed progress. ARC has no response to this failure that is known "
                    "to help, so it has not acted, and it will stop repeating the same "
                    "diagnosis. Training continues untouched. Worth checking whether this "
                    "run is still worth its GPU time."
                    if report_only else
                    f"{attempts} recovery attempts for '{kind}' did not restore healthy "
                    "training. Every retained checkpoint predates the collapse's onset or "
                    "is itself degenerate, so further rollbacks cannot help. "
                    "Recommend stopping this run and lowering the base learning rate."
                ),
            })
            emit_log(
                (f"'{kind}' reported {attempts} times with no change. Silencing further "
                 "reports for this kind. ARC has not intervened and is not judging the "
                 "run unrecoverable — it has no basis to, having taken no action."
                 if report_only else
                 f"Run judged unrecoverable after {attempts} attempts at '{kind}'. "
                 "ARC has stopped intervening; the run continues untouched so the failure "
                 "stays visible rather than being masked."),
                "error" if not report_only else "warn",
            )
        return

    if kind in REPORT_ONLY_KINDS:
        # Detect-only. Deliberately placed above the cooldown and baseline gates
        # so both A/B arms behave identically for this kind, and above every
        # path that could touch the run — in particular the zero_grad() at the
        # bottom of this function, which would discard the user's update and is
        # exactly the kind of silent intervention the control arm exists to rule
        # out. See REPORT_ONLY_KINDS for the measurement behind this.
        emit({"type": "thought", "phase": "observation",
              "message": f"Reporting '{kind}' without acting: no response to this failure has "
                         "been shown to help, and the one that was tried made a recoverable "
                         "run terminal. Training continues untouched."})
        STATE.attempts_by_kind[kind] = attempts + 1
        return

    if step - STATE.last_intervention_step < COOLDOWN_STEPS:
        emit({"type": "thought", "phase": "observation",
              "message": f"Within cooldown ({COOLDOWN_STEPS} steps) of the last intervention — "
                         "letting the previous recovery take effect."})
        return

    if not INTERVENTIONS_ENABLED:
        # Baseline arm of the A/B: detect and report, change nothing at all.
        emit({"type": "thought", "phase": "observation",
              "message": "Baseline mode — failure recorded, interventions suppressed."})
        return

    try:
        from arc_agent import run_recovery_agent
        run_recovery_agent(
            step=step,
            epoch=STATE.epoch,
            loss=loss_val,
            grad_norm=grad_norm,
            lr=lr,
            loss_history=list(STATE.loss_history)[-20:],
            optimizer=optimizer,
            monitor=monitor,
            advanced=advanced,
            kind=kind,
            reason=reason,
            emit_thought=lambda m, p="reasoning": emit({"type": "thought", "phase": p, "message": m}),
            emit_intervention=_emit_intervention,
            enable_clipping=_enable_clipping,
            clipping_active=lambda: STATE.grad_clip_enabled,
        )
    except Exception as exc:
        warn_once("agent", f"Recovery agent unavailable ({exc}); applying default recovery.")
        _default_recovery(optimizer, monitor)

    STATE.last_intervention_step = step
    STATE.attempts_by_kind[kind] = STATE.attempts_by_kind.get(kind, 0) + 1
    # The pre-failure losses describe weights that no longer exist after a
    # rollback; keeping them would keep the risk heuristic pinned high.
    STATE.loss_history.clear()

    # The gradients that produced a NaN must never reach the weights. The caller
    # is about to run its own optimizer.step(); zeroing here makes that a no-op.
    try:
        optimizer.zero_grad(set_to_none=True)
    except Exception:
        pass


def _emit_intervention(action: str, detail: str) -> None:
    STATE.intervention_count += 1
    emit({"type": "intervention", "action": action, "detail": detail, "step": STATE.step})


def _enable_clipping(max_norm: float) -> None:
    """Turn on gradient clipping inside the harness.

    The backward-anchored design could only ever *recommend* this, because it
    did not own the caller's ``optimizer.step()``. Anchoring on the optimizer
    means the harness runs immediately before the update and can actually apply
    the clip, so the recommendation became an intervention.
    """
    STATE.grad_clip_enabled = True
    STATE.clip_max_norm = max_norm


def _default_recovery(optimizer, monitor) -> None:
    steps_back = monitor.store.restore() if monitor is not None else 0
    if monitor is not None:
        old_lr, new_lr = monitor.scale_lr(0.2)
    else:
        old_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
        for group in optimizer.param_groups:
            group["lr"] *= 0.2
        new_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
    _emit_intervention(
        "rollback_and_reduce_lr",
        f"Emergency rollback of {steps_back} steps. LR {old_lr:.2e} -> {new_lr:.2e}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Patch installation
# ─────────────────────────────────────────────────────────────────────────────

def _install_optimizer_patch() -> None:
    original_init = torch.optim.Optimizer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            if getattr(self, "_arc_instrumented", False):
                return
            self._arc_instrumented = True

            bound_step = self.step

            def arc_step(_self, *a, **kw):
                # Only the outermost step is the update.
                #
                # A wrapper optimizer (Lookahead, SAM, and much of
                # `torch_optimizer`) subclasses Optimizer and delegates to an
                # inner one, so both objects get instrumented — they are
                # genuinely two objects, which no per-instance flag can reconcile.
                # Counting both reported two steps for one update and left the
                # inner hook with no loss to read, which then looked like a NaN
                # failure on the first step of a healthy run.
                if STATE.in_step:
                    return bound_step(*a, **kw)
                STATE.in_step = True
                try:
                    on_optimizer_step(_self)
                    return bound_step(*a, **kw)
                finally:
                    STATE.in_step = False

            # Bound as a method, not assigned as a plain function.
            #
            # `LRScheduler.__init__` reaches into `optimizer.step.__func__` to
            # install its own step counter. A plain function has no `__func__`,
            # so assigning one makes *constructing any scheduler* raise
            # `AttributeError: 'function' object has no attribute '__func__'` —
            # before training starts, on the majority of real scripts. Binding
            # through MethodType gives the attribute back and keeps the wrapper
            # indistinguishable from the method it replaced.
            self.step = types.MethodType(arc_step, self)  # type: ignore[method-assign]
        except Exception as exc:
            warn_once("optimizer_patch", f"Could not instrument optimizer: {exc}")

    torch.optim.Optimizer.__init__ = patched_init  # type: ignore[method-assign]


def _install_loss_patch() -> None:
    """Record which loss belongs to the upcoming update. Never emits."""
    original_backward = torch.Tensor.backward

    def patched_backward(self, *args, **kwargs):
        try:
            STATE.backward_calls += 1
            # scale() already recorded the unscaled loss; do not overwrite it
            # with the scaled tensor backward() is being called on.
            if not STATE.pending_is_scaled and self.numel() == 1:
                STATE.pending_loss = self.detach()
        except Exception:
            pass
        result = original_backward(self, *args, **kwargs)
        STATE.pending_is_scaled = False
        # A new backward begins a new iteration, so any "already unscaled" marks
        # from the previous one are stale.
        STATE.unscaled_optimizers.clear()
        return result

    torch.Tensor.backward = patched_backward  # type: ignore[method-assign]


def _install_amp_patches() -> None:
    try:
        scaler_cls = torch.amp.GradScaler
    except AttributeError:  # pragma: no cover - very old torch
        return

    original_scale = scaler_cls.scale
    original_step = scaler_cls.step
    original_unscale = getattr(scaler_cls, "unscale_", None)

    def patched_scale(self, outputs):
        try:
            STATE.active_scaler = self
            if torch.is_tensor(outputs) and outputs.numel() == 1:
                # This is the only place the *unscaled* loss is available
                # without having to guess at the scale factor.
                STATE.pending_loss = outputs.detach()
                STATE.pending_is_scaled = True
        except Exception:
            pass
        return original_scale(self, outputs)

    def patched_step(self, optimizer, *args, **kwargs):
        # scaler.step() unscales this optimizer's gradients before running it,
        # so the norm read inside the step hook needs no scale correction.
        already = id(optimizer) in STATE.unscaled_optimizers
        STATE.unscaled_optimizers.add(id(optimizer))
        try:
            return original_step(self, optimizer, *args, **kwargs)
        finally:
            # Leave the mark in place if unscale_ set it — the user may still be
            # mid-iteration, and it is cleared when the next backward starts.
            if not already:
                STATE.unscaled_optimizers.discard(id(optimizer))

    def patched_unscale(self, optimizer, *args, **kwargs):
        # The standard AMP-with-clipping recipe is
        #   scaler.scale(loss).backward(); scaler.unscale_(opt); clip_grad_norm_(...)
        # and only then scaler.step(opt). A user who follows it but calls
        # `opt.step()` directly never went through `scaler.step`, so ARC still
        # believed the gradients were scaled and divided the norm by ~65536 a
        # second time — under-reporting it to the point where no gradient rule
        # could ever fire, and suppressing ARC's own clipping for that step.
        result = original_unscale(self, optimizer, *args, **kwargs)
        STATE.unscaled_optimizers.add(id(optimizer))
        return result

    scaler_cls.scale = patched_scale  # type: ignore[method-assign]
    scaler_cls.step = patched_step  # type: ignore[method-assign]
    if original_unscale is not None:
        scaler_cls.unscale_ = patched_unscale  # type: ignore[method-assign]


_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    if not HAS_TORCH:
        warn_once("no_torch", "PyTorch not found — ARC monitoring is inactive.")
        return

    _install_optimizer_patch()
    _install_loss_patch()
    _install_amp_patches()

    if not HAS_ARC:
        warn_once(
            "no_arc",
            "arc-training not installed — effective rank, gradient entropy and "
            "update-ratio diagnostics are unavailable. Install with: pip install arc-training",
        )

    device = None
    if torch.cuda.is_available():
        try:
            device = torch.cuda.get_device_name(0)
        except Exception:
            device = None

    emit({
        "type": "environment",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": device,
        "arc": getattr(_arc_pkg, "__version__", None) if HAS_ARC else None,
        "mode": MODE,
        "python": sys.version.split()[0],
    })
    emit_log(
        f"ARC instrumentation installed (anchor: Optimizer.step, mode: {MODE})."
        + ("" if HAS_ARC else " Running without arc-training signals.")
    )


def set_epoch(epoch: int) -> None:
    STATE.epoch = int(epoch)


_finished = False


def finish(status: str = "complete", message: str | None = None) -> None:
    # Emitted exactly once. Two run summaries would double the step counts a
    # reader takes at face value, and the extension host treats the last one it
    # sees as authoritative.
    global _finished
    if _finished:
        return
    _finished = True

    wall = time.time() - STATE.train_start
    emit({
        "type": "run_summary",
        "steps": STATE.step,
        "backward_calls": STATE.backward_calls,
        "interventions": STATE.intervention_count,
        "wall_seconds": round(wall, 3),
        # Wall-clock time spent inside ARC's hooks. Deliberately NOT reported as
        # an overhead percentage: the hook reads the loss and gradient norm, and
        # that read blocks until the GPU has finished work that was already
        # queued and that the training script's own loss.item() would have waited
        # on moments later. Charging that wait to ARC overstates its cost several
        # times over — this figure read 54% on a run whose true wall-clock cost,
        # measured by running the same loop with and without instrumentation,
        # was 8.4%. For the real number see python/benchmark_overhead.py.
        "instrumentation_seconds": round(STATE.instrument_seconds, 3),
        "degraded": list(_degraded),
        "unrecoverable": sorted(STATE.abandoned_kinds),
        "mode": MODE,
    })
    emit({"type": "status", "status": status, "message": message or "Training finished."})
