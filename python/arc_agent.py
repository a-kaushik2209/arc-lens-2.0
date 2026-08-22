"""
ARC Lens — recovery agent
=========================

A **deterministic rule engine**, presented as a ReAct trace.

It emits ``thought`` events shaped like an agent's reasoning — perception,
action, observation — but it selects tools by threshold, not by inference, and
it makes no network call. That is a deliberate choice: the reflex path of a
reliability tool must not depend on a remote service being up, and a NaN needs
handling in microseconds, not in a round trip. The LLM in this product sits in
the *analysis* path (the Failure Analyst panel), where latency is acceptable and
a wrong answer costs nothing.

It runs at full speed. Pacing the trace for readability is the dashboard's job;
the training process is holding a GPU while this executes.
"""

from __future__ import annotations

import json
import math
from typing import Callable, List, Optional

# Thresholds. Named rather than inlined because these are the tool's actual
# operating policy, and a reviewer should be able to find and argue with them.
GRAD_EXPLOSION_NORM = 50.0
CLIP_MAX_NORM = 1.0
NAN_LR_FACTOR = 0.2
SOFT_LR_FACTOR = 0.5
LOSS_EXPLOSION = 1e6

# Structural triggers raised by the detector in _arc_bootstrap, with the
# response each one warrants.
#
# One entry, and it is worth saying plainly that there used to be three.
#
# `update_ratio_high` and `gradient_entropy_collapse` were both removed after
# measurement, and both had cost real accuracy first: 1.74 and 0.78 points on
# healthy runs for the former, and an entire run — 87.4% to 10% — for the
# latter. In each case the statistic simply did not separate a healthy run from
# a failing one on a real workload. The measurements are recorded in
# `_arc_bootstrap.check_structural`.
#
# The lesson generalises, and is the reason this table is now hard to add to:
# every one of these signals changes by orders of magnitude in a run's opening
# steps purely because the model goes from random to structured. A rule written
# against that transient fires on healthy training. A new entry here needs a
# measured trajectory showing separation on both a healthy and a failing run,
# not a plausible story about what the signal means.
#
# `representation_collapse` gets a rollback as well as an LR cut: by the time a
# genuine collapse is confirmed the network has usually been degenerate for a
# while, and lowering the learning rate on a dead network only makes it die
# slowly. It has never fired in validation — the threshold sits at 50% of
# baseline while a healthy run bottoms at 96% and a damaged one at 83% — so it
# is conservative, and correspondingly unexercised.
STRUCTURAL_RESPONSES = {
    "representation_collapse": (
        "The layers are collapsing onto a low-dimensional subspace, so the model is "
        "losing capacity it will not recover on its own. This produces no NaN and no "
        "gradient spike. Restoring the last healthy checkpoint and reducing the "
        "learning rate.",
        "rollback_and_reduce",
    ),
}
STRUCTURAL_KINDS = frozenset(STRUCTURAL_RESPONSES)


def _snapshot(step, epoch, loss, grad_norm, lr, loss_history, advanced) -> dict:
    is_nan = math.isnan(loss) or math.isinf(loss)
    trend = "stable"
    if len(loss_history) >= 3:
        trend = "rising" if loss_history[-1] > loss_history[-3] else "falling"
    snapshot = {
        "step": step,
        "epoch": epoch,
        "loss": "NaN/Inf" if is_nan else round(loss, 6),
        "is_nan_or_inf": is_nan,
        "grad_norm": round(grad_norm, 4),
        "learning_rate": lr,
        "loss_trend": trend,
        "recent_losses": [round(x, 4) for x in loss_history[-8:]],
    }
    if advanced:
        snapshot["advanced_telemetry"] = {k: round(v, 6) for k, v in advanced.items()}
    return snapshot


def _scale_lr(optimizer, monitor, factor: float) -> tuple[float, float]:
    """Scale the learning rate durably.

    Delegates to the monitor, which scales the live optimizer, every stored
    checkpoint, and every future step (so the user's scheduler cannot silently
    undo the reduction on the next iteration). The fallback path exists only for
    the case where no model could be matched to the optimizer.
    """
    if monitor is not None:
        return monitor.scale_lr(factor)
    old_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
    for group in optimizer.param_groups:
        group["lr"] *= factor
    return old_lr, (optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0)


def run_recovery_agent(
    step: int,
    epoch: int,
    loss: float,
    grad_norm: float,
    lr: float,
    loss_history: List[float],
    optimizer,
    monitor,
    advanced: Optional[dict],
    emit_thought: Callable,
    emit_intervention: Callable,
    enable_clipping: Callable,
    clipping_active: Callable,
    kind: str = "numerical",
    reason: str = "",
) -> None:
    emit_thought(
        f"Anomaly detected ({kind}). Initialising ARC recovery agent."
        + (f" Trigger: {reason}." if reason else ""),
        "perception",
    )

    emit_thought("Calling tool: get_training_snapshot()", "action")
    snapshot = _snapshot(step, epoch, loss, grad_norm, lr, loss_history, advanced)
    emit_thought(f"Tool result: {json.dumps(snapshot)}", "observation")

    is_nan = math.isnan(loss) or math.isinf(loss)
    # A loss of 4e15 is no more recoverable than a NaN: the weights that produced
    # it are already destroyed, and no learning-rate change repairs them. Gating
    # rollback on NaN alone leaves an exploded-but-finite run to keep diverging
    # while the log cheerfully reports successful interventions.
    weights_unrecoverable = is_nan or abs(loss) > LOSS_EXPLOSION
    recovered = False

    # ── 1. Numerical failure: roll back and cut the LR hard ──────────────────
    if weights_unrecoverable:
        state = "non-finite" if is_nan else f"exploded to {loss:.3e}"
        emit_thought(
            f"Analysis: loss is {state}. The current weights cannot be recovered by "
            "tuning — reducing the learning rate alone would keep optimising from a "
            "destroyed state. Restoring the last healthy checkpoint and scaling the "
            "learning rate down before resuming.",
            "reasoning",
        )
        emit_thought("Calling tool: rollback_and_reduce_lr()", "action")
        steps_back = monitor.store.restore() if monitor is not None else 0
        old_lr, new_lr = _scale_lr(optimizer, monitor, NAN_LR_FACTOR)
        emit_intervention(
            "rollback_and_reduce_lr",
            f"Rolled back {steps_back} steps. LR {old_lr:.2e} -> {new_lr:.2e}",
        )
        emit_thought(
            f"Tool result: {json.dumps({'success': True, 'steps_back': steps_back, 'old_lr': old_lr, 'new_lr': new_lr})}",
            "observation",
        )
        recovered = True

    # ── 2. Gradient explosion: actually clip, don't just suggest it ──────────
    # Clipping is a latch, not a per-step action. Re-emitting it every step would
    # flood the log with an intervention that changes nothing after the first.
    if grad_norm > GRAD_EXPLOSION_NORM and not clipping_active():
        emit_thought(
            f"Analysis: gradient L2 norm {grad_norm:.2f} exceeds the stability threshold "
            f"({GRAD_EXPLOSION_NORM:.0f}). Enabling gradient clipping inside the harness so "
            "the constraint applies to every subsequent update.",
            "reasoning",
        )
        emit_thought(f"Calling tool: enable_grad_clipping(max_norm={CLIP_MAX_NORM})", "action")
        enable_clipping(CLIP_MAX_NORM)
        emit_intervention(
            "enable_grad_clipping",
            f"Gradient clipping enabled at max_norm={CLIP_MAX_NORM} (applied by ARC, no user code change).",
        )
        emit_thought(
            f"Tool result: {json.dumps({'success': True, 'max_norm': CLIP_MAX_NORM, 'applied_by': 'arc_harness'})}",
            "observation",
        )
        recovered = True

    # ── 3. Structural pathologies that never produce a NaN ───────────────────
    # These arrive as their own trigger kinds now. The detector upstream has
    # already confirmed the condition held for several consecutive samples
    # against the run's own baseline, so the agent's job here is to choose the
    # response, not to re-test the threshold.
    if kind in STRUCTURAL_KINDS:
        analysis, action = STRUCTURAL_RESPONSES[kind]
        emit_thought(f"Analysis: {reason}. {analysis}", "reasoning")

        if action == "rollback_and_reduce":
            emit_thought(f"Calling tool: rollback_and_reduce_lr(factor={SOFT_LR_FACTOR})", "action")
            steps_back = monitor.store.restore() if monitor is not None else 0
            old_lr, new_lr = _scale_lr(optimizer, monitor, SOFT_LR_FACTOR)
            emit_intervention(
                "rollback_and_reduce_lr",
                f"Rolled back {steps_back} steps. LR {old_lr:.2e} -> {new_lr:.2e} ({kind})",
            )
            emit_thought(
                f"Tool result: {json.dumps({'success': True, 'steps_back': steps_back, 'old_lr': old_lr, 'new_lr': new_lr})}",
                "observation",
            )
        else:
            emit_thought(f"Calling tool: reduce_learning_rate(factor={SOFT_LR_FACTOR})", "action")
            old_lr, new_lr = _scale_lr(optimizer, monitor, SOFT_LR_FACTOR)
            emit_intervention("reduce_lr", f"LR {old_lr:.2e} -> {new_lr:.2e} ({kind})")
            emit_thought(
                f"Tool result: {json.dumps({'success': True, 'old_lr': old_lr, 'new_lr': new_lr})}",
                "observation",
            )
        recovered = True

    if not recovered and not weights_unrecoverable:
        emit_thought(
            "Analysis: failure threshold tripped but no specific rule matched. "
            "Applying the conservative default: rollback plus learning-rate reduction.",
            "reasoning",
        )
        steps_back = monitor.store.restore() if monitor is not None else 0
        old_lr, new_lr = _scale_lr(optimizer, monitor, NAN_LR_FACTOR)
        emit_intervention(
            "rollback_and_reduce_lr",
            f"Fallback: rolled back {steps_back} steps. LR {old_lr:.2e} -> {new_lr:.2e}",
        )

    emit_thought("Recovery applied. Resuming training.", "action")
