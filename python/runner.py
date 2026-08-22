"""
ARC Lens — backend runner
=========================

Spawned by the VS Code extension with the user's training script as argv[1].

Installs the instrumentation from ``_arc_bootstrap`` and then executes the
target script **unmodified** via ``runpy``. Nothing is prepended to the user's
source, so line numbers in a traceback are the line numbers in their file — for
a tool whose whole purpose is diagnosing training failures, reporting the wrong
line is a direct hit on the core value proposition.

All communication with the extension is newline-delimited JSON on stdout.
"""

from __future__ import annotations

import builtins
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _arc_bootstrap as arc_bootstrap  # noqa: E402


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def _install_helpers() -> None:
    """Expose the optional helpers without requiring an import in user code.

    Both are opt-in. ``arc_watch`` is the escape hatch for topologies where
    parameter-identity matching cannot resolve a model; ``arc_set_epoch`` lets a
    script label its epochs so the dashboard x-axis is meaningful. A script that
    calls neither is fully monitored — that is the default path.
    """
    builtins.arc_watch = arc_bootstrap.watch  # type: ignore[attr-defined]
    builtins.arc_set_epoch = arc_bootstrap.set_epoch  # type: ignore[attr-defined]


def _user_traceback(exc: BaseException, target: str) -> str:
    """Format the traceback with this runner's own frames removed.

    The user did not write runner.py and cannot act on its frames; leaving them
    in the report just buries the line that actually matters.
    """
    entries = traceback.extract_tb(exc.__traceback__)
    runner_dir = str(Path(__file__).resolve().parent)
    user_entries = [e for e in entries if not str(Path(e.filename).parent) == runner_dir]
    lines = ["Traceback (most recent call last):"]
    lines += traceback.format_list(user_entries or entries)
    lines += traceback.format_exception_only(type(exc), exc)
    return "".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        emit({"type": "error", "message": "Usage: runner.py <training_script.py>"})
        return 2

    target = sys.argv[1]
    if not os.path.exists(target):
        emit({"type": "error", "message": f"File not found: {target}"})
        return 2

    emit({"type": "log", "level": "info", "message": f"ARC Lens starting: {Path(target).name}"})

    arc_bootstrap.install()
    _install_helpers()

    emit({"type": "status", "status": "running"})

    target_dir = str(Path(target).resolve().parent)
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
        arc_bootstrap.finish("complete" if code == 0 else "error", f"Script exited with code {code}.")
        return code
    except BaseException as exc:  # noqa: BLE001 - report anything the script raises
        emit({
            "type": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": _user_traceback(exc, target),
        })
        arc_bootstrap.finish("error", str(exc))
        return 1

    arc_bootstrap.finish("complete", "Training finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
