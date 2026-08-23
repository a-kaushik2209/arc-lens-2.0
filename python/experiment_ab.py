"""
ARC Lens — baseline vs. active A/B experiment
=============================================

Runs the same training script twice per configuration, seeded identically:

  * **baseline** — full telemetry, every intervention suppressed
  * **active**   — identical, interventions allowed

Any difference in the outcome is therefore caused by the interventions and by
nothing else. The seed, the data order, the initialisation and the schedule are
all held fixed; the harness itself runs in both arms, so even instrumentation
effects cancel.

This exists because "our tool recovers training runs" is a claim, and a claim a
skeptical reader cannot reproduce is worth very little. The result is whatever
it is — including the configurations where ARC detects a failure and cannot
save it, which are reported alongside the ones where it can.

    python python/experiment_ab.py --lrs 0.05 0.15 0.5 --epochs 8

Writes results/experiment_ab.json and prints a comparison table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import data_cache

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def run_arm(script: Path, mode: str, lr: float, epochs: int, seed: int, data_root: str) -> dict:
    env = dict(os.environ)
    env.update({
        "PYTHONUNBUFFERED": "1",
        "ARC_MODE": mode,
        "ARC_DEMO_LR": str(lr),
        "ARC_DEMO_EPOCHS": str(epochs),
        "ARC_DEMO_SEED": str(seed),
        "ARC_STEP_DELAY": "0",
    })
    if data_root:
        env["ARC_DEMO_DATA"] = data_root

    started = time.time()
    proc = subprocess.run(
        [sys.executable, str(HERE / "runner.py"), str(script)],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )
    wall = time.time() - started

    events, epochs_seen = [], []
    result = None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "final_result":
                result = event
            elif event.get("type") != "metric":
                events.append(event)
        elif stripped.startswith("[epoch"):
            epochs_seen.append(stripped)

    failures = [e for e in events if e.get("type") == "failure_detected"]
    interventions = [e for e in events if e.get("type") == "intervention"]
    unrecoverable = [e for e in events if e.get("type") == "unrecoverable"]
    summary = next((e for e in events if e.get("type") == "run_summary"), {})

    return {
        "mode": mode,
        "lr": lr,
        "seed": seed,
        "exit_code": proc.returncode,
        "wall_seconds": round(wall, 1),
        "best_val_acc": (result or {}).get("best_val_acc"),
        "final_val_acc": (result or {}).get("final_val_acc"),
        "final_val_loss": (result or {}).get("final_val_loss"),
        "steps": summary.get("steps"),
        "failures": len(failures),
        "interventions": len(interventions),
        "intervention_actions": [i.get("action") for i in interventions],
        "failure_kinds": sorted({f.get("kind", "numerical") for f in failures}),
        "first_failure_step": failures[0]["step"] if failures else None,
        "unrecoverable": bool(unrecoverable),
        "degraded": summary.get("degraded", []),
        "epoch_lines": epochs_seen,
        "stderr_tail": proc.stderr[-400:] if proc.returncode != 0 else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.05, 0.15, 0.5])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--script", default=str(HERE / "train_demo.py"))
    parser.add_argument("--data-root", default=data_cache.cifar_root())
    args = parser.parse_args()

    script = Path(args.script).resolve()
    pairs = []

    print(f"ARC Lens A/B — {len(args.lrs)} learning rate(s) x 2 arms x {args.epochs} epochs")
    print("Identical seed and data order in both arms; only interventions differ.\n")

    out = REPO / "results" / "experiment_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        """Persist after every arm.

        A full sweep is the better part of an hour of GPU time. Writing the file
        only at the end means one interruption — a reboot, a killed shell, a
        laptop lid — discards every completed arm along with the unfinished one.
        Partial results are still results.
        """
        out.write_text(json.dumps({
            "generated": time.strftime("%Y-%m-%d %H:%M"),
            "epochs": args.epochs,
            "seed": args.seed,
            "script": str(script.name),
            "complete": len(pairs) == len(args.lrs) * 2,
            "runs": pairs,
        }, indent=2), encoding="utf-8")

    for lr in args.lrs:
        for mode in ("baseline", "active"):
            print(f"  running lr={lr} mode={mode} ...", flush=True)
            row = run_arm(script, mode, lr, args.epochs, args.seed, args.data_root)
            status = "unrecoverable" if row["unrecoverable"] else ("ok" if row["exit_code"] == 0 else "crashed")
            print(f"    -> best_val_acc={row['best_val_acc']} "
                  f"failures={row['failures']} interventions={row['interventions']} "
                  f"kinds={','.join(row['failure_kinds']) or '-'} "
                  f"[{status}] {row['wall_seconds']}s", flush=True)
            pairs.append(row)
            save()

    print()
    header = (f"| {'peak LR':>8} | {'arm':<9} | {'best val acc':>12} | {'failures':>8} | "
              f"{'interv.':>7} | {'first fail':>10} | {'verdict':<16} |")
    print(header)
    print("|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 14 + "|" + "-" * 10 + "|"
          + "-" * 9 + "|" + "-" * 12 + "|" + "-" * 18 + "|")

    for lr in args.lrs:
        arms = [r for r in pairs if r["lr"] == lr]
        for row in arms:
            verdict = ("unrecoverable" if row["unrecoverable"]
                       else "crashed" if row["exit_code"] != 0
                       else "completed")
            acc = "—" if row["best_val_acc"] is None else f"{row['best_val_acc']:.2f}%"
            first = "—" if row["first_failure_step"] is None else str(row["first_failure_step"])
            print(f"| {lr:>8} | {row['mode']:<9} | {acc:>12} | {row['failures']:>8} | "
                  f"{row['interventions']:>7} | {first:>10} | {verdict:<16} |")

        base = next((r for r in arms if r["mode"] == "baseline"), None)
        act = next((r for r in arms if r["mode"] == "active"), None)
        if base and act and base["best_val_acc"] is not None and act["best_val_acc"] is not None:
            delta = act["best_val_acc"] - base["best_val_acc"]
            sign = "+" if delta >= 0 else ""
            print(f"| {'':>8} | {'delta':<9} | {sign}{delta:>11.2f}% | {'':>8} | {'':>7} | {'':>10} | "
                  f"{'ARC effect':<16} |")

    save()
    print(f"\nWrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
