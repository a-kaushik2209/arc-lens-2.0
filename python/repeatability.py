"""
ARC Lens — run-to-run variance check
====================================

Runs the *same* configuration N times and reports the spread.

This exists because the A/B in `experiment_ab.py` compares two arms and
attributes the difference to the interventions. That inference is only valid if
two identical runs land closer together than the effect being measured — and on
a GPU they may not. cuDNN selects non-deterministic kernels and CUDA reductions
are not associative, so a fixed seed does not fix the arithmetic. Near the edge
of stability, where a learning rate is high enough for interventions to matter,
those tiny differences are exactly the ones that amplify.

Without this number the A/B has no error bar and every delta is unfalsifiable.

    python python/repeatability.py --lr 0.25 --repeats 4 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import data_cache

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def run_once(script: Path, lr: float, epochs: int, seed: int, data_root: str) -> dict:
    env = dict(os.environ)
    env.update({
        "PYTHONUNBUFFERED": "1",
        "ARC_MODE": "baseline",   # observe only — this measures the workload, not ARC
        "ARC_DEMO_LR": str(lr),
        "ARC_DEMO_EPOCHS": str(epochs),
        "ARC_DEMO_SEED": str(seed),
        "ARC_STEP_DELAY": "0",
    })
    if data_root:
        env["ARC_DEMO_DATA"] = data_root

    proc = subprocess.run(
        [sys.executable, str(HERE / "runner.py"), str(script)],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )
    result, failures = None, 0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "final_result":
            result = event
        elif event.get("type") == "failure_detected":
            failures += 1
    return {
        "best_val_acc": (result or {}).get("best_val_acc"),
        "failures": failures,
        "exit_code": proc.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234,
                        help="Same seed every run — the point is that the seed is not enough.")
    parser.add_argument("--script", default=str(HERE / "train_demo.py"))
    parser.add_argument("--data-root", default=data_cache.cifar_root())
    args = parser.parse_args()

    script = Path(args.script).resolve()
    print(f"Repeatability: lr={args.lr}, {args.repeats} runs, identical seed {args.seed}, "
          f"{args.epochs} epochs, interventions off.\n")

    rows = []
    for i in range(args.repeats):
        print(f"  run {i + 1}/{args.repeats} ...", flush=True)
        row = run_once(script, args.lr, args.epochs, args.seed, args.data_root)
        print(f"    -> best_val_acc={row['best_val_acc']} failures={row['failures']}", flush=True)
        rows.append(row)

    accs = [r["best_val_acc"] for r in rows if isinstance(r["best_val_acc"], (int, float))]
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "lr": args.lr,
        "seed": args.seed,
        "epochs": args.epochs,
        "runs": rows,
    }
    if len(accs) >= 2:
        out["mean"] = round(statistics.mean(accs), 3)
        out["stdev"] = round(statistics.stdev(accs), 3)
        out["spread"] = round(max(accs) - min(accs), 3)
        print(f"\n  mean {out['mean']}%   stdev {out['stdev']}   "
              f"min {min(accs)}   max {max(accs)}   spread {out['spread']}pp")
        print(f"\n  Any A/B delta smaller than ~{out['spread']}pp at this learning rate "
              f"cannot be attributed to interventions.")
    else:
        print("\n  Not enough completed runs to compute a spread.")

    path = REPO / "docs" / f"repeatability_lr{args.lr}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
