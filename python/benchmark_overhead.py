"""
ARC Lens — instrumentation overhead benchmark
=============================================

Measures what ARC actually costs by running the *same* training loop twice: once
bare, once through the instrumentation, and comparing wall-clock time.

This is the only overhead number worth quoting. The harness can time its own
hooks, but that figure is misleading in both directions: it charges ARC for GPU
work that was already queued and would have been waited on by the training
script's own ``loss.item()`` a moment later, and it misses cache and scheduling
effects. Wall-clock A/B has neither problem.

    python python/benchmark_overhead.py --steps 300 --batch 128

Prints a table suitable for pasting into the README, and writes
``docs/benchmark_overhead.json``.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

WORKLOAD = '''
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, r"{here}")
from train_demo import DemoCNN

STEPS = {steps}
BATCH = {batch}
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = DemoCNN().to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
x = torch.randn(BATCH, 3, 32, 32, device=dev)
y = torch.randint(0, 10, (BATCH,), device=dev)

for _ in range(10):
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    opt.step()
if dev == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(STEPS):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    loss.item()
if dev == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start
sys.stderr.write("ELAPSED %.6f\\n" % elapsed)
'''


def run_once(script_path: Path, instrumented: bool, env_extra: dict) -> float:
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = ([sys.executable, str(HERE / "runner.py"), str(script_path)]
           if instrumented else [sys.executable, str(script_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
    for line in proc.stderr.splitlines():
        if line.startswith("ELAPSED "):
            return float(line.split()[1])
    raise RuntimeError(
        f"workload did not report timing (instrumented={instrumented})\n"
        f"stdout tail: {proc.stdout[-500:]}\nstderr tail: {proc.stderr[-500:]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--advanced-every", type=int, default=25)
    args = parser.parse_args()

    tmp = REPO / ".bench_workload.py"
    tmp.write_text(WORKLOAD.format(here=str(HERE), steps=args.steps, batch=args.batch), encoding="utf-8")

    try:
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        torch_version = torch.__version__
    except Exception:
        device, torch_version = "unknown", "unknown"

    configs = [
        ("bare (no ARC)", False, {}),
        ("ARC core metrics only", True, {"ARC_ADVANCED_EVERY": "10000000"}),
        (f"ARC full (advanced every {args.advanced_every})", True,
         {"ARC_ADVANCED_EVERY": str(args.advanced_every)}),
        ("ARC full (advanced every step)", True, {"ARC_ADVANCED_EVERY": "1"}),
    ]

    results = []
    try:
        for label, instrumented, extra in configs:
            times = [run_once(tmp, instrumented, extra) for _ in range(args.repeats)]
            results.append({
                "config": label,
                "median_seconds": statistics.median(times),
                "samples": times,
            })
    finally:
        tmp.unlink(missing_ok=True)

    baseline = results[0]["median_seconds"]
    for row in results:
        row["ms_per_step"] = row["median_seconds"] / args.steps * 1000
        row["overhead_percent"] = (row["median_seconds"] / baseline - 1.0) * 100

    print()
    print(f"Device: {device} | torch {torch_version} | "
          f"{args.steps} steps x batch {args.batch}, median of {args.repeats}")
    print()
    print(f"| {'Configuration':<38} | {'s/run':>8} | {'ms/step':>8} | {'overhead':>9} |")
    print(f"| {'-' * 38} | {'-' * 8} | {'-' * 8} | {'-' * 9} |")
    for row in results:
        print(f"| {row['config']:<38} | {row['median_seconds']:>8.3f} | "
              f"{row['ms_per_step']:>8.2f} | {row['overhead_percent']:>8.1f}% |")
    print()

    out = REPO / "docs" / "benchmark_overhead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": device,
        "torch": torch_version,
        "steps": args.steps,
        "batch": args.batch,
        "repeats": args.repeats,
        "model": "DemoCNN (2.79M params)",
        "generated": time.strftime("%Y-%m-%d"),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
