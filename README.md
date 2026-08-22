# ARC Lens

ARC Lens is a real-time training monitor and automated recovery controller for PyTorch, inside
VS Code. It watches a training run, streams its optimisation telemetry to a dashboard, and when
the run starts to fail it restores the model to a healthy checkpoint, lowers the learning rate,
or turns on gradient clipping — without any change to your training code.

It is the IDE frontend for the **ARC (Autonomic Recovery Controller)** framework, built on the
[`arc-training`](https://pypi.org/project/arc-training/) package.

**Everything in this repository is the real implementation.** Nothing is simulated: no injected
failures, no scripted curves, no synthesised metrics. Charts show measurements or they show
gaps.

---

## What makes it different

Every monitoring tool plots a loss curve. Two things here are not standard:

**It intervenes.** When a run diverges, ARC restores weights from a checkpoint, scales the
learning rate, and resumes — automatically, mid-run.

**It knows when to stop.** After three failed recoveries of the same kind it declares the run
unrecoverable and says so, instead of rolling back forever — because at that point the useful
answer is "kill this run", not a fourth rollback.

**It is measured against a control arm, and the measurements have repeatedly gone against it.**
Three structural detection rules were built, and two were deleted after an A/B showed each one
intervening on healthy runs: one cost 1.74 and 0.78 points of validation accuracy, the other
took a run from 87.4% to chance. Both were removed rather than retuned, because the underlying
statistics do not separate a healthy run from a failing one on real workloads — see
[`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md).

What survives is deliberately narrow: divergence that is unambiguous (non-finite or exploded
loss), gradient explosion, and a representation-collapse rule whose threshold is conservative
enough that it has never fired in validation. The structural signals are still collected and
charted — they are informative to a human reading a run — they just no longer get to act on
their own.

---

## How it works

Three tiers:

1. **Extension host** (`src/`) — resolves your interpreter, spawns the run, parses telemetry,
   drives the dashboard and the LLM features.
2. **Instrumentation harness** (`python/_arc_bootstrap.py`) — patches PyTorch to measure every
   weight update, owns a host-resident checkpoint store, and detects failures.
3. **Recovery agent** (`python/arc_agent.py`) — a deterministic rule engine that chooses and
   applies the response.

The measurement anchor is `Optimizer.step`, not `loss.backward()`. That means one recorded step
is one *weight update*, which is what makes gradient accumulation, AMP and multi-optimizer
setups (GANs) correct rather than merely non-crashing. Your source is executed unmodified via
`runpy`, so tracebacks report the line numbers that are actually in your file.

Full detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Install and run

```bash
pip install torch arc-training      # in the interpreter you have selected in VS Code
npm install && npm run compile
```

Press `F5` to launch the Extension Development Host, open a `.py` training script, and click
**▶ Run with ARC Lens** in the editor toolbar.

ARC Lens uses the interpreter the Python extension has selected for the file — the venv your
`torch` and `arc-training` are actually installed in. It works without `arc-training`; you lose
the structural diagnostics, and it tells you so rather than leaving a blank chart.

### Try it on the reference script

`python/train_demo.py` is a real 9-layer CNN on real CIFAR-10, with a deliberately aggressive
learning rate. No failure is injected — whether it diverges, and at which step, depends on the
data order and the initialisation.

```bash
python python/runner.py python/train_demo.py
```

---

## Commands

| Command | What it does |
| :--- | :--- |
| **▶ Run with ARC Lens** | Monitor and recover the active script |
| **Run Baseline (interventions off)** | Same run, recovery suppressed — the A/B control arm |
| **Export Run Report** | Self-contained HTML incident report |
| **Open AI Failure Analyst** | Chat about the run, with its telemetry attached |
| **Generate ARC-Tested Script** | Generate a pre-instrumented training script |

---

## Telemetry

**Core** — loss, learning rate, gradient L2 norm, GPU memory.

**Structural** (needs `arc-training`) — effective rank (representation collapse), gradient
entropy (whether gradients still carry information), weight update ratio (‖ΔW‖/‖W‖), gradient
flow ratio (early vs late layer gradients; needs ≥4 parameterised layers).

## Interventions

| Action | Trigger |
| :--- | :--- |
| `rollback_and_reduce_lr` | Loss non-finite or exploded past 1e6, or a confirmed representation collapse |
| `enable_grad_clipping` | Gradient norm above 50 — applied by ARC, not just recommended |

**Two rules were removed after measurement, and neither is coming back without new evidence.**

*Weight update ratio* fired above an absolute ceiling. Measured across four learning rates, its
distribution on a healthy run overlaps a failing one almost completely: the p90 values are
effectively identical (0.089 healthy vs 0.088 damaged), the peaks barely separate (0.285 vs
0.322), and the healthy run sustained a *longer* consecutive breach than the damaged one — 31
samples against 26. It was a proxy for "the learning rate is large", not "training is failing".

*Gradient entropy* fired below 1% of an opening baseline. A healthy run and a dead one settle to
the same value (~1.45e-05) from around step 70, so no threshold separates them; the upstream
estimator bins a heavy-tailed distribution linearly and saturates for every run.

The pattern behind both: these signals change by orders of magnitude in a run's opening steps
simply because the model goes from random to structured, so a rule written against that
transient fires on healthy training. Structural checks therefore wait 200 steps before capturing
a baseline, thresholds are relative to that baseline rather than absolute, and every rule must
hold for several consecutive samples.

---

## Overhead

Measured, not asserted — the same loop run with and without the harness
(`python python/benchmark_overhead.py`). RTX 3050, 2.79M-parameter CNN, 200 steps × batch 128,
median of 3:

| Configuration | ms/step | Overhead |
| :--- | ---: | ---: |
| bare (no ARC) | 49.09 | — |
| ARC core metrics only | 49.97 | **1.8%** |
| ARC full (advanced every 25 steps) | 53.20 | **8.4%** |
| ARC full (advanced every step) | 132.55 | 170.0% |

The last row is why expensive signals are sampled rather than collected every step.

## Does it actually help?

`ARC_MODE=baseline` runs the identical instrumented code path with every intervention
suppressed, so a comparison against a normal run isolates the interventions and nothing else.
Both arms use the same seed and the same data order.

```bash
python python/experiment_ab.py --lrs 0.03 0.1 0.25 0.5 --epochs 10
```

Measured results — including the configurations where ARC detects the failure and **cannot**
save the run — are in [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md).

---

## Configuration

| Setting | Default | Description |
| :--- | :--- | :--- |
| `arcAgent.pythonPath` | `python3` | Overrides the Python extension's interpreter. Machine-scope: a workspace cannot set it |
| `arcAgent.stepDelay` | `0` | Artificial per-step delay for demos. Non-zero is wasted GPU time |
| `arcAgent.gpuHourlyRate` | `0` | Your GPU's hourly cost. `0` estimates from the detected device |
| `arcAgent.openRouterKey` | `""` | API key for the AI features. Provider inferred from the prefix |
| `arcAgent.llmModel` | `google/gemini-2.5-flash:free` | Model for the AI features |

Harness behaviour is tunable through environment variables (`ARC_ADVANCED_EVERY`,
`ARC_CHECKPOINT_EVERY`, `ARC_MAX_ATTEMPTS`, …) — see
[`docs/ARCHITECTURE.md` §9](docs/ARCHITECTURE.md).

---

## Tests

```bash
npm test                       # TypeScript + dashboard suites
python tests/test_harness.py   # harness, detector, checkpointing, end-to-end
```

98 tests — 54 TypeScript/dashboard, 44 Python. The Python suite includes nine end-to-end tests
that run the real harness against real training loops, asserting that gradient accumulation does
not inflate the step count, that an LR intervention survives a scheduler rewriting the learning
rate every step, that baseline mode never intervenes, and that tracebacks point at the user's
own line numbers.

Several of these tests were written before the code they cover and found real bugs doing it —
including one where an optimizer was matched to a wrapper module rather than the submodule it
actually updates, which would have rolled back both halves of a GAN.

CI additionally fails the build on any secret-shaped literal in source.

---

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pieces fit, and why the hook is where
  it is
* [`docs/SECURITY_AUDIT.md`](docs/SECURITY_AUDIT.md) — full audit with remediation status
* [`docs/FUTURE_IMPROVEMENTS.md`](docs/FUTURE_IMPROVEMENTS.md) — roadmap, and what is
  deliberately still open
* [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md) — measured A/B results
* [`docs/WASTE_REDUCTION.md`](docs/WASTE_REDUCTION.md) — measured waste-reduction
  numbers (telemetry bytes, post-verdict compute, accessibility score), both arms,
  including the two results that came back negative

## License

GNU Affero General Public License v3.0 (AGPL-3.0).
