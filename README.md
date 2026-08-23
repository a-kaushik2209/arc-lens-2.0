<p align="center">
  <img src="media/logo.png" alt="" width="96">
</p>

<h1 align="center">ARC Lens</h1>

<p align="center">
  Real-time training monitor and automated recovery controller for PyTorch, inside VS Code.
</p>

<p align="center">
  <a href="https://github.com/a-kaushik2209/arc-lens-2.0/actions/workflows/ci.yml"><img src="https://github.com/a-kaushik2209/arc-lens-2.0/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/tests-186-brightgreen" alt="186 tests">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="AGPL-3.0">
  <img src="https://img.shields.io/badge/PyTorch-instrumented%20externally-ee4c2c" alt="PyTorch">
</p>

---

ARC Lens watches a training run and streams its optimisation telemetry to a dashboard. When the
run starts to fail, it restores the model to a healthy checkpoint, lowers the learning rate, or
turns on gradient clipping. Your training code does not change.

It is the IDE frontend for the **ARC (Autonomic Recovery Controller)** framework, built on the
[`arc-training`](https://pypi.org/project/arc-training/) package.

> **Everything here is the real implementation.** No injected failures, no scripted curves, no
> synthesised metrics. Charts show measurements or they show gaps.

**Contents**: [Quick start](#quick-start) · [What it does](#what-it-does) ·
[How it works](#how-it-works) · [Telemetry](#telemetry) · [Interventions](#interventions) ·
[Overhead](#overhead) · [Evidence](#evidence) · [Configuration](#configuration) ·
[Tests](#tests)

---

## Quick start

```bash
pip install torch arc-training      # in the interpreter VS Code has selected
npm install && npm run compile
```

Press `F5` to launch the Extension Development Host, open a `.py` training script, and click
**▶ Run with ARC Lens** in the editor toolbar.

ARC Lens uses whichever interpreter the Python extension has selected for the file, which is the
venv your `torch` and `arc-training` are actually installed in. It runs without `arc-training`.
You lose the structural diagnostics in that case, and it tells you so rather than leaving a
blank chart.

**Try it on the reference script.** `python/train_demo.py` is a real 9-layer CNN on real
CIFAR-10 with a deliberately aggressive learning rate. No failure is injected. Whether it
diverges, and at which step, depends on the data order and the initialisation.

```bash
python python/runner.py python/train_demo.py
```

### Commands

| Command | What it does |
| :--- | :--- |
| **▶ Run with ARC Lens** | Monitor and recover the active script |
| **Run Baseline (interventions off)** | Same run, recovery suppressed. The A/B control arm |
| **Export Run Report** | Self-contained HTML incident report |
| **Open AI Failure Analyst** | Chat about the run, with its telemetry attached |
| **Generate ARC-Tested Script** | Generate a pre-instrumented training script |

---

## What it does

Every monitoring tool plots a loss curve. Three things here are not standard.

**It intervenes.** When a run diverges, ARC restores weights from a checkpoint, scales the
learning rate, and resumes, automatically and mid-run.

**It knows when to stop.** After three failed recoveries of the same kind it declares the run
unrecoverable and says so, instead of rolling back forever. At that point the useful answer is
"kill this run", not a fourth rollback.

**It is measured against a control arm, and the measurements have repeatedly gone against it.**
Four structural detection rules were built. Two were deleted and two were demoted to
report-only, each time because an A/B said so. The full record is under
[Evidence](#evidence).

That last point is why the interface, and not the automation, is the product. Where ARC can
detect a silent death but cannot safely fix one, what it produces is a person told in time, in
terms they can act on. In a diverging baseline run, ARC knew the run had failed at **3.81 s**.
The run continued to **50.96 s** because nothing stopped it, so **92.5% of that compute was
spent after the answer was already known**. Closing that gap is a legibility problem rather than
a detection one. That is what the status strip, the ARIA live region, the preflight and the
chart data tables are for, and it is why the accessibility work and the compute saving turned
out to be the same work.

---

## How it works

Three tiers:

1. **Extension host** (`src/`) resolves your interpreter, spawns the run, parses telemetry, and
   drives the dashboard and the LLM features.
2. **Instrumentation harness** (`python/_arc_bootstrap.py`) patches PyTorch to measure every
   weight update, owns a host-resident checkpoint store, and detects failures.
3. **Recovery agent** (`python/arc_agent.py`) is a deterministic rule engine that chooses and
   applies the response.

The measurement anchor is `Optimizer.step`, not `loss.backward()`. One recorded step is one
*weight update*, which is what makes gradient accumulation, AMP and multi-optimizer setups
(GANs) correct rather than merely non-crashing. Your source runs unmodified via `runpy`, so
tracebacks report the line numbers that are actually in your file.

---

## Telemetry

**Core**: loss, learning rate, gradient L2 norm, GPU memory.

**Structural** (needs `arc-training`): effective rank for representation collapse, gradient
entropy for whether gradients still carry information, weight update ratio (‖ΔW‖/‖W‖), and
gradient flow ratio between early and late layers, which needs at least 4 parameterised layers.

---

## Interventions

| Detected as | Trigger | Response |
| :--- | :--- | :--- |
| `numerical` | Loss non-finite or exploded past 1e6 | `rollback_and_reduce_lr`, **and** `enable_grad_clipping` if the gradient norm is also above 50 |
| `loss_plateau` | Loss stalled 300+ steps **and** never improved past 60% of its opening value | **Report only, no action** |
| `representation_collapse` | Effective rank below 50% of the run's own baseline | **Report only, no action** |

The first column is the `kind` shown in the action log and the status strip, so the two
report-only rules are identifiable there instead of appearing as unexplained anomalies. Both are
excluded from the recovery path in code (`REPORT_ONLY_KINDS` in `_arc_bootstrap.py`), not merely
left untriggered.

**Gradient explosion is not an independent trigger**, and an earlier version of this table
implied it was. `run_recovery_agent` has one call site, reached only once a `numerical` failure
is already being handled, meaning a non-finite or exploded *loss*. The `grad_norm > 50` test
then runs inside the agent and latches clipping on for every later update. A run whose gradients
spike while its loss stays finite is charted and scored but never clipped, because nothing
brings the agent in. Whether that is the right boundary is still open.

---

## Overhead

Measured rather than asserted, by running the same loop with and without the harness
(`python python/benchmark_overhead.py`). RTX 3050, 2.79M-parameter CNN, 200 steps × batch 128,
median of 3:

| Configuration | ms/step | Overhead |
| :--- | ---: | ---: |
| bare (no ARC) | 49.09 | |
| ARC core metrics only | 49.97 | **1.8%** |
| ARC full (advanced every 25 steps) | 53.20 | **8.4%** |
| ARC full (advanced every step) | 132.55 | 170.0% |

The last row is why expensive signals are sampled rather than collected every step.

---

## Evidence

`ARC_MODE=baseline` runs the identical instrumented code path with every intervention
suppressed, so a comparison against a normal run isolates the interventions and nothing else.
Both arms use the same seed and the same data order.

```bash
python python/experiment_ab.py --lrs 0.03 0.1 0.25 0.5 --epochs 10
```

The raw output of every sweep quoted below is committed under [`results/`](results/), so these
numbers can be checked against the files the scripts actually wrote. What the sweeps found, in
order:

**Two rules were deleted for firing on healthy runs.** One cost 1.74 and 0.78 points of
validation accuracy. The other took a run from 87.4% to chance.

*Weight update ratio* fired above an absolute ceiling. Across four learning rates its
distribution on a healthy run overlaps a failing one almost completely. The p90 values are
effectively identical (0.089 healthy against 0.088 damaged), the peaks barely separate (0.285
against 0.322), and the healthy run sustained a *longer* consecutive breach than the damaged
one, 31 samples against 26. It was a proxy for "the learning rate is large", not for "training
is failing".

*Gradient entropy* fired below 1% of an opening baseline. A healthy run and a dead one settle to
the same value (about 1.45e-05) from around step 70, so no threshold separates them. The
upstream estimator bins a heavy-tailed distribution linearly and saturates for every run.

Behind both cases: these signals change by orders of magnitude in a run's opening steps simply
because the model goes from random to structured. Structural checks therefore wait 200 steps
before capturing a baseline, use thresholds relative to that baseline, and must hold across
several consecutive samples.

**The rule that replaced them failed its own standard twice.** A real CIFAR-10 run at lr=0.5
finished at 10.00%, which is chance, with its loss pinned at ln(10), and ARC reported zero
failures across all 780 steps. That silent death is exactly what this tool exists to catch, and
every rule stayed quiet while the dashboard stayed green. The fix was a loss-plateau rule.

Replaying both arms, a healthy run's longest stall was 82 steps against the dead run's 764. A
longer A/B then caught a false positive immediately: over 3900 steps a run reaching 87.5%
tripped the rule twice. The counter keys off the best-ever batch loss, so as a run converges its
own record gets harder to beat and stalls grow without bound. No patience value fixes that. What
separates the two cases is whether the run ever got anywhere. A dead run stalls having never
improved (best/first = 0.888), a converged one stalls having improved enormously (0.271).

**Then the sweeps went against the rule's *response*.** Cutting the learning rate on a plateau
made things worse:

| epoch | baseline (no action) | active (3 × `reduce_lr` from step 316) |
| ---: | :--- | :--- |
| 1 | 10.00%, lr 4.91e-01 | 10.00%, lr 2.45e-01 |
| 5 | **26.73%**, lr 2.56e-01 | 10.00%, lr 3.20e-02 |
| 10 | **73.19%** | 10.00%, loss 2.3026 = ln(10) |

Both arms sat at chance for four epochs. The control arm escaped once cosine decay walked the LR
down by itself. The arm ARC "helped" never did. Large steps were the only thing that could carry
the weights out of the dead region, and cutting the LR removed them. The same pattern appeared
in `representation_collapse`.

**A third sweep then showed those deltas cannot be attributed to ARC at all.** With nothing
intervening in either arm, the same configuration still split **72.58% against 10.00%**, on
identical code, an identical seed and an identical data order, agreeing to four decimals for
four epochs before one escaped the dead region and the other never did. Escape at that learning
rate is a coin flip decided by floating-point nondeterminism. Across every `lr=0.5` arm run, 3
of 4 untouched runs escaped and 0 of 2 intervened ones did, which is consistent with the
responses hurting but nowhere near enough to establish it.

**So no structural rule acts any more.** The reason is stronger than the deltas that prompted
it: where an intervention would matter, the run-to-run spread is larger than any effect a single
A/B pair can measure, so no response can be validated that way. Acting on evidence that cannot
exist yet is the mistake this project keeps catching in itself. What still intervenes is
unambiguous divergence, meaning a non-finite or exploded loss. The structural signals are still
collected, charted and reported when they trip. They just no longer get to steer the run.

---

## Configuration

| Setting | Default | Description |
| :--- | :--- | :--- |
| `arcAgent.pythonPath` | `python3` | Overrides the Python extension's interpreter. Machine-scope, so a workspace cannot set it |
| `arcAgent.stepDelay` | `0` | Artificial per-step delay for demos. Non-zero is wasted GPU time |
| `arcAgent.gpuHourlyRate` | `0` | Your GPU's hourly cost. `0` estimates from the detected device |
| `arcAgent.openRouterKey` | `""` | API key for the AI features. Provider inferred from the prefix |
| `arcAgent.llmModel` | `google/gemini-2.5-flash:free` | Model for the AI features |
| `arcAgent.telemetryEvery` | `1` | Emit one metric event every N optimizer steps. Raising it cuts telemetry volume roughly proportionally, **72.2% lower measured at N=10**, at no wall-clock cost, in exchange for coarser chart resolution. Risk detection is unaffected, since loss history and risk are computed every step regardless |
| `arcAgent.maxCheckpointMB` | `512` | Host RAM ceiling for the rollback ring buffer. Oldest snapshots are pruned first |

Harness behaviour is tunable through environment variables such as `ARC_ADVANCED_EVERY`,
`ARC_CHECKPOINT_EVERY` and `ARC_MAX_ATTEMPTS`.

---

## Tests

```bash
npm test                       # TypeScript + dashboard suites
python tests/test_harness.py   # harness, detector, checkpointing, end-to-end
```

186 tests, of which 111 are TypeScript and dashboard and 75 are Python. Thirty of the Python
tests need PyTorch and skip without it, a handful of those additionally needing CUDA. The runner
reports the skips rather than counting them as passes.

Ten of the Python tests are end-to-end against real training loops. They assert that gradient
accumulation does not inflate the step count, that an LR intervention survives a scheduler
rewriting the learning rate every step, that baseline mode never intervenes, and that tracebacks
point at the user's own line numbers.

Several were written before the code they cover and found real bugs doing it, including one
where an optimizer was matched to a wrapper module rather than the submodule it actually
updates, which would have rolled back both halves of a GAN.

CI additionally fails the build on any secret-shaped literal in source.

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0).
