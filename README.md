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
Four structural detection rules have been built and two were deleted after an A/B showed each
one intervening on healthy runs: one cost 1.74 and 0.78 points of validation accuracy, the
other took a run from 87.4% to chance. Both were removed rather than retuned, because the
underlying statistics do not separate a healthy run from a failing one on real workloads.

The fourth was added the same way it would have been deleted: by measurement. A real CIFAR-10
run at lr=0.5 finished at 10.00% — chance — with its loss pinned at ln(10), and ARC reported
**zero failures across all 780 steps**. No NaN, no gradient spike, no rank collapse; every rule
was silent while the dashboard stayed green. That is the failure this tool exists to catch, and
it did not. The fix is a loss-plateau rule, held to the same evidence standard the deleted rules
failed — which it then failed twice itself, in both directions that matter.

Replaying both arms, a healthy run's longest stall was 82 steps against the dead run's 764. But
a longer A/B immediately caught a false positive: over 3900 steps a run reaching **87.5%**
tripped it twice, because the counter keys off the best-ever batch loss, so as a run converges
its own record gets harder to beat and stalls grow without bound. No patience value fixes that.
What separates the cases is whether the run ever got anywhere — a dead run stalls having never
improved (best/first = 0.888) while a converged one stalls having improved enormously (0.271).

Then the sweeps went against the rule's *response*, twice. At `lr=0.5` the control arm sat at
chance for several epochs and then escaped on its own — cosine decay lowered the LR and it
climbed past 70% — while the arms ARC cut, an order of magnitude further down, finished at
10.00% and 30.84%. The same pattern appeared in `representation_collapse`, the last structural
rule still allowed to act.

**And then a third sweep showed those deltas cannot be attributed to ARC.** With nothing
intervening in either arm, the same configuration still split **72.58% vs 10.00%** — identical
code, identical seed, identical data order, agreeing to four decimals for four epochs before one
escaped the dead region and the other never did. Escape at that learning rate is a coin flip
decided by floating-point nondeterminism. Across every `lr=0.5` arm run: 3 of 4 untouched runs
escaped, 0 of 2 intervened ones did — consistent with the responses hurting, nowhere near enough
to establish it.

**So no structural rule acts any more**, and the reason is stronger than the deltas that
prompted it: where an intervention would matter, the run-to-run spread is larger than any effect
a single A/B pair can measure, so no response can be validated that way at all. Acting on
evidence that cannot exist yet is the mistake this project keeps catching in itself. What still
intervenes is unambiguous divergence — a non-finite or exploded loss — which also latches
gradient clipping on when the gradient norm is high at that moment. Gradient explosion is not a
trigger in its own right; see the note under *Interventions*. The structural signals are still
collected, charted and reported when they trip — they are informative to a human reading a run —
they just no longer get to steer it.

Every sweep is logged, including the two killed mid-run and the withdrawn claims.

**Which is why the interface is the product, not the packaging.** If ARC can reliably detect a
silent death and cannot safely fix one, then what it produces is not a rescue — it is a person
being told, in time, in terms they can act on. Measured: in a diverging baseline run ARC knew the
run had failed at **3.81 s**, and the run continued to **50.96 s** because nothing stopped it.
**92.5 % of that compute was spent after the answer was already known.** Closing that gap is not
a detection problem; it is a question of whether the state of the run is legible fast enough to
act on. That is what the status strip, the ARIA live region, the preflight and the chart data
tables are for, and it is why the accessibility work and the compute saving are the same work.
The full argument and every measurement, including the two that came back negative,.

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

| Detected as | Trigger | Response |
| :--- | :--- | :--- |
| `numerical` | Loss non-finite or exploded past 1e6 | `rollback_and_reduce_lr`, **and** `enable_grad_clipping` if the gradient norm is also above 50 |
| `loss_plateau` | Loss stalled 300+ steps **and** never improved past 60% of its opening value | **Report only — no action** |
| `representation_collapse` | Effective rank below 50% of the run's own baseline | **Report only — no action** |

The first column is the `kind` you see in the action log and the status strip, so the two
report-only rules are identifiable there rather than appearing as unexplained anomalies. Both
are excluded from the recovery path in code (`REPORT_ONLY_KINDS` in `_arc_bootstrap.py`), not
merely left untriggered.

**Gradient explosion is not an independent trigger, and this table used to imply it was.**
`run_recovery_agent` has one call site, reached only once a `numerical` failure is already
being handled — a non-finite or exploded *loss*. The `grad_norm > 50` test then runs inside
the agent and latches clipping on for every later update. So a run whose gradients spike while
its loss stays finite is charted and scored but never clipped: nothing brings the agent in.
Whether that is the right boundary is open; what it is not is what the
earlier two-row table described.

**A plateau is reported and not acted on, because acting on it was measured to make things
worse.** It used to cut the learning rate. On the `lr=0.5` arm of the A/B that took a run
which recovered to **73.19%** on its own and left it at **10.00%** — chance — for all ten
epochs:

| epoch | baseline (no action) | active (3 × `reduce_lr` from step 316) |
| ---: | :--- | :--- |
| 1 | 10.00%, lr 4.91e-01 | 10.00%, lr 2.45e-01 |
| 5 | **26.73%**, lr 2.56e-01 | 10.00%, lr 3.20e-02 |
| 10 | **73.19%** | 10.00%, loss 2.3026 = ln(10) |

Both arms sat at chance for four epochs. The control arm escaped once cosine decay walked the
LR down by itself; the arm ARC "helped" never did. Large steps were the only thing that could
carry the weights out of the dead region, and cutting the LR removed them. The detection was
correct — the run really was dead — but no available response is known to help, so the rule
reports and stops there.

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
save the run —.

---

## Configuration

| Setting | Default | Description |
| :--- | :--- | :--- |
| `arcAgent.pythonPath` | `python3` | Overrides the Python extension's interpreter. Machine-scope: a workspace cannot set it |
| `arcAgent.stepDelay` | `0` | Artificial per-step delay for demos. Non-zero is wasted GPU time |
| `arcAgent.gpuHourlyRate` | `0` | Your GPU's hourly cost. `0` estimates from the detected device |
| `arcAgent.openRouterKey` | `""` | API key for the AI features. Provider inferred from the prefix |
| `arcAgent.llmModel` | `google/gemini-2.5-flash:free` | Model for the AI features |
| `arcAgent.telemetryEvery` | `1` | Emit one metric event every N optimizer steps. Raising it cuts telemetry volume roughly proportionally — **−72.2 % measured at N=10**, at no wall-clock cost — for coarser chart resolution. Risk detection is unaffected: loss history and risk are computed every step regardless |
| `arcAgent.maxCheckpointMB` | `512` | Host RAM ceiling for the rollback ring buffer. Oldest snapshots are pruned first |

Harness behaviour is tunable through environment variables (`ARC_ADVANCED_EVERY`,
`ARC_CHECKPOINT_EVERY`, `ARC_MAX_ATTEMPTS`, …).

---

## Tests

```bash
npm test                       # TypeScript + dashboard suites
python tests/test_harness.py   # harness, detector, checkpointing, end-to-end
```

186 tests — 111 TypeScript/dashboard, 75 Python. Thirty of the Python tests need PyTorch and
skip without it, and a handful of those additionally need CUDA; the runner reports the skips
rather than counting them as passes. The Python suite includes ten end-to-end tests
that run the real harness against real training loops, asserting that gradient accumulation does
not inflate the step count, that an LR intervention survives a scheduler rewriting the learning
rate every step, that baseline mode never intervenes, and that tracebacks point at the user's
own line numbers.

Several of these tests were written before the code they cover and found real bugs doing it —
including one where an optimizer was matched to a wrapper module rather than the submodule it
actually updates, which would have rolled back both halves of a GAN.

CI additionally fails the build on any secret-shaped literal in source.

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0).
