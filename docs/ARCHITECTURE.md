# ARC Lens — Architecture

Technical reference for the ARC Lens VS Code extension. This describes what the code actually
does. There is no longer a difference between the public repository and the packaged `.vsix`:
what is checked in is what ships.

**Companion documents**

| Document | Purpose |
| :--- | :--- |
| [`../README.md`](../README.md) | User-facing overview and install instructions |
| [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) | Audit findings and their remediation |
| [`FUTURE_IMPROVEMENTS.md`](FUTURE_IMPROVEMENTS.md) | Roadmap and status |
| [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) | Measured baseline-vs-active results |

---

## 1. What this is

ARC Lens watches a PyTorch training script while it runs, streams its optimisation telemetry
into a webview dashboard, and — when the run starts to fail — restores the model to a healthy
checkpoint, lowers the learning rate, or enables gradient clipping, without the user touching
their code.

It is the IDE frontend for [`arc-training`](https://pypi.org/project/arc-training/), which
supplies the structural signal collectors. ARC Lens contributes the instrumentation harness,
the transport, the checkpoint store, the recovery agent, the dashboard and the LLM
diagnostics layer.

---

## 2. Component map

```
arc-lens/
├── src/                          TypeScript — runs in the VS Code extension host
│   ├── extension.ts              Commands, process spawn, stdout parsing, webview wiring
│   └── pro/
│       ├── licenseManager.ts     Config reads and provider-key detection (no license gate)
│       ├── chatManager.ts        Streaming SSE client (OpenRouter/Groq/Anthropic/Gemini/OpenAI)
│       ├── contextBuilder.ts     Turns run telemetry into an LLM system prompt
│       ├── scriptGenerator.ts    Prompt construction + code-fence extraction
│       └── reportBuilder.ts      Self-contained HTML post-mortem report
├── python/                       Shipped as plain, readable .py files
│   ├── runner.py                 Entry point: installs hooks, runs the target via runpy
│   ├── _arc_bootstrap.py         The instrumentation engine
│   ├── arc_agent.py              Deterministic recovery rule engine
│   ├── train_demo.py             Real CIFAR-10 CNN used as the reference script
│   ├── benchmark_overhead.py     Wall-clock overhead benchmark
│   └── experiment_ab.py          Baseline-vs-active A/B harness
├── media/
│   ├── dashboard.html            Markup, CSS, ECharts setup, event handling
│   └── vendor/echarts.min.js     Vendored — the dashboard renders with no network
└── tests/                        124 tests across six suites
```

The Python sources ship as files. They were previously base64-encoded into `extension.js`,
written to `globalStorage` at runtime and executed from there — the textbook packer pattern,
and the thing a reviewer most needs to be able to read. See
[`SECURITY_AUDIT.md` §C-3](SECURITY_AUDIT.md).

---

## 3. Execution flow

```mermaid
sequenceDiagram
    participant User
    participant Ext as extension.ts<br/>(extension host)
    participant WV as dashboard.html<br/>(webview)
    participant Run as runner.py
    participant Boot as _arc_bootstrap
    participant Script as user's train.py
    participant Agent as arc_agent.py

    User->>Ext: "▶ Run with ARC Lens"
    Ext->>Ext: save file; ask ms-python.python for the interpreter
    Ext->>WV: createWebviewPanel, inject dashboard.html (nonce CSP)
    Ext->>Run: spawn(python, [runner.py, target.py], ARC_MODE=active)

    Run->>Boot: install()
    Note over Boot: patch Optimizer.__init__ (wrap each instance's step)<br/>patch Tensor.backward + GradScaler.scale (record loss only)
    Run->>Script: runpy.run_path(target)   — source unmodified

    loop every optimizer.step()
        Script->>Boot: wrapped step fires (before the update)
        Boot->>Boot: enforce_lr, foreach grad norm, one sync
        Boot->>Boot: sample arc collectors every N steps
        Boot-->>Ext: {"type":"metric"} on stdout
        Ext->>WV: postMessage (batched every 100 ms)
    end

    alt loss non-finite or exploded
        Boot->>Agent: run_recovery_agent(kind="numerical")
    else structural pathology sustained
        Boot-->>Ext: {"type":"failure_detected"} — reported, no agent call
    end

    Agent->>Boot: restore checkpoint / scale LR / enable clipping
    Agent-->>Ext: {"type":"thought"} / {"type":"intervention"}
    Ext->>WV: reasoning trace, markers, savings ledger

    Run-->>Ext: {"type":"run_summary"} then {"type":"status"}
```

---

## 4. Tier 1 — The extension host (`src/extension.ts`)

### Commands

| Command ID | Title | Notes |
| :--- | :--- | :--- |
| `arc-lens.run` | ▶ Run with ARC Lens | Active editor must be `.py` |
| `arc-lens.runBaseline` | Run Baseline (interventions off) | Control arm for the A/B |
| `arc-lens.stop` | ⏹ Stop ARC Lens | — |
| `arc-lens.exportReport` | Export Run Report | Self-contained HTML |
| `arc-lens.openChat` | Open AI Failure Analyst | Needs an API key |
| `arc-lens.generateScript` | Generate ARC-Tested Script | Needs an API key |

### Interpreter resolution

`resolveInterpreter()` prefers, in order: an explicitly configured `arcAgent.pythonPath`, the
environment `ms-python.python` has selected for the target file, then a PATH search over
`python3`/`python`/`py`.

The Python-extension path is not just a convenience. The selected interpreter is the venv the
user's `torch` and `arc-training` are installed into; a bare name off PATH resolves to the
system Python, which is the one environment where those dependencies are reliably missing.

`arcAgent.pythonPath` is `"scope": "machine"` — **not** `"machine-overridable"`, whose name
reads like a safe middle ground but which VS Code documents as workspace-settable. This value
is executed; a cloned repo's `.vscode/settings.json` must not be able to point it anywhere.

### Transport

Newline-delimited JSON on stdout. The host keeps a partial-line buffer, parses complete lines,
and batches events into one `postMessage` every 100 ms so a fast loop cannot saturate the
webview bridge. On close it drains the buffered tail and flushes synchronously before emitting
`done` — the last line is usually `run_summary`, which is exactly the one that used to be lost.

Any line that fails `JSON.parse` is forwarded as `{type:"log"}`, so `print()` in the user's
script still surfaces.

### Event schema

| `type` | Fields | Meaning |
| :--- | :--- | :--- |
| `environment` | `gpu`, `torch`, `cuda`, `arc`, `python`, `mode` | Emitted once at startup |
| `log` | `level`, `message` | Free-form line |
| `metric` | `step`, `epoch`, `loss`, `grad_norm`, `lr`, `gpu_mem_mb`, `optimizer`, `advanced` | One weight update |
| `risk` | `score` (0–1), `label` | Heuristic instability score |
| `failure_detected` | `step`, `kind`, `reason`, `loss`, `grad_norm` | A threshold was crossed |
| `thought` | `phase`, `message` | One agent step |
| `intervention` | `action`, `detail`, `step` | A recovery tool ran |
| `unrecoverable` | `step`, `kind`, `attempts`, `message` | Futility breaker tripped after repeated failed recoveries |
| `detection_silenced` | `step`, `kind`, `attempts`, `message` | A report-only rule has stopped repeating itself. Deliberately *not* `unrecoverable`: ARC never acted, so it cannot claim the run is past saving |
| `degraded` | `component`, `message` | Instrumentation is impaired |
| `checkpoint_budget` | `bytes_per_checkpoint`, `total_bytes`, `location` | Memory cost of checkpointing |
| `run_summary` | `steps`, `backward_calls`, `optimizer_steps`, `interventions`, `degraded` | Emitted at the end |
| `status` | `status`, `message` | `running` / `complete` / `error` / `stopped` |
| `batch` | `events[]` | Host-side envelope, unwrapped by the webview |

`loss` is `null` rather than `NaN`, and any non-finite advanced value is dropped rather than
clamped. JSON cannot represent `NaN` or `Infinity`, and a gap in a chart is honest where an
invented number is not.

### Telemetry accumulation

`metricHistory` is a `RingBuffer<MetricPoint>` capped at 10 000 — a fixed-slot overwrite, so
insertion stays O(1) on long runs. `contextBuilder.buildSystemPrompt()` reduces it to a compact
prompt: min/max/final loss, peak gradient norm, final LR, a 40-row evenly-sampled trace, and
the full intervention log.

---

## 5. Tier 2 — The instrumentation engine (`python/_arc_bootstrap.py`)

### No injection at all

`runner.py` imports `_arc_bootstrap`, calls `install()`, then executes the target with
`runpy.run_path(target, run_name="__main__")`. The user's source is compiled **unmodified**,
so a traceback reports the line that is actually in their file. The runner's own frames are
stripped from reported tracebacks.

### The measurement anchor is `Optimizer.step`

This is the central design decision.

* `self` is definitionally the optimizer being stepped — no guessing.
* `self.param_groups` are definitionally the parameters being updated, so the gradient norm
  covers the right tensors.
* One call is one *weight update*, which is what a training step means.

That last point is what makes gradient accumulation and multi-optimizer setups correct rather
than merely non-crashing. A 4×-accumulation loop emits one metric per update, not four.

**Concrete optimizers define their own `step`.** `Adam`, `SGD` and the rest never call
`Optimizer.step`, so patching the base class intercepts nothing. `Optimizer.__init__` is
patched instead and each *instance* has its `step` wrapped, which covers third-party and custom
optimizers too.

**The wrapper must be a bound method, not a plain function.** `LRScheduler.__init__` reaches
into `optimizer.step.__func__` to install its own step counter, and a plain function has no
`__func__` — so assigning one makes *constructing any scheduler* (`StepLR`,
`CosineAnnealingLR`, `OneCycleLR`, `ReduceLROnPlateau`, …) raise `AttributeError` before the
first batch. The wrapper is installed with `types.MethodType`, which keeps it
indistinguishable from the method it replaced. An earlier version of this document claimed
instance-wrapping was "the same technique PyTorch's own LR schedulers use"; that was the
inverse of the truth — it is the technique they reject. See [C-4](SECURITY_AUDIT.md).

**Each optimizer is instrumented once.** Wrapper optimizers (Lookahead, SAM) subclass
`Optimizer` and delegate to an inner one, so both objects pass through the patched `__init__`.
An `_arc_instrumented` instance flag keeps the hook on the outermost one only — the one whose
`step` is a user-level weight update. Without it a single update emitted two metrics and the
second hook, finding the loss already consumed, read it as a NaN failure on step 1. See
[H-5](SECURITY_AUDIT.md).

### Which loss belongs to which update

`Tensor.backward` and `GradScaler.scale` are patched, but only to record a loss tensor. Neither
emits.

**An absent loss is not a bad loss.** Those two call sites are the only writers, so the pending
loss is legitimately missing for `torch.autograd.backward(loss)` (the function form, not a
`Tensor` method), a non-scalar backward, and closure-driven optimizers such as `LBFGS` that run
`backward()` inside `step()`. Missing is reported as *unknown*: gradient, LR and memory
telemetry continue, loss-based detection is skipped for that step, the loss renders as a gap,
and a one-time `degraded` event names the cause. Treating it as a NaN instead — which the code
did until the second review pass — made ARC diagnose a failure on every step of a healthy run
and freeze the model while logging successful interventions. See [C-5](SECURITY_AUDIT.md).

Under AMP, `scaler.scale(loss)` receives the **unscaled** loss — the only place that value is
available without guessing at the scale factor. And `scaler.step(optimizer)` unscales gradients
before running the optimizer, so the norm read at the anchor needs no correction; an explicit
`get_scale()` division covers a raw `optimizer.step()` under a live scaler.

### Model resolution by parameter identity

`WeightRollback` and the arc collectors need a model, which `Optimizer.step` does not carry. A
frame walk produces *candidates*; the winner is the module whose parameters actually overlap
this optimizer's param set, ties broken toward the smaller module so a wrapper holding both a
generator and a discriminator does not outrank the specific submodule. Resolved once per
optimizer and cached. `arc_watch(model, optimizer)` is the explicit escape hatch.

### Per-step work

1. `enforce_lr()` — re-assert any ARC learning-rate reduction on top of what the user's
   scheduler just wrote (see below).
2. Gradient norm via `torch._foreach_norm` — one fused kernel. Stacked with the loss and read
   with a single `.tolist()`, so a step costs **one** device sync.
3. Health test: `isnan or isinf or |loss| > 1e6`.
4. Checkpoint every `ARC_CHECKPOINT_EVERY` steps while healthy.
5. Advanced signals from the arc collectors, sampled every `ARC_ADVANCED_EVERY` steps and
   densified while risk is elevated.
6. Risk score, then `risk` and `metric` events.
7. Gradient clipping, if a previous intervention latched it on.
8. Failure handling — numerical, then structural.

### The learning-rate guard

An LR intervention that only divides `group['lr']` is erased on the next iteration by any
scheduler, because schedulers recompute the LR from their own base every step — while the log
still reports the intervention as successful.

Rather than trying to discover and rewrite every scheduler's internal base (they differ, and
third-party ones are unbounded), ARC treats its reduction as a multiplier it re-asserts each
step. A group whose LR still holds the value ARC last wrote was not touched from outside and is
left alone, so a run with no scheduler never compounds.

### The checkpoint store

`CheckpointStore` keeps snapshots in **host** memory. Deep-copying `state_dict()` in place
leaves every checkpoint on the GPU — roughly `max_checkpoints × 3 × model size` with Adam,
which is how a tool that exists to warn about OOM causes one. It estimates its own footprint
before the first save and emits `checkpoint_budget`.

RNG state (`torch`, `cuda`, `numpy`) travels with each snapshot, so a rollback resumes the same
data order and the same dropout masks. That is also what makes the A/B comparison fair.

Owning this rather than reaching into `WeightRollback`'s internals means an `arc-training`
rename cannot silently disable rollback.

### Silent-failure detection

A run can be completely dead while its loss stays finite and its gradient norm stays small —
the loss simply sits at `ln(num_classes)` forever. Nothing in the numerical path sees that,
which is the entire reason the structural signals are worth collecting.

`check_structural()` runs on every advanced sample, independent of loss:

| Trigger | Condition | Response |
| :--- | :--- | :--- |
| `loss_plateau` | stalled 300+ steps **and** `best/first loss > 0.60` | **Report only — no action** |
| `representation_collapse` | `effective_rank < 50%` of baseline | **Report only — no action** |

That is the whole table, and **neither remaining row is allowed to act.** It has had four rows
over its life: two were deleted after measurement showed them harming healthy runs, one was
added after measurement showed a real failure getting past every row that remained, and both
survivors were then demoted to report-only after measurement showed their responses harming
failing runs.

That is not a gap in the product. Four independent attempts to let a structural signal steer a
run all made things worse, which is itself the finding — see §4 and §5 of
[`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md).

**`loss_plateau` is the rule that catches a silent death.** A CIFAR-10 run at `lr=0.5` finished
at chance accuracy with its loss pinned at `ln(10)`, and every other row above stayed silent for
all 780 steps — the loss was finite, the gradient norm was 0.07, and the rank never fell far
enough. Replaying both arms, a healthy run's longest stall was 82 steps against the dead run's
764, and patience sits at 300.

**Patience alone is not sufficient, and a longer A/B proved it.** Over 3900 steps a run reaching
87.5% tripped the rule twice. The counter keys off the *best-ever* batch loss, so as a run
converges its own record gets harder to beat and stalls grow without bound — convergence *is* a
plateau. No patience value separates them, because the stall length on a successful run is
unbounded.

The second condition is what makes the rule sound: it fires only when `best_loss / first_loss`
is above 0.60, meaning the run stalled *and* never improved more than 40% from where it started.
Measured, a healthy run reaches 0.271 and a dead one 0.888. That test needs no knowledge of the
class count, or even that the task is classification.

**It reports and takes no action, because the action it used to take was measured to destroy
the run.** It cut the learning rate. On the `lr=0.5` arm that turned a run which recovered to
73.19% by itself into one that finished at chance:

| epoch | baseline (no action) | active (3 × `reduce_lr` from step 316) |
| ---: | :--- | :--- |
| 1 | 10.00%, lr 4.91e-01 | 10.00%, lr 2.45e-01 |
| 4 | 9.76%, lr 3.34e-01 | 10.00%, lr 4.18e-02 |
| 5 | **26.73%**, lr 2.56e-01 | 10.00%, lr 3.20e-02 |
| 10 | **73.19%** | 10.00%, loss 2.3026 |

Both arms were pinned at chance for four epochs — the detection was right. But the control arm
escaped once cosine decay walked the LR down on its own, and the intervened arm never did:
large steps were the only thing that could carry the weights out of the dead region, and
cutting the LR removed them. A −63.19pp delta in ARC's disfavour, from a *correct* detection.

Rolling back is no better. Confirming a plateau takes 300 stalled steps, so by the time the
verdict lands every checkpoint in the ring is already post-collapse; restoring one returns the
model to the state it is in and spends a recovery attempt doing it. With no response known to
help, the rule reports and stops. Acting again requires a measured trajectory showing some
action actually rescues a plateaued run — the bar every other row here had to clear.

Report-only is enforced in `_handle_failure` above the cooldown and baseline gates, so both A/B
arms execute identical code for this kind, and above the `optimizer.zero_grad()` at the end of
that function — which would otherwise discard the user's update and make "no action" false.

**`representation_collapse` fired for the first time, and the arm that acted on it lost 44 points.** For a long time it
never fired at all: the threshold sits far from anything a working run reaches — a healthy
model's rank bottoms at 97.2% of its step-1 baseline — and further than the *dead* run we had
measured, which bottoms at 87.4%. `mean_effective_rank` is the SVD entropy of the weight
matrices, and a network can emit a constant output while every weight matrix stays
well-conditioned, so it measures weight conditioning rather than representational rank.

Then the sweep confirming the plateau fix produced a run where it did fire, in both arms, and
the arm it was allowed to act on was wrecked:

| epoch | baseline (no action) | active (3 × `rollback_and_reduce_lr`) |
| ---: | :--- | :--- |
| 3 | 10.00%, lr 4.04e-01 | 10.00%, lr 4.04e-01 |
| 4 | **19.35%**, lr 3.34e-01 | 10.00%, lr 3.34e-01 |
| 5 | 28.32%, lr 2.56e-01 | 21.44%, lr **3.20e-02** |
| 10 | **75.18%** | **30.84%** |

−44.34pp, apparently by the same mechanism as the plateau. A later sweep withdrew that
attribution: with nothing intervening in either arm the same configuration split by 62.58
points, so escape here is bistable and a single pair cannot tell a bad response from a bad coin
flip. What stands regardless is that no measurement shows the response helping, and the rollback
cannot in principle — the checkpoints in the ring are from inside the collapsed region, which is
where the model already is. See [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) §4c.

It is now report-only too. **No structural rule is allowed to act.** Every rule that was given
the power either fired on healthy runs or damaged failing ones; what still acts is numerical
divergence and gradient clipping, both verified working.

**The baseline used to be captured after the collapse.** Structural baselines were taken from
the first samples after the 200-step warmup, so a run that died earlier had its reference
measured on the corpse — the dead arm scored 99.72% of its own baseline against the healthy
arm's 98.69%, ranking the corpse as the more stable of the two. The baseline is now captured
from the opening samples while the verdict still waits for the warmup.

**`gradient_entropy_collapse` was removed after it destroyed a run.** It fired below 1% of an
opening baseline. On a CIFAR-10 run at `lr=0.25` it fired at step 125 and three rollbacks took
the model from the control arm's 87.43% to 10.00% — chance — inside the first epoch, after
which ARC correctly reported the run unrecoverable, having made it so. Measuring the trajectory
afterwards showed no threshold could have worked: at step 70 the healthy run and a genuinely
dead one both read 1.44e-05, and they stay together for the rest of training. The cause is
upstream — `GradientCollector._compute_entropy` bins a heavy-tailed distribution with
`torch.histc` on a linear scale, so outliers set the range, nearly all mass lands in one bin,
and the normalised entropy saturates near zero for any run. It measures outlier spread, not
information content. The signal is still collected and charted; it cannot trigger.
([C-7](SECURITY_AUDIT.md#second-review-pass))

**`weight_update_ratio` was removed before it.** One rule existed, firing above an absolute
0.05; measured across four learning rates its distributions overlap almost completely between
healthy and dead runs — a healthy run peaks higher and sustains a longer breach than a damaged
one — so it was a proxy for "the learning rate is large", not for "training is failing". It
cost 1.74 and 0.78 points of validation accuracy in back-to-back A/B runs that needed no help.
The ratio is still collected and charted for the human reading the run; it does not act.

Two rules removed for the same underlying reason is the useful generalisation here: in both
cases a signal's natural early-training trajectory resembled the pathology it was meant to
detect. A new row in this table needs a measured trajectory showing separation on both a
healthy and a failing run, not a plausible story about what the signal means.

Three properties matter:

**Nothing is judged until the run is past its opening transient.** `check_structural()` returns
immediately while `STATE.step < STRUCTURAL_WARMUP_STEPS` (default 200, `ARC_STRUCTURAL_WARMUP`),
and only then begins capturing a baseline. Every structural signal moves by orders of magnitude
in the first few dozen steps, on healthy and failing runs alike, simply because the model goes
from random to structured. A baseline taken inside that transient makes normal early learning
look like collapse — which is how the entropy rule came to fire at step 125 on a healthy run.

**Thresholds are relative to the run's own baseline**, captured from the samples after that
warmup. A fixed `effective_rank < 3.0` was tuned on a small MLP and is meaningless on a CNN
whose layers are 256 wide — the real value there is around 70.

**Each rule must hold for several consecutive samples.** One bad sample is noise.

### The futility circuit-breaker

After `ARC_MAX_ATTEMPTS` (default 3) failed recoveries of the same kind, ARC emits
`unrecoverable` and stops intervening. Once a network has genuinely collapsed, every checkpoint
still in the ring is collapsed too, so another rollback produces a slower dead run. The user's
real decision at that point is whether to keep paying for the GPU, and saying so is worth more
than a fourth rollback.

### Declining to act means touching nothing

Three branches of the failure handler decline to intervene: unrecoverable, cooldown, and
baseline mode. All three now `return` without touching gradients, because **dropping an update
is an intervention** — zeroing gradients immediately before the caller's `optimizer.step()`
makes that step a no-op, and on a diverging run discarding the bad update is most of what a
rollback achieves. All three used to call `optimizer.zero_grad()` on the way out, which made
each of their log lines false: "the run continues untouched" froze it, the cooldown suppressed
up to 15 healthy updates, and baseline mode performed the intervention it announced it was
suppressing. `zero_grad` now appears exactly once in the failure path, after a recovery has
actually been applied. See [C-6](SECURITY_AUDIT.md).

### Modes

`ARC_MODE=baseline` suppresses every intervention while leaving telemetry, detection and
reporting fully active. Both arms run identical instrumented code, so a comparison between them
isolates the interventions — which only holds because the baseline branch does not modify the
run in any way, gradients included (C-6).

---

## 6. Tier 3 — The recovery agent (`python/arc_agent.py`)

`run_recovery_agent()` is a **deterministic rule engine presented in ReAct form**. It emits
`thought` events shaped like an agent's reasoning but selects tools by threshold, not by
inference, and makes no network call.

That is deliberate, and the docstring says so. The reflex path of a reliability tool must not
depend on a remote service being reachable, and a NaN needs handling in microseconds. The LLM
sits in the *analysis* path — the Failure Analyst panel — where latency is acceptable.

**Tools**

| Tool | Trigger | Effect |
| :--- | :--- | :--- |
| `get_training_snapshot` | always, first | Builds the state the rules read |
| `rollback_and_reduce_lr` | non-finite **or** `|loss| > 1e6` | Restore checkpoint, LR × 0.2 |
| `enable_grad_clipping` | `grad_norm > 50`, once — **only while a `numerical` failure is already being handled**, since that is the sole path into the agent | Clip at 1.0 on every later update |
| `reduce_learning_rate` | — | LR × 0.5. **Unreachable:** no structural rule reaches the agent any more (see §*Report only*), so nothing triggers this tool. Kept as a cross-file safety net, not as live behaviour |

Two changes from the original design are worth noting.

**Loss explosion now triggers rollback, not just NaN.** A loss of 4e15 is no more recoverable
than a NaN — the weights that produced it are already destroyed. Gating rollback on NaN alone
left an exploded-but-finite run to keep diverging while the log reported successful
interventions.

**Gradient clipping is a real intervention, not advice.** The old backward-anchored design
could only recommend it, because it did not own the caller's `optimizer.step()`. The optimizer
anchor runs immediately before the update, so ARC applies the clip itself. It latches once
rather than re-firing every step.

---

## 7. Tier 4 — The dashboard (`media/dashboard.html`)

One self-contained file: markup, CSS custom properties for both themes, four ECharts canvases,
and the message handler. ECharts is vendored locally (`media/vendor/echarts.min.js`, ~1 MB) and
served through `webview.asWebviewUri`, so the dashboard renders with no network at all.

> Apache ECharts is Apache-2.0, which is compatible with this project's AGPL-3.0 and permits
> redistribution provided the licence notice travels with the file. The upstream header is
> preserved verbatim at the top of the vendored copy; it is unmodified.

**Charts**

| Canvas | Series |
| :--- | :--- |
| Vitals | loss, learning rate, plus the baseline loss curve when an A/B run exists |
| Dynamics | gradient L2 norm, gradient entropy |
| Structural | effective rank, weight update ratio |
| Flow | weight norm, gradient flow ratio |

**Rendering strategy.** Incoming metrics only push into arrays and set `chartsNeedUpdate`; a
single `setInterval(…, 100)` performs the redraw. This decouples render cost from event rate.

**Provenance.** Each failure draws a labelled red `markLine` naming its kind, each intervention
a green one naming the action, and `markAreaFor()` shades the span between a failure and the
intervention that resolved it — so the time spent in a bad state reads as an interval.

**Security.** `script-src` is nonce-only, with no `'unsafe-inline'`. A nonce cannot authorise
inline event-handler attributes, so all 22 `onclick=` handlers were converted to `data-act`
attributes behind one delegated listener.

**Honesty.** There is no synthetic fallback anywhere. A metric with no real source pushes
`null` and renders as a gap, and the two structural charts show an empty state naming the
missing package. The compute-savings figure always displays the source of its hourly rate
next to it. Tests assert `Math.random()` never reappears in this file.

---

## 8. The LLM layer

### Provider routing (`chatManager.ts`)

The provider is inferred from the API key's shape, so one setting works across five vendors:

| Key prefix | Host |
| :--- | :--- |
| `sk-or-` | `openrouter.ai` |
| `gsk_` | `api.groq.com` |
| `sk-ant-` | `api.anthropic.com` |
| `AIzaSy` | `generativelanguage.googleapis.com` |
| `sk-` (other) | `api.openai.com` |

Streaming is raw `https.request` with manual SSE parsing — no SDK, so the extension ships with
zero runtime dependencies. Anthropic gets a distinct body shape because it takes `system` as a
top-level field.

### Report builder (`reportBuilder.ts`)

Renders a finished run as one self-contained HTML incident report: verdict, summary tiles, a
log-scaled loss chart as hand-drawn inline SVG with failure and intervention markers, the event
timeline, and the environment block. Log scale because a curve that survives a divergence spans
many orders of magnitude. No external references, no `<script>` tags, all backend strings
escaped — each asserted by tests, because the artifact is meant to still open years later.

### Licensing

There is none, and deliberately no code that resembles one. See
[`SECURITY_AUDIT.md` §M-10](SECURITY_AUDIT.md).

---

## 9. Configuration

| Setting | Type | Default | Notes |
| :--- | :--- | :--- | :--- |
| `arcAgent.pythonPath` | string | `"python3"` | Executed. `scope: machine` — not workspace-settable |
| `arcAgent.stepDelay` | number | `0` | Artificial pacing only; non-zero is wasted GPU time |
| `arcAgent.gpuHourlyRate` | number | `0` | `0` estimates from the detected GPU |
| `arcAgent.openRouterKey` | string | `""` | Any supported provider's key; `scope: machine` |
| `arcAgent.llmModel` | string | `google/gemini-2.5-flash:free` | Overridden if incompatible with the detected provider |

Harness environment variables (set by the extension, or by hand when running `runner.py`
directly): `ARC_MODE`, `ARC_ADVANCED_EVERY`, `ARC_CHECKPOINT_EVERY`, `ARC_MAX_CHECKPOINTS`,
`ARC_COOLDOWN_STEPS`, `ARC_MAX_ATTEMPTS`, `ARC_RANK_COLLAPSE`, `ARC_STRUCTURAL_SUSTAIN`,
`ARC_STRUCTURAL_WARMUP`, `ARC_BASELINE_SAMPLES`, `ARC_LOSS_EXPLOSION`.
`ARC_UPDATE_RATIO_MAX` and `ARC_ENTROPY_COLLAPSE` are gone with the rules they tuned.

---

## 10. Adding a metric end to end

1. **Collect** — read it in `OptimizerMonitor.collect_advanced()` in `_arc_bootstrap.py` and
   add it to the `advanced` dict. Pass it through `_finite()`.
2. **Type** — add the field to `MetricPoint.advanced` in `src/pro/contextBuilder.ts`.
3. **Prompt** — add it to the trace row builder in `buildSystemPrompt()` if the LLM should
   reason over it.
4. **Chart** — in `media/dashboard.html`, add a series array, register a dataset on the
   relevant chart, and push in the `case 'metric':` branch. Push `null` when absent.
5. **Detect** — if it can indicate a pathology, add a rule to `check_structural()` with a
   baseline-relative threshold and a sustain requirement, and a response in
   `arc_agent.STRUCTURAL_RESPONSES`.
6. **Do not** add a synthetic fallback. A metric with no real source renders as a gap.
