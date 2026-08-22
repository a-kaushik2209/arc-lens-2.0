# ARC Lens — Future Improvements

A prioritised roadmap. Tiers are ordered by return on effort, not by ambition.
Cross-references to [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) use its finding IDs.

**Guiding constraint:** this product's claim is *"we save your training run."* Every item
below is judged against whether it makes that claim more **verifiable**. Features that make
the dashboard prettier without making the claim more provable are ranked lower than they
feel like they should be.

> **Status, 2026-08-22.** Tiers 0, 1 and 2 are complete. Tier 3 is deliberately open, with
> reasons given at the end. Two things about how that went are worth recording, because the
> plan did not predict either:
>
> * **The most valuable item was not on this list.** Making instrumentation failures visible
>   (2.5) revealed that the structural rules — the capability this whole product is
>   differentiated on — were unreachable in practice. See *The item that was not on the list*.
> * **Three bugs turned out to live in `arc-training`, not here.** Two of them were being
>   hidden by a bare `except: pass`, and one of those was crashing any model built with
>   `inplace=True` activations. The fixes are upstream.
>
> Claims that used to be assertions are now measurements: overhead is benchmarked
> (`docs/benchmark_overhead.json`) and the recovery claim is A/B tested
> ([`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md)), including the cases where ARC does not
> help.

---

## Tier 0 — Do these first

These are small, they are almost all deletions, and each removes something that actively
damages the project's credibility.

### 0.1 Delete `enrichEvent()` — replace fabricated metrics with honest gaps
**Effort: 30 min · Impact: decisive**

See [C-2](SECURITY_AUDIT.md). Four charts currently show `Math.random()` output styled
identically to real measurements. Nothing else on this list matters if a reviewer finds this.

The replacement is not "show nothing" — it is a deliberate empty state:

```
┌─ Structural Diagnostics ──────────────────────┐
│                                               │
│   Effective Rank · Weight Update Ratio        │
│                                               │
│   Requires arc-training                       │
│   $ pip install arc-training      [ Copy ]    │
│                                               │
└───────────────────────────────────────────────┘
```

This converts the project's biggest liability into an install prompt for its own package.
The empty state also does honest work in the demo: install `arc-training` live, re-run, and
the charts fill in. That contrast *is* a feature demonstration.

### 0.2 Vendor the chart libraries
**Effort: 1 h · Impact: high — Status: done, via a bigger fix than planned**

The plan was to vendor Chart.js + Hammer.js + `chartjs-plugin-zoom` as three pinned local
files. A live reproduction during the audit found that exact three-library combination
intermittently throws a JS error on a specific data transition — a version-mismatch race,
not something re-pinning alone reliably closes. Rebuilt the whole chart layer on
**ECharts**, vendored as one dependency-free file (`media/vendor/echarts.min.js`), with
`dataZoom` (pan/zoom) built in rather than a separate plugin. Served via
`asWebviewUri` + `webview.cspSource` (see [H-2](SECURITY_AUDIT.md) for the verification —
full event sequence, zero console errors, in a real browser). No CDN dependency remains
anywhere in the dashboard.

### 0.3 Resolve the interpreter through the Python extension
**Effort: 1 h · Impact: high — Status: done**

See [M-9](SECURITY_AUDIT.md) and [H-1](SECURITY_AUDIT.md). `resolveInterpreter()` in
`extension.ts` now asks `ms-python.python` for the environment selected for the target file,
supporting both the current `environments.getActiveEnvironmentPath` /
`resolveEnvironment` API and the legacy `settings.getExecutionDetails` shape. An explicitly
configured `arcAgent.pythonPath` still wins (the user said so); the PATH name-guessing chain
survives only as the last fallback.

This matters more than "python3 doesn't exist on Ubuntu": the interpreter the user selected
is the venv their `torch` and `arc-training` are actually installed into, and a bare name off
PATH finds the system Python — the one environment where the dependencies are guaranteed
missing.

### 0.4 Strip secrets, delete `context.md`, delete the patch scripts
**Effort: 30 min · Impact: high — Status: done in-repo; one item needs the account owner**

See [C-1](SECURITY_AUDIT.md), [C-3](SECURITY_AUDIT.md), [L-8](SECURITY_AUDIT.md).
`context.md` and the thirteen `scripts/patch_*.py` files are gone, source maps are excluded,
and the hardcoded JWT that the dashboard's "Go Pro" button wrote into global settings has
been deleted along with the dead `validateLicense`/`getLicenseStatus` stubs. A CI job
(`.github/workflows/ci.yml`) now fails the build on any secret-shaped literal or embedded
JWT, so this cannot regress silently.

**Still open, and not fixable from this repository:** the four published `.vsix` files
(0.1.0–0.1.3) contain the old signing secret in their bytes. Revoking the token at its
issuer and superseding those releases requires the marketplace account owner.

---

## Tier 1 — The differentiators

Every monitoring tool has charts. TensorBoard, W&B, Neptune, Aim — all of them plot loss
better than this does, and they have years of polish. Competing on charts is a losing
position.

What none of them do is **intervene**. These four items are built entirely on that gap.

### 1.1 The compute-savings ledger ⭐ *highest-impact feature on this list*
**Effort: 4–6 h · Impact: decisive — Status: shipped, in a smaller form than specced**

Live in the dashboard as a banner that appears on the first intervention: elapsed wall-clock
time since run start, dollars "not re-spent," the failure step, and the intervention detail
— e.g. *"Failure @ step 20 → rollback_and_reduce_lr: ... Assumes $2.50/hr GPU — without ARC:
full restart from step 0."*

**The rate is now derived from the GPU the run actually reported**, closing the gap the
earlier pass left open. The harness emits an `environment` event carrying
`torch.cuda.get_device_name(0)`; the dashboard matches it against a small rate table
(H100 / A100 / L40 / A10G / V100 / T4 / consumer) and falls back to a labelled generic
estimate for anything unrecognised. `arcAgent.gpuHourlyRate` overrides everything.

The figure is never shown bare: the source string beside it always says where the number came
from — *"your configured rate"*, *"A100 on-demand list"*, or *"not in rate table, generic
estimate"*. An unlabelled dollar amount would be the same class of problem as an unlabelled
metric, which is the mistake [C-2](SECURITY_AUDIT.md) was about. Tests assert the source
string is never empty for any input, including `null`.

The funding proposal already argues in dollars: *"a NaN at hour 47 of a 48-hour run on a
$3/hr GPU."* The dashboard never makes that argument. It should compute it live.

At each intervention, ARC knows the wall-clock elapsed time and the step it rolled back to.
That is enough to state what was preserved:

```
┌─ Run Recovered ───────────────────────────────┐
│                                               │
│   4h 12m  of training preserved               │
│   $12.60  of A100 time not re-spent           │
│                                               │
│   NaN at step 4,821 → rolled back to 4,810    │
│   LR 3.00e-4 → 6.00e-5 · resumed in 0.4s      │
│   Without ARC: restart from step 0            │
│                                               │
└───────────────────────────────────────────────┘
```

Implementation is small: record `runStartTime`, read the GPU name from
`torch.cuda.get_device_name()`, map it against a small hourly-rate table (user-overridable
via a setting), and multiply. Ship the rate table as a config default so nobody has to
believe your numbers.

This converts an abstract technical claim into a number a non-ML judge understands instantly.
It is the single highest-leverage item here.

### 1.2 Intervention provenance on the charts
**Effort: 3 h · Impact: high — Status: shipped**

Every chart now carries a vertical red `markLine` at each `failure_detected` step and a
vertical green `markLine` at each `intervention` step (ECharts' built-in `markLine` API,
rebuilt from the `failures`/`interventions` arrays on every metric update — no separate
annotation plugin needed, unlike the Chart.js-era plan below). Verified live: both markers
render at the correct step and with the correct color.

Both remaining pieces are now done too: `markAreaFor()` shades the span between each failure
and the intervention that resolved it, so the time spent in a bad state reads as an interval
rather than two unrelated vertical lines, and each marker carries an inline rotated label —
the failure kind (`NaN`, `rank collapse`) on the failure line and the action
(`rollback + LR`, `LR down`, `clip`) on the intervention line.

The labels still matter, though less than when this was written: two of the three structural
rules that could produce a distinct marker have since been deleted for firing on healthy runs,
so in practice a marker is almost always `numerical`.

~~Right now the action log says a rollback happened, and the loss chart shows a curve.
Nothing connects them.~~ The inflection point in the loss curve now visibly lines up with
the intervention marker without narration — this is the demo moment 1.1 and 1.2 together
were meant to create.

### 1.3 Side-by-side A/B — the proof, not the claim
**Effort: 8–12 h · Impact: decisive — Status: shipped**

`ARC_MODE=baseline` suppresses every intervention while leaving telemetry, detection and
reporting fully active, so the control arm runs the *same* instrumented code path as the
treatment arm and even instrumentation effects cancel. Exposed three ways:

* **Command** — *ARC Lens: Run Baseline (interventions off)*. The dashboard keeps the
  baseline loss curve and overlays it, dotted, on the next active run.
* **Harness** — `python python/experiment_ab.py --lrs 0.03 0.1 0.25 0.5 --epochs 10`
  runs both arms at each learning rate and writes `docs/experiment_ab.json`.
* **Test** — `test_baseline_mode_reports_but_never_intervenes` asserts the control arm
  detects failures and applies exactly zero interventions, so the two arms cannot silently
  converge.

Measured results are in [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) — including the
configurations where ARC detects the failure and *cannot* save the run, which are reported
next to the ones where it can.

~~Currently a judge has to take it on faith that the run would have died without ARC.~~ The
original spec below is kept for the reasoning.

```
        Loss
   3.0 ┤        ╭──╴ ✕  baseline (NaN @ 4821)
   2.0 ┤     ╭──╯
   1.0 ┤ ╭───╯╰──────────────────  with ARC
   0.0 ┼─────────────────────────
       0    2k    4k    6k    8k
```

The mechanism already exists: `arc-training`'s `WeightRollback` saves full RNG state
(`torch`, `torch.cuda`, `numpy`) with every checkpoint. Determinism across the two runs is
therefore achievable, which is the hard part of an honest A/B — and it is already done.

A "monitor-only" flag that emits telemetry while suppressing interventions is a ~20-line
change to `_arc_on_loss`. The comparison view is the bulk of the work.

**This is the feature that wins arguments.** It converts *"our tool recovers training runs"*
from a claim into a reproducible demonstration a skeptical judge can run themselves.

### 1.4 Exportable post-mortem report
**Effort: 4 h · Impact: high — Status: shipped**

**Export Report** in the dashboard header (and the *ARC Lens: Export Run Report* command)
writes one self-contained HTML file via `src/pro/reportBuilder.ts`: verdict, summary tiles,
a log-scaled loss chart as hand-drawn inline SVG with failure/intervention markers, the full
event timeline, and the environment block (GPU, torch, CUDA, arc-training, Python).

Log scale rather than linear because a loss curve that survives a divergence spans many
orders of magnitude — on a linear axis a single 1e15 point flattens everything that matters.
Zero external references and zero `<script>` tags, both asserted by tests, so the artifact
opens years later on a machine with no network and no extension. Backend strings are escaped,
also asserted, since intervention details reach the report as text.

Three reasons this outperforms its cost:

1. The artifact outlives the demo. A judge scoring twenty projects later opens a file, not a
   memory.
2. It is the natural shape of an *incident report* — the framing ML infrastructure teams
   already use, and the one that makes this feel like infrastructure rather than a toy.
3. It is a shareable object with your name on it. That is organic distribution.

Reuse `contextBuilder.buildSystemPrompt()`'s summarisation logic, which already reduces a run
to its essentials.

---

## Tier 2 — Make the engine trustworthy

Tier 1 sells the product. Tier 2 is what survives someone actually adopting it.

### 2.1 Replace the frame walk with an explicit handle
**Effort: 6 h · Impact: high — Status: done, B as the automatic path and A as the escape hatch**

See [M-1](SECURITY_AUDIT.md). Implemented in `python/_arc_bootstrap.py`.

**The anchor moved to `Optimizer.step`.** One call is one weight update, `self` is
definitionally the right optimizer and `self.param_groups` the right parameters.

One wrinkle worth recording: patching `torch.optim.Optimizer.step` does **not** work, because
`Adam`, `SGD` and every other concrete optimizer define their own `step` and never call the
base implementation. `Optimizer.__init__` is patched instead and each *instance* gets its
`step` wrapped, which covers third-party and custom optimizers too.

**And that fix shipped a Critical defect of its own, caught only by a second review pass.** The
wrapper was assigned as a plain function. `LRScheduler.__init__` reads `optimizer.step.__func__`
to install its step counter, and a plain function has no `__func__` — so *constructing any
scheduler* (`StepLR`, `CosineAnnealingLR`, `OneCycleLR`, `ReduceLROnPlateau`, …) raised
`AttributeError` before the first batch, on the majority of real training scripts. It is now
bound with `types.MethodType`. Two things about this are worth recording rather than quietly
correcting:

* This document and `ARCHITECTURE.md` both justified the instance-wrapping as "the same
  technique PyTorch's own LR schedulers use". That sentence was the exact inverse of the truth:
  it is the technique the schedulers *reject*. It read as a reassuring detail and was never
  checked against `LRScheduler.__init__`.
* Nothing in the project could have caught it. `train_demo.py` sets `group["lr"]` by hand and
  constructs no scheduler, and the benchmark and A/B harness both run that script; the two
  scheduler tests simulate one by writing `group['lr']` directly. The demo, the benchmark, the
  experiment and the tests all shared one blind spot, so the anchor change looked fully
  validated while being broken for most users. **The highest-value remaining work on this item
  is a fixture that constructs a real `torch.optim.lr_scheduler` object** — see
  [C-4](SECURITY_AUDIT.md).

A second defect from the same fix: wrapper optimizers (Lookahead, SAM) subclass `Optimizer` and
delegate to an inner one, so both ran through the patched `__init__` and both got wrapped —
double-counted steps and a false NaN failure on step 1. An `_arc_instrumented` instance flag
now keeps only the outermost one ([H-5](SECURITY_AUDIT.md)).

**`backward` and `GradScaler.scale` still get patched, but only to record which loss belongs
to the pending update.** Neither emits. `scaler.scale(loss)` is the one place the *unscaled*
loss is available without guessing at the scale factor, and `scaler.step(optimizer)` unscales
gradients before the optimizer runs — so anchoring on the optimizer gets correct AMP numbers
almost for free, with an explicit `get_scale()` division as the fallback for a raw
`optimizer.step()` under a live scaler. Those two call sites are the *only* writers, so the
recorded loss is legitimately absent for `torch.autograd.backward(loss)`, a non-scalar backward
and closure-based optimizers like LBFGS — absent is now reported as unknown rather than as a
NaN, which is a third second-pass fix to this item ([C-5](SECURITY_AUDIT.md)).

**Model resolution is by parameter identity.** The frame walk survives only to produce
*candidates*; the winner is the module whose parameters actually overlap this optimizer's
param set, with ties broken toward the smaller module. A GAN's discriminator optimizer can no
longer be matched to the generator. `arc_watch(model, optimizer)` is the explicit escape
hatch (option A) for topologies where even that cannot resolve.

Verified on a real GPU: a 4×-accumulation AMP loop produces **20 backward calls, 5 optimizer
steps, 5 metrics**, with loss ≈0.17 rather than 0.17 × 65536, and gradient norms unscaled.
Locked in by `test_one_metric_per_optimizer_step_not_per_backward`.

**Also fixed here, and not in the original spec:** an LR intervention was being silently
erased. Any scheduler recomputes `group['lr']` from its own base every step, so dividing
`group['lr']` once is undone on the very next iteration — while the log still reports the
intervention as successful. `OptimizerMonitor.enforce_lr()` re-asserts ARC's reduction as a
multiplier each step, and leaves a group alone when its LR still holds the value ARC last
wrote, so a run with no scheduler never compounds. Most real runs use a scheduler, so this
was the common case, not a corner case.

### 2.2 Make instrumentation cheap enough to leave on
**Effort: 4 h · Impact: high — Status: done, and the claim is now measured**

See [M-2](SECURITY_AUDIT.md). All three changes shipped:

- `torch._foreach_norm(grads, 2)` — one fused kernel over the optimizer's own parameters.
  Loss and gradient norm are then stacked into one tensor and read with a single `.tolist()`,
  so a step costs **one** device sync instead of one per parameter tensor.
- Expensive collectors sample every 25 steps by default (`ARC_ADVANCED_EVERY`).
- Sampling goes dense automatically whenever risk is elevated, and always fires on step 1.

**Measured**, `python/benchmark_overhead.py`, RTX 3050, DemoCNN (2.79M params), 200 steps ×
batch 128, median of 3, same loop run with and without the harness:

| Configuration | s/run | ms/step | overhead |
| :--- | ---: | ---: | ---: |
| bare (no ARC) | 9.818 | 49.09 | — |
| ARC core metrics only | 9.994 | 49.97 | **1.8%** |
| ARC full (advanced every 25) | 10.640 | 53.20 | **8.4%** |
| ARC full (advanced every step) | 26.510 | 132.55 | 170.0% |

The last row is why sampling is not optional.

Two findings from doing this properly:

**The harness's own self-timing was badly misleading and has been removed.** It reported 54%
on a run whose true wall-clock cost was 8.4%, because the hook reads the loss and blocks on
GPU work that was already queued and that the training script's own `loss.item()` would have
waited for moments later. `run_summary` now reports `instrumentation_seconds` with that
caveat stated inline and no percentage at all. Wall-clock A/B is the only honest number.

**Two upstream fixes in `arc-training` were needed to get here** — see 2.5.

### 2.3 Fix traceback line numbers
**Effort: 2 h · Impact: medium-high — Status: done, exactly rather than approximately**

See [M-3](SECURITY_AUDIT.md). The header is not shortened — it is gone. `runner.py` installs
the instrumentation from `_arc_bootstrap`, then executes the target with
`runpy.run_path(target, run_name="__main__")`. The user's source is compiled **unmodified**,
so line numbers are exact rather than off-by-one, and this runner's own frames are stripped
from any reported traceback because the user cannot act on them.

Verified on a real error during development: a crash in `train_demo.py` reported
`line 180, in main / out = model(x)`, which is the actual line.
`test_traceback_points_at_the_user_script` locks it in.

### 2.4 Bound the checkpoint memory cost
**Effort: 3 h · Impact: medium-high — Status: done, by owning the checkpoint store**

See [M-5](SECURITY_AUDIT.md). `CheckpointStore` in `_arc_bootstrap.py` keeps snapshots in
**host** memory: `copy.deepcopy(state_dict())` in place leaves every checkpoint on the GPU,
which is how a tool that exists to warn about OOM causes one. The store also estimates its own
footprint before the first save and emits a `checkpoint_budget` event the dashboard surfaces.

Owning this rather than reaching into `WeightRollback`'s internals also closes
[M-6](SECURITY_AUDIT.md): the harness no longer calls `_save_checkpoint`,
`_restore_checkpoint`, `state.step_count` or `state.checkpoints`, so an `arc-training` rename
cannot silently disable rollback. RNG state (`torch`, `cuda`, `numpy`) travels with each
snapshot, which is what makes a rollback resume the same data order — and what makes the 1.3
A/B a fair comparison. `arc-training` is still used for the thing it is genuinely good at:
the structural signal collectors.

### 2.5 Make degradation visible
**Effort: 2 h · Impact: medium-high — Status: done, and it immediately found two real bugs**

See [M-6](SECURITY_AUDIT.md), [M-7](SECURITY_AUDIT.md). `warn_once(key, message)` reports the
first occurrence of each distinct instrumentation failure as a `degraded` event, suppresses
repeats so a per-step exception cannot flood the transport, lights a **DEGRADED** badge in the
dashboard header listing the affected components, and lists them in `run_summary`.

Turning the silence off immediately surfaced two genuine bugs in `arc-training` that the bare
`except: pass` had been hiding — and which were the actual root cause of the fabricated
telemetry in [C-2](SECURITY_AUDIT.md), because the collector threw every step and the
dashboard invented plausible numbers to fill the gap:

1. **`WeightCollector` raised on every CUDA model.** It cached previous weights on the host
   and then computed `weight - prev` across devices:
   `RuntimeError: Expected all tensors to be on the same device`. Every update-ratio and
   norm-growth signal silently vanished on any GPU run.
2. **`GradientCollector` crashed models using `inplace=True` activations.**
   `register_full_backward_hook` raises *"Output 0 of BackwardHookFunctionBackward is a view
   and is being modified inplace"* — which covers torchvision's ResNet, VGG and MobileNet as
   shipped. A monitoring collector was aborting the runs it was meant to observe. Fixed by
   dropping backward hooks entirely and reading `param.grad` at collect time: identical
   numbers, no autograd interference, one less callback per step.

Both fixed in `arc-training` (same author, AGPL). A third fix there —
`_compute_effective_rank` was copying every weight to the host and running the SVD on CPU —
was a large part of the overhead in 2.2.

### 2.6 Test the pure functions
**Effort: 3 h · Impact: medium — Status: done, 108 tests plus CI**

See [L-6](SECURITY_AUDIT.md).

**`tests/pure.test.js`** (node:test, no framework) — `extractCodeBlock`,
`buildScriptGenMessages`, `buildSystemPrompt`, `buildReportHtml`. Includes the 10 000-entry
case that guards the `reduce()` replacing `Math.max(...array)`, a NaN-rendering check, and an
XSS-escaping check on report output.

**`tests/dashboard.test.js`** — extracts the dashboard's inline script, compiles it after
placeholder substitution (a leftover `{{...}}` inside the script block is a page that does not
run at all), asserts the CSP is nonce-based with no `'unsafe-inline'`, asserts no inline
`on*=` handler attributes survive, asserts no external `<script src>`, asserts
`Math.random()` never reappears in the dashboard, and unit-tests the GPU rate table.

**`tests/ring-buffer.test.js`** and **`tests/model-name.test.js`** — the two helpers extracted
from `extension.ts` so they could be tested at all.

**`tests/test_harness.py`** — risk heuristic, the JSON finite-guard, the structural detector
(including that a single bad sample is treated as noise and that the precursor rule is not
starved), loss-trend robustness against mini-batch noise, the fused gradient norm checked
against the naive per-parameter computation it replaced, AMP unscaled-loss capture, model
resolution including a two-optimizer GAN, `CheckpointStore` round-trip and RNG determinism, the
LR guard against a scheduler, plus six end-to-end tests that run the real runner on real
training loops.

**`.github/workflows/ci.yml`** runs all of it on push, and adds a secret-scan job that fails
the build on a secret-shaped literal or an embedded JWT — so [C-1](SECURITY_AUDIT.md) cannot
regress silently.

Worth noting what this actually bought, since "add tests" is easy to treat as hygiene: writing
them found **four bugs that reading the code had not**. The most serious was in the
optimizer-to-model match — it preferred a wrapper module over the submodule an optimizer
actually updates, so a GAN's discriminator failure would have rolled back the generator too.
That is the exact class of bug [M-1](SECURITY_AUDIT.md) was about, surviving in a new form
inside its own fix.

---

## Tier 3 — Beyond the hackathon

### 3.1 Put a real LLM in the recovery loop
`arc_agent.py` is a threshold-based rule engine that *renders* as a ReAct trace
(see [ARCHITECTURE.md §6](ARCHITECTURE.md)). That is the right call for a live demo —
deterministic, offline, no latency, no API key, no failure mode on stage.

But `chatManager.ts` already implements streaming across five providers, and
`contextBuilder.ts` already formats run telemetry into a prompt. The pieces for a genuine
agent are present and connected to the wrong endpoint.

The honest version is **hybrid**: rules handle the reflex (NaN → rollback, immediately, no
network), and the LLM handles the diagnosis afterwards — *why* it diverged, what to change in
the architecture, whether the LR schedule is wrong to begin with. Reflex must never depend on
a network call, and analysis benefits from one.

Until that ships, describe the current loop as *deterministic* in all materials. Calling a
rule engine an AI agent is the kind of thing that gets found, and being found beats disclosing.

### 3.2 Distributed training
DDP and FSDP are where expensive failures actually live — nobody loses $12 of compute, they
lose $12,000. Currently the patch fires per-rank with no rank awareness, so telemetry from 8
GPUs interleaves into one stream and rollback is uncoordinated.

Needs: rank-0-only emission, `all_reduce` for global gradient norms, and a barrier so all
ranks roll back to the same checkpoint. This is the largest single item on this list and the
one with the clearest commercial pull.

### 3.3 Import from TensorBoard / W&B
Point ARC at existing event files and let it replay historical runs through the same
diagnostics. Instant value with zero instrumentation — and a way to demonstrate the tool on a
failure the user already remembers, which is far more persuasive than a synthetic one.

### 3.4 Run history and regression detection
Persist runs to the workspace, then compare: *"this run's gradient norm at step 500 is 3×
last week's."* Failure prediction becomes possible once there is a baseline, and prediction is
a substantially stronger product than reaction.

### 3.5 Notebook support
Kaggle and Colab are where the target users actually train, and the script generator already
emits `.ipynb`. A cell-magic (`%%arc_watch`) reaches them where they are and is a much shorter
path to real usage than the VS Code extension.

---

## Summary matrix

| # | Item | Effort | Impact | Tier | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.1 | Delete `enrichEvent()` | 30 m | **Decisive** | Do first | ✅ Done |
| 0.2 | Vendor chart libraries | 1 h | High | Do first | ✅ Done (rebuilt on ECharts) |
| 0.3 | Resolve interpreter via Python ext | 1 h | High | Do first | ✅ Done |
| 0.4 | Strip secrets, delete `context.md` + patch scripts | 30 m | High | Do first | ✅ Done in-repo — published `.vsix` needs the account owner |
| 1.1 | Compute-savings ledger | 4–6 h | **Decisive** | Differentiator | ✅ Done (rate derived from detected GPU, source always labelled) |
| 1.2 | Intervention markers on charts | 3 h | High | Differentiator | ✅ Done (markers, labels, shaded failure→recovery span) |
| 1.3 | Side-by-side A/B comparison | 8–12 h | **Decisive** | Differentiator | ✅ Done (`ARC_MODE=baseline`, command, harness, test) — baseline arm was silently intervening until C-6 |
| 1.4 | Exportable post-mortem report | 4 h | High | Differentiator | ✅ Done (`reportBuilder.ts`, self-contained HTML) |
| 2.1 | Explicit handle / patch `Optimizer.step` | 6 h | High | Trust | ✅ Done + verified on GPU (AMP, accumulation, multi-optimizer) — second pass fixed three defects this change introduced (C-4, C-5, H-5) |
| 2.2 | Cheap instrumentation + published benchmark | 4 h | High | Trust | ✅ Done — measured 1.8% / 8.4% |
| 2.3 | Fix traceback line numbers | 2 h | Med-High | Trust | ✅ Done (`runpy`, zero injected lines) |
| 2.4 | Bound checkpoint memory | 3 h | Med-High | Trust | ✅ Done (host-resident store, budget reported) |
| 2.5 | Visible degradation | 2 h | Med-High | Trust | ✅ Done — found 2 real upstream bugs |
| 2.6 | Tests + CI | 3 h | Medium | Trust | ✅ Done (108 tests, 3 CI jobs incl. secret scan) |
| — | **Structural detection reachable at all** *(not in the original plan)* | — | Medium | Trust | ⚠️ Done, then mostly walked back — see below |
| 3.1 | Hybrid LLM recovery loop | 2 d | High | Later | Open — deliberate, see below |
| 3.2 | DDP / FSDP support | 1 w | High | Later | Open — deliberate, see below |
| 3.3 | TensorBoard / W&B import | 3 d | Medium | Later | Open |
| 3.4 | Run history + regression detection | 4 d | High | Later | Open |
| 3.5 | Notebook cell magic | 2 d | Medium | Later | Open |

### The item that was not on the list — and how much of it survived

Running the fixed harness against real CIFAR-10 exposed something the roadmap had not
anticipated. It looked at the time like the most valuable thing in this document. After
measurement, most of it is gone, and the residue is smaller than the original write-up claimed.

**What was found.** `weight_update_ratio`, `gradient_entropy` and `effective_rank` were only
ever consulted *inside* the failure handler, which only ran on a NaN or an exploded loss. So the
signals that were supposed to distinguish this product from a plotting library could not fire
unless a numerical failure had already happened — at which point they are redundant. A real run
made it concrete: at `lr=0.5` the network died around step 60, sat at chance for the remaining
330 steps, and ARC reported **zero** failures.

Structural health is now evaluated on every advanced sample, independent of loss. That part
stands. What did not stand is every rule the section was originally built on.

**The update-ratio rule was removed first.** Measured across four learning rates, its
distribution on a healthy run overlaps a failing one almost completely — the healthy run peaks
*higher* (0.322 vs 0.285) and sustains a *longer* consecutive breach. It was a proxy for "the
learning rate is large". Acting on it cost 1.74 and 0.78 points of validation accuracy on A/B
runs that needed no help.

**The entropy rule was removed second, and it had cost far more.** This document previously
argued that "a high update ratio is the actionable early warning; entropy collapse is the
obituary", and treated reordering the rules so entropy could not starve the update-ratio rule as
the fix that made detection work. Both halves of that were wrong. On a CIFAR-10 A/B at
`lr=0.25`, with the control arm C-6 had just made trustworthy:

| Arm | Final val accuracy | Failures | Interventions |
| :--- | ---: | ---: | :--- |
| baseline | **87.43%** | 0 | — |
| active | **10.00%** — chance | first at step **125** | 3 × rollback + LR cut, then `unrecoverable` |

Three rollbacks inside epoch 1 took the LR from 2.45e-01 to 3.07e-02 and the model to chance,
where it stayed. ARC destroyed a healthy run and then correctly reported it unrecoverable,
because ARC had made it so. Measuring the trajectory afterwards showed no threshold would have
helped — at step 70 the healthy run and a genuinely dead one both read 1.44e-05 and stay
together for the rest of training, because `GradientCollector._compute_entropy` bins a
heavy-tailed distribution linearly and saturates near zero on any run. The rule is deleted; the
signal is still charted. Full detail: [C-7](SECURITY_AUDIT.md#second-review-pass).

**And deleting it exposed that the tests had been certifying the bug.** Three integration tests
broke on the deletion. The cause was not a regression: their divergence fixture never diverged —
its loss peaks at 1.93 in plain PyTorch and never approaches the 1e6 threshold — and they had
been passing on the entropy rule's false positive, because `assertTrue(failures)` cannot tell a
real detection from a spurious one. Fixed with a fixture verified to diverge with ARC detached
(loss > 1e6 at step 10) and an assertion on the failure *kind*. See
[C-8](SECURITY_AUDIT.md#second-review-pass).

**What is actually left.** Two structural rules.

`loss_plateau` fires when the loss has not beaten its best value for 300 consecutive steps *and*
the run never improved past 60% of its opening loss. Both conditions are needed. It
exists because a real lr=0.5 run finished at 10.00% — chance — with its loss pinned at ln(10),
and ARC reported **zero failures across all 780 steps**: the loss was finite, the gradient norm
was 0.07, and the rank barely moved. Measured on both arms, a healthy run's longest stall is 82
steps against the dead run's 764, a 9.3x separation. It fires at step 316–330 on the dead arm.

The progress condition was added after a 10-epoch A/B caught the patience-only version firing
twice on a run that reached 87.5%. Convergence is itself a plateau — the best-ever-loss counter
makes stalls grow without bound as a run succeeds — so no patience value separates the two. What
does is whether the run ever got anywhere: best/first loss is 0.271 on a healthy run and 0.888
on a dead one.

**It is report-only, and that is the second thing the A/B corrected.** The rule shipped with
`reduce_lr` as its response. On the `lr=0.5` arm the control sat at chance for four epochs and
then escaped by itself — cosine decay walked the LR down to ~2.5e-01 and the run climbed to
73.19% — while the intervened arm, already cut to 3.2e-02, stayed at 10.00% for all ten epochs.
Large steps were the only mechanism that could leave the dead region, and the intervention
removed them: −63.19pp from a correct detection. Rolling back instead is no better, since
confirmation takes 300 steps and every retained checkpoint is post-collapse by then. With no
response known to help, the rule reports and stops.

`representation_collapse` fires at effective rank below 50% of the run's own baseline. For a
long time it **never fired at all**, and that had a measured reason: a healthy run bottoms at
97.2% of its step-1 baseline and the dead run we had measured at 87.4%, so the trigger sat four
times further away than a real collapse reached. `mean_effective_rank` is the SVD entropy of the
weight matrices, and a network can emit a constant output while every weight matrix stays
well-conditioned — it measures weight conditioning, not representational rank.

**It is also report-only now**, because the one sweep in which it did fire showed its response
doing the damage: at `lr=0.5` the control arm recovered to 75.18% while the arm it rolled back
and cut three times finished at 30.84%. −44.34pp, the same mechanism as the plateau rule — the
control escaped once cosine decay lowered the LR by itself, and the cut arm never did.

A related bug is fixed: baselines used to be captured *after* the 200-step warmup, so a run that
died earlier had its reference measured on the corpse — the dead arm scored 99.72% of its own
baseline against the healthy arm's 98.69%, ranking the corpse as more stable. Baselines are now
captured from the opening samples while the verdict still waits for the warmup.

Alongside those: numerical divergence (non-finite or `|loss| > 1e6`), which is verified working,
and gradient-norm clipping above 50. That is the shipped detection surface.

### Open, and the most promising lead we have: `grad_flow_ratio`

**Status: measured, not implemented. Needs the four-LR sweep before it becomes a rule.**

While validating the plateau rule we found that `grad_flow_ratio` — late-layer gradient norm
over early-layer, already collected and already charted — separates the two arms harder than
anything else, and does it **266 steps earlier**:

| step | healthy lr=0.03 | dead lr=0.50 |
| ---: | ---: | ---: |
| 1 | 2.08 | 2.08 |
| 25 | 3.10 | 5.43 |
| 50 | 2.36 | **50.11** |
| 75 → 375 | 1.36 – 2.72 | **absent (non-finite)** |

The healthy run stays inside a 1.36–3.10 band for all 16 samples. The dead run is 16x above the
healthy maximum by step 50 and non-finite from step 75 on, because early-layer gradient norms
fall below the `1e-10` floor upstream and the ratio becomes `inf`. That is textbook vanishing
gradient: the early layers stop receiving signal entirely while the late layers still have one.

Two things follow.

**The absence is currently thrown away.** `_finite()` correctly refuses to emit `inf` into JSON,
so the chart shows a gap exactly where the signal is screaming. Dropping a non-finite value is
right; dropping the *fact* that it went non-finite is information loss. The harness should
record that the ratio diverged without inventing a number for it.

**This is the first candidate that could actually rescue a run rather than report it.** The
network dies around step 45. `loss_plateau` cannot confirm before step ~316, by which point
every retained checkpoint is post-collapse — which is precisely why its response is an LR cut
and not a rollback. A trigger at step 50 lands *at* the collapse, while checkpoints from step 40
may still predate it. That is the difference between "we detected a death" and "we prevented
one", and it is the single highest-value item left on this list.

It is not a rule yet, and it must not become one on this evidence. This is one seed on one
workload, which is exactly the standard the update-ratio and gradient-entropy rules met right
before they were deleted for harming healthy runs. It needs the four-learning-rate A/B, and the
threshold has to be relative to the run's own early band rather than an absolute ceiling —
`grad_flow_ratio` is architecture-dependent, and a value tuned on a 9-layer CNN means nothing on
a transformer.

Two structural rules removed for the same reason is the generalisable result, and it is worth
more than the feature would have been: in both cases a signal's natural early-training
trajectory resembled the pathology it was meant to detect. Hence the 200-step
`STRUCTURAL_WARMUP_STEPS` gate before any baseline is captured, and hence the bar for adding a
rule — a measured trajectory showing separation on a healthy *and* a failing run, not a
plausible story about what the signal means.

The honest summary of this item: it was overclaimed, the measurement that corrected it cost a
run and three tests' worth of credibility, and the capability that remains is materially smaller
than the paragraph it replaced but is the part that survives contact with a control arm.

Gradient clipping also became a **real** intervention rather than an advisory one. The old
backward-anchored design could only recommend it, because it did not own the caller's
`optimizer.step()`. The optimizer anchor runs immediately before the update, so ARC now
applies the clip itself, with no user code change.

**And then the A/B caught the detector over-firing.** The first full sweep showed a clean win
at `lr=0.25` and a **1.74-point loss at `lr=0.1`** — ARC intervening once on a run that reached
87.86% without help. The log for that run looked like success: a failure detected, an
intervention applied. Only the control arm showed it was harm.

Three causes, all fixed, all in the direction of "do not act unless the evidence is real":

* The progress guard estimated a trend from **three** noisy samples. A per-step mini-batch loss
  bounces by a large fraction of its own value, so that is close to a coin flip — and each
  wrong flip is an intervention on a healthy run. Now 60 samples averaged in thirds, requiring
  a *measurable* improvement rather than merely "not worse".
* Representation collapse was not gated on progress at all, though effective rank falls during
  healthy training as layers specialise. The later fix is broader: no structural rule is
  evaluated at all until step 200, because the opening transient is where all of these signals
  look pathological.
* Adaptive sampling densified to *every* step under elevated risk — the 170%-overhead regime,
  applied longest to the runs already in trouble. Capped at 5× the normal rate.

The pattern in all of this — including the two rule deletions above — is the one thing here
worth keeping: every correction came from a measurement that contradicted the tool, and each
one made the product's claimed capability smaller. That is the opposite direction from a feature
that makes the demo look better, and it is the only reason the remaining claims are worth
anything.

**And then a second review pass caught the A/B itself.** With all of the above marked done, the
remediated code was reviewed again instead of being declared finished. It found three further
Critical defects, and two of them land squarely on the work described in this section:

* The optimizer anchor — item 2.1, the fix this whole tier is built on — made *constructing any
  LR scheduler* raise `AttributeError` before training started, because the wrapper was a plain
  function and `LRScheduler.__init__` reads `optimizer.step.__func__`. Most real scripts use a
  scheduler. The fix that made instrumentation correct had made the extension unusable on the
  majority of its target scripts, and every piece of validation the project owned ran a demo
  script that constructs no scheduler.
* Baseline mode was **not** intervention-free. The branch that logs "interventions suppressed"
  called `optimizer.zero_grad()` before returning, and dropping the update is the single most
  effective intervention ARC has. The control arm of the experiment in
  `docs/EXPERIMENT_RESULTS.md` was being protected by the mechanism the experiment exists to
  measure — so the measured benefit of interventions was being compared against a partially
  treated control. Same failure shape as the two above: a log line asserting inaction while the
  code acted.

**And then validating the remediated detector caught the detector.** The A/B that C-6 had just
made trustworthy showed the entropy rule taking a run from 87.43% to chance and then declaring
it unrecoverable, and removing the rule showed that three integration tests had been passing on
its false positive. Two more Critical findings, C-7 and C-8, both above.

The pattern across all four passes is the same and worth naming. Every one of these defects was
in code that had just been fixed, was described correctly in prose, and was believed done. The
useful conclusion is not "we fixed more bugs" — it is that "this component has been reviewed" is
worth very little, and that the validation surface (one demo script, reused by the benchmark and
the A/B harness) was narrow enough to certify a broken anchor as verified. Widening it is the
top remaining trust item, ahead of every feature in Tier 3. C-8 sharpens that: the suite was
not merely narrow, it was green *because* of a defect. Full detail:
[Second review pass](SECURITY_AUDIT.md#second-review-pass).

---

## What is deliberately still open

**3.1 — hybrid LLM recovery loop.** The reflex path stays deterministic on purpose: a NaN
needs handling in microseconds and must not depend on a remote service being reachable. The
LLM belongs in the *analysis* path, where it already is. `arc_agent.py`'s docstring states
plainly that it is a rule engine rendered as a ReAct trace, so nothing here is claimed to be
an agent that isn't.

**3.2 — DDP / FSDP.** Genuinely the highest-value remaining item, and not something to claim
without testing. It needs rank-aware emission, `all_reduce` for global gradient norms, and a
barrier so every rank rolls back to the same checkpoint. Verifying that requires a multi-GPU
machine; this work was done on a single RTX 3050, and shipping untested distributed rollback
would be worse than not shipping it.

**C-1's published artifacts.** The four released `.vsix` files still carry the old signing
secret. That needs the marketplace account owner, not a code change.

---

## If there is time for exactly three things

All of Tier 0, Tier 1 and Tier 2 are done, so the question has moved on.

The three things that would now add the most:

1. **3.2, DDP/FSDP** — where the expensive failures actually live. Nobody loses $12 of
   compute; they lose $12,000.
2. **3.4, run history and regression detection** — prediction is a much stronger product than
   reaction, and it only becomes possible once there is a baseline to compare against.
3. **Widen the A/B** — the results in [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) cover
   one architecture on one dataset on one GPU. A second architecture (a transformer, where
   AMP and accumulation actually matter) would test the 2.1 fixes on the setup they were
   written for.
