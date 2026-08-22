# ARC Lens — Judge Q&A Prep

Internal prep doc, not marketing copy. Answers are grounded in `README.md`,
`ARCHITECTURE.md`, `SECURITY_AUDIT.md` (completed this session), `ARC_FUNDING_PROPOSAL.md`,
`arc_lens_business_plan.md`, and `FUTURE_IMPROVEMENTS.md`. Bracketed notes point to the
source so you can re-verify before repeating a number out loud.

**Ground rule for the room:** if a judge asks something not covered here, the safe move is
"we haven't verified that — here's what we do know," not a confident guess. Judges reward
that answer more than they reward a smooth deflection.

---

## How it works

**Q1. What does ARC Lens actually do, in one sentence?**
It watches a PyTorch training loop in real time, streams telemetry (loss, gradient norm,
learning rate, GPU memory, and a few advanced signals) to a VS Code dashboard, and — when a
run starts to diverge — automatically rolls the model back to the last healthy checkpoint and
lowers the learning rate, without the user editing their training script.
[`README.md`, `ARCHITECTURE.md` §1]

**Q2. How does "zero code changes" actually work?**
`runner.py` installs the instrumentation, then executes your script **unmodified** through
`runpy` — nothing is prepended, so a traceback reports the line numbers that are actually in
your file. The instrumentation patches `Optimizer.__init__` and wraps each optimizer instance's
`step`, so one recorded step is one weight update.

The anchor choice is the interesting part. Hooking `Optimizer.step` rather than
`Tensor.backward` is what makes gradient accumulation, mixed precision and multi-optimizer
setups correct: `self` is definitionally the right optimizer and `self.param_groups` the right
parameters. The model is matched to it by **parameter identity** — a frame walk only proposes
candidates, and the module that actually owns the optimizer's parameters wins.
[`ARCHITECTURE.md` §5; `SECURITY_AUDIT.md` §M-1]

**Q3. Walk me through what happens the moment a run starts to fail.**
The step hook runs immediately before the update. It computes the gradient norm over that
optimizer's own parameters with one fused kernel and one device sync, then tests
`isnan(loss) or isinf(loss) or |loss| > 1e6`. On failure it emits `failure_detected`, hands
control to the recovery agent — which restores the last healthy checkpoint and scales the LR
down in the live optimizer *and* inside every stored checkpoint, so a later rollback cannot
resurrect the learning rate that caused the divergence — and then zeroes the gradients so the
poisoned update the caller is about to apply becomes a no-op. Training continues in the same
process.

Two details we got wrong and fixed: an LR cut is re-asserted every step, because any scheduler
recomputes the rate from its own base and would otherwise erase the intervention on the next
iteration; and the paths where ARC decides *not* to act now return without touching the
gradients, because dropping an update is itself an intervention.
[`ARCHITECTURE.md` §5; `SECURITY_AUDIT.md` §M-1, §C-6]

**Q4. Where's the line between `arc-training` (PyPI) and ARC Lens (the extension)?**
`arc-training` is the signal-collector and checkpoint/rollback library — the actual math.
ARC Lens is the instrumentation harness that wires it into a live script, the transport layer,
the dashboard, and the optional LLM diagnostics on top. If you deleted ARC Lens, `arc-training`
would still work as a library; if you deleted `arc-training`, ARC Lens would have no signals
to show.
[`ARCHITECTURE.md` §1]

**Q5. What are the "advanced diagnostics" — Effective Rank, Gradient Entropy, etc. — actually measuring?**
They come from `arc-training`'s collectors: effective rank estimates representation collapse in
a layer's weight matrix, gradient entropy measures how much information the gradients still
carry, weight update ratio is `‖ΔW‖/‖W‖`, and gradient flow ratio compares early- against
late-layer gradient magnitude (it needs ≥4 parameterised layers to mean anything).

**One of the four drives an intervention; three are telemetry only, and that distinction is
measured rather than assumed.** An earlier version of this answer said gradient entropy was the
one that earned its place, and cited it catching a dead network. That was wrong, and the way it
was wrong is the most useful thing on this page.

**Gradient entropy triggered an intervention and cost us an entire run.** On a CIFAR-10 A/B at
`lr=0.25`, the control arm finished at 87.43%. The ARC-protected arm fired
`gradient_entropy_collapse` at step 125, rolled back and cut the LR three times inside the first
epoch, and finished at 10.00% — chance — after which ARC correctly declared it unrecoverable,
having made it so. Measuring the trajectory afterwards showed no threshold could have saved the
rule: by step 70 the healthy run and a genuinely dead one both read 1.44e-05 and stay together
for the rest of training. The cause is upstream — `_compute_entropy` bins a heavy-tailed
gradient distribution with `torch.histc` on a linear scale, so outliers set the range, nearly
all the mass lands in one bin, and the normalised entropy saturates near zero on *any* run. It
measures outlier spread, not information content. The rule is deleted. The signal is still
charted, because a human can read it; it cannot act.

**Weight update ratio went the same way, earlier.** It *used* to trigger an LR cut above 0.05.
Measured across four learning rates, its distribution on a healthy run overlaps a failing one
almost completely — the healthy run peaked higher (0.322 vs 0.285) and sustained a *longer*
consecutive breach (31 samples vs 26). A proxy for "the learning rate is large", not "training
is failing", and acting on it cost 1.74 and 0.78 points of validation accuracy in A/B runs that
needed no help.

**What is left is effective rank**, firing below 50% of a baseline that is not captured until
step 200. We will say plainly that it has **never fired in validation** — a healthy run bottoms
at 96% of baseline and a damaged one at 83%, against a 50% trigger. It is conservative and
therefore unexercised. It is not a demonstrated capability, and we do not present it as one.

Two rules removed for the same underlying reason is the honest through-line: a signal's natural
early-training trajectory resembled the pathology it was meant to detect, in both cases. That is
also why structural checks now wait 200 steps before capturing any baseline at all.
[`ARCHITECTURE.md` §5; `SECURITY_AUDIT.md` §C-7; `EXPERIMENT_RESULTS.md`]

---

## Claims & benchmarks

**Q6. Your funding proposal says 100% recovery rate, zero false positives, across 9 architectures. Can I trust that?**
That's an internal result from a controlled, synthetic-failure protocol (programmatically
injected NaN/Inf/LR-spike failures on CPU), not an independently audited benchmark and not yet
validated on GPU or on organic (naturally-occurring) failures. We're confident in the mechanism
because we can show you the recovery happening live, but "100% on synthetic injected failures"
and "100% on real training runs at scale" are different claims, and we've only measured the
first one.
[`ARC_FUNDING_PROPOSAL.md` §3.1, §3.4; `FUTURE_IMPROVEMENTS.md` §3.2 lists GPU/distributed validation as still open]

**Q7. What's your actual measured runtime overhead, not the claimed number?**
Measured on GPU, by running the same loop with and without the harness — which is the only
overhead number worth quoting. RTX 3050, 2.79M-parameter CNN, 200 steps × batch 128, median
of 3: **1.8%** for core metrics, **8.4%** with the structural diagnostics. Reproduce it with
`python python/benchmark_overhead.py`; the raw numbers are in `docs/benchmark_overhead.json`.

The original code called `.item()` per parameter tensor — 161 forced GPU syncs per step on
ResNet-50 — which would have blown the "<10%" claim on exactly the architectures the product
targets. That is one fused `torch._foreach_norm` and a single sync now.

Two things worth volunteering here. Sampling the structural signals *every* step costs 170%,
which is why they are sampled every 25 and why "densify when unstable" is capped at 5× rather
than every step. And the harness's own self-timing is deliberately not reported as a
percentage: it read 54% on the run that wall-clock A/B measured at 8.4%, because reading the
loss blocks on GPU work the training script would have waited for anyway.
[`docs/benchmark_overhead.json`; `SECURITY_AUDIT.md` §M-2]

**Q8. The 9-architecture, 10M–117M parameter validation — was that on real hardware?**
CPU only. The funding proposal itself lists "GPU / CUDA validation suite" as the top (P0)
immediate roadmap item and names GPU compute as "the single remaining gate to enterprise
credibility." We're not hiding that — it's stated in our own proposal.
[`ARC_FUNDING_PROPOSAL.md` §4.1, §5.1, §8]

**Q9. What's the 97.5% MLP failure-prediction accuracy, and does ARC Lens use it today?**
That's a separate result for `arc-training`'s early-warning classifier (12 engineered
features, 5-fold cross-validation on synthetic failure scenarios), not something wired into
the recovery path you'd see in ARC Lens right now. The rule engine in the current recovery
loop is threshold-based, not the ML classifier described in the proposal — don't conflate the
two when answering benchmark questions.
[`ARC_FUNDING_PROPOSAL.md` §3.2; `ARCHITECTURE.md` §6]

---

## Limitations & honesty

**Q10. Does this work with mixed-precision (AMP) training?**
Yes, and it is tested on GPU. `GradScaler.scale(loss)` is patched, which is the one place the
**unscaled** loss is available without guessing at the scale factor. And because
`scaler.step(optimizer)` unscales the gradients before running the optimizer, the norm read at
the step anchor is already correct — an explicit `get_scale()` division covers the case where
someone calls `optimizer.step()` directly under a live scaler.

`scaler.unscale_()` is patched too, for the standard unscale-then-clip recipe: without that,
a user following the documented AMP pattern but calling `opt.step()` directly had the norm
divided by the scale a *second* time and reported ~65536× too small — low enough that no
gradient rule could ever fire.

Verified: a 4×-accumulation AMP loop reports 20 backward calls, 5 optimizer steps, 5 metrics,
with the loss at its true value rather than ×65536.
[`SECURITY_AUDIT.md` §M-1; `tests/test_harness.py::TestAmpLossCapture`]

**Q11. What happens if I have two optimizers — say, a GAN with a generator and discriminator?**
Handled, and tested. Because the anchor is `Optimizer.step`, `self` is definitionally the
optimizer being stepped — there is nothing to guess. The model is matched to it by **parameter
identity**: a frame walk only proposes candidates, and the module whose parameters actually
overlap that optimizer's param set wins, with the *smallest* exact owner preferred.

That last clause exists because writing the test found a bug. Only frame *locals* reach the
walk, so a model held as an attribute — `self.discriminator`, `pair.disc` — was invisible and
the enclosing container won by virtue of owning every parameter. Rolling that back would have
discarded the other network's progress: the same misattribution in a new disguise, inside its
own fix. Candidates are now expanded through submodules.

Wrapper optimizers (Lookahead, SAM) are also handled — they subclass `Optimizer` and delegate
to an inner one, so both were instrumented and one update counted as two. A re-entrancy guard
makes the outermost `step()` the update.
[`SECURITY_AUDIT.md` §M-1, §H-5; `tests/test_harness.py::TestModelResolution`]

**Q12. Does it handle gradient accumulation correctly?**
Yes. One `Optimizer.step` is one weight update, so four backwards followed by one step record
one metric, not four. The loss attributed to that update is the one from the last backward
before it.

Verified on GPU and locked in by a test: a 4×-accumulation loop reports **20 backward calls,
5 optimizer steps, 5 metrics**. The run summary carries both counts precisely so this stays
checkable.
[`SECURITY_AUDIT.md` §M-1; `tests/test_harness.py::test_one_metric_per_optimizer_step_not_per_backward`]

**Q12b. What *doesn't* work?**
Distributed training. DDP/FSDP needs rank-aware emission, `all_reduce` for global gradient
norms, and a barrier so every rank rolls back to the same checkpoint. None of it is tested,
because it needs multi-GPU hardware we do not have, and we would rather say that than claim it.
Single-process multi-optimizer is fine and tested; distributed is the next major piece.

**Q13. Why should I trust a reasoning trace that looks like an LLM agent if it's actually just if/else rules?**
Because we're telling you it is one. `run_llm_agent()` in the recovery path is a deterministic
rule engine that emits `thought` events shaped like a ReAct trace — perception, action,
observation — but it selects tools by threshold, not by inference, and it makes zero network
calls. We chose that on purpose, and not only for the demo: the reflex path of a reliability
tool must not depend on a remote service being reachable, and a NaN needs handling in
microseconds rather than a round trip. The LLM belongs in the *analysis* path, which is where
the Failure Analyst panel already is. Calling a rule engine an "AI agent" without saying so is
exactly the kind of thing that gets found in review, so `arc_agent.py`'s own docstring says it.
[`ARCHITECTURE.md` §6, `python/arc_agent.py` — no `requests`/`http`/provider calls in the
recovery loop; `FUTURE_IMPROVEMENTS.md` §3.1]

**Q14. What did you find wrong with your own project, and what have you actually fixed?**
We audited our own code and found 3 critical, 4 high, 10 medium and 11 low findings. Then we had
the *remediated* code reviewed independently, which found six more — three of them critical.
Then we validated the twice-remediated detector against a real A/B, which found two more, both
critical, and both in the detection logic itself. **All 36 are now fixed in the repository**,
with one exception noted below.

The ones worth naming: the dashboard's `Math.random()` fallback rendered fabricated telemetry
identically to real measurements; the public build was a scripted simulation rather than a
monitor; the `backward()` hook misreported every value under AMP and double-counted under
gradient accumulation; checkpoints were held on the GPU by a tool whose job is warning about
OOM; and every instrumentation failure was swallowed by a bare `except: pass`.

Fixing that last one is what made the audit worth doing. With the silence turned off, two
genuine bugs surfaced in `arc-training` itself that had been hidden on every single step — and
they were the *root cause* of the fabricated telemetry. One made `WeightCollector` throw on
every CUDA model; the other made `GradientCollector` **crash any model using
`inplace=True` activations**, which is torchvision's ResNet, VGG and MobileNet as shipped. A
monitoring collector was aborting the runs it existed to protect.

**The later passes are the part we would point a judge at.** Having fixed all 28, we had the
remediated code reviewed again rather than calling it done, and it found six more defects:

* **Constructing any LR scheduler crashed.** The fix that moved the anchor to `Optimizer.step`
  assigned a plain function to `optimizer.step`. `LRScheduler.__init__` reads
  `optimizer.step.__func__`, which a plain function does not have — so `StepLR`,
  `CosineAnnealingLR`, `OneCycleLR` and `ReduceLROnPlateau` all raised `AttributeError` before
  the first batch. Most real training scripts use a scheduler, so our headline fix had broken the
  extension for the majority of its users. Nothing caught it because the demo script, which the
  benchmark and the A/B harness both run, sets the LR by hand and constructs no scheduler.
* **A loss we could not observe was treated as a NaN.** For `torch.autograd.backward(loss)`, a
  non-scalar backward, or LBFGS, the loss is legitimately unavailable. ARC read that as a
  numerical failure on *every* step of a healthy run — rolling back, cutting the LR and zeroing
  the gradients, so the model never moved while the dashboard reported successful interventions.
* **Baseline mode was not intervention-free, which compromised our own A/B methodology.** All
  three "we are not intervening" branches called `optimizer.zero_grad()` before returning, and
  dropping the update is an intervention — the most effective one we have on a diverging run. The
  control arm of the experiment was being protected by the exact mechanism the experiment exists
  to measure.

Plus three more: wrapper optimizers (Lookahead, SAM) got instrumented twice; one surviving line
of the C-2 fabrication (`msg.grad_norm ?? 0.001`) still drew an invented gradient norm as a real
point; and the script generator's prompt told the model to emit `import arc` / `arc.wrap(model)`,
an API that does not exist, so every generated script contained a call that could only fail.

**Then, with a control arm we could finally trust, we caught the detector.** Two more critical
findings, and these are the ones that changed what the product claims:

* **A detection rule destroyed a healthy run.** `gradient_entropy_collapse` fired at step 125 on
  a CIFAR-10 run whose control arm finished at 87.43%. Three rollbacks and LR cuts inside epoch 1
  took it to 10.00% — chance — and ARC then correctly reported it unrecoverable, because ARC had
  made it so. No threshold would have worked: by step 70 a healthy run and a dead one both read
  1.44e-05, because the upstream entropy computation bins a heavy-tailed distribution linearly
  and saturates near zero on any run. The rule is deleted.
* **Removing it made three tests fail, and the tests were the bug.** Their divergence fixture
  never diverged — loss peaks at 1.93 in plain PyTorch — and they had been passing on the entropy
  rule's false positive, because `assertTrue(failures)` cannot tell a real detection from a
  spurious one. The suite was certifying the defect. The fixture is rebuilt and verified to
  diverge with ARC detached, and the assertion now checks the failure *kind*.

We would rather a judge hear this from us: our detection surface is smaller than it was. It is
numerical divergence, gradient clipping, and one conservative rank rule that has never fired.

We would rather show this than hide it. The honest summary is that three of the six were
*introduced or left behind by the first round of fixes*, they were all in code that had just been
audited and was described correctly in prose, and they were found because the work was reviewed
again instead of being declared done. The concrete lesson is about the validation surface, not
about care: one demo script, reused by the benchmark and the experiment, was narrow enough to
certify a broken anchor as verified.

**The one thing not fixed here:** the four already-published `.vsix` files contain the old
signing secret in their bytes. That needs the marketplace account owner to revoke the token and
supersede those releases — no code change can reach them. CI now fails the build on any
secret-shaped literal so it cannot recur.
[`SECURITY_AUDIT.md` — remediation status on every finding]

**Q15. If I clone the public repo right now and run it, does it actually monitor my training?**
Yes. This used to be "no", and it was the single most damaging thing about the project: the
checked-in `runner.py` ignored your script and replayed a fixed NaN at step 20 regardless of
what you opened.

The repository now ships the real harness. `python/runner.py` executes your actual script,
`_arc_bootstrap.py` instruments it, `arc_agent.py` is the real rule engine — as plain,
readable `.py` files rather than base64 blobs. The stub/`private_backup` split and both
`dev:enable`/`dev:disable` scripts are deleted, so there is no longer any difference between
what is checked in and what ships. Nothing in the harness needed protecting: it is an
`Optimizer` patch plus calls into the public `arc-training` API, and keeping it private was
hiding the most impressive part of the project.
[`README.md`; `ARCHITECTURE.md` §5; `SECURITY_AUDIT.md` §H-4, §C-3]

---

## Technical depth / "could I break this"

**Q16. What stops someone from just using gradient clipping instead?**
Nothing stops them from also using it — but clipping only addresses gradient explosion, and it
doesn't recover state. It can't undo a NaN that's already propagated into the weights, and it
does nothing for representation collapse (falling effective rank with no gradient spike) or
silent optimizer corruption. ARC's rollback is the piece gradient clipping structurally can't
do: restoring to a known-good checkpoint rather than just capping the next update. That said,
our own agent's `enable_grad_clipping` tool is advisory-only today — it recommends a threshold
but can't insert a `clip_grad_norm_()` call into a loop it doesn't own, so the two approaches
are complementary, not competing.
[`ARC_FUNDING_PROPOSAL.md` §1, competitive table; `ARCHITECTURE.md` §6, "advisory because the harness ... cannot insert a clip_grad_norm_ call"]

**Q17. How is this different from just calling `torch.save` periodically yourself?**
Three things a manual checkpoint script doesn't give you for free: automatic *detection*
(NaN/Inf/divergence thresholds firing without you polling loss), automatic *recovery*
(restoring weights and adjusting LR without you writing that logic per project), and signals
you likely aren't logging yourself — effective rank, gradient entropy, update ratio and flow
ratio, charted live. The caveat on that third item is real: only effective rank is wired to a
trigger, and that trigger has never fired in validation. The gradient-entropy rule was deleted
after it cost a healthy run 77 points of accuracy (Q5). So the honest version is "signals you
aren't charting", not "failures we catch before the loss does". The honest caveat: our current checkpoint interval
is fixed (every 10 steps, 3 checkpoints retained) and calls private/underscore-prefixed
`arc-training` APIs wrapped in bare `except: pass`, so if that internal API changes in a future
`arc-training` release, checkpointing can silently go inert while the dashboard keeps looking
healthy. That's a real gap versus a battle-tested manual save loop, and it's on our fix list.
[`ARCHITECTURE.md` §5; `SECURITY_AUDIT.md` §M-5, §M-6, §M-7]

**Q18. What happens if `arc-training` renames or changes those private APIs?**
It used to fail silently, and that is fixed. The harness no longer touches `arc-training`
internals at all: it owns its checkpoint store outright (host-resident, bounded, RNG-preserving),
so no rename upstream can disable rollback. `arc-training` is still used for the structural
signal collectors — all public API — and those calls now report the first exception through
`warn_once`, light a **DEGRADED** badge in the dashboard header, and list the failed component
in the run summary, instead of being swallowed.

Turning that silence off is what found two real bugs in `arc-training` that had been throwing
on every step: a device-mismatch in `WeightCollector` that killed every structural signal on
any CUDA run, and backward hooks in `GradientCollector` that crashed models using
`inplace=True` activations. Both are fixed upstream.
[`SECURITY_AUDIT.md` §M-6, §M-7 — both "Status: fixed"]

**Q19. Could a malicious repo hijack this via a workspace setting?**
That was a real finding and it's fixed now. `arcAgent.pythonPath` previously had no `scope`,
which defaults to `window` — meaning a cloned repo's `.vscode/settings.json` could silently
set it to an arbitrary binary path, and clicking Run would execute it with the user's full
privileges. It's now `scope: "machine"` (workspace settings can't touch it), and the
shell-based `cp.exec` install-check was replaced with argv-array `cp.execFile`, which closes
the associated shell-quoting injection path.
[`SECURITY_AUDIT.md` §H-1, "Status: fixed"]

**Q20. Why does the extension write Python files to disk at runtime instead of shipping them normally?**
Historically the stated reason was marketplace-scanner avoidance, which we've since recognized
and corrected — that document is deleted, because "base64-encode source, exclude it from the
package, decode and execute at runtime" is structurally the same pattern malware uses, and
writing it down as an intentional evasion strategy was indefensible regardless of intent. The
underlying problem (ML tooling trips scanner false positives) is real; evasion was the wrong
answer.

**The mechanism is gone too, not just the document.** The `.py` files now ship as plain,
readable source in the package. `scripts/embed_python.js`, the three base64 constants and the
write-to-globalStorage-then-execute path are all deleted, so the drop-and-execute pattern no
longer exists anywhere in the extension. That is the correct posture for an AGPL project, and
if a scanner flags us now the appeal is "here is the source, here is why it spawns Python" —
which is an argument that wins.
[`SECURITY_AUDIT.md` §C-3; `ARCHITECTURE.md` §2]

**Q21. Does the dashboard ever show fake data?**
It used to, and we caught it ourselves. `enrichEvent()` filled in Effective Rank, Gradient
Entropy, Weight Update Ratio, and Gradient Flow Ratio with `Math.random()`-generated values,
rendered identically to real measurements, whenever the backend didn't send real ones — which
happened silently whenever `arc-training` wasn't installed or the collector attach failed. It's
deleted now: missing data renders as a genuine gap in the chart with a one-time log message
telling the user to `pip install arc-training`, instead of a fabricated line. A test asserts
`Math.random` cannot reappear in that file.

One line of it outlived that fix, and the second review pass caught it: `gradNorms.push(msg.grad_norm ?? 0.001)`
drew an invented `0.001` as a measured point whenever the field was missing, and the stat tile
showed `0.000`. That is not a neutral default — ~1e-3 is the signature of vanishing gradients, so
the fabricated number told a specific and unsupported story. It pushes `null` now, and the tile
shows an em-dash ([H-6](SECURITY_AUDIT.md)). Deleting a bug's main body is not the same as
deleting the bug.

The part we did not understand at the time is *why* it had been added. The advanced metrics
were not occasionally missing — they were missing on **every CUDA run**, because
`arc-training`'s `WeightCollector` threw a device-mismatch error on every single step and a
bare `except: pass` hid it. Someone saw four permanently empty charts and filled them in. Fix
the silent exception and the real numbers appear; those real numbers turned out to be strong
enough to detect failures the loss curve cannot show, which is now our best feature. Deleting
the fake data is what made the real data worth having.
[`SECURITY_AUDIT.md` §C-2 and §M-7, both "Status: fixed"]

**Q22. Checkpointing means you're keeping model copies in memory — what's the overhead cost there?**
It used to be ~9× model size **in VRAM** — three deep copies of model plus Adam's two moment
buffers, all GPU-resident, on a memory budget the tool exists to protect. A tool that can cause
the OOM it warns about is not a reliability tool.

ARC Lens now owns its checkpoint store instead of using `arc-training`'s, and keeps snapshots in
**host** memory. It also estimates its own footprint before the first save and emits a
`checkpoint_budget` event the dashboard displays, so the cost is stated up front rather than
discovered as a crash. RNG state travels with each snapshot, which is what lets a rollback
resume the same data order — and what makes our baseline-vs-active comparison fair.

Owning it also removed a dependency on four underscore-prefixed `arc-training` internals, so an
upstream rename can no longer silently disable rollback while the dashboard renders a healthy
run.
[`SECURITY_AUDIT.md` §M-5, §M-6, both "Status: fixed"]

---

## Business model

**Q23. Is this open source? What's the license?**
Yes — AGPL-3.0. The core `arc-training` package on PyPI and the ARC Lens extension repo are
both under that license. Source maps are no longer generated at all (`sourceMap: false`) and
are excluded from packaging as a second line of defence — that is a security fix rather than a
licensing one, since a `.js.map` embeds the complete original TypeScript next to the code it
maps. Historically the released `.vsix` packages did ship
`out/**/*.js.map` source maps, so the full original TypeScript — including the pro/license
code — is recoverable regardless of AGPL; that's flagged as something to strip from future
release builds, not a licensing violation.
[`README.md` License section; `package.json` `"license": "AGPL-3.0-only"`; `SECURITY_AUDIT.md` §C-1]

**Q24. What's the Pro tier, and is it actually enforced right now?**
Pro ($2.99/mo) is the AI Failure Analyst chat and the ARC Script Generator — both call an LLM
through a user-supplied API key (BYOK). Right now the gate is not enforced: `isPro()` returns
`true` unconditionally in both the current public build and the private original, so Pro
features are unlocked for everyone. We're calling that "unlocked for evaluation" rather than
pretending it's a working paywall — a real gate needs an asymmetric-signature license check
(the current one uses a symmetric secret that's already exposed in the published `.vsix`
files, so it can't be trusted as-is even when re-enabled).
[`arc_lens_business_plan.md` §3–4; `SECURITY_AUDIT.md` §M-10, §C-1]

**Q25. What's the BYOK model and why does it matter for your margins?**
Bring-Your-Own-Key: Pro users paste their own OpenRouter (or OpenAI/Anthropic/Groq/Gemini) API
key, so we never pay for their LLM usage — the $2.99/mo is a pure software fee with no token
liability on our side, which is why the projected net margin is ~85%. The honest caveat is this
is a projection (Year 1 ARR ~$7K scaling to ~$215K by Year 3 in the funding proposal), based on
assumed conversion rates, not actuals — we have no paying users yet to validate the 2-3%
free-to-paid conversion assumption against.
[`arc_lens_business_plan.md` §4, §6; `ARC_FUNDING_PROPOSAL.md` §4.3]

---

## Quick-reference: what's fixed vs. still open

All 36 audit findings are fixed in the repository — 28 from the first pass, 6 from a second
review pass over the remediated code, and 2 from validating that code against a real A/B run.
One item cannot be closed from here.

| Status | Finding | One-line |
| :--- | :--- | :--- |
| Fixed | C-2 | Dashboard fabricated advanced metrics with `Math.random()` — and the upstream crash that caused it |
| Fixed | C-3 | Scanner-evasion framing *and* the base64 drop-and-execute mechanism, both gone |
| Fixed | H-1 | `pythonPath` workspace-override + shell injection |
| Fixed | H-2 | ECharts vendored locally; dashboard renders with no network |
| Fixed | H-3 | Removed the stub-swap scripts entirely rather than guarding them |
| Fixed | H-4 | Public repo now ships the real harness, not a simulation |
| Fixed | M-1 | Anchor moved to `Optimizer.step`; AMP, accumulation and multi-optimizer verified on GPU |
| Fixed | M-2 | One fused `_foreach_norm`, one sync — measured 1.8% / 8.4% overhead |
| Fixed | M-3 | Zero injected lines; tracebacks report the user's real line numbers |
| Fixed | M-4 | Recovery loop no longer sleeps real GPU time |
| Fixed | M-5 | Checkpoints host-resident and budget-reported |
| Fixed | M-6/M-7 | No private-API dependency; failures are visible, with a DEGRADED badge |
| Fixed | M-8 | Nonce CSP, all 22 inline handlers converted |
| Fixed | M-9 | Interpreter resolved via the Python extension |
| Fixed | M-10 | Dead license code and the hardcoded JWT backdoor deleted |
| Fixed | L-1…L-11 | Tail flush, batch-timer race, ring buffer, panel leak, reduce, 91 tests + CI |
| **Owner action** | C-1 | Source side fully clean and CI-guarded; the four *published* `.vsix` files still contain the old secret and need revocation + supersession |

Second review pass, over the code the table above describes as fixed:

| Status | Finding | One-line |
| :--- | :--- | :--- |
| Fixed | C-4 | Plain-function `step` wrapper made *constructing any LR scheduler* raise `AttributeError` before training — bound with `types.MethodType` |
| Fixed | C-5 | An unobservable loss was read as NaN, so ARC froze healthy runs while logging successful interventions — absent is now reported as unknown |
| Fixed | C-6 | Unrecoverable, cooldown and **baseline** branches all zeroed gradients while claiming not to intervene — the A/B control arm was being protected |
| Fixed | H-5 | Wrapper optimizers (Lookahead, SAM) instrumented twice; `_arc_instrumented` guard |
| Fixed | H-6 | `msg.grad_norm ?? 0.001` — the last surviving line of C-2's fabrication; now `null` and an em-dash |
| Fixed | H-7 | Script generator prompted for `import arc` / `arc.wrap(model)`, an API that does not exist; now plain PyTorch |
| Fixed | — | The blind spot C-4 came through: `test_lr_schedulers_can_be_constructed` now builds a real `CosineAnnealingLR` and `ReduceLROnPlateau` through the real runner and fails against the pre-fix code |
| Open | — | `train_demo.py` still sets `group["lr"]` by hand, so the *shipped reference script* — which the benchmark and A/B harness both run — still exercises no scheduler |

Validation pass, from running the code the two tables above describe as fixed:

| Status | Finding | One-line |
| :--- | :--- | :--- |
| Fixed | C-7 | `gradient_entropy_collapse` took a run from 87.43% to 10.00% and then declared it unrecoverable; the signal converges to the same value on healthy and dead runs, so the rule is deleted and structural baselines now wait 200 steps |
| Fixed | C-8 | Three tests were passing on C-7's false positive against a fixture that never diverged — the suite was certifying the bug; fixture rebuilt and verified without ARC, assertion now checks the failure `kind` |
| Open | — | `representation_collapse`, the one remaining structural rule, has never fired in validation (healthy run bottoms at 96% of baseline, damaged at 83%, trigger at 50%) — conservative, and therefore unexercised |

Two things worth saying out loud with this table:

**The most valuable findings were not on the original list, and they made the product smaller.**
Fixing the silent-`except` problem (M-7) exposed that the structural rules were unreachable dead
code — they only ran *after* a NaN. Making them reachable then showed that two of the three were
harmful: the update-ratio rule cost 1.74 and 0.78 accuracy points on healthy runs, and the
gradient-entropy rule took a run from 87.43% to chance and then declared it unrecoverable (C-7).
Both are deleted. Removing the second one also revealed that three integration tests had been
passing on its false positive, against a fixture that never actually diverged — the suite was
certifying the bug (C-8). What survives is numerical divergence detection, which is verified
working, gradient clipping, and one conservative rank rule that has never fired.

**We report the runs ARC cannot save.** `docs/EXPERIMENT_RESULTS.md` includes configurations
where ARC detects the failure and the run still ends at chance accuracy, because a collapsed
network cannot be rolled back to a healthy checkpoint that no longer exists. Knowing that at
step 200 is still worth more than finding out at hour six.
