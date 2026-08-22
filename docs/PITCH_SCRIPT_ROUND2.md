# ARC Lens — Round 2 Pitch Script

**Team Heisen-bug (U333WKR8) · Challenge #171: Accessibility — Resource Waste Reduction**

~1,350 words. 7.5–8 minutes. Slide cues in brackets — not spoken.

Every number traces to `docs/WASTE_REDUCTION.md`, with the method that produced it.

---

## 0:00 — The problem

[Title]

Training runs fail without crashing, and they keep burning GPU after they're already dead.

A learning rate slightly too high sends the loss to NaN. The process stays alive, the GPU stays
at 100%, the job still reports "running." It stopped being training and nothing in the stack
says so.

We measured that gap on our own demo run. Real CIFAR-10, real CNN, no injected failure. ARC
detected the run was dead at **15.3 seconds**. The run continued to **86 seconds**, because
nothing stopped it. **82% of that compute was spent after the answer was already known.**

The seconds are small — it's an 86-second demo. The fraction is the transferable part. On a
48-hour run, 82% is a day and a half spent after the answer was in.

---

## 0:50 — ARC Lens

[Dashboard, live]

ARC Lens is a VS Code extension. Open a PyTorch script, click one button, get live loss,
learning rate, gradient norm and GPU memory while the run happens.

The live chart is not the novelty. TensorBoard has one. Weights & Biases has a better one.

The difference is what happens when the loss goes non-finite. ARC Lens doesn't send an alert.
It restores the model from the last healthy checkpoint held in memory, cuts the learning rate,
latches gradient clipping on, and lets the same run continue. Same process, no restart, no human
in the loop.

Every other tool in this space watches. This one acts.

---

## 1:45 — ARC, the backbone

[Three-tier diagram]

Under the extension is the framework — ARC, Autonomic Recovery Controller. Ships separately on
PyPI as `arc-training`. Three tiers: the extension host, the instrumentation harness, and a
deterministic recovery agent.

The harness monkey-patches PyTorch at `Optimizer.step`. Not `loss.backward()` — the optimizer.
That's the load-bearing choice in the whole system: one recorded step is one weight update. So
gradient accumulation doesn't inflate the step count, mixed precision reports unscaled values,
and a GAN with two optimizers doesn't get confused. Hook `backward()` instead and all three of
those are wrong, not just noisy.

Rollback is the hard part. Checkpoints sit in host RAM in a ring buffer under a byte budget, and
restoring one means writing weights back into a live model mid-run and getting the optimizer to
continue against them — while an LR scheduler rewrites the learning rate every step and tries to
undo our correction. That case is tested.

The user's script runs unmodified through `runpy`, so their tracebacks report their own line
numbers. Zero code changes to integrate — no import, no callback, no decorator.

---

## 3:00 — What the demo run is

[Demo config]

Twenty seconds on what's actually running, because it matters for how you read the next slide.

Nine-layer CNN, 2.8 million parameters, BatchNorm throughout. Real CIFAR-10 — not synthetic, not
a subset. SGD with momentum, cosine decay, no gradient clipping.

The one unusual thing is a peak learning rate of 5.0. And that's not as absurd as it looks: it's
what you get copying a learning rate from a paper that used a different model and a much larger
batch, and keeping your own warmup. It's a mistake people actually make.

**Nothing is injected.** No NaN bomb, no scripted curve. The failure is just what that learning
rate does to this network on this data, and ARC has to detect it rather than be told where to
look.

---

## 3:25 — It actually preserves the compute

[A/B table: 10.00% vs 46.59%]

Same seed, same data order, 1,950 steps. The only difference is whether interventions are
allowed.

The loss goes non-finite at step 6 in both arms — 2.5 times ten to the twelve by the end of the
first epoch in the control.

**The control never comes back.** Five epochs, 85 seconds, final accuracy 10.00%. CIFAR-10 has
ten classes, so 10.00% is random guessing. Every second of that run is waste, and the user's next
move is to restart from step 0.

**The intervened arm rolls back at step 6, cuts the LR, latches clipping, and climbs out.**
18% by epoch 3, 32% by epoch 4, **46.6% by epoch 5** and still rising when the schedule ends.
**Plus 36.6 points against an identically seeded control.**

One caveat I'll say before anyone asks: the recovered arm is 33% *slower* in wall clock, because
recovering costs more than letting a dead run coast. ARC didn't make this run cheaper. It turned
85 wasted seconds into 113 productive ones. The alternative was never "a faster good run" — it
was "restart and hope."

---

## 4:30 — Legibility is the waste reduction

[Resources Conserved panel]

Here's the part that ties the challenge together, and it comes from something ARC *can't* do.

We built four structural detection rules for failures that never produce a NaN. Two were deleted
after an A/B caught them firing on healthy runs. The other two detect reliably but are
report-only, because every response we tried made a recoverable run worse.

So for the silent death — the run pinned at chance accuracy for four epochs — ARC's output is
not a rescue. It's a person being told, in time, in terms they can act on.

Which means the accessibility work isn't decoration around the feature. For that failure, **it
is the entire feature.** The status strip that names the current state and the next step. The
ARIA live region that escalates to assertive only on failure. The preflight that fails with a
named cause instead of a traceback. A data-table equivalent behind every canvas chart. That's
what converts 15 seconds of detection into 70 seconds not wasted.

And it measures both ways. Telemetry was emitting one event per optimizer step. Coalescing it
while the run is healthy cuts stdout by **72%** at zero wall-clock cost, and densifies again when
risk rises. That saving exists *because* the display is driven by what a human needs to see
rather than by the training loop's step rate.

Accessibility and resource waste aren't two deliverables here. They're the same work.

---

## 5:40 — The four states, and the honesty audit

[Status strip, then the provenance table]

The brief asked that success, failure, current status and next steps be clearly visible.

**Status** is the strip: running, at what step, at what risk. **Success** is the resume, marked
on the chart at the exact step. **Failure** is the one most tools skip — after three failed
recoveries of the same kind, ARC declares the run unrecoverable and says so, because the useful
answer then is "kill it," not a fourth rollback. **Next steps** is the Stop button and an
exported incident report.

Then we audited every number the dashboard displays, and labelled each one *measured*, *derived*
or *estimated*, on screen.

That audit found four fabrications. The worst: when a risk score was absent, the gauge rendered
`0.00` — which reads as **safe**. On a gauge whose entire job is saying when a run is not safe,
that's fabricating in the one direction that matters. Fixed, and the test now checks the field's
presence rather than its truthiness, because zero is a legitimate value for lr and GPU memory.

Dollar and energy figures are labelled estimates every time they appear. An unlabelled dollar
amount would be the same class of problem.

---

## 6:35 — How it's built, and against the field

[Test output, then comparison table]

134 tests — 75 Python, 59 TypeScript and dashboard. Ten Python tests run the real harness against
real training loops end to end. Several were written before the code they cover and found real
bugs doing it, including one where an optimizer was matched to a wrapper module instead of the
submodule it updates — that would have rolled back both halves of a GAN. CI fails the build on
any secret-shaped literal. LLM features are bring-your-own-key, so no credential and no token
cost sits on our side.

Against the field: W&B alerts. Comet watches deployed models for drift. Lightning's
`EarlyStopping` stops the run. Composer restarts the whole job after a hardware fault. None of
them correct a numerical pathology mid-flight and let the same run continue.

Two things stated plainly, because a well-read judge will otherwise state them for us.
Rollback-plus-LR-cut is not a new technique — it's documented practice in the OPT-175B and BLOOM
logbooks, done by hand. The contribution is automating it for people without a frontier lab's
monitoring team. And the detector is deterministic thresholds, not ML, which for a live demo is
a feature: fast and fully reproducible.

---

## 7:25 — Close

Half of what we built, we deleted or demoted, because our own measurements said it didn't work.
Every one of those sweeps is in the repo — including the two we killed mid-run and the claims we
withdrew after they failed to replicate.

What survived is narrow and it's real: for the failure ARC can fix, a run goes from chance
accuracy to 46% on the same seed. For the failure it can't, it tells you at second 15 instead of
second 86, and tells you it can't fix it.

Anything that intervenes in someone's training run has to earn it with evidence. Half the time
ours said no.

---

## Rubric coverage

Where each judging parameter gets hit. If a beat gets cut for time, check what it was carrying.

| Parameter | Beat | Carried by |
|:---|:---|:---|
| Task implementation | 3:00, 4:05, 5:15 | Compute preserved, telemetry −72%, a11y 87→100, the four required states named |
| Task complexity | 1:45 | Optimizer-level patch, in-process rollback against a live scheduler, GAN/AMP/accumulation correctness |
| Technical execution | 6:10 | 134 tests, ten end-to-end, TDD-found GAN bug, CI secret scan |
| Innovation & creativity | 0:50, 4:05, 6:10 | Acts instead of alerting; legibility reframed as the waste reduction |
| Functionality & reliability | 3:00, 5:15 | +36.6 points on a seeded A/B; unrecoverable verdict instead of infinite retry |
| Documentation & presentation | 5:15, 7:00 | On-screen provenance for every figure; every sweep published, withdrawals included |
| Architecture | 1:45 | Three tiers, one measurement anchor |
| Code quality | 5:15, 6:10 | Tests written first; the fabrication audit and its regression tests |
| User experience | 0:50, 4:05 | One button, zero code changes, named cause and named fix |
| Scalability | 4:05, 6:10 | Adaptive telemetry, checkpoint byte budget, 1.8% overhead, BYOK zero token cost |
| Technical sophistication | 1:45, 7:00 | The anchor choice, and the A/B methodology that deleted our own rules |

---

## Demo notes

Default `train_demo.py` is now `ARC_DEMO_LR=5.0`, warmup 5. A default run produces **both**
failure kinds: `numerical` around step 6 — ARC acts, and the run climbs off chance — then a
`loss_plateau` later, which ARC deliberately does not act on. Show both. The second is the more
convincing half.

Do **not** demo at `ARC_DEMO_LR=0.5` (the old default). It only produces `loss_plateau`, so the
rescue path is never exercised and nobody sees a recovery.

Dry-run both arms before the slot:

```bash
ARC_MODE=baseline python python/runner.py python/train_demo.py   # ends at 10.00 %
ARC_MODE=active   python python/runner.py python/train_demo.py   # ends at ~46 %
```

---

## Q&A

| Ask | Answer |
|:---|:---|
| Is one seeded pair enough? | No, and we say so. Sweep 6 is why: at lr=0.5 the run-to-run spread swamped effects this size. This case is more defensible — a deterministic numerical explosion at step 6, not a bistable escape regime — but no distribution over repeated runs has been measured and no error bar is claimed. |
| 46% is a bad CIFAR-10 result. | It is. It's a rescued run, not a tuned one. The claim is that the compute was preserved, not that the hyperparameters were fixed. |
| Distributed training? | Single-process, single-GPU today. Multi-optimizer within that process is tested. DDP/FSDP needs rank-aware emission and a barrier so every rank rolls back to the same checkpoint — not run on multi-GPU hardware, so not claimed. |
| Does preflight save time? | No. Measured, and it didn't — `runner.py` surfaces the same errors in about the same time. It saves effort, not seconds: a named cause and a named fix instead of a traceback. Published as a null result. |
| Why report-only instead of acting? | Because the fix couldn't be attributed. At the learning rate where it matters, two identical runs — same seed, same data order — split by 62 points on floating-point nondeterminism alone. |
| Walk me through the ML prediction. | The shipped detector is deterministic thresholds. A learned classifier exists in the core research library; it isn't wired into the live path, so it isn't claimed. |
| Is the telemetry saving on by default? | No. `arcAgent.telemetryEvery` defaults to 1. The −72% is what it buys at 10, and risk detection is unaffected either way — loss history and risk are computed every step regardless. |
| Business model? | Core free — dashboard and recovery. Pro $2.99/month for the LLM failure analyst, bring-your-own-key, so no token cost carried. |
