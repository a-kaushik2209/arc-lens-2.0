# ARC Lens — Measured Results

Everything here is measured on real hardware with real data. No injected failures, no scripted
curves, no synthesised metrics. Where ARC did not help — or made a run worse — that is reported
in the same table as the cases where it did.

**Reproduce:**

```bash
python python/benchmark_overhead.py --steps 200 --repeats 3
python python/experiment_ab.py --lrs 0.03 0.1 0.25 0.5 --epochs 10
```

Raw output: [`benchmark_overhead.json`](benchmark_overhead.json), [`experiment_ab.json`](experiment_ab.json).

`experiment_ab.json` holds only the most recent sweep. Every sweep run against the
loss-plateau rule — including the two that were killed mid-run and the one whose numbers
this document quotes for the pre-fix `lr=0.5` comparison — is listed in
[`SWEEP_LOG.md`](SWEEP_LOG.md), with the superseded results kept alongside it so the
figures below can be checked against the run that produced them.

---

## Setup

| | |
| :--- | :--- |
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop |
| Software | PyTorch 2.6.0+cu124, CUDA 12.4, arc-training 5.0.0, Python 3.12.6 |
| Model | `DemoCNN` — VGG-style, 7 conv blocks + 2 FC, 2.79M parameters |
| Data | CIFAR-10, standard crop/flip augmentation |
| Optimizer | SGD, momentum 0.9, weight decay 5e-4 |
| Schedule | 60-step linear warmup, then cosine decay |
| Run length | 10 epochs = 3,900 optimizer steps per arm |

Nothing is injected. The runs fail — when they fail — because the peak learning rate is too
high for this architecture, which is the most common real cause of a diverged run.

---

## 1. Instrumentation overhead

The same training loop, run with and without the harness. Wall-clock A/B is the only overhead
number worth quoting: the harness can time its own hooks, but that figure charges ARC for GPU
work that was already queued and that the training script's own `loss.item()` would have waited
on a moment later. On one run, self-timing said 54% where wall-clock A/B said 8.4%.

200 steps × batch 128, median of 3 runs:

| Configuration | s/run | ms/step | Overhead |
| :--- | ---: | ---: | ---: |
| bare (no ARC) | 9.818 | 49.09 | — |
| ARC core metrics only | 9.994 | 49.97 | **1.8%** |
| ARC full (structural signals every 25 steps) | 10.640 | 53.20 | **8.4%** |
| ARC full (structural signals every step) | 26.510 | 132.55 | 170.0% |

The last row is why the expensive signals are sampled rather than collected every step, and why
"sample densely while unstable" was capped at 5× the normal rate rather than left at every step
— an unstable run would otherwise spend the longest time paying the worst rate.

---

## 2. Does intervening actually help?

`ARC_MODE=baseline` suppresses every intervention while leaving telemetry, detection, logging
and checkpointing fully active. Both arms therefore execute the *same* instrumented code path,
with the same seed and the same data order, so the only difference between them is whether ARC
was allowed to act. Instrumentation effects cancel; the delta is the intervention.

That claim only became true late. Baseline mode used to call `optimizer.zero_grad()` on a
detected failure — discarding the update, which is itself an intervention, and the single most
consequential one available on a diverging run. Every number measured before that was fixed
compared against a control arm that ARC was quietly acting on. Those numbers have been withdrawn
rather than adjusted; see §3.

**What ARC can act on, as of these results:** a non-finite or exploded loss, and a gradient
norm above 50. That is the whole list.

Four structural rules have been built and none of them survives with the power to act. Two —
weight update ratio and gradient entropy — fired on healthy runs and were deleted (§3, §4). The
other two, loss plateau and representation collapse, detect real failures correctly, but no
measurement shows their responses helping and two sweeps showed failing runs ending far worse
with them applied. They are report-only (§2, §4b), and §4c explains why that call does not rest
on the size of those deltas — at these learning rates a single pair cannot attribute one. They
still detect, chart and report; they do not steer the run.

<!-- RESULTS_TABLE -->

---

## 2b. The failure the structural tier missed entirely

The plateau rule exists because of a run that exposed a hole in everything above it.

A real CIFAR-10 run at `lr=0.5` (seed 1234, 2 epochs, 780 steps) finished at **10.00%
validation accuracy** — chance on ten classes — with its final loss at 2.302725 against
`ln(10) = 2.302585`. The loss fell for about fifteen steps, spiked as the warmup pushed the
learning rate past ~0.08, and from step 25 onward never moved again.

ARC reported **zero failures and zero interventions across all 780 steps**, with the risk score
at `LOW / 0.0` throughout.

Every rule was correct to stay silent, which is the point:

| rule | why it did not fire |
| :--- | :--- |
| non-finite / exploded loss | loss stayed at 2.3027 — finite, and nowhere near 1e6 |
| gradient explosion | gradient norm sat around 0.07, threshold is 50 |
| representation collapse | rank fell only to 87.4% of its step-1 value; trigger needs 50% |

Two separate defects were behind the third row, and both are now fixed.

**The baseline was captured on the corpse.** Structural baselines were taken from the first
samples *after* a 200-step warmup. This run died by step ~45 and its rank flatlined by step 100,
so the "healthy" reference was measured on an already-dead model. Ratios then compared dead
against dead:

| arm | captured baseline | floor | floor / baseline |
| :--- | ---: | ---: | ---: |
| lr=0.03 — healthy, 73.61% | 28.312 | 27.940 | 98.69% |
| lr=0.50 — dead, 10.00% | 25.207 | 25.136 | **99.72%** |

Measured against its own baseline the dead run scored *more stable than the healthy one*,
because a dead model is perfectly steady. Any collapse beginning before step 200 was invisible
by construction. The baseline is now captured from the run's opening samples while the verdict
still waits for the warmup — the two concerns were conflated, and the conflation was the bug.

**Effective rank cannot see this failure at all.** Even with an oracle baseline taken at step 1,
the dead run bottoms at 87.4% against the healthy run's 97.2% — real separation, about ten
points of it, and a trigger sitting four times further away. That is not a threshold to tune.
`mean_effective_rank` is the SVD entropy of the *weight* matrices, and a network can emit a
nearly constant output while every weight matrix stays well-conditioned. That is exactly what
happened: weight norm halved from 147.9 to 71.4 while rank held. Weight conditioning is not
representational rank, and they diverge precisely in this failure mode.

### The signal that does separate

Replaying both arms' per-batch losses through a best-loss-with-patience counter:

| arm | max consecutive steps without improvement |
| :--- | ---: |
| lr=0.03 — healthy, 73.61% | 82 |
| lr=0.50 — dead, 10.00% | **764** |

A 9.3x gap, against 1.1x for effective rank and none at all for gradient entropy. Patience is
set to 300 — 3.7x above the measured healthy maximum rather than just above it, because 82 is
one seed on one workload and a noisier task will stall longer while training perfectly well.

Re-running both arms with the rule in place:

| arm | best val acc | failures | interventions |
| :--- | ---: | ---: | :--- |
| lr=0.03 — healthy | 74.67% | 0 | none |
| lr=0.50 — dead | 10.00% | 2 | step 330 LR 3.46e-01→1.73e-01; step 630 LR 2.58e-02→1.29e-02 |

A 1-epoch repeat fired at **step 316**, matching the offline replay's prediction for patience
300 exactly.

### And then the longer A/B said the rule was wrong

Everything above was measured on **780-step** runs. Extending to 10 epochs — 3900 steps — broke
it on the first arm:

```
lr=0.03 baseline -> best_val_acc=87.5  failures=2  kinds=loss_plateau
```

A run reaching **87.5%** tripped the rule twice. No harm followed, because the baseline arm
suppresses interventions, but the detector fired on a plainly healthy run — which is exactly
what got the update-ratio and gradient-entropy rules deleted.

**Patience cannot fix this, and that is the important part.** The counter keys off the
*best-ever* batch loss. As a run converges, its own record gets harder to beat, so the stalls
grow without bound — convergence *is* a plateau. There is no patience value that separates a
converged run from a dead one, because the stall length on a successful run is unbounded. The
780-step measurement of "82" was not wrong; it was just far too short to see the shape of the
thing.

What separates them is whether the run ever got anywhere:

| arm | first loss | best loss | best / first |
| :--- | ---: | ---: | ---: |
| lr=0.03 — healthy | 2.3018 | 0.6233 | **0.271** |
| lr=0.50 — dead | 2.3221 | 2.0632 | **0.888** |

A dead run stalls having never improved. A converged run stalls having improved enormously. The
rule now requires both conditions — stalled 300+ steps **and** `best/first > 0.60`. That test
needs no knowledge of the class count, or even that the task is classification, which
"loss near ln(num_classes)" would have.

This is the third time a measurement has gone against a rule that was already written, and the
second time it happened *after* the rule was committed and documented. That is the argument for
the A/B existing at all: the rule looked correct, the reasoning was sound, the short-run evidence
supported it, and it was still wrong.

### And then the sweep said the *response* was wrong too

The rule went into the four-learning-rate sweep with `reduce_lr` as its response. Six arms
behaved: `lr=0.03`, `0.1` and `0.25` all finished 86.8–87.8% with the rule silent in both arms,
which is what the progress guard was added to achieve. The `lr=0.5` pair did not.

| epoch | baseline — no action | active — 3 × `reduce_lr` from step 316 |
| ---: | :--- | :--- |
| 1 | 10.00%, lr 4.91e-01 | 10.00%, lr 2.45e-01 |
| 2 | 10.00%, lr 4.58e-01 | 10.00%, lr 1.14e-01 |
| 4 | 9.76%, lr 3.34e-01 | 10.00%, lr 4.18e-02 |
| 5 | **26.73%**, lr 2.56e-01 | 10.00%, lr 3.20e-02 |
| 10 | **73.19%** | **10.00%**, loss 2.3026 = ln(10) |

**Delta: −63.19pp, in ARC's disfavour, from a detection that was entirely correct.** Both arms
really were pinned at chance for four solid epochs; there was nothing wrong with the diagnosis.

> **Withdrawn as a causal claim — see §4c.** A later sweep produced a 62.58pp gap on this same
> configuration with *no* intervention in either arm. The number below is real; the attribution
> to ARC is not established. The conclusion drawn from it — that the rule should report and not
> act — survives, for a stronger reason.

The proposed mechanism was visible in the LR column. The baseline's cosine schedule walked the
learning rate down on its own, and somewhere around 2.5e-01 the run escaped the dead region and
climbed to 73%. The active arm had been cut to 3.2e-02 by that point and never escaped. The
reading at the time was that large steps were the only thing that could carry the weights out,
so reducing the learning rate at the moment of the plateau removes the one mechanism that was
going to fix it.

That reading is plausible and it is not proven. It was argued here that the epoch trace
established causation where the magnitude alone could not — a deterministic schedule driving a
visible recovery over epochs 5–10, against an intervened arm an order of magnitude below the
escape value. §4c shows why that was not enough: the same configuration, with nothing
intervening in either arm, splits by 62.58 points. Escape at this learning rate is bistable, and
a single pair cannot tell a bad intervention from a bad coin flip.

So `loss_plateau` is now **report-only**: it detects, it says so, and it changes nothing.

Rolling back instead is no improvement. Confirming a plateau takes 300 stalled steps and the
network dies around step 45, so the verdict lands ~285 steps too late for any checkpoint in the
ring to be worth restoring — restoring one returns the model to the state it is already in and
spends a recovery attempt to do it. Neither available response helps, and §4 already sets the
standard: a rule may act only once a measured trajectory shows the action helps. There is no
such measurement for a plateaued run, and there is now a measurement against one.

**Reporting is not automatically passive**, which is the part that nearly slipped through.
`_handle_failure` ends by calling `optimizer.zero_grad()` so that gradients which produced a NaN
cannot reach the weights. On a report-only path that would discard the user's update — on a
diverging run, the single most consequential intervention available, and one that would have
made "changes nothing" false while looking passive in the log. The report-only gate therefore
sits above that call, and above the cooldown and baseline gates, so both arms run identical
code for this kind. Three of the six regression tests fail if the gate is removed.

### What this does not do

**Accuracy is still 10.00%.** ARC sees the failure and does not reverse it — now by design
rather than by limitation. The claim this earns is *"ARC reports a silent death instead of
showing a green dashboard for 780 steps"* — not *"ARC recovers it."*

That is the honest ceiling of this rule, and the reason `grad_flow_ratio` is the highest-value
open item: it separated the same pair 266 steps earlier (healthy 1.36–3.10 throughout, dead
50.11 at step 50 and non-finite from step 75), which is early enough that the ring buffer still
holds a pre-collapse checkpoint. Detection that early is the only route to an intervention worth
attempting.

---

## 3. What the first run found, and what it changed

The detector was tuned *because of* earlier runs of this same experiment, and the sequence is
worth recording because it is the argument for having built the A/B at all.

That first sweep produced what looked like a decisive win at `lr=0.25` — baseline 76.19%,
active 83.53%, **+7.34pp** — and a **1.74-point loss at `lr=0.1`**, on a run whose baseline
reached 87.86% and plainly needed no help. ARC intervened once and made a healthy run worse.
Without a control arm that would have been invisible: the log showed a detected failure and a
successful intervention, which reads like the tool working.

> **The +7.34pp figure has since been withdrawn, and the reason is instructive.** It was
> measured before finding that baseline mode called `optimizer.zero_grad()` on every detection
> (C-6). Suppressing an update is an intervention — so the control arm was not a control. On a
> run that did not need rescuing, discarding updates is straightforward *harm*, which means
> that number was partly ARC crippling its own baseline and then taking credit for the gap.
>
> In the final sweep, with the control arm genuinely untouched, the same `lr=0.25` configuration
> reached **87.43% with zero failures detected**. The learning rate we had been calling
> "damaged" was not damaged; our instrumentation was.
>
> This is the strongest single argument in this document for measuring rather than asserting —
> and for treating a flattering result with the same suspicion as an unflattering one. The
> harmful `lr=0.1` result was investigated immediately; the favourable `lr=0.25` result sat
> unexamined for far longer, and it was the one that was wrong.

A second sweep, with a first attempt at a fix, made it **worse**: at `lr=0.03` — a run that
finished at 87.24% and could not have been healthier — ARC intervened once and cost 0.78 points.
The attempted fix had required a *minimum* improvement over the trend window, and real training
at that point improves by roughly 1% per 60 steps, so "not improving fast enough" fired on a
model that was learning perfectly well.

That second failure is the more useful one, because it showed the guard was not mistuned but
**unusable in principle**. Per-batch loss on this task is noisy at roughly 40% of its own mean.
Averaging 20 samples leaves about 14% noise. The signal being measured — real improvement over
60 steps — is about 1%. No choice of window or margin recovers a 1% signal from 14% noise; the
comparison is a coin flip, and every wrong flip is an intervention on a healthy run.

Three defects behind it, all now fixed:

1. **The progress guard was estimating a trend from three noisy samples.** A per-step
   mini-batch loss bounces by a large fraction of its own value, so a 10-step window made
   "is the loss improving?" close to a coin flip — and every wrong flip is an intervention on
   a healthy run. The window is now 60 steps, averaged in thirds, and requires a *measurable*
   improvement rather than merely "not worse".
2. **Representation collapse was not gated on progress at all.** Effective rank falls during
   healthy training too; that is what specialisation looks like. It is only a collapse when
   capacity is lost and nothing is gained for it.
3. **Adaptive sampling densified to every step under elevated risk** — the 170% regime in the
   table above, applied for the longest on exactly the runs already in trouble. Capped at 5×
   the normal rate.

The guard that resulted has to make one distinction: large steps *with* progress (healthy —
leave it alone) versus large steps *without* progress (diverging — intervene). Both cases are
present in this table, which is what makes it a test rather than a demo.

---

## 4. The entropy rule destroyed a healthy run

The final sweep produced the worst result in this document, and it is the one worth reading
first.

At `lr=0.25`, with the control arm finally clean, the two arms went:

| end of epoch 1 | baseline | active |
| :--- | ---: | ---: |
| train accuracy | 19.19% *(→ 88.54% by epoch 10)* | 11.49% *(→ 9.91%, flat forever)* |
| learning rate | 2.45e-01 | **3.07e-02** |
| final val accuracy | **87.43%** | **10.00%** — chance |

ARC raised `gradient_entropy_collapse` at **step 125**, inside the first epoch, and applied
three rollbacks with an LR cut each. The baseline arm proves what the run was doing at that
moment: learning normally, 19% and climbing. The detection was a false positive, and the
response — an 8× learning-rate cut in the first epoch — killed the model outright. It then
correctly reported the run as unrecoverable, which it was, because ARC had made it so.

**The cause is the same mistake as the update-ratio rule, in a different signal.** Gradient
entropy is at its highest at initialisation, when gradients are random and unstructured, and
falls by orders of magnitude as soon as the model begins to learn anything. The detector
captures its baseline from the run's opening samples — steps 1, 25 and 50 — which sit inside
exactly that transient. Normal early learning therefore reads as a 100× "collapse" against a
baseline measured before learning started.

### No threshold would have saved it

The obvious response is to tighten the threshold and move the baseline. Measuring the actual
trajectory first — sampled every 10 steps over two epochs — showed that would not have worked:

| step | `lr=0.25` — healthy, 87.4% | `lr=0.50` — dead, 10% |
| ---: | ---: | ---: |
| 1 | 2.95e-01 | 2.95e-01 |
| 30 | 1.09e-02 | 1.26e-01 |
| 70 | **1.44e-05** | **1.45e-05** |
| 200+ | 1.44e-05 | 1.29e-05 |

The healthy run and the dead run **settle to the same value**. After roughly step 70 the signal
carries no information about run health at all, so no threshold and no baseline window can
separate the cases.

The cause is upstream. `GradientCollector._compute_entropy` bins a heavy-tailed gradient
distribution with `torch.histc` on a linear scale: a few outliers set the range, essentially all
mass falls into one bin, and the normalised entropy saturates near zero — for any run. It is
measuring outlier spread, not information content. Making it useful needs log-magnitude binning
*and* fresh evidence that the result separates the cases; until someone does that, it is a chart
line, not a trigger.

So the rule was deleted rather than retuned, and the structural checks now wait 200 steps before
capturing any baseline.

This is the second rule in a row where a signal's *natural* early trajectory resembles the
pathology it is supposed to detect. That is now treated as the default failure mode for this
class of detector rather than a coincidence, and it is why a new rule needs a measured
trajectory showing separation on both a healthy and a failing run before it is allowed to act.

### The test suite was certifying the bug

Removing the rule made three integration tests fail — and the cause was not a regression.

The fixture those tests used, `SCRIPT_DIVERGE`, **never diverged**. Its loss peaks at 1.93 in
plain PyTorch and never approaches the 1e6 threshold. The tests passed because the entropy false
positive fired on it, and `assertTrue(failures)` cannot tell a real detection from a spurious
one. The suite had been confirming that ARC detects divergence by observing ARC hallucinate one.

That is worse than the original defect: a wrong detector is a bug, a test that certifies it is a
process failure. Both are fixed — the fixture is now verified to exceed 1e6 at step 10 in plain
PyTorch *before* being used to test detection, and the assertion checks the failure `kind` rather
than merely that something fired.

---

## 4b. The rank rule fired for the first time, and the arm that acted lost 44 points

The sweep run to *confirm* the plateau fix found the same defect in the last structural rule
that could still act. This is the shipped sweep — the one in `experiment_ab.json`.

| peak LR | baseline | active | interventions | ARC effect |
| ---: | ---: | ---: | ---: | ---: |
| 0.03 | 87.53% | 87.22% | 0 | −0.31pp |
| 0.10 | 87.49% | 87.73% | 0 | +0.24pp |
| 0.25 | 10.00% | 10.00% | 0 | **0.00pp** |
| 0.50 | 75.18% | 30.84% | 3 | **−44.34pp** |

**The plateau fix worked.** `loss_plateau` fired at step 316 in both `lr=0.5` arms and at steps
315/330 in both `lr=0.25` arms, and took no action in any of them. `lr=0.25` is the cleanest
demonstration: both arms collapsed to chance and landed on *exactly* the same number, which is
what a report-only detector must produce — no arm difference is possible when nothing is done.

**`representation_collapse` did the damage instead.** It had never fired in any previous
validation run. Here it fired in both `lr=0.5` arms, and the arm allowed to act on it was
rolled back and cut three times:

| epoch | baseline (no action) | active (3 × `rollback_and_reduce_lr`) |
| ---: | :--- | :--- |
| 3 | 10.00%, lr 4.04e-01 | 10.00%, lr 4.04e-01 |
| 4 | **19.35%**, lr 3.34e-01 | 10.00%, lr 3.34e-01 |
| 5 | 28.32%, lr 2.56e-01 | 21.44%, lr **3.20e-02** |
| 10 | **75.18%** | **30.84%** |

> **Withdrawn as a causal claim — see §4c**, on the same grounds as §2's: the following sweep
> split this configuration by 62.58 points with no intervention in either arm.

The mechanism proposed was identical to §2's plateau finding, which is what made it look like a
pattern rather than a one-off — and that remains the best argument for the conclusion, just not
for the number. The control arm sat at chance for three epochs and escaped
when cosine decay lowered the learning rate on its own. The intervened arm was cut to an order
of magnitude below the value at which that escape happened, and never got back. The rollback
contributes nothing either: every checkpoint in the ring is from inside the collapsed region,
which is where the model already is.

**So no structural rule acts any more.** Four rules have now been given the power to steer a
run on a structural signal. Two fired on healthy runs and were deleted; two damaged failing
runs and are now report-only. That is a complete record of the attempts, not a selection of
them, and the pattern across all four is the actual result:

> A signal that distinguishes a healthy run from a failing one still does not tell you what to
> do about it. Every response available here — lower the learning rate, restore a checkpoint —
> assumes the run needs to be slowed down or moved back. A run stuck at chance needs neither.
> It needs large steps to escape, which is exactly what both responses remove.

What still intervenes is a non-finite or exploded loss and a gradient norm above 50 — both
verified working, both reading the loss and gradient directly rather than a structural proxy.

**One caveat on the `lr=0.25` pair, stated because it cuts against the table.** In the previous
sweep that configuration finished at 86.83% / 87.75%; here both arms collapsed to chance. Same
seed, same data order, same code. That is the run-to-run variance described in §5 arriving at
full force, and it is the reason the `0.00pp` delta on that row is the honest number to quote
rather than evidence of anything. The `−44.34pp` row is different in kind: the two arms diverge
*within* the run, at the step the intervention lands, and the per-epoch trace shows it.

---

## 4c. The correction: that delta was not attributable

The sweep run to confirm §4b's fix produced the most important result in this document, and it
goes against §2 and §4b.

With **both** structural rules report-only, no arm in the sweep intervened at all — the fix
works. And the `lr=0.5` pair still split:

| epoch | baseline | active |
| ---: | :--- | :--- |
| 2 | train_loss 2.3110, 10.00% | train_loss 2.3110, 10.00% |
| 3 | train_loss 2.3105, 10.00% | train_loss 2.3105, 10.00% |
| 4 | train_loss 2.3098, 10.00% | train_loss 2.3098, 10.00% |
| 5 | **2.1726, 21.07%** | 2.3091, 10.00% |
| 10 | **72.58%** | **10.00%** |

Identical code, identical seed, identical data order, nothing intervening in either arm, and
their first four epochs agree to four decimal places. Then one escaped the dead region and the
other never did — **62.58 points apart, from floating-point nondeterminism alone.**

**So the −63.19pp and −44.34pp deltas in §2 and §4b are withdrawn as measurements of ARC's
effect.** They are real numbers from real runs and the per-epoch traces really do show the
intervened arms sitting far below the learning rate at which the control escaped. But a gap of
the same magnitude occurs with no intervention at all, so a single seeded pair cannot separate
the two explanations. Every `lr=0.5` arm run to date:

| sweep | arm | interventions | escaped? | final |
| :--- | :--- | ---: | :--- | ---: |
| 3 | baseline | 0 | yes | 73.19% |
| 3 | active | 3 (`reduce_lr`) | no | 10.00% |
| 5 | baseline | 0 | yes | 75.18% |
| 5 | active | 3 (`rollback_and_reduce_lr`) | partly | 30.84% |
| 6 | baseline | 0 | yes | 72.58% |
| 6 | active | 0 | **no** | 10.00% |

Three of four untouched runs escaped and neither intervened run did, which is *consistent with*
the responses hurting and *not sufficient to establish it*.

**This strengthens the report-only decision rather than weakening it.** At the learning rates
where an intervention would matter, the run-to-run spread is larger than any effect a single
pair can measure — so a response cannot be validated this way at all, in either direction.
Letting a rule act on evidence that cannot exist yet is precisely the mistake this document
records four times.

What it would take: `python/repeatability.py --lr 0.5 --repeats N`, reported as a distribution
rather than a pair. That has not been run, and nothing here depends on it.

The false-positive fix is untouched by this. It was measured across six healthy arms in two
independent sweeps, and the rule stayed silent in every one.

**§5 predicted this exactly** — "the regime where interventions matter is by definition near the
edge of stability, and that is precisely where tiny numerical differences amplify." It was
written before the run that demonstrated it, and it was still not enough to stop a causal claim
being made from a single pair. That is the more useful lesson than the number.

---

## 5. Honest caveats

**Run-to-run variance, and why it is the biggest threat to this table.** Both arms share a
seed, but a seed does not fix the arithmetic: cuDNN selects non-deterministic kernels and CUDA
reductions are not associative. Two "identical" runs are not identical.

That matters more than it first appears. On the pairs where ARC takes no action the spread is
small — 0.45pp and 0.29pp — which looks like a comfortable noise floor. But those are the
*stable* learning rates. The regime where interventions matter is by definition near the edge
of stability, and that is precisely where tiny numerical differences amplify.

We have direct evidence of this. An earlier sweep measured `lr=0.25` baseline at **76.19%**;
the same configuration in the final sweep measured **87.43%** — an 11-point gap between runs
that were supposed to be the same. Part of that is a real bug we fixed in between (baseline
mode was discarding updates — see §3), but we cannot currently apportion how much, and that
is the honest answer.

`python python/repeatability.py --lr 0.25 --repeats 4` runs one configuration repeatedly and
reports the spread, so the A/B has an error bar instead of an implied one. **Any delta smaller
than that spread is not evidence of anything.** A comparison without it is unfalsifiable, and
the whole point of building the control arm was to avoid unfalsifiable claims.

**One architecture, one dataset, one GPU.** These numbers describe a 2.79M-parameter CNN on
CIFAR-10 on a laptop 3050. A transformer with AMP and gradient accumulation would exercise
different parts of the harness — those paths are unit- and integration-tested, but they are not
represented in this table.

**Best validation accuracy, not final.** `best_val_acc` is the maximum across epochs. For a run
that collapses partway through, this *flatters* the baseline, because it credits the accuracy
reached before the collapse. The comparison is therefore conservative in ARC's disfavour.

**Ten epochs is short.** A production run is hours or days. The failure modes ARC targets are
more likely, not less, over a longer horizon — but that is an argument, not a measurement, and
it is not what this table shows.
