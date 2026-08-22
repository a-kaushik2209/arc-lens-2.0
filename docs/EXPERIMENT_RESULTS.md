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

**What ARC can act on, as of these results:** a non-finite or exploded loss, a gradient norm
above 50, a loss plateau sustained past 300 steps, and a representation collapse below half the
run's own baseline effective rank. Two other rules were built and deleted after measurement,
after they demonstrably harmed healthy runs. The rank rule has never fired in validation, so it
is untested in both directions and is described that way rather than claimed.

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

### What this does not do

**Accuracy is still 10.00%.** ARC now sees the failure and cannot reverse it. Confirming a
plateau takes 300 stalled steps and the network dies around step 45, so the earliest possible
verdict lands ~285 steps too late for any checkpoint in the ring to be worth restoring. That is
why the response is `reduce_lr` rather than a rollback: restoring a post-collapse checkpoint
returns the model to the state it is already in and spends a recovery attempt to do it.

The claim this earns is *"ARC reports a silent death instead of showing a green dashboard for
780 steps"* — not *"ARC recovers it."* The 9.3x separation is also one seed on one workload and
has not yet been through the four-learning-rate sweep, which is the standard the two deleted
rules failed.

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
