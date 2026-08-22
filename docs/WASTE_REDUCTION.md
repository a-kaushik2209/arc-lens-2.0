# Waste Reduction — Measured

FAR AWAY 2026, Challenge #171 (*Accessibility: Resource Waste Reduction*).
Team Heisen-bug (U333WKR8).

This document exists because the project's standing rule is **no claim without a
control arm**. Every number below was produced by running both arms on the same
machine in the same session. Where the win is small, or absent, that is what is
reported. Numbers that were not measured are marked as not measured rather than
estimated.

---

## The argument: legibility *is* the waste reduction

The challenge asks for the part of the MVP most related to accessibility to be
improved so that it cuts waste. Those read like two separate deliverables — make
it accessible, and separately make it cheaper. In this product they are the same
work, and the measurement is what shows it.

**Start from the thing ARC cannot do.** Four structural detection rules have been
built here. Two were deleted for firing on healthy runs; the two that remain were
stripped of the power to act, because every response that was tried made a
recoverable run worse — and a later sweep showed the run-to-run spread at those
learning rates is larger than any effect a single A/B pair can measure
([`SWEEP_LOG.md`](SWEEP_LOG.md)). ARC detects a silent death reliably. It cannot
fix one.

**So the product's output is not a rescue. It is a human being told, in time.**
That reframes what the accessibility work is for. A status strip that names the
current state and the next step, an ARIA live region that escalates to assertive
only on failure, a preflight that fails with a named cause instead of a
traceback, and a data-table equivalent for every canvas chart — none of that is
decoration around the real feature. For a failure ARC has decided not to act on,
**it is the entire feature.**

**And the waste it saves is measurable, because the gap it closes is.** In a
baseline run at `lr=3.0`, ARC knew the run had failed at **3.81 s**. The run
continued to **50.96 s**, because nothing stopped it — **92.5 % of that run's
compute was spent after the answer was already known** (§3, reproduced at 89.8 %
on a second run). Nothing about that gap is a detection problem. It is entirely a
question of whether the person watching can tell what happened and what to do,
fast enough to act — which is the definition of the accessibility work.

The same holds in the other direction. The telemetry that feeds the dashboard was
emitting one event per optimizer step; coalescing it while healthy cuts stdout by
**72.2 %** at **zero** wall-clock cost (§2). That is a bandwidth and storage
saving that exists *because* the display is driven by a policy about what a human
needs to see, rather than by the training loop's step rate.

**What this section is not claiming.** That the 90 % figure is money already
saved — it is the size of the opportunity the Stop button addresses, on a
53-second demo, and it scales with run length rather than being large in absolute
terms here. That preflight saves time — measured, and it does not (§1). That the
telemetry saving is switched on — it is not, by default (§2). Those three are
recorded as prominently as the wins, because a submission whose favourable
numbers are checkable and whose unfavourable ones are volunteered is the only
kind worth putting in front of a judge who will check.

---

## Provenance of every figure the dashboard displays

Audited value by value on 2026-08-22. The standard is the one this project set
for itself in [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) C-2, where four charts
were found rendering `Math.random()` output styled identically to measurements:
**a number that looks like a measurement must be one, and an absent value must
read as absent.** The same table is in the dashboard itself, under *How each
figure is arrived at*, so a reader can check the claim without leaving the page.

| Figure | Basis | How it is arrived at |
|:---|:---|:---|
| Loss, gradient norm, LR, GPU memory, step, epoch | **measured** | Read straight off the harness event. Absent fields render `—`, never `0` |
| Risk score and label | **measured** | Computed by `_risk_score()` in the harness. An unscored event renders `score —` |
| Steps not re-run | **measured** | Highest reported step minus the steps the rollback discarded, parsed from the intervention's own detail line |
| Time not re-spent | **derived** | Wall clock since run start × the fraction of steps that survived the rollback |
| Compute not re-bought | **estimate** | Time × hourly rate. Your configured `arcAgent.gpuHourlyRate` if set, else a published on-demand list price matched to the reported GPU. Source named on screen |
| Energy | **estimate** | Time × tier-typical board wattage. No power sensor is read |
| Telemetry not sent | **estimate** | Suppressed emissions (exact) × 277 B (measured mean, §2: 108,053 ÷ 390) |
| Checkpoint storage | **measured** | Byte counts reported by the harness's own checkpoint store |

**Four fabrications were found and fixed in this audit**, all of them the C-2
pattern rather than deliberate invention:

* an absent `lr` rendered `0.00e0` and an absent `gpu_mem_mb` rendered `0.0 MB`.
  Both fields also have legitimate zero values — the cosine schedule ends at
  `lr=0`, a CPU run reports 0 MB — so the test now checks the field's
  *presence*, not its truthiness.
* an absent risk score rendered `score 0.00`, which reads as **safe**. That is
  fabricating in the one direction that matters on a gauge whose purpose is to
  say when a run is not safe.
* the telemetry tile displayed a number with **no stated basis at all**, and
  could over-report without bound on a long run because it derived the
  suppressed-emission count from an array that is pruned at 100,000 entries.

**What is deliberately not claimed.** The dollar and energy figures are
assumptions and are labelled as such every time they appear — an unlabelled
dollar amount would be the same class of problem C-2 was about. The telemetry
figure is the only tile whose *magnitude* is estimated rather than counted, and
it says so on screen; when nothing is coalesced it reads `0 B` with an
explanation, so a zero tile is legible as "switched off" rather than "broken".

---

## Test environment

| | |
|:---|:---|
| GPU | NVIDIA GeForce RTX 4060 Laptop |
| torch / CUDA | 2.10.0+cu128 / 12.8 |
| arc-training | 2.0.0 |
| Python | 3.12.7 (anaconda3) |
| Workload | `python/train_demo.py` — DemoCNN, 2,788,042 params, CIFAR-10, batch 128 |
| Dataset cache | warm (already downloaded) for all timing runs |
| Lighthouse | 13.4.1, desktop preset, `navigation` mode |
| Date | 2026-08-22 |

The demo workload is small and the dataset cache is warm. That makes the
absolute seconds here *smaller* than a realistic workload would produce, not
larger — the fractions are the transferable part.

---

## 1. Preflight Doctor — repeated-effort waste

Before launching, ARC Lens runs two checks in parallel: one subprocess importing
`torch` / `arc` and probing CUDA, and one `py_compile` of the training script.

| Arm | Method | Result (n) |
|:---|:---|:---|
| Preflight import probe | `python -c "<probe>"` | **1.60 s** (1.58–1.62, n=5) |
| Preflight syntax check | `python -m py_compile` | **0.02 s** (n=5) |
| No preflight, syntax error | full `runner.py` launch until the error surfaces | **1.58 s** (1.57–1.59, n=3) |
| No preflight, missing torch | full `runner.py` launch until the error surfaces | **0.02 s** (n=3) |
| Time to reach training step 1 | `runner.py` on a healthy run, warm cache | **3.30 s** (3.20–3.36, n=3) |

**Honest reading: on this machine the preflight does not save meaningful wall
clock.** `runner.py` imports torch and compiles the user's script early, so a
syntax error or a missing interpreter already surfaces in about the same time
the probe takes. The missing-torch case is a wash at 0.02 s either way.

What preflight actually buys is not seconds, it is *what the user is handed*:

- **Without it:** a Python traceback, and the user has to work out which of
  interpreter / torch / arc / CUDA / syntax was the problem.
- **With it:** a named cause and a named fix, before anything spawns.

The seconds argument only becomes real as the gap between launch and step 1
grows — 3.30 s here on a warm CIFAR-10 cache and a 2.8M-param model, but that is
the number that scales with dataset size, model size, and a cold cache. A
misconfiguration that only surfaces *at* step 0 wastes that entire window, and
the window is minutes on a real workload. We did not measure a real workload, so
we are not claiming a figure for one.

---

## 2. Adaptive telemetry — bandwidth and storage waste

Metric emission can be coalesced while a run is healthy and densified while risk
is elevated (`ARC_METRIC_EVERY` / `ARC_METRIC_EVERY_DENSE`). Measured as total
stdout bytes over a fixed 390 steps (one epoch), same seed, same workload.

| `ARC_METRIC_EVERY` | Dense rate | `metric` events | stdout bytes | Reduction | Wall clock |
|:---|:---|---:|---:|---:|---:|
| 1 (shipped default) | 1 | 390 | 108,053 | — | 17.54 s |
| 5 | 1 | 79 | 40,090 | **−62.9 %** | 17.33 s |
| 10 | 2 | 40 | 30,019 | **−72.2 %** | 17.62 s |

Two things this measurement establishes:

1. **The reduction is real and large** — roughly a 2.7× to 3.6× cut in telemetry
   volume over the wire and in any stored run log.
2. **It is free.** Wall clock is flat across all three arms (17.3–17.6 s, inside
   run-to-run noise). Coalescing gates only the *emit* call; loss history and the
   risk score are still computed every step, so risk detection is not delayed.

**Gap, reported not fixed:** `ARC_METRIC_EVERY` defaults to `1`, and it is not
exposed as a VS Code setting in `package.json`. A default install therefore gets
**0 % of this reduction**. The mechanism is built, tested, and measured — it is
simply not switched on by default, and turning it on is a product decision about
dashboard chart resolution, not a code change. Risk score stayed `LOW` for the
whole 390-step window in these runs, so the densify path was never exercised;
its behaviour under sustained elevated risk is untested.

---

## 3b. The A/B that shows compute actually preserved

Sections 2 and 3 measure waste *avoided*. This one measures a run *saved* — the
only path where ARC acts, exercised end to end. Default settings of
`train_demo.py` as of this commit (`lr=5.0`, warmup 5, 5 epochs), both arms same
seed and same data order, 1,950 steps each. Measured 2026-08-22:

| | `baseline` (control) | `active` |
|:---|---:|---:|
| First failure | `numerical` @ step 6, 4.19 s | `numerical` @ step 6, 4.35 s |
| Interventions applied | **0** | **2** — `rollback_and_reduce_lr`, `enable_grad_clipping` |
| Epoch 1 train loss | 2.52e+12 | 4.22e+05 |
| Final val accuracy | **10.00 %** — chance | **46.59 %** |
| Wall clock | 85.27 s | 113.49 s |

Per-epoch validation accuracy:

| epoch | baseline | active |
|---:|---:|---:|
| 1 | 10.00 % | 10.00 % |
| 2 | 10.00 % | 10.00 % |
| 3 | 10.00 % | **18.28 %** |
| 4 | 10.00 % | **32.09 %** |
| 5 | 10.00 % | **46.59 %** |

The loss goes non-finite at step 6 in both arms — 2.5 × 10¹² by the end of the
first epoch in the control. The control never comes back: five epochs, 85
seconds, and a model at random-guess accuracy. **Every second of that run is
waste**, and the user's next move is to restart it from step 0.

The intervened arm rolls back to the last healthy checkpoint at step 6, cuts the
learning rate, latches gradient clipping on, and climbs out — 46.59 % by epoch 5
and still rising when the schedule ends. **+36.59 points against an identically
seeded control**, and the compute that produced it was not re-spent.

**The honest accounting.** The active arm is 28.2 s slower (113.49 vs 85.27,
+33 %), because recovering a run costs more than letting a dead one coast: the
clipping is real work and the rolled-back steps are re-run. That is the right
comparison to make out loud — ARC did not make this run cheaper, it made
85 wasted seconds into 113 productive ones. The alternative was not "a faster
good run", it was "restart from scratch and hope".

**Caveats, both real.** This is a single seeded pair, and
[`SWEEP_LOG.md`](SWEEP_LOG.md) sweep 6 is the reason that matters: at `lr=0.5`
run-to-run spread swamped effects of this size. This case is more defensible
than that one — the failure is a deterministic numerical explosion at step 6
rather than the bistable escape-or-not regime — but a distribution over repeated
runs (`python/repeatability.py`) has not been measured and no error bar is
claimed. And 46.59 % is not a good CIFAR-10 result; it is a rescued run, not a
tuned one. The claim is that the compute was preserved, not that the
hyperparameters were fixed.

**Reproduce it:**

```bash
ARC_MODE=baseline python python/runner.py python/train_demo.py   # ends at 10.00 %
ARC_MODE=active   python python/runner.py python/train_demo.py   # ends at ~46 %
```

---

## 3a. The demo run — a silent death, caught, and 82 % still burned

The single most demonstrable run in this project. Real CIFAR-10, real CNN,
`ARC_MODE=baseline`, `lr=0.5`, 4 epochs — no injected failure, no scripted
curve. Measured on the shipped code, 2026-08-22:

```
detected           loss_plateau @ step 316, at 15.32 s
run finished       85.91 s  (1,560 steps, all 4 epochs)
burned after ARC   70.59 s
already knew       = 82.2 % of the run
```

What the run actually did, epoch by epoch:

| epoch | step | train loss | train acc | val acc | lr |
|---:|---:|---:|---:|---:|---:|
| 1 | 390 | 2.3076 | 10.54 % | **10.00 %** | 4.43e-01 |
| 2 | 780 | 2.3090 | 9.96 % | **10.00 %** | 2.66e-01 |
| 3 | 1170 | 2.3054 | 10.06 % | **10.00 %** | 7.89e-02 |
| 4 | 1560 | 2.3030 | 9.97 % | **10.00 %** | 0.00e+00 |

CIFAR-10 has ten classes. **10.00 % validation accuracy is random guessing, and
the loss is pinned at 2.303 = ln(10).** This model learned nothing, for four
epochs, while every loss curve on the dashboard looked flat-but-plausible and no
NaN, no gradient spike and no rank collapse ever fired. It is exactly the failure
mode that motivated the `loss_plateau` rule, and exactly the one a human watching
a loss curve does not catch quickly.

ARC knew at **15.32 s**. The run kept going for another **70.59 s** — 82.2 % of
its total compute — because nothing stopped it.

**That gap is the product.** ARC does not act on this failure and says so: no
response to a plateau has survived measurement, and the one that was tried made a
recoverable run terminal. What it does instead is put a named verdict and a live
cost counter on screen at second 15, so the 70 seconds after it are a choice
rather than an accident. On this 86-second demo that is fractions of a cent. On
the 48-hour run the funding proposal describes, 82 % is a day and a half of GPU
time spent after the answer was already known.

**Reproduce it:**

```bash
ARC_MODE=baseline ARC_DEMO_LR=0.5 ARC_DEMO_EPOCHS=4 \
  python python/runner.py python/train_demo.py
```

Whether the plateau lands at step 316 exactly depends on data order and
initialisation — the rule is not told where to look. What reproduces is the
shape: a run pinned at chance, detected early, and the large majority of the
compute spent after detection.

---

## 3. Unrecoverable Stop — computation waste

The question: once ARC knows a run is failing, how much compute does the run
burn before anyone stops it? Measured in `baseline` mode (interventions
suppressed, telemetry only) with deliberately divergent hyperparameters
(`lr=3.0`, `warmup=5`, 3 epochs) so a failure is actually reached.

| | |
|:---|:---|
| First `failure_detected` event | **3.81 s** |
| Total run wall clock | **50.96 s** |
| Burned *after* the failure was already known | **47.16 s** |
| Fraction of the run wasted post-verdict | **92.5 %** |

Repeated across a second run of the same configuration: failure at 4.12 s,
completion at 52.55 s — 89.8 % post-verdict. Both runs completed all 1,170 steps
without anyone intervening, because nothing stops them.

Priced at the tier-typical figures the dashboard uses for an RTX-class board
(250 W, $0.20/hr cloud-equivalent), 47.16 s is about **0.0033 kWh** and
**$0.0026**. Those absolute numbers are negligible — this is a 53-second demo.
The transferable figure is the **~90 % fraction**: on a run where the failure is
detected early, nine tenths of the compute is spent after the answer is already
known. The Stop button is what converts that fraction into a saving, and it
scales linearly with run length.

---

## 4. Intervention vs baseline — A/B on the same configuration

Same divergent configuration (`lr=3.0`, `warmup=5`, 3 epochs, 1,170 steps),
run once with interventions suppressed and once with them active.

| | `baseline` | `active` |
|:---|:---|:---|
| `failure_detected` events | **7** (at 4.1, 4.2, 4.3, 4.4, 18.1, 18.1, 19.7 s) | **1** (at 4.0 s) |
| Interventions applied | 0 | 2 — `rollback_and_reduce_lr`, `enable_grad_clipping` |
| Failures after the first | 6 more, spread over the run | **0**, for the remaining 49 s |
| First NaN/Inf loss | none | none |
| Wall clock | 53.13 s | 54.07 s |
| Loss (first / min / last) | 2.322 / 2.242 / 2.306 | 2.322 / 2.171 / 2.301 |

One detection plus two interventions at the 4-second mark eliminated every
subsequent failure event for the rest of the run, at a wall-clock cost of under
1 second (1.8 % overhead).

**Honest caveat, and it is a significant one:** *neither arm learned anything.*
CIFAR-10 has 10 classes, so a loss of 2.30 is random chance, and both arms
finished there. ARC prevented the divergence it was asked to prevent, but the
recovered run then sat flat for the remaining 49 seconds without ARC flagging
that as a problem — its risk heuristic scores NaN/Inf, loss-doubling explosions
and gradient blowup, and has no concept of "loss stopped decreasing." That gap
is documented in `docs/FUTURE_IMPROVEMENTS.md` and is not fixed. Read this table
as *"intervention suppressed the failure mode"*, not as *"intervention rescued
the run."*

---

## 5. Checkpoint budget — storage waste

`arcAgent.maxCheckpointMB` caps host RAM used by the rollback ring buffer, on top
of the existing count cap.

| Budget | Checkpoints retained | Bytes held |
|:---|---:|---:|
| 512 MB (default) | 3 × 11.15 MB | 33.5 MB |
| 20 MB | 1 × 11.15 MB | 11.2 MB |

The byte budget correctly overrides the count cap and never collapses below one
checkpoint, which would lose rollback capacity entirely. Storage held scales with
model size, so the cap matters most exactly where it should — large models.

---

## 6. Accessibility — literal a11y

Lighthouse 13.4.1, desktop, `navigation` mode, against `media/dashboard.html`
rendered standalone (webview placeholders substituted, CSP meta removed for the
audit; no other changes).

| Revision | Accessibility score | Remaining failures |
|:---|---:|:---|
| Pre-Round-2 (`82ad245^`) | **87** | `color-contrast`, `landmark-one-main` |
| Round 2 as shipped (`82ad245`) | **93** | `color-contrast`, `landmark-one-main` |
| After this pass | **100** | none |

The Round 2 work (ARIA live regions, status strip, chart data-table equivalents,
focus-visible, `prefers-reduced-motion`, two badge contrast fixes) moved the
score 87 → 93, and raised the number of passing audits from 34 to 42.

Two failures survived it, both found by this measurement pass and fixed here:

- **`color-contrast`** — `.stepper-label` and `.risk-hero-score` used
  `--text-tertiary` (`#5a5a63` on `#131316` = **2.72:1**, below the 4.5:1 AA
  threshold for small text). Both now use `--text-secondary`, which measures
  **5.49:1** in dark and **5.28:1** in light. No new token was needed — the
  existing secondary token already passes in both themes.
- **`landmark-one-main`** — the document had a `<header>` but no `<main>`. The
  `.dashboard-container` wrapper is now a `<main>` element. No styling change:
  the class and all CSS are unchanged.

Both fixes are in `media/dashboard.html`.

**Not measured:** a manual screen-reader pass (NVDA / Orca). Lighthouse's
automated checks are a floor, not a ceiling — they cannot tell you whether the
ARIA live announcements are actually *useful* in sequence, only that they exist.
That pass has not been done and no claim is made about it.

---

## Summary

| Dimension | Measured result | Confidence |
|:---|:---|:---|
| Bandwidth / storage (telemetry) | −62.9 % to −72.2 % bytes, zero wall-clock cost | High — direct byte count, control arm |
| Computation (Stop) | ~90 % of a failing run burns after the verdict | High — two runs, consistent |
| Computation (intervention) | 7 failure events → 1, at 1.8 % overhead | High — A/B, same config |
| Accessibility | 87 → 100 Lighthouse | High — automated, reproducible |
| Storage (checkpoint cap) | Budget correctly overrides count cap | High — direct observation |
| Time (preflight) | No meaningful wall-clock saving on this workload | High — and reported as a null result |

The two results worth defending are the telemetry reduction and the ~90 %
post-verdict burn fraction. The preflight time saving did not materialise on this
workload and is reported as a null result rather than dropped. The intervention
A/B prevented a failure mode but did not produce a converging run, and is
qualified accordingly.

### Reproducing

Measurement scripts are not committed — they are short subprocess harnesses that
count stdout bytes and timestamp event types from `runner.py`'s JSON stream.
Each measurement above states its method, arm and sample count so it can be
rebuilt directly. The Lighthouse audits need only a static file server pointed at
a placeholder-substituted copy of `media/dashboard.html`.
