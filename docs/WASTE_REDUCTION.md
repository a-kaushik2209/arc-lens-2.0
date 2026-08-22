# Waste Reduction — Measured

FAR AWAY 2026, Challenge #171 (*Accessibility: Resource Waste Reduction*).
Team Heisen-bug (U333WKR8).

This document exists because the project's standing rule is **no claim without a
control arm**. Every number below was produced by running both arms on the same
machine in the same session. Where the win is small, or absent, that is what is
reported. Numbers that were not measured are marked as not measured rather than
estimated.

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
