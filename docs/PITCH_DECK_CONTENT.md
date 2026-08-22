# ARC Lens — Pitch Deck Content

**FAR AWAY 2026, Round 2 · Challenge #171: Accessibility — Resource Waste Reduction**
**Team Heisen-bug (U333WKR8)**

Source material: `README.md`, `ARCHITECTURE.md`, `WASTE_REDUCTION.md`, `SWEEP_LOG.md`,
`EXPERIMENT_RESULTS.md`, `ARC_FUNDING_PROPOSAL.md`, `arc_lens_business_plan.md`,
`SECURITY_AUDIT.md`, `FUTURE_IMPROVEMENTS.md`. Every number below traces back to one of those
files. Anything not independently verified is labeled as such — **say it that way out loud too.
A judge who catches an oversold claim will not un-hear it, and this deck's strongest asset is
that its unfavourable numbers are volunteered.**

Spoken script: [`PITCH_SCRIPT_ROUND2.md`](PITCH_SCRIPT_ROUND2.md). Demo runbook:
[`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).

---

## Elevator Pitch (one sentence)

> ARC Lens is a VS Code extension that watches your PyTorch training run in real time and, the
> moment the loss goes non-finite, rolls the model back to its last healthy checkpoint and cuts
> the learning rate — on a seeded A/B that took a run from 10.00% accuracy, which is chance, to
> 46.59% — and for the failures it can detect but *not* safely fix, it tells you at second 15
> instead of second 86 rather than guessing at a response.

---

## The Problem (2–3 sentences)

Training runs fail without crashing, and then keep burning GPU after they are already dead: on
our own demo run ARC knew the run had failed at 15.3 seconds, and the run continued to 86
seconds because nothing stopped it — **82% of the compute spent after the answer was known**, a
fraction that scales with run length until it is a day and a half of a 48-hour job. Today's
tooling only watches — TensorBoard and W&B show you the crash after it happens, and gradient
clipping guards one failure mode — so ML engineers end up as full-time babysitters, manually
restarting and re-diagnosing the same class of failure across every project.

---

## Slide-by-Slide Outline

Fourteen slides of content. If you have ten, cut 8, 12, 13 and merge 3 into 2 — the cut list is
at the bottom.

---

### Slide 1 — Title

**ARC Lens**
*Training runs that recover themselves — and say so when they can't.*

A real-time PyTorch monitor and autonomous recovery system, built into VS Code.

*(subtext)* Powered by `arc-training` (PyPI) · Challenge #171

---

### Slide 2 — Problem

**Headline:** Training fails silently, and keeps billing you.

- A NaN gradient at hour 47 of a 48-hour run doesn't slow you down — it erases the run.
- Worse is the failure that never crashes: **our demo run sat at 10.00% accuracy — random
  guessing on 10 classes — for four full epochs**, loss pinned at ln(10), while every chart
  looked flat-but-plausible. No NaN, no gradient spike, no rank collapse.
- **ARC knew at 15.3 s. The run continued to 86 s. 82% of its compute was spent after the
  answer was already known** — because nothing stopped it.
- A single failed long training run costs **$150–$5,000** in wasted cloud compute. The MLOps
  market is **$1.4B (2023) → $13B (2030)**, and today's tools tell you something went wrong.
  None of them act.

*(Footer)* *"We didn't build another dashboard. We built the part that acts — and the part that
admits when it shouldn't."*

**Provenance note for the presenter:** the 82% and the 10.00% are ours, measured, reproducible
(`WASTE_REDUCTION.md` §3a). The dollar and market figures are third-party estimates. Don't blur
them together.

---

### Slide 3 — The brief, and our answer to it

**Headline:** Accessibility and resource waste are the same work here.

Challenge #171 asks us to improve the part of the MVP most related to **accessibility** so that
it reduces **waste**, and to clearly show **success, failure, current status and next steps**.
Those read like two deliverables. In this product they are one, and the reason is a limitation,
not a boast:

> **ARC detects a silent death reliably. It cannot fix one.**
> Four structural detection rules were built. Two were deleted for firing on healthy runs. The
> two that remain are report-only, because every response we tried made a recoverable run worse.

So for that failure, the product's output is not a rescue — it is **a person told, in time, in
terms they can act on.** Which makes the status strip, the ARIA live region, the preflight's
named cause, and the data-table equivalent behind every chart not decoration around the feature.
**For that failure they are the entire feature**, and the 82% gap they close is the waste.

---

### Slide 4 — Solution

**Headline:** ARC Lens watches, detects, recovers — and reports when recovery isn't safe.

- Hooks any PyTorch training loop with **zero code changes**. No import, no callback, no
  decorator.
- Streams live telemetry to a VS Code dashboard: loss, learning rate, gradient norm, GPU memory,
  plus structural signals — effective rank, gradient entropy, weight update ratio, gradient flow
  ratio.
- **The moment the loss goes non-finite or past 1e6**, a local recovery agent rolls back to the
  last healthy checkpoint, cuts the learning rate, and latches gradient clipping on. Training
  resumes in the same process, no restart.
- **That is the one entry point, and saying so is the honest version.** A structural signal can
  raise a report; only the loss can trigger an action. Two structural rules were deleted after
  measurement showed them firing on healthy runs; the remaining two detect and report but no
  longer act.
- After three failed recoveries of the same kind it declares the run **unrecoverable** and says
  so, instead of rolling back forever.

---

### Slide 5 — How It Works

**Headline:** Three tiers, one measurement anchor.

1. **VS Code Extension** — dashboard, the "Run with ARC Lens" button, chat and script generator
   (Pro).
2. **Instrumentation Harness** — monkey-patches PyTorch at **`Optimizer.step`, not
   `loss.backward()`**. One recorded step is one *weight update*, which is what makes gradient
   accumulation, mixed precision and multi-optimizer GANs correct rather than merely
   non-crashing. Hook `backward()` instead and all three are wrong.
3. **Recovery Agent** — a deterministic rule engine. Restores weights into the live model
   mid-run and continues against them, *while an LR scheduler rewrites the learning rate every
   step and tries to undo the correction.* That case is tested.

Your script runs unmodified through `runpy`, so tracebacks report **your** line numbers.

```
 [Your PyTorch script]          ← unmodified; no import, no callback
        │  (Optimizer.step intercepted = one weight update)
        ▼
 [Telemetry Engine]───────────► streams metrics ───────►[VS Code Dashboard]
        │
        │  loss NaN or exploded?          (structural signals report only)
        ▼
 [Recovery Agent] ──rolls back weights, cuts LR, clips grads──► [training resumes]
```

Draw it as a loop, not a line.

---

### Slide 6 — What the demo run actually is

**Headline:** A real model, a real dataset, and a plausible mistake.

Before showing anything, spend twenty seconds on what's running. This is the slide that stops a
judge wondering whether the failure was staged.

| | |
|:---|:---|
| Model | `DemoCNN` — 9-layer CNN, **2,788,042 parameters**, BatchNorm throughout |
| Data | **Real CIFAR-10.** Not synthetic, not a subset |
| Optimizer | SGD, momentum 0.9, weight decay 5e-4, cosine decay, **no gradient clipping** |
| The one unusual thing | **Peak LR 5.0**, warmup 5 steps |
| Seeded | Yes — reproducible run to run |

**Nothing is injected.** There is no NaN bomb, no scripted curve, no synthesised metric. The
failure is what a learning rate of 5.0 does to this network on this data, and ARC has to
*detect* it rather than be told where it is. Whether it fails and at which step depends on data
order and initialisation.

**And 5.0 is not an absurd number in the way it looks.** It's what you get copying a learning
rate from a paper that used a different model and a much larger batch, and keeping your own
warmup. That is a mistake people actually make.

**Why not 0.5** (the previous default, and worth knowing if asked): BatchNorm renormalises every
block, so at 0.5 the network doesn't explode — it *saturates* into a flat run at chance accuracy.
ARC detects that as `loss_plateau` and deliberately does not act. Correct behaviour, but the
rescue path is never exercised and a viewer sees detection without recovery. At 5.0 the loss goes
non-finite, which is the failure ARC does act on — so both kinds show up in a single run.

---

### Slide 6b — Live Demo

**Headline:** Watch it happen. Nothing is injected.

1. Open a training script, click **▶ Run with ARC Lens**. No config, no code changes.
2. Dashboard opens — four live chart panels, plus the **Resources Conserved** panel counting
   from step one.
3. **Default `train_demo.py` is now `ARC_DEMO_LR=5.0`, warmup 5, and a default run produces both
   failure kinds.** The loss goes non-finite around step 6 — ARC rolls back, cuts the LR, latches
   clipping, and the run climbs off chance accuracy. Later, a `loss_plateau` fires and ARC
   deliberately does **not** act.
4. **Show both. The second is the more convincing half.** The honest line is: *"we detected a
   death that every loss-curve tool would have shown as a flat green line, and we have no
   evidence any response we can make helps, so we report it instead of guessing."*

**Do not demo at `ARC_DEMO_LR=0.5`** (the old default). It only produces `loss_plateau`, so the
rescue path is never exercised and nobody sees a recovery.

**Presenter notes:** the recovery trace is a deterministic rule engine, not a live LLM call —
fast and 100% reproducible on stage. That's a feature for a demo, not a limitation to hide.
Do the dry run in `DEMO_SCRIPT.md` §1.4 and know your step number before you're on stage.

---

### Slide 7 — Proof: compute actually preserved

**Headline:** Same seed. Same data order. One difference.

`train_demo.py` defaults, 1,950 steps, the only variable is whether interventions are allowed.
The loss goes non-finite at step 6 in **both** arms.

| | `baseline` (control) | `active` |
|:---|---:|---:|
| Interventions applied | **0** | **2** — rollback + LR cut, grad clipping |
| Epoch 1 train loss | 2.52e+12 | 4.22e+05 |
| Epoch 3 val accuracy | 10.00% | **18.28%** |
| Epoch 5 val accuracy | **10.00%** — chance | **46.59%** |
| Wall clock | 85.27 s | 113.49 s |

**+36.59 points against an identically seeded control**, and still rising when the schedule ends.

**Say the caveats out loud, before a judge finds them:**
- The active arm is **33% slower**. Recovering costs more than letting a dead run coast. ARC
  didn't make this run cheaper — it turned 85 wasted seconds into 113 productive ones. The
  alternative was never "a faster good run", it was "restart and hope".
- **This is a single seeded pair.** `SWEEP_LOG.md` sweep 6 is exactly why that matters. No
  distribution over repeated runs has been measured and **no error bar is claimed.**
- 46.59% is not a good CIFAR-10 result. It's a **rescued** run, not a tuned one.

---

### Slide 8 — Waste reduction, measured in four dimensions

**Headline:** No claim without a control arm.

| Dimension | Result | Confidence |
|:---|:---|:---|
| **Computation** (Stop) | ~90% of a failing run burns after the verdict | High — two runs, consistent |
| **Computation** (intervention) | 7 failure events → 1, at 1.8% overhead | High — A/B, same config |
| **Bandwidth / storage** | **−72.2%** telemetry bytes, **zero** wall-clock cost | High — direct byte count |
| **Storage** | Checkpoint byte budget correctly overrides the count cap | High — direct observation |
| **Accessibility** | Lighthouse **87 → 100** | High — automated, reproducible |
| **Time** (preflight) | **No meaningful saving.** Null result. | High — and published as one |

**Two results came back against us and are on the slide anyway:** the preflight saves *effort*
(a named cause and a named fix instead of a traceback), not seconds. And the telemetry saving is
**off by default** — `arcAgent.telemetryEvery` is 1; −72.2% is what it buys at 10. Risk detection
is unaffected either way, because loss history and risk are computed every step regardless.

**Overhead of the whole system: 1.8% core, 8.4% with full structural diagnostics.** GPU-measured,
same loop with and without the harness, median of 3. Sampling the expensive signals every step
instead of every 25 costs 170% — which is why they're sampled.

---

### Slide 9 — Accessibility, and the honesty audit

**Headline:** 87 → 100. Then we audited whether the numbers were real.

**Lighthouse 87 → 100** (13.4.1, desktop, `navigation`): ARIA live regions that escalate to
assertive only on failure, a data-table equivalent behind every canvas chart, `focus-visible`,
`prefers-reduced-motion`, and contrast fixes. Passing audits went 34 → 42.

**Not claimed:** a manual NVDA/Orca pass. Lighthouse confirms the live region *exists*, not that
what it announces is useful in sequence. That gap is documented.

**Then we audited every figure the dashboard displays** and labelled each one *measured*,
*derived* or *estimated* — on screen, in the page itself. **The audit found four fabrications**,
all of the same pattern:

- an absent `lr` rendered `0.00e0`; an absent `gpu_mem_mb` rendered `0.0 MB`
- **an absent risk score rendered `0.00` — which reads as *safe*.** On a gauge whose entire job
  is saying when a run is *not* safe, that's fabricating in the one direction that matters.
- the telemetry tile displayed a number with no stated basis and could over-report without bound

All fixed. The tests now check a field's **presence**, not its truthiness — because zero is a
legitimate value for `lr` (cosine schedules end there) and for GPU memory (CPU runs report 0).

*The precedent: `SECURITY_AUDIT.md` C-2, where four charts were found rendering `Math.random()`
styled identically to measurements. The standard since: a number that looks like a measurement
must be one, and an absent value must read as absent.*

---

### Slide 10 — The four states

**Headline:** Success, failure, status, next steps — all on screen.

| State | Where it lives |
|:---|:---|
| **Current status** | Status strip: running, at what step, at what risk |
| **Success** | The resume — failure detected, response applied, marked on the chart at the exact step (`markLine`/`markArea`) |
| **Failure** | After three failed recoveries of a kind: **"unrecoverable"**, stated plainly. The useful answer then is "kill it", not a fourth rollback |
| **Next steps** | The Stop button, and an exportable self-contained HTML incident report |

That Stop button is the 82% figure converted into an actual saving. Detection is worth nothing
if nothing acts on it.

---

### Slide 11 — Differentiation

**Headline:** Everyone else shows you the crash.

| Capability | TensorBoard | W&B / Neptune | Lightning | Composer | ARC Lens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Real-time telemetry | Yes | Yes | via logger | via logger | Yes |
| Structural failure signals charted | No | No | No | No | **Yes** |
| Detects failure | After the fact | Alerts a human | `check_finite` | — | In real time |
| Response | None | Notify | **Stops the run** | Restart whole job | **Corrects in-process** |
| IDE-native | Partial | No | No | No | **Yes** |
| Code changes | Minimal | Minimal | Trainer rewrite | Trainer rewrite | **Zero** |

**The defensible claim, stated narrowly:** *no dashboard-based competitor surveyed converts a
detected training failure into an automatic, in-process correction that lets the same run
continue.* They alert, they stop, or they restart the whole job after a hardware fault.

**Two things to say before a well-read judge says them for you:**
1. **Rollback + LR cut is not a new technique.** It's documented practice in the OPT-175B and
   BLOOM logbooks — done by hand, by an engineer watching a dashboard. Our contribution is
   automating a known manual practice for people without a frontier lab's monitoring team. Real
   contribution; not a new algorithm.
2. **The shipped detector is deterministic thresholds, not ML.** A learned classifier exists in
   the core research library; it is **not** wired into the live path, so we don't claim it.

---

### Slide 12 — Validation

**Headline:** 136 tests. 8 architectures. Every sweep published, including the ones that lost.

*(Footer, and say it aloud: "Internal testing — not independently verified.")*

- **136 tests** — 75 Python, 61 TypeScript/dashboard. Ten Python tests run the **real harness
  against real training loops** end to end: gradient accumulation doesn't inflate the step count,
  an LR intervention survives a scheduler rewriting the LR every step, baseline mode never
  intervenes, tracebacks land on the user's lines.
- Several tests were **written before the code they cover and found real bugs doing it** —
  including an optimizer matched to a wrapper module instead of the submodule it updates, which
  would have rolled back both halves of a GAN.
- **8 architectures, 10M–117M parameters** (NanoGPT, ResNet-50, YOLOv11, GPT-2 Small/Medium,
  Stable Diffusion U-Net, Llama-style, ViT-Base) — **8 of 8 recovered**. Read that as what it is:
  **programmatically injected** failures, measured **by us**, on CPU. "100% on injected failures"
  is a different claim from "100% on real runs at scale", and we make only the first.
- **CI fails the build on any secret-shaped literal** in source.
- `arc-training` is live on PyPI; ARC Lens `0.3.9` is packaged and submission-ready.

**Do not put "zero false positives" on this slide.** Two of our own detection rules were deleted
*for* false positives on healthy runs. The stronger story is that we caught them.

---

### Slide 13 — Business Model

**Headline:** Free for every developer. Paid for the AI layer.

**Free — $0:** live telemetry dashboard, automatic rollback & recovery (the whole self-healing
engine), chart export.

**Pro — $2.99/mo:** AI Failure Analyst (chat that explains *why* a run failed, with live
telemetry as context), ARC Script Generator, deep telemetry trend explanations.

**Why it scales:** Pro is **bring-your-own-key** — the user supplies their LLM API key
(OpenRouter, Groq, Anthropic, OpenAI, Gemini; provider inferred from the prefix). ARC never pays
for tokens and never holds a credential. Roughly **85% net margin**, with **zero token-cost
liability as usage grows.**

---

### Slide 14 — Roadmap

**Headline:** What ships next — turning the claim into proof.

*Two items previously here have shipped and moved into the demo: the compute-savings ledger is
now the live **Resources Conserved** panel, and intervention markers are annotated on three
charts. **Don't present shipped work as roadmap** — a judge who has seen the demo will catch it.*

- **Error bars on the rescue claim.** +36.59 points is one seeded pair.
  `python/repeatability.py --repeats N` turns it into a distribution. Sweep 6 is why this
  matters: at lr=0.5, run-to-run spread swamped effects of this size.
- **Side-by-side A/B in the UI.** The control arm already exists as a command; overlaying both
  curves in one chart turns our strongest evidence into something a skeptic watches live.
- **Distributed training (DDP/FSDP).** Rank-aware emission, `all_reduce` for global gradient
  norms, a barrier so every rank rolls back to the same checkpoint. **Untested** — needs
  multi-GPU hardware we don't have. This is where the expensive failures live, so it doubles as
  the honest answer to "does this scale?"
- **Manual screen-reader pass (NVDA / Orca).**

---

### Slide 15 — Team / Ask

**Team:** *[names, roles, one-line bio each]*

**The Ask:** *[judging consideration, pilot users, a specific prize track, mentorship]*

---

## If you only have 10 slides

Cut **8** (fold the −72.2% into slide 3), **12**, **13**, and merge **3** into **2**.
Never cut **7** or **9** — 7 is the only slide that proves the product works, and 9 is the only
one that proves the numbers are real.

---

## Talking Points

### The three most defensible things

1. **It acts, and we can show the delta.** TensorBoard, W&B and Neptune are observability. ARC
   Lens intercepts the training loop and takes a corrective action without a human. On a seeded
   A/B that's 10.00% → 46.59%. Load-bearing architectural difference, not marketing spin.

2. **It knows what it can't fix, and that's measured, not modest.** Four structural rules built;
   two deleted for firing on healthy runs (one drove a run from 87.43% to chance), two demoted to
   report-only after their responses were measured harming recoverable runs. Claim the
   silent-failure story as **detection**, never as rescue — the evidence points the other way, and
   overstating it is the single easiest way to lose a technical judge.

3. **Zero code changes.** The `Optimizer.step` anchor means an existing script needs no rewrite.
   Demonstrable live in under a minute, and a real adoption-cost advantage over tools that need
   logging calls wired through the loop.

### The honest caveat to have ready

**Distributed training.** GANs, gradient accumulation and AMP are handled and tested — verified
on GPU, a 4×-accumulation AMP loop reports 20 backward calls, 5 optimizer steps, 5 metrics, with
correctly unscaled values. DDP/FSDP is not. Say it plainly:

> *"Single-process, single-GPU today. Multi-optimizer within that process is fine and tested.
> Distributed is the next major piece and I'm not going to claim it works when I haven't run it."*

That answer is stronger than a hedge, and it doubles as the roadmap.

### If a judge asks "why should I trust any of these numbers?"

Point at slide 9. We audited our own dashboard and found four places it displayed a number it
didn't have — including a missing risk score rendering as `0.00`, which reads as safe. We fixed
them and wrote the regression tests. Then point at `SWEEP_LOG.md`: every sweep, including the two
killed mid-run and the causal claims we withdrew after they failed to replicate.

> *"Anything that intervenes in someone's training run has to earn it with evidence. Half the
> time ours said no, and that's in the repo too."*
