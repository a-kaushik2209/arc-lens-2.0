# ARC Lens — Pitch Deck Content

Source material: `README.md`, `ARCHITECTURE.md`, `ARC_FUNDING_PROPOSAL.md`,
`arc_lens_business_plan.md`, `SECURITY_AUDIT.md`, `FUTURE_IMPROVEMENTS.md`. Every number
below traces back to one of those files. Anything not independently verified is labeled as
such — say it that way out loud too, a judge who catches an oversold claim will not un-hear it.

---

## Elevator Pitch (one sentence)

> ARC Lens is a VS Code extension that watches your PyTorch training run in real time and,
> the moment it starts to fail — NaN loss, exploding gradients, silent representation
> collapse — automatically rolls the model back to its last healthy checkpoint and lowers the
> learning rate, so the run keeps going without you ever touching a restart button.

---

## The Problem (2–3 sentences)

Neural network training fails constantly and expensively: a single NaN gradient at hour 47 of
a 48-hour run on a $3/hr A100 is not an inconvenience, it's a complete loss of the compute,
time, and momentum already spent, and industry-wide estimates put wasted GPU compute from
these failures in the hundreds of millions of dollars annually. Today's tooling only watches —
TensorBoard and Weights & Biases show you the crash after it happens, and gradient clipping
only guards against one failure mode — so ML engineers end up as full-time babysitters,
manually restarting and re-diagnosing the same class of failure over and over across every
project.

---

## Slide-by-Slide Outline

### Slide 1 — Title

**ARC Lens**
*Training runs that recover themselves.*

A real-time PyTorch monitor and autonomous recovery system, built into VS Code.

*(subtext line, optional)* Powered by `arc-training` (PyPI)

---

### Slide 2 — Problem

**Headline:** Training fails. Nobody catches it in time.

- A NaN gradient at hour 47 of a 48-hour run doesn't just slow you down — it erases the run.
- A single failed long training run can cost **$150–$5,000** in wasted cloud compute.
- The average ML engineer juggles **5–15 concurrent training runs**, watching dashboards for
  failures that show up hours apart.
- The MLOps market is valued at **$1.4B (2023)**, projected to reach **$13B by 2030** — but
  today's tools only tell you something went wrong. None of them fix it.

**One-liner for the slide footer:** *"We didn't build another dashboard. We built the fix."*

---

### Slide 3 — Solution

**Headline:** ARC Lens watches, detects, and recovers — automatically.

- Hooks into any PyTorch training loop with zero/minimal code changes.
- Streams live telemetry to a VS Code dashboard: loss, learning rate, gradient norm, GPU
  memory, plus deeper signals — effective rank, gradient entropy, weight update ratio,
  gradient flow ratio. Of those four, only effective rank is wired to an intervention; the other
  three are charted for the user to read. Two of them used to trigger and were deleted after
  measurement showed they fired on healthy runs.
- The moment a run crosses a failure threshold (NaN/Inf loss, gradient explosion,
  representation collapse), a local recovery agent **rolls back to the last healthy
  checkpoint and reduces the learning rate** — no restart, no manual intervention.
- Training resumes automatically, in the same process, in under a second.

---

### Slide 4 — How It Works

**Headline:** Three tiers, one loop.

1. **VS Code Extension** — the dashboard, the "Run with ARC Lens" button, the chat and script
   generator (Pro).
2. **Telemetry Engine** — hooks `Optimizer.step`, so it measures once per *weight update*.
   Computes gradient norm / learning rate / GPU memory, plus the structural signals from
   `arc-training` (PyPI), and streams it all out as JSON.
3. **Local Recovery Agent** — a rule-based loop that watches those signals. When a threshold
   trips it restores the last healthy checkpoint, scales down the learning rate, or turns on
   gradient clipping, live, on the running process — and if three attempts don't work it says
   the run is unrecoverable instead of retrying forever.

**Suggested diagram (simple, drawable by hand or in slides):**

```
 [Your PyTorch script]          ← unmodified; no import, no callback
        │  (optimizer.step() intercepted = one weight update)
        ▼
 [Telemetry Engine]───────────► streams metrics ───────►[VS Code Dashboard]
        │
        │  loss NaN or exploded? grad norm > 50? effective rank collapsed?
        ▼
 [Recovery Agent] ──rolls back weights, cuts LR, clips grads──► [training resumes]
```

Draw it as a loop, not a line: telemetry flows up to the dashboard continuously; the recovery
agent sits beside the training loop and only acts when a threshold trips, then hands control
straight back.

---

### Slide 5 — Live Demo

**Headline:** Watch it happen.

What to show, in order:
1. Open a Python training script in VS Code, click **▶ Run with ARC Lens**. No config, no
   code changes.
2. Dashboard opens — point out the four live chart panels (Vitals, Dynamics, Structural,
   Flow) updating in real time.
3. Let the run hit a failure. **Nothing is injected** — the demo runs a real CIFAR-10 CNN at a
   deliberately aggressive learning rate, so whether it fails and at which step depends on the
   data order and the initialisation. Seeded runs are reproducible, so do the dry run in
   `DEMO_SCRIPT.md` §1.4 and know your number before you are on stage.
4. Narrate the trace as it appears: failure detected → response chosen → applied, all inside the
   same run, no restart.
5. Be precise about which failure you got. A `numerical` failure gets a rollback and the loss
   curve resumes. A `loss_plateau` ("stalled") gets a learning-rate cut and **no** rollback —
   that run ends at chance accuracy, and the honest line is "we detected a death that every
   loss-curve tool would have shown as a flat green line", not "we saved it".

**Presenter note:** the recovery trace is a deterministic rule engine, not a live LLM call —
it's fast and 100% reproducible on stage, which is a feature for a demo, not a limitation to
hide. Say so if asked.

---

### Slide 6 — Differentiation

**Headline:** Everyone else shows you the crash. We fix it.

| Capability | TensorBoard | Weights & Biases / Neptune | ARC Lens |
| :--- | :---: | :---: | :---: |
| Real-time telemetry | Yes | Yes | Yes |
| Advanced failure signals charted (effective rank, gradient entropy) | No | No | Yes |
| Detects failure | After the fact | After the fact | In real time |
| **Automatic rollback & recovery** | **No** | **No** | **Yes** |
| Requires code changes | Minimal | Minimal | Zero/minimal |

**The point to hammer:** TensorBoard, W&B, and Neptune are observability tools — they log and
visualize what already happened. None of them take action. ARC Lens is the only one of these
that closes the loop: it doesn't just tell you the run died, it stops the run from dying in
the first place.

---

### Slide 7 — Traction / Validation

**Headline:** Tested across 9 architectures, 10M–117M parameters.

*(Label this slide clearly: "Internal testing — not yet independently verified" in the
footer, and say the words out loud when presenting it.)*

- **100% recovery rate**, zero false positives, across 4 protection methods × 5 failure types
  × 5 random seeds (claimed in internal testing).
- **Runtime overhead measured on GPU: 1.8% core, 8.4% with full structural diagnostics.**
  RTX 3050, 2.79M-parameter CNN, 200 steps × batch 128, median of 3, same loop run with and
  without the harness. Reproducible with `python python/benchmark_overhead.py`; raw numbers in
  `docs/benchmark_overhead.json`. This supersedes the earlier CPU-only "<10%" claim — the
  number now has a method behind it, and sampling the expensive signals every step instead of
  every 25 costs 170%, which is why they are sampled.
- **Baseline-vs-active A/B on real CIFAR-10**, identical seeds, interventions the only
  difference — including the configurations where ARC detects the failure and cannot save the
  run. Results in `docs/EXPERIMENT_RESULTS.md`.
- Validated against 9 real architectures from 10M to 117M parameters (NanoGPT, ResNet-50,
  YOLOv11, GPT-2 Small/Medium, Stable Diffusion U-Net, Llama-style, ViT-Base), each recovering
  from an injected failure (LR spikes, NaN bombs, gradient explosions).
- `arc-training` is published and live on PyPI (`5.0.0`); ARC Lens `v0.1.0`+ is packaged and
  submission-ready.

---

### Slide 8 — Business Model

**Headline:** Free for every developer. Paid for the AI layer.

**Free tier — $0/month:**
- Live telemetry dashboard (loss, learning rate, gradient norms, effective rank, and more)
- Automatic rollback & recovery — the core self-healing engine, free for everyone
- Chart export (PNG)

**Pro tier — $2.99/month:**
- AI Failure Analyst — a chat that explains *why* a run failed, using live telemetry context
- ARC Script Generator — generates ARC-instrumented PyTorch training scripts from a form
- Deep telemetry trend explanations

**Why the economics work:** Pro uses a **Bring Your Own Key (BYOK)** model — users supply
their own LLM API key (OpenRouter, Groq, Anthropic, OpenAI, or Gemini, auto-detected from the
key), so ARC never pays for token usage. That's roughly **85% net margin** on every
subscription, with zero token-cost liability as usage scales.

---

### Slide 9 — Roadmap

**Headline:** What ships next — turning the claim into proof.

- **Compute-savings ledger** — every time ARC recovers a run, show the number a non-ML judge
  understands instantly: *"4h 12m of training preserved · $12.60 of A100 time not re-spent."*
- **Side-by-side A/B comparison** — run the same script twice, identically seeded, once with
  recovery on and once off, and plot both curves. This turns "ARC recovers your training run"
  from a claim into something a skeptic can watch happen with their own eyes.
- **Intervention markers on the charts** — annotate the exact step where a failure was
  detected and where recovery kicked in, directly on the loss curve, so the before/after is
  visible at a glance with no narration needed.

---

### Slide 10 — Team / Ask

**Team:**
*[Add team member names, roles, and a one-line bio each here.]*

**The Ask:**
*[State clearly what you want from the audience — judging consideration, feedback, pilot
users, a specific prize track, mentorship, etc.]*

---

## Talking Points

### The 3 most compelling, most defensible things to emphasize

1. **It's the only one that acts, not just watches.** TensorBoard, W&B, and Neptune are all
   observability — they log and chart what already happened. ARC Lens is architecturally
   different: it intercepts the training loop and takes a corrective action (checkpoint
   restore + learning-rate cut) without a human in the loop. This is a real, load-bearing
   design difference documented in the architecture, not a marketing spin on the same feature
   set — lean on it hard.

2. **It watches for failures that produce no NaN and no gradient spike.** Representation
   collapse — the model quietly losing dimensionality — doesn't show up in a loss curve at all,
   and ARC acts on effective rank falling below half the run's own baseline. Claim this as a
   capability that is *built and wired*, not as one that is proven: the rule is deliberately
   conservative and has **never fired** in our validation runs (a healthy run bottoms at 97% of
   baseline, a damaged one at 83%, against a 50% trigger). We had a second silent-failure rule on
   gradient entropy and deleted it — it fired at step 125 on a healthy CIFAR-10 run and drove it
   from 87.43% to chance accuracy, and the signal turns out to converge to the same value on
   healthy and dead runs alike. The verified, demonstrable capability is numerical-divergence
   detection and recovery, which the A/B measures directly. Overstating the silent-failure story
   is the single easiest way to lose a technical judge, because the evidence points the other
   way.

3. **Zero code changes to integrate.** The instrumentation anchors on `Optimizer.step`, so a
   user's existing training script needs no rewrite to get monitored and protected. This is a
   real adoption-cost advantage over tools that require wiring logging calls throughout a
   training loop, and it's demonstrable live in under a minute.

   The anchor choice is worth one sentence if a technical judge asks: hooking the optimizer
   rather than `backward()` means one recorded step is one *weight update*, which is what makes
   gradient accumulation, mixed precision and multi-optimizer setups correct rather than merely
   non-crashing.

### One honest caveat to have ready

The caveat has moved. GANs, gradient accumulation and AMP are **now handled and tested** —
`self` at the optimizer anchor is definitionally the right optimizer, the model is matched by
parameter identity rather than by grabbing the first `nn.Module` on the stack, and
`GradScaler.scale` hands over the unscaled loss directly. Verified on GPU: a 4×-accumulation
AMP loop reports 20 backward calls, 5 optimizer steps and 5 metrics, with correctly unscaled
values.

What is **genuinely still open is distributed training.** DDP/FSDP needs rank-aware emission,
`all_reduce` for global gradient norms, and a barrier so every rank rolls back to the same
checkpoint. None of that has been tested, because it needs multi-GPU hardware we don't have.
Say that plainly: *"single-process, single-GPU today — multi-optimizer within that process is
fine and tested; distributed is the next major piece and I'm not going to claim it works when
I haven't run it."* That answer is stronger than a hedge, and it is where the expensive
failures actually live, so it doubles as the roadmap.
