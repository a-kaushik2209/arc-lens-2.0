# ARC Lens — Round 2 Critique & Revised Suggestions

**Challenge #171 · Accessibility: Resource Waste Reduction**
**Team: Heisen-bug (U333WKR8)**
**Document purpose:** Critical review of `ROUND2_PLAN.md` — feasibility assessment, what to cut, what to keep, and new ideas for a unique standpoint.

---

## Section 1 — Honest Assessment of the Existing Plan

### ✅ What's 100% Buildable and Should Stay

| Item | Why It Survives |
|:---|:---|
| **P0.1 — Status Strip + ARIA live region** | Pure HTML/CSS/JS change to `dashboard.html`. No Python changes. 2–3 hours max. Non-negotiable per the plan's own framing. |
| **P1.3 — Unrecoverable Stop + waste numbers** | The Stop button (`arc-lens.stop` command) already exists. Waste numbers (elapsed time, cost at `GPU_HOURLY_RATE`) are already partially computed for the savings ledger. This is mostly a UI wiring job. |
| **P1.4 — Chart `<table>` equivalents** | Zero backend changes. Pure dashboard HTML addition. ECharts canvas is genuinely invisible to screen readers — this is a real, needed fix. |
| **P2.7 — Keyboard + motion + contrast pass** | `prefers-reduced-motion` is a one-liner CSS rule. Focus rings are CSS. Tab order is HTML attribute order. Very cheap, real checkbox value. |

---

### 🔴 What's NOT Feasible — Problems With the Current Build

#### P0.2 — Preflight Doctor *(the plan's best waste story — but has a problem)*

This is the most compelling demo item but has a serious implementation constraint:

**The public repo runner is a simulation.** `runner.py` (public build) ignores the target script entirely and replays a fixed 30-step sequence. This means:

- Checking `torch` imports → meaningless, torch is never actually called
- Checking `arc-training` installation → not used in the simulation path
- Checking syntax errors in the user's script → runner never parses it
- Checking CIFAR-10 cache → `train_demo.py` does not use any dataset

**Fix:** Preflight Doctor needs to be a *separate* pre-run subprocess — a `preflight.py` script spawned by `extension.ts` before the main runner, which does the checks and returns structured JSON results. This is buildable (~4–6 hours) but requires:
1. Writing `python/preflight.py` from scratch
2. Modifying `extension.ts` to spawn the preflight process first, await results, render the panel, then conditionally launch the runner

This is non-trivial but worth doing — it's the most concrete, measurable "repeated effort" waste story in the submission.

---

#### P2.5 — Adaptive Telemetry Rate *(not demonstrable in simulation build)*

The simulation `runner.py` has hardcoded `time.sleep(step_delay)` — the only rate knob is a fixed delay. A coalescing layer can be added in `extension.ts` on the receiving end, but:

- It only reduces webview update frequency, not stdout emission rate
- The measurable claim ("bytes over a fixed 10-epoch run: before vs after") cannot be demonstrated in the simulation

**Skip or descope to a receiving-end coalescing filter only.**

---

#### P2.6 — CIFAR-10 Cache Reuse *(does not apply to this build)*

`train_demo.py` is a pure-Python unstable MLP — no dataset, no downloads. CIFAR-10 is never touched in the simulation. **Remove from the plan entirely.**

---

#### P2.8 — `arcAgent.maxCheckpointMB` + Auto-Pruning *(wrong layer)*

Checkpoint pruning lives in `arc-training` (the PyPI library), not in ARC Lens. The simulation runner performs no real checkpointing. This item needs the dev-enabled runner state + real GPU hardware to validate. **Not buildable in the simulation build. Remove from Round 2 scope.**

---

## Section 2 — What's Missing: New Ideas for a Unique Standpoint

These are items not in the original plan that fit the challenge PS better, are fully buildable in the simulation build, and would differentiate ARC Lens from any other submission.

---

### 🔥 Idea 1 — Early Warning "Runway" Indicator *(Genuinely Unique)*

**Effort: ~2h · Works in simulation: ✅**

The plan has a status strip. Go one level further: **predict how many steps remain before an intervention threshold is crossed**, shown live from the moment risk elevates.

This is a pure client-side extrapolation on the last N gradient norm / loss values — no backend changes, no Python changes.

**UI mockup:**
```
┌─ Risk Monitor ──────────────────────────────────────────┐
│  ⚠️  RISK: MEDIUM                                        │
│  Grad norm: 18.4  →  Threshold: 50                      │
│  Trend: +4.2 per step (last 5 steps)                    │
│  Estimated: ~7 steps to intervention                    │
│  Next step: Monitoring. No action yet.                  │
└─────────────────────────────────────────────────────────┘
```

**Why this is unique:** No competitor (TensorBoard, W&B, Neptune) shows a forward-looking estimate. They all show what already happened. This shifts ARC Lens's narrative from *reactive* to *predictive* — a genuinely different product position.

**Implementation:** In `dashboard.html`, on every `metric` event with risk > LOW, compute `slope = (recent_grad_norms[-1] - recent_grad_norms[-N]) / N` and `steps_to_threshold = (threshold - current) / slope`. Clamp and display. 20–30 lines of JS.

---

### 🔥 Idea 2 — Failure Taxonomy Cards *(Human-Readable "What Happened")*

**Effort: ~3-4h · Works in simulation: ✅**

After an intervention, render a rich, structured card that translates technical metrics into plain English. The simulation already emits structured `thought` events with reasoning trace content — use those to drive a card that satisfies Requirement 2 verbatim.

**UI mockup:**
```
┌─ Failure Diagnosed ────────────────────────────────────────┐
│                                                            │
│  Type:       Gradient Explosion                            │
│  Signal:     L2 norm = 95.4  (threshold: 50)               │
│  Likely cause: Learning rate too high for loss landscape   │
│                                                            │
│  What ARC did:                                             │
│    → Rolled back 10 steps to last healthy checkpoint       │
│    → Cut learning rate  0.001 → 0.0002  (–80%)            │
│    → Cleared poisoned gradients (zero_grad)                │
│                                                            │
│  Status:     ✅ RECOVERED — Training resumed at step 10    │
│                                                            │
│  Next step:  Monitor the next 5 steps. If grad norm        │
│              exceeds 30 again, lower your base LR further. │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Why this matters:**
1. Satisfies the PS requirement "clearly show success, failure, current status, next steps" as a *designed UX element*, not just a status strip field.
2. Makes the product legible to a non-ML judge — they can read it without knowing what "L2 norm" means.
3. Driven entirely from existing `thought` and `intervention` events already emitted by the simulation.

**Implementation:** Add a section in `dashboard.html` that listens for `intervention` events, maps `action` to a human-readable failure type + cause description (small lookup table), and renders the card. No backend changes.

---

### 🔥 Idea 3 — Live Dual Compute Meter *(Waste Visible in Real Time)*

**Effort: ~2h · Works in simulation: ✅**

Instead of a savings ledger that only appears at intervention time, show **two concurrent meters from run start** — so a judge watching the demo sees dollars being preserved continuously.

**UI mockup — during healthy run:**
```
┌─ Resource Tracker ─────────────────────────────────────┐
│  Compute Invested:   ████████░░░░  0h 14m · $0.58      │
│  Currently At Risk:  ░░░░░░░░░░░░  $0.00  (HEALTHY ✅)  │
└────────────────────────────────────────────────────────┘
```

**UI mockup — after recovery:**
```
┌─ Resource Tracker ─────────────────────────────────────┐
│  Compute Protected:  ████████████  0h 14m · $0.58  ✅   │
│  Cost of Recovery:   ██░░░░░░░░░░  0h 02m · $0.08      │
│  Net Saved vs Restart:            0h 12m · $0.50  🎉   │
└────────────────────────────────────────────────────────┘
```

**Why this is better than the current savings ledger:**
- The ledger shows one number at one moment. The dual meter tells the story of the *entire run* — you watch value being preserved continuously.
- "At Risk" field ties directly to the risk score already emitted every step.
- "Net Saved vs Restart" is the number a non-ML judge understands instantly.

**Implementation:** Extend the existing compute-savings ledger logic in `dashboard.html`. Track `runStartTime`, update elapsed cost every second via `setInterval`. On `failure_detected`, record `failureTime` and `failureStep`. On `intervention`, compute net saved. Entirely client-side.

---

### 🔥 Idea 4 — `WASTE_REDUCTION.md` with Concrete Before/After Table

**Effort: ~1h · Required proof layer**

The plan mentions this doc but describes it vaguely. Make it extremely concrete using numbers directly from the simulation run.

**Proposed table structure:**

| Scenario | Without ARC Lens | With ARC Lens | Waste Prevented |
|:---|:---|:---|:---|
| NaN at step 20, 30-step run | Full restart from step 0 | Rollback to step 10, resume | 10 steps = 33% of run |
| Misconfigured Python path | Fail at ~5 min, cryptic traceback | Preflight fails in 2s with named fix | ~4m 58s × N repeated attempts |
| `arc-training` not installed (pre-fix) | Charts show fabricated `Math.random()` data silently | Preflight warns; charts show honest gaps | User trust + correct diagnostic decisions |
| Unrecoverable failure (3 failed rollbacks) | Run continues burning GPU indefinitely | Stop button + waste numbers shown immediately | Minutes to hours of wasted compute |

> **Note on methodology:** All timing figures from the simulation use actual wall-clock measurements from the demo run at `arcAgent.stepDelay=0.08`. Frame them exactly that way — honest, reproducible benchmark.

---

## Section 3 — Revised Priority Stack

```
Rank  Item                                     Effort   Feasible?  Requirement hit
────────────────────────────────────────────────────────────────────────────────────
 1    Status Strip + ARIA live region           2-3h     ✅         Req 2 checklist. Non-negotiable.
 2    Failure Taxonomy Cards                    3-4h     ✅         Req 2 as designed UX. Unique.
 3    Preflight Doctor (preflight.py)           4-6h     ✅*        Best concrete waste story.
 4    Early Warning / Runway Indicator          2h       ✅         No competitor has this. Predictive.
 5    Unrecoverable Stop + waste accounting     2-3h     ✅         Computation waste, tangible numbers.
 6    Dual Compute Meter (live)                 2h       ✅         Visual impact for judges.
 7    Chart <table> equivalents                 2h       ✅         Literal a11y, cheap, real signal.
 8    Keyboard + motion + contrast pass         1h       ✅         Checkbox a11y value.
 9    WASTE_REDUCTION.md                        1h       ✅         Required proof layer.
────────────────────────────────────────────────────────────────────────────────────
Skip  CIFAR-10 cache reuse                      —        ❌         Simulation doesn't use a dataset.
Skip  Adaptive telemetry (emission side)        —        ❌         Needs real runner to measure.
Skip  Checkpoint budget cap / auto-pruning      —        ❌         Wrong layer; needs real GPU.
Skip  LLM context trimming                      —        ❌         Touches paid path; off-PS.
```

> *Preflight Doctor requires writing `python/preflight.py` and modifying `extension.ts`. Non-trivial but the most important waste story to have.

---

## Section 4 — Revised Sequencing

```
Day 1  Status Strip + ARIA live region        (Req 2 done. Everything else builds on this.)
       Failure Taxonomy Cards                  (Works off existing thought/intervention events.)

Day 2  Preflight Doctor                        (preflight.py + extension.ts changes)
       Early Warning / Runway indicator        (Pure JS, fast to build after Day 1)

Day 3  Unrecoverable Stop + waste accounting   (Stop button already exists; wire numbers)
       Dual Compute Meter                      (Extend existing savings ledger)

Day 4  Chart <table> equivalents               (Pure HTML inside dashboard.html)
       Keyboard + contrast pass                (CSS-level changes)

Day 5  WASTE_REDUCTION.md                      (Run the demo, measure, record)
       Adaptive telemetry coalescing           (Receiving-end filter in extension.ts if time)
```

---

## Section 5 — Narrative Reframe for the Pitch

The PS gives you explicit cover to reframe ARC Lens beyond just "ML monitoring tool."

> **"ML training failure is an accessibility problem."**
>
> When a run dies at 3 AM and you're not watching, the tool should be readable to you — and act for you — regardless of whether you're an ML expert, a student training on a free Colab GPU, or a researcher who can't afford to babysit a 48-hour run. Every failed run is a resource taken away from someone who couldn't afford to waste it.

This framing is genuinely unique in the competitive space:
- TensorBoard, W&B, Neptune all pitch to experienced ML engineers.
- ARC Lens can pitch to **ML-adjacent developers, students, and researchers** — people who are *learning* deep learning and can't afford repeated catastrophic failures.

The "accessibility" dual reading in the plan (literal a11y + access to compute) is the right instinct. This narrative makes it explicit and makes it the opening line of the pitch.

---

## Section 6 — What Makes This Submission Win

A judge reviewing this in 60 seconds should see:

1. **Live status strip** — they can read the run state without knowing what gradient norm means.
2. **Early warning indicator** — they see "7 steps to intervention" and understand the tool is *predicting*, not just reacting.
3. **Failure taxonomy card** — they read "Gradient Explosion → Rolled back → Recovered" in plain English.
4. **Dual compute meter** — they watch dollars being preserved in real time.
5. **`WASTE_REDUCTION.md`** — they can verify the claims with actual before/after numbers.

None of items 1–4 exist in TensorBoard, W&B, or any other tool in the competitive landscape table. That is the moat for this round.

---

*Document generated: 2026-08-22*
*Based on: `ROUND2_PLAN.md`, `ARCHITECTURE.md`, `COMPETITIVE_LANDSCAPE.md`, `FUTURE_IMPROVEMENTS.md`, `SECURITY_AUDIT.md`*
