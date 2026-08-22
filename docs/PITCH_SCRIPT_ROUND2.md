# ARC Lens — Live Demo Script

**Team Heisen-bug (U333WKR8) · Challenge #171: Accessibility — Resource Waste Reduction**

Two parts. Part 1 is said before anything is on screen. Part 2 is not a monologue on a clock —
it's commentary keyed to what the dashboard actually does, in the order a real run produces it.
The run is not scripted, so the exact step numbers move a little run to run; the sequence of
events does not. Do the dry run in `DEMO_SCRIPT.md` §1.4 so you know your numbers before you're
live.

Every figure traces to `docs/WASTE_REDUCTION.md`.

---

## Part 1 — Before opening anything

**Problem.** Training runs fail without crashing. A learning rate slightly too high sends the
loss to NaN, and the process stays alive, the GPU stays at 100%, the job still says "running" —
nothing in the stack says it's dead. On our own demo run, ARC knew the run had failed at 15.3
seconds. The run kept going to 86 seconds. 82% of that run's compute was spent after the answer
was already known.

**Solution.** ARC Lens is a VS Code extension. It hooks a PyTorch training loop with zero code
changes. When the loss goes non-finite, it rolls the model back to the last healthy checkpoint,
cuts the learning rate, latches gradient clipping on, and lets the same run continue — no
restart, no human in the loop. On a seeded A/B, that took a run from 10.00% accuracy, chance, to
46.59%. For a failure it detects but can't safely fix — a run pinned at chance for four epochs,
looking completely healthy on any ordinary chart — it says so instead of guessing at a response.

**Challenge #171.** The brief asks for the part of the MVP closest to accessibility to cut
waste, and for success, failure, status and next steps to be visible. Those aren't two
deliverables here. ARC can't fix the failure above — every response we tried made it worse — so
what the product produces for it is not a rescue, it's a person told in time. The status strip,
the live region, the named cause on a failed preflight, the data table behind every chart: for
that failure, that's the whole feature, and it's what turns 15 seconds of detection into
something acted on instead of 70 more seconds burned.

Opening it now.

---

## Part 2 — The demo, as it happens

| When this happens on screen | Say this |
|:---|:---|
| Script open, before clicking Run | "Ordinary PyTorch loop. `zero_grad`, forward, `cross_entropy`, `backward`, `optimizer.step`. No ARC import, no callback, no wrapper. The only unusual thing is the learning rate — 5.0, which is well past stable for this network. Nothing is injected. ARC has to find this failure, not be told where it is." |
| Click ▶ Run with ARC Lens | "Zero code changes to get here." |
| Dashboard opens, panels populate | "Loss, learning rate, gradient norm, GPU memory — live. Resources Conserved panel bottom right, counting from zero." |
| First few steps tick by | "It hooks `Optimizer.step`, not `backward()`. One recorded step is one weight update — that's what keeps gradient accumulation, mixed precision and multi-optimizer GANs correct instead of just not-crashing." |
| Loss goes non-finite (around step 6) | "There it is. Loss just went non-finite." |
| Action log: rollback + reduce_lr + grad clipping | "Read the log: rolled back to the last checkpoint, cut the learning rate, clipping latched on. Same process. I didn't touch anything. That's the only path where ARC is allowed to act — an unambiguous divergence." |
| Loss climbing back down over subsequent steps | "It's recovering off that rollback now." |
| Val accuracy ticking up epoch to epoch (10% → 18% → 32% → 46%) | "This is the number that matters. A run I've run before with this exact setup, interventions switched off, sits at 10.00% — chance — for all five epochs, because the loss goes non-finite at this same step and nothing brings it back. This one's climbing past 46 and still rising when the schedule ends." |
| If a `loss_plateau` fires later in the run | "Different failure, and watch what happens — nothing. No rollback line. That's deliberate: this rule used to cut the learning rate, and in testing that made a recoverable run worse, not better. So it reports the diagnosis and stops. It's the only honest thing to do when you've measured that your fix doesn't help." |
| Resources Conserved panel, run either running or finished | "Steps not re-run, time not re-spent, compute not re-bought — all measured off this run, not assumed. Every number on this panel is labelled measured, derived or estimated, in the page itself, because we audited it and found four places it was displaying numbers it didn't actually have — including a risk score that rendered as a safe 0.00 when the value was simply missing. Fixed, and there's a test for it now." |
| Run complete | "That's the whole loop. Detect, act if it's safe to act, say so if it isn't." |

---

## Close

Half of what we built, we deleted or demoted, because our own measurements said it didn't work.
Two structural rules got deleted for firing on healthy runs — one took a run from 87% down to
chance. Two more got demoted to report-only after we measured their fix making things worse.
Every one of those sweeps is in the repo, including the ones we killed mid-run and the causal
claims we withdrew after they didn't replicate.

What's left is narrow and it's real. For the failure ARC can fix: chance to 46% on the same
seed. For the failure it can't: told at second 15 instead of second 86, and told it can't fix it.

Anything that touches someone's training run has to earn that with evidence. Half the time ours
said no.

---

## Rubric coverage

| Parameter | Carried by |
|:---|:---|
| Task implementation | The A/B result; telemetry −72%; a11y 87→100; all four required states named on screen |
| Task complexity | `Optimizer.step` anchor; in-process rollback against a live LR scheduler; GAN/AMP/accumulation correctness |
| Technical execution | 136 tests, ten end-to-end, a TDD-found GAN bug, CI secret scan, package builds clean on a second machine |
| Innovation & creativity | Acts instead of alerting; legibility is the waste-reduction mechanism, not decoration around it |
| Functionality & reliability | +36.6 points on a seeded A/B; unrecoverable verdict instead of infinite retry |
| Documentation & presentation | On-screen provenance for every figure — five fabrications found and fixed; every sweep published, withdrawals included |
| Architecture | Three tiers, one measurement anchor |
| Code quality | Tests written before the code; the fabrication audit and its regression tests |
| User experience | One button, zero code changes, named cause and named fix |
| Scalability | Adaptive telemetry, checkpoint byte budget, 1.8% overhead, BYOK zero token cost |
| Technical sophistication | The anchor choice; the A/B methodology that deleted our own rules |

---

## Demo notes

Default `train_demo.py` is `ARC_DEMO_LR=5.0`, warmup 5. A default run produces **both** failure
kinds: `numerical` around step 6 — ARC acts — then a `loss_plateau` later, which it deliberately
doesn't. Do **not** demo at `ARC_DEMO_LR=0.5` (the old default); it only produces `loss_plateau`,
so the rescue path never fires and nobody sees a recovery.

The 10.00%-vs-46.59% comparison is not run side by side live — it's the baseline arm from a prior
seeded run, quoted from memory or a screenshot. Know the number before you say it:

```bash
ARC_MODE=baseline python python/runner.py python/train_demo.py   # ends at 10.00 %
ARC_MODE=active   python python/runner.py python/train_demo.py   # ends at ~46 %  — the live demo
```

---

## Q&A

| Ask | Answer |
|:---|:---|
| Is one seeded pair enough? | No, and we say so. Sweep 6 is why: at lr=0.5 the run-to-run spread swamped effects this size. This case is more defensible — a deterministic explosion at step 6, not a bistable escape regime — but no distribution over repeated runs has been measured and no error bar is claimed. |
| 46% is a bad CIFAR-10 result. | It is. Rescued run, not a tuned one. The claim is the compute was preserved, not that the hyperparameters were fixed. |
| Distributed training? | Single-process, single-GPU today. Multi-optimizer within that process is tested. DDP/FSDP needs rank-aware emission and a barrier so every rank rolls back to the same checkpoint — not run on multi-GPU hardware, so not claimed. |
| Does preflight save time? | No. Measured, and it didn't — `runner.py` surfaces the same errors in about the same time. It saves effort, not seconds: a named cause and a named fix instead of a traceback. Published as a null result. |
| Why report-only instead of acting? | Because the fix couldn't be attributed. At the learning rate where it matters, two identical runs — same seed, same data order — split by 62 points on floating-point nondeterminism alone. |
| Walk me through the ML prediction. | The shipped detector is deterministic thresholds. A learned classifier exists in the core research library; it isn't wired into the live path, so it isn't claimed. |
| Is the telemetry saving on by default? | No. `arcAgent.telemetryEvery` defaults to 1. −72% is what it buys at 10, and risk detection is unaffected either way — loss history and risk are computed every step regardless. |
| Business model? | Core free — dashboard and recovery. Pro $2.99/month for the LLM failure analyst, bring-your-own-key, so no token cost carried. |
