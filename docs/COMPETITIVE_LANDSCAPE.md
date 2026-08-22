# ARC Lens — Competitive Landscape (Research Brief)

**Purpose:** internal research for pitch-deck prep and judge Q&A. This is deliberately
unflattering where the evidence is unflattering — better to find the holes now than in front
of a judge.

**Scope reminder (what ARC Lens actually does today, per `ARCHITECTURE.md`):** a VS Code
extension that monkey-patches `Optimizer.step` in the user's process, streams loss /
gradient norm / LR / GPU memory / effective rank / gradient entropy / weight-update-ratio to an
in-editor webview dashboard over stdout JSON, and — on a **deterministic threshold rule**
(NaN/Inf loss or `|loss| > 1e6`, grad norm > 50) — rolls the live model back to the last
checkpoint held in memory and cuts the optimizer's LR, in-process, without the user restarting
the script. No *structural* rule is allowed to act any more: the entropy and update-ratio rules
were deleted, and the plateau and effective-rank rules were demoted to report-only after their
responses were measured harming failing runs. See §2. The Pro tier adds a BYOK LLM chat that explains
failures using run telemetry as context, and a script generator.

---

## 1. Comparison Table

| Tool | Live dashboard | Anomaly / failure detection | Automatic recovery (rollback/LR) | IDE-native | Pricing model |
|---|---|---|---|---|---|
| **ARC Lens** | Yes — in-editor webview, ~10 Hz push over stdout | Yes — rule-based thresholds on loss/grad-norm/effective-rank/update-ratio | **Yes** — in-process weight rollback + LR cut, training resumes without a restart | Yes — dashboard is a VS Code webview, launched from the editor | Free core; Pro $2.99/mo (BYOK for LLM chat) |
| **TensorBoard** | Yes, but pull-based (reads event files off disk on a refresh interval, not a push stream) | No built-in anomaly detection | No | Partial — VS Code's Python extension embeds the TensorBoard web UI in a webview | Free, open source |
| **Weights & Biases (W&B)** | Yes — real-time streaming, richest of the hosted dashboards | Yes — `run.alert()` / W&B Automations fire on a metric threshold (e.g., loss > X) | **No** — alerts notify (Slack/email/webhook) or can trigger a *user-written* webhook; W&B does not itself touch model weights or the run process. A "stop run from the API" feature request has been open on their GitHub | No — browser dashboard (desktop app wraps it, but it's the same web UI) | Free tier; Teams ~$50/user/mo; Enterprise custom |
| **Neptune.ai** | Yes — real-time | Basic threshold monitoring / model registry checks | No | No — browser dashboard | From ~$49/mo; free tier for individuals |
| **Comet ML** | Yes — real-time | Yes, but aimed at **production/deployed-model drift** (KS test, PSI, Jensen-Shannon divergence on served data), not training-time gradient pathology | No | No — browser dashboard | Enterprise-oriented, custom/seat pricing |
| **Aim** | Yes (self-hosted) | Basic real-time alerting/notifications | No | No — self-hosted web UI | Free, open source (Apache-2.0) |
| **MLflow** | Weak — primarily a run/artifact log viewer, not a push-streaming live view | No built-in anomaly detection | No | No — browser dashboard (or Databricks-hosted) | Free, open source (Databricks offers managed hosting) |
| **PyTorch Lightning** (`Trainer`, `EarlyStopping`, `detect_anomaly`) | No dashboard of its own (delegates to TensorBoard/W&B loggers) | Yes — `EarlyStopping(check_finite=True)` detects NaN/Inf metrics; `detect_anomaly=True` enables autograd's NaN-source tracing (debug-only, high overhead) | **Stops, does not recover** — both mechanisms halt training; neither restores weights nor resumes | No | Free, open source |
| **MosaicML Composer / Databricks Mosaic AI** | No (pairs with a logger) | Indirect — checkpoint-based | Auto-**resumption** from last checkpoint after a job restart (built for spot-instance/hardware interruption, and cited for LLM loss-spike recovery too) — but this is a whole-job restart, not an in-process, same-run correction | No | Free (Composer OSS); Databricks-hosted training is paid |

**Note on academic prior art (not shipped products, but worth knowing for Q&A):**
- **ZClip** (Kumar et al., arXiv:2504.02507, Apr 2025) and **AdaGC** (arXiv:2502.11034) are
  EMA/z-score-based *adaptive gradient clipping* techniques that reshape gradients before the
  optimizer step to prevent spikes proactively. They never detect a completed failure and never
  roll back weights — they're a preventive alternative to fixed-threshold clipping, not a
  detect-and-recover system.
- Frontier labs' actual practice for LLM pretraining (OPT-175B, PaLM, BLOOM logbooks) is
  **rollback to a checkpoint ~100-500 steps before a spike + LR reduction**, done by a human
  engineer watching dashboards and making a judgment call — the same mechanism ARC Lens
  automates, but as a manual, run-specific decision rather than shipped, reusable software.
- Infra-fault-tolerance systems (ByteDance's training infra paper, TorchPass, TrainMover,
  Chameleon, TierCheck) auto-recover from **hardware/node failures** (GPU falls off the bus,
  spot preemption) via fast checkpoint-restore — a different failure class from gradient/loss
  pathology, solved by different code, and not something an indie/single-GPU user benefits from.

---

## 2. What's Actually Different (the honest version)

The genuinely defensible claim is narrow, so state it narrowly:

> **No dashboard-based competitor surveyed converts a detected training failure into an
> automatic, in-process correction that lets the same run continue.** Every one of them either
> (a) shows you the anomaly and expects a human to act (W&B alerts, Comet drift detection, Aim
> notifications), or (b) stops the run (`EarlyStopping(check_finite=True)`), or (c) recovers from
> a *hardware* fault by restarting the whole job from a checkpoint (Composer auto-resumption,
> the various fault-tolerance infra papers) rather than correcting a *numerical* pathology
> mid-flight.

Two things are true and worth saying plainly:

1. **The rollback+LR-cut idea itself is not novel.** It's exactly what OPT-175B, PaLM, and BLOOM
   engineers did by hand. ARC Lens's contribution is **automating a known manual practice** for
   a Python/PyTorch dev who is not running a frontier lab's monitoring team — not inventing a
   new recovery technique. Frame it that way in the pitch; claiming novelty of the *technique*
   invites a well-read judge to cite the OPT logbook at you.
2. **The detection is rule-based thresholds, not ML.** `ARC_FUNDING_PROPOSAL.md` cites a "97.5%
   accuracy MLP failure predictor" from the core `arc-training` research; that classifier is
   *not* what ships in ARC Lens's live intervention path per `ARCHITECTURE.md` — the extension
   trips on `isnan(loss) or isinf(loss) or |loss| > 1e6` (with `grad_norm > 50` latching clipping
   on once that has fired, rather than as a trigger of its own), and **two**
   report-only structural rules measured relative to each run's own baseline (effective rank
   below 50% of baseline, with that baseline captured only after a 200-step warmup; and a
   loss plateau against the run's own opening loss). If a judge asks "walk me through your ML-based prediction," the honest answer is
   "the shipped detector is deterministic thresholds; the learned classifier is a separate
   research result in the core library, not yet wired into the live product." Decide before the
   pitch whether to mention the MLP result at all — citing it without that caveat is the fastest
   way to lose credibility if probed.

   Note that the live intervention path is now **only** the loss and gradient-norm tests. The
   structural rules detect and report; none of them acts.

   The nuance worth volunteering, because it is the most credible thing here: four structural
   rules have been built, two were deleted outright and the two that remain were stripped of
   the power to act — every one of those decisions forced by measurement.
   The update-ratio rule's distribution overlapped almost completely between healthy and failing
   runs — the healthy run peaked *higher* and breached *longer* — and acting on it cost 1.74 and
   0.78 accuracy points on A/B runs that needed no help. The gradient-entropy rule was worse: it
   fired at step 125 on a healthy CIFAR-10 run and three rollbacks took it from the control arm's
   87.43% to 10.00% — chance — after which ARC declared the run unrecoverable, having caused that
   itself. Measured afterwards, entropy converges to 1.44e-05 on healthy and dead runs alike,
   because the upstream computation bins a heavy-tailed gradient distribution linearly and
   saturates near zero for any run. Both signals are still charted; neither can act. Thresholds
   are baseline-relative *because* absolute ones were wrong too: a fixed `effective_rank < 3.0`
   was tuned on a small MLP and could never fire on a CNN whose real value is around 70.

   Expect the follow-up "so what actually fires?" and answer it before it is asked: numerical
   divergence, which is verified working; gradient clipping; and a **loss-plateau** rule that
   fires when the loss has stalled for 300 steps *and* the run never improved past 60% of its
   opening loss. That last one catches a run that is dead but numerically healthy — measured at
   82 stalled steps on a healthy run against 764 on a dead one, with the progress condition added
   after the patience-only version fired on a healthy 87.5% run — but it detects rather than
   rescues, and it takes **no action at all**. That is measured, not missing: its original
   response cut the learning rate, and on the `lr=0.5` A/B pair that turned a run which recovered
   to 73.19% by itself into one that finished at chance. If a judge asks why a detector does
   nothing, that answer is the strongest thing on this page. The rank rule has **never fired in
   validation** and demonstrably cannot catch that
   case: a dead run only loses 12.6% of its effective rank against a 50% trigger. Presenting
   either as a proven rescue is the fastest way to lose the room.

**What is still a defensible differentiator, stated at the size it actually is:** the *effective
rank* trigger targets **representation collapse**, a failure mode that produces no NaN and no
gradient spike, and is invisible to every loss-curve-only tool in the table
(TensorBoard, MLflow, and the alerting layers of W&B/Neptune/Comet all watch scalar metrics the
user picks — none of them compute effective rank by default). That's a concrete, checkable claim
a judge can verify against the code, unlike "we do anomaly detection" (everyone in the table
does some version of that).

Say "targets", not "catches". The rank rule fired for the first time in a recent sweep, and the
arm it was allowed to act on went from a control-arm 75.18% to 30.84% — so it is now
report-only, like the plateau rule. If pushed on whether the intervention caused that, say no:
a later sweep split the same configuration by 62 points with nothing intervening at all, so the
attribution does not hold and the rule is report-only for the stronger reason that no response
has ever been shown to help. For most runs it does not fire at all: the threshold sits at
50% of baseline while a healthy run bottoms at 97.2%, and it measures weight conditioning rather
than representational rank. The silent failure we *do* catch is a loss plateau, read off the
loss itself rather than off any structural signal.
The claim that survives scrutiny is "we detect a silent failure and report it", not "we recover
from one". We have caught exactly one class in a real run — a model that trained to chance
accuracy with a perfectly finite loss — and the rule that caught it reads the loss, not any
structural signal. It cut the learning rate and the run still finished at 10.00%.

A judge who asks for the evidence gets the full sequence, which is not flattering and is the
point: the entropy rule fired on a *healthy* run and destroyed it, the rank rule has never fired
at all, and the failure that motivated the whole structural tier went undetected for 780 steps
until we measured why and replaced the rule with a simpler one. The differentiator is the
measurement discipline, not a catalogue of caught failures.

---

## 3. Why Now

**GPU cost and scarcity are rising, not falling, in 2026** — the opposite of the "GPUs get
cheaper every year" assumption a skeptical judge might reach for:
- AWS raised H200 instance prices 15% on January 4, 2026 — described as the first GPU cloud
  price *increase* in roughly two decades (IntuitionLabs, H100/H200/B200 pricing guide).
- H100 cloud rental sits at a market median of $2.29–$3.12/hr on-demand in 2026; direct
  purchase runs $25,000–$40,000 per card (CloudZero, Accio H100 market analysis).
- The GPU rental market is valued at ~$52B in 2026, projected to reach ~$199B by 2031
  (30.7% CAGR) — more spend flowing through metered compute means failed runs are metered
  losses, not sunk capital purchased once.

**Wasted compute from failed/interrupted runs is large and increasingly well-documented:**
- Meta's Llama 3 training on 16,384 GPUs over 54 days experienced **419 unexpected
  interruptions** — roughly 8/day (widely cited from Meta's Llama 3 paper, via CoreWeave's
  "Why Distributed Training Fails at Scale").
- CoreWeave reports **10.4% of allocated GPU-hours** wasted to stragglers alone — compute paid
  for that produced nothing.
- Clockwork.io (TorchPass) cites failure costs of **$300K+/month** on a 1,024-GPU cluster, and
  ~3 hours of training progress lost per day without automated recovery tooling.
- Anasim's GPU-failure economics piece notes that **if corruption isn't caught quickly, hours
  or days of progress get discarded once the loss curve finally reveals the anomaly** — i.e.,
  detection lag, not just failure occurrence, is the expensive part. That's precisely the gap
  a live, in-editor dashboard with automatic intervention targets, versus a periodic-refresh
  dashboard a human has to be watching.
- Epoch AI estimates total development compute (including dead-end runs) runs **1.2–4x** the
  final training run's compute — a $40M model can be a $160M project once failed attempts are
  counted.

**Caveat for the pitch:** most of the large, dollar-figure stats above (Llama 3, CoreWeave,
TorchPass, TierCheck, ByteDance's infra paper) are about **large distributed training clusters**
recovering from **hardware/node failures**, which is a different failure class from what ARC
Lens targets (single-process, single-GPU/dev-box numerical instability). Don't present those
figures as "the market ARC Lens addresses" without the distinction — they're evidence that
*training-run reliability is a well-funded, actively-worked problem industry-wide right now*
(the "why now" market-timing signal), not evidence that ARC Lens itself saves that money. ARC
Lens's actual addressable pain is the smaller-scale version: a solo/small-team PyTorch
developer's single run dying to a NaN or silent collapse with nobody watching the dashboard at
3 a.m.

---

## 4. Threats & Objections (with honest responses)

**"Couldn't W&B just ship this next quarter?"**
Technically, yes — W&B already has the telemetry pipe and the alerting/Automations
infrastructure; adding "on alert, call a rollback hook" is a plausible roadmap item, not a
research problem, for a company with a hooks-based Automations system already in place. Honest
response: the moat isn't the mechanism, it's that this is a small, unglamorous feature for a
platform-scale company whose roadmap is dominated by Weave/LLM-agent observability right now —
and it requires them to reach *into* the user's training process (own the checkpoint, own the
optimizer's param groups) rather than just watching metrics from outside, which is a different
integration model than their current SaaS-first architecture. It's a real threat, not a
non-threat; the honest pitch framing is "this is a 6-12 month execution window, not a permanent
moat," and the defensibility has to come from being first, from the IDE-native distribution
(VS Code Marketplace discovery), and from going deeper on PyTorch-specific signals (effective
rank, gradient entropy) than a framework-agnostic platform is likely to prioritize — with the
caveat that going deeper is exactly what removed two of our four structural rules, so "deeper
signals" is a research direction here, not a shipped advantage.

**"Why not just use gradient clipping / `EarlyStopping(check_finite=True)`?"**
Because clipping only addresses gradient-norm blowups (it does nothing for NaN loss that's
already propagated, and nothing for representation collapse, which shows no gradient spike at
all — while conceding that our rank rule for that case detects but no longer acts, because no
measurement shows the response helping and the noise floor where it matters is too large for a
single A/B pair to settle), and
`EarlyStopping` **stops** the run rather than recovering it — the user still loses the
run and has to manually restart, re-tune LR, and hope it doesn't recur. Both are real, standard,
free tools already in every practitioner's toolbox — say so, don't pretend they don't exist.
ARC Lens's pitch is the layer *above* both: catching what clipping misses (NaN, collapse) and
resuming automatically where `EarlyStopping` gives up.

**"Doesn't Composer/torchtitan/big-lab infra already do checkpoint-restart recovery?"**
Yes, at the job level, for hardware failures. That's solving "the GPU died," not "the loss went
to NaN mid-run on a healthy GPU." Different failure surface, different user (distributed
training infra teams at large labs, not a solo dev in VS Code). Say this plainly if asked —
conflating the two is the easiest way to look like the team hasn't done its homework.

**"Is the failure detection actually machine-learned, like the funding proposal implies?"**
No — see §2. Get the story straight *before* the pitch: either drop the 97.5%-accuracy MLP claim
from the deck entirely, or explicitly scope it as "a research result in the core `arc-training`
library, on the roadmap to replace the current threshold rules in ARC Lens" — do not let a judge
discover the mismatch between `ARC_FUNDING_PROPOSAL.md` and `ARCHITECTURE.md` themselves.

**"Can you show it working live, on my script, right now?"**
Yes — and this used to be the weakest answer in the deck. A clean clone now ships the real
harness: `runner.py` executes the user's actual script through `runpy`, unmodified. There is no
simulation build any more, and no divergence between the repository and the published package.
Point them at `python/train_demo.py` (a real CNN on real CIFAR-10 with no injected failure) or
let them run their own script.

**"What happens on multi-GPU / DDP, or a GAN with two optimizers?"**
GANs and multi-optimizer setups are handled. The measurement anchor is `Optimizer.step`, so
`self` is definitionally the right optimizer and `self.param_groups` the right parameters, and
the model is matched by **parameter identity** rather than by grabbing the first `nn.Module` on
the stack — so a discriminator's optimizer cannot be attributed to the generator. Gradient
accumulation and AMP are correct for the same reason, and all three are covered by tests.

DDP/FSDP is genuinely **not** supported, and we say so rather than hedging: it needs rank-aware
emission, `all_reduce` for global gradient norms and a barrier so every rank rolls back to the
same checkpoint, and none of that has been tested on multi-GPU hardware. Today's ARC Lens is a
**single-process, single-GPU tool** (multi-optimizer within that process is fine). That's a
real scope limit, not a secret — better to state it than get caught by a pointed question.

**"Is the rollback actually free / does it not cost anything?"**
No — checkpointing every 10 steps and holding 3 checkpoints in GPU memory has a real memory and
step-time cost (the funding proposal's own overhead numbers show ~10% at 250K+ params, higher at
small scale). That's a fair trade to state confidently, not hide.

---

## 5. Sources

- [Introducing run metrics notifications for W&B Models](https://wandb.ai/wandb_fc/product-announcements-fc/reports/Introducing-run-metrics-notifications-for-W-B-Models--VmlldzoxMjEwMTQxNA)
- [Send an alert — W&B Documentation](https://docs.wandb.ai/models/runs/alert)
- [[Feature]: Stop Training from wandb API · Issue #3366 · wandb/wandb](https://github.com/wandb/wandb/issues/3366)
- [Weights & Biases Pricing 2026 — TrustRadius](https://www.trustradius.com/products/weights-biases/pricing)
- [Weights & Biases Pricing 2026 — AISO Tools](https://aisotools.com/pricing/wandb)
- [Neptune.ai — Pricing Model: Usage & Seat Costs (2026)](https://www.rfp.wiki/artificial-intelligence/data-science-machine-learning-platforms/neptune-ai)
- [Neptune AI Review 2026 — AIChief](https://aichief.com/ai-productivity-tools/neptune-ai/)
- [Comet ML Review 2026 — Model Evaluation Platform, Modern DataTools](https://www.modern-datatools.com/tools/comet-ml)
- [Comet ML Review 2026 — SoftwareSuggest](https://www.softwaresuggest.com/comet-ml)
- [MLflow — Types of AI Experiment Tracking Tools: 2026 Guide](https://mlflow.org/articles/types-of-ai-experiment-tracking-tools-2026-guide/)
- [MLflow — Open Source AI Platform](https://mlflow.org/)
- [GitHub — aimhubio/aim](https://github.com/aimhubio/aim)
- [Aim — AimStack Home](https://aimstack.io/)
- [PyTorch support in Visual Studio Code (TensorBoard integration)](https://code.visualstudio.com/docs/datascience/pytorch-support)
- [Early Stopping — PyTorch Lightning 2.6.1 documentation](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)
- [`detect_anomaly` / autograd anomaly detection discussion — PyTorch Forums](https://discuss.pytorch.org/t/how-to-trace-back-from-anomaly-detection-errors/124051)
- [PyTorch NaNs Are Silent Killers — Towards Data Science](https://towardsdatascience.com/pytorch-nans-are-silent-killers-i-built-a-3ms-hook-to-catch-them-at-the-exact-layer/)
- [Auto Resumption — Composer, Databricks Mosaic AI Training](https://docs.mosaicml.com/projects/composer/en/stable/notes/resumption.html)
- [ZClip: Adaptive Spike Mitigation for LLM Pre-Training (arXiv:2504.02507)](https://arxiv.org/abs/2504.02507)
- [ZClip — GitHub (bluorion-com/ZClip)](https://github.com/bluorion-com/ZClip)
- [AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping (arXiv:2502.11034)](https://arxiv.org/html/2502.11034v3)
- [A Theory on Adam Instability in Large-Scale Machine Learning (arXiv:2304.09871)](https://arxiv.org/pdf/2304.09871)
- [OPT-175B Logbook](https://files.catbox.moe/u1836w.pdf)
- [Training Stability: Loss Spikes, Gradient Norms & Debugging](https://mbrenndoerfer.com/writing/training-stability-loss-spikes-gradient-norm-debugging)
- [Why Distributed Training Fails at Scale — CoreWeave Blog](https://www.coreweave.com/blog/why-distributed-training-fails-at-scale)
- [TorchPass AI Fault Tolerance — Clockwork.io](https://clockwork.io/blog/torchpass-workload-fault-tolerance/)
- [When Your AI Training Cluster Crashes at 3 AM: TrainMover — UCCL Project](https://uccl-project.github.io/posts/continuum-blog/)
- [The Economics of GPU Failure in Data Centers — Anasim](https://www.anasim.com/articles/gpu-failure-economics)
- [AutoClusters: Automated GPU Failure Remediation — Crusoe](https://www.crusoe.ai/resources/blog/autoclusters-minimizing-hardware-failures-in-large-gpu-clusters)
- [Robust LLM Training Infrastructure at ByteDance (arXiv:2509.16293)](https://arxiv.org/pdf/2509.16293)
- [TierCheck: Tiered Checkpointing for Fault Tolerance (arXiv:2605.17821)](https://arxiv.org/pdf/2605.17821)
- [NVIDIA H100 Price Guide 2026 — IntuitionLabs](https://intuitionlabs.ai/articles/nvidia-ai-gpu-pricing-guide)
- [H100 GPU Cost In 2026 — CloudZero](https://www.cloudzero.com/blog/h100-gpu-cost/)
- [AI GPU Rental Market Trends (August 2026) — Thunder Compute](https://www.thundercompute.com/blog/ai-gpu-rental-market-trends)
- [GPU Half-Idle: The Hundred-Billion-Dollar Race — BigGo Finance](https://finance.biggo.com/news/3682e4ece0bc23d0)
