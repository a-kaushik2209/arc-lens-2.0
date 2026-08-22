# ARC Lens — Live Demo Script

Operational script for whoever is at the keyboard during the presentation.

**What changed since the last version of this document:** there is no simulation build any
more. The old demo relied on a scripted NaN at step 20, which meant the timing was predictable
but the demo was not real. It is now a genuine CIFAR-10 training run, so **the exact step at
which it fails is not known in advance**. That is a better demo and a slightly riskier one, and
this script is written around that trade.

---

## 1. Pre-demo checklist

Do this at least 30 minutes before your slot, not 2 minutes before.

1. **Install and compile**
   ```bash
   npm install && npm run compile
   npm test                        # 45 tests, all green
   ```
   On Windows, if `npm run compile` reports `'tsc' is not recognized`, `npm install` was run
   from a shell that skipped the `.cmd` shims. Re-run `npm install` from PowerShell or cmd —
   it regenerates `node_modules/.bin/tsc.cmd` in a couple of seconds. Worth knowing because it
   looks like a broken repository and is not one.

2. **Install the Python side into the interpreter VS Code has selected**
   ```bash
   pip install torch arc-training torchvision
   python tests/test_harness.py    # 54 tests, ~50s
   ```
   Both suites green means the harness, the detector, checkpointing and the end-to-end path
   all work on this machine. This is the single best use of five minutes before a demo.

3. **Pre-download CIFAR-10.** `train_demo.py` downloads it on first run (~170 MB). Do not let
   that happen on conference wifi in front of judges.
   ```bash
   python -c "from torchvision import datasets; datasets.CIFAR10('./data/cifar', download=True)"
   ```

4. **Do one full dry run of the exact thing you will demo**, end to end, and note the step at
   which it actually fails. Seeded runs are reproducible, so the number you see in the dry run
   is the number you will see live.
   ```bash
   ARC_DEMO_EPOCHS=3 ARC_DEMO_LR=0.25 python python/runner.py python/train_demo.py
   ```

5. **Check the GPU is visible.** The `environment` event at the top of the stream names the
   device. If it says `null`, you are on CPU and everything will be far slower.

---

## 2. The 60-second version

If you have one minute, do this and nothing else.

> "This is a real ResNet-style CNN on real CIFAR-10. Nothing here is simulated — no injected
> failure, no scripted curve."

Click **▶ Run with ARC Lens**. While it warms up:

> "The learning rate is deliberately too high. That is the most common way real runs die, and
> it's a mistake, not a special case."

When the failure marker appears:

> "There's the failure. ARC caught it, rolled the weights back to the last healthy checkpoint,
> cut the learning rate, and resumed — the training loop didn't stop and I didn't touch it."

Point at the shaded band between the red and green markers:

> "That shaded region is how long the run spent broken. The number underneath is the training
> time that rollback preserved, priced at this GPU's hourly rate."

---

## 3. The full version (~4 minutes)

### 3.1 Open the script first, not the dashboard

Show `python/train_demo.py` and scroll to the training loop.

> "This is an ordinary PyTorch loop. `zero_grad`, forward, `cross_entropy`, `backward`,
> `optimizer.step`. There is no ARC import, no callback, no decorator, no wrapper. That matters
> for the next thing I'm going to say."

Scroll to the hyperparameters.

> "The only unusual thing is the learning rate — 0.5 with SGD momentum on a 9-layer CNN, with
> a very short warmup and no gradient clipping. That's past the edge of stability. Whether it
> diverges and when depends on the data order and the initialisation."

### 3.2 Run it

Click **▶ Run with ARC Lens**.

> "ARC hooks `Optimizer.step`, not `loss.backward()`. That sounds like a detail and it is
> actually the whole design. One optimizer step is one weight update — which is what makes this
> correct under gradient accumulation, under mixed precision, and on GANs with two optimizers.
> A backward-based hook gets all three of those wrong."

Point at the four charts while it runs.

> "Loss and learning rate. Gradient norm and gradient entropy. Effective rank and weight update
> ratio. Those last four come from `arc-training`. Effective rank is the one wired to an
> intervention; the other three are there for you to read, and we can say exactly why — two of
> them used to trigger and we deleted both after measuring what they did to healthy runs."

Do not claim the four signals catch failures a loss curve cannot show. One of them is wired to a
trigger; it fires rarely, and when it did fire, acting on it cost 44 points of accuracy against
the control arm — so it now reports without acting. The silent failure we *do* catch is a loss
plateau, and that is read off the loss itself, not off these four. If a judge pushes, Q5 in
`FAQ_JUDGES.md`, and it is a better story than the claim it replaced.

### 3.3 The failure

When the red marker lands, stop talking about architecture and read the Action Log aloud. The
log names the failure kind and the reason — quote it verbatim, it is more convincing than
paraphrase.

> "It rolled back N steps and cut the learning rate from X to Y."

That line is for a `numerical` failure. **A `loss_plateau` produces no intervention at all** —
the Action Log will show the detection and an explicit "reporting without acting" entry, and
there is no rollback or LR change to read out. Do not promise one and then have the log
contradict you on screen; the reason it takes no action is the strongest part of the story
(see §3.3 below).

**Expect `numerical` or `loss_plateau`.** Those are the two kinds this demo produces and the
only two verified working. At `ARC_DEMO_LR=0.5` the run dies silently rather than exploding, and
the marker you get is `loss_plateau` ("stalled" on the chart) at around step 316–330.
`gradient_entropy_collapse` no longer exists — deleted after it took a healthy CIFAR-10 run from
87.43% to chance — so do not build the talk track around it. `representation_collapse` still
exists, fires rarely, and no longer acts — the one sweep where it fired, its rollback took a run
that recovered to 75.18% on its own down to 30.84%. Do not promise it as a rescue.

**If you demo the plateau, say what it does and does not do.** It reports the death; it does not
reverse it. The run still finishes at 10.00%. Claiming a rescue here is the one thing that will
get caught, because the accuracy is on screen.

If a judge asks whether ARC catches failures that produce no NaN, answer honestly rather than
reaching for a marker that will not appear:

> "Yes, one — a loss plateau. We had a run finish at chance accuracy with its loss sitting at
> ln(10), perfectly finite, and ARC reported nothing for 780 steps. That is the failure this
> tool exists to catch and it missed it. So we measured: a healthy run's longest stall is 82
> steps, the dead one's is 764. That rule is in now, and it fires at step 316.
>
> "And it was wrong twice. Over a longer run it fired on a healthy model at 87.5% —
> convergence looks exactly like a plateau. So it now also checks whether the run ever improved
> at all: dead runs stall having gone nowhere, converged ones stall having gone a long way.
>
> "Then the sweep said the *response* was wrong. It used to cut the learning rate. On the
> lr=0.5 arm the control sat at chance for four epochs and then escaped on its own — cosine
> decay walked the LR down and it climbed to 73%. The arm we intervened on had already been
> cut an order of magnitude below that and never escaped: 10%, all ten epochs. The detection
> was right and the fix made it 63 points worse. So it reports now and touches nothing."
>
> It detects it — it does not save it, deliberately. Rolling back is no better: by the time 300
> stalled steps confirm a plateau, every checkpoint we hold is already post-collapse. Neither
> response we have is known to help, and we have a measurement saying one of them hurts.
>
> There is also a rank rule, still in, still never fired — a dead run only loses 12.6% of its
> effective rank, so it cannot reach a 50% threshold. We left it conservative rather than tuning
> it to a knife-edge, because the last rule we tuned that way took a healthy run to chance."

### 3.4 The proof

This is the part that separates you from every other monitoring demo.

> "You have no reason to believe the run would have died without this. So here's the control."

Run **ARC Lens: Run Baseline (interventions off)**, or show the pre-generated table from
`docs/EXPERIMENT_RESULTS.md`.

> "Same script, same seed, same data order, same instrumentation — every intervention
> suppressed. The baseline curve is the dotted red line. That's the counterfactual, measured,
> not argued."

### 3.5 Close on the report

Click **Export Report**.

> "One self-contained HTML file. No network, no dependencies, opens in five years. This is the
> shape an incident report takes at an infra team, and it's what makes this feel like
> infrastructure rather than a toy."

---

## 4. Anticipated questions

**"Is this a simulation?"**
No. Real CIFAR-10, real CNN, real GPU, no injected failure. The repository ships the harness
that is running right now — there is no separate "real" build. Earlier versions of this project
did ship a simulation and we removed it; it is finding H-4 in our own audit.

**"Is that an LLM agent making these decisions?"**
No, and we're specific about it. The recovery loop is a deterministic rule engine that renders
its decisions as a ReAct trace. That's deliberate: the reflex path must not depend on a network
call, and a NaN needs handling in microseconds. The LLM is in the Failure Analyst panel, where
latency is fine. `arc_agent.py`'s docstring says exactly this.

**"How much does the monitoring cost me?"**
Measured, not estimated: 1.8% for core metrics, 8.4% with the structural diagnostics, on an
RTX 3050 with a 2.79M-parameter CNN. Sampling those signals every step instead costs 170%,
which is why they're sampled every 25 steps and only densify while risk is elevated.
`python python/benchmark_overhead.py` reproduces the table.

**"What if ARC is wrong and hurts my run?"**
Two answers. First, the detector is guarded against exactly that — a high update ratio does not
fire while the loss is still improving, because large steps early in training are normal and
firing there would "rescue" runs that were working. A healthy run to 87% accuracy produces zero
interventions. Second, `ARC_MODE=baseline` lets you measure it yourself, which is the honest
version of this answer.

**"What if it can't fix the run?"**
Then it says so and stops. After three failed recoveries of the same kind it emits an
unrecoverable verdict and stops intervening, because once a network has genuinely collapsed
every checkpoint still in the ring is collapsed too. Our own results include those cases —
we didn't cut them.

**"Does it work on my ResNet?"**
Yes, and there's a story there. `arc-training`'s gradient collector used to crash on any model
with `inplace=True` activations — which is torchvision's ResNet, VGG and MobileNet as shipped.
We only found it because we replaced the silent `except: pass` with visible degradation. A
monitoring tool that crashes the run it's monitoring is the worst possible bug, and it was
invisible by construction.

**"The structural charts are empty."**
Then `arc-training` isn't importable in the selected interpreter, and the dashboard says so
rather than drawing something. An earlier version filled those charts with `Math.random()`;
that's finding C-2 in our audit and it's gone, with a test asserting it can't come back.

---

## 5. If something goes wrong

**No failure occurs during the demo.** Possible on a short run. Say so plainly — "this seed
happens to survive; the interesting behaviour is the detection, and here's the run where it
fires" — and switch to `docs/EXPERIMENT_RESULTS.md`. Do not pretend a healthy run is a rescue.

**A DEGRADED badge appears.** Hover it; it names the component. Most likely `arc-training` isn't
installed or a collector failed. This is the tool working as intended — say that, don't hide it.

**The run is too slow to show.** Cut `ARC_DEMO_EPOCHS`. Do not raise `arcAgent.stepDelay`; it
defaults to 0 now precisely because a non-zero value is real GPU time thrown away for pacing.

**Nothing renders at all.** Charts are vendored locally, so this is not a network problem.
Reload the webview (Developer: Reload Window). If it persists, the dry run in §1.4 would have
caught it — which is why §1.4 exists.
