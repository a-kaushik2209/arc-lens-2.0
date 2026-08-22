# ARC Lens — Live Demo Script

**Team Heisen-bug (U333WKR8) · Challenge #171: Accessibility — Resource Waste Reduction**

This is written to be practiced word for word, for a room that knows nothing about machine
learning. Every technical term gets explained the first time it's said — you don't need to
assume anyone in the audience trains models. `[Brackets]` are stage directions, not spoken.

Two parts. **Part 1** is said before you touch the keyboard. **Part 2** is not a monologue on a
clock — it's what you say as the dashboard actually does things, in the order a real run
produces them. The run isn't scripted, so exact step numbers shift a little each time; the
sequence of events doesn't. Do a dry run first (`DEMO_SCRIPT.md` §1.4) so the numbers you say
out loud are numbers you've actually seen.

Every figure traces to `docs/WASTE_REDUCTION.md`.

---

## Part 1 — Before you open anything

Here's a question for you. Has anyone here ever left something running overnight — a big
download, a render, anything — and come back the next morning to find it had failed six hours
in, and you just... didn't know until now?

[Pause. Let it land, whether or not anyone answers.]

That's the entire premise of what we built, except the thing that fails is more expensive, and
it fails more quietly.

So — quick version of what "training a model" even is, because I want everyone in this room
with me for the next few minutes, not just the ML people. You've got a program that's trying to
learn a pattern — say, "what does a cat look like in a photo." It does this by adjusting a huge
pile of internal numbers, a little bit, thousands of times, based on how wrong its last guess
was. That "how wrong" number is called the **loss**. Lower loss, better model. Every one of those
thousands of adjustments is called a **step**. That's the whole loop: guess, measure how wrong,
nudge the numbers, repeat.

Here's the part nobody tells you when you start: that loop breaks. Constantly. Not with a
crash — with silence. One of those nudges can be too aggressive, the internal numbers spiral
into garbage, and the program keeps running, keeps saying "step 4,001... step 4,002..." for the
next six hours, producing absolutely nothing useful. No error. No red text. It just quietly
stopped being training.

We measured exactly this on our own demo run, the one I'm about to show you. Our tool knew the
run was dead at fifteen seconds in. The run itself kept going for another seventy. That's
**82% of the compute spent after the answer was already known** — and that ratio doesn't care
how long the run is. Stretch that to a real 48-hour training job and 82% is closing in on two
full days of GPU time spent training nothing, because nobody was watching closely enough, fast
enough, to pull the plug.

So — what did we actually build?

ARC Lens is a VS Code extension. You open your training script, you click one button, and it
watches the run live — no code changes, nothing to import, nothing to configure. The moment the
loss goes non-finite — programmers call that a **NaN**, "not a number," which is exactly what it
sounds like, the math has broken — ARC Lens rolls the model back to the last point it was
healthy, something we call a **checkpoint**, turns down how aggressively it's learning, and lets
the same run keep going. No restart. Nobody has to be at the keyboard. We tested this pair
side by side, same starting conditions: the run left alone finished at 10% accuracy, which for
this dataset is literally a coin flip with ten sides — random guessing. The run ARC caught and
fixed finished at 46.59%, and was still climbing.

Now — here's the honest part, and it's the part I actually want you to remember. Not every
failure is one ARC can safely fix. We have a run that goes quietly wrong — never crashes, never
throws an error, just sits at random-guess accuracy for hours looking completely normal on any
ordinary chart — and every fix we tried for it made things *worse*, not better. So for that one,
ARC doesn't guess. It tells you, clearly, immediately, "this is dead, I'm not touching it,"
instead of pretending it has a solution it doesn't.

Quick show of hands, actually — who here has heard today's challenge is about accessibility and
reducing waste, and thought those sound like two different problems?

[Pause.]

They're not, in this product. If the tool can't fix a failure, the only thing left that matters
is whether a human can *see* it fast enough to act — clearly, in plain language, the moment it
happens. That's accessibility work: a status strip that always says what's happening, screen
reader support, a table version of every chart. And it turns out that's *also* the waste
reduction. The faster a person can read "this run is dead," the less compute gets burned after
they already knew.

Alright — let's go watch it actually happen.

---

## Part 2 — The demo: say this as it happens

[Have the script open, dashboard closed, before you start talking.]

| When this happens on screen | Say this |
|:---|:---|
| Script open, before clicking Run | "This is a completely ordinary training script — nothing special, no ARC code in it anywhere. The one deliberately reckless thing in here is how aggressively it's set to learn — turned up higher than this network can actually handle. That's not a rigged demo. That's the single most common way real training runs die: someone copies a setting from a paper or a tutorial that used a different model, and it's just slightly too aggressive for theirs." |
| Click ▶ Run with ARC Lens | "One click. That's the entire setup." |
| Dashboard opens, panels start filling in | "This is everything ARC is watching, live: the loss — remember, that's 'how wrong is it right now' — how aggressively it's learning, how big its corrections are getting, and down here, bottom right, a running tally of exactly what this tool is saving you, ticking up from zero as we go." |
| First several steps tick by, numbers still sane | "Right now everything's fine. It's just learning. This is the boring part, and boring is what you want — boring means healthy." |
| Loss goes non-finite — watch for it, usually within the first handful of steps | "There. Right there. See how the loss just went from a normal number to garbage? That's the moment I told you about — the math broke. On literally any other tool, this is where someone finds out six hours from now." |
| Action log shows the rollback line | "Read this with me: it rolled back to the last point where things were healthy, turned down the learning rate, and turned on a safety limiter for future steps. I have touched nothing. The training loop is still running, right now, on its own." |
| Loss dropping back down over the next several steps | "And there it is recovering. Same run, same process, no restart." |
| Accuracy climbing across epochs — you'll see it move roughly 10 → 18 → 32 → 46 percent | "Watch this number specifically — it's the one that actually matters. I ran this exact setup once with the safety net switched off, nothing intervening. That run sat at 10% — random guessing — for the entire time, because it never got the chance to recover. This run is already past 46% and it's still going up when we stop watching." |
| If a `loss_plateau` warning appears later in the run | "Different situation, and I want you to notice ARC does *nothing* here — no rollback line, nothing. This is that quiet failure I mentioned at the start: it's not crashing, it's just stuck. We tried fixing this automatically, measured what happened, and our fix made it worse. So now it just tells you, honestly, 'I see this, I'm not going to guess at a fix' — and that's not a missing feature, that's the more trustworthy behavior." |
| Pointing at the Resources Conserved panel, any time it has real numbers | "Every number here — steps that didn't need to be re-run, time not re-spent, compute not re-bought — is measured off this exact run, not estimated after the fact. And I mean that literally: we went back and audited every single number this dashboard shows, and found four places it was quietly showing a number it didn't actually have — including a safety gauge that showed 'safe' when the real answer was 'we don't know.' We fixed every one, and wrote a test so it can't happen again quietly." |
| Run finishes | "That's the whole thing. Watch, catch it, fix it if fixing it is safe, and if it isn't — say so, out loud, immediately, instead of guessing." |

---

## Close

Here's what I actually want you to take away, more than any single number.

About half of what we originally built, we ended up deleting or turning off — not because
someone else told us to, but because we tested it against a control group and it made things
*worse*. One rule we built took a perfectly healthy run and dropped it to random guessing, by
itself. We found that, and we killed it. Every single one of those tests is sitting in this
repo right now, including the ones where we were wrong first.

So what's left standing is small, and it's real. For the failure this tool can fix: random
guessing to 46%, on the exact same starting conditions. For the failure it can't: told at
fifteen seconds, not seventy, and told honestly that it can't fix it.

If something's going to reach into your training run and change it, it should have to earn that
right with evidence first. Ours didn't always earn it. That's in here too.

Thank you — happy to take questions.

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
