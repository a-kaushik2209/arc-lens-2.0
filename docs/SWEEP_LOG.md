# Sweep log

Every A/B sweep, in order, including the ones that were abandoned.
`docs/experiment_ab.json` only ever holds the **latest** sweep — this file is the record of what
came before it and why each one was superseded. Sweeps 1 onward tested the `loss_plateau` rule;
the *prior era* section below covers the earlier sweeps that produced the numbers still quoted
for the two deleted rules.

It exists because several numbers quoted throughout the docs come from sweeps that are no longer
the shipped artifact. Without this page, a reader checking the JSON against the prose would find
a mismatch and have no way to tell whether it was a stale doc or an invented number. It is
neither, and this is the proof.

**Read sweep 6 before quoting sweeps 3 or 5.** It withdrew the causal claims both of them
produced. Their numbers are real; the attribution of those numbers to ARC's interventions is
not established, because a sweep with nothing intervening reproduced a gap of the same size.

All sweeps: `python python/experiment_ab.py --lrs 0.03 0.1 0.25 0.5 --epochs 10 --seed 1234`,
`train_demo.py` on CIFAR-10, laptop RTX 3050, ~4.5 min per arm, ~36 min for a full 8-arm run.

---

## Sweep 1 — killed, code changed mid-run

**Rule under test:** `loss_plateau`, patience only, no progress guard. Response `reduce_lr`.

Killed partway through. The patience-only rule was firing on a healthy converging run, and I
edited the detector while the sweep was still going — which invalidates every arm after the
edit, since the two arms of a pair would no longer be running the same code. Discarded entirely
rather than partially reported.

**Outcome:** no usable data. Process failure, mine.

---

## Sweep 2 — `lr=0.03` arm only, aborted

**Rule under test:** same as sweep 1.

```
lr=0.03 baseline -> best_val_acc=87.5  failures=2  kinds=loss_plateau
```

A run reaching **87.5% validation accuracy** tripped the rule twice. Convergence is itself a
plateau: the counter keys off the best-ever batch loss, so as a run succeeds its own record gets
harder to beat and stalls grow without bound. No patience value separates the cases.

**Outcome:** false positive confirmed on the first arm; sweep stopped, no point continuing.
Fix was the progress guard — fire only when `best/opening loss > 0.60`, measured 0.271 on a
healthy run against 0.888 on a dead one. See `EXPERIMENT_RESULTS.md` §2b.

---

## Sweep 3 — complete, and it found the second defect

**Rule under test:** patience **+** progress guard. Response `reduce_lr`.

```
|  peak LR | arm      | best val acc | failures | interv. | first fail | verdict       |
|     0.03 | baseline |       87.21% |        0 |       0 |          — | completed     |
|     0.03 | active   |       87.24% |        0 |       0 |          — | completed     |
|      0.1 | baseline |       87.48% |        0 |       0 |          — | completed     |
|      0.1 | active   |       87.62% |        0 |       0 |          — | completed     |
|     0.25 | baseline |       86.83% |        0 |       0 |          — | completed     |
|     0.25 | active   |       87.75% |        0 |       0 |          — | completed     |
|      0.5 | baseline |       73.19% |       10 |       0 |        308 | completed     |
|      0.5 | active   |       10.00% |        4 |       3 |        316 | unrecoverable |
```

The progress guard worked: six healthy arms, rule silent in all of them. The false positive from
sweep 2 was gone.

Then the `lr=0.5` pair, which is the reason this file exists. Per-epoch:

| epoch | baseline (no action) | active (3 × `reduce_lr` from step 316) |
| ---: | :--- | :--- |
| 1 | 10.00%, lr 4.91e-01 | 10.00%, lr 2.45e-01 |
| 2 | 10.00%, lr 4.58e-01 | 10.00%, lr 1.14e-01 |
| 4 | 9.76%, lr 3.34e-01 | 10.00%, lr 4.18e-02 |
| 5 | **26.73%**, lr 2.56e-01 | 10.00%, lr 3.20e-02 |
| 10 | **73.19%** | 10.00%, loss 2.3026 = ln(10) |

Both arms sat at chance for four epochs — the detection was correct. The control arm escaped on
its own once cosine decay walked its LR down; the intervened arm, already an order of magnitude
below that, never did. **−63.19pp.**

> **Sweep 6 withdrew the attribution.** The same configuration split by 62.58 points with
> *nothing* intervening in either arm, so this delta cannot be read as the intervention's
> effect. The conclusion it prompted — report, do not act — survives on different grounds.

**Outcome:** the *response* was falsified. `loss_plateau` became report-only. The full raw
results are kept at
[`experiment_ab_sweep3_reduce_lr.json`](experiment_ab_sweep3_reduce_lr.json) — per-epoch lines
and all — because the numbers quoted in `README.md`,
`EXPERIMENT_RESULTS.md`, `ARCHITECTURE.md`, `FAQ_JUDGES.md`, `FUTURE_IMPROVEMENTS.md` and
`DEMO_SCRIPT.md` for the pre-fix `lr=0.5` comparison all come from here.

---

## Sweep 4 — killed at arm 2, deliberately

**Rule under test:** report-only, first version.

Killed 2 arms in, before any `lr=0.5` arm ran. The report-only path still emitted
`type: "unrecoverable"` after three reports, which would have labelled **both** `lr=0.5` arms
unrecoverable in the results table — including the baseline that recovers to 73.19%. ARC takes no
action on a report-only kind, so it has no evidence the run is past saving, and sweep 3 proves
that claim would have been false.

Killed rather than patched mid-run, because a sweep whose arms ran different code is worth
nothing. Cost ~9 minutes.

**Outcome:** no usable data, by choice. Fix: report-only kinds are tracked in a separate
`silenced_kinds` set and emit `detection_silenced`, which makes no claim about the run's fate.

---

## Sweep 5 — the shipped one, and it found one more defect

**Rule under test:** `loss_plateau` report-only, with `detection_silenced`. This is the sweep in
`docs/experiment_ab.json`.

```
|  peak LR | arm      | best val acc | failures | interv. | first fail | verdict       |
|     0.03 | baseline |       87.53% |        0 |       0 |          — | completed     |
|     0.03 | active   |       87.22% |        0 |       0 |          — | completed     |
|      0.1 | baseline |       87.49% |        0 |       0 |          — | completed     |
|      0.1 | active   |       87.73% |        0 |       0 |          — | completed     |
|     0.25 | baseline |       10.00% |        4 |       0 |        315 | completed     |
|     0.25 | active   |       10.00% |        4 |       0 |        330 | completed     |
|      0.5 | baseline |       75.18% |        5 |       0 |        316 | completed     |
|      0.5 | active   |       30.84% |        8 |       3 |        316 | unrecoverable |
```

**The plateau fix works.** `loss_plateau` fired in four arms and acted in none. `lr=0.25` is the
clearest evidence in this log: both arms collapsed to chance and finished on *exactly* the same
number. A report-only rule cannot produce an arm difference, and it did not.

**`representation_collapse` then did what the plateau rule used to do.** It was the last
structural rule still permitted to act, and it had never fired in any earlier sweep. Here it
fired in both `lr=0.5` arms:

| epoch | baseline (no action) | active (3 × `rollback_and_reduce_lr`) |
| ---: | :--- | :--- |
| 3 | 10.00%, lr 4.04e-01 | 10.00%, lr 4.04e-01 |
| 4 | **19.35%**, lr 3.34e-01 | 10.00%, lr 3.34e-01 |
| 5 | 28.32%, lr 2.56e-01 | 21.44%, lr **3.20e-02** |
| 10 | **75.18%** | **30.84%** |

−44.34pp, apparently the same mechanism as sweep 3: the control arm escaped once cosine decay
lowered the learning rate by itself; the arm ARC cut sat an order of magnitude below that and
never caught up.

> **Sweep 6 withdrew this attribution too**, on the same evidence. Two sweeps showing the same
> pattern looked like confirmation; a third showed the pattern occurs without any intervention
> at all.

Raw results: [`experiment_ab_sweep5_rank_rule.json`](experiment_ab_sweep5_rank_rule.json).

**Outcome:** `representation_collapse` is report-only too. No structural rule is allowed to act
any more. What still intervenes is a non-finite or exploded loss and a gradient norm above 50.

**Also worth recording, because it cuts against the table:** `lr=0.25` finished at 86.83% /
87.75% in sweep 3 and collapsed to chance in both arms here — same seed, same data order. That
is the run-to-run variance of `EXPERIMENT_RESULTS.md` §5 at full size, and it is why the
`0.00pp` delta on that row is the honest reading rather than evidence of anything. The
`−44.34pp` row is different in kind: those two arms diverge *within* the run, at the step the
intervention lands.

---

## Sweep 6 — confirms the fix, and withdraws the causal claim

**Rule under test:** both structural rules report-only. This is the sweep in
`docs/experiment_ab.json`.

```
|  peak LR | arm      | best val acc | failures | interv. | first fail | verdict   |
|     0.03 | baseline |       87.23% |        0 |       0 |          — | completed |
|     0.03 | active   |       87.38% |        0 |       0 |          — | completed |
|      0.1 | baseline |       87.14% |        0 |       0 |          — | completed |
|      0.1 | active   |       87.37% |        0 |       0 |          — | completed |
|     0.25 | baseline |       87.20% |        0 |       0 |          — | completed |
|     0.25 | active   |       86.70% |        0 |       0 |          — | completed |
|      0.5 | baseline |       72.58% |        8 |       0 |        316 | completed |
|      0.5 | active   |       10.00% |        4 |       0 |        316 | completed |
```

**Zero interventions in all eight arms.** That is the change working: both structural rules
detected and neither acted.

**And the `lr=0.5` pair still split by 62.58 points.** Both arms ran identical code on an
identical seed with an identical data order — nothing intervened in either — and their first
four epochs agree to four decimal places:

| epoch | baseline | active |
| ---: | :--- | :--- |
| 2 | train_loss 2.3110, 10.00% | train_loss 2.3110, 10.00% |
| 3 | train_loss 2.3105, 10.00% | train_loss 2.3105, 10.00% |
| 4 | train_loss 2.3098, 10.00% | train_loss 2.3098, 10.00% |
| 5 | **2.1726, 21.07%** | 2.3091, 10.00% |
| 10 | **72.58%** | **10.00%** |

At epoch 5 one escaped the dead region and the other never did. Escape at this learning rate is
**bistable**, and which side a run lands on is decided by floating-point nondeterminism —
non-deterministic cuDNN kernels and non-associative CUDA reductions, which a seed does not fix.

### This withdraws the causal claim from sweeps 3 and 5

Those sweeps attributed −63.19pp and −44.34pp to ARC's interventions, and the per-epoch traces
did show the intervened arms sitting at a much lower learning rate at the moment the control arm
escaped. That reasoning is no longer sufficient, because a gap of the same size occurs with no
intervention at all.

Every `lr=0.5` arm run so far, 10 epochs, seed 1234:

| sweep | arm | interventions | escaped? | final |
| :--- | :--- | ---: | :--- | ---: |
| 3 | baseline | 0 | yes | 73.19% |
| 3 | active | 3 (`reduce_lr`) | no | 10.00% |
| 5 | baseline | 0 | yes | 75.18% |
| 5 | active | 3 (`rollback_and_reduce_lr`) | partly | 30.84% |
| 6 | baseline | 0 | yes | 72.58% |
| 6 | active | 0 | **no** | 10.00% |

Three of four untouched runs escaped; neither intervened run did. That is **consistent with**
the interventions hurting and is **not sufficient to establish it** at these sample sizes. The
honest statement is that a single seeded A/B pair cannot attribute a difference at this learning
rate, because the run-to-run spread is as large as any effect being measured.

`EXPERIMENT_RESULTS.md` §5 said exactly this would happen — "the regime where interventions
matter is by definition near the edge of stability, and that is precisely where tiny numerical
differences amplify" — and it was still worth writing down when it did.

### What does not change

**No structural rule acts.** If anything sweep 6 strengthens that decision rather than
undermining it: at the learning rates where an intervention would matter, the noise floor is
larger than any effect a single pair can measure, so a response cannot be validated this way at
all. Letting a rule act on evidence that cannot exist yet is the mistake this project keeps
finding in itself.

Validating any response needs repeated runs per configuration — `python/repeatability.py --lr 0.5
--repeats N` — reported as a distribution, not a pair. That has not been done and no claim here
depends on it.

The false-positive fix is unaffected: it was measured across six healthy arms in two independent
sweeps, and the rule stayed silent in all of them.

---

## Prior era — the entropy and update-ratio sweeps

These predate the `loss_plateau` work and tested rules that **no longer exist in the code**.
They are recorded here because two of their numbers are still quoted across the docs and appear
in no committed JSON, which would otherwise look like invention.

**The numbers:**

| Figure | Where it came from | Quoted in |
| :--- | :--- | :--- |
| **87.43%** | `lr=0.25` control arm, final entropy-era sweep — the run the `gradient_entropy_collapse` rule took to 10.00% (chance) and then declared unrecoverable | `EXPERIMENT_RESULTS.md`, `FAQ_JUDGES.md`, `SECURITY_AUDIT.md` (C-7), `ARCHITECTURE.md`, `COMPETITIVE_LANDSCAPE.md`, `PITCH_DECK_CONTENT.md`, `DEMO_SCRIPT.md`, `FUTURE_IMPROVEMENTS.md` |
| **76.19%** | `lr=0.25` control arm, an *earlier* entropy-era sweep — the same configuration that later measured 87.43%, which is the 11-point run-to-run gap cited as evidence for seeded pairs being noisy | `EXPERIMENT_RESULTS.md` |

**Why there is no JSON.** `experiment_ab.json` holds only the latest sweep, and these were
overwritten by later ones before the practice of archiving each sweep to its own file started.
The archived files that do exist (`experiment_ab_sweep3_reduce_lr.json`,
`experiment_ab_sweep5_rank_rule.json`) all postdate the entropy rule's deletion.

**Why they cannot be re-measured.** `gradient_entropy_collapse` was deleted from
`_arc_bootstrap.py`. Re-running the sweep today exercises different code and cannot reproduce a
result produced by a rule that is gone. This is not a gap that more GPU time closes — the
measurement is unreproducible by construction.

**How to treat them.** They are real measurements of code that no longer ships, and every claim
resting on them is a claim about *why a rule was deleted* — history, not current behaviour.
Quote them as history and say so. Do not present either figure as a property of the shipped
product, and do not use the 87.43% ↔ 76.19% spread as a current noise estimate: sweep 6 measured
run-to-run spread on the shipped code and that is the number to use.

---

## Sweep 7 — pending

**What it would measure:** repeatability at `lr=0.5`, several runs of one configuration, to get
an error bar instead of an implied one. Until that exists, no delta at this learning rate should
be quoted as an ARC effect in either direction.
