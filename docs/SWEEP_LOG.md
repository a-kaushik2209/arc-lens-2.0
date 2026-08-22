# Sweep log

Every A/B sweep run against the `loss_plateau` rule, in order, including the ones that were
abandoned. `docs/experiment_ab.json` only ever holds the **latest** sweep — this file is the
record of what came before it and why each one was superseded.

It exists because two of the numbers quoted throughout the docs (73.19% and 10.00% at `lr=0.5`)
come from a sweep that is no longer the shipped artifact. Without this page, a reader checking
the JSON against the prose would find a mismatch and have no way to tell whether it was a stale
doc or an invented number. It is neither, and this is the proof.

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

Both arms sat at chance for four epochs — the detection was correct. But the control arm escaped
on its own once cosine decay walked its LR down, and the intervened arm, already an order of
magnitude below that, never did. **−63.19pp from a correct detection.**

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

−44.34pp, identical mechanism to sweep 3: the control arm escaped once cosine decay lowered the
learning rate by itself; the arm ARC cut sat an order of magnitude below that and never caught
up.

**Outcome:** `representation_collapse` is report-only too. No structural rule is allowed to act
any more. What still intervenes is a non-finite or exploded loss and a gradient norm above 50.

**Also worth recording, because it cuts against the table:** `lr=0.25` finished at 86.83% /
87.75% in sweep 3 and collapsed to chance in both arms here — same seed, same data order. That
is the run-to-run variance of `EXPERIMENT_RESULTS.md` §5 at full size, and it is why the
`0.00pp` delta on that row is the honest reading rather than evidence of anything. The
`−44.34pp` row is different in kind: those two arms diverge *within* the run, at the step the
intervention lands.

---

## Sweep 6 — pending

**Rule under test:** both structural rules report-only.

Not yet run. The code change is covered by tests (reverting `REPORT_ONLY_KINDS` in memory fails
the behavioural ones), but no sweep has yet confirmed it end to end, and this row stays here
saying so until one has. The expected result is that the `lr=0.5` pair converges to a single
number the way `lr=0.25` did in sweep 5.
