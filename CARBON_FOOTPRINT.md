# ARC Lens — Carbon & Energy Footprint Tracking

## What Is Carbon Footprint in ML Training?

Every training run consumes electricity. That electricity has a carbon cost — grams of CO₂
emitted at the power plant for every kilowatt-hour the GPU draws. A single CIFAR-10 run on a
laptop GPU is negligible. But the same user running 40 experiments a week, or a team running
distributed training on 8× A100s for days, accumulates a measurable environmental cost —
and a measurable dollar cost — that no tool currently surfaces.

The standard ML monitoring stack (TensorBoard, W&B, Neptune, Aim) shows loss curves, gradient
norms, GPU utilisation. None of them answer: *"How much energy did this run actually consume,
and how much of that was wasted on a run that was failing?"*

ARC Lens already answers the dollar question (the compute-savings ledger). Adding energy and
carbon turns the same data into environmental accountability — and it costs almost nothing to
implement, because every input is already available.

---

## Why This Matters for the Challenge

> *"Improve the part of your existing MVP most related to accessibility so that it can reduce
> unnecessary use of time, bandwidth, storage, **computation**, or repeated user effort."*

Energy tracking makes computational waste *tangible in physical units*. A judge may not
intuitively know whether "$0.21 of GPU time" is a lot. Everyone understands "left a phone
charging for 17 hours" or "drove 0.4 km in a car."

It also serves the four-state UX requirement:

| State | What Energy Shows |
|:---|:---|
| **Success** | "This run consumed 0.08 kWh — equivalent to 35g CO₂" |
| **Failure** | "0.04 kWh were consumed after the run became unrecoverable" |
| **Current status** | Live energy counter updates every step |
| **Next steps** | "Stop now to save ~0.02 kWh" |

---

## What Data We Already Have

ARC Lens already captures everything needed. Zero new sensors, zero new APIs, zero backend
changes to `_arc_bootstrap.py`.

### From `_arc_bootstrap.py` → `install()` (lines 1281–1317)

The `environment` event, emitted once at run start:

```python
emit({
    "type": "environment",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": device,          # ← torch.cuda.get_device_name(0)
    "arc": ...,
    "mode": MODE,
    "python": sys.version.split()[0],
})
```

The `gpu` field carries the device name string (e.g. `"NVIDIA GeForce RTX 3050"`) that the
dashboard already uses to look up the GPU hourly rate.

### From `_arc_bootstrap.py` → `finish()` (lines 1331–1361)

The `run_summary` event, emitted once at run end:

```python
emit({
    "type": "run_summary",
    "steps": STATE.step,
    "wall_seconds": round(wall, 3),
    "interventions": STATE.intervention_count,
    "instrumentation_seconds": ...,
    ...
})
```

`wall_seconds` is the total wall-clock time of the run.

### From `_arc_bootstrap.py` → `_RunState` (line 210)

```python
self.train_start = time.time()
```

This is the run start timestamp, used to compute elapsed time at any point during the run.

### From `dashboard.html` → `GPU_RATE_TABLE` (lines 493–501)

```js
const GPU_RATE_TABLE = [
  [/h100/i,            3.50, 'H100 on-demand list'],
  [/a100/i,            2.00, 'A100 on-demand list'],
  [/l40|l4\b/i,        1.00, 'L40/L4 on-demand list'],
  [/a10\b|a10g/i,      0.75, 'A10G on-demand list'],
  [/v100/i,            0.90, 'V100 on-demand list'],
  [/t4\b/i,            0.35, 'T4 on-demand list'],
  [/rtx\s*(40|30)\d0/i, 0.20, 'consumer GPU, cloud-equivalent estimate'],
];
```

This table already matches the GPU name to a cost tier. We extend it with a second column
for power draw.

### From `dashboard.html` → `updateSavingsLedger()` (lines 538–568)

The ledger already computes `elapsedMs`, `gpuRate`, `survivingFraction`, and `preservedMs`.
Energy is one more multiply on the same inputs.

---

## The Implementation

### Step 1: Add a GPU TDP Table (Dashboard JS)

This table sits alongside the existing `GPU_RATE_TABLE`. It maps GPU name patterns to their
**Thermal Design Power (TDP)** in watts — the maximum sustained power the GPU draws under
load. These are published specifications from NVIDIA; they do not change.

```js
// GPU TDP in watts — NVIDIA published specifications.
// The TDP is an upper bound on sustained power draw; actual draw under
// training load is typically 70–90% of TDP. We use TDP and label it as
// such, so the figure is honestly an upper estimate.
const GPU_TDP_TABLE = [
  [/h100.*sxm/i,     700, 'H100 SXM TDP'],
  [/h100.*pcie/i,    350, 'H100 PCIe TDP'],
  [/h100/i,          700, 'H100 TDP (assumed SXM)'],
  [/a100.*80/i,      400, 'A100 80GB TDP'],
  [/a100/i,          300, 'A100 40GB TDP'],
  [/l40/i,           300, 'L40 TDP'],
  [/l4\b/i,           72, 'L4 TDP'],
  [/a10g/i,          150, 'A10G TDP'],
  [/a10\b/i,         150, 'A10 TDP'],
  [/v100/i,          300, 'V100 TDP'],
  [/t4\b/i,           70, 'T4 TDP'],
  [/rtx\s*4090/i,    450, 'RTX 4090 TDP'],
  [/rtx\s*4080/i,    320, 'RTX 4080 TDP'],
  [/rtx\s*4070/i,    200, 'RTX 4070 TDP'],
  [/rtx\s*4060/i,    115, 'RTX 4060 TDP'],
  [/rtx\s*3090/i,    350, 'RTX 3090 TDP'],
  [/rtx\s*3080/i,    320, 'RTX 3080 TDP'],
  [/rtx\s*3070/i,    220, 'RTX 3070 TDP'],
  [/rtx\s*3060/i,    170, 'RTX 3060 TDP'],
  [/rtx\s*3050/i,     80, 'RTX 3050 TDP'],  // ← our dev machine
];
const DEFAULT_TDP_WATTS = 150;  // labelled "generic estimate"
```

### Step 2: Add a Grid Carbon Intensity Constant

```js
// IEA 2023 world average grid carbon intensity.
// Regional values range from ~20 (Norway, hydro) to ~900 (coal-heavy grids).
// Users in a specific region can override via a future setting; until then
// the world average is labelled as such rather than presented bare.
const GRID_CO2_GRAMS_PER_KWH = 436;
```

**Source:** International Energy Agency, *CO2 Emissions from Fuel Combustion Highlights*,
2023 edition. The 436 gCO₂/kWh figure is the global electricity generation weighted
average. It is an estimate, not a reading, and is labelled that way in every surface.

### Step 3: Add Dashboard State Variables

```js
let gpuTdpWatts = DEFAULT_TDP_WATTS;
let gpuTdpSource = 'generic estimate — set your GPU in the environment event';
```

### Step 4: Resolve TDP from the Environment Event

In the existing `applyGpuRate()` function (or a new parallel `applyGpuTdp()`), match the
GPU name against `GPU_TDP_TABLE`:

```js
function applyGpuTdp(gpuName) {
  if (!gpuName) {
    gpuTdpWatts = DEFAULT_TDP_WATTS;
    gpuTdpSource = 'no GPU detected — generic estimate';
    return;
  }
  for (const [pattern, tdp, label] of GPU_TDP_TABLE) {
    if (pattern.test(gpuName)) {
      gpuTdpWatts = tdp;
      gpuTdpSource = label;
      return;
    }
  }
  gpuTdpWatts = DEFAULT_TDP_WATTS;
  gpuTdpSource = `${gpuName} · not in TDP table, generic estimate`;
}
```

This is called from the existing `environment` event handler, right after `applyGpuRate()`.

### Step 5: The Math

Three derived values, all pure arithmetic, no API call, no network:

```js
function computeEnergy(elapsedMs) {
  const elapsedHours = elapsedMs / 3_600_000;
  const kWh = (gpuTdpWatts * elapsedHours) / 1000;
  const co2Grams = kWh * GRID_CO2_GRAMS_PER_KWH;
  return { kWh, co2Grams };
}
```

**Why TDP and not actual power draw?**

`torch.cuda.power_draw()` is not a standard PyTorch function. `nvidia-smi` can report
instantaneous wattage, but:
- Polling it adds a subprocess spawn per sample — overhead we cannot afford.
- The harness runs inside a VS Code extension, not a root shell.
- `pynvml` would be a new dependency for one number.

TDP is an upper bound. Training load typically draws 70–90% of TDP. We label the figure
as "TDP-based estimate" and never present it as a measurement. This is consistent with
ARC Lens's rule: an unlabelled number is the same class of problem as a fabricated metric.

### Step 6: Human-Readable Comparisons

Static lookup, no computation required:

```js
function humanize(co2Grams) {
  // Average petrol car: ~120g CO₂/km (EU fleet average, 2023)
  const carKm = co2Grams / 120;
  // Average smartphone charge: ~8.5g CO₂ (IEA)
  const phoneCharges = co2Grams / 8.5;
  // LED lightbulb (10W): 4.36g CO₂/hr at world-average grid
  const ledHours = co2Grams / 4.36;

  if (carKm >= 1)   return `≈ driving ${carKm.toFixed(1)} km`;
  if (phoneCharges >= 1) return `≈ ${phoneCharges.toFixed(1)} smartphone charges`;
  return `≈ ${ledHours.toFixed(1)} hours of a 10W LED bulb`;
}
```

The comparisons scale automatically — a short laptop run gets the phone analogy,
a 24-hour A100 run gets the driving analogy.

---

## Where It Appears in the UI

### 1. Dashboard: Energy Metric Card

A new metric card in the existing `.metrics-container` grid, alongside Loss, Gradient L2
Norm, Learning Rate, Step/Epoch, and System Risk:

```html
<div class="metric-card" id="card-energy" style="--metric-accent: #22c55e;">
  <div class="metric-label">Energy</div>
  <div class="metric-value" id="val-energy">0.00 kWh</div>
  <div class="metric-subtext" id="val-co2">0g CO₂ · —</div>
</div>
```

Updated every metric batch (reuses the same `refreshMetrics()` call that updates Loss).

```
┌─ Energy ─────────────────┐
│  0.14 kWh                │
│  61g CO₂ · ≈ driving     │
│  0.4 km                  │
└──────────────────────────┘
```

### 2. Compute-Savings Ledger: Energy Saved

The existing ledger card (`.ledger-card`) already shows "Training Preserved" and
"Compute Not Re-Spent." Add a third stat:

```html
<div>
  <div class="ledger-stat-value" id="ledger-energy">0 Wh</div>
  <div class="ledger-stat-label">Energy Not Wasted</div>
</div>
```

Shows the energy equivalent of the preserved compute. Uses the same
`survivingFraction` math already in `updateSavingsLedger()`:

```js
const preservedKwh = computeEnergy(preservedMs).kWh;
document.getElementById('ledger-energy').textContent =
  preservedKwh >= 0.01
    ? preservedKwh.toFixed(3) + ' kWh'
    : (preservedKwh * 1000).toFixed(1) + ' Wh';
```

### 3. Unrecoverable Banner: Waste Since Last Healthy

When ARC declares a run unrecoverable, the banner already appears. Add the energy cost
of the wasted tail:

```
RUN JUDGED UNRECOVERABLE
3 recovery attempts failed.

⚡ 0.04 kWh wasted since last healthy checkpoint
🌿 17g CO₂ — stop now to avoid further waste

[ Stop Run ]  [ Export Report ]
```

### 4. Exported HTML Report: Energy Section

In `reportBuilder.ts`, add a new summary card and a line in the Environment section:

```html
<div class="card">
  <div class="label">Energy</div>
  <div class="value">0.14 kWh</div>
</div>
<div class="card">
  <div class="label">Carbon</div>
  <div class="value">61g CO₂eq</div>
</div>
```

And in the `<dl>` environment block:

```html
<dt>Energy (TDP-based)</dt><dd>0.14 kWh</dd>
<dt>Carbon (IEA world avg)</dt><dd>61g CO₂eq</dd>
```

The source is always stated. The report inherits the same "never an unlabelled number"
rule as the dashboard.

---

## Integration Points — Exact File Locations

### `python/_arc_bootstrap.py`

**No changes required.** The `environment` event at line 1309 already emits `"gpu"` and
`finish()` at line 1341 already emits `"wall_seconds"`. All energy math is derived on
the frontend.

### `media/dashboard.html`

| What | Where | Change |
|:---|:---|:---|
| `GPU_TDP_TABLE` constant | After line 501 (after `GPU_RATE_TABLE`) | Add the TDP table |
| `gpuTdpWatts` / `gpuTdpSource` state | After line 506 (`runStartTime`) | Add two variables |
| `applyGpuTdp()` function | After `applyGpuRate()` (line 528) | Add the TDP resolver |
| `computeEnergy()` function | After `applyGpuTdp()` | Add the energy math |
| Energy metric card HTML | After the Risk card (line 351) | Add the card |
| Energy update in metric handler | In the batch metric processing | Add one call |
| Ledger energy stat | Inside `.ledger-stats` (line 358) | Add the third stat |
| Unrecoverable banner energy line | In unrecoverable handler | Add the text |

### `src/pro/reportBuilder.ts`

| What | Where | Change |
|:---|:---|:---|
| Import energy math | Top of `buildReportHtml` | Compute from `summary.wall_seconds` and `env.gpu` |
| Energy + Carbon summary cards | After the "Wall clock" card (line 259) | Add two cards |
| Energy line in Environment `<dl>` | After "Instrumented time" (line 276) | Add two `<dt>/<dd>` pairs |

### `tests/dashboard.test.js`

| What | Change |
|:---|:---|
| GPU TDP table test | Assert every entry has a positive TDP and a non-empty label |
| Energy computation test | Assert `computeEnergy(3600000)` with a known TDP returns the expected kWh |
| Humanize test | Assert correct comparison text for various CO₂ values |

---

## The Constants — Sources and Justification

### GPU TDP Values

| GPU | TDP (W) | Source |
|:---|---:|:---|
| H100 SXM | 700 | [NVIDIA H100 datasheet](https://www.nvidia.com/en-us/data-center/h100/) |
| H100 PCIe | 350 | Same datasheet, PCIe variant |
| A100 80 GB | 400 | [NVIDIA A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/) |
| A100 40 GB | 300 | Same datasheet |
| V100 | 300 | [NVIDIA V100 datasheet](https://www.nvidia.com/en-us/data-center/v100/) |
| T4 | 70 | [NVIDIA T4 datasheet](https://www.nvidia.com/en-us/data-center/tesla-t4/) |
| RTX 3050 | 80 | NVIDIA spec (our dev machine — verified) |
| Default | 150 | Conservative middle estimate, labelled as such |

### Grid Carbon Intensity

| Value | Meaning | Source |
|:---|:---|:---|
| 436 gCO₂/kWh | Global electricity weighted average | IEA, 2023 |
| ~20 gCO₂/kWh | Norway (near-100% hydro) | Low end |
| ~900 gCO₂/kWh | Coal-heavy grids (parts of India, Poland) | High end |

The world average is a reasonable default. A future `arcAgent.gridCarbonIntensity` setting
would let users in specific regions set their own value.

### Human Comparisons

| Comparand | CO₂/unit | Source |
|:---|:---|:---|
| Petrol car | 120 g/km | EU fleet average, 2023 |
| Smartphone charge | 8.5 g | IEA estimate for average battery + grid |
| 10W LED bulb | 4.36 g/hr | 10W × 436 gCO₂/kWh |

---

## Example Scenarios — What the Numbers Actually Look Like

### Scenario 1: Our reference run (RTX 3050, CIFAR-10, 10 epochs)

```
GPU:          RTX 3050 (TDP: 80W)
Wall clock:   6m 30s (390s)
kWh:          80W × 0.108 hr / 1000 = 0.0087 kWh
CO₂:          0.0087 × 436 = 3.8g
Comparison:   ≈ 0.4 smartphone charges
```

This is small. **Report it anyway.** ARC's credibility comes from reporting honest numbers,
not impressive ones. The point is the run that *didn't* need to be restarted.

### Scenario 2: A 6-hour A100 run that diverges at hour 4

```
GPU:          A100 80GB (TDP: 400W)
Wall clock:   6 hours
kWh:          400W × 6hr / 1000 = 2.4 kWh
CO₂:          2.4 × 436 = 1046g ≈ 1.05 kg
Comparison:   ≈ driving 8.7 km

ARC detects NaN at hour 4, rolls back to hour 3:58.
Energy saved: 400W × 2hr / 1000 = 0.8 kWh = 349g CO₂ = driving 2.9 km
Without ARC:  restart from hour 0 = another 2.4 kWh = 1.05 kg CO₂ more
```

### Scenario 3: Unrecoverable run — waste since last healthy checkpoint

```
GPU:          RTX 3050 (TDP: 80W)
Verdict:      unrecoverable at step 350, last healthy at step 200
Wasted time:  150 steps × ~49ms/step = 7.4s
Wasted kWh:   80W × 0.002hr / 1000 = 0.00016 kWh
Wasted CO₂:   0.07g
```

This number is tiny. **Show it anyway**, because the real waste is the user's time
staring at a dead run — and the next restart's energy cost.

---

## What NOT to Claim

1. **Do not call this a "carbon measurement."** It is an estimate based on published TDP and
   a global-average grid factor. Actual power draw, actual grid intensity, and cooling overhead
   are not measured.

2. **Do not hide the source.** Every energy figure must show where it came from:
   "RTX 3050 TDP · IEA world avg grid." An unlabelled kWh is the same class of problem as
   the fabricated metrics that were deleted in Tier 0.

3. **Do not use this to make ethical claims about the product.** ARC Lens saves compute
   when a run diverges. It does not make ML training "green." The honest framing is
   *"waste made visible,"* not *"sustainability achieved."*

4. **Do not inflate the numbers.** TDP is an upper bound. Typical training load is 70–90%
   of TDP. Using TDP means our estimates are slightly high, which is acceptable as long as
   it is stated. Correcting downward with a 0.8× factor is also honest if labelled
   as "estimated 80% load factor."

---

## Testing Plan

### Unit Tests (`tests/dashboard.test.js`)

```js
// TDP table: every entry is valid
test('GPU_TDP_TABLE entries have positive TDP and non-empty label', () => {
  for (const [pattern, tdp, label] of GPU_TDP_TABLE) {
    assert(tdp > 0, `TDP must be positive: ${label}`);
    assert(label.length > 0, 'Label must not be empty');
    assert(pattern instanceof RegExp, 'Pattern must be a RegExp');
  }
});

// Energy computation is correct
test('computeEnergy returns correct kWh for 1 hour at 300W', () => {
  gpuTdpWatts = 300;
  const { kWh, co2Grams } = computeEnergy(3_600_000); // 1 hour in ms
  assert.strictEqual(kWh, 0.3);
  assert.strictEqual(co2Grams, 0.3 * 436);
});

// Zero elapsed = zero energy
test('computeEnergy returns 0 for 0 elapsed time', () => {
  const { kWh, co2Grams } = computeEnergy(0);
  assert.strictEqual(kWh, 0);
  assert.strictEqual(co2Grams, 0);
});

// Humanize picks the right comparison
test('humanize selects car analogy for large values', () => {
  assert(humanize(500).includes('driving'));
  assert(humanize(500).includes('km'));
});

test('humanize selects phone analogy for small values', () => {
  assert(humanize(20).includes('smartphone'));
});
```

### Integration: Report Builder

- Assert the exported HTML contains the Energy and Carbon cards when `wall_seconds` and
  `environment.gpu` are present.
- Assert the source labels are never empty.

### Manual Validation

Run `python/train_demo.py` on the RTX 3050. Verify:
- The Energy card updates live during the run.
- The ledger shows "Energy Not Wasted" after an intervention.
- The exported report contains the Energy section.
- The numbers are consistent with `80W × elapsed hours / 1000`.

---

## Summary: What Changes, What Stays

| Component | Change | Lines of code |
|:---|:---|---:|
| `_arc_bootstrap.py` | **None** | 0 |
| `dashboard.html` — constants | Add `GPU_TDP_TABLE`, `GRID_CO2`, `computeEnergy`, `humanize` | ~50 |
| `dashboard.html` — UI | Add energy card, ledger stat, unrecoverable line | ~20 |
| `reportBuilder.ts` | Add 2 summary cards + 2 `<dl>` lines | ~15 |
| `dashboard.test.js` | TDP table, energy math, humanize tests | ~30 |
| Total | | **~115** |

Zero new dependencies. Zero new API calls. Zero backend changes. ~115 lines of frontend code
to add a feature that no other ML monitoring tool has.
