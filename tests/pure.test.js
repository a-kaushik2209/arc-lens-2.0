/**
 * Tests for the pure input->output functions in src/pro.
 *
 * Run with `npm test` (node:test, no framework). These compile from TypeScript
 * first via `npm run compile`, so they exercise the same JavaScript that ships.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const OUT = path.join(__dirname, "..", "out", "pro");
const { extractCodeBlock, buildScriptGenMessages } = require(path.join(OUT, "scriptGenerator.js"));
const { buildSystemPrompt } = require(path.join(OUT, "contextBuilder.js"));
const { buildReportHtml } = require(path.join(OUT, "reportBuilder.js"));

// ── extractCodeBlock ────────────────────────────────────────────────────────

test("extractCodeBlock pulls a language-tagged python fence", () => {
  const res = extractCodeBlock("chatter\n```python\nimport torch\n```\ntrailing", "py");
  assert.equal(res, "import torch");
});

test("extractCodeBlock falls back to any fence when the tag is missing", () => {
  const res = extractCodeBlock("```\nimport torch\n```", "py");
  assert.equal(res, "import torch");
});

test("extractCodeBlock returns null when there is no fence at all", () => {
  assert.equal(extractCodeBlock("no code here", "py"), null);
});

test("extractCodeBlock selects the json fence for notebook output", () => {
  const res = extractCodeBlock('```json\n{"cells": []}\n```', "ipynb");
  assert.equal(res, '{"cells": []}');
});

// ── buildScriptGenMessages ──────────────────────────────────────────────────

test("buildScriptGenMessages produces a system+user pair mentioning the arch", () => {
  const msgs = buildScriptGenMessages({
    architecture: "resnet",
    task: "CIFAR-10",
    platform: "kaggle",
    outputFormat: "py",
    epochs: 20,
    optimizer: "AdamW",
    extraNotes: "",
  });
  assert.equal(msgs.length, 2);
  assert.equal(msgs[0].role, "system");
  assert.equal(msgs[1].role, "user");
  assert.match(msgs[1].content, /ResNet-50/);
  assert.match(msgs[1].content, /CIFAR-10/);
});

// ── buildSystemPrompt ───────────────────────────────────────────────────────

const metric = (step, loss, grad = 1, lr = 0.1) => ({
  step, epoch: 0, loss, grad_norm: grad, lr,
});

test("buildSystemPrompt reports min/max/final loss", () => {
  const prompt = buildSystemPrompt(
    [metric(1, 5), metric(2, 2), metric(3, 9)],
    [],
    "train.py"
  );
  assert.match(prompt, /min=2\.000e\+0/);
  assert.match(prompt, /max=9\.000e\+0/);
  assert.match(prompt, /final=9\.000e\+0/);
});

test("buildSystemPrompt renders a NaN loss as NaN, never as a number", () => {
  const prompt = buildSystemPrompt([metric(1, 1), metric(2, null)], [], "train.py");
  assert.match(prompt, /\tNaN\t/);
});

test("buildSystemPrompt survives an empty run without throwing", () => {
  const prompt = buildSystemPrompt([], [], "train.py");
  assert.match(prompt, /No data yet/);
  assert.match(prompt, /N\/A/);
});

test("buildSystemPrompt caps the trace at 40 rows for a long run", () => {
  const metrics = Array.from({ length: 5000 }, (_, i) => metric(i + 1, 1 / (i + 1)));
  const prompt = buildSystemPrompt(metrics, [], "train.py");
  const traceSection = prompt.split("## Sampled Metric Trace")[1].split("## ARC Agent Event Log")[0];
  // Data rows only — the first line of that section is the column header.
  const rows = traceSection.trim().split("\n").filter((l) => l.includes("\t"));
  assert.ok(rows.length <= 40, `expected <=40 sampled rows, got ${rows.length}`);
  assert.ok(rows.length >= 30, `expected a meaningful sample, got ${rows.length}`);
});

test("buildSystemPrompt does not stack-overflow on a full-capacity history", () => {
  // Guards the reduce() that replaced Math.max(...array): 10 000 entries is
  // close enough to the engine's argument limit that the spread form was one
  // config change away from throwing here.
  const metrics = Array.from({ length: 10000 }, (_, i) => metric(i + 1, i + 1));
  const prompt = buildSystemPrompt(metrics, [], "train.py");
  assert.match(prompt, /max=1\.000e\+4/);
});

test("buildSystemPrompt includes the intervention log", () => {
  const prompt = buildSystemPrompt(
    [metric(1, 1)],
    [{ type: "intervention", step: 12, message: "", action: "reduce_lr", detail: "0.1 -> 0.05" }],
    "train.py"
  );
  assert.match(prompt, /INTERVENTION/);
  assert.match(prompt, /reduce_lr/);
});

// ── buildReportHtml ─────────────────────────────────────────────────────────

const baseRun = () => ({
  file: "train.py",
  startedAt: "2026-01-01T00:00:00Z",
  environment: { gpu: "RTX 3050", torch: "2.6.0", arc: "5.0.0" },
  events: [],
  summary: { steps: 3, wall_seconds: 12 },
  mode: "active",
});

test("buildReportHtml emits a self-contained document with no external refs", () => {
  const html = buildReportHtml(baseRun(), [metric(1, 1), metric(2, 0.5), metric(3, 0.25)]);
  assert.match(html, /^<!doctype html>/);
  assert.ok(!/https?:\/\//.test(html), "report must not reference any external resource");
  assert.ok(!/<script/i.test(html), "report must not contain script tags");
});

test("buildReportHtml escapes hostile text rather than rendering it", () => {
  const run = baseRun();
  run.events.push({
    type: "intervention",
    step: 2,
    action: "<img src=x onerror=alert(1)>",
    detail: "</td><script>alert(2)</script>",
  });
  const html = buildReportHtml(run, [metric(1, 1)]);
  assert.ok(!html.includes("<script>alert(2)"), "script payload must be escaped");
  assert.ok(!html.includes("<img src=x"), "img payload must be escaped");
  assert.match(html, /&lt;img src=x/);
});

test("buildReportHtml reports an unrecoverable verdict when one was reached", () => {
  const run = baseRun();
  run.events.push({ type: "unrecoverable", step: 200, kind: "gradient_entropy_collapse", attempts: 3 });
  const html = buildReportHtml(run, [metric(1, 1)]);
  assert.match(html, /unrecoverable/i);
  assert.match(html, /class="verdict bad"/);
});

test("buildReportHtml handles a run where every loss was NaN", () => {
  const html = buildReportHtml(baseRun(), [metric(1, null), metric(2, null)]);
  assert.match(html, /No finite loss values/);
});

test("buildReportHtml draws the baseline arm when an A/B run exists", () => {
  const run = baseRun();
  run.baselineMetrics = {
    label: "baseline (train.py)",
    points: [{ step: 1, loss: 2 }, { step: 2, loss: 4 }, { step: 3, loss: 9 }],
  };
  const html = buildReportHtml(run, [metric(1, 2), metric(2, 1), metric(3, 0.5)]);
  assert.match(html, /baseline \(train\.py\)/);
  // Two <path> elements: the active arm and the control arm.
  assert.equal((html.match(/<path d="M/g) || []).length, 2);
});

test("buildReportHtml labels the unrecoverable waste estimate, never a bare number", () => {
  const run = baseRun();
  run.events.push({
    type: "unrecoverable",
    step: 200,
    kind: "gradient_entropy_collapse",
    attempts: 3,
    elapsed_seconds: 125,
    last_checkpoint_step: 150,
  });
  const html = buildReportHtml(run, [metric(1, 1)]);
  assert.match(html, /2m 5s burned/);
  assert.match(html, /since last healthy checkpoint \(step 150\)/);
  assert.match(html, /class="assumption"/);
  assert.match(html, /~\$[\d.]+ at \$[\d.]+\/hr \(RTX 3050/);
  assert.match(html, /~[\d.]+ kWh assuming \d+W typical draw/);
});

test("buildReportHtml surfaces a checkpoint_budget event with its pruned count", () => {
  const run = baseRun();
  run.events.push({ type: "checkpoint_budget", step: 50, budget_mb: 512, pruned_count: 4 });
  const html = buildReportHtml(run, [metric(1, 1)]);
  assert.match(html, /Checkpoint budget/);
  assert.match(html, /512 MB/);
  assert.match(html, /4 pruned beyond the ring buffer/);
});

test("buildReportHtml breaks the line at a missing point instead of bridging it", () => {
  // A gap must read as absent data, not as a straight line through it — the
  // same principle that removed the dashboard's fabricated metrics.
  const html = buildReportHtml(baseRun(), [metric(1, 1), metric(2, null), metric(3, 1)]);
  const path = html.match(/<path d="([^"]+)"/)[1];
  assert.equal((path.match(/M/g) || []).length, 2, "expected the path to restart after the gap");
});
