/**
 * Dashboard checks.
 *
 * The dashboard is a single HTML file with an inline script, so it cannot be
 * imported. These tests extract the script, syntax-check it, and exercise the
 * pure helpers inside it against a minimal DOM stub. That covers the regressions
 * that actually happen when editing this file: a syntax error, a leftover
 * template placeholder, a CSP that silently permits inline script again, or a
 * fabricated-metric fallback creeping back in.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const HTML_PATH = path.join(__dirname, "..", "media", "dashboard.html");
const html = fs.readFileSync(HTML_PATH, "utf8");

/** The same substitution getDashboardHtml() performs before handing HTML to the webview. */
function render(source = html) {
  return source
    .split("{{NONCE}}").join("test-nonce")
    .split("{{CSP_SOURCE}}").join("vscode-resource:")
    .replace("{{ECHARTS_URI}}", "vendor/echarts.min.js")
    .replace("{{LOGO_URI}}", "logo.png")
    .replace("{{IS_PRO_PLACEHOLDER}}", "true");
}

function inlineScript() {
  // Compile what actually reaches the browser, not the template.
  const match = render().match(/<script nonce="test-nonce">([\s\S]*?)<\/script>/);
  assert.ok(match, "expected a nonce-tagged inline script block");
  return match[1];
}

// ── Structure and policy ────────────────────────────────────────────────────

test("inline script parses as valid JavaScript", () => {
  // `new vm.Script` compiles without executing — a syntax error throws here.
  assert.doesNotThrow(() => new vm.Script(inlineScript(), { filename: "dashboard-inline.js" }));
});

test("rendering leaves no placeholder behind", () => {
  // An unsubstituted {{...}} is not cosmetic: one of them sits inside the
  // script block, so a missed replacement is a page that does not run at all.
  const leftovers = render().match(/\{\{[A-Z_]+\}\}/g) || [];
  assert.deepEqual(leftovers, [], "these tokens would ship unsubstituted");
});

test("CSP is nonce-based and does not allow inline script", () => {
  const csp = render().match(/Content-Security-Policy" content="([^"]+)"/)[1];
  const scriptSrc = csp.split(";").find((d) => d.trim().startsWith("script-src"));
  assert.ok(scriptSrc.includes("'nonce-test-nonce'"), "script-src must carry a nonce");
  assert.ok(!scriptSrc.includes("'unsafe-inline'"), "script-src must not re-enable inline script");
});

test("every script tag carries the nonce", () => {
  const tags = render().match(/<script[^>]*>/g) || [];
  for (const tag of tags) {
    assert.ok(tag.includes('nonce="test-nonce"'), `script tag without a nonce will be blocked: ${tag}`);
  }
});

test("no inline event handler attributes remain", () => {
  // These are inline script and would need 'unsafe-inline'; a nonce cannot
  // authorise them, so one of these attributes silently breaks a control.
  const handlers = html.match(/\son(click|input|keydown|change|load)=/g) || [];
  assert.deepEqual(handlers, [], "inline handlers cannot run under a nonce CSP");
});

test("every data-act has a handler and vice versa", () => {
  const used = new Set([...html.matchAll(/data-act="([a-z]+)"/g)].map((m) => m[1]));
  const script = inlineScript();
  const declared = new Set(
    [...script.matchAll(/^\s{2}([a-z]+):\s*(?:\(|el =>|\(\) =>)/gm)].map((m) => m[1])
  );
  for (const action of used) {
    assert.ok(declared.has(action), `data-act="${action}" has no handler in ACTIONS`);
  }
});

test("dashboard loads no external scripts", () => {
  const srcs = [...html.matchAll(/<script[^>]*src="([^"]+)"/g)].map((m) => m[1]);
  for (const src of srcs) {
    assert.ok(!/^https?:/.test(src), `external script ${src} — the dashboard must render offline`);
  }
});

test("no fabricated telemetry fallback exists", () => {
  // This is the regression that mattered most: synthesised numbers rendered
  // identically to measurements. Math.random() has no legitimate use here.
  const script = inlineScript();
  assert.ok(!/Math\.random/.test(script), "dashboard must never synthesise metric values");
  assert.ok(!/enrichEvent/.test(script), "enrichEvent fabricated advanced metrics");
});

// ── Behaviour of the pure helpers ───────────────────────────────────────────

function loadHelpers() {
  const script = inlineScript();
  // Take only the declarations we want to exercise, so the module-level DOM
  // and ECharts calls at the bottom of the file never run.
  const wanted = [
    /const GPU_RATE_TABLE = \[[\s\S]*?\];/,
    /const DEFAULT_GPU_RATE = [^\n]+/,
    /const DEFAULT_GPU_WATTS = [^\n]+/,
    /let gpuRate = [^\n]+/,
    /let gpuRateSource = [^\n]+/,
    /let gpuWatts = [^\n]+/,
    /function applyGpuRate\([\s\S]*?\n\}/,
    /function escapeText\([\s\S]*?\n\}/,
    /const KIND_LABELS = \{[\s\S]*?\};/,
    /function shortKind\([^\n]*\n?[^\n]*\}/,
    /function shortAction\([\s\S]*?\n\}/,
    /const REPORT_ONLY_KINDS = new Set\([^\n]+\);/,
    /function isReportOnly\([^\n]*\}/,
    /const RISK_COLORS = [^\n]+/,
  ];
  let source = "";
  for (const pattern of wanted) {
    const match = script.match(pattern);
    assert.ok(match, `could not extract ${pattern}`);
    source += match[0] + "\n";
  }
  source += "\nmodule.exports = { applyGpuRate, escapeText, shortKind, shortAction, isReportOnly, RISK_COLORS, getRate: () => ({ gpuRate, gpuRateSource, gpuWatts }) };";
  const sandbox = { module: { exports: {} } };
  sandbox.exports = sandbox.module.exports;
  vm.createContext(sandbox);
  new vm.Script(source).runInContext(sandbox);
  return sandbox.module.exports;
}

test("GPU rate is derived from the detected device", () => {
  const h = loadHelpers();
  h.applyGpuRate("NVIDIA A100-SXM4-40GB", 0);
  assert.equal(h.getRate().gpuRate, 2.0);
  h.applyGpuRate("NVIDIA H100 80GB HBM3", 0);
  assert.equal(h.getRate().gpuRate, 3.5);
  h.applyGpuRate("NVIDIA GeForce RTX 3050 6GB Laptop GPU", 0);
  assert.equal(h.getRate().gpuRate, 0.2);
});

test("a configured rate always beats the estimate", () => {
  const h = loadHelpers();
  h.applyGpuRate("NVIDIA A100", 7.25);
  assert.equal(h.getRate().gpuRate, 7.25);
  assert.match(h.getRate().gpuRateSource, /configured/);
});

test("an unknown GPU falls back and says so", () => {
  const h = loadHelpers();
  h.applyGpuRate("Some Future GPU 9000", 0);
  assert.equal(h.getRate().gpuRate, 1.0);
  assert.match(h.getRate().gpuRateSource, /not in rate table/);
});

test("the rate source is always populated, so no bare dollar figure is shown", () => {
  const h = loadHelpers();
  for (const gpu of ["NVIDIA A100", "", null, "Weird Device"]) {
    h.applyGpuRate(gpu, 0);
    assert.ok(h.getRate().gpuRateSource.length > 0);
  }
});

test("escapeText neutralises markup from backend strings", () => {
  const h = loadHelpers();
  assert.equal(h.escapeText("<img src=x onerror=alert(1)>"), "&lt;img src=x onerror=alert(1)&gt;");
  assert.equal(h.escapeText(null), "");
});

test("failure kinds and actions get readable chart labels", () => {
  const h = loadHelpers();
  assert.equal(h.shortKind("gradient_entropy_collapse"), "entropy collapse");
  assert.equal(h.shortKind("numerical"), "NaN");
  // The silent-death kind. Without an entry here the chart marker falls back to
  // the raw snake_case kind, which is the one label a judge is most likely to
  // be looking at — it is the only failure a healthy-looking run produces.
  assert.equal(h.shortKind("loss_plateau"), "stalled");
  assert.equal(h.shortKind("something_new"), "something_new");
  assert.equal(h.shortAction("rollback_and_reduce_lr"), "rollback + LR");
  assert.equal(h.shortAction("reduce_lr"), "LR down");
});

test("the report-only kinds match the harness", () => {
  // Mirrors REPORT_ONLY_KINDS in python/_arc_bootstrap.py. A report-only kind
  // emits failure_detected with no intervention event ever following, so the
  // status strip must not promise "attempting rollback / learning-rate
  // reduction" — the run is about to continue untouched and the promise would
  // sit on screen unfulfilled for the rest of the demo.
  const h = loadHelpers();
  assert.equal(h.isReportOnly("loss_plateau"), true);
  assert.equal(h.isReportOnly("numerical"), false);
  assert.equal(h.isReportOnly("representation_collapse"), false);
  assert.equal(h.isReportOnly(undefined), false);
});

test("every risk label the backend can emit has a colour", () => {
  const h = loadHelpers();
  for (const label of ["LOW", "MEDIUM", "HIGH", "CRITICAL"]) {
    assert.ok(h.RISK_COLORS[label], `no colour for ${label}`);
  }
  // MEDIUM must not be green: the badge used to be coloured from the score,
  // so a score of 0.45 labelled MEDIUM rendered green.
  assert.notEqual(h.RISK_COLORS.MEDIUM, h.RISK_COLORS.LOW);
});
