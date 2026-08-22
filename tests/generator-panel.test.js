/**
 * Script Generator panel checks.
 *
 * getGeneratorHtml() is a module-private template literal inside src/extension.ts
 * (compiled to out/extension.js), and extension.ts imports vscode so it cannot
 * be required directly. As with tests/dashboard.test.js, we extract the inline
 * script, syntax-check it, and exercise it against a minimal hand-rolled DOM
 * stub. We also brace-match extract getGeneratorHtml/makeNonce straight out of
 * out/extension.js so the test always tracks the shipped build.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const EXT_PATH = path.join(__dirname, "..", "out", "extension.js");
if (!fs.existsSync(EXT_PATH)) {
  throw new Error(
    `${EXT_PATH} is missing — run \`npm run compile\` first (out/ is a build artifact, not checked in).`
  );
}
const extSrc = fs.readFileSync(EXT_PATH, "utf8");

/** Brace-match extract a top-level `function <name>` from compiled source. */
function extractFn(src, name) {
  const start = src.indexOf("function " + name);
  assert.ok(start >= 0, `could not find function ${name} in out/extension.js`);
  const open = src.indexOf("{", start);
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  throw new Error(`unbalanced braces extracting ${name}`);
}

const makeNonceSrc = extractFn(extSrc, "makeNonce");
const getGeneratorHtmlSrc = extractFn(extSrc, "getGeneratorHtml");

// Evaluate the two functions in a throwaway realm; makeNonce needs `require("crypto")`.
const { getGeneratorHtml } = new Function(
  "require",
  `${makeNonceSrc}\n${getGeneratorHtmlSrc}\nreturn {getGeneratorHtml};`
)(require);

const html = getGeneratorHtml("Test Model");

function inlineScript() {
  const match = html.match(/<script nonce="[^"]+">([\s\S]*)<\/script>/);
  assert.ok(match, "expected a nonce-tagged inline script block");
  return match[1];
}

// ── Structure and policy ────────────────────────────────────────────────────

test("inline script parses as valid JavaScript", () => {
  assert.doesNotThrow(() => new vm.Script(inlineScript(), { filename: "generator-inline.js" }));
});

test("CSP is intact: default-src none, nonce-scoped script-src, no unsafe-inline", () => {
  const csp = html.match(/Content-Security-Policy" content="([^"]+)"/)[1];
  const defaultSrc = csp.split(";").find((d) => d.trim().startsWith("default-src"));
  const scriptSrc = csp.split(";").find((d) => d.trim().startsWith("script-src"));
  assert.ok(defaultSrc.includes("'none'"), "default-src must be 'none'");
  assert.ok(scriptSrc, "expected a script-src directive");
  assert.match(scriptSrc, /'nonce-[^']+'/, "script-src must carry a nonce");
  assert.ok(!scriptSrc.includes("'unsafe-inline'"), "script-src must not allow unsafe-inline");
});

// ── DOM stub ─────────────────────────────────────────────────────────────

/** Just enough DOM for the generator panel's script to run against. */
function makeElement(id) {
  const el = {
    id,
    value: "",
    textContent: "",
    className: "",
    dataset: {},
    _listeners: {},
    addEventListener(type, fn) {
      (this._listeners[type] = this._listeners[type] || []).push(fn);
    },
    click() {
      (this._listeners.click || []).forEach((fn) => fn.call(this));
    },
  };
  return el;
}

function makeDom() {
  const elements = {};
  for (const id of [
    "arch",
    "task",
    "platform",
    "epochs",
    "optimizer",
    "notes",
    "btn-gen",
    "status-gen",
    "status-err",
  ]) {
    elements[id] = makeElement(id);
  }
  elements["arch"].value = "resnet";
  elements["platform"].value = "kaggle";
  elements["optimizer"].value = "AdamW";
  elements["epochs"].value = "20";

  const btnPy = makeElement("btn-py");
  btnPy.className = "format-btn active";
  btnPy.dataset.fmt = "py";
  const btnIpynb = makeElement("btn-ipynb");
  btnIpynb.className = "format-btn";
  btnIpynb.dataset.fmt = "ipynb";
  elements["btn-py"] = btnPy;
  elements["btn-ipynb"] = btnIpynb;

  const fmtButtons = [btnPy, btnIpynb];

  const posted = [];
  const windowListeners = {};
  const documentStub = {
    getElementById: (id) => {
      assert.ok(elements[id], `no stub element with id ${id}`);
      return elements[id];
    },
    querySelectorAll: (sel) => {
      assert.equal(sel, "[data-fmt]");
      return fmtButtons;
    },
  };
  const windowStub = {
    addEventListener: (type, fn) => {
      (windowListeners[type] = windowListeners[type] || []).push(fn);
    },
    dispatchMessage: (data) => {
      (windowListeners.message || []).forEach((fn) => fn({ data }));
    },
    acquireVsCodeApi: () => ({
      postMessage: (m) => posted.push(m),
      getState: () => undefined,
      setState: () => {},
    }),
  };
  return { elements, posted, documentStub, windowStub };
}

function run() {
  const dom = makeDom();
  const sandbox = {
    document: dom.documentStub,
    window: dom.windowStub,
    acquireVsCodeApi: dom.windowStub.acquireVsCodeApi,
  };
  vm.createContext(sandbox);
  new vm.Script(inlineScript()).runInContext(sandbox);
  return dom;
}

// ── Behaviour ────────────────────────────────────────────────────────────

test("form serializes correctly: epochs is a number and fields reflect the form", () => {
  const dom = run();
  dom.elements.arch.value = "transformer";
  dom.elements.task.value = "sequence classification";
  dom.elements.platform.value = "colab";
  dom.elements.optimizer.value = "SGD with momentum";
  dom.elements.epochs.value = "42";
  dom.elements.notes.value = "gradient clipping";

  dom.elements["btn-gen"].click();

  assert.equal(dom.posted.length, 1);
  const msg = dom.posted[0];
  assert.equal(msg.command, "generate");
  assert.equal(msg.request.architecture, "transformer");
  assert.equal(msg.request.task, "sequence classification");
  assert.equal(msg.request.platform, "colab");
  assert.equal(msg.request.optimizer, "SGD with momentum");
  assert.equal(msg.request.extraNotes, "gradient clipping");
  assert.equal(msg.request.epochs, 42);
  assert.equal(typeof msg.request.epochs, "number");
});

test("an empty task falls back to a default instead of posting an empty string", () => {
  const dom = run();
  dom.elements.task.value = "";
  dom.elements["btn-gen"].click();
  assert.equal(dom.posted[0].request.task, "image classification");
});

test("format toggle is mutually exclusive and the toggled value reaches the request", () => {
  const dom = run();
  dom.elements["btn-ipynb"].click();
  assert.equal(dom.elements["btn-ipynb"].className, "format-btn active");
  assert.equal(dom.elements["btn-py"].className, "format-btn");

  dom.elements["btn-gen"].click();
  assert.equal(dom.posted[0].request.outputFormat, "ipynb");
});

test("the generate button doubles as Cancel while a generation is in flight", () => {
  const dom = run();
  assert.equal(dom.elements["btn-gen"].textContent, "");

  dom.elements["btn-gen"].click();
  assert.equal(dom.elements["btn-gen"].textContent, "Cancel");

  dom.elements["btn-gen"].click();
  assert.equal(dom.posted.length, 2);
  // dom.posted[1] is a plain object literal created inside the vm realm, so it
  // does not share Object.prototype with this file's objects — compare fields,
  // not with deepEqual/deepStrictEqual (which checks prototype identity too).
  assert.equal(dom.posted[1].command, "cancel");
  assert.equal(Object.keys(dom.posted[1]).length, 1);
  assert.equal(dom.elements["btn-gen"].textContent, "Generate Training Script");
});

test("a 'done' message re-enables the generate label", () => {
  const dom = run();
  dom.elements["btn-gen"].click();
  assert.equal(dom.elements["btn-gen"].textContent, "Cancel");
  dom.windowStub.dispatchMessage({ type: "done" });
  assert.equal(dom.elements["btn-gen"].textContent, "Generate Training Script");
});

test("an 'error' message re-enables the generate label", () => {
  const dom = run();
  dom.elements["btn-gen"].click();
  dom.windowStub.dispatchMessage({ type: "error", text: "boom" });
  assert.equal(dom.elements["btn-gen"].textContent, "Generate Training Script");
});

test("error text is inert: shown literally via textContent, no markup created", () => {
  const dom = run();
  dom.windowStub.dispatchMessage({ type: "error", text: "API error 400: <b>bad model</b>" });
  assert.match(dom.elements["status-err"].textContent, /API error 400: <b>bad model<\/b>/);
  // The stub's textContent is a plain string property, never parsed as markup;
  // this assertion only holds because the panel script uses `.textContent =`
  // (never `.innerHTML =`) to write it, which is what keeps a <b> from being
  // instantiated as a real element in the actual DOM.
  assert.doesNotMatch(inlineScript(), /status-err['"]\)\.innerHTML/);
});

// ── Prompt contract (buildScriptGenMessages) ────────────────────────────

const SG_OUT = path.join(__dirname, "..", "out", "pro");
const { buildScriptGenMessages } = require(path.join(SG_OUT, "scriptGenerator.js"));

function baseRequest(overrides = {}) {
  return {
    architecture: "resnet",
    task: "CIFAR-10 classification",
    platform: "kaggle",
    outputFormat: "py",
    epochs: 15,
    optimizer: "AdamW",
    extraNotes: "mixed precision",
    ...overrides,
  };
}

test("system prompt tells the model NOT to import or wrap ARC, plain PyTorch only", () => {
  const [system] = buildScriptGenMessages(baseRequest());
  // The prompt does mention the literal text "import arc" — as a thing to forbid
  // ("No `import arc`"). What must not appear is an instruction telling the
  // model TO do it.
  assert.doesNotMatch(system.content, /\bwrite\b[^.]*\bimport arc\b/i);
  assert.match(system.content, /no arc-specific imports/i);
  assert.match(system.content, /No `import arc`/);
});

test("ipynb output format asks for one notebook object, not a bare cell array", () => {
  const [system] = buildScriptGenMessages(baseRequest({ outputFormat: "ipynb" }));
  assert.match(system.content, /one complete notebook object/i);
  assert.match(system.content, /"cells"/);
  assert.match(system.content, /"nbformat"/);
  assert.match(system.content, /[Nn]ot a bare array of cells/);
});

test("epochs, optimizer, task and extraNotes all appear in the user message", () => {
  const [, user] = buildScriptGenMessages(
    baseRequest({ epochs: 77, optimizer: "SGD with momentum", task: "distinctive-task-marker", extraNotes: "distinctive-notes-marker" })
  );
  assert.match(user.content, /77/);
  assert.match(user.content, /SGD with momentum/);
  assert.match(user.content, /distinctive-task-marker/);
  assert.match(user.content, /distinctive-notes-marker/);
});
