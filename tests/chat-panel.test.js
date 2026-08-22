/**
 * AI Failure Analyst chat webview checks.
 *
 * getChatHtml() is a template literal buried in src/extension.ts, which imports
 * `vscode` and so cannot be required directly. Following tests/dashboard.test.js's
 * approach: pull the compiled function's source out of out/extension.js with
 * brace matching, eval it in a vm context, extract the inline <script> the
 * function returns, and run that script against a minimal hand-rolled DOM stub.
 * No new dependencies (no jsdom).
 *
 * These lock down bugs that were just fixed:
 *  - send used to latch `streaming` only on the stream_start round-trip, so a
 *    double-Enter before that reply landed fired two concurrent requests.
 *  - stream_error was only rendered `if(streamEl)`, so an error with no live
 *    stream silently vanished.
 *  - stream_error concatenated the provider's untrusted error text into
 *    innerHTML.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const OUT_PATH = path.join(__dirname, "..", "out", "extension.js");

let compiledSrc = null;
try {
  compiledSrc = fs.readFileSync(OUT_PATH, "utf8");
} catch {
  // handled below
}

if (!compiledSrc) {
  test("out/extension.js exists", () => {
    assert.fail(
      `${OUT_PATH} not found. Run \`npm run compile\` (tsc -p ./) to build the ` +
        `extension before running this test file.`
    );
  });
} else {
  runSuite(compiledSrc);
}

/** Depth-first collection of every node's textContent, for a circularity-safe search. */
function flattenText(el, out = []) {
  if (el._text) out.push(el._text);
  for (const c of el.children || []) flattenText(c, out);
  return out;
}

function runSuite(src) {
  /** Brace-match a top-level function's full source out of the compiled file. */
  function extractFn(name) {
    const start = src.indexOf("function " + name);
    assert.ok(start >= 0, `could not find function ${name} in out/extension.js`);
    let i = src.indexOf("{", start);
    let depth = 0;
    for (let j = i; j < src.length; j++) {
      if (src[j] === "{") depth++;
      else if (src[j] === "}") {
        depth--;
        if (depth === 0) return src.slice(start, j + 1);
      }
    }
    throw new Error("unbalanced braces extracting " + name);
  }

  const makeNonce = extractFn("makeNonce");
  const getChatHtmlSrc = extractFn("getChatHtml");
  // makeNonce() uses require("crypto"); pass require through.
  const { getChatHtml } = new Function(
    "require",
    `${makeNonce}\n${getChatHtmlSrc}\nreturn {getChatHtml};`
  )(require);

  const html = getChatHtml("Gemini 2.5 Flash", []);

  function inlineScript() {
    const match = html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/);
    assert.ok(match, "expected a nonce-tagged inline script block in the chat panel");
    return match[1];
  }

  // ── Structure and policy ────────────────────────────────────────────────

  test("chat panel inline script parses as valid JavaScript", () => {
    assert.doesNotThrow(() => new vm.Script(inlineScript(), { filename: "chat-inline.js" }));
  });

  test("CSP locks down default-src and scopes script-src to the nonce", () => {
    const csp = html.match(/Content-Security-Policy" content="([^"]+)"/);
    assert.ok(csp, "expected a CSP meta tag");
    const directives = csp[1].split(";").map((d) => d.trim());
    assert.ok(directives.includes("default-src 'none'"), "default-src must be 'none'");
    const scriptSrc = directives.find((d) => d.startsWith("script-src"));
    assert.ok(scriptSrc, "expected a script-src directive");
    assert.match(scriptSrc, /'nonce-[^']+'/, "script-src must be nonce-scoped");
    assert.ok(!scriptSrc.includes("'unsafe-inline'"), "script-src must not allow unsafe-inline");
  });

  test("the inline script tag carries the same nonce as the CSP", () => {
    const cspNonce = html.match(/script-src 'nonce-([^']+)'/)[1];
    const tagNonce = html.match(/<script nonce="([^"]+)">/)[1];
    assert.equal(tagNonce, cspNonce);
  });

  // ── DOM stub ─────────────────────────────────────────────────────────────
  // Small enough to cover only what the chat script actually touches:
  // getElementById, createElement/appendChild/remove, per-element
  // addEventListener, dataset/style/textContent/innerHTML, and a
  // window that can dispatch 'message' events the way the extension host does.

  function makeElement(tag) {
    const listeners = {};
    return {
      tagName: tag,
      id: "",
      className: "",
      style: {},
      dataset: {},
      disabled: false,
      value: "",
      _text: "",
      _html: "",
      children: [],
      parentNode: null,
      get textContent() {
        return this._text;
      },
      set textContent(v) {
        this._text = v;
        this._html = "";
      },
      get innerHTML() {
        return this._html;
      },
      set innerHTML(v) {
        this._html = v;
      },
      appendChild(child) {
        child.parentNode = this;
        this.children.push(child);
        return child;
      },
      remove() {
        if (this.parentNode) {
          this.parentNode.children = this.parentNode.children.filter((c) => c !== this);
          this.parentNode = null;
        }
      },
      addEventListener(type, fn) {
        (listeners[type] = listeners[type] || []).push(fn);
      },
      dispatch(type, evt) {
        for (const fn of listeners[type] || []) fn(evt);
      },
    };
  }

  /** Fresh DOM + vscode api + a running copy of the chat script for each test. */
  function setupPanel() {
    const byId = {};
    function register(id, tag) {
      const el = makeElement(tag);
      el.id = id;
      byId[id] = el;
      return el;
    }
    const messages = register("messages", "div");
    const emptyState = register("empty-state", "div");
    messages.appendChild(emptyState);
    register("btn-clear", "button");
    const btnCancel = register("btn-cancel", "button");
    btnCancel.style.display = "none";
    const btnSend = register("btn-send", "button");
    register("user-input", "textarea");

    const createdTags = [];
    const document_ = {
      getElementById: (id) => byId[id] || null,
      querySelector: () => null,
      querySelectorAll: () => [],
      createElement: (tag) => {
        createdTags.push(tag);
        return makeElement(tag);
      },
    };

    const messageListeners = [];
    const window_ = {
      addEventListener: (type, fn) => {
        if (type === "message") messageListeners.push(fn);
      },
      dispatchEvent: () => {},
    };

    const posted = [];
    function acquireVsCodeApi() {
      return {
        postMessage: (m) => posted.push(m),
        getState: () => undefined,
        setState: () => {},
      };
    }

    const sandbox = {
      document: document_,
      window: window_,
      acquireVsCodeApi,
      JSON,
      Math,
      console,
    };
    vm.createContext(sandbox);
    new vm.Script(inlineScript(), { filename: "chat-inline.js" }).runInContext(sandbox);

    return {
      byId,
      posted,
      createdTags,
      sendMessage: (text) => {
        byId["user-input"].value = text;
        byId["btn-send"].dispatch("click");
      },
      clickCancel: () => byId["btn-cancel"].dispatch("click"),
      deliver: (data) => {
        for (const fn of messageListeners) fn({ data });
      },
    };
  }

  // ── Behaviour 1: send latches `streaming` synchronously ──────────────────

  test("send disables the send button and reveals cancel on the same tick, before any reply", () => {
    const p = setupPanel();
    p.sendMessage("why did the loss spike?");
    assert.equal(p.byId["btn-send"].disabled, true, "send must be disabled synchronously");
    assert.equal(
      p.byId["btn-cancel"].style.display,
      "inline-flex",
      "cancel must be revealed synchronously"
    );
    assert.equal(p.posted.length, 1);
    assert.equal(p.posted[0].command, "chat");
  });

  test("two sends with no stream_start in between post exactly one chat message", () => {
    const p = setupPanel();
    p.sendMessage("first question");
    p.sendMessage("second question, sent before any reply");
    assert.equal(
      p.posted.filter((m) => m.command === "chat").length,
      1,
      "a second send while still streaming must not fire a second request"
    );
  });

  // ── Behaviour 2 & 3: stream_error with no live stream, and inert text ────

  test("stream_error renders even when no stream element is live", () => {
    const p = setupPanel();
    // No stream_start was ever sent, so streamEl is null going into this.
    p.deliver({ type: "stream_error", text: "API error 429: rate limited" });
    const texts = flattenText(p.byId["messages"]);
    assert.ok(
      texts.some((t) => t.includes("API error 429: rate limited")),
      "the error text must reach the DOM even with no live stream element"
    );
  });

  test("stream_error text is rendered as inert text, not markup", () => {
    const p = setupPanel();
    const payload = "<img src=x onerror=alert(1)>";
    p.deliver({ type: "stream_error", text: payload });

    // Find the assistant message appended for this error.
    const assistantMsgs = p.byId["messages"].children.filter((c) => c.className === "msg msg-assistant");
    assert.equal(assistantMsgs.length, 1);
    const errorParagraph = assistantMsgs[0].children[0];
    assert.equal(errorParagraph.textContent, "Error: " + payload, "text must be literal");
    assert.ok(!p.createdTags.includes("img"), "no <img> element must ever be created from error text");
  });

  // ── Behaviour 4: chunk rendering escapes HTML, keeps markdown ────────────

  test("stream_chunk escapes raw HTML but still renders markdown", () => {
    const p = setupPanel();
    p.deliver({ type: "stream_start" });
    const streamEl = p.byId["messages"].children[p.byId["messages"].children.length - 1];

    p.deliver({ type: "stream_chunk", text: "<script>alert(1)</script> **bold** `code`" });

    assert.ok(
      streamEl.innerHTML.includes("&lt;script&gt;alert(1)&lt;/script&gt;"),
      "raw <script> must be escaped, not injected as a tag"
    );
    assert.ok(!/<script>alert/.test(streamEl.innerHTML), "no live <script> tag must appear");
    assert.ok(streamEl.innerHTML.includes("<strong>bold</strong>"), "**bold** must render as <strong>");
    assert.ok(streamEl.innerHTML.includes("<code>code</code>"), "`code` must render as <code>");
  });

  // ── Behaviour 5: lifecycle restores send/cancel state ────────────────────

  test("stream_done re-enables send and hides cancel", () => {
    const p = setupPanel();
    p.deliver({ type: "stream_start" });
    p.deliver({ type: "stream_done" });
    assert.equal(p.byId["btn-send"].disabled, false);
    assert.equal(p.byId["btn-cancel"].style.display, "none");
  });

  test("stream_error also re-enables send and hides cancel", () => {
    const p = setupPanel();
    p.deliver({ type: "stream_start" });
    p.deliver({ type: "stream_error", text: "boom" });
    assert.equal(p.byId["btn-send"].disabled, false);
    assert.equal(p.byId["btn-cancel"].style.display, "none");
  });

  test("clicking cancel posts a cancel command", () => {
    const p = setupPanel();
    p.clickCancel();
    assert.equal(p.posted.length, 1);
    assert.equal(p.posted[0].command, "cancel");
    assert.deepEqual(Object.keys(p.posted[0]), ["command"]);
  });
}
