/**
 * The model id in `arcAgent.llmModel` defaults to OpenRouter's format
 * ("google/gemini-2.5-flash:free"). Only OpenRouter understands that shape, so
 * every native provider has to be handed a bare model name instead.
 *
 * Three providers happened to get this right by accident — the OpenRouter
 * default contains none of the substrings their guards look for, so each fell
 * through to its own fallback. Gemini's guard looks for "gemini", which the
 * OpenRouter string *does* contain, so the check passed and the unusable id
 * went to Google verbatim. That was the default path for anyone pasting an
 * AIzaSy… key.
 *
 * The property worth locking down is therefore not "gemini is fixed" but
 * "no provider is ever sent OpenRouter formatting", which is what would have
 * caught the original bug.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const { providerFor, modelForProvider, isSupportedKey, SHIPPED_DEFAULT_MODEL } = require(path.join(__dirname, "..", "out", "pro", "providerRouting.js"));

/** The shipped default for `arcAgent.llmModel` (see package.json). */
const SHIPPED_DEFAULT = "google/gemini-2.5-flash:free";

const KEYS = {
  groq: "gsk_abc123",
  anthropic: "sk-ant-abc123",
  gemini: "AIzaSyAbc123",
  openai: "sk-abc123",
  openrouter: "sk-or-abc123",
};

test("routes each key prefix to its provider", () => {
  for (const [expected, key] of Object.entries(KEYS)) {
    assert.equal(providerFor(key), expected, `${key} should route to ${expected}`);
  }
});

test("every accepted key prefix is one the router recognises", () => {
  // The drift that caused the bug: the allowlist and the routing were in
  // different files, one of them untestable, and they disagreed. A prefix
  // accepted here but unknown to providerFor is silently sent to OpenRouter.
  const samples = {
    "sk-or-v1-abc": "openrouter",
    "gsk_abc": "groq",
    "sk-ant-abc": "anthropic",
    "AIzaSyAbc": "gemini",
    "AQ.Ab8Abc": "gemini",
    "sk-proj-abc": "openai",
  };
  for (const [key, provider] of Object.entries(samples)) {
    assert.equal(isSupportedKey(key), true, `${key} should be accepted`);
    assert.equal(providerFor(key), provider, `${key} should route to ${provider}`);
  }
});

test("a key from a provider we cannot route is refused", () => {
  assert.equal(isSupportedKey(""), false);
  assert.equal(isSupportedKey("xai-abc123"), false);
  assert.equal(isSupportedKey("hf_abc123"), false);
});

test("both of Google's key formats route to Gemini", () => {
  // `AQ.…` is the newer one. It was missing from the prefix allowlist and from
  // the routing here, so a key that works against
  // generativelanguage.googleapis.com was refused in settings — and would have
  // been routed to OpenRouter had it got that far.
  assert.equal(providerFor("AIzaSyAbc123"), "gemini");
  assert.equal(providerFor("AQ.Ab8Abc123"), "gemini");
});

test("an Anthropic key is not mistaken for OpenAI", () => {
  // Both start with "sk-"; anthropic has to win.
  assert.equal(providerFor("sk-ant-abc123"), "anthropic");
  assert.equal(providerFor("sk-or-abc123"), "openrouter");
});

test("no native provider is ever sent OpenRouter formatting", () => {
  // The regression that motivated this file.
  //
  // "no slash" is not the property — Groq's own model ids carry a vendor
  // prefix ("openai/gpt-oss-120b"), so asserting that would force a wrong
  // model on the one provider whose ids legitimately contain one. What must
  // never happen is the OpenRouter string reaching a native API, and the
  // `:tier` suffix is OpenRouter's alone.
  for (const provider of ["groq", "anthropic", "gemini", "openai"]) {
    const model = modelForProvider(provider, SHIPPED_DEFAULT);
    assert.notEqual(model, SHIPPED_DEFAULT, `${provider} got the OpenRouter default verbatim`);
    assert.ok(!model.includes(":"), `${provider} got a tier suffix: ${model}`);
  }
});

test("Groq keeps the vendor prefix its own model ids require", () => {
  // Stripping it turned "openai/gpt-oss-120b" into "gpt-oss-120b", which Groq
  // does not serve.
  assert.equal(modelForProvider("groq", "openai/gpt-oss-120b"), "openai/gpt-oss-120b");
  assert.equal(modelForProvider("groq", "meta-llama/llama-prompt-guard-2-86m"), "meta-llama/llama-prompt-guard-2-86m");
});

test("the Groq fallback is not a model Groq has retired", () => {
  // "llama-3.3-70b-versatile" was the fallback until Groq withdrew it and
  // began answering "does not exist or you do not have access to it" — which
  // reads as a rejected key rather than a stale default.
  assert.notEqual(modelForProvider("groq", SHIPPED_DEFAULT), "llama-3.3-70b-versatile");
});

test("the constant here matches the default package.json actually ships", () => {
  // If these drift, the "no preference" rule below silently stops firing.
  const manifest = require(path.join(__dirname, "..", "package.json"));
  assert.equal(
    SHIPPED_DEFAULT_MODEL,
    manifest.contributes.configuration.properties["arcAgent.llmModel"].default
  );
});

test("an untouched default is not treated as a choice of Gemini model", () => {
  // Stripping it to "gemini-2.5-flash" sent Google a pinned version nobody
  // selected, which it then began refusing with a 404 for new projects.
  assert.equal(modelForProvider("gemini", SHIPPED_DEFAULT), "gemini-flash-latest");
});

test("an explicitly configured model still wins over the fallback", () => {
  assert.equal(modelForProvider("gemini", "gemini-3.6-flash"), "gemini-3.6-flash");
  assert.equal(modelForProvider("gemini", "google/gemini-3.6-flash:free"), "gemini-3.6-flash");
});

test("OpenRouter keeps the configured id untouched", () => {
  assert.equal(modelForProvider("openrouter", SHIPPED_DEFAULT), SHIPPED_DEFAULT);
});

test("a model already valid for its provider is preserved", () => {
  assert.equal(modelForProvider("anthropic", "claude-opus-5"), "claude-opus-5");
  assert.equal(modelForProvider("gemini", "gemini-1.5-pro"), "gemini-1.5-pro");
  assert.equal(modelForProvider("openai", "gpt-4o"), "gpt-4o");
  assert.equal(modelForProvider("groq", "llama-3.3-70b-versatile"), "llama-3.3-70b-versatile");
});

test("a model meant for another provider falls back rather than being sent on", () => {
  // Pointing an Anthropic key at a Gemini model must not send "gemini-…" to
  // api.anthropic.com — the fallback is wrong-but-valid, which is recoverable;
  // a foreign id is a hard API error.
  assert.equal(modelForProvider("anthropic", "google/gemini-2.5-flash:free"), "claude-opus-5");
  assert.equal(modelForProvider("gemini", "anthropic/claude-opus-5"), "gemini-flash-latest");
  assert.equal(modelForProvider("openai", "meta-llama/llama-3.3-70b"), "gpt-4o-mini");
});
