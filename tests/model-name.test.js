/**
 * friendlyModelName replaced a hardcoded lookup table that had already gone
 * stale, so the property worth testing is that it produces a sensible name for
 * models nobody has heard of yet — that is the whole reason the table went.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const { friendlyModelName } = require(path.join(__dirname, "..", "out", "pro", "modelName.js"));

test("formats a vendor-prefixed model id", () => {
  assert.equal(friendlyModelName("anthropic/claude-opus-5"), "Claude Opus 5");
  assert.equal(friendlyModelName("openai/gpt-4o-mini"), "Gpt 4o Mini");
});

test("marks free tiers", () => {
  assert.equal(friendlyModelName("google/gemini-2.5-flash:free"), "Gemini 2.5 Flash (Free)");
});

test("handles a bare model id with no vendor prefix", () => {
  assert.equal(friendlyModelName("llama-3.3-70b-versatile"), "Llama 3.3 70b Versatile");
});

test("produces a usable name for a model that does not exist yet", () => {
  assert.equal(friendlyModelName("somevendor/brand-new-model-9:free"), "Brand New Model 9 (Free)");
});

test("falls back rather than returning an empty label", () => {
  for (const input of [undefined, null, "", "/", ":free"]) {
    assert.ok(friendlyModelName(input).length > 0, `empty name for ${JSON.stringify(input)}`);
  }
});
