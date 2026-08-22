/**
 * RingBuffer backs the metric history the LLM prompt and the exported report
 * both read, so "oldest first, nothing duplicated, nothing lost" is a
 * correctness property, not an implementation detail.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const { RingBuffer } = require(path.join(__dirname, "..", "out", "pro", "ringBuffer.js"));

test("returns entries in insertion order before wrapping", () => {
  const rb = new RingBuffer(5);
  [1, 2, 3].forEach((n) => rb.push(n));
  assert.deepEqual(rb.toArray(), [1, 2, 3]);
  assert.equal(rb.length, 3);
});

test("keeps exactly the newest N once full", () => {
  const rb = new RingBuffer(3);
  [1, 2, 3, 4, 5].forEach((n) => rb.push(n));
  assert.deepEqual(rb.toArray(), [3, 4, 5]);
  assert.equal(rb.length, 3);
});

test("stays ordered across many wraps", () => {
  const rb = new RingBuffer(4);
  for (let i = 0; i < 103; i++) rb.push(i);
  assert.deepEqual(rb.toArray(), [99, 100, 101, 102]);
});

test("capacity of one keeps only the latest", () => {
  const rb = new RingBuffer(1);
  rb.push("a");
  rb.push("b");
  assert.deepEqual(rb.toArray(), ["b"]);
});

test("clear resets both contents and the write position", () => {
  const rb = new RingBuffer(3);
  [1, 2, 3, 4].forEach((n) => rb.push(n));
  rb.clear();
  assert.deepEqual(rb.toArray(), []);
  assert.equal(rb.length, 0);
  rb.push(9);
  assert.deepEqual(rb.toArray(), [9], "a stale write index would misorder this");
});

test("toArray returns a copy, so callers cannot corrupt the buffer", () => {
  const rb = new RingBuffer(3);
  [1, 2].forEach((n) => rb.push(n));
  rb.toArray().push(999);
  assert.deepEqual(rb.toArray(), [1, 2]);
});

test("rejects a nonsensical capacity instead of silently misbehaving", () => {
  assert.throws(() => new RingBuffer(0), RangeError);
  assert.throws(() => new RingBuffer(-1), RangeError);
  assert.throws(() => new RingBuffer(2.5), RangeError);
});

test("holds a full-capacity run without loss", () => {
  // The real cap. The old shift()-based version re-indexed 10 000 elements on
  // every step past this point.
  const rb = new RingBuffer(10000);
  for (let i = 0; i < 25000; i++) rb.push(i);
  const out = rb.toArray();
  assert.equal(out.length, 10000);
  assert.equal(out[0], 15000);
  assert.equal(out[out.length - 1], 24999);
});
