/**
 * StreamSession / ChatSession.
 *
 * These own the generation token that decides which in-flight request may
 * touch shared state. The bug they exist to prevent is not reachable from the
 * webview tests: it lives on the extension host, where a superseded request's
 * socket emits its terminal event a tick *after* its replacement has already
 * started. That is what the fake stream below reproduces — a launch that hands
 * back its callbacks so a test can fire them late, out of order, or from a
 * request that was abandoned long ago.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const OUT = path.join(__dirname, "..", "out", "pro");
const { StreamSession, ChatSession } = require(path.join(OUT, "streamSession.js"));

/**
 * A stream that never completes on its own. Every launched request is kept,
 * so a test can fire a callback on a request the session has already
 * superseded — which is exactly what a destroyed socket does.
 */
function fakeStream() {
  const launched = [];
  const launch = (handlers) => {
    const rec = { handlers, cancelled: false };
    launched.push(rec);
    return () => { rec.cancelled = true; };
  };
  return { launch, launched };
}

const noop = () => {};
const handlers = (sink = {}) => ({
  onChunk: sink.onChunk || noop,
  onDone: sink.onDone || noop,
  onError: sink.onError || noop,
});

// ── StreamSession ───────────────────────────────────────────────────────────

test("a superseded request cannot fire its callbacks", () => {
  const { launch, launched } = fakeStream();
  const session = new StreamSession();
  const seen = [];

  session.start(launch, handlers({ onChunk: (t) => seen.push("first:" + t), onDone: () => seen.push("first:done") }));
  session.start(launch, handlers({ onChunk: (t) => seen.push("second:" + t) }));

  // The first request's socket is destroyed, but its events are still queued.
  launched[0].handlers.onChunk("late");
  launched[0].handlers.onDone();
  launched[1].handlers.onChunk("live");

  assert.deepEqual(seen, ["second:live"]);
});

test("starting a request cancels the one it supersedes", () => {
  const { launch, launched } = fakeStream();
  const session = new StreamSession();
  session.start(launch, handlers());
  session.start(launch, handlers());
  assert.equal(launched[0].cancelled, true);
  assert.equal(launched[1].cancelled, false);
});

test("a superseded request cannot release the live request's cancel handle", () => {
  const { launch, launched } = fakeStream();
  const session = new StreamSession();

  session.start(launch, handlers());
  session.start(launch, handlers());
  // The stale request finishes late. If it were allowed to clear the handle,
  // the live request would become un-cancelable and leak.
  launched[0].handlers.onDone();

  assert.equal(session.active, true, "live request must still be cancelable");
  session.cancel();
  assert.equal(launched[1].cancelled, true);
});

test("cancel abandons the request so its late completion is ignored", () => {
  const { launch, launched } = fakeStream();
  const session = new StreamSession();
  let done = 0;

  session.start(launch, handlers({ onDone: () => done++ }));
  session.cancel();
  launched[0].handlers.onDone();

  assert.equal(done, 0);
  assert.equal(session.active, false);
});

test("a handler may start the next request without losing its cancel handle", () => {
  const { launch, launched } = fakeStream();
  const session = new StreamSession();

  session.start(launch, handlers({
    onDone: () => session.start(launch, handlers()),
  }));
  launched[0].handlers.onDone();

  assert.equal(session.active, true, "the request started from onDone must be cancelable");
  session.cancel();
  assert.equal(launched[1].cancelled, true);
});

test("active tracks the live request", () => {
  const { launch, launched } = fakeStream();
  const session = new StreamSession();
  assert.equal(session.active, false);
  session.start(launch, handlers());
  assert.equal(session.active, true);
  launched[0].handlers.onDone();
  assert.equal(session.active, false);
});

test("an errored request releases the session", () => {
  const { launch, launched } = fakeStream();
  const session = new StreamSession();
  let err = null;
  session.start(launch, handlers({ onError: (e) => { err = e; } }));
  launched[0].handlers.onError("API error 429");
  assert.equal(err, "API error 429");
  assert.equal(session.active, false);
});

// ── ChatSession ─────────────────────────────────────────────────────────────

/** Runs one full turn against a fresh fake stream. */
function turn(chat, text, reply, stream) {
  const events = [];
  chat.send(text, "SYSTEM", stream.launch, (e) => events.push(e));
  const rec = stream.launched[stream.launched.length - 1];
  if (reply !== null) rec.handlers.onChunk(reply);
  rec.handlers.onDone();
  return events;
}

test("a completed turn appends user then assistant, in order", () => {
  const chat = new ChatSession();
  const stream = fakeStream();
  const events = turn(chat, "why plateau?", "the gradient died", stream);

  assert.deepEqual(chat.history.map((m) => m.role), ["system", "user", "assistant"]);
  assert.equal(chat.history[2].content, "the gradient died");
  assert.deepEqual(events.map((e) => e.type), ["stream_start", "stream_chunk", "stream_done"]);
});

test("the system prompt is rebuilt in place, never appended twice", () => {
  const chat = new ChatSession();
  const stream = fakeStream();
  turn(chat, "one", "a", stream);
  chat.send("two", "FRESHER SYSTEM", stream.launch, noop);

  assert.equal(chat.history.filter((m) => m.role === "system").length, 1);
  assert.equal(chat.history[0].content, "FRESHER SYSTEM");
});

test("a turn that streamed nothing leaves no empty assistant message", () => {
  const chat = new ChatSession();
  const stream = fakeStream();
  turn(chat, "question", null, stream);
  assert.deepEqual(chat.history.map((m) => m.role), ["system", "user"]);
});

test("a cancelled turn does not commit its partial reply", () => {
  const chat = new ChatSession();
  const stream = fakeStream();

  chat.send("question", "SYSTEM", stream.launch, noop);
  stream.launched[0].handlers.onChunk("half an ans");
  chat.cancel();
  stream.launched[0].handlers.onDone(); // the socket's late event

  assert.deepEqual(chat.history.map((m) => m.role), ["system", "user"]);
  assert.equal(chat.streaming, false);
});

test("clear cancels the live turn instead of leaving it writing into a fresh history", () => {
  const chat = new ChatSession();
  const stream = fakeStream();

  chat.send("question", "SYSTEM", stream.launch, noop);
  stream.launched[0].handlers.onChunk("reply");
  chat.clear();
  stream.launched[0].handlers.onDone();

  assert.deepEqual(chat.history, []);
  assert.equal(stream.launched[0].cancelled, true);
});

test("a superseded turn's reply never lands behind the newer user turn", () => {
  const chat = new ChatSession();
  const stream = fakeStream();

  chat.send("first", "SYSTEM", stream.launch, noop);
  stream.launched[0].handlers.onChunk("first answer");
  chat.send("second", "SYSTEM", stream.launch, noop); // supersedes
  stream.launched[0].handlers.onDone();               // the late socket event
  stream.launched[1].handlers.onChunk("second answer");
  stream.launched[1].handlers.onDone();

  assert.deepEqual(chat.history.map((m) => m.role), ["system", "user", "user", "assistant"]);
  assert.equal(chat.history[3].content, "second answer");
});

test("a superseded turn cannot emit webview events that close the live stream", () => {
  const chat = new ChatSession();
  const stream = fakeStream();
  const events = [];

  chat.send("first", "SYSTEM", stream.launch, (e) => events.push(e));
  chat.send("second", "SYSTEM", stream.launch, (e) => events.push(e));
  events.length = 0;

  stream.launched[0].handlers.onDone();
  stream.launched[0].handlers.onChunk("stale");

  assert.deepEqual(events, [], "a stale request must not post stream_done mid-answer");
});

// ── Stress ──────────────────────────────────────────────────────────────────

/**
 * Randomised interleaving of sends, cancels and late callbacks.
 *
 * Turn n sends the text "U<n>" and its stream replies in chunks tagged
 * "A<n>". The invariant that catches the original corruption is positional:
 * the assistant message following user "U<n>" must be "A<n>". Merely checking
 * that every assistant follows a user is too weak — the bug produced
 * [U1, U2, A1], which satisfies that and is still wrong.
 */
test("stress: interleaved sends, cancels and late callbacks keep the transcript coherent", () => {
  // Deterministic PRNG so a failure is reproducible from the seed alone.
  let seed = 0x5eed1e;
  const rnd = () => {
    seed ^= seed << 13; seed >>>= 0;
    seed ^= seed >> 17;
    seed ^= seed << 5;  seed >>>= 0;
    return seed / 0x100000000;
  };
  const pick = (n) => Math.floor(rnd() * n);

  const chat = new ChatSession();
  const stream = fakeStream();
  const tagOf = new Map(); // launched index -> turn number
  const terminated = new Set();
  let turns = 0;
  let liveEvents = 0;

  for (let i = 0; i < 4000; i++) {
    const op = pick(5);
    const n = stream.launched.length;

    if (op === 0) {
      turns++;
      const t = turns;
      chat.send("U" + t, "SYSTEM", stream.launch, () => { liveEvents++; });
      tagOf.set(stream.launched.length - 1, t);
    } else if (op === 1) {
      chat.cancel();
    } else if (n > 0) {
      // Fire a callback on an arbitrary request — very often one that was
      // superseded many turns ago and whose socket is only now reporting.
      const idx = pick(n);
      const h = stream.launched[idx].handlers;
      if (op === 2) h.onChunk("A" + tagOf.get(idx));
      else if (op === 3) { h.onDone(); terminated.add(idx); }
      else { h.onError("boom"); terminated.add(idx); }
    }

    // Invariant 1: the transcript is well-formed.
    const roles = chat.history.map((m) => m.role);
    if (roles.length) {
      assert.equal(roles[0], "system", `iteration ${i}: history must open with the system prompt`);
      assert.equal(roles.indexOf("system", 1), -1, `iteration ${i}: only one system message`);
    }

    // Invariant 2: every assistant message answers the user turn directly
    // above it, and carries only that turn's chunks.
    for (let j = 1; j < chat.history.length; j++) {
      if (chat.history[j].role !== "assistant") continue;
      assert.equal(chat.history[j - 1].role, "user", `iteration ${i}: assistant at ${j} does not follow a user turn`);
      const expected = chat.history[j - 1].content.replace("U", "A");
      const tags = new Set(chat.history[j].content.match(/A\d+/g));
      assert.deepEqual([...tags], [expected], `iteration ${i}: assistant at ${j} carries the wrong turn's reply`);
    }

    // Invariant 3: nothing leaks. Every request but the newest has either been
    // cancelled or been handed a terminal event — a request that is neither is
    // an open socket nobody will ever close.
    for (let k = 0; k < stream.launched.length - 1; k++) {
      assert.ok(
        stream.launched[k].cancelled || terminated.has(k),
        `iteration ${i}: request ${k} was left open`
      );
    }
  }

  assert.ok(turns > 500, `expected the stress loop to exercise many turns, got ${turns}`);
  assert.ok(liveEvents > 0, "expected the live request to emit webview events");
});
