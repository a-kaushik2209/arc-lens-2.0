/**
 * Ownership of a single in-flight LLM stream, and the chat turn built on top
 * of one.
 *
 * Both Pro panels had the same race, written out twice. Destroying a request's
 * socket makes it emit `error` on a *later* tick, so a superseded request still
 * runs its terminal callback after its replacement has started. That callback
 * committed the abandoned reply, and nulled the shared cancel handle out from
 * under the live request — leaving it un-cancelable and leaking past disposal.
 *
 * The fix is a generation token, and it lives here rather than in extension.ts
 * so it can be tested: extension.ts imports `vscode` and cannot be loaded
 * outside the extension host.
 */

// Type-only, so requiring this module does not pull in chatManager's `https`
// and `vscode` dependencies.
import type { ChatMessage } from "./chatManager";

export interface StreamHandlers {
  onChunk: (text: string) => void;
  onDone: () => void;
  onError: (err: string) => void;
}

/** Starts a request and returns a function that abandons it. */
export type LaunchStream = (handlers: StreamHandlers) => () => void;

export class StreamSession {
  private generation = 0;
  private cancelCurrent: (() => void) | null = null;

  /**
   * Supersedes any live request and starts a new one.
   *
   * `launch` is handed guarded callbacks: they no-op once the request that
   * owns them has been superseded, so a late callback from an abandoned
   * socket cannot touch state the live request now owns.
   */
  start(launch: LaunchStream, handlers: StreamHandlers): void {
    this.cancelCurrent?.();
    const generation = ++this.generation;
    // A terminal callback fires at most once per request. The token alone does
    // not cover this: it stays current after a request completes, so a second
    // `onDone` — which an SSE stream that sends both `[DONE]` and a socket
    // `end` will produce — was still treated as a second finished turn and
    // appended a second reply. chatManager guards its own callbacks today,
    // but this is the layer that owns the invariant.
    let settled = false;
    const owns = () => generation === this.generation && !settled;
    const settle = () => {
      settled = true;
      // Released before the handler runs, not after: a handler is allowed to
      // start the next request, and clearing afterwards would discard its
      // cancel function.
      this.cancelCurrent = null;
    };

    this.cancelCurrent = launch({
      onChunk: (text) => { if (owns()) handlers.onChunk(text); },
      onDone: () => { if (!owns()) return; settle(); handlers.onDone(); },
      onError: (err) => { if (!owns()) return; settle(); handlers.onError(err); },
    });
  }

  /**
   * Abandons the live request. Bumping the generation is what actually
   * abandons it — the destroyed socket still runs its terminal callback a
   * tick later, and without this that callback would be treated as a
   * completed turn.
   */
  cancel(): void {
    this.cancelCurrent?.();
    this.generation++;
    this.cancelCurrent = null;
  }

  /** True while a request owns this session. */
  get active(): boolean {
    return this.cancelCurrent !== null;
  }
}

/** What the chat panel's webview is told as a turn progresses. */
export type ChatEvent =
  | { type: "stream_start" }
  | { type: "stream_chunk"; text: string }
  | { type: "stream_done" }
  | { type: "stream_error"; text: string };

/**
 * The chat transcript and the stream that appends to it.
 *
 * `history[0]` is the system prompt, rebuilt from current telemetry on every
 * turn so the model sees the run as it stands now rather than as it was when
 * the panel opened.
 */
export class ChatSession {
  history: ChatMessage[] = [];
  private session = new StreamSession();

  get streaming(): boolean {
    return this.session.active;
  }

  send(text: string, systemPrompt: string, launch: LaunchStream, emit: (e: ChatEvent) => void): void {
    if (this.history.length === 0) {
      this.history.push({ role: "system", content: systemPrompt });
    } else {
      this.history[0] = { role: "system", content: systemPrompt };
    }
    this.history.push({ role: "user", content: text });

    emit({ type: "stream_start" });

    let reply = "";
    this.session.start(launch, {
      onChunk: (chunk) => {
        reply += chunk;
        emit({ type: "stream_chunk", text: chunk });
      },
      onDone: () => {
        // A turn cut short can end with nothing streamed. An empty assistant
        // message would be replayed to the model as though it had answered.
        if (reply) this.history.push({ role: "assistant", content: reply });
        emit({ type: "stream_done" });
      },
      onError: (err) => emit({ type: "stream_error", text: err }),
    });
  }

  /** Abandons the live turn without committing its partial reply. */
  cancel(): void {
    this.session.cancel();
  }

  clear(): void {
    this.session.cancel();
    this.history = [];
  }

  /** The messages to send for the current turn. */
  get messages(): ChatMessage[] {
    return this.history;
  }
}
