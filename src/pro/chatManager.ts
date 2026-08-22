import * as https from "https";
import { getOpenRouterKey, getLLMModel } from "./licenseManager";

const OPENROUTER_HOST = "openrouter.ai";
const OPENROUTER_PATH = "/api/v1/chat/completions";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/**
 * Sends a streaming chat completion request to OpenRouter.
 * Calls `onChunk` with each text delta as it arrives,
 * and `onDone` when the stream ends.
 * Returns a cancel function to abort the request.
 */
export function streamChatCompletion(
  messages: ChatMessage[],
  onChunk: (text: string) => void,
  onDone: () => void,
  onError: (err: string) => void
): () => void {
  const apiKey = getOpenRouterKey();
  if (!apiKey) {
    onError(
      "No API key configured. Please set arcAgent.openRouterKey in VS Code settings."
    );
    onDone();
    return () => {};
  }

  const isGroq = apiKey.startsWith("gsk_") || apiKey.includes("groq");
  const isAnthropic = apiKey.startsWith("sk-ant-") || apiKey.includes("anthropic");
  const isGemini = apiKey.startsWith("AIzaSy") || apiKey.includes("gemini") || apiKey.includes("google");
  const isOpenAI = (apiKey.startsWith("sk-") && !apiKey.startsWith("sk-or-") && !apiKey.startsWith("sk-ant-")) || apiKey.includes("openai");

  let hostname = OPENROUTER_HOST;
  let path = OPENROUTER_PATH;
  let model = getLLMModel();
  let headers: { [key: string]: string } = {
    "Content-Type": "application/json"
  };

  if (isGroq) {
    hostname = "api.groq.com";
    path = "/openai/v1/chat/completions";
    headers["Authorization"] = `Bearer ${apiKey}`;
    if (!model.includes("llama") && !model.includes("mixtral")) {
      model = "llama-3.3-70b-versatile";
    }
  } else if (isAnthropic) {
    hostname = "api.anthropic.com";
    path = "/v1/messages";
    headers["x-api-key"] = apiKey;
    headers["anthropic-version"] = "2023-06-01";
    if (!model.includes("claude")) {
      model = "claude-opus-5";
    }
  } else if (isGemini) {
    hostname = "generativelanguage.googleapis.com";
    path = "/v1beta/openai/chat/completions";
    headers["Authorization"] = `Bearer ${apiKey}`;
    if (!model.includes("gemini")) {
      model = "gemini-1.5-flash";
    }
  } else if (isOpenAI) {
    hostname = "api.openai.com";
    path = "/v1/chat/completions";
    headers["Authorization"] = `Bearer ${apiKey}`;
    if (!model.startsWith("gpt-")) {
      model = "gpt-4o-mini";
    }
  } else {
    // OpenRouter (default)
    hostname = OPENROUTER_HOST;
    path = OPENROUTER_PATH;
    headers["Authorization"] = `Bearer ${apiKey}`;
    headers["HTTP-Referer"] = "https://arc-lens.dev";
    headers["X-Title"] = "ARC Lens Pro";
  }

  let body: string;
  if (isAnthropic) {
    const systemMessage = messages.find(m => m.role === "system")?.content;
    const chatMessages = messages.filter(m => m.role !== "system");
    body = JSON.stringify({
      model,
      messages: chatMessages,
      system: systemMessage,
      stream: true,
      max_tokens: 2048,
      temperature: 0.4,
    });
  } else {
    body = JSON.stringify({
      model,
      messages,
      stream: true,
      max_tokens: 2048,
      temperature: 0.4,
    });
  }

  headers["Content-Length"] = String(Buffer.byteLength(body));

  const options: https.RequestOptions = {
    hostname,
    path,
    method: "POST",
    headers,
  };

  let buffer = "";
  // Anthropic's SSE stream pairs an `event: <type>` line with the `data: `
  // line that follows it; the event type is what tells you whether the data
  // is a text delta or an end-of-stream marker, so it has to be tracked
  // across chunks (the two lines can land in different `data` events).
  let currentEvent = "";

  // Every terminal path below can be reached more than once for a single
  // request: the SSE stream sends `data: [DONE]`, and then the socket also
  // fires `end`. Calling onDone twice appended the assistant's reply to the
  // chat history twice, which then went back to the model as context on the
  // next turn. Errors compound it further — onError is always followed by
  // onDone, and `end` can still arrive after both.
  let settled = false;
  let cancelled = false;
  const finish = () => {
    if (settled) return;
    settled = true;
    onDone();
  };
  const fail = (message: string) => {
    if (settled) return;
    onError(message);
    finish();
  };

  const req = https.request(options, (res) => {
    if (res.statusCode && res.statusCode >= 400) {
      let errBody = "";
      res.on("data", (d: Buffer) => (errBody += d.toString()));
      res.on("end", () => fail(`API error ${res.statusCode}: ${errBody}`));
      return;
    }

    res.on("data", (chunk: Buffer) => {
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        if (trimmed.startsWith("event: ")) {
          currentEvent = trimmed.slice(7).trim();
          continue;
        }
        if (!trimmed.startsWith("data: ")) continue;
        const data = trimmed.slice(6);
        if (data === "[DONE]") {
          finish();
          return;
        }
        try {
          const parsed = JSON.parse(data);
          if (isAnthropic) {
            // Anthropic never sends "data: [DONE]" — the stream ends with a
            // `message_stop` event, and text only ever arrives on
            // `content_block_delta` events shaped as
            // {type:"content_block_delta", delta:{type:"text_delta", text}}.
            if (currentEvent === "content_block_delta" && typeof parsed?.delta?.text === "string") {
              onChunk(parsed.delta.text);
            } else if (currentEvent === "message_stop") {
              finish();
              return;
            }
          } else {
            const delta = parsed?.choices?.[0]?.delta?.content || parsed?.delta?.text;
            if (delta) {
              onChunk(delta);
            }
          }
        } catch {
          // skip malformed SSE lines
        }
      }
    });

    res.on("end", finish);
    res.on("error", (err: Error) => fail(err.message));
  });

  req.on("error", (err: Error) => {
    // Destroying the request to cancel a stream also emits 'error'. That is the
    // caller's own doing, so it must not surface as an API failure in the chat.
    if (cancelled) {
      finish();
      return;
    }
    fail(err.message);
  });

  req.write(body);
  req.end();

  return () => {
    cancelled = true;
    req.destroy();
  };
}
