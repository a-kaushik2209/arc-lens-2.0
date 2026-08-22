/**
 * Which LLM provider an API key belongs to, and which model id that provider
 * will accept.
 *
 * Kept free of any `vscode` import so it stays directly testable — the routing
 * decision is pure, and it is the part that has actually been wrong.
 */

export type Provider = "groq" | "anthropic" | "gemini" | "openai" | "openrouter";

/**
 * Which provider an API key routes to, decided by its prefix.
 *
 * Order matters: an Anthropic key also starts with `sk-`, so it has to be
 * claimed before the OpenAI check sees it.
 */
export function providerFor(apiKey: string): Provider {
  if (apiKey.startsWith("gsk_") || apiKey.includes("groq")) return "groq";
  if (apiKey.startsWith("sk-ant-") || apiKey.includes("anthropic")) return "anthropic";
  if (apiKey.startsWith("AIzaSy") || apiKey.includes("gemini") || apiKey.includes("google")) return "gemini";
  if ((apiKey.startsWith("sk-") && !apiKey.startsWith("sk-or-")) || apiKey.includes("openai")) return "openai";
  return "openrouter";
}

/**
 * The model id to send, given the configured one and the provider it is going to.
 *
 * `arcAgent.llmModel` defaults to an OpenRouter model string
 * ("google/gemini-2.5-flash:free"). Only OpenRouter understands that shape —
 * the `vendor/` prefix and the `:tier` suffix are its conventions — so every
 * native provider needs the bare name.
 *
 * Groq, Anthropic and OpenAI each replaced that default by accident: it
 * contains none of the substrings their guards look for, so the fallback fired.
 * Gemini's guard looks for "gemini", which the OpenRouter string *does*
 * contain, so it passed the check and the unusable id went to Google verbatim.
 * That is the default path for anyone pasting an AIzaSy… key, and it failed
 * with a raw API error. Stripping OpenRouter formatting before the check fixes
 * it, and is a no-op for a model id that was already bare.
 *
 * Falling back to a known-good model for the provider is deliberate: sending a
 * foreign id ("gemini-…" to api.anthropic.com) is a hard API error, while the
 * wrong-but-valid model still answers and is obvious in the reply.
 */
export function modelForProvider(provider: Provider, configured: string): string {
  if (provider === "openrouter") return configured;

  // "vendor/model:tier" -> "model". A bare id is unchanged.
  const bare = configured.split("/").pop()!.split(":")[0];

  switch (provider) {
    case "groq":
      return bare.includes("llama") || bare.includes("mixtral") ? bare : "llama-3.3-70b-versatile";
    case "anthropic":
      return bare.includes("claude") ? bare : "claude-opus-5";
    case "gemini":
      return bare.includes("gemini") ? bare : "gemini-1.5-flash";
    case "openai":
      return bare.startsWith("gpt-") ? bare : "gpt-4o-mini";
  }
}
