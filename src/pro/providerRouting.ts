/**
 * Which LLM provider an API key belongs to, and which model id that provider
 * will accept.
 *
 * Kept free of any `vscode` import so it stays directly testable — the routing
 * decision is pure, and it is the part that has actually been wrong.
 */

export type Provider = "groq" | "anthropic" | "gemini" | "openai" | "openrouter";

/**
 * Prefixes we know how to route.
 *
 * This lives beside `providerFor` rather than in licenseManager because the two
 * must agree: a prefix accepted here that `providerFor` does not recognise gets
 * silently routed to OpenRouter. It was in licenseManager, which imports
 * `vscode` and so could never be tested, and it went stale exactly as you would
 * expect — Google's newer `AQ.…` keys were refused outright.
 */
const SUPPORTED_KEY_PREFIXES = [
  "sk-or-",   // OpenRouter
  "gsk_",     // Groq
  "sk-ant-",  // Anthropic
  "AIzaSy",   // Google AI Studio / Gemini
  "AQ.",      // Google, newer format
  "sk-",      // OpenAI (and compatible); checked last, it is the loosest
];

/**
 * True if the key looks like one we can route.
 *
 * This is guesswork about someone else's key format, and it earns its place
 * only by catching an obviously-wrong paste before a request is made. When in
 * doubt it should let the key through and leave the provider as the authority
 * on its own format — refusing a working key is the worse failure, and is the
 * one that actually happened.
 */
export function isSupportedKey(key: string): boolean {
  return SUPPORTED_KEY_PREFIXES.some((prefix) => key.startsWith(prefix));
}

/**
 * The shipped default for `arcAgent.llmModel`.
 *
 * Duplicated from package.json, which cannot be imported here without dragging
 * the manifest into the bundle. A test asserts the two stay equal.
 */
export const SHIPPED_DEFAULT_MODEL = "google/gemini-2.5-flash:free";

/** The prefixes, for the message shown when a key is refused. */
export const SUPPORTED_KEY_HINT =
  "sk-or-… (OpenRouter), gsk_… (Groq), sk-ant-… (Anthropic), AIzaSy… or AQ.… (Google), or sk-… (OpenAI)";

/**
 * Which provider an API key routes to, decided by its prefix.
 *
 * Order matters: an Anthropic key also starts with `sk-`, so it has to be
 * claimed before the OpenAI check sees it.
 */
export function providerFor(apiKey: string): Provider {
  if (apiKey.startsWith("gsk_") || apiKey.includes("groq")) return "groq";
  if (apiKey.startsWith("sk-ant-") || apiKey.includes("anthropic")) return "anthropic";
  // Google issues both `AIzaSy…` and the newer `AQ.…`. The second was missing
  // here and in the supported-prefix list, so a working Google key was refused
  // in settings and, had it got past that, routed to OpenRouter.
  if (apiKey.startsWith("AIzaSy") || apiKey.startsWith("AQ.") || apiKey.includes("gemini") || apiKey.includes("google")) return "gemini";
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

  // Leaving `arcAgent.llmModel` alone is not a choice of a Gemini model.
  //
  // Stripping the shipped OpenRouter default to a bare id yielded
  // "gemini-2.5-flash" and sent it to Google as though the user had asked for
  // it — a pinned version nobody selected, which Google then began refusing
  // for new projects with a 404. Treat the untouched default as "no
  // preference" and use the provider's own current model instead.
  const chosen = configured !== SHIPPED_DEFAULT_MODEL;

  // "vendor/model:tier" -> "model". A bare id is unchanged.
  const bare = configured.split("/").pop()!.split(":")[0];

  switch (provider) {
    case "groq":
      return chosen && (bare.includes("llama") || bare.includes("mixtral")) ? bare : "llama-3.3-70b-versatile";
    case "anthropic":
      return chosen && bare.includes("claude") ? bare : "claude-opus-5";
    case "gemini":
      // A moving alias, deliberately. Every pinned Gemini id this file has
      // held has eventually been retired underneath it.
      return chosen && bare.includes("gemini") ? bare : "gemini-flash-latest";
    case "openai":
      return chosen && bare.startsWith("gpt-") ? bare : "gpt-4o-mini";
  }
}
