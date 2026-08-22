import * as vscode from "vscode";
import { isSupportedKey, SUPPORTED_KEY_HINT } from "./providerRouting";

export { isSupportedKey };

/**
 * Feature gating.
 *
 * There is no license check, and deliberately no code that looks like one.
 * `validateLicense` / `getLicenseStatus` used to exist here as unreachable
 * stubs that no caller invoked, which is worse than nothing: a reviewer reads
 * them as a security mechanism and reasons about a gate that never runs.
 *
 * If a real gate is ever added it must verify an **asymmetric** signature —
 * ship the Ed25519/RSA *public* key and keep the private key on the issuing
 * server. A symmetric secret cannot be shipped to a client: a .vsix is a zip,
 * and anyone who unzips it can mint unlimited licenses.
 */
export function isPro(): boolean {
  return true;
}

/**
 * The API key configured by the user, or "" if none is usable.
 */
export function getOpenRouterKey(): string {
  const config = vscode.workspace.getConfiguration("arcAgent");
  const key = (config.get<string>("openRouterKey") || "").trim();
  return isSupportedKey(key) ? key : "";
}

/**
 * Returns the configured OpenRouter model string.
 */
export function getLLMModel(): string {
  const config = vscode.workspace.getConfiguration("arcAgent");
  return config.get<string>("llmModel") ?? "google/gemini-2.5-flash:free";
}

/**
 * Unreachable while isPro() is unconditional; kept so the call sites read
 * honestly rather than implying a paywall that does not exist.
 */
export function promptUpgrade(featureName: string): void {
  vscode.window.showInformationMessage(`ARC Lens: ${featureName} is unlocked for evaluation.`);
}

/**
 * Ensures a usable API key is configured, explaining precisely what is wrong.
 */
export function requireOpenRouterKey(): boolean {
  if (getOpenRouterKey()) return true;

  const raw = (vscode.workspace.getConfiguration("arcAgent").get<string>("openRouterKey") || "").trim();
  // "No key configured" is the wrong message when a key *is* configured and was
  // rejected — the user then has no way to tell that the prefix is the problem.
  const message = raw
    ? `ARC Lens: the configured API key was not recognised. Expected one of: ${SUPPORTED_KEY_HINT}.`
    : "ARC Lens: no API key configured. The AI features use your own key from OpenRouter, Groq, Anthropic, Google or OpenAI.";

  vscode.window.showErrorMessage(message, "Open Settings").then((sel) => {
    if (sel === "Open Settings") {
      vscode.commands.executeCommand("workbench.action.openSettings", "arcAgent.openRouterKey");
    }
  });
  return false;
}

/**
 * Whether to skip the `import arc` preflight before launching a run.
 *
 * The check stays on: ARC Lens works without arc-training (core loss, gradient
 * norm, LR and rollback all function), but the structural diagnostics that
 * distinguish it do not, and the honest place to say so is before the run
 * starts rather than through an empty chart afterwards.
 */
export function shouldBypassArcCheck(): boolean {
  return false;
}
