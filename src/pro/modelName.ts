/**
 * Display name for an LLM model identifier.
 *
 * This used to be a hardcoded lookup table, which had already gone stale: it
 * named models that were current when it was written and are not now, and every
 * new release silently added another wrong entry while the code carried on
 * looking authoritative. Deriving the name from the identifier is correct for
 * models that have not shipped yet, which is the only property that matters for
 * a table nobody will remember to update.
 */
export function friendlyModelName(model: string | undefined | null): string {
  if (!model) return "AI Analyst";

  // 'vendor/model-name-here:free' -> 'Model Name Here (Free)'
  const parts = model.split("/");
  let rawName = parts[parts.length - 1] || model;

  let isFree = false;
  if (rawName.endsWith(":free")) {
    isFree = true;
    rawName = rawName.slice(0, -5);
  }

  const formatted = rawName
    .split(/[-_]/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

  if (!formatted) return "AI Analyst";
  return isFree ? `${formatted} (Free)` : formatted;
}
