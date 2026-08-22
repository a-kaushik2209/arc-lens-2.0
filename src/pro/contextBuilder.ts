/**
 * contextBuilder.ts
 * Formats ARC Lens telemetry and intervention logs into a structured
 * system prompt that is prepended to every LLM chat request.
 */

export interface MetricPoint {
  step: number;
  epoch: number;
  loss: number | null;
  lr: number;
  grad_norm: number;
  gpu_mem_mb?: number;
  advanced?: {
    weight_norm?: number;
    effective_rank?: number;
    weight_update_ratio?: number;
    gradient_entropy?: number;
    grad_flow_ratio?: number;
  };
}

export interface AgentLogEntry {
  type: "failure_detected" | "intervention" | "thought";
  step: number;
  message: string;
  action?: string;
  detail?: string;
}

/**
 * Builds the LLM system prompt from the full run telemetry.
 */
export interface BaselineSummary {
  label: string;
  points: Array<{ step: number; loss: number | null }>;
}

export function buildSystemPrompt(
  metrics: MetricPoint[],
  agentLog: AgentLogEntry[],
  targetFile: string,
  baseline?: BaselineSummary
): string {
  const totalSteps = metrics.length;
  const failures = agentLog.filter((e) => e.type === "failure_detected");
  const interventions = agentLog.filter((e) => e.type === "intervention");

  // Summarize loss trajectory
  // `l !== null` alone let `undefined` through while the predicate claimed
  // otherwise. Ingest now normalises absent fields to null, but this is the
  // formatter's own guard: it must not depend on that to avoid throwing.
  const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
  const lossValues = metrics.map((m) => m.loss).filter(isNum);
  // reduce, not Math.max(...array): the metric history is capped at 10 000
  // entries, and spreading an array that size into a call is close enough to
  // the engine's argument limit that raising the cap would turn this into a
  // stack overflow rather than a slow function.
  const minOf = (xs: number[]) => xs.reduce((a, b) => (b < a ? b : a), Infinity);
  const maxOf = (xs: number[]) => xs.reduce((a, b) => (b > a ? b : a), -Infinity);

  const lossMin = lossValues.length ? minOf(lossValues).toExponential(3) : "N/A";
  const lossMax = lossValues.length ? maxOf(lossValues).toExponential(3) : "N/A";
  const lossFinal = lossValues.length
    ? lossValues[lossValues.length - 1].toExponential(3)
    : "N/A";

  // Summarize gradient norms
  const gradValues = metrics.map((m) => m.grad_norm).filter(isNum);
  const gradMax = gradValues.length ? maxOf(gradValues).toExponential(3) : "N/A";

  // Last few LR values
  const lrValues = metrics.map((m) => m.lr).filter(isNum);
  const lrFinal = lrValues.length
    ? lrValues[lrValues.length - 1].toExponential(3)
    : "N/A";

  // Every field is treated as possibly absent.
  //
  // `grad_norm` and `lr` were formatted unguarded, so a single metric missing
  // either — which happens when ARC cannot observe a value, or when a line
  // arrived truncated — threw inside the async message handler. That surfaced as
  // an unhandled rejection and a chat panel that simply never replied.
  const exp = (v: unknown, digits: number): string =>
    typeof v === "number" && isFinite(v) ? v.toExponential(digits) : "-";
  const fixed = (v: unknown, digits: number): string =>
    typeof v === "number" && isFinite(v) ? v.toFixed(digits) : "-";

  // Build a compact step-sampled trace (max 40 rows to keep token usage low)
  const sample = sampleTrace(metrics, 40);
  const traceRows = sample
    .map((m) =>
      [
        m.step,
        m.loss === null || m.loss === undefined ? "NaN" : exp(m.loss, 3),
        exp(m.grad_norm, 3),
        exp(m.lr, 2),
        fixed(m.advanced?.effective_rank, 2),
        exp(m.advanced?.weight_update_ratio, 3),
      ].join("\t")
    )
    .join("\n");

  // Format intervention log
  const agentSummary =
    agentLog.length === 0
      ? "No events recorded."
      : agentLog
          .map(
            (e) =>
              `[Step ${e.step}] [${e.type.toUpperCase()}] ${
                e.action ? `${e.action}: ` : ""
              }${e.detail || e.message}`
          )
          .join("\n");

  const baselineSection = baseline?.points?.length
    ? `\n## Baseline Comparison ("${baseline.label}")\n${baseline.points
        .map((p) => `[Step ${p.step}] loss=${p.loss === null || p.loss === undefined ? "NaN" : exp(p.loss, 3)}`)
        .join("\n")}\n`
    : "";

  return `You are ARC Analyst, an expert ML training diagnostics AI integrated into ARC Lens, a VS Code extension for PyTorch training monitoring.

## Role
You help ML engineers understand training failures, diagnose pathologies, explain ARC's recovery decisions, and suggest architectural improvements. Be precise, technical, and concise. When suggesting code changes, always show concrete examples.

## Current Training Run
- **File**: ${targetFile}
- **Total Steps Recorded**: ${totalSteps}
- **Failure Events**: ${failures.length}
- **ARC Interventions**: ${interventions.length}

## Metric Summary
- Loss: min=${lossMin}, max=${lossMax}, final=${lossFinal}
- Peak Gradient L2 Norm: ${gradMax}
- Final Learning Rate: ${lrFinal}

## Sampled Metric Trace (step / loss / grad_norm / lr / eff_rank / update_ratio)
${traceRows || "No data yet."}

## ARC Agent Event Log
${agentSummary}
${baselineSection}
---
Answer the user's question with full technical context. If they ask about a specific step, reference the trace data above. If you suggest code changes, show a before/after diff or a complete snippet.`;
}

/**
 * Evenly samples up to `n` points from the metric trace.
 */
function sampleTrace(metrics: MetricPoint[], n: number): MetricPoint[] {
  if (metrics.length <= n) return metrics;
  const step = Math.floor(metrics.length / n);
  const sampled: MetricPoint[] = [];
  for (let i = 0; i < metrics.length; i += step) {
    sampled.push(metrics[i]);
    if (sampled.length >= n) break;
  }
  return sampled;
}
