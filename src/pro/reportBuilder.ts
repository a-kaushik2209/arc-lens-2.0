/**
 * reportBuilder.ts
 *
 * Renders a finished run as one self-contained HTML incident report.
 *
 * Self-contained matters: the artifact outlives the demo, gets attached to a
 * ticket, and is opened later on a machine that has neither this extension nor
 * a network. So the charts are inline SVG drawn here rather than a charting
 * library, and there is not a single external reference in the output.
 */

export interface RunRecord {
  file: string;
  startedAt: string;
  environment?: Record<string, unknown>;
  events: any[];
  summary?: Record<string, unknown>;
  baselineMetrics?: { label: string; points: Array<{ step: number; loss: number | null }> };
  mode: "active" | "baseline";
}

export interface ReportMetric {
  step: number;
  loss: number | null;
  grad_norm: number;
  lr: number;
}

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmt(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (!isFinite(value)) return "—";
    if (value !== 0 && (Math.abs(value) >= 1e4 || Math.abs(value) < 1e-3)) {
      return value.toExponential(3);
    }
    return String(Math.round(value * 1e6) / 1e6);
  }
  return escapeHtml(value);
}

/**
 * A log-scaled line chart as inline SVG.
 *
 * Loss curves that survive a divergence span many orders of magnitude — the
 * interesting early detail is invisible on a linear axis once a single point
 * hits 1e15.
 */
function sparkChart(
  series: Array<{ label: string; color: string; points: Array<[number, number | null]> }>,
  markers: Array<{ step: number; color: string; label: string }>,
  width = 860,
  height = 260
): string {
  const pad = { left: 58, right: 16, top: 16, bottom: 34 };
  const all = series.flatMap((s) => s.points.filter((p) => p[1] !== null && p[1]! > 0) as Array<[number, number]>);
  if (all.length === 0) {
    return `<p class="empty">No finite loss values were recorded for this run.</p>`;
  }
  // reduce, not Math.min(...array). A report can cover two arms of up to 10 000
  // points each, and spreading 20 000 arguments into a call is the same
  // argument-limit hazard `contextBuilder` already avoids.
  const minOf = (xs: number[]) => xs.reduce((a, b) => (b < a ? b : a), Infinity);
  const maxOf = (xs: number[]) => xs.reduce((a, b) => (b > a ? b : a), -Infinity);

  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const xMin = minOf(xs);
  const xMax = maxOf(xs) || 1;
  const yMin = Math.log10(Math.max(1e-12, minOf(ys)));
  const yMax = Math.log10(maxOf(ys));
  const ySpan = yMax - yMin || 1;

  const px = (x: number) => pad.left + ((x - xMin) / (xMax - xMin || 1)) * (width - pad.left - pad.right);
  const py = (y: number) =>
    height - pad.bottom - ((Math.log10(Math.max(1e-12, y)) - yMin) / ySpan) * (height - pad.top - pad.bottom);

  const paths = series
    .map((s) => {
      let d = "";
      let pen = false;
      for (const [x, y] of s.points) {
        if (y === null || y <= 0 || !isFinite(y)) {
          pen = false; // a gap, not a straight line through missing data
          continue;
        }
        d += `${pen ? "L" : "M"}${px(x).toFixed(1)},${py(y).toFixed(1)}`;
        pen = true;
      }
      return d
        ? `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.6" stroke-linejoin="round"/>`
        : "";
    })
    .join("");

  const markerEls = markers
    .map(
      (m) =>
        `<line x1="${px(m.step).toFixed(1)}" y1="${pad.top}" x2="${px(m.step).toFixed(1)}" y2="${
          height - pad.bottom
        }" stroke="${m.color}" stroke-width="1" stroke-dasharray="4 3"/>`
    )
    .join("");

  const ticks = [0, 0.25, 0.5, 0.75, 1]
    .map((f) => {
      const value = Math.pow(10, yMin + f * ySpan);
      const y = height - pad.bottom - f * (height - pad.top - pad.bottom);
      return `<line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${width - pad.right}" y2="${y.toFixed(
        1
      )}" stroke="var(--grid)" stroke-width="1"/><text x="${pad.left - 8}" y="${(y + 3).toFixed(
        1
      )}" text-anchor="end" class="tick">${value.toExponential(1)}</text>`;
    })
    .join("");

  const legend = series
    .map(
      (s, i) =>
        `<g transform="translate(${pad.left + i * 190},${height - 8})"><line x1="0" y1="-4" x2="16" y2="-4" stroke="${
          s.color
        }" stroke-width="2"/><text x="22" y="0" class="tick">${escapeHtml(s.label)}</text></g>`
    )
    .join("");

  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Loss over training steps (log scale)">
    ${ticks}${markerEls}${paths}${legend}
  </svg>`;
}

export function buildReportHtml(run: RunRecord, metrics: ReportMetric[]): string {
  const failures = run.events.filter((e) => e.type === "failure_detected");
  const interventions = run.events.filter((e) => e.type === "intervention");
  const unrecoverable = run.events.filter((e) => e.type === "unrecoverable");
  const degraded = run.events.filter((e) => e.type === "degraded");

  const losses = metrics.map((m) => m.loss).filter((l): l is number => l !== null && isFinite(l));
  const finalLoss = losses.length ? losses[losses.length - 1] : null;
  const minLoss = losses.length ? losses.reduce((a, b) => Math.min(a, b), Infinity) : null;
  const peakGrad = metrics.reduce((a, m) => Math.max(a, m.grad_norm ?? 0), 0);

  const series = [
    {
      label: run.mode === "baseline" ? "loss (baseline)" : "loss (ARC active)",
      color: "#3b82f6",
      points: metrics.map((m) => [m.step, m.loss] as [number, number | null]),
    },
  ];
  if (run.baselineMetrics?.points?.length) {
    series.push({
      label: run.baselineMetrics.label,
      color: "#ef4444",
      points: run.baselineMetrics.points.map((p) => [p.step, p.loss] as [number, number | null]),
    });
  }

  const markers = [
    ...failures.map((f) => ({ step: f.step ?? 0, color: "#ef4444", label: "failure" })),
    ...interventions.map((i) => ({ step: i.step ?? 0, color: "#22c55e", label: "intervention" })),
  ];

  const timeline = run.events
    .filter((e) =>
      ["failure_detected", "intervention", "unrecoverable", "degraded"].includes(e.type)
    )
    .map((e) => {
      const label =
        e.type === "failure_detected"
          ? `Failure — ${escapeHtml(e.kind ?? "numerical")}`
          : e.type === "intervention"
          ? `Intervention — ${escapeHtml(e.action)}`
          : e.type === "unrecoverable"
          ? "Judged unrecoverable"
          : `Degraded — ${escapeHtml(e.component)}`;
      const detail = escapeHtml(e.detail ?? e.reason ?? e.message ?? "");
      return `<tr><td>${fmt(e.step)}</td><td class="k k-${escapeHtml(e.type)}">${label}</td><td>${detail}</td></tr>`;
    })
    .join("");

  const env = run.environment ?? {};
  const summary = run.summary ?? {};

  const verdict = unrecoverable.length
    ? {
        cls: "bad",
        text: `ARC judged this run unrecoverable after ${fmt(
          unrecoverable[0]?.attempts
        )} recovery attempts. Its recommendation was to stop the run rather than continue paying for it.`,
      }
    : interventions.length
    ? {
        cls: "ok",
        text: `ARC detected ${failures.length} failure event(s) and applied ${interventions.length} intervention(s); the run continued to completion.`,
      }
    : {
        cls: "neutral",
        text: "No failure thresholds were crossed. ARC observed the run without intervening.",
      };

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARC Lens — Run Report — ${escapeHtml(run.file.split(/[\\/]/).pop() ?? "")}</title>
<style>
:root{--bg:#ffffff;--fg:#111114;--muted:#5c5c66;--panel:#f7f7f9;--border:#e3e3e8;--grid:#ececef}
@media (prefers-color-scheme:dark){:root{--bg:#0c0c0f;--fg:#f2f2f4;--muted:#9a9aa4;--panel:#141418;--border:#26262c;--grid:#1e1e24}}
*{box-sizing:border-box}
body{margin:0;padding:40px 24px;background:var(--bg);color:var(--fg);
 font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:920px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:15px;margin:32px 0 10px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px;margin:0 0 24px}
.verdict{padding:14px 16px;border-radius:8px;border:1px solid var(--border);background:var(--panel);margin:0 0 24px}
.verdict.ok{border-left:3px solid #22c55e}
.verdict.bad{border-left:3px solid #ef4444}
.verdict.neutral{border-left:3px solid var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.card .label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.card .value{font-size:17px;font-variant-numeric:tabular-nums;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);vertical-align:top}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
td:first-child{font-variant-numeric:tabular-nums;width:64px;color:var(--muted)}
.k{white-space:nowrap;font-weight:600}
.k-failure_detected{color:#ef4444}.k-intervention{color:#22c55e}
.k-unrecoverable{color:#ef4444}.k-degraded{color:#f59e0b}
svg{width:100%;height:auto;background:var(--panel);border:1px solid var(--border);border-radius:8px}
.tick{fill:var(--muted);font-size:10px}
.empty{color:var(--muted);font-style:italic}
footer{margin-top:36px;padding-top:14px;border-top:1px solid var(--border);color:var(--muted);font-size:12px}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 16px;margin:0;font-size:13px}
dt{color:var(--muted)}dd{margin:0;font-variant-numeric:tabular-nums}
</style></head><body><main>

<h1>ARC Lens — Run Report</h1>
<p class="sub">${escapeHtml(run.file)} · started ${escapeHtml(run.startedAt)} · mode <strong>${escapeHtml(
    run.mode
  )}</strong></p>

<div class="verdict ${verdict.cls}">${verdict.text}</div>

<h2>Summary</h2>
<div class="grid">
  <div class="card"><div class="label">Steps</div><div class="value">${fmt(summary.steps ?? metrics.length)}</div></div>
  <div class="card"><div class="label">Failures</div><div class="value">${failures.length}</div></div>
  <div class="card"><div class="label">Interventions</div><div class="value">${interventions.length}</div></div>
  <div class="card"><div class="label">Final loss</div><div class="value">${fmt(finalLoss)}</div></div>
  <div class="card"><div class="label">Best loss</div><div class="value">${fmt(minLoss)}</div></div>
  <div class="card"><div class="label">Peak grad norm</div><div class="value">${fmt(peakGrad)}</div></div>
  <div class="card"><div class="label">Wall clock</div><div class="value">${fmt(summary.wall_seconds)} s</div></div>
</div>

<h2>Loss (log scale)</h2>
${sparkChart(series, markers)}

<h2>Event timeline</h2>
${timeline ? `<table><thead><tr><th>Step</th><th>Event</th><th>Detail</th></tr></thead><tbody>${timeline}</tbody></table>`
           : `<p class="empty">No failures, interventions or degradations were recorded.</p>`}

<h2>Environment</h2>
<dl>
  <dt>GPU</dt><dd>${fmt(env.gpu) || "—"}</dd>
  <dt>PyTorch</dt><dd>${fmt(env.torch)}</dd>
  <dt>CUDA</dt><dd>${fmt(env.cuda)}</dd>
  <dt>arc-training</dt><dd>${fmt(env.arc)}</dd>
  <dt>Python</dt><dd>${fmt(env.python)}</dd>
  <dt>Instrumented time</dt><dd>${fmt(summary.instrumentation_seconds)} s</dd>
</dl>

${degraded.length ? `<h2>Degraded components</h2><ul>${degraded
      .map((d) => `<li><strong>${escapeHtml(d.component)}</strong> — ${escapeHtml(d.message)}</li>`)
      .join("")}</ul>` : ""}

<footer>
Generated by ARC Lens. Every number in this report is measured from the run
described above; nothing is simulated or interpolated. Gaps in the chart are
steps where a signal was genuinely unavailable, not smoothed over.
</footer>
</main></body></html>`;
}
