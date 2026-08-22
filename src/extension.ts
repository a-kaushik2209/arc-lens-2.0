import * as vscode from "vscode";
import * as path from "path";
import * as cp from "child_process";
import * as fs from "fs";
import { isPro, promptUpgrade, getLLMModel, requireOpenRouterKey, shouldBypassArcCheck } from "./pro/licenseManager";
import { buildSystemPrompt, MetricPoint, AgentLogEntry } from "./pro/contextBuilder";
import { streamChatCompletion, ChatMessage } from "./pro/chatManager";
import { buildScriptGenMessages, extractCodeBlock, normalizeNotebook, ScriptGenRequest } from "./pro/scriptGenerator";
import { ChatSession, StreamSession } from "./pro/streamSession";
import { buildReportHtml, RunRecord } from "./pro/reportBuilder";
import { friendlyModelName } from "./pro/modelName";
import { RingBuffer } from "./pro/ringBuffer";

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
let panel: vscode.WebviewPanel | undefined;
let chatPanel: vscode.WebviewPanel | undefined;
let generatorPanel: vscode.WebviewPanel | undefined;
const generatorStream = new StreamSession();
let activeProcess: cp.ChildProcess | undefined;
/** Set by the Stop command so a deliberate SIGTERM is not reported as a crash. */
let stoppedByUser = false;

const metricHistory = new RingBuffer<MetricPoint>(10000);
// Capped like the metric history, and for a stronger reason: every entry is
// inlined verbatim into the system prompt (metrics are sampled down to 40
// rows first), so an unstable run that trips ARC repeatedly used to grow the
// prompt without bound until each chat turn was mostly stale log.
const agentLog = new RingBuffer<AgentLogEntry>(2000);
let activeTargetFile = "";
const chat = new ChatSession();

/** Everything needed to render a post-mortem report for the finished run. */
let currentRun: RunRecord = emptyRun();

function emptyRun(): RunRecord {
  return {
    file: "",
    startedAt: new Date().toISOString(),
    environment: undefined,
    events: [],
    summary: undefined,
    baselineMetrics: undefined,
    mode: "active",
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Activate
// ─────────────────────────────────────────────────────────────────────────────
export function activate(context: vscode.ExtensionContext) {
  // Invalidate the resolved-interpreter cache the instant the setting actually changes.
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("arcAgent.pythonPath")) {
        invalidatePythonPathCache();
      }
    })
  );

  // Register "Run with ARC Lens" command
  context.subscriptions.push(
    vscode.commands.registerCommand("arc-lens.run", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || !editor.document.fileName.endsWith(".py")) {
        vscode.window.showErrorMessage(
          "ARC Lens: Open a Python (.py) training script first."
        );
        return;
      }

      // Save the file before running
      editor.document.save().then(() => {
        launchAgent(editor.document.fileName, context);
      });
    })
  );

  // Register "Stop" command
  context.subscriptions.push(
    vscode.commands.registerCommand("arc-lens.stop", () => {
      if (activeProcess) {
        // `activeProcess` stays set so the close handler still recognises this
        // as the current run and drains the buffered tail. The flag is what
        // stops it reporting the SIGTERM exit code as a training failure — a
        // deliberate stop used to end in a red ERROR banner.
        stoppedByUser = true;
        activeProcess.kill("SIGTERM");
        sendToPanel({ type: "status", status: "stopped", message: "Training stopped by user." });
      }
    })
  );

  // Register "Open Pro Chat" command
  context.subscriptions.push(
    vscode.commands.registerCommand("arc-lens.openChat", () => {
      if (!isPro()) {
        promptUpgrade("AI Failure Analyst");
        return;
      }
      if (!requireOpenRouterKey()) return;
      openChatPanel(context);
    })
  );

  // Register "Generate Script" command
  context.subscriptions.push(
    vscode.commands.registerCommand("arc-lens.generateScript", () => {
      if (!isPro()) {
        promptUpgrade("ARC Script Generator");
        return;
      }
      if (!requireOpenRouterKey()) return;
      openGeneratorPanel(context);
    })
  );

  // Register "Run Baseline (no interventions)" — the A/B control arm.
  context.subscriptions.push(
    vscode.commands.registerCommand("arc-lens.runBaseline", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || !editor.document.fileName.endsWith(".py")) {
        vscode.window.showErrorMessage("ARC Lens: Open a Python (.py) training script first.");
        return;
      }
      editor.document.save().then(() => {
        launchAgent(editor.document.fileName, context, { mode: "baseline" });
      });
    })
  );

  // Register "Export Run Report"
  context.subscriptions.push(
    vscode.commands.registerCommand("arc-lens.exportReport", () => exportReport())
  );
}

async function exportReport(): Promise<void> {
  const metrics = metricHistory.toArray();
  if (metrics.length === 0 && currentRun.events.length === 0) {
    vscode.window.showWarningMessage("ARC Lens: no run recorded yet — run a training script first.");
    return;
  }

  const html = buildReportHtml(currentRun, metrics as any);
  const base = path.basename(currentRun.file || "run", ".py");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const defaultName = `arc-report-${base}-${stamp}.html`;

  const folders = vscode.workspace.workspaceFolders;
  const defaultUri = folders?.length
    ? vscode.Uri.joinPath(folders[0].uri, defaultName)
    : vscode.Uri.file(path.join(require("os").homedir(), defaultName));

  const uri = await vscode.window.showSaveDialog({
    defaultUri,
    filters: { "HTML report": ["html"] },
  });
  if (!uri) return;

  await vscode.workspace.fs.writeFile(uri, Buffer.from(html, "utf8"));
  const open = await vscode.window.showInformationMessage(
    `ARC Lens: report saved to ${path.basename(uri.fsPath)}.`,
    "Open"
  );
  if (open === "Open") {
    vscode.env.openExternal(uri);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch Agent
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Tries the configured interpreter first, then the other common names, so the
 * extension works whether the platform only provides `python3` (most Linux,
 * modern macOS) or only `python`/`py` (the standard Windows installer). Falls
 * back to the configured value untouched if nothing on PATH responds, so
 * downstream error messages still reference what the user actually asked for.
 *
 * Cached per configured value: this spawns up to 4 synchronous child
 * processes, which would otherwise block the extension host on every single
 * "Run with ARC Lens" click. Invalidated by the onDidChangeConfiguration
 * listener registered in activate() the instant arcAgent.pythonPath actually
 * changes — no guessed TTL, no staleness window.
 */
let _resolvedPythonPathCache: { configured: string; resolved: string } | null = null;
let _pythonPathSubstitutionWarned = false;
function invalidatePythonPathCache(): void {
  _resolvedPythonPathCache = null;
  _pythonPathSubstitutionWarned = false;
}

/**
 * Ask the official Python extension which interpreter this file should use.
 *
 * This is the interpreter the user already selected for the workspace — the
 * venv their `torch` and `arc-training` are actually installed into. Guessing a
 * bare name off PATH finds the system Python instead, which is usually the one
 * environment where the dependencies are missing.
 *
 * Returns undefined when the Python extension is absent or exposes no
 * selection, in which case the caller falls back to name resolution.
 */
async function resolveFromPythonExtension(target: vscode.Uri): Promise<string | undefined> {
  try {
    const ext = vscode.extensions.getExtension("ms-python.python");
    if (!ext) return undefined;
    const api = ext.isActive ? ext.exports : await ext.activate();

    // Current API surface.
    const envApi = api?.environments;
    if (envApi?.getActiveEnvironmentPath) {
      const envPath = envApi.getActiveEnvironmentPath(target);
      const resolved = await envApi.resolveEnvironment?.(envPath);
      const executable = resolved?.executable?.uri?.fsPath ?? resolved?.path ?? envPath?.path;
      if (typeof executable === "string" && executable.length > 0) return executable;
    }

    // Legacy API, still present in older Python extension builds.
    const legacy = api?.settings?.getExecutionDetails?.(target)?.execCommand;
    if (Array.isArray(legacy) && typeof legacy[0] === "string" && legacy[0].length > 0) {
      return legacy[0];
    }
  } catch {
    // A failure here is never fatal — fall back to name resolution.
  }
  return undefined;
}

function resolvePythonPath(configured: string): string {
  if (_resolvedPythonPathCache && _resolvedPythonPathCache.configured === configured) {
    return _resolvedPythonPathCache.resolved;
  }
  const candidates = Array.from(new Set([configured, "python3", "python", "py"]));
  for (const candidate of candidates) {
    try {
      cp.execFileSync(candidate, ["--version"], { stdio: "ignore" });
      _resolvedPythonPathCache = { configured, resolved: candidate };
      if (candidate !== configured && !_pythonPathSubstitutionWarned) {
        _pythonPathSubstitutionWarned = true;
        vscode.window.showWarningMessage(
          `ARC Lens: configured Python interpreter "${configured}" wasn't found. Using "${candidate}" instead.`
        );
      }
      return candidate;
    } catch {
      continue;
    }
  }
  return configured;
}

/**
 * The interpreter to run, preferring the user's selected environment.
 *
 * Order: an explicitly configured `arcAgent.pythonPath` wins (the user said so),
 * then the Python extension's selection, then PATH name resolution.
 */
async function resolveInterpreter(targetFile: string): Promise<string> {
  const inspected = vscode.workspace.getConfiguration("arcAgent").inspect<string>("pythonPath");
  const configured = (
    inspected?.globalValue ??
    inspected?.workspaceValue ??
    inspected?.workspaceFolderValue ??
    ""
  ).trim();

  // "Did the user set this?" is answered by `inspect()`, not by comparing
  // against the default string. Comparing meant a user who deliberately set
  // `pythonPath` to "python3" was treated as not having configured anything,
  // and the Python extension's selection silently overrode their explicit
  // choice — the one case where they had actually said what they wanted.
  if (configured) return resolvePythonPath(configured);

  const fromExtension = await resolveFromPythonExtension(vscode.Uri.file(targetFile));
  if (fromExtension) return fromExtension;

  return resolvePythonPath("python3");
}

/**
 * Prompts for the missing arc-training package. `arcKnownPresent` lets a
 * caller that already knows the answer (the preflight doctor's combined
 * import check) skip re-running `import arc` in a second subprocess — the
 * whole point of preflight running its checks in one shot.
 */
function ensureArcTrainingInstalled(pythonPath: string, arcKnownPresent?: boolean): Promise<boolean> {
  if (shouldBypassArcCheck() || arcKnownPresent) {
    return Promise.resolve(true);
  }
  return new Promise((resolve) => {
    cp.execFile(pythonPath, ["-c", "import arc"], (err) => {
      if (!err) {
        resolve(true);
        return;
      }

      vscode.window
        .showWarningMessage(
          "ARC Lens: 'arc-training' is not installed in the selected interpreter. " +
            "Loss, gradient norm, learning rate, checkpointing and rollback all still work — " +
            "the structural diagnostics (effective rank, gradient entropy, update ratio) do not.",
          "Continue Without It",
          "Copy Install Command"
        )
        .then((selection) => {
          if (selection === "Copy Install Command") {
            const cmd = `"${pythonPath}" -m pip install arc-training`;
            vscode.env.clipboard.writeText(cmd).then(
              () => {
                vscode.window.showInformationMessage(
                  "Installation command copied to clipboard! Paste and run it in your active terminal."
                );
              },
              () => {
                vscode.window.showErrorMessage(
                  `Could not access the clipboard. Run this command manually: ${cmd}`
                );
              }
            );
            resolve(false);
          } else {
            // Dismissing proceeds. The missing package degrades the run, it does
            // not invalidate it, and the degraded state is reported in the
            // dashboard header either way.
            resolve(true);
          }
        });
    });
  });
}

/**
 * The check script run inside the target interpreter. Combines the
 * torch/arc import probe with a CUDA device query into one subprocess and
 * one line of JSON, instead of the several round trips a naive version would
 * need — this is what keeps preflight's added latency low.
 */
const PREFLIGHT_CHECK_SCRIPT = `
import json
result = {"torch": False, "arc": False, "cuda": False}
try:
    import torch
    result["torch"] = True
    result["torch_version"] = getattr(torch, "__version__", None)
    if torch.cuda.is_available():
        result["cuda"] = True
        try:
            result["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
except Exception:
    pass
try:
    import arc
    result["arc"] = True
except Exception:
    pass
print(json.dumps(result))
`.trim();

/**
 * Runs the torch/arc/CUDA probe and a syntax check on the target script in
 * parallel, so a misconfigured environment or a broken script fails in
 * ~2 seconds instead of after minutes of training reaching step 0.
 */
async function runPreflightChecks(
  pythonPath: string,
  targetFile: string
): Promise<{
  ok: boolean;
  torch: boolean;
  torchVersion?: string;
  arc: boolean;
  cuda: boolean;
  cudaDevice?: string;
  syntaxError?: string;
}> {
  const importCheck = new Promise<{ torch: boolean; torchVersion?: string; arc: boolean; cuda: boolean; cudaDevice?: string }>(
    (resolve) => {
      cp.execFile(pythonPath, ["-c", PREFLIGHT_CHECK_SCRIPT], (err, stdout) => {
        if (err) {
          // Interpreter couldn't even run the probe — treat as everything missing;
          // the spawn() error handler downstream will surface the real reason.
          resolve({ torch: false, arc: false, cuda: false });
          return;
        }
        try {
          const line = stdout.trim().split("\n").pop() ?? "{}";
          const parsed = JSON.parse(line);
          resolve({
            torch: !!parsed.torch,
            torchVersion: parsed.torch_version,
            arc: !!parsed.arc,
            cuda: !!parsed.cuda,
            cudaDevice: parsed.cuda_device,
          });
        } catch {
          resolve({ torch: false, arc: false, cuda: false });
        }
      });
    }
  );

  const syntaxCheck = new Promise<string | undefined>((resolve) => {
    cp.execFile(pythonPath, ["-m", "py_compile", targetFile], (err, _stdout, stderr) => {
      resolve(err ? (stderr?.trim() || "Syntax error in training script.") : undefined);
    });
  });

  const [imports, syntaxError] = await Promise.all([importCheck, syntaxCheck]);

  return {
    ok: !syntaxError && imports.torch,
    ...imports,
    syntaxError,
  };
}

async function launchAgent(
  targetFile: string,
  context: vscode.ExtensionContext,
  options: { mode?: "active" | "baseline" } = {}
) {
  const config = vscode.workspace.getConfiguration("arcAgent");
  const pythonPath = await resolveInterpreter(targetFile);

  // Preflight: fail in ~2s on a bad interpreter/script instead of minutes
  // into a training run reaching step 0.
  const preflight = await runPreflightChecks(pythonPath, targetFile);

  if (preflight.syntaxError) {
    vscode.window.showErrorMessage(`ARC Lens: syntax error in ${path.basename(targetFile)} — ${preflight.syntaxError}`);
    return;
  }
  if (!preflight.torch) {
    vscode.window.showErrorMessage(
      "ARC Lens: PyTorch not found in the selected interpreter. Install torch in this environment, or point arcAgent.pythonPath at one that has it."
    );
    return;
  }
  // CUDA absence is informational only — CPU-only is a supported mode.

  // arc-training is a soft dependency: preflight already knows whether it's
  // there, so this only prompts (no duplicate `import arc` subprocess).
  const shouldProceed = await ensureArcTrainingInstalled(pythonPath, preflight.arc);
  if (!shouldProceed) {
    return;
  }

  // Kill any existing run
  if (activeProcess) {
    activeProcess.kill("SIGTERM");
    activeProcess = undefined;
  }

  // Create or reveal the dashboard panel
  if (!panel) {
    panel = vscode.window.createWebviewPanel(
      "arcAgentDashboard",
      "ARC Lens - Training Dashboard",
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        localResourceRoots: [
          vscode.Uri.file(path.join(context.extensionPath, "media")),
        ],
        retainContextWhenHidden: true,
      }
    );

    panel.iconPath = {
      light: vscode.Uri.file(path.join(context.extensionPath, "media", "logo_dark.png")),
      dark: vscode.Uri.file(path.join(context.extensionPath, "media", "logo_light.png")),
    };

    // Load the dashboard HTML
    panel.webview.html = getDashboardHtml(context, panel.webview);

    // Handle panel disposal
    panel.onDidDispose(() => {
      panel = undefined;
      if (activeProcess) {
        activeProcess.kill("SIGTERM");
        activeProcess = undefined;
      }
    });

    // Handle messages FROM the webview (e.g., user clicks Stop inside the panel)
    panel.webview.onDidReceiveMessage(async (msg) => {
      if (msg.command === "ready") {
        // The webview's script has attached its message listener; anything
        // queued while it was loading can be delivered now.
        releasePendingMessages();
      } else if (msg.command === "stop") {
        vscode.commands.executeCommand("arc-lens.stop");
      } else if (msg.command === "upgrade") {
        // No license is written here. Writing a hardcoded signed token into the
        // user's global settings to "unlock" a gate that isPro() already leaves
        // open was a backdoor pretending to be a purchase flow. All features are
        // available; the AI ones need the user's own API key and nothing else.
        vscode.window
          .showInformationMessage(
            "ARC Lens: all features are unlocked for evaluation. The AI features need your own OpenRouter/Anthropic/OpenAI API key.",
            "Open Settings"
          )
          .then((sel) => {
            if (sel === "Open Settings") {
              vscode.commands.executeCommand("workbench.action.openSettings", "arcAgent.openRouterKey");
            }
          });
      } else if (msg.command === "openChat") {
        vscode.commands.executeCommand("arc-lens.openChat");
      } else if (msg.command === "openGenerator") {
        vscode.commands.executeCommand("arc-lens.generateScript");
      } else if (msg.command === "exportReport") {
        vscode.commands.executeCommand("arc-lens.exportReport");
      } else if (msg.command === "download") {
        try {
          if (!msg.dataUrl || !msg.dataUrl.includes(',')) {
            throw new Error("Invalid image data received.");
          }
          const workspaceFolders = vscode.workspace.workspaceFolders;
          let defaultUri: vscode.Uri;
          if (workspaceFolders && workspaceFolders.length > 0) {
            defaultUri = vscode.Uri.joinPath(workspaceFolders[0].uri, msg.filename || 'chart.png');
          } else {
            const homedir = require('os').homedir();
            defaultUri = vscode.Uri.file(path.join(homedir, msg.filename || 'chart.png'));
          }

          const uri = await vscode.window.showSaveDialog({
            defaultUri: defaultUri,
            filters: { 'Images': ['png'] }
          });
          if (uri) {
            const base64Data = msg.dataUrl.split(',')[1];
            const buffer = Buffer.from(base64Data, 'base64');
            await vscode.workspace.fs.writeFile(uri, buffer);
            vscode.window.showInformationMessage(`Saved ${msg.filename} successfully!`);
          }
        } catch (err: any) {
          vscode.window.showErrorMessage(`Failed to save chart: ${err.message || err}`);
        }
      }
    });
  } else {
    panel.reveal(vscode.ViewColumn.Beside);
    // Refresh the HTML so the dashboard always has the latest markup
    panel.webview.html = getDashboardHtml(context, panel.webview);
  }

  const mode = options.mode ?? "active";
  stoppedByUser = false;

  // Reset Pro telemetry on new run
  metricHistory.clear();
  agentLog.clear();
  activeTargetFile = targetFile;
  chat.clear();
  const previousBaseline = currentRun.baselineMetrics;
  currentRun = emptyRun();
  currentRun.file = targetFile;
  currentRun.mode = mode;
  // Keep a baseline arm from an earlier run so the active arm can be drawn
  // against it. Comparing the two is the whole point of baseline mode.
  currentRun.baselineMetrics = mode === "baseline" ? undefined : previousBaseline;

  // Wait for the webview to say it is listening, rather than guessing.
  //
  // This was a 500 ms timer. Messages posted before the webview's script has
  // registered its listener are dropped outright, and on a cold panel load —
  // or when the HTML is re-set for a second run — 500 ms is not reliably
  // enough. The dashboard would then never reset: stale file name, stale
  // charts, status stuck on the previous run. In the other direction, any
  // event that arrived *before* the timer fired was wiped by the `start`
  // handler clearing the series.
  //
  // The webview posts `ready` as the last thing its script does. Everything
  // else queues until then, so nothing is dropped and nothing is wiped.
  const startMessage = {
    type: "start",
    file: path.basename(targetFile),
    timestamp: new Date().toISOString(),
    isPro: isPro(),
    mode,
    baseline: currentRun.baselineMetrics,
  };
  panelReady = false;
  pendingUntilReady = [];
  sendToPanel(startMessage);
  // Belt and braces: if a panel is already open and loaded it will not send a
  // fresh `ready`, so release the queue shortly regardless.
  setTimeout(() => releasePendingMessages(), 800);

  const stepDelay: number = config.get("stepDelay") ?? 0;
  const maxCheckpointMB: number = config.get("maxCheckpointMB") ?? 512;
  // Guarded: a value below 1 would make the modulo gate in the harness emit
  // nothing at all, and a fractional one would never match.
  const telemetryEvery: number = Math.max(1, Math.floor(config.get<number>("telemetryEvery") ?? 1));
  const runnerScript = path.join(context.extensionPath, "python", "runner.py");

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    ARC_STEP_DELAY: stepDelay.toString(),
    ARC_MODE: mode,
    ARC_MAX_CHECKPOINT_MB: maxCheckpointMB.toString(),
    // The coalescing policy existed in the harness but had no way in: it read
    // ARC_METRIC_EVERY, which nothing set and no setting exposed, so a default
    // install emitted one event per step and realised none of the measured
    // -72.2% reduction. Surfacing it is the difference between a built feature
    // and a usable one.
    ARC_METRIC_EVERY: telemetryEvery.toString(),
  };

  // ── Spawn the Python backend ───────────────────────────────────────────────
  activeProcess = cp.spawn(pythonPath, [runnerScript, targetFile], {
    env,
    cwd: path.dirname(targetFile),
  });
  const runProcess = activeProcess;

  // An unhandled 'error' on a ChildProcess is re-thrown as an uncaught exception
  // in the extension host. `resolvePythonPath` returns the configured value
  // untouched when nothing on PATH answers, so a stale venv path in settings
  // reaches spawn() and fails here — the user would see VS Code throw rather
  // than a message telling them which interpreter could not be started.
  runProcess.on("error", (err: NodeJS.ErrnoException) => {
    if (activeProcess === runProcess) activeProcess = undefined;
    const hint =
      err.code === "ENOENT"
        ? `Could not start the Python interpreter "${pythonPath}". Check arcAgent.pythonPath, or select an interpreter with the Python extension.`
        : `Failed to start the Python interpreter "${pythonPath}": ${err.message}`;
    vscode.window.showErrorMessage(`ARC Lens: ${hint}`);
    sendToPanel({ type: "error", message: hint });
    sendToPanel({ type: "status", status: "error", message: hint });
  });

  let stdoutBuffer = "";
  let messageBatch: any[] = [];
  let batchTimer: NodeJS.Timeout | null = null;

  const flushBatch = () => {
    if (batchTimer) {
      clearTimeout(batchTimer);
      batchTimer = null;
    }
    if (messageBatch.length > 0) {
      sendToPanel({ type: "batch", events: messageBatch });
      messageBatch = [];
    }
  };

  const ingest = (raw: string) => {
    const trimmed = raw.trim();
    if (!trimmed) return;

    let parsed: any;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      // A telemetry event can arrive with the user's own output glued to the
      // front of it. `print(".", end="", flush=True)` — progress dots, hand-
      // rolled bars — leaves a fragment with no newline in the pipe, so the
      // next emit reads as `.{"type":"metric",...}`. Parsing the whole line
      // strictly meant that metric was demoted to a log string and lost from
      // the history, the report and the chart.
      //
      // ARC's own events are always a single JSON object on one line, so the
      // first `{` is a safe split point: text before it is the user's, and the
      // remainder is retried as an event.
      const brace = trimmed.indexOf("{");
      if (brace > 0) {
        const prefix = trimmed.slice(0, brace).trim();
        try {
          parsed = JSON.parse(trimmed.slice(brace));
          if (prefix) messageBatch.push({ type: "log", message: prefix });
        } catch {
          messageBatch.push({ type: "log", message: trimmed });
          return;
        }
      } else {
        messageBatch.push({ type: "log", message: trimmed });
        return;
      }
    }

    if (parsed === null || typeof parsed !== "object" || typeof parsed.type !== "string") {
      // Valid JSON that is not one of our events — a user printing a dict, say.
      messageBatch.push({ type: "log", message: trimmed });
      return;
    }

    if (parsed.type === "metric") {
      // Normalise before storing, rather than guarding at each of the six
      // places that later format these fields.
      //
      // Anything on stdout that parses as JSON with type "metric" lands here,
      // including a user script printing its own dict — `{"type":"metric",
      // "step":1}` is enough. Every consumer then assumed the numeric fields
      // were present: `lr` was read unfiltered and `.toExponential()` called on
      // it, throwing inside the async chat handler and leaving the AI Analyst
      // silent with no error shown. The loss filter had the same hole, its
      // `l is number` predicate rejecting only `null` while letting `undefined`
      // through.
      //
      // A missing value becomes `null`, which every formatter already renders
      // as a gap. An absent measurement is not a bad one — see the finite
      // guard in the harness for the same rule applied at the source.
      const num = (v: unknown): number | null =>
        typeof v === "number" && Number.isFinite(v) ? v : null;
      metricHistory.push({
        ...parsed,
        step: typeof parsed.step === "number" ? parsed.step : metricHistory.length + 1,
        loss: num(parsed.loss),
        grad_norm: num(parsed.grad_norm),
        lr: num(parsed.lr),
        gpu_mem_mb: num(parsed.gpu_mem_mb),
      } as MetricPoint);
    } else if (
      parsed.type === "failure_detected" ||
      parsed.type === "intervention" ||
      parsed.type === "thought"
    ) {
      agentLog.push({
        type: parsed.type,
        step: parsed.step ?? metricHistory.length,
        message: parsed.message ?? parsed.reason ?? "",
        action: parsed.action,
        detail: parsed.detail,
      } as AgentLogEntry);
    } else if (parsed.type === "environment") {
      // The dashboard prices preserved compute from the detected GPU; the user's
      // own configured rate always wins over the built-in estimate.
      parsed.gpuHourlyRate = config.get<number>("gpuHourlyRate") ?? 0;
      currentRun.environment = parsed;
    } else if (parsed.type === "run_summary") {
      currentRun.summary = parsed;
    }

    if (parsed.type !== "metric" && parsed.type !== "risk") {
      currentRun.events.push(parsed);
    }
    messageBatch.push(parsed);
  };

  runProcess.stdout?.on("data", (chunk: Buffer) => {
    stdoutBuffer += chunk.toString();
    const lines = stdoutBuffer.split("\n");
    stdoutBuffer = lines.pop() || ""; // keep incomplete line in buffer
    for (const line of lines) ingest(line);
    if (!batchTimer) {
      batchTimer = setTimeout(flushBatch, 100);
    }
  });

  runProcess.stderr?.on("data", (chunk: Buffer) => {
    const text = chunk.toString().trim();
    if (text) {
      sendToPanel({ type: "error", message: text });
    }
  });

  runProcess.on("close", (code) => {
    // A superseded run must not narrate over the run that replaced it.
    //
    // `launchAgent` kills any previous process, but the dead one's close handler
    // still fires afterwards — against the *new* run's globals. It reported
    // `exitCode: null`, flipping the freshly-started run to ERROR, and if the
    // killed run was a baseline it overwrote `baselineMetrics` with the new
    // run's points, so the A/B overlay compared the active run against itself.
    // The same path made every user-initiated Stop end in a red ERROR banner.
    if (activeProcess !== runProcess) {
      // This run was killed by a newer one starting. Its own batchTimer must
      // not survive to flush stale events into the new run's panel later.
      if (batchTimer) {
        clearTimeout(batchTimer);
        batchTimer = null;
      }
      return;
    }
    activeProcess = undefined;

    // A final write with no trailing newline stays parked in stdoutBuffer.
    // That last line is very often the one that matters — the run summary or
    // the completion status — so it has to be drained before `done` is sent.
    if (stdoutBuffer.trim()) {
      ingest(stdoutBuffer);
      stdoutBuffer = "";
    }
    // Flush synchronously rather than letting a timer fire after `done`, which
    // is how the dashboard used to show COMPLETED and then keep appending
    // metrics behind it.
    flushBatch();

    if (stoppedByUser) {
      // `stopped` has to be its own field. exitCode 0 is used here because a
      // deliberate SIGTERM is not a crash, but the dashboard reads exitCode
      // alone to choose COMPLETED vs ERROR — so a killed run rendered as a
      // successful one, in the status badge, the strip, and the exported
      // report. Neither COMPLETED nor ERROR is true; stopped is a third state.
      sendToPanel({
        type: "done",
        exitCode: 0,
        stopped: true,
        mode,
        message: "Training stopped by user.",
      });
      return;
    }

    if (mode === "baseline") {
      currentRun.baselineMetrics = {
        label: `baseline (${path.basename(targetFile)})`,
        points: metricHistory.toArray().map((m) => ({ step: m.step, loss: m.loss })),
      };
    }

    sendToPanel({
      type: "done",
      exitCode: code,
      mode,
      message: code === 0 ? "Training completed successfully." : `Process exited with code ${code}.`,
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Whether the dashboard webview has reported that its listener is attached.
 *
 * Anything posted before that is silently discarded by the webview, so events
 * queue here instead of being lost.
 */
let panelReady = false;
let pendingUntilReady: Record<string, unknown>[] = [];

function sendToPanel(event: Record<string, unknown>) {
  if (!panel) return;
  if (!panelReady) {
    pendingUntilReady.push(event);
    return;
  }
  panel.webview.postMessage(event);
}

function releasePendingMessages() {
  if (panelReady) return;
  panelReady = true;
  const queued = pendingUntilReady;
  pendingUntilReady = [];
  for (const event of queued) panel?.webview.postMessage(event);
}

/**
 * Per-load nonce for the webview CSP.
 *
 * `script-src 'unsafe-inline'` disables exactly the protection the header
 * exists to provide, which matters most in the chat panel where model output is
 * rendered into the DOM. A nonce restores it: only the script tags this file
 * stamped can run, so an injected `<script>` from any source cannot.
 */
function makeNonce(): string {
  return require("crypto").randomBytes(16).toString("base64");
}

function getDashboardHtml(
  context: vscode.ExtensionContext,
  webview: vscode.Webview
): string {
  // We inline the HTML directly (avoids URI issues). Load from disk.
  const fs = require("fs");
  try {
    let html = fs.readFileSync(
      path.join(context.extensionPath, "media", "dashboard.html"),
      "utf8"
    );
    const logoUri = webview.asWebviewUri(
      vscode.Uri.file(path.join(context.extensionPath, "media", "logo.png"))
    );
    const echartsUri = webview.asWebviewUri(
      vscode.Uri.file(path.join(context.extensionPath, "media", "vendor", "echarts.min.js"))
    );
    const nonce = makeNonce();
    html = html.replace("{{LOGO_URI}}", logoUri.toString());
    html = html.replace("{{IS_PRO_PLACEHOLDER}}", isPro() ? "true" : "false");
    html = html.replace("{{ECHARTS_URI}}", echartsUri.toString());
    html = html.split("{{CSP_SOURCE}}").join(webview.cspSource);
    html = html.split("{{NONCE}}").join(nonce);
    return html;
  } catch {
    return `<html><body><h1>ARC Lens</h1><p>Could not load dashboard.html</p></body></html>`;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pro: Chat Panel
// ─────────────────────────────────────────────────────────────────────────────
function openChatPanel(context: vscode.ExtensionContext) {
  if (chatPanel) {
    chatPanel.reveal(vscode.ViewColumn.Beside);
    return;
  }

  chatPanel = vscode.window.createWebviewPanel(
    "arcLensChat",
    "ARC Analyst",
    vscode.ViewColumn.Beside,
    { enableScripts: true, retainContextWhenHidden: true }
  );

  chatPanel.webview.html = getChatHtml(getFriendlyModelName(), chat.history);

  chatPanel.onDidDispose(() => {
    // chat.history outlives the panel, so a turn truncated by closing it would
    // otherwise be replayed to the model as a complete answer on reopen.
    chat.cancel();
    chatPanel = undefined;
  });

  chatPanel.webview.onDidReceiveMessage(async (msg) => {
    if (msg.command === "chat") {
      const systemPrompt = buildSystemPrompt(
        metricHistory.toArray(),
        agentLog.toArray(),
        activeTargetFile,
        currentRun.baselineMetrics
      );

      chat.send(
        msg.text,
        systemPrompt,
        (h) => streamChatCompletion(chat.messages, h.onChunk, h.onDone, h.onError),
        (event) => chatPanel?.webview.postMessage(event)
      );
    } else if (msg.command === "clear") {
      chat.clear();
    } else if (msg.command === "cancel") {
      chat.cancel();
      chatPanel?.webview.postMessage({ type: "stream_done" });
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Pro: Script Generator Panel
// ─────────────────────────────────────────────────────────────────────────────
function getFriendlyModelName(): string {
  return friendlyModelName(getLLMModel());
}

function openGeneratorPanel(context: vscode.ExtensionContext) {
  if (generatorPanel) {
    generatorPanel.reveal(vscode.ViewColumn.Beside);
    return;
  }

  const genPanel = vscode.window.createWebviewPanel(
    "arcLensGenerator",
    "ARC Script Generator",
    vscode.ViewColumn.Beside,
    { enableScripts: true }
  );
  generatorPanel = genPanel;

  genPanel.webview.html = getGeneratorHtml(getFriendlyModelName());

  // Without this the panel is unreachable after being closed, its in-flight
  // request keeps streaming, and postMessage fires at a disposed webview.
  genPanel.onDidDispose(() => {
    generatorStream.cancel();
    generatorPanel = undefined;
  });

  genPanel.webview.onDidReceiveMessage(async (msg) => {
    if (msg.command === "cancel") {
      generatorStream.cancel();
      genPanel.webview.postMessage({ type: "done" });
      return;
    }
    if (msg.command !== "generate") return;

    const req = msg.request as ScriptGenRequest;
    const messages = buildScriptGenMessages(req);

    genPanel.webview.postMessage({ type: "generating" });

    let fullResponse = "";
    generatorStream.start(
      (h) => streamChatCompletion(messages, h.onChunk, h.onDone, h.onError),
      {
      onChunk: (chunk) => { fullResponse += chunk; },
      onDone: async () => {
        if (!generatorPanel) return; // panel closed mid-stream
        const code = extractCodeBlock(fullResponse, req.outputFormat);
        if (!code) {
          const tail = fullResponse.slice(-200);
          genPanel.webview.postMessage({
            type: "error",
            text: `Failed to extract code from response — no closing code fence found. The response may have been cut off. Last part of response: ${tail}`,
          });
          return;
        }

        // .py is checked with py_compile; .ipynb is JSON, so it gets a schema
        // check instead. `fileText` is what actually reaches disk — the
        // notebook path rewrites it with the envelope filled in.
        let fileText = code;
        let syntaxError: string | undefined;
        if (req.outputFormat === "ipynb") {
          const nb = normalizeNotebook(code);
          if ("error" in nb) {
            syntaxError = nb.error;
          } else {
            fileText = nb.json;
          }
        } else if (req.outputFormat === "py") {
          const tmpFile = path.join(require("os").tmpdir(), `arc_gen_${Date.now()}.py`);
          fs.writeFileSync(tmpFile, code, "utf8");
          try {
            // A tmpdir path belongs to no workspace folder, so resolving the
            // interpreter *from* it falls through to bare "python3" on PATH
            // instead of the project's configured venv/conda env. Resolve
            // from a real in-workspace file instead, falling back to the
            // tmp file only when there is no workspace context at all.
            const referenceFile =
              vscode.window.activeTextEditor?.document.uri.fsPath ??
              vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ??
              tmpFile;
            const pythonPath = await resolveInterpreter(referenceFile);
            syntaxError = await new Promise<string | undefined>((resolve) => {
              cp.execFile(pythonPath, ["-m", "py_compile", tmpFile], (err, _stdout, stderr) => {
                if (!err) {
                  resolve(undefined);
                } else if ((err as NodeJS.ErrnoException).code === "ENOENT") {
                  resolve(`could not verify — Python interpreter "${pythonPath}" not found. Check arcAgent.pythonPath.`);
                } else {
                  resolve(stderr?.trim() || "Syntax error in generated script.");
                }
              });
            });
          } finally {
            fs.unlink(tmpFile, () => {});
          }
        }

        if (syntaxError) {
          const choice = await vscode.window.showErrorMessage(
            req.outputFormat === "ipynb"
              ? `ARC Script Generator: the generated notebook failed its schema check — ${syntaxError}`
              : `ARC Script Generator: the generated script failed a syntax check — ${syntaxError}`,
            "Save Anyway (Unverified)"
          );
          if (choice !== "Save Anyway (Unverified)") {
            genPanel.webview.postMessage({ type: "done" });
            return;
          }
        }

        const ext = req.outputFormat === "py" ? "py" : "ipynb";
        const defaultName = `arc_train.${ext}`;
        const uri = await vscode.window.showSaveDialog({
          defaultUri: vscode.Uri.file(path.join(require("os").homedir(), defaultName)),
          filters: req.outputFormat === "py"
            ? { "Python Script": ["py"] }
            : { "Jupyter Notebook": ["ipynb"] },
        });

        if (uri) {
          fs.writeFileSync(uri.fsPath, fileText, "utf8");
          vscode.window.showInformationMessage(
            // "ARC-tested" overclaimed what ran: py_compile parses the file, it
            // never executes it and never puts it under ARC.
            syntaxError
              ? `Script saved (not verified): ${path.basename(uri.fsPath)}`
              : req.outputFormat === "py"
                ? `Syntax verified — script saved: ${path.basename(uri.fsPath)}`
                : `Notebook schema verified — saved: ${path.basename(uri.fsPath)}`
          );
          vscode.window.showTextDocument(uri);
        }
        genPanel.webview.postMessage({ type: "done" });
      },
      onError: (err) => {
        if (!generatorPanel) return;
        genPanel.webview.postMessage({ type: "error", text: err });
      },
      }
    );
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Pro: Webview HTML helpers
// ─────────────────────────────────────────────────────────────────────────────
function getChatHtml(modelName: string, history: ChatMessage[]): string {
  const nonce = makeNonce();
  // Replayed into the webview's JS state below so the visible transcript
  // matches chatHistory, which survives panel close/reopen on the extension
  // host even though the webview HTML is rebuilt empty each time.
  const replayHistory = history
    .filter((m) => m.role !== "system")
    .map((m) => ({ role: m.role, content: m.content }));
  const replayHistoryJson = JSON.stringify(replayHistory)
    .replace(/</g, "\\u003c")
    .replace(/'/g, "\\u0027");
  return `<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'nonce-${nonce}';">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<title>ARC Analyst</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  background: #000000;
  color: #ededed;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}
header{
  padding: 16px 20px;
  border-bottom: 1px solid #27272a;
  background: #000000;
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  z-index: 10;
}
.title{
  font-size: 14px;
  font-weight: 600;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  color: #ededed;
}
.pro-badge {
  background: transparent;
  border: 1px solid #333333;
  color: #a1a1aa;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
  display: inline-flex;
  align-items: center;
}
.model-badge {
  font-size: 11px;
  color: #888888;
  font-weight: 500;
  margin-left: auto;
}
.btn-clear{
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #a1a1aa;
  padding: 4px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 11px;
  font-weight: 500;
  transition: all 0.2s ease;
}
.btn-clear:hover{
  border-color: rgba(255, 255, 255, 0.2);
  color: #ffffff;
  background: rgba(255, 255, 255, 0.06);
}
#messages{
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
#messages::-webkit-scrollbar{
  width: 5px;
}
#messages::-webkit-scrollbar-thumb{
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}
.msg{
  max-width: 90%;
  animation: fadeIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes fadeIn{
  from{opacity:0;transform:translateY(6px)}
  to{opacity:1;transform:translateY(0)}
}
.msg-user{
  align-self: flex-end;
  background: rgba(139, 92, 246, 0.08);
  border: 1px solid rgba(139, 92, 246, 0.25);
  border-radius: 12px 12px 2px 12px;
  padding: 12px 16px;
  font-size: 13px;
  line-height: 1.5;
  color: #f4f4f5;
}
.msg-assistant{
  align-self: flex-start;
  background: #18181b;
  border: 1px solid #27272a;
  border-radius: 12px 12px 12px 2px;
  padding: 16px;
  max-width: 95%;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.msg-assistant pre{
  background: #09090b;
  border: 1px solid #27272a;
  border-radius: 8px;
  padding: 14px;
  overflow-x: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  margin: 12px 0;
  color: #e4e4e7;
}
.msg-assistant p{
  font-size: 13px;
  line-height: 1.6;
  color: #d4d4d8;
  margin: 8px 0;
}
.msg-assistant p:first-child{
  margin-top: 0;
}
.msg-assistant p:last-child{
  margin-bottom: 0;
}
.msg-assistant code{
  background: rgba(255, 255, 255, 0.06);
  color: #f472b6;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}
.msg-system{
  align-self: center;
  color: #a1a1aa;
  font-size: 11px;
  padding: 6px 14px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  backdrop-filter: blur(8px);
}
.cursor{
  display: inline-block;
  width: 2px;
  height: 14px;
  background: #8b5cf6;
  margin-left: 2px;
  animation: blink 0.8s infinite;
}
@keyframes blink{
  0%,100%{opacity:1}
  50%{opacity:0}
}
.input-area{
  padding: 20px;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
  background: #09090b;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}
.prompt-container {
  border: 1px solid #27272a;
  background: #18181b;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  padding: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  transition: border-color 0.2s, box-shadow 0.2s;
}
.prompt-container {
  background: #000000;
  border: 1px solid #333333;
  border-radius: 8px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  transition: border-color 0.15s ease;
}
.prompt-container:focus-within {
  border-color: #888888;
}
#user-input {
  background: transparent;
  border: none;
  color: #ededed;
  padding: 4px 6px;
  font-family: inherit;
  font-size: 13px;
  resize: none;
  outline: none;
  max-height: 200px;
  line-height: 1.5;
  width: 100%;
}
#user-input::placeholder {
  color: #888888;
}
.prompt-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 6px 0;
}
.telemetry-pill {
  font-size: 11px;
  color: #a1a1aa;
  background: transparent;
  padding: 2px 0px;
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
  user-select: none;
}
.btn-send {
  background: #ededed;
  border: none;
  color: #000000;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s ease;
}
.btn-send:hover {
  background: #ffffff;
}
.btn-send:disabled {
  background: #333333;
  color: #888888;
  cursor: not-allowed;
}
.empty-state{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  gap: 16px;
  color: #a1a1aa;
  text-align: center;
  max-width: 320px;
  margin: 0 auto;
}
.empty-icon-wrapper {
  width: 48px;
  height: 48px;
  border-radius: 8px;
  background: #0a0a0a;
  border: 1px solid #27272a;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #a1a1aa;
  margin-bottom: 8px;
}
.empty-title {
  font-size: 16px;
  font-weight: 600;
  color: #ededed;
}
.empty-desc {
  font-size: 13px;
  line-height: 1.5;
  color: #888896;
}
.empty-sub {
  font-size: 11px;
  color: #52526b;
  border-top: 1px solid rgba(255,255,255,0.03);
  padding-top: 12px;
  width: 100%;
}
</style></head><body>
<header>
  <div class="title">ARC Analyst <span class="pro-badge">PRO</span> <span class="model-badge">${modelName}</span></div>
  <button class="btn-clear" id="btn-cancel" style="display:none" aria-label="Cancel response">Cancel</button>
  <button class="btn-clear" id="btn-clear" aria-label="Clear chat">Clear</button>
</header>
<div id="messages" aria-live="polite">
  <div class="empty-state" id="empty-state">
    <div class="empty-icon-wrapper">
      <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
    </div>
    <div class="empty-title">AI Failure Analyst</div>
    <div class="empty-desc">Ask ARC why your training failed, or request architecture suggestions.</div>
    <div class="empty-sub">Telemetry from the active training run is attached automatically.</div>
  </div>
</div>
<div class="input-area">
  <div class="prompt-container">
    <textarea id="user-input" placeholder="Why did the gradient explode at step 40?" rows="1"
      ></textarea>
    <div class="prompt-footer">
      <div class="telemetry-pill">
        <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
        Telemetry Attached
      </div>
      <button class="btn-send" id="btn-send" aria-label="Send message">
        <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
      </button>
    </div>
  </div>
</div>
<script nonce="${nonce}">
const vscode = acquireVsCodeApi();
let streaming = false;
let streamEl = null;

function clearChat(){
  vscode.postMessage({command:'clear'});
  document.getElementById('messages').innerHTML='<div class="empty-state" id="empty-state"><div class="empty-icon-wrapper"><svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg></div><div class="empty-title">AI Failure Analyst</div><div class="empty-desc">Ask ARC why your training failed, or request architecture suggestions.</div><div class="empty-sub">Telemetry from the active training run is attached automatically.</div></div>';
}

function autoResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,120)+'px'}

function handleKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()}}

function sendMessage(){
  if(streaming) return;
  const input=document.getElementById('user-input');
  const text=input.value.trim();
  if(!text) return;
  hideEmpty();
  appendMsg('user',text);
  input.value='';input.style.height='auto';
  // Latch before the post, not on the stream_start that comes back: the
  // round-trip left a window where a second Enter queued a second request.
  setStreaming(true);
  vscode.postMessage({command:'chat',text});
}

function hideEmpty(){const e=document.getElementById('empty-state');if(e)e.remove()}

function appendMsg(role,text){
  const div=document.createElement('div');
  div.className='msg msg-'+role;
  if(role==='assistant'){div.innerHTML=renderMarkdown(text);}
  else{div.textContent=text;}
  document.getElementById('messages').appendChild(div);
  scrollBottom();
  return div;
}

function renderMarkdown(text){
  // Escape HTML entities first to prevent XSS
  const escaped = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  return escaped
    .replace(/\\\`\\\`\\\`([\\s\\S]*?)\\\`\\\`\\\`/g,'<pre><code>$1</code></pre>')
    .replace(/\\\`([^\\\`]+)\\\`/g,'<code>$1</code>')
    .replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>')
    .replace(/\\*(.+?)\\*/g,'<em>$1</em>')
    .replace(/^### (.+)$/gm,'<h4 style="font-size:13px;font-weight:600;margin:12px 0 4px;color:#fff">$1</h4>')
    .replace(/^## (.+)$/gm,'<h3 style="font-size:14px;font-weight:600;margin:12px 0 4px;color:#fff">$1</h3>')
    .replace(/^- (.+)$/gm,'<li style="margin:4px 0 4px 16px;list-style:disc">$1</li>')
    .replace(/\\n/g,'<br>');
}

function scrollBottom(){const m=document.getElementById('messages');m.scrollTop=m.scrollHeight}

// Listeners rather than inline handlers: this panel renders model output into
// the DOM, so it is the one webview where a nonce-only CSP matters most.
document.getElementById('btn-clear').addEventListener('click', clearChat);
document.getElementById('btn-cancel').addEventListener('click', () => vscode.postMessage({command:'cancel'}));
document.getElementById('btn-send').addEventListener('click', sendMessage);
const inputEl = document.getElementById('user-input');
inputEl.addEventListener('keydown', handleKey);
inputEl.addEventListener('input', () => autoResize(inputEl));

function setStreaming(isStreaming){
  streaming=isStreaming;
  document.getElementById('btn-send').disabled=isStreaming;
  document.getElementById('btn-cancel').style.display=isStreaming?'inline-flex':'none';
}

window.addEventListener('message',e=>{
  const msg=e.data;
  if(msg.type==='stream_start'){
    setStreaming(true);
    streamEl=appendMsg('assistant','');
    streamEl.innerHTML='<span class="cursor"></span>';
  } else if(msg.type==='stream_chunk'){
    if(streamEl){
      const cur=streamEl.dataset.raw||'';
      streamEl.dataset.raw=cur+msg.text;
      streamEl.innerHTML=renderMarkdown(streamEl.dataset.raw)+'<span class="cursor"></span>';
      scrollBottom();
    }
  } else if(msg.type==='stream_done'){
    if(streamEl){streamEl.innerHTML=renderMarkdown(streamEl.dataset.raw||'');}
    streamEl=null;
    setStreaming(false);
  } else if(msg.type==='stream_error'){
    // Two fixes here. The error was only rendered when a stream element
    // happened to exist, so an error arriving outside a live turn vanished
    // and the panel just sat there having answered nothing. And msg.text is
    // whatever the provider put in its error body — concatenating that into
    // innerHTML let a remote host inject markup into the panel.
    hideEmpty();
    const target = streamEl || appendMsg('assistant','');
    target.innerHTML='';
    const p=document.createElement('p');
    p.style.color='#ff4444';
    p.textContent='Error: '+msg.text;
    target.appendChild(p);
    streamEl=null;
    setStreaming(false);
  }
});

// Replay chat history: chatHistory survives on the extension host across
// panel close/reopen, but this HTML is rebuilt empty each time, so without
// this the user sees a blank panel even though the model still has context.
const initialHistory = JSON.parse('${replayHistoryJson}');
if(initialHistory.length){
  hideEmpty();
  for(const m of initialHistory){ appendMsg(m.role==='assistant'?'assistant':'user', m.content); }
}
</script></body></html>`;
}

function getGeneratorHtml(modelName: string): string {
  const nonce = makeNonce();
  return `<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'nonce-${nonce}';">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<title>ARC Script Generator</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  background: #000000;
  color: #ededed;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  padding: 24px 32px;
  min-height: 100vh;
}
.header-container {
  margin-bottom: 32px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
h1{
  font-size: 20px;
  font-weight: 600;
  margin: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  letter-spacing: -0.02em;
  color: #ededed;
}
.pro-badge {
  background: transparent;
  border: 1px solid #333333;
  color: #a1a1aa;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
  display: inline-flex;
  align-items: center;
}
.sub{
  color: #a1a1aa;
  font-size: 13px;
  line-height: 1.5;
}
.card {
  background: #0a0a0a;
  border: 1px solid #27272a;
  border-radius: 8px;
  padding: 24px;
}
.sub{
  color: #888896;
  font-size: 13.5px;
  line-height: 1.5;
}
.card {
  background: rgba(24, 24, 27, 0.4);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
  border-radius: 16px;
  padding: 24px;
  animation: fadeIn 0.5s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
.form-group{
  margin-bottom: 16px;
}
label{
  display: block;
  font-size: 12px;
  color: #a1a1aa;
  margin-bottom: 8px;
  font-weight: 500;
}
select,input,textarea{
  width: 100%;
  background: #000000;
  border: 1px solid #333333;
  border-radius: 6px;
  color: #ededed;
  padding: 10px 12px;
  font-family: inherit;
  font-size: 13px;
  outline: none;
  transition: border-color 0.15s ease;
}
select:focus,input:focus,textarea:focus{
  border-color: #888888;
}
select option {
  background: #000000;
  color: #ededed;
}
textarea{
  resize: vertical;
  min-height: 72px;
  line-height: 1.5;
}
.row{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.format-toggle {
  display: flex;
  background: #000000;
  border: 1px solid #333333;
  padding: 2px;
  border-radius: 6px;
}
.format-btn {
  flex: 1;
  background: transparent;
  border: none;
  color: #a1a1aa;
  padding: 8px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.15s ease;
}
.format-btn:hover {
  color: #ededed;
}
.format-btn.active {
  background: #27272a;
  color: #ededed;
}
.btn-generate {
  width: 100%;
  background: #ededed;
  border: 1px solid #ededed;
  color: #000000;
  padding: 10px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  margin-top: 8px;
  transition: background 0.15s ease;
}
.btn-generate:hover {
  background: #ffffff;
}
.btn-generate:disabled {
  background: #333333;
  border-color: #333333;
  color: #888888;
  cursor: not-allowed;
}
.status{
  margin-top: 16px;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 12.5px;
  display: none;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  animation: fadeIn 0.2s ease;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
.status.generating{
  display: flex;
  background: rgba(16, 185, 129, 0.05);
  border: 1px solid rgba(16, 185, 129, 0.2);
  color: #10b981;
}
.status.error{
  display: flex;
  background: rgba(239, 68, 68, 0.05);
  border: 1px solid rgba(239, 68, 68, 0.2);
  color: #fca5a5;
}
</style></head><body>
<div class="header-container">
  <h1>🛠 ARC Script Generator <span class="pro-badge">PRO</span></h1>
  <p class="sub">Generate ARC-instrumented training scripts ready for Kaggle, Colab, or local GPU.</p>
</div>
<div class="card">
  <div class="form-group">
    <label>Architecture</label>
    <select id="arch">
      <option value="resnet">ResNet-50 (Image Classification)</option>
      <option value="transformer">Transformer (Sequence Tasks)</option>
      <option value="custom_cnn">Custom CNN</option>
      <option value="mlp">MLP / Tabular</option>
      <option value="custom">Custom (placeholder)</option>
    </select>
  </div>
  <div class="form-group">
    <label>Describe your task</label>
    <textarea id="task" placeholder="e.g. Fine-tune ResNet-50 on CIFAR-10 for image classification with 10 classes"></textarea>
  </div>
  <div class="row">
    <div class="form-group">
      <label>Platform</label>
      <select id="platform">
        <option value="kaggle">Kaggle</option>
        <option value="colab">Google Colab</option>
        <option value="local">Local GPU</option>
      </select>
    </div>
    <div class="form-group">
      <label>Optimizer</label>
      <select id="optimizer">
        <option value="AdamW">AdamW</option>
        <option value="Adam">Adam</option>
        <option value="SGD with momentum">SGD + Momentum</option>
      </select>
    </div>
  </div>
  <div class="row">
    <div class="form-group">
      <label>Epochs</label>
      <input type="number" id="epochs" value="20" min="1" max="1000">
    </div>
    <div class="form-group">
      <label>Output Format</label>
      <div class="format-toggle">
        <div class="format-btn active" id="btn-py" data-fmt="py">.py Script</div>
        <div class="format-btn" id="btn-ipynb" data-fmt="ipynb">.ipynb Notebook</div>
      </div>
    </div>
  </div>
  <div class="form-group">
    <label>Extra requirements (optional)</label>
    <input type="text" id="notes" placeholder="e.g. Mixed precision, cosine LR schedule, gradient clipping">
  </div>
  <button class="btn-generate" id="btn-gen">Generate Training Script</button>
  <div class="status" id="status-gen">🔄 Generating with ${modelName}... this may take 15–30 seconds.</div>
  <div class="status" id="status-err"></div>
</div>
<script nonce="${nonce}">
const vscode=acquireVsCodeApi();
let fmt='py';
function setFormat(f){fmt=f;document.getElementById('btn-py').className='format-btn'+(f==='py'?' active':'');document.getElementById('btn-ipynb').className='format-btn'+(f==='ipynb'?' active':'');}
// The one button doubles as Cancel while a stream is open. Closing the panel
// was previously the only way out of a stuck generation.
let busy=false;
function setBusy(on){
  busy=on;
  const btn=document.getElementById('btn-gen');
  btn.textContent=on?'Cancel':'Generate Training Script';
}
function generate(){
  if(busy){setBusy(false);hideStatus('status-gen');vscode.postMessage({command:'cancel'});return;}
  setBusy(true);
  hideStatus('status-gen');
  hideStatus('status-err');
  vscode.postMessage({command:'generate',request:{
    architecture:document.getElementById('arch').value,
    task:document.getElementById('task').value||'image classification',
    platform:document.getElementById('platform').value,
    outputFormat:fmt,
    epochs:parseInt(document.getElementById('epochs').value)||20,
    optimizer:document.getElementById('optimizer').value,
    extraNotes:document.getElementById('notes').value,
  }});
}
function showStatus(id, cls, text) {
  const el = document.getElementById(id);
  el.className = 'status ' + cls;
  if (text) el.textContent = text;
}
function hideStatus(id) {
  document.getElementById(id).className = 'status';
}
document.getElementById('btn-gen').addEventListener('click', generate);
document.querySelectorAll('[data-fmt]').forEach(el =>
  el.addEventListener('click', () => setFormat(el.dataset.fmt)));
window.addEventListener('message',e=>{
  const msg=e.data;
  if(msg.type==='generating'){
    showStatus('status-gen','generating',null);
    hideStatus('status-err');
  } else if(msg.type==='done'){
    setBusy(false);
    hideStatus('status-gen');
    hideStatus('status-err');
  } else if(msg.type==='error'){
    setBusy(false);
    hideStatus('status-gen');
    showStatus('status-err','error','⚠ Error: '+msg.text);
  }
});
</script></body></html>`;
}

export function deactivate() {
  if (activeProcess) {
    activeProcess.kill("SIGTERM");
    activeProcess = undefined;
  }
  // Both streams, not just the chat one. An in-flight generator request used to
  // outlive deactivation, holding an open HTTPS connection with nothing left to
  // deliver its response to.
  chat.cancel();
  generatorStream.cancel();

  chatPanel?.dispose();
  generatorPanel?.dispose();
  panel?.dispose();
}

