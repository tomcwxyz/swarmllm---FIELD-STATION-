(() => {
  const MAX_ENTRIES = 600;
  const MAX_RUNS = 250;
  const RUN_STORAGE_KEY = "field-station:experiment-runs:v1";
  const entries = [];
  const startedAt = Date.now();
  let lastStatus = "";
  let activeRun = null;

  function safe(value, depth = 0) {
    if (depth > 3) return "[depth]";
    if (value == null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
    if (Array.isArray(value)) return value.slice(0, 30).map((v) => safe(v, depth + 1));
    if (typeof value === "object") {
      const out = {};
      for (const [key, val] of Object.entries(value).slice(0, 40)) {
        if (/prompt|message|reply|content|text/i.test(key)) continue; // never capture chat content by default
        out[key] = safe(val, depth + 1);
      }
      return out;
    }
    return String(value);
  }

  function loadRuns() {
    try {
      const parsed = JSON.parse(localStorage.getItem(RUN_STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed.slice(-MAX_RUNS) : [];
    } catch {
      return [];
    }
  }

  let experimentRuns = loadRuns();

  function persistRuns() {
    try {
      localStorage.setItem(RUN_STORAGE_KEY, JSON.stringify(experimentRuns.slice(-MAX_RUNS)));
    } catch {}
  }

  function record(type, data = {}) {
    entries.push({
      at: new Date().toISOString(),
      ms: Date.now() - startedAt,
      type,
      data: safe(data),
    });
    if (entries.length > MAX_ENTRIES) entries.splice(0, entries.length - MAX_ENTRIES);
    render();
  }

  function parseNumber(value) {
    const match = String(value || "").match(/-?\d+(?:\.\d+)?/);
    return match ? Number(match[0]) : null;
  }

  function currentModel() {
    const select = document.getElementById("ai-model");
    const option = select?.selectedOptions?.[0];
    return {
      key: select?.value || null,
      label: option?.textContent?.trim() || null,
    };
  }

  function wireSnapshot() {
    const wire = (new URLSearchParams(location.search).get("wire") || "stripe4").toLowerCase();
    const stripes = wire === "off" ? 0 : wire.startsWith("stripe") ? Math.max(1, Math.min(8, Number.parseInt(wire.slice(6), 10) || 1)) : 1;
    return { wire, stripes };
  }

  function clusterSnapshot() {
    const cards = [...document.querySelectorAll(".peer-card")].map((card) => {
      const gpu = card.querySelector(".peer-gpu")?.textContent?.trim() || null;
      const rtt = card.querySelector(".rtt")?.textContent || "";
      const bandwidth = card.querySelector(".bw")?.textContent || "";
      const contribution = card.querySelector(".buf")?.textContent || "";
      return {
        self: card.classList.contains("self"),
        gpu,
        rttMs: /ms/i.test(rtt) ? parseNumber(rtt) : null,
        bandwidthMbps: /mbps/i.test(bandwidth) ? parseNumber(bandwidth) : null,
        contributionGB: /gb/i.test(contribution) ? parseNumber(contribution) : null,
      };
    });

    const summary = document.getElementById("cluster-summary")?.textContent || "";
    const summaryMatch = summary.match(/(\d+)\s+devices?\s+·\s+(\d+)\s+WebGPU\s+·\s+([\d.]+)\s+GB pledged/i);
    const rtts = cards.map((card) => card.rttMs).filter(Number.isFinite).sort((a, b) => a - b);
    const bandwidths = cards.map((card) => card.bandwidthMbps).filter(Number.isFinite);
    const medianRttMs = rtts.length ? rtts[Math.floor(rtts.length / 2)] : null;

    return {
      devices: summaryMatch ? Number(summaryMatch[1]) : cards.length || null,
      webgpuDevices: summaryMatch ? Number(summaryMatch[2]) : cards.filter((card) => card.gpu && !/no WebGPU/i.test(card.gpu)).length,
      pledgedGB: summaryMatch ? Number(summaryMatch[3]) : cards.reduce((sum, card) => sum + (card.contributionGB || 0), 0) || null,
      medianRttMs,
      minBandwidthMbps: bandwidths.length ? Math.min(...bandwidths) : null,
      maxBandwidthMbps: bandwidths.length ? Math.max(...bandwidths) : null,
      devicesDetail: cards,
    };
  }

  function runId() {
    return crypto.randomUUID?.() || `run-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function beginRun(promptTokens) {
    const model = currentModel();
    activeRun = {
      id: runId(),
      startedAt: new Date().toISOString(),
      startedPerf: performance.now(),
      promptTokens,
      modelKey: model.key,
      modelLabel: model.label,
      ...wireSnapshot(),
      clusterAtStart: clusterSnapshot(),
      firstOutputObservedMs: null,
      lastObservedOutputTokens: 0,
      lastObservedTokensPerSecond: null,
    };
    record("experiment:run-start", {
      id: activeRun.id,
      promptTokens,
      modelKey: activeRun.modelKey,
      devices: activeRun.clusterAtStart.devices,
      wire: activeRun.wire,
    });
  }

  function saveRun(run) {
    const clean = safe(run);
    experimentRuns.push(clean);
    if (experimentRuns.length > MAX_RUNS) experimentRuns.splice(0, experimentRuns.length - MAX_RUNS);
    persistRuns();
    record("experiment:run", clean);
  }

  function completeRun({ prefillSeconds, outputTokens, outputTokensPerSecond, devices }) {
    if (!activeRun) beginRun(null);
    const finishedPerf = performance.now();
    const prefillMs = Number.isFinite(prefillSeconds) ? Math.round(prefillSeconds * 1000) : null;
    const decodeMs = Number.isFinite(outputTokensPerSecond) && outputTokensPerSecond > 0
      ? Math.round((outputTokens / outputTokensPerSecond) * 1000)
      : null;
    const clusterAtEnd = clusterSnapshot();
    const run = {
      id: activeRun.id,
      startedAt: activeRun.startedAt,
      finishedAt: new Date().toISOString(),
      status: "complete",
      modelKey: activeRun.modelKey,
      modelLabel: activeRun.modelLabel,
      promptTokens: activeRun.promptTokens,
      outputTokens,
      prefillMs,
      prefillTokensPerSecond: Number.isFinite(prefillSeconds) && prefillSeconds > 0 && Number.isFinite(activeRun.promptTokens)
        ? Number((activeRun.promptTokens / prefillSeconds).toFixed(2))
        : null,
      decodeMs,
      outputTokensPerSecond,
      firstOutputObservedMs: activeRun.firstOutputObservedMs,
      totalObservedMs: Math.round(finishedPerf - activeRun.startedPerf),
      devices: devices || clusterAtEnd.devices || activeRun.clusterAtStart.devices,
      webgpuDevices: clusterAtEnd.webgpuDevices,
      pledgedGB: clusterAtEnd.pledgedGB,
      medianRttMs: clusterAtEnd.medianRttMs,
      minBandwidthMbps: clusterAtEnd.minBandwidthMbps,
      maxBandwidthMbps: clusterAtEnd.maxBandwidthMbps,
      wire: activeRun.wire,
      stripes: activeRun.stripes,
      hostDevice: clusterAtEnd.devicesDetail.find((device) => device.self)?.gpu || null,
      cluster: clusterAtEnd.devicesDetail,
    };
    saveRun(run);
    activeRun = null;
  }

  function failRun(error) {
    if (!activeRun) return;
    const cluster = clusterSnapshot();
    saveRun({
      id: activeRun.id,
      startedAt: activeRun.startedAt,
      finishedAt: new Date().toISOString(),
      status: "failed",
      error: String(error || "generation failed").slice(0, 240),
      modelKey: activeRun.modelKey,
      modelLabel: activeRun.modelLabel,
      promptTokens: activeRun.promptTokens,
      outputTokens: activeRun.lastObservedOutputTokens,
      outputTokensPerSecond: activeRun.lastObservedTokensPerSecond,
      totalObservedMs: Math.round(performance.now() - activeRun.startedPerf),
      devices: cluster.devices || activeRun.clusterAtStart.devices,
      webgpuDevices: cluster.webgpuDevices,
      pledgedGB: cluster.pledgedGB,
      medianRttMs: cluster.medianRttMs,
      minBandwidthMbps: cluster.minBandwidthMbps,
      maxBandwidthMbps: cluster.maxBandwidthMbps,
      wire: activeRun.wire,
      stripes: activeRun.stripes,
      hostDevice: cluster.devicesDetail.find((device) => device.self)?.gpu || null,
      cluster: cluster.devicesDetail,
    });
    activeRun = null;
  }

  function observeAiStatus() {
    const status = document.getElementById("ai-status");
    if (!status) return;

    const inspect = () => {
      const value = status.textContent?.trim() || "";
      if (!value || value === lastStatus) return;
      lastStatus = value;

      const prefill = value.match(/^prefill:\s*(\d+)\s+tokens/i);
      if (prefill) {
        if (activeRun) failRun("superseded by a new generation");
        beginRun(Number(prefill[1]));
        return;
      }

      const generating = value.match(/^generating…?\s*(\d+)\s+tok\s*·\s*([\d.]+)\s+tok\/s/i);
      if (generating) {
        if (!activeRun) beginRun(null);
        if (activeRun.firstOutputObservedMs == null) activeRun.firstOutputObservedMs = Math.round(performance.now() - activeRun.startedPerf);
        activeRun.lastObservedOutputTokens = Number(generating[1]);
        activeRun.lastObservedTokensPerSecond = Number(generating[2]);
        return;
      }

      const ready = value.match(/^ready\s+[—-]\s+prefill\s+([\d.]+)s,\s*(\d+)\s+tok\s*·\s*([\d.]+)\s+tok\/s\s*·\s*(\d+)\s+devices?/i);
      if (ready) {
        completeRun({
          prefillSeconds: Number(ready[1]),
          outputTokens: Number(ready[2]),
          outputTokensPerSecond: Number(ready[3]),
          devices: Number(ready[4]),
        });
        return;
      }

      const failed = value.match(/^generation failed:\s*(.+)$/i);
      if (failed) failRun(failed[1]);
    };

    new MutationObserver(inspect).observe(status, { childList: true, subtree: true, characterData: true });
    inspect();
  }

  function snapshot() {
    return {
      fieldStation: "collective-compute",
      capturedAt: new Date().toISOString(),
      page: location.origin + location.pathname,
      userAgent: navigator.userAgent,
      online: navigator.onLine,
      experimentRuns: [...experimentRuns],
      entries: [...entries],
    };
  }

  async function copy() {
    const text = JSON.stringify(snapshot(), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      record("diagnostics:copied", { entries: entries.length, runs: experimentRuns.length });
      return true;
    } catch {
      return false;
    }
  }

  function download() {
    const blob = new Blob([JSON.stringify(snapshot(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `field-station-diagnostics-${new Date().toISOString().replaceAll(":", "-")}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    record("diagnostics:exported", { entries: entries.length, runs: experimentRuns.length });
  }

  function csvCell(value) {
    if (value == null) return "";
    const string = String(value);
    return /[",\n]/.test(string) ? `"${string.replaceAll('"', '""')}"` : string;
  }

  function downloadRunsCsv() {
    const fields = [
      "startedAt", "status", "modelKey", "modelLabel", "promptTokens", "outputTokens",
      "prefillMs", "prefillTokensPerSecond", "decodeMs", "outputTokensPerSecond",
      "firstOutputObservedMs", "totalObservedMs", "devices", "webgpuDevices", "pledgedGB",
      "medianRttMs", "minBandwidthMbps", "maxBandwidthMbps", "wire", "stripes", "hostDevice", "error",
    ];
    const rows = [fields.join(","), ...experimentRuns.map((run) => fields.map((field) => csvCell(run[field])).join(","))];
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `field-station-runs-${new Date().toISOString().replaceAll(":", "-")}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    record("experiment:csv-exported", { runs: experimentRuns.length });
  }

  function clearRuns() {
    experimentRuns = [];
    persistRuns();
    render();
    record("experiment:runs-cleared");
  }

  function ensureDialog() {
    let dialog = document.getElementById("fs-diagnostics-dialog");
    if (dialog) return dialog;
    dialog = document.createElement("dialog");
    dialog.id = "fs-diagnostics-dialog";
    dialog.innerHTML = `
      <div class="diag-head">
        <div>
          <div class="fs-type">APPARATUS · EXPERIMENT LOG</div>
          <h2>What is the room doing?</h2>
        </div>
        <button type="button" class="diag-close" aria-label="Close diagnostics">Close</button>
      </div>
      <p class="diag-note">Run metrics are kept in this browser so experiments can be compared later. Prompt and answer text are deliberately excluded.</p>
      <div class="diag-run-summary" aria-live="polite"></div>
      <div class="diag-actions">
        <button type="button" data-diag-runs-csv>Export runs CSV</button>
        <button type="button" data-diag-export>Export JSON</button>
        <button type="button" data-diag-copy>Copy JSON</button>
        <button type="button" data-diag-clear>Clear session log</button>
        <button type="button" data-diag-clear-runs>Clear run history</button>
      </div>
      <pre class="diag-log" aria-live="polite"></pre>`;
    document.body.appendChild(dialog);
    dialog.querySelector(".diag-close").addEventListener("click", () => dialog.close());
    dialog.querySelector("[data-diag-copy]").addEventListener("click", async () => {
      const ok = await copy();
      if (!ok) dialog.querySelector(".diag-note").textContent = "Clipboard access failed. Use Export JSON instead.";
    });
    dialog.querySelector("[data-diag-export]").addEventListener("click", download);
    dialog.querySelector("[data-diag-runs-csv]").addEventListener("click", downloadRunsCsv);
    dialog.querySelector("[data-diag-clear]").addEventListener("click", () => { entries.length = 0; render(); });
    dialog.querySelector("[data-diag-clear-runs]").addEventListener("click", clearRuns);
    return dialog;
  }

  function runLine(run) {
    const when = run.startedAt ? new Date(run.startedAt).toLocaleString() : "unknown time";
    const model = run.modelLabel || run.modelKey || "model";
    const performanceBits = run.status === "complete"
      ? `${run.outputTokens ?? "?"} tok · ${run.outputTokensPerSecond ?? "?"} tok/s · prefill ${run.prefillMs ?? "?"} ms`
      : `failed${run.error ? ` · ${run.error}` : ""}`;
    return `${when}  ${model}  ${run.devices || "?"} device${run.devices === 1 ? "" : "s"}  ${performanceBits}`;
  }

  function render() {
    const log = document.querySelector("#fs-diagnostics-dialog .diag-log");
    if (log) {
      log.textContent = entries.slice(-120).map((entry) => {
        const seconds = (entry.ms / 1000).toFixed(1).padStart(7);
        const data = Object.keys(entry.data || {}).length ? `  ${JSON.stringify(entry.data)}` : "";
        return `${seconds}s  ${entry.type}${data}`;
      }).join("\n");
      log.scrollTop = log.scrollHeight;
    }

    const summary = document.querySelector("#fs-diagnostics-dialog .diag-run-summary");
    if (summary) {
      if (!experimentRuns.length) {
        summary.textContent = "No completed runs recorded yet. Run the model once and the useful performance measures will appear here.";
      } else {
        const recent = experimentRuns.slice(-5).reverse().map(runLine).join("\n");
        summary.textContent = `${experimentRuns.length} saved run${experimentRuns.length === 1 ? "" : "s"}\n${recent}`;
      }
    }
  }

  function open() {
    const dialog = ensureDialog();
    render();
    dialog.showModal();
  }

  window.fieldStationDiagnostics = {
    record,
    snapshot,
    copy,
    download,
    downloadRunsCsv,
    experimentRuns: () => [...experimentRuns],
    open,
  };
  window.addEventListener("error", (event) => record("browser:error", {
    name: event.error?.name || "Error",
    error: event.error?.message || event.message || "unknown error",
    filename: event.filename?.split("/").pop() || null,
    line: event.lineno || null,
    column: event.colno || null,
  }));
  window.addEventListener("unhandledrejection", (event) => record("browser:rejection", {
    error: event.reason?.message || String(event.reason || "unknown rejection"),
  }));
  window.addEventListener("online", () => record("network:online"));
  window.addEventListener("offline", () => record("network:offline"));
  document.addEventListener("visibilitychange", () => record("page:visibility", { state: document.visibilityState }));
  document.addEventListener("click", (event) => {
    if (event.target?.closest?.("#diagnostics-btn")) open();
  });
  observeAiStatus();
  record("session:start", { webgpu: !!navigator.gpu, savedRuns: experimentRuns.length });
})();