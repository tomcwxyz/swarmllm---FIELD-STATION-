(() => {
  const MAX_ENTRIES = 600;
  const MAX_RUNS = 250;
  const RUN_STORAGE_KEY = "field-station:experiment-runs:v1";
  const PRIVATE_KEY = /^(prompt|promptText|message|messages|reply|answer|response|content|contents|text)$/i;
  const PRIVATE_SUFFIX = /(prompt|message|reply|answer|response)(Text|Content)$/i;

  const entries = [];
  const startedAt = Date.now();
  let experimentRuns = loadRuns();
  let activeRun = null;
  let lastStatus = "";

  function safe(value, depth = 0) {
    if (depth > 3) return "[depth]";
    if (value == null || ["string", "number", "boolean"].includes(typeof value)) return value;
    if (Array.isArray(value)) return value.slice(0, 30).map((item) => safe(item, depth + 1));
    if (typeof value === "object") {
      const out = {};
      for (const [key, val] of Object.entries(value).slice(0, 50)) {
        if (PRIVATE_KEY.test(key) || PRIVATE_SUFFIX.test(key)) continue;
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

  function persistRuns() {
    try { localStorage.setItem(RUN_STORAGE_KEY, JSON.stringify(experimentRuns.slice(-MAX_RUNS))); } catch {}
  }

  function record(type, data = {}) {
    entries.push({ at: new Date().toISOString(), ms: Date.now() - startedAt, type, data: safe(data) });
    if (entries.length > MAX_ENTRIES) entries.splice(0, entries.length - MAX_ENTRIES);
    render();
  }

  function parseNumber(value) {
    const match = String(value || "").match(/-?\d+(?:\.\d+)?/);
    return match ? Number(match[0]) : null;
  }

  function currentModel() {
    const select = document.getElementById("ai-model");
    return {
      key: select?.value || null,
      label: select?.selectedOptions?.[0]?.textContent?.trim() || null,
    };
  }

  function currentWire() {
    const wire = (new URLSearchParams(location.search).get("wire") || "stripe4").toLowerCase();
    const stripes = wire === "off" ? 0 : wire.startsWith("stripe")
      ? Math.max(1, Math.min(8, Number.parseInt(wire.slice(6), 10) || 1))
      : 1;
    return { wire, stripes };
  }

  function clusterSnapshot() {
    const devicesDetail = [...document.querySelectorAll(".peer-card")].map((card) => {
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
    const match = summary.match(/(\d+)\s+devices?\s+·\s+(\d+)\s+WebGPU\s+·\s+([\d.]+)\s+GB pledged/i);
    const rtts = devicesDetail.map((device) => device.rttMs).filter(Number.isFinite).sort((a, b) => a - b);
    const bandwidths = devicesDetail.map((device) => device.bandwidthMbps).filter(Number.isFinite);

    return {
      devices: match ? Number(match[1]) : devicesDetail.length || null,
      webgpuDevices: match ? Number(match[2]) : devicesDetail.filter((device) => device.gpu && !/no WebGPU/i.test(device.gpu)).length,
      pledgedGB: match ? Number(match[3]) : devicesDetail.reduce((sum, device) => sum + (device.contributionGB || 0), 0) || null,
      medianRttMs: rtts.length ? rtts[Math.floor(rtts.length / 2)] : null,
      minBandwidthMbps: bandwidths.length ? Math.min(...bandwidths) : null,
      maxBandwidthMbps: bandwidths.length ? Math.max(...bandwidths) : null,
      devicesDetail,
    };
  }

  function beginRun(promptTokens) {
    const model = currentModel();
    activeRun = {
      id: crypto.randomUUID?.() || `run-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      startedAt: new Date().toISOString(),
      startedPerf: performance.now(),
      promptTokens,
      modelKey: model.key,
      modelLabel: model.label,
      ...currentWire(),
      clusterAtStart: clusterSnapshot(),
      firstOutputObservedMs: null,
      lastObservedOutputTokens: 0,
      lastObservedTokensPerSecond: null,
    };
    record("experiment:run-start", {
      id: activeRun.id,
      modelKey: activeRun.modelKey,
      promptTokens,
      devices: activeRun.clusterAtStart.devices,
      wire: activeRun.wire,
    });
  }

  function saveRun(run) {
    experimentRuns.push(safe(run));
    if (experimentRuns.length > MAX_RUNS) experimentRuns.splice(0, experimentRuns.length - MAX_RUNS);
    persistRuns();
    record("experiment:run", run);
    activeRun = null;
  }

  function completeRun({ prefillSeconds, outputTokens, outputTokensPerSecond, devices }) {
    if (!activeRun) beginRun(null);
    const cluster = clusterSnapshot();
    const prefillMs = Number.isFinite(prefillSeconds) ? Math.round(prefillSeconds * 1000) : null;
    saveRun({
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
      decodeMs: Number.isFinite(outputTokensPerSecond) && outputTokensPerSecond > 0
        ? Math.round((outputTokens / outputTokensPerSecond) * 1000)
        : null,
      outputTokensPerSecond,
      firstOutputObservedMs: activeRun.firstOutputObservedMs,
      totalObservedMs: Math.round(performance.now() - activeRun.startedPerf),
      devices: devices || cluster.devices || activeRun.clusterAtStart.devices,
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
    try {
      await navigator.clipboard.writeText(JSON.stringify(snapshot(), null, 2));
      record("diagnostics:copied", { entries: entries.length, runs: experimentRuns.length });
      return true;
    } catch {
      return false;
    }
  }

  function downloadJson() {
    downloadBlob(JSON.stringify(snapshot(), null, 2), "application/json", `field-station-diagnostics-${timestamp()}.json`);
    record("diagnostics:exported", { entries: entries.length, runs: experimentRuns.length });
  }

  function timestamp() {
    return new Date().toISOString().replaceAll(":", "-");
  }

  function downloadBlob(content, type, filename) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
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
    downloadBlob(rows.join("\n"), "text/csv;charset=utf-8", `field-station-runs-${timestamp()}.csv`);
    record("experiment:csv-exported", { runs: experimentRuns.length });
  }

  function clearRuns() {
    experimentRuns = [];
    persistRuns();
    record("experiment:runs-cleared");
  }

  function ensureDialog() {
    let dialog = document.getElementById("fs-diagnostics-dialog");
    if (dialog) return dialog;
    dialog = document.createElement("dialog");
    dialog.id = "fs-diagnostics-dialog";
    dialog.innerHTML = `
      <div class="diag-head">
        <div><div class="fs-type">APPARATUS · EXPERIMENT LOG</div><h2>What is the room doing?</h2></div>
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
      if (!(await copy())) dialog.querySelector(".diag-note").textContent = "Clipboard access failed. Use Export JSON instead.";
    });
    dialog.querySelector("[data-diag-export]").addEventListener("click", downloadJson);
    dialog.querySelector("[data-diag-runs-csv]").addEventListener("click", downloadRunsCsv);
    dialog.querySelector("[data-diag-clear]").addEventListener("click", () => { entries.length = 0; render(); });
    dialog.querySelector("[data-diag-clear-runs]").addEventListener("click", clearRuns);
    return dialog;
  }

  function runLine(run) {
    const when = run.startedAt ? new Date(run.startedAt).toLocaleString() : "unknown time";
    const model = run.modelLabel || run.modelKey || "model";
    const result = run.status === "complete"
      ? `${run.outputTokens ?? "?"} tok · ${run.outputTokensPerSecond ?? "?"} tok/s · prefill ${run.prefillMs ?? "?"} ms`
      : `failed${run.error ? ` · ${run.error}` : ""}`;
    return `${when}  ${model}  ${run.devices || "?"} device${run.devices === 1 ? "" : "s"}  ${result}`;
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
      const recent = experimentRuns.slice(-5).reverse().map(runLine).join("\n");
      summary.textContent = experimentRuns.length
        ? `${experimentRuns.length} saved run${experimentRuns.length === 1 ? "" : "s"}\n${recent}`
        : "No completed runs recorded yet. Run the model once and the useful performance measures will appear here.";
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
    download: downloadJson,
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