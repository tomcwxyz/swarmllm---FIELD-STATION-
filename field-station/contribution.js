const TELEMETRY_EVENT = "field-station-telemetry";

export function initialContributionState() {
  return {
    computeMs: 0,
    sentBytes: 0,
    recvBytes: 0,
    transportTokens: 0,
    engineTokens: 0,
    modelBytes: 0,
    layerStart: null,
    layerEnd: null,
    battery: null,
    lastActivity: "connected",
    lastTelemetryAt: 0,
  };
}

export function formatBytes(bytes) {
  const n = Math.max(0, Number(bytes) || 0);
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  const value = n / 1024 ** i;
  return `${value >= 100 || i === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[i]}`;
}

export function formatDuration(ms) {
  const n = Math.max(0, Number(ms) || 0);
  if (n < 1000) return `${Math.round(n)} ms`;
  if (n < 60000) return `${(n / 1000).toFixed(1)} s`;
  const mins = Math.floor(n / 60000);
  const secs = Math.round((n % 60000) / 1000);
  return `${mins}m ${secs}s`;
}

export function reduceContribution(state, detail = {}) {
  const next = { ...state };
  const type = detail.type || "";
  const now = Number(detail.atPerf) || 0;

  if (type === "engine:operation") {
    next.computeMs += Math.max(0, Number(detail.ms) || 0);
    if (Number.isFinite(detail.layerStart) && Number.isFinite(detail.layerEnd)) {
      next.layerStart = detail.layerStart;
      next.layerEnd = detail.layerEnd;
    }
    if (["embedRun", "embedRunBatch", "prefillTokens"].includes(detail.operation) && Number.isFinite(detail.tokens)) {
      next.engineTokens += Math.max(0, detail.tokens);
    }
    next.lastActivity = "working";
  } else if (type === "transport:frame-send") {
    next.sentBytes += Math.max(0, Number(detail.wireBytes) || 0);
    next.lastActivity = "exchanging activations";
  } else if (type === "transport:frame-receive") {
    next.recvBytes += Math.max(0, Number(detail.wireBytes) || 0);
    next.transportTokens += Math.max(0, Number(detail.tokens) || 0);
    next.lastActivity = "exchanging activations";
  } else if (type === "model:resource") {
    const bytes = Number(detail.encodedBytes) || Number(detail.transferBytes) || Number(detail.decodedBytes) || 0;
    next.modelBytes += Math.max(0, bytes);
    next.lastActivity = "loading model";
  } else if (type === "device:battery") {
    next.battery = {
      level: Number.isFinite(detail.level) ? detail.level : null,
      charging: !!detail.charging,
    };
  } else if (type === "engine:operation-error" || type === "engine:spec-error") {
    next.lastActivity = "needs attention";
  } else if (type === "engine:run-start") {
    next.lastActivity = "starting work";
  }

  if (now) next.lastTelemetryAt = now;
  return next;
}

function modelLabel() {
  const select = document.getElementById("ai-model");
  const text = select?.selectedOptions?.[0]?.textContent || "Shared model";
  return text.split("·")[0].trim();
}

function pledgedMemory() {
  const self = document.querySelector(".peer-card.self .buf")?.textContent || "";
  const match = self.match(/(?:gives\s+)?([0-9.]+)\s*GB/i);
  if (match) return `${match[1]} GB`;
  const input = document.getElementById("join-gb");
  return input?.value ? `${input.value} GB` : "—";
}

function layerLabel(state) {
  if (!Number.isFinite(state.layerStart) || !Number.isFinite(state.layerEnd)) return "Waiting for model";
  const last = Math.max(state.layerStart, state.layerEnd - 1);
  return state.layerStart === last ? `Layer ${last}` : `Layers ${state.layerStart}–${last}`;
}

function tokensProcessed(state) {
  const n = state.transportTokens || state.engineTokens;
  return n ? `${Math.round(n).toLocaleString("en-GB")} tokens` : "—";
}

function statusActivity(state) {
  const status = (document.getElementById("ai-status")?.textContent || "").toLowerCase();
  if (/failed|error/.test(status)) return "needs attention";
  if (/generating|prefill|thinking/.test(status)) return "working together";
  if (/loading|downloading|reading|building|testing|tuning|starting|syncing/.test(status)) return "preparing model";
  if (typeof performance !== "undefined" && state.lastTelemetryAt && performance.now() - state.lastTelemetryAt < 1300) return state.lastActivity;
  if (/ready/.test(status)) return "ready";
  return "connected";
}

function devicePath() {
  const names = [...document.querySelectorAll("#peers .pname")]
    .map((el) => (el.textContent || "").replace(/\s*\(you\)\s*/i, "").trim())
    .filter(Boolean);
  if (!names.length) return "This device";
  return names.join("  →  ");
}

function collectiveSummary() {
  return document.getElementById("cluster-summary")?.textContent?.trim() || "Waiting for other devices";
}

function lastAnswerSummary() {
  const status = document.getElementById("ai-status")?.textContent || "";
  const match = status.match(/ready\s*[—-]\s*prefill\s+([0-9.]+)s,\s*(.+)$/i);
  if (!match) return "";
  return `Last answer · prefill ${match[1]}s · ${match[2]}`;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function render(state) {
  const panel = document.getElementById("contribution-panel");
  if (!panel) return;

  setText("contrib-activity", statusActivity(state));
  setText("contrib-model", modelLabel());
  setText("contrib-layers", layerLabel(state));
  setText("contrib-memory", pledgedMemory());
  setText("contrib-compute", state.computeMs ? formatDuration(state.computeMs) : "—");
  setText("contrib-data", state.sentBytes || state.recvBytes ? formatBytes(state.sentBytes + state.recvBytes) : "—");
  setText("contrib-data-detail", state.sentBytes || state.recvBytes ? `↑ ${formatBytes(state.sentBytes)} · ↓ ${formatBytes(state.recvBytes)}` : "activation traffic");
  setText("contrib-tokens", tokensProcessed(state));
  setText("contrib-model-data", state.modelBytes ? formatBytes(state.modelBytes) : "—");
  setText("contrib-together", collectiveSummary());
  setText("contrib-path", devicePath());

  const batteryRow = document.getElementById("contrib-battery-row");
  if (batteryRow) {
    batteryRow.hidden = !state.battery || state.battery.level == null;
    if (!batteryRow.hidden) {
      const pct = Math.round(state.battery.level * 100);
      setText("contrib-battery", `${pct}%${state.battery.charging ? " · charging" : ""}`);
    }
  }

  const last = lastAnswerSummary();
  const lastEl = document.getElementById("contrib-last-answer");
  if (lastEl) {
    lastEl.hidden = !last;
    lastEl.textContent = last;
  }

  panel.dataset.activity = statusActivity(state).replace(/\s+/g, "-");
}

function install() {
  if (typeof window === "undefined" || typeof document === "undefined") return;
  const panel = document.getElementById("contribution-panel");
  if (!panel) return;

  let state = initialContributionState();
  const redraw = () => render(state);

  window.addEventListener(TELEMETRY_EVENT, (event) => {
    state = reduceContribution(state, event.detail || {});
    redraw();
    window.clearTimeout(install.activityTimer);
    install.activityTimer = window.setTimeout(redraw, 1400);
  });

  for (const id of ["ai-model", "join-gb"]) {
    const el = document.getElementById(id);
    el?.addEventListener("change", redraw);
    el?.addEventListener("input", redraw);
  }

  const observer = new MutationObserver(redraw);
  for (const id of ["peers", "cluster-summary", "ai-status"]) {
    const el = document.getElementById(id);
    if (el) observer.observe(el, { childList: true, subtree: true, characterData: true });
  }

  window.__FIELD_STATION_CONTRIBUTION__ = {
    get state() { return { ...state }; },
    render: redraw,
  };
  redraw();
}

install();
