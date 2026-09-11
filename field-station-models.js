import { MODELS, NEED_GB } from "./room/models.js";
import { parseGGUFHeader } from "./engine/gguf.js";
import { inspectGGUFCompatibility } from "./engine/model-capabilities.js";
import { denseConfigFromGGUF } from "./engine/model-adapters.js";

const $ = (id) => document.getElementById(id);
const MIB = 2 ** 20;
const GIB = 2 ** 30;
let inspected = null;

function fmtBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "unknown";
  return bytes >= GIB ? `${(bytes / GIB).toFixed(2)} GB` : `${Math.round(bytes / MIB)} MB`;
}

function filenameFromURL(url) {
  try {
    const name = decodeURIComponent(new URL(url).pathname.split("/").pop() || "custom.gguf");
    return name.replace(/\.gguf$/i, "") || "custom GGUF";
  } catch { return "custom GGUF"; }
}

function normaliseModelURL(raw) {
  const url = new URL(String(raw || "").trim());
  if (url.protocol !== "https:") throw new Error("Use an HTTPS GGUF URL.");
  if (!/\.gguf(?:$|[?#])/i.test(url.href)) throw new Error("This does not look like a .gguf file URL.");
  return url.toString();
}

function isRunnableReport(report) {
  return ["qwen3", "llama"].includes(report?.architecture)
    && (report.status === "supported" || report.status === "supported-with-conversion");
}

async function fetchGGUFHeaderForInspection(url) {
  let size = 12 * MIB;
  for (;;) {
    const response = await fetch(url, { headers: { Range: `bytes=0-${size - 1}` } });
    if (response.status !== 206) {
      const host = new URL(url).host;
      throw new Error(response.ok
        ? `The model host ${host} did not honour HTTP range requests (HTTP ${response.status}). FIELD STATION needs range access so devices fetch only assigned layers.`
        : `Model host ${host} returned HTTP ${response.status}.`);
    }
    const buf = await response.arrayBuffer();
    try {
      return { G: parseGGUFHeader(buf, { skipTokenizer: true }), headerBytesFetched: buf.byteLength };
    } catch (error) {
      if (size >= 256 * MIB) throw new Error(`Could not read the GGUF header: ${error?.message || error}`);
      size *= 2;
    }
  }
}

function statusCopy(report) {
  if (isRunnableReport(report)) {
    if (report.architecture === "llama") {
      const scaled = report.reasons.some((reason) => reason.includes("scaled RoPE"));
      if (scaled) return report.status === "supported"
        ? "COMPATIBLE / Llama 3.x dense adapter + GGUF scaled RoPE"
        : "COMPATIBLE / Llama 3.x + scaled RoPE + quant conversion";
      return report.status === "supported"
        ? "COMPATIBLE / Llama 3 dense adapter"
        : "COMPATIBLE / Llama dense adapter + conversion path";
    }
    return report.status === "supported"
      ? "COMPATIBLE / current Qwen3 dense adapter"
      : "COMPATIBLE / conversion path required";
  }
  if (report.status === "architecture-needed") return `NOT YET / ${report.architecture} adapter is ${report.architectureStatus}`;
  if (report.status === "integration-needed") return "NOT YET / quantisation integration required";
  return "NOT COMPATIBLE / current runtime";
}

function renderReport(result) {
  const reportEl = $("model-report");
  if (!reportEl) return;
  const { report, headerBytesFetched } = result;
  const useable = isRunnableReport(report);
  reportEl.dataset.tone = useable ? "ok" : report.status === "architecture-needed" || report.status === "integration-needed" ? "warn" : "bad";
  const types = report.tensorTypes.map((t) => `${t.name} × ${t.count}${t.status === "supported-with-conversion" ? " → Q8" : ""}`).join(" · ");
  const reasons = report.reasons.length ? `<div>${report.reasons.map(escapeHTML).join(" · ")}</div>` : "";
  reportEl.innerHTML = `
    <div class="model-verdict">${escapeHTML(statusCopy(report))}</div>
    <div>${escapeHTML(report.architecture)} · ${report.layerCount ?? "?"} layers · ${report.tensorCount} tensors</div>
    <div class="model-types">${escapeHTML(types || "tensor types unavailable")}</div>
    <div>GGUF tensors ${escapeHTML(fmtBytes(report.totalTensorBytes))} · estimated runtime weights ${escapeHTML(fmtBytes(report.estimatedRuntimeBytes))}</div>
    <div>inspection read ${escapeHTML(fmtBytes(headerBytesFetched))} of header/index only</div>
    ${reasons}`;
  const use = $("model-use");
  if (use) { use.hidden = !useable; use.disabled = !useable; }
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function showFailure(error) {
  inspected = null;
  const report = $("model-report");
  if (report) {
    report.dataset.tone = "bad";
    report.innerHTML = `<div class="model-verdict">INSPECTION FAILED</div><div>${escapeHTML(error?.message || error)}</div><div>The host must permit browser CORS and HTTP Range requests.</div>`;
  }
  if ($("model-use")) $("model-use").hidden = true;
}

async function inspectModel() {
  const button = $("model-inspect");
  const use = $("model-use");
  try {
    const url = normaliseModelURL($("model-url")?.value);
    button.disabled = true;
    if (use) use.hidden = true;
    if ($("model-report")) $("model-report").innerHTML = '<div class="model-verdict">READING GGUF HEADER…</div><div>No model weights are being loaded yet.</div>';
    const { G, headerBytesFetched } = await fetchGGUFHeaderForInspection(url);
    const report = inspectGGUFCompatibility(G);
    if (isRunnableReport(report)) {
      try { denseConfigFromGGUF(G); }
      catch (error) { report.status = "unsupported"; report.reasons.push(error.message); }
    }
    inspected = { url, G, report, headerBytesFetched };
    renderReport(inspected);
    window.fieldStationDiagnostics?.record("model:preflight", {
      architecture: report.architecture, status: report.status, tensorCount: report.tensorCount,
      layerCount: report.layerCount, tensorTypes: report.tensorTypes.map((t) => t.name), headerBytesFetched,
    });
  } catch (error) {
    showFailure(error);
    window.fieldStationDiagnostics?.record("model:preflight-error", { error: String(error?.message || error) });
  } finally { button.disabled = false; }
}

function useInspectedModel() {
  if (!inspected) return;
  const { url, G, report } = inspected;
  if (!isRunnableReport(report)) return;
  const key = "field-custom-gguf";
  const name = G.meta["general.name"] || filenameFromURL(url);
  MODELS[key] = { label: `${name} · custom GGUF`, kind: "gguf", gguf: url, fieldCustom: true };
  NEED_GB[key] = Math.max(0.6, Math.ceil(((report.estimatedRuntimeBytes * 1.3) / GIB + 0.15) * 10) / 10);
  const select = $("ai-model");
  let option = [...select.options].find((o) => o.value === key);
  if (!option) { option = document.createElement("option"); option.value = key; select.appendChild(option); }
  option.textContent = `${name} · inspected custom GGUF`;
  select.value = key;
  select.dispatchEvent(new Event("change", { bubbles: true }));
  $("model-preflight").hidden = true;
  $("model-preflight-toggle").textContent = "Inspect another GGUF…";
  window.dispatchEvent(new CustomEvent("field-station-toast", { detail: "inspected GGUF selected for this room" }));
}

function addMissingCatalogueOptions() {
  const select = $("ai-model");
  if (!select) return;
  const existing = new Set([...select.options].map((o) => o.value));
  for (const [key, model] of Object.entries(MODELS)) {
    if (existing.has(key)) continue;
    const option = document.createElement("option"); option.value = key; option.textContent = model.label; select.appendChild(option);
  }
}

function installModelPreflight() {
  addMissingCatalogueOptions();
  $("model-preflight-toggle")?.addEventListener("click", () => {
    const panel = $("model-preflight"); panel.hidden = !panel.hidden;
    $("model-preflight-toggle").textContent = panel.hidden ? "Inspect another GGUF…" : "Close model inspection";
    if (!panel.hidden) $("model-url")?.focus();
  });
  $("model-inspect")?.addEventListener("click", inspectModel);
  $("model-use")?.addEventListener("click", useInspectedModel);
  $("model-url")?.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); inspectModel(); } });
}

installModelPreflight();
