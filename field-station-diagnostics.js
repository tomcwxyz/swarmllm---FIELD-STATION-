(() => {
  const MAX_ENTRIES = 600;
  const entries = [];
  const startedAt = Date.now();

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

  function snapshot() {
    return {
      fieldStation: "collective-compute",
      capturedAt: new Date().toISOString(),
      page: location.origin + location.pathname,
      userAgent: navigator.userAgent,
      online: navigator.onLine,
      entries: [...entries],
    };
  }

  async function copy() {
    const text = JSON.stringify(snapshot(), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      record("diagnostics:copied", { entries: entries.length });
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
    record("diagnostics:exported", { entries: entries.length });
  }

  function ensureDialog() {
    let dialog = document.getElementById("fs-diagnostics-dialog");
    if (dialog) return dialog;
    dialog = document.createElement("dialog");
    dialog.id = "fs-diagnostics-dialog";
    dialog.innerHTML = `
      <div class="diag-head">
        <div>
          <div class="fs-type">APPARATUS · LOCAL DIAGNOSTICS</div>
          <h2>What is the room doing?</h2>
        </div>
        <button type="button" class="diag-close" aria-label="Close diagnostics">Close</button>
      </div>
      <p class="diag-note">Kept in this browser only. Prompt and answer text are deliberately excluded.</p>
      <div class="diag-actions">
        <button type="button" data-diag-copy>Copy JSON</button>
        <button type="button" data-diag-export>Export JSON</button>
        <button type="button" data-diag-clear>Clear</button>
      </div>
      <pre class="diag-log" aria-live="polite"></pre>`;
    document.body.appendChild(dialog);
    dialog.querySelector(".diag-close").addEventListener("click", () => dialog.close());
    dialog.querySelector("[data-diag-copy]").addEventListener("click", async () => {
      const ok = await copy();
      if (!ok) dialog.querySelector(".diag-note").textContent = "Clipboard access failed. Use Export JSON instead.";
    });
    dialog.querySelector("[data-diag-export]").addEventListener("click", download);
    dialog.querySelector("[data-diag-clear]").addEventListener("click", () => { entries.length = 0; render(); });
    return dialog;
  }

  function render() {
    const log = document.querySelector("#fs-diagnostics-dialog .diag-log");
    if (!log) return;
    log.textContent = entries.slice(-120).map((entry) => {
      const seconds = (entry.ms / 1000).toFixed(1).padStart(7);
      const data = Object.keys(entry.data || {}).length ? `  ${JSON.stringify(entry.data)}` : "";
      return `${seconds}s  ${entry.type}${data}`;
    }).join("\n");
    log.scrollTop = log.scrollHeight;
  }

  function open() {
    const dialog = ensureDialog();
    render();
    dialog.showModal();
  }

  window.fieldStationDiagnostics = { record, snapshot, copy, download, open };
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
  record("session:start", { webgpu: !!navigator.gpu });
})();
