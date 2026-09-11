import { buildSourceContext, chunkSource, normaliseSourceText } from "./field-station/sources.js";

const $ = (id) => document.getElementById(id);
const files = new Map();
const MAX_FILES = 6;
const MAX_FILE_BYTES = 2 * 1024 * 1024;
const ACCEPTED = new Set(["txt", "md", "markdown", "csv", "json"]);

function toast(message) {
  window.dispatchEvent(new CustomEvent("field-station-toast", { detail: message }));
}

function extension(name) {
  const parts = String(name || "").toLowerCase().split(".");
  return parts.length > 1 ? parts.pop() : "";
}

function displayBytes(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

function render() {
  const list = $("sources-list");
  const count = $("sources-count");
  if (count) count.textContent = files.size ? String(files.size) : "";
  if (!list) return;
  list.innerHTML = "";
  for (const item of files.values()) {
    const row = document.createElement("div");
    row.className = "source-row";
    const meta = document.createElement("div");
    meta.className = "source-row-meta";
    const name = document.createElement("strong");
    name.textContent = item.name;
    const detail = document.createElement("span");
    detail.textContent = `${displayBytes(item.bytes)} · ${item.chunks.length} excerpt${item.chunks.length === 1 ? "" : "s"}`;
    meta.append(name, detail);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "source-remove";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => { files.delete(item.id); render(); });
    row.append(meta, remove);
    list.appendChild(row);
  }
  if (!files.size) {
    const empty = document.createElement("div");
    empty.className = "source-empty";
    empty.textContent = "No local sources added yet.";
    list.appendChild(empty);
  }
}

async function textForFile(file) {
  const raw = await file.text();
  if (extension(file.name) !== "json") return normaliseSourceText(raw);
  try { return JSON.stringify(JSON.parse(raw), null, 2); }
  catch { return normaliseSourceText(raw); }
}

async function addFiles(fileList) {
  for (const file of [...fileList]) {
    if (files.size >= MAX_FILES) { toast(`Sources supports up to ${MAX_FILES} files at once.`); break; }
    const ext = extension(file.name);
    if (!ACCEPTED.has(ext)) { toast(`${file.name}: use TXT, Markdown, CSV or JSON for now.`); continue; }
    if (file.size > MAX_FILE_BYTES) { toast(`${file.name}: keep local source files under 2 MB for this experiment.`); continue; }
    try {
      const text = await textForFile(file);
      const id = `${file.name}:${file.size}:${file.lastModified}`;
      const chunks = chunkSource({ name: file.name, text });
      if (!chunks.length) { toast(`${file.name}: no readable text found.`); continue; }
      files.set(id, { id, name: file.name, bytes: file.size, chunks });
    } catch (error) {
      toast(`${file.name}: could not read file (${error?.message || error}).`);
    }
  }
  render();
}

function allChunks() {
  return [...files.values()].flatMap((file) => file.chunks);
}

function contextFor(query) {
  if (!files.size) return { text: "", labels: [], chunks: 0 };
  return buildSourceContext(allChunks(), query, { maxChars: 2200, maxChunks: 4 });
}

function install() {
  const toggle = $("sources-toggle");
  const panel = $("sources-panel");
  const input = $("sources-input");
  const add = $("sources-add");
  const drop = $("sources-drop");

  toggle?.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    toggle.setAttribute("aria-expanded", String(!panel.hidden));
  });
  add?.addEventListener("click", () => input?.click());
  input?.addEventListener("change", async () => {
    if (input.files?.length) await addFiles(input.files);
    input.value = "";
  });
  for (const eventName of ["dragenter", "dragover"]) {
    drop?.addEventListener(eventName, (event) => { event.preventDefault(); drop.dataset.drag = "1"; });
  }
  for (const eventName of ["dragleave", "drop"]) {
    drop?.addEventListener(eventName, (event) => { event.preventDefault(); delete drop.dataset.drag; });
  }
  drop?.addEventListener("drop", async (event) => {
    if (event.dataTransfer?.files?.length) await addFiles(event.dataTransfer.files);
  });

  render();
  window.fieldStationSources = Object.freeze({
    contextFor,
    count() { return files.size; },
    clear() { files.clear(); render(); },
  });
}

install();
