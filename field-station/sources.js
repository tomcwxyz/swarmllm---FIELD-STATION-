// Local document and dataset retrieval for FIELD STATION Sources.
// Pure functions only: files are read by field-station-sources.js in the browser.

const STOP = new Set(["the","and","for","that","this","with","from","into","about","what","when","where","which","who","how","does","did","are","was","were","have","has","had","can","could","would","should","file","document","source","attached","attachment","please","tell","give","using","use","data","dataset","analyse","analyze","analysis","csv"]);

export function normaliseSourceText(text) {
  return String(text ?? "").replace(/\r\n?/g, "\n").replace(/[\t\f\v]+/g, " ").replace(/\n{4,}/g, "\n\n\n").trim();
}

export function chunkSource({ name = "source", text = "" }, { chunkChars = 900, overlapChars = 120 } = {}) {
  const clean = normaliseSourceText(text);
  if (!clean) return [];
  const chunks = [];
  let start = 0, index = 0;
  while (start < clean.length) {
    let end = Math.min(clean.length, start + chunkChars);
    if (end < clean.length) {
      const floor = Math.max(start + Math.floor(chunkChars * 0.55), end - 180);
      const candidates = [clean.lastIndexOf("\n\n", end), clean.lastIndexOf("\n", end), clean.lastIndexOf(". ", end) + 1];
      const boundary = Math.max(...candidates.filter((n) => n >= floor));
      if (boundary >= floor) end = boundary;
    }
    const body = clean.slice(start, end).trim();
    if (body) chunks.push({ id: `${name}:${index}`, name, index, text: body, kind: "document" });
    if (end >= clean.length) break;
    start = Math.max(start + 1, end - overlapChars);
    index++;
  }
  return chunks;
}

export function parseCSV(text, { maxRows = 20000 } = {}) {
  const raw = String(text ?? "").replace(/^\uFEFF/, "");
  const rows = [];
  let row = [], field = "", quoted = false;
  const pushRow = () => {
    row.push(field); field = "";
    if (row.some((v) => String(v).trim() !== "")) rows.push(row);
    row = [];
  };
  for (let i = 0; i < raw.length && rows.length < maxRows + 1; i++) {
    const ch = raw[i];
    if (quoted) {
      if (ch === '"' && raw[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
      continue;
    }
    if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") pushRow();
    else if (ch !== "\r") field += ch;
  }
  if (field || row.length) pushRow();
  if (!rows.length) return { headers: [], rows: [], truncated: false };

  const seen = new Map();
  const headers = rows[0].map((value, i) => {
    const base = String(value || "").trim() || `column_${i + 1}`;
    const n = (seen.get(base) || 0) + 1; seen.set(base, n);
    return n === 1 ? base : `${base}_${n}`;
  });
  const width = headers.length;
  const body = rows.slice(1).map((r) => Array.from({ length: width }, (_, i) => String(r[i] ?? "").trim()));
  return { headers, rows: body, truncated: rows.length >= maxRows + 1 };
}

function numericValue(value) {
  let s = String(value ?? "").trim();
  if (!s) return null;
  let neg = false;
  if (/^\(.*\)$/.test(s)) { neg = true; s = s.slice(1, -1); }
  s = s.replace(/[£$€,%\s]/g, "").replace(/,/g, "");
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(s)) return null;
  const n = Number(s);
  return Number.isFinite(n) ? (neg ? -n : n) : null;
}

function dateValue(value) {
  const s = String(value ?? "").trim();
  if (!s || !/(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})/.test(s)) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
}

function compactNumber(n) {
  if (!Number.isFinite(n)) return "?";
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}bn`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}m`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(2)}k`;
  return Number.isInteger(n) ? String(n) : String(Number(n.toFixed(3)));
}

function clip(value, n = 56) {
  const s = String(value ?? "").replace(/\s+/g, " ").trim();
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

export function profileCSV({ name = "dataset.csv", text = "" }) {
  const parsed = parseCSV(text);
  const { headers, rows } = parsed;
  const columns = headers.map((header, ci) => {
    const values = rows.map((r) => r[ci] ?? "");
    const present = values.filter((v) => String(v).trim() !== "");
    const missing = rows.length - present.length;
    const nums = present.map(numericValue).filter((v) => v !== null);
    const dates = present.map(dateValue).filter((v) => v !== null);
    const counts = new Map();
    for (const v of present) { const k = clip(v, 80); counts.set(k, (counts.get(k) || 0) + 1); }
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 3);
    const unique = counts.size;

    if (present.length && nums.length / present.length >= 0.85) {
      const sorted = nums.slice().sort((a, b) => a - b);
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      const median = sorted[Math.floor((sorted.length - 1) / 2)];
      return { name: header, kind: "numeric", missing, present: present.length, unique, min: sorted[0], max: sorted.at(-1), mean, median };
    }
    if (present.length && dates.length / present.length >= 0.85) {
      return { name: header, kind: "date", missing, present: present.length, unique, minDate: new Date(Math.min(...dates)).toISOString().slice(0, 10), maxDate: new Date(Math.max(...dates)).toISOString().slice(0, 10) };
    }
    return { name: header, kind: unique <= Math.max(20, present.length * 0.2) ? "categorical" : "text", missing, present: present.length, unique, top };
  });
  return { name, rowCount: rows.length, columnCount: headers.length, headers, rows, columns, truncated: parsed.truncated };
}

export function chunkCSVRows(profile, { rowsPerChunk = 8 } = {}) {
  if (!profile?.headers?.length || !profile?.rows?.length) return [];
  const chunks = [];
  for (let start = 0; start < profile.rows.length; start += rowsPerChunk) {
    const block = profile.rows.slice(start, start + rowsPerChunk);
    const lines = [`Columns: ${profile.headers.join(" | ")}`];
    for (let i = 0; i < block.length; i++) {
      const row = block[i];
      lines.push(`Row ${start + i + 1}: ${row.map((v, ci) => `${profile.headers[ci]}=${clip(v, 72)}`).join(" | ")}`);
    }
    chunks.push({ id: `${profile.name}:rows:${start}`, name: profile.name, index: chunks.length, text: lines.join("\n"), kind: "csv-rows" });
  }
  return chunks;
}

export function formatDatasetProfile(profile, { maxChars = 2200 } = {}) {
  if (!profile?.headers?.length) return "";
  const lines = [
    `[Dataset: ${profile.name}]`,
    `${profile.rowCount}${profile.truncated ? "+" : ""} data rows · ${profile.columnCount} columns`,
    `Columns: ${profile.headers.join(" | ")}`,
  ];
  for (const col of profile.columns) {
    let line = `- ${col.name}: ${col.kind}; ${col.missing} missing`;
    if (col.kind === "numeric") line += `; min ${compactNumber(col.min)}, max ${compactNumber(col.max)}, mean ${compactNumber(col.mean)}, median ${compactNumber(col.median)}`;
    else if (col.kind === "date") line += `; ${col.minDate} to ${col.maxDate}`;
    else {
      line += `; ${col.unique} distinct`;
      if (col.top?.length) line += `; common ${col.top.map(([v, n]) => `${clip(v, 34)} (${n})`).join(", ")}`;
    }
    if (lines.join("\n").length + line.length + 1 > maxChars) { lines.push(`- … ${profile.columns.length - (lines.length - 3)} more columns omitted from compact profile`); break; }
    lines.push(line);
  }
  return lines.join("\n").slice(0, maxChars);
}

export function queryTerms(query) {
  return [...new Set((String(query || "").toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}'’-]{2,}/gu) || [])
    .map((t) => t.replace(/[’']/g, ""))
    .filter((t) => t.length >= 3 && !STOP.has(t)))];
}

function occurrences(haystack, needle) {
  let n = 0, at = 0;
  while ((at = haystack.indexOf(needle, at)) >= 0) { n++; at += needle.length; }
  return n;
}

export function rankSourceChunks(chunks, query) {
  const terms = queryTerms(query);
  return chunks.map((chunk) => {
    const text = chunk.text.toLowerCase();
    const name = chunk.name.toLowerCase();
    let score = 0;
    for (const term of terms) {
      score += Math.min(4, occurrences(text, term));
      if (name.includes(term)) score += 2;
    }
    return { ...chunk, score };
  }).sort((a, b) => b.score - a.score || a.index - b.index || a.name.localeCompare(b.name));
}

export function selectSourceChunks(chunks, query, { maxChars = 2200, maxChunks = 4, requireMatch = false } = {}) {
  const ranked = rankSourceChunks(chunks, query);
  const matched = ranked.filter((c) => c.score > 0);
  const useful = matched.length ? matched : (requireMatch ? [] : ranked);
  const picked = [];
  let used = 0;
  for (const chunk of useful) {
    if (picked.length >= maxChunks) break;
    const cost = chunk.text.length + chunk.name.length + 24;
    if (picked.length && used + cost > maxChars) continue;
    picked.push(chunk);
    used += cost;
    if (used >= maxChars) break;
  }
  return picked;
}

export function buildSourceContext(chunks, query, options = {}) {
  const maxChars = options.maxChars ?? 3000;
  const datasets = Array.isArray(options.datasets) ? options.datasets : [];
  const datasetParts = [];
  let usedChars = 0;
  for (const dataset of datasets) {
    const remaining = Math.max(400, maxChars - usedChars);
    const part = formatDatasetProfile(dataset, { maxChars: Math.min(1800, remaining) });
    if (!part) continue;
    if (datasetParts.length && usedChars + part.length > maxChars * 0.72) break;
    datasetParts.push(part); usedChars += part.length + 2;
  }

  const rowBudget = Math.max(0, maxChars - usedChars - 260);
  const selected = rowBudget > 200
    ? selectSourceChunks(chunks, query, { maxChars: rowBudget, maxChunks: options.maxChunks ?? 3, requireMatch: datasetParts.length > 0 })
    : [];
  if (!selected.length && !datasetParts.length) return { text: "", labels: [], chunks: 0 };

  const labels = [...new Set([...datasets.map((d) => d.name), ...selected.map((c) => c.name)])];
  const excerpts = selected.map((c) => `[Source: ${c.name} · ${c.kind === "csv-rows" ? "matching rows" : `excerpt ${c.index + 1}`}]\n${c.text}`).join("\n\n");
  const datasetText = datasetParts.join("\n\n");
  const body = [datasetText, excerpts].filter(Boolean).join("\n\n");
  return {
    text: `LOCAL SOURCE MATERIAL\nThe user has attached the local source material below and you HAVE received this material for the current question. Answer from it when relevant. For datasets, the profile describes the whole parsed dataset; matching rows are only examples and must not be treated as the whole dataset. Do not claim that no source or data was provided. Do not invent analysis that is not supported by the profile or rows.\n\n${body}`.slice(0, maxChars + 520),
    labels,
    chunks: selected.length + datasetParts.length,
  };
}
