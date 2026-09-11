// Local document retrieval for FIELD STATION Sources.
// Pure functions only: files are read by field-station-sources.js in the browser.

const STOP = new Set(["the","and","for","that","this","with","from","into","about","what","when","where","which","who","how","does","did","are","was","were","have","has","had","can","could","would","should","file","document","source","attached","attachment","please","tell","give","using","use"]);

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
    if (body) chunks.push({ id: `${name}:${index}`, name, index, text: body });
    if (end >= clean.length) break;
    start = Math.max(start + 1, end - overlapChars);
    index++;
  }
  return chunks;
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

export function selectSourceChunks(chunks, query, { maxChars = 2200, maxChunks = 4 } = {}) {
  const ranked = rankSourceChunks(chunks, query);
  const useful = ranked.some((c) => c.score > 0) ? ranked.filter((c) => c.score > 0) : ranked;
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
  const selected = selectSourceChunks(chunks, query, options);
  if (!selected.length) return { text: "", labels: [], chunks: 0 };
  const labels = [...new Set(selected.map((c) => c.name))];
  const excerpts = selected.map((c) => `[Source: ${c.name} · excerpt ${c.index + 1}]\n${c.text}`).join("\n\n");
  return {
    text: `LOCAL SOURCE EXCERPTS\nThese excerpts were selected locally on the asker's device. Use them when relevant. Do not imply you read parts of a file that are not present here.\n\n${excerpts}`,
    labels,
    chunks: selected.length,
  };
}
