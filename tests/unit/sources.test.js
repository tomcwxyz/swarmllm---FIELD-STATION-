import { assert, assertEquals, assertStringIncludes } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { buildSourceContext, chunkSource, queryTerms, selectSourceChunks } from "../../field-station/sources.js";

Deno.test("source chunking keeps names and bounded excerpts", () => {
  const text = Array.from({ length: 30 }, (_, i) => `Paragraph ${i}: apples, pears and community data.`).join("\n\n");
  const chunks = chunkSource({ name: "notes.md", text }, { chunkChars: 220, overlapChars: 30 });
  assert(chunks.length > 3);
  assertEquals(chunks[0].name, "notes.md");
  assert(chunks.every((c) => c.text.length <= 240));
});

Deno.test("retrieval prefers chunks matching the question", () => {
  const chunks = [
    ...chunkSource({ name: "alpha.txt", text: "Cats sleep often.\n\nDogs enjoy walks." }, { chunkChars: 30, overlapChars: 0 }),
    ...chunkSource({ name: "budget.csv", text: "Programme,Cost\nHousing,12000\nTraining,5000" }, { chunkChars: 80, overlapChars: 0 }),
  ];
  const selected = selectSourceChunks(chunks, "What is the housing cost?", { maxChars: 300, maxChunks: 2 });
  assertEquals(selected[0].name, "budget.csv");
  assertStringIncludes(selected[0].text, "Housing");
});

Deno.test("generic document questions still return initial excerpts", () => {
  const chunks = chunkSource({ name: "brief.md", text: "First section explains the project.\n\nSecond section lists risks." }, { chunkChars: 45, overlapChars: 0 });
  const built = buildSourceContext(chunks, "Summarise the attached document", { maxChars: 300, maxChunks: 2 });
  assertEquals(built.labels, ["brief.md"]);
  assert(built.chunks >= 1);
  assertStringIncludes(built.text, "LOCAL SOURCE EXCERPTS");
});

Deno.test("query term extraction removes generic attachment words", () => {
  assertEquals(queryTerms("What does the attached document say about workforce planning?"), ["workforce", "planning"]);
});
