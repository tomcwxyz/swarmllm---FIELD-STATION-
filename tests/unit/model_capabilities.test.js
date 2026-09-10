import { assert, assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { GGML_Q6_K } from "../../engine/gguf.js";
import { GGML_Q4_K, Q4_K_BLOCK_BYTES, dequantQ4K, q4KTypeBytes } from "../../engine/q4k.js";
import { inspectGGUFCompatibility } from "../../engine/model-capabilities.js";

Deno.test("Q4_K block size follows ggml block_q4_K", () => {
  assertEquals(q4KTypeBytes(256), Q4_K_BLOCK_BYTES);
  assertEquals(q4KTypeBytes(512), Q4_K_BLOCK_BYTES * 2);
  assertEquals(q4KTypeBytes(32), -1);
});

Deno.test("Q4_K decoder reproduces simple scale/min blocks", () => {
  const block = new Uint8Array(Q4_K_BLOCK_BYTES);
  block[0] = 0x00; block[1] = 0x3c; // f16 1.0 scale
  block[2] = 0x00; block[3] = 0x00; // zero min
  // Eight sub-block scales = 1, mins = 0 in ggml's packed 6-bit layout.
  block.set([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], 4);
  block.fill(0x21, 16); // low nibble 1, high nibble 2

  const out = dequantQ4K(block, 256);
  assertEquals(out.length, 256);
  for (let group = 0; group < 4; group++) {
    const start = group * 64;
    for (let i = 0; i < 32; i++) assertEquals(out[start + i], 1);
    for (let i = 32; i < 64; i++) assertEquals(out[start + i], 2);
  }
});

Deno.test("GGUF preflight separates architecture support from quant support", () => {
  const report = inspectGGUFCompatibility({
    meta: { "general.architecture": "qwen3", "qwen3.block_count": 28 },
    tensors: {
      "blk.0.attn_q.weight": { ggmlType: GGML_Q4_K, nElems: 256, byteLength: -1 },
      "blk.0.ffn_up.weight": { ggmlType: GGML_Q6_K, nElems: 256, byteLength: 210 },
    },
  });

  assertEquals(report.architecture, "qwen3");
  assertEquals(report.adapter, "dense");
  assertEquals(report.layerCount, 28);
  assertEquals(report.status, "integration-needed");
  assert(report.reasons.some((r) => r.includes("Q4_K")));
  assert(report.reasons.some((r) => r.includes("Q6_K")));
  assertEquals(report.totalTensorBytes, 144 + 210);
});

Deno.test("GGUF preflight reports Gemma as architecture work, not a quant failure", () => {
  const report = inspectGGUFCompatibility({
    meta: { "general.architecture": "gemma3", "gemma3.block_count": 26 },
    tensors: {
      "blk.0.attn_q.weight": { ggmlType: GGML_Q4_K, nElems: 256, byteLength: -1 },
    },
  });
  assertEquals(report.adapter, "gemma");
  assertEquals(report.status, "architecture-needed");
  assertEquals(report.architectureStatus, "planned");
});
