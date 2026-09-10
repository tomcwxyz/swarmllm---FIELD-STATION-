// Q4_K codec helpers kept separate from the main GGUF loader while FIELD STATION
// grows broader quantisation support. The byte layout follows ggml block_q4_K:
// 256 weights -> f16 d + f16 dmin + 12 scale/min bytes + 128 packed nibbles.

export const GGML_Q4_K = 12;
export const QK_K = 256;
export const Q4_K_BLOCK_BYTES = 144;

function f16ToF32(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

function scaleMin(j, scales) {
  if (j < 4) return [scales[j] & 63, scales[j + 4] & 63];
  return [
    (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
    (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
  ];
}

export function q4KTypeBytes(nElems) {
  if (!Number.isInteger(nElems) || nElems < 0 || nElems % QK_K !== 0) return -1;
  return (nElems / QK_K) * Q4_K_BLOCK_BYTES;
}

/**
 * Decode ggml Q4_K blocks into f32. This is the correctness path, intended to
 * feed FIELD STATION's existing streaming requant-to-Q8 path before a native
 * Q4_K WebGPU kernel is added.
 */
export function dequantQ4K(bytes, nElems) {
  if (!(bytes instanceof Uint8Array)) bytes = new Uint8Array(bytes);
  const expected = q4KTypeBytes(nElems);
  if (expected < 0) throw new Error(`Q4_K element count must be a multiple of ${QK_K}`);
  if (bytes.byteLength < expected) throw new Error(`short Q4_K buffer: ${bytes.byteLength}/${expected} bytes`);

  const out = new Float32Array(nElems);
  const blocks = nElems / QK_K;
  for (let b = 0; b < blocks; b++) {
    const base = b * Q4_K_BLOCK_BYTES;
    const d = f16ToF32(bytes[base] | (bytes[base + 1] << 8));
    const dmin = f16ToF32(bytes[base + 2] | (bytes[base + 3] << 8));
    const scales = bytes.subarray(base + 4, base + 16);
    let q = base + 16;
    let y = b * QK_K;
    let is = 0;

    for (let j = 0; j < QK_K; j += 64) {
      const [sc1, m1q] = scaleMin(is, scales);
      const [sc2, m2q] = scaleMin(is + 1, scales);
      const d1 = d * sc1, m1 = dmin * m1q;
      const d2 = d * sc2, m2 = dmin * m2q;
      for (let l = 0; l < 32; l++) out[y++] = d1 * (bytes[q + l] & 0x0f) - m1;
      for (let l = 0; l < 32; l++) out[y++] = d2 * (bytes[q + l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
  return out;
}
