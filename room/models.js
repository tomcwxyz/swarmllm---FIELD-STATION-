// Model catalogue for the room: URLs, layer counts, memory needs, context length.

export const NEED_GB = { "qwen3-0.6b-q4": 0.6, "qwen3-0.6b-q4k": 0.9, "qwen3-0.6b": 0.8, "qwen3-1.7b": 2.0, "qwen3-4b": 4.6, "qwen3.8-27b": 16.5, "smollm-135m": 0.6 };

export const MODELS = {
  // Web-first quick start: Q4_0 is natively streamable by the engine and cuts the
  // transfer by roughly a third versus the Q8 build while keeping the same model.
  "qwen3-0.6b-q4": { label: "Qwen3 0.6B · Q4 · quick", kind: "gguf",
    gguf: "https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json" },
  // First FIELD STATION K-quant field test. Q4_K_M is a mixed GGUF recipe; its
  // Q4_K/Q5_K/Q6_K matrices are converted to the existing Q8 GPU representation
  // during load. Keep this explicitly experimental until the real-model golden lands.
  "qwen3-0.6b-q4k": { label: "Qwen3 0.6B · Q4_K_M · experimental", kind: "gguf",
    gguf: "https://huggingface.co/QuantFactory/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B.Q4_K_M.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json" },
  "qwen3-0.6b": { label: "Qwen3 0.6B · Q8 · quality", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json" },
  "qwen3-1.7b": { label: "Qwen3 1.7B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json" },
  "qwen3-4b": { label: "Qwen3 4B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json" },
  "qwen3.8-27b": { label: "Qwen 3.8 27B · Q4", kind: "qwen35",
    gguf: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_0.gguf" },
  "smollm-135m": { label: "SmolLM 135M · bf16", kind: "safetensors",
    st: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors",
    cfg: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json",
    tok: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json" },
};

// Context window per room, in tokens: prompt + answer. Each full-attention layer keeps K and V
// for this many positions (4 KB per position each for the 27B, so 16 MiB per attention layer at
// 2048); the kernels only use it as a stride. Generation stops before the cache would overflow.
export const MAX_SEQ = 2048;
// The upstream 400-token demonstration limit could terminate otherwise healthy answers mid-sentence.
// FIELD STATION allows a substantially fuller response while retaining a hard guard against runaway
// generation on slower distributed models. The runtime reports explicitly when this limit is hit.
export const MAX_NEW = 1024;
export const MIN_ROOM = 32;    // a prompt must leave at least this many tokens for the answer
