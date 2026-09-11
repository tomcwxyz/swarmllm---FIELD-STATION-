// Model catalogue for the room: URLs, layer counts, memory needs, context length.

export const NEED_GB = {
  "qwen3-0.6b-q4": 0.6, "qwen3-0.6b-q4k": 0.9, "qwen3-0.6b": 0.8,
  "qwen3-1.7b": 2.0, "qwen3-4b": 4.6,
  "llama32-1b-q4": 1.1, "llama32-3b-q4": 2.5, "llama3-8b-q4": 5.2,
  "llama3-70b-q4": 41.5,
  "qwen3.8-27b": 16.5, "smollm-135m": 0.6,
};

export const MODELS = {
  "qwen3-0.6b-q4": { label: "Qwen3 0.6B · Q4 · quick", kind: "gguf",
    gguf: "https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json" },
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

  // Cheap scaled-RoPE verification targets. Both use llama.cpp's exact
  // rope_freqs.weight tensor and native Q4_0 streaming, isolating architecture work
  // from K-quant conversion while keeping download/device requirements modest.
  "llama32-1b-q4": { label: "Llama 3.2 1B · Q4 · experimental", kind: "gguf",
    gguf: "https://huggingface.co/QuantFactory/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct.Q4_0.gguf" },
  "llama32-3b-q4": { label: "Llama 3.2 3B · Q4 · experimental", kind: "gguf",
    gguf: "https://huggingface.co/QuantFactory/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct.Q4_0.gguf" },
  "llama3-8b-q4": { label: "Llama 3 8B · Q4 · experimental", kind: "gguf",
    gguf: "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf" },
  // Original Llama 3 70B deliberately stays on native Q4_0: the dense architecture
  // and unscaled RoPE are already understood, so this isolates the collective-compute
  // question. Its untied output head needs a ~501 MiB storage binding on the host.
  "llama3-70b-q4": { label: "Llama 3 70B · Q4 · collective experiment", kind: "gguf",
    gguf: "https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.Q4_0.gguf",
    hostMinBindGB: 0.55 },

  "qwen3.8-27b": { label: "Qwen 3.8 27B · Q4", kind: "qwen35",
    gguf: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_0.gguf" },
  "smollm-135m": { label: "SmolLM 135M · bf16", kind: "safetensors",
    st: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors",
    cfg: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json",
    tok: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json" },
};

export const MAX_SEQ = 2048;
export const MAX_NEW = 1024;
export const MIN_ROOM = 32;
