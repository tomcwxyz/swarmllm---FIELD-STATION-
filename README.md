<p align="center">
  <img src="favicon.svg" width="76" alt="Shared AI mark">
</p>
<h1 align="center">Shared AI</h1>
<p align="center"><b>Can several ordinary devices become one AI computer?</b></p>
<p align="center">A <a href="https://fieldstation.xyz/">FIELD STATION</a> experiment in collective browser inference.</p>

<p align="center">
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/experiments/">Experiments</a> ·
  <a href="roadmap/">Roadmap</a> ·
  <a href="SECURITY.md">Security</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-16171c">
  <img alt="runtime" src="https://img.shields.io/badge/runtime-WebGPU%20%2B%20WebRTC-16171c">
  <img alt="status" src="https://img.shields.io/badge/status-experimental-f06f8b">
</p>

## What is Shared AI?

**Shared AI** is an experiment from [FIELD STATION](https://fieldstation.xyz/), the action lab exploring the future of work, the reach of technology and the shape of society.

The experiment asks a deliberately simple question:

> **What becomes possible when several ordinary devices can become one AI computer?**

A room of laptops, desktops, phones and other WebGPU-capable devices can pool memory and GPU work so that, together, they can run an open language model that may be uncomfortable or impossible for one device alone.

There is no inference server doing the thinking in the middle. Model layers are distributed across participating browsers and intermediate activations move directly between them over WebRTC.

Shared AI began from [SwarmLLM](https://github.com/Nehanth/swarmllm), created by Nehanth Narendrula. SwarmLLM supplied the core distributed-browser inference idea and a substantial part of the engine that remains underneath this project. The experiment has since diverged substantially, adding its own interface, model-compatibility layer, local Sources workflow, diagnostics, safety framing, field protocols and large-model experiments.

The current edge test makes the question literal: **can a room of ordinary browser devices collectively run Llama 3 70B Q4_0 — roughly 40 GB of model — when none of those devices would comfortably run it alone?** See the [70B field protocol](docs/experiments/llama3-70b.md).

## What it can do now

- **Distributed browser inference.** Split model layers across several devices; each participant contributes part of the compute and memory.
- **Run locally on one device too.** The same room can be used as a solo WebGPU runtime for comparison and testing.
- **Use multiple model families.** Qwen models remain available from the SwarmLLM base; Shared AI adds experimental Llama 3 / Llama 3.2 support and a model-adapter layer.
- **Attempt room-scale Llama.** Llama 3 70B Q4_0 is an explicit collective-compute target, with 80 layers distributed by pledged capacity, K/V-cache-aware planning and a capable-host check for the large output head.
- **Bring local Sources.** TXT, Markdown, JSON and CSV files are parsed in the browser. The full file stays on the device that selected it; only retrieved material enters the distributed inference prompt.
- **Understand CSVs as datasets.** CSV Sources are profiled locally for row/column counts, inferred types, missingness, numeric summaries, date ranges and common categorical values, with matching rows retrieved for specific questions.
- **Inspect unfamiliar GGUFs.** Preflight checks model architecture and metadata before committing to a large download or unsupported runtime path.
- **Expose the experiment.** Device, model, loading, context, performance and first-failure diagnostics are recorded so unsuccessful runs are useful evidence rather than silent breakage.

## Experimental model status

The catalogue changes as experiments progress.

| Model | Shared AI status | Purpose |
|---|---|---|
| Qwen3 0.6B / 1.7B / 4B | inherited / working paths | Small dense baselines and compatibility tests |
| Qwen 3.8 27B | inherited SwarmLLM path | Large distributed-compute baseline |
| Llama 3 8B Instruct Q4_0 | experimental, working in field tests | First native Llama target |
| Llama 3.2 1B Instruct Q4_0 | experimental | Cheap scaled-RoPE test target |
| Llama 3.2 3B Instruct Q4_0 | experimental | More useful small-model scaled-RoPE target |
| **Llama 3 70B Instruct Q4_0** | **collective experiment; field run pending** | ~40 GB model / ~41.5 GB room target across multiple devices |
| SmolLM2 135M | inherited | Tiny runtime/demo target |

“Experimental” matters. A model appearing in the picker does not mean deterministic reference testing has been completed across every supported topology. In particular, **70B is engineered for a first attempt, not claimed as proven working**.

### 70B requirements

The first Llama 3 70B route uses the QuantFactory Q4_0 GGUF and deliberately stays on original Llama 3's already-understood dense/unscaled-RoPE architecture.

Shared AI currently asks for approximately **41.5 GB pledged across the room**. One participating desktop/laptop must also be capable of acting as host for the separate Llama output head; the catalogue gate requires at least **0.55 GB per WebGPU storage binding** on that host. The planner accounts for the K/V cache as well as model weights.

The host keeps the quantised token embedding CPU-side for row lookup and streams the separate output head to GPU, avoiding a duplicate GPU copy of the embedding. Very large model ranges are excluded from the browser weight-cache clone path.

See [`docs/experiments/llama3-70b.md`](docs/experiments/llama3-70b.md) for the exact target, topology, milestones and evidence to capture.

## Sources and privacy

Shared AI is **local-first, not confidential**.

When you add a Source, the complete file is read and processed locally in that browser. It is not uploaded to a Shared AI or FIELD STATION server. For a question, the browser builds a compact source context — for example selected text passages or a dataset profile — and only that material is added to the model prompt.

Once source material enters a prompt, participating devices help compute on it. Treat prompts, selected source excerpts and model activations as potentially visible to other room participants. **Do not use Shared AI for sensitive, confidential or personal data.** WebRTC encrypts traffic in transit; it does not make an untrusted participant a private compute environment. See [SECURITY.md](SECURITY.md).

## Quick start

Shared AI is a static browser application with a build step that applies and verifies the experiment's runtime patches.

```bash
git clone https://github.com/tomcwxyz/swarmllm---FIELD-STATION-.git
cd swarmllm---FIELD-STATION-
npm install
npm run serve:field
```

Then open `http://localhost:8080`, create a room and optionally invite more devices with the protected room link.

Useful commands:

```bash
npm run build:field   # build the distributable into dist/
npm run serve:field   # build and serve it locally
npm test              # unit tests (requires Deno 2.x)
npm run test:gpu      # upstream/engine GPU tests
npm run e2e           # browser room test
```

## How collective inference works

```text
host      token / embedding
   ↓
peer A    first slice of transformer layers
   ↓
peer B    next slice
   ↓
peer C    next slice
   ↓
host      final norm → model head → next token
```

Each device downloads and retains only the weights assigned to its layer slice. The host coordinates the generation loop, while WebRTC carries hidden activations between peers.

Shared AI currently builds on SwarmLLM's custom WebGPU/WGSL inference engine rather than wrapping WebLLM, llama.cpp or a remote inference API. Additions made for this experiment include model adapters and chat framing, Llama GGUF rotary-layout/scaling support, local Source processing, room UX, diagnostics, large-model memory planning and experiment-oriented guardrails.

## FIELD STATION

[FIELD STATION](https://fieldstation.xyz/) is the **action lab**, not the software in this repository. Shared AI is one FIELD STATION experiment.

The lab investigates questions around the future of work, technology and society through practical experiments, research and action. This repository is one piece of that wider programme rather than the identity of FIELD STATION itself.

## Relationship to SwarmLLM

This repository would not exist without [SwarmLLM](https://github.com/Nehanth/swarmllm).

In practical terms:

- the **distributed WebGPU inference foundation** is derived from SwarmLLM;
- Shared AI has substantially changed the **experiment, interface and runtime around that foundation**;
- upstream SwarmLLM performance claims, demos and test results are not automatically claims about Shared AI;
- inherited work retains its attribution and licence history.

If work here is generally useful to distributed browser inference, upstreaming it back to SwarmLLM should be considered where practical.

## Repository layout

```text
field-station/            experiment-specific conversation, model-capability and Source logic
field-station-*.js/css    browser shell, diagnostics, model and Sources UI
field-room.html           shared-compute room interface
engine/                   WebGPU inference engine inherited from / evolved from SwarmLLM
room/ + room.js           distributed room runtime, transport and model catalogue
scripts/                  build patches and build-time verification
docs/                     architecture, technical notes and experiment records
roadmap/                  active research and engineering roadmap
tests/                    unit, GPU, reference and end-to-end tests
benchmarks/               engine benchmark harnesses
```

Some internal filenames still use `field-station` because they grew out of the experiment's original implementation. They are implementation details rather than the public name of the tool.

## Current research directions

- deterministic Llama 3.x reference and multi-device equivalence testing;
- the first reproducible **Llama 3 70B multi-device load and reference run**;
- streaming host-side embedding repack and, if needed, global-tensor sharding for lower-limit devices;
- richer local analysis of attached datasets;
- PDF and other local Source types;
- evidence about trade-offs between device count, memory, network latency, energy and useful model size.

See [`roadmap/`](roadmap/) and [`docs/experiments/`](docs/experiments/).

## Contributing

Shared AI benefits particularly from **real hardware evidence**: different browsers, GPUs, phones, networks and model files expose assumptions quickly. Reproducible failures are useful results.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [GOVERNANCE.md](GOVERNANCE.md).

## Upstream credit and acknowledgements

**SwarmLLM** — Nehanth Narendrula  
[github.com/Nehanth/swarmllm](https://github.com/Nehanth/swarmllm)

```bibtex
@software{swarmllm2026,
  author = {Narendrula, Nehanth},
  title  = {SwarmLLM: peer-to-peer LLM inference across browser tabs},
  year   = {2026},
  url    = {https://github.com/Nehanth/swarmllm}
}
```

Shared AI also depends on work from the open model and inference ecosystem, including [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), [Qwen](https://huggingface.co/Qwen), Meta's Llama models, Hugging Face model hosting and PeerJS.

## License

[MIT](LICENSE). Existing upstream attribution and licence history remain part of the repository.
