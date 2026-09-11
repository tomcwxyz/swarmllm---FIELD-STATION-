<p align="center">
  <img src="favicon.svg" width="76" alt="FIELD STATION contour mark">
</p>
<h1 align="center">FIELD STATION</h1>
<p align="center"><b>Collective compute for real-world questions.</b></p>
<p align="center">Pool ordinary devices together to run open language models in the browser.</p>

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

## What is FIELD STATION?

FIELD STATION is an experimental, browser-based tool for **collective AI compute**. A room of laptops, desktops and other WebGPU-capable devices can pool memory and GPU work so that, together, they can run a model that may be uncomfortable or impossible for one device alone.

There is no account and no inference server. Model layers are distributed across participating browsers and intermediate activations move directly between them over WebRTC.

FIELD STATION began as a fork of [SwarmLLM](https://github.com/Nehanth/swarmllm), created by Nehanth Narendrula. SwarmLLM supplied the core idea and a substantial part of the distributed browser inference engine that remains underneath this project.

It is now a substantially separate tool rather than a lightly modified SwarmLLM demo. FIELD STATION has its own interface, room and safety model, model-compatibility layer, local Sources workflow, diagnostics, experiment protocols, build verification and roadmap. We continue to credit SwarmLLM wherever inherited work remains and do not treat upstream benchmark results as FIELD STATION results.

### The question behind it

> What becomes possible when several ordinary devices can become one AI computer?

The aim is not simply to run ever-bigger models. FIELD STATION is a place to test collective compute in realistic settings: different hardware, open models, local files and datasets, imperfect networks, and people trying to answer actual questions.

## What it can do now

- **Distributed browser inference.** Split model layers across several devices; each participant contributes part of the compute and memory.
- **Run locally on one device too.** The same room can be used as a solo WebGPU runtime for comparison and testing.
- **Use multiple model families.** Qwen models remain available from the SwarmLLM base; FIELD STATION adds a model-adapter layer and experimental Llama 3 / Llama 3.2 support.
- **Bring local Sources.** TXT, Markdown, JSON and CSV files are parsed in the browser. The full file stays on the device that selected it; only retrieved material enters the distributed inference prompt.
- **Understand CSVs as datasets.** CSV Sources are profiled locally for row/column counts, inferred types, missingness, numeric summaries, date ranges and common categorical values, with matching rows retrieved for specific questions.
- **Inspect unfamiliar GGUFs.** Preflight checks model architecture and metadata before committing to a large download or unsupported runtime path.
- **Expose the experiment.** FIELD STATION records model, device, loading, context, performance and first-failure diagnostics so unsuccessful runs are useful evidence rather than silent breakage.

## Experimental model status

The catalogue changes as experiments progress. At the moment it includes:

| Model | FIELD STATION status | Purpose |
|---|---|---|
| Qwen3 0.6B / 1.7B / 4B | inherited / working paths | Small dense baselines and compatibility tests |
| Qwen 3.8 27B | inherited SwarmLLM path | Large distributed-compute baseline |
| Llama 3 8B Instruct Q4_0 | experimental, working in field tests | First native Llama target |
| Llama 3.2 1B Instruct Q4_0 | experimental | Cheap scaled-RoPE test target |
| Llama 3.2 3B Instruct Q4_0 | experimental | More useful small-model scaled-RoPE target |
| SmolLM2 135M | inherited | Tiny runtime/demo target |

“Experimental” matters. A model appearing in the picker does not mean we have completed deterministic reference testing across every supported device topology. See the experiment notes and roadmap for current evidence rather than treating the table as a compatibility guarantee.

## Sources and privacy

FIELD STATION is **local-first, not confidential**.

When you add a Source, the complete file is read and processed locally in that browser. It is not uploaded to a FIELD STATION server. For a question, the browser builds a compact source context — for example selected text passages or a dataset profile — and only that material is added to the model prompt.

Once source material enters a prompt, participating devices help compute on it. Treat prompts, selected source excerpts and model activations as potentially visible to other room participants. **Do not use FIELD STATION for sensitive, confidential or personal data.** WebRTC encrypts traffic in transit; it does not make an untrusted participant a private compute environment. See [SECURITY.md](SECURITY.md).

## Quick start

FIELD STATION is a static browser application with a build step that applies and verifies FIELD STATION's runtime patches.

```bash
git clone https://github.com/tomcwxyz/swarmllm---FIELD-STATION-.git
cd swarmllm---FIELD-STATION-
npm install
npm run serve:field
```

Then open `http://localhost:8080`, create a room and optionally invite more devices with the protected room link.

Useful commands:

```bash
npm run build:field   # build the FIELD STATION distributable into dist/
npm run serve:field   # build and serve it locally
npm test              # unit tests (requires Deno 2.x)
npm run test:gpu      # upstream/engine GPU tests
npm run e2e           # browser room test
```

## How collective inference works

At a high level:

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

The important unit sent between devices is the model's hidden activation, not the entire model state. Each device downloads and retains only the weights assigned to its layer slice. The host coordinates the generation loop, while WebRTC carries activations between peers.

FIELD STATION currently builds on SwarmLLM's custom WebGPU/WGSL inference engine rather than wrapping WebLLM, llama.cpp or a remote inference API. FIELD STATION additions include model adapters and chat framing, Llama GGUF rotary-layout/scaling support, local Source processing, room UX, diagnostics and experiment-oriented guardrails.

## Repository layout

```text
field-station/            FIELD STATION conversation, model-capability and Source logic
field-station-*.js/css    FIELD STATION browser shell, diagnostics, model and Sources UI
field-room.html           FIELD STATION room interface
engine/                   WebGPU inference engine inherited from / evolved from SwarmLLM
room/ + room.js           distributed room runtime, transport and model catalogue
scripts/                  FIELD STATION build patches and build-time verification
docs/                     architecture, security/technical notes and experiment records
roadmap/                  active research and engineering roadmap
tests/                    unit, GPU, reference and end-to-end tests
benchmarks/               engine benchmark harnesses
```

A deliberate project convention is that several FIELD STATION changes are applied to the copied runtime during `npm run build:field`. Those patches have exact markers and the build fails when upstream/runtime structure drifts instead of silently producing a partially patched application.

## Relationship to SwarmLLM

This repository would not exist without [SwarmLLM](https://github.com/Nehanth/swarmllm). The project began by asking whether SwarmLLM's distributed browser inference could form the basis of a FIELD STATION experiment, and then diverged as the research questions widened.

In practical terms:

- the **distributed WebGPU inference foundation** is derived from SwarmLLM;
- FIELD STATION has substantially changed the **tool, interface and experimental runtime around that foundation**;
- upstream SwarmLLM performance claims, demos and test results are not automatically claims about FIELD STATION;
- where we modify inherited code, we aim to preserve attribution and make the boundary visible rather than erase the history of the fork.

If work here is generally useful to distributed browser inference, upstreaming it back to SwarmLLM should be considered where practical.

## Current research directions

The active work is less about polishing a conventional chat product and more about learning from the system:

- deterministic Llama 3.x reference and multi-device equivalence testing;
- richer local analysis of attached datasets rather than simply placing rows in context;
- PDF and other local Source types;
- larger Llama experiments, including what has to change when model components themselves exceed a single browser/GPU buffer;
- clearer evidence about the practical trade-offs between device count, memory, network latency, energy and useful model size.

See [`roadmap/`](roadmap/) for the working plan and [`docs/experiments/`](docs/experiments/) for field protocols and results.

## Contributing

FIELD STATION benefits particularly from **real hardware evidence**: different browsers, GPUs, phones, networks and model files expose assumptions quickly. Reproducible failures are useful results.

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

FIELD STATION also depends on work from the open model and inference ecosystem, including [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), [Qwen](https://huggingface.co/Qwen), Meta's Llama models, Hugging Face model hosting, and PeerJS. Prior work referenced by the inherited SwarmLLM engine includes Petals, exo, WebLLM, LlamaWeb, Gated DeltaNet and PipeInfer.

## License

[MIT](LICENSE). Existing upstream attribution and licence history remain part of the repository.
