# FIELD STATION: collective compute

This fork wraps [SwarmLLM](https://github.com/Nehanth/swarmllm) as a FIELD STATION experiment: **can several ordinary devices become one AI computer?**

The inference engine remains upstream SwarmLLM. FIELD STATION adds a thin experiment and safety layer around it so the runtime can continue to track upstream without maintaining a second inference implementation.

## What this fork changes

- A simple FIELD STATION landing page and experiment room.
- A persistent warning that prompts are not confidential to room participants.
- Explicit consent before a browser creates or joins a room.
- The four-character room code is now only a human-friendly label, not the security boundary.
- Each room receives a random 128-bit invitation key.
- Room code + invitation key derive an opaque PeerJS host ID; the raw key is not used as the network ID.
- Every FIELD STATION peer link validates a separate, domain-separated room proof over the encrypted WebRTC data channel before releasing the connection to the SwarmLLM runtime. The reusable proof is not placed in PeerJS signalling metadata.
- Protected invitation links keep the invitation key in the URL fragment (`#key=...`) so the browser does not send it as part of the HTTP request or normal referrer.
- PeerJS 1.5.4 is pinned as a package dependency and copied into the static build. FIELD STATION pages do not execute PeerJS from a runtime CDN.
- Vercel adds basic browser hardening headers and disables camera, microphone and geolocation for the experiment.

This is pragmatic experiment hardening, **not a claim of zero-trust security**. Once a device is admitted, SwarmLLM still coordinates distributed inference between peers.

## Privacy boundary

Treat everything entered into a room as visible to its participants.

WebRTC encrypts peer traffic in transit, but the intermediate model activations move between participating devices. Do not use personal, confidential, client or otherwise sensitive prompts. The chat visibility setting only controls where rendered text is shown; it does not stop devices taking part in computation.

Do not treat the room code itself as a password. Share the full protected invitation link. Anyone who receives that link has the credentials needed to join while the room exists.

## Supply-chain boundary

The FIELD STATION shell vendors and pins PeerJS at build time. Model files are still fetched by the upstream runtime from the model locations defined in `room/models.js`, currently Hugging Face. SwarmLLM does not yet give this fork a cryptographic model-weight manifest, so model-weight integrity remains an upstream/open item.

The model runtime has no shell, arbitrary filesystem or agent tool access. It can use browser capabilities required by the experiment, principally WebGPU, network access, storage/cache and wake lock.

## Run locally

Requirements: a recent Node.js, Deno 2+, and a Chromium-family browser with WebGPU support.

```bash
npm install
```

For the simplest one-machine local test using PeerJS' public signalling service:

```bash
npm run serve:field
```

Then open `http://localhost:8080/room`. `localhost` is treated as a secure browser context, so WebGPU can be available. **Do not assume `http://192.168.x.x` will work on phones or other computers:** WebGPU normally requires HTTPS (or localhost). For a real multi-device session, serve the FIELD STATION build over HTTPS. A Vercel deployment is the simplest first test.

For a FIELD STATION-controlled signalling server, run two terminals:

```bash
# terminal 1
npm run signal

# terminal 2
npm run serve:field
```

On the host machine open:

```text
http://localhost:8080/room?signal=localhost:9000
```

This plain-HTTP signalling setup is useful for development on the same computer. For a multi-device HTTPS session, the signalling server also needs TLS (`wss://`) and must be reachable by every device. An HTTPS deployment cannot connect to an insecure `ws://` LAN signalling endpoint because browsers block mixed content. Until that is configured, use the default PeerJS signalling service; inference data still travels peer-to-peer over WebRTC.

## Recommended FIELD STATION test sequence

1. Start with 2–3 FIELD STATION-controlled devices on the same trusted network.
2. Use only dummy/public prompts.
3. Begin with Qwen3 0.6B to validate WebGPU and networking before trying larger models.
4. Test joining only from the protected invitation link.
5. Confirm a code without the invitation key cannot join.
6. Observe memory, heat, battery and bandwidth on each device.
7. Only then move to a participant-facing session.

## Upstream relationship

SwarmLLM remains the underlying distributed inference project and is MIT licensed. The FIELD STATION-specific files are intentionally isolated (`field-station/`, `field-station-room.js`, `field-room.html`, `field-station.css`) to keep upstream rebases understandable.
