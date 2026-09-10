import { cp, mkdir, readFile, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { patchFieldRuntime } from "./patch-field-runtime.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "dist");
const peerPackage = join(root, "node_modules", "peerjs", "package.json");
const peerBundle = join(root, "node_modules", "peerjs", "dist", "peerjs.min.js");

async function copy(path) {
  await cp(join(root, path), join(dist, path), { recursive: true });
}

await rm(dist, { recursive: true, force: true });
await mkdir(join(dist, "vendor"), { recursive: true });
await mkdir(join(dist, "room"), { recursive: true });

const peer = JSON.parse(await readFile(peerPackage, "utf8"));
if (peer.version !== "1.5.4") {
  throw new Error(`Expected PeerJS 1.5.4, found ${peer.version}. Refusing an unreviewed runtime dependency.`);
}

for (const path of [
  "index.html",
  "field-room.html",
  "field-station-system.css",
  "field-station.css",
  "field-station-entry.css",
  "field-station-diagnostics.css",
  "field-station-shell.js",
  "field-station-diagnostics.js",
  "field-station-room.js",
  "room.js",
  "engine",
  "room",
  "field-station",
  "favicon.svg",
  "apple-touch-icon.png",
]) await copy(path);

// FIELD STATION keeps upstream room.js mergeable: small behavioural differences are
// asserted and applied only to the distributable copy.
await patchFieldRuntime(join(dist, "room.js"));

// Make /room a real static route as well as a Vercel rewrite target. This keeps the
// FIELD STATION build portable to any static host and avoids depending on platform routing.
await cp(join(root, "field-room.html"), join(dist, "room", "index.html"));
await cp(peerBundle, join(dist, "vendor", "peerjs.min.js"));
console.log("FIELD STATION static build created in dist/ with PeerJS 1.5.4 vendored locally.");
