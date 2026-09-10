import {
  deriveJoinProof,
  deriveRoomPeerId,
  normaliseInviteKey,
  normaliseRoomCode,
  randomInviteKey,
} from "./field-station/security.js";

const UPSTREAM_PREFIX = "swarmllm-room-";
const INVITE_KEY_LENGTH = 32;
const $ = (id) => document.getElementById(id);

function parseInvitation() {
  const rawFragment = location.hash.startsWith("#") ? location.hash.slice(1) : location.hash;
  const fragment = new URLSearchParams(rawFragment);
  const compact = fragment.get("invite") || "";
  if (compact) {
    const dot = compact.indexOf(".");
    if (dot > 0) {
      const code = normaliseRoomCode(compact.slice(0, dot));
      const key = normaliseInviteKey(compact.slice(dot + 1));
      if (code && key.length === INVITE_KEY_LENGTH) return { code, key, legacy: false };
    }
  }

  // Backwards compatibility with the first FIELD STATION wrapper. New links no longer
  // expose the short room label as a query parameter or ask people to type it manually.
  const code = normaliseRoomCode(new URLSearchParams(location.search).get("code") || "");
  const key = normaliseInviteKey(fragment.get("key") || "");
  if (code && key.length === INVITE_KEY_LENGTH) return { code, key, legacy: true };
  return null;
}

const invitation = parseInvitation();

function currentKey() {
  return normaliseInviteKey($("key-input")?.value || invitation?.key || "");
}

function currentCode() {
  return normaliseRoomCode($("code-input")?.value || invitation?.code || "");
}

function setStatus(message, tone = "") {
  const el = $("join-status");
  if (!el) return;
  el.textContent = message;
  el.dataset.tone = tone;
}

function createSecureInvite(roomCode, inviteKey) {
  const url = new URL(location.origin + "/room");
  const code = normaliseRoomCode(roomCode);
  const key = normaliseInviteKey(inviteKey);
  // Keep both pieces of the bearer invitation in the fragment. Fragments are not sent in
  // HTTP requests or normal referrers, and people never need to understand either value.
  url.hash = `invite=${code}.${key}`;
  return url.toString();
}

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return true;
  }
  return false;
}

function copyInvite() {
  const code = window.__FIELD_STATION_ROOM__?.code || currentCode();
  const key = window.__FIELD_STATION_ROOM__?.key || currentKey();
  if (!code || key.length !== INVITE_KEY_LENGTH) return;
  const invite = createSecureInvite(code, key);
  copyText(invite)
    .then((ok) => {
      if (ok) window.dispatchEvent(new CustomEvent("field-station-toast", { detail: "invitation link copied" }));
      else window.prompt("Copy this FIELD STATION invitation", invite);
    })
    .catch(() => window.prompt("Copy this FIELD STATION invitation", invite));
}

function configureEntrySurface() {
  const invited = !!invitation;
  document.body.dataset.entry = invited ? "invite" : "create";

  if (invited) {
    $("code-input").value = invitation.code;
    $("key-input").value = invitation.key;
    $("create-btn").hidden = true;
    $("join-btn").hidden = false;
    if ($("entry-type")) $("entry-type").textContent = "INVITATION · COLLECTIVE COMPUTE";
    if ($("entry-title")) $("entry-title").innerHTML = '<span class="lo">You have been invited to</span> join the room.';
    if ($("entry-lede")) $("entry-lede").textContent = "This device will become part of a shared AI runtime with the other devices in the room. Read the notice below, then join.";
    if ($("entry-action-note")) $("entry-action-note").textContent = "By joining, you acknowledge the notice above.";
    setStatus("Protected invitation recognised.", "ok");

    // Normalise old links in-place without a navigation or leaking the secret into history.
    if (invitation.legacy) history.replaceState(null, "", createSecureInvite(invitation.code, invitation.key));
  } else {
    $("code-input").value = "";
    $("key-input").value = "";
    $("create-btn").hidden = false;
    $("join-btn").hidden = true;
    if ($("entry-type")) $("entry-type").textContent = "START · COLLECTIVE COMPUTE";
    if ($("entry-title")) $("entry-title").innerHTML = '<span class="lo">Create a room and</span> invite other devices.';
    if ($("entry-lede")) $("entry-lede").textContent = "A protected invitation link will be created automatically. Share that link with the other people or devices you want to include.";
    if ($("entry-action-note")) $("entry-action-note").textContent = "By creating the room, you acknowledge the notice above.";
    setStatus("");
  }
}

function validateEntry(event, action) {
  // The action button itself is the acknowledgement: its label explicitly says so.
  if ($("safety-check")) $("safety-check").checked = true;
  if (action === "join" && (!currentCode() || currentKey().length !== INVITE_KEY_LENGTH)) {
    event?.preventDefault?.();
    event?.stopImmediatePropagation?.();
    setStatus("This invitation is incomplete. Ask the host to copy a fresh invitation link.", "warn");
    return false;
  }
  return true;
}

function installPreflight() {
  configureEntrySurface();

  $("join-btn")?.addEventListener("click", (event) => validateEntry(event, "join"), true);
  $("create-btn")?.addEventListener("click", (event) => validateEntry(event, "create"), true);

  for (const id of ["room-badge", "side-code"]) {
    $(id)?.addEventListener("click", (event) => {
      const key = window.__FIELD_STATION_ROOM__?.key || currentKey();
      if (key.length !== INVITE_KEY_LENGTH) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      copyInvite();
    }, true);
  }

  window.addEventListener("field-station-toast", (event) => {
    const tray = $("toasts");
    if (!tray) return;
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = event.detail;
    tray.appendChild(toast);
    setTimeout(() => toast.remove(), 3600);
  });
}

function gateConnection(connection, { outgoingProof = null, expectedProof = null, onAuthorised = null } = {}) {
  const originalOn = connection.on.bind(connection);
  const originalSend = connection.send.bind(connection);
  const queuedOpen = [];
  let openSeen = false;
  let authorised = false;
  let timer = null;

  const authorise = () => {
    if (authorised) return;
    authorised = true;
    if (timer) clearTimeout(timer);
    onAuthorised?.();
    for (const handler of queuedOpen.splice(0)) handler();
  };

  // SwarmLLM treats DataConnection.open as the point at which it can start its own protocol.
  // Hold that event until authentication has completed over the encrypted WebRTC data channel.
  connection.on = function fieldStationOn(eventName, handler) {
    if (eventName === "open") {
      if (authorised && openSeen) queueMicrotask(handler);
      else queuedOpen.push(handler);
      return connection;
    }
    return originalOn(eventName, handler);
  };

  originalOn("open", () => {
    openSeen = true;
    timer = setTimeout(() => connection.close(), 8000);
    if (outgoingProof) originalSend({ __fieldStationAuth: 1, proof: outgoingProof });
  });

  originalOn("data", (message) => {
    if (authorised || !message || typeof message !== "object") return;
    if (outgoingProof && message.__fieldStationAuthAck === 1) {
      authorise();
      return;
    }
    if (expectedProof && message.__fieldStationAuth === 1) {
      if (message.proof !== expectedProof) {
        connection.close();
        return;
      }
      originalSend({ __fieldStationAuthAck: 1 });
      authorise();
    }
  });

  return connection;
}

function activeCredentials(hostCredentials = null) {
  if (hostCredentials?.code && hostCredentials?.key) return hostCredentials;
  const code = currentCode();
  const key = currentKey();
  if (!code || key.length !== INVITE_KEY_LENGTH) return null;
  return { code, key, proof: deriveJoinProof(code, key) };
}

function installPeerBoundary() {
  const OriginalPeer = window.Peer;
  if (typeof OriginalPeer !== "function") throw new Error("PeerJS did not load from the local vendor bundle");

  class FieldStationPeer extends OriginalPeer {
    constructor(idOrOptions, maybeOptions) {
      const isExplicitId = typeof idOrOptions === "string";
      const upstreamCode = isExplicitId && idOrOptions.startsWith(UPSTREAM_PREFIX)
        ? normaliseRoomCode(idOrOptions.slice(UPSTREAM_PREFIX.length))
        : "";

      let credentials = null;
      if (upstreamCode) {
        const key = randomInviteKey();
        credentials = {
          code: upstreamCode,
          key,
          peerId: deriveRoomPeerId(upstreamCode, key),
          proof: deriveJoinProof(upstreamCode, key),
        };
        super(credentials.peerId, maybeOptions);
      } else {
        super(idOrOptions, maybeOptions);
      }

      this.__fieldStationCredentials = credentials;
      if (credentials) {
        window.__FIELD_STATION_ROOM__ = credentials;
        $("code-input").value = credentials.code;
        $("key-input").value = credentials.key;
        setStatus("Room created. Copy the protected invitation link to add another device.", "ok");
      }
    }

    connect(target, options = {}) {
      const credentials = activeCredentials(this.__fieldStationCredentials);
      if (typeof target === "string" && target.startsWith(UPSTREAM_PREFIX)) {
        if (!credentials) throw new Error("FIELD STATION invitation missing");
        const code = normaliseRoomCode(target.slice(UPSTREAM_PREFIX.length));
        const key = credentials.key;
        const connection = super.connect(deriveRoomPeerId(code, key), options);
        return gateConnection(connection, { outgoingProof: deriveJoinProof(code, key) });
      }

      const connection = super.connect(target, options);
      // SwarmLLM creates extra chain and stripe links after the initial join. Protect those too.
      return credentials
        ? gateConnection(connection, { outgoingProof: credentials.proof || deriveJoinProof(credentials.code, credentials.key) })
        : connection;
    }

    on(eventName, handler) {
      if (eventName !== "connection") return super.on(eventName, handler);
      return super.on(eventName, (connection) => {
        const credentials = activeCredentials(this.__fieldStationCredentials);
        if (!credentials) {
          connection.close();
          return;
        }
        gateConnection(connection, {
          expectedProof: credentials.proof || deriveJoinProof(credentials.code, credentials.key),
        });
        handler(connection);
      });
    }
  }

  window.Peer = FieldStationPeer;
}

function polishUpstreamCopy() {
  const observer = new MutationObserver(() => {
    const log = $("chat-log");
    if (log) {
      for (const node of log.querySelectorAll("div")) {
        for (const textNode of node.childNodes) {
          if (textNode.nodeType !== Node.TEXT_NODE) continue;
          if (textNode.textContent?.includes("share this code with your other devices")) {
            textNode.textContent = textNode.textContent.replace(
              "share this code with your other devices",
              "copy the invitation link to add another device",
            );
          }
        }
      }
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

installPreflight();
installPeerBoundary();
polishUpstreamCopy();
await import("./room.js");