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

function fragmentKey() {
  const raw = location.hash.startsWith("#") ? location.hash.slice(1) : location.hash;
  return normaliseInviteKey(new URLSearchParams(raw).get("key") || "");
}

function currentKey() {
  return normaliseInviteKey($("key-input")?.value || fragmentKey());
}

function currentCode() {
  return normaliseRoomCode($("code-input")?.value || new URLSearchParams(location.search).get("code") || "");
}

function setStatus(message, tone = "") {
  const el = $("join-status");
  if (!el) return;
  el.textContent = message;
  el.dataset.tone = tone;
}

function createSecureInvite(roomCode, inviteKey) {
  const url = new URL(location.href);
  url.pathname = "/room";
  url.searchParams.set("code", normaliseRoomCode(roomCode));
  url.hash = `key=${normaliseInviteKey(inviteKey)}`;
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
      if (ok) window.dispatchEvent(new CustomEvent("field-station-toast", { detail: "secure invite copied" }));
      else window.prompt("Copy this FIELD STATION invitation", invite);
    })
    .catch(() => window.prompt("Copy this FIELD STATION invitation", invite));
}

function requireConsent(event, action) {
  if (!$("safety-check")?.checked) {
    event?.preventDefault?.();
    event?.stopImmediatePropagation?.();
    setStatus("Confirm that you understand this is a public experiment before continuing.", "warn");
    $("safety-check")?.focus();
    return false;
  }
  if (action === "join" && currentKey().length !== INVITE_KEY_LENGTH) {
    event?.preventDefault?.();
    event?.stopImmediatePropagation?.();
    setStatus("Use the full invitation link, or enter its 32-character invitation key.", "warn");
    $("key-input")?.focus();
    return false;
  }
  return true;
}

function installPreflight() {
  const key = fragmentKey();
  if (key.length === INVITE_KEY_LENGTH) {
    $("key-input").value = key;
    $("key-wrap").hidden = false;
    setStatus("Invitation loaded. This key grants access to the experiment room.", "ok");
  }

  $("join-btn")?.addEventListener("click", (event) => requireConsent(event, "join"), true);
  $("create-btn")?.addEventListener("click", (event) => requireConsent(event, "create"), true);
  $("code-input")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") requireConsent(event, "join");
  }, true);
  $("key-input")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && requireConsent(event, "join")) $("join-btn")?.click();
  });
  $("show-key")?.addEventListener("click", () => {
    $("key-wrap").hidden = false;
    $("key-input")?.focus();
  });
  $("safety-check")?.addEventListener("change", () => {
    if ($("safety-check").checked && $("join-status")?.dataset.tone === "warn") setStatus("");
  });

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
  // We hold that event until this tiny authentication exchange has completed over the encrypted
  // WebRTC data channel, keeping the reusable proof out of the signalling messages.
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
        $("key-input").value = credentials.key;
        $("key-wrap").hidden = false;
        setStatus("Room created with a protected invitation. Share the full invitation link, not just the room code.", "ok");
      }
    }

    connect(target, options = {}) {
      const credentials = activeCredentials(this.__fieldStationCredentials);
      if (typeof target === "string" && target.startsWith(UPSTREAM_PREFIX)) {
        if (!credentials) throw new Error("FIELD STATION invitation key missing");
        const code = normaliseRoomCode(target.slice(UPSTREAM_PREFIX.length));
        const key = credentials.key;
        const connection = super.connect(deriveRoomPeerId(code, key), options);
        return gateConnection(connection, { outgoingProof: deriveJoinProof(code, key) });
      }

      const connection = super.connect(target, options);
      // SwarmLLM creates extra chain and stripe links after the initial join. Protect those too,
      // otherwise possession of a transient worker PeerJS ID would be enough to attach to it.
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
        // Let SwarmLLM attach its listeners now; its queued "open" handler is released only
        // after the FIELD STATION authentication exchange succeeds.
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
        if (node.textContent?.includes("share this code with your other devices")) {
          for (const textNode of node.childNodes) {
            if (textNode.nodeType === Node.TEXT_NODE && textNode.textContent?.includes("share this code with your other devices")) {
              textNode.textContent = textNode.textContent.replace(
                "share this code with your other devices",
                "click the room code to copy the protected invitation link",
              );
            }
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
