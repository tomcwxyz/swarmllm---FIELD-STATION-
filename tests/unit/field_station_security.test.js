import {
  deriveJoinProof,
  deriveRoomPeerId,
  normaliseInviteKey,
  normaliseRoomCode,
  randomInviteKey,
  sha256Hex,
} from "../../field-station/security.js";

function assert(condition, message = "assertion failed") {
  if (!condition) throw new Error(message);
}

Deno.test("FIELD STATION sha256 matches known vector", () => {
  assert(
    sha256Hex("abc") === "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    "sha256 implementation drifted",
  );
});

Deno.test("room identity is deterministic but does not reveal credentials", () => {
  const code = "ABCD";
  const key = "0123456789abcdef0123456789abcdef";
  const id = deriveRoomPeerId(code, key);
  assert(id === deriveRoomPeerId(code, key));
  assert(!id.includes(code));
  assert(!id.includes(key));
  assert(id.startsWith("field-station-room-"));
});

Deno.test("different invitation keys create different network rooms", () => {
  const a = deriveRoomPeerId("ABCD", "0123456789abcdef0123456789abcdef");
  const b = deriveRoomPeerId("ABCD", "1123456789abcdef0123456789abcdef");
  assert(a !== b);
});

Deno.test("join proof is separately domain-separated", () => {
  const code = "ABCD";
  const key = "0123456789abcdef0123456789abcdef";
  const idDigest = deriveRoomPeerId(code, key).replace("field-station-room-", "");
  const proof = deriveJoinProof(code, key);
  assert(proof.length === 64);
  assert(!proof.startsWith(idDigest));
});

Deno.test("credentials are normalised", () => {
  assert(normaliseRoomCode(" ab-cd ") === "ABCD");
  assert(normaliseInviteKey(" 0123-4567-89AB-CDEF-0123-4567-89AB-CDEF ") === "0123456789abcdef0123456789abcdef");
});

Deno.test("invite key is 128 bits encoded as hex", () => {
  assert(/^[a-f0-9]{32}$/.test(randomInviteKey()));
});
