# SPDX-License-Identifier: GPL-3.0-or-later
"""Legacy VideoDat cipher: DES round function, no IP/FP, parity removal only.

Recovered from WINVD.EXE segment 36 and DECRYPT.DLL segment 2.
This is an archival compatibility implementation, not modern cryptography.
"""

P = (16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10, 2, 8, 24, 14, 32, 27, 3,
 9, 19, 13, 30, 6, 22, 11, 4, 25)

PC2 = (14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10, 23, 19, 12, 4, 26, 8, 16, 7, 27, 20, 13, 2,
 41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48, 44, 49, 39, 56, 34, 53, 46, 42, 50, 36,
 29, 32)

E = (32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 14, 15, 16, 17, 16,
 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25, 24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32,
 1)

PC1 = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27,
 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52,
 53, 54, 55, 57, 58, 59, 60, 61, 62, 63)

SHIFTS = (1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1)

SBOX = ((14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7, 0, 15, 7, 4, 14, 2, 13, 1, 10,
  6, 12, 11, 9, 5, 3, 8, 4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0, 15, 12,
  8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13),
 (15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10, 3, 13, 4, 7, 15, 2, 8, 14, 12,
  0, 1, 10, 6, 9, 11, 5, 0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15, 13, 8,
  10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9),
 (10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8, 13, 7, 0, 9, 3, 4, 6, 10, 2, 8,
  5, 14, 12, 11, 15, 1, 13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7, 1, 10, 13,
  0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12),
 (7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15, 13, 8, 11, 5, 6, 15, 0, 3, 4, 7,
  2, 12, 1, 10, 14, 9, 10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4, 3, 15, 0,
  6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14),
 (2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9, 14, 11, 2, 12, 4, 7, 13, 1, 5,
  0, 15, 10, 3, 9, 8, 6, 4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14, 11, 8,
  12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3),
 (12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11, 10, 15, 4, 2, 7, 12, 9, 5, 6, 1,
  13, 14, 0, 11, 3, 8, 9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6, 4, 3, 2,
  12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13),
 (4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1, 13, 0, 11, 7, 4, 9, 1, 10, 14,
  3, 5, 12, 2, 15, 8, 6, 1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2, 6, 11,
  13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12),
 (13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7, 1, 15, 13, 8, 10, 3, 7, 4, 12,
  5, 6, 11, 0, 14, 9, 2, 7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8, 2, 1, 14,
  7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11))


def _permute(value, width, table):
    result = 0
    for position in table:
        result = (result << 1) | ((value >> (width - position)) & 1)
    return result


def _subkeys(key, rounds):
    if len(key) != 8:
        raise ValueError("VideoDat keys must contain exactly 8 bytes")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or not 1 <= rounds <= 16:
        raise ValueError("rounds must be an integer from 1 to 16")
    packed = _permute(int.from_bytes(key, "big"), 64, PC1)
    left, right = packed >> 28, packed & 0xFFFFFFF
    keys = []
    for shift in SHIFTS[:rounds]:
        left = ((left << shift) | (left >> (28 - shift))) & 0xFFFFFFF
        right = ((right << shift) | (right >> (28 - shift))) & 0xFFFFFFF
        keys.append(_permute((left << 28) | right, 56, PC2))
    return keys


def crypt_blocks(data, key, rounds=8, *, decrypt=True):
    """Transform complete 8-byte blocks, with no padding or byte-order changes.

    The original omits DES's initial/final permutations and replaces PC-1
    with removal of each key byte's low bit. All other DES tables are standard.
    Independent blocks use the first `rounds` subkeys (reversed for decryption).
    """
    data, key = bytes(data), bytes(key)
    if len(data) % 8:
        raise ValueError("ciphertext/plaintext length must be a multiple of 8")
    keys = _subkeys(key, rounds)
    if decrypt:
        keys.reverse()
    result = bytearray()
    for offset in range(0, len(data), 8):
        block = int.from_bytes(data[offset:offset + 8], "big")
        left, right = block >> 32, block & 0xFFFFFFFF
        for subkey in keys:
            expanded = _permute(right, 32, E) ^ subkey
            substituted = 0
            for i, box in enumerate(SBOX):
                six = (expanded >> (42 - 6 * i)) & 63
                row = ((six >> 4) & 2) | (six & 1)
                column = (six >> 1) & 15
                substituted = (substituted << 4) | box[16 * row + column]
            left, right = right, left ^ _permute(substituted, 32, P)
        result.extend(((right << 32) | left).to_bytes(8, "big"))
    return bytes(result)


# WINVD.EXE expanded automatic data segment 47:292A. Deliberately overlaps
# the Pascal string "Disk Error"; the receiver copies these exact eight bytes.
SERVICE_KEY_TEMPLATE = bytes.fromhex("0a4469736b204572")


def decrypt_service_payload(stream, payload):
    """Decode an already outer-descrambled service payload without mutation.

    Recognized: stream 03 types 07/08 (account credit) and 0A (limit update).
    Returns a JSON-compatible record. A passing additive checksum is evidence
    of consistent decoding, not authentication. Unknown types raise ValueError.
    """
    data = bytes(payload)
    if stream != 3 or not data or data[0] not in (7, 8, 10):
        raise ValueError("unsupported stream/type (expected stream 03, type 07/08/0A)")
    kind = data[0]
    key = bytearray(SERVICE_KEY_TEMPLATE)
    if kind in (7, 8):
        if len(data) < 16 or len(data) & (len(data) - 1):
            raise ValueError("type 07/08 payload must be a power of two, at least 16 bytes")
        key[2:6] = data[1:5]
        clear = data[:6] + crypt_blocks(data[6:-2], key, 8) + data[-2:]
        description = "account_credit"
    else:
        if len(data) != 16:
            raise ValueError("type 0A payload must contain exactly 16 bytes")
        key[2:4] = data[2:4]
        clear = data[:4] + crypt_blocks(data[4:12], key, 8) + data[12:]
        description = "account_limit_update"
    expected = int.from_bytes(clear[-2:], "little")
    actual = sum(clear[:-2]) & 0xFFFF
    record = dict(stream=stream, type=kind, kind=description,
                  length=len(data), checksum_expected=expected,
                  checksum_actual=actual, checksum_ok=expected == actual)
    if expected != actual:
        record["status"] = "inner_checksum_failed"
        return record
    record.update(status="inner_checksum_passed", plaintext_hex=clear.hex())
    if kind in (7, 8):
        # WINVD 27:18FA multiplies the 24-bit field by 256. The following
        # signed word is the number of 256-account groups covered.
        first = int.from_bytes(clear[1:4], "little") * 256
        count = int.from_bytes(clear[4:6], "little", signed=True) * 256
        record.update(first_account=first, account_count=count,
                      last_account=first + count - 1,
                      credit_word_raw=int.from_bytes(clear[6:10], "little"),
                      account_class=clear[10], eligibility_data_hex=clear[11:-2].hex())
    else:
        record.update(sequence=int.from_bytes(clear[2:4], "little"),
                      limit_raw=int.from_bytes(clear[4:8], "little", signed=True),
                      secondary_limit_raw=int.from_bytes(clear[8:10], "little"))
    return record


def write_service_report(packets, output):
    """Write unique, outer-checksum-valid service packets as JSON lines."""
    import collections
    import json
    unique = collections.OrderedDict()
    for packet in packets:
        if (packet.ok and packet.A == 3 and packet.pay
                and packet.pay[0] in (7, 8, 10)):
            unique.setdefault(bytes(packet.pay), []).append(packet.pos)
    counts = dict(passed=0, failed=0, malformed=0)
    with open(output, "w", encoding="utf-8") as handle:
        for payload, positions in unique.items():
            try:
                record = decrypt_service_payload(3, payload)
                counts["passed" if record["checksum_ok"] else "failed"] += 1
            except ValueError as error:
                record = dict(stream=3, type=payload[0], status="malformed",
                              error=str(error))
                counts["malformed"] += 1
            record.update(positions=positions, copies=len(positions),
                          ciphertext_payload_hex=payload.hex())
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return counts
