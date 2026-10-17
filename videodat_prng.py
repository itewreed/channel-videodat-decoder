# SPDX-License-Identifier: GPL-3.0-or-later
"""Recognize the exact 15-bit PRNG output observed in stream 01, type 05.

The state is determined modulo 2**31; bit 31 of the original 32-bit state
cannot affect these outputs. A match proves this byte sequence is generated
by the recurrence, not what the broadcaster intended to use it for.
"""
import struct
from bisect import bisect_left, bisect_right
from functools import lru_cache

MULTIPLIER = 0x015A4E35
MASK = 0x7FFFFFFF
INVERSE = pow(MULTIPLIER, -1, 1 << 31)


@lru_cache(maxsize=1)
def _low_bit_index():
    # Multiplication is bijective modulo 2**31. Sort the 65536 possible
    # low-bit contributions once so each packet needs only two range lookups.
    pairs = sorted(((MULTIPLIER * low) & MASK, low) for low in range(65536))
    return tuple(x[0] for x in pairs), tuple(x[1] for x in pairs)


def _candidate_low_bits(first, second):
    residues, lows = _low_bit_index()
    base = (MULTIPLIER * (first << 16) + 1) & MASK
    start = ((second << 16) - base) & MASK
    end = (start + 65535) & MASK
    intervals = ((start, end),) if start <= end else ((start, MASK), (0, end))
    for lo, hi in intervals:
        yield from lows[bisect_left(residues, lo):bisect_right(residues, hi)]


def generate_type05(first_state, length=128):
    """Generate LE16 words, overwriting byte zero with packet type 05.

    `first_state` is the state used to produce word zero, after the first
    generator update. It is not the state preceding the first draw.
    """
    if not isinstance(first_state, int) or not 0 <= first_state <= MASK:
        raise ValueError("first_state must be an integer in [0, 2**31)")
    if length not in (16, 128):
        raise ValueError("supported type 05 lengths are 16 and 128 bytes")
    state = first_state
    output = bytearray()
    for _ in range(length // 2):
        output.extend((state >> 16).to_bytes(2, "little"))
        state = (MULTIPLIER * state + 1) & MASK
    output[0] = 5
    return bytes(output)


def identify_type05(payload):
    """Require an exact match of every observed byte, not just its distribution.

    Recover the 16 unknown low state bits from complete words 1 and 2,
    then check all remaining words and the surviving high byte of word 0.
    Returns diagnostics only; no speculative plaintext is produced.
    """
    payload = bytes(payload)
    if len(payload) not in (16, 128) or payload[0] != 5:
        raise ValueError("expected a 16- or 128-byte type 05 payload")
    words = struct.unpack("<%dH" % (len(payload) // 2 - 1), payload[2:])
    result = dict(kind="unknown", matches_prng=False, length=len(payload))
    if payload[1] & 0x80 or any(word & 0x8000 for word in words):
        return result
    matches = []
    high = words[0] << 16
    for low in _candidate_low_bits(words[0], words[1]):
        state1 = high | low
        state = (MULTIPLIER * state1 + 1) & MASK
        if state >> 16 != words[1]:
            continue
        first_state = ((state1 - 1) * INVERSE) & MASK
        if first_state >> 24 != payload[1]:
            continue
        for word in words[2:]:
            state = (MULTIPLIER * state + 1) & MASK
            if state >> 16 != word:
                break
        else:
            matches.append((first_state, (MULTIPLIER * state + 1) & MASK))
    if len(matches) != 1:
        return result
    first_state, next_state = matches[0]
    result.update(kind="pseudorandom_sequence", matches_prng=True,
                  multiplier=MULTIPLIER, increment=1, state_modulus=1 << 31,
                  first_state_mod_2_31=first_state,
                  next_state_mod_2_31=next_state,
                  overwritten_low_byte=(first_state >> 16) & 255,
                  observed_bytes_matched=len(payload))
    return result


def analyse_type05(payload):
    """Classify exact matches and conservative corruption candidates.

    Diagnostics never rewrite the input or call an altered packet an exact
    match. Near matches require a unique generator state within four changed
    bytes of a 128-byte payload. The purpose of the broadcast data remains an
    inference, independently of whether its bytes match this generator.
    """
    payload = bytes(payload)
    result = identify_type05(payload)
    if result["matches_prng"]:
        return result
    permutations = []
    for multiplier in (3, 5, 7):
        order = tuple((multiplier * column) % 8 for column in range(8))
        candidate = bytes(payload[base + column]
                          for base in range(0, len(payload), 8) for column in order)
        found = identify_type05(candidate)
        if found["matches_prng"]:
            permutations.append((order, found))
    if len(permutations) == 1:
        order, found = permutations[0]
        result.update(kind="prng_permutation_mismatch", permutation=list(order),
                      candidate_first_state_mod_2_31=found["first_state_mod_2_31"],
                      sum_preserved=True)
        return result
    if len(payload) != 128:
        return result
    words = struct.unpack("<63H", payload[2:])
    seeds = set()
    for i, (first, second) in enumerate(zip(words, words[1:])):
        if first > 32767 or second > 32767:
            continue
        for low in _candidate_low_bits(first, second):
            state = (first << 16) | low
            for _ in range(i + 1):
                state = ((state - 1) * INVERSE) & MASK
            seeds.add(state)
    candidates = []
    for seed in seeds:
        predicted = generate_type05(seed)
        differences = [dict(offset=i, observed=a, predicted=b)
                       for i, (a, b) in enumerate(zip(payload, predicted)) if a != b]
        if 1 <= len(differences) <= 4:
            candidates.append((seed, predicted, differences))
    if len(candidates) == 1:
        seed, predicted, differences = candidates[0]
        result.update(kind="prng_byte_mismatch", candidate_first_state_mod_2_31=seed,
                      differences=differences, observed_sum=sum(payload),
                      predicted_sum=sum(predicted), sum_preserved=sum(payload) == sum(predicted))
    return result
