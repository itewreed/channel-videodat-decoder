# SPDX-License-Identifier: GPL-3.0-or-later
"""CHVDFILE recovery using the long type-01 end-of-transmission key packets.

All offsets are half-open byte ranges. Missing ciphertext is never decrypted
into invented plaintext: the whole affected cipher block remains a marked hole.
"""
import hashlib
import json
import re
from pathlib import Path

from videodat_crypto import SERVICE_KEY_TEMPLATE, crypt_blocks


def unwrap_transmission(payload):
    """Decode type 09 metadata or the 16-byte type 01 key-bearing end packet.

    Key derivation: WINVD 32:0E01..0E41 and 32:1431..1474.
    Type 01 has no independent inner checksum; use the file's key check too.
    """
    p = bytes(payload)
    if not ((len(p) == 16 and p[0] == 1 and p[5:7] == b'\x01K') or
            (len(p) >= 16 and p[0] == 9)):
        raise ValueError("expected a key-bearing type 01 or paid-header type 09")
    wrapping_key = bytearray(SERVICE_KEY_TEMPLATE)
    wrapping_key[2:6] = p[1:5]
    clear = p[:7] + crypt_blocks(p[7:15], wrapping_key, 8) + p[15:]
    result = dict(type=p[0], payload_hex=p.hex(), wrapping_key_hex=wrapping_key.hex())
    if p[0] == 1:
        result.update(kind="file_key", key_hex=clear[7:15].hex(),
                      inner_checksum_checked=False)
    else:
        valid = sum(clear[7:14]) & 255 == clear[14]
        result.update(kind="paid_header", checksum_ok=valid)
        if valid:
            result.update(plaintext_hex=clear.hex(), price_raw=int.from_bytes(clear[7:9], 'little'))
    return result


def parse_file_header(data):
    data = bytes(data)
    if len(data) < 32 or data[:8] != b'CHVDFILE':
        raise ValueError("missing original CHVDFILE header")
    if not 1 <= data[8] <= 16:
        raise ValueError("invalid cipher round count")
    size = int.from_bytes(data[15:19], 'little')
    if len(data) != 32 + ((size + 7) // 8) * 8:
        raise ValueError("file length does not match padded size in header")
    name_field = data[19:32]
    if b'\0' not in name_field:
        raise ValueError("unterminated original filename")
    return dict(rounds=data[8], key_check_byte=data[9], size=size,
                name=name_field.split(b'\0', 1)[0].decode('cp437'))


def merge_ranges(ranges):
    merged = []
    for lo, hi in sorted(ranges):
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    return merged


def affected_plaintext_ranges(ranges, file_size, plaintext_size):
    out = []
    for lo, hi in ranges:
        if not 0 <= lo < hi <= file_size:
            raise ValueError("invalid ciphertext uncertainty range")
        if lo < 32:
            raise ValueError("CHVDFILE header contains missing or conflicting bytes")
        out.append([(lo - 32) // 8 * 8, min(plaintext_size, ((hi - 32 + 7) // 8) * 8)])
    return merge_ranges(out)


def decrypt_file(data, key, missing_ranges=(), conflicting_ranges=()):
    """Return plaintext plus explicit uncertainty; do not claim content integrity."""
    data, key = bytes(data), bytes(key)
    header = parse_file_header(data)
    if len(key) != 8 or (sum(key) + data[8] + data[9]) & 255:
        raise ValueError("file key fails the original eight-bit checksum")
    holes = affected_plaintext_ranges(missing_ranges, len(data), header['size'])
    conflicts = affected_plaintext_ranges(conflicting_ranges, len(data), header['size'])
    # Decrypt surviving intervals in full cipher blocks; trim final padding later.
    body = data[32:]
    plain = bytearray(len(body))
    cursor = 0
    for lo, hi in holes + [[len(body), len(body)]]:
        plain[cursor:lo] = crypt_blocks(body[cursor:lo], key, header['rounds'])
        cursor = min(len(body), ((hi + 7) // 8) * 8)
    plain = bytes(plain[:header['size']])
    return plain, dict(header, missing_plaintext_ranges=holes,
                       conflicting_plaintext_ranges=conflicts,
                       complete_capture=not holes and not conflicts,
                       key_checksum_ok=True, content_integrity_verified=False,
                       sha256=hashlib.sha256(plain).hexdigest())


def _safe_filename(name):
    # Treat the embedded name as untrusted data on every host platform.
    name = re.sub(r'[^A-Za-z0-9_.!()-]', '_', name).strip('. ')
    if not name or name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL'} or \
            re.match(r'^(COM|LPT)[1-9](\.|$)', name, re.I):
        name = 'recovered_' + name
    return name


def write_file_report(packets, manifest, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys, metadata = [], []
    for packet in packets:
        if not packet.ok or not packet.B & 128 or not packet.A:
            continue
        try:
            record = unwrap_transmission(packet.pay)
        except ValueError:
            continue
        record.update(stream=packet.A, position=packet.pos, outer_checksum_checked=True)
        metadata.append(record)
        if record['kind'] == 'file_key':
            keys.append(record)
    reports = []
    counts = dict(decrypted=0, unavailable_key=0, rejected=0)
    for item in manifest:
        if not item['name'].upper().endswith('.VD$'):
            continue
        result = dict(item)
        if not item['path']:
            result.update(status='not_written', reason='Use --partial to retain incomplete files')
            reports.append(result)
            continue
        data = Path(item['path']).read_bytes()
        result['ciphertext_sha256'] = hashlib.sha256(data).hexdigest()
        try:
            header = parse_file_header(data)
            start = min(item['header_positions']) if item['header_positions'] else None
            # Bound the key to this transmission when its start position is known.
            next_start = min((m['position'] for m in metadata if m['kind'] == 'paid_header'
                              and m['stream'] == item['stream'] and start is not None
                              and m['position'] > start), default=float('inf'))
            candidates = {}
            for record in keys:
                key = bytes.fromhex(record['key_hex'])
                if record['stream'] == item['stream'] and \
                        (start is None or start <= record['position'] < next_start) and \
                        (sum(key) + data[8] + data[9]) & 255 == 0:
                    candidates.setdefault(key, []).append(record['position'])
            if len(candidates) != 1:
                counts['unavailable_key'] += 1
                result.update(status='key_unavailable' if not candidates else 'ambiguous_key',
                              matching_keys=len(candidates))
            else:
                key, positions = next(iter(candidates.items()))
                plain, info = decrypt_file(data, key, item['missing_ranges'], item['conflicting_ranges'])
                folder = output_dir / ('stream_%02X' % item['stream'])
                folder.mkdir(exist_ok=True)
                suffix = '.partial' if not info['complete_capture'] else '.decrypted'
                name = _safe_filename(header['name'])
                path = folder / (name + suffix)
                count = 1
                while path.exists():
                    count += 1
                    path = folder / (name + '.%d' % count + suffix)
                path.write_bytes(plain)
                result.update(status='decrypted', key_hex=key.hex(), key_packet_positions=positions,
                              output=str(path), plaintext=info)
                counts['decrypted'] += 1
        except ValueError as error:
            result.update(status='rejected', reason=str(error))
            counts['rejected'] += 1
        reports.append(result)
    report = dict(counts=counts, transmission_packets=metadata, files=reports,
                  note='Key checks are eight-bit sums, not authentication. Verify archive/file integrity independently. '
                  'Missing cipher blocks are zero-filled and explicitly mapped; original bytes are unchanged.')
    (output_dir / 'file_decryption.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return counts
