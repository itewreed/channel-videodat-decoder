#!/usr/bin/env python3
"""Decode dienst_03_typ*.bin dumps produced by videodat_decoder.py."""
import argparse
import json
import struct
from pathlib import Path

from videodat_crypto import decrypt_service_payload


def read_records(path):
    """Read strict LE16-length-prefixed records, retaining file offsets."""
    with open(path, "rb") as handle:
        while True:
            offset = handle.tell()
            header = handle.read(2)
            if not header:
                return
            if len(header) != 2:
                raise ValueError(f"{path}: truncated length at offset {offset}")
            length = struct.unpack("<H", header)[0]
            if not length:
                raise ValueError(f"{path}: zero length at offset {offset}")
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"{path}: truncated payload at offset {offset}")
            yield offset, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in args.inputs}:
        parser.error("output must not overwrite an input")
    # Validate the input framing before opening the output file.
    records = []
    try:
        for path in args.inputs:
            for offset, payload in read_records(path):
                record = decrypt_service_payload(3, payload)
                record.update(source=str(path), offset=offset,
                              outer_checksum_checked=False,
                              ciphertext_payload_hex=payload.hex())
                records.append(record)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    passed = sum(record["checksum_ok"] for record in records)
    print(f"{passed}/{len(records)} inner checksums passed; output: {args.output}")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
