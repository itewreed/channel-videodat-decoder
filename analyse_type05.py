#!/usr/bin/env python3
"""Identify PRNG output and corruption candidates in a dienst_01_typ05.bin dump."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from decrypt_services import read_records
from videodat_prng import analyse_type05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="JSON summary with source hash and counts")
    parser.add_argument("--details", type=Path,
                        help="optional JSONL report, one record per payload")
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.input, args.output, args.details) if p is not None]
    if len(set(paths)) != len(paths):
        parser.error("input and output paths must differ")
    counts, lengths, anomalies = collections.Counter(), collections.Counter(), []
    total = 0
    details = None
    try:
        # Hash in bounded memory; framing is checked by read_records below.
        digest = hashlib.sha256()
        with args.input.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if args.details:
            args.details.parent.mkdir(parents=True, exist_ok=True)
            details = args.details.open("w", encoding="utf-8")
        for offset, payload in read_records(args.input):
            total += 1
            lengths[len(payload)] += 1
            try:
                record = analyse_type05(payload)
            except ValueError as error:
                record = dict(kind="unknown", matches_prng=False, error=str(error))
            record.update(offset=offset, length=len(payload), outer_checksum_checked=False)
            counts[record["kind"]] += 1
            if not record["matches_prng"]:
                anomalies.append(record)
            if details:
                details.write(json.dumps(record) + "\n")
        summary = dict(source=str(args.input), sha256=digest.hexdigest(), packets=total,
                       lengths=dict(lengths), classifications=dict(counts), anomalies=anomalies,
                       interpretation="Exact PRNG matches are consistent with filler/test data; "
                       "intended broadcast purpose is not independently established.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    finally:
        if details:
            details.close()
    print(f"{total} packets: {dict(counts)} -> {args.output}")


if __name__ == "__main__":
    main()
