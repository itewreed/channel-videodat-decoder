# channel-videodat-decoder

A decoder for Pro7 VideoDat (1990–1994), a data service broadcast in the
vertical blanking interval (VBI) of the German TV channel Pro7. It decodes
raw VBI captures from bt8x8 cards (bttv `/dev/vbi0`, PAL, 2048 samples per
line, 32 lines per frame) and TBC files from vhs-decode/ld-decode (16 bit,
4·fsc = 17.73 MHz, 1135 samples × 313 lines per field, including sync).

Two transmission variants are supported and detected automatically from the
signal (see [1990 variant](#1990-variant)):

| Variant | Bit rate | Bytes per line | Packets |
|---|---|---|---|
| 1993 | 2 Mbit/s | 10 | sync `56 AE`, scrambled payload, 16-bit checksum |
| 1990 | 800 kbit/s | 4 | SOH packets with escape bytes, 8-bit checksum, several channels on separate lines |

> Program output and the comments in the source code are in German.

## Contents

| Script | Purpose |
|---|---|
| [`videodat_decoder.py`](#videodat_decoderpy) | Main program: VBI/TBC → bit stream → packets → received files and services |
| `videodat1990.py` | Library: packet and file layer of the 1990 variant (used by `videodat_decoder.py`) |
| [`vdb_extract.py`](#vdb_extractpy) | Extract VideoDat databases (`.VDB`/`.IDX`/`.TTL`) and rebuild pictures and screen pages as PNG |
| [`decrypt_services.py`](#decrypt_servicespy) | Decrypt encrypted services (stream 03, e.g. account credit) from `dienst_03_typ*.bin` |
| [`analyse_type05.py`](#analyse_type05py) | Check stream 01 type 05 packets against the known pseudo-random sequence |
| `videodat_crypto.py` | Library: VideoDat DES (DES round function without IP/FP) for services and files |
| `videodat_files.py` | Library: decryption of paid software (`CHVDFILE`, `.VD$`) |
| `videodat_prng.py` | Library: detection of the 15-bit pseudo-random sequence in stream 01 type 05 |

## Requirements

- Python 3.8 or newer
- `numpy` (all scripts), `Pillow` (for `vdb_extract.py` and the PNG output of bit lists)

```sh
python3 -m pip install numpy pillow
```

## Typical workflow

```sh
# 1. decode a VBI capture or TBC file
python3 videodat_decoder.py capture.vbi -o out

# 2. extract a received database (incomplete files work too)
python3 vdb_extract.py out/strom_41/CHIP6.VDB -o chip6
```

---

## videodat_decoder.py

Reads the raw capture, slices the VideoDat lines into bits (NRZ, 2 Mbit/s,
10 bytes per line, each byte with start and stop bit), finds the packets
(sync `56 AE`), descrambles and verifies them, repairs damaged packets from
multiple copies and assembles the transmitted files. It also evaluates the
other services (clock, broadcast schedule, bit lists, encrypted data).
Recordings of the 1990 variant (800 kbit/s, 4 bytes per line) are detected
automatically and decoded with their own packet format, see
[1990 variant](#1990-variant).

```sh
python3 videodat_decoder.py INPUT [options]
```

### Input

| Option | Default | Meaning |
|---|---|---|
| `INPUT` | – | Raw VBI capture (bttv), TBC file (extension `.tbc`) or, with `--bytes`, an already decoded byte stream |
| `--format {auto,vbi,tbc}` | `auto` | Input format. `auto` picks `tbc` for the `.tbc` extension, otherwise `vbi` |
| `--variant {auto,1993,1990}` | `auto` | Transmission variant. `auto` compares how many lines of a sample (start of the recording, up to `--scan-seconds`) are correctly framed with each variant. With `--bytes`, `auto` means `1993`; pass `--variant 1990` for a 1990 channel stream |
| `--bytes` | off | Input is a byte stream, e.g. written by `--save-bytes`. Skips signal processing |
| `--lines LIST` | automatic | Lines carrying VideoDat. VBI: line index in the capture buffer (0–31, 0–15 = first field), e.g. `4,5,6,20,21,22`. TBC: PAL frame lines, e.g. `11,12,13,324,325,326` |
| `--scan-seconds S` | `30` | Search for the data lines over at most this many seconds from the start of the recording. The search stops earlier once the lines are reliably detected |

### VBI captures (bt8x8) only

| Option | Default | Meaning |
|---|---|---|
| `--line-length N` | `2048` | Samples per line in the VBI buffer |
| `--lines-per-frame N` | `32` | Lines per frame in the VBI buffer (half of them per field) |
| `--no-line-tracking` | off | Do not track the lines per field; always decode exactly the detected or given lines |

With bt848 captures the vertical sync is often unstable at the start of a
recording: the data lines jump by a few lines from frame to frame, or the
buffer still holds picture content. The decoder therefore identifies data
lines by their byte framing (start/stop bits) and tracks them per field.
Shifted lines are only used when they are cleanly framed, so no picture
content ends up in the data stream.

### TBC files (vhs-decode / ld-decode) only

| Option | Default | Meaning |
|---|---|---|
| `--tbc-width N` | `1135` | Samples per line (PAL) |
| `--tbc-field-lines N` | `313` | Lines per field (PAL) |
| `--no-sync` | off | Do not use the H-sync as timing reference. By default the distance from H-sync to the data start edge is learned from the file, so lines with a destroyed start edge can still be read |
| `--no-eq` | off | Do not train an equalizer, use the adaptive threshold only. By default a linear equalizer against ringing, ghosting and blur is learned from error-free lines of the recording (1993 variant only; the 1990 signal is decoded with the adaptive threshold) |

The field order is read from the matching `.tbc.json` if present.

### Output

| Option | Default | Meaning |
|---|---|---|
| `-o`, `--outdir DIR` | `videodat1993_out` | Output directory |
| `--save-bytes FILE` | – | Save the decoded byte stream (can be re-analysed later with `--bytes`). 1990: one file per channel, named `FILE_zeilen_<lines>.<ext>` |
| `--partial` | off | Write incomplete files anyway. Missing 255-byte blocks are filled with zero bytes and the file name ends in `.unvollstaendig` ("incomplete") |
| `--keine-dienste` | off | Skip the other services (clock, schedule, bit lists, encrypted data; 1990: text between the packets) |
| `--dump-other` | off | Write raw packets of other services to `andere_dienste/strom_XX_typ_YY.bin` (1993 only) |
| `--decrypt-services` | off | Decrypt stream 03 types 07/08/0A (account credit, limit) and write them with their inner checksum to `dienste/decrypted_services.jsonl` (1993 only) |
| `--decrypt-files` | off | Decrypt paid software (`CHVDFILE`, `.VD$`) with the keys from the long type 01 end packets. Output goes to `decrypted_files/`; gaps stay marked. Combine with `--partial` for incomplete files (1993 only) |

### Output directory

```
out/
  strom_XX/                    received files per data stream
    NAME                       completely received
    NAME.unvollstaendig        only with --partial
    ohne_dateikopf_N.bin...    data for which no file header was received
  dienste/
    dienste.txt                report on clock, schedule and other services
    bitliste_XX.bin / .png     bit lists (e.g. entitlements)
    dienst_XX_typYY.bin        raw data of further services (per packet: 2-byte LE length + payload)
    dienst_01_typ05_analysis.jsonl
    decrypted_services.jsonl   only with --decrypt-services
  andere_dienste/              only with --dump-other
  decrypted_files/             only with --decrypt-files (file_decryption.json + plaintext)
```

For the 1990 variant the files go to one directory per channel address:

```
out/
  kanal_XX/                    received files per channel (e.g. 24, 23)
    NAME                       completely received
    NAME.txt                   readable version of a text file (e.g. DDP.MSG)
    NAME.unvollstaendig        only with --partial
    ohne_dateikopf_N.bin...    only with --partial: blocks without file header
  dienste/
    zwischentext_<lines>.txt   text between the packets (clock, announcements)
```

The console shows a summary: detected lines, number of packets with a valid
checksum, repaired packets, the list of files with their completeness,
clock timestamps, the broadcast schedule and further services.

### Examples

```sh
# bt848 capture, lines detected automatically
python3 videodat_decoder.py capture.vbi -o out

# TBC file from vhs-decode, keep incomplete files
python3 videodat_decoder.py tape.tbc -o out --partial

# fixed lines without per-field tracking
python3 videodat_decoder.py capture.vbi --lines 4,5,6,20,21,22 --no-line-tracking

# save the byte stream and analyse it later without signal processing
python3 videodat_decoder.py tape.tbc --save-bytes tape.bytes
python3 videodat_decoder.py tape.bytes --bytes -o out2 --decrypt-files --partial

# 1990 recording (variant detected automatically; vhs-decode luma file)
python3 videodat_decoder.py tape455.tbcy --format tbc -o out1990

# re-analyse one saved 1990 channel
python3 videodat_decoder.py tape455_zeilen_11-324+12-325.bytes --bytes --variant 1990 -o out3
```

### 1990 variant

The early VideoDat transmissions use a different bit layer and packet
format. Reconstructed from a VHS recording of 5 November 1990
(tape455, Pro7):

- **Bit layer:** NRZ 800 kbit/s instead of 2 Mbit/s, 4 bytes per line
  instead of 10. Each byte has a start and a stop bit and is sent inverted,
  LSB first, as in 1993. In the TBC the data start ~224 samples after the
  start of the line and end after 40 bits (50 µs).
- **Lines and channels:** lines 11/12/13 and 324/325/326 carry data. Each
  line position, or each group of neighbouring positions, carries an
  independent byte stream (channel). The decoder tries every grouping and
  keeps the one that yields the most valid packets. In the 1990 sample:
  lines 11+12 / 324+325 carry channel `24` (files, 8 bytes per field =
  400 bytes/s), line 13 / 326 carries channel `23` (a ddp news ticker,
  otherwise filler bytes `2E`).
- **Packet:** `01` (SOH), address, `0x20 + block number`,
  `0x20 + n`, n data bytes, checksum. Control bytes never appear inside a
  packet: they are sent as `18 xx` with `xx = byte xor 20`. The checksum is
  the sum of the n bytes as sent, mod 256; if that is below `0x20` it is
  sent inverted (`xor FF`).
- **Files:** block number `0x20` with data is the file header
  `[size, 4 bytes big endian, or four spaces][group size][name]`, without
  data it ends the transmission. Data blocks carry the block numbers
  `0x21…0xFF` and then start again at `0x21`; each block holds the next
  ~210 bytes of the file. Blocks are sent in groups of *group size* blocks,
  and each group is repeated (2× for ATEX.ZIP, 4× for DDP.MSG); the last
  group is padded with blocks of `*`. Files run in a carousel, and
  repeated transmissions of the same file are merged.
- **Between the packets:** idle bytes (`FF`) and plain text such as a
  clock (`07:55:50`), a test pattern (`ABC…xyz`) and
  `Videodat-Blockformat..`; they are written to `dienste/zwischentext_*.txt`.
- **Contents of the sample:** `ATEX.ZIP` (50,141 bytes, PKZIP 1.x
  "imploded", contains the HP PCL font file `WELTDEM1.C00`; `unzip -t`
  passes) and `DDP.MSG` news items, XOR `C5` with German umlauts encoded as
  in DIN 66003 (`{|}~` = ä ö ü ß). A readable `.txt` version is written
  next to such files.

Damaged packets are repaired by soft decision over their repeated copies.
A single copy is never "repaired" on suspicion, because the 8-bit checksum
is too weak for that. bttv captures of the 1990 variant are supported in
principle (the bit rate is derived from the sample rate), but no such
capture was available for testing.

---

## vdb_extract.py

Extracts VideoDat databases, such as `CHIP6.VDB` sent in stream 41:
pictures (`VDPC`), screen pages (`VR`), page templates (`VT`), texts,
program entries and accompanying documents. Screen pages are rebuilt as
640×480 PNG images. Incompletely received files are handled too: missing
blocks are detected, several received copies of the same file are merged
byte by byte, and in pictures the position of the data after a gap is
estimated from neighbouring picture lines. Missing picture parts are shown
in grey.

```sh
python3 vdb_extract.py FILE.VDB [more copies.VDB ...] [FILE.IDX] [FILE.TTL] [options]
```

| Option | Default | Meaning |
|---|---|---|
| `dateien` | – | One or more received copies of the same `.VDB` (merged), optionally the `.IDX` (record offsets) and `.TTL` (program catalogue). The file type is detected from its content |
| `-o`, `--outdir DIR` | `<name>_inhalt` | Output directory |
| `--ohne-wiederholung` | off | Do not fill gaps from complete repetitions of the same picture |
| `--luecken FILE` | – | File listing missing ranges (lines `from-to`). Without it, 255-byte zero blocks count as missing (single VDB only) |
| `--nullbloecke` | off | Treat 255-byte zero blocks as missing even if the file name does not end in `.unvollstaendig` (always on with several copies) |
| `--palette {vdi,ega}` | `vdi` | Interpret the colour numbers of page objects as GEM VDI colours or directly as EGA colours |
| `--software DIR` | automatic | Program folder of the receiving software (VD 3.3 or WinVideodat 3.2 with `XFONTS\`, `GMICONS.DAT`, `GM8X16_*.FNT`) for the original fonts and icons. If not given, it is searched next to the VDB and next to the script; without it a TrueType font is used (Liberation/DejaVu/Arial) |
| `--ohne-seiten` | off | Do not rebuild screen pages as PNG |

Output: `inhalt.txt` (overview), `bild_*.png` (pictures), `text_*.txt`,
`seite_*.png` / `vorlage_*.png` (rebuilt pages and templates),
`seite_*.txt`, `datei_*.txt` (accompanying documents), `seiten.html`
(clickable overview), `sendungen.txt` (program list with price and size).

```sh
# merge two incomplete copies, using the original fonts
python3 vdb_extract.py a/CHIP6.VDB.unvollstaendig b/CHIP6.VDB.unvollstaendig \
    --software ~/VD33 -o chip6
```

---

## decrypt_services.py

Decrypts the encrypted services of stream 03 (types 07/08 account credit,
type 0A limit update) from the `dienst_03_typ*.bin` files written by
`videodat_decoder.py`. Each output line is a JSON record with source,
offset, plaintext and the result of the inner checksum. The exit code is 0
if all inner checksums pass, otherwise 1.

```sh
python3 decrypt_services.py out/dienste/dienst_03_typ07.bin -o services.jsonl
```

| Option | Meaning |
|---|---|
| `inputs` | One or more `dienst_03_typ*.bin` files (per packet: 2-byte LE length + payload) |
| `-o`, `--output FILE` | Output file (JSON Lines), required. Must not overwrite an input |

`videodat_decoder.py --decrypt-services` does the same while decoding.

---

## analyse_type05.py

Checks the packets of stream 01 type 05 (`dienst_01_typ05.bin`) for an exact
match with the known 15-bit pseudo-random sequence and reports deviations.
Exact matches are consistent with filler or test data.

```sh
python3 analyse_type05.py out/dienste/dienst_01_typ05.bin -o type05.json --details type05.jsonl
```

| Option | Meaning |
|---|---|
| `input` | `dienst_01_typ05.bin` |
| `-o`, `--output FILE` | JSON summary (SHA-256 of the source, packet count, lengths, classification, anomalies), required |
| `--details FILE` | Optional JSON Lines report with one record per packet |

`videodat_decoder.py` already writes this analysis automatically as
`dienste/dienst_01_typ05_analysis.jsonl`.

---

## Technical background

Details on the bit layer, packet format, descrambling, error correction and
the individual services are in the header comment of `videodat_decoder.py`
(1990 variant: `videodat1990.py`);
the database format is described in the header comment of `vdb_extract.py`
(both in German).

## License

GPL-3.0, see [LICENSE](LICENSE).
