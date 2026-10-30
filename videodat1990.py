#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
videodat1990.py - Paket- und Dateiebene der fruehen Pro7-VideoDat-
Uebertragung (1990). Wird von videodat_decoder.py verwendet
(--variant 1990 bzw. automatisch erkannt).

Durch Analyse einer VHS-Aufnahme vom 05.11.1990 (tape455, Pro7,
"Vagabund") rekonstruiert.

Bitebene (gegenueber 1993):
  NRZ 800 kbit/s statt 2 Mbit/s, pro VBI-Zeile 4 statt 10 Bytes, je Byte
  Startbit + 8 Datenbits (LSB zuerst) + Stopbit, Byte invertiert (wie
  1993). In der TBC liegt die Startflanke ~224 Samples nach Zeilenbeginn,
  die Zeile endet nach 40 Bit = 50 us.
  Belegt waren die Zeilen 11/12/13 und 324/325/326. Jeder Platz bzw. jede
  Gruppe benachbarter Plaetze traegt einen eigenen Kanal (Byte-Strom):
    Zeilen 11+12 / 324+325  Kanal 24: Dateien (z.B. ATEX.ZIP), 8 Bytes je
                            Halbbild = 400 Bytes/s
    Zeile 13 / 326          Kanal 23: ddp-Nachrichtenticker (DDP.MSG),
                            sonst Fuellbytes 2E
  Die Aufteilung der Zeilen auf Kanaele wird aus den Daten bestimmt
  (die Aufteilung mit den meisten gueltigen Paketen gewinnt).

Kanal (Byte-Strom):
  Zwischen den Paketen stehen Fuellbytes (FF, 20, 2E) und Klartext,
  z.B. eine Uhrzeit "07:55:50\\r\\n", ein Testmuster "ABC...xyz\\r\\n" und
  "Videodat-Blockformat..".

Paketformat (Steuerzeichen werden im Paket nie direkt gesendet):

  Off   Inhalt
  0     01           SOH, Paketanfang
  1     A            Kanaladresse (24 Dateien, 23 Nachrichten)
  2     S            0x20 + Blocknummer; 0x20 = Dateikopf bzw. Ende
  3     L            0x20 + Anzahl Datenbytes n (wie gesendet, mit Escapes)
  4     n Bytes      Daten. Escape: 18 xx steht fuer xx xor 20 (fuer die
                     Bytes 00 01 05 07 09 0A 0D 11 13 18 1A 1B beobachtet)
  4+n   P            Summe der n gesendeten Bytes & FF; ist sie kleiner als
                     20, wird sie invertiert gesendet (P = Summe xor FF)

  Blocknummer: Die Datenbloecke einer Datei tragen S = 21..FF, danach
  wieder 21 (also Block k -> S = 21 + k mod 223). Datenblock = naechste
  bis zu ~210 Bytes der Datei (gesendet mit Escapes hoechstens 219 Bytes).
  Dateikopf (S = 20, n > 0):
     [Groesse 4, big endian, "    " = unbekannt][G][Name]
     G = Gruppengroesse: Die Bloecke werden in Gruppen zu G gesendet und
     jede Gruppe mehrfach (ATEX.ZIP: G = 12, 2x; DDP.MSG: G = 2, 4x).
     Der Kopf selbst wird mehrfach direkt hintereinander gesendet.
  Ende (S = 20, n = 0): Ende der Uebertragung. Danach folgen Fuellbytes;
     die letzte Gruppe wird mit Bloecken aus '*' (2A) aufgefuellt.
  Die Dateien laufen im Karussell: ATEX.ZIP wurde in der Aufnahme
  fuenfmal gesendet (ca. alle 4:50 min).

Inhalt:
  ATEX.ZIP     ZIP-Archiv (PKZIP 1.x, "imploded") mit WELTDEM1.C00
  DDP.MSG      ddp-Meldungen, Bytes xor C5, Umlaute nach DIN 66003
               ({|}~[\\] = aeoeuess AeOeUe), Zeilenende CR CR LF
"""

import collections
import io
import os
import re
import struct
import zipfile

import numpy as np

SOH = 0x01
ESC = 0x18
HEADER = 0x20          # S-Byte des Dateikopfs / Endepakets
NSEQ = 223             # Blocknummern 21..FF
FILL = 0x2A            # Fuellblock '*'


def unescape(raw):
    out = bytearray()
    i = 0
    while i < len(raw):
        if raw[i] == ESC and i + 1 < len(raw):
            out.append(raw[i + 1] ^ 0x20)
            i += 2
        else:
            out.append(raw[i])
            i += 1
    return bytes(out)


def check_ok(data_raw, ck):
    s = sum(data_raw) & 0xFF
    return ck == s or (s < 0x20 and ck == s ^ 0xFF)


class Pkt90:
    __slots__ = ("pos", "A", "S", "raw", "data", "ok", "soft", "how")

    def length(self):
        return len(self.raw)


def _make(stream, soft, p):
    """Paket ab SOH-Position p oder None, wenn es nicht hineinpasst."""
    if p + 5 > len(stream):
        return None
    A, S, Lb = stream[p + 1], stream[p + 2], stream[p + 3]
    n = Lb - 0x20
    if n < 0 or A < 0x20 or S < 0x20 or p + 5 + n > len(stream):
        return None
    x = Pkt90()
    x.pos = p
    x.raw = bytes(stream[p:p + 5 + n])
    x.A, x.S = A, S
    x.ok = check_ok(x.raw[4:4 + n], x.raw[4 + n])
    x.data = unescape(x.raw[4:4 + n])
    x.soft = soft[8 * p:8 * (p + 5 + n)] if soft is not None else None
    x.how = "ok" if x.ok else ""
    return x


def find_packets(stream):
    """stream: Stream (mit Bit-Sicherheiten, siehe videodat_decoder) oder
    bytes. Gueltige Pakete werden uebersprungen, defekte bleiben fuer
    repair() in der Liste."""
    soft = None
    if hasattr(stream, "soft"):
        soft, stream = stream.soft, stream.data
    pk = []
    p = stream.find(SOH)
    while p >= 0:
        x = _make(stream, soft, p)
        nxt = p + 1
        if x is not None:
            pk.append(x)
            if x.ok:
                nxt = p + len(x.raw)
        p = stream.find(SOH, nxt)
    return pk


def _soft_raw(soft):
    b = (soft.reshape(-1, 8) >= 0).astype(np.int64)
    return bytes(int(v) ^ 0xFF for v in b @ (1 << np.arange(8)))


def repair(pk, window=8000):
    """Defekte Pakete per Soft-Decision ueber Kopien reparieren.

    Kopien = Pakete mit gleichem Kopf (A, S, L) innerhalb von window Bytes
    (jede Gruppe wird mehrfach direkt hintereinander gesendet). Gibt es dort
    schon eine gueltige Kopie, ist nichts zu tun. Sonst werden die
    Sicherheiten aller Kopien addiert und neu entschieden. Eine einzelne
    Kopie wird nicht auf Verdacht repariert: die 8-Bit-Summe ist dafuer zu
    schwach. Rueckgabe: Anzahl reparierter Pakete."""
    groups = collections.defaultdict(list)
    for x in pk:
        groups[x.raw[1:4]].append(x)
    fixed = 0
    for xs in groups.values():
        for x in xs:
            if x.ok:
                continue
            near = [y for y in xs if abs(y.pos - x.pos) <= window]
            if any(y.ok for y in near):
                continue
            cop = [y for y in near if y.soft is not None and
                   len(y.soft) == 8 * len(y.raw) and len(y.raw) == len(x.raw)]
            if len(cop) < 2:
                continue
            raw = bytearray(_soft_raw(np.sum([y.soft for y in cop], axis=0)))
            raw[0:4] = x.raw[0:4]
            n = len(raw) - 5
            if not check_ok(raw[4:4 + n], raw[4 + n]):
                continue
            data = unescape(bytes(raw[4:4 + n]))
            for y in cop:
                y.raw, y.data, y.ok, y.how = bytes(raw), data, True, "soft"
                fixed += 1
    return fixed


def parse_header(data):
    """Dateikopf -> (Name, Groesse oder None, Gruppengroesse) oder None."""
    if len(data) < 6:
        return None
    name = data[5:].split(b"\0")[0]
    name = re.sub(rb"[^\x21-\x7e]", b"", name).decode("ascii")
    if not name:
        return None
    size = None if data[:4] == b"    " else struct.unpack(">I", data[:4])[0]
    return name, size, data[4]


def _safe(name):
    n = re.sub(r'[\\/:*?"<>|]', "_", name).strip(" .")
    return n or "unbenannt"


# DIN 66003 (deutscher 7-Bit-Zeichensatz)
DIN66003 = str.maketrans({"[": "Ä", "\\": "Ö", "]": "Ü", "{": "ä",
                          "|": "ö", "}": "ü", "~": "ß", "@": "§"})


TEXT_OK = set(range(32, 127)) | {1, 2, 3, 4, 9, 10, 13}
# Gewichte fuer die Schluesselsuche: Leerzeichen und Kleinbuchstaben
# kennzeichnen Text am deutlichsten
TEXT_W = dict([(c, 1.0) for c in b"abcdefghijklmnopqrstuvwxyz"] +
              [(c, 0.5) for c in b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"] +
              [(0x20, 3.0), (0x0D, 1.0), (0x0A, 1.0)])


def text_view(data):
    """Klartext einer Nachrichtendatei oder None. Probiert alle 256
    Schluessel (Byte xor k) und nimmt den mit den meisten Buchstaben;
    1990: C5 fuer DDP.MSG."""
    if len(data) < 16:
        return None
    cnt = collections.Counter(data)
    k = max(range(256), key=lambda k: sum(n * TEXT_W.get(b ^ k, 0)
                                          for b, n in cnt.items()))
    t = bytes(b ^ k for b in data)
    good = sum(c in TEXT_OK for c in t)
    if good < 0.95 * len(t):
        return None
    s = t.decode("ascii", "replace").translate(DIN66003)
    s = s.replace("\r\r\n", "\n").replace("\r\n", "\n")
    s = re.sub(r"[\x01-\x04]", "", s)
    return k, s


def check_zip(data):
    """CRC-Pruefung eines ZIP-Archivs: (Namen, Fehler) oder None.
    'imploded' (PKZIP 1.x) kann Python nicht entpacken; dann wird nur das
    Verzeichnis geprueft."""
    try:
        z = zipfile.ZipFile(io.BytesIO(bytes(data)))
    except zipfile.BadZipFile:
        return None
    names = z.namelist()
    try:
        bad = z.testzip()
        err = "CRC falsch in %s" % bad if bad else None
    except NotImplementedError:
        err = "Verzeichnis ok; Kompression nicht pruefbar (unzip -t)"
    except Exception as e:      # defektes Archiv
        err = str(e)
    return names, err


def _unwrap(last, s):
    """Blocknummer s (0..222) relativ zum zuletzt gesehenen Block last
    entfalten: kleine Ruecksprunge (Gruppenwiederholung), sonst vorwaerts."""
    if last is None:
        return s
    d = (s - last) % NSEQ
    if d > NSEQ - 40:
        d -= NSEQ
    return last + d


def _block_data(counter):
    return counter.most_common(1)[0][0]


def _is_fill(b):
    return b.strip(bytes([FILL])) == b""


def assemble(pk, outdir, keep_partial, log):
    """Dateien aus den gueltigen Paketen eines Kanals zusammensetzen.

    Mehrere Aussendungen derselben Datei (Name und Groesse gleich) werden
    vereint (Mehrheitsentscheid je Block). Dateien ohne Groessenangabe
    (Nachrichten) aendern sich von Aussendung zu Aussendung und werden
    einzeln geschrieben; ihr Ende ist das Endepaket bzw. der erste
    Fuellblock.
    Rueckgabe: Liste (A, Name, Groesse, fehlende Bloecke, Blockzahl, Pfad,
    Aussendungen)."""
    files = collections.OrderedDict()
    orphans = []            # Bloecke ohne empfangenen Dateikopf
    cur = None              # dict(key, last)
    ntx = collections.Counter()
    for x in pk:
        if not x.ok:
            continue
        if x.S == HEADER:
            if not x.data:                     # Endepaket
                cur = None
                continue
            h = parse_header(x.data)
            if h is None:
                continue
            key = (x.A, h[0], h[1])
            if cur is not None and cur["key"][:3] == key and \
                    cur["last"] is None:
                continue                       # Wiederholung des Kopfs
            if h[1] is None:                   # Nachricht: je Aussendung
                ntx[key] += 1
                key += (ntx[key],)
            f = files.setdefault(key, dict(group=h[2], count=0, blocks={}))
            f["count"] += 1
            cur = dict(key=key, last=None)
            continue
        s = x.S - 0x21
        if cur is None:
            o = orphans[-1] if orphans else None
            if o is None or o["A"] != x.A or \
                    abs(_unwrap(o["last"], s) - o["last"]) > 40:
                o = dict(A=x.A, last=None, blocks={})
                orphans.append(o)
            tgt = o
        else:
            tgt = cur
            f = files[cur["key"]]
        k = _unwrap(tgt["last"], s)
        tgt["last"] = k
        blocks = (tgt if cur is None else f)["blocks"]
        blocks.setdefault(k, collections.Counter())[x.data] += 1

    results = []
    used = collections.Counter()
    for key, f in files.items():
        A, name, size = key[:3]
        blocks = {k: _block_data(c) for k, c in f["blocks"].items() if k >= 0}
        lens = [len(b) for b in blocks.values() if not _is_fill(b)]
        est = int(np.median(lens)) if lens else 210
        data = bytearray()
        missing, k = 0, 0
        last = max(blocks) if blocks else -1
        while True:
            if size is not None and len(data) >= size:
                break
            b = blocks.get(k)
            if size is None and (k > last or (b is not None and _is_fill(b))):
                break
            if b is None:
                if size is not None and k > last:
                    # Rest nicht empfangen: geschaetzte Zahl Bloecke
                    rest = -(-(size - len(data)) // max(est, 1))
                    missing += rest
                    k += rest
                    data += bytes(size - len(data))
                    break
                missing += 1
                b = bytes(est)
            data += b
            k += 1
        if size is not None:
            data = data[:size]
        complete = missing == 0
        used[(A, name.lower())] += 1
        fname = _safe(name)
        if used[(A, name.lower())] > 1:
            stem, ext = os.path.splitext(fname)
            fname = "%s_%d%s" % (stem, used[(A, name.lower())], ext)
        path = None
        if complete or keep_partial:
            sd = os.path.join(outdir, "kanal_%02X" % A)
            os.makedirs(sd, exist_ok=True)
            path = os.path.join(sd, fname + ("" if complete
                                             else ".unvollstaendig"))
            with open(path, "wb") as fh:
                fh.write(data)
        results.append((A, fname, size if size is not None else len(data),
                        missing, k, path, f["count"]))
        if not complete:
            continue
        z = check_zip(data) if data[:2] == b"PK" else None
        if z:
            log("  Kanal %02X: %s ist ein ZIP-Archiv mit %s%s" %
                (A, fname, ", ".join(z[0]),
                 "" if z[1] is None else " (%s)" % z[1]))
        tv = text_view(data)
        if tv and path:
            with open(path + ".txt", "w", encoding="utf-8") as fh:
                fh.write(tv[1])
            log("  Kanal %02X: %s ist Text (xor %02X) -> %s" %
                (A, fname, tv[0], os.path.basename(path) + ".txt"))
    if keep_partial:
        for i, o in enumerate(orphans):
            blocks = {k: _block_data(c) for k, c in o["blocks"].items()}
            sd = os.path.join(outdir, "kanal_%02X" % o["A"])
            os.makedirs(sd, exist_ok=True)
            path = os.path.join(sd, "ohne_dateikopf_%d.bin.unvollstaendig"
                                % i)
            with open(path, "wb") as fh:
                for k in range(min(blocks), max(blocks) + 1):
                    b = blocks.get(k, bytes(210))
                    if _is_fill(b):
                        break
                    fh.write(b)
            log("  Kanal %02X: %d Datenbloecke ohne Dateikopf -> %s" %
                (o["A"], len(blocks), os.path.basename(path)))
    return results


CLOCK_RE = re.compile(r"([0-2]\d):([0-5]\d):([0-5]\d)\r\n")
IDLE = (0xFF,)                  # Ruhezustand der Leitung, kein Zeichen


def zwischentexte(stream, pk, min_len=8):
    """Klartext zwischen den Paketen sammeln (Uhrzeit, Ankuendigungen).
    FF-Bytes sind Ruhezustand und werden vorher entfernt; Fuellzeichen
    (Leerzeichen, '*', '.') und das Testmuster "ABC...xyz" werden
    zusammengefasst. Rueckgabe: Liste (Byteposition, Text)."""
    data = stream.data if hasattr(stream, "data") else stream
    spans = sorted((x.pos, x.pos + len(x.raw)) for x in pk if x.ok)
    gaps, p = [], 0
    for a, b in spans:
        if a > p:
            gaps.append((p, a))
        p = max(p, b)
    gaps.append((p, len(data)))
    out = []
    seen = set()
    for a, b in gaps:
        seg = bytes(c for c in data[a:b] if c not in IDLE)
        for m in re.finditer(rb"[\x20-\x7e\r\n]{%d,}" % min_len, seg):
            t = m.group().decode("ascii")
            for c in CLOCK_RE.finditer(t):
                out.append((a, "Uhrzeit %s:%s:%s" % c.groups()))
            t = CLOCK_RE.sub("", t)
            if sum(ch in " *.\r\n" for ch in t) >= 0.8 * len(t):
                continue
            if "ABCDEFGHIJKLMNOPQRSTUVWXYZ" in t or \
                    "abcdefghijklmnopqrstuvwxyz" in t:
                t = "Testmuster ABC...xyz"
            t = t.strip()
            if len(t) < min_len or t in seen:
                continue
            seen.add(t)
            out.append((a, t))
    return sorted(out)


def run(channels, args, log):
    """channels: Liste (Bezeichnung, Stream). Wertet alle Kanaele aus."""
    res_all = []
    for label, st in channels:
        pk = find_packets(st)
        n_ok = sum(x.ok for x in pk)
        fixed = repair(pk)
        addrs = collections.Counter(x.A for x in pk if x.ok)
        log("Kanal %s: %d Bytes, %d Pakete, %d Pruefsumme ok, %d per "
            "Soft-Decision repariert, Adressen %s" %
            (label, len(st.data if hasattr(st, "data") else st), len(pk),
             n_ok, fixed,
             ", ".join("%02X" % a for a in sorted(addrs)) or "-"))
        res_all += assemble(pk, args.outdir, args.partial, log)
        if not args.keine_dienste:
            tx = zwischentexte(st, pk)
            if tx:
                sd = os.path.join(args.outdir, "dienste")
                os.makedirs(sd, exist_ok=True)
                fn = os.path.join(sd, "zwischentext_%s.txt" % label)
                with open(fn, "w", encoding="utf-8") as fh:
                    for p, t in tx:
                        fh.write("%9d  %s\n" % (p, t.replace("\r", "\\r")
                                                 .replace("\n", "\\n")))
                clocks = [t[8:] for _, t in tx if t.startswith("Uhrzeit ")]
                log("  Kanal %s: %d Zwischentexte -> %s%s" %
                    (label, len(tx), os.path.relpath(fn, args.outdir),
                     "; Uhrzeit %s" % ", ".join(clocks) if clocks else ""))
    return res_all


def split_channels(cap, log):
    """Datenzeilen-Plaetze zu Kanaelen gruppieren: Probiert alle
    Aufteilungen in Gruppen benachbarter Plaetze und waehlt die mit den
    meisten gueltigen Paketen (bei Gleichstand weniger Gruppen).
    Rueckgabe: Liste (Bezeichnung, Stream)."""
    slots = cap.slots()
    n = len(slots)
    best = None
    for mask in range(1 << max(n - 1, 0)):
        groups, g = [], [slots[0]] if slots else []
        for i in range(1, n):
            if mask >> (i - 1) & 1:
                groups.append(g)
                g = []
            g.append(slots[i])
        if g:
            groups.append(g)
        score = 0
        for g in groups:
            data = b"".join(r[0] for _, s, r in cap.items if s in g)
            score += sum(x.ok for x in find_packets(data))
        cand = (score, -len(groups), groups)
        if best is None or cand[:2] > best[:2]:
            best = cand
    groups = best[2] if best else []
    out = []
    for g in groups:
        label = "zeilen_" + "+".join(cap.slot_names.get(s, str(s + 1))
                                     for s in g)
        out.append((label, cap.stream(set(g))))
    log("Kanaele: %s" % ", ".join(l for l, _ in out))
    return out
