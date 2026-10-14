#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
videodat1993.py - Decoder fuer Pro7-VideoDat (1993) aus Roh-VBI-Aufnahmen
(bttv /dev/vbi0, PAL, 2048 Samples/Zeile, 32 Zeilen/Bild) oder aus
TBC-Dateien von vhs-decode/ld-decode (16 Bit, 4*fsc = 17,73 MHz,
1135 Samples x 313 Zeilen je Halbbild, inkl. Sync).

TBC: Die Startflanke der Daten liegt fest ~123,7 Samples nach der
Rueckflanke des H-Sync. Dieser Abstand wird aus der Datei gelernt und als
Zeitbezug je Zeile verwendet; die Feinphase kommt aus allen Bitflanken.
Dadurch werden auch Zeilen mit zerstoerter Startflanke gelesen.
Zeilen werden als PAL-Bildzeilen angegeben (z.B. 11,12,13,324,325,326);
die Halbbild-Reihenfolge wird aus der .tbc.json gelesen, falls vorhanden.

Fehlerkorrektur: Pro Bit wird eine Sicherheit (Abstand zur Schwelle)
gespeichert. Defekte Pakete werden aus der Summe der Sicherheiten aller
Kopien neu entschieden; danach werden bis zu 3 der 12 unsichersten Bits
gekippt, wenn genau eine Kombination die Pruefsumme erfuellt.

Bitebene (identisch mit MD-SE.exe, siehe videodat_decode.py):
  NRZ 2 Mbit/s, pro VBI-Zeile 10 Bytes, je Byte Startbit + 8 Datenbits
  (LSB zuerst) + Stopbit, Byte invertiert.

Paketformat 1993 (durch Analyse der Aufnahme rekonstruiert):

  Off  Inhalt
  0    56 AE          Sync
  2    A              Datenstrom/Dienst (z.B. 03 Software, 10 ddp, 65 Info)
  3    B              Bit0..3: n, Nutzlast = 2^n Bytes
                      Bit4..6: Gruppenzaehler, Bit7: Unterkanal
  4    C              Paketzaehler (8 Bit, pro Datenstrom)
  5    D              Bit0..2: d  \
  6    E              Bit2   : e   }  Permutation der 8-Byte-Bloecke
  7    F              Bit1..2: f  /
  8    S              Schluessel: k = S xor 0x58
  9    2^n Bytes      verwuerfelte Nutzlast
  9+2^n  Summe lo/hi  (A + B + C + Summe(Nutzlast_klar)) & 0xFFFF

  Entwuerfeln: Nutzlast_klar[8g+c] = Roh[9 + 8g + (q[c]-1) mod 8] xor k
  wobei q[c] = (((c xor e) + (d xor e) + 1) * (2f + 1 + 6e)) mod 8
  (Die 8-Byte-Fenster beginnen bei Offset 1, d.h. das letzte Byte eines
  Fensters ist jeweils Spalte 0 des naechsten 8er-Blocks.)

Nutzlast, Byte 0 = Typ:
  00  Info (z.B. Paketname "DSW05U93")
  02 / 0C  Dateikopf: [Typ][Sendezeit 4][Groesse LE32][DOS-Zeit 4]
                       [Namenslaenge][Name]
  03  Dateiende/Ankuendigung   [03][Sendezeit 4]
  04  Dateidaten: 2^n - 1 Bytes (meist 255)
  sonstige (01,05,06,07,0A,81,82): andere Dienste, werden roh abgelegt
Jedes Paket wird i.d.R. zweimal gesendet (Gruppen zu 4 Paketen).
"""

import argparse
import collections
import datetime
import os
import re
import struct
import sys

import numpy as np

FS = 35.46895e6
SPB = FS / 2.0e6
SYNC = b"\x56\xae"


# ---------------------------------------------------------------------------
# Bit-Slicer
# ---------------------------------------------------------------------------
class Fmt:
    """Abtastparameter einer Quelle."""
    def __init__(self, fs, lo_rng, search, min_swing):
        self.spb = fs / 2.0e6          # Samples pro Bit
        self.lo_rng = lo_rng           # Bereich fuer Austastpegel
        self.search = search           # Suchbereich Startflanke
        self.min_swing = min_swing     # Mindesthub Daten


FMT_BTTV = Fmt(35.46895e6, (16, 100), (100, 0x180), 30)
FMT_TBC_PAL = Fmt(17.734475e6, (100, 190), (150, 400), 3000)


def slice_line(l, fmt=FMT_BTTV, prior=None):
    """Eine VBI-Zeile -> (10 Bytes, 80 Bit-Sicherheiten) oder None.

    prior: erwartete Position der Startflanke (z.B. aus dem H-Sync);
    ohne prior wird die erste steigende Flanke gesucht.
    Sicherheit = (Abtastwert - Schwelle) / Hub, >0 bedeutet 'hell'."""
    l = np.asarray(l, dtype=float)
    spb = fmt.spb
    n = len(l)
    s0 = fmt.search[0]
    lo = np.median(l[fmt.lo_rng[0]:fmt.lo_rng[1]])
    hi = np.percentile(l[s0:min(n, s0 + int(105 * spb))], 90)
    if hi - lo < fmt.min_swing:
        return None
    thr = (lo + hi) / 2
    if prior is None:
        # echte steigende Flanke suchen: l[p-1] < thr <= l[p]
        p = s0 + 1
        while p < fmt.search[1] and not (l[p - 1] < thr <= l[p]):
            p += 1
        if p >= fmt.search[1]:
            return None
        e0 = p - 1 + (thr - l[p - 1]) / (l[p] - l[p - 1])
    else:
        e0 = float(prior)
    # Phase ueber alle Flanken der Zeile verfeinern
    s = l >= thr
    a = max(int(e0) - 3, 0)
    E = min(a + int(101 * spb), n - 2)
    idx = np.nonzero(s[a:E] != s[a + 1:E + 1])[0] + a
    if len(idx) > 4:
        fr = idx + (thr - l[idx]) / (l[idx + 1] - l[idx])
        ph = np.angle(np.mean(np.exp(2j * np.pi * (fr - e0) / spb)))
        e0 += ph * spb / (2 * np.pi)
    if not np.isfinite(e0) or e0 < 0 or e0 + 100 * spb >= n - 1:
        return None
    x = e0 + (np.arange(10)[:, None] * 10 + 1.5 + np.arange(8)[None, :]) * spb
    ia = x.astype(int)
    f = x - ia
    v = l[ia] * (1 - f) + l[ia + 1] * f
    soft = ((v - thr) / (hi - lo)).astype(np.float32)     # 10 x 8
    bits = soft >= 0
    out = bytes(int(np.dot(bits[k], 1 << np.arange(8))) ^ 0xFF
                for k in range(10))
    return out, soft.reshape(-1)


class Stream:
    """Byte-Strom + Bit-Sicherheiten (8 pro Byte)."""
    def __init__(self):
        self.data = bytearray()
        self.soft = []

    def add(self, r):
        self.data += r[0]
        self.soft.append(r[1])

    def finish(self):
        self.data = bytes(self.data)
        self.soft = (np.concatenate(self.soft) if self.soft
                     else np.zeros(0, np.float32))
        return self


def read_vbi(path, line_len, lines_per_frame, lines, log):
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n = len(raw) // line_len
    L = raw[:n * line_len].reshape(n, line_len)
    if lines is None:                          # Zeilen automatisch finden
        cnt = collections.Counter()
        frames = min(n // lines_per_frame, 400)
        for i in range(frames * lines_per_frame):
            r = slice_line(L[i], FMT_BTTV)
            if r is not None and SYNC in r[0]:
                cnt[i % lines_per_frame] += 1
        best = max(cnt.values()) if cnt else 0
        lines = sorted(k for k, v in cnt.items() if v >= best * 0.2)
        log("VideoDat-Zeilen (automatisch): %s" % lines)
    sel = set(lines)
    st = Stream()
    for i in range(n):
        if i % lines_per_frame not in sel:
            continue
        r = slice_line(L[i], FMT_BTTV)
        if r is not None:          # leere Zeile: nichts gesendet
            st.add(r)
    log("%d Bilder, %d Datenzeilen -> %d Bytes" %
        (n // lines_per_frame, len(st.soft), len(st.data)))
    return st.finish()


# ---------------------------------------------------------------------------
# TBC (vhs-decode / ld-decode): 16 Bit, 4*fsc, ganzes Halbbild inkl. Sync
# ---------------------------------------------------------------------------
def hsync_end(l):
    """Rueckflanke des H-Sync (50 % zwischen Sync-Boden und Austastwert),
    Sub-Sample genau, oder None."""
    tip = np.median(l[10:60])
    blank = np.median(l[100:190])
    if blank - tip < 3000:
        return None
    mid = (tip + blank) / 2
    seg = l[60:130]
    k = np.nonzero((seg[:-1] < mid) & (seg[1:] >= mid))[0]
    if not len(k):
        return None
    p = 60 + k[0]
    return p + (mid - l[p]) / (l[p + 1] - l[p])


def tbc_field_order(path, nf):
    """Liest isFirstField aus der .tbc.json (falls vorhanden)."""
    import json
    for js in (path + ".json", os.path.splitext(path)[0] + ".tbc.json"):
        if os.path.exists(js):
            try:
                meta = json.load(open(js))
                return [bool(f.get("isFirstField", i % 2 == 0))
                        for i, f in enumerate(meta.get("fields", []))][:nf]
            except Exception:
                pass
    return None


def read_tbc(path, width, field_lines, lines, use_sync, log):
    raw = np.memmap(path, dtype="<u2", mode="r")
    fsz = width * field_lines
    nf = len(raw) // fsz
    F = raw[:nf * fsz].reshape(nf, field_lines, width)
    order = tbc_field_order(path, nf)
    if order is None or len(order) < nf:
        order = [i % 2 == 0 for i in range(nf)]
        log("Keine .tbc.json gefunden: 1. Halbbild der Datei = 1. Halbbild")

    def idx_sets_from_frame_lines(fl):
        f1 = {n - 1 for n in fl if 1 <= n <= 313}
        f2 = {n - 314 for n in fl if 314 <= n <= 625}
        return f1, f2

    # 1) Zeilen bestimmen (Halbbild-Zeilenindex)
    if lines is None:
        cnt = [collections.Counter(), collections.Counter()]
        for fi in range(min(nf, 150)):
            par = 0 if order[fi] else 1
            for li in range(min(40, field_lines)):
                r = slice_line(F[fi, li], FMT_TBC_PAL)
                if r is not None and SYNC in r[0]:
                    cnt[par][li] += 1
        sets = []
        for c in cnt:
            best = max(c.values()) if c else 0
            sets.append({k for k, v in c.items() if v >= best * 0.2})
        log("VideoDat-Zeilen (automatisch, Bildzeilen PAL): %s" %
            (sorted(k + 1 for k in sets[0]) + sorted(k + 314 for k in sets[1])))
    else:
        sets = list(idx_sets_from_frame_lines(lines))

    # 2) Abstand H-Sync -> Startflanke lernen
    offset = None
    if use_sync:
        offs = []
        for fi in range(min(nf, 100)):
            for li in sorted(sets[0 if order[fi] else 1]):
                l = np.asarray(F[fi, li], dtype=float)
                s = hsync_end(l)
                r = slice_line(l, FMT_TBC_PAL)
                if s is None or r is None:
                    continue
                # Startflanke aus erster Flanke
                lo = np.median(l[100:190])
                hi = np.percentile(l[150:1100], 90)
                thr = (lo + hi) / 2
                p = 151
                while p < 400 and not (l[p - 1] < thr <= l[p]):
                    p += 1
                if p < 400:
                    offs.append(p - 1 + (thr - l[p - 1]) / (l[p] - l[p - 1]) - s)
        if len(offs) >= 10:
            offset = float(np.median(offs))
            log("Startflanke %.2f Samples nach H-Sync-Rueckflanke "
                "(Streuung %.2f)" % (offset, float(np.std(offs))))

    # 3) dekodieren
    st = Stream()
    nsync = 0
    for fi in range(nf):
        for li in sorted(sets[0 if order[fi] else 1]):
            l = np.asarray(F[fi, li], dtype=float)
            prior = None
            if offset is not None:
                s = hsync_end(l)
                if s is not None:
                    prior = s + offset
                    nsync += 1
            r = slice_line(l, FMT_TBC_PAL, prior)
            if r is not None:
                st.add(r)
    log("%d Halbbilder, %d Datenzeilen (%d mit H-Sync-Bezug) -> %d Bytes" %
        (nf, len(st.soft), nsync, len(st.data)))
    return st.finish()


# ---------------------------------------------------------------------------
# Paketebene
# ---------------------------------------------------------------------------
def qperm(D, E, F):
    e = (E >> 2) & 1
    d = D & 7
    f = (F >> 1) & 3
    m = (2 * f + 1 + 6 * e) % 8
    return [(((c ^ e) + (d ^ e) + 1) * m) % 8 for c in range(8)]


def descramble(raw):
    """raw = komplettes Paket inkl. Sync und Summe -> (Nutzlast, ok)"""
    A, B, C, D, E, F, S = raw[2:9]
    n = 1 << (B & 15)
    if len(raw) < n + 11:
        return b"", False
    k = S ^ 0x58
    q = qperm(D, E, F)
    pay = bytearray(n)
    for i in range(n):
        g, c = divmod(i, 8)
        pos = 9 + 8 * g + (q[c] - 1) % 8
        pay[i] = raw[pos] ^ k if pos < 9 + n else 0
    if n < 8:                       # sehr kurze Pakete: keine Permutation
        pay = bytearray(b ^ k for b in raw[9:9 + n])
    ck = raw[9 + n] | raw[10 + n] << 8
    ok = ((A + B + C + sum(pay)) & 0xFFFF) == ck
    return bytes(pay), ok


class Pkt:
    __slots__ = ("pos", "A", "B", "C", "raw", "pay", "ok", "soft", "how")


def find_packets(stream):
    """stream: Stream (mit Bit-Sicherheiten) oder bytes"""
    soft = None
    if isinstance(stream, Stream):
        soft, stream = stream.soft, stream.data
    pk = []
    for m in re.finditer(SYNC, stream):
        p = m.start()
        if p + 9 > len(stream):
            break
        B = stream[p + 3]
        L = (1 << (B & 15)) + 11
        if p + L > len(stream):
            continue
        x = Pkt()
        x.pos = p
        x.raw = stream[p:p + L]
        x.A, x.B, x.C = x.raw[2], x.raw[3], x.raw[4]
        x.pay, x.ok = descramble(x.raw)
        x.soft = soft[8 * p:8 * (p + L)] if soft is not None else None
        x.how = "ok" if x.ok else ""
        pk.append(x)
    return pk


_BITW = 1 << np.arange(8)


def soft_to_raw(soft):
    b = (soft.reshape(-1, 8) >= 0).astype(np.int64)
    return bytes(int(v) ^ 0xFF for v in b @ _BITW)


def soft_fix(xs, nflip=12, maxflips=3):
    """Soft-Decision: Bit-Sicherheiten aller Kopien addieren, danach die
    unsichersten Bits probeweise kippen. Nur eine eindeutige Loesung wird
    akzeptiert (die Pruefsumme ist eine reine Summe)."""
    xs = [x for x in xs if x.soft is not None and len(x.soft) == 8 * len(x.raw)]
    if not xs:
        return None
    m = np.sum([x.soft for x in xs], axis=0)
    # Sync (2 Bytes) ist bekannt
    raw = bytearray(soft_to_raw(m))
    raw[0:9] = xs[0].raw[0:9]       # Sync + Kopf sind allen Kopien gemeinsam
    if descramble(bytes(raw))[1]:
        return bytes(raw)
    cand = np.argsort(np.abs(m[72:]))[:nflip] + 72
    sols = []
    import itertools
    for k in range(1, maxflips + 1):
        for comb in itertools.combinations(cand, k):
            r = bytearray(raw)
            for bit in comb:
                r[bit // 8] ^= 1 << (bit % 8)
            if descramble(bytes(r))[1]:
                sols.append(bytes(r))
                if len(sols) > 1:
                    return None
        if sols:
            return sols[0]
    return None


def repair(pk, max_diff=14):
    """Defekte Pakete aus mehreren Kopien (gleicher Kopf) zusammensetzen."""
    groups = collections.defaultdict(list)
    for x in pk:
        groups[x.raw[2:9]].append(x)
    fixed = collections.Counter()
    for key, xs in groups.items():
        if any(x.ok for x in xs):
            continue
        L = max(len(x.raw) for x in xs)
        xs = [x for x in xs if len(x.raw) == L]
        good = None
        how = "soft"
        # 0) Soft-Decision ueber alle Kopien
        good = soft_fix(xs)
        if good is None and len(xs) < 2:
            continue
        if good is None:
            how = "kopien"
        # 1) einzelne Bytefehler: Kombinationen der abweichenden Bytes
        cols = [sorted(set(x.raw[i] for x in xs)) for i in range(L)]
        var = [i for i in range(L) if len(cols[i]) > 1]
        if good is None and 0 < len(var) <= max_diff:
            base = bytearray(xs[0].raw)
            for combo in range(1 << len(var)):
                for j, i in enumerate(var):
                    ch = cols[i]
                    base[i] = ch[(combo >> j) & 1] if len(ch) == 2 else ch[0]
                if descramble(bytes(base))[1]:
                    good = bytes(base)
                    break
        # 2) Anfang aus einer Kopie, Rest aus der anderen; ausgefallenes
        #    Bild (60 Bytes) in einer Kopie
        if good is None:
            for a in xs:
                for b in xs:
                    if a is b or good:
                        continue
                    for s in range(9, L):
                        r = a.raw[:s] + b.raw[s:]
                        if descramble(r)[1]:
                            good = r
                            break
                    for s in range(9, L - 60):
                        if good:
                            break
                        r = a.raw[:s] + b.raw[s:s + 60] + a.raw[s:L - 60]
                        if descramble(r)[1]:
                            good = r
        if good:
            pay, ok = descramble(good)
            for x in xs:
                x.raw, x.pay, x.ok, x.how = good, pay, True, how
            fixed[how] += 1
    return fixed


# ---------------------------------------------------------------------------
# Dateien zusammensetzen
# ---------------------------------------------------------------------------
def dos_time(v):
    date, tm = v >> 16, v & 0xFFFF
    try:
        return datetime.datetime(1980 + (date >> 9), (date >> 5) & 15,
                                 date & 31, tm >> 11, (tm >> 5) & 63,
                                 (tm & 31) * 2)
    except ValueError:
        return None


def safe_name(b):
    n = b.split(b"\0")[0].decode("cp437", "replace")
    n = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", n).strip(" .")
    return n or "unbenannt"


def assemble(pk, outdir, keep_partial, dump_other, log):
    os.makedirs(outdir, exist_ok=True)
    subs = collections.defaultdict(list)
    for x in pk:
        if x.ok:
            subs[(x.A, x.B >> 7)].append(x)
    results = []
    for (A, sub), xs in sorted(subs.items()):
        # Zaehler C entfalten (Empfangsreihenfolge, Ausreisser ignorieren)
        seq = {}
        last = None
        for x in xs:
            if last is None:
                u = x.C
            else:
                d = (x.C - last[0]) & 0xFF
                if d > 128:
                    d -= 256
                u = last[1] + d
            if last is None or abs(d) <= 64:
                last = (x.C, u)
            seq.setdefault(u, x.pay)
        order = sorted(seq)
        other = collections.Counter()
        orphan = {}
        cur = None

        def finish(cur):
            if cur is None:
                return
            data = bytearray()
            missing = 0
            for i in range(cur["nblk"]):
                blk = cur["blocks"].get(i)
                if blk is None:
                    missing += 1
                    blk = bytes(cur["blksz"])
                data += blk
            data = data[:cur["size"]]
            complete = missing == 0 and len(data) == cur["size"]
            name = cur["name"]
            if not complete:
                if not keep_partial:
                    results.append((A, name, cur["size"], missing,
                                    cur["nblk"], None))
                    return
                name += ".unvollstaendig"
            sd = os.path.join(outdir, "strom_%02X" % A)
            os.makedirs(sd, exist_ok=True)
            path = os.path.join(sd, name)
            with open(path, "wb") as f:
                f.write(data)
            if cur["mtime"]:
                t = cur["mtime"].timestamp()
                os.utime(path, (t, t))
            results.append((A, cur["name"], cur["size"], missing,
                            cur["nblk"], path))

        for u in order:
            p = seq[u]
            t = p[0]
            if t in (0x02, 0x0C) and len(p) >= 14:
                finish(cur)
                size = struct.unpack_from("<I", p, 5)[0]
                nl = p[13]
                cur = dict(name=safe_name(p[14:14 + nl]), size=size,
                           mtime=dos_time(struct.unpack_from("<I", p, 9)[0]),
                           start=u, blocks={}, blksz=None, nblk=None)
            elif t == 0x04 and cur is None:
                orphan[u] = p[1:]
            elif t == 0x04 and cur is not None:
                blk = p[1:]
                if cur["blksz"] is None:
                    cur["blksz"] = len(blk)
                    cur["nblk"] = -(-cur["size"] // len(blk)) if blk else 0
                i = u - cur["start"] - 1
                if 0 <= i < cur["nblk"]:
                    cur["blocks"][i] = blk
                if i == cur["nblk"] - 1:
                    finish(cur)
                    cur = None
            elif t == 0x03:
                pass
            else:
                other[t] += 1
                if dump_other:
                    sd = os.path.join(outdir, "andere_dienste")
                    os.makedirs(sd, exist_ok=True)
                    with open(os.path.join(sd, "strom_%02X_typ_%02X.bin" %
                                           (A, t)), "ab") as f:
                        f.write(p)
            if t == 0x00 and len(p) > 8 and 0 < p[7] < 40 and \
                    all(32 < c < 127 for c in p[8:8 + p[7]]):
                info = p[8:8 + p[7]].decode("ascii")
                log("  Strom %02X: Infoblock '%s'" % (A, info))
        if cur is not None:
            if cur["nblk"] is None:
                cur["nblk"] = -(-cur["size"] // 255)
                cur["blksz"] = 255
            finish(cur)
        if orphan and keep_partial:
            # Daten ohne Dateikopf (Uebertragung lief schon vor Aufnahmebeginn)
            sd = os.path.join(outdir, "strom_%02X" % A)
            os.makedirs(sd, exist_ok=True)
            u0, u1 = min(orphan), max(orphan)
            bs = max(len(b) for b in orphan.values())
            path = os.path.join(sd, "ohne_dateikopf_%d.bin.unvollstaendig" % u0)
            with open(path, "wb") as f:
                for u in range(u0, u1 + 1):
                    f.write(orphan.get(u, bytes(bs)))
            log("  Strom %02X/%d: %d Datenbloecke ohne Dateikopf -> %s" %
                (A, sub, len(orphan), os.path.basename(path)))
        if other:
            log("  Strom %02X/%d: andere Pakettypen %s" %
                (A, sub, ", ".join("%02X:%d" % kv for kv in sorted(other.items()))))
    return results


def main():
    ap = argparse.ArgumentParser(description="Pro7-VideoDat (1993) aus VBI")
    ap.add_argument("input", help="VBI-Rohdatei (bttv), TBC-Datei "
                    "(vhs-decode, Endung .tbc) oder mit --bytes "
                    "ein bereits dekodierter Byte-Strom")
    ap.add_argument("--format", choices=["auto", "vbi", "tbc"],
                    default="auto", help="Eingabeformat (auto: nach Endung)")
    ap.add_argument("--tbc-width", type=int, default=1135,
                    help="TBC: Samples pro Zeile (PAL 1135)")
    ap.add_argument("--tbc-field-lines", type=int, default=313,
                    help="TBC: Zeilen pro Halbbild (PAL 313)")
    ap.add_argument("--no-sync", action="store_true",
                    help="TBC: H-Sync nicht als Zeitbezug verwenden")
    ap.add_argument("-o", "--outdir", default="videodat1993_out")
    ap.add_argument("--bytes", action="store_true",
                    help="Eingabe ist ein Byte-Strom (10 Byte je Zeile)")
    ap.add_argument("--save-bytes", metavar="DATEI",
                    help="dekodierten Byte-Strom speichern")
    ap.add_argument("--line-length", type=int, default=2048)
    ap.add_argument("--lines-per-frame", type=int, default=32)
    ap.add_argument("--lines", help="VBI: Zeilenindex im Puffer, z.B. "
                    "4,5,6,20,21,22; TBC: PAL-Bildzeilen, z.B. "
                    "11,12,13,324,325,326")
    ap.add_argument("--partial", action="store_true",
                    help="unvollstaendige Dateien trotzdem schreiben "
                    "(fehlende Bloecke = Nullbytes)")
    ap.add_argument("--dump-other", action="store_true",
                    help="Pakete anderer Dienste roh ablegen")
    args = ap.parse_args()
    log = print

    fmt = args.format
    if fmt == "auto":
        fmt = "tbc" if args.input.lower().endswith(".tbc") else "vbi"
    lines = [int(v) for v in args.lines.split(",")] if args.lines else None
    if args.bytes:
        stream = open(args.input, "rb").read()
    elif fmt == "tbc":
        stream = read_tbc(args.input, args.tbc_width, args.tbc_field_lines,
                          lines, not args.no_sync, log)
    else:
        stream = read_vbi(args.input, args.line_length, args.lines_per_frame,
                          lines, log)
    if args.save_bytes:
        open(args.save_bytes, "wb").write(
            stream.data if isinstance(stream, Stream) else stream)

    pk = find_packets(stream)
    n_ok = sum(x.ok for x in pk)
    fixed = repair(pk)
    log("Pakete: %d gefunden, %d Pruefsumme ok, repariert: %d per "
        "Soft-Decision, %d aus Kopien" %
        (len(pk), n_ok, fixed["soft"], fixed["kopien"]))
    res = assemble(pk, args.outdir, args.partial, args.dump_other, log)
    log("")
    log("Dateien:")
    for A, name, size, missing, nblk, path in res:
        st = "vollstaendig" if missing == 0 else \
            "unvollstaendig (%d von %d Bloecken fehlen)" % (missing, nblk)
        log("  Strom %02X  %-14s %8d Bytes  %s%s" %
            (A, name, size, st, "" if path else "  [nicht geschrieben]"))


if __name__ == "__main__":
    main()
