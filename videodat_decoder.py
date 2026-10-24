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
Kopien neu entschieden; danach werden Bits gekippt, in denen sich die
Kopien widersprechen, wenn genau eine Kombination die Pruefsumme erfuellt.
Einzelne Kopien werden nicht auf Verdacht repariert (die Summen-Pruefsumme
kann einen Fehler nicht orten).

Bitebene (identisch mit MD-SE.exe, siehe videodat_decode.py):
  NRZ 2 Mbit/s, pro VBI-Zeile 10 Bytes, je Byte Startbit + 8 Datenbits
  (LSB zuerst) + Stopbit, Byte invertiert.
  Die fruehe Variante von 1990 (800 kbit/s, 4 Bytes pro Zeile, eigenes
  Paketformat) ist in videodat1990.py beschrieben; die Variante wird am
  Signal erkannt (--variant).

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
  sonstige: andere Dienste, siehe analyse_dienste() (Uhrzeit, Sendungs-
  ablauf, Bitlisten, verschluesselte Dienste)
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
MAX_FILE = 64 * 1024 * 1024     # groessere Dateikoepfe gelten als Fehler


# ---------------------------------------------------------------------------
# Bit-Slicer
# ---------------------------------------------------------------------------
class Fmt:
    """Abtastparameter einer Quelle und einer Uebertragungsvariante."""
    def __init__(self, fs, lo_rng, search, min_swing, variant="1993"):
        v = VARIANTS[variant]
        self.fs = fs
        self.variant = variant
        self.spb = fs / v["bitrate"]   # Samples pro Bit
        self.nbytes = v["nbytes"]      # Bytes pro Zeile
        self.nbits = 10 * self.nbytes  # inkl. Start- und Stopbits
        # Mindestzahl korrekter Start-/Stopbits fuer eine Datenzeile
        self.frame_min = v["frame_min"]
        self.lo_rng = lo_rng           # Bereich fuer Austastpegel
        self.search = search           # Suchbereich Startflanke
        self.min_swing = min_swing     # Mindesthub Daten

    def with_variant(self, variant):
        return Fmt(self.fs, self.lo_rng, self.search, self.min_swing, variant)


# Uebertragungsvarianten (Bitebene):
#   1993: 2 Mbit/s, 10 Bytes pro Zeile, Pakete mit Sync 56 AE (siehe oben)
#   1990: 800 kbit/s, 4 Bytes pro Zeile, SOH-Pakete (siehe videodat1990.py)
# Bei nur 8 Rahmungsbits (1990) muss die Rahmung fehlerfrei sein.
VARIANTS = {
    "1993": dict(bitrate=2.0e6, nbytes=10, frame_min=18),
    "1990": dict(bitrate=0.8e6, nbytes=4, frame_min=8),
}

FMT_BTTV = Fmt(35.46895e6, (16, 100), (100, 0x180), 30)
FMT_TBC_PAL = Fmt(17.734475e6, (100, 190), (150, 400), 3000)


def line_phase(l, fmt=FMT_BTTV, prior=None):
    """Startflanke einer VBI-Zeile -> (e0, Schwelle, lo, hi) oder None.

    prior: erwartete Position der Startflanke (z.B. aus dem H-Sync);
    ohne prior wird die erste steigende Flanke gesucht."""
    l = np.asarray(l, dtype=float)
    spb = fmt.spb
    nb = fmt.nbits
    n = len(l)
    s0 = fmt.search[0]
    lo = np.median(l[fmt.lo_rng[0]:fmt.lo_rng[1]])
    hi = np.percentile(l[s0:min(n, s0 + int((nb + 5) * spb))], 90)
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
    E = min(a + int((nb + 1) * spb), n - 2)
    idx = np.nonzero(s[a:E] != s[a + 1:E + 1])[0] + a
    if len(idx) > 4:
        fr = idx + (thr - l[idx]) / (l[idx + 1] - l[idx])
        ph = np.angle(np.mean(np.exp(2j * np.pi * (fr - e0) / spb)))
        e0 += ph * spb / (2 * np.pi)
    if not np.isfinite(e0) or e0 < 0 or e0 + nb * spb >= n - 1:
        return None
    return e0, thr, lo, hi


def framing(l, ph, fmt):
    """Anzahl korrekter Start-/Stopbits (0..2*Bytes, 1993: 0..20) einer
    Zeile mit Phase ph (aus line_phase). Start- und Stopbit haben in jedem
    Byte entgegengesetzten Pegel; zufaellige Signale erreichen 1993 selten
    mehr als 15."""
    e0, thr = ph[0], ph[1]
    k = np.arange(fmt.nbytes) * 10
    st = l[(e0 + (k + 0.5) * fmt.spb).astype(int)] >= thr
    sp = l[(e0 + (k + 9.5) * fmt.spb).astype(int)] >= thr
    a = int(st.sum()) + int((~sp).sum())
    return max(a, 2 * fmt.nbytes - a)


def framing_full(fmt):
    return 2 * fmt.nbytes


def slice_line(l, fmt=FMT_BTTV, prior=None, ph=None):
    """Eine VBI-Zeile -> (Bytes, 8 Bit-Sicherheiten je Byte) oder None.

    prior: siehe line_phase; ph: bereits bestimmte Phase (aus line_phase).
    Sicherheit = (Abtastwert - Schwelle) / Hub, >0 bedeutet 'hell'."""
    l = np.asarray(l, dtype=float)
    if ph is None:
        ph = line_phase(l, fmt, prior)
    if ph is None:
        return None
    e0, thr, lo, hi = ph
    spb = fmt.spb
    x = e0 + (np.arange(fmt.nbytes)[:, None] * 10 + 1.5 +
              np.arange(8)[None, :]) * spb
    ia = x.astype(int)
    f = x - ia
    v = l[ia] * (1 - f) + l[ia + 1] * f
    soft = ((v - thr) / (hi - lo)).astype(np.float32)     # Bytes x 8
    bits = soft >= 0
    out = bytes(int(np.dot(bits[k], 1 << np.arange(8))) ^ 0xFF
                for k in range(fmt.nbytes))
    return out, soft.reshape(-1)


class Capture:
    """Gelesene Datenzeilen in Empfangsreihenfolge: (Halbbild bzw. Bild,
    Platz, (Bytes, Sicherheiten)). Platz = Rang der Zeile unter den
    Datenzeilen ihres Halbbilds (0 = oberste); 1990 traegt jeder Platz
    bzw. jede Gruppe von Plaetzen einen eigenen Kanal."""
    def __init__(self, fmt):
        self.fmt = fmt
        self.items = []
        self.slot_names = {}       # Platz -> Zeilen, z.B. "11-324"

    def add(self, fi, slot, r):
        self.items.append((fi, slot, r))

    def slots(self):
        return sorted({s for _, s, _ in self.items})

    def stream(self, slots=None):
        """Byte-Strom aller (bzw. der angegebenen) Plaetze."""
        st = Stream()
        for _, s, r in self.items:
            if slots is None or s in slots:
                st.add(r)
        return st.finish()


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


def vbi_framing(l, fmt=FMT_BTTV):
    """(Phase, Rahmung) einer bttv-Zeile; Rahmung -1 ohne Daten."""
    l = np.asarray(l, dtype=float)
    ph = line_phase(l, fmt)
    return ph, (-1 if ph is None else framing(l, ph, fmt))


def choose_variant(lines, fmt, log, variant="auto"):
    """Uebertragungsvariante aus einer Stichprobe von Zeilen bestimmen:
    die Variante, bei der die meisten Zeilen einwandfrei gerahmt sind.
    Rueckgabe: Fmt der gewaehlten Variante."""
    if variant != "auto":
        return fmt.with_variant(variant)
    hits = collections.Counter()
    fmts = {v: fmt.with_variant(v) for v in VARIANTS}
    for l in lines:
        l = np.asarray(l, dtype=float)
        for v, f in fmts.items():
            ph = line_phase(l, f)
            if ph is not None and framing(l, ph, f) >= f.frame_min:
                hits[v] += 1
    best = max(VARIANTS, key=lambda v: (hits[v], v))
    log("Variante %s (gerahmte Zeilen in der Stichprobe: %s)" %
        (best, ", ".join("%s: %d" % (v, hits[v]) for v in sorted(VARIANTS))))
    return fmts[best]


def detect_vbi_lines(L, nfr, lpf, max_frames, log, block=25, need=50,
                     stable_blocks=5, fmt=FMT_BTTV):
    """VideoDat-Zeilen je Halbbild im bttv-Puffer finden.

    Bei bt848-Aufnahmen springt die Lage der Datenzeilen am Aufnahmeanfang
    oft von Bild zu Bild um einige Zeilen (instabile V-Synchronisation),
    oder es steht noch ein falscher Bildinhalt im Puffer. Gezaehlt wird daher
    je Halbbild die zusammenhaengende Gruppe einwandfrei gerahmter Zeilen
    (siehe framing); Anzahl und Lage der Zeilen sind die haeufigsten Werte.
    Gesucht wird blockweise (1 s), bis sich das Ergebnis stable_blocks Bloecke
    lang nicht mehr aendert und genug Stimmen vorliegen, hoechstens
    max_frames Bilder.
    Rueckgabe: Liste der Zeilenindizes je Halbbild."""
    half = lpf // 2
    runs = [collections.Counter(), collections.Counter()]

    def result():
        out, votes = [], []
        for c in runs:
            if not c:
                out.append([])
                votes.append(0)
                continue
            k = collections.Counter()
            for (_, n), v in c.items():
                k[n] += v
            k = k.most_common(1)[0][0]
            (s0, _), v = max(((r, v) for r, v in c.items() if r[1] == k),
                             key=lambda x: x[1])
            out.append(list(range(s0, s0 + k)))
            votes.append(v)
        return out, votes

    last, stable, end, fi = None, 0, min(nfr, max_frames), 0
    for b0 in range(0, end, block):
        for fi in range(b0, min(b0 + block, end)):
            for h in (0, 1):
                ok = [li for li in range(h * half, (h + 1) * half)
                      if vbi_framing(L[fi * lpf + li], fmt)[1] >=
                      fmt.frame_min]
                if ok and ok[-1] - ok[0] + 1 == len(ok):
                    runs[h][(ok[0], len(ok))] += 1
        sets, votes = result()
        if min(votes) >= need and sets == last:
            stable += 1
        else:
            stable = 0
        last = sets
        if stable >= stable_blocks:
            break
    t = min(fi + 1, end)
    if last is None:
        last, votes = [[], []], [0, 0]
    log("Zeilensuche ueber %d Bilder (%.1f s), Stimmen je Halbbild: %s" %
        (t, t / 25.0, ", ".join("%d" % v for v in votes)))
    for h in (0, 1):
        if not last[h]:
            log("WARNUNG: im %d. Halbbild keine VideoDat-Zeile gefunden "
                "(ggf. --lines angeben oder --scan-seconds erhoehen)" % (h + 1))
    return last


def track_vbi_lines(L, base, nominal, half_rng, fmt=FMT_BTTV):
    """Zeilen eines Halbbilds, die dekodiert werden sollen.

    Sind alle erwarteten Zeilen einwandfrei gerahmt, bleibt es bei ihnen.
    Sonst wird die Zeilengruppe innerhalb des Halbbilds verschoben, bis
    moeglichst viele ihrer Zeilen gerahmt sind (bei Gleichstand die kleinste
    Verschiebung). Bei einer Verschiebung werden nur gerahmte Zeilen
    verwendet, damit kein Bildinhalt in den Datenstrom geraet; findet sich
    nirgends eine gerahmte Zeile, bleibt es bei den erwarteten Zeilen.
    Rueckgabe: (Liste (Zeilenindex, Phase), Verschiebung)."""
    q = {}

    def get(li):
        if li not in q:
            q[li] = vbi_framing(L[base + li], fmt)
        return q[li]

    if all(get(li)[1] >= fmt.frame_min for li in nominal):
        return [(li, q[li][0]) for li in nominal], 0
    lo, hi = half_rng

    def framed(d):          # gerahmte Zeilen der Gruppe innerhalb des Halbbilds
        return [li + d for li in nominal
                if lo <= li + d < hi and get(li + d)[1] >= fmt.frame_min]

    best = (0, 0)
    for d in sorted(range(lo - nominal[-1], hi - nominal[0]), key=abs):
        c = len(framed(d))
        if c > best[0]:
            best = (c, d)
    d = best[1]
    if d == 0:
        return [(li, get(li)[0]) for li in nominal], 0
    return [(li, q[li][0]) for li in framed(d)], d


def read_vbi(path, line_len, lines_per_frame, lines, log,
             scan_seconds=30.0, track=True, variant="auto"):
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n = len(raw) // line_len
    L = raw[:n * line_len].reshape(n, line_len)
    nfr = n // lines_per_frame
    half = lines_per_frame // 2
    # Stichprobe fuer die Variantenwahl: alle Zeilen jedes 5. Bilds
    probe = (L[fi * lines_per_frame + li]
             for fi in range(0, min(nfr, int(scan_seconds * 25)), 5)
             for li in range(lines_per_frame))
    fmt = choose_variant(probe, FMT_BTTV, log, variant)
    if lines is None:                          # Zeilen automatisch finden
        sets = detect_vbi_lines(L, nfr, lines_per_frame,
                                int(scan_seconds * 25), log, fmt=fmt)
        log("VideoDat-Zeilen (automatisch): %s" % (sets[0] + sets[1]))
    else:
        sets = [sorted(li for li in lines if li // half == h)
                for h in (0, 1)]
        rest = sorted(li for li in lines if li >= 2 * half)
        if rest:
            log("WARNUNG: Zeilen ausserhalb des Puffers ignoriert: %s" % rest)
    cap = Capture(fmt)
    for i in range(max(len(sets[0]), len(sets[1]))):
        cap.slot_names[i] = "-".join(str(z[i]) for z in sets if i < len(z))
    shifted = 0
    for fi in range(nfr):
        base = fi * lines_per_frame
        for h, nominal in enumerate(sets):
            if not nominal:
                continue
            if track:
                sel, d = track_vbi_lines(L, base, nominal,
                                         (h * half, (h + 1) * half), fmt)
                shifted += d != 0
            else:
                sel, d = [(li, None) for li in nominal], 0
            for li, ph in sel:
                r = slice_line(L[base + li], fmt, ph=ph)
                if r is not None:          # leere Zeile: nichts gesendet
                    cap.add(fi, nominal.index(li - d), r)
    log("%d Bilder, %d Datenzeilen -> %d Bytes" %
        (nfr, len(cap.items), len(cap.items) * fmt.nbytes))
    if shifted:
        log("  in %d Halbbildern lagen die VideoDat-Zeilen verschoben" %
            shifted)
    return cap


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


def detect_tbc_lines(F, order, nf, max_fields, log, maxline=40,
                     block=50, need=100, stable_blocks=5, fmt=FMT_TBC_PAL):
    """VideoDat-Zeilen je Halbbild-Paritaet finden.

    Kriterium ist die Byte-Rahmung (1993: Start-/Stopbit in allen 10 Bytes,
    hoechstens 2 Fehler; 1990: alle 4 Bytes fehlerfrei), nicht das Sync-Wort: 56 AE liegt nur selten
    vollstaendig in einer Zeile, und in gestoerten Aufnahmen reichten die
    wenigen Treffer am Anfang nicht aus. Gesucht wird blockweise (1 s),
    bis beide Paritaeten genug Treffer haben und sich die Auswahl
    stable_blocks Bloecke lang nicht mehr aendert, hoechstens max_fields
    Halbbilder.
    Rueckgabe: (Zeilenmengen je Paritaet, Liste H-Sync-Offsets aus
    einwandfrei gerahmten Zeilen)."""
    score = [np.zeros(maxline, int), np.zeros(maxline, int)]
    offs = []

    def select(sc):
        best = sc.max()
        if best < 10:
            return frozenset()
        return frozenset(int(k) for k in np.nonzero(sc >= best * 0.3)[0])

    last, stable, end = None, 0, min(nf, max_fields)
    for b0 in range(0, end, block):
        for fi in range(b0, min(b0 + block, end)):
            par = 0 if order[fi] else 1
            rows = np.asarray(F[fi, :maxline], dtype=float)
            for li in range(min(maxline, len(rows))):
                l = rows[li]
                ph = line_phase(l, fmt)
                if ph is None:
                    continue
                q = framing(l, ph, fmt)
                if q >= fmt.frame_min:
                    score[par][li] += 1
                if q == framing_full(fmt):
                    s = hsync_end(l)
                    if s is not None:
                        offs.append((par, li, ph[0] - s))
        sets = [select(score[0]), select(score[1])]
        if min(score[0].max(), score[1].max()) >= need and sets == last:
            stable += 1
        else:
            stable = 0
        last = sets
        if stable >= stable_blocks:
            break
    t = min(b0 + block, end)
    log("Zeilensuche ueber %d Halbbilder (%.1f s), Treffer je Zeile: %s" %
        (t, t / 50.0, ", ".join(
            "%d:%d" % (li + 1 + 313 * p, score[p][li])
            for p in (0, 1) for li in sorted(last[p]))))
    for p in (0, 1):
        if not last[p]:
            log("WARNUNG: im %d. Halbbild keine VideoDat-Zeile gefunden "
                "(ggf. --lines angeben oder --scan-seconds erhoehen)" % (p + 1))
    offs = [o for p, li, o in offs if li in last[p]]
    return [set(last[0]), set(last[1])], offs


# Entzerrer fuer TBC-Zeilen. Analyse einer LP-Aufnahme (tape304): Die
# 0-Bits liegen bei -0,6, die 1-Bits nur bei +0,3..+0,5 (bezogen auf
# (lo+hi)/2 und den Hub), isolierte 1-Bits sind am schwaechsten
# (Intersymbol-Interferenz), und fast alle Restfehler sind gelesene 0 statt 1
# (Einbrueche im hellen Pegel). Daher:
#   - Schwelle je Zeile aus den Mittelwerten beider Pegel (2-Means),
#   - linearer Entzerrer (9 Abtastwerte im Abstand 1/2 Bit), per kleinste
#     Quadrate aus Zeilen fehlerfreier Pakete der Aufnahme selbst gelernt,
#   - Entscheidungsschwelle zugunsten '1' verschoben; die Verschiebung wird
#     so gewaehlt, dass im Lernabschnitt die meisten Pruefsummen stimmen.
# Gegenueber der festen Schwelle (lo+hi)/2 ergab das in tape304 je nach
# Abschnitt 40..100 % mehr fehlerfreie Pakete.
EQ_TAPS = np.arange(-4, 5) * 0.5          # Abtastpunkte relativ zur Bitmitte


def tbc_features(l, ph, fmt=FMT_TBC_PAL):
    """Abtastwerte um jede der Bitmitten (Start-/Daten-/Stopbits, 1993: 100),
    normiert mit der 2-Means-Schwelle -> Matrix (Bits, len(EQ_TAPS)+1)."""
    e0, thr, lo, hi = ph
    nb = fmt.nbits
    x = e0 + (np.arange(nb)[:, None] + 0.5 + EQ_TAPS[None, :]) * fmt.spb
    x = np.clip(x, 0, len(l) - 2)
    ia = x.astype(int)
    f = x - ia
    s = l[ia] * (1 - f) + l[ia + 1] * f
    c = s[:, len(EQ_TAPS) // 2]
    t = thr
    for _ in range(5):
        a, b = c[c < t], c[c >= t]
        if not len(a) or not len(b):
            break
        t = (a.mean() + b.mean()) / 2
    return np.hstack([(s - t) / (hi - lo), np.ones((nb, 1))])


class Equalizer:
    def __init__(self, w, bias):
        self.w, self.bias = w, bias

    def soft100(self, X):
        return X @ self.w + self.bias


def soft100_to_line(y):
    """Sicherheiten aller Bits einer Zeile (inkl. Start-/Stopbits, 1993: 100)
    -> (Bytes, 8 Sicherheiten je Byte)"""
    soft = np.asarray(y, np.float32).reshape(-1, 10)[:, 1:9]
    out = (np.packbits(soft >= 0, axis=1, bitorder="little")[:, 0] ^ 0xFF)
    return out.tobytes(), soft.reshape(-1)


def slice_tbc(l, prior=None, eq=None, fmt=FMT_TBC_PAL):
    """TBC-Zeile -> (Bytes, 8 Bit-Sicherheiten je Byte) oder None.
    Ohne Entzerrer: 2-Means-Schwelle in der Bitmitte."""
    l = np.asarray(l, dtype=float)
    ph = line_phase(l, fmt, prior)
    if ph is None:
        return None
    X = tbc_features(l, ph, fmt)
    y = eq.soft100(X) if eq is not None else X[:, len(EQ_TAPS) // 2]
    return soft100_to_line(y)


def _good_lines(stream, nlines):
    """Zeilen, die vollstaendig in Paketen mit richtiger Pruefsumme liegen,
    und Anzahl dieser Pakete."""
    good = np.zeros(nlines, bool)
    n = 0
    for x in find_packets(stream):
        if x.ok:
            good[-(-x.pos // 10):(x.pos + len(x.raw)) // 10] = True
            n += 1
    return good, n


def train_equalizer(F, order, sets, offset, start, nf, log,
                    chunk=2500, max_fields=30000, need=3000):
    """Entzerrer aus der Aufnahme lernen (ab Halbbild start).
    Liest Abschnitte zu chunk Halbbildern, bis need fehlerfreie Zeilen
    vorliegen (hoechstens max_fields). Rueckgabe Equalizer oder None."""
    X, good, n_simple = [], np.zeros(0, bool), 0
    end = min(nf, start + max_fields)
    fi = start
    while fi < end:
        for fj in range(fi, min(fi + chunk, end)):
            for li in sorted(sets[0 if order[fj] else 1]):
                l = np.asarray(F[fj, li], dtype=float)
                prior = None
                if offset is not None:
                    s = hsync_end(l)
                    if s is not None:
                        prior = s + offset
                ph = line_phase(l, FMT_TBC_PAL, prior)
                if ph is not None:
                    X.append(tbc_features(l, ph))
        fi = min(fi + chunk, end)
        st = Stream()
        for Xi in X:
            st.add(soft100_to_line(Xi[:, len(EQ_TAPS) // 2]))
        good, n_simple = _good_lines(st.finish(), len(X))
        if good.sum() >= need:
            break
    if good.sum() < 200:
        log("Entzerrer: zu wenige fehlerfreie Zeilen (%d), verwende "
            "einfache Schwelle" % good.sum())
        return None
    A = np.vstack([X[i] for i in np.nonzero(good)[0]])
    Y = np.where(A[:, len(EQ_TAPS) // 2] >= 0, 1.0, -1.0)
    w = np.linalg.lstsq(A, Y, rcond=None)[0]
    # Schwellenverschiebung: meiste Pruefsummen im Lernabschnitt
    Yall = np.stack([Xi @ w for Xi in X])
    biases = np.round(np.arange(0, 0.75, 0.05), 2)
    hits = []
    for b in biases:
        st = Stream()
        for y in Yall:
            st.add(soft100_to_line(y + b))
        hits.append(sum(x.ok for x in find_packets(st.finish())))
    hits = np.array(hits, float)
    # leicht geglaettet: das Optimum faellt zu grossen Werten steil ab
    sm = np.convolve(np.pad(hits, 1, mode="edge"), np.ones(3) / 3, "valid")
    k = int(np.argmax(sm))
    log("Entzerrer aus %d fehlerfreien Zeilen (Halbbilder %d-%d) gelernt, "
        "Schwellenverschiebung %.2f: %d statt %d fehlerfreie Pakete im "
        "Lernabschnitt" % (good.sum(), start, fi, biases[k], hits[k],
                           n_simple))
    return Equalizer(w, float(biases[k]))


def read_tbc(path, width, field_lines, lines, use_sync, log,
             scan_seconds=30.0, use_eq=True, variant="auto"):
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

    # 0) Variante bestimmen (Stichprobe: Zeilen 1..40 jedes 10. Halbbilds)
    max_fields = int(scan_seconds * 50)
    maxline = min(40, field_lines)
    probe = (F[fi, li] for fi in range(0, min(nf, max_fields), 10)
             for li in range(maxline))
    fmt = choose_variant(probe, FMT_TBC_PAL, log, variant)

    # 1) Zeilen bestimmen (Halbbild-Zeilenindex)
    if lines is None:
        sets, offs = detect_tbc_lines(F, order, nf, max_fields, log,
                                      maxline=maxline, fmt=fmt)
        log("VideoDat-Zeilen (automatisch, Bildzeilen PAL): %s" %
            (sorted(k + 1 for k in sets[0]) + sorted(k + 314 for k in sets[1])))
    else:
        sets = list(idx_sets_from_frame_lines(lines))
        offs = None

    # 2) Abstand H-Sync -> Startflanke lernen (nur aus einwandfrei
    # gerahmten Zeilen; gestoerte Zeilen am Aufnahmeanfang verfaelschen
    # sonst den Wert)
    offset = None
    if use_sync:
        if offs is None:
            offs = []
            for fi in range(min(nf, max_fields)):
                for li in sorted(sets[0 if order[fi] else 1]):
                    l = np.asarray(F[fi, li], dtype=float)
                    ph = line_phase(l, fmt)
                    s = hsync_end(l)
                    if ph is not None and s is not None and \
                            framing(l, ph, fmt) == framing_full(fmt):
                        offs.append(ph[0] - s)
                if len(offs) >= 500:
                    break
        if len(offs) >= 10:
            offset = float(np.median(offs))
            log("Startflanke %.2f Samples nach H-Sync-Rueckflanke "
                "(Streuung %.2f)" % (offset, float(np.std(offs))))

    # 3) Entzerrer lernen (ab Ende der Zeilensuche, dort ist das Signal
    # i.d.R. stabil). Nur 1993: Das 800-kbit/s-Signal von 1990 ist auch
    # in VHS-Aufnahmen sauber genug fuer die einfache Schwelle.
    eq = None
    if use_eq and fmt.variant == "1990":
        log("Entzerrer: fuer Variante 1990 nicht verwendet")
    elif use_eq:
        eq = train_equalizer(F, order, sets, offset,
                             min(max_fields, nf // 2) if lines is None else 0,
                             nf, log)

    # 4) dekodieren
    cap = Capture(fmt)
    l1, l2 = sorted(k + 1 for k in sets[0]), sorted(k + 314 for k in sets[1])
    for i in range(max(len(l1), len(l2))):
        cap.slot_names[i] = "-".join(str(z[i]) for z in (l1, l2)
                                     if i < len(z))
    nsync = 0
    for fi in range(nf):
        for slot, li in enumerate(sorted(sets[0 if order[fi] else 1])):
            l = np.asarray(F[fi, li], dtype=float)
            prior = None
            if offset is not None:
                s = hsync_end(l)
                if s is not None:
                    prior = s + offset
                    nsync += 1
            r = slice_tbc(l, prior, eq, fmt)
            if r is not None:
                cap.add(fi, slot, r)
    log("%d Halbbilder, %d Datenzeilen (%d mit H-Sync-Bezug) -> %d Bytes" %
        (nf, len(cap.items), nsync, len(cap.items) * fmt.nbytes))
    return cap


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


def _unstuff_packet(stream, start):
    """Read one frame, removing FF escaped after an embedded 56 AE marker.

    Keep byte indexes so VBI soft-bit evidence still addresses the right bits.
    Callers accept this interpretation only if the complete checksum passes.
    """
    raw = bytearray(stream[start:start + 2])
    indexes = [start, start + 1]
    pos, length = start + 2, None
    while pos < len(stream) and (length is None or len(raw) < length):
        raw.append(stream[pos])
        indexes.append(pos)
        pos += 1
        if len(raw) == 4:
            length = (1 << (raw[3] & 15)) + 11
        if len(raw) > 3 and raw[-2:] == SYNC and pos < len(stream) and stream[pos] == 255:
            pos += 1
    return bytes(raw), indexes, pos


def find_packets(stream):
    """stream: Stream (mit Bit-Sicherheiten) oder bytes"""
    soft = None
    if isinstance(stream, Stream):
        soft, stream = stream.soft, stream.data
    pk = []
    valid_end = 0
    for m in re.finditer(SYNC, stream):
        p = m.start()
        if p < valid_end:
            continue
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
        if not x.ok:
            raw, indexes, end = _unstuff_packet(stream, p)
            if end - p > len(raw):
                pay, ok = descramble(raw)
                if ok:
                    x.raw, x.pay, x.ok, x.how = raw, pay, True, "unstuffed"
                    x.A, x.B, x.C = raw[2:5]
                    if soft is not None:
                        bit_indexes = (np.array(indexes)[:, None] * 8 + np.arange(8)).ravel()
                        x.soft = soft[bit_indexes]
                    valid_end = end
        elif x.ok:
            valid_end = p + L
        pk.append(x)
    return pk


_BITW = 1 << np.arange(8)


def soft_to_raw(soft):
    b = (soft.reshape(-1, 8) >= 0).astype(np.int64)
    return bytes(int(v) ^ 0xFF for v in b @ _BITW)


def soft_fix(xs, maxvar=12):
    """Soft-Decision ueber mehrere Kopien eines Pakets.

    1. Bit-Sicherheiten aller Kopien addieren und neu entscheiden.
    2. Nur bei mindestens zwei Kopien: Bits, in denen sich die Kopien
       widersprechen (hoechstens maxvar), probeweise kippen. Akzeptiert
       wird nur eine eindeutige Loesung.
    Bei einer einzelnen Kopie werden KEINE Bits auf Verdacht gekippt: Die
    Pruefsumme ist eine reine Summe und kann den Ort eines Fehlers nicht
    bestimmen (gleiche Abweichung durch dasselbe Bit in einem anderen Byte);
    das fuehrte in Simulationen zu ~10 % Fehlreparaturen."""
    import itertools
    xs = [x for x in xs if x.soft is not None and len(x.soft) == 8 * len(x.raw)]
    if not xs:
        return None
    m = np.sum([x.soft for x in xs], axis=0)
    raw = bytearray(soft_to_raw(m))
    raw[0:9] = xs[0].raw[0:9]       # Sync + Kopf sind allen Kopien gemeinsam
    if sum_ok(raw):
        return bytes(raw)
    if len(xs) < 2:
        return None
    hard = np.array([x.soft >= 0 for x in xs])
    dis = [int(b) for b in np.nonzero(hard.min(0) != hard.max(0))[0] if b >= 72]
    if not dis or len(dis) > maxvar:
        return None
    sol = None
    for k in range(1, len(dis) + 1):
        for comb in itertools.combinations(dis, k):
            r = bytearray(raw)
            for bit in comb:
                r[bit // 8] ^= 1 << (bit % 8)
            if sum_ok(r):
                if sol is not None and sol != bytes(r):
                    return None             # mehrdeutig
                sol = bytes(r)
    return sol


def sum_ok(raw):
    """Schnelle Pruefsummenkontrolle ohne Entwuerfeln: Die Permutation
    aendert die Summe nicht, also reicht die Summe der Rohbytes xor k."""
    A, B, C = raw[2], raw[3], raw[4]
    n = 1 << (B & 15)
    if len(raw) < n + 11:
        return False
    k = raw[8] ^ 0x58
    s = A + B + C + sum(b ^ k for b in raw[9:9 + n])
    return (s & 0xFFFF) == (raw[9 + n] | raw[10 + n] << 8)


def _repair_pair(a, b):
    """a, b: zwei Kopien gleicher Laenge mit gleichem Kopf.
    Probiert 'Anfang aus a, Rest aus b' und 'in a fehlt ein Bild (60 Bytes),
    das aus b eingesetzt wird'. Alles ueber kumulierte Summen."""
    L = len(a)
    n = 1 << (a[3] & 15)
    k = a[8] ^ 0x58
    base = a[2] + a[3] + a[4]
    A = np.frombuffer(a, np.uint8).astype(np.int64)
    Bv = np.frombuffer(b, np.uint8).astype(np.int64)
    ca = np.concatenate([[0], np.cumsum(A[9:9 + n] ^ k)])
    cb = np.concatenate([[0], np.cumsum(Bv[9:9 + n] ^ k)])
    lo, hi = 9 + n, 10 + n

    def ck_of(src_lo, src_hi):
        return src_lo | src_hi << 8

    sols = set()
    # 1) a[:s] + b[s:]
    for s in range(9, L):
        j = min(max(s - 9, 0), n)
        tot = base + ca[j] + (cb[n] - cb[j])
        c = ck_of(a[lo] if lo < s else b[lo], a[hi] if hi < s else b[hi])
        if (int(tot) & 0xFFFF) == c:
            r = a[:s] + b[s:]
            if sum_ok(r):
                sols.add(r)
    # 2) a[:s] + b[s:s+60] + a[s:L-60]
    if L > 70:
        As = np.concatenate([np.zeros(60, np.int64), A[:-60]])   # As[i]=a[i-60]
        cas = np.concatenate([[0], np.cumsum(As[9:9 + n] ^ k)])
        for s in range(9, L - 60):
            j1 = min(s - 9, n)
            j2 = min(s + 60 - 9, n)
            tot = base + ca[j1] + (cb[j2] - cb[j1]) + (cas[n] - cas[j2])

            def pick(p):
                if p < s:
                    return a[p]
                if p < s + 60:
                    return b[p]
                return a[p - 60]
            if (int(tot) & 0xFFFF) == ck_of(pick(lo), pick(hi)):
                r = a[:s] + b[s:s + 60] + a[s:L - 60]
                if sum_ok(r):
                    sols.add(r)
    return sols.pop() if len(sols) == 1 else None


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
        how = "soft"
        # 0) Soft-Decision ueber alle Kopien
        good = soft_fix(xs)
        if good is None and len(xs) < 2:
            continue
        if good is None:
            how = "kopien"
        # 1) einzelne Bytefehler: Kombinationen der abweichenden Bytes
        if good is None:
            cols = [sorted(set(x.raw[i] for x in xs)) for i in range(L)]
            var = [i for i in range(L) if len(cols[i]) > 1]
            if 0 < len(var) <= max_diff and all(len(cols[i]) == 2 for i in var):
                n = 1 << (xs[0].raw[3] & 15)
                k = xs[0].raw[8] ^ 0x58
                base = bytearray(xs[0].raw)
                for i in var:
                    base[i] = cols[i][0]
                combos = np.arange(1 << len(var))[:, None] >> np.arange(len(var)) & 1
                rawB = np.array([[cols[i][0], cols[i][1]] for i in var])
                vals = np.where(combos == 1, rawB[:, 1], rawB[:, 0])  # (2^v, v)
                ref = np.frombuffer(bytes(base), np.uint8).astype(np.int64)
                tot = np.full(len(combos), base[2] + base[3] + base[4] +
                              int((ref[9:9 + n] ^ k).sum()))
                cklo = np.full(len(combos), base[9 + n])
                ckhi = np.full(len(combos), base[10 + n])
                for j, i in enumerate(var):
                    if 9 <= i < 9 + n:
                        tot += (vals[:, j] ^ k) - (base[i] ^ k)
                    elif i == 9 + n:
                        cklo = vals[:, j]
                    elif i == 10 + n:
                        ckhi = vals[:, j]
                hit = np.nonzero((tot & 0xFFFF) == (cklo | ckhi << 8))[0]
                if len(hit) == 1:          # nur eindeutige Loesungen
                    for j, i in enumerate(var):
                        base[i] = int(vals[hit[0], j])
                    good = bytes(base)
        # 2) Anfang/Rest aus verschiedenen Kopien, ausgefallenes Bild
        if good is None:
            for a in xs:
                for b in xs:
                    if a is b or good:
                        continue
                    good = _repair_pair(a.raw, b.raw)
        if good:
            pay, ok = descramble(good)
            if not ok:
                continue
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


def chvd_kopf(b):
    """Kopf einer Kaufsoftware-Datei ('CHVDFILE', 32 Bytes), wie sie der
    ShareShop des CHANNEL VIDEODAT Ende 1993 verschickte:
       0  'CHVDFILE'
       8  04          Version
       9  ..          unbekannt (je Datei verschieden)
      10  20          DOS-Dateiattribut (Archiv)
      11  uint16 Zeit, uint16 Datum (DOS) der enthaltenen Datei
      15  uint32      Laenge der enthaltenen Datei
      19  '!' Name\\0 (Rest bis Byte 31 ohne Bedeutung)
      32  Inhalt, verschluesselt in 8-Byte-Bloecken (Laenge auf 8
          aufgerundet); lesbar nur mit der Empfangssoftware und
          einer KontoDisk, mit der die Software bezahlt wurde.
    Rueckgabe dict oder None."""
    if len(b) < 32 or b[:8] != b"CHVDFILE":
        return None
    t, d, n = struct.unpack_from("<HHI", b, 11)
    name = b[20:32].split(b"\0")[0].decode("ascii", "replace")
    return dict(version=b[8], name=name, laenge=n, zeit=dos_time(d << 16 | t),
                bloecke8=(n + 7) // 8)


def safe_name(b):
    n = b.split(b"\0")[0].decode("cp437", "replace")
    n = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", n).strip(" .")
    return n or "unbenannt"


NAME_RE = re.compile(rb"[A-Za-z0-9_!$#&@%'(){}~^`\-]{1,8}"
                     rb"(\.[A-Za-z0-9_!$#&@%'(){}~^`\-]{0,3})?")
NAME_CH = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
              b"0123456789._$!#&-")


def header_fields(p):
    """Felder eines Dateikopfs (Typ 02/0C) oder None, wenn unplausibel."""
    if len(p) < 15 or p[0] not in (0x02, 0x0C):
        return None
    size = struct.unpack_from("<I", p, 5)[0]
    nl = p[13]
    name = p[14:14 + nl]
    if not (1 <= nl <= 12) or len(name) != nl or size > MAX_FILE:
        return None
    return name, size, struct.unpack_from("<I", p, 9)[0]


def vote_header(copies):
    """Bytewise Mehrheitsentscheid ueber defekte Kopien eines Dateikopfs.
    Bei Gleichstand wird ein fuer Dateinamen typisches Zeichen bevorzugt."""
    L = min(len(c) for c in copies)
    out = bytearray(L)
    for i in range(L):
        cnt = collections.Counter(c[i] for c in copies)
        best = max(cnt.values())
        cands = [v for v, n in cnt.items() if n == best]
        if len(cands) > 1 and 14 <= i:
            nice = [v for v in cands if v in NAME_CH]
            if nice:
                cands = nice
        out[i] = cands[0]
    return bytes(out)


def unwrap_counter(xs, back=16, fwd=128):
    """Paketzaehler C (8 Bit) in Empfangsreihenfolge entfalten.

    Normal sind Vorwaertsspruenge (verlorene Pakete, auch Aufnahmepausen,
    in denen die Sendung weiterlief) und Ruecksprunge um wenige Pakete
    (jede Vierergruppe wird zweimal gesendet). Ein groesserer Ruecksprung
    bedeutet einen Schnitt im Band bzw. einen Neustart der Sendung: Dann
    beginnt ein neuer Abschnitt, der mit dem vorherigen nicht ueberlappen
    darf. Einzelne Pakete mit abweichendem
    Zaehler, auf die keine passenden Pakete folgen, gelten als Ausreisser
    (u = None)."""
    def jump(c_from, c_to):
        d = (c_to - c_from) & 0xFF
        return d - 256 if d > 128 else d

    res = []
    last = None            # (C, u)
    maxu = None
    for i, x in enumerate(xs):
        if last is None:
            u = x.C
        else:
            d = jump(last[0], x.C)
            if -back <= d <= fwd:
                u = last[1] + d
            else:
                # Schnitt oder Ausreisser? Folgende Pakete pruefen
                follow = [jump(x.C, y.C) for y in xs[i + 1:i + 4]]
                if follow and sum(-back <= f <= fwd for f in follow) >= \
                        max(1, len(follow) - 1):
                    u = maxu + 1024 + (x.C - maxu) % 256   # neuer Abschnitt
                else:
                    res.append(None)
                    continue
        last = (x.C, u)
        maxu = u if maxu is None else max(maxu, u)
        res.append(u)
    return res


def rescue_headers(bad, okx, oku, seq, blksz_hint, log, A):
    """Defekte Dateikoepfe (Pruefsumme falsch) retten.

    bad: defekte Pakete dieses Stroms mit Kopfstruktur
    okx/oku: gute Pakete dieses Stroms (Empfangsreihenfolge) + entfaltete
    Zaehler; seq: u -> Nutzlast der guten Pakete.
    Ein Kopf wird nur uebernommen, wenn
      - Name/Groesse plausibel sind,
      - der Zaehler zum nachfolgenden guten Paket passt (hoechstens 1 Bit
        Abweichung) und an dieser Stelle noch kein guter Kopf liegt,
      - die Anzahl Datenbloecke bis zur naechsten Endemarke (Typ 03) zur
        Dateigroesse passt (falls eine Endemarke empfangen wurde)."""
    import bisect
    pos = [x.pos for x in okx]
    cand = collections.defaultdict(list)
    for x in bad:
        if header_fields(x.pay) is None:
            continue
        j = bisect.bisect_right(pos, x.pos)
        if j >= len(okx) or okx[j].pos - x.pos > 4000:
            continue
        # Position des Kopfs: zwischen vorherigem und folgendem guten Paket.
        # Bevorzugt die Stelle, deren Zaehler exakt passt (Folgepakete koennen
        # fehlen), sonst direkt vor dem Folgepaket bei hoechstens 1 Bitfehler.
        lo_u = oku[j - 1] + 1 if j > 0 else oku[j] - 8
        u = None
        for w in range(oku[j] - 1, max(lo_u, oku[j] - 8) - 1, -1):
            if (w & 0xFF) == x.C:
                u = w
                break
        if u is None:
            u = oku[j] - 1
            if bin((x.C ^ u) & 0xFF).count("1") > 1:
                continue
        if seq.get(u, b"\xff")[0] in (0x02, 0x0C):
            continue                           # guter Kopf vorhanden
        if any(seq.get(w, b"\x04")[0] != 0x04 for w in range(u + 1, oku[j])):
            continue
        cand[u].append(x.pay)
    rescued = {}
    for u, copies in cand.items():
        p = vote_header(copies) if len(copies) > 1 else copies[0]
        hf = header_fields(p)
        if hf is None or not NAME_RE.fullmatch(hf[0]):
            continue
        name, size, ts = hf
        if dos_time(ts) is None:
            continue
        # Plausibilitaet ueber die Endemarke
        nblk = -(-size // blksz_hint) if blksz_hint else None
        end = None
        for v in range(u + 1, u + 1 + (nblk or 0) + 64):
            t = seq.get(v)
            if t is not None and t[0] in (0x02, 0x03, 0x0C):
                end = v
                break
        if end is not None and seq[end][0] == 0x03 and nblk is not None:
            if end - u - 1 != nblk:
                continue
        rescued[u] = p
        log("  Strom %02X: defekter Dateikopf repariert: %s (%d Bytes, "
            "%d Kopie%s)" % (A, safe_name(name), size, len(copies),
                              "n" if len(copies) > 1 else ""))
    return rescued


def assemble(pk, outdir, keep_partial, dump_other, log, file_manifest=None):
    os.makedirs(outdir, exist_ok=True)
    subs = collections.defaultdict(list)
    badsubs = collections.defaultdict(list)
    for x in pk:
        if x.ok:
            subs[(x.A, x.B >> 7)].append(x)
        elif x.pay and x.pay[0] in (0x02, 0x0C):
            badsubs[(x.A, x.B >> 7)].append(x)
    results = []
    files = {}
    n_rescued = 0
    for (A, sub), xs in sorted(subs.items()):
        us = unwrap_counter(xs)
        keep = [(x, u) for x, u in zip(xs, us) if u is not None]
        xs = [x for x, u in keep]
        us = [u for x, u in keep]
        seq = {}
        positions = collections.defaultdict(list)
        votes = collections.defaultdict(collections.Counter)
        for x, u in keep:
            positions[u].append(x.pos)
            seq.setdefault(u, x.pay)
            # unreparierte Pakete zaehlen doppelt
            votes[u][x.pay] += 2 if getattr(x, "how", "ok") in ("ok", "unstuffed") else 1
        bl = collections.Counter(len(p) - 1 for p in seq.values() if p[0] == 4)
        # Blockgroesse = groesster Datenblock im Strom (normal 255). Der letzte
        # Block einer Datei wird oft in einem kleineren Paket gesendet und darf
        # nicht die Blockgroesse bestimmen.
        blksz_hint = max(bl) if bl else 255
        resc = rescue_headers(badsubs.get((A, sub), []), xs, us, seq,
                              blksz_hint, log, A)
        n_rescued += len(resc)
        seq.update(resc)
        order = sorted(seq)
        other = collections.Counter()
        orphan_runs = []          # Daten ohne Dateikopf, in Abschnitten
        orphan = None
        cur = None

        def finish(cur):
            """Datei-Uebertragung abschliessen und mit frueheren
            Aussendungen derselben Datei (Name, Groesse, Zeit) vereinen."""
            if cur is None:
                return
            if cur["nblk"] is None:
                # Dateikopf ohne ein einziges empfangenes Datenpaket
                cur["blksz"] = blksz_hint
                cur["nblk"] = -(-cur["size"] // blksz_hint) if blksz_hint else 0
            key = (cur["name"], cur["size"], cur["mtime"])
            f = files.get(key)
            if f is None:
                f = files[key] = dict(cur, blocks={}, count=0, A=A, header_positions=[])
            f["header_positions"].extend(cur["header_positions"])
            if not f["blocks"] and cur["blocks"]:
                f["blksz"], f["nblk"] = cur["blksz"], cur["nblk"]
            if cur["blocks"] and f["blksz"] != cur["blksz"]:
                return               # unterschiedliche Blockgroesse: ignorieren
            f["count"] += 1
            for i, cnt in cur["blocks"].items():
                f["blocks"].setdefault(i, collections.Counter()).update(cnt)

        for u in order:
            p = seq[u]
            t = p[0]
            if t in (0x02, 0x0C) and len(p) >= 14:
                finish(cur)
                cur = None
                orphan = None
                size = struct.unpack_from("<I", p, 5)[0]
                nl = p[13]
                if size > MAX_FILE:
                    log("  Strom %02X: Dateikopf mit unplausibler Groesse "
                        "%d ignoriert" % (A, size))
                    continue
                cur = dict(name=safe_name(p[14:14 + nl]), size=size,
                           header_positions=positions.get(u, []),
                           mtime=dos_time(struct.unpack_from("<I", p, 9)[0]),
                           start=u, blocks={}, blksz=blksz_hint,
                           nblk=-(-size // blksz_hint) if blksz_hint else 0)
            elif t == 0x04:
                blk = p[1:]
                i = None
                if cur is not None:
                    i = u - cur["start"] - 1
                    if not 0 <= i < cur["nblk"]:
                        # Daten jenseits des Dateiendes: Kopf der naechsten
                        # Datei fehlt -> als Daten ohne Dateikopf sammeln
                        finish(cur)
                        cur = None
                        i = None
                    else:
                        cur["blocks"][i] = collections.Counter(
                            {p[1:]: n for p, n in votes[u].items()
                             if p[0] == 0x04})
                        if i == cur["nblk"] - 1:
                            finish(cur)
                            cur = None
                if i is None:
                    if orphan is not None and orphan and \
                            u - max(k for k in orphan if k != "_ende") > 512:
                        orphan = None          # Schnitt im Band
                    if orphan is None:
                        orphan = {}
                        orphan_runs.append(orphan)
                    orphan[u] = blk
            elif t == 0x03:
                # Endemarke / Ankuendigung der naechsten Datei
                finish(cur)
                cur = None
                if orphan is not None:
                    orphan["_ende"] = u
                orphan = None
            else:
                other[t] += 1
                if dump_other:
                    sd = os.path.join(outdir, "andere_dienste")
                    os.makedirs(sd, exist_ok=True)
                    with open(os.path.join(sd, "strom_%02X_typ_%02X.bin" %
                                           (A, t)), "ab") as f:
                        f.write(p)
            if t == 0x00 and len(p) > 9 and 2 < p[7] < 40 and \
                    all(32 <= c < 127 for c in p[8:8 + p[7]]):
                info = p[8:8 + p[7]].decode("ascii")
                log("  Strom %02X: Infoblock '%s'" % (A, info))
        finish(cur)
        if keep_partial:
            for run in orphan_runs:
                end = run.pop("_ende", None)
                if not run:
                    continue
                # Daten ohne Dateikopf (Kopf vor Aufnahmebeginn gesendet
                # oder nicht zu retten)
                sd = os.path.join(outdir, "strom_%02X" % A)
                os.makedirs(sd, exist_ok=True)
                u0, u1 = min(run), max(run)
                bs = max(len(b) for b in run.values())
                path = os.path.join(sd, "ohne_dateikopf_%d.bin.unvollstaendig"
                                    % u0)
                with open(path, "wb") as f:
                    for u in range(u0, u1 + 1):
                        f.write(run.get(u, bytes(bs)))
                log("  Strom %02X/%d: %d Datenbloecke ohne Dateikopf%s -> %s" %
                    (A, sub, len(run),
                     ", Dateiende empfangen" if end == u1 + 1 else "",
                     os.path.basename(path)))
        if other:
            log("  Strom %02X/%d: andere Pakettypen %s" %
                (A, sub, ", ".join("%02X:%d" % kv for kv in sorted(other.items()))))

    # Dateien schreiben (gleichnamige, aber verschiedene Dateien bekommen
    # einen Zaehler angehaengt)
    used = collections.Counter()
    conflicts = 0
    for (name, size, mtime), f in files.items():
        data = bytearray()
        missing = 0
        missing_ranges, conflicting_ranges = [], []
        for i in range(f["nblk"]):
            cnt = f["blocks"].get(i)
            blk = None
            if cnt:
                blk = cnt.most_common(1)[0][0]
                if len(cnt) > 1:
                    conflicts += 1
                    conflicting_ranges.append([i * f["blksz"], min(size, (i + 1) * f["blksz"])])
            if blk is None:
                missing += 1
                blk = bytes(f["blksz"])
                missing_ranges.append([i * f["blksz"], min(size, (i + 1) * f["blksz"])])
            elif len(blk) < min(f["blksz"], size - i * f["blksz"]):
                missing_ranges.append([i * f["blksz"] + len(blk), min(size, (i + 1) * f["blksz"])])
            data += blk[:f["blksz"]].ljust(f["blksz"], b"\0")
        data = data[:size]
        complete = missing == 0 and len(data) == size
        used[(f["A"], name.lower())] += 1
        fname = name
        if used[(f["A"], name.lower())] > 1:
            stem, ext = os.path.splitext(name)
            fname = "%s_%d%s" % (stem, used[(f["A"], name.lower())], ext)
        path = None
        if complete or keep_partial:
            if not complete:
                fname += ".unvollstaendig"
            sd = os.path.join(outdir, "strom_%02X" % f["A"])
            os.makedirs(sd, exist_ok=True)
            path = os.path.join(sd, fname)
            with open(path, "wb") as fh:
                fh.write(data)
            if mtime:
                try:
                    t = mtime.timestamp()
                    os.utime(path, (t, t))
                except (OverflowError, OSError, ValueError):
                    pass
        results.append((f["A"], name, size, missing, f["nblk"], path,
                        f["count"]))
        if file_manifest is not None:
            file_manifest.append(dict(stream=f["A"], name=name, size=size, path=path,
                                      header_positions=f["header_positions"],
                                      missing_ranges=missing_ranges,
                                      conflicting_ranges=conflicting_ranges))
    if n_rescued:
        log("  %d defekte Dateikoepfe repariert" % n_rescued)
    if conflicts:
        log("  %d Dateibloecke lagen in abweichenden Versionen vor "
            "(Mehrheitsentscheid)" % conflicts)
    return results


# ---------------------------------------------------------------------------
# Andere Dienste (neben der Dateiuebertragung)
#
# Durch Analyse einer Aufnahme vom 11.04.1993 rekonstruiert:
#   Strom 01 Typ 06  Uhrzeit: 06 JJ MM TT hh mm ss 00, alle 5 Sekunden
#   Strom 01 Typ 05  Im Sample exakt 64 LE16-PRNG-Werte (15 Bit), Byte 0
#                    mit Typ 05 ueberschrieben; siehe videodat_prng.py.
#                    Andere Aufnahmen werden auf exakte Uebereinstimmung geprueft.
#   Strom 03 Typ 07/08  Kontoguthaben, VideoDat-DES mit 8 Runden
#   Strom 03 Typ 0A  Dispo-Aenderung, 16 Byte, VideoDat-DES mit 8 Runden
#                    (aus WINVD.EXE rekonstruiert; DESCRAMBLER_NOTES.md)
#   Strom 00 Typ 00  Bitlisten in 32-Byte-Bloecken:
#                    00 [Liste] [Block LE16] 00 [Anzahl LE16] Daten
#                    Daten komprimiert: 00 n = n Nullbytes, FF n = n FF-
#                    Bytes, sonst Byte unveraendert. Liste 00 und 11 haben
#                    genau 65536 Bit; die Listen 03/10/11 sind (fast)
#                    Teilmengen von Liste 00 -> vermutlich Freischaltung je
#                    Dienst, ein Bit pro Decoder-/Teilnehmernummer.
#   Strom 00 Typ 01/02/81/82  Steuermeldung [Typ][Strom]: 82/81 = Beginn/
#                    Ende einer Sendung im Unterkanal 1 (z.B. 82 94, 81 F5),
#                    02/01 entsprechend im Unterkanal 0 (Bedeutung unsicher)
#   In Sendungsstroemen (Unterkanal 1):
#     Typ 00  Sendungskopf: 00 [DOS-Zeit 4] [2 Byte] [Laenge] [Kennung]
#     Typ 01  Sendungsende: 01 [DOS-Zeit 4] ...
# ---------------------------------------------------------------------------
def _unrun(b):
    out = bytearray()
    i = 0
    while i < len(b):
        if b[i] in (0x00, 0xFF):
            if i + 1 >= len(b):
                break
            out += bytes([b[i]]) * b[i + 1]
            i += 2
        else:
            out.append(b[i])
            i += 1
    return out


def _dos_dt(b):
    t, d = struct.unpack("<HH", bytes(b[:4]))
    try:
        return datetime.datetime(1980 + (d >> 9), (d >> 5) & 15, d & 31,
                                 t >> 11, (t >> 5) & 63, (t & 31) * 2)
    except ValueError:
        return None


class Uhr:
    """Bytepositionen -> Uhrzeit, aus den Uhrzeit-Paketen."""

    def __init__(self, ticks):
        self.t = sorted(ticks)                     # (pos, datetime)
        rates = []
        for (p0, t0), (p1, t1) in zip(self.t, self.t[1:]):
            dt = (t1 - t0).total_seconds()
            if 0 < dt <= 30:
                rates.append((p1 - p0) / dt)
        self.rate = float(np.median(rates)) if rates else None

    def __call__(self, pos):
        if not self.t or not self.rate:
            return None
        i = int(np.searchsorted([p for p, _ in self.t], pos))
        cand = [self.t[k] for k in (i - 1, i) if 0 <= k < len(self.t)]
        p, t = min(cand, key=lambda c: abs(c[0] - pos))
        return t + datetime.timedelta(seconds=(pos - p) / self.rate)

    def schnitte(self):
        """Luecken der Aufnahme: Uhr springt weiter als die Bytes erlauben."""
        out = []
        for (p0, t0), (p1, t1) in zip(self.t, self.t[1:]):
            dt = (t1 - t0).total_seconds()
            erwartet = (p1 - p0) / self.rate
            if dt - erwartet > 30 or dt < 0:
                out.append((p0, t0, t1, dt - erwartet))
        return out


def analyse_dienste(pk, outdir, log):
    ok = [x for x in pk if x.ok and x.pay]
    os.makedirs(outdir, exist_ok=True)
    rep = []
    w = rep.append
    hm = lambda t: t.strftime("%H:%M:%S") if t else "?"

    # --- Uhrzeit --------------------------------------------------------
    ticks = []
    for x in ok:
        p = x.pay
        if x.A == 0x01 and p[0] == 0x06 and len(p) == 8:
            try:
                ticks.append((x.pos, datetime.datetime(
                    1900 + p[1], p[2], p[3], p[4], p[5], p[6])))
            except ValueError:
                pass
    # Ausreisser (falsche Pruefsumme durchgerutscht) entfernen
    ticks.sort()
    gut = []
    for k, (pos, t) in enumerate(ticks):
        nb = [tt for _, tt in ticks[max(0, k - 3):k + 4] if tt is not t]
        if sum(abs((t - tt).total_seconds()) < 900 for tt in nb) >= \
                min(2, len(nb)):
            gut.append((pos, t))
    uhr = Uhr(gut)
    w("UHRZEIT (Strom 01, Typ 06)")
    if gut:
        w("  %d Zeitstempel, %s bis %s%s" %
          (len(gut), gut[0][1].strftime("%d.%m.%Y %H:%M:%S"),
           gut[-1][1].strftime("%H:%M:%S"),
           ", %.0f Bytes/s" % uhr.rate if uhr.rate else ""))
        for p0, t0, t1, fehl in uhr.schnitte():
            w("  Aufnahmeluecke bei Byte %d: %s -> %s (ca. %d s fehlen)" %
              (p0, hm(t0), hm(t1), fehl))
    else:
        w("  keine Uhrzeit-Pakete gefunden")

    # --- Sendungsablauf ---------------------------------------------------
    w("")
    w("SENDUNGSABLAUF")
    ereig = []
    for x in ok:
        p = x.pay
        if x.A == 0x00 and p[0] in (0x01, 0x02, 0x81, 0x82) and len(p) == 8:
            art = {0x82: "Beginn", 0x81: "Ende", 0x02: "Beginn?",
                   0x01: "Ende?"}[p[0]]
            ereig.append((x.pos, "Steuerung: %-7s Strom %02X" % (art, p[1])))
        elif x.B & 0x80 and p[0] == 0x00 and len(p) > 9:
            n = p[7]
            name = bytes(p[8:8 + n]).decode("cp437", "replace").strip()
            ereig.append((x.pos, "Strom %02X Sendungskopf '%s', Sendezeit %s"
                          % (x.A, name, hm(_dos_dt(p[1:5])))))
        elif x.B & 0x80 and p[0] == 0x01 and len(p) == 8:
            ereig.append((x.pos, "Strom %02X Sendungsende, Zeit %s" %
                          (x.A, hm(_dos_dt(p[1:5])))))
    letzte = None
    for pos, txt in sorted(ereig):
        if txt == letzte:
            continue                      # Wiederholung desselben Pakets
        letzte = txt
        w("  %s  %s" % (hm(uhr(pos)), txt))
    w("")
    w("  Aktive Datenstroeme (Dateiuebertragung):")
    akt = collections.defaultdict(list)
    for x in ok:
        if x.A != 0x00 and x.pay[0] in (0x02, 0x03, 0x04, 0x0C):
            akt[(x.A, x.B >> 7)].append(x.pos)
    for (A, sub), ps in sorted(akt.items()):
        if len(ps) < 3:
            continue
        w("    Strom %02X/%d: %s - %s, %d Pakete" %
          (A, sub, hm(uhr(min(ps))), hm(uhr(max(ps))), len(ps)))

    # --- Bitlisten ----------------------------------------------------
    W = 32
    lists = collections.defaultdict(dict)
    for x in ok:
        p = x.pay
        if x.A == 0x00 and p[0] == 0x00 and len(p) >= 16:
            blk = p[2] | p[3] << 8
            n = p[5] | p[6] << 8
            u = _unrun(p[7:])
            for k in range(n):
                seg = bytes(u[k * W:(k + 1) * W])
                if len(seg) == W:
                    lists[p[1]].setdefault(blk + k,
                                           collections.Counter())[seg] += 1
    w("")
    w("BITLISTEN (Strom 00, Typ 00) - vermutlich Freischaltung je Dienst,")
    w("ein Bit je Decoder-/Teilnehmernummer (Bit 7 des ersten Bytes = Nr. 0)")
    bits = {}
    for lid, d in sorted(lists.items()):
        H = max(d) + 1
        data = b"".join(d[y].most_common(1)[0][0] if y in d else bytes(W)
                        for y in range(H))
        fehl = H - len(d)
        b = np.unpackbits(np.frombuffer(data, np.uint8))
        bits[lid] = b
        name = "bitliste_%02X.bin" % lid
        open(os.path.join(outdir, name), "wb").write(data)
        nz = np.nonzero(b)[0]
        w("  Liste %02X: %d Bit (%d Bloecke, %d fehlen), %d gesetzt, "
          "Nummern %s -> %s" %
          (lid, len(b), H, fehl, len(nz),
           ("%d-%d" % (nz[0], nz[-1])) if len(nz) else "-", name))
        try:
            from PIL import Image
            h = (len(b) + 255) // 256
            img = np.zeros(h * 256, np.uint8)
            img[:len(b)] = b * 255
            Image.fromarray(255 - img.reshape(h, 256)).save(
                os.path.join(outdir, "bitliste_%02X.png" % lid))
        except ImportError:
            pass
    if 0x00 in bits:
        null = bits[0x00] == 0
        nz = np.nonzero(null)[0]
        if len(nz):
            w("    Liste 00: %d Nummern NICHT gesetzt (%d-%d); davon in anderen "
              "Listen: %s -> Liste 00 wirkt wie eine Sperrliste "
              "(0 = gesperrt)" % (len(nz), nz[0], nz[-1], ", ".join(
                  "%02X: %d" % (k, int((bits[k][:len(null)] &
                                        null[:len(bits[k])]).sum()))
                  for k in sorted(bits) if k)))
    for a in sorted(bits):
        for c in sorted(bits):
            if a == c:
                continue
            n = min(len(bits[a]), len(bits[c]))
            ba, bc = bits[a][:n], bits[c][:n]
            if ba.sum() and (ba & ~bc).sum() <= 0.1 * ba.sum():
                w("    Liste %02X ist (fast) Teilmenge von Liste %02X "
                  "(%d von %d Bit ausserhalb)" %
                  (a, c, int((ba & ~bc).sum()), int(ba.sum())))

    # --- verschluesselte / unbekannte Dienste ------------------------------
    w("")
    w("WEITERE DIENSTE (Rohdaten in dienst_*.bin)")
    gruppen = collections.defaultdict(list)
    for x in ok:
        p = x.pay
        if (x.A, p[0]) in ((0x01, 0x05), (0x03, 0x07), (0x03, 0x08), (0x03, 0x0A)):
            gruppen[(x.A, p[0])].append(x)
    texte = {(0x01, 0x05): "unbekannt; jedes 2. Byte 7 Bit (15-Bit-"
             "Werte), keine Wiederholungen",
             (0x03, 0x07): "Kontoguthaben; VideoDat-DES, 8 Runden",
             (0x03, 0x08): "Kontoguthaben; VideoDat-DES, 8 Runden",
             (0x03, 0x0A): "Dispo-Aenderung; VideoDat-DES, 8 Runden"}
    for (A, t), xs in sorted(gruppen.items()):
        verschieden = collections.OrderedDict()
        for x in xs:
            verschieden.setdefault(bytes(x.pay), []).append(x.pos)
        name = "dienst_%02X_typ%02X.bin" % (A, t)
        with open(os.path.join(outdir, name), "wb") as f:
            for pay in verschieden:
                f.write(struct.pack("<H", len(pay)) + pay)
        if (A, t) == (0x01, 0x05):
            import json
            from videodat_prng import analyse_type05
            matched = 0
            classifications = collections.Counter()
            report_name = "dienst_01_typ05_analysis.jsonl"
            with open(os.path.join(outdir, report_name), "w", encoding="utf-8") as f:
                for pay, positions in verschieden.items():
                    try:
                        record = analyse_type05(pay)
                    except ValueError as error:
                        record = dict(kind="unknown", matches_prng=False, error=str(error))
                    matched += int(record["matches_prng"])
                    classifications[record["kind"]] += 1
                    record.update(positions=positions, copies=len(positions))
                    f.write(json.dumps(record) + "\n")
            texte[(A, t)] = "%d/%d exakt passende PRNG-Folgen; %s -> %s" % (
                matched, len(verschieden), dict(classifications), report_name)
        w("  Strom %02X Typ %02X: %d Pakete, %d verschieden, %s - %s; %s"
          " -> %s" % (A, t, len(xs), len(verschieden), hm(uhr(xs[0].pos)),
                      hm(uhr(xs[-1].pos)), texte[(A, t)], name))
    w("  (Format der .bin: je Paket 2 Byte Laenge LE + Nutzlast)")
    with open(os.path.join(outdir, "dienste.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    log("")
    for line in rep:
        log(line)
    return uhr


def main():
    ap = argparse.ArgumentParser(description="Pro7-VideoDat (1990/1993) "
                                 "aus VBI")
    ap.add_argument("input", help="VBI-Rohdatei (bttv), TBC-Datei "
                    "(vhs-decode, Endung .tbc) oder mit --bytes "
                    "ein bereits dekodierter Byte-Strom")
    ap.add_argument("--format", choices=["auto", "vbi", "tbc"],
                    default="auto", help="Eingabeformat (auto: nach Endung)")
    ap.add_argument("--variant", choices=["auto", "1993", "1990"],
                    default="auto", help="Uebertragungsvariante: 1993 = "
                    "2 Mbit/s, 10 Bytes je Zeile; 1990 = 800 kbit/s, 4 Bytes "
                    "je Zeile (auto: aus dem Signal; mit --bytes gilt 1993)")
    ap.add_argument("--tbc-width", type=int, default=1135,
                    help="TBC: Samples pro Zeile (PAL 1135)")
    ap.add_argument("--tbc-field-lines", type=int, default=313,
                    help="TBC: Zeilen pro Halbbild (PAL 313)")
    ap.add_argument("--no-sync", action="store_true",
                    help="TBC: H-Sync nicht als Zeitbezug verwenden")
    ap.add_argument("--no-eq", action="store_true",
                    help="TBC: keinen Entzerrer lernen (nur adaptive Schwelle)")
    ap.add_argument("--no-line-tracking", action="store_true",
                    help="VBI: Zeilen nicht je Halbbild nachfuehren, sondern "
                    "immer die erkannten bzw. angegebenen Zeilen dekodieren")
    ap.add_argument("--scan-seconds", type=float, default=30.0,
                    help="Zeilensuche hoechstens ueber so viele Sekunden "
                    "vom Aufnahmeanfang (bricht frueher ab, sobald die "
                    "Zeilen sicher erkannt sind)")
    ap.add_argument("-o", "--outdir", default="videodat1993_out")
    ap.add_argument("--bytes", action="store_true",
                    help="Eingabe ist ein Byte-Strom (z.B. aus --save-bytes)")
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
    ap.add_argument("--keine-dienste", action="store_true",
                    help="andere Dienste (Uhrzeit, Sendungsablauf, Bitlisten, "
                    "verschluesselte Daten) nicht auswerten")
    ap.add_argument("--dump-other", action="store_true",
                    help="Pakete anderer Dienste roh ablegen")
    ap.add_argument("--decrypt-services", action="store_true",
                    help="Strom 03 Typ 07/08/0A entschluesseln und mit innerer "
                    "Pruefsumme als JSONL protokollieren")
    ap.add_argument("--decrypt-files", action="store_true",
                    help="CHVDFILE-Dateien mit Schluesseln aus langen Typ-01-"
                    "Endpaketen entschluesseln; Luecken bleiben markiert")
    args = ap.parse_args()
    log = print

    fmt = args.format
    if fmt == "auto":
        fmt = "tbc" if args.input.lower().endswith(".tbc") else "vbi"
    lines = [int(v) for v in args.lines.split(",")] if args.lines else None
    cap = None
    if args.bytes:
        stream = open(args.input, "rb").read()
        variant = "1990" if args.variant == "1990" else "1993"
    else:
        if fmt == "tbc":
            cap = read_tbc(args.input, args.tbc_width, args.tbc_field_lines,
                           lines, not args.no_sync, log, args.scan_seconds,
                           not args.no_eq, args.variant)
        else:
            cap = read_vbi(args.input, args.line_length,
                           args.lines_per_frame, lines, log,
                           args.scan_seconds, not args.no_line_tracking,
                           args.variant)
        variant = cap.fmt.variant

    if variant == "1990":
        import videodat1990
        if cap is not None:
            channels = videodat1990.split_channels(cap, log)
        else:
            channels = [("bytes", stream)]
        if args.save_bytes and cap is not None:
            stem, ext = os.path.splitext(args.save_bytes)
            for label, st in channels:
                fn = "%s_%s%s" % (stem, label, ext)
                open(fn, "wb").write(st.data)
                log("Byte-Strom %s -> %s" % (label, fn))
        res = videodat1990.run(channels, args, log)
        log("")
        log("Dateien:")
        for A, name, size, missing, nblk, path, count in res:
            st = "vollstaendig" if missing == 0 else \
                "unvollstaendig (%d von %d Bloecken fehlen)" % (missing, nblk)
            rep = "  (%dx gesendet)" % count if count > 1 else ""
            log("  Kanal %02X  %-14s %8d Bytes  %s%s%s" %
                (A, name, size, st, rep, "" if path else
                 "  [nicht geschrieben]"))
        log("%d Dateien, davon %d vollstaendig" %
            (len(res), sum(1 for r in res if r[3] == 0)))
        return

    if cap is not None:
        stream = cap.stream()
    if args.save_bytes:
        open(args.save_bytes, "wb").write(
            stream.data if isinstance(stream, Stream) else stream)

    pk = find_packets(stream)
    n_ok = sum(x.ok for x in pk)
    n_unstuffed = sum(x.how == "unstuffed" for x in pk)
    fixed = repair(pk)
    log("Pakete: %d gefunden, %d Pruefsumme ok, repariert: %d per "
        "Soft-Decision, %d aus Kopien" %
        (len(pk), n_ok, fixed["soft"], fixed["kopien"]))
    if n_unstuffed:
        log("  %d Pakete nach Entfernen von Sync-Escape-Bytes gueltig" % n_unstuffed)
    file_manifest = [] if args.decrypt_files else None
    res = assemble(pk, args.outdir, args.partial, args.dump_other, log, file_manifest)
    if args.decrypt_files:
        from videodat_files import write_file_report
        file_counts = write_file_report(pk, file_manifest,
                                       os.path.join(args.outdir, "decrypted_files"))
        log("Datei-Entschluesselung: %s" % file_counts)
    log("")
    log("Dateien:")
    for A, name, size, missing, nblk, path, count in res:
        st = "vollstaendig" if missing == 0 else \
            "unvollstaendig (%d von %d Bloecken fehlen)" % (missing, nblk)
        rep = "  (%dx gesendet)" % count if count > 1 else ""
        log("  Strom %02X  %-14s %8d Bytes  %s%s%s" %
            (A, name, size, st, rep, "" if path else "  [nicht geschrieben]"))
        if path:
            with open(path, "rb") as fh:
                k = chvd_kopf(fh.read(32))
            if k:
                z = k["zeit"]
                log("            Kaufsoftware (CHVDFILE): enthaelt %s, %d Bytes"
                    "%s, verschluesselt" %
                    (k["name"], k["laenge"],
                     z.strftime(" vom %d.%m.%Y %H:%M") if z else ""))
    n_ok = sum(1 for r in res if r[3] == 0)
    log("%d Dateien, davon %d vollstaendig" % (len(res), n_ok))
    if not args.keine_dienste:
        analyse_dienste(pk, os.path.join(args.outdir, "dienste"), log)
    if args.decrypt_services:
        from videodat_crypto import write_service_report
        service_dir = os.path.join(args.outdir, "dienste")
        os.makedirs(service_dir, exist_ok=True)
        report = os.path.join(service_dir, "decrypted_services.jsonl")
        counts = write_service_report(pk, report)
        log("Entschluesselung: %d innere Pruefsummen ok, %d falsch, "
            "%d ungueltige Pakete -> %s" %
            (counts["passed"], counts["failed"], counts["malformed"], report))


if __name__ == "__main__":
    main()
