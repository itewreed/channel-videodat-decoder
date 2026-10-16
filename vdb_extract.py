#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vdb_extract.py - Inhalte aus VideoDat-Datenbanken (.VDB/.IDX/.TTL, Pro7 1993)
herausloesen und die Bildschirmseiten als Bilder nachbauen, auch aus
unvollstaendig empfangenen Dateien.

Aufbau einer .VDB (durch Analyse rekonstruiert):
  Folge von Datensaetzen, jeder beginnt auf einer 16-Byte-Grenze:
    0  uint32  Laenge L des Inhalts
    4  uint16  0
    6  uint16  Satznummer
    8  uint32  Flags (oberes Byte: 00 Bild, 01 Seite/Text, 02 Programm-
               eintrag, 04 Begleitdokument; 00100000 bei Satz 1 =
               Seitenfolge)
   12  L Bytes Inhalt:
        'VDPC'  Bild: Version (00/01), Zahl der Bitebenen (04 = 16 Farben,
                01 = einfarbig), Breite, Hoehe (je uint16), danach die
                Bilddaten zeilenweise je Ebene ceil(Breite/8) Bytes,
                EGA-Standardpalette.
                Version 00: PCX-Lauflaengen (Byte >= C0: Anzahl = Byte & 3F,
                naechstes Byte = Wert). Version 01 (ab Mitte 1993): GEM-IMG-
                artig mit Zeilenwiederholung und Musterlaeufen, Ebenen
                invertiert (rle2_decode). Jede Bildzeile beginnt dort mit
                einem neuen Token; das erlaubt nach Luecken einen exakten
                Wiedereinstieg am Zeilenanfang.
        'VR'    Bildschirmseite 640x480: Texte + Objektbaum im Stil von
                GEM-Resourcen (Beschreibung weiter unten bei 'Seite').
        'VT'    Seitenvorlage, gleicher Aufbau, mit Platzhaltern
                (DDDDDD, NNNNNN, ZZZ, BBBB...).
        Text    Fliesstext (CP437, Zeilenumbruch 0D) fuer Textfelder.
        94 Byte Programmeintrag: 00 80, Preis (Pfennig) und Groesse (kB)
                als zwei uint16, DOS-Zeit, Kennung, weitere DOS-Zeiten,
                Titel.
        'DT'    Begleitdokument (Text, CP437, CR/LF), z. B. ein Formular;
                eine Schaltflaeche vom Typ 0x2C auf der Seite druckt es.
        Satz 1  Liste der Seitennummern in Anzeigereihenfolge.
  Die .IDX beginnt mit 'GS' 01, danach je 3 Bytes (little endian) der
  Offset von Satz 1, 2, 3, ... in der .VDB.
  Die .TTL ist der Programmkatalog (Kennung, Groesse, Sendetermine, Titel).

Fehlende Bloecke (vom Decoder mit Nullbytes aufgefuellte 255-Byte-Bloecke)
werden erkannt. Mehrere Empfangskopien derselben Datei werden Byte fuer Byte
zusammengefuehrt. In Bildern wird nach einer Luecke die Lage der folgenden
Daten ueber die Aehnlichkeit benachbarter Bildzeilen bestimmt; fehlende
Bildteile werden grau dargestellt.

Aufruf:
  python3 vdb_extract.py DATEI.VDB [weitere Kopien.VDB ...] [DATEI.IDX]
                         [DATEI.TTL] [-o ausgabeordner] [--software ORDNER]
Ausgabe: inhalt.txt, bild_*.png, text_*.txt, seite_*.png / vorlage_*.png
(nachgebaute Seiten), seite_*.txt, datei_*.txt (Begleitdokumente),
seiten.html (anklickbare Uebersicht), sendungen.txt (Programmliste mit
Preis und Groesse).
Originalgetreue Seiten: mit --software ORDNER (Programmordner von
VD 3.3 bzw. WinVideodat 3.2 mit XFONTS, GMICONS.DAT, GM8X16_*.FNT) werden
die Original-Bitmapschriften und -Symbole verwendet; der Ordner wird auch
neben der VDB und neben diesem Skript gesucht. Farbtabelle und Fuellmuster
stammen aus VDGEM.DLL.
Benoetigt: numpy, Pillow; ohne Originalschriften eine TrueType-Schrift
(Liberation/DejaVu/Arial, wird automatisch gesucht).
"""

import argparse
import os
import re
import struct

import numpy as np

BLK = 255                       # Blockgroesse der VideoDat-Uebertragung
EGA = [(0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170), (170, 0, 0),
       (170, 0, 170), (170, 85, 0), (170, 170, 170), (85, 85, 85),
       (85, 85, 255), (85, 255, 85), (85, 255, 255), (255, 85, 85),
       (255, 85, 255), (255, 255, 85), (255, 255, 255)]
GRAU = (128, 128, 128)          # Farbe fuer fehlende Bildteile
POPC = np.array([bin(i).count("1") for i in range(256)], np.float32)


# ---------------------------------------------------------------------------
def missing_mask(data, gaps_file=None):
    """Bool-Array: True = Byte stammt aus einem fehlenden Block."""
    miss = np.zeros(len(data), bool)
    if gaps_file and os.path.exists(gaps_file):
        for line in open(gaps_file):
            m = re.match(r"\s*(\d+)\s*-\s*(\d+)", line)
            if m:
                miss[int(m.group(1)):int(m.group(2))] = True
        return miss
    a = np.frombuffer(data, np.uint8)
    for k in range(0, len(data), BLK):
        if not a[k:k + BLK].any():
            miss[k:k + BLK] = True
    return miss


def read_idx(path):
    """IDX lesen: 'GS' 01, danach je 3 Bytes (little endian) den Offset des
    Datensatzes mit der Nummer 1, 2, 3, ... in der VDB (0 ist ein gueltiger
    Offset, kein Trenner). Rueckgabe: {Satznummer: Offset}."""
    d = open(path, "rb").read()
    if d[:2] != b"GS":
        return {}
    return {k + 1: int.from_bytes(d[i:i + 3], "little")
            for k, i in enumerate(range(3, len(d) - 2, 3))}


def merge_copies(datas):
    """Mehrere Empfangskopien derselben Datei Byte fuer Byte zusammenfuehren.
    Fehlende Bloecke einer Kopie werden aus den anderen gefuellt; weichen
    vorhandene Bytes voneinander ab, entscheidet die Mehrheit (bei
    Gleichstand die zuerst angegebene Datei).
    Rueckgabe: (Daten, Fehlmaske, Anzahl abweichender Bytes)."""
    n = max(len(d) for d in datas)
    arrs, masks = [], []
    for d in datas:
        a = np.zeros(n, np.uint8)
        a[:len(d)] = np.frombuffer(d, np.uint8)
        m = np.ones(n, bool)
        m[:len(d)] = missing_mask(d)
        arrs.append(a)
        masks.append(m)
    A, M = np.array(arrs), np.array(masks)
    votes = np.zeros(A.shape, np.int32)
    for i in range(len(A)):
        for j in range(len(A)):
            votes[i] += (~M[j]) & (A[j] == A[i])
    votes[M] = -1
    best = votes.argmax(0)
    out = A[best, np.arange(n)]
    miss = M.all(0)
    present = (~M).sum(0)
    differ = int(((present >= 2) & (votes.max(0) < present)).sum())
    return bytes(out), miss, differ


# Satzarten (oberstes Byte bzw. oberes Halbbyte der Flags)
FLAG_ARTEN = {0x000, 0x001, 0x010, 0x014, 0x020, 0x040, 0x800}


def record_kind(L, nr, flags, rec):
    if rec[:4] == b"VDPC":
        return "bild"
    if rec[:2] == b"VR":
        return "seite"
    if rec[:2] == b"VT":
        return "vorlage"
    if L == 94 and (flags >> 24 == 2 or rec[1:2] == b"\x80"):
        return "sendung"
    if nr == 1 and (flags == 0x00100000 or (
            L >= 4 and L == 4 + 4 * struct.unpack_from("<I", rec)[0])):
        return "liste"
    if flags >> 24 == 1:
        return "text"
    if flags >> 24 == 4 and rec[:2] == b"DT":
        return "datei"
    return "unbekannt"


def find_records(data, miss, idx=None):
    """Datensaetze finden: alle 16-Byte-Grenzen mit plausiblem Kopf, dazu
    die Offsets aus der IDX. Rueckgabe {Offset: dict} bzw. None, wenn der
    Kopf eines laut IDX vorhandenen Satzes fehlt."""
    idx = idx or {}
    idx_offs = {o: nr for nr, o in idx.items() if o + 16 <= len(data)}
    recs = {}
    for off in sorted(set(range(0, len(data) - 16, 16)) | set(idx_offs)):
        if miss[off:off + 12].any():
            if off in idx_offs:
                recs[off] = None
            continue
        L, z, nr, fl = struct.unpack_from("<IHHI", data, off)
        ok = (z == 0 and 1 <= L < 4000000 and off + 12 + L <= len(data) + BLK
              and 1 <= nr < 0x8000 and fl & 0xFFFFF == 0
              and fl >> 20 in FLAG_ARTEN)
        if not ok:
            if off in idx_offs:
                recs[off] = None
            continue
        rec = data[off + 12:off + 12 + L]
        kind = record_kind(L, nr, fl, rec)
        if off not in idx_offs and (kind == "unbekannt" or
                                    (idx and nr > max(idx))):
            continue
        recs[off] = dict(off=off, L=L, nr=nr, flags=fl, kind=kind)
    return recs


def records_by_nr(recs, miss, idx=None):
    """Satznummer -> Offset. Die IDX hat Vorrang; sonst wird von mehreren
    Saetzen gleicher Nummer (alte Fassungen im freien Speicher) der mit den
    wenigsten fehlenden Bytes genommen."""
    best = {}
    for off, r in recs.items():
        if r is None:
            continue
        lost = int(miss[off:off + 12 + r["L"]].sum())
        if r["nr"] not in best or lost < best[r["nr"]][0]:
            best[r["nr"]] = (lost, off)
    out = {nr: off for nr, (lost, off) in best.items()}
    for nr, off in (idx or {}).items():
        if off in recs and recs[off] is not None and recs[off]["nr"] == nr:
            out[nr] = off
    return out


def dos_zeit(b):
    """4 Bytes DOS-Zeit+Datum (little endian) -> 'TT.MM.JJJJ hh:mm'."""
    t, dt = struct.unpack("<HH", b)
    if not dt or (dt >> 5) & 15 not in range(1, 13):
        return None
    return "%02d.%02d.%04d %02d:%02d" % (dt & 31, (dt >> 5) & 15,
                                         1980 + (dt >> 9), t >> 11,
                                         (t >> 5) & 63)


def dos_zeit_ok(b):
    """Wie dos_zeit, aber nur fuer plausible Werte (1990-2000)."""
    if len(b) != 4:
        return None
    t, dt = struct.unpack("<HH", b)
    if not (10 <= dt >> 9 <= 20 and 1 <= (dt >> 5) & 15 <= 12
            and 1 <= dt & 31 <= 31 and t >> 11 < 24 and (t >> 5) & 63 < 60
            and t & 31 < 30):
        return None
    return dos_zeit(b)


def cstr(b, p, maxlen=200):
    e = b.find(b"\0", p, p + maxlen)
    return b[p:e if e >= 0 else p + maxlen].decode("cp437", "replace")


def preis_groesse(v):
    """Feld 'Groesse' der Programmeintraege: oberes Wort = Preis in Pfennig
    (KontoDisk), unteres Wort = Dateigroesse in kB (aufgerundet auf 1000
    Bytes). Geprueft an acht empfangenen Dateien (Groesse exakt) und an
    Preisangaben in den Beschreibungstexten (DM 1,00 / 1,50 / 1,99).
    Rueckgabe (Pfennig oder None, kB)."""
    if v >= 0x10000:
        return v >> 16, v & 0xFFFF
    return None, v


def parse_sendung(rec):
    """94-Byte-Programmeintrag: Preis und Groesse (siehe preis_groesse),
    bis zu vier Sendetermine, Kennung und Titel."""
    size = struct.unpack_from("<I", rec, 2)[0]
    zeiten = [z for z in (dos_zeit(rec[p:p + 4]) for p in (6, 39, 43, 47))
              if z]
    return dict(kennung=cstr(rec, 10, 17), groesse=size, zeiten=zeiten,
                titel=cstr(rec, 53, 40))


def read_ttl(path):
    """Programmkatalog (.TTL): 'V0E', dann je Eintrag
    [n][Kennung n Bytes] 80 [Groesse 1-2 Bytes] [k x DOS-Zeit] [b]
    [DOS-Zeit Katalog] [b] [m][Titel m Bytes].
    Wurde die Datei unterwegs von Windows-1252 nach UTF-8 umkodiert, wird
    das rueckgaengig gemacht."""
    b = open(path, "rb").read()
    if b"\xe2\x82\xac" in b or b"\xc2" in b[:200]:
        try:
            out = bytearray()
            for ch in b.decode("utf-8"):
                try:
                    out += ch.encode("cp1252")
                except UnicodeEncodeError:
                    out.append(ord(ch))
            b = bytes(out)
        except UnicodeDecodeError:
            pass
    if b[:2] != b"V0":
        return []
    # Fehlen alle Nullbytes, wurden sie unterwegs entfernt (z. B. beim
    # Hochladen als Text); Zeitangaben werden dann auch 3-bytig gesucht.
    ohne_null = b.count(0) == 0
    alle = []

    def is_id(p):
        n = b[p] if p < len(b) else 0
        return 4 <= n <= 20 and all(c in b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                    b"0123456789_ " for c in b[p + 1:p + 1 + n])
    out, i = [], 3
    while i < len(b) and is_id(i):
        n = b[i]
        kennung = b[i + 1:i + 1 + n].decode("ascii")
        j = k = i + 1 + n
        while k < len(b):
            m = b[k]
            if (2 <= m < 80 and k + 1 + m <= len(b)
                    and all(c >= 32 for c in b[k + 1:k + 1 + m])
                    and (k + 1 + m == len(b) or is_id(k + 1 + m))):
                break
            k += 1
        seg = b[j:k]
        titel = b[k + 1:k + 1 + b[k]].decode("cp437") if k < len(b) else ""
        size, zeiten, p, groesse_bytes = 0, [], 1, b""
        while p < len(seg):
            if ohne_null and p < 2:          # mind. ein Byte Groesse
                groesse_bytes += seg[p:p + 1]
                p += 1
                continue
            z = dos_zeit_ok(seg[p:p + 4])
            if z:
                zeiten.append(z)
                p += 4
                continue
            if ohne_null:
                z = (dos_zeit_ok(b"\0" + seg[p:p + 3]) or
                     dos_zeit_ok(seg[p:p + 1] + b"\0" + seg[p + 1:p + 3]))
                if z and p + 3 <= len(seg):
                    zeiten.append(z)
                    p += 3
                    continue
            if not zeiten:
                groesse_bytes += seg[p:p + 1]
            p += 1
        size = int.from_bytes(groesse_bytes[:4], "little")
        alle.append(zeiten)
        out.append(dict(kennung=kennung, groesse=size, zeiten=zeiten,
                        titel=titel))
        i = k + 1 + (b[k] if k < len(b) else 0)
    # Der in (fast) allen Eintraegen gleiche letzte Zeitstempel ist das
    # Erstellungsdatum des Katalogs, kein Sendetermin.
    from collections import Counter
    letzte = Counter(z[-1] for z in alle if z)
    if letzte:
        kat, n = letzte.most_common(1)[0]
        if n > len(out) // 2:
            for e in out:
                if e["zeiten"] and e["zeiten"][-1] == kat:
                    e["zeiten"].pop()
            if out:
                out[0]["katalog"] = kat
    return out


def rle_decode(b):
    out = bytearray()
    i = 0
    n = len(b)
    while i < n:
        c = b[i]
        i += 1
        if c >= 0xC0:
            if i >= n:
                break
            out += bytes([b[i]]) * (c & 0x3F)
            i += 1
        else:
            out.append(c)
    return out


def rle2_decode(b, rowlen):
    """Bildkompression Version 1 (VDPC 01 xx, ab etwa Mitte 1993), ein
    GEM-IMG-artiges Verfahren. rowlen = Bytes einer Bildzeile ueber alle
    Bitebenen (Ebenenzahl * ceil(Breite/8)).
    00 00 FF k  die folgende Bildzeile (alle Ebenen) steht k+1-mal da
    00 n p1 p2  Muster p1 p2 n-mal (Raster-/Ditherflaechen)
    01..7F      n Bytes 00
    80 n ...    n Bytes unveraendert uebernehmen
    81..FE      (n-80h) Bytes FF
    Die Bitebenen sind invertiert gespeichert; zurueckgegeben werden sie
    wie bei Version 0 (1 = Bit gesetzt). Ueberzaehlige Zeilen am Ende
    schneidet der Aufrufer ab."""
    b = bytes(b)
    n = len(b)

    def lauf(i, dst):
        t = b[i]
        if t == 0:
            if i + 3 < n and b[i + 1]:
                dst += b[i + 2:i + 4] * b[i + 1]
                return i + 4
            return i + 1
        if t == 0x80:
            m = b[i + 1] if i + 1 < n else 0
            dst += b[i + 2:i + 2 + m]
            return i + 2 + m
        if t == 0xFF:
            return i + 1                 # kommt nicht vor; ueberspringen
        if t > 0x80:
            dst += b"\xff" * (t - 0x80)
        else:
            dst += b"\x00" * t
        return i + 1

    out = bytearray()
    i = 0
    while i < n:
        if b[i] == 0 and i + 3 < n and b[i + 1] == 0 and b[i + 2] == 0xFF:
            k = b[i + 3]
            i += 4
            zeile = bytearray()
            while len(zeile) < rowlen and i < n:
                i = lauf(i, zeile)
            out += bytes(zeile[:rowlen]) * (k + 1)
            if len(zeile) > rowlen:          # gestoerte Daten: Rest behalten
                out += zeile[rowlen:]
        else:
            i = lauf(i, out)
    return bytearray(x ^ 0xFF for x in out)


def _v1_token(b, i, n):
    """Ein Token der Version-1-Kompression: (naechste Position, Bytes)."""
    t = b[i]
    if t == 0:
        if i + 3 < n and b[i + 1]:
            return i + 4, 2 * b[i + 1]
        return i + 1, 0
    if t == 0x80:
        m = b[i + 1] if i + 1 < n else 0
        return i + 2 + m, min(m, max(0, n - i - 2))
    if t == 0xFF:
        return i + 1, 0
    return i + 1, (t - 0x80 if t > 0x80 else t)


def v1_zeilenanfang(b, rowlen, bpl=None, mindest=3):
    """Bei Version 1 beginnt jede Bildzeile mit einem neuen Token (an allen
    vollstaendigen Bildern geprueft; Ebenengrenzen dagegen nicht). Nach
    einer Luecke liefert das die genaue Lage innerhalb der Zeile: gesucht
    ist der Rest modulo rowlen, auf den in jeder Zeile eine Tokengrenze
    faellt. In gleichfoermigen Bildteilen erfuellen das auch Ebenengrenzen;
    dann gibt es mehrere Kandidaten. Rueckgabe: Liste der moeglichen
    Startpositionen (im entpackten Abschnitt) einer vollen Bildzeile,
    jeweils die frueheste, ab der alle Zeilenanfaenge Tokengrenzen sind;
    [] wenn nichts sicher ist."""
    b = bytes(b)
    n = len(b)
    i = pos = 0
    g = [0]
    while i < n:
        if b[i] == 0 and i + 3 < n and b[i + 1] == 0 and b[i + 2] == 0xFF:
            k = b[i + 3]
            i += 4
            z = 0
            while z < rowlen and i < n:
                i, d = _v1_token(b, i, n)
                z += d
            g.extend(pos + z * j for j in range(1, k + 2))
            pos += z * (k + 1)
            continue
        i, d = _v1_token(b, i, n)
        pos += d
        if d:
            g.append(pos)
    warm = min(pos // 4, 2 * rowlen)       # Anfang noch nicht eingerastet
    zeilen = (pos - warm) // rowlen
    if zeilen < mindest:
        return []
    gs = set(g)
    treffer = np.bincount([x % rowlen for x in set(g) if warm <= x < pos],
                          minlength=rowlen)
    kand = []
    for r in np.nonzero(treffer >= 0.95 * zeilen)[0]:
        x = max(x for x in g if x % rowlen == r and x <= pos)
        start = x
        while x - rowlen >= 0 and x - rowlen in gs:
            x -= rowlen
            start = x
        kand.append(int(start))
    return sorted(kand)

def v1_ebene_waehlen(kand, dec, ref, bpl):
    """Unter den Startkandidaten den waehlen, dessen erste Zeilen in allen
    Ebenen am besten zu den Zeilen ref (n, rowlen) darueber passen."""
    if len(kand) == 1 or ref is None or not len(ref):
        return kand[0], 0.0
    line = ref.shape[1]
    a = plane_profiles(ref.astype(np.uint8), bpl)
    decb = np.frombuffer(bytes(dec), np.uint8)
    best = (kand[0], -2.0)
    for z in kand:
        nrow = min(3, (len(decb) - z) // line)
        if nrow < 1:
            continue
        b = plane_profiles(decb[z:z + nrow * line].reshape(nrow, line), bpl)
        cs = []
        for i in range(len(a)):
            if a[i].std() < 1e-6 or b[i].std() < 1e-6:
                cs.append(1.0 if abs(a[i].mean() - b[i].mean()) < 0.3
                          else 0.0)
            else:
                cs.append(float(np.corrcoef(a[i], b[i])[0, 1]))
        sc = float(np.mean(cs))
        if sc > best[1]:
            best = (z, sc)
    return best

def bild_ebenen(rec):
    """Zahl der Bitebenen laut VDPC-Kopf (1 = einfarbig, 4 = 16 Farben)."""
    p = rec[5] if len(rec) > 5 else 4
    return p if 1 <= p <= 4 else 4


def image_decoder(rec):
    """Passenden Entpacker zur Versionskennung des VDPC-Kopfes."""
    if rec[4:5] == b"\x01":
        w = struct.unpack_from("<H", rec, 6)[0]
        rowlen = bild_ebenen(rec) * ((w + 7) // 8)
        return lambda b: rle2_decode(b, rowlen)
    return rle_decode


def ebenen_zu_index(a, w):
    """(Zeilen, Ebenen, bpl) Bytes -> Farbindex je Pixel (Zeilen, w)."""
    bits = np.unpackbits(a, axis=2)[:, :, :w].astype(np.int16)
    idx = np.zeros(bits[:, 0].shape, np.int16)
    for e in range(a.shape[1]):
        idx |= bits[:, e] << e
    if a.shape[1] == 1:
        idx *= 15                       # einfarbig: gesetzt = weiss
    return idx


def best_phase(seg, line):
    """Lage (mod Zeilenlaenge), bei der benachbarte Zeilen am aehnlichsten
    sind. Gemessen an der Bitdichte je Byte, geglaettet (robust gegen
    Rasterung). Liefert (phase, guete 0..1)."""
    a = POPC[np.frombuffer(bytes(seg), np.uint8)]
    if len(a) < 3 * line:
        return None, 0.0
    k = np.ones(3) / 3
    a = np.convolve(a, k, "same")
    best = (None, -1.0)
    for ph in range(line):
        rows = (len(a) - ph) // line
        if rows < 3:
            continue
        m = a[ph:ph + rows * line].reshape(rows, line)
        x = m[1:] - m[1:].mean(axis=1, keepdims=True)
        y = m[:-1] - m[:-1].mean(axis=1, keepdims=True)
        den = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
        ok = den > 1e-9
        if not ok.any():
            continue
        s = float(((x * y).sum(axis=1)[ok] / den[ok]).mean())
        if s > best[1]:
            best = (ph, s)
    return best


def color_hist(buf, line, bpl, w):
    """Farbhistogramm (16 Klassen) eines zeilenausgerichteten Puffers."""
    rows = len(buf) // line
    if rows == 0:
        return None
    a = np.frombuffer(bytes(buf[:rows * line]), np.uint8)
    idx = ebenen_zu_index(a.reshape(rows, line // bpl, bpl), w)
    hst = np.bincount(idx.ravel(), minlength=16).astype(float)
    return hst / max(hst.sum(), 1)


def plane_profiles(rows, bpl, win=4):
    """rows: (n, 4*bpl) Bytes -> (4, bpl) geglaettete Bitdichte je Ebene."""
    npl = rows.shape[1] // bpl
    pc = POPC[rows].reshape(rows.shape[0], npl, bpl).mean(axis=0)
    k = np.ones(win) / win
    return np.array([np.convolve(pc[i], k, "same") for i in range(npl)])


def plane_distance(out, known, dec, p, line, maxgap=80, nref=4):
    """Vergleicht die Helligkeitsverlaeufe der vier Bitebenen in den ersten
    vollen Zeilen des Abschnitts (an Position p) mit den letzten bekannten
    Zeilen darueber. Liefert 1 - mittlere Korrelation (kleiner = besser)."""
    bpl = line // 4
    off = (-p) % line
    r0 = (p + off) // line
    nrow = min(nref, (len(dec) - off) // line)
    if nrow < 1:
        return None
    rows_known = known.reshape(-1, line).all(axis=1)
    above = np.nonzero(rows_known[:r0])[0]
    if not len(above) or r0 - above[-1] > maxgap:
        return None
    ref = out.reshape(-1, line)[above[-nref:]].astype(np.uint8)
    seg = np.frombuffer(bytes(dec[off:off + nrow * line]), np.uint8)
    seg = seg.reshape(nrow, line)
    a = plane_profiles(ref, bpl)
    b = plane_profiles(seg, bpl)
    cs = []
    for i in range(4):
        if a[i].std() < 1e-6 or b[i].std() < 1e-6:
            cs.append(1.0 if abs(a[i].mean() - b[i].mean()) < 0.5 else 0.0)
        else:
            cs.append(float(np.corrcoef(a[i], b[i])[0, 1]))
    return 1.0 - float(np.mean(cs))


def hist_distance(out, known, dec, p, line, bpl, w):
    """Abstand der Farbverteilung des Abschnitts (an Position p) zur
    Verteilung des bereits bekannten Bildteils."""
    ref_rows = np.nonzero(known.reshape(-1, line).all(axis=1))[0]
    if len(ref_rows) == 0:
        return 0.0
    ref = out.reshape(-1, line)[ref_rows].astype(np.uint8).tobytes()
    off = (-p) % line                     # erste volle Zeile im Abschnitt
    seg = bytes(dec[off:])
    h1 = color_hist(ref, line, bpl, w)
    h2 = color_hist(seg, line, bpl, w)
    if h1 is None or h2 is None:
        return 0.0
    return float(np.abs(h1 - h2).sum())


def _prof(v):
    """geglaettete Bitdichte eines Byte-Vektors (letzte Achse)."""
    p = POPC[v]
    return (p + np.roll(p, 1, -1) + np.roll(p, -1, -1)) / 3


def _corr(x, y):
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    return np.where(den > 1e-9, (x * y).sum(-1) / np.maximum(den, 1e-9), 0.0)


def place_tracked(out, known, dec, p, line, total, D=48, gain=0.25,
                  minc=0.45):
    """Abschnitt ab Position p ins Bild schreiben. Vor jeder neuen Bildzeile
    wird geprueft, ob eine Verschiebung um bis zu D Bytes den Anschluss an
    die Zeile darueber deutlich verbessert (unerkannt verfaelschte
    Lauflaenge). Liefert (Endposition, Anzahl Korrekturen)."""
    dec = np.frombuffer(bytes(dec), np.uint8)
    c = 0
    q = p
    nfix = 0
    # angebrochene erste Zeile
    first = min((-q) % line, len(dec), total - q)
    if first:
        out[q:q + first] = dec[:first]
        known[q:q + first] = True
        c += first
        q += first
    while c < len(dec) and q < total:
        n = min(line, len(dec) - c, total - q)
        if D and n == line and q >= line and known[q - line:q].all() and \
                c + D + line <= len(dec):
            above = _prof(out[q - line:q].astype(np.uint8))
            lo = max(-D, -c)
            offs = np.arange(lo, D + 1)
            win = np.lib.stride_tricks.sliding_window_view(
                dec[c + lo:c + D + line], line)[:len(offs)]
            cs = _corr(_prof(win), above[None, :])
            i0 = int(np.nonzero(offs == 0)[0][0])
            ib = int(np.argmax(cs))
            if offs[ib] != 0 and cs[ib] > minc and cs[ib] - cs[i0] > gain:
                c += int(offs[ib])
                nfix += 1
        n = min(line, len(dec) - c, total - q)
        if n <= 0:
            break
        out[q:q + n] = dec[c:c + n]
        known[q:q + n] = True
        c += n
        q += n
    return q, nfix


def place_segment(out, known, dec, est, pos, line, bpl, w, total,
                  maxgap=80):
    """Startposition eines Abschnitts nach einer Luecke bestimmen.

    Bevorzugt: Vergleich der Bitdichte-Verlaeufe aller vier Ebenen der
    ersten Zeilen des Abschnitts mit den letzten bekannten Zeilen darueber,
    fuer jede moegliche Lage innerhalb der Zeile. Die Zeilennummer wird aus
    der Menge der verlorenen Daten geschaetzt."""
    if len(dec) < line + 8:
        return None, 0, ""
    est = max(est, pos)
    r_est = est // line
    rows_known = known.reshape(-1, line).all(axis=1)
    above = np.nonzero(rows_known[:max(r_est, 1)])[0]
    decb = np.frombuffer(bytes(dec), np.uint8)
    if len(above) and r_est - above[-1] <= maxgap:
        ref = out.reshape(-1, line)[above[-3:]].astype(np.uint8)
        a = plane_profiles(ref, bpl)                    # (4, bpl)
        best = None
        nrow = 3
        for off in range(line):                         # erste volle Zeile
            if off + nrow * line > len(decb):
                nrow = max(1, (len(decb) - off) // line)
                if nrow < 1 or off + line > len(decb):
                    break
            seg = decb[off:off + nrow * line].reshape(nrow, line)
            b = plane_profiles(seg, bpl)
            cs = []
            for i in range(len(a)):
                if a[i].std() < 1e-6 or b[i].std() < 1e-6:
                    cs.append(1.0 if abs(a[i].mean() - b[i].mean()) < 0.3
                              else 0.0)
                else:
                    cs.append(float(np.corrcoef(a[i], b[i])[0, 1]))
            sc = float(np.mean(cs))
            if best is None or sc > best[0]:
                best = (sc, off)
        sc, off = best
        if sc < 0.3:
            return None, sc, ""
        # Position p so, dass dec[off] auf einen Zeilenanfang faellt
        r = max(r_est, above[-1] + 1)
        p = r * line - off
        if p < pos:
            p += line * (-(-(pos - p) // line))
        if p + len(dec) > total + line:
            p = max(pos, total - len(dec))
            p -= (p + off) % line
        return p, sc, "Anschluss an Zeile %d (Guete %.2f)" % (above[-1], sc)
    return place_by_edges(decb, est, pos, line, bpl, total)


def place_by_edges(decb, est, pos, line, bpl, total):
    """Ersatzverfahren ohne bekannte Zeile darueber: An jeder Ebenen- und
    Zeilengrenze springt das Bild vom rechten an den linken Rand; die
    Bitdichte aendert sich dort im Mittel staerker als innerhalb der Zeile.
    Bestimmt die Lage modulo bpl; die Ebene wird nicht bestimmt (0)."""
    rows = len(decb) // line
    if rows < 4:
        return None, 0, ""
    pc = POPC[decb[:rows * line]]
    dif = np.abs(np.diff(pc))
    prof = np.array([dif[j::bpl].mean() for j in range(bpl)])
    j = int(np.argmax(prof))
    sc = float((prof[j] - prof.mean()) / (prof.std() + 1e-9))
    if sc < 4:
        return None, 0, ""
    off = (j + 1) % bpl                 # dec[off] liegt auf einer Ebenengrenze
    r = max(est // line, -(-pos // line))
    p = r * line - off
    while p < pos:
        p += line
    return p, 0.0, "Ebenengrenzen (Sicherheit gering)"


def find_in_refs(dec, refs, probe=96):
    """Eindeutige Position eines Abschnitts in einer vollstaendigen
    Wiederholung desselben Bildes (zwei unabhaengige Stichproben)."""
    db = bytes(dec)
    if len(db) < 3 * probe:
        return None
    for ref in refs:
        pts = []
        for m in (len(db) // 3, 2 * len(db) // 3):
            pr = db[m:m + probe]
            t = ref.find(pr)
            if t < 0 or ref.find(pr, t + 1) >= 0:
                pts = None
                break
            pts.append(t - m)
        if pts and pts[0] == pts[1]:
            return pts[0]
    return None


def decode_image(rec, rec_miss, refs=()):
    """rec = Inhalt ab 'VDPC', rec_miss = Fehlend-Maske dazu.
    Liefert (Breite, Hoehe, Farbindex-Array mit -1 = unbekannt, Info)."""
    w, h = struct.unpack_from("<HH", rec, 6)
    bpl = (w + 7) // 8
    npl = bild_ebenen(rec)
    line = npl * bpl
    total = line * h
    comp = rec[10:]
    cm = rec_miss[10:]
    decoder = image_decoder(rec)
    if not cm.any():
        # vollstaendig empfangen: direkt entpacken, keine Heuristik
        dec = np.frombuffer(bytes(decoder(comp)), np.uint8)
        if rec[4:5] == b"\x01" and len(dec) > total and \
                (len(dec) - total) % line == 0:
            # Manche Bilder enthalten eine Zeile zu viel; sie gehoert an
            # den oberen Rand (Vergleich mit derselben Grafik in Version 0
            # aus einer anderen Aufnahme)
            dec = dec[len(dec) - total:]
        out = np.zeros(total, np.uint8)
        known = np.zeros(total, bool)
        out[:min(total, len(dec))] = dec[:total]
        known[:len(dec)] = True
        idx = ebenen_zu_index(out.reshape(h, npl, bpl), w)
        kn = np.repeat(known.reshape(h, npl, bpl).all(axis=1), 8,
                       axis=1)[:, :w]
        idx[~kn] = -1
        info = "" if len(dec) >= total else \
            "Bilddaten enden nach Zeile %d" % (len(dec) // line)
        return w, h, idx, info
    # bekannte Abschnitte des komprimierten Stroms
    segs = []
    i = 0
    while i < len(comp):
        if cm[i]:
            i += 1
            continue
        j = i
        while j < len(comp) and not cm[j]:
            j += 1
        segs.append((i, j))
        i = j
    out = np.zeros(total, np.int16)
    known = np.zeros(total, bool)
    info = []
    if not segs:
        return w, h, None, "keine Bilddaten empfangen"
    ratio = None
    pos = 0
    drift = 0
    v1 = rec[4:5] == b"\x01"
    # Version 1 mit bekanntem Anfang und Ende: die Lage der Abschnitte
    # dazwischen folgt aus dem freien Platz, aufgeteilt im Verhaeltnis der
    # verlorenen Datenmengen (Zeilenanfaenge sind exakt bekannt)
    plan, starts = {}, {}
    if v1 and len(segs) >= 3 and segs[0][0] == 0 and \
            segs[-1][1] == len(comp):
        laengen = []
        d0 = decoder(comp[segs[0][0]:segs[0][1]])
        nr0 = len(d0) // line
        ref0 = np.frombuffer(bytes(d0[max(0, nr0 - 3) * line:nr0 * line]),
                             np.uint8).reshape(-1, line)
        for k, (s, e) in enumerate(segs):
            d = decoder(comp[s:e])
            if k == 0:
                z = 0
            else:
                kand = v1_zeilenanfang(comp[s:e], line, bpl)
                if not kand:
                    laengen = None
                    break
                if k == len(segs) - 1:
                    passt = [z for z in kand if (len(d) - z) % line == 0]
                    z = passt[0] if passt else kand[0]
                else:
                    z = v1_ebene_waehlen(kand, d, ref0, bpl)[0]
            laengen.append(len(d) - z)
            starts[k] = z
        if laengen:
            frei = total - laengen[-1] - sum(laengen[:-1])
            verl = [segs[k][0] - segs[k - 1][1] for k in range(1, len(segs))]
            if frei >= 0 and sum(verl):
                q = laengen[0]
                acc = 0
                for k in range(1, len(segs) - 1):
                    acc += verl[k - 1]
                    p = laengen[0] + sum(laengen[1:k]) + frei * acc / sum(verl)
                    p = max(int(round(p / line)) * line, -(-q // line) * line)
                    plan[k] = p
                    q = p + laengen[k]
    for k, (s, e) in enumerate(segs):
        dec = decoder(comp[s:e])
        if k == 0 and s == 0:
            p = 0
        else:
            lost = s - (segs[k - 1][1] if k else 0)
            last = (k == len(segs) - 1 and e == len(comp))
            z0 = None
            if k in starts:
                z0 = starts[k]
            elif v1:
                kand = v1_zeilenanfang(comp[s:e], line, bpl)
                if kand and last:
                    passt = [z for z in kand if (len(dec) - z) % line == 0]
                    z0 = passt[0] if passt else None
                elif kand:
                    rk = known.reshape(-1, line).all(axis=1)
                    ob = np.nonzero(rk)[0]
                    ref = out.reshape(-1, line)[ob[-3:]] if len(ob) else None
                    z0 = v1_ebene_waehlen(kand, dec, ref, bpl)[0]
            if z0 is not None:
                # Bytes vor dem ersten sicheren Zeilenanfang verwerfen
                # (Decoder noch nicht eingerastet); sie bleiben unbekannt
                dec = dec[z0:]
            if last and len(dec) <= total:
                p = total - len(dec)               # Ende passt genau
                how = "Ende"
                if z0 is not None:
                    how += ", Zeilenanfang exakt"
            elif k in plan and z0 is not None:
                p = plan[k]
                how = "Zeilenanfang exakt, Zeile aus den Lueckengroessen"
            elif z0 is not None and find_in_refs(dec, refs) is None:
                r_cur = len(dec) / max(e - s, 1) if e - s >= 200 else None
                est = pos + int(lost * (r_cur or ratio or 2.0))
                p = max(-(-pos // line), int(round(est / line))) * line
                how = "Zeilenanfang exakt, Zeile geschaetzt"
            elif find_in_refs(dec, refs) is not None:
                p = find_in_refs(dec, refs)
                how = "exakt ueber Wiederholung des Bildes"
            else:
                # verlorene Bildbytes schaetzen: Kompressionsrate des
                # folgenden Abschnitts (aehnelt meist dem verlorenen Teil)
                r_cur = len(dec) / max(e - s, 1) if e - s >= 200 else None
                est = pos + int(lost * (r_cur or ratio or 2.0))
                p, q, how = place_segment(out, known, dec, est, pos, line,
                                          bpl, w, total)
                if p is None:
                    info.append("Abschnitt %d nicht einzuordnen" % k)
                    continue
            info.append("Luecke vor Byte %d: weiter bei Bildzeile %d (%s)" %
                        (s, p // line, how))
        if p >= total:
            continue
        # Die Zeilenkorrektur (D > 0) verschlechterte in Tests sowohl
        # vollstaendige als auch lueckenhafte Bilder; daher abgeschaltet.
        pos, nfix = place_tracked(out, known, dec, p, line, total, D=0)
        drift += nfix
        if e - s > 1000:
            ratio = len(dec) / (e - s)
    filled = 0
    for ref in refs:
        rb = np.frombuffer(ref[:total], np.uint8)
        if len(rb) < total or known.sum() < total * 0.2:
            continue
        agree = (out[known] == rb[known]).mean()
        if agree >= 0.97:
            fill = ~known
            out[fill] = rb[fill]
            filled = int(fill.sum())
            known[:] = True
            info.append("%.0f %% aus einer vollstaendigen Wiederholung "
                        "ergaenzt (Uebereinstimmung %.1f %%)" %
                        (100.0 * filled / total, 100 * agree))
            break
    idx = ebenen_zu_index(out.astype(np.uint8).reshape(h, npl, bpl), w)
    kn = known.reshape(h, npl, bpl).all(axis=1)
    kn = np.repeat(kn, 8, axis=1)[:, :w]
    idx[~kn] = -1
    if drift:
        info.append("%d Verschiebungen innerhalb von Abschnitten korrigiert"
                    % drift)
    return w, h, idx, "; ".join(info)


def save_png(idx, path):
    from PIL import Image
    pal = np.array(EGA + [GRAU], np.uint8)
    Image.fromarray(pal[np.where(idx < 0, 16, idx)]).save(path)


def texts(rec):
    """Lesbare Zeichenketten (CP437) aus einem VR/VT-Datensatz."""
    out = []
    if rec[:2] in (b"VR", b"VT") and len(rec) >= 16:
        # Kopfbytes 4..15 koennen Speicherreste enthalten; Texte stehen
        # zwischen Kopf und Objektbaum
        T = struct.unpack_from("<H", rec, 2)[0]
        rec = rec[:2] + rec[16:T if 16 < T <= len(rec) else len(rec)]
    for m in re.finditer(rb"[\x20-\x7e\x80-\xa5]{3,}", rec[2:]):
        t = m.group().decode("cp437")
        if sum(ch.isalpha() for ch in t) >= 2:
            out.append(t.strip())
    return out


# ---------------------------------------------------------------------------
# Bildschirmseiten (VR/VT) nachbauen
#
# Aufbau eines VR-/VT-Satzes (durch Analyse rekonstruiert):
#    0  'VR' bzw. 'VT'
#    2  uint16  Offset T des Objektbaums im Satz
#    4  ...     Seitenkopf (z. B. 7f 02 df 01 = 639 x 479)
#   16  Texte und Textbeschreibungen (TEDINFO-artig, Zahlen big endian)
#    T  Objektbaum wie bei GEM-Resourcen: je Objekt 24 Bytes
#         int16 BE  naechstes, erstes Kind, letztes Kind
#         uint16 BE Typ, Flags, Status
#         4 Bytes   Spezifikation (je nach Typ)
#         int16 LE  x, y, Breite, Hoehe (relativ zum Elternobjekt)
#   Verweise auf Texte im Satz: Position = Wert - 36.
#
#   Typ 0x14 Kasten       spec: 00, Randstaerke, Farbwort (BE)
#   Typ 0x16 Kastentext   spec: Verweis auf Textbeschreibung (28 Bytes):
#                         Textverweis(4) . . Schrift(2) . Ausrichtung(2)
#                         Farbwort(2) . Randstaerke(2) Textlaenge(2) .
#   Typ 0x1A Schaltflaeche spec: Zielseite (Satznr., uint16 LE), Beschriftung
#   Typ 0x21 Bild         spec: Satznummer des VDPC-Bildes (uint16 LE)
#   Typ 0x23 Symbol 32x32 spec: Symbolnummer (Symbole selbst nicht in der VDB)
#   Typ 0x25 Sendungsknopf spec: Satznr. des Programmeintrags, Beschriftung
#   Typ 0x26 Textfeld     spec: Satznummer des Textes, Textbeschreibung
#
# Farbwort wie bei GEM: Rand(4) Text(4) Modus(1) Muster(3) Fuellung(4),
# Farbnummern nach GEM-VDI (0 weiss, 1 schwarz, 2 rot, 3 gruen, 4 blau ...).
# Unsicher sind die Schriftnummern (Schriften liegen nicht in der VDB, daher
# Ersatzschriften mit an die Objektgroesse angepasster Groesse) und die
# Symbole (werden als Platzhalter gezeichnet).

# Zuordnung Objektfarbe -> EGA-Farbe, Tabelle aus VDGEM.DLL (Offset 0x269E)
# der Original-Software (weicht bei 7 und 13 von GEM-Standard ab)
VDI2EGA = [15, 0, 4, 2, 1, 3, 6, 11, 7, 8, 12, 10, 9, 5, 14, 13]
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                   [3, 11, 1, 9], [15, 7, 13, 5]])
MUSTER = [0, 2, 4, 6, 8, 11, 13, 16]          # Dichte von 16 je Muster 0..7
# Fuellmuster 1..7 aus VDGEM.DLL (Offset 0x241A, dort in umgekehrter Folge):
# je 4 Zeilen zu 8 Bildpunkten, gesetztes Bit = Fuellfarbe
MUSTER_BITS = {7: (0xff, 0xff, 0xff, 0xff), 6: (0xff, 0xdd, 0xff, 0x77),
               5: (0xff, 0x55, 0xff, 0x55), 4: (0xbb, 0x55, 0xee, 0x55),
               3: (0xaa, 0x55, 0xaa, 0x55), 2: (0xaa, 0x11, 0xaa, 0x44),
               1: (0xaa, 0x00, 0xaa, 0x00)}
GRAU_IDX = 16
WEISS, SCHWARZ, HELLGRAU = 115, 100, 107     # feste EGA-Farben (+100)

FONT_DATEIEN = {
    "sans": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf",
             "LiberationSans-Regular.ttf", "DejaVuSans.ttf", "arial.ttf"],
    "fett": ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "C:/Windows/Fonts/arialbd.ttf", "/Library/Fonts/Arial Bold.ttf",
             "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
    "mono": ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
             "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf",
             "DejaVuSansMono.ttf", "cour.ttf"],
}
_font_cache = {}


def get_font(art, size):
    from PIL import ImageFont
    key = (art, size)
    if key not in _font_cache:
        f = None
        for p in FONT_DATEIEN[art]:
            try:
                f = ImageFont.truetype(p, size)
                break
            except OSError:
                continue
        if f is None:
            try:
                f = ImageFont.load_default(size)
            except TypeError:
                f = ImageFont.load_default()
        _font_cache[key] = f
    return _font_cache[key]


def font_art(font_id):
    if font_id < 0x80:
        return "mono"                  # Systemschrift (8x16-Raster)
    return "fett" if font_id >= 0xD0 else "sans"


def text_width(text, art, size):
    f = get_font(art, size)
    try:
        return f.getlength(text)
    except AttributeError:
        return f.getsize(text)[0]


def fit_size(text, art, w, h):
    """Groesste Schriftgroesse, mit der der Text in w x h passt."""
    if art == "mono":
        s = 14
        while s > 8 and text_width(text, art, s) > w + 2:
            s -= 1
        return s
    s = max(8, int(h * 0.8))
    while s > 8 and text_width(text, art, s) > w:
        s -= 1
    return s


class Seite:
    """Ein VR-/VT-Satz mit Objektbaum."""

    def __init__(self, rec, rec_miss=None):
        self.rec = rec
        self.miss = (rec_miss if rec_miss is not None
                     else np.zeros(len(rec), bool))
        self.T = struct.unpack_from("<H", rec, 2)[0]
        self.objs = []
        for i in range((len(rec) - self.T) // 24):
            p = self.T + 24 * i
            b = rec[p:p + 24]
            nx, hd, tl, ty, fl, st = struct.unpack(">hhhHHH", b[:12])
            x, y, w, h = struct.unpack("<hhhh", b[16:])
            self.objs.append(dict(i=i, next=nx, head=hd, tail=tl, typ=ty,
                                  flags=fl, state=st, spec=b[12:16],
                                  x=x, y=y, w=w, h=h,
                                  fehlt=bool(self.miss[p:p + 24].any())))

    def ref(self, spec):
        """Verweis im unteren Wort der Spezifikation -> Position im Satz."""
        p = struct.unpack(">H", spec[2:4])[0] - 36
        return p if 0 <= p < len(self.rec) else None

    def string(self, p):
        if p is None:
            return None
        e = self.rec.find(b"\0", p, p + 300)
        if e < 0 or self.miss[p:e + 1].any():
            return None
        return self.rec[p:e].decode("cp437")

    def tedinfo(self, p):
        if p is None or p + 28 > len(self.rec) or self.miss[p:p + 28].any():
            return None
        w = struct.unpack(">14H", self.rec[p:p + 28])
        tp = (w[0] << 16 | w[1]) - 36
        return dict(text=self.string(tp) if 0 <= tp < len(self.rec) else None,
                    font=w[6], just=w[8], color=w[9], dicke=w[11])

    def walk(self):
        """Objekte in Zeichenreihenfolge mit absoluter Lage."""
        out, seen = [], set()

        def rec_(i, ox, oy, tiefe):
            if not (0 <= i < len(self.objs)) or i in seen or tiefe > 30:
                return
            seen.add(i)
            o = self.objs[i]
            ax, ay = ox + o["x"], oy + o["y"]
            out.append((o, ax, ay))
            c = o["head"]
            n = 0
            while c not in (-1, i) and 0 <= c < len(self.objs) and n < 300:
                rec_(c, ax, ay, tiefe + 1)
                c = self.objs[c]["next"]
                n += 1
        rec_(0, 0, 0, 0)
        # Fehlt ein Elternobjekt (Luecke im Empfang), die erreichbaren Reste
        # relativ zum ueblichen Seitenrahmen (7,7) zeichnen.
        self.verwaist = 0
        for o in self.objs:
            if o["i"] not in seen and not o["fehlt"] and o["typ"]:
                n0 = len(out)
                rec_(o["i"], 7, 7, 1)
                self.verwaist += len(out) - n0
        return out


class Zeichner:
    """Zeichnet in ein Feld von EGA-Farbnummern (16 = grau/fehlt)."""

    def __init__(self, w=640, h=480, palette="vdi"):
        self.c = np.zeros((h, w), np.int16)
        self.pal = palette

    def farbe(self, v):
        """Objektfarbe -> EGA-Nummer; Werte >= 100 sind feste EGA-Farben."""
        if v >= 100:
            return v - 100
        v &= 15
        return VDI2EGA[v] if self.pal == "vdi" else v

    def clip(self, x, y, w, h):
        H, W = self.c.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def fill(self, x, y, w, h, col, muster=7, ersetzen=True):
        r = self.clip(x, y, w, h)
        if not r or (muster == 0 and not ersetzen):
            return
        x0, y0, x1, y1 = r
        # Muster 0 im Ersetzen-Modus: einfarbig in der Fuellfarbe (so
        # ergeben z.B. die Handbuchseiten schwarze Schrift auf Hellgrau)
        if muster >= 7 or muster == 0:
            self.c[y0:y1, x0:x1] = self.farbe(col)
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rows = np.array(MUSTER_BITS[muster], np.uint8)
        an = ((rows[yy & 3] >> (7 - (xx & 7))) & 1).astype(bool)
        sub = self.c[y0:y1, x0:x1]
        sub[an] = self.farbe(col)
        if ersetzen:
            sub[~an] = WEISS - 100

    def rahmen(self, x, y, w, h, col, dicke):
        if dicke == 0:
            return
        if dicke > 0:                       # nach innen
            for k in range(dicke):
                self._rect(x + k, y + k, w - 2 * k, h - 2 * k, col)
        else:                               # nach aussen
            for k in range(1, -dicke + 1):
                self._rect(x - k, y - k, w + 2 * k, h + 2 * k, col)

    def _rect(self, x, y, w, h, col):
        if w <= 0 or h <= 0:
            return
        f = self.farbe(col)
        for (a, b, c, d) in ((x, y, w, 1), (x, y + h - 1, w, 1),
                             (x, y, 1, h), (x + w - 1, y, 1, h)):
            r = self.clip(a, b, c, d)
            if r:
                self.c[r[1]:r[3], r[0]:r[2]] = f

    def grau(self, x, y, w, h):
        r = self.clip(x, y, w, h)
        if r:
            self.c[r[1]:r[3], r[0]:r[2]] = GRAU_IDX

    def text(self, s, x, y, w, h, col, art, size, just=2, ega=False):
        from PIL import Image, ImageDraw
        f = get_font(art, size)
        tw = int(text_width(s, art, size)) + 2
        asc, desc = f.getmetrics()
        th = asc + desc
        m = Image.new("1", (max(1, tw), max(1, th)), 0)
        ImageDraw.Draw(m).text((0, 0), s, font=f, fill=1)
        a = np.array(m, bool)
        if just == 2:
            tx = x + (w - tw) // 2
        elif just == 1:
            tx = x + w - tw
        else:
            tx = x + 1
        ty = y + (h - th) // 2 + 1
        self.blit_mask(a, tx, ty, col if ega else self.farbe(col))

    def blit_mask(self, a, tx, ty, f):
        H, W = self.c.shape
        sy0, sx0 = max(0, -ty), max(0, -tx)
        sy1, sx1 = min(a.shape[0], H - ty), min(a.shape[1], W - tx)
        if sy1 <= sy0 or sx1 <= sx0:
            return
        sub = self.c[ty + sy0:ty + sy1, tx + sx0:tx + sx1]
        sub[a[sy0:sy1, sx0:sx1]] = f

    def text_bitmap(self, mask, x, y, w, h, col, just=2, vcenter=True):
        tw = mask.shape[1]
        if just == 2:
            tx = x + (w - tw) // 2
        elif just == 1:
            tx = x + w - tw
        else:
            tx = x
        ty = y + (h - mask.shape[0]) // 2 if vcenter else y
        self.blit_mask(mask, tx, ty, self.farbe(col))

    def textblock_bitmap(self, text, x, y, w, h, col, font):
        zeilen = wrap_bitmap(text, font, w)
        for k, z in enumerate(zeilen):
            if (k + 1) * font.h > h + font.h // 3:
                break
            self.text_bitmap(font.render(z), x + 2, y + k * font.h, w, font.h,
                             col, 0, False)

    def textblock(self, text, x, y, w, h, col, art="sans", size=None):
        """Fliesstext mit Umbruch in der Groesse size (sonst passend)."""
        s = size or textblock_size(text, w, h, art)
        zeilen, lh = wrap_text(text, w, art, s)
        for k, z in enumerate(zeilen):
            if (k + 1) * lh > h:
                break
            self.text(z, x + 2, y + 2 + k * lh, w - 4, lh, col, art, s, 0)


def wrap_text(text, w, art, s):
    zeilen = []
    for a in text.replace("\r\n", "\r").split("\r"):
        z = ""
        for wd in a.split(" "):
            t = (z + " " + wd) if z else wd
            if text_width(t, art, s) <= w - 6 or not z:
                z = t
            else:
                zeilen.append(z)
                z = wd
        zeilen.append(z)
    return zeilen, int(s * 1.2)


def textblock_size(text, w, h, art="sans"):
    """Groesste Schrift (8..18), bei der der Fliesstext ins Feld passt."""
    for s in range(18, 8, -1):
        zeilen, lh = wrap_text(text, w, art, s)
        if (lh * len(zeilen) <= h - 4 and
                max(text_width(z, art, s) for z in zeilen) <= w - 4):
            return s
    return 8



# ---------------------------------------------------------------------------
# Original-Ressourcen der Empfangssoftware (VD 3.3 DOS / WinVideodat 3.2)
#
#   XFONTS/VDxxFNT.XFN  Proportionalschriften. 'VD' 00, uint16 Dateilaenge,
#                       256 x uint16 Offset je Zeichen (CP437, 0 = fehlt),
#                       je Zeichen: Breite, Hoehe, Zeilen zu ceil(B/8) Byte.
#                       Schriftnummer im Seitenobjekt = 0x80 + xx.
#   GM8X16_n.FNT        Systemschriften 8x16 (Schriftnummer n < 0x80),
#                       16 Zeilen zu je 256 Byte (Zeile r aller Zeichen).
#   GMICONS.DAT         767 Symbole 32x32, je Zeile 4 Ebenen zu 4 Byte.
# ---------------------------------------------------------------------------
class XfnFont:
    def __init__(self, path):
        b = open(path, "rb").read()
        tab = struct.unpack_from("<256H", b, 5)
        self.g = {}
        for c, o in enumerate(tab):
            if not o or o + 2 > len(b):
                continue
            w, h = b[o], b[o + 1]
            bpr = (w + 7) // 8
            raw = np.frombuffer(b[o + 2:o + 2 + bpr * h], np.uint8)
            if w == 0 or len(raw) < bpr * h:
                continue
            self.g[c] = np.unpackbits(raw.reshape(h, bpr),
                                      axis=1)[:, :w].astype(bool)
        self.h = max((g.shape[0] for g in self.g.values()), default=16)
        sp = self.g.get(0x20)
        self.space = sp.shape[1] if sp is not None else self.h // 3

    def glyph(self, c):
        g = self.g.get(c)
        if g is None:
            g = np.zeros((self.h, self.space), bool)
        return g

    def width(self, bs):
        return sum(self.glyph(c).shape[1] for c in bs)

    def render(self, bs):
        if not bs:
            return np.zeros((self.h, 0), bool)
        gl = []
        for c in bs:
            g = self.glyph(c)
            if g.shape[0] < self.h:
                g = np.vstack([g, np.zeros((self.h - g.shape[0], g.shape[1]),
                                           bool)])
            gl.append(g[:self.h])
        return np.concatenate(gl, axis=1)


class SysFont(XfnFont):
    def __init__(self, path, rows):
        d = np.frombuffer(open(path, "rb").read(), np.uint8)
        d = d[:rows * 256].reshape(rows, 256)
        self.g = {c: np.unpackbits(d[:, c][:, None], axis=1).astype(bool)
                  for c in range(256)}
        self.h = rows
        self.space = 8


class Original:
    """Schriften und Symbole aus dem Programmverzeichnis der Software."""

    def __init__(self, root):
        self.root = root
        self._fonts = {}
        self.icons = None
        p = self._find("GMICONS.DAT")
        if p:
            d = np.frombuffer(open(p, "rb").read(), np.uint8)
            n = len(d) // 512
            bits = np.unpackbits(d[:n * 512].reshape(n, 32, 4, 4), axis=3)
            self.icons = (bits[:, :, 0] | bits[:, :, 1] << 1 |
                          bits[:, :, 2] << 2 | bits[:, :, 3] << 3)

    def _find(self, *parts):
        for cand in (os.path.join(self.root, *parts),
                     os.path.join(self.root, *[x.lower() for x in parts])):
            if os.path.exists(cand):
                return cand
        return None

    def font(self, fid):
        if fid not in self._fonts:
            f = None
            try:
                if fid >= 0x80:
                    p = self._find("XFONTS", "VD%02XFNT.XFN" % (fid - 0x80))
                    f = XfnFont(p) if p else None
                else:
                    p = (self._find("GM8X16_%d.FNT" % fid) or
                         self._find("GM8X16_0.FNT"))
                    f = SysFont(p, 16) if p else None
            except (OSError, struct.error, ValueError):
                f = None
            self._fonts[fid] = f
        return self._fonts[fid]

    def anzahl(self):
        xf = self._find("XFONTS")
        n = len([x for x in os.listdir(xf) if x.upper().endswith(".XFN")]) \
            if xf else 0
        return n, (len(self.icons) if self.icons is not None else 0)

    @staticmethod
    def suchen(kandidaten):
        for c in kandidaten:
            if c and (os.path.exists(os.path.join(c, "GMICONS.DAT")) or
                      os.path.isdir(os.path.join(c, "XFONTS"))):
                return Original(c)
        return None


def _bytes(s):
    return s.encode("cp437", "replace")


def wrap_bitmap(text, font, w):
    zeilen = []
    for a in text.replace("\r\n", "\r").split("\r"):
        z = b""
        for wd in _bytes(a).split(b" "):
            t = (z + b" " + wd) if z else wd
            if font.width(t) <= w - 4 or not z:
                z = t
            else:
                zeilen.append(z)
                z = wd
        zeilen.append(z)
    return zeilen


def font_groessen(seiten, holen=None):
    """Je Schriftnummer eine einheitliche Groesse schaetzen (Median der
    Groessen, mit denen die Texte in ihre Objekte passen). Textfelder
    (Schluessel ('feld', Nummer)) werden getrennt geschaetzt."""
    werte = {}
    for s in seiten:
        for o in s.objs:
            if o["typ"] == 0x26 and holen and not o["fehlt"]:
                p = s.ref(o["spec"])
                art, txt = holen(struct.unpack("<H", o["spec"][:2])[0])
                if (art == "text" and txt and txt.strip("\0 ") and
                        p is not None and p + 14 <= len(s.rec)):
                    f = struct.unpack(">H", s.rec[p + 12:p + 14])[0]
                    werte.setdefault(("feld", f), []).append(
                        textblock_size(txt.strip("\0 "), o["w"], o["h"]))
                continue
            if o["typ"] != 0x16 or o["fehlt"]:
                continue
            t = s.tedinfo(s.ref(o["spec"]))
            if not t or not t["text"] or len(t["text"].strip()) < 6:
                continue
            art = font_art(t["font"])
            werte.setdefault(t["font"], []).append(
                fit_size(t["text"], art, o["w"], o["h"]))
    # Textfelder: unteres Quartil, damit auch laengere Texte meist passen
    return {k: int(np.percentile(v, 25) if isinstance(k, tuple)
                   else np.median(v)) for k, v in werte.items()}


def render_seite(seite, holen, groessen, palette="vdi", orig=None):
    """Seite zeichnen. holen(nr) liefert (art, inhalt) eines Satzes.
    Rueckgabe: (Farbfeld, Liste der Schaltflaechen, Hinweise)."""
    z = Zeichner(palette=palette)
    knoepfe, hinweise = [], []
    for o, x, y in seite.walk():
        w, h, sp = o["w"], o["h"], o["spec"]
        ty = o["typ"] & 0xFF            # oberes Byte: Zusatzangabe
        xt = o["typ"] >> 8
        if o["fehlt"]:
            hinweise.append("Objekte fehlen")
            continue
        if ty == 0x14:                                  # Kasten
            cw = struct.unpack(">H", sp[2:4])[0]
            z.fill(x, y, w, h, cw & 15, (cw >> 4) & 7, bool(cw & 0x80))
            z.rahmen(x, y, w, h, cw >> 12, struct.unpack("b", sp[1:2])[0])
        elif ty == 0x16:                                # Kastentext
            t = seite.tedinfo(seite.ref(sp))
            if not t or t["text"] is None:
                z.grau(x, y, w, h)
                hinweise.append("Text von Objekt %d fehlt" % o["i"])
                continue
            cw = t["color"]
            z.fill(x, y, w, h, cw & 15, (cw >> 4) & 7, bool(cw & 0x80))
            z.rahmen(x, y, w, h, cw >> 12,
                     struct.unpack("b", struct.pack("B", t["dicke"] & 255))[0])
            s = t["text"]
            f = orig.font(t["font"]) if orig else None
            if s.strip() and f is not None:
                z.text_bitmap(f.render(_bytes(s)), x, y, w, h,
                              (cw >> 8) & 15, t["just"])
            elif s.strip():
                art = font_art(t["font"])
                size = groessen.get(t["font"]) or fit_size(s, art, w, h)
                size = max(8, min(size, fit_size(s, art, int(w * 1.1), h)))
                z.text(s, x, y, w, h, (cw >> 8) & 15, art, size, t["just"])
        elif ty in (0x1A, 0x25, 0x2C):                  # Schaltflaechen
            nr = struct.unpack("<H", sp[:2])[0]
            label = seite.string(seite.ref(sp)) or ""
            dicke = 1 + bool(o["flags"] & 4) + bool(o["flags"] & 2)
            z.fill(x, y, w, h, WEISS)
            z.rahmen(x, y, w, h, SCHWARZ, -dicke)
            f = orig.font(0) if orig else None
            if label and f is not None:
                z.text_bitmap(f.render(_bytes(label)), x, y, w, h, SCHWARZ, 2)
            elif label:
                z.text(label, x, y, w, h, SCHWARZ, "mono",
                       fit_size(label, "mono", w - 4, h), 2)
            knoepfe.append(dict(x=x, y=y, w=w, h=h, label=label,
                                seite=nr if ty == 0x1A else None,
                                sendung=nr if ty == 0x25 else None,
                                datei=nr if ty == 0x2C else None))
        elif ty in (0x21, 0x27):                        # Bild
            nr = struct.unpack("<H", sp[:2])[0]
            art, img = holen(nr)
            if art != "bild" or img is None:
                z.grau(x, y, w, h)
                hinweise.append("Bild Nr %d fehlt" % nr)
                continue
            ih, iw = img.shape
            r = z.clip(x, y, min(w, iw), min(h, ih))
            if r:
                x0, y0, x1, y1 = r
                part = img[y0 - y:y1 - y, x0 - x:x1 - x]
                neu = np.where(part < 0, GRAU_IDX, part)
                if ty == 0x27:
                    # einfarbiges Bild: gesetzte Punkte in der Farbe aus
                    # dem oberen Typ-Byte, der Rest bleibt durchsichtig
                    neu = np.where(part > 0, xt & 15,
                                   np.where(part < 0, GRAU_IDX,
                                            z.c[y0:y1, x0:x1]))
                # Typ 0x21 mit oberem Byte (z. B. 0x421, 0xB21): Bedeutung
                # unbekannt; das Bild wird deckend gezeichnet
                z.c[y0:y1, x0:x1] = neu
        elif ty == 0x23 and orig is not None and orig.icons is not None \
                and struct.unpack("<H", sp[2:4])[0] < len(orig.icons):
            ic = orig.icons[struct.unpack("<H", sp[2:4])[0]]
            r = z.clip(x, y, 32, 32)
            if r:
                x0, y0, x1, y1 = r
                z.c[y0:y1, x0:x1] = ic[y0 - y:y1 - y, x0 - x:x1 - x]
        elif ty == 0x23:                                # Symbol
            z.fill(x + 2, y + 2, w - 4, h - 4, HELLGRAU)
            z.rahmen(x + 2, y + 2, w - 4, h - 4, SCHWARZ, 1)
            z.text("%d" % struct.unpack("<H", sp[2:4])[0], x, y, w, h,
                   SCHWARZ, "mono", 9, 2)
        elif ty == 0x26:                                # Textfeld
            nr = struct.unpack("<H", sp[:2])[0]
            p = seite.ref(sp)
            col = 1
            size = None
            f = None
            if p is not None and p + 20 <= len(seite.rec):
                col = (struct.unpack(">H", seite.rec[p + 18:p + 20])[0]
                       >> 8) & 15
                fid = struct.unpack(">H", seite.rec[p + 12:p + 14])[0]
                size = groessen.get(("feld", fid))
                f = orig.font(fid) if orig else None
            art, txt = holen(nr)
            if art != "text" or txt is None:
                z.grau(x, y, w, h)
                hinweise.append("Text Nr %d fehlt" % nr)
                continue
            txt = txt.strip("\0 ")
            if f is not None:
                z.textblock_bitmap(txt, x, y, w, h, col, f)
                continue
            if size:
                size = min(size, textblock_size(txt, w, h))
            z.textblock(txt, x, y, w, h, col, size=size)
        elif ty == 0:
            continue
        else:
            hinweise.append("unbekannter Objekttyp 0x%02x" % ty)
    if getattr(seite, "verwaist", 0):
        hinweise.append("%d Objekte ohne Elternobjekt, Lage geschaetzt" %
                        seite.verwaist)
    return z.c, knoepfe, hinweise


def save_canvas(c, path):
    from PIL import Image
    pal = np.array(EGA + [GRAU], np.uint8)
    Image.fromarray(pal[c]).save(path)


# ---------------------------------------------------------------------------
def html_seiten(seiten_info, sendungen_nr, outdir, dateien=None):
    """Kleine HTML-Uebersicht: alle Seiten untereinander, Schaltflaechen
    sind anklickbar (Sprung zur Zielseite, Sendungsdaten als Tooltip)."""
    import html
    vorhanden = {s["nr"] for s in seiten_info}
    teile = ["<!doctype html><html lang='de'><head><meta charset='utf-8'>",
             "<title>VideoDat-Seiten</title><style>",
             "body{background:#222;color:#ddd;font-family:sans-serif;"
             "margin:16px}img{max-width:100%;height:auto;image-rendering:"
             "pixelated;border:1px solid #555}h2{font-size:1rem;margin:"
             "24px 0 6px}a{color:#9cf}</style></head><body>",
             "<h1>VideoDat-Seiten</h1><p>"]
    teile.append(" ".join("<a href='#s%d'>%d</a>" % (s["nr"], s["nr"])
                          for s in seiten_info))
    teile.append("</p>")
    for s in seiten_info:
        teile.append("<h2 id='s%d'>Seite Nr %d (%s)</h2>" %
                     (s["nr"], s["nr"], html.escape(s["titel"])))
        teile.append("<img src='%s' usemap='#m%d' width='640' height='480' "
                     "alt=''>" % (s["datei"], s["nr"]))
        teile.append("<map name='m%d'>" % s["nr"])
        for k in s["knoepfe"]:
            coords = "%d,%d,%d,%d" % (k["x"], k["y"], k["x"] + k["w"],
                                      k["y"] + k["h"])
            if k["seite"] is not None and k["seite"] in vorhanden:
                teile.append("<area shape='rect' coords='%s' href='#s%d' "
                             "title='%s'>" % (coords, k["seite"],
                                              html.escape(k["label"])))
            elif k.get("datei") is not None:
                d = (dateien or {}).get(k["datei"])
                teile.append("<area shape='rect' coords='%s' title='%s' "
                             "href='%s'>" % (
                                 coords, html.escape(
                                     "%s: Dokument Nr %d" % (k["label"],
                                                             k["datei"])),
                                 d or "#s%d" % s["nr"]))
            elif k["sendung"] is not None:
                e = sendungen_nr.get(k["sendung"])
                if e:
                    pf, kb = preis_groesse(e["groesse"])
                    t = "%s: %s, %d kB%s, %s" % (
                        e["kennung"], e["titel"], kb,
                        ", DM %d,%02d" % divmod(pf, 100) if pf is not None
                        else "", "; ".join(e["zeiten"]))
                else:
                    t = "Programmeintrag Nr %d" % k["sendung"]
                teile.append("<area shape='rect' coords='%s' title='%s' "
                             "href='#s%d'>" % (coords, html.escape(t),
                                               s["nr"]))
        teile.append("</map>")
    teile.append("</body></html>")
    with open(os.path.join(outdir, "seiten.html"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(teile))


def main():
    ap = argparse.ArgumentParser(
        description="VideoDat-VDB auswerten: Bilder, Texte, Programm"
        "eintraege und Bildschirmseiten (als PNG nachgebaut)")
    ap.add_argument("dateien", nargs="+",
                    help="eine oder mehrere Empfangskopien derselben .VDB "
                    "(werden zusammengefuehrt), optional .IDX und .TTL")
    ap.add_argument("-o", "--outdir")
    ap.add_argument("--ohne-wiederholung", action="store_true",
                    help="Luecken nicht aus vollstaendigen Wiederholungen "
                    "desselben Bildes ergaenzen")
    ap.add_argument("--luecken", help="Datei mit fehlenden Bereichen "
                    "(Zeilen 'von-bis'); sonst werden 255-Byte-Nullbloecke "
                    "als fehlend angenommen (nur bei einer VDB-Datei)")
    ap.add_argument("--nullbloecke", action="store_true",
                    help="255-Byte-Nullbloecke als fehlend behandeln, auch "
                    "wenn der Dateiname nicht auf .unvollstaendig endet "
                    "(bei mehreren Kopien immer)")
    ap.add_argument("--palette", choices=("vdi", "ega"), default="vdi",
                    help="Farbnummern der Seitenobjekte als GEM-VDI-Farben "
                    "(Vorgabe) oder direkt als EGA-Farben deuten")
    ap.add_argument("--software", metavar="ORDNER",
                    help="Programmordner der Empfangssoftware (mit XFONTS\\, "
                    "GMICONS.DAT, GM8X16_*.FNT) fuer Originalschriften und "
                    "-symbole; sonst wird neben der VDB und dem Skript gesucht")
    ap.add_argument("--ohne-seiten", action="store_true",
                    help="Bildschirmseiten nicht als PNG nachbauen")
    args = ap.parse_args()

    vdbs, idx, ttl = [], {}, []
    for p in args.dateien:
        kopf = open(p, "rb").read(3)
        if kopf[:2] == b"GS":
            idx = read_idx(p)
        elif p.upper().endswith(".TTL") or kopf == b"V0E":
            ttl = read_ttl(p)
        else:
            vdbs.append(p)
    if not vdbs:
        ap.error("keine VDB-Datei angegeben")
    datas = [open(p, "rb").read() for p in vdbs]
    base = os.path.basename(vdbs[0]).split(".unvollstaendig")[0]
    outdir = args.outdir or base + "_inhalt"
    os.makedirs(outdir, exist_ok=True)
    if len(datas) > 1:
        data, miss, differ = merge_copies(datas)
        beide = int(sum((~missing_mask(x)).sum() for x in datas) -
                    (~miss).sum())
        if differ > max(50, 0.01 * beide):
            print("Die VDB-Dateien weichen in %d Bytes voneinander ab - das "
                  "sind offenbar verschiedene Ausgaben, keine Kopien derselben "
                  "Sendung. Bitte einzeln auswerten." % differ)
            raise SystemExit(1)
        einzeln = [int(missing_mask(d).sum()) // BLK for d in datas]
        print("%d Kopien zusammengefuehrt: fehlende Bloecke %s -> %d, "
              "%d Bytes abweichend (Mehrheit/erste Datei gewinnt)" %
              (len(datas), "/".join(map(str, einzeln)),
               int(miss.sum()) // BLK, differ))
    else:
        data = datas[0]
        if (args.luecken or args.nullbloecke or
                "unvollst" in vdbs[0].lower()):
            miss = missing_mask(data, args.luecken)
        else:
            # vollstaendig empfangene Datei: Nullbloecke sind echte Daten
            miss = np.zeros(len(data), bool)
    recs = find_records(data, miss, idx)
    by_nr = records_by_nr(recs, miss, idx)
    print("%s: %d Bytes, %d fehlende Bloecke (%.0f %%), %d Datensaetze%s" %
          (base, len(data), int(miss.sum()) // BLK, 100.0 * miss.mean(),
           len(recs), " (IDX mit %d Eintraegen)" % len(idx) if idx else ""))

    # vollstaendig empfangene Bilder als Referenz fuer Wiederholungen
    refs = {}
    if not args.ohne_wiederholung:
        for off, r in recs.items():
            if r is None or r["kind"] != "bild":
                continue
            if miss[off:off + 12 + r["L"]].any():
                continue
            rec = data[off + 12:off + 12 + r["L"]]
            w, h = struct.unpack_from("<HH", rec, 6)
            dec = bytes(image_decoder(rec)(rec[10:]))
            soll = bild_ebenen(rec) * ((w + 7) // 8) * h
            if rec[4:5] == b"\x01" and len(dec) > soll:
                dec = dec[len(dec) - soll:]
            if len(dec) >= soll:
                refs.setdefault((w, h), []).append(dec)

    bilder = {}

    def bild(off):
        if off not in bilder:
            L = recs[off]["L"]
            rec = data[off + 12:off + 12 + L]
            rm = miss[off + 12:off + 12 + L]
            w, h = struct.unpack_from("<HH", rec, 6)
            bilder[off] = decode_image(
                rec, rm, refs.get((w, h), []) if rm.any() else [])
        return bilder[off]

    def text_von(off):
        L = recs[off]["L"]
        rec = bytearray(data[off + 12:off + 12 + L])
        for k in np.nonzero(miss[off + 12:off + 12 + L])[0]:
            rec[k] = ord("?")
        return bytes(rec).split(b"\0")[0].decode("cp437")

    def holen(nr):
        off = by_nr.get(nr)
        if off is None or recs.get(off) is None:
            return None, None
        k = recs[off]["kind"]
        if k == "bild":
            return k, bild(off)[2]
        if k == "text":
            return k, text_von(off)
        return k, None

    summary, seiten, sendungen, dateien = [], [], {}, {}
    for off, r in sorted(recs.items()):
        if r is None:
            summary.append("%8d  ?        Kopf fehlt (laut IDX vorhanden)"
                           % off)
            continue
        L, nr, kind = r["L"], r["nr"], r["kind"]
        rec = data[off + 12:off + 12 + L]
        rm = miss[off + 12:off + 12 + L]
        fehl = 100.0 * rm.mean() if len(rm) else 0
        alt = "" if by_nr.get(nr) == off else "  (weitere Fassung)"
        if kind == "bild":
            w, h, img, info = bild(off)
            name = "bild_%07d.png" % off
            if img is not None:
                save_png(img, os.path.join(outdir, name))
                unbek = 100.0 * (img < 0).mean()
                summary.append("%8d  Bild     Nr %d, %dx%d, %.0f %% der Daten "
                               "fehlen, %.0f %% des Bildes unbekannt -> %s%s%s"
                               % (off, nr, w, h, fehl, unbek, name, alt,
                                  ("\n            " + info)
                                  if info and rm.any() else ""))
            else:
                summary.append("%8d  Bild     Nr %d, %dx%d: %s" %
                               (off, nr, w, h, info))
        elif kind == "text":
            name = "text_%03d.txt" % nr
            t = text_von(off)
            with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
                f.write(t.replace("\r", "\n") + "\n")
            summary.append("%8d  Text     Nr %d, %d Zeichen, %.0f %% fehlen "
                           "-> %s%s: %s" % (off, nr, len(t), fehl, name, alt,
                                            t.replace("\r", " ").strip()[:60]))
        elif kind == "datei":
            # Begleitdokument ('DT' + Text mit CR/LF, CP437), z. B. ein
            # Formular, das die Seite ueber eine Schaltflaeche ausdruckt
            name = "datei_%03d.txt" % nr
            t = bytes(rec[2:]).decode("cp437")
            with open(os.path.join(outdir, name), "w", encoding="utf-8",
                      newline="") as f:
                f.write(t.replace("\r\n", "\n").replace("\r", "\n"))
            dateien[nr] = name
            erste = next((z for z in t.splitlines() if z.strip()), "")
            summary.append("%8d  Dokument Nr %d, %d Zeichen, %.0f %% fehlen "
                           "-> %s%s: %s" % (off, nr, len(t), fehl, name, alt,
                                            erste.strip()[:60]))
        elif kind == "sendung":
            e = parse_sendung(rec)
            if not rm.any():
                sendungen[nr] = e
            summary.append("%8d  Sendung  Nr %d: %-16s %s  [%s]%s" %
                           (off, nr, e["kennung"], e["titel"],
                            ", ".join(e["zeiten"]), alt))
        elif kind == "liste":
            n = struct.unpack_from("<I", rec, 0)[0] if L >= 4 else 0
            nrs = [struct.unpack_from("<I", rec, 4 + 4 * k)[0]
                   for k in range(min(n, (L - 4) // 4))]
            summary.append("%8d  Liste    Nr %d: Seitenfolge %s" %
                           (off, nr, " ".join(map(str, nrs))))
        elif kind in ("seite", "vorlage"):
            t = texts(rec)
            name = "%s_%03d" % (kind, nr)
            with open(os.path.join(outdir, name + ".txt"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(t) + "\n")
            summary.append("%8d  %-8s Nr %d, %d Bytes, %.0f %% fehlen%s: %s" %
                           (off, kind.capitalize(), nr, L, fehl, alt,
                            " | ".join(t[:5])[:90]))
            if alt == "":
                seiten.append((nr, kind, off, rec, rm, t))
        else:
            summary.append("%8d  ?        Nr %d, %d Bytes, Flags %08x" %
                           (off, nr, L, r["flags"]))

    if not args.ohne_seiten and seiten:
        objs = []
        for nr, kind, off, rec, rm, t in seiten:
            try:
                objs.append(Seite(rec, rm))
            except struct.error:
                objs.append(None)
        groessen = font_groessen([s for s in objs if s], holen)
        here = os.path.dirname(os.path.abspath(vdbs[0]))
        me = os.path.dirname(os.path.abspath(__file__))
        orig = Original.suchen([args.software, here, os.path.dirname(here),
                                os.path.join(here, "VIDEODAT"), me,
                                os.path.join(me, "VIDEODAT")])
        if args.software and orig is None:
            print("Hinweis: im Ordner %s keine Schriften/Symbole gefunden"
                  % args.software)
        elif orig:
            nf, ni = orig.anzahl()
            print("Originalschriften (%d) und Symbole (%d) aus %s" %
                  (nf, ni, orig.root))
        info = []
        for (nr, kind, off, rec, rm, t), s in zip(seiten, objs):
            if s is None:
                continue
            c, knoepfe, hinweise = render_seite(s, holen, groessen,
                                                args.palette, orig)
            name = "%s_%03d.png" % (kind, nr)
            save_canvas(c, os.path.join(outdir, name))
            info.append(dict(nr=nr, datei=name, knoepfe=knoepfe,
                             titel=(t[0] if t else kind)))
            summary.append("          %s Nr %d nachgebaut -> %s%s" %
                           (kind.capitalize(), nr, name,
                            (" (" + "; ".join(sorted(set(hinweise))) + ")")
                            if hinweise else ""))
        html_seiten(info, sendungen, outdir, dateien)

    katalog = {}
    for e in ttl:
        katalog.setdefault(e["kennung"], dict(e, quelle="TTL"))
    for k in sorted(sendungen):           # Eintraege der VDB sind genauer
        e = sendungen[k]
        alt_e = katalog.get(e["kennung"], {})
        katalog[e["kennung"]] = dict(alt_e, kennung=e["kennung"],
                                     groesse=e["groesse"], zeiten=e["zeiten"],
                                     titel=alt_e.get("titel") or e["titel"],
                                     quelle="VDB" + ("+TTL" if "TTL" in
                                                     alt_e.get("quelle", "")
                                                     else ""))
    if katalog:
        with open(os.path.join(outdir, "sendungen.txt"), "w",
                  encoding="utf-8") as f:
            f.write("Kennung            Preis   Groesse  Titel                "
                    "              Quelle   Sendetermine\n")
            for kenn in sorted(katalog):
                e = katalog[kenn]
                pf, kb = preis_groesse(e["groesse"])
                f.write("%-17s %9s %6d kB  %-34s %-8s %s\n" % (
                    kenn, "DM %d,%02d" % divmod(pf, 100) if pf is not None
                    else "", kb, e["titel"], e["quelle"],
                    ", ".join(e["zeiten"])))
            kat = next((e["katalog"] for e in ttl if "katalog" in e), None)
            if kat:
                f.write("\nKatalog erstellt: %s\n" % kat)
            f.write("Preis = Kaufpreis per KontoDisk; Groesse in kB "
                    "(1000 Bytes, aufgerundet).\n")
    with open(os.path.join(outdir, "inhalt.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
