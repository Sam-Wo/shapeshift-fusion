"""PRISM Repurposing 20Q2 (figshare 20564034, CC BY 4.0): fetch A549 / HepG2 / U2OS rows only.

- primary replicate-collapsed LFC (46.7 MB, rows NOT sorted): streamed once, only target rows kept.
- secondary replicate-collapsed LFC (114.7 MB): rows are 'PR500_ACH-*' (sorted) then 'PR300_ACH-*' (sorted),
  so each target row is located by binary search with HTTP range reads.
Writes data_probe/prism20q2/{primary,secondary}_LFC_<LINE>.csv (secondary: one file per pool row found).
Usage: py -3.13 -I -X utf8 fetch_prism20q2_rows.py
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import range_rows as R  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data_probe" / "prism20q2"
PRIMARY = "https://ndownloader.figshare.com/files/36794580"
SECONDARY = "https://ndownloader.figshare.com/files/36794616"
TARGETS = {"A549": "ACH-000681", "HEPG2": "ACH-000739", "U2OS": "ACH-000364"}
BLOCK = {"PR500": 0, "PR300": 1}


def key(name):
    pool, _, ach = name.partition("_")
    return (BLOCK.get(pool, 9), ach)


def find(url, N, target):
    tk = key(target)
    lo, hi = 0, N
    while hi - lo > 600_000:
        mid = (lo + hi) // 2
        st, name = R.name_at(url, mid)
        if name is None or key(name) > tk:
            hi = mid
        else:
            lo = mid
    pos = lo
    while True:
        st, name = R.name_at(url, pos)
        if name is None or key(name) > tk:
            return None
        if name == target:
            return st
        pos = st


def stream_primary():
    want = set(TARGETS.values())
    got, header, nbytes = {}, None, 0
    with requests.get(PRIMARY, stream=True, timeout=300) as r:
        r.raise_for_status()
        buf = b""
        for chunk in r.iter_content(1 << 20):
            nbytes += len(chunk)
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if header is None:
                    header = ln
                    continue
                name = ln.split(b",", 1)[0].decode()
                if name in want:
                    got[name] = ln
        if buf:
            name = buf.split(b",", 1)[0].decode()
            if name in want:
                got[name] = buf
    for lab, ach in TARGETS.items():
        if ach in got:
            (OUT / f"primary_LFC_{lab}.csv").write_bytes(header + b"\n" + got[ach] + b"\n")
            print("primary", lab, "row bytes", len(got[ach]))
        else:
            print("primary", lab, "NOT FOUND")
    return nbytes


def main():
    nb = stream_primary()
    N = R.size(SECONDARY)
    header = R.read_line(SECONDARY, 0)
    for lab, ach in TARGETS.items():
        for pool in BLOCK:
            st = find(SECONDARY, N, f"{pool}_{ach}")
            if st is None:
                print("secondary", pool, lab, "NOT FOUND")
                continue
            row = R.read_line(SECONDARY, st)
            (OUT / f"secondary_LFC_{lab}_{pool}.csv").write_bytes(header + b"\n" + row + b"\n")
            print("secondary", pool, lab, "row bytes", len(row))
    print("bytes: primary streamed", nb, "| secondary range reads", R.BYTES)


if __name__ == "__main__":
    main()
