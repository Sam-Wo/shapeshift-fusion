"""Fetch the A549 (ACH-000681), HepG2 (ACH-000739) and U2OS (ACH-000364) rows of the PRISM 19Q4 replicate-collapsed
log-fold-change matrices by HTTP range requests (files are sorted by DepMap ID).
Writes data_probe/prism/{primary,secondary}_LFC_{A549,HEPG2,U2OS}.csv. Usage: py -3.13 -I -X utf8 fetch_prism_rows.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import range_rows as R  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data_probe" / "prism"
FILES = {"primary": "https://ndownloader.figshare.com/files/20237709",
         "secondary": "https://ndownloader.figshare.com/files/20237757"}
TARGETS = {"A549": "ACH-000681", "HEPG2": "ACH-000739", "U2OS": "ACH-000364"}


def find(url, N, target):
    lo, hi = 0, N
    while hi - lo > 300_000:  # narrow to a few rows
        mid = (lo + hi) // 2
        st, name = R.name_at(url, mid)
        if name is None or name > target:
            hi = mid
        else:
            lo = mid
    pos = lo
    while True:
        st, name = R.name_at(url, pos)
        if name is None:
            return None
        if name == target:
            return st
        if name > target:
            return None
        pos = st


for scr, url in FILES.items():
    N = R.size(url)
    header = R.read_line(url, 0)
    for lab, ach in TARGETS.items():
        st = find(url, N, ach)
        if st is None:
            print(scr, lab, "NOT FOUND"); continue
        row = R.read_line(url, st)
        (OUT / f"{scr}_LFC_{lab}.csv").write_bytes(header + b"\n" + row + b"\n")
        print(scr, lab, "row bytes", len(row))
print("total bytes read", R.BYTES)
