"""Range-read probe of wide PRISM CSVs (rows = cell lines, ~70-190 kB each).

Finds the row name at sampled byte offsets and fetches whole rows for target cell lines,
so the 41 MB / 92 MB matrices are never downloaded whole.
Usage: py -3.13 -I -X utf8 range_rows.py URL N_PROBES
"""
import sys

import requests

S = requests.Session()
BYTES = 0


def size(url):
    r = S.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    return int(r.headers["Content-Range"].split("/")[1])


def rng(url, a, b):
    global BYTES
    r = S.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
    r.raise_for_status()
    BYTES += len(r.content)
    return r.content


def name_at(url, off, step=65536):
    """(start offset, row name) of the first line beginning after byte `off`."""
    pos = off
    while True:
        c = rng(url, pos, pos + step - 1)
        i = c.find(b"\n")
        if i >= 0:
            break
        if len(c) < step:
            return None, None
        pos += step
    st = pos + i + 1
    return st, rng(url, st, st + 40).split(b",", 1)[0].decode()


def read_line(url, start, step=1 << 18):
    buf = b""
    while True:
        c = rng(url, start + len(buf), start + len(buf) + step - 1)
        j = c.find(b"\n")
        if j >= 0:
            return buf + c[:j]
        buf += c
        if len(c) < step:
            return buf


if __name__ == "__main__":
    url, n = sys.argv[1], int(sys.argv[2])
    N = size(url)
    print("size", N)
    for k in range(n):
        print(name_at(url, int(N * k / n)))
    print("bytes read", BYTES)
