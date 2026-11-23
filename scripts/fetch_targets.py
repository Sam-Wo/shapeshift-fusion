"""Download the two PRISM 20Q2 files the real run needs beyond the design-stage probe (DESIGN.md section 5).

PRISM Repurposing 20Q2, figshare article 20564034 (CC BY 4.0; Corsello et al. 2020):
  - secondary-screen-dose-response-curve-parameters.csv (290 MB): streamed once; only the rows of the three
    target lines (A549 ACH-000681, HepG2 ACH-000739, U2OS ACH-000364) are kept, and only the numeric curve
    columns plus IDs (the Repurposing Hub columns name/moa/target/indication/phase are dropped on the fly).
  - primary-screen-replicate-collapsed-logfold-change.csv (46.7 MB): kept whole under FUSION_DATA, then reduced
    to the per-column median LFC over all PRISM cell lines (needed for the line-selective residual target).
Writes FUSION_DATA/prism20q2/{secondary_curve_params_targets.csv, primary_line_median.csv} and appends
url / bytes / SHA-256 to FUSION_DATA/MANIFEST.json. Skips a file that is already present.
Usage: py -3.13 -I -X utf8 scripts/fetch_targets.py
"""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion import paths  # noqa: E402

CURVES = "https://ndownloader.figshare.com/files/36794595"
PRIMARY = "https://ndownloader.figshare.com/files/36794580"
LINES = {"ACH-000681", "ACH-000739", "ACH-000364"}
KEEP = ["broad_id", "depmap_id", "screen_id", "row_name", "upper_limit", "lower_limit", "slope", "r2", "auc",
        "ec50", "ic50", "passed_str_profiling"]


def manifest(entry):
    f = paths.data_dir() / "MANIFEST.json"
    m = json.loads(f.read_text()) if f.exists() else []
    m = [e for e in m if e.get("file") != entry["file"]] + [entry]
    f.write_text(json.dumps(m, indent=1))


def fetch_curves(out):
    if out.exists():
        print("present:", out.name)
        return
    h, nbytes, rows, header = hashlib.sha256(), 0, [], None
    with requests.get(CURVES, stream=True, timeout=600) as r:
        r.raise_for_status()
        buf = b""
        for chunk in r.iter_content(1 << 20):
            h.update(chunk)
            nbytes += len(chunk)
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if header is None:
                    header = next(csv.reader([ln.decode()]))
                    idx = [header.index(c) for c in KEEP]
                    continue
                if any(a.encode() in ln[:200] for a in LINES):
                    rec = next(csv.reader([ln.decode()]))
                    if rec[header.index("depmap_id")] in LINES:
                        rows.append([rec[i] for i in idx])
        if buf.strip():
            rec = next(csv.reader([buf.decode()]))
            if rec[header.index("depmap_id")] in LINES:
                rows.append([rec[i] for i in idx])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=KEEP).to_csv(out, index=False)
    manifest(dict(file=out.name, url=CURVES, bytes_transferred=nbytes, sha256_source=h.hexdigest(),
                  note=f"rows for {sorted(LINES)} only; Hub columns dropped"))
    print("curves:", nbytes, "bytes,", len(rows), "rows kept")


def fetch_primary(raw, out):
    if not raw.exists():
        r = requests.get(PRIMARY, timeout=600)
        r.raise_for_status()
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(r.content)
        manifest(dict(file=raw.name, url=PRIMARY, bytes_transferred=len(r.content),
                      sha256_source=hashlib.sha256(r.content).hexdigest()))
        print("primary:", len(r.content), "bytes")
    if out.exists():
        print("present:", out.name)
        return
    m = pd.read_csv(io.BytesIO(raw.read_bytes()), index_col=0)
    m = m[m.index.astype(str).str.startswith("ACH-")]
    med = pd.DataFrame({"column_name": m.columns, "median_lfc_all_lines": np.nanmedian(m.to_numpy(float), axis=0),
                        "n_lines": np.isfinite(m.to_numpy(float)).sum(0)})
    med.to_csv(out, index=False)
    print("line medians:", len(med), "columns over", len(m), "lines")


def main():
    d = paths.data_dir() / "prism20q2"
    fetch_curves(d / "secondary_curve_params_targets.csv")
    fetch_primary(d / "primary-screen-replicate-collapsed-logfold-change.csv", d / "primary_line_median.csv")


if __name__ == "__main__":
    main()
