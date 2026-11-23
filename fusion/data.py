"""Assemble per-tier feature tables from the ShapeShift Step-2 outputs (DESIGN.md section 2).

Morphology comes from ``compare_tiers.load_morph`` and the plate-centred Tahoe signatures from
``compare_tiers.Tahoe().sig(which, 5.0, centered=True)``; structures from ``overlap.std_key`` / ``std_mol``.
The result per tier is cached as FUSION_DATA/cache/features_<tier>.npz plus a JSON sidecar (unit order, column
names, SHA-256 of every Step-2 input read). No target is touched here.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from . import paths
from . import shapeshift as SS
from .splits import merge_labels

TIERS = {
    "T1": dict(tier="T1_LINCS_A549", morph_dose=3.33, tahoe="A549", cvcl="CVCL_0023"),
    "T2": dict(tier="T2_EUOS_HepG2", morph_dose=10.0, tahoe="HepG2C3A", cvcl="CVCL_1098"),
    "T3": dict(tier="T3_JUMP_U2OS", morph_dose=10.0, tahoe="consensus", cvcl=None),
}
LINCS_DOSES = (0.0412, 0.123, 0.37, 1.11, 3.33, 10.0)
TAHOE_DOSE = 5.0
INPUTS = ["step2/units.csv", "step2/morph_lincs.parquet", "step2/morph_lincs_activity.csv",
          "step2/morph_euos.parquet", "step2/morph_euos_activity.csv", "morph_profiles_high.parquet",
          "step2/morph_activity.csv", "tahoe_signatures.parquet", "step2/tahoe_perline.parquet",
          "step2/tahoe_plate_line_mean.parquet", "step2/tahoe_consistency.csv", "data/tahoe/drug_metadata.parquet"]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def near(a, b, tol=0.02):
    return np.abs(np.log(np.asarray(a, float)) - np.log(b)) < tol


def fcols(df):
    return [c for c in df.columns if c[:1] == "f" and c[1:].isdigit()]


# ------------------------------------------------------------------ chemistry
def chem_features(smiles):
    """Morgan r=2 2048 bits + counts, RDKit 2-D descriptors, generic Murcko scaffold (SMILES, '' if acyclic)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    OV = SS.overlap()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bits, counts, desc, scaf = [], [], [], []
    for s in smiles:
        m = OV.std_mol(s)
        if m is None:
            raise RuntimeError(f"std_mol failed for {s}")
        bits.append(gen.GetFingerprintAsNumPy(m).astype(np.uint8))
        counts.append(gen.GetCountFingerprintAsNumPy(m).astype(np.uint16))
        d = Descriptors.CalcMolDescriptors(m)
        desc.append(d)
        try:
            sc = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(m))
            scaf.append(Chem.MolToSmiles(sc) if sc.GetNumAtoms() else "")
        except Exception:
            scaf.append("")
    desc = pd.DataFrame(desc)
    return np.array(bits), np.array(counts), desc, scaf


def scaffold_clusters(scaffolds, cutoff_sim=0.6):
    """Butina clustering of the distinct generic scaffolds on Morgan bits (Tanimoto >= cutoff_sim joins a
    cluster). Acyclic compounds (empty scaffold) are each their own cluster. Returns one label per compound."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    uniq = sorted({s for s in scaffolds if s})
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in uniq]
    dists = []
    for i in range(1, len(fps)):
        dists.extend(1 - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])))
    clusters = Butina.ClusterData(dists, len(fps), 1 - cutoff_sim, isDistData=True)
    lab = {}
    for c, members in enumerate(clusters):
        for m in members:
            lab[uniq[m]] = f"scaf:{c}"
    return np.array([lab[s] if s else f"acyclic:{i}" for i, s in enumerate(scaffolds)], dtype=object)


# ------------------------------------------------------------------ Tahoe
_TAHOE = {}


def tahoe():
    if "t" not in _TAHOE:
        _TAHOE["t"] = SS.compare_tiers().Tahoe()
    return _TAHOE["t"]


def tahoe_block(which, units):
    """Plate-centred 5 uM signature (compare_tiers.Tahoe.sig(..., centered=True)) for the given units."""
    t = tahoe()
    s = t.sig(which, TAHOE_DOSE, centered=True)
    if s.unit.duplicated().any():
        raise RuntimeError(f"Tahoe {which}: duplicate unit rows at 5 uM")
    s = s.set_index("unit")
    return s.loc[units, t.genes].to_numpy(np.float64), list(t.genes)


def tahoe_log2_cellratio(cvcl):
    """log2(n_cells_trt / median over (first plate, dose) of the line), as in scripts/probe_targets.py.
    cvcl=None -> mean over all lines (cross-line consensus). Returns Series unit -> value at 5 uM."""
    t = tahoe()
    pl = t.pl[["unit", "dose_uM", t.cid, t.pcol, "n_cells_trt"]].copy()
    pl["p1"] = pl[t.pcol].astype(str).str.split(";").str[0]
    pl["ratio"] = pl.n_cells_trt / pl.groupby([t.cid, "p1", "dose_uM"]).n_cells_trt.transform("median")
    pl["lr"] = np.log2(pl.ratio.clip(lower=1e-3))
    pl = pl[near(pl.dose_uM, TAHOE_DOSE)]
    if cvcl is not None:
        pl = pl[pl[t.cid] == cvcl]
    return pl.groupby("unit").lr.mean()


# ------------------------------------------------------------------ morphology
def morph_block(key, units):
    """-> (X at the tier's dose, nuisance DataFrame [morph_norm, morph_n_wells(, block_reliable)], 6-dose stack
    list or None)."""
    ct = SS.compare_tiers()
    cfg = TIERS[key]
    U = pd.read_csv(paths.shapeshift_data() / "step2" / "units.csv")
    p, a, _ = ct.load_morph(cfg["tier"], U)
    cols = fcols(p)
    pd_ = p[near(p.dose_uM.astype(float), cfg["morph_dose"])].drop_duplicates("unit").set_index("unit")
    X = pd_.loc[units, cols].to_numpy(np.float64)
    X = X[:, ~np.all(np.isnan(X), 0)]
    X = np.nan_to_num(X)
    nu = pd.DataFrame(index=units)
    nu["morph_n_wells"] = pd_.loc[units, "n_wells"].astype(float).to_numpy()
    stack = None
    if key in ("T1", "T2"):
        aa = a[near(a.dose_uM.astype(float), cfg["morph_dose"])].drop_duplicates("unit").set_index("unit")
        nu["morph_norm"] = aa.loc[units, "norm_z"].astype(float).to_numpy()
        if key == "T1":
            nu["morph_block_reliable"] = aa.loc[units, "block_reliable"].astype(float).to_numpy()
            stack = []
            for d in LINCS_DOSES:
                q = p[near(p.dose_uM.astype(float), d)].drop_duplicates("unit").set_index("unit")
                Xd = np.nan_to_num(q.loc[units, cols].to_numpy(np.float64))
                stack.append(Xd)
    else:
        # morph_activity.csv (JUMP) has no norm_z column: log L2 norm of the consensus profile instead
        nu["morph_norm"] = np.log(np.linalg.norm(X, axis=1) + 1e-9)
    return X, nu, stack


# ------------------------------------------------------------------ units
def tier_units(key):
    """Tier units that have both a morphology profile at the tier dose and a 5 uM Tahoe signature."""
    cfg = TIERS[key]
    ct = SS.compare_tiers()
    U = pd.read_csv(paths.shapeshift_data() / "step2" / "units.csv")
    u = U[U.tier == cfg["tier"]].copy()
    p, _, _ = ct.load_morph(cfg["tier"], U)
    has_m = set(p.loc[near(p.dose_uM.astype(float), cfg["morph_dose"]), "unit"].dropna())
    s = tahoe().sig(cfg["tahoe"], TAHOE_DOSE, centered=False)
    has_t = set(s.unit)
    u = u[u.unit.isin(has_m) & u.unit.isin(has_t)]
    return u.sort_values("unit").reset_index(drop=True)


def compound_groups(u):
    """Step-2 unit, merged across units that share a dataset compound ID (name-bridge / prodrug pairs such as
    ixazomib / ixazomib citrate, which share LINCS wells). Stereo-flagged units are already stereo-flat."""
    ids = [set(str(x).split(";")) for x in u.dataset_ids]
    pairs = [(i, j) for i in range(len(ids)) for j in range(i) if ids[i] & ids[j]]
    lab = np.arange(len(u))
    from .splits import union_find
    lab = union_find(len(u), pairs)
    return lab


def unit_smiles(units):
    OV = SS.overlap()
    dm = pd.read_parquet(paths.shapeshift_data() / "data" / "tahoe" / "drug_metadata.parquet")
    dm["unit"] = dm.canonical_smiles.map(OV.std_key)
    sm = dm.dropna(subset=["unit"]).groupby("unit").canonical_smiles.first()
    missing = [x for x in units if x not in sm.index]
    if missing:
        raise RuntimeError(f"{len(missing)} units without a Tahoe SMILES: {missing[:3]}")
    return [sm[x] for x in units]


# ------------------------------------------------------------------ build + cache
def build_tier(key, force=False):
    cache = paths.data_dir() / "cache"
    cache.mkdir(exist_ok=True)
    fz, fj = cache / f"features_{key}.npz", cache / f"features_{key}.json"
    if fz.exists() and fj.exists() and not force:
        return load_tier(key)
    cfg = TIERS[key]
    u = tier_units(key)
    units = list(u.unit)
    Xm, nu, stack = morph_block(key, units)
    Xt, genes = tahoe_block(cfg["tahoe"], units)
    cons = pd.read_csv(paths.shapeshift_data() / "step2" / "tahoe_consistency.csv")
    cons = cons[near(cons.dose_uM, TAHOE_DOSE)].drop_duplicates("unit").set_index("unit")
    nu["tx_magnitude"] = np.linalg.norm(Xt, axis=1)
    nu["tx_log2_cellratio"] = tahoe_log2_cellratio(cfg["cvcl"]).reindex(units).to_numpy()
    npl = cons.n_plates.reindex(units).to_numpy(float)
    nu["tx_two_plate"] = (npl >= 2).astype(float)
    smiles = unit_smiles(units)
    bits, counts, desc, scaf = chem_features(smiles)
    cg = compound_groups(u)
    sc = scaffold_clusters(scaf)
    scaffold_group = merge_labels(sc, cg)
    nuis_cols = ["tx_magnitude", "tx_log2_cellratio", "tx_two_plate", "morph_norm", "morph_n_wells"] + \
        (["morph_block_reliable"] if "morph_block_reliable" in nu else [])
    arrays = dict(morph=Xm, tx=Xt.astype(np.float32), bits=bits, counts=counts,
                  desc=desc.to_numpy(np.float64), nuis=nu[nuis_cols].to_numpy(np.float64),
                  compound_group=cg, scaffold_group=scaffold_group)
    if stack is not None:
        for i, Xd in enumerate(stack):
            arrays[f"morph_d{i}"] = Xd
    np.savez_compressed(fz, **arrays)
    meta = dict(tier=key, tier_name=cfg["tier"], units=units, moa=u.moa.fillna("unclear").tolist(),
                two_plate=[bool(x) for x in (npl >= 2)], n_plates=[int(x) if np.isfinite(x) else None for x in npl],
                nuis_cols=nuis_cols, desc_cols=list(desc.columns), n_genes=len(genes),
                generic_scaffold=scaf, inputs_sha256={f: sha256(paths.shapeshift_data() / f) for f in INPUTS},
                shapeshift_code_commit=SS.code_commit())
    fj.write_text(json.dumps(meta, indent=0))
    return load_tier(key)


def load_tier(key):
    cache = paths.data_dir() / "cache"
    z = np.load(cache / f"features_{key}.npz")
    meta = json.loads((cache / f"features_{key}.json").read_text())
    arr = {k: z[k] for k in z.files}
    return arr, meta
