"""Read authoritative joint-bootstrap/null products for legacy paper plots.

Avoids a new plot-only resampling or an accidental node (instead of spatial
block) bootstrap for the full-range markers. Raw mean and h_i are unchanged.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .provenance import file_digest
from .resampling import bootstrap_summary


def find_revision_output(arm_path: Path, catalog: str) -> Path | None:
    if catalog not in {"chen", "carron"}: return None
    for parent in Path(arm_path).parents:
        out = parent / "referee_multiplicity" / catalog
        meta = out / "multiplicity_metadata.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            if m.get("analysis_arm_sha256") != file_digest(Path(arm_path)):
                return None  # not the data that generated this plot
            return out
    return None


def fullrange_markers(out: Path, *, angle: bool) -> dict:
    stats = pd.read_csv(out / "multiplicity_statistics.csv")
    idx = stats.index[(stats.stratum == "all") & (stats.zbin_id == -1)]
    if len(idx) != 1: raise ValueError("Missing/duplicated full-range arm statistic")
    i = int(idx[0]); row = stats.loc[i]
    boot = np.load(out / "multiplicity_bootstrap_samples.npz")
    nullpath = out / "multiplicity_signflip_samples.npz"
    null = np.load(nullpath) if nullpath.exists() else {}
    answer = {}
    for stat,panel,factor in [("H","H",1.),("A_sign","A",100.),("S","S",1.)]:
        value = factor * float(row[stat]); draws = factor * boot[stat][:,i]
        nd = factor * np.abs(null[f"{stat}_all_-1"]) if f"{stat}_all_-1" in null else np.array([])
        nd = nd[np.isfinite(nd)]
        q68,q95 = np.percentile(nd,[68.2689492137,95.4499736104]) if len(nd) else [np.nan,np.nan]
        if angle and stat != "A_sign":
            def transform(x):
                x=np.asarray(x,float)
                return np.rad2deg(np.arcsin(np.where(np.abs(x)<=1,x,np.nan)))
            value=float(transform(value)); draws=transform(draws)
            q68=float(transform(q68));q95=float(transform(q95))
        answer[panel]={"value":value,"err":bootstrap_summary(draws)["err"],"mock68":float(q68),"mock95":float(q95)}
    boot.close()
    if hasattr(null,"close"): null.close()
    return answer


def binned_null_table(out: Path, summary: pd.DataFrame, *, statistic: str, catalog: str) -> pd.DataFrame:
    p=out/"multiplicity_null_summary.csv"
    if not p.exists() or p.stat().st_size < 3: return pd.DataFrame()
    t=pd.read_csv(p);stat="A_sign" if statistic.lower().startswith("sign") else "H"
    t=t[(t.stratum=="all") & (t.zbin_id>=0) & (t.statistic==stat)]
    meta=json.loads((out/"multiplicity_metadata.json").read_text())
    joined=summary[["zbin_id","z_min","z_max","z_center"]].merge(t,on="zbin_id",how="left",validate="one_to_one")
    result=joined[["zbin_id","z_min","z_max","z_center"]].copy()
    result["catalog_key"]=catalog;result["statistic"]=statistic
    result["B"]=meta["null_B"];result["block_nside"]=meta["nside"]
    factor=100. if stat=="A_sign" else 1.
    for dst,src in [("null_q025","q2sigma_low"),("null_q16","q1sigma_low"),("null_q50","q50"),("null_q84","q1sigma_high"),("null_q975","q2sigma_high")]:
        result[dst]=factor*joined[src]
    return result
