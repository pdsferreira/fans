"""Explicit parent-FANS selection and consistency checks (schema v88).

Raw arm/segment tables retain diagnostic rows; analysis tables contain ONLY arms
of accepted multi-arm parents. No fallback may promote rejected/unknown parents.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA_VERSION = "88.1"


def effective_redshift(d: pd.DataFrame) -> np.ndarray:
    z = np.full(len(d), np.nan)
    for col in ("z_eff", "z_mid", "z", "z_center", "z_low"):
        if col in d:
            v = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
            z = np.where(np.isfinite(z), z, v)
    return z


def redshift_bin_ids(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Left-closed/right-open bins, with the final right boundary included."""
    z, edges = np.asarray(z, float), np.asarray(edges, float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError("Redshift edges must be finite and strictly increasing")
    ids = np.searchsorted(edges, z, side="right") - 1
    ids[z == edges[-1]] = len(edges) - 2
    ids[(~np.isfinite(z)) | (z < edges[0]) | (z > edges[-1])] = -1
    return ids.astype(int)


def validate_nodes(nodes: pd.DataFrame, min_arms: int = 2) -> None:
    for col in ("node_id", "status", "n_arms_valid"):
        if col not in nodes:
            raise ValueError(f"Node table missing mandatory column {col!r}; use v88 construction products")
    if nodes.node_id.isna().any() or nodes.node_id.duplicated().any():
        raise ValueError("Parent node_id must be unique and non-null within each catalogue")
    ok = nodes.status.eq("ok")
    m = pd.to_numeric(nodes.n_arms_valid, errors="coerce")
    if ((~np.isfinite(m[ok])) | (m[ok] < min_arms) | (m[ok] != np.floor(m[ok]))).any():
        raise ValueError("An accepted parent has invalid/noninteger multiplicity or fewer than two arms")


def attach_parent_metadata(arms: pd.DataFrame, nodes: pd.DataFrame, *,
                           accepted_only: bool = True, validate_complete: bool = True) -> pd.DataFrame:
    """Join strictly many-to-one; true parent multiplicity never equals row count 1 by convention."""
    validate_nodes(nodes)
    a = arms.copy()
    if len(a) == 0:
        if validate_complete and nodes.status.eq("ok").any():
            raise ValueError("Accepted FANS exist, but the arm table is empty")
        for c in ("node_id", "arm_local_id", "parent_status", "parent_n_arms_valid", "fans_n_arms", "status"):
            if c not in a:
                a[c] = pd.Series(dtype="object" if "status" in c else "float64")
        return a
    if not {"node_id", "arm_local_id"}.issubset(a):
        raise ValueError("Arm table must include node_id and arm_local_id")
    if a[["node_id", "arm_local_id"]].isna().any().any() or a.duplicated(["node_id", "arm_local_id"]).any():
        raise ValueError("Arm keys (node_id, arm_local_id) must be unique and non-null")
    meta = nodes[["node_id", "status", "n_arms_valid"]].rename(columns={
        "status": "parent_status", "n_arms_valid": "parent_n_arms_valid"})
    a = a.drop(columns=["parent_status", "parent_n_arms_valid", "fans_n_arms"], errors="ignore")
    a = a.merge(meta, on="node_id", how="left", validate="many_to_one", indicator="_parent_join")
    if not a._parent_join.eq("both").all():
        bad = a.loc[~a._parent_join.eq("both"), "node_id"].head().tolist()
        raise ValueError(f"Orphan arms: no parent record for node IDs {bad}")
    a = a.drop(columns="_parent_join")
    a["fans_n_arms"] = a.parent_n_arms_valid
    a["status"] = a.parent_status
    a["analysis_schema"] = SCHEMA_VERSION
    selected = a.loc[a.parent_status.eq("ok")].copy()
    if validate_complete:
        expected = nodes.loc[nodes.status.eq("ok")].set_index("node_id").n_arms_valid.astype(int).sort_index()
        found = selected.groupby("node_id").size().reindex(expected.index, fill_value=0)
        if not np.array_equal(found.to_numpy(int), expected.to_numpy(int)):
            bad = pd.DataFrame({"expected": expected, "found": found}).query("expected != found").head()
            raise ValueError(f"Incomplete accepted FANS: retained arm counts disagree with parent multiplicity:\n{bad}")
    return (selected if accepted_only else a).reset_index(drop=True)


def audit_catalog(nodes: pd.DataFrame, raw_arms: pd.DataFrame, analysis_arms: pd.DataFrame,
                  segments: pd.DataFrame | None = None) -> dict:
    expected_arms = attach_parent_metadata(raw_arms, nodes, accepted_only=True)
    selected = attach_parent_metadata(analysis_arms, nodes, accepted_only=True)
    if len(selected) != len(analysis_arms):
        raise ValueError("Analysis table still contains arms of rejected parents")
    keys = ["node_id", "arm_local_id"]
    expected_keys = set(map(tuple, expected_arms[keys].to_numpy()))
    if set(map(tuple, selected[keys].to_numpy())) != expected_keys:
        raise ValueError("Analysis arm keys differ from the accepted subset of raw arms")
    for col in ("S_tangential", "ra", "dec"):
        if col not in selected or not np.isfinite(pd.to_numeric(selected[col], errors="coerce")).all():
            raise ValueError(f"Nonfinite or missing {col} in accepted analysis arms")
    if not np.isfinite(effective_redshift(selected)).all():
        raise ValueError("Nonfinite redshift in accepted analysis arms")
    if (selected.S_tangential.abs() > 1.0 + 1e-10).any():
        raise ValueError("Arm winding outside [-1,1]")
    if "mean_t_phi" in expected_arms:
        check = expected_arms[keys + ["mean_t_phi"]].merge(selected[keys + ["S_tangential"]], on=keys, validate="one_to_one")
        if not np.allclose(check.mean_t_phi, check.S_tangential, rtol=1e-12, atol=1e-13):
            raise ValueError("Analysis h_i differs from raw arm winding")
    if segments is not None and len(segments):
        if segments.duplicated(keys + ["arm_segment_index"]).any():
            raise ValueError("Duplicate arm-segment keys")
        seg = segments.merge(expected_arms[keys], on=keys, how="inner", validate="many_to_one")
        nseg = seg.groupby(keys).size()
        for _, row in expected_arms.iterrows():
            nk = int(nseg.get((row.node_id, row.arm_local_id), 0))
            if nk != int(row.n_arm_segments_valid):
                raise ValueError("Accepted arm segment count does not match its segment table")
        # Recompute Equation (2), using ridge-point counts rather than density weights.
        if {"n_points", "t_phi"}.issubset(seg):
            seg = seg.copy()
            seg["_wtphi"] = seg.n_points * seg.t_phi
            sums = seg.groupby(keys)[["_wtphi", "n_points"]].sum()
            sums["_h_recomputed"] = sums._wtphi / sums.n_points
            chk = expected_arms.merge(sums[["_h_recomputed"]], left_on=keys, right_index=True, validate="one_to_one")
            if not np.allclose(chk.mean_t_phi, chk._h_recomputed, rtol=1e-10, atol=1e-12):
                raise ValueError("Equation (2) disagrees between segment and arm tables")
    ok = nodes.status.eq("ok")
    return {"schema_version": SCHEMA_VERSION, "passed": True,
            "n_nodes_processed": int(len(nodes)), "n_fans_accepted": int(ok.sum()),
            "n_raw_arms": int(len(raw_arms)), "n_analysis_arms": int(len(selected)),
            "n_raw_arms_rejected_parent": int(len(raw_arms) - len(expected_arms)),
            "sum_accepted_parent_multiplicities": int(nodes.loc[ok, "n_arms_valid"].sum()),
            "checks": ["unique parents", "unique arms", "no orphans", "accepted parents only",
                       "complete multiplicities", "finite h/coordinates/redshift", "arm/segment consistency"]}


def write_construction_diagnostics(nodes: pd.DataFrame, outdir: Path) -> None:
    """Expand per-node JSON audit records without exposing them as science arms."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in nodes if c.startswith("audit_") and c != "audit_candidates_json"]
    basecols = [c for c in ("node_id", "status", "ra", "dec", "z_low", "z_mid", "z_high",
                             "edge_keep", "edge_distance_mpc_h", "n_arms_valid") if c in nodes]
    nodes[basecols + cols].to_csv(outdir / "construction_node_audit.csv", index=False)
    rows = []
    if "audit_candidates_json" in nodes:
        for _, row in nodes.iterrows():
            val = row.get("audit_candidates_json")
            if not isinstance(val, str) or not val:
                continue
            for rec in json.loads(val):
                rows.append({"node_id": int(row.node_id), "parent_status": row.status,
                             "edge_keep": row.get("edge_keep", False), **rec})
    pd.DataFrame(rows, columns=["node_id", "parent_status", "edge_keep", "candidate_id", "n_seed_points",
                               "seed_phi_deg", "n_segments_valid", "passes_arm_cut", "stop_reason",
                               "stop_segment", "n_points_at_stop"]).to_csv(outdir / "construction_arm_audit.csv", index=False)
