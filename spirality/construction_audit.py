"""Read-only audit of the unchanged FANS geometric construction.

The audit also evaluates edge-rejected nodes and single candidate arms. Such
counterfactual traces NEVER enter science products. Multiplicity is operational,
not graph connectivity; unmatched/missing input slices remain unassessable.
"""
from __future__ import annotations
import json
import math
import numpy as np


def audit_node(node, sd, cfg, *, project, cluster, circular_mean, trace) -> dict:
    nx, ny = project(np.array([float(node["ra"])]), np.array([float(node["dec"])]),
                     ra0_deg=sd.ra0_deg, dec0_deg=sd.dec0_deg, chi_mpc_h=sd.chi_mpc_h)
    nx, ny = float(nx[0]), float(ny[0])
    idx = np.asarray(sd.tree.query_ball_point([nx, ny], r=cfg.arm_radius_mpc_h + cfg.node_link_radius_mpc_h), dtype=int)
    dx, dy = sd.x_mpc_h[idx] - nx, sd.y_mpc_h[idx] - ny
    rho, phi = np.hypot(dx, dy), np.arctan2(dy, dx)
    seed = np.flatnonzero((rho >= max(float(cfg.inner_exclusion_mpc_h), 1e-6)) & (rho <= cfg.node_link_radius_mpc_h))
    # A common angular grouping before vs after the orientation filter; thinning
    # may SPLIT a chain of angles, so the group counts need not decrease.
    raw = cluster(phi[seed], gap_deg=cfg.cluster_gap_deg, min_points=cfg.min_seed_points,
                  max_clusters=max(1, len(seed)))
    dot = np.abs(sd.e_hat[idx[seed]] * dx[seed] / np.maximum(rho[seed], 1e-12)
                 + sd.n_hat[idx[seed]] * dy[seed] / np.maximum(rho[seed], 1e-12))
    oriented = seed[dot >= math.cos(math.radians(cfg.seed_max_tangent_angle_deg))]
    groups_all = cluster(phi[oriented], gap_deg=cfg.cluster_gap_deg, min_points=cfg.min_seed_points,
                         max_clusters=max(1, len(oriented)))
    groups = groups_all[:int(cfg.max_arms_node)]
    candidates = []
    for i, group in enumerate(groups):
        loc = oriented[group]
        seed_phi = circular_mean(phi[loc])
        audit = {}
        segs = trace(sd, node_x=nx, node_y=ny, all_idx=idx, seed_phi=seed_phi,
                     arm_local_id=i, cfg=cfg, audit=audit)
        candidates.append({"candidate_id": i, "n_seed_points": int(len(loc)),
                           "seed_phi_deg": float(np.rad2deg(seed_phi)),
                           "n_segments_valid": len(segs),
                           "passes_arm_cut": len(segs) >= cfg.min_valid_arm_segments, **audit})
    nvalid = sum(r["passes_arm_cut"] for r in candidates)
    return {"audit_evaluated": True, "audit_trace_evaluated": True,
            "audit_n_near_points": int(len(idx)), "audit_n_seed_raw": int(len(seed)),
            "audit_n_seed_oriented": int(len(oriented)), "audit_m_raw_groups": len(raw),
            "audit_m_oriented_uncapped": len(groups_all), "audit_m_candidates": len(groups),
            "audit_m_valid": int(nvalid), "audit_n_removed_by_cap": len(groups_all) - len(groups),
            "audit_candidates_json": json.dumps(candidates, allow_nan=False)}
