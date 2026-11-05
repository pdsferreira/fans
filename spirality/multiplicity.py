"""Referee revision: FANS multiplicity, conditional spirality, composition control.

All scientific estimates remain ARM weighted (W = valid segment count). Node
fractions and estimator-weight fractions are different products. One spatial
block draw is shared by all bins and multiplicity classes within a catalogue.
No independence is assumed between Chen and Carron, or between grid choices.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .selection import attach_parent_metadata, effective_redshift, redshift_bin_ids, SCHEMA_VERSION
from .resampling import healpix_blocks, block_draw_counts, safe_ratio, bootstrap_summary, covariance_complete

CATALOG_DIRS = {
    "chen": ["chen_dr12_native"],
    "carron": ["carron_dr17_proxy/carron_proxy_ridges", "carron_dr16_proxy/carron_proxy_ridges"],
}


def catalog_directory(run_root: Path, catalog: str) -> Path:
    for rel in CATALOG_DIRS[catalog]:
        d = Path(run_root) / rel
        if (d / "node_spirality.csv").exists():
            return d
    raise FileNotFoundError(f"No {catalog} node table under {run_root}")


def construction_tables(nodes: pd.DataFrame, edges: np.ndarray):
    """Stage distributions with explicit populations and unknown denominators."""
    zbin = redshift_bin_ids(effective_redshift(nodes), edges)
    status = nodes.status.astype(str)
    # Missing-slice rows must not be treated as evaluated zero-arm systems.
    evaluated = nodes.get("audit_evaluated", pd.Series(False, index=nodes.index)).astype(str).str.lower().eq("true").to_numpy()
    edge = nodes.get("edge_keep", pd.Series(False, index=nodes.index)).astype(str).str.lower().eq("true").to_numpy()
    accepted = status.eq("ok").to_numpy()
    stagecols = {"raw_seed_groups": "audit_m_raw_groups", "oriented_uncapped_groups": "audit_m_oriented_uncapped",
                 "candidate_arms": "audit_m_candidates", "trace_survivors": "audit_m_valid", "science_valid_arms": "n_arms_valid"}
    rows, transitions, statuses = [], [], []
    scopes = [(-1, "full", np.ones(len(nodes), bool))] + [(i, "zbin", zbin == i) for i in range(len(edges)-1)]
    for bid, scope, scope_mask in scopes:
        for st, n in status[scope_mask].value_counts(dropna=False).items():
            statuses.append({"scope": scope, "zbin_id": bid, "status": st, "n_nodes": int(n)})
        for population, mask in [("all_processed", scope_mask), ("edge_safe", scope_mask & edge),
                                  ("accepted_fans", scope_mask & accepted)]:
            for stage, col in stagecols.items():
                v = pd.to_numeric(nodes.get(col, pd.Series(np.nan, index=nodes.index)), errors="coerce").to_numpy(float)
                known = np.isfinite(v) & mask
                if stage != "science_valid_arms":
                    known &= evaluated
                levels = sorted(set([0, 1, 2, 3, 4]) | set(v[known].astype(int)))
                for m in levels + [-1]:
                    count = int(np.count_nonzero(known & (v == m))) if m >= 0 else int(np.count_nonzero(mask & ~known))
                    rows.append({"population": population, "scope": scope, "zbin_id": bid, "stage": stage,
                                 "multiplicity": m, "multiplicity_label": str(m) if m >= 0 else "not_evaluated",
                                 "n_nodes": count, "n_population": int(mask.sum()), "n_evaluated": int(known.sum()),
                                 "fraction_population": count / mask.sum() if mask.any() else np.nan,
                                 "fraction_evaluated": count / known.sum() if m >= 0 and known.any() else np.nan})
        # Paired transition on the SAME parent population; no mixing denominators.
        for edge_state in [False, True]:
            mask = scope_mask & evaluated & (edge == edge_state)
            if not {"audit_m_candidates", "audit_m_valid"}.issubset(nodes):
                continue
            t = nodes.loc[mask, ["audit_m_candidates", "audit_m_valid", "status"]].copy()
            for (before, after, st), n in t.groupby(["audit_m_candidates", "audit_m_valid", "status"], dropna=False).size().items():
                transitions.append({"scope": scope, "zbin_id": bid, "edge_safe": edge_state,
                                    "m_candidates": before, "m_trace_survivors": after, "parent_status": st,
                                    "n_nodes": int(n)})
    return pd.DataFrame(rows), pd.DataFrame(transitions), pd.DataFrame(statuses)


def analyze_catalog(nodes: pd.DataFrame, arms: pd.DataFrame, *, edges: np.ndarray,
                    B: int = 1000, null_B: int = 1000, seed: int = 2026, nside: int = 16,
                    target_multiplicities=(2, 3), min_fans: int = 10, min_blocks: int = 5,
                    block_ids: np.ndarray | None = None) -> dict:
    """Measure conditional and standardized statistics, with shared block draws.

    block_ids is an explicit array for *accepted nodes within edges*, useful for
    deterministic unit tests. Production CLI always computes actual HEALPix IDs.
    Fixed-composition weights are pooled arm weights in the stated target
    classes; they are re-estimated in each bootstrap draw to propagate reference
    estimation uncertainty. The same reference is used for EVERY bin in a draw.
    """
    edges = np.asarray(edges, float)
    a = attach_parent_metadata(arms, nodes, accepted_only=True)
    n = nodes.loc[nodes.status.eq("ok")].copy()
    n["z_eff"] = effective_redshift(n)
    n["zbin_id"] = redshift_bin_ids(n.z_eff.to_numpy(), edges)
    n = n.loc[n.zbin_id >= 0].reset_index(drop=True)
    if len(n) == 0:
        raise ValueError("No accepted FANS in the requested redshift interval")
    a = a.loc[a.node_id.isin(n.node_id)].reset_index(drop=True)
    use_healpix = block_ids is None
    if use_healpix:
        block_ids = healpix_blocks(n.ra.to_numpy(float), n.dec.to_numpy(float), nside)
    if len(block_ids) != len(n):
        raise ValueError("block_ids must correspond to accepted in-range parent nodes")
    unique, inv, counts = block_draw_counts(block_ids, B, seed)
    n["_block"] = inv
    nmeta = n.set_index("node_id")
    ai = nmeta.index.get_indexer(a.node_id)
    if np.any(ai < 0):
        raise ValueError("Unexpected orphan arm after selection")
    ablock = inv[ai]
    abin = n.zbin_id.to_numpy(int)[ai]
    hcol = "S_tangential" if "S_tangential" in a else "mean_t_phi"
    h = pd.to_numeric(a[hcol], errors="coerce").to_numpy(float)
    wcol = "n_arm_segments_total" if "n_arm_segments_total" in a else "n_arm_segments_valid"
    w = pd.to_numeric(a[wcol], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(h) & np.isfinite(w) & (w > 0)):
        raise ValueError("Nonfinite winding or invalid arm weights; do not silently change a FANS multiplicity")
    mult = a.parent_n_arms_valid.to_numpy(int)
    classes = np.array(sorted(set([2, 3, 4]) | set(mult)), dtype=int)
    ng, nb, nm = len(unique), len(edges) - 1, len(classes)
    mi = {m: i for i, m in enumerate(classes)}
    acell = abin * nm + np.array([mi[m] for m in mult])
    ncell = n.zbin_id.to_numpy(int) * nm + np.array([mi[int(m)] for m in n.n_arms_valid])
    nc = nb * nm

    def reduce(values, blocks=ablock, cells=acell):
        return np.bincount(blocks * nc + cells, weights=values, minlength=ng * nc).reshape(ng, nc)
    block = {"W": reduce(w), "WH": reduce(w*h), "WS": reduce(w*np.abs(h)),
             "N_arm": reduce(np.ones(len(a))), "sign": reduce(np.sign(h)), "N_sign": reduce((h != 0).astype(float)),
             "N_node": reduce(np.ones(len(n)), inv, ncell), "WZ": reduce(w*n.z_eff.to_numpy(float)[ai])}
    # Dimensions: draws (observation then B replicates), bins, multiplicity classes.
    draws = np.vstack([np.ones((1, ng)), counts])
    totals = {key: (draws @ val).reshape(B+1, nb, nm) for key, val in block.items()}
    cell_blocks = (block["N_node"] > 0).sum(axis=0).reshape(nb, nm)
    rows, Hdraws, Sdraws, Adraws, refs = [], [], [], [], []
    specs = [("all", np.ones(nm, bool))] + [(f"m{m}", classes == m) for m in classes]
    scopes = [(-1, np.ones(nb, bool))] + [(i, np.arange(nb) == i) for i in range(nb)]
    for label, cmask in specs:
        for bid, bmask in scopes:
            sel = bmask[:, None] & cmask[None, :]
            v = {key: arr[:, sel].sum(axis=1) for key, arr in totals.items()}
            hs, ss, aa = safe_ratio(v["WH"],v["W"]), safe_ratio(v["WS"],v["W"]), safe_ratio(v["sign"],v["N_sign"])
            row = {"stratum": label, "scope": "full" if bid < 0 else "zbin", "zbin_id": bid,
                   "z_min": edges[0] if bid < 0 else edges[bid], "z_max": edges[-1] if bid < 0 else edges[bid+1],
                   "z_center": float(safe_ratio(v["WZ"][0],v["W"][0])), "H": hs[0], "S": ss[0], "A_sign": aa[0],
                   "n_fans": int(v["N_node"][0]), "n_arms": int(v["N_arm"][0]), "weight_sum": v["W"][0],
                   "n_blocks": int(np.count_nonzero(block["N_node"][:, sel.ravel()].sum(axis=1)>0))}
            row["precision_support"] = row["n_fans"] >= min_fans and row["n_blocks"] >= min_blocks
            for stat, arr in [("H", hs), ("S", ss), ("A_sign", aa)]:
                info = bootstrap_summary(arr[1:])
                if row["n_blocks"] < 2 or row["n_fans"] < 2:
                    for q in ("err", "q16", "q84", "q025", "q975"):
                        info[q] = np.nan
                row.update({f"{stat}_{key}": val for key, val in info.items()})
            rows.append(row); Hdraws.append(hs[1:]); Sdraws.append(ss[1:]); Adraws.append(aa[1:])
    stat_table = pd.DataFrame(rows)
    # Node-number fraction vs estimator-weight fraction, with their own errors.
    fraction_rows = []
    for bid, bmask in scopes:
        cnt = totals["N_node"][:, bmask, :].sum(axis=1)
        wg = totals["W"][:, bmask, :].sum(axis=1)
        f = safe_ratio(cnt, cnt.sum(axis=1)[:,None]); pi = safe_ratio(wg,wg.sum(axis=1)[:,None])
        for j,m in enumerate(classes):
            r={"scope":"full" if bid<0 else "zbin", "zbin_id":bid, "multiplicity":int(m),
               "n_fans":int(cnt[0,j]), "n_total_fans":int(cnt[0].sum()), "fraction_nodes":f[0,j],
               "fraction_weight":pi[0,j], "weight_sum":wg[0,j],
               "n_blocks":int(np.count_nonzero(block["N_node"].reshape(ng,nb,nm)[:,bmask,j].sum(axis=1)>0))}
            for prefix, val in [("fraction_nodes",f[:,j]),("fraction_weight",pi[:,j])]:
                r.update({f"{prefix}_{k}":v for k,v in bootstrap_summary(val[1:]).items()})
            fraction_rows.append(r)
    fractions = pd.DataFrame(fraction_rows)

    # Exact full-mixture identity over ALL observed classes, including rare tails.
    scell = safe_ratio(totals["WS"],totals["W"])
    picell = safe_ratio(totals["W"],totals["W"].sum(axis=2)[:,:,None])
    reconstructed = np.sum(np.where(totals["W"]>0,picell*scell,0),axis=2)
    sall = safe_ratio(totals["WS"].sum(axis=2),totals["W"].sum(axis=2))
    if not np.allclose(sall[0], reconstructed[0], equal_nan=True, atol=1e-12):
        raise RuntimeError("S_all != sum_m pi_m S_m")

    # Predeclared standardization over two- and three-arm systems, never pooling
    # an absent 4+ category or silently inventing its S value.
    target = tuple(int(m) for m in target_multiplicities)
    present = all(m in mi for m in target)
    ix = [mi[m] for m in target if m in mi]
    target_supported = present and all((totals["N_node"][0,:,j] >= min_fans).all() and (cell_blocks[:,j] >= min_blocks).all() for j in ix)
    pooled_weight = totals["W"][:,:,ix].sum(axis=1)
    reference = safe_ratio(pooled_weight,pooled_weight.sum(axis=1)[:,None])
    sfix = np.full((B+1,nb),np.nan)
    spooled = safe_ratio(totals["WS"][:,:,ix].sum(axis=2),totals["W"][:,:,ix].sum(axis=2))
    # Conditional on observed target weights, saved separately from primary.
    sfix_cond = np.full_like(sfix,np.nan)
    if target_supported:
        sfix = np.sum(reference[:,None,:] * scell[:,:,ix],axis=2)
        sfix_cond = np.sum(reference[0,None,None,:]*scell[:,:,ix],axis=2)
    for j,m in enumerate(target):
        refs.append({"multiplicity":m,"reference_weight_fraction":reference[0,j] if j < reference.shape[1] else np.nan,
                     "reference_population":"pooled accepted target arms across full z range", "supported":bool(target_supported)})
    composition=[]
    for bid in range(nb):
        r={"zbin_id":bid,"z_min":edges[bid],"z_max":edges[bid+1],"z_center":.5*(edges[bid]+edges[bid+1]),
           "S_all":sall[0,bid],"S_reconstructed":reconstructed[0,bid], "S_target_observed_mix":spooled[0,bid],
           "S_fixed_composition":sfix[0,bid],"mixing_shift":spooled[0,bid]-sfix[0,bid],
           "target_multiplicities":",".join(map(str,target)),"supported":bool(target_supported)}
        for label,val in [("S_fixed_composition",sfix[:,bid]),("S_fixed_conditional_reference",sfix_cond[:,bid]),
                          ("S_target_observed_mix",spooled[:,bid]),("mixing_shift",spooled[:,bid]-sfix[:,bid])]:
            r.update({f"{label}_{k}":v for k,v in bootstrap_summary(val[1:]).items()})
        composition.append(r)
    # Evolution summary: paired endpoint contrasts and descriptive linear slope.
    # Bootstrap tail fractions are explicitly NOT calibrated p-values.
    Sboot = np.asarray(Sdraws).T
    trends=[]; centers=.5*(edges[:-1]+edges[1:]); x=centers-centers.mean(); xnorm=np.sum(x*x)
    trend_series={}
    for label,_ in specs:
        inds=stat_table.index[(stat_table.stratum==label)&(stat_table.scope=="zbin")].to_numpy()
        trend_series[label]=np.vstack([stat_table.loc[inds,"S"].to_numpy(),Sboot[:,inds]])
    trend_series["fixed_composition"]=sfix
    trend_series["target_observed_mix"]=spooled
    for label,arr in trend_series.items():
        delta=arr[:,-1]-arr[:,0]
        slope=(arr @ x)/xnorm
        r={"stratum":label,"delta_S_last_minus_first":delta[0],"slope_S_per_z_descriptive":slope[0]}
        for prefix,val in [("delta_S",delta),("slope_S",slope)]:
            r.update({f"{prefix}_{k}":v for k,v in bootstrap_summary(val[1:]).items()})
        sd=r.get("delta_S_err",np.nan)
        r["delta_S_over_bootstrap_se_descriptive"]=delta[0]/sd if np.isfinite(sd) and sd>0 else np.nan
        trends.append(r)
    # One set of flips across all bins/classes. S never gets a sign-flip test.
    null_rows=[]; nulls={}
    if null_B>0:
        flips=np.random.default_rng(seed+100000).choice([-1.,1.],size=(null_B,ng))
        nfH=(flips@block["WH"]).reshape(null_B,nb,nm)
        nfA=(flips@block["sign"]).reshape(null_B,nb,nm)
        for label,cmask in specs:
            for bid,bmask in scopes:
                sel=bmask[:,None]&cmask[None,:]
                obs=stat_table[(stat_table.stratum==label)&(stat_table.zbin_id==bid)].iloc[0]
                for stat,num,den in [("H",nfH,totals["W"][0]),("A_sign",nfA,totals["N_sign"][0])]:
                    dist=safe_ratio(num[:,sel].sum(axis=1),den[sel].sum())
                    finite=dist[np.isfinite(dist)]; value=float(obs[stat])
                    p=(1+np.count_nonzero(np.abs(finite)>=abs(value)))/(len(finite)+1) if len(finite) and np.isfinite(value) else np.nan
                    qq=np.percentile(finite,[2.2750131948,15.8655253931,50,84.1344746069,97.7249868052]) if len(finite) else [np.nan]*5
                    null_rows.append({"stratum":label,"zbin_id":bid,"statistic":stat,"observed":value,"p_two_sided":p,
                                      "null_B":len(finite),"p_resolution":1/(null_B+1),
                                      **{k:float(v) for k,v in zip(["q2sigma_low","q1sigma_low","q50","q1sigma_high","q2sigma_high"],qq)}})
                    nulls[f"{stat}_{label}_{bid}"]=dist
    return {"statistics":stat_table,"fractions":fractions,"composition":pd.DataFrame(composition),
            "reference":pd.DataFrame(refs),"trends":pd.DataFrame(trends),"null_summary":pd.DataFrame(null_rows),
            "bootstrap":{"H":np.asarray(Hdraws).T,"S":Sboot,"A_sign":np.asarray(Adraws).T,
                         "S_fixed_composition":sfix[1:],"S_fixed_conditional_reference":sfix_cond[1:],
                         "S_target_observed_mix":spooled[1:],"reference_weight_fractions":reference[1:]},
            "null_samples":nulls,
            "metadata":{"schema":SCHEMA_VERSION,"B":B,"null_B":null_B,"seed":seed,"nside":nside,"n_blocks":ng,
                        "weight_mode":"n_segments","block_scheme":"HEALPix RING" if use_healpix else "explicit test/group IDs",
                        "target_multiplicities":list(target),"target_supported":bool(target_supported),
                        "min_fans_per_target_bin":min_fans,"min_blocks_per_target_bin":min_blocks,
                        "reference_bootstrap":"re-estimate pooled reference each draw; shared across all z bins",
                        "notes":"Conditional observed multiplicity, not physical graph connectivity. No cross-catalogue significance combination."}}


def save_analysis(result: dict, nodes: pd.DataFrame, *, edges: np.ndarray, outdir: Path) -> None:
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    for key in ["statistics","fractions","composition","reference","trends","null_summary"]:
        result[key].to_csv(outdir/f"multiplicity_{key}.csv",index=False)
    dist,trans,stat=construction_tables(nodes,edges)
    dist.to_csv(outdir/"multiplicity_selection_stages.csv",index=False)
    trans.to_csv(outdir/"multiplicity_transitions.csv",index=False)
    stat.to_csv(outdir/"selection_status_counts.csv",index=False)
    np.savez_compressed(outdir/"multiplicity_bootstrap_samples.npz",**result["bootstrap"])
    if result["null_samples"]:
        np.savez_compressed(outdir/"multiplicity_signflip_samples.npz",**result["null_samples"])
    result["statistics"][["stratum","scope","zbin_id"]].reset_index(names="column_id").to_csv(outdir/"bootstrap_column_index.csv",index=False)
    covrows=[]
    # Full covariance matrices: same complete draw set per matrix, PSD by construction.
    for key,samples in result["bootstrap"].items():
        cov,ncomplete=covariance_complete(samples)
        np.save(outdir/f"covariance_{key}.npy",cov)
        covrows.append({"statistic":key,"n_complete_draws":ncomplete,"n_variables":samples.shape[1]})
    pd.DataFrame(covrows).to_csv(outdir/"covariance_diagnostics.csv",index=False)
    (outdir/"multiplicity_metadata.json").write_text(json.dumps(result["metadata"],indent=2))


def plot_analysis(outdir: Path, catalog: str, fmt: str="pdf") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir=Path(outdir); label={"carron":"Carrón","chen":"Chen"}.get(catalog,catalog)
    stats=pd.read_csv(outdir/"multiplicity_statistics.csv")
    fractions=pd.read_csv(outdir/"multiplicity_fractions.csv")
    composition=pd.read_csv(outdir/"multiplicity_composition.csv")
    # Full accepted multiplicity distribution: log count makes rare tails visible.
    full = fractions[fractions.scope.eq("full") & fractions.n_fans.gt(0)]
    fig,ax=plt.subplots(figsize=(5.4,3.8))
    bars=ax.bar(full.multiplicity.to_numpy(int),full.n_fans)
    ax.bar_label(bars,labels=[f"{int(n):,}" for n in full.n_fans],padding=3)
    ax.set_xticks(full.multiplicity.to_numpy(int))
    ax.set_yscale("log")
    ax.set(xlabel="Arms per accepted FANS",ylabel="Number of FANS",title=label+": accepted sample",
           ylim=(.7,max(2,float(full.n_fans.max())*2)))
    fig.tight_layout();fig.savefig(outdir/f"{catalog}_accepted_multiplicity_counts.{fmt}");plt.close(fig)
    # Each file is an individual figure, useful in a two-column Letter layout.
    for metric,ylabel in [("fraction_nodes","Fraction of accepted FANS"),("fraction_weight","Fraction of estimator weight")]:
        fig,ax=plt.subplots(figsize=(5.8,4.4))
        for m,s in fractions[fractions.scope.eq("zbin")].groupby("multiplicity"):
            if not s.n_fans.sum(): continue
            b=s.zbin_id.to_numpy(int); c=composition.set_index("zbin_id").loc[b,"z_center"].to_numpy()
            ax.errorbar(c,s[metric],yerr=s[f"{metric}_err"].fillna(0),marker="o",label=f"{m} arms",capsize=2)
        ax.set(xlabel="Redshift z",ylabel=ylabel,title=label,ylim=(0,1)); ax.legend(loc="upper center",bbox_to_anchor=(.5,-.19),ncol=3,frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(outdir/f"{catalog}_{metric}.{fmt}"); plt.close(fig)
    for stat,ylabel in [("S",r"Projected spirality $S_{\rm proj}$"),("H",r"Projected handedness $H_{\rm proj}$"),("A_sign",r"Projected sign asymmetry $A_{\rm sign}$")]:
        fig,ax=plt.subplots(figsize=(5.8,4.4))
        for cl,desc in [("all","All accepted FANS"),("m2","Two-arm FANS"),("m3","Three-arm FANS")]:
            s=stats[stats.stratum.eq(cl)&stats.scope.eq("zbin")]
            if not len(s): continue
            x=.5*(s.z_min+s.z_max)
            ax.errorbar(x,s[stat],yerr=s[f"{stat}_err"].fillna(0),marker="o",capsize=2,label=desc)
        if stat=="S" and composition.supported.astype(str).str.lower().eq("true").all():
            ax.errorbar(composition.z_center,composition.S_fixed_composition,yerr=composition.S_fixed_composition_err.fillna(0),
                        marker="s",linestyle="--",capsize=2,label="Fixed 2/3-arm composition")
        if stat!="S": ax.axhline(0,linewidth=.7,linestyle=":")
        ax.set(xlabel="Redshift z",ylabel=ylabel,title=label); ax.legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.19),ncol=2,frameon=False); fig.tight_layout(); fig.savefig(outdir/f"{catalog}_{stat}_fixed_multiplicity.{fmt}"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.8,4.4))
    if composition.mixing_shift.notna().any():
        ax.errorbar(composition.z_center,composition.mixing_shift,yerr=composition.mixing_shift_err.fillna(0),marker="o",capsize=2)
    else:
        ax.text(.5,.5,"Insufficient common support for 2/3-arm standardization",ha="center",va="center",transform=ax.transAxes,fontsize=8)
    ax.axhline(0,linewidth=.7,linestyle=":"); ax.set(xlabel="Redshift z",ylabel=r"$S_{2+3,\,observed\ mix}-S_{2+3,\,fixed\ mix}$",title=label)
    fig.tight_layout(); fig.savefig(outdir/f"{catalog}_composition_shift.{fmt}"); plt.close(fig)
    dist=pd.read_csv(outdir/"multiplicity_selection_stages.csv")
    fig,ax=plt.subplots(figsize=(5.8,4.4))
    for stage,desc in [("raw_seed_groups","Raw seed groups"),("candidate_arms","Candidate arms"),("trace_survivors","Surviving arms")]:
        s=dist[dist.population.eq("edge_safe")&dist.scope.eq("full")&dist.stage.eq(stage)&dist.multiplicity.ge(0)]
        ax.plot(s.multiplicity,s.fraction_evaluated,"o-",label=desc)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="Operational arm multiplicity",ylabel="Fraction of edge-safe evaluated nodes",title=label)
    ax.legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.19),ncol=2,frameon=False); fig.tight_layout(); fig.savefig(outdir/f"{catalog}_selection_multiplicity.{fmt}"); plt.close(fig)
    trans=pd.read_csv(outdir/"multiplicity_transitions.csv")
    from .publication_matrices import plot_transition_matrix
    plot_transition_matrix(trans,catalog,outdir/f"{catalog}_multiplicity_transitions.{fmt}")
