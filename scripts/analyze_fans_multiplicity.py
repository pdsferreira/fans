#!/usr/bin/env python3
"""Run the referee diagnostics on a completed v88 projected catalogue."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from spirality.multiplicity import catalog_directory, analyze_catalog, save_analysis, plot_analysis
from spirality.selection import audit_catalog
from spirality.provenance import file_digest


def update_primary_projected_summary(result, catalog_dir: Path, run_root: Path, key: str):
    """Replace main error columns with the same paired bootstrap used for controls."""
    stats=result["statistics"]
    sub=stats[stats.stratum.eq("all")&stats.scope.eq("zbin")].set_index("zbin_id")
    paths=[catalog_dir/"zbin_summary_arms_n_segments.csv"]
    if key=="chen": paths.append(run_root/"chen_dr12_native_arms/zbin_summary_arms_n_segments.csv")
    for path in paths:
        if not path.exists(): continue
        old=pd.read_csv(path)
        for idx,row in old.iterrows():
            r=sub.loc[int(row.zbin_id)]
            for new,mean_col,err_col in [("H","H_S_weighted_mean","H_S_bootstrap_err"),
                                          ("S","mean_abs_projected_twist","mean_abs_projected_twist_bootstrap_err"),
                                          ("A_sign","sign_asymmetry","sign_asymmetry_err")]:
                if mean_col in old and np.isfinite(row.get(mean_col,np.nan)) and not np.isclose(row[mean_col],r[new],rtol=1e-10,atol=1e-12):
                    raise RuntimeError(f"Independent projected summaries disagree in {path}, bin {row.zbin_id}, {new}")
                old.loc[idx,mean_col]=r[new]; old.loc[idx,err_col]=r[f"{new}_err"]
                if new in {"H","S"}:
                    prefix="H_S" if new=="H" else "mean_abs_projected_twist"
                    old.loc[idx,f"{prefix}_bootstrap_q16"]=r[f"{new}_q16"]
                    old.loc[idx,f"{prefix}_bootstrap_q84"]=r[f"{new}_q84"]
            old.loc[idx,"sign_asymmetry_percent"]=100*r.A_sign
            old.loc[idx,"sign_asymmetry_percent_err"]=100*r.A_sign_err
            old.loc[idx,"sign_asymmetry_percent_bootstrap_q16"]=100*r.A_sign_q16
            old.loc[idx,"sign_asymmetry_percent_bootstrap_q84"]=100*r.A_sign_q84
            old.loc[idx,"error_model"]="joint_spatial_block_bootstrap"
            old.loc[idx,"bootstrap_nside"]=result["metadata"]["nside"]
            old.loc[idx,"n_fans"]=r.n_fans; old.loc[idx,"n_arms"]=r.n_arms
        old.to_csv(path,index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",required=True); p.add_argument("--catalogs",nargs="+",default=["carron","chen"],choices=["carron","chen"])
    p.add_argument("--zbin-edges",default="0.1,0.225,0.35,0.475,0.6")
    p.add_argument("--bootstrap-B",type=int,default=1000); p.add_argument("--null-B",type=int,default=1000)
    p.add_argument("--nside",type=int,default=16); p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--min-fans",type=int,default=10); p.add_argument("--min-blocks",type=int,default=5)
    p.add_argument("--plot-format",default="pdf",choices=["pdf","png"])
    p.add_argument("--refresh-projected-errors",action="store_true")
    p.add_argument("--skip-plots",action="store_true")
    args=p.parse_args(); root=Path(args.run_root).expanduser().resolve()
    edges=np.array([float(v) for v in args.zbin_edges.split(",")])
    all_stats=[]; all_frac=[]; all_null=[]; all_trends=[]
    for key in args.catalogs:
        cat=catalog_directory(root,key)
        nodes=pd.read_csv(cat/"node_spirality.csv"); raw=pd.read_csv(cat/"arm_spirality.csv")
        arms=pd.read_csv(cat/"arm_level_spirality.csv"); seg=pd.read_csv(cat/"arm_segments.csv")
        audit=audit_catalog(nodes,raw,arms,seg)
        if "audit_m_candidates" not in nodes:
            raise ValueError("Construction audit missing. Rebuild v88 geometry; old products cannot supply before-selection multiplicities.")
        result=analyze_catalog(nodes,arms,edges=edges,B=args.bootstrap_B,null_B=args.null_B,
                               seed=args.seed+(1000 if key=="chen" else 0),nside=args.nside,
                               min_fans=args.min_fans,min_blocks=args.min_blocks)
        result["metadata"]["analysis_arm_sha256"]=file_digest(cat/"arm_level_spirality.csv")
        result["metadata"]["parent_sha256"]=file_digest(cat/"node_spirality.csv")
        out=root/"referee_multiplicity"/key
        save_analysis(result,nodes,edges=edges,outdir=out)
        (out/"sample_integrity.json").write_text(json.dumps(audit,indent=2))
        if args.refresh_projected_errors: update_primary_projected_summary(result,cat,root,key)
        if not args.skip_plots: plot_analysis(out,key,args.plot_format)
        for name,target in [("statistics",all_stats),("fractions",all_frac),("null_summary",all_null),("trends",all_trends)]:
            tab=result[name].copy(); tab.insert(0,"catalog",key); target.append(tab)
        print(f"[{key}] accepted FANS={audit['n_fans_accepted']:,}; arms={audit['n_analysis_arms']:,}; rejected-parent raw arms={audit['n_raw_arms_rejected_parent']:,}")
    tabledir=root/"paper_tables"; tabledir.mkdir(exist_ok=True)
    for name,parts in [("multiplicity_statistics",all_stats),("multiplicity_fractions",all_frac),("projected_significance_joint",all_null),("spirality_trends",all_trends)]:
        pd.concat(parts,ignore_index=True).to_csv(tabledir/f"{name}.csv",index=False)
    if args.refresh_projected_errors and not args.skip_plots:
        from spirality.pipeline import make_joint_plots
        make_joint_plots(sconce_dir=root/"sconce_dr17_native",carron_dir=catalog_directory(root,"carron"),
                         chen_node_dir=catalog_directory(root,"chen"),chen_arm_dir=root/"chen_dr12_native_arms",
                         outdir=root/"three_catalog_plots",fmt=args.plot_format,zmin=None,zmax=None,filter_mode="none")

if __name__=="__main__": main()
