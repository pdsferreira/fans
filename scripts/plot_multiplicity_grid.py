#!/usr/bin/env python3
"""Collect the 27 projected-only runs and plot original plus conditional statistics."""
from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spirality.paper_products import collect_grid_tables, grid_input_digest



def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--grid-root",required=True)
    p.add_argument("--fiducial-run"); p.add_argument("--plot-format",default="pdf",choices=["pdf","png"])
    args=p.parse_args(); root=Path(args.grid_root).expanduser().resolve(); out=root/"grid_summary"; out.mkdir(exist_ok=True)
    merged,edges,meta=collect_grid_tables(root)
    nrun=meta["n_runs"]
    for k,tab in merged.items():tab.to_csv(out/f"grid_{k}.csv",index=False)
    (out/"grid_manifest.json").write_text(json.dumps({"n_runs":nrun,"expected_runs":27,"extrinsic_scans":False,
       "science_input_sha256":grid_input_digest(meta["dependencies"])},indent=2))
    s=merged["statistics"]
    for cat in ["carron","chen"]:
        fid=None
        if args.fiducial_run:
            path=Path(args.fiducial_run)/"referee_multiplicity"/cat/"multiplicity_statistics.csv"
            if not path.exists():raise FileNotFoundError(path)
            fid=pd.read_csv(path)
        for cl in ["all","m2","m3"]:
            for metric in ["H","A_sign","S"]:
                fig,ax=plt.subplots(figsize=(6.1,4.5))
                subset=s[s.catalog.eq(cat)&s.stratum.eq(cl)&s.scope.eq("zbin")]
                for run,g in subset.groupby("run_name"):
                    g=g.sort_values("zbin_id"); r=g.iloc[0]
                    ls={30.:"--",35.:"-",40.:":"}[float(r.arm_opening_deg)]
                    marker={8.:"o",10.:"s",12.:"^"}[float(r.node_link_radius_mpc_h)]
                    ax.plot(.5*(g.z_min+g.z_max),g[metric],linestyle=ls,marker=marker,alpha=.42,lw={7.5:.6,10.:1.2,12.5:1.9}[float(r.segment_length_mpc_h)],ms=3)
                if fid is not None:
                    f=fid[fid.stratum.eq(cl)&fid.scope.eq("zbin")]
                    ax.errorbar(.5*(f.z_min+f.z_max),f[metric],yerr=f[f"{metric}_err"].fillna(0),fmt="o-",lw=2,capsize=3,label="Fiducial")
                if metric!="S":ax.axhline(0,linestyle=":",linewidth=.7)
                ax.set(xlabel="Redshift z",ylabel={"S":r"$S_{\rm proj}$","H":r"$H_{\rm proj}$","A_sign":r"$A_{\rm sign}$"}[metric],
                       title=f"{cat.capitalize()}: {cl}; {nrun} construction choices")
                from matplotlib.lines import Line2D
                handles = [Line2D([], [], linewidth=w, label=f"L={L:g}") for L,w in [(7.5,.6),(10,1.2),(12.5,1.9)]]
                handles += [Line2D([], [], marker=m, linestyle="none", label=f"R={r:g}") for r,m in [(8,"o"),(10,"s"),(12,"^")]]
                handles += [Line2D([], [], linestyle=l, label=f"opening={a:g}°") for a,l in [(30,"--"),(35,"-"),(40,":")]]
                ax.legend(handles=handles,fontsize=7,ncol=3,loc="upper center",bbox_to_anchor=(.5,-.18),frameon=False)
                fig.tight_layout(); fig.savefig(out/f"grid_{cat}_{cl}_{metric}.{args.plot_format}");plt.close(fig)
        fig,ax=plt.subplots(figsize=(5.8,3.9))
        c=merged["composition"];c=c[c.catalog.eq(cat)]
        for run,g in c.groupby("run_name"):
            g=g.sort_values("zbin_id");ax.plot(g.z_center,g.S_fixed_composition,marker="o",alpha=.4,lw=.8,ms=3)
        ax.set(xlabel="Redshift z",ylabel=r"$S_{\rm fixed\ composition}$",title=f"{cat.capitalize()}: common 2/3-arm composition")
        fig.tight_layout();fig.savefig(out/f"grid_{cat}_fixed_composition.{args.plot_format}");plt.close(fig)
    # The 27 alternatives are correlated choices, never independent detections.
    t=merged["trends"]
    summary=t.groupby(["catalog","stratum"]).agg(n_runs=("run_name","nunique"),
              delta_min=("delta_S_last_minus_first","min"),delta_max=("delta_S_last_minus_first","max"),
              n_finite=("delta_S_last_minus_first","count"),
              n_positive=("delta_S_last_minus_first",lambda x:int(np.sum(x>0)))).reset_index()
    summary.to_csv(out/"grid_trend_envelope_descriptive.csv",index=False)
    print(f"[grid] {nrun} runs; figures and tables in {out}")

if __name__=="__main__":main()
