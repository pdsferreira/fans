#!/usr/bin/env python3
"""Reproducible v88.2 fiducial or 27-point projected-only revision runs.

The default never overwrites a run. Explicit --overwrite rebuilds v88 catalogue directories;
existing partial runs require --overwrite;
completed projected stages can be resumed only with matching code/input metadata
and options. The grid NEVER invokes the extrinsic executable.
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spirality.provenance import code_fingerprint, environment_versions, input_metadata


def command(cmd, *, env, log: Path | None, dry_run: bool):
    text=shlex.join(map(str,cmd));print(text,flush=True)
    if dry_run:return
    if log:
        with log.open("a") as f:f.write(text+"\n")
    subprocess.run(list(map(str,cmd)),check=True,env=env,cwd=ROOT)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["fiducial","grid"]);p.add_argument("--outroot",required=True)
    p.add_argument("--stage",choices=["projected","extrinsic","all"],default="projected")
    p.add_argument("--data-dir",default=str(ROOT / "data"))
    p.add_argument("--sconce-fits");p.add_argument("--chen-intersections");p.add_argument("--chen-fmaps")
    p.add_argument("--carron-files",nargs=3)
    p.add_argument("--n-jobs",type=int,default=32);p.add_argument("--blas-threads",type=int,default=1)
    p.add_argument("--bootstrap-B",type=int,default=1000);p.add_argument("--null-B",type=int,default=1000)
    p.add_argument("--nside-bootstrap",type=int,default=16);p.add_argument("--nside-scan",type=int,default=32)
    p.add_argument("--seed",type=int,default=2026);p.add_argument("--plot-format",choices=["pdf","png"],default="pdf")
    p.add_argument("--galleries",action="store_true",help="Fiducial native/random galleries (not repeated over the grid)")
    p.add_argument("--fiducial-run",help="Completed fiducial root for grid plot reference error bars")
    p.add_argument("--resume",action="store_true");p.add_argument("--overwrite",action="store_true")
    p.add_argument("--dry-run",action="store_true");p.add_argument("--hash-inputs",action="store_true")
    p.add_argument("--skip-fits-export",action="store_true")
    p.add_argument("--paper-outdir", help="Common paper collection. Default: <outroot parent>/paper_final_plots_and_tables")
    p.add_argument("--skip-paper-products", action="store_true", help="Leave paper plotting/collection for a separate postprocessing command")
    p.add_argument("--no-paper-footprints", action="store_true", help="Do not regenerate the three fiducial sky-map crops")
    p.add_argument("--paper-style-json", help="Optional author-style plot configuration")
    return p


def main():
    args=parser().parse_args()
    if args.mode=="grid" and args.stage!="projected":
        raise SystemExit("The grid is PROJECTED ONLY. Extrinsic work is supported only for the fiducial run.")
    if args.n_jobs<1 or args.bootstrap_B<0 or args.null_B<0:raise SystemExit("Invalid jobs/bootstrap/null count")
    d=Path(args.data_dir).expanduser()
    inputs=[Path(args.sconce_fits) if args.sconce_fits else d/"nodes_scms.fits",
            Path(args.chen_intersections) if args.chen_intersections else d/"dr12_Fint.txt",
            Path(args.chen_fmaps) if args.chen_fmaps else d/"dr12_FMaps_full.txt"]
    inputs += [Path(x) for x in args.carron_files] if args.carron_files else [d/f"Block{i}.csv" for i in [1,2,3]]
    versions=environment_versions()
    if not args.dry_run:
        missing=[k for k,v in versions.items() if v=="not installed"]
        if missing:raise SystemExit(f"Missing dependencies: {', '.join(missing)}. Install requirements.txt first.")
        for path in inputs:
            if not path.exists():raise FileNotFoundError(path)
    meta=input_metadata(inputs,hash_contents=args.hash_inputs) if not args.dry_run else [{"path":str(p)} for p in inputs]
    source=code_fingerprint(ROOT);outroot=Path(args.outroot).expanduser().resolve()
    env=os.environ.copy();env["PYTHONPATH"]=str(ROOT)+os.pathsep+env.get("PYTHONPATH","")
    for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:env[key]=str(args.blas_threads)
    edges="0.1,0.225,0.35,0.475,0.6"
    params=list(itertools.product([7.5,10.,12.5],[8.,10.,12.],[30.,35.,40.])) if args.mode=="grid" else [(10.,10.,35.)]
    for L,R,A in params:
        def tag(x):return f"{x:g}".replace(".","p")
        name=f"v88_{args.mode}_z010_z060_L{tag(L)}_N3_R{tag(R)}_A{tag(A)}"
        run=outroot/name;state=run/"revision_projected_complete.json"
        signature={"source_sha256":source,"inputs":meta,"L":L,"R":R,"A":A,"B":args.bootstrap_B,
                   "null_B":args.null_B,"nside":args.nside_bootstrap,"seed":args.seed,"edges":edges,
                   "galleries":bool(args.galleries and args.mode=="fiducial"),"plot_format":args.plot_format,
                   "export_fits":not args.skip_fits_export,"schema":"88.1"}
        projected=args.stage in ["projected","all"]
        if args.resume and state.exists():
            stored=json.loads(state.read_text())
            if stored.get("signature")!=signature:raise ValueError(f"Refusing incompatible resume of {run}")
            if not args.dry_run:
                for cat in ["chen","carron"]:
                    if not (run/"referee_multiplicity"/cat/"sample_integrity.json").exists():raise FileNotFoundError("Resume products missing")
            projected=False
        if not args.dry_run:
            if projected and run.exists() and any(run.iterdir()) and not args.overwrite:
                raise FileExistsError(f"{run} exists. Use a new --outroot, --resume for matching completed runs, or --overwrite for v88 products.")
            run.mkdir(parents=True,exist_ok=True)
            (run/"revision_manifest.json").write_text(json.dumps({"signature":signature,"environment":versions,
                "launcher_args":vars(args),"blas_threads":args.blas_threads},indent=2))
        log=run/"revision_commands.log"
        if projected:
            if not args.dry_run and state.exists(): state.unlink()
            baseB=args.bootstrap_B if args.mode=="fiducial" else 0
            baseNull=args.null_B if args.mode=="fiducial" else 0
            cmd=[sys.executable,"scripts/run_spirality_pipeline.py","--outroot",str(outroot),"--run-name",name,
                 "--pipelines","carron","chen","--parallel-pipelines","--pipeline-workers","2","--n-jobs",args.n_jobs,
                 "--sconce-fits",inputs[0],"--chen-intersections",inputs[1],"--chen-fmaps",inputs[2],"--carron-files",*inputs[3:],
                 "--segment-length-mpc-h",L,"--n-arm-segments","3","--node-link-radius-mpc-h",R,"--arm-opening-deg",A,
                 "--edge-margin-mpc-h","70","--knot-cluster-radius-mpc-h","5","--analysis-product","fiducial",
                 "--node-weight","mean_density","--summary-error-model","bootstrap","--summary-bootstrap-mode","block",
                 "--summary-bootstrap-nside",args.nside_bootstrap,"--arm-summary-weights","n_segments",
                 "--bootstrap-B",baseB,"--signflip-B",baseNull,"--seed",args.seed,"--plot-format",args.plot_format]
            for cat in ["carron","chen"]:
                cmd += [f"--{cat}-z-min","0.1",f"--{cat}-z-max","0.6",f"--{cat}-zbin-edges",edges]
            if args.overwrite:cmd += ["--overwrite"]
            if args.galleries and args.mode=="fiducial":
                cmd += ["--make-gallery","--gallery-seed","2026","--gallery-ncols","8","--gallery-nrows","12",
                        "--make-random-fans-gallery","--random-fans-gallery-n-per-arm-count","1000","--random-fans-gallery-arm-counts","2","3","4"]
            else:cmd += ["--no-make-gallery"]
            command(cmd,env=env,log=log,dry_run=args.dry_run)
            command([sys.executable,"scripts/analyze_fans_multiplicity.py","--run-root",run,"--bootstrap-B",args.bootstrap_B,
                     "--null-B",args.null_B,"--nside",args.nside_bootstrap,"--seed",args.seed,"--plot-format",args.plot_format,
                     "--refresh-projected-errors"],env=env,log=log,dry_run=args.dry_run)
            command([sys.executable,"scripts/summarize_fans_counts.py","--run-root",run,"--catalogs","carron","chen"],env=env,log=log,dry_run=args.dry_run)
            if not args.skip_fits_export:
                command([sys.executable,"scripts/export_fans_fits.py","--run-root",run,"--catalogs","carron","chen",
                         "--write-quality-copy","--overwrite"],env=env,log=log,dry_run=args.dry_run)
            if not args.dry_run:state.write_text(json.dumps({"signature":signature,"completed_unix_time":time.time()},indent=2))
        if args.stage in ["extrinsic","all"]:
            if not args.dry_run:
                if not state.exists(): raise FileNotFoundError("Run the corrected projected stage first")
                completed = json.loads(state.read_text())["signature"]
                # Plot/gallery/export choices are irrelevant when adding inference to identical data.
                invariant = ["source_sha256", "inputs", "L", "R", "A", "B", "null_B", "nside", "seed", "edges", "schema"]
                if any(completed.get(k) != signature.get(k) for k in invariant):
                    raise ValueError("Projected run does not match this code/input/statistical configuration. Rebuild in a new output root.")
            cmd=[sys.executable,"scripts/run_extrinsic_spirality.py","--run-root",run,"--catalogs","carron","chen",
                 "--weight-mode","n_segments","--nside-scan",args.nside_scan,"--bootstrap-B",args.bootstrap_B,
                 "--nside-bootstrap",args.nside_bootstrap,"--seed",args.seed,"--extrinsic-null-B",args.null_B,
                 "--extrinsic-null-modes","block","--extrinsic-null-block-nside",args.nside_bootstrap,
                 "--carron-zbin-edges",edges,"--chen-zbin-edges",edges,"--extrinsic-axis-objective","dipole",
                 "--zbin-axis-mode","fixed_global","--panel-extrinsic-sigma-source","global",
                 "--axis-legend-error-source","bootstrap","--plot-format",args.plot_format]
            command(cmd,env=env,log=log,dry_run=args.dry_run)
        print(f"[revision] {run}",flush=True)
    if args.mode=="grid":
        cmd=[sys.executable,"scripts/plot_multiplicity_grid.py","--grid-root",outroot,"--plot-format",args.plot_format]
        if args.fiducial_run:cmd += ["--fiducial-run",args.fiducial_run]
        command(cmd,env=env,log=None,dry_run=args.dry_run)
        if not args.skip_fits_export:
            command([sys.executable,"scripts/consolidate_fans_fits.py","--grid-root",outroot,"--catalogs","carron","chen",
                     "--accepted-only","--output",outroot/"fans_grid27_quality_consolidated.fits","--overwrite"],env=env,log=None,dry_run=args.dry_run)

    if not args.skip_paper_products:
        paper_out=Path(args.paper_outdir).expanduser().resolve() if args.paper_outdir else outroot.parent/"paper_final_plots_and_tables"
        cmd=[sys.executable,"scripts/build_paper_products.py","--outdir",paper_out]
        if args.mode=="fiducial":cmd += ["--fiducial-run",run]
        else:
            cmd += ["--grid-root",outroot]
            if args.fiducial_run:cmd += ["--fiducial-run",args.fiducial_run]
        if args.no_paper_footprints:cmd += ["--no-footprints"]
        if args.paper_style_json:cmd += ["--style-json",args.paper_style_json]
        command(cmd,env=env,log=None,dry_run=args.dry_run)

if __name__=="__main__":main()
