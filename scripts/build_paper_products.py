#!/usr/bin/env python3
"""Generate author-style plots and collect final tables, WITHOUT redoing inference."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spirality.paper_products import build_paper_bundle

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fiducial-run',help='Completed v88 fiducial run, not its parent directory')
    p.add_argument('--grid-root',help='Directory containing the 27 completed revised runs')
    p.add_argument('--outdir',required=True,help='Common final folder; flat managed plots/tables, manual additions preserved')
    p.add_argument('--style-json',help='Optional editable plotting choices; defaults match the attached author script')
    p.add_argument('--no-footprints',action='store_true',help='Only omit sky maps, e.g. in an environment without healpy/astropy')
    p.add_argument('--allow-partial-grid',action='store_true',help='Diagnostic incomplete grid; visibly labelled, never called grid27')
    p.add_argument('--require-extrinsic',action='store_true',help='Fail when corrected fiducial extrinsic products are missing')
    p.add_argument('--reset-sources',action='store_true',help='Forget stored source paths before this update')
    args=p.parse_args()
    result=build_paper_bundle(args.outdir,fiducial_run=args.fiducial_run,grid_root=args.grid_root,style_path=args.style_json,
            footprints=not args.no_footprints,allow_partial_grid=args.allow_partial_grid,
            require_extrinsic=args.require_extrinsic,reset_sources=args.reset_sources)
    print(f"[paper] {len(result['files'])} files in {Path(args.outdir).expanduser().resolve()}")
    for status in result['status']:print(f"[paper-status] {status['product']}: {status['status']}")

if __name__=='__main__':main()
