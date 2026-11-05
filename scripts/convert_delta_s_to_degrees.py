#!/usr/bin/env python3
"""Convert Figures 5/6 and the referee contrast table from saved paired bootstrap draws.

No reconstruction, bootstrap rerun, sign-flip null or axis search. The completed
fiducial/grid folders must retain their multiplicity_bootstrap_samples.npz files.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spirality.angular_contrasts import convert_collection, FID_TABLE, GRID_TABLE, REF_TABLE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper-dir',required=True)
    p.add_argument('--fiducial-run',help='Completed fiducial run; defaults to collection manifest')
    p.add_argument('--grid-root',help='Root containing all 27 completed grid runs; defaults to manifest')
    p.add_argument('--style-json',help='Optional existing publication style (layout is otherwise preserved)')
    p.add_argument('--check-only',action='store_true',help='Validate all paired archives and calculate summaries without writing anything')
    a=p.parse_args()
    try:
        result=convert_collection(a.paper_dir,fiducial_run=a.fiducial_run,grid_root=a.grid_root,
                                  style_path=a.style_json,check_only=a.check_only)
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as e:
        p.exit(2,f'ERROR: {e}\n')
    if a.check_only:
        print('[OK] Saved paired bootstrap archives validated:',result)
    else:
        print('[OK] Converted only the two intensity matrices and their captions; other scientific files/plots preserved.')
        for name in (FID_TABLE,GRID_TABLE,REF_TABLE):print(Path(a.paper_dir).resolve()/name)
        meta=result['metadata']['angular_contrasts_88_12']
        print('[OK] No new bootstrap draws; stored B values:',meta['bootstrap_B'])
        print('[INFO] 95% interval flags changed after exact angular conversion:',meta['angular_vs_dimensionless_95_flags_changed'])


if __name__=='__main__':main()
