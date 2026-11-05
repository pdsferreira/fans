#!/usr/bin/env python3
"""Refresh publication figures, with an optional v88.9 sample-only update.

Works with paper_final_plots_and_tables alone; no raw catalogues, bootstrap
reruns, sign-flip simulations or axis scans are needed. Managed files are
verified before writing; manually added files (including FANS panels) survive.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spirality.publication_refresh import refresh_paper_folder

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir',required=True,help='Existing managed paper_final_plots_and_tables directory')
    parser.add_argument('--style-json',help='Optional plot settings, e.g. configs/paper_plots.json')
    parser.add_argument('--fiducial-run',help='Completed fiducial root: supplies original individual node redshifts for the multiplicity KDE; no science rerun')
    parser.add_argument('--only-sample-panel',action='store_true',
                        help='Redraw only fans_sample_multiplicity_panel and its sidecars; preserve all other figures and tables')
    args=parser.parse_args()
    result=refresh_paper_folder(Path(args.paper_dir),style_path=args.style_json,fiducial_run=args.fiducial_run,only_sample_panel=args.only_sample_panel)
    print(f'[plot-only] Collection now contains {len(result["files"])} managed products in {Path(args.paper_dir).resolve()}')
    print('[plot-only] No measurements or statistical ensembles were recomputed.')
    key='appearance_update_88_9' if args.only_sample_panel else 'appearance_update_88_8'
    if args.only_sample_panel:
        print('[plot-only] Only fans_sample_multiplicity_panel and its sidecars were redrawn; other products preserved.')
    state=result.get('metadata',{}).get(key,{}).get('redshift_multiplicity_kde')
    print(f'[redshift KDE] {state}')
    panels=result.get('metadata',{}).get(key,{}).get('editorial_panels',{})
    print(f'[native panels] {panels}')
    if state=='pending_individual_redshifts':
        print('[redshift KDE] Exact four-bin plots were created. Add --fiducial-run for the smooth KDE; no redshifts were invented.')

if __name__=='__main__':main()
