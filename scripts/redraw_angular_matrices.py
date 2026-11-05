#!/usr/bin/env python3
"""Redraw angular Figures 5/6 with a common symmetric colour scale; no new measurements."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spirality.angular_plot_refresh import redraw_angular_matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', required=True, help='Completed compact publication collection')
    parser.add_argument('--colour-limit', type=float, default=4.5, help='Symmetric colour limit in degrees (default: 4.5)')
    parser.add_argument('--style-json', help='Optional publication style overrides')
    parser.add_argument('--check-only', action='store_true', help='Validate without rewriting files')
    args = parser.parse_args()
    try:
        result = redraw_angular_matrices(args.paper_dir, colour_limit=args.colour_limit,
                                        style_path=args.style_json, check_only=args.check_only)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')
    info = result if args.check_only else result['metadata']['angular_colourbar_88_13']
    print('[OK] Common limits [deg]:', info['colour_limits_deg'])
    print('[INFO] Supported cell colours saturated:', info['colour_saturated_cells'])
    print('[OK] Exact numerical data unchanged; no new draws or catalogue processing.')
    if not args.check_only:
        print('[OK] Updated the two angular PDFs, their presentation metadata and the manifest.')


if __name__ == '__main__':
    main()
