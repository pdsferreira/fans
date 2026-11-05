"""Redraw only the two angular matrices from validated cached summaries.

No NPZ files or original catalogue directories are needed. All cell values,
errors, support flags and captions remain untouched. Only colour normalization,
PDFs, presentation metadata and the managed manifest are refreshed.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from .angular_contrasts import cached_display_cells, PLOT_NAMES, GRID_TABLE, FID_TABLE, CACHE_META
from .paper_products import Bundle, _commit_bundle, load_style
from .publication_matrices import plot_delta_s_matrices
from .provenance import file_digest


def redraw_angular_matrices(paper_dir: str | Path, *, colour_limit: float = 4.5,
                            style_path: Optional[str | Path] = None,
                            check_only: bool = False) -> dict:
    """Verify the collection, stage a colour-only redraw, then commit the two PDFs."""
    folder = Path(paper_dir).expanduser().resolve()
    if not np.isfinite(colour_limit) or colour_limit <= 0:
        raise ValueError('colour-limit must be finite and strictly positive')
    previous = json.loads((folder / 'manifest.json').read_text())
    if previous.get('science_schema') != '88.1':
        raise ValueError('Expected scientific schema 88.1')
    old = {}
    for entry in previous['files']:
        name = entry['file']
        if Path(name).name != name or name in old:
            raise ValueError('Unsafe or duplicate managed filename')
        p = folder / name
        if not p.is_file() or file_digest(p) != entry['sha256']:
            raise ValueError(f'Missing/modified managed file: {p}; preserve manual edits under a different name')
        old[name] = entry
    cells = cached_display_cells(folder)
    if cells is None:
        raise FileNotFoundError('Validated angular summaries are required; run convert_delta_s_to_degrees.py first')
    for stem in PLOT_NAMES:
        for suffix in ('.pdf', '.metadata.json', '.used_values.csv'):
            if stem + suffix not in old:
                raise ValueError(f'Expected managed angular figure/sidecar: {stem + suffix}')
    saturation = {}
    for cat in ('carron', 'chen'):
        c = cells[(cells.catalog == cat) & cells.plot_support]
        saturation[cat] = int((c.delta_S_last_minus_first.abs() > colour_limit).sum())
    summary = {'colour_limits_deg': [-float(colour_limit), float(colour_limit)],
               'colour_saturated_cells': saturation, 'n_cells': len(cells),
               'no_new_resampling': True, 'numbers_unchanged': True}
    if check_only:
        return {**summary, 'check_only': True}
    style = load_style()
    style.update({k: v for k, v in previous.get('plot_style', {}).items() if k in style})
    if style_path:
        extra = json.loads(Path(style_path).read_text())
        unknown = set(extra) - set(style)
        if unknown:
            raise ValueError(f'Unknown style settings: {sorted(unknown)}')
        style.update(extra)
    style['delta_matrix_angular_vmax'] = float(colour_limit)
    style['delta_matrix_angular_allow_colour_saturation'] = True
    with tempfile.TemporaryDirectory(prefix='.angular_colourbar_', dir=folder.parent) as tmp:
        bundle = Bundle(tmp)
        bundle.status = list(previous.get('status', []))
        bundle.deps = {d['path']: d for d in previous.get('dependencies', []) if 'path' in d}
        for entry in previous['files']:
            bundle.add(folder / entry['file'], role=entry.get('role', 'support'),
                       description=entry.get('description', ''))
        # Freeze old CSV/JSON bytes in memory: we only allow two metadata JSONs to change.
        numerical_bytes = {(stem + '.used_values.csv'): (folder / (stem + '.used_values.csv')).read_bytes()
                           for stem in PLOT_NAMES}
        drawstyle = {**style, 'delta_matrix_versions': ['intensity']}
        paths = plot_delta_s_matrices(pd.read_csv(folder / 'grid27_trends.csv'),
                                     pd.read_csv(folder / 'grid27_statistics.csv'),
                                     bundle.stage, style=drawstyle, display_cells=cells, unit='deg')
        for name, data in numerical_bytes.items():
            # Assert numerical equality before preserving exact original serialization.
            pd.testing.assert_frame_equal(pd.read_csv(folder / name),
                                          pd.read_csv(bundle.stage / name), check_exact=False, rtol=1e-12, atol=1e-12)
            (bundle.stage / name).write_bytes(data)
        allowed = {stem + suffix for stem in PLOT_NAMES for suffix in ('.pdf', '.metadata.json')}
        for name, entry in old.items():
            if name not in allowed and file_digest(bundle.stage / name) != entry['sha256']:
                raise RuntimeError(f'Unrelated product would change: {name}')
        for name in allowed:
            bundle.entries = [e for e in bundle.entries if e['file'] != name]
            bundle.add(bundle.stage / name, generated=True, role=old[name].get('role', 'support'),
                       description=old[name].get('description', ''))
        for name in ('angular_plot_refresh.py', 'publication_matrices.py'):
            bundle.dependency(Path(__file__).with_name(name))
        metadata = dict(previous.get('metadata', {}))
        metadata['angular_colourbar_88_13'] = summary
        result = _commit_bundle(bundle, folder, previous.get('sources', {}), style, metadata)
    return result
