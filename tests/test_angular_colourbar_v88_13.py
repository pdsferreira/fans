"""Presentation-only regression tests for the angular shared colour scale."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from spirality.provenance import file_digest
from spirality.publication_matrices import delta_intensity_limit
from spirality.angular_plot_refresh import redraw_angular_matrices
from spirality.angular_contrasts import convert_collection, GRID_TABLE, FID_TABLE, CACHE_META
from test_angular_contrasts_v88_12 import fixture_collection


def test_explicit_saturation_never_changes_cell_values():
    cells = pd.DataFrame({'plot_support': [True, True, False],
                          'delta_S_last_minus_first': [4.55503192482597, -.31, 10.1]})
    original = cells.copy(deep=True)
    assert delta_intensity_limit(cells, 4.5, allow_saturation=True) == 4.5
    pd.testing.assert_frame_equal(cells, original)
    with pytest.raises(ValueError, match='clip'):
        delta_intensity_limit(cells, 4.5)
    # An automatic scale still ignores the grey, unsupported cell.
    assert delta_intensity_limit(cells) == 5.


@pytest.mark.parametrize('bad', [0., -4.5, float('nan'), float('inf')])
def test_invalid_colour_limit_is_rejected_before_reading(tmp_path, bad):
    with pytest.raises(ValueError, match='strictly positive'):
        redraw_angular_matrices(tmp_path, colour_limit=bad)


def test_redraw_uses_cache_preserves_all_data_and_handles_moved_runs(tmp_path):
    paper, fid, grid = fixture_collection(tmp_path)
    convert_collection(paper)
    shutil.rmtree(fid)
    shutil.rmtree(grid)
    before = {p.name: file_digest(p) for p in paper.iterdir() if p.is_file()}
    checked = redraw_angular_matrices(paper, check_only=True)
    assert checked['colour_limits_deg'] == [-4.5, 4.5]
    assert before == {p.name: file_digest(p) for p in paper.iterdir() if p.is_file()}
    result = redraw_angular_matrices(paper)
    allowed = {'manifest.json', 'manifest.csv', 'paper_sources.json', 'README.md'}
    allowed |= {f'grid27_{c}_delta_S_matrices_intensity{x}'
                for c in ('carron', 'chen') for x in ('.pdf', '.metadata.json')}
    for name, digest in before.items():
        if name not in allowed:
            assert file_digest(paper / name) == digest
    for cat in ('carron', 'chen'):
        m = json.loads((paper / f'grid27_{cat}_delta_S_matrices_intensity.metadata.json').read_text())
        assert m['intensity_vmin'] == -4.5 and m['intensity_vmax'] == 4.5
        assert m['unit'] == 'deg' and m['no_new_resampling']
        assert m['title_x_figure'] == m['subplot_grid_centre_x']
        assert not m['explanatory_notes_drawn']
        assert m['x_ticklabels_bottom_row_only']
    for e in result['files']:
        assert file_digest(paper/e['file']) == e['sha256']
    assert (paper/'manual_plot.pdf').read_bytes() == b'Manual asset, do not touch'
    assert all(file_digest(paper/n) == before[n] for n in (GRID_TABLE, FID_TABLE, CACHE_META))
    # Preflight protects manual changes in managed products.
    (paper/GRID_TABLE).write_text('tampered')
    with pytest.raises(ValueError, match='modified managed'):
        redraw_angular_matrices(paper)
