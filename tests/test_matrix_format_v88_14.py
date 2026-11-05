"""Regression tests for two-decimal angular display and superscript spacing."""
import json
import re

import pandas as pd
import pytest

from spirality import publication_matrices as pm
from spirality.angular_contrasts import cached_display_cells, convert_collection
from spirality.paper_products import load_style
from test_angular_contrasts_v88_12 import fixture_collection


def test_angular_annotations_have_two_decimals_and_spaced_colourbar(tmp_path, monkeypatch):
    paper, fid, grid = fixture_collection(tmp_path)
    convert_collection(paper)
    cells = cached_display_cells(paper)
    original = cells.copy(deep=True)
    saved = []
    original_save = pm.save_publication_figure
    def inspect(fig, output, **kwargs):
        axes = fig.axes[:9]
        primary = [t.get_text() for ax in axes for t in ax.texts
                   if re.match(r'^[+-]\d', t.get_text())]
        errors = [t.get_text() for ax in axes for t in ax.texts
                  if t.get_text().startswith(r'$\pm')]
        assert primary and errors
        assert all(re.match(r'^[+-]\d+\.\d{2}(?:\*|\$|$)', t) for t in primary)
        assert all(re.match(r'^\$\\pm\\,\d+\.\d{2}\$$', t) for t in errors)
        label = fig.axes[-1].get_xlabel()
        assert label == r'$\Delta\mathcal{S}^{\,(m)}$ [deg] (shared linear colour scale)'
        saved.append(str(output))
        original_save(fig, output, **kwargs)
    monkeypatch.setattr(pm, 'save_publication_figure', inspect)
    # The legacy setting can remain four: it must not override angular precision.
    style = {'delta_matrix_decimals': 4, 'delta_matrix_versions': ['intensity']}
    pm.plot_delta_s_matrices(pd.read_csv(paper/'grid27_trends.csv'),
                            pd.read_csv(paper/'grid27_statistics.csv'), tmp_path/'plots',
                            style=style, display_cells=cells, unit='deg')
    assert len(saved) == 2
    pd.testing.assert_frame_equal(cells, original)
    for name in ('carron', 'chen'):
        m = json.loads((tmp_path/'plots'/f'grid27_{name}_delta_S_matrices_intensity.metadata.json').read_text())
        assert m['display_decimals'] == 2
        assert m['intensity_vmin'] == -4.5 and m['intensity_vmax'] == 4.5
        assert m['colourbar_label'] == r'$\Delta\mathcal{S}^{\,(m)}$ [deg] (shared linear colour scale)'


def test_dimensionless_default_is_preserved():
    style = load_style()
    assert style['delta_matrix_decimals'] == 4
    assert style['delta_matrix_angular_decimals'] == 2
    assert style['delta_matrix_angular_vmax'] == 4.5
