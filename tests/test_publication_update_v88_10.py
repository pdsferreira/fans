"""Final layout controls; no estimator or statistical selection changes."""
import json
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from spirality.paper_products import load_style
from spirality.publication_panels import sample_panel_only_settings, sample_settings
from spirality.publication_matrices import plot_delta_s_matrices
from test_publication_update import grid_tables


def test_fraction_reduction_is_exactly_five_percent_and_local():
    style=load_style()
    old=dict(style,paper_sample_panel_fraction_increment=1.1)
    now=sample_panel_only_settings(style);before=sample_panel_only_settings(old)
    assert now['paper_fraction_annotation_size']==pytest.approx(.95*before['paper_fraction_annotation_size'])
    assert now['paper_fraction_annotation_size']==pytest.approx(19.828875)
    for key in ['paper_panel_label_size','paper_panel_tick_size','_paper_frame_linewidth','_sample_cell_size']:
        assert now[key]==before[key]
    assert sample_settings(style)['paper_fraction_annotation_size']==sample_settings(old)['paper_fraction_annotation_size']


def test_new_grid_ranges_do_not_change_dimensionless_or_asymmetry_defaults():
    c=load_style()
    assert c['grid_angular_y_limits']=={'H_proj':[-.7,.52],'A_sign':[-2.2,4.3],'S_proj':[9.1,17.3]}
    assert c['grid_y_limits']['S_proj']==[.18,.33]
    assert c['grid_angular_y_limit_policy']=='strict'


def test_only_intensity_matrices_share_x_numbers_and_reduce_gaps(tmp_path,monkeypatch):
    import spirality.publication_matrices as m
    seen=[]
    def capture(fig,output,**kwargs):
        fig.canvas.draw()
        axes=fig.axes[:9]
        labels=[[x.get_text() for x in a.get_xticklabels() if x.get_visible()] for a in axes]
        seen.append((str(output),labels,fig.subplotpars.hspace,fig.subplotpars.wspace))
    monkeypatch.setattr(m,'save_publication_figure',capture)
    t,s=grid_tables();beforet=t.copy(deep=True);befores=s.copy(deep=True)
    paths=plot_delta_s_matrices(t,s,tmp_path,style=load_style())
    assert len(paths)==12 and len(seen)==4
    for name,labels,h,w in seen:
        if name.rsplit('/',1)[-1].endswith('_intensity.pdf'):
            assert all(not x for x in labels[:6])
            assert all(x==['8','10','12'] for x in labels[6:])
            assert h==.08 and w==.07
        else:
            assert all(x==['8','10','12'] for x in labels)
            assert h==.32 and w==.13
    for cat in ['carron','chen']:
        a=pd.read_csv(tmp_path/f'grid27_{cat}_delta_S_matrices.used_values.csv')
        b=pd.read_csv(tmp_path/f'grid27_{cat}_delta_S_matrices_intensity.used_values.csv')
        pd.testing.assert_frame_equal(a,b)
        meta=json.loads((tmp_path/f'grid27_{cat}_delta_S_matrices_intensity.metadata.json').read_text())
        assert meta['x_ticklabels_bottom_row_only']
        assert meta['subplot_hspace']==.08 and meta['figure_size_inches']==[13.,10.7]
    pd.testing.assert_frame_equal(t,beforet);pd.testing.assert_frame_equal(s,befores)
