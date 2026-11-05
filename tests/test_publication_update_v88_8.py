"""View-only regression coverage for fonts, hatches, saturation and frames."""
import json
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from spirality.paper_products import load_style,plot_grid_panel
from spirality.publication_panels import sample_settings,draw_transition_axis,draw_distribution_axis
from spirality.publication_matrices import transition_display_spec,hatch_zero_transition_cells
from spirality.publication_style import save_publication_figure,publication_frame_scope
from test_publication_update_v88_7 import transition_fixture
from test_publication_update import grid_tables


def test_requested_colour_ticks_saturate_only_the_colours():
    for cat,peak,maximum,ticks in [('carron',8510,8500,[0,2125,4250,6375,8500]),('chen',5057,5000,[0,1250,2500,3750,5000])]:
        original=transition_fixture(peak);copy=original.copy(deep=True)
        counts,cmap,norm,bounds,values,meta=transition_display_spec(original,cat)
        np.testing.assert_array_equal(values,ticks)
        assert bounds is None and norm.vmin==0 and norm.vmax==maximum
        assert meta['n_saturated_cells']==1 and meta['maximum_observed_count']==peak
        assert counts.loc[2,2]==peak
        assert norm(peak)==1 and np.allclose(cmap(norm(peak)),cmap(norm(maximum)))
        pd.testing.assert_frame_equal(original,copy)


def test_unrequested_transition_clipping_still_rejected():
    with pytest.raises(ValueError,match='permission'):
        transition_display_spec(transition_fixture(8510),'carron',style={'transition_allow_colour_saturation':False})
    with pytest.raises(ValueError,match='permission'):
        transition_display_spec(transition_fixture(8510),'carron',vmax=8000)


def test_zeros_hatched_and_not_missing_or_positive_cells():
    fig,ax=plt.subplots();a=np.array([[0,1,4],[2,0,0]])
    patches=hatch_zero_transition_cells(ax,a)
    assert len(patches)==3 and all(p.get_hatch()=='///' for p in patches)
    assert all(p.get_gid()=='zero-count-cell' for p in patches)
    positions={(p.get_y()+.5,p.get_x()+.5) for p in patches}
    assert positions=={tuple(i) for i in np.argwhere(a==0)}
    plt.close(fig)


def test_sample_headings_and_other_text_have_different_increments():
    cfg=sample_settings(load_style())
    assert cfg['paper_panel_label_size']==pytest.approx(18*1.5*1.20)
    assert cfg['paper_panel_tick_size']==pytest.approx(16*1.5*1.10)
    assert cfg['paper_fraction_annotation_size']==pytest.approx(11.5*1.5*1.10)
    fig,ax=plt.subplots();im,rows,bounds=draw_transition_axis(ax,transition_fixture(),'carron',cfg)
    assert ax.title.get_fontsize()==pytest.approx(16*1.5*1.20)
    assert all(t.get_fontsize()==pytest.approx(12.8*1.5*1.10) for t in ax.texts)
    assert cfg['paper_sample_legend_gap_inches']>.84
    plt.close(fig)


def test_distribution_strokes_thicker_with_bin_edges_only():
    edges=np.array([.1,.225,.35,.475,.6]);cfg=sample_settings(load_style())
    curves=pd.DataFrame([dict(catalog=c,stratum=p,z=z,density=2) for c in ['chen','carron'] for p in ['all','m2','m3'] for z in [.1,.3,.6]])
    ann=pd.DataFrame([dict(catalog=c,z_min=edges[i],z_max=edges[i+1],text='2A:75%/3A:25%') for c in ['chen','carron'] for i in range(4)])
    fig,ax=plt.subplots(figsize=(13.7,8));draw_distribution_axis(ax,curves,ann,edges,cfg)
    assert [line.get_linewidth() for line in ax.lines[:6]]==[3.,2.5,2.5,3.,2.5,2.5]
    assert sum(len(line.get_xdata())==2 and np.ptp(line.get_xdata())==0 for line in ax.lines)==3
    assert not any(x.get_visible() for x in ax.get_xgridlines()+ax.get_ygridlines())
    plt.close(fig)


def test_every_frame_and_colourbar_uses_fiducial_width(tmp_path,monkeypatch):
    fig,ax=plt.subplots();im=ax.imshow(np.arange(4).reshape(2,2));cb=fig.colorbar(im,ax=ax)
    for a in fig.axes:
        for spine in a.spines.values():spine.set_linewidth(3)
    captured=[]
    def inspect(self,*a,**k):
        assert all(sp.get_linewidth()==1.1 for ax in self.axes for sp in ax.spines.values())
        assert cb.outline.get_linewidth()==1.1
        captured.append(True)
    monkeypatch.setattr(Figure,'savefig',inspect)
    save_publication_figure(fig,tmp_path/'frame.pdf');assert captured
    plt.close(fig)


def test_scoped_legacy_save_patch_is_restored_on_exception():
    original=Figure.savefig
    with pytest.raises(RuntimeError):
        with publication_frame_scope():
            assert Figure.savefig is not original
            raise RuntimeError('intentional')
    assert Figure.savefig is original


def test_exact_requested_all_sample_angular_ranges_and_units(tmp_path,monkeypatch):
    tr,st=grid_tables()
    st['H']=.001;st['A_sign']=.002
    for k in ('H','S','A_sign'):st[k+'_err']=.001
    st['n_arms']=2*st.n_fans;st['z_center']=(st.z_min+st.z_max)/2
    before=st.copy(deep=True);captured=[]
    def inspect(fig,*a,**k):
        for row,limits in enumerate([(-.7,.52),(-2.2,4.3),(9.1,17.3)]):
            for ax in fig.axes[2*row:2*row+2]:
                np.testing.assert_allclose(ax.get_ylim(),limits)
                assert all(x.get_linewidth()==1.1 for x in ax.spines.values())
        captured.append(True)
    monkeypatch.setattr(Figure,'savefig',inspect)
    p=tmp_path/'grid.pdf';plot_grid_panel(st,None,[.1,.225,.35,.475,.6],p,style=load_style(),angle=True)
    assert captured;pd.testing.assert_frame_equal(st,before)
    meta=json.loads(p.with_suffix('.metadata.json').read_text())
    assert 'percent' in meta['units']


def test_requested_grid_limits_do_not_silently_clip_new_data(tmp_path):
    tr,st=grid_tables();st['H']=.2;st['A_sign']=0.
    for k in ('H','S','A_sign'):st[k+'_err']=.001
    st['n_arms']=2*st.n_fans;st['z_center']=(st.z_min+st.z_max)/2
    cfg=load_style();cfg['grid_angular_y_limit_policy']='strict'
    with pytest.raises(ValueError,match='clip data'):
        plot_grid_panel(st,None,[.1,.225,.35,.475,.6],tmp_path/'bad.pdf',style=cfg,angle=True)
    plt.close('all')
