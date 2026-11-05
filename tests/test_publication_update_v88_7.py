"""Presentation-only regression tests for the v88.7 requests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from spirality.publication_panels import (sample_settings,draw_distribution_axis,
    equal_axes_grid,axes_geometry)
from spirality.publication_matrices import (transition_counts,transition_display_spec,
    plot_transition_matrix,plot_delta_s_matrices)
from spirality.paper_products import (load_style,grid_display_values,plot_grid_panel)
from test_publication_update import grid_tables


def transition_fixture(peak=8000,zeros=30000):
    return pd.DataFrame([dict(scope='full',edge_safe=True,m_candidates=m,
        m_trace_survivors=n,n_nodes=v) for m,n,v in [(0,0,zeros),(1,0,39),(1,1,160),
        (2,0,40),(2,1,200),(2,2,peak),(3,2,5),(3,3,2)]])


def test_view_filter_keeps_input_and_zero_survivor_column():
    table=transition_fixture();original=table.copy(deep=True)
    counts,_,_,_,_,meta=transition_display_spec(table,'carron')
    pd.testing.assert_frame_equal(table,original)
    assert transition_counts(table).loc[0,0]==30000
    assert counts.index.tolist()==[1,2,3] and 0 in counts.columns
    assert counts.loc[1,0]==39
    assert meta['n_nodes_not_drawn']==30000 and meta['vmax']==8500
    assert meta['n_nodes_all_evaluated']==meta['n_nodes']+30000


def test_independent_ranges_and_catalogue_colours():
    a,ca,na,_,_,ma=transition_display_spec(transition_fixture(8510),'carron')
    b,cb,nb,_,_,mb=transition_display_spec(transition_fixture(5057),'chen')
    assert ma['vmax']==8500 and mb['vmax']==5000
    np.testing.assert_allclose(ca(na(8510))[:3],to_rgb('royalblue'))
    np.testing.assert_allclose(cb(nb(5057))[:3],to_rgb('orangered'))
    np.testing.assert_allclose(ca(na(0))[:3],[1,1,1])
    np.testing.assert_allclose(ca.get_bad()[:3],to_rgb('#ededed'))
    assert not np.allclose(ca.get_bad()[:3],ca(na(1))[:3])


def test_zero_candidate_peak_never_controls_colours():
    for v in (100,10**8):
        _,_,_,_,_,meta=transition_display_spec(transition_fixture(1234,v),'chen',style={'transition_vmax_by_catalogue':{}})
        assert meta['vmax']==1234


def test_all_zero_and_legacy_optional_scales():
    t=transition_fixture();t['n_nodes']=0
    a,c,n,_,_,m=transition_display_spec(t,'chen',style={'transition_vmax_by_catalogue':{}})
    assert m['vmax']==2 and a.to_numpy().sum()==0
    for scale in ['log','discrete']:
        _,_,_,_,_,m=transition_display_spec(transition_fixture(),'chen',style={'transition_scale':scale,'transition_palette':'named'})
        assert m['colour_scale']==scale


def test_sample_fonts_scale_not_results_style():
    s=load_style();cfg=sample_settings(s)
    assert cfg['paper_panel_label_size']==pytest.approx(1.5*1.2*s['paper_panel_label_size'])
    assert cfg['paper_panel_tick_size']==pytest.approx(1.5*1.1*s['paper_panel_tick_size'])
    assert cfg['paper_panel_box_aspect']==pytest.approx(7/9.5)
    assert s['paper_panel_box_aspect']==.55


def test_distribution_only_has_bin_boundaries_no_grey_grid():
    cfg=sample_settings(load_style());edges=np.array([.1,.225,.35,.475,.6])
    curves=pd.DataFrame([dict(catalog=c,stratum=s,z=z,density=2.) for c in ['chen','carron'] for s in ['all','m2','m3'] for z in [.1,.3,.6]])
    ann=pd.DataFrame([dict(catalog=c,z_min=edges[i],z_max=edges[i+1],text='2A:75%/3A:25%') for c in ['chen','carron'] for i in range(4)])
    fig,ax=plt.subplots();draw_distribution_axis(ax,curves,ann,edges,cfg)
    assert not any(x.get_visible() for x in ax.get_xgridlines()+ax.get_ygridlines())
    v=[line.get_xdata()[0] for line in ax.lines if len(line.get_xdata())==2 and np.ptp(line.get_xdata())==0]
    np.testing.assert_allclose(v,edges[1:-1]);plt.close(fig)


def test_angular_grid_transforms_each_endpoint_and_asymmetry():
    v,lo,hi=grid_display_values([.8],[.1],'S_proj',True)
    np.testing.assert_allclose([lo[0],v[0],hi[0]],np.degrees(np.arcsin([.7,.8,.9])))
    assert not np.isclose(v-lo,hi-v)
    v,lo,hi=grid_display_values([-.2],[.03],'H_proj',True)
    assert v[0]<0
    v,lo,hi=grid_display_values([.12],[.02],'A_sign',True)
    np.testing.assert_allclose([lo[0],v[0],hi[0]],[10,12,14])
    v,lo,hi=grid_display_values([.99],[.05],'S_proj',True)
    assert np.isnan(hi[0]) and np.isfinite(lo[0])


def test_grid_versions_preserve_canonical_rows_and_disable_grid(tmp_path,monkeypatch):
    tr,st=grid_tables()
    # Minimal scientific columns used by the actual display function.
    for key in ('H','A_sign'):st[key]=.002
    for key in ('H_err','S_err','A_sign_err'):st[key]=.005
    st['n_arms']=2*st.n_fans
    st['z_center']=(st.z_min+st.z_max)/2
    before=st.copy(deep=True)
    saved=[]
    def capture(self,*a,**k):
        for ax in self.axes[:6]:
            assert not any(x.get_visible() for x in ax.get_xgridlines()+ax.get_ygridlines())
            assert sum(x.get_gid()=='redshift-bin-edge' for x in ax.lines)==3
        saved.append(1)
    monkeypatch.setattr(Figure,'savefig',capture)
    for subset in ['all','m2','m3']:
        for angle in (False,True):
            p=tmp_path/f'{subset}_{angle}.pdf'
            plot_grid_panel(st,None,[.1,.225,.35,.475,.6],p,stratum=subset,angle=angle,style=load_style())
            used=pd.read_csv(p.with_suffix('.used_values.csv'))
            q=used[used.plot_role!='highlighted_central_grid']
            ref=before[(before.scope=='zbin') & (before.stratum==subset)]
            np.testing.assert_allclose(q.S_proj,ref.S,equal_nan=True)
            if angle:np.testing.assert_allclose(q.S_proj_display,np.degrees(np.arcsin(ref.S)),equal_nan=True)
            m=json.loads(p.with_suffix('.metadata.json').read_text());assert m['angular_display']==angle
    assert len(saved)==6;pd.testing.assert_frame_equal(st,before)


def test_intensity_labels_move_without_changing_cells(tmp_path,monkeypatch):
    tr,st=grid_tables();xs=[]
    def capture(self,*a,**k):
        vals=[t.get_position()[0] for t in self.texts if t.get_text() in ['All accepted FANS','Two-arm FANS','Three-arm FANS']]
        assert len(vals)==3;xs.extend(vals)
    monkeypatch.setattr(Figure,'savefig',capture)
    plot_delta_s_matrices(tr,st,tmp_path,style={'delta_matrix_versions':['intensity']})
    assert xs==[.058]*6
    for cat in ['chen','carron']:
        d=pd.read_csv(tmp_path/f'grid27_{cat}_delta_S_matrices_intensity.used_values.csv')
        expected=tr[tr.catalog.eq(cat)&tr.stratum.isin(['all','m2','m3'])]
        a=d.merge(expected,on=['catalog','run_name','stratum'],suffixes=('_figure','_input'),validate='one_to_one')
        np.testing.assert_allclose(a.delta_S_last_minus_first_figure,a.delta_S_last_minus_first_input,equal_nan=True)
