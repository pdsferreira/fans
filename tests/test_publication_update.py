from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from spirality.publication_matrices import (matrix_cells, plot_delta_s_matrices,
    transition_counts, plot_transition_matrix, L_VALUES, R_VALUES, A_VALUES)
from spirality.publication_refresh import prepare_cached_panel
from spirality.paper_products import load_style
from spirality import extrinsic_spirality as ext
from spirality import pipeline


def grid_tables():
    trends=[];stats=[]
    for cat in ('carron','chen'):
        for index,(l,r,a) in enumerate(itertools.product(L_VALUES,R_VALUES,A_VALUES)):
            name=f'run{index:02d}'
            pars=dict(segment_length_mpc_h=l,node_link_radius_mpc_h=r,arm_opening_deg=a)
            for stratum in ('all','m2','m3'):
                delta=.02+.001*(a-35)+.002*(l-10)
                if index==0:delta=-.00001
                rare=stratum=='m3' and index<3
                trends.append(dict(catalog=cat,run_name=name,stratum=stratum,**pars,
                    delta_S_last_minus_first=delta,delta_S_err=np.nan if rare else .004,
                    delta_S_q025=np.nan if rare else delta-.008,delta_S_q975=np.nan if rare else delta+.008))
                for b in range(4):
                    stats.append(dict(catalog=cat,run_name=name,stratum=stratum,**pars,scope='zbin',zbin_id=b,
                        z_min=[.1,.225,.35,.475][b],z_max=[.225,.35,.475,.6][b],S=.2+delta*b/3,
                        n_fans=4 if rare and b==3 else 20,n_blocks=4 if rare and b==3 else 10,
                        precision_support='false' if rare and b==3 else 'true'))
    return pd.DataFrame(trends),pd.DataFrame(stats)


def test_matrix_geometry_and_signed_values_preserved():
    t,s=grid_tables();out=matrix_cells(t,s)
    assert len(out)==162
    assert out.groupby('catalog').size().to_dict()=={'carron':81,'chen':81}
    assert out.groupby('catalog').plot_support.sum().to_dict()=={'carron':78,'chen':78}
    assert np.isclose(out[(out.run_name=='run00')&(out.stratum=='m2')].delta_S_last_minus_first,-.00001).all()
    limited=out[~out.plot_support]
    assert limited.delta_S_last_minus_first.notna().all()
    assert not limited.interval_excludes_zero.any()
    assert set(limited.n_fans_last)=={4}


def test_mismatched_endpoint_contrast_is_not_silently_plotted():
    t,s=grid_tables();t.loc[0,'delta_S_last_minus_first']+=.01
    with pytest.raises(ValueError,match='does not match'):matrix_cells(t,s)


def test_duplicate_grid_cell_rejected():
    t,s=grid_tables();t=pd.concat([t,t.iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicated'):matrix_cells(t,s)


def test_partial_grid_requires_explicit_request():
    t,s=grid_tables();t=t[t.run_name!='run26'];s=s[s.run_name!='run26']
    with pytest.raises(ValueError,match='27 distinct'):matrix_cells(t,s)
    cells=matrix_cells(t,s,allow_partial=True)
    assert len(cells)==162
    assert (cells.support_reason=='configuration_not_available').sum()==6


def test_matrix_pdf_and_tables_have_no_measurement_mutation(tmp_path):
    t,s=grid_tables();tc=t.copy(deep=True);sc=s.copy(deep=True)
    paths=plot_delta_s_matrices(t,s,tmp_path)
    assert len(paths)==12 and all(p.is_file() for p in paths)
    pd.testing.assert_frame_equal(t,tc);pd.testing.assert_frame_equal(s,sc)
    saved=pd.read_csv(tmp_path/'grid27_chen_delta_S_matrices.used_values.csv')
    assert len(saved)==81 and (saved.n_fans_last==4).sum()==3
    assert json.loads((tmp_path/'grid27_chen_delta_S_matrices.metadata.json').read_text())['n_supported']==78


def sample_transitions():
    return pd.DataFrame([
        dict(scope='full',edge_safe=True,m_candidates=2,m_trace_survivors=2,n_nodes=1000,parent_status='ok'),
        dict(scope='full',edge_safe='true',m_candidates=2,m_trace_survivors=1,n_nodes=3,parent_status='too_few_valid_arms'),
        dict(scope='full',edge_safe=False,m_candidates=2,m_trace_survivors=2,n_nodes=700,parent_status='edge_reject'),
        dict(scope='zbin',edge_safe=True,m_candidates=2,m_trace_survivors=2,n_nodes=500,parent_status='ok'),
        dict(scope='full',edge_safe=True,m_candidates=3,m_trace_survivors=3,n_nodes=1,parent_status='ok')])


def test_transitions_population_and_logarithmic_scale(tmp_path):
    t=sample_transitions();a=transition_counts(t)
    assert a.to_numpy().sum()==1004 and a.loc[2,1]==3 and a.loc[3,3]==1 and a.loc[0,0]==0
    paths=plot_transition_matrix(t,'chen',tmp_path/'transition.pdf',vmax=10000)
    assert len(paths)==3
    m=json.loads((tmp_path/'transition.metadata.json').read_text())
    assert m['vmax']==10000 and m['n_nodes']==1004 and m['counts_are_normalized'] is False
    assert pd.read_csv(tmp_path/'transition.used_values.csv').n_nodes.sum()==1004


def test_transition_counts_never_clipped(tmp_path):
    with pytest.raises(ValueError,match='clip'):plot_transition_matrix(sample_transitions(),'chen',tmp_path/'bad.pdf',vmax=100)


def test_requested_angular_upper_limits_keep_lower_endpoints():
    fig,axs=plt.subplots(1,5)
    for ax in axs:ax.plot([0,1],[-.2,.3]);ax.margins(y=.1)
    lows=[ax.get_ylim()[0] for ax in axs]
    ext._apply_paper_angle_panel_limits(angle=True,ax_pH=axs[0],ax_eH=axs[1],ax_pA=axs[2],ax_eA=axs[3],ax_pS=axs[4])
    assert [ax.get_ylim()[0] for ax in axs]==lows
    assert [ax.get_ylim()[1] for ax in axs[:4]]==[.79,8.8,5.1,5.3]
    assert axs[4].get_ylim()[1]==pytest.approx(15.4)
    plt.close(fig)


def test_limits_expand_rather_than_hide_future_data():
    fig,ax=plt.subplots();ax.plot([0,1],[-.3,10.])
    ext._apply_paper_angle_panel_limits(angle=True,ax_pH=ax,ax_eH=None)
    assert ax.get_ylim()[1]>10
    plt.close(fig)


def test_dimensionless_limits_not_changed():
    fig,ax=plt.subplots();ax.plot([0,1],[-.2,.3]);ax.margins(y=.1);limits=ax.get_ylim()
    ext._apply_paper_angle_panel_limits(angle=False,ax_pH=ax,ax_eH=None)
    assert ax.get_ylim()==limits
    plt.close(fig)


def test_cached_panel_bypasses_all_inference(tmp_path,monkeypatch):
    from test_paper_products import make_run
    import shutil
    root=make_run(tmp_path/'fiducial');folder=tmp_path/'compact';folder.mkdir()
    for cat in ('chen','carron'):
        for key in ('statistics','null_summary'):
            shutil.copy2(root/'referee_multiplicity'/cat/f'multiplicity_{key}.csv',folder/f'{cat}_multiplicity_{key}.csv')
        stats=pd.read_csv(folder/f'{cat}_multiplicity_statistics.csv')
        z=stats[(stats.scope=='zbin')&(stats.stratum=='all')][['zbin_id','z_min','z_max','z_center','n_arms']].copy()
        z['H_ext']=[.001,-.003,.004,-.001];z['H_ext_err']=.01
        z['sign_asym_ext_percent']=[.2,-.3,.1,-.4];z['sign_asym_ext_percent_err']=1.
        z['zbin_axis_mode']='fixed_global'
        z.to_csv(folder/f'{cat}_extrinsic_zbins.csv',index=False)
        best=dict(analysis_schema='88.1',axis_objective='dipole',zbin_axis_mode='fixed_global',ra_deg=20.,dec_deg=30.,
            H_ext=.003,A_dip=.003,H_ext_err=.01,sign_asym_ext_percent=.3,sign_asym_ext_percent_err=1.,
            p_global_lookelsewhere=.5,p_sign_asym_global_lookelsewhere=.6,axis_bootstrap_radius_95_deg=40.)
        pd.DataFrame([best]).to_csv(folder/f'{cat}_extrinsic_bestfit.csv',index=False)
    datasets,prepared,_,meta=prepare_cached_panel(folder,fiducial_run=root)
    assert len(prepared['used_values'])==40
    def forbidden(*args,**kwargs):raise AssertionError('A plot called a scientific inference function')
    for name in ('scan_axis_grid','bootstrap_axes','run_catalog_extrinsic','_projected_fullrange_totals_by_catalog',
                 '_projected_global_signflip_sigmas_by_catalog','_cluster_bootstrap_stat'):
        monkeypatch.setattr(ext,name,forbidden)
    monkeypatch.setattr(pipeline,'_compute_block_signflip_null_zbins',forbidden)
    out=folder/'angular.pdf'
    ext.plot_projected_and_dipole_panel(run_root=folder,datasets=datasets,out=out,weight_mode='n_segments',angle=True,
        panel_extrinsic_sigma_source='global',axis_legend_error_source='bootstrap',prepared_plot_data=prepared)
    m=json.loads(out.with_suffix('.display_metadata.json').read_text())
    assert m['y_limits']['A_dip'][1]==8.8
    assert meta['legend_origin'].startswith('existing_saved_bootstrap')
    assert out.is_file()
