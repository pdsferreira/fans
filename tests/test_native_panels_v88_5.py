from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from spirality.paper_products import load_style
from spirality.publication_panels import (angular_interval,spirality_display_rows,_groups,
    fraction_annotations,distribution_curves,plot_sample_panel,plot_spirality_standalone,
    equal_axes_grid,axes_geometry,plot_results_panel)
from spirality.publication_redshift import binned_multiplicity,prepare_redshift_cache
from spirality.publication_refresh import prepare_cached_panel
from test_paper_products import make_run


@pytest.fixture
def compact(tmp_path):
    raw=make_run(tmp_path/'fiducial')
    out=tmp_path/'compact';out.mkdir()
    for cat in ('carron','chen'):
        for key in ('statistics','fractions','selection_stages','transitions','metadata','composition','null_summary'):
            ext='json' if key=='metadata' else 'csv'
            shutil.copy2(raw/'referee_multiplicity'/cat/f'multiplicity_{key}.{ext}',out/f'{cat}_multiplicity_{key}.{ext}')
    return out,raw


def test_angular_endpoints_are_transformed_not_symmetrised():
    v,lo,hi=angular_interval(.8,.7,.9)
    np.testing.assert_allclose([v,lo,hi],np.degrees(np.arcsin([.8,.7,.9])))
    assert not np.isclose(v-lo,hi-v)
    v,lo,hi=angular_interval(.9,.7,1.1)
    assert np.isnan(hi) and not np.isnan(v)
    with pytest.raises(ValueError,match='Reversed'):
        angular_interval(.2,.3,.1)


def test_conditioned_intervals_default_retain_bootstrap_standard_error(compact):
    out,_=compact;r,_=spirality_display_rows(out,'chen',load_style())
    np.testing.assert_allclose(r.S_low,r.S-r.S_err,equal_nan=True)
    np.testing.assert_allclose(r.theta_low_deg,np.degrees(np.arcsin(r.S_low)),equal_nan=True)
    np.testing.assert_allclose(r.theta_high_deg,np.degrees(np.arcsin(r.S_high)),equal_nan=True)


def test_saved_percentile_interval_is_an_explicit_option(compact):
    out,_=compact;cfg=load_style();cfg['paper_spirality_interval']='percentile68'
    r,_=spirality_display_rows(out,'chen',cfg)
    np.testing.assert_allclose(r.S_low,r.S_q16,equal_nan=True)
    np.testing.assert_allclose(r.theta_high_deg,np.degrees(np.arcsin(r.S_q84)),equal_nan=True)


def test_low_support_is_not_deleted_but_omitted_from_graph(compact):
    out,_=compact
    path=out/'chen_multiplicity_statistics.csv';d=pd.read_csv(path)
    d.loc[d.scope.eq('zbin') & d.stratum.eq('m3'),'precision_support']=True
    mask=d.scope.eq('zbin')&d.stratum.eq('m3')&d.zbin_id.eq(3)
    d.loc[mask,'precision_support']=False;d.loc[mask,'n_fans']=4
    d.to_csv(path,index=False);before=path.read_bytes()
    r,_=spirality_display_rows(out,'chen',load_style())
    q=r[r.stratum.eq('m3')]
    assert len(q)==4 and q.plot_support.sum()==3
    assert [g.zbin_id.tolist() for g in _groups(q)]==[[0,1,2]]
    plot_spirality_standalone(out,load_style())
    used=pd.read_csv(out/'chen_S_fixed_multiplicity.used_values.csv')
    assert len(used[used.stratum.eq('m3')])==4
    assert not bool(used[used.stratum.eq('m3') & used.zbin_id.eq(3)].plot_support.iloc[0])
    assert path.read_bytes()==before


def test_interpolation_never_bridges_a_missing_middle_bin():
    d=pd.DataFrame({'zbin_id':[0,1,2,3],'plot_support':[True,False,True,True]})
    assert [g.zbin_id.tolist() for g in _groups(d)]==[[0],[2,3]]


def test_fractions_use_full_bin_counts_including_four_arms():
    rows=[]
    for cat in ('carron','chen'):
        for b in range(4):
            for m,n in [(2,60),(3,30),(4,10)]:
                rows.append(dict(catalog=cat,zbin_id=b,z_min=.1+b*.125,z_max=.225+b*.125,
                    multiplicity=m,n_fans=n,n_total_fans=100))
    a=fraction_annotations(pd.DataFrame(rows))
    assert len(a)==8
    assert set(a.text)=={'2A:60.0%/3A:30.0%'}
    assert (a.n_other==10).all()


def test_six_kdes_come_from_their_own_individual_redshifts(compact):
    folder,raw=compact;b,e,_=binned_multiplicity(folder)
    nodes,_,_,_=prepare_redshift_cache(folder,b,e,raw)
    curves=distribution_curves(nodes,b,e,load_style())
    assert len(curves)==6*700
    assert curves.groupby(['catalog','stratum']).ngroups==6
    for (cat,pop),d in curves.groupby(['catalog','stratum']):
        n=nodes[nodes.catalog.eq(cat)]
        if pop!='all':n=n[n.multiplicity.eq(int(pop[1:]))]
        np.testing.assert_allclose(d.density,gaussian_kde(n.z.to_numpy())(d.z.to_numpy()),rtol=1e-12,atol=1e-13)


def test_sample_panel_does_not_invent_kdes_when_only_list_or_counts_exist(compact):
    folder,_=compact;paths,_,status=plot_sample_panel(folder,load_style())
    assert not paths and status=='pending_individual_redshifts'
    assert not (folder/'fans_sample_multiplicity_panel.pdf').exists()


def test_native_sample_panel_all_three_axes_equal_not_resized_by_colourbar(compact):
    folder,raw=compact;paths,_,status=plot_sample_panel(folder,load_style(),fiducial_run=raw)
    assert len(paths)==6 and status=='validated_fiducial_individual_redshifts'
    m=json.loads((folder/'fans_sample_multiplicity_panel.metadata.json').read_text())
    assert len(m['data_axes_geometry'])==3
    assert m['pdf_or_image_compositing'] is False
    q=m['data_axes_geometry'];np.testing.assert_allclose([x['height_over_width'] for x in q],7/9.5,atol=1e-9)
    np.testing.assert_allclose([x['width_px'] for x in q],q[0]['width_px'],atol=1e-7)
    used=pd.read_csv(folder/'fans_sample_multiplicity_panel.used_values.csv')
    assert (used.record_type=='bin_annotation').sum()==8
    assert (used.record_type=='kde_curve').sum()==4200


def test_equal_aspect_in_every_panel_including_six_axes():
    fig,axes=equal_axes_grid(3,2,width=14,aspect=.55,bottom=1.4)
    boxes=axes_geometry(fig,axes,aspect=.55)
    assert len(boxes)==6
    plt.close(fig)


def _prepare_extrinsic(folder,raw):
    for cat in ('chen','carron'):
        stats=pd.read_csv(folder/f'{cat}_multiplicity_statistics.csv')
        z=stats[stats.scope.eq('zbin')&stats.stratum.eq('all')][['zbin_id','z_min','z_max','z_center','n_arms']].copy()
        z['H_ext']=[.001,-.003,.004,-.001];z['H_ext_err']=.01
        z['sign_asym_ext_percent']=[.2,-.3,.1,-.4];z['sign_asym_ext_percent_err']=1.
        z['zbin_axis_mode']='fixed_global';z.to_csv(folder/f'{cat}_extrinsic_zbins.csv',index=False)
        best=dict(analysis_schema='88.1',axis_objective='dipole',zbin_axis_mode='fixed_global',ra_deg=20.,dec_deg=30.,
            H_ext=.003,A_dip=.003,H_ext_err=.01,sign_asym_ext_percent=.3,sign_asym_ext_percent_err=1.,
            p_global_lookelsewhere=.5,p_sign_asym_global_lookelsewhere=.6,axis_bootstrap_radius_95_deg=40.)
        pd.DataFrame([best]).to_csv(folder/f'{cat}_extrinsic_bestfit.csv',index=False)
    ds,p,_,_=prepare_cached_panel(folder,fiducial_run=raw)
    return ds,p


def test_native_results_panel_uses_no_scientific_inference_or_pdf_composition(compact,monkeypatch):
    from spirality import extrinsic_spirality as ext,pipeline
    folder,raw=compact;ds,p=_prepare_extrinsic(folder,raw)
    def forbidden(*a,**kw):raise AssertionError('Unexpected scientific or PDF assembly call')
    for name in ('scan_axis_grid','bootstrap_axes','run_catalog_extrinsic','_projected_fullrange_totals_by_catalog',
                 '_projected_global_signflip_sigmas_by_catalog','_cluster_bootstrap_stat','plot_projected_and_dipole_panel'):
        monkeypatch.setattr(ext,name,forbidden)
    monkeypatch.setattr(pipeline,'_compute_block_signflip_null_zbins',forbidden)
    paths,_=plot_results_panel(folder,load_style(),datasets=ds,prepared=p)
    assert len(paths)==4  # PDF, values, metadata and external caption.
    meta=json.loads((folder/'fiducial_results_multiplicity_panel_angle.metadata.json').read_text())
    assert len(meta['data_axes_geometry'])==6 and meta['pdf_or_image_compositing'] is False
    assert meta['parity_y_limits']['A_dip'][1]==8.8
    used=pd.read_csv(folder/'fiducial_results_multiplicity_panel_angle.used_values.csv')
    assert len(used[used.record_type.eq('parity_panel_source')])==32
    # Endpoint/source values remain tied to the original raw summaries.
    np.testing.assert_allclose(used.loc[used.record_type.eq('spirality_display'),'theta_deg'],
        np.degrees(np.arcsin(used.loc[used.record_type.eq('spirality_display'),'S'])),equal_nan=True)
