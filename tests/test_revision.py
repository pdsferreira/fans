from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree
from spirality import chen_node_spirality as chen
from spirality import sconce_node_spirality as shared
from spirality import pipeline as pipe
from spirality import extrinsic_spirality as ext
from spirality.selection import attach_parent_metadata, audit_catalog, redshift_bin_ids
from spirality.multiplicity import analyze_catalog, save_analysis, plot_analysis, construction_tables

ROOT=Path(__file__).resolve().parents[1]
EDGES=np.array([.1,.225,.35,.475,.6])


def geometry(module, *, one_arm_fails=False, edge_reject=False):
    r1=np.array([2.,4.,6.,12.,14.,16.,22.,24.,26.])
    r2=np.array([2.,4.,6.]) if one_arm_fails else r1
    x=np.r_[r1,-r2];y=np.zeros(len(x)); E=np.where(x>0,1.,-1.)*np.cos(.2);N=np.where(x>0,1.,-1.)*np.sin(.2)
    hull=np.array([[1,0,-10 if edge_reject else -100],[-1,0,-100],[0,1,-100],[0,-1,-100]],float)
    kw=dict(z_key=.2,z_value=.2,chi_mpc_h=1000.,ra0_deg=0.,dec0_deg=0.,points=pd.DataFrame({"ra":x,"dec":y}),
            x_mpc_h=x,y_mpc_h=y,e_hat=E,n_hat=N,density=np.ones(len(x)),uncertainty=np.ones(len(x)),tree=cKDTree(np.c_[x,y]),hull_equations=hull)
    sd=module.SliceData(**kw)
    cfg=module.Config(node_link_radius_mpc_h=10.,segment_length_mpc_h=10.,arm_opening_deg=35.,edge_margin_mpc_h=70,
                      bootstrap_B=0,signflip_B=0)
    node=pd.Series(dict(node_id=1,ra=0.,dec=0.,z_low=.2))
    return node,sd,cfg


@pytest.mark.parametrize("module",[chen,shared])
def test_original_failure_path_now_excluded(module):
    node,sd,cfg=geometry(module,one_arm_fails=True)
    result,arms,segs=module.process_one_node(node,sd,cfg)
    assert result['status']=='too_few_valid_arms' and result['n_arms_valid']==1
    assert result['audit_m_candidates']==2 and result['audit_m_valid']==1
    assert len(arms)==1
    n=pd.DataFrame([result]);a=pd.DataFrame(arms)
    out=pipe.make_arm_level_table_generic(a,n)
    assert len(out)==0
    if module is shared:assert len(shared.make_arm_level_table(a,n))==0


@pytest.mark.parametrize("module",[chen,shared])
def test_audit_does_not_change_accepted_geometry(module):
    node,sd,cfg=geometry(module)
    old,a0,s0=module._process_one_node_science(node,sd,cfg)
    new,a1,s1=module.process_one_node(node,sd,cfg)
    assert old['status']==new['status']=='ok'
    pd.testing.assert_frame_equal(pd.DataFrame(a0),pd.DataFrame(a1))
    pd.testing.assert_frame_equal(pd.DataFrame(s0),pd.DataFrame(s1))
    for key in old:
        if isinstance(old[key],float) and np.isnan(old[key]):assert np.isnan(new[key])
        else:assert new[key]==old[key]
    assert new['audit_m_candidates']==new['audit_m_valid']==2
    nodes=pd.DataFrame([new]);raw=pd.DataFrame(a1);arms=pipe.make_arm_level_table_generic(raw,nodes)
    assert (arms.parent_n_arms_valid==2).all()
    audit=audit_catalog(nodes,raw,arms,pd.DataFrame(s1));assert audit['passed']


@pytest.mark.parametrize("module",[chen,shared])
def test_early_edge_rejection_has_separate_diagnostic_multiplicity(module):
    node,sd,cfg=geometry(module,edge_reject=True)
    n,a,s=module.process_one_node(node,sd,cfg)
    assert n['status']=='edge_reject' and not a and not s
    assert np.isnan(n['n_arms_valid']) and n['audit_m_valid']==2
    assert n['audit_evaluated']


def test_arm_stop_reason_and_first_gap():
    node,sd,cfg=geometry(chen,one_arm_fails=True)
    n,_,_=chen.process_one_node(node,sd,cfg)
    cand=json.loads(n['audit_candidates_json'])
    rejected=[c for c in cand if not c['passes_arm_cut']][0]
    assert rejected['stop_reason']=='too_few_points' and rejected['stop_segment']==1
    assert rejected['n_segments_valid']==1


def toy_catalog(*,evolution=0.,missing_three=False):
    nodes=[];arms=[];groups=[];nid=0
    for b,(n2,n3) in enumerate(zip([80,60,40,20],[20,40,60,80])):
        z=.5*(EDGES[b]+EDGES[b+1])
        for m,num in [(2,n2),(3,0 if missing_three else n3)]:
            for k in range(num):
                group=k%8;nd={"node_id":nid,"status":"ok","n_arms_valid":m,"ra":10+group*20,"dec":10.,"z_low":z,
                     "audit_evaluated":True,"audit_trace_evaluated":True,"audit_m_raw_groups":m+1,"audit_m_candidates":m,
                     "audit_m_oriented_uncapped":m,"audit_m_valid":m,"edge_keep":True}
                nodes.append(nd);groups.append(group)
                for j in range(m):
                    val=(.1 if m==2 else .3)+evolution*z
                    sign=1 if k%2==0 else -1
                    arms.append({"node_id":nid,"arm_local_id":j,"mean_t_phi":sign*val,"mean_t_phi_err":.01,
                                 "mean_t_radial":.98,"n_arm_segments_valid":2+(j%2),"n_points_total":6,"mean_arm_density":1.})
                nid+=1
    n=pd.DataFrame(nodes);a=pipe.make_arm_level_table_generic(pd.DataFrame(arms),n)
    return n,a,np.array(groups)


def test_complete_parent_invariants():
    n,a,groups=toy_catalog()
    assert len(attach_parent_metadata(a,n))==len(a)
    with pytest.raises(ValueError,match="Incomplete"):
        attach_parent_metadata(a.iloc[1:],n)
    with pytest.raises(ValueError,match="unique"):
        attach_parent_metadata(pd.concat([a,a.iloc[:1]]),n)
    with pytest.raises(ValueError,match="Orphan"):
        b=a.copy();b.loc[0,'node_id']=-999;attach_parent_metadata(b,n)
    with pytest.raises(ValueError,match="unique"):
        attach_parent_metadata(a,pd.concat([n,n.iloc[:1]]))


def test_no_promotion_of_all_rejected_rows():
    d=pd.DataFrame(dict(status=['edge_reject','edge_reject'],node_id=[1,2],z_low=[.2,.3],S_tangential=[.1,.2]))
    r=pipe.build_rigorous_zbin_summary(d,edges=EDGES,weight_mode='uniform',B=0,seed=1,jackknife_nside=4,
                                      error_mode='bootstrap',cluster_col='node_id')
    assert r.n_nodes_ok.sum()==0


def test_halfopen_bins_exact_boundaries():
    z=np.array([.099,.1,.225,.35,.475,.6,.601,np.nan])
    np.testing.assert_array_equal(redshift_bin_ids(z,EDGES),[-1,0,1,2,3,3,-1,-1])


def test_composition_mixture_only_control():
    n,a,g=toy_catalog()
    r=analyze_catalog(n,a,edges=EDGES,B=100,null_B=40,block_ids=g)
    c=r['composition']
    assert c.supported.all()
    assert c.S_all.iloc[-1]>c.S_all.iloc[0]
    np.testing.assert_allclose(c.S_fixed_composition,c.S_fixed_composition.iloc[0],atol=1e-14)
    np.testing.assert_allclose(c.S_all,c.S_reconstructed,atol=1e-14)
    fix=r['trends'].set_index('stratum').loc['fixed_composition']
    assert abs(fix.delta_S_last_minus_first)<1e-14
    assert fix.delta_S_err<1e-13
    f=r['fractions'].query('scope=="zbin" and multiplicity==3')
    assert (f.fraction_weight>f.fraction_nodes).all()


def test_fixed_class_evolution_survives_composition_control():
    n,a,g=toy_catalog(evolution=.1)
    r=analyze_catalog(n,a,edges=EDGES,B=80,null_B=0,block_ids=g)
    t=r['trends'].set_index('stratum')
    np.testing.assert_allclose(t.loc[['m2','m3','fixed_composition'],'slope_S_per_z_descriptive'],.1,atol=1e-12)


def test_sign_flip_preserves_spirality_but_reverses_handedness():
    n,a,g=toy_catalog(evolution=.1)
    r1=analyze_catalog(n,a,edges=EDGES,B=30,null_B=30,seed=5,block_ids=g)
    a.S_tangential *= -1
    r2=analyze_catalog(n,a,edges=EDGES,B=30,null_B=30,seed=5,block_ids=g)
    np.testing.assert_allclose(r1['bootstrap']['S'],r2['bootstrap']['S'],equal_nan=True)
    np.testing.assert_allclose(r1['bootstrap']['H'],-r2['bootstrap']['H'],equal_nan=True)
    assert 'S' not in r1['null_summary'].statistic.unique()


def test_zero_bootstrap_and_unsupported_reference():
    n,a,g=toy_catalog(missing_three=True)
    r=analyze_catalog(n,a,edges=EDGES,B=0,null_B=0,block_ids=g)
    assert not r['composition'].supported.any()
    assert r['composition'].S_fixed_composition.isna().all()
    assert r['statistics'].S_err.isna().all()
    assert r['bootstrap']['S'].shape[0]==0


def test_true_arm_weighting_not_node_average():
    n,a,g=toy_catalog()
    r=analyze_catalog(n,a,edges=EDGES,B=0,null_B=0,block_ids=g)
    globalrow=r['statistics'].query('stratum=="all" and scope=="full"').iloc[0]
    expected=np.average(a.S_tangential.abs(),weights=a.n_arm_segments_total)
    assert globalrow.S==pytest.approx(expected)


def test_unknown_is_not_zero_in_stage_tables():
    n,a,g=toy_catalog();unknown=n.iloc[:1].copy();unknown['node_id']=99999;unknown['status']='no_matched_ridge_slice'
    unknown['audit_evaluated']=False;unknown['n_arms_valid']=np.nan
    nodes=pd.concat([n,unknown],ignore_index=True)
    d,_,_=construction_tables(nodes,EDGES)
    s=d.query('population=="all_processed" and scope=="full" and stage=="candidate_arms"')
    assert s.loc[s.multiplicity==-1,'n_nodes'].iloc[0]==1
    assert s.loc[s.multiplicity==0,'n_nodes'].iloc[0]==0


def directions():
    ra=np.array([0,30,90,160,220,300.]);dec=np.array([15,35,55,70,25,40.])
    xyz=ext.unit_xyz_from_radec(ra,dec)
    return pd.DataFrame({'pix':np.arange(len(ra)),'ra_deg':ra,'dec_deg':dec,'x':xyz[:,0],'y':xyz[:,1],'z':xyz[:,2]})


def extrinsic_toy():
    rng=np.random.default_rng(19);n=80
    ra=rng.uniform(0,180,n);dec=rng.uniform(-40,70,n);xyz=ext.unit_xyz_from_radec(ra,dec)
    h=.04+.10*(xyz@np.array([0.,0.,1.]))+rng.normal(0,.08,n)
    d=pd.DataFrame({'ra':ra,'dec':dec,'S_tangential':h,'n_arm_segments_total':rng.integers(2,4,n),
                    'z_eff':np.tile([.15,.28,.4,.55],n//4),'node_id':np.arange(n)//2})
    return d,xyz,np.arange(n)//4


def test_wls_intercept_vs_linear_algebra():
    d,xyz,g=extrinsic_toy();w=ext.arm_weights(d,'n_segments');h=d.S_tangential.to_numpy();mu=xyz[:,2]
    A,H0,*_=ext.weighted_dipole_fit_with_intercept(mu,h,w)
    b=np.linalg.lstsq(np.c_[np.ones(len(mu)),mu]*np.sqrt(w[:,None]),h*np.sqrt(w),rcond=None)[0]
    np.testing.assert_allclose([H0,A],b,atol=1e-13)
    A2,H02,*_=ext.weighted_dipole_fit_with_intercept(mu,h+.5,w)
    assert A2==pytest.approx(A);assert H02==pytest.approx(H0+.5)


def test_raw_sign_equals_paper_not_residual():
    mu=np.array([-.8,-.5,.1,.7]);xyz=np.c_[np.sqrt(1-mu*mu),np.zeros(len(mu)),mu]
    h=.4+.1*mu;w=np.ones(len(h));axis={'ra_deg':0.,'dec_deg':90.}
    st=ext._stats_for_fixed_axis_subset(xyz,h,w,axis)
    assert st['sign_asym_ext']==pytest.approx(0.)
    assert st['sign_asym_ext_residual']==pytest.approx(1.)
    assert np.isnan(st['A_dip_err']) and np.isnan(st['sign_asym_ext_percent_err'])


def test_axis_sign_convention_not_amplitude_oriented():
    ds=directions().iloc[:1].copy();ds['A_dip']=-.2;ds['A_dip_abs']=.2;ds['H0_dip']=.1;ds['mu_bar']=.3;ds['dipole_design_den']=2
    ds['sign_asym_ext']=-.4;ds['sign_asym_ext_percent']=-40.;ds['sign_asym_ext_residual']=-.5
    b=ext.best_axis_from_scan(ds)
    assert b['dec_deg']>0 and b['A_dip']==pytest.approx(-.2) and b['A_dip_abs']==pytest.approx(.2)
    assert b['sign_asym_ext']==pytest.approx(-.4)


def test_axis_antipode_transforms_both_estimators():
    d,xyz,g=extrinsic_toy();w=ext.arm_weights(d,'n_segments');h=d.S_tangential.to_numpy()
    s1=ext._stats_for_fixed_axis_subset(xyz,h,w,{'ra_deg':10.,'dec_deg':40.})
    s2=ext._stats_for_fixed_axis_subset(xyz,h,w,{'ra_deg':190.,'dec_deg':-40.})
    for key in ['A_dip','sign_asym_ext','sign_asym_ext_residual']:assert s1[key]==pytest.approx(-s2[key])
    assert s1['H0_dip']==pytest.approx(s2['H0_dip'])


def test_bootstrap_counts_equal_explicit_repeated_arms():
    d,xyz,g=extrinsic_toy();w=ext.arm_weights(d,'n_segments');h=d.S_tangential.to_numpy();counts=np.tile([0,1,2,3],20)
    idx=np.repeat(np.arange(len(d)),counts)
    axis={'ra_deg':120.,'dec_deg':50.}
    a=ext._stats_for_fixed_axis_subset(xyz,h,w*counts,axis,count_weights=counts)
    b=ext._stats_for_fixed_axis_subset(xyz[idx],h[idx],w[idx],axis)
    for key in ['A_dip','H0_dip','sign_asym_ext','sign_asym_ext_residual']:assert a[key]==pytest.approx(b[key])
    x=ext.scan_axis_grid(arm_xyz=xyz,sphi=h,weights=w*counts,count_weights=counts,directions=directions(),progress=False).sort_values('pix')
    y=ext.scan_axis_grid(arm_xyz=xyz[idx],sphi=h[idx],weights=w[idx],directions=directions(),progress=False).sort_values('pix')
    np.testing.assert_allclose(x[['A_dip','sign_asym_ext','sign_asym_ext_residual']],y[['A_dip','sign_asym_ext','sign_asym_ext_residual']],atol=1e-12)


def test_extrinsic_bootstrap_errors_are_actual_sample_scatter(tmp_path):
    d,xyz,g=extrinsic_toy();dirs=directions();w=ext.arm_weights(d,'n_segments')
    scan=ext.scan_axis_grid(arm_xyz=xyz,sphi=d.S_tangential.to_numpy(),weights=w,directions=dirs,progress=False)
    best=ext.best_axis_from_scan(scan)
    stats=ext.bin_extrinsic_stats(d,directions=dirs,edges=EDGES,weight_mode='n_segments',fixed_axis=best)
    boot=ext.bootstrap_axes(d=d,directions=dirs,nside_block=16,B=40,seed=5,weight_mode='n_segments',chunk_size=4,
                            fixed_axis=best,edges=EDGES,block_ids=g)
    best,stats=ext.apply_spatial_bootstrap_errors(boot,best,stats,outdir=tmp_path,key='test')
    assert stats.loc[0,'A_dip_err']==pytest.approx(boot.A_dip_zbin_0.std(ddof=1))
    assert stats.loc[0,'sign_asym_ext_percent_err']==pytest.approx(boot.sign_asym_ext_percent_zbin_0.std(ddof=1))
    assert 'conditional_A_dip_zbin_0' in boot
    assert best['axis_bootstrap_radius_95_deg']>=0
    c=np.load(tmp_path/'test_extrinsic_bootstrap_covariance.npy')
    assert np.linalg.eigvalsh(c).min()>-1e-9


def test_null_repeats_selection_and_calibrates_residual_control():
    d,xyz,g=extrinsic_toy();dirs=directions();w=ext.arm_weights(d,'n_segments');h=d.S_tangential.to_numpy()
    scan=ext.scan_axis_grid(arm_xyz=xyz,sphi=h,weights=w,directions=dirs,progress=False)
    best=ext.best_axis_from_scan(scan)
    null,summ=ext.extrinsic_signflip_lookelsewhere_null(arm_xyz=xyz,sphi=h,directions=dirs,weights=w,best=best,
               B=20,seed=6,chunk_size=4,group_codes=g,z_eff=d.z_eff.to_numpy(),zbin_edges=EDGES)
    assert len(null)==20
    expected=(1+(null.max_abs_A_dip>=abs(best['A_dip'])).sum())/21
    assert summ.p_global_lookelsewhere.iloc[0]==pytest.approx(expected)
    assert summ.p_residual_axial_sign_at_dipole_axis.notna().all()
    assert (null.max_abs_A_dip>=0).all()


def test_save_outputs_and_new_figures(tmp_path):
    n,a,g=toy_catalog(evolution=.05)
    result=analyze_catalog(n,a,edges=EDGES,B=20,null_B=20,block_ids=g)
    save_analysis(result,n,edges=EDGES,outdir=tmp_path)
    plot_analysis(tmp_path,'chen',fmt='png')
    assert len(list(tmp_path.glob('*.png')))>=7
    assert (tmp_path/'multiplicity_bootstrap_samples.npz').exists()
    assert json.loads((tmp_path/'multiplicity_metadata.json').read_text())['target_supported']


def test_grid_launcher_never_calls_extrinsic():
    cmd=[sys.executable,str(ROOT/'scripts/run_referee_revision.py'),'grid','--outroot','/unused/grid','--dry-run','--skip-fits-export']
    proc=subprocess.run(cmd,text=True,capture_output=True,check=True)
    assert proc.stdout.count('scripts/run_spirality_pipeline.py')==27
    assert proc.stdout.count('scripts/analyze_fans_multiplicity.py')==27
    assert 'run_extrinsic_spirality.py' not in proc.stdout
    assert '--pipelines carron chen' in proc.stdout


def test_launcher_refuses_grid_extrinsic():
    proc=subprocess.run([sys.executable,str(ROOT/'scripts/run_referee_revision.py'),'grid','--outroot','/unused','--stage','all','--dry-run'],capture_output=True,text=True)
    assert proc.returncode!=0 and 'PROJECTED ONLY' in proc.stderr


@pytest.mark.skipif(importlib.util.find_spec('healpy') is None,reason='healpy not installed in this environment')
def test_real_healpix_partition():
    import healpy as hp
    from spirality.resampling import healpix_blocks
    ra=np.array([0,20,200,359.]);dec=np.array([-60,0,20,80.])
    np.testing.assert_array_equal(healpix_blocks(ra,dec,16),hp.ang2pix(16,ra,dec,lonlat=True))


def _load_export_module():
    spec=importlib.util.spec_from_file_location('export_fans',ROOT/'scripts/export_fans_fits.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_fits_parent_quality_preparation_without_io(monkeypatch):
    """Dataframe join/filter test; no astropy/FITS serialization is simulated."""
    mod=_load_export_module()
    node,sd,cfg=geometry(chen); n,a,s=chen.process_one_node(node,sd,cfg)
    rejected=dict(n,node_id=2,status='too_few_valid_arms',n_arms_valid=1)
    nodes=pd.DataFrame([n,rejected]);seg=pd.DataFrame(s)
    rejected_seg=seg[seg.arm_local_id==0].copy();rejected_seg.node_id=2
    raw=pd.concat([seg,rejected_seg],ignore_index=True)
    parent=mod._prepare_nodes('chen',nodes,accepted_only=False)
    monkeypatch.setattr(mod,'_radec_from_node_offsets',lambda ra,dec,z,x,y:(np.asarray(ra),np.asarray(dec)))
    q=mod._prepare_segments('chen',parent,raw,accepted_only=True)
    assert len(q)==len(seg) and q.node_id.eq(1).all() and q.is_quality_ok.all()
    with pytest.raises(ValueError,match='Orphan'):
        mod._prepare_segments('chen',parent[parent.node_id_out==1],raw,accepted_only=True)
    with pytest.raises(ValueError,match='status'):
        mod._prepare_nodes('chen',nodes.drop(columns='status'),accepted_only=False)


def test_actual_proxy_dispatch_preserves_unassessed_nodes():
    from spirality import carron_proxy_spirality as proxy
    node,sd,cfg=geometry(shared)
    missing=node.copy();missing['node_id']=99;missing['z_low']=.55
    n,a,s=proxy.process_nodes_with_slices(shared,pd.DataFrame([node,missing]),{shared.zkey(.2):sd},cfg,
                                         n_jobs=1,timing_interval=0,catalog_label='Carron')
    unk=n.set_index('node_id').loc[99]
    assert unk.status=='no_matched_slice' and np.isnan(unk.n_arms_valid)
    assert not unk.audit_evaluated
    assert len(a)==2 and a.node_id.eq(1).all()


def test_legacy_plot_adapter_reuses_exact_joint_draws(tmp_path):
    from spirality.provenance import file_digest
    from spirality.revision_plot_data import find_revision_output,fullrange_markers,binned_null_table
    n,a,g=toy_catalog(evolution=.05)
    src=tmp_path/'chen_dr12_native';src.mkdir();a.to_csv(src/'arm_level_spirality.csv',index=False)
    result=analyze_catalog(n,a,edges=EDGES,B=30,null_B=40,block_ids=g)
    result['metadata']['analysis_arm_sha256']=file_digest(src/'arm_level_spirality.csv')
    out=tmp_path/'referee_multiplicity'/'chen';save_analysis(result,n,edges=EDGES,outdir=out)
    assert find_revision_output(src/'arm_level_spirality.csv','chen')==out
    totals=fullrange_markers(out,angle=False)
    full=result['statistics'].query('stratum=="all" and zbin_id==-1').iloc[0]
    assert totals['S']['value']==pytest.approx(full.S)
    assert totals['S']['err']==pytest.approx(full.S_err)
    assert totals['A']['err']==pytest.approx(100*full.A_sign_err)
    bins=result['statistics'].query('stratum=="all" and zbin_id>=0')
    tab=binned_null_table(out,bins,statistic='sign_asymmetry',catalog='chen')
    official=result['null_summary'].query('stratum=="all" and zbin_id>=0 and statistic=="A_sign"')
    np.testing.assert_allclose(tab.null_q84,100*official.q1sigma_high)
    with (src/'arm_level_spirality.csv').open('a') as f:f.write('\n')
    assert find_revision_output(src/'arm_level_spirality.csv','chen') is None


@pytest.mark.skipif(importlib.util.find_spec('astropy') is None,reason='astropy not installed: actual FITS serialization requires local validation')
def test_actual_fits_quality_roundtrip(tmp_path):
    from astropy.io import fits
    mod=_load_export_module();node,sd,cfg=geometry(chen)
    n,a,s=chen.process_one_node(node,sd,cfg)
    rejected=dict(n,node_id=2,status='too_few_valid_arms',n_arms_valid=1)
    cat=tmp_path/'chen_dr12_native';cat.mkdir()
    pd.DataFrame([n,rejected]).to_csv(cat/'node_spirality.csv',index=False)
    seg=pd.DataFrame(s);rej=seg[seg.arm_local_id==0].copy();rej.node_id=2
    pd.concat([seg,rej]).to_csv(cat/'arm_segments.csv',index=False)
    path=mod.export_catalog(tmp_path,'chen',tmp_path/'fans_fits',accepted_only=True,overwrite=True)
    with fits.open(path,checksum=True) as f:
        assert len(f['NODES'].data)==1
        assert len(f['SEGMENTS'].data)==len(seg)
        assert np.all(f['SEGMENTS'].data['QUALITY']==1)
        assert set(f['SEGMENTS'].data['FANS_ID'])<=set(f['NODES'].data['FANS_ID'])
        assert f[0].header['SCHEMA']=='88.1'


def test_extrinsic_orchestration_with_explicit_test_partition(tmp_path,monkeypatch):
    """Exercise real inference/CSV/cache flow; supply explicit axes/groups, not fake HEALPix."""
    d,xyz,g=extrinsic_toy();d['arm_local_id']=np.arange(len(d))%2
    d['z_low']=np.repeat([.15,.28,.4,.55],len(d)//4)
    # Coordinates/redshift of the parent are shared by its two arms.
    for col in ['ra','dec','z_low']:
        d[col]=d.groupby('node_id')[col].transform('first')
    n=d.groupby('node_id',as_index=False).first()[['node_id','ra','dec','z_low']]
    n['status']='ok';n['n_arms_valid']=2
    n.to_csv(tmp_path/'node_spirality.csv',index=False)
    path=tmp_path/'arm_level_spirality.csv';d.to_csv(path,index=False)
    out=tmp_path/'extrinsic';out.mkdir()
    monkeypatch.setattr(ext,'healpix_direction_grid',lambda nside:directions())
    monkeypatch.setattr(ext,'make_extrinsic_null_groups',lambda **kwargs:(g,len(np.unique(g)),'explicit numerical-test groups'))
    original_boot=ext.bootstrap_axes
    monkeypatch.setattr(ext,'bootstrap_axes',lambda **kwargs:original_boot(**kwargs,block_ids=g))
    monkeypatch.setattr(ext,'plot_healpix_h_map',lambda *args,**kwargs:None)
    monkeypatch.setattr(ext,'plot_bootstrap_axis_map',lambda *args,**kwargs:pd.DataFrame())
    args=dict(key='chen',label='Synthetic numerical test',arm_path=path,outdir=out,zbin_edges=EDGES,nside_scan=32,
              nside_bootstrap=16,bootstrap_B=20,seed=3,weight_mode='n_segments',chunk_size=4,no_bootstrap=False,
              axis_display_mode='none',extrinsic_null_B=20,extrinsic_null_modes=('block',),extrinsic_null_block_nside=16)
    z,b,boot,scan=ext.run_catalog_extrinsic(**args)
    assert len(z)==4 and len(boot)==20
    assert z.A_dip_err.notna().all()
    assert (out/'chen_extrinsic_completed_signature.json').exists()
    nul=pd.read_csv(out/'chen_extrinsic_lookelsewhere_signflip_block_null.csv')
    assert nul.max_abs_sign_asym_ext_percent.isna().all()  # no extra sign-axis optimization
    assert b['lookelsewhere_sign_percent_q1sigma']==pytest.approx(np.percentile(np.abs(nul.selected_axis_sign_asym_ext_percent),ext.SIGNED_ABS_1SIGMA_PCT))
    zz,bb,_,_=ext.run_catalog_extrinsic(**args,reuse_existing_extrinsic=True)
    assert bb['A_dip']==pytest.approx(b['A_dip'])
    with path.open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='stale'):
        ext.run_catalog_extrinsic(**args,reuse_existing_extrinsic=True)


def test_no_saturation_of_out_of_domain_dipole_angles():
    assert np.isnan(ext._scalar_asin_deg(1.2))
    np.testing.assert_allclose(ext._asin_deg_clip([0,.5,2]),[0,30,np.nan],equal_nan=True)
