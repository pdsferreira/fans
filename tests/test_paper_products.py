from __future__ import annotations
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import pytest
from spirality.paper_products import (validate_revision_run,load_accepted_nodes,collect_grid_tables,
    load_style,visible_limits,plot_redshift_distribution,plot_grid_panel,build_paper_bundle,
    DEFAULT_STYLE,EXPECTED_GRID)
from spirality.multiplicity import analyze_catalog,save_analysis
from spirality.provenance import file_digest
from spirality import pipeline

EDGES=np.array([.1,.225,.35,.475,.6])


def make_run(root,pars=(10.,10.,35.),*,test_seed=7):
    """Actual tiny synthetic h_i -> bootstrap -> tables; not SDSS measurements."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    L,R,A=pars
    sig={'schema':'88.1','source_sha256':'SYNTHETIC_VALIDATION_ONLY','inputs':[],
         'L':L,'R':R,'A':A,'B':8,'null_B':8,'nside':16,'seed':test_seed,
         'edges':','.join(map(str,EDGES))}
    (root/'revision_projected_complete.json').write_text(json.dumps({'signature':sig}))
    (root/'driver_config.json').write_text(json.dumps({'segment_length_mpc_h':L,'node_link_radius_mpc_h':R,
                                                     'arm_opening_deg':A,'n_arm_segments':3}))
    rng=np.random.default_rng(2026)
    for cat in ['carron','chen']:
        rows=[];arms=[];blocks=[];nid=0
        for b in range(4):
            for k in range(16):
                m=2 if k<12 else 3
                z=.5*(EDGES[b]+EDGES[b+1])+.02*np.sin(k)
                rows.append({'node_id':nid,'status':'ok','n_arms_valid':m,'ra':(10+k*23)%360,'dec':10+15*np.cos(k),
                             'z_low':z,'edge_keep':True,'audit_evaluated':True,'audit_m_candidates':m+1,'audit_m_valid':m,
                             'audit_m_raw_groups':m+1,'audit_m_oriented_uncapped':m+1})
                blocks.append(k%8)
                for a in range(m):
                    amplitude=.18+.06*b+.01*a+(.006*(L-10)+.003*(R-10)+.0004*(A-35))
                    val=amplitude*(1 if rng.random()>.5 else -1)
                    arms.append({'node_id':nid,'arm_local_id':a,'mean_t_phi':val,'mean_t_phi_err':.01,
                                 'n_arm_segments_valid':2+(a%2),'n_points_total':6,'mean_arm_density':1.,'mean_t_radial':.95})
                nid+=1
        n=pd.DataFrame(rows);a=pipeline.make_arm_level_table_generic(pd.DataFrame(arms),n)
        rejected=n.iloc[:1].copy();rejected['node_id']=99999;rejected['status']='edge_reject';rejected['n_arms_valid']=np.nan
        n=pd.concat([n,rejected],ignore_index=True)
        d=root/('chen_dr12_native' if cat=='chen' else 'carron_dr17_proxy/carron_proxy_ridges');d.mkdir(parents=True,exist_ok=True)
        n.to_csv(d/'node_spirality.csv',index=False);a.to_csv(d/'arm_level_spirality.csv',index=False)
        r=analyze_catalog(n,a,edges=EDGES,B=8,null_B=8,seed=test_seed,block_ids=np.array(blocks))
        r['metadata'].update(parent_sha256=file_digest(d/'node_spirality.csv'),analysis_arm_sha256=file_digest(d/'arm_level_spirality.csv'))
        out=root/'referee_multiplicity'/cat
        save_analysis(r,n,edges=EDGES,outdir=out)
        (out/'sample_integrity.json').write_text(json.dumps({'passed':True,'n_fans_accepted':64,'n_analysis_arms':len(a),
            'sum_accepted_parent_multiplicities':len(a),'n_raw_arms_rejected_parent':0}))
    return root


@pytest.fixture
def fid(tmp_path):return make_run(tmp_path/'fiducial')


def test_distribution_uses_accepted_nodes_once(fid,tmp_path):
    info=validate_revision_run(fid);data={}
    for cat in ['carron','chen']:
        data[cat],_=load_accepted_nodes(fid,cat,info['edges'])
        assert len(data[cat]['z'])==64
    files=plot_redshift_distribution(data,EDGES,tmp_path/'plots',DEFAULT_STYLE)
    assert all(p.exists() for p in files)
    counts=pd.read_csv(tmp_path/'plots/fans_redshift_bin_counts.csv')
    assert counts.groupby('catalog').n_fans.sum().to_dict()=={'carron':64,'chen':64}
    hist=pd.read_csv(tmp_path/'plots/fans_redshift_distribution_histogram.csv')
    for _,g in hist.groupby('catalog'):assert np.sum(g.density*(g.z_max-g.z_min))==pytest.approx(1.)


def test_tampered_input_rejected(fid):
    p=fid/'chen_dr12_native/node_spirality.csv';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='Stale'):validate_revision_run(fid)


def test_noninteger_or_missing_multiplicity_never_accepted(fid):
    p=fid/'chen_dr12_native/node_spirality.csv';d=pd.read_csv(p);d.loc[0,'n_arms_valid']=1;d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='accepted parent'):load_accepted_nodes(fid,'chen',EDGES)


def test_no_legacy_or_partial_grid_fallback(fid,tmp_path):
    grid=tmp_path/'grid';grid.mkdir();shutil.copytree(fid,grid/'anything')
    with pytest.raises(ValueError,match='27 distinct'):collect_grid_tables(grid)
    data,_,meta=collect_grid_tables(grid,allow_partial=True)
    assert meta['n_runs']==1 and len(meta['missing'])==26
    assert len(data['statistics'])>0
    shutil.copytree(fid,grid/'duplicate')
    with pytest.raises(ValueError,match='Duplicate grid'):collect_grid_tables(grid,allow_partial=True)


def test_undefined_errors_stay_nan_and_axes_do_not_clip(tmp_path):
    assert visible_limits([-.5,1.2],[np.nan,.3],[-.1,.1],'expand')[1]>1.5
    with pytest.raises(ValueError,match='clip'):visible_limits([-.5,1.2],[0,.3],[-.1,.1],'strict')


def test_kde_fallback_is_recorded(tmp_path):
    d={c:{'z':np.repeat(.3,3)} for c in ['chen','carron']}
    plot_redshift_distribution(d,EDGES,tmp_path,DEFAULT_STYLE)
    m=json.loads((tmp_path/'fans_redshift_distribution_metadata.json').read_text())
    assert m['chen']['method']=='histogram_density'


def test_flat_bundle_repeated_updates_preserve_author_panel(fid,tmp_path):
    out=tmp_path/'paper_final_plots_and_tables'
    manifest=build_paper_bundle(out,fiducial_run=fid,footprints=False)
    assert all((out/e['file']).exists() for e in manifest['files'])
    assert (out/'fans_redshift_distribution.pdf').exists()
    assert (out/'projected_sign_asymmetry_joint_bootstrap.pdf').exists()
    panel=out/'MY_FANS_PANEL.txt';panel.write_text('manually supplied')
    build_paper_bundle(out,footprints=False)
    assert panel.read_text()=='manually supplied'
    managed=out/'fans_redshift_distribution.pdf';managed.write_bytes(b'manually modified')
    with pytest.raises(FileExistsError,match='Manually modified'):build_paper_bundle(out,footprints=False)


def test_require_extrinsic_does_not_run_any_inference(fid,tmp_path):
    with pytest.raises(FileNotFoundError,match='extrinsic'):build_paper_bundle(tmp_path/'out',fiducial_run=fid,footprints=False,require_extrinsic=True)


def test_full_grid_panel_three_strata_and_exact_source_values(tmp_path):
    grid=tmp_path/'grid';fid=make_run(tmp_path/'fid')
    for i,pars in enumerate(sorted(EXPECTED_GRID)):make_run(grid/f'configuration_{i}',pars)
    tables,edges,meta=collect_grid_tables(grid)
    f=[]
    for cat in ['carron','chen']:
        d=pd.read_csv(fid/'referee_multiplicity'/cat/'multiplicity_statistics.csv');d['catalog']=cat;d['run_name']='fid'
        for col,val in zip(['segment_length_mpc_h','node_link_radius_mpc_h','arm_opening_deg'],[10,10,35]):d[col]=val
        f.append(d)
    f=pd.concat(f,ignore_index=True)
    for stratum in ['all','m2','m3']:
        p=tmp_path/f'panel_{stratum}.pdf'
        plot_grid_panel(tables['statistics'],f,edges,p,stratum=stratum,style=DEFAULT_STYLE,n_runs=meta['n_runs'])
        used=pd.read_csv(p.with_suffix('.used_values.csv'))
        h=used[used.plot_role=='highlighted_fiducial']
        expected=f[(f.scope=='zbin')&(f.stratum==stratum)]
        np.testing.assert_allclose(h.S_proj_err,expected.S_err,equal_nan=True)
        assert len(h)==8 and used.run_name.nunique()==28
    # Central geometry mismatch must not be hidden by a substituted curve.
    f.loc[(f.stratum=='all')&(f.scope=='zbin'),'S'] += .1
    with pytest.raises(ValueError,match='disagree'):plot_grid_panel(tables['statistics'],f,edges,tmp_path/'bad.pdf')


@pytest.mark.skipif(importlib.util.find_spec('healpy') is None or importlib.util.find_spec('astropy') is None,
                    reason='actual author-footprint rendering requires healpy and astropy')
def test_real_author_footprint_maps(fid,tmp_path):
    from spirality.paper_footprints import build_footprints
    info=validate_revision_run(fid)
    d={cat:load_accepted_nodes(fid,cat,info['edges'])[0] for cat in ['carron','chen']}
    files,meta=build_footprints(d,tmp_path,nside=8,font_size=14)
    assert len(files)==3 and all(p.stat().st_size>1000 for p in files)
    assert sum(r['n_fans'] for r in meta['counts'])==128
