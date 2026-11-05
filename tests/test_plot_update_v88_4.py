from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import pytest

from spirality.paper_products import load_style
from spirality.publication_matrices import (delta_intensity_limit, matrix_cells, plot_delta_s_matrices,
    plot_transition_matrix, transition_upper_bound)
from spirality.publication_redshift import (binned_multiplicity, before_after_counts, prepare_redshift_cache,
    validate_redshift_cache, plot_kde_multiplicity, generate_redshift_products, CACHE, CACHE_META)
from spirality.publication_refresh import refresh_paper_folder
from test_paper_products import make_run
from test_publication_update import grid_tables, sample_transitions


@pytest.fixture
def compact(tmp_path):
    raw=make_run(tmp_path/'fiducial')
    out=tmp_path/'compact';out.mkdir()
    for cat in ('carron','chen'):
        for key in ('statistics','fractions','selection_stages','transitions','metadata'):
            ext='json' if key=='metadata' else 'csv'
            shutil.copy2(raw/'referee_multiplicity'/cat/f'multiplicity_{key}.{ext}',out/f'{cat}_multiplicity_{key}.{ext}')
    return out,raw


def test_intensity_range_ignores_diagnostic_outliers():
    cells=pd.DataFrame({'plot_support':[True,True,False],'delta_S_last_minus_first':[-.015,.034,99.]})
    assert delta_intensity_limit(cells)==pytest.approx(.04)
    with pytest.raises(ValueError,match='clip'):
        delta_intensity_limit(cells,.03)
    assert delta_intensity_limit(cells,.06)==.06


def test_both_matrix_styles_preserve_same_numbers_and_shared_range(tmp_path):
    t,s=grid_tables();paths=plot_delta_s_matrices(t,s,tmp_path)
    assert len(paths)==12
    ranges=[]
    for cat in ('carron','chen'):
        base=f'grid27_{cat}_delta_S_matrices'
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path/f'{base}.used_values.csv'),
                                      pd.read_csv(tmp_path/f'{base}_intensity.used_values.csv'))
        meta=json.loads((tmp_path/f'{base}_intensity.metadata.json').read_text())
        assert meta['intensity_vmin']==-meta['intensity_vmax']
        assert meta['colour_mode']=='intensity'
        ranges.append(meta['intensity_vmax'])
    assert ranges[0]==ranges[1]


def test_sign_only_choice_retains_original_filenames(tmp_path):
    t,s=grid_tables()
    paths=plot_delta_s_matrices(t,s,tmp_path,style={'delta_matrix_versions':['sign']})
    assert len(paths)==6 and not list(tmp_path.glob('*intensity*'))
    with pytest.raises(ValueError,match='unique list'):
        plot_delta_s_matrices(t,s,tmp_path,style={'delta_matrix_versions':['sign','sign']})


def test_discrete_transition_range_and_counts(tmp_path):
    t=sample_transitions();paths=plot_transition_matrix(t,'chen',tmp_path/'t.pdf',vmax=30000,style={'transition_scale':'discrete','transition_palette':'named'})
    meta=json.loads((tmp_path/'t.metadata.json').read_text())
    assert meta['colour_scale']=='discrete'
    assert meta['count_boundaries']==[1,10,30,100,300,600,1000,2000,4000,10000,30000]
    assert meta['n_nodes']==1004
    assert transition_upper_bound(15087,{'transition_scale':'discrete'})==30000
    assert transition_upper_bound(30000,{'transition_scale':'discrete'})>30000


def test_log_transition_option_preserved(tmp_path):
    plot_transition_matrix(sample_transitions(),'chen',tmp_path/'t.pdf',style={'transition_scale':'log','transition_cmap':'cividis'})
    meta=json.loads((tmp_path/'t.metadata.json').read_text())
    assert meta['colour_scale']=='log' and meta['count_boundaries'] is None


def test_conditional_redshift_density_normalizes_per_class(compact):
    folder,raw=compact;b,edges,_=binned_multiplicity(folder)
    for _,g in b[b.multiplicity.isin([2,3])].groupby(['catalog','multiplicity']):
        assert np.sum(g.density_given_multiplicity*(g.z_max-g.z_min))==pytest.approx(1.)
        assert np.allclose(g.fraction_nodes, g.n_fans/g.n_total_fans)


def test_fraction_denominator_includes_four_arm_systems(compact):
    folder,_=compact
    path=folder/'chen_multiplicity_fractions.csv';d=pd.read_csv(path)
    d.loc[(d.scope=='zbin')&(d.zbin_id==0)&(d.multiplicity==4),'n_fans']=1
    d.loc[(d.scope=='zbin')&(d.zbin_id==0),'n_total_fans']=17
    d.loc[(d.scope=='full')&(d.multiplicity==4),'n_fans']=1
    d.loc[d.scope=='full','n_total_fans']=65
    d['fraction_nodes']=d.n_fans/d.n_total_fans;d.to_csv(path,index=False)
    p=folder/'chen_multiplicity_statistics.csv';s=pd.read_csv(p)
    s.loc[(s.stratum=='all')&(s.scope=='zbin')&(s.zbin_id==0),'n_fans']=17
    s.loc[(s.stratum=='all')&(s.scope=='full'),'n_fans']=65;s.to_csv(p,index=False)
    b,_,_=binned_multiplicity(folder)
    r=b[(b.catalog=='chen')&(b.zbin_id==0)&(b.multiplicity==2)].iloc[0]
    assert r.fraction_nodes==pytest.approx(12/17)
    assert r.fraction_nodes!=pytest.approx(12/16)


def test_before_after_are_not_same_class_survival_counts(compact):
    folder,_=compact;b,_,_=binned_multiplicity(folder);d,_=before_after_counts(folder,b)
    # Synthetic accepted two-arm FANS had three candidates: N2 can increase.
    assert (d.query('stage=="before" and multiplicity==2').n_nodes==0).all()
    assert (d.query('stage=="after" and multiplicity==2').n_nodes==12).all()
    assert set(d.multiplicity_definition)=={'candidate_arms','accepted_valid_arms'}


def test_unknown_candidate_multiplicity_is_not_zero(compact):
    folder,_=compact
    p=folder/'chen_multiplicity_selection_stages.csv';d=pd.read_csv(p)
    mask=(d.population=='edge_safe')&(d.stage=='candidate_arms')&(d.scope=='zbin')&(d.zbin_id==0)
    d.loc[mask,'n_population']+=1
    d.loc[mask&(d.multiplicity==-1),'n_nodes']+=1
    d.to_csv(p,index=False)
    b,_,_=binned_multiplicity(folder);r,_=before_after_counts(folder,b)
    assert (r.query('catalog=="chen" and zbin_id==0').n_unknown_candidate_multiplicity==1).all()
    assert r.query('catalog=="chen" and zbin_id==0 and stage=="before" and multiplicity==2').n_nodes.iloc[0]==0


def test_transition_mismatch_rejected(compact):
    folder,_=compact;p=folder/'chen_multiplicity_transitions.csv'
    t=pd.read_csv(p);t.loc[(t.scope=='zbin')&(t.zbin_id==0),'n_nodes']+=1;t.to_csv(p,index=False)
    b,_,_=binned_multiplicity(folder)
    with pytest.raises(ValueError,match='histogram disagrees'):
        before_after_counts(folder,b)


def test_no_individual_redshifts_means_no_invented_kde(compact):
    folder,_=compact;paths,status,_=generate_redshift_products(folder,load_style())
    assert status=='pending_individual_redshifts'
    assert not (folder/'fans_redshift_distribution_by_multiplicity.pdf').exists()
    assert (folder/'fans_redshift_distribution_by_multiplicity_step.pdf').exists()
    assert (folder/'fans_multiplicity_counts_before_after_step.pdf').exists()
    assert len(paths)==12


def test_real_individual_cache_and_kde_generation(compact):
    folder,raw=compact;paths,status,_=generate_redshift_products(folder,load_style(),raw)
    assert status=='validated_fiducial_individual_redshifts'
    nodes=pd.read_csv(folder/CACHE)
    assert len(nodes)==128 and not nodes.duplicated(['catalog','node_id']).any()
    assert (nodes.node_id!=99999).all()
    meta=json.loads((folder/'fans_redshift_distribution_by_multiplicity.metadata.json').read_text())
    assert {m['method'] for m in meta['methods']}=={'gaussian_kde_scott'}
    assert {(m['catalog'],m['multiplicity'],m['n_fans']) for m in meta['methods']}=={
        ('carron',2,48),('carron',3,16),('chen',2,48),('chen',3,16)}
    assert len(paths)==17
    from scipy.stats import gaussian_kde
    curves=pd.read_csv(folder/'fans_redshift_distribution_by_multiplicity.used_values.csv')
    for (cat,m),c in curves.groupby(['catalog','multiplicity']):
        z=nodes[(nodes.catalog==cat)&(nodes.multiplicity==m)].z.to_numpy()
        np.testing.assert_allclose(c.density,gaussian_kde(z)(c.z.to_numpy()),rtol=1e-12,atol=1e-13)


def test_cached_redshifts_work_without_raw_run(compact):
    folder,raw=compact;b,e,_=binned_multiplicity(folder)
    n,paths,_,_=prepare_redshift_cache(folder,b,e,raw)
    shutil.rmtree(raw)
    reread,_,_,status=prepare_redshift_cache(folder,b,e)
    assert status=='cached_individual_redshifts'
    np.testing.assert_allclose(n.z,reread.z)


def test_wrong_node_source_rejected_before_kde(compact):
    folder,raw=compact;b,e,_=binned_multiplicity(folder)
    p=raw/'chen_dr12_native/node_spirality.csv';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='differs from'):
        prepare_redshift_cache(folder,b,e,raw)


def test_modified_redshift_cache_rejected(compact):
    folder,raw=compact;b,e,_=binned_multiplicity(folder)
    prepare_redshift_cache(folder,b,e,raw)
    p=folder/CACHE;p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='Modified redshift cache'):
        prepare_redshift_cache(folder,b,e)


def test_cached_counts_must_match_saved_bins(compact):
    folder,raw=compact;b,e,_=binned_multiplicity(folder)
    n,_,_,_=prepare_redshift_cache(folder,b,e,raw)
    n.loc[0,'z']=.59
    with pytest.raises(ValueError,match='do not reproduce'):
        validate_redshift_cache(n,b,e)


def test_spirality_upper_limit_and_defaults():
    cfg=load_style()
    assert cfg['panel_angular_upper_limits']['S_proj']==15.4
    assert cfg['delta_matrix_versions']==['sign','intensity']
    assert cfg['transition_scale']=='linear'
