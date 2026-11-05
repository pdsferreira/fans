from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd
import pytest

from spirality.angular_contrasts import (angular_difference, summarize_saved_endpoints,
    display_cells, convert_collection, cached_display_cells, write_referee_table,
    GRID_TABLE, FID_TABLE, REF_TABLE, CACHE_META, SOURCE_NAMES)
from spirality.resampling import bootstrap_summary
from spirality.publication_matrices import matrix_cells, PARAMETERS, L_VALUES, R_VALUES, A_VALUES
from spirality.provenance import file_digest


def example(B=200, seed=18, delta=.02, rare=False, empty=False):
    rng=np.random.default_rng(seed)
    records=[];draws=[]
    for k,stratum in enumerate(('all','m2','m3')):
        base=.2+.01*k
        common=rng.normal(0,.008,B)
        for b in range(4):
            central=base+delta*b/3
            arr=central+common+rng.normal(0,.004,B)
            if stratum=='m3' and b==3 and empty:
                arr[:B//5]=np.nan
            r=dict(stratum=stratum,scope='zbin',zbin_id=b,z_min=[.1,.225,.35,.475][b],z_max=[.225,.35,.475,.6][b],
                   S=central,n_fans=4 if rare and stratum=='m3' and b==3 else 30,n_blocks=4 if rare and stratum=='m3' and b==3 else 15,
                   precision_support=not(rare and stratum=='m3' and b==3))
            r.update({'S_'+n:v for n,v in bootstrap_summary(arr).items()})
            records.append(r);draws.append(arr)
    stats=pd.DataFrame(records);samples=np.asarray(draws).T;tr=[]
    for stratum in ('all','m2','m3'):
        sub=stats[stats.stratum==stratum]
        i,j=sub.index[0],sub.index[-1]
        tr.append(dict(stratum=stratum,delta_S_last_minus_first=stats.loc[j,'S']-stats.loc[i,'S'],
            **{'delta_S_'+n:v for n,v in bootstrap_summary(samples[:,j]-samples[:,i]).items()}))
    return stats,pd.DataFrame(tr),stats[['stratum','scope','zbin_id']].reset_index(names='column_id'),samples,{'B':B,'schema':'88.1'}


def test_exact_arcsin_endpoints_not_arcsin_of_delta():
    assert angular_difference(.7,.8)==pytest.approx(np.degrees(np.arcsin(.8)-np.arcsin(.7)))
    assert abs(float(angular_difference(.7,.8))-float(np.degrees(np.arcsin(.1))))>2


def test_exact_paired_standard_error_and_percentiles():
    st,tr,idx,s,md=example()
    out=summarize_saved_endpoints(st,tr,idx,s,md)
    d=angular_difference(s[:,0],s[:,3])
    assert out.iloc[0].delta_S_deg_err==pytest.approx(np.std(d,ddof=1))
    assert out.iloc[0].delta_S_deg_q025==pytest.approx(np.percentile(d,2.5))
    assert out.iloc[0].delta_S_deg_q975==pytest.approx(np.percentile(d,97.5))
    assert out.iloc[0].bootstrap_B==200
    assert out.iloc[0].delta_S_deg_err < np.hypot(np.std(np.degrees(np.arcsin(s[:,0]))),np.std(np.degrees(np.arcsin(s[:,3]))))*.7


def test_no_random_sampling_is_invoked(monkeypatch):
    args=example()
    def fail(*a,**kw):raise AssertionError('No new draws permitted')
    monkeypatch.setattr(np.random,'default_rng',fail)
    summarize_saved_endpoints(*args)


def test_reject_shuffled_column_index():
    st,tr,idx,s,md=example();idx=idx.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError,match='column IDs'):summarize_saved_endpoints(st,tr,idx,s,md)


def test_reject_unpaired_endpoint_archive_even_if_marginals_match():
    st,tr,idx,s,md=example();s[:,3]=s[::-1,3]
    with pytest.raises(ValueError,match='paired raw delta'):summarize_saved_endpoints(st,tr,idx,s,md)


def test_missing_draws_retain_same_pairs_and_valid_fraction_gate():
    out=summarize_saved_endpoints(*example(empty=True))
    assert np.isnan(out.iloc[2].delta_S_deg_err)
    assert not out.iloc[2].plot_support
    assert out.iloc[2].delta_S_deg_bootstrap_valid_fraction==pytest.approx(.8)


def test_no_support_upgrade_for_rare_cells():
    out=summarize_saved_endpoints(*example(rare=True))
    assert not out.iloc[2].plot_support
    assert np.isfinite(out.iloc[2].delta_S_deg)


@pytest.mark.parametrize('first,last',[(1.01,.4),(.2,-.01),(np.inf,.3)])
def test_out_of_domain_rejected_instead_of_clipped(first,last):
    with pytest.raises(ValueError,match='outside'):angular_difference(first,last)


def test_actual_B_not_hardcoded():
    out=summarize_saved_endpoints(*example(B=1000))
    assert out.bootstrap_B.eq(1000).all()


def test_wrong_metadata_B_rejected():
    st,tr,idx,s,md=example();md['B']=1000
    with pytest.raises(ValueError,match='size differs'):summarize_saved_endpoints(st,tr,idx,s,md)


def fixture_collection(root):
    paper=root/'paper';fid=root/'fiducial';grid=root/'grid';paper.mkdir()
    grid_stats=[];grid_trends=[]
    fid_data={}
    for cat_i,cat in enumerate(('carron','chen')):
        for i,(l,r,a) in enumerate(itertools.product(L_VALUES,R_VALUES,A_VALUES)):
            run=f'run{i:02d}'
            args=example(seed=cat_i*100+i+1,delta=.015+.001*i,rare=(cat=='chen' and i%5==0))
            st,tr,idx,draw,meta=args
            pars=dict(zip(PARAMETERS,(l,r,a)))
            grid_stats.append(st.assign(catalog=cat,run_name=run,**pars));grid_trends.append(tr.assign(catalog=cat,run_name=run,**pars))
            folder=grid/run/'referee_multiplicity'/cat;folder.mkdir(parents=True)
            st.to_csv(folder/SOURCE_NAMES[0],index=False);tr.to_csv(folder/SOURCE_NAMES[1],index=False)
            idx.to_csv(folder/SOURCE_NAMES[2],index=False);(folder/SOURCE_NAMES[3]).write_text(json.dumps(meta))
            np.savez_compressed(folder/SOURCE_NAMES[4],S=draw)
            if (l,r,a)==(10.,10.,35.):fid_data[cat]=args
        folder=fid/'referee_multiplicity'/cat;folder.mkdir(parents=True)
        st,tr,idx,draw,meta=fid_data[cat]
        st.to_csv(folder/SOURCE_NAMES[0],index=False);tr.to_csv(folder/SOURCE_NAMES[1],index=False)
        idx.to_csv(folder/SOURCE_NAMES[2],index=False);(folder/SOURCE_NAMES[3]).write_text(json.dumps(meta))
        np.savez_compressed(folder/SOURCE_NAMES[4],S=draw)
        st.to_csv(paper/f'{cat}_multiplicity_statistics.csv',index=False);tr.to_csv(paper/f'{cat}_multiplicity_trends.csv',index=False)
    pd.concat(grid_stats).to_csv(paper/'grid27_statistics.csv',index=False)
    pd.concat(grid_trends).to_csv(paper/'grid27_trends.csv',index=False)
    (paper/'manual_plot.pdf').write_bytes(b'Manual asset, do not touch')
    (paper/'other.pdf').write_bytes(b'Unrelated managed PDF, do not touch')
    entries=[dict(file=p.name,sha256=file_digest(p),role='science',size_bytes=p.stat().st_size) for p in paper.iterdir() if p.name!='manual_plot.pdf']
    (paper/'manifest.json').write_text(json.dumps({'science_schema':'88.1','files':entries,'sources':dict(fiducial_run=str(fid),grid_root=str(grid))}))
    return paper,fid,grid


def test_preflight_missing_arrays_does_not_modify_any_file(tmp_path):
    paper,fid,grid=fixture_collection(tmp_path)
    (fid/'referee_multiplicity/carron/multiplicity_bootstrap_samples.npz').unlink()
    hashes={p.name:file_digest(p) for p in paper.iterdir()}
    with pytest.raises(FileNotFoundError,match='paired bootstrap'):convert_collection(paper)
    assert hashes=={p.name:file_digest(p) for p in paper.iterdir()}


def test_check_only_does_not_write(tmp_path):
    paper,fid,grid=fixture_collection(tmp_path);hashes={p.name:file_digest(p) for p in paper.iterdir()}
    result=convert_collection(paper,check_only=True)
    assert result['grid_rows']==162 and result['fiducial_rows']==6
    assert hashes=={p.name:file_digest(p) for p in paper.iterdir()}


def test_full_conversion_and_cache_preserve_other_products(tmp_path):
    paper,fid,grid=fixture_collection(tmp_path)
    old={p.name:file_digest(p) for p in paper.iterdir() if p.name!='manifest.json'}
    result=convert_collection(paper)
    assert all(file_digest(paper/k)==v for k,v in old.items())
    assert all((paper/n).is_file() for n in (FID_TABLE,GRID_TABLE,REF_TABLE,CACHE_META))
    assert r'$\Delta\mathcal{S}^{(m)}$ [deg]' in (paper/REF_TABLE).read_text()
    assert 'arcsin' in (paper/REF_TABLE).read_text()
    cells=cached_display_cells(paper)
    assert len(cells)==162 and cells.unit.eq('deg').all()
    a=json.loads((paper/'grid27_carron_delta_S_matrices_intensity.metadata.json').read_text())
    b=json.loads((paper/'grid27_chen_delta_S_matrices_intensity.metadata.json').read_text())
    assert a['intensity_vmax']==b['intensity_vmax']
    assert a['unit']==b['unit']=='deg'
    assert a['colourbar_gap_below_xlabel_inches']==pytest.approx(.22)
    assert a['title_x_figure']==a['subplot_grid_centre_x']
    assert a['x_ticklabels_bottom_row_only'] is True and not a['explanatory_notes_drawn']
    assert 'dimensionless' not in (paper/'grid27_carron_delta_S_matrices_intensity.caption.tex').read_text()
    for e in result['files']:assert file_digest(paper/e['file'])==e['sha256']
    # Cached exact angular summaries remain usable after moving/removing the runs.
    import shutil
    shutil.rmtree(fid);shutil.rmtree(grid)
    assert len(cached_display_cells(paper))==162
    (paper/'grid27_statistics.csv').write_text('modified')
    with pytest.raises(ValueError,match='Stale angular'):cached_display_cells(paper)


def test_angular_plots_require_paired_summary_not_just_unit_relabelling(tmp_path):
    from test_publication_update import grid_tables
    from spirality.publication_matrices import plot_delta_s_matrices
    tr,st=grid_tables()
    with pytest.raises(ValueError,match='require exact'):plot_delta_s_matrices(tr,st,tmp_path,unit='deg')


def test_conversion_keeps_other_caption_blocks_and_is_repeatable(tmp_path):
    from spirality.publication_captions import CARRON, CHEN, CARRON_TEX, CHEN_TEX
    paper,fid,grid=fixture_collection(tmp_path)
    other='% other.pdf\n\\caption{Author text unchanged.}\n\\label{fig:other}\n\n'
    combined=other+f'% {CARRON}.pdf\n\\caption{{{CARRON_TEX}}}\n\\label{{fig:delta_s_carron}}\n\n'+f'% {CHEN}.pdf\n\\caption{{{CHEN_TEX}}}\n\\label{{fig:delta_s_chen}}\n'
    path=paper/'captions_fans_revision.tex';path.write_text(combined)
    manifest=json.loads((paper/'manifest.json').read_text())
    manifest['files'].append(dict(file=path.name,sha256=file_digest(path),role='figure_caption',size_bytes=path.stat().st_size))
    (paper/'manifest.json').write_text(json.dumps(manifest))
    convert_collection(paper)
    text=path.read_text();assert text.startswith(other)
    assert 'dimensionless contrasts' not in text and 'applying $\\arcsin$ in each bin' in text
    first=pd.read_csv(paper/FID_TABLE)
    convert_collection(paper)
    pd.testing.assert_frame_equal(first,pd.read_csv(paper/FID_TABLE))
    assert path.read_text().startswith(other)
