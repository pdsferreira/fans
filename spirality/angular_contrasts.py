"""Exact angular endpoint contrasts from the *saved paired* bootstrap draws.

This module never reconstructs arms, draws samples, assumes independent bins,
or transforms the dimensionless contrast itself. Original science tables are
read-only. The published quantity is degrees(arcsin(S4) - arcsin(S1)).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .provenance import file_digest
from .resampling import bootstrap_summary
from .publication_matrices import STRATA, PARAMETERS, matrix_cells

GRID_TABLE = 'grid27_angular_contrasts.csv'
FID_TABLE = 'fiducial_angular_contrasts.csv'
CACHE_META = 'angular_contrasts_provenance.json'
REF_TABLE = 'referee_spirality_contrasts_degrees.tex'
PLOT_NAMES = tuple(f'grid27_{c}_delta_S_matrices_intensity' for c in ('carron','chen'))
SOURCE_NAMES = ('multiplicity_statistics.csv', 'multiplicity_trends.csv',
                'bootstrap_column_index.csv', 'multiplicity_metadata.json',
                'multiplicity_bootstrap_samples.npz')
SUMMARY_FIELDS = ('bootstrap_n_valid', 'bootstrap_valid_fraction', 'err', 'q16', 'q84', 'q025', 'q975')
STAT_KEYS = ['stratum','scope','zbin_id']
DEFINITION = '(180/pi) * (arcsin(S_bin4) - arcsin(S_bin1))'


def _same(a, b, context: str) -> None:
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if aa.shape != bb.shape or not np.allclose(aa, bb, rtol=2e-8, atol=2e-11, equal_nan=True):
        raise ValueError(f'{context}: saved quantities do not match. Use the summaries and bootstrap archive from the same execution; no output was replaced.')


def _check_s(values, context: str) -> np.ndarray:
    a = np.asarray(values, float)
    if np.any(np.isinf(a)) or np.any(np.isfinite(a) & ((a < 0.) | (a > 1.))):
        raise ValueError(f'{context}: spirality outside [0,1]; not clipped for arcsin')
    return a


def angular_difference(first, last) -> np.ndarray:
    """Apply the angular transformation to each endpoint *before* subtraction."""
    first, last = _check_s(first, 'first bin'), _check_s(last, 'last bin')
    if first.shape != last.shape:
        raise ValueError('Paired endpoints must have the same shape')
    return np.degrees(np.arcsin(last) - np.arcsin(first))


def _supported(value) -> bool:
    from .publication_matrices import _bool
    return _bool(value)


def _compare_collected(local, collected, *, context):
    """The collection must match the actual run, including bootstrap summaries."""
    local = local[local.stratum.isin(STRATA)].copy()
    collected = collected[collected.stratum.isin(STRATA)].copy()
    if local.duplicated(STAT_KEYS).any() or collected.duplicated(STAT_KEYS).any():
        raise ValueError(f'{context}: duplicate statistic rows')
    a, b = (x.set_index(STAT_KEYS).sort_index() for x in (local, collected))
    if not a.index.equals(b.index):
        raise ValueError(f'{context}: different row identities in run and collection')
    fields=['z_min','z_max','S','n_fans','n_blocks','precision_support','S_err',
            'S_q025','S_q975','S_bootstrap_n_valid','S_bootstrap_valid_fraction']
    for key in fields:
        if key not in a or key not in b:
            raise ValueError(f'{context}: missing statistic column {key}')
        if key=='precision_support':
            if a[key].map(_supported).tolist()!=b[key].map(_supported).tolist():
                raise ValueError(f'{context}: endpoint support flags differ')
        else:
            _same(a[key], b[key], f'{context}/{key}')


def summarize_saved_endpoints(statistics: pd.DataFrame, trends: pd.DataFrame,
                              index: pd.DataFrame, samples: np.ndarray,
                              metadata: dict) -> pd.DataFrame:
    """Validate archive/index against saved summaries; derive exact angular CIs."""
    required=set(STAT_KEYS+['S','z_min','z_max','n_fans','n_blocks','precision_support'])
    if not required.issubset(statistics) or not set(STAT_KEYS+['column_id']).issubset(index):
        raise ValueError('Missing bootstrap/statistic row-index columns')
    if statistics.duplicated(STAT_KEYS).any() or index.duplicated(STAT_KEYS).any():
        raise ValueError('Duplicate bootstrap index or statistic rows')
    n=len(statistics)
    if not np.array_equal(index.column_id.to_numpy(), np.arange(n)) or len(index)!=n:
        raise ValueError('bootstrap_column_index must contain consecutive ordered column IDs')
    if index[STAT_KEYS].astype(str).to_dict('records') != statistics[STAT_KEYS].astype(str).to_dict('records'):
        raise ValueError('Bootstrap column index does not match statistic row order')
    samples=_check_s(samples, 'saved S bootstrap array')
    if samples.ndim!=2 or samples.shape[1]!=n or samples.shape[0] < 2:
        raise ValueError(f'Expected S array [B,{n}] with B>=2; found {samples.shape}')
    if int(metadata.get('B',-1))!=samples.shape[0]:
        raise ValueError('Saved bootstrap archive size differs from metadata B')
    if str(metadata.get('schema'))!='88.1':
        raise ValueError('Only science schema 88.1 is supported')
    # This establishes that the index pairs the original draws, not marginal
    # quantiles or bootstrap means generated with another seed/version.
    for pos,r in statistics.iterrows():
        if r.stratum not in STRATA:
            continue
        summ=bootstrap_summary(samples[:,int(pos)])
        for field in SUMMARY_FIELDS:
            col='S_'+field
            if col in statistics:
                _same(summ[field], r[col], f'Archive validation/{r.stratum}/{r.scope}/{r.zbin_id}/{col}')
    rows=[]
    for stratum in STRATA:
        st=statistics[(statistics.stratum==stratum)&(statistics.scope=='zbin')].sort_values('zbin_id')
        if st.zbin_id.tolist()!=[0,1,2,3]:
            raise ValueError(f'{stratum}: expected exactly four redshift bins')
        first,last=st.iloc[0],st.iloc[-1]
        _same([first.z_min,first.z_max,last.z_min,last.z_max],[.1,.225,.475,.6], 'Endpoint bin boundaries')
        tr=trends[trends.stratum==stratum]
        if len(tr)!=1:
            raise ValueError(f'{stratum}: missing/duplicate saved contrast')
        tr=tr.iloc[0]
        a,b=samples[:,int(first.name)],samples[:,int(last.name)]
        raw=bootstrap_summary(b-a)
        _same(float(last.S)-float(first.S),tr.delta_S_last_minus_first,f'{stratum}/central raw delta')
        for key in SUMMARY_FIELDS:
            name='delta_S_'+key
            if name in trends:
                _same(raw[key],tr[name],f'{stratum}/paired raw delta/{key}')
        deg=angular_difference(a,b)
        summary=bootstrap_summary(deg)
        central=float(angular_difference(float(first.S),float(last.S)))
        first_ok,last_ok=_supported(first.precision_support),_supported(last.precision_support)
        finite_interval=np.isfinite([central,summary['err'],summary['q025'],summary['q975']]).all()
        support=bool(first_ok and last_ok and finite_interval)
        rows.append(dict(stratum=stratum,unit='deg',definition=DEFINITION,
            S_first=float(first.S),S_last=float(last.S),
            delta_S_dimensionless=float(last.S)-float(first.S),
            delta_S_deg=central,**{'delta_S_deg_'+k:v for k,v in summary.items()},
            bootstrap_B=int(samples.shape[0]),
            n_fans_first=int(first.n_fans),n_fans_last=int(last.n_fans),
            n_blocks_first=int(first.n_blocks),n_blocks_last=int(last.n_blocks),
            precision_support_first=first_ok,precision_support_last=last_ok,
            plot_support=support,
            interval_excludes_zero=bool(support and (summary['q025']>0 or summary['q975']<0)),
            dimensionless_interval_excludes_zero=bool(first_ok and last_ok and
                np.isfinite([raw['q025'],raw['q975']]).all() and (raw['q025']>0 or raw['q975']<0))))
    return pd.DataFrame(rows)


def _read_run(folder, expected_stats, expected_trends, *, catalog, run_name):
    paths={n:Path(folder)/n for n in SOURCE_NAMES}
    local_stats=pd.read_csv(paths['multiplicity_statistics.csv']).reset_index(drop=True)
    local_trends=pd.read_csv(paths['multiplicity_trends.csv'])
    index=pd.read_csv(paths['bootstrap_column_index.csv'])
    meta=json.loads(paths['multiplicity_metadata.json'].read_text())
    _compare_collected(local_stats,expected_stats,context=f'{catalog}/{run_name}')
    for stratum in STRATA:
        a=local_trends[local_trends.stratum==stratum]
        b=expected_trends[expected_trends.stratum==stratum]
        if len(a)!=1 or len(b)!=1:
            raise ValueError(f'Missing contrast for {catalog}/{run_name}/{stratum}')
        for col in ['delta_S_last_minus_first','delta_S_err','delta_S_q025','delta_S_q975']:
            _same(a.iloc[0][col], b.iloc[0][col],f'{catalog}/{run_name}/{stratum}/{col}')
    with np.load(paths['multiplicity_bootstrap_samples.npz'],allow_pickle=False) as archive:
        if 'S' not in archive:
            raise ValueError(f'Missing S array in {paths["multiplicity_bootstrap_samples.npz"]}')
        result=summarize_saved_endpoints(local_stats,local_trends,index,archive['S'],meta)
    result.insert(0,'run_name',run_name);result.insert(0,'catalog',catalog)
    return result,list(paths.values())


def prepare_angular_tables(folder, fiducial_run, grid_root):
    """Read 2 fiducial + 54 grid archives. Fail before writing if any is missing."""
    folder,fiducial_run,grid_root=map(Path,(folder,fiducial_run,grid_root))
    gs=pd.read_csv(folder/'grid27_statistics.csv');gt=pd.read_csv(folder/'grid27_trends.csv')
    raw_cells=matrix_cells(gt,gs)  # validates complete grid and all endpoint identities
    jobs=[]
    for cat in ('carron','chen'):
        jobs.append(('fiducial',cat,fiducial_run.name,fiducial_run/'referee_multiplicity'/cat,
             pd.read_csv(folder/f'{cat}_multiplicity_statistics.csv'),
             pd.read_csv(folder/f'{cat}_multiplicity_trends.csv')))
    for (cat,run),stats in gs.groupby(['catalog','run_name'],sort=True):
        if Path(run).name!=run:
            raise ValueError(f'Unsafe run name: {run}')
        jobs.append(('grid',cat,run,grid_root/run/'referee_multiplicity'/cat,
                     stats,gt[(gt.catalog==cat)&(gt.run_name==run)]))
    missing=[p/n for _,_,_,p,_,_ in jobs for n in SOURCE_NAMES if not (p/n).is_file()]
    if missing:
        short='\n'.join('  '+str(p) for p in missing[:8])
        raise FileNotFoundError(f'{len(missing)} required saved files are missing:\n{short}\nUse the completed fiducial/grid folders with multiplicity_bootstrap_samples.npz. Marginal errors cannot replace paired bootstrap draws. No figures or tables were replaced.')
    data={'grid':[],'fiducial':[]};deps=[]
    for kind,cat,run,path,stats,trends in jobs:
        result,used=_read_run(path,stats,trends,catalog=cat,run_name=run)
        if kind=='grid':
            pars=stats[list(PARAMETERS)].drop_duplicates()
            if len(pars)!=1:raise ValueError('Run has inconsistent parameters')
            for k in PARAMETERS:result[k]=float(pars.iloc[0][k])
        data[kind].append(result);deps+=used
    grid=pd.concat(data['grid'],ignore_index=True);fid=pd.concat(data['fiducial'],ignore_index=True)
    plotted=display_cells(raw_cells,grid)
    return grid,fid,plotted,deps


def display_cells(raw_cells, angular):
    """Keep row geometry/support, replacing only displayed delta/error/interval."""
    keys=['catalog','run_name','stratum']
    if angular.duplicated(keys).any():raise ValueError('Duplicate angular contrasts')
    if set(map(tuple,angular[keys].to_numpy()))!=set(map(tuple,raw_cells[keys].to_numpy())):
        raise ValueError('Angular cache does not cover exactly the matrix cells')
    extras=angular.set_index(keys);out=raw_cells.copy(deep=True)
    for dest,src in [('delta_S_last_minus_first','delta_S_deg'),('delta_S_err','delta_S_deg_err'),
                     ('delta_S_q025','delta_S_deg_q025'),('delta_S_q975','delta_S_deg_q975')]:
        out[dest+'_dimensionless']=out[dest]
        out[dest]=[float(extras.loc[tuple(r),src]) for r in out[keys].to_numpy()]
    for i,r in out.iterrows():
        ar=extras.loc[(r.catalog,r.run_name,r.stratum)]
        _same([r.S_first,r.S_last],[ar.S_first,ar.S_last], 'Angular cache/endpoint means')
        _same(r.delta_S_last_minus_first,angular_difference(r.S_first,r.S_last),'Angular cache/central angle')
        if _supported(r.plot_support)!=_supported(ar.plot_support):
            raise ValueError('Angular conversion changed support: inspect archive/metadata mismatch')
        # Recomputed from angular realizations, never copied from raw delta.
        out.loc[i,'interval_excludes_zero']=_supported(ar.interval_excludes_zero)
        out.loc[i,'bootstrap_B']=int(ar.bootstrap_B)
    out['unit']='deg';out['definition']=DEFINITION
    return out


def cached_display_cells(folder):
    """Reuse exact converted summaries without rereading the bootstrap archives."""
    folder=Path(folder);path=folder/CACHE_META
    if not path.exists():return None
    meta=json.loads(path.read_text())
    for name,digest in meta['collection_inputs'].items():
        p=folder/name
        if not p.is_file() or file_digest(p)!=digest:
            raise ValueError(f'Stale angular contrasts: {name} changed; rerun convert_delta_s_to_degrees.py')
    for name,digest in meta['angular_tables'].items():
        if not (folder/name).is_file() or file_digest(folder/name)!=digest:
            raise ValueError(f'Modified angular cache: {name}')
    return display_cells(matrix_cells(pd.read_csv(folder/'grid27_trends.csv'),
                                     pd.read_csv(folder/'grid27_statistics.csv')),
                         pd.read_csv(folder/GRID_TABLE))


def write_referee_table(table, output):
    """A pasteable table, not a rewritten response letter. One prose line/paragraph."""
    rows=[]
    for cat,stratum,label in [('carron','m2',r"Carr\'on & Two arms"),('chen','m2','Chen & Two arms'),('carron','m3',r"Carr\'on & Three arms")]:
        part=table[(table.catalog==cat)&(table.stratum==stratum)]
        if len(part)!=1:raise ValueError('Incomplete fiducial referee table')
        r=part.iloc[0]
        if _supported(r.plot_support):
            rows.append(label+f' & ${r.delta_S_deg:.2f}\\pm{r.delta_S_deg_err:.2f}$ & $[{r.delta_S_deg_q025:.2f},\\,{r.delta_S_deg_q975:.2f}]$ '+r'\\')
        else:rows.append(label+r' & --- & Insufficient support \\')
    paragraph=r"Here $\Delta\mathcal{S}^{(m)}$ [deg] denotes the difference after applying $\arcsin$ separately to the spirality in each endpoint bin. Errors and 95\% percentile intervals are computed from these angular differences in the saved paired bootstrap realizations."
    body=paragraph+'\n\n'+r'\begin{center}'+'\n'+r'\begin{tabular}{llcc}'+'\n'+r'\toprule'+'\n'+r'Sample & Multiplicity & $\Delta\mathcal{S}^{(m)}$ [deg] & 95\% interval [deg] \\'+'\n'+r'\midrule'+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{center}'+'\n'
    output=Path(output);output.write_text(body,encoding='utf-8')
    return output


def convert_collection(folder, *, fiducial_run=None, grid_root=None, style_path=None, check_only=False):
    """Convert only the two intensity matrices + captions and add angular tables.

    The operation preflights every input and stages all outputs before replacing
    any managed product. Unrelated PDFs/data and manually named files survive.
    """
    from .paper_products import Bundle,_commit_bundle,load_style
    from .publication_matrices import plot_delta_s_matrices
    from .publication_captions import write_caption_files
    folder=Path(folder).expanduser().resolve();manifest=folder/'manifest.json'
    previous=json.loads(manifest.read_text())
    if previous.get('science_schema')!='88.1':raise ValueError('Expected scientific schema 88.1')
    old={}
    for e in previous['files']:
        name=e['file'];p=folder/name
        if Path(name).name!=name or name in old:raise ValueError('Unsafe/duplicate managed product')
        if not p.is_file() or file_digest(p)!=e['sha256']:
            raise ValueError(f'Missing/modified managed file: {p}; preserve manual edits under a separate name')
        old[name]=e
    source=previous.get('sources',{})
    fiducial_run=fiducial_run or source.get('fiducial_run');grid_root=grid_root or source.get('grid_root')
    if not fiducial_run or not grid_root:raise ValueError('Provide --fiducial-run and --grid-root')
    fiducial_run=Path(fiducial_run).expanduser().resolve();grid_root=Path(grid_root).expanduser().resolve()
    style=load_style()
    style.update({k:v for k,v in previous.get('plot_style',{}).items() if k in style})
    if style_path:
        given=json.loads(Path(style_path).read_text());unknown=set(given)-set(style)
        if unknown:raise ValueError(f'Unknown style settings: {sorted(unknown)}')
        style.update(given)
    grid,fid,cells,deps=prepare_angular_tables(folder,fiducial_run,grid_root)
    if check_only:
        return {'check_only':True,'grid_rows':len(grid),'fiducial_rows':len(fid),'bootstrap_B':sorted(set(grid.bootstrap_B)|set(fid.bootstrap_B))}
    with tempfile.TemporaryDirectory(prefix='.angular_contrasts_',dir=folder.parent) as tmp:
        bundle=Bundle(tmp);bundle.status=list(previous.get('status',[]))
        # Preserve archived provenance even when original source roots moved.
        bundle.deps={d['path']:d for d in previous.get('dependencies',[]) if 'path' in d}
        for e in previous['files']:bundle.add(folder/e['file'],role=e.get('role','support'),description=e.get('description',''))
        out=bundle.stage
        paths=[]
        grid.to_csv(out/GRID_TABLE,index=False);fid.to_csv(out/FID_TABLE,index=False)
        paths.extend([out/GRID_TABLE,out/FID_TABLE,write_referee_table(fid,out/REF_TABLE)])
        raw_names=['grid27_statistics.csv','grid27_trends.csv']+[f'{c}_multiplicity_{k}.csv' for c in ('carron','chen') for k in ('statistics','trends')]
        pm={'schema':'88.12','unit':'deg','definition':DEFINITION,
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'bootstrap_method':'Transform each endpoint in the SAME saved bootstrap row, then subtract; std ddof=1, percentile 2.5/97.5; original valid-fraction threshold retained.',
            'collection_inputs':{n:file_digest(folder/n) for n in raw_names},
            'angular_tables':{n:file_digest(out/n) for n in (GRID_TABLE,FID_TABLE)},
            'bootstrap_sources':[{'path':str(p),'sha256':file_digest(p)} for p in deps],
            'bootstrap_B':sorted(set(grid.bootstrap_B)|set(fid.bootstrap_B)),
            'no_new_resampling':True,'original_scientific_files_unchanged':True,
            'angular_vs_dimensionless_95_flags_changed':int((grid.interval_excludes_zero!=grid.dimensionless_interval_excludes_zero).sum())}
        (out/CACHE_META).write_text(json.dumps(pm,indent=2,ensure_ascii=False));paths.append(out/CACHE_META)
        drawstyle=dict(style);drawstyle['delta_matrix_versions']=['intensity']
        # Numeric scale now has units of degrees; keep automatic/shared scaling.
        drawstyle['delta_matrix_intensity_vmax']=None
        paths+=plot_delta_s_matrices(pd.read_csv(out/'grid27_trends.csv'),pd.read_csv(out/'grid27_statistics.csv'),out,
                                      style=drawstyle,display_cells=cells,unit='deg')
        cp,cd=write_caption_files(out,only=PLOT_NAMES);paths+=cp;deps+=cd
        # Update only the two caption blocks in the existing combined file.
        combined=out/'captions_fans_revision.tex'
        if combined.exists():
            import re
            text=combined.read_text()
            for name in PLOT_NAMES:
                pattern=r'(% '+re.escape(name)+r'\.pdf\s*\n)\\caption\{[^\n]*\}\n\\label\{[^\n]*\}\n?'
                block=(out/f'{name}.caption.tex').read_text()
                text,n=re.subn(pattern,lambda m:m.group(1)+block,text)
                if n!=1:raise ValueError(f'Cannot isolate existing caption for {name}; update it manually rather than overwrite unrelated captions')
            combined.write_text(text);paths.append(combined)
        allowed={p.name for p in paths}
        for name,e in old.items():
            if name not in allowed and file_digest(out/name)!=e['sha256']:
                raise RuntimeError(f'Unrelated product changed: {name}')
        for p in paths:
            bundle.entries=[e for e in bundle.entries if e['file']!=p.name]
            bundle.add(p,generated=True,role='robustness_figure' if p.suffix=='.pdf' else 'angular_contrast_data')
        for p in deps:bundle.dependency(p)
        for n in ('angular_contrasts.py','publication_matrices.py','publication_captions.py'):
            bundle.dependency(Path(__file__).with_name(n))
        meta=dict(previous.get('metadata',{}));meta['angular_contrasts_88_12']=pm
        source=dict(source);source.update(fiducial_run=str(fiducial_run),grid_root=str(grid_root))
        result=_commit_bundle(bundle,folder,source,style,meta)
    return result
