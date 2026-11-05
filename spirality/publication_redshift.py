"""Publication-only multiplicity/redshift figures from the revised saved tables.

The four-bin products need only the compact publication collection. A Gaussian
KDE additionally requires the *individual* accepted node redshifts, either from
a validated fiducial node table or the small cache this module exports. Coarse
bin counts, bin centres and the all-multiplicity KDE are never used to invent
individual redshifts or a multiplicity-resolved KDE.
"""
from __future__ import annotations
from .publication_style import save_publication_figure, FRAME_LINEWIDTH

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from .provenance import file_digest
from .selection import effective_redshift, redshift_bin_ids, validate_nodes
from .publication_matrices import _bool

CATS = ('carron', 'chen')
LABEL = {'carron': 'Carrón DR16', 'chen': 'Chen DR12'}
COLOR = {'carron': 'royalblue', 'chen': 'orangered'}
MULTIPLICITIES = (2, 3)
LINE_M = {2: '-', 3: '--'}
LINE_STAGE = {'before': '--', 'after': '-'}
CACHE = 'fans_redshift_multiplicity_nodes.csv'
CACHE_META = 'fans_redshift_multiplicity_nodes.metadata.json'
DEFAULTS = {'multiplicity_before_after_figsize': [17.8, 7.2]}


def _integers(series, label):
    x = pd.to_numeric(series, errors='coerce').to_numpy(float)
    if not np.isfinite(x).all() or (x < 0).any() or not np.equal(x, np.rint(x)).all():
        raise ValueError(f'{label} must be finite nonnegative integers')
    return x.astype(int)


def binned_multiplicity(folder: Path) -> tuple[pd.DataFrame, np.ndarray, list[Path]]:
    """Validate number fractions; retain all multiplicities in the denominator."""
    folder = Path(folder); pieces = []; dependencies = []; edges = None
    for cat in CATS:
        sf = folder/f'{cat}_multiplicity_statistics.csv'
        ff = folder/f'{cat}_multiplicity_fractions.csv'
        stat = pd.read_csv(sf); frac = pd.read_csv(ff)
        dependencies.extend([sf, ff])
        bins = stat[(stat.scope=='zbin') & (stat.stratum=='all')].sort_values('zbin_id')
        if bins.zbin_id.tolist() != [0,1,2,3]:
            raise ValueError(f'{cat}: four unique redshift bins required')
        ee = np.r_[bins.z_min.to_numpy(float), float(bins.z_max.iloc[-1])]
        if not np.isfinite(ee).all() or not (np.diff(ee)>0).all() or not np.allclose(bins.z_max, ee[1:]):
            raise ValueError('Invalid/noncontiguous redshift bins')
        if edges is not None and not np.array_equal(edges,ee):
            raise ValueError('Catalogue redshift bins disagree')
        edges = ee
        if frac.duplicated(['scope','zbin_id','multiplicity']).any():
            raise ValueError('Duplicate multiplicity count rows')
        full = frac[frac.scope=='full'].set_index('multiplicity')
        d = frac[frac.scope=='zbin'].copy()
        d['n_fans'] = _integers(d.n_fans, 'Multiplicity counts')
        d['n_total_fans'] = _integers(d.n_total_fans, 'Total counts')
        d['multiplicity'] = _integers(d.multiplicity, 'Accepted multiplicity')
        if (d.multiplicity<2).any():
            raise ValueError('A final FANS cannot have fewer than two arms')
        if sorted(d.zbin_id.unique()) != [0,1,2,3]:
            raise ValueError('Fractions have inconsistent redshift bins')
        for b,g in d.groupby('zbin_id'):
            n = int(bins[bins.zbin_id==b].n_fans.iloc[0])
            if not (g.n_total_fans==n).all() or int(g.n_fans.sum())!=n:
                raise ValueError('Multiplicity counts do not partition the accepted sample')
            if not np.allclose(g.fraction_nodes, g.n_fans/n, rtol=1e-10, atol=1e-12):
                raise ValueError('Stored number fractions disagree with counts')
        for m,g in d.groupby('multiplicity'):
            if m not in full.index or int(g.n_fans.sum())!=int(full.loc[m,'n_fans']):
                raise ValueError('Full-range and binned multiplicity counts disagree')
        d['catalog'] = cat
        d['z_min'] = d.zbin_id.map(dict(zip(bins.zbin_id,bins.z_min)))
        d['z_max'] = d.zbin_id.map(dict(zip(bins.zbin_id,bins.z_max)))
        total_by_m = d.groupby('multiplicity').n_fans.sum()
        d['n_multiplicity_fullrange'] = d.multiplicity.map(total_by_m)
        denominator = d.n_multiplicity_fullrange*(d.z_max-d.z_min)
        d['density_given_multiplicity'] = np.divide(d.n_fans, denominator,
            out=np.full(len(d),np.nan), where=denominator.to_numpy()>0)
        pieces.append(d)
    return pd.concat(pieces,ignore_index=True), edges, dependencies


def before_after_counts(folder: Path, binned: pd.DataFrame) -> tuple[pd.DataFrame,list[Path]]:
    """Operational candidates vs accepted multiplicities on the edge-safe sample.

    'Before' is after ridge/seed orientation and angular grouping, but before
    radial trace and final FANS cuts. It is NOT pre-reconstruction physical
    connectivity. Counts compare changing multiplicities, not a survival rate.
    """
    rows = []; dependencies = []; folder = Path(folder)
    for cat in CATS:
        path = folder/f'{cat}_multiplicity_selection_stages.csv'
        stages = pd.read_csv(path); dependencies.append(path)
        pre = stages[(stages.population=='edge_safe') & (stages.stage=='candidate_arms') & (stages.scope=='zbin')]
        if pre.duplicated(['zbin_id','multiplicity']).any():
            raise ValueError('Duplicate before-selection multiplicity rows')
        trpath = folder/f'{cat}_multiplicity_transitions.csv'
        tr = pd.read_csv(trpath); dependencies.append(trpath)
        tr = tr[(tr.scope=='zbin') & tr.edge_safe.map(_bool)].copy()
        for b in range(4):
            p = pre[pre.zbin_id==b]
            if p.empty:
                raise ValueError(f'Missing edge-safe candidate audit for {cat}/bin{b}')
            if p.n_population.nunique()!=1 or p.n_evaluated.nunique()!=1:
                raise ValueError('Inconsistent audit population denominator')
            known = p[p.multiplicity>=0]; unknown = p[p.multiplicity<0]
            n_unknown = int(unknown.n_nodes.sum())
            if int(known.n_nodes.sum()) != int(p.n_evaluated.iloc[0]) or int(p.n_population.iloc[0]) != int(known.n_nodes.sum()) + n_unknown:
                raise ValueError('Audit unknown/evaluated counts do not partition the population')
            t = tr[tr.zbin_id==b]
            for _,r in known.iterrows():
                if int(t.loc[t.m_candidates==r.multiplicity,'n_nodes'].sum())!=int(r.n_nodes):
                    raise ValueError('Candidate histogram disagrees with paired transition audit')
            final = binned[(binned.catalog==cat)&(binned.zbin_id==b)]
            for _,r in final.iterrows():
                tn = t[(t.parent_status=='ok') & (t.m_trace_survivors==r.multiplicity)].n_nodes.sum()
                if int(tn)!=int(r.n_fans):
                    raise ValueError('Accepted transitions disagree with final FANS counts')
            before_total = int(known.loc[known.multiplicity>=2,'n_nodes'].sum())
            after_total = int(final.n_fans.sum())
            zlo,zhi = float(final.z_min.iloc[0]),float(final.z_max.iloc[0])
            for m in MULTIPLICITIES:
                np_before = int(known.loc[known.multiplicity==m,'n_nodes'].sum())
                np_after = int(final.loc[final.multiplicity==m,'n_fans'].sum())
                for stage,n,denom in [('before',np_before,before_total),('after',np_after,after_total)]:
                    rows.append(dict(catalog=cat,zbin_id=b,z_min=zlo,z_max=zhi,multiplicity=m,stage=stage,
                        n_nodes=n,n_stage_multige2=denom,fraction_within_stage_multige2=n/denom if denom else np.nan,
                        n_edge_safe=int(p.n_population.iloc[0]),n_evaluated=int(p.n_evaluated.iloc[0]),
                        n_unknown_candidate_multiplicity=n_unknown,
                        multiplicity_definition='candidate_arms' if stage=='before' else 'accepted_valid_arms'))
    return pd.DataFrame(rows), dependencies


def _finish_axes(ax, edges, title, ylabel):
    ax.set(xlim=(float(edges[0]),float(edges[-1])), xlabel=r'$z$', ylabel=ylabel, title=title)
    for ze in edges[1:-1]:
        ax.axvline(ze,color='black',lw=.9,alpha=.65,zorder=0)
    ax.grid(True,color='.82',lw=.75,alpha=.35)
    ax.tick_params(direction='in',top=True,right=True,length=7,width=1.4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)


def _legends(ax, *, stages=False, fontsize=18):
    catalogs=[Line2D([0],[0],color=COLOR[c],lw=3,label=LABEL[c]) for c in CATS]
    first=ax.legend(handles=catalogs,loc='upper right',fontsize=fontsize,frameon=True)
    first.get_frame().set_linewidth(1.1);ax.add_artist(first)
    if stages:
        handles=[Line2D([0],[0],color='.2',lw=2.6,ls='--',label='Before trace cuts'),
                 Line2D([0],[0],color='.2',lw=2.6,ls='-',label='After FANS cuts')]
    else:
        handles=[Line2D([0],[0],color='.2',lw=2.6,ls=LINE_M[m],label=f'{m} arms') for m in MULTIPLICITIES]
    second=ax.legend(handles=handles,loc='center right',fontsize=fontsize,frameon=True)
    second.get_frame().set_linewidth(1.1)


def _save(fig, out, data, metadata):
    out = Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    save_publication_figure(fig,out,bbox_inches='tight',dpi=300);plt.close(fig)
    values=out.with_suffix('.used_values.csv');data.to_csv(values,index=False)
    meta=out.with_suffix('.metadata.json')
    meta.write_text(json.dumps({**metadata,'no_scientific_inference_recomputed':True},indent=2,ensure_ascii=False))
    return [out,values,meta]


def plot_binned_multiplicity(binned, edges, outdir, style):
    """Two distinct quantities: p(z|m) and P(m|z,accepted), never interchanged."""
    from .paper_products import style_context
    outdir=Path(outdir);data=binned[binned.multiplicity.isin(MULTIPLICITIES)].copy();paths=[]
    with style_context(style):
        for mode in ('density','fraction'):
            fig,ax=plt.subplots(figsize=style['redshift_figsize'])
            for cat in CATS:
                for m in MULTIPLICITIES:
                    d=data[(data.catalog==cat)&(data.multiplicity==m)].sort_values('zbin_id')
                    if len(d)!=len(edges)-1:raise ValueError('Missing accepted multiplicity/bin count')
                    key='density_given_multiplicity' if mode=='density' else 'fraction_nodes'
                    ax.stairs(d[key],edges,baseline=None,color=COLOR[cat],ls=LINE_M[m],lw=3)
            name='fans_redshift_distribution_by_multiplicity_step' if mode=='density' else 'fans_multiplicity_fractions_step'
            title='FANS redshift distribution by arm number' if mode=='density' else 'FANS arm-multiplicity fractions'
            _finish_axes(ax,edges,title,'Density' if mode=='density' else 'Fraction of accepted FANS')
            # Legend outside the data area so the near-unity two-arm fraction is visible.
            handles=[Line2D([0],[0],color=COLOR[c],lw=3,label=LABEL[c]) for c in CATS]
            handles += [Line2D([0],[0],color='.2',lw=2.6,ls=LINE_M[m],label=f'{m} arms') for m in MULTIPLICITIES]
            ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,
                      fontsize=style['font_size']-5,frameon=True)
            ax.set_ylim(bottom=0) if mode=='density' else ax.set_ylim(-.025,1.025)
            fig.tight_layout()
            paths.extend(_save(fig,outdir/f'{name}.pdf',data,{
                'quantity':'p(z | catalogue, m, accepted)' if mode=='density' else 'P(m | catalogue, z-bin, accepted)',
                'definition':'N_m,b / (sum_b N_m,b * bin_width)' if mode=='density' else 'N_m,b / N_all_accepted,b',
                'all_multiplicities_retained_in_fraction_denominator':True,'one_count_per_node':True,
                'edges':edges.tolist(),'source_resolution':'four saved analysis bins; no KDE or synthetic individual redshifts',
                'colours':COLOR,'line_styles':{str(m):LINE_M[m] for m in MULTIPLICITIES},
                'uncertainty_display':'not displayed; saved fraction bootstrap columns retained in used_values'}))
    return paths


def plot_before_after(data, edges, outdir, style):
    from .paper_products import style_context
    paths=[]
    with style_context(style):
        for mode in ('counts','fractions'):
            fig,axes=plt.subplots(1,2,figsize=style.get('multiplicity_before_after_figsize',DEFAULTS['multiplicity_before_after_figsize']))
            for ax,m in zip(axes,MULTIPLICITIES):
                for cat in CATS:
                    for stage in ('before','after'):
                        d=data[(data.catalog==cat)&(data.multiplicity==m)&(data.stage==stage)].sort_values('zbin_id')
                        key='n_nodes' if mode=='counts' else 'fraction_within_stage_multige2'
                        ax.stairs(d[key],edges,baseline=None,color=COLOR[cat],ls=LINE_STAGE[stage],lw=3)
                _finish_axes(ax,edges,f'{m} arms','Number of nodes' if mode=='counts' else r'Fraction within stage ($m\geq2$)')
                if mode=='counts':ax.set_ylim(bottom=0)
                else:ax.set_ylim(-.025,1.025)
            handles=[Line2D([0],[0],color=COLOR[c],lw=3,label=LABEL[c]) for c in CATS]
            handles += [Line2D([0],[0],color='.2',ls=LINE_STAGE[s],lw=2.6,
                        label='Before trace cuts' if s=='before' else 'After FANS cuts') for s in ('before','after')]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.08),ncol=4,frameon=True,
                       fontsize=style['font_size']-5,handlelength=2.2)
            fig.text(.5,.027,'Edge-safe nodes: candidate arms before tracing; accepted FANS after cuts. Multiplicity can change.',
                     ha='center',fontsize=14)
            fig.subplots_adjust(left=.075,right=.98,top=.9,bottom=.27,wspace=.22)
            paths.extend(_save(fig,Path(outdir)/f'fans_multiplicity_{mode}_before_after_step.pdf',data,{
                'quantity':'absolute numbers' if mode=='counts' else 'stage-specific fractions among multiplicities >=2',
                'before':'edge_safe evaluated nodes, candidate_arms after initial selection/grouping and before radial trace cuts',
                'after':'accepted FANS; valid-arm multiplicity; verified against paired transitions and final counts',
                'before_is_physical_connectivity':False,'before_includes_all_quality_cuts':False,
                'unknown_multiplicity':'excluded from known classes; counts retained explicitly, never treated as zero',
                'denominator':'none (raw counts)' if mode=='counts' else 'all nodes with m>=2 at the same stage, including m>=4',
                'not_a_survival_fraction':True,'edges':edges.tolist(),'colours':COLOR,'stage_line_styles':LINE_STAGE,
                'uncertainty_display':'none; no new counting-error model or bootstrap introduced'}))
    return paths


def validate_redshift_cache(nodes: pd.DataFrame, binned: pd.DataFrame, edges) -> None:
    required={'catalog','node_id','z','multiplicity'}
    if not required.issubset(nodes):raise ValueError(f'Redshift cache missing columns: {required-set(nodes)}')
    if set(nodes.catalog)!=set(CATS) or nodes.duplicated(['catalog','node_id']).any():
        raise ValueError('Redshift cache must contain both catalogues, once per unique node')
    m=_integers(nodes.multiplicity,'Cached multiplicity');z=pd.to_numeric(nodes.z,errors='coerce').to_numpy(float)
    if (m<2).any() or not np.isfinite(z).all():raise ValueError('Invalid accepted-node redshift/multiplicity')
    bid=redshift_bin_ids(z,edges)
    if (bid<0).any():raise ValueError('Cached node redshift outside the scientific bins')
    counts=nodes.assign(zbin_id=bid).groupby(['catalog','zbin_id','multiplicity']).size()
    expected=binned.set_index(['catalog','zbin_id','multiplicity']).n_fans
    joint=counts.index.union(expected.index)
    if not np.array_equal(counts.reindex(joint,fill_value=0),expected.reindex(joint,fill_value=0)):
        raise ValueError('Individual redshifts do not reproduce the saved multiplicity/bin counts')


def prepare_redshift_cache(folder, binned, edges, fiducial_run=None):
    """Return genuine individual redshifts or an explicit missing-data status."""
    from .multiplicity import catalog_directory
    folder=Path(folder);cache=folder/CACHE;metadata=folder/CACHE_META
    if cache.exists():
        if not metadata.exists():raise ValueError('Redshift cache lacks provenance metadata')
        meta=json.loads(metadata.read_text())
        if meta.get('cache_sha256')!=file_digest(cache):raise ValueError('Modified redshift cache')
        nodes=pd.read_csv(cache);validate_redshift_cache(nodes,binned,edges)
        for cat in CATS:
            current=json.loads((folder/f'{cat}_multiplicity_metadata.json').read_text())
            if meta.get('parent_sha256',{}).get(cat)!=current.get('parent_sha256'):
                raise ValueError('Cached redshift source differs from the current scientific sample')
        return nodes,[cache,metadata],[cache,metadata],'cached_individual_redshifts'
    if fiducial_run is None:
        return None,[],[],'pending_individual_redshifts'
    run=Path(fiducial_run).expanduser().resolve();parts=[];dependencies=[];parents={}
    for cat in CATS:
        source=catalog_directory(run,cat)/'node_spirality.csv'
        if not source.is_file():raise FileNotFoundError(f'Individual node table required for KDE: {source}')
        meta_path=folder/f'{cat}_multiplicity_metadata.json'
        meta=json.loads(meta_path.read_text())
        digest=file_digest(source)
        if not meta.get('parent_sha256') or digest!=meta['parent_sha256']:
            raise ValueError(f'Fiducial node table differs from the collected scientific sample: {source}')
        n=pd.read_csv(source);validate_nodes(n);n=n[n.status.eq('ok')].copy()
        zz=effective_redshift(n)
        if not np.isfinite(zz).all():raise ValueError('Nonfinite redshift in accepted nodes')
        keep=redshift_bin_ids(zz,edges)>=0
        parts.append(pd.DataFrame(dict(catalog=cat,node_id=n.loc[keep,'node_id'].to_numpy(),
                     z=zz[keep],multiplicity=n.loc[keep,'n_arms_valid'].to_numpy(int))))
        parents[cat]=digest;dependencies.extend([source,meta_path])
    nodes=pd.concat(parts,ignore_index=True);validate_redshift_cache(nodes,binned,edges)
    nodes.to_csv(cache,index=False)
    metadata.write_text(json.dumps({'source':'accepted individual node rows; no redshift synthesis',
        'parent_sha256':parents,'cache_sha256':file_digest(cache),'edges':edges.tolist(),
        'selection':'parent status=ok and same effective_redshift within the scientific interval',
        'multiplicity':'n_arms_valid of the accepted node','n_nodes':len(nodes),
        'sources':[str(p) for p in dependencies]},indent=2))
    return nodes,[cache,metadata],dependencies,'validated_fiducial_individual_redshifts'


def plot_kde_multiplicity(nodes, edges, outdir, style):
    from .paper_products import style_context
    grid=np.linspace(edges[0],edges[-1],int(style['kde_samples']));curve=[];methods=[]
    with style_context(style):
        fig,ax=plt.subplots(figsize=style['redshift_figsize'])
        for cat in CATS:
            for m in MULTIPLICITIES:
                z=nodes.loc[(nodes.catalog==cat)&(nodes.multiplicity==m),'z'].to_numpy(float)
                method='gaussian_kde_scott';reason=None
                try:
                    if len(z)<5 or np.ptp(z)==0:raise ValueError('Fewer than five nodes or zero redshift variance')
                    y=gaussian_kde(z)(grid)
                    ax.plot(grid,y,color=COLOR[cat],ls=LINE_M[m],lw=3)
                    curve.extend(dict(catalog=cat,multiplicity=m,z=float(x),density=float(v),method=method,n_fans=len(z)) for x,v in zip(grid,y))
                except (ValueError,np.linalg.LinAlgError) as exc:
                    reason=str(exc);method='histogram_density'
                    if len(z):
                        bins=np.linspace(edges[0],edges[-1],int(style['histogram_bins'])+1)
                        count,_=np.histogram(z,bins=bins);y=count/(len(z)*np.diff(bins))
                        ax.stairs(y,bins,baseline=None,color=COLOR[cat],ls=LINE_M[m],lw=3)
                        curve.extend(dict(catalog=cat,multiplicity=m,z=float((a+b)/2),z_min=float(a),z_max=float(b),
                            density=float(v),method=method,n_fans=len(z)) for a,b,v in zip(bins[:-1],bins[1:],y))
                methods.append(dict(catalog=cat,multiplicity=m,n_fans=len(z),method=method,fallback_reason=reason))
        _finish_axes(ax,edges,'FANS redshift distribution by arm number','Density')
        ax.set_ylim(bottom=0)
        handles=[Line2D([0],[0],color=COLOR[c],lw=3,label=LABEL[c]) for c in CATS]
        handles += [Line2D([0],[0],color='.2',lw=2.6,ls=LINE_M[m],label=f'{m} arms') for m in MULTIPLICITIES]
        ax.legend(handles=handles,loc='upper right',fontsize=style['font_size']-5,frameon=True)
        fig.tight_layout()
        return _save(fig,Path(outdir)/'fans_redshift_distribution_by_multiplicity.pdf',pd.DataFrame(curve),{
            'quantity':'p(z | catalogue, m, accepted)','methods':methods,'edges':edges.tolist(),
            'density_note':'Each catalogue/multiplicity KDE has unit integral on the real line; shown only within the analysis bounds. No boundary renormalization, as in the original overall plot.',
            'source':'genuine individual accepted node redshifts, one count per node',
            'colours':COLOR,'multiplicity_line_styles':LINE_M})


def generate_redshift_products(folder: Path, style: Mapping, fiducial_run=None):
    binned,edges,dependencies=binned_multiplicity(folder)
    before_after,deps=before_after_counts(folder,binned);dependencies+=deps
    paths=plot_binned_multiplicity(binned,edges,folder,style)
    paths+=plot_before_after(before_after,edges,folder,style)
    nodes,cachepaths,deps,status=prepare_redshift_cache(folder,binned,edges,fiducial_run)
    dependencies+=deps;paths+=cachepaths
    if nodes is not None:paths+=plot_kde_multiplicity(nodes,edges,folder,style)
    return paths,status,dependencies
