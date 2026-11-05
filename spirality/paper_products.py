"""Publication plots and curated exports from completed v88 scientific products.

No reconstruction, bootstrap, sign flips or axis scans are performed here.
Author-provided colours/layouts are retained. Legacy v66/v87 statistics are
intentionally NOT a fallback for revised figures. All plot ordinates and errors
come from the joint-bootstrap multiplicity tables.
"""
from __future__ import annotations
from .publication_style import save_publication_figure, FRAME_LINEWIDTH, publication_frame_scope
import itertools
import json
import math
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, to_hex
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.stats import gaussian_kde
from .selection import effective_redshift, redshift_bin_ids, validate_nodes
from .multiplicity import catalog_directory
from .provenance import file_digest

CATS=('carron','chen')
LABEL={'carron':'Carrón DR16','chen':'Chen DR12'}
COLOR={'carron':'royalblue','chen':'orangered'}
MARKER={8.:'o',10.:'s',12.:'^'}
LINE={30.:'-',35.:'--',40.:':'}
PARAM_KEYS=('segment_length_mpc_h','node_link_radius_mpc_h','arm_opening_deg')
EXPECTED_GRID=set(itertools.product((7.5,10.,12.5),(8.,10.,12.),(30.,35.,40.)))
METRICS=(('H','H_proj',r'$H_{\rm proj}$'),('A_sign','A_sign',r'$A_{\rm sign}$'),('S','S_proj',r'$\mathcal{S}_{\rm proj}$'))
DEFAULT_STYLE={
    'font_size':24,'grid_figsize':[15.6,18.0],'redshift_figsize':[9.5,7.0],
    'grid_y_limits':{'H_proj':[-.018,.014],'A_sign':[-.032,.027],'S_proj':[.180,.330]},
    'grid_angular_y_limits':{'H_proj':[-.7,.52],'A_sign':[-2.2,4.3],'S_proj':[9.1,17.3]},
    'grid_angular_y_limit_policy':'strict',
    'y_limit_policy':'expand','footprint_nside':64,'density_per_deg2':False,
    'kde_samples':700,'histogram_bins':20,
    'panel_angular_upper_limits': {'H_proj': .79, 'A_sign': 5.1, 'A_dip': 8.8, 'A_ax': 5.3, 'S_proj': 15.4},
}
from .publication_matrices import DEFAULTS as _MATRIX_DEFAULTS
DEFAULT_STYLE.update(_MATRIX_DEFAULTS)
from .publication_redshift import DEFAULTS as _REDSHIFT_DEFAULTS
DEFAULT_STYLE.update(_REDSHIFT_DEFAULTS)
from .publication_panels import DEFAULTS as _PANEL_DEFAULTS
DEFAULT_STYLE.update(_PANEL_DEFAULTS)


def read_json(p):
    return json.loads(Path(p).read_text())


def load_style(path=None):
    cfg=json.loads(json.dumps(DEFAULT_STYLE))
    if path:
        given=read_json(path)
        unknown=set(given)-set(cfg)
        if unknown: raise ValueError(f'Unknown paper style settings: {sorted(unknown)}')
        cfg.update(given)
    if cfg['y_limit_policy'] not in {'expand','auto','strict'}:
        raise ValueError('y_limit_policy must be expand, auto or strict')
    if cfg['grid_angular_y_limit_policy'] not in {'expand','auto','strict'}:
        raise ValueError('Invalid grid angular y-limit policy')
    for key,lim in {**cfg['grid_y_limits'],**cfg['grid_angular_y_limits']}.items():
        if key not in {'H_proj','A_sign','S_proj'}:raise ValueError(f'Unknown metric {key}')
        if lim is not None and (len(lim)!=2 or not np.isfinite(lim).all() or lim[0]>=lim[1]):
            raise ValueError(f'Invalid axis limits for {key}')
    return cfg


def style_context(style):
    fs=style['font_size']
    return plt.rc_context({'font.family':'serif','font.serif':['Times New Roman','Times','Nimbus Roman','Liberation Serif','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':fs,'axes.labelsize':fs,'axes.titlesize':fs+1,
        'xtick.labelsize':fs-2,'ytick.labelsize':fs-2,'axes.linewidth':FRAME_LINEWIDTH,
        'xtick.major.width':1.2,'ytick.major.width':1.2,'xtick.direction':'in','ytick.direction':'in',
        'xtick.top':True,'ytick.right':True,'pdf.fonttype':42,'ps.fonttype':42})


def validate_revision_run(run, *, check_hashes=True):
    """Check completed revised science products before ever using their values."""
    run=Path(run).expanduser().resolve()
    state=run/'revision_projected_complete.json'
    if not state.exists(): raise FileNotFoundError(f'Completed revised projected stage required: {state}')
    state_data=read_json(state)
    if not str(state_data.get('signature',{}).get('schema','')).startswith('88.'):
        raise ValueError(f'Unsupported/legacy science schema in {state}; no fallback to v66/v87')
    edges=None; deps=[state]
    for cat in CATS:
        folder=run/'referee_multiplicity'/cat
        meta_path=folder/'multiplicity_metadata.json'
        audit_path=folder/'sample_integrity.json'
        meta=read_json(meta_path); audit=read_json(audit_path)
        if audit.get('passed') is not True: raise ValueError(f'Failed sample integrity: {audit_path}')
        if audit.get('n_analysis_arms')!=audit.get('sum_accepted_parent_multiplicities'):
            raise ValueError(f'Inconsistent arm/parent counts: {audit_path}')
        catdir=catalog_directory(run,cat)
        for filename,mkey in [('node_spirality.csv','parent_sha256'),('arm_level_spirality.csv','analysis_arm_sha256')]:
            p=catdir/filename
            if check_hashes and (not meta.get(mkey) or file_digest(p)!=meta[mkey]):
                raise ValueError(f'Stale/mismatched revised data: {p}; recompute science products, not only plots')
            deps.append(p)
        stats=pd.read_csv(folder/'multiplicity_statistics.csv')
        span=stats[(stats.stratum=='all')&(stats.scope=='zbin')].sort_values('zbin_id')
        if len(span)==0:raise ValueError(f'No redshift bins in {folder}')
        ee=np.r_[span.z_min.to_numpy(float),float(span.z_max.iloc[-1])]
        specified=state_data.get('signature',{}).get('edges')
        if specified is not None:
            se=np.array([float(x) for x in specified.split(',')]) if isinstance(specified,str) else np.asarray(specified,float)
            if not np.array_equal(ee,se):raise ValueError(f'Stored bins disagree with run signature: {folder}')
        if edges is not None and not np.array_equal(edges,ee):raise ValueError('Catalogue redshift bins disagree')
        edges=ee
        deps += [meta_path,audit_path,folder/'multiplicity_statistics.csv']
        _check_stats(pd.read_csv(folder/'multiplicity_statistics.csv'),edges,cat)
    return {'run':run,'edges':edges,'state':state_data,'dependencies':deps}


def _check_stats(stats,edges,context):
    required={'stratum','scope','zbin_id','z_min','z_max','H','H_err','S','S_err','A_sign','A_sign_err','n_fans','n_arms'}
    if not required.issubset(stats):raise ValueError(f'Missing revised statistical columns in {context}: {required-set(stats)}')
    if stats.duplicated(['stratum','scope','zbin_id']).any():raise ValueError(f'Duplicated statistical cells in {context}')
    for cl in ['all','m2','m3']:
        g=stats[(stats.stratum==cl)&(stats.scope=='zbin')].sort_values('zbin_id')
        if list(g.zbin_id)!=list(range(len(edges)-1)):
            raise ValueError(f'Incomplete redshift bins for {context}:{cl}')
        if not np.allclose(g.z_min,edges[:-1],rtol=0,atol=1e-12) or not np.allclose(g.z_max,edges[1:],rtol=0,atol=1e-12):
            raise ValueError(f'Redshift edges disagree in {context}:{cl}')
        for m,_,_ in METRICS:
            if (g[f'{m}_err'].dropna()<0).any():raise ValueError(f'Negative uncertainty in {context}:{m}')
    full=stats.query('stratum == "all" and scope == "full"')
    if len(full)!=1:raise ValueError(f'Full-range statistic missing/duplicated: {context}')


def load_accepted_nodes(run,cat,edges):
    """Use authoritative CSVs, not the old FITS QUALITY.astype(bool) fallback.

    The same effective-redshift priority as the revised selection is used. This
    counts nodes once, unlike the arm-weighted spirality estimators.
    """
    p=catalog_directory(run,cat)/'node_spirality.csv'
    n=pd.read_csv(p);validate_nodes(n)
    n=n[n.status.eq('ok')].copy()
    z=effective_redshift(n);ra=pd.to_numeric(n.ra,errors='coerce').to_numpy(float);dec=pd.to_numeric(n.dec,errors='coerce').to_numpy(float)
    finite=np.isfinite(z)&np.isfinite(ra)&np.isfinite(dec)&(np.abs(dec)<=90)
    if not finite.all():raise ValueError(f'Invalid coordinates/redshift in accepted nodes: {p}')
    inrange=redshift_bin_ids(z,edges)>=0
    # analyze_catalog restricts accepted parents to the declared redshift range.
    # Apply exactly that final selection here, even if a construction CSV also
    # retains edge-of-range parents for diagnostics.
    n=n.loc[inrange].copy();z=z[inrange];ra=ra[inrange];dec=dec[inrange]
    s=pd.read_csv(Path(run)/'referee_multiplicity'/cat/'multiplicity_statistics.csv')
    count=int(s.query('stratum=="all" and scope=="full"').n_fans.iloc[0])
    if len(n)!=count:raise ValueError(f'Distribution count disagrees with the statistical sample: {p}')
    return {'ra':np.mod(ra,360),'dec':dec,'z':z,'node_id':n.node_id.to_numpy()},p


def plot_redshift_distribution(nodes,edges,out,style):
    """Author's Gaussian KDE (no boundary renormalization), with explicit fallback."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    grid=np.linspace(edges[0],edges[-1],int(style['kde_samples']))
    histogram_edges=np.linspace(edges[0],edges[-1],int(style['histogram_bins'])+1)
    curves=[];histrows=[];countrows=[];metadata={}
    with style_context(style):
        fig,ax=plt.subplots(figsize=style['redshift_figsize'])
        for cat in CATS:
            z=np.asarray(nodes[cat]['z'],float)
            if not len(z):raise ValueError(f'Cannot draw an empty final catalogue: {cat}')
            used='gaussian_kde_scott';reason=None
            try:
                if len(z)<5 or np.ptp(z)==0:raise ValueError('Fewer than five nodes or zero redshift variance')
                density=gaussian_kde(z)(grid)
                ax.plot(grid,density,color=COLOR[cat],lw=3,label=LABEL[cat])
                curves += [{'catalog':cat,'z':float(x),'density':float(y),'method':used} for x,y in zip(grid,density)]
            except (ValueError,np.linalg.LinAlgError) as exc:
                used='histogram_density';reason=str(exc)
                ax.hist(z,bins=histogram_edges,density=True,histtype='step',color=COLOR[cat],lw=3,label=LABEL[cat])
            counts,_=np.histogram(z,bins=histogram_edges)
            dens=counts/(len(z)*np.diff(histogram_edges))
            histrows += [{'catalog':cat,'z_min':float(a),'z_max':float(b),'n_fans':int(c),'density':float(d)}
                         for a,b,c,d in zip(histogram_edges[:-1],histogram_edges[1:],counts,dens)]
            bid=redshift_bin_ids(z,edges)
            countrows += [{'catalog':cat,'zbin_id':i,'z_min':float(edges[i]),'z_max':float(edges[i+1]),
                           'n_fans':int(np.sum(bid==i)),'fraction':float(np.mean(bid==i))} for i in range(len(edges)-1)]
            metadata[cat]={'n_fans':len(z),'method':used,'fallback_reason':reason}
        for ze in edges[1:-1]:ax.axvline(ze,color='black',lw=.9,alpha=.85)
        ax.set(xlim=(edges[0],edges[-1]),xlabel=r'$z$',ylabel='Density',title='FANS redshift distribution')
        ax.grid(False,which='both')
        ax.tick_params(direction='in',top=True,right=True,length=7,width=1.4)
        for sp in ax.spines.values():sp.set_linewidth(FRAME_LINEWIDTH)
        leg=ax.legend(fontsize=style['font_size']-3,frameon=True,handlelength=1.8,handletextpad=.55)
        leg.get_frame().set_linewidth(1.1)
        path=out/'fans_redshift_distribution.pdf';save_publication_figure(fig,path,bbox_inches='tight',dpi=300);plt.close(fig)
    pd.DataFrame(curves,columns=['catalog','z','density','method']).to_csv(out/'fans_redshift_distribution_kde.csv',index=False)
    pd.DataFrame(histrows).to_csv(out/'fans_redshift_distribution_histogram.csv',index=False)
    pd.DataFrame(countrows).to_csv(out/'fans_redshift_bin_counts.csv',index=False)
    metadata.update({'edges':list(map(float,edges)),'unit':'one count per accepted FANS node',
                     'kde_note':'Default Scott bandwidth, Gaussian KDE normalized on the real line; displayed only inside analysis bounds, no boundary correction.'})
    (out/'fans_redshift_distribution_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    return [out/name for name in ['fans_redshift_distribution.pdf','fans_redshift_distribution_kde.csv',
            'fans_redshift_distribution_histogram.csv','fans_redshift_bin_counts.csv','fans_redshift_distribution_metadata.json']]


def _parameters(run,info):
    config_path=run/'driver_config.json'
    cfg=read_json(config_path)
    values=tuple(float(cfg[k]) for k in PARAM_KEYS)
    if int(cfg.get('n_arm_segments',3))!=3:raise ValueError(f'The paper grid requires N_seg=3: {run}')
    sig=info['state']['signature']
    for key,val in zip(('L','R','A'),values):
        if key in sig and not np.isclose(float(sig[key]),val):raise ValueError(f'Manifest/driver mismatch: {run}:{key}')
    return values,config_path


def collect_grid_tables(root,*,allow_partial=False):
    root=Path(root).expanduser().resolve()
    seen={};tables={k:[] for k in ['statistics','fractions','composition','trends']};deps=[];edges=None
    signatures=[]
    for run in sorted(root.iterdir()):
        if not run.is_dir() or not (run/'driver_config.json').exists():continue
        if not (run/'revision_projected_complete.json').exists():
            raise ValueError(f'Incomplete grid run: {run}; finish it or move it outside the grid root')
        info=validate_revision_run(run)
        pars,cp=_parameters(run,info)
        if pars not in EXPECTED_GRID:raise ValueError(f'Unexpected construction parameters {pars} in {run}')
        if pars in seen:raise ValueError(f'Duplicate grid parameters {pars}: {seen[pars]}, {run}')
        seen[pars]=str(run);deps+=info['dependencies']+[cp]
        signatures.append(info['state']['signature'])
        if edges is not None and not np.array_equal(edges,info['edges']):raise ValueError('Grid redshift edges differ')
        edges=info['edges']
        for cat in CATS:
            for kind in tables:
                p=run/'referee_multiplicity'/cat/f'multiplicity_{kind}.csv'
                t=pd.read_csv(p);t.insert(0,'catalog',cat);t.insert(0,'run_name',run.name)
                for k,v in zip(PARAM_KEYS,pars):t[k]=v
                tables[kind].append(t);deps.append(p)
    if not seen:raise ValueError(f'No completed revision grid found: {root}')
    missing=sorted(EXPECTED_GRID-set(seen))
    if missing and not allow_partial:raise ValueError(f'Expected 27 distinct configurations; found {len(seen)}. Missing {missing}. Use --allow-partial-grid only for diagnostics.')
    # Do not pool choices with different input catalogues/resampling settings.
    for key in ['source_sha256','inputs','B','null_B','nside','seed','edges','schema']:
        if any(s.get(key)!=signatures[0].get(key) for s in signatures[1:]):
            raise ValueError(f'Grid mixes incompatible {key}; do not combine these runs')
    return {k:pd.concat(v,ignore_index=True) for k,v in tables.items()},edges,{'n_runs':len(seen),'expected_runs':27,'missing':missing,'runs':seen,'dependencies':deps}


def standardize_statistics(table):
    d=table.copy()
    for old,new in [('H','H_proj'),('S','S_proj'),('H_err','H_proj_err'),('S_err','S_proj_err')]:d=d.rename(columns={old:new})
    for old,new in zip(PARAM_KEYS,('Lseg','Rlink','Aopen')):d=d.rename(columns={old:new})
    d['Nseg']=3;d['z_center_statistic']=d.get('z_center',np.nan)
    d['z_center']=.5*(d.z_min+d.z_max) # Keep author's midpoint abscissae for this panel.
    return d


def color_for_length(cat,length):
    base=np.asarray(to_rgb(COLOR[cat]))
    if np.isclose(length,7.5):base=(1-.67)*base+.67
    elif np.isclose(length,12.5):base=(1-.48)*base
    return to_hex(base)


def visible_limits(values,errors,requested,policy):
    vals=np.asarray(values,float);err=np.asarray(errors,float)
    # Missing errors are not imputed in any exported table; points still count
    # when deriving non-clipping limits.
    delta=np.where(np.isfinite(err),err,0.)
    lo=vals-delta;hi=vals+delta;lo=lo[np.isfinite(lo)];hi=hi[np.isfinite(hi)]
    if not len(lo):return tuple(requested) if requested is not None else (-1.,1.)
    a,b=float(lo.min()),float(hi.max());margin=.045*max(b-a,abs(a)*.05,abs(b)*.05,1e-5)
    auto=(a-margin,b+margin)
    if policy=='auto' or requested is None:return auto
    if policy=='strict':
        if a<requested[0] or b>requested[1]:raise ValueError(f'Figure limits {requested} would clip data/error bars [{a}, {b}]')
        return tuple(requested)
    return min(auto[0],requested[0]),max(auto[1],requested[1])



def grid_display_values(values, errors, metric, angle=False):
    """Plot transform only: the saved mean and each error endpoint separately.

    H and S use degrees(arcsin(x)); sign asymmetry uses percent. Undefined
    endpoints outside [-1,1] are not clipped to 90 degrees or substituted.
    """
    y=np.asarray(values,float);err=np.asarray(errors,float)
    if np.any(np.isfinite(err)&(err<0)):raise ValueError('Negative saved grid error')
    low=y-err;high=y+err
    if not angle:return y,low,high
    if metric=='A_sign':return 100*y,100*low,100*high
    if metric not in {'H_proj','S_proj'}:raise ValueError(f'Unknown angular grid metric: {metric}')
    def convert(v):
        with np.errstate(invalid='ignore'):
            return np.degrees(np.arcsin(np.where(np.abs(v)<=1,v,np.nan)))
    return convert(y),convert(low),convert(high)


def plot_grid_panel(grid,fid,edges,out,*,stratum='all',style=None,n_runs=27,angle=False):
    style=style or DEFAULT_STYLE;out=Path(out);fs=style['font_size']
    d=standardize_statistics(grid);d=d[(d.scope=='zbin')&(d.stratum==stratum)].copy()
    central=np.isclose(d.Lseg,10)&np.isclose(d.Rlink,10)&np.isclose(d.Aopen,35)
    high=standardize_statistics(fid) if fid is not None else d[central].copy()
    high=high[(high.scope=='zbin')&(high.stratum==stratum)].copy()
    if high.empty:raise ValueError('No fiducial or central configuration to highlight')
    if fid is not None and central.any():
        joined=d[central].merge(high,on=['catalog','zbin_id'],suffixes=('_grid','_fid'),validate='one_to_one')
        if len(joined)!=len(high) or len(joined)!=int(central.sum()):
            raise ValueError('Incomplete fiducial/central grid pairing')
        for key in ['H_proj','A_sign','S_proj','n_fans','n_arms']:
            if not np.allclose(joined[key+'_grid'],joined[key+'_fid'],rtol=1e-9,atol=1e-12,equal_nan=True):
                raise ValueError(f'Fiducial and central grid disagree for {stratum}:{key}; verify their inputs/selections')
    limits={};display_undefined={}
    angular_labels={'H_proj':r'$\arcsin(H_{\rm proj})\ [\mathrm{deg}]$',
                    'A_sign':r'$A_{\rm sign}\ [\%]$',
                    'S_proj':r'$\arcsin(\mathcal{S}_{\rm proj})\ [\mathrm{deg}]$'}
    for t in [d,high]:
        for _,metric,_ in METRICS:
            yy,lo,hi=grid_display_values(t[metric],t[metric+'_err'],metric,angle)
            t[metric+'_display']=yy;t[metric+'_display_low']=lo;t[metric+'_display_high']=hi
            t[metric+'_display_error_defined']=np.isfinite(yy)&np.isfinite(lo)&np.isfinite(hi)
    with style_context(style):
        fig,axes=plt.subplots(3,2,figsize=style['grid_figsize'],sharex=True,gridspec_kw={'wspace':.045,'hspace':.06})
        for row,(_,metric,ylabel) in enumerate(METRICS):
            coly=metric+'_display';collo=metric+'_display_low';colhi=metric+'_display_high'
            visible=np.r_[d[coly],high[coly],high[collo],high[colhi]]
            requested=style['grid_y_limits'].get(metric) if stratum=='all' else None
            policy=style['y_limit_policy']
            if stratum=='all' and angle:
                requested=style.get('grid_angular_y_limits',DEFAULT_STYLE['grid_angular_y_limits']).get(metric)
                policy=style.get('grid_angular_y_limit_policy','strict')
            lim=visible_limits(visible,np.zeros(len(visible)),requested,policy);limits[metric]=lim
            display_undefined[metric]=int((~high[metric+'_display_error_defined']).sum())
            for col,cat in enumerate(CATS):
                ax=axes[row,col];catdata=d[(d.catalog==cat)&(~central)]
                for _,g in catdata.groupby('run_name'):
                    g=g.sort_values('zbin_id');r=g.iloc[0]
                    ax.plot(g.z_center,g[coly],color=color_for_length(cat,r.Lseg),ls=LINE[r.Aopen],marker=MARKER[r.Rlink],lw=1.15,ms=3.9,alpha=.88,zorder=2)
                h=high[high.catalog==cat].sort_values('zbin_id')
                if len(h):
                    yy=h[coly].to_numpy(float);lo=h[collo].to_numpy(float);hi=h[colhi].to_numpy(float);xx=h.z_center.to_numpy(float)
                    ax.plot(xx,yy,ls='--',marker='s',color=COLOR[cat],lw=2.9,ms=6.6,mec='black',mew=.7,zorder=8)
                    ok=np.isfinite(yy)&np.isfinite(lo)&np.isfinite(hi)
                    if ok.any():ax.errorbar(xx[ok],yy[ok],yerr=np.vstack([yy[ok]-lo[ok],hi[ok]-yy[ok]]),fmt='none',ecolor=COLOR[cat],lw=2,capsize=3,zorder=8)
                if metric in {'H_proj','A_sign'}:ax.axhline(0,color='.15',lw=.85,alpha=.75,zorder=1)
                ax.set(xlim=(edges[0],edges[-1]),ylim=lim)
                ticks=[v for v in [.2,.3,.4,.5] if edges[0]<=v<=edges[-1]]
                ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]))
                ax.grid(False,which='both')
                for ze in edges[1:-1]:ax.axvline(ze,color='black',lw=.75,alpha=.95,zorder=1.5,gid='redshift-bin-edge')
                if col==0:ax.set_ylabel(angular_labels[metric] if angle else ylabel)
                else:ax.tick_params(labelleft=False)
                if row==0:ax.set_title('Carrón' if cat=='carron' else 'Chen')
                if row==2:ax.set_xlabel(r'$z$')
                else:ax.tick_params(labelbottom=False)
                ax.tick_params(direction='in',top=True,right=True,length=7,width=1.2)
        handlesL=[Line2D([],[],color=c,lw=3,label=rf'${L:g}\,h^{{-1}}{{\rm Mpc}}$') for L,c in [(7.5,'#d6d6d6'),(10.,'#7f7f7f'),(12.5,'#2b2b2b')]]
        handlesR=[Line2D([],[],color='.25',marker=m,lw=0,ms=7.5,label=rf'${R:g}\,h^{{-1}}{{\rm Mpc}}$') for R,m in MARKER.items()]
        handlesA=[Line2D([],[],color='.25',lw=3,ls=ls,label=rf'${a:g}^\circ$') for a,ls in LINE.items()]
        for handles,title,x0 in [(handlesL,r'Tone: $L_{\rm seg}$',.06),(handlesR,r'Marker: $R_{\rm link}$',.36),(handlesA,r'Line: $\Delta\phi_{\rm arm}$',.66)]:
            lax=fig.add_axes([x0,.045,.28,.105]);lax.axis('off')
            lax.legend(handles=handles,title=title,loc='center',frameon=True,fontsize=fs-4,title_fontsize=fs-3,borderpad=.75,handlelength=2.2,handletextpad=.8,labelspacing=.55)
        if stratum!='all' or n_runs!=27:
            title=(f'{stratum[1:]}-arm FANS' if stratum!='all' else 'All FANS')
            if n_runs!=27:title+=f' — PARTIAL GRID: {n_runs}/27'
            fig.suptitle(title,fontsize=fs,y=.994)
        fig.subplots_adjust(left=.115 if angle else .095,right=.985,top=.965,bottom=.21,wspace=.045,hspace=.06)
        save_publication_figure(fig,out,bbox_inches='tight',pad_inches=.32);plt.close(fig)
    used=pd.concat([d.assign(plot_role=np.where(central,'central_grid_reference','variation')),high.assign(plot_role='highlighted_fiducial' if fid is not None else 'highlighted_central_grid')],ignore_index=True)
    used.to_csv(out.with_suffix('.used_values.csv'),index=False)
    note={'n_runs':n_runs,'stratum':stratum,'y_limits':limits,'y_policy':style.get('grid_angular_y_limit_policy','strict') if angle and stratum=='all' else style['y_limit_policy'],
          'frame_linewidth_pt':FRAME_LINEWIDTH,
          'units':'H and S: degrees(arcsin(x)); A_sign: percent' if angle else 'dimensionless; A_sign is a fraction; H and S are not arcsin transformed',
          'angular_display':bool(angle),'grey_grid_drawn':False,'bin_boundaries_drawn':True,
          'abscissa':'midpoint of declared bin edges, as in supplied author panel',
          'error_source':'joint spatial block bootstrap; only fiducial error bars displayed',
          'error_transformation':'transform saved lower and upper endpoints separately; no derivative approximation or re-symmetrisation',
          'undefined_highlight_intervals':display_undefined,
          'support_policy':'same saved finite measurements as the existing grid plot; support flags retained in used_values, no scientific selection changed',
          'missing_error_note':'undefined errors remain NaN; angular endpoints outside [-1,1] not clipped to 90 degrees'}
    out.with_suffix('.metadata.json').write_text(json.dumps(note,indent=2,ensure_ascii=False))
    return [out,out.with_suffix('.used_values.csv'),out.with_suffix('.metadata.json')]


def plot_projected_sign_asymmetry(fid,edges,out,style):
    """Replace the legacy isolated analytic-error figure in the final collection."""
    rows=[]
    with style_context(style):
        fig,ax=plt.subplots(figsize=(9.,6.5))
        for cat in CATS:
            t=fid[(fid.catalog==cat)&(fid.stratum=='all')&(fid.scope=='zbin')].sort_values('zbin_id')
            x=t.z_center.to_numpy(float);y=100*t.A_sign.to_numpy(float);e=100*t.A_sign_err.to_numpy(float)
            ax.plot(x,y,marker='o',linestyle='--' if cat=='carron' else ':',color=COLOR[cat],label=LABEL[cat],lw=2)
            ok=np.isfinite(y)&np.isfinite(e)
            ax.errorbar(x[ok],y[ok],yerr=e[ok],fmt='none',ecolor=COLOR[cat],capsize=3)
            rows.append(t)
        ax.axhline(0,color='.25',lw=1)
        for z in edges[1:-1]:ax.axvline(z,color='black',lw=.7,alpha=.7)
        ax.set(xlabel=r'$z$',ylabel=r'$A_{\rm sign}$ [%]',xlim=(edges[0],edges[-1]))
        ax.legend(fontsize=style['font_size']-4);ax.grid(alpha=.25)
        p=Path(out)/'projected_sign_asymmetry_joint_bootstrap.pdf';save_publication_figure(fig,p,bbox_inches='tight');plt.close(fig)
    q=p.with_suffix('.used_values.csv');pd.concat(rows,ignore_index=True).to_csv(q,index=False)
    return [p,q]


class Bundle:
    """Build a managed, flat collection; manually added files are never removed."""
    def __init__(self,stage):
        self.stage=Path(stage);self.entries=[];self.deps={};self.status=[]
    def dependency(self,p):
        p=Path(p).resolve()
        if p.is_file() and str(p) not in self.deps:
            self.deps[str(p)]={'path':str(p),'sha256':file_digest(p),'size_bytes':p.stat().st_size}
    def add(self,p,*,name=None,role='support',description='',generated=False):
        p=Path(p);name=name or p.name
        if Path(name).name!=name:raise ValueError('Final names must be flat file names')
        if any(e['file']==name for e in self.entries):raise ValueError(f'Duplicate final filename: {name}')
        target=self.stage/name
        if p.resolve()!=target.resolve():shutil.copy2(p,target)
        if not generated:self.dependency(p)
        self.entries.append({'file':name,'role':role,'description':description,
             'source':'generated by build_paper_products.py' if generated else str(p.resolve()),
             'sha256':file_digest(target),'size_bytes':target.stat().st_size})
    def optional(self,p,**kwargs):
        p=Path(p)
        if p.exists():self.add(p,**kwargs);return True
        self.status.append({'product':p.name,'status':'not_available','source':str(p)})
        return False


def _collect_fiducial(bundle,run,*,style,footprints):
    info=validate_revision_run(run);edges=info['edges'];run=info['run']
    for p in info['dependencies']:bundle.dependency(p)
    pars,cp=_parameters(run,info)
    if pars!=(10.,10.,35.):raise ValueError(f'Expected adopted fiducial L=10,R=10,A=35; found {pars}')
    bundle.dependency(cp)
    data={};stats=[]
    for cat in CATS:
        data[cat],p=load_accepted_nodes(run,cat,edges);bundle.dependency(p)
        folder=run/'referee_multiplicity'/cat
        s=pd.read_csv(folder/'multiplicity_statistics.csv');s.insert(0,'catalog',cat)
        for k,v in zip(PARAM_KEYS,pars):s[k]=v
        s['run_name']=run.name;stats.append(s)
        # Exclude raw construction rows and resampling arrays from paper folder.
        for pattern in ['*.csv','*.json','*.pdf','*.png']:
            for p in sorted(folder.glob(pattern)):
                role='referee_figure' if p.suffix in {'.pdf','.png'} else 'referee_table'
                if p.name.endswith('metadata.json') or p.name=='sample_integrity.json':role='audit'
                name=p.name if p.name.startswith(cat+'_') else f'{cat}_{p.name}'
                bundle.add(p,name=name,role=role,description='Fiducial revised multiplicity/composition product')
    fid=pd.concat(stats,ignore_index=True)
    for p in plot_redshift_distribution(data,edges,bundle.stage,style):
        bundle.add(p,role='main_figure' if p.suffix=='.pdf' else 'figure_data',generated=True,description='Accepted FANS redshift distribution; one count per node')
    for p in plot_projected_sign_asymmetry(fid,edges,bundle.stage,style):
        bundle.add(p,role='support_figure' if p.suffix=='.pdf' else 'figure_data',generated=True,description='Joint spatial bootstrap, not legacy analytic counting error')
    if footprints:
        from .paper_footprints import build_footprints
        paths,meta=build_footprints(data,bundle.stage,nside=int(style['footprint_nside']),density_per_deg2=style['density_per_deg2'],font_size=style['font_size'])
        for p in paths:bundle.add(p,role='footprint_figure',generated=True,description='Author-calibrated Galactic N/S footprint crop')
        p=bundle.stage/'fans_footprints_metadata.json';p.write_text(json.dumps(meta,indent=2,ensure_ascii=False))
        bundle.add(p,role='audit',generated=True)
        bundle.status.extend({'product':'footprint','status':'empty_region','source':x} for x in meta['skipped'])
    else:bundle.status.append({'product':'footprint maps','status':'disabled_by_user'})
    # Rebuild compact combined tables from the validated per-catalogue tables.
    # Do not trust a stale paper_tables CSV from a different analysis invocation.
    for key,target in [('statistics','fiducial_multiplicity_statistics.csv'),('fractions','fiducial_multiplicity_fractions.csv'),
                       ('null_summary','fiducial_projected_significance_joint.csv'),('trends','fiducial_spirality_trends.csv')]:
        parts=[]
        for cat in CATS:
            source=run/'referee_multiplicity'/cat/f'multiplicity_{key}.csv'
            part=pd.read_csv(source);part.insert(0,'catalog',cat);parts.append(part);bundle.dependency(source)
        p=bundle.stage/target;pd.concat(parts,ignore_index=True).to_csv(p,index=False)
        bundle.add(p,role='paper_table',generated=True,description='Combined from the validated per-catalogue revision products')
    p=bundle.stage/'fiducial_accepted_fans_counts.csv'
    fid[fid.stratum.eq('all')][['catalog','scope','zbin_id','z_min','z_max','n_fans','n_arms','weight_sum','n_blocks']].to_csv(p,index=False)
    bundle.add(p,role='paper_table',generated=True,description='Accepted sample only; not a before-selection connectivity proxy')
    for filename in ['fans_count_summary.csv','fans_count_summary_allz_compact.csv','fans_status_summary_long.csv']:
        source=run/'paper_tables'/filename
        if source.exists():bundle.add(source,name='inventory_'+filename,role='audit',description='Legacy inventory; before_quality is not pre-reconstruction physical connectivity. Use multiplicity_selection_stages for the referee.')
    for stem in ['three_catalog_arms_n_segments_angle_pchiperr_extband','three_catalog_arms_n_segments_pchiperr_extband',
                 'three_catalog_arms_n_segments_shape_geometry_pchiperr_extband']:
        for suffix in ['.pdf','.png']:
            p=run/'three_catalog_plots'/(stem+suffix)
            if p.exists():bundle.add(p,role='support_figure',description='Revised arm-level projected or geometry panel')
        for p in sorted((run/'three_catalog_plots').glob(stem+'*.csv')):
            bundle.add(p,role='figure_data')
    return fid,info


def _collect_extrinsic(bundle,run,*,required=False):
    folder=Path(run)/'extrinsic_spirality'
    available=True
    for cat in CATS:
        sigpath=folder/f'{cat}_extrinsic_completed_signature.json'
        if not sigpath.exists():
            available=False;bundle.status.append({'product':f'{cat} extrinsic','status':'pending','source':str(sigpath)});continue
        sig=read_json(sigpath);d=catalog_directory(run,cat)
        if sig.get('analysis_schema')!='88.1' or sig.get('axis_objective')!='dipole' or sig.get('zbin_axis_mode')!='fixed_global':
            raise ValueError(f'Extrinsic outputs do not use the revised paper method: {sigpath}')
        for file,key in [('arm_level_spirality.csv','arm_sha256'),('node_spirality.csv','parent_sha256')]:
            if file_digest(d/file)!=sig.get(key):raise ValueError(f'Stale extrinsic results for {cat}; selected catalogue changed')
        for filename in [f'{cat}_extrinsic_bestfit.csv',f'{cat}_extrinsic_zbins.csv']:
            if not (folder/filename).exists():raise FileNotFoundError(f'Completed extrinsic signature but missing output: {folder/filename}')
        bundle.dependency(sigpath)
    if not available:
        if required:raise FileNotFoundError('Both completed fiducial extrinsic products are required')
        return
    for cat in CATS:
        for suffix in ['bestfit.csv','zbins.csv','bootstrap_summary.csv','axis_hpd_summary.csv','completed_signature.json',
                       'lookelsewhere_signflip_block_summary.csv','lookelsewhere_signflip_block_zbin_summary.csv',
                       'bootstrap_covariance_index.csv']:
            bundle.optional(folder/f'{cat}_extrinsic_{suffix}',role='extrinsic_table')
        for p in sorted(folder.glob(f'{cat}_extrinsic*map.pdf')):bundle.add(p,role='axis_figure')
    stem='extrinsic_arms_n_segments'
    for tail in ['_projected_vs_dipole_panel_angle','_projected_vs_dipole_panel','_bootstrap_axis_contours_combined']:
        found=False
        for ext in ['.pdf','.png']:
            p=folder/(stem+tail+ext)
            if p.exists():bundle.add(p,role='main_figure' if 'panel_angle' in tail else 'support_figure');found=True
        for suffix in ['.used_values.csv','_bingham_axis_summary.csv']:
            p=folder/(stem+tail+suffix)
            if p.exists():bundle.add(p,role='figure_data')
        if not found:bundle.status.append({'product':stem+tail,'status':'missing_figure'})
    for tail in ['_combined_bestfits.csv','_combined_zbins.csv','_bootstrap_axis_contours_combined_hpd_summary.csv']:
        bundle.optional(folder/(stem+tail),role='extrinsic_table')


def _collect_grid(bundle,root,fid,*,style,allow_partial):
    tables,edges,meta=collect_grid_tables(root,allow_partial=allow_partial)
    for p in meta['dependencies']:bundle.dependency(p)
    n=meta['n_runs'];prefix='grid27' if n==27 else f'grid_partial{n}'
    for key,t in tables.items():
        p=bundle.stage/f'{prefix}_{key}.csv';t.to_csv(p,index=False)
        bundle.add(p,role='robustness_table',generated=True)
    std=standardize_statistics(tables['statistics'])
    p=bundle.stage/f'{prefix}_projected_all_results.csv';std.to_csv(p,index=False)
    bundle.add(p,role='figure_data',generated=True,description='Fresh canonical grid values, not a cached legacy CSV')
    for stratum in ['all','m2','m3']:
        name=f'{prefix}_projected_panel'+('' if stratum=='all' else '_'+stratum)+'.pdf'
        for p in plot_grid_panel(tables['statistics'],fid,edges,bundle.stage/name,stratum=stratum,style=style,n_runs=n):
            bundle.add(p,role='main_figure' if stratum=='all' and p.suffix=='.pdf' else 'robustness_figure' if p.suffix=='.pdf' else 'figure_data',generated=True)
    t=tables['trends']
    desc=t.groupby(['catalog','stratum']).agg(n_runs=('run_name','nunique'),delta_min=('delta_S_last_minus_first','min'),
            delta_max=('delta_S_last_minus_first','max'),n_finite=('delta_S_last_minus_first','count'),
            n_positive=('delta_S_last_minus_first',lambda x:int(np.sum(x>0)))).reset_index()
    p=bundle.stage/f'{prefix}_trend_envelope_descriptive.csv';desc.to_csv(p,index=False);bundle.add(p,role='robustness_table',generated=True,
        description='Correlated construction choices, not independent detections')
    # Retain the author's optional centrality diagnostic but never use it to pick
    # the fiducial or add a significance. The adopted parameters stay 10/10/35.
    rows=[]
    d=std[(std.scope=='zbin')&(std.stratum=='all')]
    for (cat,b),g in d.groupby(['catalog','zbin_id']):
        for _,metric,_ in METRICS:
            val=g[metric].to_numpy(float);finite=np.isfinite(val)
            if not finite.any():continue
            med=np.median(val[finite]);scale=np.median(np.abs(val[finite]-med))
            if not np.isfinite(scale) or scale<=0:scale=np.std(val[finite])
            if not np.isfinite(scale) or scale<=0:scale=1.
            for name,score in zip(g.run_name,np.abs(val-med)/scale):rows.append({'run_name':name,'distance':score})
    if rows:
        central=pd.DataFrame(rows).groupby('run_name',as_index=False).distance.mean().rename(columns={'distance':'mean_median_distance'})
        central=central.merge(d[['run_name','Lseg','Rlink','Aopen']].drop_duplicates(),on='run_name',validate='one_to_one').sort_values('mean_median_distance')
        p=bundle.stage/f'{prefix}_centrality_scores_descriptive.csv';central.to_csv(p,index=False)
        bundle.add(p,role='audit',generated=True,description='Descriptive only; never changes the preselected fiducial')
    # Existing grid-summary plots are ancillary. Check their provenance manifest
    # rather than copying potentially stale curves from an earlier grid.
    old=Path(root)/'grid_summary';mp=old/'grid_manifest.json'
    if mp.exists() and read_json(mp).get('science_input_sha256')==grid_input_digest(meta['dependencies']):
        for p in sorted(old.glob('*.pdf'))+sorted(old.glob('*.png')):bundle.add(p,role='robustness_figure')
    else:bundle.status.append({'product':'individual grid-summary figures','status':'not_copied_without_matching_input_hash','source':str(old)})
    return meta


def grid_input_digest(paths):
    import hashlib
    h=hashlib.sha256()
    for p in sorted(set(map(Path,paths))):
        h.update(str(p.resolve()).encode());h.update(file_digest(p).encode())
    return h.hexdigest()


def _commit_bundle(bundle,outdir,source_settings,style,metadata):
    outdir=Path(outdir);previous=outdir/'manifest.json'
    prev=read_json(previous) if previous.exists() else {}
    managed={r['file']:r for r in prev.get('files',[])}
    special={'manifest.json','manifest.csv','paper_sources.json','README.md'}
    current={r['file'] for r in bundle.entries}
    # Never silently overwrite a manual addition or an edited managed plot.
    for name in set(managed)|current:
        p=outdir/name
        if p.exists() and name in managed and file_digest(p)!=managed[name]['sha256']:
            raise FileExistsError(f'Manually modified collected file: {p}. Move it to a new name before updating.')
        if p.exists() and name in current and name not in managed:
            raise FileExistsError(f'Unmanaged file would be overwritten: {p}')
    manifest={'bundle_version':'88.10','created_utc':datetime.now(timezone.utc).isoformat(),
              'science_schema':'88.1','sources':source_settings,'plot_style':style,
              'files':bundle.entries,'dependencies':list(bundle.deps.values()),'status':bundle.status,
              'metadata':metadata,'no_scientific_recomputation':True,
              'manual_fans_example_panel':'not generated, copied or deleted'}
    (bundle.stage/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str))
    pd.DataFrame(bundle.entries).to_csv(bundle.stage/'manifest.csv',index=False)
    (bundle.stage/'paper_sources.json').write_text(json.dumps(source_settings,indent=2))
    byrole={}
    for e in bundle.entries:byrole.setdefault(e['role'],[]).append(e['file'])
    text=['# Paper final plots and tables','',
          'This flat folder contains managed publication products from the revised FANS analysis.',
          'Figure numbers are not hard-coded: use filenames in LaTeX. The FANS example panel is managed by the author.',
          'One node per count in the redshift distribution; arm weighting remains unchanged in the winding estimators.',
          'Undefined bootstrap errors remain undefined. Partial grids are explicitly labelled and never called 27 runs.',
          'Files manually added under new names are preserved. Edit originals or use new names rather than editing managed copies.',
          'No raw FITS catalogues, bootstrap arrays, massive null tables, full galleries or legacy node-level estimators are copied.',
          '', '## Available products']
    for role,names in byrole.items():
        text+=['',f'### {role}']+[f'- `{n}`' for n in names]
    text+=['','## Pending, disabled or unavailable products']
    text += [f"- {x['product']}: {x['status']}" for x in bundle.status] or ['None recorded.']
    (bundle.stage/'README.md').write_text('\n'.join(text)+'\n')
    outdir.mkdir(parents=True,exist_ok=True)
    for name in set(managed)-current:
        p=outdir/name
        if p.exists():p.unlink()
    for name in current|special:
        shutil.copy2(bundle.stage/name,outdir/name)
    return manifest


@publication_frame_scope()
def build_paper_bundle(outdir,*,fiducial_run=None,grid_root=None,style_path=None,footprints=True,
                       allow_partial_grid=False,require_extrinsic=False,reset_sources=False):
    """Refresh generated/curated products without any scientific rerun.

    Stored source roots allow fiducial and grid launches to accumulate into the
    same folder. --reset-sources starts a new source selection explicitly.
    """
    outdir=Path(outdir).expanduser().resolve()
    prev=read_json(outdir/'paper_sources.json') if (outdir/'paper_sources.json').exists() and not reset_sources else {}
    fiducial_run=fiducial_run or prev.get('fiducial_run');grid_root=grid_root or prev.get('grid_root')
    if not fiducial_run and not grid_root:raise ValueError('Provide --fiducial-run or --grid-root')
    if require_extrinsic and not fiducial_run:raise ValueError('--require-extrinsic requires a fiducial run')
    fiducial_run=Path(fiducial_run).expanduser().resolve() if fiducial_run else None
    grid_root=Path(grid_root).expanduser().resolve() if grid_root else None
    # A new explicit fiducial must not silently retain a different source's grid.
    if prev.get('fiducial_run') and fiducial_run and Path(prev['fiducial_run'])!=fiducial_run and not reset_sources:
        raise ValueError('Changing the fiducial source requires --reset-sources or a new final folder')
    style_path=style_path or prev.get('style_path')
    style=load_style(style_path)
    outdir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.paper_products_build_',dir=outdir.parent) as tmp:
        bundle=Bundle(tmp);fid=None;metadata={}
        bundle.dependency(Path(__file__))
        bundle.dependency(Path(__file__).with_name('paper_footprints.py'))
        if style_path:bundle.dependency(style_path)
        if fiducial_run:
            fid,info=_collect_fiducial(bundle,fiducial_run,style=style,footprints=footprints)
            metadata['redshift_edges']=info['edges'].tolist()
            _collect_extrinsic(bundle,fiducial_run,required=require_extrinsic)
        else:bundle.status.append({'product':'fiducial redshift and extrinsic figures','status':'no_fiducial_source'})
        if grid_root:
            gridmeta=_collect_grid(bundle,grid_root,fid,style=style,allow_partial=allow_partial_grid)
            metadata['grid_n_runs']=gridmeta['n_runs'];metadata['grid_missing_parameters']=gridmeta['missing']
        else:bundle.status.append({'product':'27-configuration robustness panel','status':'pending_grid'})
        settings={'fiducial_run':str(fiducial_run) if fiducial_run else None,'grid_root':str(grid_root) if grid_root else None,
                  'style_path':str(Path(style_path).resolve()) if style_path else None}
        # Rendering only: use the collected calibrated tables, never repeat inference.
        from .publication_refresh import refresh_bundle_appearance
        appearance = refresh_bundle_appearance(bundle, style=style, fiducial_run=fiducial_run)
        metadata['appearance_update_88_7'] = appearance
        return _commit_bundle(bundle,outdir,settings,style,metadata)
