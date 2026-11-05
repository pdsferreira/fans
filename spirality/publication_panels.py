"""Native Matplotlib panels for the revised Letter (presentation only).

All plots are drawn from numeric products onto newly allocated, equal-aspect
Axes. No PDF/image composition is used. Missing individual redshifts never
trigger synthetic KDEs. Unsupported measurements are omitted from curves, not
from the science tables; disconnected support is never interpolated across.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, to_rgb
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde

from .publication_matrices import _bool, transition_counts, transition_upper_bound, _transition_boundaries, transition_display_spec
from .publication_redshift import CATS, COLOR, binned_multiplicity, prepare_redshift_cache, validate_redshift_cache
from .provenance import file_digest
from .publication_style import save_publication_figure, FRAME_LINEWIDTH
from .publication_matrices import hatch_zero_transition_cells

DEFAULTS = {
    'paper_panel_box_aspect': 0.55,
    'paper_results_width': 14.0,
    'paper_sample_width': 13.7,
    'paper_sample_box_aspect': 7.0/9.5,
    'paper_sample_font_scale': 1.5,
    'paper_sample_text_increment': 1.10,
    'paper_sample_heading_increment': 1.20,
    'paper_sample_linewidth_all': 3.0,
    'paper_sample_linewidth_conditioned': 2.5,
    'paper_sample_legend_gap_inches': 1.12,
    # Only fans_sample_multiplicity_panel uses these v88.9 adjustments.
    'paper_sample_panel_letter_increment': 1.20,
    'paper_sample_panel_legend_increment': 1.20,
    'paper_sample_panel_legend_extra_gap_inches': 0.14,
    'paper_sample_panel_fraction_increment': 1.045,  # v88.9 size multiplied by 0.95
    'paper_sample_panel_cells_match_ticks': True,
    'paper_sample_panel_density_upper_limit': 11.5,
    'paper_sample_panel_hide_density_zero': True,
    'paper_sample_panel_frame_increment': 1.40,
    'paper_panel_label_size': 18,
    'paper_panel_tick_size': 16,
    'paper_fraction_annotation_size': 11.5,
    'paper_fraction_decimals': 1,
    'paper_spirality_interval': 'bootstrap_standard_error',
    'paper_spirality_shared_limits': True,
    'paper_spirality_upper_limit_deg': 19.1,
    'paper_spirality_show_fixed_composition': False,
    'paper_results_show_bin_edges': False,
    'paper_results_legend_gap_inches': 0.80,
}
STRATA = ('all', 'm2', 'm3', 'fixed_composition')
POP_LABEL = {'all': 'All arms', 'm2': '2 arms', 'm3': '3 arms', 'fixed_composition': 'Fixed mixture (2+3)'}
# Catalogue is encoded by hue; population by stroke/marker and shade.
POP_LINE = {'all': '--', 'm2': '-', 'm3': ':', 'fixed_composition': '-.'}
POP_MARKER = {'all': 'o', 'm2': 's', 'm3': '^', 'fixed_composition': 'D'}
DIST_LINE = {'all': '-', 'm2': '--', 'm3': ':'}


def _settings(style: Mapping | None) -> dict:
    cfg = {**DEFAULTS, **dict(style or {})}
    q = float(cfg['paper_panel_box_aspect'])
    if not np.isfinite(q) or q <= 0:
        raise ValueError('paper_panel_box_aspect must be positive')
    if cfg['paper_spirality_interval'] not in {'bootstrap_standard_error', 'percentile68'}:
        raise ValueError('paper_spirality_interval must be bootstrap_standard_error or percentile68')
    upper = float(cfg['paper_spirality_upper_limit_deg'])
    if not np.isfinite(upper) or not 0 < upper <= 90:
        raise ValueError('paper_spirality_upper_limit_deg must be in (0, 90]')
    for name in ('paper_spirality_show_fixed_composition', 'paper_results_show_bin_edges'):
        if not isinstance(cfg[name], (bool, np.bool_)):
            raise ValueError(f'{name} must be a boolean')
    gap = float(cfg['paper_results_legend_gap_inches'])
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError('paper_results_legend_gap_inches must be positive')
    return cfg


@contextmanager
def panel_style(style):
    """Same font, tick, line and fill language as the projected/extrinsic panel."""
    from .extrinsic_spirality import setup_matplotlib
    cfg = _settings(style)
    with plt.rc_context():
        setup_matplotlib()
        plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
            'mathtext.fontset': 'stix', 'axes.labelsize': cfg['paper_panel_label_size'],
            'xtick.labelsize': cfg['paper_panel_tick_size'], 'ytick.labelsize': cfg['paper_panel_tick_size']})
        yield


def equal_axes_grid(rows, cols, *, width, aspect, bottom, left=1.15, right=.45,
                    top=.60, hgap=.38, wgap=1.05):
    """Allocate equal physical plot boxes; colourbars never resize data Axes."""
    aw = (float(width)-left-right-(cols-1)*wgap)/cols
    if aw <= 1 or rows < 1 or cols < 1:
        raise ValueError('Figure geometry leaves no usable data area')
    ah = aw*float(aspect)
    height = top+bottom+rows*ah+(rows-1)*hgap
    fig = plt.figure(figsize=(width,height))
    axes = np.empty((rows,cols),dtype=object)
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_axes([(left+c*(aw+wgap))/width,
                (height-top-(r+1)*ah-r*hgap)/height, aw/width, ah/height])
            ax.set_box_aspect(aspect)
            axes[r,c] = ax
    return fig, axes


def axes_geometry(fig, axes, *, aspect):
    fig.canvas.draw()
    boxes=[]
    for ax in np.ravel(axes):
        box=ax.get_window_extent()
        boxes.append(dict(width_px=float(box.width),height_px=float(box.height),
                          height_over_width=float(box.height/box.width)))
    if not np.allclose([b['height_over_width'] for b in boxes],aspect,rtol=0,atol=1e-9):
        raise RuntimeError('Publication plot boxes have inconsistent aspect ratios')
    if not np.allclose([b['width_px'] for b in boxes],boxes[0]['width_px'],atol=1e-7,rtol=0):
        raise RuntimeError('A colourbar/legend resized a publication data panel')
    return boxes


def _save(fig, out, values, metadata, axes, cfg):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    geometry=axes_geometry(fig,axes,aspect=float(cfg['paper_panel_box_aspect']))
    linewidth=float(cfg.get('_paper_frame_linewidth', FRAME_LINEWIDTH))
    if '_paper_frame_linewidth' in cfg:
        # Explicit figure-local override survives the outer legacy save scope.
        # Other figures retain the common 1.1 pt frame.
        fig._publication_frame_override=linewidth
    metadata={**metadata,'native_matplotlib_axes':True,'pdf_or_image_compositing':False,
        'no_scientific_inference_recomputed':True,'box_aspect':float(cfg['paper_panel_box_aspect']),
        'data_axes_geometry':geometry,'figure_size_inches':fig.get_size_inches().tolist(),
        'frame_linewidth_pt':linewidth}
    save_publication_figure(fig,out,bbox_inches='tight',pad_inches=.26,dpi=300)
    plt.close(fig)
    vp=out.with_suffix('.used_values.csv');values.to_csv(vp,index=False)
    mp=out.with_suffix('.metadata.json');mp.write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    return [out,vp,mp]


def angular_interval(value, lo, hi):
    """Transform the actual endpoints; never symmetrise or clip to 90 deg."""
    x,a,b=np.broadcast_arrays(np.asarray(value,float),np.asarray(lo,float),np.asarray(hi,float))
    if np.any(np.isfinite(a)&np.isfinite(b)&(a>b)):
        raise ValueError('Reversed saved uncertainty interval')
    def angle(v):
        with np.errstate(invalid='ignore'):
            return np.degrees(np.arcsin(np.where(np.abs(v)<=1,v,np.nan)))
    return angle(x),angle(a),angle(b)


def spirality_display_rows(folder, catalog, style=None):
    """Prepare all saved rows, with an explicit plotted/omitted flag."""
    cfg=_settings(style);folder=Path(folder)
    source=folder/f'{catalog}_multiplicity_statistics.csv'
    d=pd.read_csv(source)
    required={'stratum','scope','zbin_id','z_min','z_max','z_center','S','S_err','precision_support','n_fans','n_blocks'}
    if not required.issubset(d):raise ValueError(f'Missing spirality columns: {required-set(d)}')
    keep=d[(d.scope=='zbin') & d.stratum.isin(['all','m2','m3'])].copy()
    if keep.duplicated(['stratum','zbin_id']).any():raise ValueError('Duplicate conditioned spirality bins')
    if keep[keep.stratum.eq('all')].sort_values('zbin_id').zbin_id.tolist()!=[0,1,2,3]:
        raise ValueError('Four unique full-sample bins are required')
    rows=[]
    for _,r in keep.iterrows():
        rows.append(dict(catalog=catalog,stratum=str(r.stratum),zbin_id=int(r.zbin_id),
            z_min=float(r.z_min),z_max=float(r.z_max),z_center=float(r.z_center),S=float(r.S),S_err=float(r.S_err),
            S_q16=float(r.get('S_q16',np.nan)),S_q84=float(r.get('S_q84',np.nan)),
            n_fans=int(r.n_fans),n_blocks=int(r.n_blocks),precision_support=_bool(r.precision_support),
            x_definition='saved mean arm redshift',source_file=source.name))
    deps=[source]
    cp=folder/f'{catalog}_multiplicity_composition.csv'
    if cp.is_file():
        comp=pd.read_csv(cp);deps.append(cp)
        for _,r in comp.iterrows():
            rows.append(dict(catalog=catalog,stratum='fixed_composition',zbin_id=int(r.zbin_id),
                z_min=float(r.z_min),z_max=float(r.z_max),z_center=float(r.z_center),
                S=float(r.S_fixed_composition),S_err=float(r.S_fixed_composition_err),
                S_q16=float(r.get('S_fixed_composition_q16',np.nan)),S_q84=float(r.get('S_fixed_composition_q84',np.nan)),
                n_fans=np.nan,n_blocks=np.nan,precision_support=_bool(r.supported),
                x_definition='saved geometric bin centre for the standardized population',source_file=cp.name))
    out=pd.DataFrame(rows).sort_values(['stratum','zbin_id']).reset_index(drop=True)
    if cfg['paper_spirality_interval']=='bootstrap_standard_error':
        out['S_low']=out.S-out.S_err;out['S_high']=out.S+out.S_err
    else:
        out['S_low']=out.S_q16;out['S_high']=out.S_q84
    y,lo,hi=angular_interval(out.S,out.S_low,out.S_high)
    out['theta_deg']=y;out['theta_low_deg']=lo;out['theta_high_deg']=hi
    out['supported_interval']=out.precision_support & np.isfinite(y)&np.isfinite(lo)&np.isfinite(hi)
    out['display_requested']=~out.stratum.eq('fixed_composition') | bool(cfg['paper_spirality_show_fixed_composition'])
    out['plot_support']=out.supported_interval & out.display_requested
    out['omission_reason']=np.where(~out.display_requested,'not_requested_for_figure',
        np.where(~out.precision_support,'insufficient_saved_support',
        np.where(~np.isfinite(y)|~np.isfinite(lo)|~np.isfinite(hi),'undefined_or_outside_arcsin_domain','')))
    out['display_mode']=np.where(~out.plot_support,'omitted',
        np.where(out.stratum.eq('all'),'curve_filled_band_markers_errorbars','markers_errorbars_only'))
    out['interval_definition']=cfg['paper_spirality_interval']
    # Check bins, not pointwise x positions: each stratum has its own saved mean z.
    bb=out[out.stratum.eq('all')].set_index('zbin_id')
    for _,r in out.iterrows():
        if r.zbin_id not in bb.index or not np.allclose([r.z_min,r.z_max],bb.loc[r.zbin_id,['z_min','z_max']].to_numpy(float),atol=1e-12,rtol=0):
            raise ValueError('Conditioned curves use different redshift bins')
    return out,deps


def _groups(d):
    """Consecutive *supported* bins only, preserving holes in support."""
    supported=d[d.plot_support].sort_values('zbin_id')
    if supported.empty:return []
    cut=np.flatnonzero(np.diff(supported.zbin_id.to_numpy(int))!=1)+1
    limits=np.r_[0,cut,len(supported)]
    return [supported.iloc[a:b] for a,b in zip(limits[:-1],limits[1:]) if b>a]


def _pop_colour(cat,pop):
    c=np.asarray(to_rgb(COLOR[cat]))
    if pop=='m2':return tuple(.67*c)
    if pop=='m3':return tuple(.72*c+.28)
    if pop=='fixed_composition':return '.20'
    return COLOR[cat]


def _interpolated(x,y,first,last,n=260):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)==1:return np.array([first,last]),np.repeat(y[0],2)
    xx=np.r_[first,x,last];yy=np.r_[y[0],y,y[-1]]
    xx,idx=np.unique(xx,return_index=True);yy=yy[idx]
    grid=np.unique(np.r_[np.linspace(first,last,n),x])
    return grid,PchipInterpolator(xx,yy,extrapolate=False)(grid)


def _finish_z_axes(ax, edges, *, xlabel=True, show_bin_edges=True, show_grid=True):
    ax.set_xlim(edges[0],edges[-1]);ax.set_xticks([.1,.2,.3,.4,.5,.6])
    ax.grid(False,which='both')
    if show_grid:ax.grid(alpha=.22)
    ax.tick_params(axis='both',which='major',direction='in',top=True,right=True,length=5.5,width=1.1)
    if show_bin_edges:
        for v in edges[1:-1]:ax.axvline(v,color='black',lw=.65,alpha=.42,zorder=0)
    if xlabel:ax.set_xlabel(r'redshift $z$')
    else:ax.tick_params(labelbottom=False)


def draw_spirality_axis(ax, rows, catalog, cfg, *, show_legend=True):
    from .extrinsic_spirality import _add_left_total_marker
    handles=[]
    for pop in STRATA:
        d=rows[rows.stratum.eq(pop)]
        if not d.plot_support.any():continue
        colour=_pop_colour(catalog,pop)
        for g in _groups(d):
            x=g.z_center.to_numpy(float);y=g.theta_deg.to_numpy(float)
            lo=g.theta_low_deg.to_numpy(float);hi=g.theta_high_deg.to_numpy(float)
            # Only the unrestricted sample is connected/interpolated.  The
            # conditioned samples retain their exact point and endpoint bars,
            # without filled areas or connecting curves.
            if pop == 'all':
                first,last=float(g.z_min.iloc[0]),float(g.z_max.iloc[-1])
                xb,low=_interpolated(x,lo,first,last)
                _,high=_interpolated(x,hi,first,last)
                if np.any(low>high):
                    _,mid=_interpolated(x,(lo+hi)/2,first,last)
                    _,half=_interpolated(x,(hi-lo)/2,first,last)
                    low,high=mid-half,mid+half
                ax.fill_between(xb,low,high,color=colour,alpha=.22,lw=0,zorder=2,
                                gid='spirality-band-all')
                ax.plot(xb,low,color=colour,alpha=.65,lw=.75,zorder=2.1,
                        gid='spirality-boundary-all')
                ax.plot(xb,high,color=colour,alpha=.65,lw=.75,zorder=2.1,
                        gid='spirality-boundary-all')
                if len(x)>=2:
                    xx,yy=_interpolated(x,y,first,last);valid=(xx>=x[0])&(xx<=x[-1])
                    ax.plot(xx[valid],yy[valid],color=colour,ls=POP_LINE[pop],lw=1.45,zorder=4,
                            gid='spirality-curve-all')
            # Exact endpoint bars; no assumption that a percentile interval is
            # symmetric around (or even contains) its original point estimate.
            for _,r in g.iterrows():
                ax.plot([r.z_center,r.z_center],[r.theta_low_deg,r.theta_high_deg],color=colour,lw=1.0,alpha=.85,zorder=4.2,
                        gid=f'spirality-yerr-{pop}')
                ax.plot([r.z_min,r.z_max],[r.theta_deg,r.theta_deg],color=colour,lw=.85,alpha=.47,zorder=3,
                        gid=f'spirality-xerr-{pop}')
            ax.plot(x,y,ls='none',marker=POP_MARKER[pop],color=colour,ms=6.6,zorder=5,
                    gid=f'spirality-markers-{pop}')
        handles.append(Line2D([0],[0],color=colour,ls=POP_LINE[pop] if pop=='all' else 'none',marker=POP_MARKER[pop],lw=1.6,ms=6,
            label='All multiplicities' if pop=='all' else POP_LABEL[pop]))
    bb=rows[rows.stratum.eq('all')].sort_values('zbin_id')
    edges=np.r_[bb.z_min,float(bb.z_max.iloc[-1])]
    _finish_z_axes(ax,edges,show_bin_edges=bool(cfg['paper_results_show_bin_edges']))
    ax.set_ylabel(r'$\sin^{-1}\mathcal{S}_{\rm proj}$ [deg]')
    ax.text(.5,.96,f"Spirality — {'Carrón' if catalog=='carron' else 'Chen'}",transform=ax.transAxes,
        ha='center',va='top',fontsize=14,bbox=dict(fc='white',ec='none',alpha=.8,pad=1))
    if show_legend:
        ax.legend(handles=handles,loc='lower right',fontsize=10.8,ncol=2,frameon=True,
            facecolor='white',framealpha=.9,handlelength=2.5,columnspacing=1.1)
    return handles


def _spirality_limits(axes, cfg):
    ranges=[]
    requested=float(cfg['paper_spirality_upper_limit_deg'])
    for ax in axes:
        ax.relim();ax.autoscale(enable=True,axis='y');ax.margins(y=.10)
        # relim ignores collections, but explicit boundary lines retain all
        # uncertainty endpoints. No unsupported point contributes to these limits.
        low,automatic_high=ax.get_ylim()
        # Use the requested ceiling exactly when all supported intervals fit.
        # Expand only for actual data beyond it, not for autoscale padding alone.
        data_high=float(ax.dataLim.ymax)
        high=requested if np.isfinite(data_high) and data_high<=requested else max(requested,automatic_high)
        ranges.append([low,high])
    if cfg['paper_spirality_shared_limits'] and len(axes)>1:
        common=[min(x[0] for x in ranges),max(x[1] for x in ranges)]
        ranges=[common]*len(axes)
    for ax,lim in zip(axes,ranges):ax.set_ylim(*lim)
    return ranges


def _omitted_note(rows):
    bad=rows[~rows.plot_support & rows.stratum.isin(['m2','m3'])]
    if bad.empty:return ''
    parts=[]
    for _,r in bad.iterrows():
        parts.append(f"{r.catalog.title()}, {r.stratum[1:]} arms, {r.z_min:g}<z<{r.z_max:g} (N={int(r.n_fans)})")
    return 'Not drawn: '+ '; '.join(parts)+'. Insufficient saved support.'



def _spirality_caption(rows, cfg):
    from .publication_captions import spirality_caption
    return spirality_caption(rows, cfg)


def _results_caption(rows, cfg, axis_lines):
    from .publication_captions import results_caption
    return results_caption(rows, cfg, axis_lines)


def plot_spirality_standalone(folder, cfg):
    folder=Path(folder);paths=[];deps=[];datasets={}
    for cat in CATS:
        datasets[cat],dd=spirality_display_rows(folder,cat,cfg);deps+=dd
    # Establish common y range from native rendered data, for both individual PDFs.
    with panel_style(cfg):
        figs=[];axs=[]
        for cat in CATS:
            fig,grid=equal_axes_grid(1,1,width=8.4,aspect=cfg['paper_panel_box_aspect'],bottom=1.0,left=1.05,right=.30,top=.40)
            draw_spirality_axis(grid[0,0],datasets[cat],cat,cfg);figs.append(fig);axs.append(grid[0,0])
        limits=_spirality_limits(axs,cfg)
        for cat,fig,ax,lim in zip(CATS,figs,axs,limits):
            note=_omitted_note(datasets[cat])
            paths+=_save(fig,folder/f'{cat}_S_fixed_multiplicity.pdf',datasets[cat],{
                'catalog':cat,'unit':'degrees','value':'arcsin(S_proj), not mean of arm-level angles',
                'interval':cfg['paper_spirality_interval'],'endpoint_transform':'arcsin of each endpoint; no symmetrisation',
                'unsupported':'omitted from plot, retained in original scientific tables and used_values',
                'omission_caption':note,'y_limits':lim,'support_gaps':'no unsupported point is drawn',
                'fixed_composition_shown':bool(cfg['paper_spirality_show_fixed_composition']),
                'filled_band_populations':['all'],'connected_populations':['all'],
                'bin_boundaries_drawn':bool(cfg['paper_results_show_bin_edges']),
                'omission_text_drawn':False},[ax],cfg)
            cp=folder/f'{cat}_S_fixed_multiplicity.caption.txt'
            cp.write_text(_spirality_caption(datasets[cat],cfg),encoding='utf-8');paths.append(cp)
    return paths,deps,datasets


def draw_parity_axes(axes, datasets, prepared, cfg):
    """Reuse native curve/null drawing helpers, with no analysis fallbacks."""
    from . import extrinsic_spirality as e
    from . import pipeline as p
    pH,eH,pA,eA=axes[0,0],axes[0,1],axes[1,0],axes[1,1]
    for cat,label,zext,best in datasets:
        d=prepared['projected'][cat].copy()
        d=e._add_angle_value_error_columns(d,'H_S_weighted_mean','H_S_bootstrap_err','H_angle','H_angle_err')
        p._plot_block_signflip_null_background_pchip(pH,d,key=cat,statistic='H',angle=True,
            extend_band_to_bin_edges=True,cached_null=prepared['nulls'][cat]['H'])
        p._plot_block_signflip_null_background_pchip(pA,d,key=cat,statistic='sign_asymmetry',angle=False,
            extend_band_to_bin_edges=True,cached_null=prepared['nulls'][cat]['A_sign'])
        e.plot_one_series_pchip(pH,d,cat,label,'H_angle','H_angle_err',angle=True,extband=True)
        e.plot_one_series_pchip(pA,d,cat,label,'sign_asymmetry_percent','sign_asymmetry_percent_err',extband=True)
        z=e._add_angle_value_error_columns(zext,'H_ext','H_ext_err','H_angle','H_angle_err')
        e._draw_extrinsic_lookelsewhere_lines(eH,z,cat,panel='H',angle=True,extband=True)
        e._draw_extrinsic_lookelsewhere_lines(eA,z,cat,panel='sign',angle=False,extband=True)
        e.plot_one_series_pchip(eH,z,cat,label,'H_angle','H_angle_err',angle=True,extband=True)
        e.plot_one_series_pchip(eA,z,cat,label,'sign_asym_ext_percent','sign_asym_ext_percent_err',extband=True)
    names=('Handedness','Dipole handedness','CCW/CW sign asymmetry','Axial CCW/CW sign asymmetry')
    labels=(r'$\sin^{-1}H_{\rm proj}$ [deg]',r'$\sin^{-1}A_{\rm dip}$ [deg]',
        r'$(N_{\rm CCW}-N_{\rm CW})/(N_{\rm CCW}+N_{\rm CW})$ [%]',
        r'$(N_{\rm CCW}^{\rm ax}-N_{\rm CW}^{\rm ax})/(N_{\rm CCW}^{\rm ax}+N_{\rm CW}^{\rm ax})$ [%]')
    bins=prepared['projected'][CATS[0]].sort_values('zbin_id');edges=np.r_[bins.z_min,bins.z_max.iloc[-1]]
    for ax,name,label in zip((pH,eH,pA,eA),names,labels):
        ax.axhline(0,color='.25',lw=1.0);ax.set_ylabel(label)
        _finish_z_axes(ax,edges,xlabel=False,show_bin_edges=bool(cfg['paper_results_show_bin_edges']))
        ax.text(.50,.95,name,transform=ax.transAxes,ha='center',va='top',fontsize=14,
            bbox=dict(boxstyle='round,pad=.14',fc='white',ec='none',alpha=.72))
    pH.set_title('Projected',fontsize=16,fontweight='bold');eH.set_title('Dipolar extrinsic',fontsize=16,fontweight='bold')
    e._apply_paper_angle_panel_limits(angle=True,ax_pH=pH,ax_eH=eH,ax_pA=pA,ax_eA=eA,
        upper_limits=cfg.get('panel_angular_upper_limits'))
    for cat,label,zext,best in datasets:
        e._add_left_total_marker(pH,cat,prepared['totals'][cat]['H']['value'])
        e._add_left_total_marker(pA,cat,prepared['totals'][cat]['A']['value'])
        e._add_global_total_markers(eH,best,cat,panel='H',angle=True)
        e._add_global_total_markers(eA,best,cat,panel='sign',angle=False)
    e._add_sigma_annotation_multi(pH,prepared['sigma_items']['H'])
    e._add_sigma_annotation_multi(pA,prepared['sigma_items']['A_sign'])
    e._add_sigma_annotation_multi(eH,e._sigmas_from_best_pvalues_by_catalog(datasets,'p_global_lookelsewhere',one_sided=True))
    e._add_sigma_annotation_multi(eA,e._sigmas_from_best_pvalues_by_catalog(datasets,'p_sign_asym_global_lookelsewhere',one_sided=False))
    return {key:list(ax.get_ylim()) for key,ax in zip(('H_proj','A_dip','A_sign','A_ax'),(pH,eH,pA,eA))}


def plot_results_panel(folder, style, *, datasets=None, prepared=None, fiducial_run=None):
    from .publication_refresh import prepare_cached_panel
    from . import extrinsic_spirality as ext
    folder=Path(folder);cfg=_settings(style);deps=[]
    if datasets is None or prepared is None:
        datasets,prepared,dd,_=prepare_cached_panel(folder,fiducial_run=fiducial_run);deps+=dd
    records={}
    for cat in CATS:records[cat],dd=spirality_display_rows(folder,cat,cfg);deps+=dd
    with panel_style(cfg):
        fig,axes=equal_axes_grid(3,2,width=cfg['paper_results_width'],aspect=cfg['paper_panel_box_aspect'],
            bottom=1.30,hgap=.22,top=.65)
        ylimits=draw_parity_axes(axes,datasets,prepared,cfg)
        for ax,cat in zip(axes[2],CATS):
            draw_spirality_axis(ax,records[cat],cat,cfg)
            ext._add_left_total_marker(ax,cat,prepared['totals'][cat]['S']['value'])
        slims=_spirality_limits(axes[2],cfg)
        for ax,letter in zip(axes.flat,'abcdef'):
            ax.text(.022,.958,f'({letter})',transform=ax.transAxes,ha='left',va='top',fontsize=12,
                    bbox=dict(fc='white',ec='none',alpha=.7,pad=.2))
        handles=[Line2D([0],[0],color=COLOR[c],ls=ext._catalog_linestyle(c),marker='o',ms=5,lw=1.5,
                   label='Carrón' if c=='carron' else 'Chen') for c in CATS]
        handles += [Line2D([0],[0],color='black',lw=1.25,alpha=.43,label='Null 1σ'),
            Line2D([0],[0],color='black',lw=1.85,alpha=.78,label='Null 2σ'),
            Line2D([0],[0],color='.3',ls='none',marker='>',label='Full range')]
        height=fig.get_figheight()
        legend_y=(1.30-float(cfg['paper_results_legend_gap_inches']))/height
        fig.legend(handles=handles,loc='center',bbox_to_anchor=(.54,legend_y),ncol=5,frameon=False,fontsize=12.2,handlelength=2.5)
        axis_lines=[ext._format_axis_95ci_line(label,best) for _,label,_,best in datasets]
        omitted=' '.join(_omitted_note(x) for x in records.values()).strip()
        part=prepared['used_values'].copy();part['record_type']='parity_panel_source'
        sr=pd.concat(records.values(),ignore_index=True);sr['record_type']='spirality_display'
        values=pd.concat([part[part.panel.ne('S_proj')],sr],ignore_index=True)
        paths=_save(fig,folder/'fiducial_results_multiplicity_panel_angle.pdf',values,{
            'layout':'3x2; four parity panels followed by Carron and Chen spirality',
            'parity_curves':'same native drawing functions and exact saved inputs as the previous angular panel',
            'parity_y_limits':ylimits,'spirality_y_limits':dict(zip(CATS,slims)),
            'spirality_interval':cfg['paper_spirality_interval'],
            'spirality_endpoints':'arcsin(S +/- saved bootstrap standard error) without symmetrisation' if cfg['paper_spirality_interval']=='bootstrap_standard_error' else 'arcsin of stored 16th/84th bootstrap percentiles',
            'omission_caption':omitted,'axis_legend':axis_lines,
            'omission_text_drawn':False,'axis_summary_text_drawn':False,
            'fixed_composition_shown':bool(cfg['paper_spirality_show_fixed_composition']),
            'filled_band_populations':['all'],'connected_populations':['all'],
            'bin_boundaries_drawn':bool(cfg['paper_results_show_bin_edges']),
            'legend_gap_inches':float(cfg['paper_results_legend_gap_inches']),
            'fullrange_values_legend_preserved_in_metadata':prepared['fullrange_legend_text'],
            'scientific_tables_unchanged':True},axes,cfg)
        cp=folder/'fiducial_results_multiplicity_panel_angle.caption.txt'
        cp.write_text(_results_caption(sr,cfg,axis_lines),encoding='utf-8');paths.append(cp)
    return paths,deps


def fraction_annotations(binned, decimals=1):
    """Numbers are binwise P(m|z), never ratios of independently normalised KDEs."""
    rows=[]
    for cat in CATS:
        for b in range(4):
            g=binned[(binned.catalog==cat)&(binned.zbin_id==b)]
            total=int(g.n_total_fans.iloc[0]);vals={m:int(g.loc[g.multiplicity.eq(m),'n_fans'].sum()) for m in (2,3)}
            p2,p3=100*vals[2]/total,100*vals[3]/total
            rows.append(dict(catalog=cat,zbin_id=b,z_min=float(g.z_min.iloc[0]),z_max=float(g.z_max.iloc[0]),
                n_total_fans=total,n_m2=vals[2],n_m3=vals[3],n_other=total-vals[2]-vals[3],fraction_m2_percent=p2,
                fraction_m3_percent=p3,text=f'2A:{p2:.{decimals}f}%/3A:{p3:.{decimals}f}%',
                denominator='all accepted FANS in this catalogue/bin, including multiplicities >=4'))
    return pd.DataFrame(rows)


def distribution_curves(nodes,binned,edges,style):
    """Six actual Gaussian KDEs: total, m=2 and m=3, independently normalized."""
    validate_redshift_cache(nodes,binned,edges)
    grid=np.linspace(edges[0],edges[-1],int(style.get('kde_samples',700)))
    result=[]
    for cat in CATS:
        for pop in ('all','m2','m3'):
            part=nodes[nodes.catalog.eq(cat)]
            if pop!='all':part=part[part.multiplicity.eq(int(pop[1:]))]
            z=part.z.to_numpy(float)
            if len(z)<5 or np.ptp(z)==0:
                raise ValueError(f'Insufficient individual redshifts for a KDE: {cat}/{pop}; no smoothed substitute is invented')
            y=gaussian_kde(z)(grid)
            result.extend(dict(catalog=cat,stratum=pop,z=float(x),density=float(v),n_fans=len(z),
                               method='gaussian_kde_scott') for x,v in zip(grid,y))
    return pd.DataFrame(result)


def draw_distribution_axis(ax,curves,annotations,edges,cfg):
    for cat in CATS:
        for pop in ('all','m2','m3'):
            d=curves[(curves.catalog==cat)&(curves.stratum==pop)]
            ax.plot(d.z,d.density,color=COLOR[cat],ls=DIST_LINE[pop],lw=float(cfg['paper_sample_linewidth_all'] if pop=='all' else cfg['paper_sample_linewidth_conditioned']))
    _finish_z_axes(ax,edges,show_grid=False);ax.set_xlabel(r'$z$');ax.set_ylabel('Density')
    ymax=cfg.get('_density_upper_limit')
    if ymax is None:
        ymax=float(curves.density.max())/.72
    else:
        ymax=float(ymax)
        if not np.isfinite(ymax) or ymax<=0:
            raise ValueError('Density upper limit must be finite and positive')
        # Respect the requested range; explicitly record/warn about any clipping.
        n_above=int((curves.density>ymax).sum())
        if n_above:
            import warnings
            warnings.warn(f'{n_above} KDE grid values exceed the requested sample-panel '
                          f'density upper limit {ymax:g}; see plot metadata.', RuntimeWarning)
    ax.set_ylim(0,ymax)
    if cfg.get('_hide_density_zero',False):
        from matplotlib.ticker import FuncFormatter
        # Hide the numeral, not the origin, data, or other tick labels.
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value,position: '' if np.isclose(value,0,atol=1e-12,rtol=0) else f'{value:g}'))
    ax.set_title('FANS redshift distribution',fontsize=16*float(cfg.get('paper_sample_heading_scale',cfg.get('paper_sample_font_scale',1))),pad=15)
    texts=[]
    for _,r in annotations.iterrows():
        yy=.942 if r.catalog=='carron' else .866
        text=ax.text((r.z_min+r.z_max)/2,yy,r.text,transform=ax.get_xaxis_transform(),
            color=COLOR[r.catalog],ha='center',va='center',fontsize=cfg['paper_fraction_annotation_size'],
            bbox=dict(fc='white',ec='none',alpha=.9,pad=.4))
        texts.append((text,r.z_min,r.z_max))
    # Check the annotation fits in the bin; reduce its font only as necessary.
    ax.figure.canvas.draw();renderer=ax.figure.canvas.get_renderer()
    for text,zlo,zhi in texts:
        width=(ax.transData.transform((zhi,0))[0]-ax.transData.transform((zlo,0))[0])*.97
        bbox=text.get_window_extent(renderer)
        if bbox.width>width:
            if cfg.get('_require_fraction_font_size',False):
                raise ValueError('Fraction annotation does not fit its bin at the requested font size; '
                                 'increase paper_sample_width rather than shrink its text.')
            text.set_fontsize(text.get_fontsize()*width/bbox.width)
    return texts


def _distribution_handles(cfg=None):
    cfg=cfg or DEFAULTS
    return ([Line2D([0],[0],color=COLOR[c],lw=cfg['paper_sample_linewidth_all'],label='Carrón' if c=='carron' else 'Chen') for c in CATS]+
        [Line2D([0],[0],color='.2',lw=cfg['paper_sample_linewidth_conditioned'],ls=DIST_LINE[p],label={'all':'All multiplicities','m2':'2 arms','m3':'3 arms'}[p]) for p in ('all','m2','m3')])



def draw_transition_axis(ax,table,cat,cfg,vmax=None):
    counts,cmap,norm,bounds,ticks,metadata=transition_display_spec(table,cat,style=cfg,vmax=vmax)
    if counts.empty:raise ValueError(f'No candidate >=1 transitions to plot: {cat}')
    a=counts.to_numpy(float);scale=float(cfg.get('paper_sample_font_scale',1.))
    image=ax.imshow(np.ma.masked_equal(a,0),origin='lower',aspect='auto',interpolation='nearest',cmap=cmap,norm=norm)
    hatch_zero_transition_cells(ax,a,cfg)
    ax.set_box_aspect(cfg['paper_panel_box_aspect'])
    ax.set_xticks(np.arange(a.shape[1]),[str(v) for v in counts.columns]);ax.set_yticks(np.arange(a.shape[0]),[str(v) for v in counts.index])
    ax.set_xticks(np.arange(-.5,a.shape[1]),minor=True);ax.set_yticks(np.arange(-.5,a.shape[0]),minor=True)
    ax.grid(False,which='major');ax.grid(which='minor',color='.8',lw=.6);ax.tick_params(which='minor',length=0)
    for i,j in np.ndindex(a.shape):
        val=int(a[i,j]);rgb=np.asarray(cmap(norm(val))[:3])
        colour='white' if val and np.dot(rgb,[.2126,.7152,.0722])<.48 else ('.42' if not val else '.08')
        ax.text(j,i,f'{val:,}',ha='center',va='center',fontsize=float(cfg.get('_sample_cell_size',12.8*scale)),color=colour)
    ax.set_xlabel('Arms surviving trace cuts');ax.set_ylabel('Candidate arms')
    ax.set_title('Carrón' if cat=='carron' else 'Chen',fontsize=16*float(cfg.get('paper_sample_heading_scale',scale)),pad=14)
    data=counts.rename_axis(index='m_candidates',columns='m_trace_survivors').stack().rename('n_nodes').reset_index()
    data['catalog']=cat
    image._fans_transition_ticks=ticks
    image._fans_transition_metadata=metadata
    return image,data,bounds


def sample_settings(style):
    """Enlarge sample-panel typography only; keep the results panel unchanged."""
    cfg=_settings(style);base=float(cfg['paper_sample_font_scale'])
    factor=base*float(cfg['paper_sample_text_increment'])
    heading=base*float(cfg['paper_sample_heading_increment'])
    aspect=float(cfg['paper_sample_box_aspect'])
    numbers=[base,factor,heading,aspect,float(cfg['paper_sample_legend_gap_inches']),
             float(cfg['paper_sample_linewidth_all']),float(cfg['paper_sample_linewidth_conditioned'])]
    if not np.isfinite(numbers).all() or min(numbers)<=0:
        raise ValueError('Sample font scales, line widths, legend gap and box aspect must be positive')
    cfg['paper_panel_box_aspect']=aspect
    cfg['paper_sample_font_scale']=factor
    cfg['paper_sample_heading_scale']=heading
    cfg['paper_panel_label_size']=float(cfg['paper_panel_label_size'])*heading
    for key in ['paper_panel_tick_size','paper_fraction_annotation_size']:
        cfg[key]=float(cfg[key])*factor
    return cfg


def sample_panel_only_settings(style):
    """Resolve only the three-row sample-panel overrides (not standalone PDFs)."""
    cfg=sample_settings(style)
    keys=('paper_sample_panel_letter_increment','paper_sample_panel_legend_increment',
          'paper_sample_panel_fraction_increment','paper_sample_panel_frame_increment',
          'paper_sample_panel_density_upper_limit')
    for key in keys:
        value=float(cfg[key])
        if not np.isfinite(value) or value<=0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('paper_sample_panel_cells_match_ticks','paper_sample_panel_hide_density_zero'):
        if not isinstance(cfg[key],(bool,np.bool_)):
            raise ValueError(f'{key} must be a boolean')
    cfg['paper_fraction_annotation_size']*=float(cfg['paper_sample_panel_fraction_increment'])
    if cfg['paper_sample_panel_cells_match_ticks']:
        cfg['_sample_cell_size']=float(cfg['paper_panel_tick_size'])
    extra_gap=float(cfg['paper_sample_panel_legend_extra_gap_inches'])
    if not np.isfinite(extra_gap) or extra_gap<0:
        raise ValueError('paper_sample_panel_legend_extra_gap_inches must be finite and nonnegative')
    cfg['paper_sample_legend_gap_inches']+=extra_gap
    cfg['_sample_hgap_inches']=2.08+extra_gap
    cfg['_density_upper_limit']=float(cfg['paper_sample_panel_density_upper_limit'])
    cfg['_hide_density_zero']=bool(cfg['paper_sample_panel_hide_density_zero'])
    cfg['_require_fraction_font_size']=True
    cfg['_paper_frame_linewidth']=FRAME_LINEWIDTH*float(cfg['paper_sample_panel_frame_increment'])
    return cfg


def plot_sample_panel(folder,style,*,fiducial_run=None,include_standalone=True):
    folder=Path(folder);cfg=sample_settings(style);fontscale=float(cfg['paper_sample_font_scale'])
    binned,edges,deps=binned_multiplicity(folder)
    nodes,cachepaths,ndeps,status=prepare_redshift_cache(folder,binned,edges,fiducial_run);deps+=ndeps
    if nodes is None:return [],deps,'pending_individual_redshifts'
    curves=distribution_curves(nodes,binned,edges,cfg)
    ann=fraction_annotations(binned,int(cfg['paper_fraction_decimals']))
    tables={c:pd.read_csv(folder/f'{c}_multiplicity_transitions.csv') for c in CATS}
    deps += [folder/f'{c}_multiplicity_transitions.csv' for c in CATS]
    commonmeta={'curves':'Six Gaussian KDEs from genuine individual accepted-node redshifts',
        'normalization':'p(z|catalogue) and p(z|catalogue,m); each density integrates to one on the real line; no boundary correction',
        'fractions':'P(m|catalogue,z bin), computed from counts, not KDE height ratios',
        'fraction_denominator':'all accepted FANS in each bin, including >=4 arms; 2A+3A can be below 100%',
        'annotation_order':'Carron (royalblue) above Chen (orangered) in each bin',
        'edges':edges.tolist(),'kde_source_status':status,'unsupported_individual_redshifts':'no approximation or synthetic replacement',
        'grey_grid_drawn':False,'bin_boundaries_drawn':True,'font_scale':fontscale,
        'heading_font_scale':cfg['paper_sample_heading_scale'],
        'legend_gap_inches':cfg['paper_sample_legend_gap_inches'],
        'curve_linewidth_all_pt':cfg['paper_sample_linewidth_all'],
        'curve_linewidth_conditioned_pt':cfg['paper_sample_linewidth_conditioned'],
        'transition_population':'scope=full, edge_safe=True, candidate arms >=1; parent status summed',
        'transition_accepted_fans_warning':'0/1 surviving arms are not accepted FANS',
        'transition_scale':cfg.get('transition_scale'),'transition_scales_shared':False,
        'transition_zero':'light grey with diagonal hatching = exactly zero nodes; white = lowest nonzero counts',
        'transition_not_physical_connectivity':True}
    values=pd.concat([curves.assign(record_type='kde_curve'),ann.assign(record_type='bin_annotation')],ignore_index=True)
    paths=[]
    standalone_cfg=cfg
    cfg=sample_panel_only_settings(style)
    with panel_style(cfg):
        fig,grid=equal_axes_grid(3,1,width=cfg['paper_sample_width'],aspect=cfg['paper_panel_box_aspect'],
            bottom=1.05,left=1.55,right=1.85,top=1.00,hgap=cfg['_sample_hgap_inches'])
        axes=grid[:,0];texts=draw_distribution_axis(axes[0],curves,ann,edges,cfg)
        transitions=[];transmeta={}
        for ax,cat in zip(axes[1:],CATS):
            image,data,bounds=draw_transition_axis(ax,tables[cat],cat,cfg);transitions.append(data)
            transmeta[cat]=image._fans_transition_metadata
            pos=ax.get_position()
            cax=fig.add_axes([pos.x1+.022,pos.y0,.018,pos.height])
            ticks=image._fans_transition_ticks
            cb=fig.colorbar(image,cax=cax,boundaries=bounds,ticks=ticks)
            cb.ax.set_yticklabels([f'{x:,.0f}' for x in ticks]);cb.ax.tick_params(labelsize=11*fontscale)
            cb.set_label('Number of nodes',fontsize=13*cfg['paper_sample_heading_scale'],labelpad=11)
        pos=axes[0].get_position()
        fig.legend(handles=_distribution_handles(cfg),loc='center',bbox_to_anchor=((pos.x0+pos.x1)/2,pos.y0-float(cfg['paper_sample_legend_gap_inches'])/fig.get_figheight()),
            ncol=5,frameon=False,
            fontsize=11.5*fontscale*float(cfg['paper_sample_panel_legend_increment']),
            handlelength=2.0,columnspacing=1.0)
        for ax,letter in zip(axes,'abc'):
            ax.text(.008,1.025,f'({letter})',transform=ax.transAxes,ha='left',
                    fontsize=13*fontscale*float(cfg['paper_sample_panel_letter_increment']))
        tv=pd.concat(transitions,ignore_index=True).assign(record_type='transition_count')
        paths+=_save(fig,folder/'fans_sample_multiplicity_panel.pdf',pd.concat([values,tv],ignore_index=True),
                    {**commonmeta,'layout':'one column, three equal-aspect data panels; independent catalogue colourbars',
                     'transition_details':transmeta,
                     'legend_gap_inches':cfg['paper_sample_legend_gap_inches'],
                     'density_y_limits':list(axes[0].get_ylim()),
                     'density_zero_tick_label_hidden':cfg['_hide_density_zero'],
                     'density_curve_maximum':float(curves.density.max()),
                     'density_grid_values_above_upper_limit':int((curves.density>cfg['_density_upper_limit']).sum()),
                     'subplot_letter_font_size_pt':13*fontscale*float(cfg['paper_sample_panel_letter_increment']),
                     'line_legend_font_size_pt':11.5*fontscale*float(cfg['paper_sample_panel_legend_increment']),
                     'transition_cell_font_size_pt':float(cfg.get('_sample_cell_size',12.8*fontscale)),
                     'axis_tick_font_size_pt':float(cfg['paper_panel_tick_size']),
                     'annotation_requested_font_size_pt':float(cfg['paper_fraction_annotation_size']),
                     'annotation_actual_font_sizes':[t.get_fontsize() for t,_,_ in texts]},axes,cfg)
    if not include_standalone:
        return paths,deps,status
    cfg=standalone_cfg
    with panel_style(cfg):
        fig,grid=equal_axes_grid(1,1,width=cfg['paper_sample_width'],aspect=cfg['paper_panel_box_aspect'],
            bottom=1.50,left=1.55,right=.45,top=.90)
        draw_distribution_axis(grid[0,0],curves,ann,edges,cfg)
        pos=grid[0,0].get_position()
        fig.legend(handles=_distribution_handles(cfg),loc='center',
                   bbox_to_anchor=((pos.x0+pos.x1)/2,pos.y0-float(cfg['paper_sample_legend_gap_inches'])/fig.get_figheight()),ncol=5,frameon=False,
                   fontsize=11.5*fontscale,handlelength=2.0)
        paths+=_save(fig,folder/'fans_redshift_distribution_total_multiplicity_annotated.pdf',values,commonmeta,grid,cfg)
    return paths,deps,status


def generate_editorial_panels(folder,style,*,fiducial_run=None,datasets=None,prepared=None):
    """Called by the existing atomic plot refresh and the full collector."""
    folder=Path(folder);cfg=_settings(style)
    paths,deps,_=plot_spirality_standalone(folder,cfg)
    pp,dd,status=plot_sample_panel(folder,cfg,fiducial_run=fiducial_run);paths+=pp;deps+=dd
    ext_ready=all((folder/f'{c}_extrinsic_zbins.csv').exists() and (folder/f'{c}_extrinsic_bestfit.csv').exists() for c in CATS)
    if ext_ready:
        pp,dd=plot_results_panel(folder,cfg,datasets=datasets,prepared=prepared,fiducial_run=fiducial_run);paths+=pp;deps+=dd
    metadata={'native_matplotlib':True,'box_aspect':cfg['paper_panel_box_aspect'],
        'sample_panel':status,'results_panel':'generated' if ext_ready else 'pending_extrinsic',
        'unsupported_spirality_points':'omitted, explained in caption and metadata; no scientific row removed'}
    return paths,deps,metadata
