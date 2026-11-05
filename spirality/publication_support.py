"""Redraw inherited support figures from compact tables for a uniform frame.

No new statistics or resampling. Existing tabular science products are never
written. Later native renderers replace S and transitions with the chosen style.
"""
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .publication_style import publication_frame_scope, save_publication_figure


@publication_frame_scope()
def refresh_support_figures(folder, style, fiducial_run=None):
    from .paper_products import style_context, COLOR, LABEL, plot_projected_sign_asymmetry
    from .multiplicity import plot_analysis
    folder=Path(folder);paths=[];deps=[];report={}
    for cat in ('carron','chen'):
        names=['statistics','fractions','composition','selection_stages','transitions']
        originals=[folder/f'{cat}_multiplicity_{key}.csv' for key in names]
        if not all(p.is_file() for p in originals):continue
        with tempfile.TemporaryDirectory(prefix='frame_support_') as tmp:
            for key,src in zip(names,originals):
                shutil.copy2(src,Path(tmp)/f'multiplicity_{key}.csv');deps.append(src)
            plot_analysis(Path(tmp),cat)
            for p in Path(tmp).glob('*.pdf'):
                # The dedicated S/transition renderers run next. Avoid downgrading
                # the angular S plot even if a later optional product is absent.
                if p.name in {f'{cat}_S_fixed_multiplicity.pdf',f'{cat}_multiplicity_transitions.pdf'}:continue
                target=folder/p.name
                if target.exists():shutil.copy2(p,target);paths.append(target)
    fidpath=folder/'fiducial_multiplicity_statistics.csv'
    fid=pd.read_csv(fidpath) if fidpath.exists() else None
    if fid is not None:
        bins=fid[(fid.catalog=='carron') & (fid.stratum=='all') & (fid.scope=='zbin')].sort_values('zbin_id')
        edges=np.r_[bins.z_min,bins.z_max.iloc[-1]];deps.append(fidpath)
        paths+=plot_projected_sign_asymmetry(fid,edges,folder,style)
        kdepath=folder/'fans_redshift_distribution_kde.csv'
        if kdepath.exists():
            curves=pd.read_csv(kdepath)
            if set(curves.method)=={'gaussian_kde_scott'}:
                with style_context(style):
                    fig,ax=plt.subplots(figsize=style['redshift_figsize'])
                    for cat in ('carron','chen'):
                        d=curves[curves.catalog==cat].sort_values('z')
                        ax.plot(d.z,d.density,color=COLOR[cat],label=LABEL[cat],lw=3)
                    for z in edges[1:-1]:ax.axvline(z,color='black',lw=.9,alpha=.85)
                    ax.set(xlim=(edges[0],edges[-1]),xlabel=r'$z$',ylabel='Density',title='FANS redshift distribution')
                    ax.grid(True,color='.82',lw=.75,alpha=.35)
                    ax.legend(fontsize=style['font_size']-3,frameon=True,handlelength=1.8,handletextpad=.55)
                    p=folder/'fans_redshift_distribution.pdf';save_publication_figure(fig,p,bbox_inches='tight',dpi=300);plt.close(fig)
                    paths.append(p);deps.append(kdepath)
    gp=folder/'grid27_statistics.csv';cp=folder/'grid27_composition.csv'
    if gp.exists():
        stats=pd.read_csv(gp);deps.append(gp)
        for cat in ('carron','chen'):
            fcat=fid[fid.catalog==cat] if fid is not None else None
            for cl in ('all','m2','m3'):
                for metric in ('H','A_sign','S'):
                    out=folder/f'grid_{cat}_{cl}_{metric}.pdf'
                    if not out.exists():continue
                    with plt.rc_context():
                        fig,ax=plt.subplots(figsize=(6.1,4.5))
                        subset=stats[(stats.catalog==cat)&(stats.stratum==cl)&(stats.scope=='zbin')]
                        for _,g in subset.groupby('run_name'):
                            g=g.sort_values('zbin_id');r=g.iloc[0]
                            ax.plot(.5*(g.z_min+g.z_max),g[metric],linestyle={30.:'--',35.:'-',40.:':'}[float(r.arm_opening_deg)],
                                marker={8.:'o',10.:'s',12.:'^'}[float(r.node_link_radius_mpc_h)],alpha=.42,
                                lw={7.5:.6,10.:1.2,12.5:1.9}[float(r.segment_length_mpc_h)],ms=3)
                        if fcat is not None:
                            f=fcat[(fcat.stratum==cl)&(fcat.scope=='zbin')].sort_values('zbin_id')
                            ax.errorbar(.5*(f.z_min+f.z_max),f[metric],yerr=f[f'{metric}_err'],fmt='o-',lw=2,capsize=3,label='Fiducial')
                        if metric!='S':ax.axhline(0,linestyle=':',lw=.7)
                        ax.set(xlabel='Redshift z',ylabel={'S':r'$S_{\rm proj}$','H':r'$H_{\rm proj}$','A_sign':r'$A_{\rm sign}$'}[metric],
                            title=f'{cat.capitalize()}: {cl}; {stats.run_name.nunique()} construction choices')
                        handles=[Line2D([],[],linewidth=w,label=f'L={L:g}') for L,w in [(7.5,.6),(10,1.2),(12.5,1.9)]]
                        handles += [Line2D([],[],marker=m,ls='none',label=f'R={r:g}') for r,m in [(8,'o'),(10,'s'),(12,'^')]]
                        handles += [Line2D([],[],ls=l,label=f'opening={a:g}°') for a,l in [(30,'--'),(35,'-'),(40,':')]]
                        ax.legend(handles=handles,fontsize=7,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False)
                        fig.tight_layout();save_publication_figure(fig,out);plt.close(fig);paths.append(out)
            out=folder/f'grid_{cat}_fixed_composition.pdf'
            if cp.exists() and out.exists():
                comp=pd.read_csv(cp);c=comp[comp.catalog==cat]
                fig,ax=plt.subplots(figsize=(5.8,3.9))
                for _,g in c.groupby('run_name'):
                    g=g.sort_values('zbin_id');ax.plot(g.z_center,g.S_fixed_composition,marker='o',alpha=.4,lw=.8,ms=3)
                ax.set(xlabel='Redshift z',ylabel=r'$S_{\rm fixed\ composition}$',title=f'{cat.capitalize()}: common 2/3-arm composition')
                fig.tight_layout();save_publication_figure(fig,out);plt.close(fig);paths.append(out);deps.append(cp)
    # Footprints require real sky positions, absent from the compact z-only cache.
    # Regenerate them when the validated fiducial is present; otherwise preserve
    # the archived images and report that precise scope rather than edit a PDF.
    skyfiles=list(folder.glob('fans_density_*.pdf'))
    if skyfiles and fiducial_run is not None and Path(fiducial_run).is_dir():
        from .paper_products import load_accepted_nodes
        from .paper_footprints import build_footprints
        try:
            data={}
            for cat in ('carron','chen'):
                data[cat],src=load_accepted_nodes(Path(fiducial_run),cat,edges);deps.append(src)
            # Generate only PDFs; do not replace scientific footprint metadata.
            with tempfile.TemporaryDirectory(prefix='frame_footprints_') as tmp:
                plots,_=build_footprints(data,Path(tmp),nside=int(style['footprint_nside']),
                    density_per_deg2=style['density_per_deg2'],font_size=style['font_size'])
                for p in plots:
                    if p.suffix=='.pdf':
                        target=folder/p.name;shutil.copy2(p,target);paths.append(target)
            report['footprints']='redrawn_from_validated_fiducial_nodes'
        except ImportError:
            report['footprints']='archived_PDFs_preserved_missing_optional_sky_dependencies'
    elif skyfiles:
        report['footprints']='archived_PDFs_preserved_no_individual_sky_positions'
    report['frame_linewidth_pt']=1.1
    report['support_figures_redrawn']=sum(p.suffix=='.pdf' for p in paths)
    return paths,deps,report
