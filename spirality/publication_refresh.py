"""Update publication figures from existing compact results, never rerun science.

The original 88.2 compact bundle did not include the full-range transformed
bootstrap draws or absolute-null quantiles used in its small legend. If those
arrays are unavailable, preserve the already formatted legend verbatim from
its display metadata, or (for 88.2 only) extract selectable PDF text. No OCR,
quantile approximation, or newly generated null/bootstraps are used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import norm

from .provenance import file_digest
from .publication_style import publication_frame_scope
from .publication_matrices import plot_delta_s_matrices, plot_transition_matrix, transition_counts, transition_upper_bound

CATS=('carron','chen')
PANEL='extrinsic_arms_n_segments_projected_vs_dipole_panel_angle'


def _one(table: pd.DataFrame, mask, label: str) -> pd.Series:
    selection=table.loc[mask]
    if len(selection)!=1:raise ValueError(f'Expected one {label}; found {len(selection)}')
    return selection.iloc[0]


def _cached_legend(folder: Path) -> tuple[str, Path, str]:
    sidecar=folder/f'{PANEL}.display_metadata.json'
    if sidecar.exists():
        meta=json.loads(sidecar.read_text())
        text=meta.get('fullrange_legend_text')
        if not isinstance(text,str) or not text.startswith('Full-range values:'):
            raise ValueError(f'Invalid cached display legend: {sidecar}')
        return text,sidecar,'verbatim_saved_display_metadata'
    pdf=folder/f'{PANEL}.pdf'
    if not pdf.exists():
        raise FileNotFoundError('Full-range display cache unavailable: provide the existing angular PDF or the fiducial run with saved draws')
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError('One-time legacy PDF-text import requires PyMuPDF: python -m pip install pymupdf') from exc
    with fitz.open(pdf) as document:
        text='\n'.join(page.get_text('text') for page in document)
    start=text.find('Full-range values:')
    if start<0:raise ValueError(f'Could not find selectable full-range legend text in {pdf}; no OCR/fallback estimation is attempted')
    lines=text[start:].splitlines()
    if len(lines)<6 or not all(lines[i].startswith(p) for i,p in enumerate(('Full-range values:','H:','CCW/CW:','S:','Dip.:','Axial:'))):
        raise ValueError(f'Unexpected full-range legend format in {pdf}; preserve the source or supply the full fiducial products')
    return '\n'.join(lines[:6]),pdf,'verbatim_selectable_text_from_existing_88_2_pdf'


def prepare_cached_panel(folder: Path, *, fiducial_run: Path|None=None) -> tuple[list,dict,list[Path],dict]:
    from .extrinsic_spirality import STYLE, _format_fullrange_values_legend
    folder=Path(folder)
    projected={}; nulls={};totals={};sigma={'H':[],'A_sign':[]};datasets=[];dependencies=[];allrows=[]
    for cat in CATS:
        names={key:folder/f'{cat}_{value}.csv' for key,value in {
            'stats':'multiplicity_statistics','null':'multiplicity_null_summary',
            'bins':'extrinsic_zbins','best':'extrinsic_bestfit'}.items()}
        loaded={key:pd.read_csv(path) for key,path in names.items()};dependencies.extend(names.values())
        stats,nt,zext,besttable=(loaded[k] for k in ('stats','null','bins','best'))
        if len(besttable)!=1:raise ValueError(f'Expected one extrinsic best fit for {cat}')
        best=besttable.iloc[0].to_dict()
        if str(best.get('analysis_schema'))!='88.1' or best.get('zbin_axis_mode')!='fixed_global' or best.get('axis_objective')!='dipole':
            raise ValueError(f'Not revised fixed-global-axis dipole results: {cat}')
        st=stats[(stats.stratum=='all')&(stats.scope=='zbin')].sort_values('zbin_id').copy()
        zext=zext.sort_values('zbin_id').copy()
        if st.zbin_id.tolist()!=[0,1,2,3] or zext.zbin_id.tolist()!=[0,1,2,3]:
            raise ValueError('Angular publication panel requires four complete bins')
        for field in ('z_min','z_max','n_arms'):
            if not np.allclose(st[field],zext[field],rtol=0,atol=1e-12):
                raise ValueError(f'Sample/bin mismatch between projected and extrinsic: {cat}/{field}')
        # Both original summary and zext use mean arm redshift (not weighted mean).
        d=zext[['zbin_id','z_min','z_max','z_center']].reset_index(drop=True).copy()
        for source,target,factor in [('H','H_S_weighted_mean',1.),('S','mean_abs_projected_twist',1.),('A_sign','sign_asymmetry_percent',100.)]:
            errtarget={'H':'H_S_bootstrap_err','S':'mean_abs_projected_twist_bootstrap_err','A_sign':'sign_asymmetry_percent_err'}[source]
            d[target]=factor*st[source].to_numpy(float);d[errtarget]=factor*st[f'{source}_err'].to_numpy(float)
        d['catalog_key']=cat;d['catalog_label']=STYLE[cat]['label']
        projected[cat]=d;nulls[cat]={};totals[cat]={}
        full=_one(stats,(stats.stratum=='all')&(stats.scope=='full'),'full-range statistic')
        for source,key,factor in [('H','H',1.),('A_sign','A',100.),('S','S',1.)]:
            value=factor*float(full[source])
            if source!='A_sign':value=float(np.rad2deg(np.arcsin(value)))
            # Only exact central markers are required here. The existing display
            # legend or saved draws supply the uncertainties; no approximation.
            totals[cat][key]={'value':value,'err':np.nan,'mock68':np.nan,'mock95':np.nan}
        for statistic,factor in [('H',1.),('A_sign',100.)]:
            n=nt[(nt.stratum=='all')&(nt.statistic==statistic)&(nt.zbin_id>=0)].sort_values('zbin_id')
            if n.zbin_id.tolist()!=[0,1,2,3]:raise ValueError(f'Missing cached projected null bins: {cat}/{statistic}')
            n=n.reset_index(drop=True)
            expected=st[statistic].to_numpy(float)
            if not np.allclose(n.observed,expected,rtol=1e-10,atol=2e-12):
                raise ValueError('Projected null statistics do not match saved observations')
            tab=d[['zbin_id','z_min','z_max','z_center']].copy()
            for dest,src in [('null_q025','q2sigma_low'),('null_q16','q1sigma_low'),('null_q50','q50'),('null_q84','q1sigma_high'),('null_q975','q2sigma_high')]:
                tab[dest]=factor*n[src].to_numpy(float)
            nulls[cat][statistic]=tab
            nfull=_one(nt,(nt.stratum=='all')&(nt.statistic==statistic)&(nt.zbin_id==-1),'global null')
            if not np.isclose(nfull.observed,full[statistic],rtol=1e-10,atol=2e-12):raise ValueError('Global null/observation mismatch')
            p=float(nfull.p_two_sided)
            if not 0<p<=1:raise ValueError('Invalid stored global p-value')
            sigma[statistic].append((cat,STYLE[cat]['label'],float(norm.isf(p/2))))
        datasets.append((cat,STYLE[cat]['label'],zext,best))
        # Reusable exact values displayed in all five quantitative panels.
        for raw,col,errcol in [('H_proj','H_S_weighted_mean','H_S_bootstrap_err'),('S_proj','mean_abs_projected_twist','mean_abs_projected_twist_bootstrap_err'),('A_sign','sign_asymmetry_percent','sign_asymmetry_percent_err')]:
            for _,row in d.iterrows():
                allrows.append(dict(catalog=cat,panel=raw,zbin_id=int(row.zbin_id),z_min=row.z_min,z_max=row.z_max,z_center=row.z_center,
                                    raw_value=row[col],raw_error=row[errcol],raw_unit='percent' if raw=='A_sign' else 'dimensionless'))
        for raw,col,errcol in [('A_dip','H_ext','H_ext_err'),('A_ax','sign_asym_ext_percent','sign_asym_ext_percent_err')]:
            for _,row in zext.iterrows():
                allrows.append(dict(catalog=cat,panel=raw,zbin_id=int(row.zbin_id),z_min=row.z_min,z_max=row.z_max,z_center=row.z_center,
                                    raw_value=row[col],raw_error=row[errcol],raw_unit='percent' if raw=='A_ax' else 'dimensionless'))
    prepared={'projected':projected,'nulls':nulls,'totals':totals,'sigma_items':sigma}
    raw_available=(fiducial_run is not None and all((Path(fiducial_run)/'referee_multiplicity'/cat/'multiplicity_bootstrap_samples.npz').is_file()
                   and (Path(fiducial_run)/'referee_multiplicity'/cat/'multiplicity_signflip_samples.npz').is_file() for cat in CATS))
    if raw_available:
        from .revision_plot_data import fullrange_markers
        for cat in CATS:
            rawfolder=Path(fiducial_run)/'referee_multiplicity'/cat
            saved=pd.read_csv(rawfolder/'multiplicity_statistics.csv')
            compact=pd.read_csv(folder/f'{cat}_multiplicity_statistics.csv')
            pd.testing.assert_frame_equal(saved,compact)
            prepared['totals'][cat]=fullrange_markers(rawfolder,angle=True)
            dependencies.extend(rawfolder/name for name in ('multiplicity_statistics.csv','multiplicity_bootstrap_samples.npz','multiplicity_signflip_samples.npz'))
        origin='existing_saved_bootstrap_and_null_arrays; no new draws'
        prepared['fullrange_legend_text']=_format_fullrange_values_legend(prepared['totals'],datasets,angle=True)
    else:
        legend,legendpath,origin=_cached_legend(folder)
        dependencies.append(legendpath)
        prepared['fullrange_legend_text']=legend
        # Verify the displayed H/A/S central numbers against the actual tables.
        # This is a provenance cross-check, not extraction of scientific errors.
        for label,panel in [('H','H'),('CCW/CW','A'),('S','S')]:
            line=next(x for x in legend.splitlines() if x.startswith(label+':'))
            for cat,displayname in [('carron','Carron'),('chen','Chen')]:
                match=re.search(rf'{displayname}:\s*([-+\d.]+)±',line)
                if not match or abs(float(match.group(1))-prepared['totals'][cat][panel]['value'])>.00501:
                    raise ValueError('Existing PDF legend disagrees with compact scientific tables')
    meta={'legend_origin':origin,'inputs':[{'file':p.name,'sha256':file_digest(p)} for p in dependencies],
          'no_scientific_recomputation':True,'n_binned_values':len(allrows)}
    prepared['used_values']=pd.DataFrame(allrows)
    return datasets,prepared,dependencies,meta


@publication_frame_scope()
def refresh_bundle_appearance(bundle, *, style: dict, fiducial_run: Path|None=None) -> dict:
    """In-memory staging hook used by both the pipeline collector and offline CLI."""
    out=bundle.stage
    for filename in ('publication_refresh.py','publication_captions.py','publication_matrices.py','publication_redshift.py','publication_panels.py','paper_products.py','publication_style.py','publication_support.py','extrinsic_spirality.py','pipeline.py'):
        bundle.dependency(Path(__file__).with_name(filename))
    def replace(paths,role):
        for p in paths:
            bundle.entries=[e for e in bundle.entries if e['file']!=p.name]
            bundle.add(p,role=role if p.suffix in {'.pdf','.png'} else 'figure_data',generated=True)
    metadata={'updated_utc':datetime.now(timezone.utc).isoformat(),'no_scientific_recomputation':True}
    from .publication_support import refresh_support_figures
    support_paths,support_deps,support_meta=refresh_support_figures(out,style,fiducial_run)
    replace(support_paths,'support_figure')
    for p in support_deps:bundle.dependency(p)
    metadata['uniform_frames']=support_meta
    transition_tables={cat:pd.read_csv(out/f'{cat}_multiplicity_transitions.csv') for cat in CATS if (out/f'{cat}_multiplicity_transitions.csv').exists()}
    if transition_tables:
        for cat,t in transition_tables.items():
            bundle.dependency(out/f'{cat}_multiplicity_transitions.csv')
            replace(plot_transition_matrix(t,cat,out/f'{cat}_multiplicity_transitions.pdf',style=style),'referee_figure')
        metadata['transition_colour_scales']='independent per catalogue; candidate>=1; explicit ticks and permitted colour-only saturation; zero hatching'
    # Number fractions and before/after counts require only compact tables.
    # The smooth KDE is deliberately conditional on genuine individual z values.
    redshift_files = [out/f'{cat}_multiplicity_{key}.csv' for cat in CATS
                      for key in ('statistics','fractions','selection_stages','transitions')]
    if all(p.is_file() for p in redshift_files):
        from .publication_redshift import generate_redshift_products
        paths, status, dependencies = generate_redshift_products(out,style,fiducial_run=fiducial_run)
        replace(paths,'referee_figure')
        for p in dependencies:bundle.dependency(p)
        metadata['redshift_multiplicity_kde']=status
        product='Multiplicity-resolved redshift KDE'
        bundle.status=[s for s in bundle.status if s.get('product')!=product]
        if status=='pending_individual_redshifts':
            bundle.status.append({'product':product,'status':status,
                'required':'--fiducial-run with the original node_spirality.csv files, or the validated individual-redshift cache'})
    else:
        metadata['redshift_multiplicity_kde']='pending_fiducial'

    # Support complete and explicitly labelled partial grids.
    candidates=list(out.glob('grid27_trends.csv'))+list(out.glob('grid_partial*_trends.csv'))
    if len(candidates)>1:raise ValueError('Ambiguous multiple grid summaries in publication folder')
    if candidates:
        trpath=candidates[0];prefix=trpath.name.removesuffix('_trends.csv');stpath=out/f'{prefix}_statistics.csv'
        trends=pd.read_csv(trpath);stats=pd.read_csv(stpath)
        from .angular_contrasts import cached_display_cells
        angular_cells=cached_display_cells(out) if prefix=='grid27' else None
        if angular_cells is None:
            replace(plot_delta_s_matrices(trends,stats,out,style=style,allow_partial=prefix!='grid27'),'robustness_figure')
        else:
            original_modes=style.get('delta_matrix_versions',['sign','intensity'])
            if 'sign' in original_modes:
                scfg={**style,'delta_matrix_versions':['sign']}
                replace(plot_delta_s_matrices(trends,stats,out,style=scfg),'robustness_figure')
            if 'intensity' in original_modes:
                acfg={**style,'delta_matrix_versions':['intensity'],'delta_matrix_intensity_vmax':None}
                replace(plot_delta_s_matrices(trends,stats,out,style=acfg,display_cells=angular_cells,unit='deg'),'robustness_figure')
        bundle.dependency(trpath);bundle.dependency(stpath)
        metadata['delta_S_matrices']='generated_from_paired_bootstrap_trends'
        # Rebuild all grid display variants from saved summaries, never from PDF
        # extraction or a scientific pipeline call. Keep dimensionless originals.
        from .paper_products import plot_grid_panel
        fp=out/'fiducial_multiplicity_statistics.csv'
        fid=pd.read_csv(fp) if fp.exists() else None
        if fp.exists():bundle.dependency(fp)
        bb=stats[(stats.catalog=='carron')&(stats.stratum=='all')&(stats.scope=='zbin')][['zbin_id','z_min','z_max']].drop_duplicates().sort_values('zbin_id')
        if bb.zbin_id.tolist()!=[0,1,2,3]:raise ValueError('Four identical grid bins required')
        edges=np.r_[bb.z_min.to_numpy(float),float(bb.z_max.iloc[-1])]
        nr=int(stats.run_name.nunique())
        for st in ['all','m2','m3']:
            base=prefix+'_projected_panel'+('' if st=='all' else '_'+st)
            for angle in [False,True]:
                target=out/(base+('_angle' if angle else '')+'.pdf')
                replace(plot_grid_panel(stats,fid,edges,target,stratum=st,style=style,n_runs=nr,angle=angle),'robustness_figure')
        metadata['grid_panels']='dimensionless and angular versions; H/S in degrees, A_sign percent; bin boundaries only'

    else:metadata['delta_S_matrices']='pending_grid'
    datasets=prepared=None
    have_extrinsic=all((out/f'{cat}_extrinsic_zbins.csv').exists() and (out/f'{cat}_extrinsic_bestfit.csv').exists() for cat in CATS)
    if have_extrinsic:
        from .extrinsic_spirality import plot_projected_and_dipole_panel
        datasets,prepared,dependencies,detail=prepare_cached_panel(out,fiducial_run=fiducial_run)
        for p in dependencies:bundle.dependency(p)
        target=out/f'{PANEL}.pdf'
        # Consume already prepared values; all inference/fallback paths are bypassed.
        plot_projected_and_dipole_panel(run_root=out,datasets=datasets,out=target,weight_mode='n_segments',
            angle=True,panel_extrinsic_sigma_source='global',axis_legend_error_source='bootstrap',
            prepared_plot_data=prepared,panel_upper_limits=style.get('panel_angular_upper_limits'))
        values=out/f'{PANEL}.used_values.csv';prepared['used_values'].to_csv(values,index=False)
        provenance=out/f'{PANEL}.refresh_provenance.json';provenance.write_text(json.dumps(detail,indent=2))
        replace([target,target.with_suffix('.display_metadata.json'),values,provenance],'main_figure')
        metadata['angular_panel']='regenerated_from_stored_science_and_cached_display_legend'
    else:metadata['angular_panel']='pending_extrinsic'
    if all(p.is_file() for p in redshift_files):
        from .publication_panels import generate_editorial_panels
        paths, dependencies, editorial = generate_editorial_panels(out,style,
            fiducial_run=fiducial_run,datasets=datasets,prepared=prepared)
        replace(paths,'main_figure')
        for p in dependencies:bundle.dependency(p)
        metadata['editorial_panels']=editorial
        product='Native sample/multiplicity panel'
        bundle.status=[s for s in bundle.status if s.get('product')!=product]
        if editorial['sample_panel']=='pending_individual_redshifts':
            bundle.status.append({'product':product,'status':'pending_individual_redshifts',
                'required':'validated individual-redshift cache or --fiducial-run; no distribution curve is invented'})
    from .publication_captions import write_caption_files
    caption_paths, caption_dependencies = write_caption_files(out)
    replace(caption_paths,'figure_caption')
    for p in caption_dependencies:bundle.dependency(p)
    metadata['short_captions']='one LaTeX paragraph per source line; numerical details from saved panel metadata'
    return metadata


def refresh_paper_folder(folder: Path, *, style_path=None, fiducial_run=None,
                         only_sample_panel=False) -> dict:
    """Atomically stage a figure-only refresh of an existing managed collection.

    Original CSV/JSON science products are byte-preserved. Only requested PDFs,
    new matrix/plot provenance files and the collection manifest are replaced.
    Hand-edited managed products cause a clear failure; manual additions survive.
    """
    from .paper_products import Bundle, _commit_bundle, load_style
    folder=Path(folder).expanduser().resolve()
    manifestpath=folder/'manifest.json'
    if not manifestpath.exists():raise FileNotFoundError(f'Expected managed publication collection: {manifestpath}')
    previous=json.loads(manifestpath.read_text())
    if previous.get('science_schema')!='88.1':raise ValueError('Only revised 88.1 scientific products are supported')
    names=[]
    for entry in previous.get('files',[]):
        name=entry['file']
        if Path(name).name!=name or name in names:raise ValueError('Unsafe/duplicate manifest entry')
        names.append(name);p=folder/name
        if not p.is_file() or file_digest(p)!=entry['sha256']:
            raise ValueError(f'Missing/modified managed product: {p}; restore it or preserve your edit under a different name')
    style=load_style(style_path)
    if fiducial_run is not None:
        fiducial_run=Path(fiducial_run).expanduser().resolve()
        if not fiducial_run.is_dir():raise FileNotFoundError(f'Fiducial run not found: {fiducial_run}')
    else:
        saved=previous.get('sources',{}).get('fiducial_run')
        if saved and Path(saved).expanduser().is_dir():
            fiducial_run=Path(saved).expanduser().resolve()
    original_science={name:file_digest(folder/name) for name in names if name.endswith(('.csv','.json'))}
    with tempfile.TemporaryDirectory(prefix='.paper_plot_refresh_',dir=folder.parent) as tmp:
        bundle=Bundle(tmp)
        bundle.status=list(previous.get('status',[]))
        for name in names:
            entry=next(e for e in previous['files'] if e['file']==name)
            bundle.add(folder/name,role=entry.get('role','support'),description=entry.get('description',''))
        metadata=dict(previous.get('metadata',{}))
        if only_sample_panel:
            from .publication_panels import plot_sample_panel
            from .publication_redshift import CACHE, CACHE_META
            # This path must not call the full appearance refresh, standalone
            # plots, extrinsic routines, or grid renderers.
            paths,deps,status=plot_sample_panel(bundle.stage,style,
                fiducial_run=fiducial_run,include_standalone=False)
            if status=='pending_individual_redshifts':
                raise FileNotFoundError('Sample panel requires the validated individual-redshift '
                    'cache or --fiducial-run with the original node_spirality.csv tables. '
                    'No output or manifest has been replaced.')
            for name in (CACHE,CACHE_META):
                path=bundle.stage/name
                if path.exists() and name not in names:
                    paths.append(path)
            from .publication_captions import write_caption_files
            caption_paths, caption_deps = write_caption_files(bundle.stage, only=['fans_sample_multiplicity_panel'])
            paths.extend(caption_paths); deps.extend(caption_deps)
            for path in paths:
                bundle.entries=[e for e in bundle.entries if e['file']!=path.name]
                bundle.add(path,role='main_figure' if path.suffix=='.pdf' else 'figure_data',generated=True)
            for path in deps:
                bundle.dependency(path)
            for filename in ('publication_panels.py','publication_style.py','publication_refresh.py','publication_captions.py'):
                bundle.dependency(Path(__file__).with_name(filename))
            if style_path:
                bundle.dependency(Path(style_path).expanduser().resolve())
            allowed={'fans_sample_multiplicity_panel.pdf',
                     'fans_sample_multiplicity_panel.caption.tex',
                     'fans_sample_multiplicity_panel.used_values.csv',
                     'fans_sample_multiplicity_panel.metadata.json',CACHE,CACHE_META}
            for name in names:
                if name not in allowed and file_digest(bundle.stage/name)!=file_digest(folder/name):
                    raise RuntimeError(f'Sample-only refresh changed unrelated product: {name}')
            bundle.status=[r for r in bundle.status if r.get('product')!='Native sample/multiplicity panel']
            metadata['appearance_update_88_9']={
                'scope':'fans_sample_multiplicity_panel only',
                'updated_utc':datetime.now(timezone.utc).isoformat(),
                'redshift_multiplicity_kde':status,
                'other_managed_products_preserved_byte_for_byte':True,
                'no_scientific_recomputation':True}
        else:
            metadata['appearance_update_88_8']=refresh_bundle_appearance(bundle,style=style,fiducial_run=fiducial_run)
        # Scientific summaries are preserved exactly. Only plot sidecars may change.
        plot_suffixes=('.used_values.csv','.metadata.json','.display_metadata.json','.refresh_provenance.json')
        for name,digest in original_science.items():
            if name.endswith(plot_suffixes):continue
            if file_digest(bundle.stage/name)!=digest:raise RuntimeError(f'Figure refresh altered scientific product: {name}')
        result=_commit_bundle(bundle,folder,previous.get('sources',{}),style,metadata)
    return result
