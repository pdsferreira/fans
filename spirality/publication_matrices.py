"""Publication matrices from saved estimates; no scientific resampling is done.

The original sign-only matrices and an alternative linear-intensity version
are both generated. Colours never encode catalogue identity or significance. Sparse endpoint strata remain diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from .publication_style import save_publication_figure, FRAME_LINEWIDTH
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import LogFormatterMathtext

PARAMETERS = ('segment_length_mpc_h', 'node_link_radius_mpc_h', 'arm_opening_deg')
L_VALUES = (7.5, 10.0, 12.5)
R_VALUES = (8.0, 10.0, 12.0)
A_VALUES = (30.0, 35.0, 40.0)
CAT_LABEL = {'carron': 'Carrón DR16', 'chen': 'Chen DR12'}
STRATA = ('all', 'm2', 'm3')
STRATUM_LABEL = {'all': 'All accepted FANS', 'm2': 'Two-arm FANS', 'm3': 'Three-arm FANS'}
DEFAULTS = {
    'delta_matrix_figsize': [13.0, 12.2],
    'delta_matrix_intensity_figsize': [13.0, 10.7],
    'delta_matrix_intensity_hspace': 0.08,
    'delta_matrix_intensity_wspace': 0.07,
    'delta_matrix_intensity_share_xlabels': True,
    'delta_matrix_intensity_colorbar_gap_inches': 0.22,
    'delta_matrix_intensity_show_notes': False,
    'delta_matrix_positive': '#cfe5f5',
    'delta_matrix_negative': '#f5d3d0',
    'delta_matrix_low_support': '#f0f0f0',
    'delta_matrix_decimals': 4,
    # Angular figures use only two displayed decimals; keep full precision in data.
    'delta_matrix_angular_decimals': 2,
    'delta_matrix_show_errors': True,
    'delta_matrix_mark_95_interval': True,
    'delta_matrix_versions': ['sign', 'intensity'],
    'delta_matrix_intensity_positive': '#79afd5',
    'delta_matrix_intensity_negative': '#d77972',
    'delta_matrix_intensity_vmax': None,
    'delta_matrix_angular_vmax': 4.5,
    'delta_matrix_angular_allow_colour_saturation': True,
    'transition_cmap': 'viridis',
    'transition_scale': 'linear',
    'transition_palette': 'catalogue',
    'transition_min_candidates': 1,
    'transition_zero_colour': '#ededed',
    'transition_zero_hatch': '///',
    'transition_vmax_by_catalogue': {'carron':8500.,'chen':5000.},
    'transition_ticks_by_catalogue': {'carron':[0,2125,4250,6375,8500], 'chen':[0,1250,2500,3750,5000]},
    'transition_allow_colour_saturation': True,
    'delta_matrix_row_label_x': 0.058,
    'transition_count_boundaries': [1, 10, 30, 100, 300, 600, 1000, 2000, 4000, 10000, 30000],
    'transition_figsize': [7.6, 6.0],
}


def _bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    if str(value).strip().lower() not in {'true', 'false', '1', '0'}:
        raise ValueError(f'Invalid boolean support flag: {value!r}')
    return str(value).strip().lower() in {'true', '1'}


def matrix_cells(trends: pd.DataFrame, statistics: pd.DataFrame, *, allow_partial=False) -> pd.DataFrame:
    """Validate and join stored paired endpoint contrasts with endpoint support.

    Delta/error/CI come verbatim from the saved *paired* bootstrap trend table.
    Endpoint means are used only as a consistency check, never to estimate errors.
    """
    tr_required = {'catalog', 'run_name', 'stratum', *PARAMETERS,
                   'delta_S_last_minus_first', 'delta_S_err', 'delta_S_q025', 'delta_S_q975'}
    st_required = {'catalog', 'run_name', 'stratum', *PARAMETERS, 'scope', 'zbin_id',
                   'z_min', 'z_max', 'S', 'n_fans', 'n_blocks', 'precision_support'}
    for table, required, label in [(trends, tr_required, 'trends'), (statistics, st_required, 'statistics')]:
        if not required.issubset(table.columns):
            raise ValueError(f'Missing {label} columns: {sorted(required-set(table.columns))}')
    keys = ['catalog', 'run_name', 'stratum']
    tr = trends[trends.stratum.isin(STRATA)].copy()
    st = statistics[(statistics.scope == 'zbin') & statistics.stratum.isin(STRATA)].copy()
    if tr.duplicated(keys).any() or st.duplicated(keys+['zbin_id']).any():
        raise ValueError('Duplicated trend/endpoint cells')
    rows = []
    for cat in ('carron', 'chen'):
        ct = tr[tr.catalog == cat]
        configs = ct[['run_name', *PARAMETERS]].drop_duplicates()
        if configs.duplicated(list(PARAMETERS)).any() or configs.duplicated('run_name').any():
            raise ValueError(f'Duplicate/inconsistent parameter grid for {cat}')
        actual = {tuple(map(float, v)) for v in configs[list(PARAMETERS)].to_numpy()}
        expected = {(l, r, a) for l in L_VALUES for r in R_VALUES for a in A_VALUES}
        if actual - expected or (not allow_partial and actual != expected):
            raise ValueError(f'{cat}: expected 27 distinct configurations; found {len(actual)}')
        for l in L_VALUES:
            for r in R_VALUES:
                for a in A_VALUES:
                    for stratum in STRATA:
                        match = ct[(ct.stratum == stratum) & np.isclose(ct[PARAMETERS[0]], l)
                                   & np.isclose(ct[PARAMETERS[1]], r) & np.isclose(ct[PARAMETERS[2]], a)]
                        if len(match) == 0:
                            if not allow_partial or (l, r, a) in actual:
                                raise ValueError(f'Missing stratum in completed configuration: {cat}, {(l,r,a)}, {stratum}')
                            rows.append(dict(catalog=cat, stratum=stratum, **dict(zip(PARAMETERS,(l,r,a))),
                                             plot_support=False, support_reason='configuration_not_available',
                                             delta_S_last_minus_first=np.nan, delta_S_err=np.nan,
                                             delta_S_q025=np.nan, delta_S_q975=np.nan, interval_excludes_zero=False))
                            continue
                        if len(match) != 1:
                            raise ValueError('Duplicate matrix cell')
                        row = match.iloc[0].to_dict()
                        ss = st[(st.catalog == cat) & (st.run_name == row['run_name']) & (st.stratum == stratum)].sort_values('zbin_id')
                        if ss.zbin_id.tolist() != [0, 1, 2, 3]:
                            raise ValueError(f'Missing four-bin statistics for {cat}/{row["run_name"]}/{stratum}')
                        for p in PARAMETERS:
                            if not np.allclose(ss[p].to_numpy(float), float(row[p]), rtol=0, atol=1e-12):
                                raise ValueError('Parameter mismatch between trend and statistics table')
                        first, last = ss.iloc[0], ss.iloc[-1]
                        delta = float(row['delta_S_last_minus_first'])
                        calculated = float(last.S) - float(first.S)
                        if not np.isclose(delta, calculated, rtol=1e-10, atol=2e-12, equal_nan=True):
                            raise ValueError('Stored Delta S does not match the same first/last bins')
                        reasons = []
                        for tag, record in [('first',first), ('last',last)]:
                            row[f'n_fans_{tag}'] = int(record.n_fans)
                            row[f'n_blocks_{tag}'] = int(record.n_blocks)
                            row[f'precision_support_{tag}'] = _bool(record.precision_support)
                            row[f'S_{tag}'] = float(record.S)
                            row[f'z_min_{tag}'] = float(record.z_min)
                            row[f'z_max_{tag}'] = float(record.z_max)
                            if not row[f'precision_support_{tag}']:
                                reasons.append(f'insufficient_{tag}_bin_support')
                        err, lo, hi = (float(row[k]) for k in ('delta_S_err','delta_S_q025','delta_S_q975'))
                        if np.isfinite(err) and err < 0:
                            raise ValueError('Negative stored bootstrap error')
                        if np.isfinite(lo) and np.isfinite(hi) and lo > hi:
                            raise ValueError('Reversed stored bootstrap interval')
                        if not np.isfinite([delta,err,lo,hi]).all():
                            reasons.append('undefined_paired_bootstrap_interval')
                        if not np.isfinite(delta):
                            reasons.append('undefined_endpoint_mean')
                        row['plot_support'] = not reasons
                        row['support_reason'] = ';'.join(reasons) if reasons else 'supported'
                        row['interval_excludes_zero'] = bool(not reasons and (lo > 0 or hi < 0))
                        row['fiducial_configuration'] = (l,r,a)==(10.,10.,35.)
                        rows.append(row)
    cells = pd.DataFrame(rows)
    for cat in ('carron','chen'):
        q = cells[(cells.catalog==cat)&cells.plot_support]
        bounds = q[['z_min_first','z_max_first','z_min_last','z_max_last']].drop_duplicates()
        if len(bounds)>1:
            raise ValueError('Endpoint bin definitions vary across grid cells')
    return cells


def _plot_delta_s_matrices_mode(trends: pd.DataFrame, statistics: pd.DataFrame, outdir: Path,
                          *, style: Mapping | None=None, allow_partial=False, colour_mode="sign", intensity_limit=None, display_cells=None, unit="dimensionless") -> list[Path]:
    cfg={**DEFAULTS, **(dict(style) if style else {})}
    cells=matrix_cells(trends,statistics,allow_partial=allow_partial) if display_cells is None else display_cells.copy(deep=True)
    if unit not in {"dimensionless", "deg"}:
        raise ValueError("Unknown contrast display unit")
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    paths=[]
    intensity = colour_mode == 'intensity'
    cmap = LinearSegmentedColormap.from_list('delta_S_red_white_blue',
        [cfg['delta_matrix_intensity_negative'], 'white', cfg['delta_matrix_intensity_positive']])
    allow_saturation = unit == 'deg' and bool(cfg['delta_matrix_angular_allow_colour_saturation'])
    norm = Normalize(-intensity_limit, intensity_limit, clip=allow_saturation) if intensity else None
    rc={'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
        'mathtext.fontset':'stix','pdf.fonttype':42,'font.size':13,'axes.titlesize':17,
        'axes.labelsize':15,'xtick.labelsize':13,'ytick.labelsize':13}
    for cat in ('carron','chen'):
        c=cells[cells.catalog==cat].copy()
        prefix='grid27' if c.run_name.nunique()==27 else f'grid_partial{c.run_name.nunique()}'
        output=outdir/f'{prefix}_{cat}_delta_S_matrices{"_intensity" if intensity else ""}.pdf'
        with plt.rc_context(rc):
            compact_x = intensity and bool(cfg['delta_matrix_intensity_share_xlabels'])
            figsize = cfg['delta_matrix_intensity_figsize'] if intensity else cfg['delta_matrix_figsize']
            hspace = float(cfg['delta_matrix_intensity_hspace']) if intensity else .32
            wspace = float(cfg['delta_matrix_intensity_wspace']) if intensity else .13
            if not np.isfinite([hspace,wspace]).all() or min(hspace,wspace)<0:
                raise ValueError('Matrix subplot gaps must be finite and nonnegative')
            fig, axes=plt.subplots(3,3,figsize=figsize,sharex=compact_x)
            fig.subplots_adjust(left=.135,right=.978,top=.888,bottom=.21 if intensity else .161,hspace=hspace,wspace=wspace)
            grid_centre = .5 * (axes[0,0].get_position().x0 + axes[0,-1].get_position().x1)
            title_x = grid_centre if intensity else .5
            centre_x = grid_centre if intensity else .55
            fig.suptitle(CAT_LABEL[cat],fontsize=24,x=title_x,y=.986)
            subtitle = (r'$\Delta\mathcal{S}^{(m)}$ [deg]' if unit=='deg' else r'$\Delta\mathcal{S}=\mathcal{S}_{\rm proj}(z_4)-\mathcal{S}_{\rm proj}(z_1)$')
            fig.text(centre_x,.942,subtitle+'  |  first and last redshift bins',ha='center',fontsize=17)
            for outerrow,stratum in enumerate(STRATA):
                for col,a in enumerate(A_VALUES):
                    ax=axes[outerrow,col]
                    ax.set(xlim=(-.5,2.5),ylim=(2.5,-.5))
                    ax.set_xticks(range(3),[f'{r:g}' for r in R_VALUES])
                    ax.set_yticks(range(3),[f'{l:g}' for l in L_VALUES])
                    ax.tick_params(length=0,pad=5,top=False,right=False)
                    if compact_x:
                        ax.tick_params(axis='x',labelbottom=(outerrow==2),labeltop=False)
                    if col==0:
                        ax.set_ylabel(r'$L_{\rm seg}\ [h^{-1}\,{\rm Mpc}]$',labelpad=9)
                        pos=ax.get_position()
                        fig.text(float(cfg['delta_matrix_row_label_x']) if intensity else .031,(pos.y0+pos.y1)/2,STRATUM_LABEL[stratum],rotation=90,
                                 ha='center',va='center',fontsize=17,fontweight='bold')
                    else:
                        ax.tick_params(labelleft=False)
                    if outerrow==2:
                        ax.set_xlabel(r'$R_{\rm link}\ [h^{-1}\,{\rm Mpc}]$',labelpad=8)
                    if outerrow==0:
                        ax.set_title(rf'$\Delta\phi_{{\rm arm}}={a:g}^\circ$',pad=9)
                    for i,l in enumerate(L_VALUES):
                        for j,r in enumerate(R_VALUES):
                            cell=c[(c.stratum==stratum)&np.isclose(c[PARAMETERS[0]],l)
                                   &np.isclose(c[PARAMETERS[1]],r)&np.isclose(c[PARAMETERS[2]],a)].iloc[0]
                            value=float(cell.delta_S_last_minus_first)
                            supported=_bool(cell.plot_support)
                            face=(cfg['delta_matrix_positive'] if value>0 else cfg['delta_matrix_negative'] if value<0 else 'white') if supported else cfg['delta_matrix_low_support']
                            if intensity and supported:
                                face = cmap(norm(value))
                            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=face,edgecolor='.76',linewidth=.65,
                                                  hatch=None if supported else '///'))
                            star='*' if cfg['delta_matrix_mark_95_interval'] and _bool(cell.interval_excludes_zero) else ''
                            dagger='' if supported else r'$^{\dagger}$'
                            decimals=int(cfg['delta_matrix_angular_decimals'] if unit == 'deg' else cfg['delta_matrix_decimals'])
                            if np.isfinite(value):
                                formatted=(f'{value:+.1e}' if value!=0 and round(value,decimals)==0
                                           else f'{value:+.{decimals}f}')
                                primary=f'{formatted}{star}{dagger}'
                            else:
                                primary='—'
                            if supported and cfg['delta_matrix_show_errors']:
                                secondary=rf'$\pm\,{float(cell.delta_S_err):.{decimals}f}$'
                            elif not supported and 'n_fans_last' in cell and pd.notna(cell.n_fans_last):
                                secondary=rf'$N_1={int(cell.n_fans_first)},\ N_4={int(cell.n_fans_last)}$'
                            elif not supported:
                                secondary='not available'
                            else:
                                secondary=''
                            ax.text(j,i-.13 if secondary else i,primary,ha='center',va='center',fontsize=14,
                                    fontweight='bold' if supported else 'normal',color='.10',
                                    bbox={'facecolor':face,'edgecolor':'none','pad':.5} if not supported else None)
                            if secondary:
                                ax.text(j,i+.20,secondary,ha='center',va='center',fontsize=10.4,color='.22',
                                        bbox={'facecolor':face,'edgecolor':'none','pad':.4} if not supported else None)
                            if (l,r,a)==(10.,10.,35.):
                                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='none',edgecolor='.08',linewidth=2.0,zorder=6))
                    for spine in ax.spines.values():spine.set_color('.35');spine.set_linewidth(.8)
            legend=[Patch(facecolor=cfg['delta_matrix_positive'],edgecolor='.6',label=r'$\Delta\mathcal{S}>0$'),
                    Patch(facecolor=cfg['delta_matrix_negative'],edgecolor='.6',label=r'$\Delta\mathcal{S}<0$'),
                    Patch(facecolor=cfg['delta_matrix_low_support'],hatch='///',edgecolor='.76',label='Limited support (†)'),
                    Patch(facecolor='none',edgecolor='.08',linewidth=2,label='Fiducial parameters')]
            colorbar_gap_inches = None
            colorbar_position = None
            legend_y = .071
            if intensity:
                legend = legend[2:]
                # Measure the native x-axis labels, so larger fonts cannot make
                # the shared colour bar touch R_link. No PDF manipulation.
                gap = float(cfg['delta_matrix_intensity_colorbar_gap_inches'])
                if not np.isfinite(gap) or gap <= 0:
                    raise ValueError('Colour-bar gap must be positive and finite')
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                lowest_label_px = min(a.xaxis.label.get_window_extent(renderer).y0 for a in axes[-1])
                bar_height = .014
                bar_top = lowest_label_px / fig.bbox.height - gap / fig.get_figheight()
                bar_bottom = bar_top - bar_height
                if bar_bottom <= 0:
                    raise ValueError('Insufficient bottom margin for colour bar; increase figure height or subplot bottom')
                cax = fig.add_axes([centre_x-.57/2, bar_bottom, .57, bar_height])
                cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
                cb.set_ticks(np.linspace(-intensity_limit, intensity_limit, 5))
                cb.ax.tick_params(labelsize=10, length=3)
                cb.set_label((r'$\Delta\mathcal{S}^{\,(m)}$ [deg]' if unit=='deg' else r'$\Delta\mathcal{S}$')+' (shared linear colour scale)', fontsize=11, labelpad=2)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                colorbar_gap_inches = (lowest_label_px-cax.get_window_extent(renderer).y1)/fig.dpi
                colorbar_position = list(cax.get_position().bounds)
                # The legend lies below the complete bar (including its labels).
                legend_y = cax.get_tightbbox(renderer).y0 / fig.bbox.height - .07/fig.get_figheight()
            fig.legend(handles=legend,loc='upper center' if intensity else 'lower center',
                       bbox_to_anchor=(centre_x,legend_y),ncol=4,frameon=False,
                       fontsize=12.5,columnspacing=1.4,handlelength=1.4)
            notes_drawn = not intensity or bool(cfg['delta_matrix_intensity_show_notes'])
            if notes_drawn:
                fig.text(.55,.043,'Cells: '+r'$\Delta\mathcal{S}\ \pm$ bootstrap standard error; *: 95% percentile interval excludes zero.',
                         ha='center',fontsize=11.5)
                fig.text(.55,.022,('Colour intensity encodes the magnitude of the signed contrast; hatched cells are diagnostic.' if intensity else 'Colours encode sign only. Hatched estimates are diagnostic, not evidence for a trend.'),
                         ha='center',fontsize=11.5)
            save_publication_figure(fig,output,bbox_inches='tight',pad_inches=.12)
            plt.close(fig)
        data=output.with_suffix('.used_values.csv');c.to_csv(data,index=False)
        metadata=output.with_suffix('.metadata.json')
        metadata.write_text(json.dumps({'catalog':cat,'quantity':('(180/pi) * (arcsin(S_bin4) - arcsin(S_bin1))' if unit=='deg' else 'Delta_S = S_bin4 - S_bin1'),
            'display_decimals':int(cfg['delta_matrix_angular_decimals'] if unit == 'deg' else cfg['delta_matrix_decimals']),
            'colourbar_label':((r'$\Delta\mathcal{S}^{\,(m)}$ [deg]' if unit=='deg' else r'$\Delta\mathcal{S}$')+' (shared linear colour scale)') if intensity else None,
            'unit':unit,'outer_rows':list(STRATA),'outer_columns_delta_phi_arm_deg':A_VALUES,
            'inner_rows_L_seg_mpc_h':L_VALUES,'inner_columns_R_link_mpc_h':R_VALUES,
            'positive_colour':cfg['delta_matrix_positive'],'negative_colour':cfg['delta_matrix_negative'],
            'colour_mode': colour_mode,
            'figure_size_inches': list(figsize),
            'subplot_hspace': hspace, 'subplot_wspace': wspace,
            'x_ticklabels_bottom_row_only': compact_x,
            'title_x_figure': title_x, 'subplot_grid_centre_x': grid_centre,
            'colourbar_gap_below_xlabel_inches': colorbar_gap_inches,
            'colourbar_bounds_figure': colorbar_position,
            'explanatory_notes_drawn': notes_drawn,
            'row_label_x': float(cfg['delta_matrix_row_label_x']) if intensity else .031,
            'colour_meaning':('linear signed Delta_S; common symmetric scale across both catalogues and all strata' if intensity else 'sign only; no colour encodes catalogue, amplitude or significance'),
            'intensity_vmin': -intensity_limit if intensity else None,
            'intensity_vmax': intensity_limit if intensity else None,
            **({'colour_saturation_allowed': allow_saturation,
                'n_colour_saturated': int((c.plot_support & (c.delta_S_last_minus_first.abs() > intensity_limit)).sum()),
                'supported_max_abs_deg': float(c.loc[c.plot_support, 'delta_S_last_minus_first'].abs().max())}
               if intensity and unit == 'deg' else {}),
            'intensity_negative_colour': cfg['delta_matrix_intensity_negative'] if intensity else None,
            'intensity_positive_colour': cfg['delta_matrix_intensity_positive'] if intensity else None,
            'star':('angular paired-bootstrap 95% percentile interval excludes zero; not a calibrated p-value' if unit=='deg' else 'stored paired-bootstrap 95% percentile interval excludes zero; not a calibrated p-value'),
            'limited_support':'either endpoint precision_support=False or paired-bootstrap interval undefined',
            'no_scientific_recomputation':True,'no_new_resampling':True,'uncertainty_source':('exact transformed paired bootstrap draws' if unit=='deg' else 'saved dimensionless paired-bootstrap summaries'),'n_supported':int(c.plot_support.sum()),'n_cells':len(c)},indent=2))
        paths += [output,data,metadata]
    return paths



def delta_intensity_limit(cells: pd.DataFrame, requested=None, *, allow_saturation=False) -> float:
    """One symmetric linear scale for supported cells in both catalogues.

    Limited-support cells are grey and do not set the range. Explicitly permitted
    saturation affects colours only: cell values, errors and intervals are retained.
    """
    values = pd.to_numeric(cells.loc[cells.plot_support, 'delta_S_last_minus_first'], errors='coerce').to_numpy(float)
    values = np.abs(values[np.isfinite(values)])
    largest = float(values.max()) if len(values) else 0.
    if requested is not None:
        limit = float(requested)
        if not np.isfinite(limit) or limit <= 0 or (limit < largest and not allow_saturation):
            raise ValueError('Intensity colour limit would clip supported Delta S values')
        return limit
    if largest == 0:
        return .01
    step = 10. ** np.floor(np.log10(largest))
    return float(np.ceil(largest / step) * step)


def plot_delta_s_matrices(trends: pd.DataFrame, statistics: pd.DataFrame, outdir: Path,
                          *, style: Mapping | None=None, allow_partial=False,
                          display_cells=None, unit="dimensionless") -> list[Path]:
    """Generate the selectable sign-only and intensity alternatives side by side."""
    cfg = {**DEFAULTS, **(dict(style) if style else {})}
    modes = cfg['delta_matrix_versions']
    if not isinstance(modes, (list, tuple)) or not modes or len(set(modes)) != len(modes) or set(modes)-{'sign','intensity'}:
        raise ValueError('delta_matrix_versions must be a nonempty unique list of sign and/or intensity')
    cells = matrix_cells(trends, statistics, allow_partial=allow_partial) if display_cells is None else display_cells.copy(deep=True)
    if unit == "deg":
        if display_cells is None or "unit" not in cells or not cells.unit.eq("deg").all():
            raise ValueError("Angular matrices require exact converted paired-bootstrap summaries")
        expected = matrix_cells(trends, statistics, allow_partial=allow_partial)
        from .angular_contrasts import angular_difference, _same
        _same(cells.delta_S_last_minus_first, angular_difference(expected.S_first, expected.S_last), "Angular display order")
    requested = cfg.get('delta_matrix_angular_vmax') if unit == 'deg' else cfg.get('delta_matrix_intensity_vmax')
    allow_saturation = unit == 'deg' and bool(cfg.get('delta_matrix_angular_allow_colour_saturation'))
    limit = delta_intensity_limit(cells, requested, allow_saturation=allow_saturation)
    paths = []
    for mode in modes:
        paths.extend(_plot_delta_s_matrices_mode(trends, statistics, outdir, style=cfg,
            allow_partial=allow_partial, colour_mode=mode, intensity_limit=limit,
            display_cells=cells,unit=unit))
    return paths


def transition_upper_bound(largest: float, style: Mapping | None=None) -> float:
    cfg = {**DEFAULTS, **(dict(style) if style else {})}
    if cfg['transition_scale'] == 'linear':
        return float(max(2., largest))
    if cfg['transition_scale'] == 'log':
        return float(10**np.ceil(np.log10(max(10., largest))))
    bounds = _transition_boundaries(cfg)
    while bounds[-1] <= largest:
        bounds = np.r_[bounds, bounds[-1]*10.]
    return float(bounds[np.searchsorted(bounds, largest, side='right')])


def _transition_boundaries(cfg, vmax=None):
    bounds = np.asarray(cfg['transition_count_boundaries'], float)
    if bounds.ndim != 1 or len(bounds) < 2 or bounds[0] != 1 or not np.isfinite(bounds).all() or not (np.diff(bounds)>0).all():
        raise ValueError('Transition count boundaries must strictly increase from 1')
    if vmax is not None:
        bounds = np.r_[bounds[bounds < vmax], vmax]
    return bounds


def transition_counts(table: pd.DataFrame) -> pd.DataFrame:
    required={'scope','edge_safe','m_candidates','m_trace_survivors','n_nodes'}
    if not required.issubset(table):raise ValueError(f'Missing transition columns: {required-set(table)}')
    keep=(table.scope=='full') & table.edge_safe.map(_bool)
    t=table.loc[keep].copy()
    if t.empty:return pd.DataFrame()
    vals=t[['m_candidates','m_trace_survivors','n_nodes']].to_numpy(float)
    if not np.isfinite(vals).all() or (vals<0).any() or not np.equal(vals,np.rint(vals)).all():
        raise ValueError('Transition multiplicities/counts must be finite nonnegative integers')
    pivot=t.pivot_table(index='m_candidates',columns='m_trace_survivors',values='n_nodes',aggfunc='sum',fill_value=0)
    return pivot.reindex(index=range(int(pivot.index.max())+1),columns=range(int(pivot.columns.max())+1),fill_value=0).astype(int)


def transition_display_spec(table: pd.DataFrame, catalog: str, *, style=None, vmax=None):
    """View-only candidate filter and an independent, labelled colour scale.

    transition_counts remains the unfiltered audit. A zero surviving-arm column
    stays visible: zero survivors of a nonzero candidate system is informative.
    """
    cfg={**DEFAULTS, **dict(style or {})}
    raw=transition_counts(table)
    minimum=cfg['transition_min_candidates']
    if isinstance(minimum,bool) or not float(minimum).is_integer() or int(minimum)<0:
        raise ValueError('transition_min_candidates must be a nonnegative integer')
    counts=raw.loc[raw.index>=int(minimum)].copy()
    largest=float(counts.to_numpy().max()) if counts.size else 0.
    scale=cfg['transition_scale']
    overrides=cfg.get('transition_vmax_by_catalogue',{})
    explicit=overrides.get(catalog) if scale=='linear' and cfg['transition_palette']=='catalogue' else None
    maximum=float(vmax) if vmax is not None else (float(explicit) if explicit is not None else transition_upper_bound(largest,cfg))
    saturated=largest>maximum
    allow_saturation=bool(cfg.get('transition_allow_colour_saturation',False)) and explicit is not None and vmax is None
    if not np.isfinite(maximum) or maximum<=1 or (saturated and not allow_saturation):
        raise ValueError('Transition colour limit would clip counts without explicit display permission')
    if scale not in {'linear','log','discrete'}:
        raise ValueError('transition_scale must be linear, log or discrete')
    palette=cfg['transition_palette']
    if palette=='catalogue':
        colour={'carron':'royalblue','chen':'orangered'}.get(catalog)
        if colour is None: raise ValueError(f'Unknown catalogue colour: {catalog}')
        base=LinearSegmentedColormap.from_list(f'{catalog}_white_to_colour',['white',colour])
    elif palette=='named':
        base=plt.get_cmap(cfg['transition_cmap']).copy();colour=cfg['transition_cmap']
    else: raise ValueError('transition_palette must be catalogue or named')
    bounds=_transition_boundaries(cfg,maximum) if scale=='discrete' else None
    if bounds is not None:
        cmap=ListedColormap(base(np.linspace(0,1,len(bounds)-1)))
        norm=BoundaryNorm(bounds,cmap.N,clip=False)
        ticks=bounds
    else:
        cmap=base
        norm=Normalize(0,maximum,clip=True) if scale=='linear' else LogNorm(1,maximum)
        ticks=np.linspace(0,maximum,5) if scale=='linear' else np.unique(np.r_[1.,[10.**k for k in range(1,int(np.ceil(np.log10(maximum))))],maximum])
    if explicit is not None and vmax is None:
        custom=cfg.get('transition_ticks_by_catalogue',{}).get(catalog)
        if custom is not None:
            ticks=np.asarray(custom,float)
            if (len(ticks)<2 or not np.isfinite(ticks).all() or np.any(np.diff(ticks)<=0)
                or ticks[0]!=float(norm.vmin) or ticks[-1]!=maximum):
                raise ValueError('Transition ticks must be increasing and include the colour-range endpoints')
    cmap.set_bad(cfg['transition_zero_colour'])
    metadata={'catalog':catalog,'population':'scope=full and edge_safe=True; sum over parent_status',
        'display_selection':f'm_candidates >= {int(minimum)}; audit/input table unchanged',
        'min_candidates_drawn':int(minimum),'survivor_zero_column_retained':True,
        'colour_scale':scale,'vmin':float(norm.vmin),'vmax':maximum,'cmap':cmap.name,
        'palette':palette,'catalogue_colour':colour,'zero_colour':cfg['transition_zero_colour'],
        'colour_range_population':'displayed candidate rows only; independent per catalogue',
        'colour_comparison_warning':'equal shades across catalogues need not represent equal counts',
        'count_boundaries':bounds.tolist() if bounds is not None else None,
        'zero_cells':'exactly zero nodes, not missing measurements','zero_hatch':cfg['transition_zero_hatch'],
        'colour_saturation_allowed':allow_saturation,'n_saturated_cells':int(np.count_nonzero(counts.to_numpy()>maximum)),
        'maximum_observed_count':largest,'colourbar_ticks':ticks.tolist(),'frame_linewidth_pt':FRAME_LINEWIDTH,
        'counts_are_normalized':False,
        'n_nodes':int(counts.to_numpy().sum()),'n_nodes_all_evaluated':int(raw.to_numpy().sum()),
        'n_nodes_not_drawn':int(raw.loc[raw.index<int(minimum)].to_numpy().sum()),
        'no_scientific_recomputation':True}
    return counts,cmap,norm,bounds,ticks,metadata


def hatch_zero_transition_cells(ax, values, style=None):
    """Mark only exact-zero counts; hatching does not denote missing data here."""
    cfg={**DEFAULTS, **dict(style or {})};a=np.asarray(values)
    result=[]
    for i,j in np.argwhere(a==0):
        patch=Rectangle((j-.5,i-.5),1,1,facecolor=cfg['transition_zero_colour'],
                        edgecolor='.72',hatch=cfg['transition_zero_hatch'],linewidth=0,zorder=1.2)
        patch.set_gid('zero-count-cell');ax.add_patch(patch);result.append(patch)
    return result


def plot_transition_matrix(table: pd.DataFrame, catalog: str, out: Path, *, vmax=None, style: Mapping|None=None) -> list[Path]:
    """Independent catalogue-coloured counts; light grey denotes exactly zero."""
    cfg={**DEFAULTS, **dict(style or {})}
    counts,cmap,norm,bounds,ticks,metadata=transition_display_spec(table,catalog,style=cfg,vmax=vmax)
    if counts.empty:return []
    a=counts.to_numpy(float)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
                         'mathtext.fontset':'stix','pdf.fonttype':42,'font.size':14}):
        fig,ax=plt.subplots(figsize=cfg['transition_figsize'])
        im=ax.imshow(np.ma.masked_equal(a,0),cmap=cmap,norm=norm,origin='lower',aspect='equal',interpolation='nearest')
        hatch_zero_transition_cells(ax,a,cfg)
        ax.set_xticks(range(len(counts.columns)),[str(int(x)) for x in counts.columns])
        ax.set_yticks(range(len(counts.index)),[str(int(x)) for x in counts.index])
        ax.set_xticks(np.arange(-.5,len(counts.columns)),minor=True);ax.set_yticks(np.arange(-.5,len(counts.index)),minor=True)
        ax.grid(False,which='major');ax.grid(which='minor',color='.75',linewidth=.65);ax.tick_params(which='both',length=0)
        ax.set(xlabel='Arms surviving trace cuts',ylabel='Candidate arms',title=CAT_LABEL.get(catalog,catalog))
        ax.set_title(CAT_LABEL.get(catalog,catalog),fontsize=20,pad=13)
        for i,j in np.ndindex(a.shape):
            value=int(a[i,j]);rgb=np.asarray(cmap(norm(value))[:3]);lum=float(rgb@[.2126,.7152,.0722])
            color=('.48' if value==0 else 'white' if lum<.48 else '.06')
            ax.text(j,i,f'{value:,}',ha='center',va='center',fontsize=13,color=color,fontweight='bold' if value else 'normal')
        cb=fig.colorbar(im,ax=ax,fraction=.052,pad=.045,ticks=ticks,boundaries=bounds)
        cb.ax.set_yticklabels([f'{v:,.0f}' for v in ticks]);cb.ax.tick_params(labelsize=10)
        cb.set_label('Number of nodes'+(' (log scale)' if cfg['transition_scale']=='log' else ''),labelpad=11)
        fig.text(.48,.035,'Candidate arms ≥ '+str(cfg['transition_min_candidates'])+'. Hatched grey cells: zero nodes. Independent colour scales.',ha='center',fontsize=11)
        fig.tight_layout(rect=(0,.075,1,1));save_publication_figure(fig,out,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    data=out.with_suffix('.used_values.csv')
    counts.rename_axis(index='m_candidates',columns='m_trace_survivors').stack().rename('n_nodes').reset_index().to_csv(data,index=False)
    meta=out.with_suffix('.metadata.json');meta.write_text(json.dumps(metadata,indent=2))
    return [out,data,meta]
