"""Short publication captions, kept separate from scientific inference.

Each LaTeX paragraph occupies one source line. Numerical axis/omission details
are read from saved plot products; they are never hard-coded for a new run.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable
import pandas as pd

SAMPLE = 'fans_sample_multiplicity_panel'
RESULTS = 'fiducial_results_multiplicity_panel_angle'
CARRON = 'grid27_carron_delta_S_matrices_intensity'
CHEN = 'grid27_chen_delta_S_matrices_intensity'
GRID = 'grid27_projected_panel_angle'
LABELS = {SAMPLE:'fig:fans_redshift', RESULTS:'fig:results',
          CARRON:'fig:delta_s_carron', CHEN:'fig:delta_s_chen', GRID:'fig:robustness'}

SAMPLE_TEX = r"FANS redshift distributions and arm multiplicity. (a) Separately normalized distributions for all accepted FANS and the two- and three-arm subsamples; annotations give their fractions among all accepted FANS in each redshift bin. (b,c) Candidate-to-surviving arm-multiplicity transitions for edge-safe nodes in Carr\'on and Chen, respectively."
CARRON_TEX = r"Spirality contrast $\Delta\mathcal{S}=\mathcal{S}_{\rm proj}(z_4)-\mathcal{S}_{\rm proj}(z_1)$ between the last and first redshift bins for Carr\'on across the 27 constructions and three multiplicity selections. Cells give dimensionless contrasts and paired-bootstrap standard errors; asterisks mark 95\% percentile intervals excluding zero. Colour intensity encodes the signed contrast on a common scale with Fig.~\ref{fig:delta_s_chen}. Hatched cells have insufficient support or an undefined interval, with endpoint counts $N_1,N_4$ shown; dashes denote undefined contrasts. Black outlines mark the fiducial construction."
CARRON_ANGULAR_TEX = r"Spirality contrast $\Delta\mathcal{S}^{(m)}$ [deg] between the last and first redshift bins for Carr\'on across the 27 constructions and three multiplicity selections. The angular contrast is obtained by applying $\arcsin$ in each bin before subtraction. Cells give contrasts and paired-bootstrap standard errors in degrees; asterisks mark 95\% percentile intervals excluding zero, calculated from the angular contrasts in the same bootstrap realizations. Colour intensity encodes the signed contrast on a common scale with Fig.~\ref{fig:delta_s_chen}. Hatched cells have insufficient support or an undefined interval, with endpoint counts $N_1,N_4$ shown; dashes denote undefined contrasts. Black outlines mark the fiducial construction."
CHEN_TEX = r"Same as Fig.~\ref{fig:delta_s_carron}, for Chen."
GRID_TEX = r"Projected statistics across the 27 FANS constructions for Carr\'on (left) and Chen (right). Only the fiducial bootstrap errors are shown. Vertical lines mark the redshift-bin boundaries."


def one_line(text: str) -> str:
    return ' '.join(text.split())


def omitted_records(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty or not {'plot_support','stratum'}.issubset(rows):
        return pd.DataFrame()
    keep = rows.stratum.isin(['m2','m3'])
    support = rows.plot_support.astype(str).str.lower().isin(['true','1','1.0'])
    if 'record_type' in rows:
        keep &= rows.record_type.eq('spirality_display')
    return rows.loc[keep & ~support].copy()


def omission_text(rows: pd.DataFrame, *, tex: bool = False) -> str:
    sentences = []
    for _, r in omitted_records(rows).iterrows():
        name = (r"Carr\'on" if tex else 'Carrón') if r.catalog == 'carron' else 'Chen'
        count = f'{int(r.n_fans)} FANS in {int(r.n_blocks)} spatial blocks'
        bounds = f'{r.z_min:g}<z<{r.z_max:g}'
        if tex:
            bounds = '$'+bounds+'$'
        precision = str(r.get('precision_support', False)).lower() in {'true','1','1.0'}
        why = 'undefined angular interval' if precision else 'insufficient support'
        sentences.append(f'The {name} {int(str(r.stratum)[1:])}-arm point at {bounds} is omitted ({count}; {why}).')
    return ' '.join(sentences)


def spirality_caption(rows: pd.DataFrame, cfg: dict) -> str:
    interval = ('spatial block-bootstrap standard errors' if cfg['paper_spirality_interval']=='bootstrap_standard_error'
                else '16th--84th bootstrap percentile intervals')
    return one_line('Spirality for the full and fixed-multiplicity FANS samples. '
        f'Errors show {interval}, with endpoints transformed separately into degrees. '
        + omission_text(rows))+'\n'


def results_caption(rows: pd.DataFrame, cfg: dict, axis_lines: Iterable[str], *, tex: bool = False) -> str:
    interval = ('spatial block-bootstrap standard errors' if cfg['paper_spirality_interval']=='bootstrap_standard_error'
                else 'the stored bootstrap intervals')
    text = ('Fiducial projected and axis-conditioned handedness (upper rows) and spirality at fixed arm multiplicity (bottom row). '
            f'Errors are {interval}; grey bands show the sign-flip null ranges. '
            + (r'Left-margin triangles and quoted significances refer to $0.1<z<0.6$. ' if tex
               else 'Left-margin triangles and quoted significances refer to 0.1<z<0.6. ')
            + omission_text(rows, tex=tex))
    lines = list(axis_lines)
    if lines:
        if not tex:
            text += ' Fitted axes and 95% bootstrap angular radii: '+'; '.join(lines)+'.'
        else:
            axis_re = re.compile(r'^(.*?): RA=([-+\d.eE]+)°, Dec=([-+\d.eE]+)°; bootstrap r95=([-+\d.eE]+)°$')
            values=[]
            for line in lines:
                m=axis_re.fullmatch(line)
                if m:
                    label,ra,dec,r95=m.groups()
                    label=label.replace('Carrón',r"Carr\'on")
                    values.append(rf'{label}: $(\alpha,\delta)=({ra}^\circ,{dec}^\circ)$, $r_{{95}}={r95}^\circ$')
                else:
                    # Do not invent axis uncertainties when the source lacks them.
                    safe=line.replace('Carrón',r"Carr\'on").replace('%',r'\%').replace('°',r'$^\circ$')
                    values.append(safe)
            text += r' Fitted axes and 95\% bootstrap angular radii: '+'; '.join(values)+'.'
    return one_line(text)+'\n'


def caption_bodies(folder: Path) -> tuple[dict[str,str], list[Path]]:
    folder=Path(folder)
    bodies={SAMPLE:SAMPLE_TEX, CARRON:CARRON_TEX, CHEN:CHEN_TEX, GRID:GRID_TEX}
    deps=[]
    for name in (CARRON,CHEN):
        mp=folder/f'{name}.metadata.json'
        if mp.is_file():
            md=json.loads(mp.read_text());deps.append(mp)
            if md.get('unit')=='deg' and name==CARRON:
                bodies[CARRON]=CARRON_ANGULAR_TEX
    mp=folder/f'{RESULTS}.metadata.json';vp=folder/f'{RESULTS}.used_values.csv'
    if mp.is_file() and vp.is_file():
        meta=json.loads(mp.read_text());rows=pd.read_csv(vp);deps.extend([mp,vp])
        cfg={'paper_spirality_interval':meta['spirality_interval']}
        bodies[RESULTS]=results_caption(rows,cfg,meta.get('axis_legend',[]),tex=True).strip()
    return bodies,deps


def write_caption_files(folder: Path, *, only: Iterable[str] | None = None) -> tuple[list[Path],list[Path]]:
    """Write one-line LaTeX captions only for existing managed figure products."""
    folder=Path(folder);bodies,deps=caption_bodies(folder)
    names=list(only) if only is not None else list(LABELS)
    paths=[];blocks=[]
    for name in names:
        if name not in bodies or not (folder/f'{name}.pdf').is_file():
            continue
        body=one_line(bodies[name])
        block=rf'\caption{{{body}}}'+'\n'+rf'\label{{{LABELS[name]}}}'+'\n'
        p=folder/f'{name}.caption.tex';p.write_text(block,encoding='utf-8');paths.append(p)
        blocks.append('% '+name+'.pdf\n'+block)
    if only is None and blocks:
        p=folder/'captions_fans_revision.tex'
        p.write_text('% Insert each caption after its matching includegraphics command.\n\n'+'\n'.join(blocks),encoding='utf-8')
        paths.append(p)
    return paths,deps
