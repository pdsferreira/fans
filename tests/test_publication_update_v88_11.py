"""Presentation-only fixes and concise-caption regression tests."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from spirality.paper_products import load_style
from spirality import publication_matrices as matrices
from spirality import publication_captions as captions
from test_publication_update import grid_tables


def test_intensity_title_centre_bar_gap_and_removed_footer(tmp_path, monkeypatch):
    seen=[]
    def inspect(fig, output, **kwargs):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        axes=fig.axes[:9]
        centre=(axes[0].get_position().x0+axes[2].get_position().x1)/2
        is_intensity=str(output).endswith('_intensity.pdf')
        if is_intensity:
            assert fig._suptitle.get_position()[0]==pytest.approx(centre)
            title_bbox=fig._suptitle.get_window_extent(renderer)
            assert (title_bbox.x0+title_bbox.x1)/2==pytest.approx(centre*fig.bbox.width)
            cax=fig.axes[9]
            lowest=min(a.xaxis.label.get_window_extent(renderer).y0 for a in axes[-3:])
            gap=(lowest-cax.get_window_extent(renderer).y1)/fig.dpi
            assert gap==pytest.approx(.22)
            assert not any(('Cells:' in t.get_text() or 'Colour intensity encodes' in t.get_text()) for t in fig.texts)
            legend_bbox=fig.legends[0].get_window_extent(renderer)
            assert legend_bbox.y1 < cax.get_tightbbox(renderer).y0
        else:
            assert fig._suptitle.get_position()[0]==.5
            assert any(t.get_text().startswith('Cells:') for t in fig.texts)
        seen.append(is_intensity)
    monkeypatch.setattr(matrices, 'save_publication_figure', inspect)
    tr,st=grid_tables();tr0=tr.copy(deep=True);st0=st.copy(deep=True)
    matrices.plot_delta_s_matrices(tr,st,tmp_path,style=load_style())
    assert sorted(seen)==[False,False,True,True]
    pd.testing.assert_frame_equal(tr,tr0);pd.testing.assert_frame_equal(st,st0)
    for cat in ('chen','carron'):
        meta=json.loads((tmp_path/f'grid27_{cat}_delta_S_matrices_intensity.metadata.json').read_text())
        assert meta['explanatory_notes_drawn'] is False
        assert meta['colourbar_gap_below_xlabel_inches']==pytest.approx(.22)
        a=pd.read_csv(tmp_path/f'grid27_{cat}_delta_S_matrices.used_values.csv')
        b=pd.read_csv(tmp_path/f'grid27_{cat}_delta_S_matrices_intensity.used_values.csv')
        pd.testing.assert_frame_equal(a,b)


def sample_rows():
    return pd.DataFrame([dict(catalog='chen',stratum='m3',plot_support=False,precision_support=False,
                             z_min=.475,z_max=.6,n_fans=4,n_blocks=4,record_type='spirality_display')])


def test_short_caption_preserves_only_required_numerical_details():
    rows=sample_rows();cfg={'paper_spirality_interval':'bootstrap_standard_error'}
    axis=['Carrón: RA=12°, Dec=34°; bootstrap r95=56°','Chen: RA=78°, Dec=10°; bootstrap r95=32°']
    text=captions.results_caption(rows,cfg,axis)
    assert len(text.split())<160
    assert '4 FANS in 4 spatial blocks' in text
    assert 'RA=12°' in text and 'r95=32°' in text
    assert len(text.strip().splitlines())==1
    tex=captions.results_caption(rows,cfg,axis,tex=True)
    assert r'(12^\circ,34^\circ)' in tex and r'r_{95}=32^\circ' in tex
    assert r'95\%' in tex and r"Carr\'on" in tex
    assert '329' not in tex
    assert len(tex.strip().splitlines())==1
    assert captions.CHEN_TEX==r'Same as Fig.~\ref{fig:delta_s_carron}, for Chen.'


def test_caption_generation_skips_missing_figures_and_writes_single_paragraph_lines(tmp_path):
    # File presence is enough for caption generation; no inference or PDF parsing.
    for name in captions.LABELS:
        (tmp_path/f'{name}.pdf').write_bytes(b'not read by caption generator')
    sample_rows().to_csv(tmp_path/f'{captions.RESULTS}.used_values.csv',index=False)
    mp=tmp_path/f'{captions.RESULTS}.metadata.json'
    mp.write_text(json.dumps({'spirality_interval':'bootstrap_standard_error',
                             'axis_legend':['Chen: RA=9°, Dec=11°; bootstrap r95=23°']}))
    before=mp.read_bytes()
    paths,deps=captions.write_caption_files(tmp_path)
    assert len(paths)==6
    bundle=(tmp_path/'captions_fans_revision.tex').read_text()
    assert bundle.count(r'\caption{')==5
    assert bundle.count(r'\label{')==5
    for line in bundle.splitlines():
        if line.startswith(r'\caption{'):
            assert line.endswith('}')
            assert line.count('{')==line.count('}')
    assert r'r_{95}=23^\circ' in bundle
    assert mp.read_bytes()==before
    (tmp_path/f'{captions.SAMPLE}.pdf').unlink()
    paths,deps=captions.write_caption_files(tmp_path,only=[captions.SAMPLE])
    assert not paths
