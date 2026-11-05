"""Sample-only finishing: exact fonts/limits, local frames, no unrelated writes."""
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from spirality.paper_products import load_style, Bundle, _commit_bundle
from spirality.publication_panels import (
    sample_settings, sample_panel_only_settings, panel_style,
    draw_distribution_axis, draw_transition_axis, plot_sample_panel,
)
from spirality.publication_style import (
    FRAME_LINEWIDTH, save_publication_figure, publication_frame_scope,
)
from spirality.publication_refresh import refresh_paper_folder
from spirality.publication_redshift import prepare_redshift_cache, binned_multiplicity
from spirality.provenance import file_digest
from test_native_panels_v88_5 import compact
from test_publication_update_v88_7 import transition_fixture


def _curves_annotations():
    edges=np.array([.1,.225,.35,.475,.6])
    curves=pd.DataFrame([dict(catalog=c,stratum=p,z=z,density=y)
        for c in ['chen','carron'] for p in ['all','m2','m3']
        for z,y in zip([.1,.3,.6],[2.,7.,1.])])
    ann=pd.DataFrame([dict(catalog=c,z_min=edges[i],z_max=edges[i+1],
        text='2A:75.4%/3A:24.4%') for c in ['chen','carron'] for i in range(4)])
    return curves,ann,edges


def test_only_composite_settings_get_requested_increments():
    style=load_style();before=json.dumps(style,sort_keys=True)
    baseline=sample_settings(style);new=sample_panel_only_settings(style)
    assert new['paper_panel_tick_size']==baseline['paper_panel_tick_size']
    assert new['paper_panel_label_size']==baseline['paper_panel_label_size']
    assert new['paper_fraction_annotation_size']==pytest.approx(1.045*baseline['paper_fraction_annotation_size'])
    assert new['_sample_cell_size']==baseline['paper_panel_tick_size']
    assert new['_paper_frame_linewidth']==pytest.approx(1.54)
    assert new['_density_upper_limit']==11.5
    assert new['paper_sample_panel_letter_increment']==1.2
    assert new['paper_sample_panel_legend_increment']==1.2
    assert '_density_upper_limit' not in baseline
    assert '_paper_frame_linewidth' not in baseline
    assert json.dumps(style,sort_keys=True)==before


@pytest.mark.parametrize('key,value',[
    ('paper_sample_panel_letter_increment',0),
    ('paper_sample_panel_frame_increment',-1),
    ('paper_sample_panel_density_upper_limit',np.nan),
    ('paper_sample_panel_legend_extra_gap_inches',-1),
    ('paper_sample_panel_cells_match_ticks','yes'),
])
def test_invalid_panel_only_controls_rejected(key,value):
    s=load_style();s[key]=value
    with pytest.raises(ValueError):sample_panel_only_settings(s)


def test_density_origin_is_kept_but_zero_numeral_removed():
    c,a,e=_curves_annotations();cfg=sample_panel_only_settings(load_style())
    with panel_style(cfg):
        fig,ax=plt.subplots(figsize=(13.7,8))
        before=c.copy(deep=True);texts=draw_distribution_axis(ax,c,a,e,cfg);fig.canvas.draw()
        assert ax.get_ylim()==(0,11.5)
        assert ax.yaxis.get_major_formatter()(0,0)==''
        assert ax.yaxis.get_major_formatter()(2,1)=='2'
        assert all(t.get_fontsize()==pytest.approx(20.8725*.95) for t,_,_ in texts)
        assert len(ax.lines)==9
        assert not any(x.get_visible() for x in ax.get_xgridlines()+ax.get_ygridlines())
        pd.testing.assert_frame_equal(c,before)
        plt.close(fig)


def test_above_range_values_are_preserved_and_warned():
    c,a,e=_curves_annotations();c.loc[c.index[0],'density']=12
    cfg=sample_panel_only_settings(load_style())
    with panel_style(cfg):
        fig,ax=plt.subplots(figsize=(13.7,8))
        with pytest.warns(RuntimeWarning,match='upper limit'):
            draw_distribution_axis(ax,c,a,e,cfg)
        assert c.density.max()==12 and ax.get_ylim()==(0,11.5)
        assert ax.lines[0].get_ydata().max()==12 or ax.lines[3].get_ydata().max()==12
        plt.close(fig)


def test_cells_and_axis_tick_numerals_have_identical_size():
    cfg=sample_panel_only_settings(load_style())
    with panel_style(cfg):
        fig,ax=plt.subplots()
        _,data,_=draw_transition_axis(ax,transition_fixture(),'chen',cfg)
        fig.canvas.draw()
        ticks=[t for t in ax.get_xticklabels()+ax.get_yticklabels() if t.get_text()]
        sizes=[t.get_fontsize() for t in ticks]
        assert sizes and all(v==pytest.approx(26.4) for v in sizes)
        assert all(t.get_fontsize()==pytest.approx(26.4) for t in ax.texts)
        assert data.n_nodes.max()==8000  # raw count is never clipped by its colour scale
        plt.close(fig)


def test_sample_frame_override_survives_scoped_save_and_does_not_leak(tmp_path,monkeypatch):
    seen=[]
    def capture(fig,*args,**kwargs):
        seen.append([[s.get_linewidth() for s in a.spines.values()] for a in fig.axes])
    monkeypatch.setattr(Figure,'savefig',capture)
    with publication_frame_scope():
        fig,ax=plt.subplots();im=ax.imshow([[1,2],[3,4]]);fig.colorbar(im,ax=ax)
        fig._publication_frame_override=1.54
        save_publication_figure(fig,tmp_path/'sample.pdf');plt.close(fig)
        fig,ax=plt.subplots();save_publication_figure(fig,tmp_path/'unchanged.pdf');plt.close(fig)
    assert all(x==pytest.approx(1.54) for row in seen[0] for x in row)
    assert all(x==pytest.approx(1.1) for row in seen[1] for x in row)
    assert FRAME_LINEWIDTH==1.1


def _managed(compact,tmp_path,with_cache):
    folder,raw=compact
    if with_cache:
        b,e,_=binned_multiplicity(folder);prepare_redshift_cache(folder,b,e,raw)
    # Unrelated files are deliberately opaque: target-only must never read/render them.
    (folder/'other_figure.pdf').write_bytes(b'unchanged unrelated PDF bytes')
    (folder/'other.caption.txt').write_text('Do not touch this caption.')
    stage=tmp_path/'bundle_stage';stage.mkdir()
    bundle=Bundle(stage)
    for p in folder.iterdir():bundle.add(p)
    out=tmp_path/'managed'
    _commit_bundle(bundle,out,{},load_style(),{})
    (out/'author_manual_panel.pdf').write_bytes(b'user owned panel')
    return out,raw


def test_targeted_refresh_preserves_every_unrelated_product(compact,tmp_path,monkeypatch):
    out,raw=_managed(compact,tmp_path,True)
    before={p.name:file_digest(p) for p in out.iterdir()}
    import spirality.publication_refresh as r
    monkeypatch.setattr(r,'refresh_bundle_appearance',lambda *a,**k:pytest.fail('Full refresh must not run'))
    import spirality.publication_panels as p
    monkeypatch.setattr(p,'plot_results_panel',lambda *a,**k:pytest.fail('Results panel must not run'))
    monkeypatch.setattr(p,'plot_spirality_standalone',lambda *a,**k:pytest.fail('Standalone must not run'))
    result=refresh_paper_folder(out,only_sample_panel=True)
    exempt={'manifest.json','manifest.csv','paper_sources.json','README.md'}
    assert all(file_digest(out/name)==digest for name,digest in before.items() if name not in exempt)
    assert not (out/'fans_redshift_distribution_total_multiplicity_annotated.pdf').exists()
    meta=json.loads((out/'fans_sample_multiplicity_panel.metadata.json').read_text())
    assert meta['frame_linewidth_pt']==pytest.approx(1.54)
    assert meta['density_y_limits']==[0,11.5]
    assert meta['subplot_letter_font_size_pt']==pytest.approx(25.74)
    assert meta['line_legend_font_size_pt']==pytest.approx(22.77)
    assert meta['transition_cell_font_size_pt']==pytest.approx(meta['axis_tick_font_size_pt'])
    assert result['metadata']['appearance_update_88_9']['scope']=='fans_sample_multiplicity_panel only'
    for entry in result['files']:
        assert file_digest(out/entry['file'])==entry['sha256']


def test_targeted_missing_individual_z_fails_without_writing(compact,tmp_path):
    out,_=_managed(compact,tmp_path,False)
    before={p.name:file_digest(p) for p in out.iterdir()}
    with pytest.raises(FileNotFoundError,match='requires the validated'):
        refresh_paper_folder(out,only_sample_panel=True)
    assert before=={p.name:file_digest(p) for p in out.iterdir()}


def test_targeted_refresh_can_build_cache_from_valid_original_nodes(compact,tmp_path):
    out,raw=_managed(compact,tmp_path,False)
    result=refresh_paper_folder(out,fiducial_run=raw,only_sample_panel=True)
    assert (out/'fans_redshift_multiplicity_nodes.csv').exists()
    assert (out/'fans_sample_multiplicity_panel.pdf').exists()
    assert not (out/'fans_redshift_distribution_total_multiplicity_annotated.pdf').exists()
    assert result['metadata']['appearance_update_88_9']['redshift_multiplicity_kde']=='validated_fiducial_individual_redshifts'


def test_targeted_modified_managed_file_stops_before_writing(compact,tmp_path):
    out,raw=_managed(compact,tmp_path,True)
    (out/'other_figure.pdf').write_bytes(b'author modified file')
    before={p.name:file_digest(p) for p in out.iterdir()}
    with pytest.raises(ValueError,match='Missing/modified managed product'):
        refresh_paper_folder(out,only_sample_panel=True)
    assert before=={p.name:file_digest(p) for p in out.iterdir()}
