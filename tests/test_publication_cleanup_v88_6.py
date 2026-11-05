"""Presentation-only regression tests; all inputs are synthetic."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from test_native_panels_v88_5 import compact, _prepare_extrinsic
from spirality import publication_panels as panels
from spirality.paper_products import load_style


def _supported_fixture(folder):
    """Two fully sampled classes, one deliberately unsupported last m3 bin."""
    for cat in ('carron','chen'):
        p=folder/f'{cat}_multiplicity_statistics.csv';d=pd.read_csv(p)
        d.loc[d.scope.eq('zbin'), 'precision_support']=True
        if cat=='chen':
            mask=d.scope.eq('zbin')&d.stratum.eq('m3')&d.zbin_id.eq(3)
            d.loc[mask,['precision_support','n_fans','n_blocks']]=[False,4,4]
        d.to_csv(p,index=False)
        p=folder/f'{cat}_multiplicity_composition.csv';c=pd.read_csv(p)
        c['supported']=True
        c['S_fixed_composition']=.20;c['S_fixed_composition_err']=.01
        c.to_csv(p,index=False)


def test_fixed_composition_kept_in_data_but_not_requested_for_default_figure(compact):
    folder,_=compact;_supported_fixture(folder)
    source=folder/'carron_multiplicity_composition.csv';before=source.read_bytes()
    rows,_=panels.spirality_display_rows(folder,'carron',load_style())
    fixed=rows[rows.stratum.eq('fixed_composition')]
    assert len(fixed)==4 and fixed.supported_interval.all()
    assert not fixed.plot_support.any() and not fixed.display_requested.any()
    assert set(fixed.omission_reason)=={'not_requested_for_figure'}
    assert source.read_bytes()==before
    cfg=load_style();cfg['paper_spirality_show_fixed_composition']=True
    optional,_=panels.spirality_display_rows(folder,'carron',cfg)
    assert optional.loc[optional.stratum.eq('fixed_composition'),'plot_support'].all()


def test_only_all_multiplicities_has_fill_or_connecting_curve(compact):
    folder,_=compact;_supported_fixture(folder);cfg=load_style()
    rows,_=panels.spirality_display_rows(folder,'chen',cfg)
    fig,ax=plt.subplots();handles=panels.draw_spirality_axis(ax,rows,'chen',cfg)
    assert [c.get_gid() for c in ax.collections]==['spirality-band-all']
    ids=[line.get_gid() for line in ax.lines]
    assert ids.count('spirality-curve-all')==1
    for pop,n in [('m2',4),('m3',3)]:
        assert ids.count(f'spirality-yerr-{pop}')==n
        assert ids.count(f'spirality-xerr-{pop}')==n
        markers=[l for l in ax.lines if l.get_gid()==f'spirality-markers-{pop}']
        assert sum(len(l.get_xdata()) for l in markers)==n
        assert all(l.get_linestyle()=='None' for l in markers)
        assert f'spirality-curve-{pop}' not in ids
        assert f'spirality-boundary-{pop}' not in ids
    assert [h.get_label() for h in handles]==['All multiplicities','2 arms','3 arms']
    assert [h.get_linestyle() for h in handles[1:]]==['None','None']
    plt.close(fig)


def test_requested_19_1_ceiling_keeps_data_and_expands_only_when_needed():
    cfg=load_style()
    fig,axes=plt.subplots(1,2)
    for ax in axes:ax.plot([.1,.6],[12,18.1])
    limits=panels._spirality_limits(axes,cfg)
    assert all(lim[1]==19.1 for lim in limits)
    axes[0].plot([.2,.4],[15,22])
    limits=panels._spirality_limits(axes,cfg)
    assert all(lim[1]>22 for lim in limits)
    plt.close(fig)


def test_results_have_no_bin_lines_no_footnotes_and_lowered_legend(compact,monkeypatch):
    folder,raw=compact;ds,prepared=_prepare_extrinsic(folder,raw);_supported_fixture(folder)
    cfg=load_style();seen={};original=panels._save
    def inspect(fig,out,values,metadata,axes,cfg):
        arr=np.asarray(axes).ravel()
        for ax in arr:
            for line in ax.lines:
                x=np.asarray(line.get_xdata(),float)
                if len(x)==2 and x[0]==x[1]:
                    assert not any(np.isclose(x[0],edge,rtol=0,atol=1e-12) for edge in (.225,.35,.475))
        assert not fig.texts
        assert len(fig.legends)==1
        fig.canvas.draw()
        legend=fig.legends[0]
        anchor=legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        gap=(axes[2,0].get_position().y0-anchor.y0)*fig.get_figheight()
        assert np.isclose(gap,.80,atol=1e-10)
        seen['inspected']=True
        return original(fig,out,values,metadata,axes,cfg)
    monkeypatch.setattr(panels,'_save',inspect)
    paths,_=panels.plot_results_panel(folder,cfg,datasets=ds,prepared=prepared)
    assert seen['inspected'] and len(paths)==4
    meta=json.loads((folder/'fiducial_results_multiplicity_panel_angle.metadata.json').read_text())
    assert meta['bin_boundaries_drawn'] is False
    assert meta['fixed_composition_shown'] is False
    assert meta['axis_summary_text_drawn'] is False
    assert meta['omission_text_drawn'] is False
    caption=(folder/'fiducial_results_multiplicity_panel_angle.caption.txt').read_text()
    assert '4 FANS in 4 spatial blocks' in caption
    assert 'RA=20°' in caption and 'r95=40°' in caption
    assert 'fixed-composition control' not in caption


def test_standalones_have_caption_not_note_and_unchanged_science(compact,monkeypatch):
    folder,_=compact;_supported_fixture(folder)
    before={p.name:p.read_bytes() for p in folder.glob('*.csv')}
    original=panels._save;seen=[]
    def inspect(fig,*a,**kw):
        assert not fig.texts;seen.append(True)
        return original(fig,*a,**kw)
    monkeypatch.setattr(panels,'_save',inspect)
    paths,_,_=panels.plot_spirality_standalone(folder,load_style())
    assert len(paths)==8 and len(seen)==2
    for name,old in before.items():assert (folder/name).read_bytes()==old
    for cat in ('carron','chen'):
        meta=json.loads((folder/f'{cat}_S_fixed_multiplicity.metadata.json').read_text())
        assert meta['y_limits'][1]>=19.1  # Synthetic inputs can exceed the requested ceiling.
        assert len(meta['data_axes_geometry'])==1
        assert meta['data_axes_geometry'][0]['height_over_width']==pytest.approx(.55)


def test_sample_distribution_keeps_bin_boundaries():
    fig,ax=plt.subplots()
    panels._finish_z_axes(ax,np.array([.1,.225,.35,.475,.6]))
    xs=[np.asarray(line.get_xdata(),float) for line in ax.lines]
    assert all(any(np.allclose(x,[edge,edge]) for x in xs) for edge in (.225,.35,.475))
    plt.close(fig)


def test_old_style_json_inherits_new_presentation_defaults(tmp_path):
    p=tmp_path/'old.json';p.write_text(json.dumps({'paper_panel_box_aspect':.55,'paper_spirality_interval':'bootstrap_standard_error'}))
    cfg=load_style(p)
    assert cfg['paper_spirality_show_fixed_composition'] is False
    assert cfg['paper_results_show_bin_edges'] is False
    assert cfg['paper_spirality_upper_limit_deg']==19.1
