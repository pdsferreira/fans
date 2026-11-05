from .publication_style import save_publication_figure, FRAME_LINEWIDTH
"""Footprint rendering adapted from the author-provided plotting script.

The calibrated crops, RA/Dec labels, common greyscale and Galactic N/S split
are retained. Called only for the fiducial sample; never during a grid rebuild.
"""
from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.text import Text
# Optional astronomy imports are local to the footprint builder below.
hp = None
def use_times_or_fallback():
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    ]

    for p in candidates:
        if os.path.exists(p):
            fm.fontManager.addfont(p)

    plt.rcParams["font.family"] = [
        "Times New Roman",
        "Liberation Serif",
        "Noto Serif",
        "DejaVu Serif",
        "serif",
    ]

def make_truncated_greys(
    name="Greys_not_black",
    base_cmap="Greys",
    gray_min=0.12,
    gray_max=0.76,
    n=256,
):
    base = plt.get_cmap(base_cmap)
    colors = base(np.linspace(gray_min, gray_max, n))
    return LinearSegmentedColormap.from_list(name, colors)

def make_fans_density_map(
    nodes,
    nside=64,
    nest=False,
    density_per_deg2=False,
    mask_zero_pixels=True,
):
    ra = np.asarray(nodes["ra"], dtype=float)
    dec = np.asarray(nodes["dec"], dtype=float)

    good = np.isfinite(ra) & np.isfinite(dec)
    ra = np.mod(ra[good], 360.0)
    dec = dec[good]

    npix = hp.nside2npix(nside)

    if len(ra) == 0:
        m = np.zeros(npix, dtype=float)
        m[:] = hp.UNSEEN
        return m, np.zeros(npix, dtype=float), r"$N_{\rm FANS}$"

    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(nside, theta, phi, nest=nest)

    counts = np.bincount(pix, minlength=npix).astype(float)

    if density_per_deg2:
        pix_area_deg2 = hp.nside2pixarea(nside, degrees=True)
        m = counts / pix_area_deg2
        cbar_label = r"$N_{\rm FANS}\,{\rm deg}^{-2}$"
    else:
        m = counts.copy()
        cbar_label = r"$N_{\rm FANS}$"

    m_plot = m.copy()

    if mask_zero_pixels:
        m_plot[counts == 0] = hp.UNSEEN

    return m_plot, counts, cbar_label

def style_healpy_colorbar(
    fig,
    label,
    labelsize=26,
    ticksize=20,
    tick_width=1.5,
    tick_length=5,
):
    """
    Healpy colorbar is usually the last axis of the figure.
    """
    if len(fig.axes) < 2:
        return

    cax = fig.axes[-1]

    cax.tick_params(
        axis="both",
        which="both",
        labelsize=ticksize,
        width=tick_width,
        length=tick_length,
        direction="in",
    )

    cax.set_xlabel(label, fontsize=labelsize, labelpad=8)

    for spine in cax.spines.values():
        spine.set_linewidth(1.3)

def remove_equatorial_text(fig):
    for txt in fig.findobj(Text):
        if "equatorial" in txt.get_text().strip().lower():
            txt.set_visible(False)

def force_graticule_black(ax, lw=1.4, alpha=1.0):
    for line in ax.lines:
        line.set_color("black")
        line.set_linewidth(lw)
        line.set_alpha(alpha)

def add_manual_labels_projected(
    ax,
    xlim,
    ylim,
    title,
    ra_label_xy=None,
    dec_label_xy=None,
    coord_color="black",
    coord_fontsize=22,
    axis_label_fontsize=25,
    title_fontsize=26,
    ra_axis_label_pos=(0.96, -0.01),
    dec_axis_label_pos=(1.047, 0.09),
    title_pos=(0.50, 1.055),
    frame_lw=2.0,
):
    """
    Manual labels in projected coordinates, following the style/crops
    already calibrated in your footprint example.
    """

    if ra_label_xy is not None:
        for ra_val, (x, y) in ra_label_xy.items():
            ax.text(
                x,
                y,
                rf"{int(ra_val)}°",
                ha="center",
                va="top",
                fontsize=coord_fontsize,
                color=coord_color,
                zorder=20,
                clip_on=False,
            )

    if dec_label_xy is not None:
        for dec_val, (x, y) in dec_label_xy.items():
            ax.text(
                x,
                y,
                rf"{int(dec_val)}°",
                ha="left",
                va="center",
                fontsize=coord_fontsize,
                color=coord_color,
                zorder=20,
                clip_on=False,
            )

    ax.text(
        ra_axis_label_pos[0],
        ra_axis_label_pos[1],
        "RA",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=axis_label_fontsize,
        color="black",
        clip_on=False,
    )

    ax.text(
        dec_axis_label_pos[0],
        dec_axis_label_pos[1],
        "Dec.",
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=axis_label_fontsize,
        color="black",
        clip_on=False,
    )

    ax.text(
        title_pos[0],
        title_pos[1],
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=title_fontsize,
        color="black",
        clip_on=False,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    rect = Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        linewidth=frame_lw,
        edgecolor="black",
        facecolor="none",
        zorder=30,
    )
    rect.set_gid("publication-plot-frame")
    ax.add_patch(rect)

def plot_fans_density_crop(
    nodes,
    output,
    title,
    ra_center,
    figsize,
    nside=64,
    nest=False,
    density_per_deg2=False,
    gray_min=0.12,
    gray_max=0.76,
    vmin=0.0,
    vmax=None,
    vmax_percentile=99.0,
    xlim=(-0.85, 0.70),
    ylim=(-0.05, 0.85),
    dpar=30,
    dmer=30,
    graticule_lw=1.4,
    graticule_alpha=1.0,
    ra_label_xy=None,
    dec_label_xy=None,
    coord_fontsize=22,
    axis_label_fontsize=25,
    title_fontsize=26,
    ra_axis_label_pos=(0.96, -0.01),
    dec_axis_label_pos=(1.047, 0.09),
    title_pos=(0.50, 1.055),
    cbar_labelsize=26,
    cbar_ticksize=20,
):
    """
    Individual cropped healpy Mollweide density plot.
    """

    m_plot, counts, cbar_label = make_fans_density_map(
        nodes,
        nside=nside,
        nest=nest,
        density_per_deg2=density_per_deg2,
        mask_zero_pixels=True,
    )

    valid = m_plot[m_plot != hp.UNSEEN]
    valid = valid[np.isfinite(valid)]

    if valid.size == 0:
        raise ValueError(f"Nenhum pixel válido encontrado para plotar: {title}")

    if vmax is None:
        positive = valid[valid > 0]
        if positive.size > 0:
            vmax = np.nanpercentile(positive, vmax_percentile)
        else:
            vmax = 1.0

    if vmax <= vmin:
        vmax = vmin + 1.0

    cmap = make_truncated_greys(
        gray_min=gray_min,
        gray_max=gray_max,
    )

    fig = plt.figure(figsize=figsize)

    hp.mollview(
        m_plot,
        fig=fig.number,
        hold=True,
        title="",
        coord="C",
        rot=(ra_center, 0, 0),
        flip="astro",
        cmap=cmap,
        unit="",
        badcolor="white",
        nest=nest,
        min=vmin,
        max=vmax,
        cbar=True,
        notext=True,
    )

    map_ax = fig.get_axes()[0]
    plt.sca(map_ax)

    hp.graticule(
        dpar=dpar,
        dmer=dmer,
        lw=graticule_lw,
        color="black",
        alpha=graticule_alpha,
    )

    force_graticule_black(
        map_ax,
        lw=graticule_lw,
        alpha=graticule_alpha,
    )

    remove_equatorial_text(fig)

    add_manual_labels_projected(
        map_ax,
        xlim=xlim,
        ylim=ylim,
        title=title,
        ra_label_xy=ra_label_xy,
        dec_label_xy=dec_label_xy,
        coord_fontsize=coord_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        title_fontsize=title_fontsize,
        ra_axis_label_pos=ra_axis_label_pos,
        dec_axis_label_pos=dec_axis_label_pos,
        title_pos=title_pos,
    )

    style_healpy_colorbar(
        fig,
        label=cbar_label,
        labelsize=cbar_labelsize,
        ticksize=cbar_ticksize,
    )

    save_publication_figure(fig,output, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print("[write]", output)

def build_footprints(nodes_by_catalog, outdir, *, nside=64, density_per_deg2=False, font_size=24):
    """Render the three author-calibrated equatorial crops, split in Galactic b.

    Zero-count pixels are masked, not an estimate of the survey completeness
    mask. An empty region is recorded as absent rather than fabricating a map.
    """
    global hp
    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if not hp.isnsideok(nside):
        raise ValueError(f"Invalid HEALPix NSIDE: {nside}")
    outdir = Path(outdir)
    north_ra = {90:(.547,.902),120:(.370,.902),150:(.190,.902),180:(.0135,.902),
                210:(-.165,.902),240:(-.340,.902),270:(-.515,.902),300:(-.695,.902)}
    north_dec = {0:(-.910,0),30:(-.9411,.405),60:(-.9411,.765)}
    south_ra = {330:(.304,.499),0:(.0075,.499),30:(-.285,.499)}
    south_dec = {0:(-.530,0),30:(-.550,.405)}
    regions = {}
    counts = []
    for cat, data in nodes_by_catalog.items():
        sky=SkyCoord(ra=np.asarray(data['ra'])*u.deg,dec=np.asarray(data['dec'])*u.deg,frame='icrs')
        b=sky.galactic.b.deg
        for reg,mask in [('galnorth',b>=0),('galsouth',b<0)]:
            subset={k:np.asarray(v)[mask] for k,v in data.items() if k in {'ra','dec','z'}}
            regions[(cat,reg)]=subset
            counts.append({'catalog':cat,'region':reg,'n_fans':int(mask.sum())})
    selected=[('carron','galnorth'),('carron','galsouth'),('chen','galnorth')]
    vals=[]
    for key in selected:
        m,_,_=make_fans_density_map(regions[key],nside=nside,density_per_deg2=density_per_deg2)
        good=m[np.isfinite(m)&(m!=hp.UNSEEN)&(m>0)]
        if len(good): vals.append(good)
    vmax=max(1.,float(np.percentile(np.concatenate(vals),99))) if vals else 1.
    paths=[]; skipped=[]
    with plt.rc_context():
        use_times_or_fallback()
        plt.rcParams.update({'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'font.size':font_size})
        for cat,reg in selected:
            if not len(regions[(cat,reg)]['z']):
                skipped.append(f'{cat}_{reg}: no accepted nodes in region'); continue
            north=reg=='galnorth'
            path=outdir/f'fans_density_{cat}_{reg}_crop.pdf'
            title=('Carrón DR16' if cat=='carron' else 'Chen DR12')+(' North FANS' if north else ' South FANS')
            plot_fans_density_crop(regions[(cat,reg)],output=path,title=title,ra_center=180. if north else 0.,
                figsize=(10.,7.) if north else (8.5,6.3),nside=nside,density_per_deg2=density_per_deg2,vmax=vmax,
                xlim=(-.85,.70) if north else (-.49,.465),ylim=(-.078084,.878084) if north else (-.132062,.457062),
                ra_label_xy=north_ra if north else south_ra,dec_label_xy=north_dec if north else south_dec,
                coord_fontsize=font_size-2,axis_label_fontsize=font_size+1,title_fontsize=font_size+1,
                cbar_labelsize=font_size-3,cbar_ticksize=font_size-7)
            paths.append(path)
    return paths,{'nside':nside,'density_per_deg2':density_per_deg2,'common_vmax':vmax,
                  'galactic_split_b_deg':0.,'counts':counts,'skipped':skipped,
                  'selection':'accepted revised FANS; one count per node',
                  'mask_note':'zero-count pixels masked; not a completeness map'}
