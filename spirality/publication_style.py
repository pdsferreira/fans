"""One physical frame linewidth for publication figures (points, not pixels)."""
from __future__ import annotations
import math
from matplotlib.figure import Figure

FRAME_LINEWIDTH = 1.1  # Same value as fiducial_results_multiplicity_panel_angle.

def apply_publication_frames(fig: Figure, linewidth: float | None = None) -> None:
    """Standardise axes and colourbar outlines; do not thicken data/grid lines."""
    if linewidth is None:
        linewidth = getattr(fig, "_publication_frame_override", FRAME_LINEWIDTH)
    linewidth = float(linewidth)
    if not math.isfinite(linewidth) or linewidth <= 0:
        raise ValueError("Publication frame linewidth must be finite and positive")
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth)
        cb = getattr(ax, "_colorbar", None)
        if cb is not None:
            cb.outline.set_linewidth(linewidth)
        # Footprint boundaries are explicit rectangles, not Matplotlib spines.
        for patch in ax.patches:
            if patch.get_gid() == "publication-plot-frame":
                patch.set_linewidth(linewidth)
    fig._publication_frame_linewidth = linewidth

def _safe_save_kwargs(kwargs):
    # Rotated mathtext can have a larger glyph bbox than Matplotlib reports.
    # Keep a small blank margin; do not shift or crop any data/axis object.
    result = dict(kwargs)
    if result.get("bbox_inches") == "tight":
        pad = result.get("pad_inches", .1)
        if isinstance(pad, (int, float)):
            result["pad_inches"] = max(.26, float(pad))
    return result


def save_publication_figure(fig: Figure, *args, **kwargs):
    """Save after frame normalisation, leaving every plotted datum untouched."""
    apply_publication_frames(fig)
    return fig.savefig(*args, **_safe_save_kwargs(kwargs))

from contextlib import contextmanager
from functools import wraps
from threading import RLock

_FRAME_LOCK = RLock()

@contextmanager
def publication_frame_scope():
    """Scope frame normalisation to a sequential publication rendering batch.

    Needed for inherited plotting helpers that save internally. The original
    Figure.savefig is restored even when a renderer fails; no PDF is edited.
    Publication CLI rendering is single-threaded. Do not run independent GUI
    renderers concurrently in this process while this context is active.
    """
    with _FRAME_LOCK:
        original = Figure.savefig
        @wraps(original)
        def save_with_uniform_frames(fig, *args, **kwargs):
            apply_publication_frames(fig)
            return original(fig, *args, **_safe_save_kwargs(kwargs))
        Figure.savefig = save_with_uniform_frames
        try:
            yield
        finally:
            Figure.savefig = original
