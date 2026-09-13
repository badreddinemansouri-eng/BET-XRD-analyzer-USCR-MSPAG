"""
EXPORT UTILITIES FOR PUBLICATION FIGURES
========================================================================
Centralised handling of figure export across formats and downloads.

Provides:
    - fig_to_bytes(fig, fmt, dpi)              -> bytes
    - fig_to_download(fig, label, filename,...) -> Streamlit button
    - figs_to_zip(figures_dict, fmt, dpi)      -> ZIP bytes
    - panel_export_buttons(fig, base_filename) -> Streamlit buttons row
    - save_all_to_zip(...)                     -> Streamlit button
    - available_export_formats()               -> list of (label, fmt, dpi)

All functions are deterministic and safe with empty figures.

References:
    Matplotlib documentation, savefig section.
========================================================================
"""

import io
import zipfile
import logging
import re
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


# ============================================================================
# FORMAT REGISTRY
# ============================================================================
FORMAT_SPECS = [
    ("PNG (600 DPI)",   "png",  600),
    ("PDF (Vector)",    "pdf",  None),
    ("SVG (Vector)",    "svg",  None),
    ("TIFF (1200 DPI)", "tiff", 1200),
]

FORMAT_MIME = {
    "png":  "image/png",
    "pdf":  "application/pdf",
    "svg":  "image/svg+xml",
    "tiff": "image/tiff",
    "zip":  "application/zip",
}


def available_export_formats() -> List[Tuple[str, str, Optional[int]]]:
    """Return the list of (label, format, dpi) tuples."""
    return list(FORMAT_SPECS)


def _sanitize_filename(name: str) -> str:
    """Remove characters that are invalid in file names."""
    name = re.sub(r'[^\w\.\-]+', '_', name)
    return name.strip('_') or "figure"


# ============================================================================
# LOW-LEVEL
# ============================================================================
def fig_to_bytes(fig, fmt: str = "png", dpi: Optional[int] = 600) -> bytes:
    """
    Serialize a matplotlib figure to bytes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    fmt : 'png', 'pdf', 'svg', 'tiff'
    dpi : int or None. For vector formats, None is required.

    Returns
    -------
    bytes of the rendered figure. Empty bytes on failure.
    """
    if fig is None:
        return b""

    fmt = fmt.lower()
    if fmt not in FORMAT_MIME:
        fmt = "png"

    buf = io.BytesIO()
    try:
        if fmt in ("pdf", "svg"):
            fig.savefig(buf, format=fmt, bbox_inches='tight')
        else:
            fig.savefig(buf, format=fmt, dpi=dpi or 600,
                        bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Figure export to %s failed: %s", fmt, exc)
        return b""


def close_fig_safely(fig):
    """Close a matplotlib figure and release memory."""
    try:
        import matplotlib.pyplot as plt
        if fig is not None:
            plt.close(fig)
    except Exception:
        pass


# ============================================================================
# STREAMLIT HELPERS
# ============================================================================
def fig_to_download(fig,
                    label: str,
                    filename: str,
                    fmt: str = "png",
                    dpi: Optional[int] = 600,
                    use_container_width: bool = True,
                    key: Optional[str] = None):
    """
    Render a Streamlit download button for a single figure.
    Returns True if the button was rendered, False otherwise.
    """
    try:
        import streamlit as st
    except ImportError:
        logger.warning("Streamlit not available. fig_to_download is a no-op.")
        return False

    data = fig_to_bytes(fig, fmt=fmt, dpi=dpi)
    if not data:
        st.warning(f"Could not export {label} as {fmt}.")
        return False

    safe_name = _sanitize_filename(filename)
    mime = FORMAT_MIME.get(fmt, "application/octet-stream")

    st.download_button(
        label=label,
        data=data,
        file_name=safe_name,
        mime=mime,
        use_container_width=use_container_width,
        key=key,
    )
    return True


def panel_export_buttons(fig,
                         base_filename: str,
                         key_prefix: str = "panel",
                         columns: int = 4):
    """
    Render a row of Streamlit buttons, one per export format, for a
    single figure.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    cols = st.columns(columns)
    for idx, (label, fmt, dpi) in enumerate(FORMAT_SPECS):
        with cols[idx % columns]:
            fig_to_download(
                fig=fig,
                label=label,
                filename=f"{base_filename}.{fmt}",
                fmt=fmt,
                dpi=dpi,
                use_container_width=True,
                key=f"{key_prefix}_{base_filename}_{fmt}",
            )


def save_all_to_zip(figures: Dict[str, object],
                    zip_filename: str = "all_figures.zip",
                    fmt: str = "png",
                    dpi: Optional[int] = 600,
                    key: Optional[str] = None):
    """
    Render a single Streamlit download button that produces a ZIP
    containing every figure in `figures`, one file per entry.

    Parameters
    ----------
    figures : dict
        Mapping {figure_name: matplotlib.figure.Figure}
    """
    try:
        import streamlit as st
    except ImportError:
        return

    data = figs_to_zip(figures, fmt=fmt, dpi=dpi)
    if not data:
        st.warning("No figures available for ZIP export.")
        return

    st.download_button(
        label=f"Download all figures (ZIP, {fmt.upper()})",
        data=data,
        file_name=_sanitize_filename(zip_filename),
        mime="application/zip",
        use_container_width=True,
        key=key,
    )


def figs_to_zip(figures: Dict[str, object],
                fmt: str = "png",
                dpi: Optional[int] = 600) -> bytes:
    """
    Serialize a dict of figures into a ZIP archive.

    Parameters
    ----------
    figures : dict
        Mapping {figure_name: matplotlib.figure.Figure}

    Returns
    -------
    bytes of the ZIP archive, or empty bytes on failure.
    """
    if not figures:
        return b""

    buf = io.BytesIO()
    try:
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            used_names = {}
            for name, fig in figures.items():
                if fig is None:
                    continue
                data = fig_to_bytes(fig, fmt=fmt, dpi=dpi)
                if not data:
                    continue
                safe_base = _sanitize_filename(name)
                counter = used_names.get(safe_base, 0)
                used_names[safe_base] = counter + 1
                suffix = "" if counter == 0 else f"_{counter}"
                arcname = f"{safe_base}{suffix}.{fmt}"
                zf.writestr(arcname, data)
        buf.seek(0)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("ZIP packaging failed: %s", exc)
        return b""


# ============================================================================
# CONVENIENCE: EXPORT A FIGURE TO MULTIPLE FILES
# ============================================================================
def export_fig_to_multiple_formats(fig,
                                   base_filename: str,
                                   formats: Optional[List[str]] = None,
                                   dpi: int = 600) -> Dict[str, bytes]:
    """
    Serialize a figure into several formats at once.

    Returns
    -------
    dict {format: bytes}
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]

    out = {}
    for fmt in formats:
        fmt = fmt.lower()
        d = None if fmt in ("pdf", "svg") else dpi
        data = fig_to_bytes(fig, fmt=fmt, dpi=d)
        if data:
            out[fmt] = data
    return out
