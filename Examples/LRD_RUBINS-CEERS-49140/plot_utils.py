"""
Plotting utilities that exactly reproduce the GalfitS 2D-image display
convention (see src/galfits/gsutils.py: `normimg` + `standard_display`).
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from astropy.stats import sigma_clipped_stats



def set_plot_style(fontscale=1.0, tick_direction="out", usetex=True):
    """
    One-call global plotting style, summarizing the settings used in
    isophot_for_agnhost.ipynb (which itself builds on set_mpl_style()).

    - usetex + serif + stix mathtext (bold via B() helper below)
    - axis labels 30, ticks 20 (scaled by fontscale)
    - major ticks width=2 length=8, minor width=1 length=5, direction out
    - minor ticks on, top/right ticks on
    - axes spine linewidth 1.2, legend frameless, numpoints=1
    - figure.dpi 120, savefig 300 tight
    """
    fs = fontscale
    mpl.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 20 * fs,

        "axes.titlesize": 30 * fs,
        "axes.labelsize": 30 * fs,
        "axes.labelpad": 8.0,
        "axes.linewidth": 1.2,

        "xtick.direction": tick_direction,
        "ytick.direction": tick_direction,
        "xtick.labelsize": 20 * fs,
        "ytick.labelsize": 20 * fs,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.size": 8.0,
        "ytick.major.size": 8.0,
        "xtick.minor.size": 5.0,
        "ytick.minor.size": 5.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,

        "legend.fontsize": 20 * fs,
        "legend.frameon": False,
        "legend.numpoints": 1,
        "legend.handlelength": 2.5,
        "legend.borderpad": 0.5,
        "legend.labelspacing": 0.4,

        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def B(s):
    """Bold text+math under usetex (fontweight='bold' does not work there)."""
    return r"{\boldmath\bfseries " + s + "}"


def fig_grid(nrow, ncol, size=6):
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    gridspec = GridSpec(nrow, ncol, figure=fig)
    return fig, gridspec


def normimg(image, min=0, max=None, sky=0, frac=0.5):
    """
    Normalize image with arcsinh stretch for visualization.

    Numpy port of galfits.gsutils.normimg (identical algorithm).

    Parameters
    ----------
    image : array_like
        Input image array.
    min : float, default=0
        Lower threshold of the arcsinh stretch. Below this the scaling
        is linear (image * frac / min).
    max : float, optional
        Upper threshold of the arcsinh stretch. If None, uses image maximum.
    sky : float, default=0
        Sky background to subtract before scaling.
    frac : float, default=0.5
        Fraction of the [0, 1] output range given to the linear part
        (0 <= image <= min maps linearly onto [0, frac]).

    Returns
    -------
    numpy.ndarray
        Normalized image, roughly in [-1, 1]; display with vmin=-1, vmax=1.
    """
    if max is None:
        max     = np.nanmax(image)
    image   = np.array(image, dtype=float) - sky
    image   = np.where(
        image > min,
        frac + (1 - frac) *
        (np.arcsinh(image) - np.arcsinh(min)) / (np.arcsinh(max) - np.arcsinh(min)),
        image * frac / min,
    )
    image   = np.where(
        image < -frac,
        -frac -(1 - frac) * 
        (np.arcsinh(-image) - np.arcsinh(frac)) / (np.arcsinh(frac * max / min) - np.arcsinh(frac)),
        image,
    )
    return image


def show_image(img, mask=None,
               fig=None, grid=None, pos=(0, 0),
               nmin=5.0, nmax=None,  frac=0.4,  
               mode="sci", cmap="seismic", title=None, colorbar=False,):

    if grid is None:
        fig     = plt.figure(figsize=(6,6))
        grid    = GridSpec(1, 1, figure=fig)
    ax  = grid.figure.add_subplot(grid[pos[0], pos[1]])

    if mode == "sci":
        _, _, sky_std   = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
        immin   = nmin * sky_std
        immax   = np.nanmax(img) if nmax is None else nmax * sky_std
        im  = ax.imshow(normimg(img, immin, immax, frac=frac),
                       cmap=cmap, origin="lower", vmin=-1, vmax=1,
                       interpolation="nearest")
    elif mode == "resid":
        im  = ax.imshow(img, cmap=cmap, origin="lower", vmin=-10, vmax=10, interpolation="nearest")
    else:
        raise ValueError("mode must be 'sci' or 'resid'")

    if mask is not None:
        mask    = np.array(mask, dtype=float)
        m_vis   = np.ma.masked_where(mask == 0, mask)  # hide unmasked pixels (0)
        shade   = mcolors.ListedColormap(["grey"])
        ax.imshow(m_vis, cmap=shade, origin="lower", alpha=0.6, interpolation="nearest", vmin=0, vmax=1)

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2D images: hide top/right ticks
    ax.tick_params(which="both", top=False, right=False)

    if title is not None:
        # upright bold sans-serif, same as the "MOCK"/filter labels
        # in the 4x8 figure: B(r"\textsf{...}")
        if mpl.rcParams["text.usetex"]:
            ax.set_title(B(rf"\textsf{{{title}}}"))
        else:
            ax.set_title(title, fontweight="bold", family="sans-serif")

    return ax, 


def forward(x):
    EPS     = 1e-8
    x       = np.asarray(x, dtype=float)
    out     = np.empty_like(x)
    mask    = x<=1
    x_log   = np.where(x[mask] > 0, x[mask], EPS)   # 0 → EPS
    out[mask]  = np.log10(x_log)
    out[~mask] = x[~mask] - 1
    return out    

def inverse(y):
    
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    mask = y <= 0
    out[mask] = 10**y[mask]              # y<=0: comes from log side
    out[~mask] = y[~mask] + 1           # y>=0: comes from linear side
    return out
