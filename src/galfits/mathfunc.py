"""
Mathematical Functions and Utilities Module
============================================

This module provides mathematical utility functions for image analysis,
geometric calculations, fitting, and statistical operations used throughout
the GalfitS package.

Key Components
--------------
- **Geometric functions**: Ellipse calculations, coordinate transformations
- **Photometry**: Aperture photometry within elliptical apertures
- **Fitting utilities**: Polynomial fitting, robust fitting methods
- **Masking**: Create masks for various geometric shapes
- **Cross-matching**: Match astronomical catalogs
- **Sampling**: Grid sampling for parameter space exploration
- **Statistics**: Loss functions, total variation regularization

Functions
---------
- aperture_photometry : Measure flux within elliptical apertures
- elliptical_func : Distance function for elliptical geometry
- Maskellipse : Create elliptical mask arrays
- polynomialfit : Robust polynomial fitting
- cross_match : Match positions between catalogs
- loss_total_variation_isotropic : TV regularization for image processing

These functions support both NumPy and JAX arrays for flexibility and
GPU acceleration where applicable.
"""

import jax.numpy as jnp
from astropy.modeling import models, fitting
from astropy.table import Table, Column, join, join_skycoord
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from scipy.spatial import cKDTree
import warnings

def aperture_photometry(image, posi, a, ellipticity, PA, mask = None, sky_mean = None, sky_std = None):
    r"""
    Measure flux within elliptical aperture with sky subtraction.
    
    Computes aperture photometry in an ellipse with optional sky background
    estimation from surrounding annulus (1.25a to 1.6a). Returns flux and
    uncertainty accounting for sky noise and Poisson statistics.
    
    Parameters
    ----------
    image : array_like, shape (ny, nx)
        2D image array in flux units.
    posi : tuple of float
        Aperture center (x, y) in pixel coordinates.
    a : float
        Semi-major axis in pixels.
    ellipticity : float
        Ellipticity = 1 - b/a, where b is semi-minor axis. Range [0, 1).
    PA : float
        Position angle in degrees, measured counter-clockwise from +x axis.
    mask : array_like, shape (ny, nx), optional
        Boolean mask (True = bad pixel to exclude). Default is None.
    sky_mean : float, optional
        Pre-determined sky background level. If None, estimated from annulus.
    sky_std : float, optional
        Pre-determined sky noise. If None, estimated from annulus.
    
    Returns
    -------
    flux : float
        Total sky-subtracted flux within aperture.
    flux_err : float
        Flux uncertainty from: :math:`\sqrt{F + (A \\sigma_{\\rm sky})^2 / A_{\\rm annulus} + A \\sigma_{\\rm sky}^2}`
        
        Includes source photon noise, sky uncertainty, and sky subtraction error.
    
    Notes
    -----
    Sky annulus extends from 1.25a to 1.6a. Masked pixels excluded from both
    aperture and annulus. Useful for measuring galaxy/star fluxes in crowded fields.
    
    Examples
    --------
    >>> # Measure flux in ellipse at (50, 60) with a=10, e=0.3, PA=45°
    >>> flux, err = aperture_photometry(image, (50, 60), 10, 0.3, 45)
    >>> print(f"Flux = {flux:.2f} ± {err:.2f}")
    """
    ny,nx=image.shape
    xaxis = jnp.arange(nx)
    yaxis = jnp.arange(ny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    inellip = elliptical_func(xmesh,posi,a,ellipticity,PA,ymesh)<0
    if mask is not None:
        image = jnp.where(mask, 0., image)
    annulus1 = elliptical_func(xmesh,posi,a*1.601,ellipticity,PA,ymesh)<0
    annulus2 = elliptical_func(xmesh,posi,a*1.25,ellipticity,PA,ymesh)<0
    sky_annulus = jnp.where(annulus1 & ~annulus2, 1., 0.)
    sky_mean_t = jnp.sum(image*sky_annulus)/(jnp.sum(sky_annulus)-jnp.sum(sky_annulus*mask))
    sky_std_t = jnp.sqrt(jnp.sum((image-sky_mean)**2*sky_annulus)/(jnp.sum(sky_annulus)-jnp.sum(sky_annulus*mask)))
    if sky_mean is None:
        sky_mean = sky_mean_t
    if sky_std is None:
        sky_std = sky_std_t
    #print (sky_mean,sky_std)
    area = jnp.sum(inellip)
    sky_area = jnp.sum(sky_annulus)
    flux = jnp.sum((image-sky_mean)*inellip)
    flux_err = (flux + (area * sky_std)**2 / sky_area +  area * sky_std**2)**0.5
    return flux, flux_err


@dataclass
class Trial:
    posi: Tuple[float, float]
    PA: float


def sampler(
    mask: Union[np.ndarray, jnp.ndarray],
    aperture: Dict[str, Union[Sequence[float], float]],
    threshold: float = 0.8,
    overlap: float = 0.2,
    sample: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    max_iterations: int = 50000,
    max_inner_iterations: int = 2000,
) -> List[Trial]:
    """Sample unmasked regions using elliptical apertures.

    Parameters
    ----------
    mask : array_like
        Two dimensional mask array where masked pixels should be truthy (1 or True).
    aperture : dict
        Dictionary describing the aperture with keys ``posi`` (center ``[x, y]`` in pixels),
        ``sma`` (semi-major axis), ``ellipticity`` (0-1) and ``PA`` (degrees, anti-clockwise from
        the *y* axis to the major axis).
    threshold : float, optional
        Required fraction of unmasked pixels within an aperture before accepting it.
    overlap : float, optional
        Maximum allowed fraction of already sampled pixels within the aperture.
    sample : float, optional
        Fraction of the total unmasked pixels to cover before terminating.
    rng : numpy.random.Generator or None, optional
        Random generator to use. If ``None`` a new default generator is created.
    max_iterations : int, optional
        Safeguard on the number of outer loop iterations.
    max_inner_iterations : int, optional
        Safeguard on the number of proposals per accepted aperture.

    Returns
    -------
    list[Trial]
        Sequence of accepted apertures represented by their center and position angle
        (still referenced to the *y* axis for compatibility with legacy behaviour).
    """

    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    mask_array = np.asarray(mask, dtype=float)
    if mask_array.ndim != 2:
        raise ValueError("mask must be a 2D array")

    ny, nx = mask_array.shape
    mask_bool = mask_array.astype(bool)

    total_unmasked = float(np.sum(1.0 - mask_array))
    if total_unmasked <= 0:
        return []

    try:
        posi = np.asarray(aperture["posi"], dtype=float)
    except KeyError as exc:
        raise KeyError("aperture dictionary must include 'posi'") from exc
    if posi.shape != (2,):
        raise ValueError("aperture['posi'] must contain two elements [x, y]")

    try:
        a = float(aperture["sma"])
        ellipticity = float(aperture["ellipticity"])
        pa_y = float(aperture.get("PA", 0.0))
    except KeyError as exc:
        raise KeyError("aperture dictionary must include 'sma' and 'ellipticity'") from exc

    if a <= 0:
        raise ValueError("Semi-major axis 'sma' must be positive.")
    if not (0.0 <= ellipticity < 1.0):
        raise ValueError("Ellipticity must be within [0, 1).")

    b = max(a * (1.0 - ellipticity), 1e-6)

    sample_fraction = float(np.clip(sample, 0.0, 1.0))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    overlap = float(np.clip(overlap, 0.0, 1.0))

    xaxis = np.arange(nx)
    yaxis = np.arange(ny)
    xmesh, ymesh = np.meshgrid(xaxis, yaxis)

    sample_image = np.zeros_like(mask_array, dtype=float)
    accepted_trials: List[Trial] = []
    k = 4.0

    def _theta_from_pa(pa_y_deg: float) -> float:
        """Convert position angle measured from the y-axis to radians from the x-axis."""
        return np.deg2rad((pa_y_deg + 90.0) % 180.0)

    def _ellipse_mask(cx: float, cy: float, theta: float) -> np.ndarray:
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        A = (a * sin_t) ** 2 + (b * cos_t) ** 2
        B = 2.0 * (b ** 2 - a ** 2) * sin_t * cos_t
        C = (a * cos_t) ** 2 + (b * sin_t) ** 2
        D = -2.0 * A * cx - B * cy
        E = -B * cx - 2.0 * C * cy
        F = A * (cx ** 2) + B * cx * cy + C * (cy ** 2) - (a ** 2) * (b ** 2)
        delta = A * (xmesh ** 2) + B * xmesh * ymesh + C * (ymesh ** 2) + D * xmesh + E * ymesh + F
        return delta < 0.0

    def _coverage(frac_image: np.ndarray, region_mask: np.ndarray) -> float:
        area_pix = float(region_mask.sum())
        if area_pix == 0.0:
            return 0.0
        return float(frac_image[region_mask].sum()) / area_pix

    target_pixels = total_unmasked * sample_fraction
    covered_pixels = float(sample_image[~mask_bool].sum())

    outer_loops = 0
    while covered_pixels < target_pixels and outer_loops < max_iterations:
        outer_loops += 1
        accepted = False
        inner_loops = 0
        while not accepted and inner_loops < max_inner_iterations:
            inner_loops += 1

            theta_move = _theta_from_pa(pa_y)
            step = rng.uniform(0.8 * a, 2.5 * a)
            x_candidate = (posi[0] + step * np.cos(theta_move)) % nx
            y_candidate = (posi[1] + step * np.sin(theta_move)) % ny

            theta_current = _theta_from_pa(pa_y)
            current_mask = _ellipse_mask(posi[0], posi[1], theta_current)
            area_current = current_mask.sum()
            if area_current == 0:
                continue
            cc0 = _coverage(sample_image, current_mask)

            traPA = rng.uniform(-90.0, 90.0)
            theta_candidate = _theta_from_pa(traPA)
            ellipse_mask = _ellipse_mask(x_candidate, y_candidate, theta_candidate)
            area_candidate = ellipse_mask.sum()
            if area_candidate == 0:
                continue

            cc = _coverage(sample_image, ellipse_mask)
            masked_fraction = float(mask_array[ellipse_mask].sum()) / area_candidate
            dd = 1.0 - masked_fraction

            if cc < cc0:
                posi = np.array([x_candidate, y_candidate])
                pa_y = traPA
            else:
                prob = np.exp(-k * (cc - cc0))
                if prob >= 1.0 or rng.random() < prob:
                    posi = np.array([x_candidate, y_candidate])
                    pa_y = traPA

            if dd > threshold and cc < overlap:
                sample_image[ellipse_mask] = 1.0
                covered_pixels = float(sample_image[~mask_bool].sum())
                accepted_trials.append(Trial(posi=(float(posi[0]), float(posi[1])), PA=float(pa_y)))
                accepted = True

        if not accepted:
            warnings.warn(
                "sampler reached the maximum number of inner iterations without finding a new aperture.",
                RuntimeWarning,
            )
            break

    if covered_pixels < target_pixels:
        warnings.warn(
            "sampler terminated before achieving the desired sampling fraction.",
            RuntimeWarning,
        )

    return accepted_trials

def elliptical_func(xmesh,posi,a,ellipticity,PA,ymesh=None):
    r"""
    Compute ellipse distance function or solve for y-coordinates.
    
    Returns signed distance from ellipse boundary (negative inside) or solves
    for y-coordinates on ellipse given x. Used for masking, photometry, and
    profile extraction.
    
    Parameters
    ----------
    xmesh : array_like
        X-coordinates of pixels or grid.
    posi : tuple of float
        Ellipse center (x0, y0) in pixels.
    a : float
        Semi-major axis in pixels.
    ellipticity : float
        Ellipticity = 1 - b/a, where b is semi-minor axis.
    PA : float
        Position angle in degrees from +x axis (counter-clockwise).
    ymesh : array_like, optional
        Y-coordinates. If provided, returns distance function.
        If None, solves for y given x.
    
    Returns
    -------
    array_like
        If ymesh provided: :math:`Ax^2 + Bxy + Cy^2 - a^2b^2`
        
        - Negative inside ellipse
        - Zero on ellipse
        - Positive outside
        
        If ymesh=None: y-coordinates on ellipse for given x
    
    Notes
    -----
    Distance formula for rotated ellipse:
    
    .. math::
        A = a^2 \\sin^2\\theta + b^2 \\cos^2\\theta
        
        B = 2(b^2 - a^2) \\sin\\theta \\cos\\theta
        
        C = a^2 \\cos^2\\theta + b^2 \\sin^2\\theta
    
    See Also
    --------
    Maskellipse : Create elliptical mask
    aperture_photometry : Flux measurement in ellipse
    """
    x=xmesh-posi[0]
    b=a*(1.-ellipticity)
    theta=PA*jnp.pi/180.
    A=(a**2)*(jnp.sin(theta))**2+(b**2)*(jnp.cos(theta))**2
    B=2*(b**2-a**2)*jnp.sin(theta)*jnp.cos(theta)
    C=(a**2)*(jnp.cos(theta))**2+(b**2)*(jnp.sin(theta))**2
    if ymesh is not None:
        y=ymesh-posi[1]
        return A*x**2+B*x*y+C*y**2-(a**2)*(b**2)
    else:
        coeff = [C, B*x, A*x**2-(a**2)*(b**2)]
        return jnp.roots(coeff)+posi[1]
    
def elliptical_cross_mask(image,posi,a,ellipticity,PA):
    """
    Create mask for pixels intersected by ellipse boundary.
    
    Identifies pixels where ellipse boundary crosses, useful for surface
    brightness profile extraction along ellipse perimeter.
    
    Parameters
    ----------
    image : array_like, shape (ny, nx)
        Input image (used only for shape).
    posi : tuple of float
        Ellipse center (x, y).
    a : float
        Semi-major axis in pixels.
    ellipticity : float
        Ellipticity = 1 - b/a.
    PA : float
        Position angle in degrees.
    
    Returns
    -------
    array_like, shape (ny, nx)
        Binary mask: 1 where ellipse crosses pixel, 0 elsewhere.
    
    See Also
    --------
    ellipse_profile : Extract flux along ellipse
    """
    ny,nx=image.shape
    xaxis = jnp.arange(nx)
    yaxis = jnp.arange(ny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    inellip1 = elliptical_func(xmesh,posi,a,ellipticity,PA,ymesh) * elliptical_func(xmesh+1,posi,a,ellipticity,PA,ymesh+1)
    inellip2 = elliptical_func(xmesh+1,posi,a,ellipticity,PA,ymesh) * elliptical_func(xmesh,posi,a,ellipticity,PA,ymesh+1)
    cross = jnp.where((inellip1<0) | (inellip2<0), 1., 0.)
    return cross 

def ellipse_profile(image,posi,a,ellipticity,PA,resolution=200):
    '''
    measuing the pixel value along the ellipse

    Parameters:
    image: 2D array
    posi: the center of the ellipse
    a: the major axis of the ellipse
    ellipticity: the ellipticity of the ellipse
    PA: the position angle of the ellipse, degree, since the x axis, anti-clockwise
    
    Return: flux_list, the flux along the ellipse, ordered by the angle
    '''
    PA_rad = jnp.deg2rad(PA)
    # Calculate minor axis
    b = a * (1 - ellipticity)
    # Create an array of angles (theta values) from 0 to 2*pi
    thetas = jnp.linspace(0, 2 * jnp.pi, resolution)  # You can adjust the resolution as needed
    # Precompute sin and cos for efficiency
    cos_PA, sin_PA = jnp.cos(PA_rad), jnp.sin(PA_rad)
    # Compute (x, y) positions on the ellipse for each angle
    x = posi[0] + a * jnp.cos(thetas) * cos_PA - b * jnp.sin(thetas) * sin_PA
    y = posi[1] + a * jnp.cos(thetas) * sin_PA + b * jnp.sin(thetas) * cos_PA
    # Extract pixel values at each (x, y) position
    flux_list = [image[int(round(yi)), int(round(xi))] for xi, yi in zip(x, y)]
    return thetas,flux_list



def elliptical_surface_brightness(image,posi,a_grid,ellipticity,PA):
    '''
    measuing the mean pixel value along the ellipse
    sample the a_grid points 
    
    Parameters:
    image: 2D array
    posi: the center of the ellipse
    a_grid: the major axis of the ellipse
    ellipticity: the ellipticity of the ellipse
    PA: the position angle of the ellipse, degree, since the x axis, anti-clockwise
    
    Return: surface_brightness, the mean flux along the a_grid
    '''
    surface_brightness=[]
    surface_brightness_err=[]
    for a in a_grid:
        crossmask = elliptical_cross_mask(image,posi,a,ellipticity,PA)
        meanflux = jnp.sum(image*crossmask)/jnp.sum(crossmask)
        fluxstd = jnp.sqrt(jnp.sum(((image-meanflux)*crossmask)**2)/jnp.sum(crossmask))
        surface_brightness.append(meanflux)
        surface_brightness_err.append(fluxstd)
    return jnp.array(surface_brightness),jnp.array(surface_brightness_err)

def Maskellipse(mask,posi,a,ellipticity,PA,antimask=False):
    """
    Apply elliptical mask to existing mask array.
    
    Adds elliptical region to mask (masking interior) or clears it (antimask=True).
    
    Parameters
    ----------
    mask : array_like, shape (ny, nx)
        Existing mask array (modified in-place conceptually).
    posi : tuple of float
        Ellipse center (x, y).
    a : float
        Semi-major axis in pixels.
    ellipticity : float
        Ellipticity = 1 - b/a.
    PA : float
        Position angle in degrees from +x axis (counter-clockwise).
    antimask : bool, default=False
        If False, set interior to 1 (mask it).
        If True, set interior to 0 (unmask it).
    
    Returns
    -------
    array_like
        Updated mask with same dtype as input.
    
    See Also
    --------
    elliptical_func : Underlying distance function
    """
    #PA from x axis anti-clockwise, degree
    masktemp=mask.copy()
    ny,nx=mask.shape
    xaxis = jnp.arange(nx)
    yaxis = jnp.arange(ny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    inellip = elliptical_func(xmesh+0.5,posi,a,ellipticity,PA,ymesh+0.5)<0
    if not antimask:
        masktemp = jnp.where(inellip, 1, masktemp)
    else:
        masktemp = jnp.where(inellip, 0, masktemp)
    return masktemp.astype(mask.dtype)

def polynomialfit(data,mask,order=3):
    """
    Fit and subtract 2D polynomial background from image.
    
    Uses robust least-squares fitting on unmasked regions. Common for sky
    subtraction in spectroscopy or removing large-scale gradients.
    
    Parameters
    ----------
    data : array_like, shape (ny, nx)
        Input image.
    mask : array_like, shape (ny, nx)
        Boolean mask (True/1 = exclude from fit).
    order : int, default=3
        Polynomial degree. order=3 includes terms up to x³, y³, x²y, xy².
    
    Returns
    -------
    dict
        Dictionary with keys:
        
        - 'bkgfunc' : Polynomial2D model
            Fitted polynomial function
        - 'bkg' : array_like
            2D background model
        - 'residual' : array_like
            Background-subtracted image (data - bkg)
    
    Examples
    --------
    >>> result = polynomialfit(image, mask, order=2)
    >>> clean_image = result['residual']
    >>> background = result['bkg']
    
    See Also
    --------
    astropy.modeling.models.Polynomial2D : Underlying model
    """
    i, j = jnp.mgrid[:data.shape[0], :data.shape[1]]
    i=data.shape[0]-i
    imask=i[~mask.astype(bool)]
    jmask=j[~mask.astype(bool)]
    datamask=data[~mask.astype(bool)]
    p_init = models.Polynomial2D(degree=order)
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = fit_p(p_init, imask, jmask, datamask)
    background=p(i,j)
    datap=data-background
    result={
        'bkgfunc':p,
        'bkg':background,
        'residual':datap
    }
    return result

def loss_total_variation_isotropic(image: Union[np.ndarray, jnp.ndarray]) -> Union[float, jnp.floating]:
    r"""
    Compute the isotropic Total Variation (TV) loss of a 2D image.

    The isotropic TV is defined as the sum of gradient magnitudes across the image:

    .. math::
        \mathrm{TV}(I) = \sum_{i,j} \sqrt{\left(\nabla_x I_{i,j}\right)^2 + \left(\nabla_y I_{i,j}\right)^2 + \epsilon}

    where finite differences are used for the gradients, and a small :math:`\epsilon` is added
    for numerical stability when gradients are zero.

    Parameters
    ----------
    image : array_like, shape (ny, nx)
        Input 2D image.

    Returns
    -------
    float
        Isotropic TV loss as a scalar value.

    Notes
    -----
    This implementation uses JAX NumPy if available to support JIT and autodiff.
    """
    # Use jnp throughout for JAX compatibility; np inputs are implicitly converted
    img = jnp.asarray(image)
    # Forward finite differences
    dx = img[:, 1:] - img[:, :-1]
    dy = img[1:, :] - img[:-1, :]
    # Pad to original shape for consistent summation
    dx_pad = jnp.pad(dx, ((0, 0), (0, 1)))
    dy_pad = jnp.pad(dy, ((0, 1), (0, 0)))
    eps = 1e-12
    tv = jnp.sum(jnp.sqrt(dx_pad * dx_pad + dy_pad * dy_pad + eps))
    return tv

def cross_match(catlist, angular_sep = 2.5, key0='sc',keyF='scF'):
    """
    Cross-match multiple astronomical catalogs by sky coordinates.
    
    Recursively joins catalogs using SkyCoord matching. Useful for combining
    multi-wavelength observations of same sources.
    
    Parameters
    ----------
    catlist : list of Table
        Astropy tables with SkyCoord column (default name 'sc').
    angular_sep : float, default=2.5
        Matching radius in arcseconds.
    key0 : str, default='sc'
        Column name for SkyCoord in input catalogs.
    keyF : str, default='scF'
        Column name for SkyCoord in output merged catalog.
    
    Returns
    -------
    Table
        Merged catalog with matched sources. Columns from all input
        catalogs included. Unmatched sources excluded.
    
    Notes
    -----
    For N catalogs, recursively pairs: (1,2), (3,4), ..., then pairs results.
    
    Examples
    --------
    >>> cat1 = Table({'sc': coords1, 'flux_r': [...]})
    >>> cat2 = Table({'sc': coords2, 'flux_g': [...]})
    >>> matched = cross_match([cat1, cat2], angular_sep=1.0)
    
    See Also
    --------
    make_union_catalog : Find all detections across catalogs
    astropy.table.join_skycoord : Underlying matching function
    """
    lencat = len(catlist)
    if lencat == 2:
        join_funcs={key0: join_skycoord(angular_sep/3600. * u.deg)}
        tab = join(catlist[0], catlist[1], join_funcs=join_funcs)
        tab.remove_column('{0}_2'.format(key0))
        tab.remove_column('{0}_id'.format(key0))
        tab.rename_column('{0}_1'.format(key0), keyF)
        #print (tab)
        return tab
    else:
        catnew = []
        if (lencat-2*(lencat//2)) == 1:
            for loop in range(lencat//2):
                catnew.append(cross_match([catlist[2*loop],catlist[2*loop+1]],angular_sep,key0,key0))
            catnew.append(catlist[lencat-1])
        else:
            for loop in range(lencat//2):
                catnew.append(cross_match([catlist[2*loop],catlist[2*loop+1]],angular_sep,key0,key0))
        return cross_match(catnew,angular_sep,key0,keyF)

def make_union_catalog(cataloglist, distance=5):
    """
    Cluster sources from multiple catalogs based on their xcentroid and ycentroid coordinates.
    
    Parameters:
    cataloglist (list): List of source catalogs (each a dictionary or table with 'xcentroid' and 'ycentroid' keys).
    distance (float): Maximum distance to consider sources as part of the same cluster.
    
    Returns:
    list: A (M, N) list where M is the number of unique clusters and N is the number of catalogs.
    The element of (m,n) indicate the index of source labeled by the m-th cluster at the n-th catalog. 
    And if in the n-th catalog, label m source is not in it (not detected), (m,n)=-1.
    """
    all_sources = []
    catalog_indices = []
    
    # Collect all sources and their catalog indices
    for i, catalog in enumerate(cataloglist):
        x = catalog['xcentroid']
        y = catalog['ycentroid']
        for j in range(len(x)):
            all_sources.append((x[j], y[j]))
            catalog_indices.append(i)
    
    # Build a KD-Tree for fast spatial clustering
    kdtree = cKDTree(all_sources)
    clusters = kdtree.query_ball_tree(kdtree, r=distance)
    
    # Assign labels to each unique cluster
    labels = np.full(len(all_sources), -1)
    current_label = 0
    for i, cluster in enumerate(clusters):
        if labels[i] == -1:
            for idx in cluster:
                if labels[idx] == -1:
                    labels[idx] = current_label
            current_label += 1
    
    # Create the (M, N) result array
    M = current_label
    N = len(cataloglist)
    result = [[-1] * N for _ in range(M)]
    
    for idx, label in enumerate(labels):
        catalog_idx = catalog_indices[idx]
        source_idx = idx - sum(len(catalog['xcentroid']) for catalog in cataloglist[:catalog_idx])
        result[label][catalog_idx] = source_idx
    
    return result

 

