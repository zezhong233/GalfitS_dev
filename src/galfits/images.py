"""
Image Processing and Photometry Module
=======================================

This module provides tools for astronomical image processing, source detection,
background estimation, PSF extraction, and photometric measurements. It integrates
with photutils and astropy for standard astronomical data reduction tasks.

Key Features
------------
- **Source detection**: Detect and segment sources in images
- **Background estimation**: 2D background modeling and subtraction
- **PSF extraction**: Build empirical PSFs from star stacks
- **Aperture photometry**: Measure fluxes within various apertures
- **Image manipulation**: Resampling, matching, convolution
- **Cutout extraction**: Create image cutouts around targets
- **Visualization**: Image display with astronomical conventions

Functions
---------
- normimg : Normalize image for display
- grow_segmentation : Expand segmentation maps
- match_psf : Match PSF between images
- extract_psf : Build PSF from stellar sources
- measure_background : Estimate and subtract background
- detect_sources : Source detection and cataloging

The module is designed to work seamlessly with the main fitting pipeline,
preparing data for galaxy modeling and SED fitting.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from astropy.wcs.utils import proj_plane_pixel_scales
from .plot import plot_image
from scipy import ndimage
from .galaxy import coordinates_transfer
from .gsutils import effective_wave
from .sed_interp import wavegrid_dlnlam
from photutils.background import Background2D
from . import profiles as pro
from astropy.visualization import simple_norm, make_lupton_rgb
from .mathfunc import Maskellipse,polynomialfit,cross_match,elliptical_func,loss_total_variation_isotropic
from photutils.segmentation import deblend_sources
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.psf import EPSFStar,EPSFStars,EPSFBuilder
from photutils.segmentation import SourceCatalog,detect_threshold,detect_sources
from astropy.table import Table, Column, join, join_skycoord, vstack
from astropy.wcs import WCS
from astropy.nddata import NDData
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder
from . import Constant as C
import jax.numpy as jnp
import jax,optax,evosax
from astroquery.xmatch import XMatch
from jax import random
jaxkey = random.PRNGKey(0)
from photutils.centroids import centroid_com
from scipy.interpolate import RectBivariateSpline
from numpy import fft
from reproject import reproject_interp

def sep_to_photutils_like_table(objects, *, assume_sep_ab_are_sigmas=True):
    '''
    Convert SEP `objects` (structured array from sep.extract) to an Astropy Table
    with photutils-like column names used in your workflow.

    Mapping/assumptions:
    - xcentroid, ycentroid <- objects['x'], objects['y']
    - semimajor_sigma, semiminor_sigma <- objects['a'], objects['b']
    - orientation <- objects['theta']  (radians; SEP uses SExtractor convention)
    - segment_flux <- objects['flux']  (isophotal flux in SEP)
    - fwhm <- 2.3548 * sqrt(a*b)  (equivalent circular sigma from geometric mean)
    - maxval_xindex, maxval_yindex <- objects['xpeak'], objects['ypeak'] if present
        otherwise set to -1

    If your SEP build uses different definitions for a/b, set assume_sep_ab_are_sigmas=False
    and adjust accordingly.
    '''
    t = Table()

    # centroids
    t["xcentroid"] = np.array(objects["x"], dtype=float)
    t["ycentroid"] = np.array(objects["y"], dtype=float)

    # ellipse parameters (SEP typically reports RMS-like ellipse sigmas, similar spirit to photutils *_sigma)
    a = np.array(objects["a"], dtype=float)
    b = np.array(objects["b"], dtype=float)
    t["semimajor_sigma"] = a
    t["semiminor_sigma"] = b

    # orientation (radians)
    t["orientation"] = np.array(objects["theta"], dtype=float)

    # flux analogous to segment_flux
    if "flux" in objects.dtype.names:
        t["segment_flux"] = np.array(objects["flux"], dtype=float)
    else:
        t["segment_flux"] = np.nan

    # FWHM (use geometric-mean sigma as a single-number proxy, like many pipelines do)
    # If a/b are already sigmas, this is fine. Otherwise tweak the formula here.
    sigma_eq = np.sqrt(np.clip(a * b, 0.0, np.inf))
    t["fwhm"] = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_eq

    # peak location (closest analog to photutils maxval_*index)
    if ("xpeak" in objects.dtype.names) and ("ypeak" in objects.dtype.names):
        # SEP xpeak/ypeak are pixel indices (often ints); enforce int
        t["maxval_xindex"] = np.array(objects["xpeak"], dtype=int)
        t["maxval_yindex"] = np.array(objects["ypeak"], dtype=int)
    else:
        t["maxval_xindex"] = -np.ones(len(objects), dtype=int)
        t["maxval_yindex"] = -np.ones(len(objects), dtype=int)

    return t
def to_native_endian(a):
    a = np.asarray(a)
    if a.dtype.byteorder in ('>', '!'):  # big-endian
        return a.astype(a.dtype.newbyteorder('='), copy=False)
    return a
def normimg(image, min=0, max=None, sky=0, frac = 0.5):
    if max is None:
        max = np.nanmax(image)
    image = np.array(image - sky)
    image = np.where(image > min, (1-frac)*(np.arcsinh(image)-np.arcsinh(min))/(np.arcsinh(max)-np.arcsinh(min))+frac, image*frac/min)
    image = np.where(image < -frac, -(1-frac)*(np.arcsinh(-image)-np.arcsinh(frac))/(np.arcsinh(frac*max/min)-np.arcsinh(frac))-frac, image)
    return image

def choose_kernel_half_width(
    wave, lsf_wave, lsf_R, Nlogv=2000, nsig=6.0,
    wave_unit_scale=1.0,  # multiply wave by this to match lsf_wave units if needed
):
    """
    Pick a fixed kernel half-width K (in log-grid pixels) large enough for the
    broadest LSF across the covered wavelength range.

    This is a *Python-side* helper (not jitted). Use once, then pass K into jit fn.
    """
    # log-grid spacing (same as inside conv)
    logw = jnp.log(wave * wave_unit_scale)
    newlogw = jnp.linspace(jnp.min(logw), jnp.max(logw), Nlogv)
    dlogw = float(newlogw[1] - newlogw[0])

    # Interpolate R onto the log grid (in Python helper; OK to use jnp here too)
    lam_new = jnp.exp(newlogw)
    R_new = jnp.interp(lam_new, lsf_wave, lsf_R, left=lsf_R[0], right=lsf_R[-1])

    # Broadest sigma is at smallest R
    R_min = float(jnp.min(R_new))
    sigma_ln_max = (1.0 / R_min) * C._FWHM_TO_SIGMA
    sigma_pix_max = sigma_ln_max / dlogw

    K = int(jnp.ceil(nsig * sigma_pix_max))
    return max(K, 3)  # ensure not tiny

def grow_segmentation(segm, data, source_label, source_dia = 0.5, max_grow_size=5, sky_level=0.1, nearbygrow = 0, addgrow=0):
    """
    Adaptively grow segmentation regions to encompass full source extent.
    
    Iteratively dilates source footprints until reaching sky background level,
    with special handling for target source and nearby companions. More accurate
    than fixed-size elliptical masking as it follows actual isophotes.

    Parameters
    ----------
    segm : photutils.segmentation.SegmentationImage
        Initial segmentation map from source detection.
    data : ndarray
        Image data (background-subtracted) to guide growth.
    source_label : int or None
        Label of target source to protect from masking. If None, grows all sources
        without protection.
    source_dia : float, default: 0.5
        Protected region scale factor. Neighbors within `source_dia * sqrt(area/π)`
        pixels of target are treated specially.
    max_grow_size : int, default: 5
        Maximum number of dilation iterations for contaminating sources.
    sky_level : float, default: 0.1
        Flux threshold for growth termination. Growth stops when median of newly
        added pixels < `sky_level`.
    nearbygrow : int, default: 0
        Number of dilation iterations for sources near target (within protected
        region). Set > 0 to also mask nearby companions.
    addgrow : int, default: 0
        Additional dilation iterations for all contaminating sources after adaptive
        growth.

    Returns
    -------
    segm_grow : photutils.segmentation.SegmentationImage
        Grown segmentation map with same labels as input.
    mask : ndarray of bool
        Boolean mask where True indicates contaminating source pixels (excludes
        target source and sky). Shape matches `data`.

    Notes
    -----
    **Growth Algorithm**:
    
    1. **Identify protected region**: 
       - Dilate target source by `source_dia * sqrt(area/π)` pixels
       - Find all source labels overlapping this region
    
    2. **Handle nearby sources**:
       - Apply `nearbygrow` iterations (usually 0 to leave untouched)
       - Do not allow growth into target source region
    
    3. **Grow contaminating sources**:
       - For each label not in protected region:
         a. Dilate by 1 pixel
         b. Compute median flux of new pixels
         c. If median < `sky_level` OR iteration > `max_grow_size`: stop
         d. Otherwise: accept growth and repeat
       - Apply `addgrow` iterations unconditionally
       - Never grow into target source region
    
    4. **Create mask**: 
       - Mask = (grown segmentation ≠ 0) AND (grown segmentation ≠ source_label)
       - Also mask NaN/Inf pixels
    
    **Usage Scenarios**:
    
    - **Isolate target** (`source_label` set): Mask companions but preserve target
    - **Background estimation** (`source_label=None`): Mask all sources uniformly
    
    The adaptive growth is particularly valuable in crowded fields where fixed
    ellipses may miss faint wings or overlap neighboring sources.

    Examples
    --------
    Grow all sources to sky level for background estimation:
    
    >>> from photutils.segmentation import detect_sources
    >>> segm = detect_sources(data, threshold=3*sky_std, npixels=5)
    >>> segm_grown, mask_all = grow_segmentation(segm, data, source_label=None,
    ...                                          max_grow_size=20, sky_level=0.05)
    >>> background = np.median(data[~mask_all])
    
    Isolate galaxy at label=5 by masking grown companions:
    
    >>> segm_grown, mask_contam = grow_segmentation(segm, data, source_label=5,
    ...                                             source_dia=1.0, max_grow_size=50,
    ...                                             sky_level=0.1, addgrow=5)
    >>> galaxy_only = np.where(mask_contam, 0, data)
    
    Mask nearby companions but leave distant sources untouched:
    
    >>> segm_grown, mask = grow_segmentation(segm, data, source_label=5,
    ...                                      source_dia=2.0, nearbygrow=10)
    
    See Also
    --------
    image.generate_cutmask : High-level masking interface
    photutils.segmentation.detect_sources : Initial segmentation
    scipy.ndimage.binary_dilation : Underlying dilation operation
    """
    segm_grow = segm.copy()
    # Create a mask for the source
    if source_label is None:
        segm_grow.data = np.array(ndimage.binary_dilation(segm_grow.data > 0, iterations= max_grow_size), dtype = np.int32) 
        mask = (segm_grow.data != 0)
    else:
        source_mask = (segm.data == source_label)
        # Dilate the source mask to find labels that are neighbors to the source
        sourcearea = np.sum(source_mask)
        #print (int(source_dia*np.sqrt(sourcearea/np.pi)))
        dilated_source_mask = ndimage.binary_dilation(source_mask, iterations=int(source_dia*np.sqrt(sourcearea/np.pi)))
        # Identify neighboring labels
        neighboring_labels = np.unique(segm.data[dilated_source_mask])
        for label in segm.labels:
            if label in neighboring_labels:
                if nearbygrow > 0:
                    dilation_result = ndimage.binary_dilation(segm_grow.data == label, iterations=nearbygrow)
                    safe_to_grow = np.logical_and(~dilated_source_mask, segm_grow.data==0)
                    segm_grow.data[np.logical_and(dilation_result, safe_to_grow)] = label
                continue
            count = 0
            while True:
                dilation_result = ndimage.binary_dilation(segm_grow.data == label, iterations=1)
                # Only grow the segmentation where it's not reaching the source or dilated_source_mask
                safe_to_grow = np.logical_and(~dilated_source_mask, segm_grow.data==0)
                grown_pixels = np.logical_and(dilation_result, safe_to_grow)
                segm_grow.data[grown_pixels] = label
                count += 1
                if (np.nanmedian(data[grown_pixels]) < sky_level) or count == max_grow_size:      
                    break
            if addgrow > 0:
                dilation_result = ndimage.binary_dilation(segm_grow.data == label, iterations=addgrow)
                safe_to_grow = np.logical_and(~dilated_source_mask, segm_grow.data==0)
                segm_grow.data[np.logical_and(dilation_result, safe_to_grow)] = label
        mask = (segm_grow.data != source_label) & (segm_grow.data != 0)
    
    masknan = np.isnan(data) | np.isinf(data)
    mask = mask | masknan
    return segm_grow, mask

def fittingPSF(cutPSF,cutMask,cutSigma=None,oversampling = 1,a_TV = 1e6, a_sky=1, verbose = False,  learning_rate=1e-4, num_steps=14000, perturbate = 1. , makestackpsf = False, fit_method = 'adam'):
    N_star = len(cutPSF)
    norm = []
    dof = 0
    for loop in range(N_star):
        cutPSF[loop] = jnp.where(cutMask[loop],0,cutPSF[loop])
        if loop == 0:
            sumimage = cutPSF[loop] 
        else:
            sumimage += cutPSF[loop] 
        #print (loop)
        dof += np.sum(cutMask[loop]==False)
        norm.append(jnp.sum(cutPSF[loop]))
    if cutSigma is None:
        cutSigma = jnp.ones(N_star,dtype=jnp.float32)
    ny0,nx0 = cutPSF[0].shape
    ny = oversampling*ny0
    nx = oversampling*nx0
    mask = np.zeros((ny,nx),dtype=bool)
    circle = Maskellipse(mask,[0.5*ny,0.5*nx],0.5*nx,0,0)
    circle = jnp.array(circle,dtype=bool)
    isize = ny*nx
    stackpsf = sumimage/(jnp.sum(sumimage))
    if makestackpsf:
        return regulate_PSF(stackpsf)
    if oversampling > 1:
        overpsf = jax.image.scale_and_translate(stackpsf,(ny,nx),(0,1),jnp.array([oversampling,oversampling]),jnp.array([0,0]),'cubic')/oversampling**2
    else:
        overpsf = stackpsf
    
    def cal_chisq(theta):
        psf_flat = theta[0:isize]
        psf_image = jnp.reshape(psf_flat,(ny,nx))
        psfsky = jnp.sum((psf_image*circle)**2)
        zeros = jnp.zeros((ny,nx))
        left = zeros.at[0:ny,1:nx].set(psf_image[0:ny,0:nx-1])
        right = zeros.at[0:ny,0:nx-1].set(psf_image[0:ny,1:nx])
        up = zeros.at[1:ny,0:nx].set(psf_image[0:ny-1,0:nx])
        down = zeros.at[0:ny-1,0:nx].set(psf_image[1:ny,0:nx])
        # calculate the TV loss
        TV_loss = jnp.sum((left-psf_image)*(left-psf_image)+(right-psf_image)*(right-psf_image)+(up-psf_image)*(up-psf_image)+(down-psf_image)*(down-psf_image))
        #V = (left+right+up+down - 4*psf_image) #/(psf_image+1e-8)
        #TV_loss = jnp.sum(V[1:ny-1,1:nx-1]*V[1:ny-1,1:nx-1])
        Norm = theta[isize:isize+N_star]
        xy = theta[isize+N_star:isize+3*N_star]
        #sky = theta[isize+3*N_star:isize+4*N_star]
        for loop in range(N_star):
            resi = cutPSF[loop] - Norm[loop]*pro.add_PSF(xy[2*loop],xy[2*loop+1],nx0,ny0,psf_image,scalefactor=1./oversampling)
            resi = resi/cutSigma[loop]
            resi = jnp.where(cutMask[loop],0,resi)
            if loop == 0:
                residu_flat = resi.ravel()
            else:
                residu_flat=jnp.append(residu_flat,resi.ravel())
        # # centeroid calculation from photutils
        # def centroid_com(data):
        #     data = data.copy()
        #     total = jnp.sum(data)
        #     indices = jnp.ogrid[tuple(slice(0, i) for i in data.shape)]
        #     return jnp.array([jnp.sum(indices[axis] * data) / total
        #                     for axis in range(data.ndim)])[::-1]
        # xcenter, ycenter = centroid_com(psf_image)
        # center_loss = jnp.sum((xcenter-nx/2)**2) + jnp.sum((ycenter-ny/2)**2)
        # chisq = ... + center_loss*1e8
        chisq = 0.5*jnp.sum(residu_flat*residu_flat)+a_TV*N_star*TV_loss + a_sky*N_star*psfsky/oversampling**2
        #print (0.5*jnp.sum(residu_flat*residu_flat), a_TV*N_star*TV_loss)
        return chisq
    
    theta0 = jnp.zeros(isize+3*N_star)
    flat = jnp.array(overpsf.ravel())
    theta0 = theta0.at[0:isize].set(flat) 
    theta0 = theta0.at[isize:isize+N_star].set(jnp.array(norm)) 
    theta0 = theta0.at[isize+N_star:isize+3*N_star].set(0.5*perturbate*nx0+0.5*random.normal(jaxkey, shape = (2*N_star,))) 
    
    if fit_method == 'adam':
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(theta0)
        #return cal_chisq(theta0)
        @jax.jit
        def step(logitu, opt_state):
            g = jax.grad(cal_chisq)(logitu)
            updates, opt_state = optimizer.update(g, opt_state)
            new_logitu = optax.apply_updates(logitu, updates)
            return new_logitu, opt_state, cal_chisq(logitu)
        logitu = theta0
        for i in range(num_steps):
            #last = logitu
            logitu, opt_state, loss = step(logitu, opt_state)
            if i % 100 == 0:  # print every 100 steps
                if verbose:
                    print(f"Step {i}, Chisq: {loss}")
                if jnp.isnan(loss):
                    break
                else:
                    theta = logitu

    psf_flat = theta[0:isize]
    psf_image = jnp.reshape(psf_flat,(ny,nx))

    def cal_chisq(theta):
        psf_flat = theta[0:isize]
        psf_image = jnp.reshape(psf_flat,(ny,nx))
        psfsky = jnp.sum((psf_image*circle)**2)
        zeros = jnp.zeros((ny,nx))
        left = zeros.at[0:ny,1:nx].set(psf_image[0:ny,0:nx-1])
        right = zeros.at[0:ny,0:nx-1].set(psf_image[0:ny,1:nx])
        up = zeros.at[1:ny,0:nx].set(psf_image[0:ny-1,0:nx])
        down = zeros.at[0:ny-1,0:nx].set(psf_image[1:ny,0:nx])
        # calculate the TV loss
        TV_loss = jnp.sum((left-psf_image)*(left-psf_image)+(right-psf_image)*(right-psf_image)+(up-psf_image)*(up-psf_image)+(down-psf_image)*(down-psf_image))
        #V = (left+right+up+down - 4*psf_image) #/(psf_image+1e-8)
        #TV_loss = jnp.sum(V[1:ny-1,1:nx-1]*V[1:ny-1,1:nx-1])
        Norm = theta[isize:isize+N_star]
        xy = theta[isize+N_star:isize+3*N_star]
        #sky = theta[isize+3*N_star:isize+4*N_star]
        for loop in range(N_star):
            resi = cutPSF[loop] - Norm[loop]*pro.add_PSF(xy[2*loop],xy[2*loop+1],nx0,ny0,psf_image,scalefactor=1./oversampling)
            resi = resi/cutSigma[loop]
            resi = jnp.where(cutMask[loop],0,resi)
            if loop == 0:
                residu_flat = resi.ravel()
            else:
                residu_flat=jnp.append(residu_flat,resi.ravel())
            print (loop,jnp.sum(resi.ravel()*resi.ravel()))
        print (0.5*jnp.sum(residu_flat*residu_flat)/dof, a_TV*N_star*TV_loss/dof, a_sky*N_star*psfsky/oversampling**2/dof)
        return 
    cal_chisq(theta)
    
    return regulate_PSF(psf_image,oversampling)

def fittingPSF_new(cutPSF,cutMask,cutSigma=None,oversampling = 1,a_TV = 0.1, verbose = False,  learning_rate=1e-4, num_steps=14000, perturbate = 1. , makestackpsf = False, fit_method = 'adam'):
    N_star = len(cutPSF)
    norm = []
    dof = 0
    for loop in range(N_star):
        cutPSF[loop] = jnp.where(cutMask[loop],0,cutPSF[loop])
        if loop == 0:
            sumimage = cutPSF[loop] 
        else:
            sumimage += cutPSF[loop] 
        dof += np.sum(cutMask[loop]==False)
        norm.append(jnp.sum(cutPSF[loop]))
    if cutSigma is None:
        cutSigma = jnp.ones(N_star,dtype=jnp.float32)
    ny0,nx0 = cutPSF[0].shape
    ny = oversampling*ny0
    nx = oversampling*nx0
    mask = np.zeros((ny,nx),dtype=bool)
    circle = Maskellipse(mask,[0.5*ny,0.5*nx],0.5*nx,0,0)
    circle = jnp.array(circle,dtype=bool)
    isize = ny*nx
    stackpsf = sumimage/(jnp.sum(sumimage))
    if makestackpsf:
        return regulate_PSF(stackpsf)
    if oversampling > 1:
        overpsf = jax.image.scale_and_translate(stackpsf,(ny,nx),(0,1),jnp.array([oversampling,oversampling]),jnp.array([0,0]),'cubic')/oversampling**2
    else:
        overpsf = stackpsf
    
    def cal_chisq(theta):
        psf_flat = theta[0:isize]
        psf_image = jnp.reshape(psf_flat,(ny,nx))
        psf_image = jnp.exp(psf_image)
        TV_loss = loss_total_variation_isotropic(psf_image)
        Norm = theta[isize:isize+N_star]
        xy = theta[isize+N_star:isize+3*N_star]
        for loop in range(N_star):
            resi = cutPSF[loop] - Norm[loop]*pro.add_PSF(xy[2*loop],xy[2*loop+1],nx0,ny0,psf_image,scalefactor=1./oversampling)
            resi = resi/cutSigma[loop]
            resi = jnp.where(cutMask[loop],0,resi)
            if loop == 0:
                residu_flat = resi.ravel()
            else:
                residu_flat=jnp.append(residu_flat,resi.ravel())
        chisq = 0.5*jnp.sum(residu_flat*residu_flat)+a_TV*TV_loss 
        return chisq
    
    theta0 = jnp.zeros(isize+3*N_star)
    flat = jnp.array(overpsf.ravel())
    flat = jnp.abs(flat)
    flat = jnp.log(flat+1e-8)
    theta0 = theta0.at[0:isize].set(flat) 
    theta0 = theta0.at[isize:isize+N_star].set(jnp.array(norm)) 
    theta0 = theta0.at[isize+N_star:isize+3*N_star].set(0.5*perturbate*nx0+0.5*random.normal(jaxkey, shape = (2*N_star,))) 
    
    if fit_method == 'adam':
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(theta0)
        @jax.jit
        def step(logitu, opt_state):
            g = jax.grad(cal_chisq)(logitu)
            updates, opt_state = optimizer.update(g, opt_state)
            new_logitu = optax.apply_updates(logitu, updates)
            return new_logitu, opt_state, cal_chisq(logitu)
        logitu = theta0
        for i in range(num_steps):
            logitu, opt_state, loss = step(logitu, opt_state)
            if i % 100 == 0:  # print every 100 steps
                if verbose:
                    print(f"Step {i}, Chisq: {loss}")
                if jnp.isnan(loss):
                    break
                else:
                    theta = logitu

    psf_flat = jnp.exp(theta[0:isize])
    psf_image = jnp.reshape(psf_flat,(ny,nx))

    def cal_chisq(theta):
        psf_flat = theta[0:isize]
        psf_image = jnp.reshape(psf_flat,(ny,nx))
        psf_image = jnp.exp(psf_image)
        TV_loss = loss_total_variation_isotropic(psf_image)
        Norm = theta[isize:isize+N_star]
        xy = theta[isize+N_star:isize+3*N_star]
        for loop in range(N_star):
            resi = cutPSF[loop] - Norm[loop]*pro.add_PSF(xy[2*loop],xy[2*loop+1],nx0,ny0,psf_image,scalefactor=1./oversampling)
            resi = resi/cutSigma[loop]
            resi = jnp.where(cutMask[loop],0,resi)
            if loop == 0:
                residu_flat = resi.ravel()
            else:
                residu_flat=jnp.append(residu_flat,resi.ravel())
            print (loop,jnp.sum(resi.ravel()*resi.ravel()))
        print (0.5*jnp.sum(residu_flat*residu_flat)/dof, a_TV*N_star*TV_loss/dof)
        return 
    cal_chisq(theta)
    
    return regulate_PSF(psf_image,oversampling)

def regulate_PSF(epsf,oversampling=1,subfrac = 50, normalize = True, xys=None, interpolate=False):
    """
    Post-process PSF: recenter, remove sky pedestal, normalize.
    
    Finalizes PSF construction by centering peak at array center, subtracting
    background pedestal, masking edges, and normalizing to unit sum. Essential
    for accurate PSF convolution in galaxy modeling.

    Parameters
    ----------
    epsf : ndarray
        Input PSF array, potentially off-center or unnormalized.
    oversampling : int, default: 1
        PSF oversampling factor. Edges within `oversampling` pixels are zeroed
        to avoid wrap-around artifacts.
    subfrac : float, default: 50
        Percentile for sky level estimation from outer (non-circular) regions.
        Subtracted from PSF to remove pedestal.
    normalize : bool, default: True
        If True, normalize PSF to sum=1 after processing.
    xys : ndarray, optional
        Array of [x, y] positions (shape N×2) to shift by same amounts as PSF
        recentering. Used to update star positions after PSF adjustment.
    interpolate : bool, default: False
        If True, use bi-cubic spline interpolation for sub-pixel recentering
        (photutils method). If False, use integer pixel roll (faster).

    Returns
    -------
    epsf : ndarray
        Processed PSF, centered and normalized.
    xys1 : ndarray, optional
        Updated positions (only returned if `xys` provided).

    Notes
    -----
    **Processing Steps**:
    
    1. **Estimate sky pedestal**:
       - Create circular mask with radius = 0.5 * image size
       - Compute `subfrac` percentile of pixels outside circle
       - Subtract from entire PSF
    
    2. **Recenter PSF**:
       - Find brightest pixel
       - Shift to array center via:
         * Integer roll if `interpolate=False` (preserves flux exactly)
         * Bi-cubic spline if `interpolate=True` (sub-pixel accuracy, smooths slightly)
    
    3. **Mask edges**:
       - Zero out pixels within `max(oversampling, shift_distance//2 + oversampling)`
         of edges to remove artifacts from rolling/interpolation
    
    4. **Normalize** (if `normalize=True`):
       - Scale PSF so `sum(PSF) = 1`
    
    **When to Use Interpolation**:
    
    - `interpolate=False`: Default, preserves Poisson noise properties
    - `interpolate=True`: Better centering accuracy for undersampled PSFs, but
      introduces slight smoothing
    
    If `xys` provided, applies inverse of PSF shift to all positions, keeping
    star cutouts aligned with new PSF center.

    Examples
    --------
    Basic PSF regulation after fitting:
    
    >>> raw_psf = fittingPSF(cutouts, masks, oversampling=2)
    >>> psf = regulate_PSF(raw_psf, oversampling=2, normalize=True)
    >>> print(psf.sum())
    1.0
    
    Recenter PSF and update star positions:
    
    >>> xys = np.array([[15.3, 15.7], [15.1, 16.2], ...])  # star centers
    >>> psf, xys_updated = regulate_PSF(raw_psf, xys=xys, interpolate=True)
    >>> print(f"Shift applied: {xys_updated[0] - xys[0]}")
    
    Aggressive sky subtraction for PSF with strong background:
    
    >>> psf = regulate_PSF(raw_psf, subfrac=20, normalize=True)
    
    See Also
    --------
    fittingPSF : JAX-based PSF construction
    photutils.centroids.centroid_com : Center-of-mass centroiding
    scipy.interpolate.RectBivariateSpline : Sub-pixel shifting
    """
    shifty = 0
    shiftx = 0
    if interpolate==False: 
        epsf = np.array(epsf,dtype=np.float32)
        size,_ = epsf.shape
        mask = np.zeros((size,size),dtype=bool)
        circle = Maskellipse(mask,[0.5*size,0.5*size],0.5*size,0,0)
        skylevel = np.percentile(epsf[~circle],subfrac)
        epsf -= skylevel
        #epsf[epsf < 0.] = 0
        epsf[0:oversampling,:] = 0
        epsf[-oversampling:,:] = 0
        epsf[:,0:oversampling] = 0
        epsf[:,-oversampling:] = 0
        centerpix = [size//2,size//2]
        # move max pixel to center
        maxpix = np.unravel_index(np.argmax(epsf), epsf.shape)
        shiftx = centerpix[0] - maxpix[0]
        shifty = centerpix[1] - maxpix[1]
        print (f"Shift PSF by {shiftx} pixels in x and {shifty} pixels in y")
        epsf = np.roll(epsf,shiftx,axis=0)
        epsf = np.roll(epsf,shifty,axis=1)
        regur = int(0.5*np.sqrt(shiftx**2+shifty**2)+oversampling)
        epsf[0:regur,:] = 0
        epsf[-regur:,:] = 0
        epsf[:,0:regur] = 0
        epsf[:,-regur:] = 0
    else: 
        def shift_image_bivariate_spline(image, shift_y, shift_x):
            ny, nx = image.shape
            y = jnp.arange(ny)
            x = jnp.arange(nx)
            spline = RectBivariateSpline(y, x, image)
            y_shifted = y - shift_y
            x_shifted = x - shift_x
            shifted_image = spline.ev(x_shifted[:, jnp.newaxis], y_shifted[jnp.newaxis, :])
            return shifted_image
        ny, nx = epsf.shape
        y, x = centroid_com(epsf)
        shifty = ny//2 - y
        shiftx = nx//2 - x
        # By default, Photutils called this function for 15 times for each intermediate psf, but in our case 1 time should be enough because this is the final psf
        epsf = shift_image_bivariate_spline(epsf, shift_y=shifty, shift_x=shiftx)
    if normalize:
        epsf /= np.sum(epsf)
    if xys is not None:
        xys1 = np.array(xys,dtype=np.float32)
        for i in range(len(xys)//2):
            xys1[2*i] = xys[2*i] - shifty
            xys1[2*i+1] = xys[2*i+1] - shiftx
        return epsf,xys1
    else:
        return epsf

def fit_star(cutpsf, cutmask, cutsigma, psf_image, oversampling, learning_rate, num_steps):
   
    cutpsf = jnp.where(cutmask, 0, cutpsf)
    dof = jnp.sum(~cutmask)
    ny0, nx0 = cutpsf.shape

    def cal_chisq(theta):
        resi = cutpsf - theta[0] * pro.add_PSF(
            theta[1], theta[2], nx0, ny0, psf_image, scalefactor=1. / oversampling
        ) - theta[3]
        resi = jnp.where(cutmask, 0, resi)
        residu_flat = resi.ravel() / cutsigma.ravel()
        chisq = 0.5 * jnp.sum(residu_flat * residu_flat)
        return chisq

    theta0 = jnp.array([jnp.sum(cutpsf)*oversampling**2, 0.5 * nx0, 0.5 * ny0, 0.])
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(theta0)

    @jax.jit
    def step(theta, opt_state):
        g = jax.grad(cal_chisq)(theta)
        updates, opt_state = optimizer.update(g, opt_state)
        new_theta = optax.apply_updates(theta, updates)
        return new_theta, opt_state, cal_chisq(theta)

    theta = theta0
    lossgood = 0.
    for i in range(num_steps):
        theta, opt_state, loss = step(theta, opt_state)
        if i % 100 == 0:
            if jnp.isnan(loss):
                break
            else:
                #print(f"Step {i}, Chisq: {loss}")
                lossgood = loss

    # Compute the model image with optimized parameters
    model_image = theta[0] * pro.add_PSF(
        theta[1], theta[2], nx0, ny0, psf_image, scalefactor=1. / oversampling
    ) + theta[3]

    return model_image, lossgood, dof

def test_PSF(cutPSF, cutMask, cutSigma, psf_image, oversampling=1, learning_rate=5e-4, num_steps=15000):
    lossgood_total = 0.
    model_images = []
    DOF = 0
    for i in range(len(cutPSF)):
        model_image, lossgood, dof = fit_star(
            cutPSF[i], cutMask[i], cutSigma[i], psf_image, oversampling, learning_rate, num_steps
        )
        lossgood_total += lossgood
        avg_loss = lossgood / dof
        print(f"Average loss for item {i}: {avg_loss}")
        DOF += dof
        model_images.append(model_image)
    print("Total loss:", lossgood_total)
    print("Overall average loss:", lossgood_total / DOF)
    return model_images

def mapping_mask(img1, img2, mask1, grow = 1):
    # transform the mask1 in img1's cordinate to img2's cordinate
    # img1 and img2 are 2D array
    mask1 = np.array(mask1,dtype=bool)
    masky,maskx = np.where(mask1)
    hdu = fits.open(img1)
    header1 = hdu[0].header
    wcs1 = WCS(header1)
    hdu = fits.open(img2)
    header2 = hdu[0].header
    shape2 = hdu[0].data.shape
    wcs2 = WCS(header2)
    x1,y1 = wcs1.all_pix2world(maskx,masky,1)
    x2,y2 = wcs2.all_world2pix(x1,y1,1)
    x2 = np.array(x2,dtype=np.int32)
    y2 = np.array(y2,dtype=np.int32)
    pixelin = np.where((x2>=0)&(x2<shape2[1])&(y2>=0)&(y2<shape2[0]))
    x2 = x2[pixelin]
    y2 = y2[pixelin]
    mask2 = np.zeros(hdu[0].data.shape,dtype=bool)
    mask2[y2,x2] = True
    # grow the mask
    if grow > 0:
        mask2 = np.array(ndimage.binary_dilation(mask2, iterations=grow), dtype = np.int32)
    return mask2

__all__ = ['image', 'image_atlas']

def get_wcs_rotation(wcs):
    """
    Compute position angle of WCS coordinate frame.
    
    Calculates the rotation of the pixel grid relative to equatorial coordinates
    (RA/Dec), measured as position angle east of north. Essential for matching
    galaxy model orientations to observed images.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        World Coordinate System object with CD or PC matrix.

    Returns
    -------
    rot : astropy.units.Quantity (angle)
        Position angle in radians. Positive values indicate counterclockwise
        rotation from north toward east in sky coordinates.

    Notes
    -----
    **Calculation Method**:
    
    Extracts the CD matrix (or PC * CDELT) and computes rotation via:
    
    .. math::
    
        {{\\rm PA}} = \\arctan2({{\\rm sgn}} \\cdot CD_{10}, {{\\rm sgn}} \\cdot CD_{00})
    
    where :math:`{{\\rm sgn}} = {{\\rm sign}}({{\\rm det}}(CD))` ensures correct
    orientation for parity-flipped coordinates (e.g., FITS convention with decreasing
    RA from left to right).
    
    **Position Angle Convention**:
    
    - 0° = North (up in sky coordinates)
    - 90° = East (left in RA/Dec, right in sky Az/El)
    - 180° = South
    - 270° = West
    
    This matches the standard astronomical convention for slit/galaxy position angles.
    
    **Applicability**:
    
    Works for:
    - Simple rotations + scaling (most HST/JWST images)
    - Distorted WCS (returns average rotation)
    
    Does not account for:
    - Non-linear distortions (SIP, TPV polynomials)
    - Differential rotation across field
    
    For modeling, the rotation is typically applied to the entire cutout region,
    so local variations are acceptable.

    Examples
    --------
    Get PA for HST/ACS image:
    
    >>> from astropy.io import fits
    >>> hdu = fits.open('hst_acs.fits')
    >>> wcs = WCS(hdu[0].header)
    >>> pa = get_wcs_rotation(wcs)
    >>> print(f"Image PA: {pa.to('deg'):.2f}")
    
    Check if image is north-up:
    
    >>> pa = get_wcs_rotation(wcs)
    >>> if abs(pa.to('deg').value) < 1:
    ...     print("Image is approximately north-up")
    
    See Also
    --------
    astropy.wcs.utils.proj_plane_pixel_scales : Pixel scale extraction
    galfits.galaxy.coordinates_transfer : Apply rotation to model coordinates
    """
    cdmat = wcs.pixel_scale_matrix
    sgn = np.sign(np.linalg.det(cdmat))
    rot = np.arctan2(sgn * cdmat[1, 0], sgn * cdmat[0, 0]) * u.radian
    return rot

def generate_3D_PSF(data,reference_image,emap,psfFWHM,wcs,throughput,nsigma=2.0, a_TV = 1e6, a_sky=1., fluxthreshould = 2, isolate = 0.1, qt = 0.8, sigfwhm = 2,
                    size = 31, oversampling=1, plot=True, filepaths=None, droppsf=[], notfit=False, crossmatch = False, makestackpsf = False):
    sky_mean, sky_median, sky_std = sigma_clipped_stats(reference_image, sigma=3.0, maxiters=5)
    '''
    Generate 3D PSF from IFS data cube. Unfinished.
    data: 2D array, IFS image at a single wavelength (slice)
    reference_image: 2D array, reference image for source detection. Must have the same shape as data. Can be white light image or continuum image.
    emap: 2D array, exposure time map at a single wavelength (slice)
    wcs: celestial wcs for IFS cube
    Currently still restricted to 2D PSF generation. Can be vectorized so as to generate 3D PSF
    '''
    reference_image -= sky_median
    mask = np.zeros_like(reference_image, dtype=bool)
    mask[np.isnan(reference_image)] = True
    mask[np.isinf(reference_image)] = True
    psfFWHMpix = psfFWHM / np.abs(wcs.pixel_scale_matrix[0,0] * 3600)
    thresholder = detect_threshold(reference_image, nsigma=nsigma,mask=mask)
    sigma = psfFWHMpix * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
    kernel.normalize()
    segm = detect_sources(reference_image, thresholder, npixels=5,mask=mask)
    segm_data = segm.data
    props = SourceCatalog(reference_image, segm)
    stars = Table(props.to_table(['xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation','segment_flux','fwhm','maxval_xindex','maxval_yindex']))
#    wcs = self.data.wcs
    ra, dec = wcs.all_pix2world(stars['xcentroid'], stars['ycentroid'], 0)
    Column_ra = Column(name='RA', data=ra)
    Column_dec = Column(name='DEC', data=dec)
    stars.add_columns([Column_ra, Column_dec])
    if crossmatch:
        t_o = XMatch.query(cat1=stars, cat2='vizier:I/355/gaiadr3', max_distance=1.5*psfFWHM*u.arcsec, 
                        colRA1='RA', colDec1='DEC')  # Gaia xmatch.
    else:
        t_o = stars
    if len(t_o) == 0:
        print ("original table have length {0}, become 0 after Gaia cross match".format(len(stars)))
        t_o = stars
    # filter source > fluxthreshould%
    print ('total sources: ',len(t_o))
    t_o = t_o[t_o['segment_flux'] > fluxthreshould*np.mean(t_o['segment_flux'])]
    print ('sources above thereshold flux :',len(t_o))
    q = t_o['semiminor_sigma']/t_o['semimajor_sigma']
    t_o = t_o[q>qt]
    print ('sources above thereshold axis ratio :',len(t_o))
#    mean, median, std = sigma_clipped_stats(np.array(t_o['fwhm']), sigma=3.0, maxiters=5)
 #   t_o = t_o[t_o['fwhm'] < median+sigfwhm*std]
 #   print(t_o['fwhm'])
 #   print ('sources within threshould fwhm :',len(t_o))



    cutPSF = []
    cutMask = []
    posi_sequence = []
    for i in range(len(t_o)):
        x = t_o['maxval_xindex'][i]
        y = t_o['maxval_yindex'][i]
        nlabel = segm_data[y,x]
        minx = np.max([0,x-size//2])
        maxx = np.min([reference_image.shape[1],x+size//2+1])
        miny = np.max([0,y-size//2])
        maxy = np.min([reference_image.shape[0],y+size//2+1])
        reachbound = False
        if minx == 0 or maxx == data.shape[1] or miny == 0 or maxy == data.shape[0]:
            reachbound = True
        localdata = reference_image[miny:maxy,minx:maxx]
        # scale the count rate by exposure time map
        localsege = segm_data[miny:maxy,minx:maxx]
        localmask = mask[miny:maxy,minx:maxx]
        localmask[(localsege!=nlabel)&(localsege!=0)] = True
        localmask[np.isnan(localdata)] = True
        localmask[np.isinf(localdata)] = True
        if np.sum(localmask/(size**2)) > isolate or reachbound:
            continue
        #cutPSF.append(jnp.array(localdata,dtype=jnp.float32))
        posi_sequence.append(jnp.array([miny,maxy,minx,maxx]))
        cutMask.append(jnp.array(localmask,dtype=bool))
    print ('sources isolate :',len(cutPSF))

#   circle = np.zeros([size,size],dtype=bool)
#   circle1 = Maskellipse(circle,[0.5*size,0.5*size],0.25*size,0,0)
#   removeloop = []
#   N.B. over-ride the central bad removal routine at this time
#    for loopx in range(len(cutPSF)):
#        lmask = np.array(cutMask[loopx])
#        if np.sum(lmask[circle1]) > 0:
#            removeloop.append(loopx)
#    cutPSF = [i for j, i in enumerate(cutPSF) if j not in removeloop]
#    cutMask = [i for j, i in enumerate(cutMask) if j not in removeloop]
#    print ('sources remove central bad :',len(cutPSF))

    if len(droppsf) > 0:
        cutPSF = [i for j, i in enumerate(cutPSF) if j not in droppsf]
        posi_sequence = [i for j, i in enumerate(posi_sequence) if j not in droppsf]
        cutMask = [i for j, i in enumerate(cutMask) if j not in droppsf]


    if plot:
        tlens = len(cutPSF)
        if (((tlens//5)+1)*5-tlens) < (((tlens//4)+1)*4-tlens):
            ncols = 5
            nrows = (tlens//5)+1 
        else:
            ncols = 4
            nrows = (tlens//4)+1
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows),squeeze=True)
        ax = ax.ravel()
        for i in range(tlens):
            image = cutPSF[i]
            norm = simple_norm(image, 'log', percent=99.)
            ax[i].imshow(image,cmap='Greys', origin='lower', norm=norm,interpolation='nearest')
            ax[i].imshow(np.array(cutMask[i],dtype=float), cmap='Blues', origin='lower', alpha=0.5*np.array(cutMask[i],dtype=float))
            ax[i].text(0.95, 0.05, str(i), color='r',transform=ax[i].transAxes,ha='right', va='bottom')
        plt.show()

    if notfit:
        return None
    
    epsf = jnp.vmap(generate_3D_PSF_sub,in_axes=(0,0,None,None))(data,emap,cutMask,posi_sequence, a_TV = a_TV, a_sky=a_sky, oversampling=oversampling, makestackpsf = makestackpsf)
    

    if filepaths is not None:
        hdu = fits.PrimaryHDU(epsf)
        After = fits.HDUList([hdu])
        After.writeto(filepaths,overwrite= True)

    if plot:
        fig=plt.figure(figsize=(10,10))
        image = epsf
        norm = simple_norm(image, 'log', percent=99.)
        plt.imshow(image,cmap='Greys', origin='lower', norm=norm,interpolation='nearest')
        plt.show()

    return epsf

def generate_3D_PSF_sub(data_slice,emap_slice,cutMask,posi_sequence, a_TV = 1e6, a_sky=1., oversampling=1, makestackpsf = False):
    cutPSF = []
    for  j , posi in enumerate(posi_sequence):
        miny,maxy,minx,maxx = posi
        localdata = data_slice[miny:maxy,minx:maxx]
        localemap = emap_slice[miny:maxy,minx:maxx]
        # scale the count rate by exposure time map
        weight=localemap/localemap.sum()
        localdata = localdata/weight
        cutPSF.append(jnp.array(localdata,dtype=jnp.float32))
    epsf = fittingPSF(cutPSF,cutMask,oversampling = oversampling, a_TV = a_TV, a_sky=a_sky, makestackpsf = makestackpsf)
    return epsf

def generate_ISF_PSF(data,emap,psfFWHM,wcs,nsigma=2.0, a_TV = 1e6, a_sky=1., fluxthreshould = 2, isolate = 0.1, qt = 0.8, sigfwhm = 2,
                    size = 31, oversampling=1, plot=True, filepaths=None, droppsf=[], notfit=False, crossmatch = False, makestackpsf = False):
    '''
    Generate ISF PSF using the same method as generate_3D_PSF, but only generate 2D PSF
    read flattened data and emap
    '''
    sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    data -= sky_median
    mask = np.zeros_like(data, dtype=bool)
    mask[np.isnan(data)] = True
    mask[np.isinf(data)] = True
    psfFWHMpix = psfFWHM / np.abs(wcsinfo.pixel_scale_matrix[0,0] * 3600)
    thresholder = detect_threshold(data, nsigma=nsigma,mask=mask)
    sigma = psfFWHMpix * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
    kernel.normalize()
    segm = detect_sources(data, thresholder, npixels=5,mask=mask)
    segm_data = segm.data
    props = SourceCatalog(data, segm)
    stars = Table(props.to_table(['xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation','segment_flux','fwhm','maxval_xindex','maxval_yindex']))
#    wcs = self.data.wcs
    ra, dec = wcs.all_pix2world(stars['xcentroid'], stars['ycentroid'], 0)
    Column_ra = Column(name='RA', data=ra)
    Column_dec = Column(name='DEC', data=dec)
    stars.add_columns([Column_ra, Column_dec])
    if crossmatch:
        t_o = XMatch.query(cat1=stars, cat2='vizier:I/355/gaiadr3', max_distance=1.5*psfFWHM*u.arcsec, 
                        colRA1='RA', colDec1='DEC')  # Gaia xmatch.
    else:
        t_o = stars
    if len(t_o) == 0:
        print ("original table have length {0}, become 0 after Gaia cross match".format(len(stars)))
        t_o = stars
    # filter source > fluxthreshould%
    print ('total sources: ',len(t_o))
    t_o = t_o[t_o['segment_flux'] > fluxthreshould*np.mean(t_o['segment_flux'])]
    print ('sources above thereshold flux :',len(t_o))
    q = t_o['semiminor_sigma']/t_o['semimajor_sigma']
    t_o = t_o[q>qt]
    print ('sources above thereshold axis ratio :',len(t_o))
#    mean, median, std = sigma_clipped_stats(np.array(t_o['fwhm']), sigma=3.0, maxiters=5)
 #   t_o = t_o[t_o['fwhm'] < median+sigfwhm*std]
 #   print(t_o['fwhm'])
 #   print ('sources within threshould fwhm :',len(t_o))
    cutPSF = []
    cutMask = []
    for i in range(len(t_o)):
        x = t_o['maxval_xindex'][i]
        y = t_o['maxval_yindex'][i]
        nlabel = segm_data[y,x]
        minx = np.max([0,x-size//2])
        maxx = np.min([data.shape[1],x+size//2+1])
        miny = np.max([0,y-size//2])
        maxy = np.min([data.shape[0],y+size//2+1])
        reachbound = False
        if minx == 0 or maxx == data.shape[1] or miny == 0 or maxy == data.shape[0]:
            reachbound = True
        localdata = data[miny:maxy,minx:maxx]
        localemap = emap[miny:maxy,minx:maxx]
        # scale the count rate by exposure time map
        weight=localemap/localemap.sum()
        localdata = localdata/weight
        localsege = segm_data[miny:maxy,minx:maxx]
        localmask = mask[miny:maxy,minx:maxx]
        localmask[(localsege!=nlabel)&(localsege!=0)] = True
        localmask[np.isnan(localdata)] = True
        localmask[np.isinf(localdata)] = True
        if np.sum(localmask/(size**2)) > isolate or reachbound:
            continue
        cutPSF.append(jnp.array(localdata,dtype=jnp.float32))
        cutMask.append(jnp.array(localmask,dtype=bool))
    print ('sources isolate :',len(cutPSF))

    circle = np.zeros([size,size],dtype=bool)
    circle1 = Maskellipse(circle,[0.5*size,0.5*size],0.25*size,0,0)
    removeloop = []
#   over-ride the central bad removal routine for this time
#    for loopx in range(len(cutPSF)):
#        lmask = np.array(cutMask[loopx])
#        if np.sum(lmask[circle1]) > 0:
#            removeloop.append(loopx)
    cutPSF = [i for j, i in enumerate(cutPSF) if j not in removeloop]
    cutMask = [i for j, i in enumerate(cutMask) if j not in removeloop]
    print ('sources remove central bad :',len(cutPSF))

    if len(droppsf) > 0:
        cutPSF = [i for j, i in enumerate(cutPSF) if j not in droppsf]
        cutMask = [i for j, i in enumerate(cutMask) if j not in droppsf]


    if plot:
        tlens = len(cutPSF)
        if (((tlens//5)+1)*5-tlens) < (((tlens//4)+1)*4-tlens):
            ncols = 5
            nrows = (tlens//5)+1 
        else:
            ncols = 4
            nrows = (tlens//4)+1
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows),squeeze=True)
        ax = ax.ravel()
        for i in range(tlens):
            image = cutPSF[i]
            norm = simple_norm(image, 'log', percent=99.)
            ax[i].imshow(image,cmap='Greys', origin='lower', norm=norm,interpolation='nearest')
            ax[i].imshow(np.array(cutMask[i],dtype=float), cmap='Blues', origin='lower', alpha=0.5*np.array(cutMask[i],dtype=float))
            ax[i].text(0.95, 0.05, str(i), color='r',transform=ax[i].transAxes,ha='right', va='bottom')
        plt.show()

    if notfit:
        return None

    epsf = fittingPSF(cutPSF,cutMask,oversampling = oversampling, a_TV = a_TV, a_sky=a_sky, makestackpsf = makestackpsf)

    if filepaths is not None:
        hdu = fits.PrimaryHDU(epsf)
        After = fits.HDUList([hdu])
        After.writeto(filepaths,overwrite= True)

    if plot:
        fig=plt.figure(figsize=(10,10))
        image = epsf
        norm = simple_norm(image, 'log', percent=99.)
        plt.imshow(image,cmap='Greys', origin='lower', norm=norm,interpolation='nearest')
        plt.show()

    return epsf


class image(object):
    '''
    A single image object.

    Functions
    ---------
    * Read from fits file use CCDData.
    * get_size : Get the image size.
    * plot : Plot the image.
    * sigma_clipped_stats : Calculate the basic statistics of the image.
    * set_data : Load from numpy array.
    * set_mask : Set image mask.
    * set_pixel_scales : Set the pixel scales along two axes.
    * set_zero_point : Set magnitude zero point.
    '''
    def __init__(self, filename=None, IFS=False, hdu=0,wcs=None, unit=None, zero_point=None, pixel_scales=None, wcs_rotation=None, mask=None,psfFWHM_map=None, fitSED = True):
        '''
        Parameters
        ----------
        filename (optional) : string
            FITS file name of the image.
        IFS : bool (default: False)
            If the image is a IFS(integral field spectrograph) data cube.
        hdu : int or string (default: 0)
            The number of extension to load from the FITS file.
        unit (optional) : string
            Unit of the image flux for CCDData.
        zero_point (optional) : float
            Magnitude zero point.
        pixel_scales (optional) : tuple
            Pixel scales along the first and second directions, units: arcsec.
        wcs_rotation (optional) : float
            WCS rotation, east of north, units: radian.
        mask (optional) : 2D bool array
            The image mask.
        verbose : bool (default: True)
            Print out auxiliary data.
        '''
        if filename is None:
            self.data = None
        elif IFS:
            from spectral_cube import SpectralCube
            self.cube = SpectralCube.read(filename)
            self.data = SpectralCube.read(filename).unmasked_data[:,:,:]
            self.data.mask = SpectralCube.read(filename).mask
            hdul = fits.open(filename)
            header = hdul[hdu].header
            self.data.header = header
            if wcs is not None:
                self.data.wcs = wcs
            else:
                self.data.wcs = WCS(self.data.header)
        else:
            self.data = CCDData.read(filename, hdu=hdu, unit=unit, mask=mask)
            hdul = fits.open(filename)
            header = hdul[hdu].header
            self.data.header = header
            if 'photop' in header:
                self.data.wcs = None
                self.photopoint = True
            else:
                if wcs is not None:
                    self.data.wcs = wcs
                else:
                    self.data.wcs = WCS(self.data.header,naxis=2)
                self.photopoint = False
            # naxis=2 is to avoid the error of WCS when reading squashed IFS data,
            # taking only the first two axes (spatial axes)          
        if self.data.wcs and (pixel_scales is None):
            pixel_scales = proj_plane_pixel_scales(self.data.wcs) * u.degree.to('arcsec')
        if self.data is not None:  
            if (pixel_scales is  None) & (not self.photopoint):
                a1 = abs(self.data.header['CDELT1']) * 3600.
                a2 = abs(self.data.header['CDELT2']) * 3600.
                self.pixel_scales = (a1*u.arcsec, a2*u.arcsec)

        if pixel_scales is not None:
            self.pixel_scales = (pixel_scales[0]*u.arcsec, pixel_scales[1]*u.arcsec)

        self.zero_point = zero_point
        #if self.data.wcs and (wcs_rotation is None):
        #    self.wcs_rotation = get_wcs_rotation(self.data.wcs)
        if wcs_rotation is not None:
            self.wcs_rotation = wcs_rotation * u.radian
        else:
            self.wcs_rotation = None
        self.grism_cut = False
        self.grism_cpi = None
        self.grism_cpg = None 
        if self.data.header is not None:
            if self.data.header.get('CUT') == True:
                self.grism_cut = True
                try:
                    self.grism_cpg = [self.data.header['CPG_XMIN'],self.data.header['CPG_XMAX'],
                                      self.data.header['CPG_YMIN'],self.data.header['CPG_YMAX']]
                except KeyError:
                    pass
                try:
                    self.grism_cpi = [self.data.header['CPI_XMIN'],self.data.header['CPI_XMAX'],
                                      self.data.header['CPI_YMIN'],self.data.header['CPI_YMAX']]
                except KeyError:
                    pass
        self.sources_catalog = None
        self.sigma_image = None
        self.sources_skycord = None
        self.sources_skycord_cut = None
        self.ss_data = None
        self.PSF = None
        self.psfFWHM_map=psfFWHM_map
        self.psfFWHM_arcsec = None
        self.cut_image = None
        self.cut_sigma_image = None
        self.cut_mask_image = None
        self.coordinates_transfer_para = None
        self.cshape0 = None
        self.sky_median = 0.
        self.phys_to_counts_rate = 1.
        self.band = None
        self.model_image = None
        self.wcsshift = False
        self.magzp = 20.
        self.fitSED = fitSED
        self.skycord = None
        self.disperser = None
        self.dcord = [0., 0.]

    def __del__(self):
        # Nullify object's properties
        self.data = None
        self.pixel_scales = None
        self.zero_point = None
        self.wcs_rotation = None
        self.sources_catalog = None
        self.sigma_image = None
        self.sources_skycord = None
        self.sources_skycord_cut = None
        self.ss_data = None
        self.PSF = None
        self.psfFWHM_map = None
        self.psfFWHM_arcsec = None
        self.cube = None
        self.cut_image = None
        self.cut_sigma_image = None
        self.cut_mask_image = None
        self.coordinates_transfer_para = None
        self.sky_median = None
        self.phys_to_counts_rate = None
        self.band = None
        self.model_image = None
        self.wcsshift = None
        self.magzp = None
        self.fitSED = None
        self.disperser = None

    def img_cut(self,ra,dec,cutsize,cutposition=None,IFS=False,usemaxpix = False, sigma_clipped=True,  referpix = None,zoomsize = 100, spcorrection=[0.,0.]):
        """
        Extract image cutout around a celestial position.
        
        This method creates a cutout from the full image centered on specified RA/Dec 
        coordinates, handles coordinate transformation between world and pixel coordinates,
        and calculates transformation parameters needed for model generation. It can
        optionally recenter on the brightest pixel within a search region.

        Parameters
        ----------
        ra : float
            Right ascension of the source position (degrees).
        dec : float
            Declination of the source position (degrees).
        cutsize : float or array-like
            Cut size in arcsec. If float, symmetric box. If array [xmin, xmax, ymin, ymax],
            asymmetric cutout around source position.
        cutposition : list of int, optional
            Direct pixel coordinates [minx, maxx, miny, maxy]. If provided, overrides
            `cutsize` calculation.
        IFS : bool, default: False
            If True, treat data as IFS (Integral Field Spectrograph) cube with shape
            (nwave, ny, nx).
        usemaxpix : bool, default: False
            If True, recenter cutout on brightest pixel within `zoomsize` of initial
            position. Useful when catalog position is inaccurate.
        sigma_clipped : bool, default: True
            If True, estimate sky median using sigma-clipped statistics. Otherwise use
            simple median.
        referpix : list of float, optional
            Reference pixel [x, y] for centering search when `usemaxpix=True`. If None,
            uses WCS-derived position from RA/Dec.
        zoomsize : float, default: 100
            Search radius in arcsec for brightest pixel when `usemaxpix=True`.
        spcorrection : list of float, default: [0., 0.]
            Manual position correction [dx, dy] in pixels to apply to WCS-derived
            position before cutout.

        Returns
        -------
        imcut : jax.numpy.ndarray
            Cutout image with shape (ny, nx) or (nwave, ny, nx) for IFS data.
        cutposition : list of int
            Pixel bounds [minx, maxx, miny, maxy] of the cutout in the full image.

        Notes
        -----
        This method sets several instance attributes used for modeling:
        
        - `cut_image` : The extracted cutout
        - `sources_skycord` : [x, y] position in full image pixel coordinates
        - `sources_skycord_cut` : [x, y] position in cutout pixel coordinates
        - `coordinates_transfer_para` : Dictionary with coordinate transformation info::
        
            {
                'x0': float,        # X position in cutout (pixels)
                'x0shift': float,   # X alignment offset (pixels)
                'y0': float,        # Y position in cutout (pixels)
                'y0shift': float,   # Y alignment offset (pixels)
                'dxra': float,      # factor of ra contribute to x axis
                'dxdec': float,     # factor of dec contribute to x axis
                'dyra': float,      # factor of ra contribute to y axis
                'dydec': float,     # factor of dec contribute to y axis
                'pixsc': float,     # Pixel scale (arcsec/pixel)
                'delta_ang': float  # Position angle east of north (degrees)
            }
        
        - `sky_median` : Background level estimated from cutout
        - `cut_mask_image` : Mask applied to cutout (if data has mask)
        - `cut_sigma_image` : Uncertainty map for cutout (if sigma_image exists)
        - `dcord` : [dx, dy] offset of CRPIX from cutout origin

        The transformation parameters are calculated from WCS information and allow
        conversion between sky offsets (arcsec) and pixel coordinates in arbitrary
        orientation, essential for modeling rotated galaxies.

        Examples
        --------
        Extract 10 arcsec cutout around a galaxy:
        
        >>> img = image('data.fits')
        >>> cutout, bounds = img.img_cut(ra=150.123, dec=2.456, cutsize=10.0)
        >>> print(cutout.shape)
        (100, 100)  # if pixel scale ~0.1 arcsec
        
        Asymmetric cutout extending further in +x direction:
        
        >>> cutout, bounds = img.img_cut(ra=150.123, dec=2.456, 
        ...                              cutsize=[5, 15, 10, 10])
        
        Recenter on brightest pixel within 2 arcsec:
        
        >>> cutout, bounds = img.img_cut(ra=150.123, dec=2.456, cutsize=20.0,
        ...                              usemaxpix=True, zoomsize=2.0)
        
        See Also
        --------
        write_imcut : Save cutout and ancillary data to FITS file
        read_imcut : Load previously saved cutout
        """
        shape=self.data.shape
        self.skycord = [ra,dec]

        if IFS:
            nz,ny,nx=shape
        else:
            ny,nx=shape
        if cutsize > 0.:
            if IFS:
                srcPstXY = self.data.wcs.celestial.all_world2pix([ra], [dec], 1)
                srcXp = srcPstXY[0][0]
                srcYp = srcPstXY[1][0]
                srcPstXY = self.data.wcs.celestial.all_world2pix([ra+1./60], [dec], 1)
                srcXpra = srcPstXY[0][0]
                srcYpra = srcPstXY[1][0]
                srcPstXY = self.data.wcs.celestial.all_world2pix([ra], [dec+1./60], 1)
                srcXpdec = srcPstXY[0][0]
                srcYpdec = srcPstXY[1][0]
            else:
                srcPstXY = self.data.wcs.all_world2pix([ra], [dec], 1) # unit of ra and dec is "degree"。 pixel coordinate
                srcXp = srcPstXY[0][0]
                srcYp = srcPstXY[1][0]
                srcPstXY = self.data.wcs.all_world2pix([ra+1./60], [dec], 1)
                srcXpra = srcPstXY[0][0]
                srcYpra = srcPstXY[1][0]
                srcPstXY = self.data.wcs.all_world2pix([ra], [dec+1./60], 1)
                srcXpdec = srcPstXY[0][0]
                srcYpdec = srcPstXY[1][0]
            dxra = (srcXpra-srcXp)/60. # pixels / arcsec in ra position.
            dyra = (srcYpra-srcYp)/60.
            dxdec = (srcXpdec-srcXp)/60.
            dydec = (srcYpdec-srcYp)/60.
            self.sources_skycord = [srcXp,srcYp]

            dx, dy = srcXpdec - srcXp, srcYpdec - srcYp
            pa_north = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
            delta_ang = pa_north 
            
            # Normalize the North and East direction vectors
            vN = np.array([dxdec, dydec])
            vE = np.array([dxra,  dyra])

            vN /= np.hypot(*vN)
            vE /= np.hypot(*vE)

            self.skydirection = [vE, vN]  # East, North unit vectors in pixel coordinates

            #''' old version
            #vector = [srcXpdec-srcXp,srcYpdec-srcYp]
            #normv = jnp.sqrt(vector[0]**2+vector[1]**2)
            #delta_ang = np.arcsin((srcYpdec-srcYp)/normv)*180./np.pi
            #'''
            #print ('delta_ang: ', delta_ang)
            #print(f"PA delta_ang: {delta_ang} degrees")
            if np.isnan(delta_ang):
                print ("careful with the delta_ang")
                delta_ang = 90.
            srcXp = srcXp + spcorrection[0]
            srcYp = srcYp + spcorrection[1] 
            if referpix is None:
                referpix = [srcXp,srcYp]
            if usemaxpix:
                data2 = np.array(self.data.data)
                masky = np.zeros((ny,nx),dtype=bool)
                circle = Maskellipse(masky,referpix,zoomsize,0,0)
                print ("before correction:",srcXp,srcYp)
                data2[~circle] = np.nanmin(data2)
                maxpix = np.unravel_index(np.nanargmax(data2), data2.shape)
                srcXp = maxpix[1]
                srcYp = maxpix[0]
                print("after correction:",srcXp,srcYp)
                self.sources_skycord[0] = maxpix[1]
                self.sources_skycord[1] = maxpix[0]

           

            if type(cutsize) is int:
                cutsize = float(cutsize)
            if type(cutsize) is float:
                cutsize_int = int(cutsize/self.pixel_scales[0].value)
                minx = np.max([int(srcXp)-cutsize_int,0])
                maxx = np.min([int(srcXp)+cutsize_int,nx])
                miny = np.max([int(srcYp)-cutsize_int,0])
                maxy = np.min([int(srcYp)+cutsize_int,ny])
            else:
                cutsize_intx1 = int(cutsize[0]/self.pixel_scales[0].value)
                cutsize_intx2 = int(cutsize[1]/self.pixel_scales[0].value)
                cutsize_inty1 = int(cutsize[2]/self.pixel_scales[0].value)
                cutsize_inty2 = int(cutsize[3]/self.pixel_scales[0].value)
                minx = np.max([int(srcXp)-cutsize_intx1,0])
                maxx = np.min([int(srcXp)+cutsize_intx2,nx])
                miny = np.max([int(srcYp)-cutsize_inty1,0])
                maxy = np.min([int(srcYp)+cutsize_inty2,ny])
            if cutposition is not None:
                minx = cutposition[0]
                maxx = cutposition[1]
                miny = cutposition[2]
                maxy = cutposition[3]
            else:
                cutposition = [minx,maxx,miny,maxy]


            self.sources_skycord_cut = [srcXp-minx,srcYp-miny]
            x0 = self.data.header['CRPIX1']
            y0 = self.data.header['CRPIX2']
            #print ('check cords')
            #print (self.sources_skycord, x0, y0, minx, miny)
            self.dcord = [x0-minx, y0-miny]

            
            # case 1: data is a IFS cube; 2: is a 2D image read by CCDData from typical telescope like HST;
            # 3: is a 2D image squashed from IFS data cube in a given wavelength range
            if IFS:
                imcut = jnp.array(self.data[:,miny:maxy,minx:maxx].value,dtype=jnp.float32)
            elif isinstance(self.data, CCDData):
                imcut = jnp.array(self.data.data[miny:maxy,minx:maxx],dtype=jnp.float32)
            else:
                imcut = jnp.array(self.data[miny:maxy,minx:maxx].value,dtype=jnp.float32)
            
            self.cut_image = imcut
            
            #x/y0shift was added for relative alignment calibration between different instruments
            self.coordinates_transfer_para = {'x0': srcXp-minx-0.5,'x0shift':0.0, 'y0': srcYp-miny-0.5,'y0shift':0.0, 'dxra':dxra, 'dxdec':dxdec,'dyra':dyra,'dydec':dydec,'pixsc':self.pixel_scales[0].value, 'delta_ang':delta_ang}
        else:
            minx = 0
            maxx = nx
            miny = 0
            maxy = ny
            cutposition = [minx,maxx,miny,maxy]
            # case 1: image is a IFS data cube; 2: image is a 2D image from typical CCD like HST;
            # 3: image is a 2D image squashed from IFS data cube in a given wavelength range
            if IFS:
                imcut = jnp.array(self.data[:,miny:maxy,minx:maxx].value,dtype=jnp.float32)
            elif isinstance(self.data, CCDData):
                imcut = jnp.array(self.data.data[miny:maxy,minx:maxx],dtype=jnp.float32)
            else:
                imcut = jnp.array(self.data[miny:maxy,minx:maxx],dtype=jnp.float32)
            self.cut_image = imcut
            self.coordinates_transfer_para = {'x0': 0,'x0shift':0.0, 'y0': 0,'y0shift':0.0, 'dxra':1, 'dxdec':0,'dyra':0,'dydec':1,'pixsc':1., 'delta_ang':0.}
        if sigma_clipped:
            sky_mean, sky_median, sky_std = sigma_clipped_stats(imcut, sigma=3.0, maxiters=5)
        else:
            sky_median = np.nanmedian(imcut)
            sky_std = np.nanstd(imcut)
        self.sky_median = sky_median
        if self.data.mask is not None and IFS:
            self.cut_mask_image = self.data.mask[:,miny:maxy,minx:maxx]
        elif self.data.mask is not None:
            self.cut_mask_image = self.data.mask[miny:maxy,minx:maxx]
        if self.sigma_image is not None and IFS:
            self.cut_sigma_image = self.sigma_image[:,miny:maxy,minx:maxx]
        elif self.sigma_image is not None:
            self.cut_sigma_image = self.sigma_image[miny:maxy,minx:maxx]

        return imcut,cutposition


    def img_cut_align(self, ra, dec, cutsize, cutposition=None, IFS=False, usemaxpix=False, 
            sigma_clipped=True, referpix=None, zoomsize=100, spcorrection=[0., 0.]):
        """
        Extract a cutout around a celestial position, accurately resampled to a 
        strict North-Up, East-Left standard WCS grid.
        """



        self.skycord = [ra, dec]
        shape = self.data.shape
        
        wcs_2d = self.data.wcs.celestial if IFS else self.data.wcs
        pix_scale_deg = self.pixel_scales[0].value / 3600.  # Assuming pixel_scales is in arcsec
        
        # ---------------------------------------------------------
        # 1. Target Coordinate Refinement
        # ---------------------------------------------------------
        # Initial pixel guess for target
        init_px, init_py = wcs_2d.all_world2pix(ra, dec, 1)
        target_px = float(init_px + spcorrection[0])
        target_py = float(init_py + spcorrection[1])
        
        if usemaxpix:
            if referpix is None:
                referpix = [target_px, target_py]
            
            # Fast local search: slice a small box to find the max instead of the whole image
            z_rad = int(zoomsize / self.pixel_scales[0].value)
            ymin, ymax = max(0, int(referpix[1])-z_rad), min(shape[-2], int(referpix[1])+z_rad+1)
            xmin, xmax = max(0, int(referpix[0])-z_rad), min(shape[-1], int(referpix[0])+z_rad+1)
            
            search_area = self.data.data[ymin:ymax, xmin:xmax] if hasattr(self.data, 'data') else self.data[ymin:ymax, xmin:xmax]
            # Handle IFS (collapse Z axis to find max)
            if IFS: search_area = np.nansum(search_area, axis=0)
                
            my, mx = np.unravel_index(np.nanargmax(search_area), search_area.shape)
            target_px, target_py = xmin + mx, ymin + my

        # Update absolute center in RA/Dec based on final pixel
        target_ra, target_dec = wcs_2d.all_pix2world(target_px, target_py, 1)
        target_ra, target_dec = float(target_ra), float(target_dec)
        self.sources_skycord = [target_px, target_py]

        # ---------------------------------------------------------
        # 2. Define the Output Shape and Perfect Output WCS
        # ---------------------------------------------------------
        if type(cutsize) in [int, float]:
            cs_arcsec = float(cutsize)
            nx_out = ny_out = int(cs_arcsec / self.pixel_scales[0].value) * 2
        else: # Asymmetric cutsize logic
            nx_out = int((cutsize[0] + cutsize[1]) / self.pixel_scales[0].value)
            ny_out = int((cutsize[2] + cutsize[3]) / self.pixel_scales[0].value)
            
        # Build strict Standard WCS (North-Up, East-Left)
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crpix = [nx_out / 2.0 + 0.5, ny_out / 2.0 + 0.5]
        target_wcs.wcs.cdelt = [-pix_scale_deg, pix_scale_deg] # East-Left(-), North-Up(+)
        target_wcs.wcs.crval = [target_ra, target_dec]
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # --- NEW: Generate and store the FITS-compliant header ---
        target_header = target_wcs.to_header()
        target_header['HISTORY'] = "Cutout strictly aligned to North-Up, East-Left."
        self.cut_wcs_header = target_header
        # ---------------------------------------------------------

        if cutsize <= 0:
            # Fallback if no cut size is defined
            imcut = jnp.array(self.data.data if hasattr(self.data, 'data') else self.data, dtype=jnp.float32)
            minx, maxx, miny, maxy = 0, shape[-1], 0, shape[-2]
        else:
            # ---------------------------------------------------------
            # 3. Efficient Bounding Box Extraction (The Speed Trick)
            # ---------------------------------------------------------
            # Extract a generously padded box from original array to feed to reproject
            # Padding by sqrt(2) guarantees corners won't clip during rotation
            pad_factor = 1.5 
            size_px = max(nx_out, ny_out) * pad_factor
            
            target_coord = SkyCoord(target_ra, target_dec, unit='deg')
            
            # Helper func for reprojection layer by layer
            def process_layer(img_2d):
                # Fast rough crop
                cutout = Cutout2D(img_2d, target_coord, size_px, wcs=wcs_2d, mode='partial', fill_value=np.nan)
                # Precision sub-pixel mapping
                reproj_arr, _ = reproject_interp((cutout.data, cutout.wcs), target_wcs, shape_out=(ny_out, nx_out))
                return reproj_arr

            def process_mask_layer(mask_2d):
                cutout = Cutout2D(mask_2d, target_coord, size_px, wcs=wcs_2d, mode='partial', fill_value=1)
                # order='nearest-neighbor' is CRITICAL for masks to avoid spline artifacting
                reproj_arr, _ = reproject_interp((cutout.data, cutout.wcs), target_wcs, shape_out=(ny_out, nx_out), order='nearest-neighbor')
                return reproj_arr > 0.5

            # ---------------------------------------------------------
            # 4. Apply transformations to Data, Mask, and Sigma
            # ---------------------------------------------------------
            data_arr = self.data.data if hasattr(self.data, 'data') else self.data
            
            if IFS:
                imcut_list = [process_layer(data_arr[z]) for z in range(shape[0])]
                imcut = jnp.array(np.stack(imcut_list), dtype=jnp.float32)
                
                if self.data.mask is not None:
                    self.cut_mask_image = np.stack([process_mask_layer(self.data.mask[z]) for z in range(shape[0])])
                if self.sigma_image is not None:
                    self.cut_sigma_image = np.stack([process_layer(self.sigma_image[z]) for z in range(shape[0])])
            else:
                imcut = jnp.array(process_layer(data_arr), dtype=jnp.float32)
                
                #if self.data.mask is not None:
                #    self.cut_mask_image = process_mask_layer(self.data.mask)
                #if self.sigma_image is not None:
                #    self.cut_sigma_image = process_layer(self.sigma_image)

            # Store Cut bounds logic in original frame (approximate, for legacy tracking)
            cutposition = [int(target_px - nx_out/2), int(target_px + nx_out/2), 
                        int(target_py - ny_out/2), int(target_py + ny_out/2)]

        self.cut_image = imcut
        
        # Update coordinates tracking
        self.sources_skycord_cut = [nx_out / 2.0 - 0.5, ny_out / 2.0 - 0.5]
        
        # Overwrite Coordinate transfer parameters to the strictly aligned WCS values
        pixsc_val = self.pixel_scales[0].value
        pixsc_arcmin = pixsc_val / 60.0
        self.coordinates_transfer_para = {
            'x0': self.sources_skycord_cut[0], 'x0shift': 0.0, 
            'y0': self.sources_skycord_cut[1], 'y0shift': 0.0, 
            'dxra': -pixsc_arcmin, 'dxdec': 0.0, 
            'dyra': 0.0,           'dydec': pixsc_arcmin, 
            'pixsc': pixsc_val, 
            'delta_ang': 90.0
        }

        # Extract Stats
        if sigma_clipped:
            sky_mean, sky_median, sky_std = sigma_clipped_stats(imcut, sigma=3.0, maxiters=5)
        else:
            sky_median = np.nanmedian(imcut)
        self.sky_median = sky_median
        return imcut, cutposition

    def write_imcut_align(self, ra=None, dec=None, outputname='cutout.fits', header=None, addpixsc=False, addmodel=False, Alamda=None):
        """
        Write cutout image and ancillary data to multi-extension FITS file.
        (Updated for strictly aligned standard WCS).
        """
        # 1. USE THE NEW ALIGNED HEADER BY DEFAULT
        if header is None:
            if hasattr(self, 'cut_wcs_header'):
                header = self.cut_wcs_header.copy()
            else:
                header = self.data.header.copy()

        # 2. CREATE PRIMARY HDU
        if addmodel:
            hdu_temp = fits.PrimaryHDU(np.float32(self.cut_image - self.model_image), header=header)
        else:
            hdu_temp = fits.PrimaryHDU(np.float32(self.cut_image), header=header)

        # 3. LEGACY WCS OVERRIDES (Only applied if not using the new clean header)
        if not hasattr(self, 'cut_wcs_header'): #or header is not self.cut_wcs_header:
            print("falling back to leagacy WCS")
            if hasattr(self, 'dcord'):
                hdu_temp.header['CRPIX1'] = self.dcord[0]
                hdu_temp.header['CRPIX2'] = self.dcord[1]
            if addpixsc:
                # Note: CDELT1 must be negative for standard East-Left convention
                hdu_temp.header['CDELT1'] = -self.pixel_scales[0].value / 3600.0 
                hdu_temp.header['CDELT2'] = self.pixel_scales[1].value / 3600.0

        # 4. EXTINCTION CORRECTION
        factor = 10**(0.4 * Alamda) if Alamda is not None else 1

        # 5. BUILD EXTENSIONS
        # Note: Applying factor to a boolean mask is unconventional, but preserved from original logic
        hdu_temp2 = fits.ImageHDU(np.float32(self.cut_mask_image) * factor, name='MASK')
        hdu_temp3 = fits.ImageHDU(np.float32(self.cut_sigma_image) * factor, name='SIGMA')
        
        if not addmodel:
            hdul = fits.HDUList([hdu_temp, hdu_temp2, hdu_temp3])
        else:
            hdu_temp4 = fits.ImageHDU(np.float32(self.model_image), name='MODEL')
            hdu_temp5 = fits.ImageHDU(np.float32(self.cut_image), name='DATA')
            hdul = fits.HDUList([hdu_temp, hdu_temp2, hdu_temp3, hdu_temp4, hdu_temp5])
            
        # 6. WRITE TO DISK
        hdul.writeto(outputname, overwrite=True)
        return
    def write_imcut(self, ra, dec, outputname, header = None, addpixsc = False, addmodel = False, Alamda = None):
        """
        Write cutout image and ancillary data to multi-extension FITS file.
        
        Saves the cutout data, mask, uncertainty, and optionally model to a FITS file
        with proper WCS keywords. Useful for archiving intermediate results or manual
        inspection of fitting regions.

        Parameters
        ----------
        ra : float, optional
            Right ascension of source (degrees). If None, uses stored `self.skycord[0]`.
        dec : float, optional
            Declination of source (degrees). If None, uses stored `self.skycord[1]`.
        outputname : str
            Output FITS file path.
        header : astropy.io.fits.Header, optional
            Base header for primary HDU. If None, uses original image header.
        addpixsc : bool, default: False
            If True, add CDELT1/CDELT2 keywords with pixel scale (degrees/pixel).
        addmodel : bool, default: False
            If True, write 5-extension FITS with residual (data-model) as primary HDU,
            plus mask, sigma, model, and original data extensions.

        FITS Structure
        --------------
        Without model (3 extensions):
            - HDU 0: Cut image data (PRIMARY, float32)
            - HDU 1: Cut mask (IMAGE, float32, bool converted to float)
            - HDU 2: Cut sigma/uncertainty (IMAGE, float32)
        
        With model (5 extensions):
            - HDU 0: Residual = data - model (PRIMARY, float32)
            - HDU 1: Cut mask (IMAGE, float32)
            - HDU 2: Cut sigma/uncertainty (IMAGE, float32)
            - HDU 3: Model image (IMAGE, float32)
            - HDU 4: Original cut image (IMAGE, float32)

        Notes
        -----
        The primary HDU header is updated with correct WCS reference pixel (CRPIX1/2)
        to reflect the cutout's position within the original image. This allows proper
        coordinate conversion even for cutouts.

        Examples
        --------
        Save cutout with default 3-extension format:
        
        >>> img.img_cut(ra=150.123, dec=2.456, cutsize=10.0)
        >>> img.write_imcut(ra=150.123, dec=2.456, outputname='cutout.fits')
        
        Save cutout with model and updated pixel scale:
        
        >>> img.model_image = create_model(...)  # after fitting
        >>> img.write_imcut(ra=150.123, dec=2.456, outputname='cutout_model.fits',
        ...                 addpixsc=True, addmodel=True)
        
        See Also
        --------
        img_cut : Extract cutout from full image
        read_imcut : Load previously saved cutout
        """
        if header is None:
            header=self.data.header
        if addmodel:
            hdu_temp = fits.PrimaryHDU(np.float32(self.cut_image-self.model_image), header=header)
        else:
            hdu_temp = fits.PrimaryHDU(np.float32(self.cut_image), header=header)
        if ra is None or dec is None:
            ra = self.skycord[0]
            dec = self.skycord[1]
        #hdu_temp.header['CRVAL1'] = ra
        #hdu_temp.header['CRVAL2'] = dec
        #hdu_temp.header['CRPIX1'] = self.sources_skycord_cut[0]
        #hdu_temp.header['CRPIX2'] = self.sources_skycord_cut[1]
        hdu_temp.header['CRPIX1'] = self.dcord[0]
        hdu_temp.header['CRPIX2'] = self.dcord[1]
        if addpixsc:
            hdu_temp.header['CDELT1'] = self.pixel_scales[0].value/3600
            hdu_temp.header['CDELT2'] = self.pixel_scales[1].value/3600
        if Alamda is not None:
            factor = 10**(0.4*Alamda)
        else:
            factor = 1
        hdu_temp2 = fits.ImageHDU(np.float32(self.cut_mask_image)*factor)
        hdu_temp3 = fits.ImageHDU(np.float32(self.cut_sigma_image)*factor)
        if not addmodel:
            hdul = fits.HDUList([hdu_temp,hdu_temp2,hdu_temp3])
        else:
            hdu_temp4 = fits.ImageHDU(np.float32(self.model_image))
            hdu_temp5 = fits.ImageHDU(np.float32(self.cut_image))
            hdul = fits.HDUList([hdu_temp,hdu_temp2,hdu_temp3,hdu_temp4,hdu_temp5])
        hdul.writeto(outputname, overwrite=True)
        return

    def read_imcut(self, read_hdu, transpar, cutimage = 0, sigmaimage = None, sigma_clipped=True):
        #cutsize is arcsec unit
        #delta_ang point to east direction
        #transpar = {'x0': srcXp-minx,'x0shift':0.0, 'y0': srcYp-miny,'y0shift':0.0, 'dxra':dxra, 'dxdec':dxdec,'dyra':dyra,'dydec':dydec,'pixsc':self.pixel_scales[0].value, 'delta_ang':delta_ang}
        self.sources_skycord = [transpar['x0'],transpar['y0']]
        self.cut_image = read_hdu[cutimage].data
        if sigma_clipped:
            sky_mean, sky_median, sky_std = sigma_clipped_stats(self.cut_image, sigma=3.0, maxiters=5)
        else:
            sky_median = np.nanmedian(self.cut_image)
            sky_std = np.nanstd(self.cut_image)
        self.sky_median = sky_median
        if sigmaimage is not None:
            self.cut_sigma_image = read_hdu[sigmaimage].data
        self.coordinates_transfer_para = transpar
        return self.cut_image

    def get_size(self, units='pixel'):
        '''
        Get the size of the image.

        Parameters
        ----------
        units : string
            Units of the size (pixel or angular units).

        Returns
        -------
        x, y : float
            Size along X and Y axes.
        '''
        nrow, ncol = self.data.shape
        if units == 'pixel':
            x = ncol
            y = nrow
        else:
            x = ncol * self.pixel_scales[0].to(units).value
            y = nrow * self.pixel_scales[1].to(units).value
        return (x, y)

    def get_data_info(self):
        '''
        Data information to generate model image.

        Returns
        -------
        d : dict
            shape : (ny, nx)
                Image array shape.
            pixel_scale : (pixelscale_x, pixelscale_y), default units: arcsec
                Pixel scales.
            wcs_rotation : angle, default units: radian
                WCS rotation, east of north.
        '''
        d = dict(shape=self.data.shape,
                 pixel_scale=self.pixel_scale,
                 wcs_rotation=self.wcs_rotation)
        return d

    def sigma_clipped_stats(self, **kwargs):
        '''
        Run astropy.stats.sigma_clipped_stats to get the basic statistics of
        the image.

        Parameters
        ----------
        All of the parameters go to astropy.stats.sigma_clipped_stats().

        Returns
        -------
        mean, median, stddev : float
            The mean, median, and standard deviation of the sigma-clipped data.
        '''
        return sigma_clipped_stats(self.data.data, mask=self.data.mask, **kwargs)

    def plot(self, stretch='asinh', units='arcsec', vmin=None, vmax=None, a=None, ax=None, plain=False, **kwargs):
        '''
        Plot an image.

        Parameters
        ----------
        stretch : string (default: 'asinh')
            Choice of stretch: asinh, linear, sqrt, log.
        units : string (default: 'arcsec')
            Units of pixel scale.
        vmin (optional) : float
            Minimal value of imshow.
        vmax (optional) : float
            Maximal value of imshow.
        a (optional) : float
            Scale factor of some stretch function.
        ax (optional) : matplotlib.Axis
            Axis to plot the image.
        plain : bool (default: False)
            If False, tune the image.
        **kwargs : Additional parameters goes into plt.imshow()

        Returns
        -------
        ax : matplotlib.Axis
            Axis to plot the image.
        '''
        assert self.data is not None, 'Set data first!'
        ax = plot_image(self.data, self.pixel_scales, stretch=stretch,
                        units=units, vmin=vmin, vmax=vmax, a=a, ax=ax,
                        plain=plain, **kwargs)
        if plain is False:
            ax.set_xlabel(r'$\Delta X$ ({0})'.format(units), fontsize=24)
            ax.set_ylabel(r'$\Delta Y$ ({0})'.format(units), fontsize=24)
        return ax

    def plot_direction(self, ax, xy=(0, 0), len_E=None, len_N=None, color='k', fontsize=20,linewidth=2, frac_len=0.1, units='arcsec', backextend=0.05):
        '''
        Plot the direction arrow. Only applied to plots using WCS.
        Parameters
        ----------
        ax : Axis
            Axis to plot the direction.
        xy : (x, y)
            Coordinate of the origin of the arrows.
        length : float
            Length of the arrows, units: pixel.
        units: string (default: arcsec)
            Units of xy.
        '''
        xlim = ax.get_xlim()
        len_total = np.abs(xlim[1] - xlim[0])
        pixelscale = self.pixel_scales[0].to('degree').value
        if len_E is None:
            len_E = len_total * frac_len / pixelscale
        if len_N is None:
            len_N = len_total * frac_len / pixelscale

        wcs = self.data.wcs
        header = wcs.to_header()
        d_ra = len_E * pixelscale
        d_dec = len_N * pixelscale
        ra = [header['CRVAL1'], header['CRVAL1']+d_ra, header['CRVAL1']]
        dec = [header['CRVAL2'], header['CRVAL2'], header['CRVAL2']+d_dec]
        ra_pix, dec_pix = wcs.all_world2pix(ra, dec, 1)
        d_arrow1 = [ra_pix[1]-ra_pix[0], dec_pix[1]-dec_pix[0]]
        d_arrow2 = [ra_pix[2]-ra_pix[0], dec_pix[2]-dec_pix[0]]
        l_arrow1 = np.sqrt(d_arrow1[0]**2 + d_arrow1[1]**2)
        l_arrow2 = np.sqrt(d_arrow2[0]**2 + d_arrow2[1]**2)
        d_arrow1 = np.array(d_arrow1) / l_arrow1 * len_E * pixelscale
        d_arrow2 = np.array(d_arrow2) / l_arrow2 * len_N * pixelscale

        def sign_2_align(sign):
            '''
            Determine the alignment of the text.
            '''
            if sign[0] < 0:
                ha = 'right'
            else:
                ha = 'left'
            if sign[1] < 0:
                va = 'top'
            else:
                va = 'bottom'
            return ha, va
        ha1, va1 = sign_2_align(np.sign(d_arrow1))
        ha2, va2 = sign_2_align(np.sign(d_arrow2))

        xy_e = (xy[0] - d_arrow1[0] * backextend, xy[1] - d_arrow1[1] * backextend)
        ax.annotate('E', xy=xy_e, xycoords='data', fontsize=fontsize,
                    xytext=(d_arrow1[0]+xy[0], d_arrow1[1]+xy[1]), color=color,
                    arrowprops=dict(color=color, arrowstyle="<-", lw=linewidth),
                    ha=ha1, va=va1)
        xy_n = (xy[0] - d_arrow2[0] * backextend, xy[1] - d_arrow2[1] * backextend)
        ax.annotate('N', xy=xy_n, xycoords='data', fontsize=fontsize,
                    xytext=(d_arrow2[0]+xy[0], d_arrow2[1]+xy[1]), color=color,
                    arrowprops=dict(color=color, arrowstyle="<-", lw=linewidth),
                    ha=ha2, va=va2)

    def set_data(self, data, unit):
        '''
        Parameters
        ----------
        data : 2D array
            Image data.
        unit : string
            Unit for CCDData.
        '''
        self.data = CCDData(data, unit=unit)

    def generate_cutmask(self, psfFWHM, data = None, sdmask = None, nsigma=3.0, magnification=3., npixels = 5, nlevels=32, contrast=0.001, deblend=False, sky_level = None, source_dia = 0.5 ,max_grow_size=100, growmethod = True, nearbygrow = 0, addgrow=0, left_source = True):
        """
        Generate pixel mask for contaminating sources in cutout region.
        
        Detects and masks background/foreground sources to isolate the target object
        for fitting. Uses photutils segmentation to identify sources, then optionally
        grows segmentation regions adaptively based on local flux levels.

        Parameters
        ----------
        psfFWHM : float
            Point spread function FWHM (arcsec). Used to set detection kernel size and
            minimum source separation.
        data : ndarray, optional
            Image data to analyze. If None, uses `self.cut_image`.
        sdmask : ndarray of bool, optional
            Pre-existing mask (e.g., bad pixels). Pixels marked True are excluded from
            source detection.
        nsigma : float, default: 3.0
            Detection threshold in units of background RMS. Sources must have peak 
            flux > nsigma * sky_std.
        magnification : float, default: 3.0
            Factor by which to expand detected source regions when `growmethod=False`.
            Mask semi-major axis becomes `magnification * semimajor_sigma`.
        npixels : int, default: 5
            Minimum connected pixels above threshold to constitute a source.
        nlevels : int, default: 32
            Number of multi-thresholding levels for deblending (if `deblend=True`).
        contrast : float, default: 0.001
            Deblending contrast parameter. Smaller values → more aggressive separation.
        deblend : bool, default: False
            If True, attempt to separate blended sources using multi-level segmentation.
        sky_level : float, optional
            Flux threshold for adaptive growth termination. If None, set to 3*sky_std.
            Growth stops when median flux of new pixels < `sky_level`.
        source_dia : float, default: 0.5
            Protected region diameter (arcsec) around target source position. Sources
            within this distance are treated specially (not masked if `left_source=True`).
        max_grow_size : int, default: 100
            Maximum number of dilation iterations for adaptive growth.
        growmethod : bool, default: True
            If True, use adaptive growth algorithm (`grow_segmentation`). If False,
            simply expand all sources by `magnification` factor using their ellipse
            parameters.
        nearbygrow : int, default: 0
            Number of dilation iterations applied to sources near target (within
            `source_dia`). Set > 0 to also mask nearby companions.
        addgrow : int, default: 0
            Additional dilation iterations applied to all contaminating sources after
            adaptive growth converges.
        left_source : bool, default: True
            If True, protect the central source at `self.sources_skycord_cut` from
            being masked. If False, mask all detected sources.

        Returns
        -------
        mask : ndarray of bool
            Boolean mask with shape matching `data`. True indicates pixels to exclude
            from fitting (contaminating sources, bad pixels, NaN/Inf).

        Notes
        -----
        **Adaptive Growth Algorithm** (when `growmethod=True`):
        
        1. Detect sources using `photutils.detect_sources` with Gaussian kernel
        2. Identify target source label at `self.sources_skycord_cut`
        3. For each contaminating source:
           - Iteratively dilate segmentation by 1 pixel
           - Stop when median of new pixels < `sky_level` OR iteration > `max_grow_size`
           - Apply additional `addgrow` iterations
        4. Combine all masks, excluding target source region
        
        **Elliptical Masking** (when `growmethod=False`):
        
        - Measure ellipse parameters (a, b, PA) for each source
        - Mask ellipse with semi-major axis = `magnification * a`
        - Preserve target source by checking overlap with protected circle
        
        The adaptive method is recommended for crowded fields as it follows source
        isophotes more accurately than fixed ellipses.

        Examples
        --------
        Basic contamination masking for isolated galaxy:
        
        >>> img.img_cut(ra=150.123, dec=2.456, cutsize=20.0)
        >>> mask = img.generate_cutmask(psfFWHM=0.15, nsigma=2.0)
        >>> print(f"Masked {np.sum(mask)/mask.size*100:.1f}% of pixels")
        
        Crowded field with aggressive masking:
        
        >>> mask = img.generate_cutmask(psfFWHM=0.12, nsigma=2.5, magnification=5.0,
        ...                             max_grow_size=50, addgrow=5)
        
        Mask all sources including target (for background estimation):
        
        >>> mask_all = img.generate_cutmask(psfFWHM=0.15, left_source=False)
        
        See Also
        --------
        grow_segmentation : Adaptive segmentation growth algorithm
        sky_subtraction : Background subtraction using source masking
        """
        if data is None:
            data = np.array(self.cut_image.copy())
        if sky_level is None:
            sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
            sky_level = 3*sky_std
        if sdmask is None:
            sdmask = np.zeros_like(data,dtype = bool)
        psfFWHMpix = psfFWHM / self.pixel_scales[0].value
        thresholder = detect_threshold(data, nsigma=nsigma)
        sigma = psfFWHMpix * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
        kernel.normalize()
        segm = detect_sources(data, thresholder, npixels=npixels, mask = sdmask)
        if deblend is True:
            segm = deblend_sources(data, segm, npixels=npixels, nlevels=nlevels, contrast=contrast,progress_bar = False)
        if left_source:
            if growmethod is True:
                sourcelabel = segm.data[int(self.sources_skycord_cut[1]),int(self.sources_skycord_cut[0])]
                if sourcelabel == 0:
                    print ("No source detected at the position of the source")
                    sourcelabel = None
                segmnew, mask = grow_segmentation(segm, data, sourcelabel, source_dia = source_dia ,max_grow_size=max_grow_size, sky_level=sky_level, nearbygrow = nearbygrow, addgrow=addgrow)
            else:
                props = SourceCatalog(data, segm)
                intsourceposi = [int(self.sources_skycord_cut[0]),int(self.sources_skycord_cut[1])]
                mask=np.zeros_like(data, dtype=bool)
                mask[np.isnan(data)] = True
                mask[np.isinf(data)] = True
                sources = Table(props.to_table(['xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation','segment_flux','fwhm','maxval_xindex','maxval_yindex']))
                for loop in range(len(sources)):
                    position = (sources['xcentroid'][loop],sources['ycentroid'][loop])
                    a = sources['semimajor_sigma'][loop]
                    b = sources['semiminor_sigma'][loop]
                    theta = sources['orientation'][loop]*180./np.pi
                    masktmp=Maskellipse(mask,position,magnification*a,(1-b/a),theta)
                    # Y.D.: This criteria is to make sure the source will not be masked out. 
                    xmesh, ymesh = jnp.meshgrid(jnp.arange(masktmp.shape[1]), jnp.arange(masktmp.shape[0]))
                    source_mask = elliptical_func(xmesh+0.5,intsourceposi,source_dia/2,0,0,ymesh+0.5)<0
                    if not any(masktmp[source_mask]):
                        # execute if masktmp is all False in the source region
                        # Previous version was too loose such that for KCWI it didn't work. Must be rewritten.
                        mask = masktmp
        else:
            if growmethod is True:
                segmnew, mask = grow_segmentation(segm, data, None, source_dia = source_dia ,max_grow_size=max_grow_size, sky_level=sky_level, nearbygrow = nearbygrow, addgrow=addgrow)
            else:
                props = SourceCatalog(data, segm)
                mask=np.zeros_like(data, dtype=bool)
                mask[np.isnan(data)] = True
                mask[np.isinf(data)] = True
                sources = Table(props.to_table(['xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation','segment_flux','fwhm','maxval_xindex','maxval_yindex']))
                for loop in range(len(sources)):
                    position = (sources['xcentroid'][loop],sources['ycentroid'][loop])
                    a = sources['semimajor_sigma'][loop]
                    b = sources['semiminor_sigma'][loop]
                    theta = sources['orientation'][loop]*180./np.pi
                    mask=Maskellipse(mask,position,magnification*a,(1-b/a),theta)       
        return mask    

    def source_detection_individual(self, psfFWHM, nsigma=3.0, sc_key=''):
        '''
        Parameters
        ----------
        psfFWHM : float
            FWHM of the imaging point spread function
        nsigma : float
            source detection threshold
        '''
        data = np.array(self.data.copy())
        psfFWHMpix = psfFWHM / self.pixel_scales[0].value
        thresholder = detect_threshold(data, nsigma=nsigma)
        sigma = psfFWHMpix * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
        kernel.normalize()
        segm = detect_sources(data, thresholder, npixels=5)
        props = SourceCatalog(data, segm)
        tab = Table(props.to_table(['xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation','segment_flux','fwhm','maxval_xindex','maxval_yindex']))
        self.sources_catalog = tab
        srcPstradec = self.data.wcs.all_pix2world(tab['xcentroid'], tab['ycentroid'],1)
        sc = SkyCoord(srcPstradec[0], srcPstradec[1], unit='deg')
        sctab = Table([sc,np.arange(len(sc))],names=['sc','sloop_{0}'.format(sc_key)])
        self.sources_skycord = sctab

    def source_detection_residual(self, psfFWHM, nsigma=3.0, sc_key=''):
        '''
        Copied from `source_detection_individual` but detects both positive and negative sources
        '''
        # positive sources
        data = np.array(self.data.copy())
        psfFWHMpix = psfFWHM / self.pixel_scales[0].value
        thresholder = detect_threshold(data, nsigma=nsigma)
        sigma = psfFWHMpix * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
        kernel.normalize()
        segm = detect_sources(data, thresholder, npixels=5)
        if segm is not None: 
            props = SourceCatalog(data, segm)
            tab_posi = Table(props.to_table(['xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation','segment_flux','fwhm','maxval_xindex','maxval_yindex']))
            srcPstradec = self.data.wcs.all_pix2world(tab_posi['xcentroid'], tab_posi['ycentroid'],1)
            sc = SkyCoord(srcPstradec[0], srcPstradec[1], unit='deg')
            sctab_posi = Table([sc,np.arange(len(sc))],names=['sc','sloop_{0}'.format(sc_key)])
        else: 
            tab_posi = Table(names=['xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation','segment_flux','fwhm','maxval_xindex','maxval_yindex'])
            sctab_posi = Table(names=['sc','sloop_{0}'.format(sc_key)])
        # negative sources
        data = -np.array(self.data.copy())
        psfFWHMpix = psfFWHM / self.pixel_scales[0].value
        thresholder = detect_threshold(data, nsigma=nsigma)
        sigma = psfFWHMpix * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
        kernel.normalize()
        segm = detect_sources(data, thresholder, npixels=5)
        if segm is not None: 
            props = SourceCatalog(data, segm)
            tab_nega = Table(props.to_table(['xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation','segment_flux','fwhm','maxval_xindex','maxval_yindex']))
            srcPstradec = self.data.wcs.all_pix2world(tab_nega['xcentroid'], tab_nega['ycentroid'],1)
            sc = SkyCoord(srcPstradec[0], srcPstradec[1], unit='deg')
            sctab_nega = Table([sc,np.arange(len(sc))],names=['sc','sloop_{0}'.format(sc_key)])
        else: 
            tab_nega = Table(names=['xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation','segment_flux','fwhm','maxval_xindex','maxval_yindex'])
            sctab_nega = Table(names=['sc','sloop_{0}'.format(sc_key)])
        # combine them together
        tab_nega['segment_flux'] = -tab_nega['segment_flux']
        tab_stacked = vstack([tab_posi, tab_nega])
        self.sources_catalog = tab_stacked
        sctab_stacked = vstack([sctab_posi, sctab_nega])
        sctab_stacked['sloop_'] = np.arange(len(sctab_stacked))
        self.sources_skycord = sctab_stacked

    def set_mask(self, mask):
        '''
        Set mask for the extension.

        Parameters
        ----------
        mask : 2D array
            The mask.
        '''
        assert self.data.shape == mask.shape, 'Mask shape incorrect!'
        self.data.mask = mask
        if self.ss_data is not None:
            self.ss_data.mask = mask

    def set_pixel_scales(self, pixel_scales):
        '''
        Parameters
        ----------
        pixel_scales (optional) : tuple
            Pixel scales along the first and second directions, units: arcsec.
        '''
        self.pixel_scales = (pixel_scales[0]*u.arcsec, pixel_scales[1]*u.arcsec)

    def set_zero_point(self, zp):
        '''
        Set magnitude zero point.
        '''
        self.zero_point = zp

    def sky_subtraction(self, sdmask = None, polyfit = True, order=3 , filepath = None, psfFWHM = 1.5, nsigma=3.0, npixels = 5 ,max_grow_size=100, addgrow=0, usemedian = False):
        """
        Estimate and subtract sky background from image.
        
        Masks sources using segmentation, then fits 2D polynomial or computes median/2D
        background model from unmasked pixels. Useful for removing spatially varying
        sky before galaxy modeling.

        Parameters
        ----------
        sdmask : ndarray of bool, optional
            Pre-existing mask (e.g., bad pixels). Combined with detected source mask.
        polyfit : bool, default: True
            If True, fit 2D polynomial to unmasked sky pixels. If False, use photutils
            `Background2D` with median filter boxes (unless `usemedian=True`).
        order : int, default: 3
            Polynomial order for `polyfit=True`. Order 1 = plane, 2 = paraboloid, etc.
            Higher orders can capture complex gradients but may overfit.
        filepath : str, optional
            If provided, save sky-subtracted image to this FITS file.
        psfFWHM : float, default: 1.5
            PSF FWHM (arcsec) for source detection kernel.
        nsigma : float, default: 3.0
            Detection threshold (sigma) for masking sources.
        npixels : int, default: 5
            Minimum connected pixels to constitute a source.
        max_grow_size : int, default: 100
            Maximum dilation iterations for growing source masks.
        addgrow : int, default: 0
            Additional dilation iterations after adaptive growth.
        usemedian : bool, default: False
            If True, subtract simple median of unmasked pixels (fastest, least accurate).

        Returns
        -------
        maskplus : ndarray of bool
            Combined mask of sources, bad pixels, and boundary artifacts.

        Notes
        -----
        **Background Subtraction Methods**:
        
        1. **Median** (`usemedian=True`):
           - Background = median(unmasked pixels)
           - Fastest, suitable for flat fields
        
        2. **2D Polynomial** (`polyfit=True`):
           - Fits polynomial surface to unmasked pixels via least squares
           - Good for gradients from scattered light, moon, etc.
           - Equation for order=2: :math:`B(x,y) = c_0 + c_1 x + c_2 y + c_3 x^2 + c_4 xy + c_5 y^2`
        
        3. **2D Median Filter** (`polyfit=False, usemedian=False`):
           - Uses photutils `Background2D` with 30x30 pixel boxes
           - Adapts to local sky variations
           - Slower but handles complex backgrounds
        
        **Masking Strategy**:
        
        - Detects sources with `detect_sources` and `detect_threshold`
        - Grows segmentation masks to exclude faint wings and artifacts
        - Additionally masks pixels with values exactly equal to the 4 image corners
          (often indicates bad data or padding)
        
        The sky-subtracted data overwrites `self.data.data`. Original data is lost
        unless saved beforehand.

        Examples
        --------
        Polynomial sky subtraction (most common):
        
        >>> img = image('science.fits')
        >>> mask = img.sky_subtraction(psfFWHM=0.15, polyfit=True, order=2)
        >>> print(f"Sky median after subtraction: {np.median(img.data.data):.6f}")
        
        2D median filter for variable background:
        
        >>> mask = img.sky_subtraction(psfFWHM=0.15, polyfit=False, usemedian=False)
        
        Simple median subtraction for quick processing:
        
        >>> mask = img.sky_subtraction(psfFWHM=0.15, usemedian=True)
        
        Save sky-subtracted image:
        
        >>> mask = img.sky_subtraction(psfFWHM=0.15, order=3, 
        ...                            filepath='science_skysub.fits')
        
        See Also
        --------
        generate_cutmask : Source masking algorithm
        polynomialfit : 2D polynomial fitting function
        photutils.background.Background2D : Photutils 2D background estimator
        """
        data = np.array(self.data.data)
        maskplus = self.generate_cutmask(psfFWHM = psfFWHM, data = data, sdmask = sdmask, nsigma=nsigma, npixels = npixels, deblend=False ,max_grow_size=max_grow_size, addgrow=addgrow, left_source = False)
        # mask pixels have exact values as the 4 boundary points
        maskplus = np.where(data == data[0,0], True, maskplus)
        maskplus = np.where(data == data[0,-1], True, maskplus)
        maskplus = np.where(data == data[-1,0], True, maskplus)
        maskplus = np.where(data == data[-1,-1], True, maskplus)
        print ("finish mask")
        if usemedian:
            background = np.nanmedian(data[~maskplus])
        else:
            if polyfit:
                backR=polynomialfit(data,maskplus,order=order)
                background=backR['bkg']
            else:
                backR=Background2D(data,30,mask=maskplus)
                background = backR.background
        self.data.data = data-background
        if filepath is not None:
            hdu_temp = fits.PrimaryHDU(data-background, header = self.data.header)
            hdu_temp.writeto(filepath, overwrite=True)
        return maskplus

    def read_ss_image(self,filepath):
        '''
        read sky subtracted image from "filepath"
        '''
        hdu = fits.open(filepath)
        self.ss_data = CCDData(hdu[0].data, unit=self.data.unit)
        self.ss_data.mask = self.data.mask.copy()

    def cal_sigma_image(self, expt, data = None, gain = 1., fac1 = 1., fac2 = 1.,filepath=None):
        """
        Construct uncertainty map from photon statistics and sky background.
        
        Calculates per-pixel uncertainties following GALFIT's error model: quadrature
        sum of Poisson noise from source photons and sky background variance. Assumes
        input data is in count rate (e-/s) units.

        Parameters
        ----------
        expt : float
            Exposure time (seconds). Converts count rate to total counts.
        data : ndarray, optional
            Image data in count rate units (e-/s). If None, uses `self.data.data`.
        gain : float, default: 1.0
            Detector gain (e-/ADU). For pre-calibrated data in electrons, set gain=1.
        fac1 : float, default: 1.0
            Multiplicative factor for Poisson noise term. Can account for sub-pixel
            resampling effects (set > 1) or correlation (set < 1).
        fac2 : float, default: 1.0
            Multiplicative factor for sky background variance. Adjust for non-ideal
            background estimation.
        filepath : str, optional
            If provided, save sigma map as FITS file to this path.

        Returns
        -------
        sigmap : ndarray
            Uncertainty map in count rate units (e-/s). Same shape as input data.

        Notes
        -----
        **Uncertainty Model**:
        
        .. math::
        
            \\sigma_{{\\rm total}} = \\sqrt{{f_1 N_{{\\rm source}} + f_2 \\sigma_{{\\rm sky}}^2}}
        
        where:
        
        - :math:`N_{{\\rm source}} = |I| \\times G \\times t` (total electrons from source)
        - :math:`\\sigma_{{\\rm sky}}` = sky RMS in electrons (from sigma-clipped stats)
        - :math:`f_1, f_2` = empirical correction factors (`fac1`, `fac2`)
        - :math:`G` = gain, :math:`t` = exposure time
        
        Final uncertainty is converted back to count rate by dividing by :math:`G \\times t`.
        
        The absolute value of intensity handles both positive flux and negative
        residuals during iterative fitting.
        
        **Typical Values**:
        
        - HST/ACS: gain ~1-2 e-/ADU, already in e-/s → set gain=1
        - Ground-based CCD: gain ~2-5 e-/ADU, units in ADU/s → provide gain
        - Drizzled images: set fac1 ~1.5 to account for pixel correlation
        
        This method sets `self.sigma_image` for later use by fitting routines.

        Examples
        --------
        Calculate uncertainties for HST/WFC3 (gain=1, units already e-/s):
        
        >>> img = image('hst_wfc3.fits')
        >>> sigma = img.cal_sigma_image(expt=1000.0, gain=1.0)
        >>> print(f"Median S/N: {np.median(img.data.data/sigma):.1f}")
        
        Ground-based image with gain=2.5 e-/ADU:
        
        >>> sigma = img.cal_sigma_image(expt=600.0, gain=2.5, fac1=1.2)
        
        Save uncertainty map for inspection:
        
        >>> sigma = img.cal_sigma_image(expt=1000.0, filepath='sigma.fits')
        
        See Also
        --------
        cal_sigma_image_of_cutout : Calculate sigma for cutout region only
        read_sigmap : Load externally computed sigma map
        """
        # Uncertainty due to photon noise
        if data is None:
            data = np.array(self.data.data)
        image_data_electrons = np.abs(data) * gain * expt  # convert to electrons
        sky_mean, sky_median, sky_std = sigma_clipped_stats(data*gain*expt, sigma=3.0, maxiters=5)
        #print (sky_std/(gain * expt))
        total_noise = np.sqrt(fac1 * image_data_electrons + fac2 * sky_std **2)
        sigmap = total_noise / (gain * expt)
        self.sigma_image = sigmap
        if filepath is not None:
            hdu_temp = fits.PrimaryHDU(sigmap)
            hdu_temp.writeto(filepath, overwrite=True)
        return sigmap
    
    def cal_sigma_image_of_cutout(self, expt, gain = 1., fac1 = 1., fac2 = 1.,filepath=None):
        '''
    	Construct sigma map following the same procedure as Galfit (quadruture sum of sigma at each pixel from source and sky background).
    	Note
        ----------
    	'GAIN' keyword must be available in the image header and ADU x GAIN = electron
        assume a counts rate unit of image
    	Parameters
        ----------
    		filepath:
    			Whether and where to save sigma map
        '''
        # Uncertainty due to photon noise

        image_data_electrons = np.abs(self.cut_image) * gain * expt  # convert to electrons
        sky_mean, sky_median, sky_std = sigma_clipped_stats(self.cut_image*gain *expt, sigma=3.0, maxiters=5)
        total_noise = np.sqrt(fac1 * image_data_electrons + fac2 * sky_std **2)
        sigmap = total_noise / (gain * expt)
        self.cut_sigma_image = sigmap
        if filepath is not None:
            hdu_temp = fits.PrimaryHDU(sigmap)
            hdu_temp.writeto(filepath, overwrite=True)
        return sigmap
        
    def read_sigmap(self, filepath):
        '''
        read sigma image from "filepath"
        '''
        hdu = fits.open(filepath)
        self.sigma_image = hdu[0].data

    def read_PSF(self, filepath):
        '''
        read PSF image from "filepath"
        '''
        hdu = fits.open(filepath)
        self.PSF = hdu[0].data

    def set_psf_fwhm(self, fwhm=None):
        if fwhm is None:
            self.psfFWHM_arcsec=self.psfFWHM_map[int(self.sources_skycord[1]),int(self.sources_skycord[0])]
        else:
            self.psfFWHM_arcsec = fwhm
        return

    def gaussian_psf(self, size=15):
        """
        Generate analytic Gaussian PSF kernel.
        
        Creates a 2D Gaussian PSF normalized to unit sum, using the image's FWHM
        value (`self.psfFWHM_arcsec`). Useful as fallback when empirical PSF cannot
        be built from stars.

        Parameters
        ----------
        size : int, default: 15
            PSF kernel size (pixels). Should be odd for symmetric centering. Typically
            5-7 times FWHM in pixels is sufficient for accurate convolution.

        Returns
        -------
        psf : ndarray
            2D Gaussian kernel with shape (size, size), normalized to sum=1.

        Notes
        -----
        The Gaussian width is set by:
        
        .. math::
        
            \\sigma = {{\\rm FWHM}} / (2\\sqrt{{2\\ln 2}}) \\approx {{\\rm FWHM}} / 2.355
        
        where FWHM is converted from arcsec to pixels using `self.pixel_scales[0]`.
        
        The kernel is generated via astropy's `Gaussian2DKernel` with default centering
        and normalization.
        
        This method sets `self.PSF` for use by convolution routines in galaxy modeling.

        Examples
        --------
        Create Gaussian PSF for HST/WFC3 (FWHM ~ 0.15 arcsec):
        
        >>> img.psfFWHM_arcsec = 0.15
        >>> psf = img.gaussian_psf(size=21)
        >>> print(psf.shape, psf.sum())
        (21, 21) 1.0
        
        Use larger kernel for ground-based seeing (FWHM ~ 0.8 arcsec):
        
        >>> img.psfFWHM_arcsec = 0.8
        >>> psf = img.gaussian_psf(size=51)  # ~6 * FWHM/pixscale
        
        See Also
        --------
        generate_PSFs : Build empirical PSF from stellar sources
        read_PSF : Load pre-computed PSF from file
        set_psf_fwhm : Set FWHM value for PSF generation
        """
        psfFWHM = self.psfFWHM_arcsec/self.pixel_scales[0].value
        kernel = Gaussian2DKernel(psfFWHM/C.tl2, x_size=size, y_size=size)
        kernel.normalize()
        self.PSF = np.array(kernel)
        return self.PSF
    
    

    
    def generate_PSFs(self, psfFWHM, data = None,mheader=None, sigmaimg = None, nsigma=2.0, npixels=50, a_TV = 1e6, a_sky=1., learning_rate=1e-4, fluxthreshould = 2, isolate = 0.1, qt = 0.8, medianfwhm = None, sigfwhm = 2, starcord_list = None, central_tol = 0.0, figurepath = None,
                    size = 31, oversampling=1, plot=True,crossmatch_tolerance=None, filepaths=None, droppsf=[],use_sep=False, notfit=False, crossmatch = False, star_savepath = None, makestackpsf = False, photutils_method = False,num_steps = 14000,verbose = False, referimg = None, referthre = 10.):
        """
        Build empirical point spread function from stellar sources in image.
        
        Detects point sources, extracts cutouts, and constructs high-fidelity PSF model
        using either photutils EPSFBuilder or custom JAX-based optimizer with TV
        regularization. The PSF can be oversampled for sub-pixel accuracy.

        Parameters
        ----------
        psfFWHM : float
            Expected PSF FWHM (arcsec). Used for source detection kernel and star
            selection criteria.
        data : ndarray, optional
            Image data for source detection. If None, uses `self.data.data`.
        sigmaimg : ndarray, optional
            Uncertainty map. If provided, used for weighted PSF fitting.
        nsigma : float, default: 2.0
            Detection threshold (sigma above background) for candidate stars.
        npixels : int, default: 50
            Minimum connected pixels above threshold to constitute a source.
        a_TV : float, default: 1e6
            Total variation regularization strength (JAX method only). Higher values
            enforce smoother PSF. Typical range: 1e4 - 1e7.
        a_sky : float, default: 1.0
            Sky background penalty weight (JAX method only). Prevents PSF wings from
            becoming negative.
        learning_rate : float, default: 1e-4
            Adam optimizer learning rate (JAX method only).
        fluxthreshould : float, default: 2
            Flux selection: keep stars with flux > `fluxthreshould * mean(all_fluxes)`.
        isolate : float, default: 0.1
            Isolation criterion: reject stars if > `isolate` fraction of cutout is
            masked by neighbors or boundaries.
        qt : float, default: 0.8
            Axis ratio threshold: keep stars with q = b/a > `qt` (rejects elongated
            sources likely to be galaxies or blends).
        medianfwhm : float, optional
            Override median FWHM for star selection. If None, computed from detections.
        sigfwhm : float, default: 2
            FWHM selection: keep stars within `sigfwhm * std` of median FWHM, or
            set None to disable.
        starcord_list : list of [RA, Dec], optional
            Manually specified star coordinates (degrees). If provided, uses these
            instead of automatic detection.
        central_tol : float, default: 0.0
            Fraction of central region (0.25*size radius) allowed to be masked. Rejects
            stars with bad pixels near peak.
        figurepath : str, optional
            Directory path for saving diagnostic plots. If None and `plot=True`, shows
            interactively.
        size : int, default: 31
            Cutout size (pixels) for star extraction. Should encompass PSF wings.
        oversampling : int, default: 1
            PSF oversampling factor. E.g., oversampling=2 creates PSF with 2x finer
            pixel grid. Useful for dithered observations.
        plot : bool, default: True
            If True, display cutouts of all stars and final PSF.
        filepaths : str, optional
            FITS file path to save final PSF.
        droppsf : list of int, optional
            Indices of stars to exclude from PSF building (after initial filtering).
        notfit : bool, default: False
            If True, return None after showing star cutouts (diagnostic mode).
        crossmatch : bool, default: False
            If True, cross-match detections with Gaia DR3 to confirm stars. Requires
            astroquery.
        star_savepath : str, optional
            Directory path to save individual star cutouts as FITS files (for debugging).
        makestackpsf : bool, default: False
            If True, return simple median-stacked PSF instead of fitting.
        photutils_method : bool, default: False
            If True, use photutils EPSFBuilder. If False, use JAX-based fitter.
        num_steps : int, default: 14000
            Number of optimization iterations (JAX method only).
        verbose : bool, default: False
            If True, print loss every 100 iterations (JAX method).
        referimg : ndarray, optional
            Reference PSF for quality check. Stars with chi-square > `referthre` when
            fit with `referimg` are rejected.
        referthre : float, default: 10.0
            Reduced chi-square threshold for reference PSF filtering.

        Returns
        -------
        epsf : ndarray
            Empirical PSF with shape `(size*oversampling, size*oversampling)`. 
            Normalized to unit sum.

        Notes
        -----
        **Star Selection Pipeline**:
        
        1. **Detection**: Use photutils `detect_sources` with Gaussian kernel
        2. **Gaia cross-match** (optional): Confirm point sources
        3. **Flux cut**: Keep bright sources (> `fluxthreshould * mean_flux`)
        4. **Morphology cuts**: 
           - Axis ratio q > `qt` (reject elongated sources)
           - FWHM within `sigfwhm` sigma of median (reject outliers)
        5. **Isolation check**: Fraction masked < `isolate`
        6. **Central quality**: Central region masking < `central_tol`
        7. **Reference fit** (optional): Chi-square with `referimg` < `referthre`
        
        **PSF Building Methods**:
        
        - **Photutils** (`photutils_method=True`): Anderson & King (2000) ePSF 
          algorithm with 150 iterations
        - **JAX** (`photutils_method=False`): Simultaneous fit of PSF + positions
          + normalizations using Adam optimizer with total variation regularization::
        
              Loss = 0.5 * sum((data - model)^2 / sigma^2) 
                     + a_TV * TV(PSF) 
                     + a_sky * sum(PSF[outside_circle]^2)
        
        The JAX method typically produces sharper PSFs by avoiding smoothing inherent
        in interpolation, but requires careful tuning of `a_TV` and `a_sky`.
        
        This method sets `self.PSF` for use in galaxy model convolution.

        Examples
        --------
        Build PSF from automatic star detection:
        
        >>> img = image('science.fits')
        >>> img.cal_sigma_image(expt=1000.0, gain=1.0)
        >>> psf = img.generate_PSFs(psfFWHM=0.15, sigmaimg=img.sigma_image,
        ...                         size=31, oversampling=2)
        >>> print(psf.shape)
        (62, 62)
        
        Use manual star list (e.g., from catalog):
        
        >>> star_coords = [[150.123, 2.456], [150.145, 2.478], ...]
        >>> psf = img.generate_PSFs(psfFWHM=0.15, starcord_list=star_coords,
        ...                         crossmatch=False)
        
        Build PSF with stricter selection and save diagnostics:
        
        >>> psf = img.generate_PSFs(psfFWHM=0.12, fluxthreshould=3, qt=0.9, 
        ...                         sigfwhm=1.5, filepaths='psf.fits',
        ...                         figurepath='./psf_diagnostics/')
        
        Quick stacked PSF (no fitting):
        
        >>> psf = img.generate_PSFs(psfFWHM=0.15, makestackpsf=True)
        
        See Also
        --------
        gaussian_psf : Create analytic Gaussian PSF
        fittingPSF : Low-level JAX PSF fitter
        photutils.psf.EPSFBuilder : Photutils ePSF construction
        """
        if data is None:
            data = np.array(self.data.data)
        if mheader is None:
            wcs = self.data.wcs
        else:
            wcs = WCS(mheader)

        sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
        data -= sky_median
        mask = np.zeros_like(data, dtype=bool)
        mask[np.isnan(data)] = True
        mask[np.isinf(data)] = True
        psfFWHMpix = psfFWHM / self.pixel_scales[0].value
        sigma = psfFWHMpix * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
        kernel.normalize()

        if use_sep:
            import sep
            objects,segm = sep.extract(to_native_endian(data), nsigma, err=to_native_endian(sigmaimg),segmentation_map=True\
                                  ,mask=mask,minarea=npixels)
            stars = sep_to_photutils_like_table(objects)
        else:
            thresholder = detect_threshold(data, nsigma=nsigma,error=sigmaimg,mask=mask)
            segm = detect_sources(data, thresholder, npixels=npixels,mask=mask)
            # deblend sources
            segm = deblend_sources(data, segm, npixels=npixels, nlevels=8, contrast=0.001,progress_bar = False)
            segm_data = segm.data
            props = SourceCatalog(data, segm)
            stars = Table(props.to_table(['xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation','segment_flux','fwhm','maxval_xindex','maxval_yindex']))
        
        ra, dec = wcs.all_pix2world(stars['xcentroid'], stars['ycentroid'], 0)
        Column_ra = Column(name='RA', data=ra)
        Column_dec = Column(name='DEC', data=dec)
        stars.add_columns([Column_ra, Column_dec])

        if crossmatch:
            if crossmatch_tolerance is None:
                max_distance = 1.5*psfFWHM*u.arcsec
            else:
                max_distance = crossmatch_tolerance*u.arcsec
            t_o = XMatch.query(cat1=stars, cat2='vizier:I/355/gaiadr3', max_distance=max_distance, 
                            colRA1='RA', colDec1='DEC')  # Gaia xmatch.
        else:
            t_o = stars
        if len(t_o) == 0:
            print ("original table have length {0}, become 0 after Gaia cross match".format(len(stars)))
            t_o = stars
        if starcord_list is not None:
            rows = t_o[0]
            for i in range(len(starcord_list)):
                star_skyc = wcs.all_world2pix(starcord_list[i][0],starcord_list[i][1],0)
                rows['maxval_xindex'] = star_skyc[0]
                rows['maxval_yindex'] =  star_skyc[1]
                rows['fwhm'] = np.nan
                t_o.add_row(rows)
            t_o.remove_rows(np.arange(len(t_o)-len(starcord_list)))
        else:
            # filter source > fluxthreshould%
            print ('total sources: ',len(t_o))
            t_o = t_o[t_o['segment_flux'] > fluxthreshould*np.mean(t_o['segment_flux'])]
            print ('sources above thereshold flux :',len(t_o))
            q = t_o['semiminor_sigma']/t_o['semimajor_sigma']
            t_o = t_o[q>qt]
            print ('sources above thereshold axis ratio :',len(t_o))
            if sigfwhm is not None:
                t_o = t_o[t_o['fwhm'] < 7.5*psfFWHMpix]
                mean, median, std = sigma_clipped_stats(np.array(t_o['fwhm']), sigma=3.0, maxiters=5)
                if medianfwhm is not None:
                    median = medianfwhm
                print ('PSF FWHM: {0:.2f}, median: {1:.2f}, std: {2:.2f}'.format(psfFWHMpix,median,std))
                masker = (t_o['fwhm'] < median+sigfwhm*std) & (t_o['fwhm'] > median-sigfwhm*std)
                t_o = t_o[masker]
                print ('sources within threshould fwhm :',len(t_o))
        cutPSF = []
        cutMask = []
        cutSigma = []
        cutResidual = []
        Ptstars = []
        fwhmlist = []
        SMAlist = []
        qlist = []
        Chi_r = []
        C_x = []
        C_y = []
        for i in range(len(t_o)):
            x = t_o['maxval_xindex'][i]
            y = t_o['maxval_yindex'][i]
            nlabel = segm_data[y,x]
            minx = np.max([0,x-size//2])
            maxx = np.min([data.shape[1],x+size//2+1])
            miny = np.max([0,y-size//2])
            maxy = np.min([data.shape[0],y+size//2+1])
            reachbound = False
            if minx == 0 or maxx == data.shape[1] or miny == 0 or maxy == data.shape[0]:
                reachbound = True
            localdata = data[miny:maxy,minx:maxx]
            localsege = segm_data[miny:maxy,minx:maxx]
            localsigma = sigmaimg[miny:maxy,minx:maxx]
            localmask = mask[miny:maxy,minx:maxx]
            localmask[(localsege!=nlabel)&(localsege!=0)] = True
            localmask[np.isnan(localdata)] = True
            localmask[np.isinf(localdata)] = True
            localsigma[localsigma == 0] = 1.
            localdata[localmask] = 0.
            localsigma[localmask] = 1.
            #print ("check local data",np.sum(localdata),np.sum(localsigma))
            if np.sum(localmask/(size**2)) > isolate or reachbound:
                continue
            if referimg is not None:
                image_vec = localdata[~localmask].flatten()
                refer_vec = referimg[~localmask].flatten()
                sigma_vec = localsigma[~localmask].flatten()
                #scaled = np.dot(image_vec,refer_vec)/np.dot(refer_vec,refer_vec)
                weights = 1.0 / (sigma_vec ** 2)
                numerator = np.sum(weights * image_vec * refer_vec)
                denominator = np.sum(weights * refer_vec ** 2)
                scaled = numerator / denominator
                residual = (localdata-scaled*referimg)/sigmaimg[miny:maxy,minx:maxx]
                chi_r = np.sum(residual**2)/size**2
                if (chi_r > referthre)|(np.isnan(chi_r)):
                    continue
                cutResidual.append(jnp.array(residual,dtype=jnp.float32))
            else:
                cutResidual.append(None) 
                chi_r = 0
            cutPSF.append(jnp.array(localdata,dtype=jnp.float32))
            cutMask.append(jnp.array(localmask,dtype=bool))
            if sigmaimg is not None:
                cutSigma.append(jnp.array(sigmaimg[miny:maxy,minx:maxx],dtype=jnp.float32))
                weight = 1./cutSigma[-1]**2
            else:
                weight = jnp.ones_like(cutPSF[-1])
            #print ("shape check",cutMask.shape,weight.shape)
            weight = jnp.where(cutMask[-1],0,weight)
            Ptstars.append(EPSFStar(cutPSF[-1],weights=weight,cutout_center=(x-minx,y-miny)))
            fwhmlist.append(t_o['fwhm'][i])
            SMAlist.append(t_o['semimajor_sigma'][i])
            qlist.append(t_o['semiminor_sigma'][i]/t_o['semimajor_sigma'][i])
            Chi_r.append(chi_r)
            C_x.append(x)
            C_y.append(y)
        print ('sources isolate (or look like the input PSF):',len(cutPSF))

        circle = np.zeros([size,size],dtype=bool)
        circle1 = Maskellipse(circle,[0.5*size,0.5*size],0.25*size,0,0)
        removeloop = []
        for loopx in range(len(cutPSF)):
            lmask = np.array(cutMask[loopx])
            if np.sum(lmask[circle1]) > central_tol:
                removeloop.append(loopx)
        cutPSF = [i for j, i in enumerate(cutPSF) if j not in removeloop]
        cutMask = [i for j, i in enumerate(cutMask) if j not in removeloop]
        cutResidual = [i for j, i in enumerate(cutResidual) if j not in removeloop]
        Ptstars = [i for j, i in enumerate(Ptstars) if j not in removeloop]
        fwhmlist = [i for j, i in enumerate(fwhmlist) if j not in removeloop]
        SMAlist = [i for j, i in enumerate(SMAlist) if j not in removeloop]
        qlist = [i for j, i in enumerate(qlist) if j not in removeloop]
        Chi_r = [i for j, i in enumerate(Chi_r) if j not in removeloop]
        C_x = [i for j, i in enumerate(C_x) if j not in removeloop]
        C_y = [i for j, i in enumerate(C_y) if j not in removeloop]
        if sigmaimg is not None:
            cutSigma = [i for j, i in enumerate(cutSigma) if j not in removeloop]
        else:
            cutSigma = None
        print ('sources remove central bad :',len(cutPSF))

        if len(droppsf) > 0:
            cutPSF = [i for j, i in enumerate(cutPSF) if j not in droppsf]
            cutMask = [i for j, i in enumerate(cutMask) if j not in droppsf]
            cutResidual = [i for j, i in enumerate(cutResidual) if j not in droppsf]
            Ptstars = [i for j, i in enumerate(Ptstars) if j not in droppsf]
            fwhmlist = [i for j, i in enumerate(fwhmlist) if j not in droppsf]
            SMAlist = [i for j, i in enumerate(SMAlist) if j not in droppsf]
            qlist = [i for j, i in enumerate(qlist) if j not in droppsf]
            Chi_r = [i for j, i in enumerate(Chi_r) if j not in droppsf]
            C_x = [i for j, i in enumerate(C_x) if j not in droppsf]
            C_y = [i for j, i in enumerate(C_y) if j not in droppsf]
            if sigmaimg is not None:
                cutSigma = [i for j, i in enumerate(cutSigma) if j not in droppsf]
        
        final_cordinates = []
        if star_savepath is not None:
            for i in range(len(cutPSF)):
                sky_cord = wcs.all_pix2world(C_x[i],C_y[i],0)
                final_cordinates.append(sky_cord)
                hdu_temp = fits.PrimaryHDU(np.array(cutPSF[i],dtype=np.float32))
                hdu_mask = fits.ImageHDU(np.array(cutMask[i],dtype=np.float32))
                if sigmaimg is not None:
                    hdu_error = fits.ImageHDU(np.array(cutSigma[i],dtype=np.float32))
                else:
                    hdu_error = fits.ImageHDU(np.ones_like(cutPSF[i])*np.nan)
                weight_i = 1./cutSigma[i]**2
                weight_i = np.where(cutMask[i],0,weight_i)
                hduweight = fits.ImageHDU(np.array(weight_i,dtype=np.float32))
                hdulist = fits.HDUList([hdu_temp,hdu_mask,hdu_error,hduweight])
                hdulist.writeto(star_savepath+'star_{0}.fits'.format(i), overwrite=True)
            tab_star = Table(np.array(final_cordinates),names=['RA','DEC'])
            tab_star.write(star_savepath+'star_cord.ipac',format='ipac',overwrite=True)
        
        if plot:
            tlens = len(cutPSF)
            if (((tlens//5)+1)*5-tlens) < (((tlens//4)+1)*4-tlens):
                ncols = 5
                nrows = (tlens//5)+1
            else:
                ncols = 4
                nrows = (tlens//4)+1
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows),squeeze=True)
            ax = ax.ravel()
            for i in range(tlens):
                image = cutPSF[i]
                sky_mean, sky_median, sky_std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
                immin = 5*sky_std
                frac = 0.4
                ax[i].imshow(normimg(image, immin, sky = 0.,frac=frac), cmap='seismic', origin='lower', vmin=-1, vmax=1,interpolation='nearest')
                ax[i].imshow(np.array(cutMask[i],dtype=float), cmap='Blues', origin='lower', alpha=0.5*np.array(cutMask[i],dtype=float))
                ax[i].text(0.95, 0.05, str(i), color='k',transform=ax[i].transAxes,ha='right', va='bottom')
                ax[i].text(0.05, 0.05, 'FWHM: {0:.2f}'.format(fwhmlist[i]), color='k',transform=ax[i].transAxes,ha='left', va='bottom')
                ax[i].text(0.05, 0.2, 'a: {0:.2f} q: {1:.2f}'.format(SMAlist[i],qlist[i]), color='k',transform=ax[i].transAxes,ha='left', va='bottom')
                ax[i].text(0.05, 0.95, 'Chi_r: {0:.2f}'.format(Chi_r[i]), color='k',transform=ax[i].transAxes,ha='left', va='top')
            if figurepath is not None:
                plt.savefig(figurepath + 'stars.png', dpi = 200, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            if referimg is not None:
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows),squeeze=True)
                ax = ax.ravel()
                for i in range(tlens):
                    image = cutResidual[i]
                    ax[i].imshow(image,cmap='seismic', origin='lower',vmin=-10, vmax=10,interpolation='nearest')
                    ax[i].imshow(np.array(cutMask[i],dtype=float), cmap='Blues', origin='lower', alpha=0.5*np.array(cutMask[i],dtype=float))
                    ax[i].text(0.95, 0.05, str(i), color='k',transform=ax[i].transAxes,ha='right', va='bottom')
                    ax[i].text(0.05, 0.05, 'FWHM: {0:.2f}'.format(fwhmlist[i]), color='k',transform=ax[i].transAxes,ha='left', va='bottom')
                    ax[i].text(0.05, 0.95, 'Chi_r: {0:.2f}'.format(Chi_r[i]), color='k',transform=ax[i].transAxes,ha='left', va='top')
                if figurepath is not None:
                    plt.savefig(figurepath + 'residuals.png', dpi = 200, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
                
        if len(cutPSF) < 3:
            print ('not enough stars to fit ePSF, use stack PSF')
            makestackpsf = True

        if notfit:
            return None
        if photutils_method:
            epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=150,progress_bar=False)
            epsf,fitted_stars = epsf_builder(EPSFStars(Ptstars))
            epsf = jnp.array(epsf.data,dtype=jnp.float32)
        else:
            #cutSigma = None
            epsf = fittingPSF(cutPSF,cutMask,cutSigma,oversampling = oversampling, a_TV = a_TV, learning_rate=learning_rate, a_sky=a_sky, makestackpsf = makestackpsf,num_steps = num_steps, verbose = verbose)
        if filepaths is not None:
            hdu = fits.PrimaryHDU(epsf)
            After = fits.HDUList([hdu])
            After.writeto(filepaths,overwrite= True)

        if plot:
            fig=plt.figure(figsize=(10,10))
            image = epsf
            sky_mean, sky_median, sky_std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
            immin = 5*sky_std
            frac = 0.4
            plt.imshow(normimg(image, immin, sky = 0.,frac=frac), cmap='seismic', origin='lower', vmin=-1, vmax=1,interpolation='nearest')
            if figurepath is not None:
                plt.savefig(figurepath + 'ePSF.png', dpi = 200, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        return epsf

    def write_model(self, filepath):
        '''
        Write model image to a fits file.
        '''
        hdu = fits.PrimaryHDU(np.array(self.model_image, dtype=np.float32))
        hdu.writeto(filepath, overwrite=True)
        return
    
    def reproject(self, ra, dec, r, oversampling=1):
        """
        Extract and resample image cutout to arbitrary pixel scale and orientation.
        
        Uses `reproject.reproject_interp` to create cutout centered on specified
        coordinates with custom pixel scale. Useful for matching images to common grid,
        zooming into regions, or aligning data from different instruments.

        Parameters
        ----------
        ra : float
            Right ascension of cutout center (degrees).
        dec : float
            Declination of cutout center (degrees).
        r : int
            Half-width of cutout (pixels in *output* frame). Output shape will be
            (2*r+1, 2*r+1).
        oversampling : float, default: 1
            Pixel scale reduction factor. Output pixel scale = input pixel scale / 
            oversampling. E.g., oversampling=2 makes pixels twice as fine (0.1" → 0.05").

        Returns
        -------
        imcut : ndarray
            Reprojected image cutout with shape (2*r+1, 2*r+1). Pixels outside original
            image footprint are NaN.

        Notes
        -----
        **WCS Construction**:
        
        Creates target WCS with:
        - CRVAL = (RA, Dec) - cutout center
        - CRPIX = (r, r) - places center at image center
        - CDELT = ±pixel_scale / oversampling (degrees/pixel)
        - CTYPE = "RA---TAN", "DEC--TAN" (tangent plane projection)
        
        **Interpolation Method**:
        
        Uses `reproject_interp` which:
        - Transforms pixel coordinates between input and output WCS
        - Performs bi-linear interpolation of flux values
        - Conserves surface brightness (not flux)
        - Handles rotation and scaling simultaneously
        
        This is **not** a simple cutout - it resamples the data, so noise properties
        change. For analysis requiring independent pixels, consider drizzling instead.
        
        Currently only supports 2D images (not IFS cubes).

        Examples
        --------
        Extract 5 arcsec cutout at native resolution:
        
        >>> img = image('hst_wfc3.fits')  # pixel scale 0.06 arcsec
        >>> cutout = img.reproject(ra=150.123, dec=2.456, r=42, oversampling=1)
        >>> print(cutout.shape)
        (85, 85)  # 85 pixels * 0.06 arcsec/pix = 5.1 arcsec
        
        Zoom in 4x to resolve fine structure:
        
        >>> cutout_zoom = img.reproject(ra=150.123, dec=2.456, r=84, 
        ...                             oversampling=4)
        >>> print(cutout_zoom.shape)
        (169, 169)  # pixel scale now 0.015 arcsec
        
        Match JWST NIRCam (0.03"/pix) to HST/ACS (0.05"/pix):
        
        >>> jwst = image('jwst_nircam.fits')
        >>> cutout = jwst.reproject(ra=150.123, dec=2.456, r=100, 
        ...                         oversampling=0.6)  # 0.03/0.6 = 0.05
        
        See Also
        --------
        img_cut : Fast native-resolution cutout extraction
        reproject.reproject_interp : Underlying reprojection function
        """
        # assuming x and y has the same pixel scale
        app = np.abs(self.pixel_scales[0].value)/oversampling
        scale = app / 3600.0  # degrees per pixel 
        naxis = 2*r+1
        wcs_cut = WCS(naxis=2)
        wcs_cut.wcs.crpix = [r, r] 
        wcs_cut.wcs.cdelt = [-scale, scale]
        wcs_cut.wcs.crval = [ra, dec]  
        wcs_cut.wcs.ctype = ["RA---TAN", "DEC--TAN"] 
        wcs_cut.wcs.equinox = 2000.0
        wcs_cut.wcs.radesys = 'ICRS'
        # TODO: add IFS support?
        if isinstance(self.data, CCDData):
            data = self.data.data
        else:
            data = self.data
        wcs = self.data.wcs
        imcut, footprint = reproject_interp((data, wcs), wcs_cut, shape_out=(naxis, naxis))
        return imcut
        

class image_atlas(object):
    """
    Container for multi-band imaging and spectroscopy observations.
    
    Groups multiple `image` objects (photometric bands) and optional grism spectra
    for simultaneous multi-band SED fitting. Handles cross-matching catalogs,
    coordinated masking, PSF building, and residual tracking across all bands.

    Parameters
    ----------
    image_list : list of image, optional
        List of `image` objects, one per photometric band. Should all observe the
        same target region (though pixel grids may differ).
    band_list : list of str, optional
        Band identifiers (e.g., ['F435W', 'F606W', 'F814W']). Must match length of
        `image_list`. Check `instrument_info` for standard naming.
    grism_image_list : list of image, optional
        List of `image` objects for grism/IFU spectroscopy. Spectral dimension handled
        separately from photometry.
    grism_band_list : list of str, optional
        Identifiers for grism observations.
    same_grid : bool, default: True
        If True, all images share the same pixel grid (pre-aligned). Enables faster
        processing. If False, coordinate transformations handled per-image.
    refere_agn : int, default: 0
        Index of reference band for AGN modeling (if fitting AGN+host decomposition).
    Nlogv : int, default: 2000
        Number of logarithmic velocity bins for emission line fitting.
    name : str, default: ''
        Dataset identifier for tracking multiple image atlases.

    Attributes
    ----------
    image_list : list of image
        Photometric band images.
    band_list : list of str
        Band names.
    common_catalog : astropy.table.Table
        Cross-matched source catalog from `make_common_catalog()`. Contains positions,
        shapes, and master (maximum across bands) semi-major/minor axes.
    spectra : list
        Attached spectra from `add_spectrum()`. Can be fiber, longslit, or IFU.
    specapermask : list of ndarray
        Aperture masks for spectral extraction, one per spectrum.
    img_residual : ndarray
        Chi-square residuals per band after fitting, shape (N_bands, 1).
    img_dof : ndarray
        Degrees of freedom per band, shape (N_bands, 1).
    spec_residual : list
        Chi-square residuals per spectrum.
    spec_dof : list
        Degrees of freedom per spectrum.

    Methods
    -------
    add_spectrum(spectrum, referimag, sposition, sapersize, ...)
        Attach spectroscopic data with aperture definition.
    source_detection(nsigma)
        Detect sources in all bands.
    make_common_catalog(CM_separation, magnification)
        Cross-match multi-band catalogs.
    master_mask(magnification, applylist)
        Apply unified masking based on common catalog.
    sky_subtraction(order, filepaths)
        Background subtraction across all bands.
    generate_PSFs(equivalent_radius, size, oversampling, ...)
        Build PSF from stars common to all bands.

    Notes
    -----
    **Typical Workflow**:
    
    1. Create `image` objects for each band::
    
        img_f435w = image('f435w.fits', zero_point=25.68, ...)
        img_f606w = image('f606w.fits', zero_point=26.51, ...)
        atlas = image_atlas([img_f435w, img_f606w], ['F435W', 'F606W'])
    
    2. Sky subtraction and source detection::
    
        atlas.sky_subtraction(order=3)
        atlas.source_detection(nsigma=3.0)
    
    3. Cross-match and mask contaminating sources::
    
        atlas.make_common_catalog(CM_separation=2.5)
        atlas.master_mask(magnification=3.0)
    
    4. Build PSF from common stars::
    
        atlas.generate_PSFs(equivalent_radius=1.5, size=31, oversampling=2)
    
    5. Cut target regions and attach to fitter::
    
        for img in atlas.image_list:
            img.img_cut(ra=150.123, dec=2.456, cutsize=10.0)
    
    **Residual Tracking**:
    
    After fitting, `img_residual` and `spec_residual` store chi-square values for
    each band/spectrum, enabling identification of poor fits or missing model components.

    Examples
    --------
    Create atlas from HST/ACS multi-band images:
    
    >>> from galfits.images import image, image_atlas
    >>> f435w = image('acs_f435w.fits', zero_point=25.68)
    >>> f606w = image('acs_f606w.fits', zero_point=26.51)
    >>> f814w = image('acs_f814w.fits', zero_point=25.94)
    >>> atlas = image_atlas([f435w, f606w, f814w], ['F435W', 'F606W', 'F814W'])
    >>> print(len(atlas))
    3
    
    Access individual bands:
    
    >>> img_v = atlas['F606W']
    >>> img_v.sigma_clipped_stats()
    
    See Also
    --------
    image : Single-band image class
    galfitS_data : Multi-atlas container for handling multiple targets/instruments
    """
    def __init__(self, image_list=None, band_list=None, grism_image_list=None, grism_band_list=None, same_grid = True, refere_agn = 0, Nlogv = 2000, name = ''):
        """Initialize multi-band image atlas."""
        if image_list is None:
            self.image_list = []
        else:
            self.image_list = image_list

        if band_list is None:
            self.band_list = []
            for i in range(len(self.image_list)):
                self.band_list.append(self.image_list[i].band)
        else:
            self.band_list = band_list
        self.__length = len(image_list)
        self.common_catalog = None
        self.same_grid = same_grid
        self.spectra = []
        self.modelspec = []
        self.specapermask = []
        self.specprefix = []
        self.specaper = []
        self.hires = []
        self.snorm = []
        self.lsfs = []
        self.lsfK = []
        self.referimag = 0
        self.refere_agn = refere_agn
        self.Nlogv = Nlogv
        self.name = name
        self.img_residual = np.zeros((len(self.image_list),1),dtype=np.float32)
        self.spec_residual = []
        self.img_dof = np.zeros((len(self.image_list),1),dtype=np.float32)
        self.spec_dof = []
        self.grism_image_list = grism_image_list
        self.grism_band_list = grism_band_list
        self.grism_residual = np.zeros((len(self.grism_image_list),1),dtype=np.float32)
        self.grism_dof = np.zeros((len(self.grism_image_list),1),dtype=np.float32)

    def __del__(self):
        # Nullify object's properties
        if self.image_list is not None:
            for im in self.image_list:
                im.__del__()
        self.image_list = None
        self.band_list = None
        self.__length = None
        self.common_catalog = None
        self.same_grid = None
        self.spectra = None
        self.modelspec = None
        self.specapermask = None
        self.specprefix = None
        self.specaper = None
        self.hires = None
        self.snorm = None
        self.referimag = None
        self.grism_band_list = None
        self.grism_image_list = None

    def add_spectrum(self, spectrum, referimag=-1, sposition=[0,0], sapersize=2.,prefix='', stype = 'fiber', PA = 0, a=1 , b=1, hires = False, norm = 1., lsf=None):
        """
        Attach spectroscopic observation with aperture definition.
        
        Links spectrum to photometric reference image and defines extraction aperture
        geometry. Enables joint image+spectrum SED fitting with spatially-matched
        constraints.

        Parameters
        ----------
        spectrum : array-like or spectrum object
            Spectroscopic data (wavelength, flux, error arrays or spectrum object).
        referimag : int, default: -1
            Index of reference image in `self.image_list` for aperture definition.
            If -1, no aperture mask created (whole image).
        sposition : list of float, default: [0, 0]
            Aperture center in sky offset coordinates (arcsec) from reference image
            cutout center. [0,0] = use galaxy center from photometry.
        sapersize : float, default: 2.0
            Aperture size parameter (arcsec). Interpretation depends on `stype`:
            - 'fiber': radius
            - 'longslit': half-width perpendicular to slit
        prefix : str, default: ''
            Identifier prefix for this spectrum (e.g., ``MOS1_``, ``IFU_``).
        stype : {'fiber', 'longslit'}, default: 'fiber'
            Aperture geometry type:
            - 'fiber': circular aperture
            - 'longslit': rectangular slit
        PA : float, default: 0
            Position angle of longslit (degrees, east of north). Only used if
            `stype='longslit'`.
        a : float, default: 1
            Longslit half-length along slit direction (arcsec). Only used if
            `stype='longslit'`.
        b : float, default: 1
            Longslit half-width perpendicular to slit (arcsec). Only used if
            `stype='longslit'`.
        hires : bool, default: False
            If True, use high-resolution spectral templates (e.g., for echelle data).
        norm : float, default: 1.0
            Flux normalization factor applied to spectrum before fitting.
        lsf : array-like or None, optional 
            2D array defining line spread function (LSF) for spectral [wave, R]
        Notes
        -----
        **Aperture Masking**:
        
        Creates boolean mask with shape matching reference image cutout. Mask is
        flattened and stored in `self.specapermask` for efficient aperture integration
        during model generation.
        
        For fiber aperture:
        
        .. math::
        
            {{\\rm mask}} = \\sqrt{{(x - x_c)^2 + (y - y_c)^2}} < r_{{\\rm aper}}
        
        For longslit:
        
        .. math::
        
            {{\\rm mask}} = (|{{\\bf r}} \\cdot {{\\bf \\hat{{s}}}}| < a) \\land 
                           (|{{\\bf r}} \\cdot {{\\bf \\hat{{n}}}}| < b)
        
        where :math:`{{\\bf \\hat{{s}}}}` is unit vector along slit (PA) and 
        :math:`{{\\bf \\hat{{n}}}}` is perpendicular.
        
        **Residual Tracking**:
        
        Initializes `spec_residual` and `spec_dof` entries for this spectrum,
        updated during fitting.

        Examples
        --------
        Add fiber spectrum centered on galaxy:
        
        >>> from galfits.spectrum import spectrum
        >>> spec = spectrum('muse_spectrum.fits')
        >>> atlas.add_spectrum(spec, referimag=0, sposition=[0,0], 
        ...                    sapersize=1.5, stype='fiber')
        
        Add off-center longslit along PA=45°:
        
        >>> atlas.add_spectrum(spec, referimag=0, sposition=[0.5, -0.3],
        ...                    stype='longslit', PA=45, a=3.0, b=0.5)
        
        Add spectrum without spatial matching (no aperture):
        
        >>> atlas.add_spectrum(spec, referimag=-1)
        
        See Also
        --------
        image.img_cut : Define cutout region for aperture mapping
        """
        self.spectra.append(spectrum)
        self.referimag = referimag
        if referimag == -1:
            self.specapermask.append(1)
        elif stype == 'fiber':
            im = self.image_list[referimag]
            ny,nx = im.cut_image.shape
            xaxis = np.arange(nx)
            yaxis = np.arange(ny)
            xmesh, ymesh = np.meshgrid(xaxis, yaxis)
            xcen, ycen = coordinates_transfer(sposition[0],sposition[1],im.coordinates_transfer_para)
            r = np.sqrt( (xmesh+0.5 - xcen)**2. + (ymesh+0.5 - ycen)**2.)
            aperture_mask = r < (sapersize/im.pixel_scales[0].value)
            self.specapermask.append(jnp.array(aperture_mask, dtype=jnp.bool_).ravel())
        elif stype == 'longslit':
            im = self.image_list[referimag]
            ny,nx = im.cut_image.shape
            xaxis = np.arange(nx)
            yaxis = np.arange(ny)
            xmesh, ymesh = np.meshgrid(xaxis, yaxis)
            xcen, ycen = coordinates_transfer(sposition[0],sposition[1],im.coordinates_transfer_para)
            # along the slit
            xdir, ydir = coordinates_transfer(sposition[0]+np.cos(PA*np.pi/180.),sposition[1]+np.sin(PA*np.pi/180.),im.coordinates_transfer_para)
            dvector = np.array([xdir-xcen,ydir-ycen])
            dvector = dvector/np.sqrt(np.sum(dvector**2))
            dx = xmesh+0.5 - xcen
            dy = ymesh+0.5 - ycen
            r = np.abs(dx*dvector[0]+dy*dvector[1]) 
            aperture_mask = r < (a/im.pixel_scales[0].value)
            # perpendicular to the slit
            xdir, ydir = coordinates_transfer(sposition[0]-np.sin(PA*np.pi/180.),sposition[1]+np.cos(PA*np.pi/180.),im.coordinates_transfer_para)
            dvector = np.array([xdir-xcen,ydir-ycen])
            dvector = dvector/np.sqrt(np.sum(dvector**2))
            dx = xmesh+0.5 - xcen
            dy = ymesh+0.5 - ycen
            r = np.abs(dx*dvector[0]+dy*dvector[1])
            aperture_mask = aperture_mask & (r < (b/im.pixel_scales[0].value))
            self.specapermask.append(jnp.array(aperture_mask, dtype=jnp.bool_).ravel())
        self.specprefix.append(prefix)
        self.hires.append(hires)
        self.snorm.append(norm) 
        self.spec_residual.append(0)
        self.spec_dof.append(0)
        self.lsfs.append(lsf)
        if lsf is not None:
            K = choose_kernel_half_width(
                spectrum[0], lsf[0], lsf[1],
                Nlogv=4000, nsig=6.0)
        else:
            K = 50
        self.lsfK.append(K)
        return

    def __getitem__(self, key):
        '''
        Get the image data using the filter name or number index.
        '''
        if type(key) is str:
            idx = self.band_list.index(key)
        elif type(key) is int:
            idx = key
        return self.image_list[idx]

    def __len__(self):
        '''
        Get the length of the data list.
        '''
        return self.__length

    def init_sed(self):
        '''
        Initialize the SED model
        '''
        resps = []
        photos = []
        errors = []
        for im in self.image_list:
            resps.append(im.resp)
            photos.append(jnp.sum(im.cut_image))
            errors.append(jnp.sum(im.cut_sigma_image))
        self.resps = resps
        self.photos = jnp.array(photos,dtype=jnp.float32)
        self.photoerrs = jnp.array(errors,dtype=jnp.float32)

    def source_detection(self,nsigma=3.0):
        '''
        Do multi-band source detection
        Parameters
        ----------
        nsigma : float, or a array with same size as image_atlas
            source detection threshold
        '''
        if type(nsigma) == float:
            nsigma = nsigma * np.ones(self.__length,dtype=float)
        for loop in range(self.__length):
            fwhm = self.image_list[loop].psfFWHM_arcsec/self.image_list[loop].pixel_scales[0].value
            self.image_list[loop].source_detection_individual(fwhm,nsigma=nsigma[loop],sc_key=loop+1)
        return

    def make_common_catalog(self,CM_separation=2.5,magnification=3.0,applylist=None):
        '''
        Do multi-band source detection
        Parameters
        ----------
        CM_separation : float
        angular separation used to do sky coordinates crossmatching, unit in deg
        magnification : float, or a array with same size as image_atlas
                        magnification for generating mask foe each image
        applylist : [list of index]
        None for all images
        '''
        if type(magnification) == float:
            magnification = magnification * np.ones(self.__length,dtype=float)
        if applylist is None:
            applylist = np.arange(self.__length)
        cats = []
        for loop in applylist:
            cats.append(self.image_list[loop].sources_skycord)
        comc = cross_match(cats,angular_sep = 2.5)
        lencc = len(comc)
        master_a = np.zeros(lencc, dtype = float)
        master_b = np.zeros(lencc, dtype = float)
        for loop in range(len(comc)):
            a = []
            b = []
            for loop2 in applylist:
                a.append(self.image_list[loop2].sources_catalog['semimajor_sigma'][comc['sloop_{0}'.format(loop2+1)][loop]]
                *magnification[loop2]*self.image_list[loop2].pixel_scales[0].value)
                b.append(self.image_list[loop2].sources_catalog['semiminor_sigma'][comc['sloop_{0}'.format(loop2+1)][loop]]
                *magnification[loop2]*self.image_list[loop2].pixel_scales[0].value)
            master_a[loop] = np.max(np.array(a))
            master_b[loop] = np.max(np.array(b))
        comc.add_column(Column(master_a, name = 'master_a'))
        comc.add_column(Column(master_b, name = 'master_b'))
        self.common_catalog = comc

    def sky_subtraction(self,order=3,filepaths=None):
        '''
        Do multi-band sky subtration
        Parameters
        ----------
        order (optional) : int
            order of the polynomial
        filepaths : filepath to store the sky subtracted images
        '''
        if type(order) == int:
            order = order * np.ones(self.__length,dtype=int)
        for loop in range(self.__length):
            if filepaths is None:
                self.image_list[loop].sky_subtraction(order[loop])
            else:
                self.image_list[loop].sky_subtraction(order[loop],filepath=filepaths[loop])
        return

    def master_mask(self, magnification=3.0, applylist=None):
        '''
        Do multi-band source masking
        Parameters
        ----------
        magnification : float, or a array with same size as image_atlas
                        magnification for generating mask foe each image
        applylist : [list of index]
        None for all images
        '''
        if type(magnification) == float:
            magnification = magnification * np.ones(self.__length,dtype=float)
        if applylist is None:
            applylist = np.arange(self.__length)
        comc = self.common_catalog.copy()
        commonsourcelist = []
        for loop2 in applylist:
            newsc = self.image_list[loop2].sources_catalog.copy()
            for loop in range(len(comc)):
                self.image_list[loop2].sources_catalog['semimajor_sigma'][comc['sloop_{0}'.format(loop2+1)][loop]] = comc['master_a'][loop]/(magnification[loop2]*self.image_list[loop2].pixel_scales[0].value)
                self.image_list[loop2].sources_catalog['semiminor_sigma'][comc['sloop_{0}'.format(loop2+1)][loop]] = comc['master_b'][loop]/(magnification[loop2]*self.image_list[loop2].pixel_scales[0].value)
            indexes = np.delete(np.arange(len(self.image_list[loop2].sources_catalog)), comc['sloop_{0}'.format(loop2+1)])
            newsc.remove_rows(indexes)
            commonsourcelist.append(newsc)
        for loop2 in range(self.__length):
            self.image_list[loop2].make_mask(sources=commonsourcelist[loop2],magnification=magnification[loop2])

    def generate_PSFs(self, equivalent_radius=1.5, threshold = 50., axrit=0.75, size = 20.,oversampling=1, plot=None, filepaths=None, photutils = False):
        '''
        Generate effective point spread fuctions (ePSFs) for each image
        Parameters
        ----------
        equivalent_radius : float,
                            radius criteria to indentify star: fwhm of sourve < equivalent_radius*psfFWHM
        size : float, unit pixel
               use what size box to extract stars
        oversampling : int
                       oversample the ePSF
        plot : None for not plot stars & ePSF
               list like [1,2,3] to plot rgb image
        filepaths : filepath to store the ePSFs
        '''
        stars = self.common_catalog.copy()
        remolist = []
        flux_thre = []
        for loop2 in range(self.__length):
            flux_thre.append(np.percentile(self.image_list[loop2].sources_catalog['segment_flux'],threshold))
        for loop in range(len(stars)):
            for loop2 in range(self.__length):
                catloop = stars['sloop_{0}'.format(loop2+1)][loop]
                a = (self.image_list[loop2].sources_catalog['fwhm'][catloop])*self.image_list[loop2].pixel_scales[0].value/self.image_list[loop2].psfFWHM_arcsec
                ar = self.image_list[loop2].sources_catalog['semiminor_sigma'][catloop]/self.image_list[loop2].sources_catalog['semimajor_sigma'][catloop]
                fls = self.image_list[loop2].sources_catalog['segment_flux'][catloop]
                sigvalue = self.image_list[loop2].sigma_image[self.image_list[loop2].sources_catalog['maxval_yindex'][catloop],self.image_list[loop2].sources_catalog['maxval_xindex'][catloop]]
                if (a > equivalent_radius)|(ar<axrit)|(fls<flux_thre[loop2])|np.isinf(sigvalue):
                    remolist.append(loop)
                    break
        stars.remove_rows(remolist)
        star_images = []
        PSFs = []
        for loop2 in range(self.__length):
            newsc = self.image_list[loop2].sources_catalog.copy()
            indexes = np.delete(np.arange(len(self.image_list[loop2].sources_catalog)), stars['sloop_{0}'.format(loop2+1)])
            newsc.remove_rows(indexes)
            stars_tbl = Table()
            stars_tbl['x']=np.array(newsc['maxval_xindex'])
            stars_tbl['y']=np.array(newsc['maxval_yindex'])
            if self.image_list[loop2].ss_data is not None:
                nddata = NDData(data=np.array(self.image_list[loop2].ss_data))
            else:
                nddata = NDData(data=np.array(self.image_list[loop2].data.data-self.image_list[loop2].sky_median))
            Tstar = extract_stars(nddata, stars_tbl, size=size)
            epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=15,progress_bar=False)
            epsf, fitted_stars = epsf_builder(Tstar)
            self.image_list[loop2].PSF = epsf.data
            if filepaths is not None:
                hdu = fits.PrimaryHDU(epsf.data.astype('float32'))
                After = fits.HDUList([hdu])
                After.writeto(filepaths[loop2],overwrite= True)
            if plot is not None:
                star_images.append(Tstar)
                PSFs.append(epsf.data)
        #print ('len',len(Tstar),len(stars))
        if plot is not None:
            tlens = len(Tstar)
            if (((tlens//5)+1)*5-tlens) < (((tlens//4)+1)*4-tlens):
                ncols = 5
                nrows = (tlens//5)+1
            else:
                ncols = 4
                nrows = (tlens//4)+1
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows),squeeze=True)
            ax = ax.ravel()
            for i in range(tlens):
                if len(plot) > 2:
                    star_b = star_images[plot[0]][i].data*100./np.sum(star_images[plot[0]][i].data)
                    star_g = star_images[plot[1]][i].data*100./np.sum(star_images[plot[1]][i].data)
                    star_r = star_images[plot[2]][i].data*100./np.sum(star_images[plot[2]][i].data)
                    norm = simple_norm(star_b, 'log', percent=99.)
                    image = make_lupton_rgb(star_r, star_g, star_b, Q=10)
                else:
                    image = star_images[plot[0]][i].data
                    norm = simple_norm(image, 'log', percent=99.)
                ax[i].imshow(image,norm=norm ,origin='lower')
            plt.show()
            fig=plt.figure(figsize=(10,10))
            if len(plot) > 2:
                star_b = PSFs[plot[0]]*100./np.sum(PSFs[plot[0]])
                star_g = PSFs[plot[1]]*100./np.sum(PSFs[plot[1]])
                star_r = PSFs[plot[2]]*100./np.sum(PSFs[plot[2]])
                norm = simple_norm(star_b, 'log', percent=99.)
                image = make_lupton_rgb(star_r, star_g, star_b, Q=10)
            else:
                image = PSFs[plot[0]]
                norm = simple_norm(image, 'log', percent=99.)
            plt.imshow(image,norm=norm ,origin='lower')
            plt.show()

class galfitS_data(object):
    """
    Top-level data container for GalfitS multi-instrument, multi-target fitting.
    
    Aggregates multiple `image_atlas` objects (e.g., HST, JWST, ground-based for
    different targets or instruments) into unified interface. Handles coordinate
    transformations between datasets, tracks all bands across atlases, and manages
    flux conversion factors for heterogeneous data.

    Parameters
    ----------
    image_atlas_list : list of image_atlas, optional
        List of `image_atlas` objects. Each atlas typically represents one target
        observed with one instrument, or one instrument observing multiple targets.
    image_atlas_names : list of str, optional
        Identifiers for each atlas (e.g., ['HST_ACS', 'JWST_NIRCam', 'MUSE']).
    fitMode : str, default: 'undefined'
        Fitting mode determined by data types present. Automatically set by
        `read_config_file` to one of:
        - 'images - photometry': Multi-band photometry only
        - 'images - SED': Multi-band images with SED constraints
        - 'images + spectra - SED': Combined imaging and spectroscopy
        - 'spectrum - SED': Spectroscopy only

    Attributes
    ----------
    image_atlas_list : list of image_atlas
        All image atlases.
    image_atlas_names : list of str
        Atlas identifiers.
    allbands : list of str
        Flattened list of all photometric band names across all atlases.
    listindex : list of int
        Atlas index for each band in `allbands`.
    atlasindex : list of int
        Image index within atlas for each band in `allbands`.
    allgrismbands : list of str
        Flattened list of all grism band names.
    grismlistindex : list of int
        Atlas index for each grism.
    grismatlasindex : list of int
        Grism index within atlas.
    Nimages : int
        Number of true 2D images (excluding photometry points).
    image_index : list of int
        Indices in `allbands` corresponding to 2D images.
    imageset : list of list of int
        Grouping of images by orientation/pixel scale. Images in same group share
        coordinate transformation parameters.
    imagesizes : list of tuple
        Reference image shape for each group in `imageset`.
    coordinates_transfer_paras : list of dict
        Coordinate transformation parameters for each group.
    filterwaves : list of ndarray
        Wavelength grids for filter response functions, one per imageset group.

    Methods
    -------
    get_image(key)
        Retrieve individual image by band name or flat index.
    get_grism_image(key)
        Retrieve grism image by band name or flat index.
    calculate_tranformation(angle)
        Group images by orientation and set coordinate transformations.

    Notes
    -----
    **Coordinate Transformation Grouping**:
    
    Images from different instruments often have different orientations (e.g., HST at
    PA=45° vs JWST at PA=120°). `calculate_tranformation` groups images by position
    angle and selects the highest-resolution image in each group as reference.
    
    This enables efficient model evaluation: generate model on reference grid, then
    apply rotation/scaling transformations for other images in group.
    
    **Flux Conversion**:
    
    Each image has `phys_to_counts_rate` attribute set during initialization to convert
    physical flux units (e.g., erg/s/cm²/Å) to observed count rates (e-/s). This
    accounts for:
    - Filter throughput curves
    - Telescope effective area
    - Detector quantum efficiency
    
    Photometry points (no 2D data) have `phys_to_counts_rate=1` as they're already
    integrated fluxes.

    Examples
    --------
    Create multi-instrument dataset:
    
    >>> from galfits.images import image, image_atlas, galfitS_data
    >>> # HST observations
    >>> hst_imgs = [image(f'hst_{b}.fits') for b in ['F435W', 'F606W', 'F814W']]
    >>> hst_atlas = image_atlas(hst_imgs, ['F435W', 'F606W', 'F814W'])
    >>> # JWST observations  
    >>> jwst_imgs = [image(f'jwst_{b}.fits') for b in ['F150W', 'F277W']]
    >>> jwst_atlas = image_atlas(jwst_imgs, ['F150W', 'F277W'])
    >>> # Combined dataset
    >>> data = galfitS_data([hst_atlas, jwst_atlas], ['HST', 'JWST'])
    >>> print(f"Total bands: {len(data.allbands)}")
    Total bands: 5
    
    Access images by band name:
    
    >>> img_f606w = data.get_image('F606W')
    >>> img_f277w = data.get_image('F277W')
    
    Access by flat index:
    
    >>> img_0 = data.get_image(0)  # First image (F435W)
    >>> img_4 = data.get_image(4)  # Last image (F277W)
    
    Group images by orientation:
    
    >>> data.calculate_tranformation(angle=3.0)
    >>> print(f"Image groups: {data.imageset}")
    [[0, 1, 2], [3, 4]]  # HST and JWST have different PAs
    
    See Also
    --------
    image_atlas : Multi-band container for single instrument
    image : Single-band image class
    gsutils.read_config_file : Create galfitS_data from configuration file
    """
    def __init__(self, image_atlas_list=None, image_atlas_names=None, fitMode = 'undefined'):
        """Initialize multi-atlas data container."""
        if image_atlas_list is None:
            self.image_atlas_list = []
        else:
            self.image_atlas_list = image_atlas_list

        if image_atlas_names is None:
            self.image_atlas_names = []
        else:
            self.image_atlas_names = image_atlas_names
        self.init_grid()
        self.__length = len(image_atlas_list)
        self.allbands   = []
        self.wavephots  = []
        self.listindex  = []
        self.atlasindex = []
        self.allgrismbands   = []
        self.grismlistindex  = []
        self.grismatlasindex = []
        self.Nimages = 0
        self.image_index = []
        for loop1,ia in enumerate(image_atlas_list):
            for loop2,im in enumerate(ia):
                self.allbands.append(im.band)
                self.wavephots.append(effective_wave[im.band])
                self.listindex.append(loop1)
                self.atlasindex.append(loop2)
            for loop2,im in enumerate(ia.grism_image_list):
                self.allgrismbands.append(im.band)
                self.grismlistindex.append(loop1)
                self.grismatlasindex.append(loop2)
        fitMode = fitMode
        if len(self.allbands) != 0:
            self.calculate_tranformation()
            
            fwave_NV = []
            for loop in range(len(self.allbands)):
                fwave_NV.append(self.get_image(loop).resp[:,0])
            fwave_NV = jnp.concatenate(fwave_NV)
            fwave_NV = np.unique(np.sort(np.array(fwave_NV)))
            self.filterwavesNV = jnp.array(fwave_NV, dtype=jnp.float32)
            
            self.respsNV = []
            self.respboundsNV = []
            self.respnormsNV = []
            for loop in range(len(self.allbands)):
                im = self.get_image(loop)
                self.respsNV.append(im.resp)
                self.respnormsNV.append(im.respnorm)
                filterx = im.resp[:,0]
                idx_start = jnp.searchsorted(fwave_NV, filterx[0], side='left')
                idx_end = jnp.searchsorted(fwave_NV, filterx[-1], side='right')
                self.respboundsNV.append([idx_start, idx_end])
                
        for loop in range(len(self.allbands)):
            im = self.get_image(loop)
            if not im.photopoint:
                self.Nimages += 1
                self.image_index.append(loop)
            else:
                im.phys_to_counts_rate = 1.

    def __del__(self):
        '''
        Parameters
        ----------
        image_atlas_list (optional) : List
            List of `image_atlas`.
        image_atlas_names (optional) : List
            List of image_atlas name.
        '''
        if self.image_atlas_list is not None:
            for ia in self.image_atlas_list:
                ia.__del__()
        self.image_atlas_names = None
        self.allbands = None
        self.listindex = None
        self.atlasindex = None

    def __getitem__(self, key):
        '''
        Get the image data using the filter name or number index.
        '''
        if type(key) is str:
            idx = self.image_atlas_names.index(key)
        elif type(key) is int:
            idx = key
        return self.image_atlas_list[idx]

    def get_image(self, key):
        if type(key) is str:
            idx = self.allbands.index(key)
        elif type(key) is int:
            idx = key
        return self.image_atlas_list[self.listindex[idx]][self.atlasindex[idx]]
    
    def get_grism_image(self, key):
        if type(key) is str:
            idx = self.allgrismbands.index(key)
        elif type(key) is int:
            idx = key
        return self.image_atlas_list[self.grismlistindex[idx]].grism_image_list[self.grismatlasindex[idx]]

    def calculate_tranformation(self, angle = 3.):
        thetas = []
        pscales = []
        num_images = len(self.allbands)
        first_trans = self.get_image(0).coordinates_transfer_para
        for loop in range(len(self.allbands)):
            im = self.get_image(loop)
            if not im.photopoint:
                first_trans = im.coordinates_transfer_para
                break

        for loop in range(len(self.allbands)):
            im = self.get_image(loop)
            if im.photopoint:
                im.coordinates_transfer_para['delta_ang'] = first_trans['delta_ang']+0.1
                im.coordinates_transfer_para['pixsc'] = 100.
            thetas.append(im.coordinates_transfer_para['delta_ang'])
            pscales.append(im.coordinates_transfer_para['pixsc'])
        imageset = []

        # 2. Sort image indices by their theta values.
        sorted_indices = sorted(range(num_images), key=lambda i: thetas[i])
        
        # 3. Group images: each group contains images with angle differences < given threshold.
        groups = []
        current_group = [sorted_indices[0]]
        # Using the first image in the group as the base for comparison.
        for idx in sorted_indices[1:]:
            # If the difference between this image’s theta and the first image in the group is less than threshold,
            # add it to the current group; otherwise, start a new group.
            if abs(thetas[idx] - thetas[current_group[0]]) < angle:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        groups.append(current_group)  # add the final group

        # Initialize lists to save the final results.
        self.imageset = []               # List of groups; each group is a list of image indices.
        self.imagesizes = []             # List of reference image sizes (e.g., image dimensions).
        self.coordinates_transfer_paras = []  # List of transformation parameters corresponding to each group.
        # self.filterwaves    = []  # List of filter wavelengths for each imageset
        self.sub_wavegrids  = []  # List of wavegrid for each imageset.
        self.subgroup_resps     = [] # resps for sub 

        # 4. For each group, select the best image (with minimum pixel scale) as the reference,
        resps   = []
        for group in groups:
            # Identify the index of the reference image with best (i.e., smallest) pixscale.
            ref_index = min(group, key=lambda idx: pscales[idx])
            ref_image = self.get_image(ref_index)
            ref_para = ref_image.coordinates_transfer_para.copy()
            
            # Here we assume each image has a "size" attribute;
            # Adjust as necessary if the size is stored differently.
            ref_size = ref_image.cut_image.shape

            # Save the group information.
            self.imageset.append(group)
            self.imagesizes.append(ref_size)
            self.coordinates_transfer_paras.append(ref_para)

            # Save wavegrid for each sub group.
            # fwave = []
            # for loop in group:
            #     fwave.append(self.get_image(loop).resp[:,0])
            # fwave = np.concatenate(fwave)
            # fwave = np.array(fwave)
            # fwave = np.unique(fwave)
            # fwave = np.sort(fwave) 

            resps_temp  = []
            for loop in group:
                im  = self.get_image(loop)
                resp_im     = im.resp
                resps_temp.append(resp_im)
            sub_wavegrid, sub_resps_new     = wavegrid_dlnlam(resps_temp)
            self.sub_wavegrids.append(sub_wavegrid) # maybe useless.
            self.subgroup_resps.append(sub_resps_new)


            # self.filterwaves.append(np.array(fwave, dtype=np.float32))
            # for loop in group:
            #     im = self.get_image(loop)
            #     filterx = im.resp[:,0]
            #     min_wave, max_wave = filterx[0], filterx[-1]
            #     idx_start = jnp.searchsorted(fwave, min_wave, side='left')
            #     idx_end = jnp.searchsorted(fwave, max_wave, side='right')
            #     im.respbound = [idx_start, idx_end]
        return
    def init_grid(self):
        resps   = []
        for loop, al in enumerate(self.image_atlas_list):
            for loop2 in range(len(al)):
                # read resp from image objects directly: image_atlas.resps is only
                # set by image_atlas.init_sed(), which runs later and only in sedmode.
                resps.append(al.image_list[loop2].resp)
        # self.resps = resps ## store resps_new, which have been resampled.
        wavegrid, resps_new     = wavegrid_dlnlam(resps, )
        self.wavegrid   = wavegrid ## this wavegrid is used for subsequent sample and integrate.
        self.resps  = resps_new

    
    def init_sed(self):
        atlasnamelist = []
        photos = []
        photoerrs = []
        resps = []
        for loop, al in enumerate(self.image_atlas_list):
            for loop2 in range(len(al)):
                atlasnamelist.append(self.image_atlas_names[loop])
                photos.append(al.photos[loop2])
                photoerrs.append(al.photoerrs[loop2])
                resps.append(al.resps[loop2])
        self.atlasnamelist = atlasnamelist
        self.photos = jnp.array(photos,dtype=jnp.float32)
        self.photoerrs = jnp.array(photoerrs,dtype=jnp.float32)


