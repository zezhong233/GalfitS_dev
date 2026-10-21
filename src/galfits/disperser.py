
import jax.numpy as jnp
from collections import OrderedDict
import numpy as np
import jax
import os
from astropy.io import fits
from astropy.table import Table
from pathlib import Path
jax.config.update("jax_enable_x64", False)

current_dir = os.path.dirname(os.path.abspath(__file__))
conffilepath = os.path.join(current_dir, '..', 'data', 'grism')
conffilepath = Path(conffilepath)

# grism image resolution
R_grism = {
    'HST_G102': 210,
    'HST_G141': 130,
    'CSST_GI': 1000,
    'CSST_GV': 1000,
    'CSST_GU': 1000,
    'JWST_F356W_R_A':1500,
    'JWST_F444W_R_B':1600
}

ccd_size = {'JWST_F356W_R_A': (2048, 2048),'JWST_F444W_R_B': (2048, 2048)}

def _interp1d_conserve_flux_trapz_sum(
    x, y, nx,
    ix, nix,
    x_bs, ix_bs):
    """
    Flux-conserving 1D interpolation using trapezoidal integration (returns integrated flux).
    
    This function performs flux-conserving interpolation from one wavelength grid to another
    by computing the integrated flux within each output bin. It uses the trapezoidal rule
    for numerical integration, ensuring total flux is conserved during resampling.
    
    Parameters
    ----------
    x : array_like, shape (nx,)
        Input wavelength/coordinate grid (must be monotonically increasing).
    y : array_like, shape (nx,)
        Input flux density values (flux per unit wavelength).
    nx : int
        Number of points in the input grid.
    ix : array_like, shape (nix,)
        Output wavelength/coordinate grid centers.
    nix : int
        Number of points in the output grid.
    x_bs : array_like, shape (nx+1,)
        Workspace array for input bin boundaries (will be computed internally).
    ix_bs : array_like, shape (nix+1,)
        Workspace array for output bin boundaries (will be computed internally).
    
    Returns
    -------
    iy : array_like, shape (nix,)
        Integrated flux in each output bin (not flux density).
        Total integrated flux: sum(iy) ≈ integral of y(x) over overlapping range.
    
    Notes
    -----
    The function:
    
    1. Computes bin boundaries by averaging adjacent grid points
    2. Uses trapezoidal rule to integrate y(x) within each output bin
    3. Handles edge cases where output bins extend beyond input range (flux=0)
    4. Returns integrated flux, not flux density
    
    This is suitable for resampling spectra when you want to preserve total counts
    or energy in each bin, such as for detector response modeling.
    
    The algorithm uses JAX control flow (lax.while_loop, lax.fori_loop) for
    JIT compilation and GPU acceleration.
    
    See Also
    --------
    _interp1d_conserve_flux_trapz_nosum : Returns flux density instead of integrated flux.
    _interp1d_conserve_flux_trapz_sum_rewritten : Alternative implementation without loops.
    """
    # Calculate boundaries for x and ix
    x_bs = x_bs.at[1:nx].set((x[:-1] + x[1:]) / 2)
    x_bs = x_bs.at[0].set(1.5 * x[0] - 0.5 * x[1])
    x_bs = x_bs.at[nx].set(1.5 * x[nx - 1] - 0.5 * x[nx - 2])

    nix_bs = nix + 1
    ix_bs = ix_bs.at[1:nix].set((ix[:-1] + ix[1:]) / 2)
    ix_bs = ix_bs.at[0].set(1.5 * ix[0] - 0.5 * ix[1])
    ix_bs = ix_bs.at[nix].set(1.5 * ix[nix - 1] - 0.5 * ix[nix - 2])

    # Initialize variables
    iy = jnp.zeros_like(ix)

    # Use JAX's while_loop to skip ix values below x[0]
    def skip_cond(state):
        j, _ix = state
        return (_ix <= x[0]) & (j < nix_bs - 1)

    def skip_body(state):
        j, _ix = state
        return j + 1, ix_bs[j + 1]

    j, _ix = jax.lax.while_loop(skip_cond, skip_body, (0, ix_bs[0]))

    # Check if all ix values are less than x[0]
    iy = jnp.where(_ix <= x[0], jnp.zeros_like(iy), iy)

    # Initialize iflux, _xleft, and _yleft
    iflux = 0.0
    _xleft = x[0]
    _yleft = y[0]

    def process_segment(i, carry):
        j, _ix, _xleft, _yleft, iflux, iy = carry
        _xright, _yright = x[i + 1], y[i + 1]

        def process_cond(state):
            j, _ix, _xleft, _yleft, iflux, iy = state
            return (_xright > _ix) & (j < nix_bs)

        def process_body(state):
            j, _ix, _xleft, _yleft, iflux, iy = state
            _ymid = _yleft + (_yright - _yleft) / (_xright - _xleft) * (_ix - _xleft)
            new_flux = iflux + (_ix - _xleft) * (_yleft + _ymid) / 2
            new_iy = iy.at[j - 1].set( new_flux)
            # Update variables
            _xleft = _ix
            _yleft = _ymid
            iflux = 0.0
            j += 1
            _ix = jnp.where(j < nix_bs, ix_bs[j], _ix)
            return j, _ix, _xleft, _yleft, iflux, new_iy

        j, _ix, _xleft, _yleft, iflux, iy = jax.lax.while_loop(
            process_cond, process_body, (j, _ix, _xleft, _yleft, iflux, iy)
        )


        # Update iflux and iy after the inner loop
        update_condition = (j < nix_bs)
        additional_flux = jnp.where(
            update_condition,
            (_xright - _xleft) * (_yleft + _yright) / 2,
            0.0
        )
        iflux += additional_flux
        iy = iy.at[j - 1].set(
            jnp.where(
                update_condition & (j > 0), iflux,
                iy[j - 1])
        )
        # Update _xleft and _yleft for the next segment
        _xleft = _xright
        _yleft = _yright

        return j, _ix, _xleft, _yleft, iflux, iy

    # Run the main loop
    j, _ix, _, _, _, iy = jax.lax.fori_loop(
        0, nx - 1, process_segment, (j, _ix, _xleft, _yleft, iflux, iy)
    )

    # Handle remaining elements
    def finalize_cond(state):
        j, iy = state
        return j < nix

    def finalize_body(state):
        j, iy = state
        return j + 1, iy.at[j].set(0)

    _, iy = jax.lax.while_loop(finalize_cond, finalize_body, (j, iy))

    return iy

def _interp1d_conserve_flux_trapz_nosum(
    x, y, nx,
    ix, nix,
    x_bs, ix_bs):
    """
    Flux-conserving 1D interpolation using trapezoidal integration (returns flux density).
    
    Similar to `_interp1d_conserve_flux_trapz_sum`, but returns flux density (flux per
    unit wavelength) rather than integrated flux. This is appropriate for resampling
    spectral energy distributions where you want to preserve the shape of the spectrum.
    
    Parameters
    ----------
    x : array_like, shape (nx,)
        Input wavelength/coordinate grid (must be monotonically increasing).
    y : array_like, shape (nx,)
        Input flux density values (flux per unit wavelength).
    nx : int
        Number of points in the input grid.
    ix : array_like, shape (nix,)
        Output wavelength/coordinate grid centers.
    nix : int
        Number of points in the output grid.
    x_bs : array_like, shape (nx+1,)
        Workspace array for input bin boundaries (computed internally).
    ix_bs : array_like, shape (nix+1,)
        Workspace array for output bin boundaries (computed internally).
    
    Returns
    -------
    iy : array_like, shape (nix,)
        Flux density at each output bin center (flux per unit wavelength).
        Computed as integrated flux divided by bin width.
    
    Notes
    -----
    The relationship between this function and `_interp1d_conserve_flux_trapz_sum`:
    
    .. math::
        y_{density}[i] = \\frac{\\int_{\\lambda_{i,left}}^{\\lambda_{i,right}} y(\\lambda) d\\lambda}{\\lambda_{i,right} - \\lambda_{i,left}}
    
    This preserves the spectral energy distribution shape while allowing different
    wavelength samplings. Total flux is conserved when comparing integrals over
    the same wavelength range.
    
    Uses JAX control flow for JIT compilation and GPU acceleration.
    
    See Also
    --------
    _interp1d_conserve_flux_trapz_sum : Returns integrated flux instead of density.
    _interp1d_conserve_flux_trapz_nosum_rewritten : Alternative implementation.
    """
    # Calculate boundaries for x and ix
    x_bs = x_bs.at[1:nx].set((x[:-1] + x[1:]) / 2)
    x_bs = x_bs.at[0].set(1.5 * x[0] - 0.5 * x[1])
    x_bs = x_bs.at[nx].set(1.5 * x[nx - 1] - 0.5 * x[nx - 2])

    nix_bs = nix + 1
    ix_bs = ix_bs.at[1:nix].set((ix[:-1] + ix[1:]) / 2)
    ix_bs = ix_bs.at[0].set(1.5 * ix[0] - 0.5 * ix[1])
    ix_bs = ix_bs.at[nix].set(1.5 * ix[nix - 1] - 0.5 * ix[nix - 2])

    # Initialize variables
    iy = jnp.zeros_like(ix)

    # Use JAX's while_loop to skip ix values below x[0]
    def skip_cond(state):
        j, _ix = state
        return (_ix <= x[0]) & (j < nix_bs - 1)

    def skip_body(state):
        j, _ix = state
        return j + 1, ix_bs[j + 1]

    j, _ix = jax.lax.while_loop(skip_cond, skip_body, (0, ix_bs[0]))

    # Check if all ix values are less than x[0]
    iy = jnp.where(_ix <= x[0], jnp.zeros_like(iy), iy)

    # Initialize iflux, _xleft, and _yleft
    iflux = 0.0
    _xleft = x[0]
    _yleft = y[0]

    def process_segment(i, carry):
        j, _ix, _xleft, _yleft, iflux, iy = carry
        _xright, _yright = x[i + 1], y[i + 1]

        def process_cond(state):
            j, _ix, _xleft, _yleft, iflux, iy = state
            return (_xright > _ix) & (j < nix_bs)

        def process_body(state):
            j, _ix, _xleft, _yleft, iflux, iy = state
            _ymid = _yleft + (_yright - _yleft) / (_xright - _xleft) * (_ix - _xleft)
            new_flux = iflux + (_ix - _xleft) * (_yleft + _ymid) / 2
            new_iy = iy.at[j - 1].set(new_flux / (ix_bs[j] - ix_bs[j - 1]))
            # Update variables
            _xleft = _ix
            _yleft = _ymid
            iflux = 0.0
            j += 1
            _ix = jnp.where(j < nix_bs, ix_bs[j], _ix)
            return j, _ix, _xleft, _yleft, iflux, new_iy

        j, _ix, _xleft, _yleft, iflux, iy = jax.lax.while_loop(
            process_cond, process_body, (j, _ix, _xleft, _yleft, iflux, iy)
        )


        # Update iflux and iy after the inner loop
        update_condition = (j < nix_bs)
        additional_flux = jnp.where(
            update_condition,
            (_xright - _xleft) * (_yleft + _yright) / 2,
            0.0
        )
        iflux += additional_flux
        iy = iy.at[j - 1].set(
            jnp.where(
                update_condition & (j > 0), iflux / (ix_bs[j] - ix_bs[j - 1]),
                iy[j - 1])
        )
        # Update _xleft and _yleft for the next segment
        _xleft = _xright
        _yleft = _yright

        return j, _ix, _xleft, _yleft, iflux, iy

    # Run the main loop
    j, _ix, _, _, _, iy = jax.lax.fori_loop(
        0, nx - 1, process_segment, (j, _ix, _xleft, _yleft, iflux, iy)
    )

    # Handle remaining elements
    def finalize_cond(state):
        j, iy = state
        return j < nix

    def finalize_body(state):
        j, iy = state
        return j + 1, iy.at[j].set(0)

    _, iy = jax.lax.while_loop(finalize_cond, finalize_body, (j, iy))


    return iy


def _interp1d_conserve_flux_trapz_nosum_rewritten(x, y, nx, ix, nix, x_bs, ix_bs):
    """
    Flux-conserving interpolation using trapezoidal rule, rewritten without dynamic loops.
    Args:
        x: Input grid points (sorted, shape: [nx])
        y: Input values (shape: [nx])
        nx: Number of x points
        ix: Output grid points (sorted, shape: [nix])
        nix: Number of ix points
        x_bs: Unused boundary array for x (shape: [nx + 1])
        ix_bs: Unused boundary array for ix (shape: [nix + 1])
    Returns:
        iy: Interpolated values (shape: [nix])
    """
    # Compute ix_bs inside (boundaries for output bins)
    ix_bs = jnp.zeros(nix + 1)
    ix_bs = ix_bs.at[1:nix].set((ix[:-1] + ix[1:]) / 2)  # Interior midpoints
    ix_bs = ix_bs.at[0].set(1.5 * ix[0] - 0.5 * ix[1])    # Left extrapolation
    ix_bs = ix_bs.at[nix].set(1.5 * ix[nix - 1] - 0.5 * ix[nix - 2])  # Right extrapolation

    # Cumulative integral at x points from x[0] to x[i]
    cum_int_x = jnp.cumsum(jnp.concatenate([
        jnp.array([0.0]),  # Starting at zero
        (x[1:] - x[:-1]) * (y[:-1] + y[1:]) / 2  # Trapezoidal rule segments
    ]))

    # For each ix_bs[k], find the segment it falls into
    i = jnp.searchsorted(x, ix_bs, side='left') - 1  # i such that x[i] <= ix_bs[k] < x[i+1]

    # Extract values at segment boundaries, clipping indices to avoid bounds errors
    x_i = jnp.take(x, i, mode='clip')
    y_i = jnp.take(y, i, mode='clip')
    x_ip1 = jnp.take(x, i + 1, mode='clip')
    y_ip1 = jnp.take(y, i + 1, mode='clip')
    cum_int_i = jnp.take(cum_int_x, i, mode='clip')

    # Interpolate y at ix_bs[k] within the segment [x[i], x[i+1]]
    delta_x = x_ip1 - x_i  # Assume x is strictly increasing; delta_x > 0 for valid i
    f_ix = y_i + (y_ip1 - y_i) / delta_x * (ix_bs - x_i)  # Linear interpolation

    # Integral from x[i] to ix_bs[k]
    integral_add = (ix_bs - x_i) * (y_i + f_ix) / 2

    # Total cumulative integral up to ix_bs[k]
    cum_int_intermediate = cum_int_i + integral_add
    cum_int = jnp.where(i < 0, 0.0,  # Before x[0], integral is 0
                        jnp.where(i >= nx - 1, cum_int_x[nx - 1],  # At or beyond x[nx-1]
                                  cum_int_intermediate))    # Within range

    # Compute iy as flux per bin, normalized by bin width
    iy = (cum_int[1:] - cum_int[:-1]) / (ix_bs[1:] - ix_bs[:-1])

    return iy


def _interp1d_conserve_flux_trapz_sum_rewritten(x, y, nx, ix, nix, x_bs, ix_bs):
    """
    Flux-conserving interpolation using trapezoidal rule, rewritten without dynamic loops.
    Computes the integral over each bin [ix_bs[k], ix_bs[k+1]] ∩ [x[0], x[nx-1]].
    
    Args:
        x: Input grid points (sorted, shape: [nx])
        y: Input values (shape: [nx])
        nx: Number of x points
        ix: Output grid points (sorted, shape: [nix])
        nix: Number of ix points
        x_bs: Unused boundary array for x (shape: [nx + 1])
        ix_bs: Unused boundary array for ix (shape: [nix + 1])
    
    Returns:
        iy: Integrated values over each bin (shape: [nix])
    """
    # Compute ix_bs boundaries
    ix_bs = jnp.zeros(nix + 1)
    ix_bs = ix_bs.at[1:nix].set((ix[:-1] + ix[1:]) / 2)
    ix_bs = ix_bs.at[0].set(1.5 * ix[0] - 0.5 * ix[1])
    ix_bs = ix_bs.at[nix].set(1.5 * ix[nix - 1] - 0.5 * ix[nix - 2])

    # Compute cumulative integral at x points
    cum_int_x = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), (x[1:] - x[:-1]) * (y[:-1] + y[1:]) / 2]))

    # Find segment indices for ix_bs
    i = jnp.searchsorted(x, ix_bs, side='left') - 1

    # Define masks for different regions
    mask_before = i < 0  # ix_bs < x[0]
    mask_after = i > nx - 2  # ix_bs >= x[nx-1]
    mask_within = ~mask_before & ~mask_after  # x[0] <= ix_bs < x[nx-1]

    # Compute cumulative integral components within x range
    i_clipped = jnp.clip(i, 0, nx - 2)
    x_i = x[i_clipped]
    y_i = y[i_clipped]
    x_ip1 = x[i_clipped + 1]
    y_ip1 = y[i_clipped + 1]
    delta_x = x_ip1 - x_i
    slope = (y_ip1 - y_i) / delta_x
    t = ix_bs - x_i
    y_at_t = y_i + slope * t
    integral_add = t * (y_i + y_at_t) / 2

    # Compute cumulative integral at ix_bs
    cum_int_within = cum_int_x[i] + integral_add
    cum_int = jnp.where(mask_before, 0.0,
                        jnp.where(mask_after, cum_int_x[nx - 1],
                                  cum_int_within))

    # Compute output as differences of cumulative integrals
    iy = jnp.abs(cum_int[1:] - cum_int[:-1])

    return iy


def dispersion_ifu_with_flux_jax(
    dymin,
    nxs, nys, nxd,
    seg,
    dy3d, lam3d,
    ifu_wave, ifu_flux, nwave,
    t_area_factor,
    sens_wave, sens_sens, nsens,
    gsimage,
    convert_to_photo,
):
    """
    JAX-accelerated dispersion of IFU data cube to create grism image.
    
    This function performs the core dispersion calculation, taking an input IFU
    (Integral Field Unit) data cube with wavelength-dependent flux and dispersing
    it onto a 2D detector image according to the instrument's wavelength-dependent
    trace and sensitivity.
    
    Parameters
    ----------
    dymin : int
        Minimum y-offset for the dispersed spectrum.
    nxs, nys : int
        Dimensions of the input IFU spatial grid.
    nxd : int
        Number of wavelength pixels in the dispersion direction.
    seg : array_like
        Segmentation map (not currently used, reserved for future).
    dy3d : array_like, shape (nys, nxs, nxd)
        3D array of y-trace offsets for each spatial pixel and wavelength.
    lam3d : array_like, shape (nys, nxs, nxd)
        3D array of wavelengths for each spatial pixel and dispersion pixel.
    ifu_wave : array_like
        Wavelength grid of the input IFU cube (Angstroms).
    ifu_flux : array_like, shape (nys, nxs, nwave)
        Input IFU flux cube in erg/s/cm²/Å or similar units.
    nwave : int
        Number of wavelength points in the IFU cube.
    t_area_factor : float
        Product of exposure time and telescope area for flux-to-counts conversion.
    sens_wave : array_like
        Wavelength grid for the sensitivity curve (Angstroms).
    sens_sens : array_like
        Sensitivity curve values (throughput).
    nsens : int
        Number of points in the sensitivity curve.
    gsimage : array_like
        Output dispersed image array to be updated.
    convert_to_photo : int
        If >0, convert flux to photoelectrons.
    
    Returns
    -------
    gsimage : array_like
        Updated dispersed image with flux from all IFU pixels added.
    
    Notes
    -----
    The dispersion process:
    
    1. For each spatial pixel in the IFU:
       
       a. Resample the input spectrum onto the instrument's wavelength grid
       b. Apply the sensitivity curve
       c. Convert to detector counts (if requested)
       d. Distribute flux along the wavelength-dependent trace
    
    2. Use linear interpolation between integer pixel positions
    
    The function is JAX-accelerated and can run on GPU for significant speedup
    on large data cubes.
    
    See Also
    --------
    GSDisperser._disperse_ifu_with_flux : Wrapper function that calls this dispatcher.
    """
    unit_factor = 503411656754.271
    #print(t_area_factor)
    def process_pixel(ix, iy):
        # Convert to photo-electrons if required
        iflux_raw = jnp.where(
            convert_to_photo > 0,
            ifu_flux[iy, ix] * ifu_wave * unit_factor,
            ifu_flux[iy, ix]
        )

        # Get wavelength array for this pixel
        lam = lam3d[iy, ix]

        # Allocate boundary arrays
        x_bs_wave = jnp.zeros(nwave + 1)
        x_bs_sens = jnp.zeros(nsens + 1)
        ix_bs = jnp.zeros(nxd + 1)

        # Interpolate flux and sensitivity
        iflux = _interp1d_conserve_flux_trapz_nosum_rewritten(
            ifu_wave, iflux_raw, nwave,
            lam, nxd, x_bs_wave, ix_bs
        )
        isens = _interp1d_conserve_flux_trapz_sum_rewritten(
            sens_wave, sens_sens, nsens,
            lam, nxd, x_bs_sens, ix_bs
        )
        # Calculate electrons and positions
        electrons = isens * iflux * t_area_factor
        y_positions = dy3d[iy, ix] - dymin + iy
        x_positions = jnp.arange(nxd) + ix

        def process_lambda(ilam):
            _x = x_positions[ilam]
            _y = y_positions[ilam]
            _electrons = electrons[ilam]

            # Compute floor and ceil
            _floor = jnp.floor(_y).astype(int)
            _ceil = jnp.ceil(_y).astype(int)

            # Compute fractions
            ceil_fraction = jnp.where(_floor == _ceil, 1.0, 1 - (_ceil - _y))
            floor_fraction = jnp.where(_floor == _ceil, 0.0, 1 - (_y - _floor))

            # Return updates for floor and ceil pixels
            return jnp.stack([
                jnp.array([_floor, _x, floor_fraction * _electrons]),
                jnp.array([_ceil, _x, ceil_fraction * _electrons])
            ])

        # Vectorize process_lambda over all wavelength steps
        updates = jax.vmap(process_lambda)(jnp.arange(nxd))  # Shape: (nxd, 2, 3)
        updates = updates.reshape(-1, 3)  # Shape: (nxd * 2, 3)

        return updates

    # Create a grid of all (ix, iy) pixel indices
    ix_indices = jnp.arange(nxs)
    iy_indices = jnp.arange(nys)
    ix_grid, iy_grid = jnp.meshgrid(ix_indices, iy_indices, indexing='ij')  # Shape: (nxs, nys)

    # Flatten the grid to obtain lists of ix and iy
    ix_flat = ix_grid.flatten()
    iy_flat = iy_grid.flatten()

    # Vectorize process_pixel over all pixels using vmap
    all_updates = jax.vmap(process_pixel)(ix_flat, iy_flat)  # Shape: (nxs * nys, nxd * 2, 3)
    all_updates = all_updates.reshape(-1, 3)  # Shape: (nxs * nys * nxd * 2, 3)

    # Extract row, column indices and corresponding values
    row_indices = all_updates[:, 0].astype(int)
    col_indices = all_updates[:, 1].astype(int)
    values = all_updates[:, 2]

    # Apply scatter_add to update the gsimage
    gsimage = gsimage.at[row_indices, col_indices].add(values)

    return gsimage

class GSDisperser(object):
    """
    Grism/Slitless spectroscopy disperser for IFU data cubes.
    
    This class handles the forward modeling of grism/prism observations by dispersing
    input IFU (Integral Field Unit) data cubes according to instrument-specific
    dispersion relations and sensitivity curves. It accounts for wavelength-dependent
    PSF, spectral resolution, and detector response.
    
    Parameters
    ----------
    axeconf : aXeConfBase
        Configuration object containing instrument dispersion and sensitivity information.
    nx : int
        Total width of the output dispersed image in pixels.
    ny : int
        Total height of the output dispersed image in pixels.
    ixmin : int
        Minimum x-coordinate of the input IFU cutout.
    iymin : int
        Minimum y-coordinate of the input IFU cutout.
    nxs : int
        Width of the input IFU cutout in pixels.
    nys : int
        Height of the input IFU cutout in pixels.
    beam : str, optional
        Spectral order/beam to model (e.g., 'A', 'B', 'C'). Default is 'A'.
    bin : int, optional
        Binning factor for dispersion calculation (for speed). Default is 1.
    convert_to_photos : int, optional
        Whether to convert flux to photoelectrons (1) or keep in flux units (0).
        Default is 0.
    t_area_factor : float, optional
        Product of exposure time and telescope area for flux-to-counts conversion.
        Default is 1.
    
    Attributes
    ----------
    nx, ny : int
        Output image dimensions.
    axeconf : aXeConfBase
        Instrument configuration object.
    sens_wave : array_like
        Wavelength grid for sensitivity curve (Angstroms).
    sens_sens : array_like
        Sensitivity values in erg/s/cm²/Å per detected count.
    wave_grid : array_like
        Wavelength grid for spectral rebinning based on instrument resolution.
    dy3d, lam3d : array_like
        3D arrays of trace offsets and wavelengths for each pixel.
    nxd, nyd : int
        Dimensions of the dispersed spectrum.
    dymin, dxmin : int
        Minimum offsets for dispersion geometry.
    
    Notes
    -----
    The disperser uses JAX for GPU-accelerated dispersion calculation. It performs:
    
    1. Wavelength-dependent trace calculation
    2. Flux-conserving spectral rebinning to detector resolution
    3. Sensitivity curve application
    4. Conversion to detector counts (if requested)
    
    The class assumes that each spatial pixel in the input IFU can have a different SED,
    enabling modeling of spatially resolved spectroscopy.
    
    Examples
    --------
    >>> from galfits.disperser import GSDisperser, aXeConfBase
    >>> conf = aXeConfBase('HST_G141')
    >>> disperser = GSDisperser(conf, nx=100, ny=100, ixmin=0, iymin=0,
    ...                          nxs=10, nys=10, beam='A')
    >>> wavelength = np.arange(8000, 17000, 10)  # Angstroms
    >>> ifu_cube = np.ones((10, 10, len(wavelength)))  # Flat spectrum
    >>> dispersed_image = disperser.get_spec_orders(wavelength, ifu_cube)
    
    See Also
    --------
    aXeConfBase : Configuration parser for aXe-format instrument files.
    """
    def __init__(self, axeconf, nx,ny, ixmin, iymin, nxs, nys, beam='A', bin=1, convert_to_photos = 0, t_area_factor = 1):
    
        self.nx = nx
        self.ny = ny
        if isinstance(axeconf, aXeConfBase):
            self.axeconf_file = axeconf.conf_file
            self.axeconf = axeconf
        else:
            raise Exception('axeconf should be aXeConfBase instance')

        sens = self.axeconf.get_beam_sens(beam)
        self.sens_sens = jnp.array(sens['SENSITIVITY']*1e-17, dtype = jnp.float32)
        self.sens_wave =  jnp.array(sens['WAVELENGTH'], dtype = jnp.float32)

        npwave = np.array(sens['WAVELENGTH'])
        npsens = np.array(sens['SENSITIVITY'])
        wmin = np.min(npwave[npsens>0.005*max(npsens)])
        wmax = np.max(npwave[npsens>0.005*max(npsens)])
        r_grism = R_grism[self.axeconf.bandname]
        dlambda = 0.75*wmin/r_grism
        wave_grid = np.arange(wmin,wmax,dlambda)
        self.wave_grid = jnp.array(wave_grid, dtype=jnp.float32)

        self._dxdylam_cache = None
        self.convert_to_photos = jnp.array(convert_to_photos,dtype=jnp.float32)
        self.t_area_factor = jnp.array(t_area_factor,dtype=jnp.float32)

        self._get_cached_dxdylam(ixmin, iymin, nxs, nys, beam=beam, bin=bin)
    

    def _get_cached_dxdylam(self,  ixmin, iymin, nxs, nys, beam='A', bin=1):
        if self._dxdylam_cache is None:
            
            dx, dy, lam = self.axeconf.get_dxdylam_3d(
              xmin=ixmin,
              ymin=iymin,
              nx=nxs,ny=nys,
              beam=beam,
              bin=bin,
              force_x_int=False,
            )
            dx = jnp.int16(dx)

            dymin = int(jnp.floor(dy.min()))
            dymax = int(jnp.ceil(dy.max()))
            nxd = dx.shape[2]
            nyd = dymax - dymin + 1
            dxmin = dx.min()

            self.dy3d = jnp.array(dy, dtype=jnp.int16)
            self.lam3d = jnp.array(lam, dtype=jnp.float32)
            self.nxd = nxd
            self.nyd = nyd
            self.nxs = nxs
            self.nys = nys
            self.dymin = dymin
            self.dxmin = dxmin
            self.igxmin = ixmin + dxmin
            self.igymin = iymin + dymin

            self._dxdylam_cache = dx, dy, lam, nxd, nyd, dymin, dymax, dxmin, self.sens_wave, self.sens_sens
        
        return self._dxdylam_cache


    def _disperse_ifu_with_flux(self, wave, ifu_flux):
        '''
          just to pack these operations and make the structure of 
            IFUDisperser.get_spec_orders_with_fixed_image 
              the same as 
            FixedImageDisperser.get_spec_orders
        '''

        
        nxg = self.nxd + self.nxs
        nyg = self.nyd + self.nys
        gsimage = jnp.zeros((nyg, nxg), dtype=jnp.float32)
        seg = jnp.ones((self.nys, self.nxs), dtype=jnp.int16)

        gsimage = dispersion_ifu_with_flux_jax(
            self.dymin,
            self.nxs, self.nys, self.nxd,
            seg,
            self.dy3d, self.lam3d,
            wave, ifu_flux, wave.size,
            self.t_area_factor,
            self.sens_wave, self.sens_sens, self.sens_wave.size,
            gsimage,
            self.convert_to_photos,
        )

        result = gsimage
          
        return result
    
    def get_spec_orders(self, wave, ifu):
        beamimg = self._disperse_ifu_with_flux(wave=wave, ifu_flux=ifu)
        #nys, nxs, nwave = ifu.shape

        return beamimg

class aXeConfBase(object):
    """
    Configuration parser for aXe-format grism/prism instrument files.
    
    This class reads and parses aXe-style configuration files that describe the
    dispersion solution, trace geometry, and sensitivity curves for slitless
    spectroscopy instruments. It provides methods to compute wavelength solutions
    and trace positions for any location in the detector field of view.
    
    Parameters
    ----------
    band : str, optional
        Instrument/mode identifier (e.g., 'HST_G102', 'HST_G141', 'CSST_GI').
        Default is 'HST_G102'.
    
    Attributes
    ----------
    conf : OrderedDict
        Parsed configuration parameters from the .conf file.
    conf_file : str
        Path to the configuration file.
    bandname : str
        Instrument/mode name.
    instrument : str
        Instrument identifier.
    beams : list
        List of available spectral orders/beams.
    dydx_orders, dldp_orders : dict
        Polynomial orders for trace and dispersion solutions per beam.
    dydx_corders, dldp_corders : dict
        Cross-term polynomial orders per beam.
    dydx_params, dldp_params : dict
        Polynomial coefficients for trace and dispersion per beam.
    xoff, yoff : float
        Global coordinate offsets.
    
    Notes
    -----
    The aXe configuration format defines:
    
    - Trace solution: dy/dx as a function of position and wavelength offset
    - Dispersion solution: λ as a function of position and pixel offset
    - Sensitivity curves: Instrument throughput vs wavelength
    - Beam geometry: Valid wavelength ranges and pixel offsets
    
    The polynomial solutions typically have the form:
    
    .. math::
        f(x, y, \\Delta x) = \\sum_{i=0}^{N} \\left(\\sum_{j=0}^{M_i} c_{ij} x^{(M_i-j)} y^j\\right) (\\Delta x)^i
    
    where N is the order and M_i is the cross-term order.
    
    Examples
    --------
    >>> from galfits.disperser import aXeConfBase
    >>> conf = aXeConfBase('HST_G141')
    >>> print(conf.beams)
    ['A', 'B', 'C', 'D', 'E']
    >>> sens = conf.get_beam_sens('A')
    >>> dx, dy, lam = conf.get_dxdylam(x=512, y=512, beam='A')
    
    See Also
    --------
    GSDisperser : Uses aXeConfBase for forward modeling of dispersed spectra.
    
    References
    ----------
    .. [1] aXe Software: https://www.stsci.edu/hst/instrumentation/acs/data-analysis/slitless-spectroscopy
    .. [2] Kümmel et al. 2009, PASP, 121, 59
    """

    def __init__(self, band='HST_G102'):
        conf_file = conffilepath / (band + '.conf')
        conf_file = str(conf_file)
        self.conf = self.read_conf_file(conf_file)
        self.conf_file = conf_file
        self.bandname = band
        self.instrument = self.conf['INSTRUMENT']
        self.ccd_sx = ccd_size[self.bandname][0]
        self.ccd_sy = ccd_size[self.bandname][1]
        self._update_cls_by_config()

    def get_beam_sens(self, beam=None):
        sens_file = self.conf[f'SENSITIVITY_{beam}']
        conf_dir = Path(os.path.dirname(self.conf_file))
        sens_file = str(conf_dir/sens_file)
        
        data = fits.open(sens_file, memmap=False)
        data = Table.read(data[1])
        
        data['SENSITIVITY'] = data['SENSITIVITY'].astype(np.float32)
        data['WAVELENGTH']  = data['WAVELENGTH'].astype(np.float32)
        return data

    def read_conf_file(self, conf_file):

        conf = OrderedDict()
        with open(conf_file) as f:
            lines = f.readlines()
        for line in lines:
            ## empty / commented lines
            if (line.startswith('#')) | (line.strip() == '') | ('"' in line):
                continue

            ## split the line, taking out ; and # comments
            spl = line.split(';')[0].split('#')[0].split()
            param = spl[0]
            if len(spl) > 2:
                value = np.asarray(spl[1:],dtype=np.float32)
            else:
                try:
                    value = float(spl[1])
                except:
                    value = spl[1]
            conf[param] = value
        return conf
    
    def _update_cls_by_config(self):

        self.band = self.conf
        if 'XOFF' in self.conf.keys():
            self.xoff = np.float(self.conf['XOFF'])
        else:
            self.xoff = 0.

        if 'YOFF' in self.conf.keys():
            self.yoff = np.float(self.conf['YOFF'])
        else:
            self.yoff = 0.

        beams = []
        # in HST G141 band, the beam is A~G, but the config do not have DISP_ORDER_G parameter
        # for both CSST and HST, we just calcualte up to E beam. We do not calculate F and G beam
        for beam in ['A', 'B', 'C', 'D', 'E', ]:
            if f'BEAM{beam}' in self.conf:
                beams.append(beam)
        self.beams = beams

        # update orders
        # \Deltax and \Delta l
        self.dydx_orders  = {}
        self.dldp_orders  = {}
        for todo in ['dydx', 'dldp']:
            attr = getattr(self, f'{todo}_orders')
            for beam in self.beams:
                attr[beam] = self._compute_orders(beam, todo.upper())

        # update corders
        # (x_0, y_0)
        self.dydx_corders  = {}
        self.dldp_corders  = {}
        for todo in ['dydx', 'dldp']:
            attr = getattr(self, f'{todo}_corders')
            orders = getattr(self, f'{todo}_orders')
            for beam in self.beams:
                order = orders[beam]
                attr[beam] = self._compute_corders(beam, todo.upper(), order)

        # update params
        # length is (corder+1)*(corder+2) / 2 * (order+1)
        self._update_params()

    def _compute_orders(self, beam, type):
        order = 0
        while f'{type}_{beam}_{order}' in self.conf.keys():
            order += 1
        if order>0:
            if type == 'DLDP':
                assert self.conf[f'DISP_ORDER_{beam}'] == order - 1
            else:
                assert self.conf[f'{type}_ORDER_{beam}'] == order - 1
            return order - 1
        else:
            raise Exception(f'beam {beam} not exist')
  
    def _compute_corders(self, beam, type, order):
    
        orders = []
        for i in range(order+1):
            key = f'{type}_{beam}_{i}'
            if key not in self.conf:
                raise Exception(f'coefficient {key} not in conf')
            coeffs = self.conf[key]
            ## number of coefficients for a given polynomial order
            ## 1:1, 2:3, 3:6, 4:10, order:order*(order+1)/2
            o = int(-1 + np.sqrt(1 + 8 * len(coeffs))) // 2 - 1
            orders.append(o)
        return np.max(orders)
  

    def _update_params(self):
        '''
        update parameters by self.conf
        '''
        self.dydx_params = {}
        self.dldp_params = {}
        for todo in ['dydx', 'dldp']:
            params = getattr(self, f'{todo}_params')
            attr   = getattr(self, f'{todo}_orders')
            cattr  = getattr(self, f'{todo}_corders')
            for beam in self.beams:
                order  = attr[beam]
                corder = cattr[beam]
                N = int((corder+1)*(corder+2) / 2)
                size = (order+1) * N
                coeffs = []
                for o in range(order+1):
                    key = f'{todo.upper()}_{beam}_{o}'
                    if key in self.conf:
                        coeff = self.conf[key]
                        if len(coeff) != N:
                            coeff = np.concatenate([coeff, np.zeros(N-len(coeff))])
                    else:
                        coeff = np.zeros(N)
                    coeffs.extend(coeff)
                assert len(coeffs) == size
                params[beam] = np.array(coeffs, dtype=np.float64)
  

    def get_dxdylam_3d(self, xmin, ymin,nx, ny, beam, force_x_int=False,dydx_params=None,dldp_params=None,dx=None,bin=1):
        '''
        to speed up dispersion,
            the output 3d results MUST be in shape of 
            (ny, nx, nlam)
        '''

        if dydx_params is None:
            dydx_params = self.dydx_params[beam]
            #dydx_order    = self.dydx_orders[beam]
            #dydx_corder   = self.dydx_corders[beam]
        if dldp_params is None:
            dldp_params = self.dldp_params[beam]
            #dldp_order    = self.dldp_orders[beam]
            #dldp_corder   = self.dldp_corders[beam]

        #xoff_beam = self.conf[f'XOFF_{beam}']
        #yoff_beam = self.conf[f'YOFF_{beam}']
        dxmin, dxmax = self.conf[f'BEAM{beam}']
        ndx =  int(dxmax - dxmin)

        dx3d  = np.zeros((ny, nx, ndx), dtype=np.float32)
        dy3d  = np.zeros((ny, nx, ndx), dtype=np.float32)
        lam3d = np.zeros((ny, nx, ndx), dtype=np.float32)


        sxx = np.arange(xmin, xmin+nx, bin)
        syy = np.arange(ymin, ymin+ny, bin)
        snx = int(np.ceil(nx/bin))
        sny = int(np.ceil(ny/bin))
        assert snx == sxx.size
        assert sny == syy.size
        _i = 0
        _j = 0
        for j in range(sny):
            for i in range(snx):
                _x = xmin + i*bin + (bin-1)/2.
                _y = ymin + j*bin + (bin-1)/2.
                dx, dy, lam = self.get_dxdylam(
                    x=_x, y=_y, beam=beam, force_x_int=force_x_int,
                    dydx_params=dydx_params,
                    dldp_params=dldp_params)
                assert ndx == dx.size
                if bin == 1:
                    dx3d[ j, i, :]  = dx[:]
                    dy3d[ j, i, :]  = dy[:]
                    lam3d[j, i, :] = lam[:]
                else: # bin >= 2
                    for _j in range(j*bin, j*bin+bin):
                        if _j>=ny: continue
                        for _i in range(i*bin, i*bin+bin):
                            if _i>=nx: continue
                            dx3d[ _j, _i, :]  = dx[:]
                            dy3d[ _j, _i, :]  = dy[:]
                            lam3d[_j, _i, :] = lam[:]        
        return dx3d, dy3d, lam3d
        

    def get_dxdylam(self, x, y, beam, force_x_int=False, dydx_params=None, dldp_params=None, dx=None):
   
        xoff_beam = self.conf[f'XOFF_{beam}']
        yoff_beam = self.conf[f'YOFF_{beam}']

        delta_dx = 0.
        if self.bandname == 'HST_G141' and beam == 'A':
            delta_dx = -0.5 # necessary for WFC3/IR G141, v4.32

        
        x, y = x - self.xoff, y - self.yoff # global offsets
        if dydx_params is None:
            dydx_params = self.dydx_params[beam]
        order       = self.dydx_orders[beam]
        corder      = self.dydx_corders[beam]
        dx, dy = self.get_dxdy(x=x, y=y, beam=beam, force_x_int=force_x_int, dydx_params=dydx_params, dx=dx)

        # dp
        dydx = self._get_dx_coeffects(order, corder, x, y, dydx_params)
        dp = self._get_dp(dx-xoff_beam, dydx)


        # wave
        if dldp_params is None:
            dldp_params = self.dldp_params[beam]
        order       = self.dldp_orders[beam]
        corder      = self.dldp_corders[beam]
        N = int((corder + 1) * (corder + 2) / 2)
        m = dp.shape[0]
        size = N * (order + 1) * m
        dldpa = self._get_A(order, corder, x, y, dp)
        dldpA = dldpa.reshape((m, N * (order+1)))
        #lam = (np.matrix(dldpA) * np.matrix(dldp_params).T).A.flatten()
        lam = (dldpA @ np.array([dldp_params]).T).flatten()

        dx = dx - delta_dx # for G141 special

        return dx, dy, lam
   

    def get_dxdy(self, *, x, y, beam, force_x_int=False, dydx_params=None, dx=None):
    
        xmin, xmax = self.conf[f'BEAM{beam}']
        # if dx is auto generated, the length of it is `BEAM{X}[1] - BEAM{X}[0] + 1`
        if dx is None:
        # len is xmax - xmin, left close right open
        # same behavior as HST aXeConf
            if force_x_int:
                _delta = int(x) - x
                dx = np.arange(xmin+_delta, xmax+_delta)
            else:
                dx = np.arange(xmin, xmax)

        delta_dx = 0.
        if self.bandname == 'HST_G141' and beam == 'A':
            delta_dx = -0.5 # necessary for WFC3/IR G141, v4.32
        dx = dx + delta_dx

        # beam offsets
        # currently, all ?OFF_{beam} in the config file is 0
        xoff_beam = self.conf[f'XOFF_{beam}']
        yoff_beam = self.conf[f'YOFF_{beam}']

        # trace
        if dydx_params is None:
            dydx_params = self.dydx_params[beam]
        order       = self.dydx_orders[beam]
        corder      = self.dydx_corders[beam]
        N = int((corder + 1) * (corder + 2) / 2)
        m = dx.shape[0]
        size = N * (order + 1) * m
        dydxa = self._get_A(order, corder, x, y, dx-xoff_beam)
        dydxA = dydxa.reshape((m, N * (order+1)))
        #dy = (np.matrix(dydxA) * np.matrix(dydx_params).T).A.flatten()
        dy = (dydxA @ np.array([dydx_params]).T).flatten()
        dy += yoff_beam

        return dx, dy
  

    def _get_A(self, order, corder, x, y, dx):   
        N = int((corder + 1) * (corder + 2) / 2)
        m = dx.shape[0]
        size = N * (order + 1) * m
        D = np.zeros(N, dtype=np.float64)
        Dx = np.zeros(order+1)
        A = np.zeros(size, dtype=np.float64)

        index = 0
        for i in range(corder+1):
            for j in range(i+1):
                value = x**(i-j) * y**j
                D[index] = value
                index += 1

        index = 0
        for i in range(m):
            deltax = dx[i]
            DELTAX = 1.
            for j in range(order+1):
                for k in range(N):
                    A[index] = DELTAX * D[k]
                    index += 1
                DELTAX *= deltax
        return A
    

    def _get_dx_coeffects(self, order, corder, x, y, params):
        N = int((corder + 1) * (corder + 2) / 2)
        Dx = np.zeros(order+1, dtype=np.float64)
        D = np.zeros(N, dtype=np.float64)

        index = 0
        for i in range(corder+1):
            for j in range(i+1):
                value = x**(i-j) * y**j
                D[index] = value
                index += 1

        I = 0
        for i in range(order+1):
            value = 0
            for j in range(N):
                value += D[j] * params[j+I]
            Dx[i] = value
            I += N
        return Dx
    
    def _get_dp(self, dx, dydx):
        order = len(dydx) - 1

        if order == 0:  ## dy=0
            dp = dx
        elif order == 1:  ## constant dy/dx
            dp = np.sqrt(1 + dydx[1] ** 2) * (dx)
        elif order == 2:  ## quadratic trace, copy from Grizli, NOT TESTED!
            u0 = dydx[1] + 2 * dydx[2] * (0)
            dp0 = (u0 * np.sqrt(1 + u0 ** 2) + np.arcsinh(u0)) / (4 * dydx[2])
            u = dydx[1] + 2 * dydx[2] * (dx)
            dp = (u * np.sqrt(1 + u ** 2) + np.arcsinh(u)) / (4 * dydx[2]) - dp0
        else:
            raise NotImplementedError('higher order not supported')
        return dp