"""
Structural Profile Functions Module
====================================

This module provides implementations of various 2D surface brightness profiles
used in galaxy modeling, including Sersic, exponential, Ferrer, edge-on disks,
Gaussian rings, and Fourier-mode perturbations for spiral structure.

All functions are JAX-compatible for automatic differentiation and GPU acceleration.

Supported Profiles
------------------
- **Sersic**: Generalized Sersic profile with arbitrary index n
- **Ferrer**: Truncated profile for bars and rings
- **Edge-on disk**: Vertical structure for edge-on galaxies
- **Gaussian ring**: Ring-like structures (resonance rings)
- **Fourier modes**: Spiral arms and non-axisymmetric features
- **Broken/Ring variants**: Truncated and ring-like modifications
- **PSF**: Point spread function modeling

Key Functions
-------------
- sersic2D : 2D Sersic profile
- Ferrer : Ferrer (truncated) profile
- EdgeonDisk : Edge-on disk with vertical structure
- GaussianRing : Gaussian ring profile
- sersic2D_fourier : Sersic + Fourier spiral modes
- R_3d : 3D deprojected radius calculation
- R_fourier : Fourier-mode perturbations

Geometric Utilities
-------------------
- R_2d, R_3d : 2D and 3D radius calculations
- xy_3d : Deproject coordinates accounting for inclination
- xpyp : Rotate coordinates by position angle
- alpha_tanh, hypotan : Smooth transition functions

All angles are in radians internally. Profiles return normalized surface brightness
or can be scaled by total flux/mass normalization.
"""

from . import Constant as C
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jax.lax import clamp
from functools import partial
from scipy.special import kn

# make all unit of angle to rad
oversample = 2

def generate_subpixels(loopx, loopy, sapf):
    x1 = jnp.arange(sapf)
    xm, ym = jnp.meshgrid(x1, x1)
    dx = (xm + 0.5) / sapf + loopx
    dy = (ym + 0.5) / sapf + loopy
    return dx.flatten(), dy.flatten()

grp1_dx, grp1_dy = generate_subpixels(0, 0, 50)
grp1_loopx = jnp.array([0])
grp1_loopy = jnp.array([0])
grp1_dx = jnp.array(grp1_dx)
grp1_dy = jnp.array(grp1_dy)

grp2_dx, grp2_dy, grp2_lx, grp2_ly = [], [], [], []
for lx, ly in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    dx, dy = generate_subpixels(lx, ly, 25)
    grp2_dx.append(dx)
    grp2_dy.append(dy)
    grp2_lx.append(lx)
    grp2_ly.append(ly)
grp2_dx = jnp.array(grp2_dx)
grp2_dy = jnp.array(grp2_dy)
grp2_loopx = jnp.array(grp2_lx)
grp2_loopy = jnp.array(grp2_ly)

grp3_dx, grp3_dy, grp3_lx, grp3_ly = [], [], [], []
for lx in range(-2, 3):
    for ly in range(-2, 3):
        if (lx==0 and ly==0) or (abs(lx)+abs(ly) < 2):
            continue
        dx, dy = generate_subpixels(lx, ly, 5)
        grp3_dx.append(dx)
        grp3_dy.append(dy)
        grp3_lx.append(lx)
        grp3_ly.append(ly)
grp3_dx = jnp.array(grp3_dx)
grp3_dy = jnp.array(grp3_dy)
grp3_loopx = jnp.array(grp3_lx)
grp3_loopy = jnp.array(grp3_ly)

def alpha_tanh(r_3d, r_in, r_out, alpha, theta_out):
    theta_out_rad = theta_out
    return theta_out_rad*hypotan(r_3d, r_in, r_out, alpha, theta_out_rad)*jnp.power(0.5*(r_3d/r_out+1.),alpha)

def hypotan(r_3d, r_in, r_out, alpha, theta_out_rad):
    CDEF = 0.23
    A = 2.*CDEF/(jnp.abs(theta_out_rad)+CDEF)-1.0001
    B = (2.-jnp.arctanh(A))*r_out/(r_out-r_in)
    return 0.5*jnp.tanh(B*(r_3d/r_out-1.)+2.)+0.5

def R_2d(x,y):
    return (jnp.sqrt(x**2+y**2)+0.0001)

def R_3d(x,y,PA,i):
    phi = PA
    jax.lax.cond(i < 0.3,lambda x: 0.3, lambda x: x, i) ##？
    dis_proj_square = jnp.abs(jnp.cos(phi)*y-jnp.sin(phi)*x)**2
    return jnp.sqrt(R_2d(x,y)**2+(1./jnp.cos(i)**2-1)*dis_proj_square)

def R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm):
    r_3d = R_3d(x_p,y_p,PA,i)
    theta_r = alpha_tanh(r_3d, r_in, r_out, alpha_rc, theta_out)
    x_p,y_p = xy_3d(x_p,y_p,PA,i)
    x_i = x_p*jnp.cos(theta_r) + y_p*jnp.sin(theta_r)
    y_i = -x_p*jnp.sin(theta_r) + y_p*jnp.cos(theta_r)
    return R_fourier(x_i,y_i,m,am,theta_m,PA,i_arm)

def xy_3d(x,y,PA,i):
    phi = PA
    dis_proj = jnp.cos(phi)*y-jnp.sin(phi)*x
    projvec = (1./jnp.cos(i)-1.)*dis_proj
    return x-projvec*jnp.sin(phi),y+projvec*jnp.cos(phi)

def xpyp(x,y,PA):
    phi = PA
    return jnp.cos(phi)*x+jnp.sin(phi)*y,jnp.cos(phi)*y-jnp.sin(phi)*x

def R_fourier(x,y,m,am,theta_m,PA,i):
    ac = x/R_2d(x,y)
    ac = clamp(-1.+1e-6, ac, 1.-1e-6)
    ang = jnp.arccos(ac)
    phi_map = jnp.where(y < 0.,C.twopi-ang, ang)
    
    #print ('check phi_map')
    #print (jnp.sum(phi_map))

    return R_3d(x,y,PA,i)*(1.+am*jnp.cos(m*(phi_map+theta_m)))

def sersic_L_to_I(alpha,n):
    return C.twopi*alpha**2*n*jnp.exp(jax.scipy.special.gammaln(2*n))

def sersic1D(I,r,alpha,n):
    return I*jnp.exp(-jnp.power(r/alpha,1/n))

def sersic1D_broken(I,r,alpha,n,Rout):
    flux = jnp.exp(-jnp.power(r/alpha,1/n))
    transition_width = 0.05 * Rout
    smooth_transition = 0.5 * (1 + jnp.tanh((Rout - r) / transition_width))
    return smooth_transition*flux*I

def sersic1D_ring(I,r,alpha,n,Rin):
    flux = jnp.exp(-jnp.power(r/alpha,1/n))
    transition_width = 0.05 * Rin
    smooth_transition = 1. - 0.5 * (1 + jnp.tanh((Rin - r) / transition_width))
    return smooth_transition*flux*I

def sersic1D_ringf(I,r,alpha,n,Rin, width):
    flux = jnp.exp(-jnp.power(r/alpha,1/n))
    transition_width = width * Rin
    smooth_transition = 1. - 0.5 * (1 + jnp.tanh((Rin - r) / transition_width))
    return smooth_transition*flux*I

def Ferrer1D(r,R_out,a,beta):
    # Original Ferrer function
    flux = jnp.power(jnp.abs(1 - (r / R_out)**(2 - beta)), a)
    #flux = jnp.where(r < R_out, jnp.power(1 - (r / R_out)**(2 - beta), a), 0.)
    #print (flux)
    # Transition function
    transition_width = 0.05 * R_out
    smooth_transition = 0.5 * (1 + jnp.tanh((R_out - r) / transition_width))
    return smooth_transition*flux

scipyx = jnp.logspace(-5,jnp.log10(30),1000)
scipyk1 = kn(1,scipyx)
def jaxk1(x):
    return jnp.interp(x,scipyx,scipyk1)

def EdgeonDisk1D(r,h,rs,hs):
    return (r/rs)*jaxk1(r/rs)*(1/jnp.cosh(h/hs))**2

def bn(n):
    return 2.*n-1./3+4./405/n+46./25515/n**2+131./1148175/n**3

@partial(jit, static_argnums=(2,3))
def sersic2D(x,y,imnx,imny,axrat,PA,Re,n,L):
    """
    Generate 2D Sérsic surface brightness profile.
    
    Creates elliptical Sérsic profile I(R) ∝ exp(-(R/Re)^(1/n)) for
    modeling bulges, disks, and other galaxy components. JIT-compiled
    for performance.
    
    Parameters
    ----------
    x, y : float
        Center position in pixels (can be sub-pixel).
    imnx, imny : int
        Image dimensions (width, height) in pixels.
        Static arguments for JIT compilation.
    axrat : float
        Axis ratio q = b/a (minor/major axis). Range (0, 1].
        q = 1 for circular, q → 0 for edge-on disk.
    PA : float
        Position angle in radians (measured counter-clockwise from +x axis).
    Re : float
        Effective (half-light) radius in pixels.
        Contains half the total luminosity.
    n : float
        Sérsic index controlling concentration.
        
        - n = 0.5: Gaussian profile
        - n = 1: Exponential disk
        - n = 4: de Vaucouleurs bulge
        - n > 4: Concentrated bulge/elliptical
    L : float
        Total integrated luminosity (arbitrary units).
    
    Returns
    -------
    array_like, shape (imny, imnx)
        2D surface brightness map in flux units (not mag/arcsec^2).
    
    Notes
    -----
    **Sérsic profile formula:**
    
    .. math::
        I(R) = I_e \\exp\\left(-b_n\\left[\\left(\\frac{R}{R_e}\\right)^{1/n} - 1\\right]\\right)
    
    where :math:`b_n \\approx 2n - 1/3` ensures :math:`R_e` encloses half the light.
    
    The deprojected radius :math:`R` accounts for ellipticity and inclination:
    
    .. math::
        R^2 = x'^2 + (y'/q)^2
    
    where :math:`(x', y')` are rotated by PA and :math:`q` = axis ratio.
    
    **Oversampling:** If global `oversample` > 0, central pixel is
    sub-sampled for accurate peak flux (important for PSF convolution).
    
    **Special cases:**
    
    - Exponential disk (n=1): :math:`I(R) \\propto \\exp(-R/h)` with :math:`h = R_e/1.678`
    - de Vaucouleurs (n=4): :math:`I(R) \\propto \\exp(-(R/R_e)^{1/4})`
    
    See Also
    --------
    sersic1D : 1D radial profile
    sersic2D_broken : Sérsic with truncation
    sersic2D_ring : Sérsic with inner hole
    Ferrer : Truncated profile for tidal features
    
    Examples
    --------
    >>> # Exponential disk
    >>> disk = sersic2D(x=50.0, y=50.0, imnx=100, imny=100,
    ...                 axrat=0.6, PA=jnp.pi/4, Re=15.0, n=1.0, L=100.0)
    
    >>> # de Vaucouleurs bulge
    >>> bulge = sersic2D(x=50.0, y=50.0, imnx=100, imny=100,
    ...                  axrat=0.9, PA=0.0, Re=5.0, n=4.0, L=50.0)
    
    References
    ----------
    .. [1] Sérsic, J. L. 1963, Bol. Asoc. Arg. Astron., 6, 41
    .. [2] Graham & Driver 2005, PASA, 22, 118
    """
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    i = jnp.arccos(axrat)
    r_3d = R_3d(x_p,y_p,PA,i)
    
    alpha = Re/bn(n)**n
    I = L/sersic_L_to_I(alpha,n)/axrat
    #print ("check I")
    #print (I)
    fluxmap = sersic1D(I,r_3d,alpha,n)
    #print ("check fluxmap1")
    #print (jnp.sum(fluxmap))
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean1 = jnp.mean(sersic1D(I,r_3d,alpha,n))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean2 = jnp.mean(sersic1D(I,r_3d,alpha,n), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean3 = jnp.mean(sersic1D(I,r_3d,alpha,n), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    #print ("check fluxmap1")
    #print (jnp.sum(fluxmap))
    return fluxmap

@partial(jit, static_argnums=(2,3))
def sersic2D_broken(x,y,imnx,imny,axrat,PA,Re,n,L,Rout):
    """
    Sérsic profile with smooth outer truncation.
    
    Same as sersic2D but with smooth exponential cutoff at Rout using tanh transition.
    Useful for modeling truncated disks or tidally stripped galaxies.
    
    Additional Parameters
    ---------------------
    Rout : float
        Truncation radius in pixels. Flux smoothly drops to zero beyond Rout.
    
    See Also
    --------
    sersic2D : Standard Sérsic profile
    Ferrer : Alternative truncation function
    """
    # sersic with an outer cutoff 
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    i = jnp.arccos(axrat)
    r_3d = R_3d(x_p,y_p,PA,i)
    alpha = Re/bn(n)**n
    fluxmap = sersic1D_broken(1.,r_3d,alpha,n,Rout)
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean1 = jnp.mean(sersic1D_broken(1.,r_3d,alpha,n,Rout))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean2 = jnp.mean(sersic1D_broken(1.,r_3d,alpha,n,Rout), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean3 = jnp.mean(sersic1D_broken(1.,r_3d,alpha,n,Rout), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    mass_map = L*fluxmap/jnp.sum(fluxmap)
    return mass_map

@partial(jit, static_argnums=(2,3))
def sersic2D_ring(x,y,imnx,imny,axrat,PA,Re,n,L,Rin,width):
    """
    Sérsic profile with inner hole (ring/shell morphology).
    
    Models ring galaxies, shell structures, or centrally depleted systems.
    Flux smoothly suppressed inside Rin with transition width controlled by width parameter.
    
    Additional Parameters
    ---------------------
    Rin : float
        Inner hole radius in pixels.
    width : float
        Transition width as fraction of Rin (0 = sharp, 0.1 = smooth).
    
    See Also
    --------
    GaussianRing : Pure ring with Gaussian radial profile
    """
    # sersic with a inner cutoff
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    i = jnp.arccos(axrat)
    r_3d = R_3d(x_p,y_p,PA,i)
    alpha = Re/bn(n)**n
    fluxmap = sersic1D_ringf(1.,r_3d,alpha,n,Rin,width)
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean1 = jnp.mean(sersic1D_ringf(1.,r_3d,alpha,n,Rin,width))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean2 = jnp.mean(sersic1D_ringf(1.,r_3d,alpha,n,Rin,width), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean3 = jnp.mean(sersic1D_ringf(1.,r_3d,alpha,n,Rin,width), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    mass_map = L*fluxmap/jnp.sum(fluxmap)
    return mass_map

@partial(jit, static_argnums=(2,3))
def Ferrer(x,y,imnx,imny,axrat,PA,R_out,a,beta,L):
    """
    Ferrer (truncated power-law) profile for bars and tidal features.
    
    Models sharply truncated structures with flexible shape control.
    
    Parameters
    ----------
    x, y : float
        Center position in pixels.
    imnx, imny : int
        Image dimensions (width, height) in pixels.
    axrat : float
        Axis ratio q = b/a (minor/major axis). Range (0, 1].
    PA : float
        Position angle in radians.
    R_out : float
        Outer truncation radius in pixels.
    a : float
        Central concentration parameter (typical: 2-4).
    beta : float
        Edge sharpness (typical: 0-2). :math:`\\beta=0` gives steepest edge.
    L : float
        Total integrated luminosity.
    
    Returns
    -------
    array_like, shape (imny, imnx)
        2D surface brightness map.
    
    Notes
    -----
    **Ferrer profile formula:**
    
    .. math::
        I(R) \\propto \\left[1 - \\left(\\frac{R}{R_{\\rm out}}\\right)^{2-\\beta}\\right]^a
    
    for :math:`R < R_{\\rm out}`, and :math:`I(R) = 0` otherwise.
    
    Common for modeling galactic bars, tidal tails, and shells.
    Smooth tanh transition prevents numerical issues at boundary.
    
    References
    ----------
    Ferrer et al. 2012, A&A, 545, A59
    """
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    i = jnp.arccos(axrat)
    r_3d = R_3d(x_p,y_p,PA,i)
    fluxmap = Ferrer1D(r_3d,R_out,a,beta)
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean1 = jnp.mean(Ferrer1D(r_3d,R_out,a,beta))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean2 = jnp.mean(Ferrer1D(r_3d,R_out,a,beta), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean3 = jnp.mean(Ferrer1D(r_3d,R_out,a,beta), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    mass_map = L*fluxmap/jnp.sum(fluxmap)
    return mass_map

@partial(jit, static_argnums=(2,3))
def GaussianRing(x,y,imnx,imny,axrat,PA,r0,Sig,L):
    """
    Narrow Gaussian ring profile.
    
    I(R) ∝ exp[-(R-r0)²/(2σ²)]. Models narrow ring structures
    like Einstein rings, resonance rings, or circular shell features.
    
    Parameters (in addition to standard x, y, imnx, imny, axrat, PA, L)
    ------------------------------------------------------------------------
    r0 : float
        Ring radius in pixels.
    Sig : float
        Ring width (σ of Gaussian) in pixels.
    
    Notes
    -----
    For broader rings with Sérsic-like profiles, use sersic2D_ring instead.
    """
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    i = jnp.arccos(axrat)
    r_3d = R_3d(x_p,y_p,PA,i)
    fluxmap = jnp.exp(-0.5*(r_3d-r0)**2/Sig**2)
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean1 = jnp.mean(jnp.exp(-0.5*(r_3d-r0)**2/Sig**2))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean2 = jnp.mean(jnp.exp(-0.5*(r_3d-r0)**2/Sig**2), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        r_3d = R_3d(x_p,y_p,PA,i)
        mean3 = jnp.mean(jnp.exp(-0.5*(r_3d-r0)**2/Sig**2), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    mass_map = L*fluxmap/jnp.sum(fluxmap)
    return mass_map

@partial(jit, static_argnums=(2,3))
def EdgeonDisk(x,y,imnx,imny,PA,rs,hs,L):
    """
    Edge-on exponential disk with vertical structure.
    
    Models edge-on disk galaxies with exponential radial decline and
    sech²(z/hs) vertical profile. Uses modified Bessel function :math:`K_1`
    for proper radiative transfer in edge-on geometry.
    
    Parameters ----------
    x, y : float
        Center position in pixels.
    imnx, imny : int
        Image dimensions (width, height) in pixels.
    PA : float
        Position angle in radians.
    rs : float
        Radial scale length in pixels.
    hs : float
        Vertical scale height in pixels. Typical rs/hs ~ 5-10 for disk galaxies.
    L : float
        Total integrated luminosity.
    
    Returns
    -------
    array_like, shape (imny, imnx)
        2D surface brightness map.
    
    Notes
    -----
    **Edge-on disk formula:**
    
    .. math::
        I(r,z) \\propto \\frac{r}{r_s} K_1\\left(\\frac{r}{r_s}\\right) {\\rm sech}^2\\left(\\frac{z}{h_s}\\right)
    
    where :math:`K_1` is modified Bessel function of second kind.
    
    See Also
    --------
    sersic2D : For face-on or moderately inclined disks
    
    References
    ----------
    van der Kruit & Searle 1981, A&A, 95, 105
    """
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    xpp,ypp = xpyp(x_p,y_p,PA)
    fluxmap = EdgeonDisk1D(jnp.abs(xpp),jnp.abs(ypp),rs,hs)
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        xpp,ypp = xpyp(x_p,y_p,PA)
        mean1 = jnp.mean(EdgeonDisk1D(jnp.abs(xpp),jnp.abs(ypp),rs,hs))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        xpp,ypp = xpyp(x_p,y_p,PA)
        mean2 = jnp.mean(EdgeonDisk1D(jnp.abs(xpp),jnp.abs(ypp),rs,hs), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        xpp,ypp = xpyp(x_p,y_p,PA)
        mean3 = jnp.mean(EdgeonDisk1D(jnp.abs(xpp),jnp.abs(ypp),rs,hs), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    mass_map = L*fluxmap/jnp.sum(fluxmap)
    return mass_map

@partial(jit, static_argnums=(2,3))
def sersic2D_fourier(x,y,imnx,imny,axrat,PA,Re,n,L, r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm):
    """
    Sérsic profile with Fourier-mode spiral arms.
    
    Models spiral galaxies by modulating base Sérsic profile with cos(mφ) terms.
    
    Parameters
    ----------
    x, y, imnx, imny, axrat, PA, Re, n, L : standard Sérsic parameters
        See sersic2D documentation.
    r_in : float
        Inner radius where spiral pattern starts (pixels).
    r_out : float
        Outer radius where spiral pattern ends (pixels).
    alpha_rc : float
        Rotation curve power-law index (affects arm winding).
    theta_out : float
        Azimuthal phase at r_out (controls arm pitch angle).
    m : int
        Number of spiral arms (m=2 for grand-design, m>2 for multi-arm).
    am : float
        Relative arm amplitude (0.1-0.5 typical). Larger = more prominent arms.
    theta_m : float
        Phase offset of spiral pattern (radians).
    i_arm : float
        Inclination of spiral plane (radians). Use for tilted spirals.
    
    Returns
    -------
    array_like, shape (imny, imnx)
        2D surface brightness map with spiral structure.
    
    Notes
    -----
    **Spiral modulation formula:**
    
    .. math::
        R_{\\rm spiral} = R \\times \\left[1 + a_m \\cos(m(\\phi + \\theta_m))\\right]
    
    Logarithmic spiral pitch angle :math:`\\approx \\arctan(1/m)`. Grand-design spirals 
    (m=2) have shallow pitch (~10-20°), multi-arm spirals (m>2) are tighter.
    
    See Also
    --------
    sersic2D : Base profile without spiral structure
    
    References
    ----------
    Rix & Zaritsky 1995, ApJ, 447, 82
    """
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    i = jnp.arccos(axrat)
    r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
    #print ('check cord')
    #print (jnp.sum(r_3d))
    alpha = Re/bn(n)**n
    I = 1.
    fluxmap = sersic1D(I,r_3d,alpha,n)
    #print (jnp.sum(fluxmap))
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean1 = jnp.mean(sersic1D(I,r_3d,alpha,n))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean2 = jnp.mean(sersic1D(I,r_3d,alpha,n), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean3 = jnp.mean(sersic1D(I,r_3d,alpha,n), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    #print (jnp.sum(fluxmap))
    mass_map = L*fluxmap/jnp.sum(fluxmap)
    #print (jnp.sum(mass_map))
    return mass_map

@partial(jit, static_argnums=(2,3))
def sersic2D_ringfourier(x,y,imnx,imny,axrat,PA,Re,n,L, r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm):
    '''
    Re,n: control the mass distribution of the arms
    i, PA galaxy projection
    r_in, r_out, alpha_rc, theta_out: rotation curve
    m, am, theta_m, i_arm: arm number, relative strength
    '''
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    i = jnp.arccos(axrat)
    r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
    alpha = Re/bn(n)**n
    I = 1.
    fluxmap = sersic1D_ring(I,r_3d,alpha,n,r_in)
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean1 = jnp.mean(sersic1D_ring(I,r_3d,alpha,n,r_in))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean2 = jnp.mean(sersic1D_ring(I,r_3d,alpha,n,r_in), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean3 = jnp.mean(sersic1D_ring(I,r_3d,alpha,n,r_in), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    mass_map = L*fluxmap/jnp.sum(fluxmap)
    return mass_map

@partial(jit, static_argnums=(2,3))
def sersic2D_brokenfourier(x,y,imnx,imny,axrat,PA,Re,n,L, r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm):
    '''
    Re,n: control the mass distribution of the arms
    i, PA galaxy projection
    r_in, r_out, alpha_rc, theta_out: rotation curve
    m, am, theta_m, i_arm: arm number, relative strength
    '''
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    i = jnp.arccos(axrat)
    r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
    alpha = Re/bn(n)**n
    I = 1.
    fluxmap = sersic1D_broken(I,r_3d,alpha,n,r_out)
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean1 = jnp.mean(sersic1D_broken(I,r_3d,alpha,n,r_out))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean2 = jnp.mean(sersic1D_broken(I,r_3d,alpha,n,r_out), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean3 = jnp.mean(sersic1D_broken(I,r_3d,alpha,n,r_out), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    mass_map = L*fluxmap/jnp.sum(fluxmap)
    return mass_map


@partial(jit, static_argnums=(2,3))
def Ferrer_fourier(x,y,imnx,imny,axrat,PA,R_out,a,beta,L,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm):
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    i = jnp.arccos(axrat)
    r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
    fluxmap = Ferrer1D(r_3d,R_out,a,beta)
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean1 = jnp.mean(Ferrer1D(r_3d,R_out,a,beta))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean2 = jnp.mean(Ferrer1D(r_3d,R_out,a,beta), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean3 = jnp.mean(Ferrer1D(r_3d,R_out,a,beta), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    mass_map = L*fluxmap/jnp.sum(fluxmap)
    return mass_map

@partial(jit, static_argnums=(2,3))
def GaussianRing_fourier(x,y,imnx,imny,axrat,PA,r0,Sig,L,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm):
    xaxis = jnp.arange(imnx)
    yaxis = jnp.arange(imny)
    xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
    x_p = xmesh + 0.5 - x
    y_p = ymesh + 0.5 - y
    i = jnp.arccos(axrat)
    r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
    fluxmap = jnp.exp(-0.5*(r_3d-r0)**2/Sig**2)
    intx = x.astype(int)
    inty = y.astype(int)
    if oversample > 0:
        # Group 1 (1 block, 2500 subs)
        x_p = grp1_dx + intx - x
        y_p = grp1_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean1 = jnp.mean(jnp.exp(-0.5*(r_3d-r0)**2/Sig**2))
        fluxmap = fluxmap.at[inty+grp1_loopy[0], intx+grp1_loopx[0]].set(mean1)
        
        # Group 2 (4 blocks, 625 subs each)
        x_p = grp2_dx + intx - x
        y_p = grp2_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean2 = jnp.mean(jnp.exp(-0.5*(r_3d-r0)**2/Sig**2), axis=1)
        fluxmap = fluxmap.at[inty + grp2_loopy, intx + grp2_loopx].set(mean2)

        # Group 3 (20 blocks, 25 subs each)
        x_p = grp3_dx + intx - x
        y_p = grp3_dy + inty - y
        r_3d = R_3d_fourier(x_p,y_p,PA,i,r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
        mean3 = jnp.mean(jnp.exp(-0.5*(r_3d-r0)**2/Sig**2), axis=1)
        fluxmap = fluxmap.at[inty + grp3_loopy, intx + grp3_loopx].set(mean3)
    mass_map = L*fluxmap/jnp.sum(fluxmap)
    return mass_map

def add_PSF(x,y,imnx,imny,psfimage,scalefactor=1.):
    ny0,nx0=psfimage.shape
    shiftx = x-0.5*nx0*scalefactor+0.5
    shifty = y-0.5*ny0*scalefactor+0.5
    scale = jnp.array([scalefactor,scalefactor], dtype=jnp.float32)
    trans = jnp.array([shifty,shiftx], dtype=jnp.float32)
    return jax.image.scale_and_translate(psfimage,(imny,imnx),(0,1),scale,trans,'cubic')/(scalefactor**2)

def add_PSF_2D(x, y, imnx, imny, psfimage_slice, scalefactor):
    ny0, nx0 = psfimage_slice.shape
    shiftx = x - 0.5 * nx0 * scalefactor + 0.5
    shifty = y - 0.5 * ny0 * scalefactor + 0.5
    scale = jnp.array([scalefactor, scalefactor], dtype=jnp.float32)
    trans = jnp.array([shifty, shiftx], dtype=jnp.float32)
    return jax.image.scale_and_translate(psfimage_slice, (imny, imnx), (0, 1), scale, trans, 'cubic')/ (scalefactor**2)

def add_PSF_3D(x, y, imnx, imny, psfimage, scalefactor):
    # Use vmap to apply the add_PSF_2D function across the third dimension of psfimage
    return  jax.vmap(add_PSF_2D, in_axes=(None, None, None, None, 2, None))(x, y, imnx, imny, psfimage, scalefactor)

def Fnu_to_Fl(data, unit='epHz'):
    #mJy to erg/A
    if len(data) == 2:
        xold = data[0]
        xnew = C.ckm*1e13/xold
        yold = data[1]
        ynew = yold*xold/xnew
        if type(xold) is not float:
            xnew = xnew[::-1]
            ynew = ynew[::-1]
        if unit == 'mJy':
            return [xnew,ynew*1e-26]
        else:
            return [xnew,ynew]
    elif len(data) == 3:
        xold = data[0]
        xnew = C.ckm*1e13/xold
        yold = data[1]
        ynew = yold*xold/xnew
        yerr = ynew*data[2]/data[1]
        if type(xold) is not float:
            xnew = xnew[::-1]
            ynew = ynew[::-1]
            yerr = yerr[::-1]
        if unit == 'mJy':
            return [xnew,ynew*1e-26,yerr*1e-26]
        else:
            return [xnew,ynew,yerr]