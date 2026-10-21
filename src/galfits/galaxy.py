import numpy as np
import sys
from . import sed_interp as SEDs
from . import emission_lines as EL
from pathlib import Path
from . import profiles as Pro
from . import Constant as C
import jax.numpy as jnp
from jax import grad, jit, vmap
import os
import jax
from jax import lax
import lmfit
from astropy.convolution import Gaussian1DKernel,convolve
from typing import Literal

jax.config.update("jax_enable_x64", False)

'''
"allbands":
'acs_f625w', '4star_m_j2', 'wfc3_f555w', 'wfc3_f139m', 'acs_f475w', 'ukidss_h', 'ndwfs_r', '4star_m_j3', 'acs_f435w',
    'ch1', 'ndwfs_i', '4star_j', 'galex_nuv', 'wfc3_f606w', '4star_m_hlong', 'wfc3_f125w', 'newfirm_j', 'newfirm_ks',
    'sloan_z', 'wfc3_f140w', 'sloan_g', 'sloan_i', 'wfc3_f814w', 'sloan_u', 'wfc3_f775w', 'sloan_r', 'r', 'wise_ch1',
    '4star_m_hshort', 'i', '4star_ks', 'wfpc2_f450w', 'README', 'h', 'wfc3_f275w', '4star_h', 'ch3', 'ukidss_k',
    'wfc3_f218w', 'ch4', 'ukidss_y', '4star_m_j1', 'wfc3_f110w', 'ukidss_j', 'ch2', 'wfc3_f153m', 'acs_f606w', 'ndwfs_bw',
    'wfpc2_f814w', 'galex_fuv', 'wfc3_f225w', 'wfpc2_f675w', 'ks', 'acs_f555w', 'wfc3_f625w', 'wfc3_f127m', 'wfc3_f475w',
    'wfpc2_f555w', 'wfc3_f438w', 'wfc3_f105w', 'newfirm_h', 'wfc3_f160w', 'j', 'v', 'acs_f775w', 'wfpc2_f606w', 'wise_ch2',
    'acs_f814w', 'wfc3_f850lp', 'b', 'wise_ch3', 'wise_ch4', 'acs_f850lp', 'KCWI_squashed'
'''

current_dir = os.path.dirname(os.path.abspath(__file__))
filterpath = os.path.join(current_dir, '..', 'data', 'filters')
filterpath = Path(filterpath)


def indentify_xy(x,y):
    """
    Identity transformation for coordinate systems.
    
    This function serves as a placeholder that returns coordinates unchanged. It is used
    when no coordinate transformation is needed between (RA, Dec) offsets and pixel coordinates.
    
    Parameters
    ----------
    x : float or array_like
        X-coordinate (typically RA offset in arcseconds or pixel position).
    y : float or array_like
        Y-coordinate (typically Dec offset in arcseconds or pixel position).
    
    Returns
    -------
    x : float or array_like
        Unchanged x-coordinate.
    y : float or array_like
        Unchanged y-coordinate.
    
    See Also
    --------
    coordinates_transfer : Applies affine transformation to convert from sky to pixel coordinates.
    coordinates_transfer_inverse : Inverse transformation from pixel to sky coordinates.
    """
    return x,y

def coordinates_transfer(x, y, kwargs):
    """
    Transform coordinates from sky offsets (RA, Dec) to pixel coordinates.
    
    This function applies an affine transformation to convert from celestial coordinate offsets
    (in arcseconds) to pixel coordinates in the image plane. The transformation accounts for
    rotation, scale, and pixel offset.
    
    Parameters
    ----------
    x : float or array_like
        RA offset from reference position (in arcseconds).
    y : float or array_like
        Dec offset from reference position (in arcseconds).
    kwargs : dict
        Dictionary containing transformation parameters:
        
        - 'x0' : float
            Reference x-pixel coordinate.
        - 'y0' : float
            Reference y-pixel coordinate.
        - 'x0shift' : float
            Additional x-offset in pixels.
        - 'y0shift' : float
            Additional y-offset in pixels.
        - 'dxra' : float
            Change in x-pixel per RA offset (pixel/arcsec).
        - 'dxdec' : float
            Change in x-pixel per Dec offset (pixel/arcsec).
        - 'dyra' : float
            Change in y-pixel per RA offset (pixel/arcsec).
        - 'dydec' : float
            Change in y-pixel per Dec offset (pixel/arcsec).
    
    Returns
    -------
    xp : float or array_like
        Transformed x-coordinate in pixels.
    yp : float or array_like
        Transformed y-coordinate in pixels.
    
    Notes
    -----
    The transformation follows:
    
    .. math::
        x_p = x_0 + x_{0,shift} + x \cdot dx_{RA} + y \cdot dx_{Dec}
        
        y_p = y_0 + y_{0,shift} + x \cdot dy_{RA} + y \cdot dy_{Dec}
    
    See Also
    --------
    coordinates_transfer_inverse : Inverse transformation from pixel to sky coordinates.
    indentify_xy : Identity transformation (no coordinate conversion).
    """
    xp = kwargs['x0'] + kwargs['x0shift'] + x*kwargs['dxra'] + y*kwargs['dxdec']
    yp = kwargs['y0'] + kwargs['y0shift'] + x*kwargs['dyra'] + y*kwargs['dydec']
    return xp,yp

def SFR_to_Ha(SFR, Z):
    """
    Convert star formation rate to H-alpha luminosity.
    
    This function converts a star formation rate (SFR) to H-alpha line luminosity
    using metallicity-dependent calibration factors. The conversion accounts for
    the fact that H-alpha emissivity depends on the gas-phase metallicity.
    
    Parameters
    ----------
    SFR : float or array_like
        Star formation rate in solar masses per year (M_sun/yr).
    Z : float or array_like
        Gas-phase metallicity (mass fraction, not log). Should be in range [0.0001, 0.05].
    
    Returns
    -------
    L : float or array_like
        H-alpha luminosity in units of 10^38 erg/s.
    
    Notes
    -----
    The conversion uses metallicity-dependent calibration factors interpolated from
    a grid spanning Z = [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05]. The calibration
    is based on nebular emission models and follows:
    
    .. math::
        L_{H\\alpha} = \\frac{SFR}{10^{K(Z)}}
    
    where K(Z) is the metallicity-dependent conversion factor in units of log(erg/s/(M_sun/yr)).
    
    References
    ----------
    The conversion factors are derived from photoionization models similar to those
    in Kennicutt (1998) and Kennicutt & Evans (2012), adjusted for metallicity.
    """
    Zg = jnp.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])
    Kha = jnp.array([-34.388, -34.274, -34.190, -34.137, -34.054, -33.945])+31.
    L = SFR/10**jnp.interp(Z,Zg,Kha)
    return L

def coordinates_transfer_inverse(xp, yp, kwargs):
    """
    Transform coordinates from pixel space back to sky offsets (RA, Dec).
    
    This function applies the inverse affine transformation to convert from pixel
    coordinates back to celestial coordinate offsets (in arcseconds). It inverts
    the transformation performed by `coordinates_transfer`.
    
    Parameters
    ----------
    xp : float or array_like
        X-coordinate in pixels.
    yp : float or array_like
        Y-coordinate in pixels.
    kwargs : dict
        Dictionary containing transformation parameters (same as `coordinates_transfer`):
        
        - 'x0' : float
            Reference x-pixel coordinate.
        - 'y0' : float
            Reference y-pixel coordinate.
        - 'dxra' : float
            Change in x-pixel per RA offset (pixel/arcsec).
        - 'dxdec' : float
            Change in x-pixel per Dec offset (pixel/arcsec).
        - 'dyra' : float
            Change in y-pixel per RA offset (pixel/arcsec).
        - 'dydec' : float
            Change in y-pixel per Dec offset (pixel/arcsec).
    
    Returns
    -------
    x : float or array_like
        RA offset from reference position (in arcseconds).
    y : float or array_like
        Dec offset from reference position (in arcseconds).
    
    Notes
    -----
    The inverse transformation solves the linear system:
    
    .. math::
        x = \\frac{(x_p - x_0) \\cdot dy_{Dec} - (y_p - y_0) \\cdot dx_{Dec}}{dx_{RA} \\cdot dy_{Dec} - dy_{RA} \\cdot dx_{Dec}}
        
        y = \\frac{(x_p - x_0) \\cdot dy_{RA} - (y_p - y_0) \\cdot dx_{RA}}{dy_{RA} \\cdot dx_{Dec} - dx_{RA} \\cdot dy_{Dec}}
    
    See Also
    --------
    coordinates_transfer : Forward transformation from sky to pixel coordinates.
    """
    x = ((xp-kwargs['x0'])*kwargs['dydec'] - (yp-kwargs['y0'])*kwargs['dxdec'])/(kwargs['dxra']*kwargs['dydec']-kwargs['dyra']*kwargs['dxdec'])
    y = ((xp-kwargs['x0'])*kwargs['dyra'] - (yp-kwargs['y0'])*kwargs['dxra'])/(kwargs['dyra']*kwargs['dxdec']-kwargs['dxra']*kwargs['dydec'])
    return x,y

class switch(object):
    """
    Switch-case statement implementation for Python.
    
    This class provides switch-case functionality similar to C/C++ for Python.
    It allows cleaner handling of multiple conditional branches compared to
    chained if-elif statements.
    
    Parameters
    ----------
    value : any
        The value to match against in case statements.
    
    Attributes
    ----------
    value : any
        The value being switched on.
    fall : bool
        Flag indicating if a case has been matched (enables fall-through behavior).
    
    Examples
    --------
    >>> for case in switch(value):
    ...     if case('option1'):
    ...         # Handle option1
    ...         break
    ...     if case('option2', 'option3'):
    ...         # Handle option2 or option3
    ...         break
    ...     if case():  # default case
    ...         # Handle default
    ...         break
    
    Notes
    -----
    The switch class mimics C-style switch statements. Once a match is found,
    subsequent cases will also match (fall-through) unless break is used.
    An empty case() serves as the default case.
    """
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """
        Return the match method once, then stop.
        
        Returns
        -------
        generator
            Yields the match method for use in iteration.
        """
        yield self.match
        raise StopIteration

    def match(self, *args):
        """
        Indicate whether or not to enter a case suite.
        
        Parameters
        ----------
        *args : any
            Values to match against. If empty, acts as default case.
        
        Returns
        -------
        bool
            True if this case should execute, False otherwise.
        """
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

def Cal_map(r, type, paradic):
    """
    Calculate spatially-resolved parameter maps using various functional forms.
    
    This function computes physical parameter distributions (e.g., age, metallicity,
    star formation history) as a function of radius using different mathematical
    profiles. It supports linear, exponential, constant, and rotation curve profiles.
    
    Parameters
    ----------
    r : float or array_like
        Radial distance or normalized radius (units depend on the chosen profile type).
    type : str
        Type of radial profile to use. Options are:
        
        - 'linear' : Linear interpolation between inner and outer values
        - 'exp' : Exponential decay profile
        - 'const' : Constant value (no radial dependence)
        - 'arctan' : Arctangent profile (typically used for rotation curves)
    
    paradic : dict
        Dictionary of parameters specific to the chosen profile type:
        
        For 'linear':
            - 'in' : float - Value at r=0 (inner region)
            - 'out' : float - Value at r=1 (outer region)
        
        For 'exp':
            - 'in' : float - Value at r=0
            - 'out' : float - Asymptotic value at large r
            - 'k' : float - Scale length for exponential decay
        
        For 'const':
            - 'value' : float - Constant value at all radii
        
        For 'arctan':
            - 'r0' : float - Reference radius
            - 'rt' : float - Turnover radius scale
            - 'v0' : float - Velocity offset
            - 'vc' : float - Characteristic velocity
    
    Returns
    -------
    result : float or array_like
        The computed parameter value(s) at the given radius/radii.
    
    Raises
    ------
    ValueError
        If an unrecognized profile type is specified.
    
    Notes
    -----
    Profile equations:
    
    - Linear: :math:`f(r) = f_{in}(1-r) + f_{out} \\cdot r`
    - Exponential: :math:`f(r) = (f_{in} - f_{out}) e^{-r/k} + f_{out}`
    - Constant: :math:`f(r) = c`
    - Arctangent: :math:`f(r) = v_0 + \\frac{2v_c}{\\pi} \\arctan\\left(\\frac{r-r_0}{r_t}\\right)`
    
    The arctangent profile is commonly used to model rotation curves of galaxies.
    
    See Also
    --------
    Cal_gradient_map : Calculate gradient correction maps for spatially varying SEDs.
    """
    for case in switch(type):
        if case('linear'):
            return paradic['in']*(1-r) + paradic['out']*r
            break
        if case('exp'):
            return (paradic['in']-paradic['out'])*jnp.exp(-r/paradic['k'])+paradic['out']
            break
        if case('const'):
            return jnp.ones_like(r,dtype=float)*paradic['value']
            break
        if case('arctan'):
            R = (r-paradic['r0'])/paradic['rt']
            return paradic['v0'] + 2.*paradic['vc']*jnp.arctan(R)/jnp.pi
            break
        if case():
            raise ValueError("Unidentified method for calculate age or Z map")

def Cal_gradient_map(r_grid, r_map, method, type, paradic, flux_band, f2, interX, ax, **kwargs):
    """
    Calculate flux correction maps for spatially varying stellar population parameters.
    
    This function computes how the integrated flux in a photometric band changes as a
    function of stellar population parameters (age, metallicity, dust extinction, or
    star formation history) across the galaxy. This is used to correct for gradient
    effects when fitting spatially resolved observations.
    
    Parameters
    ----------
    r_grid : array_like
        Radial grid points at which to evaluate the parameter gradients (typically
        normalized radii from 0 to 1).
    r_map : array_like
        2D map of radial distances for each pixel in the image.
    method : str
        The parameter whose gradient is being calculated. Options:
        
        - 'age' : Stellar population age gradient
        - 'f_cont' : Continuous star formation fraction gradient
        - 'Z' : Metallicity gradient
        - 'Av' : Dust extinction gradient
    
    type : str
        Functional form of the gradient (see `Cal_map` for options).
    paradic : dict
        Parameters defining the gradient profile (see `Cal_map` for structure).
    flux_band : float
        Reference flux in the photometric band for normalization.
    f2 : callable
        Filter transmission function f2(wavelength).
    interX : array_like
        Wavelength grid for SED integration (in Angstroms).
    ax : float
        Normalization factor for the filter (integral of transmission curve).
    **kwargs : dict
        Additional parameters needed for SED calculation:
        
        - 'f_cont_zero' : float - Reference star formation fraction
        - 'age_zero' : float - Reference age (Gyr)
        - 'Z_zero' : float - Reference metallicity (mass fraction)
        - 'Av_zero' : float - Reference extinction (magnitudes)
    
    Returns
    -------
    gradient_map : ndarray
        2D map of flux correction factors relative to the central value.
        Shape matches `r_map`. Values near 1.0 indicate similar flux to center.
    
    Raises
    ------
    ValueError
        If an unrecognized method is specified.
    
    Notes
    -----
    The function works by:
    
    1. Computing SEDs at different radii with varying parameters
    2. Integrating these SEDs through the filter bandpass
    3. Creating a flux ratio map relative to the central flux
    4. Interpolating this onto the full 2D radial map
    
    This is essential for accurate photometric modeling when stellar populations
    vary spatially within galaxies.
    
    See Also
    --------
    Cal_map : Calculate parameter values as a function of radius.
    """
    if type =='const':
        return np.ones_like(r_map,dtype=float)
    else:
        fratio_list = []
        targe_para = Cal_map(r_grid,type,paradic)
        for loopr in range(len(r_grid)):
            if method =='age':
                centerSED = SEDs.get_host_SED(interX, 0., kwargs['f_cont_zero'], targe_para[loopr], kwargs['Z_zero'],kwargs['Av_zero'],1.)
            elif method =='f_cont':
                centerSED = SEDs.get_host_SED(interX, 0., targe_para[loopr], kwargs['age_zero'], kwargs['Z_zero'],kwargs['Av_zero'],1.)
            elif method =='Z':
                centerSED = SEDs.get_host_SED(interX, 0., kwargs['f_cont_zero'], kwargs['age_zero'], targe_para[loopr],kwargs['Av_zero'],1.)
            elif method =='Av':
                centerSED = SEDs.get_host_SED(interX, 0., kwargs['f_cont_zero'], kwargs['age_zero'], kwargs['Z_zero'],targe_para[loopr],1.)
            else:
                raise ValueError("Unidentified method for calculate gradient map")
            flux = jnp.trapezoid(centerSED*f2(interX),x=interX)/ax
            fratio_list.append(flux/flux_band)
        fratio_list = np.array(fratio_list)
        targe_map = Cal_map(r_map,type,paradic)
        gradient_map = np.interp(targe_map,np.array(targe_para),fratio_list)
        return gradient_map

class Galaxy(object):
    """
    Galaxy object for modeling multi-component stellar populations and nebular emission.
    
    The Galaxy class represents a galaxy as a collection of subcomponents (e.g., bulge, disk)
    each with its own structural profile, stellar population properties (age, metallicity,
    star formation history), and dust attenuation. It handles spatially resolved modeling
    including radial gradients in stellar population parameters.
    
    Parameters
    ----------
    regionz : float
        Redshift of the region being modeled (can differ from galaxy redshift for
        cosmological calculations).
    z : float, optional
        Redshift of the galaxy. Default is 0.
    ebv_G : float, optional
        Galactic foreground extinction E(B-V) in magnitudes. Default is 0.
    name : str, optional
        Name identifier for the galaxy. Default is 'undefined'.
    currentage : float, optional
        Current age of the universe at the galaxy's redshift in Gyr. Default is 13.8.
    wave_temp : array_like, optional
        Wavelength template for high-resolution spectral modeling. Default is None.
    
    Attributes
    ----------
    name : str
        Galaxy identifier.
    Nsub : int
        Number of subcomponents in the galaxy.
    subnames : list
        Names of all subcomponents.
    subCs : dict
        Structural parameters for each subcomponent (Sersic, exponential, etc.).
    Nebular : ISM or None
        Nebular emission model object if initialized.
    ageparams : dict
        Age distribution parameters for each subcomponent.
    Zparams : dict
        Metallicity distribution parameters for each subcomponent.
    f_cont : dict
        Continuous star formation fraction parameters for each subcomponent.
    Avparams : dict
        Dust extinction parameters for each subcomponent.
    logMlist : dict
        Log stellar mass for each subcomponent (log10(M/Msun)).
    mass_map : dict
        2D spatial mass distribution for each subcomponent.
    SFR_map : dict
        2D star formation rate maps for each subcomponent.
    redshift : float
        Galaxy redshift.
    ebv_G : float
        Galactic extinction.
    rnorm : float
        Normalization factor for cosmological distance effects.
    Re : float
        Effective radius for defining radial maps (kpc).
    massimg : array_like
        Total 2D stellar mass map summed over all components.
    currentage : float
        Current age of universe (Gyr).
    
    Examples
    --------
    >>> from galfits import Galaxy
    >>> gal = Galaxy(regionz=0.1, z=0.1, name='NGC1234')
    >>> gal.add_subC('bulge', logM=10.5, Sparams={'type': 'sersic', ...},
    ...              ageparam={'type': 'const', 'paradic': {'value': 8.0}},
    ...              Zparam={'type': 'const', 'paradic': {'value': 0.02}},
    ...              f_cont={'type': 'const', 'paradic': {'value': 0.1}},
    ...              Avparam={'type': 'const', 'paradic': {'value': 0.5}})
    
    Notes
    -----
    The Galaxy class uses JAX for efficient computation and supports:
    
    - Multiple structural components (Sersic, exponential, Ferrer, etc.)
    - Radial gradients in stellar population properties
    - Nebular emission (continuum and lines)
    - Dust attenuation with various attenuation curves
    - Star formation histories (constant, exponential, burst, binned)
    - Energy balance between absorbed starlight and dust emission
    
    See Also
    --------
    AGN : AGN modeling class for active galactic nuclei.
    ISM : Interstellar medium and nebular emission modeling.
    """
    def __init__(self, regionz, z=0., ebv_G=0., name = 'undefined', currentage = 13.8, wave_temp = None):
        """
        Initialize a Galaxy object.
        
        Parameters
        ----------
        regionz : float
            Redshift of the region being modeled.
        z : float, optional
            Galaxy redshift. Default is 0.
        ebv_G : float, optional
            Galactic foreground extinction E(B-V). Default is 0.
        name : str, optional
            Galaxy name identifier. Default is 'undefined'.
        currentage : float, optional
            Current age of universe in Gyr. Default is 13.8.
        wave_temp : array_like, optional
            Wavelength grid for spectral templates. Default is None.
        """
        self.name = name
        self.Nsub=0
        self.subnames = []
        self.subCs = {}
        self.Nebular = None
        self.ageparams={}
        self.Zparams={}
        self.Zneb_params={}
        self.f_cont={}
        self.Avparams={}
        self.logMlist = {}
        self.shape = None
        self.mass_map = {}
        self.SFR_map = {}
        self.sigma = {}
        self.r_map = None
        self.rot_scale = None
        self.imshape = None
        self.redshift = z
        self.ebv_G = ebv_G
        self.Re = 1.
        self.massimg = None
        self.SFH_list = {}
        self.npSFH_times = {}
        self.currentage = currentage
        self.allconstant = True
        self.r_comname=None
        self.rotation = {}
        self.logUlist = {}
        self.AVbumplist = {}
        self.sedmodelist = {}
        self.energy_balance = {}
        self.useEZ = False
        self.redshift0 = z
        self.wave_temp = wave_temp
        self.table_model_dic = {}
        self.dl0 = SEDs.luminosity_distance(z)/(1+z)
        self.dlr = SEDs.luminosity_distance(regionz)/(1+regionz)
        self.rnorm = (self.dlr/self.dl0)**2


    
    def __del__(self):
        # Nullify object's properties
        self.name = None
        self.Nsub = None
        self.subnames = None
        self.subCs = None
        self.Nebular = None
        self.ageparams = None
        self.Zparams = None
        self.Zneb_params = None
        self.f_cont = None
        self.Avparams = None
        self.logMlist = None
        self.shape = None
        self.mass_map = None
        self.SFR_map = None
        self.sigma = None
        self.r_map = None
        self.rot_scale = None
        self.imshape = None
        self.redshift = None
        self.ebv_G = None
        self.rnorm = None
        self.massimg = None
        self.SFH_list = None
        self.currentage = None
        self.allconstant = None
        self.r_comname = None
        self.logUlist = None
        self.AVbumplist = None
        self.sedmodelist = None
        self.sortwave = None
        self.energy_balance = None
        self.useEZ = None
        self.table_model_dic = None

    def reset_info(self, paradic, z=0., ebv_G=0.):
        """
        Update galaxy redshift and distance-dependent properties.
        
        This method updates the galaxy's redshift, galactic extinction, and
        recalculates distance-dependent quantities including the luminosity
        distance normalization and the physical scale at the galaxy's redshift.
        
        Parameters
        ----------
        paradic : dict
            Parameter dictionary that will be updated with the new scale factor.
            The key 'scale_{name}' will be added/updated with kpc per arcsec at z.
        z : float, optional
            New galaxy redshift. Default is 0.
        ebv_G : float, optional
            New galactic foreground extinction E(B-V). Default is 0.
        
        Notes
        -----
        This method updates:
        
        - rnorm : Luminosity distance normalization factor
        - redshift : Galaxy redshift
        - ebv_G : Galactic extinction
        - paradic['scale_{name}'] : Physical scale (kpc/arcsec)
        - currentage : Age of universe at redshift z
        
        The normalization factor accounts for the difference between the region
        redshift and the galaxy redshift: rnorm = (D_L(z_region)/D_L(z_galaxy))^2
        """
        dnew=SEDs.luminosity_distance(z)/(1+z)
        self.rnorm = (self.dlr/dnew)**2
        self.redshift = z
        self.ebv_G = ebv_G
        paradic['scale_{0}'.format(self.name)] = SEDs.kpc_per_arcsec(z)
        self.currentage = SEDs.cosmo_age(z)

    def add_subC(self,Pro_names,logM,Sparams,ageparam,Zparam,Zneb_param,f_cont,Avparam,logU=-2,sigma=150.,SFH='conti', Avbump=3,agnbin=[], SEDmode = 0,table_model = None):
        """
        Add a subcomponent (e.g., bulge, disk) to the galaxy model.
        
        This method adds a structural and stellar population component to the galaxy.
        Each component has its own spatial profile, stellar mass, age distribution,
        metallicity, dust extinction, and star formation history.
        
        Parameters
        ----------
        Pro_names : str
            Name identifier for the subcomponent (e.g., 'bulge', 'disk', 'bar').
        logM : float
            Log10 of total stellar mass for this component in solar masses.
        Sparams : dict
            Structural parameters dictionary with keys:
            
            - 'type' : str - Profile type ('sersic', 'ferrer', 'edgeondisk', etc.)
            - 'paradic' : dict - Profile-specific parameters
            
            Example for Sersic:
            {'type': 'sersic', 'paradic': {'xcen': 0.0, 'ycen': 0.0, 'Re': 2.0,
             'n': 4.0, 'ang': 45.0, 'axrat': 0.7}}
        
        ageparam : dict
            Age distribution parameters:
            
            - 'type' : str - Distribution type ('const', 'linear', 'exp', 'bins')
            - 'paradic' : dict - Type-specific parameters
            
            Example: {'type': 'const', 'paradic': {'value': 8.0}} for 8 Gyr uniform age
        
        Zparam : dict
            Metallicity distribution parameters (same structure as ageparam).
            Example: {'type': 'const', 'paradic': {'value': 0.02}} for solar metallicity
        
        f_cont : dict
            Continuous star formation fraction parameters (same structure).
            Values between 0 (all stars formed instantaneously) and 1 (continuous SF).
            Example: {'type': 'const', 'paradic': {'value': 0.1}}
        
        Avparam : dict
            V-band dust extinction parameters in magnitudes (same structure).
            Example: {'type': 'const', 'paradic': {'value': 0.5}}
        
        logU : float, optional
            Log10 of ionization parameter for nebular emission. Default is -3.
        sigma : float, optional
            Velocity dispersion in km/s for kinematic modeling. Default is 150.
        SFH : str, optional
            Star formation history type: 'conti' (continuous), 'burst', 'bins'.
            Default is 'conti'.
        Avbump : float, optional
            Strength of the 2175 Angstrom extinction feature. Default is 3.
        agnbin : list, optional
            Age bins for non-parametric SFH (when SFH='bins'). Default is [].
        SEDmode : int or str, optional
            SED components to include:
            
            - 0 : stars + nebular (default)
            - 1 : stars only
            - 2 : nebular only
            - 3 : dust only
            - 4 : stars + nebular + dust
            - 'T' : table model
        
        table_model : object, optional
            Custom table-based SED model. Default is None.
        
        Notes
        -----
        Multiple subcomponents can be added to represent complex galaxy morphologies.
        Each component contributes independently to the total SED and surface brightness
        distribution. Radial gradients are supported for age, Z, f_cont, and Av.
        
        The method automatically updates internal tracking of whether all parameters
        are constant (for computational optimization).
        
        Examples
        --------
        >>> gal.add_subC('bulge', logM=10.5,
        ...              Sparams={'type': 'sersic', 'paradic': {'xcen': 0, 'ycen': 0,
        ...                       'Re': 1.0, 'n': 4.0, 'ang': 0, 'axrat': 0.8}},
        ...              ageparam={'type': 'const', 'paradic': {'value': 10.0}},
        ...              Zparam={'type': 'const', 'paradic': {'value': 0.02}},
        ...              f_cont={'type': 'const', 'paradic': {'value': 0.05}},
        ...              Avparam={'type': 'const', 'paradic': {'value': 0.3}})
        
        See Also
        --------
        update_subC_para : Update parameters of an existing subcomponent.
        fix_loose_subC : Fix or free parameters for fitting.
        """
        self.subCs.update({Pro_names : Sparams})
        self.ageparams.update({Pro_names : ageparam})
        self.Zparams.update({Pro_names : Zparam})
        self.Zneb_params.update({Pro_names : Zneb_param})
        self.Avparams.update({Pro_names : Avparam})
        self.f_cont.update({Pro_names : f_cont})
        self.mass_map.update({Pro_names : 0.})
        self.SFR_map.update({Pro_names : 0.})
        self.logMlist.update({Pro_names : logM})
        self.logUlist.update({Pro_names : logU})
        self.AVbumplist.update({Pro_names : Avbump})
        self.sigma.update({Pro_names : sigma})
        self.subnames.append(Pro_names)
        self.SFH_list.update({Pro_names : SFH})

        # NOTE: keep sedmod as plain Python ints, NOT jnp arrays — fiducial_sed
        # branches on these values with Python `if` inside jitted code paths, and
        # jnp scalars would become tracers -> TracerBoolConversionError.
        if SEDmode == 0:
            # star 
            sedmod = (1, 0, 0)
        elif SEDmode == 1:
            # steNeb 
            sedmod = (0, 1, 0)
        elif SEDmode == 2:
            # star + dust
            sedmod = (1, 0, 1)
        elif SEDmode == 3:
            # star + nebular + dust
            sedmod = (0, 1, 1)
        elif SEDmode == 'T':
            sedmod = (0, 0, 0)
        elif SEDmode == 'Z':
            # EAZY template mode is handled at galaxy level in generate_group_image
            sedmod = (0, 0, 0)
            self.useEZ = True
        else:
            raise ValueError(f"Incorrect sed moode {SEDmode}.")
        
        self.sedmodelist.update({Pro_names : sedmod})
        self.table_model_dic.update({Pro_names : table_model})
        self.energy_balance.update({Pro_names : False})
        if self.r_comname == None:
            self.r_comname = Pro_names
        if ageparam['type'] != 'const' or Zparam['type'] != 'const' or f_cont['type'] != 'const' or Avparam['type'] != 'const':
            if not ageparam['type'] == 'bins':
                self.allconstant = False
                self.r_comname = Pro_names
        self.npSFH_times.update({Pro_names : jnp.array(agnbin)})
        self.Nsub += 1

    def update_subC_para(self,Pro_names,pardict):
        """
        Update all parameters for a specific sub-component from parameter dictionary.
        
        Synchronizes internal parameter storage with values from a flat parameter
        dictionary (e.g., from lmfit fitting results). Updates structural parameters,
        SED parameters (age, metallicity, attenuation, SFH), and normalization.
        
        Parameters
        ----------
        Pro_names : str
            Sub-component name (e.g., 'disk', 'bulge', 'comp1').
            Only one profile name.
        pardict : dict
            Flat parameter dictionary with keys following naming convention:
            
            - Structure: '{Pro_names}_{param}' (e.g., 'disk_n', 'disk_Re')
            - Age: '{Pro_names}_age_{param}' (e.g., 'disk_age_value')
            - Metallicity: '{Pro_names}_Z_{param}' (e.g., 'disk_Z_value')
            - Attenuation: '{Pro_names}_Av_{param}' (e.g., 'disk_Av_value')
            - SFH: '{Pro_names}_f_cont_{param}' (e.g., 'disk_f_cont_value')
            - Normalization: 'logM_{Pro_names}', 'logU_{Pro_names}', 'AVbump_{Pro_names}'
            - Optional: 'sigma_{Pro_names}' (velocity dispersion)
        
        Returns
        -------
        None
            Updates internal dictionaries in place.
        
        Notes
        -----
        This method is typically called during iterative fitting to apply
        parameter updates from the optimizer. It:
        
        1. Updates structural parameters (Sérsic index, effective radius, etc.)
        2. Updates SED gradient parameters (age, Z, Av, f_cont profiles)
        3. Resets mass map to force regeneration
        4. Updates global parameters (mass, ionization parameter, UV bump, σ)
        
        The mass_map is set to 0 to flag that it needs regeneration with
        new parameters via `generate_mass_map()`.
        
        Parameter naming follows hierarchical convention to avoid conflicts
        when multiple components exist (e.g., 'disk' vs 'bulge').
        
        See Also
        --------
        fix_loose_subC : Lock/unlock parameters in lmfit
        generate_mass_map : Regenerate spatial mass distribution
        
        Examples
        --------
        >>> galaxy = Galaxy(...)
        >>> galaxy.add_subC('disk', ...)
        >>> # After fitting, update with new parameters
        >>> params = {'disk_n': 1.5, 'disk_Re': 3.0, 'logM_disk': 10.5,
        ...          'disk_age_value': 1.0, 'disk_Z_value': 0.02, ...}
        >>> galaxy.update_subC_para('disk', params)
        """
        for key in self.subCs[Pro_names]['paradic']:
            self.subCs[Pro_names]['paradic'][key]= pardict[Pro_names+'_'+key]
        for key in self.ageparams[Pro_names]['paradic']:
            self.ageparams[Pro_names]['paradic'][key]= pardict[Pro_names+'_age_'+key]
        for key in self.Zparams[Pro_names]['paradic']:
            self.Zparams[Pro_names]['paradic'][key]= pardict[Pro_names+'_Z_'+key]
        for key in self.Avparams[Pro_names]['paradic']:
            self.Avparams[Pro_names]['paradic'][key]= pardict[Pro_names+'_Av_'+key]
        for key in self.f_cont[Pro_names]['paradic']:
            self.f_cont[Pro_names]['paradic'][key]= pardict[Pro_names+'_f_cont_'+key]
        for key in self.Zneb_params[Pro_names]['paradic']:
            self.Zneb_params[Pro_names]["paradic"][key] = pardict[Pro_names + "_Zneb_" + key] 
        self.mass_map[Pro_names] = 0.
        self.logMlist[Pro_names] = pardict['logM_'+Pro_names]
        self.logUlist[Pro_names] = pardict['logU_'+Pro_names]
        self.AVbumplist[Pro_names] = pardict['AVbump_'+Pro_names]
        try:
            self.sigma[Pro_names] = pardict['sigma_'+Pro_names]
        except KeyError:
            a = 1
        return

    def get_eazy_lognorms(self, Pro_names, pardict):
        return jnp.array(
            [pardict['ezlogn{0}_{1}'.format(i_ez, Pro_names)] for i_ez in range(1, 19)],
            dtype=jnp.float32,
        )

    def fix_loose_subC(self,Pro_names,lmfit_params, fix = True, structure = True, SED = True, norm = True):
        """

        Examples
        --------
        >>> # Stage 1: Fit morphology with fixed SED
        >>> galaxy.fix_loose_subC('disk', params, fix=True, structure=False, SED=True)
        >>> fitter.fit()
        >>> # Stage 2: Fit SED with fixed morphology  
        >>> galaxy.fix_loose_subC('disk', params, fix=True, structure=True, SED=False)
        >>> fitter.fit()
        >>> # Stage 3: Unlock everything for joint fit
        >>> galaxy.fix_loose_subC('disk', params, fix=False)
        >>> fitter.fit()
        """
        if fix:
            if structure:
                for key in self.subCs[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_'+key].vary = False
            if SED:
                for key in self.ageparams[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_age_'+key].vary = False
                for key in self.Zparams[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_Z_'+key].vary = False
                for key in self.Zneb_params[Pro_names]['paradic']:
                    lmfit_params[Pro_names+"_Zneb_"+key].vary = False
                for key in self.Avparams[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_Av_'+key].vary = False
                for key in self.f_cont[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_f_cont_'+key].vary = False  
            if norm:  
                lmfit_params['logM_'+Pro_names].vary = False
        else:
            if structure:
                for key in self.subCs[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_'+key].vary = True
            if SED:
                for key in self.ageparams[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_age_'+key].vary = True
                for key in self.Zparams[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_Z_'+key].vary = True
                for key in self.Zneb_params[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_Zneb_'+key].vary = True
                for key in self.Avparams[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_Av_'+key].vary = True
                for key in self.f_cont[Pro_names]['paradic']:
                    lmfit_params[Pro_names+'_f_cont_'+key].vary = True  
            if norm:  
                lmfit_params['logM_'+Pro_names].vary = True
        return

    def add_arbsubC(self,Pro_names,mass_map,ageparam,Zparam,Zneb_param, f_cont,Avparam,sigmaparam={'type': "const", 'paradic':{'value': 100}},arbi_para={'func':'Users'}):
        ''' 
        To add a subcomponent for a galaxy object

        Pro_names: the name of the profiles
        e.g. = "sersic" "coresersic" "brokenexp" "moffat" "ferrer" "king" "pointsource"

        params: a dictionary of the parameters for this subcomponent
        e.g. for sersic: {'xcen': 50.0, 'ycen': 50.0, 'frac': 0.704, 're': 10.0,
        'nser': 3.0, 'ang': -32.70422048691768, 'axrat': 1.0, 'box': 0.0, 'convolve': False}

        ageparam: a dictionary of the age dsitribution parameters for this subcomponent
        e.g. {'type': 'linear', 'paradic': {'k': -0.05, 'b': 9.0}}
             {'type': 'const', 'paradic': {'value': 5.0}}

        Zparam: a dictionary of the matallicity dsitribution parameters for this subcomponent
        e.g. {'type': 'linear', 'paradic': {'k': 0.0, 'b': 0.02}}

        f_cont: a dictionary of the fraction of starformation dsitribution parameters for this subcomponent
        between [0.,1.], fraction of starformation
        '''
        if Pro_names in self.subCs.keys():
            self.subCs[Pro_names].append(arbi_para)
            self.ageparams[Pro_names].append(ageparam)
            self.Zparams[Pro_names].append(Zparam)
            self.Zneb_params[Pro_names].append(Zneb_param)
            self.Avparams[Pro_names].append(Avparam)
            self.f_cont[Pro_names].append(f_cont)
        else:
            self.subCs.update({Pro_names : [arbi_para]})
            self.ageparams.update({Pro_names : [ageparam]})
            self.Zparams.update({Pro_names : [Zparam]})
            self.Zneb_params.update({Pro_names: [Zneb_param]})
            self.Avparams.update({Pro_names : [Avparam]})
            self.f_cont.update({Pro_names : [f_cont]})
            self.mass_map.update({Pro_names : []})
        self.mass_map[Pro_names].append(mass_map)
        self.shape = mass_map.shape

    def generate_mass_map(self, shape, transpar, aperturemask=[]):

        ny,nx   = shape
        self.imshape    = shape
        image   = jnp.zeros(shape,dtype=jnp.float32)
        Apertures   = []
        for key in self.subCs:
            params  = self.subCs[key]['paradic']
            xpix,ypix   = coordinates_transfer(params['xcen'],params['ycen'],transpar)
            testang     = params['ang']+transpar['delta_ang']
            PA  = testang*C.degree_to_rad

            if self.subCs[key]['type'] == 'sersic':  
                Re  = params['Re'] /transpar['pixsc']
                axrat   = params['axrat']
                n   = params['n']
                mass_map    = Pro.sersic2D(xpix,ypix,nx,ny,axrat,PA,Re,n,10**self.logMlist[key])
                self.mass_map[key]  = mass_map
                image  += mass_map

            elif self.subCs[key]['type'] == 'ferrer':
                rout = params['Rout'] /transpar['pixsc']
                axrat = params['axrat']
                mass_map = Pro.Ferrer(xpix,ypix,nx,ny,axrat,PA,rout,params['alpha'],params['beta'],10**self.logMlist[key])
                self.mass_map[key] = mass_map
                image += mass_map

            elif self.subCs[key]['type'] == 'GauRing':
                r0 = params['r0'] /transpar['pixsc']
                sig = params['sig'] /transpar['pixsc']  
                axrat = params['axrat']
                mass_map = Pro.GaussianRing(xpix,ypix,nx,ny,axrat,PA,r0,sig,10**self.logMlist[key])
                self.mass_map[key] = mass_map
                image += mass_map

            elif self.subCs[key]['type'] == 'edgeondisk':
                rs = params['rs'] /transpar['pixsc']
                hs = params['hs'] /transpar['pixsc']
                mass_map = Pro.EdgeonDisk(xpix,ypix,nx,ny,PA,rs,hs,10**self.logMlist[key])
                self.mass_map[key] = mass_map
                image += mass_map

            elif self.subCs[key]['type'] == 'sersic_f':
                Re = params['Re']/transpar['pixsc']
                r_in = params['r_in']/transpar['pixsc']
                r_out = params['r_out']/transpar['pixsc']
                axrat = params['axrat']
                n = params['n']
                m = params['m']
                am = params['am']
                alpha_rc = params['alpha_rc']
                theta_out = params['theta_out']*C.degree_to_rad
                #testang = params['ang']-61.674422
                #PA = testang*C.degree_to_rad
                theta_m = (params['theta_m']-transpar['delta_ang']-61.674422)*C.degree_to_rad
                i_arm = params['i_arm']*C.degree_to_rad
                mass_map = Pro.sersic2D_fourier(xpix,ypix,nx,ny,axrat,PA,Re,n,10**self.logMlist[key], r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
                self.mass_map[key] = mass_map
                image += mass_map  

            elif self.subCs[key]['type'] == 'sersic_bf':
                Re = params['Re']/transpar['pixsc']
                r_in = params['r_in']/transpar['pixsc']
                r_out = params['r_out']/transpar['pixsc']
                axrat = params['axrat']
                n = params['n']
                m = params['m']
                am = params['am']
                alpha_rc = params['alpha_rc']
                theta_out = params['theta_out']*C.degree_to_rad
                #testang = params['ang']-61.674422
                #PA = testang*C.degree_to_rad
                theta_m = (params['theta_m']-transpar['delta_ang']-61.674422)*C.degree_to_rad
                i_arm = params['i_arm']*C.degree_to_rad
                mass_map = Pro.sersic2D_brokenfourier(xpix,ypix,nx,ny,axrat,PA,Re,n,10**self.logMlist[key], r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
                self.mass_map[key] = mass_map
                image += mass_map   

            elif self.subCs[key]['type'] == 'sersic_rf':
                Re = params['Re']/transpar['pixsc']
                r_in = params['r_in']/transpar['pixsc']
                r_out = params['r_out']/transpar['pixsc']
                axrat = params['axrat']
                n = params['n']
                m = params['m']
                am = params['am']
                alpha_rc = params['alpha_rc']
                theta_out = params['theta_out']*C.degree_to_rad
                #testang = params['ang']-61.674422
                #PA = testang*C.degree_to_rad
                theta_m = (params['theta_m']-transpar['delta_ang']-61.674422)*C.degree_to_rad
                i_arm = params['i_arm']*C.degree_to_rad
                mass_map = Pro.sersic2D_ringfourier(xpix,ypix,nx,ny,axrat,PA,Re,n,10**self.logMlist[key], r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
                self.mass_map[key] = mass_map
                image += mass_map  

            elif self.subCs[key]['type'] == 'sersic_b': 
                Re = params['Re']/transpar['pixsc']
                r_out = params['r_out']/transpar['pixsc']
                testang = params['ang']+transpar['delta_ang']
                axrat = params['axrat']
                n = params['n']
                mass_map = Pro.sersic2D_broken(xpix,ypix,nx,ny,axrat,PA,Re,n,10**self.logMlist[key], r_out)
                self.mass_map[key] = mass_map
                image += mass_map  

            elif self.subCs[key]['type'] == 'sersic_r': 
                Re = params['Re']/transpar['pixsc']
                r_in = params['r_in']/transpar['pixsc']
                width = params['width']
                axrat = params['axrat']
                n = params['n']
                mass_map = Pro.sersic2D_ring(xpix,ypix,nx,ny,axrat,PA,Re,n,10**self.logMlist[key], r_in, width)
                self.mass_map[key] = mass_map
                image += mass_map  

            elif self.subCs[key]['type'] == 'ferrer_f':
                rout = params['Rout'] /transpar['pixsc']
                r_in = params['r_in']/transpar['pixsc']
                r_out = params['r_out']/transpar['pixsc']
                axrat = params['axrat']
                m = params['m']
                am = params['am']
                alpha_rc = params['alpha_rc']
                theta_out = params['theta_out']*C.degree_to_rad
                testang = params['ang']-61.674422
                PA = testang*C.degree_to_rad
                theta_m = (params['theta_m']-transpar['delta_ang']-61.674422)*C.degree_to_rad
                i_arm = params['i_arm']*C.degree_to_rad
                mass_map = Pro.Ferrer_fourier(xpix,ypix,nx,ny,axrat,PA,rout,params['alpha'],params['beta'],10**self.logMlist[key], r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
                self.mass_map[key] = mass_map
                image += mass_map

            elif self.subCs[key]['type'] == 'GauRing_f':
                r0 = params['r0'] /transpar['pixsc']
                sig = params['sig'] /transpar['pixsc']
                r_in = params['r_in']/transpar['pixsc']
                r_out = params['r_out']/transpar['pixsc']
                axrat = params['axrat']
                m = params['m']
                am = params['am']
                alpha_rc = params['alpha_rc']
                theta_out = params['theta_out']*C.degree_to_rad
                testang = params['ang']-61.674422
                PA = testang*C.degree_to_rad
                theta_m = (params['theta_m']-transpar['delta_ang']-61.674422)*C.degree_to_rad
                i_arm = params['i_arm']*C.degree_to_rad
                mass_map = Pro.GaussianRing_fourier(xpix,ypix,nx,ny,axrat,PA,r0,sig,10**self.logMlist[key], r_in, r_out, alpha_rc, theta_out, m, am, theta_m, i_arm)
                self.mass_map[key] = mass_map
                image += mass_map

            elif self.subCs[key]['type'] == 'const':
                mass_map = jnp.ones((ny,nx),dtype=jnp.float32)*10**self.logMlist[key]
                self.mass_map[key] = mass_map
                image += mass_map 
            
            if key == self.r_comname:
                xaxis = jnp.arange(nx)
                yaxis = jnp.arange(ny)
                xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
                x_p = xmesh + 0.5 - xpix
                y_p = ymesh + 0.5 - ypix
                i = jnp.arccos(axrat)
                self.r_map = Pro.R_3d(x_p,y_p,PA,i)*transpar['pixsc']
                self.Re = jnp.max(jnp.array([params['Re'],0.3*jnp.sqrt(nx*ny)*transpar['pixsc']]))
            if len(aperturemask) > 0:
                for apm in aperturemask:
                    mass_map_flat = mass_map.ravel()
                    mass_map_flat_aper = jnp.where(apm, mass_map_flat, 0)
                    Apertures.append(jnp.sum(mass_map_flat_aper)/ jnp.sum(mass_map_flat))
            #print (key,mass_map)
        #change here for rotation curve fitting?
        xaxis = jnp.arange(nx)
        yaxis = jnp.arange(ny)
        xmesh, ymesh = jnp.meshgrid(xaxis, yaxis)
        x_p = xmesh + 0.5 - xpix
        y_p = ymesh + 0.5 - ypix
        i = jnp.arccos(axrat)
        self.r_map = Pro.R_3d(x_p,y_p,PA,i)*transpar['pixsc'] 
        ### v_obs = v_c * cos(phi) * sini (cosphi = x_int/r)
        self.rot_scale = jnp.sin(i) * ((jnp.cos(PA)*x_p+jnp.sin(PA)*y_p)/(self.r_map/transpar['pixsc'])) 
        apertures = []
        if len(aperturemask) > 0:
            lap = len(aperturemask)
            for loop2 in range(lap):
                ap = []
                for loop,key in enumerate(self.subCs):
                    ap.append(Apertures[loop*lap+loop2])
                apertures.append(ap)
        self.massimg = image
        
        return image,apertures

    def generate_SFR_map(self, inte_step=10):

        ny = self.imshape[0]
        nx = self.imshape[1]
        modimage = jnp.zeros((ny,nx),dtype=jnp.float32)
        for key in self.subCs:
            if self.SFH_list[key] == 'bins':
                age_zero = self.npSFH_times[key]
                f_cont_zero = jnp.ones(len(age_zero)-1,dtype=jnp.float32)
                for loop in range(len(age_zero)-1):
                    f_cont_zero = f_cont_zero.at[loop].set(self.f_cont[key]['paradic']['bin{0}'.format(loop+1)])
            else:
                age_zero = Cal_map(0.,self.ageparams[key]['type'],self.ageparams[key]['paradic'])
                f_cont_zero = Cal_map(0.,self.f_cont[key]['type'],self.f_cont[key]['paradic'])
            self.SFR_map[key] = jnp.zeros((ny,nx),dtype=jnp.float32)
            self.SFR_map[key] += 10**SEDs.f_cont_to_logSFR(f_cont_zero, jnp.log10(self.mass_map[key]), age_zero, present_age = self.currentage, SFH = self.SFH_list[key])
        if not self.allconstant:
            r_grid = jnp.linspace(0,1,inte_step)
            for loop in range(inte_step-1):
                testim = jnp.zeros((ny,nx),dtype=jnp.float32)
                for key in self.subCs:
                    r = 0.5*(r_grid[loop]+r_grid[loop+1])
                    f_cont_zero = Cal_map(r,self.f_cont[key]['type'],self.f_cont[key]['paradic'])
                    age_zero = Cal_map(r,self.ageparams[key]['type'],self.ageparams[key]['paradic'])
                    testim += 10**SEDs.f_cont_to_logSFR(f_cont_zero, jnp.log10(self.mass_map[key]), age_zero, present_age = self.currentage, SFH = self.SFH_list[key])
                self.SFR_map[key] = jnp.where((self.r_map>(2*self.Re*r_grid[loop])),testim,self.SFR_map[key])
        for key in self.subCs:
            modimage += self.SFR_map[key]
        return modimage

    def fiducial_sed(self, waveintrin, apertures=None, pardict = None, button: Literal[0,1,2] = 0):
        '''
        This function returns spectrum. different returning type for different button.
        ==========
        button: literal[0,1,2]
        0: total flux for all subcomponents and sub sed components. -> spec_tot
        1: sub-component fluxs for all sed components -> [spec_disk, spec_bulge]
        2: fluxs for every sed components -> [spec_disk_steNeb, spec_disk_duste, spec_bulge_steNeb, spec_bulge_duste]
        
        '''

        f_tot = jnp.zeros_like(waveintrin) ## total flux
        f_subtots   = [] ## tot flux for each sub-component.
        subtot_labels   = [] ## labels for subcomponents e.g.[disk, bulge]
        f_subs  = [] ## flux for each sub-components and sed components 
        sub_labels  = [] ## labels for sub sed e.g. [disk_steNeb, disk_duste, bulge_steNeb, bulge_duste]
        # waveintrin = wavegrid/(1.+self.redshift)

        for loop, key in enumerate(self.subCs): ## loop for every subcomponents of this galaxy.
            ## key: name for profiles Px1) 

            if self.SFH_list[key] == 'bins':
                age     = self.npSFH_times[key]
                f_cont  = jnp.ones(len(age)-1,dtype=jnp.float32)
                for loop in range(len(age)-1):
                    f_cont  = f_cont.at[loop].set(self.f_cont[key]['paradic']['bin{0}'.format(loop+1)])
            else:
                age     = Cal_map(0., self.ageparams[key]['type'], self.ageparams[key]['paradic'])
                f_cont  = Cal_map(0., self.f_cont[key]['type'], self.f_cont[key]['paradic'])

            Zste    = Cal_map(0., self.Zparams[key]['type'], self.Zparams[key]['paradic'])
            Zneb    = Cal_map(0., self.Zneb_params[key]['type'], self.Zneb_params[key]['paradic']) # In principle, this two parameters should be passed by pardict. because sed gredient is not fully implemented yet.
            # Av      = Cal_map(0., self.Avparams[key]['type'], self.Avparams[key]['paradic'])

            # fixed params are passed through self. 
            present_age     = self.currentage
            SFH             = self.SFH_list[key]
            sed_mode        = self.sedmodelist[key]  
            zred            = self.redshift
            ebv             = self.ebv_G
            rnorm           = self.rnorm
            Gal_name        = self.name
                                                 
            Stellar, SteNeb, Duste   = False, False, False
            if sed_mode[0] == 1: Stellar    = True  
            if sed_mode[1] == 1: SteNeb     = True 
            if sed_mode[2] == 1: Duste      = True

            # variable params are passed through nebularpar
            ## basic sed params
            logM    = pardict["logM_{0}".format(key)]
            sigma   = pardict["{0}_sigma".format(key)]
            ## dust extinction params
            dust2   = pardict['{0}_dust2'.format(key)]
            dust_ratio  = pardict['{0}_dust_ratio'.format(key)]
            dust_index  = pardict['{0}_dust_index'.format(key)]
            ## dust emission 
            qPAH    = pardict['qPAH_{0}'.format(key)]
            umin    = pardict['Umin_{0}'.format(key)]
            gamma   = pardict['gammad_{0}'.format(key)] ## TODO: standardize the parameter read format. -> add_subC, update_subC_para

            # sed components.
            ## stellar Nebular flux ## TODO: add Stellar interpolate option.
            fc_steNeb   = SEDs.get_host_SED(waveintrin, logM, f_cont, age, Zste, Zneb, C_unit=1, SFH=SFH, present_age=present_age)
            ## dust extinction, which is always done.
            fc_att      = SEDs.apply_dust_extinction(waveintrin, f_cont, logM, age, Zste, Zneb, fc_steNeb, dust2, dust_ratio, dust_index, SFH=SFH)
            ## dust emission 
            # if EB:  ## TODO: add energy balance option..
            duste_L     = jnp.trapezoid(fc_steNeb-fc_att, x=waveintrin)
            fc_duste    = SEDs.get_dustEmi(waveintrin, gamma, umin, qPAH, duste_L)
            fc  = fc_att + sed_mode[2] * fc_duste

            # buttion branches
            f_tot += fc
            f_subtots.append(fc); subtot_labels.append(key)
            # if Stellar: xxx # TODO.
            #   f_subs.append(fs); sub_labels.append("{0}_Stellar_att".format(key))
            if SteNeb: 
                # f_subs.append(fc_steNeb); sub_labels.append("{0}_steNeb".format(key))
                f_subs.append(fc_att); sub_labels.append("{0}_SteNeb_att".format(key))
            if Duste: 
                f_subs.append(fc_duste); sub_labels.append("{0}_duste".format(key))

        if button == 0:
            f_tot_obs        = rnorm * SEDs.sed_to_obse(waveintrin, f_tot, zred, ebv)[1]   
            return f_tot_obs, Gal_name
        elif button == 1:
            f_subtots_obs    = [rnorm * SEDs.sed_to_obse(waveintrin, f_temp, zred, ebv)[1] for f_temp in f_subtots]
            return f_subtots_obs, subtot_labels
        elif button == 2:
            f_subs_obs       = [rnorm * SEDs.sed_to_obse(waveintrin, f_temp, zred, ebv)[1] for f_temp in f_subs]
            return f_subs_obs, sub_labels
        else:
            raise ValueError(f"Incorrect button value:{button}")
    
    def generate_SED(self, resps, wavegrid, fll_tot):
        '''
        This function is used to convolve spectrum with resp curve.
        wavegrid: observed frame.
        '''

        SEDs_array  = []
        for resp in resps:
            filterx = resp[:,0]
            filtery = resp[:,1]
            filtery_intp    = jnp.interp(wavegrid, filterx, filtery, left=0., right=0.)
            ax  = jnp.trapezoid(filtery_intp, x=wavegrid)
            flux_band   = jnp.trapezoid(filtery_intp * fll_tot, x=wavegrid) / ax
            SEDs_array.append(flux_band)

        SEDs_array  = jnp.array(SEDs_array,dtype=jnp.float32)
        return SEDs_array
            

    def generate_image(self, GSdata, band, noSED = False,  pardict = None):
        '''
        Function used to calculate model image for THIS galaxy.
        -----------
        GSdata: class <galfitS_data> in `images.py`
            Class storing infomation related with IMAGE / GRISM, fetching through `Myfitter.GSdata`
        band  : str
            Filter name of the model image you want to calculate. 
            Must be within gsutils.effective_wave.keys(), like `nircam_f115w`.
        noSED : boolean 
            label to branch flux calculation.
            `True` means using simple normalization method to account for flux.
            `False` means using `fiducial_sed` method to calculate flux. 
        pardict: dict
            Dict stores all params value. 
            Fetching through Myfitter.pardict. 
            NOTE: please first update gssummary, then update pardict through `Myfitter.loose_fix_params()`. 
                Don't forget update params of Galaxy class through `gal_model.update_subC_para()`.
        '''
        im  = GSdata.get_image(band)
        band_index  = GSdata.allbands.index(band)
        resp        = GSdata.resps[band_index]
        wavegrid    = GSdata.wavegrid

        PSF     = im.PSF
        PHY2CR  = im.phys_to_counts_rate
        img_label   = im.imglabel

        # global flat index of THIS image (identity-based: robust to duplicate band names)
        loop_image  = next(i for i in range(len(GSdata.allbands)) if GSdata.get_image(i) is im)
        loop_group  = next(g for g, grp in enumerate(GSdata.imageset) if loop_image in grp) # PA 

        group_trans_pardict     = GSdata.coordinates_transfer_paras[loop_group]
        im_trans_pardict    = im.coordinates_transfer_para

        if im.wcsshift:
            im_trans_pardict["x0shift"]     = pardict["csx_{0}".format(im.imglabel)]
            im_trans_pardict["y0shift"]     = pardict["csy_{0}".format(im.imglabel)]

        ny, nx  = GSdata.imagesizes[loop_group]
        _, _    = self.generate_mass_map((ny, nx), transpar=group_trans_pardict) # calculate mass map

        model_img   = jnp.zeros((ny, nx), dtype=jnp.float32)

        if noSED:
            for subC_label in self.subCs:
                flux_subC_temp  = 10**pardict["logNorm_{0}_{1}".format(subC_label, band)]
                model_img  += flux_subC_temp * self.mass_map[subC_label]
                ## convolve when returning.
        else:
            waveintrin  = wavegrid / (1+self.redshift)
            flux_subC_list, subC_labels     = self.fiducial_sed(waveintrin, pardict=pardict, button=1)

            for flux_subC, subC_label in zip(flux_subC_list, subC_labels):
                logM_temp   = pardict["logM_{0}".format(subC_label)]
                flux_subC_unit  = flux_subC / 10**logM_temp
                sed_subC    = self.generate_SED([resp], wavegrid, flux_subC_unit)[0]
                sed_subC    = jnp.array(sed_subC, dtype=jnp.float32)
                model_img  += sed_subC * self.mass_map[subC_label]
        ## downgrade to local frame.

        ny_l, nx_l  = im.cut_image.shape[0], im.cut_image.shape[1] # local frame shape
        scale0  = group_trans_pardict["pixsc"] / im_trans_pardict["pixsc"]
        scale   = jnp.array([scale0, scale0], dtype=jnp.float32)
        shiftx  = (im_trans_pardict['x0shift'] + im_trans_pardict['x0']) - (group_trans_pardict['x0shift'] + group_trans_pardict['x0']) * scale0 + 0.5
        shifty  = (im_trans_pardict['y0shift'] + im_trans_pardict['y0']) - (group_trans_pardict['y0shift'] + group_trans_pardict['y0']) * scale0 + 0.5
        shift   = jnp.array([shifty, shiftx], dtype=jnp.float32)
        model_img_l     = jax.image.scale_and_translate(model_img, [ny_l, nx_l], [0,1], scale, shift, "cubic")
        model_img_l    /= scale0**2
        model_img_l_conv    = jax.scipy.signal.fftconvolve(model_img_l, PSF, mode="same")

        ## unit convert
        model_img_l_conv_CR     = model_img_l_conv * PHY2CR

        return model_img_l_conv_CR

    
    def generate_subcomponent_image(self, GSdata, band, subC_name, noSED=False, pardict=None):
        '''
        Function used to calculate subcomponent model image of THIS galaxy.
        -----------
        GSdata: class <galfitS_data> in `images.py`
            Class storing infomation related with IMAGE / GRISM, fetching through `Myfitter.GSdata`
        band  : str
            Filter name of the model image you want to calculate. 
            Must be within gsutils.effective_wave.keys(), like `nircam_f115w`.
        subC_name: str
            profile name you want to calculate.
        noSED : boolean 
            label to branch flux calculation.
            `True` means using simple normalization method to account for flux.
            `False` means using `fiducial_sed` method to calculate flux. 
        pardict: dict
            Dict stores all params value. 
            Fetching through Myfitter.pardict. 
            NOTE: please first update gssummary, then update pardict through `Myfitter.loose_fix_params()`
                Don't forget update params of Galaxy class through `gal_model.update_subC_para()`.
        '''

        im  = GSdata.get_image(band)
        band_index  = GSdata.allbands.index(band)
        resp    = GSdata.resps[band_index]
        wavegrid    = GSdata.wavegrid
        
        PSF     = im.PSF
        PHY2CR  = im.phys_to_counts_rate

        loop_image  = next(i for i in range(len(GSdata.allbands)) if GSdata.get_image(i) is im)
        loop_group  = next(g for g, grp in enumerate(GSdata.imageset) if loop_image in grp)

        group_trans_pardict     = GSdata.coordinates_transfer_paras[loop_group]
        im_trans_pardict        = im.coordinates_transfer_para

        if im.wcsshift:
            im_trans_pardict["x0shift"]     = pardict["csx_{0}".format(im.imglabel)]
            im_trans_pardict["y0shift"]     = pardict["csy_{0}".format(im.imglabel)]

        # regenerate the mass maps on THIS group's reference frame
        ny, nx  = GSdata.imagesizes[loop_group]
        self.generate_mass_map((ny, nx), transpar=group_trans_pardict)
        model_img   = jnp.zeros((ny, nx), dtype=jnp.float32)
        if noSED:
            flux_subC_temp  = 10**pardict["logNorm_{0}_{1}".format(subC_name, band)]
            model_img      += flux_subC_temp * self.mass_map[subC_name]
        else:
            waveintrin  = wavegrid / (1+self.redshift)
            flux_subC_list, subC_labels = self.fiducial_sed(waveintrin, pardict=pardict, button=1)
            flux_subC   = flux_subC_list[subC_labels.index(subC_name)]
            logM_temp   = pardict["logM_{0}".format(subC_name)]
            flux_subC_unit  = flux_subC / 10**logM_temp  # avoid double-counting logM (in mass map)
            sed_subC    = self.generate_SED([resp], wavegrid, flux_subC_unit)[0]
            sed_subC    = jnp.array(sed_subC, dtype=jnp.float32)
            model_img  += sed_subC * self.mass_map[subC_name]
        ## downgrade to local frame (identical to generate_image).

        ny_l, nx_l  = im.cut_image.shape[0], im.cut_image.shape[1] # local frame shape
        scale0  = group_trans_pardict["pixsc"] / im_trans_pardict["pixsc"]
        scale   = jnp.array([scale0, scale0], dtype=jnp.float32)
        shiftx  = (im_trans_pardict['x0shift'] + im_trans_pardict['x0']) - (group_trans_pardict['x0shift'] + group_trans_pardict['x0']) * scale0 + 0.5
        shifty  = (im_trans_pardict['y0shift'] + im_trans_pardict['y0']) - (group_trans_pardict['y0shift'] + group_trans_pardict['y0']) * scale0 + 0.5
        shift   = jnp.array([shifty, shiftx], dtype=jnp.float32)
        model_img_l     = jax.image.scale_and_translate(model_img, [ny_l, nx_l], [0,1], scale, shift, "cubic")
        model_img_l    /= scale0**2
        model_img_l_conv    = jax.scipy.signal.fftconvolve(model_img_l, PSF, mode="same")

        ## unit convert
        model_img_l_conv_CR     = model_img_l_conv * PHY2CR

        return model_img_l_conv_CR

    def generate_group_image(self, wavegrid, resps, pardict=None):
        '''
        assuming a constant image frame. equal to sed in principle.
        '''
        ny = self.imshape[0]
        nx = self.imshape[1]
        Nbands  = len(resps)
        imm0s   = jnp.zeros((Nbands, ny, nx), dtype=jnp.float32)
        waveintrin  = wavegrid / (1. + self.redshift)
        subKeys     = list(self.subCs.keys())
        Ncomp   = len(subKeys)

        flux_subC_list, subC_labels     = self.fiducial_sed(waveintrin, pardict=pardict, button=1)
        # sed_subC_list   = [self.generate_SED(resps, wavegrid, flux_subC) for flux_subC in flux_subC_list]
        for flux_subC, subC_label in zip(flux_subC_list, subC_labels):
            logM_temp   = pardict["logM_{0}".format(subC_label)]
            flux_subC_unit  = flux_subC / 10**logM_temp ## avoid double logM calculation against massmap.
            sed_subC    = self.generate_SED(resps, wavegrid, flux_subC_unit)
            sed_subC    = jnp.array(sed_subC, dtype=jnp.float32)
            imm0s  += sed_subC[:, None, None] * self.mass_map[subC_label]
        return imm0s


    def make_nebular(self, obsspec,lines_narrow=[], withline = True, nnarrow=1, strict=0.01, D3 = False):
        """
        Initialize nebular emission module and calibrate line templates.
        
        Creates ISM object for nebular continuum and emission lines, optionally
        calibrating line profiles from observed spectrum. Supports both direct
        template usage and data-driven line fitting.
        
        Parameters
        ----------
        obsspec : list of array_like
            Observed spectrum as [wavelength, flux] in observed frame.
            Used for line template calibration when withline=False.
            Wavelength in Angstroms, flux arbitrary units.
        lines_narrow : list of str, optional
            Names of narrow emission lines to model separately (e.g., ['[OIII]5007', '[NII]6584']).
            These get distinct kinematic components. Default: [] (no narrow lines).
        withline : bool, optional
            If True, use pre-computed line templates from CLOUDY models.
            If False, fit line templates directly to obsspec. Default: True.
        nnarrow : int, optional
            Number of narrow-line kinematic components (e.g., 1 for single,
            2 for outflow+disk). Default: 1.
        strict : float, optional
            Strictness parameter for line fitting (0-1). Lower values allow
            more flexible line ratios. Default: 0.01.
        D3 : bool, optional
            If True, use 3D photoionization grids (spatially resolved).
            If False, use integrated templates. Default: False.
        
        Returns
        -------
        params : lmfit.Parameters or None
            Initialized parameters for line fitting (if withline=False).
            None otherwise.
        
        Notes
        -----
        Three operational modes:
        
        1. **Pre-computed lines** (withline=True, D3=False):
           Uses CLOUDY line templates indexed by Z and logU.
           Fast, physically motivated line ratios.
           
        2. **Data-driven lines** (withline=False, D3=False):
           Fits Gaussian profiles to observed lines in obsspec.
           Flexible for unusual line ratios or kinematics.
           Returns initialized parameters for subsequent fitting.
           
        3. **3D photoionization** (D3=True):
           Spatially resolved nebular emission from 3D CLOUDY grids.
           Most sophisticated but computationally expensive.
        
        The ISM object is stored as self.Nebular and used by generate_SED(),
        generate_image(), and generate_SEDIFU() to add nebular emission.
        
        Narrow lines (lines_narrow) are treated as separate kinematic component
        with independent velocity and width, useful for modeling forbidden lines
        from extended narrow-line regions distinct from BLR.
        
        See Also
        --------
        ISM : Nebular emission class
        generate_SED : Uses self.Nebular to add emission lines
        
        Examples
        --------
        >>> # Mode 1: Use CLOUDY templates
        >>> galaxy.make_nebular(obsspec=None, withline=True)
        
        >>> # Mode 2: Fit lines to observed spectrum
        >>> wave_obs = jnp.linspace(4000, 9000, 1000)
        >>> flux_obs = observed_flux  # User's data
        >>> params = galaxy.make_nebular([wave_obs, flux_obs], withline=False,
        ...                              lines_narrow=['[OIII]5007', '[NII]6584'])
        
        >>> # Mode 3: 3D photoionization for IFU
        >>> galaxy.make_nebular(obsspec=None, D3=True)
        """
        prefix = self.name+'nb'
        self.Nebular = ISM(withline,lines_narrow, nnarrow, prefix, D3 = D3) # -> self.Nebular.withline=True for image + sed.
        if (not withline) & (not D3):
            x, sed_obs = SEDs.sed_to_rest(obsspec[0],obsspec[1],self.redshift,self.ebv_G)
            restspec = [x,sed_obs]
            return self.Nebular.model_initialized(restspec, strict=strict)
        elif D3:
            return self.Nebular.model_initialized_3D()
        else:
            pass
    
    def get_SFH(self):
        times = jnp.linspace(10**6.05*1e-9,self.currentage-0.2,1000) # Smallest bin of the SSP model (BPASS v2.3) is 10**6.05 yr
        SFHs = []
        for key in self.subCs:
            logM = self.logMlist[key]
            sfh = jnp.zeros_like(times)
            if self.SFH_list[key] == 'bins':
                age_zero = self.npSFH_times[key]
                f_cont_zero = jnp.ones(len(age_zero)-1,dtype=jnp.float32)
                for loop in range(len(age_zero)-1):
                    f_cont_zero = f_cont_zero.at[loop].set(self.f_cont[key]['paradic']['bin{0}'.format(loop+1)])
                f_cont_real = 10**f_cont_zero
                f_cont_real = f_cont_real/jnp.sum(f_cont_real)
                for loop in range(len(age_zero)-1):
                    sfr = f_cont_real[loop]*10**logM/(age_zero[loop+1]-age_zero[loop])/1e9
                    sfh = jnp.where((times>=age_zero[loop])&(times<age_zero[loop+1]),sfr,sfh)
            else:
                age_zero = Cal_map(0.,self.ageparams[key]['type'],self.ageparams[key]['paradic'])
                f_cont_zero = Cal_map(0.,self.f_cont[key]['type'],self.f_cont[key]['paradic'])
                log_M_cont = logM + f_cont_zero
                log_M_inst = logM + jnp.log10(1-10**f_cont_zero)
                if self.SFH_list[key] == 'conti':
                    logSFRc = log_M_cont - 9 - jnp.log10(self.currentage - 0.2) 
                    sfh = jnp.where((times>0.)&(times<self.currentage-0.2),10**logSFRc,sfh)
                elif self.SFH_list[key] == 'burst':
                    logSFRc = log_M_cont - 8
                    sfh = jnp.where((times>0.)&(times<0.1),10**logSFRc,sfh)
                width = 5e-2 #bin width of a single-year SSP representing instantaneous starburst in log(age/yr)
                # 1. Convert times to years (assuming 'times' and 'age_zero' are in Gyr)
                # We use jnp.maximum to prevent log10(0) errors if the time grid starts at exactly 0.
                times_yr = jnp.maximum(times * 1e9, 10**6.05)
                age_zero_yr = jnp.maximum(age_zero * 1e9, 10**6.05)

                # 2. Calculate the Log-Normal Star Formation Rate
                # Formula: SFR(t) = [M_total / (t * ln(10) * sigma * sqrt(2pi))] * exp(-0.5 * [(log10(t) - log10(t0)) / sigma]^2)

                # Calculate the normalization factor (note: this is an array dependent on times_yr, not a scalar peak)
                norm = (10**log_M_inst) / (times_yr * jnp.log(10) * width * jnp.sqrt(2 * jnp.pi))

                # Calculate the log-normal exponential term
                exponent = -0.5 * ((jnp.log10(times_yr) - jnp.log10(age_zero_yr)) / width)**2
                #print("sanity check 1:",np.trapezoid(norm/(10**log_M_inst) * jnp.exp(exponent),x=times_yr))
                # 3. Add to the overall SFH array
                sfh += norm * jnp.exp(exponent)
                #peak = 10**log_M_inst/jnp.sqrt(2*jnp.pi*(width*1e9)**2) # 1e9 is for converting Gyr to yr since peak is in Msun/yr
                #sfh += SEDs.gaussian_jax(times, peak, age_zero, width)
            sfh.at[0].set(sfh[1])
            SFHs.append(sfh)
        return times, SFHs

    def cal_dynamics(self, paradic=None):
        # this part need to be integrated by dismalPY
        # volmap = jnp.zeros_like(self.massimg,dtype=jnp.float32)
        r_mapkpc = self.r_map*paradic['scale_{0}'.format(self.name)]
        rot_par = {
            'r0': paradic['r0_{0}'.format(self.name)],
            'rt': paradic['rt_{0}'.format(self.name)],
            'vc': paradic['vc_{0}'.format(self.name)],
            'v0': paradic['v0_{0}'.format(self.name)]
        }
        volmap = Cal_map(r_mapkpc,'arctan',rot_par)
        sigmap = paradic['LineSigma_{0}'.format(self.name)]*jnp.ones_like(self.massimg,dtype=jnp.float32)
        return sigmap, volmap

    def generate_SEDIFU(self, wavelength, nebularpar = None):
        """
        Generate full 3D spectral datacube (IFU forward model).
        
        Computes spatially resolved spectroscopy by combining stellar population
        SEDs with spatial gradients in age, metallicity, and attenuation. Produces
        synthetic IFU datacube matching observations from JWST/NIRSpec, VLT/MUSE, etc.
        
        Parameters
        ----------
        wavelength : array_like
            Observed-frame wavelength grid in Angstroms. Length N_λ.
        nebularpar : dict, optional
            Nebular and dust emission parameters. Required if dust emission
            or table models are enabled. Expected keys:
            
            - 'logMdust_{comp}': Log10 dust mass for each component
            - 'qPAH_{comp}', 'Umin_{comp}', 'alphad_{comp}', 'gammad_{comp}': DL14 parameters
            - Additional keys for table models if configured
        
        Returns
        -------
        tot_IFU : array_like, shape (ny, nx, nλ)
            Synthetic IFU datacube in 10^38 erg/s/Å units.
            Axes: [spatial_y, spatial_x, wavelength].
        
        Notes
        -----
        **Algorithm overview:**
        
        1. For each sub-component:
           
           a. Compute spatial maps of age(x,y), Z(x,y), Av(x,y), f_cont(x,y)
           b. Generate stellar SED at each spaxel: SED(λ | age, Z, Av, f_cont)
           c. Add nebular continuum and emission lines
           d. Add dust emission (if enabled)
           e. Apply IGM absorption and Galactic extinction
           f. Weight by stellar mass map
        
        2. Sum all components
        
        **Performance optimization:**
        
        - If allconstant=True: Parameters uniform across galaxy, compute once
        - If allconstant=False: Uses JAX vmap for parallel spaxel computation
        - JIT compilation provides ~10-100× speedup on GPU
        
        The function respects energy balance: if energy_balance[comp]=True,
        dust luminosity equals absorbed stellar light.
        
        Output datacube can be compared directly to observed IFU data or
        convolved with instrumental line-spread function (LSF) and point-spread
        function (PSF) for realistic simulations.
        
        **Memory usage:**
        
        For 100×100 spaxels, 1000 wavelength points: ~40 MB (float32)
        
        See Also
        --------
        generate_image : 2D imaging (integrated over wavelength)
        generate_SED : 1D spectrum (integrated over space)
        get_component_SED : Extract individual component spectra
        
        Examples
        --------
        >>> # Generate IFU cube for JWST/NIRSpec
        >>> wave = jnp.linspace(10000, 50000, 2000)  # 1-5 μm
        >>> cube = galaxy.generate_SEDIFU(wave)
        >>> print(f"Cube shape: {cube.shape}")
        Cube shape: (100, 100, 2000)
        
        >>> # Extract spaxel spectrum at galaxy center
        >>> center_y, center_x = ny//2, nx//2
        >>> central_spec = cube[center_y, center_x, :]
        >>> plt.plot(wave, central_spec)
        
        >>> # With dust emission
        >>> dust_params = {'logMdust_disk': 7.0, 'qPAH_disk': 3.5,
        ...                'Umin_disk': 1.0, 'alphad_disk': 2.0, 'gammad_disk': 0.1}
        >>> cube = galaxy.generate_SEDIFU(wave, nebularpar=dust_params)
        """
        ny, nx = self.imshape
        nw = len(wavelength)
        tot_IFU = jnp.zeros((ny, nx, nw))
        waveintrin = wavelength / (1. + self.redshift)
        wave3D = jnp.tile(waveintrin, (ny, nx, 1))
        rmap = self.r_map
        subKeys = list(self.subCs.keys())
        Ncomp = len(subKeys)
        if self.allconstant:
            for loop in range(Ncomp):
                key = subKeys[loop]
                if self.SFH_list[key] == 'bins':
                    age = self.npSFH_times[key]
                    f_cont = jnp.ones(len(age)-1,dtype=jnp.float32)
                    for loop in range(len(age)-1):
                        f_cont = f_cont.at[loop].set(self.f_cont[key]['paradic']['bin{0}'.format(loop+1)])
                else:
                    age = self.ageparams[key]['paradic']['value']
                    f_cont = self.f_cont[key]['paradic']['value']
                Z = self.Zparams[key]['paradic']['value']
                Av = self.Avparams[key]['paradic']['value']
                # add stellar + dust SED
                if not self.energy_balance[key] :
                    fc = self.sedmodelist[key][0] * SEDs.get_host_SED(waveintrin, 0, f_cont, age, Z, Av, 1., self.SFH_list[key],self.currentage, self.AVbumplist[key])
                    fc += self.sedmodelist[key][2]*SEDs.get_cold_dust_sed(waveintrin, nebularpar['logMdust_{0}'.format(key)]-self.logMlist[key], nebularpar['qPAH_{0}'.format(key)], nebularpar['Umin_{0}'.format(key)], nebularpar['alphad_{0}'.format(key)], nebularpar['gammad_{0}'.format(key)])
                else:
                    fc, Latten = SEDs.get_host_SED_withL(waveintrin, 0, f_cont, age, Z, Av, 1., self.SFH_list[key],self.currentage, self.AVbumplist[key])
                    fc *= self.sedmodelist[key][0]
                    dustL = SEDs.get_cold_dust_L(nebularpar['qPAH_{0}'.format(key)], nebularpar['Umin_{0}'.format(key)], nebularpar['alphad_{0}'.format(key)], nebularpar['gammad_{0}'.format(key)])
                    nebularpar['logMdust_{0}'.format(key)] = self.logMlist[key] + jnp.log10(Latten)-dustL
                    fc += self.sedmodelist[key][2]*SEDs.get_cold_dust_sed(waveintrin, nebularpar['logMdust_{0}'.format(key)]-self.logMlist[key], nebularpar['qPAH_{0}'.format(key)], nebularpar['Umin_{0}'.format(key)], nebularpar['alphad_{0}'.format(key)], nebularpar['gammad_{0}'.format(key)])
                # add nebular SED
                normmap = self.sedmodelist[key][1] * SFR_to_Ha(self.SFR_map[key], Z)
                fnebular = self.Nebular.generate_continuum(waveintrin, self.logUlist[key], Z, Av)
                if not self.Nebular.withline:
                    fnebular += self.Nebular.generate_line(waveintrin, nebularpar)
                if self.table_model_dic[key] is not None:
                    fc += self.table_model_dic[key].get_sed(nebularpar, waveintrin,0.)
                _, obs_sed = SEDs.sed_to_obse(waveintrin, fc, self.redshift, self.ebv_G)
                _, obs_nebular = SEDs.sed_to_obse(waveintrin, fnebular, self.redshift, self.ebv_G)
                mass_cov = self.mass_map[key]
                tot_IFU += mass_cov[..., None] * obs_sed
                tot_IFU += normmap[..., None] * obs_nebular
        else:
            agemap = jnp.zeros((ny, nx, self.Nsub))
            Zmap = jnp.zeros((ny, nx, self.Nsub))
            f_contmap = jnp.zeros((ny, nx, self.Nsub))
            Avmap = jnp.zeros((ny, nx, self.Nsub))
            massmap = jnp.zeros((ny, nx, self.Nsub))
            def set_maps(i, _):
                key = subKeys[i]
                agemap_val = Cal_map(rmap, self.ageparams[key]['type'], self.ageparams[key]['paradic'])
                Zmap_val = Cal_map(rmap, self.Zparams[key]['type'], self.Zparams[key]['paradic'])
                f_cont_val = Cal_map(rmap, self.f_cont[key]['type'], self.f_cont[key]['paradic'])
                Av_val = Cal_map(rmap, self.Avparams[key]['type'], self.Avparams[key]['paradic'])
                mass_val = self.mass_map[key]
                return (agemap.at[:,:,i].set(agemap_val),
                        Zmap.at[:,:,i].set(Zmap_val),
                        f_contmap.at[:,:,i].set(f_cont_val),
                        Avmap.at[:,:,i].set(Av_val),
                        massmap.at[:,:,i].set(mass_val))
            def init_carry():
                return agemap, Zmap, f_contmap, Avmap, massmap
            def loop_body(i, carry):
                agemap_c, Zmap_c, f_contmap_c, Avmap_c, massmap_c = carry
                agemap_c, Zmap_c, f_contmap_c, Avmap_c, massmap_c = set_maps(i, None)
                return (agemap_c, Zmap_c, f_contmap_c, Avmap_c, massmap_c)
            agemap, Zmap, f_contmap, Avmap, massmap = lax.fori_loop(0, Ncomp, loop_body, init_carry())
            SFH_list = [self.SFH_list[k] for k in subKeys]
            Avbumplist = [self.AVbumplist[k] for k in subKeys]
            SFH_array = jnp.stack(SFH_list, axis=0)  
            Avbump_array = jnp.stack(Avbumplist, axis=0) 
            def compute_sed_per_component(inputs):
                loopy, loopx, ncom = inputs
                seds_total = SEDs.get_host_SED(
                    wave3D[loopy, loopx, :],
                    massmap[loopy, loopx, ncom],
                    f_contmap[loopy, loopx, ncom],
                    agemap[loopy, loopx, ncom],
                    Zmap[loopy, loopx, ncom],
                    Avmap[loopy, loopx, ncom],
                    SFH_array[ncom],        
                    self.currentage,
                    Avbump_array[ncom]
                )*self.sedmodelist[subKeys[ncom]][0]
                # add nebular SED
                normmap = self.sedmodelist[subKeys[ncom]][1] * SFR_to_Ha(SFH_array[ncom], Zmap[loopy, loopx, ncom])
                seds_total += normmap * self.Nebular.generate_continuum(wave3D[loopy, loopx, :], self.logUlist[subKeys[ncom]], Zmap[loopy, loopx, ncom], Avmap[loopy, loopx, ncom])
                if not self.Nebular.withline:
                    seds_total += self.Nebular.generate_line(wave3D[loopy, loopx, :], nebularpar)
                # add the dust emission
                seds_total += self.sedmodelist[subKeys[ncom]][2]*SEDs.get_cold_dust_sed(wave3D[loopy, loopx, :], nebularpar['logMdust_{0}'.format(subKeys[ncom])]-self.logMlist[subKeys[ncom]], nebularpar['qPAH_{0}'.format(subKeys[ncom])], nebularpar['Umin_{0}'.format(subKeys[ncom])], nebularpar['alphad_{0}'.format(subKeys[ncom])], nebularpar['gammad_{0}'.format(subKeys[ncom])])
                # add table model SED
                if self.table_model_dic[subKeys[ncom]] is not None:
                    seds_total += self.table_model_dic[subKeys[ncom]].get_sed(nebularpar, wave3D[loopy, loopx, :],0.)
                _, sed_obs = SEDs.sed_to_obse(waveintrin, seds_total, self.redshift, self.ebv_G)
                return sed_obs  
            def compute_sed_pixel(loopy, loopx):
                sed_all_components = jax.vmap(lambda ncom: compute_sed_per_component((loopy, loopx, ncom)))(jnp.arange(len(subKeys)))
                return jnp.sum(sed_all_components, axis=0)
            compute_sed_for_row = jax.vmap(lambda y: jax.vmap(lambda x: compute_sed_pixel(y, x))(jnp.arange(nx)))(jnp.arange(ny))
            tot_IFU = compute_sed_for_row
        tot_IFU *= self.rnorm
        return tot_IFU

    def generate_image_fromIFU(self, IFU, wavefull, resp, respbound):
        wavelenth = wavefull[respbound[0]:respbound[1]]
        filter_x = resp[:,0]
        filter_y = resp[:,1]
        full_resp = jnp.interp(wavelenth, filter_x, filter_y, left=0, right=0)
        IFU_resp = IFU[:, :, respbound[0]:respbound[1]] * full_resp
        image = jnp.trapezoid(IFU_resp, x=wavelenth, axis=2)
        return image

    def generate_dynamicalIFU(self, wavelength, impsf, allconst=True, nebularpar = None):
        ny, nx = self.imshape
        nw = len(wavelength)
        tot_IFU = jnp.zeros((ny, nx, nw))
        waveintrin = wavelength / (1. + self.redshift)
        wave3D = jnp.tile(waveintrin, (ny, nx, 1))
        rmap = self.r_map
        subKeys = list(self.subCs.keys())
        Ncomp = len(subKeys)
        if allconst:
            for loop in range(Ncomp):
                key = subKeys[loop]
                if self.SFH_list[key] == 'bins':
                    age = self.npSFH_times[key]
                    f_cont = jnp.ones(len(age)-1,dtype=jnp.float32)
                    for loop in range(len(age)-1):
                        f_cont = f_cont.at[loop].set(self.f_cont[key]['paradic']['bin{0}'.format(loop+1)])
                else:
                    age = self.ageparams[key]['paradic']['value']
                    f_cont = self.f_cont[key]['paradic']['value']
                Z = self.Zparams[key]['paradic']['value']
                Av = self.Avparams[key]['paradic']['value']
                fc = self.sedmodelist[key][0] * SEDs.get_host_SED(waveintrin, 0, f_cont, age, Z, Av, 1., self.SFH_list[key],self.currentage, self.AVbumplist[key])
                _, obs_sed = SEDs.sed_to_obse(waveintrin, fc, self.redshift, self.ebv_G)
                mass_cov = jax.scipy.signal.fftconvolve(self.mass_map[key],impsf,mode='same')
                tot_IFU += mass_cov[..., None] * obs_sed
        else:
            agemap = jnp.zeros((ny, nx, self.Nsub))
            Zmap = jnp.zeros((ny, nx, self.Nsub))
            f_contmap = jnp.zeros((ny, nx, self.Nsub))
            Avmap = jnp.zeros((ny, nx, self.Nsub))
            massmap = jnp.zeros((ny, nx, self.Nsub))
            def set_maps(i, _):
                key = subKeys[i]
                agemap_val = Cal_map(rmap, self.ageparams[key]['type'], self.ageparams[key]['paradic'])
                Zmap_val = Cal_map(rmap, self.Zparams[key]['type'], self.Zparams[key]['paradic'])
                f_cont_val = Cal_map(rmap, self.f_cont[key]['type'], self.f_cont[key]['paradic'])
                Av_val = Cal_map(rmap, self.Avparams[key]['type'], self.Avparams[key]['paradic'])
                mass_val = self.mass_map[key]
                return (agemap.at[:,:,i].set(agemap_val),
                        Zmap.at[:,:,i].set(Zmap_val),
                        f_contmap.at[:,:,i].set(f_cont_val),
                        Avmap.at[:,:,i].set(Av_val),
                        massmap.at[:,:,i].set(mass_val))
            def init_carry():
                return agemap, Zmap, f_contmap, Avmap, massmap
            def loop_body(i, carry):
                agemap_c, Zmap_c, f_contmap_c, Avmap_c, massmap_c = carry
                agemap_c, Zmap_c, f_contmap_c, Avmap_c, massmap_c = set_maps(i, None)
                return (agemap_c, Zmap_c, f_contmap_c, Avmap_c, massmap_c)
            agemap, Zmap, f_contmap, Avmap, massmap = lax.fori_loop(0, Ncomp, loop_body, init_carry())
            SFH_list = [self.SFH_list[k] for k in subKeys]
            Avbumplist = [self.AVbumplist[k] for k in subKeys]
            SFH_array = jnp.stack(SFH_list, axis=0)  
            Avbump_array = jnp.stack(Avbumplist, axis=0) 
            def compute_sed_per_component(inputs):
                loopy, loopx, ncom = inputs
                seds_total = SEDs.get_host_SED(
                    wave3D[loopy, loopx, :],
                    massmap[loopy, loopx, ncom],
                    f_contmap[loopy, loopx, ncom],
                    agemap[loopy, loopx, ncom],
                    Zmap[loopy, loopx, ncom],
                    Avmap[loopy, loopx, ncom],
                    SFH_array[ncom],        
                    self.currentage,
                    Avbump_array[ncom]
                )
                _, sed_obs = SEDs.sed_to_obse(waveintrin, seds_total, self.redshift, self.ebv_G)
                return sed_obs  
            def compute_sed_pixel(loopy, loopx):
                sed_all_components = jax.vmap(lambda ncom: compute_sed_per_component((loopy, loopx, ncom)))(jnp.arange(len(subKeys)))
                return jnp.sum(sed_all_components, axis=0)
            compute_sed_for_row = jax.vmap(lambda y: jax.vmap(lambda x: compute_sed_pixel(y, x))(jnp.arange(nx)))(jnp.arange(ny))
            tot_IFU = compute_sed_for_row

        sigmap, volmap = self.cal_dynamics(paradic = nebularpar)
        vrmap = volmap * self.rot_scale
        wave3D_rest = wave3D/(1.+volmap[..., None]/C.ckm)
        SFR_map = self.generate_SFR_map()
        tot_IFU += self.Nebular.generate_line3D(wave3D_rest,nebularpar,self.massimg,SFR_map,sigmap)

        # multiplying the spread function (difficulty)

        return tot_IFU*self.rnorm
    


class ISM(object):
    
    """
    Interstellar medium model for nebular emission (continuum and lines).
    
    The ISM class handles nebular emission from ionized gas, including both
    continuum emission (free-free, free-bound, two-photon) and emission lines
    (H, He, metals). It can model emission from both narrow-line regions (NLR)
    and broad-line regions (BLR) in AGN.
    
    Parameters
    ----------
    withlines : bool, optional
        If True, nebular lines are included in the continuum model (single component).
        If False, lines are modeled separately with the Line_Emitter. Default is True.
    lines : list, optional
        List of emission line dictionaries for separate line modeling.
        Each dict should have 'name' and 'wave' keys. Default is [].
    nnarrow : int, optional
        Number of kinematic components for narrow lines. Default is 1.
    prefix : str, optional
        Prefix for parameter names (e.g., 'nb' for narrow, 'bb' for broad).
        Default is 'nb'.
    D3 : bool, optional
        If True, use 3D modeling for spatially resolved IFU data. Default is False.
    
    Attributes
    ----------
    withline : bool
        Whether lines are included in continuum.
    D3 : bool
        Whether 3D modeling is enabled.
    nnarrow : int
        Number of kinematic components.
    lineemit : Line_Emitter or None
        Line emitter object for separate line modeling.
    
    Examples
    --------
    >>> from galfits.galaxy import ISM
    >>> from galfits import emission_lines as EL
    >>> # Simple continuum + lines (integrated in SED models)
    >>> ism_simple = ISM(withlines=True)
    >>> 
    >>> # Separate line modeling for kinematics
    >>> lines = [EL.Ha, EL.NII_6583, EL.OIII_5007]
    >>> ism_lines = ISM(withlines=False, lines=lines, nnarrow=2)
    
    Notes
    -----
    The ISM class provides two modes:
    
    1. **withlines=True**: Nebular continuum and lines are computed together
       from photoionization models. Suitable for broadband SED fitting.
    
    2. **withlines=False**: Lines are modeled separately as Gaussians with
       adjustable kinematics. Suitable for spectroscopic fitting where line
       profiles contain kinematic information.
    
    Nebular continuum includes:
    - Free-free emission (Bremsstrahlung)
    - Free-bound emission (recombination continua)
    - Two-photon emission
    - Balmer continuum
    
    The emission depends on ionization parameter (U), metallicity (Z), and
    dust extinction (Av).
    
    See Also
    --------
    Line_Emitter : Handles kinematic modeling of emission lines.
    Galaxy.make_nebular : Initialize nebular emission for a Galaxy object.
    """
    def __init__(self, withlines = True, lines = [], nnarrow = 1, prefix='nb', D3 = False):
        self.withline = withlines
        self.D3 = D3
        self.nnarrow = nnarrow
        if not withlines:
            self.lines = lines
            self.prefix = prefix
            self.lineemit = Line_Emitter(lines, nnarrow, prefix)
        pass

    def model_initialized(self, restspec, strict=0.01):
        return self.lineemit.model_initialized(restspec, narrow=True, strict=strict)

    def model_initialized_3D(self):
        return self.lineemit.model_initialized_3D()

    def generate_continuum(self, wave, logU, Z, Av):
        return SEDs.get_nebular(wave, 1, logU, Z, Av, withline = self.withline)
    
    def generate_line(self, wave, pardict):
        return self.lineemit(wave, pardict)
    
    def generate_line3D(self, wave3D, pardict, massmap, SFRmap, sigmap):
        return self.lineemit.get_3Dflux(wave3D, pardict, massmap, SFRmap, sigmap)

class Line_Emitter(object):
    """
    Emission line profile modeler for kinematic analysis.
    
    This class models emission lines as Gaussian profiles with multiple kinematic
    components. It handles line ratios, velocity offsets, and widths, making it
    suitable for spectroscopic fitting of galaxies and AGN where line profiles
    contain kinematic information (rotation, outflows, etc.).
    
    Parameters
    ----------
    lines : list
        List of emission line dictionaries. Each dictionary should contain:
        
        - 'name' : str - Line identifier (e.g., 'Ha', 'OIII_5007')
        - 'wave' : float - Rest wavelength in Angstroms
    
    ncomp : int, optional
        Number of kinematic components (e.g., 1=single component, 2=core+wing).
        Default is 1.
    prefix : str, optional
        Prefix for parameter names to distinguish different line regions
        (e.g., 'b' for broad, 'n' for narrow). Default is 'b'.
    
    Attributes
    ----------
    lines : list
        List of emission line dictionaries.
    ncomp : int
        Number of kinematic components.
    prefix : str
        Parameter name prefix.
    nprefix : dict or None
        Reference line for tying kinematics across multiple lines.
    
    Notes
    -----
    The Line_Emitter models each line as a sum of Gaussian components:
    
    .. math::
        F(\\lambda) = \\sum_{i=1}^{N_{comp}} \\frac{A_i}{\\sqrt{2\\pi}\\sigma_i} 
                      \\exp\\left(-\\frac{(\\lambda - \\lambda_{c,i})^2}{2\\sigma_i^2}\\right)
    
    Where:
    - A_i is the integrated line flux
    - λ_c,i is the line center (can be offset from rest wavelength)
    - σ_i is the Gaussian width (related to velocity dispersion)
    
    Key features:
    
    - Automatic line ratio constraints (e.g., [OIII]5007/4959 = 2.98)
    - Tied kinematics across lines (same velocity, dispersion)
    - Multiple kinematic components (narrow + broad)
    - Support for 3D IFU modeling with spatially varying kinematics
    
    Examples
    --------
    >>> from galfits.galaxy import Line_Emitter
    >>> from galfits import emission_lines as EL
    >>> lines = [EL.Ha, EL.NII_6583, EL.NII_6549, EL.SII_6717, EL.SII_6731]
    >>> emitter = Line_Emitter(lines=lines, ncomp=2, prefix='n')
    >>> # Initialize parameters from an observed spectrum
    >>> params = emitter.model_initialized(restspec, narrow=True)
    
    See Also
    --------
    ISM : Parent class that uses Line_Emitter for nebular modeling.
    """
    def __init__(self, lines = [], ncomp = 1, prefix='b'):
        self.lines = lines
        self.ncomp = ncomp
        self.prefix = prefix
        self.nprefix = None

    def model_initialized(self, restspec, narrow=False, strict=0.01, broader=12000.,  bcentershift = 0., havespec=True):
        par = lmfit.Parameters()
        prefix = self.prefix
        nprefix=None
        for loopline, line in enumerate(self.lines):
            lp = jnp.abs(jnp.interp(line['wave'],restspec[0],restspec[1]))
            if not narrow: 
                wguess = 2500.
                if not havespec:
                    wguess = 7000.
                for loop in range(self.ncomp):
                    if loop ==0:
                        par.add('{0}{1}{2}wid'.format(line['name'],prefix,loop+1),value=line['wave']*wguess/C.ckm/C.tl2,min=line['wave']*1200./C.ckm/C.tl2,max=line['wave']*broader/C.ckm/C.tl2,brute_step=0.1,vary=havespec)
                        par.add('{0}{1}{2}peak'.format(line['name'],prefix,loop+1),value=lp,min=0.,max=25*lp,brute_step=0.01*lp)
                        par.add('{0}{1}{2}cen'.format(line['name'],prefix,loop+1),value=line['wave'],min=line['wave']-50.,max=line['wave']+50.,brute_step=0.1,vary=havespec)
                    else:
                        censhi = (bcentershift*loop)/(self.ncomp-1)
                        par.add('{0}{1}{2}wid'.format(line['name'],prefix,loop+1),value=line['wave']*5000./C.ckm/C.tl2,min=line['wave']*2000./C.ckm/C.tl2,max=line['wave']*broader/C.ckm/C.tl2,brute_step=0.1,vary=havespec)
                        par.add('{0}{1}{2}peak'.format(line['name'],prefix,loop+1),value=0.1*lp,min=0.,max=25*lp,brute_step=0.01*lp)
                        par.add('{0}{1}{2}cen'.format(line['name'],prefix,loop+1),value=line['wave']+censhi,min=line['wave']+censhi-200.,max=line['wave']+censhi+200.,brute_step=0.1,vary=havespec)
            else:
                if nprefix is None:
                    nprefix=line
                    self.nprefix = line
                if loopline == 0:
                    par.add('{0}{1}1cen'.format(line['name'],prefix),value=line['wave'],min=(1.-strict)*line['wave'],max=(1.+strict)*line['wave'],brute_step=0.1,vary=havespec)
                    par.add('{0}{1}1wid'.format(line['name'],prefix),value=5.,min=line['wave']*100./C.ckm/C.tl2,max=line['wave']*1060./C.ckm/C.tl2,brute_step=0.1,vary=havespec)
                    par.add('{0}{1}1peak'.format(line['name'],prefix),value=lp,min=0.,max=100*lp,brute_step=0.01*lp)
                    for loop in range(self.ncomp-1):
                        par.add('{0}{1}{2}cen'.format(line['name'],prefix,loop+2),value=line['wave']-5.*(loop+1),min=line['wave']-35.,max=line['wave']+20.,brute_step=0.1,vary=havespec)
                        par.add('{0}{1}{2}wid'.format(line['name'],prefix,loop+2),value=line['wave']*1500./C.ckm/C.tl2,min=line['wave']*200./C.ckm/C.tl2,max=line['wave']*2500./C.ckm/C.tl2,brute_step=0.1,vary=havespec)
                        par.add('{0}{1}{2}peak'.format(line['name'],prefix,loop+2),value=1.*lp/(0.8**(loop+1)),min=0.,max=10*lp,brute_step=0.01*lp)
                else:
                    for loop in range(self.ncomp):
                        par.add('{0}{1}{2}wid'.format(line['name'],prefix,loop+1),expr='{0}*{1}{2}{3}wid'.format(line['wave']/nprefix['wave'],nprefix['name'],prefix,loop+1),min=line['wave']*200./C.ckm/C.tl2,max=line['wave']*2500./C.ckm/C.tl2,brute_step=0.1)
                        par.add('{0}{1}{2}cen'.format(line['name'],prefix,loop+1),expr='{0}*{1}{2}{3}cen'.format(line['wave']/nprefix['wave'],nprefix['name'],prefix,loop+1),min=line['wave']-35.,max=line['wave']+20.,brute_step=0.1)
                        if loop == 0:
                            par.add('{0}{1}{2}peak'.format(line['name'],prefix,loop+1),value=lp,min=0.,max=100*lp,brute_step=0.01*lp)
                        else:
                            par.add('{0}{1}{2}peak'.format(line['name'],prefix,loop+1),expr='1.*{0}{1}{2}peak*{3}{1}1peak/{0}{1}1peak'.format(nprefix['name'],prefix,loop+1,line['name']),min=0.,max=100*lp,brute_step=0.01*lp)
        for loopline, line in enumerate(self.lines):
            for loop in range(self.ncomp):
                if loop == 0:
                    if line['name'] == EL.OIII_4959['name']:
                        par['{0}{1}{2}peak'.format(EL.OIII_4959['name'],prefix,loop+1)].set(expr='0.33557*{0}{1}{2}peak'.format(EL.OIII_5007['name'],prefix,loop+1),min=0.,max=100*lp,brute_step=0.01*lp)
                    elif line['name'] == EL.NII_6549['name']:
                        par['{0}{1}{2}peak'.format(EL.NII_6549['name'],prefix,loop+1)].set(expr='0.337838*{0}{1}{2}peak'.format(EL.NII_6583['name'],prefix,loop+1),min=0.,max=100*lp,brute_step=0.01*lp)
        return par

    def model_initialized_3D(self, logLs0 = None, initfac = 1, varyfrac = False, varysig = False):
        if logLs0 is None:
            logLs0 = jnp.zeros(len(self.lines)) + 39.
        par = lmfit.Parameters()
        prefix = self.prefix
        for loopline, line in enumerate(self.lines):
            lp = logLs0[loopline]
            par.add('{0}{1}frac'.format(line['name'],prefix),value=initfac,min=0,max=1,brute_step=0.1, vary=varyfrac)
            par.add('{0}{1}fsig'.format(line['name'],prefix),value=1,min=0.1,max=10,brute_step=0.01, vary=varysig)
            for loop in range(self.ncomp):
                par.add('{0}{1}{2}logL'.format(line['name'],prefix,loop+1),value=lp,min=lp-4,max=lp+4,brute_step=0.05)           
        return par

    def __call__(self, wave, pardict):
        flux = jnp.zeros_like(wave)
        prefix = self.prefix
        for loopline, line in enumerate(self.lines):
            for loop in range(self.ncomp):
                flux += SEDs.gaussian_jax(wave,pardict['{0}{1}{2}peak'.format(line['name'],prefix,loop+1)],pardict['{0}{1}{2}cen'.format(line['name'],prefix,loop+1)],pardict['{0}{1}{2}wid'.format(line['name'],prefix,loop+1)])
        return flux
    
    def get_3Dflux(self, wave3D, pardict, massmap, SFRmap, sigmap):
        prefix = self.prefix
        MIFU = jnp.zeros_like(wave3D, dtype=jnp.float32)
        norm_mass = massmap/(jnp.sum(massmap))
        norm_SFR = SFRmap/(jnp.sum(SFRmap))
        for loopline, line in enumerate(self.lines):
            normmap = pardict['{0}{1}frac'.format(line['name'],prefix)]*norm_SFR + (1.-pardict['{0}{1}frac'.format(line['name'],prefix)])*norm_mass
            sigmapp = sigmap * pardict['{0}{1}fsig'.format(line['name'],prefix)] * line['wave']/C.ckm
            for loop in range(self.ncomp):
                MIFU += 10**(pardict['{0}{1}{2}logL'.format(line['name'],prefix,loop+1)]-38)*SEDs.gaussianL_jax3D(wave3D,normmap[...,None],line['wave'],sigmapp[...,None])
        return MIFU

class AGN(object):
    """
    Active Galactic Nucleus (AGN) spectral and imaging model.
    
    Comprehensive AGN modeling including accretion disk continuum, broad and
    narrow emission lines, FeII emission, Balmer continuum, torus emission,
    and hot dust. Supports multiple disk models and custom table models.
    
    Parameters
    ----------
    regionz : float
        Redshift of the observed region (may differ from AGN redshift for
        gravitationally lensed systems).
    logM_BH : float, optional
        Log10 of black hole mass in solar masses (default: 8.0).
        Range typically [5, 10].
    logLedd : float, optional
        Log10 of Eddington ratio L_bol/L_Edd (default: -1.0).
        Range typically [-3, 0] for type 1 AGN.
    astar : float, optional
        Black hole spin parameter (0 = Schwarzschild, 0.998 = maximal).
        Default: 0.0. Range [0, 0.998].
    Av : float, optional
        V-band attenuation in magnitudes (default: 0.0).
    z : float, optional
        AGN redshift (default: 0.0).
    ebv_G : float, optional
        Galactic E(B-V) color excess for foreground extinction (default: 0.0).
    torus : bool, optional
        If True, include CAT3D torus emission model (default: False).
    prefix : str, optional
        Name prefix for parameters (default: ''). Used to distinguish
        multiple AGN components.
    
    Attributes
    ----------
    logM_BH : float
        Black hole mass (log10 M_sun).
    logLedd : float
        Eddington ratio (log10 L_bol/L_Edd).
    astar : float
        Spin parameter.
    Av : float
        Internal dust attenuation.
    redshift : float
        AGN redshift.
    ebv_G : float
        Galactic extinction.
    BLR : Line_Emitter or None
        Broad-line region emission model.
    NLR : Line_Emitter or None
        Narrow-line region emission model.
    FeII : bool
        Include FeII emission if True.
    BaC : bool
        Include Balmer continuum if True.
    sedModel : str
        SED model type: 'TD' (thin disk), 'SP' (single power-law),
        'BP' (broken power-law), 'SBD' (starburst disk), 'table', 'NP' (non-parametric).
    torus : bool
        Include torus emission if True.
    hotdust : bool
        Include hot dust blackbody if True.
    table_model : TableModel or None
        Custom SED table model if configured.
    rnorm : float
        Normalization factor for flux scaling: (D_L,region / D_L,AGN)^2.
    
    Methods
    -------
    reset_BH(logM_BH, logLedd, astar, Av)
        Update black hole parameters.
    reset_info(z, ebv_G)
        Update redshift and Galactic extinction.
    change_SED(sedmodel)
        Switch between disk models.
    set_full_model(obsspec, lines_broad, lines_narrow, par_tot, ...)
        Initialize BLR, NLR, FeII, and BaC from observed spectrum.
    generate_image(posi, shape, band, psfimage, resp, ...)
        Create 2D AGN image in specified band.
    fiducial_sed(wavelength, pardict, delta)
        Generate AGN SED at given parameters.
    get_component_SED(wavelength, pardict, delta)
        Extract individual SED components (disk, BLR, NLR, FeII, etc.).
    
    Notes
    -----
    The AGN class is designed to be combined with Galaxy class for
    host-AGN decomposition. It handles both unresolved (point source)
    and resolved (extended NLR) emission.
    
    **SED models:**
    
    - 'TD': Thin accretion disk (Shakura-Sunyaev with relativistic corrections)
    - 'SP': Single power-law (λ^-α)
    - 'BP': Broken power-law (different slopes UV vs optical)
    - 'SBD': Starburst disk (optically thick, dust-obscured)
    - 'table': Custom table model
    - 'NP': Non-parametric (directly fit fluxes in bands)
    
    **Emission components:**
    
    - Accretion disk continuum
    - Broad-line region (BLR): H, He recombination lines with FWHM ~ 1000-10000 km/s
    - Narrow-line region (NLR): Forbidden lines with FWHM ~ 100-500 km/s
    - FeII emission: Complex blended iron transitions
    - Balmer continuum: Recombination continuum from BLR
    - Torus: IR emission from dusty torus (1-100 μm)
    - Hot dust: Near-IR blackbody (1-3 μm)
    
    Examples
    --------
    >>> # Create AGN with thin disk model
    >>> agn = AGN(regionz=2.0, logM_BH=8.5, logLedd=-0.5, astar=0.7,
    ...           z=2.0, ebv_G=0.03, torus=True, prefix='agn')
    
    >>> # Initialize emission line models from observed spectrum
    >>> wave_obs = jnp.linspace(3000, 9000, 1000)
    >>> flux_obs = observed_spectrum
    >>> lines_broad = ['Ha', 'Hb', 'MgII2798']
    >>> lines_narrow = ['[OIII]5007', '[NII]6584']
    >>> params = lmfit.Parameters()
    >>> agn.set_full_model([wave_obs, flux_obs], lines_broad, lines_narrow,
    ...                    params, addFeII=True, addBaC=True)
    
    >>> # Generate SED
    >>> wave = jnp.logspace(2, 5, 1000)
    >>> pardict = {param.name: param.value for param in params.values()}
    >>> sed = agn.fiducial_sed(wave, pardict)
    
    >>> # Generate point-source image
    >>> posi = [50.0, 50.0]  # pixel coordinates
    >>> shape = (100, 100)
    >>> img = agn.generate_image(posi, shape, 'F200W', psf, response)
    
    See Also
    --------
    Galaxy : Host galaxy modeling
    Line_Emitter : Emission line fitting
    cat3Dtorus : AGN torus emission
    get_AGN_SED : Thin disk SED interpolation
    
    References
    ----------
    .. [1] Shakura & Sunyaev 1973, A&A, 24, 337 (thin disk)
    .. [2] Hoenig & Kishimoto 2017, ApJ, 838, 94 (CAT3D torus)
    .. [3] Vestergaard & Wilkes 2001, ApJS, 134, 1 (FeII templates)
    """
    def __init__(self,regionz,logM_BH=8.,logLedd=-1.,astar=0., Av=0., z=0., ebv_G=0., torus = False, prefix=''):
        '''
        galaxy object is initialed from a given mass
        '''
        self.logM_BH = logM_BH
        self.logLedd=logLedd
        self.astar = astar
        self.Av = Av
        self.redshift = z
        self.ebv_G = ebv_G
        self.BLR = None
        self.NLR = None
        self.FeII = False
        self.BaC =  False
        self.sedModel = 'TD'
        self.prefix = prefix
        self.torus = torus
        self.bandslist = []
        self.iwaves = None
        self.hotdust = False
        self.name = prefix
        self.table_model = None
        self.dl0 = SEDs.luminosity_distance(z)/(1+z)
        self.dlr = SEDs.luminosity_distance(regionz)/(1+regionz)
        self.rnorm = (self.dlr/self.dl0)**2

    def __del__(self):
        # Nullify object's properties
        self.logM_BH = None
        self.logLedd = None
        self.astar = None
        self.Av = None
        self.redshift = None
        self.ebv_G = None
        self.BLR = None
        self.NLR = None
        self.FeII = None
        self.BaC = None
        self.sedModel = None
        self.prefix = None
        self.rnorm = None
        self.torus = None
        self.sortwave = None
        self.table_model = None

    def reset_BH(self, logM_BH,logLedd, astar, Av):
        self.logM_BH = logM_BH
        self.logLedd=logLedd
        self.astar = astar
        self.Av = Av
        return
    
    def reset_info(self, z=0., ebv_G=0.):
        '''
        reset the z of a agn object
        '''
        dnew=SEDs.luminosity_distance(z)/(1+z)
        self.rnorm = (self.dlr/dnew)**2
        self.redshift = z
        self.ebv_G = ebv_G

    def change_SED(self, sedmodel):
        self.sedModel = sedmodel

    def set_full_model(self, obsspec, lines_broad, lines_narrow, par_tot, addFeII = True,  addBaC = True, nbroad=2, nnarrow=1, strict=0.01,broader=12000., havespec=False, **kwargs):
        '''
        set the BLR NLR models for the AGN component
        ------
        obsspec: observed optical spectrum [wave,flux]
            it is used to make an initial guess
        lines_broad: list
            a line list in BLR
        lines_narrow: list
            a line list in NLR
        ------
        '''
        prefix = self.prefix

        if ('bcentershift' in kwargs.keys()):
            bcentershift = kwargs['bcentershift']
        else:
            bcentershift = 0.
        if ('stdhb' in kwargs.keys()):
            stdhb = kwargs['stdhb']
        else:
            stdhb = 100.
        if addFeII:
            self.FeII = True
            par_tot.add('FeII{0}A_uv'.format(prefix),value=0.7*stdhb/4000.,min=0.7*stdhb/400000.,max=0.7*stdhb/4.,brute_step=0.01*stdhb/4000.)
            par_tot.add('FeII{0}A_op'.format(prefix),value=stdhb/4000.,min=stdhb/400000.,max=stdhb/4.,brute_step=0.01*stdhb/4000.)
            par_tot.add('FeII{0}fwhm'.format(prefix),value=2000.,min=900.,max=9000.,brute_step=100.,vary=havespec)
            par_tot.add('FeII{0}dcen'.format(prefix),value=0.,min=-3000.,max=3000.,brute_step=100.,vary=havespec)
        if addBaC:
            self.BaC = True
            par_tot.add('BaC{0}logL3600'.format(prefix),value = 40,min=36,max=44,brute_step=0.01)
            par_tot.add('BaC{0}logM'.format(prefix),value=7,min=5,max=9,brute_step=0.01,vary=False)
            par_tot.add('BaC{0}logMdot'.format(prefix),value=-1,min=-3,max=2,brute_step=0.01,vary=False)
            par_tot.add('BaC{0}spin'.format(prefix),value=0,min=0,max=0.998,brute_step=0.01,vary=False)
            par_tot.add('BaC{0}dcen'.format(prefix), value=0.,min=-3000.,max=3000.,brute_step=100.,vary=havespec)
            par_tot.add('BaC{0}fwhm'.format(prefix),value = 2000.,min=900.,max=9000.,brute_step=100.,vary=havespec)
        x, sed_obs = SEDs.sed_to_rest(obsspec[0],obsspec[1],self.redshift,self.ebv_G)
        restspec = [x,sed_obs]
        if len(lines_broad) > 0:
            self.BLR = Line_Emitter(lines_broad, nbroad, prefix+'b')
            par_tot.update(self.BLR.model_initialized(restspec, narrow=False, strict=strict, broader=broader,  bcentershift = bcentershift,havespec=havespec))
        if len(lines_narrow) > 0:
            self.NLR = Line_Emitter(lines_narrow, nnarrow, prefix+'n')
            par_tot.update(self.NLR.model_initialized(restspec, narrow=True, strict=strict, broader=broader,  bcentershift = bcentershift,havespec=havespec))
        return par_tot

    def generate_image(self, posi, shape, band, psfimage, resp, wavegrid=None, transpar=None, pardict=None, psfimscale = 1.):
        '''
        Parameters:
        shape: (y,x) of the output image

        band: band of the output image

        {psftype: [psfparams]}: a dict, the point spread function
        eg.  {'psf': [{'xcen':50, 'ycen':50}]}     stands for a point sources which have same shape as the empirical PSF
             {'moffat': [{'xcen':50, 'ycen':50, 'fwhm':3., 'con':'5.'}]}
        '''
        ny = shape[0]
        nx = shape[1]
        prefix = self.prefix

        if self.sedModel == 'NP':
            flux_band = 10**(pardict['{0}_{1}_logL'.format(prefix,band)] - 38) * self.rnorm #luminosity_\lambda_obs
        else:
            waveintrin  = wavegrid / (1 + self.redshift)
            agnsed      = self.fiducial_sed(waveintrin, pardict)
            flux_band   = self.generate_SED([resp], wavegrid, agnsed)[0]
        if nx*ny == 1:
            agnimg  = flux_band * jnp.ones((ny,nx), dtype=jnp.float32)
        else:
            if transpar is None:
                xpix,ypix = indentify_xy(posi[0], posi[1])
            else:
                xpix,ypix = coordinates_transfer(posi[0], posi[1], transpar)
                
            agnimg = Pro.add_PSF(xpix, ypix, nx, ny, psfimage, scalefactor=psfimscale) * flux_band

        return agnimg


    def fiducial_sed(self, waveintrin, pardict=None, delta=0.):
        """
        Computes the AGN spectral energy distribution (SED) at rest-frame wavelengths given a set of model parameters.

        Parameters:
        -----------
        wavelength : numpy.ndarray
            Array of wavelengths in Angstroms.
        pardict : dict, optional
            Dictionary of model parameters. If None, the default parameters are used.
        delta : float, optional
            2175 bump strength. Default is 0.

        Returns:
        --------
        agnsed : numpy.ndarray
            Array of AGN SED values in units of 1e38 erg/s/A at the observer frame.
        """
        prefix = self.prefix
        # waveintrin = wavegrid/(1.+self.redshift)
        wavegrid    = waveintrin * (1+self.redshift)
        agnsed_rest = jnp.zeros_like(waveintrin,dtype=jnp.float32)
        if self.sedModel == 'TD':
            agnsed_rest += SEDs.get_AGN_SED(waveintrin,pardict['agn_logM{0}'.format(prefix)],pardict['agn_logLedd{0}'.format(prefix)],pardict['agn_spin{0}'.format(prefix)],1.)
        elif self.sedModel == 'SP':
            agnsed_rest += SEDs.spowerlaw(waveintrin,10**(pardict['{0}logL5100'.format(prefix)]-38),pardict['{0}plC'.format(prefix)])
        elif self.sedModel == 'BP':
            agnsed_rest += SEDs.brokenpower(waveintrin,10**(pardict['{0}logL5100'.format(prefix)]-38),pardict['{0}plC1'.format(prefix)],pardict['{0}plC2'.format(prefix)])
        elif self.sedModel == 'NP':
            fluxes =[]
            for band in self.bandslist:
                fluxes.append(10**(pardict['{0}_{1}_logL'.format(prefix,band)]-38))
            fluxes = jnp.array(fluxes)
            return jnp.interp(wavegrid,self.iwaves,fluxes,)*self.rnorm # linear 
        elif self.sedModel == 'SBD':
            agnsed_rest += SEDs.starburst_disk(waveintrin,pardict['agn_logM{0}'.format(prefix)],pardict['agn_logLedd{0}'.format(prefix)],pardict['agn_amax{0}'.format(prefix)])
        elif self.sedModel == 'table':
            agnsed_rest += self.table_model.get_sed(tuple(pardict[self.prefix+name] for name in self.table_model.param_names)+(waveintrin,),norm_key=pardict['{0}logL5100'.format(prefix)])
        if self.BLR is not None:
            agnsed_rest += self.BLR(waveintrin,pardict)
        if self.NLR is not None:
            agnsed_rest += self.NLR(waveintrin,pardict)
        if self.FeII:
            agnsed_rest += SEDs.FeII(waveintrin,pardict['FeII{0}A_uv'.format(prefix)],pardict['FeII{0}A_op'.format(prefix)],pardict['FeII{0}dcen'.format(prefix)],pardict['FeII{0}fwhm'.format(prefix)])
        if self.BaC:
            agnsed_rest += SEDs.BaC(waveintrin,pardict['BaC{0}logL3600'.format(prefix)],pardict['BaC{0}logM'.format(prefix)],pardict['BaC{0}logMdot'.format(prefix)],pardict['BaC{0}spin'.format(prefix)],pardict['BaC{0}dcen'.format(prefix)],pardict['BaC{0}fwhm'.format(prefix)])
        if self.torus:
            agnsed_rest += SEDs.cat3Dtorus(waveintrin,10.0**(pardict['{0}torus_logL'.format(prefix)]-38), pardict['{0}torus_a'.format(prefix)], pardict['{0}torus_h'.format(prefix)], pardict['{0}torus_N0'.format(prefix)], pardict['{0}torus_i'.format(prefix)])
        if self.hotdust:
            agnsed_rest += SEDs.Bbody(waveintrin,10.0**(pardict['{0}Lhotdust'.format(prefix)]-38),jnp.array(pardict['{0}Thotdust'.format(prefix)],dtype=jnp.float32))
        
        #E(B-V) = Av/Rv, Rv assumed to be 3.1 (MW value, see Calzetti et al. 2000)
        # TODO: How to understand this?
        cm=SEDs.mycigale(waveintrin,delta)*pardict['agn_Av{0}'.format(prefix)]/3.1/2.5
        agnsed_rest /= 10**cm
        x,agnsed = SEDs.sed_to_obse(waveintrin,agnsed_rest,self.redshift,self.ebv_G)
        return agnsed*self.rnorm
    
    def generate_SED(self, resps, wavegrid, fll_tot):
        # wavelength = self.sortwave   
        '''
        keep same format with Galaxy class.
        '''
        SEDs_array = []
        for resp in resps:
            filterx     = resp[:,0]
            filtery     = resp[:,1]
            filtery_intp    = jnp.interp(wavegrid, filterx, filtery, left=0., right=0.)
            ax  = jnp.trapezoid(filtery_intp, x=wavegrid)
            flux_band   = jnp.trapezoid(filtery_intp*fll_tot, x=wavegrid) / ax
            SEDs_array.append(flux_band)
        SEDs_array = jnp.array(SEDs_array,dtype=jnp.float32)
        return SEDs_array

    def get_component_SED(self, wavegrid, pardict=None,delta=0.):
        """
        Returns the spectral energy distribution (SED) of the galaxy's components at a given wavelength.
        NOTE: don't used in galfits pipeline.
        Args:
            wavelength (float): The wavelength at which to calculate the SED.
            pardict (dict, optional): A dictionary of parameters for the SED model. Defaults to None.
            delta (float, optional): 2175 A bump amplitude. Defaults to 0.

        Returns:
            tuple: A tuple containing two lists.
            The first list contains the SEDs of the galaxy's components at the given wavelength. 
            The second list contains the names of the components.
        """
        prefix = self.prefix
        waveintrin = wavegrid/(1.+self.redshift)
        agnsed_rest = []
        sedcomp = []
        if self.sedModel == 'TD':
            agnsed_rest.append(SEDs.get_AGN_SED(waveintrin,pardict['agn_logM{0}'.format(prefix)],pardict['agn_logLedd{0}'.format(prefix)],pardict['agn_spin{0}'.format(prefix)],1.))
            sedcomp.append('Disk')
        elif self.sedModel == 'SP':
            agnsed_rest.append(SEDs.spowerlaw(waveintrin,10**(pardict['{0}logL5100'.format(prefix)]-38),pardict['{0}plC'.format(prefix)]))
            sedcomp.append('Disk')
        elif self.sedModel == 'BP':
            agnsed_rest.append(SEDs.brokenpower(waveintrin,10**(pardict['{0}logL5100'.format(prefix)]-38),pardict['{0}plC1'.format(prefix)],pardict['{0}plC2'.format(prefix)]))
            sedcomp.append('Disk')
        elif self.sedModel == 'NP':
            fluxes =[]
            for band in self.bandslist:
                fluxes.append(10**(pardict['{0}_{1}_logL'.format(prefix,band)]-38))
            fluxes = jnp.array(fluxes)
            spec = jnp.interp(wavegrid, self.iwaves, fluxes)*self.rnorm
            agnsed_rest.append(spec)
            sedcomp.append('Total')
            return agnsed_rest, sedcomp
        elif self.sedModel == 'table':
            agnsed_rest.append(self.table_model.get_sed(tuple(pardict[self.prefix+name] for name in self.table_model.param_names)+(waveintrin,),norm_key=pardict['{0}logL5100'.format(prefix)]))
            sedcomp.append('Disk')
        if self.BLR is not None:
            agnsed_rest.append(self.BLR(waveintrin,pardict))
            sedcomp.append('Broad lines')
        if self.NLR is not None:
            agnsed_rest.append(self.NLR(waveintrin,pardict))
            sedcomp.append('Narrow lines')
        if self.FeII:
            agnsed_rest.append(SEDs.FeII(waveintrin,pardict['FeII{0}A_uv'.format(prefix)],pardict['FeII{0}A_op'.format(prefix)],pardict['FeII{0}dcen'.format(prefix)],pardict['FeII{0}fwhm'.format(prefix)]))
            sedcomp.append('FeII')    
        if self.BaC:
            agnsed_rest.append(SEDs.BaC(waveintrin,pardict['BaC{0}logL3600'.format(prefix)],pardict['BaC{0}logM'.format(prefix)],pardict['BaC{0}logMdot'.format(prefix)],pardict['BaC{0}spin'.format(prefix)],pardict['BaC{0}dcen'.format(prefix)],pardict['BaC{0}fwhm'.format(prefix)]))
            sedcomp.append('BaC')
        if self.torus:
            agnsed_rest.append(SEDs.cat3Dtorus(waveintrin,10.**(pardict['{0}torus_logL'.format(prefix)]-38), pardict['{0}torus_a'.format(prefix)], pardict['{0}torus_h'.format(prefix)], pardict['{0}torus_N0'.format(prefix)], pardict['{0}torus_i'.format(prefix)]))
            sedcomp.append('Torus')
        if self.hotdust:
            agnsed_rest.append(SEDs.Bbody(waveintrin,10.**(pardict['{0}Lhotdust'.format(prefix)]-38),jnp.array(pardict['{0}Thotdust'.format(prefix)],dtype=jnp.float32)))
            sedcomp.append('Hot dust')
        cm=SEDs.mycigale(waveintrin,delta)*pardict['agn_Av{0}'.format(prefix)]/3.1/2.
        for loop in range(len(agnsed_rest)):
            agnsed_rest[loop] /= 10**cm
            x,agnsed = SEDs.sed_to_obse(waveintrin,agnsed_rest[loop],self.redshift,self.ebv_G)
            agnsed_rest[loop] = agnsed*self.rnorm
        return agnsed_rest, sedcomp
    
    def generate_IFU(self, wavelength, posi, shape, psfimage, LSF_FWHM=1.6, GaussianLSF=True , transpar=None, pardict=None):
        # wavelength: in observed frame
        # psfimage: 3D array, the shape of empirical PSF, [y, x, wave]
        # posi: [x,y] of the position of the AGN, put in the IFU
        # LSF_FWHM: Full width at half maximum of the line spread function (bin size; 1 A=1 bin for KCWI)
        # GaussianLSF: True or False, whether the line spread function is approximated by Gaussian
        # the function 
        nx = shape[0]
        ny = shape[1]
        ns = len(wavelength)
        IFU =  jnp.zeros((ny,nx,ns),dtype=jnp.float32)
        #waverest = wavelength/(1.+self.redshift)
        # There was a bug. should give observed wavelength to fiducial_sed
        agnsed_obs = self.fiducial_sed(wavelength,pardict)
        # The extinction already applied in fiducial_sed, no need to do it again
        #if self.Av > 0.:
        #    cm=SEDs.jccm89(waverest,self.Av,3.1)/2.5
        #    agnsed_rest /= 10**cm
        #_,agnsed_obs = SEDs.sed_to_obse(waverest,agnsed_rest,self.redshift,self.ebv_G)
        if transpar is None:
            xpix,ypix = indentify_xy(posi[0],posi[1])
        else:
            xpix,ypix = coordinates_transfer(posi[0],posi[1],transpar)
        # jax approach
        #def loop_body(carry, loop_vars):
        #    w_idx, agnsed_obs_value = loop_vars
        #    psf_slice = psfimage[:, :, w_idx]
        #    IFU_slice = Pro.add_PSF(xpix, ypix, nx, ny, psf_slice) * self.rnorm * agnsed_obs_value
        #    return carry, IFU_slice
        #_, IFU_slices = jax.lax.scan(loop_body, None, (jnp.arange(len(wavelength)), agnsed_obs))
        #IFU = jnp.stack(IFU_slices, axis=-1)
        # addPSF3D approach
        if GaussianLSF:
            kernel = Gaussian1DKernel(LSF_FWHM/(2*np.sqrt(2*np.log(2))))
            agnsed_obs=convolve(agnsed_obs,kernel)
        transformed_psfimage = Pro.add_PSF_3D(xpix, ypix, nx, ny, psfimage, 1.)  # using the new 3D function
        IFU = transformed_psfimage * agnsed_obs[:, None, None] * self.rnorm
    
        return IFU
    
    def generate_grism_IFU(self,wavelength,posi,shape,psfimage,pardict=None,transpar = None):
        nx = shape[1]
        ny = shape[0]
        if transpar is None:
            xpix,ypix = indentify_xy(posi[0],posi[1])
        else:
            xpix,ypix = coordinates_transfer(posi[0],posi[1],transpar)

        psf_proj = Pro.add_PSF(xpix, ypix, nx, ny, psfimage) 
        IFU = psf_proj[...,None] * self.fiducial_sed(wavelength, pardict = pardict)
        return IFU

class Star(object):
    """
    Point-source stellar object for foreground/companion star modeling.
    
    Models individual stars as unresolved point sources using either synthetic
    stellar atmosphere models (PHOENIX) or non-parametric flux measurements.
    Used for fitting foreground stars, bright companions, or calibration stars
    in galaxy images.
    
    Parameters
    ----------
    prefix : str, optional
        Name prefix for parameters (default: ''). Use to distinguish multiple stars.
    useSED : bool, optional
        If True, use PHOENIX stellar atmosphere models; if False, use
        non-parametric flux measurements per band (default: True).
    
    Attributes
    ----------
    name : str
        Star name (same as prefix).
    prefix : str
        Parameter name prefix.
    sedModel : str
        SED model type: 'PHOENIX' (synthetic spectra) or 'NP' (non-parametric).
    bandslist : list
        List of photometric bands for non-parametric mode.
    
    Methods
    -------
    generate_image(posi, shape, band, psfimage, pardict, resp, ...)
        Create 2D star image as PSF-convolved point source.
    fiducial_sed(wavelength, pardict)
        Generate stellar SED from atmosphere model or interpolation.
    generate_SED(resps, pardict)
        Compute multi-band photometry from SED.
    
    Notes
    -----
    **PHOENIX mode** (useSED=True):
    
    - Uses pre-computed stellar atmosphere grids
    - Parameters: Teff (temperature), logg (surface gravity), logZ (metallicity)
    - Valid ranges: Teff [3000, 12000] K, logg [-0.5, 5.5], logZ [-4, 1]
    - Normalization: logL38 (log10 of luminosity in 10^38 erg/s)
    
    **Non-parametric mode** (useSED=False):
    
    - Directly fit flux in each photometric band
    - Parameters: {prefix}_{band}_logL for each band
    - Interpolates between bands for SED
    - Flexible but requires many bands for good spectral coverage
    
    The Star class is typically used in combination with Galaxy/AGN classes
    for full scene modeling. Stars appear as point sources (convolved with PSF)
    while galaxies are extended.
    
    Examples
    --------
    >>> # Create foreground star with PHOENIX model
    >>> star = Star(prefix='star1', useSED=True)
    >>> params = {'star1_logL38': 0.5,  # L = 3.16e38 erg/s
    ...           'star1_Teff': 5800,    # Solar temperature
    ...           'star1_logg': 4.5,     # Solar gravity
    ...           'star1_logZ': 0.0}     # Solar metallicity
    
    >>> # Generate SED
    >>> wave = jnp.linspace(3000, 10000, 1000)
    >>> sed = star.fiducial_sed(wave, params)
    
    >>> # Generate point-source image
    >>> posi = [75.0, 120.0]  # pixel coordinates
    >>> shape = (200, 200)
    >>> img = star.generate_image(posi, shape, 'F200W', psf, params, response)
    
    >>> # Non-parametric mode for calibration star
    >>> star_cal = Star(prefix='cal', useSED=False)
    >>> star_cal.bandslist = ['F150W', 'F200W', 'F277W', 'F356W']
    >>> params_cal = {'cal_F150W_logL': 2.0, 'cal_F200W_logL': 2.1,
    ...               'cal_F277W_logL': 2.2, 'cal_F356W_logL': 2.3}
    
    See Also
    --------
    AGN : Point-source AGN modeling
    Galaxy : Extended galaxy modeling
    get_stellar_sed : PHOENIX SED interpolation
    
    References
    ----------
    .. [1] Husser et al. 2013, A&A, 553, A6 (PHOENIX)
    """
    def __init__(self, prefix='', useSED = True):
        self.name = prefix
        self.prefix = prefix
        if useSED:
            self.sedModel = 'PHOENIX'
        else:
            self.sedModel = 'NP'
        self.bandslist = []
        

    def __del__(self):
        self.name = None
        self.sedModel = None
        self.sortwave = None

    def generate_image(self, posi, shape, band, psfimage, pardict, resp, transpar=None, psfimscale = 1.):
        ny = shape[0]
        nx = shape[1]
        prefix = self.name
        if self.sedModel == 'NP':
            flux_band = 10**(pardict['{0}_{1}_logL'.format(prefix,band)]-38)
        else:
            resp = np.genfromtxt(filterpath / band, dtype=float, names=None)
            filter_x=resp[:,0]
            filter_y=resp[:,1]
            ax=jnp.trapezoid(filter_y,x=filter_x)
            starsed = self.fiducial_sed(filter_x,pardict)
            flux_band = jnp.trapezoid(starsed*filter_y,x=filter_x)/ax
        if nx*ny == 1:
            starimg = flux_band*jnp.ones((ny,nx),dtype=jnp.float32)
        else:
            if transpar is None:
                xpix,ypix = indentify_xy(posi[0],posi[1])
            else:
                xpix,ypix = coordinates_transfer(posi[0],posi[1],transpar)
            starimg = Pro.add_PSF(xpix,ypix,nx,ny,psfimage,scalefactor=psfimscale)*flux_band
        return starimg
    
    def fiducial_sed(self, wavelength, pardict=None):
        prefix = self.name
        if self.sedModel == 'PHOENIX':
            starsed =SEDs.get_stellar_sed(wavelength,10**pardict['{0}_logL38'.format(prefix)],pardict['{0}_Teff'.format(prefix)],pardict['{0}_logg'.format(prefix)],pardict['{0}_logZ'.format(prefix)])
        elif self.sedModel == 'NP':
            fluxes =[]
            for band in self.bandslist:
                fluxes.append(10**(pardict['{0}_{1}_logL'.format(prefix,band)]-38))
            fluxes = jnp.array(fluxes)
            return jnp.interp(wavelength,self.iwaves,fluxes)
        return starsed

    def generate_SED(self, resps, pardict = None):
        wavelength = self.sortwave   
        fll_tot = self.fiducial_sed(wavelength,pardict=pardict)
        SEDs = []
        for resp in resps:
            filterx = resp[:,0]
            filtery = resp[:,1]
            fluxfiler = jnp.interp(filterx,wavelength,fll_tot)
            ax = jnp.trapezoid(filtery,x=filterx)
            flux_band = jnp.trapezoid(fluxfiler*filtery,x=filterx)/ax
            SEDs.append(flux_band)
        SEDs = jnp.array(SEDs,dtype=jnp.float32)
        return SEDs

