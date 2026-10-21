"""
Spectral Energy Distribution Interpolation Module
==================================================

This module provides fast interpolation of pre-computed SED templates for
stellar populations, AGN, dust emission, and nebular emission. It uses JAX
for GPU-accelerated interpolation and automatic differentiation support.

The module loads various SED libraries at import time and provides functions
to interpolate these templates for arbitrary physical parameters.

SED Templates
-------------
- **Stellar populations**: BC03-based continuum and instantaneous burst
- **AGN**: Thin disk, starburst disk, broad-line region
- **Dust emission**: Draine & Li (2007) models with power-law and constant U
- **Nebular emission**: Continuum (free-free, free-bound) and emission lines
- **Torus**: CAT3D torus models for AGN
- **High-resolution**: Templates for IFU/spectroscopy

Key Functions
-------------
- get_host_SED : Interpolate stellar population SED
- get_AGN_SED : Interpolate AGN thin disk SED
- get_nebular : Interpolate nebular continuum and lines
- get_cold_dust_sed : Interpolate dust emission SED
- cat3Dtorus : Interpolate AGN torus emission
- sed_to_obse : Transform rest-frame to observed-frame
- mycigale : Calzetti+ dust attenuation curve
- luminosity_distance : Compute luminosity distance at redshift
- kpc_per_arcsec : Physical scale at redshift
- cosmo_age : Age of universe at redshift

Cosmology
---------
Default cosmology: Flat ΛCDM with H0=70, Ωm=0.3, ΩΛ=0.7

Units
-----
- Wavelength: Angstroms
- Luminosity: 10^38 erg/s (or per Å for spectral luminosity)
- Mass: Solar masses (M_sun)
- Age: Gyr
- Metallicity: Z (mass fraction, not [Z/H])
- Dust extinction: Av (magnitudes)

Environment Variable
--------------------
Requires GS_DATA_PATH environment variable pointing to template directory.
Set in your shell configuration:

    export GS_DATA_PATH=/path/to/galfits/data

Template Data
-------------
Templates are stored as .npz files with regular grids for fast interpolation.
All interpolators use RegularGridInterpolator from JAX for differentiability.

Notes
-----
All luminosity values are in units of 10^38 erg/s unless otherwise noted.
This module is automatically imported and initialized when galfits is loaded.
"""

import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from . import Constant as C
from jax import grad, jit, vmap
import os
import warnings
import jax
from jax import lax
import numpy as np
import pickle
from functools import partial
from jax.scipy.special import erf

# ALL the Luminosity are in L38

def _find_data_path():
    env_path = os.environ.get("GS_DATA_PATH")
    if env_path:
        if os.path.isdir(os.path.join(env_path, "templates")):
            return env_path
        return env_path

    module_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(module_dir, "..", "data")),
        os.path.abspath(os.path.join(module_dir, "..", "..", "data")),
        os.path.abspath(os.path.join(module_dir, "..", "..", "..", "data")),
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "templates")):
            return path
    return None


def _template_file(filename):
    return os.path.join(DATA_PATH, "templates", filename)


DATA_PATH = _find_data_path()
SPHINX_BUILD = os.environ.get("SPHINX_BUILD", "0") == "1"

# Default SFRs (Msun/yr) for EAZY templates #1-18 from eazy_templates.csv.
EZY_SFR = jnp.array([7.8729427e-06, 6.3261302e-04, 1.0946556e+01, 7.8539217e-01,
       5.9802396e-08, 8.7987081e-07, 1.8755735e-09, 3.8053772e-06,
       2.3488412e-05, 7.0655952e+01, 2.2518292e-05, 3.3370560e+01,
       4.1507337e-07, 3.5962694e-07, 1.9262400e-03, 2.3268182e+02,
       4.2486950e-05, 5.7516351e+00], dtype=jnp.float32)
EZY_logM = jnp.array([10.042975, 11.458659, 10.107912, 12.426558,  8.539674, 10.459066,
        8.559923, 10.116205, 10.909261, 11.708006,  9.943432, 10.250462,
       10.161845, 10.06802 , 10.873773, 10.198579, 11.178163, 10.885424], dtype=jnp.float32)

if DATA_PATH is None:
    if not SPHINX_BUILD:
        warnings.warn(
            "GS_DATA_PATH is not set and no local templates directory was found. "
            "Template-dependent functions in galfits.sed_interp will not run.",
            RuntimeWarning,
        )

    intp_thindisk = None
    intp_SBdisk = None
    intp_BLRDC = None
    stellar_seds = None
    stellar_params = None
    stellar_wave = None
    intp_host_cont = None
    WAVEHOST = None
    intp_host_inst = None
    intp_nebularDC = None
    intp_nebularline = None
    intp_dl2014Uconst = None
    intp_Ldl2014Uconst = None
    intp_dl2014Upower = None
    intp_Ldl2014Upower = None
    cat3Dtemplate = None
    cat3DparList = None
    cat3DluminList = None
    cat3Dwave = None
    z_to_Mpcx = None
    z_to_Mpcy = None
    z_to_scale = None
    z_to_age = None
    intp_host_hres_cont = None
    intp_host_hres_inst = None
    EZY_SED = None
    EZY_WAVE = None
    feii_template_op = None
    feii_template_uv = None
    wave_bac = None
elif not os.path.isdir(os.path.join(DATA_PATH, "templates")):
    if not SPHINX_BUILD:
        warnings.warn(
            f"GS_DATA_PATH is set to '{DATA_PATH}', but templates were not found. "
            "Template-dependent functions in galfits.sed_interp will not run.",
            RuntimeWarning,
        )

    intp_thindisk = None
    intp_SBdisk = None
    intp_BLRDC = None
    stellar_seds = None
    stellar_params = None
    stellar_wave = None
    intp_host_cont = None
    WAVEHOST = None
    intp_host_inst = None
    intp_nebularDC = None
    intp_nebularline = None
    intp_dl2014Uconst = None
    intp_Ldl2014Uconst = None
    intp_dl2014Upower = None
    intp_Ldl2014Upower = None
    cat3Dtemplate = None
    cat3DparList = None
    cat3DluminList = None
    cat3Dwave = None
    z_to_Mpcx = None
    z_to_Mpcy = None
    z_to_scale = None
    z_to_age = None
    intp_host_hres_cont = None
    intp_host_hres_inst = None
    EZY_SED = None
    EZY_WAVE = None
    feii_template_op = None
    feii_template_uv = None
    wave_bac = None
else:
    sed_data = jnp.load(_template_file('thin_disk_log.npz'))
    points = (sed_data['spin'], sed_data['logMdot'], sed_data['logM'], sed_data['wave'])
    intp_thindisk = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

    sed_data = jnp.load(_template_file('starburst_disk.npz'))
    points = (sed_data['amax'], sed_data['logMdot'], sed_data['logM'], sed_data['wave'])
    intp_SBdisk = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

    sed_data = jnp.load(_template_file('BLRDC_ev.npz'))
    points = (sed_data['spin'], sed_data['logM'], sed_data['logMdot'], sed_data['wave'])
    intp_BLRDC = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

    sed_data = jnp.load(_template_file('stellar.npz'))
    stellar_seds = jnp.array(sed_data['sed'])
    stellar_params = jnp.array(sed_data['params'])
    stellar_wave = jnp.array(sed_data['wave'])

    sed_data    = jnp.load(_template_file("host_conti_fsps_fsps_mist_miles_chab_neb_freeGasz_oldfsps.npz"))
    points  = (sed_data["Zste"], sed_data["Zneb"], sed_data["age"], sed_data["wave"])
    sed     = sed_data["sed"]
    intp_host_cont  = RegularGridInterpolator(points, sed, bounds_error=False,fill_value=0.)
    WAVEHOST = sed_data['wave']
    DUST_TSEC  = 10.0**7.025         # young / old population dividing point for two dust extinction components model.

    sed_data    = jnp.load(_template_file("dl07_unit_dustEmi.npz"))
    points  = (sed_data["gamma"], sed_data["umin"], sed_data["qpah"], sed_data["wave"])
    sed    = sed_data["sed"]
    intp_dl07_dustEmi    = RegularGridInterpolator(points, sed,bounds_error=False,fill_value=0.)

    sed_data    = jnp.load(_template_file("host_inst_fsps_fsps_mist_miles_chab_neb_freeGasz_oldfsps.npz"))
    points  = (sed_data["Zste"], sed_data["Zneb"], sed_data["age"], sed_data["wave"])
    sed     = sed_data["sed"]
    intp_host_inst  = RegularGridInterpolator(points, sed, bounds_error=False,fill_value=0.)
 

    sed_data = jnp.load(_template_file('nebular_DC.npz'))
    points = (sed_data['Z'], sed_data['logU'], sed_data['wave'])
    intp_nebularDC = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

    sed_data = jnp.load(_template_file('nebular_line.npz'))
    points = (sed_data['Z'], sed_data['logU'], sed_data['wave'])
    intp_nebularline = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

    sed_data = jnp.load(_template_file('dl2014Uconst.npz'))
    points = (sed_data['qpah'], sed_data['Umin'], sed_data['wave'])
    intp_dl2014Uconst = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)
    intp_Ldl2014Uconst = RegularGridInterpolator((sed_data['qpah'], sed_data['Umin']), sed_data['lumin'],bounds_error=False,fill_value=0.)

    sed_data = jnp.load(_template_file('dl2014Upower.npz'))
    points = (sed_data['qpah'], sed_data['Umin'], sed_data['alpha'], sed_data['wave'])
    intp_dl2014Upower = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)
    intp_Ldl2014Upower = RegularGridInterpolator((sed_data['qpah'], sed_data['Umin'], sed_data['alpha']), sed_data['lumin'],bounds_error=False,fill_value=0.)

    with open(_template_file('cat3D.template'), "rb") as cat3d_file:
        cat3Dtemplate = pickle.load(cat3d_file)
    cat3DparList = jnp.array(cat3Dtemplate['parList'])
    cat3DluminList = jnp.array(cat3Dtemplate['luminList'])
    cat3Dwave = jnp.array(cat3Dtemplate['wave'])

    tab_z = np.genfromtxt(_template_file('z_to_Mpc.txt'))
    z_to_Mpcx   = jnp.array(tab_z[:,0])
    z_to_Mpcy   = jnp.array(tab_z[:,1])
    z_to_scale  = jnp.array(tab_z[:,2])
    z_to_age    = jnp.array(tab_z[:,3])
    # scale means kpc per arcsec

    # hires 
    sed_data_hires = jnp.load(_template_file('host_conti_hires.npz'))
    points_hires = (sed_data_hires['Z'], sed_data_hires['age'], sed_data_hires['wave'])
    intp_host_hres_cont = RegularGridInterpolator(points_hires, sed_data_hires['sed'],bounds_error=False,fill_value=0.)

    sed_data = jnp.load(_template_file('host_inst_hires.npz'))
    points = (sed_data['Z'], sed_data['age'], sed_data['wave'])
    intp_host_hres_inst = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

    sed_data = jnp.load(_template_file('eaZY.npz'))
    EZY_SED = sed_data['sed']
    EZY_WAVE = sed_data['wave']

    feii_template_op = jnp.array(np.genfromtxt(_template_file('irontemplate_op_ev.txt'), dtype=float, names=None))
    feii_template_uv = jnp.array(np.genfromtxt(_template_file('irontemplate_uv_ev.txt'), dtype=float, names=None))
    wave_bac = jnp.array(np.genfromtxt(_template_file('BLRDC_ev.wavegrid'), dtype=float, names=None))


def wavegrid_from_template(waverange, zred=0.0, wrange=(0.85, 1.15), fallback_n=4000):
    """Return an observed-frame wavelength grid for plotting/output.

    Prefer the native rest-frame template wave grid (`WAVEHOST`, non-uniform,
    ~0.9 A in the optical) shifted to the observed frame via x*(1+z), so that
    emission lines survive interpolation at the correct observed wavelengths.
    Fall back to a log grid when the band is too narrow (< 2 template points).
    """
    wmin = float(wrange[0] * waverange[0])
    wmax = float(wrange[1] * waverange[1])
    if WAVEHOST is None:
        return np.logspace(np.log10(wmin), np.log10(wmax), fallback_n)
    wt_obs = np.asarray(WAVEHOST, dtype=float) * (1.0 + float(zred))
    m = (wt_obs >= wmin) & (wt_obs <= wmax)
    if m.sum() >= 2:
        return wt_obs[m]
    return np.logspace(np.log10(wmin), np.log10(wmax), fallback_n)

def interp_resps(resps, wavegrid, wrange_sub=[0.9, 1.1]):

    resps_new    = []
    for resp in resps:
        wave_native     = resp[:,0]
        wmin    = wrange_sub[0] * wave_native[0]
        wmax    = wrange_sub[1] * wave_native[-1]

        grid_mask   = np.where((wavegrid > wmin) & (wavegrid < wmax))
        wavegrid_tem    = wavegrid[grid_mask]

        resp_v  = resp[:, 1]
        resp_i  = np.interp(wavegrid_tem, wave_native, resp_v, left=0., right=0.) ## interp resp_v from wave_native to wavegrid_temp.
        resp_n  = np.column_stack((wavegrid_tem, resp_i))
        resps_new.append(resp_n)

    return  resps_new

def wavegrid_dlnlam(resps, wrange=[0.85, 1.15]):

    diffs   = [np.diff(np.log(resp[:, 0])).min() for resp in resps]
    diffs   = np.array(diffs)
    min_diff    = min(diffs.min(), 1e-3) ## 1e-3 means delta lambda ~ 1\AA for 1000\AA
    dlnlam  = min_diff

    wmin    = np.min([np.min(resp[:, 0]) for resp in resps]) * wrange[0]
    wmin    = min(910, wmin) ## for dust extinction/emission energy conservation calculation.
    wmax    = np.max([np.max(resp[:, 0]) for resp in resps]) 
    wmax    = np.exp(np.log(wmax) + dlnlam) * wrange[1] ## np.arange() will lose final element.

    wavegrid_log    = np.arange(np.log(wmin), np.log(wmax), dlnlam)
    wavegrid   = np.exp(wavegrid_log)

    resps_new   = interp_resps(resps, wavegrid) ## interplate resps to new grid
    return wavegrid, resps_new

# define Table model class
# The TableModel class is used to load and interpolate SEDs from a table model  
# table_path: path to the .npz file containing the table model
# param_names: list of parameter names to interpolate, same as in the .npz file
# prefix: prefix to add to the parameter names
# sed_data['sed'] is in log10(L_lambda) in ergs/s/A
# wavelength is in Angstrom, as 'wave' in the .npz file
# if logNorm is in param_names, it will be used for normalization

class TableModel:
    """
    Generic SED table model with multi-parameter interpolation.
    
    Loads pre-computed SED grids from .npz files and provides fast
    interpolation for arbitrary parameter values. Supports optional
    normalization parameter for flexible scaling.
    
    Parameters
    ----------
    table_path : str
        Path to .npz file containing SED table.
        Must contain 'sed' (log10 L_λ), 'wave', and parameter grids.
    param_names : list of str
        Parameter names to interpolate (must match keys in .npz file).
        If 'logNorm' is included, it will be used for normalization.
    prefix : str
        Prefix to add to parameter names for external interface
        (e.g., 'AGN' → 'AGN_spin', 'AGN_logM').
    
    Attributes
    ----------
    norm : bool
        True if normalization parameter is present.
    norm_key : str
        Name of normalization parameter (with prefix).
    param_names : list of str
        Parameter names without 'logNorm'.
    pnames : list of str
        Display names with prefix applied.
    intp_table_model : RegularGridInterpolator
        JAX interpolator for fast SED lookup.
    
    Methods
    -------
    get_sed(pardict, x)
        Interpolate SED at given parameter values and wavelengths.
    
    Notes
    -----
    Table files (.npz) must contain:
    
    - 'sed': SED grid as log10(L_λ) in erg/s/Å (shape: [N_p1, N_p2, ..., N_wave])
    - 'wave': Wavelength grid in Angstroms (1D array)
    - Parameter grids: 1D arrays for each parameter (e.g., 'age', 'Z', 'logM')
    
    If 'logNorm' parameter is included, SEDs are scaled by 10^logNorm
    during interpolation, allowing flexible normalization.
    
    The class uses JAX's RegularGridInterpolator for GPU-accelerated
    interpolation with automatic differentiation support.
    
    Examples
    --------
    >>> # Load custom AGN template table
    >>> agn_model = TableModel('custom_agn.npz', 
    ...                        param_names=['logM', 'logMdot', 'spin', 'logNorm'],
    ...                        prefix='AGN')
    >>> # Interpolate at specific parameters
    >>> params = {'AGN_logM': 8.0, 'AGN_logMdot': -0.5, 
    ...           'AGN_spin': 0.7, 'AGN_logNorm': 44}
    >>> wave = jnp.logspace(2, 5, 1000)
    >>> sed = agn_model.get_sed(params, wave)
    
    See Also
    --------
    get_AGN_SED : Specialized AGN disk interpolation
    get_host_SED : Specialized stellar population interpolation
    """
    def __init__(self, table_path, param_names, prefix):
        self.table_path     = table_path
        self.prefix         = prefix
        self.param_names    = list(param_names)
        self.pnames         = [f"{prefix}_{name}" if prefix else name for name in self.param_names]
        self.load_table_model()

    def load_table_model(self):
        sed_data = jnp.load(self.table_path)
        points = tuple(sed_data[name] for name in self.param_names)
        # append wavelength axis
        points += (sed_data['wave'],)
        self.intp_table_model = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

    def get_sed(self, allparams, norm_key):
        """
        Interpolate SED at given parameters and wavelengths.
        
        Parameters
        ----------
        pardict : dict
            Dictionary mapping parameter names (with prefix) to values.
            Must include all parameters in self.param_names.
            If self.norm=True, must also include normalization parameter.
        x : array_like
            Wavelength grid in Angstroms for interpolation.
        
        Returns
        -------
        array_like
            Interpolated SED normalized to L5100 = 1. Output in units of 10^38 erg/s/Å.
        
        Examples
        --------
        >>> params = {'prefix_param1': 1.0, 'prefix_param2': 2.0}
        >>> wave = jnp.linspace(1000, 10000, 1000)
        >>> sed = model.get_sed(params, wave)
        """
        # param_values should be in the same order as param_names
        sed = 10**(self.intp_table_model(allparams) + norm_key - 38)
        
        return sed

def get_eazy_sed(wave, lognorms):
    """
    Get the EAZY SEDs for a given set of wavelengths and normalizations.

    Parameters
    ----------
    wave : array-like
        The wavelengths at which to evaluate the SEDs.
    lognorms : array-like
        Log10 normalizations for each template. (18,) array-like.

    Returns
    -------
    sed : array-like
        The SEDs evaluated at the given wavelengths and normalizations.
        by simply linearly combining the EAZY templates with the provided normalizations.
    """

    wave = jnp.asarray(wave)
    lognorms = jnp.asarray(lognorms)

    # Interpolate each template in log-space, then apply per-template log-normalization.
    logL_temp = jax.vmap(lambda sed_i: jnp.interp(wave, EZY_WAVE, sed_i))(EZY_SED)
    sed_full = 10**(logL_temp + lognorms[:, None] - 38.0)
    sed = jnp.sum(sed_full, axis=0)
    return sed


def eazy_lognorm_to_logSFR(lognorms, template_sfr=EZY_SFR):
    """
    Convert EAZY template log-normalizations to total log10(SFR).

    Parameters
    ----------
    lognorms : array-like
        Per-template log10 normalization coefficients.
    template_sfr : array-like, optional
        SFR of each EAZY template at logM=10, in Msun/yr.

    Returns
    -------
    float
        Total log10(SFR/Msun yr^-1).
    """
    lognorms = jnp.asarray(lognorms)
    template_sfr = jnp.asarray(template_sfr)
    sfr_tot = jnp.sum((10**lognorms) * template_sfr)
    return jnp.log10(jnp.maximum(sfr_tot, 1e-30))


def eazy_logw_to_logSFR(logw, logM, template_sfr=EZY_SFR):
    """
    Convert EAZY template log-weights to total log10(SFR).

    Parameters
    ----------
    logw : array-like
        Raw per-template log10 weights (before normalization).
    logM : float
        Total stellar mass in log10(Msun), consistent with GalfitS profile logM.
    template_sfr : array-like, optional
        SFR of each EAZY template at logM=10, in Msun/yr.

    Returns
    -------
    float
        Total log10(SFR/Msun yr^-1).
    """
    logw = jnp.asarray(logw)
    template_sfr = jnp.asarray(template_sfr)
    max_logw = jnp.max(logw)
    log_denom = max_logw + jnp.log10(jnp.sum(10**(logw - max_logw)))
    lognorms = (logM - 10.0) + logw - log_denom
    return eazy_lognorm_to_logSFR(lognorms, template_sfr=template_sfr)


def read_table_model(table_path, param_names, prefix):
    """
    Factory function to create TableModel instance.
    
    Convenience wrapper for TableModel initialization.
    
    Parameters
    ----------
    table_path : str
        Path to .npz file containing SED table.
    param_names : list of str
        Parameter names to interpolate.
    prefix : str
        Prefix for parameter names.
    
    Returns
    -------
    TableModel
        Initialized table model ready for interpolation.
    
    Examples
    --------
    >>> model = read_table_model('templates/custom.npz', 
    ...                          ['age', 'Z', 'Av'], prefix='host')
    """
    return TableModel(table_path, param_names, prefix)

def luminosity_distance(z):
    """
    Compute luminosity distance at redshift z.
    
    Uses pre-computed lookup table for flat ΛCDM cosmology (H0=70, Ωm=0.3).
    
    Parameters
    ----------
    z : float or array_like
        Redshift value(s).
    
    Returns
    -------
    float or array_like
        Luminosity distance in Mpc.
    
    Examples
    --------
    >>> dL = luminosity_distance(2.0)
    >>> print(f"Luminosity distance at z=2: {dL:.1f} Mpc")
    """
    return jnp.interp(z, z_to_Mpcx, z_to_Mpcy)

def kpc_per_arcsec(z):
    """
    Compute physical scale at redshift z.
    
    Returns the angular scale (kpc per arcsec) for flat ΛCDM cosmology.
    
    Parameters
    ----------
    z : float or array_like
        Redshift value(s).
    
    Returns
    -------
    float or array_like
        Physical scale in kpc per arcsecond.
    
    Examples
    --------
    >>> scale = kpc_per_arcsec(2.0)
    >>> print(f"At z=2: 1 arcsec = {scale:.2f} kpc")
    """
    return jnp.interp(z, z_to_Mpcx, z_to_scale)

def cosmo_age(z):
    """
    Compute age of universe at redshift z.
    
    Uses pre-computed lookup table for flat ΛCDM cosmology (H0=70, Ωm=0.3).
    
    Parameters
    ----------
    z : float or array_like
        Redshift value(s).
    
    Returns
    -------
    float or array_like
        Age of universe in Gyr.
    
    Examples
    --------
    >>> age = cosmo_age(2.0)
    >>> print(f"Universe age at z=2: {age:.2f} Gyr")
    """
    return jnp.interp(z, z_to_Mpcx, z_to_age)

def ccm89ab_ir_invum(x, Av, Rv):
    """
    Cardelli, Clayton & Mathis (1989) extinction function.
    ccm89 a, b parameters for 0.3 < x < 1.1 (infrared)

    Parameters:
    ------------
    x: float
        wavelength
    Av:float
        extinction in magnitudes at characteristic V band wavelength.
    Rv:float 
        The ratio between total extinction and selective extinction.
    """
    y = x**1.61
    a = 0.574 * y
    b = -0.527 * y
    return Av * (a + b / Rv)

def ccm89ab_opt_invum(x, Av, Rv):
    '''

    ccm89 a, b parameters for 1.1 < x < 3.3 (optical)
    '''
    y = x - 1.82
    a = ((((((0.329990*y - 0.77530)*y + 0.01979)*y + 0.72085)*y - 0.02427)*y
             - 0.50447)*y + 0.17699)*y + 1.0
    b = ((((((-2.09002*y + 5.30260)*y - 0.62251)*y - 5.38434)*y + 1.07233)*y
             + 2.28305)*y + 1.41338)*y
    return Av * (a + b / Rv)


def ccm89ab_uv1_invum(x, Av, Rv):
    """ccm89 a, b parameters for 3.3 < x < 8.0 (ultraviolet)"""
    y = x - 4.67
    a = 1.752 - 0.316*x - (0.104 / (y*y + 0.341))
    y = x - 4.62
    b = -3.090 + 1.825*x + (1.206 / (y*y + 0.263))
    return Av * (a + b / Rv)

def ccm89ab_uv2_invum(x, Av, Rv):
    """ccm89 a, b parameters for 3.3 < x < 8.0 (ultraviolet)"""
    y = x - 4.67
    a = 1.752 - 0.316*x - (0.104 / (y*y + 0.341))
    y = x - 4.62
    b = -3.090 + 1.825*x + (1.206 / (y*y + 0.263))
    y = x - 5.9
    y2 = y * y
    y3 = y2 * y
    a += -0.04473*y2 - 0.009779*y3
    b += 0.2130*y2 + 0.1207*y3
    return Av * (a + b / Rv)

def ccm89ab_fuv_invum(x, Av, Rv):
    """ccm89 a, b parameters for 8 < x < 11 (far-UV)"""
    y = x - 8.
    y2 = y * y
    y3 = y2 * y
    a = -0.070*y3 + 0.137*y2 - 0.628*y - 1.073
    b = 0.374*y3 - 0.420*y2 + 4.257*y + 13.670
    return Av * (a + b / Rv)

def ccm89(x,Av,Rv):
    x = 1e4 / x
    out = ccm89ab_ir_invum(x,Av,Rv)
    out = jnp.where((x>1.1)&(x<3.3),ccm89ab_opt_invum(x,Av,Rv),out)
    out = jnp.where((x>3.3)&(x<5.9),ccm89ab_uv1_invum(x,Av,Rv),out)
    out = jnp.where((x>5.9)&(x<8),ccm89ab_uv2_invum(x,Av,Rv),out)
    out = jnp.where((x>8)&(x<11),ccm89ab_fuv_invum(x,Av,Rv),out)
    return out

jccm89 = jit(ccm89)

def gaussian_jax(x, peak, cen, wid):
    return (peak * jnp.exp(-(x-cen)**2 / (2*wid**2)))

def gaussianL_jax3D(x, norm, cen, wid):
    return norm/(jnp.sqrt(2*jnp.pi)*wid)*jnp.exp(-0.5*(x-cen)**2/wid**2)

feii_dlogv = 1.293869
#templates in feii_dlogv = 1.293869, where feii_dlogv = dlogv*4000
bac_dlogv = 9.159824
hires_dlogv = 2.e-01

def FeII(x,A_uv,A_op,dcen,fwhm):
    """
    Generate optical/UV FeII emission template for AGN.
    
    Constructs iron emission spectrum using empirical templates with
    independent UV and optical normalization. Applies velocity broadening
    and Doppler shift.
    
    Parameters
    ----------
    x : array_like
        Rest-frame wavelength grid in Angstroms.
    A_uv : float
        Amplitude of UV FeII template (λ < 3500 Å).
    A_op : float
        Amplitude of optical FeII template (λ > 3500 Å).
    dcen : float
        Velocity offset in km/s (positive = redshift).
    fwhm : float
        Full-width at half-maximum velocity width in km/s.
        Intrinsic template FWHM = 900 km/s is deconvolved.
    
    Returns
    -------
    array_like
        FeII emission flux (arbitrary normalization).
    
    Notes
    -----
    FeII emission arises from collisionally excited iron in the AGN
    broad-line region. It creates broad pseudo-continuum with thousands
    of blended transitions, especially prominent at:
    
    - UV: 2200-3200 Å (FeII multiplet 191)
    - Optical: 4400-5500 Å (FeII multiplets around Hβ)
    
    Templates are empirical from composite quasar spectra. The UV and
    optical components scale independently, reflecting different excitation
    conditions.
    
    Convolution is performed in log-wavelength space for uniform velocity
    resolution. The 900 km/s intrinsic width is quadratically subtracted
    before applying user-specified FWHM.
    
    See Also
    --------
    BaC : Balmer continuum emission
    get_AGN_SED : AGN accretion disk continuum
    
    Examples
    --------
    >>> wave = jnp.linspace(4000, 5500, 1000)
    >>> # Moderate FeII with 3000 km/s FWHM
    >>> feii = FeII(wave, A_uv=1.0, A_op=1.0, dcen=0, fwhm=3000)
    
    References
    ----------
    Templates from Vestergaard & Wilkes 2001, ApJS, 134, 1
    """
    wavemodel = jnp.append(feii_template_uv[:,0],feii_template_op[:,0])
    fluxmodel = jnp.append(A_uv*feii_template_uv[:,1],A_op*feii_template_op[:,1])
    fwhm = jnp.sqrt(fwhm**2-900**2)
    referwave = 4000.
    # in logwavelength space
    std = fwhm*referwave/C.ckm/C.tl2/feii_dlogv
    xk = jnp.linspace(-100, 100, 201)
    kernel = gaussian_jax(xk,1/jnp.sqrt(C.twopi)/std,0.,std)
    f_conv = jnp.convolve(fluxmodel, kernel, mode='same')
    return jnp.interp(x, wavemodel*(1+dcen/C.ckm), f_conv)

def BaC(x,logL3600,logM,logMdot,spin,dcen,fwhm):
    """
    Generate Balmer continuum emission from AGN broad-line region.
    
    Computes BLR Balmer continuum using pre-computed templates that include
    high-order Balmer recombination edges (Balmer jump at 3646 Å plus
    higher-order series). Applies velocity broadening and Doppler shift.
    
    Parameters
    ----------
    x : array_like
        Rest-frame wavelength grid in Angstroms.
    logL3600 : float
        Log10 of continuum luminosity at 3600 Å in 10^38 erg/s/Å.
        Used for normalization.
    logM : float
        Log10 of black hole mass in solar masses [5, 10].
    logMdot : float
        Log10 of accretion rate in Eddington units [-4, 2].
    spin : float
        Black hole spin parameter [0, 0.99].
    dcen : float
        Velocity offset in km/s (positive = redshift).
    fwhm : float
        Full-width at half-maximum velocity width in km/s.
    
    Returns
    -------
    array_like
        Balmer continuum flux in 10^38 erg/s/Å.
    
    Notes
    -----
    The Balmer continuum arises from recombination of hydrogen in the
    dense, ionized gas of the AGN broad-line region. Key features:
    
    - Balmer jump at 3646 Å (flux discontinuity)
    - Series of higher-order Balmer edges converging to Balmer limit
    - Contributes blue "pseudo-continuum" in optical spectra
    - Depends on BLR density, temperature, covering factor
    
    Templates are computed from photoionization models and depend on
    black hole parameters through the ionizing continuum shape.
    
    The function normalizes to luminosity at 3600 Å (common AGN
    reference point below Balmer jump).
    
    See Also
    --------
    FeII : Iron emission template
    get_AGN_SED : AGN disk continuum
    
    Examples
    --------
    >>> wave = jnp.linspace(3000, 5000, 1000)
    >>> bac = BaC(wave, logL3600=3.0, logM=8.0, logMdot=-0.5,
    ...           spin=0.7, dcen=0, fwhm=5000)
    """
    normL3600 = intp_BLRDC((spin,logM,logMdot,3600))-logL3600+38
    flux = 10**(intp_BLRDC((spin,logM,logMdot,wave_bac))-normL3600)
    referwave = 4000.
    std = fwhm*referwave/C.ckm/C.tl2/bac_dlogv
    xk = jnp.linspace(-100, 100, 201)
    kernel = gaussian_jax(xk,1/jnp.sqrt(C.twopi)/std,0.,std)
    f_conv = jnp.convolve(flux, kernel, mode='same')
    return jnp.interp(x, wave_bac*(1+dcen/C.ckm), f_conv)

def nearest_neighbors_jax(X, point):
    min_vals = jnp.min(X, axis=0)
    max_vals = jnp.max(X, axis=0)
    ranges = max_vals - min_vals
    norm_X = (X - min_vals) / ranges
    norm_point = (point - min_vals) / ranges
    distance_matrix = jnp.sum((norm_X - norm_point) ** 2, axis=-1)
    return jnp.argmin(distance_matrix)

def get_stellar_sed(x, L38, Teff, logg, logZ):
    parList = jnp.array([Teff, logg, logZ])
    distance_matrix = jnp.sum((stellar_params - parList) ** 2, axis=-1)
    nn_index = jnp.argmin(distance_matrix)
    flux = 10**stellar_seds[nn_index]*L38
    y = jnp.interp(x, stellar_wave, flux)
    return y

def get_cold_dust_L(qpah, Umin, alpha, gamma):
    """
    Compute total dust luminosity for Draine & Li (2007) models.
    
    Returns integrated IR luminosity (total power emitted) for given
    dust parameters, normalized to dust mass.
    
    Parameters
    ----------
    qpah : float or array_like
        PAH abundance parameter (percentage of dust mass in PAHs).
    Umin : float
        Minimum radiation field intensity (Mathis field units).
    alpha : float or array_like
        Power-law index for radiation field distribution.
    gamma : float
        Fraction of dust mass in power-law component [0, 1].
    
    Returns
    -------
    float or array_like
        Log10 of total dust luminosity normalized to 10^38 erg/s
        per solar mass of dust: log10(L_dust / M_dust / 10^38).
    
    Notes
    -----
    Total luminosity combines constant and power-law components:
    
    L_tot = (1 - γ) × L(U_min) + γ × L(U_power)
    
    where L(U) is the emissivity integrated over wavelength.
    
    Used to convert dust mass to IR luminosity or vice versa.
    
    See Also
    --------
    get_cold_dust_sed : Full SED for same parameters
    
    Examples
    --------
    >>> logL_per_Msun = get_cold_dust_L(qpah=3.5, Umin=1.0, alpha=2.0, gamma=0.3)
    >>> # Total IR luminosity for 10^7 Msun of dust
    >>> logMdust = 7.0
    >>> logL_IR = logL_per_Msun + logMdust
    """
    qpah = jnp.array(qpah)
    alpha = jnp.array(alpha)
    logUconst = intp_Ldl2014Uconst((qpah, Umin)) 
    logUpower = intp_Ldl2014Upower((qpah, Umin, alpha)) 
    L = gamma*10**logUpower + (1-gamma)*10**logUconst
    return jnp.log10(L)-38.

def get_cold_dust_sed(x, logMdust, qpah, Umin, alpha, gamma):
    """
    Interpolate dust emission SED using Draine & Li (2007) models.
    
    Computes IR dust emission as combination of radiation fields with
    constant intensity (U_min) and power-law distribution (dM/dU ∝ U^-α).
    
    Parameters
    ----------
    x : array_like
        Rest-frame wavelength in Angstroms.
    logMdust : float
        Log10 of dust mass in solar masses.
    qpah : float
        PAH abundance parameter (percentage of dust mass in PAH grains).
        Range [0.47, 4.58]. Solar neighborhood: qpah ~ 3.5%.
    Umin : float
        Minimum radiation field intensity (in units of Mathis field).
        Range [0.1, 25]. ISRF: U = 1.
    alpha : float
        Power-law index for radiation field distribution dM/dU ∝ U^-α.
        Range [1.0, 3.0]. Typical: α ~ 2.
    gamma : float
        Fraction of dust mass in power-law component. Range [0, 1].
        gamma = 0: all dust at U_min (quiescent)
        gamma = 1: all dust in power-law (PDR-dominated)
    
    Returns
    -------
    array_like
        Dust emission spectral luminosity density in 10^38 erg/s/Å.
    
    Notes
    -----
    Total SED is combination of two components:
    
    - Constant U: L_const = (1 - γ) × M_dust × ε(U_min)
    - Power-law U: L_power = γ × M_dust × ∫ ε(U) U^-α dU
    
    where ε(U) is the emissivity per unit dust mass at radiation field U.
    
    PAH emission dominates at λ < 20 μm (features at 3.3, 6.2, 7.7, 11.3 μm).
    Larger grains emit modified blackbody at λ > 50 μm.
    
    See Also
    --------
    get_cold_dust_L : Total dust luminosity for same parameters
    
    Examples
    --------
    >>> wave = jnp.logspace(4, 6, 200)  # 1-100 μm
    >>> dust_sed = get_cold_dust_sed(wave, logMdust=7.0, qpah=3.5,
    ...                              Umin=1.0, alpha=2.0, gamma=0.3)
    
    References
    ----------
    .. [1] Draine & Li 2007, ApJ, 657, 810
    """
    logUconst_sed = intp_dl2014Uconst((qpah, Umin, x)) + logMdust - 38
    logUpower_sed = intp_dl2014Upower((qpah, Umin, alpha, x)) + logMdust - 38
    return gamma*10**logUpower_sed + (1-gamma)*10**logUconst_sed

def cat3Dtorus(x, L38, a, h, N0, i): 
    """
    Interpolate AGN torus emission SED using CAT3D models.
    
    Uses nearest-neighbor lookup in pre-computed grid of clumpy torus
    radiative transfer models (Hoenig & Kishimoto 2017).
    
    Parameters
    ----------
    x : array_like
        Rest-frame wavelength in Angstroms.
    L38 : float
        Normalization luminosity in units of 10^38 erg/s.
    a : float
        Radial density power-law index. Typical range [0, -2].
        ρ ∝ r^a where r is distance from center.
    h : float
        Scale height (ratio of torus height to radius at inner edge).
        Range [0.1, 1.0]. Larger h = more extended vertically.
    N0 : float
        Average number of clouds along equatorial ray. Range [1, 15].
        Controls optical depth (more clouds = higher τ).
    i : float
        Inclination angle in degrees [0, 90].
        i = 0: face-on (Type 1)
        i = 90: edge-on (Type 2)
    
    Returns
    -------
    array_like
        Torus emission spectral luminosity density in 10^38 erg/s/Å.
        Valid wavelength range: 1-100 μm (returns 0 outside).
    
    Notes
    -----
    - CAT3D models include clumpiness, 3D radiative transfer, anisotropy
    - Torus heated by central AGN UV/optical emission
    - Re-emits in near-to-far IR (peak around 10-30 μm)
    - Type 1/2 viewing angle dependence built into models
    - Uses nearest-neighbor interpolation (not smooth interpolation)
    
    The torus provides obscuration and IR re-emission in AGN unified model.
    
    See Also
    --------
    get_AGN_SED : AGN accretion disk emission
    get_cold_dust_sed : Extended cold dust emission
    
    Examples
    --------
    >>> wave = jnp.logspace(4, 5, 100)  # 1-10 μm
    >>> # Type 1 (face-on) torus
    >>> torus_sed = cat3Dtorus(wave, L38=1000, a=-0.5, h=0.5, N0=10, i=30)
    
    References
    ----------
    .. [1] Hoenig & Kishimoto 2017, ApJ, 838, 94 (CAT3D)
    """
    parList = jnp.array([a, h, N0, i])
    nn_index = nearest_neighbors_jax(cat3DparList, parList)
    flux = cat3DluminList[nn_index]*L38
    y = jnp.interp(x, cat3Dwave,flux)
    y = jnp.where((x>1e4) & (x<1e7), y, 0.)
    return y

def starburst_disk(x, logM, logMdot, amax):
    return intp_SBdisk((amax, logMdot, logM, x))

def get_host_SED_3D(x, logM, f_cont, age, Z, Av, sigma, C_unit, wave_temp, SFH = 'conti', present_age = 13.8, delta = 3.):
    # intp_burst_cont normalized by SFR = 10 Ms/yr
    # intp_burst_inst normalized by M_ast = 10^7 Ms
    # x: wavelength [91, 1.6e6] Angs
    # M: overall stellar mass;
    # f_cont: float, [0, 1.], fraction of the continuum starformation component
    # age: the age of the starburst component [1e-5, 13.69] Gyr
    # Z : metalicity, [0.001,0.04], 0.02 for solar ?unit as solar???
    # sigma  : velocity dispersion km/s
    # Av : V Attenuation
    # Return: restframe L_lambda in ergs/s
    # 3D means high resolution .
    if SFH == 'bins':
        f_cont_real = 10**f_cont
        f_cont_real = f_cont_real/jnp.sum(f_cont_real)
        sfr = f_cont_real[0]*10**logM/(age[1]-age[0])/1e9
        fluxmodel = 10**(intp_host_hres_cont((Z, 1e9*(age[1]), wave_temp))+jnp.log10(sfr)-39) # 这里为啥不是(age[1]+age[0])/2
        for loop in range(len(age)-2): # 这里的Z是不是也应该是一个list?
            sfr = f_cont_real[loop+1]*10**logM/(age[loop+2]-age[loop+1])/1e9
            fluxmodel += 10**(intp_host_hres_cont((Z, 1e9*(age[loop+2]), wave_temp))+jnp.log10(sfr)-39)-10**(intp_host_hres_cont((Z, 1e9*(age[loop+1]), wave_temp))+jnp.log10(sfr)-39)
        fluxmodel = fluxmodel*C_unit
    else:
        age_yr = age*1e9
        log_M_cont = logM + f_cont
        log_M_inst = logM + jnp.log10(1-10**f_cont)
        sed1 = 10**(intp_host_hres_inst((Z, age_yr, wave_temp))+log_M_inst-45)
        if SFH == 'conti':
            sed2 = 10**(intp_host_hres_cont((Z, 1e9*(present_age-0.3), wave_temp))+log_M_cont-48-jnp.log10(present_age-0.3))
        elif SFH == 'burst':
            sed2 = 10**(intp_host_hres_cont((Z, 1e8, wave_temp))+log_M_cont-47)
        fluxmodel = C_unit*(sed1+sed2)

    cm=mycigale(wave_temp,delta)*Av/3.1/2.5
    higrss = fluxmodel/(10**cm)
    referwave = 4000.
    std = sigma*referwave/C.ckm/hires_dlogv
    xk = jnp.linspace(-100, 100, 201)
    kernel = gaussian_jax(xk,1/jnp.sqrt(C.twopi)/std,0.,std)
    f_conv = jnp.convolve(higrss, kernel, mode='same')
    higrss_conv = jnp.interp(x, wave_temp, f_conv)
    
    return  higrss_conv

@partial(jax.jit, static_argnames=['Nlogv']) ## Nlogv should be stated as static_argname because length of jax array should be fixed. This params should not be traced.
def conv_spec(wave, flux, sigma, dcenlog=0, Nlogv = 10000):
    """
    Convolve spectrum with Gaussian velocity kernel.
    
    Performs convolution in log-wavelength space to model velocity dispersion
    and radial velocity shifts. JIT-compiled for performance.
    
    Parameters
    ----------
    wave : array_like
        Wavelength array in Angstroms.
    flux : array_like
        Flux array (same length as wave).
    sigma : float
        Velocity dispersion in km/s (1-sigma width of Gaussian kernel).
    dcenlog : float, optional
        Velocity offset in km/s (positive = redshift, default: 0).
    Nlogv : int, optional
        Number of points in log-wavelength interpolation grid (default: 2000).
        Higher = more accurate but slower.
    
    Returns
    -------
    array_like
        Convolved flux array at original wavelength grid.
    
    Notes
    -----
    Algorithm:
    
    1. Convert wavelength to log(λ) for uniform velocity spacing
    2. Interpolate flux onto regular log-wavelength grid
    3. Convolve with Gaussian kernel: K(v) ∝ exp(-v²/2σ²)
    4. Shift by dcenlog (applies Doppler formula)
    5. Interpolate back to original wavelength grid
    
    Edge effects are handled by extending flux with constant values
    (100 pixels on each side).
    
    The Gaussian kernel in log-wavelength is equivalent to velocity-space
    convolution: Δv/c = Δ(ln λ).
    
    See Also
    --------
    get_host_SED_3D : Uses this for velocity dispersion in stellar SEDs
    
    Examples
    --------
    >>> wave = jnp.linspace(5000, 5500, 1000)
    >>> flux = jnp.ones_like(wave)
    >>> flux = flux.at[500].set(10)  # Emission line
    >>> # Broaden line by 200 km/s dispersion
    >>> flux_conv = conv_spec(wave, flux, sigma=200)
    >>> # Also apply +500 km/s redshift
    >>> flux_shifted = conv_spec(wave, flux, sigma=200, dcenlog=500)
    """
    '''
    convolve the give spectra ['wave', 'flux'] with any given 'kernel'
    return the flux respect to wave
    sigma: velocity dispersion in km/s
    dcenlog: the offset of the central wavelength in velocity space
    '''
    logw = jnp.log(wave)
    newlogw = jnp.linspace(jnp.min(logw),jnp.max(logw),Nlogv)
    newflux = jnp.interp(newlogw,logw,flux)
    left_extrapolated = newflux[0]*jnp.ones(100)
    right_extrapolated = newflux[-1]*jnp.ones(100)
    extended_flux = jnp.concatenate([left_extrapolated, newflux, right_extrapolated])
    dlogv = newlogw[1]-newlogw[0]
    std = sigma/C.ckm/dlogv
    xk = jnp.linspace(-100, 100, 201)
    kernel = gaussian_jax(xk,1/jnp.sqrt(C.twopi)/std,0.,std)
    f_conv = jnp.convolve(extended_flux, kernel, mode='same')
    flux_conv = jnp.interp(logw-dcenlog/C.ckm, newlogw, f_conv[100:-100])
    return flux_conv



@partial(jax.jit, static_argnames=("Nlogv", "K"))
def conv_spec_lsf_R(
    wave,
    flux,
    lsf_wave,
    lsf_R,
    *,
    dcen_kms=0.0,
    extra_sigma_kms=0.0,
    Nlogv=2000,
    K=50,
):
    logw = jnp.log(wave)
    newlogw = jnp.linspace(jnp.min(logw), jnp.max(logw), Nlogv)
    dlogw = newlogw[1] - newlogw[0]

    newflux = jnp.interp(newlogw, logw, flux)

    lam_new = jnp.exp(newlogw)
    R_new = jnp.interp(lam_new, lsf_wave, lsf_R, left=lsf_R[0], right=lsf_R[-1])

    sigma_ln_inst = (1.0 / R_new) * C._FWHM_TO_SIGMA
    sigma_ln_extra = extra_sigma_kms / C.ckm
    sigma_ln_tot = jnp.sqrt(sigma_ln_inst**2 + sigma_ln_extra**2)

    std_pix = sigma_ln_tot / dlogw
    std_pix = jnp.clip(std_pix, 1e-3, 1e6)

    pad = K
    flux_pad = jnp.pad(newflux, (pad, pad), mode="edge")
    x = jnp.arange(-K, K + 1, dtype=newflux.dtype)

    def one_point(i_center, s):
        # IMPORTANT: center the window on i_center by starting at i_center - K
        start = (i_center - K).astype(jnp.int32)
        seg = lax.dynamic_slice(flux_pad, (start,), (2 * K + 1,))
        w = jnp.exp(-0.5 * (x / s) ** 2)
        w = w / jnp.sum(w)
        return jnp.sum(seg * w)

    i_center = (jnp.arange(Nlogv, dtype=jnp.int32) + pad)
    f_conv_log = jax.vmap(one_point)(i_center, std_pix)

    flux_conv = jnp.interp(logw - dcen_kms / C.ckm, newlogw, f_conv_log)
    return flux_conv


# 1.87531 is raised though fitting of cloudy output
def get_nebular(x, Lha, logU, Z, Av, withline = True, delta = 3.):
    '''
    NOTE: Don't use this function now.
    '''
    sed = intp_nebularDC((Z, logU, x))
    if withline:
        sed += intp_nebularline((Z, logU, x))
    cm=mycigale(x,delta)*Av/3.1/2.5
    return 1.87531*Lha*sed/(10**cm)


def get_host_SED(wavegrid, logM, f_cont, age, Zste, Zneb, C_unit, SFH = 'bins', present_age = 13.8,):
    '''
    # intp_burst_cont normalized by SFR = 10 Ms/yr
    # intp_burst_inst normalized by M_ast = 10^7 Ms
    # x: wavelength [91, 1.6e6] Angs
    # M: overall stellar mass;
    # f_cont: float, [0, 1.], fraction of the continuum starformation component
    # age: the age of the starburst component [1e-5, 13.69] Gyr
    # Z : metalicity, [0.001,0.04], 0.02 for solar
    # Av : V Attenuation
    # Return: restframe L_lambda in ergs/s


    NOTE: This function interplates stellar template only.
    '''

    ### interplate stellar emission for different `SFH`.
    if SFH == 'bins':
        f_cont_real = 10**f_cont
        f_cont_real = f_cont_real/jnp.sum(f_cont_real)
        sfr = f_cont_real[0] * 10**logM / (age[1]-age[0]) / 1e9
        fluxmodel = 10**(intp_host_cont((Zste, Zneb, 1e9*(age[1]), wavegrid)) + jnp.log10(sfr)-39)
        for loop in range(len(age)-2):
            sfr = f_cont_real[loop+1]* 10**logM / (age[loop+2]-age[loop+1]) / 1e9 
            fluxmodel += 10**(intp_host_cont((Zste, Zneb, 1e9*(age[loop+2]), wavegrid)) + jnp.log10(sfr) - 39) - 10**(intp_host_cont((Zste, Zneb, 1e9*(age[loop+1]), wavegrid)) + jnp.log10(sfr) - 39)
        fluxmodel = fluxmodel*C_unit
    else: ## TODO: standardlize parameters. `Zneb`
        age_yr = age*1e9
        log_M_cont = logM + f_cont
        log_M_inst = logM + jnp.log10(1-10**f_cont)
        sed1 = 10**(intp_host_inst((Zste, age_yr, wavegrid))+log_M_inst-45) 
        if SFH == 'conti':
            sed2 = 10**(intp_host_cont((Zste, 1e9*(present_age-0.3), wavegrid))+log_M_cont-48-jnp.log10(present_age-0.3))
        elif SFH == 'burst':
            sed2 = 10**(intp_host_cont((Zste, 1e8, wavegrid))+log_M_cont-47)
        fluxmodel = C_unit*(sed1+sed2)

    return fluxmodel


# ========== DUST EXTINCTION ==========

def apply_dust_extinction(wavegrid, f_cont, logM, age, Zste, Zneb, fluxmodel, 
                        dust2, dust_ratio, dust_index, dust1_index=-1.,dust_tesc=DUST_TSEC, 
                        SFH="bins",intp_external=None):

    '''
    This function apply dust extinction using two component model. (Charlot & Fall 2000)

    ======
    DUST_TSEC ~ 1e7.025 yr 
    '''
    # first seperate ste_POP into young POP and old POP.
    intp_use = intp_external if intp_external is not None else intp_host_cont
    edges  = [a * 1e9 for a in age]      # Gyr -> yr; list length is static
    f_cont_real     = 10**f_cont
    f_cont_real    /= jnp.sum(f_cont_real)

    # NOTE (jax): bin edges may be TRACED (fitted) values, so the young/old
    # split must not use Python `if` / `min()` / `break` on them.  We clip
    # with jnp.minimum and zero out fully-old bins with jnp.where instead. [kimi-k3]
    young_spec  = 0.0

    if SFH=="bins":
        for loop_a in range(len(edges) - 1):
            e_lo = edges[loop_a]
            e_hi = edges[loop_a + 1]
            sfr_tem = 10**logM * f_cont_real[loop_a] / (e_hi - e_lo)
            # young part of this bin = [e_lo, min(e_hi, dust_tesc)]
            hi = jnp.minimum(e_hi, dust_tesc)
            seg = sfr_tem * (10**(intp_use((Zste, Zneb, hi,   wavegrid)) - 39)
                           - 10**(intp_use((Zste, Zneb, e_lo, wavegrid)) - 39))
            # bins starting beyond the boundary contribute exactly zero
            young_spec = young_spec + jnp.where(e_lo < dust_tesc, seg, 0.0)
    else:
        pass # TODO: add other SFH types (conti/burst: truncate the
             # constant-SFR part at dust_tesc, burst is young iff age<=tesc)

    # apply dust extinction for different populations.
    old_spec    = fluxmodel - young_spec
    f_y, f_o = _attenuation_factors(wavegrid, dust2, dust_ratio, dust_index, dust1_index)
    fluxmodel_atten     = f_y * young_spec + f_o * old_spec

    return fluxmodel_atten


def _attenuation_factors(wavegrid, dust2, dust_ratio, dust_index,
                        dust1_index=-1.0):

    lam = jnp.asarray(wavegrid)
    tau_bc = dust2 * dust_ratio * (lam / 5500.0)**dust1_index ## birth dust: dust1 and dust1_index for young population.
    tau_diff = dust2 * _kc13_tau_unit(lam, dust_index) ##diffuse dust: dust 2 and dust_index for all population
    return jnp.exp(-tau_bc - tau_diff), jnp.exp(-tau_diff)


def _kc13_tau_unit(wave_aa, dust_index):
    """Kriek & Conroy (2013) diffuse curve: tau per unit dust2.

    tau     = [k_Calzetti/Rv + Drude_2175(E_b)/Rv] * (lam/5500)^dust_index
    E_b     = 0.85 - 1.9*dust_index  (KC13 slope-bump linkage).
        Equals 1.0003 at 5500 A (tiny excess is the Drude bump tail), so A_5500[mag] = 1.086*dust2; to convert a magnitudes-style Av: dust2 = Av/1.086.
    """
    lam = jnp.asarray(wave_aa)
    eb = 0.85 - 1.9 * dust_index
    drude = eb * (lam * 350.0)**2 / ((lam**2 - 2175.0**2)**2 + (lam * 350.0)**2)
    return (_calzetti_k_over_rv(lam) + drude / 4.05) * (lam / 5500.0)**dust_index


def _calzetti_k_over_rv(wavegrid):
    """Calzetti et al. (2000) k'(lam)/(0.44*4.05), zeroed past ~2.2 um root.
    """
    lam = jnp.asarray(wavegrid)
    x = 1e4 / lam
    red  = 1.17 * (-1.857 + 1.04 * x) + 1.78
    blue = 1.17 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + 1.78
    cal00 = jnp.where(lam > 6300.0, red, blue) / 0.44 / 4.05
    return jnp.maximum(cal00, 0.0)


#   ========== DUST EMISSION ==========

def get_dustEmi(wavegrid, gamma, umin, qpah, L_abs, intp_external=None):
    intp_use    = intp_external if intp_external is not None else intp_dl07_dustEmi
    ## TODO: add boundary check prompt.
    return L_abs * intp_use((gamma, umin, qpah, wavegrid))



# def get_host_SED_withL(x, logM, f_cont, age, Z, Av, C_unit, SFH = 'conti', present_age = 13.8, delta = 3.):
#     if SFH == 'bins':
#         f_cont_real = 10**f_cont
#         f_cont_real = f_cont_real/jnp.sum(f_cont_real)
#         sfr = f_cont_real[0]*10**logM/(age[1]-age[0])/1e9
#         fluxmodel = 10**(intp_host_cont((Z, 1e9*(age[1]), WAVEHOST))+jnp.log10(sfr)-39)
#         for loop in range(len(age)-2):
#             sfr = f_cont_real[loop+1]*10**logM/(age[loop+2]-age[loop+1])/1e9
#             fluxmodel += 10**(intp_host_cont((Z, 1e9*(age[loop+2]), WAVEHOST))+jnp.log10(sfr)-39)-10**(intp_host_cont((Z, 1e9*(age[loop+1]), WAVEHOST))+jnp.log10(sfr)-39)
#         fluxmodel = fluxmodel*C_unit
#     else:
#         age_yr = age*1e9
#         log_M_cont = logM + f_cont
#         log_M_inst = logM + jnp.log10(1-10**f_cont)
#         sed1 = 10**(intp_host_inst((Z, age_yr, WAVEHOST))+log_M_inst-45)
#         if SFH == 'conti':
#             sed2 = 10**(intp_host_cont((Z, 1e9*(present_age-0.3), WAVEHOST))+log_M_cont-48-jnp.log10(present_age-0.3))
#         elif SFH == 'burst':
#             sed2 = 10**(intp_host_cont((Z, 1e8, WAVEHOST))+log_M_cont-47)
#         fluxmodel = C_unit*(sed1+sed2)
#     cm=mycigale(WAVEHOST,delta)*Av/3.1/2.5
#     flux_extin =  fluxmodel/(10**cm)
#     absorbed_flux = jnp.trapezoid(fluxmodel-flux_extin, WAVEHOST)
#     flux_x = jnp.interp(x, WAVEHOST, flux_extin)
#     return flux_x, absorbed_flux


def f_cont_to_logSFR(f_cont, logM, age, present_age = 13.8, SFH = 'conti'):
    """
    Convert continuum fraction to instantaneous star formation rate.
    
    Derives contemporary SFR from the mass fraction in continuous star
    formation component for composite SFH models.
    
    Parameters
    ----------
    f_cont : float or array_like
        Log10 of mass fraction in continuous SF component.
        For 'bins': array of log10 mass fractions per bin.
    logM : float
        Log10 of total stellar mass in solar masses.
    age : float or array_like
        For 'conti'/'burst': age of burst in Gyr.
        For 'bins': array of bin edges in Gyr.
    present_age : float, optional
        Current age of universe in Gyr (default: 13.8).
    SFH : {'conti', 'burst', 'bins'}, optional
        Star formation history model (default: 'conti').
    
    Returns
    -------
    float
        Log10 of star formation rate in M_sun/yr.
    
    Notes
    -----
    For 'conti' SFH:
        SFR = M_cont / (t_present - 0.2 Gyr)
    
    For 'burst' SFH:
        SFR = M_cont / (0.1 Gyr)  [τ-model timescale]
    
    Also accounts for instantaneous burst component with smoothed
    error function to handle recent starbursts (< 100 Myr).
    
    See Also
    --------
    logSFR_to_f_cont : Inverse transformation (SFR → f_cont)
    get_host_SED : Uses f_cont to build SED
    
    Examples
    --------
    >>> # 10^10 Msun galaxy, 10% in continuous SF
    >>> logSFR = f_cont_to_logSFR(f_cont=-1.0, logM=10.0, age=0.1, SFH='conti')
    >>> print(f"SFR = {10**logSFR:.2f} Msun/yr")
    """
    # calculate the comtempereous SFR from the continuum fraction
    if SFH == 'bins':
        f_cont_real = 10**f_cont 
        f_cont_real = f_cont_real/jnp.sum(f_cont_real) ## 这里不对，没有考虑fix 的情况（假设输入的时候sum(f_cont_real)=1） 应该是sum(f_cont_real_free)/factor + sum(f_cont_real_fix)=1, 
                                                            ## 所以分母应该是factor = sum(f_cont_real_free)/(1-sum(f_cont_real_fix)), 
                                                            ## 最后f_cont_real=[ ( 1 - sum(f_cont_real_fix) ) * f_cont_real_free / sum(f_cont_real_free),  f_cont_real_fix]
        sfr = f_cont_real[0]*10**logM/(age[1]-age[0])/1e9
        logSFR = jnp.log10(sfr)
    else:
        log_M_cont = logM + f_cont
        log_M_inst = logM + jnp.log10(1-10**f_cont)
        #print("sanity check 2:",10**log_M_inst,10**log_M_cont)
        width = 5e-2 #bin width of a single-year SSP representing instantaneous starburst in log(age/yr)
        minage = 8-jnp.log10(age*1e9)
        cdf_end = 0.5 * (1 + erf((minage) / (width * jnp.sqrt(2))))
        cdf_start = 0.5 * (1 + erf((-jnp.log10(age*1e9)) / (width * jnp.sqrt(2))))
        #print("minage,log(age):",minage,jnp.log10(age*1e9))
        fac = cdf_end - cdf_start
        #print("sanity check 3:",cdf_end,cdf_start,fac)
        lsingle = 10**log_M_inst*fac*1e-8
        if SFH == 'conti':
            logSFRc = log_M_cont - 9 - jnp.log10(present_age - 0.2) 
        elif SFH == 'burst':
            logSFRc = log_M_cont - 8
        logSFR = jnp.log10(10**logSFRc + lsingle)
    return logSFR


def logSFR_to_f_cont(logSFR, logM, present_age = 13.8, SFH = 'conti'):
    # assume age is old
    if SFH == 'conti':
        log_M_cont = 9 + logSFR + jnp.log10(present_age - 0.2)
    elif SFH == 'burst':
        log_M_cont = 8 + logSFR
    f_cont = log_M_cont - logM
    return f_cont


def get_AGN_SED(x, logM,logMdot,spin,C_unit):
    """
    Interpolate AGN accretion disk SED for given black hole parameters.
    
    Uses pre-computed thin disk models (Shakura-Sunyaev) with relativistic
    ray-tracing corrections for spinning black holes.
    
    Parameters
    ----------
    x : array_like
        Rest-frame wavelength grid in Angstroms [125, 24700].
    logM : float
        Log10 of black hole mass in solar masses. Range [5, 10].
    logMdot : float
        Log10 of accretion rate in Eddington units. Range [-4, 2].
        Mdot_Edd = L_Edd / (η c^2).
    spin : float
        Dimensionless black hole spin parameter (0 = Schwarzschild, 0.998 = maximal).
        Range [0, 0.99].
    C_unit : float
        Normalization constant (typically 10^38 for erg/s units).
    
    Returns
    -------
    array_like
        Rest-frame AGN spectral luminosity density L_λ in units of C_unit
        (typically 10^38 erg/s/Å).
    
    Notes
    -----
    - Templates assume standard thin disk with Novikov-Thorne efficiency
    - Includes relativistic Doppler boosting and gravitational redshift
    - Higher spin → more luminous at fixed Mdot (higher efficiency)
    - Includes color correction for disk atmosphere
    - Valid for radiatively efficient accretion (Mdot/Mdot_Edd > 0.01)
    
    See Also
    --------
    starburst_disk : Alternative starburst-heated disk model
    cat3Dtorus : AGN torus emission component
    
    Examples
    --------
    >>> wave = jnp.logspace(2, 4.4, 500)  # 100 Å to 25000 Å
    >>> agn_sed = get_AGN_SED(wave, logM=8.0, logMdot=-0.5, spin=0.7, C_unit=1e38)
    
    References
    ----------
    .. [1] Shakura & Sunyaev 1973, A&A, 24, 337
    .. [2] Novikov & Thorne 1973, in Black Holes
    """
    # derive a SED of standard thin disk
    # x:       wavelength, in range[125,24700]
    # logM:    log10(M_BH), in range[5,10]
    # logMdot: log10(dotM), in range[-4,2]
    # spin:    BH spin, in range[0,0.99]
    # Return: restframe L_lambda in ergs/s
    fluxmodel=C_unit*10**(intp_thindisk((spin, logMdot, logM, x))-38)
    return fluxmodel

def sed_to_obse(x,y,z,ebv):
    """
    Transform rest-frame SED to observed frame with IGM and Galactic extinction.
    
    Applies cosmological redshift, IGM absorption (Meiksin 2006), and
    Milky Way foreground extinction (CCM89).
    
    Parameters
    ----------
    x : array_like
        Rest-frame wavelength in Angstroms.
    y : array_like
        Rest-frame spectral luminosity density in 10^38 erg/s/Å.
    z : float
        Source redshift.
    ebv : float
        Milky Way E(B-V) color excess in magnitudes.
    
    Returns
    -------
    xnew : array_like
        Observed-frame wavelength in Angstroms: x_obs = x_rest × (1 + z).
    ynew : array_like
        Observed-frame spectral luminosity density in 10^38 erg/s/Å.
        Includes (1+z)^-3 cosmological dimming factor.
    
    Notes
    -----
    Transformation steps:
    
    1. Redshift wavelength: λ_obs = λ_rest × (1 + z)
    2. Apply IGM absorption: L_obs *= T_IGM(λ_obs, z)
    3. Apply (1+z)^-3 cosmological dimming
    4. Apply Galactic extinction: L_obs *= 10^(-A_λ/2.5)
    
    The IGM transmission T_IGM accounts for Lyman series absorption and
    Lyman continuum opacity from intervening neutral hydrogen.
    
    See Also
    --------
    sed_to_rest : Inverse transformation (observed → rest frame)
    igm_transmission : IGM absorption implementation
    ccm89 : Galactic extinction curve
    
    Examples
    --------
    >>> wave_rest = jnp.logspace(2, 4, 500)
    >>> sed_rest = get_AGN_SED(wave_rest, 8.0, -0.5, 0.7, 1e38)
    >>> wave_obs, sed_obs = sed_to_obse(wave_rest, sed_rest, z=2.0, ebv=0.05)
    """
    '''
    Convert restframe SED to observed frame,
    applying IGM and MW local extinction
    x: wavelength in restframe
    y: L_38 in restframe
    z: redshift
    ebv: Milky Way E(B-V)

    Returns:
    xnew: wavelength in observed frame
    ynew: L_38 in observed frame
    '''
    xnew = x.copy()
    ynew = y.copy()
    av=3.1*ebv
    xnew*=(1+z)
    # fac_igm = igm_transmission(xnew, z) ## xnew is observe frame
    fac_igm     = igm_transmission_madau95(xnew, z)
    ynew*=fac_igm
    ynew/=(1+z)**3
    cm=jccm89(xnew,av,3.1)
    ynew/=jnp.power(10,cm/(2.5))
    return xnew,ynew

def sed_to_rest(x,y,z,ebv):
    '''
    Convert observed frame SED to restframe,
    applying IGM and MW local extinction
    x: wavelength in observed frame
    y: L_38 in observed frame
    z: redshift
    ebv: Milky Way E(B-V)

    Returns:
    xnew: wavelength in restframe
    ynew: L_38 in restframe
    '''
    # fac_igm = igm_transmission(x, z)
    fac_igm     = igm_transmission_madau95(x, z)
    xnew = x.copy()
    ynew = y.copy()
    av=3.1*ebv
    cm=jccm89(xnew,av,3.1)
    ynew*=jnp.power(10,cm/(2.5))
    xnew/=(1+z)
    ynew*=(1+z)**3
    ynew /= fac_igm
    return xnew,ynew

def spowerlaw(x,A,C):
    y = A*jnp.power(x/5100.,-C)/5100
    return y

def brokenpower(x,A,C1,C2):
    y = jnp.where(x<5100.,jnp.power(x/5100.,-C2),jnp.power(x/5100.,-C1))*A/5100
    return y

def Bbody(x, Ltot, Teff):
    # Normalized to R_s
    fx = 6.5992166e+31/(x**5*(jnp.exp(143876866./(Teff*x))-1))/(Teff**4)
    return fx*Ltot




@jit
def igm_transmission(wavelength, redshift):
    """
    Compute intergalactic medium (IGM) transmission.
    
    Calculates wavelength-dependent IGM opacity from Lyman series absorption
    and Lyman continuum blanketing by intervening neutral hydrogen along
    the line of sight. Based on Meiksin (2006) prescriptions.
    
    Parameters
    ----------
    wavelength : array_like
        Observed-frame wavelength(s) in Angstroms.
    redshift : float
        Source redshift. Must be strictly positive.
    
    Returns
    -------
    array_like
        IGM transmission factor T_IGM(λ) ∈ [0, 1].
        Multiply SED by this to apply absorption: F_obs = F_int × T_IGM.
    
    Notes
    -----
    Three opacity components contribute:
    
    1. **Lyman series** (Lyα, Lyβ, ...): Discrete absorption features
       redshifted into spectrum. Each higher-order line has lower optical depth.
       
    2. **Lyman continuum** (λ < 912 Å rest): Strong blanketing blueward of
       Lyman limit. Creates sharp absorption edge at λ_obs = 912(1+z) Å.
       
    3. **Lyman limit systems (LLS)**: Additional opacity from dense clouds
       with high HI column densities.
    
    Total transmission: T = exp(-τ_Lyα - τ_Lyβ - ... - τ_LC - τ_LLS)
    
    At λ_obs < 912 Å (rest UV), transmission → 0 for z > 0.
    IGM becomes transparent at λ_obs > Lyα(1+z) ≈ 1216(1+z) Å.
    
    The function is JIT-compiled for performance with JAX.
    
    See Also
    --------
    sed_to_obse : Applies IGM transmission to rest-frame SEDs
    
    Examples
    --------
    >>> wave_obs = jnp.linspace(1000, 5000, 500)
    >>> T_igm = igm_transmission(wave_obs, z=3.0)
    >>> # Strong absorption blueward of Lyα at 1216*(1+3) = 4864 Å
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(wave_obs, T_igm)
    >>> plt.xlabel('Observed wavelength (Å)')
    >>> plt.ylabel('IGM transmission')
    
    References
    ----------
    .. [1] Meiksin, A. 2006, MNRAS, 365, 807
    """
    n_transitions_low = 10
    n_transitions_max = 31
    gamma = 0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
    n0 = 0.25
    lambda_limit = 912.  # Lyman limit in Angstroms

    lambda_n = jnp.empty(n_transitions_max)
    z_n = jnp.empty((n_transitions_max, len(wavelength)))
    for n in range(2, n_transitions_max):
        lambda_n = lambda_n.at[n].set(lambda_limit / (1. - 1. / float(n * n)))
        z_n = z_n.at[n, :].set((wavelength / lambda_n[n]) - 1.)

    fact = jnp.array([1., 1., 1., 0.348, 0.179, 0.109, 0.0722, 0.0508, 0.0373, 0.0283])

    tau_n = jnp.zeros((n_transitions_max, len(wavelength)))
    
    tau_a = lax.cond(redshift <= 4, 
                     lambda _: 0.00211 * jnp.power(1. + redshift, 3.7), 
                     lambda _: 0.00058 * jnp.power(1. + redshift, 4.5),
                     operand=None)
    
    for n in range(2, n_transitions_max):
        tau_n = tau_n.at[n, :].set(
            lax.cond(
                redshift <= 4,
                lambda _: 0.00211 * jnp.power(1. + z_n[n, :], 3.7),
                lambda _: 0.00058 * jnp.power(1. + z_n[n, :], 4.5),
                operand=None))

    for n in range(3, n_transitions_max):
        condition = z_n[n, :] < 3
        tau_n = tau_n.at[n, :].set(
            jnp.where(condition, 
                      tau_a * fact[n] * jnp.power(0.25 * (1. + z_n[n, :]), (1. / 3.)),
                      tau_n[n, :]))
        condition = z_n[n, :] >= 3
        tau_n = tau_n.at[n, :].set(
            jnp.where(condition, 
                      tau_a * fact[n] * jnp.power(0.25 * (1. + z_n[n, :]), (1. / 6.)),
                      tau_n[n, :]))
        if 5 < n <= 9:
            tau_n = tau_n.at[n, :].set((tau_a * fact[n] * jnp.power(0.25 * (1. + z_n[n, :]), (1. / 3.))))
        elif n > 9:
            tau_n = tau_n.at[n, :].set((tau_n[9, :] * 720. / (float(n) * (float(n * n - 1.)))))

    for n in range(2, n_transitions_max):
        condition = (z_n[n, :] >= redshift) | (z_n[n, :] < 0)
        tau_n = tau_n.at[n, :].set(jnp.where(condition, 0., tau_n[n, :]))


    tau_taun = jnp.sum(tau_n[2:n_transitions_max, :], axis=0)

    z_l = wavelength / lambda_limit - 1.
    condition = z_l < redshift
    tau_l_igm = jnp.where(condition, 
                        0.805 * jnp.power(1. + z_l, 3) * (1. / (1. + z_l) - 1. / (1. + redshift)), 
                        0.)
    term1 = gamma - jnp.exp(-1.)
    term2 = -1.861527558090058
    term3 = jnp.where(condition, 
                    ((1. + redshift) * jnp.power(wavelength / lambda_limit, 1.5) - jnp.power(wavelength / lambda_limit, 2.5)), 
                    0.)
    Term4coefficients = jnp.array([-2.00000000e+00 ,4.76190476e-02 ,-5.12820513e-03 , 6.26566416e-04,
        -7.40740741e-05 , 8.14597589e-06, -8.25000825e-07 , 7.69041467e-08,-6.61640318e-09])
    term4 = jnp.sum(jnp.array(
        [(Term4coefficients[n-1] *
        (jnp.where(condition, 
                    (1. + redshift) ** (2.5 - (3 * n)) * (wavelength / lambda_limit) ** (3 * n) - (wavelength / lambda_limit) ** 2.5, 
                    0.)))
        for n in jnp.arange(1, n_transitions_low)]), axis=0)
    tau_l_lls = jnp.where(condition, n0 * ((term1 - term2) * term3 - term4), 0.)
    condition = z_l < 0
    tau_norm_l_igm = jnp.interp(0, z_l, tau_l_igm)
    tau_norm_l_lls = jnp.interp(0, z_l, tau_l_lls)
    damp_factor = jnp.where(condition, (z_l + 1.) ** 2.75, 0.)
    tau_l_igm = jnp.where(condition, tau_norm_l_igm * damp_factor, tau_l_igm)
    tau_l_lls = jnp.where(condition, tau_norm_l_lls * damp_factor, tau_l_lls)

    return jnp.exp(- tau_taun - tau_l_igm - tau_l_lls)

def calzetti2000(wavelength):
    """Compute the Calzetti et al. (2000) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Calzetti at al. (2000) and Leitherer et al. (2002). The formula
    is given for wavelengths between 91.2 nm and 2200 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in angstrom but converted to nm in the function.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = 0.1*jnp.array(wavelength)

    # Leitherer et al. (2002); between 91.2 nm and 180 nm
    result = (5.472 + 0.671e3 / wavelength -
              9.218e3 / wavelength ** 2 +
              2.620e6 / wavelength ** 3)
    
    # Calzetti at al. (2000) Attenuation between 180 nm and 630 nm
    mask = (wavelength < 630) & (wavelength >= 180)
    result = jnp.where(mask, 2.659 * (-2.156 + 1.509e3 / wavelength -
                                      0.198e6 / wavelength ** 2 +
                                      0.011e9 / wavelength ** 3) + 4.05, result)

    # Calzetti at al. (2000) Attenuation between 630 nm and 2200 nm
    mask = (wavelength >= 630)
    result = jnp.where(mask, 2.659 * (-1.857 + 1.040e3 / wavelength) + 4.05, result)
    return result

def cigale(wavelength, delta, amplitude):
    bump = (1e-2*amplitude * wavelength ** 2 * 35. ** 2 /
            ((1e-2*wavelength ** 2 - 217.5 ** 2) ** 2 +
             1e-2*wavelength ** 2 * 35. ** 2))
    result = calzetti2000(wavelength) * jnp.power(wavelength/5500.,delta) + bump
    return result

def mycigale(wavelength, amplitude):
    """
    Calzetti et al. (2000) attenuation curve with UV bump.
    
    Implements the Calzetti+ (2000) and Leitherer+ (2002) starburst
    attenuation law with optional UV bump feature (2175 Å bump).
    
    Parameters
    ----------
    wavelength : array_like
        Wavelength array in Angstroms.
    amplitude : float
        Amplitude of the UV bump (2175 Å feature).
        amplitude = 0: no bump (pure Calzetti curve)
        amplitude = 3: typical value for Milky Way-like bump
    
    Returns
    -------
    array_like
        Attenuation curve k(λ) where A(λ) = k(λ) × E(B-V).
        To apply: F_obs = F_int × 10^(-A(λ)/2.5).
    
    Notes
    -----
    The Calzetti law applies to starburst galaxies where dust is mixed
    with stars. It differs from Galactic extinction curves (CCM, RV=3.1):
    
    - Grayer (flatter) in UV/optical
    - No 2175 Å bump by default (added optionally here)
    - Normalized to E(B-V) of stellar continuum
    
    Wavelength ranges:
    
    - λ < 180 nm: Leitherer+ (2002) extrapolation
    - 180-630 nm: Calzetti+ (2000) law (optical/UV)
    - λ > 630 nm: Calzetti+ (2000) law (NIR)
    
    The UV bump is parameterized as Drude profile centered at 2175 Å with
    FWHM = 350 Å.
    
    See Also
    --------
    calzetti2000 : Base Calzetti curve without bump
    ccm89 : Galactic extinction curve (for Milky Way foreground)
    
    Examples
    --------
    >>> wave = jnp.linspace(1000, 10000, 1000)
    >>> # No UV bump (pure Calzetti)
    >>> k_cal = mycigale(wave, amplitude=0)
    >>> # With UV bump
    >>> k_bump = mycigale(wave, amplitude=3)
    >>> # Apply to SED with E(B-V) = 0.5
    >>> ebv = 0.5
    >>> flux_atten = flux_int * 10**(-k_bump * ebv / 2.5)
    
    References
    ----------
    .. [1] Calzetti et al. 2000, ApJ, 533, 682
    .. [2] Leitherer et al. 2002, ApJS, 140, 303
    .. [3] Boquien et al. 2019, A&A, 622, A103 (CIGALE implementation)
    """
    bump = (1e-2*amplitude * wavelength ** 2 * 35. ** 2 /
            ((1e-2*wavelength ** 2 - 217.5 ** 2) ** 2 +
             1e-2*wavelength ** 2 * 35. ** 2))
    result = calzetti2000(wavelength) + bump
    return result



# --- FSPS constants (igm_absorb.f90) ---
LYW = jnp.array([1215.67, 1025.72, 972.537, 949.743, 937.803,
                 930.748, 926.226, 923.150, 920.963, 919.352,
                 918.129, 917.181, 916.429, 915.824, 915.329,
                 914.919, 914.576])
LYCOEFF = jnp.array([0.0036, 0.0017, 0.0011846, 0.0009410, 0.0007960,
                     0.0006967, 0.0006236, 0.0005665, 0.0005200, 0.0004817,
                     0.0004487, 0.0004200, 0.0003947, 0.000372, 0.000352,
                     0.0003334, 0.00031644])
LYLIM = 911.75
A_METAL = 0.0017


@partial(jit, static_argnums=())
def igm_transmission_madau95(wavelength, redshift, igm_factor=1.0):
    """
    Madau (1995) IGM transmission, exact port of FSPS igm_absorb.f90.

    Parameters
    ----------
    wavelength : array_like
        Observed-frame wavelength(s) in Angstroms, ascending order
        (the tau cap assumes an ascending grid, as in FSPS/GalfitS).
    redshift : float
        Source redshift.
    igm_factor : float
        FSPS fudge factor scaling the optical depth (default 1.0).

    Returns
    -------
    array_like
        Transmission T ∈ [0, 1]; multiply the SED by it.
    """
    lam = jnp.asarray(wavelength, dtype=jnp.float32)
    z1 = 1.0 + redshift
    xc = lam / LYLIM            # = lobs/lylim in FSPS (lobs = lam_rest*(1+z) = lam_obs)

    tau = jnp.zeros_like(lam)

    # --- Ly-series line blanketing ---
    for i in range(17):
        contrib = LYCOEFF[i] * jnp.power(lam / LYW[i], 3.46)
        tau = tau + jnp.where(lam <= LYW[i] * z1, contrib, 0.0)
    # metal blanketing (Ly-alpha only)
    tau = tau + jnp.where(lam <= LYW[0] * z1,
                          A_METAL * jnp.power(lam / LYW[0], 1.68), 0.0)

    # --- LyC absorption (Madau 1995 eq.16 footnote 3) ---
    xcp = jnp.maximum(xc, 1e-30)   # guard powers of ~0
    t_lyc = (0.25 * xcp**3 * (z1**0.46 - xcp**0.46)
             + 9.4 * xcp**1.5 * (z1**0.18 - xcp**0.18)
             - 0.7 * xcp**3 * (xcp**-1.32 - z1**-1.32)
             - 0.023 * (z1**1.68 - xcp**1.68))
    tau = tau + jnp.where(lam <= LYLIM * z1, t_lyc, 0.0)

    # --- cap: tau(<= argmax) = tau_max (FSPS: vv=MAXLOC(tau,1); tau(1:vv)=tau(vv)) ---
    # The LyC polynomial breaks down blueward of its peak; FSPS flattens tau there.
    idx = jnp.argmax(tau)
    tau = jnp.where(jnp.arange(tau.size) <= idx, tau[idx], tau)

    return jnp.exp(-tau * igm_factor)
