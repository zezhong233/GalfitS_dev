"""
GalfitS Utilities Module
=========================

This module provides utility functions and classes for multi-band galaxy fitting,
including data preparation, parameter initialization, visualization, and result analysis.

Key Components
--------------
- **GSData**: Multi-band imaging and spectroscopic data container
- **Parameter initialization**: Create lmfit.Parameters with priors and constraints
- **Visualization**: Plotting tools for data, models, and residuals
- **Unit conversions**: Flux calibration and magnitude systems
- **PSF handling**: PSF loading and manipulation
- **Result analysis**: Extract and visualize fit results

The module interfaces with the main fitting classes (gsfitter, vdfitter, sedfitter)
and provides high-level workflows for common analysis tasks.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import lmfit
from astropy import units as u
from astropy import constants
from astropy.cosmology import FlatLambdaCDM,z_at_value
from astropy.stats import sigma_clipped_stats
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.table import Table
from photutils.segmentation import detect_sources
from . import sed_interp as sedI
from .sed_interp import _find_data_path
from . import galaxy as GALA
from . import emission_lines as EL
from . import disperser
# NOTE: Avoid top-level import of gsfit to prevent circular imports with gsfit.py
# We'll import the fitter classes lazily inside read_config_file right before use.
from astropy.io import fits
from . import images as IM
import jax.numpy as jnp
import jax,shutil
import matplotlib.pyplot as plt
import re
jax.config.update("jax_enable_x64", False)
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import UnivariateSpline
from importlib.resources import files

#plt.style.use("classic")
#plt.rc('font',family='Times New Roman')

cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.308)


phys_to_image ={
    # flux is Flambda
    # counts_rate/flux for galex
    'galex_fuv' : 0.73e15, # 1538.6
    'galex_nuv' : 5.30e15, # 2315.7
    # counts_rate/flux for panstarrs
    'panstarrs_g'  : 2.12e18,  #4810
    'panstarrs_r'  : 3.50e18,  #6170
    'panstarrs_i'  : 5.20e18,  #7520
    'panstarrs_z'  : 6.89e18,  #8660
    'panstarrs_y'  : 8.50e18,  #9620
    # nanomaggies/flux for SDSS
    'sloan_u_nm' : 1.15e17, #3543 nanomaggies/flux for SDSS
    'sloan_g_nm' : 2.09e17, #4770
    'sloan_r_nm' : 3.57e17, #6231
    'sloan_i_nm' : 5.34e17, #7625
    'sloan_z_nm' : 7.66e17, #9134
    # below for stripe82 images, counts rate units
    'sloan_u' :2.4747728902890766e+19,
    'sloan_g' :4.4856956453039735e+19,
    'sloan_r' :7.654354072010808e+19,
    'sloan_i' :1.1462325610262926e+20,
    'sloan_z' :1.6448074325343234e+20,
    # nanomaggies/flux for DESI
    'desi_s_g' : 2.14e17, # 4829
    'desi_s_r' : 3.80e17, # 6436
    'desi_s_z' : 7.74e17, # 9178
    'desi_n_g' : 2.10e17, # 4776
    'desi_n_r' : 3.78e17, #6412
    'desi_n_z' : 7.88e17, #9264
    # CGS survey
    'cgs_r' : 2.611592e+18,
    # 2MASS use dN
    'j' : 7.657e17, #12350
    'h' : 1.674e18, #16608
    'ks' : 2.459e18, #21590
    # ALLWISE image, counts rate units
    'wise_ch1' : 2.021673e+19, #34000
    'wise_ch2' : 2.6562486e+19, #46000
    'wise_ch3' : 2.4610223e+20, #120000
    'wise_ch4' : 3.1332743e+19, #220000
    # hsc
    'hsc_g' : 1.342599746262021e+19, # 4816.12
    'hsc_r' : 2.2495771089545773e+19, # 6234.11
    'hsc_i' : 3.468158753845135e+19, # 7740.58
    'hsc_z' : 4.819885768625678e+19, # 9125.20
    'hsc_y' : 5.5363489398266675e+19, # 9779.93
    # DECam
    'decam_u' : 1.1577365689039841e+20,
    'decam_g' : 2.055300494539333e+20,
    'decam_r' : 3.7863704438463935e+20,
    'decam_i' : 5.646575905544513e+20,
    'decam_z' : 7.877257069291907e+20,
    'decam_y' : 9.352664977634097e+20,
    # JWST Wen's image 
    'nircam_f070w': 1.6549452784062153e+20, # 7000.00
    'nircam_f090w': 2.72284697e+20, # 8984.98
    'nircam_f115w': 2.100850e+20,  # 11542.61
    'nircam_f140m': 6.588963655011429e+20,
    'nircam_f150w': 3.5514056e+20, # 15007.44
    'nircam_f162m': 8.83146025299659e+20,
    'nircam_f182m': 1.1357407856536832e+21,
    'nircam_f187n': 9.252032957757727e+21,
    'nircam_f200w': 6.2359538e+20, # 19886.48
    'nircam_f210m': 1.4649173809667134e+21, # 27617.40
    'nircam_f250m': 2.090223943512105e+21, # 27617.40
    'nircam_f277w': 1.202687048e+21, # 27617.40
    'nircam_f300m': 2.994370201538456e+21, # 35683.62
    'nircam_f356w': 2.00782041e+21, # 35683.62
    'nircam_f360m': 4.3791610308612635e+21,
    'nircam_f410m': 2.627747889e+21, # 40822.38
    'nircam_f444w': 3.05874870647e+21, # 44043.15 
    'miri_f770w':2.3141492028087479e+21,
    # swift UVOT
    'swift_u':6.11729639e15,
    'swift_b':6.925923785e15,
    'swift_v':3.89771479e15,
    'swift_w1':2.361020409e15,
    'swift_m2':1.20773749e15,
    'swift_w2':1.5044131e15,
    # XMM OM
    'xmmom_v': 4.016064257e+15,
    'xmmom_b': 7.751937984e+15,
    'xmmom_u': 5.154639175e+15,
    'xmmom_w1': 2.100840336e+15,
    'xmmom_m2': 4.545454545e+14,
    'xmmom_w2': 1.751313485e+14,
    # hst
    'wfc3_f438w': 1.5061352645923459e+18,
    'wfc3_f105w': 3.2909271048457126e+19,
    'wfc3_f140w': 6.775526338130881e+19,
    'wfc3_f475w': 4.0688115786659354e+18,
    'wfc3_f814w': 6.769243266295261e+18,
    'wfc3_f555w': 5.526021732406707e+18,
    'wfc3_f125w': 4.447722514776113e+19,
    'wfc3_f110w': 6.547018163850784e+19,
    'wfc3_f547m': 2.2079148359556813e+18,
    'wfc3_f606w': 8.791595234955382e+18,
    # Spitzer-IRAC MJy/Sr to Flambda/pixel
    'ch1':5.10895484e+18,
    'ch2':7.98274194e+18,
    'ch3':1.32612069e+19,
    'ch4':2.52294066e+19
    }
effective_wave ={
    # flux is Flambda
    # counts_rate/flux for galex
    'galex_fuv' : 1538.6, 
    'galex_nuv' : 2315.7,
    # counts_rate/flux for panstarrs 
    'panstarrs_g' : 4810,  
    'panstarrs_r' : 6170,
    'panstarrs_i' : 7520,
    'panstarrs_z' : 8660,
    'panstarrs_y' : 9620,
    # nanomaggies/flux for SDSS
    'sloan_u' : 3543,
    'sloan_g' : 4770,
    'sloan_r' : 6231, 
    'sloan_i' : 7625,
    'sloan_z' : 9134,
    # nanomaggies/flux for DESI
    'desi_s_g' : 4829,
    'desi_s_r' : 6436,
    'desi_s_z' : 9178,
    'desi_n_g' : 4776,
    'desi_n_r' : 6412,
    'desi_n_z' : 9264,
    'desi_i': 7840,
    # CGS survey
    'cgs_u' : 3551.05,
    'cgs_b' : 4369.53,
    'cgs_v' : 5467.57,
    'cgs_r' : 6357.35,
    'cgs_i' : 7828.65,
    # 2MASS use dN  
    'j' : 12350,
    'h' : 16608, 
    'ks' : 21590,
    # ALLWISE image, counts rate units
    'wise_ch1' : 34000,
    'wise_ch2' : 46000,
    'wise_ch3' : 120000,
    'wise_ch4' : 220000,
    # Subaru-HSC
    'hsc_g' : 4816.12,
    'hsc_r' : 6234.11,
    'hsc_i' : 7740.58,
    'hsc_z' : 9125.20,
    'hsc_y' : 9779.93,
    # DECam
    'decam_u' : 3550,
    'decam_g' : 4730,
    'decam_r' : 6420,
    'decam_i' : 7840,
    'decam_z' : 9260,
    'decam_y' : 10090,
    # Subaru FOCAS
    'focas_U' : 3600.,
    'focas_B' : 4400.,
    'focas_V' : 5500.,
    'focas_R' : 6600.,
    'focas_I' : 8050.,
    #Subaru Suprime
    'suprime_B' : 4439.35,
    'suprime_V' : 5448.15,
    'suprime_i' : 7629.33,
    'suprime_z' : 9017.60,
    # JWST 
    'nircam_f070w': 7043.721411041347,
    'nircam_f090w': 8984.98,
    'nircam_f115w': 11542.61,  
    'nircam_f140m': 14054.613516500445,
    'nircam_f150w': 15007.44,
    'nircam_f162m': 16271.463297896555,
    'nircam_f182m': 18452.27687805039,
    'nircam_f187n': 18737.22,
    'nircam_f200w': 19886.48,
    'nircam_f210m': 20956.41149309253,
    'nircam_f250m': 25032.646167918316,
    'nircam_f277w': 27617.40,
    'nircam_f300m': 29961.4686458197,
    'nircam_f335m': 33623.44734205056,
    'nircam_f356w': 35683.62,
    'nircam_f360m': 36233.126420440145,
    'nircam_f410m': 40822.38,
    'nircam_f430m': 42818.637,
    'nircam_f444w': 44043.15,
    'nircam_f460m': 46305.45,
    'nircam_f480m': 48152.79,
    'miri_f770w': 76604.7,
    'miri_f1000w': 99620.625,
    'miri_f1280w': 128313.970,
    'miri_f1500w': 150760.77,
    'miri_f1800w': 179922.53,
    'miri_f2100w': 208425.266,
    # HST
    'acs_f435w':4317.61,
    'acs_f606w':5809.26,
    'acs_f775w':7665.08,
    'acs_f814w':7973.39,
    'acs_f850lp':9143.95,
    'wfc3_f225w': 2380.02,
    'wfc3_f275w': 2704.27,
    'wfc3_f336w':3358.18,
    'wfc3_f438w':4327.16,
    'wfc3_f475w':4777.38,
    'wfc3_f547m':5471.32,
    'wfc3_f555w':5305.85,
    'wfc3_f606w':5921.97,
    'wfc3_f814w':8024.22,
    'wfc3_f105w':10430.83,
    'wfc3_f110w':11534.52,
    'wfc3_f125w':12363.55,
    'wfc3_f140w':13734.66,
    'wfc3_f160w':15278.47,
    # swift UVOT
    'swift_u':3465.00,
    'swift_b':4392.00,
    'swift_v':5468.00,
    'swift_w1':2600.00,
    'swift_m2':2246.00,
    'swift_w2':1928.00,
    # XMM-OM
    'xmmom_u':3440.00,
    'xmmom_b':4500.00,
    'xmmom_v':5500.00,
    'xmmom_m2':2310.00,
    'xmmom_w1':2910.00,
    'xmmom_w2':2120.00,
    # Spitzer-IRAC
    'ch1':36000.,
    'ch2':45000.,
    'ch3':58000.,
    'ch4':80000.,
    # Spitzer-MIPS
    'mips_24um':236800.,
    # Keck-LRIS
    'Un_lris':3450.,
    #'B_lris':4370.,
    'G_lris':4731.,
    #'V_lris':5437.,
    'B_lris':4377.,
    'V_lris':5473.,
    'R_lris':6417.,
    'I_lris':7599.,
    'Rs_lris':6809.,
    # Keck-MOSFIRE
    'mosfire_Y':10480.,
    'mosfire_J':12530.,
    'mosfire_H':16370.,
    'mosfire_Ks':21470.,
    # Herschel
    'herschel_psw': 2516960.,
    'herschel_pmw': 3524879.,
    'herschel_plw': 5116700.,
    # JCMT SCUBA-2
    'JCMT_450um': 4500057.081,
    'JCMT_850um': 8633461.375,
    # NOEMA bands
    'NOEMA_1mm': 13050697.157,
    'NOEMA_3mm': 33996815.682,
    # VISTA
    'vista_z' : 8798.309,
    'vista_y' : 10214.932,
    'vista_j' : 12535.58,
    'vista_h' : 16454.893,
    'vista_ks' : 21540.533,
    # UKIDSS
    'ukidss_i' : 8510.397444745571,
    'ukidss_z' : 9576.269546031692,
    'ukidss_y' : 10193.748930096594,
    'ukidss_j' : 12468.323570710187,
    'ukidss_h' : 16315.757397676653,
    'ukidss_k' : 21992.18163118593,
}

def normimg(image, min=0, max=None, sky=0, frac = 0.5):
    """
    Normalize image with arcsinh stretch for visualization.
    
    Applies arcsinh scaling for better dynamic range display. Handles both
    positive and negative pixel values (e.g., residuals). Sky-subtracted before
    scaling.
    
    Parameters
    ----------
    image : array_like
        Input image array.
    min : float, default=0
        Lower threshold for arcsinh stretch.
    max : float, optional
        Upper threshold. If None, uses image maximum.
    sky : float, default=0
        Sky background to subtract before scaling.
    frac : float, default=0.5
        Fraction parameter controlling stretch behavior near zero.
    
    Returns
    -------
    array_like
        Normalized image in range suitable for display.
    
    Notes
    -----
    Arcsinh stretch preserves faint features better than linear or log scaling.
    Commonly used for displaying astronomical images with large dynamic range.
    """
    if max is None:
        max = jnp.nanmax(image)
    image = jnp.array(image - sky)
    image = jnp.where(image > min, (1-frac)*(jnp.arcsinh(image)-jnp.arcsinh(min))/(jnp.arcsinh(max)-jnp.arcsinh(min))+frac, image*frac/min)
    image = jnp.where(image < -frac, -(1-frac)*(jnp.arcsinh(-image)-jnp.arcsinh(frac))/(jnp.arcsinh(frac*max/min)-jnp.arcsinh(frac))-frac, image)
    return image

def parse_config_file(filename):
    """
    Parse GalfitS configuration file into dictionary.
    
    Reads configuration file with format "KEY) value # comment" and converts
    to Python dictionary. Used by read_config_file to load fitting parameters.
    
    Parameters
    ----------
    filename : str
        Path to configuration file.
    
    Returns
    -------
    dict
        Configuration dictionary with keys like 'R1)', 'I1)', 'M1)', etc.
        Values auto-converted to Python types (lists, floats, strings).
    
    Notes
    -----
    Config file format:
    
    - R1) target_name
    - R2) [ra, dec]
    - R3) redshift
    - I1) [image_file, ext]
    - M1) model_type
    
    See Also
    --------
    read_config_file : Main config loader
    write_data_config : Create config files
    """
    config_data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Remove comments and whitespace
            line = line.split('#')[0].strip()
            if not line:  # Skip empty lines
                continue
            # Split into key and value
            #print (line)
            key, value = line.split(')', 1)
            key = key.strip() + ')'  # add the closing bracket back to the key
            # Try to evaluate value as Python literal, else leave as string
            try:
                value = eval(value.strip())
            except (NameError, SyntaxError):
                value = value.strip()
            # Store in dictionary
            config_data[key] = value
    return config_data

def generate_cons_func(par_total,additionals=[]):
    pardict = par_total.valuesdict()
    exelist = []
    exelist.append("def Update_Constraints(pardictlc):")
    Haveconstraints = False
    for el in pardict:
        if par_total[el].expr is not None:
            par_total[el].vary = False
            expr = par_total[el].expr
            opra = []
            oploop = []
            transexp = "pardictlc['{0}']=".format(el)
            for loop, a in enumerate(expr):
                if (a == '*') | (a == '/'):
                    opra.append(a)
                    oploop.append(loop)
            oploop.append(len(expr))
            transexp += expr[0:oploop[0]]
            for loop, a in enumerate(opra):
                transexp += a
                transexp += "pardictlc['{0}']".format(expr[oploop[loop]+1:oploop[loop+1]])
            exelist.append("    "+transexp)
            Haveconstraints = True

    # Keep Z-mode mass normalization even when constraints are regenerated.
    # These constants are copied from sed_interp.EZY_logM and rounded to 3 decimals.
    ez_logm_vals = [
        10.043, 11.459, 10.108, 12.427, 8.540, 10.459,
        8.560, 10.116, 10.909, 11.708, 9.943, 10.250,
        10.162, 10.068, 10.874, 10.199, 11.178, 10.885,
    ]
    for el in pardict:
        if not el.startswith('logM_'):
            continue
        pname = el[5:]
        ez_keys = ['ezlogn{0}_{1}'.format(i_ez + 1, pname) for i_ez in range(18)]
        if all(k in pardict for k in ez_keys):
            terms = '+'.join([
                "10**(pardictlc['{0}']+{1:.3f})".format(k, ez_logm_vals[i_ez])
                for i_ez, k in enumerate(ez_keys)
            ])
            exelist.append("    pardictlc['{0}'] = jnp.log10({1})".format(el, terms))
            par_total[el].vary = False
            Haveconstraints = True

    for addc in additionals:
        exelist.append("    "+addc)
        Haveconstraints = True
    function_string = "\n".join(exelist)
    #print ("we go here constraints")
    if Haveconstraints is False:
        #print ("No constraints")
        function_string = "def Update_Constraints(pardictlc):\n    pass"
    return function_string

def _extract_section_label(key, prefix):
    """Extract section label from keys like Iaa10), Aab2), etc.
    key is "Iaa1)", prefix is I.
    """
    match = re.match(r'^' + re.escape(prefix) + r'([A-Za-z]+)\d+\)$', key)
    if match is None:
        return None
    return match.group(1)

def _unique_keep_order(values):
    """Remove duplicates while preserving insertion order."""
    return list(dict.fromkeys(values))

def _purify_labels(labels):
    """Remove spurious labels caused by sub-prefix keys (e.g., 'aT' when 'a' exists).

    When config keys like NaT1) are parsed, _extract_section_label returns 'aT',
    but 'T' is a sub-prefix (template parameter), not part of the component label.
    If both 'a' and 'aT' exist, 'aT' is spurious and should be removed.
    """
    sorted_labels = sorted(labels, key=len)
    purified = []
    for label in sorted_labels:
        if not any(label.startswith(existing) and len(label) > len(existing)
                   for existing in purified):
            purified.append(label)
    return purified

def _z_to_agebin_rescale(z_fit, n_agebin=7, manual_agelist=None):
    '''
    set agebin used for cosmic prior calculation.

    Params
    ----------
    z_fit: float
        zred of galaxy in interest.
    n_agebin: int, default 7
        number of agebin and age_list
        Should be same with age_list.
    manual_agelist: list or None
        you can manually add a lookback time list(zp is z_fit) to t_cur(z_fit), for debug.
    Return
    ----------
    age bin: lookback time bins with Tzp is local univ t.
    '''
    z_max   = 15 # time for star formation start
    a_min1, a_min2  = 0.03, 0.1
    if manual_agelist is not None:
        t_cur   = cosmo.lookback_time(z_fit).to(u.Gyr).value
        age_bin     = manual_agelist + t_cur
        return age_bin
    age_bin     = np.zeros(n_agebin)
    age_bin[0]  = cosmo.lookback_time(z_fit).to(u.Gyr).value
    t_max   = cosmo.lookback_time(z_max).to(u.Gyr).value
    tbin_max    = age_bin[0] + 0.9 * (t_max - age_bin[0])
    age_bin[-1]     = t_max
    age_bin[-2]     = tbin_max

    if z_fit <= 3.:
        age_bin[1]  = age_bin[0] + a_min1
        age_bin[2]  = age_bin[0] + a_min2
        i_age   = 3
        n_bins  = n_agebin  - 2
    else:
        age_bin[1]  = age_bin[0] + 10**(7.1295-9)
        i_age   = 2
        n_bins  = n_agebin  - 1
    
    age_bin = (
        age_bin[:i_age].tolist()[:-1] + np.logspace(np.log10(age_bin[i_age-1]), np.log10(tbin_max), n_bins).tolist() + 
        [t_max]
    )
    return age_bin


def _generate_cosmic_sfh_prior(gal_z, age_list, logM, Cosmic_SFR_prior_path=None, plot: bool = False, outpath = None, manual_agelist=None):

    '''
    return the sfh prior based on the Cosmic SFRD in Behroozi(2019) and method in Bingjie Wang(2023).
    ---------------
    Parameters:
    gal_z: float the redshift of profile's galaxy.
    age_list: list
        the lookback time list in Gyr.
    logM: float 
        the log10(mass) of galaxy to considering mass dependence.

    Cosmic_SFR_prior_path: str
        the path to cosmic sfr curve.

    =========================
    Notes: this function will not be used. see galfits/templates/csfr_grid, you can directly interplate with gal_z, 
                    and logM to derive the sfr list, then multiple with age_list to get f_cont_prior -> log10() -> final results.
    '''
    if Cosmic_SFR_prior_path is None:
        Cosmic_SFR_prior_path = os.path.join(_find_data_path(), "templates/behroozi_19_sfrd.tab")

    csfr_tb     = Table.read(Cosmic_SFR_prior_path, format="ascii")
    _zred_b19, _csfrd_b19, _t_univ_b19, _t_lb_b19   = csfr_tb["zred"].value, csfr_tb["sfrd"].value, csfr_tb["t_univ"].value, csfr_tb["t_lookback"].value
    sfrd_z_interp   = UnivariateSpline(_t_lb_b19[::-1], _csfrd_b19[::-1], s=0, ext=3)

    t_univ  = cosmo.age(gal_z).to(u.Gyr).value
    t_z0    = cosmo.age(0).to(u.Gyr).value

    def shift_factor(logM):
        logM    = min(max(logM, 9), 12) 
        return 1/3 * logM - 3.2  ##
    
    tmin_thres  = cosmo.age(10).to(u.Gyr).value
    tmax_thres  = cosmo.age(0.15).to(u.Gyr).value

    shift_delta     = 10**shift_factor(logM)
    
    t_univ_shift    = t_univ * shift_delta
    if manual_agelist: 
        manual_agelist  = np.array(manual_agelist)
        manual_agelist *= shift_delta
    t_univ_shift    = np.clip(t_univ_shift, a_min=tmin_thres, a_max=tmax_thres)
    z_shift     = z_at_value(cosmo.age, t_univ_shift*u.Gyr)
        

    agelim_shift  = _z_to_agebin_rescale(z_shift, n_agebin=len(age_list)-1, manual_agelist=manual_agelist)
    n_sfrbin    = len(agelim_shift)-1

    f_cont_list     = np.zeros(n_sfrbin)
    sfr_shifted     = np.zeros(n_sfrbin)

    for i in range(n_sfrbin):
        a   = agelim_shift[i]
        b   = agelim_shift[i+1]
        sfr_shifted[i]  = sfrd_z_interp.integral(a=a, b=b) / (b-a)
        f_cont_list[i]  = sfr_shifted[i] * (age_list[i+1]-age_list[i])
    f_cont_list     /= jnp.sum(f_cont_list)
    logf_cont_list  = jnp.log10(f_cont_list)

    if plot: 
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(_t_univ_b19, _csfrd_b19, label="True Cosmic SFR", color="grey", alpha = 0.7)
        ax.set_xlabel(r"$t_{univ}$ (Gyr)", fontsize=15)
        ax.set_ylabel(r"$Cosmic\,\, SFR$ $(M_{\odot}\cdot yr^{-1} \cdot Mpc^{-3})$", fontsize=15)
        ax.set_title(f"Cosmic SFRD in Behroozi(2019), logM: {logM:.2f}, gal_z: {gal_z:.2f}", fontsize=15)
        ax.axvline(t_univ_shift, c = "b", lw = 2, label = "t_start", alpha = 0.5)
        ax.axvline(t_univ, c = "r", lw = 2, label = "t_universe", alpha = 0.5)
        ax.set_ylim(-0.005, 0.072)
        for i in range(len(age_list)):
            label = "age_list" if i == 0 else None
            ax.axvline(t_z0 - agelim_shift[i],ymin = 0.06, ymax = 1, alpha = 0.3, c = "green", label = label)
        ax.legend()

        if outpath is not None:
            plt.savefig(outpath, dpi=200)
        return logf_cont_list, fig

    return logf_cont_list

def read_config_file(config, workplace, targname = None, priorpath = None, fixagnlinepro = False, dataonly = False, vectorize = True, silent=False):
    """
    Load GalfitS configuration file and initialize fitting setup.
    
    Main entry point for setting up fit from configuration file. Creates galaxy
    models, loads images/spectra, initializes parameters with priors. Returns
    fully configured fitter object ready for optimization.
    
    Parameters
    ----------
    config : str
        Path to configuration file.
    workplace : str
        Working directory for data files.
    priorpath : str, optional
        Path to prior configuration file. Default None (no priors).
    fixagnlinepro : bool, default=False
        Whether to fix AGN line profiles during fit.
    dataonly : bool, default=False
        If True, only load data without creating models/fitter.
    vectorize : bool, default=True
        Use vectorized operations for speed.
    
    Returns
    -------
    Myfitter : gsfitter or specialized fitter
        Configured fitter object (mbifitter, imagefitter_phot, imagefitter_SED,
        vdfitter, or sedfitter depending on data type).
    par_total : lmfit.Parameters
        Full parameter object with bounds and priors.
    GS_data : galfitS_data
        Data container with loaded images/spectra.
    
    Notes
    -----
    **Config file structure:**
    
    - R1-R3: Region info (target, coordinates, redshift)
    - I sections: Image definitions
    - S sections: Spectrum definitions
    - A sections: Atlas groupings
    - M sections: Model definitions
    - P sections: Parameter values and bounds
    
    **Fitting modes auto-detected:**
    
    - 'images - photometry': Multi-band images, photometry-only
    - 'images - SED': Multi-band with SED modeling
    - 'images + spectra - SED': Combined imaging + spectroscopy
    - 'spectrum - SED': Spectrum-only fitting
    
    Examples
    --------
    >>> # Basic fit
    >>> fitter, params, data = read_config_file('config.txt', './')
    >>> result = fitter.optimizer(num_steps=5000)
    >>> 
    >>> # With priors
    >>> fitter, params, data = read_config_file(
    ...     'config.txt', './', priorpath='priors.txt'
    ... )
    
    See Also
    --------
    parse_config_file : Config file parser
    write_data_config : Create config files
    """
    config_dict     = parse_config_file(config)

    if targname is None:
        targname    = config_dict['R1)']
    ra,dec  = config_dict['R2)']
    z       = config_dict['R3)']
    regionz     = z
    cosmo   = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.308)
    d       = cosmo.luminosity_distance(z)/(1+z)
    da      = cosmo.angular_diameter_distance(z)
    dc      = d.to(u.cm)
    dis     = dc.value
    C_unit  = 1e38/(4*np.pi*dis**2)
    scale   = da.to(u.kpc).value*np.pi/180/3600 # kpc per arcsec
    # Find how many keys start with 'I'
    imagelabel  = []
    speclabel   = []
    profilelabel = []
    imatalslabel = []
    galaxylabel = []
    nucleilabel = []
    fgslabel    = []
    
    for key in config_dict.keys():
        ilabel = _extract_section_label(key, 'I')
        if ilabel is not None:
            imagelabel.append(ilabel)
        slabel = _extract_section_label(key, 'S')
        if slabel is not None:
            speclabel.append(slabel)
        alabel = _extract_section_label(key, 'A')
        if alabel is not None:
            imatalslabel.append(alabel)
        plabel = _extract_section_label(key, 'P')
        if plabel is not None:
            profilelabel.append(plabel)
        glabel = _extract_section_label(key, 'G')
        if glabel is not None:
            galaxylabel.append(glabel)
        nlabel = _extract_section_label(key, 'N')
        if nlabel is not None:
            nucleilabel.append(nlabel)
        flabel = _extract_section_label(key, 'F')
        if flabel is not None:
            fgslabel.append(flabel)
    imagelabel      = _unique_keep_order(imagelabel)
    speclabel       = _unique_keep_order(speclabel)
    imatalslabel    = _unique_keep_order(imatalslabel)
    profilelabel    = _purify_labels(_unique_keep_order(profilelabel))
    galaxylabel     = _unique_keep_order(galaxylabel)
    nucleilabel     = _purify_labels(_unique_keep_order(nucleilabel))
    fgslabel        = _unique_keep_order(fgslabel)

    msr_label   = []
    mar_label   = []
    mmr_label   = []
    agnp_label  = []
    sfhp_label  = []
    sfh_tp_label = []
    
    if priorpath is not None: # read prior path 
        pfile = parse_config_file(priorpath)
        for key in pfile.keys():
            msrl = _extract_section_label(key, 'MSR')
            if msrl is not None:
                msr_label.append(msrl)
            mmrl = _extract_section_label(key, 'MMR')
            if mmrl is not None:
                mmr_label.append(mmrl)
            marl = _extract_section_label(key, 'MAR')
            if marl is not None:
                mar_label.append(marl)
            agnl = _extract_section_label(key, 'AGN')
            if agnl is not None:
                agnp_label.append(agnl)
            sfhl = _extract_section_label(key, 'SFH')
            if sfhl is not None:
                sfhp_label.append(sfhl)
            sfh_tl = _extract_section_label(key, "SFH_T")
            if sfh_tl is not None:
                sfh_tp_label.append(sfh_tl)
        msr_label   = _unique_keep_order(msr_label)
        mmr_label   = _unique_keep_order(mmr_label)
        mar_label   = _unique_keep_order(mar_label)
        agnp_label  = _unique_keep_order(agnp_label)
        sfhp_label  = _unique_keep_order(sfhp_label)
        sfh_tp_label = _unique_keep_order(sfh_tp_label)

    # read images
    Imagelist   = []
    Allbands    = []
    par_total   = lmfit.Parameters() ## lmfit parameters objects.
    totpix      = 0
    imsum   = {}
    filter_wave = []
    BandDuplication = False

    for image_ke in imagelabel:
        image_key   = 'I' + image_ke
        image_config    = {key: config_dict[image_key + str(i+1) + ')'] 
                        for i, key in enumerate(['image', 'band', 'sigma', 'psf', 'psf_sampling', 'mask', 'unit', 'fitting_area', 'conversion', 'magzp', 'skymodel', 'skyparameter', 'shift', 'shift_param', 'use_sed'])}
        # The image and psf values are lists with 2 elements, so we need to parse them.
        image_file, image_hdu   = image_config['image'][1:-1].split(',')
        psf_file, psf_hdu   = image_config['psf'][1:-1].split(',')
        mask_file, mask_hdu = image_config['mask'][1:-1].split(',')
        sigtemp     = image_config['sigma'][1:-1].split(',')

        if len(sigtemp) == 2:
            sigma_fie   = sigtemp[0]
            sigma_hdu   = sigtemp[1]
            sigma_norm  = 1
        elif len(sigtemp) == 3:
            sigma_fie = sigtemp[0]
            sigma_hdu = sigtemp[1]
            sigma_norm = jnp.array(sigtemp[2],dtype=jnp.float32) # scale factor for sigma image.

        if image_config['use_sed'] > 0:
            im = IM.image(image_file, hdu=int(image_hdu), unit=image_config['unit'],fitSED = True)
        else:
            im = IM.image(image_file, hdu=int(image_hdu), unit=image_config['unit'],fitSED = False)

        if im.photopoint & (not silent):
            print ("{0} is a photometric point".format(image_config['band']))
            
        imcut,cp = im.img_cut(ra,dec, image_config['fitting_area'],sigma_clipped=False)
        #print (im.coordinates_transfer_para)
        ny,nx = imcut.shape
        im.cshape0 = (ny,nx)
        im.band = image_config['band']
        

        # If the image is a grism image
        if image_config['use_sed'] == 2:
            axeconf = disperser.aXeConfBase(im.band)
            nyi,nxi = im.data.data.shape
            if im.grism_cut and im.grism_cpi is not None:
                cp = im.grism_cpi # the cut position of cpi on the original image
                ny = cp[3]-cp[2]
                nx = cp[1]-cp[0]
                im.cshape0 = (ny,nx)
                idisperser = disperser.GSDisperser(
                                nx=axeconf.ccd_sx, ny=axeconf.ccd_sy,
                                axeconf=axeconf,
                                nys=ny,
                                nxs=nx,
                                ixmin=cp[0],iymin=cp[2],
                                bin =3)
            else:
                idisperser = disperser.GSDisperser(
                nx=nxi, ny=nyi,
                axeconf=axeconf,
                nys=ny,
                nxs=nx,
                ixmin=cp[0],iymin=cp[2],
                bin =3)
                im.grism_cpi = cp
            sxmin = int(idisperser.igxmin)
            symin = int(idisperser.igymin)
            nyss = int(idisperser.nyd + idisperser.nys)
            nxss = int(idisperser.nxd + idisperser.nxs)
            totpix += nyss*nxss
            ### Need to change here!
            sxmin_c = max(sxmin,0)
            sxmax_c = min(sxmin+nxss,idisperser.axeconf.ccd_sx) #ccd_sx is the size of ccd in x direction
            symin_c = max(symin,0)
            symax_c = min(symin+nyss,idisperser.axeconf.ccd_sy) #ccd_sy is the size of ccd in y direction
            if im.grism_cut:
                if im.grism_cpg is not None:
                    cp = im.grism_cpg
                im.cut_image = jnp.array(im.data.data,dtype=jnp.float32)
            else:
                cp = [sxmin_c,sxmax_c,symin_c,symax_c]
                im.grism_cpg = cp
                im.cut_image = jnp.array(im.data.data[cp[2]:cp[3],cp[0]:cp[1]],dtype=jnp.float32)
            imcut = im.cut_image
            im.disperser = idisperser
            if im.band not in Allbands:
                filter_wave.append(im.disperser.sens_wave)
        else:
            totpix += ny*nx
            im.resp = jnp.array(np.genfromtxt(GALA.filterpath / im.band, dtype=float, names=None),dtype=jnp.float32)
            filterx = im.resp[:,0]
            filtery = im.resp[:,1]
            im.respnorm = jnp.trapezoid(filtery,x=filterx)
            if im.band not in Allbands:
                filter_wave.append(im.resp[:,0])

        if im.band in Allbands:
            BandDuplication = True
        else:
            Allbands.append(im.band)

        #print (image_config['band'],ny,nx)
        psf = fits.open(psf_file)
        psfdata = jnp.array(psf[int(psf_hdu)].data,dtype=jnp.float32)
        psfdata = jnp.where(jnp.isnan(psfdata),0,psfdata)
        psfdata = jnp.where(jnp.isinf(psfdata),0,psfdata)
        im.PSF = psfdata/jnp.sum(psfdata)
        if image_config['conversion'] == -1:
            conversion = phys_to_image[im.band]
        else:
            conversion = jnp.array(image_config['conversion'],dtype=jnp.float32)

        im.phys_to_counts_rate =  jnp.float32(conversion * C_unit)
        imsum[im.band] = jnp.nansum(imcut)
        if not silent:
            print ("conversion factor is ", conversion)
            print ("{0} band image unit to counts rate is ".format(im.band), im.phys_to_counts_rate)
        try:
            maskimg = fits.open(mask_file)[int(mask_hdu)].data
            if im.grism_cut:
               maskcut = maskimg
            else:
                maskcut = maskimg[cp[2]:cp[3],cp[0]:cp[1]]
            cutmask = jnp.array(maskcut,dtype=bool)
        except FileNotFoundError:
            if not silent:print ("mask image not found, use the zero mask image")
            cutmask = jnp.zeros_like(imcut,dtype=bool)
        im.cut_mask_image=cutmask
        sky_mean, sky_median, sky_std = sigma_clipped_stats(imcut, mask = cutmask, sigma=3.0, maxiters=5)
        if (im.band == 'galex_fuv') | (im.band == 'galex_nuv'):
            sky_std = 0.25*np.nanstd(imcut[imcut>3*sky_std])
        # print (im.band, 'std', sky_std)
        
        im.cut_image = jnp.where(jnp.isnan(im.cut_image),0,im.cut_image)
        im.cut_image = jnp.where(jnp.isinf(im.cut_image),0,im.cut_image)
        try:
            sigmaimg = fits.open(sigma_fie)[int(sigma_hdu)].data
            if im.grism_cut:
                sigmacut = sigmaimg
            else:
                sigmacut = sigmaimg[cp[2]:cp[3],cp[0]:cp[1]]
            im.cut_sigma_image = jnp.array(sigmacut)
        except FileNotFoundError:
            print ("sigma image not found, use the default calculated sigma image")
            sigmaimg = im.cal_sigma_image(500)
            if im.grism_cut:
                sigmacut = sigmaimg
            else:
                sigmacut = sigmaimg[cp[2]:cp[3],cp[0]:cp[1]]
            im.cut_sigma_image = jnp.array(sigmacut)
        im.cut_sigma_image = jnp.where(im.cut_sigma_image < 0 , 1, im.cut_sigma_image)
        im.cut_sigma_image = jnp.where(jnp.isinf(1./im.cut_sigma_image) , 1, im.cut_sigma_image)
        im.cut_sigma_image = jnp.where(jnp.isnan(im.cut_sigma_image),1,im.cut_sigma_image)
        im.cut_sigma_image = jnp.where(jnp.isinf(im.cut_sigma_image),1,im.cut_sigma_image)
        im.cut_sigma_image = sigma_norm*im.cut_sigma_image

        Imagelist.append(im)
        

    for loopii,image_ke in enumerate(imagelabel):
        image_key = 'I' + image_ke
        image_config = {key: config_dict[image_key + str(i+1) + ')'] for i, key in enumerate(['image', 'band', 'sigma', 'psf', 'psf_sampling', 'mask', 'unit', 'fitting_area', 'conversion', 'magzp', 'skymodel', 'skyparameter', 'shift', 'shift_param', 'use_sed'])}
        if BandDuplication:
            Imagelist[loopii].imglabel = '{0}_{1}'.format(Imagelist[loopii].band,image_ke)
        else:
            Imagelist[loopii].imglabel = '{0}'.format(Imagelist[loopii].band)
        im = Imagelist[loopii]
        im.wcsshift = bool(image_config['shift'])
        im.magzp = image_config['magzp']
        csp = image_config['shift_param']
        par_total.add('csx_{0}'.format(im.imglabel),csp[0][0],min=csp[0][1],max=csp[0][2],brute_step=csp[0][3],vary=bool(csp[0][4]))
        par_total.add('csy_{0}'.format(im.imglabel),csp[1][0],min=csp[1][1],max=csp[1][2],brute_step=csp[1][3],vary=bool(csp[1][4]))
        skypar = image_config['skyparameter']
        if image_config['skymodel'] == 'uniform':
            #sky_mean, sky_median, sky_std = sigma_clipped_stats(im.cut_image, mask = cutmask, sigma=3.0, maxiters=5)
            if bool(skypar[0][4]):
                if not silent:print("Using uniform sky model for {0} band with sigma clipped median value {1:.2e} and std {2:.2e}".format(im.band, sky_median, sky_std))
                par_total.add('sky_{0}'.format(im.imglabel),sky_median,min=skypar[0][1],max=skypar[0][2],brute_step=sky_std,vary=bool(skypar[0][4]))
            else:
                if not silent:print("Using uniform sky model for {0} band with fixed value {1}".format(im.band, skypar[0][0]))
                par_total.add('sky_{0}'.format(im.imglabel),skypar[0][0],min=skypar[0][1],max=skypar[0][2],brute_step=skypar[0][3],vary=bool(skypar[0][4]))
        Imagelist.append(im)
        #print ("cut image of {0} is : {1}".format(im.band, im.cut_image))

    sedmode = False
    if len(filter_wave) > 0:
        combined_wave = jnp.concatenate(filter_wave)
        sortwave    = np.sort(np.asarray(combined_wave))
          ## # im.resp 还没改呢 (N x 300 for jwst nircam filter.) ## inreasonable, too sparse to sample well.
    else:
        sortwave = None
    for im in Imagelist:
        if im.photopoint:
            sedmode = True
        else:    
            sedmode = False
            break #???



    # read spectra
    Spectra = []
    hires = []
    snorm = []
    lsf = []
    swrange = None
    wave_temp = None
    feUV = True
    for spec_ke in speclabel:
        spec_key = 'S' + spec_ke
        holder = config_dict[spec_key + '1)']
        if type(holder) == str:
            sfname = holder
            enorm = 1.
        else:            
            sfname = holder[0]
            enorm = holder[1]
        spec = np.genfromtxt(sfname)
        spec = spec.transpose()
        fact = config_dict[spec_key + '2)']*1e-17/C_unit
        spec[1] = spec[1]*fact
        spec[2] = spec[2]*fact*enorm
        windows = config_dict[spec_key + '3)']
        if len(windows) > 0:
            if windows[0][0] > 3550:
                feUV = False
        mask = np.zeros_like(spec[0],dtype=bool)
        for window in windows:
            mask1 = (spec[0]/(1+z) > window[0]) & (spec[0]/(1+z) < window[1])
            mask = mask | mask1
        spec = jnp.array([spec[0][mask],spec[1][mask],spec[2][mask]])
        swrange = [jnp.min(spec[0]/(1+z))*(1.-1000./3e5),jnp.max(spec[0]/(1+z))*(1.+1000./3e5)]
        hires.append(config_dict[spec_key + '4)'])
        Spectra.append(spec)
        snorm.append(fact)
        # lsf, optional in 5)
        lsf_key = spec_key + '5)'
        if lsf_key in config_dict:
            lsf_spec = np.genfromtxt(config_dict[lsf_key])
            lsf.append(jnp.array(lsf_spec))
        else:
            lsf.append(None)
    hires_array = np.array(hires)

    if swrange is not None:
        wave_temp = np.array(sedI.sed_data_hires['wave'])
        masker = (wave_temp > swrange[0]) & (wave_temp < swrange[1])
        #print (swrange, masker)
        wave_temp = wave_temp[masker]
        wave_temp = jnp.array(wave_temp,dtype = jnp.float32)

    # define image_atlas
    Atlas = []
    Atlasnames = []
    grismfit = False
    
    for loopa,atlas_ke in enumerate(imatalslabel):
        atlas_key = 'A' + atlas_ke
        Atlasnames.append(config_dict[atlas_key + '1)'])
        imgs =  config_dict[atlas_key + '2)']
        imgindex = [imagelabel.index(img) for img in imgs]
        samegrid = config_dict[atlas_key + '3)']
        linkshift = config_dict[atlas_key + '4)']
        pimagelist= []
        pbandlist = []
        gimagelist = []
        gbandlist = []
        for loopii,i in enumerate(imgindex):
            if Imagelist[i].disperser is None:
                pimagelist.append(Imagelist[i])
                pbandlist.append(Imagelist[i].band) 
            else:
                gimagelist.append(Imagelist[i])
                gbandlist.append(Imagelist[i].band)
                grismfit = True
        atl = IM.image_atlas(pimagelist,pbandlist,grism_image_list = gimagelist,grism_band_list = gbandlist,same_grid=bool(samegrid),name=Atlasnames[-1])
        if (linkshift == 1)&(len(atl) > 1):  # for multi atlases, link their frame shift together. But how atlas shift affect im shift? only for samegrid?
            for loop in range(len(atl)-1):
                par_total.add('csx_{0}'.format(atl[loop+1].imglabel),expr='1*csx_{0}'.format(atl[0].imglabel),vary=False)
                par_total.add('csy_{0}'.format(atl[loop+1].imglabel),expr='1*csy_{0}'.format(atl[0].imglabel),vary=False)
        speclist = config_dict[atlas_key + '5)']
        aperturelist = config_dict[atlas_key + '6)']
        refimglist = config_dict[atlas_key + '7)']
        for loop in range(len(speclist)):
            indice = speclabel.index(speclist[loop])
            sprefix = 'S'+speclist[loop]
            if len(aperturelist[loop]) == 1:
                atl.add_spectrum(Spectra[indice],referimag=int(refimglist[loop]),stype = 'fiber',sapersize=float(aperturelist[loop][0]),hires=hires[indice],prefix=sprefix,norm=snorm[indice],lsf=lsf[indice])
            elif len(aperturelist[loop]) == 3:
                atl.add_spectrum(Spectra[indice],referimag=int(refimglist[loop]),stype = 'longslit', PA = float(aperturelist[loop][0]), a=float(aperturelist[loop][1]) , b=float(aperturelist[loop][2]),hires=hires[indice],prefix=sprefix,norm=snorm[indice],lsf=lsf[indice])
            elif len(aperturelist[loop]) == 0:
                atl.add_spectrum(Spectra[indice],hires=hires[indice],prefix=sprefix,norm=snorm[indice],lsf=lsf[indice])
            for nuke in nucleilabel:
                nuke_key = 'N' + nuke
                par_total.add('Ns_{0}_{1}'.format(config_dict[nuke_key + '1)'],sprefix),config_dict[nuke_key + '19)'][0],min=config_dict[nuke_key + '19)'][1],max=config_dict[nuke_key + '19)'][2],brute_step=config_dict[nuke_key + '19)'][3],vary=bool(config_dict[nuke_key + '19)'][4]))
            for gake in galaxylabel:
                gake_key = 'G' + gake
                par_total.add('Ns_{0}_{1}'.format(config_dict[gake_key + '1)'],sprefix),config_dict[gake_key + '5)'][0],min=config_dict[gake_key + '5)'][1],max=config_dict[gake_key + '5)'][2],brute_step=config_dict[gake_key + '5)'][3],vary=bool(config_dict[gake_key + '5)'][4]))
        for nuke in nucleilabel:
                nuke_key = 'N' + nuke
                if atlas_key + '9)' in config_dict:
                    atlasnorm=config_dict[atlas_key + '9)']
                    print("fixing Ni_{0}_{1} to requested value {2}".format(config_dict[nuke_key + '1)'],Atlasnames[-1],atlasnorm))
                else:
                    atlasnorm=config_dict[nuke_key + '26)'][0]
                if atlas_key + '8)' in config_dict:
                    varya = bool(config_dict[atlas_key + '8)'])
                else: 
                    varya = True
                varyn = bool(config_dict[nuke_key + '26)'][4])
                par_total.add('Ni_{0}_{1}'.format(config_dict[nuke_key + '1)'],Atlasnames[-1]),atlasnorm,min=config_dict[nuke_key + '26)'][1],max=config_dict[nuke_key + '26)'][2],brute_step=config_dict[nuke_key + '26)'][3],vary=varyn&varya)
        Atlas.append(atl)

          

    GS_data = IM.galfitS_data(Atlas,Atlasnames)
    GS_data.sortwave = sortwave ## TODO: delete
    
    if (len(imagelabel) == 0) & (len(speclabel) > 0):
        if np.sum(hires_array) == 0:
            GS_data.fitMode = 'spectrum - SED'
        else:
            GS_data.fitMode = 'high resolution spectrum - SED'
    elif (len(imagelabel) > 0) & (len(speclabel) == 0):
        allnotSED = True
        for im in Imagelist:
            if im.fitSED:
                allnotSED = False
        if allnotSED:
            GS_data.fitMode = 'images - photometry'
            for im in Imagelist:
                im.phys_to_counts_rate = 1.
        else:
            GS_data.fitMode = 'images - SED'
    elif (len(imagelabel) > 0) & (len(speclabel) > 0):
        GS_data.fitMode = 'images + spectra - SED'
    else:
        raise ValueError('No data to fit')
    if sedmode:
        GS_data.fitMode = 'photometric - SED'
        if len(speclabel) > 0:
            GS_data.fitMode = 'photometric + spectrum - SED'
        for atl in Atlas:
            atl.init_sed()
        GS_data.init_sed() ## here set self.resps and self.wavegrid in resampled grid.
    if grismfit:
        GS_data.fitMode = 'grism - SED'
    
    if dataonly:
        return GS_data, par_total

    # loading profiles
    Profiles    = []
    Profilesnames   = []
    Sparams_list    = []
    SFHlist     = []
    SEDmodelist = []
    TableModellist  = []
    has_eazy_profile = False
    agetype     = []
    agebin      = []
    sfrtype     = []
    ztype   = []
    Znebtype    = []
    Avtype  = []
    for pke in profilelabel:
        profile_key = 'P' + pke
        Profilesnames.append(config_dict[profile_key + '1)'])
        # define stellar population
        # TODO:change the below code
        SFHlist.append(config_dict[profile_key + '15)'])
        sed_mode = config_dict[profile_key + '27)']
        SEDmodelist.append(sed_mode)
        if sed_mode == 'Z':
            has_eazy_profile = True
            for i_ez in range(18):
                par_total.add(
                    'ezlogn{0}_{1}'.format(i_ez + 1, Profilesnames[-1]),
                    -1.0,
                    min=-3.0,
                    max=2.0,
                    brute_step=0.1,
                    vary=True,
                )
        if sed_mode == 'T':
            tablepath = config_dict[profile_key + 'T1)']
            param_names = config_dict[profile_key + 'T2)']
            paran_init = config_dict[profile_key + 'T3)']
            for par in param_names:
                par_total.add('{0}_{1}'.format(Profilesnames[-1],par),paran_init[par][0],min=paran_init[par][1],max=paran_init[par][2],brute_step=paran_init[par][3],vary=bool(paran_init[par][4]))
            model = sedI.read_table_model(tablepath,param_names,Profilesnames[-1])
            TableModellist.append(model)
        else:
            TableModellist.append(None)
        if config_dict[profile_key + '15)'] == 'bins':
            for loopage in range(len(config_dict[profile_key + '10)'])-1):
                par_total.add('{0}_f_cont_bin{1}'.format(Profilesnames[-1],loopage+1),config_dict[profile_key + '9)'][loopage][0],min=config_dict[profile_key + '9)'][loopage][1],max=config_dict[profile_key + '9)'][loopage][2],brute_step=config_dict[profile_key + '9)'][loopage][3],vary=bool(config_dict[profile_key + '9)'][loopage][4]))
            sfrtype.append('bins')
            agetype.append('bins')
            agebin.append(config_dict[profile_key + '10)'])
        else:
            agebin.append([])
            if len(config_dict[profile_key + '9)']) == 1: ## TODO:We can make it more elegent.
                sfrtype.append('const')
                par_total.add('{0}_f_cont_value'.format(Profilesnames[-1]),config_dict[profile_key + '9)'][0][0],min=config_dict[profile_key + '9)'][0][1],max=config_dict[profile_key + '9)'][0][2],brute_step=config_dict[profile_key + '9)'][0][3],vary=bool(config_dict[profile_key + '9)'][0][4]))
            elif len(config_dict[profile_key + '9)']) == 2:
                sfrtype.append('linear')
                par_total.add('{0}_f_cont_in'.format(Profilesnames[-1]),config_dict[profile_key + '9)'][0][0],min=config_dict[profile_key + '9)'][0][1],max=config_dict[profile_key + '9)'][0][2],brute_step=config_dict[profile_key + '9)'][0][3],vary=bool(config_dict[profile_key + '9)'][0][4]))
                par_total.add('{0}_f_cont_out'.format(Profilesnames[-1]),config_dict[profile_key + '9)'][1][0],min=config_dict[profile_key + '9)'][1][1],max=config_dict[profile_key + '9)'][1][2],brute_step=config_dict[profile_key + '9)'][1][3],vary=bool(config_dict[profile_key + '9)'][1][4]))
            if len(config_dict[profile_key + '10)']) == 1:
                agetype.append('const')
                par_total.add('{0}_age_value'.format(Profilesnames[-1]),config_dict[profile_key + '10)'][0][0],min=config_dict[profile_key + '10)'][0][1],max=config_dict[profile_key + '10)'][0][2],brute_step=config_dict[profile_key + '10)'][0][3],vary=bool(config_dict[profile_key + '10)'][0][4]))
            elif len(config_dict[profile_key + '10)']) == 2:
                agetype.append('linear')
                par_total.add('{0}_age_in'.format(Profilesnames[-1]),config_dict[profile_key + '10)'][0][0],min=config_dict[profile_key + '10)'][0][1],max=config_dict[profile_key + '10)'][0][2],brute_step=config_dict[profile_key + '10)'][0][3],vary=bool(config_dict[profile_key + '10)'][0][4]))
                par_total.add('{0}_age_out'.format(Profilesnames[-1]),config_dict[profile_key + '10)'][1][0],min=config_dict[profile_key + '10)'][1][1],max=config_dict[profile_key + '10)'][1][2],brute_step=config_dict[profile_key + '10)'][1][3],vary=bool(config_dict[profile_key + '10)'][1][4]))
        ## for stellar metallicity
        if len(config_dict[profile_key + '11)']) == 1:
            ztype.append('const')
            par_total.add('{0}_Z_value'.format(Profilesnames[-1]),config_dict[profile_key + '11)'][0][0],min=config_dict[profile_key + '11)'][0][1],max=config_dict[profile_key + '11)'][0][2],brute_step=config_dict[profile_key + '11)'][0][3],vary=bool(config_dict[profile_key + '11)'][0][4]))
        elif len(config_dict[profile_key + '11)']) == 2:
            ztype.append('linear')
            par_total.add('{0}_Z_in'.format(Profilesnames[-1]),config_dict[profile_key + '11)'][0][0],min=config_dict[profile_key + '11)'][0][1],max=config_dict[profile_key + '11)'][0][2],brute_step=config_dict[profile_key + '11)'][0][3],vary=bool(config_dict[profile_key + '11)'][0][4]))
            par_total.add('{0}_Z_out'.format(Profilesnames[-1]),config_dict[profile_key + '11)'][1][0],min=config_dict[profile_key + '11)'][1][1],max=config_dict[profile_key + '11)'][1][2],brute_step=config_dict[profile_key + '11)'][1][3],vary=bool(config_dict[profile_key + '11)'][1][4]))
        ## for nebular metallicity 
            ## TODO: reorganize the parameters sequence. temporarily use 33 as neb metallicity.
        if len(config_dict[profile_key + "33)"]) == 1:
            Znebtype.append("const")
            par_total.add('{0}_Zneb_value'.format(Profilesnames[-1]), config_dict[profile_key + '33)'][0][0], min=config_dict[profile_key + '33)'][0][1], max=config_dict[profile_key + '33)'][0][2], brute_step=config_dict[profile_key + '33)'][0][3], vary=bool(config_dict[profile_key + '33)'][0][4]))
        elif len(config_dict[profile_key + "33)"]) == 2:
            Znebtype.append("linear")
            par_total.add('{0}_Zneb_in'.format(Profilesnames[-1]), config_dict[profile_key + '33)'][0][0], min=config_dict[profile_key + '33)'][0][1], max=config_dict[profile_key + '33)'][0][2], brute_step=config_dict[profile_key + '33)'][0][3], vary=bool(config_dict[profile_key + '33)'][0][4]))
            par_total.add('{0}_Zneb_out'.format(Profilesnames[-1]), config_dict[profile_key + '33)'][1][0], min=config_dict[profile_key + '33)'][1][1], max=config_dict[profile_key + '33)'][1][2], brute_step=config_dict[profile_key + '33)'][1][3], vary=bool(config_dict[profile_key + '33)'][1][4]))
        ## for dust extinction 

        par_total.add('{0}_dust2'.format(Profilesnames[-1]), config_dict[profile_key + '34)'][0], min=config_dict[profile_key + '34)'][1], max=config_dict[profile_key + '34)'][2], brute_step=config_dict[profile_key + '34)'][3], vary=bool(config_dict[profile_key + '34)'][4]))
        par_total.add('{0}_dust_ratio'.format(Profilesnames[-1]), config_dict[profile_key + '35)'][0], min=config_dict[profile_key + '35)'][1],max=config_dict[profile_key + '35)'][2], brute_step=config_dict[profile_key + '35)'][3], vary=bool(config_dict[profile_key + '35)'][4]))
        par_total.add('{0}_dust_index'.format(Profilesnames[-1]), config_dict[profile_key + '36)'][0], min=config_dict[profile_key + '36)'][1], max=config_dict[profile_key + '36)'][2], brute_step=config_dict[profile_key + '36)'][3], vary=bool(config_dict[profile_key + '36)'][4]))
        ## for Av
        if len(config_dict[profile_key + '12)']) == 1:
            Avtype.append('const')
            par_total.add('{0}_Av_value'.format(Profilesnames[-1]),config_dict[profile_key + '12)'][0][0],min=config_dict[profile_key + '12)'][0][1],max=config_dict[profile_key + '12)'][0][2],brute_step=config_dict[profile_key + '12)'][0][3],vary=bool(config_dict[profile_key + '12)'][0][4]))
        elif len(config_dict[profile_key + '12)']) == 2:
            Avtype.append('linear')
            par_total.add('{0}_Av_in'.format(Profilesnames[-1]),config_dict[profile_key + '12)'][0][0],min=config_dict[profile_key + '12)'][0][1],max=config_dict[profile_key + '12)'][0][2],brute_step=config_dict[profile_key + '12)'][0][3],vary=bool(config_dict[profile_key + '12)'][0][4]))
            par_total.add('{0}_Av_out'.format(Profilesnames[-1]),config_dict[profile_key + '12)'][1][0],min=config_dict[profile_key + '12)'][1][1],max=config_dict[profile_key + '12)'][1][2],brute_step=config_dict[profile_key + '12)'][1][3],vary=bool(config_dict[profile_key + '12)'][1][4]))
        #if len(config_dict[profile_key + '13)']) == 1:
        #    sigtype.append('const')
        par_total.add('{0}_sigma'.format(Profilesnames[-1]),config_dict[profile_key + '13)'][0],min=config_dict[profile_key + '13)'][1],max=config_dict[profile_key + '13)'][2],brute_step=config_dict[profile_key + '13)'][3],vary=bool(config_dict[profile_key + '13)'][4]))
        #elif len(config_dict[profile_key + '13)']) == 2:
        #    sigtype.append('linear')
        #    par_total.add('{0}_sigma_out'.format(Profilesnames[-1]),config_dict[profile_key + '13)'][0][0],min=config_dict[profile_key + '13)'][0][1],max=config_dict[profile_key + '13)'][0][2],brute_step=config_dict[profile_key + '13)'][0][3],vary=bool(config_dict[profile_key + '13)'][0][4]))
        #    par_total.add('{0}_sigma_in'.format(Profilesnames[-1]),config_dict[profile_key + '13)'][1][0],min=config_dict[profile_key + '13)'][1][1],max=config_dict[profile_key + '13)'][1][2],brute_step=config_dict[profile_key + '13)'][1][3],vary=bool(config_dict[profile_key + '13)'][1][4]))
        if sed_mode == 'Z': ## Why eazy fixs logM?
            par_total.add('logM_{0}'.format(Profilesnames[-1]),config_dict[profile_key + '14)'][0],min=config_dict[profile_key + '14)'][1],max=config_dict[profile_key + '14)'][2],brute_step=config_dict[profile_key + '14)'][3],vary=False)
        else:
            par_total.add('logM_{0}'.format(Profilesnames[-1]),config_dict[profile_key + '14)'][0],min=config_dict[profile_key + '14)'][1],max=config_dict[profile_key + '14)'][2],brute_step=config_dict[profile_key + '14)'][3],vary=bool(config_dict[profile_key + '14)'][4]))
        mass0 = config_dict[profile_key + '14)'][0]
        par_total.add('logU_{0}'.format(Profilesnames[-1]),config_dict[profile_key + '16)'][0],min=config_dict[profile_key + '16)'][1],max=config_dict[profile_key + '16)'][2],brute_step=config_dict[profile_key + '16)'][3],vary=bool(config_dict[profile_key + '16)'][4]))
        par_total.add('AVbump_{0}'.format(Profilesnames[-1]),config_dict[profile_key + '26)'][0],min=config_dict[profile_key + '26)'][1],max=config_dict[profile_key + '26)'][2],brute_step=config_dict[profile_key + '26)'][3],vary=bool(config_dict[profile_key + '26)'][4]))
        try:
            use_dust_model = int(sed_mode) > 2 ## TODO: change to a more beautiful criteria....
        except (TypeError, ValueError):
            use_dust_model = False
        if use_dust_model:
            # include the cold dust
            par_total.add('logMdust_{0}'.format(Profilesnames[-1]),config_dict[profile_key + '28)'][0],min=config_dict[profile_key + '28)'][1],max=config_dict[profile_key + '28)'][2],brute_step=config_dict[profile_key + '28)'][3],vary=bool(config_dict[profile_key + '28)'][4]))
            par_total.add('Umin_{0}'.format(Profilesnames[-1]),config_dict[profile_key + '29)'][0],min=config_dict[profile_key + '29)'][1],max=config_dict[profile_key + '29)'][2],brute_step=config_dict[profile_key + '29)'][3],vary=bool(config_dict[profile_key + '29)'][4]))
            par_total.add('qPAH_{0}'.format(Profilesnames[-1]),config_dict[profile_key + '30)'][0],min=config_dict[profile_key + '30)'][1],max=config_dict[profile_key + '30)'][2],brute_step=config_dict[profile_key + '30)'][3],vary=bool(config_dict[profile_key + '30)'][4]))
            par_total.add('alphad_{0}'.format(Profilesnames[-1]),config_dict[profile_key + '31)'][0],min=config_dict[profile_key + '31)'][1],max=config_dict[profile_key + '31)'][2],brute_step=config_dict[profile_key + '31)'][3],vary=bool(config_dict[profile_key + '31)'][4]))
            par_total.add('gammad_{0}'.format(Profilesnames[-1]),config_dict[profile_key + '32)'][0],min=config_dict[profile_key + '32)'][1],max=config_dict[profile_key + '32)'][2],brute_step=config_dict[profile_key + '32)'][3],vary=bool(config_dict[profile_key + '32)'][4]))
        else: ## TODO: delete. NO NEED actually.
            # give a default value for the cold dust. 
            par_total.add('logMdust_{0}'.format(Profilesnames[-1]),8.14,min=4.5,max=10.,brute_step=0.1,vary=False)
            par_total.add('Umin_{0}'.format(Profilesnames[-1]),1.0,min=0.1,max=50.,brute_step=0.1,vary=False)
            par_total.add('qPAH_{0}'.format(Profilesnames[-1]),1.0,min=0.47,max=7.32,brute_step=0.1,vary=False)
            par_total.add('alphad_{0}'.format(Profilesnames[-1]),1.0,min=1.0,max=3.0,brute_step=0.1,vary=False)
            par_total.add('gammad_{0}'.format(Profilesnames[-1]),0.1,min=0,max=1.0,brute_step=0.1,vary=False)  
        prefix = Profilesnames[-1]
        if GS_data.fitMode == 'images - photometry':
            for loopim in range(len(GS_data.allbands)):
                im = GS_data.get_image(loopim)
                magzp = im.magzp 
                Ncomp = len(profilelabel) + len(nucleilabel)
                fnorm = jnp.log10(jnp.abs(imsum[im.band]))-mass0-jnp.log10(Ncomp)
                par_total.add('logNorm_{0}_{1}'.format(Profilesnames[-1],im.band), fnorm, min=fnorm-3., max=fnorm+2., brute_step=0.1, vary=True)
                par_total.add('Mag_{0}_{1}'.format(Profilesnames[-1],im.band), expr='{0}-2.5*logM_{1}-2.5*logNorm_{1}_{2}'.format(magzp,Profilesnames[-1],im.band), vary=False)
        if config_dict[profile_key + '2)'] == 'sersic':
            par_total.add('{0}_xcen'.format(Profilesnames[-1]),config_dict[profile_key + '3)'][0],min=config_dict[profile_key + '3)'][1],max=config_dict[profile_key + '3)'][2],brute_step=config_dict[profile_key + '3)'][3],vary=bool(config_dict[profile_key + '3)'][4]))
            par_total.add('{0}_ycen'.format(Profilesnames[-1]),config_dict[profile_key + '4)'][0],min=config_dict[profile_key + '4)'][1],max=config_dict[profile_key + '4)'][2],brute_step=config_dict[profile_key + '4)'][3],vary=bool(config_dict[profile_key + '4)'][4]))
            par_total.add('{0}_Re'.format(Profilesnames[-1]),config_dict[profile_key + '5)'][0],min=config_dict[profile_key + '5)'][1],max=config_dict[profile_key + '5)'][2],brute_step=config_dict[profile_key + '5)'][3],vary=bool(config_dict[profile_key + '5)'][4]))
            par_total.add('{0}_n'.format(Profilesnames[-1]),config_dict[profile_key + '6)'][0],min=config_dict[profile_key + '6)'][1],max=config_dict[profile_key + '6)'][2],brute_step=config_dict[profile_key + '6)'][3],vary=bool(config_dict[profile_key + '6)'][4]))
            par_total.add('{0}_ang'.format(Profilesnames[-1]),config_dict[profile_key + '7)'][0],min=config_dict[profile_key + '7)'][1],max=config_dict[profile_key + '7)'][2],brute_step=config_dict[profile_key + '7)'][3],vary=bool(config_dict[profile_key + '7)'][4]))
            par_total.add('{0}_axrat'.format(Profilesnames[-1]),config_dict[profile_key + '8)'][0],min=config_dict[profile_key + '8)'][1],max=config_dict[profile_key + '8)'][2],brute_step=config_dict[profile_key + '8)'][3],vary=bool(config_dict[profile_key + '8)'][4]))
            pardict = par_total.valuesdict()
            Sparams = {'type': 'sersic', 'paradic': {'xcen': pardict['{0}_xcen'.format(prefix)], 'ycen': pardict['{0}_ycen'.format(prefix)], 'Re': pardict['{0}_Re'.format(prefix)],
                    'n': pardict['{0}_n'.format(prefix)], 'ang':pardict['{0}_ang'.format(prefix)], 'axrat': pardict['{0}_axrat'.format(prefix)]}}
        elif config_dict[profile_key + '2)'] == 'sersic_b':
            par_total.add('{0}_xcen'.format(Profilesnames[-1]),config_dict[profile_key + '3)'][0],min=config_dict[profile_key + '3)'][1],max=config_dict[profile_key + '3)'][2],brute_step=config_dict[profile_key + '3)'][3],vary=bool(config_dict[profile_key + '3)'][4]))
            par_total.add('{0}_ycen'.format(Profilesnames[-1]),config_dict[profile_key + '4)'][0],min=config_dict[profile_key + '4)'][1],max=config_dict[profile_key + '4)'][2],brute_step=config_dict[profile_key + '4)'][3],vary=bool(config_dict[profile_key + '4)'][4]))
            par_total.add('{0}_Re'.format(Profilesnames[-1]),config_dict[profile_key + '5)'][0],min=config_dict[profile_key + '5)'][1],max=config_dict[profile_key + '5)'][2],brute_step=config_dict[profile_key + '5)'][3],vary=bool(config_dict[profile_key + '5)'][4]))
            par_total.add('{0}_n'.format(Profilesnames[-1]),config_dict[profile_key + '6)'][0],min=config_dict[profile_key + '6)'][1],max=config_dict[profile_key + '6)'][2],brute_step=config_dict[profile_key + '6)'][3],vary=bool(config_dict[profile_key + '6)'][4]))
            par_total.add('{0}_ang'.format(Profilesnames[-1]),config_dict[profile_key + '7)'][0],min=config_dict[profile_key + '7)'][1],max=config_dict[profile_key + '7)'][2],brute_step=config_dict[profile_key + '7)'][3],vary=bool(config_dict[profile_key + '7)'][4]))
            par_total.add('{0}_axrat'.format(Profilesnames[-1]),config_dict[profile_key + '8)'][0],min=config_dict[profile_key + '8)'][1],max=config_dict[profile_key + '8)'][2],brute_step=config_dict[profile_key + '8)'][3],vary=bool(config_dict[profile_key + '8)'][4]))
            par_total.add('{0}_r_out'.format(Profilesnames[-1]),config_dict[profile_key + '18)'][0],min=config_dict[profile_key + '18)'][1],max=config_dict[profile_key + '18)'][2],brute_step=config_dict[profile_key + '18)'][3],vary=bool(config_dict[profile_key + '18)'][4]))
            pardict = par_total.valuesdict()
            Sparams = {'type': 'sersic_b', 'paradic': {'xcen': pardict['{0}_xcen'.format(prefix)], 'ycen': pardict['{0}_ycen'.format(prefix)], 'Re': pardict['{0}_Re'.format(prefix)],
                    'n': pardict['{0}_n'.format(prefix)], 'ang':pardict['{0}_ang'.format(prefix)], 'axrat': pardict['{0}_axrat'.format(prefix)], 'r_out': pardict['{0}_r_out'.format(prefix)]}}
        elif config_dict[profile_key + '2)'] == 'sersic_r':
            par_total.add('{0}_xcen'.format(Profilesnames[-1]),config_dict[profile_key + '3)'][0],min=config_dict[profile_key + '3)'][1],max=config_dict[profile_key + '3)'][2],brute_step=config_dict[profile_key + '3)'][3],vary=bool(config_dict[profile_key + '3)'][4]))
            par_total.add('{0}_ycen'.format(Profilesnames[-1]),config_dict[profile_key + '4)'][0],min=config_dict[profile_key + '4)'][1],max=config_dict[profile_key + '4)'][2],brute_step=config_dict[profile_key + '4)'][3],vary=bool(config_dict[profile_key + '4)'][4]))
            par_total.add('{0}_Re'.format(Profilesnames[-1]),config_dict[profile_key + '5)'][0],min=config_dict[profile_key + '5)'][1],max=config_dict[profile_key + '5)'][2],brute_step=config_dict[profile_key + '5)'][3],vary=bool(config_dict[profile_key + '5)'][4]))
            par_total.add('{0}_n'.format(Profilesnames[-1]),config_dict[profile_key + '6)'][0],min=config_dict[profile_key + '6)'][1],max=config_dict[profile_key + '6)'][2],brute_step=config_dict[profile_key + '6)'][3],vary=bool(config_dict[profile_key + '6)'][4]))
            par_total.add('{0}_ang'.format(Profilesnames[-1]),config_dict[profile_key + '7)'][0],min=config_dict[profile_key + '7)'][1],max=config_dict[profile_key + '7)'][2],brute_step=config_dict[profile_key + '7)'][3],vary=bool(config_dict[profile_key + '7)'][4]))
            par_total.add('{0}_axrat'.format(Profilesnames[-1]),config_dict[profile_key + '8)'][0],min=config_dict[profile_key + '8)'][1],max=config_dict[profile_key + '8)'][2],brute_step=config_dict[profile_key + '8)'][3],vary=bool(config_dict[profile_key + '8)'][4]))
            par_total.add('{0}_r_in'.format(Profilesnames[-1]),config_dict[profile_key + '17)'][0],min=config_dict[profile_key + '17)'][1],max=config_dict[profile_key + '17)'][2],brute_step=config_dict[profile_key + '17)'][3],vary=bool(config_dict[profile_key + '17)'][4]))
            par_total.add('{0}_width'.format(Profilesnames[-1]),config_dict[profile_key + '19)'][0],min=config_dict[profile_key + '19)'][1],max=config_dict[profile_key + '19)'][2],brute_step=config_dict[profile_key + '19)'][3],vary=bool(config_dict[profile_key + '19)'][4]))
            pardict = par_total.valuesdict()
            Sparams = {'type': 'sersic_r', 'paradic': {'xcen': pardict['{0}_xcen'.format(prefix)], 'ycen': pardict['{0}_ycen'.format(prefix)], 'Re': pardict['{0}_Re'.format(prefix)],
                    'n': pardict['{0}_n'.format(prefix)], 'ang':pardict['{0}_ang'.format(prefix)], 'axrat': pardict['{0}_axrat'.format(prefix)], 'r_in': pardict['{0}_r_in'.format(prefix)], 'width': pardict['{0}_width'.format(prefix)]}}            
        elif (config_dict[profile_key + '2)'] == 'sersic_f')|(config_dict[profile_key + '2)'] == 'sersic_rf')|(config_dict[profile_key + '2)'] == 'sersic_bf'):
            par_total.add('{0}_xcen'.format(Profilesnames[-1]),config_dict[profile_key + '3)'][0],min=config_dict[profile_key + '3)'][1],max=config_dict[profile_key + '3)'][2],brute_step=config_dict[profile_key + '3)'][3],vary=bool(config_dict[profile_key + '3)'][4]))
            par_total.add('{0}_ycen'.format(Profilesnames[-1]),config_dict[profile_key + '4)'][0],min=config_dict[profile_key + '4)'][1],max=config_dict[profile_key + '4)'][2],brute_step=config_dict[profile_key + '4)'][3],vary=bool(config_dict[profile_key + '4)'][4]))
            par_total.add('{0}_Re'.format(Profilesnames[-1]),config_dict[profile_key + '5)'][0],min=config_dict[profile_key + '5)'][1],max=config_dict[profile_key + '5)'][2],brute_step=config_dict[profile_key + '5)'][3],vary=bool(config_dict[profile_key + '5)'][4]))
            par_total.add('{0}_n'.format(Profilesnames[-1]),config_dict[profile_key + '6)'][0],min=config_dict[profile_key + '6)'][1],max=config_dict[profile_key + '6)'][2],brute_step=config_dict[profile_key + '6)'][3],vary=bool(config_dict[profile_key + '6)'][4]))
            par_total.add('{0}_ang'.format(Profilesnames[-1]),config_dict[profile_key + '7)'][0],min=config_dict[profile_key + '7)'][1],max=config_dict[profile_key + '7)'][2],brute_step=config_dict[profile_key + '7)'][3],vary=bool(config_dict[profile_key + '7)'][4]))
            par_total.add('{0}_axrat'.format(Profilesnames[-1]),config_dict[profile_key + '8)'][0],min=config_dict[profile_key + '8)'][1],max=config_dict[profile_key + '8)'][2],brute_step=config_dict[profile_key + '8)'][3],vary=bool(config_dict[profile_key + '8)'][4]))
            par_total.add('{0}_r_in'.format(Profilesnames[-1]),config_dict[profile_key + '17)'][0],min=config_dict[profile_key + '17)'][1],max=config_dict[profile_key + '17)'][2],brute_step=config_dict[profile_key + '17)'][3],vary=bool(config_dict[profile_key + '17)'][4]))
            par_total.add('{0}_r_out'.format(Profilesnames[-1]),config_dict[profile_key + '18)'][0],min=config_dict[profile_key + '18)'][1],max=config_dict[profile_key + '18)'][2],brute_step=config_dict[profile_key + '18)'][3],vary=bool(config_dict[profile_key + '18)'][4]))
            par_total.add('{0}_alpha_rc'.format(Profilesnames[-1]),config_dict[profile_key + '19)'][0],min=config_dict[profile_key + '19)'][1],max=config_dict[profile_key + '19)'][2],brute_step=config_dict[profile_key + '19)'][3],vary=bool(config_dict[profile_key + '19)'][4]))
            par_total.add('{0}_theta_out'.format(Profilesnames[-1]),config_dict[profile_key + '20)'][0],min=config_dict[profile_key + '20)'][1],max=config_dict[profile_key + '20)'][2],brute_step=config_dict[profile_key + '20)'][3],vary=bool(config_dict[profile_key + '20)'][4]))
            par_total.add('{0}_m'.format(Profilesnames[-1]),config_dict[profile_key + '21)'],min=0,max=10,brute_step=0.1,vary=False)
            par_total.add('{0}_am'.format(Profilesnames[-1]),config_dict[profile_key + '22)'][0],min=config_dict[profile_key + '22)'][1],max=config_dict[profile_key + '22)'][2],brute_step=config_dict[profile_key + '22)'][3],vary=bool(config_dict[profile_key + '22)'][4]))
            par_total.add('{0}_theta_m'.format(Profilesnames[-1]),config_dict[profile_key + '23)'][0],min=config_dict[profile_key + '23)'][1],max=config_dict[profile_key + '23)'][2],brute_step=config_dict[profile_key + '23)'][3],vary=bool(config_dict[profile_key + '23)'][4]))
            par_total.add('{0}_i_arm'.format(Profilesnames[-1]),config_dict[profile_key + '24)'][0],min=config_dict[profile_key + '24)'][1],max=config_dict[profile_key + '24)'][2],brute_step=config_dict[profile_key + '24)'][3],vary=bool(config_dict[profile_key + '24)'][4]))
            pardict = par_total.valuesdict()
            Sparams = {'type': config_dict[profile_key + '2)'], 'paradic': {'xcen': pardict['{0}_xcen'.format(prefix)], 'ycen': pardict['{0}_ycen'.format(prefix)], 'Re': pardict['{0}_Re'.format(prefix)],
                    'n': pardict['{0}_n'.format(prefix)], 'ang':pardict['{0}_ang'.format(prefix)], 'axrat': pardict['{0}_axrat'.format(prefix)], 'r_in': pardict['{0}_r_in'.format(prefix)],
                    'r_out': pardict['{0}_r_out'.format(prefix)], 'alpha_rc': pardict['{0}_alpha_rc'.format(prefix)], 'theta_out': pardict['{0}_theta_out'.format(prefix)], 'm': int(config_dict[profile_key + '21)']),
                    'am': pardict['{0}_am'.format(prefix)], 'theta_m': pardict['{0}_theta_m'.format(prefix)], 'i_arm': pardict['{0}_i_arm'.format(prefix)]}}
        elif config_dict[profile_key + '2)'] == 'ferrer':
            par_total.add('{0}_xcen'.format(Profilesnames[-1]),config_dict[profile_key + '3)'][0],min=config_dict[profile_key + '3)'][1],max=config_dict[profile_key + '3)'][2],brute_step=config_dict[profile_key + '3)'][3],vary=bool(config_dict[profile_key + '3)'][4]))
            par_total.add('{0}_ycen'.format(Profilesnames[-1]),config_dict[profile_key + '4)'][0],min=config_dict[profile_key + '4)'][1],max=config_dict[profile_key + '4)'][2],brute_step=config_dict[profile_key + '4)'][3],vary=bool(config_dict[profile_key + '4)'][4]))
            par_total.add('{0}_Rout'.format(Profilesnames[-1]),config_dict[profile_key + '5)'][0],min=config_dict[profile_key + '5)'][1],max=config_dict[profile_key + '5)'][2],brute_step=config_dict[profile_key + '5)'][3],vary=bool(config_dict[profile_key + '5)'][4]))
            par_total.add('{0}_alpha'.format(Profilesnames[-1]),config_dict[profile_key + '6)'][0],min=config_dict[profile_key + '6)'][1],max=config_dict[profile_key + '6)'][2],brute_step=config_dict[profile_key + '6)'][3],vary=bool(config_dict[profile_key + '6)'][4]))
            par_total.add('{0}_ang'.format(Profilesnames[-1]),config_dict[profile_key + '7)'][0],min=config_dict[profile_key + '7)'][1],max=config_dict[profile_key + '7)'][2],brute_step=config_dict[profile_key + '7)'][3],vary=bool(config_dict[profile_key + '7)'][4]))
            par_total.add('{0}_axrat'.format(Profilesnames[-1]),config_dict[profile_key + '8)'][0],min=config_dict[profile_key + '8)'][1],max=config_dict[profile_key + '8)'][2],brute_step=config_dict[profile_key + '8)'][3],vary=bool(config_dict[profile_key + '8)'][4]))
            par_total.add('{0}_beta'.format(Profilesnames[-1]),config_dict[profile_key + '25)'][0],min=config_dict[profile_key + '25)'][1],max=config_dict[profile_key + '25)'][2],brute_step=config_dict[profile_key + '25)'][3],vary=bool(config_dict[profile_key + '25)'][4]))
            pardict = par_total.valuesdict()
            Sparams = {'type': 'ferrer', 'paradic': {'xcen': pardict['{0}_xcen'.format(prefix)], 'ycen': pardict['{0}_ycen'.format(prefix)], 'Rout': pardict['{0}_Rout'.format(prefix)],
                    'alpha': pardict['{0}_alpha'.format(prefix)], 'ang':pardict['{0}_ang'.format(prefix)], 'axrat': pardict['{0}_axrat'.format(prefix)], 'beta': pardict['{0}_beta'.format(prefix)]}}                                                            
        elif config_dict[profile_key + '2)'] == 'edgeondisk':
            par_total.add('{0}_xcen'.format(Profilesnames[-1]),config_dict[profile_key + '3)'][0],min=config_dict[profile_key + '3)'][1],max=config_dict[profile_key + '3)'][2],brute_step=config_dict[profile_key + '3)'][3],vary=bool(config_dict[profile_key + '3)'][4]))
            par_total.add('{0}_ycen'.format(Profilesnames[-1]),config_dict[profile_key + '4)'][0],min=config_dict[profile_key + '4)'][1],max=config_dict[profile_key + '4)'][2],brute_step=config_dict[profile_key + '4)'][3],vary=bool(config_dict[profile_key + '4)'][4]))
            par_total.add('{0}_rs'.format(Profilesnames[-1]),config_dict[profile_key + '5)'][0],min=config_dict[profile_key + '5)'][1],max=config_dict[profile_key + '5)'][2],brute_step=config_dict[profile_key + '5)'][3],vary=bool(config_dict[profile_key + '5)'][4]))
            par_total.add('{0}_hs'.format(Profilesnames[-1]),config_dict[profile_key + '6)'][0],min=config_dict[profile_key + '6)'][1],max=config_dict[profile_key + '6)'][2],brute_step=config_dict[profile_key + '6)'][3],vary=bool(config_dict[profile_key + '6)'][4]))
            par_total.add('{0}_ang'.format(Profilesnames[-1]),config_dict[profile_key + '7)'][0],min=config_dict[profile_key + '7)'][1],max=config_dict[profile_key + '7)'][2],brute_step=config_dict[profile_key + '7)'][3],vary=bool(config_dict[profile_key + '7)'][4]))
            pardict = par_total.valuesdict()
            Sparams = {'type': 'edgeondisk', 'paradic': {'xcen': pardict['{0}_xcen'.format(prefix)], 'ycen': pardict['{0}_ycen'.format(prefix)], 'rs': pardict['{0}_rs'.format(prefix)],
                    'hs': pardict['{0}_hs'.format(prefix)], 'ang':pardict['{0}_ang'.format(prefix)]}}
        elif config_dict[profile_key + '2)'] == 'GauRing':
            par_total.add('{0}_xcen'.format(Profilesnames[-1]),config_dict[profile_key + '3)'][0],min=config_dict[profile_key + '3)'][1],max=config_dict[profile_key + '3)'][2],brute_step=config_dict[profile_key + '3)'][3],vary=bool(config_dict[profile_key + '3)'][4]))
            par_total.add('{0}_ycen'.format(Profilesnames[-1]),config_dict[profile_key + '4)'][0],min=config_dict[profile_key + '4)'][1],max=config_dict[profile_key + '4)'][2],brute_step=config_dict[profile_key + '4)'][3],vary=bool(config_dict[profile_key + '4)'][4]))
            par_total.add('{0}_r0'.format(Profilesnames[-1]),config_dict[profile_key + '5)'][0],min=config_dict[profile_key + '5)'][1],max=config_dict[profile_key + '5)'][2],brute_step=config_dict[profile_key + '5)'][3],vary=bool(config_dict[profile_key + '5)'][4]))
            par_total.add('{0}_sig'.format(Profilesnames[-1]),config_dict[profile_key + '6)'][0],min=config_dict[profile_key + '6)'][1],max=config_dict[profile_key + '6)'][2],brute_step=config_dict[profile_key + '6)'][3],vary=bool(config_dict[profile_key + '6)'][4]))
            par_total.add('{0}_ang'.format(Profilesnames[-1]),config_dict[profile_key + '7)'][0],min=config_dict[profile_key + '7)'][1],max=config_dict[profile_key + '7)'][2],brute_step=config_dict[profile_key + '7)'][3],vary=bool(config_dict[profile_key + '7)'][4]))
            par_total.add('{0}_axrat'.format(Profilesnames[-1]),config_dict[profile_key + '8)'][0],min=config_dict[profile_key + '8)'][1],max=config_dict[profile_key + '8)'][2],brute_step=config_dict[profile_key + '8)'][3],vary=bool(config_dict[profile_key + '8)'][4]))
            pardict = par_total.valuesdict()
            Sparams = {'type': 'GauRing', 'paradic': {'xcen': pardict['{0}_xcen'.format(prefix)], 'ycen': pardict['{0}_ycen'.format(prefix)], 'r0': pardict['{0}_r0'.format(prefix)],
                    'sig': pardict['{0}_sig'.format(prefix)], 'ang':pardict['{0}_ang'.format(prefix)], 'axrat': pardict['{0}_axrat'.format(prefix)]}}
        elif config_dict[profile_key + '2)'] == 'ferrer_f':
            par_total.add('{0}_xcen'.format(Profilesnames[-1]),config_dict[profile_key + '3)'][0],min=config_dict[profile_key + '3)'][1],max=config_dict[profile_key + '3)'][2],brute_step=config_dict[profile_key + '3)'][3],vary=bool(config_dict[profile_key + '3)'][4]))
            par_total.add('{0}_ycen'.format(Profilesnames[-1]),config_dict[profile_key + '4)'][0],min=config_dict[profile_key + '4)'][1],max=config_dict[profile_key + '4)'][2],brute_step=config_dict[profile_key + '4)'][3],vary=bool(config_dict[profile_key + '4)'][4]))
            par_total.add('{0}_Rout'.format(Profilesnames[-1]),config_dict[profile_key + '5)'][0],min=config_dict[profile_key + '5)'][1],max=config_dict[profile_key + '5)'][2],brute_step=config_dict[profile_key + '5)'][3],vary=bool(config_dict[profile_key + '5)'][4]))
            par_total.add('{0}_alpha'.format(Profilesnames[-1]),config_dict[profile_key + '6)'][0],min=config_dict[profile_key + '6)'][1],max=config_dict[profile_key + '6)'][2],brute_step=config_dict[profile_key + '6)'][3],vary=bool(config_dict[profile_key + '6)'][4]))
            par_total.add('{0}_ang'.format(Profilesnames[-1]),config_dict[profile_key + '7)'][0],min=config_dict[profile_key + '7)'][1],max=config_dict[profile_key + '7)'][2],brute_step=config_dict[profile_key + '7)'][3],vary=bool(config_dict[profile_key + '7)'][4]))
            par_total.add('{0}_axrat'.format(Profilesnames[-1]),config_dict[profile_key + '8)'][0],min=config_dict[profile_key + '8)'][1],max=config_dict[profile_key + '8)'][2],brute_step=config_dict[profile_key + '8)'][3],vary=bool(config_dict[profile_key + '8)'][4]))
            par_total.add('{0}_r_in'.format(Profilesnames[-1]),config_dict[profile_key + '17)'][0],min=config_dict[profile_key + '17)'][1],max=config_dict[profile_key + '17)'][2],brute_step=config_dict[profile_key + '17)'][3],vary=bool(config_dict[profile_key + '17)'][4]))
            par_total.add('{0}_r_out'.format(Profilesnames[-1]),config_dict[profile_key + '18)'][0],min=config_dict[profile_key + '18)'][1],max=config_dict[profile_key + '18)'][2],brute_step=config_dict[profile_key + '18)'][3],vary=bool(config_dict[profile_key + '18)'][4]))
            par_total.add('{0}_alpha_rc'.format(Profilesnames[-1]),config_dict[profile_key + '19)'][0],min=config_dict[profile_key + '19)'][1],max=config_dict[profile_key + '19)'][2],brute_step=config_dict[profile_key + '19)'][3],vary=bool(config_dict[profile_key + '19)'][4]))
            par_total.add('{0}_theta_out'.format(Profilesnames[-1]),config_dict[profile_key + '20)'][0],min=config_dict[profile_key + '20)'][1],max=config_dict[profile_key + '20)'][2],brute_step=config_dict[profile_key + '20)'][3],vary=bool(config_dict[profile_key + '20)'][4]))
            par_total.add('{0}_m'.format(Profilesnames[-1]),config_dict[profile_key + '21)'],min=0,max=10,brute_step=0.1,vary=False)
            par_total.add('{0}_am'.format(Profilesnames[-1]),config_dict[profile_key + '22)'][0],min=config_dict[profile_key + '22)'][1],max=config_dict[profile_key + '22)'][2],brute_step=config_dict[profile_key + '22)'][3],vary=bool(config_dict[profile_key + '22)'][4]))
            par_total.add('{0}_theta_m'.format(Profilesnames[-1]),config_dict[profile_key + '23)'][0],min=config_dict[profile_key + '23)'][1],max=config_dict[profile_key + '23)'][2],brute_step=config_dict[profile_key + '23)'][3],vary=bool(config_dict[profile_key + '23)'][4]))
            par_total.add('{0}_i_arm'.format(Profilesnames[-1]),config_dict[profile_key + '24)'][0],min=config_dict[profile_key + '24)'][1],max=config_dict[profile_key + '24)'][2],brute_step=config_dict[profile_key + '24)'][3],vary=bool(config_dict[profile_key + '24)'][4]))
            par_total.add('{0}_beta'.format(Profilesnames[-1]),config_dict[profile_key + '25)'][0],min=config_dict[profile_key + '25)'][1],max=config_dict[profile_key + '25)'][2],brute_step=config_dict[profile_key + '25)'][3],vary=bool(config_dict[profile_key + '25)'][4]))
            pardict = par_total.valuesdict()
            Sparams = {'type': 'ferrer_f', 'paradic': {'xcen': pardict['{0}_xcen'.format(prefix)], 'ycen': pardict['{0}_ycen'.format(prefix)], 'Rout': pardict['{0}_Rout'.format(prefix)],
                    'alpha': pardict['{0}_alpha'.format(prefix)], 'ang':pardict['{0}_ang'.format(prefix)], 'axrat': pardict['{0}_axrat'.format(prefix)], 'r_in': pardict['{0}_r_in'.format(prefix)],
                    'r_out': pardict['{0}_r_out'.format(prefix)], 'alpha_rc': pardict['{0}_alpha_rc'.format(prefix)], 'theta_out': pardict['{0}_theta_out'.format(prefix)], 'm': int(config_dict[profile_key + '21)']),
                    'am': pardict['{0}_am'.format(prefix)], 'theta_m': pardict['{0}_theta_m'.format(prefix)], 'i_arm': pardict['{0}_i_arm'.format(prefix)], 'beta': pardict['{0}_beta'.format(prefix)]}}
        elif config_dict[profile_key + '2)'] == 'GauRing_f':
            par_total.add('{0}_xcen'.format(Profilesnames[-1]),config_dict[profile_key + '3)'][0],min=config_dict[profile_key + '3)'][1],max=config_dict[profile_key + '3)'][2],brute_step=config_dict[profile_key + '3)'][3],vary=bool(config_dict[profile_key + '3)'][4]))
            par_total.add('{0}_ycen'.format(Profilesnames[-1]),config_dict[profile_key + '4)'][0],min=config_dict[profile_key + '4)'][1],max=config_dict[profile_key + '4)'][2],brute_step=config_dict[profile_key + '4)'][3],vary=bool(config_dict[profile_key + '4)'][4]))
            par_total.add('{0}_r0'.format(Profilesnames[-1]),config_dict[profile_key + '5)'][0],min=config_dict[profile_key + '5)'][1],max=config_dict[profile_key + '5)'][2],brute_step=config_dict[profile_key + '5)'][3],vary=bool(config_dict[profile_key + '5)'][4]))
            par_total.add('{0}_sig'.format(Profilesnames[-1]),config_dict[profile_key + '6)'][0],min=config_dict[profile_key + '6)'][1],max=config_dict[profile_key + '6)'][2],brute_step=config_dict[profile_key + '6)'][3],vary=bool(config_dict[profile_key + '6)'][4]))
            par_total.add('{0}_ang'.format(Profilesnames[-1]),config_dict[profile_key + '7)'][0],min=config_dict[profile_key + '7)'][1],max=config_dict[profile_key + '7)'][2],brute_step=config_dict[profile_key + '7)'][3],vary=bool(config_dict[profile_key + '7)'][4]))
            par_total.add('{0}_axrat'.format(Profilesnames[-1]),config_dict[profile_key + '8)'][0],min=config_dict[profile_key + '8)'][1],max=config_dict[profile_key + '8)'][2],brute_step=config_dict[profile_key + '8)'][3],vary=bool(config_dict[profile_key + '8)'][4]))
            par_total.add('{0}_r_in'.format(Profilesnames[-1]),config_dict[profile_key + '17)'][0],min=config_dict[profile_key + '17)'][1],max=config_dict[profile_key + '17)'][2],brute_step=config_dict[profile_key + '17)'][3],vary=bool(config_dict[profile_key + '17)'][4]))
            par_total.add('{0}_r_out'.format(Profilesnames[-1]),config_dict[profile_key + '18)'][0],min=config_dict[profile_key + '18)'][1],max=config_dict[profile_key + '18)'][2],brute_step=config_dict[profile_key + '18)'][3],vary=bool(config_dict[profile_key + '18)'][4]))
            par_total.add('{0}_alpha_rc'.format(Profilesnames[-1]),config_dict[profile_key + '19)'][0],min=config_dict[profile_key + '19)'][1],max=config_dict[profile_key + '19)'][2],brute_step=config_dict[profile_key + '19)'][3],vary=bool(config_dict[profile_key + '19)'][4]))
            par_total.add('{0}_theta_out'.format(Profilesnames[-1]),config_dict[profile_key + '20)'][0],min=config_dict[profile_key + '20)'][1],max=config_dict[profile_key + '20)'][2],brute_step=config_dict[profile_key + '20)'][3],vary=bool(config_dict[profile_key + '20)'][4]))
            par_total.add('{0}_m'.format(Profilesnames[-1]),config_dict[profile_key + '21)'],min=0,max=10,brute_step=0.1,vary=False)
            par_total.add('{0}_am'.format(Profilesnames[-1]),config_dict[profile_key + '22)'][0],min=config_dict[profile_key + '22)'][1],max=config_dict[profile_key + '22)'][2],brute_step=config_dict[profile_key + '22)'][3],vary=bool(config_dict[profile_key + '22)'][4]))
            par_total.add('{0}_theta_m'.format(Profilesnames[-1]),config_dict[profile_key + '23)'][0],min=config_dict[profile_key + '23)'][1],max=config_dict[profile_key + '23)'][2],brute_step=config_dict[profile_key + '23)'][3],vary=bool(config_dict[profile_key + '23)'][4]))
            par_total.add('{0}_i_arm'.format(Profilesnames[-1]),config_dict[profile_key + '24)'][0],min=config_dict[profile_key + '24)'][1],max=config_dict[profile_key + '24)'][2],brute_step=config_dict[profile_key + '24)'][3],vary=bool(config_dict[profile_key + '24)'][4]))
            pardict = par_total.valuesdict()
            Sparams = {'type': 'GauRing_f', 'paradic': {'xcen': pardict['{0}_xcen'.format(prefix)], 'ycen': pardict['{0}_ycen'.format(prefix)], 'r0': pardict['{0}_r0'.format(prefix)],
                    'sig': pardict['{0}_sig'.format(prefix)], 'ang':pardict['{0}_ang'.format(prefix)], 'axrat': pardict['{0}_axrat'.format(prefix)], 'r_in': pardict['{0}_r_in'.format(prefix)],
                    'r_out': pardict['{0}_r_out'.format(prefix)], 'alpha_rc': pardict['{0}_alpha_rc'.format(prefix)], 'theta_out': pardict['{0}_theta_out'.format(prefix)], 'm': int(config_dict[profile_key + '21)']),
                    'am': pardict['{0}_am'.format(prefix)], 'theta_m': pardict['{0}_theta_m'.format(prefix)], 'i_arm': pardict['{0}_i_arm'.format(prefix)]}}
        elif config_dict[profile_key + '2)'] == 'const':
            Sparams = {'type': 'const', 'paradic': {}}
        else:
            raise ValueError('Profile type not recognized')                                                                                                                                                                                                                                  
        Sparams_list.append(Sparams)

    if has_eazy_profile and ((GS_data.fitMode != 'images - SED') or (not vectorize)):
        raise ValueError("SED mode 'Z' is only allowed when fitMode is 'images - SED' with vectorize=True")

    # load Nucleus
    Nucleus = []
    Nunames = []
    mtype_list = []
    for nuke in nucleilabel:
        mtype_list.append('agn')
        nuke_key = 'N' + nuke
        Nunames.append(config_dict[nuke_key + '1)'])
        par_total.add('xcen_{0}'.format(Nunames[-1]),config_dict[nuke_key + '4)'][0],min=config_dict[nuke_key + '4)'][1],max=config_dict[nuke_key + '4)'][2],brute_step=config_dict[nuke_key + '4)'][3],vary=bool(config_dict[nuke_key + '4)'][4]))
        par_total.add('ycen_{0}'.format(Nunames[-1]),config_dict[nuke_key + '5)'][0],min=config_dict[nuke_key + '5)'][1],max=config_dict[nuke_key + '5)'][2],brute_step=config_dict[nuke_key + '5)'][3],vary=bool(config_dict[nuke_key + '5)'][4]))
        par_total.add('agn_logM{0}'.format(Nunames[-1]),config_dict[nuke_key + '6)'][0],min=config_dict[nuke_key + '6)'][1],max=config_dict[nuke_key + '6)'][2],brute_step=config_dict[nuke_key + '6)'][3],vary=bool(config_dict[nuke_key + '6)'][4]))
        par_total.add('agn_logLedd{0}'.format(Nunames[-1]),config_dict[nuke_key + '7)'][0],min=config_dict[nuke_key + '7)'][1],max=config_dict[nuke_key + '7)'][2],brute_step=config_dict[nuke_key + '7)'][3],vary=bool(config_dict[nuke_key + '7)'][4]))
        if config_dict[nuke_key + '18)'] == 5:
            par_total.add('agn_amax{0}'.format(Nunames[-1]),config_dict[nuke_key + '8)'][0],min=config_dict[nuke_key + '8)'][1],max=config_dict[nuke_key + '8)'][2],brute_step=config_dict[nuke_key + '8)'][3],vary=bool(config_dict[nuke_key + '8)'][4]))
            par_total.add('agn_spin{0}'.format(Nunames[-1]),0.0,min=0.0,max=0.998,brute_step=0.01,vary=False)
        else:
            par_total.add('agn_spin{0}'.format(Nunames[-1]),config_dict[nuke_key + '8)'][0],min=config_dict[nuke_key + '8)'][1],max=config_dict[nuke_key + '8)'][2],brute_step=config_dict[nuke_key + '8)'][3],vary=bool(config_dict[nuke_key + '8)'][4]))
        par_total.add('agn_Av{0}'.format(Nunames[-1]),config_dict[nuke_key + '9)'][0],min=config_dict[nuke_key + '9)'][1],max=config_dict[nuke_key + '9)'][2],brute_step=config_dict[nuke_key + '9)'][3],vary=bool(config_dict[nuke_key + '9)'][4]))
        par_total.add('{0}logL5100'.format(Nunames[-1]),config_dict[nuke_key + '10)'][0],min=config_dict[nuke_key + '10)'][1],max=config_dict[nuke_key + '10)'][2],brute_step=config_dict[nuke_key + '10)'][3],vary=bool(config_dict[nuke_key + '10)'][4]))
        par_total.add('z_{0}'.format(Nunames[-1]),config_dict[nuke_key + '2)'][0],min=config_dict[nuke_key + '2)'][1],max=config_dict[nuke_key + '2)'][2],brute_step=config_dict[nuke_key + '2)'][3],vary=bool(config_dict[nuke_key + '2)'][4]))
        myagn = GALA.AGN(regionz=regionz,logM_BH=par_total['agn_logM{0}'.format(Nunames[-1])].value,logLedd= par_total['agn_logLedd{0}'.format(Nunames[-1])].value,
                        z=config_dict[nuke_key + '2)'][0],ebv_G=config_dict[nuke_key + '3)'],prefix=Nunames[-1], torus = bool(config_dict[nuke_key + '20)']))
        myagn.sortwave = sortwave ## TODO: delete
        if config_dict[nuke_key + '18)'] == 0:
            myagn.change_SED('SP')
            par_total.add('{0}plC'.format(Nunames[-1]),config_dict[nuke_key + '11)'][0][0],min=config_dict[nuke_key + '11)'][0][1],max=config_dict[nuke_key + '11)'][0][2],brute_step=config_dict[nuke_key + '11)'][0][3],vary=bool(config_dict[nuke_key + '11)'][0][4]))
        elif config_dict[nuke_key + '18)'] == 1:
            myagn.change_SED('BP')
            par_total.add('{0}plC1'.format(Nunames[-1]),config_dict[nuke_key + '11)'][0][0],min=config_dict[nuke_key + '11)'][0][1],max=config_dict[nuke_key + '11)'][0][2],brute_step=config_dict[nuke_key + '11)'][0][3],vary=bool(config_dict[nuke_key + '11)'][0][4]))
            par_total.add('{0}plC2'.format(Nunames[-1]),config_dict[nuke_key + '11)'][1][0],min=config_dict[nuke_key + '11)'][1][1],max=config_dict[nuke_key + '11)'][1][2],brute_step=config_dict[nuke_key + '11)'][1][3],vary=bool(config_dict[nuke_key + '11)'][1][4]))
        elif config_dict[nuke_key + '18)'] == 3:
            myagn.change_SED('T2')
        elif config_dict[nuke_key + '18)'] == 4:
            myagn.change_SED('NP')
            nimage = len(GS_data.allbands)
            bandlist = []
            wavels = []
            elvis94 = Table.read('{0}/templates/radioquiet_Elvis94.csv'.format(sedI.DATA_PATH),format='csv') ## TODO: we should write a script to load all needed templates.
            for i in range(nimage):
                im = GS_data.get_image(i)
                logLint = jnp.log10(jnp.interp(effective_wave[im.band],jnp.array(elvis94['lambda']),jnp.array(elvis94['L38lambda'])))+38
                par_total.add('{0}_{1}_logL'.format(Nunames[-1],im.band),logLint,min=logLint-5,max=logLint+5,brute_step=0.1,vary=True)
                bandlist.append(im.band)
                wavels.append(effective_wave[im.band])
            # sort the wavels and reorder the bandlist based on new wavels
            bandlist = [x for _,x in sorted(zip(wavels,bandlist))]
            wavels = sorted(wavels)
            myagn.bandslist = bandlist
            myagn.iwaves = np.array(wavels) 
        elif config_dict[nuke_key + '18)'] == 5:
            myagn.change_SED('SBD')
        elif config_dict[nuke_key + '18)'] == 'T':
            myagn.change_SED('table')
            tablepath = config_dict[nuke_key + 'T1)']
            param_names = config_dict[nuke_key + 'T2)']
            paran_init = config_dict[nuke_key + 'T3)']
            for loopp, par in enumerate(param_names):
                par_total.add('{0}{1}'.format(Nunames[-1],par),paran_init[par][0],min=paran_init[par][1],max=paran_init[par][2],brute_step=paran_init[par][3],vary=bool(paran_init[par][4]))
            model = sedI.read_table_model(tablepath,param_names,Nunames[-1])
            myagn.table_model = model
        par_total.add('{0}torus_logL'.format(Nunames[-1]),config_dict[nuke_key + '21)'][0],min=config_dict[nuke_key + '21)'][1],max=config_dict[nuke_key + '21)'][2],brute_step=config_dict[nuke_key + '21)'][3],vary=bool(config_dict[nuke_key + '21)'][4]))
        par_total.add('{0}torus_a'.format(Nunames[-1]),config_dict[nuke_key + '22)'][0],min=config_dict[nuke_key + '22)'][1],max=config_dict[nuke_key + '22)'][2],brute_step=config_dict[nuke_key + '22)'][3],vary=bool(config_dict[nuke_key + '22)'][4]))
        par_total.add('{0}torus_h'.format(Nunames[-1]),config_dict[nuke_key + '23)'][0],min=config_dict[nuke_key + '23)'][1],max=config_dict[nuke_key + '23)'][2],brute_step=config_dict[nuke_key + '23)'][3],vary=bool(config_dict[nuke_key + '23)'][4]))
        par_total.add('{0}torus_N0'.format(Nunames[-1]),config_dict[nuke_key + '24)'][0],min=config_dict[nuke_key + '24)'][1],max=config_dict[nuke_key + '24)'][2],brute_step=config_dict[nuke_key + '24)'][3],vary=bool(config_dict[nuke_key + '24)'][4]))
        par_total.add('{0}torus_i'.format(Nunames[-1]),config_dict[nuke_key + '25)'][0],min=config_dict[nuke_key + '25)'][1],max=config_dict[nuke_key + '25)'][2],brute_step=config_dict[nuke_key + '25)'][3],vary=bool(config_dict[nuke_key + '25)'][4]))
        if (nuke_key + '27)') in config_dict:
            myagn.hotdust = True
            print ("host dust is added to the model")
            par_total.add('{0}Lhotdust'.format(Nunames[-1]),config_dict[nuke_key + '27)'][0][0],min=config_dict[nuke_key + '27)'][0][1],max=config_dict[nuke_key + '27)'][0][2],brute_step=config_dict[nuke_key + '27)'][0][3],vary=bool(config_dict[nuke_key + '27)'][0][4]))
            par_total.add('{0}Thotdust'.format(Nunames[-1]),config_dict[nuke_key + '27)'][1][0],min=config_dict[nuke_key + '27)'][1][1],max=config_dict[nuke_key + '27)'][1][2],brute_step=config_dict[nuke_key + '27)'][1][3],vary=bool(config_dict[nuke_key + '27)'][1][4]))
        lines_broad = []
        lines_narrow = []
        for name in config_dict[nuke_key + '12)']:
            lines_broad.append(EL.ALLlinedit[name])
        for name in config_dict[nuke_key + '13)']:
            lines_narrow.append(EL.ALLlinedit[name])
        if len(Spectra) == 0:
            waves = []
            L38s = []
            for loopim in range(len(GS_data.allbands)):
                im = GS_data.get_image(loopim)
                waves.append(effective_wave[im.band])
                l38 = imsum[im.band]/im.phys_to_counts_rate
                L38s.append(np.log10(l38))
            # sort waves and L38s based on waves
            waves, L38s = zip(*sorted(zip(waves, L38s)))
            # AGN initilized at rest frame
            wave_obs = np.linspace(min(waves)*0.9,max(waves)*1.1,300)
            flux_obs = np.interp(wave_obs, waves, L38s)
            wave, flux = sedI.sed_to_rest(wave_obs,flux_obs,z=config_dict[nuke_key + '2)'][0],ebv=config_dict[nuke_key + '3)'])
            havespec = False
        else:
            Wave = []
            Flux = []
            for i in range(len(Spectra)):
                Wave.append(Spectra[i][0])
                Flux.append(Spectra[i][1])
            wave = np.unique(np.concatenate(Wave))
            flux = np.zeros_like(wave)
            for i in range(len(Spectra)):
                flux += np.interp(wave, Wave[i], Flux[i])
            flux /= len(Spectra)
            havespec = True
        par_total = myagn.set_full_model([wave,flux],lines_broad=lines_broad,lines_narrow=lines_narrow, par_tot = par_total,nbroad = config_dict[nuke_key + '14)'],
                                nnarrow = config_dict[nuke_key + '15)'], addBaC=bool(config_dict[nuke_key + '16)']),addFeII =  bool(config_dict[nuke_key + '17)']),havespec=havespec)
        if (not feUV) & (bool(config_dict[nuke_key + '17)'])):
            par_total['FeII{0}A_uv'.format(Nunames[-1])].set(value = 0, vary = False)
        if fixagnlinepro:
            if bool(config_dict[nuke_key + '17)']):
                par_total['FeII{0}fwhm'.format(Nunames[-1])].set(vary = False)
                par_total['FeII{0}dcen'.format(Nunames[-1])].set(vary = False)
            for line in lines_broad:
                for loopl in range(config_dict[nuke_key + '14)']):
                    par_total['{0}{1}b{2}peak'.format(line['name'],Nunames[-1],loopl+1)].set(vary = False)
                    par_total['{0}{1}b{2}cen'.format(line['name'],Nunames[-1],loopl+1)].set(vary = False)
                    par_total['{0}{1}b{2}wid'.format(line['name'],Nunames[-1],loopl+1)].set(vary = False)
            for line in lines_narrow:
                for loopl in range(config_dict[nuke_key + '15)']):
                    par_total['{0}{1}n{2}peak'.format(line['name'],Nunames[-1],loopl+1)].set(vary = False)
                    par_total['{0}{1}n{2}cen'.format(line['name'],Nunames[-1],loopl+1)].set(vary = False)
                    par_total['{0}{1}n{2}wid'.format(line['name'],Nunames[-1],loopl+1)].set(vary = False)
                    
        Nucleus.append(myagn)

    # load Stellar components
    FGstars = []
    FGnames = []
    for fg in fgslabel:
        mtype_list.append('star')
        fg_key = 'F' + fg
        FGnames.append(config_dict[fg_key + '1)'])
        useSED = bool(config_dict[fg_key + '6)'])
        par_total.add('xcen_{0}'.format(FGnames[-1]),config_dict[fg_key + '2)'][0],min=config_dict[fg_key + '2)'][1],max=config_dict[fg_key + '2)'][2],brute_step=config_dict[fg_key + '2)'][3],vary=bool(config_dict[fg_key + '2)'][4]))
        par_total.add('ycen_{0}'.format(FGnames[-1]),config_dict[fg_key + '3)'][0],min=config_dict[fg_key + '3)'][1],max=config_dict[fg_key + '3)'][2],brute_step=config_dict[fg_key + '3)'][3],vary=bool(config_dict[fg_key + '3)'][4]))            
        mystar = GALA.Star(prefix=FGnames[-1],useSED=useSED)
        mystar.sortwave = sortwave ## TODO: delete
        
        if useSED:
            par_total.add('{0}_Teff'.format(FGnames[-1]),config_dict[fg_key + '4)'][0],min=config_dict[fg_key + '4)'][1],max=config_dict[fg_key + '4)'][2],brute_step=config_dict[fg_key + '4)'][3],vary=bool(config_dict[fg_key + '4)'][4]))
            par_total.add('{0}_logL38'.format(FGnames[-1]),config_dict[fg_key + '5)'][0],min=config_dict[fg_key + '5)'][1],max=config_dict[fg_key + '5)'][2],brute_step=config_dict[fg_key + '5)'][3],vary=bool(config_dict[fg_key + '5)'][4]))
            if fg_key + '7)' in config_dict:
                par_total.add('{0}_logg'.format(FGnames[-1]),config_dict[fg_key + '7)'][0],min=config_dict[fg_key + '7)'][1],max=config_dict[fg_key + '7)'][2],brute_step=config_dict[fg_key + '7)'][3],vary=bool(config_dict[fg_key + '7)'][4]))
                par_total.add('{0}_logZ'.format(FGnames[-1]),config_dict[fg_key + '8)'][0],min=config_dict[fg_key + '8)'][1],max=config_dict[fg_key + '8)'][2],brute_step=config_dict[fg_key + '8)'][3],vary=bool(config_dict[fg_key + '8)'][4]))
        else:
            nimage = len(GS_data.allbands)
            wavels = []
            bandlist = []
            for i in range(nimage):
                im = GS_data.get_image(i)
                bandlist.append(im.band)
                flux = jnp.where(im.cut_mask_image,0,im.cut_image) 
                logLint = jnp.log10(jnp.nansum(jnp.abs(flux))/im.phys_to_counts_rate)+37
                par_total.add('{0}_{1}_logL'.format(FGnames[-1],im.band),logLint,min=logLint-4,max=logLint+4,brute_step=0.1,vary=True)
                bandlist.append(im.band)
                wavels.append(effective_wave[im.band])
            bandlist = [x for _,x in sorted(zip(wavels,bandlist))]
            wavels = sorted(wavels)
            mystar.bandslist = bandlist
            mystar.iwaves = np.array(wavels)
        FGstars.append(mystar)

    varnames=[]
    pardict = par_total.valuesdict()
    exelist = []
    exelist.append("def Update_Constraints(pardictlc):")
    Haveconstraints = False
    for el in pardict:
        varnames.append(el) # Are varnames not those variable parameters? 

    # Z-mode mass normalization with template-specific EZY_logM constants.
    # These constants are copied from sed_interp.EZY_logM and rounded to 3 decimals.
    ez_logm_vals = [
        10.043, 11.459, 10.108, 12.427, 8.540, 10.459,
        8.560, 10.116, 10.909, 11.708, 9.943, 10.250,
        10.162, 10.068, 10.874, 10.199, 11.178, 10.885,
    ] ## TODO: put this at the very front.
    for pname, sed_mode in zip(Profilesnames, SEDmodelist):
        if sed_mode == 'Z':
            ez_terms = '+'.join([
                "10**(pardictlc['ezlogn{0}_{1}']+{2:.3f})".format(i_ez + 1, pname, ez_logm_vals[i_ez])
                for i_ez in range(18)
            ])
            exelist.append("    pardictlc['logM_{0}'] = jnp.log10({1})".format(pname, ez_terms))
            Haveconstraints = True

    # apply typical AGN constraints
    gprior = []
    for agnp in agnp_label:
        Haveconstraints = True
        pname = pfile['AGN{0}1)'.format(agnp)]
        parname = '{0}logL5100'.format(pname)
        # fix width from Ho 2015, 6.91 for both bulge types
        # set a minimum width 2000 km/s and max 20000 km/s
        linehbname = 'Hb{0}b1wid'.format(pname)
        if linehbname in varnames:
            exelist.append("    pardictlc['Hb{0}b1wid'] = 6.88*jnp.sqrt(jnp.power(10, pardictlc['agn_logM{0}']-6.91)* jnp.power(10**(pardictlc['{1}']-44),-0.533))".format(pname,parname))
            exelist.append("    pardictlc['Hb{0}b1wid'] = jnp.maximum(pardictlc['Hb{0}b1wid'], 13.762)".format(pname))
            exelist.append("    pardictlc['Hb{0}b1wid'] = jnp.minimum(pardictlc['Hb{0}b1wid'], 137.62)".format(pname))
        linehaname = 'Ha{0}b1wid'.format(pname)
        if linehaname in varnames:
            exelist.append("    pardictlc['Ha{0}b1wid'] = 1.35*pardictlc['Hb{0}b1wid']".format(pname))
        # constrain strong line luminosity

        if linehaname in varnames:
            pbHa = pfile['AGN{0}4)'.format(agnp)]
            if pbHa[2] > 0. :
                par_total.add('Sigma_bHa_{0}'.format(pname),0,min=-4*pbHa[2]- 0.05,max=4*pbHa[2] + 0.05,brute_step=0.1*pbHa[2],vary=True)
                gprior.append(['Sigma_bHa_{0}'.format(pname), 0., pbHa[2]])
                exelist.append("    pardictlc['Ha{0}b1peak'] = 10**({1}*pardictlc['{2}']+pardictlc['Sigma_bHa_{0}']+{3})/(2.5*pardictlc['Ha{0}b1wid'])".format(pname,pbHa[1],parname,pbHa[0]))
                par_total['Ha{0}b1peak'.format(pname)].set(vary=False)
            elif pbHa[2] == 0. :
                par_total.add('Sigma_bHa_{0}'.format(pname),0,min=-4*pbHa[2]-0.05,max=4*pbHa[2]+0.05,brute_step=0.1*pbHa[1],vary=False)
                exelist.append("    pardictlc['Ha{0}b1peak'] = 10**({1}*pardictlc['{2}']+pardictlc['Sigma_bHa_{0}']+{3})/(2.5*pardictlc['Ha{0}b1wid'])".format(pname,pbHa[1],parname,pbHa[0]))
                par_total['Ha{0}b1peak'.format(pname)].set(vary=False)
            par_total['Ha{0}b1wid'.format(pname)].set(vary=False)
            par_total['Ha{0}b1peak'.format(pname)].set(vary=False)
            par_total['Ha{0}b1cen'.format(pname)].set(vary=False)
        if linehbname in varnames:
            pbHb = pfile['AGN{0}5)'.format(agnp)]
            if pbHb[2] > 0. :
                par_total.add('Sigma_bHb_{0}'.format(pname),0,min=-4*pbHb[2]-0.05,max=4*pbHb[2]+0.05,brute_step=0.1*pbHb[2],vary=True)
                gprior.append(['Sigma_bHb_{0}'.format(pname), 0., pbHb[2]])
                exelist.append("    pardictlc['Hb{0}b1peak'] = 10**({1}*pardictlc['{2}']+pardictlc['Sigma_bHb_{0}']+{3})/(2.5*pardictlc['Hb{0}b1wid'])".format(pname,pbHb[1],parname,pbHb[0]))
                par_total['Hb{0}b1peak'.format(pname)].set(vary=False)
            elif pbHb[2] == 0. :
                par_total.add('Sigma_bHb_{0}'.format(pname),0,min=-4*pbHb[2]-0.05,max=4*pbHb[2]+0.05,brute_step=0.1*pbHb[1],vary=False)
                exelist.append("    pardictlc['Hb{0}b1peak'] = 10**({1}*pardictlc['{2}']+pardictlc['Sigma_bHb_{0}']+{3})/(2.5*pardictlc['Hb{0}b1wid'])".format(pname,pbHb[1],parname,pbHb[0]))
                par_total['Hb{0}b1peak'.format(pname)].set(vary=False)
            par_total['Hb{0}b1wid'.format(pname)].set(vary=False)
            par_total['Hb{0}b1peak'.format(pname)].set(vary=False)
            par_total['Hb{0}b1cen'.format(pname)].set(vary=False)
        lineoIIIname = 'OIII_5007{0}n1wid'.format(pname)
        if lineoIIIname in varnames:
            pnOIII = pfile['AGN{0}8)'.format(agnp)]
            if pnOIII[2] > 0. :
                par_total.add('Sigma_nOIII_{0}'.format(pname),0,min=-4*pnOIII[2]-0.05,max=4*pnOIII[2]+0.05,brute_step=0.1*pnOIII[2],vary=True)
                gprior.append(['Sigma_nOIII_{0}'.format(pname), 0., pnOIII[2]])
                exelist.append("    pardictlc['OIII_5007{0}n1peak'] = 10**({1}*pardictlc['{2}']+pardictlc['Sigma_nOIII_{0}']+{3})/(2.5*pardictlc['OIII_5007{0}n1wid'])".format(pname,pnOIII[1],parname,pnOIII[0]))
                par_total['OIII_5007{0}n1peak'.format(pname)].set(vary=False)
            elif pnOIII[2] == 0. :
                par_total.add('Sigma_nOIII_{0}'.format(pname),0,min=-4*pnOIII[2]-0.05,max=4*pnOIII[2]+0.05,brute_step=0.1*pnOIII[1],vary=False)
                exelist.append("    pardictlc['OIII_5007{0}n1peak'] = 10**({1}*pardictlc['{2}']+pardictlc['Sigma_nOIII_{0}']+{3})/(2.5*pardictlc['OIII_5007{0}n1wid'])".format(pname,pnOIII[1],parname,pnOIII[0]))
                par_total['OIII_5007{0}n1peak'.format(pname)].set(vary=False)
            par_total['OIII_5007{0}n1wid'.format(pname)].set(vary=False)
            par_total['OIII_5007{0}n1cen'.format(pname)].set(vary=False)
        # constrain of mass mass relation
        gname = pfile['AGN{0}2)'.format(agnp)]
        gmparm = pfile['AGN{0}3)'.format(agnp)]
        if gmparm[2] > 0. :
            par_total.add('Sigma_gm_{0}'.format(pname),0,min=-4*gmparm[2]-0.05,max=4*gmparm[2]+0.05,brute_step=0.1*gmparm[2],vary=True)
            gprior.append(['Sigma_gm_{0}'.format(pname), 0., gmparm[2]])
            exelist.append("    pardictlc['agn_logM{0}'] = {1}*pardictlc['logM_{2}']+pardictlc['Sigma_gm_{0}']+{3}".format(pname,gmparm[1],gname,gmparm[0]))
            par_total['agn_logM{0}'.format(pname)].set(vary=False)
        elif gmparm[2] == 0. :
            par_total.add('Sigma_gm_{0}'.format(pname),0,min=-4*gmparm[2]-0.05,max=4*gmparm[2]+0.05,brute_step=0.1*gmparm[1],vary=False)
            exelist.append("    pardictlc['agn_logM{0}'] = {1}*pardictlc['logM_{2}']+pardictlc['Sigma_gm_{0}']+{3}".format(pname,gmparm[1],gname,gmparm[0]))
            par_total['agn_logM{0}'.format(pname)].set(vary=False)
        # constrain of AGN center
        gname = pfile['AGN{0}11)'.format(agnp)]
        if gname != 'none':
            exelist.append("    pardictlc['xcen_{0}'] = pardictlc['{1}_xcen']".format(pname,gname))
            exelist.append("    pardictlc['ycen_{0}'] = pardictlc['{1}_ycen']".format(pname,gname))
            par_total['xcen_{0}'.format(pname)].set(vary=False)
            par_total['ycen_{0}'.format(pname)].set(vary=False)
        # constrain emission from BLR
        agnfe = 'FeII{0}A_op'.format(pname)
        RfeII = pfile['AGN{0}12)'.format(agnp)]
        if RfeII != 'none':
            par_total.add('RFeII_{0}_op'.format(pname),RfeII[0][0],min=RfeII[0][1],max=RfeII[0][2],brute_step=RfeII[0][3],vary=bool(RfeII[0][4]))
            par_total.add('RFeII_{0}_uv'.format(pname),RfeII[1][0],min=RfeII[1][1],max=RfeII[1][2],brute_step=RfeII[1][3],vary=bool(RfeII[1][4]))
        if agnfe in varnames:
            par_total['FeII{0}dcen'.format(pname)].set(value=0., vary=False)
            exelist.append("    pardictlc['FeII{0}A_op'] = 10**(1.133*pardictlc['{1}']-45.70)*pardictlc['RFeII_{0}_op']/43111".format(pname,parname))
            exelist.append("    pardictlc['FeII{0}A_uv'] = pardictlc['FeII{0}A_op']*pardictlc['RFeII_{0}_uv']".format(pname))
            exelist.append("    pardictlc['FeII{0}fwhm'] = 145.33*pardictlc['Hb{0}b1wid']".format(pname))
            par_total['FeII{0}fwhm'.format(pname)].set(vary=False)
            par_total['FeII{0}A_op'.format(pname)].set(vary=False)
            par_total['FeII{0}A_uv'.format(pname)].set(vary=False)
        agnBac = 'BaC{0}logL3600'.format(pname)
        if agnBac in varnames:
            par_total['BaC{0}dcen'.format(pname)].set(value=0., vary=False)
            exelist.append("    pardictlc['BaC{0}fwhm'] = 145.33*pardictlc['Hb{0}b1wid']".format(pname))
            par_total['BaC{0}fwhm'.format(pname)].set(vary=False)
        # set up other Broad lines
        linenames = pfile['AGN{0}6)'.format(agnp)]
        lineratios = pfile['AGN{0}7)'.format(agnp)]
        agntypicallines = []
        for name1 in linenames:
            agntypicallines.append(EL.ALLlinedit[name1])
        for i in range(len(agntypicallines)):
            line = agntypicallines[i]
            lname = '{0}{1}b1peak'.format(line['name'],pname)
            if lname in varnames:
                if lineratios[i] > 0. :
                    exelist.append("    pardictlc['{0}{1}b1peak'] = pardictlc['Hb{1}b1peak']*{2}".format(line['name'],pname,lineratios[i]))
                    par_total['{0}{1}b1peak'.format(line['name'],pname)].set(vary=False)
                else:
                    par_total['{0}{1}b1peak'.format(line['name'],pname)].set(vary=True)
                exelist.append("    pardictlc['{0}{1}b1wid'] = pardictlc['Hb{1}b1wid']".format(line['name'],pname))
                par_total['{0}{1}b1wid'.format(line['name'],pname)].set(vary=False)
                par_total['{0}{1}b1cen'.format(line['name'],pname)].set(vary=False)
        # set up other narrow lines
        linenames = pfile['AGN{0}9)'.format(agnp)]
        lineratios = pfile['AGN{0}10)'.format(agnp)]
        agntypicallines = []
        for name1 in linenames:
            agntypicallines.append(EL.ALLlinedit[name1])
        for i in range(len(agntypicallines)):
            line = agntypicallines[i]
            lname = '{0}{1}n1peak'.format(line['name'],pname)
            if lname in varnames:
                exelist.append("    pardictlc['{0}{1}n1peak'] = pardictlc['OIII_5007{1}n1peak']*{2}".format(line['name'],pname,lineratios[i]))
                par_total['{0}{1}n1peak'.format(line['name'],pname)].set(vary=False)
                par_total['{0}{1}n1wid'.format(line['name'],pname)].set(value = par_total['OIII_5007{0}n1wid'.format(pname)].value, vary=False)
                par_total['{0}{1}n1cen'.format(line['name'],pname)].set(vary=False)                

    #print ("check AGN priors: ", exelist)
    # apply mass size relation
    for msrp in msr_label:
        Haveconstraints = True
        pname = pfile['MSR{0}1)'.format(msrp)]
        pparam = pfile['MSR{0}2)'.format(msrp)] 
        gname = ''
        for gake in galaxylabel:
            gakey = 'G' + gake
            pros =  config_dict[gakey + '2)']
            proindex = [profilelabel.index(pro) for pro in pros]
            for i in proindex:
                if Profilesnames[i] == pname:
                    gname = config_dict[gakey + '1)']
        if gname == '':
            raise ValueError('Profile name not recognized')
        if pparam[2] > 0. :
            par_total.add('Sigma_MSR_{0}'.format(pname),0,min=-4*pparam[2]-0.05,max=4*pparam[2]+0.05,brute_step=0.1*pparam[2],vary=True)
            gprior.append(['Sigma_MSR_{0}'.format(pname), 0., pparam[2]])
            exelist.append("    pardictlc['{0}_Re'] = 10**({1}*pardictlc['logM_{0}']+pardictlc['Sigma_MSR_{0}']+{2})/pardictlc['scale_{3}']".format(pname,pparam[1],pparam[0]-pparam[1]*10.7,gname))
            par_total['{0}_Re'.format(pname)].set(vary=False)
        elif pparam[2] == 0. :
            par_total.add('Sigma_MSR_{0}'.format(pname),0,min=pparam[2]-0.05,max=-pparam[2]+0.05,brute_step=0.1*pparam[2],vary=False)
            exelist.append("    pardictlc['{0}_Re'] = 10**({1}*pardictlc['logM_{0}']+pardictlc['Sigma_MSR_{0}']+{2})/pardictlc['scale_{3}']".format(pname,pparam[1],pparam[0]-pparam[1]*10.7,gname))
            par_total['{0}_Re'.format(pname)].set(vary=False)

    # apply mass attenuation relation
    for marp in mar_label:
        Haveconstraints = True
        pname = pfile['MAR{0}1)'.format(marp)]
        pparam = pfile['MAR{0}2)'.format(marp)]
        gname = ''
        for gake in galaxylabel:
            gakey = 'G' + gake
            pros =  config_dict[gakey + '2)']
            proindex = [profilelabel.index(pro) for pro in pros]
            for i in proindex:
                if Profilesnames[i] == pname:
                    gname = config_dict[gakey + '1)']
        if gname == '':
            raise ValueError('Profile name not recognized')
        if pparam[5] > 0. :
            par_total.add('Sigma_MAR_{0}'.format(pname),0,min=-4*pparam[5]-0.05,max=4*pparam[5]+0.05,brute_step=0.1*pparam[5],vary=True)
            gprior.append(['Sigma_MAR_{0}'.format(pname), 0., pparam[5]])
            exelist.append("    pardictlc['{0}_Av_value'] = {1}+{2}*pardictlc['logM_{0}']+{3}*pardictlc['logM_{0}']**2+{4}*pardictlc['logM_{0}']**3+{5}*pardictlc['logM_{0}']**4+pardictlc['Sigma_MAR_{0}']".format(pname,pparam[4],pparam[3],pparam[2],pparam[1],pparam[0]))
            par_total['{0}_Av_value'.format(pname)].set(vary=False)
        elif pparam[5] == 0. :
            par_total.add('Sigma_MAR_{0}'.format(pname),0,min=pparam[5]-0.05,max=-pparam[5]+0.05,brute_step=0.1*pparam[5],vary=False)
            exelist.append("    pardictlc['{0}_Av_value'] = {1}+{2}*pardictlc['logM_{0}']+{3}*pardictlc['logM_{0}']**2+{4}*pardictlc['logM_{0}']**3+{5}*pardictlc['logM_{0}']**4+pardictlc['Sigma_MAR_{0}']".format(pname,pparam[4],pparam[3],pparam[2],pparam[1],pparam[0]))
            par_total['{0}_Av_value'.format(pname)].set(vary=False)

    # apply mass metalicity relation
    for mmrp in mmr_label:
        Haveconstraints = True
        pname = pfile['MMR{0}1)'.format(mmrp)]
        pparam = pfile['MMR{0}2)'.format(mmrp)]
        if len(pparam) == 4: # adopt the KD02 z = 0 MZR
            exelist.append("    pardictlc['{0}_Z_value'] = 0.02*10**({1}+{2}*pardictlc['logM_{0}']+{3}*pardictlc['logM_{0}']**2+{4}*pardictlc['logM_{0}']**3-8.69)".format(pname,pparam[0],pparam[1],pparam[2],pparam[3]))
        if len(pparam) == 3: # adopt the Nakajima23 z=4-8 MZR
            if pparam[2] > 0.:
                par_total.add('Sigma_MZR_{0}'.format(pname),0,min=-4*pparam[2]-0.05,max=4*pparam[2]+0.05,brute_step=0.1*pparam[2],vary=True)
                gprior.append(['Sigma_MZR_{0}'.format(pname), 0., pparam[2]])
                exelist.append("    pardictlc['{0}_Z_value'] = 0.02*10**({1}+{2}*pardictlc['logM_{0}']+pardictlc['Sigma_MZR_{0}']-8.69)".format(pname,pparam[0],pparam[1]))
            elif pparam[2] == 0.:
                par_total.add('Sigma_MZR_{0}'.format(pname),0,min=pparam[2]-0.05,max=-pparam[2]+0.05,brute_step=0.1*pparam[2],vary=False)
                exelist.append("    pardictlc['{0}_Z_value'] = 0.02*10**({1}+{2}*pardictlc['logM_{0}']+pardictlc['Sigma_MZR_{0}']-8.69)".format(pname,pparam[0],pparam[1]))
        # metalicity should in range , [0.00001,0.04]
        exelist.append("    pardictlc['{0}_Z_value'] = jnp.maximum(pardictlc['{0}_Z_value'], 0.00001)".format(pname))
        exelist.append("    pardictlc['{0}_Z_value'] = jnp.minimum(pardictlc['{0}_Z_value'], 0.04)".format(pname))
        par_total['{0}_Z_value'.format(pname)].set(vary=False)
    
    # apply SFH constraints
    for sfhp in sfhp_label: 
        Haveconstraints = True
        pname = pfile['SFH{0}1)'.format(sfhp)]
        ftype = pfile['SFH{0}2)'.format(sfhp)]
        pparams = pfile['SFH{0}3)'.format(sfhp)] 
        pparam_gprior = pfile['SFH{0}4)'.format(sfhp)]
        for sploop in range(len(pparams)):
            par_total.add('{0}_SFH_p{1}'.format(pname,sploop+1),pparams[sploop][0],min=pparams[sploop][1],max=pparams[sploop][2],brute_step=pparams[sploop][3],vary=bool(pparams[sploop][4]))
            if len(pparam_gprior[sploop]) == 2:
                gprior.append(['{0}_SFH_p{1}'.format(pname,sploop+1), pparam_gprior[sploop][0], pparam_gprior[sploop][1]])    
        for i in range(len(Profilesnames)):
            if Profilesnames[i] == pname:
                abin = np.array(agebin[i])
                ages = 0.5*(abin[0:len(abin)-1]+abin[1:len(abin)])
                period = abin[1:len(abin)]-abin[0:len(abin)-1]
                ages = np.max(abin)-ages
                for loopage in range(len(ages)):
                    if ftype == 'exponetial':
                        exelist.append("    pardictlc['{0}_f_cont_bin{1}'] = jnp.log10({3}) + jnp.log10( jnp.exp(({2}-pardictlc['{0}_SFH_p3'])/pardictlc['{0}_SFH_p2']) - jnp.exp(-pardictlc['{0}_SFH_p3']/pardictlc['{0}_SFH_p2']))".format(pname,loopage+1,ages[loopage],period[loopage]))
                    elif ftype == 'delayed':
                        exelist.append("    pardictlc['{0}_f_cont_bin{1}'] = jnp.log10({3}) + jnp.log10( jnp.exp((-{2}/pardictlc['{0}_SFH_p2'])*{2}/pardictlc['{0}_SFH_p2']**2))".format(pname,loopage+1,ages[loopage],period[loopage]))
                    elif ftype == 'exponetial2':
                        # parameter [logSFR0, tau0, f, tau1, t1]
                        # numerator   = f * tau0 * (1 - np.exp(-t0/tau0))
                        # denominator = (1 - f) * tau1 * (1 - np.exp(-t1/tau1))
                        # k = numerator / denominator
                        if loopage == 0:
                            exelist.append("    numerator = pardictlc['{0}_SFH_p3']*pardictlc['{0}_SFH_p2']*(1.-jnp.exp(-{1}/pardictlc['{0}_SFH_p2']))".format(pname,np.max(abin)))
                            exelist.append("    denominator = (1.-pardictlc['{0}_SFH_p3'])*pardictlc['{0}_SFH_p4']*(1.-jnp.exp(-pardictlc['{0}_SFH_p5']/pardictlc['{0}_SFH_p4']))".format(pname))
                            exelist.append("    k = numerator/denominator")
                        exelist.append("    pardictlc['{0}_f_cont_bin{1}'] = jnp.log10({3}) + jnp.log10(jnp.exp(-{2}/pardictlc['{0}_SFH_p2'])+k*jnp.exp(({4}-pardictlc['{0}_SFH_p5']-{2})/pardictlc['{0}_SFH_p4'])*jnp.where(({2}-{4}+pardictlc['{0}_SFH_p5'])>0.,1.,0.))".format(pname,loopage+1,ages[loopage],period[loopage],np.max(abin)))
                    else:
                        raise ValueError('SFH type not recognized')
                if len(pparam_gprior[0]) > 0:
                    par_total['logM_{0}'.format(pname)].set(vary=False)
                    newcons = "    summass_{0} = ".format(pname)
                    for loopage in range(len(ages)-1):
                        newcons += "10**pardictlc['{0}_f_cont_bin{1}']+".format(pname,loopage+1)
                    newcons += "10**pardictlc['{0}_f_cont_bin{1}']".format(pname,len(ages))
                    exelist.append(newcons)
                    exelist.append("    mass_frac_bin1_{0} = 10**pardictlc['{0}_f_cont_bin1']/summass_{0}".format(pname))
                    exelist.append("    pardictlc['logM_{0}'] = 9. + jnp.log10(10**pardictlc['{0}_SFH_p1']*{1}/mass_frac_bin1_{0})".format(pname,period[0]))

    # student-t Prior for SFH
    # f_cont_t_adj_priors: list of (name_i, name_j, center, scale, df)
    #   center = ratio of adjacent bins (delta log mass_fraction)
    #   constraint: (theta[j] - theta[i]) ~ StudentT(center, scale, df)

    try:
        pname_list, ftype_list  = [], []
        for sfh_tp in sfh_tp_label:
            pname_list.append(pfile[f"SFH_T{sfh_tp}1)"])
            ftype_list.append(pfile[f"SFH_T{sfh_tp}2)"])
        scale   = 0.3
        df  = 2
        if not pname_list:
            f_tprior_info   = None
        else:
            f_tprior_info    = (pname_list, ftype_list, scale, df)


    except Exception as e:
        f_tprior_info   = None
        print(f"=="*10)
        raise ValueError(f"An error occured:{e}, please check your prior setting.")


    # Gaussian priors
    try:
        GP_list = pfile['GP)']
        for gpname in GP_list:
            value = par_total[gpname].value
            std = par_total[gpname].brute_step
            gprior.append([gpname, value, std])
    except:
        print ("No additional Gaussian priors")

    # FIX) module: fix specified parameters to given default values
    # Reads two parallel lists from the prior file:
    #   FIX1) ['param_name_a', 'param_name_b', ...]
    #   FIX2) [value_a, value_b, ...]
    # Each matched parameter is set to its FIX2 value and vary=False so that
    # the downstream fitter treats it as a fixed (non-free) parameter.
    try:
        FIX_names = list(pfile['FIX1)'])
        FIX_values = list(pfile['FIX2)'])
        if len(FIX_names) != len(FIX_values):
            raise ValueError("FIX1) and FIX2) have different lengths: {0} vs {1}".format(len(FIX_names), len(FIX_values)))
        print ("\n===== FIX) module: fixing parameters =====")
        for fix_name, fix_value in zip(FIX_names, FIX_values):
            if fix_name not in par_total:
                print ("  [FIX] WARNING: parameter '{0}' not found in par_total, skipped".format(fix_name))
                continue
            par_total[fix_name].set(value=float(fix_value), vary=False)
            print ("  [FIX] {0} = {1:.6g}  (vary=False)".format(fix_name, float(fix_value)))
        print ("===========================================\n")
    except (KeyError, UnboundLocalError):
        # KeyError: prior file lacks FIX1)/FIX2) section
        # UnboundLocalError: no --priorpath given at all, so pfile was never assigned
        # Either way, silently skip the FIX) module.
        pass
    except Exception as exc:
        print ("[FIX] module encountered an error: {0}".format(exc))

    # Energy balances
    energy_balance = []
    try:
        EB_list = pfile['EB)']
        for ebname in EB_list:
            energy_balance.append(ebname)
    except:
        print ("No energy balances assigned")


    # define galaxy model
    Galaxies = []
    ganames = []
    for gake in galaxylabel:
        mtype_list.append('galaxy')
        gakey = 'G' + gake
        ganames.append(config_dict[gakey + '1)'])
        pros =  config_dict[gakey + '2)'] # profile 
        proindex = [profilelabel.index(pro) for pro in pros]
        par_total.add('z_{0}'.format(ganames[-1]),config_dict[gakey + '3)'][0],min=config_dict[gakey + '3)'][1],max=config_dict[gakey + '3)'][2],brute_step=config_dict[gakey + '3)'][3],vary=bool(config_dict[gakey + '3)'][4]))
        zg = config_dict[gakey + '3)'][0]
        da = cosmo.angular_diameter_distance(zg)
        scale = da.to(u.kpc).value*np.pi/180/3600 # kpc per arcsec
        par_total.add('scale_{0}'.format(ganames[-1]),scale,min=0.02,max=12,brute_step=0.01,vary=False) ## What is this?
        magal = GALA.Galaxy(regionz=regionz,z=zg,ebv_G=config_dict[gakey + '4)'], name=config_dict[gakey + '1)'], wave_temp = wave_temp)
        magal.currentage = cosmo.age(magal.redshift).value
        magal.sortwave = sortwave ## TODO: delete
        g_sed_modes = [SEDmodelist[i] for i in proindex]
        has_eazy_in_galaxy = any([mode == 'Z' for mode in g_sed_modes])
        if has_eazy_in_galaxy and (not all([mode == 'Z' for mode in g_sed_modes])):
            raise ValueError("All profiles in a galaxy must use SED mode 'Z' if any profile uses 'Z'")
        magal.useEZ = has_eazy_in_galaxy
        lines_narrow = []
        for name in config_dict[gakey + '6)']:
            lines_narrow.append(EL.ALLlinedit[name])
        if (gakey + '8)') in config_dict:
            par_total.add('r0_{0}'.format(ganames[-1]),config_dict[gakey + '8)'][0][0],min=config_dict[gakey + '8)'][0][1],max=config_dict[gakey + '8)'][0][2],brute_step=config_dict[gakey + '8)'][0][3],vary=bool(config_dict[gakey + '8)'][0][4]))
            par_total.add('rt_{0}'.format(ganames[-1]),config_dict[gakey + '8)'][1][0],min=config_dict[gakey + '8)'][1][1],max=config_dict[gakey + '8)'][1][2],brute_step=config_dict[gakey + '8)'][1][3],vary=bool(config_dict[gakey + '8)'][1][4]))
            par_total.add('v0_{0}'.format(ganames[-1]),config_dict[gakey + '8)'][2][0],min=config_dict[gakey + '8)'][2][1],max=config_dict[gakey + '8)'][2][2],brute_step=config_dict[gakey + '8)'][2][3],vary=bool(config_dict[gakey + '8)'][2][4]))
            par_total.add('vc_{0}'.format(ganames[-1]),config_dict[gakey + '8)'][3][0],min=config_dict[gakey + '8)'][3][1],max=config_dict[gakey + '8)'][3][2],brute_step=config_dict[gakey + '8)'][3][3],vary=bool(config_dict[gakey + '8)'][3][4]))
            par_total.add('LineSigma_{0}'.format(ganames[-1]),config_dict[gakey + '8)'][4][0],min=config_dict[gakey + '8)'][4][1],max=config_dict[gakey + '8)'][4][2],brute_step=config_dict[gakey + '8)'][4][3],vary=bool(config_dict[gakey + '8)'][4][4]))
        else:
            par_total.add('r0_{0}'.format(ganames[-1]),0,min=0,max=1,brute_step=0.1,vary=False)
            par_total.add('rt_{0}'.format(ganames[-1]),3,min=0.5,max=15,brute_step=0.1,vary=False)
            par_total.add('v0_{0}'.format(ganames[-1]),30,min=150,max=300,brute_step=1.,vary=False)
            par_total.add('vc_{0}'.format(ganames[-1]),0,min=-100,max=100,brute_step=1.,vary=False)
            par_total.add('LineSigma_{0}'.format(ganames[-1]),50,min=10.,max=300.,brute_step=1.,vary=False)
        if (len(Spectra) == 0)&(not grismfit): ## image + sed fit.
            magal.make_nebular(None, lines_narrow=[], withline = True)
        elif grismfit:
            par_nebu = magal.make_nebular(None,lines_narrow, nnarrow = config_dict[gakey + '7)'],withline = False, D3=True)
            par_total.update(par_nebu)
        else:
            wave = Spectra[0][0]
            flux = Spectra[0][1]
            par_nebu = magal.make_nebular([wave,flux],lines_narrow, nnarrow = config_dict[gakey + '7)'],withline = False)
            par_total.update(par_nebu)
        for i in proindex:
            prefix = Profilesnames[i]
            Sparams = Sparams_list[i]
            if agetype[i] == 'const':
                age = {'type': "const", 'paradic':{'value': pardict['{0}_age_value'.format(prefix)]}}
            elif agetype[i] == 'linear':
                age = {'type': "linear", 'paradic':{'in': pardict['{0}_age_in'.format(prefix)], 'out': pardict['{0}_age_out'.format(prefix)]}}
            elif agetype[i] == 'bins':
                age = {'type': "bins", 'paradic':{}}
            if ztype[i] == 'const':
                Z = {'type': "const", 'paradic':{'value':  pardict['{0}_Z_value'.format(prefix)]}}
            elif ztype[i] == 'linear':
                Z = {'type': "linear", 'paradic':{'in': pardict['{0}_Z_in'.format(prefix)], 'out': pardict['{0}_Z_out'.format(prefix)]}}
            if Znebtype[i] == "const":
                Zneb = {'type': "const", 'paradic':{'value': pardict['{0}_Zneb_value'.format(prefix)]}}
            elif Znebtype[i] == 'linear':
                Zneb = {'type':"linear", 'paradic':{'in':pardict['{0}_Zneb_in'.format(prefix)], 'out': pardict['{0}_Zneb_out'.format(prefix)]}}
            if Avtype[i] == 'const':
                Av = {'type': "const", 'paradic':{'value': pardict['{0}_Av_value'.format(prefix)]}}
            elif Avtype[i] == 'linear':
                Av = {'type': "linear", 'paradic':{'in': pardict['{0}_Av_in'.format(prefix)], 'out': pardict['{0}_Av_out'.format(prefix)]}}
            if sfrtype[i] == 'const':
                f_cont = {'type': "const", 'paradic':{'value': pardict['{0}_f_cont_value'.format(prefix)]}}
            elif sfrtype[i] == 'linear':
                f_cont = {'type': "linear", 'paradic':{'in': pardict['{0}_f_cont_in'.format(prefix)], 'out': pardict['{0}_f_cont_out'.format(prefix)]}}
            elif sfrtype[i] == 'bins':
                pdict = {}
                for loopage in range(len(agebin[i])-1):
                    pdict['bin{0}'.format(loopage+1)] = pardict['{0}_f_cont_bin{1}'.format(prefix,loopage+1)]
                f_cont = {'type': "bins", 'paradic':pdict}
            sed_mode = SEDmodelist[i]
            try:
                sed_mode = int(sed_mode)
            except (TypeError, ValueError):
                pass ## not raise error?
            magal.add_subC(prefix,pardict['logM_{0}'.format(prefix)],Sparams,age,Z,Zneb,f_cont,Av,SFH=SFHlist[i],logU = pardict['logU_{0}'.format(prefix)],Avbump = pardict['AVbump_{0}'.format(prefix)],agnbin = agebin[i], SEDmode = sed_mode,table_model=TableModellist[i])
            if prefix in energy_balance:
                if prefix in magal.energy_balance:
                    magal.energy_balance[prefix] = True
        if magal.useEZ and (not magal.allconstant):
            raise ValueError("SED mode 'Z' currently requires all associated profiles to be allconstant in each galaxy")
        Galaxies.append(magal)

    if has_eazy_profile and (not all([magal.allconstant for magal in Galaxies])):
        raise ValueError("SED mode 'Z' is only supported by imagefitter_SED_allconst; please use allconstant profiles for all galaxies")

    varnames = []
    for el in pardict:
        if par_total[el].expr is not None:
            par_total[el].vary = False
            expr = par_total[el].expr
            opra = []
            oploop = []
            transexp = "pardictlc['{0}']=".format(el)
            for loop, a in enumerate(expr):
                if (a == '*') | (a == '/') | (a == '+') | (a == '-'):
                    opra.append(a)
                    oploop.append(loop)
            oploop.append(len(expr))
            transexp += expr[0:oploop[0]]
            for loop, a in enumerate(opra):
                transexp += a
                expre = expr[oploop[loop]+1:oploop[loop+1]]
                # check if it is a parameter
                if expre in pardict:
                    transexp += "pardictlc['{0}']".format(expre)
                else:
                    transexp += expre
            exelist.append("    "+transexp)
            Haveconstraints = True
        if par_total[el].vary:
            varnames.append(el)

    if len(exelist) == 1:
        Haveconstraints = False
    #print (exelist, len(exelist))
    #if Haveconstraints:
    #    print ("The current model has parameter constraints applied.")
    #else:
    #    print ("The current model has NO parameter constraints applied.")

    function_string = "\n".join(exelist)
    if Haveconstraints is False:
        function_string = "def Update_Constraints(pardictlc):\n    pass"

    print(f"=="*10)
    print(f"workplace is {workplace}")
    print(f"targname is {targname}")
    workplace   = str(workplace)
    f = open(workplace + '/{0}.constrain'.format(targname), 'w')
    print(function_string, file=f)
    f.close()

    # Lazy import to avoid circular dependency during module import
    from .gsfit import gsfitter, sedfitter, imagefitter_phot, imagefitter_SED_allconst

    if (GS_data.fitMode == 'photometric - SED')|(GS_data.fitMode == 'photometric + spectrum - SED'):
        Myfitter = sedfitter(GSdata=GS_data, model_list = (Nucleus+FGstars+Galaxies),mtype_list=mtype_list,lmParameters = par_total, g_prior=gprior, f_tprior_info = f_tprior_info)
        print ('Using sedfitter for photometric - SED')
    elif GS_data.fitMode == 'images - photometry':
        Myfitter = imagefitter_phot(GSdata=GS_data, model_list = (Nucleus+FGstars+Galaxies),mtype_list=mtype_list,lmParameters = par_total, g_prior=gprior, f_tprior_info = f_tprior_info)
        print ('Using imagefitter_phot for images - photometry')
    elif (GS_data.fitMode == 'images - SED') & vectorize:
        Myfitter = imagefitter_SED_allconst(GSdata=GS_data, model_list = (Nucleus+FGstars+Galaxies),mtype_list=mtype_list,lmParameters = par_total, g_prior=gprior, f_tprior_info = f_tprior_info)
        print ('Using imagefitter_SED_allconst for images - SED')
    else:
        Myfitter = gsfitter(GSdata=GS_data, model_list = (Nucleus+FGstars+Galaxies),mtype_list=mtype_list,lmParameters = par_total, g_prior=gprior, f_tprior_info = f_tprior_info)
        print ('Using default gsfitter')

    f = open(workplace + '/{0}.params'.format(targname), 'w')
    print ('name value min max step vary expr', file=f)
    for el in Myfitter.pardict:
            print('{0} {1} {2} {3} {4} {5} {6}'.format(el,Myfitter.pardict[el],Myfitter.lmParameters[el].min,Myfitter.lmParameters[el].max,Myfitter.lmParameters[el].brute_step,Myfitter.lmParameters[el].vary,Myfitter.lmParameters[el].expr), file=f)
    f.close()

    return Myfitter, targname, function_string

def add_Fourier_config(config, config_out, refer = 'b', target = 'c', galaxy = 'a', name = 'arm', m = 2, separ = 0.5, phiout = 360 , Re = None):
    config_dict = parse_config_file(config)
    f = open(config_out, 'w')
    f_old = open(config, 'r')
    for line in f_old:
        if line[0:4] == 'G{0}2)'.format(galaxy):
            plist = config_dict['G{0}2)'.format(galaxy)]
            plist.append(target)
            print('G{0}2) {1}'.format(galaxy, plist), file=f)
        else:
            print(line, file=f, end='')
    f_old.close()
    print ('\n# Profile {0}'.format(target),file=f)
    print ('P{0}1) {1}'.format(target, name),file=f)
    print ('P{0}2) sersic_f'.format(target),file=f)
    for loop in range(14):
        print ('P{0}{1}) {2}'.format(target,loop+3,config_dict['P{0}{1})'.format(refer,loop+3)]),file=f)
    if Re is None:
        Re = config_dict['P{0}5)'.format(refer)][0]
    r_sep = Re*separ 
    print ('P{0}17) [{1}, {2}, {3}, 0.5, 1]'.format(target, 0.5*r_sep, 0.1*r_sep, r_sep),file=f)
    print ('P{0}18) [{1}, {2}, {3}, 0.5, 1]'.format(target, 2*r_sep, r_sep, 10*r_sep),file=f)
    print ('P{0}19) [ 1, 0.5, 3 , 0.5, 1]'.format(target),file=f)
    print ('P{0}20) [{1}, {2}, {3}, 1, 1]'.format(target, 0.5*phiout, 0.1*phiout, 2*phiout),file=f)
    print ('P{0}21) {1}'.format(target, m),file=f)
    print ('P{0}22) [0.9,0.,1,0.01,1]'.format(target),file=f)
    print ('P{0}23) [120,0,180,1,1]'.format(target),file=f)
    print ('P{0}24) [30,0,90,1,1]'.format(target),file=f)
    f.close()
    return

def write_config_models(confignamedata, confignamemodel, outputname):
    '''
    Write the model parameters to a config file.
    Parameters:
    configname (str): The name of the config file to write.
    Myfitter (object): The fitter object containing the model parameters.
    outputname (str): The name of the output file to write the parameters to.
    '''
    f_old1 = open(confignamedata, 'r')
    f_old2 = open(confignamemodel, 'r')
    # first keep the original config that begins with 'R', 'I', 'S', 'A' and empty lines
    f_new = open(outputname, 'w')
    for line in f_old1:
        if line[0] in ['R', 'I', 'S', 'A'] or line.strip() == '':
            print(line, file=f_new, end='')
    f_old1.close()
    # then write the model parameters
    # i.e. keep beginning with 'P', 'G', 'N', 'F' and empty lines
    #f_new.write('\n# Model parameters\n')
    for line in f_old2:
        if line[0] in ['P', 'G', 'N', 'F'] or line.strip() == '':
            print(line, file=f_new, end='')
    f_old2.close()
    f_new.close()
    return

def free_sed_config(config, config_out, Myfitter, plabel, SEDmode=0, geovary = False):
    '''
    Free the SED parameters in the config file.
    Parameters:
    config (str): The name of the config file to read.
    config_out (str): The name of the output config file to write.
    Myfitter (object): The fitter object containing the model parameters.
    pname (str): The name of the profile to free the SED parameters for.
    '''
    config_dict = parse_config_file(config)
    lmpara = Myfitter.lmParameters
    f_old = open(config, 'r')
    f_new = open(config_out, 'w')
    writed =  False
    pflabel = 'P{0}'.format(plabel)
    if 'P{0}1)'.format(plabel) not in config_dict:
        shutil.copy(config, config_out)
        return
    for line in f_old:
        if line.startswith(pflabel):
            if not writed:
                pfname = config_dict['P{0}1)'.format(plabel)]
                ptype = config_dict['P{0}2)'.format(plabel)]
                writed = True
                print ('P{0}1) {1}'.format(plabel, config_dict['P{0}1)'.format(plabel)]), file=f_new)
                print ('P{0}2) {1}'.format(plabel, ptype), file=f_new)    
                if not geovary:
                    if ptype == 'sersic':
                        print ('P{0}3) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_xcen'.format(pfname)].value, lmpara['{0}_xcen'.format(pfname)].min, lmpara['{0}_xcen'.format(pfname)].max, lmpara['{0}_xcen'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}4) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_ycen'.format(pfname)].value, lmpara['{0}_ycen'.format(pfname)].min, lmpara['{0}_ycen'.format(pfname)].max, lmpara['{0}_ycen'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}5) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_Re'.format(pfname)].value, lmpara['{0}_Re'.format(pfname)].min, lmpara['{0}_Re'.format(pfname)].max, lmpara['{0}_Re'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}6) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_n'.format(pfname)].value, lmpara['{0}_n'.format(pfname)].min, lmpara['{0}_n'.format(pfname)].max, lmpara['{0}_n'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}7) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_ang'.format(pfname)].value, lmpara['{0}_ang'.format(pfname)].min, lmpara['{0}_ang'.format(pfname)].max, lmpara['{0}_ang'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}8) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_axrat'.format(pfname)].value, lmpara['{0}_axrat'.format(pfname)].min, lmpara['{0}_axrat'.format(pfname)].max, lmpara['{0}_axrat'.format(pfname)].brute_step, int(geovary)), file=f_new)
                    elif (ptype == 'sersic_f')|(ptype == 'sersic_rf')|(ptype == 'sersic_bf'):
                        print ('P{0}3) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_xcen'.format(pfname)].value, lmpara['{0}_xcen'.format(pfname)].min, lmpara['{0}_xcen'.format(pfname)].max, lmpara['{0}_xcen'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}4) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_ycen'.format(pfname)].value, lmpara['{0}_ycen'.format(pfname)].min, lmpara['{0}_ycen'.format(pfname)].max, lmpara['{0}_ycen'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}5) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_Re'.format(pfname)].value, lmpara['{0}_Re'.format(pfname)].min, lmpara['{0}_Re'.format(pfname)].max, lmpara['{0}_Re'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}6) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_n'.format(pfname)].value, lmpara['{0}_n'.format(pfname)].min, lmpara['{0}_n'.format(pfname)].max, lmpara['{0}_n'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}7) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_ang'.format(pfname)].value, lmpara['{0}_ang'.format(pfname)].min, lmpara['{0}_ang'.format(pfname)].max, lmpara['{0}_ang'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}8) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_axrat'.format(pfname)].value, lmpara['{0}_axrat'.format(pfname)].min, lmpara['{0}_axrat'.format(pfname)].max, lmpara['{0}_axrat'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}17) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_r_in'.format(pfname)].value, lmpara['{0}_r_in'.format(pfname)].min, lmpara['{0}_r_in'.format(pfname)].max, lmpara['{0}_r_in'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}18) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_r_out'.format(pfname)].value, lmpara['{0}_r_out'.format(pfname)].min, lmpara['{0}_r_out'.format(pfname)].max, lmpara['{0}_r_out'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}19) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_alpha_rc'.format(pfname)].value, lmpara['{0}_alpha_rc'.format(pfname)].min, lmpara['{0}_alpha_rc'.format(pfname)].max, lmpara['{0}_alpha_rc'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}20) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_theta_out'.format(pfname)].value, lmpara['{0}_theta_out'.format(pfname)].min, lmpara['{0}_theta_out'.format(pfname)].max, lmpara['{0}_theta_out'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}21) {1}'.format(plabel, config_dict['P{0}21)'.format(plabel)]), file=f_new)
                        print ('P{0}22) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_am'.format(pfname)].value, lmpara['{0}_am'.format(pfname)].min, lmpara['{0}_am'.format(pfname)].max, lmpara['{0}_am'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}23) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_theta_m'.format(pfname)].value, lmpara['{0}_theta_m'.format(pfname)].min, lmpara['{0}_theta_m'.format(pfname)].max, lmpara['{0}_theta_m'.format(pfname)].brute_step, int(geovary)), file=f_new)
                        print ('P{0}24) [{1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5}]'.format(plabel, lmpara['{0}_i_arm'.format(pfname)].value, lmpara['{0}_i_arm'.format(pfname)].min, lmpara['{0}_i_arm'.format(pfname)].max, lmpara['{0}_i_arm'.format(pfname)].brute_step, int(geovary)), file=f_new)
                else:
                    # write exact the original values
                    for i in range(3, 9):
                        pvalue = config_dict['P{0}{1})'.format(plabel, i)]
                        print ('P{0}{1}) {2}'.format(plabel, i, pvalue), file=f_new)
                    if ptype.endswith('f'):
                        for i in range(17, 25):
                            pvalue = config_dict['P{0}{1})'.format(plabel, i)]
                            print ('P{0}{1}) {2}'.format(plabel, i, pvalue), file=f_new)
                # for p9-p16, we just copy the original values
                p9 = config_dict['P{0}9)'.format(plabel)]
                # whether p9 is 2d list or not
                if isinstance(p9[0], list):
                    for li9 in p9:
                        li9[-1] = 1
                print ('P{0}9) {1}'.format(plabel, p9), file=f_new)
                p10 = config_dict['P{0}10)'.format(plabel)]
                if isinstance(p10[0], list):
                    for li10 in p10:
                        li10[-1] = 1
                print ('P{0}10) {1}'.format(plabel, p10), file=f_new)
                p11 = config_dict['P{0}11)'.format(plabel)]
                if isinstance(p11[0], list):
                    for li11 in p11:
                        li11[-1] = 1
                print ('P{0}11) {1}'.format(plabel, p11), file=f_new)
                p12 = config_dict['P{0}12)'.format(plabel)]
                if isinstance(p12[0], list):
                    for li12 in p12:
                        li12[-1] = 1
                print ('P{0}12) {1}'.format(plabel, p12), file=f_new)
                p13 = config_dict['P{0}13)'.format(plabel)]
                print ('P{0}13) {1}'.format(plabel, p13), file=f_new)
                p14 = config_dict['P{0}14)'.format(plabel)]
                p14[-1] = 1
                print ('P{0}14) {1}'.format(plabel, p14), file=f_new)
                print ('P{0}15) {1}'.format(plabel, config_dict['P{0}15)'.format(plabel)]), file=f_new)
                print ('P{0}16) {1}'.format(plabel, config_dict['P{0}16)'.format(plabel)]), file=f_new)
                print ('P{0}26) {1}'.format(plabel, config_dict['P{0}26)'.format(plabel)]), file=f_new)
                print ('P{0}27) {1}'.format(plabel, SEDmode), file=f_new)
                if SEDmode > 2:
                    # add dust
                    print ('P{0}28) [7.14,4.5,10,0.1,0]'.format(plabel), file=f_new)  
                    print ('P{0}29) [2.0, 0.1, 50, 0.1, 0]'.format(plabel), file=f_new)  
                    print ('P{0}30) [2.5, 0.47, 7.32, 0.1, 0]'.format(plabel), file=f_new)  
                    print ('P{0}31) [2.0, 1.0, 3.0, 0.1, 0]'.format(plabel), file=f_new) 
                    print ('P{0}32) [0.02, 0.01, 1.0, 0.01, 0]'.format(plabel), file=f_new) 
        else:
            print(line, file=f_new, end='')
    f_old.close()
    f_new.close()
    return

def write_data_config(outname, regioninfo, groupinfo, images, bands, cutsizes, convf, magzps = None, varyshift = True, varysky=False, useSED = True, skyrange = 1., lenlabel = 1):
    '''
    #im = [[image, image_ext, mask, mask_ext, sigimg, sigimg_ext, psf, psf_ext, psf_sampling, kerror]...]
    # groupinfo = [['hst', [1,2]], ['sdss', [3,4,5,6]] ...]
    '''
    alphabet = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    if lenlabel == 1:
        allindexes = alphabet
    elif lenlabel == 2:
        allindexes = [a + b for a in alphabet for b in alphabet]
    else:
        raise ValueError('lenlabel must be 1 or 2')
    if len(images) > len(allindexes):
        raise ValueError('Number of images exceeds available labels for lenlabel={0}'.format(lenlabel))
    if len(groupinfo) > len(allindexes):
        raise ValueError('Number of groups exceeds available labels for lenlabel={0}'.format(lenlabel))
    targ = regioninfo['target']
    ra, dec = regioninfo['ra'], regioninfo['dec']
    z_refe = regioninfo['z_ref']
    f = open(outname, 'w')
    print ('# Data config file for {0}, automatically generated by GalfitS'.format(targ), file=f)
    print ('R1) {0}'.format(targ), file=f)
    print ('R2) [{0:.5f},{1:.5f}]'.format(ra, dec), file=f)
    print ('R3) {0:.4f} \n'.format(z_refe), file=f)
    for i, im in enumerate(images):
        index = allindexes[i]
        print ('# Image {0}'.format(index), file=f)
        print ('I{0}1) [{1},{2}]'.format(index, im[0], im[1]), file=f)
        print ('I{0}2) {1}'.format(index, bands[i]), file=f)
        print ('I{0}3) [{1},{2},{3}]'.format(index, im[4], im[5], im[9]), file=f)  # sigma image
        print ('I{0}4) [{1},{2}]'.format(index, im[6], im[7]), file=f)  # PSF image
        print ('I{0}5) {1}'.format(index, im[8]), file=f)  # PSF fine sampling factor
        print ('I{0}6) [{1},{2}]'.format(index, im[2], im[3]), file=f)  # bad pixel mask image
        print ('I{0}7) cR'.format(index), file=f)  # unit of the image
        print ('I{0}8) {1:.4f}'.format(index, cutsizes[i]), file=f)  # size to make cutout image region for fitting, unit arcsec
        print ('I{0}9) {1:.4e}'.format(index, convf[i]), file=f)  # conversion factor from erg/s/cm^2/A to image unit, -1 for default
        if magzps is not None:
            print ('I{0}10) {1:.2f}'.format(index, magzps[i]), file=f)
        else:
            print ('I{0}10) 26.67'.format(index), file=f)
        print ('I{0}11) uniform'.format(index), file=f)  # sky model
        if varysky:
            print ('I{0}12) [[0,-{1},{1},0.1,1]]'.format(index, skyrange), file=f)
        else:
            print ('I{0}12) [[0,-0.5,0.5,0.1,0]]'.format(index), file=f)
        if (varyshift):
            print ('I{0}13) 1'.format(index), file=f)
            print ('I{0}14) [[0,-5,5,0.1,1],[0,-5,5,0.1,1]]'.format(index), file=f)  # [shiftx, shifty]
        else:
            print ('I{0}13) 0'.format(index), file=f)
            print ('I{0}14) [[0,-5,0.5,0.1,0],[0,-5,5,0.1,0]]'.format(index), file=f)  # [shiftx, shifty]
        if useSED:
            print ('I{0}15) 1 \n'.format(index), file=f)
        else:
            print ('I{0}15) 0 \n'.format(index), file=f)
    print ('# Image atlas', file=f)
    for groupi in range(len(groupinfo)):
        groupname = groupinfo[groupi][0]
        imageindexes = groupinfo[groupi][1]
        print ('A{0}1) {1}'.format(allindexes[groupi], groupname), file=f)
        print ("A{0}2) ['{1}']".format(allindexes[groupi], "','".join(allindexes[i] for i in imageindexes)), file=f)
        print ('A{0}3) 0'.format(allindexes[groupi]), file=f)  # whether the images have same pixel size
        print ('A{0}4) 0'.format(allindexes[groupi]), file=f)  # link relative shiftings
        print ('A{0}5) []'.format(allindexes[groupi]), file=f)  # spectra
        print ('A{0}6) []'.format(allindexes[groupi]), file=f)  # aperture size
        print ('A{0}7) [] \n'.format(allindexes[groupi]), file=f)  # references images
    
    f.close()
            
def Fnu_to_Fl(Fnu,lambd):
    """
    Convert flux from F_ν (mJy) to F_λ (erg/s/cm²/Å).
    
    Parameters
    ----------
    Fnu : float or array_like
        Flux density in milliJanskys.
    lambd : float or array_like
        Wavelength in Angstroms.
    
    Returns
    -------
    float or array_like
        Flux in erg/s/cm²/Å.
    
    Notes
    -----
    Conversion: :math:`F_\\lambda = F_\\nu \\times \\frac{c}{\\lambda^2}`
    where c = 2.998×10⁵ km/s.
    """
    #mJy to erg/A 
    c=2.9979246e5
    mJy=1e-26
    a=Fnu*mJy
    Fl=a*(c*1e13)/(lambd**2)
    return Fl

def Fl_to_Fnu(Fl,lambd):
    """
    Convert flux from F_λ (erg/s/cm²/Å) to F_ν (mJy).
    
    Parameters
    ----------
    Fl : float or array_like
        Flux in erg/s/cm²/Å.
    lambd : float or array_like
        Wavelength in Angstroms.
    
    Returns
    -------
    float or array_like
        Flux density in milliJanskys.
    """
    #erg/A to mJy
    c=2.9979246e5
    mJy=1e-26
    a=Fl/(c*1e13)
    Fnu=a*lambd**2/mJy
    return Fnu

def ABmag_to_covf(mzp_AB, wave):
    """
    Convert AB magnitude zeropoint to flux conversion factor.
    
    Parameters
    ----------
    mzp_AB : float
        AB magnitude zeropoint.
    wave : float
        Effective wavelength in Angstroms.
    
    Returns
    -------
    float
        Conversion factor: counts = flux_Flambda × factor.
    """
    flux_mJy = 10**(-0.4*(mzp_AB))*3631*1000
    flux_fl_fuv = Fnu_to_Fl(flux_mJy,wave)
    return 1/flux_fl_fuv

def covf_to_ABmag(covf, wave):
    """
    Convert flux conversion factor to AB magnitude zeropoint.
    
    Parameters
    ----------
    covf : float
        Flux conversion factor.
    wave : float
        Effective wavelength in Angstroms.
    
    Returns
    -------
    float
        AB magnitude zeropoint.
    """
    flux_fl_fuv = 1/covf
    flux_mJy = Fl_to_Fnu(flux_fl_fuv,wave)
    mzp_AB = -2.5*np.log10(flux_mJy/3631/1000) 
    return mzp_AB

def get_L_band(wave, flux, resp):
    """
    Integrate flux through filter response curve.
    
    Computes band-integrated flux by convolving SED with filter transmission.
    
    Parameters
    ----------
    wave : array_like
        SED wavelength grid in Angstroms.
    flux : array_like
        SED flux in F_λ units.
    resp : array_like, shape (N, 2)
        Filter response: [:, 0] = wavelength, [:, 1] = transmission.
    
    Returns
    -------
    float
        Band-integrated flux (same units as input flux).
    
    Notes
    -----
    Formula: :math:`\\langle F \\rangle = \\frac{\\int F(\\lambda) R(\\lambda) d\\lambda}{\\int R(\\lambda) d\\lambda}`
    """
    filter_x = resp[:,0]
    filter_y = resp[:,1]
    flux_L = np.trapz(np.interp(filter_x, wave, flux)*filter_y, filter_x)
    norm = np.trapz(filter_y, filter_x)
    return flux_L/norm

def get_effwave(resp):
    """
    Compute effective wavelength of filter.
    
    Parameters
    ----------
    resp : array_like, shape (N, 2)
        Filter response curve [:, 0] = wavelength, [:, 1] = transmission.
    
    Returns
    -------
    float
        Effective wavelength in Angstroms.
    """
    filter_x = resp[:,0]
    filter_y = resp[:,1]
    effwave = np.trapz(filter_x*filter_y, filter_x)/np.trapz(filter_y, filter_x)
    return effwave

def MAB2Flam(MAB,Lambda):
    """
    Convert AB magnitude to flux density in cgs units of erg/s/cm^3.
    M_{AB}=-2.5\log_{10}f_{\nu}-48.60
    
    Parameters:
    MAB (float): AB magnitude
    Lambda (float): Wavelength in cm

    Returns:
    float: Flux density in cgs units of erg/s/cm^3
    """
    ccgs=constants.c.to(u.cm/u.s).value
    fnu=10**((MAB+48.6)/(-2.5))
    return fnu*ccgs/Lambda**2

def standard_display(Myfitter, workplace, targ, ndisage = False, addtional_scatters=[],wrange=[0.85,1.15],saveimgs=False, sed_multiplot = False):
    # plot multi-band image
    nimage = Myfitter.GSdata.Nimages
    pltwave, Sedcomp, Sedlabel, zred_list = Myfitter.cal_model_image(ndisagn=ndisage,wrange=wrange)
    if (nimage > 0) & (Myfitter.GSdata.fitMode != "photometric - SED"):
        if nimage < 16:
            nrows = nimage
            ncols = 3
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nimage),squeeze=True)
            ax = ax.ravel()
            for i in range(nrows):
                im = Myfitter.GSdata.get_image(Myfitter.GSdata.image_index[i])
                sky_mean, sky_median, sky_std = sigma_clipped_stats(im.cut_image, sigma=3.0, maxiters=5)
                if (im.band == 'galex_nuv') | (im.band == 'galex_fuv'):
                    sky_std = np.std(np.array(im.cut_image,dtype=float)[np.array(im.cut_mask_image,dtype=float)==0])
                sky_median = 0.
                immin = 5*sky_std
                #immax = 500*sky_std
                immax = jnp.nanmax(im.cut_image)
                frac = 0.4
                m = np.array(im.cut_mask_image, dtype=float)
                m_vis = np.ma.masked_where(m == 0, m)   # hide unmasked pixels (0)
                ax[3*i].imshow(normimg(im.cut_image, immin, immax, sky = sky_median,frac=frac), cmap='seismic', origin='lower', vmin=-1, vmax=1,interpolation='nearest')
                ax[3*i+1].imshow(normimg(im.model_image, immin, immax,frac=frac), cmap='seismic', origin='lower', vmin=-1, vmax=1,interpolation='nearest')
                cb = ax[3*i+2].imshow( (im.cut_image-im.model_image)/im.cut_sigma_image, cmap='seismic', origin='lower',vmin=-10, vmax=10,interpolation='nearest')
                ax[3*i].imshow(m_vis, cmap="Blues", origin="lower",alpha=0.5, interpolation="nearest", vmin=0, vmax=1)
                ax[3*i+2].imshow(m_vis, cmap="Blues", origin="lower",alpha=0.5, interpolation="nearest", vmin=0, vmax=1)
                if i == 0:
                    divider = make_axes_locatable(ax[3*i+2])
                    cax = divider.append_axes("top", size="5%", pad=0.05)
                    cbar = plt.colorbar(cb, cax=cax, orientation='horizontal')
                    cbar.ax.tick_params(labelsize=20)
                ny,nx = im.cut_image.shape
                ax[3*i+0].text(0.03*nx,0.06*ny, "{0}".format(Myfitter.GSdata.allbands[i]), size = 25, color = 'k', weight = "light" )
                if i == 0:
                    ax[0].text(0.03*nx,0.85*ny, "data", size = 25, color = 'k', weight = "light" )
                    ax[1].text(0.03*nx,0.85*ny, "model", size = 25, color = 'k', weight = "light" )
                    ax[2].text(0.03*nx,0.85*ny, "residual/error", size = 25, color = 'k', weight = "light" )

                # add compass
                size1 = np.sqrt(nx**2+ny**2)
                x0 = 0.8*nx
                y0 = 0.8*ny
                L_arrow = 0.12
                L = L_arrow*size1
                skydire = im.skydirection
                vE = skydire[0]
                vN = skydire[1]
                xN, yN = x0 + L * vN[0], y0 + L * vN[1]
                xE, yE = x0 + L * vE[0], y0 + L * vE[1]

                xN_text = xN - 0.03*size1*vE[0]
                yN_text = yN - 0.03*size1*vE[1]
                xE_text = xE + 0.03*size1*vN[0]
                yE_text = yE + 0.03*size1*vN[1]
                # Draw arrows
                ax[3*i].annotate(
                    "", xy=(xN, yN), xytext=(x0+0.06*(x0-xN), y0+0.06*(y0-yN)),
                    arrowprops=dict(arrowstyle='-|>', lw=1.5, color='k')
                )
                ax[3*i].text(xN_text, yN_text, "N", color='k', ha='center', va='bottom')

                ax[3*i].annotate(
                    "", xy=(xE, yE), xytext=(x0+0.06*(x0-xE), y0+0.06*(y0-yE)),
                    arrowprops=dict(arrowstyle='-|>', lw=1.5, color='k')
                )
                ax[3*i].text(xE_text, yE_text, "E", color='k', ha='center', va='bottom')

                if saveimgs:
                    outname = workplace + '/{0}_{1}_result.fits'.format(targ,Myfitter.GSdata.allbands[i])
                    im.write_imcut(None, None, outname, addmodel = True)
        else:
            ngrid = int(np.ceil(np.sqrt(nimage)))
            nrows = ngrid
            ncols = 3 * ngrid
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows), squeeze=False)
            for i in range(nimage):
                row = i // ngrid
                col = i % ngrid
                ax_data = ax[row, col]
                ax_model = ax[row, col + ngrid]
                ax_res = ax[row, col + 2*ngrid]

                im = Myfitter.GSdata.get_image(Myfitter.GSdata.image_index[i])
                sky_mean, sky_median, sky_std = sigma_clipped_stats(im.cut_image, sigma=3.0, maxiters=5)
                if (im.band == 'galex_nuv') | (im.band == 'galex_fuv'):
                    sky_std = np.std(np.array(im.cut_image,dtype=float)[np.array(im.cut_mask_image,dtype=float)==0])
                sky_median = 0.
                immin = 5*sky_std
                #immax = 500*sky_std
                immax = jnp.nanmax(im.cut_image)
                frac = 0.4
                m = np.array(im.cut_mask_image, dtype=float)
                m_vis = np.ma.masked_where(m == 0, m)   # hide unmasked pixels (0)

                ax_data.imshow(normimg(im.cut_image, immin, immax, sky = sky_median, frac=frac), cmap='seismic', origin='lower', vmin=-1, vmax=1, interpolation='nearest')
                ax_model.imshow(normimg(im.model_image, immin, immax, frac=frac), cmap='seismic', origin='lower', vmin=-1, vmax=1, interpolation='nearest')
                cb = ax_res.imshow((im.cut_image-im.model_image)/im.cut_sigma_image, cmap='seismic', origin='lower', vmin=-10, vmax=10, interpolation='nearest')
                ax_data.imshow(m_vis, cmap="Blues", origin="lower", alpha=0.5, interpolation="nearest", vmin=0, vmax=1)
                ax_res.imshow(m_vis, cmap="Blues", origin="lower", alpha=0.5, interpolation="nearest", vmin=0, vmax=1)

                if i == 0:
                    divider = make_axes_locatable(ax_res)
                    cax = divider.append_axes("top", size="5%", pad=0.05)
                    cbar = plt.colorbar(cb, cax=cax, orientation='horizontal')
                    cbar.ax.tick_params(labelsize=20)

                ny,nx = im.cut_image.shape
                ax_data.text(0.03*nx, 0.06*ny, "{0}".format(Myfitter.GSdata.allbands[i]), size=20, color='k', weight="light")
                if i == 0:
                    ax_data.text(0.03*nx, 0.85*ny, "data", size=20, color='k', weight="light")
                    ax_model.text(0.03*nx, 0.85*ny, "model", size=20, color='k', weight="light")
                    ax_res.text(0.03*nx, 0.85*ny, "residual/error", size=20, color='k', weight="light")

                # add compass
                size1 = np.sqrt(nx**2+ny**2)
                x0 = 0.8*nx
                y0 = 0.8*ny
                L_arrow = 0.12
                L = L_arrow*size1
                skydire = im.skydirection
                vE = skydire[0]
                vN = skydire[1]
                xN, yN = x0 + L * vN[0], y0 + L * vN[1]
                xE, yE = x0 + L * vE[0], y0 + L * vE[1]

                xN_text = xN - 0.03*size1*vE[0]
                yN_text = yN - 0.03*size1*vE[1]
                xE_text = xE + 0.03*size1*vN[0]
                yE_text = yE + 0.03*size1*vN[1]
                ax_data.annotate(
                    "", xy=(xN, yN), xytext=(x0+0.06*(x0-xN), y0+0.06*(y0-yN)),
                    arrowprops=dict(arrowstyle='-|>', lw=1.5, color='k')
                )
                ax_data.text(xN_text, yN_text, "N", color='k', ha='center', va='bottom')

                ax_data.annotate(
                    "", xy=(xE, yE), xytext=(x0+0.06*(x0-xE), y0+0.06*(y0-yE)),
                    arrowprops=dict(arrowstyle='-|>', lw=1.5, color='k')
                )
                ax_data.text(xE_text, yE_text, "E", color='k', ha='center', va='bottom')

                if saveimgs:
                    outname = workplace + '/{0}_{1}_result.fits'.format(targ,Myfitter.GSdata.allbands[i])
                    im.write_imcut(None, None, outname, addmodel = True)

            for i in range(nimage, ngrid*ngrid):
                row = i // ngrid
                col = i % ngrid
                ax[row, col].set_visible(False)
                ax[row, col + ngrid].set_visible(False)
                ax[row, col + 2*ngrid].set_visible(False)
         
        plt.savefig(workplace+'/{0}image_fit.png'.format(targ),dpi=200.,bbox_inches='tight')
        plt.close()

    ngimage = len(Myfitter.GSdata.allgrismbands)
    if (ngimage > 0) & (Myfitter.GSdata.fitMode == "grism - SED"):
        nrows = ngimage
        ncols = 3
        im0 = Myfitter.GSdata.get_grism_image(0)
        shape0 = im0.cut_image.shape
        axisratio = shape0[0]/shape0[1]
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*ngimage*axisratio),squeeze=True)
        ax = ax.ravel()
        for i in range(nrows):
            im = Myfitter.GSdata.get_grism_image(i)
            sky_mean, sky_median, sky_std = sigma_clipped_stats(im.cut_image, sigma=3.0, maxiters=5)
            sky_median = 0.
            immin = 5*sky_std
            #immax = 500*sky_std
            immax = jnp.nanmax(im.cut_image)
            frac = 0.4
            ax[3*i].imshow(normimg(im.cut_image, immin, immax, sky = sky_median,frac=frac), cmap='seismic', origin='lower', vmin=-1, vmax=1,interpolation='nearest')
            ax[3*i+1].imshow(normimg(im.model_image, immin, immax,frac=frac), cmap='seismic', origin='lower', vmin=-1, vmax=1,interpolation='nearest')
            cb = ax[3*i+2].imshow( (im.cut_image-im.model_image)/im.cut_sigma_image, cmap='seismic', origin='lower',vmin=-10, vmax=10,interpolation='nearest')
            ax[3*i].imshow(np.array(im.cut_mask_image,dtype=float), cmap='Blues', origin='lower', alpha=0.5*np.array(im.cut_mask_image,dtype=float))
            ax[3*i+2].imshow(np.array(im.cut_mask_image,dtype=float), cmap='Blues', origin='lower', alpha=0.5*np.array(im.cut_mask_image,dtype=float))
            if i == 0:
                divider = make_axes_locatable(ax[3*i+2])
                cax = divider.append_axes("top", size="5%", pad=0.05)
                cbar = plt.colorbar(cb, cax=cax, orientation='horizontal')
                cbar.ax.tick_params(labelsize=20)
            ny,nx = im.cut_image.shape
            ax[3*i+0].text(3,0.85*ny, "{0}".format(Myfitter.GSdata.allgrismbands[i]), size = 25, color = 'r', weight = "light" )
            ax[0].text(3,3, "data", size = 25, color = 'b', weight = "light" )
            ax[1].text(3,3, "model", size = 25, color = 'b', weight = "light" )
            ax[2].text(3,3, "residual/error", size = 25, color = 'b', weight = "light" )
            if saveimgs:
                outname = workplace + '/{0}_{1}_result.fits'.format(targ,Myfitter.GSdata.allgrismbands[i])
                im.write_imcut(None, None, outname, addmodel = True)
        plt.savefig(workplace+'/{0}grism_fit.png'.format(targ),dpi=200.,bbox_inches='tight')
        plt.close()


    if not Myfitter.GSdata.fitMode == "single image - photometry":
        if pltwave is not None:
            # plot SED model
            pltwave[0] = np.array(pltwave[0])
            pltwave[1] = np.array(pltwave[1])
            pltwave[3] = np.array(pltwave[3])
            # waverange = [np.min(pltwave[0]),np.max(pltwave[0])]
            # sortwave   = Myfitter.GSdata.sortwave # ## we should fix this as quickly as possible.
            # waverange   = [sortwave[0], sortwave[-1]]
            wavegrid    = Myfitter.GSdata.wavegrid
            waverange   = [wavegrid[0], wavegrid[-1]]

            if not sed_multiplot: # plot single sed figure for all profile.
                fig = plt.figure(figsize=(9,6))
                ax = plt.gca()
                plt.xlim([wrange[0]*waverange[0],wrange[1]*waverange[1]])
                plt.errorbar(pltwave[0],pltwave[1]*pltwave[0],yerr=pltwave[2]*pltwave[0],fmt='o',color='k',label=r'$L_\mathrm{tot}$ cut image',alpha=0.7)
                plt.errorbar(pltwave[0],pltwave[3]*pltwave[0],fmt='o',color='r',label=r'$L_\mathrm{tot}$ model image',alpha=0.7)
                sedtot = np.zeros(len(wavegrid))
                # plot spectra is any
                for image_atlas in Myfitter.GSdata.image_atlas_list:
                    if len(image_atlas.spectra) > 0:
                        for loop, spec in enumerate(image_atlas.spectra):
                            spec = image_atlas.spectra[loop]
                            plt.step(spec[0], spec[1]*spec[0], 'k-', label='{0} {1}th spec'.format(image_atlas.name, loop), linewidth=2, alpha=0.5)
                for loop in range(len(Sedcomp)):
                    plt.plot(wavegrid,wavegrid*Sedcomp[loop],label=Sedlabel[loop],lw=2,alpha=0.7)
                    sedtot += Sedcomp[loop]
                for adds in addtional_scatters:
                    if len(adds) == 5:
                        plt.errorbar(adds[0], adds[1],yerr=adds[2], fmt='o',color=adds[3],label=adds[4],alpha=0.7)
                plt.plot(wavegrid,wavegrid*sedtot,label='sed_total',lw=2,alpha=0.7,linestyle='--')  
                plt.xlabel(r'Observed Wavelength ($\mathring{\rm A}$)', fontsize=19)
                plt.ylabel(r'$\lambda L_\lambda \; (10^{38} \, \rm erg\,s^{-1})$', fontsize=19)
                plt.ylim([0.33*np.min(pltwave[3]*pltwave[0]),3*np.max(pltwave[3]*pltwave[0])])

                if np.max(pltwave[0])/np.min(pltwave[0]) > 10:
                    plt.xscale('log')
                plt.yscale('log')
                plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12, numpoints=1, frameon=False)


                # make a upper x-axis labeled by rest wavelength
                ax2 = plt.twiny()
                ax2.set_xlim([wrange[0]*waverange[0],wrange[1]*waverange[1]])
                if np.max(pltwave[0])/np.min(pltwave[0]) > 10:
                    ax2.set_xscale('log')
                ax2.set_xticks(pltwave[0])
                ax2.set_xticklabels(np.array(pltwave[0]/(1.+zred_list[0]),dtype=np.int32))
                ax2.set_xlabel(r'Rest Wavelength ($\mathring{\rm A}$)', fontsize=19)
                plt.savefig(workplace + '/{0}SED_model.png'.format(targ),dpi=200,bbox_inches='tight')
                plt.close()
            else: ## plot sed figure for each sources in isolated region.
                pass
            #     sedp_name = np.array(["Stellar_att", "SteNeb_att", "duste"])
            #     for loop in range(len(Myfitter.model_list)):
            #         band_list = Myfitter.GSdata.allbands
            #         fluxphots_iso, modelphots_iso, errphots_iso, sedscale_ratios = [], [], [], []            
            #         plt_filter = Myfitter.GSdata.allbands[0]
                    
            #         if Myfitter.mtype_list[loop] == "galaxy":
            #             gal_model = Myfitter.model_list[loop]
            #             for band in band_list:
                            
            #                 im = Myfitter.GSdata.get_image(band)
            #                 gal_img, _ = load_gal_image(Myfitter, band, gal_model.name) # too
            #                 ## iso flux
            #                 cut_img = im.cut_image
            #                 err_img = im.cut_sigma_image
            #                 mask_img = im.cut_mask_image
            #                 segm = isophot_segm(gal_img, mask = mask_img)

            #                 if band == plt_filter:
            #                     plt_cut = im.cut_image
            #                     plt_seg = segm.data
            #                 fluxphot_iso = np.sum(cut_img[segm.data > 0]) / im.phys_to_counts_rate
            #                 modelphot_iso = np.sum(gal_img[segm.data > 0]) / im.phys_to_counts_rate
            #                 errphot_iso = np.sqrt(np.sum(err_img[segm.data>0] ** 2)) / im.phys_to_counts_rate

            #                 fluxphots_iso.append(fluxphot_iso)
            #                 modelphots_iso.append(modelphot_iso)
            #                 errphots_iso.append(errphot_iso)
            #                 sedscale_ratios.append(modelphot_iso / (np.sum(gal_img)/im.phys_to_counts_rate))
            #             fluxphots_iso = np.array(fluxphots_iso)
            #             modelphots_iso = np.array(modelphots_iso)
            #             errphots_iso = np.array(errphots_iso)

            #             # sed 
            #             sedscale_ratio = np.mean(sedscale_ratios)
            #             gal_name = gal_model.name
            #             sed_total = np.zeros(len(wavegrid))
            #             for subname in gal_model.subnames:
            #                 sedmode_array = np.array(gal_model.sedmodelist[subname], dtype = bool)
            #                 sedmode_name = sedp_name[sedmode_array]
            #                 sedlabel = [f"{subname}_{mode}" for mode in sedmode_name]
            #                 sedlabel_idx = [Sedlabel.index(sedlabel_el) for sedlabel_el in sedlabel]
            #             ## plot
            #             fig = plt.figure(figsize=(9,6))
            #             ax = plt.gca()
            #             plt.xlim([wrange[0]*waverange[0],wrange[1]*waverange[1]])
            #             # flux photometry
            #             plt.errorbar(pltwave[0],fluxphots_iso*pltwave[0],yerr=errphots_iso*pltwave[0],fmt='o',color='k',label=r'$L_\mathrm{tot}$ cut image',alpha=0.7)
            #             plt.errorbar(pltwave[0],modelphots_iso*pltwave[0],fmt='o',color='r',label=r'$L_\mathrm{tot}$ model image',alpha=0.7)
            #             for sedlabel_idx_el in sedlabel_idx:
            #                 sed = Sedcomp[sedlabel_idx_el] * sedscale_ratio
            #                 sed_total += sed
            #                 plt.plot(wavegrid,wavegrid*sed,label=Sedlabel[sedlabel_idx_el],lw=2,alpha=0.7)
            #             plt.plot(wavegrid,wavegrid*sed_total,label='sed_total',lw=2,alpha=0.7,linestyle='--')  
            #             plt.xlabel(r'Observed Wavelength ($\mathring{\rm A}$)', fontsize=19)
            #             plt.ylabel(r'$\lambda L_\lambda \; (10^{38} \, \rm erg\,s^{-1})$', fontsize=19)
            #             ylim_min = min(np.min(modelphots_iso*pltwave[0]), np.min(fluxphots_iso*pltwave[0]))
            #             ylim_max = max(np.max(modelphots_iso*pltwave[0]), np.max(fluxphots_iso*pltwave[0]))
            #             plt.ylim([0.33*ylim_min,3*ylim_max])
            #             plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12, numpoints=1, frameon=False)
            #             plt.yscale("log")

                        
            #             ## upper axis
            #             ax2 = plt.twiny()
            #             ax2.set_xlim([wrange[0]*waverange[0],wrange[1]*waverange[1]])
            #             if np.max(pltwave[0])/np.min(pltwave[0]) > 10:
            #                 ax2.set_xscale('log')
            #             ax2.set_xticks(pltwave[0])
            #             ax2.set_xticklabels(np.array(pltwave[0]/(1.+gal_model.redshift),dtype=np.int32))
            #             ax2.set_xlabel(r'Rest Wavelength ($\mathring{\rm A}$)', fontsize=19)

            #             mean, median, std = sigma_clipped_stats(plt_cut, sigma = 3)
            #             immin = 5*std
            #             immax = jnp.nanmax(plt_cut)
            #             ax_insert = inset_axes(ax, width=1.5, height=1.2, loc='center left',bbox_to_anchor=(1.0, 0.8), bbox_transform=ax.transAxes)
            #             ax_insert.imshow(np.flipud(normimg(plt_cut, min=immin, max=immax, sky=0, frac=0.5)),
            #                             cmap='seismic', origin='lower', vmin=-1, vmax=1,
            #                             interpolation='nearest')

            #             ax_insert.imshow(np.flipud(plt_seg), alpha = 0.3)
            #             ax_insert.set_title(plt_filter, fontsize = 10)
            #             ax_insert.set_xticks([])
            #             ax_insert.set_yticks([])
            #             plt.savefig(workplace + '/{0}_{1}SED_model.png'.format(targ,gal_name),dpi=200,bbox_inches='tight')
                        # pass/
                    # else: # for star ...
                    #     print(f"don't support now.")
        # plot spectra
        else:
            for image_atlas in Myfitter.GSdata.image_atlas_list:
                count = 0
                if len(image_atlas.spectra) > 0:
                    for loop, spec in enumerate(image_atlas.spectra):
                        spec = image_atlas.spectra[loop]
                        fig = plt.figure(figsize=(12, 12))
                        frame1=fig.add_axes((.15,.3,.8,.65)) # left, bottom, width, height
                        plt.step(spec[0], spec[1], 'k-', label='data', linewidth=2, alpha=0.5)
                        mospec = image_atlas.modelspec[loop]
                        for loop2 in range(len(Sedcomp[count])):  
                                plt.plot(spec[0],Sedcomp[count][loop2],label = Sedlabel[count][loop2],linewidth=1.5,alpha = 0.7)
                        plt.plot(spec[0],mospec,label='Model',linewidth=2,c='r',alpha = 0.7)
                        frame1.minorticks_on()
                        frame1.tick_params(axis="y", which="major", length=8, labelsize=18, width=1.)
                        frame1.tick_params(axis="y", which="minor", length=5, labelsize=18, width=1.)
                        frame1.tick_params(axis="x", labelsize=0.)
                        plt.ylabel(r'$L_\lambda \;( 10^{38} \rm erg \, s^{-1} \, \mathring{A}^{-1}) $',fontsize=24)
                        plt.legend(loc='upper right',fontsize=14,numpoints=1)
                        yliml=-0.25*np.max(spec[1])
                        ylimu=1.5*np.max(spec[1])
                        plt.ylim([yliml,ylimu])
                        plt.xlim([wrange[0]*np.min(spec[0]),wrange[1]*np.max(spec[0])])
                        tickwave = np.linspace(wrange[0]*np.min(spec[0]),wrange[1]*np.max(spec[0]),6)
                        ax2 = plt.twiny()
                        ax2.set_xlim([wrange[0]*np.min(spec[0]),wrange[1]*np.max(spec[0])])
                        ax2.set_xticks(tickwave)
                        ax2.set_xticklabels(np.int32(np.round(tickwave/(1.+zred_list[0]),0)))
                        ax2.set_xlabel(r'Rest Wavelength ($\mathring{A}$)', fontsize=24)
                        ax2.tick_params(axis="both", which="major", length=8, labelsize=18, width=1.)
                        ax2.tick_params(axis="both", which="minor", length=5, labelsize=18, width=1.)
                        resid=spec[1]-mospec
                        frame2=fig.add_axes((.15,.1,.8,.2))
                        frame2.minorticks_on()
                        frame2.tick_params(axis="both", which="major", length=8, labelsize=18, width=1.)
                        frame2.tick_params(axis="both", which="minor", length=5, labelsize=18, width=1.)
                        plt.errorbar(spec[0],resid,marker='.',capsize=2,capthick=1,color='k',alpha=0.5)
                        plt.axhline(y=0., linestyle='--', color='r', label='',linewidth=3.)
                        plt.xlabel(r'Observed Wavelength ($\mathring{A}$)', fontsize=24)
                        ylim=1.5*np.max(resid)
                        plt.ylim([-ylim,ylim])
                        plt.xlim([wrange[0]*np.min(spec[0]),wrange[1]*np.max(spec[0])])
                        plt.savefig(workplace+'/{0}spec_{1}.png'.format(targ,image_atlas.specprefix[loop]),dpi=200.,bbox_inches='tight')
                        plt.close()
                        count += 1
    return

def photometry_to_img(flux, flux_err, z, outputname,band=None,effectivewave=None, unit = 'mJy'):
    '''
    converting a photometry point to a image with a given band
    first, convert the flux to a Luminosity, in unit of 1e38erg/s/A
    unit: mJy or flambda, or MagAB
    '''
    data = np.ones((1,1))
    if effectivewave is not None:
        wave = effectivewave
    else:
        wave = effective_wave[band]
    cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.308)
    d=cosmo.luminosity_distance(z)/(1+z)
    dc=d.to(u.cm)
    dis=dc.value
    C_unit=1e38/(4*np.pi*dis**2)
    #C_unit=1
    if unit == 'mJy':
        flux = Fnu_to_Fl(flux,wave)
        flux_err = Fnu_to_Fl(flux_err,wave)
    elif unit == 'MagAB':
        flux = 1/ABmag_to_covf(flux, wave)
        flux_err = flux*flux_err*np.log(10)/2.5
    elif unit == 'flambda':
        flux = flux
        flux_err = flux_err
    L = flux/C_unit
    L_err = flux_err/C_unit
    if unit == 'L38':
        L = flux
        L_err = flux_err
    hdu_temp = fits.PrimaryHDU(np.float32(L*data), header = None)
    hdu_temp.header['band'] = band
    hdu_temp.header['photop'] = 1
    hdu_temp2 = fits.ImageHDU(0*np.float32(data))
    hdu_temp3 = fits.ImageHDU(np.float32(L_err*data))
    hdu_temp4 = fits.ImageHDU(np.float32(data))
    hdul = fits.HDUList([hdu_temp,hdu_temp2,hdu_temp3,hdu_temp4])
    hdul.writeto(outputname, overwrite=True)
    return 

def get_line_flux(peak, width):
    return peak*width*np.sqrt(2*np.pi)


def z_to_agebin(z_fit, n_agebin=7):


    '''
    Follow prospector beta method to generate age bin
    '''
    amin1, amin2    = 0.03, 0.10 # Gyr
    t_univ  = float(cosmo.age(z_fit).value)
    tbin_max    = 0.9 * t_univ
    if z_fit <= 3.:
        age_bin     = [0, amin1] + np.logspace(np.log10(amin2), np.log10(tbin_max), n_agebin-2).tolist() + [t_univ] 
    else:
        # age_bin     = [0, 0.01347] + np.logspace(np.log10(0.01347), np.log10(tbin_max), n_agebin-1).tolist()[1:] + [t_univ]
        age_bin     = np.logspace(np.log10(amin1), np.log10(tbin_max), n_agebin).tolist() + [t_univ]
        age_bin[0]  = 0
    return age_bin
