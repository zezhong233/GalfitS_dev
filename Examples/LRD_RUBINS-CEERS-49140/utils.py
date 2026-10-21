
import numpy as np
from astropy.io import fits, ascii
from astropy.stats import sigma_clipped_stats
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants  as const
c = const.c.to("AA/s").value
cosmo   = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.308)

def parse_gs_params(gssm_path, targ_params):
    gssm_dict   = ascii.read(gssm_path)
    gssm_parlist    = list(gssm_dict["pname"].value)
    value_list  = []

    for targ_p in targ_params:
        assert targ_p in gssm_parlist, f"{targ_p} not in galfits params file."
        value_index     = gssm_parlist.index(targ_p)
        value_temp      = gssm_dict["best_value"][value_index]
        value_list.append(value_temp)
    value_list  = np.array(value_list)
    return value_list


def Ll38_to_flam(Ll38, z):
    dc      = cosmo.comoving_distance(z).to(u.cm).value
    C_unit  = 1e38 / (4*np.pi*dc**2)
    flam      = Ll38 * C_unit
    return flam


def fconts_to_SFH(f_conts_list, age_list, logM):

    f_conts_list = np.asarray(f_conts_list)  
    f_reals = 10.0**f_conts_list
    f_reals = f_reals / np.sum(f_reals)

    sfh_list = []
    for i in range(len(f_reals)):
        sfh_list.append(f_reals[i] * 10.0**logM / (1e9 * (age_list[i+1] - age_list[i])))

    sfh_list    = np.array(sfh_list)
    return sfh_list