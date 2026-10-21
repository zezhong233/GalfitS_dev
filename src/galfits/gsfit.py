import os
from . import gsutils
from . import sed_interp as SED
from .sed_interp import _find_data_path
from . import galaxy as GALA
import numpy as np
import jax.numpy as jnp
import dynesty
import jax
from jax import random, grad, jit, vmap, lax
from jax.scipy.interpolate import RegularGridInterpolator
import jaxopt
from functools import partial
import jax.random as jrandom
import lmfit
import multiprocessing
from . import profiles as pro
import importlib
#import blackjax
#import numpyro
#import numpyro.distributions as dist
#from numpyro.infer import HMC, HMCECS, MCMC, NUTS, SA, SVI, Trace_ELBO, init_to_value,init_to_uniform
from jnesty import NestedSampler as JNestySampler
from jnesty import DynamicNestedSampler as JNestyDynamicSampler
from jnesty import plotting as jnesty_plotting
from jnesty import save_results as jnesty_save_results
import tqdm
import optax
from evosax import algorithms
import dynesty.utils as dyutils

jax.config.update("jax_enable_x64", False)

_template_path   = _find_data_path()

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def soft_bound(x, bound, steepness):
    return 1 / (1 + jnp.exp(-steepness * (x - bound)))

def logit(p):
    return jnp.log(p / (1 - p))

def gradient_clipping(gradients, max_norm):
    total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        gradients = [g * clip_coef for g in gradients]
    return gradients

def _detect_devices():
    devs = jax.devices()
    platforms = {d.platform for d in devs}
    if "gpu" in platforms:
        return [d for d in devs if d.platform == "gpu"], "gpu"
    if "cpu" in platforms:
        return [d for d in devs if d.platform == "cpu"], "cpu"
    return devs, devs[0].platform

def _load_npz(filename, points_name, y_name):
    '''
    load npz file into interpolater
    Be cautious for the sequence of points_name.
    '''
    data    = jnp.load(filename)
    points  = tuple(data[p_name] for p_name in points_name)
    interp  = RegularGridInterpolator(points, data[y_name], bounds_error=False,fill_value=0.)
    return interp

class gsfitter(object):
    """
    Multi-band SED and image fitting engine for galaxies, AGN, and stars.
    
    The gsfitter class provides a unified framework for fitting galaxy models to
    multi-wavelength photometric and spectroscopic data. It supports simultaneous
    fitting of structural parameters, stellar populations, and nebular emission using
    various optimization and sampling algorithms including gradient descent, nested
    sampling, and MCMC.
    
    Parameters
    ----------
    GSdata : GSData object, optional
        Multi-band imaging and spectroscopic data to fit. Should contain photometric
        images, error maps, PSFs, and optionally spectra. Default is None.
    model_list : list, optional
        List of model objects (Galaxy, AGN, Star) to fit to the data.
        Default is [].
    mtype_list : list of str, optional
        Type identifier for each model ('galaxy', 'agn', 'star'). Must have same
        length as model_list. Default is [].
    lmParameters : lmfit.Parameters, optional
        Parameter object defining all fit parameters with bounds, initial values,
        and constraints. Default is None.
    g_prior : list of tuples, optional
        Gaussian priors on parameters. Each tuple: (param_name, mean, std).
        Default is [].
    wight_spec : float, optional
        Relative weight for spectroscopic data vs. photometric data in chi-squared.
        Default is 1.0.
    
    Attributes
    ----------
    GSdata : GSData
        Data object containing observations.
    model_list : list
        All model objects being fit.
    gmodel_list : list
        Galaxy model objects only.
    psfmodel_list : list
        Point source (AGN, star) model objects.
    lmParameters : lmfit.Parameters
        Parameter definitions with bounds and constraints.
    theta0 : array_like
        Initial parameter values for free parameters.
    ndim : int
        Number of free parameters.
    varnames : list
        Names of free parameters.
    pardict : dict
        Dictionary of all parameter values.
    gprior : array_like
        Boolean mask indicating which parameters have Gaussian priors.
    
    Methods
    -------
    lnprior(theta)
        Compute log-prior probability.
    lnlike(theta)
        Compute log-likelihood given data.
    lnprob(theta)
        Compute log-posterior probability.
    run_lmfit(**kwargs)
        Perform least-squares optimization using lmfit.
    run_dynesty(**kwargs)
        Perform nested sampling using dynesty.
    run_jaxopt(**kwargs)
        Perform gradient-based optimization using JAX.
    
    Notes
    -----
    The fitting process:
    
    1. **Forward Model**: For given parameters:
       
       - Generate mass maps for each galaxy component
       - Compute SEDs for each component
       - Create model images in each band
       - Convolve with PSFs
       - Generate model spectra if available
    
    2. **Likelihood**: Compare to data:
       
       .. math::
           \\chi^2 = \\sum_{bands} \\frac{(I_{data} - I_{model})^2}{\\sigma^2}
                   + w_{spec} \\sum_{\\lambda} \\frac{(F_{data} - F_{model})^2}{\\sigma_{spec}^2}
    
    3. **Optimization/Sampling**: Find best-fit or posterior distribution
    
    The class uses JAX for automatic differentiation, enabling efficient gradient-based
    optimization. It supports:
    
    - Multi-component galaxy decomposition (bulge+disk+bar)
    - Simultaneous galaxy + AGN fitting
    - Spatially resolved stellar population gradients
    - Nebular emission modeling
    - Energy balance dust emission
    
    Examples
    --------
    >>> from galfits import gsfitter, GSData, Galaxy, AGN
    >>> # Setup data
    >>> data = GSData()
    >>> data.add_image('sdss_r', image, error, psf)
    >>> # Setup models
    >>> gal = Galaxy(z=0.1, name='gal')
    >>> gal.add_subC('bulge', logM=10.5, ...)
    >>> agn = AGN(z=0.1, name='agn')
    >>> # Setup fitter
    >>> fitter = gsfitter(GSdata=data, 
    ...                   model_list=[gal, agn],
    ...                   mtype_list=['galaxy', 'agn'],
    ...                   lmParameters=params)
    >>> # Run fit
    >>> result = fitter.run_lmfit(method='least_squares')
    
    See Also
    --------
    GSData : Data container for multi-band observations.
    Galaxy : Galaxy model class.
    AGN : AGN model class.
    """
    def __init__(self, GSdata=None, model_list=[], mtype_list=[], lmParameters=None, g_prior=[], f_tprior_info = None,wight_spec = 1.0):
        """
        Initialize the fitter with data, models, and parameters.
        
        Parameters
        ----------
        GSdata : GSData, optional
            Observation data object.
        model_list : list, optional
            List of model objects to fit.
        mtype_list : list of str, optional
            Model types ('galaxy', 'agn', 'star').
        lmParameters : lmfit.Parameters, optional
            Parameter definitions.
        g_prior : list of tuples, optional
            Gaussian priors [(name, mean, std), ...].
        wight_spec : float, optional
            Spectroscopic data weight.
        """
        self.GSdata = GSdata
        self.model_list = model_list
        self.mtype_list = mtype_list
        self.gmodel_list = []
        self.psfmodel_list = []
        for loop,model in enumerate(self.model_list):
            if mtype_list[loop] == 'galaxy':
                self.gmodel_list.append(model)
            elif mtype_list[loop] == 'agn':
                self.psfmodel_list.append(model)
            elif mtype_list[loop] == 'star':
                self.psfmodel_list.append(model)
        self.lmParameters = lmParameters
        self.weight_spec = wight_spec
        self.loglike_norm = jnp.array(0.,dtype=jnp.float32)
        theta0 = []
        bounds = []
        gbounds = []
        self.varnames=[] # self.lmparameters -> value_dict() -> self.pardict -> vary? -> self.varnames
        mnames = [] # model name
        for model in self.model_list:
            mnames.append(model.name)
        self.mnames = mnames
        if lmParameters is not None:
            self.pardict = self.lmParameters.valuesdict()
            for el in self.pardict:
                if self.lmParameters[el].expr is not None:
                    self.lmParameters[el].vary = False
                if self.lmParameters[el].vary:
                    theta0.append(self.pardict[el])
                    self.varnames.append(el)
                    lb, ub = self.lmParameters[el].min, self.lmParameters[el].max
                    bounds.append(jnp.array([lb, ub]))
                    gbounds.append(jnp.array([0., 1.]))
            self.theta0 = jnp.array(theta0)
            self.ndim = len(self.varnames)
            self.gprior = jnp.zeros(self.ndim,dtype=jnp.bool_)
            for loop, el in enumerate(self.varnames):
                for gpp in g_prior:
                    if el == gpp[0]:
                        gbounds[loop] = gbounds[loop].at[0].set(gpp[1])
                        gbounds[loop] = gbounds[loop].at[1].set(gpp[2])
                        self.gprior = self.gprior.at[loop].set(True)
            self.bounds = jnp.array(bounds)
            self.gbounds = jnp.array(gbounds)
        else:
            self.ndim = 0

        ## used for cosmic sfr calculation.
        self.np_agedict     = {} ## agelist of profiles with non-parametric SFH setting
        self.np_z_logMidx_dict  = {} ## profile redshift `value` and logM `index` in varname
        if f_tprior_info is not None:
            for gal_model in self.gmodel_list:
                SFH_type_dict   = gal_model.SFH_list
                p_keys    = SFH_type_dict.keys()
                for p_key in p_keys:
                    if SFH_type_dict[p_key] == "bins":
                        logM_n  = f"logM_{p_key}"
                        logM_i  = self.varnames.index(logM_n) ## TODO: I will fix the bug if logM is not variable.
                        zred    = gal_model.redshift
                        np_age  = gal_model.npSFH_times[p_key]
                        self.np_agedict[p_key]  = np_age
                        self.np_z_logMidx_dict[p_key]   = [logM_i, zred]

        self.UC_func = None
        self.fitresult = None
        self.mcmc_samples = None
        self.dof = 0
        for image_atlas in self.GSdata.image_atlas_list:
            for loop in range(len(image_atlas.band_list)):
                im = image_atlas[loop]
                self.dof += jnp.sum(jnp.where(im.cut_mask_image,0.,1.)) #？？？？ 这个对吗？全部image pixel 都是自由参数
                image_atlas.img_dof[loop] = jnp.sum(jnp.where(im.cut_mask_image,0.,1.))
                self.loglike_norm += 0.5*jnp.sum((1.-im.cut_mask_image)*jnp.log(2*jnp.pi*im.cut_sigma_image*im.cut_sigma_image))
            for loop in range(len(image_atlas.grism_band_list)):
                gim = image_atlas.grism_image_list[loop]
                self.dof += jnp.sum(jnp.where(gim.cut_mask_image,0.,1.))
                image_atlas.grism_dof[loop] = jnp.sum(jnp.where(gim.cut_mask_image,0.,1.))
                self.loglike_norm += 0.5*jnp.sum((1.-gim.cut_mask_image)*jnp.log(2*jnp.pi*gim.cut_sigma_image*gim.cut_sigma_image))
            if len(image_atlas.spectra) > 0:
                for loop, spec in enumerate(image_atlas.spectra):
                    data1D_De = spec[1]
                    self.dof += len(data1D_De)
                    image_atlas.spec_dof[loop] = len(data1D_De)  
                    self.loglike_norm += 0.5*jnp.sum(jnp.log(2*jnp.pi*spec[2]*spec[2]))
        self.dof -= self.ndim
        self.chisq = None

        self._init_ftp_params(f_tprior_info)



    def __del__(self):
        # Nullify object's properties
        if self.GSdata is not None:
            self.GSdata.__del__()
        for model in self.model_list:
            if model is not None:
                model.__del__()
        self.GSdata = None
        self.model_list = None
        self.mtype_list = None
        self.lmParameters = None
        self.weight_spec = None
        self.UC_func = None
        self.fitresult = None
        self.chisq = None


    def _init_ftp_params(self, f_tprior_info):

        if f_tprior_info is None:
            self.use_f_cont_t_prior = False
            self.ftp_name_list  = []
            self.ftp_type_list  = []
            self.ftp_scale  = []
            self.ftp_df     = []
            self.ftp_adj_pairs  = []
            self.csfr_interp_dict = {}
            return
        
        self.use_f_cont_t_prior = True
        ftp_name_list, ftp_type_list, ftp_scale, ftp_df   = f_tprior_info

        ftp_adj_pairs   = []
        for ftpname, ftptype in zip(ftp_name_list, ftp_type_list):
            agebin_length   = len(self.np_agedict[ftpname])

            for k in range(agebin_length-2):
                name_i  = f"{ftpname}_f_cont_bin{k+1}"
                name_j  = f"{ftpname}_f_cont_bin{k+2}"
                idx_i   = self.varnames.index(name_i)
                idx_j   = self.varnames.index(name_j)
                ftp_adj_pairs.append([idx_i, idx_j])
        

        self.ftp_name_list  = ftp_name_list
        self.ftp_type_list  = ftp_type_list
        self.ftp_scale  = ftp_scale
        self.ftp_df     = ftp_df

        self.ftp_adj_pairs  = jnp.array(ftp_adj_pairs)


        ## load the interpolater for different age bin length
        n_list  = [5,6,7,8,9]
        points_name     = ("gal_z", "logM")
        y_name  = "sfr"
        csfr_grid_files      = [os.path.join(_template_path, f"templates/csfr_grids/sfr_grid_n{nv}.npz") for nv in n_list]
        self.csfr_interp_dict    = {nv: _load_npz(file, points_name, y_name) for nv, file in zip(n_list, csfr_grid_files)}


    def update_fpt_params(self, theta):


        ftp_adj_params  = []
        for loop_p, p_key in enumerate(self.ftp_name_list):
            ftp_type    = self.ftp_type_list[loop_p]

            logM_i, zred    = self.np_z_logMidx_dict[p_key]
            age_list    = jnp.array(self.np_agedict[p_key])
            age_bin_list    = age_list[1:] - age_list[:-1]
            logM    = theta[logM_i]

            if ftp_type == 'cosmic':
                # f_cont_tp_means = gsutils._generate_cosmic_sfh_prior(zred, age_list, logM)
                sfr_means     = self.csfr_interp_dict[len(age_list)-1]((zred, logM))
                f_cont_tp_means     = age_bin_list * sfr_means 
                f_cont_tp_means    /= jnp.sum(f_cont_tp_means) # normalized to 1
                f_cont_tp_means     = jnp.log10(f_cont_tp_means)# in logscale 


            elif ftp_type == "flat":
                f_cont_tp_means     = jnp.zeros(len(age_list)-1)

            else:
                raise ValueError("Don't support this prior.")
                
            delta_f_cont_tp_means = f_cont_tp_means[1:] - f_cont_tp_means[:-1]

            for delta_mean in delta_f_cont_tp_means:
                ftp_adj_params.append([delta_mean, self.ftp_scale, self.ftp_df])

        ftp_adj_params  = jnp.array(ftp_adj_params)
        return ftp_adj_params


    def get_model(self, name):
        find = False
        for model in self.model_list:
            if model.name == name:
                find = True
                index = self.model_list.index(model)
                print ("Find {0} model {1}".format(self.mtype_list[index],name))
                return model
        if not find:
            raise ValueError('Model {0} not found'.format(name))
        return None

    def loose_fix_pars(self, loose_list = [], fix_list = [], g_prior = []):
        # gprior: [[parname, mean, sigma],...]
        theta0 = []
        bounds = []
        gbounds = []
        self.varnames=[]
        for el in loose_list:
            self.lmParameters[el].vary = True
        for el in fix_list:
            self.lmParameters[el].vary = False
        self.pardict = self.lmParameters.valuesdict()
        for el in self.pardict:
            if self.lmParameters[el].expr is not None:
                self.lmParameters[el].vary = False
            if self.lmParameters[el].vary:
                theta0.append(self.pardict[el])
                self.varnames.append(el)
                lb, ub = self.lmParameters[el].min, self.lmParameters[el].max
                bounds.append(jnp.array([lb, ub]))
                gbounds.append(jnp.array([0., 1.]))
        self.theta0 = jnp.array(theta0)
        self.ndim = len(self.varnames)
        self.gprior = jnp.zeros(self.ndim,dtype=jnp.bool_)
        for loop, el in enumerate(self.varnames):
            for gpp in g_prior:
                if el == gpp[0]:
                    gbounds[loop] = gbounds[loop].at[0].set(gpp[1])
                    gbounds[loop] = gbounds[loop].at[1].set(gpp[2])
                    self.gprior = self.gprior.at[loop].set(True)
        self.bounds = jnp.array(bounds)
        self.gbounds = jnp.array(gbounds)
        return

    @partial(jax.jit, static_argnames=['self'])
    def ptranform(self, u):
        '''
        transform the unit cube to the prior
        u: unit cube, [0,1] array, shape = (ndim,)
        bounds: 
        for uniform prior (gprior = False), prior bounds, [min, max], shape = (ndim,2)
        for gaussian prior (gprior = True), prior center and stddev, [cen, wid], shape = (ndim,2)
        '''
        priu = u*self.bounds[:,1]+(1.-u)*self.bounds[:,0]
        priu = jnp.where(self.gprior,jax.scipy.stats.norm.ppf(u,self.bounds[:,0],self.bounds[:,1]),priu)
        return priu

    def Set_Update_Constraints(self, UC_func = None):
        self.UC_func = UC_func
    
    def set_wight_spec(self, wight_spec):
        self.weight_spec = wight_spec

    def cal_residual(self,theta):
        for loop,key in enumerate(self.varnames):
            self.pardict[key] = theta[loop]
        if self.UC_func is not None:
            self.UC_func(self.pardict)
        for loop, model in enumerate(self.model_list):
            if self.mtype_list[loop] == 'galaxy':
                if self.lmParameters['z_{0}'.format(model.name)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.name)])
                for com in model.subnames:
                    model.update_subC_para(com,self.pardict)
            elif self.mtype_list[loop] == 'agn':
                if self.lmParameters['z_{0}'.format(model.prefix)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.prefix)])
        first = True
        noSED = False
        if (self.GSdata.fitMode == 'single image - photometry')|(self.GSdata.fitMode == 'images - photometry'):
            noSED = True
        for image_atlas in self.GSdata.image_atlas_list:
            specaper = []
            for loop,band in enumerate(image_atlas.band_list):
                im = image_atlas[loop]
                if im.wcsshift:
                    im.coordinates_transfer_para['x0shift'] = self.pardict['csx_{0}'.format(im.imglabel)]
                    im.coordinates_transfer_para['y0shift'] = self.pardict['csy_{0}'.format(im.imglabel)]
                if (loop == 0)|(not image_atlas.same_grid) :
                    ny,nx = im.cut_image.shape
                    for loop2, model in enumerate(self.model_list):
                        if self.mtype_list[loop2] == 'galaxy':
                            if loop == image_atlas.referimag:
                                totalmass,apertures = model.generate_mass_map((ny,nx),transpar=im.coordinates_transfer_para,aperturemask=image_atlas.specapermask)
                                specaper.append(apertures)
                            else:
                                totalmass,apertures = model.generate_mass_map((ny,nx),transpar=im.coordinates_transfer_para)
                        else:
                            specaper.append(None)
                imm = jnp.zeros((ny,nx),dtype=jnp.float32)
                for loop2, model in enumerate(self.model_list):
                    if self.mtype_list[loop2] == 'galaxy':
                        imm += model.generate_image(band,im.PSF,resp=im.resp,noSED=noSED,nebularpar = self.pardict)
                    elif self.mtype_list[loop2] == 'star':
                        imm += model.generate_image([self.pardict['xcen_{0}'.format(model.prefix)],self.pardict['ycen_{0}'.format(model.prefix)]],[ny,nx], band, im.PSF, resp=im.resp,transpar=im.coordinates_transfer_para, pardict = self.pardict)
                    elif self.mtype_list[loop2] == 'agn':
                        imm += self.pardict['Ni_{0}_{1}'.format(model.prefix,image_atlas.name)]*model.generate_image([self.pardict['xcen_{0}'.format(model.prefix)],self.pardict['ycen_{0}'.format(model.prefix)]],[ny,nx],band,im.PSF,resp=im.resp,transpar=im.coordinates_transfer_para,pardict = self.pardict)
                imm *= im.phys_to_counts_rate
                imm += jnp.ones_like(imm)*self.pardict['sky_{0}'.format(im.imglabel)]
                resi = (im.cut_image-imm)/im.cut_sigma_image
                resi = jnp.where(im.cut_mask_image,0.,resi)
                if first:
                    first = False
                    residu_flat = resi.ravel()
                else:
                    residu_flat=jnp.append(residu_flat,resi.ravel())
            if len(image_atlas.spectra) > 0:
                for loop, spec in enumerate(image_atlas.spectra):
                    flm = jnp.zeros(len(spec[0]),dtype=jnp.float32)
                    for loop2, model in enumerate(self.model_list):
                        if self.mtype_list[loop2] == 'galaxy':
                            if len(image_atlas) == 0:
                                apertures = jnp.ones(len(model.subnames),dtype=jnp.float32)
                            else:
                                apertures = specaper[loop2][loop]
                            fl = model.fiducial_sed(spec[0],apertures=apertures, nebularpar = self.pardict)
                            flm += fl*self.pardict['Ns_{0}_{1}'.format(model.name,image_atlas.specprefix[loop])]
                        elif self.mtype_list[loop2] == 'agn':
                            flm += self.pardict['Ns_{0}_{1}'.format(model.prefix,image_atlas.specprefix[loop])]*model.fiducial_sed(spec[0],pardict=self.pardict)
                    # do convolution if lsf is not None
                    if image_atlas.lsfs[loop] is not None:
                        lsf = image_atlas.lsfs[loop]
                        K = image_atlas.lsfK[loop]
                        flm =SED.conv_spec_lsf_R(spec[0],flm,lsf[0],lsf[1],dcen_kms=0.0, extra_sigma_kms=0.0,Nlogv=4000,K=K)
                    residu_spec = (flm-spec[1])*self.weight_spec/spec[2]
                    if first:
                        first = False
                        residu_flat = residu_spec
                    else:
                        residu_flat=jnp.append(residu_flat,residu_spec)
        return residu_flat

    def cal_model_image(self, pardict=None, ndisagn=False, wrange=[0.85,1.15]):
        if pardict is not None:
            self.pardict = pardict
        for loop, model in enumerate(self.model_list):
            if self.mtype_list[loop] == 'galaxy':
                if self.lmParameters['z_{0}'.format(model.name)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.name)])
                for com in model.subnames:
                    model.update_subC_para(com,self.pardict)
            elif self.mtype_list[loop] == 'agn':
                if self.lmParameters['z_{0}'.format(model.prefix)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.prefix)])
        noSED = False
        if (self.GSdata.fitMode == 'single image - photometry')|(self.GSdata.fitMode == 'images - photometry'):
            pltwave = None
            noSED = True
            z = 0.
        Sedcomp = []
        Sedlabel = []
        for image_atlas in self.GSdata.image_atlas_list:
            specaper = []
            for loop,band in enumerate(image_atlas.band_list):
                im = image_atlas[loop]
                if im.wcsshift:
                    im.coordinates_transfer_para['x0shift'] = self.pardict['csx_{0}'.format(im.imglabel)]
                    im.coordinates_transfer_para['y0shift'] = self.pardict['csy_{0}'.format(im.imglabel)]
                if (loop == 0)|(not image_atlas.same_grid) :
                    ny,nx = im.cut_image.shape
                    for loop2, model in enumerate(self.model_list):
                        if self.mtype_list[loop2] == 'galaxy':
                            if loop == image_atlas.referimag:
                                totalmass,apertures = model.generate_mass_map((ny,nx),transpar=im.coordinates_transfer_para,aperturemask=image_atlas.specapermask)
                                specaper.append(apertures)
                            else:
                                totalmass,apertures = model.generate_mass_map((ny,nx),transpar=im.coordinates_transfer_para)
                            #print ("total mass :", jnp.sum(totalmass))
                        else:
                            specaper.append(None)
                imm = jnp.zeros((ny,nx),dtype=jnp.float32)
                for loop2, model in enumerate(self.model_list):
                    if self.mtype_list[loop2] == 'galaxy':
                        mimg = model.generate_image(band,im.PSF,resp=im.resp,noSED=noSED,nebularpar = self.pardict)
                        imm += mimg
                    elif self.mtype_list[loop2] == 'star':
                        imm += model.generate_image([self.pardict['xcen_{0}'.format(model.prefix)],self.pardict['ycen_{0}'.format(model.prefix)]],[ny,nx], band, im.PSF, resp=im.resp,transpar=im.coordinates_transfer_para, pardict = self.pardict)
                    elif (self.mtype_list[loop2] == 'agn')&(not ndisagn):
                        mimg = model.generate_image([self.pardict['xcen_{0}'.format(model.prefix)],self.pardict['ycen_{0}'.format(model.prefix)]],[ny,nx],band,im.PSF,resp=im.resp,transpar=im.coordinates_transfer_para,pardict = self.pardict)
                        imm += mimg*self.pardict['Ni_{0}_{1}'.format(model.prefix,image_atlas.name)]
                imm *= im.phys_to_counts_rate
                imm += jnp.ones_like(imm)*self.pardict['sky_{0}'.format(im.imglabel)]
                im.model_image = imm
                resi = (im.cut_image-imm)/im.cut_sigma_image
                resi = jnp.where(im.cut_mask_image,0.,resi)
                image_atlas.img_residual[loop] = jnp.sum(resi*resi)
            if len(image_atlas.spectra) > 0:
                mospec = []
                pltwave = None
                for loop, spec in enumerate(image_atlas.spectra):
                    sedcomp = []
                    sedlabel = []
                    flm = jnp.zeros(len(spec[0]),dtype=jnp.float32)
                    z = 0.
                    for loop2, model in enumerate(self.model_list):
                        if self.mtype_list[loop2] == 'galaxy':
                            if len(image_atlas) == 0:
                                apertures = jnp.ones(len(model.subnames),dtype=jnp.float32)
                            else:
                                apertures = specaper[loop2][loop]
                            flist, namelist = model.get_component_SED(spec[0],apertures=apertures,nebularpar = self.pardict)
                            for loop3,fl in enumerate(flist):
                                tf = flist[loop3]*self.pardict['Ns_{0}_{1}'.format(model.name,image_atlas.specprefix[loop])]
                                sedcomp.append(tf)
                                sedlabel.append(namelist[loop3])
                                flm += fl*self.pardict['Ns_{0}_{1}'.format(model.name,image_atlas.specprefix[loop])]
                            z = model.redshift
                        elif self.mtype_list[loop2] == 'agn':
                            agnsed_rest, tsedcomp = model.get_component_SED(spec[0], pardict = self.pardict)
                            for loop3,fl in enumerate(agnsed_rest):
                                if self.pardict['Ns_{0}_{1}'.format(model.prefix,image_atlas.specprefix[loop])] != 0:
                                    tf = agnsed_rest[loop3]*self.pardict['Ns_{0}_{1}'.format(model.prefix,image_atlas.specprefix[loop])]
                                    sedcomp.append(tf)
                                    sedlabel.append(model.prefix+ tsedcomp[loop3])
                                    flm += tf
                            z = model.redshift
                    # do convolution if lsf is not None
                    if image_atlas.lsfs[loop] is not None:
                        lsf = image_atlas.lsfs[loop]
                        K = image_atlas.lsfK[loop]
                        flm =SED.conv_spec_lsf_R(spec[0],flm,lsf[0],lsf[1],dcen_kms=0.0, extra_sigma_kms=0.0,Nlogv=4000,K=K)
                        for loop3 in range(len(sedcomp)):
                            sedcomp[loop3] = SED.conv_spec_lsf_R(spec[0],sedcomp[loop3],lsf[0],lsf[1],dcen_kms=0.0, extra_sigma_kms=0.0,Nlogv=4000,K=K)
                    mospec.append(flm)
                    Sedcomp.append(sedcomp)
                    Sedlabel.append(sedlabel)
                    resi = (flm-spec[1])/spec[2]
                    image_atlas.spec_residual[loop] = jnp.sum(resi*resi*0.5)
                image_atlas.modelspec = mospec
                image_atlas.aper = specaper
                
        if (not noSED) & (len(Sedcomp)==0):
            wavephot = []
            fluxphot = []
            modelphot = []
            errorphot = []
            for loop in range(len(self.GSdata.allbands)):
                im = self.GSdata.get_image(loop)
                sky = self.pardict['sky_{0}'.format(im.imglabel)]
                wavephot.append(gsutils.effective_wave[im.band])
                fluxphot.append(jnp.sum(jnp.where(im.cut_mask_image, 0., im.cut_image-sky))/im.phys_to_counts_rate)
                modelphot.append(jnp.sum(jnp.where(im.cut_mask_image, 0., im.model_image-sky))/im.phys_to_counts_rate)
                if im.photopoint:
                    errorphot.append(jnp.sum(im.cut_sigma_image)/im.phys_to_counts_rate)    
                else:
                    emap = jnp.where(im.cut_mask_image, 0., im.cut_sigma_image)
                    errorphot.append(jnp.sqrt(jnp.sum(emap**2))/im.phys_to_counts_rate)
            wavephot = jnp.array(wavephot)
            fluxphot = jnp.array(fluxphot)
            errorphot = jnp.array(errorphot)
            
        zred_list   = []
        for loop2,model in enumerate(self.model_list):
            sortwave    = model.sortwave
            wmin, wmax  = jnp.min(sortwave), jnp.max(sortwave)
            waverange   = [wmin, wmax]
            
            if self.mtype_list[loop2] == 'galaxy':
                zred    = model.redshift
                wavegrid    = jnp.asarray(SED.wavegrid_from_template(waverange, zred, wrange=[0.9, 1.1]))
                flist, namelist = model.get_component_SED(wavegrid, nebularpar = self.pardict)
                for loop, fl in enumerate(flist):
                    Sedcomp.append(fl)
                    Sedlabel.append(namelist[loop])   
                    zred_list.append(zred)
            elif self.mtype_list[loop2] == 'agn':
                zred    = model.redshift
                wavegrid    = jnp.asarray(SED.wavegrid_from_template(waverange, zred, wrange=[0.9, 1.1]))
                fl  = model.fiducial_sed(wavegrid,pardict = self.pardict)
                Sedcomp.append(fl)
                Sedlabel.append(model.prefix)
                zred_list.append(zred)
            elif self.mtype_list[loop2] == 'star':
                zred    = 0
                wavegrid    = jnp.asarray(SED.wavegrid_from_template(waverange, zred, wrange=[0.9, 1.1]))
                fl = model.fiducial_sed(wavegrid, pardict = self.pardict)
                Sedcomp.append(fl)
                Sedlabel.append(model.prefix)
                zred_list.append(zred)
        pltwave = [wavephot,fluxphot,errorphot,modelphot]
        return pltwave, Sedcomp, Sedlabel, zred_list

    def paracall(self, lfparam):
        theta = jnp.zeros(self.ndim,dtype=jnp.float32)
        for loop,el in enumerate(self.varnames):
            theta = theta.at[loop].set(lfparam[el].value)
        return self.cal_residual(theta)

    @partial(jax.jit, static_argnames=['self'])
    def lnprob(self, theta):
        """
        Compute log-posterior probability (for MCMC/nested sampling).
        
        Combines log-prior and log-likelihood following Bayes' theorem:
        log P(θ|D) = log P(D|θ) + log P(θ) + const
        
        Parameters
        ----------
        theta : array_like
            Parameter vector (free parameters only).
        
        Returns
        -------
        float
            Log-posterior probability.
        
        Notes
        -----
        Used by dynesty nested sampling and MCMC samplers.
        Not used by lmfit (uses chi-squared) or jaxopt (uses loss).
        """
        prior = self.logprior(theta)
        return prior + self.loglike(theta)

    @partial(jax.jit, static_argnames=['self'])
    def loglike(self, theta):
        """
        Compute log-likelihood from data-model residuals.
        
        Assumes Gaussian errors: log L = -0.5 Σ(residual²).
        JIT-compiled for performance.
        
        Parameters
        ----------
        theta : array_like
            Parameter vector.
        
        Returns
        -------
        float
            Log-likelihood (negative chi-squared / 2).
        """
        residu_flat = self.cal_residual(theta)
        return -0.5 * jnp.sum(residu_flat * residu_flat)

    def logprior(self, theta):
        """
        Compute log-prior probability with soft bounds and Gaussian priors.
        
        Combines uniform priors (with smooth boundaries) and Gaussian priors
        for specified parameters. Uses sigmoid transitions at bounds to avoid
        hard cutoffs that cause gradient issues.
        
        Parameters
        ----------
        theta : array_like
            Parameter vector.
        
        Returns
        -------
        float
            Log-prior probability.
        
        Notes
        -----
        For parameters without Gaussian priors: P(θ) = Uniform(min, max) with
        smooth sigmoid transitions. For parameters with priors: P(θ) = N(μ, σ).
        """
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        steepness = 10/(upper_bounds-lower_bounds)
        lower_soft = soft_bound(theta, lower_bounds, steepness)
        upper_soft = soft_bound(upper_bounds, theta, steepness)
        in_bounds = lower_soft * upper_soft
        # Calculate the Gaussian log probabilities
        gps = jax.scipy.stats.norm.logpdf(theta, loc=self.gbounds[:, 0], scale=jnp.abs(self.gbounds[:, 1]))
        # Combine the Gaussian and uniform parts
        uniform_logp = jnp.log(in_bounds + jnp.finfo(float).eps)  # Adding epsilon for numerical stability
        logpList = jnp.where(self.gprior, gps, uniform_logp)
        logp = jnp.sum(logpList)
        return logp

    def _ftp_penalty(self, theta):
        """
        Compute Student-t prior penalty for adjacent f_cont bin differences.

        For each pair (i, j): penalty -= log StudentT(theta[j] - theta[i]; center, scale, df)
        This is added to chisq in the fitness function.

        Parameters
        ----------
        theta : jnp.array
            Current parameter vector.

        Returns
        -------
        float
            Penalty value (positive = increases objective function).
        """
    
        theta_i = theta[self.ftp_adj_pairs[:, 0]]
        theta_j = theta[self.ftp_adj_pairs[:, 1]]

        ftp_adj_params  = self.update_fpt_params(theta)

        diffs = theta_j - theta_i
        tps = jax.scipy.stats.t.logpdf(
            diffs, df=ftp_adj_params[:, 2],
            loc=ftp_adj_params[:, 0], scale=ftp_adj_params[:, 1])
        return -jnp.sum(tps)

    def _calculate_fisher_uncertainties(self, u, cal_chisq):
        """
        Calculate parameter uncertainties using the Fisher Information Matrix.
        More stable than direct Hessian calculation.
        """
        # Calculate gradients for each parameter separately
        n_params = len(u)
        param_errors = jnp.zeros(n_params)
        
        # Function to calculate gradient with respect to one parameter
        def grad_fn(idx):
            @jax.jit
            def g(u_i):
                u_new = u.at[idx].set(u_i)
                return cal_chisq(u_new)
            return g
        
        # Calculate curvature for each parameter independently
        for i in range(n_params):
            try:
                # Use finite differences for second derivative - more stable
                eps = 1e-4  # Small perturbation
                u_plus = u.at[i].add(eps)
                u_minus = u.at[i].add(-eps)
                
                g_plus = jax.grad(cal_chisq)(u_plus)[i]
                g_minus = jax.grad(cal_chisq)(u_minus)[i]
                
                # Second derivative approximation
                hessian_ii = (g_plus - g_minus) / (2 * eps)
                
                if hessian_ii > 0:  # Check for positive curvature
                    # Transform to parameter space
                    u_i = u[i]
                    jacobian_i = (self.bounds[i, 1] - self.bounds[i, 0])
                    
                    # Standard error in parameter space
                    param_errors = param_errors.at[i].set(jacobian_i / jnp.sqrt(hessian_ii))
                else:
                    param_errors = param_errors.at[i].set(jnp.inf)
            except:
                param_errors = param_errors.at[i].set(jnp.inf)
        
        return param_errors

    def _calculate_MCMC_uncertainties(self,best_u,fitness_fn,n_samples=10000,burn_in=1000,r_step=1.0,seed=42,adapt_interval=100,target_acceptance=0.234,use_half_chi2=True,u_eps=1e-8):
        """
        Fast MH MCMC uncertainties (block-jitted with lax.scan) in u∈(0,1)^D.

        Parameters
        ----------
        best_u : (D,) array
            Starting point in u-space.
        fitness_fn : callable
            If use_half_chi2=True: returns chi^2(u)  (so posterior ∝ exp(-0.5*chi^2))
            Else: returns negative log posterior/likelihood (so posterior ∝ exp(-fitness)).
            Must be JAX-compatible; can be jitted outside.
        r_step : float
            Global multiplier applied to per-parameter brute_step.
        adapt_interval : int
            Block size for step adaptation during burn-in.
        target_acceptance : float
            Target acceptance rate for RWM (0.234 is classic high-D optimum).
        use_half_chi2 : bool
            True if fitness_fn returns chi^2. False if it returns -log posterior/likelihood.
        """

        # ---------- build per-dimension step sizes in u-space ----------
        # brute_step in theta-space -> convert to u-space by dividing by parameter range
        step_theta = jnp.array(
            [self.lmParameters[self.varnames[i]].brute_step for i in range(self.ndim)],
            dtype=jnp.float32,
        )
        bounds = jnp.asarray(self.bounds, dtype=jnp.float32)
        ranges = jnp.maximum(bounds[:, 1] - bounds[:, 0], 1e-20)
        step_u_base = (step_theta * float(r_step)) / ranges

        # Avoid pathological scales (optional but usually helpful)
        step_u_base = jnp.clip(step_u_base, 1e-12, 0.5)

        # ---------- initialization ----------
        rng = jax.random.PRNGKey(seed)
        best_u = jnp.asarray(best_u, dtype=jnp.float32)
        best_u = jnp.clip(best_u, u_eps, 1.0 - u_eps)

        # small, scale-aware perturbation so you don't start exactly at a mode
        rng, k0 = jax.random.split(rng)
        jitter = 0.1 * step_u_base * jax.random.normal(k0, shape=(self.ndim,))
        current_u = jnp.clip(best_u + jitter, u_eps, 1.0 - u_eps)
        current_fit = fitness_fn(current_u)

        # ---------- one MH step (used inside scan) ----------
        def mh_step(carry, key):
            u, fit, step_u = carry

            # SPLIT THE KEY
            key_prop, key_acc = jax.random.split(key)
            # propose
            step = jax.random.normal(key_prop, shape=u.shape) * step_u
            prop = u + step

            # reflect into [0,1] then clip (your original behavior)
            prop = jnp.where(prop < 0.0, -prop, prop)
            prop = jnp.where(prop > 1.0, 2.0 - prop, prop)
            prop = jnp.clip(prop, u_eps, 1.0 - u_eps)

            prop_fit = fitness_fn(prop)

            # acceptance log-ratio
            # If fitness is chi^2: log p ∝ -0.5 chi^2  => log_accept = -0.5*(prop-current)
            if use_half_chi2:
                log_alpha = -0.5 * (prop_fit - fit)
            else:
                # fitness is already -log p (up to const): log_alpha = -(prop - current)
                log_alpha = -(prop_fit - fit)

            # accept/reject
            ukey = jax.random.uniform(key_acc, ())
            accept = jnp.log(ukey) < log_alpha

            new_u = jnp.where(accept, prop, u)
            new_fit = jnp.where(accept, prop_fit, fit)

            # return accepted flag for stats + sample
            return (new_u, new_fit, step_u), (new_u, accept)

        # ---------- JIT a block runner ----------
        @partial(jax.jit, static_argnames=("n_steps",))
        def run_block(u0, fit0, step_u, key, n_steps: int):
            keys = jax.random.split(key, n_steps)
            (uT, fitT, _), (u_chain, acc_flags) = jax.lax.scan(
                mh_step,
                (u0, fit0, step_u),
                keys
            )
            acc_count = jnp.sum(acc_flags.astype(jnp.int32))
            return uT, fitT, u_chain, acc_count

        # ---------- burn-in with adaptation (scalar multiplier) ----------
        global_scale = 1.0
        step_u = step_u_base  # Start with your base guesses

        n_burn = int(burn_in)
        n_blocks = n_burn // adapt_interval
        rem = n_burn - n_blocks * adapt_interval

        for _ in tqdm.tqdm(range(n_blocks)):
            rng, kb = jax.random.split(rng)
            # The actual step size used is the per-dimension step * global scale
            current_step = step_u * global_scale

            current_u, current_fit, u_chain, acc_count = run_block(
                current_u, current_fit, current_step, kb, n_steps=adapt_interval
            )
            acc_rate = float(acc_count) / float(adapt_interval)

            # simple multiplicative adaptation of *scalar* step_scale
            if acc_rate > target_acceptance:
                global_scale *= 1.1
            else:
                global_scale *= 0.9
            # Calculate standard deviation of the chain in this block
            chain_std = jnp.std(u_chain, axis=0)
            
            # Smoothly update the step size (50% old, 50% new) to prevent wild fluctuations
            # We add a tiny epsilon to prevent any dimension from collapsing to zero
            step_u = 0.5 * step_u + 0.5 * (chain_std + 1e-8)
            
            # Keep the base relative steps sane in u-space
            step_u = jnp.clip(step_u, 1e-6, 0.5)
        
        # leftover burn-in steps (no adaptation needed, but we run them)
        if rem > 0:
            rng, kr = jax.random.split(rng)
            current_step = step_u * global_scale
            current_u, current_fit, _, _ = run_block(
                current_u, current_fit, current_step, kr, n_steps=rem
            )

        # ---------- production sampling (jitted in one go; or chunk if memory huge) ----------
        rng, kp = jax.random.split(rng)
        final_step_u = step_u * global_scale
        current_u, current_fit, samples_u, _ = run_block(
            current_u, current_fit, final_step_u, kp, n_steps=int(n_samples)
        )

        # ---------- transform back to theta and compute std ----------
        self.mcmc_samples = samples_u * ranges + bounds[:, 0]
        param_errors = jnp.std(self.mcmc_samples, axis=0)

        # MCMC evaluates a jitted fitness function; if cal_residual mutates
        # object state, traced values can leak into self.pardict. Rebuild the
        # fitter state from a concrete best-theta before returning.
        best_theta = best_u * ranges + bounds[:, 0]
        self.update_parameter(best_theta)
        return param_errors

    def _calculate_gn_uncertainties(self, bestu):
   
        # Convert to theta
        besttheta = bestu * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        
        # Define residual function
        @jax.jit
        def residuals(theta):
            return self.cal_residual(theta)
        
        # Compute Jacobian
        J_theta = jax.jacobian(residuals)(besttheta)
        
        # Gauss-Newton Hessian approximation
        H_theta = J_theta.T @ J_theta
        
        # Add small regularization to ensure invertibility (optional)
        H_theta += 1e-6 * jnp.eye(H_theta.shape[0])
        
        # Covariance matrix
        cov_theta = jnp.linalg.inv(H_theta)
        param_errors = jnp.sqrt(jnp.diag(cov_theta))
        
        return param_errors

    def chisq_fit(self):
        fitresult = lmfit.minimize(self.paracall,self.lmParameters,nan_policy='omit' )
        self.lmParameters = fitresult.params
        self.pardict = self.lmParameters.valuesdict()
        self.fitresult = fitresult
        return fitresult

    def update_parameter(self, theta):
        self.theta0 = theta
        self.pardict = self.lmParameters.valuesdict()
        for loop,el in enumerate(self.varnames):
            self.pardict[el] = theta[loop]
        if self.UC_func is not None:
            self.UC_func(self.pardict)
        for el in self.lmParameters:
            self.lmParameters[el].set(value=self.pardict[el])
        par = lmfit.Parameters()
        for loop, model in enumerate(self.model_list):
            if self.mtype_list[loop] == 'galaxy':
                if self.lmParameters['z_{0}'.format(model.name)].vary:
                    model.redshift = model.redshift0
                    model.rnorm = (model.dlr/model.dl0)**2
                    model.currentage = float(SED.cosmo_age(model.redshift0))
                    #model.redshift = self.pardict['z_{0}'.format(model.name)]
                # also report star formation history
                for key in model.subCs:
                    model.update_subC_para(key,self.pardict)
                    if getattr(model, 'useEZ', False):
                        lognorms = model.get_eazy_lognorms(key, self.pardict)
                        logsfr = SED.eazy_lognorm_to_logSFR(lognorms)
                    else:
                        if model.SFH_list[key] == 'bins':
                            age_zero = model.npSFH_times[key]
                            f_cont_zero = jnp.ones(len(age_zero)-1,dtype=jnp.float32)
                            for loop in range(len(age_zero)-1):
                                f_cont_zero = f_cont_zero.at[loop].set(model.f_cont[key]['paradic']['bin{0}'.format(loop+1)])
                        else:
                            age_zero = GALA.Cal_map(0.,model.ageparams[key]['type'],model.ageparams[key]['paradic'])
                            f_cont_zero = GALA.Cal_map(0.,model.f_cont[key]['type'],model.f_cont[key]['paradic'])
                        logsfr = SED.f_cont_to_logSFR(f_cont_zero, model.logMlist[key], age_zero, present_age = model.currentage, SFH = model.SFH_list[key])
                    logsfr = float(logsfr)
                    par.add('logsfr_{0}'.format(key),value=logsfr,min=logsfr-0.5,max=logsfr+0.5,brute_step=0.1,vary=False)
        self.lmParameters.update(par)  
        self.pardict = self.lmParameters.valuesdict()          
        r = self.cal_residual(theta)
        self.chisq = jnp.sum(r*r) 
        self.BIC = self.chisq + self.ndim*jnp.log(self.dof + self.ndim) # BIC = chi2 + k ln n = -2lnL + kln n ? 自由度？
        return

    def optimizer(self, num_steps=1000, learning_rate = 1e-3, gtransformation = False, calsigma = False, optimizer = 'adam', debugmode = False, clip=5., bayesian=False):
        """
        Gradient-based optimization using JAX and Optax optimizers.
        
        Performs chi-squared minimization using automatic differentiation for 
        fast, local optimization. Supports Adam, SGD, AdaBelief, and Adamax
        optimizers with optional gradient clipping and Bayesian priors.
        
        Parameters
        ----------
        num_steps : int, default=1000
            Number of optimization steps (gradient updates).
        learning_rate : float, default=1e-3
            Step size for parameter updates. Smaller = more stable, slower.
        gtransformation : bool, default=False
            Whether to apply gradient clipping based on parameter step sizes.
            Recommended for SGD to prevent divergence.
        calsigma : bool, default=False
            Whether to calculate parameter uncertainties after optimization.
            Uses Fisher information matrix approximation.
        optimizer : str, default='adam'
            Optimizer algorithm to use. Options:
            
            - 'adam': Adaptive moment estimation (recommended for most cases)
            - 'sgd': Stochastic gradient descent (requires gtransformation=True)
            - 'adabelief': AdaBelief optimizer (adaptive learning rate)
            - 'adamax': Adamax variant (more robust to sparse gradients)
        debugmode : bool, default=False
            Whether to save optimization trace and detect NaN parameters.
        clip : float, default=5.
            Maximum absolute value for logit-transformed parameters.
            Prevents extreme parameter values.
        bayesian : bool, default=False
            Whether to include Gaussian prior terms in loss function.
            Uses g_prior list from initialization.
        
        Returns
        -------
        dict
            Dictionary containing:
            
            - 'best_theta' : array_like
                Best-fit parameter values
            - 'uncertainty' : array_like
                Parameter uncertainties (if calsigma=True)
            - 'chisq' : float
                Final chi-squared value
            - 'trace' : list of arrays
                Parameter evolution (if debugmode=True)
        
        Notes
        -----
        **Optimization strategy:**
        
        Parameters are transformed to unconstrained space via logit transform:
        
        .. math::
            u = \\frac{\\theta - \\theta_{\\rm min}}{\\theta_{\\rm max} - \\theta_{\\rm min}}
        
        .. math::
            {\\rm logit}(u) = \\log\\left(\\frac{u}{1-u}\\right)
        
        This ensures parameters stay within bounds without penalty terms.
        
        **Optimizer comparison:**
        
        - **Adam**: Best for most problems, adapts learning rate per parameter
        - **SGD**: Fastest but needs careful tuning, use with gtransformation=True
        - **AdaBelief**: Good for noisy data, adapts to gradient magnitude
        - **Adamax**: More stable for problems with sparse/large gradients
        
        **When to use:**
        
        - Initial fits: Use optimizer with ~1000-5000 steps
        - After nested sampling: Refine best-fit with optimizer(num_steps=500)
        - Multi-component models: May need 10000+ steps for convergence
        
        Examples
        --------
        >>> # Basic optimization
        >>> result = fitter.optimizer(num_steps=2000, learning_rate=1e-3)
        >>> 
        >>> # With uncertainties
        >>> result = fitter.optimizer(num_steps=5000, calsigma=True)
        >>> print(result['uncertainty'])
        >>> 
        >>> # With Bayesian priors
        >>> result = fitter.optimizer(bayesian=True, num_steps=3000)
        
        See Also
        --------
        evolution_strategies : Global optimization for multi-modal problems
        nested_sampling : Full posterior sampling
        """
        u0 = (self.theta0-self.bounds[:,0])/(self.bounds[:,1]-self.bounds[:,0])
        u0 = jnp.where(u0 > 1., (u0-1)/u0, u0)
        u0 = jnp.where(u0 < 0., -u0/(1-u0), u0)
        logitu = logit(u0)
        logitu = jnp.where(jnp.isnan(logitu), 0., logitu)
        logitu = jnp.where(jnp.isinf(logitu), 0., logitu)
        logitu = jnp.where(jnp.abs(logitu) > clip, jnp.sign(logitu)*clip, logitu)
        
        if not bayesian:
            @jit
            def cal_chisq(logitu):
                u = sigmoid(logitu)
                theta = u*self.bounds[:,1]+(1.-u)*self.bounds[:,0]
                r = self.cal_residual(theta)
                chisq = 0.5*jnp.sum(r*r)
                if self.use_f_cont_t_prior:
                    chisq += self._ftp_penalty(theta)
                return chisq
        else:
            # when Gaussian prior is used
            @jit
            def cal_chisq(logitu):
                u = sigmoid(logitu)
                theta = u*self.bounds[:,1]+(1.-u)*self.bounds[:,0]
                r = self.cal_residual(theta)
                chisq = 0.5*jnp.sum(r*r)
                gps = jax.scipy.stats.norm.logpdf(theta, loc=self.gbounds[:, 0], scale=jnp.abs(self.gbounds[:, 1]))
                logpList = jnp.where(self.gprior, gps, 0.)
                logp = -jnp.sum(logpList)
                chisq += logp
                if self.use_f_cont_t_prior:
                    chisq += self._ftp_penalty(theta)
                return chisq
        
        if optimizer == 'adam':
            optimizer = optax.adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            gtransformation = True
            optimizer = optax.sgd(learning_rate=learning_rate)
        elif optimizer == 'adabelief':
            optimizer = optax.adabelief(learning_rate=learning_rate)
        elif optimizer == 'adamax':
            optimizer = optax.adamax(learning_rate=learning_rate)

        if gtransformation:
            step_size = []
            for loop in range(self.ndim):
                step_size.append(self.lmParameters[self.varnames[loop]].brute_step)
            step_size = jnp.array(step_size)
            max_delta = 10*step_size/(self.bounds[:,1]-self.bounds[:,0])
            gradient_transformation = optax.chain(
            optax.clip(max_delta),
            optimizer)
            opt_state = gradient_transformation.init(logitu)
            @jax.jit
            def step(logitu, opt_state):
                g = jax.grad(cal_chisq)(logitu)
                g = jnp.where(jnp.isnan(g) | jnp.isinf(g), 0., g)
                updates, opt_state = gradient_transformation.update(g, opt_state)
                new_logitu = optax.apply_updates(logitu, updates)
                return new_logitu, opt_state, cal_chisq(logitu)
        else:
            opt_state = optimizer.init(logitu)
            @jax.jit
            def step(logitu, opt_state):
                g = jax.grad(cal_chisq)(logitu)
                g = jnp.where(jnp.isnan(g) | jnp.isinf(g), 0., g)
                updates, opt_state = optimizer.update(g, opt_state)
                new_logitu = optax.apply_updates(logitu, updates)
                return new_logitu, opt_state, cal_chisq(logitu)
        logitugood = logitu
        if not debugmode:
            for i in range(num_steps):
                #last = logitu
                logitu, opt_state, loss = step(logitu, opt_state)
                if i % 100 == 0:  # print every 100 steps
                    print(f"Step {i}, Chisq: {loss}")
                    if jnp.isnan(loss):
                        break
                    else:
                        logitugood = logitu
            trace = None
            bad = None
        else:
            trace = []
            bad = []
            for i in range(num_steps):
                #last = logitu
                logitu, opt_state, loss = step(logitu, opt_state)
                if i % 100 == 0:  # print every 100 steps
                    print(f"Step {i}, Chisq: {loss}")
                if jnp.sum(jnp.isnan(logitu)) > 0:
                    bestu = sigmoid(logitu)
                    besttheta = bestu*self.bounds[:,1]+(1.-bestu)*self.bounds[:,0]
                    print ("opt_state_stop at: ", i)
                    for loop2 in range(len(self.varnames)):
                        if jnp.isnan(besttheta[loop2]):
                            bad.append(loop2)
                            print ('varname: ', self.varnames[loop2], ' is nan')
                    break
                else:
                    logitugood = logitu
                    bestu = sigmoid(logitu)
                    trace.append(bestu)
            trace = jnp.array(trace)


        bestu = sigmoid(logitugood)
        besttheta = bestu*self.bounds[:,1]+(1.-bestu)*self.bounds[:,0]

        param_errors = None
        sigmethod = "MCMC"  # 'fisher' or 'MCMC'
        if calsigma:
            if sigmethod == "fisher":
                param_errors = self._calculate_fisher_uncertainties(bestu, cal_chisq)
            elif sigmethod == "MCMC":
                burn_in = 1000 + 500 * self.ndim
                n_samples = 10000 + 10000 * self.ndim
                param_errors = self._calculate_MCMC_uncertainties(
                    bestu, cal_chisq, burn_in=burn_in, n_samples=n_samples
                )

        self.update_parameter(besttheta)
        

        fitresult = {
            'theta': besttheta,
            'chisq': loss*2,
            'param_errors': param_errors,
            'trace':trace,
            'bad':bad,
        }

        return fitresult

    def optimizer_zgrid(self, Z_grid, num_steps=1000, learning_rate=1e-3, calsigma=False, optimizer='adam', bayesian=False):
        u0 = (self.theta0-self.bounds[:,0])/(self.bounds[:,1]-self.bounds[:,0])
        u0 = jnp.where(u0 > 1., (u0-1)/u0, u0)
        u0 = jnp.where(u0 < 0., -u0/(1-u0), u0)
        
        if not bayesian:
            @jit
            def cal_chisq(u, z):
                # First reset model info with the current z
                for loop, model in enumerate(self.model_list):
                    if self.mtype_list[loop] == 'galaxy':
                        model.reset_info(self.pardict, z=z)
                
                # Then calculate chi-square
                theta = u*self.bounds[:,1]+(1.-u)*self.bounds[:,0]
                r = self.cal_residual(theta)
                chisq = 0.5*jnp.sum(r*r)
                return chisq
        else:
            # Similar changes for the Bayesian case
            @jit
            def cal_chisq(u, z):
                # First reset model info with the current z
                for loop, model in enumerate(self.model_list):
                    if self.mtype_list[loop] == 'galaxy':
                        model.reset_info(self.pardict, z=z)
                
                theta = u*self.bounds[:,1]+(1.-u)*self.bounds[:,0]
                r = self.cal_residual(theta)
                chisq = 0.5*jnp.sum(r*r)
                gps = jax.scipy.stats.norm.logpdf(theta, loc=self.gbounds[:, 0], scale=jnp.abs(self.gbounds[:, 1]))
                logpList = jnp.where(self.gprior, gps, 0.)
                logp = -jnp.sum(logpList)
                chisq += logp
                return chisq
        
        # Set up optimizer as before
        if optimizer == 'adam':
            optimizer = optax.adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer = optax.sgd(learning_rate=learning_rate)
        elif optimizer == 'adabelief':
            optimizer = optax.adabelief(learning_rate=learning_rate)
        elif optimizer == 'adamax':
            optimizer = optax.adamax(learning_rate=learning_rate)
        
        @jax.jit
        def step(state, _):
            u, opt_state, z = state
            g = jax.grad(lambda u: cal_chisq(u, z))(u)
            g = jnp.where(jnp.isnan(g), 0., g)
            updates, new_opt_state = optimizer.update(g, opt_state)
            new_u = optax.apply_updates(u, updates)
            # give a rebound to the u
            overflow_low = jnp.minimum(0, new_u)
            overflow_high = jnp.maximum(0, new_u - 1.)
            new_u = new_u - 1.1*overflow_low - 1.1*overflow_high
            return (new_u, new_opt_state, z), None
        
        @jax.jit
        def run_steps(u_init, z):
            opt_state_init = optimizer.init(u_init)
            init_state = (u_init, opt_state_init, z)
            final_state, _ = jax.lax.scan(step, init_state, None, length=num_steps)
            final_u, _, _ = final_state
            return final_u, cal_chisq(final_u, z)
        
        besttheta = []
        best_loss = []
        
        for z in Z_grid:
            u = u0
            # No need to call reset_info here anymore, it's inside cal_chisq
            bestu, loss = run_steps(u, z)
            besttheta.append(bestu*self.bounds[:,1]+(1.-bestu)*self.bounds[:,0])
            best_loss.append(loss)
        
        fitresult = {
            'theta': besttheta,
            'chisq': best_loss*2,
        }
        
        return fitresult

    def optimizer_multi(
        self,
        num_starts=10,
        num_steps=15000,
        learning_rate=1e-4,
        optimizer="adamw",
        use_schedule=True,
        bayesian=False,
        calsigma=False,
        warmup_steps=1000,
        decay_steps=10000,
        seed=42,
        NsGPU=10,
        ):
        """
        Multi-start optimizer that scales across *all* available GPUs.

        Logic:
        - Detect NGPU automatically
        - Each loop launches NGPU devices in parallel (pmap)
        - Each device runs NsGPU starting points in parallel (vmap)
        - Repeat loops until NGPU * NsGPU * n_loops >= num_starts

        If no GPU is available, it falls back to CPU devices (typically 1).
        """

        # ----------------------------
        # Detect devices
        # ----------------------------
        devices, backend = _detect_devices()

        NGPU = len(devices)
        if NGPU < 1:
            raise RuntimeError("No JAX devices found.")

        batch_per_loop = NGPU * int(NsGPU)
        if batch_per_loop <= 0:
            raise ValueError("NsGPU must be >= 1.")

        print(
            f"Starting multi-start optimization: num_starts={num_starts}, "
            f"num_steps={num_steps}, NGPU={NGPU} ({backend}), NsGPU={NsGPU}, "
            f"batch_per_loop={batch_per_loop}"
        )

        # ----------------------------
        # Generate starting points
        # ----------------------------
        key = jrandom.PRNGKey(seed)

        keys = jrandom.split(key, num_starts)
        u_starts = jnp.stack(
            [jrandom.uniform(k, (len(self.bounds),), minval=0.01, maxval=0.99) for k in keys]
        )

        # include the original starting point at index 0
        u0 = (self.theta0 - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        u0 = jnp.clip(u0, 0.01, 0.99)
        u_starts = u_starts.at[0].set(u0)

        # ----------------------------
        # Objective function
        # ----------------------------
        if not bayesian:

            @jit
            def cal_chisq(u):
                theta = u * self.bounds[:, 1] + (1.0 - u) * self.bounds[:, 0]
                r = self.cal_residual(theta)
                chisq = 0.5 * jnp.sum(r * r)
                if self.use_f_cont_t_prior:
                    chisq += self._ftp_penalty(theta)
                return chisq

        else:

            @jit
            def cal_chisq(u):
                theta = u * self.bounds[:, 1] + (1.0 - u) * self.bounds[:, 0]
                r = self.cal_residual(theta)
                chisq = 0.5 * jnp.sum(r * r)
                gps = jax.scipy.stats.norm.logpdf(
                    theta, loc=self.gbounds[:, 0], scale=jnp.abs(self.gbounds[:, 1])
                )
                logpList = jnp.where(self.gprior, gps, 0.0)
                logp = -jnp.sum(logpList)
                if self.use_f_cont_t_prior:
                    chisq += self._ftp_penalty(theta)
                return chisq + logp

        # ----------------------------
        # Learning rate schedule
        # ----------------------------
        if use_schedule:
            warmup_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=learning_rate,
                transition_steps=warmup_steps,
            )
            cosine_schedule = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=decay_steps,
                alpha=0.1,
            )
            schedule = optax.join_schedules(
                schedules=[warmup_schedule, cosine_schedule],
                boundaries=[warmup_steps],
            )
        else:
            schedule = learning_rate

        # ----------------------------
        # Optimizer
        # ----------------------------
        if optimizer == "adamw":
            base_opt = optax.adamw(learning_rate=schedule, weight_decay=1e-6)
        elif optimizer == "adam":
            base_opt = optax.adam(learning_rate=schedule, b1=0.9, b2=0.999, eps=1e-8)
        elif optimizer == "lion":
            base_opt = optax.lion(learning_rate=schedule * 0.3)  # lion usually needs lower LR
        elif optimizer == "sgd":
            base_opt = optax.sgd(learning_rate=schedule, momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            base_opt,
        )

        # ----------------------------
        # Single-start optimization
        # ----------------------------
        @jit
        def optimize_single_start(u_init):
            u = u_init
            opt_state = opt.init(u)

            def step_fn(carry, _):
                u, opt_state, best_u, best_loss = carry

                loss, grads = jax.value_and_grad(cal_chisq)(u)
                grads = jnp.where(jnp.isnan(grads) | jnp.isinf(grads), 0.0, grads)

                updates, opt_state = opt.update(grads, opt_state, u)
                new_u = optax.apply_updates(u, updates)

                # reflect & clip back to bounds
                new_u = jnp.where(new_u < 0, -new_u, new_u)
                new_u = jnp.where(new_u > 1, 2 - new_u, new_u)
                new_u = jnp.clip(new_u, 0.001, 0.999)

                # track best
                best_u = jnp.where(loss < best_loss, new_u, best_u)
                best_loss = jnp.minimum(loss, best_loss)

                return (new_u, opt_state, best_u, best_loss), loss

            init_loss = cal_chisq(u)
            init_carry = (u, opt_state, u, init_loss)
            final_carry, losses = jax.lax.scan(step_fn, init_carry, None, length=num_steps)
            _, _, best_u, best_loss = final_carry
            return best_u, best_loss, losses

        # ----------------------------
        # Multi-device + multi-start batching
        #   pmap over devices, vmap over NsGPU starts per device
        # ----------------------------
        def optimize_on_one_device(u_inits):  # u_inits: (NsGPU, ndim)
            # vmap across NsGPU starts on the same device
            return vmap(optimize_single_start)(u_inits)

        p_optimize = jax.pmap(
            optimize_on_one_device,
            axis_name="devices",
            in_axes=0,
            out_axes=0,
            devices=devices,
        )

        # ----------------------------
        # Pad + reshape starts into loops x NGPU x NsGPU x ndim
        # ----------------------------
        ndim = u_starts.shape[-1]
        n_loops = int(jnp.ceil(num_starts / batch_per_loop))
        padded = n_loops * batch_per_loop
        pad_n = padded - num_starts

        if pad_n > 0:
            pad_block = jnp.tile(u_starts[-1][None, :], (pad_n, 1))
            u_all = jnp.concatenate([u_starts, pad_block], axis=0)
            valid_mask = jnp.concatenate(
                [jnp.ones((num_starts,), dtype=bool), jnp.zeros((pad_n,), dtype=bool)],
                axis=0,
            )
        else:
            u_all = u_starts
            valid_mask = jnp.ones((num_starts,), dtype=bool)

        u_all = u_all.reshape(n_loops, NGPU, NsGPU, ndim)
        valid_mask = valid_mask.reshape(n_loops, NGPU, NsGPU)

        # Decide whether to keep convergence curves (can be huge!)
        keep_curves = (num_starts <= batch_per_loop)

        all_best_u_chunks = []
        all_best_loss_chunks = []
        all_curve_chunks = [] if keep_curves else None

        for li in range(n_loops):
            # Run one batched launch across all devices
            best_u, best_loss, losses = p_optimize(u_all[li])

            # Move to host (still DeviceArrays, but fine for concatenation)
            # Mask out padded entries by setting loss=+inf
            best_loss = jnp.where(valid_mask[li], best_loss, jnp.inf)

            all_best_u_chunks.append(best_u.reshape(-1, ndim))
            all_best_loss_chunks.append(best_loss.reshape(-1))

            if keep_curves:
                all_curve_chunks.append(losses.reshape(-1, num_steps))

            # Optional: light progress (no device-side prints)
            done = min((li + 1) * batch_per_loop, num_starts)
            print(f"  Loop {li+1}/{n_loops}: finished {done}/{num_starts} starts")

        all_best_u = jnp.concatenate(all_best_u_chunks, axis=0)[:num_starts]
        all_best_losses = jnp.concatenate(all_best_loss_chunks, axis=0)[:num_starts]

        if keep_curves:
            all_losses = jnp.concatenate(all_curve_chunks, axis=0)[:num_starts]
        else:
            all_losses = None

        # ----------------------------
        # Global best
        # ----------------------------
        all_best_losses = jnp.where(jnp.isnan(all_best_losses),1e20,all_best_losses)
        global_best_idx = jnp.argmin(all_best_losses)
        global_best_u = all_best_u[global_best_idx]
        global_best_loss = all_best_losses[global_best_idx]

        global_best_theta = global_best_u * self.bounds[:, 1] + (1.0 - global_best_u) * self.bounds[:, 0]

        print("\nGlobal optimization completed:")
        print(f"Best loss: {global_best_loss:.6f}")
        print(f"Best found at start #{int(global_best_idx) + 1}")
        print(f"Loss range: [{jnp.nanmin(all_best_losses):.6f}, {jnp.nanmax(all_best_losses):.6f}]")
        print("all Loss:", all_best_losses)

        # ----------------------------
        # Optional uncertainties
        # ----------------------------
        param_errors = None
        sigmethod = "MCMC"  # 'fisher' or 'MCMC'
        if calsigma:
            if sigmethod == "fisher":
                param_errors = self._calculate_fisher_uncertainties(global_best_u, cal_chisq)
            elif sigmethod == "MCMC":
                burn_in = 1000 + 1000 * self.ndim
                n_samples = 1000 + 3000 * self.ndim
                print(f"Calculating MCMC uncertainties with burn-in={burn_in} and n_samples={n_samples}...")
                param_errors = self._calculate_MCMC_uncertainties(
                    global_best_u, cal_chisq, burn_in=burn_in, n_samples=n_samples
                )

        self.update_parameter(global_best_theta)

        fitresult = {
            "theta": global_best_theta,
            "chisq": global_best_loss,
            "param_errors": param_errors,
            "all_losses": all_best_losses*2,
            "all_u": all_best_u,
            # keep full convergence curves only when small enough to be reasonable
            "convergence_curves": all_losses,
            "NGPU": NGPU,
            "NsGPU": NsGPU,
            "n_loops": n_loops,
        }

        return fitresult

    def evolution_strategies(self, num_generations=10000, popsize=10, seed=9527, inisize=0.4, 
                     verbose=True, method='CMA_ES', bayesian=False, logsigmainits=-3,
                     early_stop_threshold=None, early_stop_patience=50, calsigma = False,
                     es_params=None):
        """
        Evolution Strategies for global optimization using Evosax.
        
        Derivative-free optimization that evolves a population of solutions
        toward the global minimum. Excellent for multi-modal problems and when
        gradients are unavailable or unreliable. Uses CMA-ES (Covariance Matrix
        Adaptation) by default.
        
        Parameters
        ----------
        num_generations : int, default=10000
            Maximum number of generations (iterations) to evolve population.
            Each generation evaluates popsize solutions.
        popsize : int, default=10
            Population size (number of candidate solutions per generation).
            Larger = better exploration but slower. Recommended:
            
            - Simple problems: 10-20
            - Complex/multi-modal: 20-50
            - High-dimensional (>20 params): 50-100
        seed : int, default=9527
            Random seed for reproducibility.
        inisize : float, default=0.4
            Initial search range in logit-transformed space.
            Larger = wider exploration, smaller = local search.
        verbose : bool, default=True
            Whether to show progress bar during evolution.
        method : str, default='CMA_ES'
            Evolution strategy algorithm. Options:
            
            - 'CMA_ES': Covariance Matrix Adaptation (best for most problems)
            - 'Sep_CMA_ES': Separable CMA-ES (faster for high-D, no correlations)
            - 'OpenES': OpenAI's Evolution Strategy (simpler, faster)
            - 'PGPE': Policy Gradients with Parameter Exploration
        bayesian : bool, default=False
            Whether to include Gaussian prior terms in fitness function.
        logsigmainits : float, default=-3
            Log10 of initial standard deviation for parameter search.
            -3 means σ=0.001 in logit space.
        early_stop_threshold : float or None
            Fitness (chi-squared) threshold for early termination.
            Stops if best fitness < threshold.
        early_stop_patience : int, default=50
            Number of generations without improvement before early stopping.
        calsigma : bool, default=False
            Whether to calculate parameter uncertainties after optimization.
        es_params : dict or None
            Algorithm-specific parameters to override defaults. For CMA-ES:
            
            - 'elite_ratio': Fraction of top performers to use (default: 0.4)
            - 'use_negative_weights': Use negative weights (default: True)
            - 'std_init', 'std_min', 'std_max': Adaptation limits
        
        Returns
        -------
        dict
            Dictionary containing:
            
            - 'best_theta' : array_like
                Best-fit parameter values across all generations
            - 'uncertainty' : array_like
                Parameter uncertainties (if calsigma=True)
            - 'chisq' : float
                Best chi-squared value achieved
            - 'fitness_history' : array_like
                Best fitness per generation (for convergence diagnostics)
            - 'state' : ES state object
                Final evolution strategy state (population, covariance, etc.)
        
        Notes
        -----
        **CMA-ES algorithm overview:**
        
        1. Initialize population around starting point
        2. Evaluate fitness (chi-squared) for all candidates
        3. Select elite candidates (top 40% by default)
        4. Update search distribution (mean + covariance matrix)
        5. Adapt step size based on success rate
        6. Generate new population from updated distribution
        7. Repeat until convergence or max generations
        
        **Advantages over gradient methods:**
        
        - Finds global minimum (not trapped in local minima)
        - No derivatives needed (works with non-smooth chi-squared)
        - Automatically adapts search direction and scale
        - Robust to noise and parameter correlations
        
        **When to use:**
        
        - Initial exploration with poor starting guess
        - Multi-component models with degenerate solutions
        - When optimizer gets stuck in local minimum
        - Before nested sampling to find good starting point
        
        **Performance comparison:**
        
        - **vs optimizer**: 10-100× slower, but finds global minimum
        - **vs nested_sampling**: 2-5× faster, but no full posterior
        - **vs grid search**: 100-1000× more efficient
        
        Examples
        --------
        >>> # Basic global optimization
        >>> result = fitter.evolution_strategies(
        ...     num_generations=5000,
        ...     popsize=20,
        ...     method='CMA_ES'
        ... )
        >>> 
        >>> # Large parameter space with custom settings
        >>> result = fitter.evolution_strategies(
        ...     num_generations=10000,
        ...     popsize=50,
        ...     inisize=0.8,  # wider search
        ...     es_params={'elite_ratio': 0.3}
        ... )
        >>> 
        >>> # Quick exploration then refine with optimizer
        >>> es_result = fitter.evolution_strategies(num_generations=3000)
        >>> opt_result = fitter.optimizer(num_steps=2000)
        
        See Also
        --------
        optimizer : Local gradient-based optimization
        nested_sampling : Bayesian posterior sampling
        evolution_strategies_multi : Multi-start ES for very hard problems
        
        References
        ----------
        Hansen & Ostermeier 2001, Evolutionary Computation, 9, 159 (CMA-ES)
        Salimans et al. 2017, arXiv:1703.03864 (OpenES)
        """
        
        ndims = len(self.bounds)
        num_devices = jax.device_count()

        # Prepare initial parameters
        u0 = (self.theta0 - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        u0 = jnp.where(u0 > 1., (u0 - 1) / u0, u0)
        u0 = jnp.where(u0 < 0., -u0 / (1 - u0), u0)
        logitu = logit(u0)
        minpar = logitu - inisize
        minpar = jnp.where(minpar < -4., -4., minpar)
        maxpar = logitu + inisize
        maxpar = jnp.where(maxpar > 4., 4., maxpar)
        
        # Define the fitness function with or without Bayesian priors
        if not bayesian:
            @jax.jit
            def fitness_fn(logitu):
                u = sigmoid(logitu)
                theta = u * self.bounds[:, 1] + (1. - u) * self.bounds[:, 0]
                r = self.cal_residual(theta)
                chisq = 0.5 * jnp.sum(r * r)
                return chisq
        else:
            @jax.jit
            def fitness_fn(logitu):
                u = sigmoid(logitu)
                theta = u * self.bounds[:, 1] + (1. - u) * self.bounds[:, 0]
                r = self.cal_residual(theta)
                chisq = 0.5 * jnp.sum(r * r)
                gps = jax.scipy.stats.norm.logpdf(theta, loc=self.gbounds[:, 0], scale=jnp.abs(self.gbounds[:, 1]))
                logpList = jnp.where(self.gprior, gps, 0.)
                logp = -jnp.sum(logpList)
                chisq += logp
                return chisq
        
        # Define dummy solution for algorithm initialization
        dummy_solution = jnp.zeros(ndims)
        
        # Initialize the algorithm based on the method
        if method == 'CMA_ES':
            # Default algorithm parameters
            algorithm_params = {
                'elite_ratio': 0.4,
                'use_negative_weights': True,
                'std_init': 10**logsigmainits,
                'std_min': 10**(logsigmainits-2),
                'std_max': 10**(logsigmainits+2)
            }
            
            # Update with user-provided parameters if any
            if es_params is not None:
                algorithm_params.update(es_params)
            
            # Create ES instance with specified parameters
            es = algorithms.CMA_ES(
                population_size=popsize,
                solution=dummy_solution
            )
            
            # Set algorithm attributes from parameters
            es.elite_ratio = algorithm_params.get('elite_ratio')
            es.use_negative_weights = algorithm_params.get('use_negative_weights')
            
            # Get default parameters and update them
            params = es.default_params
            
            # Update specific CMA-ES parameters if provided
            if 'std_init' in algorithm_params:
                params = params.replace(std_init=algorithm_params.get('std_init'))
            if 'std_min' in algorithm_params:
                params = params.replace(std_min=algorithm_params.get('std_min'))
            if 'std_max' in algorithm_params:
                params = params.replace(std_max=algorithm_params.get('std_max'))
                
        elif method == 'OpenES':
            # Add OpenES implementation here
            raise NotImplementedError("OpenES not implemented yet")
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        # Initialize the strategy
        rng = jax.random.PRNGKey(seed)
        state = es._init(rng, params)
        
        # Set initial mean to our logitu value if possible
        if hasattr(state, 'mean'):
            state = state.replace(mean=logitu)
        
        # Initialize early stopping
        best_fitness = float('inf')
        patience_counter = 0
        
        # Start the optimization loop
        progress_bar = tqdm.tqdm(range(num_generations), "Generation: ") if verbose else range(num_generations)
        
        for i in progress_bar:
            rng, rng_ask, rng_eval = jax.random.split(rng, 3)
            
            # Ask for new candidates
            population, state = es.ask(rng_ask, state, params)
            
            # Evaluate fitness for all candidates
            fitness = jax.vmap(fitness_fn)(population)
            
            # Tell the results back to the algorithm
            state, metrics = es.tell(rng_eval, population, fitness, state, params)  # Note the key parameter and capturing metrics
            
            # Display progress
            current_fitness = jnp.min(fitness)
            if verbose:
                progress_bar.set_description(f"Generation: {i}, Best Fitness: {current_fitness:.4f}")
            
            # Handle NaN fitness values
            if jnp.isnan(current_fitness):
                print("Fitness is NaN, stopping optimization")
                n = jnp.argmax(jnp.isnan(fitness))
                print(f"NaN at index {n}")
                tu = sigmoid(population[n])
                problem_theta = tu * self.bounds[:, 1] + (1. - tu) * self.bounds[:, 0]
                for loop in range(len(self.varnames)):
                    print(f"{self.varnames[loop]}: {problem_theta[loop]}")
                break
            
            # Early stopping check
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                patience_counter = 0
            else:
                patience_counter += 1
                
            if early_stop_threshold is not None and current_fitness < early_stop_threshold:
                if verbose:
                    print(f"Early stopping: reached threshold {early_stop_threshold}")
                break
                
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"Early stopping: no improvement for {early_stop_patience} generations")
                break
        
        # Get the best member from the final state
        best_solution = state.best_solution if hasattr(state, 'best_solution') else state.mean
        bestu = sigmoid(best_solution)
        besttheta = bestu * self.bounds[:, 1] + (1. - bestu) * self.bounds[:, 0]
        
        # Update parameters and return result
        self.update_parameter(besttheta)
        
        # Calculate final fitness for the best solution
        final_fitness = fitness_fn(best_solution)

        if calsigma:
            '''
            hessian_fn = jax.hessian(fitness_fn)
            hessian_matrix = hessian_fn(best_solution)
            eps = 1e-8
            hessian_matrix = hessian_matrix + eps * jnp.eye(len(best_solution))
            try:
                cov_matrix_logit = jnp.linalg.inv(hessian_matrix)
                u_prime = bestu * (1 - bestu)  # Derivative of sigmoid
                jacobian_diag = (self.bounds[:, 1] - self.bounds[:, 0]) * u_prime
                # Transform covariance matrix to parameter space using the Jacobian
                # For diagonal elements, the variance is (J * Cov * J^T)_ii = J_i^2 * Cov_ii
                param_var = jnp.diag(cov_matrix_logit) * jacobian_diag**2
                # Standard errors are square roots of the variances
                param_errors = jnp.sqrt(param_var)
                # Handle any numerical issues
                param_errors = jnp.where(jnp.isnan(param_errors), jnp.inf, param_errors)
                param_errors = jnp.where(param_errors < 0, jnp.inf, param_errors)
                     
            except jnp.linalg.LinAlgError:
                # If inversion fails, return infinity for all errors
                print("Warning: Hessian inversion failed. Parameter errors cannot be calculated.")
            '''
            sigma_logit = state.std * jnp.sqrt(jnp.diag(state.C))
            # 2) Jacobian of theta w.r.t. logitu: dθ/d(logitu) = (bounds_diff) * sigmoid * (1 - sigmoid)
            bounds_diff = (self.bounds[:, 1] - self.bounds[:, 0])
            jacobian = bounds_diff * bestu * (1 - bestu)  # shape (ndims,)
            # 3) Propagate errors: σ_θ = σ_logit * |dθ/d(logitu)|
            param_errors = sigma_logit * jacobian
            # Handle any numerical issues
            param_errors = jnp.where(jnp.isnan(param_errors), jnp.inf, param_errors)
            param_errors = jnp.where(param_errors < 0, jnp.inf, param_errors)
            
        else:
            param_errors = None
        self.update_parameter(besttheta)
        

        fitresult = {
            'theta': besttheta,
            'param_errors': param_errors,
            'fitness': float(final_fitness),
            'iterations': i + 1,
            'early_stopped': patience_counter >= early_stop_patience,
        }
        
        return fitresult

    def evolution_strategies_theta(self, num_generations=10000, popsize=20, seed=9527,
                    verbose=True, method='Sep_CMA_ES', bayesian=False,
                    early_stop_threshold=None, early_stop_patience=500,
                    calsigma=False, es_params=None):
        """
        Evolution Strategies optimization with hard box-constraints via clipping.

        Parameters:
        -----------
        num_generations : int
            Maximum number of generations to run
        popsize : int
            Population size for the evolutionary algorithm
        seed : int
            Random seed for reproducibility
        verbose : bool
            Whether to show progress information
        method : str
            ES algorithm to use - supported options: 'CMA_ES', 'OpenES', 'PGPE', 'SimpleGA', 'DES', etc.
            Any algorithm available in evosax.algorithms can be specified
        bayesian : bool
            Whether to add Gaussian priors in logpdf
        early_stop_threshold : float
            Fitness threshold for early stopping
        early_stop_patience : int
            Maximum generations without improvement before stopping
        calsigma : bool
            Whether to estimate parameter uncertainties (Fisher or MCMC)
        es_params : dict
            Override default algorithm settings
        """
        
        # Transform parameters to unit hypercube
        u0 = (self.theta0-self.bounds[:,0])/(self.bounds[:,1]-self.bounds[:,0])
        u0 = jnp.where(u0 > 1., (u0-1)/u0, u0)
        u0 = jnp.where(u0 < 0., -u0/(1-u0), u0)
        
        # Define fitness function in θ-space
        if not bayesian:
            @jax.jit
            def fitness_fn(u):
                theta = u * self.bounds[:, 1] + (1. - u) * self.bounds[:, 0] # change back parameters from unit hypercube.
                r = self.cal_residual(theta)
                chisq = 0.5 * jnp.sum(r*r)
                if self.use_f_cont_t_prior:
                    print(f"using f_cont_t_prior")
                    chisq += self._ftp_penalty(theta)
                return chisq

        else:
            @jax.jit
            def fitness_fn(u):
                theta = u * self.bounds[:, 1] + (1. - u) * self.bounds[:, 0]
                r = self.cal_residual(theta)
                chisq = 0.5 * jnp.sum(r*r)
                gps = jax.scipy.stats.norm.logpdf(theta,
                        loc=self.gbounds[:,0], scale=jnp.abs(self.gbounds[:,1]))
                logp = -jnp.sum(jnp.where(self.gprior, gps, 0.))
                if self.use_f_cont_t_prior:
                    print(f"using f_cont_t_prior")
                    chisq += self._ftp_penalty(theta)
                return chisq + logp

        
        AlgorithmClass = getattr(algorithms, method)
        # Create algorithm instance
        es = AlgorithmClass(
            population_size=popsize,
            solution=u0  # Pass initial solution for dimension inference
        )

        
        # Set algorithm-specific parameters
        # First prepare default parameters for specific algorithm types
        if method == 'CMA_ES':
            # CMA-ES specific parameters
            std_init = es_params.get('std_init', 5e-2) if es_params else 5e-2
            std_min = es_params.get('std_min', 1e-6) if es_params else 1e-6
            std_max = es_params.get('std_max',1) if es_params else 1
            elite_ratio = es_params.get('elite_ratio', 0.4) if es_params else 0.4
            
            # Apply parameters
            es.elite_ratio = elite_ratio
            es.use_negative_weights = es_params.get('use_negative_weights', True) if es_params else True
            params = es.default_params.replace(std_init=std_init, std_min=std_min, std_max=std_max)
            
            if verbose:
                print(f"Using CMA-ES with elite_ratio={elite_ratio}, std_init={std_init}, "
                    f"std_min={std_min}, std_max={std_max}")
        
        elif method == 'Open_ES':
            # OpenES specific parameters
            sigma_init = es_params.get('sigma_init', 0.01) if es_params else 0.01
            sigma_decay = es_params.get('sigma_decay', 0.999) if es_params else 0.999
            sigma_limit = es_params.get('sigma_limit', 0.001) if es_params else 0.001
            
            # Apply parameters
            params = es.default_params.replace(
                sigma_init=sigma_init, 
                sigma_decay=sigma_decay, 
                sigma_limit=sigma_limit
            )
            
            if verbose:
                print(f"Using OpenES with sigma_init={sigma_init}, "
                    f"sigma_decay={sigma_decay}, sigma_limit={sigma_limit}")
        
        elif method == 'PGPE':
            # PGPE specific parameters
            sigma_init = es_params.get('sigma_init', 0.1) if es_params else 0.1
            sigma_decay = es_params.get('sigma_decay', 0.999) if es_params else 0.999
            sigma_limit = es_params.get('sigma_limit', 0.01) if es_params else 0.01
            
            # Apply parameters
            params = es.default_params.replace(
                sigma_init=sigma_init, 
                sigma_decay=sigma_decay, 
                sigma_limit=sigma_limit
            )
            
            if verbose:
                print(f"Using PGPE with sigma_init={sigma_init}, "
                    f"sigma_decay={sigma_decay}, sigma_limit={sigma_limit}")
        
        elif method == 'SimpleGA':
            # SimpleGA specific parameters
            mutation_power = es_params.get('mutation_power', 0.02) if es_params else 0.02
            elite_ratio = es_params.get('elite_ratio', 0.1) if es_params else 0.1
            
            # Apply parameters
            params = es.default_params.replace(
                mutation_power=mutation_power,
                elite_ratio=elite_ratio
            )
            
            if verbose:
                print(f"Using SimpleGA with mutation_power={mutation_power}, elite_ratio={elite_ratio}")
                
        elif method == 'DES':
            # DES specific parameters
            sigma_init = es_params.get('sigma_init', 0.1) if es_params else 0.1
            sigma_decay = es_params.get('sigma_decay', 0.999) if es_params else 0.999
            sigma_limit = es_params.get('sigma_limit', 0.01) if es_params else 0.01
            
            # Apply parameters
            params = es.default_params.replace(
                sigma_init=sigma_init, 
                sigma_decay=sigma_decay, 
                sigma_limit=sigma_limit
            )
            
            if verbose:
                print(f"Using DES with sigma_init={sigma_init}, "
                    f"sigma_decay={sigma_decay}, sigma_limit={sigma_limit}")
        
        elif method == 'Sep_CMA_ES':

            std_init = es_params.get('std_init', 5e-4) if es_params else 5e-4
            std_min = es_params.get('std_min', 1e-6) if es_params else 1e-6
            std_max = es_params.get('std_max',1) if es_params else 1
            elite_ratio = es_params.get('elite_ratio', 0.4) if es_params else 0.4
            
            # Apply parameters
            es.elite_ratio = elite_ratio
            es.use_negative_weights = es_params.get('use_negative_weights', True) if es_params else True
            params = es.default_params.replace(std_init=std_init, std_min=std_min, std_max=std_max)
            
            if verbose:
                print(f"Using Sep_CMA-ES with elite_ratio={elite_ratio}, std_init={std_init}, "
                    f"std_min={std_min}, std_max={std_max}")

        else:
            # Generic case - just use default parameters or override with es_params
            params = es.default_params
            
            # Apply any custom parameters from es_params dict
            if es_params:
                param_updates = {}
                for key, value in es_params.items():
                    if hasattr(params, key):
                        param_updates[key] = value
                
                if param_updates:
                    params = params.replace(**param_updates)
            
            if verbose:
                print(f"Using {method} with default parameters")
                if es_params:
                    print(f"Custom parameters: {es_params}")

        # Initialize state with mean=u0
        rng = jax.random.PRNGKey(seed)
        state = es._init(rng, params)
        state = state.replace(mean=u0)
        
        # Early stopping and progress bar
        best_fitness = float('inf')
        patience_counter = 0
        progress_bar = tqdm.tqdm(range(num_generations), "Generation: ") if verbose else range(num_generations)

        for i in progress_bar:
            rng, ask_key, eval_key = jax.random.split(rng, 3)
            population, state = es.ask(ask_key, state, params)

            # Enforce bounds by soft-clipping
            overflow_low = jnp.minimum(0, population)
            overflow_high = jnp.maximum(0, population - 1.)
            population = population - 1.1*overflow_low - 1.1*overflow_high

            # Evaluate fitness
            fitness = jax.vmap(fitness_fn)(population)
            state, _ = es.tell(eval_key, population, fitness, state, params)

            current_fitness = float(jnp.min(fitness))
            if verbose:
                progress_bar.set_description(f"Generation: {i}, Best Fitness: {current_fitness:.4f}")

            # NaN check
            if jnp.isnan(current_fitness):
                print("Fitness is NaN, stopping optimization")
                break

            # Early stopping
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                patience_counter = 0
            else:
                patience_counter += 1

            if early_stop_threshold is not None and current_fitness < early_stop_threshold:
                if verbose:
                    print(f"Early stopping: reached threshold {early_stop_threshold}")
                break

            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"Early stopping: no improvement for {early_stop_patience} generations")
                break

        # Extract best solution
        # Different algorithms might store the best solution differently
        if hasattr(state, 'best_solution'):
            best_u = state.best_solution
        else:
            # Fallback: use best from last population
            best_idx = jnp.argmin(fitness)
            best_u = population[best_idx]

        # Convert back to original parameter space
        best_theta = best_u * self.bounds[:, 1] + (1. - best_u) * self.bounds[:, 0]
        self.update_parameter(best_theta)

        # Calculate parameter uncertainties
        param_errors = None
        sigmethod = 'MCMC'  # Can be 'fisher' or 'MCMC'
        if calsigma:
            if sigmethod == 'fisher':
                param_errors = self._calculate_fisher_uncertainties(best_u, fitness_fn)
            elif sigmethod == 'MCMC':
                burn_in = 1000 + 100*self.ndim
                n_samples = 1000 + 1000*self.ndim
                param_errors = self._calculate_MCMC_uncertainties(best_u, fitness_fn, burn_in=burn_in, n_samples=n_samples)

        # Restore concrete parameter state from best_theta to clear any JAX tracers leaked during optimization or MCMC
        self.update_parameter(best_theta)

        result_dict = {
            'theta': best_theta,
            'param_errors': param_errors,
            'fitness': float(best_fitness),
            'iterations': i+1,
        }

        return result_dict

    def evolution_strategies_multi(
        self,
        num_starts=10,
        num_generations=8000,
        popsize=20,
        seed=42,
        calsigma=False,
        std_init=5e-2,
        std_min=1e-6,
        std_max=1.0,
        elite_ratio=0.4,
        NsGPU=10,
        bayesian=False,
    ):
        """
        Multi-start Evolution Strategies optimization using Sep_CMA_ES.
        
        This follows the same multi-device batching structure as GalfitS:
        - pmap over devices
        - vmap over NsGPU starts per device
        - repeat loops until all starts are processed

        Notes (per your requirement):
        - Starting points are drawn randomly in the *parameter space* (theta).
        - ES hyperparameters (std_init/std_min/std_max/elite_ratio, etc.) are fixed
        and NOT varied across starting points.

        Args:
            num_starts: Total number of starting points.
            NsGPU: Number of starting points per device per loop.
        """

        num_starts = int(num_starts)
        num_generations = int(num_generations)
        popsize = int(popsize)
        NsGPU = int(NsGPU)
        if num_starts < 1:
            raise ValueError("num_starts must be >= 1")
        if NsGPU < 1:
            raise ValueError("NsGPU must be >= 1")
        if getattr(self, 'ndim', 0) <= 0:
            raise ValueError("No free parameters: cannot run evolution_strategies_multi")

        # ----------------------------
        # Detect devices (GPU preferred)
        # ----------------------------
        devices, backend = _detect_devices()

        NGPU = len(devices)
        if NGPU < 1:
            raise RuntimeError("No JAX devices found")

        batch_per_loop = NGPU * NsGPU
        print(
            f"Starting ES multi-start optimization: num_starts={num_starts}, "
            f"num_generations={num_generations}, NGPU={NGPU} ({backend}), NsGPU={NsGPU}, "
            f"batch_per_loop={batch_per_loop}"
        )

        # ----------------------------
        # Generate starting points in *parameter space* theta
        # ----------------------------
        key = jrandom.PRNGKey(int(seed))
        keys = jrandom.split(key, num_starts)
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        span = (ub - lb) + 1e-30

        u_rand = vmap(lambda k: jrandom.uniform(k, (self.ndim,), minval=0.0, maxval=1.0))(keys)
        theta_starts = lb[None, :] + u_rand * span[None, :]

        # Include the original starting point at index 0
        theta0 = jnp.asarray(self.theta0)
        theta_starts = theta_starts.at[0].set(theta0)

        # Convert theta starts into unit-cube u starts for ES
        u_starts = (theta_starts - lb[None, :]) / span[None, :]
        u_starts = jnp.clip(u_starts, 0.01, 0.99)

        u0 = jnp.clip((theta0 - lb) / span, 0.01, 0.99)
        
        # Define fitness function
        if not bayesian:
            @jax.jit
            def fitness_fn(u):
                theta = u * self.bounds[:, 1] + (1. - u) * self.bounds[:, 0]
                r = self.cal_residual(theta)
                return 0.5 * jnp.sum(r * r)
        else:
            @jax.jit
            def fitness_fn(u):
                theta = u * self.bounds[:, 1] + (1. - u) * self.bounds[:, 0]
                r = self.cal_residual(theta)
                chisq = 0.5 * jnp.sum(r * r)
                gps = jax.scipy.stats.norm.logpdf(theta,
                        loc=self.gbounds[:,0], scale=jnp.abs(self.gbounds[:,1]))
                logp = -jnp.sum(jnp.where(self.gprior, gps, 0.))
                return chisq + logp
        
        # Setup Sep_CMA_ES
        AlgorithmClass = getattr(algorithms, 'Sep_CMA_ES')
        es = AlgorithmClass(population_size=popsize, solution=u0)
        
        # Set parameters
        es.elite_ratio = elite_ratio
        es.use_negative_weights = True
        params = es.default_params.replace(
            std_init=std_init, 
            std_min=std_min, 
            std_max=std_max
        )
        
        @jit
        def optimize_single_start_es(u_init, rng_key):
            """Optimize from a single starting point using ES"""
            # Initialize state with mean=u_init
            state = es._init(rng_key, params)
            state = state.replace(mean=u_init)
            
            best_fitness = jnp.inf
            best_u = u_init
            
            def step_fn(carry, _):
                state, best_u, best_fitness, rng = carry
                
                rng, ask_key, eval_key = jrandom.split(rng, 3)
                population, state = es.ask(ask_key, state, params)
                
                # Enforce bounds by soft-clipping (same as your original method)
                overflow_low = jnp.minimum(0, population)
                overflow_high = jnp.maximum(0, population - 1.)
                population = population - 1.1 * overflow_low - 1.1 * overflow_high
                
                # Evaluate fitness
                fitness = jax.vmap(fitness_fn)(population)
                state, _ = es.tell(eval_key, population, fitness, state, params)
                
                # Track best solution
                current_best_idx = jnp.argmin(fitness)
                current_best_fitness = fitness[current_best_idx]
                current_best_u = population[current_best_idx]
                
                # Update global best
                is_better = current_best_fitness < best_fitness
                best_fitness = jnp.where(is_better, current_best_fitness, best_fitness)
                best_u = jnp.where(is_better, current_best_u, best_u)
                
                return (state, best_u, best_fitness, rng), current_best_fitness
            
            init_carry = (state, u_init, fitness_fn(u_init), rng_key)
            final_carry, fitness_history = jax.lax.scan(
                step_fn, init_carry, None, length=num_generations
            )
            
            _, final_best_u, final_best_fitness, _ = final_carry
            return final_best_u, final_best_fitness

        # ----------------------------
        # Multi-device + multi-start batching (pmap over devices, vmap over NsGPU)
        # ----------------------------
        # One RNG key per start
        run_keys = jrandom.split(jrandom.PRNGKey(int(seed) + 12345), num_starts)

        def optimize_on_one_device(u_inits, rng_keys):
            return vmap(optimize_single_start_es)(u_inits, rng_keys)

        p_optimize = jax.pmap(
            optimize_on_one_device,
            axis_name="devices",
            in_axes=(0, 0),
            out_axes=(0, 0),
            devices=devices,
        )

        # Pad + reshape starts into loops x NGPU x NsGPU x ndim
        ndim = int(u_starts.shape[-1])
        n_loops = int(jnp.ceil(num_starts / batch_per_loop))
        padded = n_loops * batch_per_loop
        pad_n = padded - num_starts

        if pad_n > 0:
            pad_u = jnp.tile(u_starts[-1][None, :], (pad_n, 1))
            pad_k = jnp.tile(run_keys[-1][None, :], (pad_n, 1))
            u_all = jnp.concatenate([u_starts, pad_u], axis=0)
            k_all = jnp.concatenate([run_keys, pad_k], axis=0)
            valid_mask = jnp.concatenate(
                [jnp.ones((num_starts,), dtype=bool), jnp.zeros((pad_n,), dtype=bool)],
                axis=0,
            )
        else:
            u_all = u_starts
            k_all = run_keys
            valid_mask = jnp.ones((num_starts,), dtype=bool)

        u_all = u_all.reshape(n_loops, NGPU, NsGPU, ndim)
        k_all = k_all.reshape(n_loops, NGPU, NsGPU, 2)
        valid_mask = valid_mask.reshape(n_loops, NGPU, NsGPU)

        all_best_u_chunks = []
        all_best_loss_chunks = []

        for li in range(n_loops):
            best_u, best_loss = p_optimize(u_all[li], k_all[li])
            best_loss = jnp.where(valid_mask[li], best_loss, jnp.inf)

            all_best_u_chunks.append(best_u.reshape(-1, ndim))
            all_best_loss_chunks.append(best_loss.reshape(-1))

            done = min((li + 1) * batch_per_loop, num_starts)
            print(f"  Loop {li+1}/{n_loops}: finished {done}/{num_starts} starts")

        all_best_u = jnp.concatenate(all_best_u_chunks, axis=0)[:num_starts]
        all_best_losses = jnp.concatenate(all_best_loss_chunks, axis=0)[:num_starts]
        
        # Find globally best solution
        all_best_losses = jnp.where(jnp.isnan(all_best_losses), jnp.inf, all_best_losses)
        global_best_idx = jnp.argmin(all_best_losses)
        global_best_u = all_best_u[global_best_idx]
        global_best_loss = all_best_losses[global_best_idx]
        
        # Convert back to parameter space
        global_best_theta = (global_best_u * self.bounds[:,1] + 
                            (1. - global_best_u) * self.bounds[:,0])
        
        print(f"\nES Global optimization completed:")
        print(f"Best loss: {float(global_best_loss):.6f}")
        print(f"Best found at start #{int(global_best_idx) + 1}")
        print(f"Loss range: [{float(jnp.nanmin(all_best_losses)):.6f}, {float(jnp.nanmax(all_best_losses)):.6f}]")
        



        self.update_parameter(global_best_theta)

        # Calculate parameter uncertainties
        param_errors = None
        sigmethod = 'MCMC'  # Can be 'fisher' or 'MCMC'
        if calsigma:
            if sigmethod == 'fisher':
                param_errors = self._calculate_fisher_uncertainties(global_best_u, fitness_fn)
            elif sigmethod == 'MCMC':
                burn_in = 1000 + 100*self.ndim
                n_samples = 1000 + 1000*self.ndim
                param_errors = self._calculate_MCMC_uncertainties(global_best_u, fitness_fn, burn_in=burn_in, n_samples=n_samples)

        # Restore concrete parameter state from global_best_theta to clear any JAX tracers leaked during ES or MCMC
        self.update_parameter(global_best_theta)

        return {
            'theta': global_best_theta,
            'param_errors': param_errors,
            'best_u': global_best_u,
            'fitness': global_best_loss,
            'all_losses': all_best_losses,
            'all_u': all_best_u,
            'NGPU': NGPU,
            'NsGPU': NsGPU,
            'n_loops': n_loops,
        }

    def jaxopt(self, method='scipy',maxiter=1000,tol=1e-4,**kwargs):
        # map initial theta into unconstrained u0 in (0,1)
        lo, hi = self.bounds[:,0], self.bounds[:,1]
        u0 = (self.theta0 - lo) / (hi - lo)
        # avoid exact 0/1
        u0 = jnp.where(u0 >= 1., (u0 - 1) / u0, u0)
        u0 = jnp.where(u0 <= 0., -u0 / (1 - u0), u0)
        def cal_chisq(logitu):
            u = sigmoid(logitu)
            theta = u * hi + (1. - u) * lo
            r = self.cal_residual(theta)
            return 0.5 * jnp.sum(r * r)

        logitu0 = logit(u0)

        if method == 'scipy':
            solver = jaxopt.ScipyMinimize(fun=cal_chisq,
                                            method='L-BFGS-B',
                                            tol=tol,
                                            maxiter=maxiter,
                                            jit=True,
                                            **kwargs)
            result = solver.run(logitu0)
            logitu_opt = result.params
        elif method == 'GradientDescent':
            solver = jaxopt.GradientDescent(fun=cal_chisq,
                                                tol=tol,
                                                maxiter=maxiter,
                                                jit=True,
                                                **kwargs)
            result = solver.run(init=logitu0)
            logitu_opt = result.params
        elif method == 'NonlinearCG':
            solver = jaxopt.NonlinearCG(fun=cal_chisq,
                                            tol=tol,
                                            maxiter=maxiter,
                                            jit=True,
                                            **kwargs)
            result = solver.run(init=logitu0)
            logitu_opt = result.params
        elif method == 'LBFGS':
            solver = jaxopt.LBFGS(fun=cal_chisq,
                                    tol=tol,
                                    maxiter=maxiter,
                                    jit=True,
                                    **kwargs)
            result = solver.run(init=logitu0)
            logitu_opt = result.params
        else:
            raise ValueError(f"Unknown method '{method}' for unconstrained minimization")

        u_opt = sigmoid(logitu_opt)
        tbest = u_opt * hi + (1. - u_opt) * lo

        # update parameters and return solver result
        self.update_parameter(tbest)
        return result

    def jaxopt_zgrid(self,
                 Z_grid,
                 method: str = 'scipy',scipymethod='L-BFGS-B',
                 maxiter: int = 1000,
                 tol: float = 1e-4,
                 bayesian: bool = False,
                 **solver_kwargs):
        # transform initial theta0 → u0 in (0,1)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        u0 = (self.theta0 - lo) / (hi - lo)
        u0 = jnp.where(u0 >= 1., (u0 - 1) / u0, u0)
        u0 = jnp.where(u0 <= 0., -u0 / (1 - u0), u0)
        logitu0 = logit(u0)

        # define loss in logitu space, resetting model info for each z
        def make_cal(logitu, z):
            u = sigmoid(logitu)
            # reset galaxy models with this z
            for loop, model in enumerate(self.model_list):
                if self.mtype_list[loop] == 'galaxy':
                    model.reset_info(self.pardict, z=z)
            theta = u * hi + (1. - u) * lo
            r = self.cal_residual(theta)
            chisq = 0.5 * jnp.sum(r * r)
            if bayesian:
                gps = jax.scipy.stats.norm.logpdf(theta,
                                                loc=self.gbounds[:, 0],
                                                scale=jnp.abs(self.gbounds[:, 1]))
                logp = -jnp.sum(jnp.where(self.gprior, gps, 0.))
                chisq = chisq + logp
            return chisq

        @jax.jit
        def make_res(logitu, z):
            u = sigmoid(logitu)
            for loop, model in enumerate(self.model_list):
                if self.mtype_list[loop] == 'galaxy':
                    model.reset_info(self.pardict, z=z)
            theta = u * hi + (1. - u) * lo
            return self.cal_residual(theta)
        
        # JIT it for speed
        cal_log = jax.jit(make_cal)

        # pick solver
        if method == 'scipy':
            solver = jaxopt.ScipyMinimize(
                fun=cal_log,
                method=scipymethod,
                tol=tol,
                maxiter=maxiter,
                jit=True,
                **solver_kwargs
            )
        elif method == 'GradientDescent':
            solver = jaxopt.GradientDescent(
                fun=cal_log,
                tol=tol,
                maxiter=maxiter,
                jit=True,
                **solver_kwargs
            )
        elif method == 'NonlinearCG':
            solver = jaxopt.NonlinearCG(
                fun=cal_log,
                tol=tol,
                maxiter=maxiter,
                jit=True,
                **solver_kwargs
            )
        elif method == 'LBFGS':
            solver = jaxopt.LBFGS(
                fun=cal_log,
                tol=tol,
                maxiter=maxiter,
                jit=True,
                **solver_kwargs
            )
        elif method == 'leastsq':
            # Levenberg-Marquardt least-squares
            solver = jaxopt.LevenbergMarquardt(
            residual_fun=make_res,
            maxiter=maxiter,
            tol=tol,
            xtol=tol,
            gtol=tol,
            jac_fun=None,
            materialize_jac=False,
            implicit_diff=True,
            jit=True,
            **solver_kwargs)
        else:
            raise ValueError(f"Unknown JAXOPT method '{method}'")

        best_theta = []
        best_chisq = []


        # run solver for each redshift
        for z in Z_grid:
            # run(init=..., z=z) passes z into our loss above
            result = solver.run(logitu0, z=z)
            logitu_opt = result.params
            u_opt = sigmoid(logitu_opt)
            theta_opt = u_opt * hi + (1. - u_opt) * lo
            best_theta.append(theta_opt)
            # evaluate final chisq
            best_chisq.append(float(cal_log(logitu_opt, z)))

        return {
            'theta': best_theta,   # list of arrays, one per z
            'chisq': best_chisq,   # list of scalars
        }

    def minimize(self,
                 method='scipy',
                 scipymethod='L-BFGS-B',
                 maxiter=1000,
                 tol=1e-4,
                 constrain=False,
                 **kwargs):
        # map initial theta into unconstrained u0 in (0,1)
        lo, hi = self.bounds[:,0], self.bounds[:,1]
        u0 = (self.theta0 - lo) / (hi - lo)
        # avoid exact 0/1
        u0 = jnp.where(u0 >= 1., (u0 - 1) / u0, u0)
        u0 = jnp.where(u0 <= 0., -u0 / (1 - u0), u0)

        if not constrain:
            # optimize over logit(u) unconstrained
            def cal_chisq(logitu):
                u = sigmoid(logitu)
                theta = u * hi + (1. - u) * lo
                r = self.cal_residual(theta)
                return 0.5 * jnp.sum(r * r)

            logitu0 = logit(u0)

            if method == 'scipy':
                solver = jaxopt.ScipyMinimize(fun=cal_chisq,
                                              method=scipymethod,
                                              tol=tol,
                                              maxiter=maxiter,
                                              jit=True,
                                              **kwargs)
                result = solver.run(logitu0)
                logitu_opt = result.params
            elif method == 'GradientDescent':
                solver = jaxopt.GradientDescent(fun=cal_chisq,
                                                 tol=tol,
                                                 maxiter=maxiter,
                                                 jit=True,
                                                 **kwargs)
                result = solver.run(init=logitu0)
                logitu_opt = result.params
            elif method == 'NonlinearCG':
                solver = jaxopt.NonlinearCG(fun=cal_chisq,
                                             tol=tol,
                                             maxiter=maxiter,
                                             jit=True,
                                             **kwargs)
                result = solver.run(init=logitu0)
                logitu_opt = result.params
            elif method == 'LBFGS':
                solver = jaxopt.LBFGS(fun=cal_chisq,
                                      tol=tol,
                                      maxiter=maxiter,
                                      jit=True,
                                      **kwargs)
                result = solver.run(init=logitu0)
                logitu_opt = result.params
            else:
                raise ValueError(f"Unknown method '{method}' for unconstrained minimization")

            u_opt = sigmoid(logitu_opt)
            tbest = u_opt * hi + (1. - u_opt) * lo
            final_chisq = cal_chisq(logitu_opt)

        else:
            # constrained: optimize directly over u in [0,1]
            def cal_chisq_u(u):
                theta = u * hi + (1. - u) * lo
                r = self.cal_residual(theta)
                return 0.5 * jnp.sum(r * r)

            bounds_u = (jnp.zeros(self.ndim), jnp.ones(self.ndim))

            if method == 'ProjectedGradient':
                solver = jaxopt.ProjectedGradient(
                    fun=cal_chisq_u,
                    projection=projection_box,
                    implicit_diff=True,
                    tol=tol,
                    maxiter=maxiter,
                    jit=True,
                    **kwargs
                )
                result = solver.run(u0, hyperparams_proj=bounds_u)
                u_opt = result.params
            elif method == 'MirrorDescent':
                solver = jaxopt.MirrorDescent(
                    fun=cal_chisq_u,
                    projection=projection_box,
                    implicit_diff=True,
                    tol=tol,
                    maxiter=maxiter,
                    jit=True,
                    **kwargs
                )
                result = solver.run(u0, hyperparams_proj=bounds_u)
                u_opt = result.params
            elif method == 'scipy':
                solver = jaxopt.ScipyBoundedMinimize(
                    fun=cal_chisq_u,
                    method=scipymethod,
                    tol=tol,
                    maxiter=maxiter,
                    jit=True,
                    **kwargs
                )
                result = solver.run(u0,bounds=bounds_u)
                u_opt = result.params
            else:
                raise ValueError(f"Unknown method '{method}' for constrained minimization")

            tbest = u_opt * hi + (1. - u_opt) * lo
            final_chisq = cal_chisq_u(u_opt)

        # update parameters and return solver result
        self.update_parameter(tbest)
        return {
            'theta': tbest,
            'chisq': float(final_chisq),
            'param_errors': None,
        }

    def nested_sampling(self, dynamical=False, maxiters=1000, dlogz = 0.02 , workers=1, nlive=150, sample_method='slice',bound='multi', queue_size=None,progress=True,dynesty_kwargs={}, n_effective=None, use_stop=None):
        """
        Nested sampling for Bayesian parameter estimation using Dynesty.
        
        Computes full posterior distributions, parameter uncertainties, and 
        Bayesian evidence. Explores multi-modal posteriors more robustly than
        MCMC. Recommended for final publication-quality uncertainties.
        
        Parameters
        ----------
        dynamical : bool, default=False
            Whether to use dynamic nested sampling. Dynamic mode adaptively
            allocates live points to maximize posterior resolution.
            
            - False: Static nested sampling (faster, less accurate)
            - True: Dynamic nested sampling (slower, better uncertainties)
        maxiters : int, default=1000
            Maximum number of likelihood evaluations. Increase for complex
            multi-component models or tight convergence.
        dlogz : float, default=0.02
            Target evidence tolerance for termination. Smaller = more accurate
            but slower. Typical range: 0.01-0.5.
        workers : int, default=1
            Number of parallel processes for likelihood evaluation.
            Set to cpu_count() for maximum speed.
        nlive : int, default=150
            Number of live points. More points = better posterior resolution
            but slower. Recommended:
            - Simple models (1-5 params): 50-100
            - Moderate models (5-15 params): 150-300
            - Complex models (>15 params): 300-500
        sample_method : str, default='slice'
            Sampling algorithm for new live points. Options:
            - 'slice': Slice sampling (robust, recommended)
            - 'rwalk': Random walk (fast but may struggle with correlations)
            - 'unif': Uniform sampling (for simple/low-D problems)
        bound : str, default='multi'
            Bounding method to constrain live point proposals. Options:
            
            - 'multi': Multiple ellipsoids (handles multi-modal posteriors)
            - 'single': Single ellipsoid (for unimodal posteriors)
            - 'balls': Overlapping balls (robust but slower)
        queue_size : int or None
            Size of parallel queue. Auto-set if None.
        progress : bool, default=True
            Whether to show progress bar during sampling.
        dynesty_kwargs : dict, default={}
            Additional keyword arguments passed to dynesty.DynamicNestedSampler.
        n_effective : int or None, default=None
            Target number of effective posterior samples for dynamic-mode
            early stopping (forwarded to dynesty's run_nested). None uses
            dynesty's default (max(ndim**2, 10000)).
        use_stop : bool or None, default=None
            Whether to enable dynamic-mode early stopping. None leaves
            dynesty's default behavior unless overridden via dynesty_kwargs.
            
        Returns
        -------
        dict
            Dictionary containing:
            
            - 'fitresu' : dynesty.results.Results
                Full dynesty results object with evidence, samples, weights
            - 'samples' : array_like, shape (n_samples, n_params)
                Posterior samples (weighted by evidence)
            - 'best_theta' : array_like
                Posterior mean parameter values
            - 'uncertainty' : array_like
                Posterior standard deviations
        
        Notes
        -----
        **Nested sampling algorithm:**
        
        1. Initialize `nlive` points uniformly in prior
        2. Find point with lowest likelihood
        3. Replace with new point from prior with L > L_min
        4. Repeat until evidence converges (ΔlogZ < dlogz)
        
        **Advantages over MCMC:**
        
        - Computes Bayesian evidence for model comparison
        - Handles multi-modal posteriors naturally
        - No burn-in required
        - Better for high-dimensional problems (>10 params)
        
        **When to use:**
        
        - Final parameter uncertainties for publication
        - Model comparison (evidence ratios)
        - Multi-component fits with degeneracies
        - After optimizer to explore parameter space
        
        **Performance tips:**
        
        - Use workers > 1 for CPU parallelization
        - Start with dynamical=False for quick exploration
        - Use dynamical=True for final uncertainties
        - Increase nlive if posterior is multi-modal
        
        Examples
        --------
        >>> # Quick nested sampling
        >>> result = fitter.nested_sampling(nlive=100, dlogz=0.1)
        >>> print(f"logZ = {result['fitresu'].logz[-1]:.2f}")
        >>> 
        >>> # Publication-quality with parallelization
        >>> result = fitter.nested_sampling(
        ...     dynamical=True,
        ...     nlive=300,
        ...     dlogz=0.01,
        ...     workers=8
        ... )
        >>> samples = result['samples']
        >>> 
        >>> # Model comparison
        >>> result1 = fitter1.nested_sampling()
        >>> result2 = fitter2.nested_sampling()
        >>> delta_logZ = result1['fitresu'].logz[-1] - result2['fitresu'].logz[-1]
        >>> print(f"Bayes factor: {jnp.exp(delta_logZ):.2f}")
        
        See Also
        --------
        optimizer : Fast local optimization
        evolution_strategies : Global optimization without full posterior
        
        References
        ----------
        Skilling 2006, Bayesian Analysis, 1, 833
        Speagle 2020, MNRAS, 493, 3132 (dynesty paper)
        """
        auto_pool = None
        sampler_kwargs = {}
        if isinstance(workers, int) and workers > 1:
            auto_pool = multiprocessing.Pool(workers)
            sampler_kwargs['pool'] = auto_pool
            if queue_size is not None:
                sampler_kwargs['queue_size'] = queue_size
            else:
                sampler_kwargs['queue_size'] = workers
        sampler_kwargs['bound'] = bound

        # Student-t SFH prior: wrap loglike if needed
        if self.use_f_cont_t_prior:
            _orig_loglike = self.loglike
            def loglike_with_prior(theta):
                logl = _orig_loglike(theta)
                logl -= self._ftp_penalty(theta)
                return logl
        else:
            loglike_with_prior = self.loglike

        if not dynamical:
            self.sampler = dynesty.NestedSampler(loglike_with_prior, self.ptranform, self.ndim,
                                                     nlive=nlive,sample=sample_method,
                                    first_update={'min_ncall': 100, 'min_eff': 50.},
                                                     **sampler_kwargs)
            self.sampler.run_nested(dlogz = dlogz, maxiter=maxiters)
        else:
            self.sampler = dynesty.DynamicNestedSampler(loglike_with_prior, self.ptranform, self.ndim,
                                                    sample=sample_method,
                                    first_update={'min_ncall': 100, 'min_eff': 50.},
                                                     **sampler_kwargs)
            try:
                dynesty_run_kwargs = dict(dynesty_kwargs)
                if use_stop is not None:
                    dynesty_run_kwargs['use_stop'] = use_stop
                if n_effective is not None:
                    dynesty_run_kwargs['n_effective'] = n_effective
                self.sampler.run_nested(dlogz_init=dlogz, nlive_init=nlive, **dynesty_run_kwargs)
            except RuntimeError:
                print ('dynamical sampling runtime error')
        self.fitresult = self.sampler.results
        samples = self.fitresult.samples
        weight = jnp.exp(self.fitresult.logwt-jnp.max(self.fitresult.logwt))
        resu = dyutils.mean_and_cov(samples,weight)
        quantiles = resu[0]
        cov = resu[1]
        sigmas = jnp.sqrt(jnp.diag(cov))
        self.update_parameter(quantiles)
        resultdict = {
            'fitresu': self.fitresult,
            'samples' : samples,
            'best_theta': quantiles,
            'uncertainty': sigmas,
            'theta': quantiles,
            'param_errors': sigmas,
        }
        return resultdict

    def jnesty_nested_sampling(self, maxiters=1000000, dlogz=0.02, nlive=500,
                               bound='multi', rwalk_K=None, target_acceptance=0.5,
                               queue_size=None, device='gpu', verbose=True,
                               workplace=None, dynamical=False, maxbatch=100,
                               n_effective=None, use_stop=False,
                               jnesty_kwargs=None):
        """
        Nested sampling using JNesty for Bayesian parameter estimation.

        GPU-accelerated nested sampling with JAX. Computes full posterior
        distributions, parameter uncertainties, and Bayesian evidence.

        Parameters
        ----------
        maxiters : int, default=1000000
            Maximum number of iterations per static run / per dynamic batch
            (forwarded to jnesty's internal static sub-runs in dynamic mode).
        dlogz : float, default=0.02
            Convergence threshold on delta_logZ.
        nlive : int, default=500
            Number of live points.
        bound : str, default='multi'
            Bounding method: 'multi' (recommended), 'single', or 'none'.
        rwalk_K : int or None, default=None
            Random walk steps per iteration. Auto-tuned if None.
        target_acceptance : float, default=0.5
            Target acceptance rate for scale adaptation.
        queue_size : int or None, default=None
            Number of parallel queue entries for GPU likelihood evals.
            Auto-set to 8 for bound='multi' (Dynesty default).
        device : str, default='gpu'
            Device: 'gpu' or 'cpu'. If 'gpu' is requested but no GPU device
            is available, it automatically falls back to 'cpu'.
        verbose : bool, default=True
            Whether to show progress during sampling.
        workplace : str, optional
            Directory in which to save the JNesty results FITS file.
        dynamical : bool, default=False
            If True, use jnesty.DynamicNestedSampler instead of the static
            NestedSampler. Mirrors the '--dynamic' mode of the dynesty path.
        maxbatch : int, default=100
            Maximum number of batches for dynamic nested sampling.
        n_effective : int or None, default=None
            Target number of effective posterior samples for dynamic-mode
            early stopping (forwarded to jnesty's DynamicNestedSampler.
            run_nested). None uses jnesty's default (max(ndim**2, 10000)).
            Only applies when use_stop=True and dynamical=True.
        use_stop : bool, default=False
            Whether to enable dynamic-mode early stopping via n_effective.
        jnesty_kwargs : dict, optional
            Additional keyword arguments passed to jnesty.NestedSampler.
            Default is None (no extra arguments).

        Returns
        -------
        dict
            Dictionary with 'fitresu', 'samples', 'best_theta', 'uncertainty',
            'theta', 'param_errors', 'sampler'.
        """
        if jnesty_kwargs is None:
            jnesty_kwargs = {}

        # Auto-switch to CPU when no GPU device is available.
        if device.lower() == 'gpu':
            _, detected_platform = _detect_devices()
            if detected_platform != 'gpu':
                if verbose:
                    print("JNesty: no GPU device detected; switching device to 'cpu'.")
                device = 'cpu'

        # Prior transform: unit cube [0,1]^ndim -> physical space.
        # Use the same transform as the dynesty path (self.ptranform) so that
        # Gaussian priors (self.gprior) are applied identically in both samplers.
        prior_transform = self.ptranform

        # Log-likelihood: JNesty passes a single parameter array.
        # Apply the identical Student-t SFH adjacent-bin prior penalty that the
        # dynesty path uses, so both samplers target the same posterior.
        if self.use_f_cont_t_prior:
            _orig_loglike = self.loglike

            def loglikelihood(theta):
                logl = _orig_loglike(theta)
                logl -= self._ftp_penalty(theta)
                return logl
        else:
            loglikelihood = self.loglike

        # Create JNesty sampler
        if dynamical:
            sampler = JNestyDynamicSampler(
                loglikelihood=loglikelihood,
                prior_transform=prior_transform,
                ndim=self.ndim,
                nlive=nlive,
                bound=bound,
                rwalk_K=rwalk_K,
                target_acceptance=target_acceptance,
                queue_size=queue_size,
                device=device,
                verbose=verbose,
                **jnesty_kwargs
            )
            self.sampler = sampler

            # Run dynamic nested sampling. Mirror the dynesty dynamic call in
            # self.nested_sampling (dlogz_init=dlogz, nlive_init=nlive,
            # maxbatch=maxbatch, use_stop=False) so both methods are directly
            # comparable; maxiters is forwarded to every internal static
            # sub-run as the per-run iteration cap.
            sampler.run_nested(
                nlive_init=nlive,
                dlogz_init=dlogz,
                maxbatch=maxbatch,
                max_iterations=maxiters,
                n_effective=n_effective,
                use_stop=use_stop,
                print_progress=verbose,
            )
        else:
            sampler = JNestySampler(
                loglikelihood=loglikelihood,
                prior_transform=prior_transform,
                ndim=self.ndim,
                nlive=nlive,
                rwalk_K=rwalk_K,
                target_acceptance=target_acceptance,
                batch_size=None,
                queue_size=queue_size,
                device=device,
                verbose=verbose,
                bound=bound,
                **jnesty_kwargs
            )
            self.sampler = sampler

            # Run nested sampling
            sampler.run_nested(
                max_iterations=maxiters,
                delta_logZ_threshold=dlogz,
                print_progress=verbose,
            )

        # Extract results
        results = sampler.results
        self.fitresult = results

        # Compute weighted statistics using importance weights
        # (matches Dynesty's dyutils.mean_and_cov)
        from jnesty.utils import mean_and_cov
        logwt = results['logwt']
        logwt_max = np.max(logwt)
        weights = np.exp(logwt - logwt_max)
        weights /= weights.sum()

        if dynamical:
            # Dynamic results only carry unit-cube coordinates (samples_u) and
            # trajectory logz/logzerr; transform to physical space and take
            # the final evidence values.
            param_samples = np.asarray(self.ptranform(np.asarray(results['samples_u'])))
            logz_val = float(np.asarray(results['logz'])[-1])
            logzerr_val = float(np.asarray(results['logzerr'])[-1])
            niter_val = int(len(results['logl']))
        else:
            param_samples = results['samples']
            logz_val = float(results['logz'])
            logzerr_val = float(results['logzerr'])
            niter_val = int(results['niter'])
        theta_mean, cov = mean_and_cov(param_samples, weights)
        theta_std = np.sqrt(np.diag(cov))

        # Update model parameters with weighted mean
        self.update_parameter(theta_mean)

        # Display summary
        if verbose:
            print(f"\nlogZ = {logz_val:.4f} +/- {logzerr_val:.4f}")
            if dynamical:
                nbatch = len(results.get('batch_nlive', [])) - 1
                print(f"niter = {niter_val} (dynamic, {nbatch} batches)")
            else:
                print(f"niter = {niter_val}, efficiency = {results['eff']:.2f}%")

        resultdict = {
            'fitresu': results,
            'samples': param_samples,
            'best_theta': theta_mean,
            'uncertainty': theta_std,
            'theta': theta_mean,
            'param_errors': theta_std,
            'sampler': sampler,
        }

        # Save results to FITS file
        if workplace is not None:
            if dynamical:
                fits_path = os.path.join(workplace, 'jnesty_dynamic_results.fits')
                # jnesty.save_results expects a static-style Results dict with
                # scalar logz/logzerr and a physical-space 'samples' array;
                # normalize the dynamic output before saving.
                save_dict = dict(results)
                save_dict['samples'] = param_samples
                save_dict['logz'] = logz_val
                save_dict['logzerr'] = logzerr_val
                save_dict['niter'] = niter_val
                save_dict['logz_trajectory'] = np.asarray(results['logz'])
                save_dict['logzerr_trajectory'] = np.asarray(results['logzerr'])
            else:
                fits_path = os.path.join(workplace, 'jnesty_results.fits')
                save_dict = results
            try:
                jnesty_save_results(save_dict, fits_path)
                if verbose:
                    print(f"JNesty results saved to {fits_path}")
            except Exception as e:
                print(f"Warning: Failed to save JNesty results to FITS: {e}")

        return resultdict

    def plot_jnesty_diagnostics(self, result, save_dir='./'):
        """
        Plot JNesty diagnostic figures.

        Parameters
        ----------
        result : dict
            Results dictionary from jnesty_nested_sampling()
        save_dir : str, default='./'
            Directory to save plots

        Returns
        -------
        list
            List of file paths to saved plots
        """
        import matplotlib.pyplot as plt
        from pathlib import Path

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if 'sampler' not in result:
            print("Warning: sampler not in results. Cannot plot diagnostics.")
            return []

        sampler = result['sampler']
        results = result['fitresu']

        # Dynamic JNesty results carry only unit-cube coordinates (samples_u)
        # and trajectory logz/logzerr. Build a plotting-friendly dict with
        # physical-space samples and scalar evidence summaries so that the
        # jnesty plotting functions (which expect the static format) work.
        if ('samples' not in results) and ('samples_u' in results):
            results = dict(results)
            results['samples'] = np.asarray(
                self.ptranform(np.asarray(results['samples_u'])))
            results['logz_trajectory'] = np.asarray(results['logz'])
            results['logz'] = float(np.asarray(results['logz'])[-1])
            results['logzerr'] = float(np.asarray(results['logzerr'])[-1])

        saved_files = []

        try:
            print("\nGenerating JNesty diagnostic plots...")

            # Run plot (evidence evolution, likelihood, weights)
            run_file = Path(save_dir) / 'jnesty_runplot.png'
            fig, axes = jnesty_plotting.runplot(results)
            fig.savefig(str(run_file), dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(str(run_file))
            print(f"Run plot saved: {run_file}")

            # Trace plot (parameter traces + marginals)
            ndim = results['samples'].shape[1] if results['samples'].ndim > 1 else 1
            trace_dims = tuple(range(min(ndim, 10)))
            trace_file = Path(save_dir) / 'jnesty_traceplot.png'
            fig, axes = jnesty_plotting.traceplot(results, dims=trace_dims,
                                                   max_samples=5000)
            fig.savefig(str(trace_file), dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(str(trace_file))
            print(f"Trace plot saved: {trace_file}")

            # Diagnostic summary (convergence, efficiency)
            diag_file = Path(save_dir) / 'jnesty_diagnostics.png'
            fig, axes = jnesty_plotting.diagnostics(results)
            fig.savefig(str(diag_file), dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(str(diag_file))
            print(f"Diagnostic plot saved: {diag_file}")

            # Corner plot
            corner_file = Path(save_dir) / 'jnesty_cornerplot.png'
            fig, axes = jnesty_plotting.cornerplot(results)
            fig.savefig(str(corner_file), dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(str(corner_file))
            print(f"Corner plot saved: {corner_file}")

            print(f"Total {len(saved_files)} plots saved to {save_dir}\n")

        except Exception as e:
            print(f"Warning: Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

        return saved_files

     
class imagefitter_phot(gsfitter):
    """
    Photometry-based multi-band image fitter.
    
    Optimized for fitting multi-wavelength imaging without SED modeling.
    Uses band-specific normalization parameters instead of full SED computation.
    Faster than imagefitter_SED for pure morphology fitting.
    
    Inherits all optimization methods from gsfitter.
    
    See Also
    --------
    imagefitter_SED : Full SED-based image fitting
    mbifitter : Image + grism fitting
    """
    # specific for multi-band image fitting, developed version
    def cal_residual(self,theta):
        for loop,key in enumerate(self.varnames):
            self.pardict[key] = theta[loop]
        if self.UC_func is not None:
            self.UC_func(self.pardict)
        for loop, model in enumerate(self.gmodel_list):
            for com in model.subnames:
                model.update_subC_para(com,self.pardict)
        first = True
        GSdata = self.GSdata
        for loop_group in range(len(GSdata.imageset)):
            ny,nx = GSdata.imagesizes[loop_group]
            for gmodel in self.gmodel_list:
                totalmass,apertures = gmodel.generate_mass_map((ny,nx),transpar=GSdata.coordinates_transfer_paras[loop_group])
            for loop_image in GSdata.imageset[loop_group]:
                im = GSdata.get_image(loop_image)
                if im.wcsshift:
                    im.coordinates_transfer_para['x0shift'] = self.pardict['csx_{0}'.format(im.imglabel)]
                    im.coordinates_transfer_para['y0shift'] = self.pardict['csy_{0}'.format(im.imglabel)]
                imm0 = jnp.zeros((ny,nx),dtype=jnp.float32)
                for gmodel in self.gmodel_list:
                    for key in gmodel.subCs:
                        imm0 += 10**self.pardict['logNorm_{0}_{1}'.format(key,im.band)]*gmodel.mass_map[key]
                # downgrade the image to its local frame  
                nyl,nxl = im.cut_image.shape
                scale0 = GSdata.coordinates_transfer_paras[loop_group]['pixsc']/im.coordinates_transfer_para['pixsc']
                scale = jnp.array([scale0,scale0], dtype=jnp.float32)
                shiftx = im.coordinates_transfer_para['x0shift']+im.coordinates_transfer_para['x0']-(GSdata.coordinates_transfer_paras[loop_group]['x0shift']+GSdata.coordinates_transfer_paras[loop_group]['x0'])*scale0+0.5
                shifty = im.coordinates_transfer_para['y0shift']+im.coordinates_transfer_para['y0']-(GSdata.coordinates_transfer_paras[loop_group]['y0shift']+GSdata.coordinates_transfer_paras[loop_group]['y0'])*scale0+0.5
                trans = jnp.array([shifty,shiftx], dtype=jnp.float32)
                imm = jax.image.scale_and_translate(imm0,(nyl,nxl),(0,1),scale,trans,'cubic')
                imm /= scale0**2
                imm = jax.scipy.signal.fftconvolve(imm,im.PSF,mode='same')
                # now add the PSFs
                for psfmodel in self.psfmodel_list:
                    imm += psfmodel.generate_image([self.pardict['xcen_{0}'.format(psfmodel.prefix)],self.pardict['ycen_{0}'.format(psfmodel.prefix)]],[nyl,nxl],im.band,im.PSF,resp=im.resp,transpar=im.coordinates_transfer_para,pardict = self.pardict)
                imm *= im.phys_to_counts_rate
                imm += jnp.ones_like(imm)*self.pardict['sky_{0}'.format(im.imglabel)]
                resi = (im.cut_image-imm)/im.cut_sigma_image
                resi = jnp.where(im.cut_mask_image,0.,resi)
                if first:
                    first = False
                    residu_flat = resi.ravel()
                else:
                    residu_flat=jnp.append(residu_flat,resi.ravel())        
                                   
        return residu_flat
    
    def cal_model_image(self, pardict=None, ndisagn=False, wrange=[0.85, 1.15]):
        if self.UC_func is not None:
            self.UC_func(self.pardict)
        for loop, model in enumerate(self.gmodel_list):
            for com in model.subnames:
                model.update_subC_para(com,self.pardict)
        first = True
        GSdata = self.GSdata
        for loop_group in range(len(GSdata.imageset)):
            ny,nx = GSdata.imagesizes[loop_group]
            for gmodel in self.gmodel_list:
                totalmass,apertures = gmodel.generate_mass_map((ny,nx),transpar=GSdata.coordinates_transfer_paras[loop_group])
            for loop_image in GSdata.imageset[loop_group]:
                im = GSdata.get_image(loop_image)
                if im.wcsshift:
                    im.coordinates_transfer_para['x0shift'] = self.pardict['csx_{0}'.format(im.imglabel)]
                    im.coordinates_transfer_para['y0shift'] = self.pardict['csy_{0}'.format(im.imglabel)]
                imm0 = jnp.zeros((ny,nx),dtype=jnp.float32)
                for gmodel in self.gmodel_list:
                    for key in gmodel.subCs:
                        imm0 += 10**self.pardict['logNorm_{0}_{1}'.format(key,im.band)]*gmodel.mass_map[key]

                # downgrade the image to its local frame
                nyl,nxl = im.cut_image.shape
                scale0 = GSdata.coordinates_transfer_paras[loop_group]['pixsc']/im.coordinates_transfer_para['pixsc']
                scale = jnp.array([scale0,scale0], dtype=jnp.float32)
                shiftx = im.coordinates_transfer_para['x0shift']+im.coordinates_transfer_para['x0']-(GSdata.coordinates_transfer_paras[loop_group]['x0shift']+GSdata.coordinates_transfer_paras[loop_group]['x0'])*scale0+0.5
                shifty = im.coordinates_transfer_para['y0shift']+im.coordinates_transfer_para['y0']-(GSdata.coordinates_transfer_paras[loop_group]['y0shift']+GSdata.coordinates_transfer_paras[loop_group]['y0'])*scale0+0.5 # in im frame。 imm0's orign point is the leftest corner of image.
                trans = jnp.array([shifty,shiftx], dtype=jnp.float32)
                imm = jax.image.scale_and_translate(imm0,(nyl,nxl),(0,1),scale,trans,'cubic') # col'+0.5 = (col+0.5) * scale + trans, so trans = col'+0.5-scale * (col+0.5）
                imm /= scale0**2
                imm = jax.scipy.signal.fftconvolve(imm,im.PSF,mode='same')
                #print ('mass2:', jnp.sum(imm))
                # now add the PSFs
                for psfmodel in self.psfmodel_list:
                    imm += psfmodel.generate_image([self.pardict['xcen_{0}'.format(psfmodel.prefix)],self.pardict['ycen_{0}'.format(psfmodel.prefix)]],[nyl,nxl],im.band,im.PSF,resp=im.resp,transpar=im.coordinates_transfer_para,pardict = self.pardict)
                imm *= im.phys_to_counts_rate
                imm += jnp.ones_like(imm)*self.pardict['sky_{0}'.format(im.imglabel)]
                #print ('mass3:', jnp.sum(imm))
                im.model_image = imm
                resi = (im.cut_image-imm)/im.cut_sigma_image
                resi = jnp.where(im.cut_mask_image,0.,resi)
                image_atlas = GSdata[GSdata.listindex[loop_image]]
                image_atlas.img_residual[GSdata.atlasindex[loop_image]] = jnp.sum(resi*resi)
        pltwave = None
        Sedcomp = []
        Sedlabel = []
        z = []

        return pltwave, Sedcomp, Sedlabel, z

class imagefitter_SED_allconst(gsfitter):
    """
    SED-based multi-band image fitter optimized for constant SEDs.
    """
    def cal_residual(self,theta):

        for loop,key in enumerate(self.varnames):
            self.pardict[key] = theta[loop]
        
        if self.UC_func is not None:
            self.UC_func(self.pardict)

        for loop, model in enumerate(self.model_list):
            if self.mtype_list[loop] == 'galaxy':
                if self.lmParameters['z_{0}'.format(model.name)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.name)])

                for com in model.subnames:
                    model.update_subC_para(com,self.pardict)

            elif self.mtype_list[loop] == 'agn':
                if self.lmParameters['z_{0}'.format(model.prefix)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.prefix)])

        first   = True
        GSdata  = self.GSdata
        wavegrid    = GSdata.wavegrid
        for loop_group in range(len(GSdata.imageset)):
            ny,nx   = GSdata.imagesizes[loop_group]
            resps_tem   = GSdata.subgroup_resps[loop_group]
            imm0s_total = jnp.zeros((len(resps_tem), ny, nx), dtype=jnp.float32)

            for gmodel in self.gmodel_list:
                totalmass, apertures    = gmodel.generate_mass_map((ny,nx),transpar=GSdata.coordinates_transfer_paras[loop_group])
                # sfrmap = gmodel.generate_SFR_map() # Maybe not needed? But harmless.
                imm0s_total += gmodel.generate_group_image(wavegrid, resps_tem, pardict=self.pardict)

            for loopb, loop_image in enumerate(GSdata.imageset[loop_group]):
                im = GSdata.get_image(loop_image)
                resp_tem    = resps_tem[loopb] # resampled resp for loop_image.
                if im.wcsshift:
                    im.coordinates_transfer_para['x0shift'] = self.pardict['csx_{0}'.format(im.imglabel)]
                    im.coordinates_transfer_para['y0shift'] = self.pardict['csy_{0}'.format(im.imglabel)]
                imm0 = imm0s_total[loopb]
                
                if not im.photopoint:
                    # downgrade the image to its local frame
                    nyl,nxl = im.cut_image.shape
                    scale0 = GSdata.coordinates_transfer_paras[loop_group]['pixsc']/im.coordinates_transfer_para['pixsc']
                    scale = jnp.array([scale0,scale0], dtype=jnp.float32)
                    shiftx = im.coordinates_transfer_para['x0shift']+im.coordinates_transfer_para['x0']-(GSdata.coordinates_transfer_paras[loop_group]['x0shift']+GSdata.coordinates_transfer_paras[loop_group]['x0'])*scale0+0.5
                    shifty = im.coordinates_transfer_para['y0shift']+im.coordinates_transfer_para['y0']-(GSdata.coordinates_transfer_paras[loop_group]['y0shift']+GSdata.coordinates_transfer_paras[loop_group]['y0'])*scale0+0.5
                    trans = jnp.array([shifty,shiftx], dtype=jnp.float32)
                    imm = jax.image.scale_and_translate(imm0,(nyl,nxl),(0,1),scale,trans,'cubic')
                    imm /= scale0**2
                    imm = jax.scipy.signal.fftconvolve(imm,im.PSF,mode='same')
                    # now add the PSFs (AGNs, stars.)
                    for psfmodel in self.psfmodel_list:
                        posi_psf    = [self.pardict['xcen_{0}'.format(psfmodel.prefix)],self.pardict['ycen_{0}'.format(psfmodel.prefix)]]
                        shape_img   = [nyl,nxl]
                        imm += psfmodel.generate_image(posi_psf, shape_img, im.band, im.PSF ,resp=resp_tem, wavegrid=wavegrid, transpar=im.coordinates_transfer_para, pardict = self.pardict)
                else:
                    imm = jnp.sum(imm0)*jnp.array([[1.]])
                    for psfmodel in self.psfmodel_list:
                        zred_temp   = psfmodel.redshift
                        waveintrin  = wavegrid / (1+zred_temp)
                        agn_sed     = psfmodel.fiducial_sed(waveintrin, pardict=self.pardict)
                        imm += psfmodel.generate_SED([resp_tem], wavegrid, agn_sed)[0]

                imm *= im.phys_to_counts_rate
                imm += jnp.ones_like(imm)*self.pardict['sky_{0}'.format(im.imglabel)]
                resi = (im.cut_image-imm)/im.cut_sigma_image
                resi = jnp.where(im.cut_mask_image,0.,resi)
                if first:
                    first = False
                    residu_flat = resi.ravel()
                else:
                    residu_flat=jnp.append(residu_flat,resi.ravel())
        return residu_flat

    def cal_model_image(self, pardict=None, ndisagn=False, wrange=[0.85, 1.15]):
        if self.UC_func is not None:
            self.UC_func(self.pardict)
            
        for loop, model in enumerate(self.model_list):
            if self.mtype_list[loop] == 'galaxy':
                if self.lmParameters['z_{0}'.format(model.name)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.name)])
                for com in model.subnames:
                    model.update_subC_para(com,self.pardict)
            elif self.mtype_list[loop] == 'agn':
                if self.lmParameters['z_{0}'.format(model.prefix)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.prefix)])

        GSdata = self.GSdata
        wavegrid    = GSdata.wavegrid
        for loop_group in range(len(GSdata.imageset)):
            ny,nx = GSdata.imagesizes[loop_group]
            resps_tem   = GSdata.subgroup_resps[loop_group]
            imm0s_total = jnp.zeros((len(resps_tem), ny, nx), dtype=jnp.float32)

            for gmodel in self.gmodel_list:
                totalmass,apertures = gmodel.generate_mass_map((ny, nx),transpar=GSdata.coordinates_transfer_paras[loop_group])
                # sfrmap = gmodel.generate_SFR_map()
                imm0s_total += gmodel.generate_group_image(wavegrid, resps_tem, pardict=self.pardict)
                
            for loopb, loop_image in enumerate(GSdata.imageset[loop_group]):
                im = GSdata.get_image(loop_image)
                resp_tem    = resps_tem[loopb] # resampled resp for loop_image.

                if im.wcsshift:
                    im.coordinates_transfer_para['x0shift'] = self.pardict['csx_{0}'.format(im.imglabel)]
                    im.coordinates_transfer_para['y0shift'] = self.pardict['csy_{0}'.format(im.imglabel)]
                imm0 = imm0s_total[loopb]
                
                if not im.photopoint:
                    # downgrade the image to its local frame
                    nyl,nxl = im.cut_image.shape
                    scale0 = GSdata.coordinates_transfer_paras[loop_group]['pixsc']/im.coordinates_transfer_para['pixsc']
                    scale = jnp.array([scale0,scale0], dtype=jnp.float32)
                    shiftx = im.coordinates_transfer_para['x0shift']+im.coordinates_transfer_para['x0']-(GSdata.coordinates_transfer_paras[loop_group]['x0shift']+GSdata.coordinates_transfer_paras[loop_group]['x0'])*scale0+0.5
                    shifty = im.coordinates_transfer_para['y0shift']+im.coordinates_transfer_para['y0']-(GSdata.coordinates_transfer_paras[loop_group]['y0shift']+GSdata.coordinates_transfer_paras[loop_group]['y0'])*scale0+0.5
                    trans = jnp.array([shifty,shiftx], dtype=jnp.float32)
                    imm = jax.image.scale_and_translate(imm0,(nyl,nxl),(0,1),scale,trans,'cubic')
                    imm /= scale0**2
                    imm = jax.scipy.signal.fftconvolve(imm,im.PSF,mode='same')
                    # now add the PSFs
                    for psfmodel in self.psfmodel_list:
                        posi_psf    = [self.pardict['xcen_{0}'.format(psfmodel.prefix)],self.pardict['ycen_{0}'.format(psfmodel.prefix)]]
                        shape_img   = [nyl,nxl]
                        imm += psfmodel.generate_image(posi_psf, shape_img, im.band, im.PSF ,resp=resp_tem, wavegrid=wavegrid, transpar=im.coordinates_transfer_para, pardict = self.pardict)
                else:
                    imm = jnp.sum(imm0)*jnp.array([[1.]])
                    for psfmodel in self.psfmodel_list:
                        zred_temp   = psfmodel.redshift
                        waveintrin  = wavegrid / (1+zred_temp)
                        agn_sed     = psfmodel.fiducial_sed(waveintrin, pardict=self.pardict)
                        imm += psfmodel.generate_SED([resp_tem], wavegrid, agn_sed)[0]

                imm *= im.phys_to_counts_rate
                imm += jnp.ones_like(imm)*self.pardict['sky_{0}'.format(im.imglabel)]
                im.model_image = imm
                resi = (im.cut_image-imm)/im.cut_sigma_image
                resi = jnp.where(im.cut_mask_image,0.,resi)
                image_atlas = GSdata[GSdata.listindex[loop_image]]
                image_atlas.img_residual[GSdata.atlasindex[loop_image]] = jnp.sum(resi*resi)
        

        pltwave = None
        Sedcomp = []
        Sedlabel = []     
        if len(image_atlas.grism_image_list)==0:
            wavephot = []
            fluxphot = []
            modelphot = []
            errorphot = []
            for loop in range(len(self.GSdata.allbands)):
                im = self.GSdata.get_image(loop)
                sky = self.pardict['sky_{0}'.format(im.imglabel)]
                wavephot.append(gsutils.effective_wave[im.band])
                fluxphot.append(jnp.sum(jnp.where(im.cut_mask_image, 0., im.cut_image-sky))/im.phys_to_counts_rate)
                modelphot.append(jnp.sum(jnp.where(im.cut_mask_image, 0., im.model_image-sky))/im.phys_to_counts_rate)
                if im.photopoint:
                    errorphot.append(jnp.sum(im.cut_sigma_image)/im.phys_to_counts_rate)
                else:## TODO: find a method to account the correlation of pixels.
                    emap = jnp.where(im.cut_mask_image, 0., im.cut_sigma_image)
                    errorphot.append(jnp.sqrt(jnp.sum(emap**2))/im.phys_to_counts_rate)
            wavephot = jnp.array(wavephot)
            fluxphot = jnp.array(fluxphot)
            errorphot = jnp.array(errorphot)

        zred_list   = []
        wavegrid    = self.GSdata.wavegrid
        for loop2,model in enumerate(self.model_list):

            if self.mtype_list[loop2] == 'galaxy':
                zred    = model.redshift
                waveintrin  = wavegrid / (1 + zred)
                f_subs_list, sub_labels     = model.fiducial_sed(waveintrin, pardict=self.pardict, button=2)
                for loop, fl in enumerate(f_subs_list):
                    Sedcomp.append(fl)
                    Sedlabel.append(sub_labels[loop])   
                    zred_list.append(zred)
            elif self.mtype_list[loop2] == 'agn':
                zred    = model.redshift
                waveintrin  = wavegrid / (1+zred)
                fl  = model.fiducial_sed(waveintrin, pardict=self.pardict)
                Sedcomp.append(fl)
                Sedlabel.append(model.prefix)
                zred_list.append(zred)
            elif self.mtype_list[loop2] == 'star':
                zred    = 0
                fl = model.fiducial_sed(wavegrid, pardict = self.pardict)
                Sedcomp.append(fl)
                Sedlabel.append(model.prefix)
                zred_list.append(zred)
        pltwave = [wavephot,fluxphot,errorphot,modelphot]
        return pltwave, Sedcomp, Sedlabel, zred_list



class sedfitter(gsfitter):
    """
    Photometric SED fitter without spatial information. (with optional spectroscopy)
    
    Fits multi-band photometry to galaxy and AGN SED models. Fastest 
    fitting mode - uses integrated photometry without morphology. Ideal 
    for large surveys or unresolved sources.
    
    Inherits all optimization methods from gsfitter.
    
    See Also
    --------
    imagefitter_SED : SED fitting with resolved morphology
    vdfitter : Spectroscopic fitting
    """
    # specific for photometric SED fitting
    def cal_residual(self,theta):
        for loop,key in enumerate(self.varnames):
            self.pardict[key] = theta[loop] # theta is jit traced -> self.pardict 
        if self.UC_func is not None:
            self.UC_func(self.pardict)
        for loop, model in enumerate(self.model_list):
            if self.mtype_list[loop] == 'galaxy':
                if self.lmParameters['z_{0}'.format(model.name)].vary: ## TODO: check z-free mode later.
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.name)]) # update redshift and distance parameters.
                for com in model.subnames:
                    model.update_subC_para(com,self.pardict)
            elif self.mtype_list[loop] == 'agn':
                if self.lmParameters['z_{0}'.format(model.prefix)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.prefix)])
        
        atnames     = self.GSdata.atlasnamelist
        resps       = self.GSdata.resps ## NOTE: have been resampled. 
        wavegrid    = self.GSdata.wavegrid
        photos      = self.GSdata.photos
        photoerrs   = self.GSdata.photoerrs
        sedm = jnp.zeros_like(photos, dtype=jnp.float32)

        for loop2, model in enumerate(self.model_list):    
            if self.mtype_list[loop2] == 'agn':
                normfactor = jnp.array([self.pardict[('Ni_{0}_{1}'.format(model.prefix,atname))] for atname in atnames], dtype=jnp.float32)
                zred_temp   = model.redshift
                waveintrin  = wavegrid / (1+zred_temp)
                agn_sed_temp    = model.fiducial_sed(waveintrin, pardict=self.pardict, )
                sedm   += normfactor*model.generate_SED(resps, wavegrid, agn_sed_temp)  

            elif self.mtype_list[loop2] == "galaxy":
                zred_temp    = model.redshift
                waveintrin  = wavegrid / (1+zred_temp)
                ftot_obs, _    =  model.fiducial_sed(waveintrin, pardict=self.pardict, button=0)
                sedm += model.generate_SED(resps, wavegrid, ftot_obs)


        # for loop2, model2 in enumerate(self.model_list):   
        #     if self.mtype_list[loop2] != "agn":
        #         continue
        #     agn_name    = model2.prefix
        #     logfagn     = self.pardict["agn_logM{0}".format(agn_name)] # borrow this param, which is not used in `BP` agn model, as container to pass fagn_host.
        #     plC1    = self.pardict['{0}plC1'.format(agn_name)]
        #     for loop3, model3 in enumerate(self.model_list):
        #         if self.mtype_list[loop3] != "galaxy":
        #             continue # first calculate galaxy 
        #         zred_temp   = model3.redshift
        #         waveintrin  = wavegrid / (1+zred_temp)  
        #         # subC_key    = model3.subCs[0] # only one subprofile for one galaxy.
        #         subC_key    = model3.subnames[0]
        #         ftot_obs, _     =  model3.fiducial_sed(waveintrin, pardict=self.pardict, button=0)
        #         sedm    += model3.generate_SED(resps, wavegrid, ftot_obs) 

        #         ## reset the dust2 as 0 follow Wang+24 
        #         pardict_temp    = self.pardict.copy() # .copy() feasible here??
        #         pardict_temp["{0}_dust2".format(subC_key)]  = 0.
        #         ftot_obs_temp ,_    = model3.fiducial_sed(waveintrin, pardict=pardict_temp, button=0)
        #         ftot_temp   = ftot_obs_temp * (1+zred_temp)**3 # negligible milky way extinction for rest frame jwst coverage. 
        #         fhost5500   = jnp.interp(5500, waveintrin, ftot_temp,) # first convert to rest frame, so here waveintrin instead of wavegrid.

        #         # invert brokenpower red branch: F(5500)=A*(5500/5100)^(-plC1)/5100,
        #         # A = 10^(logL5100-38), require F(5500) = f * Fhost5500
        #         self.pardict['{0}logL5100'.format(agn_name)] = (
        #             logfagn + jnp.log10(fhost5500) + jnp.log10(5100.)
        #             + plC1*jnp.log10(5500./5100.) + 38.)
                
        #     # then calculate AGN model.
        #     normfactor = jnp.array([self.pardict[('Ni_{0}_{1}'.format(agn_name,atname))] for atname in atnames], dtype=jnp.float32)
        #     zred_temp   = model2.redshift
        #     waveintrin  = wavegrid / (1+zred_temp)
        #     agn_sed_temp    = model2.fiducial_sed(waveintrin, pardict=self.pardict, )
        #     sedm   += normfactor*model2.generate_SED(resps, wavegrid, agn_sed_temp)  

        
        residu_flat = (photos-sedm)/photoerrs

        return residu_flat
    
    def cal_model_image(self, pardict=None, ndisagn=False, wrange=[0.85,1.15]):

        '''
        only used for gsutils display....
        '''
        if pardict is not None:
            self.pardict = pardict
        for loop, model in enumerate(self.model_list):
            if self.mtype_list[loop] == 'galaxy':
                if self.lmParameters['z_{0}'.format(model.name)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.name)])
                for com in model.subnames:
                    model.update_subC_para(com,self.pardict)
            elif self.mtype_list[loop] == 'agn':
                if self.lmParameters['z_{0}'.format(model.prefix)].vary:
                    model.reset_info(self.pardict, z = self.pardict['z_{0}'.format(model.prefix)])
        
        pltwave     = []
        Sedcomp     = []
        Sedlabel    = []
        wavephot    = []
        fluxphot    = []
        modelphot   = []
        errorphot   = []

        atnames     = self.GSdata.atlasnamelist
        wavegrid    = self.GSdata.wavegrid
        resps   = self.GSdata.resps
        photos  = self.GSdata.photos

        sedm = jnp.zeros_like(photos, dtype=jnp.float32)
        for loop2, model in enumerate(self.model_list):
            if self.mtype_list[loop2] == 'agn':
                normfactor  = jnp.array([self.pardict[('Ni_{0}_{1}'.format(model.prefix,atname))] for atname in atnames], dtype=jnp.float32)
                zred_temp   = model.redshift
                waveintrin  = wavegrid / (1+zred_temp)
                agn_sed     = model.fiducial_sed(waveintrin, pardict=self.pardict, ) 
                sedm    += normfactor*model.generate_SED(resps, wavegrid, agn_sed)  
            else:
                zred_temp   = model.redshift
                waveintrin  = wavegrid / (1+zred_temp)
                ftot_obs, _    =  model.fiducial_sed(waveintrin, pardict=self.pardict, button=0)
                sedm += model.generate_SED(resps, wavegrid, ftot_obs) ## get sed.
        modelphot   = sedm

        for image_atlas in self.GSdata.image_atlas_list:

            for loopb, band in enumerate(image_atlas.band_list):
                wavephot.append(gsutils.effective_wave[band])
                fluxphot.append(image_atlas.photos[loopb])
                errorphot.append(image_atlas.photoerrs[loopb])
                sed_index   = len(fluxphot) - 1
                image_atlas.img_residual[loopb]     = ((fluxphot[sed_index] - modelphot[sed_index]) / errorphot[sed_index]) ** 2


        zred_list   = []
        wavegrid    = self.GSdata.wavegrid 

        for loop2,model in enumerate(self.model_list):
            
            if self.mtype_list[loop2] == 'galaxy':
                zred    = model.redshift
                waveintrin  = wavegrid / (1 + zred)
                f_subs_list, sub_labels     = model.fiducial_sed(waveintrin, pardict=self.pardict, button=2)
                for loop, fl in enumerate(f_subs_list):
                    Sedcomp.append(fl)
                    Sedlabel.append(sub_labels[loop])   
                    zred_list.append(zred)
            elif self.mtype_list[loop2] == 'agn':
                zred    = model.redshift
                waveintrin  = wavegrid / (1+zred)
                fl  = model.fiducial_sed(waveintrin, pardict=self.pardict)
                Sedcomp.append(fl)
                Sedlabel.append(model.prefix)
                zred_list.append(zred)
            elif self.mtype_list[loop2] == 'star':
                zred    = 0
                fl = model.fiducial_sed(wavegrid, pardict = self.pardict)
                Sedcomp.append(fl)
                Sedlabel.append(model.prefix)
                zred_list.append(zred)
        pltwave = [wavephot,fluxphot,errorphot,modelphot]
        return pltwave, Sedcomp, Sedlabel, zred_list
