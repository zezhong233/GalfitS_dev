# This is a todo list for GalfitS framework refinement. 


## 1. BUGs to fix

- [v]  `generate_image` & `generate_subcomponent_image` of <class Galaxy>.
    - [ ] fix that also for <class AGN>. 
- [ ] the Extinction part of <class AGN> -> `fiducial_sed()`. 
- [ ] <class star> should be aligned with Galaxy and AGN.
- [ ] Fix the jax +- 0.5 pixel offset. 


## 2. optimization points.

- [ ] generate `generate_group_image` image for AGN model too. 
    - so `generate_image` is only used for user manually plot, then I can directly time `self.phys_to_count_rate` to image in flux to get True image in count rate.
- [ ] delete unnecessary methods of <class Galaxy>, like `make_nebular'.
- [ ] add other fitters and fitting methods back to <gsfit.py>.
- [ ] parameters pass routine in `fiducial_sed`.
- [ ] gredient map.
- [ ] Check whether zred-free fitting is not TRUELY not robust.  


## 3. Necessary Functions

- [ ] Automatically derive geometry parameters from <Image-photometry> fitter to feed image-sed fitting.
    - `--readsummary objxxx_geo.gssummary`
- [ ] Convenient wrapped parameter settings or commands for different fitters, like what ruancun did. 
- [ ] we should summary the naming format for every parameters, like {}