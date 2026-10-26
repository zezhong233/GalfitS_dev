# This is a todo list for GalfitS framework refinement. 


## 1. BUGs to fix

- [v]  `generate_image` & `generate_subcomponent_image` of <class Galaxy>.
    - [ ] fix that also for <class AGN>. 
- [v] Fix the jax +- 0.5 pixel offset. 
- [ ] find ghost point source bug. 
- [ ] change the params index format.
- [ ] the Extinction part of <class AGN> -> `fiducial_sed()`. 
- [ ] <class star> should be aligned with Galaxy and AGN.



## 2. optimization points.

- [ ] find `prospector` logM - metallicity relation.
- [ ] generate `generate_group_image` image for AGN model too. 
    - so `generate_image` is only used for user manually plot, then I can directly time `self.phys_to_count_rate` to image in flux to get True image in count rate.
- [ ] delete unnecessary methods of <class Galaxy>, like `make_nebular'.
- [ ] add other fitters and fitting methods back to <gsfit.py>.
- [ ] parameters pass routine in `fiducial_sed`.
- [ ] gredient map.


## 3. Necessary Functions

- [ ] Automatically derive geometry parameters from <Image-photometry> fitter to feed image-sed fitting.
    - `--readsummary objxxx_geo.gssummary`
- [ ] Convenient wrapped parameter settings or commands for different fitters, like what ruancun did. 
- [ ] a python format lyric. 
- [v] we should summary the naming format for every parameters. 


## 4. mock test 
- [ ] Alighment test
    - targ: confirm the necessary alignment accuracy for PSF fitting and extended sersic fitting.
    - compare the BIC of `xcen`, `ycen` free/fix of two fitting?
        - how to define correct BIC for image with correlation?

- [ ] sed fitting mock
    - targ: see whether and to what extent current fitting setting can revover input parameters. 

- [ ] Z-free fitting.
    - Whether zred-free fitting is not TRUELY not robust?