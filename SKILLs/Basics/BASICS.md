# This is a markdown recording some kind and necessary notes for galfits usage.

## 1. SPS model 

**host_conti_fsps_fsps_mist_miles_chab_neb_freeGasz_oldfsps.npz**
- Stellar library: MIST + MILES
- IMF: Chabrier(2003)
- fsps version: 0.4.8
- $Z_{sol}$ = 0.142 # NOTE: a little bit werid...



## 2. --readsummary

NOTE: `--readsummary` a gssummary file will overwrite the parameter vary range from [`min` - `max`] to 
[`init - 10*step` - `init + 10*step`]. 


## 3. Gaussian prior 

NOTE: Don't forget to input `--baysian`!