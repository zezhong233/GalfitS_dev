# GalfitS Config Parameter Guide

> Hand-written guide (maintained by humans, never regenerated).
> Sources: `Examples/LRD_RUBINS-CEERS-49140/lyric_writing.py` (working
> example), verified against the parser `gsutils.read_config_file`
> (`src/galfits/gsutils.py:634`).

## Conventions

- **Key format:** `X<n><k>) value` — `X` = section letter, `<n>` = item letter
  (`a`, `b`, … = 1st, 2nd, … item of that section), number = field index.
  Example: `Ia3)` = first image's 3rd field (sigma image).
- **Values are Python literals** — they are `eval()`'d
  (`gsutils.parse_config_file`, `gsutils.py:415-419`). Use `[...]`, numbers,
  `'strings'`.
- **Fit-parameter list format:** `[value, min, max, step, vary]`
  (`value` initial; `min`/`max` bounds; `step` = lmfit `brute_step`;
  `vary`: 1 = free, 0 = fixed). Confirmed at `gsutils.py:962-972,1181`.
- `#` starts a comment.

---

## Region — `Rx)` (exactly one per config)

| Key | Definition |
|---|---|
| `R1)` | Target name |
| `R2)` | Sky coordinate `[RA, Dec]` in degrees. |
| `R3)` | Redshift of the region (scalar). |

---

## Image — `Ix)` (one block per band/image)

One letter per image: `Ia…)` first, `Ib…)` second, etc.
Parser mapping verified at `gsutils.py:952`
(`['image','band','sigma','psf','psf_sampling','mask','unit','fitting_area','conversion','magzp','skymodel','skyparameter','shift','shift_param','use_sed']`).

| Key | Definition |
|---|---|
| `Ix1)` | Science image: `[fits_path, extension]`. |
| `Ix2)` | Band / filter name. Name format see `effective_wave` in gsutils.py. e.g. `nircam_f115w` |
| `Ix3)` | Sigma (1σ error) image: `[fits_path, extension, scale]`. `scale` is a arbitary scale factor for whole sigma image. default `1` if not argued.|
| `Ix4)` | PSF image: `[fits_path, extension]`. |
| `Ix5)` | PSF fine-sampling (oversampling) factor relative to the data. |
| `Ix6)` | Mask image: `[fits_path, extension]`; `[Noimg,0]` = no mask. |
| `Ix7)` | Image unit string (e.g. `MJy/sr`, `cR`). |
| `Ix8)` | Cutout half-size of the fitting region [arcsec]; `-1` = no cut. |
| `Ix9)` | Conversion from image unit to F_λ; `-1` = default (from `Ix10)`). For pure-SED mock images use `1`. |
| `Ix10)` | Photometric magnitude zeropoint. |
| `Ix11)` | Sky model, e.g. `uniform`. |
| `Ix12)` | Sky parameter(s): list of 5-element fit lists, e.g. `[[0,-1e5,1e5,0.1,0]]`. |
| `Ix13)` | Allow relative astrometric shifting of this image: 1 = yes, 0 = no. |
| `Ix14)` | Shift parameters `[[x…],[y…]]`, each a 5-element fit list (pixels). |
| `Ix15)` | Use SED information for this band: 1 = yes, 0 = no. |

---

## ImageAtlas — `Ax)` (groups images + spectra into one fitting unit)

Verified at `gsutils.py:1049-1098`.

| Key | Definition |
|---|---|
| `Ax1)` | Atlas name (string). |
| `Ax2)` | List of image letters in this atlas, e.g. `['a','b','c']`. |
| `Ax3)` | Images share the same pixel grid: 1 = yes, 0 = no. |
| `Ax4)` | Link relative shiftings across the atlas: 1 = yes, 0 = no. |
| `Ax5)` | Spectra belonging to this atlas (list of spectrum letters; `[]` = none). |
| `Ax6)` | Aperture sizes for the spectra (list; `[]` = none). |
| `Ax7)` | Reference image list (`[]` = none). <TODO: exact semantics> |
| `Ax8)` | *(optional)* vary flag for the AGN image normalization in this atlas. |
| `Ax9)` | *(optional)* initial value of the AGN image normalization in this atlas. |

---

## Profile — `Px)` (one structural component, e.g. one Sérsic)

Verified at `gsutils.py:1158-1400`. All fit params use the 5-element list
unless noted. "1 vs 2 entries" below: many SED params accept either one
5-element list (global value → param suffix `_value`) or two (radial inner/
outer pair → suffixes `_in` / `_out`).

| Key | Definition |
|---|---|
| `Px1)` | Component name (string, e.g. `obj0`). |
| `Px2)` | Profile type, e.g. `sersic`. <TODO: full list of valid types> |
| `Px3)` | x-center offset [arcsec]. |
| `Px4)` | y-center offset [arcsec]. |
| `Px5)` | Effective (half-light) radius Re [arcsec]. |
| `Px6)` | Sérsic index n. |
| `Px7)` | Position angle [deg] (up = 0, left = 90). |
| `Px8)` | Axis ratio q = b/a (1 = round). |
| `Px9)` | SFH fractions. SFH `bins`: one 5-list per age bin (log fraction of contemporary SFR). Otherwise 1 entry → `f_cont_value`, 2 → `f_cont_in/out`. |
| `Px10)` | Stellar ages [Gyr]. `bins`: the age-bin list (e.g. from `z_to_agebin`). Otherwise 1 → `age_value`, 2 → `age_in/out`. |
| `Px11)` | Stellar metallicity Z (solar ≈ 0.0142). 1 → `Z_value`, 2 → `Z_in/out`. |
| `Px12)` | Dust extinction Av [mag]. 1 → `Av_value`, 2 → `Av_in/out`. |
| `Px13)` | Stellar velocity dispersion σ [km/s] (single 5-list). |
| `Px14)` | log10 stellar mass [M☉]. (Forced fixed when `Px27) = 'Z'`.) |
| `Px15)` | SFH type: `bins` / `burst` / `conti` … <TODO: full list> |
| `Px16)` | logU — nebular ionization parameter. |
| `Px17)`–`Px25)` | Profile-type-dependent extras (ring radius `r_out`; Fourier/spiral-arm `r_in`, `width`/`alpha_rc`, `theta_out`, mode `m`, amplitude `am`, phase `theta_m`). See `gsutils.py:1296-1325`. <TODO: document per profile type> |
| `Px26)` | 2175 Å bump amplitude on the extinction curve (`AVbump`). |
| `Px27)` | SED model mode: `0` stellar; `1` stellar+nebular; `2` stellar+dust emission; `3` stellar+nebular+dust; `'Z'` eazy-style (logM fixed). |
| `Px28)` | log10 dust mass [M☉]. |
| `Px29)` | Umin — minimum starlight radiation field (dust emission). |
| `Px30)` | qPAH — PAH mass fraction [%]. |
| `Px31)` | α — power-law slope of the radiation-field distribution. |
| `Px32)` | γ — mass fraction illuminated by star-forming regions. |
| `Px33)` | Nebular metallicity Zneb. 1 → `Zneb_value`, 2 → `Zneb_in/out`. |
| `Px34)` | dust2 — diffuse-dust attenuation optical depth (~N(0.3, 1) on [0,4]). |
| `Px35)` | dust_ratio — birth-cloud vs diffuse dust (dust1/dust2, ~1). |
| `Px36)` | dust_index — diffuse-dust power-law index (~U[-1, 0.4]). |
| `PxT1)`–`PxT3)` | Table-based profile: table path, parameter names, initial values (`gsutils.py:1176-1181`). <TODO> |

---

## Galaxy — `Gx)` (assembles profiles into a physical galaxy)

Verified at `gsutils.py:1919-1959`.

| Key | Definition |
|---|---|
| `Gx1)` | Galaxy name (string). |
| `Gx2)` | List of profile letters composing this galaxy, e.g. `['a']`. |
| `Gx3)` | Galaxy redshift (5-element fit list). |
| `Gx4)` | E(B−V) of Galactic (Milky Way) foreground reddening (scalar). |
| `Gx5)` | Spectrum normalization `Ns` for images+spectrum joint fitting (5-element list). |
| `Gx6)` | Narrow emission lines to include in the nebular component (list; `[]` = none). |
| `Gx7)` | Number of components per narrow line (`nnarrow`). |
| `Gx8)` | *(optional)* line kinematics: `[r0, rt, v0, vc, LineSigma]`, each a 5-element fit list (`gsutils.py:1939-1943`). <TODO: meaning of each> |

---

## Other sections (stubs — not exercised by the lyric example)

| Section | Item | Status |
|---|---|---|
| `Nx)` | Nucleus / AGN component (keys `N1)`–`N27)`; creates `AGN` model, `gsutils.py:1417-1432`) | <TODO: document> |
| `Fx)` | Foreground star (creates `Star` model, `gsutils.py:1541-1546`) | <TODO> |
| `Sx)` | Spectrum definition (referenced from `Ax5)`) | <TODO> |

## Prior file (separate `.prior` file, via `--priorpath`)

Not part of the config; written alongside in the example
(`lyric_writing.py:138-143, 182-186`):

| Key | Definition |
|---|---|
| `SFH_Tx1)` | Profile name the SFH prior applies to. |
| `SFH_Tx2)` | Prior type, e.g. `cosmic` (cosmic-SFR prior). |
| `GP)` | Gaussian priors: list of `[param_name, mean, std]`-style entries. <TODO: exact format> |
