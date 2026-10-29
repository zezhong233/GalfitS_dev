# GalfitS Parameter Naming Format (need to be checked again by MAN!)

> Hand-written. How internal parameter names (`lmParameters` / `pardict`
> keys) are built from config names. Verified against
> `gsutils.read_config_file` (`src/galfits/gsutils.py`) — every pattern below
> comes from an actual `par_total.add(...)` call.
>
> Companion doc: `PARAM_GUIDE.md` (what each config key means).

## Name sources

| Config key | Provides the name | Example name |
|---|---|---|
| `Px1)` | profile / component name `{p}` | `obj0` |
| `Gx1)` | galaxy name `{g}` | `mygal` |
| `Nx1)` | AGN (nucleus) name `{n}` | — |
| `Fx1)` | foreground star name `{f}` | — |
| image letter | image label `{img}` | `a`, `b`, … |
| `Ax1)` | atlas name `{atlas}` | `jwst` |

## Two conventions (mixed — historical)

1. **`{name}_{key}`** — e.g. `obj0_Re`, `obj0_dust2`.
2. **`{key}_{name}`** — e.g., `logM_obj0`, `z_mygal`, `sky_a`.

⚠️ No separator at all in most AGN names: `agn_logM{n}` → `agn_logMnu1`,
`{n}torus_a` → `nu1torus_a`. Grep before you assume.

---

## Profile-owned parameters (name from `Px1)`)

### Morphology — pattern `{p}_…`

| Parameter name | Config key | Definition |
|---|---|---|
| `{p}_xcen`, `{p}_ycen` | `Px3)`, `Px4)` | center offset [arcsec] |
| `{p}_Re` | `Px5)` | effective radius [arcsec] |
| `{p}_n` | `Px6)` | Sérsic index |
| `{p}_ang` | `Px7)` | position angle [deg] |
| `{p}_axrat` | `Px8)` | axis ratio b/a |
| `{p}_r_in` | `Px17)` | inner radius (e.g. `sersic_r`) |
| `{p}_r_out` | `Px18)` | outer radius (`sersic_b`, ring) |
| `{p}_width` / `{p}_alpha_rc` | `Px19)` | ring width / arm RC slope <TODO> |
| `{p}_theta_out` | `Px20)` | outer PA (arms) <TODO> |
| `{p}_m`, `{p}_am`, `{p}_theta_m` | `Px21)`–`Px23)` | Fourier mode, amplitude, phase (`m` forced `vary=False`) |
| `{p}_rs`, `{p}_hs`, `{p}_beta`, `{p}_alpha`, `{p}_sig`, `{p}_Rout`, `{p}_i_arm` | profile-type specific | <TODO: map per type> |
| `{p}_{par}` | `PxT2)` names | table-profile params (free-form names) |

### Stellar population / SED — pattern `{p}_…`

| Parameter name | Config key | Definition |
|---|---|---|
| `{p}_f_cont_bin{i}` | `Px9)` (SFH `bins`) | log SFR fraction in age bin i |
| `{p}_f_cont_value` / `_in` / `_out` | `Px9)` (non-bins) | SFR fraction, global / inner / outer |
| `{p}_age_value` / `_in` / `_out` | `Px10)` | stellar age [Gyr] |
| `{p}_Z_value` / `_in` / `_out` | `Px11)` | stellar metallicity |
| `{p}_Av_value` / `_in` / `_out` | `Px12)` | extinction Av [mag] |
| `{p}_sigma` | `Px13)` | velocity dispersion [km/s] (`_in/_out` linear mode exists but is commented out) |
| `{p}_SFH_p{i}` | (parametric SFH) | SFH coefficients <TODO> |
| `{p}_Zneb_value` / `_in` / `_out` | `Px33)` | nebular metallicity |
| `{p}_dust2` | `Px34)` | diffuse-dust attenuation |
| `{p}_dust_ratio` | `Px35)` | dust1/dust2 ratio |
| `{p}_dust_index` | `Px36)` | diffuse-dust slope |
| `ezlogn{i}_{p}` | auto when `Px27)='Z'` | 18 eazy-template log norms (i = 1..18) |

### Pattern `{key}_{p}`

| Parameter name | Config key | Definition |
|---|---|---|
| `logM_{p}` | `Px14)` | log10 stellar mass (fixed if `Px27)='Z'`) |
| `logU_{p}` | `Px16)` | nebular ionization parameter |
| `AVbump_{p}` | `Px26)` | 2175 Å bump amplitude |
| `logMdust_{p}` | `Px28)` | log10 dust mass |
| `Umin_{p}` | `Px29)` | min radiation field |
| `qPAH_{p}` | `Px30)` | PAH mass fraction |
| `alphad_{p}` | `Px31)` | radiation-field power-law slope |
| `gammad_{p}` | `Px32)` | SF-region illuminated fraction |

### Per-band (auto-added in photometry mode, `gsutils.py:1277-1278`)

| Parameter name | Definition |
|---|---|
| `logNorm_{p}_{band}` | per-band log flux normalization (free) |
| `Mag_{p}_{band}` | AB magnitude — derived via `expr = magzp − 2.5·logM_{p} − 2.5·logNorm_{p}_{band}`, always `vary=False` |

---

## Galaxy-owned parameters (name from `Gx1)`) — pattern `{key}_{g}`

| Parameter name | Config key | Definition |
|---|---|---|
| `z_{g}` | `Gx3)` | galaxy redshift |
| `Ns_{g}_{specprefix}` | `Gx5)` | spectrum normalization per spectrum |
| `r0_{g}`, `rt_{g}`, `v0_{g}`, `vc_{g}`, `LineSigma_{g}` | `Gx8)` (optional) | narrow-line kinematics <TODO: physical meaning of each> |
| `scale_{g}` | <TODO: key> | spectrum flux scale <TODO> |

---

## AGN-owned parameters (name from `Nx1)`)

⚠️ Inconsistent separators: `agn_*{n}` (no `_` before name) vs `{n}torus_*`
(no separator at all) vs `{n}_logL`.

| Parameter name | Definition |
|---|---|
| `agn_logM{n}` | log10 BH mass |
| `agn_logLedd{n}` | log10 Eddington ratio |
| `agn_spin{n}` | BH spin |
| `agn_amax{n}` | max torus/disk radius param <TODO> |
| `agn_Av{n}` | AGN extinction Av |
| `z_{n}` | AGN redshift (own, independent of galaxy) |
| `xcen_{n}`, `ycen_{n}` | AGN center (note: key-name order, unlike profiles!) |
| `Ns_{n}_{specprefix}` | AGN spectrum normalization (`Nx19)`) |
| `Ni_{n}_{atlas}` | AGN image normalization per atlas (`Nx26)`, `Ax8)`/`Ax9)`) |
| `{n}logL5100` | log luminosity at 5100 Å |
| `{n}Lhotdust`, `{n}Thotdust` | hot-dust luminosity / temperature |
| `{n}plC`, `{n}plC1`, `{n}plC2` | power-law continuum coefficients |
| `{n}torus_logL`, `{n}torus_N0`, `{n}torus_a`, `{n}torus_h`, `{n}torus_i` | torus model params <TODO: definitions> |
| `{n}_{x}_logL` | line luminosity `{x}` <TODO: key source> |

## Star-owned parameters (name from `Fx1)`)

| Parameter name | Definition |
|---|---|
| `xcen_{f}`, `ycen_{f}` | star center (key-name order, like AGN) |
| `{f}_Teff`, `{f}_logg`, `{f}_logZ` | stellar atmosphere params |
| `{f}_logL38`, `{f}_{x}_logL` | luminosity params <TODO> |

## Image-owned parameters (name = image label `{img}`)

| Parameter name | Config key | Definition |
|---|---|---|
| `sky_{img}` | `Ix12)` | sky background level |
| `csx_{img}`, `csy_{img}` | `Ix14)` | astrometric shift [pixels] |

---

## Prior-file parameters (from `.prior` sections, not the config)

Prior file has its own sections: `AGN{x}`, `MSR{x}`, `MAR{x}`, `MMR{x}`,
`SFH{x}`, `SFH_T{x}`, `GP` — each `{x}1)` gives the `pname` used below
(`gsutils.py:1601-1802`).

| Parameter name | Definition |
|---|---|
| `Sigma_bHa_{p}`, `Sigma_bHb_{p}`, `Sigma_nOIII_{p}` | scatter around broad-Hα/Hβ/[OIII] scaling relations |
| `Sigma_MZR_{p}`, `Sigma_MSR_{p}`, `Sigma_MAR_{p}`, `Sigma_gm_{p}` | scatter: mass–metallicity / mass–SFR / MAR / gas-mass relations |
| `RFeII_{p}_op`, `RFeII_{p}_uv` | optical / UV Fe II strength ratios |

<TODO: document prior-file section syntax fully — keys `GP)`,
`SFH_Tx1/2)`, `MSR/MAR/MMR/SFH/AGN` blocks>

---

## Quick lookup cheatsheet

```
morphology : {p}_xcen {p}_ycen {p}_Re {p}_n {p}_ang {p}_axrat
stars/SFH  : {p}_f_cont_* {p}_age_* {p}_Z_* logM_{p} {p}_sigma
dust       : {p}_Av_* {p}_dust2 {p}_dust_ratio {p}_dust_index AVbump_{p} logMdust_{p}
nebular    : logU_{p} {p}_Zneb_*
dust emis. : Umin_{p} qPAH_{p} alphad_{p} gammad_{p}
per-band   : logNorm_{p}_{band}  Mag_{p}_{band}(derived)
galaxy     : z_{g} Ns_{g}_{spec} [r0 rt v0 vc LineSigma]_{g}
AGN        : agn_*{n} z_{n} xcen_{n} ycen_{n} Ns_{n}_{spec} Ni_{n}_{atlas} {n}torus_*
star       : xcen_{f} ycen_{f} {f}_Teff {f}_logg {f}_logZ
image      : sky_{img} csx_{img} csy_{img}
```
