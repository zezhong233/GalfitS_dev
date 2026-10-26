# Prospector-β (Wang+2024 UNCOVER) parameter settings

Extracted from `files/prospector_catalog/params_prosp_fsps.py` (`params_fsps_phisfh`,
`free_gas_logu=False` — the configuration used for the released UNCOVER DR4 SPS catalog).
Solar metallicity of the FSPS data used: **zsun = 0.0142** (old FSPS/MIST data).

## Free parameters (18 = 9 nzsfh + 9 physical)

| # | prospector param | free/fix | init | prior | range | galfits equivalent |
|---|---|---|---|---|---|---|
| 1 | `zred` | free (inside `nzsfh`) | 0.5 | Uniform | 1e-3 – 20 | `z_mygal` (usually fix to spec-z) |
| 2 | `logmass` (formed mass, M☉) | free (inside `nzsfh`) | 8.0 | cosmic mass function (truncated) | 6.0 – 12.5 | `logM` + `f_cont_bin1..7` |
| 3 | `logzsol` (log10 Z/Z☉) | free (inside `nzsfh`) | -0.5 | mass–metallicity TruncNormal(μ=MZR(logM), σ=MZR scale) | -1.98 – 0.19 | `Z_value` = 10^logzsol × 0.0142 |
| 4–9 | `logsfr_ratios_1..6` | free (inside `nzsfh`) | 0.0 | StudentT(df=2, loc=SFRD(z,M), scale=0.3, truncated) | -5 – 5 | → `f_cont` via SFH construction |
| 10 | `dust2` | free | 0.0 | ClippedNormal(mean=0.3, σ=1.0) | 0 – 4 | `dust2` |
| 11 | `dust_index` | free | 0.7 ⚠️(1) | Uniform | -1.0 – 0.4 | `dust_index` |
| 12 | `dust1_fraction` | free | 1.0 | ClippedNormal(mean=1.0, σ=0.3) | 0 – 2 | `dust_ratio` |
| 13 | `log_fagn` | free | -7e-5 | Uniform | -5 – 0.477 (=log10 3) | — (no galfits counterpart) |
| 14 | `log_agn_tau` | free | 1.301 (=log10 20) | Uniform | 0.699 – 2.176 (log10 5–150) | — (no galfits counterpart) |
| 15 | `gas_logz` | free | -0.5 | Uniform | -2.0 – 0.5 | `Zneb_value` = 10^gas_logz × 0.0142 |
| 16 | `duste_qpah` | free | 2.0 | ClippedNormal(mean=2.0, σ=2.0) | 0 – 7 | `qPAH` |
| 17 | `duste_umin` | free | 1.0 | ClippedNormal(mean=1.0, σ=10.0) | 0.1 – 25 | `Umin` |
| 18 | `log_duste_gamma` | free | -2.0 | ClippedNormal(mean=-2.0, σ=1.0) | -4 – 0 | `gammad` = 10^log_duste_gamma |

## Fixed parameters

| prospector param | free/fix | value | note |
|---|---|---|---|
| `gas_logu` | **fix** | **-1.0** | paper text says -2.0 — the released fits used -1.0 |
| `imf_type` | fix | 1 | Chabrier |
| `sfh` | fix | 3 | tabular SFH, 7 bins |
| `dust_type` | fix | 4 | Kriek & Conroy 2013 |
| `pmetals` | fix | -99 | with `zcontinuous=2`: triangular smoothing + log-Z interp |
| `zcontinuous` | fix | 2 | (build-time FSPS flag) |
| `sigma_smooth` | fix | 0.0 | no velocity smoothing |
| `add_igm_absorption` | fix | True | Madau 1995 |
| `add_agb_dust_model` | fix | True | |
| `add_neb_emission` | fix | True | |
| `add_neb_continuum` | fix | True | |
| `add_agn_dust` | fix | True | Nenkova torus |
| `mass_units` | fix | 'mformed' | mass = formed, not surviving |
| `peraa` | fix | False | |
| MW extinction | — | none | set `ebv=0` in galfits |
| cosmology | — | WMAP9 | H0=69.32, Ωm=0.2865 (agebins + d_L) |

## Derived (depends_on) parameters

| param | definition |
|---|---|
| `dust1` | = `dust1_fraction` × `dust2` (birth-cloud attenuation, t ≤ 10^7.025 yr) |
| `fagn` | = 10^log_fagn |
| `agn_tau` | = 10^log_agn_tau |
| `duste_gamma` | = 10^log_duste_gamma |
| `mass` (7 bins) | = `logsfr_ratios_to_masses(logmass, logsfr_ratios, agebins)` |
| `agebins` (7) | = prospector `zred_to_agebins` (WMAP9); z≤3: [0, 10^7.4771, 10^8.0] + logspace(8, log10(0.9·tuniv), 5) + tuniv |

(1) `dust_index` 的 `init=0.7` 是 Wang 代码中的笔误（超出 prior 范围 [-1.0, 0.4]，应为 **-0.7**，即 Kriek & Conroy 2013 标准值）。dynesty 采样不使用 init，因此发布目录未受影响（链中实测范围恰好 [-1.0, 0.4]）。**在 galfits 中配置时初值请用 -0.7。**

## Notes for galfits configuration

- `Z_value` / `Zneb_value` in galfits are **absolute Z**: `Z = 10^logzsol × 0.0142`
  (and likewise for gas). Prior ranges: `Z_value` ≈ [1.5e-4, 2.2e-2], `Zneb_value` ∈ [1.4e-4, 4.5e-2].
- The `nzsfh` joint prior (`PhiSFH` in `prospect/models/priors_beta.py`) couples
  (z, logM, logZ, SFH): p(logM|z) = cosmic mass function, p(logZ|logM) = mass–metallicity
  relation, p(SFH|z,logM) = SFRD-informed StudentT. For per-object fitting, using the
  ranges above with flat/gaussian priors is the practical approximation.
- AGN (`log_fagn`, `log_agn_tau`) has no galfits component; impact on UNCOVER
  optical/NIR photometry <1.5% (for objid 67659, log_fagn=-0.71).
- With dynesty the init values matter little; they are sensible starting points for
  a galfits optimizer.
