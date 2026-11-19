# Columns of the sample tables

Both tables come from `build_tables.py` (19 Nov 2026). Load them with pandas:

```python
import pandas as pd
stars = pd.read_csv('stars.csv', dtype={'gaia_dr3': str})
spectra = pd.read_csv('spectra.csv.gz', dtype={'gaia_dr3': str})   # pandas reads the .gz directly
both = spectra.merge(stars, on='star_id', suffixes=('', '_star'))
```

`star_id` links the two tables. Missing values are empty cells (NaN).

## stars.csv: one row per group (7,994 rows)

A group is the set of spectra that the grouping step assigned to one object (see the README, section 3).

| Column | Meaning |
|---|---|
| `star_id` | Group identifier. `GaiaDR3 <id>` when the star has a Gaia DR3 id; `SIMBAD <name>` when SIMBAD has no Gaia id for it (e.g. very bright stars such as alf Cen B); `nonstellar:<name>` for the Sun, planets, moons, asteroids, comets and calibration frames; `unresolved:<name>` for archive names that could not be matched to one star. |
| `star_label` | Readable name (the SIMBAD main identifier). |
| `gaia_dr3` | Gaia DR3 source id (read as text: it is too long for a float). |
| `folder` | Name of the star's folder in our local per-star dataset (only stars with at least 3 spectra have one). |
| `sp_type` | Spectral type from SIMBAD, as given (e.g. `K1V`, `A5IV-V`, `G8III/IV`). Empty when SIMBAD has none. |
| `type_letter` | First letter of the MK type: O, B, A, F, G, K or M. `other or unknown` for no type, white dwarfs, carbon stars, etc. |
| `luminosity_class` | Luminosity class read from `sp_type` (V dwarf, IV subgiant, III giant, II, Ib, Ia, I supergiants). Empty when not given. Parsed by a simple rule, so check odd cases against `sp_type`. |
| `search_class` | The bundle used by the search: `OB` (O0–B9), `A_earlyF` (A0–F4), `lateF_G_K` (F5–K9), `M` (M0–M9), else `other or unknown`. |
| `vmag` | V magnitude (SIMBAD). |
| `ra`, `dec` | Position in degrees (SIMBAD, ICRS). |
| `pmra`, `pmdec` | Proper motion, mas/yr (SIMBAD). |
| `parallax_mas` | Parallax, mas (SIMBAD, mostly Gaia). |
| `distance_pc` | 1000 / parallax. A naive inverse: unreliable for tiny or noisy parallaxes (a few values are absurdly large). Only for parallax > 0. |
| `simbad_rv_kms`, `simbad_rv_quality` | Radial velocity from SIMBAD and its quality grade (A best to E worst). Only queried for the stars the search can use. |
| `n_spectra_grouped` | Spectra assigned to the group by the grouping step. |
| `n_spectra` | Spectra in the table for this group, after removing the 51 spectra with a bad wavelength grid. |
| `n_nights` | Distinct observing nights (see `night` below). |
| `first_mjd`, `last_mjd`, `span_years` | First and last observation (Modified Julian Date) and the years between them. |
| `median_snr` | Median signal to noise of its spectra (archive value, see `snr` below). |
| `n_pre_upgrade`, `n_post_upgrade` | Spectra before and after the HARPS fibre upgrade (29 May 2015). |
| `n_eggs` | Spectra taken in the EGGS mode (lower resolution, see the README). |
| `n_programmes` | Distinct ESO observing programmes that observed it. |
| `match_method` | How the group was identified: `name` (the archive target name matched SIMBAD and the position agrees), `cone` (position only), `unresolved`, `nonstellar`. |
| `grouping_flags` | Warnings from the grouping step, `;`-separated: `name_unmatched` (name not found in SIMBAD, matched by position), `wide_match` (matched beyond the main radius), `close_pair` (another star nearby), `position_outlier`, `name_position_mismatch`, `name_multiple_stars`, `targ_ignored` (telescope target coordinates not used), `component_ambiguous` (binary component unclear). |
| `group_kind` | `star`, `nonstellar` or `unresolved name`. |
| `in_final_sample` | True for the 2,642 stars with at least 15 spectra: the sample we work with. |
| `search_eligible` | In the final sample and with an O–M type: the 2,236 stars the search tries. |
| `searched` | Scored by the current search (1,740 stars). Some eligible stars drop out because a quality cut on noisy spectra leaves them with too few. |

## spectra.csv.gz: one row per spectrum (349,533 rows)

All public HARPS spectra we downloaded that went through the grouping step.

| Column | Meaning |
|---|---|
| `dp_id` | ESO archive identifier of the spectrum (`ADP.<date>`), unique. |
| `star_id` | Its group (links to `stars.csv`). |
| `target_name` | Target name typed by the observer (raw, often inconsistent; this is why grouping was needed). |
| `ra_pointing`, `dec_pointing` | Telescope position of the observation, degrees. |
| `mjd` | Start of the exposure, Modified Julian Date (ESO ObsCore `t_min`). |
| `night` | Observing night: floor(mjd − 0.5). All spectra of one La Silla night share it. |
| `date_obs` | Start of the exposure, UTC (header `DATE-OBS`). |
| `exptime_s` | Exposure time, seconds. |
| `snr` | Signal to noise from the archive (ESO ObsCore). It is one number for the whole spectrum, usually measured far from Ca II; the noise near Ca II H and K (blue, and inside deep lines) is worse, especially for cool stars. |
| `spec_res` | Nominal resolving power: 115,000 (HAM mode) or 80,000 (EGGS mode). |
| `mode` | `HAM` (high accuracy, the standard mode) or `EGGS` (larger fibres, more light, lower resolution). |
| `berv_kms` | Barycentric Earth radial velocity at that time, km/s (the pipeline already corrects the spectrum for it). |
| `drs_version` | Version of the ESO HARPS reduction pipeline that made the spectrum. |
| `proposal_id` | ESO programme of the observation. |
| `match_method`, `sep_arcsec`, `grouping_flags` | How this spectrum was assigned to its group, and its distance from the star's position (arcsec). |
| `epoch_fibre` | `pre2015` or `post2015` (fibre upgrade, 29 May 2015). |
| `upgrade_2015_window` | True during the upgrade and recommissioning (20 May to 3 June 2015); the search skips these. |
| `bad_grid` | True for the 51 spectra with a broken wavelength grid (excluded everywhere). |
| `search_class`, `in_final_sample`, `search_eligible`, `searched` | Copied from the star, for convenience. |
