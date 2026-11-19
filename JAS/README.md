# JAS: getting started with the HARPS exocomet sample

This folder is your starting point. It explains what the project is about, where the data come from and how the
star sample was built, and gives you the sample as small tables, so you can work on it without the rest of the
code. Last updated 19 November 2026.

```
JAS/
  README.md               this file
  download/
    harps_fetch.py        downloads the Ca II H & K windows of every public HARPS spectrum
    harps_fetch.json      its settings
  sample/
    stars.csv             one row per star (spectral type, magnitude, distance, number of spectra, ...)
    spectra.csv.gz        one row per spectrum (date, signal to noise, exposure, instrument mode, ...)
    COLUMNS.md            what every column means
    summary.md            counts: spectral types, classes, totals
    figures/              distribution plots
    build_tables.py       how the tables were made (needs our local data, for reference only)
```

You do not need to download any spectra to use the tables in `sample/`: they hold everything about the sample
except the spectra themselves.

---

## 1. The project in broad strokes

**Exocomets** are comets orbiting other stars. When one passes in front of its star, the gas it releases absorbs a
little starlight at very specific wavelengths. The clearest of these are the two **Ca II lines, K (393.4 nm) and
H (396.8 nm)**. The absorption shows up as an extra narrow dip in the star's spectrum. It lasts from minutes to
hours, and it is usually shifted in wavelength because the comet moves fast towards or away from the star
(tens to hundreds of km/s). It appears in one spectrum and is gone in the next. The best-known example is
β Pictoris, where such dips have been seen for decades.

The project searches the whole public archive of the **HARPS** spectrograph for these transient dips, star by
star, in a uniform and statistically controlled way. Earlier searches looked at chosen stars or smaller samples.
This one covers every star HARPS observed often enough.

How the search works, in one paragraph (you will not need the details):
- For each star we build a reference spectrum: the median of all its spectra.
- Every spectrum is compared with that reference, and we look for an extra absorption that appears in **both** K
  and H at the **same velocity**, with the depth ratio expected for real gas.
- How unusual a dip has to be before we believe it is set from the star's own noise, using a "fake" version of
  the data in which no comet can exist.
- The false alarm rate is controlled over the whole survey (with ~2,000 stars, rare chance events become likely
  somewhere, so the bar is set for the survey as a whole).
- Every candidate then goes through tests that rule out known impostors (stellar activity, instrument effects,
  binary stars, ...).
- As a check on the whole method, the same search is run on spectra turned upside down. Comets only absorb, so
  anything found there is false and tells us the real false alarm rate.

Your part is the **sample itself**: which stars are in it, how they are distributed, and what that means for what
the search can and cannot say.

---

## 2. The data: HARPS

**HARPS** (High Accuracy Radial velocity Planet Searcher) is a high-resolution spectrograph on the ESO 3.6 m
telescope at La Silla, Chile, in use since 2003. It was built to find exoplanets by measuring tiny stellar velocity
changes, which has two consequences for us:

- **The sample is shaped by planet hunting.** Most of the time went to nearby, quiet, Sun-like stars (F, G and K
  dwarfs), many observed hundreds or thousands of times. Hot stars (O, B, A) and cool M dwarfs are fewer. Yet
  known exocomets are mostly around A-type stars. This mismatch is one of the main statistical issues of the sample.
- **Observations are uneven in time.** Some stars have one night, some have 20 years of monitoring; some nights
  have one spectrum, some have a hundred (`n_spectra`, `n_nights`, `span_years` in `stars.csv`).

Instrument facts that matter for the data:
- **Resolving power** R = 115,000 in the standard **HAM** mode, 80,000 in the **EGGS** mode (larger fibres, more
  light, lower resolution). About 4% of spectra (13,565) are EGGS (`mode` column).
- **Fibre upgrade, 29 May 2015.** The fibres were replaced, which changed the instrument profile slightly. Spectra
  before and after are tagged (`epoch_fibre`); the two weeks of the upgrade itself are excluded.
- The archive spectra are **reduced by ESO's pipeline**: extracted to 1D, wavelength calibrated, and moved to the
  **barycentric frame** (corrected for the Earth's motion, `berv_kms`). We use them as delivered.
- **Signal to noise (`snr`)** is a single archive number per spectrum, measured over the whole spectrum. Near Ca II
  (blue end, inside deep absorption lines) the real noise is worse, much worse for cool M stars, which are faint in
  the blue.

---

## 3. How the sample was built

### 3.1 Download (`download/harps_fetch.py`)
- The ESO archive lists **355,115** public HARPS 1D spectra (Sept 2003 to Oct 2026).
- A full spectrum is ~5 MB, but we only need two narrow windows. The script asks ESO's cut-out service (SODA)
  for **±1.0 nm around K and H** only, the windows used by Bendahan-West et al. (2025).
- Result: **349,845** spectra downloaded, all covering both lines; 5,252 still proprietary (restricted); 18 failed.
- It takes **~24 GB** on disk instead of ~200 GB for full spectra.
- Each spectrum is saved as a small `.npz` file (wavelength in nm, flux and error for K and H, plus its full FITS
  header, gzipped). Key header values also go into a SQLite table, `state.sqlite`, which is where most of the
  columns in `spectra.csv.gz` come from.

To download yourself (only needed if you want the spectra, not for the tables):
```
pip install numpy pandas requests astropy
# edit download/harps_fetch.json: set "data_dir" to a folder with ~25 GB free
python download/harps_fetch.py manifest   # list all HARPS spectra from the ESO archive (a few minutes)
python download/harps_fetch.py run        # download; safe to stop with Ctrl+C and run again later
python download/harps_fetch.py status     # progress counters
```
Use `python download/harps_fetch.py run --limit 200` for a quick test. Stars listed in `priority_targets` (β Pic and other known exocomet hosts) are fetched first;
the rest come in a random order, so a partial download is a fair random sample.

### 3.2 Grouping spectra into stars
The archive gives a target name per spectrum, typed by each observer, so one star can appear under many names
(`HD 39060`, `bet Pic`, `beta Pic`, ...) and names can be wrong. Each spectrum was therefore matched to a star in
**SIMBAD** using:
- its name, accepted when the SIMBAD position agrees with the telescope pointing (`match_method = name`);
- otherwise its position alone, a cone search (`cone`);
- warnings are kept in `grouping_flags` (close pairs, wide matches, mismatches between name and position, ...).

Non-stellar targets (Sun, planets, moons, asteroids, comets, calibrations) are kept apart, and so are names that
could not be resolved.

Result: **7,994 groups**: 7,520 stars (7,332 with a Gaia DR3 id), 29 non-stellar, 445 unresolved names. The
spectral type, magnitude, position, proper motion and parallax of each star come from SIMBAD at this step.

### 3.3 The final sample
- Stars with at least **15 spectra** are kept, because the method needs enough spectra per star to learn what is
  normal for it. A further 51 spectra on a different wavelength grid were dropped.
- **Final sample: 2,642 stars and 298,587 spectra** (`in_final_sample`). Median 35 spectra per star, up to 23,041
  (α Cen B).
- The search uses the stars with an O to M spectral type: **2,236 stars** (`search_eligible`), in four classes:

| Class | Types | Stars | Note |
|---|---|---|---|
| OB | O0–B9 | 92 | hot, few lines; few exocomets known |
| A_earlyF | A0–F4 | 299 | where most known exocomets are |
| lateF_G_K | F5–K9 | 1,598 | the planet-search stars |
| M | M0–M9 | 247 | cool; Ca II is faint and often in emission |

- **1,740** of them are actually scored by the current search (`searched`); the others lose spectra to a quality cut
  on noisy data and fall below the minimum. Note how unevenly this hits the classes (see `summary.md`).
- The 406 final-sample stars with no usable type are not searched.

---

## 4. What the statistics of the sample could address

Some starting points (to discuss with Guilherme):
- **Representativeness.** How does the HARPS sample compare with an unbiased (e.g. volume-limited) sample of stars
  by type, luminosity class and distance? What population could a result be generalised to?
- **Selection effects.** Planet-search programmes pick quiet dwarfs; exocomet programmes pick young A stars. How
  much of the sample is targeted (by programme) and how does that bias the classes?
- **Sensitivity per star.** Number of spectra, nights, time span and signal to noise set how detectable a comet
  is. Their distribution per class decides how many stars are effectively searched.
- **Class sizes.** The search gives each class a share of the false alarm budget. Whether those shares and the
  class definitions suit the sample is a statistical question.
- **Upper limits.** If few or no new exocomets are found, the sample sizes, cadence and sensitivity are what turn
  that into an honest rate limit per spectral type.

`sample/summary.md` and the figures give the first look; `COLUMNS.md` explains each column, including its caveats
(e.g. `distance_pc` is a naive 1/parallax, and `luminosity_class` is parsed by a simple rule).

## 5. References
- Mayor et al. 2003, *The Messenger* 114, 20: HARPS.
- ESO HARPS User Manual (instrument modes, 2015 fibre upgrade).
- Kiefer et al. 2014, *Nature* 514, 462: exocomet families of β Pic from HARPS.
- Bendahan-West et al. 2025: an earlier search for exocomets in HARPS Ca II spectra.
