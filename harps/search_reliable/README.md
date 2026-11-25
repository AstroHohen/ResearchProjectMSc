# Reliable exocomet search (harps/search_reliable)

This folder is a copy of `harps/search_by_type/` (the overnight search of 1 November 2026) with a stricter way of
deciding what counts as a detection, and new tests that separate a comet from the star itself. The old folder is
untouched, so both can be run and compared.

Goal: a "high confidence" result should mean that the absorption is (a) very unlikely to be noise for the whole
survey, (b) seen twice independently, and (c) not explained by the star changing.

Group B of the ideas (moonlight, fibre change and detector, side effects of PCA) is **not** done here, on purpose.

## What did not change

Steps 1 to 5 and 7 of the search are the same as in `search_by_type`: preparation (resolution, normalisation, noise
cut), local noise, removal of the star's normal behaviour (line shift and PCA), temporal scatter, matched filter, the
K and H doublet rule and the night stacks. The score of a spectrum, D = -(S_K + S_H) / sqrt(2), is the same number as
before.

## What changed, step by step

### 1. The null of each star now comes from the upside-down spectra (idea 8)

Before: H was shifted against K by a wrong velocity, 200 times, and the largest doublets were the "chance level".
In a star with many real comets (bet Pic) the shifted H still holds comets, so the chance level rose (149 for bet Pic)
and a cap of 1.5 times the typical level of the class had to lower it (to 57). The cap then lowered the bar for every
star with a high null, which is why 13 of the 20 old high-confidence stars were below 3 sigma under their own null.

Now: the same shifts are done on the upside-down scores (-S_K, -S_H). Upside down, a comet becomes an emission bump,
which the doublet rule (both scores <= -2) ignores, so real comets no longer enter the null. Noise is symmetric, so
it enters as before.

The null is the maximum doublet per spectrum. A generalised Pareto tail is fitted by moments to its top 5%
(`tail_fit`), which gives the chance p_spec that one spectrum of this star reaches a score this large by chance:

    p_spec = P(null maximum >= D)

There is no cap any more. Two choices made after the first test run:

- The shape of the tail is kept >= 0. A negative shape (a tail that ends) made any score past the end
  "impossible", which gave 37 sigma to scores that were not remarkable.
- The aligned upside-down doublets (no shift) were meant as a second null for the K-H variability of the star. For
  bet Pic they reach 98, because its comets are so frequent that the median reference itself carries comet
  absorption: a spectrum without a comet then looks like emission, and upside down like absorption. That null set a
  threshold of 479, so it rises with the comets exactly as the old one did. It is now only reported (`p_var`, at 400+
  spectra). The decision keeps it in the old form: a score must exceed the largest emission doublets of its star
  (the 99.8% level), as before.

### 2. One false alarm budget for the whole survey (idea 8)

Before: 1% false alarms per star. With 2,348 stars that allows about 23 false stars from noise alone, about what the
flipped control found.

Now: 1% for the whole survey (`false_alarm_survey`). For a star of n spectra in a survey of N stars:

    p_star   = n * p_spec          (any of its spectra)
    p_survey = N * p_star          (any star)
    detection if p_survey < 0.01

With N = 2,348 (the 7 Nov run; now N = 2,236 and the budget split over 2 searches, about 5.5 sigma per spectrum) this is about 4.5 sigma for the whole survey. The night stacks are treated the same way with n nights.
Each row gets sigma_star and sigma_survey, the one-sided Gaussian significance of p_star and p_survey.

### 3. Confirmation by an independent repeat (idea 9)

Before: "confirmed" meant 2 or more spectra of the night above a fixed score of 5.5.

Now it has a probability. While building the null, stage 1 also counts the doublet values at single velocities, so it
knows how often a single spectrum reaches a value d at one fixed velocity and width by chance (`local_probability`).
For a detection in spectrum i, at velocity v and width w:

    p_repeat = min over the other spectra k of the night of p_local(D_k at v, w)  *  m  *  (window / reach)

where m is the number of other spectra that night, and the window allows the comet to drift by about one FWHM
(the last factor counts those extra velocities). A detection found by the night stack is confirmed instead by
stacking each half of the night on its own: both halves must show it, p_halves = max(p_half1, p_half2) * window.
Confirmed means p_repeat < 0.01 or p_halves < 0.01 (`repeat_false_alarm`). The two spectra have independent noise, so
together with the survey threshold this is a strong condition.

### 4. Photospheric control lines (idea 1)

A comet absorbs in Ca II (and a few other gas species). A pulsation, a spot or a companion changes every line of the
star. The cut-outs (+-1 nm) hold other lines of the star: Fe I and Al I lines, more in cooler stars.

- `control_lines` finds the lines of the reference spectrum deeper than 1.5%, away from Ca II (and from H epsilon), up
  to 6 per window.
- `control_scores` runs a compensated Gaussian filter of the same width at the same offset x from the centre of each
  of those lines, in every spectrum, in units of its noise. The lines are combined with their depths as weights:

      S_ctrl = sum(d_c S_c) / sqrt(sum(d_c^2))

- Fail if |S_ctrl| >= 3 in the detection spectrum, or if over all spectra the Ca II feature and S_ctrl are
  correlated (|Spearman| > 0.5). Either sign fails: a pulsation can deepen or fill in the photospheric lines. (The
  first version failed only on absorption, and the first test run then promoted eta Aps and GV Car, whose
  photospheric lines changed by +20 and +9 noise units at the moment of the feature.)
- `control_spread` is the spread of S_ctrl over all spectra of the star. It should be about 1 if the noise of the
  control lines is right; a much larger value means the star's lines vary all the time.
- Is the test able to see anything? If the feature were the star, the control lines would change by about
  depth_K / depth of Ca II at x, times their own depth. `control_expected` is the score that change would give. The
  test is "informative" when that is <= -3. High confidence needs an informative test.

A detection whose control position would land on the Ca II line itself is skipped for that line, so the comet or
the activity of the core cannot leak in. A very strong comet could also absorb in Fe I; that would make it fail,
which errs on the safe side.

### 5. Inside the photosphere (idea 2)

Pulsation and spots only move the line profile within +-v sin i of the line centre. The half width at 10% depth of
the strongest control line measures that width (`photosphere_kms`). A feature inside it is labelled
`inside_photosphere`. It is a label, not a test: many real comets of bet Pic sit inside its v sin i of ~120 km/s.

### 6. Periodicity (idea 3)

At the velocity of the feature, the mean of the K and H residuals over all spectra forms a time series. A Lomb-Scargle
periodogram of it, without the night of the detection, finds the best period from the baseline down to ~5 minutes
(roAp stars). Fail if its false alarm probability is below 0.001 and the period predicts half or more of the feature
at that moment. Run only for scores that can pass the survey threshold, because it is slow.

### 7. H epsilon (idea 4)

H epsilon (397.007 nm) is +121 km/s from Ca II H. A feature within 40 km/s of that offset has an H depth that is not
only Ca II, so the K/H ratio is unreliable there. It is labelled `near_h_epsilon` and kept out of the top tier.

### 8. Tiers

In A to F4 and F5 to K:
- high confidence: passes every test, confirmed by a repeat, informative control test, not near H epsilon
- confirmed, no control lines: the same, but the star has no photospheric line the control test can use. Fast
  rotating A stars such as bet Pic and HD 172555 have no line deeper than about 1% in either window, so for them a
  change of the star is not ruled out by this test. Without this tier the known hosts could never reach the top
- candidate: passes every test, but not confirmed, or near H epsilon, or its control lines are too weak to see a
  change of that size (`missing_for_top` says which)

In OB and M: anomaly, for anything that passes every test.

### 9. Speed

The repeat, control and period tests run only on scores that can pass the survey threshold. bet Pic, with 9,315
spectra, still takes about 18 minutes on one core, as before; a full run takes about 25 minutes on 12 cores.

### 10. At least 15 spectra per group (9 November 2026)

`min_spectra` in `spectral_classes.json` is now 15 for every bundle (it was 3 for OB and A to F4, 10 for F5 to K and
M). A group is the spectra of one star; a bundle is one spectral class. The reference of a group is the median of its
spectra, so with 3 to 9 spectra it is little more than one spectrum and its noise goes into every residual. Several
false detections in the flipped run came from such groups (3, 5 and 7 spectra in A to F4).

The same number is applied twice: to the spectra in the dataset (which groups are eligible, and so the survey size N
in the false alarm budget) and again after the noise cut, counting only spectra kept in both K and H. Before the noise
cut, 15 is close to the median of every bundle, so about half the groups and 91 to 95% of the spectra stay:

| Bundle | Groups | Groups with 15 or more spectra | Spectra kept | Still 15 or more after the noise cut |
| --- | --- | --- | --- | --- |
| OB | 180 | 92 (51.1%) | 91.3% | 91 |
| A to F4 | 554 | 299 (54.0%) | 94.4% | 280 |
| F5 to K | 3,123 | 1,598 (51.2%) | 95.3% | 1,323 |
| M | 527 | 247 (46.9%) | 90.7% | 49 |

M loses the most: most M dwarf spectra are too faint near Ca II to pass the noise cut. The figure
`../results/Reliable/figures/spectra_per_star_by_group.png` shows the spectra per group in each bundle. The results
below are from the run with the old minimums. The 15-spectra run replaced them in `../results/Reliable/` and
`../results/Reliable_flip/`, and the old outputs were deleted.

### 11. Veto lines (9 to 11 November 2026, `veto_check.py`)

Does the star itself explain a Ca II detection? The veto runs on every detection above threshold, in the real and the
flipped run. It also runs on the 15 best nights of bet Pic and HD 172555, as positive controls. The lines are chosen by
spectral type, inside each bundle. A failed check only **flags** the detection: nothing is removed, and no star
changes tier (the user's choice). Stage 2 (`calibrate_by_type.py`) runs the veto itself, after the tests, so every
run of stage 2 is vetoed. `scored_detections.pkl` gets a `veto` column for each detection, and `candidates.csv` gets
`veto` (the label of the best detection) and `veto_flagged_share` (the share of the star's detections that were
flagged). The flipped run is recognised from "flip" in the output folder name, or with `--flip`. Without the veto data
the step is skipped with a message. `veto_check.py` alone reruns only the veto. The telluric template is built once and
kept in `C:/data/harps/telluric_template.npz`.

**The 6 eV rule.** Comet gas around bet Pic has been seen absorbing from levels up to 4.1 eV (Fe II; Vrignaud &
Lecavelier des Etangs 2024, arXiv:2409.15247). The population of a level falls as exp(-E/kT), with kT = 0.86 eV at
10,000 K, and thin gas has too few collisions to fill high levels. A line from a level at 6 eV or more therefore cannot
be absorbed by a comet. A dip in such a line at the comet velocity comes from the star: pulsation, spots, a companion or
a changing blend. The 6 eV margin above the 4.1 eV observed is our own choice; no paper states it. Every line used starts
at 7.7 eV or more (NIST ASD lower levels, checked on 9 November 2026).

**Two checks on the same spectrum:**

1. **Stellar dip.** A compensated Gaussian filter, with the FWHM of the detection, is run on the spectrum divided by the
   median of up to 300 spectra from other nights. It is placed at the comet velocity in each dip line. The score is in
   units of the scatter of the same filter on 60 reference spectra. At 3 sigma or more the detection is flagged "stellar
   dip". The check needs the line in the star: if the line is shallower than 1%, or the comet velocity is outside its
   profile, the result is "not testable".
2. **Activity.** An H alpha or He I D3 index (a core band against continuum bands, after Gomes da Silva et al. 2011)
   more than 3 MAD from the star's other spectra, of either sign, flags the detection as "activity".

In the flipped run a detection is a brightening of the real Ca II, so the veto lines are flipped too. Young or
accreting stars (Herbig Ae/Be and T Tauri in SIMBAD, or an "e" in the spectral type) are labelled "young star", since
infalling gas also changes H alpha and He I there (Mora et al. 2002).

**Telluric lines.** Water lines move across the stellar spectrum through the year with the barycentric correction.
The template comes from 2,000 bet Pic spectra. Each is shifted to the Earth frame, high-pass filtered over 0.3 nm, and
the spectra are median-combined: the star's broad lines smear out over the year, and the narrow telluric lines stay.
Pixels where the template is more than 1% deep are ignored. That is 7.5% of the H alpha band, 13.1% of the He I D3/Na D
band, and about 1% of the Si II and C I bands.

**Lines by spectral type** (lower-level energies from NIST):

| Type | Dip lines (lower level) | Activity index | Why | Sources |
| --- | --- | --- | --- | --- |
| O | He II 454.2 nm (51.0 eV), He I 447.1 nm (21.0 eV) | H alpha | He II 4542 / He I 4471 is the main classification ratio of O stars, so both lines are strong. H alpha traces the variable wind | Sota et al. 2011; Puls et al. 1996 |
| B0 to B3 | Si III 455.3, 456.8, 457.5 nm (19.0 eV), He I 447.1 nm | H alpha | Si III is the strongest metal line of early B stars. H alpha traces Be disc emission | Walborn & Fitzpatrick 1990; Rivinius et al. 2013 |
| B4 to B9 | Mg II 448.1 nm (8.9 eV), He I 447.1 nm | H alpha | He I 4471 / Mg II 4481 is the temperature ratio of B stars | Gray, Digital Spectral Classification Atlas, section 6 (the Gray & Corbally 2009 page is unverified); Rivinius et al. 2013 |
| A | Mg II 448.1 nm, Si II 634.7, 637.1 nm (8.1 eV) | none | Mg II 4481 is the line that survives fast rotation. A stars have no convective chromosphere; their variability is pulsation, which shows in photospheric lines | Royer et al. 2002 (B8 to F2); Eiroa et al. 2021 (phi Leo: pulsation and circumstellar shell variability). No source checked for Si II in A stars |
| F0 to F4 | Mg II 448.1 nm | none | Mg II 4481 is measured in F dwarfs down to F8V | Gebran et al. 2008. Activity: unverified (the Noyes et al. 1984 calibration starts at B-V = 0.44, later than F4) |
| F5 to G9 | C I 505.2, 538.0 nm (7.7 eV) | He I D3 | C I gives clean, high-excitation lines in HARPS spectra of FGK dwarfs. He I D3 is an activity proxy | Nissen et al. 2014; Gomes da Silva et al. 2011; Andretta et al. 2017 |
| K0 to K1 | C I 505.2, 538.0 nm | He I D3, H alpha | C I is reliable down to Teff = 5200 K, about K1. K dwarfs are highly variable | Delgado Mena et al. 2021; Gomes da Silva et al. 2021 (Ca II, not H alpha) |
| K2 to K9 | none | H alpha | Below 5200 K, C I is unreliable (blends, scatter). No verified high-excitation photospheric line | Delgado Mena et al. 2021. H alpha in K dwarfs: no specific source checked |
| M | none | H alpha, He I D3 | H alpha and He I D3 rise together in flares. No usable photospheric line from 6 eV or more was found in 378 to 691 nm (unverified) | Muheki et al. 2020; Gizis et al. 2002; Gomes da Silva et al. 2011 |

H alpha (10.2 eV) and He I D3 (21.0 eV) also start above 6 eV, so a comet cannot absorb in them either. In K2 to M
dwarfs the dip check and the activity check therefore use the same lines.

**Data.** `scripts/veto_fetch.py` downloads 8 bands for every searched spectrum, in one SODA request per spectrum. Only
the flux is kept, on a uniform grid (the same idea as the Ca II cut-outs). The bands are: He I 4471/Mg II, He II/Si III,
C I 5052, C I 5380, He I D3/Na D, O I 6158, Si II and H alpha (`config/harps.json`, block `veto`). The files are in
`C:/data/harps/veto/` (33.8 GB), with their state in `veto_state.sqlite`. The download finished on 11 November 2026:
250,296 of 250,397 spectra. 100 spectra are now proprietary, and 1 failed; detections in them get "no veto data".

**Result (11 November 2026, full veto data).** Stars above threshold with at least one flagged detection:

| Run | Stars above threshold | Flagged | After tests: best detection flagged |
| --- | --- | --- | --- |
| Real | 6 | 5 | 2 of 3: GV Car (stellar dip in Mg II and Si II, likely a companion) and TY CrA (stellar dip, young star). mu Col is clean |
| Flipped | 13 | 5 | 2 of 4: CD-49 451 (activity) and tau01 Lup (stellar dip). HD 97507 and HD 109536 are clean |
| Controls | bet Pic, HD 172555 | bet Pic 1 of 15 nights | |

After the tests and the veto, 1 real star is unflagged (mu Col, an OB anomaly) against 2 flipped stars (HD 97507 and
HD 109536, both candidates). So there is still no excess of real detections over the flipped control, and the veto does
not remove every false alarm.

### 12. Two searches: PCA on and PCA off (13 November 2026, `combine_runs.py`)

bet Pic was never recovered: its threshold was 456 against a best score of 108 (single search; 511 against 107.5 with the budget split over the two searches). A test on 61 stars (11 November 2026)
found why. PCA learns what the star does at many epochs from the other nights. In a star with comets at most epochs, it
learns the comets as part of the star and removes them, and the leftover comet signal also raises the star's noise and
so its threshold. With PCA off, bet Pic's threshold fell to 51, and it passed every test with 355 detections. But PCA
helps ordinary stars: without it, about 5 points fewer of the ordinary fake comets were found (37.5% against 32.5% at
10% depth).

| 61 test stars | Real after tests | Flipped after tests | bet Pic | Fake comets found, 10% depth |
| --- | --- | --- | --- | --- |
| PCA on, noise scaling on (before) | 3 | 4 | missed | 37.5% |
| PCA off | 3 | 2 | recovered | 32.5% |
| Noise scaling off | 3 | 0 | missed | 32.5% |
| Both off | 3 | 1 | recovered | 27.5% |

The flipped test stars were chosen as false alarms of the search before this change, so the drops in the flipped
column are partly chance.

So every star is now searched twice, with PCA on and with PCA off (`exocomet_search.py --no-pca`), and a star counts if
either search finds it. Two searches are two chances of a false alarm, so the survey budget is shared:
`tests['searches'] = 2` gives each search half of the 1% (stage 1 threshold and stage 2 decision). The flipped run is
searched both ways too, and combined the same way, so the control has the same two chances. `combine_runs.py` writes
`../results/TwoRuns/`: the stars above threshold in either search, the best candidate of each star (with `found_by` and
`best_from`), and a summary by bundle for real and flipped. The fake comets (`validate_by_type.py --two-runs`) go
through both searches, and a comet-rich fake star is added: comets at one velocity in 40% of a star's spectra.

**Full-survey results (13 November 2026, 1,743 stars).** bet Pic is now recovered by the PCA-off search: 42 spectra
above its threshold (best score 100 against a threshold of 53), tier "confirmed, no control lines", veto clean, at
+28 km/s. The PCA-on search still misses it. HD 172555 is still not found.

| Stars | Real above threshold | Real after tests | Flipped above threshold | Flipped after tests |
| --- | --- | --- | --- | --- |
| OB | 3 | 2 | 1 | 1 |
| A to F4 | 6 | 2 | 2 | 1 |
| F5 to K | 1 | 1 | 8 | 2 |
| M | 2 | 2 | 5 | 2 |
| All | 12 | 7 | 16 | 6 |

Real after the tests: bet Pic, mu Col, GV Car (stellar dip), TY CrA (stellar dip, young), HD 209290, HD 35650 and
CD-51 6859 (young). Flipped after the tests: ups Aqr (tier **high confidence**, PCA off, veto clean), HD 109536,
AD Leo, AO Men (young), tau01 Lup (stellar dip) and CD-49 451 (activity). So the combined search still shows no excess
over the control, and the flipped run now gives a high-confidence false alarm in a late-type star.

Fake comets, share of injections that pass the tests:

| Bundle | Depth | PCA on | PCA off | Either |
| --- | --- | --- | --- | --- |
| A to F4 | 10% | 25.6% | 23.9% | 27.8% |
| A to F4 | comet-rich | 26.7% | 26.7% | 30.0% |
| OB | 10% | 48.3% | 43.9% | 53.3% |
| OB | comet-rich | 35.0% | 58.3% | 58.3% |
| F5 to K | 20% | 10.0% | 11.1% | 13.3% |
| M | any | 0% | 0% | 0% |

Taking either search raises recovery by 2 to 5 points for ordinary comets, and PCA off recovers far more of the
comet-rich OB stars (58% against 35%).

## 13. Decision rule and cleanup (14 November 2026)

- Stage 2 now uses the rule in `tests.rule` (default **type_weight**, `decision_rules.py`): the survey budget is shared
  by spectral type (A to F4 1, OB 0.1, F5 to K and M 0.01). Chosen as the pre-registered winner of the overnight test
  (`results/Overnight/PREREGISTRATION.md`, addendum 4). `baseline` gives the old decision. pairs and combined are
  sensitivity checks only (`evaluate_rules.py`).
- Shape test: `tests.shape_mode` = weighted, (shape_K + 0.5 shape_H) / 1.5 >= 0.5 (addendum 3).
- Stage 1 writes `class_counts.json` (stars per class, the trials of the weighted budget).
- Removed: the clean-reference option (tested 10 Nov, did not help) and the unused `SNR_cut` key (never applied; the
  real cut is the noise cut sigma <= 0.25).
- On the existing two-run files: real 7 stars after the tests (1 high confidence, HD 96146; bet Pic confirmed, no
  control lines), flipped 5 (0 high confidence). HD 172555 is found only by the pair route.

### HARPS fixes and a bug fix (14 November 2026, addendum 5 and bug fix 2)

- Spectra taken during the fibre upgrade (20 May to 3 June 2015) are not searched. In a star with at least 5 spectra on
  each side, the median change of the residuals across the upgrade is removed like the line-shift modes (HARPS manual
  4.7: the instrumental profile "significantly changed").
- `near_berv` flag (stage 2, flag only): detection within 10 km/s of the BERV, where moonlight or sky would sit (the
  DRS does no sky subtraction, manual 3.2).
- Star velocity: where the symmetry estimate sits within 10 km/s of its +-200 km/s limit (35 stars, mostly OB and A),
  the SIMBAD velocity in `catalogue_rv.csv` is used, or 0 km/s (column `v_star_source`).
- `slow_budget_factor` 10,000 is now the default, so the slow tests exist for every spectrum the type-weighted rule
  can accept.
- Check run: `results/FinalCheck/` (four searches, `run.sh`).
- **Star velocity, bug fix 3.** FinalCheck showed the symmetry estimate off by > 20 km/s from SIMBAD in 74% of OB,
  34% of M and 9% of A to F4 stars, so the activity mask and comet window were centred on the wrong velocity and the
  star's own Ca II core was searched as a comet (all flipped M-star false alarms sat at the true stellar velocity).
  Now: SIMBAD velocity of quality A to C first (`catalogue_rv.csv`, 2,060 of 2,236 stars), else the symmetry estimate,
  else 0 km/s. Safety net from the old code: emission above the cutoff within +-150 km/s is masked +-50 km/s around its
  peak. Re-run: `results/FinalCheck2/`.

## Follow-up scripts

| Script | Idea | What it does |
| --- | --- | --- |
| `vet_candidates.py` | 10, and the tail check | Joint Gaussian fit of K and H for each candidate (depths, velocity, FWHM, K/H ratio with errors, delta chi2), and the null redrawn with 2,000 shifts to check the Pareto tail |
| `blind_test.py` | 11 | `freeze` records a hash of the code and settings and the last observation; `run` refuses to run if anything changed, and reports detections only in newer spectra |
| `sodium_check.py` | 12 | Downloads Na I D cut-outs for the candidate stars only, and measures Na D2 and D1 at the comet velocity |
| `other_instruments.py` | 13 | Lists UVES, FEROS, ESPRESSO and X-shooter spectra of each candidate on the same night |
| `population_tests.py` | 14 | Infrared excess (AllWISE W3 - W4), v sin i (SIMBAD) and the redshift share, detections against all searched stars, real against flipped |

## How to run

    cd harps/search_reliable
    py -3 exocomet_search.py                 # stage 1, real, PCA on, to ../results/Reliable/
    py -3 exocomet_search.py --flip --r ../results/Reliable_flip/
    py -3 exocomet_search.py --no-pca --r ../results/Reliable_nopca/
    py -3 exocomet_search.py --no-pca --flip --r ../results/Reliable_nopca_flip/
    py -3 ../../scripts/veto_fetch.py run    # once: veto cut-outs of every searched spectrum
    py -3 calibrate_by_type.py               # stage 2, real, with the veto flags
    py -3 calibrate_by_type.py --r ../results/Reliable_flip/
    py -3 calibrate_by_type.py --r ../results/Reliable_nopca/
    py -3 calibrate_by_type.py --r ../results/Reliable_nopca_flip/
    py -3 combine_runs.py                    # either search, to ../results/TwoRuns/
    py -3 validate_by_type.py --two-runs     # fake comets, both searches
    py -3 vet_candidates.py
    py -3 sodium_check.py fetch ; py -3 sodium_check.py measure
    py -3 other_instruments.py
    py -3 population_tests.py

## Results of the full run (7 November 2026)

> **Superseded.** This section is kept as history. Its output files were replaced by the 15-spectra and two-run
> searches (sections 10 and 12), so these numbers cannot be reproduced from `results/Reliable/` any more. Current
> results: section 12 and `results/Overnight/RESULTS.md`; current decision rule: section 13.

3,021 stars could be searched; 2,348 kept enough spectra after the quality cuts; 256,188 spectra scored. Results are
in `harps/results/Reliable/` and the flipped control in `harps/results/Reliable_flip/`. Each run takes 25 minutes.

### Real against flipped

| Class | Stars | Real: above threshold | Real: after tests | Flipped: above threshold | Flipped: after tests |
| --- | --- | --- | --- | --- | --- |
| A to F4 | 532 | 4 | 1 | 7 | 1 |
| F5 to K | 1,585 | 1 | 0 | 13 | 2 |
| M | 57 | 2 | 2 | 4 | 2 |
| OB | 174 | 4 | 3 | 1 | 1 |
| All | 2,348 | 11 | 6 | 25 | 6 |

No star reaches the high-confidence tier. The real run has no excess over the flipped control. The budget allows 0.01
false stars in the whole survey, but the flipped run gives 25 above threshold and 6 after the tests. So the chance
levels from the shifted null are too optimistic for some stars, or the flipped spectra hold real events. Both are
likely: Ca II emission flares of late-type stars are real K and H events, and upside down they look like comets, which
fits the 13 late-type stars in the flipped run against 1 in the real one.

### The six real candidates

| Star | Class | Tier | What stops it | Notes |
| --- | --- | --- | --- | --- |
| mu Col | OB | anomaly | control lines too weak | -142 km/s, same velocity for months; Na I D also absorbs there (8 to 16 sigma) |
| GV Car | A to F4 | candidate | near H epsilon | +128 to +133 km/s, the H epsilon offset |
| CD-29 4446 | M | anomaly | (passes all) | +186 km/s, 5.8 sigma for the survey; Na I D not seen |
| zet01 Sco | OB | anomaly | no repeat | a B1.5 hypergiant, whose winds this tier exists for; Na I D 14 to 45 sigma |
| L 49-19 | M | anomaly | no repeat, H epsilon | |
| TY CrA | OB | anomaly | H epsilon | a young eclipsing binary in a reflection nebula |

### The known hosts

- bet Pic: best spectrum 2.0 sigma for its own star, 0 for the survey; best night 3.8 sigma for the star. Its threshold
  is 479 for a best score of 107. Not recovered.
- HD 172555: best spectrum 2.3 sigma for its own star, 0 for the survey; best night 3.2 sigma. Not recovered.

Why bet Pic fails: its comets are present in most spectra, so the median reference itself holds comet absorption.
A spectrum with fewer comets than usual then has positive bumps, and those enter the upside-down null too. The null of
bet Pic has a heavy tail (Pareto shape 0.16), and extrapolated to the survey budget (3.6e-10 per spectrum) it reaches
479. Changing the null cannot fix this; a reference that is not filled with comets would.

### Fake comets (`injection_recovery.csv`)

The same test as before: one fake comet in up to 3 consecutive spectra of one night, 60 stars per class, depths 1 to
20%, FWHM 5, 10 and 20 km/s. "Confirmed" = passes all tests and the repeat. Percent of fake comets recovered:

| Class | Depth 2% | 5% | 10% | 20% | Old search at 10% | Old at 20% |
| --- | --- | --- | --- | --- | --- | --- |
| A to F4 | 1.7 | 9.4 | 20.0 | 28.9 | 50.0 | 65.0 |
| F5 to K | 0.0 | 0.0 | 1.1 | 6.7 | 6.1 | 23.3 |
| OB | 6.1 | 22.8 | 41.1 | 43.9 | 73.3 | 76.7 |
| M | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The strict budget costs about 2.5 times the recoveries in A to F4. Even a 20% deep comet is found less than a third of
the time. Why so few has not been broken down yet (by number of spectra, by star), and that is the next thing to look
at before changing the threshold.

### Follow-up checks

- Other instruments on the same night: none for any of the six (`other_instruments.csv`).
- Joint fit (`vetting.csv`): K/H ratios from 0.93 +- 0.11 (GV Car, below the range of thin or thick gas) to
  1.61 +- 0.16. Redrawing the null with 2,000 shifts changed the tail probabilities by less than a factor 2.5, except
  L 49-19 (2.5e-8 to 5.1e-11, in the safe direction). So the Pareto extrapolation itself is not the problem.
- Na I D (`sodium.csv`): 188 cut-outs downloaded (1 failed). Na absorbs at the Ca II velocity in mu Col and zet01 Sco,
  the two OB stars, where an interstellar or wind origin is more likely than a comet. GV Car had too few reference
  spectra that night. The reference comes from other nights for most stars, so telluric lines can leak in; treat the
  Na values as indicative.
- Population tests (`population_tests.csv`): they run, but with 1 real detection in the classes that can make
  detections they say nothing yet. 1,749 of 2,348 stars have a usable AllWISE W3 - W4, 1,506 a v sin i.

### Blind test

`blind_test.py` is ready but the method is **not frozen**, because the results above show it still has to change.
The archive refresh of 7 November found 312 newly public spectra (observed in November 2024, out of their
proprietary year), 256 of them on 3 searched late-type stars. The star folders have not been rebuilt with them.

## What this means

The search is now honest about noise, and with that the archive shows no Ca II doublet beyond what the flipped
control also produces. Two things block a real detection, and both are fixable:

1. The chance level should be calibrated on the control, not only modelled. The flipped run gives the real false
   alarm rate per class; the threshold can be set where the flipped run gives the budget. Late types need a control
   that is not full of flares (for example the flipped run with the activity tests applied first).
2. Stars with frequent comets need a reference without comets, for example an upper envelope of the spectra
   (comets only absorb) or a reference from the quietest nights. Until then bet Pic cannot be recovered.

Group B (moonlight, fibre change, PCA side effects) is still open, and the PCA question is tied to point 2.
