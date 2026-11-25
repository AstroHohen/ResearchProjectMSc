# Overnight test of three search improvements (written 13 Nov 2026, 23:16, before any run)

Agreed with the user: three ideas, judged on the flipped control, the fake comets and the known hosts. The real
run is reported but is never used to choose a method or a number. Every number below is fixed now.

## Common to all methods
- Same stage 1 as the two-run search (PCA on and PCA off, real and flipped), with one change that does not alter
  any result: the slow tests (repeat, control lines, period) are computed for more spectra (`slow_budget_factor`
  = 10,000, so every spectrum with p_survey < 25 gets them), because the new methods can accept detections the
  old threshold would not have tested. Check: the baseline decision on these files must give the same stars as
  results/TwoRuns.
- Same tests, veto (flag only), bundles and tiers, unless the method says otherwise.
- Total false alarm budget of the survey stays 1%, split over the 2 searches (PCA on, PCA off). A method that adds
  a second route to a detection splits its budget equally between the routes.

## Method 1: budget weighted by spectral type (literature prior)
Weighted Bonferroni: each star gets budget alpha * w / sum(w over the 2,236 stars), with w fixed from where Ca II
exocomets are known: A to F4 = 1 (beta Pic, HD 172555, 49 Cet, HR 10 and most known hosts are A type), OB = 0.1
(a few late B hosts), F5 to K = 0.01 and M = 0.01 (no Ca II exocomet host known). No velocity prior.

## Method 2: star-level evidence, then the star's own spectra
Route A (alpha/2): the baseline spectrum and night-stack detection.
Route B (alpha/2): count k = nights of the star whose night stack has p_night < 0.001. Under the null
k ~ Binomial(n_nights, 0.001); the star passes if N_stars * P(K >= k) < alpha/2. In a star that passes, a spectrum
(or night stack) is detected if its per-star chance n * p_spec (or n_nights * p_night) < alpha/2: the survey trial
factor has been paid by the star test.
Tier: a star that passes route B and has a confirmed detection (repeat), not near H epsilon, is high confidence if
its control lines are informative, or if it has no usable control lines (n_control = 0), since the star test is then
the evidence the control lines would give.

## Method 3: paired exposures of one night
Route A (alpha/2): the baseline detection.
Route B (alpha/2): a spectrum is detected if N_stars * n * p_spec * p_repeat < alpha/2, where p_repeat (stage 1) is
the chance that another spectrum of the same night shows a doublet as large at the same velocity. Only spectra with
p_survey < 25 are considered (the slow tests are computed only there).

## Choosing the winner (user: "control first")
1. The flipped run must give 0 high-confidence stars.
2. beta Pic must be found, in the top tier the method allows it.
3. HD 172555 must be found (user: both required).
Among the methods that meet 1 to 3, the winner recovers the most fake comets that pass the tests (A to F4 at 10%
depth first, then all bundles). If none meets all three, I report which fails what, and the best one on 1 then 2
then 3 then recovery. The real run is only reported.

## Fake comets
validate_by_type.py --two-runs settings (60 stars per bundle, depths 1-20%, FWHM 5/10/20, comet-rich star), the
same injected spectra scored once and judged by each method.

## Addendum (13 Nov 2026 23:26), before any full-survey result
In the flipped run of a star with comets at most epochs (bet Pic), the median reference holds comet absorption, so
its comet-free epochs become dips when flipped (stage 1 already notes this). Such a flipped detection is caused by
real comets and is not a false alarm of the method. For criterion 1, flipped high-confidence stars are counted
without the two known hosts (bet Pic, HD 172555), and the flipped hosts are reported separately.

## Addendum 2 (14 Nov 2026 00:19), after the three methods, before any result of method 4
Result of methods 1-3 on the criteria (full survey): none meets all three. type_weight: 0 flipped high confidence,
bet Pic top tier, HD 172555 missed. star_level: 2 flipped high confidence (both F5 to K), bet Pic high confidence,
HD 172555 missed. pairs: 2 flipped high confidence (both F5 to K), HD 172555 found. Every flipped high-confidence star
is F5 to K, the bundle the type weights suppress. Seen only on the flipped run and the two hosts.

Method 4 (combined), fixed now: the type weights of method 1 give each star its budget alpha * w / sum(w). That
budget is split in three equal parts: route A (spectrum or night stack, as the baseline), route B (the star test of
method 2, then the star's own spectra at its per-star level), route C (pairs of method 3). Tiers as in method 2.
Judged by the same criteria and the same fake comets (a second injection run, same stars and plans, same seeds).

## Bug fix (14 Nov 2026 00:22)
The pair routes (methods 3 and 4) multiplied p_repeat by the survey chance after it was capped at 1, so a spectrum
with no significance at all could pass on its repeat alone. Fixed to use the uncapped chance N * n * p_spec, and the
p_survey < 25 limit now applies to the uncapped value as written above. Methods 3 and 4 are re-evaluated; the
injection run was stopped and restarted with the fix. No rule or number was changed.

## Addendum 3 (14 Nov 2026 11:32), before any result of this test
Attrition test: which stage-2 test rejects the fake comets (validate_attrition.py, same fake comets and seeds) and
the flipped control. User request: test a weighted shape test. Fixed now: shape_w = (shape_K + 0.5 shape_H) / 1.5
must be >= 0.5 (the same min_shape_corr), in place of min(shape_K, shape_H) >= 0.5. The weight 0.5 is the H/K depth
ratio of optically thin Ca II gas, not tuned. Judged by the same criteria: flipped high-confidence stars (without the
hosts), bet Pic, HD 172555, then fake comets that pass (A to F4 at 10% depth first), with the baseline and combined
rules. Adopted only if the flipped high-confidence count does not rise.

## Addendum 4 (14 Nov 2026 14:38), decision by the user
- Main decision rule: **type_weight** (the pre-registered fallback winner). pairs and combined are reported only as
  labelled sensitivity checks; combined was chosen after seeing the flipped run and is exploratory.
- Shape test: weighted (K + 0.5 H) / 1.5 >= 0.5 (addendum 3; it met its adoption condition).
- Both now defaults in search_reliable/spectral_classes.json (tests.rule, tests.shape_mode).
- Reporting rule (user): a known host found below the top tier (e.g. HD 172555 only through the pair route, bet Pic
  "confirmed, no control lines") is reported with the tier the code gives and why it is not higher; the paper may
  raise it one level only for literature confirmation, stated explicitly as a raise from outside the search.
- No other change to rules or numbers. The flipped run has been used for tuning, so its count is not an independent
  false alarm rate; the confirmatory run uses fresh fake comets and a frozen method.

## Addendum 5 (14 Nov 2026 15:10), HARPS fixes agreed with the user, before any run with them
Reason: HARPS User Manual 2.4 section 4.7 ("the instrumental profile (IP) of the spectrograph significantly
changed" after the 2015 fibre exchange) and section 3.2 ("The sky correction/subtraction is left to the user").
- Fix 2a: spectra taken during the upgrade (20 May to 3 June 2015, MJD 57162 to 57177) are not searched.
- Fix 2b: in a star with at least 5 spectra on each side of the upgrade, the median difference of the residuals
  after and before it (one velocity pattern per line) is removed like the line-shift modes, so a profile change
  cannot look like an absorption.
- Fix 3: flag only, never a cut: a detection whose barycentric velocity is within 10 km/s of the spectrum's BERV
  (where scattered moonlight or sky would sit) is flagged near_berv.
- EGGS spectra stay in (fix 1 skipped: the documented effects are small and no detection depends on it).
Check before freezing: full real and flipped run (PCA on and off) with the fixes and the type_weight rule; reported
with bet Pic, HD 172555 and the flipped count, but no rule or number is changed after seeing it, except to undo a fix
that breaks the search (stated if it happens).

## Bug fix 2 (14 Nov 2026 15:13), agreed with the user, before the check run
The star velocity (symmetry of the Ca II K reference) fails in 35 of 1,743 stars, mostly OB and A (weak, broad
photospheric line): the estimate sits at the +-200 km/s limit of its search (e.g. tau Sco, alf Vir, 58 Aql), so the
comet window is centred on the wrong velocity. Fix: an estimate within 10 km/s of that limit counts as failed and the
SIMBAD radial velocity is used (cached once in search_reliable/catalogue_rv.csv), or 0 km/s if SIMBAD has none.
No other star changes.

## Bug fix 3 (14 Nov 2026 17:31), agreed with the user, before the second check run
The first check run (results/FinalCheck) showed the symmetry star velocity off by > 20 km/s from SIMBAD in 74% of OB,
34% of M, 9% of A to F4 and 1% of F5 to K stars (1,586 stars with SIMBAD quality A to C). The activity mask and the
comet window were then centred on the wrong velocity, and the star's own Ca II core was searched as a comet.
Fix: star velocity = SIMBAD radial velocity of quality A to C where it exists (catalogue_rv.csv, all searched stars),
else the symmetry estimate, else 0 km/s if that sits within 10 km/s of its limit. Safety net from the old code
(asset.py X_corr): if the median spectrum shows emission above the activity cutoff within +-150 km/s, +-50 km/s
around the emission peak is masked as well. Judged as before; FinalCheck is kept as the before-fix run.

## Addendum 6 (14 Nov 2026 19:33), agreed with the user, before any result of these two
Seen first: GV Car, PT Vel and HD 96146 (the real candidates after bug fix 3) are spectroscopic binaries: the whole
stellar Ca II line moves between epochs.
- Binary flag: in each spectrum of a candidate star, the minimum of Ca II K smoothed with a Gaussian of sigma 15 km/s
  (so narrow interstellar or comet lines do not set it), within +-250 km/s of the star. If the 10th to 90th percentile
  spread of that position over the star's spectra exceeds 20 km/s, the star is flagged binary and its tier becomes
  anomaly (as for OB and M). Checked on bet Pic and HD 172555, which must not be flagged; if either is, the flag is
  reported as failed and not used.
- Exploratory level: type_weight with false_alarm_survey = 1 (the model expects about one false star in the survey,
  100 times looser than the main level). Real and flipped are judged at the same level. Everything found there is
  called a candidate in the paper, never a detection, and is reported next to the flipped count at that level.
  The main result stays at 0.01. No other level is tried.

## Addendum 7 (14 Nov 2026 19:36), user decisions
- Binary flag withdrawn by the user (it also flagged active late-type stars, whose split emission core moves the
  minimum). Code removed; the binaries GV Car, PT Vel and HD 96146 are reported from the plots.
- Sensitivity sweep requested by the user, after seeing the 0.01 and 1 levels: type_weight at survey budgets 0.001,
  0.01, 0.1, 1 and 10, real and flipped, every level reported. This replaces "no other level is tried" in addendum 6.
  The main result stays at 0.01; the sweep shows how real and flipped grow together, not a level to pick.

## Addendum 8 (14 Nov 2026 20:01), user decision
Minimum transit duration lowered from 30 to 10 minutes (tests.min_transit_minutes). Re-evaluated on FinalCheck2
(stage 2 only, files tagged _transit10).

## Addendum 9 (14 Nov 2026 20:31), diagnostic checks requested by the user
Goal: find steps that hide real comets. (1) Fake comets with the latest code (type_weight and pairs, weighted shape),
with and without the mirror level, recording each fake comet's best score and threshold
(results/FinalCheck2/attrition_*). (2) bet Pic and HD 172555 with and without the 2015 upgrade term (scratchpad).
A step is changed only if the fake comets show it hides comets and the flipped control shows the change does not add
false alarms; any change is logged here before the next full run.

## Bug fix 4 and addendum 10 (16 Nov 2026), agreed with the user, before any result of these changes
Seen first: the fake comets of addendum 9 (results/FinalCheck2/attrition_*) are lost in stage 1, not in the tests
(20%-deep comets found in 55% of A-F4, 50% of OB, 17% of F5-K, 0% of M stars). Per-star thresholds are often 10 to 500
times the largest null value. A read-only trace of 16 stars (scratchpad null_source.csv) shows the top 0.1% of the null
of each bad star comes from one to three spectra (reused by all 200 H shifts), often a 3 km/s spike. A review by the
detection statistician advised against simply switching to a maximum likelihood fit. The real results (FinalCheck2)
have already been seen, so these changes are a method change, judged only by the checks below, never by real
detections or by recovering bet Pic or HD 172555.
- Bug fix 4 (tail bookkeeping): the null holds many exact zeros (the doublet floor), so the start of the tail can sit
  in the ties and the share above it is not TAIL_FRACTION. The tail is now fitted above the 95th percentile of the
  positive null values, and the probability uses the share of the whole null actually above that start.
- Method change A (declustering, Ferro & Segers 2003): the excesses of one epoch over its shifts are one cluster. The
  shape and scale of the tail are fitted to the largest excess of each epoch above the start (one value per epoch); the
  rate above the start stays that of the whole null. This removes the weight of one bad spectrum repeated 200 times,
  and is conservative where clusters are long. Same for the night null (one cluster per night). Estimator unchanged
  (moments, shape clipped to 0 to 0.5).
- Method change B (budget): the spectrum and night routes each spent the full budget, so the stated survey level was
  about half the true one. With tests.split_spectrum_night each gets half.
- Checks, in this order, and the outcome rules:
  1. Hold-out calibration: for each star, fit the null on a random half of its epochs and count exceedances of the
     other half at 1e-3 and 1e-4, pooled by class. Pass if observed / expected is within the Poisson 95% interval or
     below it (conservative). If any class is anti-conservative beyond the interval, the change is not used as is.
  2. The four searches (on, off, on_flip, off_flip) re-run into results/FinalCheck3/ with the main rule (type_weight,
     0.01, weighted shape, 10 min). Flipped stars after tests are reported whatever they are; if they exceed the
     Poisson 95% upper bound for an expected count of 0.01 (i.e. 1 star is already the limit, 2 or more fails), the
     change is reported as failed and FinalCheck2 stays the main result.
  3. Fake comets (validate_attrition.py) with depths 0.05, 0.1, 0.2, 0.3, 0.4 and the comet-rich plan, compared with
     FinalCheck2.
  Whatever the outcome, both FinalCheck2 and FinalCheck3 are reported.
- Detail of method change A (19 Nov 2026), seen on the thresholds of the 16 traced stars only (no detection, real or
  flipped, seen): with one cluster per epoch, fewer than 20 epochs often lie above the start, so the fit fell back to
  "no tail" and the star could detect nothing. The start is now the lower of the 95th percentile of the positive values
  and the 20th largest epoch maximum (a fixed number of upper order statistics, k = 20); stars with fewer than 20
  epochs use all of them, and the fallback is kept only below 10 clusters. Judged by the same checks as above.
- Detail withdrawn (19 Nov 2026): lowering the start to the 20th largest epoch maximum made the thresholds far larger
  (one epoch's maximum over 200 shifts is far above its typical value, so the cluster maxima overstate the tail). Seen on
  the 16 traced stars' thresholds only. Replaced by cluster weighting: the tail starts as in bug fix 4; each excess is
  weighted by 1 / (number of excesses of its epoch), so every epoch weighs one in total in the moments of the shape and
  scale while its whole distribution is used; the rate stays that of the whole null. Fallback below 10 epochs in the
  tail. This is the last change to the estimator before the hold-out calibration, which decides.

## Addendum 11 (19 Nov 2026), spike mask, agreed with the user, rule from the detection statistician, before any result
Seen first: the top null epoch of each of the 12 worst stars holds one spike in one line only (z = +27 to +224,
3 to 9 px, absent from every other spectrum at that pixel): a cosmic ray, which the negated null turns into deep
one-line absorption. Cluster weighting (addendum 10) is kept; unweighted moments are reported as a sensitivity check.
Rule (search.spike_mask, every search identically: real, PCA on and off, flipped, injections), in prepare_line before
the epoch mode and mode removal:
1. Pixel j of spectrum i is an outlier where |z| > 6 (z before mode removal) in a run of at most 12 px.
2. It is kept when the other line of the same spectrum, at the same velocity, shows the same sign with a physical
   ratio (1 <= tau_K / tau_H <= 2), in fractional depth d = (spectrum - reference - baseline) / reference smoothed by
   the narrowest kernel: a K outlier is kept if s * d_H >= |d_K| / 2 - 3 sigma_H, an H outlier if
   s * d_K >= |d_H| - 3 sigma_K (s the sign of the outlier, sigma the MAD scatter of the smoothed d of that spectrum).
3. The other outliers are grown by +-2 px and masked: z = 0 before the modes, and after them residual 0 and weight 0.
4. The masked pixel count of each star is reported; every candidate within one kernel of a masked pixel is listed.
Fixed now, not tuned after candidates. Validation before the full run: (a) injections through the mask: recovery
unchanged and about 0 injected comet pixels masked; synthetic one-line spikes of both signs removed. (b) The hold-out
calibration of addendum 10. If the masked tails are still heavy (shape near 0.5), probability-weighted moments are the
pre-registered fallback estimator. Then FinalCheck3 as in addendum 10.

## Outcome of addenda 10 and 11, and bug fix 5 (21 Nov 2026), before any result of bug fix 5
- Addenda 10 and 11 are not used. The hold-out calibration gave observed / expected 3-7 at 1e-3 and up to 400 at
  1e-5, and the spike mask removed up to 72% of one injected OB comet's pixels. A review by the detection statistician
  then found the calibration itself was mis-designed (null values of one epoch over 200 shifts are clustered, so the
  Poisson interval is invalid, and it tested the null against itself rather than the real statistic). Both outcomes are
  reported; the switches stay in the code, off: tail_method 'legacy', spike_mask false, split_spectrum_night false.
- Bug fix 5 (same review): the real score keeps a doublet only where the K and H depths have a physical ratio
  (ratio_ok, depth K / depth H from 1 to 2 within errors); the null (star_null) never applied that check, so a one-line
  cosmic ray paired with any H value above the floor passed in the null and inflated the thresholds. The null now
  applies the same check, with the depths and errors of H rolled together with its scores and the depths negated with
  the scores (search.null_ratio_check). The night null needs no change (the night score has no ratio check).
- Judged only by: the flipped control (FinalCheck3 on_flip and off_flip, main rule; fail if 2 or more flipped stars
  after tests, the Poisson limit for the expected 0.01) and the fake comets (validate_attrition.py, compared with
  FinalCheck2). Real detections are not used to judge it. If it fails, FinalCheck2 stays the main result.
- The survey level is stated as <= 2% (spectrum and night routes each spend the 1% budget; union bound), and the
  flipped control (1 flipped star vs 4 real after tests in FinalCheck2) is given as the empirical false alarm measure.

## Addendum 12 (21 Nov 2026): event analysis, agreed with the user, before any event is characterised
Pivot of the article: from counting host stars to characterising individual events. The detection stage is frozen
(FinalCheck2, or FinalCheck3 if bug fix 5 passes its checks above).
- Events analysed: every detection that passes the stage-2 tests in the real search by the main rule (type_weight) or
  the pair route; one event = one star, one night, one velocity (detections within 2 FWHM merged). The flipped
  control goes through the same analysis as a check of the scorecard.
- Measurements per event: joint Gaussian fit of K and H at a common velocity and width in each spectrum of the night;
  depth ratio, apparent optical depth and covering fraction (doublet-ratio method, Savage & Sembach 1991); velocity
  relative to the star; change of velocity and depth through the night; duration; veto lines (H alpha, He I D3 and
  Na D, high-excitation photospheric lines) at the same velocity and time; Earth velocity (BERV).
- Scorecard, fixed now. Required (an event failing any is "not supported"): (R1) depth ratio K/H between 1 and 2 within
  2 sigma; (R2) no matching absorption or emission change at the event velocity in H alpha or He I D3 (activity), more
  than 3 sigma; (R3) not within 10 km/s of the Earth velocity in the star frame; (R4) width at least the instrument
  resolution (FWHM >= 2.6 km/s). Points (one each): (P1) redshifted (v > 0 relative to the star); (P2) transient: absent
  in the median and in spectra of other nights at that velocity; (P3) in two or more consecutive spectra; (P4) velocity
  or depth changes during the night; (P5) depth, width and velocity inside the range of published events; (P6) star not
  a known spectroscopic binary; (P7) Na D absorption at the same velocity (comets can show it; no point is removed if
  absent). Level: 6-7 strong, 4-5 probable, 0-3 possible.
- The reference ranges of P5 come from published HARPS events (Kiefer et al. 2014a for bet Pic, Kiefer et al. 2014b
  for HD 172555, Bendahan-West et al. 2025), compiled and checked before the scorecard is applied.
- Addition (21 Nov 2026, seen on the thresholds of the 16 traced stars only): bug fix 4 (tail bookkeeping) is kept
  on its own, without the cluster weighting of addendum 10 (search.tail_method 'bookkeeping'), because it is a bug
  (one star: null maximum 4.3, threshold 2348). FinalCheck3 = bug fixes 4 and 5, nothing else.

## Addendum 13 (21 Nov 2026), agreed with the user, before any result of these
- Event statistics (descriptive): for velocity, depth K, FWHM, K/H ratio and duration, the mean, median, standard
  deviation and robust spread (MAD) of two reference samples: (a) published HARPS events, per event where the papers
  list them (literature_events_table.csv: Kiefer et al. 2014 Nature and A&A, Bendahan-West et al. 2025), and (b) all
  bet Pic events measured by events.py, the same way as every other event. Each event of the search is placed against
  both as a z-score (and a robust z from median and MAD), and the two references are compared with each other (two-
  sample KS test) to show how much the measurement method changes the distribution. No event is kept or dropped by
  these numbers; they are reported only. The scorecard (addendum 12) is unchanged.
- Bendahan-West pipeline on our data: his original code, unchanged in its detection logic (only the reading of our
  HARPS cut-outs is adapted), run on the same stars as our search, on the real spectra and on the flipped spectra
  (2 x median - spectrum), with his published default settings, no tuning. Reported: stars and detections, real vs
  flipped, by spectral class, next to ours (FinalCheck2 / FinalCheck3), and which known hosts each recovers.
- Detail of addendum 13 (21 Nov 2026), before running it: the "our bet Pic" reference is one feature per bet Pic
  night, not only the nights the search flagged: at the velocity and width of the night stack's best doublet (stage 1,
  PCA-on run), measured by events.py exactly as an event; kept when the night-mean depth of K exceeds 5 times its
  error and that of H 3 times. Scorecard and levels are not applied to these (reference only).

## Outcome of FinalCheck3 (21 Nov 2026)
- Main route (type_weight): 6 real and 6 flipped stars after tests (FinalCheck2: 4 and 1). The rule set before the run
  (fail at 2 or more flipped stars after tests) fails it. Pair route: 9 real, 7 flipped. bet Pic recovered on both
  routes; HD 172555 not found.
- Fake comets (type_weight, weighted shape): recovery rises 25 to 40% (e.g. A_earlyF depth 0.05: 71/219 vs 57/211;
  OB depth 0.1: 213/343 vs 152/341); M still 0.
- Decision: FinalCheck2 remains the main result; the event analysis (addenda 12 and 13) stays on FinalCheck2. Bug
  fixes 4 and 5 are reported as found and tested, with their cost to the false-alarm rate, not adopted.
