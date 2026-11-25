'''
07/11/26
Exocomet search by spectral class, reliable version. Stage 1 of 2: scores every spectrum, then calibrate_by_type.py decides.
Same steps 1-5 and 7 as harps/search_by_type (01/11/26). What changed is how a score becomes a detection, and the
tests run on each detection. README.md in this folder explains every change.

For each star:
1. local noise of quicksearch_v3 and removal of the slow local median of the residual
2. removal of what the star does at many epochs: a line shift (the first two derivatives of the reference) at any n,
   plus principal components at 20+ spectra. Components come from other nights, so a transient cannot remove itself
3. noise scaled by the scatter of each velocity over time, at 10+ spectra (the star's normal variability)
4. matched filter of Eq. 6 of the article with T = Gaussian * reference, since a comet removes a fraction of the
   stellar flux, and a local box subtracted, so structure broader than the feature scores nothing
5. one feature in K and H: same velocity and width, depth(K) / depth(H) between 1 and 2 within errors
6. chance level of each star, NEW: from the upside-down spectra, K against H at a wrong velocity. Upside down, real
   comets turn into emission, which the doublet rule ignores, so a star with many comets no longer raises its own
   null. Each score gets a probability, so stage 2 can use one false alarm budget for the whole survey (no cap). The
   aligned upside-down spectra measure the K-H variability of the star; at 400+ spectra their tail is reported as p_var
7. the same score on the stack of each night: a comet lasts hours, so a faint one adds up over the exposures of a night

Each spectrum gets one row. Spectra with a large score also get the tests of false positives, NEW ones included:
- repeat: probability that another spectrum of the night, or the other half of the night, shows it by chance
- control lines: photospheric lines of the same window (Fe I, Al I...) at the same velocity from their centre.
  A comet absorbs in Ca II only, a change of the star shows in all its lines
- photosphere: is the feature inside the velocity width of the photospheric lines (where pulsation lives)
- period: does a periodic signal at that velocity, fitted without the night, predict the feature
- H epsilon: H epsilon is +121 km/s from Ca II H, so the depth of H there is not only Ca II
'''
import argparse
import json
import multiprocessing
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.ndimage import binary_dilation, convolve1d, label, median_filter
from scipy.signal import fftconvolve, find_peaks, peak_widths
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'search'))
from quicksearch_v3 import ASSET_LocalNoise, MAD_TO_SIGMA

FWHM_TO_SIGMA = 2.354820045
SPECTRAL_ORDER = 'OBAFGKM'
EDGE_KMS = 420  # work on |v| below this. 300-420 km/s holds no signal and measures the noise of the scores
UPGRADE_MJD = (57162, 57177)  # 20 May to 3 June 2015: fibre exchange (HARPS upgrade page); not searched
EMISSION_MASK_KMS = 50  # half width masked around an emission peak (Bendahan-West, asset.py)
STAR_VELOCITY_LIMIT = 200  # star_velocity searches centres within this
STAR_VELOCITY_FAILED = 10  # an estimate this close to the limit counts as failed
MIN_EPOCH_SPECTRA = 5  # spectra needed on each side of the upgrade to remove the profile change
BOX_SIGMAS = 8  # half width of the box of the compensated filter
FOLDS = 8
MIN_NIGHTS_PCA = 5
MIN_EPOCHS_PCA = 20
MIN_EPOCHS_TEMPORAL = 10
PRIOR_EPOCHS = 20  # strength of the shrinkage of the temporal scatter towards 1 (Smyth 2004)
PCA_MAX_EPOCHS = 300
TAIL_FRACTION = 0.05  # share of the null used to fit its tail
SPIKE_Z, SPIKE_MAX_PX, SPIKE_GROW = 6, 12, 2  # one-line outlier mask (addendum 11)
TAIL_CLUSTERS_MIN = 10  # fewest epochs in the tail of a cluster-weighted null, else no tail fit (addendum 10)
MIN_EPOCHS_VARIABILITY = 400  # aligned upside-down null: 5% tail of 400 epochs = 20 values to fit
LOCAL_BINS = np.geomspace(2 * np.sqrt(2) - 1e-6, 200, 400)  # doublet values at one velocity, from the floor 2+2
H_EPSILON_KMS = 121.0  # H epsilon 3970.07 A from Ca II H 3968.47 A
H_EPSILON_HALF_KMS = 40.0
CA_EXCLUDE_KMS = {'K': (-250, 250), 'H': (-250, 300)}  # control lines are taken outside these, H epsilon included
CONTROL_PROMINENCE = 0.015  # depth of a photospheric line to use it as a control
MAX_CONTROL_LINES = 6
PERIOD_MIN_EPOCHS = 30
PERIOD_MAX_FREQUENCY = 300.0  # 1/day: periods down to ~5 min (roAp stars)
PERIOD_MAX_GRID = 2_000_000

Config = None
Searches = None


def spectral_index(code):
    '''A5 -> 25. A type with no digit counts as the middle of its letter'''
    return SPECTRAL_ORDER.index(code[0]) * 10 + (int(code[1]) if len(code) > 1 else 5)


def class_of(sp_type, classes):
    '''SIMBAD spectral type -> class name, None if it is not an O-M type'''
    match = re.match(r'^(?:sd|esd|d|g|k)?([OBAFGKM]\d?)', str(sp_type).strip())
    if match is None:
        return None
    index = spectral_index(match.group(1))
    for name, spectral_class in classes.items():
        if spectral_index(spectral_class['from']) <= index <= spectral_index(spectral_class['to']):
            return name
    return None


def local_noise(resid, window, guard, step):
    '''Local noise and local median of a 2Darray, measured every step pixels and interpolated between them'''
    half = window // 2
    centres = np.arange(0, resid.shape[1], step)
    padded = np.pad(resid, ((0, 0), (half, half)), mode='reflect')
    keep = np.abs(np.arange(window) - half) > guard
    sigma, baseline = np.empty((len(resid), len(centres))), np.empty((len(resid), len(centres)))
    for i, centre in enumerate(centres):
        pixels = padded[:, centre:centre + window][:, keep]
        baseline[:, i] = np.median(pixels, axis=1)
        sigma[:, i] = MAD_TO_SIGMA * np.median(np.abs(pixels - baseline[:, i, None]), axis=1)
    pixels = np.arange(resid.shape[1])
    return (np.array([np.interp(pixels, centres, row) for row in sigma]),
            np.array([np.interp(pixels, centres, row) for row in baseline]))


def pca_rank(z, cap, rng):
    '''Parallel analysis: components stronger than the same epochs each shifted at random along the velocity axis'''
    if cap < 1:
        return 0
    values = np.linalg.svd(z, compute_uv=False)
    shifted = np.array([np.roll(row, shift) for row, shift in zip(z, rng.integers(0, z.shape[1], len(z)))])
    return min(int(np.sum(values > np.linalg.svd(shifted, compute_uv=False)[0])), cap)


def remove_modes(z, nights, fixed, kernels, rng):
    '''Removes the shared behaviour of the star from z. Returns cleaned z, and for each epoch its fold, with the
    share of each template that survives the removal (fold, width, pixel)'''
    unique = np.unique(nights)
    use_pca = len(z) >= MIN_EPOCHS_PCA and len(unique) >= MIN_NIGHTS_PCA and Config['search'].get('pca', True)
    folds = rng.permutation(len(unique)) % (FOLDS if use_pca else 1)
    fold_of = folds[np.searchsorted(unique, nights)]
    clipped = np.clip(np.nan_to_num(z), -5, 5)
    rank = 0
    if use_pca:
        sample = rng.permutation(len(z))[:PCA_MAX_EPOCHS]
        rank = pca_rank(clipped[sample], min(len(unique) // 5, Config['search']['max_pca_rank']), rng)

    cleaned = z.copy()
    survives = []
    for fold in range(fold_of.max() + 1):
        inside = fold_of == fold
        columns = fixed
        if rank > 0:
            train = np.where(~inside)[0]
            train = rng.permutation(train)[:PCA_MAX_EPOCHS]
            modes = np.linalg.svd(clipped[train], full_matrices=False)[2][:rank]
            columns = np.vstack([fixed, modes])
        basis = np.linalg.qr(columns.T)[0]
        cleaned[inside] -= (np.nan_to_num(z[inside]) @ basis) @ basis.T
        kept = [1 - sum(np.convolve(mode, kernel / np.linalg.norm(kernel), mode='same') ** 2 for mode in basis.T)
                for kernel in kernels]
        survives.append(np.sqrt(np.clip(kept, 0.1, 1)))
    return cleaned, fold_of, np.array(survives), rank


def temporal_scatter(z):
    '''Scatter of each velocity over time in units of the local noise, smoothed and shrunk towards 1'''
    if len(z) < MIN_EPOCHS_TEMPORAL or not Config['search'].get('temporal_scatter', True):
        return np.ones(z.shape[1])
    middle = np.median(z, axis=0)
    scatter = MAD_TO_SIGMA * np.median(np.abs(z - middle), axis=0)
    scatter = median_filter(scatter, size=9, mode='nearest')
    return np.sqrt((PRIOR_EPOCHS + len(z) * scatter ** 2) / (PRIOR_EPOCHS + len(z)))


def compensated_filter(resid, ref, inv_var, kernel, box):
    '''Eq. 6 with a box of the local mean removed. Returns score (negative for absorption), depth and its error'''
    convolve = lambda data, shape: fftconvolve(data, shape[None, :], mode='same', axes=1)
    template_box = convolve(ref * inv_var, kernel)
    box_box = np.maximum(convolve(inv_var, box), 1e-12)
    match = convolve(resid * ref * inv_var, kernel) - template_box * convolve(resid * inv_var, box) / box_box
    weight = np.maximum(convolve(ref ** 2 * inv_var, kernel ** 2) - template_box ** 2 / box_box, 1e-12)
    return match / np.sqrt(weight), -match / weight, 1 / np.sqrt(weight)


def epoch_mode(z, post):
    '''Change of the line profile at the 2015 fibre upgrade (HARPS manual 4.7): median residual after minus before,
    as one velocity pattern. None if either side has fewer than MIN_EPOCH_SPECTRA spectra'''
    if post is None or post.sum() < MIN_EPOCH_SPECTRA or (~post).sum() < MIN_EPOCH_SPECTRA:
        return None
    clipped = np.clip(np.nan_to_num(z), -5, 5)
    return np.median(clipped[post], axis=0) - np.median(clipped[~post], axis=0)


def standardise(spectra, ref):
    '''Step 1 for one line: residual from the reference in units of the local noise. Returns z, sigma, baseline'''
    search = Config['search']
    sigma, baseline = [], []
    for start in range(0, len(spectra), search['chunk']):
        s, b = local_noise(spectra[start:start + search['chunk']] - ref, search['local_window'], search['local_guard'],
                           search['local_step'])
        sigma.append(s)
        baseline.append(b)
    sigma, baseline = np.vstack(sigma), np.vstack(baseline)
    sigma[sigma <= 1e-3 * np.median(sigma, axis=1, keepdims=True)] = np.inf
    return (spectra - ref - baseline) / sigma, sigma, baseline


def spike_masks(zK, zH, dK, dH, kernel):
    '''Addendum 11: one-line outliers (cosmic rays, bad pixels) of either sign. An outlier is |z| > SPIKE_Z in a run of
    at most SPIKE_MAX_PX pixels; it is kept when the other line at the same velocity has the same sign with a physical
    ratio (1 <= tau_K / tau_H <= 2), in fractional depth d smoothed by the narrowest kernel. Returns the K and H masks,
    grown by SPIKE_GROW pixels'''
    kernel = kernel / kernel.sum()
    smooth = [convolve1d(np.nan_to_num(d), kernel, axis=1, mode='nearest') for d in (dK, dH)]
    spread = [MAD_TO_SIGMA * np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1, keepdims=True)
              for d in smooth]
    row = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    masks = []
    for z, own, other, other_spread, ratio in ((zK, smooth[0], smooth[1], spread[1], 0.5),
                                               (zH, smooth[1], smooth[0], spread[0], 1.0)):
        z = np.nan_to_num(z, posinf=0, neginf=0)
        big = np.abs(z) > SPIKE_Z
        runs, _ = label(big, structure=row)
        short = big & (np.bincount(runs.ravel())[runs] <= SPIKE_MAX_PX)
        doublet_like = np.sign(z) * other >= ratio * np.abs(own) - 3 * other_spread
        masks.append(binary_dilation(short & ~doublet_like, structure=np.ones((1, 2 * SPIKE_GROW + 1), bool)))
    return masks


def prepare_line(spectra, ref, v, good, rng, nights, post=None, standard=None, spikes=None):
    '''Steps 1-3 for one line. Returns residual, reference, inverse variance and the share that survives. spikes: pixels
    masked before the modes (addendum 11)'''
    search = Config['search']
    step = np.median(np.diff(v))
    kernels = [np.exp(-np.arange(-int(4 * s) - 1, int(4 * s) + 2) ** 2 / (2 * s ** 2))
               for s in (f / FWHM_TO_SIGMA / step for f in search['fwhm_kms'])]
    z, sigma, baseline = standard if standard is not None else standardise(spectra, ref)
    if spikes is not None:
        z = np.where(spikes, 0.0, z)

    mean_sigma = np.median(np.where(np.isfinite(sigma), sigma, np.nan), axis=0)
    slope = np.gradient(ref, v)
    fixed = np.array([slope / mean_sigma, np.gradient(slope, v) / mean_sigma])
    change = epoch_mode(z, post) if search.get('epoch_mode', True) else None
    if change is not None:
        fixed = np.vstack([fixed, change])
    fixed = np.nan_to_num(fixed / np.maximum(np.linalg.norm(np.nan_to_num(fixed), axis=1, keepdims=True), 1e-12))
    cleaned, fold_of, survives, rank = remove_modes(z, nights, fixed, kernels, rng)
    scatter = temporal_scatter(cleaned)
    noise = sigma * scatter
    inv_var = good / noise ** 2
    if spikes is not None:
        cleaned = np.where(spikes, 0.0, cleaned)
        inv_var = np.where(spikes, 0.0, inv_var)
    return dict(z=cleaned / scatter, resid=cleaned * sigma, inv_var=inv_var, fold_of=fold_of, survives=survives,
                rank=rank, kernels=kernels, ref=ref)


def score_line(line, part, zone):
    '''Step 4 for the epochs in part, for all widths. Each score is scaled by its own scatter where there is no signal'''
    scores, depths, errors = [], [], []
    for w, kernel in enumerate(line['kernels']):
        half = int(BOX_SIGMAS * (len(kernel) - 1) / 8) + 1
        box = np.ones(2 * half + 1)
        S, A, E = compensated_filter(line['resid'][part], line['ref'], line['inv_var'][part], kernel, box)
        S = S / line['survives'][line['fold_of'][part], w]
        far = S[:, zone]
        scale = MAD_TO_SIGMA * np.median(np.abs(far - np.median(far, axis=1, keepdims=True)), axis=1, keepdims=True)
        scores.append(S / np.maximum(scale, 1e-6))
        depths.append(A)
        errors.append(E * scale)
    return np.array(scores), np.array(depths), np.array(errors)


def doublet(SK, SH, floor):
    '''Absorption score: both lines at least at floor. Shapes (..), positive for absorption'''
    return np.where((SK <= -floor) & (SH <= -floor), -(SK + SH) / np.sqrt(2), 0.0)


def legacy_tail():
    '''search.tail_method: 'legacy' reproduces FinalCheck2; 'bookkeeping' is bug fix 4 alone (tail of the positive values,
    true rate above its start); 'declustered' adds the cluster weighting of addendum 10 (failed, not used)'''
    return bool(Config) and Config['search'].get('tail_method', 'declustered') == 'legacy'


def tail_fit(null, clusters=None):
    '''Generalised Pareto tail (Coles 2001) of a null sample, fitted by moments. Keeps 101 quantiles for values below
    the tail. The tail starts at the top TAIL_FRACTION of the positive values (the doublet floor makes many exact
    zeros), and 'rate' is the share of the whole null above that start (bug fix 4). With clusters (the epoch of each
    value), each excess is weighted by 1 / (excesses of its epoch), so one bad spectrum reused by every shift weighs as
    one epoch in the shape and scale (cluster weighting, addendum 10; Ferro & Segers 2003 for clusters of extremes)'''
    null = np.asarray(null, dtype=float)
    quantiles = np.quantile(null, np.linspace(0, 1, 101)) if len(null) else np.zeros(101)
    if len(null) == 0:
        return dict(start=0.0, scale=0.0, shape=0.0, quantiles=quantiles, n=0, rate=TAIL_FRACTION)
    weights, fewest = None, 20
    if legacy_tail():
        ordered = np.sort(null)
        start, rate = ordered[int((1 - TAIL_FRACTION) * len(null))], TAIL_FRACTION
        excess = ordered[ordered > start] - start
    else:
        positive = np.sort(null[null > 0])
        if len(positive) < 20:
            return dict(start=float(null.max()), scale=0.0, shape=0.0, quantiles=quantiles, n=len(null),
                        rate=TAIL_FRACTION)
        start = positive[int((1 - TAIL_FRACTION) * len(positive))]
        above = null > start
        rate = float(above.mean())
        excess = null[above] - start
        if clusters is not None:
            _, index, sizes = np.unique(np.asarray(clusters)[above], return_inverse=True, return_counts=True)
            weights, fewest = 1.0 / sizes[index], TAIL_CLUSTERS_MIN
            if len(sizes) < fewest:
                excess = excess[:0]
    if len(excess) < fewest or excess.std() == 0:
        return dict(start=float(null.max()), scale=0.0, shape=0.0, quantiles=quantiles, n=len(null), rate=rate)
    mean = np.average(excess, weights=weights)
    variance = np.average((excess - mean) ** 2, weights=weights)
    ratio = mean ** 2 / variance
    # shape >= 0: a negative shape ends the tail at a finite value, so any score past it would be called impossible
    shape = float(np.clip(0.5 * (1 - ratio), 0, 0.5))
    return dict(start=float(start), scale=float(0.5 * mean * (ratio + 1)), shape=shape, quantiles=quantiles,
                n=len(null), rate=rate)


def exceed_probability(value, fit):
    '''P(null >= value) from tail_fit. Above the largest null value without a tail fit, 1 / (n + 1)'''
    value = np.asarray(value, dtype=float)
    if fit['n'] == 0:
        return np.ones_like(value)
    below = 1 - np.interp(value, fit['quantiles'], np.linspace(0, 1, 101))
    if fit['scale'] == 0:
        return np.where(value <= fit['start'], np.maximum(below, 1 / (fit['n'] + 1)), 1 / (fit['n'] + 1))
    excess = np.maximum(value - fit['start'], 0) / fit['scale']
    if abs(fit['shape']) < 1e-3:
        tail = np.exp(-excess)
    else:
        tail = np.maximum(1 + fit['shape'] * excess, 1e-300) ** (-1 / fit['shape'])
    rate = fit.get('rate', TAIL_FRACTION)
    return np.where(value <= fit['start'], np.maximum(below, rate), rate * tail)


def fit_threshold(fit, probability):
    '''Value exceeded with this probability under tail_fit'''
    rate = fit.get('rate', TAIL_FRACTION)
    if probability >= rate or fit['scale'] == 0:
        return float(np.interp(1 - min(probability, 1), np.linspace(0, 1, 101), fit['quantiles']))
    level = rate / probability
    if abs(fit['shape']) < 1e-3:
        return float(fit['start'] + fit['scale'] * np.log(level))
    return float(fit['start'] + fit['scale'] / fit['shape'] * (level ** fit['shape'] - 1))


def ratio_check(AK, AH, EK, EH):
    '''Depths of K and H consistent with real gas (depth K / depth H from 1 to 2), within the errors'''
    return (AK >= AH - 2 * np.hypot(EK, EH)) & (AK <= 2 * AH + 2 * np.hypot(EK, 2 * EH))


def star_null(SK, SH, floor, shifts, rng, depths=None):
    '''Chance level of a star from its upside-down scores (shape width, epoch, velocity): K against H at a wrong
    velocity. Returns the tail of the maximum per epoch, and the counts of single velocity values for local p-values.
    depths (AK, AH, EK, EH): the same K/H depth ratio check as the real score (bug fix 5); the depths of H are moved
    with its scores, and upside down the depths change sign'''
    SK, SH = -SK, -SH
    pixels = SK.shape[2]
    low = min(150, pixels // 3)
    maxima = []
    counts = np.zeros(len(LOCAL_BINS) - 1)
    for shift in rng.integers(low, pixels - low, shifts):
        values = doublet(SK, np.roll(SH, shift, axis=2), floor)
        if depths is not None:
            AK, AH, EK, EH = depths
            values = np.where(ratio_check(-AK, -np.roll(AH, shift, axis=2), EK, np.roll(EH, shift, axis=2)), values, 0.0)
        maxima.append(values.max(axis=(0, 2)))
        counts += np.histogram(values[values > 0], LOCAL_BINS)[0]
    survival = np.concatenate([np.cumsum(counts[::-1])[::-1], [0]]) / (shifts * SK.size)
    declustered = bool(Config) and Config['search'].get('tail_method') == 'declustered'  # addendum 10, not used
    epochs = np.tile(np.arange(SK.shape[1]), len(maxima)) if declustered else None  # one cluster per epoch
    return tail_fit(np.concatenate(maxima), epochs), survival


def local_probability(value, survival):
    '''P(doublet >= value) at one fixed velocity and width, under the null of star_null'''
    value = np.atleast_1d(np.asarray(value, dtype=float))
    out = np.ones_like(value)
    positive = value > LOCAL_BINS[0]
    index = np.clip(np.searchsorted(LOCAL_BINS, value[positive], side='right') - 1, 0, len(survival) - 1)
    out[positive] = np.maximum(survival[index], survival[survival > 0].min() if (survival > 0).any() else 1.0)
    return out


def star_velocity(ref, v):
    '''Centre of the line: the velocity about which the reference is most symmetric. The lowest point of the line fails
    in active stars, where emission splits the core (eps Eri: -66 km/s, true +16). Within 3 km/s on five known stars'''
    step = np.median(np.diff(v))
    offsets = np.arange(int(5 / step), int(120 / step))
    centres = np.where((np.abs(v) < 200) & (np.arange(len(v)) >= offsets[-1]) & (np.arange(len(v)) < len(v) - offsets[-1]))[0]
    difference = [np.mean((ref[c + offsets] - ref[c - offsets]) ** 2) for c in centres]
    return v[centres[int(np.argmin(difference))]]


CATALOGUE = {}


def catalogue_velocity(folder):
    '''SIMBAD radial velocity of quality A to C (catalogue_rv.csv), or None'''
    if not CATALOGUE:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'catalogue_rv.csv')
        table = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=['folder', 'rv_kms', 'rv_quality'])
        good = table[table['rv_quality'].isin(['A', 'B', 'C']) & (table['rv_kms'].abs() < EDGE_KMS - 100)]
        CATALOGUE.update(zip(good['folder'], good['rv_kms'].astype(float)))  # outside the grid is an error (tau Sco)
        CATALOGUE.setdefault('', None)
    return CATALOGUE.get(folder)


def emission_centre(ref, v, cutoff):
    '''Peak of the emission of the median spectrum within +-150 km/s, as asset.X_corr (Gaussian of 0.2 A), or None'''
    near = np.abs(v) < 150
    if not (ref[near] > cutoff).any():
        return None
    step = np.median(np.diff(v))
    sigma = 0.02 / 393.366 * 299792.458 / step  # 0.2 A in pixels
    kernel = np.exp(-np.arange(-int(4 * sigma) - 1, int(4 * sigma) + 2) ** 2 / (2 * sigma ** 2))
    corr = np.convolve(ref - 1, kernel, mode='same')
    return float(v[near][np.argmax(corr[near])])


def load_line(search, path):
    '''Normalised, smoothed spectra that pass the noise cut, as in QuickSearch'''
    search.ccf = False
    out = search.spec_analysis(path)
    return None if out is None else (out[0], search.df.copy(), search.radial_velocity.copy())


def transit_minutes(times, detected):
    '''Bendahan-West t_max: time between the nearest spectra with no detection, before and after'''
    if not detected.any():
        return np.inf  # only the stack of the night reaches the feature, so no single spectrum dates it
    before = np.where(~detected & (times < times[detected].min()))[0]
    after = np.where(~detected & (times > times[detected].max()))[0]
    if len(before) == 0 or len(after) == 0:
        return np.inf  # the event is at an end of the time series: its length is unknown
    return (times[after[0]] - times[before[-1]]) * 1440


def control_lines(ref, v, v_star, line):
    '''Photospheric lines of the window, away from the Ca II line (and H epsilon). Returns centres (km/s in the
    frame of the window), depths and the half width at 10% of the depth of the strongest one'''
    low, high = CA_EXCLUDE_KMS[line]
    peaks, properties = find_peaks(-ref, prominence=CONTROL_PROMINENCE, width=2)
    relative = v[peaks] - v_star
    room = np.abs(v[peaks]) < np.abs(v).max() - 250  # the test shifts the line by up to 200 km/s, plus its box
    keep = ((relative < low) | (relative > high)) & room
    peaks, depths = peaks[keep], properties['prominences'][keep]
    order = np.argsort(depths)[::-1][:MAX_CONTROL_LINES]
    peaks, depths = peaks[order], depths[order]
    half_width = np.nan
    if len(peaks):
        step = np.median(np.diff(v))
        half_width = float(peak_widths(-ref, peaks[:1], rel_height=0.9)[0][0] * step / 2)
    return v[peaks], depths, half_width


def control_scores(spectra, ref, v, centres, depths, offset, fwhm, noise, v_star, line):
    '''Matched filter of each spectrum at centre + offset of each control line, in units of its noise. Returns the
    sums over lines, weighted by line depth, of the score (negative for absorption) and of the score a change of the
    star of unit fractional depth would give, and the sum of the squared weights'''
    sigma = fwhm / FWHM_TO_SIGMA
    low, high = CA_EXCLUDE_KMS[line]
    total, expected, weights = np.zeros(len(spectra)), np.zeros(len(spectra)), 0.0
    for centre, depth in zip(centres, depths):
        if low < centre + offset - v_star < high:
            continue  # shifted onto the Ca II line: its own changes (activity, the comet) would leak in
        near = np.abs(v - centre - offset) < BOX_SIGMAS * sigma
        if near.sum() < 5:
            continue
        dip = np.exp(-(v[near] - centre - offset) ** 2 / (2 * sigma ** 2)) * ref[near]
        g = dip - dip.mean()  # compensated: a broad change of the line scores nothing
        norm = np.sqrt(np.sum(g ** 2))
        total += depth * ((spectra[:, near] - ref[near]) @ g) / (noise * norm)
        expected += depth * depth * (-(dip @ g)) / (norm * noise)
        weights += depth ** 2
    return total, expected, weights


def period_test(times, series, exclude):
    '''Lomb-Scargle of the series at the feature velocity without the night of the feature. Returns the false alarm
    probability of the best period, and a function giving what that period predicts at any time'''
    use = ~exclude & np.isfinite(series)
    if use.sum() < PERIOD_MIN_EPOCHS:
        return np.nan, None
    t, y = times[use], series[use]
    baseline = max(t.max() - t.min(), 1.0)
    per_peak = float(np.clip(PERIOD_MAX_GRID / (PERIOD_MAX_FREQUENCY * baseline), 0.5, 5))
    model = LombScargle(t, y)
    frequency, power = model.autopower(minimum_frequency=1 / baseline, maximum_frequency=PERIOD_MAX_FREQUENCY,
                                       samples_per_peak=per_peak)
    best = int(np.argmax(power))
    fap = float(model.false_alarm_probability(power[best], method='baluev', minimum_frequency=1 / baseline,
                                              maximum_frequency=PERIOD_MAX_FREQUENCY))
    middle = np.median(y)
    return fap, lambda when: float(model.model(np.atleast_1d(when), frequency[best])[0] - middle)


def add_injection(sK, sH, v, nights, times, plan, rng):
    '''Fake comet in consecutive spectra of one night, depth(K) / depth(H) = 1.5 and a slow drift. Returns their indices.
    With plan['share'], a comet-rich star instead: that share of all spectra get a comet at the same velocity, each with
    a depth drawn from plan['depth_range'] (comets at most epochs, as in bet Pic)'''
    if 'share' in plan:
        epochs = np.sort(rng.choice(len(sK), int(plan['share'] * len(sK)), replace=False))
        dip = np.exp(-(v - plan['v0']) ** 2 / (2 * (plan['fwhm'] / FWHM_TO_SIGMA) ** 2))
        for i in epochs:
            depth = rng.uniform(*plan['depth_range'])
            sK[i] = sK[i] * (1 - depth * dip)
            sH[i] = sH[i] * (1 - depth / 1.5 * dip)
        return epochs
    counts = pd.Series(nights).value_counts()
    night = rng.choice(counts.index[:max(1, len(counts) // 2)])
    epochs = np.where(nights == night)[0][:plan['epochs']]
    for i in epochs:
        dip = np.exp(-(v - plan['v0']) ** 2 / (2 * (plan['fwhm'] / FWHM_TO_SIGMA) ** 2))
        sK[i] = sK[i] * (1 - plan['depth'] * dip)
        sH[i] = sH[i] * (1 - plan['depth'] / 1.5 * dip)
    return epochs


def search_star(job, inject=None, keep_state=False):
    '''Rows of one star. keep_state also returns what vet_candidates.py needs to refit and recheck it'''
    folder, class_name = job
    search = Config['search']
    spectral_class = Config['classes'][class_name]
    tests = Config['tests']
    path = search['dataset'] + folder + '/'
    rng = np.random.default_rng(sum(ord(c) * (i + 1) for i, c in enumerate(folder)))

    K, H = load_line(Searches['K'], path), load_line(Searches['H'], path)
    if K is None or H is None:
        return []
    keep = np.intersect1d(K[1]['dp_id'], H[1]['dp_id'])  # the noise cut can keep different spectra in each line
    during = K[1]['t_min'].between(*UPGRADE_MJD, inclusive='left')  # fibre exchange and recommissioning, 2015
    keep = np.setdiff1d(keep, K[1].loc[during, 'dp_id'])
    if len(keep) < spectral_class['min_spectra']:
        return []

    inK, inH = K[1]['dp_id'].isin(keep).to_numpy(), H[1]['dp_id'].isin(keep).to_numpy()
    df = K[1][inK].reset_index(drop=True)
    times = df['t_min'].to_numpy()
    nights = np.floor(times - 0.5)  # MJD changes at midnight UT, near the middle of a La Silla night is not at 0.5
    v_full = K[2]
    pixels = np.abs(v_full) < EDGE_KMS
    v = v_full[pixels]
    sK = K[0][inK].astype(float)[:, pixels]

    # H on the velocity grid of K
    full_H = H[0][inH].astype(float)
    upper = np.clip(np.searchsorted(H[2], v), 1, len(H[2]) - 1)
    fraction = np.clip((v - H[2][upper - 1]) / (H[2][upper] - H[2][upper - 1]), 0, 1)
    sH = full_H[:, upper - 1] * (1 - fraction) + full_H[:, upper] * fraction

    mK, mH = np.median(sK, axis=0), np.median(sH, axis=0)
    # whole windows, for the control lines outside the searched velocities
    wide = {'K': (K[0][inK].astype(np.float32), K[2]), 'H': (full_H.astype(np.float32), H[2])}
    wide = {line: (s, np.median(s, axis=0), v_line) for line, (s, v_line) in wide.items()}
    if search.get('flip'):  # comets only absorb, so whatever is found in upside-down spectra is false
        sK, sH = 2 * mK - sK, 2 * mH - sH
        wide = {line: (2 * m - s, m, v_line) for line, (s, m, v_line) in wide.items()}
    # SIMBAD first: the symmetry of the line fails in many OB and M stars (bug fix 3, 14 Nov 2026)
    v_star, v_star_source = catalogue_velocity(folder), 'catalogue'
    if v_star is None:
        v_star, v_star_source = star_velocity(mK, v), 'symmetry'
        if abs(v_star) > STAR_VELOCITY_LIMIT - STAR_VELOCITY_FAILED:  # the estimate sits at the limit of its search
            v_star, v_star_source = 0.0, 'none'
    relative = v - v_star
    in_range = np.abs(relative) < search['velocity_limit_kms']
    zone = np.abs(v) > 300
    emission = (mK[np.abs(relative) < 100] > search['cutoff']).any()  # same activity rule as ASSET
    mask_kms = spectral_class['activity_mask_kms'] if spectral_class['activity_mask_kms'] else (50 if emission else 0)
    good = np.abs(relative) >= mask_kms
    centre = emission_centre(mK, v, search['cutoff'])
    if centre is not None:  # safety net of the old code: mask the emission wherever it is
        good &= np.abs(v - centre) >= EMISSION_MASK_KMS
    good = good.astype(float)

    injected = None
    if inject is not None:
        plan = dict(inject, v0=v_star + rng.choice([-1, 1]) * rng.uniform(mask_kms + 10, 150))
        injected = add_injection(sK, sH, v, nights, times, plan, rng)

    post = times >= UPGRADE_MJD[1]
    standard_K, standard_H = standardise(sK, mK), standardise(sH, mH)
    spikes_K = spikes_H = None
    if search.get('spike_mask', True):
        narrowest = search['fwhm_kms'][0] / FWHM_TO_SIGMA / np.median(np.diff(v))
        kernel = np.exp(-np.arange(-int(4 * narrowest) - 1, int(4 * narrowest) + 2) ** 2 / (2 * narrowest ** 2))
        depth = [(s - m - standard[2]) / np.maximum(m, 0.05 * np.median(m))
                 for s, m, standard in ((sK, mK, standard_K), (sH, mH, standard_H))]
        spikes_K, spikes_H = spike_masks(standard_K[0], standard_H[0], depth[0], depth[1], kernel)
    masked_px = int(spikes_K.sum() + spikes_H.sum()) if spikes_K is not None else 0
    lineK = prepare_line(sK, mK, v, good, rng, nights, post, standard_K, spikes_K)
    lineH = prepare_line(sH, mH, v, good, rng, nights, post, standard_H, spikes_H)

    n = len(df)
    floor = search['min_line_score']
    score = np.zeros((n, in_range.sum()), np.float32)
    mirror = np.zeros((n, in_range.sum()), np.float32)
    depthK, depthH = np.zeros_like(score), np.zeros_like(score)
    width = np.zeros((n, in_range.sum()), np.int8)
    sample = rng.permutation(n)[:search['null_max_epochs']]
    null_K, null_H, null_depths = {}, {}, {}
    null_ratio = search.get('null_ratio_check', True)  # bug fix 5: the null uses the K/H depth check of the real score
    unique_nights, night_of = np.unique(nights, return_inverse=True)
    stack_K = np.zeros((len(search['fwhm_kms']), len(unique_nights), in_range.sum()), np.float32)
    stack_H = np.zeros_like(stack_K)

    for start in range(0, n, search['chunk']):
        part = np.arange(start, min(start + search['chunk'], n))
        SK, AK, EK = score_line(lineK, part, zone)
        SH, AH, EH = score_line(lineH, part, zone)
        ratio_ok = ratio_check(AK, AH, EK, EH)
        joint = np.where(ratio_ok, doublet(SK, SH, floor), 0.0)[:, :, in_range]
        best = joint.argmax(axis=0)
        rows = np.arange(len(part))[:, None]
        columns = np.arange(joint.shape[2])[None, :]
        score[part] = joint[best, rows, columns]
        width[part] = best
        depthK[part] = AK[:, :, in_range][best, rows, columns]
        depthH[part] = AH[:, :, in_range][best, rows, columns]
        mirror[part] = doublet(-SK, -SH, floor)[:, :, in_range].max(axis=0)
        for w in range(len(search['fwhm_kms'])):
            np.add.at(stack_K[w], night_of[part], SK[w][:, in_range])
            np.add.at(stack_H[w], night_of[part], SH[w][:, in_range])
        chosen = np.isin(part, sample)
        if chosen.any():
            null_K[start], null_H[start] = (SK[:, chosen][:, :, in_range].astype(np.float32),
                                            SH[:, chosen][:, :, in_range].astype(np.float32))
            if null_ratio:
                null_depths[start] = [x[:, chosen][:, :, in_range].astype(np.float32) for x in (AK, AH, EK, EH)]

    # step 6: chance level, for one false alarm budget over the survey (stage 2 decides with p_spec and p_night)
    # the budget is shared by the searches of each star (PCA on and off, combine_runs.py)
    budget = tests['false_alarm_survey'] / tests.get('searches', 1) / search['n_stars']
    budget /= 2 if tests.get('split_spectrum_night', False) else 1  # spectra and night stacks share it (addendum 10)
    depths = ([np.concatenate([d[k] for d in null_depths.values()], axis=1) for k in range(4)] if null_ratio else None)
    null_fit, survival = star_null(np.concatenate(list(null_K.values()), axis=1),
                                   np.concatenate(list(null_H.values()), axis=1), floor, search['null_shifts'], rng,
                                   depths)
    # The aligned upside-down doublets hold the K-H variability of the star. They are only reported (p_var): in a star
    # with comets at most epochs (bet Pic) the reference itself holds comet absorption, so an epoch without comets looks
    # like emission, and this null rises with the comets, as the old one did. Stage 2 keeps the largest of them as a floor
    mirror_fit = tail_fit(mirror.max(axis=1)) if n >= MIN_EPOCHS_VARIABILITY else None
    threshold = fit_threshold(null_fit, budget / n)
    # the slow tests can be computed for more spectra than the threshold keeps (for other decision rules in stage 2)
    slow = search.get('slow_budget_factor', 1)
    slow_threshold = fit_threshold(null_fit, min(budget * slow / n, 0.5)) if slow > 1 else threshold

    root = np.sqrt(np.bincount(night_of))[None, :, None]
    stack_K, stack_H = stack_K / root, stack_H / root
    night_fit, _ = star_null(stack_K, stack_H, floor, search['null_shifts'], rng)
    night_joint = doublet(stack_K, stack_H, floor)
    night_width, night_best = night_joint.argmax(axis=0), night_joint.max(axis=0)
    night_pixel, night_score = night_best.argmax(axis=1), night_best.max(axis=1)
    night_mirror = doublet(-stack_K, -stack_H, floor).max(axis=(0, 2))
    p_stack = exceed_probability(night_score, night_fit)
    night_threshold = fit_threshold(night_fit, budget / len(unique_nights))
    slow_night_threshold = (fit_threshold(night_fit, min(budget * slow / len(unique_nights), 0.5)) if slow > 1
                            else night_threshold)

    grid = relative[in_range]
    step = np.median(np.diff(v))
    core = np.abs(relative) < 100
    slope = np.gradient(mK, v)[core]
    slope = slope - slope.mean()
    raw = sK[:, core] - mK[core]
    raw = raw - raw.mean(axis=1, keepdims=True)
    shift_corr = np.abs(raw @ slope) / np.maximum(np.linalg.norm(raw, axis=1) * np.linalg.norm(slope), 1e-12)
    centre = sK[:, np.abs(relative) < 10].mean(axis=1)
    centre_rank = pd.Series(centre).rank().to_numpy()
    feature_z = (lineK['z'][:, in_range] + lineH['z'][:, in_range]) / 2  # both lines, for the period test

    # control lines of each window, and the noise of each spectrum away from the Ca II line
    controls = {}
    for line, (s_wide, m_wide, v_line) in wide.items():
        centres, depths, half = control_lines(m_wide, v_line, v_star, line)
        low, high = CA_EXCLUDE_KMS[line]
        away = ((v_line - v_star) < low) | ((v_line - v_star) > high)
        resid = s_wide[:, away] - m_wide[away]
        noise = MAD_TO_SIGMA * np.median(np.abs(resid - np.median(resid, axis=1, keepdims=True)), axis=1)
        controls[line] = dict(spectra=s_wide, ref=m_wide, v=v_line, centres=centres, depths=depths, half=half,
                              noise=np.maximum(noise, 1e-6))
    strongest = [(c['depths'][0], c['half']) for c in controls.values() if len(c['depths'])]
    photosphere = float(max(strongest)[1]) if strongest else np.nan
    n_control = int(sum(len(c['centres']) for c in controls.values()))

    night_cache, period_cache, control_cache = {}, {}, {}

    def night_scores(night):
        '''Scores of every width for the spectra of one night (width, epoch, velocity)'''
        if night not in night_cache:
            members = np.where(night_of == night)[0]
            SKn = score_line(lineK, members, zone)[0][:, :, in_range]
            SHn = score_line(lineH, members, zone)[0][:, :, in_range]
            night_cache[night] = (members, SKn, SHn)
        return night_cache[night]

    rows = []
    night_sizes = np.bincount(night_of)
    for i in range(n):
        j = int(np.argmax(score[i]))
        night = night_of[i]
        night_counts = night_score[night] > min(night_threshold, search['diagnostic_score'])
        if night_counts:  # the stack found a feature: read this spectrum at that velocity
            reach = max(int(search['fwhm_kms'][night_width[night, night_pixel[night]]] / step), 1)
            first = max(night_pixel[night] - reach, 0)
            j = first + int(np.argmax(score[i, first:night_pixel[night] + reach + 1]))
        row = dict(folder=folder, class_name=class_name, n=n, n_nights=len(unique_nights), n_stars=search['n_stars'], v_star_source=v_star_source, pca=search.get('pca', True), dp_id=df['dp_id'].iloc[i],
                   t=times[i], score=float(score[i, j]), p_spec=float(exceed_probability(score[i, j], null_fit)),
                   p_var=float(exceed_probability(score[i, j], mirror_fit)) if mirror_fit else np.nan,
                   mirror=float(mirror[i].max()), threshold=threshold, v0=float(grid[j]), v_star=float(v_star),
                   rank_K=lineK['rank'], rank_H=lineH['rank'], night_score=float(night_score[night]),
                   p_night=float(p_stack[night]), night_mirror=float(night_mirror[night]),
                   night_threshold=night_threshold, night_v0=float(grid[night_pixel[night]]),
                   night_size=int(night_sizes[night]), masked_px=masked_px)
        if injected is not None:
            row['injected'] = i in injected
            row['near_injection'] = abs(row['v0'] + v_star - plan['v0']) < max(plan['fwhm'], 5)
            if spikes_K is not None and i in injected:  # share of the comet's pixels the spike mask removed
                at = np.abs(v - plan['v0']) < max(plan['fwhm'], 5)
                row['injected_masked'] = float((spikes_K[i, at] | spikes_H[i, at]).mean())
            if i in injected:  # the rule of QuickSearch V2 on the K line, at the same place
                resid = sK[i] - mK
                sig = np.where(np.abs(v - plan['v0']) < max(plan['fwhm'], 5), resid / resid.std(), np.nan)
                row['v2_hit'] = bool(np.nanmin(sig) < search['threshold'] and Searches['K'].get_width(sig) >= search['width_filt'])
        if row['score'] > search['diagnostic_score'] or (night_counts and row['score'] > 0):
            w = int(width[i, j])
            fwhm = search['fwhm_kms'][w]
            reach = max(int(fwhm / step), 1)
            near = slice(max(j - reach, 0), j + reach + 1)
            wide_part = slice(max(j - 3 * reach, 0), j + 3 * reach + 1)
            window = (near.stop - near.start) / reach  # independent velocities searched around the feature
            detected = score[:, near].max(axis=1) >= search['confirm_score']
            same_night = nights == nights[i]
            profile = np.exp(-(v[in_range] - v[in_range][j]) ** 2 / (2 * (fwhm / FWHM_TO_SIGMA) ** 2))
            K_part, H_part = lineK['z'][i][in_range][wide_part], lineH['z'][i][in_range][wide_part]
            shape = [np.corrcoef(np.nan_to_num(part), -profile[wide_part])[0, 1] for part in (K_part, H_part)]
            at_feature = lineK['z'][:, in_range][:, near].mean(axis=1)
            row.update(fwhm=fwhm, depth_K=float(depthK[i, j]), depth_H=float(depthH[i, j]),
                       transit_min=transit_minutes(times, detected), shift_corr=float(shift_corr[i]),
                       night_spectra=int(same_night.sum()), night_confirmed=int((detected & same_night).sum()),
                       emission_twin=float(mirror[i, wide_part].max() / row['score']), shape_K=float(shape[0]), shape_H=float(shape[1]),
                       activity_corr=abs(float(np.corrcoef(pd.Series(at_feature).rank().to_numpy(), centre_rank)[0, 1])))

            row.update(p_repeat=1.0, p_halves=1.0, n_control=n_control, control_score=np.nan, control_expected=np.nan,
                       control_corr=np.nan, control_spread=np.nan, photosphere_kms=photosphere,
                       period_fap=np.nan, period_share=np.nan)
            x = float(grid[j])
            row['inside_photosphere'] = bool(np.isfinite(photosphere) and abs(x) < photosphere)
            # H epsilon sits +121 km/s from Ca II H in the frame of the star
            row['near_h_epsilon'] = bool(abs(x - H_EPSILON_KMS) < H_EPSILON_HALF_KMS)
            if not (row['score'] > slow_threshold or night_score[night] > slow_night_threshold):
                rows.append(row)  # cannot pass the survey threshold: the slow tests are not needed
                continue

            # repeat: chance that another spectrum of the night shows a doublet this large at the same velocity
            members, SKn, SHn = night_scores(night)
            others = members != i
            if others.any():
                repeat = doublet(SKn[w][others][:, near], SHn[w][others][:, near], floor).max(axis=1)
                row['p_repeat'] = float(min(1.0, local_probability(repeat, survival).min() * others.sum() * window))
            # halves of the night: each stacked on its own, at the velocity and width of the night stack
            if night_counts and len(members) >= 2:
                nw, npix = int(night_width[night, night_pixel[night]]), int(night_pixel[night])
                n_reach = max(int(search['fwhm_kms'][nw] / step), 1)
                n_near = slice(max(npix - n_reach, 0), npix + n_reach + 1)
                order = np.argsort(times[members])
                half_p = []
                for half in (order[:len(order) // 2], order[len(order) // 2:]):
                    hK = SKn[nw][half].sum(axis=0) / np.sqrt(len(half))
                    hH = SHn[nw][half].sum(axis=0) / np.sqrt(len(half))
                    half_p.append(local_probability(doublet(hK[n_near], hH[n_near], floor).max(), survival)[0])
                row['p_halves'] = float(min(1.0, max(half_p) * (n_near.stop - n_near.start) / n_reach))

            # control lines: the same offset from the centre of photospheric lines, in the same spectrum
            if (w, j) not in control_cache:  # the same for every spectrum at this velocity and width
                total, expected, weights = np.zeros(n), np.zeros(n), 0.0
                for line, c in controls.items():
                    tl, el, wl = control_scores(c['spectra'], c['ref'], c['v'], c['centres'], c['depths'], x, fwhm,
                                                c['noise'], v_star, line)
                    total, expected, weights = total + tl, expected + el, weights + wl
                control_cache[(w, j)] = (total, expected, weights)
            total, expected, weights = control_cache[(w, j)]
            if weights > 0:
                control = total / np.sqrt(weights)
                line_depth = max(1 - float(mK[in_range][j]), 0.05)  # depth of Ca II K at the feature
                share = min(max(row['depth_K'], 0) / line_depth, 1.0)
                row['control_score'] = float(control[i])
                row['control_expected'] = float(expected[i] / np.sqrt(weights) * share)
                # spread over all spectra, in noise units: about 1 if the noise of the control lines is right
                row['control_spread'] = float(MAD_TO_SIGMA * np.median(np.abs(control - np.median(control))))
                if n >= 10:
                    row['control_corr'] = float(spearmanr(feature_z[:, near].mean(axis=1), control)[0])

            # period: a periodic signal at this velocity, fitted without this night, once per velocity and night
            key = (w, j // reach, night)
            if key not in period_cache:
                series = feature_z[:, near].mean(axis=1)
                period_cache[key] = (series, *period_test(times, series, same_night))
            series, fap, predict = period_cache[key]
            row['period_fap'] = fap
            if predict is not None and series[i] != 0:
                row['period_share'] = predict(times[i]) / float(series[i])
        rows.append(row)
    if keep_state:
        state = dict(lineK=lineK, lineH=lineH, v=v, in_range=in_range, grid=grid, df=df, floor=floor,
                     null_K=np.concatenate(list(null_K.values()), axis=1),
                     null_H=np.concatenate(list(null_H.values()), axis=1), null_fit=null_fit, mirror_fit=mirror_fit)
        return rows, state
    return rows


def init_worker(config):
    global Config, Searches
    Config = config
    parameters = dict(config['search'], min_spectra=3)
    Searches = {line: ASSET_LocalNoise(parameters=parameters, line=line) for line in 'KH'}


def build_jobs(config):
    '''Stars with an O-M type and enough spectra for their class'''
    index = pd.read_csv(config['search']['dataset'] + 'dataset_index.csv', dtype={'gaia_dr3': str})
    groups = pd.read_pickle(os.path.join(config['search']['dataset'], '..', 'grouping', 'v2', 'harps_groups_v2.pkl'))
    index = index.merge(groups[['star_id', 'sp_type']], on='star_id', how='left')
    index['class_name'] = [class_of(s, config['classes']) if isinstance(s, str) else None for s in index['sp_type']]
    print(index['class_name'].fillna('not searched').value_counts().to_string(), flush=True)
    index = index.dropna(subset=['class_name'])
    return index[[n >= config['classes'][c]['min_spectra'] for n, c in zip(index['n_spectra'], index['class_name'])]]


if __name__ == '__main__':
    startTime = time.time()

    parser = argparse.ArgumentParser(description='Exocomet search by spectral class, stage 1')
    parser.add_argument('--c', default='spectral_classes.json', help='config file (default: spectral_classes.json)')
    parser.add_argument('--r', default='../results/Reliable/', help='output path (default: ../results/Reliable/)')
    parser.add_argument('--flip', action='store_true', help='turn the spectra upside down to measure false detections')
    parser.add_argument('--stars', default=None, help='text file of folders: only search these (for testing)')
    parser.add_argument('--no-pca', action='store_true', help='second search without principal components, for stars '
                        'with comets at most epochs (bet Pic), where PCA learns the comets as part of the star')
    args = parser.parse_args()

    with open(args.c) as configfile:
        config = json.load(configfile)

    config['search']['flip'] = args.flip
    config['search']['pca'] = not args.no_pca
    index = build_jobs(config)
    config['search']['n_stars'] = len(index)  # trials of the survey: every star that can be searched
    if args.stars:
        with open(args.stars) as starfile:
            index = index[index['folder'].isin(starfile.read().split())]
    jobs = list(zip(index['folder'], index['class_name']))
    print('Stars to search:', len(jobs), flush=True)

    rows = []
    with multiprocessing.Pool(config['search']['cores'], initializer=init_worker, initargs=(config,)) as pool:
        for i, star_rows in enumerate(pool.imap_unordered(search_star, jobs)):
            rows += star_rows
            if (i + 1) % 250 == 0:
                print('{}/{} stars, {:.0f} s'.format(i + 1, len(jobs), time.time() - startTime), flush=True)

    os.makedirs(args.r, exist_ok=True)
    pd.DataFrame(rows).to_pickle(os.path.join(args.r, 'spectrum_scores.pkl'))
    if not args.stars:  # stars of each class in the survey, the trials of the type-weighted budget (stage 2)
        with open(os.path.join(args.r, 'class_counts.json'), 'w') as handle:
            json.dump(index['class_name'].value_counts().to_dict(), handle)
    print('Spectra scored: {}, {:.0f} s'.format(len(rows), time.time() - startTime))
