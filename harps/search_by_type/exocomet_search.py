'''
01/11/26
Exocomet search by spectral class. Stage 1 of 2: scores every spectrum, then calibrate_by_type.py decides.

Same normalisation, smoothing and noise cut as QuickSearch, on K and H together. Stars with no O-M type are not searched.
After that, for each star:
1. local noise of quicksearch_v3 and removal of the slow local median of the residual
2. removal of what the star does at many epochs: a line shift (the first two derivatives of the reference) at any n,
   plus principal components at 20+ spectra. Components come from other nights, so a transient cannot remove itself
3. noise scaled by the scatter of each velocity over time, at 10+ spectra (the star's normal variability)
4. matched filter of Eq. 6 of the article with T = Gaussian * reference, since a comet removes a fraction of the
   stellar flux, and a local box subtracted, so structure broader than the feature scores nothing
5. one feature in K and H: same velocity and width, depth(K) / depth(H) between 1 and 2 within errors
6. chance level of each star: K against H at a wrong velocity keeps the star's own noise, without a real doublet
7. the same score on the stack of each night: a comet lasts hours, so a faint one adds up over the exposures of a night

Each spectrum gets one row. Spectra above the threshold of the star also get the values of the tests of false positives.
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
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'search'))
from quicksearch_v3 import ASSET_LocalNoise, MAD_TO_SIGMA

FWHM_TO_SIGMA = 2.354820045
SPECTRAL_ORDER = 'OBAFGKM'
EDGE_KMS = 420  # work on |v| below this. 300-420 km/s holds no signal and measures the noise of the scores
BOX_SIGMAS = 8  # half width of the box of the compensated filter
FOLDS = 8
MIN_NIGHTS_PCA = 5
MIN_EPOCHS_PCA = 20
MIN_EPOCHS_TEMPORAL = 10
PRIOR_EPOCHS = 20  # strength of the shrinkage of the temporal scatter towards 1 (Smyth 2004)
PCA_MAX_EPOCHS = 300
TAIL_FRACTION = 0.05  # share of the null used to fit its tail

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
    use_pca = len(z) >= MIN_EPOCHS_PCA and len(unique) >= MIN_NIGHTS_PCA
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
    if len(z) < MIN_EPOCHS_TEMPORAL:
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


def prepare_line(spectra, ref, v, good, rng, nights):
    '''Steps 1-3 for one line. Returns residual, reference, inverse variance and the share that survives'''
    search = Config['search']
    step = np.median(np.diff(v))
    kernels = [np.exp(-np.arange(-int(4 * s) - 1, int(4 * s) + 2) ** 2 / (2 * s ** 2))
               for s in (f / FWHM_TO_SIGMA / step for f in search['fwhm_kms'])]
    sigma, baseline = [], []
    for start in range(0, len(spectra), search['chunk']):
        s, b = local_noise(spectra[start:start + search['chunk']] - ref, search['local_window'], search['local_guard'],
                           search['local_step'])
        sigma.append(s)
        baseline.append(b)
    sigma, baseline = np.vstack(sigma), np.vstack(baseline)
    sigma[sigma <= 1e-3 * np.median(sigma, axis=1, keepdims=True)] = np.inf
    z = (spectra - ref - baseline) / sigma

    mean_sigma = np.median(np.where(np.isfinite(sigma), sigma, np.nan), axis=0)
    slope = np.gradient(ref, v)
    fixed = np.array([slope / mean_sigma, np.gradient(slope, v) / mean_sigma])
    fixed = np.nan_to_num(fixed / np.maximum(np.linalg.norm(np.nan_to_num(fixed), axis=1, keepdims=True), 1e-12))
    cleaned, fold_of, survives, rank = remove_modes(z, nights, fixed, kernels, rng)
    scatter = temporal_scatter(cleaned)
    noise = sigma * scatter
    return dict(z=cleaned / scatter, resid=cleaned * sigma, inv_var=good / noise ** 2, fold_of=fold_of, survives=survives,
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


def fit_threshold(null, probability):
    '''Value exceeded with this probability, with a generalised Pareto tail (Coles 2001) fitted by moments'''
    null = np.sort(null)
    start = null[int((1 - TAIL_FRACTION) * len(null))]
    excess = null[null > start] - start
    if probability >= TAIL_FRACTION or len(excess) < 20 or excess.std() == 0:
        return float(np.quantile(null, 1 - probability))
    ratio = excess.mean() ** 2 / excess.var()
    shape = float(np.clip(0.5 * (1 - ratio), -0.2, 0.5))
    scale = 0.5 * excess.mean() * (ratio + 1)
    level = TAIL_FRACTION / probability
    return float(start + (scale * np.log(level) if abs(shape) < 1e-3 else scale / shape * (level ** shape - 1)))


def star_null(SK, SH, floor, shifts, n_stars_epochs, far_per_star, rng):
    '''Threshold of a star: maximum per epoch of K against H at a wrong velocity (shape width, epoch, velocity)'''
    pixels = SK.shape[2]
    low = min(150, pixels // 3)
    maxima = []
    for shift in rng.integers(low, pixels - low, shifts):
        best = doublet(SK, np.roll(SH, shift, axis=2), floor).max(axis=(0, 2))
        maxima.append(best)
    return fit_threshold(np.concatenate(maxima), far_per_star / n_stars_epochs)


def star_velocity(ref, v):
    '''Centre of the line: the velocity about which the reference is most symmetric. The lowest point of the line fails
    in active stars, where emission splits the core (eps Eri: -66 km/s, true +16). Within 3 km/s on five known stars'''
    step = np.median(np.diff(v))
    offsets = np.arange(int(5 / step), int(120 / step))
    centres = np.where((np.abs(v) < 200) & (np.arange(len(v)) >= offsets[-1]) & (np.arange(len(v)) < len(v) - offsets[-1]))[0]
    difference = [np.mean((ref[c + offsets] - ref[c - offsets]) ** 2) for c in centres]
    return v[centres[int(np.argmin(difference))]]


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


def add_injection(sK, sH, v, nights, times, plan, rng):
    '''Fake comet in consecutive spectra of one night, depth(K) / depth(H) = 1.5 and a slow drift. Returns their indices'''
    counts = pd.Series(nights).value_counts()
    night = rng.choice(counts.index[:max(1, len(counts) // 2)])
    epochs = np.where(nights == night)[0][:plan['epochs']]
    for i in epochs:
        dip = np.exp(-(v - plan['v0']) ** 2 / (2 * (plan['fwhm'] / FWHM_TO_SIGMA) ** 2))
        sK[i] = sK[i] * (1 - plan['depth'] * dip)
        sH[i] = sH[i] * (1 - plan['depth'] / 1.5 * dip)
    return epochs


def search_star(job, inject=None):
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
    if search.get('flip'):  # comets only absorb, so whatever is found in upside-down spectra is false
        sK, sH = 2 * mK - sK, 2 * mH - sH
    v_star = star_velocity(mK, v)
    relative = v - v_star
    in_range = np.abs(relative) < search['velocity_limit_kms']
    zone = np.abs(v) > 300
    emission = (mK[np.abs(relative) < 100] > search['cutoff']).any()  # same activity rule as ASSET
    mask_kms = spectral_class['activity_mask_kms'] if spectral_class['activity_mask_kms'] else (50 if emission else 0)
    good = (np.abs(relative) >= mask_kms).astype(float)

    injected = None
    if inject is not None:
        plan = dict(inject, v0=v_star + rng.choice([-1, 1]) * rng.uniform(mask_kms + 10, 150))
        injected = add_injection(sK, sH, v, nights, times, plan, rng)

    lineK = prepare_line(sK, mK, v, good, rng, nights)
    lineH = prepare_line(sH, mH, v, good, rng, nights)

    n = len(df)
    floor = search['min_line_score']
    score = np.zeros((n, in_range.sum()), np.float32)
    mirror = np.zeros((n, in_range.sum()), np.float32)
    depthK, depthH = np.zeros_like(score), np.zeros_like(score)
    width = np.zeros((n, in_range.sum()), np.int8)
    sample = rng.permutation(n)[:search['null_max_epochs']]
    null_K, null_H = {}, {}
    unique_nights, night_of = np.unique(nights, return_inverse=True)
    stack_K = np.zeros((len(search['fwhm_kms']), len(unique_nights), in_range.sum()), np.float32)
    stack_H = np.zeros_like(stack_K)

    for start in range(0, n, search['chunk']):
        part = np.arange(start, min(start + search['chunk'], n))
        SK, AK, EK = score_line(lineK, part, zone)
        SH, AH, EH = score_line(lineH, part, zone)
        ratio_ok = (AK >= AH - 2 * np.hypot(EK, EH)) & (AK <= 2 * AH + 2 * np.hypot(EK, 2 * EH))
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

    threshold = star_null(np.concatenate(list(null_K.values()), axis=1), np.concatenate(list(null_H.values()), axis=1),
                          floor, search['null_shifts'], n, tests['false_alarm_per_star'], rng)

    root = np.sqrt(np.bincount(night_of))[None, :, None]
    stack_K, stack_H = stack_K / root, stack_H / root
    night_threshold = star_null(stack_K, stack_H, floor, search['null_shifts'], len(unique_nights),
                                tests['false_alarm_per_star'], rng)
    night_joint = doublet(stack_K, stack_H, floor)
    night_width, night_best = night_joint.argmax(axis=0), night_joint.max(axis=0)
    night_pixel, night_score = night_best.argmax(axis=1), night_best.max(axis=1)
    night_mirror = doublet(-stack_K, -stack_H, floor).max(axis=(0, 2))

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
        row = dict(folder=folder, class_name=class_name, n=n, dp_id=df['dp_id'].iloc[i], t=times[i],
                   score=float(score[i, j]), mirror=float(mirror[i].max()), threshold=threshold, v0=float(grid[j]),
                   rank_K=lineK['rank'], rank_H=lineH['rank'], night_score=float(night_score[night]),
                   night_mirror=float(night_mirror[night]), night_threshold=night_threshold,
                   night_v0=float(grid[night_pixel[night]]), night_size=int(night_sizes[night]))
        if injected is not None:
            row['injected'] = i in injected
            row['near_injection'] = abs(row['v0'] + v_star - plan['v0']) < max(plan['fwhm'], 5)
            if i in injected:  # the rule of QuickSearch V2 on the K line, at the same place
                resid = sK[i] - mK
                sig = np.where(np.abs(v - plan['v0']) < max(plan['fwhm'], 5), resid / resid.std(), np.nan)
                row['v2_hit'] = bool(np.nanmin(sig) < search['threshold'] and Searches['K'].get_width(sig) >= search['width_filt'])
        if row['score'] > search['diagnostic_score'] or (night_counts and row['score'] > 0):
            fwhm = search['fwhm_kms'][width[i, j]]
            reach = max(int(fwhm / step), 1)
            near = slice(max(j - reach, 0), j + reach + 1)
            wide = slice(max(j - 3 * reach, 0), j + 3 * reach + 1)
            detected = score[:, near].max(axis=1) >= search['confirm_score']
            same_night = nights == nights[i]
            profile = np.exp(-(v[in_range] - v[in_range][j]) ** 2 / (2 * (fwhm / FWHM_TO_SIGMA) ** 2))
            K_part, H_part = lineK['z'][i][in_range][wide], lineH['z'][i][in_range][wide]
            shape = [np.corrcoef(np.nan_to_num(part), -profile[wide])[0, 1] for part in (K_part, H_part)]
            at_feature = lineK['z'][:, in_range][:, near].mean(axis=1)
            row.update(fwhm=fwhm, depth_K=float(depthK[i, j]), depth_H=float(depthH[i, j]),
                       transit_min=transit_minutes(times, detected), shift_corr=float(shift_corr[i]),
                       night_spectra=int(same_night.sum()), night_confirmed=int((detected & same_night).sum()),
                       emission_twin=float(mirror[i, wide].max() / row['score']), shape_K=float(shape[0]), shape_H=float(shape[1]),
                       activity_corr=abs(float(np.corrcoef(pd.Series(at_feature).rank().to_numpy(), centre_rank)[0, 1])))
        rows.append(row)
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
    parser.add_argument('--r', default='../results/ByType/', help='output path (default: ../results/ByType/)')
    parser.add_argument('--flip', action='store_true', help='turn the spectra upside down to measure false detections')
    parser.add_argument('--stars', default=None, help='text file of folders: only search these (for testing)')
    args = parser.parse_args()

    with open(args.c) as configfile:
        config = json.load(configfile)

    config['search']['flip'] = args.flip
    index = build_jobs(config)
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
    print('Spectra scored: {}, {:.0f} s'.format(len(rows), time.time() - startTime))
