'''
09/11/26
Veto step: does the star itself explain a Ca II detection?

For every detection above threshold (real run, flipped run) and for the best night of the known hosts (bet Pic,
HD 172555, as positive controls), two checks on the same spectrum, with lines chosen by the star's spectral type:

1. Dip at the comet velocity (in the flipped run, where the veto lines are flipped like Ca II: a brightening) in a photospheric line from a level at or above 6 eV. Comet gas has been seen absorbing
   from levels up to 4.1 eV (Vrignaud & Lecavelier des Etangs 2024), so a dip in such a line at the velocity of the
   detection comes from the star (pulsation, spots, a changing blend) and flags the detection as "stellar dip". The
   test needs the line in the star: if the comet velocity lies outside the photospheric profile, the result is
   "not testable".
2. Activity in the same spectrum: an H alpha or He I D3 index more than 3 MAD from the star's other spectra (either
   sign: He I D3 can anti-correlate with Ca II, Santos et al. 2010) flags the detection as "activity". Index bands
   follow Gomes da Silva et al. 2011, with local continuum bands for He I.

Both checks only flag: a detection is never removed here (the user's choice). Young or accreting stars (Herbig Ae/Be,
T Tauri) are labelled, because infalling gas can also change H alpha and He I there (Mora et al. 2002).

Telluric water lines move across the stellar spectrum through the year (barycentric correction), so pixels where the
telluric template is more than 1% deep are ignored. The template comes from the data: bet Pic's spectra shifted to
the Earth frame, high-pass filtered and median-combined; the star's broad lines smear out over the year and the
narrow telluric lines stay.

Each line score is a compensated Gaussian filter (FWHM of the detection) on the spectrum divided by the median of the
star's spectra from other nights; its noise is the MAD of the same filter on the reference spectra themselves.

Stage 2 (calibrate_by_type.py) calls veto_stage on its detections. On its own, without rerunning stage 2:
  py -3 veto_check.py                                   # real run, ../results/Reliable/
  py -3 veto_check.py --flip --r ../results/Reliable_flip/
'''
import argparse
import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from external import SIMBAD_TAP, tap

C_KMS = 299792.458
FWHM_TO_SIGMA = 2.354820045
MAD_TO_SIGMA = 1.4826
VETO_DIR = 'C:/data/harps/'
BET_PIC, HD172555 = 'GaiaDR3_4792774797545800832', 'GaiaDR3_6438274350302427776'
CONTROL_NIGHTS = 15  # best nights of each known host used as positive controls
MAX_REFERENCE = 300  # spectra in a reference median (bet Pic has 9,000)
SPREAD_SPECTRA = 60  # reference spectra used to measure the scatter of a score
DIP_SIGMA, ACTIVITY_SIGMA = 3.0, 3.0
TELLURIC_DEPTH = 0.01
HIGH_PASS_NM = 0.3  # running median that removes the broad shape before the telluric template
TEMPLATE = VETO_DIR + 'telluric_template.npz'
TELLURIC_BANDS = ('Halpha', 'HeI5876_NaD', 'SiII6347_6371', 'OI6158', 'CI5380', 'CI5052')

# rest wavelengths in air (nm) and lower-level energies from NIST ASD, checked 9 Nov 2026
LINES = {'HeII4542': ('HeII4542_SiIII4553', [454.163]), 'HeI4471': ('HeI4471_MgII4481', [447.148]),
         'SiIII': ('HeII4542_SiIII4553', [455.262, 456.784, 457.475]), 'MgII4481': ('HeI4471_MgII4481', [448.113]),
         'SiII': ('SiII6347_6371', [634.711, 637.137]), 'CI': (None, [505.215, 538.033])}
CI_BANDS = {505.215: 'CI5052', 538.033: 'CI5380'}
# activity index: core (centre, half width) and continuum bands, nm, in the stellar rest frame
INDICES = {'Halpha': ('Halpha', (656.2808, 0.08), [(655.087, 0.5375), (658.031, 0.4375)]),
           'HeID3': ('HeI5876_NaD', (587.562, 0.02), [(586.9, 0.25), (588.1, 0.25)])}


def rule_for(sp_type):
    '''Dip lines and activity indices for a spectral type (see the veto line table in the README)'''
    m = re.match(r'\s*(?:sd|d|k)?([OBAFGKM])\s*(\d+(?:\.\d+)?)?', sp_type or '')
    if not m:
        return 'unknown', [], []
    letter, sub = m.group(1), float(m.group(2)) if m.group(2) else 5.0
    if letter == 'O':
        return 'O', ['HeII4542', 'HeI4471'], ['Halpha']
    if letter == 'B':
        return ('B0-B3', ['SiIII', 'HeI4471'], ['Halpha']) if sub <= 3 else ('B4-B9', ['MgII4481', 'HeI4471'], ['Halpha'])
    if letter == 'A':
        return 'A', ['MgII4481', 'SiII'], []  # no chromosphere: pulsation shows in the dip lines
    if letter == 'F':
        return ('F0-F4', ['MgII4481'], []) if sub < 5 else ('F5-G9', ['CI'], ['HeID3'])
    if letter == 'G':
        return 'F5-G9', ['CI'], ['HeID3']
    if letter == 'K':
        # C I is reliable down to Teff = 5200 K, about K1 (Delgado Mena et al. 2021)
        return ('K0-K1', ['CI'], ['HeID3', 'Halpha']) if sub <= 1 else ('K2-K9', [], ['Halpha'])
    return 'M', [], ['Halpha', 'HeID3']


def load_band(files, band):
    '''Spectra of one band on a common grid (nm, barycentric); each spectrum divided by its median'''
    first = np.load(files[0])
    w0, step, n = float(first[f'wave0_{band}']), float(first[f'step_{band}']), first[f'flux_{band}'].size
    grid = w0 + step * np.arange(n)
    flux = np.empty((len(files), n), np.float32)
    for i, f in enumerate(files):
        d = np.load(f)
        x = float(d[f'wave0_{band}']) + float(d[f'step_{band}']) * np.arange(d[f'flux_{band}'].size)
        flux[i] = np.interp(grid, x, d[f'flux_{band}'], left=np.nan, right=np.nan)
    flux /= np.nanmedian(flux, axis=1, keepdims=True)
    return grid, flux


def telluric_template(files, berv, band):
    '''Median of high-pass filtered spectra in the Earth frame: 1 where clean, < 1 on telluric lines'''
    grid, flux = load_band(files, band)
    size = int(HIGH_PASS_NM / (grid[1] - grid[0]))
    out = []
    for f, b in zip(flux, berv):
        f = np.where(np.isfinite(f), f, 1.0)
        f = f / median_filter(f, size=size, mode='nearest')
        out.append(np.interp(grid, grid / (1 + b / C_KMS), f))  # barycentric -> observed (Earth) wavelengths
    return grid, np.median(out, axis=0)


def compensated(grid, centre, fwhm_kms):
    sigma = centre * fwhm_kms / FWHM_TO_SIGMA / C_KMS
    near = np.abs(grid - centre) < 8 * sigma
    g = np.exp(-(grid[near] - centre) ** 2 / (2 * sigma ** 2))
    return near, (g - g.mean()) / np.sqrt(np.sum((g - g.mean()) ** 2))


def flatten(ratio, grid):
    '''Remove the slow shape of a spectrum/reference ratio (blaze, order merging)'''
    size = int(HIGH_PASS_NM / (grid[1] - grid[0]))
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    return ratio / median_filter(ratio, size=size, mode='nearest')


def dip_sigma(grid, flux, good, i, ref_idx, spread_idx, centres, fwhm):
    '''Score of a dip at the given wavelengths in spectrum i (positive = deeper than the reference), in units of the
    scatter of the same score on reference spectra'''
    ref = np.nanmedian(flux[ref_idx], axis=0)
    filters = [compensated(grid, c, fwhm) for c in centres]

    def score(spectrum, reference):
        r = flatten(spectrum / reference, grid) - 1
        return -sum(float(np.where(good[near], r[near], 0) @ g) for near, g in filters) / np.sqrt(len(filters))

    value = score(flux[i], ref)
    others = [score(flux[j], np.nanmedian(flux[[k for k in ref_idx if k != j][:MAX_REFERENCE]], axis=0))
              for j in spread_idx]
    spread = MAD_TO_SIGMA * np.median(np.abs(np.array(others) - np.median(others)))
    return value / spread if spread > 0 else np.nan


def line_present(grid, ref, centre, fwhm, v0):
    '''Depth of the photospheric line in the reference and whether the comet velocity falls inside its profile'''
    r = ref / np.nanpercentile(ref, 95)  # continuum: the cut-outs are a few nm, so the top pixels are close to it
    v = (grid / centre - 1) * C_KMS
    core = np.abs(v) < 30
    if not core.any():
        return 0.0, False
    depth = float(1 - np.nanmin(r[core]))
    inside = r < 1 - 0.1 * depth  # the profile down to 10% of the depth
    width = np.abs(v[inside]).max() if inside.any() and depth > 0 else 0
    return depth, depth >= 0.01 and abs(v0) <= width + fwhm / 2


def index_values(grid, flux, good, v_star, name):
    _, (centre, half), refs = INDICES[name]
    shift = 1 + v_star / C_KMS

    masked = np.where(good, flux, np.nan)  # telluric pixels of each spectrum left out of its own index

    def band_mean(c, h):
        sel = np.abs(grid - c * shift) <= h
        enough = np.sum(np.isfinite(masked[:, sel]), axis=1) >= 5
        return np.where(enough, np.nanmean(masked[:, sel], axis=1), np.nan)

    return band_mean(centre, half) / np.nanmean([band_mean(c, h) for c, h in refs], axis=0)


def young_stars(labels):
    names = "', '".join(n.replace("'", "''") for n in labels)
    try:
        t = tap(SIMBAD_TAP, f"SELECT i.id, b.otype FROM ident AS i JOIN basic AS b ON b.oid = i.oidref "
                            f"WHERE i.id IN ('{names}')")
        young = {'Ae*', 'Be*', 'TT*', 'Y*O', 'Or*', 'HH', 'Y*?', 'TT?', 'Ae?'}
        return {r['id']: r['otype'] for _, r in t.iterrows() if str(r['otype']).strip() in young}
    except Exception as error:
        print('SIMBAD unavailable, young stars only from the spectral type:', type(error).__name__)
        return {}


def check_star(folder, detections, sp_type, files_df, mask, young, sign=1):
    rule, dips, indices = rule_for(sp_type)
    bands = sorted({LINES[d][0] for d in dips if LINES[d][0]} | {CI_BANDS[c] for d in dips if d == 'CI'
                                                                 for c in LINES['CI'][1]}
                   | {INDICES[x][0] for x in indices})
    files = [VETO_DIR + f for f in files_df['file']]
    nights = np.floor(files_df['t'].to_numpy() - 0.5)
    berv = files_df['berv'].to_numpy()
    pos = {d: k for k, d in enumerate(files_df['dp_id'])}
    data = {}
    for band in bands:
        grid, flux = load_band(files, band)
        good = np.ones((len(files), grid.size), bool)
        if band in mask:
            for k, b in enumerate(berv):  # telluric template in the Earth frame moved to each spectrum's frame
                good[k] = np.interp(grid / (1 + b / C_KMS), mask[band][0], mask[band][1]) > 1 - TELLURIC_DEPTH
        data[band] = (grid, flux, good)
    rows = []
    rng = np.random.default_rng(0)
    for _, det in detections.iterrows():
        if det['dp_id'] not in pos:
            rows.append(dict(folder=folder, dp_id=det['dp_id'], veto='no veto data'))
            continue
        i = pos[det['dp_id']]
        other = np.flatnonzero(nights != np.floor(det['t'] - 0.5))
        ref_idx = rng.choice(other, min(MAX_REFERENCE, other.size), replace=False) if other.size else other
        spread_idx = ref_idx[:SPREAD_SPECTRA]
        out = dict(folder=folder, dp_id=det['dp_id'], sp_type=sp_type, rule=rule, v0=det['v0'], fwhm=det['fwhm'],
                   score=det['score'], young=young, n_reference=len(ref_idx))
        dip_flags, testable_any = [], False
        for d in dips:
            centres = LINES[d][1]
            for c in centres:
                band = LINES[d][0] or CI_BANDS[c]
                grid, flux, good = data[band]
                if len(ref_idx) < 5:
                    continue
                ref = np.nanmedian(flux[ref_idx], axis=0)
                depth, testable = line_present(grid, ref, c * (1 + det['v_star'] / C_KMS), det['fwhm'], det['v0'])
                at = [c * (1 + (det['v_star'] + det['v0']) / C_KMS)]
                # telluric pixels of the detection spectrum are left out; in the median reference they move and wash out
                sigma = sign * dip_sigma(grid, flux, good[i], i, ref_idx, spread_idx, at, det['fwhm']) if testable else np.nan
                key = f'{d}_{c:.0f}'
                out[key + '_depth'], out[key + '_testable'], out[key + '_sigma'] = depth, testable, sigma
                testable_any |= testable
                if testable and np.isfinite(sigma):
                    dip_flags.append(sigma >= DIP_SIGMA)
        activity_flags = []
        for x in indices:
            grid, flux, good = data[INDICES[x][0]]
            values = index_values(grid, flux, good, det['v_star'], x)
            others = values[ref_idx]
            spread = MAD_TO_SIGMA * np.nanmedian(np.abs(others - np.nanmedian(others)))
            sigma = (values[i] - np.nanmedian(others)) / spread if spread > 0 else np.nan
            out[x + '_sigma'] = sigma
            if np.isfinite(sigma):
                activity_flags.append(abs(sigma) >= ACTIVITY_SIGMA)
        out['dip_flag'] = any(dip_flags)
        out['activity_flag'] = any(activity_flags)
        out['dip_testable'] = testable_any
        labels = [name for name, on in (('stellar dip', out['dip_flag']), ('activity', out['activity_flag'])) if on]
        if not labels:
            labels = ['clean'] if testable_any or activity_flags else ['not testable']
        if young:
            labels.append('young star')
        out['veto'] = ', '.join(labels)
        rows.append(out)
    return rows


def telluric_mask(db, scores):
    '''Telluric template of each band (wavelength, transmission), built once from bet Pic and kept in VETO_DIR'''
    if os.path.exists(TEMPLATE):
        saved = np.load(TEMPLATE)
        return {band: (saved[f'{band}_wave'], 1 - saved[f'{band}_depth']) for band in TELLURIC_BANDS}
    print('Telluric template from bet Pic')
    bp = star_files(db, scores, BET_PIC)
    bp = bp.sample(min(2000, len(bp)), random_state=0)  # enough to cover the year of barycentric shifts
    mask = {band: telluric_template([VETO_DIR + f for f in bp['file']], bp['berv'].to_numpy(), band)
            for band in TELLURIC_BANDS}
    np.savez(TEMPLATE, **{f'{b}_wave': m[0] for b, m in mask.items()}, **{f'{b}_depth': 1 - m[1] for b, m in mask.items()})
    for band, (_, t) in mask.items():
        print(f'  {band}: {np.mean(t < 1 - TELLURIC_DEPTH):.1%} of pixels masked')
    return mask


def star_files(db, scores, folder):
    f = pd.read_sql("SELECT dp_id, file, berv FROM spectra WHERE folder=? AND status='done'", db, params=(folder,))
    return f.merge(scores[scores['folder'] == folder][['dp_id', 't']], on='dp_id')


def control_detections(scores):
    '''Best night of each known host, up to CONTROL_NIGHTS nights: detections the veto should mostly call clean'''
    controls = []
    for host in (BET_PIC, HD172555):
        s = scores[(scores['folder'] == host) & (scores['score'] > 0) & scores['fwhm'].notna()]
        s = s.assign(night=np.floor(s['t'] - 0.5))
        controls.append(s.sort_values('score', ascending=False).drop_duplicates('night').head(CONTROL_NIGHTS))
    return pd.concat(controls)


def veto_stage(kept, scores, config, path, flipped):
    '''Veto of every detection above threshold of one run, called by stage 2 (calibrate_by_type.py).
    Writes veto.csv (and veto_controls.csv in the real run). Returns the veto table, empty without veto data'''
    if not os.path.exists(VETO_DIR + 'veto_state.sqlite'):
        print('No veto data in', VETO_DIR, '(scripts/veto_fetch.py): veto skipped')
        return pd.DataFrame()
    d = config['search']['dataset']
    idx = pd.read_csv(d + 'dataset_index.csv', dtype={'gaia_dr3': str})
    groups = pd.read_pickle(os.path.join(d, '..', 'grouping', 'v2', 'harps_groups_v2.pkl'))
    idx = idx.merge(groups[['star_id', 'sp_type']], on='star_id', how='left').set_index('folder')
    db = sqlite3.connect(VETO_DIR + 'veto_state.sqlite')
    mask = telluric_mask(db, scores)
    runs = {'flipped' if flipped else 'real': kept}
    if not flipped:
        runs['control'] = control_detections(scores)

    labels = idx.loc[list({f for r in runs.values() for f in r['folder']}), 'star_label']
    young = young_stars(list(labels)) if len(labels) else set()
    out = []
    for run, detections in runs.items():
        for folder, det in detections.groupby('folder'):
            sp = idx.loc[folder, 'sp_type']
            label = idx.loc[folder, 'star_label']
            is_young = label in young or bool(re.search(r'e\b|e$', str(sp)))
            # in the flipped run a detection is a brightening of the real Ca II, so the matching stellar signal is a
            # brightening of the veto line: the veto lines are flipped too
            rows = check_star(folder, det, sp, star_files(db, scores, folder), mask, is_young, -1 if flipped else 1)
            for r in rows:
                r.update(run=run, star=label)
            out += rows
            print(f'{run:8s} {label:14s} {str(sp):10s} ' + ' | '.join(sorted({r["veto"] for r in rows})))
    table = pd.DataFrame(out)
    for run in runs:
        part = table[table['run'] == run] if len(table) else table
        part.to_csv(os.path.join(path, 'veto_controls.csv' if run == 'control' else 'veto.csv'), index=False)
    return table


def add_veto(kept, best, veto):
    '''Veto label of each detection, and for each candidate the share of its star's detections that were flagged.
    Flags only: no detection is removed and no candidate moves tier'''
    if len(veto) == 0:
        return kept, best
    veto = veto[veto['run'] != 'control']
    labels = veto[['dp_id', 'veto']].drop_duplicates('dp_id')
    kept = kept.drop(columns=['veto'], errors='ignore').merge(labels, on='dp_id', how='left')
    if len(best):
        share = (veto.assign(flagged=veto['veto'].str.contains('stellar dip|activity'))
                 .groupby('folder')['flagged'].mean().rename('veto_flagged_share'))
        best = (best.drop(columns=['veto', 'veto_flagged_share'], errors='ignore').merge(labels, on='dp_id', how='left')
                .merge(share, on='folder', how='left'))
    return kept, best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Veto lines for the detections above threshold, without rerunning '
                                                 'stage 2 (calibrate_by_type.py runs it too)')
    parser.add_argument('--c', default='spectral_classes.json')
    parser.add_argument('--r', default='../results/Reliable/')
    parser.add_argument('--flip', action='store_true', help='the run in --r is the flipped one')
    args = parser.parse_args()
    with open(args.c) as handle:
        config = json.load(handle)
    scores = pd.read_pickle(os.path.join(args.r, 'spectrum_scores.pkl'))
    kept = pd.read_pickle(os.path.join(args.r, 'scored_detections.pkl'))
    best = pd.read_csv(os.path.join(args.r, 'candidates.csv'))
    table = veto_stage(kept, scores, config, args.r, args.flip)
    kept, best = add_veto(kept, best, table)
    kept.to_pickle(os.path.join(args.r, 'scored_detections.pkl'))
    best.to_csv(os.path.join(args.r, 'candidates.csv'), index=False)
    summary = table.assign(stars=table['folder']).groupby(['run', 'veto'])['stars'].nunique().unstack(fill_value=0)
    print(summary.to_string())
