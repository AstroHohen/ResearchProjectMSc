'''
07/11/26
Na I D at the velocity of each candidate (idea 12). The cut-outs only hold Ca II, so this asks ESO SODA for a second
small cut-out, 588.3-590.3 nm (Na I D2 588.995 nm and D1 589.592 nm), only for the candidate stars: every spectrum of
each detection night, plus up to OTHER_SPECTRA from other nights. Files go to C:/data/harps/extra/NaD/.

For each detection spectrum, each Na line is scored like a control line: a compensated Gaussian filter at the comet
velocity (barycentric: velocity of the star + offset of the feature) and width, against a reference. The reference is
the median of the other spectra of the same night when there are 2 or more (same telluric lines, same barycentric
shift), otherwise of the other nights. The score is turned into a significance with the same filter run on the
reference spectra themselves. Absorption in Na at the same velocity is strong support for gas in front of the star;
no Na says little, as many comets of bet Pic show Ca II without Na.

  py -3 sodium_check.py fetch   # download what is missing
  py -3 sodium_check.py measure # scores
'''
import argparse
import gzip
import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from astropy.io import fits

SODA = 'https://dataportal.eso.org/dataPortal/soda/sync'
OUT = 'C:/data/harps/extra/NaD/'
BAND_NM = (588.3, 590.3)
LINES_NM = {'D2': 588.995, 'D1': 589.592}
OTHER_SPECTRA = 60
C_KMS = 299792.458
FWHM_TO_SIGMA = 2.354820045


def path_of(dp_id):
    return os.path.join(OUT, dp_id.replace(':', '_') + '.npz')


def fetch(dp_id):
    if os.path.exists(path_of(dp_id)):
        return 'have'
    r = requests.get(SODA, timeout=180, params=[('ID', f'ivo://eso.org/ID?{dp_id}'),
                                                ('BAND', f'{BAND_NM[0] * 1e-9:.6E} {BAND_NM[1] * 1e-9:.6E}')])
    if r.status_code in (401, 403):
        return 'restricted'
    r.raise_for_status()
    blob = r.content
    if blob[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            blob = z.read(sorted(z.namelist())[0])
    with fits.open(io.BytesIO(blob), memmap=False) as hdul:
        row = hdul['SPECTRUM'].data[0]
        wave = np.asarray(row['WAVE'], dtype=float)
        wave = wave / 10 if np.nanmedian(wave) > 1000 else wave  # Angstrom to nm
        np.savez_compressed(path_of(dp_id), wave=wave, flux=np.asarray(row['FLUX'], np.float32),
                            header_gz=np.frombuffer(gzip.compress(hdul[0].header.tostring().encode()), np.uint8))
    return 'done'


def plan(results, dataset):
    '''Spectra to look at: every spectrum of each detection night, and up to OTHER_SPECTRA others, per star'''
    best = pd.read_csv(os.path.join(results, 'candidates.csv'))
    kept = pd.read_pickle(os.path.join(results, 'scored_detections.pkl'))
    kept = kept[kept['pass_all'] & kept['folder'].isin(best['folder'])]
    rows = []
    for folder, detections in kept.groupby('folder'):
        spectra = pd.read_pickle(os.path.join(dataset, folder, 'meta', 'group_df.pkl'))[['dp_id', 't_min']]
        night = np.floor(spectra['t_min'] - 0.5)
        on = night.isin(np.floor(detections['t'] - 0.5))
        others = spectra[~on].sample(min(OTHER_SPECTRA, int((~on).sum())), random_state=0)
        chosen = pd.concat([spectra[on], others])
        chosen = chosen.assign(folder=folder, night=np.floor(chosen['t_min'] - 0.5),
                               detection=chosen['dp_id'].isin(detections['dp_id']))
        rows.append(chosen)
    return pd.concat(rows), kept


def line_score(wave, flux, centre_nm, fwhm_kms):
    '''Compensated Gaussian filter at one wavelength, on a spectrum already divided by its reference (1 = no change)'''
    sigma = centre_nm * fwhm_kms / FWHM_TO_SIGMA / C_KMS
    near = np.abs(wave - centre_nm) < 8 * sigma
    if near.sum() < 5:
        return np.nan
    g = np.exp(-(wave[near] - centre_nm) ** 2 / (2 * sigma ** 2))
    g = g - g.mean()
    return float(((flux[near] - 1) @ g) / np.sqrt(np.sum(g ** 2)))


def load(dp_id, grid):
    data = np.load(path_of(dp_id))
    flux = np.interp(grid, data['wave'], data['flux'])
    return flux / np.median(flux)


def measure(results, dataset):
    chosen, kept = plan(results, dataset)
    chosen = chosen[[os.path.exists(path_of(d)) for d in chosen['dp_id']]]
    out = []
    for folder, group in chosen.groupby('folder'):
        first = np.load(path_of(group['dp_id'].iloc[0]))['wave']
        grid = np.linspace(first.min() + 0.05, first.max() - 0.05, len(first))
        flux = {d: load(d, grid) for d in group['dp_id']}
        for _, row in kept[kept['folder'] == folder].iterrows():
            if row['dp_id'] not in flux:
                continue
            night = np.floor(row['t'] - 0.5)
            same = group[(group['night'] == night) & ~group['detection']]['dp_id'].tolist()
            pool = same if len(same) >= 2 else group[group['night'] != night]['dp_id'].tolist()
            if len(pool) < 2:
                continue
            ref = np.median([flux[d] for d in pool], axis=0)
            result = dict(folder=folder, dp_id=row['dp_id'], v0=row['v0'], fwhm=row['fwhm'], reference=len(pool),
                          reference_same_night=len(same) >= 2)
            for name, rest in LINES_NM.items():
                centre = rest * (1 + (row['v_star'] + row['v0']) / C_KMS)
                score = line_score(grid, flux[row['dp_id']] / ref, centre, row['fwhm'])
                # the same filter on each reference spectrum against the median of the others: the scatter of the score
                scatter = [line_score(grid, flux[d] / np.median([flux[o] for o in pool if o != d], axis=0), centre,
                                      row['fwhm']) for d in pool[:30]] if len(pool) >= 3 else []
                spread = 1.4826 * np.nanmedian(np.abs(np.array(scatter) - np.nanmedian(scatter))) if scatter else np.nan
                result[name + '_sigma'] = -score / spread if spread and np.isfinite(spread) else np.nan
            out.append(result)
    table = pd.DataFrame(out)
    table.to_csv(os.path.join(results, 'sodium.csv'), index=False)
    return table


def _safe(dp_id):
    try:
        return fetch(dp_id)
    except Exception as error:  # one failed download should not stop the others
        return 'failed: ' + type(error).__name__


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Na I D at the velocity of each candidate')
    parser.add_argument('step', choices=['fetch', 'measure'])
    parser.add_argument('--c', default='spectral_classes.json', help='config file (default: spectral_classes.json)')
    parser.add_argument('--r', default='../results/Reliable/', help='results (default: ../results/Reliable/)')
    args = parser.parse_args()
    with open(args.c) as configfile:
        dataset = json.load(configfile)['search']['dataset']
    os.makedirs(OUT, exist_ok=True)
    if args.step == 'fetch':
        chosen, _ = plan(args.r, dataset)
        print('Spectra to fetch:', len(chosen), 'for', chosen['folder'].nunique(), 'stars')
        with ThreadPoolExecutor(8) as pool:
            status = list(pool.map(_safe, chosen['dp_id']))
        print(pd.Series(status).value_counts().to_string())
    else:
        table = measure(args.r, dataset)
        print(table.to_string(index=False))
