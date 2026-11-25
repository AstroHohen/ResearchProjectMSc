'''
21/11/26
Bendahan-West et al. 2025 search (ASSET + quicksearch, github.com/AstroHohen/HARPS-exocomet, a fork of the original)
run on our HARPS dataset, on the real spectra and on the flipped control (addendum 13 of the pre-registration).

His detection logic is used unchanged: his ASSET class is imported from the clone and his param.json is read as is.
Our per-star folders already have his layout (meta/group_df.pkl with Reduced, Sanitised, Object, SNR; spec/sK.npy;
wavelength/wK.npy, rvK.npy), so nothing is converted. Two things are added around his code, neither changing what is
detected:
- quicksearch returns the star at its first detection; bw_detections runs the same loop line for line but keeps every
  spectrum that passes, so the star-level answer is identical and the detections can be counted.
- flipped control: each spectrum is reflected about the star's median after removing its scale (mean beyond
  +-200 km/s, his normalisation band), then given its scale back, so his own normalisation sees 2 x median - spectrum.

Known difference of data, not of code: his cut-outs had about 1 km/s per pixel ("1 data point = 1km/s"), ours have
0.76 km/s, so his width filter of 3 pixels is 2.3 km/s here instead of about 3 km/s.

    py -3 bw_run.py                      # real spectra
    py -3 bw_run.py --flip               # flipped control
'''
import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd

BW_REPO = 'C:/Users/guilh/OneDrive/Desktop/HARPS-exocomet/search/'
sys.path.append(BW_REPO)  # last, so our own 'asset' module (harps/search) is not shadowed by his
# asset.py imports get_otypes (a SIMBAD query needing astroquery) but the search never calls it: an empty stand-in
import importlib.util  # noqa: E402
import types  # noqa: E402
sys.modules.setdefault('otypes', types.SimpleNamespace(get_otypes=None))
_spec = importlib.util.spec_from_file_location('bw_asset', BW_REPO + 'asset.py')
bw_asset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bw_asset)
ASSET = bw_asset.ASSET  # his class, unchanged

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH = os.path.join(HERE, '..', 'search_reliable')
DATASET = 'C:/data/harps/dataset/'


class FlippedASSET(ASSET):
    '''His ASSET; only the loaded spectra are flipped, before any of his steps'''

    def load_info(self, star_path, init=True):
        super().load_info(star_path, init)
        s = self.spectra.astype(float)
        band = np.abs(self.radial_velocity) > 200
        scale = s[:, band].mean(axis=1, keepdims=True)
        normed = s / scale
        self.spectra = (2 * np.median(normed, axis=0) - normed) * scale


def bw_detections(search, star):
    '''His quicksearch loop (asset.py, ASSET.quicksearch), line for line, keeping every passing spectrum'''
    search.ccf = False
    spec_param = search.spec_analysis(star)
    if spec_param is None:
        return []
    new_spectra, med, med_err = spec_param
    corr_med = med.copy()
    if search.ccf:
        rv_shift = search.X_corr(corr_med)
        cond100 = (search.radial_velocity > rv_shift - 50) & (search.radial_velocity < rv_shift + 50)
    rows = []
    rv = search.radial_velocity[search.snr_idxrange]
    for i in range(len(new_spectra)):
        spec = new_spectra[i]
        snr = search.snr(spec, med, search.spectra_err[i], med_err)
        sd = np.std(snr)
        corr_snr = snr.copy()
        if search.ccf:
            corr_snr[cond100] = np.nan
        corr_snr = corr_snr[search.snr_idxrange]
        sig = corr_snr / sd
        min_detect = np.nanmin(sig)
        if min_detect < search.threshold:
            width = search.get_width(sig)
            if width >= search.width_filter:
                rows.append(dict(dp_id=search.df['dp_id'].iloc[i], t=search.df['t_min'].iloc[i],
                                 min_sigma=float(min_detect), rv_bary=float(rv[np.nanargmin(sig)]), width_px=int(width),
                                 ccf_mask=bool(search.ccf)))
    return rows


Worker = {}


def init(param, flip):
    Worker['search'] = (FlippedASSET if flip else ASSET)(parameters=param, line='K')


def run(job):
    folder, class_name = job
    try:
        rows = bw_detections(Worker['search'], DATASET + folder + '/')
    except Exception as error:  # a star his code cannot read is reported, not hidden
        return [dict(folder=folder, class_name=class_name, error=f'{type(error).__name__}: {error}')]
    if not rows:
        return [dict(folder=folder, class_name=class_name, detected=False)]
    return [dict(folder=folder, class_name=class_name, detected=True, **r) for r in rows]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bendahan-West search on our HARPS dataset')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--cores', type=int, default=6)
    parser.add_argument('--out', default=os.path.join(HERE, 'results'))
    args = parser.parse_args()
    with open(BW_REPO + 'param.json') as handle:
        param = json.load(handle)  # his settings: threshold -4, width 3 px, cutoff 2, rv +-200 km/s
    print('Bendahan-West parameters:', param, flush=True)

    sys.path.insert(0, SEARCH)
    import exocomet_search as es
    config = json.load(open(os.path.join(SEARCH, 'spectral_classes.json')))
    jobs = es.build_jobs(config)[['folder', 'class_name']]
    jobs = list(jobs.itertuples(index=False, name=None))
    print('Stars:', len(jobs), 'flipped' if args.flip else 'real', flush=True)

    out = []
    with multiprocessing.Pool(args.cores, initializer=init, initargs=(param, args.flip)) as pool:
        for i, rows in enumerate(pool.imap_unordered(run, jobs, chunksize=4)):
            out += rows
            if (i + 1) % 250 == 0:
                print(f'{i + 1}/{len(jobs)}', flush=True)
    out = pd.DataFrame(out)
    os.makedirs(args.out, exist_ok=True)
    name = 'bw_flipped.csv' if args.flip else 'bw_real.csv'
    out.to_csv(os.path.join(args.out, name), index=False)
    stars = out.groupby('folder').agg(class_name=('class_name', 'first'), detected=('detected', 'max'))
    print(stars.groupby('class_name')['detected'].agg(['sum', 'count']).to_string())
    print('stars detected:', int(stars['detected'].fillna(False).sum()), 'of', len(stars),
          '| errors:', int(out['error'].notna().sum()) if 'error' in out else 0)
