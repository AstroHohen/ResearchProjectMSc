'''
01/11/26
QuickSearch_V3 for HARPS: QuickSearch_V2 with the local noise of section 4.2.2 of the article.

Change from V2, only in the noise:
- V2 divides each spectrum by (s - m) / sqrt(err_s^2 + err_m^2), then by one std for the whole spectrum
- V3 uses Eq. 5, r_loc(v) = r(v) / sigma_loc(v), with sigma_loc from the MAD of r in a sliding window around v
  The MAD is measured on s - m, so it already holds the noise of the reference: nothing is added in quadrature.
- the window skips local_guard pixels either side of v, so a dip does not count as noise at its own position
- r_loc is then divided by its std over the spectrum, because the MAD is too small for residuals against a median
  of few spectra (1/3 of the residuals are exactly 0 when n = 3)

Normalisation, smoothing, noise cut, activity mask, threshold and width rule are the same as V2.
spec_err is still used, but only by the noise cut.
'''
import argparse
import glob
import json
import multiprocessing
import os
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from asset import ASSET_HARPS

MAD_TO_SIGMA = 1.4826  # MAD -> standard deviation for Gaussian noise
MIN_SIGMA_FRACTION = 1e-3  # of the spectrum's median sigma: below this the window is mostly exact zeros

Search = None


class ASSET_LocalNoise(ASSET_HARPS):

    def __init__(self, line, parameters=None):
        super().__init__(line, parameters)
        if parameters != None:
            self.local_window = parameters["local_window"]
            self.local_guard = parameters["local_guard"]
            if self.local_window % 2 == 0:
                raise ValueError('local_window must be odd, so the window is centred on each pixel')

    def local_sigma(self, resid):
        '''Local noise of one residual 1Darray: scaled MAD in a window centred on each pixel
        Returns same shape array'''
        half = self.local_window // 2
        windows = sliding_window_view(np.pad(resid, half, mode='reflect'), self.local_window)
        windows = windows[:, np.abs(np.arange(self.local_window) - half) > self.local_guard]
        centre = np.median(windows, axis=1, keepdims=True)
        return MAD_TO_SIGMA * np.median(np.abs(windows - centre), axis=1)

    def local_snr(self, s, m):
        '''Eq. 5 of the article, rescaled to unit std. Input spec 1Darray, median ref
        Returns r_loc 1Darray, NaN where the MAD is too small to be a noise estimate'''
        resid = (s - m).astype(float)
        sigma = self.local_sigma(resid)
        sigma[sigma <= MIN_SIGMA_FRACTION * np.median(sigma)] = np.nan
        r_loc = resid / sigma
        return r_loc / np.nanstd(r_loc)


def init_worker(param, line):
    global Search
    Search = ASSET_LocalNoise(parameters=param, line=line)


def quicksearch_with_rv_filter(star):
    Search.ccf = False
    spec_param = Search.spec_analysis(star)
    if spec_param is None:
        return None

    new_spectra, med, med_err = spec_param

    if Search.ccf:
        rv_shift = Search.X_corr(med.copy())
        cond100 = (Search.radial_velocity > rv_shift - 50) & (Search.radial_velocity < rv_shift + 50)

    dip_detected = False
    max_peak_seen = -np.inf

    for i in range(len(new_spectra)):
        sig = Search.local_snr(new_spectra[i], med)

        if Search.ccf:
            sig[cond100] = np.nan
        sig = sig[Search.snr_idxrange]

        if not np.isfinite(sig).any():
            continue

        min_detect = np.nanmin(sig)
        max_detect = np.nanmax(sig)
        if np.isfinite(max_detect):
            max_peak_seen = max(max_peak_seen, float(max_detect))

        if min_detect < Search.threshold:
            width = Search.get_width(sig)
            if width >= Search.width_filter:
                dip_detected = True

    if dip_detected:
        if not np.isfinite(max_peak_seen):
            max_peak_seen = np.nan
        return (os.path.basename(os.path.normpath(star)), float(max_peak_seen))

    return None


if __name__ == '__main__':
    startTime = time.time()

    parser = argparse.ArgumentParser(description='How to use QuickSearch_V3', epilog='ASSET by RBW')
    parser.add_argument('--p', metavar='param_v3.json', default='param_v3.json', help='parameter file name (default:param_v3.json)')
    parser.add_argument('--line', metavar='line', default='K', help='define what atomic line to use (default: "K")')
    parser.add_argument('--r', metavar='res-path', default='../results/QuickSearch_V3/', help='output path (default: ../results/QuickSearch_V3/)')
    parser.add_argument('--tag', metavar='tag', default='V3', help='suffix tag for output filename (default: V3)')
    parser.add_argument('--limit', type=int, default=None, help='only search the first N stars (for testing)')
    args = parser.parse_args()

    with open(args.p) as paramfile:
        param = json.load(paramfile)

    all_paths = sorted(glob.glob(param["dataset"] + '*/'))

    min_spectra = int(param.get('min_spectra', 1))
    # Require a minimum number of spectra so each target has enough epochs to form a robust group reference.
    dataset_path = [p for p in all_paths if np.load(p + 'spec/sK.npy', mmap_mode='r').shape[0] >= min_spectra]
    print('Stars with >= {} spectra: {}/{}'.format(min_spectra, len(dataset_path), len(all_paths)), flush=True)
    if args.limit:
        dataset_path = dataset_path[:args.limit]

    with multiprocessing.Pool(param["cores"], initializer=init_worker, initargs=(param, args.line)) as pool:
        out = []
        for i, item in enumerate(pool.imap_unordered(quicksearch_with_rv_filter, dataset_path)):
            out.append(item)
            if (i + 1) % 250 == 0:
                print('{}/{} stars, {:.0f} s'.format(i + 1, len(dataset_path), time.time() - startTime), flush=True)

    Threshold = param["threshold"]
    Cutoff = param["cutoff"]
    Width = param["width_filt"]

    records = [item for item in out if item is not None]
    print('------------------')

    cands = np.array(sorted(item[0] for item in records), dtype=object)
    peak_lookup = {item[0]: item[1] for item in records}
    print('Raw candidates:', len(cands))

    cands_peak = np.array([peak_lookup.get(name, np.nan) for name in cands.tolist()], dtype=float)
    peak_threshold = abs(float(Threshold))

    # Split dip detections with significant positive peaks into requested bins
    bin_3p5_4_mask = (cands_peak >= peak_threshold) & (cands_peak < 4.0)
    bin_4_5_mask = (cands_peak >= 4.0) & (cands_peak < 5.0)
    bin_5_plus_mask = cands_peak >= 5.0

    peak_3p5_4 = cands[bin_3p5_4_mask]
    peak_4_5 = cands[bin_4_5_mask]
    peak_5_plus = cands[bin_5_plus_mask]

    print('Peak bins among candidates:')
    print('  +{:.1f} to +4 sigma: {}'.format(peak_threshold, len(peak_3p5_4)))
    print('  +4 to +5 sigma: {}'.format(len(peak_4_5)))
    print('  +5 sigma and above: {}'.format(len(peak_5_plus)))

    tag = args.tag.strip()
    if tag:
        cands_file = 'candidates_{}sig_{}cut_{}width_{}.npy'.format(Threshold, Cutoff, Width, tag)
        peaks_file = 'candidate_max_peak_{}sig_{}cut_{}width_{}.npy'.format(Threshold, Cutoff, Width, tag)
    else:
        cands_file = 'candidates_{}sig_{}cut_{}width.npy'.format(Threshold, Cutoff, Width)
        peaks_file = 'candidate_max_peak_{}sig_{}cut_{}width.npy'.format(Threshold, Cutoff, Width)

    os.makedirs(args.r, exist_ok=True)

    np.save(os.path.join(args.r, cands_file), cands)
    np.save(os.path.join(args.r, peaks_file), cands_peak)

    peak_bin_dirs = {
        'peak_3p5_to_4': peak_3p5_4,
        'peak_4_to_5': peak_4_5,
        'peak_5_plus': peak_5_plus,
    }

    for folder, arr in peak_bin_dirs.items():
        out_dir = os.path.join(args.r, folder)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'targets.npy'), arr)

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
