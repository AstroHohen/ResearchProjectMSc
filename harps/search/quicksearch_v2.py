'''
01/11/26
QuickSearch_V2 for HARPS, copied from HR/search/quicksearch_V2.ipynb.
Returns numpy array of candidate stars, search done according to parameters in param.json.

Changes from UVES, none of them to the search itself:
- a script, with the worker at module level: Windows multiprocessing cannot use the notebook version
- candidates are named by their dataset folder, since each folder is already one V2 star
  (so the UVES deduplication by name is not needed)
- no os.nice, which does not exist on Windows
'''
import argparse
import glob
import json
import multiprocessing
import os
import time

import numpy as np

from asset import ASSET_HARPS

Search = None


def init_worker(param, line):
    global Search
    Search = ASSET_HARPS(parameters=param, line=line)


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
        spec = new_spectra[i]

        snr = Search.snr(spec, med, Search.spectra_err[i], med_err)
        sd = np.std(snr)
        if not np.isfinite(sd) or sd == 0:
            continue

        corr_snr = snr.copy()
        if Search.ccf:
            corr_snr[cond100] = np.nan
        corr_snr = corr_snr[Search.snr_idxrange]

        sig = corr_snr / sd

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

    parser = argparse.ArgumentParser(description='How to use QuickSearch_V2', epilog='ASSET by RBW')
    parser.add_argument('--p', metavar='param.json', default='param.json', help='parameter file name (default:param.json)')
    parser.add_argument('--line', metavar='line', default='K', help='define what atomic line to use (default: "K")')
    parser.add_argument('--r', metavar='res-path', default='../results/QuickSearch_V2/', help='output path (default: ../results/QuickSearch_V2/)')
    parser.add_argument('--tag', metavar='tag', default='V2', help='suffix tag for output filename (default: V2)')
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
