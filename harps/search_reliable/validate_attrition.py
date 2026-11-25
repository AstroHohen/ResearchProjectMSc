'''
14/11/26
Which stage-2 test rejects the fake comets, and which rejects the flipped control. Same fake comets as
validate_rules.py (same stars, plans and seeds), judged by the baseline and combined rules. For each detected fake
comet, the detection closest to passing (fewest failed tests) says which tests failed, and whether one test alone
stopped it. The flipped run gives the same for the false alarms, so a test is only worth changing if it rejects
many fake comets and few flipped detections.
    py -3 validate_attrition.py                 # fake comets, writes ../results/Overnight/attrition_*.csv
    py -3 validate_attrition.py --flipped-only  # only the flipped control (fast)
'''
import argparse
import json
import multiprocessing
import os

import numpy as np
import pandas as pd

import exocomet_search
from calibrate_by_type import TESTS
from decision_rules import decide_rule
from validate_by_type import DEPTHS, FWHMS, MAX_SPECTRA, RICH_FWHM, STARS_PER_CLASS, inject_star

ATTRITION_RULES = ['type_weight', 'pairs']
SHAPE_MODES = ['weighted', 'no_mirror']  # the default, and the default without the mirror level
Extra = {}


def closest(kept):
    '''The detection with the fewest failed tests, with its failures and tier inputs'''
    fails = ~kept[TESTS]
    kept = kept.assign(n_fail=fails.sum(axis=1))
    row = kept.sort_values(['n_fail', 'confirmed'], ascending=[True, False]).iloc[0]
    out = {name: bool(not row[name]) for name in TESTS}
    out.update(n_fail=int(row['n_fail']), confirmed=bool(row['confirmed']),
               control_informative=bool(row['control_informative']), n_control=int(row['n_control']),
               near_h_epsilon=bool(row['near_h_epsilon']))
    return out


def with_mode(config, mode):
    if mode == 'no_mirror':
        return dict(config, tests=dict(config['tests'], shape_mode='weighted', use_mirror=False))
    return dict(config, tests=dict(config['tests'], shape_mode=mode))


def init(config, n_stars, class_counts):
    exocomet_search.init_worker(config)
    Extra.update(n_stars=n_stars, class_counts=class_counts)


def run(job):
    rows = inject_star(job)
    if len(rows) == 0 or 'fwhm' not in pd.DataFrame(rows):
        return []
    rows = pd.DataFrame(rows)
    out = []
    for rule, mode in [(r, m) for r in ATTRITION_RULES for m in SHAPE_MODES]:
        kept, best = decide_rule(rows, with_mode(exocomet_search.Config, mode), rule, Extra['n_stars'],
                                 Extra['class_counts'])
        base = dict(rule=rule, shape=mode, class_name=job[1], depth=job[2], fwhm=job[3], pca=job[4], folder=job[0])
        near = rows[rows['injected'] & rows['near_injection']]
        base['best_score'] = float(near['score'].max()) if len(near) else 0.0
        base['threshold'] = float(near['threshold'].iloc[0]) if len(near) and 'threshold' in near else np.nan
        hits = kept[kept['injected'] & kept['near_injection']] if len(kept) else kept
        if len(hits) == 0:
            out.append(dict(base, detected=False))
            continue
        top = bool(len(best) and (best['tier'] == 'high confidence').any() and hits['pass_all'].any())
        out.append(dict(base, detected=True, high_confidence=top, **closest(hits)))
    return out


def flipped(out_dir, config, class_counts):
    rows = []
    for folder in ('on_flip', 'off_flip'):
        scores = pd.read_pickle(os.path.join(out_dir, folder, 'spectrum_scores.pkl'))
        for rule, mode in [(r, m) for r in ATTRITION_RULES for m in SHAPE_MODES]:
            kept, _ = decide_rule(scores, with_mode(config, mode), rule, class_counts=class_counts)
            for star, group in kept.groupby('folder'):
                rows.append(dict(rule=rule, shape=mode, search=folder, folder=star, class_name=group['class_name'].iloc[0],
                                 detected=True, **closest(group)))
    rows = pd.DataFrame(rows)
    hosts = open(os.path.join(out_dir, 'hosts.txt')).read().split()
    return rows[~rows['folder'].isin(hosts)]


def summary(rows, keys):
    '''Per group: detected, passed, and for the detected that failed, the share failing each test and failing only it'''
    out = []
    for key, group in rows.groupby(keys):
        det = group[group['detected'] == True]
        failed = det[det['n_fail'] > 0]
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)), n=len(group), detected=len(det),
                   passed=int((det['n_fail'] == 0).sum()), failed=len(failed))
        for name in TESTS:
            short = name.replace('pass_', '')
            row['fail_' + short] = int(failed[name].sum())
            row['only_' + short] = int((failed[name] & (failed['n_fail'] == 1)).sum())
        ok = det[det['n_fail'] == 0].astype({'confirmed': bool, 'control_informative': bool})
        row['passed_no_repeat'] = int((~ok['confirmed']).sum())
        row['passed_control_weak'] = int((ok['confirmed'] & ~ok['control_informative'] & (ok['n_control'] > 0)).sum())
        out.append(row)
    return pd.DataFrame(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Which test rejects the fake comets and the flipped control')
    parser.add_argument('--c', default='../results/Overnight/config.json')
    parser.add_argument('--r', default='../results/Reliable/', help='where the star list comes from')
    parser.add_argument('--out', default='../results/Overnight/')
    parser.add_argument('--flipped-only', action='store_true')
    args = parser.parse_args()

    with open(args.c) as handle:
        config = json.load(handle)
    with open(os.path.join(args.out, 'class_counts.json')) as handle:
        class_counts = json.load(handle)

    flip = flipped(args.out, config, class_counts)
    flip.to_csv(os.path.join(args.out, 'attrition_flipped_rows.csv'), index=False)
    table = summary(flip, ['rule', 'shape', 'class_name'])
    table.to_csv(os.path.join(args.out, 'attrition_flipped.csv'), index=False)
    print(table.T.to_string(), flush=True)
    if args.flipped_only:
        raise SystemExit

    config['search']['n_stars'] = n_stars = sum(class_counts.values())
    scores = pd.read_pickle(os.path.join(args.r, 'spectrum_scores.pkl'))
    stars = scores[scores['n'] < MAX_SPECTRA].drop_duplicates('folder')
    stars = stars.sample(frac=1, random_state=0).groupby('class_name').head(STARS_PER_CLASS)
    plans = [(d, w) for d in DEPTHS for w in FWHMS] + [('rich', RICH_FWHM)]
    jobs = [(f, c, d, w, pca) for f, c in zip(stars['folder'], stars['class_name']) for d, w in plans
            for pca in (True, False)]
    print('Injection runs:', len(jobs), flush=True)

    out = []
    with multiprocessing.Pool(config['search']['cores'], initializer=init,
                              initargs=(config, n_stars, class_counts)) as pool:
        for i, rows in enumerate(pool.imap_unordered(run, jobs, chunksize=4)):
            out += rows
            if (i + 1) % 500 == 0:
                print(f'{i + 1}/{len(jobs)}', flush=True)
                pd.DataFrame(out).to_pickle(os.path.join(args.out, 'attrition_injection.pkl'))
    out = pd.DataFrame(out)
    out.to_pickle(os.path.join(args.out, 'attrition_injection.pkl'))
    out['depth'] = out['depth'].astype(str)
    table = summary(out, ['rule', 'shape', 'class_name', 'depth'])
    table.to_csv(os.path.join(args.out, 'attrition_injection.csv'), index=False)
    print(table.to_string(), flush=True)
