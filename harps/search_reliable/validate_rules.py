'''
13/11/26
Fake comets judged by each decision rule of decision_rules.py: the same injected spectra, scored once by stage 1 (PCA
on and off), then decided by every rule. Same plans as validate_by_type.py --two-runs.
    py -3 validate_rules.py --c ../results/Overnight/config.json --out ../results/Overnight/
'''
import argparse
import json
import multiprocessing
import os

import numpy as np
import pandas as pd

import exocomet_search
from decision_rules import RULES, decide_rule
from validate_by_type import DEPTHS, FWHMS, MAX_SPECTRA, RICH_FWHM, STARS_PER_CLASS, inject_star

Extra = {}


def init(config, n_stars, class_counts):
    exocomet_search.init_worker(config)
    Extra.update(n_stars=n_stars, class_counts=class_counts)


def run(job):
    rows = inject_star(job)
    if len(rows) == 0:
        return []
    rows = pd.DataFrame(rows)
    n_injected = max(int(rows['injected'].sum()), 1)
    out = []
    for rule in RULES:
        if 'fwhm' in rows:
            kept, best = decide_rule(rows, exocomet_search.Config, rule, Extra['n_stars'], Extra['class_counts'])
        else:
            kept, best = rows.iloc[:0].assign(pass_all=False, confirmed=False), rows.iloc[:0]
        hits = kept[kept['injected'] & kept['near_injection']] if len(kept) else kept
        passed = hits[hits['pass_all']] if len(hits) else hits
        top = bool(len(best) and (best['tier'] == 'high confidence').any() and len(passed))
        out.append(dict(rule=rule, class_name=job[1], depth=job[2], fwhm=job[3], pca=job[4], folder=job[0],
                        above_threshold=len(hits) > 0, passed=len(passed) > 0,
                        confirmed=bool(len(passed) and passed['confirmed'].any()), high_confidence=top,
                        share_found=float(len(passed) / n_injected)))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake comets judged by each decision rule')
    parser.add_argument('--c', default='../results/Overnight/config.json')
    parser.add_argument('--r', default='../results/Reliable/', help='where the star list comes from')
    parser.add_argument('--out', default='../results/Overnight/')
    args = parser.parse_args()

    with open(args.c) as handle:
        config = json.load(handle)
    with open(os.path.join(args.out, 'class_counts.json')) as handle:
        class_counts = json.load(handle)
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
                pd.DataFrame(out).to_pickle(os.path.join(args.out, 'injection_rules.pkl'))
    out = pd.DataFrame(out)
    out.to_pickle(os.path.join(args.out, 'injection_rules.pkl'))

    columns = ['above_threshold', 'passed', 'confirmed', 'high_confidence', 'share_found']
    either = out.groupby(['rule', 'class_name', 'depth', 'fwhm', 'folder'])[columns].max().reset_index()
    either['depth'] = either['depth'].astype(str)
    table = (100 * either.groupby(['rule', 'class_name', 'depth'])[columns].mean()).round(1)
    table.to_csv(os.path.join(args.out, 'injection_rules.csv'))
    print(table.to_string())
