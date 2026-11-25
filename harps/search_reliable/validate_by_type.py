'''
07/11/26
Injection test of the exocomet search: how many fake exocomets does each spectral class recover?
Run after both stages. Each run puts one fake comet in up to three consecutive spectra of one night of a star, in K and H
with depth(K) / depth(H) = 1.5, at a random velocity from the star. A comet counts as recovered at the velocity it was put.
A comet-rich star is also tested: comets at one velocity in RICH_SHARE of all its spectra (bet Pic has comets at most
epochs). With --two-runs each fake comet goes through both searches (PCA on and off, combine_runs.py), and it counts as
recovered if either finds it; the share of injected spectra found is reported for the comet-rich star.
'''
import argparse
import json
import multiprocessing
import os

import numpy as np
import pandas as pd

import exocomet_search
from calibrate_by_type import decide

DEPTHS = (0.01, 0.02, 0.05, 0.1, 0.2)
FWHMS = (5, 10, 20)
EPOCHS = 3
STARS_PER_CLASS = 60
MAX_SPECTRA = 2000
RICH_SHARE, RICH_DEPTHS, RICH_FWHM = 0.4, (0.03, 0.15), 10


def inject_star(job):
    folder, class_name, depth, fwhm, pca = job
    exocomet_search.Config['search']['pca'] = pca
    plan = (dict(share=RICH_SHARE, depth_range=RICH_DEPTHS, fwhm=fwhm, depth=depth) if depth == 'rich'
            else dict(depth=depth, fwhm=fwhm, epochs=EPOCHS))
    return exocomet_search.search_star((folder, class_name), plan)


def recovered(rows, config, n_stars):
    '''Did the search find the fake comet above threshold, after the tests, confirmed; and the share of its spectra found'''
    rows_df = pd.DataFrame(rows)
    kept = decide(rows_df, config, n_stars)[0] if 'fwhm' in rows_df else rows_df.iloc[:0].assign(pass_all=False, confirmed=False)
    hits = kept[kept['injected'] & kept['near_injection']]
    n_injected = max(int(rows_df['injected'].sum()), 1)
    return dict(above_threshold=len(hits) > 0, passed=bool(hits['pass_all'].any()),
                confirmed=bool((hits['pass_all'] & hits['confirmed']).any()),
                share_found=float(hits['pass_all'].sum() / n_injected),
                v2_hit=bool(rows_df['v2_hit'].fillna(False).any()) if 'v2_hit' in rows_df else False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Injection test of the exocomet search')
    parser.add_argument('--c', default='spectral_classes.json', help='config file (default: spectral_classes.json)')
    parser.add_argument('--r', default='../results/Reliable/', help='output path (default: ../results/Reliable/)')
    parser.add_argument('--two-runs', action='store_true', help='each fake comet through the PCA on and off searches')
    args = parser.parse_args()

    with open(args.c) as configfile:
        config = json.load(configfile)
    config['search']['n_stars'] = len(exocomet_search.build_jobs(config))
    scores = pd.read_pickle(os.path.join(args.r, 'spectrum_scores.pkl'))
    n_stars = int(scores['n_stars'].iloc[0])

    stars = scores[scores['n'] < MAX_SPECTRA].drop_duplicates('folder')
    stars = stars.sample(frac=1, random_state=0).groupby('class_name').head(STARS_PER_CLASS)
    plans = [(d, w) for d in DEPTHS for w in FWHMS] + [('rich', RICH_FWHM)]
    searches = (True, False) if args.two_runs else (True,)
    jobs = [(f, c, d, w, pca) for f, c in zip(stars['folder'], stars['class_name']) for d, w in plans for pca in searches]
    print('Injection runs:', len(jobs), flush=True)

    out = []
    with multiprocessing.Pool(config['search']['cores'], initializer=exocomet_search.init_worker, initargs=(config,)) as pool:
        for job, rows in zip(jobs, pool.imap(inject_star, jobs, chunksize=4)):
            if len(rows) == 0:
                continue
            out.append(dict(class_name=job[1], depth=job[2], fwhm=job[3], pca=job[4], folder=job[0], n=rows[0]['n'],
                            **recovered(rows, config, n_stars)))

    out = pd.DataFrame(out)
    out.to_pickle(os.path.join(args.r, 'injection_runs.pkl'))
    columns = ['v2_hit', 'above_threshold', 'passed', 'confirmed', 'share_found']
    keys = ['class_name', 'depth', 'fwhm', 'folder']
    either = out.groupby(keys)[columns].max().reset_index().assign(search='either')
    out['search'] = np.where(out['pca'], 'PCA on', 'PCA off')
    combined = pd.concat([out.drop(columns=['pca', 'n']), either]) if args.two_runs else out.drop(columns=['pca', 'n'])
    combined['depth'] = combined['depth'].astype(str)
    table = 100 * combined.groupby(['search', 'class_name', 'depth'])[columns].mean()
    table.round(1).to_csv(os.path.join(args.r, 'injection_recovery.csv'))
    print(table.round(1).to_string())
    normal = combined[combined['depth'] != 'rich']
    print((100 * normal.groupby(['search', 'class_name', 'fwhm'])['confirmed'].mean()).round(1).unstack().to_string())
