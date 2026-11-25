'''
01/11/26
Injection test of the exocomet search: how many fake exocomets does each spectral class recover?
Run after both stages. Each run puts one fake comet in up to three consecutive spectra of one night of a star, in K and H
with depth(K) / depth(H) = 1.5, at a random velocity from the star. A comet counts as recovered at the velocity it was put.
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


def inject_star(job):
    folder, class_name, depth, fwhm = job
    return exocomet_search.search_star((folder, class_name), dict(depth=depth, fwhm=fwhm, epochs=EPOCHS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Injection test of the exocomet search')
    parser.add_argument('--c', default='spectral_classes.json', help='config file (default: spectral_classes.json)')
    parser.add_argument('--r', default='../results/ByType/', help='output path (default: ../results/ByType/)')
    args = parser.parse_args()

    with open(args.c) as configfile:
        config = json.load(configfile)
    scores = pd.read_pickle(os.path.join(args.r, 'spectrum_scores.pkl'))
    curves = decide(scores, config)[2]

    stars = scores[scores['n'] < MAX_SPECTRA].drop_duplicates('folder')
    stars = stars.sample(frac=1, random_state=0).groupby('class_name').head(STARS_PER_CLASS)
    jobs = [(f, c, d, w) for f, c in zip(stars['folder'], stars['class_name']) for d in DEPTHS for w in FWHMS]
    print('Injection runs:', len(jobs), flush=True)

    out = []
    with multiprocessing.Pool(config['search']['cores'], initializer=exocomet_search.init_worker, initargs=(config,)) as pool:
        for job, rows in zip(jobs, pool.imap(inject_star, jobs, chunksize=4)):
            if len(rows) == 0:
                continue
            rows_df = pd.DataFrame(rows)
            kept = decide(rows_df, config, curves)[0] if 'fwhm' in rows_df else rows_df.iloc[:0].assign(pass_all=False, confirmed=False)
            hits = kept[kept['injected'] & kept['near_injection']]
            out.append(dict(class_name=job[1], depth=job[2], fwhm=job[3], n=rows[0]['n'], above_threshold=len(hits) > 0,
                            v2_hit=bool(rows_df['v2_hit'].fillna(False).any()),
                            passed=hits['pass_all'].any(), confirmed=(hits['pass_all'] & hits['confirmed']).any()))

    out = pd.DataFrame(out)
    out.to_pickle(os.path.join(args.r, 'injection_runs.pkl'))
    table = 100 * out.groupby(['class_name', 'depth'])[['v2_hit', 'above_threshold', 'passed', 'confirmed']].mean()
    table.round(1).to_csv(os.path.join(args.r, 'injection_recovery.csv'))
    print(table.round(1).to_string())
    print((100 * out.groupby(['class_name', 'fwhm'])['confirmed'].mean()).round(1).unstack().to_string())
