'''
21/11/26
Reference sample for event_stats.py (addendum 13): one feature per bet Pic night, at the velocity and width of the
night stack's best doublet in stage 1 (PCA-on run), measured by events.measure exactly as an event. Kept when the
night-mean depth of K is above 5 times its error and that of H above 3 times.
    py -3 bet_pic_reference.py --out ../results/FinalCheck2/
Writes <out>/events/bet_pic_reference.csv
'''
import argparse
import json
import os

import numpy as np
import pandas as pd

import events as ev
import exocomet_search as es

BET_PIC = 'GaiaDR3_4792774797545800832'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='../results/FinalCheck2/')
    args = parser.parse_args()
    es.init_worker(json.load(open(os.path.join(args.out, 'config.json'))))
    scores = pd.read_pickle(os.path.join(args.out, 'on', 'spectrum_scores.pkl'))
    scores = scores[scores['folder'] == BET_PIC].copy()
    scores['night'] = np.floor(scores['t'] - 0.5)
    fwhm_grid = es.Config['search']['fwhm_kms']
    star = ev.load_star(BET_PIC)
    rows = []
    for night, group in scores.groupby('night'):
        top = group.sort_values('night_score', ascending=False).iloc[0]
        width = top['fwhm'] if np.isfinite(top.get('fwhm', np.nan)) else 10.0
        e = pd.Series(dict(event=f'ref_{int(night)}', night=night, v0=top['night_v0'], fwhm=width,
                           v_star=top['v_star'], night_score=top['night_score']))
        m, per, _ = ev.measure(e, star, flip=False)
        rows.append({**e.to_dict(), **m})
    out = pd.DataFrame(rows)
    keep = (out['depth_K'] > 5 * out['depth_K_err']) & (out['depth_H'] > 3 * out['depth_H_err'])
    out['kept'] = keep
    out.to_csv(os.path.join(args.out, 'events', 'bet_pic_reference.csv'), index=False)
    print(f'{len(out)} nights, {int(keep.sum())} kept')
    print(out[keep][['v_fit', 'depth_K', 'fwhm_fit', 'ratio_KH', 'duration_h']].describe().round(3).to_string())
