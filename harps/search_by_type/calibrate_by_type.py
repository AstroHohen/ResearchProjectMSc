'''
01/11/26
Exocomet search by spectral class. Stage 2 of 2: from spectrum_scores.pkl to candidates, with no visual step.

Threshold: each star has its own from stage 1, for 1% false stars, changed here in two ways:
- a star with many real events raises its own null, so no threshold goes above threshold_cap * the typical threshold
  of its class at that number of spectra
- that null has no real K-H doublet, so it misses stellar variability in both lines. Comets only absorb, so a star's
  largest doublets in emission show the size of that variability, and an absorption has to be larger. A share
  MIRROR_SHARE of them is ignored, so one bad night does not close a star of thousands of spectra
A feature counts if the spectrum passes, or if the stack of its night passes at the velocity of the spectrum.

Tests on each spectrum above the threshold (Bendahan-West et al. false positive categories):
- transit:  nearest spectra without the feature are at least 30 min apart
- shift:    the residual does not follow the slope of the reference (binaries, RV shifts)
- twin:     no emission of similar size beside the feature (P Cygni, line profile changes)
- shape:    K and H both follow the Gaussian profile at that velocity
- activity: the feature does not vary with the flux of the line core (late types)
- depth:    not deeper than the star's flux

Tiers of a star:
- high confidence: passes all, seen in 2+ spectra of the same night, in a class where Ca II can make a detection
- candidate:       passes all, but seen in one spectrum, or no other spectrum that night to confirm it
- anomaly:         passes all in OB or M, where winds or activity mean Ca II alone cannot make a detection
Features at a fixed velocity for months are only labelled: bet Pic has recurring comets, so they are not rejected.
'''
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

MIRROR_SHARE = 0.002


def apply_tests(rows, tests):
    '''One boolean column per test, True = passed'''
    rows = rows.copy()
    rows['pass_transit'] = rows['transit_min'] >= tests['min_transit_minutes']
    rows['pass_shift'] = rows['shift_corr'] <= tests['max_rv_shift_corr']
    rows['pass_twin'] = rows['emission_twin'] <= tests['max_emission_twin']
    rows['pass_shape'] = rows[['shape_K', 'shape_H']].min(axis=1) >= tests['min_shape_corr']
    late = rows['class_name'].isin(['lateF_G_K', 'M'])
    rows['pass_activity'] = ~late | (rows['activity_corr'] <= tests['max_activity_corr'])
    rows['pass_depth'] = rows[['depth_K', 'depth_H']].max(axis=1) < 1  # a cloud cannot remove more than all the flux
    tests_passed = ['pass_transit', 'pass_shift', 'pass_twin', 'pass_shape', 'pass_activity', 'pass_depth']
    rows['pass_all'] = rows[tests_passed].all(axis=1)
    rows['confirmed'] = (rows['night_confirmed'] >= 2) | (rows['by_stack'] & (rows['night_size'] >= 2))
    return rows


def typical_thresholds(scores, column='threshold'):
    '''Line of the threshold of each class against log10(spectra). Theil-Sen, as bet Pic is both far out and extreme'''
    stars = scores.groupby('folder').agg(class_name=('class_name', 'first'), n=('n', 'first'), value=(column, 'first'))
    return {name: theilslopes(rows['value'], np.log10(rows['n']))[:2] for name, rows in stars.groupby('class_name')}


def final_threshold(rows, column, mirror, curves, tests):
    '''Threshold of each row after both changes'''
    line = rows['class_name'].map(curves)
    typical = line.str[0] * np.log10(rows['n']) + line.str[1]
    return np.maximum(np.minimum(rows[column], tests['threshold_cap'] * typical), mirror)


def mirror_level(rows, column):
    '''Size of the largest emission of each star, without the top MIRROR_SHARE'''
    return rows.groupby('folder')[column].transform(lambda x: x.quantile(1 - MIRROR_SHARE, interpolation='higher'))


def tier_of(row, classes):
    if classes[row['class_name']]['confidence'] == 'anomaly':
        return 'anomaly'
    return 'high confidence' if row['confirmed'] else 'candidate'


def decide(scores, config, curves=None):
    '''All detections with their tests and final threshold, and the best one of each star that passed'''
    tests = config['tests']
    if curves is None:
        curves = {column: typical_thresholds(scores, column) for column in ('threshold', 'night_threshold')}
    scores = scores.copy()
    scores['final_threshold'] = final_threshold(scores, 'threshold', mirror_level(scores, 'mirror'), curves['threshold'], tests)
    scores['final_night'] = final_threshold(scores, 'night_threshold', mirror_level(scores, 'night_mirror'),
                                            curves['night_threshold'], tests)
    scores['by_stack'] = (scores['night_score'] > scores['final_night']) & ((scores['v0'] - scores['night_v0']).abs() <= 2 * scores['fwhm'].fillna(0))
    kept = scores.dropna(subset=['fwhm'])
    kept = kept[(kept['score'] > kept['final_threshold']) | kept['by_stack']].copy()
    kept = apply_tests(kept, tests)
    passed = kept[kept['pass_all']]
    best = passed.sort_values(['confirmed', 'score'], ascending=False).groupby('folder', as_index=False).first()
    if len(best):
        best['tier'] = best.apply(tier_of, axis=1, classes=config['classes'])
        spread = passed.groupby('folder').agg(v_spread=('v0', 'std'), nights=('t', lambda t: len(np.unique(np.floor(t)))))
        best = best.merge(spread, on='folder')
        best['fixed_velocity'] = (best['v_spread'] < tests['stable_velocity_kms']) & (best['nights'] >= tests['stable_min_nights'])
    return kept, best, curves


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exocomet search by spectral class, stage 2')
    parser.add_argument('--c', default='spectral_classes.json', help='config file (default: spectral_classes.json)')
    parser.add_argument('--r', default='../results/ByType/', help='output path (default: ../results/ByType/)')
    args = parser.parse_args()

    with open(args.c) as configfile:
        config = json.load(configfile)

    scores = pd.read_pickle(os.path.join(args.r, 'spectrum_scores.pkl'))
    kept, best, curves = decide(scores, config)

    summary = pd.DataFrame({'stars': scores.groupby('class_name')['folder'].nunique(),
                            'spectra': scores.groupby('class_name').size(),
                            'above_threshold': kept.groupby('class_name')['folder'].nunique(),
                            'after_tests': best.groupby('class_name')['folder'].nunique() if len(best) else 0}).fillna(0)
    print(summary.to_string())
    for test in ('pass_transit', 'pass_shift', 'pass_twin', 'pass_shape', 'pass_activity', 'pass_depth'):
        print('{}: {:.1f}% of detections'.format(test, 100 * kept[test].mean()))

    best.to_csv(os.path.join(args.r, 'candidates.csv'), index=False)
    kept.to_pickle(os.path.join(args.r, 'scored_detections.pkl'))
    print('Candidates:', len(best), best['tier'].value_counts().to_dict() if len(best) else '')
