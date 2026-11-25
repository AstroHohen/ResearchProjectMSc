'''
07/11/26
Exocomet search by spectral class, reliable version. Stage 2 of 2: from spectrum_scores.pkl to candidates, with no
visual step. README.md in this folder explains every change from harps/search_by_type.

Detection: one false alarm budget for the whole survey, false_alarm_survey (1%: about one false star in a hundred
searches of the whole archive). Stage 1 gives each spectrum the chance p_spec of its score in one spectrum of its star,
so for N stars of n spectra each
    p_star = n * p_spec,    p_survey = N * p_star,    detection if p_survey < false_alarm_survey
The same with the night stacks (n nights). There is no cap: the null now comes from the upside-down spectra, where real
comets do not count, so a star with many comets no longer raises it. A score must also be larger than the largest
emission doublets of its star (without the top MIRROR_SHARE), as before.

Tests on each detection (Bendahan-West et al. false positive categories, then the new ones):
- transit:  nearest spectra without the feature are at least 30 min apart
- shift:    the residual does not follow the slope of the reference (binaries, RV shifts)
- twin:     no emission of similar size beside the feature (P Cygni, line profile changes)
- shape:    K and H both follow the Gaussian profile at that velocity
- activity: the feature does not vary with the flux of the line core (late types)
- depth:    not deeper than the star's flux
- control:  NEW, no change (of either sign) at the same offset in the photospheric lines of the window, and no
            correlation over time between the feature and those lines (pulsation, spots and companions change every
            line)
- period:   NEW, no periodic signal at that velocity, fitted without the night, that predicts half the feature or more
- veto:     NEW, flag only (veto_check.py): a dip at the same velocity in a photospheric line from 6 eV or more, or an
            unusual H alpha / He I D3 index in the same spectrum. Labelled in the veto column, never removed

Tiers of a star, in a class where Ca II can make a detection:
- high confidence: passes all, confirmed by an independent repeat in the same night (p < repeat_false_alarm), the
                   control test could have seen a change of the star of that size, not near H epsilon
- confirmed, no control lines: the same, but the star has no photospheric line the control test can use (fast
                   rotating A stars such as bet Pic), so a change of the star is not ruled out by it
- candidate:       passes all, but not confirmed by a repeat, or near H epsilon, or the control lines are too weak
- anomaly:         passes all in OB or M, where winds or activity mean Ca II alone cannot make a detection
Features at a fixed velocity for months, and features inside the width of the photospheric lines, are only labelled.
'''
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from veto_check import add_veto, veto_stage

MIRROR_SHARE = 0.002
TESTS = ['pass_transit', 'pass_shift', 'pass_twin', 'pass_shape', 'pass_activity', 'pass_depth', 'pass_control',
         'pass_period']


def sigma_of(p):
    '''One-sided Gaussian significance of a probability'''
    return norm.isf(np.clip(p, 1e-300, 0.5))  # 0 sigma for anything that is not significant at all


def apply_tests(rows, tests):
    '''One boolean column per test, True = passed'''
    rows = rows.copy()
    rows['pass_transit'] = rows['transit_min'] >= tests['min_transit_minutes']
    rows['pass_shift'] = rows['shift_corr'] <= tests['max_rv_shift_corr']
    rows['pass_twin'] = rows['emission_twin'] <= tests['max_emission_twin']
    if tests.get('shape_mode', 'both') == 'weighted':  # H weighted by its depth relative to K in thin gas
        rows['pass_shape'] = (rows['shape_K'] + 0.5 * rows['shape_H']) / 1.5 >= tests['min_shape_corr']
    else:
        rows['pass_shape'] = rows[['shape_K', 'shape_H']].min(axis=1) >= tests['min_shape_corr']
    late = rows['class_name'].isin(['lateF_G_K', 'M'])
    rows['pass_activity'] = ~late | (rows['activity_corr'] <= tests['max_activity_corr'])
    rows['pass_depth'] = rows[['depth_K', 'depth_H']].max(axis=1) < 1  # a cloud cannot remove more than all the flux
    no_control = rows['control_score'].isna()
    # a change of the star can have either sign at the control lines, so a large change of either sign fails
    rows['pass_control'] = no_control | ((rows['control_score'].abs() < -tests['min_control_score'])
                                         & (rows['control_corr'].abs().fillna(0) <= tests['max_control_corr']))
    rows['control_informative'] = ~no_control & (rows['control_expected'] <= tests['control_informative'])
    periodic = (rows['period_fap'] < tests['max_period_fap']) & (rows['period_share'] >= tests['max_period_share'])
    rows['pass_period'] = ~periodic.fillna(False)
    rows['pass_all'] = rows[TESTS].all(axis=1)
    rows['confirmed'] = ((rows['by_spectrum'] & (rows['p_repeat'] < tests['repeat_false_alarm']))
                         | (rows['by_stack'] & (rows['p_halves'] < tests['repeat_false_alarm'])))
    return rows


def mirror_level(rows, column):
    '''Size of the largest emission of each star, without the top MIRROR_SHARE'''
    return rows.groupby('folder')[column].transform(lambda x: x.quantile(1 - MIRROR_SHARE, interpolation='higher'))


def tier_of(row, classes):
    if classes[row['class_name']]['confidence'] == 'anomaly':
        return 'anomaly'
    if not row['confirmed'] or row['near_h_epsilon']:
        return 'candidate'
    if row['control_informative']:
        return 'high confidence'
    return 'confirmed, no control lines' if row['n_control'] == 0 else 'candidate'


def missing_for_top(row):
    '''What keeps a candidate out of the top tier'''
    reasons = [name for name, ok in (('no repeat', row['confirmed']), ('control too weak', row['control_informative']),
                                     ('H epsilon', not row['near_h_epsilon'])) if not ok]
    return ', '.join(reasons)


def significance(scores, n_stars):
    '''Probabilities and sigma of each spectrum and night stack, for its star and for the survey'''
    scores = scores.copy()
    scores['p_star'] = np.minimum(1, scores['n'] * scores['p_spec'])
    scores['p_survey'] = np.minimum(1, n_stars * scores['p_star'])
    scores['p_night_star'] = np.minimum(1, scores['n_nights'] * scores['p_night'])
    scores['p_night_survey'] = np.minimum(1, n_stars * scores['p_night_star'])
    for column in ('p_star', 'p_survey', 'p_night_star', 'p_night_survey'):
        scores['sigma' + column[1:]] = sigma_of(scores[column])
    return scores


def decide(scores, config, n_stars=None):
    '''All detections with their tests, and the best one of each star that passed'''
    tests = config['tests']
    budget = tests['false_alarm_survey'] / tests.get('searches', 1)  # shared by the PCA on and off searches
    if n_stars is None:  # every star stage 1 could search, also in a test run on a few stars
        n_stars = int(scores['n_stars'].iloc[0]) if 'n_stars' in scores else scores['folder'].nunique()
    scores = significance(scores, n_stars)
    scores['mirror_level'] = mirror_level(scores, 'mirror')
    scores['night_mirror_level'] = mirror_level(scores, 'night_mirror')
    split = 2 if tests.get('split_spectrum_night', False) else 1  # addendum 10: spectra and stacks share the budget
    scores['by_spectrum'] = (scores['p_survey'] < budget / split) & (scores['score'] > scores['mirror_level'])
    scores['by_stack'] = ((scores['p_night_survey'] < budget / split) & (scores['night_score'] > scores['night_mirror_level'])
                          & ((scores['v0'] - scores['night_v0']).abs() <= 2 * scores['fwhm'].fillna(0)))
    kept = scores.dropna(subset=['fwhm'])
    kept = kept[kept['by_spectrum'] | kept['by_stack']].copy()
    kept = apply_tests(kept, tests)
    passed = kept[kept['pass_all']]
    best = passed.sort_values(['confirmed', 'sigma_survey'], ascending=False).groupby('folder', as_index=False).first()
    if len(best):
        best['tier'] = best.apply(tier_of, axis=1, classes=config['classes'])
        best['missing_for_top'] = best.apply(missing_for_top, axis=1)
        best = add_spread(best, passed, tests)
    return kept, best


BERV_FLAG_KMS = 10  # scattered moonlight or sky sits at the BERV in the barycentric frame (no sky subtraction, manual 3.2)
STATE_DB = 'C:/data/harps/state.sqlite'


def add_berv_flag(rows, state_db=STATE_DB):
    '''Flag only: detections whose barycentric velocity is within BERV_FLAG_KMS of the BERV of their spectrum'''
    import sqlite3
    if len(rows) == 0 or not os.path.exists(state_db):
        return rows.assign(near_berv=False)
    with sqlite3.connect(f'file:{state_db}?mode=ro', uri=True) as con:
        berv = {}
        ids = list(rows['dp_id'].unique())
        for start in range(0, len(ids), 500):
            part = ids[start:start + 500]
            query = 'select dp_id, header_json from spectra where dp_id in ({})'.format(','.join('?' * len(part)))
            for dp_id, header in con.execute(query, part):
                berv[dp_id] = json.loads(header or '{}').get('HIERARCH ESO DRS BERV', np.nan)
    rows = rows.assign(berv=rows['dp_id'].map(berv).astype(float))
    return rows.assign(near_berv=((rows['v0'] + rows['v_star'] - rows['berv']).abs() < BERV_FLAG_KMS).fillna(False))


def add_spread(best, passed, tests):
    '''Flag stars whose detections keep one velocity over several nights (interstellar or circumstellar gas)'''
    spread = passed.groupby('folder').agg(v_spread=('v0', 'std'), nights=('t', lambda t: len(np.unique(np.floor(t)))))
    best = best.merge(spread, on='folder')
    best['fixed_velocity'] = (best['v_spread'] < tests['stable_velocity_kms']) & (best['nights'] >= tests['stable_min_nights'])
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exocomet search by spectral class, stage 2')
    parser.add_argument('--c', default='spectral_classes.json', help='config file (default: spectral_classes.json)')
    parser.add_argument('--r', default='../results/Reliable/', help='output path (default: ../results/Reliable/)')
    parser.add_argument('--flip', action='store_true', help='the run in --r is the flipped one (default: "flip" in --r)')
    args = parser.parse_args()
    flipped = args.flip or 'flip' in os.path.basename(os.path.normpath(args.r)).lower()

    with open(args.c) as configfile:
        config = json.load(configfile)

    scores = pd.read_pickle(os.path.join(args.r, 'spectrum_scores.pkl'))
    rule = config['tests'].get('rule', 'baseline')
    if rule == 'baseline':
        kept, best = decide(scores, config)
    else:  # decision_rules.py; type_weight is the rule chosen on 14 Nov 2026 (results/Overnight/PREREGISTRATION.md)
        from decision_rules import decide_rule, survey_class_counts
        kept, best = decide_rule(scores, config, rule, class_counts=survey_class_counts(args.r, config))
    print('Decision rule:', rule, '| shape test:', config['tests'].get('shape_mode', 'both'))
    kept, best = add_berv_flag(kept), add_berv_flag(best)
    # veto lines of the star at the velocity of each detection: flags only, no detection is removed (veto_check.py)
    kept, best = add_veto(kept, best, veto_stage(kept, scores, config, args.r, flipped))

    summary = pd.DataFrame({'stars': scores.groupby('class_name')['folder'].nunique(),
                            'spectra': scores.groupby('class_name').size(),
                            'above_threshold': kept.groupby('class_name')['folder'].nunique(),
                            'after_tests': best.groupby('class_name')['folder'].nunique() if len(best) else 0}).fillna(0)
    print('Stars in the survey:', int(scores['n_stars'].iloc[0]), '| stars with scores:', scores['folder'].nunique())
    print(summary.to_string())
    for test in TESTS + ['confirmed', 'control_informative', 'near_h_epsilon', 'inside_photosphere']:
        print('{}: {:.1f}% of detections'.format(test, 100 * kept[test].mean()))

    best.to_csv(os.path.join(args.r, 'candidates.csv'), index=False)
    kept.to_pickle(os.path.join(args.r, 'scored_detections.pkl'))
    print('Candidates:', len(best), best['tier'].value_counts().to_dict() if len(best) else '')
    if len(best) and 'veto' in best:
        print('Veto of the candidates:', best['veto'].fillna('no veto data').value_counts().to_dict())
    if len(best):
        print(best.groupby(['class_name', 'tier']).size().unstack(fill_value=0).to_string())
