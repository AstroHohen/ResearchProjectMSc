'''
13/11/26
Three other decision rules for stage 2, tested against the baseline (calibrate_by_type.decide) on the flipped control,
the fake comets and the known hosts. Fixed before any run in results/Overnight/PREREGISTRATION.md.

- baseline:    calibrate_by_type.decide
- type_weight: the survey budget shared out by spectral type, from where Ca II exocomets are known (weighted Bonferroni)
- star_level:  half the budget for the baseline, half for a star test (how many nights have a night stack with
               p_night < NIGHT_P), then the star's own spectra at the per-star level
- pairs:       half the budget for the baseline, half for pairs: a spectrum and another of its night at the same
               velocity (p_spec * p_repeat)
Stage 1 must have computed the slow tests for more spectra (search['slow_budget_factor']).
'''
import numpy as np
import pandas as pd
from scipy.stats import binom

import json
import os

from calibrate_by_type import add_spread, apply_tests, mirror_level, missing_for_top, significance, sigma_of, tier_of

RULES = ['baseline', 'type_weight', 'star_level', 'pairs', 'combined']
TYPE_WEIGHTS = {'A_earlyF': 1.0, 'OB': 0.1, 'lateF_G_K': 0.01, 'M': 0.01}
NIGHT_P = 1e-3
SLOW_P_SURVEY = 25  # the slow tests are only computed below this p_survey (slow_budget_factor 10,000)


def star_test(scores, n_stars):
    '''Chance of the survey that a star has as many nights with p_night < NIGHT_P as it has'''
    nights = scores.assign(night=np.floor(scores['t'] - 0.5)).drop_duplicates(['folder', 'night'])
    per = nights.groupby('folder').agg(k=('p_night', lambda p: int((p < NIGHT_P).sum())), n_nights=('n_nights', 'first'))
    per['p_star_test'] = binom.sf(per['k'] - 1, per['n_nights'], NIGHT_P)
    per['p_star_survey'] = np.minimum(1, n_stars * per['p_star_test'])
    return per


def detections(scores, budget, column, night_column, split=1):
    '''Spectra or night stacks below the budget, above the largest emission of their star. With split 2 the spectra and
    the night stacks each spend half of the budget (tests.split_spectrum_night, addendum 10)'''
    budget = budget / split
    by_spectrum = (scores[column] < budget) & (scores['score'] > scores['mirror_level'])
    by_stack = ((scores[night_column] < budget) & (scores['night_score'] > scores['night_mirror_level'])
                & ((scores['v0'] - scores['night_v0']).abs() <= 2 * scores['fwhm'].fillna(0)))
    return by_spectrum, by_stack


def decide_rule(scores, config, rule, n_stars=None, class_counts=None):
    '''All detections of one search with their tests, and the best one of each star that passed'''
    tests = config['tests']
    budget = tests['false_alarm_survey'] / tests.get('searches', 1)
    split = 2 if tests.get('split_spectrum_night', False) else 1
    if n_stars is None:
        n_stars = int(scores['n_stars'].iloc[0]) if 'n_stars' in scores else scores['folder'].nunique()
    scores = significance(scores, n_stars)
    scores['mirror_level'] = mirror_level(scores, 'mirror')
    scores['night_mirror_level'] = mirror_level(scores, 'night_mirror')
    if not tests.get('use_mirror', True):  # test switch: no comparison with the star's largest emission
        scores['mirror_level'] = scores['night_mirror_level'] = -np.inf
    scores['route'] = ''
    star_pass = pd.Series(False, index=scores.index)

    if rule == 'baseline':
        by_spectrum, by_stack = detections(scores, budget, 'p_survey', 'p_night_survey', split)
        scores.loc[by_spectrum | by_stack, 'route'] = 'spectrum'
        scores['by_spectrum'], scores['by_stack'] = by_spectrum, by_stack
    elif rule == 'type_weight':
        total = sum(TYPE_WEIGHTS[c] * k for c, k in class_counts.items())
        trials = total / scores['class_name'].map(TYPE_WEIGHTS)  # weighted number of stars
        scores['p_weighted'] = np.minimum(1, trials * scores['p_star'])
        scores['p_night_weighted'] = np.minimum(1, trials * scores['p_night_star'])
        by_spectrum, by_stack = detections(scores, budget, 'p_weighted', 'p_night_weighted', split)
        scores.loc[by_spectrum | by_stack, 'route'] = 'weighted'
        scores['by_spectrum'], scores['by_stack'] = by_spectrum, by_stack
    elif rule == 'star_level':
        by_spectrum, by_stack = detections(scores, budget / 2, 'p_survey', 'p_night_survey', split)
        scores.loc[by_spectrum | by_stack, 'route'] = 'spectrum'
        per = star_test(scores, n_stars)
        passing = per.index[per['p_star_survey'] < budget / 2]
        star_pass = scores['folder'].isin(passing)
        own_spectrum, own_stack = detections(scores, budget / 2, 'p_star', 'p_night_star', split)
        own = star_pass & (own_spectrum | own_stack)
        scores.loc[own & (scores['route'] == ''), 'route'] = 'star'
        scores['by_spectrum'] = by_spectrum | (star_pass & own_spectrum)
        scores['by_stack'] = by_stack | (star_pass & own_stack)
        scores = scores.merge(per[['k', 'p_star_survey']].rename(columns={'p_star_survey': 'p_star_test_survey'}),
                              left_on='folder', right_index=True, how='left')
        star_pass = scores['folder'].isin(passing)
    elif rule == 'pairs':
        by_spectrum, by_stack = detections(scores, budget / 2, 'p_survey', 'p_night_survey', split)
        scores.loc[by_spectrum | by_stack, 'route'] = 'spectrum'
        p_repeat = scores['p_repeat'] if 'p_repeat' in scores else pd.Series(1.0, index=scores.index)
        # from the uncapped chance of the spectrum: a capped p_survey of 1 times p_repeat is not a pair chance
        raw_survey = n_stars * scores['n'] * scores['p_spec']
        scores['p_pair_survey'] = np.minimum(1, raw_survey * p_repeat.fillna(1))
        pair_ok = raw_survey < SLOW_P_SURVEY
        pair = ((scores['p_pair_survey'] < budget / 2) & pair_ok
                & (scores['score'] > scores['mirror_level']))
        scores.loc[pair & (scores['route'] == ''), 'route'] = 'pair'
        scores['by_spectrum'], scores['by_stack'] = by_spectrum | pair, by_stack
    elif rule == 'combined':  # type weights on all three routes, each with a third of the budget
        third = budget / 3
        total = sum(TYPE_WEIGHTS[c] * k for c, k in class_counts.items())
        trials = total / scores['class_name'].map(TYPE_WEIGHTS)
        scores['p_weighted'] = np.minimum(1, trials * scores['p_star'])
        scores['p_night_weighted'] = np.minimum(1, trials * scores['p_night_star'])
        by_spectrum, by_stack = detections(scores, third, 'p_weighted', 'p_night_weighted', split)
        scores.loc[by_spectrum | by_stack, 'route'] = 'spectrum'
        per = star_test(scores, n_stars)
        star_trials = scores.groupby('folder')['class_name'].first().map(lambda c: total / TYPE_WEIGHTS[c])
        per['p_star_weighted'] = np.minimum(1, per['p_star_test'] * star_trials.reindex(per.index))
        passing = per.index[per['p_star_weighted'] < third]
        star_pass = scores['folder'].isin(passing)
        own_spectrum, own_stack = detections(scores, third, 'p_star', 'p_night_star', split)
        scores.loc[star_pass & (own_spectrum | own_stack) & (scores['route'] == ''), 'route'] = 'star'
        p_repeat = scores['p_repeat'] if 'p_repeat' in scores else pd.Series(1.0, index=scores.index)
        raw_weighted = trials * scores['n'] * scores['p_spec']  # uncapped, as in pairs
        scores['p_pair_weighted'] = np.minimum(1, raw_weighted * p_repeat.fillna(1))
        pair = ((scores['p_pair_weighted'] < third) & (n_stars * scores['n'] * scores['p_spec'] < SLOW_P_SURVEY)
                & (scores['score'] > scores['mirror_level']))
        scores.loc[pair & (scores['route'] == ''), 'route'] = 'pair'
        scores['by_spectrum'] = by_spectrum | (star_pass & own_spectrum) | pair
        scores['by_stack'] = by_stack | (star_pass & own_stack)
        scores = scores.merge(per[['k', 'p_star_weighted']], left_on='folder', right_index=True, how='left')
        star_pass = scores['folder'].isin(passing)
    else:
        raise ValueError(rule)

    scores['star_pass'] = star_pass
    kept = scores.dropna(subset=['fwhm'])
    kept = kept[kept['route'] != ''].copy()
    kept = apply_tests(kept, tests)
    passed = kept[kept['pass_all']]
    best = passed.sort_values(['confirmed', 'sigma_survey'], ascending=False).groupby('folder', as_index=False).first()
    if len(best):
        best['tier'] = best.apply(tier_of, axis=1, classes=config['classes'])
        if rule in ('star_level', 'combined'):  # the star test stands in for missing control lines
            top = (best['star_pass'] & best['confirmed'] & ~best['near_h_epsilon']
                   & (best['control_informative'] | (best['n_control'] == 0))
                   & (best['tier'] != 'anomaly'))
            best.loc[top, 'tier'] = 'high confidence'
        best['missing_for_top'] = best.apply(missing_for_top, axis=1)
        best = add_spread(best, passed, tests)
    return kept, best


def survey_class_counts(results, config):
    '''Stars of each class in the survey (the trials), written by stage 1; rebuilt from the dataset if missing'''
    path = os.path.join(results, 'class_counts.json')
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    from exocomet_search import build_jobs
    return build_jobs(config)['class_name'].value_counts().to_dict()
