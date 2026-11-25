'''
07/11/26
Two checks on each candidate star after stage 2, too slow to run on every star:

1. Joint fit (idea 10). The matched filter gives depths at fixed velocity and width. Here one Gaussian, the same
   velocity and width in both lines, is fitted to the cleaned residuals of K and H of the detection spectrum:
       r_K(v) = c_K - a_K ref_K(v) g(v),   r_H(v) = c_H - a_H ref_H(v) g(v),   g = exp(-(v - mu)^2 / 2 s^2)
   weighted by the noise of stage 1, over +-4 FWHM. It gives the depths, velocity and FWHM with errors, the K/H
   ratio with its error (1 for thick gas, 2 for thin gas), and delta chi2 against no feature (c_K and c_H only).

2. The tail of the null (my earlier point 5). Stage 1 fits a Pareto tail to 200 random shifts of H against K and
   extrapolates it far out. Here the same null is drawn again with SHIFTS shifts, and the share of it above the score
   is compared with what the 200-shift tail predicted, at the scores of the candidates.
'''
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import exocomet_search as es

SHIFTS = 2000
FWHM_TO_SIGMA = 2.354820045


def joint_fit(state, i, v0, fwhm):
    '''Least squares of one Gaussian in K and H. Returns its values and errors'''
    v = state['v'][state['in_range']]
    grid = state['grid']
    centre = v[int(np.argmin(np.abs(grid - v0)))]
    near = np.abs(v - centre) < 4 * fwhm
    data = []
    for line in (state['lineK'], state['lineH']):
        resid = line['resid'][i][state['in_range']][near]
        weight = np.sqrt(line['inv_var'][i][state['in_range']][near])
        data.append((resid, weight, line['ref'][state['in_range']][near]))
    x = v[near]

    def model(p, full=True):
        aK, aH, mu, s, cK, cH = p
        g = np.exp(-(x - mu) ** 2 / (2 * s ** 2)) if full else 0
        return [cK - aK * data[0][2] * g, cH - aH * data[1][2] * g]

    def misfit(p, full=True):
        m = model(p, full)
        return np.concatenate([(data[k][0] - m[k]) * data[k][1] for k in range(2)])

    start = [0.05, 0.03, centre, fwhm / FWHM_TO_SIGMA, 0, 0]
    fit = least_squares(misfit, start, bounds=([-1, -1, centre - 2 * fwhm, 0.5, -1, -1],
                                               [1, 1, centre + 2 * fwhm, 3 * fwhm, 1, 1]))
    chi2 = float(np.sum(fit.fun ** 2))
    dof = max(len(fit.fun) - 6, 1)
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * max(chi2 / dof, 1)
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov, err = None, np.full(6, np.nan)
    flat = least_squares(lambda p: misfit([0, 0, centre, 1, p[0], p[1]], full=False), [0, 0])
    aK, aH = fit.x[:2]
    ratio = aK / aH if aH != 0 else np.nan
    ratio_err = abs(ratio) * np.hypot(err[0] / aK, err[1] / aH) if aK and aH else np.nan
    return dict(fit_depth_K=aK, fit_depth_K_err=err[0], fit_depth_H=aH, fit_depth_H_err=err[1],
                fit_ratio=ratio, fit_ratio_err=ratio_err, fit_v=fit.x[2] - (centre - v0), fit_v_err=err[2],
                fit_fwhm=fit.x[3] * FWHM_TO_SIGMA, fit_fwhm_err=err[3] * FWHM_TO_SIGMA,
                fit_delta_chi2=float(np.sum(flat.fun ** 2)) - chi2, fit_chi2_dof=chi2 / dof)


def tail_check(state, score, rng):
    '''Share of a 2,000-shift null above the score, against the 200-shift Pareto tail of stage 1'''
    fit, _ = es.star_null(state['null_K'], state['null_H'], state['floor'], SHIFTS, rng)
    samples = SHIFTS * state['null_K'].shape[1]
    return dict(tail_p_200=float(es.exceed_probability(score, state['null_fit'])),
                tail_p_2000=float(es.exceed_probability(score, fit)), tail_samples=samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint fit and null tail check of the candidates')
    parser.add_argument('--c', default='spectral_classes.json', help='config file (default: spectral_classes.json)')
    parser.add_argument('--r', default='../results/Reliable/', help='results (default: ../results/Reliable/)')
    args = parser.parse_args()
    with open(args.c) as configfile:
        config = json.load(configfile)
    config['search']['n_stars'] = len(es.build_jobs(config))
    es.init_worker(config)

    best = pd.read_csv(os.path.join(args.r, 'candidates.csv'))
    out = []
    for _, row in best.iterrows():
        rows, state = es.search_star((row['folder'], row['class_name']), keep_state=True)
        i = int(np.where(state['df']['dp_id'].to_numpy() == row['dp_id'])[0][0])
        result = dict(folder=row['folder'], tier=row['tier'], dp_id=row['dp_id'], score=row['score'])
        result.update(joint_fit(state, i, row['v0'], row['fwhm']))
        result.update(tail_check(state, row['score'], np.random.default_rng(1)))
        out.append(result)
        print(row['folder'], 'ratio {:.2f} +- {:.2f}, delta chi2 {:.0f}, tail p {:.1e} (200) vs {:.1e} (2000)'.format(
            result['fit_ratio'], result['fit_ratio_err'], result['fit_delta_chi2'], result['tail_p_200'],
            result['tail_p_2000']), flush=True)
    pd.DataFrame(out).to_csv(os.path.join(args.r, 'vetting.csv'), index=False)
