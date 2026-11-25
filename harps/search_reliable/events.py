'''
21/11/26
Event analysis (addendum 12 of the pre-registration): every detection that passes the stage-2 tests, by the main
rule (type_weight) or the pair route, becomes an event (one star, one night, one velocity), measured spectrum by
spectrum and scored on a scorecard fixed before any event was looked at.

Per event:
- K and H fitted together in each spectrum of the night (ratio to the star's median of other nights, in the star
  frame): one Gaussian with a common velocity and width, a depth per line, a free local scale per line.
- Depth ratio K/H, apparent optical depth of H and covering fraction (doublet ratio, Savage & Sembach 1991:
  depth = fc (1 - exp(-tau)), tau_K = 2 tau_H, so K/H = 1 + exp(-tau_H)).
- Velocity relative to the star, change of velocity and depth through the night, duration, other nights at the same
  velocity, Earth velocity (BERV).
- Veto lines at the event velocity (H alpha, He I D3, Na D), with the telluric mask and scores of veto_check.py.
- Scorecard: required R1-R4, points P1-P7, level strong (6-7), probable (4-5), possible (0-3), or not supported.

    py -3 events.py --out ../results/FinalCheck2/
Writes <out>/events/events.csv, events_spectra.csv, figures/<event>.png
'''
import argparse
import json
import os
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

import exocomet_search as es
import veto_check as vc
from calibrate_by_type import add_berv_flag
from evaluate_rules import combined


C_KMS = 299792.458
FWHM_TO_SIGMA = 2.354820045
MAD_TO_SIGMA = 1.4826
RULES = ('type_weight', 'pairs')
RUNS = {'real': ('on', 'off'), 'flipped': ('on_flip', 'off_flip')}
MERGE_FWHM = 2  # detections of one night within this many FWHM are one event
MAX_REFERENCE = 2000  # spectra in the median reference (bet Pic has 9,000)
DETECT_SIGMA = 3  # a spectrum shows the event when its K depth is above this many errors
INSTRUMENT_FWHM = 2.6  # km/s, HARPS at R = 115,000
TRANSIENT_SHARE = 0.1  # P2: at most this share of the other nights show absorption at that velocity
VETO_SIGMA = 3
DATASET = 'C:/data/harps/dataset/'
GROUPS = 'C:/data/harps/grouping/v2/harps_groups_v2.pkl'
GROUPING_CACHE = 'C:/data/harps/grouping/v2/cache/'
BINARY_TYPES = {'SB*', 'EB*', 'El*', 'SB?', 'EB?'}
LITERATURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'literature_events.json')
# rest wavelengths in air, nm (NIST ASD)
VETO_LINES = {'Halpha': ('Halpha', 656.2808), 'HeID3': ('HeI5876_NaD', 587.562),
              'NaD2': ('HeI5876_NaD', 588.995), 'NaD1': ('HeI5876_NaD', 589.592)}


# ------------------------------------------------------------------------------------------------- events

def collect(out, config, class_counts):
    '''Detections that passed every test, real and flipped, by the main rule or the pair route'''
    rows = []
    for rule in RULES:
        for run, folders in RUNS.items():
            kept, best = combined(out, folders, config, rule, class_counts)
            if len(best) == 0:
                continue
            passed = kept[kept['pass_all'].astype(bool) & kept['folder'].isin(best['folder'])]
            rows.append(passed.assign(rule=rule, run=run))
    rows = pd.concat(rows, ignore_index=True)
    rows['night'] = np.floor(rows['t'] - 0.5)
    return rows


def merge(rows):
    '''One event per star, night and velocity; keeps the strongest detection and the routes that found it'''
    events = []
    for (run, folder, night), group in rows.groupby(['run', 'folder', 'night']):
        group = group.sort_values('v0')
        cluster = (group['v0'].diff() > MERGE_FWHM * group['fwhm'].shift().fillna(0)).cumsum()
        for _, part in group.groupby(cluster):
            top = part.sort_values('score', ascending=False).iloc[0]
            events.append(dict(top, rules=','.join(sorted(part['rule'].unique())),
                               routes=','.join(sorted(set(part['route'].astype(str)))),
                               n_detections=len(part.drop_duplicates('dp_id'))))
    events = pd.DataFrame(events)
    events['event'] = [f"{r}_{f}_{int(n)}_{v:+.0f}" for r, f, n, v in
                       zip(events['run'], events['folder'], events['night'], events['v0'])]
    return events.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- spectra

def load_star(folder):
    '''Normalised, smoothed K and H (H on the K velocity grid), as the search reads them'''
    path = es.Config['search']['dataset'] + folder + '/'
    K, H = es.load_line(es.Searches['K'], path), es.load_line(es.Searches['H'], path)
    if K is None or H is None:
        return None
    keep = np.intersect1d(K[1]['dp_id'], H[1]['dp_id'])
    inK, inH = K[1]['dp_id'].isin(keep).to_numpy(), H[1]['dp_id'].isin(keep).to_numpy()
    df = K[1][inK].reset_index(drop=True)
    sK = K[0][inK].astype(float)
    hH = H[0][inH].astype(float)
    order = pd.Series(np.arange(inH.sum()), index=H[1]['dp_id'][inH].to_numpy())[df['dp_id']].to_numpy()
    sH = np.array([np.interp(K[2], H[2], h) for h in hH[order]])
    return dict(df=df, v=K[2], K=sK, H=sH, times=df['t_min'].to_numpy())


def gauss(v, centre, sigma):
    return np.exp(-(v - centre) ** 2 / (2 * sigma ** 2))


def fit_doublet(v, qK, qH, eK, eH, v0, fwhm0):
    '''Joint fit of K and H: common velocity and width, depth per line, local scale per line. Returns dict'''
    half = max(3 * fwhm0, 40)
    sel = np.abs(v - v0) < half
    x, yK, yH = v[sel], qK[sel], qH[sel]

    def resid(p):
        aK, aH, c, s, cK, cH = p
        g = gauss(x, c, s)
        return np.concatenate([(yK - cK * (1 - aK * g)) / eK, (yH - cH * (1 - aH * g)) / eH])

    s0 = max(fwhm0, 3) / FWHM_TO_SIGMA
    lo = [-1, -1, v0 - max(fwhm0, 5), 0.5, 0.5, 0.5]
    hi = [1.5, 1.5, v0 + max(fwhm0, 5), 40 / FWHM_TO_SIGMA * 2, 1.5, 1.5]
    p0 = np.clip([0.1, 0.07, v0, s0, 1, 1], np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    fit = least_squares(resid, p0, bounds=(lo, hi))
    J = fit.jac
    try:
        cov = np.linalg.inv(J.T @ J) * max(1.0, np.sum(fit.fun ** 2) / max(len(fit.fun) - 6, 1))
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(6, np.nan)
    aK, aH, c, s, _, _ = fit.x
    return dict(depth_K=aK, depth_K_err=err[0], depth_H=aH, depth_H_err=err[1], v_fit=c, v_fit_err=err[2],
                fwhm_fit=s * FWHM_TO_SIGMA, fwhm_fit_err=err[3] * FWHM_TO_SIGMA,
                chi2_red=float(np.sum(fit.fun ** 2) / max(len(fit.fun) - 6, 1)))


def amplitude(v, q, e, centre, sigma):
    '''Depth of a Gaussian of fixed position and width, with a free level (linear least squares)'''
    sel = np.abs(v - centre) < max(4 * sigma, 20)
    g = gauss(v[sel], centre, sigma)
    A = np.vstack([np.ones_like(g), -g]).T / e
    coef, *_ = np.linalg.lstsq(A, q[sel] / e, rcond=None)
    cov = np.linalg.pinv(A.T @ A)
    level, b = coef
    return b / level, np.sqrt(cov[1, 1]) / abs(level)


def noise(v, q, v0, half):
    '''MAD scatter of a ratio spectrum away from the feature, within +-250 km/s'''
    sel = (np.abs(v - v0) > half) & (np.abs(v) < 250)
    r = q[sel]
    return MAD_TO_SIGMA * np.median(np.abs(r - np.median(r)))


def doublet_ratio(aK, aH, eK, eH):
    '''K/H depth ratio with its error; optical depth of H and covering fraction when 1 < ratio < 2'''
    ratio = aK / aH if aH != 0 else np.nan
    err = abs(ratio) * np.hypot(eK / aK, eH / aH) if aK and aH else np.nan
    tau_H = fc = np.nan
    if 1 < ratio < 2:
        tau_H = -np.log(ratio - 1)
        fc = aH / (1 - np.exp(-tau_H))
    return ratio, err, tau_H, fc


def measure(event, star, flip):
    '''All Ca II measurements of one event. Returns (event row, per-spectrum rows)'''
    v = star['v'] - event['v_star']
    sign = -1 if flip else 1
    K, H = star['K'], star['H']
    nights = np.floor(star['times'] - 0.5)
    tonight = np.flatnonzero(nights == event['night'])
    other = np.flatnonzero(nights != event['night'])
    rng = np.random.default_rng(0)
    ref_idx = rng.choice(other, min(MAX_REFERENCE, other.size), replace=False) if other.size else tonight
    refK, refH = np.median(K[ref_idx], axis=0), np.median(H[ref_idx], axis=0)
    qK = 1 + sign * (K / refK - 1)  # flipped run: the ratio upside down, as the search saw it
    qH = 1 + sign * (H / refH - 1)
    v0, fwhm0 = event['v0'], event['fwhm']
    half = max(3 * fwhm0, 40)

    stackK, stackH = qK[tonight].mean(axis=0), qH[tonight].mean(axis=0)
    eK, eH = noise(v, stackK, v0, half), noise(v, stackH, v0, half)
    stack = fit_doublet(v, stackK, stackH, eK, eH, v0, fwhm0)
    centre, sigma = stack['v_fit'], stack['fwhm_fit'] / FWHM_TO_SIGMA

    per = []
    for i in tonight:
        eKi, eHi = noise(v, qK[i], v0, half), noise(v, qH[i], v0, half)
        f = fit_doublet(v, qK[i], qH[i], eKi, eHi, centre, stack['fwhm_fit'])
        aK, aKe = amplitude(v, qK[i], eKi, centre, sigma)
        aH, aHe = amplitude(v, qH[i], eHi, centre, sigma)
        per.append(dict(event=event['event'], dp_id=star['df']['dp_id'].iloc[i], t=star['times'][i],
                        fixed_depth_K=aK, fixed_depth_K_err=aKe, fixed_depth_H=aH, fixed_depth_H_err=aHe,
                        detected=bool(aK / aKe > DETECT_SIGMA) if aKe > 0 else False, **f))
    per = pd.DataFrame(per).sort_values('t').reset_index(drop=True)

    # other nights at the same velocity and width
    other_hits, other_n = 0, 0
    for n in np.unique(nights[other]):
        members = np.flatnonzero(nights == n)
        sK = qK[members].mean(axis=0)
        aK, aKe = amplitude(v, sK, noise(v, sK, v0, half), centre, sigma)
        other_n += 1
        other_hits += int(aKe > 0 and aK / aKe > DETECT_SIGMA)

    det = per[per['detected']]
    run, longest = 0, 0
    for d in per['detected']:
        run = run + 1 if d else 0
        longest = max(longest, run)
    drift = drift_err = depth_p = np.nan
    if len(det) >= 3:
        hours = (det['t'] - det['t'].min()) * 24
        if hours.std() > 0:
            res = stats.linregress(hours, det['v_fit'])
            drift, drift_err = res.slope, res.stderr
        w = 1 / det['fixed_depth_K_err'] ** 2
        mean = np.sum(w * det['fixed_depth_K']) / np.sum(w)
        chi2 = np.sum(w * (det['fixed_depth_K'] - mean) ** 2)
        depth_p = float(stats.chi2.sf(chi2, len(det) - 1))
    duration = (det['t'].max() - det['t'].min()) * 24 if len(det) else 0.0
    ratio, ratio_err, tau_H, fc = doublet_ratio(stack['depth_K'], stack['depth_H'], stack['depth_K_err'],
                                                stack['depth_H_err'])
    out = dict(stack, n_night=len(tonight), n_detected=len(det), longest_run=longest,
               duration_h=duration, duration_open=bool(len(det) and (per['detected'].iloc[0] or per['detected'].iloc[-1])),
               drift_kms_per_h=drift, drift_err=drift_err, depth_change_p=depth_p,
               other_nights=other_n, other_nights_hit=other_hits,
               ratio_KH=ratio, ratio_KH_err=ratio_err, tau_H=tau_H, covering_fraction=fc,
               strongest_dp=per.loc[(per['fixed_depth_K'] / per['fixed_depth_K_err']).idxmax(), 'dp_id'])
    return out, per, dict(v=v, qK=qK, qH=qH, tonight=tonight, stackK=stackK, stackH=stackH)


# ------------------------------------------------------------------------------------------------- vetoes

def veto_lines(event, folder_files, mask, flip):
    '''Scores of H alpha, He I D3 and Na D at the event velocity in its strongest spectrum (positive = dip), in units
    of the scatter of the same score on other nights'''
    out = {f'{k}_sigma': np.nan for k in VETO_LINES}
    if event['strongest_dp'] not in set(folder_files['dp_id']):
        out['veto_data'] = False
        return out
    files = [vc.VETO_DIR + f for f in folder_files['file']]
    nights = np.floor(folder_files['t'].to_numpy() - 0.5)
    berv = folder_files['berv'].to_numpy()
    i = int(np.flatnonzero(folder_files['dp_id'].to_numpy() == event['strongest_dp'])[0])
    rng = np.random.default_rng(0)
    other = np.flatnonzero(nights != event['night'])
    ref_idx = rng.choice(other, min(vc.MAX_REFERENCE, other.size), replace=False)
    spread_idx = ref_idx[:vc.SPREAD_SPECTRA]
    sign = -1 if flip else 1
    cache = {}
    for name, (band, rest) in VETO_LINES.items():
        if band not in cache:
            grid, flux = vc.load_band(files, band)
            good = np.ones((len(files), grid.size), bool)
            if band in mask:
                for k, b in enumerate(berv):
                    good[k] = np.interp(grid / (1 + b / C_KMS), mask[band][0], mask[band][1]) > 1 - vc.TELLURIC_DEPTH
            cache[band] = (grid, flux, good)
        grid, flux, good = cache[band]
        at = [rest * (1 + (event['v_star'] + event['v_fit']) / C_KMS)]
        out[f'{name}_sigma'] = sign * vc.dip_sigma(grid, flux, good[i], i, ref_idx, spread_idx, at,
                                                   max(event['fwhm_fit'], INSTRUMENT_FWHM))
    out['veto_data'] = True
    return out


def binaries(labels):
    '''Main SIMBAD object type of each star, from the grouping cache (SIMBAD does not answer from here): True for
    spectroscopic, eclipsing or ellipsoidal binaries. Only the main type: a binary listed under another main type
    (e.g. TY CrA, Ae*) is not caught'''
    cache = pd.concat([pd.read_csv(GROUPING_CACHE + name, dtype=str, usecols=['main_id', 'otype'])
                       for name in ('cone_candidates.csv', 'resolved.csv')]).drop_duplicates('main_id')
    otype = dict(zip(cache['main_id'], cache['otype']))
    return {label: (otype[label] in BINARY_TYPES) if label in otype else np.nan for label in labels}


# ------------------------------------------------------------------------------------------------ scorecard

def literature():
    if not os.path.exists(LITERATURE):
        return None
    return json.load(open(LITERATURE))['ranges']


def inside(value, entry):
    if entry is None or entry.get('min') is None or entry.get('max') is None or not np.isfinite(value):
        return None
    return entry['min'] <= value <= entry['max']


def scorecard(e, ranges):
    req = dict(
        R1=bool(np.isfinite(e['ratio_KH']) and e['ratio_KH'] + 2 * e['ratio_KH_err'] >= 1
                and e['ratio_KH'] - 2 * e['ratio_KH_err'] <= 2),
        R2=bool(not (abs(e.get('Halpha_sigma', np.nan)) > VETO_SIGMA or abs(e.get('HeID3_sigma', np.nan)) > VETO_SIGMA)),
        R3=bool(not e['near_berv']),
        R4=bool(e['fwhm_fit'] >= INSTRUMENT_FWHM))
    p5 = None
    if ranges:
        checks = [inside(abs(e['v_fit']), ranges.get('velocity_kms_abs') or None),
                  inside(e['depth_K'], ranges.get('depth_K')), inside(e['fwhm_fit'], ranges.get('fwhm_kms'))]
        if ranges.get('velocity_kms') and ranges['velocity_kms'].get('min') is not None:
            checks[0] = inside(e['v_fit'], ranges['velocity_kms'])
        checks = [c for c in checks if c is not None]
        p5 = bool(checks) and all(checks)
    points = dict(
        P1=bool(e['v_fit'] > 0),
        P2=bool(e['other_nights'] > 0 and e['other_nights_hit'] / e['other_nights'] <= TRANSIENT_SHARE),
        P3=bool(e['longest_run'] >= 2),
        P4=bool((np.isfinite(e['drift_err']) and e['drift_err'] > 0 and abs(e['drift_kms_per_h']) > 3 * e['drift_err'])
                or (np.isfinite(e['depth_change_p']) and e['depth_change_p'] < 0.01)),
        P5=bool(p5) if p5 is not None else False,
        P6=bool(e['binary'] is False or e['binary'] == 0),
        P7=bool(e.get('NaD2_sigma', np.nan) > VETO_SIGMA and e.get('NaD1_sigma', np.nan) > VETO_SIGMA))
    score = sum(points.values())
    level = ('not supported' if not all(req.values()) else 'strong' if score >= 6 else 'probable' if score >= 4
             else 'possible')
    failed = [k for k, ok in req.items() if not ok]
    return dict(**req, **points, points=score, level=level, failed_required=','.join(failed),
                literature_checked=p5 is not None)


# ------------------------------------------------------------------------------------------------- figure

def figure(e, view, per, path):
    v, tonight = view['v'], view['tonight']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [2, 1]})
    for ax, line, q, stack in ((axes[0, 0], 'K', view['qK'], view['stackK']), (axes[1, 0], 'H', view['qH'], view['stackH'])):
        for i in tonight:
            ax.plot(v, q[i], color='0.75', lw=0.6)
        ax.plot(v, stack, color='tab:blue' if line == 'K' else 'tab:red', lw=1.2, label=f'Ca II {line}, night mean')
        ax.axvline(e['v_fit'], color='g', ls='--', lw=1)
        ax.axhline(1, color='k', lw=0.5)
        ax.set_xlim(-250, 250)
        lo = np.nanpercentile(stack[np.abs(v) < 250], 0.5)
        ax.set_ylim(min(lo - 0.02, 0.9), 1.1)
        ax.set_ylabel('spectrum / median of other nights')
        ax.legend(fontsize=8, loc='lower left')
    axes[1, 0].set_xlabel('velocity relative to the star (km/s)')
    hours = (per['t'] - per['t'].min()) * 24
    axes[0, 1].errorbar(hours, per['fixed_depth_K'], per['fixed_depth_K_err'], fmt='o', ms=3, color='tab:blue', label='K')
    axes[0, 1].errorbar(hours, per['fixed_depth_H'], per['fixed_depth_H_err'], fmt='s', ms=3, color='tab:red', label='H')
    axes[0, 1].axhline(0, color='k', lw=0.5)
    axes[0, 1].set_ylabel('depth at the event velocity')
    axes[0, 1].legend(fontsize=8)
    d = per[per['detected']]
    axes[1, 1].errorbar((d['t'] - per['t'].min()) * 24, d['v_fit'], d['v_fit_err'], fmt='o', ms=3, color='g')
    axes[1, 1].set_ylabel('fitted velocity (km/s)')
    axes[1, 1].set_xlabel('hours into the night')
    fig.suptitle(f"{e['star_label']} ({e['class_name']}), night of MJD {int(e['night'])}: {e['level']} "
                 f"({e['points']} points), v = {e['v_fit']:+.1f} km/s, depth K {e['depth_K']:.3f}, "
                 f"K/H {e['ratio_KH']:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------- main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Event analysis of the detections that passed the tests')
    parser.add_argument('--out', default='../results/FinalCheck2/')
    parser.add_argument('--c', default=None, help='settings file (default: config.json in --out)')
    args = parser.parse_args()
    config = json.load(open(args.c or os.path.join(args.out, 'config.json')))
    config['tests']['shape_mode'] = 'weighted'
    class_counts = json.load(open(os.path.join(args.out, 'class_counts.json')))
    es.init_worker(config)
    dest = os.path.join(args.out, 'events')
    os.makedirs(os.path.join(dest, 'figures'), exist_ok=True)

    events = merge(collect(args.out, config, class_counts))
    events = add_berv_flag(events)
    groups = pd.read_pickle(GROUPS)
    index = pd.read_csv(DATASET + 'dataset_index.csv', dtype={'gaia_dr3': str})
    info = index.merge(groups[['star_id', 'sp_type']], on='star_id', how='left')[['folder', 'star_label', 'sp_type']]
    events = events.merge(info, on='folder', how='left')
    print(f'{len(events)} events in {events["folder"].nunique()} stars '
          f'({(events["run"] == "real").sum()} real, {(events["run"] == "flipped").sum()} flipped)', flush=True)
    binary = binaries(events['star_label'].dropna().unique())
    events['binary'] = events['star_label'].map(binary)

    db = sqlite3.connect(vc.VETO_DIR + 'veto_state.sqlite')
    scores = pd.read_pickle(os.path.join(args.out, 'on', 'spectrum_scores.pkl'))[['folder', 'dp_id', 't']]
    mask = vc.telluric_mask(db, scores)
    ranges = literature()
    rows, spectra = [], []
    for folder, group in events.groupby('folder'):
        star = load_star(folder)
        if star is None:
            continue
        files = vc.star_files(db, scores, folder)
        for _, e in group.iterrows():
            flip = e['run'] == 'flipped'
            m, per, view = measure(e, star, flip)
            e = pd.Series({**e.to_dict(), **m})
            e = pd.Series({**e.to_dict(), **veto_lines(e, files, mask, flip)})
            e = pd.Series({**e.to_dict(), **scorecard(e, ranges)})
            rows.append(e)
            spectra.append(per)
            figure(e, view, per, os.path.join(dest, 'figures', f"{e['event']}.png"))
            print(f"{e['run']:8s} {str(e['star_label']):22s} night {int(e['night'])} v {e['v_fit']:+7.1f} "
                  f"depth K {e['depth_K']:.3f} K/H {e['ratio_KH']:.2f} det {e['n_detected']}/{e['n_night']} "
                  f"-> {e['level']} ({e['points']}) {e['failed_required']}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(dest, 'events.csv'), index=False)
    pd.concat(spectra).to_csv(os.path.join(dest, 'events_spectra.csv'), index=False)
    print(out.groupby(['run', 'level']).size().to_string())
