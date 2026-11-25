'''
07/11/26
Population tests (idea 14). Noise and the flipped control should not care what kind of star they land on; real comets
should prefer young stars with debris discs and edge-on systems, and fall in (redshifted, as at bet Pic). So compare the
stars with detections against all searched stars of the same class, for the real run and for the flipped control:
- infrared excess, a sign of a debris disc: W3 - W4 > 0.3 mag in AllWISE (both bands measured, quality A or B), through
  the Gaia DR3 best neighbour. W3 - W4 is about 0 for a bare photosphere of these types, and W3 and W4 saturate only
  for the very brightest stars (W1 and W2 saturate near 8 mag, so W1 - W4 is not used)
- v sin i, median of the SIMBAD measurements: discs seen edge-on are the ones whose comets cross the star, and they
  spin fast in projection
- share of redshifted detections (falling in)
Fisher exact test for the shares, Mann-Whitney for v sin i, binomial for the redshift share against 1/2.
Stars without a Gaia DR3 id have no WISE match here and count as unknown, not as no excess.
'''
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, mannwhitneyu

from external import GAIA_TAP, SIMBAD_TAP, stars, tap

CHUNK = 400
EXCESS_MAG = 0.3


def wise(gaia_ids, cache):
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={'source_id': str})
    out = []
    for start in range(0, len(gaia_ids), CHUNK):
        ids = ','.join(gaia_ids[start:start + CHUNK])
        out.append(tap(GAIA_TAP, 'SELECT n.source_id, a.w3mpro, a.w4mpro, a.ph_qual FROM gaiadr3.allwise_best_neighbour AS n '
                                 'JOIN gaiadr1.allwise_original_valid AS a ON a.allwise_oid = n.allwise_oid '
                                 f'WHERE n.source_id IN ({ids})'))
    table = pd.concat(out).astype({'source_id': str})
    table.to_csv(cache, index=False)
    return table


def vsini(gaia_ids, cache):
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={'source_id': str})
    out = []
    for start in range(0, len(gaia_ids), CHUNK):
        names = ','.join("'Gaia DR3 {}'".format(g) for g in gaia_ids[start:start + CHUNK])
        out.append(tap(SIMBAD_TAP, 'SELECT i.id, r.vsini FROM mesRot AS r JOIN ident AS i ON i.oidref = r.oidref '
                                   f'WHERE i.id IN ({names})'))
    table = pd.concat(out)
    table['source_id'] = table['id'].str.replace('Gaia DR3 ', '', regex=False)
    table = table.groupby('source_id', as_index=False)['vsini'].median()
    table.to_csv(cache, index=False)
    return table


def compare(searched, detected, name):
    '''One row of tests: detected stars against the other searched stars of the same classes'''
    s = searched[searched['class_name'].isin(detected['class_name'].unique())]
    hit = s['folder'].isin(detected['folder'])
    known = s['excess'].notna()
    a = int((hit & known & (s['excess'] == True)).sum()); b = int((hit & known).sum()) - a
    c = int((~hit & known & (s['excess'] == True)).sum()); d = int((~hit & known).sum()) - c
    row = dict(run=name, detected=int(hit.sum()), excess_detected=f'{a}/{a + b}', excess_others=f'{c}/{c + d}',
               excess_p=fisher_exact([[a, b], [c, d]], alternative='greater')[1] if a + b and c + d else np.nan)
    v_hit, v_other = s.loc[hit, 'vsini'].dropna(), s.loc[~hit, 'vsini'].dropna()
    row.update(vsini_median_detected=v_hit.median(), vsini_median_others=v_other.median(),
               vsini_p=mannwhitneyu(v_hit, v_other, alternative='greater')[1] if len(v_hit) and len(v_other) else np.nan)
    red = int((detected['v0'] > 0).sum())
    row.update(redshifted=f'{red}/{len(detected)}',
               redshift_p=binomtest(red, len(detected), 0.5, alternative='greater').pvalue if len(detected) else np.nan)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Population tests of the detections')
    parser.add_argument('--c', default='spectral_classes.json', help='config file (default: spectral_classes.json)')
    parser.add_argument('--r', default='../results/Reliable/', help='real run (default: ../results/Reliable/)')
    parser.add_argument('--f', default='../results/Reliable_flip/', help='flipped run (default: ../results/Reliable_flip/)')
    args = parser.parse_args()
    with open(args.c) as configfile:
        config = json.load(configfile)

    scores = pd.read_pickle(os.path.join(args.r, 'spectrum_scores.pkl'))
    searched = scores.drop_duplicates('folder')[['folder', 'class_name']].merge(stars(config['search']['dataset']),
                                                                                 on='folder', how='left')
    gaia = sorted(searched['gaia_dr3'].dropna().unique())
    w = wise(gaia, os.path.join(args.r, 'allwise.csv'))
    good = w['ph_qual'].str[2:4].apply(lambda q: set(q) <= set('AB'))
    w['excess'] = np.where(good, (w['w3mpro'] - w['w4mpro']) > EXCESS_MAG, np.nan)
    searched = searched.merge(w[['source_id', 'excess']], left_on='gaia_dr3', right_on='source_id', how='left')
    searched = searched.merge(vsini(gaia, os.path.join(args.r, 'vsini.csv')), on='source_id', how='left')

    rows = []
    for name, folder in (('real', args.r), ('flipped', args.f)):
        best = pd.read_csv(os.path.join(folder, 'candidates.csv'))
        best = best[best['class_name'].isin(['A_earlyF', 'lateF_G_K'])]  # the classes that can make detections
        rows.append(compare(searched, best, name + ', all tiers'))
        top = best[best['tier'] == 'high confidence']
        if len(top):
            rows.append(compare(searched, top, name + ', high confidence'))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.r, 'population_tests.csv'), index=False)
    print('Stars with a WISE answer:', int(searched['excess'].notna().sum()), 'of', len(searched),
          '| with v sin i:', int(searched['vsini'].notna().sum()))
    print(table.to_string(index=False))
