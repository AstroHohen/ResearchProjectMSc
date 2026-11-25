'''
07/11/26
Other instruments on the same night (idea 13). A comet lasts hours, so only a spectrum taken on the same night by
another spectrograph can confirm it independently. Lists, for each candidate star, the UVES, FEROS, ESPRESSO and
X-shooter spectra in the ESO archive within 10 arcsec and 0.5 day of each detection. It only lists them.
'''
import argparse
import json
import os

import pandas as pd

from external import ESO_TAP, stars, tap

INSTRUMENTS = ('UVES', 'FEROS', 'ESPRESSO', 'XSHOOTER')
RADIUS_ARCSEC = 10
WINDOW_DAYS = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectra of other instruments on the nights of the detections')
    parser.add_argument('--c', default='spectral_classes.json', help='config file (default: spectral_classes.json)')
    parser.add_argument('--r', default='../results/Reliable/', help='results (default: ../results/Reliable/)')
    args = parser.parse_args()
    with open(args.c) as configfile:
        config = json.load(configfile)

    best = pd.read_csv(os.path.join(args.r, 'candidates.csv'))
    kept = pd.read_pickle(os.path.join(args.r, 'scored_detections.pkl'))
    kept = kept[kept['pass_all'] & kept['folder'].isin(best['folder'])]
    where = stars(config['search']['dataset']).set_index('folder')
    names = ', '.join(f"'{i}'" for i in INSTRUMENTS)
    found = []
    for folder, rows in kept.groupby('folder'):
        ra, dec = where.loc[folder, ['ra', 'dec']]
        nights = sorted(set(rows['t'].round(3)))
        times = ' OR '.join(f'(t_min BETWEEN {t - WINDOW_DAYS} AND {t + WINDOW_DAYS})' for t in nights[:50])
        table = tap(ESO_TAP, f"SELECT instrument_name, dp_id, t_min, target_name FROM ivoa.ObsCore "
                             f"WHERE instrument_name IN ({names}) AND dataproduct_type = 'spectrum' "
                             f"AND CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {RADIUS_ARCSEC / 3600})) = 1 "
                             f"AND ({times})")
        if len(table):
            table.insert(0, 'folder', folder)
            table.insert(1, 'star', where.loc[folder, 'star_label'])
            found.append(table)
        print(where.loc[folder, 'star_label'], ':', len(table), 'spectra of other instruments on', len(nights), 'detections')
    out = pd.concat(found) if found else pd.DataFrame(columns=['folder', 'star', 'instrument_name', 'dp_id', 't_min'])
    out.to_csv(os.path.join(args.r, 'other_instruments.csv'), index=False)
    print('Total:', len(out))
