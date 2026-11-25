'''
07/11/26
Small helpers to ask public archives (TAP services) and to find the stars of the search. Used by the follow-up scripts.
'''
import io
import os
import time

import pandas as pd
import requests

ESO_TAP = 'https://archive.eso.org/tap_obs/sync'
GAIA_TAP = 'https://gea.esac.esa.int/tap-server/tap/sync'
SIMBAD_TAP = 'https://simbad.harvard.edu/simbad/sim-tap/sync'  # the Strasbourg mirror does not answer from here


def tap(url, adql, tries=3):
    '''One synchronous ADQL query, as a table'''
    for attempt in range(tries):
        try:
            r = requests.post(url, timeout=300, data={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv',
                                                      'MAXREC': '2000000', 'QUERY': adql})
            r.raise_for_status()
            if r.text.lstrip().startswith('<'):
                raise ValueError(r.text[:500])
            return pd.read_csv(io.StringIO(r.text))
        except (requests.RequestException, ValueError):
            if attempt == tries - 1:
                raise
            time.sleep(10)


def stars(dataset):
    '''Folder, star id, Gaia DR3 id, position and spectral type of every star folder'''
    index = pd.read_csv(os.path.join(dataset, 'dataset_index.csv'), dtype={'gaia_dr3': str})
    groups = pd.read_pickle(os.path.join(dataset, '..', 'grouping', 'v2', 'harps_groups_v2.pkl'))
    return index.merge(groups[['star_id', 'ra', 'dec', 'sp_type', 'V']], on='star_id', how='left')
