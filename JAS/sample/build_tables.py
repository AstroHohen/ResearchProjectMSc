'''
19/11/26
Builds the sample tables in this folder (stars.csv, spectra.csv.gz, summary.md, figures/) from the local HARPS products.
It needs the full local data (C:/data/harps), so it only runs on the machine that holds them; the tables it writes are
self-contained.
    py -3 build_tables.py
'''
import json
import os
import re
import sqlite3
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = 'C:/data/harps/'
SEARCH = os.path.join(REPO, 'harps', 'search_reliable')
SEARCHED = os.path.join(REPO, 'harps', 'results', 'FinalCheck2', 'on', 'spectrum_scores.pkl')
UPGRADE_MJD = 57172  # 2015-05-29, HARPS fibre upgrade (config/harps.json)
CLASS_ORDER = ['OB', 'A_earlyF', 'lateF_G_K', 'M', 'other or unknown']
LETTERS = list('OBAFGKM')

sys.path.insert(0, SEARCH)
from exocomet_search import class_of  # the spectral class rule of the search


def headers():
    '''Selected header values of every downloaded spectrum'''
    db = sqlite3.connect(DATA + 'state.sqlite')
    rows = db.execute("SELECT dp_id, header_json FROM spectra WHERE status='done'").fetchall()
    keys = {'DATE-OBS': 'date_obs', 'EXPTIME': 'exptime_s', 'SPEC_RES': 'spec_res', 'HIERARCH ESO DRS BERV': 'berv_kms',
            'HIERARCH ESO DRS VERSION': 'drs_version'}
    out = []
    for dp_id, text in rows:
        h = json.loads(text)
        out.append(dict(dp_id=dp_id, **{new: h.get(old) for old, new in keys.items()}))
    out = pd.DataFrame(out)
    out['mode'] = out['spec_res'].map({115000: 'HAM', 80000: 'EGGS'})
    return out


def markdown(frame):
    '''A DataFrame as a Markdown table (no tabulate needed)'''
    frame = frame.reset_index()
    head = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    rule = '|' + '---|' * len(frame.columns)
    body = ['| ' + ' | '.join(f'{v:,}' if isinstance(v, (int, np.integer)) else str(v) for v in row) + ' |'
            for row in frame.itertuples(index=False)]
    return '\n'.join([head, rule] + body)


def letter(sp_type):
    m = re.match(r'\s*(?:sd|d|esd|usd)?\s*([OBAFGKM])', sp_type) if isinstance(sp_type, str) else None
    return m.group(1) if m else 'other or unknown'


def luminosity(sp_type):
    '''Luminosity class from the MK type (V, IV, III, II, I), or blank'''
    if not isinstance(sp_type, str):
        return ''
    m = re.search(r'(?<![A-Za-z])(Ia|Ib|III|II|IV|V|I)(?![a-z])', sp_type.replace('IV-V', 'IV').replace('III-IV', 'III'))
    return m.group(1) if m else ''


if __name__ == '__main__':
    config = json.load(open(os.path.join(SEARCH, 'spectral_classes.json')))
    groups = pd.read_pickle(DATA + 'grouping/v2/harps_groups_v2.pkl')
    meta = pd.read_pickle(DATA + 'grouping/v2/harps_full_metadata_v2.pkl')
    index = pd.read_csv(DATA + 'dataset/dataset_index.csv', dtype={'gaia_dr3': str})
    bad = set(pd.read_csv(DATA + 'dataset/bad_grid.csv')['dp_id'])
    searched = set(pd.read_pickle(SEARCHED)['folder'].unique())
    rv = pd.read_csv(os.path.join(SEARCH, 'catalogue_rv.csv'))

    # per spectrum
    spectra = meta[['dp_id', 'star_id', 'target_name', 's_ra', 's_dec', 't_min', 'snr', 'proposal_id', 'method',
                    'sep_arcsec', 'epoch_fibre', 'flags']].merge(headers(), on='dp_id', how='left')
    spectra = spectra.rename(columns={'t_min': 'mjd', 's_ra': 'ra_pointing', 's_dec': 'dec_pointing',
                                      'method': 'match_method', 'flags': 'grouping_flags'})
    spectra['bad_grid'] = spectra['dp_id'].isin(bad)
    spectra['night'] = np.floor(spectra['mjd'] - 0.5).astype('Int64')  # one La Silla night per value
    spectra['upgrade_2015_window'] = spectra['mjd'].between(57162, 57177, inclusive='left')

    # per star
    good = spectra[spectra['star_id'].notna() & ~spectra['bad_grid']]
    per = good.groupby('star_id').agg(
        n_spectra=('dp_id', 'size'), n_nights=('night', 'nunique'), first_mjd=('mjd', 'min'), last_mjd=('mjd', 'max'),
        median_snr=('snr', 'median'), n_pre_upgrade=('mjd', lambda t: int((t < UPGRADE_MJD).sum())),
        n_post_upgrade=('mjd', lambda t: int((t >= UPGRADE_MJD).sum())),
        n_eggs=('mode', lambda m: int((m == 'EGGS').sum())), n_programmes=('proposal_id', 'nunique'))
    per['span_years'] = (per['last_mjd'] - per['first_mjd']) / 365.25
    stars = groups.rename(columns={'Nb Obs': 'n_spectra_grouped', 'method': 'match_method', 'V': 'vmag',
                                   'plx': 'parallax_mas', 'flags': 'grouping_flags'})
    stars = stars.drop(columns=['New Groups', 'raw_names', 'n_raw_names'])
    stars = stars.merge(per, left_on='star_id', right_index=True, how='left')
    stars['distance_pc'] = np.where(stars['parallax_mas'] > 0, 1000 / stars['parallax_mas'], np.nan)
    stars['type_letter'] = stars['sp_type'].map(letter)
    stars['luminosity_class'] = stars['sp_type'].map(luminosity)
    stars['search_class'] = [class_of(s, config['classes']) if isinstance(s, str) else None for s in stars['sp_type']]
    stars['search_class'] = stars['search_class'].fillna('other or unknown')
    stars = stars.merge(index[['star_id', 'folder', 'n_spectra']].rename(columns={'n_spectra': 'n_dataset'}), on='star_id',
                        how='left')
    stars['in_final_sample'] = stars['n_dataset'].fillna(0) >= 15  # final dataset: >= 15 spectra, bad grid removed
    stars['search_eligible'] = stars['in_final_sample'] & (stars['search_class'] != 'other or unknown')
    stars['searched'] = stars['folder'].isin(searched)  # scored by the search (enough spectra after its noise cut)
    stars = stars.merge(rv[['folder', 'rv_kms', 'rv_quality']].rename(columns={'rv_kms': 'simbad_rv_kms',
                                                                             'rv_quality': 'simbad_rv_quality'}),
                        on='folder', how='left')
    stars['group_kind'] = stars['star_id'].astype(str).str.split(':').str[0].map({'nonstellar': 'nonstellar', 'unresolved': 'unresolved name'}).fillna('star')
    order = ['star_id', 'star_label', 'gaia_dr3', 'folder', 'sp_type', 'type_letter', 'luminosity_class',
             'search_class', 'vmag', 'ra', 'dec', 'pmra', 'pmdec', 'parallax_mas', 'distance_pc', 'simbad_rv_kms',
             'simbad_rv_quality', 'n_spectra_grouped', 'n_spectra', 'n_nights', 'first_mjd', 'last_mjd', 'span_years',
             'median_snr', 'n_pre_upgrade', 'n_post_upgrade', 'n_eggs', 'n_programmes', 'match_method', 'grouping_flags',
             'group_kind', 'in_final_sample', 'search_eligible', 'searched']
    stars = stars[order].sort_values('n_spectra_grouped', ascending=False)
    stars.to_csv(os.path.join(HERE, 'stars.csv'), index=False)
    spectra = spectra.merge(stars[['star_id', 'search_class', 'in_final_sample', 'search_eligible', 'searched']], on='star_id', how='left')
    spectra.to_csv(os.path.join(HERE, 'spectra.csv.gz'), index=False)
    print('stars', stars.shape, 'spectra', spectra.shape)

    # summary tables
    final = stars[stars['in_final_sample']]
    lines = ['# Sample summary', '', 'Generated by build_tables.py on 19 Nov 2026. Counts are stars unless stated.', '']
    def table(series, title):
        t = pd.DataFrame({'all grouped': stars[series].value_counts(), 'final sample (>=15 spectra)':
                          final[series].value_counts(), 'search eligible': stars[stars['search_eligible']][series].value_counts(),
                          'searched': stars[stars['searched']][series].value_counts()})
        t = t.fillna(0).astype(int)
        lines.extend([f'## {title}', '', markdown(t), ''])
        return t
    t1 = table('type_letter', 'Spectral type letter (SIMBAD)')
    t2 = table('search_class', 'Search class (bundle)')
    t3 = table('luminosity_class', 'Luminosity class (blank = not given)')
    lines += ['## Totals', '',
              f'- Grouped groups: {len(stars):,}; stars {(stars["group_kind"] == "star").sum():,}, with a Gaia DR3 id {(stars["gaia_dr3"].fillna("").astype(str).str.strip() != "").sum():,}',
              f'- Spectra in the table: {len(spectra):,} (bad grid {spectra["bad_grid"].sum()}, EGGS '
              f'{(spectra["mode"] == "EGGS").sum():,})',
              f'- Final sample: {len(final):,} stars, {int(final["n_spectra"].sum()):,} spectra; median spectra per '
              f'star {final["n_spectra"].median():.0f}, max {int(final["n_spectra"].max()):,}',
              f'- Search eligible (O-M type, final sample): {stars["search_eligible"].sum():,} stars; scored by the search: '
              f'{stars["searched"].sum():,} stars', '']
    q = final[['n_spectra', 'n_nights', 'span_years', 'median_snr', 'vmag', 'distance_pc']].describe(
        percentiles=[.1, .5, .9]).T
    lines += ['## Final sample, per star', '', markdown(q.round(1)), '']
    open(os.path.join(HERE, 'summary.md'), 'w', encoding='utf-8').write('\n'.join(lines))

    # figures
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3})
    blue, grey = '#2f5fb3', '#b8c0cc'
    fig, ax = plt.subplots(figsize=(7, 4))
    letters = LETTERS + ['other or unknown']
    x = np.arange(len(letters))
    ax.bar(x - 0.2, t1['all grouped'].reindex(letters, fill_value=0), 0.4, color=grey, label='all grouped')
    ax.bar(x + 0.2, t1['final sample (>=15 spectra)'].reindex(letters, fill_value=0), 0.4, color=blue,
           label='final sample (>=15 spectra)')
    ax.set_xticks(x, letters[:-1] + ['other /\nunknown'])
    ax.set_ylabel('stars')
    ax.set_title('HARPS stars by spectral type')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'figures', 'spectral_types.png'), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(final['n_spectra'], bins=np.logspace(np.log10(15), np.log10(final['n_spectra'].max()), 40), color=blue)
    ax.set_xscale('log')
    ax.set_xlabel('spectra per star')
    ax.set_ylabel('stars')
    ax.set_title('Spectra per star, final sample')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'figures', 'spectra_per_star.png'), dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(spectra.loc[spectra['in_final_sample'] == True, 'snr'].clip(upper=400), bins=80, color=blue)
    axes[0].set_xlabel('SNR (archive value, clipped at 400)')
    axes[0].set_ylabel('spectra')
    axes[0].set_title('Signal to noise per spectrum')
    axes[1].hist(final['span_years'], bins=46, color=blue)
    axes[1].set_xlabel('years between first and last spectrum')
    axes[1].set_ylabel('stars')
    axes[1].set_title('Time span per star')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'figures', 'snr_and_span.png'), dpi=150)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    colours = {'OB': '#2f5fb3', 'A_earlyF': '#3b9e8f', 'lateF_G_K': '#d08c2c', 'M': '#b8473a',
               'other or unknown': '#b8c0cc'}
    for name in CLASS_ORDER[::-1]:
        part = final[final['search_class'] == name]
        ax.scatter(part['ra'], part['dec'], s=6, color=colours[name], label=f'{name} ({len(part)})')
    ax.set_xlim(360, 0)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_title('Final sample on the sky')
    ax.legend(frameon=False, fontsize=8, markerscale=2, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'figures', 'sky.png'), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    years = 1858.877 + spectra.loc[spectra['in_final_sample'] == True, 'mjd'] / 365.25
    ax.hist(years, bins=np.arange(2003, 2027), color=blue)
    ax.axvline(2015.41, color='k', ls='--', lw=1, label='fibre upgrade, May 2015')
    ax.set_xlabel('year')
    ax.set_ylabel('spectra')
    ax.set_title('Spectra per year, final sample')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'figures', 'spectra_per_year.png'), dpi=150)
    print('done')
