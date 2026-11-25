'''
21/11/26
Distribution statistics of exocomet events (addendum 13 of the pre-registration). Descriptive only: nothing here keeps
or drops an event.

Two reference samples:
- literature: published HARPS events, one row per event (literature_events_table.csv)
- our bet Pic: every bet Pic event measured by events.py, the same way as every other event
For velocity, depth K, FWHM, K/H ratio and duration: mean, median, standard deviation and MAD-based spread of each
reference; every search event as a z-score and a robust z (from median and MAD) against both; and a two-sample KS test
between the references (how much the measurement method itself moves the distribution).

    py -3 event_stats.py --out ../results/FinalCheck2/
Writes <out>/events/reference_stats.csv, event_zscores.csv, reference_ks.csv, figures/distributions.png
'''
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
LITERATURE = os.path.join(HERE, 'literature_events_table.csv')
BET_PIC = 'GaiaDR3_4792774797545800832'
MAD_TO_SIGMA = 1.4826
# quantity: (column in events.csv, column in the literature table, label)
QUANTITIES = {'velocity': ('v_fit', 'velocity_kms', 'velocity relative to the star (km/s)'),
              'depth_K': ('depth_K', 'depth_K', 'depth of Ca II K (fraction of stellar flux)'),
              'fwhm': ('fwhm_fit', 'fwhm_kms', 'FWHM (km/s)'),
              'ratio_KH': ('ratio_KH', 'ratio_K_over_H', 'depth K / depth H'),
              'duration_h': ('duration_h', 'duration_hours', 'duration (hours)')}


def describe(values):
    v = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy()
    if len(v) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, mad_sigma=np.nan, q16=np.nan, q84=np.nan)
    return dict(n=len(v), mean=v.mean(), median=np.median(v), std=v.std(ddof=1) if len(v) > 1 else np.nan,
                mad_sigma=MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))),
                q16=np.percentile(v, 16), q84=np.percentile(v, 84))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distribution statistics of exocomet events')
    parser.add_argument('--out', default='../results/FinalCheck2/')
    args = parser.parse_args()
    dest = os.path.join(args.out, 'events')
    events = pd.read_csv(os.path.join(dest, 'events.csv'))
    real = events[events['run'] == 'real']
    reference = os.path.join(dest, 'bet_pic_reference.csv')  # one feature per bet Pic night (bet_pic_reference.py)
    ours = pd.read_csv(reference) if os.path.exists(reference) else real[real['folder'] == BET_PIC]
    if 'kept' in ours:
        ours = ours[ours['kept']]
    lit = pd.read_csv(LITERATURE) if os.path.exists(LITERATURE) else pd.DataFrame()
    print(f'{len(real)} real events, {len(ours)} of them bet Pic; {len(lit)} published events')

    refs = {'literature': {q: lit[c[1]] if c[1] in lit else [] for q, c in QUANTITIES.items()},
            'our bet Pic': {q: ours[c[0]] for q, c in QUANTITIES.items()}}
    table = []
    for name, ref in refs.items():
        for q in QUANTITIES:
            table.append(dict(reference=name, quantity=q, **describe(ref[q])))
    table = pd.DataFrame(table)
    table.to_csv(os.path.join(dest, 'reference_stats.csv'), index=False)
    print(table.round(3).to_string(index=False))

    ks = []
    for q in QUANTITIES:
        a = pd.to_numeric(pd.Series(refs['literature'][q]), errors='coerce').dropna()
        b = pd.to_numeric(pd.Series(refs['our bet Pic'][q]), errors='coerce').dropna()
        if len(a) >= 3 and len(b) >= 3:
            res = stats.ks_2samp(a, b)
            ks.append(dict(quantity=q, n_literature=len(a), n_ours=len(b), ks_statistic=res.statistic, p_value=res.pvalue))
    ks = pd.DataFrame(ks)
    ks.to_csv(os.path.join(dest, 'reference_ks.csv'), index=False)
    if len(ks):
        print(ks.round(4).to_string(index=False))

    rows = []
    for _, e in events.iterrows():
        row = dict(event=e['event'], run=e['run'], star=e.get('star_label'), level=e.get('level'), points=e.get('points'))
        for q, (col, _, _) in QUANTITIES.items():
            value = e.get(col, np.nan)
            row[q] = value
            for name in refs:
                s = table[(table['reference'] == name) & (table['quantity'] == q)].iloc[0]
                row[f'z_{q}_{name.replace(" ", "_")}'] = (value - s['mean']) / s['std'] if s['std'] > 0 else np.nan
                row[f'rz_{q}_{name.replace(" ", "_")}'] = ((value - s['median']) / s['mad_sigma'] if s['mad_sigma'] > 0
                                                           else np.nan)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(dest, 'event_zscores.csv'), index=False)

    others = real[real['folder'] != BET_PIC]
    fig, axes = plt.subplots(1, len(QUANTITIES), figsize=(4.2 * len(QUANTITIES), 3.8))
    for ax, (q, (col, _, label)) in zip(axes, QUANTITIES.items()):
        lit_v = pd.to_numeric(pd.Series(refs['literature'][q]), errors='coerce').dropna()
        our_v = pd.to_numeric(ours[col], errors='coerce').dropna()
        values = np.concatenate([lit_v, our_v, pd.to_numeric(others[col], errors='coerce').dropna()])
        bins = np.histogram_bin_edges(values, bins=20) if len(values) else 10
        if len(our_v):
            ax.hist(our_v, bins=bins, color='#2f5fb3', alpha=0.6, label=f'our bet Pic ({len(our_v)})')
        if len(lit_v):
            ax.hist(lit_v, bins=bins, histtype='step', color='#c8741f', lw=2, label=f'published ({len(lit_v)})')
        for v in pd.to_numeric(others[col], errors='coerce').dropna():
            ax.axvline(v, color='k', lw=1, ls='--')
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('events')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Event properties: our bet Pic events, published events, and the other search events (dashed)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(dest, 'figures', 'distributions.png'), dpi=130)
    print('done')
