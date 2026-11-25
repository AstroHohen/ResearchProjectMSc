'''
14/11/26
Plot of one candidate for vetting by eye: Ca II K and H in the frame of the star, the detection spectrum and the
others of its night against the median of all spectra, and the residuals of both lines. Read only.
    py -3 plot_candidate.py --folder GaiaDR3_5338661808024777856 --dp ADP.2014-09-23T11:02:18.743 --v0 161.7 --vstar -72.47
'''
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import exocomet_search as es

C_KMS = 299792.458
MAX_GREY = 300
H_EPSILON_KMS = (397.007 - 396.847) / 396.847 * C_KMS  # H epsilon relative to Ca II H


def load(folder, config):
    es.init_worker(config)
    path = config['search']['dataset'] + folder + '/'
    out = {}
    for line in 'KH':
        spectra, df, v = es.load_line(es.Searches[line], path)
        out[line] = (spectra.astype(float), df.reset_index(drop=True), v)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a candidate for vetting by eye')
    parser.add_argument('--folder', required=True)
    parser.add_argument('--dp', required=True, help='dp_id of the detection spectrum')
    parser.add_argument('--v0', type=float, required=True, help='feature velocity relative to the star (km/s)')
    parser.add_argument('--vstar', type=float, required=True, help='star velocity used by the search (km/s)')
    parser.add_argument('--name', default='')
    parser.add_argument('--c', default='spectral_classes.json')
    parser.add_argument('--out', default='../results/FinalCheck2/figures/')
    parser.add_argument('--span', type=float, default=300, help='half width of the plot in km/s')
    args = parser.parse_args()

    with open(args.c) as handle:
        config = json.load(handle)
    data = load(args.folder, config)
    os.makedirs(args.out, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [3, 3, 2]})
    colours = {'K': 'tab:blue', 'H': 'tab:red'}
    for ax, line in zip(axes[:2], 'KH'):
        spectra, df, v = data[line]
        rel = v - args.vstar
        median = np.median(spectra, axis=0)
        i = int(np.where(df['dp_id'] == args.dp)[0][0])
        night = np.floor(df['t_min'] - 0.5) == np.floor(df['t_min'][i] - 0.5)
        shown = np.random.default_rng(0).permutation(len(spectra))[:MAX_GREY]  # a sample for stars with many spectra
        for s in spectra[shown]:
            ax.plot(rel, s, color='0.85', lw=0.4, zorder=1)
        for j in np.where(night)[0]:
            if j != i:
                ax.plot(rel, spectra[j], color='orange', lw=0.7, zorder=2)
        ax.plot(rel, median, color='k', lw=1.2, label='median of all {} spectra'.format(len(spectra)), zorder=3)
        ax.plot(rel, spectra[i], color=colours[line], lw=1.0, label='detection spectrum', zorder=4)
        ax.plot([], [], color='orange', label='other spectra of the night ({})'.format(int(night.sum()) - 1))
        ax.axvline(args.v0, color='g', ls='--', lw=1, label='feature, {:+.0f} km/s'.format(args.v0))
        if line == 'H':
            ax.axvline(H_EPSILON_KMS, color='purple', ls=':', lw=1, label='H epsilon')
        ax.set_ylabel(f'Ca II {line}, normalised flux')
        ax.legend(fontsize=8, loc='lower left')
    for line in 'KH':
        spectra, df, v = data[line]
        i = int(np.where(df['dp_id'] == args.dp)[0][0])
        axes[2].plot(v - args.vstar, spectra[i] - np.median(spectra, axis=0), color=colours[line], lw=0.8, label=line)
    axes[2].axvline(args.v0, color='g', ls='--', lw=1)
    axes[2].axvline(H_EPSILON_KMS, color='purple', ls=':', lw=1)
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_ylabel('detection - median')
    axes[2].set_xlabel('velocity relative to the star (km/s)')
    axes[2].legend(fontsize=8)
    axes[2].set_xlim(-args.span, args.span)
    _, df, _ = data['K']
    t = float(df.loc[df['dp_id'] == args.dp, 't_min'].iloc[0])
    fig.suptitle('{} ({}), detection {} (MJD {:.3f}), v_star {:+.1f} km/s'.format(args.name, args.folder, args.dp, t,
                                                                                   args.vstar), fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.out, f'{args.name or args.folder}.png'.replace(' ', '_'))
    fig.savefig(path, dpi=130)
    print(path)
