import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_PRE_DATASET = '/home/msp25gd/Downloads/res/dataset/'
DEFAULT_POST_DATASET = '/home/msp25gd/ResearchProjectMSc/ResolutionHandling/processed_candidates/'
DEFAULT_METADATA = '/home/msp25gd/Downloads/res/meta/metadata.pkl'


def _resolve_target_row(star_name, to_reduce_fn, metadata_path=DEFAULT_METADATA):
    if to_reduce_fn is None:
        raise ValueError('to_reduce_fn is required.')

    reduced_name = to_reduce_fn(str(star_name))

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'Metadata file not found: {metadata_path}')

    metadata = pd.read_pickle(metadata_path)
    reduced_col = 'Reduced' if 'Reduced' in metadata.columns else 'reduced'
    sanitised_col = (
        'Sanitised'
        if 'Sanitised' in metadata.columns
        else ('sanitised' if 'sanitised' in metadata.columns else None)
    )

    match = metadata[metadata[reduced_col].astype(str).str.lower() == str(reduced_name).lower()]
    if len(match) == 0:
        raise FileNotFoundError(f"Could not find target '{star_name}' ({reduced_name}) in metadata.")

    row = match.iloc[0]
    star_red = str(row[reduced_col])
    star_san = str(row[sanitised_col]) if sanitised_col is not None else star_red
    return star_red, star_san


def _compute_epoch_rows(search, new_spectra, med, med_err):
    rows = []
    rv_mask = None
    corr_med = med.copy()

    if search.ccf:
        rv_shift = search.X_corr(corr_med)
        rv_mask = (search.radial_velocity > rv_shift - 50) & (search.radial_velocity < rv_shift + 50)
        corr_med[rv_mask] = np.nan

    for epoch_idx, spec in enumerate(new_spectra):
        snr = search.snr(spec, med, search.spectra_err[epoch_idx], med_err)
        sd = np.std(snr)

        corr_spec = spec.copy()
        if search.ccf and rv_mask is not None:
            corr_spec[rv_mask] = np.nan

        corr_snr = snr.copy()
        if search.ccf and rv_mask is not None:
            corr_snr[rv_mask] = np.nan

        corr_snr = corr_snr[search.snr_idxrange]
        filtered_rv = search.radial_velocity[search.snr_idxrange]
        sig = corr_snr / sd

        min_detect = float(np.nanmin(sig))
        width = float(search.get_width(sig)) if np.isfinite(sig).any() else 0.0
        is_detection = (min_detect < search.threshold) and (width >= search.width_filter)

        rows.append(
            {
                'corr_spec': corr_spec,
                'filtered_rv': filtered_rv,
                'sig': sig,
                'is_detection': is_detection,
            }
        )

    return corr_med, rows


def _plot_dataset_panel(ax_flux, ax_snr, search, med, corr_med, rows, panel_title):
    ax_flux.set_title(panel_title)
    ax_flux.set_ylabel('Normalised Flux')
    ax_flux.set_xlabel('Heliocentric Velocity (km/s)')
    ax_flux.set_xlim(search.rv_min, search.rv_max)

    ax_snr.set_ylabel('SNR ($\\sigma$)')
    ax_snr.set_xlabel('Heliocentric Velocity (km/s)')
    ax_snr.hlines(-1 * search.threshold, search.rv_min, search.rv_max, linestyles='dashed', linewidth=3, colors='red')
    ax_snr.hlines(1 * search.threshold, search.rv_min, search.rv_max, linestyles='dashed', linewidth=3, colors='red')

    for row in rows:
        if row['is_detection']:
            ax_flux.plot(search.radial_velocity, row['corr_spec'], linewidth=2, alpha=0.7, color='k', zorder=5)
            ax_snr.plot(row['filtered_rv'], row['sig'], linewidth=1.5, color='k', alpha=0.5, zorder=5)
        else:
            ax_flux.plot(search.radial_velocity, row['corr_spec'], linewidth=2, alpha=0.2, color='grey', zorder=0)
            ax_snr.plot(row['filtered_rv'], row['sig'], linewidth=1, color='grey', alpha=0.3, zorder=0)

    ax_flux.plot(search.radial_velocity, corr_med, linewidth=2.5, color='r', label='Median Reference', zorder=10)
    if search.ccf:
        ax_flux.plot(
            search.radial_velocity,
            med,
            linewidth=2.5,
            color='r',
            linestyle='--',
            alpha=0.2,
            label='Original Median Reference',
            zorder=0,
        )

    handles, _ = ax_flux.get_legend_handles_labels()
    spectra_legend = Line2D([0], [0], label='Superimposed spectra', color='g', alpha=0.2)
    spectra_det_legend = Line2D([0], [0], label='Spectra with detection', color='k')
    handles.extend([spectra_legend, spectra_det_legend])
    ax_flux.legend(handles=handles, loc='upper right', fontsize=9)


def plot_resolution_comparison(
    star_names,
    asset_cls,
    to_reduce_fn,
    base_param,
    line='K',
    pre_dataset=DEFAULT_PRE_DATASET,
    post_dataset=DEFAULT_POST_DATASET,
    metadata_path=DEFAULT_METADATA,
    save=False,
    save_dir='/home/msp25gd/ResearchProjectMSc/Images/finalStarImages',
):
    """Compare per-star iPlotter views before and after resolution degradation."""
    if asset_cls is None:
        raise ValueError('asset_cls is required.')
    if base_param is None:
        raise ValueError('base_param is required.')

    if isinstance(star_names, str):
        star_names = [star_names]
    else:
        star_names = list(star_names)

    if len(star_names) == 0:
        raise ValueError('Provide at least one star name.')

    if not os.path.isdir(pre_dataset):
        raise FileNotFoundError(f'Pre-degradation dataset not found: {pre_dataset}')
    if not os.path.isdir(post_dataset):
        raise FileNotFoundError(f'Post-degradation dataset not found: {post_dataset}')

    if save:
        os.makedirs(save_dir, exist_ok=True)

    for star_name in star_names:
        star_red, star_san = _resolve_target_row(star_name, to_reduce_fn=to_reduce_fn, metadata_path=metadata_path)

        pre_param = copy.deepcopy(base_param)
        pre_param['dataset'] = pre_dataset if pre_dataset.endswith('/') else pre_dataset + '/'

        post_param = copy.deepcopy(base_param)
        post_param['dataset'] = post_dataset if post_dataset.endswith('/') else post_dataset + '/'

        search_pre = asset_cls(parameters=pre_param, line=line)
        search_pre.ccf = False
        pre_path = str(Path(pre_param['dataset']) / star_red) + '/'
        pre_spec = search_pre.spec_analysis(pre_path)

        search_post = asset_cls(parameters=post_param, line=line)
        search_post.ccf = False
        post_path = str(Path(post_param['dataset']) / star_red) + '/'
        post_spec = search_post.spec_analysis(post_path)

        if pre_spec is None:
            print(f"{star_red}: not enough spectra in pre-degradation dataset.")
            continue
        if post_spec is None:
            print(f"{star_red}: not enough spectra in post-degradation dataset.")
            continue

        pre_new_spectra, pre_med, pre_med_err = pre_spec
        post_new_spectra, post_med, post_med_err = post_spec

        pre_corr_med, pre_rows = _compute_epoch_rows(search_pre, pre_new_spectra, pre_med, pre_med_err)
        post_corr_med, post_rows = _compute_epoch_rows(search_post, post_new_spectra, post_med, post_med_err)

        fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex='col', constrained_layout=True)
        fig.suptitle(f"{star_san} (Ca II {line}) - Before vs After Degradation")

        _plot_dataset_panel(
            axs[0, 0],
            axs[1, 0],
            search_pre,
            pre_med,
            pre_corr_med,
            pre_rows,
            panel_title='Before Degradation',
        )
        _plot_dataset_panel(
            axs[0, 1],
            axs[1, 1],
            search_post,
            post_med,
            post_corr_med,
            post_rows,
            panel_title='After Degradation',
        )

        plt.show()

        if save:
            safe_name = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in str(star_san))
            out_path = os.path.join(save_dir, f'{safe_name}_{str(line).upper()}_resolution_comparison.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print('Saved:', out_path)

        plt.close(fig)
