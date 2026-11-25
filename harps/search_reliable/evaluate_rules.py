'''
13/11/26
Each decision rule of decision_rules.py on the full survey: PCA on and off combined (a star counts if either finds it,
its best candidate over both), real and flipped. Reports the tiers by bundle and the known hosts.
    py -3 evaluate_rules.py        # reads and writes ../results/Overnight/
'''
import argparse
import json
import os

import pandas as pd

from calibrate_by_type import add_berv_flag
from decision_rules import RULES, decide_rule

RUNS = {'real': ('on', 'off'), 'flipped': ('on_flip', 'off_flip')}
TIER_ORDER = ['high confidence', 'confirmed, no control lines', 'candidate', 'anomaly']
HOSTS = {'GaiaDR3_4792774797545800832': 'bet Pic', 'GaiaDR3_6438274350302427776': 'HD 172555'}


def combined(out, folders, config, rule, class_counts):
    kept_all, best_all = [], []
    for folder in folders:
        scores = pd.read_pickle(os.path.join(out, folder, 'spectrum_scores.pkl'))
        kept, best = decide_rule(scores, config, rule, class_counts=class_counts)
        kept_all.append(kept.assign(search=folder))
        best_all.append(best.assign(search=folder))
    kept, best = pd.concat(kept_all), pd.concat(best_all, ignore_index=True)
    if len(best):
        best['rank'] = best['tier'].map(TIER_ORDER.index)
        best = (best.sort_values(['rank', 'confirmed', 'sigma_survey'], ascending=[True, False, False])
                .groupby('folder', as_index=False).first())
    return kept, best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decision rules on the full survey')
    parser.add_argument('--out', default='../results/Overnight/')
    parser.add_argument('--rules', default=','.join(RULES))
    parser.add_argument('--shape', default='both', help='shape test: both (min of K and H) or weighted (K + 0.5 H)')
    parser.add_argument('--alpha', type=float, default=None, help='survey false alarm budget (exploratory level: 1)')
    parser.add_argument('--c', default=None, help='settings file (default: config.json in --out)')
    parser.add_argument('--tag', default='', help='suffix for the output files')
    args = parser.parse_args()
    with open(args.c or os.path.join(args.out, 'config.json')) as handle:
        config = json.load(handle)
    config['tests']['shape_mode'] = args.shape
    tag = '' if args.shape == 'both' else '_' + args.shape
    tag += args.tag
    if args.alpha is not None:
        config['tests']['false_alarm_survey'] = args.alpha
        tag += f'_alpha{args.alpha:g}'
    with open(os.path.join(args.out, 'class_counts.json')) as handle:
        class_counts = json.load(handle)

    summary = []
    for rule in args.rules.split(','):
        for run, folders in RUNS.items():
            kept, best = combined(args.out, folders, config, rule, class_counts)
            best = add_berv_flag(best)
            best.to_csv(os.path.join(args.out, f'candidates_{rule}_{run}{tag}.csv'), index=False)
            row = dict(rule=rule, run=run, above=kept['folder'].nunique(), after_tests=len(best))
            for tier in TIER_ORDER:
                row[tier] = int((best['tier'] == tier).sum()) if len(best) else 0
            for bundle in ('OB', 'A_earlyF', 'lateF_G_K', 'M'):
                row['after ' + bundle] = int((best['class_name'] == bundle).sum()) if len(best) else 0
            for folder, name in HOSTS.items():
                match = best[best['folder'] == folder] if len(best) else best
                row[name] = match['tier'].iloc[0] if len(match) else ('above, failed tests' if folder in set(kept['folder'])
                                                                      else 'not found')
            summary.append(row)
            print(rule, run, row, flush=True)
            if len(best):
                columns = [c for c in ('folder', 'class_name', 'tier', 'route', 'search', 'v0', 'v_star', 'near_berv', 'sigma_survey', 'n')
                           if c in best]
                print(best[columns].to_string(index=False), flush=True)
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out, f'rules_summary{tag}.csv'), index=False)
    print(summary.to_string())
