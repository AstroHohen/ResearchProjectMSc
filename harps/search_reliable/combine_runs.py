'''
13/11/26
Two searches of every star: with principal components (PCA on) and without (PCA off). PCA removes what the star does
at many epochs, which helps most stars, but in a star with comets at most epochs (bet Pic) it learns the comets as
part of the star and removes them. A star counts if either search finds it. The false alarm budget is shared
(tests['searches'] = 2: each search gets half), and the flipped searches are combined the same way, so the control
has the same two chances as the real search.

Run after stage 2 of the four searches:
    py -3 combine_runs.py       # writes ../results/TwoRuns/
'''
import argparse
import json
import os

import pandas as pd

RUNS = {'real': {'PCA on': 'Reliable', 'PCA off': 'Reliable_nopca'},
        'flipped': {'PCA on': 'Reliable_flip', 'PCA off': 'Reliable_nopca_flip'}}
TIER_ORDER = ['high confidence', 'confirmed, no control lines', 'candidate', 'anomaly']


def combine(results, folders):
    '''Best candidate of each star over both searches, which searches found it above threshold and after the tests'''
    above, candidates = [], []
    for search, folder in folders.items():
        path = os.path.join(results, folder)
        kept = pd.read_pickle(os.path.join(path, 'scored_detections.pkl'))
        above.append(kept.groupby('folder')['class_name'].first().to_frame().assign(search=search))
        best = pd.read_csv(os.path.join(path, 'candidates.csv'))
        candidates.append(best.assign(search=search))
    above = pd.concat(above).reset_index()
    above = above.groupby('folder').agg(class_name=('class_name', 'first'), above_in=('search', lambda s: ', '.join(sorted(s))))
    candidates = pd.concat(candidates, ignore_index=True)
    if len(candidates) == 0:
        return above, candidates
    found = candidates.groupby('folder')['search'].agg(lambda s: ', '.join(sorted(s))).rename('found_by')
    candidates['tier_rank'] = candidates['tier'].map(TIER_ORDER.index)
    best = (candidates.sort_values(['tier_rank', 'confirmed', 'sigma_survey'], ascending=[True, False, False])
            .groupby('folder', as_index=False).first().drop(columns='tier_rank'))
    best = best.merge(found, on='folder').rename(columns={'search': 'best_from'})
    return above, best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine the PCA on and PCA off searches')
    parser.add_argument('--c', default='spectral_classes.json')
    parser.add_argument('--results', default='../results/')
    args = parser.parse_args()
    with open(args.c) as handle:
        config = json.load(handle)

    out = os.path.join(args.results, 'TwoRuns')
    os.makedirs(out, exist_ok=True)
    summary = {}
    for run, folders in RUNS.items():
        above, best = combine(args.results, folders)
        above.to_csv(os.path.join(out, f'above_threshold_{run}.csv'))
        best.to_csv(os.path.join(out, f'candidates_{run}.csv'), index=False)
        summary[(run, 'above threshold')] = above.groupby('class_name').size()
        summary[(run, 'after tests')] = best.groupby('class_name').size() if len(best) else pd.Series(dtype=int)
        print(f'\n{run}: {len(above)} stars above threshold, {len(best)} after the tests')
        if len(best):
            columns = [c for c in ('folder', 'class_name', 'tier', 'found_by', 'best_from', 'sigma_survey', 'v0', 'veto')
                       if c in best]
            print(best[columns].to_string(index=False))
    table = pd.DataFrame(summary).fillna(0).astype(int)
    table.loc['all'] = table.sum()
    table.to_csv(os.path.join(out, 'summary.csv'))
    print('\nStars by bundle (budget split over {} searches)'.format(config['tests'].get('searches', 1)))
    print(table.to_string())
