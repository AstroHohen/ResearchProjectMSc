'''
07/11/26
Freeze the method, then test it on data it has never seen (idea 11).

The known hosts and the quiet stars helped design the search, so they are not a blind test. A blind test needs spectra
observed after the method was fixed:
  py -3 blind_test.py freeze   # writes FROZEN.json (the date and a sha256 of each code file and of the settings) and
                               # FROZEN_spectra.txt (every spectrum the method was tuned on)
  py -3 blind_test.py run      # refuses to run if any frozen file changed; then runs stages 1 and 2 unchanged on every
                               # star and keeps only detections in spectra that are not in FROZEN_spectra.txt
Unseen means not in that list, not "observed later": the archive also releases old observations when their
proprietary year ends (312 such spectra arrived on 7 November 2026, observed in November 2024). New spectra reach the
star folders only when the dataset is rebuilt (harps_fetch.py, grouping, dataset notebook). Until then `run` finds
none and says so.
'''
import argparse
import hashlib
import json
import multiprocessing
import os
import sys
from datetime import datetime, timezone

import pandas as pd

import exocomet_search as es
from calibrate_by_type import decide

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['exocomet_search.py', 'calibrate_by_type.py', 'spectral_classes.json', os.path.join('..', 'search', 'asset.py'),
         os.path.join('..', 'search', 'quicksearch_v3.py')]
FROZEN = os.path.join(HERE, 'FROZEN.json')
SEEN = os.path.join(HERE, 'FROZEN_spectra.txt')
SCORED = os.path.join(HERE, '..', 'results', 'Reliable', 'spectrum_scores.pkl')  # the spectra the method was tuned on


def digest(name):
    with open(os.path.join(HERE, name), 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def freeze():
    seen = sorted(pd.read_pickle(SCORED)['dp_id'].unique())
    with open(SEEN, 'w') as handle:
        handle.write(chr(10).join(seen))
    record = dict(frozen_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), spectra_seen=len(seen),
                  files={name: digest(name) for name in FILES + ['FROZEN_spectra.txt']})
    with open(FROZEN, 'w') as handle:
        json.dump(record, handle, indent=2)
    print('Frozen at', record['frozen_utc'], '-', len(seen), 'spectra seen')


def run(results):
    with open(FROZEN) as handle:
        record = json.load(handle)
    changed = [name for name, value in record['files'].items() if digest(name) != value]
    if changed:
        sys.exit('These files changed after the freeze, so this would not be a blind test: ' + ', '.join(changed))
    with open(os.path.join(HERE, 'spectral_classes.json')) as handle:
        config = json.load(handle)
    index = es.build_jobs(config)
    config['search']['n_stars'] = len(index)
    jobs = list(zip(index['folder'], index['class_name']))
    rows = []
    with multiprocessing.Pool(config['search']['cores'], initializer=es.init_worker, initargs=(config,)) as pool:
        for star_rows in pool.imap_unordered(es.search_star, jobs):
            rows += star_rows
    scores = pd.DataFrame(rows)
    with open(SEEN) as handle:
        seen = set(handle.read().split())
    new = ~scores['dp_id'].isin(seen)
    print('Spectra not seen when the method was frozen:', int(new.sum()), 'in', scores.loc[new, 'folder'].nunique(), 'stars')
    if not new.any():
        return
    kept, best = decide(scores, config)
    kept = kept[~kept['dp_id'].isin(seen)]
    os.makedirs(results, exist_ok=True)
    kept.to_pickle(os.path.join(results, 'blind_detections.pkl'))
    print('Detections in new spectra:', kept['folder'].nunique(), 'stars,', int(kept['pass_all'].sum()), 'passing all tests')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Freeze the search, or run it blind on newer spectra')
    parser.add_argument('step', choices=['freeze', 'run'])
    parser.add_argument('--r', default='../results/Reliable_blind/', help='output (default: ../results/Reliable_blind/)')
    args = parser.parse_args()
    freeze() if args.step == 'freeze' else run(args.r)
