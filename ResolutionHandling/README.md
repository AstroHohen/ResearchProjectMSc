# ResolutionHandling

Recreated pipeline to handle mixed spectral resolutions after QuickSearch V2 grouping.

## Files
- `01_identify_groups.ipynb`:
  - Loads QuickSearch V2 candidate groups and full metadata
  - Splits groups into `uniform` and `mixed` by `SPEC_RES` uniqueness
  - Saves:
    - `uniform_groups.npy`
    - `mixed_groups.npy`
    - `all_candidate_groups.npy`
    - `logs/group_resolution_summary.csv`

- `02_build_processed_dataset.ipynb`:
  - Rebuilds a full candidate dataset in `processed_candidates/`
  - Uniform groups are copied unchanged
  - Mixed groups are degraded to the lowest group resolution and metadata is updated
  - Saves processing log: `logs/processed_groups_log.csv`

- `03_verify_and_summarize.ipynb`:
  - Verifies mixed groups now have a single `SPEC_RES`
  - Prints issue summary and status counts

## Output dataset
- `processed_candidates/` mirrors group folders from:
  - `/home/msp25gd/ResearchProjectMSc/HR/results/Assessed/Candidates`

## Run order
1. Run notebook 1
2. Run notebook 2
3. Run notebook 3

Then run QuickSearch V2 again on `processed_candidates/` (by pointing its input base path there).
