# Overnight test of three search improvements: results (14 Nov 2026)

Rules and the winner criterion were fixed before the runs (PREREGISTRATION.md, with two addenda and one bug fix,
all timestamped). Full survey: 1,743 stars with scores, PCA on and off combined, real and flipped. The fake comets
are 60 stars per bundle (49 in M), depths 1-20%, FWHM 5/10/20 km/s, and a comet-rich star; the same injected
spectra are judged by every rule. Code: `harps/search_reliable/decision_rules.py`, `evaluate_rules.py`,
`validate_rules.py`. Stage-1 option `slow_budget_factor` (results/Overnight/config.json) computes the slow tests for
more spectra; the baseline decision on these files reproduces results/TwoRuns exactly (7 real, 6 flipped).

## The rules
| Rule | Idea |
| --- | --- |
| baseline | two-run search as it is |
| 1 type_weight | budget shared by spectral type from where Ca II exocomets are known (A to F4 1, OB 0.1, F5 to K and M 0.01) |
| 2 star_level | half the budget for a star test (how many nights have p_night < 0.001), then the star's own spectra |
| 3 pairs | half the budget for pairs: a spectrum and another of its night at the same velocity |
| 4 combined | added after 1-3 (addendum 2): type weights on the baseline, star and pair routes, a third each |

## Criteria (flipped high confidence without the two hosts; bet Pic; HD 172555)
| Rule | Flipped high confidence | bet Pic | HD 172555 | Meets all |
| --- | --- | --- | --- | --- |
| baseline | 1 (ups Aqr, F7V) | confirmed, no control lines | not found | no |
| 1 type_weight | **0** | confirmed, no control lines | not found | no (HD 172555) |
| 2 star_level | 2 (ups Aqr; HD 192310, K2V) | **high confidence** | not found | no |
| 3 pairs | 1 (ups Aqr, through the baseline route) | confirmed, no control lines | **confirmed, no control lines** | no |
| 4 combined | 1 (HD 192310, K2V, star route) | **high confidence** | **confirmed, no control lines** | no |

Every flipped high-confidence star is F5 to K. In that bundle the flipped run has 2.5 times more strong spectra than
the real run (72 against 26 stars with p_survey < 25, PCA on), the known bias of the control there (activity
emission becomes a dip when flipped).

## Fake comets that pass the tests (either search)
| Bundle, depth | baseline | 1 type_weight | 2 star_level | 3 pairs | 4 combined |
| --- | --- | --- | --- | --- | --- |
| A to F4, 5% | 13.3 | 15.6 | 12.8 | 20.6 | 21.1 |
| A to F4, 10% | 27.8 | 33.3 | 27.2 | **44.4** | **44.4** |
| A to F4, 20% | 41.7 | 47.8 | 39.4 | 56.7 | 57.2 |
| A to F4, comet-rich | 30.0 | 35.0 | 35.0 | 36.7 | **48.3** |
| OB, 10% | 53.3 | 52.2 | 51.7 | 70.0 | 68.9 |
| F5 to K, 20% | 13.3 | 11.1 | 12.8 | 20.0 | 17.8 |
| M, any | 0 | 0 | 0 | 0 | 0 |

High confidence for the fake comets in A to F4 at 10% depth: baseline 13.3%, type_weight 14.4%, star_level 12.8%,
pairs 21.1%, combined 21.1%; comet-rich: 8.3, 11.7, 25.0, 16.7, **35.0%**.

## Real run (reported only, not used to choose)
Combined: 12 stars after the tests. High confidence: bet Pic (A6V, +28 km/s), HD 179079 (G5IV, -98 km/s, star route),
HD 214686 (F7V, +200 km/s, pair route). Confirmed, no control lines: HD 172555 (A7V, -3 km/s, pair route).
Doubts: HD 179079 is F5 to K and came through the star route, the route of the flipped false alarm; HD 214686 is at
+200 km/s, the edge of the searched range.

## Verdict by the pre-registered rule
No rule meets all three criteria. By the fallback order (criterion 1, then 2, then 3, then recovery) the winner is
**type_weight**: the only one with 0 flipped high-confidence stars, but it misses HD 172555 and recovers fewer fake
comets than pairs or combined.

The strongest rule is **combined**: it finds both hosts, puts bet Pic at high confidence, and recovers 60% more fake
comets in A to F4 than the baseline. It fails criterion 1 by one flipped F5-to-K star. Making it pass would need a
change to how F5 to K stars can reach the top tier, which is a decision for the user, not a re-tuning done here.

## Veto of the candidates of pairs and combined (flag only)
Real: bet Pic 329 of 336 detections clean (7 stellar dip); HD 172555 clean; HD 179079 clean (4); HD 214686 clean;
GV Car all 6 stellar dip; TY CrA young star, 2 of 6 stellar dip. Flipped: HD 192310 (the combined false alarm) clean
in all 4, ups Aqr clean. So the veto does not separate the flipped late-type false alarms from the real late-type
candidates: it cannot replace a decision on F5 to K. Files: veto_real/, veto_flipped/.

## Attrition test and weighted shape (14 Nov 2026, addendum 3)
Script: `search_reliable/validate_attrition.py`; files attrition_injection.pkl/.csv, attrition_flipped.csv. Shares
here are of injection runs where stage 1 returned scored rows, so they are higher than the table above; compare
shape modes and steps with each other, not with that table.

Flipped control and hosts: weighted shape (K + 0.5 H)/1.5 >= 0.5 gives the same flipped high-confidence star as
min(K, H) >= 0.5 (baseline and combined), and bet Pic and HD 172555 keep their tiers. Passes the adoption condition.

Fake comets, combined rule, passed the tests (either search), min(K,H) -> weighted:
A to F4 5% 34.9 -> 36.7, 10% 60.2 -> 60.9, 20% 68.7 -> 69.3, comet-rich 64.4 -> 66.7; high confidence unchanged
except 5% (13.8 -> 14.7). A small gain at no cost.

Where the A-to-F4 fake comets are lost (combined, min(K,H), 1,075 runs PCA on and off counted apart):
- stage 1 does not detect them: 559 runs (52%), the largest loss
- a stage-2 test rejects them: 33 of 516 detected (6%): shape 19, control lines 13, twin 1
- they pass but are not high confidence: 229 of 483, mainly no control lines (141), near H epsilon (76), control
  lines too weak (60) (a comet can have more than one)
So the tests are not the obstacle; the stage-1 threshold and the tier rules (control lines) are.
