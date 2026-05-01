# Project 3 - What Pitch Should Tarik Skubal Throw?

## Exploration Report 1: Data Description and EDA

**Pitcher**: Tarik Skubal, LHP, Detroit Tigers (MLBAM ID `669373`, debut 18 Aug 2020)
**Data source**: MLB Statcast pitch-by-pitch parquet files, one per season, pulled via the `pybaseball` library
**Files**: `data/statcast_2015.parquet` through `data/statcast_2026.parquet` (~943 MB combined)
**Notebook**: `notebooks/01_eda.ipynb` (executed end-to-end without errors)

---

### 1. Data description

Each parquet holds every pitch thrown in a regular- or post-season MLB game for the given year. The 2024 file alone contains **760,248 rows and 118 columns**, which is representative of the post-2020 schema. Columns fall into seven domain groups:

| Group | Key columns |
|---|---|
| Identifiers | `game_pk`, `game_date`, `game_year`, `at_bat_number`, `pitch_number`, `pitcher`, `batter`, `player_name` |
| Pitch label | `pitch_type` (FF, SI, SL, CH, CU, FS, FC, KC, etc.), `pitch_name`, `description`, `events`, `type` |
| Pitch physics | `release_speed`, `release_spin_rate`, `release_extension`, `spin_axis`, `pfx_x`, `pfx_z`, `plate_x`, `plate_z`, `vx0/vy0/vz0`, `ax/ay/az` |
| Count and game state | `balls`, `strikes`, `outs_when_up`, `inning`, `inning_topbot`, `on_1b/2b/3b` |
| Matchup | `stand`, `p_throws`, `home_team`, `away_team`, `home_score`, `away_score`, `bat_score`, `fld_score` |
| Batted-ball outcome | `bb_type`, `launch_speed`, `launch_angle`, `hit_distance_sc`, `estimated_woba_using_speedangle` |
| Win-probability deltas | `delta_run_exp`, `delta_pitcher_run_exp`, `delta_home_win_exp`, `home_win_exp` |

`pitch_type` is the natural target for Phase 1 modelling. `description` (ball, called_strike, swinging_strike, foul, hit_into_play) and `delta_run_exp` (run-value of each pitch) are the natural targets for a Phase 2 outcome model.

---

### 2. Skubal sample size by season

Filtering on `pitcher == 669373` across all 12 parquet files yields **13,935 pitches across 161 unique games**, all in seasons 2020-2026 (zero rows pre-2020, as expected from his debut date).

| Season | Pitches | Games | Avg release speed (mph) | Avg spin rate (rpm) |
|---|---:|---:|---:|---:|
| 2020 | 591 | 8 | 89.86 | 2259 |
| 2021 | 2,776 | 34 | 89.50 | 2032 |
| 2022 | 2,141 | 25 | 90.04 | 2050 |
| 2023 | 1,219 | 15 | 89.97 | 2007 |
| 2024 | 3,295 | 36 | 92.07 | 2045 |
| 2025 | 3,304 | 36 | 93.26 | 2160 |
| 2026 | 609 | 7 | 92.41 | 2161 |
| **Total** | **13,935** | **161** | **91.39** | **2092** |

2023 is depressed because of an oblique strain that delayed his season start. 2026 is partial (data through the dataset cut-off). Velocity has stepped up sharply from 2023 (89.97 mph) to 2025 (93.26 mph), which is the single biggest signal in the data.

---

### 3. Career pitch mix (overall)

| Code | Pitch | Count | % |
|---|---|---:|---:|
| FF | 4-seam fastball | 4,906 | 35.21 |
| CH | Changeup | 3,058 | 21.94 |
| SL | Slider | 2,573 | 18.46 |
| SI | Sinker (2-seam) | 2,462 | 17.67 |
| CU | Curveball | 697 | 5.00 |
| NA | Unclassified | 120 | 0.86 |
| FS | Splitter | 97 | 0.70 |
| FC | Cutter | 19 | 0.14 |
| KC | Knuckle-curve | 3 | 0.02 |

Five pitch types (FF, CH, SL, SI, CU) cover **98.3%** of his career. FC and KC are vestigial and only appear in 2020-2021. FS appears in occasional bursts (mostly 2021).

---

### 4. Pitch mix by season (% of pitches, top 5 pitches)

| Season | FF | SI | SL | CH | CU |
|---|---:|---:|---:|---:|---:|
| 2020 | 58.9 | 0.0 | 15.7 | 16.4 | 7.8 |
| 2021 | 44.1 | 11.8 | 22.6 | 11.0 | 6.8 |
| 2022 | 28.9 | 19.6 | 29.7 | 15.2 | 6.6 |
| 2023 | 36.0 | 12.1 | 20.7 | 24.1 | 6.8 |
| 2024 | 32.6 | 21.2 | 14.1 | 27.0 | 3.9 |
| 2025 | 29.4 | 22.8 | 12.7 | 29.9 | 2.6 |
| 2026 | 38.6 | 18.9 | 13.0 | 25.9 | 3.4 |

Three structural shifts are visible:

1. **Sinker introduction (2021)**. Before 2021 Skubal threw zero sinkers; in 2025 the SI was 22.8% of his pitches, basically tied with the FF.
2. **Slider de-emphasis (2024 onward)**. SL share fell from 29.7% in 2022 to 12.7% in 2025 as the changeup took over as his go-to secondary.
3. **Changeup ascendancy (2023 onward)**. CH share roughly doubled from ~12% (2021) to ~30% (2025) and is now his most-used pitch alongside the FF.

The cutter and knuckle-curve are extinct in the modern arsenal. Curveball usage has shrunk to 2-3%.

---

### 5. Pitch profile by type (career averages)

| Pitch | n | Avg speed (mph) | Avg spin (rpm) |
|---|---:|---:|---:|
| FF | 4,906 | 95.8 | 2,252 |
| SI | 2,462 | 96.2 | 2,136 |
| FC | 19 | 94.7 | 2,283 |
| SL | 2,573 | 88.3 | 2,127 |
| CH | 3,058 | 85.9 | 1,698 |
| FS | 97 | 85.0 | 1,303 |
| CU | 697 | 77.0 | 2,295 |
| KC | 3 | 81.0 | 1,387 |

The FF and SI sit in the same velocity band (~95-96 mph) and are differentiated by movement profile, not speed. CH is ~10 mph slower than the FF, which is the textbook changeup velocity gap. CU is the slowest, highest-spin pitch (low-77 mph, ~2,300 rpm). The unclassified `NA` rows (n=120, 0.86%) appear evenly distributed and can be dropped or treated as a low-cardinality OTHER class.

---

### 6. Count situation (balls/strikes)

All 12 ball/strike states (0-0 through 3-2) are populated. Pitch selection is highly count-dependent.

Key rows (% of pitches, top 5 pitches):

| Count | FF | SI | SL | CH | CU |
|---|---:|---:|---:|---:|---:|
| 0-0 | 35.0 | 20.4 | 21.8 | 14.0 | 7.3 |
| 1-0 | 34.8 | 21.4 | 17.4 | 22.7 | 1.5 |
| 0-2 | 38.9 | 10.5 | 16.4 | 27.3 | 5.2 |
| 1-2 | 34.4 | 12.1 | 17.6 | 28.6 | 5.4 |
| 2-2 | 35.0 | 16.6 | 18.3 | 23.4 | 5.1 |
| 3-0 | 84.1 | 15.0 | 0.0 | 0.0 | 0.0 |
| 3-1 | 54.1 | 23.9 | 17.0 | 3.7 | 0.0 |
| 3-2 | 42.2 | 18.6 | 19.9 | 16.5 | 1.1 |

Behind in the count (3-0, 3-1) Skubal becomes nearly all-fastball. Ahead in the count (0-2, 1-2, 2-2) he leans much harder on the changeup as his put-away pitch. Curveballs essentially vanish in any hitter's count. **Count state is a high-signal feature for the next-pitch model.**

---

### 7. Batter-handedness split

| Stand | FF | SI | SL | CH | CU |
|---|---:|---:|---:|---:|---:|
| L (LHB) | 23.8 | 35.1 | 29.5 | 8.2 | 2.3 |
| R (RHB) | 37.7 | 13.9 | 16.1 | 24.9 | 5.6 |

Strong handedness platoon split: against same-handed (LHB) hitters Skubal leads with the SI and SL; against opposite-handed (RHB) hitters he leads with the FF and CH. This will be one of the strongest features in the model.

---

### 8. Target variable for modelling

We define `next_pitch_type` as the `pitch_type` of the next pitch in the same `(game_pk, at_bat_number)`, sorted by `pitch_number`. The last pitch of each at-bat has no successor and is dropped, leaving **10,284 of 13,935 rows (73.8%)** with a defined target.

**Majority-class baseline** (always predict FF): accuracy = **0.356**. Any model needs to clear this bar.

A current-pitch-conditioned baseline (predict the modal next pitch given the current pitch type) recovers some structure: e.g. after an FF in a 2-strike count the next pitch is often a CH or SL, which is exactly the put-away pattern visible in section 6.

---

### 9. Missingness on key columns

`pitch_type`, `release_speed`, `release_spin_rate`, `pfx_x/z`, `plate_x/z`, `balls`, `strikes`, `outs_when_up`, `inning`, `stand`, `p_throws` all have <2% missingness on the Skubal subset, with `pitch_type` itself missing on 0.86% of pitches (the `NA` bucket above). No critical columns require imputation beyond dropping or labelling the unclassified pitches.

---

### 10. Key observations for the prediction model

1. **Career sample is healthy (~13.9k pitches, 161 games) but skewed late.** 75% of the data sits in 2021-2025. Only 591 pitches in the rookie cup-of-coffee 2020 season; only 1,219 in injury-shortened 2023.

2. **Skubal's arsenal is non-stationary.** Velocity is up ~3 mph since 2023 and the SI/CH have replaced the SL as his dominant secondary pitches. **Recommendation**: train on 2022-2026 only, or apply recency weighting; modelling 2020-2021 data is misleading for predicting 2026 behaviour.

3. **Effective class space is 5 (FF, SI, SL, CH, CU).** Fold FC/KC/FS/NA into an OTHER class or drop them; they collectively represent <2% of pitches.

4. **High-signal features** (in expected importance order):
   - `balls`, `strikes` (count state, 12 levels) - drives huge mix shifts (3-0 = 84% FF; 0-2 = 28% CH)
   - `stand` (LHB vs RHB) - 14-percentage-point swing in FF usage; 21-pt swing in SI usage
   - Previous pitch type and previous pitch result within the at-bat (sequencing)
   - Velocity/spin/extension of previous pitch (proxy for pitcher fatigue and adjustment)
   - `inning`, `n_thruorder_pitcher` (times through the order, present in the schema)
   - Score differential (`bat_score_diff`, available in schema)
   - Runners on (`on_1b/2b/3b`)

5. **Target leakage to avoid**: never use any column that describes the next pitch itself (release_speed of the next pitch, `description`, `events`). Use only state available at the moment of the decision.

6. **Validation split**: stratify by `game_pk` to keep all pitches from a single game in either train or test. Random pitch-level splits will leak within-at-bat sequencing and inflate accuracy.

7. **Two-stage extension**: once next-pitch type is in hand, the natural next target is *outcome* (whiff, called strike, weak contact, hard contact) using `description` plus `delta_run_exp`. That answers the brief's actual question (which pitch maximises success) rather than just mimicking what Skubal already throws.

---

### Files produced

- `notebooks/01_eda.ipynb` - executed end-to-end, 17 code cells, 0 errors
- `reports/exploration_1.md` - this file
- `src/_inspect.py`, `src/_summary.py`, `src/_extract_metrics.py` - helper scripts used to verify the numbers above
