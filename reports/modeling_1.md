# Modeling 1 - Baseline (Skubal Next-Pitch Prediction)

## Setup

Multi-class classification of the next pitch type Tarik Skubal (MLBAM 669373) will throw, given pre-pitch context only. Source data: MLB Statcast 2020-2026 (debut Aug 2020).

After filtering to the five effective classes (FF, CH, SL, SI, CU) and dropping rows with missing pre-pitch state, the modeling table holds **13,696 pitches**.

Class prior on the full table:

| Pitch | Share |
|---|---|
| FF | 35.8% |
| CH | 22.3% |
| SL | 18.8% |
| SI | 18.0% |
| CU | 5.1% |

## Split

Time-based, no shuffle:

- Train: 2020-2024 (9,871 rows)
- Val: 2025 first half (1,625 rows)
- Test: 2025 second half + 2026 (2,200 rows)

This split mirrors the deployment task: the model must extrapolate forward to unseen games rather than peek into the future of the same season.

## Features (basic set, 28 columns)

Pre-pitch context only:

- Count: `balls`, `strikes`
- Game state: `outs_when_up`, `inning`, `on_1b/2b/3b_flag`, `score_diff`
- Batter handedness one-hot (`stand_R`)
- At-bat position counter (`ab_pitch_idx`)
- Lag-1 prior pitch type one-hot (FF / SI / SL / CH / CU / NONE)

## Baselines (test set, n=2,200)

| Model | Accuracy | Macro F1 | Weighted F1 | Log loss |
|---|---|---|---|---|
| Always-FF (majority) | 0.347 | 0.103 | 0.179 | n/a |
| Logistic Regression (basic) | 0.284 | 0.255 | 0.302 | 1.580 |
| Random Forest (basic) | 0.296 | 0.262 | 0.313 | 1.537 |

The constant-FF baseline already achieves accuracy 0.347 because of the FF prior. Both probabilistic baselines underperform on accuracy but recover meaningful per-class F1 on CH and SI:

| Class | LogReg F1 | RF F1 |
|---|---|---|
| CH | 0.41 | 0.40 |
| CU | 0.12 | 0.10 |
| FF | 0.24 | 0.26 |
| SI | 0.39 | 0.41 |
| SL | 0.12 | 0.14 |

CU is the rarest class (5.1% prior) and neither baseline learns it.

## Random Forest configuration

`n_estimators=400`, `max_depth=12`, `min_samples_leaf=2`, `class_weight='balanced'`, `n_jobs=4`.

Top features by impurity importance:

| Feature | Importance |
|---|---|
| stand_R (batter handedness) | 0.198 |
| inning | 0.113 |
| balls | 0.104 |
| prev_pitch_1_CH | 0.088 |
| score_diff | 0.085 |
| ab_pitch_idx | 0.084 |
| strikes | 0.068 |
| prev_pitch_1_FF | 0.060 |
| outs_when_up | 0.054 |
| on_1b_flag | 0.043 |

Batter handedness dominates: Skubal mixes SI / SL more vs LHB and FF / CH more vs RHB.

## Confusion (Random Forest, test)

Rows = true pitch, columns = predicted pitch, order [CH, CU, FF, SI, SL]:

```
CH:  [267, 164,  95,  76,  47]
CU:  [ 20,  32,  10,   3,   2]
FF:  [247, 206, 149, 118,  43]
SI:  [ 84,  86,  61, 174,  36]
SL:  [ 54, 104,  47,  45,  30]
```

The classifier over-predicts CH and CU at the expense of FF; SL is poorly recovered. The class imbalance plus heavily overlapping feature distributions (balls / strikes look similar across pitch types within a count) limit the basic-feature ceiling.

## Takeaways

- The class prior is informative: any non-trivial baseline must beat 0.347 accuracy.
- Logistic Regression and basic Random Forest reach macro F1 around 0.26 but lose on accuracy because they spread predictions away from FF.
- Lag-1 prior-pitch features carry the strongest signal beyond batter handedness.
- Improvements next step: deeper sequence (lag-2, lag-3), batter identity, and a calibrated boosted tree.
