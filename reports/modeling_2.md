# Modeling 2 - Improved XGBoost (Skubal Next-Pitch Prediction)

## Goal

Recover the FF class without losing the per-class F1 gains the basic Random Forest delivered on CH and SI. Strategy: deeper sequence features, batter identity, gradient boosting, and a hyperparameter search.

## What changed vs baseline

1. Added lag-2 and lag-3 prior-pitch type one-hots (extending lag-1 from modeling_1).
2. Added a frequency-encoded `batter_freq` feature, fitted on the train split only (avoids the 1000+ one-hot columns a categorical batter id would create).
3. Replaced sklearn Random Forest with XGBoost (`objective='multi:softprob'`).
4. Hyperparameter search over four configs on the validation split, then refit best on `train + val` and evaluate once on the held-out test split.

Final feature count: 29.

## Hyperparameter search (validation accuracy)

| Cfg | depth | lr | n_est | min_child | subsample | colsample | val acc | val macro F1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 0.10 | 200 | 5 | 0.9 | 0.8 | 0.344 | 0.241 |
| 1 | 6 | 0.07 | 250 | 3 | 0.9 | 0.8 | 0.329 | 0.248 |
| 2 | 6 | 0.05 | 300 | 5 | 0.8 | 0.7 | 0.329 | 0.243 |
| 3 | 8 | 0.05 | 250 | 5 | 0.8 | 0.7 | 0.309 | 0.228 |

Best config selected on macro F1: `max_depth=6, learning_rate=0.07, n_estimators=250, min_child_weight=3, subsample=0.9, colsample_bytree=0.8`. The depth-8 variant overfits on this sample size.

## Final test results (n=2,200)

| Model | Accuracy | Macro F1 | Weighted F1 | Log loss |
|---|---|---|---|---|
| Majority FF | 0.347 | 0.103 | 0.179 | n/a |
| LogReg (basic) | 0.284 | 0.255 | 0.302 | 1.580 |
| RF (basic) | 0.296 | 0.262 | 0.313 | 1.537 |
| **XGBoost (full)** | **0.387** | **0.271** | **0.368** | **1.413** |

The final XGBoost beats every baseline on every metric, including the always-FF accuracy benchmark. Log loss drops from 1.54 (RF) to 1.41 (a ~8% gain in calibration).

Per-class F1 (test):

| Class | LogReg | RF | XGB final |
|---|---|---|---|
| CH | 0.414 | 0.404 | 0.399 |
| CU | 0.116 | 0.097 | 0.027 |
| FF | 0.245 | 0.265 | **0.480** |
| SI | 0.386 | 0.406 | 0.347 |
| SL | 0.117 | 0.137 | 0.104 |

XGBoost roughly doubles FF recall (FF F1: 0.27 -> 0.48) by leaning into the prior, gives back some SI recall, and effectively gives up CU. CU at 5.1% prior with high overlap is the next modeling problem.

## Top features (XGB gain importance)

| Feature | Gain |
|---|---|
| stand_R | 0.162 |
| prev_pitch_1_CH | 0.062 |
| strikes | 0.040 |
| prev_pitch_1_FF | 0.039 |
| balls | 0.036 |
| prev_pitch_1_NONE | 0.036 |
| prev_pitch_1_SI | 0.034 |
| prev_pitch_3_SI | 0.033 |
| prev_pitch_2_FF | 0.032 |
| prev_pitch_2_CH | 0.032 |
| prev_pitch_3_NONE | 0.030 |
| prev_pitch_2_SI | 0.030 |
| prev_pitch_1_SL | 0.029 |
| prev_pitch_1_CU | 0.029 |
| on_1b_flag | 0.028 |

Lag features dominate after batter handedness, confirming the sequence-conditioned nature of pitch selection.

## Confusion (XGB final, test)

Rows = true, columns = predicted (order CH, CU, FF, SI, SL):

```
CH:  [255,  2, 293,  64,  35]
CU:  [ 22,  1,  27,  10,   7]
FF:  [191,  2, 437,  88,  45]
SI:  [ 89,  2, 160, 136,  54]
SL:  [ 71,  1, 140,  45,  23]
```

FF is now the dominant prediction across the diagonal, but CU collapses (1 of 67 correct). Two-thirds of CU true pitches are misclassified as CH or FF.

## Takeaways and limitations

- 4 percentage points of accuracy and 1.3 of weighted F1 over the basic RF, with better calibration, by adding sequence context (lag-2, lag-3) plus batter identity.
- The trade-off is CU recall. A class-weighted or focal-loss training run would close that gap; a separate "is-CU" head is also viable since CU usage is heavily count- and game-state-dependent.
- 2-class macro F1 is constrained by the limited Skubal-only sample (~13.7k pitches across 6 partial seasons). A multi-pitcher model with pitcher identity as a feature would likely transfer.

## Persisted artifacts

- `deliverables/skubal_xgb.pkl` - final XGBoost model + `feature_order` + `classes` + `label_map` + `batter_freq` table
- `deliverables/metrics.json` - all four model metric blocks for reproducibility
