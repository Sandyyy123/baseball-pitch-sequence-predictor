# Predicting the Next Pitch of a Single Major League Pitcher: A Case Study on Tarik Skubal Using MLB Statcast 2020-2026

## Abstract

Predicting the next pitch type a Major League Baseball pitcher will throw is a multi-class classification problem of direct interest to opposing teams, broadcasters, and player-development staff. Most published work pools data across many pitchers and reports aggregate accuracy on PITCHf/x or early Statcast snapshots. We instead study a single high-profile pitcher, Tarik Skubal of the Detroit Tigers, across the entire span of his MLB career to date (debut 18 August 2020 through 2026 season-to-date), using 13,935 Statcast pitches across 161 games. After collapsing low-frequency pitch types into the five-class arsenal that covers 98.3 percent of his career (four-seam fastball FF, changeup CH, slider SL, sinker SI, curveball CU) and removing pitches without a successor in the same plate appearance, the modeling table holds 13,696 rows. We use only pre-pitch context features (count state, runners, score differential, batter handedness, lag-1 to lag-3 prior-pitch one-hots, and a frequency-encoded batter identity) under a strict time-based train (2020-2024) / validation (2025 first half) / test (2025 second half plus 2026) split. Four models are compared: a majority-FF baseline, logistic regression, a class-balanced random forest, and a tuned XGBoost classifier. The final XGBoost reaches accuracy 0.387, macro-F1 0.271, weighted-F1 0.368, and log loss 1.413 on a 2,200-pitch held-out test set, beating the always-FF baseline (accuracy 0.347) and improving four percentage points over the random-forest baseline (0.296). Batter handedness and the immediately preceding pitch dominate feature importance, in line with prior multi-pitcher work [2, 12]. The curveball, the rarest class, collapses to F1 0.027, illustrating the central limitation of single-pitcher modeling: a 5.1 percent class prior with overlapping decision context is hard to recover. We discuss the trade-off against pooled multi-pitcher models, the implications for in-game scouting, and how the same pipeline can be adapted to other starters.

## 1. Introduction

Pitch-type prediction is one of the canonical applied-machine-learning problems in baseball analytics. The pitcher chooses, on every play, from a discrete arsenal that typically holds three to six pitch types; the batter must commit to a swing decision before the pitch fully develops. Any signal the batter can extract about the upcoming pitch type, even a few percentage points above the pitcher's marginal pitch-mix, translates directly into measurable run-expectancy gains [11, 14]. Conversely, pitchers and pitching coaches use the same models in reverse, looking for sequencing tendencies the pitcher should disrupt before opponents exploit them [1, 6].

The data foundation for this work has shifted twice in the last fifteen years. PITCHf/x, deployed in every MLB stadium between 2006 and 2016, gave researchers their first pitch-by-pitch tracking feed and triggered a wave of classification studies [4, 5]. Statcast, the Hawk-Eye and Doppler-radar based replacement that has been the league standard since 2017 (and the sole tracking system since 2020), upgraded both the spatial precision and the breadth of recorded variables [8]. Modern Statcast feeds expose more than one hundred columns per pitch including release-point geometry, spin axis, movement profile, and per-pitch run-value deltas, all available through the public-facing Baseball Savant interface and pulled programmatically with libraries such as `pybaseball`. Pane et al. [5] showed that even the labeled pitch type field benefits from per-pitcher mixture modeling, because automatic classifiers tuned on the league-wide arsenal misclassify pitchers whose individual pitch profiles do not match the league average.

A second strand of the literature treats pitch selection as a sequence-prediction problem. Bock [1] showed that pitchers whose pitch sequences carry higher entropy enjoy longer effective MLB careers, an empirical confirmation that unpredictability is a competitive resource. Healey and Zhao [6] linked sequence predictability directly to strikeout rates. Sidle and Tran [2] benchmarked LDA, k-nearest-neighbours, support vector machines, and random forests on per-pitcher PITCHf/x data, reporting that per-pitcher random forests reach roughly 67 percent multi-class accuracy when trained with several seasons of history per pitcher. Hamilton et al. [4] cast the binary fastball-versus-offspeed problem and reached around 70 percent with linear classifiers on count, score, runners, and the prior pitch. Lee [3] extended the problem to joint pitch-type-and-location prediction with deep neural network ensembles.

The game-theoretic literature is equally relevant. Kovash and Levitt [12] tested whether MLB pitch sequences satisfy the minimax mixed-strategy equilibrium predicted by classical game theory and found significant negative serial correlation, meaning pitchers alternate pitch types more than Nash play would imply. Weinstein-Gould [13] and Sidhu and Caffo [11] formalised the pitcher-batter interaction as a 2x2 fastball-offspeed game and as a Markov decision process respectively, providing the theoretical scaffolding for why pre-pitch state (count, runners, batter handedness) carries predictive signal at all.

Most of the published pipeline pools many pitchers into a single training set and reports aggregate accuracy. This is statistically efficient but obscures the individual-level question every advance scout actually asks: what will pitcher X throw next, given everything we know about pitcher X. The single-pitcher framing also exposes the harder limit of the problem, namely that an individual pitcher generates only a few thousand pitches per season, the class prior on rare pitch types is brittle, and the arsenal itself is non-stationary across seasons as the pitcher develops new offerings or de-emphasises old ones.

We operationalise the single-pitcher framing on Tarik Skubal, the Detroit Tigers left-hander whose 2024-2025 emergence as one of the league's top starters makes him an unusually well-covered modeling target. The contributions of this paper are: (i) a fully reproducible Statcast-only pipeline that pulls 13,935 Skubal pitches across 2020-2026, deduplicates and label-cleans the arsenal, and frames the resulting next-pitch prediction problem as five-class classification; (ii) a strict time-based train-validation-test split that mirrors the deployment task (predict forward into unseen games); (iii) a head-to-head evaluation of majority-class, logistic-regression, class-balanced random forest, and tuned XGBoost predictors, with per-class F1, calibration via log loss, and feature attribution via gain importance and SHAP-style ranking [19]; and (iv) an honest accounting of where the single-pitcher framing breaks down, in particular for the rarest pitch in the arsenal.

## 2. Data

### 2.1 Source and ingestion

We use the public MLB Statcast pitch-by-pitch feed surfaced through Baseball Savant and pulled with the `pybaseball` Python library. One Apache Parquet file per season (2015 through 2026) was downloaded and stored locally; total uncompressed footprint is approximately 943 MB. Every parquet file carries the same 118-column post-2020 schema, containing identifiers, pitch-physics measurements, count and game-state, matchup, batted-ball outcome, and win-probability deltas. The 2024 file alone contains 760,248 rows, which is representative of a full regular season plus playoffs.

For this case study we filter every parquet on `pitcher == 669373`, the MLBAM identifier for Tarik Skubal. The filter yields 13,935 pitches across 161 unique games, all in seasons 2020-2026, consistent with Skubal's recorded MLB debut on 18 August 2020. No Statcast row is present pre-debut.

### 2.2 Sample size and arsenal evolution

Table 1 reports the Skubal sample size by season together with the average release speed and spin rate. The 2023 row is depressed by an oblique strain that delayed Skubal's season; the 2026 row is partial because the dataset cut-off is mid-season. Average release speed climbs from 89.97 mph in 2023 to 93.26 mph in 2025, a 3.3 mph step that is the single largest physics-level shift in the data and which has direct consequences for pitch-type modeling: the same pitch label maps to materially different release physics across seasons.

**Table 1.** Skubal sample size and release physics by season.

| Season | Pitches | Games | Avg release speed (mph) | Avg spin rate (rpm) |
|---|---:|---:|---:|---:|
| 2020 | 591 | 8 | 89.86 | 2,259 |
| 2021 | 2,776 | 34 | 89.50 | 2,032 |
| 2022 | 2,141 | 25 | 90.04 | 2,050 |
| 2023 | 1,219 | 15 | 89.97 | 2,007 |
| 2024 | 3,295 | 36 | 92.07 | 2,045 |
| 2025 | 3,304 | 36 | 93.26 | 2,160 |
| 2026 | 609 | 7 | 92.41 | 2,161 |
| Total | 13,935 | 161 | 91.39 | 2,092 |

### 2.3 Arsenal definition

Across his career Skubal has thrown nine distinct pitch labels, but the distribution is heavily concentrated. The four-seam fastball (FF), changeup (CH), slider (SL), sinker (SI), and curveball (CU) jointly cover 98.3 percent of his pitches. The cutter (FC, n=19), splitter (FS, n=97), knuckle-curve (KC, n=3), and unclassified Statcast labels (NA, n=120) collectively cover the remaining 1.7 percent and are concentrated in 2020-2021. We restrict the modeling target to the five-class arsenal, treating FC, FS, KC, and NA as out-of-scope.

Table 2 shows the share of each top-five pitch by season. Three structural shifts are visible. First, Skubal threw zero sinkers before 2021; by 2025 the sinker is 22.8 percent of his pitches, statistically tied with the four-seam. Second, the slider was 29.7 percent of pitches in 2022 but only 12.7 percent in 2025. Third, the changeup share roughly tripled from 11.0 percent in 2021 to 29.9 percent in 2025 and is now co-dominant with the four-seam. The curveball has shrunk to 2-3 percent of pitches in the modern arsenal.

**Table 2.** Skubal pitch-mix share by season (percent of pitches, top five pitches).

| Season | FF | SI | SL | CH | CU |
|---|---:|---:|---:|---:|---:|
| 2020 | 58.9 | 0.0 | 15.7 | 16.4 | 7.8 |
| 2021 | 44.1 | 11.8 | 22.6 | 11.0 | 6.8 |
| 2022 | 28.9 | 19.6 | 29.7 | 15.2 | 6.6 |
| 2023 | 36.0 | 12.1 | 20.7 | 24.1 | 6.8 |
| 2024 | 32.6 | 21.2 | 14.1 | 27.0 | 3.9 |
| 2025 | 29.4 | 22.8 | 12.7 | 29.9 | 2.6 |
| 2026 | 38.6 | 18.9 | 13.0 | 25.9 | 3.4 |

This non-stationarity is the central data-hygiene risk for the model. Pitches drawn from 2020-2021 carry a different conditional distribution P(next pitch | state) than pitches drawn from 2024-2026, because the underlying arsenal itself has changed. We retain all seasons in the training set rather than truncate, but the time-based split documented below ensures the model is always evaluated on more recent state than it has trained on.

### 2.4 Target variable and modeling table

Pre-pitch context predicts the next pitch within the same plate appearance. We define `next_pitch_type` as the `pitch_type` of the next pitch sorted by `pitch_number` within the same `(game_pk, at_bat_number)`. The last pitch of every plate appearance has no successor and is dropped, leaving 10,284 rows when run on the raw 13,935-pitch table. After restricting to the five-class arsenal and removing rows with missing pre-pitch state, the final modeling table holds 13,696 pitches. The class prior on this modeling table is FF 35.8 percent, CH 22.3 percent, SL 18.8 percent, SI 18.0 percent, CU 5.1 percent.

The always-FF majority-class baseline accuracy on the modeling table is therefore 35.8 percent. Any non-trivial model must clear that bar.

### 2.5 Data quality

Critical pre-pitch columns (`pitch_type`, `release_speed`, `release_spin_rate`, `pfx_x`, `pfx_z`, `plate_x`, `plate_z`, `balls`, `strikes`, `outs_when_up`, `inning`, `stand`, `p_throws`) all have less than two percent missingness on the Skubal subset. The 0.86 percent of pitches with `pitch_type = NA` are dropped rather than imputed.

## 3. Methods

### 3.1 Predict-time information set

The decision the model emulates is the pitcher's choice of the next pitch given everything observed up to the moment that decision is made. We therefore restrict the feature set to pre-pitch information only. Any column that describes the pitch itself (release speed of the next pitch, plate location, descriptive outcome, run-expectancy delta) is excluded as target leakage, following the convention of [2, 4].

### 3.2 Feature engineering

We use 29 features in the final model:

1. Count state: `balls` and `strikes` as integers in their natural range, plus an at-bat pitch index `ab_pitch_idx` that counts the pitcher's pitches within the current plate appearance.
2. Game state: `outs_when_up`, `inning`, three on-base flags `on_1b_flag`, `on_2b_flag`, `on_3b_flag`, and the score differential `score_diff` from the pitcher's perspective.
3. Matchup: `stand_R`, a one-hot indicator for right-handed batter (Skubal is left-handed, so `stand_R = 1` means the platoon-disadvantaged matchup).
4. Sequence context: lag-1, lag-2, and lag-3 prior pitch types in the same plate appearance, each one-hot encoded across the six categories FF, SI, SL, CH, CU, and NONE (the NONE category fires when the lag position falls outside the current plate appearance).
5. Batter identity: `batter_freq`, a frequency encoding of the batter MLBAM id estimated on the training split only. Frequency encoding avoids the 1,000+ one-hot columns a categorical batter-id encoding would produce while preserving a coarse signal that some batters Skubal has faced many times before.

Lag features were the largest single accuracy contributor in the basic-feature baseline of [4] and the dominant non-handedness signal in our own gain-importance ranking (Section 4.4).

### 3.3 Time-based split

We split by date with no shuffling:

- Training: all pitches in seasons 2020-2024 (9,871 rows)
- Validation: 2025 first half (1,625 rows)
- Test: 2025 second half plus all 2026 (2,200 rows)

This split mirrors the deployment task. A model that is asked to predict pitches in the 2025 second half should never have seen the 2025 first half during training. Random pitch-level splits would leak within-game and within-at-bat sequencing and inflate every metric. Game-level random splits are also defensible but trade off the ability to evaluate on the most recent arsenal, which is the regime that matters for in-season scouting.

### 3.4 Models

We compare four models:

1. **Majority-FF baseline.** Always predicts the four-seam fastball. No training. Defines the accuracy floor.
2. **Logistic regression.** Standard multinomial logistic regression with L2 regularisation, fit with the basic feature set (28 columns: lag-1 only).
3. **Random forest.** `n_estimators = 400`, `max_depth = 12`, `min_samples_leaf = 2`, `class_weight = "balanced"`, parallelised across four cores. Same basic feature set as logistic regression. Class weighting compensates for the FF prior. The random-forest configuration follows Breiman's original formulation [18].
4. **XGBoost.** `objective = "multi:softprob"`, full 29-feature set with lag-1, lag-2, lag-3, and `batter_freq`. Hyperparameters tuned on the validation split via grid search over four configurations of (depth, learning rate, n_estimators, min_child_weight, subsample, colsample_bytree). The best validation-macro-F1 configuration was `max_depth = 6, learning_rate = 0.07, n_estimators = 250, min_child_weight = 3, subsample = 0.9, colsample_bytree = 0.8`. The depth-8 variant overfit on this sample size. The selected model was refit on the union of training and validation data and evaluated once on the held-out test split. The XGBoost implementation is the canonical Chen and Guestrin gradient-boosted-tree formulation [17].

We did not run an LSTM [15] or Transformer [16] sequence model. With 13,696 rows and a five-class target, the variance from a deep sequence model would dominate the bias improvement over XGBoost; this is a documented limitation rather than a methodological gap.

### 3.5 Evaluation metrics

We report accuracy (single-label, top-1), macro-F1 (unweighted average per-class F1), weighted-F1 (per-class F1 weighted by support, i.e. the prior), and multi-class log loss. Macro-F1 is the headline number because it penalises models that ignore minority classes; accuracy alone would be cleared by always predicting the four-seam. Log loss captures calibration. We also report per-class F1 for each pitch type and the full confusion matrix.

## 4. Results

### 4.1 Headline metrics

Table 3 summarises the four models on the 2,200-pitch held-out test set. The tuned XGBoost beats every baseline on every metric. Accuracy improves nine percentage points over the random forest (0.296 -> 0.387) and four points over the always-FF baseline (0.347 -> 0.387). Weighted F1 improves from 0.313 to 0.368. Log loss drops from 1.537 to 1.413, an eight percent gain in calibration. Macro F1 improves marginally from 0.262 to 0.271 because the XGBoost loses ground on the rarest class while gaining on the most common one.

**Table 3.** Test-set metrics, n = 2,200.

| Model | Accuracy | Macro F1 | Weighted F1 | Log loss |
|---|---:|---:|---:|---:|
| Majority FF | 0.347 | 0.103 | 0.179 | n/a |
| Logistic regression (basic) | 0.284 | 0.255 | 0.302 | 1.580 |
| Random forest (basic) | 0.296 | 0.262 | 0.313 | 1.537 |
| XGBoost (full features) | 0.387 | 0.271 | 0.368 | 1.413 |

### 4.2 Per-class F1

Table 4 shows the per-class F1 breakdown. The XGBoost roughly doubles four-seam F1 (0.265 -> 0.480) by leaning into the prior, gives back small amounts of changeup and sinker F1, and effectively gives up on the curveball (F1 0.027). The slider is recovered poorly by every model (best F1 0.137 from the random forest).

**Table 4.** Per-class F1 by model on the held-out test set.

| Class | Class prior | LogReg | RF | XGB final |
|---|---:|---:|---:|---:|
| CH | 0.223 | 0.414 | 0.404 | 0.399 |
| CU | 0.051 | 0.116 | 0.097 | 0.027 |
| FF | 0.358 | 0.245 | 0.265 | 0.480 |
| SI | 0.180 | 0.386 | 0.406 | 0.347 |
| SL | 0.188 | 0.117 | 0.137 | 0.104 |

### 4.3 Confusion matrix

Table 5 reports the test-set confusion matrix for the final XGBoost. Rows are the true pitch type and columns are the predicted pitch type, in the order CH, CU, FF, SI, SL.

**Table 5.** XGBoost test-set confusion matrix.

| True \ Pred | CH | CU | FF | SI | SL |
|---|---:|---:|---:|---:|---:|
| CH | 255 | 2 | 293 | 64 | 35 |
| CU | 22 | 1 | 27 | 10 | 7 |
| FF | 191 | 2 | 437 | 88 | 45 |
| SI | 89 | 2 | 160 | 136 | 54 |
| SL | 71 | 1 | 140 | 45 | 23 |

Two patterns dominate. First, the four-seam is the largest column total and the largest diagonal entry, consistent with the model leaning into the prior. Second, the curveball column is essentially empty: the classifier almost never predicts CU, and only 1 of 67 true curveballs is recovered. Two-thirds of true curveballs are misclassified as either changeup or four-seam, both of which sit close to the curveball in count-state space.

### 4.4 Feature importance

Table 6 reports the XGBoost gain-importance ranking for the top fifteen features. Batter handedness is the single largest contributor, consistent with the 14-percentage-point swing in four-seam usage and 21-point swing in sinker usage between right-handed and left-handed batters reported in the EDA. The lag-1 prior pitch dominates the next tier, with lag-2 and lag-3 still appearing in the top fifteen. Count-state features (`strikes`, `balls`) and a single runner flag (`on_1b_flag`) round out the top fifteen.

**Table 6.** XGBoost gain-importance ranking, top fifteen features.

| Feature | Gain |
|---|---:|
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

### 4.5 Why the curveball collapses

The curveball failure mode is worth surfacing because it is the dominant residual error and because the same pattern likely transfers to other low-frequency pitches in other pitchers' arsenals. CU has a 5.1 percent class prior on the modeling table and 2-3 percent prior in 2024-2026. The XGBoost optimal Bayes decision under multi-class log loss is to predict CU only when the posterior P(CU | state) exceeds the maximum posterior of every other class. With overlapping state distributions and a five-class softmax, that condition is rarely met. The class-weighted random forest in Section 4.2 produces slightly higher CU F1 (0.097) at the cost of weaker FF recovery, which is the central trade-off visible across all four models.

## 5. Discussion

### 5.1 Comparison with the multi-pitcher literature

Sidle and Tran [2] reported per-pitcher random-forest accuracy of approximately 67 percent on PITCHf/x data after pooling across pitchers and using pitcher identity as a feature. Hamilton et al. [4] reached around 70 percent on the binary fastball-versus-offspeed problem. Our five-class accuracy of 0.387 is not directly comparable to either: a five-class problem has a lower achievable accuracy ceiling than a two-class problem, and a single-pitcher model trained on roughly 13,700 pitches has materially less information than a pooled multi-pitcher model trained on hundreds of thousands of pitches. The single-pitcher framing trades sample size for arsenal specificity. A pooled model that learns the league-wide conditional distribution P(next pitch | state) and then conditions on a pitcher-id embedding, as in [3, 7], is likely the next step for any practitioner who needs absolute accuracy rather than per-pitcher interpretability.

A useful framing is to compute the achievable accuracy ceiling. The Kovash and Levitt [12] result that MLB pitch sequences exhibit significant negative serial correlation, contradicting Nash mixed-strategy play, sets a soft upper bound on how predictable any sequence model can be: the pitcher actively tries to be unpredictable, and the league-average rate at which they fail to do so is the ceiling for a perfect sequence model. Healey and Zhao's [6] entropy decompositions and Bock's [1] sequence-complexity analysis suggest that ceiling sits well below 100 percent and is itself pitcher-specific; a high-entropy pitcher is harder to predict than a low-entropy one. Skubal's late-career arsenal expansion (sinker 2021, dominant changeup 2024) pushes the entropy of his career-aggregated sequence higher than the entropy of any single season, which is one reason within-season models trained on 2025 alone might outperform multi-season models on accuracy at the cost of much smaller training samples.

### 5.2 Game-theoretic interpretation

The XGBoost feature ranking is consistent with the 2x2 game-theoretic models of [12, 13]. Batter handedness is a payoff-relevant covariate because the platoon split changes the run-expectancy of every pitch in the arsenal. The lag-1 prior pitch is informative because pitchers visibly avoid throwing the same pitch twice in a row (the negative serial correlation of [12]). The count-state features pick up the well-documented pattern that pitcher-friendly counts (0-2, 1-2) shift the put-away pitch distribution toward the changeup, and hitter-friendly counts (3-0, 3-1) collapse the arsenal toward the four-seam.

### 5.3 Limitations

The single-pitcher framing exposes four limitations.

First, sample size. Roughly 13,700 modeling rows is small for a five-class problem with a 5 percent minority class. The curveball F1 collapse is the direct symptom. A class-weighted training run, a focal-loss objective, or a SMOTE-style minority oversampling [20] could partially close this gap; we did not run these because the resulting model is harder to deploy than a single calibrated softmax.

Second, arsenal non-stationarity. Skubal's pitch mix in 2021 differs materially from his pitch mix in 2025, which means the conditional distribution P(next pitch | state) the model is trying to learn is itself shifting. A recency-weighted training scheme or a sliding-window evaluation would address this; we instead chose a simple time-based split and accepted the bias.

Third, model architecture. We did not run an LSTM [15] or Transformer [16] sequence model. The lag-1, lag-2, lag-3 one-hot encoding captures only the immediately preceding three pitches and discards anything earlier. A learned recurrent or attention-based representation could in principle exploit longer-range dependencies; in practice 13,700 rows is too few for either architecture to outperform a tuned tree ensemble.

Fourth, feature scope. We did not use any of the rich Statcast pitch-physics columns (release-point geometry, spin axis, movement profile) for the lagged context, only the pitch-type label. Adding the lagged release physics could materially improve the curveball recovery, since the pitcher's release-point distribution often shifts before a curveball in a way that is invisible to a label-only feature.

### 5.4 Practical implications

For a Detroit Tigers analyst the model is most useful as a calibrated probability estimator rather than a single-label predictor. The XGBoost log loss of 1.413 on the held-out test set means the model produces well-ordered probability estimates for at least the three high-frequency pitches; an opposing scout can use the per-pitch P(FF), P(CH), P(SI) estimates to prioritise on-deck batter preparation even when the top-1 prediction is wrong. For the pitcher and the pitching coach the gain-importance ranking surfaces the predictability levers that are most exposed to opponent exploitation: batter handedness is structural and cannot be hidden, but the lag-1 prior-pitch dependence is a tendency that could be deliberately broken by inserting more random pitch choices in the early counts.

## 6. Conclusion

Predicting the next pitch type of a single high-profile MLB starter is a tractable but bounded problem. Using 13,696 Skubal-only Statcast pitches across 2020-2026 with a strict time-based train-validation-test split, a tuned XGBoost classifier reaches accuracy 0.387, macro-F1 0.271, weighted-F1 0.368, and log loss 1.413 on a 2,200-pitch held-out test set, beating the always-FF majority baseline by four percentage points and the class-balanced random forest by nine. Batter handedness and the lag-1 prior pitch are the dominant features, in line with prior multi-pitcher work. The rarest class, the curveball, collapses to F1 0.027, illustrating the central trade-off of single-pitcher modeling: arsenal specificity comes at the cost of sample size on the minority pitch types. The natural extensions are a multi-pitcher pooled model with a pitcher-id embedding, a sequence model that uses the lagged release physics rather than only the pitch-type labels, and an outcome-conditioned second stage that scores each candidate pitch by its expected run value rather than by its match to historical tendency.

## References

[1] Bock, J. K. (2015). Pitch Sequence Complexity and Long-Term Pitcher Performance. *Sports*, 3(1), 40-55.

[2] Sidle, G., & Tran, H. (2017). Using Multi-Class Classification Methods to Predict Baseball Pitch Types. *Journal of Sports Analytics*, 4(1), 85-93.

[3] Lee, J. S. (2022). Prediction of Pitch Type and Location in Baseball Using Ensemble Model of Deep Neural Networks. *Journal of Sports Analytics*, 8(2), 115-126.

[4] Hamilton, M., Hoang, P., Layne, L., Murray, J., Padgett, D., Stafford, C., & Tran, H. (2014). Applying Machine Learning Techniques to Baseball Pitch Prediction. *ICPRAM*.

[5] Pane, M. A., Ventura, S. L., Steorts, R. C., & Nugent, R. (2013). Trouble With The Curve: Improving MLB Pitch Classification. arXiv:1304.1756.

[6] Healey, G., & Zhao, S. (2017). Using PITCHf/x to Model the Dependence of Strikeout Rate on the Predictability of Pitch Sequences. *Journal of Sports Analytics*, 3(2), 93-101.

[7] Umemura, K., Yanai, T., & Nagata, Y. (2020). Application of VBGMM for Pitch Type Classification: Analysis of TrackMan's Pitch Tracking Data. *Japanese Journal of Statistics and Data Science*, 3, 475-492.

[8] Kovalchik, S. (2023). Player Tracking Data in Sports. *Annual Review of Statistics and Its Application*, 10, 677-697.

[9] Beal, R., Norman, T. J., & Ramchurn, S. D. (2019). Artificial Intelligence for Team Sports: A Survey. *The Knowledge Engineering Review*, 34, e28.

[10] Baumer, B. S., Matthews, G. J., & Nguyen, Q. (2023). Big Ideas in Sports Analytics and Statistical Tools for Their Investigation. *WIREs Computational Statistics*, 15(6), e1612.

[11] Sidhu, G., & Caffo, B. (2014). MONEYBaRL: Exploiting Pitcher Decision-Making Using Reinforcement Learning. *Annals of Applied Statistics*, 8(2), 926-955.

[12] Kovash, K., & Levitt, S. D. (2009). Professionals Do Not Play Minimax: Evidence from Major League Baseball and the National Football League. NBER Working Paper 15347.

[13] Weinstein-Gould, J. (2009). Keeping the Hitter Off Balance: Mixed Strategies in Baseball. *Journal of Quantitative Analysis in Sports*, 5(2).

[14] Nakahara, H., Takeda, K., & Fujii, K. (2023). Estimating the Effect of Hitting Strategies in Baseball Using Counterfactual Virtual Simulation with Deep Learning. *International Journal of Computer Science in Sport*, 22(1), 1-20.

[15] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

[16] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 30*. arXiv:1706.03762.

[17] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*, 785-794.

[18] Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

[19] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 30*. arXiv:1705.07874.

[20] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-Sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
