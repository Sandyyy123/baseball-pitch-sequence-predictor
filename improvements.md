# Improvements - Project 03 (Tarik Skubal Pitch Prediction)

## Summary

Solid single-pitcher pipeline (13,696 rows, 5-class XGBoost, time-based split, 0.387 acc / 0.271 macro-F1). Headline weakness: macro-F1 is barely above the basic RF (0.271 vs 0.262) because the curveball collapses to F1 0.027 and the slider stays at 0.10. The biggest leverage is not deeper architectures but pooled multi-pitcher training with a pitcher-id embedding plus pitch-physics lag features, both of which the manuscript itself flags as future work.

## Top recommendation

**Switch from Skubal-only training to a pooled multi-pitcher LightGBM/XGBoost with a pitcher-id target-mean encoding (or a learned embedding) and add lag-1 to lag-3 pitch-physics features (release_speed, spin_rate, pfx_x, pfx_z) instead of pitch-type one-hots only.** The data already on disk (statcast_2015-2026.parquet, ~940 MB) supports this with no new download. Sidle and Tran [2] showed pooled per-pitcher RFs reach roughly 67 percent accuracy vs the 0.387 here; a pitcher-id-conditioned pooled model captures the within-Skubal signal while borrowing strength on rare classes (CU, SL) from other left-handed starters with similar arsenals (e.g. Sale, Crochet, Snell). Realistic target: macro-F1 0.35-0.40 and CU F1 0.15+, without changing the deployment surface.

## Weaknesses and actionable fixes

### 1. Curveball collapse is unaddressed - HIGH

CU F1 is 0.027 (1 of 67 recovered). The manuscript names class-weighting, focal loss, and SMOTE [20] as options but did not run any. Concrete next step: refit XGBoost with `sample_weight = compute_sample_weight('balanced', y_train)` and a `multi:softprob` + focal-loss custom objective (alpha=0.25, gamma=2). Compare CU F1 and FF F1 on the validation split; pick the operating point that keeps FF F1 above 0.40 and CU F1 above 0.10. This is a one-hour change with a directly visible payoff in Table 4.

### 2. No pitch-physics lag features - HIGH

Section 5.3 limitation 4 names this gap explicitly. Lag-1 to lag-3 should carry numeric `release_speed`, `release_spin_rate`, `pfx_x`, `pfx_z`, `release_pos_x`, `release_pos_z` from the previous pitch in the same plate appearance, not only the pitch-type label. Release-point clustering before a curveball is a documented tell. Add 18 numeric lag columns; refit XGBoost; expect CU F1 +0.05 to +0.10 because the release-physics signature of the previous pitch reveals the pitcher's set-up sequence.

### 3. No probability calibration step - HIGH

The manuscript's Section 5.4 sells the model as "a calibrated probability estimator" but no calibration was run. Log loss 1.413 is reported on raw softmax scores. Wrap the final XGBoost in `CalibratedClassifierCV(method='isotonic', cv='prefit')` fit on the validation split, then re-evaluate log loss and Brier score on the test split. Add a reliability diagram per class. Without this, the "calibrated" claim is unsupported.

### 4. Reproducibility scaffolding missing - HIGH

No `requirements.txt`, no `environment.yml`, no `README.md`, no `.gitignore` in the project root. Seeds are set in the modeling notebook (random_state=42) but not pinned for numpy/python hash. Brief explicitly requires "associated GitHub". Create `requirements.txt` (pin pandas, scikit-learn, xgboost, pybaseball, pyarrow versions), a top-level `README.md` with one-paragraph problem statement and `make all` style commands, and a `.gitignore` that excludes `data/*.parquet` (the 940 MB local cache should never be committed). Set `PYTHONHASHSEED=0` and `np.random.seed(42)` at the top of every notebook.

### 5. No ablation table - MEDIUM

The manuscript reports four models but does not isolate the contribution of each new feature. Add an ablation: (a) basic 28 features, (b) +lag-2/lag-3, (c) +batter_freq, (d) full. Report accuracy, macro-F1, and log loss for each. This converts the "we added these and the score went up" narrative into a testable claim and makes the paper publishable rather than a class report.

### 6. Single train/val/test split, no temporal CV - MEDIUM

Reported metrics rest on one held-out window (2025 H2 + 2026, n=2,200). Time-series cross-validation with rolling origins (e.g. expanding window, 6 folds: train through 2022, eval 2023; ... train through 2025 H1, eval 2025 H2) gives a confidence interval on the headline numbers and detects whether the 0.387 accuracy is a lucky split. Use `sklearn.model_selection.TimeSeriesSplit(n_splits=6)` over `game_date`-sorted rows.

### 7. Slider F1 0.10 is unexplained - MEDIUM

SL has 18.8 percent prior, comparable to SI (18.0 percent, F1 0.35), but SL F1 is 0.10. The manuscript does not investigate. Likely cause: SL-vs-SI ambiguity on count and handedness state. Action: run a binary SL-vs-rest probit using the same features, inspect SHAP [19] on the SL margin, and report the 5 most-confused (true, predicted) pairs. If SL is structurally indistinguishable from SI on pre-pitch state, that is a publishable finding, not a bug.

### 8. Presentation has zero embedded visuals - MEDIUM

`grep -cE "<img|<svg|chart|canvas" presentation.html` returns 0. Slide deck for a client/business audience must show the confusion matrix as a heatmap, the per-class F1 bar chart (Table 4), and the season pitch-mix shift (Table 2) as a stacked area plot. Inline SVG (no external assets, satisfies the self-contained-HTML rule) is the right choice. Use matplotlib `savefig(..., format='svg')` and inline the SVG bodies.

### 9. No baseline beyond majority-FF - LOW

A "current-pitch + count" lookup-table baseline (predict the modal next pitch given the previous pitch type and balls/strikes count) is mentioned in the EDA but not measured. Add it to Table 3 as a fifth row; expect accuracy near 0.36-0.38, which would clarify whether XGBoost adds anything beyond a memorisation table.

### 10. No within-season recency weighting - LOW

Section 5.3 limitation 2 flags arsenal non-stationarity (sinker share 0 percent in 2020 to 22.8 percent in 2025) but does not act on it. Train XGBoost with `sample_weight = exp(-(2026 - season) * 0.3)` so 2020 pitches get weight ~0.18 and 2025 pitches get weight ~1.0. Compare to the unweighted model on test. If the recency-weighted model wins on 2026 alone, that is the in-season-deployment configuration.
