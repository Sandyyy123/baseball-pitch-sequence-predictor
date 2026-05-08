# Validation Report - Project 03 (MLB Pitch / Tarik Skubal)

## Compact summary

**Overall status: PASS-WITH-WARNINGS.**

The manuscript is a complete, IMRaD-conformant 4,704-word write-up with all 20 inline citations resolving to entries in `manuscripts/references.md`; the spot-check of 5 DOIs against `api.crossref.org` returned HTTP 200 with title matches on all 5. The presentation HTML is fully self-contained (zero external `http` references). Both notebooks parse as valid JSON, both Python scripts in `src/` parse cleanly, and zero em-dash characters appear anywhere in the audited surface. Saved model artefacts (`skubal_xgb.pkl`, `metrics.json`) are present in `deliverables/`. The warnings concern layout drift from the standard scaffold rather than substance: this project has no `checkpoint.json`, no `model_baseline.py` / `model_advanced.py` (model code lives in `notebooks/03_modeling.ipynb`), `references.md` lives under `manuscripts/` not `reports/`, and the brief is `brief.pdf` (no `brief.md`).

---

## Findings

### 1. Notebook validity
- [PASS] `notebooks/01_eda.ipynb` parses as valid JSON (28 cells).
- [PASS] `notebooks/03_modeling.ipynb` parses as valid JSON (19 cells).
- [WARN] No `notebooks/01_EDA.ipynb` (capitalised) - file is lowercase `01_eda.ipynb`. Cosmetic deviation from scaffold.

### 2. Python script syntax
- [WARN] `src/model_baseline.py` and `src/model_advanced.py` do NOT exist. Model code lives in `notebooks/03_modeling.ipynb` instead. Per QA spec rule (#1-#8 may have been executed earlier), this is a layout deviation, not a substantive failure.
- [PASS] `src/_extract_metrics.py`: `ast.parse` succeeds.
- [PASS] `src/_inspect.py`: `ast.parse` succeeds.
- [PASS] `src/_summary.py`: `ast.parse` succeeds.

### 3. Manuscript word count
- [PASS] `wc -w manuscripts/manuscript.md` = 4,704 words. Inside the 4,000-5,000 target window.

### 4. Self-contained HTML
- [PASS] `grep -E 'href="http|src="http' deliverables/presentation.html` returns 0 hits. Presentation is fully inline.

### 5. IMRaD completeness
- [PASS] All required sections present: Title (L1), Abstract, 1. Introduction, 2. Data + 3. Methods (split between data ingestion and modeling methodology, both substantive), 4. Results, 5. Discussion, 6. Conclusion, References.
- [PASS] References block holds 20 numbered entries.

### 6. Method drift
Methods named in the manuscript Methods section: Majority-FF baseline, multinomial logistic regression with L2 regularisation, class-balanced random forest, XGBoost (`multi:softprob`, tuned hyperparameters), lag-1/lag-2/lag-3 prior-pitch one-hot encoding, frequency-encoded batter id (`batter_freq`), strict time-based train/validation/test split, macro-F1 / weighted-F1 / log-loss / accuracy reporting, gain-importance and SHAP-style attribution.
- [PASS] All four model classes verified present in `notebooks/03_modeling.ipynb`: `LogisticRegression`, `RandomForestClassifier`, `XGBClassifier`, `class_weight`, `multi:softprob`, `batter_freq`, lag features, `log_loss`, `f1_score`, `confusion_matrix`, SHAP. No drift.
- [WARN] Methods are implemented in the modeling notebook rather than in the canonical `model_baseline.py` / `model_advanced.py` scripts. Reproducibility fine via notebook kernel; pipeline scripts would be cleaner.

### 7. Citation drift
- [PASS] 20 inline citation numbers found ([1] through [20]); each maps to the correspondingly numbered entry in `manuscripts/references.md`. Zero orphans, zero unused references.

### 8. Re-verify 5 random references (CrossRef live)
All 5 returned HTTP 200 with matching titles:
- [PASS] `10.3390/sports3010040` -> "Pitch Sequence Complexity and Long-Term Pitcher Performance" (Bock 2015, ref [1]).
- [PASS] `10.3233/jsa-170171` -> "Using multi-class classification methods to predict baseball pitch types" (Sidle and Tran 2017, ref [2]).
- [PASS] `10.5220/0004763905200527` -> "Applying Machine Learning Techniques to Baseball Pitch Prediction" (Hamilton et al. 2014, ref [4]).
- [PASS] `10.1146/annurev-statistics-033021-110117` -> "Player Tracking Data in Sports" (Kovalchik 2023, ref [8]).
- [PASS] `10.1145/2939672.2939785` -> "XGBoost" (Chen and Guestrin 2016, ref [17]).

### 9. Em-dash scan
- [PASS] Total em-dash count across `manuscript.md`, both notebooks, `references.md`, all `src/*.py`, and `presentation.html` = 0.

### 10. AI-tell scan
- [PASS] `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude'` over the project tree returned zero hits.

### 11. Checkpoint schema
- [WARN] `checkpoint.json` does NOT exist at the project root. A `.tmp_summary.json` exists with EDA-side aggregates (per-season pitch counts, release-speed stats), but it lacks the `project_number`, `title`, `methodology`, `status` fields required by the QA spec. No artefact in the project advertises an explicit `status` field.

### 12. Saved model artefacts (project #1-#8 expectation)
- [PASS] `deliverables/skubal_xgb.pkl` (4.3 MB) present.
- [PASS] `deliverables/metrics.json` (5.5 KB) present.
- [PASS] `deliverables/presentation.html` (43 KB, self-contained) present.

---

## Output file path
`/root/AI/liora_projects/03_skubal_pitch/validation_report.md`

## Top 3 findings
1. Substance is solid: 4,704-word IMRaD manuscript, all 20 citations resolve, 5/5 DOIs live-verified on CrossRef, em-dash count = 0, AI-tell count = 0, presentation HTML inline-only, saved XGBoost model and metrics.json present.
2. Layout drift from QA scaffold: no `checkpoint.json`, no `model_baseline.py` / `model_advanced.py` (models live in `notebooks/03_modeling.ipynb`), `references.md` is in `manuscripts/` not `reports/`, brief is PDF only. None of these block the deliverable but they fail strict scaffold checks.
3. No method drift and no orphan citations - the modeling notebook implements every method named in the Methods section (LogReg, RF, XGBoost, class weighting, lag features, SHAP, log-loss / F1 reporting).

## Blockers
None.

Role A (VALIDATOR) complete.
