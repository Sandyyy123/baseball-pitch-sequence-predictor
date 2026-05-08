# Additional References - MLB Pitch-Type Prediction (Tarik Skubal Case Study)

Independent literature scout output. Every entry below was resolved live against `https://api.crossref.org/works/{doi}` on 2026-05-08; entries that did not resolve were dropped, not patched. Volume / issue / page numbers are intentionally omitted (Author, Title, Journal, Year, DOI only).

These references are NEW relative to `manuscripts/references.md` (the existing 20-entry bibliography). They prioritise 2024-2026 work in pitch-type prediction, single-pitcher modeling, Statcast-based outcome models, recent imbalanced-classification methods, and pitcher-state context (pitch-clock, foreign-substance policy, fatigue, biomechanics) that the current manuscript would benefit from acknowledging.

## State-of-the-art callout: gaps relative to the current bibliography

The current `references.md` ends at 2023 for everything except generic ML methods (XGBoost, SHAP, SMOTE, LSTM, Transformer). Five concrete SOTA gaps that the project should cite:

1. **Player-specific deep sequence models for pitch type** - the manuscript's core claim is single-pitcher modeling, but the existing bibliography stops at multi-pitcher Statcast/PITCHf/x classifiers (Sidle & Tran 2017, Lee 2022, Hamilton 2014). Chang et al. 2026 (CNN-LSTM for player-specific pitch type from video sequences, DOI 10.3390/asi9040075) is the direct 2026 SOTA on the same single-pitcher question and is missing.
2. **Skeleton / pose-based pitch-type classifiers (ST-GCN family)** - Huesca-Flores et al. 2026 introduce skeleton-based GCN pipelines (TKA-STAGCN and Projection-Gated ST-GCN, DOIs 10.1007/s11042-026-21476-3 and 10.3390/engproc2026123003) for the same five-class arsenal. The manuscript treats the problem as pre-pitch tabular only, but should at minimum cite these as alternative-modality SOTA.
3. **Temporal Fusion Transformer applied to MLB pitcher performance** - Lee & Kim 2025 (DOI 10.32604/cmc.2025.065413) is the 2025 reference for a deep sequence model on MLB pitcher data, directly comparable to the manuscript's tabular XGBoost; absent from the current bibliography.
4. **Context-enhanced deep learning for pitch location prediction** - Moore et al. 2025, *Sports Engineering*, DOI 10.1007/s12283-025-00497-5. The manuscript only models pitch type; the natural next step is joint type-and-location and this 2025 paper is the canonical citation.
5. **Data-driven spatiotemporal-cue framework for next-pitch prediction** - Takamido et al. 2025 (DOI 10.1101/2025.10.29.685437) operationalises pre-pitch spatiotemporal cues (release point, pitch-flight kinematics) for next-pitch prediction. The manuscript explicitly lists pre-pitch state as the information set; this paper is the closest 2025 mirror.

A secondary gap is recent multi-class imbalanced-classification methodology (the manuscript cites Chawla 2002 SMOTE only). MKC-SMOTE (Wang & Awang 2024), TS-SMOTE (Song & Yang 2025), and FS-SMOTE (Huang 2025) are the modern benchmarks for the curveball-collapse problem the paper documents in Section 4.

## Architectures and algorithms (2024-2026)

Chang C, Wei C, Li H, Hsiao S. A CNN-LSTM Framework for Player-Specific Baseball Pitch Type Prediction from Video Sequences. Applied System Innovation. 2026. DOI:10.3390/asi9040075

Huesca-Flores S, Benitez-Garcia G, Juarez-Sandoval O, Takahashi H, Nakano-Miyatake M. TKA-STAGCN: a skeleton-based graph convolutional network with temporal attention for baseball pitch type classification. Multimedia Tools and Applications. 2026. DOI:10.1007/s11042-026-21476-3

Huesca-Flores S, Benitez-Garcia G, Juarez-Sandoval O, Takahashi H, Perez-Meana H, Nakano-Miyatake M. From Pose to Pitch: Classifying Baseball Pitch Types with Projection-Gated ST-GCN. First Summer School on Artificial Intelligence in Cybersecurity. 2026. DOI:10.3390/engproc2026123003

Lin Z, Wu K. Pitch Type Prediction in MLB: A Machine Learning and Deep Learning Approach. Advances in Transdisciplinary Engineering. 2025. DOI:10.3233/atde251162

Kim J, Lee S. Comparative Analysis of Machine Learning-Based Pitch Type Prediction Models for Major League Baseball Left- and Right-Handed Pitchers: Focusing on the 2024 Season. Korean Journal of Security Convergence Management. 2025. DOI:10.24826/kscs.14.11.3

Lee W, Kim J. Pitcher Performance Prediction Major League Baseball (MLB) by Temporal Fusion Transformer. Computers, Materials & Continua. 2025. DOI:10.32604/cmc.2025.065413

Moore R, Gurchiek R, Avedesian J. A context-enhanced deep learning approach to predict baseball pitch location from ball tracking release metrics. Sports Engineering. 2025. DOI:10.1007/s12283-025-00497-5

Takamido R, Suzuki C, Nakamoto H. A data-driven analysis of spatiotemporal cues and experience accumulation effects for pitch type prediction. bioRxiv. 2025. DOI:10.1101/2025.10.29.685437

## Outcome, win-probability, and team-level baseball models (2024-2026)

Ko K, Lee H, Go S. Calculating Win Probability in Korean Professional Baseball Using Deep Learning. Asia-pacific Journal of Convergent Research Interchange. 2024. DOI:10.47116/apjcri.2024.11.06

Park K, Lim H, Lee J, Suh B. Enhancing Auto-Generated Baseball Highlights via Win Probability and Bias Injection Method. Proceedings of the CHI Conference on Human Factors in Computing Systems. 2024. DOI:10.1145/3613904.3642021

Ahn D, Kim J, Son J, Kim K. MLB's on-base and out prediction model by pitcher and batter type using machine learning. Korean Journal of Sports Science. 2024. DOI:10.35159/kjss.2024.2.33.1.655

Pandey D, Gupta R. EFSM-MLB: An Ensemble Feature Selection Model for Better Outcome Prediction in Major League Baseball Using Filter and Embedded Methods. International Journal of Electronics and Communication Engineering. 2024. DOI:10.14445/23488549/ijece-v11i5p105

Bae J, Chiu B. Machine Learning-Based Classification of Team Playoff Advancement Using Pitching Performance Metrics in Korean Professional Baseball. Applied Sciences. 2026. DOI:10.3390/app16052215

McBride M. Measuring Pitcher Production Fairly in Baseball Using the Shapley Value. Games. 2026. DOI:10.3390/g17020015

Zhang Q, Chang C, Shang C, Chang H, Roy D. Capturing captivating moments: a multi-model approach for identifying baseball strikeout highlights. Signal, Image and Video Processing. 2025. DOI:10.1007/s11760-024-03805-x

## Game-theoretic and sequencing studies (2024-2026)

Hsiao S, Hu S, Lin M, Weng W. Do Professional Baseball Players Play Mixed Strategies? Evidence from MLB. SSRN Electronic Journal. 2024. DOI:10.2139/ssrn.4691177

Bushek N, Erickson S. Weighted dyadicity for major league baseball player transaction networks. Journal of Sports Analytics. 2025. DOI:10.1177/22150218251326427

Ortega R, Levine R, Osborne J. Double steals in Major League Baseball. Journal of Sports Analytics. 2026. DOI:10.1177/22150218251414487

## Datasets, tracking, and Statcast-derived measurements (2024-2026)

Pifer N. Pitch-level college baseball data captured by optical tracking technology. Data in Brief. 2024. DOI:10.1016/j.dib.2024.111049

Kabra A. Evaluating Pitcher Fatigue Through Spin Rate Decline: A Statcast Data Analysis. PARIPEX Indian Journal of Research. 2025. DOI:10.36106/paripex/0900292

King B, Bailey E, Aspang J, Hammond K, Danilkowicz R. Statcast-based evaluation of postoperative performance in Major League Baseball position players following ulnar collateral ligament surgery. Clinics in Shoulder and Elbow. 2026. DOI:10.5397/cise.2025.01361

## Pitcher-state context: pitch clock, foreign-substance policy, perception (2024-2026)

Nichols L. From Pastoral to Precise: Fan Discourse on the Major League Baseball Pitch Clock. Sociology of Sport Journal. 2026. DOI:10.1123/ssj.2024-0249

Kriter A, Sommers P. How Did the Pitch Clock Impact the 2023 Major League Baseball Season? Journal of Student Research. 2024. DOI:10.47611/jsr.v13i4.2664

Samborski S, ElAttrache N, Karnyski S, Bergeron B, Ladnier K, Banffy M. The Effect of the 2021 MLB Foreign Substance Policy on Pitcher Injury Rates and Statistics. Orthopaedic Journal of Sports Medicine. 2025. DOI:10.1177/23259671251346984

Besler Z, Muller S, Chua R, Hodges N. Right and left-handed pitch-type recognition among hitters and pitchers in baseball: Testing the motor simulation hypothesis. PsyArXiv. 2025. DOI:10.31234/osf.io/5rpg2_v2

## Pitching biomechanics and trajectory models (2024-2026)

Lozowski B, Wang C, Oliver G. Pitching kinematics have direct and indirect effects on pitch location in NCAA baseball. International Journal of Sports Medicine. 2024. DOI:10.1055/a-2468-5645

Mine K, Jones M, Saunders S, Onofrio B, Crowther R, Milanese S. Relationships between upper trunk rotation kinematics and arm fatigue after repetitive pitching among baseball pitchers. Sports Biomechanics. 2024. DOI:10.1080/14763141.2024.2431901

Ishigaki T, Kurisuga Y, Sato R, Furuto I, Kimura R, Akuzawa H, Sekine C, Hirabayashi R. Changes in glenohumeral range of motion by repetitive pitching and their relationship with arm speed during pitching. Sports Biomechanics. 2025. DOI:10.1080/14763141.2025.2452329

Yang H, Guo Z. A Study on Factors Affecting Baseball Pitch Trajectories. Frontiers in Science and Engineering. 2026. DOI:10.54691/tbv1x605

Feng Z, Lochhead L, Kohn J, Appelbaum L. Predictors of batting and pitching performance in the USA baseball prospect development pipeline. Sports Biomechanics. 2024. DOI:10.1080/14763141.2023.2298959

## Multi-class imbalanced classification (relevant to curveball-class collapse)

Wang J, Awang N. MKC-SMOTE: A Novel Synthetic Oversampling Method for Multi-Class Imbalanced Data Classification. IEEE Access. 2024. DOI:10.1109/access.2024.3521120

Sahoo M, Sridhar R. Multi-class Classification of Class Imbalanced Skin Lesion Dataset Using a Modified SMOTE-ENN Gabor-Enhanced VGG-19 Architecture. SN Computer Science. 2025. DOI:10.1007/s42979-025-03806-8

Song S, Yang S. TS-SMOTE: An Improved SMOTE Method Based on Symmetric Triangle Scoring Mechanism for Solving Class-Imbalanced Problems. Symmetry. 2025. DOI:10.3390/sym17081326

Huang Y. FS-SMOTE: An Improved SMOTE Method Based on Feature Space Scoring Mechanism for Solving Class-Imbalanced Problems. IEEE Access. 2025. DOI:10.1109/access.2025.3597794

---

Total entries: 34, all DOIs resolved live via CrossRef on 2026-05-08. Source query log: `/root/AI/.tmp/lit_03_skubal/`.
