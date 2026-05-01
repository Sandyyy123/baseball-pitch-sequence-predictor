# References - MLB Pitch-Type Prediction (Case Study: Tarik Skubal)

Verified bibliography (20 entries) for the Skubal pitch-type prediction manuscript. All DOIs and arXiv IDs were confirmed via OpenAlex and Crossref on 2026-05-01. Each entry has a one-line abstract.

## Pitch-type and pitch-sequence prediction

1. Bock, J. K. (2015). Pitch Sequence Complexity and Long-Term Pitcher Performance. *Sports*, 3(1), 40-55. DOI: 10.3390/sports3010040.
   Sequence-entropy analysis showing that more unpredictable pitch sequences correlate with longer effective MLB pitching careers.

2. Sidle, G., & Tran, H. (2017). Using Multi-Class Classification Methods to Predict Baseball Pitch Types. *Journal of Sports Analytics*, 4(1), 85-93. DOI: 10.3233/jsa-170171.
   Compares LDA, k-NN, SVM, and random forest on per-pitcher PITCHf/x data; per-pitcher RFs reach about 67 percent multi-class accuracy.

3. Lee, J. S. (2022). Prediction of Pitch Type and Location in Baseball Using Ensemble Model of Deep Neural Networks. *Journal of Sports Analytics*, 8(2), 115-126. DOI: 10.3233/jsa-200559.
   Ensemble of DNNs that predicts both pitch type and zone location from situational features, beating single-model baselines.

4. Hamilton, M., Hoang, P., Layne, L., Murray, J., Padgett, D., Stafford, C., & Tran, H. (2014). Applying Machine Learning Techniques to Baseball Pitch Prediction. *ICPRAM*. DOI: 10.5220/0004763905200527.
   First widely cited ML pitch-type predictor: SVM and linear classifier on count, score, runners, prior pitch, achieving binary fastball-vs-offspeed accuracy around 70 percent.

5. Pane, M. A., Ventura, S. L., Steorts, R. C., & Nugent, R. (2013). Trouble With The Curve: Improving MLB Pitch Classification. arXiv:1304.1756. DOI: 10.48550/arXiv.1304.1756.
   Improves Statcast/PITCHf/x pitch-type labels via per-pitcher mixture models that respect each pitcher's pitch arsenal.

6. Healey, G., & Zhao, S. (2017). Using PITCHf/x to Model the Dependence of Strikeout Rate on the Predictability of Pitch Sequences. *Journal of Sports Analytics*, 3(2), 93-101. DOI: 10.3233/jsa-170103.
   Empirical link between sequence predictability (entropy of conditional pitch-type distributions) and strikeout production.

7. Umemura, K., Yanai, T., & Nagata, Y. (2020). Application of VBGMM for Pitch Type Classification: Analysis of TrackMan's Pitch Tracking Data. *Japanese Journal of Statistics and Data Science*, 3, 475-492. DOI: 10.1007/s42081-020-00079-8.
   Variational Bayesian Gaussian mixture clustering of TrackMan release/movement features to derive cleaner pitch-type labels than vendor labels.

## Tracking systems and sports-analytics reviews

8. Kovalchik, S. (2023). Player Tracking Data in Sports. *Annual Review of Statistics and Its Application*, 10, 677-697. DOI: 10.1146/annurev-statistics-033021-110117.
   Comprehensive review of optical and radar tracking systems (PITCHf/x, Hawk-Eye, Statcast) and the statistical pipelines built on top of them.

9. Beal, R., Norman, T. J., & Ramchurn, S. D. (2019). Artificial Intelligence for Team Sports: A Survey. *The Knowledge Engineering Review*, 34, e28. DOI: 10.1017/s0269888919000225.
   Survey of ML and AI methods for team-sport prediction, opponent modeling, and tactical analysis.

10. Baumer, B. S., Matthews, G. J., & Nguyen, Q. (2023). Big Ideas in Sports Analytics and Statistical Tools for Their Investigation. *WIREs Computational Statistics*, 15(6), e1612. DOI: 10.1002/wics.1612.
    Modern overview of expected-value frameworks, win probability, tracking-data models, and reproducibility in sports analytics.

## Game-theoretic pitcher-batter models

11. Sidhu, G., & Caffo, B. (2014). MONEYBaRL: Exploiting Pitcher Decision-Making Using Reinforcement Learning. *Annals of Applied Statistics*, 8(2), 926-955. DOI: 10.1214/13-aoas712.
    Models the pitcher-batter interaction as a Markov decision process and uses RL to find batter-optimal exploitation of pitcher tendencies.

12. Kovash, K., & Levitt, S. D. (2009). Professionals Do Not Play Minimax: Evidence from Major League Baseball and the National Football League. NBER Working Paper 15347. DOI: 10.3386/w15347.
    Tests minimax mixed-strategy equilibrium predictions on actual MLB pitch sequences and finds significant negative serial correlation, contradicting Nash play.

13. Weinstein-Gould, J. (2009). Keeping the Hitter Off Balance: Mixed Strategies in Baseball. *Journal of Quantitative Analysis in Sports*, 5(2). DOI: 10.2202/1559-0410.1173.
    Game-theoretic model of pitcher pitch-mix selection in 2x2 fastball-offspeed games conditional on count.

14. Nakahara, H., Takeda, K., & Fujii, K. (2023). Estimating the Effect of Hitting Strategies in Baseball Using Counterfactual Virtual Simulation with Deep Learning. *International Journal of Computer Science in Sport*, 22(1), 1-20. DOI: 10.2478/ijcss-2023-0001.
    Counterfactual deep-learning simulator quantifying how alternative batter strategies would change run expectancy against observed pitcher tendencies.

## Core ML methods

15. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780. DOI: 10.1162/neco.1997.9.8.1735.
    Original LSTM paper, foundational for sequence models applied to pitch-by-pitch prediction.

16. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 30*. arXiv:1706.03762. DOI: 10.48550/arXiv.1706.03762.
    Introduces the Transformer architecture; baseline for modern sequence models on play-by-play sports data.

17. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*, 785-794. DOI: 10.1145/2939672.2939785.
    Canonical reference for gradient-boosted trees, the dominant tabular baseline for Statcast-style pitch features.

18. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32. DOI: 10.1023/A:1010933404324.
    Foundational random-forest paper used as a baseline in essentially every pitch-type-prediction study.

19. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 30*. arXiv:1705.07874. DOI: 10.48550/arXiv.1705.07874.
    Introduces SHAP values for local feature attribution, used here to explain pitch-type model decisions per count and runner state.

20. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-Sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357. DOI: 10.1613/jair.953.
    Standard oversampling technique used to handle class imbalance across rare pitch types (CU, CH) in the Skubal arsenal.
