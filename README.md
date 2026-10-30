# 🎮 Steam Market Intelligence: Commercial Traction Prediction & Non-Parametric Modeling

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas%20%7C%20NumPy-brightgreen.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

## 📌 Executive Summary
With over **130,000 titles** listed on Steam, platform discovery is saturated. Nearly **34% of all games have zero reviews**, and over **81% fail to reach commercial traction** (defined as $\ge 100$ user reviews). For indie developers, publishers, and investment funds, allocating development capital without empirical demand validation represents extreme downside risk.

This project implements an end-to-end quantitative analytics and predictive machine learning pipeline on a **414 MB dataset (130,100 listings)** to:
1. **Uncover platform inequality and consumer behavior** using non-parametric statistics (Mann-Whitney U, Gini concentration, Chi-Square odds ratios).
2. **Identify market demand vs. supply mismatches** across primary genres.
3. **Predict commercial traction before launch** using only pre-release signals via temporal out-of-time benchmarking, achieving a **4.15x Lift** in hit discovery.

---

## 📊 Key Findings at a Glance

* **Extreme Winner-Take-Most Dynamic:** Review engagement displays a **Gini coefficient of 0.967**. The top **1% of titles capture 74.9% of all platform reviews**, while the bottom 80% struggle for visibility.
* **The Indie & Casual Saturation Trap:**
  * **Indie:** 70.3% of catalog supply captures only 46.3% of market demand (**0.66x Demand/Supply ratio**).
  * **Casual:** 43.1% of supply captures just 18.4% of demand (**0.43x Demand/Supply ratio**, with a 15.2% hit rate).
  * **High-Leverage Niches:** **MMO (5.79x)**, **Action (1.63x)**, and **RPG (1.48x)** show significant organic excess demand relative to listing volumes.
* **The Multi-Platform Multiplier:** Supporting macOS or Linux alongside Windows raises the 100-review traction rate from 22.2% to 37.5%—an **Odds Ratio of 2.10** ($p < 10^{-300}$).
* **Pre-Launch Predictive Power:** Using strictly launch-time attributes (localization depth, achievements, media count, cloud integration), **Histogram Gradient Boosting** achieved an **ROC-AUC of 0.832** and **PR-AUC of 0.563** (vs. a 15.4% random baseline), delivering **64.0% Precision at the Top 10% cutoff**.

---

## 🛠️ Data Architecture & Defensive Pipeline

```text
  [Raw 414 MB CSV] ---> [Defensive Schema Repair] ---> [Feature Engineering]
                               (40-Col Realignment)        (Multi-Platform, Ratios)
                                                                     │
                     ┌───────────────────────────────────────────────┘
                     │
                     ├──────────────────────────────┐
                     ▼                              ▼
        [Statistical Analysis]             [Temporal ML Split]
       - Gini Inequality (0.967)          - Train: 2014-2020 (38,867)
       - Mann-Whitney U Testing           - Test:  2021-2022 (24,083)
       - Opportunity Matrix                         │
                                                    ▼
                                       [Benchmark & Explanations]
                                       - HistGradientBoosting (0.832 AUC)
                                       - Permutation Importance
```
