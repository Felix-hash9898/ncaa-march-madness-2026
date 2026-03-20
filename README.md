# NCAA March Madness 2026 — Win Probability Prediction

**Authors**: [Xinwei Huang](https://github.com/Felix-hash9898) · Haoran Zhang

**Kaggle Competition**: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)  
**Metric**: Brier Score (MSE of predicted win probabilities, lower is better)  
**Final CV Brier**: **0.1579** — 36.8% improvement over naive baseline (0.25), 10.8% over seed-only (0.1770)

---

## Results

| Method | CV Brier |
|---|---|
| Naive (predict 0.5) | 0.2500 |
| Seed-only historical lookup | 0.1770 |
| Logistic Regression (7 features) | 0.1661 |
| LightGBM baseline (all 171 features) | 0.1709 |
| LightGBM tuned (Optuna, top-20 features) | 0.1595 |
| XGBoost | 0.1681 |
| CatBoost | 0.1647 |
| Gender-specific ensemble (LGB + CatBoost) | 0.1583 |
| **+ Temperature scaling + Seed-prior blend** | **0.1579** |

**Per-gender breakdown** (CV val set, after calibration):

| Gender | Brier | n |
|---|---|---|
| Men | 0.1861 | 401 |
| Women | 0.1294 | 394 |

**Stage 1 backtest** (2022–2025 tournaments, known results):

| Season | Brier | n |
|---|---|---|
| 2022 | 0.1077 | 134 |
| 2023 | 0.1104 | 134 |
| 2024 | 0.0893 | 134 |
| 2025 | 0.0609 | 134 |
| Men overall | 0.1262 | 268 |
| Women overall | 0.0579 | 268 |

**2026 Bracket Predictions** (simulated from Stage 2 submission):
- 🏆 Men's Champion: **Arizona** (beat Duke in championship, P(Duke)=0.442)
- 🏆 Women's Champion: **Connecticut** (beat UCLA in championship, P(UConn)=0.506)

---

## Project Structure

```
ncaa-march-madness-2026/
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # CSV loading, M/W unification, label builder
│   ├── features.py           # Basic stats, Four Factors, SOS, conf tourney
│   ├── elo.py                # Three Elo variants with MOV adjustment
│   ├── massey.py             # Massey Ordinals feature extraction (men only)
│   ├── context_features.py   # Coach, conference strength, seed features
│   └── pipeline.py           # Master assembly: team features → matchup features
├── submissions/
│   ├── stage1/               # Stage 1 submission (519,144 rows, seasons 2022-2025)
│   └── stage2/               # Stage 2 submission (132,133 rows, season 2026)
│       ├── submission_ensemble_v1.csv
│       ├── bracket_2026_mens.csv
│       └── bracket_2026_womens.csv
├── ncaa_march_madness_2026.ipynb   # Main notebook (full pipeline + results)
├── validate.py                      # Standalone CV validation script
└── README.md
```

> **Data not included** (Kaggle competition data): download `march-machine-learning-mania-2026/` from the competition page and place it in the project root.

---

## Pipeline Overview

```
Raw CSVs (35 files)
    │
    ▼
[1] Data Loading & Preprocessing          (src/data_loader.py)
    ├── Merge Men's + Women's data (add Gender column)
    ├── Build symmetric training pairs: ID = SSSS_lowTeamID_highTeamID
    └── Label = P(lower TeamID wins)
    │
    ▼
[2] Team-Season Feature Engineering       (src/features.py, elo.py, massey.py, context_features.py)
    ├── Basic stats: win rate, avg margin, momentum (last 30 days), close-game win rate
    ├── Advanced stats: Four Factors (eFG%, TO rate, OR%, FT rate), net efficiency per 100 poss, tempo
    ├── Strength of Schedule: avg/median opponent win rate, weighted wins
    ├── Elo ratings: 3 variants (standard K=32, aggressive K=48, conservative K=20) + MOV adjustment
    ├── GLM team quality: Bradley-Terry logistic regression with recency weighting
    ├── Massey Ordinals: 20 top ranking systems, trend features (men only)
    ├── Coach features: career year, tenure, prior tournament appearances
    ├── Conference strength: avg/median win rate, avg margin across conference teams
    └── Seed features: seed number, region, seed_top4/top8 binary flags
    │
    ▼
[3] Matchup Feature Assembly              (src/pipeline.py)
    ├── Vectorized merge: A_xxx, B_xxx, diff_xxx (A−B) for all 156 team features
    ├── Interaction features: interact_seed_x_elo, interact_seed_x_winrate, net_matchup_eff
    ├── Log-ratio features: log_ratio_win_rate, log_ratio_elo, cross_a_off_b_def
    └── Head-to-head history: lookback 5 years (tournament training only)
    → 171 matchup features total (154 diff + 4 h2h + 13 interaction/ratio)
    │
    ▼
[4] Regular Season Augmentation
    ├── 235,150 regular season games appended (season 2003–2025, no 2020)
    ├── Tournament games weighted 5×, regular season 1×
    └── H2H and seed features NaN-filled (imputed at train time from tournament median)
    │
    ▼
[5] Feature Selection
    ├── LGB baseline on all 171 features → extract gain-based importance
    ├── Exclude h2h features (unavailable at Stage 2 inference → train/test distribution shift)
    ├── Test Top-{10, 20, 30, 50, 80, 171} by CV Brier
    └── Top-20 selected (Brier=0.1639, best across all sizes)
    │
    ▼
[6] Model Training (Time-Series CV: train < season N, validate on season N)
    ├── Validation seasons: 2019, 2021, 2022, 2023, 2024, 2025 (skip 2020 COVID)
    ├── LightGBM — Optuna-tuned (100 trials): lr=0.102, num_leaves=20, max_depth=6
    ├── XGBoost — fixed params
    ├── CatBoost — fixed params
    └── Gender-specific models: separate Men/Women training
        ├── Women drop 2 Massey features (>50% NaN): diff_rank_KPK, diff_rank_std_all
        └── Augmented RS data used where CV Brier improves
    │
    ▼
[7] Ensemble
    ├── Optimize SLSQP weights on OOF predictions separately per gender
    ├── Men:   LGB=0.451, XGB=0.000, CatBoost=0.549  → Brier=0.1862
    ├── Women: LGB=0.287, XGB=0.000, CatBoost=0.713  → Brier=0.1299
    └── XGB weight=0 for both genders (dominated by LGB/CatBoost)
    │
    ▼
[8] Calibration
    ├── Temperature scaling: T=0.90 (sharpens predictions) → 0.1583→0.1580
    ├── Clip range: (0.01, 0.99) → no further improvement
    └── Seed-prior blending: alpha=0.925 model + 0.075 historical seed win rates → 0.1579
    │
    ▼
[9] Submission (132,133 rows, Season 2026)
```

---

## Top 20 Selected Features

| Rank | Feature | Description |
|---|---|---|
| 1 | `diff_elo_conservative` | Elo difference (K=20, mean reversion=0.85) |
| 2 | `diff_seed` | Tournament seed difference (A − B) |
| 3 | `diff_elo_standard` | Elo difference (K=32, mean reversion=0.75) |
| 4 | `diff_sos_mean` | Strength of schedule difference |
| 5 | `interact_seed_x_elo` | Seed diff × Elo diff interaction |
| 6 | `diff_elo_std` | Std dev across 3 Elo variants (uncertainty signal) |
| 7 | `diff_avg_margin` | Average scoring margin difference |
| 8 | `diff_elo_mean` | Mean of 3 Elo variants difference |
| 9 | `diff_conf_median_win_rate` | Conference median win rate difference |
| 10 | `diff_rank_std_all` | Std dev of rankings across all Massey systems (men only) |
| 11 | `diff_sos_median` | Median SOS difference |
| 12 | `diff_rank_KPK` | KenPom-adjacent ranking difference (men only) |
| 13 | `diff_eFG_pct_std` | Effective FG% volatility difference |
| 14 | `diff_opp_eFG_pct_std` | Opponent eFG% volatility difference |
| 15 | `diff_reb_margin_std` | Rebound margin volatility difference |
| 16 | `diff_efg_trend` | eFG% trend (late season vs early season) difference |
| 17 | `diff_blk_rate_std` | Block rate volatility difference |
| 18 | `diff_opp_ft_pct_mean` | Opponent FT% allowed, mean difference |
| 19 | `diff_recent_win_rate` | Win rate in last 30 days (DayNum 103–132) difference |
| 20 | `diff_tempo_std` | Tempo (possessions/game) volatility difference |

Note: Features 10 and 12 (`diff_rank_std_all`, `diff_rank_KPK`) are dropped for women's models due to >50% missing values (Massey Ordinals are men-only in the dataset).

---

## Key Design Decisions

**Gender-specific models**: Women's basketball is substantially more predictable (CV Brier 0.1294 vs Men's 0.1861). Massey Ordinals data is only available for men, so a unified model would impute ~50% of women's Massey features with noise. Separate models eliminate this cross-contamination.

**Feature count = 20, not 171**: Adding more features hurts (Top-171 Brier=0.1717 vs Top-20 Brier=0.1639). The Elo and seed signals are so dominant that extra features act as noise for a dataset of only ~2,800 tournament games.

**XGBoost weight = 0**: In both gender-specific ensembles, SLSQP optimization assigns zero weight to XGBoost. LightGBM (Optuna-tuned) and CatBoost fully dominate.

**Regular season augmentation**: 235,150 RS games added with weight=1 (tournament games weight=5). Improvement is model- and gender-dependent — selection is done automatically by comparing CV Brier with and without augmentation per model.

**H2H features excluded from inference**: Head-to-head history is available for training (historical tournament matchups) but not for Stage 2 (we don't know which teams will play each other). Including them would cause train/inference distribution shift.

---

## Reproduction

```bash
# 1. Clone and install dependencies
git clone https://github.com/Felix-hash9898/ncaa-march-madness-2026.git
cd ncaa-march-madness-2026
pip install pandas numpy lightgbm xgboost catboost optuna scikit-learn scipy matplotlib seaborn

# 2. Download competition data from Kaggle
# Place the unzipped folder as: march-machine-learning-mania-2026/

# 3. Run the full pipeline (validation mode)
python validate.py

# 4. Or open the notebook for the full pipeline + submission generation
jupyter notebook ncaa_march_madness_2026.ipynb
```

**Expected runtime**: ~45 seconds for feature pipeline, ~3 minutes for Optuna (100 trials).

---

## CV Folds (Time-Series)

| Fold | Train seasons | Val season | Train n | Val n |
|---|---|---|---|---|
| 0 | 2003–2018 | 2019 | 2056 | 130 |
| 1 | 2003–2019 | 2021 | 2186 | 129 |
| 2 | 2003–2021 | 2022 | 2315 | 134 |
| 3 | 2003–2022 | 2023 | 2449 | 134 |
| 4 | 2003–2023 | 2024 | 2583 | 134 |
| 5 | 2003–2024 | 2025 | 2717 | 134 |

Season 2020 excluded from both training and validation (COVID — no tournament held).

---

## LightGBM Best Hyperparameters (Optuna, 100 trials)

```python
{
    "learning_rate": 0.1022,
    "num_leaves": 20,
    "max_depth": 6,
    "min_child_samples": 72,
    "feature_fraction": 0.500,
    "bagging_fraction": 0.768,
    "bagging_freq": 2,
    "reg_alpha": 0.0268,
    "reg_lambda": 0.0230,
}
```

Best Optuna trial: #64 of 100, Brier=0.1595.

---

## Dependencies

```
python >= 3.10
pandas
numpy
lightgbm
xgboost
catboost
optuna
scikit-learn
scipy
matplotlib
seaborn
```

---

## Contributors

- [Xinwei Huang](https://github.com/Felix-hash9898)
- Haoran Zhang
