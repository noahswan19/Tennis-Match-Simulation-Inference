# Tennis Match Outcome Prediction via Monte Carlo Simulation

Predicts ATP match outcomes by simulating in-match statistics from player-specific Bayesian posteriors and aggregating ML classifier predictions across many simulated scenarios.

---

## Overview

Rather than predicting match results from rankings or summary statistics directly, this project builds a generative model of each player's on-court statistics and uses it to produce probabilistic predictions. For each match, both players' 12 in-match statistics are independently sampled from Beta posteriors fitted on their historical performance, the simulated stat differences are fed into a trained classifier, and win probabilities are averaged across many simulation draws.

Data comes from Jeff Sackmann's open-source [tennis_atp](https://github.com/JeffSackmann/tennis_atp) dataset covering ATP matches from 2003 through 2023. Models are trained on 2003–2022 data and evaluated on a held-out 2023 set.

This project is the evolution of a project I worked on before with a more extensive write up here: https://medium.com/@noahswan19/simulating-tennis-matches-with-maximum-likelihood-estimation-6e4c0ca1370e
The major changes in this project correspond to the modeling of player statistics. Instead of truncated normal distributions, this updated version models statstics as rates on binomial random variables with Beta priors and posteriors.

---

## Results

On the 2023 holdout data, using just rank to predict match results results in an accuracy of 63.4%. From a high level, the configuration of parameters implemented in `main.py` achieves
a peak accuracy of 64.6% for the Random Forest model simulating matches 450 times and the XGBoost model simulating matches 200 times. This marks an improvement of 1.2 percentage points or a 1.9% increase.

The `diagnostics.ipynb' notebook shares some additional plots assessing the accuracy across different categorical variables. These results show that this method performs poorly when data is sparse for the players in the model. When cutting by surface, we see a drop in accuracy for the simulation method on grass courts, a surface where few players will play enough matches to have posteriors significantly different from the priors. In addition, if we start limiting to matches where both players have played a minimum number of matches, meaning their posterior distributions are more specific to the player's results, we see a gap emerge between the results of this method and the naive rank-based method. For example, if we require both players to have both played at least 36 matches on the court against the same level of opponent (sample size of 94), we see the Random Forest method can achieve an accuracy of 71.3% compared to the rank-based approach 62.8%, equivalent to a 8.5 percentage point gap or a marginal increase of **13.5% in accuracy using the simulation approach**. 

---

## Core Idea: Simulation at Inference

The ML models are trained on **observed** match statistics from historical ATP data. At inference time, for any upcoming match between two players, each player's 12 statistics are independently sampled from their Beta posteriors, stat differences are computed, and the trained classifier produces a win probability for that simulation. Win probabilities are averaged across many simulations to produce a single match-level prediction.

This simulation-at-inference design is the core contribution: it produces a full win-probability distribution for any upcoming match between any two players, conditioned on surface and opponent quality, without requiring knowledge of in-match statistics that don't yet exist. As opposed to the previous implementation of the project, this version is flexible to use data for players with few matches played up to a point in time 

**A note on posterior concentration.** Beta posteriors for players with many career matches accumulate large α+β, so draws cluster tightly around the posterior mean. Left uncorrected, running many simulations would produce near-identical predictions, reducing the Monte Carlo average to a single deterministic prediction from career averages. The `sample_concentration` parameter caps α+β before each draw, preserving the posterior mean exactly while restoring meaningful spread in the samples. This ensures the simulation reflects genuine match-to-match variability rather than just estimation uncertainty.

The below sections describe the intimate details of the implementation. Read at the risk of your own boredom.

---

## Pipeline

### 1. Data Ingestion and Feature Engineering

Raw per-year CSVs are downloaded from Jeff Sackmann's GitHub repository and cached locally as Parquet files. The raw data is in a winner/loser row format; this is reshaped into a per-player-per-match format with one row per player per match.

Filtering removes retirements, walkovers, Carpet surface matches, and rows missing key fields (age, rank, surface, ace counts). The following 12 statistics are computed as the simulation inputs — all are proportions bounded in [0, 1]:

| Category | Statistics |
|---|---|
| Serve | `first_in_per`, `first_won_per`, `second_won_per`, `rally_svptw_per`, `ace_per`, `df_per` |
| Return | `first_rpw_per`, `second_rpw_per`, `rally_rpw_per`, `ace_per_against` |
| Pressure | `bp_face_freq`, `bp_create_freq` |

Return and pressure stats are derived by joining each player's row against their opponent's serve-side raw counts within the same match.

For ML training, match rows are collapsed to a single row per match in a **match-diff format**: player A is the alphabetically first player by name, player B is the other. Most stat features are expressed as differences (A − B), but three serve stats (`first_won_per`, `second_won_per`, `rally_svptw_per`) are expressed as levels (A + B) — their return counterparts already capture relative advantage, so the serve diff would be perfectly collinear with the return diff. Player numeric columns (`rank_points`, `age`) are kept as separate A/B columns. Player categorical columns (`seed`, `entry`, `rank_bin`, `hand`) and match-level categoricals (`surface`, `tourney_level`, `draw_size`, `best_of`, `round`) are one-hot encoded.

### 2. Bayesian Beta Posteriors

A `BayesianBetaPosterior` is fitted for each player × surface × opponent rank bin combination. The 5 rank bins are: Top 10, Top 25, Top 50, Top 100, and Outside Top 100, giving each player up to 15 separate distributions (3 surfaces × 5 bins).

**Empirical Bayes priors.** Hyperparameters α₀ and β₀ are set per surface and opponent bin from aggregate 2003 data:

- For each surface, compute the mean rate p̄ from players with ≥5 matches (aggregate successes / aggregate totals across per-player rates).
- Apply an opponent rank-bin offset: compute the bin-specific rate p_bin from 2003 aggregates, then set p̄_adjusted = p̄ + (p_bin − p̄_overall).
- Set α₀ = `prior_concentration` × p̄_adjusted and β₀ = `prior_concentration` × (1 − p̄_adjusted), pinning α₀+β₀ to `prior_concentration` while preserving the adjusted mean. (Setting `prior_concentration=None` falls back to raw method-of-moments using per-player variance to estimate α₀ and β₀.)

**Conjugate updates.** For each player, matches are sorted chronologically by `match_order = tourney_date × 10 + round_digit`. The posterior stored for match `i` uses a shifted cumulative sum of raw counts from matches `0 … i−1` only, ensuring strict no-leakage:

```
α_i = α₀ + Σ_{j<i} successes_j
β_i = β₀ + Σ_{j<i} (total_j − successes_j)
```

Posteriors are stored in a long-format Parquet file and indexed by `(match_id, player_name, surface, opp_rank_bin)` for O(1) lookup during simulation.

### 3. Train/Validation Split and Scaling

Matches from 2003–2022 are split 80/20 by match ID into train and validation sets (shuffled with `random_state=33`). The 2023 matches are held out entirely for final evaluation.

A `StandardScaler` is fit on the continuous columns of the training split (stat features and numeric player columns) and applied to both the train and validation splits. The fitted scaler and list of continuous column names are retained for use when scaling simulation inputs at inference time.

**Concentration cap.** At inference time, before drawing each Beta posterior sample, if `sample_concentration` is set on the `BayesianBetaPosterior`, α and β are rescaled so that α+β ≤ `sample_concentration`: `scale = min(1, C / (α+β))`, `α' = α·scale`, `β' = β·scale`. The mean is preserved exactly; only the spread increases. This prevents the simulation from collapsing to near-identical predictions for players with many career matches.

### 4. Model Training

Three classifiers are trained on the observed match-diff rows:

- **MLP** (PyTorch Lightning): 1 hidden layer with 16 neurons, ReLU activation, dropout=0.25, SGD optimizer (lr=0.01), early stopping on validation loss with patience=20. Saved as a `.pt` state dict.
- **Random Forest** (scikit-learn): Hyperparameter search via `RandomizedSearchCV` (15 iterations, 3-fold CV, neg log-loss) over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`. Saved with joblib (`.joblib`).
- **XGBoost**: Hyperparameter search via `RandomizedSearchCV` (15 iterations, 5-fold CV, neg log-loss) over `max_depth`, `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`. Saved in XGBoost binary JSON format (`.ubj`).

All models expose a unified interface (`fit`, `predict_proba`, `save`, `load`, `tune`) via the `TennisModel` abstract base class.

### 5. Holdout Evaluation

For each match in the 2023 holdout set, `N` simulations are drawn from each player's posterior (state as of that match). Before computing stat differences, cross-player stat pairs (e.g., `first_won_per` / `first_rpw_per`) are normalized so each pair sums to 1 per simulation, ensuring internally consistent match-up statistics. Fixed match features are joined to each simulated row, continuous columns are scaled with the training-set scaler, and `model.predict_proba` is called on all `N` rows. The `N` predicted probabilities that player A wins are averaged to produce a single win probability per match. A match is predicted correctly if `avg_prob_A > 0.5` matches the true `player_a_won` label.

Evaluation is run at every 50 simulations from N=100 to N=1000 to assess how accuracy depends on simulation count.

---

## Project Structure

```
.
├── main.py                        # End-to-end pipeline entry point
├── src/
│   ├── data_preparation.py        # TennisMatchDataset: load, reshape, feature engineering, splits
│   ├── distributions.py           # BayesianBetaPosterior: priors, conjugate updates, sampling
│   ├── models.py                  # TennisModel ABC + MLP, Random Forest, XGBoost implementations
│   └── evaluation.py              # Simulation, prediction aggregation, accuracy, charts
├── data/
│   ├── raw/                       # Per-year cached Parquet files (downloaded from GitHub)
│   └── processed/                 # player_matches_wfeat.parquet, match_diffs_complete.parquet,
│                                  #   posteriors.parquet
└── models/                        # Serialized model artifacts (.pt, .joblib, .ubj)
```

---

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for Python package management (Python 3.13).

```bash
# Install dependencies
uv sync

# Run the full pipeline
uv run main.py
```

On first run the pipeline downloads raw CSVs from GitHub and caches them to `data/raw/`. Subsequent runs load from cache. Intermediate artifacts (processed data, posteriors, trained models) are also cached — set `force_posterior_rerun = True` at the top of `main.py` to recompute posteriors, or delete a model file under `models/` to retrain that model.

---
