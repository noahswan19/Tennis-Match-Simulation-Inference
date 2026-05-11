"""
Simulation-based evaluation of tennis match prediction models.

Draws from Bayesian Beta posteriors to simulate match statistics, feeds
simulations through trained models, and aggregates predictions to win
probabilities. Generates accuracy metrics.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from sklearn.preprocessing import StandardScaler

from src.data_preparation import SERVE_LEVEL_STATS
from src.distributions import BayesianBetaPosterior, STATS
from src.models import TennisModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Simulated stat feature column names: level (A+B) for serve stats, diff (A-B) for others
STAT_DIFF_COLS = [
    f"{stat}_level" if stat in SERVE_LEVEL_STATS else f"{stat}_diff" for stat in STATS
]

# Cross-player stat pairs: (server_stat, returner_stat).
# Used to normalize stat pairings so computed differences
# reflect legitimate percentages
STAT_NORMALIZATION_PAIRS: list[tuple[str, str]] = [
    ("first_won_per", "first_rpw_per"),
    ("second_won_per", "second_rpw_per"),
    ("rally_svptw_per", "rally_rpw_per"),
    ("ace_per", "ace_per_against"),
    ("bp_face_freq", "bp_create_freq"),
]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _normalize_paired_stats(
    samples_A: dict[str, np.ndarray],
    samples_B: dict[str, np.ndarray],
) -> None:
    """Normalize cross-player stat pairs in-place so each pair sums to 1 per sim.

    For each (serve_stat, return_stat) pair, applies adj = stat / (serve + return)
    in both serve directions (A-serves and B-serves).
    """
    for serve_stat, return_stat in STAT_NORMALIZATION_PAIRS:
        s, r = samples_A[serve_stat], samples_B[return_stat]
        total = s + r
        samples_A[serve_stat] = s / total
        samples_B[return_stat] = r / total

        s, r = samples_B[serve_stat], samples_A[return_stat]
        total = s + r
        samples_B[serve_stat] = s / total
        samples_A[return_stat] = r / total


def simulate_match_stats(
    posterior: BayesianBetaPosterior,
    holdout_data: pl.DataFrame,
    player_matches_wfeat: pl.DataFrame,
    n_sims: int,
) -> pl.DataFrame:
    """
    Draw Beta posterior samples for each match and compute per-stat differences.

    For each match in holdout_data, determines player A (alphabetically first name)
    and player B, samples n_sims from each player's posterior, and computes
    stat_diff = sample_A - sample_B for each of the 12 simulated statistics.

    Args:
        posterior (BayesianBetaPosterior): Fitted posterior object.
        holdout_data (pl.DataFrame): Match-diff format holdout with match_id column.
        player_matches_wfeat (pl.DataFrame): Full player-match DataFrame for name/surface lookup.
        n_sims (int): Number of simulations per match.

    Returns:
        pl.DataFrame: Columns: match_id, sim_idx, {stat}_diff for all 12 stats.
            One row per match × simulation.
    """
    # Build lookup: match_id → [(name, surface, opp_rank_bin), ...]
    match_players = (
        player_matches_wfeat.select(["match_id", "name", "surface", "opp_rank_bin"])
        .join(
            holdout_data.select("match_id"),
            on="match_id",
            how="semi",
        )
        .unique()
    )

    dfs: list[pl.DataFrame] = []
    for match_id in holdout_data["match_id"].unique().to_list():
        players_in_match = match_players.filter(pl.col("match_id") == match_id)
        if len(players_in_match) != 2:
            raise ValueError(f"Didn't find 2 players in match {match_id}")

        rows = players_in_match.to_dicts()
        names = [r["name"] for r in rows]
        name_A, name_B = sorted(names)

        row_A = next(r for r in rows if r["name"] == name_A)
        row_B = next(r for r in rows if r["name"] == name_B)

        samples_A = posterior.sample(
            match_id, name_A, row_A["surface"], row_A["opp_rank_bin"], n_sims
        )
        samples_B = posterior.sample(
            match_id, name_B, row_B["surface"], row_B["opp_rank_bin"], n_sims
        )

        _normalize_paired_stats(samples_A, samples_B)

        chunk: dict[str, object] = {
            "match_id": [match_id] * n_sims,
            "sim_idx": np.arange(n_sims),
        }
        for stat in STATS:
            a_vals = samples_A.get(stat, np.full(n_sims, np.nan))
            b_vals = samples_B.get(stat, np.full(n_sims, np.nan))
            if stat in SERVE_LEVEL_STATS:
                chunk[f"{stat}_level"] = a_vals + b_vals
            else:
                chunk[f"{stat}_diff"] = a_vals - b_vals

        dfs.append(pl.DataFrame(chunk))

    return pl.concat(dfs) if dfs else pl.DataFrame()


def predict_with_model(
    model: TennisModel,
    simulated_stats: pl.DataFrame,
    holdout_data: pl.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
    continuous_cols: list[str] | None = None,
) -> pl.DataFrame:
    """
    Run model.predict_proba on each simulated set of match stat differences.

    Joins simulated diff columns with fixed match features from holdout_data,
    then calls model.predict_proba. Returns the probability that player A wins.

    Args:
        model (TennisModel): Fitted tennis model.
        simulated_stats (pl.DataFrame): Output of simulate_match_stats.
            Columns: match_id, sim_idx, {stat}_diff for all 12 stats.
        holdout_data (pl.DataFrame): Match-diff format holdout dataset.
        feature_cols (list[str]): Ordered list of columns the model expects.
        scaler (StandardScaler | None): Fitted scaler for continuous columns. Defaults to None.
        continuous_cols (list[str] | None): Names of continuous columns to scale. Defaults to None.

    Returns:
        pl.DataFrame: Columns: match_id, sim_idx, pred_prob (probability player A wins).
    """
    # Fixed features: all holdout cols except stat diffs, target, and identifiers
    fixed_cols = [
        c
        for c in holdout_data.columns
        if c not in STAT_DIFF_COLS + ["player_a_won", "match_id", "year"]
    ]
    fixed_feat_df = holdout_data.select(["match_id"] + fixed_cols)

    merged = simulated_stats.join(fixed_feat_df, on="match_id", how="left")

    # Build feature matrix in the expected column order
    available_cols = [c for c in feature_cols if c in merged.columns]
    X_df = merged.select(available_cols)

    # Apply scaler to continuous columns before converting to numpy
    if scaler is not None and continuous_cols is not None:
        scaled = scaler.transform(X_df.select(continuous_cols).to_numpy())
        X_df = X_df.with_columns(pl.from_numpy(scaled, schema=continuous_cols))

    X = X_df.to_numpy().astype(np.float32)

    proba = model.predict_proba(X)
    win_proba = proba[:, 1]

    return merged.select(["match_id", "sim_idx"]).with_columns(
        pred_prob=pl.Series(win_proba)
    )


def aggregate_predictions(predictions: pl.DataFrame) -> pl.DataFrame:
    """
    Average predicted win probabilities for player A across simulations per match.

    Args:
        predictions (pl.DataFrame): Output of predict_with_model with columns
            match_id, sim_idx, pred_prob.

    Returns:
        pl.DataFrame: Columns: match_id, avg_prob_A.
    """
    return predictions.group_by("match_id").agg(
        pl.col("pred_prob").mean().alias("avg_prob_A")
    )


def compute_accuracy(
    aggregated: pl.DataFrame,
    holdout_data: pl.DataFrame,
) -> float:
    """
    Compute match-level prediction accuracy from aggregated win probabilities.

    Predicts player A wins when avg_prob_A > 0.5, then compares against the
    player_a_won ground-truth label in holdout_data.

    Args:
        aggregated (pl.DataFrame): Output of aggregate_predictions.
            Columns: match_id, avg_prob_A.
        holdout_data (pl.DataFrame): Match-diff holdout containing match_id and
            player_a_won columns for ground-truth labels.

    Returns:
        float: Fraction of matches predicted correctly.
    """
    labels = holdout_data.select(["match_id", "player_a_won"])
    joined = aggregated.join(labels, on="match_id", how="inner")
    predicted = (joined["avg_prob_A"] > 0.5).cast(pl.Int64)
    correct = (predicted == joined["player_a_won"]).sum()
    return float(correct) / len(joined)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def eval_num_sims(
    model: TennisModel,
    posterior: BayesianBetaPosterior,
    holdout_data: pl.DataFrame,
    player_matches_wfeat: pl.DataFrame,
    n_sims_list: list[int],
    feature_cols: list[str] | None = None,
    scaler: StandardScaler | None = None,
    continuous_cols: list[str] | None = None,
) -> pl.DataFrame:
    """
    Evaluate model accuracy at multiple simulation counts.

    Runs the full simulate → predict → aggregate → accuracy pipeline for each
    value of n_sims in n_sims_list.

    Args:
        model (TennisModel): Fitted tennis model.
        posterior (BayesianBetaPosterior): Fitted Beta posterior object.
        holdout_data (pl.DataFrame): Match-diff format holdout set (one row per match).
        player_matches_wfeat (pl.DataFrame): Full player-match DataFrame for name/surface lookup.
        n_sims_list (list[int]): Number of simulations to evaluate.
        feature_cols (list[str] | None): Model feature column order. If None,
            uses all columns except match_id, year, player_a_won. Defaults to None.
        scaler (StandardScaler | None): Fitted scaler for continuous columns. Defaults to None.
        continuous_cols (list[str] | None): Names of the continuous columns to scale. Defaults to None.

    Returns:
        pl.DataFrame: Columns: n_sims, accuracy.
    """
    if feature_cols is None:
        feature_cols = [
            c
            for c in holdout_data.columns
            if c not in ("match_id", "year", "player_a_won")
        ]

    max_sims = max(n_sims_list)
    print(f"  Simulating {max_sims} draws per match...")
    simulated_all = simulate_match_stats(
        posterior, holdout_data, player_matches_wfeat, max_sims
    )

    results = []
    for n_sims in n_sims_list:
        simulated = simulated_all.filter(pl.col("sim_idx") < n_sims)
        preds = predict_with_model(
            model,
            simulated,
            holdout_data,
            feature_cols,
            scaler=scaler,
            continuous_cols=continuous_cols,
        )
        agg = aggregate_predictions(preds)
        acc = compute_accuracy(agg, holdout_data)
        print(f"    Accuracy at {n_sims} sims: {acc:.4f}")
        results.append({"n_sims": n_sims, "accuracy": acc})

    return pl.DataFrame(results)
