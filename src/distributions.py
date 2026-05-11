"""
Bayesian Beta posterior distributions for player-surface statistics.

Implements empirical-Bayes priors from the first data year and rolling
conjugate updates from raw count columns, maintaining strict no-leakage
ordering (state before each match only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.stats import beta as scipy_beta

# ---------------------------------------------------------------------------
# Stat → (success_expr_str, total_expr_str) mapping
# Expressions are strings evaluated against a Polars context where all raw
# count columns are available as Float64.
# ---------------------------------------------------------------------------

# For each stat we need (successes col/expr, total col/expr) to form:
#   alpha = alpha_0 + cumsum(successes)
#   beta  = beta_0  + cumsum(total - successes)
STAT_COUNT_MAP: dict[str, tuple[str, str]] = {
    "first_in_per": ("first_in", "svpt"),
    "first_won_per": ("first_won", "first_in"),
    "second_won_per": ("second_won", "svpt_minus_first_in"),  # derived below
    "rally_svptw_per": ("rally_svptw", "rally_svpt"),
    "ace_per": ("ace", "svpt"),
    "df_per": ("df", "svpt"),
    "bp_face_freq": ("bp_faced", "svpt"),
    "first_rpw_per": ("opp_first_in_minus_opp_first_won", "opp_first_in"),  # derived
    "second_rpw_per": (
        "opp_second_attempts_minus_opp_second_won",
        "opp_second_attempts",
    ),  # derived
    "rally_rpw_per": (
        "opp_rally_svpt_minus_opp_rally_svptw",
        "opp_rally_svpt",
    ),  # derived
    "ace_per_against": ("opp_ace", "opp_svpt"),
    "bp_create_freq": ("opp_bp_faced", "opp_svpt"),
}

STATS = list(STAT_COUNT_MAP.keys())

# Raw count columns needed for posterior computation
RAW_COUNT_COLS: list[str] = [
    "svpt",
    "first_in",
    "first_won",
    "second_won",
    "ace",
    "df",
    "bp_faced",
    "rally_svpt",
    "rally_svptw",
    "opp_svpt",
    "opp_first_in",
    "opp_first_won",
    "opp_second_won",
    "opp_ace",
    "opp_bp_faced",
    "opp_rally_svpt",
    "opp_rally_svptw",
]

# Surfaces supported
SURFACES = ["Hard", "Clay", "Grass"]

# Opponent ranking bins
OPP_RANK_BINS = ["Top 10", "Top 25", "Top 50", "Top 100", "Outside Top 100"]


class BayesianBetaPosterior:
    """
    Bayesian Beta conjugate posterior for player-surface match statistics.

    Empirical Bayes priors are computed from aggregate counts in the first
    data year (prior_year). For each player × surface, matches are processed
    in chronological order; the posterior stored for match i reflects evidence
    from all matches before i (strict no-leakage).

    Attributes:
        decay (float): Exponential time-decay factor (1.0 = no decay).
        prior_year (int): Year used to compute empirical Bayes priors.
        sample_concentration (float | None): If set, caps α+β to this value before
            sampling to inflate variance while preserving the posterior mean.
            None disables scaling.
        prior_concentration (float | None): Fixed concentration for empirical Bayes
            priors. None uses raw method-of-moments (sample variance) instead.
        priors (dict): Surface → stat → (alpha0, beta0).
        posterior_df (pl.DataFrame | None): Long-format posterior parameters.
    """

    def __init__(
        self,
        decay: float = 1.0,
        prior_year: int = 2003,
        sample_concentration: float | None = None,
        prior_concentration: float | None = 300.0,
    ) -> None:
        """
        Initialize the posterior calculator.

        Args:
            decay (float): Multiplicative time-decay weight per match-order step.
                           1.0 disables decay. Defaults to 1.0.
            prior_year (int): Year from which to compute empirical Bayes priors.
                              Defaults to 2003.
            sample_concentration (float | None): If set, scales α and β at sample
                time so that α+β ≤ sample_concentration, preserving the mean while
                inflating variance. None disables scaling. Defaults to None.
            prior_concentration (float | None): Fixed α₀+β₀ for empirical Bayes priors.
                The MoM-derived mean is preserved; only the concentration is pinned.
                Higher values give the prior a more lasting effect on the posterior.
                If None, uses raw method-of-moments (sample mean + variance) to
                estimate α₀ and β₀ directly. Defaults to 300.0.
        """
        self.decay = decay
        self.prior_year = prior_year
        self.sample_concentration = sample_concentration
        self.prior_concentration = prior_concentration
        self.priors: dict[str, dict[str, tuple[float, float]]] = {}
        self.posterior_df: pl.DataFrame | None = None
        self._index: dict[tuple[str, str, str], dict[str, tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_derived_count_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add derived raw-count columns needed by STAT_COUNT_MAP.

        Args:
            df (pl.DataFrame): DataFrame with raw count columns as Float64.

        Returns:
            pl.DataFrame: DataFrame with additional derived columns.
        """
        return df.with_columns(
            svpt_minus_first_in=(pl.col("svpt") - pl.col("first_in")),
            opp_first_in_minus_opp_first_won=(
                pl.col("opp_first_in") - pl.col("opp_first_won")
            ),
            opp_second_attempts=(pl.col("opp_svpt") - pl.col("opp_first_in")),
            opp_second_attempts_minus_opp_second_won=(
                (pl.col("opp_svpt") - pl.col("opp_first_in")) - pl.col("opp_second_won")
            ),
            opp_rally_svpt_minus_opp_rally_svptw=(
                pl.col("opp_rally_svpt") - pl.col("opp_rally_svptw")
            ),
        )

    def _compute_priors(self, player_matches_wfeat: pl.DataFrame) -> None:
        """
        Compute empirical Bayes priors from prior_year aggregate counts.

        For each surface, sum all successes and failures across all players
        to set the hyperparameters alpha0, beta0. Falls back to pooled
        all-surface counts if a surface has no data.

        Args:
            player_matches_wfeat (pl.DataFrame): Full player-match DataFrame
                including raw count columns.
        """
        prior_data = (
            player_matches_wfeat.filter(pl.col("year") == self.prior_year)
            .select("id", "name", "opp_rank_bin", "surface", *RAW_COUNT_COLS)
            .with_columns(pl.col(RAW_COUNT_COLS).cast(pl.Float64))
        )
        prior_data = self._add_derived_count_cols(prior_data)

        prior_data_limited = prior_data.filter(  # limit to only players with min 5 matches played for prior
            pl.col("surface").count().over(["name", "id", "surface"]) >= 5
        ).drop("opp_rank_bin")

        # Aggregate per surface
        surface_agg = prior_data_limited.group_by("surface", "id", "name").agg(
            cs.all().sum()
        )

        # get aggregated data for opponent bin offset
        rank_bin_summ = (
            prior_data.drop("id", "name")
            .group_by("opp_rank_bin", "surface")
            .agg(cs.all().sum())
        )

        for surface in SURFACES:
            self.priors[surface] = {}
            surface_rows = surface_agg.filter(pl.col("surface") == surface)
            rank_bin_rows = rank_bin_summ.filter(pl.col("surface") == surface)

            if len(surface_rows) == 0:
                raise ValueError(
                    f"No players found to compute priors for {surface} in {self.prior_year}"
                )

            for stat, (succ_col, total_col) in STAT_COUNT_MAP.items():
                total_val = surface_rows[total_col]
                succ_val = surface_rows[succ_col]
                rates = succ_val / total_val
                x_bar = rates.mean()
                x_var = rates.var() if self.prior_concentration is None else None

                # implement offset using opponent rank bin
                overall_avg_rate = (
                    rank_bin_rows[succ_col].sum() / rank_bin_rows[total_col].sum()
                )
                for bin, bin_row in rank_bin_rows.group_by("opp_rank_bin"):
                    bin_name = bin[0]
                    if bin_name not in self.priors[surface].keys():
                        self.priors[surface][bin_name] = {}
                    bin_rate = (bin_row[succ_col] / bin_row[total_col]).item()
                    offset = bin_rate - overall_avg_rate
                    adjusted_x_bar = x_bar + offset
                    if not (0 < adjusted_x_bar < 1):
                        raise ValueError(
                            f"adjusted_x_bar={adjusted_x_bar:.4f} out of (0,1) for "
                            f"{stat} on {surface}/{bin_name}"
                        )
                    if self.prior_concentration is None:
                        concentration = (
                            adjusted_x_bar * (1 - adjusted_x_bar) / x_var - 1
                        )
                        alpha_0 = float(adjusted_x_bar * concentration)
                        beta_0 = float((1 - adjusted_x_bar) * concentration)
                    else:
                        alpha_0 = self.prior_concentration * adjusted_x_bar
                        beta_0 = self.prior_concentration * (1 - adjusted_x_bar)
                    self.priors[surface][bin_name][stat] = (alpha_0, beta_0)

    def _build_index(self) -> None:
        """
        Build an in-memory lookup index from posterior_df for O(1) parameter access.

        Converts the long-format posterior_df into a nested dict keyed by
        (match_id, name, surface, opp_rank_bin) → {stat: (alpha, beta)}, replacing the
        O(rows) Polars filter in get_params with a O(1) dict lookup.
        """
        match_ids = self.posterior_df["match_id"].to_list()
        names = self.posterior_df["name"].to_list()
        surfaces = self.posterior_df["surface"].to_list()
        opp_rank_bins = self.posterior_df["opp_rank_bin"].to_list()
        stats = self.posterior_df["stat"].to_list()
        alphas = self.posterior_df["alpha"].to_list()
        betas = self.posterior_df["beta"].to_list()

        index: dict[tuple[str, str, str, str], dict[str, tuple[float, float]]] = {}
        for mid, name, surf, bin, stat, alpha, beta in zip(
            match_ids, names, surfaces, opp_rank_bins, stats, alphas, betas
        ):
            key = (mid, name, surf, bin)
            if key not in index:
                index[key] = {}
            index[key][stat] = (alpha, beta)

        self._index = index

    def fit(
        self,
        player_matches_wfeat: pl.DataFrame,
        cache_path: Path | None = None,
    ) -> Self:
        """
        Compute rolling Bayesian Beta posteriors for all player-surface pairs.

        For each player × surface group, matches are sorted by match_order.
        The posterior stored at row i is based on cumulative evidence from
        matches 0 … i-1 (shifted cumulative sum), ensuring no data leakage.

        Args:
            player_matches_wfeat (pl.DataFrame): Per-player match data with
                raw count columns and match_order.
            cache_path (Path | None): If provided, cache the posterior DataFrame
                to this Parquet path. Defaults to None.

        Returns:
            Self: Updated object with posterior_df set.
        """
        print("Computing empirical Bayes priors...")
        self._compute_priors(player_matches_wfeat)

        count_data = player_matches_wfeat.select(
            "match_id",
            "name",
            "surface",
            "opp_rank_bin",
            "match_order",
            *RAW_COUNT_COLS,
        ).with_columns(pl.col(RAW_COUNT_COLS).cast(pl.Float64))
        count_data = self._add_derived_count_cols(count_data)

        print("Computing rolling posteriors...")
        result_rows: list[dict] = []

        # Build column lists for success and total per stat
        succ_cols = [STAT_COUNT_MAP[s][0] for s in STATS]
        total_cols = [STAT_COUNT_MAP[s][1] for s in STATS]

        for surface in SURFACES:
            for opp_bin in OPP_RANK_BINS:
                alpha0_vec = np.array(
                    [self.priors[surface][opp_bin][s][0] for s in STATS]
                )
                beta0_vec = np.array(
                    [self.priors[surface][opp_bin][s][1] for s in STATS]
                )

                surface_df = count_data.filter(
                    pl.col("surface") == surface, pl.col("opp_rank_bin") == opp_bin
                ).sort(["name", "match_order"])

                for player_name, group in surface_df.group_by(
                    "name", maintain_order=True
                ):
                    player_label = player_name[0]
                    # Extract arrays (n_matches, n_stats)
                    n = len(group)
                    succ_arr = np.column_stack([group[c].to_numpy() for c in succ_cols])
                    total_arr = np.column_stack(
                        [group[c].to_numpy() for c in total_cols]
                    )

                    nan_mask = np.isnan(succ_arr) | np.isnan(total_arr)
                    if nan_mask.any():
                        bad_rows = np.where(nan_mask.any(axis=1))[0]
                        bad_match_ids = [
                            group["match_id"].to_list()[r] for r in bad_rows
                        ]
                        raise ValueError(
                            f"NaN in raw count columns for player {player_label!r} on {surface!r} in bin {opp_bin!r}."
                            f"Affected match_ids: {bad_match_ids}"
                        )

                    fail_arr = total_arr - succ_arr

                    if self.decay < 1.0:
                        # Weighted cumulative sum: most recent match gets decay^1, etc.
                        cumsucc = np.zeros((n, len(STATS)))
                        cumfail = np.zeros((n, len(STATS)))
                        for i in range(1, n):
                            weights = self.decay ** np.arange(i, 0, -1)
                            cumsucc[i] = (succ_arr[:i] * weights[:, None]).sum(axis=0)
                            cumfail[i] = (fail_arr[:i] * weights[:, None]).sum(axis=0)
                    else:
                        # No decay: simple cumsum shifted by 1
                        cumsucc = np.vstack(
                            [
                                np.zeros((1, len(STATS))),
                                np.cumsum(succ_arr, axis=0)[:-1],
                            ]
                        )
                        cumfail = np.vstack(
                            [
                                np.zeros((1, len(STATS))),
                                np.cumsum(fail_arr, axis=0)[:-1],
                            ]
                        )

                    alphas = alpha0_vec + cumsucc  # (n, n_stats)
                    betas = beta0_vec + cumfail  # (n, n_stats)

                    match_ids = group["match_id"].to_list()
                    for i, match_id in enumerate(match_ids):
                        for j, stat in enumerate(STATS):
                            result_rows.append(
                                {
                                    "match_id": match_id,
                                    "name": player_label,
                                    "surface": surface,
                                    "opp_rank_bin": opp_bin,
                                    "stat": stat,
                                    "alpha": float(alphas[i, j]),
                                    "beta": float(betas[i, j]),
                                }
                            )

        self.posterior_df = pl.DataFrame(result_rows)
        self._build_index()

        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self.posterior_df.write_parquet(cache_path)
            print(f"Posteriors cached to {cache_path}")

        return self

    def load(self, cache_path: Path) -> Self:
        """
        Load previously computed posteriors from a Parquet cache file.

        Args:
            cache_path (Path): Path to cached posteriors Parquet file.

        Returns:
            Self: Updated object with posterior_df set.
        """
        self.posterior_df = pl.read_parquet(cache_path)
        self._build_index()
        return self

    def get_params(
        self, match_id: str, name: str, surface: str, opp_rank_bin: str
    ) -> dict[str, tuple[float, float]]:
        """
        Look up posterior alpha/beta parameters for a player at a specific match.

        Args:
            match_id (str): Match identifier.
            name (str): Player name.
            surface (str): Surface name (Hard, Clay, Grass).
            opp_rank_bin (str): Opponent ranking bin used to select prior.

        Returns:
            dict[str, tuple[float, float]]: Stat name → (alpha, beta).
        """
        key = (match_id, name, surface, opp_rank_bin)
        if key not in self._index:
            raise LookupError(
                f"No posterior found for match_id={match_id!r}, name={name!r}, surface={surface!r}."
            )
        return self._index[key]

    def sample(
        self,
        match_id: str,
        name: str,
        surface: str,
        opp_rank_bin: str,
        n_samples: int,
        stat: str | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Draw random samples from the Beta posterior for a player at a specific match.

        Args:
            match_id (str): Match identifier.
            name (str): Player name.
            surface (str): Surface name (Hard, Clay, Grass).
            opp_rank_bin (str): Opponent ranking bin used to select prior.
            n_samples (int): Number of samples to draw per statistic.
            stat (str | None): If provided, sample only this statistic. Defaults to None.

        Returns:
            dict[str, np.ndarray]: Stat name → array of shape (n_samples,).
        """
        params = self.get_params(match_id, name, surface, opp_rank_bin)
        if stat is not None:
            if stat not in params:
                raise ValueError(
                    f"Unknown stat {stat!r}. Valid stats: {sorted(params)}"
                )
            params = {stat: params[stat]}

        if self.sample_concentration is not None:
            params = {
                s: (
                    a * min(1.0, self.sample_concentration / (a + b)),
                    b * min(1.0, self.sample_concentration / (a + b)),
                )
                for s, (a, b) in params.items()
            }

        return {s: scipy_beta.rvs(a, b, size=n_samples) for s, (a, b) in params.items()}
