"""
Data loading, transformation, feature engineering, and train/test splitting for ATP tennis match data.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Self

import polars as pl
import polars.selectors as cs
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Year configuration
# ---------------------------------------------------------------------------

TRAIN_YEARS = range(2003, 2024)
HOLDOUT_YEAR = 2023

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"

ROUND_DIGIT: dict[str, int] = {
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
    "BR": 7,
}

# Serve stats represented as level (A + B) rather than diff (A - B).
# Their return counterparts (first_rpw_per etc.) already capture relative advantage,
# so the serve diff would be perfectly collinear with the return diff.
SERVE_LEVEL_STATS: list[str] = ["first_won_per", "second_won_per", "rally_svptw_per"]

# The 12 simulated statistics (inputs to the simulation)
STAT_COLS: list[str] = [
    "first_in_per",
    "first_won_per",
    "second_won_per",
    "rally_svptw_per",
    "ace_per",
    "df_per",
    "first_rpw_per",
    "second_rpw_per",
    "rally_rpw_per",
    "bp_face_freq",
    "ace_per_against",
    "bp_create_freq",
]

# Match-level context features shared by both players
MATCH_LEVEL_COLS: list[str] = [
    "surface",
    "tourney_level",
    "draw_size",
    "best_of",
    "round",
]

# Player-specific numeric features — kept as separate A/B columns
PLAYER_NUMERIC_COLS: list[str] = ["rank_points", "age"]

# Player-specific categorical features — one-hot encoded per player with _A/_B suffix
PLAYER_CATEGORICAL_COLS: list[str] = ["seed", "entry", "rank_bin", "hand"]

# Features used by the ML model (identity + outcome + fixed + simulated)
# Retained for player_matches_wfeat construction (used by add_features drop_nulls)
MODEL_FEATS: list[str] = [
    "match_id",
    "result",
    # fixed match context
    "year",
    "tourney_level",
    "surface",
    "draw_size",
    "best_of",
    "round",
    "seed",
    "entry",
    "rank_points",
    "rank_bin",
    "opp_rank_points",
    "opp_rank_bin",
    "age",
    "hand",
    "opp_hand",
    # simulated / observed stats
    "first_in_per",
    "first_won_per",
    "second_won_per",
    "rally_svptw_per",
    "ace_per",
    "df_per",
    "first_rpw_per",
    "second_rpw_per",
    "rally_rpw_per",
    "bp_face_freq",
    "ace_per_against",
    "bp_create_freq",
]

# Raw count columns retained for Bayesian Beta posterior updates
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
    # opponent equivalents
    "opp_svpt",
    "opp_first_in",
    "opp_first_won",
    "opp_second_won",
    "opp_ace",
    "opp_bp_faced",
    "opp_rally_svpt",
    "opp_rally_svptw",
]


class TennisMatchDataset:
    """
    Pipeline to load, transform, and feature-engineer ATP tennis match data.

    Loads raw CSVs from Jeff Sackmann's GitHub, caches them locally, reshapes
    the winner/loser format to a per-player-per-match format, computes derived
    statistics, and prepares train/holdout splits for modeling.

    The ML training format is match-level (one row per match) with statistical
    features as differences (player A − player B), where player A is the
    alphabetically first player by name.

    Attributes:
        years (list[int]): Years to include in the dataset.
        raw_cache_dir (Path): Directory for cached raw-year Parquet files.
        processed_cache_dir (Path): Directory for cached processed Parquet files.
        player_matches_wfeat (pl.DataFrame | None): Per-player matches with all features.
            Used for posterior fitting and simulation lookup.
        match_diffs (pl.DataFrame | None): Match-level rows with stat diffs and A/B player cols.
        match_diffs_complete (pl.DataFrame | None): Dummy-encoded model-ready match-diff data.
        X_train (pl.DataFrame | None): Training features (continuous cols standardized).
        X_test (pl.DataFrame | None): Validation features (continuous cols standardized).
        y_train (pl.Series | None): Training labels (player_a_won).
        y_test (pl.Series | None): Validation labels (player_a_won).
        holdout (pl.DataFrame | None): Holdout-year model-ready match-diff data.
        scaler (StandardScaler | None): Fitted on X_train continuous cols; reuse for inference.
        continuous_feature_cols (list[str] | None): Names of the standardized columns.
    """

    def __init__(
        self,
        years: Iterable[int],
        raw_cache_dir: Path = Path("data/raw"),
        processed_cache_dir: Path = Path("data/processed"),
    ) -> None:
        """
        Initialize the dataset with year range and cache paths.

        Args:
            years (Iterable[int]): Years to load (e.g., range(2003, 2024)).
            raw_cache_dir (Path): Directory for per-year raw Parquet cache.
            processed_cache_dir (Path): Directory for processed Parquet cache.
        """
        self.years = sorted(list(set(years)))
        self.raw_cache_dir = Path(raw_cache_dir)
        self.processed_cache_dir = Path(processed_cache_dir)
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_cache_dir.mkdir(parents=True, exist_ok=True)

        self._original_df: pl.DataFrame | None = None
        self.player_matches_wfeat: pl.DataFrame | None = None
        self.match_diffs: pl.DataFrame | None = None
        self.match_diffs_complete: pl.DataFrame | None = None
        self.X_train: pl.DataFrame | None = None
        self.X_test: pl.DataFrame | None = None
        self.y_train: pl.Series | None = None
        self.y_test: pl.Series | None = None
        self.train_match_ids: pl.Series | None = None
        self.test_match_ids: pl.Series | None = None
        self.holdout: pl.DataFrame | None = None
        self.scaler: StandardScaler | None = None
        self.continuous_feature_cols: list[str] | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_year(self, year: int) -> pl.DataFrame:
        """
        Load a single year of match data, using local Parquet cache if available.

        Args:
            year (int): The year to load.

        Returns:
            pl.DataFrame: Match data for that year.
        """
        cache_path = self.raw_cache_dir / f"{year}.parquet"
        if cache_path.exists():
            return pl.read_parquet(cache_path)
        print(f"  Downloading {year}...")
        df = pl.read_csv(
            GITHUB_BASE.format(year=year),
            infer_schema_length=int(1e9),
        )
        df.write_parquet(cache_path)
        return df

    def _load_matches(self) -> pl.DataFrame:
        """
        Load all years of match data, caching each year individually.

        Returns:
            pl.DataFrame: Combined match data for all years.
        """
        print("Loading data...")
        frames = [self._load_year(y) for y in self.years]
        return pl.concat(frames, how="diagonal")

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def derive_player_matches(self) -> Self:
        """
        Reshape raw match data from winner/loser format to per-player-per-match rows.

        Adds match_id, tiebreak counts, year, and result columns.

        Returns:
            Self: Updated dataset object.
        """
        print("Transforming to player matches...")
        matches = self._original_df.rename(
            mapping=lambda x: x
            if x == "draw_size"
            else x.replace("1st", "first")
            .replace("2nd", "second")
            .replace("l_", "loser_")
            .replace("w_", "winner_")
        ).with_columns(
            match_id=pl.col("tourney_date").cast(pl.String)
            + "-"
            + pl.col("winner_id").cast(pl.String)
            + "-"
            + pl.col("loser_id").cast(pl.String),
            num_tiebreaks=pl.col("score")
            .str.extract_all(r"\(")
            .list.len()
            .cast(pl.Int64),
            winner_tiebreaks_won=pl.col("score").str.extract_all(r"7-6\(").list.len(),
            loser_tiebreaks_won=pl.col("score").str.extract_all(r"6-7\(").list.len(),
            year=pl.col("tourney_date")
            .cast(pl.String)
            .str.slice(0, length=4)
            .cast(pl.Int64),
        )

        winner_data = (
            matches.select(cs.exclude("^loser_.*$"))
            .with_columns(result=pl.lit("winner"))
            .rename(lambda x: x.removeprefix("winner_"))
            .rename(
                lambda x: re.sub(r"(\w)([A-Z])", r"\g<1>_\g<2>", x)
                .lower()
                .replace("__", "_")
            )
        )
        loser_data = (
            matches.select(cs.exclude("^winner_.*$"))
            .with_columns(result=pl.lit("loser"))
            .rename(lambda x: x.removeprefix("loser_"))
            .rename(
                lambda x: re.sub(r"(\w)([A-Z])", r"\g<1>_\g<2>", x)
                .lower()
                .replace("__", "_")
            )
        )
        unpivoted = pl.concat([winner_data, loser_data])

        self._player_matches_raw = unpivoted
        return self

    def limit_player_matches(self) -> Self:
        """
        Filter out retirements, walkovers, carpet surface, and rows with missing key fields.

        Also removes rounds ER and RR, and players with unknown/ambiguous hand.

        Returns:
            Self: Updated dataset object.
        """
        print("Limiting player matches...")
        self._player_matches_limited = self._player_matches_raw.filter(
            ~pl.col("score").str.contains("RET|W/O|Walkover"),
            pl.col("surface") != "Carpet",
            pl.col(["age", "rank", "surface", "ace"]).is_not_null(),
            pl.col("hand").is_in(["L", "R"]),
            ~pl.col("round").is_in(["ER", "RR"]),
            pl.col("svpt") != 0,
        )
        return self

    def add_features(self) -> Self:
        """
        Compute all 12 derived statistics plus raw count columns for Beta updates.

        Serve stats computed from player's own serve columns. Return/opponent stats
        derived by joining each match row against the opponent's row (flipped result).
        Adds match_order = tourney_date * 10 + round_digit for chronological ordering.

        Returns:
            Self: Updated dataset object.
        """
        print("Adding basic features...")
        intermediate = (
            self._player_matches_limited.with_columns(
                rally_svpt=(
                    pl.col("svpt").cast(pl.Float64)
                    - pl.col("ace").cast(pl.Float64)
                    - pl.col("df").cast(pl.Float64)
                ),
                seed=pl.col("seed").fill_null("Unseeded"),
                entry=pl.col("entry").fill_null("NA"),
            )
            .with_columns(
                rank_bin=pl.when(pl.col("rank") <= 10)
                .then(pl.lit("Top 10"))
                .when(pl.col("rank") <= 25)
                .then(pl.lit("Top 25"))
                .when(pl.col("rank") <= 50)
                .then(pl.lit("Top 50"))
                .when(pl.col("rank") <= 100)
                .then(pl.lit("Top 100"))
                .otherwise(pl.lit("Outside Top 100")),
                tourney_level=pl.when(pl.col("tourney_level") == "M")
                .then(pl.lit("Masters"))
                .when(pl.col("tourney_level") == "A")
                .then(pl.lit("ATP 250/500"))
                .when(pl.col("tourney_level") == "G")
                .then(pl.lit("Grand Slam"))
                .when(pl.col("tourney_level") == "F")
                .then(pl.lit("Year End Final")),
                first_in_per=pl.col("first_in").cast(pl.Float64)
                / pl.col("svpt").cast(pl.Float64),
                first_won_per=pl.col("first_won").cast(pl.Float64)
                / pl.col("first_in").cast(pl.Float64),
                second_won_per=(
                    pl.col("second_won").cast(pl.Float64)
                    / (
                        pl.col("svpt").cast(pl.Float64)
                        - pl.col("first_in").cast(pl.Float64)
                    )
                ),
                rally_svptw=(
                    pl.col("first_won").cast(pl.Float64)
                    + pl.col("second_won").cast(pl.Float64)
                    - pl.col("ace").cast(pl.Float64)
                ),
                ace_per=pl.col("ace").cast(pl.Float64)
                / pl.col("svpt").cast(pl.Float64),
                df_per=pl.col("df").cast(pl.Float64) / pl.col("svpt").cast(pl.Float64),
                bp_face_freq=pl.col("bp_faced").cast(pl.Float64)
                / pl.col("svpt").cast(pl.Float64),
            )
            .with_columns(
                rally_svptw_per=pl.col("rally_svptw") / pl.col("rally_svpt"),
                match_order=(
                    pl.col("tourney_date").cast(pl.Int64) * 10
                    + pl.col("round").replace(ROUND_DIGIT, default=0)
                ),
            )
            .with_columns(  # Consolidate some rarer categorical variable categories
                seed=pl.when(pl.col("seed").is_in([str(i) for i in [1, 2, 3, 4]]))
                .then(pl.lit("Top 4 Seed"))
                .when(pl.col("seed").is_in([str(i) for i in [5, 6, 7, 8]]))
                .then(pl.lit("Seed 5-8"))
                .when(pl.col("seed").is_in([str(i) for i in range(9, 17)]))
                .then(pl.lit("Seed 9-16"))
                .when(pl.col("seed") == "Unseeded")
                .then(pl.lit("Unseeded"))
                .otherwise(pl.lit("Seed 17+")),
                draw_size=pl.when(pl.col("draw_size") <= 28)
                .then(pl.lit("28 or fewer"))
                .otherwise(pl.col("draw_size").cast(pl.String)),
                entry=pl.when(pl.col("entry").is_in(["NA", "Q", "WC"]))
                .then(pl.col("entry"))
                .otherwise(pl.lit("Other")),
                round=pl.when(pl.col("round") == "BR")
                .then(pl.lit("F"))
                .otherwise(pl.col("round")),
            )
        )

        # Build opponent stats by flipping result and joining raw count cols
        print("Adding opponent features...")
        opp_cols = {
            "first_won_per": "first_rpw_per",
            "second_won_per": "second_rpw_per",
            "rally_svptw_per": "rally_rpw_per",
            "ace_per": "ace_per_against",
            "rank_points": "opp_rank_points",
            "rank_bin": "opp_rank_bin",
            # raw counts with opp_ prefix
            "svpt": "opp_svpt",
            "first_in": "opp_first_in",
            "first_won": "opp_first_won",
            "second_won": "opp_second_won",
            "ace": "opp_ace",
            "bp_faced": "opp_bp_faced",
            "rally_svpt": "opp_rally_svpt",
            "rally_svptw": "opp_rally_svptw",
            "hand": "opp_hand",
        }

        opponent_stats = (
            intermediate.select(["match_id", "result"] + list(opp_cols.keys()))
            .rename(opp_cols)
            .with_columns(
                # Return win rates = 1 − opponent's serve win rates
                first_rpw_per=1.0 - pl.col("first_rpw_per"),
                second_rpw_per=1.0 - pl.col("second_rpw_per"),
                rally_rpw_per=1.0 - pl.col("rally_rpw_per"),
                bp_create_freq=pl.col("opp_bp_faced").cast(pl.Float64)
                / pl.col("opp_svpt").cast(pl.Float64),
                result=pl.when(pl.col("result") == "winner")
                .then(pl.lit("loser"))
                .otherwise(pl.lit("winner")),
            )
        )

        joined = (
            intermediate.join(opponent_stats, on=["match_id", "result"], how="inner")
            .drop_nulls(MODEL_FEATS)
            .drop_nans(cs.numeric() & cs.by_name(*MODEL_FEATS))
        )

        self.player_matches_wfeat = joined
        return self

    def derive_match_diffs(self) -> Self:
        """
        Build a match-level DataFrame with one row per match.

        For each match, player A is the alphabetically first player by name.
        Statistical features are computed as differences (A − B). Numeric and
        categorical player features are kept as separate A/B columns. The target
        column player_a_won = 1 if player A won, 0 otherwise.

        Requires player_matches_wfeat to be populated.

        Returns:
            Self: Updated dataset object.
        """
        print("Deriving match-diff format...")
        pm = self.player_matches_wfeat

        winner_rows = pm.filter(pl.col("result") == "winner").select(
            ["match_id", "year", "name"]
            + MATCH_LEVEL_COLS
            + STAT_COLS
            + PLAYER_NUMERIC_COLS
            + PLAYER_CATEGORICAL_COLS
        )
        loser_rows = pm.filter(pl.col("result") == "loser").select(
            ["match_id", "name"]
            + STAT_COLS
            + PLAYER_NUMERIC_COLS
            + PLAYER_CATEGORICAL_COLS
        )

        # Inner join: one row per match
        joined = winner_rows.join(loser_rows, on="match_id", suffix="_loser")

        # Determine alphabetical player A (True = winner is alphabetically first)
        joined = joined.with_columns(
            winner_is_A=(pl.col("name") < pl.col("name_loser")),
            player_a_won=(pl.col("name") < pl.col("name_loser")).cast(pl.Int64),
        )

        # Stat features: level (A + B) for serve stats, diff (A - B) for all others
        stat_feat_exprs = []
        for stat in STAT_COLS:
            if stat in SERVE_LEVEL_STATS:
                stat_feat_exprs.append(
                    (pl.col(stat) + pl.col(f"{stat}_loser")).alias(f"{stat}_level")
                )
            else:
                stat_feat_exprs.append(
                    pl.when(pl.col("winner_is_A"))
                    .then(pl.col(stat) - pl.col(f"{stat}_loser"))
                    .otherwise(pl.col(f"{stat}_loser") - pl.col(stat))
                    .alias(f"{stat}_diff")
                )

        # Numeric and categorical cols with _A/_B suffix
        player_AB_exprs = []
        for col in PLAYER_NUMERIC_COLS + PLAYER_CATEGORICAL_COLS:
            player_AB_exprs.append(
                pl.when(pl.col("winner_is_A"))
                .then(pl.col(col))
                .otherwise(pl.col(f"{col}_loser"))
                .alias(f"{col}_A")
            )
            player_AB_exprs.append(
                pl.when(pl.col("winner_is_A"))
                .then(pl.col(f"{col}_loser"))
                .otherwise(pl.col(col))
                .alias(f"{col}_B")
            )

        joined = joined.with_columns(stat_feat_exprs + player_AB_exprs)

        final_cols = (
            ["match_id", "year"]
            + MATCH_LEVEL_COLS
            + [
                f"{stat}_level" if stat in SERVE_LEVEL_STATS else f"{stat}_diff"
                for stat in STAT_COLS
            ]
            + [f"{col}_A" for col in PLAYER_NUMERIC_COLS]
            + [f"{col}_B" for col in PLAYER_NUMERIC_COLS]
            + [f"{col}_A" for col in PLAYER_CATEGORICAL_COLS]
            + [f"{col}_B" for col in PLAYER_CATEGORICAL_COLS]
            + ["player_a_won"]
        )

        self.match_diffs = joined.select(final_cols)
        return self

    def prepare_for_model(self) -> Self:
        """
        One-hot encode categoricals and standardize numeric types on match_diffs.

        Encodes match-level categorical columns and per-player categorical columns
        (with _A/_B suffix). Casts numeric feature columns to Float64 and the
        target player_a_won to Int64.

        Returns:
            Self: Updated dataset object.
        """
        match_cat_cols = MATCH_LEVEL_COLS
        player_cat_A_cols = [f"{c}_A" for c in PLAYER_CATEGORICAL_COLS]
        player_cat_B_cols = [f"{c}_B" for c in PLAYER_CATEGORICAL_COLS]
        all_cat_cols = match_cat_cols + player_cat_A_cols + player_cat_B_cols

        numeric_AB_cols = [f"{c}_A" for c in PLAYER_NUMERIC_COLS] + [
            f"{c}_B" for c in PLAYER_NUMERIC_COLS
        ]
        stat_diff_cols = [
            f"{stat}_level" if stat in SERVE_LEVEL_STATS else f"{stat}_diff"
            for stat in STAT_COLS
        ]

        self.match_diffs_complete = self.match_diffs.to_dummies(
            all_cat_cols
        ).with_columns(
            pl.col(numeric_AB_cols + stat_diff_cols).cast(pl.Float64),
            pl.col("player_a_won").cast(pl.Int64),
        )
        return self

    def split_data(self) -> Self:
        """
        Assign 80/20 train/validation match ID split, holdout set, and feature matrices.

        Sets train_match_ids, test_match_ids, holdout, X_train, X_test, y_train, y_test,
        scaler, and continuous_feature_cols from the observed match stats in
        match_diffs_complete.

        Returns:
            Self: Updated dataset object.
        """
        train_data = self.match_diffs_complete.filter(pl.col("year") != HOLDOUT_YEAR)
        self.holdout = self.match_diffs_complete.filter(pl.col("year") == HOLDOUT_YEAR)

        match_ids = (
            train_data["match_id"].unique().sample(fraction=1.0, shuffle=True, seed=33)
        )
        n_test = max(1, int(len(match_ids) * 0.2))
        self.test_match_ids = match_ids.slice(0, n_test)
        self.train_match_ids = match_ids.slice(n_test)

        feature_cols = [
            c
            for c in self.match_diffs_complete.columns
            if c not in ("match_id", "year", "player_a_won")
        ]
        cont_cols = [
            c for c in feature_cols if self.match_diffs_complete[c].dtype == pl.Float64
        ]

        train_rows = self.match_diffs_complete.filter(
            pl.col("match_id").is_in(self.train_match_ids)
        )
        test_rows = self.match_diffs_complete.filter(
            pl.col("match_id").is_in(self.test_match_ids)
        )

        self.scaler = StandardScaler()
        scaled_train = self.scaler.fit_transform(
            train_rows.select(cont_cols).to_numpy()
        )
        scaled_test = self.scaler.transform(test_rows.select(cont_cols).to_numpy())

        self.X_train = train_rows.select(feature_cols).with_columns(
            pl.from_numpy(scaled_train, schema=cont_cols)
        )
        self.X_test = test_rows.select(feature_cols).with_columns(
            pl.from_numpy(scaled_test, schema=cont_cols)
        )
        self.y_train = train_rows["player_a_won"]
        self.y_test = test_rows["player_a_won"]
        self.continuous_feature_cols = cont_cols

        return self

    def process(self, force: bool = True) -> Self:
        """
        Run the full data pipeline, loading from cache if available.

        Steps: load → reshape → filter → features → match diffs → model prep → split.
        Caches player_matches_wfeat and match_diffs_complete to Parquet.

        Args:
            force (bool): If True, recompute even if cache exists. Defaults to True.

        Returns:
            Self: Updated dataset object.
        """
        pm_cache = self.processed_cache_dir / "player_matches_wfeat.parquet"
        complete_cache = self.processed_cache_dir / "match_diffs_complete.parquet"

        if not force and pm_cache.exists() and complete_cache.exists():
            print("Loading processed data from cache...")
            self.player_matches_wfeat = pl.read_parquet(pm_cache)
            self.match_diffs_complete = pl.read_parquet(complete_cache)
            self.split_data()
            return self

        print("Processing data...")
        self._original_df = self._load_matches()
        (
            self.derive_player_matches()
            .limit_player_matches()
            .add_features()
            .derive_match_diffs()
            .prepare_for_model()
        )

        self.player_matches_wfeat.write_parquet(pm_cache)
        self.match_diffs_complete.write_parquet(complete_cache)

        self.split_data()

        return self

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_pm_data(self) -> pl.DataFrame:
        """
        Return the per-player match DataFrame with all features including raw counts.

        Returns:
            pl.DataFrame: Player matches with features.
        """
        return self.player_matches_wfeat

    def get_model_data(self) -> pl.DataFrame:
        """
        Return the dummy-encoded model-ready match-diff DataFrame (all years).

        Returns:
            pl.DataFrame: Model-ready feature matrix (one row per match).
        """
        return self.match_diffs_complete

    def get_holdout_data(self) -> pl.DataFrame:
        """
        Return the holdout-year (2023) model-ready match-diff DataFrame.

        Returns:
            pl.DataFrame: Holdout feature matrix (one row per match, match-diff format).
        """
        return self.holdout
