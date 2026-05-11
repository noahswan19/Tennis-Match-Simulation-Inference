"""
End-to-end pipeline for tennis match outcome prediction.

Steps:
1. Load and process ATP match data (2003-2023), caching intermediates.
2. Compute or load Bayesian Beta posteriors per player × surface.
3. Train MLP, Random Forest, and XGBoost on observed match stats.
4. Evaluate each model on 2023 holdout via Monte Carlo simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_preparation import TennisMatchDataset, TRAIN_YEARS, HOLDOUT_YEAR
from src.distributions import BayesianBetaPosterior
from src.models import MLPTennisModel, RandomForestTennisModel, XGBoostTennisModel
from src.evaluation import eval_num_sims

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run the full training and evaluation pipeline."""

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading and processing data")
    print("=" * 60)
    dataset = TennisMatchDataset(
        years=list(TRAIN_YEARS) + [HOLDOUT_YEAR],
        raw_cache_dir=DATA_RAW_DIR,
        processed_cache_dir=DATA_PROCESSED_DIR,
    )
    dataset.process()

    pm_data = dataset.get_pm_data()
    holdout_data = dataset.get_holdout_data()

    # ------------------------------------------------------------------
    # 2. Bayesian posteriors
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Bayesian Beta posteriors")
    print("=" * 60)
    posterior = BayesianBetaPosterior(
        decay=1, prior_year=2003, prior_concentration=900.0, sample_concentration=150.0
    )
    posterior_cache = DATA_PROCESSED_DIR / "posteriors.parquet"
    force_posterior_rerun = False

    if not force_posterior_rerun and posterior_cache.exists():
        print("Loading posteriors from cache...")
        posterior.load(posterior_cache)
    else:
        posterior.fit(pm_data, cache_path=posterior_cache)

    # ------------------------------------------------------------------
    # 3. Model training
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Training models")
    print("=" * 60)

    feature_cols = list(dataset.X_train.columns)
    X_train_np = dataset.X_train.to_numpy().astype(np.float32)
    y_train_np = dataset.y_train.to_numpy().astype(np.int64)
    X_val_np = dataset.X_test.to_numpy().astype(np.float32)
    y_val_np = dataset.y_test.to_numpy().astype(np.int64)

    model_configs = [
        (MLPTennisModel, MODELS_DIR / "mlp.pt"),
        (RandomForestTennisModel, MODELS_DIR / "random_forest.joblib"),
        (XGBoostTennisModel, MODELS_DIR / "xgboost.ubj"),
    ]

    trained_models: dict[str, object] = {}
    for ModelClass, model_path in model_configs:
        model_name = ModelClass.__name__
        print(f"\n--- {model_name} ---")
        if model_path.exists():
            print(f"  Loading from {model_path}")
            model = ModelClass.load(model_path)
        else:
            model = ModelClass()
            if isinstance(model, MLPTennisModel):
                model.fit(X_train_np, y_train_np, X_val=X_val_np, y_val=y_val_np)
            else:
                model.tune(X_train_np, y_train_np, feature_names=feature_cols)
            model.save(model_path)
            print(f"  Saved to {model_path}")
        trained_models[model_name] = model

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Evaluating models on 2023 holdout")
    print("=" * 60)

    n_sims_list = list(range(100, 1001, 50))
    results: dict[str, object] = {}

    for model_name, model in trained_models.items():
        print(f"\n--- {model_name} ---")
        results[model_name] = eval_num_sims(
            model=model,
            posterior=posterior,
            holdout_data=holdout_data,
            player_matches_wfeat=pm_data,
            n_sims_list=n_sims_list,
            feature_cols=feature_cols,
            scaler=dataset.scaler,
            continuous_cols=dataset.continuous_feature_cols,
        )


if __name__ == "__main__":
    main()
