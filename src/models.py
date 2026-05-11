"""
Abstract base class and concrete implementations for tennis match prediction models.

Includes MLP (PyTorch Lightning), Random Forest (scikit-learn), and XGBoost classifiers.
All models expose a unified interface: fit, predict_proba, save, load, tune.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TennisModel(ABC):
    """
    Abstract base class for tennis match outcome prediction models.

    All concrete subclasses must implement fit, predict_proba, save, and load.
    A default tune() method using RandomizedSearchCV is provided but may be
    overridden (e.g., MLP raises NotImplementedError).
    """

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        """
        Train the model on the provided data.

        Args:
            X_train (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y_train (np.ndarray): Binary labels of shape (n_samples,).

        Returns:
            Self: The fitted model.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities of shape (n_samples, 2) — [P(loss), P(win)].
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Serialize the trained model to disk.

        Args:
            path (Path): Destination file path.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """
        Deserialize a model from disk.

        Args:
            path (Path): Source file path.

        Returns:
            Self: Loaded model instance.
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted class labels (argmax of predict_proba).

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels of shape (n_samples,).
        """
        return self.predict_proba(X).argmax(axis=1)

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_distributions: dict,
        n_iter: int = 50,
        cv: int = 5,
        random_state: int = 42,
    ) -> Self:
        """
        Hyperparameter search using RandomizedSearchCV; fit the best estimator.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
            param_distributions (dict): Parameter distributions for search.
            n_iter (int): Number of random candidates. Defaults to 50.
            cv (int): Cross-validation folds. Defaults to 5.
            random_state (int): Random seed. Defaults to 42.

        Returns:
            Self: Model fitted with best hyperparameters.

        Raises:
            NotImplementedError: Always; subclasses must override or use fit() directly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement tune() via RandomizedSearchCV. "
            "Call fit() directly with desired hyperparameters."
        )


# ---------------------------------------------------------------------------
# MLP (PyTorch Lightning)
# ---------------------------------------------------------------------------


class _MLPModule(L.LightningModule):
    """
    Inner LightningModule for the MLP tennis model.

    One hidden layer with ReLU activation and dropout.

    Attributes:
        net (nn.Sequential): The feed-forward network.
        lr (float): Learning rate for SGD optimizer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        dropout: float = 0.25,
        lr: float = 0.01,
    ) -> None:
        """
        Initialize the MLP architecture.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Neurons in the hidden layer. Defaults to 16.
            dropout (float): Dropout rate. Defaults to 0.25.
            lr (float): Learning rate for SGD. Defaults to 0.01.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim).

        Returns:
            torch.Tensor: Logits of shape (batch, 2).
        """
        return self.net(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Single training step.

        Args:
            batch (tuple): (X, y) tensors.
            batch_idx (int): Batch index (unused).

        Returns:
            torch.Tensor: Scalar loss.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Single validation step.

        Args:
            batch (tuple): (X, y) tensors.
            batch_idx (int): Batch index (unused).
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure SGD optimizer.

        Returns:
            torch.optim.Optimizer: SGD optimizer.
        """
        return torch.optim.SGD(self.parameters(), lr=self.lr)


class MLPTennisModel(TennisModel):
    """
    MLP classifier wrapped around PyTorch Lightning.

    Trains a single-hidden-layer network with early stopping on validation loss.
    Hyperparameter tuning via RandomizedSearchCV is not supported; use fit()
    directly with explicit hyperparameters.

    Attributes:
        hidden_dim (int): Neurons in the hidden layer.
        dropout (float): Dropout rate.
        lr (float): Learning rate.
        max_epochs (int): Maximum training epochs.
        batch_size (int): Mini-batch size.
        _module (_MLPModule | None): Underlying Lightning module after fitting.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        dropout: float = 0.25,
        lr: float = 0.01,
        max_epochs: int = 500,
        batch_size: int = 256,
    ) -> None:
        """
        Initialize the MLP model configuration.

        Args:
            hidden_dim (int): Neurons in hidden layer. Defaults to 16.
            dropout (float): Dropout probability. Defaults to 0.25.
            lr (float): SGD learning rate. Defaults to 0.01.
            max_epochs (int): Maximum training epochs. Defaults to 500.
            batch_size (int): Training batch size. Defaults to 256.
        """
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self._module: _MLPModule | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> Self:
        """
        Train the MLP with early stopping on a validation set.

        Args:
            X_train (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y_train (np.ndarray): Binary labels of shape (n_samples,).
            X_val (np.ndarray | None): Validation features for early stopping.
            y_val (np.ndarray | None): Validation labels for early stopping.

        Returns:
            Self: Fitted model.
        """
        if X_val is None or y_val is None:
            raise ValueError(
                "MLPTennisModel requires X_val and y_val for early stopping."
            )
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)

        X_v = torch.tensor(X_val, dtype=torch.float32)
        y_v = torch.tensor(y_val, dtype=torch.long)
        train_ds = TensorDataset(X_t, y_t)
        val_ds = TensorDataset(X_v, y_v)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        self._module = _MLPModule(
            input_dim=X_train.shape[1],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            lr=self.lr,
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=20, mode="min", verbose=False
        )

        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stop],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(self._module, train_loader, val_loader)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities using softmax over logits.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities of shape (n_samples, 2).
        """
        self._module.eval()
        with torch.no_grad():
            logits = self._module(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def save(self, path: Path) -> None:
        """
        Save the model state dict and hyperparameters to disk.

        Args:
            path (Path): Destination .pt file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._module.state_dict(),
                "hparams": self._module.hparams,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Load a model from a saved .pt checkpoint.

        Args:
            path (Path): Source .pt file path.

        Returns:
            Self: Loaded model instance.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hparams = ckpt["hparams"]
        instance = cls(
            hidden_dim=hparams.get("hidden_dim", 16),
            dropout=hparams.get("dropout", 0.25),
            lr=hparams.get("lr", 0.01),
        )
        instance._module = _MLPModule(**hparams)
        instance._module.load_state_dict(ckpt["state_dict"])
        instance._module.eval()
        return instance

    def tune(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Self:
        """
        Not implemented for MLP. Use fit() with explicit hyperparameters.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "MLPTennisModel does not support RandomizedSearchCV tuning. "
            "Call fit() directly."
        )


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


class RandomForestTennisModel(TennisModel):
    """
    Random Forest classifier wrapping scikit-learn RandomForestClassifier.

    Supports hyperparameter tuning via RandomizedSearchCV over n_estimators
    and max_depth.

    Attributes:
        _model (RandomForestClassifier | None): Fitted classifier.
    """

    def __init__(self) -> None:
        """Initialize with no fitted model."""
        self._model: RandomForestClassifier | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        """
        Fit the Random Forest classifier.

        Args:
            X_train (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y_train (np.ndarray): Binary labels of shape (n_samples,).

        Returns:
            Self: Fitted model.
        """
        if self._model is None:
            self._model = RandomForestClassifier(random_state=42)
        self._model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities of shape (n_samples, 2).
        """
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        """
        Save the model using joblib.

        Args:
            path (Path): Destination .joblib file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Load a joblib-serialized Random Forest model.

        Args:
            path (Path): Source file path.

        Returns:
            Self: Loaded model instance.
        """
        instance = cls()
        instance._model = joblib.load(path)
        return instance

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_distributions: dict | None = None,
        n_iter: int = 15,
        cv: int = 3,
        random_state: int = 42,
        feature_names: list[str] | None = None,
    ) -> Self:
        """
        Run RandomizedSearchCV over n_estimators and max_depth, then fit best.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
            param_distributions (dict | None): Override default search space. Defaults to None.
            n_iter (int): Number of random candidates. Defaults to 15.
            cv (int): Cross-validation folds. Defaults to 3.
            random_state (int): Random seed. Defaults to 42.
            feature_names (list[str] | None): Accepted for interface consistency; unused. Defaults to None.

        Returns:
            Self: Model fitted with best hyperparameters.
        """
        if param_distributions is None:
            param_distributions = {
                "n_estimators": [50, 100, 200, 300, 500],
                "max_depth": [5, 10, 15, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.3, 0.5],
            }

        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=random_state),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_log_loss",
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X, y)
        self._model = search.best_estimator_
        print(f"Best RF params: {search.best_params_}")
        return self


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


class XGBoostTennisModel(TennisModel):
    """
    XGBoost classifier wrapping xgboost.XGBClassifier.

    Supports hyperparameter tuning via RandomizedSearchCV over max_depth,
    n_estimators, and learning_rate (eta).

    Attributes:
        _model (XGBClassifier | None): Fitted classifier.
    """

    def __init__(self) -> None:
        """Initialize with no fitted model."""
        self._model: XGBClassifier | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> Self:
        """
        Fit the XGBoost classifier.

        Args:
            X_train (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y_train (np.ndarray): Binary labels of shape (n_samples,).
            feature_names (list[str] | None): Column names stored on the booster for
                interpretability. Defaults to None.

        Returns:
            Self: Fitted model.
        """
        if self._model is None:
            self._model = XGBClassifier(
                eval_metric="logloss",
                random_state=42,
            )
        self._model.fit(X_train, y_train)
        if feature_names is not None:
            self._model.get_booster().feature_names = list(feature_names)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities of shape (n_samples, 2).
        """
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        """
        Save the model in XGBoost binary JSON format (.ubj).

        Args:
            path (Path): Destination .ubj file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Load an XGBoost model from a .ubj file.

        Args:
            path (Path): Source .ubj file path.

        Returns:
            Self: Loaded model instance.
        """
        instance = cls()
        instance._model = XGBClassifier()
        instance._model.load_model(path)
        return instance

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_distributions: dict | None = None,
        n_iter: int = 15,
        cv: int = 5,
        random_state: int = 42,
        feature_names: list[str] | None = None,
    ) -> Self:
        """
        Run RandomizedSearchCV over max_depth, n_estimators, and learning_rate.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
            param_distributions (dict | None): Override default search space. Defaults to None.
            n_iter (int): Number of random candidates. Defaults to 15.
            cv (int): Cross-validation folds. Defaults to 5.
            random_state (int): Random seed. Defaults to 42.
            feature_names (list[str] | None): Column names stored on the booster for
                interpretability. Defaults to None.

        Returns:
            Self: Model fitted with best hyperparameters.
        """
        if param_distributions is None:
            param_distributions = {
                "max_depth": [3, 4, 5, 6, 7, 8],
                "n_estimators": [50, 100, 200, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            }

        base = XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
        )
        search = RandomizedSearchCV(
            base,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_log_loss",
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X, y)
        self._model = search.best_estimator_
        if feature_names is not None:
            self._model.get_booster().feature_names = list(feature_names)
        print(f"Best XGB params: {search.best_params_}")
        return self
