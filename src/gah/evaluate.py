from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .config import CITIES, TASKS, CityConfig
from .data import load_label


def task_arrays(
    embedding: np.ndarray,
    labels: np.ndarray,
    config: CityConfig,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(embedding, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.shape[0] != config.regions or y.shape[0] != config.regions:
        raise ValueError(f"{config.key}/{task}: embedding and label region axes do not match")
    if task == "checkIn" and config.key in {"chicago", "sf"}:
        keep = y > 0
        x, y = x[keep], y[keep]
    return x, y


def evaluate_array(
    embedding: np.ndarray,
    labels: np.ndarray,
    config: CityConfig,
    task: str,
) -> dict[str, object]:
    x, y = task_arrays(embedding, labels, config, task)
    splitter = KFold(
        n_splits=config.folds,
        shuffle=config.shuffle,
        random_state=config.cv_seed if config.shuffle else None,
    )
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    folds: list[dict[str, float]] = []
    for train_index, test_index in splitter.split(x):
        model = Ridge(alpha=1)
        model.fit(x[train_index], y[train_index])
        predicted = model.predict(x[test_index])
        predicted[predicted < 0] = 0
        truth = y[test_index]
        fold = {
            "mae": float(mean_absolute_error(truth, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
            "r2": float(r2_score(truth, predicted)),
        }
        folds.append(fold)
        predictions.append(predicted)
        truths.append(truth)
    predicted = np.concatenate(predictions)
    truth = np.concatenate(truths)
    overall = {
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
        "r2": float(r2_score(truth, predicted)),
    }
    fold_summary = {
        metric: {
            "mean": float(np.mean([fold[metric] for fold in folds])),
            "std": float(np.std([fold[metric] for fold in folds])),
        }
        for metric in ("mae", "rmse", "r2")
    }
    return {
        "city": config.key,
        "task": task,
        "protocol": {
            "regressor": "Ridge(alpha=1)",
            "folds": config.folds,
            "shuffle": config.shuffle,
            "random_state": config.cv_seed,
            "negative_predictions_clipped_to_zero": True,
        },
        "samples": int(len(y)),
        "overall": overall,
        "fold_summary": fold_summary,
        "fold_metrics": folds,
    }


def evaluate_file(embedding_path: Path, data_dir: Path, city: str, task: str) -> dict[str, object]:
    if city not in CITIES:
        raise ValueError(f"unknown city: {city}")
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    config = CITIES[city]
    return evaluate_array(np.load(embedding_path), load_label(data_dir, task), config, task)

