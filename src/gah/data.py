from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import CityConfig


ARRAY_FILES = {
    "poi_dist": "poi_dist.npy",
    "land_dist": "landUse_dist.npy",
    "mob_dist": "mob_dist.npy",
    "mob_adj": "mob-adj.npy",
    "poi_simi": "poi_simi.npy",
    "land_simi": "landUse_simi.npy",
}

LABEL_FILES = {
    "crime": "crime_counts.npy",
    "checkIn": "check_counts.npy",
    "serviceCall": "serviceCall_counts.npy",
}


def load_raw(data_dir: Path, config: CityConfig) -> dict[str, np.ndarray]:
    raw = {
        name: np.load(data_dir / filename).astype(np.float32)
        for name, filename in ARRAY_FILES.items()
    }
    n = config.regions
    if raw["mob_adj"].shape != (n, n):
        raise ValueError(f"{config.key}: expected mobility shape {(n, n)}, got {raw['mob_adj'].shape}")
    if raw["poi_dist"].shape != (n, config.poi_dim):
        raise ValueError(f"{config.key}: invalid POI feature shape {raw['poi_dist'].shape}")
    if raw["land_dist"].shape != (n, config.land_dim):
        raise ValueError(f"{config.key}: invalid land-use feature shape {raw['land_dist'].shape}")
    if raw["mob_dist"].shape != (n, n):
        raise ValueError(f"{config.key}: invalid mobility feature shape {raw['mob_dist'].shape}")
    return raw


def to_tensors(raw: dict[str, np.ndarray], device: torch.device):
    mobility = raw["mob_adj"] / max(float(raw["mob_adj"].mean()), 1e-12)
    features = [
        torch.from_numpy(raw["poi_dist"][None]).to(device),
        torch.from_numpy(raw["land_dist"][None]).to(device),
        torch.from_numpy(raw["mob_dist"][None]).to(device),
    ]
    return (
        features,
        torch.from_numpy(mobility).to(device),
        torch.from_numpy(raw["poi_simi"]).to(device),
        torch.from_numpy(raw["land_simi"]).to(device),
    )


def load_label(data_dir: Path, task: str) -> np.ndarray:
    try:
        filename = LABEL_FILES[task]
    except KeyError as exc:
        raise ValueError(f"unknown task: {task}") from exc
    return np.load(data_dir / filename)


def load_release_prior(data_root: Path, city: str, raw: dict[str, np.ndarray]) -> np.ndarray:
    """Load the frozen float32 prior, or deterministically rebuild a compatible fallback.

    The frozen matrix removes one-ULP differences caused by reduction ordering in
    independent NumPy implementations. Those differences can change a long
    stochastic optimization trajectory even when the mathematical prior is the same.
    """
    path = data_root / "priors" / f"{city}.npy"
    if path.is_file():
        prior = np.load(path)
    else:
        from .gravity import gravity_conflict_prior

        prior = gravity_conflict_prior(raw)
    expected = (CITIES_REGIONS[city], CITIES_REGIONS[city])
    if prior.shape != expected or prior.dtype != np.float32 or not np.isfinite(prior).all():
        raise ValueError(f"invalid frozen gravity prior for {city}: {prior.shape} {prior.dtype}")
    return prior


CITIES_REGIONS = {"nyc": 180, "chicago": 77, "sf": 175}
