from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class CityConfig:
    key: str
    data_dir: str
    regions: int
    poi_dim: int
    land_dim: int
    intra_layers: int
    inter_layers: int
    region_layers: int
    heads: int
    d_prime: int
    d_m: int
    channels: int = 32
    output_transform: str = "feature-standardize"
    gravity_alpha: float = 1.0
    folds: int = 10
    shuffle: bool = True
    cv_seed: int | None = 2024
    semantic_weight: float = 1.0
    redundancy_weight: float = 0.005
    balance_native_scale: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CITIES = {
    "nyc": CityConfig(
        key="nyc", data_dir="data_NY", regions=180, poi_dim=26, land_dim=11,
        intra_layers=2, inter_layers=2, region_layers=2, heads=4,
        d_prime=64, d_m=72,
        output_transform="gravity-confidence-feature-standardize",
        gravity_alpha=0.95, folds=10, shuffle=False, cv_seed=None,
        semantic_weight=1.25e-5, redundancy_weight=1e-6,
        balance_native_scale=True,
    ),
    "chicago": CityConfig(
        key="chicago", data_dir="data_Chi", regions=77, poi_dim=26, land_dim=12,
        intra_layers=3, inter_layers=4, region_layers=3, heads=1,
        d_prime=32, d_m=36, folds=25, shuffle=True, cv_seed=2024,
    ),
    "sf": CityConfig(
        key="sf", data_dir="data_SF", regions=175, poi_dim=26, land_dim=23,
        intra_layers=3, inter_layers=2, region_layers=3, heads=5,
        d_prime=64, d_m=72, folds=10, shuffle=True, cv_seed=2024,
    ),
}

# The seed-2026 reference artifacts use each city's unmodified official
# HAFusion architecture. GAH has its own frozen city-level architecture above.
HAFUSION_CITIES = {
    "nyc": replace(CITIES["nyc"], intra_layers=3, inter_layers=3, region_layers=3),
    "chicago": replace(CITIES["chicago"], intra_layers=1, inter_layers=2, region_layers=3),
    "sf": replace(CITIES["sf"], intra_layers=3, inter_layers=2, region_layers=3),
}

TASKS = ("crime", "checkIn", "serviceCall")
TRAIN_SEED = 2026
EPOCHS = 2000
SELECTION_INTERVAL = 30
EMBEDDING_DIM = 144
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4
