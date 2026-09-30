"""Download missing source data, train all three cities, then test."""

from pathlib import Path

from gah.config import CITIES
from gah.download import ensure_data
from gah.train import train_city

from test import main as test_all


ROOT = Path(__file__).resolve().parent


def main() -> None:
    ensure_data(ROOT)
    for city in CITIES:
        print(f"Training {city}...", flush=True)
        train_city("gah", city, ROOT, ROOT / "outputs" / "gah" / city)
    test_all()


if __name__ == "__main__":
    main()

