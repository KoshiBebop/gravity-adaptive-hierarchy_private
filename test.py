"""Evaluate the task-best embeddings produced by train.py."""

import json
from pathlib import Path

from gah.config import CITIES, TASKS
from gah.download import ensure_data
from gah.evaluate import evaluate_file


ROOT = Path(__file__).resolve().parent


def main() -> None:
    ensure_data(ROOT)
    results = {}
    for city, config in CITIES.items():
        results[city] = {}
        for task in TASKS:
            embedding = ROOT / "outputs" / "gah" / city / f"official_best_embedding_{task}.npy"
            result = evaluate_file(embedding, ROOT / config.data_dir, city, task)
            results[city][task] = result["overall"]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

