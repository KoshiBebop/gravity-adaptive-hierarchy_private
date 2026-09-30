from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import (
    CITIES,
    EMBEDDING_DIM,
    EPOCHS,
    LEARNING_RATE,
    SELECTION_INTERVAL,
    TASKS,
    TRAIN_SEED,
    WEIGHT_DECAY,
    HAFUSION_CITIES,
)
from .data import load_label, load_raw, load_release_prior, to_tensors
from .evaluate import evaluate_array
from .gravity import apply_output_transform, select_gravity_prior
from .losses import native_components, redundancy_loss
from .model import GravityAdaptiveHierarchy, HAFusionReference


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_state():
    return (
        random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    )


def _restore_rng(state) -> None:
    python_state, numpy_state, torch_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    torch.cuda.set_rng_state_all(cuda_state)


def train_city(
    model_name: str,
    city: str,
    data_root: Path,
    output_dir: Path,
    seed: int = TRAIN_SEED,
    epochs: int = EPOCHS,
    device_name: str = "cuda:0",
    gravity_alpha: float | None = None,
    gravity_roles: frozenset[str] | None = None,
    redundancy_multiplier: float = 1.0,
    semantic_multiplier: float = 1.0,
    output_transform: str | None = None,
    gravity_prior_variant: str = "full",
    learnable_coupling: bool = True,
    region_gravity_kernel: str = "equilibrium",
) -> dict[str, object]:
    if model_name not in {"gah", "hafusion"}:
        raise ValueError("model must be 'gah' or 'hafusion'")
    if city not in CITIES:
        raise ValueError(f"unknown city: {city}")
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("formal training requires a real CUDA device")
    if seed != TRAIN_SEED:
        raise ValueError(f"release training seed is fixed to {TRAIN_SEED}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if gravity_alpha is not None and gravity_alpha < 0.0:
        raise ValueError("gravity alpha must be non-negative")
    if redundancy_multiplier < 0.0 or semantic_multiplier < 0.0:
        raise ValueError("loss multipliers must be non-negative")

    config = CITIES[city] if model_name == "gah" else HAFUSION_CITIES[city]
    data_dir = data_root / config.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    seed_everything(seed)
    raw = load_raw(data_dir, config)
    features, mobility, poi_similarity, land_similarity = to_tensors(raw, device)
    frozen_prior = load_release_prior(data_root, city, raw)
    prior_array = select_gravity_prior(raw, frozen_prior, gravity_prior_variant)
    prior = torch.from_numpy(prior_array).to(device)
    reference = HAFusionReference(config, EMBEDDING_DIM).to(device)
    effective_alpha = config.gravity_alpha if gravity_alpha is None else gravity_alpha
    effective_roles = GravityAdaptiveHierarchy.GRAVITY_ROLES if gravity_roles is None else gravity_roles
    effective_transform = config.output_transform if output_transform is None else output_transform
    model = (
        GravityAdaptiveHierarchy(
            reference, prior, effective_alpha, effective_roles,
            learnable_coupling=learnable_coupling,
            region_gravity_kernel=region_gravity_kernel,
        ).to(device)
        if model_name == "gah"
        else reference
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    labels = {task: load_label(data_dir, task) for task in TASKS}
    best = {
        task: {"r2": 0.0, "epoch": None, "embedding": None, "metrics": None}
        for task in TASKS
    }
    trace: list[dict[str, object]] = []
    losses: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(epochs):
        model.train()
        outputs = model(features)
        mobility_component, poi_component, land_component = native_components(
            outputs, mobility, poi_similarity, land_similarity
        )
        if model_name == "gah":
            native_scale = mobility_component.detach().clamp_min(1.0)
            scale = native_scale if config.balance_native_scale else torch.ones_like(native_scale)
            regularizer = scale * redundancy_loss(model.out_feature())
            loss = mobility_component + config.semantic_weight * semantic_multiplier * scale * (
                poi_component + land_component
            ) + config.redundancy_weight * redundancy_multiplier * regularizer
        else:
            regularizer = torch.zeros_like(mobility_component)
            loss = mobility_component + poi_component + land_component
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append({
            "epoch": epoch,
            "mobility": float(mobility_component.detach()),
            "poi": float(poi_component.detach()),
            "land": float(land_component.detach()),
            "redundancy": float(regularizer.detach()),
            "total": float(loss.detach()),
        })

        if epoch % SELECTION_INTERVAL == 0:
            checkpoint = model.out_feature().detach().cpu().numpy().astype(np.float32)
            if model_name == "gah":
                checkpoint = apply_output_transform(checkpoint, effective_transform, prior_array)
            state = _rng_state()
            epoch_record: dict[str, object] = {"epoch": epoch, "tasks": {}}
            try:
                for task in TASKS:
                    result = evaluate_array(checkpoint, labels[task], config, task)
                    metrics = result["overall"]
                    assert isinstance(metrics, dict)
                    epoch_record["tasks"][task] = metrics  # type: ignore[index]
                    if best[task]["r2"] < metrics["r2"]:
                        best[task] = {
                            "r2": float(metrics["r2"]),
                            "epoch": epoch,
                            "embedding": checkpoint.copy(),
                            "metrics": metrics,
                        }
            finally:
                _restore_rng(state)
            trace.append(epoch_record)
            print(json.dumps(epoch_record, sort_keys=True), flush=True)

    model.eval()
    with torch.no_grad():
        model(features)
    final_embedding = model.out_feature().detach().cpu().numpy().astype(np.float32)
    if model_name == "gah":
        final_embedding = apply_output_transform(final_embedding, effective_transform, prior_array)
    np.save(output_dir / "final_embedding.npy", final_embedding)

    selected: dict[str, object] = {}
    for task in TASKS:
        embedding = best[task]["embedding"]
        if not isinstance(embedding, np.ndarray):
            raise RuntimeError(f"no positive-R2 checkpoint selected for {city}/{task}")
        path = output_dir / f"official_best_embedding_{task}.npy"
        np.save(path, embedding)
        evaluation = evaluate_array(embedding, labels[task], config, task)
        selected[task] = {
            "best_epoch": best[task]["epoch"],
            "metrics": evaluation["overall"],
            "embedding": path.name,
            "shape": list(embedding.shape),
            "dtype": str(embedding.dtype),
            "sha256": sha256_file(path),
        }

    payload: dict[str, object] = {
        "model": model_name,
        "city": city,
        "training_seed": seed,
        "epochs": epochs,
        "selection_interval": SELECTION_INTERVAL,
        "selection": "city-once, task-specific best R2",
        "config": config.to_dict(),
        "loss": {
            "semantic_weight": config.semantic_weight * semantic_multiplier if model_name == "gah" else 1.0,
            "redundancy_weight": config.redundancy_weight * redundancy_multiplier if model_name == "gah" else 0.0,
            "native_mobility_scale": model_name == "gah" and config.balance_native_scale,
        },
        "experiment": {
            "gravity_alpha": effective_alpha if model_name == "gah" else None,
            "gravity_roles": sorted(effective_roles) if model_name == "gah" else [],
            "redundancy_multiplier": redundancy_multiplier if model_name == "gah" else None,
            "semantic_multiplier": semantic_multiplier if model_name == "gah" else None,
            "output_transform": effective_transform if model_name == "gah" else "none",
            "gravity_prior_variant": gravity_prior_variant if model_name == "gah" else None,
            "learnable_coupling": learnable_coupling if model_name == "gah" else None,
            "region_gravity_kernel": region_gravity_kernel if model_name == "gah" else None,
        },
        "selected": selected,
        "runtime": {
            "duration_seconds": time.time() - started,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "trace": trace,
    }
    (output_dir / "selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "losses.json").write_text(json.dumps(losses, indent=2), encoding="utf-8")
    return payload
