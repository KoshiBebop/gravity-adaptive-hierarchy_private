from __future__ import annotations

import numpy as np


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 1e-30)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-30)


def sinkhorn_prior(
    logits: np.ndarray,
    row_marginal: np.ndarray,
    column_marginal: np.ndarray,
    iterations: int = 100,
) -> np.ndarray:
    kernel = np.exp(logits - np.max(logits))
    kernel = np.maximum(kernel, 1e-30)
    row_scale = np.ones_like(row_marginal)
    column_scale = np.ones_like(column_marginal)
    for _ in range(iterations):
        row_scale = row_marginal / np.maximum(kernel @ column_scale, 1e-30)
        column_scale = column_marginal / np.maximum(kernel.T @ row_scale, 1e-30)
    result = row_scale[:, None] * kernel * column_scale[None, :]
    return result / max(float(result.sum()), 1e-30)


def balanced_transport(raw: dict[str, np.ndarray]) -> np.ndarray:
    observed = np.maximum(raw["mob_adj"].astype(np.float64), 0.0)
    observed = observed / max(float(observed.sum()), 1e-30)
    row_marginal = observed.sum(axis=1)
    column_marginal = observed.sum(axis=0)
    mobility = raw["mob_dist"].astype(np.float64)
    normalized = mobility / np.maximum(np.linalg.norm(mobility, axis=1, keepdims=True), 1e-12)
    impedance = np.clip(1.0 - normalized @ normalized.T, 0.0, 2.0)
    logits = (
        np.log(np.maximum(row_marginal[:, None], 1e-30))
        + np.log(np.maximum(column_marginal[None, :], 1e-30))
        - 1.5 * impedance
    )
    coupling = sinkhorn_prior(logits, row_marginal, column_marginal)
    transport = coupling / np.maximum(row_marginal[:, None], 1e-30)
    return _row_normalize(transport)


def gravity_conflict_components(
    raw: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the transport and conflict components of the gravity prior."""
    transport = balanced_transport(raw)
    n = transport.shape[0]
    off_diagonal = ~np.eye(n, dtype=bool)
    transport = np.maximum(transport.astype(np.float64), 0.0)
    transport[~off_diagonal] = 0.0
    transport = transport / np.maximum(transport.sum(axis=1, keepdims=True), 1e-30)
    fallback = np.full((n, n), 1.0 / max(n - 1, 1), dtype=np.float64)
    fallback[~off_diagonal] = 0.0
    transport = np.where(transport.sum(axis=1, keepdims=True) > 1e-30, transport, fallback)

    poi = np.nan_to_num(raw["poi_simi"].astype(np.float64), nan=0.0)
    poi = np.clip(0.5 * (poi + poi.T), 0.0, 1.0)
    mobility = np.nan_to_num(raw["mob_dist"].astype(np.float64), nan=0.0)
    mobility = mobility / np.maximum(np.linalg.norm(mobility, axis=1, keepdims=True), 1e-12)
    mobility = np.clip(0.5 * (mobility @ mobility.T + (mobility @ mobility.T).T) * 0.5 + 0.5, 0.0, 1.0)
    agreement = np.sqrt(poi * mobility)
    agreement[~off_diagonal] = 0.0

    positive = transport * (0.25 + 0.75 * agreement)
    positive[~off_diagonal] = 0.0
    positive = positive / np.maximum(positive.sum(axis=1, keepdims=True), 1e-30)
    positive = np.where(positive.sum(axis=1, keepdims=True) > 1e-30, positive, fallback)

    land = np.nan_to_num(raw["land_simi"].astype(np.float64), nan=0.0)
    land = np.clip(0.5 * (land + land.T), 0.0, 1.0)
    gravity_peak = transport / np.maximum(transport.max(axis=1, keepdims=True), 1e-30)
    hard_negative = land * (1.0 - gravity_peak)
    hard_negative[~off_diagonal] = 0.0
    negative_weights = np.ones((n, n), dtype=np.float64) + hard_negative
    negative_weights[~off_diagonal] = 0.0

    conflict = _row_normalize(positive / (1.0 + negative_weights))
    return transport.astype(np.float32), conflict.astype(np.float32)


def gravity_conflict_prior(raw: dict[str, np.ndarray]) -> np.ndarray:
    """Numerically compatible reconstruction of the released GAH prior."""
    transport, conflict = gravity_conflict_components(raw)
    prior = _row_normalize(np.sqrt(np.maximum(transport * conflict, 1e-30)))
    return prior.astype(np.float32)


def select_gravity_prior(
    raw: dict[str, np.ndarray], frozen: np.ndarray, variant: str
) -> np.ndarray:
    """Create one controlled prior ablation while preserving float32 shape."""
    if variant == "full":
        return np.asarray(frozen, dtype=np.float32)
    if variant == "uniform":
        n = frozen.shape[0]
        return np.full((n, n), 1.0 / n, dtype=np.float32)
    if variant == "symmetric":
        return _row_normalize(0.5 * (frozen + frozen.T)).astype(np.float32)
    transport, conflict = gravity_conflict_components(raw)
    if variant == "transport-only":
        return _row_normalize(transport).astype(np.float32)
    if variant == "conflict-only":
        return _row_normalize(conflict).astype(np.float32)
    raise ValueError(f"unsupported gravity prior variant: {variant}")


def apply_output_transform(
    embedding: np.ndarray,
    transform: str,
    gravity_prior: np.ndarray,
) -> np.ndarray:
    x = np.asarray(embedding, dtype=np.float64)
    if transform == "none":
        return x.astype(np.float32)
    if transform == "feature-standardize":
        weights = np.ones(x.shape[0], dtype=np.float64)
    elif transform == "gravity-confidence-feature-standardize":
        prior = np.maximum(np.asarray(gravity_prior, dtype=np.float64), 1e-30)
        destination_mass = prior.sum(axis=0)
        entropy = -(prior * np.log(prior)).sum(axis=1)
        max_entropy = max(float(np.log(max(prior.shape[1], 2))), 1e-12)
        confidence = np.clip(1.0 - entropy / max_entropy, 0.10, 1.0)
        weights = (prior.T @ confidence) / np.maximum(destination_mass, 1e-30)
    else:
        raise ValueError(f"unsupported output transform: {transform}")
    weights = np.maximum(weights, 1e-30)
    weights /= max(float(weights.sum()), 1e-30)
    mean = (weights[:, None] * x).sum(axis=0, keepdims=True)
    centered = x - mean
    variance = (weights[:, None] * np.square(centered)).sum(axis=0, keepdims=True)
    return (centered / np.maximum(np.sqrt(variance), 1e-6)).astype(np.float32)
