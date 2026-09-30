from __future__ import annotations

import torch
import torch.nn.functional as F


def mobility_loss(source: torch.Tensor, target: torch.Tensor, mobility: torch.Tensor) -> torch.Tensor:
    return -(mobility * torch.log_softmax(source @ target.T, dim=-1)).sum() - (
        mobility.T * torch.log_softmax(target @ source.T, dim=-1)
    ).sum()


def geometry_loss(embedding: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    similarity = F.cosine_similarity(embedding.unsqueeze(1), embedding.unsqueeze(0), dim=2)
    return F.mse_loss(similarity, adjacency)


def native_components(outputs, mobility, poi_similarity, land_similarity):
    source, target, poi, land = outputs
    return (
        mobility_loss(source, target, mobility),
        geometry_loss(poi, poi_similarity),
        geometry_loss(land, land_similarity),
    )


def redundancy_loss(embedding: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embedding, dim=1, eps=1e-8)
    centered = normalized - normalized.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    off_diagonal = ~torch.eye(covariance.shape[0], device=embedding.device, dtype=torch.bool)
    decorrelation = covariance[off_diagonal].square().mean()
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance_floor = F.relu(0.50 - standard_deviation).square().mean()
    return decorrelation + variance_floor

