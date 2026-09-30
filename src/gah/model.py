from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import CityConfig, EMBEDDING_DIM


class Projection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.Linear(2 * input_dim, 2 * input_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(2 * input_dim, output_dim),
            nn.LeakyReLU(0.3),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class BaselineIntraBlock(nn.Module):
    def __init__(self, dim: int, heads: int, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dim, 2048)
        self.linear2 = nn.Linear(2048, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.expand = nn.Conv2d(1, channels, 1)
        self.pooling = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.proj = nn.Linear(channels, dim)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        attended, weights = self.self_attn(source, source, source)
        edge = self.expand(weights[:, None])
        edge = (edge.softmax(dim=-1) * edge).sum(dim=-1).transpose(-1, -2)
        attended = attended + self.proj(edge)
        source = self.norm1(source + self.dropout1(attended))
        feed = self.linear2(self.dropout(F.relu(self.linear1(source))))
        return self.norm2(source + self.dropout2(feed))


class BaselineIntra(nn.Module):
    def __init__(self, dim: int, config: CityConfig) -> None:
        super().__init__()
        self.input_dim = dim
        self.blocks = nn.ModuleList([
            BaselineIntraBlock(dim, config.heads, config.channels)
            for _ in range(config.intra_layers)
        ])
        self.fc = Projection(dim, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values)
        return self.fc(values.squeeze(0))


class BaselineRegionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dim, 2048)
        self.linear2 = nn.Linear(2048, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attn(source, source, source)
        source = self.norm1(source + self.dropout1(attended))
        feed = self.linear2(self.dropout(F.relu(self.linear1(source))))
        return self.norm2(source + self.dropout2(feed))


class BaselineRegion(nn.Module):
    def __init__(self, dim: int, config: CityConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            BaselineRegionBlock(dim, config.heads) for _ in range(config.region_layers)
        ])
        self.fc = Projection(dim, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values)
        return self.fc(values.squeeze(0))


class BaselineInterBlock(nn.Module):
    def __init__(self, dim: int, slots: int) -> None:
        super().__init__()
        self.mk = nn.Linear(dim, slots, bias=False)
        self.mv = nn.Linear(slots, dim, bias=False)
        nn.init.normal_(self.mk.weight, std=0.001)
        nn.init.normal_(self.mv.weight, std=0.001)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        allocation = torch.softmax(self.mk(queries), dim=1)
        allocation = allocation / allocation.sum(dim=2, keepdim=True)
        return self.mv(allocation)


class BaselineInter(nn.Module):
    def __init__(self, dim: int, config: CityConfig) -> None:
        super().__init__()
        self.input_dim = dim
        self.blocks = nn.ModuleList([
            BaselineInterBlock(dim, config.d_m) for _ in range(config.inter_layers)
        ])
        self.fc = Projection(dim, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values)
        return self.fc(values.squeeze())


class BaselineViewFusion(nn.Module):
    def __init__(self, dim: int, adaptive_dim: int) -> None:
        super().__init__()
        self.W = nn.Conv1d(dim, adaptive_dim, 1, bias=False)
        self.f1 = nn.Conv1d(adaptive_dim, 1, 1)
        self.f2 = nn.Conv1d(adaptive_dim, 1, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = self.W(values)
        logits = self.f1(projected) + self.f2(projected).transpose(1, 2)
        return torch.softmax(F.leaky_relu(logits, 0.3).mean(dim=-1).mean(dim=0), dim=-1)


class HAFusionReference(nn.Module):
    """Numerically faithful HAFusion reference used as the baseline and initializer."""

    def __init__(self, config: CityConfig, output_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        dim = config.regions
        self.input_dim = dim
        self.densePOI2 = nn.Linear(config.poi_dim, dim)
        self.denseLandUse3 = nn.Linear(config.land_dim, dim)
        self.encoderPOI = BaselineIntra(dim, config)
        self.encoderLandUse = BaselineIntra(dim, config)
        self.encoderMob = BaselineIntra(dim, config)
        self.regionFusionLayer = BaselineRegion(dim, config)
        self.interViewEncoder = BaselineInter(dim, config)
        self.fc = Projection(dim, output_dim)
        self.para1 = nn.Parameter(torch.tensor([0.1]))
        self.para2 = nn.Parameter(torch.tensor([0.9]))
        self.viewFusionLayer = BaselineViewFusion(dim, config.d_prime)
        self.dropout = nn.Dropout(0.1)
        self.decoder_s = nn.Linear(output_dim, output_dim)
        self.decoder_t = nn.Linear(output_dim, output_dim)
        self.decoder_p = nn.Linear(output_dim, output_dim)
        self.decoder_l = nn.Linear(output_dim, output_dim)
        self.feature: torch.Tensor | None = None

    def forward(self, features: list[torch.Tensor]):
        poi, land, mobility = features
        poi = self.dropout(F.relu(self.densePOI2(poi)))
        land = self.dropout(F.relu(self.denseLandUse3(land)))
        intra = torch.stack([
            self.encoderPOI(poi), self.encoderLandUse(land), self.encoderMob(mobility)
        ])
        inter = self.interViewEncoder(intra.transpose(0, 1)).transpose(0, 1)
        p1 = self.para1 / (self.para1 + self.para2)
        # Keep the official graph explicit: replacing p2 by 1-p1 is forward-
        # equivalent but changes gradients for para1/para2 and the trajectory.
        p2 = self.para2 / (self.para1 + self.para2)
        states = inter * p2 + intra * p1
        weights = self.viewFusionLayer(states.transpose(0, 2))
        fused = sum(weights[index] * states[index] for index in range(3))[None]
        feature = self.fc(self.regionFusionLayer(fused))
        self.feature = feature
        return (
            self.decoder_s(feature), self.decoder_t(feature),
            self.decoder_p(feature), self.decoder_l(feature),
        )

    def out_feature(self) -> torch.Tensor:
        if self.feature is None:
            raise RuntimeError("out_feature called before forward")
        return self.feature


class GravityIntraBlock(nn.Module):
    def __init__(self, reference: BaselineIntraBlock, prior: torch.Tensor) -> None:
        super().__init__()
        dim = reference.self_attn.embed_dim
        channels = reference.expand.out_channels
        self.register_buffer("prior", prior.detach() / prior.detach().sum(dim=1, keepdim=True).clamp_min(1e-12))
        self.attention = nn.MultiheadAttention(dim, reference.self_attn.num_heads, dropout=0.1, batch_first=True, bias=True)
        self.expand = nn.Conv2d(1, channels, 1)
        self.projection = nn.Linear(channels, dim)
        self.linear1 = nn.Linear(dim, reference.linear1.out_features)
        self.linear2 = nn.Linear(reference.linear1.out_features, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.attention.load_state_dict(reference.self_attn.state_dict())
        self.expand.load_state_dict(reference.expand.state_dict())
        self.projection.load_state_dict(reference.proj.state_dict())
        self.linear1.load_state_dict(reference.linear1.state_dict())
        self.linear2.load_state_dict(reference.linear2.state_dict())
        self.norm1.load_state_dict(reference.norm1.state_dict())
        self.norm2.load_state_dict(reference.norm2.state_dict())

    def forward(self, source: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
        attended, weights = self.attention(
            source, source, source,
            attn_mask=coupling * torch.log(self.prior.to(source).clamp_min(1e-30)),
            need_weights=True,
        )
        edge = self.expand(weights.unsqueeze(1))
        edge = (torch.softmax(edge, dim=-1) * edge).sum(dim=-1).transpose(-1, -2)
        attended = attended + self.projection(edge)
        source = self.norm1(source + self.dropout1(attended))
        feed = self.linear2(self.dropout(F.relu(self.linear1(source))))
        return self.norm2(source + self.dropout2(feed))


class GravityIntra(nn.Module):
    def __init__(self, reference: BaselineIntra, prior: torch.Tensor) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([GravityIntraBlock(block, prior) for block in reference.blocks])
        self.output = Projection(reference.input_dim, reference.input_dim)
        self.output.network.load_state_dict(reference.fc.network.state_dict())

    def forward(self, values: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values, coupling)
        return self.output(values.squeeze(0))


class GravityInterBlock(nn.Module):
    def __init__(self, reference: BaselineInterBlock, prior: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("prior", prior.detach() / prior.detach().sum(dim=1, keepdim=True).clamp_min(1e-12))
        self.key = nn.Linear(reference.mk.in_features, reference.mk.out_features, bias=False)
        self.value = nn.Linear(reference.mv.in_features, reference.mv.out_features, bias=False)
        self.key.load_state_dict(reference.mk.state_dict())
        self.value.load_state_dict(reference.mv.state_dict())

    def forward(self, queries: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
        transported = torch.einsum("ij,jvd->ivd", self.prior.to(queries), queries)
        routed = queries + coupling * (transported - queries)
        allocation = torch.softmax(self.key(routed), dim=1)
        allocation = allocation / allocation.sum(dim=2, keepdim=True).clamp_min(1e-12)
        return self.value(allocation)


class GravityInter(nn.Module):
    def __init__(self, reference: BaselineInter, prior: torch.Tensor) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([GravityInterBlock(block, prior) for block in reference.blocks])
        self.output = Projection(reference.input_dim, reference.input_dim)
        self.output.network.load_state_dict(reference.fc.network.state_dict())

    def forward(self, values: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values, coupling)
        return self.output(values)


class GravityRegionBlock(nn.Module):
    def __init__(self, reference: BaselineRegionBlock, prior: torch.Tensor) -> None:
        super().__init__()
        dim = reference.self_attn.embed_dim
        self.register_buffer("prior", prior.detach() / prior.detach().sum(dim=1, keepdim=True).clamp_min(1e-12))
        self.attention = nn.MultiheadAttention(dim, reference.self_attn.num_heads, dropout=0.1, batch_first=True, bias=True)
        self.linear1 = nn.Linear(dim, reference.linear1.out_features)
        self.linear2 = nn.Linear(reference.linear1.out_features, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.attention.load_state_dict(reference.self_attn.state_dict())
        self.linear1.load_state_dict(reference.linear1.state_dict())
        self.linear2.load_state_dict(reference.linear2.state_dict())
        self.norm1.load_state_dict(reference.norm1.state_dict())
        self.norm2.load_state_dict(reference.norm2.state_dict())

    def forward(self, source: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
        context, _ = self.attention(
            source, source, source,
            attn_mask=coupling * torch.log(self.prior.to(source).clamp_min(1e-30)),
            need_weights=False,
        )
        source = self.norm1(source + self.dropout1(context))
        feed = self.linear2(self.dropout(F.relu(self.linear1(source))))
        return self.norm2(source + self.dropout2(feed))


class GravityAdaptiveHierarchy(nn.Module):
    GRAVITY_ROLES = frozenset({"intra", "inter", "view", "region"})

    def __init__(
        self,
        reference: HAFusionReference,
        prior: torch.Tensor,
        alpha: float,
        gravity_roles: frozenset[str] | None = None,
        learnable_coupling: bool = True,
        region_gravity_kernel: str = "equilibrium",
    ) -> None:
        super().__init__()
        roles = self.GRAVITY_ROLES if gravity_roles is None else frozenset(gravity_roles)
        unknown = roles - self.GRAVITY_ROLES
        if unknown:
            raise ValueError(f"unknown gravity roles: {sorted(unknown)}")
        if region_gravity_kernel not in {"equilibrium", "force"}:
            raise ValueError("region gravity kernel must be 'equilibrium' or 'force'")
        self.gravity_roles = roles
        dim = reference.input_dim
        output_dim = reference.decoder_s.in_features
        force = prior.detach() / prior.detach().sum(dim=1, keepdim=True).clamp_min(1e-12)
        reverse = force.T / force.T.sum(dim=1, keepdim=True).clamp_min(1e-12)
        equilibrium = 0.5 * (force + reverse)
        equilibrium = equilibrium / equilibrium.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("force", force)
        self.register_buffer("equilibrium", equilibrium)
        initial_logit = torch.tensor(-4.59511985)
        if learnable_coupling:
            self.gravity_logit = nn.Parameter(initial_logit)
        else:
            self.register_buffer("gravity_logit", initial_logit)
        self.learnable_coupling = learnable_coupling
        self.gravity_routing_alpha = float(alpha)
        self.poi_projection = nn.Linear(reference.densePOI2.in_features, dim)
        self.land_projection = nn.Linear(reference.denseLandUse3.in_features, dim)
        self.poi_projection.load_state_dict(reference.densePOI2.state_dict())
        self.land_projection.load_state_dict(reference.denseLandUse3.state_dict())
        self.intra = nn.ModuleList([
            GravityIntra(reference.encoderPOI, force),
            GravityIntra(reference.encoderLandUse, force),
            GravityIntra(reference.encoderMob, force),
        ])
        self.inter = GravityInter(reference.interViewEncoder, force)
        self.para1 = nn.Parameter(reference.para1.detach().clone())
        self.para2 = nn.Parameter(reference.para2.detach().clone())
        adaptive_dim = reference.viewFusionLayer.W.out_channels
        self.view_projection = nn.Conv1d(dim, adaptive_dim, 1, bias=False)
        self.view_left = nn.Conv1d(adaptive_dim, 1, 1)
        self.view_right = nn.Conv1d(adaptive_dim, 1, 1)
        self.view_projection.load_state_dict(reference.viewFusionLayer.W.state_dict())
        self.view_left.load_state_dict(reference.viewFusionLayer.f1.state_dict())
        self.view_right.load_state_dict(reference.viewFusionLayer.f2.state_dict())
        region_prior = equilibrium if region_gravity_kernel == "equilibrium" else force
        self.region_gravity_kernel = region_gravity_kernel
        self.region_blocks = nn.ModuleList([
            GravityRegionBlock(block, region_prior) for block in reference.regionFusionLayer.blocks
        ])
        self.region_output = Projection(dim, dim)
        self.region_output.network.load_state_dict(reference.regionFusionLayer.fc.network.state_dict())
        self.output = Projection(dim, output_dim)
        self.output.network.load_state_dict(reference.fc.network.state_dict())
        self.heads = nn.ModuleList([nn.Linear(output_dim, output_dim) for _ in range(4)])
        for target, source in zip(
            self.heads,
            [reference.decoder_s, reference.decoder_t, reference.decoder_p, reference.decoder_l],
            strict=True,
        ):
            target.load_state_dict(source.state_dict())
        self.dropout = nn.Dropout(0.1)
        self.feature: torch.Tensor | None = None

    def forward(self, features: list[torch.Tensor]):
        coupling = (torch.sigmoid(self.gravity_logit) * self.gravity_routing_alpha).clamp(0.0, 1.0)
        zero_coupling = torch.zeros_like(coupling)
        intra_coupling = coupling if "intra" in self.gravity_roles else zero_coupling
        inter_coupling = coupling if "inter" in self.gravity_roles else zero_coupling
        view_coupling = coupling if "view" in self.gravity_roles else zero_coupling
        region_coupling = coupling if "region" in self.gravity_roles else zero_coupling
        poi, land, mobility = features
        inputs = [
            self.dropout(F.relu(self.poi_projection(poi))),
            self.dropout(F.relu(self.land_projection(land))),
            mobility,
        ]
        intra = torch.stack([
            encoder(values, intra_coupling)
            for encoder, values in zip(self.intra, inputs, strict=True)
        ])
        inter = self.inter(intra.transpose(0, 1), inter_coupling).transpose(0, 1)
        p1 = self.para1 / (self.para1 + self.para2)
        states = inter * (1.0 - p1) + intra * p1
        projected = self.view_projection(states.transpose(0, 2))
        pair_logits = self.view_left(projected) + self.view_right(projected).transpose(1, 2)
        global_logits = F.leaky_relu(pair_logits, 0.3).mean(dim=-1).mean(dim=0)
        transported = torch.stack([self.force.to(state) @ state for state in states])
        local = -(states - transported).square().mean(dim=-1).transpose(0, 1)
        local = local - local.mean(dim=1, keepdim=True)
        weights = torch.softmax(global_logits.unsqueeze(0) + view_coupling * local, dim=1)
        fused = (weights.T.unsqueeze(-1) * states).sum(dim=0).unsqueeze(0)
        for block in self.region_blocks:
            fused = block(fused, region_coupling)
        feature = self.output(self.region_output(fused.squeeze(0)))
        self.feature = feature
        return tuple(head(feature) for head in self.heads)

    def out_feature(self) -> torch.Tensor:
        if self.feature is None:
            raise RuntimeError("out_feature called before forward")
        return self.feature
