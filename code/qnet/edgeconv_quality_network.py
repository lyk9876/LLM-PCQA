"""EdgeConv patch quality networks.

This module scores one normalized patch directly from its point coordinates:

    [N, 3] -> EdgeConv(3, 64) -> EdgeConv(64, 64) -> EdgeConv(64, 128)
           -> concat per-layer tokens [N, 256] -> 5 learned quality queries.
"""

from __future__ import annotations

import torch
import torch.nn as nn


GRAPH_MODES = {"fixed", "fixed_fixed_dynamic", "dynamic"}
ENCODER_TYPES = {"edgeconv", "multiscale_edgeconv"}
QUERY_HEAD_TYPES = {"independent", "shared", "shared_affine"}
QUALITY_OUTPUT_MODES = {"direct", "global_residual", "factor_queries_global"}
FACTOR_QUERY_NAMES = ("noise", "detail", "surface", "density")


def knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return k nearest-neighbor indices for each token in x.

    Args:
        x: Tensor with shape [B, N, C].
        k: Number of neighbors. The center point itself is excluded.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected [B, N, C], got {tuple(x.shape)}")
    num_points = x.shape[1]
    if num_points < 2:
        raise ValueError("EdgeConv requires at least two points")
    k = min(k, num_points - 1)
    distances = torch.cdist(x, x)
    return distances.topk(k=k + 1, dim=-1, largest=False).indices[:, :, 1:]


def edge_features(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Build EdgeConv features [x_i, x_j - x_i]."""
    batch_size, num_points, channels = x.shape
    _, _, k = idx.shape
    batch_offsets = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1) * num_points
    flat_idx = (idx + batch_offsets).reshape(-1)
    neighbors = x.reshape(batch_size * num_points, channels)[flat_idx]
    neighbors = neighbors.reshape(batch_size, num_points, k, channels)
    centers = x.unsqueeze(2).expand(-1, -1, k, -1)
    return torch.cat([centers, neighbors - centers], dim=-1)


class EdgeConvBlock(nn.Module):
    """One dynamic-graph EdgeConv block with max aggregation."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 16):
        super().__init__()
        self.k = int(k)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        if idx is None:
            idx = knn_indices(x, self.k)
        edges = edge_features(x, idx)
        edge_hidden = self.edge_mlp(edges)
        return edge_hidden.max(dim=2).values


class MultiScaleEdgeConvBlock(nn.Module):
    """Parallel EdgeConv branches over multiple neighborhood sizes."""

    def __init__(self, in_channels: int, out_channels: int, ks: tuple[int, ...]):
        super().__init__()
        if not ks:
            raise ValueError("MultiScaleEdgeConvBlock requires at least one k")
        self.ks = tuple(int(k) for k in ks)
        self.branches = nn.ModuleList([EdgeConvBlock(in_channels, out_channels, k=k) for k in self.ks])
        self.fuse = nn.Sequential(
            nn.Linear(out_channels * len(self.ks), out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, idxs: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
        if idxs is None:
            outputs = [branch(x) for branch in self.branches]
        else:
            if len(idxs) != len(self.branches):
                raise ValueError(f"Expected {len(self.branches)} neighbor index tensors, got {len(idxs)}")
            outputs = [branch(x, idx) for branch, idx in zip(self.branches, idxs)]
        return self.fuse(torch.cat(outputs, dim=-1))


class EdgeConvEncoder(nn.Module):
    """Three-layer EdgeConv encoder returning point tokens with 256 channels."""

    def __init__(self, k: int = 16, graph_mode: str = "dynamic"):
        super().__init__()
        if graph_mode not in GRAPH_MODES:
            raise ValueError(f"graph_mode must be one of {sorted(GRAPH_MODES)}, got {graph_mode!r}")
        self.k = int(k)
        self.graph_mode = graph_mode
        self.edge1 = EdgeConvBlock(3, 64, k=k)
        self.edge2 = EdgeConvBlock(64, 64, k=k)
        self.edge3 = EdgeConvBlock(64, 128, k=k)
        self.token_dim = 64 + 64 + 128

    def forward(self, points: torch.Tensor, return_hierarchy: bool = False):
        if points.shape[-1] != 3:
            raise ValueError(f"Expected last dimension 3, got {points.shape[-1]}")
        original_shape = points.shape[:-2]
        x = points.reshape(-1, points.shape[-2], 3)

        fixed_idx = knn_indices(x, self.k) if self.graph_mode in {"fixed", "fixed_fixed_dynamic"} else None
        h1 = self.edge1(x, fixed_idx)
        h2 = self.edge2(h1, fixed_idx if self.graph_mode in {"fixed", "fixed_fixed_dynamic"} else None)
        h3 = self.edge3(h2, fixed_idx if self.graph_mode == "fixed" else None)
        tokens = torch.cat([h1, h2, h3], dim=-1)
        tokens = tokens.reshape(*original_shape, points.shape[-2], self.token_dim)
        if not return_hierarchy:
            return tokens
        h1 = h1.reshape(*original_shape, points.shape[-2], h1.shape[-1])
        h2 = h2.reshape(*original_shape, points.shape[-2], h2.shape[-1])
        h3 = h3.reshape(*original_shape, points.shape[-2], h3.shape[-1])
        return tokens, (h1, h2, h3)


class MultiScaleEdgeConvEncoder(nn.Module):
    """Three-layer multi-scale EdgeConv encoder returning 256-D point tokens."""

    def __init__(self, ks: tuple[int, ...] = (8, 16, 32), graph_mode: str = "dynamic"):
        super().__init__()
        if graph_mode not in GRAPH_MODES:
            raise ValueError(f"graph_mode must be one of {sorted(GRAPH_MODES)}, got {graph_mode!r}")
        self.ks = tuple(int(k) for k in ks)
        self.graph_mode = graph_mode
        self.edge1 = MultiScaleEdgeConvBlock(3, 64, ks=self.ks)
        self.edge2 = MultiScaleEdgeConvBlock(64, 64, ks=self.ks)
        self.edge3 = MultiScaleEdgeConvBlock(64, 128, ks=self.ks)
        self.token_dim = 64 + 64 + 128

    def forward(self, points: torch.Tensor, return_hierarchy: bool = False):
        if points.shape[-1] != 3:
            raise ValueError(f"Expected last dimension 3, got {points.shape[-1]}")
        original_shape = points.shape[:-2]
        x = points.reshape(-1, points.shape[-2], 3)

        fixed_idxs = [knn_indices(x, k) for k in self.ks] if self.graph_mode in {"fixed", "fixed_fixed_dynamic"} else None
        h1 = self.edge1(x, fixed_idxs)
        h2 = self.edge2(h1, fixed_idxs if self.graph_mode in {"fixed", "fixed_fixed_dynamic"} else None)
        h3 = self.edge3(h2, fixed_idxs if self.graph_mode == "fixed" else None)
        tokens = torch.cat([h1, h2, h3], dim=-1)
        tokens = tokens.reshape(*original_shape, points.shape[-2], self.token_dim)
        if not return_hierarchy:
            return tokens
        h1 = h1.reshape(*original_shape, points.shape[-2], h1.shape[-1])
        h2 = h2.reshape(*original_shape, points.shape[-2], h2.shape[-1])
        h3 = h3.reshape(*original_shape, points.shape[-2], h3.shape[-1])
        return tokens, (h1, h2, h3)


def build_encoder(
    encoder_type: str = "edgeconv",
    k: int = 16,
    graph_mode: str = "dynamic",
    multi_scale_ks: tuple[int, ...] = (8, 16, 32),
) -> nn.Module:
    if encoder_type not in ENCODER_TYPES:
        raise ValueError(f"encoder_type must be one of {sorted(ENCODER_TYPES)}, got {encoder_type!r}")
    if encoder_type == "edgeconv":
        return EdgeConvEncoder(k=k, graph_mode=graph_mode)
    return MultiScaleEdgeConvEncoder(ks=multi_scale_ks, graph_mode=graph_mode)


class QualityQueryHead(nn.Module):
    """Five learned queries attending to point-level quality tokens."""

    def __init__(
        self,
        token_dim: int = 256,
        num_queries: int = 5,
        num_heads: int = 4,
        hidden_dim: int = 128,
        query_interaction: bool = True,
        head_type: str = "independent",
    ):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if head_type not in QUERY_HEAD_TYPES:
            raise ValueError(f"head_type must be one of {sorted(QUERY_HEAD_TYPES)}, got {head_type!r}")
        self.query_interaction = bool(query_interaction)
        self.head_type = head_type
        self.queries = nn.Parameter(torch.empty(num_queries, token_dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.token_norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(token_dim)
        if self.query_interaction:
            self.query_self_attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True)
            self.query_interaction_norm = nn.LayerNorm(token_dim)
        else:
            self.query_self_attn = None
            self.query_interaction_norm = None
        if self.head_type == "independent":
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(token_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(num_queries)
                ]
            )
            self.shared_head = None
            self.head_scale = None
            self.head_bias = None
        else:
            self.heads = None
            self.shared_head = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.head_type == "shared_affine":
                self.head_scale = nn.Parameter(torch.ones(num_queries))
                self.head_bias = nn.Parameter(torch.zeros(num_queries))
            else:
                self.head_scale = None
                self.head_bias = None
        self.last_attention_weights: torch.Tensor | None = None
        self.last_query_interaction_weights: torch.Tensor | None = None

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
        return_query_features: bool = False,
        return_query_stages: bool = False,
    ):
        batch_size = tokens.shape[0]
        tokens = self.token_norm(tokens)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, attention = self.attn(
            queries,
            tokens,
            tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        attended = self.query_norm(attended + queries)
        before_interaction = attended
        if self.query_self_attn is not None and self.query_interaction_norm is not None:
            refined, query_attention = self.query_self_attn(
                attended,
                attended,
                attended,
                need_weights=return_attention,
                average_attn_weights=False,
            )
            attended = self.query_interaction_norm(attended + refined)
        else:
            query_attention = None
        if self.heads is not None:
            raw = torch.stack(
                [head(attended[:, query_idx]).squeeze(-1) for query_idx, head in enumerate(self.heads)],
                dim=-1,
            )
        else:
            raw = self.shared_head(attended).squeeze(-1)
            if self.head_scale is not None and self.head_bias is not None:
                raw = raw * self.head_scale.unsqueeze(0) + self.head_bias.unsqueeze(0)
        self.last_attention_weights = attention.detach() if attention is not None else None
        self.last_query_interaction_weights = query_attention.detach() if query_attention is not None else None
        if return_attention and return_query_stages:
            return raw, attention, before_interaction, attended
        if return_attention and return_query_features:
            return raw, attention, attended
        if return_attention:
            return raw, attention
        if return_query_stages:
            return raw, before_interaction, attended
        if return_query_features:
            return raw, attended
        return raw


class FactorQueryGlobalHead(nn.Module):
    """Four point-attending factor queries plus an independent global Overall head.

    The factor branches expose both their 256-D attended representations and
    point-aligned attention maps. Overall is regressed directly from mean-pooled
    point tokens and is never computed from the four factor predictions.
    """

    def __init__(
        self,
        token_dim: int = 256,
        num_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.queries = nn.Parameter(torch.empty(len(FACTOR_QUERY_NAMES), token_dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.token_norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(token_dim)
        self.factor_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(token_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in FACTOR_QUERY_NAMES
            ]
        )
        self.global_norm = nn.LayerNorm(token_dim)
        self.overall_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B,N,C], got {tuple(tokens.shape)}")
        normalized_tokens = self.token_norm(tokens)
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        factor_features, attention_heads = self.attn(
            queries,
            normalized_tokens,
            normalized_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        factor_features = self.query_norm(factor_features + queries)
        factor_logits = torch.stack(
            [
                head(factor_features[:, query_index]).squeeze(-1)
                for query_index, head in enumerate(self.factor_heads)
            ],
            dim=-1,
        )
        global_feature = self.global_norm(tokens.mean(dim=1))
        overall_logit = self.overall_head(global_feature).squeeze(-1)
        logits = torch.cat([factor_logits, overall_logit.unsqueeze(-1)], dim=-1)
        return {
            "logits": logits,
            "factor_features": factor_features,
            "global_feature": global_feature,
            "attention_heads": attention_heads,
            "attention": attention_heads.mean(dim=1),
        }


class GlobalResidualQualityHead(nn.Module):
    """One global quality query plus five dimension-residual queries."""

    def __init__(
        self,
        token_dim: int = 256,
        num_residual_queries: int = 5,
        num_heads: int = 4,
        hidden_dim: int = 128,
        query_interaction: bool = True,
        residual_mean: torch.Tensor | None = None,
        residual_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.query_interaction = bool(query_interaction)
        self.num_residual_queries = int(num_residual_queries)
        self.queries = nn.Parameter(torch.empty(self.num_residual_queries + 1, token_dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.token_norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(token_dim)
        if self.query_interaction:
            self.query_self_attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True)
            self.query_interaction_norm = nn.LayerNorm(token_dim)
        else:
            self.query_self_attn = None
            self.query_interaction_norm = None
        self.global_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(token_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.num_residual_queries)
            ]
        )
        if residual_mean is None:
            residual_mean = torch.zeros(self.num_residual_queries, dtype=torch.float32)
        if residual_std is None:
            residual_std = torch.ones(self.num_residual_queries, dtype=torch.float32)
        self.register_buffer("residual_mean", residual_mean.detach().clone().float())
        self.register_buffer("residual_std", residual_std.detach().clone().float().clamp_min(1e-6))
        self.last_attention_weights: torch.Tensor | None = None
        self.last_query_interaction_weights: torch.Tensor | None = None

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
        return_query_features: bool = False,
        return_query_stages: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        batch_size = tokens.shape[0]
        tokens = self.token_norm(tokens)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, attention = self.attn(
            queries,
            tokens,
            tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        attended = self.query_norm(attended + queries)
        before_interaction = attended
        if self.query_self_attn is not None and self.query_interaction_norm is not None:
            refined, query_attention = self.query_self_attn(
                attended,
                attended,
                attended,
                need_weights=return_attention,
                average_attn_weights=False,
            )
            attended = self.query_interaction_norm(attended + refined)
        else:
            query_attention = None

        global_feature = attended[:, 0]
        residual_features = attended[:, 1:]
        global_raw = self.global_head(global_feature).squeeze(-1)
        global_pred = torch.sigmoid(global_raw)
        residual_z = torch.stack(
            [
                head(residual_features[:, query_idx]).squeeze(-1)
                for query_idx, head in enumerate(self.residual_heads)
            ],
            dim=-1,
        )
        residual_raw = residual_z * self.residual_std.unsqueeze(0) + self.residual_mean.unsqueeze(0)
        residual_pred = residual_raw - residual_raw.mean(dim=-1, keepdim=True)
        pred_5d = (global_pred.unsqueeze(-1) + residual_pred).clamp(0.0, 1.0)

        self.last_attention_weights = attention.detach() if attention is not None else None
        self.last_query_interaction_weights = query_attention.detach() if query_attention is not None else None
        output: dict[str, torch.Tensor | None] = {
            "pred_5d": pred_5d,
            "global_raw": global_raw,
            "global_pred": global_pred,
            "residual_z": residual_z,
            "residual_pred": residual_pred,
            "global_feature": global_feature,
            "query_features": residual_features if return_query_features else None,
            "all_query_features": attended if return_query_features or return_query_stages else None,
            "attention": attention if return_attention else None,
            "query_attention": query_attention if return_attention else None,
            "before_interaction": before_interaction[:, 1:] if return_query_stages else None,
            "after_interaction": residual_features if return_query_stages else None,
            "all_before_interaction": before_interaction if return_query_stages else None,
            "all_after_interaction": attended if return_query_stages else None,
        }
        return output


class EdgeConvQualityQueryNet5D(nn.Module):
    """Predict five raw quality scores from one 500x3 candidate patch."""

    def __init__(
        self,
        k: int = 16,
        output_dim: int = 5,
        query_heads: int = 4,
        graph_mode: str = "dynamic",
        query_interaction: bool = True,
        encoder_type: str = "edgeconv",
        multi_scale_ks: tuple[int, ...] = (8, 16, 32),
        query_head_type: str = "independent",
        quality_mode: str = "direct",
        residual_mean: torch.Tensor | None = None,
        residual_std: torch.Tensor | None = None,
        factor_query_dropout: float = 0.1,
        structured_output: bool = False,
    ):
        super().__init__()
        if quality_mode not in QUALITY_OUTPUT_MODES:
            raise ValueError(f"quality_mode must be one of {sorted(QUALITY_OUTPUT_MODES)}, got {quality_mode!r}")
        self.quality_mode = quality_mode
        self.structured_output = bool(structured_output)
        self.encoder = build_encoder(
            encoder_type=encoder_type,
            k=k,
            graph_mode=graph_mode,
            multi_scale_ks=multi_scale_ks,
        )
        if self.quality_mode == "direct":
            self.query_head = QualityQueryHead(
                token_dim=self.encoder.token_dim,
                num_queries=output_dim,
                num_heads=query_heads,
                query_interaction=query_interaction,
                head_type=query_head_type,
            )
        elif self.quality_mode == "global_residual":
            self.query_head = GlobalResidualQualityHead(
                token_dim=self.encoder.token_dim,
                num_residual_queries=output_dim,
                num_heads=query_heads,
                query_interaction=query_interaction,
                residual_mean=residual_mean,
                residual_std=residual_std,
            )
        else:
            if output_dim != 5:
                raise ValueError("factor_queries_global requires output_dim=5")
            self.query_head = FactorQueryGlobalHead(
                token_dim=self.encoder.token_dim,
                num_heads=query_heads,
                hidden_dim=128,
                dropout=factor_query_dropout,
            )
        self.last_attention_weights: torch.Tensor | None = None
        self.last_query_interaction_weights: torch.Tensor | None = None

    def forward(
        self,
        points: torch.Tensor,
        return_attention: bool = False,
        return_tokens: bool = False,
        return_query_features: bool = False,
        return_query_stages: bool = False,
        return_quality_components: bool = False,
        return_structured: bool = False,
    ):
        """Score point patches.

        Args:
            points: Tensor with shape [..., N, 3].

        Returns:
            Raw quality logits with shape [..., 5].
        """
        original_shape = points.shape[:-2]
        tokens = self.encoder(points)
        flat_tokens = tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])
        if self.quality_mode == "factor_queries_global":
            if return_query_stages:
                raise ValueError("factor_queries_global has no inter-query refinement stages")
            output = self.query_head(flat_tokens)
            raw = output["logits"].reshape(*original_shape, 5)
            factor_features = output["factor_features"].reshape(*original_shape, 4, self.encoder.token_dim)
            global_feature = output["global_feature"].reshape(*original_shape, self.encoder.token_dim)
            attention_heads = output["attention_heads"].reshape(
                *original_shape,
                output["attention_heads"].shape[-3],
                4,
                points.shape[-2],
            )
            attention = output["attention"].reshape(*original_shape, 4, points.shape[-2])
            self.last_attention_weights = attention_heads.detach()
            self.last_query_interaction_weights = None

            if self.structured_output or return_structured or return_quality_components:
                scores_tensor = torch.sigmoid(raw)
                structured: dict[str, object] = {
                    "scores": {
                        name: scores_tensor[..., index]
                        for index, name in enumerate((*FACTOR_QUERY_NAMES, "overall"))
                    },
                    "scores_tensor": scores_tensor,
                    "logits": {
                        name: raw[..., index]
                        for index, name in enumerate((*FACTOR_QUERY_NAMES, "overall"))
                    },
                    "logits_tensor": raw,
                    "features": {
                        **{
                            name: factor_features[..., index, :]
                            for index, name in enumerate(FACTOR_QUERY_NAMES)
                        },
                        "global": global_feature,
                    },
                    "query_features": factor_features,
                    "attentions": {
                        name: attention[..., index, :]
                        for index, name in enumerate(FACTOR_QUERY_NAMES)
                    },
                    "attention_heads": attention_heads,
                }
                if return_tokens:
                    structured["point_features"] = tokens
                return structured
            result: tuple[torch.Tensor, ...] = (raw,)
            if return_attention:
                result += (attention_heads,)
            if return_query_features:
                result += (factor_features,)
            if return_tokens:
                result += (tokens,)
            return result[0] if len(result) == 1 else result
        if self.quality_mode == "global_residual":
            output = self.query_head(
                flat_tokens,
                return_attention=return_attention,
                return_query_features=return_query_features,
                return_query_stages=return_query_stages,
            )
            reshaped: dict[str, torch.Tensor | None] = {}
            for key, value in output.items():
                if value is None:
                    reshaped[key] = None
                elif key in {"pred_5d", "residual_z", "residual_pred"}:
                    reshaped[key] = value.reshape(*original_shape, value.shape[-1])
                elif key in {"global_raw", "global_pred"}:
                    reshaped[key] = value.reshape(*original_shape)
                elif key in {"global_feature"}:
                    reshaped[key] = value.reshape(*original_shape, value.shape[-1])
                elif key in {
                    "query_features",
                    "all_query_features",
                    "before_interaction",
                    "after_interaction",
                    "all_before_interaction",
                    "all_after_interaction",
                }:
                    reshaped[key] = value.reshape(*original_shape, value.shape[-2], value.shape[-1])
                elif key in {"attention"}:
                    reshaped[key] = value.reshape(*original_shape, value.shape[-3], value.shape[-2], value.shape[-1])
                elif key in {"query_attention"}:
                    reshaped[key] = value.reshape(*original_shape, value.shape[-3], value.shape[-2], value.shape[-1])
                else:
                    reshaped[key] = value
            self.last_attention_weights = reshaped["attention"].detach() if reshaped.get("attention") is not None else None
            self.last_query_interaction_weights = (
                reshaped["query_attention"].detach() if reshaped.get("query_attention") is not None else None
            )
            if return_quality_components:
                if return_tokens:
                    reshaped["tokens"] = tokens
                return reshaped
            if return_attention and return_query_stages:
                result = (
                    reshaped["pred_5d"],
                    reshaped["attention"],
                    reshaped["before_interaction"],
                    reshaped["after_interaction"],
                )
                return (*result, tokens) if return_tokens else result
            if return_attention and return_query_features:
                result = (reshaped["pred_5d"], reshaped["attention"], reshaped["query_features"])
                return (*result, tokens) if return_tokens else result
            if return_attention:
                result = (reshaped["pred_5d"], reshaped["attention"])
                return (*result, tokens) if return_tokens else result
            if return_query_stages:
                result = (reshaped["pred_5d"], reshaped["before_interaction"], reshaped["after_interaction"])
                return (*result, tokens) if return_tokens else result
            if return_query_features:
                result = (reshaped["pred_5d"], reshaped["query_features"])
                return (*result, tokens) if return_tokens else result
            if return_tokens:
                return reshaped["pred_5d"], tokens
            return reshaped["pred_5d"]
        if return_attention:
            query_out = self.query_head(
                flat_tokens,
                return_attention=True,
                return_query_features=return_query_features,
                return_query_stages=return_query_stages,
            )
            if return_query_stages:
                raw_flat, attention, before_flat, after_flat = query_out
            elif return_query_features:
                raw_flat, attention, query_features_flat = query_out
            else:
                raw_flat, attention = query_out
            raw = raw_flat.reshape(*original_shape, raw_flat.shape[-1])
            attention = attention.reshape(*original_shape, attention.shape[-3], attention.shape[-2], attention.shape[-1])
            self.last_attention_weights = attention.detach()
            query_attention = self.query_head.last_query_interaction_weights
            if query_attention is not None:
                query_attention = query_attention.reshape(
                    *original_shape,
                    query_attention.shape[-3],
                    query_attention.shape[-2],
                    query_attention.shape[-1],
                )
            self.last_query_interaction_weights = query_attention.detach() if query_attention is not None else None
            if return_query_stages:
                before = before_flat.reshape(*original_shape, before_flat.shape[-2], before_flat.shape[-1])
                after = after_flat.reshape(*original_shape, after_flat.shape[-2], after_flat.shape[-1])
                if return_tokens:
                    return raw, attention, before, after, tokens
                return raw, attention, before, after
            if return_query_features:
                query_features = query_features_flat.reshape(
                    *original_shape,
                    query_features_flat.shape[-2],
                    query_features_flat.shape[-1],
                )
                if return_tokens:
                    return raw, attention, query_features, tokens
                return raw, attention, query_features
            if return_tokens:
                return raw, attention, tokens
            return raw, attention
        if return_query_stages:
            raw_flat, before_flat, after_flat = self.query_head(flat_tokens, return_query_stages=True)
            raw = raw_flat.reshape(*original_shape, raw_flat.shape[-1])
            self.last_attention_weights = None
            self.last_query_interaction_weights = None
            before = before_flat.reshape(*original_shape, before_flat.shape[-2], before_flat.shape[-1])
            after = after_flat.reshape(*original_shape, after_flat.shape[-2], after_flat.shape[-1])
            if return_tokens:
                return raw, before, after, tokens
            return raw, before, after
        if return_query_features:
            raw_flat, query_features_flat = self.query_head(flat_tokens, return_query_features=True)
        else:
            raw_flat = self.query_head(flat_tokens)
        raw = raw_flat.reshape(*original_shape, raw_flat.shape[-1])
        self.last_attention_weights = None
        self.last_query_interaction_weights = None
        if return_query_features:
            query_features = query_features_flat.reshape(
                *original_shape,
                query_features_flat.shape[-2],
                query_features_flat.shape[-1],
            )
            if return_tokens:
                return raw, query_features, tokens
            return raw, query_features
        if return_tokens:
            return raw, tokens
        return raw


class PrivilegedQualityTeacher5D(nn.Module):
    """Training-only teacher that compares candidate tokens with noisy/clean references."""

    def __init__(
        self,
        k: int = 16,
        output_dim: int = 5,
        query_heads: int = 4,
        graph_mode: str = "dynamic",
        query_interaction: bool = True,
        encoder_type: str = "edgeconv",
        multi_scale_ks: tuple[int, ...] = (8, 16, 32),
    ):
        super().__init__()
        self.encoder = build_encoder(
            encoder_type=encoder_type,
            k=k,
            graph_mode=graph_mode,
            multi_scale_ks=multi_scale_ks,
        )
        token_dim = self.encoder.token_dim
        self.reference_attn = nn.MultiheadAttention(token_dim, num_heads=query_heads, batch_first=True)
        self.reference_norm = nn.LayerNorm(token_dim)
        self.query_head = QualityQueryHead(
            token_dim=token_dim,
            num_queries=output_dim,
            num_heads=query_heads,
            query_interaction=query_interaction,
        )

    def forward(self, candidate: torch.Tensor, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        original_shape = candidate.shape[:-2]
        candidate_tokens = self.encoder(candidate)
        noisy_tokens = self.encoder(noisy)
        clean_tokens = self.encoder(clean)

        flat_candidate = candidate_tokens.reshape(-1, candidate_tokens.shape[-2], candidate_tokens.shape[-1])
        flat_noisy = noisy_tokens.reshape(-1, noisy_tokens.shape[-2], noisy_tokens.shape[-1])
        flat_clean = clean_tokens.reshape(-1, clean_tokens.shape[-2], clean_tokens.shape[-1])
        reference_tokens = torch.cat([flat_noisy, flat_clean], dim=1)
        reference_context, _ = self.reference_attn(
            flat_candidate,
            reference_tokens,
            reference_tokens,
            need_weights=False,
        )
        teacher_tokens = self.reference_norm(flat_candidate + reference_context)
        raw = self.query_head(teacher_tokens)
        return raw.reshape(*original_shape, raw.shape[-1])


class EdgeConvObjectiveRanker(nn.Module):
    """Stage-1 objective ranking model: EdgeConv encoder plus scalar ranking head."""

    def __init__(
        self,
        k: int = 16,
        graph_mode: str = "dynamic",
        hidden_dim: int = 256,
        encoder_type: str = "edgeconv",
        multi_scale_ks: tuple[int, ...] = (8, 16, 32),
    ):
        super().__init__()
        self.encoder = build_encoder(
            encoder_type=encoder_type,
            k=k,
            graph_mode=graph_mode,
            multi_scale_ks=multi_scale_ks,
        )
        pooled_dim = self.encoder.token_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        original_shape = points.shape[:-2]
        tokens = self.encoder(points)
        flat_tokens = tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])
        pooled = torch.cat([flat_tokens.max(dim=1).values, flat_tokens.mean(dim=1)], dim=-1)
        raw = self.head(pooled).squeeze(-1)
        return raw.reshape(*original_shape)
