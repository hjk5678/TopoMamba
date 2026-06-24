import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphInteractionAttention(nn.Module):
    """
    Lightweight graph interaction attention over multi-scale skip features.

    Each feature scale is treated as one graph node. The module exchanges
    information between nodes, then writes the updated node state back as a
    channel gate for the corresponding feature map.
    """

    def __init__(
        self,
        channels,
        graph_dim=64,
        num_heads=4,
        dropout=0.0,
        max_gamma=0.1,
        init_gamma=1e-3,
    ):
        super().__init__()

        if graph_dim % num_heads != 0:
            raise ValueError(
                f"graph_dim must be divisible by num_heads, got "
                f"graph_dim={graph_dim}, num_heads={num_heads}"
            )

        self.channels = list(channels)
        self.max_gamma = float(max_gamma)

        self.node_proj = nn.ModuleList([
            nn.Linear(ch, graph_dim)
            for ch in self.channels
        ])

        self.attn = nn.MultiheadAttention(
            embed_dim=graph_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(graph_dim)
        self.norm2 = nn.LayerNorm(graph_dim)

        self.ffn = nn.Sequential(
            nn.Linear(graph_dim, graph_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim * 2, graph_dim),
        )

        self.gate_proj = nn.ModuleList([
            nn.Linear(graph_dim, ch)
            for ch in self.channels
        ])

        init_ratio = init_gamma / self.max_gamma
        init_ratio = max(min(init_ratio, 0.999), -0.999)
        init_raw = torch.atanh(torch.tensor(init_ratio, dtype=torch.float32))
        self.gamma_raw = nn.Parameter(
            init_raw.repeat(len(self.channels))
        )

    def forward(self, features):
        if len(features) != len(self.channels):
            raise ValueError(
                f"Expected {len(self.channels)} features, got {len(features)}"
            )

        nodes = []
        for feat, proj in zip(features, self.node_proj):
            pooled = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
            nodes.append(proj(pooled))

        nodes = torch.stack(nodes, dim=1)

        attn_out, _ = self.attn(nodes, nodes, nodes, need_weights=False)
        nodes = self.norm1(nodes + attn_out)
        nodes = self.norm2(nodes + self.ffn(nodes))

        gamma = torch.tanh(self.gamma_raw) * self.max_gamma

        refined = []
        for i, (feat, gate_proj) in enumerate(zip(features, self.gate_proj)):
            gate = torch.sigmoid(gate_proj(nodes[:, i]))
            gate = gate.view(gate.shape[0], gate.shape[1], 1, 1)
            refined.append(feat + gamma[i] * feat * gate)

        return refined
