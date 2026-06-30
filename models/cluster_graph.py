import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(num_channels, max_groups=32):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class GraphConvolution(nn.Module):
    """Dense GCN layer using a fixed, normalized adjacency matrix."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, nodes, normalized_adjacency):
        support = self.proj(nodes)
        adjacency = normalized_adjacency.to(dtype=support.dtype)
        return self.norm(torch.bmm(adjacency, support))


class HardClusterGraphBlock(nn.Module):
    """
    Attention-free spatial region graph refinement.

    Spatial features are hard-clustered with feature/coordinate descriptors.
    Touching clusters form an undirected graph, which is processed by two
    ordinary GCN layers. Refined region nodes are then broadcast back to their
    assigned pixels and fused through a bounded residual branch.

    Cluster assignments and adjacency construction are intentionally
    non-differentiable routing operations. The feature projection, node
    aggregation, graph convolutions, inverse projection, and residual scale
    remain fully trainable.
    """

    def __init__(
        self,
        in_channels,
        num_clusters,
        graph_dim=64,
        num_cluster_iters=2,
        spatial_weight=0.5,
        distance_chunk_size=8192,
        dropout=0.0,
        max_gamma=0.1,
        init_gamma=1e-3,
    ):
        super().__init__()

        if num_clusters <= 0:
            raise ValueError(f"num_clusters must be positive, got {num_clusters}")
        if graph_dim <= 0:
            raise ValueError(f"graph_dim must be positive, got {graph_dim}")
        if num_cluster_iters <= 0:
            raise ValueError(
                f"num_cluster_iters must be positive, got {num_cluster_iters}"
            )
        if spatial_weight < 0:
            raise ValueError(
                f"spatial_weight must be non-negative, got {spatial_weight}"
            )
        if max_gamma <= 0:
            raise ValueError(f"max_gamma must be positive, got {max_gamma}")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.graph_dim = int(graph_dim)
        self.num_cluster_iters = int(num_cluster_iters)
        self.spatial_weight = float(spatial_weight)
        self.distance_chunk_size = int(distance_chunk_size)
        self.max_gamma = float(max_gamma)

        self.in_proj = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.graph_dim,
                kernel_size=1,
                bias=False,
            ),
            _make_group_norm(self.graph_dim),
            nn.GELU(),
        )

        self.gcn1 = GraphConvolution(self.graph_dim, self.graph_dim)
        self.gcn2 = GraphConvolution(self.graph_dim, self.graph_dim)
        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Sequential(
            nn.Conv2d(
                self.graph_dim,
                self.in_channels,
                kernel_size=1,
                bias=False,
            ),
            _make_group_norm(self.in_channels),
        )

        init_ratio = init_gamma / self.max_gamma
        init_ratio = max(min(init_ratio, 0.999), -0.999)
        init_raw = torch.atanh(torch.tensor(init_ratio, dtype=torch.float32))
        self.gamma_raw = nn.Parameter(init_raw.view(1))

    @staticmethod
    def _grid_shape(num_clusters, height, width):
        """Choose a near-aspect-ratio grid containing at least K seeds."""
        target_h = int(
            round(math.sqrt(num_clusters * height / max(float(width), 1.0)))
        )
        grid_h = max(1, min(height, target_h))

        if math.ceil(num_clusters / grid_h) > width:
            grid_h = min(height, math.ceil(num_clusters / width))

        grid_w = max(1, min(width, math.ceil(num_clusters / grid_h)))

        while grid_h * grid_w < num_clusters:
            if grid_w < width:
                grid_w += 1
            elif grid_h < height:
                grid_h += 1
            else:
                break

        return grid_h, grid_w

    @staticmethod
    def _coordinate_grid(batch, height, width, device):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([yy, xx], dim=0).unsqueeze(0)
        return coords.expand(batch, -1, -1, -1)

    def _initial_centroids(self, descriptors, num_clusters):
        batch, channels, height, width = descriptors.shape
        grid_h, grid_w = self._grid_shape(num_clusters, height, width)
        seeds = F.adaptive_avg_pool2d(descriptors, (grid_h, grid_w))
        seeds = seeds.flatten(2).transpose(1, 2)

        if seeds.shape[1] == num_clusters:
            return seeds

        indices = torch.linspace(
            0,
            seeds.shape[1] - 1,
            num_clusters,
            device=seeds.device,
            dtype=torch.float32,
        ).round().long()
        return seeds.index_select(1, indices)

    def _assign_clusters(self, pixels, centroids):
        centroid_t = centroids.transpose(1, 2)
        centroid_norm = centroids.square().sum(dim=-1).unsqueeze(1)
        labels = []

        for start in range(0, pixels.shape[1], self.distance_chunk_size):
            chunk = pixels[:, start:start + self.distance_chunk_size]
            chunk_norm = chunk.square().sum(dim=-1, keepdim=True)
            distances = (
                chunk_norm
                + centroid_norm
                - 2.0 * torch.bmm(chunk, centroid_t)
            )
            labels.append(distances.argmin(dim=-1))

        return torch.cat(labels, dim=1)

    @staticmethod
    def _update_centroids(pixels, labels, centroids):
        batch, _, channels = pixels.shape
        num_clusters = centroids.shape[1]
        index = labels.unsqueeze(-1).expand(-1, -1, channels)

        sums = pixels.new_zeros(batch, num_clusters, channels)
        sums.scatter_add_(1, index, pixels)

        counts = pixels.new_zeros(batch, num_clusters, 1)
        counts.scatter_add_(
            1,
            labels.unsqueeze(-1),
            pixels.new_ones(batch, pixels.shape[1], 1),
        )

        updated = sums / counts.clamp_min(1.0)
        return torch.where(counts > 0, updated, centroids)

    def _cluster(self, projected):
        batch, _, height, width = projected.shape
        num_clusters = min(self.num_clusters, height * width)

        with torch.no_grad():
            normalized = F.normalize(projected.float(), dim=1, eps=1e-6)
            coords = self._coordinate_grid(
                batch,
                height,
                width,
                projected.device,
            )
            descriptors = torch.cat(
                [normalized, coords * self.spatial_weight],
                dim=1,
            )
            pixels = descriptors.flatten(2).transpose(1, 2).contiguous()
            centroids = self._initial_centroids(descriptors, num_clusters)

            labels = None
            for iteration in range(self.num_cluster_iters):
                labels = self._assign_clusters(pixels, centroids)
                if iteration + 1 < self.num_cluster_iters:
                    centroids = self._update_centroids(
                        pixels,
                        labels,
                        centroids,
                    )

        return labels, num_clusters

    @staticmethod
    def _pool_nodes(projected, labels, num_clusters):
        pixels = projected.flatten(2).transpose(1, 2).contiguous()
        batch, _, channels = pixels.shape
        index = labels.unsqueeze(-1).expand(-1, -1, channels)

        sums = pixels.new_zeros(batch, num_clusters, channels)
        sums.scatter_add_(1, index, pixels)

        counts = pixels.new_zeros(batch, num_clusters, 1)
        counts.scatter_add_(
            1,
            labels.unsqueeze(-1),
            pixels.new_ones(batch, pixels.shape[1], 1),
        )
        return sums / counts.clamp_min(1.0)

    @staticmethod
    def _build_normalized_adjacency(
        labels,
        height,
        width,
        num_clusters,
        dtype,
    ):
        with torch.no_grad():
            label_map = labels.view(labels.shape[0], height, width)

            horizontal_a = label_map[:, :, :-1].reshape(labels.shape[0], -1)
            horizontal_b = label_map[:, :, 1:].reshape(labels.shape[0], -1)
            vertical_a = label_map[:, :-1, :].reshape(labels.shape[0], -1)
            vertical_b = label_map[:, 1:, :].reshape(labels.shape[0], -1)

            sources = torch.cat(
                [horizontal_a, horizontal_b, vertical_a, vertical_b],
                dim=1,
            )
            targets = torch.cat(
                [horizontal_b, horizontal_a, vertical_b, vertical_a],
                dim=1,
            )
            edge_indices = sources * num_clusters + targets

            adjacency_flat = torch.zeros(
                labels.shape[0],
                num_clusters * num_clusters,
                device=labels.device,
                dtype=torch.float32,
            )
            adjacency_flat.scatter_add_(
                1,
                edge_indices,
                torch.ones_like(edge_indices, dtype=torch.float32),
            )
            adjacency = adjacency_flat.view(
                labels.shape[0],
                num_clusters,
                num_clusters,
            )
            adjacency = (adjacency > 0).float()

            eye = torch.eye(
                num_clusters,
                device=labels.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            adjacency = torch.maximum(adjacency, eye)

            degree_inv_sqrt = adjacency.sum(dim=-1).clamp_min(1.0).rsqrt()
            normalized = (
                degree_inv_sqrt.unsqueeze(-1)
                * adjacency
                * degree_inv_sqrt.unsqueeze(1)
            )

        return normalized.to(dtype=dtype)

    @staticmethod
    def _broadcast_nodes(nodes, labels, height, width):
        channels = nodes.shape[-1]
        index = labels.unsqueeze(-1).expand(-1, -1, channels)
        pixels = torch.gather(nodes, dim=1, index=index)
        return pixels.transpose(1, 2).reshape(
            nodes.shape[0],
            channels,
            height,
            width,
        )

    def forward(self, feature):
        _, _, height, width = feature.shape
        projected = self.in_proj(feature)
        labels, num_clusters = self._cluster(projected)

        nodes = self._pool_nodes(projected, labels, num_clusters)
        adjacency = self._build_normalized_adjacency(
            labels,
            height,
            width,
            num_clusters,
            dtype=nodes.dtype,
        )

        graph_delta = F.gelu(self.gcn1(nodes, adjacency))
        graph_delta = self.gcn2(graph_delta, adjacency)
        refined_nodes = nodes + self.dropout(graph_delta)

        graph_feature = self._broadcast_nodes(
            refined_nodes,
            labels,
            height,
            width,
        )
        graph_feature = self.out_proj(graph_feature)

        gamma = torch.tanh(self.gamma_raw) * self.max_gamma
        return feature + gamma * graph_feature


class MultiScaleClusterGraph(nn.Module):
    """Independent hard-cluster GCN refinement for four encoder scales."""

    def __init__(
        self,
        channels=(64, 128, 256, 512),
        cluster_counts=(256, 128, 64, 32),
        graph_dim=64,
        num_cluster_iters=2,
        spatial_weight=0.5,
        distance_chunk_size=8192,
        dropout=0.0,
    ):
        super().__init__()

        if len(channels) != 4 or len(cluster_counts) != 4:
            raise ValueError(
                "MultiScaleClusterGraph expects four channels and four "
                f"cluster counts, got {channels} and {cluster_counts}."
            )

        self.channels = tuple(int(value) for value in channels)
        self.cluster_counts = tuple(int(value) for value in cluster_counts)
        self.blocks = nn.ModuleList([
            HardClusterGraphBlock(
                in_channels=in_channels,
                num_clusters=num_clusters,
                graph_dim=graph_dim,
                num_cluster_iters=num_cluster_iters,
                spatial_weight=spatial_weight,
                distance_chunk_size=distance_chunk_size,
                dropout=dropout,
            )
            for in_channels, num_clusters in zip(
                self.channels,
                self.cluster_counts,
            )
        ])

    def forward(self, features):
        if len(features) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} features, got {len(features)}."
            )
        return [block(feature) for block, feature in zip(self.blocks, features)]
