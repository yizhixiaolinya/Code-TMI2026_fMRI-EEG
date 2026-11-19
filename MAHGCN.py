from pathlib import Path

import scipy.io as scio
import torch
import torch.nn as nn


def sparsify_fc(fc, keep_fraction=0.2):
    """Keep the strongest absolute off-diagonal undirected edges, with signs."""
    symmetric = (fc + fc.transpose(-1, -2)) * 0.5
    rows, columns = torch.triu_indices(fc.shape[-1], fc.shape[-1], offset=1, device=fc.device)
    values = symmetric[..., rows, columns]
    indices = values.abs().topk(int(values.shape[-1] * keep_fraction), dim=-1).indices
    retained = torch.zeros_like(values).scatter(-1, indices, values.gather(-1, indices))
    adjacency = torch.zeros_like(fc)
    adjacency[..., rows, columns] = retained
    return adjacency + adjacency.transpose(-1, -2)


def normalize_adjacency(adjacency):
    """Symmetric normalization using absolute degrees; no self-loops are added."""
    degree = adjacency.abs().sum(dim=-1)
    inverse_sqrt = degree.masked_fill(degree == 0, 1).rsqrt()
    return inverse_sqrt.unsqueeze(-1) * adjacency * inverse_sqrt.unsqueeze(-2)


class HGAA(nn.Module):
    """Finest-to-coarsest hierarchy with persistent [batch, channel, ROI, feature] states."""

    def __init__(self, act, drop_p, layer, ROI_num=100, step=100,
                 use_self_attn=True, hidden_dim=64, num_heads=4):
        super().__init__()
        self.layer = layer
        self.ROI_num = ROI_num
        self.hidden_dim = hidden_dim
        self.use_self_attn = use_self_attn
        self.sizes_asc = [ROI_num + i * step for i in range(layer)]
        self.sizes_desc = list(reversed(self.sizes_asc))
        self.net_gcn = nn.ModuleList([
            GCN(self.sizes_desc[0] + 2 if index == 0 else hidden_dim, hidden_dim, act, drop_p, bias=False)
            for index in range(layer)
        ])
        self.self_attentions = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=drop_p, batch_first=True)
            for _ in range(layer)
        ] if use_self_attn else [])
        self.attn_ln = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(layer)
        ] if use_self_attn else [])
        self.net_pool = nn.ModuleList([
            Fine2Coarse(fine, coarse, hidden_dim=hidden_dim)
            for fine, coarse in zip(self.sizes_desc[:-1], self.sizes_desc[1:])
        ])

    def prepare_graphs(self, graphs):
        """Unpack [FC row, ROI time-series mean, std] into adjacency and features."""
        graphs = sorted(graphs, key=lambda graph: graph.shape[-2], reverse=True)
        expected_shapes = [(size, size + 2) for size in self.sizes_desc]
        if [tuple(graph.shape[-2:]) for graph in graphs] != expected_shapes:
            raise ValueError(f"Expected packed [FC, mean, std] shapes {expected_shapes}")
        adjacency = [normalize_adjacency(sparsify_fc(graph[..., :size]))
                     for graph, size in zip(graphs, self.sizes_desc)]
        return adjacency, graphs[0]

    def forward(self, graphs):
        adjacency, features = self.prepare_graphs(graphs)
        graph_layers, node_layers = [], []
        for index, gcn in enumerate(self.net_gcn):
            features = gcn(adjacency[index], features)
            if self.use_self_attn:
                channel_mean = features.mean(dim=1)
                context, _ = self.self_attentions[index](
                    channel_mean, channel_mean, channel_mean, need_weights=False)
                features = self.attn_ln[index](features + context.unsqueeze(1))
            graph_layers.append(features.mean(dim=1).mean(dim=1))
            node_layers.append(features)
            if index < self.layer - 1:
                features = self.net_pool[index](features)

        # Public scale lists and concatenation follow the paper's coarse-to-fine order.
        graph_layers.reverse()
        node_layers.reverse()
        return torch.cat(graph_layers, dim=1), graph_layers, node_layers


class Fine2Coarse(nn.Module):
    """Subject-adaptive pooling restricted to positive atlas voxel overlap."""

    def __init__(self, indim, outdim, hidden_dim=64):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        filename = Path(__file__).resolve().parent / "mapping" / f"mapping_{indim}to{outdim}.mat"
        mapping = torch.as_tensor(scio.loadmat(filename)["mapping"], dtype=torch.float32)
        self.register_buffer("mapping", mapping)
        self.register_buffer("mask", mapping.T > 0)
        if not self.mask.any(dim=-1).all():
            raise ValueError(f"Every coarse ROI must have a fine contributor in {filename}")
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def attention_weights(self, features):
        scores = self.score(features.mean(dim=1)).squeeze(-1)
        scores = scores.unsqueeze(1).masked_fill(~self.mask.unsqueeze(0), float("-inf"))
        return scores.softmax(dim=-1)

    def forward(self, features):
        weights = self.attention_weights(features)
        pooled = torch.einsum("brv,bcvd->bcrd", weights, features)
        return self.norm(pooled)


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act, p=0.3, bias=True):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        # The same projection is applied to every channel within a modality.
        return self.act(self.proj(torch.matmul(g, self.drop(h))))
