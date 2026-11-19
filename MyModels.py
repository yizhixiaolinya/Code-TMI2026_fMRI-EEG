"""Single-modality HGAA branches and an interleaved paired BCMF hierarchy."""
import torch
import torch.nn as nn

from MAHGCN import HGAA
from Dual_cross import CrossInteraction


BANDS = ('FC_delta', 'FC_theta', 'FC_alpha', 'FC_beta', 'FC_gamma')


def _classifier(feature_dim, num_class):
    return nn.Sequential(
        nn.LayerNorm(feature_dim), nn.ReLU(),
        nn.Linear(feature_dim, 64), nn.LayerNorm(64), nn.ReLU(),
        nn.Linear(64, num_class),
    )


class MAHGCNNET_fMRI(nn.Module):
    def __init__(self, ROInum, layer, num_class=2, use_self_attn=True,
                 hidden_dim=64, num_heads=4):
        super().__init__()
        self.ROInum = ROInum
        self.layer = layer
        self.hidden_dim = hidden_dim
        self.graph_embed_dim = layer * hidden_dim
        self.enc = HGAA(nn.ReLU(), 0.3, layer, ROI_num=ROInum,
                        use_self_attn=use_self_attn, hidden_dim=hidden_dim, num_heads=num_heads)
        self.sizes_asc, self.sizes_desc = self.enc.sizes_asc, self.enc.sizes_desc
        self.cls = _classifier(self.graph_embed_dim, num_class)

    def prepare_inputs(self, fmri_scales):
        """Each scale is [batch, ROI, ROI + 2]: dense FC, time-series mean, std."""
        return [graph.unsqueeze(1) for graph in fmri_scales]

    def forward(self, fmri_scales):
        cat, graph_layers, node_layers = self.enc(self.prepare_inputs(fmri_scales))
        return {"logits": self.cls(cat), "feat_cat": cat,
                "feat_layers": graph_layers, "node_layers": node_layers}


class MAHGCNNET_EEG(nn.Module):
    def __init__(self, ROInum, layer, num_class=2, bands=BANDS, use_self_attn=True,
                 hidden_dim=64, num_heads=4):
        super().__init__()
        self.ROInum = ROInum
        self.layer = layer
        self.bands = list(bands)
        self.hidden_dim = hidden_dim
        self.graph_embed_dim = layer * hidden_dim
        self.enc = HGAA(nn.ReLU(), 0.3, layer, ROI_num=ROInum,
                        use_self_attn=use_self_attn, hidden_dim=hidden_dim, num_heads=num_heads)
        self.sizes_asc, self.sizes_desc = self.enc.sizes_asc, self.enc.sizes_desc
        self.cls = _classifier(self.graph_embed_dim, num_class)

    def prepare_inputs(self, eeg_scales):
        """Each band is [batch, ROI, ROI + 2]: dense FC, time-series mean, std."""
        return [torch.stack([scale[band] for band in self.bands], dim=1)
                for scale in eeg_scales]

    def forward(self, eeg_scales):
        cat, graph_layers, node_layers = self.enc(self.prepare_inputs(eeg_scales))
        return {"logits": self.cls(cat), "feat_cat": cat,
                "feat_layers": graph_layers, "node_layers": node_layers}


class MAHGCNNET(nn.Module):
    """Stage 2: GCN -> cross-attention -> readout -> pool at each descending scale."""

    def __init__(self, ROInum, layer, num_class=2, hidden_dim=64, num_heads=4, bands=BANDS):
        super().__init__()
        self.ROInum = ROInum
        self.layer = layer
        self.hidden_dim = hidden_dim
        self.graph_embed_dim = layer * hidden_dim
        self.fmri_net = MAHGCNNET_fMRI(
            ROInum, layer, num_class, hidden_dim=hidden_dim, num_heads=num_heads)
        self.eeg_net = MAHGCNNET_EEG(
            ROInum, layer, num_class, bands=bands, hidden_dim=hidden_dim, num_heads=num_heads)
        self.sizes_asc, self.sizes_desc = self.fmri_net.sizes_asc, self.fmri_net.sizes_desc
        self.cross_attentions = nn.ModuleList([
            CrossInteraction(hidden_dim, nhead=num_heads, dropout=0.3) for _ in range(layer)
        ])
        self.mix_logits = nn.Parameter(torch.zeros(layer))
        self.cls_fused = _classifier(self.graph_embed_dim, num_class)
        # Keep Stage 1 state-dict compatibility; these modules are bypassed in Stage 2.
        for branch in (self.fmri_net, self.eeg_net):
            for module in (branch.enc.self_attentions, branch.enc.attn_ln, branch.cls):
                module.requires_grad_(False)

    def load_stage1(self, fmri_state_dict, eeg_state_dict):
        """Load the two complete Stage 1 branch state dicts from the same architecture."""
        self.fmri_net.load_state_dict(fmri_state_dict)
        self.eeg_net.load_state_dict(eeg_state_dict)
        return self

    def forward(self, fmri_scales, eeg_scales):
        fmri_adj, fmri = self.fmri_net.enc.prepare_graphs(self.fmri_net.prepare_inputs(fmri_scales))
        eeg_adj, eeg = self.eeg_net.enc.prepare_graphs(self.eeg_net.prepare_inputs(eeg_scales))
        fmri_layers, eeg_layers, fused_layers = [], [], []
        for index, cross_attention in enumerate(self.cross_attentions):
            fmri = self.fmri_net.enc.net_gcn[index](fmri_adj[index], fmri)
            eeg = self.eeg_net.enc.net_gcn[index](eeg_adj[index], eeg)
            fmri, eeg = cross_attention(fmri, eeg)
            fmri_graph, eeg_graph = fmri.mean(dim=1).mean(dim=1), eeg.mean(dim=1).mean(dim=1)
            weight = self.mix_logits[index].sigmoid()
            fmri_layers.append(fmri_graph)
            eeg_layers.append(eeg_graph)
            fused_layers.append(weight * fmri_graph + (1.0 - weight) * eeg_graph)
            if index < self.layer - 1:
                fmri = self.fmri_net.enc.net_pool[index](fmri)
                eeg = self.eeg_net.enc.net_pool[index](eeg)

        fmri_layers.reverse()
        eeg_layers.reverse()
        fused_layers.reverse()
        cat = torch.cat(fused_layers, dim=1)
        return {"fused_logits": self.cls_fused(cat), "fused_features": fused_layers,
                "fused_feat_cat": cat, "fmri_features": fmri_layers, "eeg_features": eeg_layers}
