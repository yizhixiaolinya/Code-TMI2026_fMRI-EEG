import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Dict, List, Optional
import MAHGCN as HGAA  # Import HGAA module (file still named MAHGCN.py for compatibility)
from Dual_cross import BCMFBlock

def _l2_normalize(x, eps: float = 1e-6):
    return x / (x.norm(dim=1, keepdim=True) + eps)

def _cosine_loss(x, y):
    x_n = _l2_normalize(x); y_n = _l2_normalize(y)
    return (1.0 - (x_n * y_n).sum(dim=1)).mean()

def _mse_loss(x, y):
    return F.mse_loss(x, y)

def calculate_correlation(X, Y):
    """
    Calculate correlation between two tensors (reference: Max_Corr_alignment_model.py)
    
    Args:
        X: [B, N, D] or [B, N] tensor
        Y: [B, N, D] or [B, N] tensor, same shape as X
    
    Returns:
        correlation: Scalar, average correlation
    """
    # Ensure consistent dimensions
    if X.dim() == 2:
        X = X.unsqueeze(-1)  # [B, N] -> [B, N, 1]
    if Y.dim() == 2:
        Y = Y.unsqueeze(-1)  # [B, N] -> [B, N, 1]
    
    # Calculate mean
    mean_X = torch.mean(X, dim=-1, keepdim=True)  # [B, N, 1]
    mean_Y = torch.mean(Y, dim=-1, keepdim=True)  # [B, N, 1]
    
    # Center
    X_centered = X - mean_X  # [B, N, D]
    Y_centered = Y - mean_Y  # [B, N, D]
    
    # Calculate standard deviation product
    std_product = torch.sqrt(
        torch.sum(X_centered**2, dim=-1) * torch.sum(Y_centered**2, dim=-1) + 1e-8
    )  # [B, N]
    
    # Calculate correlation
    correlation = torch.sum(X_centered * Y_centered, dim=-1) / std_product  # [B, N]
    
    # Return average correlation
    return correlation.mean()

# ---------------- Utility: reorganize eeg_scales as {band: [scale_tensors...]} ----------------
def reorganize_eeg_scales(eeg_scales, bands):
    """
    eeg_scales: list[dict], each element represents a scale, dict keys are band names, values are [B,N,N] tensors
    bands: List of band names
    Returns: {band: [tensor_scale1, tensor_scale2, ...]}
    """
    eeg_by_band = {band: [] for band in bands}
    for scale_dict in (eeg_scales or []):
        if scale_dict is None:
            continue
        for band in bands:
            t = scale_dict.get(band, None)
            if t is not None:
                eeg_by_band[band].append(t)
    return eeg_by_band

# ======================= fMRI Dedicated Network =======================
class MAHGCNNET_fMRI(nn.Module):
    def __init__(self, ROInum: int, layer: int, num_class: int = 2, use_self_attn: bool = True):
        super().__init__()
        self.ROInum = ROInum
        self.layer = layer
        self.sizes_asc  = [ROInum + 100*i for i in range(layer)]
        self.sizes_desc = list(reversed(self.sizes_asc))   # e.g., [400,300,200]
        self.hidden_dim = 64  # Fixed hidden dimension d (aligned with embed_dim)
        self.graph_embed_dim = self.layer * self.hidden_dim  # Total graph embedding dimension

        self.enc = HGAA.HGAA(act=nn.ReLU(), drop_p=0.3, layer=self.layer, ROI_num=self.ROInum, 
                             use_self_attn=use_self_attn, hidden_dim=self.hidden_dim)

        self.cls = nn.Sequential(
            nn.LayerNorm(self.graph_embed_dim), nn.ReLU(),
            nn.Linear(self.graph_embed_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, num_class)
        )

    def _build_init_feat(self, B, N, device):
        I = torch.eye(N, device=device)
        return I.unsqueeze(0).expand(B, N, N)

    def _encode_with_layers(self, graphs: List[torch.Tensor], h: torch.Tensor):
        """
        Encode graphs using HGAA
        
        Returns:
            feat_cat: [B, layer * hidden_dim] - concatenated graph embeddings for classification
            feat_layers: list of [B, hidden_dim] - graph-level embeddings for each scale (for Stage3 alignment)
            node_layers: list of [B, R_s, hidden_dim] - node-level features for each scale (for fusion)
        """
        graphs = sorted(graphs, key=lambda g: g.shape[-1], reverse=True)
        feat_cat, feat_layers, node_layers = self.enc(graphs, h)
        # feat_cat: [B, layer * hidden_dim]
        # feat_layers: list of [B, hidden_dim] (graph-level embeddings)
        # node_layers: list of [B, R_s, hidden_dim] (node-level features)
        return feat_cat, feat_layers, node_layers

    def forward(self, fmri_scales: List[torch.Tensor]):
        # Statistics: device/B/N_max
        device, B, N_max = None, None, 0
        for g in (fmri_scales or []):
            if g is None: continue
            if device is None:
                device, B = g.device, g.shape[0]
            N_max = max(N_max, g.shape[1])
        if device is None or B is None or N_max == 0:
            raise RuntimeError("[MAHGCNNET_fMRI] Empty fmri_scales.")

        h = self._build_init_feat(B, N_max, device)
        cat, graph_layers, node_layers = self._encode_with_layers(fmri_scales, h)
        # cat: [B, layer * hidden_dim] - for classification
        # graph_layers: list of [B, hidden_dim] - graph-level embeddings (g^m_s)
        # node_layers: list of [B, R_s, hidden_dim] - node-level features (for fusion)

        # Preserve gradients for graph embeddings at each layer for Stage3 alignment
        for x in graph_layers:
            if not x.requires_grad:
                x.requires_grad_(True)
            try:
                x.retain_grad()
            except RuntimeError:
                pass

        logits = self.cls(cat)
        return {
            "logits": logits, 
            "feat_cat": cat,  # [B, layer * hidden_dim] for classification
            "feat_layers": graph_layers,  # list of [B, hidden_dim] - graph-level embeddings (g^m_s)
            "node_layers": node_layers  # list of [B, R_s, hidden_dim] - node-level features
        }


# ======================= EEG Dedicated Network =======================
class MAHGCNNET_EEG(nn.Module):
    def __init__(self, ROInum: int, layer: int, num_class: int = 2,
                 bands: Sequence[str] = ('FC_delta','FC_theta','FC_alpha','FC_beta','FC_gamma'),
                 use_self_attn: bool = True):
        super().__init__()
        self.ROInum = ROInum
        self.layer = layer
        self.bands = list(bands)

        self.sizes_asc  = [ROInum + 100*i for i in range(layer)]
        self.sizes_desc = list(reversed(self.sizes_asc))
        self.hidden_dim = 64  # Fixed hidden dimension d (aligned with embed_dim)
        self.graph_embed_dim = self.layer * self.hidden_dim  # Total graph embedding dimension

        self.enc = HGAA.HGAA(act=nn.ReLU(), drop_p=0.3, layer=self.layer, ROI_num=self.ROInum, 
                             use_self_attn=use_self_attn, hidden_dim=self.hidden_dim)

        # Band gating networks: for each layer, input global features (mean over ROI dimension), output [B, n_bands] scores
        # Each gate network: input [B, n_bands] (global features for each band), output [B, n_bands] scores
        self.band_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(self.bands), len(self.bands) * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(len(self.bands) * 2, len(self.bands)),
            ) for _ in self.sizes_desc
        ])
        self.mlp = nn.Sequential(
            nn.Linear(self.graph_embed_dim, self.graph_embed_dim),
            nn.ReLU(),
            nn.Linear(self.graph_embed_dim, self.graph_embed_dim),
        )
        self.cls = nn.Sequential(
            nn.LayerNorm(self.graph_embed_dim), nn.ReLU(),
            nn.Linear(self.graph_embed_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, num_class)
        )

    def _build_init_feat(self, B, N, device):
        I = torch.eye(N, device=device)
        return I.unsqueeze(0).expand(B, N, N)

    def _encode_with_layers(self, graphs: List[torch.Tensor], h: torch.Tensor):
        """
        Encode graphs using HGAA (for single band)
        
        Returns:
            feat_cat: [B, layer * hidden_dim] - concatenated graph embeddings
            feat_layers: list of [B, hidden_dim] - graph-level embeddings for each scale
            node_layers: list of [B, R_s, hidden_dim] - node-level features for each scale
        """
        graphs = sorted(graphs, key=lambda g: g.shape[-1], reverse=True)
        feat_cat, feat_layers, node_layers = self.enc(graphs, h)
        return feat_cat, feat_layers, node_layers

    def forward(self, eeg_scales: List[Dict[str, torch.Tensor]]):
        device, B, N_max = None, None, 0
        for d in (eeg_scales or []):
            if d is None: continue
            for t in d.values():
                if t is None: continue
                if device is None:
                    device, B = t.device, t.shape[0]
                N_max = max(N_max, t.shape[1])
        if device is None or B is None or N_max == 0:
            raise RuntimeError("[MAHGCNNET_EEG] Empty eeg_scales.")

        h = self._build_init_feat(B, N_max, device)
        eeg_by_band = reorganize_eeg_scales(eeg_scales, self.bands)

        # Collect graph-level embeddings and node-level features for each scale
        graph_layers: List[torch.Tensor] = []  # list of [B, hidden_dim] - graph-level embeddings
        node_layers_per_scale: List[torch.Tensor] = []  # list of [B, R_s, hidden_dim] - node-level features
        band_alphas: List[torch.Tensor] = []  # Save band weights for each layer
        
        for li, N_i in enumerate(self.sizes_desc):
            # Process each band for this scale
            band_node_features = []  # List of [B, R_s, hidden_dim] for each band
            for band in self.bands:
                band_scales = eeg_by_band[band]
                if len(band_scales) == 0:
                    # Create zero node features [B, R_s, hidden_dim]
                    band_node_features.append(torch.zeros(B, N_i, self.enc.hidden_dim, device=device))
                    continue
                _, _, node_layers = self._encode_with_layers(band_scales, h)
                band_node_features.append(node_layers[li])  # [B, R_s, hidden_dim]
            
            # Stack band features: [B, n_bands, R_s, hidden_dim]
            stack = torch.stack(band_node_features, dim=1)  # [B, len(bands), R_s, hidden_dim]
            
            # Band gating: compute global features for each band (mean over ROI dimension)
            # Paper: channel-average \bar{H}'^m_s for attention
            band_global_feat = stack.mean(dim=2)  # [B, len(bands), R_s, hidden_dim] -> [B, len(bands), hidden_dim]
            # Further reduce to [B, len(bands)] by mean over hidden_dim for gating
            band_global_feat_reduced = band_global_feat.mean(dim=-1)  # [B, len(bands)]
            
            # Gate network outputs scores, then softmax to get normalized weights
            gate_scores = self.band_gates[li](band_global_feat_reduced)  # [B, n_bands]
            alpha = F.softmax(gate_scores, dim=1)  # [B, n_bands] - normalized weights
            
            # Expand alpha to [B, n_bands, 1, 1] for broadcasting, do weighted sum
            alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1)  # [B, n_bands, 1, 1]
            fused_nodes = (alpha_expanded * stack).sum(dim=1)  # [B, n_bands, R_s, hidden_dim] -> [B, R_s, hidden_dim]
            
            # Graph-level embedding: mean pooling over ROI dimension
            fused_graph = fused_nodes.mean(dim=1)  # [B, R_s, hidden_dim] -> [B, hidden_dim]

            # Retain gradients for graph embeddings
            if not fused_graph.requires_grad:
                fused_graph.requires_grad_(True)
            try:
                fused_graph.retain_grad()
            except RuntimeError:
                pass

            graph_layers.append(fused_graph)  # [B, hidden_dim]
            node_layers_per_scale.append(fused_nodes)  # [B, R_s, hidden_dim]
            band_alphas.append(alpha)  # Save band weights for this layer

        # Concatenate graph embeddings for classification
        cat = torch.cat(graph_layers, dim=1)  # [B, layer * hidden_dim]
        cat = self.mlp(cat)
        logits = self.cls(cat)
        
        return {
            "logits": logits, 
            "feat_cat": cat,  # [B, layer * hidden_dim] for classification
            "feat_layers": graph_layers,  # list of [B, hidden_dim] - graph-level embeddings (g^m_s)
            "node_layers": node_layers_per_scale,  # list of [B, R_s, hidden_dim] - node-level features
            "band_alphas": band_alphas
        }


# ======================= Fusion Module =======================
class MAHGCNNET_Fusion(nn.Module):
    def __init__(self, sizes_desc: List[int], embed_dim: int = 64, num_class: int = 2,
                cls_input: str = "fused", use_msre: bool = False, 
                use_corr_align: bool = True, corr_align_dim: int = 16,
                temperature: float = 5.0, fusion_mode: str = "dual_cross"):
        super().__init__()
        self.sizes_desc = list(sizes_desc)
        self.embed_dim  = embed_dim  # Fixed embedding dimension d (default 64)
        self.cls_input  = cls_input  # ← Must assign
        self.use_msre    = use_msre    # MSRE module switch (optional)
        self.use_corr_align = use_corr_align  # Whether to use correlation alignment
        self.corr_align_dim = corr_align_dim  # Alignment projection dimension
        self.temperature = temperature  # CrossInteraction temperature parameter
        self.fusion_mode = fusion_mode  # "bcmf", "concat_only", or "learnable_mix" (formerly "dual_cross")
        
        # Total graph embedding dimension for classification
        self.graph_embed_dim = len(self.sizes_desc) * self.embed_dim

        # Alignment projection layers (for correlation alignment before fusion)
        # Project from hidden_dim to corr_align_dim
        if self.use_corr_align:
            self.fmri_align_proj = nn.ModuleList([
                nn.Linear(embed_dim, corr_align_dim) for _ in self.sizes_desc
            ])
            self.eeg_align_proj = nn.ModuleList([
                nn.Linear(embed_dim, corr_align_dim) for _ in self.sizes_desc
            ])

        # Projection layers: node features [B, R_s, hidden_dim] -> [B, R_s, embed_dim]
        # Note: node_layers from HGAA are already [B, R_s, hidden_dim] where hidden_dim=embed_dim
        # So we might not need projection, but keep for flexibility
        self.fmri_proj = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in self.sizes_desc])
        self.eeg_proj  = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in self.sizes_desc])
        
        # Only create BCMFBlock in bcmf or learnable_mix mode
        # Support both old ("dual_cross") and new ("bcmf") names for backward compatibility
        if fusion_mode in ("bcmf", "learnable_mix", "dual_cross"):
            bcmf_modules = nn.ModuleList([
                BCMFBlock(d_model=embed_dim, nhead=4, num_layers=2, use_msre=use_msre, num_class=num_class, temperature=temperature)
                for _ in self.sizes_desc
            ])
            self.bcmf = bcmf_modules
            # Keep dualcross as alias for backward compatibility
            self.dualcross = bcmf_modules
        else:
            self.bcmf = None  # concat_only mode doesn't need it
            self.dualcross = None
        
        # Learnable scale-specific mixing coefficients λ_s ∈ [0,1] for learnable_mix mode
        if fusion_mode == "learnable_mix":
            # Initialize with sigmoid to ensure [0,1] range
            # Use nn.Parameter with logit initialization (sigmoid(logit) gives desired initial value)
            # Initialize to 0.5 (equal mixing) -> logit = 0
            self.mix_coeffs = nn.ParameterList([
                nn.Parameter(torch.zeros(1))  # Will be passed through sigmoid
                for _ in self.sizes_desc
            ])
        else:
            self.mix_coeffs = None

        # Classification head input dimension
        # All inputs are now graph-level embeddings [B, layer * embed_dim]
        if self.cls_input == "fused":
            in_cls = self.graph_embed_dim
        elif self.cls_input == "early":
            in_cls = self.graph_embed_dim * 2
        else:  # "all"
            in_cls = self.graph_embed_dim * 3

        self.cls_fused = nn.Sequential(
            nn.LayerNorm(in_cls), nn.ReLU(),
            nn.Linear(in_cls, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, num_class)
        )

    def forward(self, fmri_node_layers: List[torch.Tensor], eeg_node_layers: List[torch.Tensor], 
                return_attn_weights: bool = False, debug: bool = False):
        """
        Forward pass of Fusion module
        
        Args:
            fmri_node_layers: list of [B, R_s, hidden_dim] - node-level features from fMRI encoder
            eeg_node_layers: list of [B, R_s, hidden_dim] - node-level features from EEG encoder
        
        Returns:
            result dict containing:
                - logits_fused: [B, num_class]
                - feat_fused_cat: [B, layer * embed_dim] - concatenated graph embeddings
                - feat_fused_layers: list of [B, embed_dim] - graph-level embeddings for each scale (g^{fuse}_s)
                - feat_fmri_cat: [B, layer * embed_dim]
                - feat_eeg_cat: [B, layer * embed_dim]
        """
        assert len(fmri_node_layers) == len(eeg_node_layers) == len(self.sizes_desc)

        fused_graph_layers = []  # List of [B, embed_dim] - graph-level fused embeddings
        msre_info_list = []  # Collect MSRE info for all layers
        attn_weights_list = []  # Collect attention weights for all layers
        corr_align_loss = torch.zeros((), device=fmri_node_layers[0].device, dtype=torch.float32)  # Correlation alignment loss
        
        # ========== Correlation alignment before fusion ==========
        if self.use_corr_align:
            corr_parts = []
            for li, N_i in enumerate(self.sizes_desc):
                f_nodes = fmri_node_layers[li]  # [B, R_s, hidden_dim]
                e_nodes = eeg_node_layers[li]    # [B, R_s, hidden_dim]

                # Project node features for correlation alignment
                # f_nodes: [B, R_s, hidden_dim], project to [B, R_s, corr_align_dim]
                X = f_nodes @ self.fmri_align_proj[li]  # [B, R_s, hidden_dim] @ [hidden_dim, corr_align_dim] -> [B, R_s, corr_align_dim]
                Y = e_nodes @ self.eeg_align_proj[li]   # [B, R_s, hidden_dim] @ [hidden_dim, corr_align_dim] -> [B, R_s, corr_align_dim]
                
                # Transpose for correlation calculation: [B, R_s, corr_align_dim] -> [B, corr_align_dim, R_s]
                X = X.transpose(1, 2)  # [B, corr_align_dim, R_s]
                Y = Y.transpose(1, 2)  # [B, corr_align_dim, R_s]
                
                # Compute correlation
                corr = calculate_correlation(X, Y)  # Scalar
                corr_parts.append(corr)
            
            # Maximize correlation = minimize negative correlation
            if len(corr_parts) > 0:
                corr_align_loss = -torch.stack(corr_parts).sum()  # Sum of negative correlations (minimize this = maximize correlation)
        
        for li, N_i in enumerate(self.sizes_desc):
            # Input: node-level features [B, R_s, hidden_dim] from HGAA
            f_nodes = fmri_node_layers[li]  # [B, R_s, hidden_dim]
            e_nodes = eeg_node_layers[li]    # [B, R_s, hidden_dim]
            
            # Project to embed_dim (usually identity if hidden_dim == embed_dim)
            f_tok = self.fmri_proj[li](f_nodes)  # [B, R_s, embed_dim]
            e_tok = self.eeg_proj[li](e_nodes)   # [B, R_s, embed_dim]
            
            # ========== concat_only mode: directly concat, no cross-attention ==========
            if self.fusion_mode == "concat_only":
                # Simple average as fusion result
                z = (f_tok + e_tok) / 2.0  # [B, R_s, embed_dim]
                # Graph-level embedding: mean pooling over ROI dimension
                fused = z.mean(dim=1)  # [B, R_s, embed_dim] -> [B, embed_dim]
            elif self.fusion_mode == "learnable_mix":
                # ========== learnable_mix mode: cross-attention + learnable mixing coefficients ==========
                layer_name = f"layer{li}"
                
                # Apply cross-attention refinement using BCMF
                if self.use_msre:
                    bcmf_result = self.bcmf[li](f_tok, e_tok, return_attn_weights=return_attn_weights,
                                                      debug=debug, layer_name=layer_name)
                    if return_attn_weights and len(bcmf_result) > 4:
                        a_out, b_out, z_temp, msre_info, attn_meta = bcmf_result
                        attn_weights_list.append(attn_meta)
                    else:
                        a_out, b_out, z_temp, msre_info = bcmf_result
                    msre_info_list.append(msre_info)
                else:
                    bcmf_result = self.bcmf[li](f_tok, e_tok, return_attn_weights=return_attn_weights,
                                                      debug=debug, layer_name=layer_name)
                    if return_attn_weights and len(bcmf_result) > 3:
                        a_out, b_out, z_temp, attn_meta = bcmf_result
                        attn_weights_list.append(attn_meta)
                    else:
                        a_out, b_out, z_temp = bcmf_result
                
                # Generate scale-level graph embeddings: mean pooling over ROI dimension
                # Paper: \tilde{g}^f_s = mean(a_out, dim=ROI), \tilde{g}^e_s = mean(b_out, dim=ROI)
                # a_out: [B, R_s, embed_dim] (fMRI enhanced features after cross-attention)
                # b_out: [B, R_s, embed_dim] (EEG enhanced features after cross-attention)
                g_f = a_out.mean(dim=1)  # [B, R_s, embed_dim] -> [B, embed_dim] - fMRI graph embedding
                g_e = b_out.mean(dim=1)  # [B, R_s, embed_dim] -> [B, embed_dim] - EEG graph embedding
                
                # Apply learnable scale-specific mixing coefficient λ_s
                # Paper: λ_s = sigmoid(tildeλ_s) ∈ [0,1] (subject-shared, scale-specific)
                lambda_s = torch.sigmoid(self.mix_coeffs[li])  # Scalar or [1] tensor
                lambda_s = lambda_s.view(1, 1)  # Ensure [1, 1] shape for broadcasting
                
                # Fusion: g^{fuse}_s = λ_s * \tilde{g}^f_s + (1-λ_s) * \tilde{g}^e_s
                fused = lambda_s * g_f + (1.0 - lambda_s) * g_e  # [B, embed_dim]
            else:
                # ========== bcmf mode: use BCMFBlock (bidirectional cross-modal fusion) ==========
                layer_name = f"layer{li}"
                
                if self.use_msre:
                    bcmf_result = self.bcmf[li](f_tok, e_tok, return_attn_weights=return_attn_weights,
                                                      debug=debug, layer_name=layer_name)
                    if return_attn_weights and len(bcmf_result) > 4:
                        _, _, z, msre_info, attn_meta = bcmf_result
                        attn_weights_list.append(attn_meta)
                    else:
                        _, _, z, msre_info = bcmf_result
                    msre_info_list.append(msre_info)
                else:
                    bcmf_result = self.bcmf[li](f_tok, e_tok, return_attn_weights=return_attn_weights,
                                                      debug=debug, layer_name=layer_name)
                    if return_attn_weights and len(bcmf_result) > 3:
                        _, _, z, attn_meta = bcmf_result
                        attn_weights_list.append(attn_meta)
                    else:
                        _, _, z = bcmf_result

                # Graph-level embedding: mean pooling over ROI dimension
                # Paper: g^{fuse}_s = mean(z, dim=ROI) where z is fused node features
                fused = z.mean(dim=1)  # [B, R_s, embed_dim] -> [B, embed_dim]

            # Retain gradients (for Stage3 alignment)
            if not fused.requires_grad:
                fused.requires_grad_(True)
            try:
                fused.retain_grad()
            except RuntimeError:
                pass

            fused_graph_layers.append(fused)  # [B, embed_dim]

        # Concatenate graph-level embeddings for classification
        # All are now [B, embed_dim] per scale, so cat -> [B, layer * embed_dim]
        fused_cat = torch.cat(fused_graph_layers, dim=1)  # [B, layer * embed_dim] - fused graph embeddings
        
        # Compute graph-level embeddings from node features for fmri and eeg
        fmri_graph_layers = [nodes.mean(dim=1) for nodes in fmri_node_layers]  # list of [B, embed_dim]
        eeg_graph_layers = [nodes.mean(dim=1) for nodes in eeg_node_layers]   # list of [B, embed_dim]
        fmri_cat = torch.cat(fmri_graph_layers, dim=1)  # [B, layer * embed_dim]
        eeg_cat = torch.cat(eeg_graph_layers, dim=1)    # [B, layer * embed_dim]
        fused_in = torch.cat([fmri_cat, eeg_cat], dim=1)  # [B, 2 * layer * embed_dim] early concatenation

        # Select classification input
        if self.cls_input == "fused":
            cls_feat = fused_cat
        elif self.cls_input == "early":
            cls_feat = fused_in
        else:  # "all"
            cls_feat = torch.cat([fused_cat, fmri_cat, eeg_cat], dim=1)

        logits_fused = self.cls_fused(cls_feat)

        result = {
            "logits_fused": logits_fused,
            "feat_fused_cat": fused_cat,  # [B, layer * embed_dim] - for classification
            "feat_fused_layers": fused_graph_layers,  # list of [B, embed_dim] - graph-level embeddings (g^{fuse}_s)
            "feat_fmri_cat": fmri_cat,  # [B, layer * embed_dim]
            "feat_eeg_cat": eeg_cat,    # [B, layer * embed_dim]
            "feat_early_cat": fused_in,   # [B, 2 * layer * embed_dim] - early concatenation
            "g_layers_fmri": fmri_graph_layers,  # list of [B, embed_dim] - graph-level embeddings (g^f_s)
            "g_layers_eeg": eeg_graph_layers,    # list of [B, embed_dim] - graph-level embeddings (g^e_s)
            "g_layers_fuse": fused_graph_layers,  # list of [B, embed_dim] - graph-level embeddings (g^{fuse}_s)
            "corr_align_loss": corr_align_loss,  # Correlation alignment loss
        }
        
        # If MSRE enabled, add MSRE-related information
        if self.use_msre and len(msre_info_list) > 0:
            # Aggregate MSRE info for all layers (can take average or last layer, here we keep all)
            # Note: Each layer's msre_info is a dict containing logits_f_a, logits_f_b, msre_a, msre_b, etc.
            result["msre_info"] = msre_info_list
            # For convenience, also provide aggregated version (take last layer as representative)
            result["msre_info_last"] = msre_info_list[-1] if msre_info_list else None
        
        # If attention weights requested, add attention weight information
        if return_attn_weights and len(attn_weights_list) > 0:
            result["attn_weights"] = attn_weights_list  # Attention weight dict for each layer

        return result
    
def _retain_layers(layer_list):
    if layer_list is None:
        return None
    kept = []
    for t in layer_list:
        if t is None:
            kept.append(None)
            continue
        # Per-layer node representations are usually non-leaf, need retain_grad()
        if not t.requires_grad:
            t.requires_grad_(True)
        try:
            t.retain_grad()
        except RuntimeError:
            pass
        kept.append(t)
    return kept


# ======================= Top Level =======================
class MAHGCNNET(nn.Module):
    def __init__(self, ROInum, layer, num_scales, num_class=2, fusion_mode="bcmf",
                 align_loss_type="mse", align_to_consensus_lambda=1.0,
                 bands: Sequence[str] = ('FC_delta','FC_theta','FC_alpha','FC_beta','FC_gamma'),
                 align_to_fused_lambda=1.0, embed_dim: int = 64,
                 pairwise_lambda: float = 0.0, use_msre: bool = False,
                 use_corr_align: bool = True, corr_align_dim: int = 16,
                 temperature: float = 5.0, use_self_attn: bool = True):   # ← New: self-attention switch
        super().__init__()
        self.ROInum = ROInum
        self.layer  = layer
        self.num_scales = num_scales
        self.bands = list(bands)
        self.use_msre = use_msre  # MSRE module switch (optional)
        self.use_corr_align = use_corr_align  # Correlation alignment switch

        self.sizes_asc  = [ROInum + 100*i for i in range(layer)]
        self.sizes_desc = list(reversed(self.sizes_asc))
        self.paranum    = sum(self.sizes_desc)

        self.fmri_net = MAHGCNNET_fMRI(ROInum, layer, num_class=num_class, use_self_attn=use_self_attn)
        self.eeg_net  = MAHGCNNET_EEG (ROInum, layer, num_class=num_class, bands=bands, use_self_attn=use_self_attn)
        
        # Create fusion module, decide whether to enable corresponding features based on use_msre and use_corr_align parameters
        self.fusion_mode = fusion_mode
        self.fuser    = MAHGCNNET_Fusion(self.sizes_desc, embed_dim=embed_dim, num_class=num_class, 
                                        cls_input="all", use_msre=use_msre,
                                        use_corr_align=use_corr_align, corr_align_dim=corr_align_dim,
                                        temperature=temperature, fusion_mode=fusion_mode)

        self.align_loss_type = align_loss_type              # "cosine" or "mse"
        self.align_to_consensus_lambda = align_to_consensus_lambda  # Can be used for layer-wise weighting
        self.align_to_fused_lambda = align_to_fused_lambda  # Top-level (cat) weight
        self.pairwise_lambda = pairwise_lambda              # Can be 0 (default disabled)

    def forward(self,
                graph_scales: Optional[List] = None,
                fmri_only: bool = False,
                eeg_only: bool  = False,
                fmri_scales: Optional[List[torch.Tensor]] = None,
                eeg_scales:  Optional[List[Dict[str, torch.Tensor]]] = None,
                return_attn_weights: bool = False):

        # ===================== fMRI Only =====================
        if fmri_only and not eeg_only:
            if fmri_scales is None:
                fmri_scales = graph_scales
            o_f = self.fmri_net(fmri_scales)

            # Expose and ensure backpropagation
            self.last_fmri_layers  = _retain_layers(o_f["feat_layers"])
            self.last_eeg_layers   = None
            self.last_fused_layers = None

            if self.use_msre:
                return (
                    o_f["logits"],                      # out_fmri
                    None,                               # out_eeg
                    None,                               # out_fused
                    torch.zeros((), device=o_f["feat_cat"].device),  # align_loss (placeholder)
                    None,                               # fused_cons (placeholder)
                    o_f["feat_cat"],                    # fmri_feat_for_proto
                    None,                               # eeg_feat_for_proto
                    None                                # msre_info (placeholder)
                )
            else:
                return (
                    o_f["logits"],                      # out_fmri
                    None,                               # out_eeg
                    None,                               # out_fused
                    torch.zeros((), device=o_f["feat_cat"].device),  # align_loss (placeholder)
                    None,                               # fused_cons (placeholder)
                    o_f["feat_cat"],                    # fmri_feat_for_proto
                    None                                # eeg_feat_for_proto
                )

        # ===================== EEG Only =====================
        if eeg_only and not fmri_only:
            if eeg_scales is None:
                eeg_scales = graph_scales
            o_e = self.eeg_net(eeg_scales)

            # Expose and ensure backpropagation
            self.last_fmri_layers  = None
            self.last_eeg_layers   = _retain_layers(o_e["feat_layers"])
            self.last_fused_layers = None

            if self.use_msre:
                return (
                    None,                               # out_fmri
                    o_e["logits"],                      # out_eeg
                    None,                               # out_fused
                    torch.zeros((), device=o_e["feat_cat"].device),  # align_loss (placeholder)
                    None,                               # fused_cons (placeholder)
                    None,                               # fmri_feat_for_proto
                    o_e["feat_cat"],                    # eeg_feat_for_proto
                    None                                # msre_info (placeholder)
                )
            else:
                return (
                    None,                               # out_fmri
                    o_e["logits"],                      # out_eeg
                    None,                               # out_fused
                    torch.zeros((), device=o_e["feat_cat"].device),  # align_loss (placeholder)
                    None,                               # fused_cons (placeholder)
                    None,                               # fmri_feat_for_proto
                    o_e["feat_cat"]                     # eeg_feat_for_proto
                )

        # ===================== Fusion (EEG + fMRI) =====================
        if (fmri_scales is not None) and (eeg_scales is not None):
            o_f = self.fmri_net(fmri_scales)
            o_e = self.eeg_net(eeg_scales)
            # Pass debug flag
            debug_attn = getattr(self, '_debug_attn', False)
            # Use node_layers for fusion (node-level features [B, R_s, hidden_dim])
            fused = self.fuser(o_f["node_layers"], o_e["node_layers"], 
                              return_attn_weights=return_attn_weights, debug=debug_attn)

            # Expose and ensure backpropagation
            # feat_layers are now graph-level embeddings [B, hidden_dim]
            self.last_fmri_layers  = _retain_layers(o_f["feat_layers"])  # list of [B, hidden_dim]
            self.last_eeg_layers   = _retain_layers(o_e["feat_layers"])  # list of [B, hidden_dim]
            self.last_fused_layers = _retain_layers(fused["feat_fused_layers"])  # list of [B, hidden_dim]

            # ---------- Select distance metric ----------
            if self.align_loss_type.lower() == "cosine":
                _dist = _cosine_loss
            else:
                _dist = _mse_loss

            # ---------- Get graph-level representations ----------
            # All are now [B, layer * embed_dim] (graph-level embeddings concatenated)
            fmri_cat  = o_f["feat_cat"]               # [B, layer * embed_dim]
            eeg_cat   = o_e["feat_cat"]               # [B, layer * embed_dim]
            fused_cat = fused["feat_fused_cat"]       # [B, layer * embed_dim]

            # ---------- Consensus teacher (stop-grad) ----------
            # For Stage3, we'll use scale-wise alignment, but keep cat-level for backward compatibility
            teacher = fused_cat.detach()

            # ---------- Top-level alignment (cat level) ----------
            # Note: Stage3 will use scale-wise alignment (L_align^m = Σ_s || g^m_s - g^{fuse}_s ||_2^2)
            # This cat-level alignment is kept for backward compatibility but may not be used in Stage3
            align_loss_cat = _dist(fmri_cat, teacher) + _dist(eeg_cat, teacher)
            align_loss_cat = self.align_to_fused_lambda * align_loss_cat

            align_loss = align_loss_cat
            
            # ---------- Correlation alignment loss (before fusion) ----------
            corr_align_loss = fused.get("corr_align_loss", torch.zeros((), device=o_f["logits"].device))

            # If MSRE enabled, extract MSRE info from fused
            msre_info = None
            if self.use_msre and "msre_info" in fused:
                msre_info = fused.get("msre_info", None)
            
            # Extract attention weights (if requested)
            attn_weights = None
            if return_attn_weights and "attn_weights" in fused:
                attn_weights = fused.get("attn_weights", None)

            # Extract scale-wise graph embeddings for Stage3 alignment
            g_layers_fmri = o_f["feat_layers"]  # list of [B, hidden_dim] - graph-level embeddings (g^f_s)
            g_layers_eeg = o_e["feat_layers"]   # list of [B, hidden_dim] - graph-level embeddings (g^e_s)
            g_layers_fuse = fused.get("g_layers_fuse", fused["feat_fused_layers"])  # list of [B, hidden_dim] - graph-level embeddings (g^{fuse}_s)
            
            result = (
                o_f["logits"],                    # out_fmri
                o_e["logits"],                    # out_eeg
                fused["logits_fused"],            # out_fused
                align_loss,                       # align_loss (cat-level, kept for backward compatibility)
                fused["feat_fused_cat"],          # fused_cons: fused representation h_cat^fused after BCMF
                o_f["feat_cat"],                  # fmri_feat_for_proto (use existing directly)
                o_e["feat_cat"],                  # eeg_feat_for_proto (use existing directly)
                msre_info,                        # msre_info: MSRE-related information (if enabled)
                corr_align_loss,                   # corr_align_loss: correlation alignment loss (before fusion)
                g_layers_fmri,                    # g_layers_fmri: list of [B, hidden_dim] for Stage3 alignment
                g_layers_eeg,                     # g_layers_eeg: list of [B, hidden_dim] for Stage3 alignment
                g_layers_fuse                     # g_layers_fuse: list of [B, hidden_dim] for Stage3 alignment
            )
            
            # If attention weights requested, add them to return result
            if return_attn_weights and attn_weights is not None:
                return (*result, attn_weights)
            
            return result

        # ---------------- Invalid parameters ----------------
            raise RuntimeError(
            "[HGAA_NET] Invalid inputs. Use one of: "
            "1) graph_scales+fmri_only=True; 2) graph_scales+eeg_only=True; "
            "3) fmri_scales=..., eeg_scales=..."
        )

