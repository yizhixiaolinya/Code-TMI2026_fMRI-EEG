
"""
BCMF.py
-------------
A Mamba-free fusion module that mirrors Sigma's Cross-SS + Concat-SS logic using
standard PyTorch blocks (MultiheadAttention/Transformer).

It is designed to drop into existing multi-modal pipelines (e.g., EEG–fMRI, RGB–X)
and can operate on either grid features (B, C, H, W) or node/sequence features (B, N, D).

Key ideas:
- CrossInteraction: mutual conditioning via cross-attention
- BCMFOnly: bidirectional cross-modal fusion with cross-attention followed by feature fusion with optional MSRE module

Outputs:
- a_out: modality A enhanced feature (shape matches input)
- b_out: modality B enhanced feature (shape matches input)
- fused: fused feature (single stream), matching input layout
- meta: dict with intermediate tensors for debugging/visualization

License: MIT
"""
from __future__ import annotations
from typing import Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_seq(x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    Convert (B,C,H,W) or (B,N,D) to a unified (B, L, D) sequence.
    Returns: (seq, meta) where meta carries shape info for restoring.
    """
    if x.dim() == 4:
        # (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return seq, {"layout": "bchw", "B": B, "C": C, "H": H, "W": W}
    elif x.dim() == 3:
        # (B, N, D) already sequence
        B, N, D = x.shape
        return x, {"layout": "bnd", "B": B, "N": N, "D": D}
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")


def _from_seq(seq: torch.Tensor, meta: Dict) -> torch.Tensor:
    """Restore back to (B,C,H,W) or (B,N,D) based on meta."""
    if meta["layout"] == "bchw":
        B, C, H, W = meta["B"], meta["C"], meta["H"], meta["W"]
        return seq.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    elif meta["layout"] == "bnd":
        B, N, _ = meta["B"], meta["N"], meta["D"]
        return seq.view(B, N, -1).contiguous()
    else:
        raise ValueError("Unknown layout meta")


class CrossInteraction(nn.Module):
    """
    Cross-interaction via shared similarity matrix + row/column softmax:
    - Compute shared similarity matrix: A_aff = (V_a @ V_b.transpose(-1, -2)) / sqrt(d)
    - Normalize A_aff and apply temperature scaling: A_aff = (A_aff - mean) / std * temperature
    - A←B direction: Apply row softmax to A_aff to get S_a, then R_a = S_a @ V_b
    - B←A direction: Apply column softmax to A_aff to get S_b, then R_b = S_b.transpose(-1, -2) @ V_a
    - Residual and normalization: A_out = ln_a(A + R_a), B_out = ln_b(B + R_b)
    
    Operates on sequences (B, L, D). Input tensors can be in grid or sequence form; we auto-convert.
    """
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.0, temperature: float = 5.0):
        super().__init__()
        # Note: nhead parameter kept for interface compatibility, but new implementation does not use multi-head attention
        # Projection layers for generating V_a and V_b
        self.proj_a = nn.Linear(d_model, d_model)  # A -> V_a
        self.proj_b = nn.Linear(d_model, d_model)  # B -> V_b
        self.ln_a = nn.LayerNorm(d_model)
        self.ln_b = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.d_model = d_model
        self.nhead = nhead  # Kept for compatibility, but not actually used
        self.temperature = temperature  # Temperature hyperparameter for amplifying logits dynamic range
        
        # ========== Explicitly initialize weights to ensure normal Xavier/Kaiming initialization ==========
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters, using Xavier uniform initialization for weights, zeros for bias"""
        nn.init.xavier_uniform_(self.proj_a.weight)
        if self.proj_a.bias is not None:
            nn.init.zeros_(self.proj_a.bias)
        
        nn.init.xavier_uniform_(self.proj_b.weight)
        if self.proj_b.bias is not None:
            nn.init.zeros_(self.proj_b.bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor, return_attn_weights: bool = False, 
                debug: bool = False, layer_name: str = "") -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        a_seq, meta_a = _to_seq(a)
        b_seq, meta_b = _to_seq(b)
        
        # Project to get V_a and V_b
        V_a = self.proj_a(a_seq)  # (B, L_a, d_model)
        V_b = self.proj_b(b_seq)  # (B, L_b, d_model)
        
        A_aff = torch.matmul(V_a, V_b.transpose(-1, -2)) / math.sqrt(self.d_model)
        
        # ========== Normalize and apply temperature scaling to increase logits dynamic range ==========
        A_aff_mean = A_aff.mean()  # Scalar - global mean
        A_aff_std = A_aff.std() + 1e-8  # Scalar - global std, add small value to avoid division by zero
        A_aff_normalized = (A_aff - A_aff_mean) / A_aff_std  # (B, L_a, L_b) - global normalization
        A_aff_scaled = A_aff_normalized * self.temperature  # (B, L_a, L_b) - apply temperature scaling
        
        # ========== Debug: compute and print statistics of V_a, V_b, A_aff ==========
        if debug:
            print(f"[CrossInteraction {layer_name}] Shared Similarity Matrix:")
            print(f"  V_a shape: {V_a.shape}, mean: {V_a.mean():.6f}, std: {V_a.std():.6f}")
            print(f"  V_b shape: {V_b.shape}, mean: {V_b.mean():.6f}, std: {V_b.std():.6f}")
            print(f"  A_aff (raw) shape: {A_aff.shape}, mean: {A_aff.mean():.6f}, std: {A_aff.std():.6f}")
            print(f"  A_aff (scaled) shape: {A_aff_scaled.shape}, mean: {A_aff_scaled.mean():.6f}, std: {A_aff_scaled.std():.6f} ⭐")
            print(f"  Temperature: {self.temperature}")
            if A_aff_scaled.std() < 1e-6:
                print(f"  ⚠️  WARNING: A_aff_scaled.std() = {A_aff_scaled.std():.6f} is very close to 0! Similarity matrix may have degenerated to constants.")
        
        # A←B direction: Apply row softmax to A_aff_scaled to get S_a, then R_a = S_a @ V_b
        # S_a: (B, L_a, L_b) - row-normalized (attention distribution of each token in A to B)
        S_a = F.softmax(A_aff_scaled, dim=-1)  # Row softmax
        S_a = self.dropout(S_a)
        R_a = torch.matmul(S_a, V_b)  # (B, L_a, L_b) @ (B, L_b, d) -> (B, L_a, d)
        
        # B←A direction: Apply column softmax to A_aff_scaled to get S_b, then R_b = S_b.transpose(-1, -2) @ V_a
        # S_b: (B, L_a, L_b) - column-normalized (attention distribution of each token in B to A)
        S_b = F.softmax(A_aff_scaled, dim=-2)  # Column softmax
        S_b = self.dropout(S_b)
        R_b = torch.matmul(S_b.transpose(-1, -2), V_a)
        
        # ========== Debug: check attention weights ==========
        if debug:
            print(f"  S_a (row softmax) shape: {S_a.shape}, mean: {S_a.mean():.6f}, std: {S_a.std():.6f} ⭐")
            print(f"  S_b (col softmax) shape: {S_b.shape}, mean: {S_b.mean():.6f}, std: {S_b.std():.6f} ⭐")
            # Check if normalized
            row_sums_a = S_a.sum(dim=-1)  # (B, L_a)
            col_sums_b = S_b.sum(dim=-2)  # (B, L_b)
            print(f"  S_a row_sums: mean={row_sums_a.mean():.6f}, std={row_sums_a.std():.6f} (should be ~1.0)")
            print(f"  S_b col_sums: mean={col_sums_b.mean():.6f}, std={col_sums_b.std():.6f} (should be ~1.0)")
            # Check if all values are identical (uniform distribution)
            unique_vals_a = torch.unique(S_a[0, :, :])  # Take first sample
            unique_vals_b = torch.unique(S_b[0, :, :])  # Take first sample
            if len(unique_vals_a) == 1:
                print(f"  ⚠️  WARNING: All S_a values are identical! Value: {unique_vals_a[0]:.6f}")
                expected_uniform = 1.0 / S_a.shape[-1]
                if abs(unique_vals_a[0].item() - expected_uniform) < 1e-6:
                    print(f"  ⚠️  This is uniform attention (1/N = {expected_uniform:.6f})!")
            else:
                print(f"  S_a unique values count: {len(unique_vals_a)} (good, not uniform)")
            if len(unique_vals_b) == 1:
                print(f"  ⚠️  WARNING: All S_b values are identical! Value: {unique_vals_b[0]:.6f}")
                expected_uniform = 1.0 / S_b.shape[-2]
                if abs(unique_vals_b[0].item() - expected_uniform) < 1e-6:
                    print(f"  ⚠️  This is uniform attention (1/N = {expected_uniform:.6f})!")
            else:
                print(f"  S_b unique values count: {len(unique_vals_b)} (good, not uniform)")
        
        # Residual and normalization
        a_out = self.ln_a(a_seq + R_a)
        b_out = self.ln_b(b_seq + R_b)
        
        meta = {"a_seq": a_out, "b_seq": b_out}
        if return_attn_weights:
            # Return attention weights for interface compatibility
            meta["attn_weights_a2b"] = S_a.unsqueeze(1)  # (B, 1, L_a, L_b)
            meta["attn_weights_b2a"] = S_b.transpose(-1, -2).unsqueeze(1)  # (B, 1, L_b, L_a)
        
        return _from_seq(a_out, meta_a), _from_seq(b_out, meta_b), meta


class MSREModule(nn.Module):
    """
    MSRE (Modality-Specific Reliability Estimator) module:
    1. Structure-specific classifier f: outputs classification logits
    2. Reliability network Conf: outputs reliability estimates
    3. MSRE computation: get Modality-Specific Reliability Estimator from Conf
    4. Feature weighting: weight features using MSRE
    """
    def __init__(self, d_model: int, num_class: int = 2, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_class = num_class
        
        # Global pooling (for getting sample-level representation from token/ROI-level features)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool over sequence length dimension
        
        # Structure-specific classifier f: input pooled features, output classification logits
        self.structure_classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class)
        )
        
        # Confidence network Conf: input pooled features, output confidence
        # Output can be per-class confidence vector (num_class dim) or scalar (1 dim)
        # Here we use per-class confidence vector form, more flexible
        self.conf_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class)  # Output per-class confidence logits
        )
        
    def forward(self, feat_seq: torch.Tensor):
        """
        Args:
            feat_seq: (B, L, D) sequence features (features after cross-attention)
        Returns:
            logits_f: (B, num_class) logits from structure-specific classifier
            msre: (B, num_class) Modality-Specific Reliability Estimator (normalized to [0,1] via softmax)
            conf_logits: (B, num_class) raw logits from Conf network (for loss computation)
            pooled_feat: (B, D) pooled global features
        """
        B, L, D = feat_seq.shape
        
        # Global pooling: (B, L, D) -> (B, D)
        # First transpose to (B, D, L) to use AdaptiveAvgPool1d
        feat_transposed = feat_seq.transpose(1, 2)  # (B, D, L)
        pooled_feat = self.pool(feat_transposed).squeeze(-1)  # (B, D)
        
        # Structure-specific classifier f
        logits_f = self.structure_classifier(pooled_feat)  # (B, num_class)
        
        # Confidence network Conf
        conf_logits = self.conf_net(pooled_feat)  # (B, num_class)
        
        # MSRE computation: apply softmax to conf_logits to get Modality-Specific Reliability Estimator
        # prob_hat = F.softmax(logits_f, dim=1)  # Original softmax probability (comment: for comparison)
        msre = F.softmax(conf_logits, dim=1)  # MSRE: Modality-Specific Reliability Estimator, range [0,1]
        
        return logits_f, msre, conf_logits, pooled_feat


class BCMFOnly(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.0,
                 residual: bool = True, fused_from_post_residual: bool = False,
                 use_ln_on_fused: bool = False, use_msre: bool = False, num_class: int = 2,
                 temperature: float = 5.0):
        super().__init__()
        self.cross = CrossInteraction(d_model, nhead, dropout, temperature=temperature)
        self.residual = residual
        self.fused_from_post_residual = fused_from_post_residual
        self.proj_fused = nn.Linear(2 * d_model, d_model)
        self.ln_a = nn.LayerNorm(d_model)
        self.ln_b = nn.LayerNorm(d_model)
        self.ln_fused = nn.LayerNorm(d_model) if use_ln_on_fused else nn.Identity()
        
        # MSRE module (optional)
        self.use_msre = use_msre
        if use_msre:
            self.msre_a = MSREModule(d_model, num_class=num_class, dropout=dropout)
            self.msre_b = MSREModule(d_model, num_class=num_class, dropout=dropout)

    def _make_fused(self, a_feat_seq: torch.Tensor, b_feat_seq: torch.Tensor, restore_meta: dict):
        fused = torch.cat([a_feat_seq, b_feat_seq], dim=-1)  # (B, L, 2D)
        fused = self.proj_fused(fused)                       # (B, L, D)
        fused = self.ln_fused(fused)
        return _from_seq(fused, restore_meta)

    def forward(self, a: torch.Tensor, b: torch.Tensor, return_attn_weights: bool = False, 
                debug: bool = False, layer_name: str = ""):
        a0, b0 = a, b
        a1, b1, cross_meta = self.cross(a, b, return_attn_weights=return_attn_weights, 
                                         debug=debug, layer_name=layer_name)

        # ========== MSRE module (after cross-attention, before fusion) ==========
        msre_info = None
        if self.use_msre:
            a1_seq, ma1 = _to_seq(a1)
            b1_seq, mb1 = _to_seq(b1)
            
            # Compute MSRE for each modality
            logits_f_a, msre_a, conf_logits_a, pooled_a = self.msre_a(a1_seq)
            logits_f_b, msre_b, conf_logits_b, pooled_b = self.msre_b(b1_seq)
            
            # Weight features using MSRE
            msre_weight_a = msre_a.max(dim=1, keepdim=True)[0].unsqueeze(1)  # (B, 1, 1)
            msre_weight_b = msre_b.max(dim=1, keepdim=True)[0].unsqueeze(1)  # (B, 1, 1)
            
            # Weight features: a1_weighted = a1 * msre_weight
            a1_weighted_seq = a1_seq * msre_weight_a  # (B, L, D)
            b1_weighted_seq = b1_seq * msre_weight_b  # (B, L, D)
            
            # Restore to original layout
            a1_weighted = _from_seq(a1_weighted_seq, ma1)
            b1_weighted = _from_seq(b1_weighted_seq, mb1)
            
            # Save MSRE info for loss computation
            msre_info = {
                "logits_f_a": logits_f_a,
                "logits_f_b": logits_f_b,
                "msre_a": msre_a,
                "msre_b": msre_b,
                "conf_logits_a": conf_logits_a,
                "conf_logits_b": conf_logits_b,
                "msre_weight_a": msre_weight_a.squeeze(),
                "msre_weight_b": msre_weight_b.squeeze(),
            }
            
            # Use weighted features to continue subsequent flow
            a1, b1 = a1_weighted, b1_weighted

        # ---- pre-residual fused ----
        a1_seq, ma1 = _to_seq(a1)
        b1_seq, mb1 = _to_seq(b1)
        fused_pre = self._make_fused(a1_seq, b1_seq, ma1)

        # ---- residual to a/b (does not affect fused_pre) ----
        if self.residual:
            a1r_seq, ma = _to_seq(a1); a0_seq, _ = _to_seq(a0)
            b1r_seq, mb = _to_seq(b1); b0_seq, _ = _to_seq(b0)
            a1r_seq = self.ln_a(a1r_seq + a0_seq)
            b1r_seq = self.ln_b(b1r_seq + b0_seq)
            a_out = _from_seq(a1r_seq, ma)
            b_out = _from_seq(b1r_seq, mb)
        else:
            a_out, b_out = a1, b1

        # ---- Optional post-residual fused ----
        if self.fused_from_post_residual:
            a_post_seq, ma_post = _to_seq(a_out)
            b_post_seq, _      = _to_seq(b_out)
            fused = self._make_fused(a_post_seq, b_post_seq, ma_post)
        else:
            fused = fused_pre

        # Build return result
        if self.use_msre:
            result = (a_out, b_out, fused, msre_info)
        else:
            result = (a_out, b_out, fused)
        
        # If attention weights requested, add them to result
        if return_attn_weights and "attn_weights_a2b" in cross_meta:
            return (*result, cross_meta)
        
        return result


# ---- Convenience wrapper modules for easy drop-in ----

class BCMFBlock(nn.Module):
    """
    Drop-in replacement for a 'CAT' style fusion box in your diagrams.
    Expected call: a_out, b_out, fused = block(a_in, b_in)
    Shapes:
      - Grid: a_in (B,C,H,W), b_in (B,C,H,W) -> fused (B,C,H,W)
      - Graph/Seq: a_in (B,N,D), b_in (B,N,D) -> fused (B,N,D)
    """
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, dropout: float = 0.0,
                 use_cross_se: bool = True, residual: bool = True, use_msre: bool = False, num_class: int = 2,
                 temperature: float = 5.0):
        super().__init__()
        # Note: num_layers and use_cross_se parameters are kept for interface compatibility but not used
        self.fuse = BCMFOnly(d_model, nhead, dropout, residual=residual,
                                  fused_from_post_residual=False, use_ln_on_fused=True,
                                  use_msre=use_msre, num_class=num_class, temperature=temperature)
        self.use_msre = use_msre

    def forward(self, a: torch.Tensor, b: torch.Tensor, return_attn_weights: bool = False,
                debug: bool = False, layer_name: str = ""):
        if self.use_msre:
            fuse_result = self.fuse(a, b, return_attn_weights=return_attn_weights, 
                                    debug=debug, layer_name=layer_name)
            if return_attn_weights and len(fuse_result) > 4:
                a_out, b_out, fused, msre_info, attn_meta = fuse_result
                return a_out, b_out, fused, msre_info, attn_meta
            else:
                a_out, b_out, fused, msre_info = fuse_result
                return a_out, b_out, fused, msre_info
        else:
            fuse_result = self.fuse(a, b, return_attn_weights=return_attn_weights,
                                     debug=debug, layer_name=layer_name)
            if return_attn_weights and len(fuse_result) > 3:
                a_out, b_out, fused, attn_meta = fuse_result
                return a_out, b_out, fused, attn_meta
            else:
                a_out, b_out, fused = fuse_result
                return a_out, b_out, fused
