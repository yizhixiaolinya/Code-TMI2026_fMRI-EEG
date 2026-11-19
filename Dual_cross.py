"""Bidirectional multi-head cross-attention with channel-specific residuals."""
import torch.nn as nn


class CrossInteraction(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.0):
        super().__init__()
        self.a_from_b = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.b_from_a = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ln_a = nn.LayerNorm(d_model)
        self.ln_b = nn.LayerNorm(d_model)

    def forward(self, a, b):
        # Inputs retain channels: [batch, channels, ROIs, hidden_dim].
        a_mean, b_mean = a.mean(dim=1), b.mean(dim=1)
        a_context, _ = self.a_from_b(a_mean, b_mean, b_mean, need_weights=False)
        b_context, _ = self.b_from_a(b_mean, a_mean, a_mean, need_weights=False)
        return (self.ln_a(a + a_context.unsqueeze(1)),
                self.ln_b(b + b_context.unsqueeze(1)))
