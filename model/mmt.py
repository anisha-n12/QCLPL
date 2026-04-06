# ==========================================
# 🔷 MMT: Multimodal Transformer Encoder
# ==========================================

import torch
import torch.nn as nn


class MMT(nn.Module):
    def __init__(self, dim=384, num_heads=4, ff_dim=1536, num_layers=1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        x: (B, T, 384)
        returns: (B, T, 384)
        """
        return self.encoder(x)