# ==========================================
# 🔷 REGION EMBEDDING MODULE
# ==========================================

import torch
import torch.nn as nn


class RegionEmbedding(nn.Module):
    def __init__(self, dim=384):
        super().__init__()

        # Learnable region embeddings: (2, 384)
        self.E_reg = nn.Parameter(torch.randn(2, dim))

    def forward(self, C, P_reg):
        """
        C      : (B, T, 384)
        P_reg  : (B, T, 2)

        returns:
        C_prime : (B, T, 384)
        """

        # -------------------------------
        # 1. Compute region info
        # -------------------------------
        # (B, T, 2) × (2, 384) → (B, T, 384)
        Region_info = torch.matmul(P_reg, self.E_reg)

        # -------------------------------
        # 2. Add to context
        # -------------------------------
        C_prime = C + Region_info

        return C_prime