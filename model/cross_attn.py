# ==========================================
# 🔷 PAPER-ALIGNED CROSS ATTENTION (F_context + Q_f)
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim=384):
        super().__init__()

        # Linear projection W (NO bias)
        self.W = nn.Linear(dim, dim, bias=False)

    def forward(self, F_context, Q_f):
        """
        F_context : (B, N, 384)
        Q_f       : (B, Lq, 384)

        returns   : (B, N, 384)
        """

        # -------------------------------
        # 1. Mean pool query
        # -------------------------------
        Q_global = Q_f.mean(dim=1)              # (B, 384)

        # -------------------------------
        # 2. Linear transform video features
        # -------------------------------
        F_proj = self.W(F_context)              # (B, N, 384)

        # -------------------------------
        # 3. Dot product similarity
        # -------------------------------
        Q_global = Q_global.unsqueeze(-1)       # (B, 384, 1)
        sim = torch.bmm(F_proj, Q_global)       # (B, N, 1)
        sim = sim.squeeze(-1)                   # (B, N)

        # -------------------------------
        # 4. L2 normalization (across N)
        # -------------------------------
        sim_norm = F.normalize(sim, p=2, dim=1) # (B, N)

        # -------------------------------
        # 5. Expand + reweight features
        # -------------------------------
        sim_norm = sim_norm.unsqueeze(-1)       # (B, N, 1)
        F_prime = F_proj * sim_norm             # (B, N, 384)

        return F_prime