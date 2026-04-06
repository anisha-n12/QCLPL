# ==========================================
# 🔷 BIDIRECTIONAL ATTENTION (FINAL CORRECT)
# ==========================================

import torch
import torch.nn as nn


class BiDAttention(nn.Module):
    def __init__(self, dim=384, num_heads=4):
        super().__init__()

        # Fusion after concatenation
        self.fusion_proj = nn.Linear(dim, dim)

        # C → Q attention (PRIMARY, per-position)
        self.c_to_q_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Optional Q → C (kept but NOT broadcast)
        self.q_to_c_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Final projection
        self.out_proj = nn.Linear(dim * 2, dim)

    def forward(self, F_prime, S_prime, Q):
        """
        F_prime : (B, N, 384)
        S_prime : (B, Ls, 384)
        Q       : (B, Lq, 384)

        returns : (B, N + Ls, 384)
        """

        # -------------------------------
        # 1. Concatenate modalities
        # -------------------------------
        C = torch.cat([F_prime, S_prime], dim=1)     # (B, N+Ls, 384)

        # -------------------------------
        # 2. Fusion projection
        # -------------------------------
        C = self.fusion_proj(C)                      # (B, N+Ls, 384)

        # -------------------------------
        # 3. Context → Query attention (MAIN)
        # -------------------------------
        C_query, _ = self.c_to_q_attn(
            query=C,
            key=Q,
            value=Q
        )                                            # (B, N+Ls, 384)

        # -------------------------------
        # 4. Query → Context attention (optional, NOT broadcast)
        # -------------------------------
        Q_context, _ = self.q_to_c_attn(
            query=Q,
            key=C,
            value=C
        )                                            # (B, Lq, 384)

        # NOTE:
        # We DO NOT force Q_context into (B, N+Ls, 384)
        # No mean, no expand, no alignment trick

        # -------------------------------
        # 5. Combine (ONLY position-aware features)
        # -------------------------------
        combined = torch.cat([C, C_query], dim=-1)
        C_final = self.out_proj(combined) + C

        return C_final