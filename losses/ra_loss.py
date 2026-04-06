# ==========================================
# 🔷 REGION-AWARE LOSS (L_ra)
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionAwareLoss(nn.Module):
    def __init__(self, lambda_orth=0.1):
        super().__init__()

        self.lambda_orth = lambda_orth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, P_reg_logits, y_reg, E_reg):
        """
        P_reg_logits : (B, T, 2)  ← BEFORE softmax
        y_reg        : (B, T)     ← {0,1}
        E_reg        : (2, 384)

        returns:
        L_ra : scalar
        """

        B, T, _ = P_reg_logits.shape

        # ----------------------------------------
        # 1. Region Classification Loss (L_reg)
        # ----------------------------------------

        # reshape for CE
        logits = P_reg_logits.reshape(B * T, 2)   # (B*T, 2)
        targets = y_reg.reshape(B * T)            # (B*T)

        L_reg = self.ce_loss(logits, targets)

        # ----------------------------------------
        # 2. Orthogonality Loss (L_orth)
        # ----------------------------------------

        # normalize embeddings
        E_norm = F.normalize(E_reg, p=2, dim=1)   # (2, 384)

        # similarity matrix
        S = torch.matmul(E_norm, E_norm.t())      # (2, 2)

        # identity matrix
        I = torch.eye(2, device=E_reg.device)

        # Frobenius norm (MSE)
        L_orth = torch.mean((S - I) ** 2)

        # ----------------------------------------
        # 3. Final Loss
        # ----------------------------------------
        L_ra = L_reg + self.lambda_orth * L_orth

        return L_ra, L_reg, L_orth