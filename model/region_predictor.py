# ==========================================
# 🔷 REGION PREDICTOR (Progressive Localization Entry)
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionPredictor(nn.Module):
    def __init__(self, dim=384, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        # Conv1: 384 → 384
        self.conv1 = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=padding
        )

        # Conv2: 384 → 2 (background / foreground)
        self.conv2 = nn.Conv1d(
            in_channels=dim,
            out_channels=2,
            kernel_size=kernel_size,
            padding=padding
        )

        self.relu = nn.ReLU()

    def forward(self, C, use_gumbel=False, tau=1.0):
        """
        C : (B, T, 384)

        returns:
        probs  : (B, T, 2)
        logits : (B, T, 2)
        """

        # -------------------------------
        # 1. Convert to (B, 384, T)
        # -------------------------------
        x = C.permute(0, 2, 1)

        # -------------------------------
        # 2. Conv + ReLU
        # -------------------------------
        x = self.conv1(x)
        x = self.relu(x)

        # -------------------------------
        # 3. Conv to logits
        # -------------------------------
        logits = self.conv2(x)               # (B, 2, T)

        # -------------------------------
        # 4. Back to (B, T, 2)
        # -------------------------------
        logits = logits.permute(0, 2, 1)

        # -------------------------------
        # 5. Softmax / Gumbel Softmax
        # -------------------------------
        if use_gumbel:
            probs = F.gumbel_softmax(logits, tau=tau, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)

        return probs, logits