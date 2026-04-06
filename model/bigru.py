# ==========================================
# 🔷 TEMPORAL LOCALIZATION MODULE (BiGRU)
# ==========================================

import torch
import torch.nn as nn


class BiGRULocalizer(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=192):
        super().__init__()

        # BiGRU
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Conv heads (separate for start and end)
        self.start_conv = nn.Conv1d(
            in_channels=hidden_dim * 2,  # 384
            out_channels=1,
            kernel_size=1
        )

        self.end_conv = nn.Conv1d(
            in_channels=hidden_dim * 2,  # 384
            out_channels=1,
            kernel_size=1
        )

    def forward(self, C_prime):
        """
        C_prime : (B, T, 384)

        returns:
        start_logits : (B, T)
        end_logits   : (B, T)
        """

        # -------------------------------
        # 1. BiGRU
        # -------------------------------
        H, _ = self.bigru(C_prime)              # (B, T, 384)

        # -------------------------------
        # 2. Convert to (B, 384, T)
        # -------------------------------
        H = H.permute(0, 2, 1)

        # -------------------------------
        # 3. Conv heads
        # -------------------------------
        start_logits = self.start_conv(H)       # (B, 1, T)
        end_logits   = self.end_conv(H)         # (B, 1, T)

        # -------------------------------
        # 4. Squeeze → (B, T)
        # -------------------------------
        start_logits = start_logits.squeeze(1)
        end_logits   = end_logits.squeeze(1)

        return start_logits, end_logits