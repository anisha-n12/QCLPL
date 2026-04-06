# ==========================================
# 🔷 MOMENT LOCALIZATION LOSS (L_ml)
# ==========================================

import torch
import torch.nn as nn


class MomentLocalizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, start_logits, end_logits, start_gt, end_gt, video_length):
        """
        start_logits : (B, T)
        end_logits   : (B, T)

        start_gt     : (B,)   ← index
        end_gt       : (B,)
        
        video_length : int or scalar (N)

        returns:
        L_ml : scalar
        """

        # ----------------------------------------
        # 1. Align to VIDEO positions only
        # ----------------------------------------
        # Keep only first N positions (video clips)
        start_logits_video = start_logits[:, :video_length]  # (B, N)
        end_logits_video   = end_logits[:, :video_length]    # (B, N)

        # ----------------------------------------
        # 2. Compute CrossEntropy losses
        # ----------------------------------------
        L_start = self.ce_loss(start_logits_video, start_gt)
        L_end   = self.ce_loss(end_logits_video, end_gt)

        # ----------------------------------------
        # 3. Final Loss
        # ----------------------------------------
        L_ml = L_start + L_end

        return L_ml, L_start, L_end