import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticContrastiveLoss(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.temperature = 0.07

    def forward(self, Q, F_pos, F_neg, S_pos, S_neg):

        # ----------------------------------------
        # 1. Global query representation
        # ----------------------------------------
        Q_global = Q.mean(dim=1)

        # ----------------------------------------
        # 2. Normalize
        # ----------------------------------------
        Q_global = F.normalize(Q_global, dim=-1)

        F_pos = F.normalize(F_pos, dim=-1)
        F_neg = F.normalize(F_neg, dim=-1)

        S_pos = F.normalize(S_pos, dim=-1)
        S_neg = F.normalize(S_neg, dim=-1)

        # ----------------------------------------
        # 3. Expand query
        # ----------------------------------------
        Q_exp_f = Q_global.unsqueeze(1)
        Q_exp_s = Q_global.unsqueeze(1)

        # ----------------------------------------
        # 4. Similarity with temperature
        # ----------------------------------------
        r_pos_f = torch.sum(Q_exp_f * F_pos, dim=-1) / self.temperature
        r_neg_f = torch.sum(Q_exp_f * F_neg, dim=-1) / self.temperature

        r_pos_s = torch.sum(Q_exp_s * S_pos, dim=-1) / self.temperature
        r_neg_s = torch.sum(Q_exp_s * S_neg, dim=-1) / self.temperature

        # ----------------------------------------
        # 5. Stable loss (log-softmax)
        # ----------------------------------------
        logits_f = torch.stack([r_pos_f, r_neg_f], dim=-1)
        log_probs_f = F.log_softmax(logits_f, dim=-1)
        L_f = -log_probs_f[..., 0].mean()

        logits_s = torch.stack([r_pos_s, r_neg_s], dim=-1)
        log_probs_s = F.log_softmax(logits_s, dim=-1)
        L_s = -log_probs_s[..., 0].mean()

        return L_f + L_s