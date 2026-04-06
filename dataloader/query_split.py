import torch
import torch.nn as nn


class QuerySplit(nn.Module):
    """
    Implements query splitting:
        Q_f = W_f Q
        Q_s = W_s Q

    where W_f, W_s are learnable projection matrices.
    """

    def __init__(self, dim=384):
        super().__init__()

        print("\n Initializing QuerySplit module...")

        # Paper-consistent: linear projection WITHOUT bias
        self.qf_proj = nn.Linear(dim, dim, bias=False)
        self.qs_proj = nn.Linear(dim, dim, bias=False)

        print(" W_f and W_s initialized (Linear layers)")
        print("W_f shape:", self.qf_proj.weight.shape)
        print("W_s shape:", self.qs_proj.weight.shape)

    def forward(self, Q):
        """
        Q: (B, L, 384)
        Returns:
            Q_f: (B, L, 384)
            Q_s: (B, L, 384)
        """

        

        # Apply projections
        Q_f = self.qf_proj(Q)
        Q_s = self.qs_proj(Q)

        
        # Optional debug: check difference
        diff = torch.mean(torch.abs(Q_f - Q_s)).item()
        # print("Avg difference between Q_f and Q_s:", diff)

        return Q_f, Q_s