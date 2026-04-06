# ==========================================
# 🔷 EVALUATION SCRIPT (evaluate.py)
# ==========================================

import torch
from tqdm import tqdm

from dataloader.didemo_loader import DiDeMoDataset
from torch.utils.data import DataLoader

from dataloader.query_split import QuerySplit
from model.mmt import MMT
from model.cross_attn import CrossAttention
from model.bid_attn import BiDAttention
from model.region_predictor import RegionPredictor
from model.region_embedding import RegionEmbedding
from model.bigru import BiGRULocalizer


# -------------------------------
# IoU function
# -------------------------------
def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
    union = max(pe, ge) - min(ps, gs) + 1
    return inter / union if union > 0 else 0

def collate_fn(batch):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    video_feat = torch.stack([b["video_feat"] for b in batch])

    query_feat = pad_sequence(
        [b["query_feat"] for b in batch],
        batch_first=True
    )

    caption_feat = pad_sequence(
        [b["caption_feat"] for b in batch],
        batch_first=True
    )

    start = torch.tensor([b["start"] for b in batch], dtype=torch.long)
    end   = torch.tensor([b["end"] for b in batch], dtype=torch.long)

    video_id = [b["video_id"] for b in batch]

    return {
        "video_feat": video_feat,
        "query_feat": query_feat,
        "caption_feat": caption_feat,
        "start": start,
        "end": end,
        "video_id": video_id
    }
# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model(checkpoint_path, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Load Test Dataset
    # -------------------------------
    test_dataset = DiDeMoDataset("test")
    test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
    )
    # -------------------------------
    # Initialize Model
    # -------------------------------
    query_split = QuerySplit(384).to(device)
    mmt = MMT(384, 4, 1536, 1).to(device)
    cross_attn = CrossAttention(384).to(device)
    bid_attn = BiDAttention(384, 4).to(device)
    region_pred = RegionPredictor(384).to(device)
    region_embed = RegionEmbedding(384).to(device)
    bigru = BiGRULocalizer().to(device)

    modules = {
        "QuerySplit": query_split,
        "MMT": mmt,
        "CrossAttention": cross_attn,
        "BiDAttention": bid_attn,
        "RegionPredictor": region_pred,
        "RegionEmbedding": region_embed,
        "BiGRULocalizer": bigru
    }

    # -------------------------------
    # Load checkpoint
    # -------------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device)

    for name, module in modules.items():
        module.load_state_dict(checkpoint["model_state"][name])
        module.eval()

    # -------------------------------
    # Evaluation
    # -------------------------------
    total = 0
    correct_05 = 0
    correct_07 = 0
    fact=10

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):

            video_feat = batch["video_feat"].to(device)
            query_feat = batch["query_feat"].to(device)
            caption_feat = batch["caption_feat"].to(device)

            start_gt = batch["start"].to(device)
            end_gt = batch["end"].to(device)

            B, N, _ = video_feat.shape

            # Forward pass
            Q_f, Q_s = query_split(query_feat)

            F_context = mmt(video_feat)
            S_context = mmt(caption_feat)

            F_prime = cross_attn(F_context, Q_f)
            S_prime = cross_attn(S_context, Q_s)

            C = bid_attn(F_prime, S_prime, query_feat)

            P_reg, _ = region_pred(C, use_gumbel=False)
            C_prime = region_embed(C, P_reg)

            start_logits, end_logits = bigru(C_prime)

            # Slice to video clips only
            start_logits = start_logits[:, :N]
            end_logits = end_logits[:, :N]

            pred_s = torch.argmax(start_logits, dim=1)
            pred_e = torch.argmax(end_logits, dim=1)

            # Compute IoU
            for i in range(B):
                iou = compute_iou(
                    pred_s[i].item(),
                    pred_e[i].item(),
                    start_gt[i].item(),
                    end_gt[i].item()
                )

                if iou >= 0.5:
                    correct_05 += 1
                if iou >= 0.7:
                    correct_07 += 1

                total += 1

    # -------------------------------
    # Results
    # -------------------------------
    r1_05 = correct_05*fact / total
    r1_07 = correct_07*fact / total

    return {
        "R@1 IoU=0.5": r1_05,
        "R@1 IoU=0.7": r1_07
    }