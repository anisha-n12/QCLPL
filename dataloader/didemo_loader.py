# ==========================================
# 🔷 COMPLETE CLEAN PIPELINE (NO VERBOSE)
# ==========================================

import os
import json
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from dataloader.query_split import QuerySplit


# ================= NORMALIZATION =================
def normalize_vid(vid):
    vid = os.path.basename(vid)
    vid = os.path.splitext(vid)[0]
    vid = vid.replace("video_", "").replace("vid_", "").replace("v_", "")
    return vid.strip()


# ================= LOAD JSON =================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ================= LOAD JSONL =================
def load_all_jsonl(paths):
    caption_map = {}
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    vid = normalize_vid(item["video"])
                    caption_map[vid] = item["caption"]
    return caption_map


# ================= DATASET =================
class DiDeMoDataset(Dataset):
    def __init__(self, split="train"):
        self.query_path = f"dataset/query/{split}_data.json"
        self.rgb_path = "dataset/features/average_rgb_feats.h5"
        self.flow_path = "dataset/features/average_flow_feats.h5"

        self.data = load_json(self.query_path)

        self.caption_map = load_all_jsonl([
            "dataset/query/train.jsonl",
            "dataset/query/test.jsonl"
        ])

        self.rgb_h5 = h5py.File(self.rgb_path, "r")
        self.flow_h5 = h5py.File(self.flow_path, "r")

        self.rgb_map = {normalize_vid(k): k for k in self.rgb_h5.keys()}
        self.flow_map = {normalize_vid(k): k for k in self.flow_h5.keys()}

        feature_vids = set(self.rgb_map.keys()) & set(self.flow_map.keys())

        self.valid_data = [
            item for item in self.data
            if normalize_vid(item["video"]) in feature_vids
        ]

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.text_model = RobertaModel.from_pretrained("roberta-base")
        self.text_model.eval()

        self.video_proj = torch.nn.Linear(5120, 384)
        self.text_proj = torch.nn.Linear(768, 384)

    def __len__(self):
        return len(self.valid_data)

    def process_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        emb = outputs.last_hidden_state.squeeze(0)
        return self.text_proj(emb)

    def get_video_feat(self, vid):
        rgb = self.rgb_h5[self.rgb_map[vid]][:]
        flow = self.flow_h5[self.flow_map[vid]][:]

        rgb = rgb.astype(np.float32)
        flow = flow.astype(np.float32)

        N = min(rgb.shape[0], flow.shape[0])
        feat = np.concatenate([rgb[:N], flow[:N]], axis=-1)

        feat = torch.tensor(feat)
        return self.video_proj(feat)

    def get_label(self, times):
        s = min([t[0] for t in times])
        e = max([t[1] for t in times])
        return s, e

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        vid = normalize_vid(item["video"])

        video_feat = self.get_video_feat(vid)
        query_feat = self.process_text(item["description"])

        caption_text = self.caption_map.get(vid, item["description"])
        caption_feat = self.process_text(caption_text)

        start, end = self.get_label(item["times"])

        return {
            "video_feat": video_feat,
            "query_feat": query_feat,
            "caption_feat": caption_feat,
            "start": start,
            "end": end,
            "video_id": vid
        }


# ================= MAIN PIPELINE =================

