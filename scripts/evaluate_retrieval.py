import os
import sys
import torch
import json
import clip
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.st_adapter import STAdapter

class SmallEvalDataset(Dataset):
    def __init__(self, feature_dir, video_ids):
        self.feature_dir = feature_dir
        self.video_ids = video_ids
        
    def __len__(self):
        return len(self.video_ids)
        
    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]
        pt_path = os.path.join(self.feature_dir, f"{vid_id}.pt")
        feat = torch.load(pt_path, weights_only=False)
        if feat.dim() == 1: feat = feat.unsqueeze(0)
        return feat, vid_id

def compute_metrics(sim_matrix):
    """
    sim_matrix: (num_queries, num_videos)
    Assumes query i matches video i (diagonal is ground truth).
    """
    num_queries = sim_matrix.shape[0]
    ranks = []
    
    for i in range(num_queries):
        # Sort scores in descending order
        scores = sim_matrix[i]
        sorted_indices = np.argsort(-scores)
        # Find rank of the correct video (index i)
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)
        
    ranks = np.array(ranks)
    r1 = 100 * np.sum(ranks <= 1) / num_queries
    r5 = 100 * np.sum(ranks <= 5) / num_queries
    r10 = 100 * np.sum(ranks <= 10) / num_queries
    medr = np.median(ranks)
    meanr = np.mean(ranks)
    
    return {"R@1": r1, "R@5": r5, "R@10": r10, "MedR": medr, "MeanR": meanr}

@torch.no_grad()
def evaluate(checkpoint_path, feature_dir, captions_json, limit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Starting Evaluation...")
    print(f"📍 Checkpoint: {checkpoint_path}")
    print(f"📍 Annotations: {captions_json}")
    
    # 1. Load Model
    st_adapter = STAdapter().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle both full state_dict and just weights
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    st_adapter.load_state_dict(state_dict)
    st_adapter.eval()
    
    # 2. Load CLIP
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # 3. Load Mapping
    with open(captions_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Organize by video_id
    id_to_captions = {}
    # Check format: official JSON has list of dicts or dict mapping?
    if isinstance(raw_data, list):
        for item in raw_data:
            vid = item['video'].replace('.mp4', '')
            if vid not in id_to_captions: id_to_captions[vid] = []
            id_to_captions[vid].append(item['caption'])
    else:
        id_to_captions = raw_data # Optimized format we saved earlier
        
    # 4. Filter available features
    local_files = [f.replace('.pt', '') for f in os.listdir(feature_dir) if f.endswith('.pt')]
    valid_ids = [vid for vid in local_files if vid in id_to_captions]
    
    if limit:
        valid_ids = valid_ids[:limit]
        
    print(f"📊 Found {len(valid_ids)} videos with features and annotations.")
    if len(valid_ids) == 0:
        print("❌ No matching videos found for evaluation. Check your feature_dir and JSON.")
        return

    # 5. Extract Video Features
    video_feats = []
    print("🎥 Extracting video temporal embeddings...")
    for vid_id in tqdm(valid_ids):
        pt_path = os.path.join(feature_dir, f"{vid_id}.pt")
        feat = torch.load(pt_path, weights_only=False).to(device).float()
        if feat.dim() == 1: feat = feat.unsqueeze(0)
        
        # Pass through STAdapter
        temp_feat = st_adapter(feat.unsqueeze(0)) # (1, T, 512)
        pooled = temp_feat.mean(dim=1) # (1, 512)
        # L2 Norm
        pooled /= pooled.norm(dim=-1, keepdim=True)
        video_feats.append(pooled)
        
    video_feats = torch.cat(video_feats, dim=0) # (N, 512)
    
    # 6. Extract Text Features
    # For evaluation, we typically match each unique video to its FIRST caption 
    # to create a clear 1-to-1 retrieval task.
    print("📝 Encoding text queries (first caption per video)...")
    text_queries = [id_to_captions[vid][0] for vid in valid_ids]
    tokens = clip.tokenize(text_queries, truncate=True).to(device)
    text_feats = clip_model.encode_text(tokens).float()
    text_feats /= text_feats.norm(dim=-1, keepdim=True) # (N, 512)
    
    # 7. Compute Similarity
    sim_matrix = (text_feats @ video_feats.T).cpu().numpy() # (N, N)
    
    # 8. Metrics
    metrics = compute_metrics(sim_matrix)
    print("\n" + "="*30)
    print("🏆 FINAL RETRIEVAL METRICS")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k:7}: {v:.2f}")
    print("="*30)
    
    # 9. Qualitative Examples
    print("\n👀 Qualitative Examples (Top-3 Retrieval):")
    for i in range(min(3, len(valid_ids))):
        query_text = text_queries[i]
        scores = sim_matrix[i]
        top_indices = np.argsort(-scores)[:3]
        
        print(f"\nQuery: \"{query_text}\"")
        for rank, idx in enumerate(top_indices):
            status = "✅ (Correct)" if idx == i else "❌"
            print(f"  {rank+1}. {valid_ids[idx]} (Score: {scores[idx]:.4f}) {status}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/st_adapter_contrastive.pt")
    parser.add_argument("--feature_dir", default="embeddings1")
    parser.add_argument("--captions_json", default="msrvtt_ret_test1k.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.feature_dir, args.captions_json, args.limit)
