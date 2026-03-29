"""
scripts/build_retrieval_index.py – Phase 3 (Steps 1-6)

This script performs the full indexing process:
1. Loads all .pt embeddings from the feature directory
2. Runs them through STAdapter to get temporal features (Shape Tx512)
3. Computes heuristic event segments (splits video when consecutive frames have distance > threshold)
4. Computes segment features by averaging frame features within each segment
5. Generates a text caption for each segment using CaptionDecoder
6. Saves events to `video_event_captions.jsonl` AND a searchable PyTorch index `retrieval_index.pt`

Run:
    python scripts/build_retrieval_index.py --feature_dir embeddings1
"""

import torch
import torch.nn.functional as F
import os
import sys
import json
import argparse
from tqdm import tqdm
from Extract_Frame import run_Caption_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.st_adapter       import STAdapter
from models.caption_decoder  import CaptionDecoder


def segment_by_heuristics(feat_enriched: torch.Tensor, threshold: float = 0.4) -> list:
    """
    Step 3 - Event Segmentation based on visual changes.
    Computes cosine distance between consecutive frames. If the distance > threshold,
    we split the video into a new segment.
    """
    T = feat_enriched.shape[0]
    segments = []
    
    # If the video is extremely short (1 frame), return it as one segment
    if T <= 1:
        return [(0, max(1, T) - 1)]

    # Calculate cosine similarity between consecutive frames
    # feats[:-1] are frames 0 to T-2, feats[1:] are frames 1 to T-1
    f1 = F.normalize(feat_enriched[:-1], dim=-1)
    f2 = F.normalize(feat_enriched[1:], dim=-1)
    
    # Cosine distance = 1 - cosine similarity
    distances = 1.0 - (f1 * f2).sum(dim=-1)
    
    # Find boundary indices where distance > threshold
    boundaries = (distances > threshold).nonzero(as_tuple=True)[0].tolist()
    
    start_idx = 0
    for b in boundaries:
        segments.append((start_idx, b))
        start_idx = b + 1
    
    # Add the final segment
    if start_idx < T:
        segments.append((start_idx, T - 1))
        
    return segments

def build_index(args):
    feature_dir = args.feature_dir
    config_path = args.config
    checkpoint  = args.checkpoint
    threshold   = args.threshold
    fps         = args.fps

    print("⚙️  Loading STAdapter...")
    st_adapter = STAdapter(config_path=config_path)
    if os.path.exists(checkpoint):
        full_sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        adapter_sd = {
            k.replace("st_adapter.", "", 1): v
            for k, v in full_sd.items() if k.startswith("st_adapter.")
        }
        if adapter_sd:
            st_adapter.load_state_dict(adapter_sd)
        else:
            st_adapter.load_state_dict(full_sd)
        print(f"   ✅ Loaded weights from {checkpoint}")
    else:
        print(f"   ⚠️  Using random STAdapter weights (No checkpoint found)")
    st_adapter.eval()

    print("🔤 Loading CaptionDecoder...")
    decoder = CaptionDecoder(device="cpu" if not torch.cuda.is_available() else "cuda")

    video_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.pt')])
    if args.limit:
        video_files = video_files[:args.limit]

    print(f"\n🎬 Indexing {len(video_files)} videos...\n")

    jsonl_results = []
    retrieval_index = []  # List of dicts representing the database

    for fname in tqdm(video_files, desc="Processing Videos"):
        video_id = fname.replace('.pt', '')
        pt_path  = os.path.join(feature_dir, fname)

        # Step 1: Load Video Embeddings
        feat = torch.load(pt_path, weights_only=False)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)  # (1, 512)
            
        T, D = feat.shape
        if D != 512:
            continue

        # Step 2: Temporal Modeling
        feat_batched = feat.unsqueeze(0)
        with torch.no_grad():
            feat_enriched = st_adapter(feat_batched).squeeze(0)  # (T, 512)

        # Step 3: Event Segmentation (Heuristic)
        frame_segments = segment_by_heuristics(feat_enriched, threshold=threshold)
        
        timeline = []
        
        for (start_f, end_f) in frame_segments:
            # Calculate segment feature by averaging frames in segment
            seg_feat = feat_enriched[start_f:end_f+1].mean(dim=0)  # (512,)
            
            # # Step 4: Caption Generation
            # caption, conf = decoder.decode(seg_feat)
            
            # Step 5 & 6 Info Prep
            start_time = round(start_f / fps, 2)
            end_time   = round((end_f + 1) / fps, 2) # end_time is exclusive edge
            
            timeline.append({
                "start":      f"{start_time}s",
                "end":        f"{end_time}s"
                # "caption":    caption,
                # "confidence": conf
            })
            
            # Add to search index database
            retrieval_index.append({
                "video_id":    video_id,
                "start_time":  start_time,
                "end_time":    end_time,
                "segment_emb": seg_feat.cpu() # 512-dim tensor
            })

        jsonl_results.append({
            "video_id": video_id,
            "timeline": timeline
        })

    # Save JSONL Output (Step 5)
    os.makedirs('data', exist_ok=True)
    with open("data/video_event_captions.jsonl", 'w') as f:
        for entry in jsonl_results:
            f.write(json.dumps(entry) + '\n')
            
    # Save Feature Index for Search (Step 6)
    torch.save(retrieval_index, "data/retrieval_index.pt")
    # Get the captions and save it (Step 7)
    print("starting caption pipeline")
    run_Caption_pipeline()
    print(f"\n✅ Done! Captioned timelines saved → data/video_event_captions.jsonl")
    print(f"✅ Fast search index saved → data/retrieval_index.pt")
    print(f"   Indexed {len(retrieval_index)} total segment events across {len(jsonl_results)} videos.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", default="embeddings1")
    parser.add_argument("--checkpoint",  default="checkpoints/st_adapter.pt")
    parser.add_argument("--config",      default="configs/model_base.json")
    parser.add_argument("--limit",       type=int, default=None)
    parser.add_argument("--threshold",   type=float, default=0.2, help="Cosine distance threshold for scene split")
    parser.add_argument("--fps",         type=float, default=1.0)
    args = parser.parse_args()
    build_index(args)
