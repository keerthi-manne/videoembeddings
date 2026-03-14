"""
scripts/demo_captioned_events.py  –  Team B (Phase 2)

Full end-to-end pipeline:
  .pt file → STAdapter → segment → CaptionDecoder → timeline → save JSONL

This is the Phase 2 equivalent of demo_events.py.
Instead of printing event0/event1 labels, it now prints real text captions.

Run:
    # Single video
    python scripts/demo_captioned_events.py --video_id video1000

    # All videos in embeddings folder
    python scripts/demo_captioned_events.py --all

    # Quick test on 5 videos
    python scripts/demo_captioned_events.py --all --limit 5

Output console:
    Video: video1000
    ─────────────────────────────────────
    [0.0s – 8.0s]   a person cooking food in a kitchen  (0.82)
    [8.0s – 16.0s]  someone chopping vegetables          (0.78)
    ─────────────────────────────────────

Output file: video_event_captions.jsonl
"""

import torch
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.st_adapter      import STAdapter
from models.event_head      import segment_features
from models.caption_decoder import (
    CaptionDecoder,
    generate_captions_for_video,
    merge_caption_segments
)



# ─────────────────────────────────────────
# Core pipeline function
# ─────────────────────────────────────────

def process_video_to_captions(
    feat: torch.Tensor,
    st_adapter: STAdapter,
    decoder: CaptionDecoder,
    seg_len: int = 8,
    fps: float   = 1.0
) -> list:
    """
    Full Phase 2 pipeline for ONE video.

    Steps:
      1. STAdapter: add temporal context to frame features
      2. segment_features: group into chunks
      3. CaptionDecoder: get text caption per chunk
      4. merge_caption_segments: collapse repeated captions
      5. Return clean timeline

    Args:
        feat: (T, 512) CLIP feature tensor for the video
        st_adapter: trained STAdapter (adds temporal context)
        decoder: CaptionDecoder (CLIP text similarity)
        seg_len: frames per segment
        fps: frames per second used during extraction

    Returns:
        List of {"start", "end", "caption", "confidence"}
    """
    T, D = feat.shape

    # Step 1: STAdapter — enrich features with temporal context
    # (1, T, D) → (1, T, D)  same shape but each frame now "knows" its neighbors
    feat_batched = feat.unsqueeze(0)          # add batch dim
    with torch.no_grad():
        feat_enriched = st_adapter(feat_batched)  # (1, T, 512)
    feat_enriched = feat_enriched.squeeze(0)      # (T, 512)

    # Step 2 & 3: Segment + caption in one call
    # (uses generate_captions_for_video from caption_decoder.py)
    segments = generate_captions_for_video(
        feat_enriched, decoder, seg_len=seg_len, fps=fps
    )

    # Step 4: Merge consecutive segments with the same caption
    timeline = merge_caption_segments(segments)

    return timeline


def print_timeline(video_id: str, timeline: list):
    """Pretty-print the event timeline to console."""
    print(f"\n📹 Video: {video_id}")
    print("─" * 55)
    if not timeline:
        print("  (no events detected)")
    for event in timeline:
        start   = event["start"]
        end     = event["end"]
        caption = event["caption"]
        conf    = event["confidence"]
        print(f"  [{start}s – {end}s]  {caption}  ({conf})")
    print("─" * 55)


# ─────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────

def run(args):
    config_path  = args.config
    feature_dir  = args.feature_dir
    output_file  = args.output
    seg_len      = args.seg_len
    fps          = args.fps
    checkpoint   = args.checkpoint

    # Load STAdapter
    print("⚙️  Loading STAdapter...")
    st_adapter = STAdapter(config_path=config_path)
    if os.path.exists(checkpoint):
        full_sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # Checkpoint was saved from STClassifier which has keys like:
        #   "st_adapter.down_proj.weight", "classifier.weight" ...
        # We only want the "st_adapter.*" keys, stripped of the prefix
        adapter_sd = {
            k.replace("st_adapter.", "", 1): v
            for k, v in full_sd.items()
            if k.startswith("st_adapter.")
        }
        if adapter_sd:
            st_adapter.load_state_dict(adapter_sd)
            print(f"   ✅ Loaded STAdapter weights from {checkpoint}")
        else:
            # Checkpoint might already be a bare STAdapter checkpoint
            st_adapter.load_state_dict(full_sd)
            print(f"   ✅ Loaded weights from {checkpoint}")
    else:
        print(f"   ⚠️  No checkpoint at '{checkpoint}', using random weights")
    st_adapter.eval()


    # Load CaptionDecoder (encodes candidate captions via CLIP)
    print("🔤 Loading CaptionDecoder...")
    decoder = CaptionDecoder()

    # Decide which videos to process
    if args.video_id:
        # Single video mode
        pt_path = os.path.join(feature_dir, f"{args.video_id}.pt")
        if not os.path.exists(pt_path):
            print(f"❌ File not found: {pt_path}")
            return
        video_files = [f"{args.video_id}.pt"]
    else:
        # All videos mode
        video_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.pt')])
        if args.limit:
            video_files = video_files[:args.limit]

    print(f"\n🎬 Processing {len(video_files)} video(s)...\n")

    all_results = []

    for fname in video_files:
        video_id = fname.replace('.pt', '')
        pt_path  = os.path.join(feature_dir, fname)

        # Load features
        feat = torch.load(pt_path, weights_only=False)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)          # (512,) → (1, 512)

        # Run full pipeline
        timeline = process_video_to_captions(
            feat, st_adapter, decoder, seg_len=seg_len, fps=fps
        )

        # Print to console
        print_timeline(video_id, timeline)

        # Collect for JSONL output
        all_results.append({
            "video_id": video_id,
            "timeline": [
                {
                    "start":      f"{e['start']}s",
                    "end":        f"{e['end']}s",
                    "caption":    e["caption"],
                    "confidence": e["confidence"]
                }
                for e in timeline
            ]
        })

    # Save to JSONL
    with open(output_file, 'w') as f:
        for entry in all_results:
            f.write(json.dumps(entry) + '\n')

    print(f"\n✅ Done! Captioned timelines saved → {output_file}")
    print(f"   {len(all_results)} videos processed\n")

    # Show one full example entry
    if all_results:
        print("📄 Sample JSONL entry:")
        print(json.dumps(all_results[0], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 demo: video embeddings → captioned event timeline"
    )
    parser.add_argument("--video_id",    default=None,
                        help="Single video ID (e.g. video1000). Omit to run all.")
    parser.add_argument("--all",         action="store_true",
                        help="Process all videos in feature_dir")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Max videos to process (for quick testing)")
    parser.add_argument("--feature_dir", default="embeddings1",
                        help="Folder with .pt files")
    parser.add_argument("--checkpoint",  default="checkpoints/st_adapter.pt",
                        help="Trained STAdapter weights")
    parser.add_argument("--config",      default="configs/model_base.json")
    parser.add_argument("--output",      default="video_event_captions.jsonl")
    parser.add_argument("--seg_len",     type=int,   default=8)
    parser.add_argument("--fps",         type=float, default=1.0)
    args = parser.parse_args()

    # Default to single video1000 if nothing specified
    if not args.video_id and not args.all:
        args.video_id = "video1000"

    run(args)
