"""
scripts/generate_captions.py  –  Team A (Phase 2)

What this does:
  Batch-processes ALL videos in the embeddings folder.
  For each video → segments → captions → saves to captions.jsonl

Run:
    python scripts/generate_captions.py --feature_dir embeddings1
    python scripts/generate_captions.py --feature_dir embeddings1 --limit 20
"""

import torch
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.caption_decoder import CaptionDecoder
from models.event_head import segment_features


def load_feature(path: str) -> torch.Tensor:
    """Load a .pt file, normalize shape to (T, 512)."""
    feat = torch.load(path, weights_only=False)
    if feat.dim() == 1:
        feat = feat.unsqueeze(0)    # (512,) → (1, 512)
    return feat                     # (T, 512)


def generate_captions_for_video(
    feat: torch.Tensor,
    decoder: CaptionDecoder,
    seg_len: int = 8,
    fps: float = 1.0
) -> list:
    """
    Given a video's feature tensor (T, 512):
    1. Group into segments of seg_len frames
    2. For each segment → best matching caption
    3. Return list of {start, end, caption, confidence}
    """
    T = feat.shape[0]

    # If video is too short to form even one full segment,
    # treat the whole video as one segment
    if T < seg_len:
        seg_feat = feat.mean(dim=0)          # (512,)
        caption, conf = decoder.decode(seg_feat)
        start = 0.0
        end   = round(T / fps, 2)
        return [{"start": start, "end": end, "caption": caption, "confidence": conf}]

    # Segment: (1, T, 512) → (1, S, 512)
    feat_batched = feat.unsqueeze(0)
    segments     = segment_features(feat_batched, seg_len=seg_len)  # (1, S, 512)
    segments     = segments.squeeze(0)                               # (S, 512)
    S            = segments.shape[0]

    results = []
    for i in range(S):
        seg_feat          = segments[i]                 # (512,)
        caption, conf     = decoder.decode(seg_feat)
        start_time        = round(i * seg_len / fps, 2)
        end_time          = round((i + 1) * seg_len / fps, 2)
        results.append({
            "start":      start_time,
            "end":        end_time,
            "caption":    caption,
            "confidence": conf
        })

    return results


def merge_caption_segments(segments: list) -> list:
    """
    Merge consecutive segments that got the same caption.
    This collapses repetitive predictions into cleaner events.

    Example:
      [0–8s "cooking", 8–16s "cooking", 16–24s "eating"]
      → [0–16s "cooking", 16–24s "eating"]
    """
    if not segments:
        return []

    merged = []
    cur = dict(segments[0])  # copy first segment

    for seg in segments[1:]:
        if seg["caption"] == cur["caption"]:
            # Same caption → extend the current event
            cur["end"]        = seg["end"]
            cur["confidence"] = round((cur["confidence"] + seg["confidence"]) / 2, 3)
        else:
            merged.append(cur)
            cur = dict(seg)

    merged.append(cur)
    return merged


def run(args):
    feature_dir = args.feature_dir
    output_file = args.output
    seg_len     = args.seg_len
    fps         = args.fps
    limit       = args.limit

    # Load decoder (does CLIP text encoding once)
    decoder = CaptionDecoder()

    # Find all .pt files
    files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.pt')])
    if limit:
        files = files[:limit]

    print(f"\n🎬 Generating captions for {len(files)} videos...\n")

    results = []

    for i, fname in enumerate(files):
        video_id = fname.replace('.pt', '')
        path     = os.path.join(feature_dir, fname)

        feat     = load_feature(path)
        segments = generate_captions_for_video(feat, decoder, seg_len=seg_len, fps=fps)
        segments = merge_caption_segments(segments)

        entry = {
            "video_id": video_id,
            "segments": segments
        }
        results.append(entry)

        # Print progress every 10 videos
        if (i + 1) % 10 == 0 or i == 0:
            ex = segments[0] if segments else {}
            cap = ex.get('caption', '')[:40]
            print(f"  [{i+1:>4}/{len(files)}]  {video_id}  →  '{cap}...'")

    # Save to JSONL
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')

    print(f"\n✅ Saved captions for {len(results)} videos → {output_file}")
    print("\nSample entry:")
    print(json.dumps(results[0], indent=2) if results else "(empty)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", default="embeddings1",
                        help="Folder with .pt files from Team A")
    parser.add_argument("--output",      default="captions.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--seg_len",     type=int,   default=8)
    parser.add_argument("--fps",         type=float, default=1.0)
    parser.add_argument("--limit",       type=int,   default=None,
                        help="Process only first N videos (for quick testing)")
    args = parser.parse_args()

    run(args)
