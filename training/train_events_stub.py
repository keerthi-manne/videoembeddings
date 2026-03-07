"""
training/train_events_stub.py  –  Team B (Person B2)

What this does:
  Tests the full event pipeline with FAKE labels (no real dataset needed).
  The goal is NOT to get good accuracy, but to prove the code path works:
    Load features → segment → EventHead → loss → backward → print predictions

When Team A delivers data + heuristic events (Day 7), we swap fake labels
for real ones (see the TODO comment in the code).

Run:
    python training/train_events_stub.py
    python training/train_events_stub.py --num_videos 5 --epochs 3
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import argparse

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.event_head import EventHead, segment_features
from utils.events import merge_segments, segments_to_time, print_timeline


def make_fake_data(num_videos: int, T: int, D: int, num_classes: int):
    """
    Generates fake CLIP features and fake event labels.

    In Phase 1 we don't have real event labels yet, so:
    - Features are random tensors (same shape as real CLIP output)
    - Labels: all segments of a video get the same label (= video index % num_classes)
      (rough approximation: the whole video is one 'event type')

    TODO (Day 7): Replace this with real data from:
        data/meta/samples_with_events.jsonl  (Team A+B heuristic events)
    """
    features_list = []
    labels_list = []

    for i in range(num_videos):
        # Fake CLIP features: (T, D) same shape as real output
        feat = torch.randn(T, D)
        features_list.append(feat)

        # Fake label: same for all segments of this video
        video_label = i % num_classes
        labels_list.append(video_label)

    return features_list, labels_list


def train_events_stub(num_videos=8, T=32, epochs=5, seg_len=8, lr=0.001):
    """
    Main training loop for event classification.

    The pipeline per batch:
      features (T, D)
        → segment_features()   group into S segments
        → EventHead            classify each segment
        → cross_entropy loss   compare to fake labels
        → backward             update EventHead weights
    """
    print("🚀 Starting Event Stub Training")
    print(f"   Videos: {num_videos}, T={T}, seg_len={seg_len}, epochs={epochs}\n")

    # Config
    D = 512
    num_classes = 10

    # Load model
    event_head = EventHead()  # reads configs/model_base.json
    optimizer = optim.Adam(event_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Generate fake data
    features_list, video_labels = make_fake_data(num_videos, T, D, num_classes)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        event_head.train()

        for vid_idx, (feat, video_label) in enumerate(zip(features_list, video_labels)):
            # feat: (T, D) → add batch dim → (1, T, D)
            feat = feat.unsqueeze(0)

            # Segment: (1, T, D) → (1, S, D)
            segments = segment_features(feat, seg_len=seg_len)
            B, S, _ = segments.shape

            # EventHead forward: (1, S, D) → (1, S, num_classes)
            logits = event_head(segments)

            # Fake labels: all segments get the same label as the video
            # Shape needed for CrossEntropyLoss: (B*S,)
            fake_labels = torch.full((B * S,), video_label, dtype=torch.long)

            # Reshape logits: (B, S, num_classes) → (B*S, num_classes)
            logits_flat = logits.view(B * S, -1)

            # Compute loss, backprop
            loss = criterion(logits_flat, fake_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_videos
        print(f"Epoch {epoch+1}/{epochs}  |  Avg Loss: {avg_loss:.4f}")

    # ── After training: print event predictions for one video ──
    print("\n📊 Sample Event Predictions (last video):")
    event_head.eval()
    with torch.no_grad():
        feat = features_list[-1].unsqueeze(0)          # (1, T, D)
        segments = segment_features(feat, seg_len=seg_len)  # (1, S, D)
        logits = event_head(segments)                       # (1, S, num_classes)

        probs = torch.softmax(logits, dim=-1)[0]            # (S, num_classes)
        pred_labels = probs.argmax(dim=-1).tolist()          # [label, label, ...]
        pred_confs  = probs.max(dim=-1)[0].tolist()          # [conf, conf, ...]

    # Print raw per-segment output
    print(f"\n{'Seg':<6} {'start_frame':<14} {'end_frame':<12} {'pred_label':<12} {'confidence'}")
    print("─" * 60)
    for i, (lbl, conf) in enumerate(zip(pred_labels, pred_confs)):
        start_f = i * seg_len
        end_f   = (i + 1) * seg_len - 1
        print(f"  {i:<4}  {start_f:<14}  {end_f:<12}  event{lbl:<8}  {conf:.3f}")

    # Merge segments and convert to timeline
    merged   = merge_segments(pred_labels, pred_confs, conf_thresh=0.0)  # 0.0 = show all
    timeline = segments_to_time(merged, seg_len=seg_len, fps=1.0)
    print_timeline(timeline)

    print("\n✅ Event stub training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_videos", type=int, default=8)
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--T",          type=int, default=32,  help="Frames per video")
    parser.add_argument("--seg_len",    type=int, default=8,   help="Frames per segment")
    parser.add_argument("--lr",         type=float, default=0.001)
    args = parser.parse_args()

    train_events_stub(
        num_videos=args.num_videos,
        T=args.T,
        epochs=args.epochs,
        seg_len=args.seg_len,
        lr=args.lr
    )
