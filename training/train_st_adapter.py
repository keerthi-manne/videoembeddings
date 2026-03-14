"""
training/train_st_adapter.py  –  Team B (Person B1)

What this does:
  Trains the STClassifier (STAdapter + mean-pool + Linear head)
  for whole-video classification.

Why?
  We need to teach the STAdapter to model temporal patterns first,
  using a simpler task (whole-video label), before the harder event task.

Data modes:
  Mode 1 (RIGHT NOW, no Team A): Uses random fake tensors. Loss should
          decrease slightly — proves training loop works.
  Mode 2 (Once Team A delivers): Point --feature_dir at data/features/
          and the script automatically uses real CLIP features.

Run:
    # Mode 1 – no data needed (test the loop works)
    python training/train_st_adapter.py

    # Mode 2 – with real Team A features
    python training/train_st_adapter.py --feature_dir data/features --samples_file data/meta/samples.jsonl
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.st_classifier import STClassifier


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────

def load_fake_dataset(num_videos=50, T=30, D=512, num_classes=10):
    """
    Mode 1: Generates random (features, label) pairs.
    Shape matches exactly what Team A will produce.
    """
    data = []
    for i in range(num_videos):
        feat  = torch.randn(T, D)           # (T, 512) ← same shape as real .pt files
        label = i % num_classes
        data.append((feat, label))
    return data


def load_real_dataset(samples_file: str, feature_dir: str, num_classes: int):
    """
    Mode 2a: Loads real CLIP features via samples.jsonl index.
    Each file: data/features/<video_id>.pt → tensor (T, 512)
    """
    data = []
    with open(samples_file) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    for i, s in enumerate(samples):
        vid_id = s["video_id"]
        feat_path = os.path.join(feature_dir, f"{vid_id}.pt")
        if not os.path.exists(feat_path):
            continue
        feat  = torch.load(feat_path, weights_only=False)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)        # (D,) → (1, D)
        if feat.shape[0] < 2:
            continue                        # skip T=1, not useful for temporal model
        label = i % num_classes
        data.append((feat, label))

    print(f"📂 Loaded {len(data)} real videos from {feature_dir}")
    return data


def load_embeddings_dir(feature_dir: str, num_classes: int, min_T: int = 2):
    """
    Mode 2b: Loads ALL .pt files directly from Team A's embeddings folder.
    No samples.jsonl needed — just point at the folder.

    Usage:
        python training/train_st_adapter.py --feature_dir embeddings1
    """
    data = []
    skipped = 0
    files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.pt')])

    for i, fname in enumerate(files):
        path = os.path.join(feature_dir, fname)
        feat = torch.load(path, weights_only=False)

        # Normalise shape
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)        # (D,) → (1, D)
        if feat.dim() != 2:
            skipped += 1
            continue
        T, D = feat.shape
        if T < min_T or D != 512:
            skipped += 1
            continue

        label = i % num_classes
        data.append((feat, label))

    print(f"📂 Loaded {len(data)} usable videos from '{feature_dir}' "
          f"(skipped {skipped} with T<{min_T} or wrong shape)")
    return data


def split_dataset(data, train_ratio=0.8):
    """Simple 80/20 split by index."""
    random.shuffle(data)
    n = int(len(data) * train_ratio)
    return data[:n], data[n:]


def make_batch(samples, batch_size=16):
    """
    Yields (features_batch, labels_batch) tuples.
    Pads sequences to the same T within each batch.
    """
    random.shuffle(samples)
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        feats, labels = zip(*batch)

        # Pad all to max T in this batch
        max_T = max(f.shape[0] for f in feats)
        D     = feats[0].shape[1]
        padded = torch.zeros(len(feats), max_T, D)
        for j, f in enumerate(feats):
            padded[j, :f.shape[0], :] = f

        yield padded, torch.tensor(labels, dtype=torch.long)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(args):
    print("🚀 ST-Adapter Training")

    # Decide data mode
    # Mode 2a: --feature_dir + --samples_file (Team A's final structured output)
    # Mode 2b: --feature_dir only             (Team A's embeddings folder, no index needed)
    # Mode 1:  nothing given                  (fake data for testing)
    use_real_indexed = (args.feature_dir is not None and args.samples_file is not None
                        and os.path.exists(args.feature_dir) and os.path.exists(args.samples_file))
    use_real_dir     = (args.feature_dir is not None and args.samples_file is None
                        and os.path.exists(args.feature_dir))

    config_path = args.config
    config = json.load(open(config_path))
    num_classes = config["num_classes"]
    lr          = config["learning_rate"]
    batch_size  = config["batch_size"]

    if use_real_indexed:
        print(f"📂 Mode 2a: Real features via samples.jsonl from {args.feature_dir}")
        dataset = load_real_dataset(args.samples_file, args.feature_dir, num_classes)
    elif use_real_dir:
        print(f"📂 Mode 2b: Loading all .pt files directly from '{args.feature_dir}'")
        dataset = load_embeddings_dir(args.feature_dir, num_classes, min_T=2)
    else:
        print("🔧 Mode 1: Fake data (no --feature_dir given)")
        dataset = load_fake_dataset(num_videos=args.num_videos, num_classes=num_classes)

    train_data, val_data = split_dataset(dataset)
    print(f"   Train: {len(train_data)} videos, Val: {len(val_data)} videos\n")

    # Model, optimizer, loss
    model     = STClassifier(config_path=config_path)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for feats, labels in make_batch(train_data, batch_size):
            logits = model(feats)                           # (B, num_classes)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * len(labels)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total   += len(labels)

        train_acc  = train_correct / train_total
        train_loss = train_loss / train_total

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for feats, labels in make_batch(val_data, batch_size):
                logits = model(feats)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total   += len(labels)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"Epoch {epoch+1:>2}/{args.epochs}  |  "
              f"Loss: {train_loss:.4f}  |  "
              f"Train Acc: {train_acc:.3f}  |  "
              f"Val Acc: {val_acc:.3f}")

        # Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/st_adapter.pt")

    print(f"\n✅ Training done! Best Val Acc: {best_val_acc:.3f}")
    print(f"💾 Saved → checkpoints/st_adapter.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/model_base.json")
    parser.add_argument("--feature_dir",  default=None,  help="data/features  (Team A output)")
    parser.add_argument("--samples_file", default=None,  help="data/meta/samples.jsonl (Team A output)")
    parser.add_argument("--epochs",       type=int, default=10)
    parser.add_argument("--num_videos",   type=int, default=50,  help="only for fake-data mode")
    args = parser.parse_args()

    train(args)
