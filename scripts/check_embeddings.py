"""
scripts/check_embeddings.py  --  Sanity check Team A's .pt files
Run this first before training to understand what you have.

Usage:
    python scripts/check_embeddings.py --feature_dir embeddings1
"""

import torch
import os
import argparse


def check(feature_dir: str, min_T: int = 2):
    files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]
    print(f"\n📂 Folder: {feature_dir}")
    print(f"   Total .pt files: {len(files)}\n")

    shapes = {}
    bad = []
    good = []

    for f in files:
        path = os.path.join(feature_dir, f)
        t = torch.load(path, weights_only=False)

        # Must be 2D: (T, D)
        if t.dim() == 1:
            t = t.unsqueeze(0)          # (D,) → (1, D)
        if t.dim() != 2:
            bad.append((f, f"unexpected dims: {t.shape}"))
            continue

        T, D = t.shape
        shapes[T] = shapes.get(T, 0) + 1

        if D != 512:
            bad.append((f, f"D={D}, expected 512"))
        elif T < min_T:
            bad.append((f, f"T={T} < min_T={min_T}, too short for temporal modeling"))
        else:
            good.append((f, T, D))

    # Report
    print("Frame count distribution (T = number of frames per video):")
    for k in sorted(shapes):
        bar = "█" * shapes[k]
        print(f"  T={k:>3}: {shapes[k]:>4} videos  {bar}")

    print(f"\n✅ Usable videos (T >= {min_T}, D=512): {len(good)}")
    print(f"❌ Skipped videos:                      {len(bad)}")

    if bad:
        print("\nSkipped details:")
        for name, reason in bad[:10]:
            print(f"  {name}: {reason}")
        if len(bad) > 10:
            print(f"  ... and {len(bad)-10} more")

    print(f"\n💡 Recommendation:")
    if len(good) < 20:
        print("  ⚠️  Very few usable videos. Ask Team A to re-extract with fps=1.0")
        print("      and ensure per-frame saving (not mean-pooled).")
    else:
        print(f"  ✅ Enough to start training. Use --feature_dir {feature_dir}")
        print(f"     For seg_len, use max that fits: check T values above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", default="embeddings1")
    parser.add_argument("--min_T",       type=int, default=2,
                        help="Minimum frames needed to be considered usable")
    args = parser.parse_args()
    check(args.feature_dir, args.min_T)
