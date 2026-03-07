import torch
import torch.nn as nn
import json
import os
import sys

# So it can find st_adapter.py from same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from st_adapter import STAdapter


class STClassifier(nn.Module):
    """
    Full pipeline for whole-video classification:
      (B, T, 512) → STAdapter → mean pool → Linear → (B, num_classes)

    Why mean pool?
      STAdapter gives us T improved frame vectors. To classify the whole
      video as one label, we average all T vectors into one summary vector,
      then feed it to a Linear classifier.

    Used in: training/train_st_adapter.py
    """

    def __init__(self, config_path: str = 'configs/model_base.json'):
        super().__init__()

        # Load config
        config = self._load_config(config_path)
        self.D = config['D']                        # 512 (CLIP output dim)
        self.num_classes = config['num_classes']    # e.g. 10

        # The temporal module (the main contribution, Team B's job)
        self.st_adapter = STAdapter(config_path)

        # Simple classifier head after mean pooling
        self.classifier = nn.Linear(self.D, self.num_classes)

        print(f"✅ STClassifier ready: D={self.D}, classes={self.num_classes}")

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            return json.load(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)  ← CLIP feature tensor from Team A
           B = batch size (how many videos at once)
           T = number of frames per video (e.g. 30)
           D = 512 (CLIP output dimension)

        Returns: (B, num_classes)  ← one score per class per video
        """
        # Step 1: Add temporal context
        # (B, T, D) → (B, T, D)  same shape, but now frames "know" about neighbors
        x = self.st_adapter(x)

        # Step 2: Mean pool across all frames → one vector per video
        # (B, T, D) → (B, D)
        x = x.mean(dim=1)

        # Step 3: Classify
        # (B, D) → (B, num_classes)
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    print("🧪 Testing STClassifier...")

    model = STClassifier()

    # Simulate a batch of 4 videos, each with 30 frames, 512-dim CLIP features
    batch = torch.randn(4, 30, 512)
    output = model(batch)

    print(f"Input shape:  {batch.shape}")   # (4, 30, 512)
    print(f"Output shape: {output.shape}")  # (4, 10)
    assert output.shape == (4, 10), "❌ Shape mismatch!"

    # Also test with different T to confirm it handles variable length
    batch2 = torch.randn(2, 15, 512)
    out2 = model(batch2)
    assert out2.shape == (2, 10)

    print("✅ STClassifier: All tests PASS!")
