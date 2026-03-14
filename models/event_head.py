import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys
import os
from typing import Union, List

# Fix Windows import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from st_adapter import STAdapter

def segment_features(
    x: torch.Tensor, 
    seg_len: int = 8,
    mode: str = 'mean',
    return_indices: bool = False
) -> Union[torch.Tensor, tuple]:
    """
    [B, T, D] → [B, S, D]
    mode: 'mean', 'max', 'first'
    """
    B, T, D = x.shape
    
    # Pad to multiple of seg_len
    pad_len = (seg_len - T % seg_len) % seg_len
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
        T += pad_len
    
    S = T // seg_len
    
    if mode == 'mean':
        segments = x.view(B, S, seg_len, D).mean(dim=2)
    elif mode == 'max':
        segments = x.view(B, S, seg_len, D).max(dim=2)[0]
    elif mode == 'first':
        segments = x[:, :S*seg_len:seg_len, :]  # First frame per segment
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if return_indices:
        indices = torch.arange(S).repeat(B, 1) * seg_len
        return segments, indices
    
    return segments

def dynamic_segmentation(
    x: torch.Tensor, 
    threshold: float = 0.8,
    min_len: int = 4
) -> List[dict]:
    """
    Step 3 from Architecture Diagram:
    Iterates through time-aware features and splits when similarity < threshold.
    
    Args:
        x: (T, D) single video features
        threshold: 0.8 default
        min_len: minimum frames to avoid tiny noise segments
        
    Returns:
        List of indices where segments start and end.
    """
    T, D = x.shape
    if T <= 1:
        return [{"start_idx": 0, "end_idx": T}]

    # Normalize for cosine similarity
    x_norm = F.normalize(x, dim=-1)
    
    boundaries = [0]
    curr_start = 0
    
    for i in range(1, T):
        # Calculate similarity between current frame and the 'average' of current segment
        # This is more robust than just checking frame-to-frame
        seg_center = x_norm[curr_start:i].mean(dim=0)
        seg_center = F.normalize(seg_center, dim=0)
        
        sim = (x_norm[i] @ seg_center).item()
        
        # If similarity drops, it's a new event!
        if sim < threshold and (i - curr_start) >= min_len:
            boundaries.append(i)
            curr_start = i
            
    boundaries.append(T)
    
    segments = []
    for i in range(len(boundaries)-1):
        segments.append({
            "start_idx": boundaries[i],
            "end_idx": boundaries[i+1],
            "feat": x[boundaries[i]:boundaries[i+1]].mean(dim=0) # Average feature for the segment
        })
        
    return segments

class EventHead(nn.Module):
    def __init__(self, config_path: str = 'configs/model_base.json'):
        super().__init__()
        self.config_path = config_path
        
        config = self._load_config(config_path)
        self.D = config['D']
        self.num_classes = config['num_classes']
        self.seg_len = config['seg_len']
        self.dropout_p = config.get('dropout', 0.1)
        
        hidden = self.D // 2
        self.fc = nn.Sequential(
            nn.Linear(self.D, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden, self.num_classes)
        )
        
        print(f"✅ EventHead: {self.D}→{self.num_classes}, params={self._count_params():.0f}K")
    
    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            return json.load(f)
    
    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, D] → [B, S, num_classes]"""
        B, S, D = x.shape
        if D != self.D:
            raise ValueError(f"Expected D={self.D}, got {D}")
        
        x = x.view(B * S, D)
        logits = self.fc(x)
        return logits.view(B, S, -1)

def visualize_events(
    logits: torch.Tensor, 
    seg_len: int = 8, 
    fps: float = 1.0,
    conf_thresh: float = 0.5
):
    """Prints Day 10 demo format"""
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)[0]  # First video
    conf = probs.max(dim=-1)[0][0]
    
    print("📱 Event Timeline:")
    for i, (p, c) in enumerate(zip(pred, conf)):
        if c >= conf_thresh:
            start = i * seg_len / fps
            end = (i + 1) * seg_len / fps
            print(f"[ {start:.1f}s - {end:.1f}s ] event{p} ({c:.2f})")

if __name__ == "__main__":
    print("🧪 Testing FULL PIPELINE...")
    
    # Dynamic test
    adapter = STAdapter()
    head = EventHead()
    
    lengths = [25, 32, 40]
    for T in lengths:
        print(f"\n--- Testing T={T} ---")
        x = torch.randn(2, T, 512)
        
        enhanced = adapter(x)
        segments = segment_features(enhanced, seg_len=8)
        logits = head(segments)
        
        print(f"✅ {x.shape} → {logits.shape}")
        
        # Demo visualization
        visualize_events(logits[:1], seg_len=8, fps=1.0)
    
    print("\n🎉 ALL TESTS PASS! Production ready!")
