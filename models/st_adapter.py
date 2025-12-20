import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from typing import Optional

class STAdapter(nn.Module):
    def __init__(self, config_path: str = 'configs/model_base.json'):
        super().__init__()
        self.config_path = config_path
        
        # Load config with error handling
        config = self._load_config(config_path)
        
        self.D = config['D']
        self.d = config['d']
        self.dropout_p = config.get('dropout', 0.1)
        
        # Core layers
        self.down_proj = nn.Linear(self.D, self.d)
        self.temporal_conv = nn.Conv1d(self.d, self.d, kernel_size=3, padding=1)
        self.up_proj = nn.Linear(self.d, self.D)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)
        self.norm = nn.LayerNorm(self.D)
        
        print(f"✅ ST-Adapter: D={self.D}→d={self.d}, params={self._count_params():.0f}K")
    
    def _load_config(self, path: str) -> dict:
        """Safe config loading"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            config = json.load(f)
        if config['D'] != 512:
            print(f"⚠️ Non-standard CLIP dim: {config['D']} (expected 512)")
        return config
    
    def _count_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] → [B, T, D]
        Works with ANY T (dynamic length)
        """
        B, T, D = x.shape
        if D != self.D:
            raise ValueError(f"Expected D={self.D}, got {D}")
        
        residual = self.norm(x)
        x = self.down_proj(residual)           # [B, T, d]
        x = x.transpose(1, 2)                  # [B, d, T]
        x = self.temporal_conv(x)              # [B, d, T]
        x = x.transpose(1, 2)                  # [B, T, d]
        x = self.act(x)
        x = self.up_proj(x)                    # [B, T, D]
        x = self.dropout(x)
        return x + residual

if __name__ == "__main__":
    # Test dynamic lengths
    adapter = STAdapter()
    lengths = [16, 25, 32, 64]
    for T in lengths:
        x = torch.randn(2, T, 512)
        out = adapter(x)
        assert out.shape == x.shape, f"Shape mismatch for T={T}"
        print(f"✅ T={T}: {x.shape} → {out.shape}")
    print("🎉 ST-Adapter: All tests PASS!")
