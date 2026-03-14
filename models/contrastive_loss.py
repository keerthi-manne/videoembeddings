import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Symmetric Cross-Entropy Loss (InfoNCE / NT-Xent)
    Used by CLIP to match Images (Videos) to Text.
    
    Given a batch of N video embeddings and N text embeddings,
    computes the cosine similarity matrix (N x N).
    The diagonal represents the correct (video, text) pairs.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # Learnable temperature parameter (like CLIP), initialized to log(1 / 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))

    def forward(self, video_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_features: shape (N, D) — e.g. D=512
            text_features:  shape (N, D)
            
        Returns:
            Scalar loss value
        """
        # 1. L2 Normalize both sets of features
        video_features = F.normalize(video_features, p=2, dim=-1)
        text_features  = F.normalize(text_features, p=2, dim=-1)

        # 2. Get the learnable temperature (clamped to prevent instability)
        logit_scale = self.logit_scale.exp().clamp(max=100)

        # 3. Compute cosine similarity matrix (N x N)
        # logits[i][j] = video_i dot text_j
        logits_per_video = logit_scale * (video_features @ text_features.T)
        logits_per_text  = logits_per_video.T  # Transpose for the other direction

        # 4. The target labels are exactly the diagonal (0, 1, 2, ..., N-1)
        N = video_features.shape[0]
        labels = torch.arange(N, dtype=torch.long, device=video_features.device)

        # 5. Compute symmetric cross-entropy loss
        loss_video = F.cross_entropy(logits_per_video, labels)
        loss_text  = F.cross_entropy(logits_per_text, labels)

        # Total loss is the average of both directions
        total_loss = (loss_video + loss_text) / 2.0
        
        return total_loss
