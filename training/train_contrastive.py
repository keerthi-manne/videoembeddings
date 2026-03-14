import os
import sys
import torch
import json
import random
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.st_adapter import STAdapter
from models.contrastive_loss import ContrastiveLoss
import clip

class MSRVTTCaptionDataset(Dataset):
    """
    Custom Dataset that loads both the video .pt embedding AND its text captions.
    """
    def __init__(self, feature_dir: str, captions_json: str):
        self.feature_dir = feature_dir
        self.samples = []
        
        # Load the official MSR-VTT Retrieval JSON splits (list of dicts)
        # Format: [ {"video": "video1000.mp4", "caption": "a person cooking"}, ... ]
        with open(captions_json, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Convert list to an optimized dictionary: { "video1000": ["caption 1", "caption 2"] }
        self.metadata = {}
        for item in raw_data:
            vid_id = item["video"].replace(".mp4", "")
            if vid_id not in self.metadata:
                self.metadata[vid_id] = []
            self.metadata[vid_id].append(item["caption"])
            
        # Only keep videos that physically exist in the embeddings folder AND have captions in the JSON
        available_files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]
        for fname in available_files:
            vid_id = fname.replace('.pt', '')
            if vid_id in self.metadata:
                self.samples.append((vid_id, os.path.join(feature_dir, fname), self.metadata[vid_id]))
                
        print(f"✅ Loaded {len(self.samples)} valid video-caption pairings out of {len(available_files)} internal .pt files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_id, pt_path, captions_list = self.samples[idx]
        
        # 1. Load video tensor
        feat = torch.load(pt_path, weights_only=False)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
            
        # 2. Random Sampling Strategy
        # MSR-VTT has ~20 captions per video. 
        # By randomly picking one on the fly, the model sees *different* valid descriptions
        # of the same video across different epochs! This creates massive data augmentation.
        chosen_caption = random.choice(captions_list)
        
        # 3. Tokenize text caption using CLIP
        # clip.tokenize returns shape (1, 77), we squeeze to (77,)
        text_tokens = clip.tokenize(chosen_caption, truncate=True).squeeze(0)
        
        return feat, text_tokens, vid_id, chosen_caption

def collate_fn(batch):
    """
    Pads variable-length video tensors to match the longest video in the batch.
    """
    feats, tokens, vid_ids, chosen_captions = zip(*batch)
    
    # Pad video features (T, 512) along the T dimension
    import torch.nn.utils.rnn as rnn_utils
    # We must ensure they are 2D tensors
    feats_2d = [f if f.dim() == 2 else f.unsqueeze(0) for f in feats]
    padded_feats = rnn_utils.pad_sequence(feats_2d, batch_first=True) # (B, max_T, 512)
    
    # Tokens are already fixed size (77,)
    stacked_tokens = torch.stack(tokens) # (B, 77)
    
    return padded_feats, stacked_tokens, list(vid_ids), list(chosen_captions)

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Contrastive Training starting on {device}")
    
    # 1. Datasets
    dataset = MSRVTTCaptionDataset(args.feature_dir, args.captions_json)
    if len(dataset) == 0:
        print("❌ Dataset empty. Exiting.")
        return
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 2. Models
    # A) Our trainable STAdapter
    st_adapter = STAdapter(config_path=args.config).to(device)
    
    # We can load existing STAdapter weights to warm-start if we want!
    if os.path.exists(args.checkpoint) and not args.from_scratch:
        print(f"🔄 Warming start with weights from {args.checkpoint}")
        full_sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Strip prefixes if it came from the old STClassifier
        adapter_sd = {k.replace("st_adapter.", "", 1): v for k, v in full_sd.items() if k.startswith("st_adapter.")}
        st_adapter.load_state_dict(adapter_sd if adapter_sd else full_sd)

    # B) Pre-trained frozen CLIP text encoder
    print("🔤 Loading Frozen CLIP Text Encoder...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval() # Never train the text encoder!
    
    # C) The Loss Function
    criterion = ContrastiveLoss(temperature=0.07).to(device)
    
    # D) Optional Projection Head (Video: 512 -> 512)
    video_proj = torch.nn.Linear(512, 512).to(device) if args.use_proj else torch.nn.Identity().to(device)
    
    # 3. Optimizer
    # We train STAdapter parameters + projection head (if used) + learnable temperature
    optimizer = optim.AdamW(
        list(st_adapter.parameters()) + list(video_proj.parameters()) + [criterion.logit_scale],
        lr=args.lr, weight_decay=0.05
    )
    
    # 4. Training Loop
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        st_adapter.train()
        total_loss = 0.0
        
        for batch_idx, (video_batch, text_tokens_batch, vid_ids, chosen_captions) in enumerate(dataloader):
            video_batch = video_batch.to(device).float() # (B, T, 512)
            text_tokens_batch = text_tokens_batch.to(device) # (B, 77)
            
            # --- Sanity Check Display (Only print once per run) ---
            if epoch == 1 and batch_idx == 0:
                print("\n🧐 SANITY CHECK: Random Sample of Ground Truth Pairings from this batch:")
                for i in range(min(3, len(vid_ids))):
                    print(f"   🎥 {vid_ids[i]}.pt  -->  📝 \"{chosen_captions[i]}\"")
                print("----------------------------------------------------------------\n")
            
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            # 1. Process video through STAdapter
            enriched_video = st_adapter(video_batch) # (B, T, 512)
            
            # Pool across Time (T) to get a single vector per video: (B, 512)
            video_features = enriched_video.mean(dim=1) 
            
            # Optional: Pass through projection head into dedicated contrastive space
            video_features = video_proj(video_features)
            
            # 2. Get Text Features from Frozen CLIP
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tokens_batch).float() # (B, 512)
                
            # 3. Compute Contrastive Loss (match video_features to text_features)
            loss = criterion(video_features, text_features)
            
            # --- Backward Pass ---
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{args.epochs}] | Contrastive Loss: {avg_loss:.4f} | Temp: {criterion.logit_scale.exp().item():.3f}")
        
        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            
            if args.use_proj:
                torch.save({
                    "st_adapter": st_adapter.state_dict(),
                    "video_proj": video_proj.state_dict()
                }, args.save_path)
            else:
                torch.save(st_adapter.state_dict(), args.save_path)
                
            print(f"   💾 Best model saved -> {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", default="embeddings1")
    parser.add_argument("--captions_json", default="data/mock_msrvtt_captions.json", help="Path to actual Ground Truth sentences")
    parser.add_argument("--checkpoint", default="checkpoints/st_adapter.pt", help="Warm start weights")
    parser.add_argument("--save_path", default="checkpoints/st_adapter_contrastive.pt", help="Where to save new weights")
    parser.add_argument("--config", default="configs/model_base.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--from_scratch", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--use_proj", action="store_true", help="Add a 512->512 projection head for video features")
    
    args = parser.parse_args()
    train(args)

