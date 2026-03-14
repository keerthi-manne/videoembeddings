# Phase 4: Model Retraining & Dataset Expansion Guide

This guide details the steps to scale the video event segmentation project by expanding the dataset, re-generating features, and retraining the temporal STAdapter.

## Step 9 — Preparation for Model Retraining
To improve temporal representation quality, caption accuracy, and retrieval precision, we must transition from Phase 1 prototyping into Phase 4 data-at-scale operations.

### For Team A (Data operations)
1. **Collect a larger open video dataset:**
   - Instead of restricting to a 5,000 video limit, index and download all applicable MSR-VTT videos (~10,000) or combine with ActivityNet Captions. All video assets go in `.mp4` into `data/raw_videos/`.
2. **Generate embeddings for the new videos:**
   - Execute `python scripts/preprocess_videos.py` to extract frames for all videos ensuring constant `fps=1.0`. **DO NOT** mean pool video frame representations. Ensure every second gets saved as an individual frame output.
   - Run `python scripts/extract_features.py` to process these frames using CLIP ViT-B/32, producing robust `(T, 512)` PyTorch `.pt` datasets inside the `embeddings1/` (or `data/features/`) folder.

### For Team B (Model Operations)
1. **Prepare True MSR-VTT Ground Truth:**
   - The system utilizes Multimodal Contrastive Learning. You **must** provide the real MSR-VTT caption dataset (`data/real_msrvtt_captions.json`) for final semantic evaluation. The mock dataset provided earlier is strictly for infrastructure testing only.
   
2. **Retrain STAdapter using the expanded dataset:**
   - With new, accurate variables of frames ($T \geq 8$ on average), Team B executes the continuous training script:
     ```bash
     python training/train_contrastive.py --feature_dir embeddings1 --captions_json data/real_msrvtt_captions.json --epochs 25
     ```
   - The expanded visual context allows the temporal convolution mechanism in `STAdapter` to establish long-term motion associations before dimension pooling. The optional video projection head maps these features accurately to the frozen text embeddings.

3. **Replace the current STAdapter weights:**
   - The training script automatically registers and produces higher-accuracy weights inside `checkpoints/st_adapter_contrastive.pt`.
   - The end-to-end framework reads sequentially from this checkpoint, immediately propagating visual accuracy updates across Phase 2 (Caption Generation) and Phase 3 (Text Video Retrieval).

By seamlessly iterating via `train_contrastive.py`, the system is effectively self-upgrading once Team A delivers comprehensive `.pt` CLIP arrays and true semantic supervision captions.
