# Team A – Phase 1 Guide
> Pull this repo first: `git pull origin main`  
> Duration: Days 1–10 | 4-5 month Samsung Work-let Project

---

## What Team A does (plain English)

Team B builds the "brain" (STAdapter + EventHead) that understands time in videos.  
**Team A's job:** feed that brain clean data.

```
.mp4 videos  →  [Team A]  →  (T, 512) feature tensors  →  [Team B]  →  Event Timeline
```

Team B's models are already in the repo. Your output plugs directly into them.

---

## Folder Structure to Create

```
video_event_segmentation/
    data/
        raw_videos/          ← put all MSR-VTT .mp4 files here
        frames/              ← extracted keyframes go here (auto-created by script)
        features/            ← CLIP feature tensors go here (auto-created by script)
        meta/
            samples.jsonl    ← your main output (Team B needs this)
            train_ids_phase1.txt
    video_pipeline/
        preprocess.py        ← write this (Step 2)
        clip_features.py     ← write this (Step 3)
    scripts/
        create_msrvtt_index.py   ← write this (Step 1)
        preprocess_videos.py     ← write this (Step 2)
        extract_features.py      ← write this (Step 3)
    training/
        dataset.py               ← write this (Step 4)
        train_baseline.py        ← write this (Step 4)
    models/
        baseline_classifier.py   ← write this (Step 4)
    api/
        inference_backbone.py    ← write this (Step 5)
```

---

## Step 1 — Download MSR-VTT + Create Index (Person A1)

### Download
- Search: **"MSR-VTT dataset download"**
- Get: video archive (`.zip` of `.mp4` files) + annotation JSON
- Put videos in `data/raw_videos/`

### Write `scripts/create_msrvtt_index.py`

**Input:** path to MSR-VTT annotations JSON and `raw_videos/` folder  
**Output:** `data/meta/samples.jsonl` — one line per video:

```json
{"video_id": "video0", "video_path": "data/raw_videos/video0.mp4", "caption": "a man cooking", "events": []}
```

**Rules:**
- Only include videos that have at least one caption
- Use only **3000–5000 videos** for Phase 1 (store IDs in `train_ids_phase1.txt`)
- Log any video where the `.mp4` file is missing → `missing_videos.txt`

---

## Step 2 — Extract Frames (Person A2)

### Write `video_pipeline/preprocess.py`

```python
def extract_frames(video_path: str, fps: float = 1.0) -> List[np.ndarray]:
    # Use cv2.VideoCapture
    # Sample 1 frame per second using frame index and video fps

def select_keyframes(frames: List[np.ndarray], hist_thresh: float = 0.3) -> List[np.ndarray]:
    # Convert frames to HSV
    # Compute histogram per frame using cv2.calcHist
    # Keep frame only if histogram distance from previous > hist_thresh
```

### Write `scripts/preprocess_videos.py`

- Reads `data/meta/samples.jsonl`
- For each video: `extract_frames` → `select_keyframes`
- Saves frames to `data/frames/<video_id>/0.jpg`, `1.jpg`, ...
- Logs: number of raw frames and keyframes kept per video

**Test on 200 videos first.** Target: 10–40 keyframes per video.

---

## Step 3 — CLIP Feature Extraction (Person A1)

### Write `video_pipeline/clip_features.py`

```python
import clip

model, preprocess = clip.load("ViT-B/32")
for p in model.parameters():
    p.requires_grad = False   # freeze CLIP, never retrain it

def encode_frame_batch(frames: List[PIL.Image]) -> torch.Tensor:
    # preprocess each frame → stack to (T, 3, 224, 224)
    # run through model.encode_image()
    # return shape: (T, 512)
```

### Write `scripts/extract_features.py`

- Reads `data/meta/samples.jsonl`
- For each video:
  - Load frames from `data/frames/<video_id>/`
  - Run `encode_frame_batch()`
  - Save tensor to `data/features/<video_id>.pt`
- Add `--limit` CLI flag (e.g. `--limit 200`)

**Verify:** open one `.pt` file → shape must be `(T, 512)`. This is exactly what Team B's training scripts expect.

---

## Step 4 — Dataset Loader + Baseline Classifier (Person A2)

### Write `training/dataset.py`

```python
class VideoFeatureDataset(Dataset):
    def __init__(self, samples_file, feature_dir, split="train"):
        # load samples.jsonl
        # 80/20 train/val split by index

    def __getitem__(self, idx):
        # load data/features/<video_id>.pt  → tensor (T, 512)
        # pseudo label = idx % num_classes  (no real labels needed yet)
        return features, label   # (T, 512), int
```

### Write `models/baseline_classifier.py`

```python
class BaselineClassifier(nn.Module):
    # Input: (B, T, 512)
    # Mean pool T frames → (B, 512)
    # Linear(512, num_classes)
    # Output: (B, num_classes)
    # No STAdapter — this is the simple baseline to compare against
```

### Write `training/train_baseline.py`

- Standard cross-entropy training loop, Adam optimizer, LR from `configs/model_base.json`
- Saves to `checkpoints/baseline.pt`
- Logs train/val accuracy per epoch

---

## Step 5 — Inference API (A1 + A2 Together)

### Write `api/inference_backbone.py`

```python
def process_video(video_path: str) -> dict:
    """
    Runs the full pipeline on a raw .mp4 and returns features.
    Team B calls this in their demo_events.py script.
    """
    frames = extract_frames(video_path, fps=1.0)
    frames = select_keyframes(frames)
    features = encode_frame_batch([PIL.Image.fromarray(f) for f in frames])
    return {
        "features": features,         # torch.Tensor (T, 512)
        "frame_indices": [...],       # list of original frame indices kept
        "fps": 1.0
    }
```

**Important:** Use the same `fps` and `hist_thresh` as the training scripts. Team B's model was trained on features extracted with these same settings.

---

## What Team B Needs From You — and When

| Deliverable | Deadline | Why Team B needs it |
|-------------|----------|---------------------|
| `data/features/` (200+ videos) | **ASAP** | Unblocks real training (`train_st_adapter.py`) |
| `training/dataset.py` | **ASAP** | Replaces fake data in training loop |
| `data/meta/samples.jsonl` | **ASAP** | Team B generates heuristic event labels from this |
| `api/inference_backbone.py` | **By Day 7** | Team B's final demo (`demo_events.py`) calls `process_video()` |

---

## How to Verify Your Work Plugs into Team B's Code

Once you have features ready, run Team B's training script:

```bash
python training/train_st_adapter.py \
  --feature_dir data/features \
  --samples_file data/meta/samples.jsonl \
  --epochs 10
```

If it prints accuracy per epoch and saves `checkpoints/st_adapter.pt` → **integration works!**

Also test event training:
```bash
python training/train_events_stub.py --epochs 5
```

---

## Run Order (Full Pipeline)

```bash
# Step 1 — create index
python scripts/create_msrvtt_index.py --json data/annotations/msrvtt.json --video_dir data/raw_videos

# Step 2 — extract frames
python scripts/preprocess_videos.py --limit 1000

# Step 3 — extract CLIP features
python scripts/extract_features.py --limit 1000

# Step 4 — train baseline
python training/train_baseline.py --config configs/model_base.json

# Step 5 — verify Team B integration
python training/train_st_adapter.py --feature_dir data/features --samples_file data/meta/samples.jsonl
```

---

## Key Technical Constraints (must match Team B's code)

| Setting | Value | Why |
|---------|-------|-----|
| CLIP model | `ViT-B/32` | Produces 512-dim features — Team B models are hard-coded for D=512 |
| Feature shape | `(T, 512)` per video | Must match `configs/model_base.json`: `"D": 512` |
| Sampling FPS | `1.0` | Must match `"fps": 1.0` in config |
| Save format | `torch.save(tensor, path)` | Team B loads with `torch.load()` |
