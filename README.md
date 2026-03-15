# Samsung Video Event Segmentation & Semantic Retrieval

![Project Status](https://img.shields.io/badge/Status-Presentation--Ready-brightgreen)
![Tech Stack](https://img.shields.io/badge/Tech-PyTorch%20%7C%20CLIP%20%7C%20OpenAI-blue)
![Params](https://img.shields.io/badge/Model%20Size-182K-blueviolet)

An end-to-end AI pipeline that adds **Temporal Intelligence** to static video features. This project transforms raw video frames into a searchable, captioned, and semantically understood event database.

## 🚀 Key Features
- **STAdapter (The Brain):** A lightweight 1D-CNN temporal bottleneck (182k parameters) that enables independent frames to "remember" actions over time.
- **Dynamic Segmentation:** Content-aware event detection using a **Similarity Threshold (< 0.8)** strategy. The system cuts videos naturally where the action changes.
- **Zero-Shot Captioning:** Automatically generates English descriptions for detected events using CLIP's shared latent space (no pre-defined labels required).
- **Semantic Text-Video Retrieval:** Search entire video libraries using natural language queries (e.g., *"someone cooking in a kitchen"*).
- **Edge Optimized:** Designed for high-speed, low-latency inference on Samsung mobile and wearable device hardware.

## 🏗️ System Architecture
1. **Feature Extraction (Team A):** Raw video frames are encoded via CLIP (ViT-B/32).
2. **Temporal Alignment:** STAdapter blends frame sequences into action-aware vectors.
3. **Contrastive Training:** Model weights are aligned with **Official MSR-VTT** ground-truth captions using InfoNCE Loss.
4. **Natural Boundary Detection:** Consecutive features are compared via Cosine Similarity to identify event transitions.
5. **Cross-Modal Retrieval:** Text queries are mapped into the video embedding space for sub-second search.

## 📊 Performance Benchmark (MSR-VTT)
| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Median Rank (MedR)** | **7.0** | The ground-truth result is typically in the Top 7 out of 224. |
| **Recall@10** | **58.04%** | >58% of queries find the exact correct video in the first page of results. |
| **Latency** | **< 1ms** | Sub-millisecond segment processing on standard CPU. |

## 🛠️ Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Run the Grand Presentation Demo
This script showcases the full pipeline: Segmentation -> Captioning -> Semantic Search.
```bash
python scripts/grand_presentation_demo.py
```

### Evaluate Retrieval Metrics
```bash
python scripts/evaluate_retrieval.py --captions_json msrvtt_ret_test1k.json
```

## 📂 Repository Structure
- `models/`: Architecture for STAdapter, Event Head, and Caption Decoder.
- `training/`: Contrastive training loop and loss functions.
- `scripts/`: Utilities for demo, index building, and evaluation.
- `checkpoints/`: Optimized semantic weights (`st_adapter_contrastive.pt`).

---
**Developed for the Samsung Advanced AI Research Project.**
