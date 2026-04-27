"""
scripts/text_video_retrieval.py

Hybrid Search = CLIP (semantic) + keyword matching

Run:
python scripts/text_video_retrieval.py \
    --query "people dressed as storm troopers"
"""

import torch
import torch.nn.functional as F
import os
import argparse
import json
import re
from collections import Counter , defaultdict
import clip
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------
# Load JSONL file
# -------------------------
def load_json(json_path):
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found at {json_path}")
        return None

    data = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


# -------------------------
# Keyword scoring (improved)
# -------------------------
def clean_text(text):
            return re.sub(r"[^\w\s]", "", text.lower()).strip()

def keyword_score(query, caption):
    q_words = re.findall(r"\w+", query)
    c_words = re.findall(r"\w+", caption)

    q_count = Counter(q_words)
    c_count = Counter(c_words)

    overlap = sum(min(q_count[w], c_count[w]) for w in q_count)
    return overlap / (len(q_words) + 1e-6)


# -------------------------
# Main search function
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
model.eval()
def search_json(json_path, query, top_k=5):
    data = load_json(json_path)
    if data is None:
        return

    # Load model ONCE

    print(f"\n🔍 Searching for: '{query}'\n")

    # Encode query properly
    query_clean = clean_text(query)
    query_emb = model.encode(query_clean, normalize_embeddings=True)

    results = []

    print("📊 Encoding captions & computing hybrid similarity...")

    # -------------------------
    # Loop through data
    # -------------------------
    for video in data:
        vid = video["video_id"]

        for seg in video["timeline"]:
            caption = seg.get("caption", "").strip()

            # ❌ Skip bad captions
            if not caption or caption.lower() in ["no caption available", "none", "n/a"]:
                continue

            caption_clean = clean_text(caption)

            # ✅ Correct encoding (NO tokenization)
            emb = model.encode(caption_clean, normalize_embeddings=True)

            # -------------------------
            # Hybrid scoring
            # -------------------------
            semantic_sim = float(query_emb @ emb)
            kw_score = keyword_score(query_clean, caption_clean)

            alpha = 0.8  # semantic weight
            beta = 0.2   # keyword weight

            sim = alpha * semantic_sim + beta * kw_score

            results.append({
                "similarity": sim,
                "semantic": semantic_sim,
                "keyword": kw_score,
                "video_id": vid,
                "start": seg["start"],
                "end": seg["end"],
                "caption": caption
            })

    # -------------------------
    # Sort results
    # -------------------------
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]

    # -------------------------
    # Print results
    # -------------------------
    print("─────────────────────────────────────────────────────")
    print(f"🏆 Top {top_k} results for: '{query}'")
    print("─────────────────────────────────────────────────────")

    for i, res in enumerate(results, 1):
        print(f"#{i} | {res['video_id']} | [{res['start']} – {res['end']}]")
        print(f"     Final Score: {res['similarity']:.3f}")
        print(f"     (Semantic: {res['semantic']:.3f}, Keyword: {res['keyword']:.3f})\n")

    print("─────────────────────────────────────────────────────")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument(
        "--json",
        default="data/final_video_event_new_tsting_captions.jsonl",
        help="Path to JSONL captions file"
    )
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()
    search_json(args.json, args.query, args.top_k)
