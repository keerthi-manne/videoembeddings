"""
scripts/text_video_retrieval_hybrid.py

Hybrid Search = CLIP (semantic) + keyword matching

Run:
python scripts/text_video_retrieval_hybrid.py \
    --query "people dressed as storm troopers"
"""

import torch
import torch.nn.functional as F
import os
import argparse
import json
import re


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
def keyword_score(query, caption):
    query_words = re.findall(r"\w+", query.lower())
    caption_words = set(re.findall(r"\w+", caption.lower()))

    score = 0
    for word in query_words:
        if word in caption_words:
            # Boost rare/important words
            score += 1 + len(word) * 0.1

    return score / (len(query_words) + 1e-6)


# -------------------------
# Main search function
# -------------------------
def search_json(json_path, query, top_k=5):
    data = load_json(json_path)
    if data is None:
        return

    print("🔤 Loading CLIP Text Encoder...")
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    print(f"\n🔍 Searching for: '{query}'\n")

    # -------------------------
    # Encode query
    # -------------------------
    tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        query_emb = model.encode_text(tokens).squeeze(0)
        query_emb = F.normalize(query_emb, dim=-1).float()

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
            if (
                not caption or
                caption.lower() in ["no caption available", "none", "n/a"]
            ):
                continue

            # Encode caption
            tokens = clip.tokenize([caption]).to(device)

            with torch.no_grad():
                emb = model.encode_text(tokens).squeeze(0)
                emb = F.normalize(emb, dim=-1).float()

            # -------------------------
            # Hybrid scoring
            # -------------------------
            semantic_sim = (query_emb @ emb).item()
            kw_score = keyword_score(query, caption)

            alpha = 0.7  # semantic weight
            beta = 0.3   # keyword weight

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
        print(f"     Caption: {res['caption']}")
        print(f"     Final Score: {res['similarity']:.3f}")
        print(f"     (Semantic: {res['semantic']:.3f}, Keyword: {res['keyword']:.3f})\n")

    print("─────────────────────────────────────────────────────")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument(
        "--json",
        default="data/video_event_new_tst_captions.jsonl",
        help="Path to JSONL captions file"
    )
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()
    search_json(args.json, args.query, args.top_k)