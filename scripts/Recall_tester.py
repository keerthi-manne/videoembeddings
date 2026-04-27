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
from collections import Counter , defaultdict
import clip
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------
# Load JSONL file
# -------------------------
import time 
start =   time.perf_counter()
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
def evaluate_from_caption_file(segment_json_path, query_json_path, top_k=10):
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
        # Load data
        # -------------------------
        data = load_json(segment_json_path)
        if data is None:
            return

        if not os.path.exists(query_json_path):
            print(f"❌ Query JSON not found: {query_json_path}")
            return

        with open(query_json_path, "r") as f:
            query_data = json.load(f)

        print("🔤 Loading Sentence Transformer...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # -------------------------
        # Prepare segment texts
        # -------------------------
        print("📦 Preparing segment captions...")
        segment_entries = []
        captions = []

        for video in data:
            vid = video["video_id"]

            for seg in video["timeline"]:
                caption = seg.get("caption", "").strip()

                if (
                    not caption or
                    caption.lower() in ["no caption available", "none", "n/a"]
                ):
                    continue

                caption_clean = clean_text(caption)

                segment_entries.append({
                    "video_id": vid,
                    "caption": caption_clean
                })

                captions.append(caption_clean)

        # -------------------------
        # Encode all captions (BATCH = FAST)
        # -------------------------
        print("⚡ Encoding all segment captions (batched)...")
        embeddings = model.encode(captions, normalize_embeddings=True, batch_size=64)

        for i in range(len(segment_entries)):
            segment_entries[i]["embedding"] = embeddings[i]

        print(f"✅ Total segments: {len(segment_entries)}\n")

        # -------------------------
        # Metrics
        # -------------------------
        top1, top5, top10 = 0, 0, 0
        total = 0

        print("🚀 Running evaluation...")

        # -------------------------
        # Loop over queries
        # -------------------------
        for i, (gt_video, query) in enumerate(query_data.items()):
            query = clean_text(query.strip())
            if not query:
                continue

            query_emb = model.encode(query, normalize_embeddings=True)

            # -------------------------
            # Video-level scoring
            # -------------------------

            video_segments = defaultdict(list)

            for item in segment_entries:
                vid = item["video_id"]
                emb = item["embedding"]

                semantic_sim = float(np.dot(query_emb, emb))
                kw_score = keyword_score(query, item["caption"])

                sim = 0.9 * semantic_sim + 0.1 * kw_score

                video_segments[vid].append(sim)

            # Top-3 pooling
            video_scores = {}
            for vid, sims in video_segments.items():
                sims.sort(reverse=True)
                video_scores[vid] = sum(sims[:3]) / min(3, len(sims))

            # Rank videos
            ranked_videos = sorted(video_scores, key=video_scores.get, reverse=True)

            # -------------------------
            # Evaluate
            # -------------------------
            if gt_video == ranked_videos[0]:
                top1 += 1
            if gt_video in ranked_videos[:5]:
                top5 += 1
            if gt_video in ranked_videos[:10]:
                top10 += 1

            total += 1

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(query_data)}")

        # -------------------------
        # Final Results
        # -------------------------
        print("\n📊 FINAL RESULTS")
        print("────────────────────────────")
        print(f"Total Queries: {total}")
        print(f"Top-1 Accuracy : {top1/total*100:.2f}%")
        print(f"Top-5 Accuracy : {top5/total*100:.2f}%")
        print(f"Top-10 Accuracy: {top10/total*100:.2f}%")
        print("────────────────────────────")

# -------------------------
# Entry point
# -------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--query", type=str, required=True)
#     parser.add_argument(
#         "--json",
#         default="data/video_event_new_tst_captions.jsonl",
#         help="Path to JSONL captions file"
#     )
#     parser.add_argument("--top_k", type=int, default=5)

#     args = parser.parse_args()
#     search_json(args.json, args.query, args.top_k)
if __name__ == "__main__":
    evaluate_from_caption_file(
        segment_json_path="data/final_video_event_new_tsting_captions.jsonl",
        query_json_path="Video_embeddings/modified_msrvtt_test_1k.json",
        top_k=10
    )
    end = time.perf_counter()
    print(" total time is " , end - start )