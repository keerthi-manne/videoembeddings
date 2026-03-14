"""
scripts/text_video_retrieval.py - Phase 3 (Steps 7-8)

1. Accepts user query.
2. Encodes via CLIP text model to 512 dimensions.
3. Computes similarity across all indexing vectors.
4. Spits out most relevant hits.

Run:
    python scripts/text_video_retrieval.py --query "person cooking food"
"""

import torch
import torch.nn.functional as F
import os
import argparse

def search_index(index_path, query, top_k=5):
    if not os.path.exists(index_path):
        print(f"❌ Index not found at {index_path}. Run build_retrieval_index.py first!")
        return
        print(f"❌ Index not found at {index_path}. Run build_retrieval_index.py first!")
        return
        
    print("📂 Loading retrieval index...")
    database = torch.load(index_path, map_location="cpu", weights_only=False)
    if len(database) == 0:
        print("❌ Database is empty!")
        return
        
    print("🔤 Loading CLIP Text Encoder...")
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    
    print(f"\n🔍 Searching for: '{query}'\n")
    
    # Encode Query
    tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        query_emb = model.encode_text(tokens).squeeze(0) # (512,)
        query_emb = F.normalize(query_emb, dim=-1).cpu()
    
    # Calculate Similarity for all events
    results = []
    for event in database:
        event_emb = event["segment_emb"]
        event_emb = F.normalize(event_emb, dim=-1) # L2 normalize
        
        sim = (query_emb @ event_emb).item()
        results.append((sim, event))
        
    # Sort top K
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]
    
    print("─────────────────────────────────────────────────────")
    print(f"🏆 Top {top_k} results for: '{query}'")
    print("─────────────────────────────────────────────────────")
    
    for rank, (sim, event) in enumerate(top_results, 1):
        vid   = event["video_id"]
        start = event["start_time"]
        end   = event["end_time"]
        
        # We fetch the exact caption generation logic just to show the original Phase 2 caption on screen
        print(f"#{rank}  |  {vid:<12} |  [{start}s – {end}s]  |  Similarity: {sim:.3f}")
        
    print("─────────────────────────────────────────────────────")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Text query to search for")
    parser.add_argument("--index", default="data/retrieval_index.pt", help="Path to database index")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to show")
    args = parser.parse_args()
    search_index(args.index, args.query, args.top_k)
