"""
models/caption_decoder.py  –  Team A (Phase 2)

What this does:
  Given a segment feature vector (512 numbers from CLIP),
  find the best matching text caption from a candidate list.

How it works:
  CLIP puts images AND text into the same 512-number space.
  So we can directly compare a video segment's feature vector
  with encoded text captions using cosine similarity.

  Highest similarity = best matching caption.

Example:
  segment feature [0.2, 0.8, ...] (512 numbers describing "someone cooking")
      vs
  "a person cooking food"    → similarity 0.82  ← winner
  "a car driving on road"    → similarity 0.11
  "someone playing football" → similarity 0.09

Run standalone test:
  python models/caption_decoder.py
"""

import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Candidate captions (100 common video activity descriptions) ──────────────
# These cover the types of scenes found in MSR-VTT (general video dataset)
# CANDIDATE_CAPTIONS = [
#     # Cooking & food
#     "a person chopping vegetables",
#     "someone cooking food in a kitchen",
#     "a person frying food in a pan",
#     "placing food into a pot",
#     "stirring food while cooking",
#     "a person eating food",
#     "someone baking in an oven",
#     "a person washing dishes",
#     "someone preparing ingredients",
#     "a person grilling meat outdoors",

#     # Sports & exercise
#     "a person running outdoors",
#     "someone playing football",
#     "a person playing basketball",
#     "someone swimming in a pool",
#     "a person cycling on a road",
#     "someone doing yoga or stretching",
#     "a person lifting weights",
#     "someone playing tennis",
#     "a person jumping or doing gymnastics",
#     "someone skateboarding",

#     # Indoor activities
#     "a person sitting and talking",
#     "someone typing on a computer",
#     "a person reading a book",
#     "someone watching television",
#     "a person using a phone",
#     "someone playing a musical instrument",
#     "a person painting or drawing",
#     "someone doing household chores",
#     "a person sleeping or resting",
#     "someone working at a desk",

#     # Outdoor & travel
#     "a person walking on a street",
#     "someone hiking in nature",
#     "a person at the beach",
#     "someone in a park or garden",
#     "a person driving a car",
#     "someone riding a motorcycle",
#     "a person on public transportation",
#     "someone at an airport",
#     "a person shopping in a store",
#     "someone at a restaurant",

#     # Social & entertainment
#     "a group of people talking",
#     "someone dancing",
#     "a person at a party or celebration",
#     "someone playing a video game",
#     "a person giving a presentation",
#     "someone laughing and smiling",
#     "a person hugging or greeting someone",
#     "someone taking a photograph",
#     "a group of people eating together",
#     "someone singing",

#     # Animals & nature
#     "a dog playing or running",
#     "a cat sitting or sleeping",
#     "birds flying in the sky",
#     "animals in a zoo or wildlife",
#     "a person playing with a pet",

#     # Construction & work
#     "a person building or fixing something",
#     "someone using tools or machinery",
#     "a person cleaning or tidying up",
#     "someone gardening outdoors",
#     "a person carrying or moving objects",

#     # Water activities
#     "a person fishing",
#     "someone surfing waves",
#     "a person rowing a boat",
#     "someone playing in water",
#     "a person at a waterfall or river",

#     # Music & performance
#     "a person playing guitar",
#     "someone playing piano",
#     "a band performing on stage",
#     "someone doing a magic trick",
#     "a person performing on stage",

#     # Children & family
#     "children playing outdoors",
#     "a baby or toddler doing something",
#     "a family spending time together",
#     "someone helping a child",
#     "children in a classroom",

#     # Medical & science
#     "a person in a laboratory",
#     "someone receiving medical treatment",
#     "a doctor or nurse working",
#     "someone doing an experiment",
#     "a person using scientific equipment",

#     # Urban & city life
#     "cars moving on a busy road",
#     "a city street with people walking",
#     "someone at a market or bazaar",
#     "a person in a crowded place",
#     "someone at a sports stadium",

#     # Art & crafts
#     "a person making handicrafts",
#     "someone knitting or sewing",
#     "a person sculpting or molding",
#     "someone doing calligraphy or writing",
#     "a person doing origami or folding",

#     # Generic descriptions (fallback)
#     "a person doing an activity",
#     "someone outdoors in daylight",
#     "an indoor scene with people",
#     "a person moving around",
#     "a close-up of an object or face",
#     "a person demonstrating something",
#     "someone interacting with objects",
#     "an outdoor nature scene",
#     "a sports or fitness activity",
#     "a cooking or food-related scene",
# ]

CANDIDATE_CAPTIONS = [

    # Motion (extended)
    "running", "walking", "jogging", "sprinting", "jumping", "climbing",
    "falling", "sliding", "crawling", "rolling", "turning", "spinning",

    # Vehicle & transport
    "driving", "accelerating", "braking", "parking", "reversing",
    "riding", "riding", "flying", "landing",
    "boarding", "exiting",

    # Object interaction
    "holding", "grabbing", "picking up", "placing", "dropping",
    "throwing", "catching", "passing", "lifting", "carrying",
    "opening", "closing", "locking", "unlocking",
    "pushing", "pulling", "pressing", "tapping",

    # Tools / work / engineering
    "repairing", "fixing", "building", "assembling", "disassembling",
    "cutting", "drilling", "hammering", "screwing", "welding",
    "testing", "measuring", "inspecting", "adjusting", "installing",

    # Cooking (expanded)
    "cooking", "chopping", "cutting", "slicing",
    "stirring", "mixing", "pouring", "frying", "boiling",
    "baking", "grilling", "serving", "plating",
    "tasting", "eating", "washing",

    # Human interaction / communication
    "talking", "speaking", "explaining", "presenting",
    "demonstrating", "teaching", "instructing",
    "listening", "arguing", "laughing", "smiling",
    "greeting", "shaking hands", "hugging",

    # Indoor activities
    "typing", "clicking", "scrolling",
    "reading", "writing", "drawing",
    "watching", "browsing",
    "using a phone", "using a computer",

    # Sports & fitness
    "playing", "playing", "playing",
    "playing", "playing badminton",
    "swimming", "cycling", "running",
    "lifting", "exercising", "stretching", "kicking", "throwing a ball",

    # Entertainment / media
    "dancing", "singing", "performing",
    "playing music", "playing", "playing",
    "acting", "recording", "filming",

    # Animals
    "dog running", "dog jumping", "cat walking",
    "animal running", "animal eating", "animal playing",

    # Water activities
    "swimming", "diving", "surfing", "rowing",
    "splashing", "floating", "washing",

    # Office / daily work
    "working", "organizing", "arranging",
    "cleaning", "wiping", "packing", "unpacking",
    "sorting", "checking",

    # Shopping / daily life
    "shopping", "paying", "selecting items",
    "carrying groceries", "waiting in line",

    # Construction / physical work
    "digging", "lifting materials", "moving objects",
    "loading", "unloading",

    # Tech / device usage
    "operating", "controlling",
    "programming", "debugging",

    # Generic but useful
    "moving", "interacting", "handling",
    "demonstrating", "doing an activity"
]
class CaptionDecoder:
    """
    Matches a segment feature vector to the best text caption
    using CLIP's shared image-text embedding space.

    Unlike a neural network, this needs NO training —
    CLIP already understands both images and text.
    """

    def __init__(self, device: str = None):
        import clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔤 Loading CLIP text encoder on {self.device}...")
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # Pre-encode all captions ONCE at startup (not for every segment)
        self.captions = CANDIDATE_CAPTIONS
        self.text_embeddings = self._encode_captions()
        self.text_embeddings = self.text_embeddings.float()
        print(f"✅ CaptionDecoder ready: {len(self.captions)} candidate captions")

    def _encode_captions(self) -> torch.Tensor:
        """
        Converts all candidate captions to 512-dim vectors.
        Done once at startup — reused for every segment.
        Returns: (num_captions, 512)
        """
        import clip
        tokens = clip.tokenize(self.captions).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
            text_emb = F.normalize(text_emb, dim=-1)   # L2 normalize
        return text_emb  # (num_captions, 512)

    def decode(self, segment_feature: torch.Tensor, top_k: int = 1):
        """
        Find the best caption for a single segment feature vector.

        Args:
            segment_feature: (512,) — one segment's CLIP feature
            top_k: return top K matches instead of just the best one

        Returns:
            If top_k=1: (caption_string, confidence_score)
            If top_k>1: list of (caption, score) tuples
        """
        # Ensure on right device and normalized
        feat = segment_feature.to(self.device).float()
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)                # (1, 512)
        feat = F.normalize(feat, dim=-1)            # L2 normalize

        # Cosine similarity = dot product (both vectors are L2 normalized)
        # text_embeddings: (num_captions, 512)
        # feat: (1, 512) → transpose → (512, 1)
        sims = (self.text_embeddings @ feat.T).squeeze()  # (num_captions,)

        if top_k == 1:
            best_idx  = sims.argmax().item()
            best_conf = sims[best_idx].item()
            return self.captions[best_idx], round(best_conf, 3)
        else:
            top_indices = sims.topk(top_k).indices.tolist()
            return [(self.captions[i], round(sims[i].item(), 3)) for i in top_indices]




def generate_captions_for_video(
    feat: torch.Tensor,
    decoder: "CaptionDecoder",
    seg_len: int = 8,
    fps: float = 1.0,
    dynamic: bool = True
) -> list:
    """
    Segment a video's feature tensor and caption each segment.

    Args:
        feat:    (T, 512) — enriched frame features (after STAdapter)
        decoder: CaptionDecoder instance
        seg_len: frames per segment (used if dynamic=False)
        fps:     frames per second
        dynamic: if True, use similarity-based segmentation (Step 3 in Diagram)

    Returns:
        List of {"start", "end", "caption", "confidence"}
    """
    from models.event_head import dynamic_segmentation
    T = feat.shape[0]

    if dynamic:
        # Use the Architectural Diagram's similarity-based logic (Similarity < 0.8)
        dyn_segments = dynamic_segmentation(feat, threshold=0.8)
        results = []
        for seg in dyn_segments:
            caption, conf = decoder.decode(seg["feat"])
            results.append({
                "start":      round(seg["start_idx"] / fps, 2),
                "end":        round(seg["end_idx"] / fps, 2),
                "caption":    caption,
                "confidence": conf
            })
        return results

    # Legacy fixed-length segmentation
    # Short video: treat whole thing as one segment
    if T < seg_len:
        seg_feat        = feat.mean(dim=0)
        caption, conf   = decoder.decode(seg_feat)
        return [{"start": 0.0, "end": round(T / fps, 2),
                 "caption": caption, "confidence": conf}]

    # Pad T to multiple of seg_len then reshape
    pad_len = (seg_len - T % seg_len) % seg_len
    if pad_len > 0:
        feat = torch.nn.functional.pad(feat.unsqueeze(0), (0, 0, 0, pad_len)).squeeze(0)
    S = feat.shape[0] // seg_len
    segments = feat.view(S, seg_len, -1).mean(dim=1)  # (S, 512)

    results = []
    for i in range(S):
        caption, conf = decoder.decode(segments[i])
        results.append({
            "start":      round(i * seg_len / fps, 2),
            "end":        round((i + 1) * seg_len / fps, 2),
            "caption":    caption,
            "confidence": conf
        })
    return results


from typing import List, Dict, Any

def merge_caption_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments that have the same caption.

    Example:
      [0–8s cooking, 8–16s cooking, 16–24s eating]
      → [0–16s cooking, 16–24s eating]
    """
    if not segments:
        return []
    merged: List[Dict[str, Any]] = []
    cur = dict(segments[0])
    for seg in segments[1:]:
        if seg["caption"] == cur["caption"]:
            cur["end"]        = seg["end"]
            cur["confidence"] = round(float((cur["confidence"] + seg["confidence"]) / 2), 3)
        else:
            merged.append(cur)
            cur = dict(seg)
    merged.append(cur)
    return merged


if __name__ == "__main__":

    print("🧪 Testing CaptionDecoder...\n")

    decoder = CaptionDecoder()

    # Simulate 3 segment features (random — in real use these come from STAdapter output)
    # In a real run, these (512,) vectors are computed from actual video frames
    test_features = [
        torch.randn(512),
        torch.randn(512),
        torch.randn(512),
    ]

    print("Sample predictions (random features — captions won't be meaningful):")
    for i, feat in enumerate(test_features):
        caption, conf = decoder.decode(feat)
        print(f"  Segment {i}: '{caption}'  (sim={conf})")

        # Also show top 3
        top3 = decoder.decode(feat, top_k=3)
        for rank, (cap, s) in enumerate(top3, 1):
            print(f"    #{rank}: {cap}  ({s})")
        print()

    print("✅ CaptionDecoder: Test PASS!")
