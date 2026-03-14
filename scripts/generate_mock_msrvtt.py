import os
import json
import random

# We'll use the candidate captions we wrote previously to generate 
# a completely fake but structural mapping for all .pt files in embeddings1.

CANDIDATES = [
    "a person chopping vegetables", "someone running outdoors", 
    "a person typing on a computer", "someone driving a car",
    "a group of people talking", "a dog playing or running",
    "a person building or fixing something", "a person fishing",
    "a person playing guitar", "children playing outdoors"
]

def generate_mock_json(feature_dir="embeddings1", output_file="data/mock_msrvtt_captions.json"):
    if not os.path.exists(feature_dir):
        print(f"Missing {feature_dir}. Run this from the project root.")
        return

    mapping = {}
    for fname in os.listdir(feature_dir):
        if fname.endswith(".pt"):
            vid_id = fname.replace(".pt", "")
            mapping[vid_id] = random.choice(CANDIDATES)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=4)
    
    print(f"✅ Generated mock Ground Truth captions for {len(mapping)} videos into {output_file}")

if __name__ == "__main__":
    generate_mock_json()
