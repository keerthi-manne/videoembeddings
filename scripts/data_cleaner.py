import json

# Input & output paths
input_file = "Video_embeddings/msrvtt_train_9k.json"
output_file = "Video_embeddings/cleaned_msrvtt_train_9k.json"

# Load original JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert format
result = {}

for item in data:
    vid = item["video_id"]
    captions = item["caption"]
    
    result[vid] = captions

# Save new JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

print("✅ Conversion complete!")