import cv2
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.VLM_Generation import VLM_Generation
# -------- CONFIG --------
VIDEO_DIR = "Video_embeddings/Videos"
JSON_FILE = "data/video_event_captions.jsonl"
OUTPUT_JSON_FILE = "data/video_event_new_captions.jsonl"

VLM = VLM_Generation()


# -------- UTILS --------
def time_to_seconds(t):
    return float(t.replace("s", ""))


def extract_frame_caption(video_path, timestamp, prompt):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = frame_count / fps if fps > 0 else 0

    # 🔥 Skip invalid timestamps
    if timestamp > duration:
        print(f"⚠️ Skipping {timestamp}s (video length: {duration:.2f}s)")
        cap.release()
        return None

    # 🔥 Better seeking
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

    success, frame = cap.read()

    if not success:
        print(f"❌ Failed at {timestamp}s for {video_path}")
        cap.release()
        return None

    caption = VLM.generate_caption(frame, prompt)

    cap.release()
    return caption


# -------- MAIN LOOP --------
with open(JSON_FILE, "r") as fin, open(OUTPUT_JSON_FILE, "w") as fout:

    for line in fin:
        data = json.loads(line.strip())

        video_id = data["video_id"]
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

        new_timeline = []

        for i, segment in enumerate(data["timeline"]):
            start = time_to_seconds(segment["start"])
            end = time_to_seconds(segment["end"])
            confidence = segment["confidence"]
            label = segment["caption"]

            # smarter prompt usage
            prompt = label if confidence > 0.4 else None

            mid = (start + end) / 2.0

            new_caption = extract_frame_caption(
                video_path, mid, prompt
            )

            # fallback if model fails
            if new_caption is None:
                new_caption = label

            new_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "old_caption": label,
                "new_caption": new_caption,
                "confidence": confidence
            }

            new_timeline.append(new_segment)

        output_data = {
            "video_id": video_id,
            "timeline": new_timeline
        }

        fout.write(json.dumps(output_data) + "\n")