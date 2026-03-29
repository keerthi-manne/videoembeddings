import cv2
import json
import os
import sys
import time 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from models.SmolVLM import VLM_Generation
from models.VLM_Generation import VLM_Generation
VIDEO_DIR = "Video_embeddings/Videos"
JSON_FILE = "data/video_event_captions.jsonl"
OUTPUT_JSON_FILE = "data/video_event_final_captions.jsonl"

VLM = VLM_Generation()
BATCH_SIZE = 16
def time_to_seconds(t):
    return float(t.replace("s", ""))


def extract_frame(cap, timestamp, duration):
    if timestamp > duration:
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = cap.read()

    if not success:
        return None

    frame = cv2.resize(frame, (224, 224))
    return frame

def run_Caption_pipeline() : 
    with open(JSON_FILE, "r") as fin, open(OUTPUT_JSON_FILE, "w") as fout:

        for line in fin:
            data = json.loads(line.strip())

            video_id = data["video_id"]
            video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0

            new_timeline = []

            batch_frames = []
            batch_meta = []

            for segment in data["timeline"]:
                start = time_to_seconds(segment["start"])
                end = time_to_seconds(segment["end"])
                # confidence = segment["confidence"]
                # label = segment["caption"]
                mid = (start + end) / 2.0
                frame = extract_frame(cap, mid, duration)

                if frame is None:
                    # new_caption = label
                    new_timeline.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        # "old_caption": label,
                        "caption": "No caption Available"
                    })
                    continue
                batch_frames.append(frame)
                batch_meta.append((segment))

                if len(batch_frames) == BATCH_SIZE:
                    captions = VLM.generate_batch(batch_frames)

                    for cap_text, (seg) in zip(captions, batch_meta):
                        new_timeline.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "caption": cap_text
                        })

                    batch_frames.clear()
                    batch_meta.clear()

            if batch_frames:
                captions = VLM.generate_batch(batch_frames)

                for cap_text, (seg) in zip(captions, batch_meta):
                    new_timeline.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "caption": cap_text
                    })

            cap.release()

            fout.write(json.dumps({
                "video_id": video_id,
                "timeline": new_timeline
            }) + "\n")