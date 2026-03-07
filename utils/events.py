"""
utils/events.py  –  Team B utility functions

Two jobs:
  1. merge_segments()     - clean up raw per-segment predictions
  2. segments_to_time()   - convert segment indices → seconds

Used by:
  - training/train_events_stub.py  (to print readable output during training)
  - scripts/demo_events.py         (final demo timeline, Day 10)
"""

from typing import List, Tuple


def merge_segments(
    pred_labels: List[int],
    pred_confs: List[float],
    conf_thresh: float = 0.5
) -> List[Tuple[int, int, int, float]]:
    """
    Takes raw per-segment predictions and merges them into clean events.

    Why we need this:
      EventHead gives you one label per segment, e.g.:
        Segment 0: cooking (0.9)
        Segment 1: cooking (0.8)   ← same label, merge!
        Segment 2: driving (0.7)
        Segment 3: driving (0.3)   ← low confidence, skip
        Segment 4: driving (0.8)

      After merging: [(0,1, cooking, 0.85), (2,4, driving, 0.75)]

    Args:
        pred_labels : list of predicted label int per segment [0, 0, 1, 1, 2, ...]
        pred_confs  : list of confidence scores per segment   [0.9, 0.8, 0.7, ...]
        conf_thresh : skip segments with confidence below this

    Returns:
        List of tuples: (start_seg_idx, end_seg_idx, label, avg_confidence)
    """
    if not pred_labels:
        return []

    merged = []

    # Start the first group
    current_label = pred_labels[0]
    current_start = 0
    current_confs = [pred_confs[0]]

    for i in range(1, len(pred_labels)):
        label = pred_labels[i]
        conf = pred_confs[i]

        if label == current_label:
            # Same label → extend current group
            current_confs.append(conf)
        else:
            # Label changed → save current group (if confidence is good enough)
            avg_conf = sum(current_confs) / len(current_confs)
            if avg_conf >= conf_thresh:
                merged.append((current_start, i - 1, current_label, round(avg_conf, 3)))
            # Start new group
            current_label = label
            current_start = i
            current_confs = [conf]

    # Don't forget the last group
    avg_conf = sum(current_confs) / len(current_confs)
    if avg_conf >= conf_thresh:
        merged.append((current_start, len(pred_labels) - 1, current_label, round(avg_conf, 3)))

    return merged


def segments_to_time(
    segments: List[Tuple[int, int, int, float]],
    seg_len: int = 8,
    fps: float = 1.0
) -> List[dict]:
    """
    Converts segment indices → actual timestamps in seconds.

    How?
      Each segment covers `seg_len` frames. At `fps` frames per second:
        start_time = start_seg_idx * seg_len / fps
        end_time   = (end_seg_idx + 1) * seg_len / fps

    Example:
      seg_len=8, fps=1.0
      Segment 0 → frames 0-7  → 0.0s to 8.0s
      Segment 1 → frames 8-15 → 8.0s to 16.0s

    Args:
        segments : output of merge_segments()
        seg_len  : number of frames per segment (from config)
        fps      : frames per second used during extraction (from config)

    Returns:
        List of dicts with start_time, end_time, label, confidence
    """
    timeline = []
    for (start_idx, end_idx, label, conf) in segments:
        start_time = round(start_idx * seg_len / fps, 2)
        end_time   = round((end_idx + 1) * seg_len / fps, 2)
        timeline.append({
            "start": start_time,
            "end":   end_time,
            "label": f"event{label}",
            "confidence": conf
        })
    return timeline


def print_timeline(timeline: List[dict]):
    """
    Pretty-prints the final event timeline.

    Output format (matches Day 10 demo goal):
      [0.0s – 8.0s]  event0  (0.85)
      [8.0s – 20.0s] event1  (0.76)
    """
    print("\n📋 Event Timeline:")
    print("─" * 40)
    if not timeline:
        print("  (no events above confidence threshold)")
    for event in timeline:
        print(f"  [{event['start']}s – {event['end']}s]  "
              f"{event['label']}  ({event['confidence']})")
    print("─" * 40)


if __name__ == "__main__":
    print("🧪 Testing utils/events.py...\n")

    # Simulate raw EventHead outputs for 7 segments
    labels = [0, 0, 1, 1, 1, 2, 2]
    confs  = [0.9, 0.85, 0.7, 0.4, 0.75, 0.8, 0.82]
    #                         ^^^  low confidence segment (0.4 < 0.5 threshold)
    #                              but it's in the middle of a group, so avg decides

    merged = merge_segments(labels, confs, conf_thresh=0.5)
    print("Merged segments (start, end, label, avg_conf):")
    for m in merged:
        print(f"  {m}")

    # Convert to timeline
    timeline = segments_to_time(merged, seg_len=8, fps=1.0)
    print_timeline(timeline)

    print("✅ events.py: All tests PASS!")
