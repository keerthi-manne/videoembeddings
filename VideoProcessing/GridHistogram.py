import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import time
import clip
import torch.nn.functional as F
from PIL import Image
from ImageEmbedding import process_image
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------------
# Extract Frames
# -----------------------------
def extract_frames(video_path, resize=(640,360), frame_skip=1):

    cap = cv2.VideoCapture(video_path)

    frames = []
    gray_frames = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip == 0:

            frame = cv2.resize(frame, resize)

            frames.append(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160,90))   # smaller for SSIM
            gray_frames.append(gray)

        frame_id += 1

    cap.release()

    return frames, gray_frames


# -----------------------------
# 3×3 Grid Histogram
# -----------------------------
def grid_histogram(frame):

    h, w, _ = frame.shape
    bh = h // 3
    bw = w // 3

    feature = []

    for i in range(3):
        for j in range(3):

            block = frame[i*bh:(i+1)*bh, j*bw:(j+1)*bw]

            hist = cv2.calcHist(
                [block],
                [0,1,2],
                None,
                [6,6,6],
                [0,256,0,256,0,256]
            )

            feature.extend(hist.flatten())

    return np.array(feature)


# -----------------------------
# Histogram Distance
# -----------------------------
def histogram_distance(h1, h2):

    return np.linalg.norm(h1 - h2)


# -----------------------------
# Keyframe Extraction
# -----------------------------
def extract_keyframes(video_path):

    frames, gray_frames = extract_frames(video_path)

    print("Total Frames:", len(frames))

    keyframes = []
    clusters = []

    prev_hist = None
    current_cluster = [0]

    histograms = []


    for i, frame in enumerate(frames):

        hist = grid_histogram(frame)
        histograms.append(hist)

    hist_threshold = 5000
    ssim_threshold = 0.9

    for i in range(1, len(frames)):

        hist_diff = histogram_distance(histograms[i], histograms[i-1])

        if hist_diff > hist_threshold:

            similarity = ssim(gray_frames[i-1], gray_frames[i])

            if similarity < ssim_threshold:

                clusters.append(current_cluster)
                current_cluster = [i]

            else:
                current_cluster.append(i)

        else:
            current_cluster.append(i)

    clusters.append(current_cluster)

    # Select representative frames
    min_cluster_size = 2

    for cluster in clusters:

        if len(cluster) >= min_cluster_size:

            keyframes.append(frames[cluster[len(cluster)//2]])

    return keyframes

def keyframes_to_embeddings(keyframes, model, preprocess, device):

    embeddings = []

    for frame in keyframes:

        # convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # preprocess for CLIP
        image = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)

        # compute embedding
        with torch.no_grad():
            emb = process_image(image, device, model, preprocess)

        embeddings.append(emb)

    # combine all embeddings
    embeddings = torch.cat(embeddings, dim=0)

    return embeddings
# -----------------------------
# Run
# -----------------------------
# video_keyframes = extract_keyframes("DTrailer.mp4")
# emb = keyframes_to_embeddings(  video_keyframes,
#     model,
#     preprocess,
#     device
# )
for vid_num in range(0, 10000) : 
    try : 
        print(f"Processing video {vid_num}...")
        video_keyframes = extract_keyframes(f"./Video_embeddings/MSRVTT_Videos/video/video{vid_num}.mp4")
        video_emb = keyframes_to_embeddings(  video_keyframes,
            model,
            preprocess,
            device
        )
        torch.save(video_emb , f"./Video_embeddings/grid_embeddings/video{vid_num}.pt")  
        print(f"video {vid_num} Saved !! ")
    except Exception as e:
        print(f"❌ Error in video {vid_num}: {e}")
        continue  