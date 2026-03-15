import cv2
import torch
import clip
import torch.nn.functional as F
from PIL import Image
from ImageEmbedding import process_image
from GridApproach import extract_keyframes
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def video_to_frames(video_path, threshold=0.5):

    cap = cv2.VideoCapture(video_path)

    prev_hist = None
    key_embeddings = []
    all_embeddings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # histogram
        hist = cv2.calcHist([frame],[0],None,[256],[0,256])
        hist = cv2.normalize(hist, hist)

        # CLIP embedding for ALL frames
        image = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
        emb = process_image(image, device, model, preprocess)

        all_embeddings.append(emb)

        select = False

        if prev_hist is None:
            select = True
        else:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff >= threshold:
                select = True

        prev_hist = hist

        if select:
            key_embeddings.append(emb)

    cap.release()

    all_embeddings = torch.cat(all_embeddings)
    key_embeddings = torch.cat(key_embeddings)

    return all_embeddings, key_embeddings
    
def compute_fidelity(all_embeddings, key_embeddings):

    similarities = []

    for emb in all_embeddings:

        sims = F.cosine_similarity(
            emb.unsqueeze(0),
            key_embeddings
        )

        best = torch.max(sims)
        similarities.append(best.item())

    fidelity = sum(similarities) / len(similarities)

    return fidelity
def keyframes_to_embeddings(keyframes):

    embeddings = []

    for frame in keyframes:

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)

        emb = process_image(image, device, model, preprocess)

        embeddings.append(emb)

    return torch.cat(embeddings, dim=0)
all_embeddings, key_embeddings = video_to_frames("DTrailer.mp4")
keyframes = extract_keyframes("DTrailer.mp4")

keyframe_embeddings = keyframes_to_embeddings(keyframes)

print(compute_fidelity(all_embeddings , keyframe_embeddings))