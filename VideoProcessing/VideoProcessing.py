import cv2
import os
from ImageEmbedding import process_image 
import torch
import clip
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_embeddings = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
        frame_embedding = process_image(image, device , model , preprocess)
        frame_embeddings.append(frame_embedding)
    features = torch.cat(frame_embeddings, dim=0)
    cap.release()
    return features 
video_emb = video_to_frames(video_path = "3195394-uhd_3840_2160_25fps.mp4" )
def get_similarity_scores(video_emb):
    text = clip.tokenize([
        "a person making medicine in the lab",
        "a car driving",
        "a cooking scene"
    ]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    video_feat = video_emb.mean(dim=0, keepdim=True)  # [1,512]
    scores = (video_feat @ text_features.T).squeeze()

    return scores
print(get_similarity_scores(video_emb))
