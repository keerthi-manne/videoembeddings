import cv2
import os
from ImageEmbedding import process_image 
import torch
import sys
import clip
from PIL import Image
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def video_to_frames(video_path, threshold=0.5):

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return None

    frame_embeddings = []
    prev_hist = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hist = cv2.calcHist([frame],[0],None,[256],[0,256])
        hist = cv2.normalize(hist, hist)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

            if diff < threshold:
                continue

        prev_hist = hist

        image = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)

        with torch.no_grad():
            frame_embedding = process_image(image, device, model, preprocess)

        frame_embeddings.append(frame_embedding)

    cap.release()

    # 🔥 CRITICAL FIX
    if len(frame_embeddings) == 0:
        print(f"⚠️ No embeddings extracted for {video_path}")
        return None

    return torch.cat(frame_embeddings, dim=0)
# def get_similarity_scores(video_emb):

#     text = clip.tokenize([
#         "a person making medicine in the lab",
#         "a car driving",
#         "a cooking scene"
#     ]).to(device)

#     with torch.no_grad():
#         text_features = model.encode_text(text)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#     video_feat = video_emb.mean(dim=0, keepdim=True)

#     scores = (video_feat @ text_features.T).squeeze()

#     return scores
# start = time.time()
# video_emb = video_to_frames("Dtrailer.mp4")
# print(len(video_emb))
# end = time.time()
# print("P time is " , end - start , "seconds")
# torch.save(video_emb , "sample_embed.pt")
for vid_num in range(0, 10000) : 
    try : 
        print(f"Processing video {vid_num}...")
        video_emb = video_to_frames(f"../Video_embeddings/MSRVTT_Videos/video/video{vid_num}.mp4")
        torch.save(video_emb , f"../Video_embeddings/embeddings_new/video{vid_num}.pt")  
        print(f"video {vid_num} Saved !! ")
    except Exception as e:
        print(f"❌ Error in video {vid_num}: {e}")
        continue  

# import cv2
# import os
# from ImageEmbedding import process_image 
# import torch
# import clip
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)


# def video_to_frames(video_path, threshold=0.5):

#     cap = cv2.VideoCapture(video_path)
#     frame_embeddings = []

#     prev_hist = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # convert frame for CLIP
#         image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)

#         frame_embedding = process_image(image, device, model, preprocess)

#         frame_embeddings.append(frame_embedding)

#     cap.release()

#     features = torch.cat(frame_embeddings, dim=0)

#     return features
# def get_similarity_scores(video_emb):

#     text = clip.tokenize([
#         "a person making medicine in the lab",
#         "a car driving",
#         "a cooking scene"
#     ]).to(device)

#     with torch.no_grad():
#         text_features = model.encode_text(text)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#     video_feat = video_emb.mean(dim=0, keepdim=True)

#     scores = (video_feat @ text_features.T).squeeze()

#     return scores

# video_emb = video_to_frames("3195394-uhd_3840_2160_25fps.mp4")
# torch.save(video_emb , "sample_embed.pt")
# # for vid_num in range( 1000 , 1302) : 
# #     try : 
# #         video_emb = video_to_frames(f"./Video_embeddings/videos/video{vid_num}.mp4")
# #         torch.save(video_emb , f"./Video_embeddings/embeddings_new/video{vid_num}.pt")  
# #     except:
# #         continue  